// Copyright 2024 KVCache.AI
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// Private implementation fragment included by tcp_transport.cpp from inside
// namespace mooncake. Keeping the protocol/session state here makes the public
// transport entry points and the lane scheduler independently reviewable
// without changing linkage or lifetime behavior.

#include "gpu_staging_pool.h"

// Debug trace for the multi-io-thread investigation (MC_TCP_DEBUG=1).
// Gated behind a cached env lookup so the hot path stays branch-only.
inline bool tcpTransportDebug() {
    static const bool enabled = [] {
        const char* env = std::getenv("MC_TCP_DEBUG");
        return env && env[0] == '1' && env[1] == '\0';
    }();
    return enabled;
}
#define TCP_DBG(...)                                        \
    do {                                                    \
        if (tcpTransportDebug()) LOG(INFO) << __VA_ARGS__;  \
    } while (0)

// Phase timing for the real-shape KV transfer benchmark (MC_TCP_PHASE_TIMING=1).
// Four durations are reported per transfer, each measured between its own pair
// of real points — none is derived from the others:
//   MC_PH1_GATHER  : source-side GPU -> pinned staging copy (D2H)
//   MC_PH2_NET     : first body byte handed to the socket on the source
//                    (BEGIN) -> last body byte received on the destination
//                    (END); the two points live in different processes but
//                    share the host CLOCK_REALTIME epoch
//   MC_PH3_SCATTER : destination-side pinned staging -> GPU copy (H2D)
//   total          : measured by the caller around batch_transfer_sync_read
// Timestamps are epoch nanoseconds (system_clock) so cross-process points on
// one host are comparable. id is the session address, for pairing points
// within one request.
inline bool tcpPhaseTiming() {
    static const bool enabled = [] {
        const char* env = std::getenv("MC_TCP_PHASE_TIMING");
        return env && env[0] == '1' && env[1] == '\0';
    }();
    return enabled;
}
inline int64_t phaseNowNs() {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(
               std::chrono::system_clock::now().time_since_epoch())
        .count();
}
#define PHASE_LOG(...)                                     \
    do {                                                   \
        if (tcpPhaseTiming())                              \
            LOG(INFO) << "[XDBG] " << __VA_ARGS__;         \
    } while (0)

static size_t getChunkSize() {
    static const size_t val = [] {
        const char* env = std::getenv("MC_TCP_SLICE_SIZE");
        if (env) {
            try {
                size_t v = std::stoull(env);
                if (v > 0) return v;
                LOG(WARNING)
                    << "Ignore non-positive MC_TCP_SLICE_SIZE value: " << env
                    << ", using default 65536";
            } catch (const std::exception& e) {
                // A non-numeric or out-of-range value makes std::stoull throw;
                // fall through to the default instead of letting the exception
                // propagate out of this static initializer and abort the
                // transfer that first reads the chunk size.
                LOG(WARNING)
                    << "Invalid MC_TCP_SLICE_SIZE value: " << env
                    << ". Error: " << e.what() << ", using default 65536";
            }
        }
        return size_t(65536);  // 64KB default
    }();
    return val;
}

struct SessionHeader {
    uint64_t size;
    uint64_t addr;
    uint8_t opcode;
};

#if defined(USE_CUDA) || defined(USE_MUSA) || defined(USE_HIP) ||  \
    defined(USE_MLU) || defined(USE_MACA) || defined(USE_HYGON) || \
    defined(USE_COREX)
// Returns the CUDA device ordinal if addr is device memory, or -1 otherwise.
// Callers must call cudaSetDevice before any cudaMemcpy to avoid implicit
// GPU 0 context creation.
static int getCudaDeviceId(void* addr) {
    cudaPointerAttributes attributes;
    auto status = cudaPointerGetAttributes(&attributes, addr);
    if (status != cudaSuccess) return -1;
    if (attributes.type == cudaMemoryTypeDevice) return attributes.device;
    return -1;
}

#ifdef USE_MACA
static cudaError_t copyTcpCudaMemory(void* dst, const void* src, size_t size) {
    cudaStream_t stream;
    cudaError_t status =
        cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
    if (status != cudaSuccess) return status;

    status = cudaMemcpyAsync(dst, src, size, cudaMemcpyDefault, stream);
    if (status == cudaSuccess) {
        status = cudaStreamSynchronize(stream);
    }

    cudaError_t destroy_status = cudaStreamDestroy(stream);
    return status == cudaSuccess ? destroy_status : status;
}
#endif
#endif

// Forward declaration
class TcpTransport;

using ValidateAddrFn = std::function<bool(uint64_t, uint64_t)>;

// --- Acknowledged framing (protocol v2, #2086) ------------------------------
// v1 framing gives the initiator no channel to learn whether the receiver
// applied (or even accepted) a WRITE: COMPLETED fires when the final chunk
// enters the initiator's kernel socket buffer, while megabytes may still be
// in flight toward destination memory, and a rejected request is silently
// "successful". v2 requests set the high bit of the opcode; the server then
// (a) prefixes every READ response with an 8-byte status frame and (b) sends
// an 8-byte status frame for WRITE only after the final chunk has been
// applied to destination memory. Initiators enable v2 only when the target
// segment advertises tcp_proto_version >= 2, so old servers never see
// flagged opcodes and old initiators keep receiving v1 framing.
static constexpr uint8_t kOpcodeV2Flag = 0x80;
// Status frames carry a magic in the high 32 bits so that a v2 initiator
// which reaches a v1 server through a stale descriptor (v1 treats unknown
// opcodes as READ and immediately streams payload bytes) fails fast on the
// first frame instead of misinterpreting the stream. Residual risk: payload
// bytes that happen to equal a valid frame (2^-64 per request, data
// dependent) are indistinguishable in-band; eliminating that would need a
// nonce/checksum handshake, which this deliberately avoids.
static constexpr uint64_t kStatusMagic = 0x4D435456ull << 32;  // "MCTV"
static constexpr uint64_t kStatusOk = kStatusMagic | 0;
static constexpr uint64_t kStatusAddrRejected = kStatusMagic | 1;
static inline bool statusFrameValid(uint64_t frame) {
    return (frame & 0xFFFFFFFF00000000ull) == kStatusMagic;
}

// Operational escape hatch: MC_TCP_PROTO=1 forces initiators to speak the
// legacy unacknowledged framing even to v2-capable servers. Also used by
// tests to cover the mixed-version matrix in one process.
static bool forceLegacyTcpProto() {
    // Read per call (startTransfer already does metadata lookups; getenv is
    // noise) so tests can cover both protocol modes in one process.
    const char* env = std::getenv("MC_TCP_PROTO");
    return env && env[0] == '1' && env[1] == '\0';
}

// Server-side session: handles transfer requests on a persistent connection.
// The session owns the socket; ending the callback chain without rearming
// (start()/next handler) drops the last reference and closes the connection.
struct ServerSession : public std::enable_shared_from_this<ServerSession> {
    explicit ServerSession(std::shared_ptr<tcpsocket> socket,
                           ValidateAddrFn validate_addr)
        : socket_(std::move(socket)),
          strand_(socket_->get_executor()),
          validate_addr_(std::move(validate_addr)) {}

    ~ServerSession() { releaseStaging(); }

    std::shared_ptr<tcpsocket> socket_;
    // All handlers of a session are bound to strand_. The request chain is
    // sequential by construction (one outstanding operation at a time), but
    // with a shared io_context on multiple threads the strand pins initiation
    // and state access to a single logical thread, matching ClientSession.
    asio::strand<asio::any_io_executor> strand_;
    ValidateAddrFn validate_addr_;
    SessionHeader header_;
    uint64_t total_transferred_bytes_;
    char* local_buffer_;
    bool v2_ = false;
    uint64_t status_frame_;
    // Pinned staging slot held while a GPU transfer uses the pool fast path.
    GpuStagingPool::Slot* staging_slot_ = nullptr;

    void releaseStaging() {
        if (staging_slot_) {
            GpuStagingPool::instance()->release(staging_slot_);
            staging_slot_ = nullptr;
        }
    }

    void start() {
        total_transferred_bytes_ = 0;
        readHeader();
    }

   private:
    // Send an 8-byte status frame, then run `next` (or end the session —
    // closing the connection — when `next` is empty or the send fails).
    void sendStatus(uint64_t status, std::function<void()> next) {
        status_frame_ = htole64(status);
        TCP_DBG("S sendStatus status=0x" << std::hex << status << std::dec
                                         << " has_next=" << (bool)next);
        auto self(shared_from_this());
        asio::async_write(
            *socket_, asio::buffer(&status_frame_, sizeof(status_frame_)),
            asio::bind_executor(
                strand_,
                [this, self, next = std::move(next)](
                    const asio::error_code& ec, std::size_t) {
                    if (ec)
                        return;  // connection closes with the session
                    if (next) next();
                }));
    }

    void readHeader() {
        auto self(shared_from_this());
        asio::async_read(
            *socket_, asio::buffer(&header_, sizeof(SessionHeader)),
            asio::bind_executor(
                strand_,
                [this, self](const asio::error_code& ec, std::size_t len) {
                    TCP_DBG("S readHeader done ec=" << ec.value()
                                                    << " len=" << len);
                    if (ec || len != sizeof(SessionHeader)) {
                        if (ec.value() != asio::error::eof) {
                            LOG(WARNING)
                                << "ServerSession::readHeader failed. Error: "
                                << ec.message() << " (value: " << ec.value()
                                << ")"
                                << ", bytes read: " << len;
                        }
                        return;
                    }

                    v2_ = (header_.opcode & kOpcodeV2Flag) != 0;
                    const uint8_t opcode = header_.opcode & ~kOpcodeV2Flag;
                    local_buffer_ = (char*)(le64toh(header_.addr));
                    uint64_t size = le64toh(header_.size);
                    TCP_DBG("S readHeader opcode=" << (int)opcode
                                                   << " v2=" << v2_
                                                   << " size=" << size);
                    if (validate_addr_ &&
                        !validate_addr_((uint64_t)local_buffer_, size)) {
                        LOG(ERROR)
                            << "ServerSession: remote-supplied address 0x"
                            << std::hex << (uint64_t)local_buffer_ << std::dec
                            << " with size " << size
                            << " is not within any registered buffer";
                        // v2 initiators learn of the rejection; v1 initiators
                        // only see the connection close (and, for small WRITEs,
                        // may have already reported success — the defect v2
                        // exists to fix).
                        if (v2_) sendStatus(kStatusAddrRejected, nullptr);
                        return;
                    }
                    if (opcode == (uint8_t)TransferRequest::WRITE) {
                        readBody();
                    } else if (v2_) {
                        // READ, v2: status frame precedes the data.
                        sendStatus(kStatusOk, [this] { writeBody(); });
                    } else {
                        writeBody();
                    }
                }));
    }

    void writeBody() {
        auto self(shared_from_this());
        uint64_t size = le64toh(header_.size);
        char* addr = local_buffer_;

        size_t buffer_size =
            std::min(getChunkSize(), size - total_transferred_bytes_);
        if (buffer_size == 0) {
            // Transfer complete, wait for next request on this connection
            releaseStaging();
            start();
            return;
        }

        char* dram_buffer = addr + total_transferred_bytes_;
        int cuda_device = -1;
        bool staged = false;

#if defined(USE_CUDA) || defined(USE_MUSA) || defined(USE_HIP) ||  \
    defined(USE_MLU) || defined(USE_MACA) || defined(USE_HYGON) || \
    defined(USE_COREX)
        if (staging_slot_) {
            // Whole request already staged into the pinned slot; stream the
            // remaining bytes straight from it.
            dram_buffer = staging_slot_->buffer + total_transferred_bytes_;
            staged = true;
        } else {
            cuda_device = getCudaDeviceId(addr);
            if (cuda_device >= 0) {
                if (total_transferred_bytes_ == 0) {
                    // Fast path: stage the entire request with one
                    // asynchronous ordered copy, then send from the pinned
                    // slot. Any acquire/stage failure falls through to the
                    // legacy per-chunk staging below.
                    auto* pool = GpuStagingPool::instance();
                    if (auto* slot =
                            pool->acquire(static_cast<size_t>(size))) {
                        const int64_t ph0 =
                            tcpPhaseTiming() ? phaseNowNs() : 0;
                        if (pool->stageDeviceToHost(slot, addr,
                                                    static_cast<size_t>(
                                                        size))) {
                            PHASE_LOG("MC_PH1_GATHER t0="
                                      << ph0 << " t1=" << phaseNowNs()
                                      << " bytes=" << size
                                      << " id=" << static_cast<void*>(this));
                            staging_slot_ = slot;
                            dram_buffer = slot->buffer;
                            staged = true;
                        } else {
                            pool->release(slot);
                        }
                    }
                }
                if (!staged) {
                    dram_buffer = new char[buffer_size];
                    cudaSetDevice(cuda_device);
#ifdef USE_MACA
                    cudaError_t cuda_status = copyTcpCudaMemory(
                        dram_buffer, addr + total_transferred_bytes_,
                        buffer_size);
#else
                    const int64_t ph0 = tcpPhaseTiming() ? phaseNowNs() : 0;
                    cudaError_t cuda_status =
                        cudaMemcpy(dram_buffer, addr + total_transferred_bytes_,
                                   buffer_size, cudaMemcpyDefault);
#endif
#ifndef USE_MACA
                    PHASE_LOG("MC_PH1_GATHER t0="
                              << ph0 << " t1=" << phaseNowNs()
                              << " bytes=" << buffer_size
                              << " id=" << static_cast<void*>(this));
#endif
                    if (cuda_status != cudaSuccess) {
                        LOG(ERROR)
                            << "ServerSession::writeBody failed to copy from "
                               "CUDA memory. "
                            << "Error: " << cudaGetErrorString(cuda_status);
                        delete[] dram_buffer;
                        return;  // Connection will be closed
                    }
                }
            }
        }
#endif

        if (total_transferred_bytes_ == 0) {
            PHASE_LOG("MC_PH2_NET_BEGIN t="
                      << phaseNowNs() << " bytes=" << size
                      << " id=" << static_cast<void*>(this));
        }
        asio::async_write(
            *socket_, asio::buffer(dram_buffer, buffer_size),
            asio::bind_executor(
                strand_,
                [this, addr, dram_buffer, cuda_device, staged, self](
                    const asio::error_code& ec, std::size_t transferred_bytes) {
                    TCP_DBG("S writeBody chunk ec=" << ec.value()
                                                    << " bytes=" << transferred_bytes
                                                    << " total=" << total_transferred_bytes_);
#if defined(USE_CUDA) || defined(USE_MUSA) || defined(USE_HIP) ||  \
    defined(USE_MLU) || defined(USE_MACA) || defined(USE_HYGON) || \
    defined(USE_COREX)
                    if (cuda_device >= 0 && !staged) {
                        delete[] dram_buffer;
                    }
#endif
                    if (ec) {
                        LOG(ERROR)
                            << "ServerSession::writeBody failed. "
                            << "Attempt to write data "
                            << static_cast<void*>(addr) << " using buffer "
                            << static_cast<void*>(dram_buffer)
                            << ". Error: " << ec.message()
                            << " (value: " << ec.value() << ")";
                        return;  // Connection will be closed
                    }
                    total_transferred_bytes_ += transferred_bytes;
                    writeBody();
                }));
    }

    void readBody() {
        auto self(shared_from_this());
        uint64_t size = le64toh(header_.size);
        char* addr = local_buffer_;

        size_t buffer_size =
            std::min(getChunkSize(), size - total_transferred_bytes_);
        if (buffer_size == 0) {
            // Destination memory now holds the complete payload. Under v2,
            // acknowledge before accepting the next request — this is what
            // makes the initiator's COMPLETED mean "applied at the
            // destination" rather than "left my socket buffer".
            if (staging_slot_) {
                // Flush the staged payload into device memory with one
                // asynchronous copy before acknowledging.
                auto* pool = GpuStagingPool::instance();
                bool ok = pool->stageHostToDevice(
                    staging_slot_, addr, static_cast<size_t>(size));
                pool->release(staging_slot_);
                staging_slot_ = nullptr;
                if (!ok) return;  // Connection will be closed
            }
            if (v2_) {
                sendStatus(kStatusOk, [this] { start(); });
            } else {
                start();
            }
            return;
        }

        char* dram_buffer = addr + total_transferred_bytes_;
        int cuda_device = -1;
        bool staged = false;

#if defined(USE_CUDA) || defined(USE_MUSA) || defined(USE_HIP) ||  \
    defined(USE_MLU) || defined(USE_MACA) || defined(USE_HYGON) || \
    defined(USE_COREX)
        if (staging_slot_) {
            // Receiving into the pinned slot; flushed to device memory at
            // completion.
            dram_buffer = staging_slot_->buffer + total_transferred_bytes_;
            staged = true;
        } else {
            cuda_device = getCudaDeviceId(addr);
            if (cuda_device >= 0) {
                if (total_transferred_bytes_ == 0) {
                    // Fast path: receive the whole request directly into a
                    // pinned slot. nullptr (disabled/oversized/exhausted)
                    // keeps the legacy per-chunk staging below.
                    staging_slot_ = GpuStagingPool::instance()->acquire(
                        static_cast<size_t>(size));
                }
                if (staging_slot_) {
                    dram_buffer =
                        staging_slot_->buffer + total_transferred_bytes_;
                    staged = true;
                } else {
                    dram_buffer = new char[buffer_size];
                }
            }
        }
#endif

        asio::async_read(
            *socket_, asio::buffer(dram_buffer, buffer_size),
            asio::bind_executor(
                strand_,
                [this, addr, dram_buffer, cuda_device, staged, self](
                    const asio::error_code& ec, std::size_t transferred_bytes) {
                    TCP_DBG("S readBody chunk ec=" << ec.value()
                                                   << " bytes=" << transferred_bytes
                                                   << " total=" << total_transferred_bytes_);
                    if (ec) {
                        // If client closed connection (EOF), this is normal -
                        // don't log
                        if (ec.value() != asio::error::eof) {
                            LOG(WARNING)
                                << "ServerSession::readBody failed. "
                                << "Attempt to read data "
                                << static_cast<void*>(addr) << " using buffer "
                                << static_cast<void*>(dram_buffer)
                                << ". Error: " << ec.message()
                                << " (value: " << ec.value() << ")";
                        }
                        if (cuda_device >= 0 && !staged) delete[] dram_buffer;
                        return;  // Connection will be closed
                    }

#if defined(USE_CUDA) || defined(USE_MUSA) || defined(USE_HIP) ||  \
    defined(USE_MLU) || defined(USE_MACA) || defined(USE_HYGON) || \
    defined(USE_COREX)
                    if (cuda_device >= 0 && !staged) {
                        cudaSetDevice(cuda_device);
#ifdef USE_MACA
                        cudaError_t cuda_status =
                            copyTcpCudaMemory(addr + total_transferred_bytes_,
                                              dram_buffer, transferred_bytes);
#else
                        cudaError_t cuda_status = cudaMemcpy(
                            addr + total_transferred_bytes_, dram_buffer,
                            transferred_bytes, cudaMemcpyDefault);
#endif
                        if (cuda_status != cudaSuccess) {
                            LOG(ERROR)
                                << "ServerSession::readBody failed to copy to "
                                   "CUDA memory. "
                                << "Error: " << cudaGetErrorString(cuda_status);
                            delete[] dram_buffer;
                            return;  // Connection will be closed
                        }
                        delete[] dram_buffer;
                    }
#endif
                    total_transferred_bytes_ += transferred_bytes;
                    readBody();
                }));
    }
};

// Client-side session: initiates one transfer request
struct ClientSession : public std::enable_shared_from_this<ClientSession> {
    using OnTerminal =
        std::function<void(TransferStatusEnum, bool connection_clean)>;

    explicit ClientSession(std::shared_ptr<tcpsocket> socket, bool use_v2,
                           OnTerminal on_terminal)
        : socket_(std::move(socket)),
          strand_(socket_->get_executor()),
          v2_(use_v2),
          on_terminal_(std::move(on_terminal)) {}

    std::shared_ptr<tcpsocket> socket_;
    // Serializes this session's handlers (v2 WRITE runs a body-write chain and
    // an ack read concurrently) and its timers. With one io thread this was
    // implicit; with a shared io_context on multiple threads, concurrent
    // socket use and session state access from different handler threads are
    // unsafe.
    asio::strand<asio::any_io_executor> strand_;
    SessionHeader header_;
    uint64_t total_transferred_bytes_;
    char* local_buffer_;
    bool v2_;
    uint64_t status_frame_;
    // v2 WRITE runs the body stream and the ack read concurrently (one
    // async op per direction; handlers serialize on the io thread). The
    // concurrent read lets a rejection — or a v1 server's bogus payload —
    // abort a large in-flight WRITE instead of deadlocking on mutually
    // full socket buffers, and delivers rejection frames before the close.
    bool write_body_done_ = false;
    bool write_acked_ok_ = false;
    // An early negative/malformed ack can arrive while asio::async_write still
    // owns a buffer pointing into the caller's source memory. Do not publish a
    // terminal status until that body operation has completed or been
    // cancelled: callers are allowed to release the source buffer as soon as
    // the transfer becomes terminal.
    bool write_body_in_flight_ = false;
    bool write_abort_requested_ = false;
    // A v2 status frame is prompt by construction: a READ status precedes
    // any payload, and a WRITE ack follows at most one chunk's apply after
    // the body is done. The only peer that never sends one is a legacy
    // server reached through a stale v2 descriptor — and for requests
    // shorter than a frame it also keeps the connection open (it streamed
    // size < 8 bytes of "READ payload" and is waiting for our next header),
    // so without a deadline both sides wait forever. Bound that wait; the
    // default is generous so no healthy slow path can trip it, since it only
    // covers the frame itself, never payload streaming.
    static int statusFrameTimeoutSec() {
        const char* env = std::getenv("MC_TCP_STATUS_TIMEOUT_SEC");
        if (env) {
            int v = std::atoi(env);
            if (v > 0) return v;
        }
        return 30;
    }
    std::optional<asio::steady_timer> status_timer_;
    bool status_deadline_disarmed_ = false;
    // Unlike the status-frame deadline above, this is a sliding liveness
    // deadline for the whole request. It is refreshed only after an Asio
    // completion reports actual protocol or payload progress.
    static int progressTimeoutSec() {
        const char* env = std::getenv("MC_TCP_PROGRESS_TIMEOUT_SEC");
        if (env) {
            int v = std::atoi(env);
            if (v > 0) return v;
        }
        return 30;
    }
    std::optional<asio::steady_timer> progress_timer_;
    uint64_t progress_epoch_ = 0;
    bool progress_deadline_disarmed_ = true;
    bool progress_timeout_committed_ = false;
    bool terminal_reported_ = false;
    OnTerminal on_terminal_;
    // Pinned staging slot held while a GPU transfer uses the pool fast path.
    GpuStagingPool::Slot* staging_slot_ = nullptr;

    ~ClientSession() { releaseStaging(); }

    void releaseStaging() {
        if (staging_slot_) {
            GpuStagingPool::instance()->release(staging_slot_);
            staging_slot_ = nullptr;
        }
    }

    void initiate(void* buffer, uint64_t dest_addr, size_t size,
                  TransferRequest::OpCode opcode) {
        local_buffer_ = (char*)buffer;
        header_.addr = htole64(dest_addr);
        header_.size = htole64(size);
        header_.opcode = (uint8_t)opcode | (v2_ ? kOpcodeV2Flag : 0);
        total_transferred_bytes_ = 0;
        TCP_DBG("C initiate size=" << size << " opcode=" << (int)opcode
                                   << " v2=" << v2_);
        writeHeader();
    }

    void cancel() noexcept {
        // Called from arbitrary threads (shutdown/retirement posts and the
        // post-join cleanup path). Route through the strand so socket and
        // session-state access cannot race in-flight handlers; when the
        // io_context is already stopped the posted handler never runs, and
        // the destructor plus the shutdown path's direct socket closes cover
        // the same cleanup.
        try {
            asio::post(strand_, [self = shared_from_this()] {
                self->cancelOnStrand();
            });
        } catch (...) {
            // Executor gone: no handler can be running concurrently.
            cancelOnStrand();
        }
    }

   private:
    void cancelOnStrand() noexcept {
        cancelStatusDeadline();
        cancelProgressDeadline();
        releaseStaging();
        if (!socket_) return;
        asio::error_code cancel_ec;
        socket_->cancel(cancel_ec);
        asio::error_code close_ec;
        socket_->close(close_ec);
    }

   private:
    // The remainder of the session is driven by Asio completion handlers; all
    // are bound to strand_ so initiation stays single-threaded per session.

    // The status-frame deadline is separate from the sliding progress
    // deadline below. Its expiry closes the socket, but a status completion
    // already ready in the executor queue retains its original result; the
    // status handler still owns the terminal path.
    void armStatusDeadline() {
        auto self(shared_from_this());
        status_deadline_disarmed_ = false;
        status_timer_.emplace(strand_);
        status_timer_->expires_after(
            std::chrono::seconds(statusFrameTimeoutSec()));
        status_timer_->async_wait([this, self](const asio::error_code& ec) {
            // The disarmed flag also covers an expiry that was already
            // queued when cancel() ran (cancel cannot revoke those, and by
            // then the socket may have been re-pooled).
            if (ec == asio::error::operation_aborted ||
                status_deadline_disarmed_) {
                return;
            }
            LOG(ERROR) << "ClientSession: no status frame within "
                       << statusFrameTimeoutSec()
                       << "s (peer likely speaks the legacy protocol); "
                          "dropping connection";
            if (socket_ && socket_->is_open()) {
                asio::error_code cec;
                socket_->close(cec);
            }
        });
    }

    void cancelStatusDeadline() {
        status_deadline_disarmed_ = true;
        if (status_timer_) {
            asio::error_code ec;
            status_timer_->cancel(ec);
        }
    }

    void refreshProgressDeadline() {
        if (terminal_reported_ || progress_timeout_committed_) return;
        auto self(shared_from_this());
        progress_deadline_disarmed_ = false;
        ++progress_epoch_;
        if (progress_epoch_ == 0) ++progress_epoch_;
        const uint64_t progress_epoch = progress_epoch_;
        if (!progress_timer_) progress_timer_.emplace(strand_);
        progress_timer_->expires_after(
            std::chrono::seconds(progressTimeoutSec()));
        progress_timer_->async_wait(
            [this, self, progress_epoch](const asio::error_code& ec) {
                handleProgressDeadline(progress_epoch, ec);
            });
    }

    void handleProgressDeadline(uint64_t progress_epoch,
                                const asio::error_code& ec) {
        if (ec == asio::error::operation_aborted ||
            progress_deadline_disarmed_ || terminal_reported_ ||
            progress_epoch != progress_epoch_) {
#ifdef MOONCAKE_TCP_TRANSPORT_TEST_HOOKS
            (void)invokeSessionProgressHook(kSessionTimeoutStale, false);
#endif
            return;
        }
        progress_timeout_committed_ = true;
        progress_deadline_disarmed_ = true;
        TCP_DBG("C progress deadline FIRE");
        ++progress_epoch_;
        if (progress_epoch_ == 0) ++progress_epoch_;
        if ((header_.opcode & ~kOpcodeV2Flag) ==
            static_cast<uint8_t>(TransferRequest::WRITE)) {
            write_abort_requested_ = true;
        }
#ifdef MOONCAKE_TCP_TRANSPORT_TEST_HOOKS
        (void)invokeSessionProgressHook(kSessionTimeoutCommitted,
                                        write_body_in_flight_);
#endif
        LOG(ERROR) << "ClientSession: no transfer progress within "
                   << progressTimeoutSec() << "s; dropping connection";
        if (socket_ && socket_->is_open()) {
            asio::error_code cancel_ec;
            socket_->cancel(cancel_ec);
            asio::error_code close_ec;
            socket_->close(close_ec);
        }
    }

    bool acceptProgress() {
        if (progress_timeout_committed_) return false;
        refreshProgressDeadline();
        return !progress_timeout_committed_;
    }

#ifdef MOONCAKE_TCP_TRANSPORT_TEST_HOOKS
    bool acceptProgressForTest(int event) {
        const uint64_t previous_epoch = progress_epoch_;
        const int action = invokeSessionProgressHook(event, false);
        if (action == kSessionCommitTimeoutBeforeProgress)
            handleProgressDeadline(previous_epoch, asio::error_code{});
        if (!acceptProgress()) return false;
        if (action == kSessionReplayPreviousTimeoutAfterProgress)
            handleProgressDeadline(previous_epoch, asio::error_code{});
        return !progress_timeout_committed_;
    }
#endif

    void cancelProgressDeadline() {
        progress_deadline_disarmed_ = true;
        ++progress_epoch_;
        if (progress_epoch_ == 0) ++progress_epoch_;
        if (progress_timer_) {
            asio::error_code ec;
            progress_timer_->cancel(ec);
        }
    }

    // Single terminal path. The invoking Asio operation has already released
    // its buffer before entering its completion handler. The lane posts any
    // follow-up pump, so a clean socket cannot be reused inline here.
    void finalize(TransferStatusEnum status, bool clean) {
        if (terminal_reported_) return;
        terminal_reported_ = true;
        TCP_DBG("C finalize status=" << (int)status << " clean=" << clean);
        cancelStatusDeadline();
        cancelProgressDeadline();
#ifdef MOONCAKE_TCP_TRANSPORT_TEST_HOOKS
        (void)invokeSessionProgressHook(kSessionTerminal, clean);
#endif
        auto on_terminal = std::move(on_terminal_);
        if (on_terminal) on_terminal(status, clean);
    }

    // Abort a v2 WRITE and cancel any body operation. If asio still owns the
    // current source buffer, its completion handler is responsible for
    // finalizing after the buffer is quiescent.
    void abortWrite() {
        write_abort_requested_ = true;
        if (socket_ && socket_->is_open()) {
            asio::error_code ec;
            socket_->close(ec);
        }
        if (!write_body_in_flight_) finalize(TransferStatusEnum::FAILED, false);
    }

    void writeHeader() {
        auto self(shared_from_this());
        asio::async_write(
            *socket_, asio::buffer(&header_, sizeof(SessionHeader)),
            asio::bind_executor(
                strand_,
                [this, self](const asio::error_code& ec, std::size_t len) {
                    TCP_DBG("C writeHeader done ec=" << ec.value()
                                                     << " len=" << len);
                    if (ec || len != sizeof(SessionHeader)) {
                        LOG(ERROR)
                            << "ClientSession::writeHeader failed. Error: "
                            << ec.message() << " (value: " << ec.value() << ")"
                            << ", bytes written: " << len;
                        finalize(TransferStatusEnum::FAILED, false);
                        return;
                    }
                    if (!acceptProgress()) {
                        if ((header_.opcode & ~kOpcodeV2Flag) ==
                            static_cast<uint8_t>(TransferRequest::WRITE)) {
                            abortWrite();
                        } else {
                            finalize(TransferStatusEnum::FAILED, false);
                        }
                        return;
                    }
                    if ((header_.opcode & ~kOpcodeV2Flag) ==
                        (uint8_t)TransferRequest::WRITE) {
                        if (v2_) readWriteAck();  // concurrent with the body
                        writeBody();
                    } else if (v2_) {
                        readReadStatus();
                    } else {
                        readBody();
                    }
                }));
        // Asio does not invoke the completion handler inline. Arm only after
        // initiation succeeds so a synchronous initiation exception cannot
        // leave a timer handler retaining this otherwise-abandoned session.
        refreshProgressDeadline();
    }

    // v2 READ: the server prefixes the data with a status frame.
    void readReadStatus() {
        auto self(shared_from_this());
        armStatusDeadline();
        asio::async_read(
            *socket_, asio::buffer(&status_frame_, sizeof(status_frame_)),
            asio::bind_executor(
                strand_,
                [this, self](const asio::error_code& ec, std::size_t len) {
                    cancelStatusDeadline();
                    if (ec || len != sizeof(status_frame_)) {
                        LOG(ERROR)
                            << "ClientSession: failed to read READ status "
                               "frame. Error: "
                            << ec.message() << " (value: " << ec.value() << ")";
                        finalize(TransferStatusEnum::FAILED, false);
                        return;
                    }
                    if (!acceptProgress()) {
                        finalize(TransferStatusEnum::FAILED, false);
                        return;
                    }
                    uint64_t frame = le64toh(status_frame_);
                    if (!statusFrameValid(frame)) {
                        LOG(ERROR) << "ClientSession: malformed READ status "
                                      "frame (peer likely speaks the legacy "
                                      "protocol); dropping connection";
                        finalize(TransferStatusEnum::FAILED, false);
                        return;
                    }
                    if (frame != kStatusOk) {
                        LOG(ERROR) << "ClientSession: READ rejected by server, "
                                      "status "
                                   << (frame & 0xFFFFFFFFull);
                        finalize(TransferStatusEnum::FAILED, false);
                        return;
                    }
                    readBody();
                }));
    }

    // v2 WRITE: completion is the server's acknowledgment that the payload
    // has been applied to destination memory. Armed concurrently with the
    // body stream; a well-behaved v2 server only sends the frame after the
    // final chunk, so a frame arriving before the body is done is either a
    // rejection or a legacy peer's payload — both close the socket immediately
    // and publish failure only after the outstanding body write is quiescent.
    void readWriteAck() {
        auto self(shared_from_this());
        asio::async_read(
            *socket_, asio::buffer(&status_frame_, sizeof(status_frame_)),
            asio::bind_executor(
                strand_,
                [this, self](const asio::error_code& ec, std::size_t len) {
                    TCP_DBG("C ack done ec=" << ec.value() << " len=" << len
                                             << " body_done=" << write_body_done_);
                    cancelStatusDeadline();
                    if (ec || len != sizeof(status_frame_)) {
                        // The body path may have already finalized a failure and
                        // closed the socket; finalize() is idempotent (moved-from
                        // callbacks are null-checked).
                        if (ec != asio::error::operation_aborted) {
                            LOG(ERROR)
                                << "ClientSession: failed to read WRITE "
                                   "ack frame. Error: "
                                << ec.message() << " (value: " << ec.value() << ")";
                        }
                        abortWrite();
                        return;
                    }
                    bool progress_accepted = false;
#ifdef MOONCAKE_TCP_TRANSPORT_TEST_HOOKS
                    progress_accepted =
                        acceptProgressForTest(kSessionWriteAckSuccess);
#else
                    progress_accepted = acceptProgress();
#endif
                    if (!progress_accepted) {
                        abortWrite();
                        return;
                    }
                    uint64_t frame = le64toh(status_frame_);
                    if (!statusFrameValid(frame)) {
                        LOG(ERROR) << "ClientSession: malformed WRITE ack frame "
                                      "(peer likely speaks the legacy protocol); "
                                      "dropping connection";
                        abortWrite();
                        return;
                    }
                    if (frame != kStatusOk) {
                        LOG(ERROR) << "ClientSession: WRITE rejected by server, "
                                      "status "
                                   << (frame & 0xFFFFFFFFull);
                        abortWrite();
                        return;
                    }
                    if (!write_body_done_) {
                        // The server's ack can legitimately overtake the final
                        // local write-completion handler (both become ready
                        // together for small writes; the io thread may run this
                        // handler first). Record it; the body path finalizes.
                        write_acked_ok_ = true;
                        return;
                    }
                    finalize(TransferStatusEnum::COMPLETED, true);
                }));
    }

    void readBody() {
        auto self(shared_from_this());
        uint64_t size = le64toh(header_.size);
        char* addr = local_buffer_;

        size_t buffer_size =
            std::min(getChunkSize(), size - total_transferred_bytes_);
        if (buffer_size == 0) {
            // Whole payload received. With staged GPU data, flush it into
            // device memory before reporting completion so that COMPLETED
            // still means "applied at the destination".
            PHASE_LOG("MC_PH2_NET_END t=" << phaseNowNs() << " bytes=" << size
                                          << " id=" << static_cast<void*>(this));
            if (staging_slot_) {
                auto* pool = GpuStagingPool::instance();
                const int64_t ph0 = tcpPhaseTiming() ? phaseNowNs() : 0;
                bool ok = pool->stageHostToDevice(
                    staging_slot_, addr, static_cast<size_t>(size));
                PHASE_LOG("MC_PH3_SCATTER t0="
                          << ph0 << " t1=" << phaseNowNs() << " bytes=" << size
                          << " id=" << static_cast<void*>(this));
                pool->release(staging_slot_);
                staging_slot_ = nullptr;
                if (!ok) {
                    finalize(TransferStatusEnum::FAILED, false);
                    return;
                }
            }
            finalize(TransferStatusEnum::COMPLETED, true);
            return;
        }

        char* dram_buffer = addr + total_transferred_bytes_;
        int cuda_device = -1;
        bool staged = false;

#if defined(USE_CUDA) || defined(USE_MUSA) || defined(USE_HIP) ||  \
    defined(USE_MLU) || defined(USE_MACA) || defined(USE_HYGON) || \
    defined(USE_COREX)
        if (staging_slot_) {
            // Receiving into the pinned slot; flushed to device memory at
            // completion.
            dram_buffer = staging_slot_->buffer + total_transferred_bytes_;
            staged = true;
        } else {
            cuda_device = getCudaDeviceId(addr);
            if (cuda_device >= 0) {
                if (total_transferred_bytes_ == 0) {
                    // Fast path: receive the whole request directly into a
                    // pinned slot. nullptr (disabled/oversized/exhausted)
                    // keeps the legacy per-chunk staging below.
                    staging_slot_ = GpuStagingPool::instance()->acquire(
                        static_cast<size_t>(size));
                }
                if (staging_slot_) {
                    dram_buffer =
                        staging_slot_->buffer + total_transferred_bytes_;
                    staged = true;
                } else {
                    dram_buffer = new char[buffer_size];
                }
            }
        }
#endif

        asio::async_read(
            *socket_, asio::buffer(dram_buffer, buffer_size),
            asio::bind_executor(
                strand_,
                [this, addr, dram_buffer, cuda_device, staged, self](
                        const asio::error_code& ec, std::size_t transferred_bytes) {
                    TCP_DBG("C readBody chunk ec=" << ec.value()
                                                   << " bytes=" << transferred_bytes
                                                   << " total=" << total_transferred_bytes_);
                    if (ec) {
                        LOG(ERROR)
                            << "ClientSession::readBody failed. "
                            << "Attempt to read data " << static_cast<void*>(addr)
                            << " using buffer " << static_cast<void*>(dram_buffer)
                            << ". Error: " << ec.message()
                            << " (value: " << ec.value() << ")";
#if defined(USE_CUDA) || defined(USE_MUSA) || defined(USE_HIP) ||  \
    defined(USE_MLU) || defined(USE_MACA) || defined(USE_HYGON) || \
    defined(USE_COREX)
                        if (cuda_device >= 0 && !staged) delete[] dram_buffer;
#endif
                        finalize(TransferStatusEnum::FAILED, false);
                        return;
                    }

                    bool progress_accepted = !progress_timeout_committed_;
                    if (progress_accepted && transferred_bytes > 0) {
#ifdef MOONCAKE_TCP_TRANSPORT_TEST_HOOKS
                        progress_accepted =
                            acceptProgressForTest(kSessionReadBodySuccess);
#else
                        progress_accepted = acceptProgress();
#endif
                    }
                    if (!progress_accepted) {
#if defined(USE_CUDA) || defined(USE_MUSA) || defined(USE_HIP) ||  \
    defined(USE_MLU) || defined(USE_MACA) || defined(USE_HYGON) || \
    defined(USE_COREX)
                        if (cuda_device >= 0 && !staged) delete[] dram_buffer;
#endif
                        finalize(TransferStatusEnum::FAILED, false);
                        return;
                    }

#if defined(USE_CUDA) || defined(USE_MUSA) || defined(USE_HIP) ||  \
    defined(USE_MLU) || defined(USE_MACA) || defined(USE_HYGON) || \
    defined(USE_COREX)
                    if (cuda_device >= 0 && !staged) {
                        cudaSetDevice(cuda_device);
#ifdef USE_MACA
                        cudaError_t cuda_status =
                            copyTcpCudaMemory(addr + total_transferred_bytes_,
                                              dram_buffer, transferred_bytes);
#else
                        const int64_t ph0 = tcpPhaseTiming() ? phaseNowNs() : 0;
                        cudaError_t cuda_status =
                            cudaMemcpy(addr + total_transferred_bytes_, dram_buffer,
                                       transferred_bytes, cudaMemcpyDefault);
#endif
#ifndef USE_MACA
                        PHASE_LOG("MC_PH3_SCATTER t0="
                                  << ph0 << " t1=" << phaseNowNs()
                                  << " bytes=" << transferred_bytes
                                  << " id=" << static_cast<void*>(this));
#endif
                        if (cuda_status != cudaSuccess) {
                            LOG(ERROR)
                                << "ClientSession::readBody failed to copy to CUDA "
                                   "memory. "
                                << "Error: " << cudaGetErrorString(cuda_status);
                            delete[] dram_buffer;
                            finalize(TransferStatusEnum::FAILED, false);
                            return;
                        }
                        delete[] dram_buffer;
                    }
#endif
                    total_transferred_bytes_ += transferred_bytes;
                    readBody();
                }));
    }

    void writeBody() {
        auto self(shared_from_this());
        uint64_t size = le64toh(header_.size);
        char* addr = local_buffer_;

        size_t buffer_size =
            std::min(getChunkSize(), size - total_transferred_bytes_);
        if (buffer_size == 0) {
            // The payload has fully entered the socket; the pinned staging
            // slot (if any) is no longer needed.
            releaseStaging();
            if (v2_) {
                if (write_abort_requested_) {
                    finalize(TransferStatusEnum::FAILED, false);
                    return;
                }
                // Completion comes from the server's acknowledgment, whose
                // read is already in flight (armed in writeHeader) and may
                // have finished first.
                write_body_done_ = true;
                if (write_acked_ok_) {
                    finalize(TransferStatusEnum::COMPLETED, true);
                } else {
                    // From here a well-behaved server owes at most one
                    // chunk's apply plus the frame; a legacy peer behind a
                    // stale descriptor may owe nothing, ever.
                    armStatusDeadline();
                }
            } else {
                // v1: no acknowledgment exists in the protocol; this only
                // means the payload left the initiator (#2086).
                finalize(TransferStatusEnum::COMPLETED, true);
            }
            return;
        }

        char* dram_buffer = addr + total_transferred_bytes_;
        int cuda_device = -1;
        bool staged = false;

#if defined(USE_CUDA) || defined(USE_MUSA) || defined(USE_HIP) ||  \
    defined(USE_MLU) || defined(USE_MACA) || defined(USE_HYGON) || \
    defined(USE_COREX)
        if (staging_slot_) {
            // Whole request already staged into the pinned slot; stream the
            // remaining bytes straight from it.
            dram_buffer = staging_slot_->buffer + total_transferred_bytes_;
            staged = true;
        } else {
            cuda_device = getCudaDeviceId(addr);
            if (cuda_device >= 0) {
                if (total_transferred_bytes_ == 0) {
                    // Fast path: stage the entire request with one
                    // asynchronous ordered copy, then send from the pinned
                    // slot. Any acquire/stage failure falls through to the
                    // legacy per-chunk staging below.
                    auto* pool = GpuStagingPool::instance();
                    if (auto* slot =
                            pool->acquire(static_cast<size_t>(size))) {
                        if (pool->stageDeviceToHost(slot, addr,
                                                    static_cast<size_t>(
                                                        size))) {
                            staging_slot_ = slot;
                            dram_buffer = slot->buffer;
                            staged = true;
                        } else {
                            pool->release(slot);
                        }
                    }
                }
                if (!staged) {
                    dram_buffer = new char[buffer_size];
                    cudaSetDevice(cuda_device);
#ifdef USE_MACA
                    cudaError_t cuda_status = copyTcpCudaMemory(
                        dram_buffer, addr + total_transferred_bytes_,
                        buffer_size);
#else
                    cudaError_t cuda_status =
                        cudaMemcpy(dram_buffer, addr + total_transferred_bytes_,
                                   buffer_size, cudaMemcpyDefault);
#endif
                    if (cuda_status != cudaSuccess) {
                        LOG(ERROR)
                            << "ClientSession::writeBody failed to copy from "
                               "CUDA memory. "
                            << "Error: " << cudaGetErrorString(cuda_status);
                        delete[] dram_buffer;
                        abortWrite();
                        return;
                    }
                }
            }
        }
#endif

        write_body_in_flight_ = true;
        asio::async_write(
            *socket_, asio::buffer(dram_buffer, buffer_size),
            asio::bind_executor(
                strand_,
                [this, addr, dram_buffer, cuda_device, staged, self](
                        const asio::error_code& ec, std::size_t transferred_bytes) {
                    write_body_in_flight_ = false;
                    TCP_DBG("C writeBody chunk ec=" << ec.value()
                                                    << " bytes=" << transferred_bytes
                                                    << " total=" << total_transferred_bytes_);
#if defined(USE_CUDA) || defined(USE_MUSA) || defined(USE_HIP) ||  \
    defined(USE_MLU) || defined(USE_MACA) || defined(USE_HYGON) || \
    defined(USE_COREX)
                    if (cuda_device >= 0 && !staged) {
                        delete[] dram_buffer;
                    }
#endif
                    if (ec) {
                        LOG(ERROR)
                            << "ClientSession::writeBody failed. "
                            << "Attempt to write data " << static_cast<void*>(addr)
                            << " using buffer " << static_cast<void*>(dram_buffer)
                            << ". Error: " << ec.message()
                            << " (value: " << ec.value() << ")";
                        abortWrite();
                        return;
                    }
                    if (write_abort_requested_) {
                        // An early ACK failure or a committed progress timeout
                        // closed the socket while this operation still owned the
                        // caller's source buffer. It is safe to publish failure
                        // now that the handler has run.
                        finalize(TransferStatusEnum::FAILED, false);
                        return;
                    }
                    if (transferred_bytes > 0 && !acceptProgress()) {
                        finalize(TransferStatusEnum::FAILED, false);
                        return;
                    }
                    total_transferred_bytes_ += transferred_bytes;
                    writeBody();
                }));
    }
};

struct TcpContext {
    TcpContext(short port, ValidateAddrFn validate_addr)
        : acceptor(io_context),
          work_guard_(asio::make_work_guard(io_context)),
          validate_addr_(std::move(validate_addr)) {
        std::error_code ec;
        asio::ip::tcp::endpoint endpoint(asio::ip::tcp::v6(), port);

        acceptor.open(endpoint.protocol(), ec);
        if (!ec) {
            acceptor.set_option(asio::ip::v6_only(false), ec);
            if (!ec) {
                acceptor.set_option(
                    asio::ip::tcp::acceptor::reuse_address(true));
                acceptor.bind(endpoint, ec);
                if (!ec) {
                    acceptor.listen();
                    return;
                }
            }
            acceptor.close();
        }
        LOG(ERROR) << "Failed to set up IPv6 dual-stack listener: "
                   << ec.message() << " (error code: " << ec.value() << ")";
        asio::ip::tcp::endpoint endpoint_v4(asio::ip::tcp::v4(), port);
        acceptor.open(endpoint_v4.protocol());
        acceptor.set_option(asio::ip::tcp::acceptor::reuse_address(true));
        acceptor.bind(endpoint_v4);
        acceptor.listen();
    }

    void doAccept() {
        acceptor.async_accept([this](asio::error_code ec, tcpsocket socket) {
            if (!ec) {
                asio::error_code nodelay_ec;
                socket.set_option(asio::ip::tcp::no_delay(true), nodelay_ec);
                auto socket_ptr =
                    std::make_shared<tcpsocket>(std::move(socket));
                auto session =
                    std::make_shared<ServerSession>(socket_ptr, validate_addr_);
                session->start();
            }
            doAccept();
        });
    }

    asio::io_context io_context;
    // Keeps run() blocked on the event loop even when no transfer is active.
    // Without it, a worker thread that observes a transient work-free moment
    // (e.g. before the first doAccept) returns from run(), after which the
    // io_context is stopped and every subsequent run() returns immediately:
    // the thread busy-spins and handlers posted later are never executed.
    // Released in shutdown before io_context.stop().
    asio::executor_work_guard<asio::io_context::executor_type> work_guard_;
    asio::ip::tcp::acceptor acceptor;
    ValidateAddrFn validate_addr_;
};
