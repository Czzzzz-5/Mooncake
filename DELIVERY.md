# gpu-staging-v1 分支交付说明

> 面向 review 的导览：改了什么、为什么、效果如何、怎么复现。
> 实验数据与实验卡在私有仓库 `Czzzzz-5/Mooncake-TTFT`。

## 一句话结论

128k token KV cache 跨节点传输（TCP），本分支比 main（e58ad953）快
**1.6–2.1 倍**（有效带宽 ~3.1 vs ~1.5 GB/s）。优势来源：把"显存↔内存拷贝"
和"网络收发"从每 64KB 小块的严格串行，改成流水线交叠。

## main 的问题

main 的 TCP transport 对每个 64KB slice 走"临时页缓存 + 同步 cudaMemcpy +
socket 收发"的严格串行路径。实测每条 lane 的时间线上 ~14% 在拷显存、
~86% 在等 socket——拷贝期间网络空等，这是带宽天花板 ~1.5 GB/s 的结构性原因。

## 本分支的改动

| commit | 内容 |
|---|---|
| bbd3f18f | engine 侧 gather：submit_gather_write + 多段池化 gather |
| 91141140 | 接收侧 scatter slots（与 gather 镜像） |
| 27a83e8a | batched gather/scatter（cudaMemcpyBatchAsync）、lane 并行 split writes、slot 上限提升 |
| 716d366d | 测量打点：`MC_TCP_PHASE_TIMING=1` 时输出 `[XDBG]` 分段事件，默认关闭 |

核心设施是 `mooncake-transfer-engine/src/transport/tcp_transport/gpu_staging_pool.h`：
每个槽位一块持久锁页内存 + 非阻塞 stream + event，整请求一次异步拷贝进/出槽，
socket 层直接读写槽内存——D2H、网络、H2D 三段由此流水交叠。
槽位过小或耗尽会回退 legacy 路径（有一次性 WARNING），部署时注意
`MC_TCP_GPU_STAGING_SLOT_MB` ≥ 单次传输上限（实验用 192MB）。

## 实验（2026-10-26，同机同轮 A/B）

- 方法：realshape bench——P/D 双进程按真实模型 KV cache 布局（仿 vLLM v1）
  分配显存并注册，D 侧 `batch_transfer_sync_read` 拉回；16 档 token
  （128 → 1M）× 每档 3 次实测 + 1 次 warmup；收端做数据图案校验
- 模型 shape：k3（24 层 MLA，27.6 KB/tok）、dsv3（61 层 MLA，70.3 KB/tok）、
  qwen7b（28 层 GQA，56 KB/tok）
- 环境：同机 2× RTX 5090，TCP loopback；A = main e58ad953（另加等效的
  测量打点 patch，见下），B = 本分支；唯一变量是 mooncake 版本

### 128k 档结果（每档 3 次实测取平均）

| 模型 | 数据量 | main | 本分支 | 提速 |
|---|---|---|---|---|
| k3 | 3.6 GB | 2056 ms | 1295 ms | **1.59x** |
| dsv3 | 9.2 GB | 5963 ms | 2896 ms | **2.06x** |
| qwen7b | 7.5 GB | 4919 ms | 2445 ms | **2.01x** |

分段解剖（128k）：main 单 lane 内拷/发串行，copy 占 lane 时间线 ~14%（求和口径）；
本分支 D2H/网络/H2D 三段 span 各自 ≈ 总时间，近乎完全交叠（span 口径）。
**口径警告**：main 分段是多 lane 求和（求和 > 墙钟，看结构占比），本分支是
事件 span；跨分支只能比总时间/带宽，分段只在分支内部解读。

## 复现

代码与数据均在 `Czzzzz-5/Mooncake-TTFT`：

- bench / 解析脚本：`scripts/`（`my_realshape_bench.py`、`run_pair.sh`、
  `parse_realshape_phases.py`、`parse_main_phases.py`、`ab_compare.py`）
- 原始数据：`results/raw/2026-10-26-realshape/`（6 份 jsonl + 两侧 P/D 日志 .gz
  + ours 分段表）
- 实验卡：`experiments/2026-10-26-main-vs-ours-realshape.md`
- main 对照组打点 patch：`patches/mooncake-main-phase-timing.patch`
  （打在 main e58ad953 上，输出 `[MC-PHASE]` copy/io 计时，纯测量）

一轮 A/B：`./run_pair.sh <k3|dsv3|qwen7b> <tag>`（脚本内已含所需环境变量；
P 侧 GPU1 后台、D 侧 GPU2 前台，跑完自动清理）。
