# FilterVector / UNG GPU 优化系统总览、性能边界与适用环境

更新时间：2026-07-01

本文是当前仓库的第一阅读入口。它只保留可审计结论，回答四个问题：

1. 这套系统包含什么。
2. 性能优势来自哪里。
3. 哪些环境下优势明显。
4. 哪些结论还不能写成学术 claim。

## 1. 系统包含什么

当前系统不是单一 GPU kernel，而是一套围绕 UNG filtered vector search 的 workload-aware 构建与查询优化。主要组成如下。

| 组成 | 当前实现 | 作用 | 状态 |
|---|---|---|---|
| build profile / runtime config | `UngBuildConfig`、`CrossEdgeGpuRuntimeConfig`、`TagoreCudaRuntimeConfig`、`SearchRuntimeConfig` | 把 CPU、naive GPU、paper fused、adaptive CUDA、diagnostic route 的选择显式化 | 已接入主流程 |
| group graph backend | CPU Vamana、bounded-complete fallback、packed exact-anchor、FastGrnndCuda / reverse-tail、adaptive route | 为每个 label group 构建组内图 | 已有 full-quality A/B，但不是 CPU Vamana 的无条件替代 |
| cross-edge backend | CPU exact / Vamana baseline、cuVS per-group baseline、SGEMM+topK baseline、grouped fused topK、universal flat double-buffer、X-side streaming | 构建跨 group 边 | grouped fused / universal 是当前主线；X-streaming 是显存边界路径 |
| output boundary | `CrossEdgeTopkOutputWriter`、`CrossEdgeHostOutputView`、`CrossEdgeGraphMaterializationWriter`、`CrossEdgeCsrOutput` | 统一 CUDA D2H / writeback / graph materialization / CSR adjacency 输出 | 已完成第一层接口化 |
| query route and search backend | `QueryRouteDecision`、`SearchEntryProvider`、`GraphSearchBackend` | 把 query route、entry group provider、邻接读取和 graph expansion 分开 | 已接入主流程 |
| query entry provider | CPU exact/min-super-set provider、GPU `gpu_cover_frontier` correct-cover provider | 从 query labels 生成 coverage-correct entry group ids | production route 已接入；GPU provider 当前是 per-query 串行 CUDA 调用 |

代码结构上，`UniNavGraph` 仍是较大的 facade，但 build、group graph、cross-edge、query feature、query route、entry provider、search backend、I/O、label graph 已经拆到独立文件。后续维护应继续把 backend 输入输出对象化，而不是再把新方法塞回 `uni_nav_graph.cpp` 或 `gpu_gemm_topk.cu`。

## 2. 性能优势来自哪里

优势主要来自三类 workload mismatch 的修正。

第一，cross-edge 不是一个全局大 GEMM，而是大量 group-level topK search。直接逐组调用 cuVS 或先 SGEMM 再 separate topK 会产生重复 packing、临时矩阵和输出写回开销。grouped fused topK / descriptor batching / universal flat double-buffer 把相同形态的 group workload 合并调度，并减少中间结果物化。

第二，group graph 的 group size 分布很不均匀。小组 GPU 化常被 H2D、packing、fill/writeback 吞掉收益；中组适合 packed exact-anchor；大组可以用 FastGrnndCuda / reverse-tail 做近似候选和质量修复。因此当前可靠设计是 adaptive router，而不是单一 GPU backend 替换 CPU Vamana。

第三，query entry group selection 在 many-group 场景中可能成为显著瓶颈。CPU exact minimal 输出少但代价高；GPU correct-cover 用 bitset/frontier/descendant coverage 批处理候选 group，保证 coverage，允许冗余 entry groups。它适合 label/group 数量大、entry scan 占 query latency 高的场景。

## 3. 已有性能证据

下表只列当前较稳的代表性结果。长表和原始 artifact 仍以 `docs/reports/TECHNICAL_REPORT_OPTIMIZATION_SPEEDUP_CN.md`、`docs/papers/EVIDENCE_MATRIX_CN.md` 和 runbook 中登记路径为准。

| 模块 | workload / 环境 | 结果 | 严谨表述 |
|---|---|---:|---|
| cross-edge | Amazon 1%x100，strong-baseline fairness | final fused cross `848.9 ms`；相对 CPU exact 128T `2.82x`，相对 cuVS per-group `4.39x`，相对 SGEMM+topK `1.64x` | 可写成 cross-edge stage 加速；不能推出所有 full-quality index 都加速 |
| cross-edge | Amazon 1%x200 full-quality | universal cross `2494.21 ms`，L1000/L5000 `0.871/0.911` | 可写成 full-quality 正结果；仍需区分 group graph / cross-edge / additional_edges |
| group graph | Amazon 1%x200 full-quality | packed exact-anchor group `3827.28 ms`，约 `2.22x` vs CPU group；L1000/L5000 `0.869/0.908` | 可写成中组 exact-anchor route 的正结果 |
| group graph | Amazon 10%x40 full-quality | Index `1.53x`，group graph `3.54x`，L100/L500/L1000 不低于 CPU | 可写成 many-group stress 下 adaptive route 有端到端 index 收益 |
| query entry | Amazon 100%x40，batch-query pair sweep | 相对 CPU scan 128T 的 `19.85x` 只是弱 baseline；宽松质量约 `2170` groups 时 CPU 更快；fused compact 后中高质量约 `1300~1500` groups 时 GPU 约 `2.7~3.0x` 优于同质量 CPU；`nq=4096/10240` scaling 稳定约 `3x` | 只能写 entry provider batch microbenchmark；GPU 不是全区间占优；production route 已接入但仍是 per-query CUDA，还缺 batch/end-to-end search A/B |
| scale boundary | Amazon 100%x40，A6000 48GB | old resident all-X 约需 `71.5GB`，X-side streaming 可完成 strict skip-additional build | 可写成显存边界绕过；不能写成性能主结果 |

## 4. 适用环境

这些优化最容易体现优势的环境：

| 环境特征 | 为什么有利 |
|---|---|
| group 数量多，cross-edge target/query 组合多 | descriptor batching 和 fused topK 能摊薄 per-group launch、packing、writeback 开销 |
| group size 分布长尾明显 | adaptive router 能避免小组 GPU 化负收益，并把中/大组交给更合适 backend |
| query labels 导致 naive CPU entry scan 占比高 | GPU correct-cover 提供 coverage-correct 批处理路径；fused compact 后 GPU 在中高质量约 `1300~1500` output groups 区间约 `2.7~3.0x` 更好，且 batch query scaling 支持高并发吞吐；宽松质量下 CPU 更好 |
| GPU 有足够带宽和显存容纳主要工作集，或 workload 可 streaming | resident all-X / direct-qid / flat output 能减少重复 H2D；显存不足时 X-streaming 至少提供可完成路径 |
| 评估使用 full-quality 口径 | 才能同时检查 additional_edges、recall、Index time、search latency，避免只优化单阶段 |

优势不明显或不应夸大的环境：

| 环境特征 | 风险 |
|---|---|
| 大量 very-small groups | GPU packing/H2D/fill 可能超过计算收益，bounded-complete 或 CPU fallback 更合理 |
| graph search 被随机访存和 cache miss 主导 | entry provider 或 build 加速不一定转化为 query latency |
| output boundary 仍需 CPU-compatible per-node graph | D2H/writeback/allocator 可能成为主瓶颈 |
| 只跑 skip-additional 或 microbenchmark | 不能直接声称 full-quality recall 或端到端性能 |
| GPU correct-cover 输出远多于 CPU exact minimal | entry group 阶段更快，但后续 entry point materialization 和 graph expansion 可能增加 |

## 5. 学术 claim 边界

可以写：

- 该系统针对 UNG 的不规则 group workload 设计了 workload-aware GPU backend 和 router。
- grouped fused topK 在 cross-edge stage 上优于强 CPU exact baseline、cuVS per-group baseline 和 SGEMM+topK baseline，具体成立于已报告的 Amazon sampled workload。
- adaptive group graph route 在 Amazon 10%x40 full-quality stress test 中实现了端到端 index 加速，并保持报告的 recall 口径不低于 CPU。
- GPU correct-cover entry provider 已在 microbenchmark 中证明 coverage-correct 语义和批处理 GPU 路径，并已接入 production search route；fused compact 后 GPU 在中高质量约 `1300~1500` output groups 区间达到约 `2.7~3.0x` 同质量 CPU 加速，但相对 CPU scan 128T 的 `19.85x` 不能作为公平 GPU-vs-CPU parallel 加速主张。
- X-side streaming 解决了 48GB A6000 上 100%x40 resident all-X cache OOM 的可完成性问题。

不能写：

- 不能写“GPU 无条件替代 CPU Vamana”。
- 不能把 entry provider 相对 CPU scan 128T 的 `19.85x` 写成相对同语义 CPU parallel baseline 的加速。
- 不能把 entry provider `19.85x` microbenchmark 写成端到端 query `19.85x`。
- 不能把 skip-additional 结果写成 full-quality recall 结果。
- 不能把 X-streaming v1 写成建图加速主结果；当前主要意义是绕过显存边界。
- 不能忽略负结果：heavy prune、小组全 GPU exact、qid-lock global merge、source-centric single-stage、x400 light prune recall gap 都必须保留为设计边界。

## 6. 当前缺口

| 优先级 | 缺口 | 最小验证 |
|---:|---|---|
| P0 | GPU query entry provider 已接入 production route，但还没有 batch/end-to-end A/B | 同 index/query/Lsearch 下比较 CPU provider vs `gpu_cover_frontier`，报告 total latency、entry ms、NumEntries、core search ms、recall |
| P0 | 100%x40 graph search 异常缺少细粒度解释 | 给 `iterate_to_fixed_point()` 增加 edge scans、visited hit/miss、queue insert/reject、cache-sensitive counters |
| P0 | 当前 best router 缺统一复测表 | 用同一脚本导出 CPU/current-best/diagnostic 的 full-quality 表 |
| P1 | flat/CSR search backend 需要端到端验证 | 对比 neighbor-list vs CSR backend 的 query latency、recall、一致性 |
| P1 | additional_edges 仍是 CPU-heavy boundary | 设计 staged edge buffer 或更低成本 writeback，并做 full-quality A/B |
| P1 | X-streaming `pack_q` 主导 | 引入 Q-side device cache/gather 或 source-centric two-stage streaming |

## 7. 推荐阅读路径

第一次阅读只需要：

1. 本文。
2. `docs/runbooks/UNG_METHOD_REGISTRY_CN.md`：确认每条方法的入口、开关和语义。
3. `docs/papers/EVIDENCE_MATRIX_CN.md`：确认每个 claim 的证据等级。
4. `docs/reports/TECHNICAL_REPORT_OPTIMIZATION_SPEEDUP_CN.md`：只在需要查原始细节时阅读。

其他长文档保留为证据库、历史记录或论文草稿，不应再作为当前状态的唯一依据。
