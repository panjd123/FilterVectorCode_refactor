# UNG GPU 优化现状、方法效果与缺口

更新时间：2026-06-26

本文是当前仓库的详细纠偏版总览。第一次阅读请先看更短、更严格的入口：

```text
docs/reports/SYSTEM_SCOPE_PERFORMANCE_AND_LIMITS_CN.md
```

本文只回答三个问题：

1. 现在代码里到底有哪些方法？
2. 哪些结果能作为主结论，哪些只能作为诊断或负例？
3. 下一步要补哪些实验或实现，才能把系统继续推进到更可信的论文状态？

如果需要直接向导师汇报，请使用更短的版本：

```text
docs/presentations/ADVISOR_BRIEFING_20260625_CN.md
```

## 0. 一句话结论

当前最稳的结论不是“GPU 无条件替代 CPU Vamana”，而是：

```text
UNG 是一个不规则 group workload。
cross-edge 适合 grouped fused topK / universal flat double-buffer；
group graph 需要按 group size 和质量风险做 workload-aware router；
query entry scan 可以用 GPU correct-cover 批处理改写，但公平加速结论必须和同语义 CPU parallel cover-frontier 比较；
但 CPU-compatible graph output、additional_edges 和大图 graph search 仍是主要边界。
```

代码结构的当前状态可以这样概括：方法层已经能用 `UngBuildConfig` / `CrossEdgeGpuRuntimeConfig` 表达主要 profile 和 route；cross-edge 的 descriptor、route-neutral planning、GPU 复用缓存/handle、topK device primitive、norm/topK update kernels、resident all-X upload/release helper、CPU tiny exact fallback helper、query packing/upload helper、bucket descriptor assembly helper、per-group dispatch helper、regular route/config builder、singleton fastpath kernel、large-group fused kernel、descriptor-batched TF32 kernel、small/medium/group-desc fused kernel、timing/result、double-buffer helper、SGEMM/separate-topK baseline helper、bucket fused descriptor launch、per-group fused launch、普通 batched output finalization、profiling summary、optional verify、X-streaming boundary helper、source-exact experimental helper、cuVS library baseline 已经从主流程中拆出；regular dispatch 已进一步收敛为 workload/config/buffers/counters 四类对象，planning、regular route、double-buffer route 和 benchmark filter 都显式消费 `CrossEdgeGpuRuntimeConfig&`，主 CUDA 路径不再通过 nullable route fallback 或本地 env parser 直接读调度 env。`GroupGraphRoute`、`GraphReserveSettings`、`CpuGroupGraphSettings`、`TagoreGroupGraphSettings`、`IntraGroupIdSettings`、`CpuHybridCrossSettings`、`GpuCrossLifecycleSettings`、`AdditionalEdgesSettings` 已移动到 `ung_build_settings.{h,cpp}`，把 graph reserve、CPU group graph、Tagore/FastGrnnd group graph、CPU hybrid cross、GPU cross lifecycle、additional_edges 的 env fallback 以及 group graph impl 到 CUDA/adaptive/prune-mode 的映射从 `uni_nav_graph.cpp` 执行循环中拿出去，并输出 `[graph_reserve] config`、`[group_graph] cpu_config`、`[TagoreCuda] config` summary 供实验审计。CPU/GPU group graph 已集中到 `uni_nav_graph_group_graph.cpp`；cross-edge backend resolve、CPU baseline/exact/hybrid、GPU route adapter、additional_edges 和 offset/merge/writeback orchestration 已集中到 `uni_nav_graph_cross_edges.cpp`；query/search 已拆成 `query_features`、`query_route`、`entry_provider`、`search_backend`、`search_orchestration` 五层文件，分别负责 selector 特征、路由策略、入口组 provider、ACORN/UNG graph expansion、线程调度和结果写回；`SearchEntryProvider`、`EntryGroupProviderRequest/Result`、`GraphSearchBackend` 和 CSR adjacency backend 已接入主流程。LNG/trie/coverage 已集中到 `uni_nav_graph_label_graph.cpp`；index save/load/statistics 已集中到 `uni_nav_graph_io.cpp`；vector-attribute graph 已集中到 `uni_nav_graph_vector_attr.cpp`；legacy ACORN augment 已集中到 `uni_nav_graph_acorn_augment.cpp`；profiling helper 已移动到 `ung_prof_log.{h,cpp}`。但 `uni_nav_graph.h` 仍是较大的 facade header，`gpu_gemm_topk.cu` 仍保留顶层高层执行分支，许多 backend helper 仍依赖 `UniNavGraph` 内部状态。也就是说，现在已经比临时实验状态清楚很多，但还没达到完全 backend 插拔化。

## 1. 当前主路径和边界

| 模块 | 当前主路径 | 可写结果 | 不能夸大 |
|---|---|---|---|
| cross-edge | `UNG_CROSS_EDGE_IMPL=1` + `UNG_GPU_TOPK_IMPL=3` + `UNG_UNIVERSAL_GPU=1` | x100 strong-baseline fairness：相对 CPU exact `2.82x`，相对 cuVS per-group `4.39x`，相对 SGEMM+topK `1.64x`；x200 full-quality universal cross `2494.21 ms`、L1000/L5000 `0.871/0.911` | 不是所有 workload 端到端加速；x100 full-quality Index 变慢，是 boundary result |
| group graph | `UNG_GROUP_GRAPH_IMPL=4` adaptive route；中组 packed exact-anchor/direct-H2D exact-anchor，大组 FastGrnndCuda / reverse-tail，小组 CPU/bounded fallback | x200 packed exact-anchor group `3827.28 ms`，相对 CPU `2.22x`，L1000/L5000 `0.869/0.908`；x200 direct-H2D+fill16 group `3204.88 ms`，相对 CPU `2.66x`，Lsearch sweep 显示 L20 低约 `0.010`、L50-L200 低 `0.001~0.0024`、L500 低 `0.004`、L1000 低 `0.002`、L2000/L5000 对齐；10%x40 full-quality Index `1.53x`，group `3.54x` | 不是 CPU Vamana 的逐边等价替代；不能写“无损普遍替代” |
| query entry scan | `gpu_cover_frontier` correct-cover provider | Amazon 100%x40、`nq=10240` 修正 pair sweep：相对 CPU scan 128T 的 `19.85x` 只是弱 baseline；宽松质量约 `2170` groups 时 CPU 更快；fused compact 后中高质量约 `1300~1500` groups 时 GPU 达到约 `2.7~3.0x` vs 同质量 CPU | 已接入 production route，但当前是 per-query 串行 CUDA provider；不能说端到端 query 已经加速；质量必须按实际输出组数比较 |
| output boundary | NeighborList64 / reserve / direct-H2D / flat-id / CSR backend ablations | 证明 `std::vector` graph materialization 和 host allocator 是真实瓶颈；x200 NeighborList64 reserve `25.4 ms`、fill `14.7 ms`；search 可显式选择 CSR adjacency backend | CSR backend 已接入但仍需端到端 latency/recall 复测；不能把 flat/CSR ablation 直接写成 full-quality 主结果 |
| 100%x40 scale | Storage 64-bit byte-count 修复 + `UNG_GPU_X_STREAMING=1` | 可完成 strict skip-additional build；解决 71.5GB resident-cache OOM 边界 | X-streaming v1 cross generate `697.1s`，`pack_q=450.3s`，不是加速主结果 |

## 2. 代码实现分类

### 2.1 Build profile

`UngBuildConfig` 已经把主要口径收敛到 profile：

| `UNG_BUILD_PROFILE` | 作用 | 论文/实验用途 |
|---|---|---|
| `original_cpu` | 尽量还原旧 CPU pipeline，使用旧 sort/LNG/cross-edge 路径 | 原始 CPU 对照或历史复核 |
| `current_cpu` | 当前优化后的 CPU baseline | 评估 GPU 之外的结构优化收益 |
| `naive_gpu` | GPU 工程 baseline，使用 SGEMM + separate topK | 对比“直接调用库/常规 GPU 化” |
| `paper_fused` | grouped fused topK 路径 | cross-edge 论文主线 |
| `custom` | 逐项环境变量控制 | ablation、调参、诊断 |

纠偏结论：这些 profile 应保留。它们不是历史垃圾，而是让 CPU、naive GPU、论文 fused 路径可比较的必要边界。

### 2.2 group graph 后端

| 实现 | 当前定位 | 效果判断 |
|---|---|---|
| `VamanaCpu` | 质量 baseline 和 fallback | 仍是最稳质量基线；不能删除 |
| `TagoreCuda` | 原始 Tagore 语义 wrapper | 用于说明 Tagore 候选生成/重 prune 的代价；不是最终推荐 |
| `GrnndLikeCuda` | 精度增强实验路径 | 保留为 ablation；不是主结果 |
| `FastGrnndCuda` | 大组候选生成 + reverse-tail/local prune | x400 reverse-tail 证明质量修复有效；但不能覆盖所有 group |
| `AdaptiveCuda` | 当前推荐 router | 用 group size 分流，避免小组 GPU 化和大组纯 exact 的风险 |

当前最清楚的 router 语义：

```text
small group  -> CPU Vamana 或 bounded-complete fallback
medium group -> packed exact-anchor
large group  -> FastGrnndCuda + reverse-tail / repair
```

重要纠偏：旧 `old batched exact kNN` 的低 recall 已被结构诊断判定为 cross/additional 口径混杂，不能继续作为 exact-anchor 不可行证据。现在应写成 confounded ablation。

### 2.3 cross-edge 后端

| 实现 | 当前定位 | 效果判断 |
|---|---|---|
| CPU Vamana cross | 原始慢 baseline | 可保留为弱 baseline |
| CPU exact scan 128T | 强 CPU baseline | x100 cross 对比必须报告，防止只打弱 baseline |
| cuVS per-group brute force | 库调用 baseline | 证明逐组调用库的 API/pack/writeback 开销大 |
| SGEMM + topK | 常规 GPU baseline | 证明 fused topK 省掉中间矩阵和独立 topK pass |
| grouped fused topK | 当前核心贡献 | x100 相对 SGEMM+topK `1.64x` |
| universal flat double-buffer | 当前工程主路线 | x200 full-quality 正结果；x100 boundary |
| X-side streaming | 100%x40 OOM 绕过路径 | scalability/boundary，不是性能主线 |

重要纠偏：`UNG_GPU_X_STREAMING=1` 是显存边界修复，不应写成主加速路径。它证明 resident all-X cache 不是唯一可行方式，但 v1 被重复 `pack_q` 主导。

### 2.4 query entry provider

`gpu_cover_frontier` 的语义是 coverage-correct，不是 exact minimal：

```text
C = {g | query_labels subset labels(g)}
F_raw = {g in C | |query_labels| <= |labels(g)| <= |query_labels| + delta}
F = first_cap(F_raw)
Covered = union(descendants(f) for f in F)
Output = F union (C - Covered)
```

这个算法能保证不漏候选 group：如果候选不被 frontier descendants 覆盖，就直接进入 `C - Covered`。代价是输出入口 groups 可能多于 CPU exact minimal。

当前 production search route 已接入 `gpu_cover_frontier` provider，但实现是 per-query 串行 CUDA 调用，用 mutex 保护共享 device state。修正后的 pair sweep 已经把 CPU/GPU 放到同一 `delta/cap` 和同一 group-id materialization 输出边界下比较。结论是 GPU 不是全区间占优：宽松质量约 `2170` groups 时 CPU 更快；fused compact 后中高质量约 `1300~1500` groups 时 GPU 达到 `2.7~3.0x` vs 同质量 CPU；最严格 `<1200` groups 当前 grid 里只有 CPU 达到。学术写作中还要补同 index/query/Lsearch 下的端到端 A/B，报告 total latency、entry ms、NumEntries、core search ms 和 recall。

## 3. 主要结果表

### 3.1 Build / index 结果

| workload | 当前方法 | Before | After | 加速比 | 质量口径 |
|---|---|---:|---:|---:|---|
| Amazon 1%x100 full-quality | adaptive CUDA | Index `6863.47 ms` | `5450.00 ms` | `1.26x` | L100/L500/L1000 `0.829/0.870/0.895`，基本对齐历史 |
| Amazon 1%x200 full-quality | packed exact-anchor | Index `17596.80 ms` | `11043.50 ms` | `1.59x` | L1000/L5000 `0.869/0.908`，对齐 CPU |
| Amazon 1%x400 coverage-query | packed exact / quality route | Index `36257.30 ms` | `28436.30 ms` | `1.27x` | L1000/L5000 `0.945/0.967`，CPU 约 `0.946/0.965` |
| Amazon 10%x40 full-quality | FastGrnndCuda + fallback | Index `36053.30 ms` | `23497.40 ms` | `1.53x` | L100/L500/L1000 `0.8668/0.8980/0.9103`，不低于 CPU |
| Amazon 100%x40 boundary | X-streaming strict skip-add | 无同口径 baseline | Index `945683 ms` | - | 只证明可完成，不证明加速 |

### 3.2 cross-edge 单阶段 / diagnostic

| workload | baseline | baseline cross | our cross | 加速比 | 口径 |
|---|---|---:|---:|---:|---|
| Amazon 1%x100 | CPU Vamana cross | `18735.8 ms` | `848.9 ms` | `22.07x` | baseline 很弱，但可作历史对比 |
| Amazon 1%x100 | CPU exact scan 128T | `2389.8 ms` | `848.9 ms` | `2.82x` | 强 CPU baseline |
| Amazon 1%x100 | cuVS per-group brute force | `3730.5 ms` | `848.9 ms` | `4.39x` | 库调用 baseline |
| Amazon 1%x100 | SGEMM + topK | `1395.9 ms` | `848.9 ms` | `1.64x` | 常规 GPU baseline |
| Amazon 1%x200 full-quality | partial DB route | `2657.23 ms` | `2494.21 ms` | `1.07x` | universal flat all-DB |
| Amazon 10%x40 diagnostic | CPU exact cross | `54879.3 ms` | `8475.4 ms` | `6.48x` | skip-additional，不能作 recall 主表 |

### 3.3 query entry scan

| workload | 方法 | total | avg output groups | QPS | speedup vs CPU scan |
|---|---|---:|---:|---:|---:|
| Amazon 100%x40, `nq=10240` | CPU scan 128T | `1187.56 ms` | `23332.2` | `8623` | `1.00x` |
| Amazon 100%x40, `nq=10240` | CPU exact minimal | `4569.83 ms` | `616.24` | `2241` | `0.26x` |
| Amazon 100%x40, `nq=10240` | CPU cover-frontier d1 cap64 | `16.06 ms` | `2172.57` | `637647` | `72.64x` |
| Amazon 100%x40, `nq=10240` | GPU cover-frontier d1 cap64 | `47.37 ms` | `2172.57` | `216187` | `24.63x` |
| Amazon 100%x40, `nq=10240` | CPU cover-frontier d2 cap256 | `195.68 ms` | `1384.20` | `52330` | `5.96x` |
| Amazon 100%x40, `nq=10240` | GPU cover-frontier d2 cap1024 | `127.32 ms` | `1336.57` | `80430` | `9.16x` |
| Amazon 100%x40, `nq=10240` | GPU fused-compact cover-frontier d2 cap1024 | `97.80 ms` | `1336.57` | `104704` | `11.63x` |
| Amazon 100%x40, `nq=10240` | GPU cover-frontier d3 cap8192 | `415.13 ms` | `1279.98` | `24667` | `2.81x` |
| Amazon 100%x40, `nq=10240` | GPU fused-compact cover-frontier d3 cap8192 | `385.50 ms` | `1279.98` | `26563` | `2.95x` |

注意：`19.85x` 只相对 CPU scan 128T。公平主比较必须同时看时间和输出组数。完整 pair sweep artifact 是 `/home/graphdb/FilterVectorResultsRefactor/organized_benchmarks/query_entry_group_pair_sweep_20260701_nocheck`。
fused compact 输出优化 artifact 是 `/home/graphdb/FilterVectorResultsRefactor/organized_benchmarks/query_entry_group_fused_compact_keypoints_20260701`；它把 full-bitset GPU 的 D2H 从约 `25 ms` 降到 `2~4 ms`，并在 `d2 cap1024` 上达到 `97.80 ms / 1336.57 groups`，相对同质量 CPU `290.70 ms / 1336.57 groups` 约 `2.97x`。重 OR 配置仍主要受 descendant bitset OR 显存带宽限制。
并行度 scaling artifact 是 `/home/graphdb/FilterVectorResultsRefactor/organized_benchmarks/query_entry_group_parallel_scaling_20260701`。固定 `d2 cap1024` 后，GPU fused compact 相对 CPU cover-frontier 的 speedup 随批量查询稳定在目标区间：`nq=4096` 为 `119.6 -> 38.87 ms`，约 `3.08x`；`nq=10240` 为 `294.4 -> 97.82 ms`，约 `3.01x`。因此 ELS 的 GPU 主张应写成 batch-query throughput，而不是 production per-query CUDA route。

端到端 query 影响的当前写法应是上界估算：

| workload | CPU entry 占比 | 替换 entry 后估算 speedup | 状态 |
|---|---:|---:|---|
| Amazon 10%x40 full-quality | `48~51%` | `1.9~2.0x` | production route 已接入；仍缺 batch/end-to-end A/B |
| Amazon 1%x200 / 1%x400 | `6~11%` | `1.07~1.12x` | scan 不是主瓶颈 |
| Amazon 100%x40 | entry scan 可高速化 | 未给正式端到端 speedup | 后续 graph search 随机访存异常仍需研究 |

## 4. 当前最重要的负结果和边界

| 负结果 / 边界 | 说明 | 对设计的影响 |
|---|---|---|
| old exact kNN 低 recall | 后续诊断显示 cross edges 缺失，不能隔离组内图质量 | 不再用它否定 exact-anchor |
| heavy prune | x200 `prune=19218.3 ms`，group graph 慢于 CPU | 大多数中组不能使用重 prune |
| all-small GPU exact | 10%x40 小组搬到 GPU 后被 pack/H2D/fill 吃掉收益 | 小组应 fallback 或用更轻的 CPU-compatible 构造 |
| qid-lock global merge | D2H/writeback 降了，但 GPU lock/atomic 让 kernel 变慢 | 输出边界要做 segmented/no-lock，而不是简单加锁 |
| source-centric single-stage | 方向合理，但当前 kernel 并行度不足 | future 应做 two-stage source grouped GEMM + reduce |
| X-streaming v1 | 解决 OOM，但 `pack_q` 是绝对主项 | 需要 Q-side device cache/gather 或 source-centric 流水 |
| CPU exact minimal entry | 输出最少但太慢 | GPU correct-cover 是合理语义折中；compact 输出已降低 D2H，但还需减少 count/compact kernel 和 CPU materialize 成本，并做 production batch/end-to-end A/B |
| 100%x40 graph search | `DistCalcs/Visited/EntryGroups` 无法解释 core time 暴涨 | 需要细粒度 CPU search counters 和 cache-friendly graph layout |

## 5. 下一步补齐清单

### 5.1 必须补的实验

| 优先级 | 缺口 | 最小动作 | 验收标准 |
|---:|---|---|---|
| P0 | query entry GPU 已接入 production route，但尚未完成 batch/end-to-end A/B | 跑同 index/query/Lsearch 下 CPU provider vs `gpu_cover_frontier` A/B，并评估 batch/stream 化 | 同时报告 total latency、entry ms、entry groups、core search ms、recall |
| P0 | 100%x40 graph search 异常缺计数 | 给 `iterate_to_fixed_point()` 加 `edge_scans`、visited hit/miss、queue inserts/rejects、memmove bytes | 能解释 `core_search_time_ms` 为什么远大于 dist calc 变化 |
| P0 | current best router 缺统一复测表 | 对 x100/x200/x400/10%x40 用相同脚本导出 CPU/current best/diagnostic 表 | 主文只引用同口径 full-quality 表 |
| P1 | packed exact-anchor 泛化 | 补 10%x100、真实多标签或 CelebA full-quality A/B | 证明中组 route 不只适用于 Amazon repeat |
| P1 | additional_edges 仍是 CPU Vamana | 尝试 flat staged edge buffer 或 CPU exact additional full-quality A/B | 不降低 recall，且 additional 本体和 writeback 下降 |
| P1 | X-streaming pack_q | 做 Q-side device cache/gather 或 source-centric two-stage streaming | 100%x40 cross generate 不再被 host `pack_q` 主导 |
| P2 | flat/CSR graph backend | 引入 GraphView/CSR 并适配 search span view | 消除 CPU-compatible per-node `NeighborList` 输出边界 |

### 5.2 必须补的文档/论文口径

| 文档位置 | 要求 |
|---|---|
| `docs/papers/EVIDENCE_MATRIX_CN.md` | 继续作为 claim gate；每个新 claim 都必须有 artifact 和不能夸大的边界 |
| `docs/reports/TECHNICAL_REPORT_OPTIMIZATION_SPEEDUP_CN.md` | 保留长证据链，但不要作为第一次阅读入口 |
| `docs/presentations/0605.md` | 作为展示材料，只放主结果和边界，不再追加流水实验 |
| `docs/runbooks/UNG_BUILD_MODE_SWITCHES_CN.md` | 每新增一个开关必须写明“主路径 / diagnostic / negative / boundary” |

## 6. 当前仓库清理建议

### 6.1 应保留

- `UngBuildConfig` 的 profile 和环境变量分流。
- CPU Vamana、CPU exact、cuVS、SGEMM+topK 等 baseline。
- `TagoreCuda` / `GrnndLikeCuda` / `FastGrnndCuda` 作为方法演进和 ablation。
- X-streaming，作为 100%x40 可完成性边界和 OOM fallback。
- 历史报告，但应通过 `docs/README.md` 标明“历史/证据来源”，不要作为当前主结论入口。

### 6.1.1 当前接口清晰度

当前代码已经能清楚表达“跑哪个方法”，但还没有完全做到“任意新增方法都只实现一个小接口”。建议按下面的成熟度理解：

| 模块 | 已经清楚的接口 | 还欠缺的接口 | 当前风险 |
|---|---|---|---|
| profile/config | `UngBuildConfig` / `UngBuildProfile` / `Ung*Impl` enums | 统一把部分底层 tuning 上升到 typed config | 自定义 env 太多时仍难复现 |
| group graph | `TagoreGroupRequest` / `TagoreBuildResult`；`TagoreGroupBuildContext` 已统一 route/settings/runtime config/prune mode/exact threshold；`TagoreCudaRuntimeConfig` 已显式传入 CUDA batch builder | 正式 `GroupGraphBackend` 对象和统一 result/writeback contract | helper 仍依赖 `UniNavGraph` 内部状态；context 还不是独立 backend object |
| cross-edge | `CrossEdgeGpuRuntimeConfig` / `CrossEdgeBuildResult` / writeback mode | `CrossEdgeBackend` 和 `CrossEdgeOutputWriter` | CUDA helper 已拆多文件，但 route/config 仍很宽 |
| query route | `SearchRuntimeConfig` / `QueryRouteDecision` | `GraphSearchBackend` | 当前仍绑定 CPU `_graph` 和 `SearchCache` |
| entry group | `EntryGroupProviderRequest` / `EntryGroupProviderResult` / `run_entry_group_provider()` | production GPU provider compact output | `gpu_cover_frontier` 仍是 benchmark/fallback，不是正式主路径；fallback 已显式记录在 result 中 |
| public header dependency | ACORN/Faiss、ONNX selector、CPU Vamana 和 Tagore build-settings 重头文件已下沉到 `.cpp`，`uni_nav_graph.h` 保留前置声明 | 继续降低 `uni_nav_graph.h` 对 backend 细节的依赖 | 头文件仍偏大，后续需要进一步拆 facade |

所以目前可以向外汇报：**方法边界已经可审计，实验入口已经清楚；但如果目标是长期维护和继续接入新 backend，还需要把 helper 层进一步对象化。**

### 6.2 应避免继续扩散

- 不要再新增一个未登记的“临时最快配置”文档。
- 不要把只用于阶段拆分的 build time 放进完整质量主表。
- 不要把独立 query-entry benchmark 写成端到端 query speedup。
- 不要继续用 old exact 低 recall 当作 exact-anchor 负例。
- 不要把 X-streaming v1 写成 100%x40 性能优化。

### 6.3 下一轮代码清理方向

当前不建议大面积删除 C++ 后端，因为这些后端仍是 baseline 或 ablation。更合理的代码清理是：

1. 把 cross-edge GPU 细分实现的入口收敛到少数 routing 函数，保留内部 helper，但减少外部可见开关。
2. 给 negative / boundary 路径的日志加明确前缀，例如 `[diagnostic]`、`[boundary]`。
3. 把 query-entry GPU provider 从 benchmark 工具接入正式 search path；不要重新引入只取 frontier、不能保证 coverage 的旧思路。
4. 把 graph output 从 `NeighborList` 过渡到 span/CSR view，减少 GPU 构建后必须回填 CPU per-node object 的成本。

2026-06-25 已完成一项低风险结构收敛：新增 `ung_cross_edge_config.{h,cpp}`，把 cross-edge GPU 的 profile-to-legacy-env bridge 和 runtime route 从 `uni_nav_graph.cpp` 中拆出。`CrossEdgeGpuRuntimeConfig` 现在统一表达 source-exact、universal flat、double-buffer、X-streaming chunk、Q device-gather、global merge、CPU tiny fallback、singleton fastpath、direct-qid fused、GEMM/cuBLAS fallback、naive CUDA tuning、DB chunk/tile 阈值、small/medium/large fused kernel shape 和 writeback mode，并显式传入 `gpu_cross_groups_search_all_batched()` 和 `gpu_cross_groups_search_x_streaming()`。这不是新的性能方法，而是为了让现有方法的接口边界更清楚，避免后续继续在 CUDA 深处裸读主要调度 env。

仍应继续收敛的开关包括更底层 debug/profiling 开关，以及少量历史 kernel 对照 micro-tuning。它们目前可以保留为 diagnostic/tuning，但不应被写成独立主方法。

## x200 exact-anchor low-L tuning note

低成本 exact-anchor head/tail 调参已测试：`head16_anchor1`、`head8_anchor1`、`head24_anchor0`、`head16_anchor4`。`head16_anchor4` 在 group `1613.47 ms` 下把 L500 recall 从 default `0.845` 提到 `0.850`，略高 CPU `0.849`，但 L20/L50 gap 仍未明显缩小。结构诊断显示出度、low-degree 和 WCC 不是问题；下一步质量修复应转向 reciprocal/reverse-tail augmentation 或局部 diversity pruning。artifact: `/home/graphdb/fv_runs/x200_quality_tune_exact_anchor_20260702/quality_tuning_summary.md`。

补充：x200 exact-anchor 低成本质量调参中，`head12_anchor4` 是当前最好点：group `1710.05 ms`，L20 `0.8118`（default `0.808867`，CPU `0.819`），L500 `0.850`（CPU `0.849`）。它在不损失建图性能的情况下修复 L500 并部分缩小 L20 gap，但 L20/L50 仍未大幅闭合，下一步应尝试 reciprocal/reverse-tail augmentation。artifact: `/home/graphdb/fv_runs/x200_quality_tune_exact_anchor_20260702/head12_anchor4`。

补充：`head12_anchor4_bidir` 是 x200 exact-anchor 当前最佳低成本质量增强点：group `2238.07 ms`，仍快于 default direct-H2D `3204.88 ms`；L20 recall `0.812067`（default `0.808867`，CPU `0.819`），L500 `0.850`（CPU `0.849`）。它修复 L500 并缩小约三成 L20 gap，但 L50/L100 gap 仍未大幅闭合。artifact: `/home/graphdb/fv_runs/x200_quality_tune_exact_anchor_20260702/head12_anchor4_bidir`。

补充：`head12_anchor4_bidir_rev4` 是 x200 exact-anchor 当前最佳质量增强点：group `1822.89 ms`，仍快于 default direct-H2D `3204.88 ms`；L20 recall `0.814533`（default `0.808867`，CPU `0.819`），把 L20 gap 缩小超过一半；L500 `0.850`（CPU `0.849`）。这是 group-aware reciprocal/reverse-tail augmentation 的正结果，但 L20/L50 仍未完全对齐 CPU。artifact: `/home/graphdb/fv_runs/x200_quality_tune_exact_anchor_20260702/head12_anchor4_bidir_rev4`。

补充：`head16_anchor4_bidir_rev4` 是 x200 exact-anchor 当前推荐性价比点：group `2063.79 ms`，仍快于 default direct-H2D `3204.88 ms`；L20 `0.813933`（default `0.808867`，CPU `0.819`），L50 `0.8164`，L500 `0.850`（CPU `0.849`）。相对 `head12_anchor4_bidir_rev4`，它的 L20 略低但 group/index 更稳，是当前更适合保留的质量增强配置。artifact: `/home/graphdb/fv_runs/x200_quality_tune_exact_anchor_20260702/head16_anchor4_bidir_rev4`。
