# UNG 构建实现切换开关

本文记录当前代码里每个构建阶段可选择的实现。目标是让 CPU baseline、工程 GPU baseline、论文 fused kernel 路径都能在同一套代码里用环境变量表达。

## 总览

| 阶段 | 环境变量 | 取值 | 含义 |
| --- | --- | --- | --- |
| profile | `UNG_BUILD_PROFILE` | `custom` | 逐项读取下面的实现开关 |
| profile | `UNG_BUILD_PROFILE` | `original_cpu` | 旧 CPU pipeline：跳过 descendants / coverage / roaring / vector-attr / GPU |
| profile | `UNG_BUILD_PROFILE` | `current_cpu` | 当前 CPU baseline |
| profile | `UNG_BUILD_PROFILE` | `naive_gpu` | 工程 GPU baseline，SGEMM topK |
| profile | `UNG_BUILD_PROFILE` | `paper_fused` | 论文 fused topK 路径 |
| 组内图构建 | `UNG_GROUP_GRAPH_IMPL` | `0` | CPU Vamana |
| 组内图构建 | `UNG_GROUP_GRAPH_IMPL` | `1` | TagoreCuda：进程内直接调用 Tagore CUDA kernels，不经过 Python、磁盘 index 或进程创建 |
| 组内图构建 | `UNG_GROUP_GRAPH_IMPL` | `2` | GrnndLikeCuda：TagoreCuda 候选生成 + GPU sampled reverse / alpha prune refine，用于精度增强实验 |
| 组内图构建 | `UNG_GROUP_GRAPH_IMPL` | `3` | FastGrnndCuda：Tagore GNN-Descent 候选生成 + reverse-augmented local RNG prune，跳过 Tagore 重 prune |
| 组内图构建 | `UNG_GROUP_GRAPH_IMPL` | `4` | AdaptiveCuda：小组 fallback，中组 packed exact-anchor，大组 FastGrnndCuda，并发 CPU fallback 与 GPU batch |
| minimal supersets | `UNG_GET_MIN_SUPER_SETS_IMPL` | `0` | 当前 bucket/缓存优化 |
| minimal supersets | `UNG_GET_MIN_SUPER_SETS_IMPL` | `1` | 旧版 sort + includes |
| LNG 构建 | `UNG_LNG_IMPL` | `0` | 当前优化版 Phase1，复用线程本地 buffer |
| LNG 构建 | `UNG_LNG_IMPL` | `1` | 兼容慢路径，每个 group 分配临时 vector |
| LNG 构建 | `UNG_LNG_IMPL` | `2` | 原始 CPU LNG 函数 |
| descendants | `UNG_DESCENDANTS_IMPL` | `0` | 当前优化版 epoch BFS |
| descendants | `UNG_DESCENDANTS_IMPL` | `1` | 兼容慢路径，unordered_set visited BFS |
| coverage | `UNG_COVERAGE_IMPL` | `0` | legacy topological merge |
| coverage | `UNG_COVERAGE_IMPL` | `1` | descendants_direct |
| cross-edge | `UNG_CROSS_EDGE_IMPL` | `0` | CPU Vamana fixed-point |
| cross-edge | `UNG_CROSS_EDGE_IMPL` | `1` | GPU batched exact topK |
| cross-edge | `UNG_CROSS_EDGE_IMPL` | `2` | 原始 CPU cross-edge 函数 |
| cross-edge | `UNG_CROSS_EDGE_IMPL` | `3` | CPU exact scan；强 CPU baseline，用于和 GPU fused / cuVS 公平比较 |
| cross-edge | `UNG_CROSS_EDGE_IMPL` | `4` | CPU hybrid scan/Vamana；启发式诊断路径 |
| cross-edge | `UNG_CROSS_EDGE_IMPL` | `5` | cuVS per-group brute force baseline；用于证明逐组调用库的外围开销 |
| additional edges | `UNG_ADDITIONAL_EDGES_IMPL` | `0` | CPU Vamana 补边 |
| additional edges | `UNG_ADDITIONAL_EDGES_IMPL` | `1` | 跳过补边，用于拆分测试 |
| additional edges | `UNG_ADDITIONAL_EDGES_IMPL` | `2` | CPU exact scan 补边；用于 additional_edges ablation，不是默认 full-quality |
| GPU topK | `UNG_GPU_TOPK_IMPL` | `0` | auto，保留底层细粒度开关 |
| GPU topK | `UNG_GPU_TOPK_IMPL` | `1` | custom naive CUDA dot + topK |
| GPU topK | `UNG_GPU_TOPK_IMPL` | `2` | SGEMM + separate topK |
| GPU topK | `UNG_GPU_TOPK_IMPL` | `3` | group fused topK 路径 |

cross-edge GPU 细分开关：

| 环境变量 | 默认值 | 含义 |
| --- | ---: | --- |
| `UNG_UNIVERSAL_GPU` | `0` | 当前 cross-edge 主工程路线开关；启用 target-centric descriptor batching、double-buffer execution、GPU global merge 和 flat-id output |
| `UNG_GPU_DB_NOSPLIT` | universal 下默认 `1` | 让可覆盖 group 优先走 double-buffer descriptor batch |
| `UNG_GPU_DB_LARGE_MAX_NX` | universal 下默认 `1048576` | universal route 中允许进入 double-buffer 的最大 `nx`；调低可强制大组 fallback |
| `UNG_GPU_FLAT_ID_WRITEBACK` | auto | flat-id 输出，避免大量 `SearchQueue`/per-group container 物化；universal route 会自动启用 |
| `UNG_GPU_SOURCE_EXACT` | `0` | source-centric no-lock 实验路径；当前不是主线 |
| `UNG_GPU_SOURCE_EXACT_MODE` | `0` | `0` 为 CUDA-core exact，`1` 为 legacy/实验性 TF32 WMMA source path |
| `UNG_GPU_X_STREAMING` | `0` | 强制使用 X-side streaming，避免 resident all-X device cache；100%x40 OOM 边界路径，不是性能主线 |
| `UNG_GPU_X_STREAMING_AUTO` | `1` | resident prepare_all 失败且输出仍为 SearchQueue 时，自动尝试 X-side streaming |
| `UNG_GPU_PREPARE_DIRECT_HOSTREG` | `1` | resident all-X prepare 阶段允许对连续 host storage 使用 `cudaHostRegister`；现在读取点在 `CrossEdgeGpuRuntimeConfig::vector_upload` |
| `UNG_GPU_PREPARE_DIRECT_PAGEABLE` | `0` | resident all-X prepare 阶段允许直接 pageable H2D；主要用于诊断 copy/host-register 成本 |
| `UNG_GPU_PREPARE_HOSTREG_MAX_MB` | `4096` | 超过该大小时不尝试 `cudaHostRegister`，转向 direct pageable 或 pinned staging |
| `UNG_COUNT_VALID_PAIRS` | `0` | optional profiling：统计回传 topK 中的有效 pair 数；现在读取点在 `CrossEdgeGpuRuntimeConfig::diagnostics` |
| `UNG_GEMM_VERIFY_SAMPLES` | `0` | optional diagnostic：抽样用 CPU exact 校验 GPU topK 输出；默认关闭 |
| `UNG_GEMM_VERIFY_STRICT` | `0` | optional diagnostic：校验 mismatch 时是否直接失败；只在 `UNG_GEMM_VERIFY_SAMPLES>0` 时生效 |
| `UNG_BENCH_MIN_NX` | `0` | benchmark-only filter：只处理 `nx >= min` 的 target group；读取点在 `CrossEdgeGpuRuntimeConfig::benchmark_filter` |
| `UNG_BENCH_MAX_NX` | `1048576` | benchmark-only filter：只处理 `nx <= max` 的 target group；不应作为论文默认配置 |
| `UNG_BENCH_MIN_WORK_M` | `0` | benchmark-only filter：只处理 `nq * nx * dim >= M * 1e6` 的 target group |
| `UNG_GPU_X_CHUNK_MB` | `8192` | X-side streaming 的 target vector chunk 上限 |
| `UNG_GPU_X_STREAM_Q_MB` | `2048` | X-side streaming 的 query vector subchunk 上限 |
| `UNG_GPU_X_STREAM_OUT_MB` | `2048` | X-side streaming 的 topK output subchunk 上限 |

当前证据边界：

```text
UNG_UNIVERSAL_GPU=1 是重要工程路线和历史实验路径；当前 full-quality best cross-edge 口径以 `docs/reports/THREE_MAINLINES_METHOD_BASELINE_DATA_SPEEDUP_CN.md` 的最终性能表为准。
SIFT30 skip-additional cross 3180.63 ms；
Amazon 1% x200 full-quality cross 2494.21 ms，L1000/L5000 0.871/0.911；
Amazon 1% x100 full-quality cross 1689.40 ms，L100/L500/L1000 0.826/0.868/0.891，但 Index 变慢。
因此它是 x200 正结果和 x100 boundary result，不能写成无条件端到端加速。
source-centric 修复后 SIFT30 cross 5926.67 ms，只作为 future two-stage source grouped GEMM + reduce 方向。
X-side streaming 已让 Amazon 100%x40 strict skip-additional build 完成，但 cross generate `697.1s`，其中 `pack_q=450.3s`，因此只能写作 scalability/boundary result。
```

TagoreCuda 相关参数：

| 环境变量 | 默认值 | 含义 |
| --- | ---: | --- |
| `UNG_TAGORE_MIN_GROUP_SIZE` | `0` | 小于该大小的 group 不走 Tagore；`nx<=max_degree` 仍使用 complete graph |
| `UNG_TAGORE_K` | `64` | Tagore GNN-Descent 候选 K |
| `UNG_TAGORE_ITER` | `10` / `4` | Tagore GNN-Descent 迭代数；`UNG_GROUP_GRAPH_IMPL=2/3/4` 默认 `4`，原始 `TagoreCuda` 默认 `10` |
| `UNG_TAGORE_M` | `64` | Tagore pruning TOPM |
| `UNG_FAST_GRNND_LIGHT_PRUNE_NX` | `256` | FastGrnndCuda 中 `nx<=该值` 的 group 使用 light prune：直接保留 GNN-Descent 近邻并轻量去重，跳过 reverse candidate + RNG occlusion 重剪枝；设为 `0` 可回到旧 heavy prune |
| `UNG_FAST_GRNND_LIGHT_HEAD` | `final_degree-8` | light prune 中保留最近邻头部的数量；默认 `R=32` 时为 `24`，尾部 8 条从 GNN 候选尾部采样以增加多样性 |
| `UNG_FAST_GRNND_BATCH_EXACT_NX` | `0` | mixed exact/GNN router 的 exact-anchor 上限；在 `AdaptiveCuda` 中通常由 `UNG_ADAPTIVE_EXACT_MAX_NX` 间接设置。旧 pure-exact 低 recall 已被 cross/additional-edge mismatch 混淆，不能再作为 exact-anchor 负面结论 |
| `UNG_ADAPTIVE_EXACT_MAX_NX` | `4096` | `UNG_GROUP_GRAPH_IMPL=4` 的中组 exact-anchor 上限；实际仍受 `UNG_TAGORE_MIN_GROUP_SIZE` 约束 |
| `UNG_TAGORE_OVERLAP_FALLBACK` | adaptive 为 `1`，其它为 `0` | 让 CPU fallback group graph 与 GPU batch 构图并发 |
| `UNG_TAGORE_FALLBACK_THREADS` | `num_threads` | fallback path 的 OpenMP 线程数；用于扫 CPU/GPU overlap 的资源竞争 |
| `UNG_TAGORE_FILL_THREADS` | `min(num_threads,16)` | Tagore/FastExact 结果回填到 `Graph::neighbors` 的 OpenMP 线程数；过高会放大 `std::vector` allocator 竞争 |
| `UNG_TAGORE_FILL_FAST` | `0` | 跳过保护性去重的快速 host fill；exact packed 结果会走直接 memcpy 填充 |
| `UNG_GRAPH_RESERVE_CAPACITY` | auto | 在 group graph 构建前给 `Graph::neighbors` 预留容量，减少 GPU fill、bounded complete fallback 和 cross-edge merge 的 per-node vector 扩容；显式设为 `0/1` 可覆盖自动策略 |
| `UNG_GRAPH_RESERVE_ADDITIONAL_SLACK` | `num_cross_edges` | 每点额外预留的追加边容量 |
| `UNG_GRAPH_RESERVE_HARD_CAP` | `1048576` | 单点 reserve 上限，防止异常大组过度预留 |
| `UNG_GRAPH_RESERVE_AUTO_MIN_POINTS` | `500000` | auto 模式下，点数达到该阈值时启用 reserve |
| `UNG_GRAPH_RESERVE_AUTO_MIN_GROUPS` | `20000` | auto 模式下，group 数达到该阈值时启用 reserve |
| `UNG_GRAPH_RESERVE_AUTO_SMALL_PCT` | `50` | auto 模式下，小组 fallback 点数占比达到该百分比时启用 reserve |
| `UNG_FAST_EXACT_ANCHOR_TAIL` | `1` | batched exact ablation 的 deterministic anchor tail 开关；不能解决 recall 问题 |
| `UNG_FAST_EXACT_DIRECT_H2D` | `1` | batched exact-anchor 直接从已重排的 base vector 连续段 H2D，跳过临时 host pack |
| `UNG_FAST_EXACT_DIRECT_H2D_MAX_RUNS` | `4096` | direct H2D 最多允许的连续段数量；超过后回退 host pack，避免大量小 copy |
| `UNG_FAST_EXACT_DEVICE_LOOKUP` | `1` | 在 GPU 上生成 exact batch 的 point->group/local lookup，避免 CPU 逐点填表 |
| `UNG_FAST_GRNND_FORWARD_CAP` | `2*max_degree` | 旧 heavy prune 的 forward candidate 预算，仅用于调参实验 |
| `UNG_FAST_GRNND_REVERSE_CAP` | `max_degree` | 旧 heavy prune 的 sampled reverse candidate 预算，仅用于调参实验 |

兼容旧开关：`UNG_CROSS_EDGE_BACKEND=0/1` 仍然可用；如果同时设置 `UNG_CROSS_EDGE_IMPL`，以 `UNG_CROSS_EDGE_IMPL` 为准。

## 推荐口径

当前 CPU baseline：

```bash
UNG_GROUP_GRAPH_IMPL=0 \
UNG_LNG_IMPL=1 \
UNG_DESCENDANTS_IMPL=1 \
UNG_COVERAGE_IMPL=0 \
UNG_CROSS_EDGE_IMPL=0 \
UNG_ADDITIONAL_EDGES_IMPL=0 \
build_UNG_index ...
```

原始 CPU profile：

```bash
UNG_BUILD_PROFILE=original_cpu \
build_UNG_index ...
```

这个 profile 使用旧版 `get_min_super_sets`、旧版 LNG 和旧版 CPU cross-edge，并跳过当前新增的 `build_vector_and_attr_graph()`、descendants、coverage、Roaring bitsets、GPU cross-edge 和 ACORN/new-edge 路径。

工程 GPU baseline：

```bash
UNG_GROUP_GRAPH_IMPL=0 \
UNG_LNG_IMPL=0 \
UNG_DESCENDANTS_IMPL=0 \
UNG_COVERAGE_IMPL=1 \
UNG_CROSS_EDGE_IMPL=1 \
UNG_GPU_TOPK_IMPL=2 \
UNG_CROSS_EDGE_GPU_STRICT=1 \
build_UNG_index ...
```

论文 fused 路径：

```bash
UNG_GROUP_GRAPH_IMPL=0 \
UNG_LNG_IMPL=0 \
UNG_DESCENDANTS_IMPL=0 \
UNG_COVERAGE_IMPL=1 \
UNG_CROSS_EDGE_IMPL=1 \
UNG_GPU_TOPK_IMPL=3 \
UNG_CROSS_EDGE_GPU_STRICT=1 \
build_UNG_index ...
```

历史/工程 cross-edge universal 路径：

```bash
UNG_GROUP_GRAPH_IMPL=0 \
UNG_LNG_IMPL=0 \
UNG_DESCENDANTS_IMPL=0 \
UNG_COVERAGE_IMPL=1 \
UNG_CROSS_EDGE_IMPL=1 \
UNG_GPU_TOPK_IMPL=3 \
UNG_UNIVERSAL_GPU=1 \
UNG_ADDITIONAL_EDGES_IMPL=0 \
UNG_CROSS_EDGE_GPU_STRICT=1 \
build_UNG_index ...
```

该配置保留 CPU Vamana group graph 和 CPU Vamana `additional_edges`，只替换主 cross-edge 路径。用于论文时应同时报告 full-quality recall；如果设置 `UNG_ADDITIONAL_EDGES_IMPL=1` 跳过补边，只能作为 cross-edge kernel/output-boundary ablation。

Tagore 组内建图后端：

```bash
UNG_GROUP_GRAPH_IMPL=1 \
UNG_TAGORE_MIN_GROUP_SIZE=1024 \
UNG_TAGORE_K=64 \
UNG_TAGORE_ITER=10 \
UNG_TAGORE_M=64 \
build_UNG_index ...
```

这一路径会改变 group 内图的构造语义：它不是 CPU Vamana 的逐边等价替换，而是使用 Tagore 的 GPU GNN-Descent + Vamana pruning 输出图。当前 wrapper 保持 Tagore 原始语义，包括 float 输入转 half、原始 kernel launch 顺序和 Vamana pruning。

GrnndLikeCuda 精度增强后端：

```bash
UNG_GROUP_GRAPH_IMPL=2 \
UNG_TAGORE_K=64 \
UNG_TAGORE_ITER=4 \
UNG_TAGORE_M=64 \
build_UNG_index ...
```

这一路径不追求 CPU Vamana 的严格语义一致，而是优先服务 UNG search recall：在 Tagore 的 GPU pruning 后，额外采样局部出边的 reverse candidates，并在 GPU 上做一轮 alpha-style refine。当前测试里，把 `UNG_TAGORE_ITER` 从 `2` 提到 `4` 是性价比最高的精度增强；继续提高到 `8` 只带来很小 recall 增益，但 GNN-Descent 时间明显增加。

FastGrnndCuda 最终推荐后端：

```bash
UNG_GROUP_GRAPH_IMPL=3 \
UNG_TAGORE_K=64 \
UNG_TAGORE_ITER=4 \
UNG_FAST_GRNND_LIGHT_PRUNE_NX=256 \
build_UNG_index ...
```

这一路径采用自适应剪枝强度：先复用 Tagore GNN-Descent 生成候选；中等组默认走 diversified light prune，保留 GNN-Descent 近邻头部，并从候选尾部采样少量多样性邻居，避免 `nx≈200` 上 reverse candidate + RNG occlusion 无法摊销；大组继续使用 reverse-augmented local RNG pruning。它不是 GRNND 的直接复现，而是面向 UNG group graph 的后端替换：用可调剪枝强度在 build time 和 recall 之间取 Pareto。

Amazon 1% x200 coverage-query 实验中，旧 heavy prune 的 `prune=19218.3 ms`，group graph `26620.6 ms`；默认 diversified light prune 后，`prune=759.66 ms`，group graph `6706.36 ms`，Index `16478.5 ms`，快于 CPU Vamana Index `17596.8 ms`。当前代价是 L5000 recall `0.9056`，低于 CPU Vamana `0.9080`。早期 batched exact kNN 后端曾出现 L5000 `0.8199`，但后续结构诊断显示这是 cross/additional 口径混杂：组内图结构与新 packed exact-anchor 一致，差异来自 cross edges 缺失。因此旧 exact 低 recall 只能作为实验教训，不能再作为 exact-anchor 组内图不可行证据。当前 packed exact-anchor `nx4096` full-quality sweep 给出 x200 group `3827.28 ms`、L1000/L5000 `0.869/0.908`，是中组 router 的强候选。

详细方法总结见：`docs/reports/FINAL_GROUP_GRAPH_METHOD_CN.md`。

AdaptiveCuda conservative 推荐后端：

```bash
UNG_GROUP_GRAPH_IMPL=4 \
UNG_TAGORE_MIN_GROUP_SIZE=128 \
UNG_ADAPTIVE_EXACT_MAX_NX=512 \
UNG_TAGORE_OVERLAP_FALLBACK=1 \
UNG_TAGORE_FILL_FAST=1 \
UNG_ADDITIONAL_EDGES_IMPL=0 \
build_UNG_index ...
```

该路径把系统级替代路线拆成三段：`nx<128` 保留 CPU fallback 语义，`128<=nx<=512` 走 packed exact-anchor，`nx>512` 走 FastGrnndCuda。Amazon 1% x100 full-quality 测试中，它把 Index time 从历史 FastGrnnd full-quality `6370.67 ms` 降到 `5450 ms`，group graph 从 `3216.21 ms` 降到 `2773.35 ms`，L100/L500/L1000 recall 为 `0.829/0.870/0.895`，与历史 `0.828/0.871/0.896` 基本一致。

激进小组替代 ablation：

```bash
UNG_GROUP_GRAPH_IMPL=4 \
UNG_TAGORE_MIN_GROUP_SIZE=128 \
UNG_ADAPTIVE_EXACT_MAX_NX=512 \
UNG_TAGORE_FALLBACK_IMPL=0 \
UNG_GROUP_GRAPH_BOUNDED_COMPLETE=1 \
build_UNG_index ...
```

该路径将 fallback 小组改为 bounded-complete，Amazon 1% x100 skip-additional build 中 group graph 可降到 `1953.07 ms`，但 skip-additional L100/L500/L1000 recall 只有 `0.816/0.823/0.827`。因此它只能作为“CPU 小组 Vamana 是剩余瓶颈”的证据，不能作为 full-quality 主线。

固定单 group 合成测试中，`dim=128, max_degree=32, K=64, iter=10, M=64` 的经验分界点：

| `nx` | CPU Vamana mean build_graph | TagoreCuda mean build_graph | 结论 |
| ---: | ---: | ---: | --- |
| `256` | `29.55 ms` | `159.94 ms` | CPU 明显更快 |
| `512` | `53.68 ms` | `149.85 ms` | CPU 明显更快 |
| `1024` | `161.32 ms` | `148.61 ms` | 接近，Tagore 略快 |
| `2048` | `335.59 ms` | `163.34 ms` | Tagore 开始明显划算 |
| `4096` | `690.47 ms` | `181.77 ms` | Tagore 明显更快 |
| `8192` | `1513.12 ms` | `201.25 ms` | Tagore 明显更快 |

测试输出目录：`/tmp/fv_tagore_threshold_results_warm` 和 `/tmp/fv_tagore_threshold_results_small_warm`。

## 注意

`legacy_*` 慢路径是当前 refactor 代码内的兼容 baseline，用于表达“优化前风格”的算法选择；它不是旧仓库逐行还原。最初 CPU 版本的严格结果仍应来自旧代码 rerun 或对应历史 commit。
