# UNG 方法注册表

更新时间：2026-07-01

本文档登记当前代码中仍应被识别的方法、选择入口、实现入口、语义边界和成熟度。后续新增方法时，先更新这里，再更新实验脚本和论文证据矩阵。

## 1. 分类约定

| 分类 | 含义 |
|---|---|
| `main` | 当前推荐主路径或论文主贡献 |
| `baseline` | 必须保留的对照方法 |
| `ablation` | 用于解释设计选择的消融 |
| `diagnostic` | 用于定位瓶颈或公平比较，不应写成主系统结果 |
| `boundary` | 用于解决规模/显存/接口边界，不代表性能主张 |
| `negative` | 已知效果不佳，但保留为反例或后续参考 |

## 1.1 源码地图

当前代码已经按职责拆成几条主要路径。定位问题时优先按下表找入口，不要直接从 `uni_nav_graph.cpp` 开始全局搜索。

| 文件 | 主要职责 | 不应继续塞入的内容 |
|---|---|---|
| `UNG/codes/src/uni_nav_graph.cpp` | build 总调度、group storage 准备、LNG summary 诊断、intra-group finalize/offset helper | 新 backend、大段 save/load、CUDA route 细节、query generator、LNG/trie/coverage 细节、vector-attribute graph 细节、legacy ACORN 补边、global_graph 持久化兼容 |
| `UNG/codes/src/uni_nav_graph_label_graph.cpp` | trie grouping、`get_min_super_sets`、entry-group selection、bitmap helpers、LNG build、descendants、coverage、roaring bitset 初始化 | group graph 构建、cross-edge 构建、search graph expansion |
| `UNG/codes/src/uni_nav_graph_group_graph.cpp` | 组内图 CPU Vamana、complete/bounded fallback、Tagore/FastGrnnd/adaptive CUDA router、batch fill/writeback | cross-edge 逻辑、query 逻辑 |
| `UNG/codes/include/ung_group_graph_build_types.h` | group graph 构建中间结果类型：fallback stats、GPU batch artifacts、group partition | cross-edge route 配置、query 统计 |
| `UNG/codes/src/uni_nav_graph_cross_edges.cpp` | cross-edge CPU/GPU orchestration、backend resolve、additional_edges、merge/writeback、original CPU path | CUDA kernel 实现、group graph 构建 |
| `UNG/codes/src/uni_nav_graph_query_features.cpp` | selector warmup、Idea1/Idea2 feature order、feature-only CSV 诊断导出、pre-trie heuristic、entry-group route stats 汇总 | ACORN/UNG graph search 执行、build-time graph construction |
| `UNG/codes/src/uni_nav_graph_query_route.cpp` | query route policy：Idea1/Idea2 selector 调用、pre-trie heuristic 应用、force route 覆盖、route 统计填充；`make_search_runtime_config()` 统一 search runtime 字段构造 | ACORN/UNG graph search、entry-group provider dispatch、selector feature 表定义 |
| `UNG/codes/src/uni_nav_graph_entry_provider.cpp` | route 后 entry-group provider dispatch、CPU min-super-set provider、production GPU correct-cover provider callback、`ung_more_entry` 扩展应用 | query route decision、ACORN/UNG graph search、selector feature 表 |
| `UNG/codes/include/ung_gpu_cover_frontier_provider.h` / `UNG/codes/src/ung_gpu_cover_frontier_provider.cu` | `gpu_cover_frontier` production provider：从 group labels/LNG descendants 构造常驻 CUDA bitset 表，按 query 输出 coverage-correct entry groups | query route policy、graph search |
| `UNG/codes/src/uni_nav_graph_search_backend.cpp` | ACORN query backend、UNG query backend、entry-point expansion、UNG fixed-point graph expansion | query route policy、provider dispatch、thread orchestration |
| `UNG/codes/src/uni_nav_graph_search.cpp` | search runtime config consumption、per-query orchestration、thread dispatch、result writeback | query route policy、selector feature 表、entry-group provider 实现、backend graph expansion、build-time graph construction |
| `UNG/codes/include/ung_query_route.h` | query route 数据契约：`QueryRouteDecision` 表达 UNG/ACORN route、entry-group 是否已计算和 pre-trie heuristic；`SearchRuntimeConfig` 收束 `Lsearch/K/efs/force_use_alg` 等一次 search run 的运行参数；`make_search_runtime_config()` 是 app 入口和历史兼容 wrapper 的唯一 runtime 构造 helper | build-time graph construction、cross-edge route policy |
| `UNG/codes/include/ung_entry_group.h` / `UNG/codes/src/ung_entry_group.cpp` | entry-group 扩展选择策略 `SelectionMode`、provider 请求实现枚举 `EntryGroupProviderImpl`、provider 输入契约 `EntryGroupProviderRequest`、provider 输出契约 `EntryGroupProviderResult`、provider impl/kind 命名 helper 和字符串解析 helper | build-time graph construction、cross-edge route policy |
| `UNG/codes/src/uni_nav_graph_io.cpp` | index save/load、vector-attribute bipartite graph persistence、ACORN inverted-index loading、index statistics | 算法 route policy、kernel 或 benchmark 实验 |
| `UNG/codes/src/uni_nav_graph_vector_attr.cpp` | vector-attribute bipartite graph 构建和比较 helper | index save/load、LNG/trie/coverage、cross-edge 构建 |
| `UNG/codes/src/uni_nav_graph_acorn_augment.cpp` | legacy ACORN/guarantee distance-oriented 补边路径 | 新 cross-edge 主路径、GPU kernel、group graph 构建、query 执行 |
| `UNG/codes/include/ung_acorn_augment_config.h` | legacy ACORN-in-UNG 补边配置结构 `AcornInUng`；默认构造即关闭 legacy augment | 新 cross-edge route 配置、paper fused 参数 |
| `UNG/codes/include/ung_query_stats.h` | query/search benchmark 统计字段契约；所有字段都有安全默认值，未被某条 route 覆盖时输出 0 | build-time graph construction、backend route policy |
| `UNG/codes/src/gpu_gemm_topk.cu` + `gpu_cross_edge_*.cuh` | cross-edge CUDA route setup、fused/topK/SGEMM/cuVS/source/X-streaming/device helper | CPU orchestration 和文档级方法选择 |
| `UNG/codes/include/ung_build_settings.h` / `UNG/codes/src/ung_build_settings.cpp` | build/group graph/output-boundary/additional 的 env fallback 收口 | 新算法主体实现 |
| `UNG/codes/include/ung_cross_edge_config.h` / `UNG/codes/src/ung_cross_edge_config.cpp` | cross-edge backend 分类、GPU runtime route config、writeback mode | kernel 细节或 CPU fallback 实现 |

历史上 `UniNavGraph` public 区域曾暴露一批 `generate_queries_*` / `query_generate` 声明，但仓库内没有对应实现，`build_UNG_index.cpp` 中唯一调用点也是整块注释代码。当前已删除这些假 API；query 生成应走独立工具，例如 `UNG/codes/tools/generate_query_labels.cpp` 或 `UNG/codes/tools/generate_mixed_queries.cpp`。

历史 pair-wise GPU cross-edge 和 GPU additional-edge helper 也只剩 `UniNavGraph` 私有声明、没有实现或调用点；当前已删除这些假扩展入口。新的 cross-edge backend 应接入 `build_cross_edges_generate_gpu_optimized()`、`gpu_cross_groups_search_all_batched()` / `gpu_cross_groups_search_x_streaming()` 这一套显式 runtime config 路径。

`UniNavGraph` 的 public 区域现在只应保留两类入口：

| 类别 | 入口 | 说明 |
|---|---|---|
| primary lifecycle/search API | `build()`、`search()`、`search_hybrid()`、`save()`、`load()` | 正式构建、查询和持久化入口 |
| diagnostic / benchmark helper | `get_min_super_sets_debug()`、`compute_attribute_bitmap()`、`compute_bitmap_from_groups()`、`batch_compute_ung_bitmaps()`、`select_entry_groups()`、`compare_graphs()` | 现有 app/bitmap/entry-cost benchmark 仍直接调用；新 production provider 不应继续扩展这一层 |

以下状态已经是 private 内部数据，不是扩展点：`_num_points`、`_vector_attr_graph`、`_attr_to_id`、`_id_to_attr`、`_num_attributes`、`_lng_descendants_bits`、`_covered_sets_bits`、`_lng_descendants_rb`、`_covered_sets_rb`。新增 backend 如果需要这些数据，应先定义清楚输入数据契约，而不是把内部成员重新公开。

`UniNavGraph` 的内部标量状态应保持显式默认初始化。当前 `_num_points`、`_num_groups`、`_num_attributes`、`_num_cross_edges`、`_max_degree`、`_Lbuild`、`_alpha`、`_num_threads` 和 legacy ACORN edge counter 都有 in-class initializer；新增内部计数器或阶段耗时字段时也应遵守这个规则，避免默认构造、load 前诊断或异常路径读到未初始化值。

legacy ACORN/guarantee 补边路径仍保留为历史实验入口。`AcornInUng{}` 的安全默认语义是 `ung_and_acorn=false`、`new_edge_policy=false`，因此 `build_UNG_index` 不再在 app 层复制这批硬编码默认值。若必须运行 `new_edge_policy=just_acorn` 或 `acorn_and_guarantee`，需要显式打开 legacy augment 并设置：

```bash
export UNG_ACORN_SCRIPT_PATH=/abs/path/to/run_acorn_for_ung.sh
```

这条路径属于 legacy/experimental，不是当前 cross-edge 主贡献；新的 cross-edge 方法应接入 `uni_nav_graph_cross_edges.cpp` 和 `gpu_cross_edge_*.cuh`，不要继续扩展 ACORN augment 文件。


## 1.2 Special Block 方法注册

当前 special block 是 x1/original 数据口径下的候选结构优化，不是默认 full-quality 主 claim。相关入口和默认如下：

| 方法/开关 | 分类 | 当前默认/推荐 | 实现入口 | 语义边界 |
|---|---|---|---|---|
| `UNG_SPECIAL_BLOCKS=1` + `UNG_SPECIAL_BLOCK_DATA_MODE=x1` | diagnostic / candidate | 仅显式打开 | `uni_nav_graph_special_blocks.cpp::build_special_blocks()` | 只用于 x1/original 语义；repeat x40/x200 不应直接构造 special block |
| special CPU intra | baseline | exact-topK `UNG_SPECIAL_INTRA_COMPLETE_NX=2048` + 少量 CPU Vamana | `build_special_edge_overlay()` | corrected CPU baseline；旧全 Vamana `155.92s` 只保留为历史慢口径 |
| special GPU intra | candidate/main for special build | split `th=128` + Tagore/FastGrnnd CUDA | `build_special_edge_overlay()` + `tagore_graph_builder.cu` | 当前 stage 收益明确；仍需端到端 filtered-search recall 约束 |
| special CPU inter | baseline | CPU hybrid exact/Vamana | `build_special_edge_overlay()` | threadfix 后约 `10-11s`，不是旧单线程慢路径 |
| special GPU inter | candidate/main for special build | source-exact cuda-core, `UNG_SPECIAL_GPU_INTER_WARPS=2` | `gpu_gemm_topk.cu::gpu_build_special_inter_edges()` | H2D/D2H 不是瓶颈；WMMA tile 是负结果 |
| binary-only sidecar | boundary/save optimization | `UNG_SPECIAL_EDGE_BINARY_ONLY=1` for experiments | `save_special_blocks()` / `load_special_blocks()` | CSV 保留调试/兼容；binary-only 已支持 auto-load light sidecar |
| two-tier heavy sidecar | approximate candidate | env-gated | `uni_nav_graph_search_backend.cpp` and special block load/save | broad query recall 可恢复，但仍是近似路线，需要更多 workload |

Reviewer 应先读 `docs/reports/REVIEW_READY_HANDOFF_CN.md`、`docs/reports/THREE_MAINLINES_METHOD_BASELINE_DATA_SPEEDUP_CN.md#0-最终性能大表` 和 `docs/reports/SPECIAL_BLOCK_BUILD_QUERY_AB_CN.md#0-reviewer-当前口径速览`，不要从历史负结果反推当前默认。

## 2. 顶层 profile

| Profile | 分类 | 选择方式 | 含义 |
|---|---|---|---|
| `original_cpu` | baseline | `UNG_BUILD_PROFILE=original_cpu` | 尽量还原旧 CPU pipeline |
| `current_cpu` | baseline | `UNG_BUILD_PROFILE=current_cpu` | 当前优化后的 CPU pipeline |
| `naive_gpu` | baseline | `UNG_BUILD_PROFILE=naive_gpu` | 常规 GPU baseline，SGEMM + separate topK |
| `paper_fused` | main | `UNG_BUILD_PROFILE=paper_fused` | grouped fused topK cross-edge 路径 |
| `custom` | diagnostic | `UNG_BUILD_PROFILE=custom` | 逐项开关，用于 ablation 和调参 |

实现入口：

```text
UNG/codes/include/ung_build_config.h
UNG/codes/src/ung_build_config.cpp
```

`UngBuildConfig::from_env(build_threads)` 的结构约定：

```text
1. 读取 UNG_BUILD_PROFILE；
2. 通过 apply_profile_defaults() 应用 profile 默认组合；
3. 只有 custom profile 继续读取逐项 env；
4. 逐项 env 的 int-to-enum 映射必须放在命名 *_from_int() helper 中。
```

新增 profile 或枚举值时，不要在 `from_env()` 主体里继续堆嵌套三目表达式；先扩展 enum / `to_string()` / `*_from_int()` / runbook，再接入 profile defaults 或 custom env。

## 3. Group Graph 方法

| 方法 | 分类 | 选择方式 | 实现入口 | 语义/用途 |
|---|---|---|---|---|
| CPU Vamana | baseline | `UNG_GROUP_GRAPH_IMPL=0` | `UniNavGraph::build_graph_for_all_groups()` in `uni_nav_graph_group_graph.cpp` | 质量 baseline 和 fallback |
| TagoreCuda | ablation | `UNG_GROUP_GRAPH_IMPL=1` | `build_tagore_vamana_cuda_batch(..., TagoreVamana)` | 原始 Tagore 风格 GPU 路径 |
| GrnndLikeCuda | ablation | `UNG_GROUP_GRAPH_IMPL=2` | `build_tagore_vamana_cuda_batch(..., TagoreVamanaWithGrnndRefine)` | 精度增强实验路径 |
| FastGrnndCuda | main component | `UNG_GROUP_GRAPH_IMPL=3` | `build_tagore_vamana_cuda_batch(..., FastGrnnd)` | 大组候选生成和轻量修复 |
| AdaptiveCuda | main/router | `UNG_GROUP_GRAPH_IMPL=4` | `UniNavGraph::build_graph_for_all_groups_tagore_cuda()` in `uni_nav_graph_group_graph.cpp` | small/medium/large workload-aware router |

Tagore/FastGrnnd batch builder 的方法选择只通过 `TagorePruneMode` 枚举表达；旧的 `bool grnnd_like_refine` overload 已删除。新增 group graph backend 不应再用布尔参数区分算法族。`tagore_graph_builder.h` 现在公开 `TagoreCudaRuntimeConfig`：`TagoreFastExactBatchConfig` 表达 packed exact-anchor 的 direct-H2D、device lookup、compact-D2H、warp kernel 等选项，`TagoreFastGrnndPruneConfig` 表达 light/reverse/repair prune 选项，batch-level 字段表达 exact-batch threshold、parallel streams 和 compact-D2H request。`tagore_cuda_config.cpp` 是 `UNG_FAST_*` / `UNG_TAGORE_*` env 到 typed config 的唯一 Tagore CUDA 读取点；`tagore_graph_builder.cu` 只消费 config 并执行 CUDA builder。旧的无 config overload 仍保留为兼容入口，但只是从 env 构造 runtime config 后转发；新的主调用链应显式传入 `TagoreCudaRuntimeConfig`。

当前推荐 router 语义：

```text
small group  -> CPU Vamana 或 bounded-complete fallback
medium group -> packed exact-anchor
large group  -> FastGrnndCuda + reverse-tail / repair
```

Group graph / output-boundary 配置现在先收敛到 `include/ung_build_settings.h` / `src/ung_build_settings.cpp`，而不是散落在 `uni_nav_graph.cpp` 执行循环中：

| settings | 负责范围 | 审计日志 |
|---|---|---|
| `GroupGraphRoute` | `UngGroupGraphImpl` 到 CPU/CUDA backend、adaptive route、Tagore prune mode 的语义映射 | 当前不单独打印 summary，作为主流程 route helper |
| `GraphReserveSettings` | Graph neighbor reserve 的 auto/manual、small-group 阈值、slack、hard cap | `[graph_reserve] config` / `[PROF] graph.reserve_config` |
| `CpuGroupGraphSettings` | CPU Vamana group graph 的 complete/bounded/profile/large-inner-thread 参数 | `[group_graph] cpu_config` / `[PROF] group_graph.cpu_config` |
| `TagoreGroupGraphSettings` | Tagore/FastGrnnd fallback、overlap fallback、exact batch、fill 参数 | `[TagoreCuda] config` / `[PROF] tagore.config` |
| `IntraGroupIdSettings` | 组内建图使用 group-local id 还是 global id 的兼容边界 | 当前主要用于统一入口判断，不单独打印 summary |
| `CpuHybridCrossSettings` | CPU hybrid cross 中 exact scan 的工作量阈值 | 当前主要用于 hybrid cross route，不单独打印 summary |
| `GpuCrossLifecycleSettings` | GPU cross-edge 构建后是否释放 resident vector/cache | 当前主要用于生命周期边界，不单独打印 summary |
| `AdditionalEdgesSettings` | additional_edges 的 direct append 和 hybrid exact scan 阈值 | 当前主要用于 additional_edges route，不单独打印 summary |

这些 settings 不是新算法，只是把历史 env fallback 变成可读、可审计的 route 配置。`uni_nav_graph.cpp` 现在只消费 settings 对象；Tagore/FastGrnnd 路径已经进一步用 `TagoreGroupBuildContext` 把 `GroupGraphRoute`、`TagoreGroupGraphSettings`、`TagoreCudaRuntimeConfig`、`TagorePruneMode` 和 exact-batch threshold 收敛成一次 build 的上下文对象。

Group graph adapter 现在集中在 `UNG/codes/src/uni_nav_graph_group_graph.cpp`。`build_graph_for_all_groups()` 是 CPU/GPU group graph 总入口：负责检查 index name、读取 `GroupGraphRoute` / `CpuGroupGraphSettings`、选择 CPU Vamana baseline 或 CUDA route。CPU 路径中的 complete、bounded-complete、小组 fallback、大组内层多线程 Vamana 和 profile bin 输出都在同一文件内。`build_graph_for_all_groups_tagore_cuda()` 负责 CUDA high-level orchestration：创建 `TagoreGroupBuildContext`、启动可选 fallback overlap、调用 GPU batch builder、汇总 timing 并写回 `_tagore_*` 指标。Group partition 已抽到 `UniNavGraph::partition_tagore_groups()`，并通过 `TagoreGroupPartition` 返回 fallback group ids、GPU group ids、Tagore requests 和 GPU path 总点数。Fallback group build 已抽到 `UniNavGraph::build_tagore_fallback_groups()`，并通过 `TagoreFallbackBuildStats` 汇总 complete/CPU fallback 组数、点数和 wall time。Exact/GNN batch split 已抽到 `UniNavGraph::build_tagore_batch_artifacts()`，并通过 `TagoreBatchBuildArtifacts` 返回 batch results、packed graph source、batch timing 和 wall time；这些 helper 现在接收同一个 `TagoreGroupBuildContext`，因此 fallback、exact/GNN split、CUDA fast-exact 判断和 fill/writeback 使用同一组阈值与 runtime config。上述中间类型已从 `UniNavGraph` 嵌套声明移到 `include/ung_group_graph_build_types.h`，为后续正式 `GroupGraphBackend` 接口留出独立数据契约。Batch result 的 CPU-compatible fill/writeback 已抽到 `UniNavGraph::fill_tagore_batch_results()`，负责 packed graph source 选择、Graph::neighbors 填充、entry point 写回和 Tagore timing 累加。下一步如果要把 group graph 做成完整插件式 backend，应把当前 `TagoreGroupBuildContext` 提升为正式 `GroupGraphBackendConfig`，并让 backend 返回统一 result/writeback contract。

关键参数：

| 参数 | 含义 |
|---|---|
| `UNG_GROUP_GRAPH_COMPLETE_NX` | 小组 complete/bounded fallback 阈值 |
| `UNG_TAGORE_MIN_GROUP_SIZE` | 小于该值不进入 GPU group graph backend |
| `UNG_FAST_GRNND_BATCH_EXACT_NX` | FastGrnnd 路径中走 packed exact 的最大 nx |
| `UNG_ADAPTIVE_EXACT_MAX_NX` | adaptive router 的 exact-anchor 上界 |
| `UNG_TAGORE_ITER` | GNN/Tagore 迭代次数；GRNND-style 默认 4，Tagore 默认 10 |
| `UNG_TAGORE_M` | prune / candidate 保留相关参数 |

## 4. Cross-Edge 方法

| 方法 | 分类 | 选择方式 | 实现入口 | 语义/用途 |
|---|---|---|---|---|
| CPU Vamana cross | baseline | `UNG_CROSS_EDGE_IMPL=0` | `build_cross_edges_generate_cpu_baseline()` in `uni_nav_graph_cross_edges.cpp` | 原始弱 baseline |
| GPU batched | main/router | `UNG_CROSS_EDGE_IMPL=1` | `build_cross_edges_generate_gpu_optimized()` in `uni_nav_graph_cross_edges.cpp` | GPU cross-edge 总入口 |
| Original CPU | baseline | `UNG_CROSS_EDGE_IMPL=2` | `build_cross_group_edges_original_cpu()` in `uni_nav_graph_cross_edges.cpp` | 旧 CPU 函数 |
| CPU exact scan | baseline/diagnostic | `UNG_CROSS_EDGE_IMPL=3` | `build_cross_edges_generate_cpu_exact_scan()` in `uni_nav_graph_cross_edges.cpp` | 强 CPU brute-force baseline |
| CPU hybrid scan/Vamana | diagnostic | `UNG_CROSS_EDGE_IMPL=4` | `build_cross_edges_generate_cpu_hybrid_scan_vamana()` in `uni_nav_graph_cross_edges.cpp` | 小工作量 exact，大工作量 Vamana |
| cuVS per-group | baseline/diagnostic | `UNG_CROSS_EDGE_IMPL=5` | `build_cross_edges_generate_cuvs_bruteforce()` | 库调用 baseline |

GPU batched 内部子路径：

| 子路径 | 分类 | 关键开关 | 实现入口 | 说明 |
|---|---|---|---|---|
| SGEMM + topK | baseline | `UNG_GPU_TOPK_IMPL=2` | route: `gpu_cross_groups_search_all_batched()`；tile 执行: `gpu_cross_edge_sgemm_baseline.cuh` | 常规 GPU baseline |
| grouped fused topK | main | `UNG_GPU_TOPK_IMPL=3` | `gpu_cross_groups_search_all_batched()` | 当前 cross-edge 核心贡献 |
| universal flat double-buffer | main/engineering | `UNG_UNIVERSAL_GPU=1` | `gpu_cross_groups_search_all_batched()` | 当前工程主路线 |
| source-centric exact | negative/experimental | `UNG_GPU_SOURCE_EXACT=1` | `build_cross_edges_generate_gpu_source_exact()` | 当前 single-stage 版本不是主结果 |
| X-side streaming | boundary | `UNG_GPU_X_STREAMING=1` | `gpu_cross_groups_search_x_streaming()` | 解决 resident all-X OOM，但 v1 被 `pack_q` 主导 |

CPU/GPU cross-edge backend 分类现在由 `CrossEdgeBackend` 表达，定义在 `ung_cross_edge_config.h`，不再作为 `UniNavGraph` 的嵌套 enum。GPU batched 的主配置边界由 `CrossEdgeGpuRuntimeConfig` 表达，入口在：

```text
UNG/codes/include/ung_cross_edge_config.h
UNG/codes/src/ung_cross_edge_config.cpp
```

CPU/GPU cross-edge 的 C++ orchestration 现在集中在：

```text
UNG/codes/src/uni_nav_graph_cross_edges.cpp
UNG/codes/include/uni_nav_graph.h: Cross-group edge orchestration / CUDA cross-edge backend entry points
```

该文件承载 `build_cross_group_edges()`、backend resolve、CPU Vamana/exact/hybrid baseline、GPU route adapter 和 original CPU path。`build_cross_group_edges()` 现在只保留 route、fallback、offset、helper 调用和 summary 汇总；additional_edges 构建已进入 `build_cross_edges_generate_additional()`，cross-edge 写回已进入 `merge_cross_edges_to_graph()`，additional_edges materialized 写回已进入 `merge_additional_edges_to_graph()`。host 输出消费已通过 `CrossEdgeHostOutputView` 统一表达 SearchQueue / id-vector / flat-id，additional coverage 检查和 graph materialization 不再散传三种容器和 mode；`CrossEdgeGraphMaterializationWriter` 已提升到 `ung_cross_edge_output_writer.{h,cpp}`，集中负责把这些输出追加到 `_graph->neighbors`，并可生成 CSR adjacency buffers。它不包含 CUDA kernels；CUDA 执行仍由 `gpu_gemm_topk.cu` 及 `gpu_cross_edge_*.cuh` helper 承担。search 侧可通过 `SearchRuntimeConfig::graph_backend` / `search_UNG_index --graph_search_backend csr` 显式选择 CSR adjacency backend。

CUDA 侧多个 cross-edge route 共用的 descriptor 放在：

```text
UNG/codes/src/gpu_cross_edge_common.cuh
UNG/codes/src/gpu_cross_edge_planning.cuh
UNG/codes/src/gpu_cross_edge_buffers.cuh
UNG/codes/src/gpu_cross_edge_topk_device.cuh
UNG/codes/src/gpu_cross_edge_update_kernels.cuh
UNG/codes/src/gpu_cross_edge_resident_vectors.cuh
UNG/codes/src/gpu_cross_edge_cpu_tiny.cuh
UNG/codes/src/gpu_cross_edge_query_pack.cuh
UNG/codes/src/gpu_cross_edge_bucket_descriptors.cuh
UNG/codes/src/gpu_cross_edge_group_dispatch.cuh
UNG/codes/src/gpu_cross_edge_regular_route.cuh
UNG/codes/src/gpu_cross_edge_singleton_kernels.cuh
UNG/codes/src/gpu_cross_edge_large_group_kernels.cuh
UNG/codes/src/gpu_cross_edge_descriptor_kernels.cuh
UNG/codes/src/gpu_cross_edge_group_fused_kernels.cuh
UNG/codes/src/gpu_cross_edge_double_buffer.cuh
UNG/codes/src/gpu_cross_edge_sgemm_baseline.cuh
UNG/codes/src/gpu_cross_edge_bucket_fused.cuh
UNG/codes/src/gpu_cross_edge_per_group_fused.cuh
UNG/codes/src/gpu_cross_edge_output.cuh
UNG/codes/src/gpu_cross_edge_profile.cuh
UNG/codes/src/gpu_cross_edge_verify.cuh
UNG/codes/src/gpu_cross_edge_x_streaming.cuh
UNG/codes/src/gpu_cross_edge_source_exact_kernels.cuh
UNG/codes/src/gpu_cross_edge_source_exact.cuh
UNG/codes/src/gpu_cross_edge_cuvs_baseline.cuh
```

`gpu_cross_edge_common.cuh` 目前只放 POD descriptor，不放 route policy 和 env parsing，避免把公共结构和调度逻辑再次耦合。

`gpu_cross_edge_planning.cuh` 放 route-neutral planning helper：`CrossEdgeTargetWorkload`、`CrossEdgeBatchCapacity`、`plan_cross_edge_target_workload()` 和 `make_cross_edge_batch_capacity()`。它统一 LNG in-neighbor query count、target nx、bench filter、CPU tiny fallback 和 flat batch split 语义，供不同 cross-edge backend 复用。

`gpu_cross_edge_buffers.cuh` 放 GPU cross-edge 复用缓存、lazy allocation helper 和 cuBLAS/cuBLASLt handle。它被包含在 `gpu_gemm_topk.cu` 的匿名 namespace 内，因此保留原来的 internal-linkage 全局状态和懒初始化语义；它不是 route policy，也不决定算法，只负责 Q/output、resident all-X、norm、dot、descriptor、global-merge、singleton、cuVS result 和 cuBLAS workspace 这些 buffer 的容量管理。

`gpu_cross_edge_topk_device.cuh` 放 device-side topK 插入 primitive：`ung_topk_insert32()`、`ung_topk_insert16()` 和 `ung_global_topk_insert_locked()`。它必须出现在所有调用这些 primitive 的 kernel include/definition 之前，包括 source-exact、singleton、large/small/medium fused 和 group-desc fused kernels。

`gpu_cross_edge_update_kernels.cuh` 放多个 cross-edge route 共用的基础 CUDA 更新操作：`l2_norm_sq_kernel()`、`init_topk_kernel()`、`topk_insert_linear()` 和 SGEMM/cuBLAS baseline 使用的 `update_topk_from_dot_*` kernels。它不是 route policy，也不是新算法；它被包含在 `gpu_gemm_topk.cu` 的匿名 namespace 内，并且必须早于 double-buffer 和 SGEMM baseline helper。

`gpu_cross_edge_resident_vectors.cuh` 放 resident all-X upload/release member functions：`gpu_prepare_all_vectors_on_device()`、`gpu_prepare_all_vectors_for_cross_edge()` 和 `gpu_release_all_vectors_on_device()`。它负责连续内存探测、direct pageable / `cudaHostRegister` / pinned staging 三种上传路径、device all-X cache、all-X norm cache 和共享 buffer 释放。上传策略由 `CrossEdgeGpuRuntimeConfig::vector_upload` 显式传入，`UNG_GPU_PREPARE_DIRECT_HOSTREG` / `UNG_GPU_PREPARE_DIRECT_PAGEABLE` / `UNG_GPU_PREPARE_HOSTREG_MAX_MB` 只在 `ung_cross_edge_config.cpp` 读取；该 helper 不选择 cross-edge 算法，只提供 source-exact、cuVS baseline、batched fused 和 double-buffer route 复用的 resident vector cache。

`gpu_cross_edge_cpu_tiny.cuh` 放 CPU tiny exact fallback 执行 helper：`run_cross_edge_cpu_tiny_groups()`。`gpu_cross_edge_planning.cuh` 决定哪些 groups 进入 tiny fallback；这个文件只执行这些 groups 的 CPU exact scan 和 host topK，并写回 `SearchQueue`。它不是新的算法变种，而是把主 all-batched loop 里的小工作量 fallback 细节显式隔离。

`gpu_cross_edge_query_pack.cuh` 放 regular all-batched route 的 query packing/upload helper：`CrossEdgeQueryUploadResult` 和 `pack_and_upload_cross_edge_queries()`。它负责 flatten Q、生成 host writeback maps、按 route 上传 host Q 或 qid、可选启动 device gather，并返回 `fill_ms/h2d_ms`。它不是 search backend，只是把 CPU packing 和 H2D boundary 从主 route 函数里隔离出来。

`gpu_cross_edge_bucket_descriptors.cuh` 放 bucket fused / TF32 路径的 host-side descriptor assembly：`CrossEdgeBucketDescriptors` 和 `build_cross_edge_bucket_descriptors()`。它只负责给 target groups 标记 small/medium/TF32 bucket，并构造 kernel 需要的 POD descriptors；kernel launch 仍在 `gpu_cross_edge_bucket_fused.cuh`。

`gpu_cross_edge_group_dispatch.cuh` 放 regular all-batched 的 per-group execution helper：`CrossEdgeRegularWorkloadView`、`CrossEdgeRegularDispatchConfig`、`CrossEdgeRegularDeviceBuffers`、`CrossEdgeRegularDispatchCounters` 和 `dispatch_cross_edge_regular_groups()`。它逐 target group 执行 heavy-SGEMM fallback 判断、immediate fused route 尝试，以及 separate-topK baseline fallback。主函数保留 route setup、global merge、output 和 profile 汇总，但不再通过几十个独立参数把 workload、route 参数、device buffer 和 profile counter 穿透到 dispatch 层。

`gpu_cross_edge_regular_route.cuh` 放 regular all-batched 的 host-side route/config assembly：`CrossEdgeRegularRouteConfig`、`make_cross_edge_regular_route_config()`、`compute_cross_edge_direct_qid_all_effective()`、`make_cross_edge_separate_topk_config()`、`make_cross_edge_regular_dispatch_config()` 和 `log_cross_edge_regular_route_config()`。它把 query gather、singleton、direct-qid fused、global merge/writeback、GEMM/cuBLAS/naive CUDA backend、small/medium/large fused 阈值和 TF32 group path 集中到一个地方，并显式消费 `CrossEdgeGpuRuntimeConfig&`；主 `.cu` 文件现在只消费结构化 route config，不再直接维护或 fallback 读取这批 env。

`gpu_cross_edge_singleton_kernels.cuh` 放 singleton target fastpath kernels：`ung_singleton_top1_global_kernel()` 和 `ung_singleton_top1_global_merge_kernel()`。它只处理 `nx == 1` 的特殊路径，和 small/medium/large fused kernel 分开维护，避免主 `.cu` 文件里把特殊 fastpath 与通用 fused route 混在一起。

`gpu_cross_edge_large_group_kernels.cuh` 放 immediate per-group route 使用的 large target-group fused kernels，包括 warp-fused、warp-query 和 TF32 WMMA 三种 large 模式及其 global-merge 变体。它只放 kernel definitions；是否启用、阈值和 launch 参数仍由 `gpu_cross_edge_per_group_fused.cuh` 与 `CrossEdgeGpuRuntimeConfig` 决定。

`gpu_cross_edge_descriptor_kernels.cuh` 放 descriptor-batched TF32 kernels：`ung_group_tile_tf32_wmma_topk_global_kernel()`、`ung_group_prefix_tf32_wmma_topk_global_kernel()` 和 `ung_group_2d_tf32_wmma_topk_global_kernel()`。这些 kernel 被 bucket fused、double-buffer 和 X-streaming route 复用；descriptor 构造、bucket 选择和 launch 策略不在这个文件里，仍分别由 planning/route helper 管理。

`gpu_cross_edge_group_fused_kernels.cuh` 放 naive group fused、small fused、medium fused 和 descriptor small/medium fused kernels，包括 global-merge 变体。它只放 kernel definitions；per-group immediate route、bucket route、double-buffer route 和 X-streaming route 分别决定何时 launch 这些 kernel。

`gpu_cross_edge_double_buffer.cuh` 是 double-buffer 主路径的 helper 文件，不是新的公开 API。它被 `gpu_gemm_topk.cu` 在 kernel/global-buffer 定义之后包含，集中放置 DB route/eligibility、slot 资源、chunk packing、enqueue、global merge finalize 和双 slot scheduler。这样主执行函数只保留 route 决策和输出汇总，不再内嵌大段 stream/event/buffer 管理。

`gpu_cross_edge_sgemm_baseline.cuh` 是 SGEMM/cuBLASLt/naive-dot + separate-topK baseline helper。它放置 `CrossEdgeSeparateTopkConfig` 和 `run_cross_edge_separate_topk_group()`，负责 dot tile 规划、cuBLAS SGEMM strided batch、cuBLASLt TF32 matmul、naive CUDA dot、tail GEMV/GEMM 和独立 topK update kernel。它不是 paper fused 路径，也不新增公开 API；它的作用是让主 group loop 只表达 route/fallback，而不是内嵌 baseline 的 300 多行 tile 执行细节。

`gpu_cross_edge_bucket_fused.cuh` 是 bucket fused descriptor launch helper。它放置 `launch_cross_edge_bucket_fused_groups()`，消费已经构造好的 small / medium / TF32 descriptors，负责 H2D descriptor copy、kernel launch 和 launch error check。descriptor 构造和 routing 仍在 `gpu_cross_groups_search_all_batched()`，避免把执行细节和 workload 决策重新耦合。

`gpu_cross_edge_per_group_fused.cuh` 是 immediate per-group fused launch helper。它放置 `try_launch_cross_edge_per_group_fused()`，集中处理不能走 descriptor batching、需要逐 target group 立即 launch 的 naive / small / medium / large fused kernel route，并维护对应 profile counter。主 group loop 只需要先做 workload/routing 判断，然后调用这个 helper；如果 helper 返回 `false`，再落到 SGEMM/cuBLASLt/naive-dot + separate-topK baseline。

`gpu_cross_edge_output.cuh` 是普通 batched 路径的 output finalization helper。CUDA host 输出契约 `CrossEdgeTopkOutputWriter` 放在 `gpu_cross_edge_common.cuh`，regular output helper 和 double-buffer global-merge finalize 共用它表达 flat-id / id-vector / SearchQueue 三种 writeback。`gpu_cross_edge_output.cuh` 放置 `CrossEdgeTopkOutputFinalizeResult` 和 `finalize_cross_edge_topk_output()`，负责 D2H、host writeback、`gpu_d2h_ms/writeback_ms` timing 和可选 `valid_topk_pairs` 统计。double-buffer 路径不走这里，仍由 `gpu_cross_edge_double_buffer.cuh` 的 finalize helper 收尾，但 global-merge finalize 也消费同一个 writer contract。

`gpu_cross_edge_profile.cuh` 是普通 batched 路径的 profiling summary helper。它放置 `CrossEdgeRegularProfileSummary` 和 `log_cross_edge_regular_profile_summary()`，集中维护尾部 profile key，避免主函数继续内嵌一长串日志格式。

`gpu_cross_edge_verify.cuh` 是 optional diagnostic helper。它放置 `verify_cross_edge_topk_output()`，只在 `CrossEdgeGpuRuntimeConfig::diagnostics.verify_samples>0` 时用 CPU exact scan 抽样校验 GPU topK 输出，并保留 `verify_strict` 的失败语义。`UNG_GEMM_VERIFY_SAMPLES` / `UNG_GEMM_VERIFY_STRICT` 只在 `ung_cross_edge_config.cpp` 解析；verify helper 本身不是性能路径，也不应作为论文默认实验配置。

`gpu_cross_edge_x_streaming.cuh` 是 X-side streaming boundary helper 文件，只放 `gpu_cross_groups_search_x_streaming()`。这个路径用于 resident all-X device cache 放不下时按 X chunk 流式处理 target vectors；它是 scalability/OOM 兜底，不是默认性能路线。

`gpu_cross_edge_source_exact_kernels.cuh` 和 `gpu_cross_edge_source_exact.cuh` 是 source-centric exact experimental path。前者放 source-exact CUDA kernels，后者放 `build_cross_edges_generate_gpu_source_exact()` host route。这条路径是 checked negative ablation / future two-stage source-grouped design anchor，不进入默认主路径。

`gpu_cross_edge_cuvs_baseline.cuh` 是 cuVS per-group library baseline，只放 `build_cross_edges_generate_cuvs_bruteforce()`。它固定 SearchQueue 写回，不参与 flat-id / id-vector output-boundary 优化。

cross-edge 构建结果类型放在：

```text
UNG/codes/include/ung_cross_edge_result.h
```

`CrossEdgeBuildResult` 目前统一表达 generate / additional / add-offset / merge / GPU H2D-kernel-D2H timing、active writeback、flat-id item 数、GPU fallback 和 additional direct append 状态。它已经是正式 result 类型；GPU optimized、cuVS baseline 和 source-exact backend 私有函数已接收 `CrossEdgeBuildTiming&`，CUDA cross search helper 已接收 `CrossEdgeBuildTiming*`，不再散传 `h2d/kernel/d2h` 三个指针。全量向量上传通过 `gpu_prepare_all_vectors_for_cross_edge(CrossEdgeBuildTiming&, const CrossEdgeGpuRuntimeConfig&)` 累计 H2D timing，并从同一份 runtime config 读取 host-register/direct-pageable 策略；裸 `gpu_prepare_all_vectors_on_device(const CrossEdgeVectorUploadConfig&, double*)` 只作为底层 copy helper 保留。`CrossEdgeHostOutputView` 已经把 CPU-side graph materialization 的输入边界收成一个 view，`CrossEdgeGraphMaterializationWriter` 已经提升到独立 output writer 模块并把 `_graph->neighbors` 写回集中到一处，同时能生成 `CrossEdgeCsrOutput` adjacency buffers；search 侧已有 `GraphSearchBackend` / `GraphNeighborView` 邻接访问边界，并支持从 `_graph` 或 `CrossEdgeCsrOutput` 读取邻接，`search_UNG_index --graph_search_backend csr` 可显式走 CSR-backed search。

`gpu_cross_groups_search_all_batched()` 的 target workload 统计已拆成 `plan_cross_edge_target_workload()`，目前位于 `gpu_cross_edge_planning.cuh`。该 helper 统一表达 raw query count、GPU query count、target nx、CPU tiny fallback groups、bench-filter skip count、total GPU queries 和 count-stage timing。GPU 执行函数只消费 planning 结果，因此后续接入新的 cross-edge backend 时可以复用同一个 LNG in-neighbor workload 语义。

flat batch 的容量边界也已经进入同一套 route config：`flat_q_cap_mb` 和 `flat_out_cap_mb` 来自 `UNG_GPU_FLAT_Q_CAP_MB` / `UNG_GPU_FLAT_OUT_CAP_MB`，CUDA 侧由 `CrossEdgeBatchCapacity` 计算 `max_flat_queries` 并决定是否 split。这样 batch-size routing 不再直接散落在执行函数里。

double-buffer 的 route 设置也已集中到 `CrossEdgeDoubleBufferRoute`，包括 request、DB no-split、device-gather、unsupported split、global merge、chunk size、medium/large nx 阈值和 warp 数。`evaluate_cross_edge_double_buffer_eligibility()` 单独判断当前 batch 是否能进入 double-buffer，以及 unsupported group 数量。这样新的 backend 或调参实验可以复用同一套 eligibility 语义，而不是在执行函数里重新拼布尔条件。这组 helper 目前位于 `gpu_cross_edge_double_buffer.cuh`。

double-buffer 的 slot 生命周期已集中到 `CrossEdgeDoubleBufferSlot` 和 `init/free/ensure_*` helper。slot 负责 stream/event、pinned host buffer、device buffer、group/tile descriptor buffer 和 query id 映射；`finish_cross_edge_double_buffer_slot()` 负责 completion timing 和非 global-merge SearchQueue 写回，主执行函数不再内嵌这组资源管理/finish lambda。

double-buffer 的 chunk packing 已集中到 `pack_cross_edge_double_buffer_chunk()`。它返回 `CrossEdgeDoubleBufferChunkPack`，包含 query count、group count、group descriptors 和 tile descriptors，并负责填充 slot 的 qid / target offset / query-global-id 映射。`enqueue_cross_edge_double_buffer_chunk()` 消费 pack 结果执行 H2D、topK 初始化、medium/large kernel launch、可选 global merge、D2H 和 done event。

global-merge 的最终输出也已集中到 `finalize_cross_edge_double_buffer_global_merge()`。它负责从 `g_d_global_idx/g_d_global_dis` D2H，并通过 `CrossEdgeTopkOutputWriter` 按输出模式写回 flat-id、id-vector 或 SearchQueue，同时累计 D2H/writeback timing。

double-buffer 的 chunk 调度已集中到 `run_cross_edge_double_buffer_chunks()`。它按 `chunk_queries` 切分 target groups，轮转两个 slots，并保持原有 `finish_slot(si)` 先于 `enqueue_chunk(si, ...)` 的顺序。

它的职责不是新增一种算法，而是把原先散落在 `build_cross_group_edges()` 和 `gpu_gemm_topk.cu` 深处的 legacy env 转成一次性 runtime route。当前已经收敛的参数包括：

`CrossEdgeGpuRuntimeConfig` 同时负责 route 语义命名和基础校验：

| helper | 含义 |
|---|---|
| `route_name()` | 输出 `source_exact`、`x_streaming`、`universal_batched` 或 `batched` |
| `route_class_name()` | 标注 `negative_experimental`、`boundary_scalability`、`main_engineering` 等用途边界 |
| `writeback_mode_name()` | 输出 `search_queue`、`id_vector` 或 `flat_id` |
| `validation_error()` | 集中校验 X-streaming/SearchQueue 等 route 约束，避免主流程散落手写判断 |

| 配置组 | 字段/开关 | 说明 |
|---|---|---|
| route | `source_exact`、`universal_route`、`x_streaming` | 选择 source-centric、universal flat 或 X-streaming boundary 路径 |
| source-centric exact | `source_exact_mode`、`source_exact_warps`、`source_exact_pad_groups` | 控制 source-centric negative/experimental 路径的 CUDA-core/TF32 WMMA、warp 数和 source-group tile padding |
| double-buffer | `db_nosplit`、`double_buffer`、`db_chunk_queries`、`db_split_unsupported` | 控制 direct-qid 双缓冲批处理是否启用、chunk 粒度以及 unsupported group 的拆分回退 |
| X-streaming boundary | `x_stream_x_chunk_mb`、`x_stream_q_chunk_mb`、`x_stream_out_chunk_mb` | 控制 resident all-X OOM 兜底路径的 X/Q/output 分块；这是 scalability/boundary 配置，不是性能主张 |
| resident vector upload | `vector_upload.direct_hostreg`、`vector_upload.direct_pageable`、`vector_upload.hostreg_max_mb` | 控制 prepare_all 阶段使用 `cudaHostRegister`、direct pageable copy 还是 pinned staging；source-exact、cuVS 和 batched fused 共用这份配置 |
| diagnostics | `diagnostics.count_valid_pairs`、`diagnostics.verify_samples`、`diagnostics.verify_strict` | 控制 optional valid-topK 计数和 CPU exact 抽样校验；只用于 profiling/debug，不应进入正常性能路径 |
| query upload | `query_upload_device_gather`、`gather_threads`、`gather_blocks` | 控制 Q 侧是 host pack 还是 qid + device gather |
| tiny/singleton | `cpu_tiny_groups`、`cpu_tiny_*`、`singleton_fastpath`、`singleton_threads` | 控制极小 workload 是否 CPU fallback，以及 nx=1 专用路径 |
| direct-qid fused | `direct_qid_fused`、`direct_qid_all_fused` | 控制 fused kernel 是否直接消费 qid / target offset，减少 materialized Q |
| GEMM fallback | `force_custom_kernel`、`gemm_impl_request`、`naive_heavy_*` | 控制自研 fused / naive CUDA / cuBLAS strided-batched / cuBLASLt 的选择和重负载回退 |
| naive CUDA tuning | `naive_warps_per_block`、`naive_shared_x`、`naive_vec4`、`naive_x_cols_per_block`、`naive_shared_kb`、`topk_block_threads` | 只影响 naive CUDA / separate topK micro-kernel 的调参，不应单独作为论文方法 |
| kernel shape | `small/medium/large_group_fused`、`*_max_nx`、`*_warps`、`large_group_fused_mode`、`db_medium/large_*` | 控制 small fused、medium fused 和 large tiled/WMMA kernel 的分界与并行粒度 |
| global merge | `global_merge`、`global_merge_direct`、`global_merge_direct_max_nx`、`db_global_merge` | 控制同一 qid 多 target group 结果是否在 GPU 侧合并 |
| writeback | `id_vector_writeback`、`flat_id_writeback`、`skip_searchqueue_storage` | 控制输出从 `SearchQueue` 降到 id-vector / flat-id 的程度 |

注意：`cuVS per-group` 是库调用 baseline，当前实现固定写回 `SearchQueue`。即使全局 route 开启 id-vector / flat-id，cuVS baseline 也不参与这些 output-boundary 优化，避免把 baseline 和论文主路径混在一起。

`UNG_BENCH_*` 这种只服务 micro-benchmark 的过滤器已经进入 `CrossEdgeGpuRuntimeConfig::benchmark_filter`，读取点在 `ung_cross_edge_config.cpp`。主 cross-edge GPU 入口、planning helper、double-buffer route helper 和 regular-route helper 都已经要求显式传入 `CrossEdgeGpuRuntimeConfig&`，正常 build 路径不会触发“未传 route 直接读 env”的行为。后续剩余清理重点不再是 env 读取散落，而是把过宽的 `CrossEdgeGpuRuntimeConfig` 拆成 main/baseline/diagnostic/boundary 子配置，或提升为正式 backend adapter。

关键写回方式：

| 写回方式 | 开关 | 含义 |
|---|---|---|
| `SearchQueue` | 默认 | CPU-compatible，最稳但重 |
| id-vector | `UNG_GPU_ID_VECTOR_WRITEBACK=1` | 降低部分 SearchQueue 开销 |
| flat-id | `UNG_GPU_FLAT_ID_WRITEBACK=1` | 更接近 GPU-friendly 输出 |
| direct append additional | `UNG_ADDITIONAL_DIRECT_APPEND=1` | 只支持 CPU exact additional_edges |

## 5. Additional Edges 方法

| 方法 | 分类 | 选择方式 | 实现位置 | 说明 |
|---|---|---|---|---|
| CPU Vamana | baseline | `UNG_ADDITIONAL_EDGES_IMPL=0` | `build_cross_edges_generate_additional()` in `uni_nav_graph_cross_edges.cpp` | full-quality 传统补边路径 |
| Skip | diagnostic | `UNG_ADDITIONAL_EDGES_IMPL=1` | `build_cross_edges_generate_additional()` in `uni_nav_graph_cross_edges.cpp` | 用于拆分 cross-edge 主体耗时，不应作为 full-quality 主结果 |
| CPU exact scan | diagnostic/main candidate | `UNG_ADDITIONAL_EDGES_IMPL=2` | `build_cross_edges_generate_additional()` in `uni_nav_graph_cross_edges.cpp` | 语义更直接，但仍是 CPU 路径 |

当前边界：

```text
cross-edge 主体已经能 GPU 化；
additional_edges 仍经常成为 full-quality 下的新瓶颈。
```

## 6. Search / Query 执行路径

正式查询路径现在集中在：

```text
UNG/codes/src/uni_nav_graph_search.cpp: thread orchestration and result writeback
UNG/codes/src/uni_nav_graph_query_features.cpp: selector features and diagnostics
UNG/codes/src/uni_nav_graph_query_route.cpp: query route policy
UNG/codes/src/uni_nav_graph_entry_provider.cpp: entry-group provider dispatch
UNG/codes/src/uni_nav_graph_search_backend.cpp: ACORN/UNG graph-search backend
UNG/codes/include/ung_query_route.h: QueryRouteDecision
UNG/codes/include/ung_entry_group.h: SelectionMode and provider request/result
UNG/codes/include/uni_nav_graph.h: grouped private declarations for the five layers above
```

这些文件共同承载 production `search_UNG_index` 会调用的 C++ 查询逻辑，不等同于下面的 query-entry benchmark。当前边界如下：

`search_UNG_index` 的 selector model 参数推荐使用 `--selector_model_prefix`。旧拼写 `--selector_modle_prefix` 只作为 deprecated CLI alias 保留，用于兼容历史脚本；C++ API 和当前 benchmark 脚本应使用正确拼写。

| 模块 | 分类 | 入口 | 说明 |
|---|---|---|---|
| query feature collection | diagnostic | `calculate_query_features_only()` in `uni_nav_graph_query_features.cpp` | 采集 Idea1/Idea2 selector 特征，写 CSV；active feature order 是训练模型契约 |
| selector warmup | engineering | `warmup_selectors()` in `uni_nav_graph_query_features.cpp` | 预热 ONNX selector，避免首批 query 抖动 |
| UNG/ACORN hybrid route | main/legacy hybrid | `search_hybrid(..., const SearchRuntimeConfig&)` / `thread_function(query_id, ...)` / `SearchRuntimeConfig` / `make_search_runtime_config()` | `search_UNG_index` 主路径和 worker 都消费 runtime config；旧 `search_hybrid` 长签名仅保留为兼容包装，并调用同一个 runtime 构造 helper；thread-pool task 直接捕获 query id，不再通过共享 queue/mutex 分发；每个 query 内先做 route decision，再调用 ACORN 或 UNG 执行 helper |
| query route selector | main/legacy hybrid | `decide_query_route()` in `uni_nav_graph_query_route.cpp` / `QueryRouteDecision` in `ung_query_route.h` | 集中处理 query feature、Idea1/Idea2 selector、pre-trie heuristic 和 `force_use_alg`；feature 计算、pre-trie heuristic 和 Idea2 entry-group route stats 已移动到 `uni_nav_graph_query_features.cpp`；pre-trie route 应用由 `QueryRouteDecision::apply_pre_trie_heuristic()` 表达，强制路由覆盖由 `QueryRouteDecision::apply_force_route()` 表达，避免 search 实现层散落裸 switch、直接改 `algorithm/use_new_trie` 或复制 descendants/coverage 统计 |
| entry group provider | main | `run_entry_group_provider()` / `prepare_entry_groups_for_execution()` in `uni_nav_graph_entry_provider.cpp` / `EntryGroupProviderImpl` / `EntryGroupProviderRequest` / `EntryGroupProviderResult` | 当前默认 CPU min-super-set provider；选择 `gpu_cover_frontier` 时注入 production CUDA correct-cover callback；`run_entry_group_provider()` 只做 request validation、provider dispatch 并返回 result，`prepare_entry_groups_for_execution()` 只把 result 应用到当前 query；`EntryGroupProviderImpl` 表达请求哪个实现，`EntryGroupProviderKind` 表达实际结果来源；request 汇集 query labels、query id、route decision、当前 entry groups、true group ids 和扩展策略，并提供 `validate()`、`has_current_group_ids()`、`true_group_id_or()`；result 汇集 requested impl、实际 provider、输出 group ids、coverage/minimal 语义、fallback 状态和 provider 耗时，并用 `mark_fallback()` 统一记录 fallback；底层 `get_min_super_sets_debug()` 是 const/read-only 查询接口 |
| entry group expansion strategy | helper | `SelectionMode` in `ung_entry_group.h` / `select_entry_groups()` | 控制额外入口组选择使用 size-only 还是 size+LNG-distance |
| ACORN query execution | main/legacy hybrid | `execute_acorn_query()` in `uni_nav_graph_search_backend.cpp` | 选择 ACORN index、设置 efs、执行 old-bitmap 或 filter-map search，并把结果写入 `SearchQueue` |
| UNG query execution | main | `execute_ung_query()` in `uni_nav_graph_search_backend.cpp` | 将 entry groups 展开为 entry points，再调用 fixed-point graph search |
| entry point expansion | main | `get_entry_points()` / `get_entry_points_given_group_id()` in `uni_nav_graph_search_backend.cpp` | 从入口 group 选取向量入口点 |
| UNG graph expansion | main | `iterate_to_fixed_point()` in `uni_nav_graph_search_backend.cpp` | 在 `_graph` 上执行 greedy fixed-point search |
| global graph persistence | legacy compatibility | `_global_graph` / `_global_vamana_entry_point` in `uni_nav_graph_io.cpp` | 当前 build/search 不构建或查询 global Vamana；只保留空图和 entry point 字段的 save/load，用于旧索引格式兼容 |

当前重要限制：

```text
Search/query 已经从 uni_nav_graph.cpp 拆出，ACORN/UNG 执行阶段、query route selector 和 entry group provider 都已经有 helper 边界。entry group provider 的 UniNavGraph-bound 实现现在集中在 `uni_nav_graph_entry_provider.cpp`，search 执行文件只调用 `prepare_entry_groups_for_execution()`。当前代码已经有 `SearchEntryProvider`、`EntryGroupProviderImpl`、`EntryGroupProviderRequest` / `EntryGroupProviderResult`：`Impl` 表达 runtime 请求哪个 provider 实现，request 表达 provider 输入边界，result 表达 requested impl、实际 provider 类型、输出 group ids、coverage-correct/exact-minimal 语义、fallback 状态和 provider 耗时；request 自己负责 `validate()`、当前 entry groups 是否存在、以及可选 true group id 读取，result 自己负责 `mark_fallback()`；`SearchEntryProvider` 负责 dispatch，并支持注入 CPU provider callback 和可选 GPU provider callback。`run_entry_group_provider()` 现在同时注入 CPU callback 和 GPU callback；GPU callback 惰性创建 `GpuCoverFrontierProvider`，从 `_group_id_to_label_set` 和 LNG descendants 构造常驻 CUDA bitset 表，按 query 输出 coverage-correct entry groups；如果 CUDA 初始化或执行失败，会标记 fallback 并调用 CPU provider。`prepare_entry_groups_for_execution()` 只负责把 result 写回 `entry_group_ids` 和 `QueryStats::num_entry_points`。`entry_group_provider_impl_name()` / `entry_group_provider_kind_name()` 统一 provider 命名，`parse_entry_group_provider_impl()` 统一 CLI/脚本/benchmark 的字符串解析，避免后续入口各自硬编码 provider 字符串。`search_UNG_index` 已暴露 `--entry_group_provider`，支持 `cpu_min_super_sets/cpu/0` 和 `gpu_cover_frontier/gpu/1`；`scripts/benchmarks/run_end_to_end_recall_ab.sh` 用 `ENTRY_GROUP_PROVIDER` 透传。`search_UNG_index` 也暴露 `--is_ung_more_entry`，默认 false；只有显式打开该 flag 时，`query_group_id_file` 才会影响 oracle/expanded entry group 选择。CPU min-super-set provider 的底层入口 `get_min_super_sets_debug()` 已经是 const/read-only 方法，`search_UNG_index` 和 bitmap benchmark 不再需要 `const_cast`，这使 provider 边界更接近“输入 query labels，输出 group ids，不修改 index”的接口契约。`QueryStats` 现在所有字段都显式默认初始化，新 route 或 provider 只需要覆盖自己负责的字段，未覆盖字段会稳定输出 0，而不是未定义值。Graph adjacency 侧已有 `GraphSearchBackend`，并可显式选择 `--graph_search_backend csr`。
`gpu_cover_frontier` 已从 benchmark provider 接入 production search route，但当前实现是 per-query 串行 CUDA provider：为了适配现有 per-query thread pool，它用 provider mutex 保护共享 device state，不等价于 benchmark 工具的 batch throughput。下一步应做 query-batch/stream 化并重新跑端到端 recall/latency，而不是直接沿用独立 benchmark 的 19.85x 入口组阶段结论。
```

## 7. Query Entry Group 方法

当前实现主要在 benchmark 工具中：

```text
UNG/codes/tools/query_entry_group_bench.cu
scripts/benchmarks/run_query_entry_group_bench.sh
```

| Provider | 分类 | 语义 | 状态 |
|---|---|---|---|
| `cpu_scan` | baseline | 输出所有满足 query label subset 的候选 group | benchmark |
| `cpu_exact` | baseline | exact minimal frontier，输出最少 | benchmark，慢 |
| `gpu_scan` | diagnostic | GPU raw candidate scan | 只用于诊断 |
| `gpu_bitset` | diagnostic | GPU bitset raw candidate | 只用于诊断 |
| `gpu_cover_frontier` | main candidate | coverage-correct，允许冗余入口组 | 已接入 production search；当前 per-query 串行 CUDA provider，待批处理化和端到端复测 |

`gpu_cover_frontier` 算法：

```text
C = all matching groups
F_raw = candidate frontier within label-size window
F = first cap groups from F_raw
Covered = union(descendants(f) for f in F)
Output = F union (C - Covered)
```

保证：

```text
不会漏掉实际候选 group。
```

不保证：

```text
输出是 exact minimal。
```

## 8. 当前推荐实验口径

### 8.1 CPU 原始版本

```bash
UNG_BUILD_PROFILE=original_cpu
```

### 8.2 当前 CPU baseline

```bash
UNG_BUILD_PROFILE=current_cpu
```

### 8.3 naive GPU baseline

```bash
UNG_BUILD_PROFILE=naive_gpu
```

### 8.4 cross-edge 论文 fused

```bash
UNG_BUILD_PROFILE=paper_fused
UNG_UNIVERSAL_GPU=1
UNG_CROSS_EDGE_GPU_STRICT=1
```

### 8.5 group graph 当前 router

```bash
UNG_BUILD_PROFILE=custom
UNG_GROUP_GRAPH_IMPL=4
```

### 8.6 100%x40 显存边界

```bash
UNG_BUILD_PROFILE=custom
UNG_CROSS_EDGE_IMPL=1
UNG_GPU_TOPK_IMPL=3
UNG_GPU_X_STREAMING=1
UNG_CROSS_EDGE_GPU_STRICT=1
```

注意：这条是 boundary/scalability 口径，不是性能主结果。

## 9. 新增方法接入规则

新增方法必须同时满足：

1. 在 `UngBuildConfig` 或对应正式 config struct 中有选择入口。
2. 在本文档登记分类：`main`、`baseline`、`ablation`、`diagnostic`、`boundary` 或 `negative`。
3. 有明确语义：exact、coverage-correct、approximate、diagnostic。
4. 有明确输出格式：`SearchQueue`、id-vector、flat-id、CSR 或 compact group ids。
5. 有至少一个复现脚本或 benchmark 命令。
6. 有 artifact 记录或被标注为“尚未实测”。

不要再新增只靠隐藏 env 和聊天上下文才能理解的实现路径。
