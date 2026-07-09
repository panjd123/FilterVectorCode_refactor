# UNG 优化代码结构审计

更新时间：2026-07-01

## TLDR

当前代码**已经从临时实验堆叠状态明显收敛，但还不能算完全简洁清晰**。更准确地说：

```text
配置层已经收敛：UngBuildConfig 能表达 CPU、naive GPU、paper fused、group graph router 等主要方法。
接口层部分清楚：tagore_graph_builder.h、SearchRuntimeConfig、EntryGroupProviderRequest/Result、CrossEdgeBuildResult 是目前最像可替换 backend 的 C++ API。
实现层仍然偏重：gpu_gemm_topk.cu 还承担 cross-edge 顶层 CUDA route，部分模块还只是“职责拆文件”而不是正式 backend 接口；但 `uni_nav_graph.cpp` 已收敛为 build 总调度、group storage 准备和 finalize/offset helper。LNG/trie/coverage 已集中进入 uni_nav_graph_label_graph.cpp，vector-attribute graph 构建/比较已集中进入 uni_nav_graph_vector_attr.cpp，legacy ACORN/guarantee 补边已集中进入 uni_nav_graph_acorn_augment.cpp，CPU/GPU group graph 已集中进入 uni_nav_graph_group_graph.cpp，cross-edge CPU/GPU orchestration 已集中进入 uni_nav_graph_cross_edges.cpp，search/query 执行路径已集中进入 uni_nav_graph_search.cpp，index save/load/statistics 和 legacy global_graph 持久化兼容已集中进入 uni_nav_graph_io.cpp，profiling helper 已进入 ung_prof_log.{h,cpp}。
```

这意味着仓库现在“能跑、能切换、能复现实验”，但还没有达到“新方法容易接入、reviewer 或后续同学一眼能看懂”的程度。最需要整理的不是删除 baseline，而是把主路径、baseline、diagnostic、boundary、negative path 的边界写清楚，并逐步把大函数拆成 backend adapter。

2026-07-01 的补充结论：**目前的代码已经足够支撑论文实验复现和方法 A/B，但还不应宣称为最终清晰架构**。现在新增一个方法时，正确路径应该是先定义或复用 typed config / request / result，再接入 router；不应该直接新增一个环境变量并在 CUDA/搜索深层函数里读取。查询侧已经把运行配置构造收敛到 `make_search_runtime_config()`，`search_UNG_index` 和历史长签名 `search_hybrid(...)` 都只作为薄入口进入同一份 `SearchRuntimeConfig`，避免后续新增 search 开关时两个入口字段漂移。

同日继续收敛 `uni_nav_graph.cpp`：文件头重复 include 已清理，只保留 build orchestration、`MethodSelector` 析构边界和 graph reserve 所需依赖；临时中文 LNG 边数分析块已改成命名 helper `print_label_nav_graph_summary()`，输出 `[label_nav_graph] groups/edges/avg_out_degree`，并同步写 `[PROF] label_nav_graph.summary`。这不改变算法，只是把主 build 流程从“夹杂临时诊断代码”收回到“调用命名阶段 helper”。

配置入口也继续收敛：`UngBuildConfig::from_env()` 不再接收未使用的 `index_name`，profile 默认值应用被拆到 `apply_profile_defaults()`，`UNG_GROUP_GRAPH_IMPL` / `UNG_LNG_IMPL` / `UNG_CROSS_EDGE_IMPL` / `UNG_ADDITIONAL_EDGES_IMPL` / `UNG_GPU_TOPK_IMPL` 的 int-to-enum 映射被拆成命名 `*_from_int()` helper。这样 profile、custom env 和 enum 映射三层关系更清楚，后续新增 profile 或枚举值时不需要改一串嵌套三目表达式。

`uni_nav_graph.h` 也补了一轮内部状态默认值：`_num_points`、`_num_groups`、`_num_attributes`、`_num_cross_edges`、`_max_degree`、`_Lbuild`、`_alpha`、`_num_threads` 和 legacy ACORN edge counter 现在都有 in-class initializer；无调用点的 `ENABLE_SEARCH_PATH_LOGGING` 已删除。这不是算法变化，而是让默认构造、load 前诊断和异常路径不再依赖未初始化标量。

本轮追加的接口清理：删除了只剩注释调用的 `save_bipartite_graph_info()` 早期调试 dump helper。vector-attribute bipartite graph 现在只保留正式的 `count_graph_edges()`、`save_bipartite_graph()`、`load_bipartite_graph()` 和 `compute_checksum()` 边界，避免后续误以为主流程还支持一个独立文本 info dump 路径。随后又删除了只剩注释调用/无调用点的 global Vamana 构建和 global fixed-point 查询 helper；`global_graph` 文件和 `_global_vamana_entry_point` 字段仅作为旧索引格式兼容持久化，不再作为可接入搜索 backend。legacy ACORN augment 中残留的个人补丁式阶段注释也已改成稳定的 ACORN augmentation / guarantee repair 阶段说明，避免把 legacy 可运行路径误读成未清理临时代码。

2026-06-26 的最新纠偏结论是：cross-edge 已经有了第一层显式 config、公共 descriptor、regular route/config builder、double-buffer helper 文件边界，以及 regular all-batched per-group dispatch 的结构化接口；但还没有完成深层 backend 接口化。静态扫描显示，`gpu_gemm_topk.cu` 和 `gpu_cross_edge_*.cuh` 不再直接读取 route env，planning、double-buffer、regular route 和 benchmark filter 都显式消费 `CrossEdgeGpuRuntimeConfig&`；`UNG_BENCH_*` 的读取点也已经进入 `CrossEdgeGpuRuntimeConfig::benchmark_filter`。`make_cross_edge_gpu_runtime_config()` 也已拆成 route flags、output flags、kernel flags、kernel limits、batch capacity、benchmark filter、writeback derivation 几个局部 helper，避免后续新增 cross-edge 路径继续往一个大函数堆环境变量。`CrossEdgeGpuRuntimeConfig::summary()` 已拆成 route/output/kernel/capacity/diagnostics/benchmark/writeback summary helper，字段顺序保留旧日志顺序；`apply_cross_edge_gpu_topk_env_defaults()` 也已拆成 `CustomNaive`、`SgemmTopk`、`FusedGroupTopk` 三个 profile defaults helper，顶层函数只负责按 `UngGpuTopkImpl` 分派。build/group graph/output-boundary 相关 settings 已从 `uni_nav_graph.cpp` 移到 `include/ung_build_settings.h` / `src/ung_build_settings.cpp`，主流程不再直接读这批 env。Tagore/FastGrnnd CUDA builder 的 runtime tuning 也已进入公开 `TagoreCudaRuntimeConfig`，其中 `TagoreFastExactBatchConfig` 表达 packed exact-anchor 的 H2D/device-lookup/compact-D2H/warp-kernel 选项，`TagoreFastGrnndPruneConfig` 表达 light/reverse/repair prune 选项；`tagore_cuda_config.cpp` 负责把 `UNG_FAST_*` / `UNG_TAGORE_*` env 转成 runtime config，`tagore_graph_builder.cu` 只消费 config 并执行 CUDA kernel，不再承担 env parsing。`build_graph_for_all_groups_tagore_cuda()` 现在创建一次 runtime config，并把 `TagoreGroupGraphSettings::exact_batch_threshold()` 写回同一份 config 后传给 `build_tagore_batch_artifacts()` 和 CUDA batch builder，避免 router split 阈值与 CUDA 内部 exact-batch 阈值不一致。旧的无 config overload 仍作为兼容入口保留，只负责从 env 构造 config 后转发。CPU Vamana group graph、complete/bounded-complete fallback、Tagore/FastGrnnd 的 CUDA group graph 主入口和四个阶段 helper 已移动到 `src/uni_nav_graph_group_graph.cpp`；cross-edge backend resolve、CPU baseline/exact/hybrid、GPU route adapter、additional_edges、offset/merge/writeback orchestration 和 original CPU path 已移动到 `src/uni_nav_graph_cross_edges.cpp`；search/query 已拆成五层：`src/uni_nav_graph_query_features.cpp` 负责 selector feature/diagnostic，`src/uni_nav_graph_query_route.cpp` 负责 route policy，`src/uni_nav_graph_entry_provider.cpp` 负责 entry-group provider，`src/uni_nav_graph_search_backend.cpp` 负责 ACORN/UNG graph expansion，`src/uni_nav_graph_search.cpp` 只保留 thread orchestration 和 result writeback；index save/load、vector-attribute bipartite graph persistence、ACORN inverted-index loading 和 index size statistics 已移动到 `src/uni_nav_graph_io.cpp`；trie grouping、`get_min_super_sets`、LNG build、descendants、coverage 和 entry-group bitmap helpers 已移动到 `src/uni_nav_graph_label_graph.cpp`；vector-attribute graph 构建和比较 helper 已移动到 `src/uni_nav_graph_vector_attr.cpp`；legacy ACORN/guarantee distance-oriented 补边已移动到 `src/uni_nav_graph_acorn_augment.cpp`，只服务它的 `NewEdgeCandidate` 也已从公开头文件降到该 `.cpp` 的匿名 namespace。原先暴露在 `UniNavGraph` public 区域的 query generator API 只有声明、没有任何实现定义，并且 `build_UNG_index.cpp` 中唯一调用点也是整块注释代码；这批假 API 已删除，避免后续误以为仍有可调用的 query generation 子系统。最新一轮又把 `_num_points`、vector-attribute bipartite graph、attribute id map、LNG descendants/coverage bitset/roaring cache 这些内部状态从 public API 移到 private，并在 public 区域显式标注正式 lifecycle/search API 与 diagnostic/benchmark helper 的边界。`uni_nav_graph.h` 的私有声明也已按 search runtime、query route、entry provider、query feature/search backend、label graph、group storage、group graph、cross-edge orchestration 和 CUDA cross-edge backend entry 分组；C++ build profiling helper 已移动到 `include/ung_prof_log.h` / `src/ung_prof_log.cpp`，避免后续拆文件继续依赖 `uni_nav_graph.cpp` 的匿名 namespace。本轮还清理了 `uni_nav_graph_io.cpp`、`uni_nav_graph_label_graph.cpp`、`trie.cpp` 和 `utils.cpp` 中残留的 `fxy_add`、`fxy_mod`、`FXY_ADD`、装饰性分隔线、死代码和临时“步骤”注释，把 coverage/entry bitmap、trie debug metrics、recall/roaring serialization 等正式路径改成稳定算法说明；`utils.cpp` 的 roaring serialization 也从裸 `new[]/delete[]` 改为 `std::vector<char>` 缓冲。随后又删除了只剩私有声明、没有定义或调用点的 legacy pair-wise GPU cross-edge helper 和 GPU additional-edge helper。这些改动说明代码边界已经更可审计，但当前代码还不是“只改一个 backend config 就能完全理解行为”的状态。

## 1. 当前结构判断

| 层次 | 当前状态 | 结论 |
|---|---|---|
| 顶层 profile | `UNG_BUILD_PROFILE={original_cpu,current_cpu,naive_gpu,paper_fused,custom}` | 清楚，应该保留；profile defaults 与 custom env 解析已在 `UngBuildConfig::from_env()` 中分层 |
| 主要枚举 | `UngGroupGraphImpl`、`UngCrossEdgeImpl`、`UngGpuTopkImpl` 等 | 基本清楚，但还不能覆盖所有深层子路径 |
| C++ backend API | `tagore_graph_builder.h` 比较干净 | 可作为后续 backend 接口模板 |
| build orchestration | `UniNavGraph::build()` | 可用但仍偏重；现在主要串联 label graph、group graph、cross-edge、I/O 等模块；LNG summary 诊断已收敛到 `print_label_nav_graph_summary()` |
| label graph / trie / coverage | `uni_nav_graph_label_graph.cpp` | trie grouping、min-super-set、entry-group selection、bitmap helpers、LNG build、descendants、coverage、roaring bitsets 已从主文件拆出；下一步可抽成 `LabelGraphBuilder` / `EntryGroupProvider` |
| vector-attribute graph | `uni_nav_graph_vector_attr.cpp` | vector-attribute bipartite graph 构建和比较 helper 已从主文件拆出；持久化仍在 `uni_nav_graph_io.cpp` |
| legacy ACORN augment | `uni_nav_graph_acorn_augment.cpp` + `ung_acorn_augment_config.h` | ACORN 脚本调用、distance-oriented edge 解析/过滤、guarantee 边修复和 `AcornInUng` 配置已从主文件/主头文件拆出；`AcornInUng{}` 默认关闭 legacy augment，`build_UNG_index` 不再复制硬编码默认值；外部脚本路径必须通过 `UNG_ACORN_SCRIPT_PATH` 显式提供；它是 legacy/experimental 路径，不应承载新的 cross-edge 主实现 |
| group graph adapter | `uni_nav_graph_group_graph.cpp` + `ung_group_graph_build_types.h` | CPU Vamana、complete/bounded-complete、Tagore/FastGrnnd 主入口、fallback、batch split、fill/writeback 已从主文件拆出；group graph 中间结果类型已从 `UniNavGraph` 类体拆出；下一步应形成独立 `GroupGraphBackend` |
| cross-edge orchestration | `uni_nav_graph_cross_edges.cpp` + `ung_cross_edge_config.{h,cpp}` + `ung_cross_edge_output_writer.{h,cpp}` | CPU exact/Vamana/hybrid、GPU route adapter、additional_edges、offset/merge/writeback 已从主文件拆出；`CrossEdgeBackend` 已从 `UniNavGraph` 类体移到 cross-edge config 层；CPU-compatible graph materialization 和 CSR adjacency buffer writer 已进入独立 output writer；search 侧可显式选择 CSR adjacency backend |
| query feature / selector support | `uni_nav_graph_query_features.cpp` | selector warmup、Idea1/Idea2 feature order、feature-only CSV 诊断导出、pre-trie heuristic 和 Idea2 entry-group route stats 已从 search 执行文件拆出；active feature order 是训练模型契约 | 新 graph search backend、build-time graph construction |
| query route policy | `uni_nav_graph_query_route.cpp` + `ung_query_route.h` | `decide_query_route()` 已从 search 执行文件拆出；集中处理 Idea1/Idea2 selector、pre-trie heuristic 应用、force route 覆盖和 route stats；`make_search_runtime_config()` 统一 app 入口和历史兼容 wrapper 的 runtime 字段构造 | ACORN/UNG graph search、entry-group provider dispatch |
| entry group provider | `uni_nav_graph_entry_provider.cpp` + `ung_entry_group.{h,cpp}` + `ung_gpu_cover_frontier_provider.{h,cu}` | provider dispatch、CPU min-super-set provider、production GPU correct-cover provider 和 `ung_more_entry` 扩展已从 search 执行文件拆出；request/result 已经表达 provider 输入输出和 fallback 状态，GPU provider 失败时会显式 fallback 到 CPU | query route decision、ACORN/UNG graph search |
| search backend | `uni_nav_graph_search_backend.cpp` | ACORN query backend、UNG query backend、entry-point expansion、UNG fixed-point graph expansion 已从 search orchestration 文件拆出 | query route policy、provider dispatch、thread orchestration |
| search/query orchestration | `uni_nav_graph_search.cpp` | search 文件现在只消费 `SearchRuntimeConfig`、route/provider 结果和 backend helper，负责 per-query orchestration、thread task creation 和 result writeback；不再直接包含 `MethodSelector` 或 ACORN/Faiss 重头文件；旧 `search_hybrid` 长签名只作为兼容包装，并通过 `make_search_runtime_config()` 转发到结构化 overload；thread-pool task 直接捕获 query id，不再通过共享 queue/mutex 分发；`thread_function()` 现在只在 UNG graph-search 分支创建 `SearchCacheList`，ACORN 路径不再做无用 cache 分配；`SearchEntryProvider` 已承载 provider impl dispatch；`--entry_group_provider` / `ENTRY_GROUP_PROVIDER` 已能选择 provider impl；`--graph_search_backend csr` 可显式选择 CSR adjacency backend；`GpuCoverFrontier` 已有 production CUDA correct-cover callback，按 query 惰性初始化常驻 device bitset 表并通过互斥锁保护 CUDA 调用；GPU 初始化/执行失败时 fallback 状态进入 result；`QueryStats` 全字段已有默认值，避免新增 route 时产生未定义统计输出；selector model 参数内部统一为 `selector_model_prefix`，旧 CLI 拼写仅作为兼容 alias；下一步是把 GPU provider 从 per-query 串行调用扩展为批处理/stream 化并做端到端复测 |
| index I/O/statistics | `uni_nav_graph_io.cpp` | index save/load、二分图 save/load、ACORN inverted-index loading、index size statistics 已从主文件拆出；下一步可把 meta/build-time CSV 写出拆成更小的 `IndexMetadataWriter` |
| public class declaration | `uni_nav_graph.h` | public 面已分成 primary lifecycle/search API 与 diagnostic/benchmark helper；`_num_points`、vector-attribute graph、attribute id map、LNG descendants/coverage cache 已降为 private 内部状态，内部标量状态已有默认初始化；私有声明已按 search/query、label graph、group storage、group graph、cross-edge orchestration、CUDA cross-edge backend entry 分组；`QueryStats` 已拆到 `ung_query_stats.h`；`QueryRouteDecision` / `SearchRuntimeConfig` 已拆到 `ung_query_route.h`；entry-group `SelectionMode` / `EntryGroupProviderImpl` / `EntryGroupProviderRequest` / `EntryGroupProviderResult` 已拆到 `ung_entry_group.h`；legacy `AcornInUng` 已拆到 `ung_acorn_augment_config.h`；group graph helper result types 已拆到 `ung_group_graph_build_types.h`；`CrossEdgeBackend` 已拆到 `ung_cross_edge_config.h`；ACORN/Faiss 重头文件已下沉到 search/io `.cpp`，头文件只保留 `faiss::IndexACORNFlat` 前置声明；ONNX `MethodSelector` 和 CPU `Vamana` 也已从 public header 下沉，`uni_nav_graph.h` 只保留前置声明，构造/析构函数 out-of-line 定义以保证 `unique_ptr` / `shared_ptr` 相关类型边界清楚；`ung_build_settings.h` 不再包含完整 Tagore builder API，只前置声明 `TagorePruneMode`，具体枚举值使用留在 `.cpp` | 比原来可读，但仍是一个大类头文件 |
| profiling helper | `ung_prof_log.{h,cpp}` | 不再依赖 `uni_nav_graph.cpp` 匿名 namespace，后续模块可共享 build profiling |
| CUDA cross-edge | `gpu_gemm_topk.cu` + `gpu_cross_edge_common.cuh` + `gpu_cross_edge_planning.cuh` + `gpu_cross_edge_buffers.cuh` + `gpu_cross_edge_topk_device.cuh` + `gpu_cross_edge_update_kernels.cuh` + `gpu_cross_edge_resident_vectors.cuh` + `gpu_cross_edge_cpu_tiny.cuh` + `gpu_cross_edge_query_pack.cuh` + `gpu_cross_edge_bucket_descriptors.cuh` + `gpu_cross_edge_group_dispatch.cuh` + `gpu_cross_edge_regular_route.cuh` + `gpu_cross_edge_singleton_kernels.cuh` + `gpu_cross_edge_large_group_kernels.cuh` + `gpu_cross_edge_descriptor_kernels.cuh` + `gpu_cross_edge_group_fused_kernels.cuh` + `gpu_cross_edge_double_buffer.cuh` + `gpu_cross_edge_sgemm_baseline.cuh` + `gpu_cross_edge_bucket_fused.cuh` + `gpu_cross_edge_per_group_fused.cuh` + `gpu_cross_edge_output.cuh` + `gpu_cross_edge_profile.cuh` + `gpu_cross_edge_verify.cuh` + `gpu_cross_edge_x_streaming.cuh` + `gpu_cross_edge_source_exact*.cuh` + `gpu_cross_edge_cuvs_baseline.cuh` | descriptor、route-neutral planning、GPU 复用缓存/handle、topK device primitive、norm/topK update kernels、resident all-X upload/release、CPU tiny exact fallback、query packing/upload、bucket descriptor assembly、per-group dispatch、regular route/config builder、singleton fastpath kernel、large-group fused kernel、descriptor-batched TF32 kernel、small/medium/group-desc fused kernel、double-buffer helper、SGEMM/separate-topK baseline、bucket fused descriptor launch、per-group fused kernel launch、普通 batched output finalization、profiling summary、optional verify、X-streaming boundary helper、source-exact experimental path 和 cuVS baseline 已拆出；resident vector upload 的 host-register/direct-pageable 策略已进入 `CrossEdgeGpuRuntimeConfig::vector_upload`；主 `.cu` 仍保留顶层 route setup 和高层执行分支 |
| query entry GPU | `query_entry_group_bench.cu` | 目前是 benchmark 工具，不是正式 search backend |
| 文档入口 | `docs/README.md` 已区分人看/AI 看 | 已改善，但代码方法注册仍需单独维护 |

## 1.1 接口成熟度结论

当前代码不能简单回答“已经清楚”或“仍然混乱”。更准确的分层结论如下：

| 成熟度 | 范围 | 现状 | 维护要求 |
|---|---|---|---|
| 已经清楚 | `UngBuildConfig` profiles、`TagoreGroupRequest/Result`、`EntryGroupProviderRequest/Result`、`SearchRuntimeConfig`、`CrossEdgeBuildResult` | 已经有结构化输入输出或统一 enum，后续方法可以按这个模式接入；`SearchRuntimeConfig` 也已有统一构造 helper，避免入口字段漂移 | 新增方法必须先登记 enum / request / result / runbook |
| 基本清楚 | `uni_nav_graph_group_graph.cpp`、`uni_nav_graph_cross_edges.cpp`、`uni_nav_graph_search.cpp` | 已经按职责拆出文件，search entry provider 和 graph adjacency backend 已有对象边界，但不少 build/cross/group helper 仍是 `UniNavGraph` 成员函数，依赖内部状态 | 下一步抽 `GroupGraphBackend`、继续收窄 `CrossEdgeBackend` |
| 半清楚 | `CrossEdgeGpuRuntimeConfig`、`gpu_cross_edge_*.cuh` | route/config/planning/dispatch/output helper 已拆出；但 config 太宽，baseline、boundary、debug、kernel tuning 还在同一个结构里 | 新增 CUDA 路径要先选分类：main/baseline/diagnostic/boundary/negative，并尽量落到独立 helper |
| 已接入但待复测 | GPU correct-cover production query provider | `ung_gpu_cover_frontier_provider.{h,cu}` 已把 benchmark correct-cover bitset/frontier 逻辑改造成 production `SearchEntryProvider` callback；当前按 query 串行调用 CUDA，保留 CPU fallback | 不能直接沿用 benchmark 加速结论；需要做批处理化和端到端 recall/latency 复测 |

面向“后续要继续接很多新方法”的标准，当前结论如下：

| 检查项 | 当前判定 | 依据 |
|---|---|---|
| 大方向方法是否有统一开关 | 基本达标 | build profile、group graph、cross-edge、GPU topK、additional edges 和 entry group provider 都已经有 enum/config 层。 |
| 单个方法是否有独立 C++ 接口 | 部分达标 | Tagore/FastGrnnd batch builder、entry-group request/result、search runtime config 比较清楚；cross-edge CUDA 内部仍主要通过 `gpu_gemm_topk.cu` 的 route helper 和 `.cuh` helper 组合。 |
| 文件结构是否容易理解 | 明显改善，但未完成 | `uni_nav_graph.cpp` 已经不再堆叠 build/query/io/group/cross-edge/label graph；但 `UniNavGraph` 仍是大 facade，许多 helper 仍依赖内部成员状态。 |
| 是否还有明显误导主路径的死入口 | 本轮范围内已清理 | query generator 假 API、legacy pair-wise GPU cross-edge 声明、GPU additional-edge 声明和 bipartite info dump helper 已删除；剩余 `legacy/experimental/fallback` 大多有明确用途或负例定位。 |
| 新同学能否直接加第四种方法 | 还不够理想 | 可以接入，但需要先读 `UniNavGraph` 内部状态和 cross-edge CUDA route；下一步应把 backend 对象化，把输入输出从类成员状态里剥离。 |

因此，当前仓库的主问题已经不是“旧方法和新方法完全混在一起”，而是**外层选择清楚、内层 backend 对象边界还不够硬**。后续清理不应优先删除 baseline，而应优先把三类接口对象化：

```text
GroupGraphBackend         -> build group-local or packed graph, return entry point + timing
CrossEdgeBackend          -> consume group workload, produce edge ids + timing
CrossEdgeOutputWriter     -> materialize SearchQueue/id-vector/flat-CSR without leaking backend details
SearchEntryProvider       -> consume query labels, produce coverage-correct entry group ids
```

## 2. 目前清楚的部分

### 2.1 `UngBuildConfig`

`UNG/codes/include/ung_build_config.h` 和 `UNG/codes/src/ung_build_config.cpp` 已经把大多数“选择哪种方法”的入口收敛到了一个结构：

```text
UngBuildProfile
UngGroupGraphImpl
UngCrossEdgeImpl
UngAdditionalEdgesImpl
UngGpuTopkImpl
```

这是正确方向。后续不应该再绕开它新增一个顶层实现开关。新增方法应先进入 enum，再在 runbook 和 method registry 中登记。

### 2.2 `tagore_graph_builder.h`

这是当前最清楚的 backend API：

```text
TagoreGroupRequest
TagoreBuildResult
TagoreBatchBuildResult
build_tagore_vamana_cuda()
build_tagore_vamana_cuda_batch()
TagorePruneMode
```

它的优点是输入输出基本独立于 `UniNavGraph` 内部状态。当前 batch API 只保留 `TagorePruneMode` 枚举入口；旧的 `bool grnnd_like_refine` overload 已删除，避免用布尔参数隐式表达方法类型。后续 group graph 的 GPU backend 应尽量维持这种接口风格：给定 group 数据和参数，返回 graph、entry point 和分阶段 timing。

### 2.3 query-entry 语义已经清楚

`gpu_cover_frontier` 的语义已经和旧 `gpu_frontier` 区分开：

```text
必须保证 coverage，不要求 exact minimal。
```

这是一个正确接口边界。但它还停留在 benchmark 工具中，正式 `search_UNG_index` 还没有以 compact output 接入这个 provider。

## 3. 主要结构问题

### 3.1 `uni_nav_graph.cpp` 仍然偏重，但 search/query 已拆出

文件规模已从约 `4.9k` 行下降到约 `0.29k` 行，`include/uni_nav_graph.h` 已从约 `0.64k` 行下降到约 `0.45k` 行，query/search 统计结构已移入 `include/ung_query_stats.h`，query route decision 已移入 `include/ung_query_route.h`，entry-group selection mode 已移入 `include/ung_entry_group.h`，legacy ACORN augment 配置已移入 `include/ung_acorn_augment_config.h`，group graph 构建中间结果类型已移入 `include/ung_group_graph_build_types.h`，cross-edge backend 分类已移入 `include/ung_cross_edge_config.h`。当前 `src/uni_nav_graph_label_graph.cpp` 约 `1.12k` 行，`src/uni_nav_graph_search.cpp` 约 `0.16k` 行，`src/uni_nav_graph_search_backend.cpp` 约 `0.31k` 行，`src/uni_nav_graph_query_route.cpp` 约 `0.12k` 行，`src/uni_nav_graph_query_features.cpp` 约 `0.61k` 行，`src/uni_nav_graph_entry_provider.cpp` 约 `0.09k` 行，`src/uni_nav_graph_io.cpp` 约 `0.89k` 行，`src/uni_nav_graph_group_graph.cpp` 约 `0.90k` 行，`src/uni_nav_graph_cross_edges.cpp` 约 `0.86k` 行，`src/gpu_gemm_topk.cu` 约 `1.08k` 行；search/query 现在分成五层：

```text
selector feature collection and diagnostics
Idea1 / Idea2 feature order contract
query route policy
entry-group provider dispatch and expansion
ACORN / UNG query backend
per-query orchestration
entry group -> entry point expansion
fixed-point graph search
```

这一步解决的是“查询逻辑混在 build/save/statistics 里”的问题；当前又进一步把 selector 支撑逻辑拆到 `uni_nav_graph_query_features.cpp`，把 active `features.push_back(...)` 顺序固定为训练模型接口契约，并集中放置 `calculate_query_features_only()`、`warmup_selectors()`、`check_pre_trie_heuristic()` 和 `populate_entry_group_route_stats()`。`decide_query_route()` 已移动到 `uni_nav_graph_query_route.cpp`，集中处理 Idea1/Idea2 selector 调用、pre-trie heuristic 应用、force route 覆盖和 route stats。route 后的 entry-group provider dispatch、CPU min-super-set provider、GPU correct-cover provider 和 `ung_more_entry` 扩展已移动到 `uni_nav_graph_entry_provider.cpp`；CUDA provider 的 device bitset 表和 kernels 位于 `ung_gpu_cover_frontier_provider.{h,cu}`。ACORN query backend、UNG query backend、entry-point expansion、UNG fixed-point graph expansion 已移动到 `uni_nav_graph_search_backend.cpp`。search orchestration 文件现在只保留 runtime config consumption、per-query orchestration、thread dispatch 和 result writeback；`thread_function()` 也只在 UNG graph-search 分支创建 `SearchCacheList`，ACORN 路径不再做无用 cache 分配，且 search 文件不再直接包含 `MethodSelector` 或 ACORN/Faiss 重头文件。ACORN 执行阶段和 UNG fixed-point 执行阶段已经拆成 `execute_acorn_query()` / `execute_ung_query()`，query route 状态收敛到 `QueryRouteDecision`，运行时参数收敛到 `SearchRuntimeConfig`，route 后的 entry group 选择和输入输出收敛到 `EntryGroupProviderImpl`、`EntryGroupProviderRequest` / `EntryGroupProviderResult`。`search_UNG_index` 的主调用点现在通过 `make_search_runtime_config()` 构造 `SearchRuntimeConfig`；CLI 参数 `--entry_group_provider` 支持 `cpu_min_super_sets/cpu/0` 和 `gpu_cover_frontier/gpu/1`，并通过 `parse_entry_group_provider_impl()` 复用同一套解析逻辑；`scripts/benchmarks/run_end_to_end_recall_ab.sh` 通过 `ENTRY_GROUP_PROVIDER` 透传这个选择；`UniNavGraph` 仍保留历史 `search_hybrid(...)` 长签名，但它只是调用同一个 `make_search_runtime_config()` 后转发到结构化 overload，避免 app 入口和兼容 wrapper 对 runtime 字段的理解漂移。`thread_function()` 现在主要保留 query dispatch、调用 route helper、构造 entry-group provider request、调用执行 helper 和结果写回，不再透传 `Lsearch/K/efs/force_use_alg` 等散参数。`uni_nav_graph.h` 中 search/query 相关私有声明也已集中到一个 `Search/query orchestration helpers` 分组，不再散落在 selector 状态和 graph-search 声明之间；`QueryStats`、`QueryRouteDecision`、`SearchRuntimeConfig`、`EntryGroupProviderImpl`、`EntryGroupProviderRequest` 和 `EntryGroupProviderResult` 已拆到独立头文件。`SearchRuntimeConfig::entry_group_provider` 目前默认 `CpuMinSuperSets`；选择 `GpuCoverFrontier` 时 production search 会惰性初始化 GPU provider，复用 `_group_id_to_label_set` 和 LNG descendants 构造常驻 device bitset 表，输出 coverage-correct 入口组；如果 GPU 初始化或 kernel 执行失败，则显式 fallback 到 CPU provider。`QueryStats` 现在所有字段都有显式默认值；新增 route/provider 即使只写自己负责的字段，CSV 和 benchmark 也会稳定输出 0，而不是未定义内存。`search_UNG_index` 和 `UniNavGraph::load()` 内部已统一使用正确拼写 `selector_model_prefix`，旧 `selector_modle_prefix` 只作为 deprecated CLI alias 兼容历史脚本。底层 `get_min_super_sets_debug()` 已经改为 const/read-only 查询接口，`search_UNG_index` 和 bitmap helper 不再通过 `const_cast` 调用它，这让 entry-group provider 的语义更明确：读取 index/trie，写输出 group ids 和 stats，不修改 index。legacy `AcornInUng` 配置已拆到独立头文件，ACORN/Faiss 的具体头文件只在 `uni_nav_graph_search_backend.cpp` / `uni_nav_graph_io.cpp` 中包含。`save()`、`load()`、`statistics()`、二分图持久化和 ACORN inverted-index loading 已移动到 `uni_nav_graph_io.cpp`。trie/LNG/coverage 相关实现已移动到 `uni_nav_graph_label_graph.cpp`，主文件不再直接承载 `get_min_super_sets`、descendants、coverage 和 roaring bitset 初始化。vector-attribute graph 构建/比较已移动到 `uni_nav_graph_vector_attr.cpp`。legacy ACORN/experimental 补边已移动到 `uni_nav_graph_acorn_augment.cpp`，`NewEdgeCandidate` 不再污染公开头文件。无实现的 query generator public API 和 `build_UNG_index.cpp` 中整块注释调用已删除；需要生成 query 时应使用 `UNG/codes/tools/generate_query_labels.cpp` 或 `UNG/codes/tools/generate_mixed_queries.cpp` 这类独立工具，而不是在 `UniNavGraph` 主类里恢复旧入口。剩余问题是 `uni_nav_graph.cpp` 仍承担 build 总调度、group storage 准备和 offset helper；GPU entry provider 当前是 per-query 串行 CUDA 调用，还需要批处理化和端到端复测。

典型例子：

| 函数 | 现状 | 问题 |
|---|---|---|
| `build_graph_for_all_groups()` | 已移动到 `uni_nav_graph_group_graph.cpp`；仍负责 CPU/GPU group graph route 选择 | 比原来清楚，但还不是独立 backend interface |
| `build_graph_for_all_groups_tagore_cuda()` | 已移动到 `uni_nav_graph_group_graph.cpp`；仍负责 fallback/GPU overlap、timing 汇总和 Tagore/FastGrnnd route 编排 | 比原来清楚，但还不是独立 backend interface |
| `build_cross_group_edges()` | 已移动到 `uni_nav_graph_cross_edges.cpp`；backend 选择和 timing 汇总保留在主调度，additional 和 merge/writeback 已拆到 `build_cross_edges_generate_additional()`、`merge_cross_edges_to_graph()`、`merge_additional_edges_to_graph()`；CPU-side graph materialization 已通过 `CrossEdgeHostOutputView` 统一消费 SearchQueue / id-vector / flat-id host 输出，并由 `CrossEdgeGraphMaterializationWriter` 集中写入 `_graph->neighbors` 或生成 CSR adjacency buffers | cross-edge 主体已离开主文件；output writer 已独立；CSR adjacency 已可作为显式 search backend 输入 |
| `build_cross_edges_generate_gpu_optimized()` | 已移动到 `uni_nav_graph_cross_edges.cpp`；把 `UngGpuTopkImpl` 映射成 runtime route，并选择 source/x-stream/all-batched | 是 routing adapter，但还不是独立 backend interface |
| `search_hybrid()` / `thread_function()` | 已移动到 `uni_nav_graph_search.cpp`；结构化 `search_hybrid(..., const SearchRuntimeConfig&)` 是当前 app 主调用入口，旧长签名只做兼容转发；`search_hybrid()` 创建 per-query task 并直接传入 query id，`thread_function()` 只执行单个 query 的 route/provider/backend 和结果写回，route decision 与 ACORN/UNG 执行细节已下沉到 helper；entry provider dispatch 已进入 `SearchEntryProvider`，graph adjacency 已进入 `GraphSearchBackend` | 查询主路径已离开主文件，app/worker 接口已收束；剩余重点是 GPU entry provider 批处理化和更完整的 backend config/result contract |
| `decide_query_route()` / `QueryRouteDecision` | `decide_query_route()` 已移动到 `uni_nav_graph_query_route.cpp`，`QueryRouteDecision` 在 `ung_query_route.h`；集中承载 Idea1/Idea2 selector route、pre-trie heuristic 应用和 force-use route；feature 计算、pre-trie heuristic 函数和 `populate_entry_group_route_stats()` 在 `uni_nav_graph_query_features.cpp`；`QueryRouteDecision::apply_pre_trie_heuristic()` 统一表达 pre-trie route 应用语义，`QueryRouteDecision::apply_force_route()` 统一表达 `force_use_alg` 覆盖语义 | 是 route selector helper，还不是可插拔 route policy |
| `SearchEntryProvider` / `run_entry_group_provider()` / `prepare_entry_groups_for_execution()` / `compute_cpu_entry_groups_for_execution()` / `compute_gpu_entry_groups_for_execution()` | `SearchEntryProvider` 已进入 `ung_entry_group.{h,cpp}`，负责按 `EntryGroupProviderImpl` dispatch 并返回 `EntryGroupProviderResult`；它支持 CPU provider callback 和可选 GPU provider callback；`uni_nav_graph_entry_provider.cpp` 当前同时注入 CPU callback 和 GPU callback，GPU callback 惰性创建 `GpuCoverFrontierProvider`，失败时通过 `EntryGroupProviderResult::mark_fallback()` 回到 CPU；`prepare_entry_groups_for_execution()` 只应用输出 group ids 和 `num_entry_points` 统计；CPU provider 消费 request/result 契约，集中承载 route 后的 CPU entry group 补算和 `ung_more_entry` 扩展 | 已经具备 provider 对象 + request/result 契约；`GpuCoverFrontier` 已是 production route，但仍是 per-query 串行 CUDA provider，下一步是批处理化和端到端复测 |
| `execute_acorn_query()` | 已移动到 `uni_nav_graph_search_backend.cpp`；负责 ACORN index 选择、efs 设置、old-bitmap/filter-map 查询和结果转入 `SearchQueue` | 是 ACORN 执行 helper，还不是正式 `SearchBackend` |
| `execute_ung_query()` / `iterate_to_fixed_point()` | 已移动到 `uni_nav_graph_search_backend.cpp`；负责 entry group 到 entry point，再通过 `GraphSearchBackend` / `GraphNeighborView` 读取邻接并执行 greedy fixed-point search；`GraphSearchBackend` 已支持 `_graph` 和 `CrossEdgeCsrOutput` 两种邻接来源；`SearchRuntimeConfig::graph_backend` 和 `search_UNG_index --graph_search_backend` 可选择默认邻接表或 CSR | 逻辑边界比原来清楚，但仍绑定 `SearchCache` / `_distance_handler`；如果继续做 GPU query，需要再抽独立 SearchEntryProvider |

### 3.2 `gpu_gemm_topk.cu` 主路径和实验路径混杂

文件规模约 `1.1k` 行，当前仍包含：

```text
naive dot / gather / local-to-global merge kernels
norm / topK update kernels
top-level all-batched group loop
resident all-X upload/release entry points
```

这会造成两个实际问题：

1. 新人很难判断哪个 kernel 是当前主路径。
2. 新增一个 GPU 实现时，容易继续往同一个文件里塞 env 和分支。

本轮已做的低风险整理首先是把 cross-edge CUDA 共享 descriptor 抽到：

```text
UNG/codes/src/gpu_cross_edge_common.cuh
```

现在 `UngGroupQueryDesc`、`UngGroupTileDesc`、`UngSourceQueryDesc`、`UngSourceTileDesc`、`UngTargetSegmentDesc` 不再藏在 `gpu_gemm_topk.cu` 文件开头。这个改动不改变行为，但给后续拆出 `gpu_cross_edge_xstream.cu`、`gpu_cross_edge_source_exact.cu`、`gpu_cross_edge_universal.cu` 建了一个可复用的公共头文件。

随后把 route-neutral workload planning 移到：

```text
UNG/codes/src/gpu_cross_edge_planning.cuh
```

这个文件集中放置 `CrossEdgeTargetWorkload`、`CrossEdgeBatchCapacity`、`plan_cross_edge_target_workload()` 和 `make_cross_edge_batch_capacity()`。它不属于某个具体 GPU backend，而是给 batched fused、double-buffer、后续 source/X streaming backend 复用同一套 LNG in-neighbor query count、target nx、bench filter、CPU tiny fallback 和 flat batch capacity 语义。

double-buffer 主路径 helper 则放在：

```text
UNG/codes/src/gpu_cross_edge_double_buffer.cuh
```

这个文件集中放置 double-buffer route、eligibility、slot 生命周期、chunk packing、enqueue、global-merge finalize 和双 slot chunk scheduler。它仍然被包含在 `gpu_gemm_topk.cu` 的匿名 namespace 内，并且 include 位置放在 `gpu_cross_edge_update_kernels.cuh` 之后，因为 enqueue helper 需要调用已经定义好的 norm/topK 初始化 CUDA kernels 和全局 device/host buffer。这次移动是结构重构，不改变 H2D/kernel/D2H 顺序、kernel 参数、writeback 语义或 fallback 条件。

X-streaming boundary 路径已经从主 `.cu` 文件移到：

```text
UNG/codes/src/gpu_cross_edge_x_streaming.cuh
```

这个文件只承载 `gpu_cross_groups_search_x_streaming()`。它的定位仍然是 100%x40 / resident all-X OOM 的可扩展性兜底路径，不是当前主性能路线。拆出它的价值是让主文件中 `gpu_cross_groups_search_all_batched()` 的主路径更容易阅读，同时避免 boundary route 和 paper fused 主路径混在一起。

source-centric exact negative/experimental 路径也已经隔离：

```text
UNG/codes/src/gpu_cross_edge_source_exact_kernels.cuh
UNG/codes/src/gpu_cross_edge_source_exact.cuh
```

前者只放 `ung_source_exact_topk_global_kernel()` 和 `ung_source_tf32_wmma_topk_global_kernel()`，后者只放 `build_cross_edges_generate_gpu_source_exact()`。这条路径保留为 checked negative ablation 和 future two-stage 设计参照，不作为默认性能路线。

cuVS per-group library baseline 已从主 `.cu` 文件移到：

```text
UNG/codes/src/gpu_cross_edge_cuvs_baseline.cuh
```

这个文件只放 `build_cross_edges_generate_cuvs_bruteforce()`。它仍固定 SearchQueue 写回，不参与 flat-id / id-vector output-boundary 优化，避免 baseline 混入论文主路径的输出优化。cuVS result buffer 的共享缓存和释放逻辑暂时仍留在 `gpu_gemm_topk.cu` 的公共缓存区域。

SGEMM/cuBLASLt/naive-dot + separate-topK baseline 的 tile 执行细节已从主 group loop 移到：

```text
UNG/codes/src/gpu_cross_edge_sgemm_baseline.cuh
```

这个 helper 只承载 baseline 执行：dot tile 规划、cuBLAS SGEMM strided batch、cuBLASLt TF32 matmul、naive CUDA dot、tail GEMV/GEMM 和 `update_topk_from_dot_*`。它不改变 `UNG_GPU_TOPK_IMPL=2` 的语义，也不参与 paper fused 路径；主循环现在只负责决定某个 group 是否 fallback 到 separate-topK baseline。

norm/topK update utility kernels 已从主 `.cu` 文件移到：

```text
UNG/codes/src/gpu_cross_edge_update_kernels.cuh
```

这个 helper 只放 `l2_norm_sq_kernel()`、`init_topk_kernel()`、`topk_insert_linear()` 和 SGEMM baseline 使用的 `update_topk_from_dot_*` kernels。它被包含在 `gpu_gemm_topk.cu` 的匿名 namespace 内，位置早于 double-buffer 和 SGEMM baseline helper，因此保留原来的 internal-linkage 和调用顺序。它不是新算法，只是把“多个 route 共用的基础 CUDA 更新操作”从顶层路由文件里剥离出来。

resident all-X upload/release member functions 已从主 `.cu` 文件移到：

```text
UNG/codes/src/gpu_cross_edge_resident_vectors.cuh
```

这个 helper 只放 `gpu_prepare_all_vectors_on_device()`、`gpu_prepare_all_vectors_for_cross_edge()` 和 `gpu_release_all_vectors_on_device()`。它负责连续内存探测、可选 `cudaHostRegister` / pageable direct copy、pinned staging fallback、resident all-X device allocation、一次性 all-X norm 计算和释放共享 buffer。host-register/direct-pageable/hostreg 上限已经收敛到 `CrossEdgeGpuRuntimeConfig::vector_upload`，source-exact、cuVS baseline 和 batched fused 都通过同一份 runtime config 进入 prepare_all，不再由 resident helper 自己读取 `UNG_GPU_PREPARE_*`。它仍在 `gpu_cross_edge_source_exact.cuh`、`gpu_cross_edge_cuvs_baseline.cuh` 和 X-streaming helper 之前 include，因此这些 backend 看到的公开 member 函数和共享缓存语义不变。

CPU tiny exact fallback 执行细节已从主 group loop 移到：

```text
UNG/codes/src/gpu_cross_edge_cpu_tiny.cuh
```

planning 仍由 `gpu_cross_edge_planning.cuh` 决定哪些 target group 满足 `cpu_tiny_*` 阈值；这个 helper 只执行已经选出的 tiny groups。它保留原来的 CPU exact scan：对每个 tiny target group，枚举 LNG in-neighbor group 中的 query 向量，扫描 target group 内全部 `nx` 个向量，维护线性 topK，并写回 `SearchQueue`。主 all-batched 函数现在只调用 `run_cross_edge_cpu_tiny_groups()`，不再内嵌 exact scan 和 host topK 细节。

regular all-batched 路径的 query packing / H2D upload 已从主函数移到：

```text
UNG/codes/src/gpu_cross_edge_query_pack.cuh
```

这个 helper 放 `CrossEdgeQueryUploadResult` 和 `pack_and_upload_cross_edge_queries()`。它负责把每个 target group 的 LNG in-neighbor vectors 展平成连续 Q rows，生成 `query_global_ids`、`query_target_index` 和可选 `query_target_offsets`，并按 route 选择 host Q H2D 或 qid H2D + device gather。主函数现在只消费 mapping vectors 和 `fill_ms/h2d_ms`，不再内嵌 packing loop 和 CUDA copy event 计时代码。

bucket fused / TF32 descriptor assembly 已从主函数移到：

```text
UNG/codes/src/gpu_cross_edge_bucket_descriptors.cuh
```

这个 helper 放 `CrossEdgeBucketDescriptors` 和 `build_cross_edge_bucket_descriptors()`。它只负责 host-side route bucket 标记和 POD descriptor 填充：small/medium group query descriptors、TF32 tile descriptors、TF32 group-prefix descriptors 和 tile prefix offsets。kernel launch 仍由 `gpu_cross_edge_bucket_fused.cuh` 负责，per-group immediate fallback 仍由 `gpu_cross_edge_per_group_fused.cuh` 决定。

regular all-batched 的 per-group dispatch loop 已从主函数移到：

```text
UNG/codes/src/gpu_cross_edge_group_dispatch.cuh
```

这个 helper 放 `CrossEdgeRegularWorkloadView`、`CrossEdgeRegularDispatchConfig`、`CrossEdgeRegularDeviceBuffers`、`CrossEdgeRegularDispatchCounters` 和 `dispatch_cross_edge_regular_groups()`。它对每个 target group 先判断 heavy-SGEMM fallback，再尝试 immediate fused route，最后落到 SGEMM/cuBLASLt/naive-dot + separate-topK baseline。主函数现在通过 workload/config/buffers/counters 四个对象进入 dispatch helper，不再把几十个 bool、threshold、buffer pointer 和 counter 逐层散传；这一步是接口收敛，不改变 route 语义。

regular all-batched 的 route/config assembly 已从主函数移到：

```text
UNG/codes/src/gpu_cross_edge_regular_route.cuh
```

这个 helper 放 `CrossEdgeRegularRouteConfig`、`make_cross_edge_regular_route_config()`、`compute_cross_edge_direct_qid_all_effective()`、`make_cross_edge_separate_topk_config()`、`make_cross_edge_regular_dispatch_config()` 和 `log_cross_edge_regular_route_config()`。它把 query gather、singleton fastpath、direct-qid fused、global merge/writeback、GEMM backend、naive CUDA tuning、heavy-SGEMM fallback、small/medium/large fused thresholds 和 TF32 group path 从主 `.cu` 文件中移出，并显式消费 `CrossEdgeGpuRuntimeConfig&`，不再保留未传 route 时的 env fallback。主函数现在只创建一次 regular route config，再派生 packing、dispatch、separate-topK 和 profiling 所需的小配置。

普通 batched 路径的 D2H 和 host writeback 已从主函数移到：

```text
UNG/codes/src/gpu_cross_edge_output.cuh
```

`CrossEdgeTopkOutputWriter` 现在放在 `gpu_cross_edge_common.cuh`，regular output helper 和 double-buffer global-merge finalize 共用同一份 writer contract。`gpu_cross_edge_output.cuh` 放置 `CrossEdgeTopkOutputFinalizeResult` 和 `finalize_cross_edge_topk_output()`，负责从 local/global topK device buffer D2H，并按单一 writer mode 写回 flat-id、id-vector 或 SearchQueue，同时保留 `valid_topk_pairs` 可选统计和 `gpu_d2h_ms/writeback_ms` timing。non-double-buffer regular path 的调用点不再传 `flat_id_writeback` / `id_vector_writeback` 布尔组合，而是构造 writer 描述输出格式和目标容器。double-buffer global-merge 的最终 D2H/writeback 也已经改为消费同一个 writer 描述。CPU-side additional-edge coverage 和 graph materialization 现在通过 `CrossEdgeHostOutputView` 消费同一组 host 输出，`CrossEdgeGraphMaterializationWriter` 已提升到 `ung_cross_edge_output_writer.{h,cpp}`，集中负责把这些输出追加到 `_graph->neighbors`，并可从同一 view 生成 CSR adjacency buffers；CSR graph-search backend adapter 已进入 `ung_graph_search_backend.{h,cpp}`，`search_UNG_index --graph_search_backend csr` 可显式启用。

bucket fused descriptor launch 已从主函数移到：

```text
UNG/codes/src/gpu_cross_edge_bucket_fused.cuh
```

这个 helper 放置 `launch_cross_edge_bucket_fused_groups()`，只负责把已构造好的 small/medium/TF32 descriptors 拷到 device 并启动对应 fused kernels。descriptor 构造、route 选择、counter 统计和输出仍留在调用方，因此它是执行细节拆分，不改变 workload routing 或图语义。

可选 GPU topK 校验逻辑已从主函数移到：

```text
UNG/codes/src/gpu_cross_edge_verify.cuh
```

这个 helper 放置 `verify_cross_edge_topk_output()`，只在 `CrossEdgeGpuRuntimeConfig::diagnostics.verify_samples>0` 时执行。它用 CPU L2 exact scan 对 global-merge 或 local-row topK 输出做抽样校验，并保留 `diagnostics.verify_strict` 严格失败语义。`UNG_GEMM_VERIFY_*` 只在 `ung_cross_edge_config.cpp` 解析；verify helper 是 diagnostic/debug 路径，不参与正常构建热路径。

普通 batched 路径的 profiling summary 已从主函数移到：

```text
UNG/codes/src/gpu_cross_edge_profile.cuh
```

这个 helper 放置 `CrossEdgeRegularProfileSummary` 和 `log_cross_edge_regular_profile_summary()`，集中维护 `topk_valid_pairs`、fused group counters、writeback mode、global-merge、heavy-SGEMM、CPU-tiny 和 stage timing 的 profile key。它不改变日志字段，只把主函数尾部的日志格式集中到一个文件。

同时，cross-edge 构建结果已经有了正式 lightweight result 类型：

```text
UNG/codes/include/ung_cross_edge_result.h
```

`CrossEdgeBuildResult` 目前承载：

```text
CrossEdgeBuildTiming:
  generate_ms
  additional_ms
  add_offset_ms
  merge_cross_ms
  merge_additional_ms
  gpu_h2d_ms / gpu_kernel_ms / gpu_d2h_ms

active_writeback_mode
route_summary
flat_id_items
gpu_backend_used / gpu_fallback_used
additional_direct_append
```

这不是性能优化，而是把原来散落在主函数里的 `gen_ms/add_ms/...` 临时变量收成一个明确的构建结果对象。后续拆 `CrossEdgeBuildRequest` / backend adapter 时，可以让 CPU exact、cuVS、SGEMM+topK、fused/universal 等路径都填同一个 result，避免每个 backend 自己发明一套 timing 和状态字段。

最新接口推进：`build_cross_edges_generate_gpu_optimized()`、`build_cross_edges_generate_cuvs_bruteforce()`、`build_cross_edges_generate_gpu_source_exact()` 这三个 cross-edge backend 私有函数已经从 `bool + double *h2d/kernel/d2h` 改成 `bool + CrossEdgeBuildTiming&`，其中 cuVS 和 source-exact 也接收 `const CrossEdgeGpuRuntimeConfig&`，避免 baseline/negative path 绕过统一配置。下一层 CUDA helper `gpu_cross_groups_search_all_batched()` / `gpu_cross_groups_search_x_streaming()` 也已经从 optional `double*` 三指针改成 optional `CrossEdgeBuildTiming*`，并且现在必须显式接收 `const CrossEdgeGpuRuntimeConfig&`；调用点需要显式传入 id-vector / flat-id 输出指针和 timing/route，不再依赖默认 `nullptr` 推导实际 writeback 形态。全量向量上传仍由底层 `gpu_prepare_all_vectors_on_device()` 执行，但 cross-edge 路径现在通过 `gpu_prepare_all_vectors_for_cross_edge(CrossEdgeBuildTiming&, const CrossEdgeGpuRuntimeConfig&)` 进入，不再直接操作裸 H2D 指针，也不在 resident helper 内直接读取 `UNG_GPU_PREPARE_*`。

`gpu_cross_groups_search_all_batched()` 的入口 workload 统计已经从执行函数中拆出为 `plan_cross_edge_target_workload()`，并放在 `gpu_cross_edge_planning.cuh`。原来散落的 `target_counts_raw`、`target_counts`、`target_group_nx`、`cpu_tiny_group_indices`、`bench_skipped_groups`、`cpu_tiny_*` 和 `total_queries_sz` 现在由 `CrossEdgeTargetWorkload` 一次性返回；执行函数只消费 `gpu_query_counts`、`group_nx`、CPU tiny 分流结果和 bench filter 统计。这个改动不改变 cross-edge 语义，但把“统计 LNG in-neighbor query workload”和“执行 GPU 搜索/writeback”拆成了两个阶段。

double-buffer route 的设置和进入条件也已从执行函数中拆出，并且已经从主 `.cu` 文件移入 `gpu_cross_edge_double_buffer.cuh`：`CrossEdgeDoubleBufferRoute` 表达 request、DB no-split、device-gather、unsupported split、global merge、chunk size、medium/large nx 阈值和 warp 数；`evaluate_cross_edge_double_buffer_eligibility()` 负责判断全量向量是否已准备、是否存在 CPU tiny 分流、topK/dim 是否支持，以及当前 batch 中有多少 target group 超出 `large_max_nx`。主函数只根据 `can_enter`、`all_groups_supported` 和 `unsupported_groups` 决定进入 double-buffer、split unsupported groups，或回退旧路径。

double-buffer slot 的资源生命周期也已从主函数中移出：`CrossEdgeDoubleBufferSlot` 统一管理 stream/event、pinned host buffer、device buffer、group/tile descriptor buffer 和 query id 映射；`init/free/ensure_*` helper 负责初始化、释放和扩容。`finish_cross_edge_double_buffer_slot()` 进一步收拢 slot completion、event timing 和非 global-merge 的 SearchQueue 写回。`pack_cross_edge_double_buffer_chunk()` 把 chunk 需求统计、slot 扩容、query id/target offset packing、group/tile descriptor 构造收进 `CrossEdgeDoubleBufferChunkPack`；`enqueue_cross_edge_double_buffer_chunk()` 收拢 H2D、topK 初始化、medium/large kernel launch、可选 global merge、D2H 和 done event；`finalize_cross_edge_double_buffer_global_merge()` 收拢 global-merge 最终 D2H 和 flat-id / id-vector / SearchQueue 三种 host writeback。`run_cross_edge_double_buffer_chunks()` 则负责按 `chunk_queries` 切分 target groups、轮转两个 slots，并按 finish-before-enqueue 顺序调度。

### 3.3 env fallback 已集中到 settings/config builder，但底层兼容层仍需继续收敛

顶层已经有 `UngBuildConfig`。cross-edge regular all-batched 路径已经把 legacy/env fallback 收敛到 `CrossEdgeGpuRuntimeConfig` 和 `ung_cross_edge_config.cpp`。`UNG_COUNT_VALID_PAIRS`、`UNG_GEMM_VERIFY_SAMPLES` 和 `UNG_GEMM_VERIFY_STRICT` 已进入 `CrossEdgeGpuRuntimeConfig::diagnostics`，`UNG_BENCH_*` 已进入 `CrossEdgeGpuRuntimeConfig::benchmark_filter`。`gpu_cross_groups_search_all_batched()` 和 `gpu_cross_groups_search_x_streaming()` 的主入口现在不再接受可空 route，也不再在 `gpu_gemm_topk.cu` 内直接读取 `UNG_UNIVERSAL_GPU`；universal、diagnostics、benchmark filter、vector upload、writeback 和 chunk/tile route 均来自调用方传入的 `CrossEdgeGpuRuntimeConfig`。进一步收紧后，`gpu_cross_edge_planning.cuh` 的 flat batch capacity / CPU tiny / benchmark filter 阈值、`gpu_cross_edge_double_buffer.cuh` 的 DB route 设置和 `gpu_cross_edge_regular_route.cuh` 的 regular route 设置也都显式消费 `const CrossEdgeGpuRuntimeConfig&`，不再保留“未传 route 时读 env”的 fallback。

现在更大的结构问题在 `uni_nav_graph.cpp`：它仍然是 build/search orchestration 大文件，但 build/group graph/output-boundary env fallback 已经移到 `include/ung_build_settings.h` / `src/ung_build_settings.cpp`，而不是散在执行循环里。当前已经收敛的配置包括：

```text
GroupGraphRoute               // group graph impl -> CPU/CUDA backend / adaptive / prune mode
GraphReserveSettings          // Graph output boundary reserve / auto enable / hard cap
CpuGroupGraphSettings         // CPU Vamana group graph complete / profile / large-group inner threading
TagoreGroupGraphSettings      // Tagore/FastGrnnd fallback / overlap / exact batch / fill
IntraGroupIdSettings          // group graph 内部 id/global id 兼容边界
CpuHybridCrossSettings        // CPU hybrid cross exact scan 阈值
GpuCrossLifecycleSettings     // GPU cross-edge 后是否释放缓存
AdditionalEdgesSettings       // additional_edges 的 direct append / hybrid scan 边界
```

主要 settings 会在运行时输出 summary，同时写入 profile log：

```text
[graph_reserve] config ...
[group_graph] cpu_config ...
[TagoreCuda] config ...
[PROF] graph.reserve_config ...
[PROF] group_graph.cpu_config ...
[PROF] tagore.config ...
```

这些日志不是性能方法，而是可审计边界：同一份实验结果可以从 log 中确认 small-group complete 阈值、large-group inner threading、Tagore fallback、overlap fallback、exact-batch 和 fill 策略。

静态扫描目前显示，`uni_nav_graph.cpp` 已经没有 `read_env_*` / `getenv()` 调用，也不再手写 `UngGroupGraphImpl` 到 CUDA/adaptive/prune-mode 的重复枚举判断；这批读取和 route 语义只保留在 `ung_build_settings.cpp` 内部。执行函数消费 settings/route 对象，不再直接读裸 env。这一步解决的是“实验开关隐藏在深层执行代码里”的问题。剩余问题是这些 settings 仍是 build settings 模块级接口，尚未提升为正式 `GroupGraphRouterConfig` / `CrossEdgeBuildOrchestrator` 级接口，因此还不能说已经完全 backend 插拔化。

Tagore/FastGrnnd 路径已新增 `TagoreGroupBuildContext`，把 `GroupGraphRoute`、`TagoreGroupGraphSettings`、`TagoreCudaRuntimeConfig`、`TagorePruneMode` 和 exact-batch threshold 收敛成一次 build 的上下文。Group partition 已抽到 `partition_tagore_groups()`，返回 `TagoreGroupPartition`，封装 fallback group ids、GPU group ids、Tagore requests 和 GPU path 总点数。Fallback group build 已抽到 `build_tagore_fallback_groups()`，返回 `TagoreFallbackBuildStats`，主函数不再维护 fallback 的 atomic counters 和 complete/CPU Vamana fallback 细节。Exact/GNN batch split 已抽到 `build_tagore_batch_artifacts()`，返回 `TagoreBatchBuildArtifacts`，封装 batch results、packed graph source、batch timing 和 wall time。CPU-compatible result fill/writeback 也已抽到 `fill_tagore_batch_results()`。这些 helper 现在消费同一个 `TagoreGroupBuildContext`，避免 fallback、batch split、CUDA fast-exact 和 fill/writeback 各自散传阈值或 runtime config；主函数只保留 context 创建、fallback 调度、GPU batch 调用、fill 调用和最终 profile 汇总。Tagore batch builder 的方法选择只通过 `TagorePruneMode` 传递，不再保留 bool 兼容 overload。这不是新算法，但减少了 group graph backend 与输出物化细节的耦合。

### 3.4 存在 legacy env bridge

`build_cross_edges_generate_gpu_optimized()` 会根据 `UngGpuTopkImpl` 给旧开关设置默认值。这是为了兼容历史优化路径，短期可以保留，但必须明确它是：

```text
profile -> legacy CUDA env bridge
```

当前代码已经做了第一步收敛：`UNG/codes/include/ung_cross_edge_config.h` 和 `UNG/codes/src/ung_cross_edge_config.cpp` 单独承载这段配置逻辑。`apply_cross_edge_gpu_topk_env_defaults()` 负责 legacy env bridge，`CrossEdgeGpuRuntimeConfig` 负责解析 source-exact、universal、DB no-split、double-buffer、X-streaming boundary、query device-gather、global merge、CPU tiny fallback、singleton fastpath、direct-qid fused、GEMM/cuBLAS fallback、naive CUDA micro-tuning、group fused kernel shape、writeback、flat batch capacity、resident vector upload、diagnostics 以及 benchmark filter，并已经作为显式参数从 `build_cross_group_edges()` 传入 `build_cross_edges_generate_gpu_optimized()`；source-exact、cuVS baseline、resident vector upload、batched fused 和 X-streaming boundary 都开始消费同一份 config。`gpu_cross_groups_search_all_batched()` 和 `gpu_cross_groups_search_x_streaming()` 现在显式接收 `const CrossEdgeGpuRuntimeConfig&`，主入口不再有“未传 route 时直接读 env”的行为；planning、double-buffer route 和 regular route helper 也已改成 reference config。`CrossEdgeGpuRuntimeConfig` 也提供 `CrossEdgeGpuWritebackMode`、`route_name()`、`route_class_name()`、`validation_error()`、`writeback_mode_name()`、`summary()`、`needs_searchqueue_storage()`、`uses_id_vector_writeback()` 和 `uses_flat_id_writeback()` 这类语义 helper，避免主流程继续手写布尔组合或分散拼日志。

目前已经进入显式 config 的 cross-edge GPU 参数包括：

| 字段 | 原 env | 含义 |
|---|---|---|
| `source_exact` | `UNG_GPU_SOURCE_EXACT` | 是否走 source-centric exact 实验路径 |
| `source_exact_mode` / `source_exact_warps` / `source_exact_pad_groups` | `UNG_GPU_SOURCE_EXACT_MODE` / `UNG_GPU_SOURCE_EXACT_WARPS` / `UNG_GPU_SOURCE_EXACT_PAD_GROUPS` | source-centric negative/experimental 路径的 CUDA-core/WMMA、warp 和 padding 配置 |
| `universal_route` | `UNG_UNIVERSAL_GPU` | 是否启用 universal flat route |
| `flat_q_cap_mb` / `flat_out_cap_mb` | `UNG_GPU_FLAT_Q_CAP_MB` / `UNG_GPU_FLAT_OUT_CAP_MB` | flat batch 的 Q/output 容量上限；CUDA 侧通过 `CrossEdgeBatchCapacity` 计算是否需要 split |
| `db_nosplit` | `UNG_GPU_DB_NOSPLIT` | double-buffer 是否不按旧 flat cap 分裂 |
| `double_buffer` | `UNG_GPU_DOUBLE_BUFFER` | 是否启用双缓冲 direct-qid 路径 |
| `x_streaming` | `UNG_GPU_X_STREAMING` | 是否走 X-side streaming boundary 路径 |
| `query_upload_device_gather` | `UNG_Q_UPLOAD_MODE` | Q 上传用 qid + device gather，还是 host pack Q |
| `global_merge` | `UNG_GPU_GLOBAL_MERGE` | 是否在 GPU 侧合并同一 qid 的跨组 topK |
| `global_merge_direct` | `UNG_GPU_GLOBAL_MERGE_DIRECT` | 是否使用 direct global merge kernel |
| `global_merge_direct_max_nx` | `UNG_GPU_GLOBAL_MERGE_DIRECT_MAX_NX` | direct merge 适用的最大 target group size |
| `db_global_merge` | `UNG_GPU_DB_GLOBAL_MERGE` | double-buffer 子路径是否强制 GPU global merge |
| `db_split_unsupported` | `UNG_GPU_DB_SPLIT_UNSUPPORTED` | double-buffer 遇到 unsupported group 时是否拆分 supported/fallback |
| `x_stream_*_chunk_mb` | `UNG_GPU_X_CHUNK_MB` / `UNG_GPU_X_STREAM_Q_MB` / `UNG_GPU_X_STREAM_OUT_MB` | X-streaming boundary 路径的 X/Q/output 分块上限 |
| `cpu_tiny_groups` / thresholds | `UNG_CPU_TINY_*` | 是否把极小工作量 group 回退 CPU exact |
| `singleton_fastpath` / `singleton_threads` | `UNG_SINGLETON_*` | nx=1 的专用写回/计算路径 |
| `direct_qid_fused` / `direct_qid_all_fused` | `UNG_DIRECT_QID_*` | fused kernel 直接消费 qid 而不是 materialized Q |
| `force_custom_kernel` / `gemm_impl_request` | `UNG_FORCE_CUSTOM_KERNEL` / `UNG_GEMM_IMPL` | 自研 fused、cuBLAS strided-batched、cuBLASLt/naive CUDA 的选择 |
| `naive_*` tuning | `UNG_NAIVE_*` | naive CUDA dot/topK 的 block、shared memory、vec4、heavy-SGEMM fallback 配置 |
| `topk_block_threads` | `UNG_TOPK_BLOCK_THREADS` | 独立 topK kernel 的线程数；`-1` 表示按 `topk` 动态默认 |
| `tf32_group_prefix` / `tf32_group_2d` | `UNG_TF32_GROUP_*` | groupGEMM fused topK 的 TF32/WMMA 相关路径 |
| `small/medium/large_group_fused` | `UNG_SMALL_GROUP_FUSED` 等 | small / medium / large fused kernel 是否启用 |
| `small/medium/large_group_*` thresholds | `UNG_*_GROUP_*` | fused kernel 的 nx 分界、warp 数和模式 |
| `gather_threads` / `gather_blocks` | `UNG_GATHER_THREADS` / `UNG_GATHER_BLOCKS` | device gather kernel 配置 |
| `db_chunk_queries` | `UNG_GPU_DB_CHUNK_QUERIES` | double-buffer 每 chunk query 上限 |
| `db_medium_max_nx` / `db_large_max_nx` | `UNG_GPU_DB_MEDIUM_MAX_NX` / `UNG_GPU_DB_LARGE_MAX_NX` | medium fused 与 large tiled kernel 的 nx 分界 |
| `db_medium_group_warps` / `db_large_group_warps` | `UNG_MEDIUM_GROUP_WARPS` / `UNG_LARGE_GROUP_WARPS` | fused/tiled kernel warp 配置 |
| `id_vector_writeback` / `flat_id_writeback` | `UNG_GPU_ID_VECTOR_WRITEBACK` / `UNG_GPU_FLAT_ID_WRITEBACK` | 输出写回格式选择 |

主流程现在用单一 `CrossEdgeGpuWritebackMode active_writeback_mode` 表示实际写回来源，而不是同时维护 `id_vector_active` 和 `flat_id_active` 两个布尔值。`cuVS per-group` baseline 固定为 `SearchQueue` 写回，即使全局 route 请求 flat/id 输出，也不会把 output-boundary 优化混入 cuVS baseline。

## 4. 方法边界审计

| 模块 | 方法 | 当前分类 | 是否应保留 | 主要入口 |
|---|---|---|---|---|
| profile | `original_cpu` | baseline | 保留 | `UngBuildConfig::from_env()` |
| profile | `current_cpu` | baseline | 保留 | `UngBuildConfig::from_env()` |
| profile | `naive_gpu` | baseline | 保留 | `UngBuildConfig::from_env()` |
| profile | `paper_fused` | main/paper | 保留 | `UngBuildConfig::from_env()` |
| group graph | CPU Vamana | quality baseline/fallback | 保留 | `build_graph_for_all_groups()` |
| group graph | TagoreCuda | ablation | 保留 | `build_graph_for_all_groups_tagore_cuda()` |
| group graph | GrnndLikeCuda | ablation | 保留 | `build_graph_for_all_groups_tagore_cuda()` |
| group graph | FastGrnndCuda | main component | 保留 | `build_tagore_vamana_cuda_batch(..., FastGrnnd)` |
| group graph | AdaptiveCuda | current router | 保留 | `build_graph_for_all_groups_tagore_cuda()` |
| cross-edge | CPU Vamana | weak baseline | 保留 | `build_cross_edges_generate_cpu_baseline()` |
| cross-edge | CPU exact scan | strong CPU baseline | 保留 | `build_cross_edges_generate_cpu_exact_scan()` |
| cross-edge | CPU hybrid | diagnostic | 保留但不要当主结果 | `build_cross_edges_generate_cpu_hybrid_scan_vamana()` |
| cross-edge | cuVS per-group | library baseline | 保留 | `build_cross_edges_generate_cuvs_bruteforce()` |
| cross-edge | SGEMM + topK | naive GPU baseline | 保留 | route 在 `gpu_cross_groups_search_all_batched()`，tile 执行在 `gpu_cross_edge_sgemm_baseline.cuh` |
| cross-edge | grouped fused topK | paper/main | 保留 | `gpu_cross_groups_search_all_batched()` |
| cross-edge | source-centric exact | negative/experimental | 可保留但应隔离 | `build_cross_edges_generate_gpu_source_exact()` |
| cross-edge | X-side streaming | boundary/scalability | 保留但不能当主加速 | `gpu_cross_groups_search_x_streaming()` |
| query entry | CPU scan | baseline | 保留 | `query_entry_group_bench.cu` |
| query entry | CPU exact minimal | quality/minimal baseline | 保留 | `query_entry_group_bench.cu` |
| query entry | GPU correct-cover | main candidate | 需要接入正式 search | `gpu_cover_frontier` provider |

## 5. 推荐的代码重构边界

### 5.1 低风险第一步

不移动 CUDA kernel，只先建立 adapter 和配置边界：

```text
UngBuildConfig
  -> GroupGraphRouterConfig
  -> CrossEdgeRouterConfig
  -> QueryEntryConfig
```

目标是让深层函数不再直接读取主要 env，而是通过 config 参数知道：

```text
我是 main path / baseline / diagnostic / boundary path
应该用哪种 writeback
是否允许 fallback
是否要求 strict
```

### 5.2 第二步拆文件

建议拆分为：

```text
UNG/codes/src/ung_group_graph_router.cpp
UNG/codes/src/ung_group_graph_cpu.cpp
UNG/codes/src/ung_cross_edge_router.cpp
UNG/codes/src/ung_cross_edge_cpu.cpp
UNG/codes/src/gpu_cross_edge_universal.cu
UNG/codes/src/gpu_cross_edge_xstream.cu
UNG/codes/src/gpu_cross_edge_cuvs.cu
UNG/codes/src/query_entry_gpu.cu
```

拆分原则：

```text
router 只做选择，不做 kernel 细节。
backend 只做一种实现，不读取顶层 profile。
output writer 独立出来，统一 SearchQueue / id-vector / flat-id / CSR 写回。
```

### 5.3 第三步定义正式 backend 接口

建议引入三个轻量接口：

```cpp
struct GroupGraphBuildRequest;
struct GroupGraphBuildResult;
struct CrossEdgeBuildRequest;
struct CrossEdgeBuildResult;
struct QueryEntryRequest;
struct QueryEntryResult;
```

每个 result 都必须带：

```text
semantic_level: exact / coverage_correct / approximate / diagnostic
timing breakdown
output format
fallback_used
```

这样后续论文实验可以自动检查：不能把 diagnostic/boundary path 混进 main result。

## 6. 哪些现在不应该删

不要因为文件乱就立即删除这些路径：

| 路径 | 原因 |
|---|---|
| CPU Vamana | 质量 baseline 和 fallback，仍是最重要对照 |
| CPU exact scan | 强 CPU baseline，防止只和弱 baseline 比 |
| cuVS per-group | 库 baseline，证明逐组调用 API 边界 |
| SGEMM + topK | naive GPU baseline，证明 fused topK 的贡献 |
| TagoreCuda / GrnndLikeCuda | 方法演进和 ablation，需要解释为什么最终不用 |
| X-streaming | 100%x40 OOM 边界修复，虽然不是主加速 |

应清理的是“它们在主路径中的位置和命名”，不是把 baseline 消失。

## 7. 面向导师的判断

如果导师问“现在代码是否足够简洁清晰”，我会回答：

```text
还不够。

我们已经把实验方法收敛到了可选择的 profile 和 enum，文档也开始区分主结果、baseline 和负结果；
但实现层仍然把 router、backend、fallback、profiling、legacy env bridge 和输出写回混在少数大文件里。

当前仓库适合继续跑实验和补论文证据，但如果要让工作变成长期可维护的系统，需要先做一次接口化重构：
把 group graph、cross-edge、query entry 三条线分别拆成 router + backend + output writer。
```

## 8. 下一轮最小整改清单

| 优先级 | 任务 | 验收标准 |
|---:|---|---|
| P0 | 建立 method registry，并让 docs/README 指向它 | 新人能从一个表看懂所有方法定位 |
| P0 | 给 `build_cross_edges_generate_gpu_optimized()` 标注为 legacy env bridge/router adapter | 已完成：`ung_cross_edge_config.{h,cpp}` 承载 typed runtime config，函数入口已用注释明确为 GPU cross-edge routing adapter / legacy env bridge，新 kernel 应接入 `gpu_cross_edge_*.cuh` helper 而不是在这里继续堆 env parsing 或算法策略 |
| P0 | 给 X-streaming/source-centric 标注 boundary/experimental | 不再被误写成主结果 |
| P1 | 抽 `CrossEdgeGpuConfig`，替代部分深层 env | 已完成 runtime route 第一层，并传入 CUDA batched search / X-streaming 的 universal / DB no-split / double-buffer / X-streaming chunk / flat batch cap / query gather / global merge / DB chunk / CPU tiny / singleton / direct-qid / GEMM fallback / naive tuning / fused kernel shape 参数；`CrossEdgeBuildResult` 已把主流程 timing/writeback/fallback 状态收拢；backend 私有函数已改为接收 `CrossEdgeBuildTiming&`，CUDA cross search helper 已改为接收 `CrossEdgeBuildTiming*`；double-buffer route、eligibility 和 slot 生命周期已有独立 helper；`CrossEdgeGpuRuntimeConfig` 已拆出 vector upload / diagnostics / debug / benchmark filter 子配置，`UNG_MEDIUM_GROUP_SYNC_DEBUG` 与 `UNG_NAIVE_DEBUG_DOT` 不再混在 kernel flags 顶层；下一步是把 main/baseline/diagnostic/boundary config 进一步对象化 |
| P1 | 抽 `GroupGraphRouterConfig`，把 small/medium/large 阈值集中 | 已先完成模块化收敛：`GroupGraphRoute`、`GraphReserveSettings`、`CpuGroupGraphSettings`、`TagoreGroupGraphSettings`、`IntraGroupIdSettings`、`CpuHybridCrossSettings`、`GpuCrossLifecycleSettings`、`AdditionalEdgesSettings` 已移到 `ung_build_settings.{h,cpp}`，并补了 `[graph_reserve] config`、`[group_graph] cpu_config`、`[TagoreCuda] config` summary 打印；CPU 顶层入口已通过 `GroupGraphBuildContext` 同时消费 route 和 CPU settings，CUDA 路径继续通过 `TagoreGroupBuildContext` 消费 route / Tagore settings / runtime config / exact-batch threshold；无调用的单组 Tagore fill helper 已删除。下一步是把 `GroupGraphBuildContext` 和 `TagoreGroupBuildContext` 统一成正式 backend config / result contract |
| P1 | 把 query-entry GPU provider 接入正式 search backend | 阶段 benchmark 变成端到端实验 |
| P2 | 拆 `gpu_gemm_topk.cu` | 已抽出 `gpu_cross_edge_common.cuh`、`gpu_cross_edge_planning.cuh`、`gpu_cross_edge_buffers.cuh`、`gpu_cross_edge_topk_device.cuh`、`gpu_cross_edge_update_kernels.cuh`、`gpu_cross_edge_resident_vectors.cuh`、`gpu_cross_edge_cpu_tiny.cuh`、`gpu_cross_edge_query_pack.cuh`、`gpu_cross_edge_bucket_descriptors.cuh`、`gpu_cross_edge_group_dispatch.cuh`、`gpu_cross_edge_regular_route.cuh`、`gpu_cross_edge_regular_backend.cuh`、`gpu_cross_edge_singleton_kernels.cuh`、`gpu_cross_edge_large_group_kernels.cuh`、`gpu_cross_edge_descriptor_kernels.cuh`、`gpu_cross_edge_group_fused_kernels.cuh`、`gpu_cross_edge_double_buffer.cuh`、`gpu_cross_edge_sgemm_baseline.cuh`、`gpu_cross_edge_bucket_fused.cuh`、`gpu_cross_edge_per_group_fused.cuh`、`gpu_cross_edge_output.cuh`、`gpu_cross_edge_profile.cuh`、`gpu_cross_edge_verify.cuh`、`gpu_cross_edge_x_streaming.cuh`、`gpu_cross_edge_source_exact*.cuh` 和 `gpu_cross_edge_cuvs_baseline.cuh`；batched helper 的入口 workload planning、GPU 复用缓存/handle、topK device primitive、norm/topK update kernels、resident all-X upload/release、CPU tiny exact fallback、query packing/upload、bucket descriptor assembly、per-group dispatch、regular route/config builder、regular backend lifecycle adapter、singleton fastpath kernels、large-group fused kernels、descriptor-batched TF32 kernels、small/medium/group-desc fused kernels、DB route/eligibility、slot、chunk pack/enqueue/finalize/schedule、SGEMM/separate-topK baseline、bucket fused descriptor launch、per-group fused kernel launch、普通 batched D2H/writeback、profiling summary、optional verify、X-streaming boundary path、source-exact experimental path、cuVS library baseline 已离开主 `.cu` 文件；regular dispatch 已从长参数列表收敛为 workload/config/buffers/counters 四类对象，route/config helper 已改成显式 `CrossEdgeGpuRuntimeConfig&`，global merge init / singleton prepass / regular global merge finalize、bucket descriptor wrapper、bucket fused launch wrapper 和 regular profile 填充已收进 `gpu_cross_edge_regular_backend.cuh`。下一步是把 remaining regular backend 输入输出进一步对象化，而不是在主入口继续传散参数 |
| P2 | 抽 flat/CSR output writer | 已完成第一步：`CrossEdgeTopkOutputWriter` 统一表达 CUDA D2H/writeback 的 SearchQueue / id-vector / flat-id 三种 host 输出，non-double-buffer regular path 和 double-buffer global-merge finalize 都已消费这个 writer；`CrossEdgeHostOutputView` 统一表达 CPU-side additional coverage 检查和 graph materialization 输入；`CrossEdgeGraphMaterializationWriter` 已提升到 `ung_cross_edge_output_writer.{h,cpp}` 并集中写入 `_graph->neighbors`，也能生成 `CrossEdgeCsrOutput` adjacency buffers；search 侧已新增 `GraphSearchBackend` / `GraphNeighborView` 作为邻接访问边界，并支持从 `CrossEdgeCsrOutput` 读取邻接；`search_UNG_index --graph_search_backend csr` 可显式走 CSR-backed search |
