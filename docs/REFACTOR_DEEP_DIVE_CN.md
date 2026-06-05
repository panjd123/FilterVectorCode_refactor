# FilterVectorCode_refactor 深度交接文档（中文）

本文面向“下一位接手优化的人”，目标是做到：

1. 不看代码也能先理解端到端在做什么。
2. 看代码时知道每个核心文件的职责、输入输出、性能瓶颈和正确性约束。
3. 能快速定位“目前写法是什么、已经怎么优化过、下一步该怎么优化”。

---

## 1. 项目端到端在做什么

这个项目实现的是 **Filtered ANNS**（带属性过滤的近邻检索），主线是 UNG，支持与 ACORN 结合。

- 构建阶段（Build）：把基础向量和标签构造成可搜索索引。
- 搜索阶段（Search）：给定查询向量 + 查询标签，返回满足标签过滤条件的近邻。

在 `FilterVectorCode_refactor` 中，关键优化主线集中在 **UNG 构建阶段**，尤其是：

- LNG 构建及派生统计（descendants / coverage）
- 跨组边构建（cross-group edges）中的 GPU 路径
- 组内图构建（group graph）的 workload-aware GPU/CPU router
- GPU 结果回填到 CPU-compatible graph 的 output boundary

当前最新主线快照：

| 模块 | 当前实现定位 | 关键证据 / 边界 |
|---|---|---|
| cross-edge | `UNG_UNIVERSAL_GPU=1` universal flat double-buffer：target-centric descriptor batching + double-buffer + GPU global merge + flat-id output | SIFT30 skip-additional cross `3180.63 ms`；Amazon 1% x200 full-quality cross `2494.21 ms`、L1000/L5000 `0.871/0.911`；Amazon 1% x100 是 boundary result，不能写成无条件端到端加速 |
| source-centric | no-lock 遍历方向保留为 future direction | 修复 id-only null-distance 写入和 stream error check 后，SIFT30 source CUDA-core cross `5926.67 ms`，慢于 universal；后续应做 two-stage source grouped GEMM + reduce |
| group graph | `UNG_GROUP_GRAPH_IMPL=4` adaptive route / `3` FastGrnndCuda：小组 CPU/bounded fallback，中组 packed exact-anchor，大组 FastGrnndCuda/reverse-tail | packed exact-anchor x200 group `3827.28 ms`、L1000/L5000 `0.869/0.908`；x400 reverse-tail+repair 是质量增强 Pareto 点；不能写成 CPU Vamana 的无损普遍替代 |
| output boundary | `NeighborList64`、reserve、direct-H2D 等已降低 host allocator/pack/fill 成本 | 仍保留 CPU-compatible `Graph::neighbors` / `SearchQueue` / Vamana 语义；flat adjacency/CSR/GraphView 还未完成 |

---

## 2. 端到端执行链路（脚本到核心函数）

### 2.1 实验脚本链路

- 顶层实验脚本：`exp.sh`
- 构建脚本：`build_hybrid.sh`
- 可执行入口（UNG 构建）：`UNG/codes/apps/build_UNG_index.cpp`
- 可执行入口（UNG 搜索）：`UNG/codes/apps/search_UNG_index.cpp`

典型构建调用链：

1. `exp.sh` 读取 `experiments.json`，逐实验循环。
2. 调用 `build_hybrid.sh`，决定 `parallel / serial / ung_only / acorn_only`。
3. `build_hybrid.sh` 运行 `build_UNG_index`。
4. `build_UNG_index.cpp` 调 `UniNavGraph::build(...)`。
5. `UniNavGraph::save(...)` 把索引和统计文件落盘到 `index_files` / `results`。

### 2.2 构建主函数（UNG）

`UNG/codes/apps/build_UNG_index.cpp` 做三件事：

1. 加载 base 向量与标签（`Storage`）。
2. 调 `index.build(...)` 真正构建索引。
3. 调 `index.save(...)` 输出索引文件。

---

## 3. 核心数据模型（必须先理解）

### 3.1 关键概念

- 向量（vector）：高维 float 向量。
- 标签集（label set）：一个向量对应的一组离散标签。
- 组（group）：**标签集完全相同** 的向量归为一组。
- Trie：索引所有标签集，用于做超集/最小超集查询。
- LNG（Label Navigating Graph）：
  - 节点：group
  - 边：从 group 指向其最小超集 group
- UNG 图：
  - 组内图（intra-group，Vamana）
  - 跨组边（cross-group）

### 3.2 关键容器（`UniNavGraph` 内）

- `_group_id_to_vec_ids`：组 -> 向量ID列表
- `_group_id_to_label_set`：组 -> 标签集
- `_group_id_to_range`：组 -> 在“重排后向量数组”中的连续区间
- `_label_nav_graph->out_neighbors/in_neighbors`：LNG 邻接
- `_label_nav_graph->_lng_descendants`：每组后代组列表
- `_label_nav_graph->covered_sets`：每组覆盖的向量列表
- `_graph`：最终 UNG 邻接（包含组内边与跨组边）

---

## 4. 模块职责图（文件级）

| 模块 | 核心文件 | 职责 | 输入 | 输出 |
|---|---|---|---|---|
| 存储层 | `UNG/codes/include/storage.h`, `UNG/codes/src/storage.cpp` | 读写向量/标签，重排数据 | `.bin` + 标签txt | 连续内存向量与标签集 |
| 距离层 | `UNG/codes/include/distance.h`, `UNG/codes/src/distance.cpp` | 距离计算（当前主要 L2） | 两个向量地址 | float 距离 |
| 图容器 | `UNG/codes/include/graph.h` | 邻接表存储/读写 | 节点数 | 邻接表 |
| 搜索队列 | `UNG/codes/include/search_queue.h`, `UNG/codes/src/search_queue.cpp` | 有序 top-L 候选队列 | (id, distance) 流 | 有序候选集 |
| 搜索缓存 | `UNG/codes/include/search_cache.h` | 线程缓存池 | 线程数/容量 | 复用 SearchCache |
| 标签索引 | `UNG/codes/include/trie.h`, `UNG/codes/src/trie.cpp` | 插入标签集、查超集入口 | 标签集 | 候选 group 终止节点 |
| LNG容器 | `UNG/codes/include/label_nav_graph.h` | 存 LNG 结构与派生量 | group 总数 | out/in/desc/coverage |
| UNG主逻辑 | `UNG/codes/include/uni_nav_graph.h`, `UNG/codes/src/uni_nav_graph.cpp` | 构建、保存、加载、搜索 | base storage + 参数 | 完整索引 |
| GPU跨组边 | `UNG/codes/src/gpu_gemm_topk.cu` | 跨组 brute-force topk GPU 实现；包含 direct-qid、double-buffer、universal flat-id、source-centric 实验路径 | group 批次 + resident base vectors + qid descriptors | cross_group_neighbors / flat-id edge buffers |
| GPU组内图 | `UNG/codes/include/tagore_graph_builder.h`, `UNG/codes/src/tagore_graph_builder.cu` | Tagore/FastGrnndCuda/packed exact-anchor group graph 后端 | group storage views + route config | 每组邻接图或 packed graph buffer |
| 构建开关 | `UNG/codes/include/ung_build_config.h`, `UNG/codes/src/ung_build_config.cpp` | 统一读取 CPU/GPU/group graph/cross-edge 实现开关 | 环境变量 | `UngBuildConfig` |

---

## 5. 构建流程逐阶段解析（`UniNavGraph::build`）

以下是构建主流程中的关键阶段，按执行顺序说明“当前写法 + 已优化点 + 优化潜力”。

### 阶段 A：分组与 Trie 建立

函数：

- `build_trie_and_divide_groups()`

当前写法：

- 遍历全部向量标签集，插入 Trie。
- 标签集相同则复用已有 group_id，否则创建新 group_id。
- 输出 `_group_id_to_vec_ids` 和 `_group_id_to_label_set`。

已做优化：

- `TrieIndex::insert` 减少冗余 `resize` 开销（仅必要时扩容）。

潜力：

- Trie 内部结构目前是 `unordered_map<LabelType, shared_ptr<TrieNode>>`，可考虑：
  - 小分支节点改紧凑结构（small-vector / flat_hash_map）
  - 节点对象池减少小对象分配

### 阶段 B：数据重排与分组存储视图

函数：

- `prepare_group_storages_graphs()`

当前写法：

- 先按 group 顺序重排全部向量，使同组向量连续。
- 构建 `_group_id_to_range`，并创建每组 storage view + graph view。

作用：

- 为后续组内构图、跨组批处理提供连续内存与更好局部性。

潜力：

- `reorder_data` 目前逐向量 `memcpy`，可评估 NUMA 绑定和并行搬运策略。

### 阶段 C：每组 Vamana 组内图

函数：

- `build_graph_for_all_groups()`

当前写法：

- 默认 CPU baseline 是 `UNG_GROUP_GRAPH_IMPL=0`：`#pragma omp parallel for schedule(dynamic,1)`，小组 complete graph，大组 CPU Vamana。
- GPU/workload-aware route 通过 `UNG_GROUP_GRAPH_IMPL` 切换：
  - `1`：TagoreCuda，进程内直接调用 Tagore CUDA kernels；
  - `3`：FastGrnndCuda，Tagore GNN-Descent 候选 + reverse-augmented local pruning / light prune；
  - `4`：AdaptiveCuda，小组 CPU fallback，中组 packed exact-anchor，大组 FastGrnndCuda，并发 CPU fallback 与 GPU batch。

当前边界：

- 这些路径不是 CPU Vamana 的逐边等价替代，必须用 full-quality search recall A/B 证明。
- x200/x400 coverage-query 支持 packed exact-anchor 是强候选；x100 证明 conservative adaptive route 兼容；真实多标签和更多 x400 参数仍是缺口。

潜力：

- 组大小分布极不均衡时，动态调度仍会有尾部长任务；可引入分层队列或 work-stealing。
- 更大收益需要 flat adjacency/CSR，减少 GPU 图回填到 CPU per-node object 的成本。

### 阶段 D：向量-属性二分图

函数：

- `build_vector_and_attr_graph()`

当前写法：

- 建立 `label -> attr_id` 映射。
- 构建双向邻接：向量节点 <-> 属性节点。

作用：

- 搜索阶段可用于属性位图过滤（对比 UNG 覆盖集位图）。

### 阶段 E：LNG 主图构建

函数：

- `build_label_nav_graph()`
- 内部核心：`get_min_super_sets(...)`

当前写法：

- Phase1（并行）：对每个 group 求最小超集入口，写入 `out_neighbors`。
- Phase2（串行）：由 out 邻接反推 `in_neighbors`。

`get_min_super_sets` 当前关键实现：

- Trie 查询候选超集入口；
- 线程本地缓存复用（`thread_local`）减少反复分配；
- 候选标签集大小跨度较小时（阈值 256）走桶化遍历，避免全量排序；
- 跨度大时回退排序路径，保证稳定性。

已做优化：

- 线程本地容器复用 + 桶化路径，是 LNG 阶段的重要优化来源之一。

### 阶段 F：后代集合（descendants）

函数：

- `get_descendants_info()`

当前写法：

- 对每个 group 在 LNG 上 BFS，得到全部后代 group。
- 使用线程本地 `visited_epoch + queue + discovered` 复用，减少分配和清零成本。
- 输出：
  - `_lng_descendants_num`
  - `_lng_descendants`
  - `avg_descendants`

已做优化：

- 从哈希容器流转改为连续 `vector` 布局，显著降低构建时间。

### 阶段 G：覆盖集合（coverage）

函数：

- `cal_f_coverage_ratio()`

当前写法（双实现）：

- `UNG_COVERAGE_IMPL=1`：`descendants_direct`，直接按后代组聚合向量。
- `UNG_COVERAGE_IMPL=0`：legacy 拓扑合并，最后 `sort+unique` 去重。
- 线程数由 `UNG_COVERAGE_THREADS` 控制。

已做优化：

- `covered_sets` 改为连续向量容器，去掉大量哈希插入热区。

### 阶段 H：Roaring 位图初始化

函数：

- `initialize_roaring_bitsets()`

当前写法：

- 并行遍历 group。
- `addMany` 批量灌入 descendants / coverage。

已做优化：

- 从逐元素 `add` 变为 `addMany`，且并行化。

### 阶段 I：跨组边构建（主优化战场）

函数：

- `build_cross_group_edges()`

当前写法：

- 先生成每个 query 向量到目标组 top-k 近邻候选。
- 再补充“未连接 out-group”的附加边。
- 再把组内局部ID转全局ID并合并进 `_graph`。

可切换后端：

- `UNG_CROSS_EDGE_BACKEND=0`：CPU baseline
- `UNG_CROSS_EDGE_BACKEND=1`：GPU optimized
- `UNG_CROSS_EDGE_GPU_STRICT=1`：GPU失败不回退；`0` 则回退 CPU

---

## 6. 跨组边 GPU 路径：输入输出与计算细节

核心入口：

- `gpu_prepare_all_vectors_on_device(...)`
- `gpu_cross_groups_search_all_batched(...)`

### 6.1 算子在算什么

对于每个目标组 `X`，其入邻组所有向量构成查询 `Q`，要做：

- 对每个 `q in Q`，在 `X` 中找 L2 最近的 top-k。

数值公式：

- `||q - x||^2 = ||q||^2 + ||x||^2 - 2 * (q·x)`

所以核心计算是 `Q * X^T`（dot），再结合 norm 做 top-k 更新。

### 6.2 输入输出契约

输入：

- `target_group_ids`：需要处理的目标组列表（通常是 `in_neighbors` 非空的组）
- `dim`：向量维度
- `topk`：每 query 选多少跨组邻居
- 内部依赖：
  - `_group_id_to_range`
  - `_label_nav_graph->in_neighbors`
  - `_base_storage` 向量数据

输出：

- `cross_group_neighbors[qid]` 插入 `(target_vid, dist)`。
- 后续由 `build_cross_group_edges()` 合并入 `_graph`。

### 6.3 当前 GPU 路径（一步一步）

`gpu_cross_groups_search_all_batched(...)` 当前流程：

1. 统计每个目标组 query 数（来自其 `in_neighbors`）。
2. 构建 target/source/qid descriptors；当前推荐 route 是 `UNG_UNIVERSAL_GPU=1`。
3. base vectors 和 base norms 通过 `gpu_prepare_all_vectors_on_device(...)` 常驻 GPU，query 主要以 qid 形式传输和解析。
4. double-buffer descriptor batch 分 chunk 执行，使 H2D/descriptor 准备、kernel 和 D2H/merge 尽量流水化。
5. 对每个 target descriptor 执行 fused exact topK：
   - singleton / small / medium / group-desc kernel；
   - direct-qid resident-vector 路径；
   - unsupported 大组可按阈值 fallback。
6. universal route 使用 GPU global merge + flat-id output，减少 `SearchQueue` / per-group container 物化。
7. full-quality 配置仍会执行 `additional_edges`；当前主证据中 additional_edges 仍是 CPU Vamana。
8. 最终把 cross edges 合并到 `_graph`，这一步仍受 CPU-compatible output boundary 约束。

### 6.4 关键缓存复用

当前 `.cu` 做了大量“常驻/复用”：

- Host pinned：`g_h_Q / g_h_idx / g_h_dis`
- Device：`g_d_Q / g_d_idx / g_d_dis / g_d_dot / g_d_q_norm`
- 全量向量常驻：`g_d_all_X`
- 全量 norm 常驻：`g_d_all_norm`
- universal / flat-id 相关 descriptor、id-only output 和 merge buffers

`gpu_prepare_all_vectors_on_device` 会：

- 优先尝试连续内存直拷（host register + H2D）
- 否则走 pinned staging
- 一次性计算并缓存全量 `x_norm`

### 6.5 Kernel 族（当前实现）

- `l2_norm_sq_kernel`：批量算范数平方
- `update_topk_from_dot_tile_kernel`：单 tile 更新 running top-k
- `update_topk_from_dot_grouped_tiles_kernel`：多 tile grouped 更新，减少 launch
- `ung_dot_batched_naive_global_kernel`：自定义 CUDA Core dot kernel
- `ung_singleton_top1_global_kernel`：`nx=1` 快路
- `ung_small_group_topk_fused_global_kernel`：小组融合核
- `ung_medium_group_topk_fused_global_kernel`：中组融合核
- `ung_group_desc_topk_fused_global_kernel`：bucket 后按描述符批处理

---

## 7. 目前“写法是什么、怎么优化的”

下面给出主干路径“原始思路 -> 当前写法 -> 收益类型”。

### 7.1 LNG / descendants / coverage

- 原始思路：
  - 大量哈希集合累积、传播、去重；
  - 临时容器频繁创建销毁。
- 当前写法：
  - descendants / covered_sets 主容器改为连续 `vector`；
  - 线程本地 BFS 缓冲复用；
  - coverage 支持 descendants-direct 快路；
  - roaring 初始化并行 + `addMany`。
- 收益类型：
  - 大幅降低分配与哈希热点；
  - 降低 cache miss；
  - 把“不可预测抖动”改成“更线性可控”的内存访问。

### 7.2 Trie + min-super-set

- 原始思路：
  - 候选后基本排序 + 逐步比较。
- 当前写法：
  - 线程本地候选容器复用；
  - `label_set_size` 小跨度时改桶化遍历；
  - Trie superset 搜索里的 BFS 容器从 queue/set 改为 vector+head / unordered_set。
- 收益类型：
  - 降低排序与容器维护成本，缩短 LNG Phase1。

### 7.3 跨组边 GPU 计算

- 原始思路：
  - 组粒度细碎，频繁 H2D/D2H 与 kernel 启动，GPU 利用率受限。
- 当前写法：
  - 全量 `X` 与 `x_norm` 常驻，query 尽量以 qid/direct-qid 方式消费；
  - grouped tile GEMM + grouped topk update；
  - 小/中/单点组专门路径；
  - universal flat double-buffer route 用 descriptor batch + GPU global merge + flat-id output 降低输出边界开销；
  - source-centric no-lock 目前只是 future direction，可信复测慢于 universal。
- 收益类型：
  - 降低 launch 与拷贝开销；
  - 降低全局重复计算（尤其 `x_norm`）；
  - 提升长批次吞吐。

### 7.4 组内图 GPU / exact-anchor router

- 原始思路：
  - 每组 CPU Vamana，长尾大组和大量中组会拖慢 build；
  - 直接全量上 GPU 又会被小组 kernel launch、pack、fill 和质量问题拖住。
- 当前写法：
  - `UNG_GROUP_GRAPH_IMPL=4` conservative adaptive：小组 CPU fallback，中组 packed exact-anchor，大组 FastGrnndCuda；
  - `UNG_FAST_EXACT_DIRECT_H2D=1` 跳过 exact-anchor 的 host pack；
  - `UNG_TAGORE_FILL_THREADS=min(num_threads,16)` 降低 host allocator 竞争；
  - `NeighborList64` 和 reserve 降低 `Graph::neighbors` per-node object 成本。
- 收益类型：
  - x200/x400 coverage-query 上 packed exact-anchor 达到 CPU 级 L5000 recall；
  - direct-H2D 证明旧 exact-anchor 慢主要是外围，不是 GPU exact kernel；
  - 但最终 search 仍消费 CPU-compatible graph，flat/CSR 是下一阶段。

---

## 8. 当前可观测性能与正确性证据

优化过程与 A/B 数据详见：

- `docs/reports/OPTIMIZATION_STEP_BY_STEP.md`
- `docs/archive/OPTIMIZATION_PROVENANCE.md`
- `docs/reports/TECHNICAL_REPORT_OPTIMIZATION_SPEEDUP_CN.md`
- `docs/papers/EVIDENCE_MATRIX_CN.md`
- `docs/papers/REBUTTAL_CHECKLIST_CN.md`

关键结论（历史记录）：

- `216cd6c`：LNG/coverage 容器重构是最大收益点（端到端显著下降）。
- `d68f3a0`：Trie + Phase1 低风险继续提速。
- `ff47638` 试验已回退（有明确退化证据）。

正确性手段：

- GPU/CPU 对照的关键文件 md5 比对（`lng_descendants_rb.bin` 等）。
- 结构统计量对比（LNG 边数、平均 descendants、target_groups）。
- GPU 路径支持抽样验证：
  - `UNG_GEMM_VERIFY_SAMPLES`
  - `UNG_GEMM_VERIFY_STRICT`
- 论文 claim / artifact gate：
  - `python3 tools/benchmarks/run_submission_gate.py --final`

---

## 9. 输入数据规模与分布（CelebA，当前本机统计）

基于：

- `/home/graphdb/aaaGPU/FilterVectorData/celeba/celeba_base.bin`
- `/home/graphdb/aaaGPU/FilterVectorData/celeba/celeba_base_labels.txt`

统计值（2026-03-04，本机脚本统计）：

- 向量数：`202,599`
- 维度：`512`
- 唯一标签数：`40`
- 每向量标签数：
  - 平均 `9.03`
  - 中位数 `9`
  - p95 `14`
- 组（按标签集完全相同）：
  - 组数 `115,114`
  - 组大小中位数 `1`
  - 单点组占比 `78.0%`
  - 最大组大小 `358`

解释：

- 这是典型“**大量小组 + 长尾少量中大组**”分布。
- 该分布直接决定了跨组边阶段需要：
  - 小任务路径（避免 GPU 过碎）
  - 中大任务路径（尽量吃满算力）

---

## 10. 为什么端到端会慢、会波动（当前代码视角）

主要来源：

1. **组规模分布极不均衡**  
   大量 tiny group + 少量长尾 group，导致阶段尾部和调度不稳定。

2. **跨组边阶段是多阶段混合开销**  
   不只是 GEMM，还包含：
   - query flatten / gather
   - topk merge
   - 回写与图合并
   - additional edges 构建

3. **CPU/GPU 协同成本**  
   数据搬运与 kernel 启动开销在小任务上占比高。

4. **OpenMP 调度与系统噪声**  
   动态调度对长尾友好，但运行间仍有抖动。

---

## 11. 后续优化建议（按优先级）

### P0：先稳态，再极限吞吐

1. 固化测试协议：固定线程绑核、固定环境变量、A/B 至少3次中位数。
2. 将 `cross_edges.generate_ms` 拆得更细（已有基础），把“算子时间”与“回写合并时间”分开看。

### P1：跨组边进一步代码级优化

1. bucket 进一步细分（按 `nq*nx*dim` work 量，不只按 `nx`）。
2. topk 更新核优化（shared 选择逻辑可替换为更高效 block-select）。
3. 对大组引入更强算子（可评估 WMMA/TensorCore 自定义核，前提先验证数值与收益）。

### P2：LNG 构建阶段

1. `build_label_nav_graph` Phase2 目前串行，可探索并行安全写入策略。
2. Trie 节点结构做内存压缩，降低遍历 cache miss。

---

## 12. 重要环境变量速查

构建主线：

- `UNG_CROSS_EDGE_BACKEND`：`0=CPU`, `1=GPU`
- `UNG_CROSS_EDGE_GPU_STRICT`：GPU失败是否禁止回退
- `UNG_COVERAGE_IMPL`：`0=legacy`, `1=descendants_direct`
- `UNG_COVERAGE_THREADS`：coverage 线程数

GPU 跨组边常用：

- `UNG_Q_UPLOAD_MODE`
- `UNG_GEMM_IMPL`
- `UNG_FORCE_CUSTOM_KERNEL`
- `UNG_SMALL_GROUP_FUSED`, `UNG_MEDIUM_GROUP_FUSED`, `UNG_BUCKET_GROUP_FUSED`
- `UNG_SINGLETON_FASTPATH`
- `UNG_CPU_TINY_GROUPS`（及 `NX/NQ/OPS` 阈值）
- `UNG_GEMM_VERIFY_SAMPLES`, `UNG_GEMM_VERIFY_STRICT`

---

## 13. 索引输出物（`save` 关键文件）

`index_files/` 主要包含：

- `meta`
- `vecs.bin`, `labels.txt`
- `group_id_to_label_set`, `group_id_to_range`, `group_id_to_vec_ids.dat`
- `trie`
- `graph`, `global_graph`
- `lng_coverage_ratio`, `covered_sets`, `lng_descendants`, `lng_descendants_num`
- `lng_out_neighbors.dat`
- `lng_descendants_rb.bin`, `covered_sets_rb.bin`
- `vector_attr_graph`
- `reordered_vecs.fvecs`, `reordered_labels.txt`（供 ACORN）

`results/` 主要包含：

- `build_time.csv`
- `trie_label_frequency.csv`

---

## 14. 建议接手顺序（最快上手）

1. 先读：`docs/reports/OPTIMIZATION_STEP_BY_STEP.md`
2. 再读：本文件（全貌+当前实现）
3. 然后看代码：
   - `UNG/codes/src/uni_nav_graph.cpp`
   - `UNG/codes/src/gpu_gemm_topk.cu`
   - `UNG/codes/src/trie.cpp`
4. 最后按 `docs/runbooks/TESTING_GUIDE.md` 复现 A/B

这样可以先建立“证据链”再动代码，避免重复做已回退方案。
