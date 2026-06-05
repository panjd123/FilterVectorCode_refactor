# Refactor 优化步骤总览（逐步、可复现、可审计）

本文档是“优化时间线 + 数据证据”文档。
它关注三件事：

1. 每一步到底改了什么。
2. 性能到底变好了还是变坏了。
3. 正确性如何被验证。

## 0. 测试基线约定

为了让不同步骤可比，绝大多数 A/B 测试使用统一设置：

- 数据集：`celeba`
- 构建参数：`--max_degree 64 --Lbuild 100 --alpha 1.2 --num_cross_edges 6`
- 环境变量：
  - `UNG_CROSS_EDGE_BACKEND=1`
  - `UNG_COVERAGE_IMPL=1`
  - `UNG_COVERAGE_THREADS=32`
- 主指标：`Index time`
- 分段指标：
  - `Finished building LNG`
  - `descendants`
  - `coverage`
  - `cross-group edges`

说明：

- 单次运行容易受系统噪声影响。
- 所以结论优先看 A/B 多轮“中位数”。

## 1. 第一步迁移：把 aaaGPU 的一批改动引入 refactor

### 1.1 变更点

- 提交：`076d990`
- 文件：`UNG/codes/src/uni_nav_graph.cpp`
- 目标：迁移 aaaGPU 的 CPU 侧流程优化
- 关键函数：
  - `get_min_super_sets`
  - `get_descendants_info`
  - `cal_f_coverage_ratio`

### 1.2 结果

- A/B CSV：`/home/graphdb/FilterVectorResultsRefactor/ab_benchmark_20260302_235803.csv`
- 对比：`87892ae` -> `076d990`（中位数）

- `Index time`: `72437 -> 84878 ms`（`+17.2%`，回退）
- `LNG`: `44578.7 -> 42171.5 ms`（下降）
- `descendants`: `2359.1 -> 2032.0 ms`（下降）
- `coverage`: `13965.8 -> 26666.0 ms`（大幅上升）
- `cross-group edges`: `1711.4 -> 1541.0 ms`（下降）

### 1.3 结论

- 该步说明“局部优化（cross-edge）有效”并不等于“端到端有效”。
- 覆盖率阶段抖动/变慢吞掉了收益。

## 2. 第二步试验：覆盖率线程上限（失败并回退）

### 2.1 变更点

- 试验提交：`ff47638`
- 回退提交：`14a0fa9`
- 思路：给 `descendants_direct` 限线程 + 改 OMP 调度，降低波动

### 2.2 结果

- A/B CSV：`/home/graphdb/FilterVectorResultsRefactor/ab_covcap_20260303_001148.csv`
- 对比：`5985bb1` -> `ff47638`（中位数）

- `Index time`: `58044.5 -> 83180.0 ms`（显著回退）
- `coverage`: `4738.65 -> 21245.55 ms`（显著回退）

### 2.3 结论

- 假设不成立，必须回退。
- 已保留失败记录，防止后续重复踩坑。

## 3. 第三步高影响优化：哈希布局改为连续布局

### 3.1 变更点

- 提交：`216cd6c`
- 文件：
  - `UNG/codes/include/label_nav_graph.h`
  - `UNG/codes/src/uni_nav_graph.cpp`

核心改法：

- `_lng_descendants` 与 `covered_sets` 从 `unordered_set` 改为 `vector`。
- descendants 构建直接写 BFS 发现序列，避免大量哈希插入。
- coverage 的 descendants-direct 路径改为连续 append。
- legacy 路径保留 `sort + unique`，保证 DAG 合并场景语义正确。
- `initialize_roaring_bitsets` 改为并行 + `addMany` 批量写入。

### 3.2 正确性验证

GPU/CPU 对照下，关键文件 md5 一致：

- `lng_descendants_rb.bin`
- `covered_sets_rb.bin`
- `vector_attr_graph`
- `lng_descendants_num`
- `lng_coverage_ratio`

路径：

- `/home/graphdb/FilterVectorResultsRefactor/opt_vecset_gpu_20260303_003040/index_files`
- `/home/graphdb/FilterVectorResultsRefactor/opt_vecset_cpu_20260303_003205/index_files`

### 3.3 性能结果

- A/B CSV：`/home/graphdb/FilterVectorResultsRefactor/ab_vecset_20260303_003348.csv`
- 对比：`dcd5dee` -> `216cd6c`（中位数）

- `Index time`: `84115.0 -> 35314.5 ms`（`-58.0%`）
- `LNG`: `39002.6 -> 32308.1 ms`（`-17.2%`）
- `descendants`: `3059.7 -> 104.55 ms`（`-96.6%`）
- `coverage`: `29665.95 -> 219.1 ms`（`-99.3%`）
- `cross-group edges`: `1540.2 -> 1413.95 ms`（`-8.2%`）

### 3.4 结论

- 这是当前历史中最关键的一步。
- 它证明“数据结构与内存访问模式”比“算子微调”更影响端到端。

## 4. 第四步低风险跟进：Trie 与 LNG Phase1 开销压缩

### 4.1 变更点

- 提交：`d68f3a0`
- 文件：
  - `UNG/codes/src/trie.cpp`
  - `UNG/codes/src/uni_nav_graph.cpp`

核心改法：

- Phase1 (`build_label_nav_graph`) 使用线程本地可复用 `min_super_set_ids`，并用 `swap` 写回。
- trie 插入时仅在必要时 resize，去掉冗余 resize。
- `get_super_set_entrances`：
  - `queue` 改 `vector + head`
  - `set` 去重改 `unordered_set`
  - 增加 `_label_to_nodes` 索引边界保护

### 4.2 语义一致性验证

Before/After 日志中关键结构统计保持一致：

- `Average number of descendants per group`: `275.4`
- `LNG edges`: `946138`
- `target_groups`: `115073`

### 4.3 性能结果

- A/B CSV：`/home/graphdb/FilterVectorResultsRefactor/ab_safe_trie_20260303_005332.csv`
- 对比：`8d1a549` -> `d68f3a0`（中位数）

- `Index time`: `49100.5 -> 36969.5 ms`（`-24.7%`）
- `LNG`: `45746.45 -> 33878.15 ms`（`-25.9%`）
- `descendants`: `131.75 -> 117.6 ms`（`-10.7%`）
- `coverage`: `261.2 -> 329.75 ms`（`+26.2%`，但绝对值仅 +68.55ms）
- `cross-group edges`: `1518.25 -> 1354.85 ms`（`-10.8%`）

### 4.4 结论

- 在保持语义不变的前提下，进一步压缩了 LNG 主耗时。
- 目前端到端主瓶颈仍是 LNG 阶段。

## 5. 被否决的激进路径（未入主线）

### 5.1 试验内容

- 思路：在 `get_min_super_sets` 做 exact-node 向下 BFS 快路径

### 5.2 问题

虽然速度极快，但语义被破坏：

- `LNG edges`: `946138 -> 20007`
- `Average descendants`: `275.4 -> 0.3`
- `target_groups`: `115073 -> 20007`

证据日志：

- `/home/graphdb/FilterVectorResultsRefactor/opt_exactbfs_gpu_20260303_004843/others/ung_build.log`

### 5.3 处理

- 未提交入主线，已完全撤销。

## 6. 当前最优稳定状态

- 性能主线提交：`d68f3a0`
- 文档主线提交：见 `git log` 中其后的 docs 提交
- 当前可复现稳定区间（CelebA）：端到端约 `35s ~ 40s`

## 7. 经验总结（给后续优化者）

1. 先抓数据结构与内存访问，再抓算子细节。
2. 任何“看起来更快”的路径必须先过语义校验（边数/后代数/target_groups）。
3. 单次结果不可信，必须 A/B 多轮看中位数。
4. 回退实验也要文档化，失败信息本身就是资产。
5. 保持“每步一个 commit + 一份 CSV + 一份日志路径”的最小证据链。
