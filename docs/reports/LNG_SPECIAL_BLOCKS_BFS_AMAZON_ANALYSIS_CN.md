# Amazon LNG Special Blocks BFS 与 Trie Special Blocks 对比实验报告

日期：2026-08-06

## 1. 实验对象

本报告整理并分析以下两个结果目录的对比：

- LNG BFS block 方法：`/home/dev/graphdb/FilterVectorResult/Amazon/results/lng_special_blocks_bfs/query_selected_recall_advantage_1000_1000_20000`
- Trie special block baseline：`/home/dev/graphdb/FilterVectorResult/Amazon/results/cpu_bruteforce_els_special_blocks/query_selected_recall_advantage_1000_1000_20000`

对应索引目录：

- LNG BFS block 索引：`/home/dev/graphdb/FilterVectorResult/Amazon/index/UNG_lng_special_blocks_hybrid_bfs/index_files/`
- Trie block 索引：`/home/dev/graphdb/FilterVectorResult/Amazon/index/UNG_special_blocks_hybrid_bdeg128_bcross10/index_files/`

两次搜索命令的主要搜索参数保持一致：

- `entry_group_provider=cpu_bruteforce_els`
- `graph_search_backend=neighbor_list`
- `num_entry_points=16`
- `force_use_alg=1`
- `K=10`
- `Lsearch=1000,2000,...,20000`
- 查询集均为 `query_selected_recall_advantage`

需要注意：Trie baseline 的 `query_details_repeat1.csv` 时间戳是 `2026-08-01 15:09:06`，LNG BFS 的 `query_details_repeat1.csv` 时间戳是 `2026-08-05 16:16:53`。严格的最终论文/汇报数据建议用同一版 `search_UNG_index` 重新跑一次 Trie baseline。不过从本次 CSV 明细看，两者的入口规模、候选规模和 query 基本统计完全一致，因此仍可用于定位 LNG 方法慢在哪里。

## 2. 一次重要的前置问题：为什么最开始两个 block 文件会完全一样

早先对比中曾发现 `lng_special_blocks_bfs` 和 `cpu_bruteforce_els_special_blocks` 的以下文件字节级相同：

- `special_blocks.csv`
- `special_block_members.csv`
- `special_block_children.csv`
- `special_edges.csv`
- `graph`
- `lng_out_neighbors.dat`

这不是算法本身导致的巧合，而是实验构建流程问题：当时 `build_UNG_index` 二进制早于 LNG block 相关源码修改，runner 只检查可执行文件是否存在，没有检查源码是否比二进制更新。因此目录名虽然叫 `UNG_lng_special_blocks_hybrid_bfs`，实际构建程序仍可能走旧逻辑或默认 trie block 逻辑。

后来已经在 `experiments/lng_special_blocks/run_lng_special_blocks.py` 中加入 `build_tools.policy=if_stale`，当 `UNG/codes` 源码或 CMake 文件比目标二进制更新时会自动重新编译。当前分析使用的是重新构建后的 LNG BFS 索引，其 meta 和 build.log 明确记录：

```text
special_block_partition=lng
special_block_tree_mode=bfs
special_block_tree_seed=1
```

## 3. 总体搜索结果

`search_time_summary.csv` 给出的端点结果如下：

| Lsearch | 方法 | Summary Time ms | Recall |
|---:|---|---:|---:|
| 1000 | LNG BFS block | 2220.76 | 0.765398 |
| 1000 | Trie block baseline | 942.248 | 0.557331 |
| 20000 | LNG BFS block | 5800.03 | 0.927194 |
| 20000 | Trie block baseline | 1993.50 | 0.982547 |

可以看到：

- 在小 `Lsearch=1000` 下，LNG BFS 更慢，但 recall 高于旧 Trie baseline。
- 在大 `Lsearch=20000` 下，LNG BFS 仍明显更慢，并且 recall 低于 Trie baseline。
- 因此当前 LNG BFS block 并不是简单的“多花时间换 recall”，在高搜索预算下存在明显的搜索效率问题。

## 4. query 明细聚合：慢在哪里

下表来自两个 `query_details_repeat1.csv` 的逐 query 聚合。这里的 `Time_ms` 是 per-query 明细平均值，和 summary 文件的总耗时口径不同；定位慢点主要看相对比例和细分计数。

### 4.1 Lsearch=1000

| 指标 | LNG BFS | Trie baseline | 比例 LNG/Trie |
|---|---:|---:|---:|
| Time_ms | 109.23 | 46.19 | 2.36x |
| search_time_ms | 53.48 | 8.96 | 5.97x |
| core_search_time_ms | 53.47 | 8.96 | 5.97x |
| Recall | 0.765 | 0.557 | 1.37x |
| DistCalcs | 18508 | 2162 | 8.56x |
| NumNodeVisited | 17608 | 1262 | 13.96x |
| SpecialRegularExpanded | 296 | 71 | 4.18x |
| SpecialFreeExpanded | 733 | 45 | 16.40x |
| SpecialRegularEdgesScanned | 2929 | 525 | 5.58x |
| SpecialEdgesScanned | 130957 | 2973 | 44.05x |
| SpecialIntraEdgesScanned | 42262 | 2059 | 20.53x |
| SpecialInterEdgesScanned | 88694 | 914 | 97.02x |

`Lsearch=1000` 时，核心慢点非常明确：LNG BFS 平均每个 query 扫描约 13.1 万条 special edge，而 Trie baseline 只扫描约 2973 条。尤其是 inter-block special edge，LNG BFS 是 baseline 的约 97 倍。

### 4.2 Lsearch=20000

| 指标 | LNG BFS | Trie baseline | 比例 LNG/Trie |
|---|---:|---:|---:|
| Time_ms | 284.83 | 96.44 | 2.95x |
| search_time_ms | 268.33 | 83.33 | 3.22x |
| core_search_time_ms | 268.32 | 83.32 | 3.22x |
| Recall | 0.927 | 0.983 | 0.94x |
| DistCalcs | 76942 | 25865 | 2.97x |
| NumNodeVisited | 69406 | 18329 | 3.79x |
| SpecialRegularExpanded | 6696 | 2635 | 2.54x |
| SpecialFreeExpanded | 12010 | 2479 | 4.84x |
| SpecialRegularEdgesScanned | 43752 | 13587 | 3.22x |
| SpecialEdgesScanned | 2464683 | 163754 | 15.05x |
| SpecialIntraEdgesScanned | 691739 | 116653 | 5.93x |
| SpecialInterEdgesScanned | 1772944 | 47102 | 37.64x |

`Lsearch=20000` 时，LNG BFS 的核心搜索时间约为 baseline 的 3.22 倍。平均每 query 扫描 special edge 约 246 万条，其中 inter-block special edge 约 177 万条。也就是说慢点仍然集中在 special overlay，尤其是 inter-block overlay。

## 5. 输入和入口并没有变

下面这些指标在两个方法中完全一致：

| 指标 | LNG BFS | Trie baseline | 说明 |
|---|---:|---:|---|
| NumEntries | 5945.88 | 5945.88 | 入口 group 数一致 |
| EntryGroupMatchedPoints | 161814 | 161814 | 入口 group 覆盖点数一致 |
| CandSize | 1037.60 | 1037.60 | query 候选规模一致 |
| QuerySize | 1.225 | 1.225 | query label 数一致 |
| SpecialSearchEnabled | 1 | 1 | 都启用 special block search |
| SpecialFreeUseRegular | 0 | 0 | free 状态都不继续走 regular 边 |

因此慢不是因为 query 不同、入口 provider 不同、普通搜索参数不同，而是 block 划分后触发的 special block 搜索状态和 special edge 结构不同。

## 6. block 构建结果对比

索引 meta 显示，LNG BFS 与 Trie block 的 special block 结构差异很大：

| 指标 | LNG BFS block | Trie block baseline | LNG/Trie |
|---|---:|---:|---:|
| special_block_count | 123 | 170 | 0.72x |
| special_block_member_groups | 467464 | 492152 | 0.95x |
| special_block_member_points | 554055 | 581472 | 0.95x |
| special_block_child_edges | 131 | 156 | 0.84x |
| special_edge_count | 120046230 | 56171630 | 2.14x |
| special_edge_intra_count | 29688060 | 31433290 | 0.94x |
| special_edge_inter_count | 90358170 | 24738340 | 3.65x |

最关键的变化是：LNG BFS block 数量更少，但 inter special edge 数量暴涨。也就是说它不是简单地少建了 block，而是把 block 合并成更大的结构后，跨 block overlay 变得非常密。

### 6.1 block 尺寸分布

| 指标 | LNG BFS | Trie baseline |
|---|---:|---:|
| block 数 | 123 | 170 |
| 总 point_count | 554055 | 581472 |
| point_count min | 1001 | 1015 |
| point_count p50 | 2169 | 1901 |
| point_count p90 | 5369 | 6952 |
| point_count p95 | 7016 | 9653 |
| point_count max | 150409 | 34221 |
| point_count mean | 4504.5 | 3420.4 |
| child_block_count p90 | 1 | 3 |
| child_block_count max | 70 | 27 |
| root_label_len p50 | 3 | 3 |
| root_label_len max | 5 | 7 |

LNG BFS 的分布有明显长尾：最大 block 达到 150409 点，而 Trie baseline 最大 block 只有 34221 点。LNG BFS 还有一个 block 的 `child_block_count=70`，远高于 Trie baseline 的最大值 27。

### 6.2 典型大 block

LNG BFS 中最大的几个 block：

| block_id | root_labels | point_count | member_group_count | child_block_count | subtree_point_count |
|---:|---|---:|---:|---:|---:|
| 45 | 20835 | 150409 | 138809 | 28 | 206910 |
| 123 | 1 2 | 65591 | 56429 | 70 | 272736 |
| 15 | 20837 | 16356 | 15079 | 2 | 20817 |
| 5 | 20836 | 13959 | 12086 | 5 | 13959 |

Trie baseline 中最大的几个 block：

| block_id | root_labels | point_count | member_group_count | child_block_count | subtree_point_count |
|---:|---|---:|---:|---:|---:|
| 169 | 1 2 | 34221 | 29791 | 27 | 272737 |
| 72 | 20835 | 28671 | 21919 | 19 | 210944 |
| 71 | 20835 20836 | 20909 | 17089 | 11 | 73844 |
| 170 | 1 | 15634 | 13915 | 3 | 290684 |

这说明 LNG BFS 生成树的自底向上切分，把某些上层标签对应的区域聚合得过大。大 block 本身会增加 intra graph 构建和搜索成本；更严重的是，大 parent block 连接多个 child block 时会造成 inter special edge 爆炸。

## 7. 代码路径分析

### 7.1 LNG block 如何形成

LNG block 构建入口在：

`/home/dev/graphdb/FilterVectorCode_refactor/UNG/codes/src/uni_nav_graph_special_blocks.cpp`

当前 LNG 分支逻辑为：

1. 使用 `_label_nav_graph->out_neighbors` 作为 LNG 图。
2. 根据 `special_block_tree_mode=bfs` 构建 LNG spanning forest。
3. 对生成树按 DFS visit order 逆序自底向上聚合。
4. 当 `cur.uncovered_points > min_points` 时生成一个 `SpecialBlock`。
5. 最后用 block 的 `root_labels` 重新套用 LNG label 包含关系，生成 block 间 `child_block_ids`。

核心代码在：

- `build_special_blocks()` 中 LNG 分支：`uni_nav_graph_special_blocks.cpp:511-559`
- `build_lng_special_blocks()`：`ung_lng_block_partition.cpp:230-300`

当前 `build_lng_special_blocks()` 中，超过阈值时直接以当前 group 作为 block root：

```cpp
if (cur.uncovered_points > input.min_points)
{
   SpecialBlock block;
   block.root_group_id = group_id;
   block.point_count = cur.uncovered_points;
   block.root_labels = sorted_unique_labels(group_labels[group_id]);
   block.member_group_ids = sorted_unique(std::move(cur.member_group_ids));
   block.child_block_ids = sorted_unique(std::move(cur.child_block_ids));
   ...
}
```

这会受到 BFS 生成树结构强烈影响。如果 BFS tree 在某些上层标签 group 下聚合了大量未被子 block 截断的 group，就会产生很大的 block，例如 `root_labels=20835` 的 150409 点 block。

### 7.2 inter special edge 如何生成

inter special edge 构建在：

`/home/dev/graphdb/FilterVectorCode_refactor/UNG/codes/src/uni_nav_graph_special_blocks.cpp:879-984`

逻辑是对每个 parent block 和它的每个 child block：

```cpp
for (const SpecialBlock &block : _special_blocks)
{
   const auto &src_points = block_points[block.block_id];
   for (IdxType child_block_id : block.child_block_ids)
   {
      const auto &dst_points = block_points[child_block_id];
      for (IdxType source : src_points)
      {
         append_cross_edges_from_target_points(
            source,
            dst_points,
            ...,
            special_block_num_cross_edges);
      }
   }
}
```

因此 inter edge 数量近似受以下因素控制：

```text
sum_over_parent_blocks(point_count(parent) * child_count(parent) * special_block_num_cross_edges)
```

当 LNG BFS 出现 `point_count=65591, child_block_count=70` 或 `point_count=150409, child_block_count=28` 这种结构时，即使 `special_block_num_cross_edges=10`，也会产生非常大的 inter overlay。

这正对应 meta 中的结果：

```text
LNG BFS special_edge_inter_count = 90358170
Trie    special_edge_inter_count = 24738340
```

### 7.3 搜索时为什么会被放大

special search 的 free-state 搜索在：

`/home/dev/graphdb/FilterVectorCode_refactor/UNG/codes/src/uni_nav_graph_search_backend.cpp:409-666`

其关键逻辑是：

1. 对入口 group 判断是否属于 query 覆盖的 block。
2. 对 regular 邻居也可通过 `cached_point_is_free()` 升级成 free 状态。
3. 一旦当前 candidate 是 free，就扫描该点的所有 `_special_edges_by_point[cur.id]`。
4. 当前实验中 `SpecialFreeUseRegular=0`，因此 free 节点扫描完 special edge 后直接 `continue`，不再走普通 graph edge。

核心片段：

```cpp
if (cur.free)
{
   scan_special_edges(_special_edges_by_point[cur.id], false);
   if (!runtime.special_block_free_use_regular)
      continue;
}
```

这意味着 special edge 的 per-point 度数越高，free 节点展开成本越高。LNG BFS 的 inter edge 总量更大，且某些大 block 覆盖了大量点，所以在搜索中一旦进入 free 状态，就会出现大量 special edge 扫描。

## 8. 为什么 LNG BFS 慢但高 Lsearch recall 反而低

这个现象由两个因素共同造成。

### 8.1 query 覆盖到的 block 更少，入口 free 点更少

虽然 LNG BFS special edge 更多，但 query 覆盖统计反而低于 Trie baseline：

| 指标 | LNG BFS | Trie baseline | LNG/Trie |
|---|---:|---:|---:|
| SpecialQueryPoints | 139125 | 158481 | 0.88x |
| SpecialQueryBlockCount | 34.64 | 49.49 | 0.70x |
| SpecialEntryFreePoints | 2971 | 6899 | 0.43x |
| SpecialEntryRegularPoints | 4565 | 637 | 7.16x |

也就是说，LNG BFS 的 block root label 分布使得同一个 query 能直接进入 free 状态的入口点更少，大量入口点变成 regular 状态。搜索需要先沿 regular graph 展开，再通过 regular 邻居升级成 free。

这也体现在：

```text
Lsearch=20000 SpecialFreeUpgrades:
LNG BFS = 1132
Trie    = 80
```

LNG BFS 的 free upgrade 次数约为 baseline 的 14.19 倍。这说明它不是一开始就精准进入有用的 free 区域，而是在搜索过程中不断从 regular 状态切换到 free 状态。

### 8.2 大量 inter edge 消耗搜索预算，但不一定导向更好的近邻

在 `Lsearch=20000` 下，LNG BFS 平均每个 query 扫描约 177 万条 inter special edge，而 Trie baseline 只有约 4.7 万条。大量边扫描带来了更多距离计算和候选插入，但最终 recall 只有 0.927，低于 Trie baseline 的 0.983。

这说明当前 LNG BFS block 的 inter overlay 更密，但有效性不足。它在更大的 block 和更多 child block 之间生成了大量跨 block top-k 边，这些边可能覆盖很广，但并没有像 Trie block 那样稳定地把搜索导向满足 query 的高质量近邻区域。

## 9. 根因总结

当前 `lng_special_blocks_bfs` 慢的根因不是入口 provider，也不是普通 graph backend，而是 LNG BFS block 切分带来的 special overlay 结构问题：

1. LNG BFS 生成的 block 更少、更大，最大 block 达到 150409 点，远大于 Trie baseline 的 34221 点。
2. LNG BFS 出现高 child fanout block，例如 `(1 2)` block 有 70 个 child block。
3. inter special edge 构建对 parent block 的所有点和每个 child block 都生成 cross edges，导致 inter edge 从 2473 万涨到 9036 万。
4. 搜索中 free 节点会扫描该点全部 special edge，且当前 `SpecialFreeUseRegular=0`，所以 free 状态基本被 special overlay 主导。
5. LNG BFS 的 query 覆盖 block 数和入口 free 点更少，导致 regular 展开和 free upgrade 也显著增加。
6. 最终表现为更多节点访问、更多距离计算、更多 special edge 扫描，尤其是 inter special edge 扫描爆炸。

最短结论：

```text
LNG BFS 慢在 special free-state 搜索阶段，直接原因是 inter-block special edge 扫描爆炸；
更深层原因是 BFS 生成树自底向上切 block 后形成了大 block 和高 child fanout，
使 block 间 overlay 过密，同时 query 直接覆盖的入口 free 点反而减少。
```

## 10. 后续建议

### 10.1 先做公平 baseline

当前 Trie baseline 搜索结果时间戳早于 LNG BFS 结果。建议用当前同一版 `search_UNG_index` 对 Trie block baseline 重新搜索一次，避免二进制版本差异影响最终结论。

### 10.2 限制 LNG block 的 inter edge 爆炸

可以考虑加入以下控制参数或策略：

- 对 parent block 的 `child_block_count` 设置上限。
- 对 `point_count(parent) * child_count(parent)` 设置 inter work cap。
- 对过大的 parent block 不生成全部 child inter edge，改为选择 top child blocks。
- 对 `special_block_num_cross_edges` 做 parent-size 自适应衰减。
- 对 `root_labels` 过浅的大 block 进行二次切分，避免 `20835`、`1 2` 这种大 block 统治 overlay。

### 10.3 调整 LNG 生成树切分

当前 BFS 生成树可能把一些上层标签区域聚合得过大。可以对 LNG block partition 增加额外切分准则：

- 不只用 `uncovered_points > min_points`，还加 `max_points`。
- 对 child fanout 太高的节点强制切分。
- 在自底向上聚合时，如果 root label 太短且点数过大，延迟到更具体 label 节点切 block。
- 比较 `bfs` 与 `random_seed` 的 block 最大尺寸、inter edge 数和 query 覆盖率，选择更稳的树构造方式。

### 10.4 增加搜索侧保护

如果暂时不改构建，可以在搜索侧减少 free-state 爆炸：

- 开启或实现 `UNG_SPECIAL_EARLY_STOP` 的更积极策略。
- 对每个 free 点扫描 special edge 时加入 per-node edge scan cap。
- 对 inter edge 和 intra edge 分别设置预算，优先 intra，再按需 inter。
- 对 query 覆盖较弱或入口 free 点较少的 query，不启用 LNG special block search。

## 11. 复查路径

主要结果文件：

- LNG summary：`/home/dev/graphdb/FilterVectorResult/Amazon/results/lng_special_blocks_bfs/query_selected_recall_advantage_1000_1000_20000/results/search_time_summary.csv`
- LNG query details：`/home/dev/graphdb/FilterVectorResult/Amazon/results/lng_special_blocks_bfs/query_selected_recall_advantage_1000_1000_20000/results/query_details_repeat1.csv`
- Trie summary：`/home/dev/graphdb/FilterVectorResult/Amazon/results/cpu_bruteforce_els_special_blocks/query_selected_recall_advantage_1000_1000_20000/results/search_time_summary.csv`
- Trie query details：`/home/dev/graphdb/FilterVectorResult/Amazon/results/cpu_bruteforce_els_special_blocks/query_selected_recall_advantage_1000_1000_20000/results/query_details_repeat1.csv`

主要构建文件：

- LNG meta：`/home/dev/graphdb/FilterVectorResult/Amazon/index/UNG_lng_special_blocks_hybrid_bfs/index_files/meta`
- LNG build log：`/home/dev/graphdb/FilterVectorResult/Amazon/index/UNG_lng_special_blocks_hybrid_bfs/others/build.log`
- LNG block CSV：`/home/dev/graphdb/FilterVectorResult/Amazon/index/UNG_lng_special_blocks_hybrid_bfs/index_files/special_blocks.csv`
- Trie meta：`/home/dev/graphdb/FilterVectorResult/Amazon/index/UNG_special_blocks_hybrid_bdeg128_bcross10/index_files/meta`
- Trie build log：`/home/dev/graphdb/FilterVectorResult/Amazon/index/UNG_special_blocks_hybrid_bdeg128_bcross10/others/build.log`
- Trie block CSV：`/home/dev/graphdb/FilterVectorResult/Amazon/index/UNG_special_blocks_hybrid_bdeg128_bcross10/index_files/special_blocks.csv`

主要代码路径：

- LNG block partition：`/home/dev/graphdb/FilterVectorCode_refactor/UNG/codes/src/ung_lng_block_partition.cpp`
- special block 构建和 special edge overlay：`/home/dev/graphdb/FilterVectorCode_refactor/UNG/codes/src/uni_nav_graph_special_blocks.cpp`
- special free-state 搜索：`/home/dev/graphdb/FilterVectorCode_refactor/UNG/codes/src/uni_nav_graph_search_backend.cpp`
