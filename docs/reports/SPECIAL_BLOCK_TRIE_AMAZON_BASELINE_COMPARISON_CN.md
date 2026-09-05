# Amazon SpecialBlockTrie 与现有 Baseline 对比报告

## 结论摘要

当前最终方案为：

`special_trie_router16_group_cap256_frontier8_preexpand2_repeat3`

它不是所有结果中的全局 recall/QPS 最优点，但在当前 UNG 特殊 block 索引体系内表现最好：

- 相比同一 `UNG_special_block_trie_regular` 索引上的 CPU ELS，四档 recall 提升 `0.0057-0.0133`，QPS 基本持平或略高。
- 相比旧 terminal-only Router，Lsearch=20,000 的 recall 从 `0.9199` 提升到 `0.9940`。
- 相比 uncapped Router，free-group 初始入口从 `4,413.5` 降到 `348.8`，距离计算减少 `15.9%/6.7%/4.5%/2.8%`。
- FAVOR 的高 recall 前沿仍然更强：最高约 `0.9960`，但它使用独立索引和 `prefilter_selectivity_threshold` 参数，不能简单视为同一种搜索预算。

因此，当前方案的定位是：在 SpecialBlockTrie/UNG 路径中取得高 recall 和较稳定 QPS；如果目标是全目录的最高 recall/QPS，FAVOR 仍是需要重点对比的外部 baseline。

## 对比范围与可比性

所有严格可比结果满足：

- Dataset：Amazon，602,453 vectors，dimension 768。
- Query task：`query_selected_recall_advantage`，1,971 queries。
- `K=10`，100 search threads，containment，L2。
- 结果目录使用 `query_selected_recall_advantage_1000_1000_20000` 的四档：1,000、5,000、10,000、20,000。
- 结果目录中同一 query task 共发现 81 条完整四档曲线；报告将同一方法的调参重复归为一族，并选取代表点，另保留完整结果文件路径。

需要区分三类结果：

1. **严格同索引**：当前方案、旧 Router、terminal-only、CPU ELS 都使用 `UNG_special_block_trie_regular`，这是判断入口策略收益的主要依据。
2. **同查询任务但不同索引**：ACORN、NaviX、FAVOR、GPU ELS special blocks、HashPrune 等可以做系统级参考，但索引结构和参数语义不同。
3. **不可直接对齐的扫描预算**：Curator 使用 `search_ef`，FAVOR 使用 `prefilter_selectivity_threshold` 与自身搜索 sweep；报告保留其原始参数，不把它们强行解释为 UNG 的 Lsearch。

大多数外部 baseline 是 1 次重复，当前方案和同索引 CPU ELS 使用 3 次重复；QPS 只作为结果目录中记录值比较，优先看 recall/QPS 的相对趋势。

## 同索引对比

### 主要曲线

| Lsearch | Terminal-only Recall / QPS | CPU ELS 同索引 Recall / QPS | Uncapped Router Recall / QPS | 最终方案 Recall / QPS |
|---:|---:|---:|---:|---:|
| 1,000 | 0.7166 / 2,220.7 | 0.8992 / 2,057.6 | 0.9076 / 1,725.3 | **0.9099 / 2,141.4** |
| 5,000 | 0.8470 / 983.2 | 0.9651 / 891.9 | 0.9782 / 890.8 | **0.9785 / 904.6** |
| 10,000 | 0.8907 / 647.7 | 0.9798 / 609.8 | 0.9898 / 611.2 | **0.9911 / 609.0** |
| 20,000 | 0.9199 / 417.4 | 0.9882 / 403.1 | 0.9936 / 402.7 | **0.9940 / 406.2** |

相对同索引 CPU ELS：

| Lsearch | Recall 增益 | QPS 变化 |
|---:|---:|---:|
| 1,000 | +0.0108 | +4.1% |
| 5,000 | +0.0133 | +1.4% |
| 10,000 | +0.0113 | -0.1% |
| 20,000 | +0.0057 | +0.8% |

这说明入口压缩没有牺牲同索引下的最终效果。L=10,000 的 QPS 差异只有约 0.1%，属于测量噪声范围；主要收益体现在 recall 和初始距离计算减少。

相对 terminal-only，最终方案的 recall 增益为：

`+0.1933 / +0.1315 / +0.1004 / +0.0741`

这部分收益来自 block coverage、frontier landmarks 和 free-state 搜索路径，而不是单纯增加 Lsearch。

## 外部 baseline 对比

下表使用各方法在相同报告档位中的原始记录值。

| Lsearch | 当前方案 | ACORN | NaviX | FAVOR 0.25 | GPU ELS special blocks |
|---:|---:|---:|---:|---:|---:|
| 1,000 | 0.9099 / 2,141 | 0.9122 / 2,199 | 0.9472 / 1,015 | 0.9958 / 985 | 0.9444 / 1,086 |
| 5,000 | 0.9785 / 905 | 0.9162 / 653 | 0.9503 / 528 | 0.9960 / 670 | 0.9740 / 983 |
| 10,000 | 0.9911 / 609 | 0.9163 / 288 | 0.9505 / 337 | 0.9960 / 517 | 0.9855 / 695 |
| 20,000 | 0.9940 / 406 | 0.9164 / 93 | 0.9506 / 212 | 0.9960 / 396 | 0.9932 / 457 |

表中每个单元格为 `Recall / QPS`。

### ACORN

ACORN 在 L=1,000 上略高于当前方案：recall `0.9122` 对 `0.9099`，QPS `2,199` 对 `2,141`。从 L=5,000 开始，当前方案同时拥有更高 recall 和更高 QPS；在 L=20,000，当前 recall 高约 `0.0775`，QPS 约为 ACORN 的 `4.4` 倍。ACORN-1 的 recall 最高约 `0.9213`，但 QPS 更低。

### NaviX

NaviX 在 L=1,000 有更高 recall，但 QPS 约为当前方案的一半。L>=5,000 后当前方案同时超过 NaviX：L=20,000 为 `0.9940 / 406`，NaviX 为 `0.9506 / 212`。

### FAVOR

FAVOR 0.25 是目录中 recall 最强的代表性曲线之一：

- L=1,000：`0.9958 / 985`
- L=5,000：`0.9960 / 670`
- L=10,000：`0.9960 / 517`
- L=20,000：`0.9960 / 396`

与当前方案相比，FAVOR 在四档都高约 `0.0021-0.0858` recall；当前方案则在低到中等 Lsearch 上有明显 QPS 优势，L=20,000 的 QPS 也略高。FAVOR 0.1/0.2/0.3 的 recall/QPS 曲线也大体位于当前方案之上或附近。

但 FAVOR 的 `Lsearch` 实际对应自身 sweep，并同时受 `prefilter_selectivity_threshold` 控制；它不是 UNG 图搜索中的同义预算。因此可以得出“FAVOR 的系统级前沿更强”，不能直接得出“当前 UNG 图搜索每次距离计算都更慢”。

### GPU ELS special blocks

CPU ELS special blocks 使用 `UNG_special_blocks_hybrid_bdeg128_bcross10` 索引，在 L=20,000 为 `0.9825 / 988.7`。它比当前方案快约 2.4 倍，但 recall 低 `0.0114`；在 L=1,000 两者 QPS 接近，而当前方案 recall 高 `0.3526`。这说明该旧混合索引更偏向低 recall/高吞吐工作点。

GPU ELS special blocks 使用另一套 `UNG_special_blocks_hybrid_bdeg128_bcross10` 索引。它在 L=20,000 达到 `0.9932 / 456.8`，比当前方案快约 12.5%，但 recall 低 `0.0008`。在 L=5,000 和 10,000，它的 QPS 也更高，但 recall 分别低 `0.0045` 和 `0.0055`。这是典型的 recall/QPS trade-off，不是完全支配关系。

### Curator

Curator 使用 `search_ef`，不能按 Lsearch 直接对齐。其代表点如下：

| Curator search_ef | Recall / QPS |
|---:|---:|
| 1,024 | 0.9013 / 486.6 |
| 2,048 | 0.9367 / 622.7 |
| 4,096 | 0.9631 / 523.4 |
| 8,192 | 0.9794 / 648.9 |
| 10,240 | 0.9840 / 721.2 |

在约 `0.978` recall 区间，当前方案为 `0.9785 / 904.6`，Curator ef=8192 为 `0.9794 / 648.9`：当前方案 recall 略低 `0.0009`，但 QPS 高约 `39%`。Curator 当前记录的最高 recall 为 `0.9840`，低于最终方案在 L=10,000 和 20,000 的 recall。

## HashPrune 与旧 UNG 路径

HashPrune 的最好代表 `hashprune_hp_mix128` 在四档为：

`0.9095/1039.9`、`0.9685/778.0`、`0.9817/535.9`、`0.9898/362.2`。

最终方案在四档都同时提高 recall 和 QPS。`hashprune_hp_navrepair` 的 L=20,000 结果为 `0.9879 / 327.6`，也低于当前方案的 `0.9940 / 406.2`。

原始 `UNG__hybrid` 在 L=20,000 只有 `0.7309` recall；当前 SpecialBlockTrie 路径已完全解决这一类低覆盖问题。旧 special-block 混合索引中的 CPU/GPU ELS 虽然可能有更高 QPS，但其 recall 曲线明显较低，不能只看吞吐排名。

## Pareto 结论

按当前结果目录的记录，可以得到以下判断：

1. **同索引最佳**：最终 SpecialBlockTrie Router。它支配 terminal-only、uncapped Router 和同索引 CPU ELS 的主要点，尤其在 recall `0.99+` 区间。
2. **最高 recall 前沿**：FAVOR 0.1/0.2/0.25/0.3，约 `0.9955-0.9960`。代价是独立索引、不同过滤参数和较低 QPS。
3. **低 Lsearch 高吞吐**：ACORN、CPU ELS special-block、GPU ELS special-block 在某些低 recall 区间更快；它们不能同时保持当前方案的高 recall。
4. **中高 recall 的平衡**：当前方案在 `0.978-0.994` 区间具有较好的折中；相对 NaviX、HashPrune、旧 UNG 和 Curator 的可比点更有优势。
5. **剩余瓶颈**：当前 L>=5,000 的边扫描中约 `98.5%-99.0%` 是 free special edges。继续减少 group 初始点的收益有限，下一步应优化 block 内 special-edge 访问和距离批处理。

### 搜索热路径优化更新

已实现并验证 free intra-edge cap：保留所有 inter-block edges，只将每个 free 节点的
intra-block special-edge 扫描限制为 16 条。反序 3-repeat A/B 中，它在 `Lsearch=2,500`
达到 `Recall=0.952714`、稳态约 `2,111 QPS`；原始路径在 `Lsearch=2,000` 为
`Recall=0.950786`、稳态约 `1,474 QPS`，提升约 `1.43x`。详细设计、heap/lazy 的负结果和
工作量对比见 [搜索热路径优化报告](SPECIAL_BLOCK_UNG_SEARCH_OPTIMIZATION_CN.md)。

## Block 大小 A/B 实验

为验证增大 block 是否能达到 `Recall >= 0.95` 下 3-5 倍提速，新增了
`UNG_SPECIAL_BLOCK_MIN_POINTS=2000/4000` 两个索引。除了该阈值外，构图参数与当前
`min_points=1000` 索引保持一致；搜索固定使用同一批 1,971 条查询、100 线程、3 次重复。
这里的阈值是自底向上切 block 时的最小未覆盖点数，不是 block 大小上限。

### 索引结构变化

| Min points | Block 数 | 直接 points 中位数 / 均值 | P90 / 最大值 | Special edges | Intra / Inter edges |
|---:|---:|---:|---:|---:|---:|
| 1,000 | 170 | 1,904 / 3,420 | 6,952 / 34,221 | 35.43M | 25.54M / 9.90M |
| 2,000 | 86 | 3,771 / 6,695 | 12,103 / 50,614 | 29.80M | 22.31M / 7.50M |
| 4,000 | 45 | 8,056 / 12,601 | 26,466 / 61,715 | 28.13M | 22.28M / 5.85M |

相对 1,000 阈值，2,000/4,000 阈值把 block 数减少 `49.4%/73.5%`，总 special edges
减少 `15.9%/20.6%`，inter-block edges 减少 `24.2%/40.9%`。但 2,000 到 4,000 时
intra edges 几乎不再下降，说明继续合并主要消除了 block 边界，没有同步降低 block 内
图的主体工作量。

### 相同 Lsearch 的速度与 Recall

| Min points | Lsearch | Recall | QPS | 相对 1,000 QPS | 每 query free blocks | Special edges/query | DistCalcs/query |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1,000 | 2,000 | 0.950786 | 1,668.7 | 1.00x | 49.5 | 96,968 | 27,589 |
| 2,000 | 2,000 | 0.941908 | 1,705.7 | 1.02x | 24.7 | 80,309 | 25,199 |
| 4,000 | 2,000 | 0.930188 | 1,996.7 | 1.20x | 12.7 | 77,078 | 21,505 |

增大 block 在固定预算下确实有效：4,000 阈值少做约 `22.1%` 距离计算，QPS 提高约
`19.7%`。但 Recall 同时下降 `0.0206`，已经不满足目标。原因不是入口 Trie 变慢，而是
block 合并后精确公共标签缩小、可用的 block 边界导航变粗；相同 beam 容量覆盖不到原来
的候选范围。

### Recall >= 0.95 的公平比较

| Min points | 首个达标 Lsearch | Recall | QPS | 相对 1,000 QPS |
|---:|---:|---:|---:|---:|
| 1,000 | 2,000 | 0.950786 | **1,668.7** | 1.00x |
| 2,000 | 2,500 | 0.950330 | 1,494.8 | 0.90x |
| 4,000 | 3,500 | 0.952968 | 1,431.3 | 0.86x |

为了补回 Recall，2,000 阈值需要把 Lsearch 增加 `25%`，4,000 阈值需要增加 `75%`。
这会逆转固定预算下的节省：

| Min points | DistCalcs/query | Free edges/query | Free nodes expanded/query | Beam shifted candidates/query |
|---:|---:|---:|---:|---:|
| 1,000 | 27,589 | 96,968 | 1,675 | 4.13M |
| 2,000 | 28,352 | 99,968 | 2,030 | 5.69M |
| 4,000 | 28,948 | 134,423 | 2,770 | 8.40M |

因此全局增大 block 不能作为目标方案。当前目录最强 baseline 约为 `1,644.5 QPS @
0.954744 recall`；本轮 1,000 阈值在最低达标点为 `1,668.7 QPS @ 0.950786`，仅约
`1.01x`，远低于 3-5 倍目标。2,000 和 4,000 阈值在达标点反而比 1,000 分别慢约
`10.4%` 和 `14.2%`。

后续不应继续全局提高 `MIN_POINTS`。若要利用大 block，应只在宽标签、高扇出的区域做
自适应合并，同时保留选择性强的子 block；更优先的主线仍是 lazy block activation、
减少 free-edge 扫描，以及替换有序 vector beam。

## 结果文件索引

- 当前最终方案：[search_time_summary_qps.csv](/home/dev/graphdb/FilterVectorResult/Amazon/results/special_trie_router16_group_cap256_frontier8_preexpand2_repeat3/query_selected_recall_advantage_1000_1000_20000/results/search_time_summary_qps.csv)
- 同索引 CPU ELS：[search_time_summary_qps.csv](/home/dev/graphdb/FilterVectorResult/Amazon/results/cpu_els_same_index_repeat3/query_selected_recall_advantage_1000_1000_20000/results/search_time_summary_qps.csv)
- Terminal-only：[search_time_summary_qps.csv](/home/dev/graphdb/FilterVectorResult/Amazon/results/special_trie_baseline_repeat3/query_selected_recall_advantage_1000_1000_20000/results/search_time_summary_qps.csv)
- ACORN：[search_time_summary_qps.csv](/home/dev/graphdb/FilterVectorResult/Amazon/results/ACORN/query_selected_recall_advantage_1000_1000_20000/results/search_time_summary_qps.csv)
- NaviX：[search_time_summary_qps.csv](/home/dev/graphdb/FilterVectorResult/Amazon/results/NaviX/query_selected_recall_advantage_1000_1000_20000/results/search_time_summary_qps.csv)
- FAVOR 0.25：[search_time_summary_qps.csv](/home/dev/graphdb/FilterVectorResult/Amazon/results/favor_0.25/query_selected_recall_advantage_1000_1000_20000/results/search_time_summary_qps.csv)
- Curator ef sweep：[search_time_summary_qps.csv](/home/dev/graphdb/FilterVectorResult/Amazon/results/Curator/query_selected_recall_advantage_search_ef64_search_ef10240/results/search_time_summary_qps.csv)
- GPU ELS special blocks：[search_time_summary_qps.csv](/home/dev/graphdb/FilterVectorResult/Amazon/results/gpu_bruteforce_els_special_blocks/query_selected_recall_advantage_1000_1000_20000/results/search_time_summary_qps.csv)
- CPU ELS special blocks：[search_time_summary_qps.csv](/home/dev/graphdb/FilterVectorResult/Amazon/results/cpu_bruteforce_els_special_blocks/query_selected_recall_advantage_1000_1000_20000/results/search_time_summary_qps.csv)
- HashPrune hp_mix128：[search_time_summary_qps.csv](/home/dev/graphdb/FilterVectorResult/Amazon/results/hashprune_hp_mix128/query_selected_recall_advantage_1000_1000_20000/results/search_time_summary_qps.csv)
- Block size A/B, min_points=1000：[search_time_summary_qps.csv](/home/dev/graphdb/FilterVectorResult/Amazon/results/special_trie_block_minp1000_ab/query_selected_recall_advantage_500_250_4000/results/search_time_summary_qps.csv)
- Block size A/B, min_points=2000：[search_time_summary_qps.csv](/home/dev/graphdb/FilterVectorResult/Amazon/results/special_trie_block_minp2000_ab/query_selected_recall_advantage_500_250_4000/results/search_time_summary_qps.csv)
- Block size A/B, min_points=4000：[search_time_summary_qps.csv](/home/dev/graphdb/FilterVectorResult/Amazon/results/special_trie_block_minp4000_ab/query_selected_recall_advantage_500_250_4000/results/search_time_summary_qps.csv)

报告生成日期：2026-08-11。
