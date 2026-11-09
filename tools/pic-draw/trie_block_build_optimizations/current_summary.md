# 新版 Trie block 三组构建对比

本表包含已完成的 24/24 个数据集/方式组合；状态：complete。

CPU、GPU+LB+TopK 沿用原表的八数据集实测结果；GPU 列替换为新测的 GPU-Naive，删除单独的 GPU+LB 列。
GPU-Naive 来自独立构建器和测量批次，输入数据及公共构建配置已核对；不再声称三组使用同一构建器二进制。
每组每数据集 1 次测量，无独立预热。

端到端墙钟包含完整基础 Trie 与块索引的加载、构建、保存；编译、GPU 锁等待、校验、汇总及样本正文清理在计时之外。
单层 500:trie、出度 32、跨块边数 4、Lbuild=100、alpha=1.2、60 CPU 线程；各组块间连边均为 CPU 并统一预分配。
GPU-Naive 使用基础距离内核、单流、完整排序和完整候选回传；GPU+LB+TopK 保留已有 GPU 优化，并启用 4 stream 负载均衡及阈值过滤、局部 Bitonic 排序与二分排名归并的 topK 优化。

## 端到端构建时间（秒）

| 数据集 | CPU | GPU-Naive | GPU+LB+TopK |
|---|---|---|---|
| Genome | 39.437 | 22.822 | 22.118 |
| Reviews | 68.579 | 15.705 | 13.754 |
| Amazon | 286.860 | 59.858 | 49.811 |
| VImg | 257.277 | 43.194 | 36.438 |
| Music | 956.817 | 133.030 | 117.799 |
| BReviews | 1562.447 | 234.526 | 212.172 |
| Tiktok | 813.320 | 236.320 | 209.810 |
| Laion | 15106.515 | 3669.595 | 3334.661 |

## 块内图构建时间（秒）

| 数据集 | CPU | GPU-Naive | GPU+LB+TopK |
|---|---|---|---|
| Genome | 18.128 | 1.671 | 0.511 |
| Reviews | 55.753 | 3.032 | 1.002 |
| Amazon | 239.281 | 11.187 | 3.279 |
| VImg | 222.942 | 9.225 | 2.845 |
| Music | 847.007 | 25.926 | 8.515 |
| BReviews | 1357.136 | 34.874 | 10.954 |
| Tiktok | 605.609 | 38.249 | 11.759 |
| Laion | 11774.883 | 282.576 | 99.792 |

## CPU 构图阶段耗时与占比

沿用上表相同的八个 CPU 样本（每数据集 1 次），无需重跑。下表括号内为占比，分母为三个阶段耗时之和：`T三阶段 = T底层 + T块内 + T跨块`。该分母只用于比较这三个阶段，不等于完整端到端时间。

- 底层图：`cpu_base_build_timer_s`，来自基础索引元数据的 `index_time_add_rb(ms) / 1000`；包括基础 Trie 分组、组内图和基础组间边等构建，不含基础数据文件加载与索引保存。
- 块内图：`cpu_intra_s`，来自 `special_edge_intra_build_time(ms) / 1000`，对应各 block 内部图的构建；不包含 block 划分等元数据准备。
- 跨块边：`cpu_inter_s`，来自 `special_edge_inter_build_time(ms) / 1000`，对应 block 之间的连边；与底层图中的基础组间边属于不同阶段。

以上均为 CPU 构建流程中的阶段墙钟耗时，不是所有 CPU 线程累计的 CPU 时间。

| 数据集 | 底层图：秒（占比） | 块内图：秒（占比） | 跨块边：秒（占比） | 三阶段合计（秒） |
|---|---:|---:|---:|---:|
| Genome | 5.082（21.18%） | 18.128（75.55%） | 0.786（3.27%） | 23.995 |
| Reviews | 2.399（4.02%） | 55.753（93.40%） | 1.541（2.58%） | 59.693 |
| Amazon | 9.838（3.81%） | 239.281（92.68%） | 9.063（3.51%） | 258.182 |
| VImg | 8.051（3.37%） | 222.942（93.32%） | 7.907（3.31%） | 238.900 |
| Music | 20.925（2.32%） | 847.007（94.02%） | 32.941（3.66%） | 900.873 |
| BReviews | 38.571（2.64%） | 1357.136（92.94%） | 64.544（4.42%） | 1460.252 |
| Tiktok | 86.140（11.59%） | 605.609（81.49%） | 51.412（6.92%） | 743.161 |
| Laion | 253.460（1.76%） | 11774.883（81.67%） | 2388.884（16.57%） | 14417.227 |

为与前面的端到端时间表对齐，另列以 `cpu_end_to_end_s` 为分母的占比。其余开销按 `T端到端 − T三阶段` 计算，包含数据加载、保存、block 元数据/Trie 准备及其他未单列开销；不能将其全部归为 I/O。

| 数据集 | 底层图 / 端到端 | 块内图 / 端到端 | 跨块边 / 端到端 | 其余开销 / 端到端 |
|---|---:|---:|---:|---:|
| Genome | 12.89% | 45.97% | 1.99% | 39.16% |
| Reviews | 3.50% | 81.30% | 2.25% | 12.96% |
| Amazon | 3.43% | 83.41% | 3.16% | 10.00% |
| VImg | 3.13% | 86.65% | 3.07% | 7.14% |
| Music | 2.19% | 88.52% | 3.44% | 5.85% |
| BReviews | 2.47% | 86.86% | 4.13% | 6.54% |
| Tiktok | 10.59% | 74.46% | 6.32% | 8.63% |
| Laion | 1.68% | 77.95% | 15.81% | 4.56% |

按三阶段之和归一化，块内图构建在八个数据集上占 75.55%–94.02%，均为主要耗时；Laion 的跨块边构建为 2388.884 秒，占三阶段的 16.57%（端到端的 15.81%）。底层图占比在 Genome（21.18%）和 Tiktok（11.59%）上相对较高。百分比用未舍入秒数计算，展示值可能因四舍五入不恰好合计 100%。

统计明细：[current_cpu_build_breakdown.csv](current_cpu_build_breakdown.csv)。来源为 `current_four_methods_wide.csv` 的 CPU 列，并已逐项核对 `current_four_methods.csv` 中八个 CPU `source_cases` 对应的基础/块索引 `meta` 和 `timing.json`；保持各数据集单独统计，不将不同数据集的时长混为平均占比。

## 观测结果（单次测量）

- CPU→GPU-Naive 端到端加速：1.73–7.19 倍。
- GPU-Naive→GPU+LB+TopK 块内加速：2.83–3.41 倍。
- GPU-Naive→GPU+LB+TopK 端到端加速：1.03–1.20 倍。
- 八个数据集的 GPU+LB+TopK 块内图和端到端时间均短于 GPU-Naive。

GPU-Naive 与最终版本还存在距离内核和回传方式等差异，因此上述收益反映完整优化组合，不能单独归因于负载均衡或 topK。
以上是各样本的实测结果；跨批次和单次测量仍有系统波动，不证明微小收益稳定，也不证明各近似图搜索质量相同。

## 数据来源

- GPU-Naive：[naive_vs_existing.csv](/home/dev/graphdb/FilterVectorResult/trie_block_gpu_naive/20261020_153118_339915_gpu_naive_eight/naive_vs_existing.csv)，取 `naive_end_to_end_s` 和 `naive_intra_s`；测量批次为 `20261020_153118_339915_gpu_naive_eight`。
- CPU、GPU+LB+TopK：本目录的 `current_four_methods_wide.csv`，保留原有实测值。
- 三组展示与结论保留于本 Markdown；`current_four_methods*.csv`、`current_speedups.csv`、来源清单及原四组 PNG/PDF 保留原样，不代表本表的三组展示。新增 CPU 阶段统计另存于 `current_cpu_build_breakdown.csv`。
- 原绘图目录：`/home/dev/graphdb/FilterVectorCode_multilevel_special/tools/pic-draw/pic/trie_block_build_optimizations`。

历史结果保留在 `historical_four_methods*.csv` 和 `historical_summary.md`，不与本表合并。
