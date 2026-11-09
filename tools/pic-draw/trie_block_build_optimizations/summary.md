# 新版 Trie block 四组构建对比

已完成 32/32 个数据集/方式；状态：complete。

各组均使用删除辅助预计算后的相同构建器。GPU+LB+TopK 复用前一批次的八个样本，CPU、GPU、GPU+LB 重新测量；两批次的构建器 SHA256、输入数据及公共构建配置已核对。
每组每数据集 1 次测量，无独立预热。未完成项留空，不填 0，不使用历史旧版数据补齐。

端到端墙钟包含完整基础 Trie 与块索引的加载、构建、保存；编译、GPU 锁等待、校验、汇总及样本正文清理在计时之外。
单层 500:trie、出度 32、跨块边数 4、Lbuild=100、alpha=1.2、60 CPU 线程；各组块间连边均为 CPU 并统一预分配。
GPU 组共用 GNN-Descent 和 FastGrnnd 剪枝，负载均衡为 4 stream；topK 为阈值过滤、局部 Bitonic 排序与二分排名归并。

## 端到端构建时间（秒）

| 数据集 | CPU | GPU | GPU+LB | GPU+LB+TopK |
|---|---|---|---|---|
| Genome | 39.437 | 22.633 | 22.016 | 22.118 |
| Reviews | 68.579 | 14.209 | 15.700 | 13.754 |
| Amazon | 286.860 | 52.829 | 52.694 | 49.811 |
| VImg | 257.277 | 36.245 | 36.836 | 36.438 |
| Music | 956.817 | 118.646 | 118.358 | 117.799 |
| BReviews | 1562.447 | 218.569 | 216.657 | 212.172 |
| Tiktok | 813.320 | 217.815 | 222.253 | 209.810 |
| Laion | 15106.515 | 3397.377 | 3383.759 | 3334.661 |

## 块内图构建时间（秒）

| 数据集 | CPU | GPU | GPU+LB | GPU+LB+TopK |
|---|---|---|---|---|
| Genome | 18.128 | 0.749 | 0.646 | 0.511 |
| Reviews | 55.753 | 1.440 | 1.295 | 1.002 |
| Amazon | 239.281 | 4.503 | 3.862 | 3.279 |
| VImg | 222.942 | 3.896 | 3.560 | 2.845 |
| Music | 847.007 | 10.675 | 9.920 | 8.515 |
| BReviews | 1357.136 | 13.930 | 12.895 | 10.954 |
| Tiktok | 605.609 | 15.597 | 16.783 | 11.759 |
| Laion | 11774.883 | 122.294 | 125.709 | 99.792 |

## 观测结果（单次测量）

- CPU→GPU 端到端加速：1.74–8.06 倍。
- GPU→GPU+LB+TopK 块内加速：1.23–1.46 倍。
- 仅负载均衡基础上再加 topK，块内图变快：Genome、Reviews、Amazon、VImg、Music、BReviews、Tiktok、Laion；变慢：无。
- 仅负载均衡基础上再加 topK，端到端变快：Reviews、Amazon、VImg、Music、BReviews、Tiktok、Laion；变慢：Genome。

以上是各样本的实测结果；跨批次和单次测量仍有系统波动，不证明微小收益稳定，也不证明各近似图搜索质量相同。

## 数据及绘图

- `current_four_methods.csv`：已完成样本的长表，包含各阶段时间、样本数和来源。
- `current_four_methods_wide.csv`：八数据集四组宽表，时间单位秒，未完成单元格为空。
- `current_speedups.csv`：仅对已完成的配对计算加速比；大于 1 为变快。
- 完整 32 个样本通过验证后，自动生成三张 PNG/PDF 并更新 `current_source_manifest.json`。
- 绘图目录：`/home/dev/graphdb/FilterVectorCode_multilevel_special/tools/pic-draw/pic/trie_block_build_optimizations`。

历史结果保留在 `historical_four_methods*.csv` 和 `historical_summary.md`，不与新版合并。
