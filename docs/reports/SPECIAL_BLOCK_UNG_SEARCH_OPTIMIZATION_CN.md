# Special Block UNG 搜索热路径优化实验

## 结论

在 Amazon `query_selected_recall_advantage`（1,971 queries，K=10，100 threads）上，当前
最有效的改动是限制每个 free 节点最多扫描 16 条 **intra-block** special edges，同时保留
全部 inter-block edges：

| 路径 | Lsearch | Recall | 稳态 QPS | 相对同轮 baseline |
|---|---:|---:|---:|---:|
| 原始 ordered queue | 2,000 | 0.950786 | 1,474.5 | 1.00x |
| ordered + intra cap 16 | 2,500 | 0.952714 | **2,111.4** | **1.43x** |

稳态值取各方法的 repeat 2/3 均值；cap16-first、baseline-second 的反序 A/B 结果在
`special_trie_ordered_intra_cap16_reverse_repeat3` 和
`special_trie_ordered_baseline_reverse_repeat3` 中。此前正序复测得到约 `2,216/1,526 QPS`，
所以 1.43x 是较保守的结论。

当前目录中 Recall>=0.95 的既有最强 baseline 约为 `1,644.5 QPS @ 0.954744`。cap16 的
`2,111.4 QPS @ 0.952714` 约为其 `1.28x`，尚未达到 3-5x 目标。下一阶段的主要瓶颈仍是
large-L ordered beam 的候选搬移与剩余 free-edge 路径，而不是 Trie 入口。

## 实现

### 1. 可切换 CandidateQueue

新增 `SpecialCandidateQueue`，由每线程 `SearchCache` 复用：

- 默认使用原有的有序连续数组语义，保留 `(distance, id, free)` 排序、retained top-L、
  preexpand 标记和最终结果顺序。
- 设置 `UNG_SPECIAL_CANDIDATE_HEAP=1` 时，切换到 result max-heap 与 expansion min-heap，
  用 token 跳过已淘汰候选。
- heap 能将 `SpecialQueueShiftedCandidates` 从约 `4.13M/query` 降到零，但同一 cap32 工作量
  下稳态 QPS 约 `1,634`，低于 ordered 的约 `1,693`。连续内存的 `memmove` 在该负载下优于
  双 heap 的随机维护，故 heap 仅保留为实验开关，不作为默认路径。

### 2. Lazy block activation

设置 `UNG_SPECIAL_TRIE_LAZY_BLOCK_ACTIVATION=1` 后，只 seed free frontier block；
`UNG_SPECIAL_TRIE_LAZY_SEED_DEPTH` 可扩展到固定层数的 child block。未 seed 的 free block
仍然 eligible，但只有 regular portal 或 parent-child special edge 实际到达时才激活。

该逻辑正确降低了初始 activation。例如 L=2,000 时：

| 策略 | Recall | free seed retained | preexpanded seeds | searched blocks | free edges/query |
|---|---:|---:|---:|---:|---:|
| 原始 | 0.950786 | 62.4 | 51.3 | 49.5 | 96,968 |
| lazy depth 0 | 0.939472 | 14.7 | 3.7 | 35.8 | 96,953 |
| lazy depth 1 | 0.941502 | 25.8 | 14.7 | 37.4 | 96,107 |
| lazy depth 2 | 0.946525 | 42.5 | 31.5 | 41.1 | 96,337 |

它没有减少主循环 edge scan，因为 beam 最终仍展开接近 Lsearch 个 free 节点；为了恢复
Recall 还要提高 Lsearch。因此 lazy 保留为可选路由策略，默认关闭。

### 3. Free-edge fanout cap

`UNG_SPECIAL_FREE_INTRA_EDGE_SCAN_CAP=N` 只裁剪 block-local graph 的 intra edges，全部
inter-block edges 保留，从而不会人为切断父 block 到 child block 的激活链。构建与加载均
保证 intra edges 在前、inter edges 在后；查询用 `partition_point` 找到分界，仅遍历前 N
条 intra edge 和整个 inter suffix，不再逐条读取被裁剪的 adjacency record。

cap16 的等 Recall 工作量对比：

| 指标/query | 原始 L=2,000 | cap16 L=2,500 | 变化 |
|---|---:|---:|---:|
| Free edges scanned | 96,968 | 58,259 | -39.9% |
| Capped intra edges | 0 | 61,582 | - |
| Distance calculations | 27,589 | 20,593 | -25.4% |
| Free distance calculations | 24,946 | 17,919 | -28.2% |
| Queue insert attempts | 23,992 | 16,996 | -29.2% |
| Queue bound rejections | 19,278 | 11,606 | -39.8% |
| Queue shifted candidates | 4.13M | 5.76M | +39.3% |

cap16 需要将 Lsearch 从 2,000 提升到 2,500 才恢复 Recall，但即使 beam 搬移增加，edge 和
distance 的降低仍带来净收益。cap24、32、40、48 也完成了 A/B；其中 cap24 在
`0.951192` 时约 `1,822 steady QPS`，cap32 在 `0.952968` 时约 `1,693 steady QPS`，均低于
cap16 的最佳点。

## 未采用的策略

1. 每 block 限制 free node expansions：即使只对 free-block 数较多的查询启用，cap48 在
   L=4,000 仍只有约 0.9465 Recall。宽查询中的大 block 仍需深度局部探索。
2. Heap 替换 ordered vector：消除了搬移统计，但实际吞吐下降，原因是 heap 的额外
   `log L`、token 失效判断和较差缓存局部性。
3. 全局增大 block：已在 `SPECIAL_BLOCK_TRIE_AMAZON_BASELINE_COMPARISON_CN.md` 记录，
   提高阈值会在目标 Recall 下减速。

## 推荐运行配置

使用 [最终反序 A/B 配置](../../experiments/search_comparison/config_amazon_special_block_cap16_reverse_ab.json)，
或在当前 Router 配置中加入：

```text
UNG_SPECIAL_FREE_INTRA_EDGE_SCAN_CAP=16
UNG_SPECIAL_CANDIDATE_HEAP unset
UNG_SPECIAL_TRIE_LAZY_BLOCK_ACTIVATION unset
Lsearch=2500
```

结果文件：

- cap16：[search_time_summary_qps.csv](/home/dev/graphdb/FilterVectorResult/Amazon/results/special_trie_ordered_intra_cap16_reverse_repeat3/query_selected_recall_advantage_2000_500_2500/results/search_time_summary_qps.csv)
- baseline：[search_time_summary_qps.csv](/home/dev/graphdb/FilterVectorResult/Amazon/results/special_trie_ordered_baseline_reverse_repeat3/query_selected_recall_advantage_2000_500_2500/results/search_time_summary_qps.csv)
