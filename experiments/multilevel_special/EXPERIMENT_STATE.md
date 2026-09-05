# 多层 Special Block 实验状态

本文只维护会影响后续决策的实验状态。完整设计和已完成的 1k/10k 实验见仓库根目录 `WORKTREE_HANDOFF.md`。

## 当前指标与公平口径

- 主指标：相同端到端 Recall 下的 batch time / QPS。
- 次指标：固定 L 下 Recall、距离计算量、构建时间、索引大小和加载时间。
- ELS 控制：核心图结构 A/B 固定使用 `cpu_bruteforce_els`，且允许计时前 warmup/reuse；这样差异来自 graph overlay，而不是入口组算法。
- 数据：Amazon 原始 100% x1，K=10，100 search threads，每个选择率 workload 使用相同 query 和精确 GT。
- 初筛：每点 3 repeats、宽 L 网格；只对 Pareto 前沿和精细 Recall 匹配点做 5+ repeats。
- 冷启动：同时保存 all-repeat mean 与排除 repeat 0 的 warm mean；主性能结论优先 warm mean，all-repeat 用于完整系统视角。

## Workload 清单

| 名称 | queries | 实际平均候选点 | 实际平均选择率 | 范围 |
|---|---:|---:|---:|---:|
| query_minlen5_avgsel05pct | 1000 | 3,007.741 | 0.499% | 20--10,810 |
| query_minlen5_avgsel1pct | 1000 | 5,440.481 | 0.903% | 20--29,771 |
| query_minlen3_avgsel10pct | 1000 | 59,682.701 | 9.907% | 6,045--143,527 |
| query_minlen2_avgsel25pct | 1000 | 150,103.543 | 24.915% | 1--548,561 |
| query_minlen1_avgsel50pct | 1000 | 301,053.926 | 49.971% | 794--582,582 |
| query_minlen1_avgsel75pct | 1000 | 451,802.516 | 74.994% | 5,759--582,582 |

注意：25%/50%/75% workload 的单 query 分布很宽，不能只用目录名解释所有结果；最终需要同时报告平均值和 coverage 分位数。

## 方法矩阵

核心消融：

1. `plain`：相同普通 UNG 主图，无 block overlay。
2. `single_1k`：中层 T1=1000。
3. `multi_1k_T2`：中层固定 T1=1000，上层阈值扫描。

候选上层阈值：4000、10000、25000、50000。10k 已构建，其余待构建。根据初筛结果决定是否需要细化到 6k/8k/16k。

系统级 baseline 在核心消融稳定后增加：至少包括现有普通 UNG/CPU ELS；若已有索引与当前数据/GT 可验证一致，再加入 NaviX、FaVOR 或现有 upstream route。不同 ELS、不同主图的方法必须单列，不能当作纯多层结构消融。

## Issue Ledger

| ID | 类型 | 状态 | 描述 | 当前证据 | 下一步 |
|---|---|---|---|---|---|
| M1 | measurement | ACTIVE | 尚无六档选择率统一结果 | 仅有 selected_recall_advantage | 跑核心矩阵 |
| M2 | tuning | ACTIVE | 上层 T2=10k 未证明全局最优 | 仅构建/查询过 10k | 扫描 4k/25k/50k |
| M3 | baseline | ACTIVE | 其他系统方法尚未统一数据与 Recall 口径 | 历史结果路径混杂 Amazon/Amazon_hybrid | 校验 index fingerprint 和 GT 后运行 |
| M4 | correctness | RESOLVED | 多 level 候选重复占槽 | `368227e` 后 Recall 恢复 | 保留回归测试 |
| M5 | compatibility | RESOLVED | loader 不接受 multilevel format | 格式白名单和 round-trip test | 保留回归测试 |
| M6 | interpretation | RESOLVED | 同 L 慢是否否定多层 | 同 L Recall 更高；等 Recall 已有 1.124x--1.331x | 后续只按等 Recall主张 |
| M7 | measurement | ACTIVE | plain UNG 的 CPU ELS warmup 曾被错误绑定到 Special Block 开关 | sel_0p5 首次 L50 为 7.29 s，后续约 14 ms；代码条件要求 `UNG_SPECIAL_BLOCK_SEARCH` | 移除错误条件并重跑旧 smoke |

## 当前假设

- H1（中高置信）：多层收益主要来自以更小 L 达到相同 Recall，而非降低单次 expansion 成本。
- H2（待验证）：中高选择率更容易完整覆盖上层 block，因此多层收益更明显；极低选择率可能无法激活上层，收益较小或为负。
- H3（待验证）：T2 太小会引入过多边和扫描开销，T2 太大则 block 太少、导航收益不足，存在 workload-dependent 中间最优值。

## 下一最小实验及判据

先用 plain、single_1k、multi_1k_10k 在六档选择率上扫描同一 L 网格。若多层在至少两个选择率区间存在更优的 Recall-time Pareto 点，再扩展 T2 参数；若完全没有 Pareto 收益，先用 detailed stats 判断是覆盖率不足还是边扫描开销过高。
