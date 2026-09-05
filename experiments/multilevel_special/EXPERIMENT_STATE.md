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

候选上层阈值：4000、10000、25000、50000。四档均已按相同参数构建并通过 fingerprint、metadata 与必要文件检查。根据查询初筛结果决定是否需要细化到 6k/8k/16k。

系统级 baseline 在核心消融稳定后增加：至少包括现有普通 UNG/CPU ELS；若已有索引与当前数据/GT 可验证一致，再加入 NaviX、FaVOR 或现有 upstream route。不同 ELS、不同主图的方法必须单列，不能当作纯多层结构消融。

## Issue Ledger

| ID | 类型 | 状态 | 描述 | 当前证据 | 下一步 |
|---|---|---|---|---|---|
| M1 | measurement | RESOLVED | 六档选择率统一结果已完成 | 18 case x 20 L x 3 repeats；验证器通过 | 保留初筛表，最终候选增加 repeats |
| M2 | tuning | ACTIVE | 上层 T2=10k 未证明全局最优 | 四档 T2 已构建；查询仅完整扫描过 10k | 在 25%/50%/75% 初筛 4k/10k/25k/50k |
| M3 | baseline | ACTIVE | 其他系统方法尚未统一数据与 Recall 口径 | 历史结果路径混杂 Amazon/Amazon_hybrid | 校验 index fingerprint 和 GT 后运行 |
| M4 | correctness | RESOLVED | 多 level 候选重复占槽 | `368227e` 后 Recall 恢复 | 保留回归测试 |
| M5 | compatibility | RESOLVED | loader 不接受 multilevel format | 格式白名单和 round-trip test | 保留回归测试 |
| M6 | interpretation | RESOLVED | 同 L 慢是否否定多层 | 同 L Recall 更高；等 Recall 已有 1.124x--1.331x | 后续只按等 Recall主张 |
| M7 | measurement | RESOLVED | plain UNG 的 CPU ELS warmup 曾被错误绑定到 Special Block 开关 | 修复后 CPU ELS 在计时前预热 156 ms；首个 L50 从 7.29 s 降到 264 ms，warm repeats 13.6--14.5 ms | 主表继续用 warm mean，并保留 all-repeat |
| M8 | measurement | ACTIVE | 搜索线程池/工作区仍使每进程第一个 L 的 repeat 0 偏高 | 修复 ELS warmup 后，sel_0p5 L50 cold 264 ms、warm 约 14 ms | 初筛按 warm mean，最终复测增加独立 warmup 或丢弃 repeat 0 |
| M9 | orchestration | RESOLVED | 旧系统异常的 `ps -o` 输出造成旧 driver 已退出的误判，两个 runner 曾短暂并发 | 进程树确认 PID 10039 在跑 sel_75、新 tmux 在跑 sel_50；已同时停止，并隔离受影响产物 | runner 增加 output-root 独占锁；只用重新单独运行的 sel_50/sel_75 |
| M10 | measurement | PARTIAL | plain UNG Recall 跨 repeat 有随机波动 | 最大 spread 0.0029；single/multi 在该矩阵中为 0 | 等 Recall 保留质量 margin；最终点增加 repeats 并报告范围 |
| M11 | algorithm | ACTIVE | 10k 上层对低选择率常有额外开销、对高选择率显著有利 | 相对 single：0.5%--10% 多数 speedup <1；50%/75% 多点为 1.06--3.19x | 扫 T2，并考虑按完整上层覆盖/选择率门控启用 |
| M12 | baseline | RESOLVED | 历史 UNG 50% Recall 显著高于当前矩阵，疑似 ELS 语义退化 | 交叉实验表明同一旧主图上 bitset ELS Recall 反而更高；旧结果使用 482,387-group/30,723-label 主图，当前使用 510,639-group/21,834-label hybrid 主图 | 历史结果只作系统级旁证，禁止混入 overlay 消融 |
| M13 | measurement | RESOLVED | T2 初筛曾受到运行中重编译和错误 instrumentation 条件影响 | 两轮结果已分别隔离至 `runs/quarantine/20260905T2218_binary_rebuild/` 与 `runs/quarantine/20260905T2225_instrumentation_bug/`；最终修正后二进制为 `ab4a94a` | 只接受重新运行的 `runs/t2_query_screen` |
| M14 | mechanism | PARTIAL | 需要证明上层不是只增加静态数据而未参与查询 | 10k、50% detail smoke：每查询平均覆盖 5.089 个上层 block、展开 52.207 个上层节点、扫描 2,834.590 条上层边、发生 46.101 次上层激活 | 最终候选另做不计入性能主表的 detail run，并报告分位数 |

## 当前假设

- H1（中高置信）：多层收益主要来自以更小 L 达到相同 Recall，而非降低单次 expansion 成本。
- H2（中高置信）：中高选择率更容易完整覆盖上层 block，因此多层收益更明显；极低选择率收益小或为负。六档 sweep 与 10k/50% activation counters 均支持该解释，仍需最终候选的多选择率分位数验证。
- H3（待验证）：T2 太小会引入过多边和扫描开销，T2 太大则 block 太少、导航收益不足，存在 workload-dependent 中间最优值。

## 下一最小实验及判据

等待基于最终 `ab4a94a` 二进制的 `runs/t2_query_screen` 完成；先做结构验证和相同 Recall 汇总，再按结果选一个或两个 T2，在低选择率验证是否需要运行时门控上层。
