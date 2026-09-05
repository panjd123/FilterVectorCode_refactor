# 多层 Special Block 实验状态

本文只维护会影响后续决策的实验状态。完整设计和已完成的 1k/10k 实验见仓库根目录 `WORKTREE_HANDOFF.md`。

## 当前指标与公平口径

- 主指标：相同端到端 Recall 下的 batch time / QPS。
- 次指标：固定 L 下 Recall、距离计算量、构建时间、索引大小和加载时间。
- ELS 控制：核心图结构 A/B 固定使用 `cpu_bruteforce_els`，且允许计时前 warmup/reuse；这样差异来自 graph overlay，而不是入口组算法。主图内嵌 labels、query labels 和 GT 必须来自同一 Amazon x1 版本。
- 数据：Amazon 原始 100% x1，K=10，100 search threads，每个选择率 workload 使用相同 query 和精确 GT。
- 初筛：每点 3 repeats、宽 L 网格；Pareto 前沿和精细 Recall 匹配点已经完成 7 repeats。
- 冷启动：同时保存 all-repeat mean 与排除 repeat 0 的 warm mean；主性能结论使用 warm median，并同时报告 warm mean/CV。

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

系统级 baseline 已接入 NaviX、FAVOR、Curator 和官方 ACORN。它们固定相同 Amazon x1 query/GT、K=10、1000 queries、100 threads；不同 ELS、主图和 route 单列，只用于系统位置比较，不能当作纯多层结构消融。

## Issue Ledger

| ID | 类型 | 状态 | 描述 | 当前证据 | 下一步 |
|---|---|---|---|---|---|
| M1 | measurement | RESOLVED | 六档选择率统一结果已完成 | 18 case x 20 L x 3 repeats；验证器通过 | 保留初筛表，最终候选增加 repeats |
| M2 | tuning | RESOLVED | 四档 T2 已完成粗网格、dense-L 和正式候选比较 | 25k/50k 在不同 workload/质量区间占优；4k 不进入最终候选 | 如论文需要再做局部阈值敏感性，不影响当前结论 |
| M3 | baseline | RESOLVED | 外部系统统一数据与 Recall 口径 | NaviX/FAVOR/Curator/官方 ACORN 均完成三档；主表按预声明 Recall 门槛选最快实测点 | 保留跨系统边界，不把不同图结构解释为局部消融 |
| M4 | correctness | RESOLVED | 多 level 候选重复占槽 | `368227e` 后 Recall 恢复 | 保留回归测试 |
| M5 | compatibility | RESOLVED | loader 不接受 multilevel format | 格式白名单和 round-trip test | 保留回归测试 |
| M6 | interpretation | RESOLVED | 同 L 慢是否否定多层 | 同 L Recall 更高；正式同 Recall 主表为 1.222x/1.530x/2.345x，高质量上限为 1.222x/4.925x/6.290x | 后续只按等 Recall 主张 |
| M7 | measurement | RESOLVED | plain UNG 的 CPU ELS warmup 曾被错误绑定到 Special Block 开关 | 修复后 CPU ELS 在计时前预热 156 ms；首个 L50 从 7.29 s 降到 264 ms，warm repeats 13.6--14.5 ms | 主表继续用 warm mean，并保留 all-repeat |
| M8 | measurement | RESOLVED | 100-thread 极短任务仍有 wall-time 长尾 | 15 repeats 中 Recall 完全稳定，低 L 的 CV 可达 0.273；正式中高质量点 CV 为 0.006--0.016 | 主结论报告 warm median/mean/CV，避免用单次低 L 时间 |
| M9 | orchestration | RESOLVED | 旧系统异常的 `ps -o` 输出造成旧 driver 已退出的误判，两个 runner 曾短暂并发 | 进程树确认 PID 10039 在跑 sel_75、新 tmux 在跑 sel_50；已同时停止，并隔离受影响产物 | runner 增加 output-root 独占锁；只用重新单独运行的 sel_50/sel_75 |
| M10 | measurement | RESOLVED | 最终候选需要增加 repeats 和质量余量 | 7-repeat 正式矩阵完成，汇总含 Recall margin、median speedup、L reduction | 主表同时给目标 Recall 与实际 Recall |
| M11 | algorithm | RESOLVED | 多层是否在不同选择率有效 | warm median：共同门槛为 1.222x/1.530x/2.345x；最高质量为 1.222x/4.925x/6.290x | 如实报告 25% 收益弱和外部负结果 |
| M12 | measurement | REGRESSED | 历史 UNG 50% Recall 显著高于当前矩阵 | provenance 审计发现 query/GT 属于 30,723-label Amazon x1，而当前 overlay 主图仅 21,834 labels；query `{1}` 匹配点为 582,582 vs 290,684 | 废弃该 overlay 的 Recall 结论，在 30,723-label 主图重建 |
| M13 | measurement | RESOLVED | T2 初筛曾受到运行中重编译和错误 instrumentation 条件影响 | 两轮结果已隔离；新 runner 把搜索程序复制为 content-addressed 只读快照并记录 SHA-256 | 只接受 `runs/t2_query_screen_amazon_x1` 且 hash 固定的结果 |
| M14 | mechanism | RESOLVED | 需要证明上层不是只增加静态数据而未参与查询 | upper on/off 对所有 workload/L 提升 Recall；10k/50% detail smoke 每查询平均展开 52.207 个上层节点、扫描 2,834.590 条上层边、发生 46.101 次上层激活 | detail counters 只解释机制，不计入性能主表 |
| M15 | provenance | RESOLVED | 所有核心 overlay 必须绑定主图 labels hash | 四档均通过 `new_to_old` 标签重排验证，绑定 602,453 points、482,387 groups 和 fingerprint `91d78580ae29f468` | 查询 runner 持续校验 hash/fingerprint |
| M16 | ablation | RESOLVED | 同一多层索引关闭 level 2 是否退化为单层语义 | upper on/off 完成；开启 level 2 对所有 workload/L 提升 Recall，证明上层真实参与 | 保留机制表，不与 fresh single 的系统时间混用 |
| M17 | measurement | RESOLVED | 粗 L 网格会误判等 Recall 性能 | 三档 dense-L 和 method-specific validator 已完成 | 正式结论只用实测匹配点，不插值外推 |
| M18 | provenance | RESOLVED | 外部系统索引与 GT 是否满足强比较要求 | NaviX labels hash、FAVOR x1 构建日志、Curator x1 隔离重建、ACORN x1 adapter/0 violations 均已留证 | 后续新增系统继续执行同一 provenance gate |
| M19 | measurement | RESOLVED | Curator 首轮时间随 `search_ef` 非单调 | 每 ef 已做 1 warmup + 5 measured repeats并保存 raw；反向预算 profile 将异常定位到 `prepare_filter`，不是 ANN search | 主表使用 median/CV；根因细化不阻塞当前系统对照 |
| M20 | provenance | RESOLVED | Curator 是否能绑定当前 x1 和统一 GT | 602,453 points、768D；隔离重建 102.207 s；持久化成功；smoke 320 IDs 无 filter 违规 | 保留 meta/hash和 smoke 证据 |
| M21 | correctness | RESOLVED | 官方 ACORN hybrid 初始 candidate 未检查 `filter_map` | nested clone `fb07f1d` 修复后所有纳入结果 0 filter violations，Recall 不变 | 保留补丁和 adapter smoke |
| M22 | tuning | RESOLVED | 单一 ACORN gamma 是否会造成不公平结论 | gamma=1/2/4/8/12 已构建和扫描；gamma 对 Recall 非单调，25%/50% 由 ACORN-1 达标，75% 最快达标点为 gamma12 | 表中标注 variant、ef、total/core 和重复次数 |

## 当前假设

- H1（高置信）：多层收益主要来自以更小 L 达到相同 Recall，而非降低同 L expansion 成本；正式矩阵中 L reduction 与 speedup 一致。
- H2（高置信）：完整覆盖上层 block 的 query 比例随选择率由 26.0% 增至 51.3%/77.3%，解释了 25% 收益弱而 50%/75% 收益强。
- H3（中高置信）：T2 存在 workload/质量相关最优点；小 T2 边多、扫描代价高，大 T2 block 少但跨越更远，当前 25k/50k 分别占优。

## 当前最终结果与后续方向

共同 Recall 门槛 `0.90/0.85/0.87` 下，多层相对单层的 warm-median speedup 为 `1.222x/1.530x/2.345x`；在单层最高质量附近为 `1.222x/4.925x/6.290x`。第二层的主要作用是降低达到同 Recall 所需的 L，不是降低固定 L 的单次扩展成本。完整表见 `docs/reports/MULTILEVEL_SPECIAL_BLOCK_PAPER_REPORT_CN.md`。

当前外部比较的主要负结果是 FAVOR 三档均更快；25% workload 上多层也慢于 NaviX/Curator。下一步若继续优化，应优先让 GPU batch scratch 携带 per-edge activation level、减少多层标量路径开销，并在完全相同 Recall 附近补更密的外部参数点；这些不是本轮实现正确性与主表交付的阻塞项。
