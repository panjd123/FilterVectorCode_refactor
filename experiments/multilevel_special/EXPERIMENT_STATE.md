# 多层 Special Block 实验状态

本文只维护会影响后续决策的实验状态。完整设计和已完成的 1k/10k 实验见仓库根目录 `WORKTREE_HANDOFF.md`。

## 当前指标与公平口径

- 主指标：相同端到端 Recall 下的 batch time / QPS。
- 次指标：固定 L 下 Recall、距离计算量、构建时间、索引大小和加载时间。
- ELS 控制：核心图结构 A/B 固定使用 `cpu_bruteforce_els`，且允许计时前 warmup/reuse；这样差异来自 graph overlay，而不是入口组算法。主图内嵌 labels、query labels 和 GT 必须来自同一 Amazon x1 版本。
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
| M2 | tuning | RESOLVED | 四档 T2 已完成粗网格、dense-L 和正式候选比较 | 25k/50k 在不同 workload/质量区间占优；4k 不进入最终候选 | 如论文需要再做局部阈值敏感性，不影响当前结论 |
| M3 | baseline | PARTIAL | 其他系统方法正在统一数据与 Recall 口径 | NaviX/FAVOR 三档粗网格完成；Curator 已从 x1 隔离重建并完成首轮三档；ACORN 尚不充分 | 完成 Curator 稳健复测、NaviX/FAVOR 局部等 Recall、ACORN 审计 |
| M4 | correctness | RESOLVED | 多 level 候选重复占槽 | `368227e` 后 Recall 恢复 | 保留回归测试 |
| M5 | compatibility | RESOLVED | loader 不接受 multilevel format | 格式白名单和 round-trip test | 保留回归测试 |
| M6 | interpretation | RESOLVED | 同 L 慢是否否定多层 | 同 L Recall 更高；等 Recall 已有 1.124x--1.331x | 后续只按等 Recall主张 |
| M7 | measurement | RESOLVED | plain UNG 的 CPU ELS warmup 曾被错误绑定到 Special Block 开关 | 修复后 CPU ELS 在计时前预热 156 ms；首个 L50 从 7.29 s 降到 264 ms，warm repeats 13.6--14.5 ms | 主表继续用 warm mean，并保留 all-repeat |
| M8 | measurement | PARTIAL | 100-thread 极短任务仍有 wall-time 长尾 | 15 repeats 中 Recall 完全稳定，低 L 的 CV 可达 0.273；中高质量点明显更稳定 | 主结论报告 warm median/mean/CV，避免用单次低 L 时间 |
| M9 | orchestration | RESOLVED | 旧系统异常的 `ps -o` 输出造成旧 driver 已退出的误判，两个 runner 曾短暂并发 | 进程树确认 PID 10039 在跑 sel_75、新 tmux 在跑 sel_50；已同时停止，并隔离受影响产物 | runner 增加 output-root 独占锁；只用重新单独运行的 sel_50/sel_75 |
| M10 | measurement | RESOLVED | 最终候选需要增加 repeats 和质量余量 | 7-repeat 正式矩阵完成，汇总含 Recall margin、median speedup、L reduction | 主表同时给目标 Recall 与实际 Recall |
| M11 | algorithm | RESOLVED | 多层是否在不同选择率有效 | 正式结果：25% 多数持平/减速、最高质量 1.220x；50% 最高 4.936x；75% 最高 6.274x | 如实报告适用区间，不筛掉负结果 |
| M12 | measurement | REGRESSED | 历史 UNG 50% Recall 显著高于当前矩阵 | provenance 审计发现 query/GT 属于 30,723-label Amazon x1，而当前 overlay 主图仅 21,834 labels；query `{1}` 匹配点为 582,582 vs 290,684 | 废弃该 overlay 的 Recall 结论，在 30,723-label 主图重建 |
| M13 | measurement | RESOLVED | T2 初筛曾受到运行中重编译和错误 instrumentation 条件影响 | 两轮结果已隔离；新 runner 把搜索程序复制为 content-addressed 只读快照并记录 SHA-256 | 只接受 `runs/t2_query_screen_amazon_x1` 且 hash 固定的结果 |
| M14 | mechanism | PARTIAL | 需要证明上层不是只增加静态数据而未参与查询 | 10k、50% detail smoke：每查询平均覆盖 5.089 个上层 block、展开 52.207 个上层节点、扫描 2,834.590 条上层边、发生 46.101 次上层激活 | 最终候选另做不计入性能主表的 detail run，并报告分位数 |
| M15 | provenance | RESOLVED | 所有核心 overlay 必须绑定主图 labels hash | 四档均通过 `new_to_old` 标签重排验证，绑定 602,453 points、482,387 groups 和 fingerprint `91d78580ae29f468` | 查询 runner 持续校验 hash/fingerprint |
| M16 | ablation | RESOLVED | 同一多层索引关闭 level 2 是否退化为单层语义 | upper on/off 完成；开启 level 2 对所有 workload/L 提升 Recall，证明上层真实参与 | 保留机制表，不与 fresh single 的系统时间混用 |
| M17 | measurement | RESOLVED | 粗 L 网格会误判等 Recall 性能 | 三档 dense-L 和 method-specific validator 已完成 | 正式结论只用实测匹配点，不插值外推 |
| M18 | provenance | ACTIVE | 外部系统索引与 GT 是否满足强比较要求 | NaviX 已有 labels hash；FAVOR 有 x1 构建日志；Curator/ACORN 尚缺完整输入 hash | 分系统校验或隔离重建，无法确认则降级为附录 |
| M19 | measurement | ACTIVE | Curator 首轮 5-repeat 只保存均值且时间随 `search_ef` 非单调 | x1 首轮三档完成；25% `ef=2048` 约 2.99 s、`ef=10240` 约 1.79 s；C++ 路径每 query 先构建临时过滤树 | 每 ef warmup 1 次并保存 5 个 raw repeats；再拆 filter/temp-index/search 时间 |
| M20 | provenance | RESOLVED | Curator 是否能绑定当前 x1 和统一 GT | 602,453 points、768D；隔离重建 102.207 s；持久化成功；smoke 320 IDs 无 filter 违规 | 保留 meta/hash和 smoke 证据 |

## 当前假设

- H1（高置信）：多层收益主要来自以更小 L 达到相同 Recall，而非降低同 L expansion 成本；正式矩阵中 L reduction 与 speedup 一致。
- H2（高置信）：完整覆盖上层 block 的 query 比例随选择率由 26.0% 增至 51.3%/77.3%，解释了 25% 收益弱而 50%/75% 收益强。
- H3（中高置信）：T2 存在 workload/质量相关最优点；小 T2 边多、扫描代价高，大 T2 block 少但跨越更远，当前 25k/50k 分别占优。

## 下一最小实验及判据

先完成 Curator raw-repeat 复测。判据是三档每个 ef 均有 1 warmup + 5 measured batches、Recall 与首轮一致、可计算 median/CV；若时间仍非单调，则用 C++ profile 的临时树构建与 ANN search 分项解释。之后补 NaviX/FAVOR 局部等 Recall 点并审计 ACORN；任何缺输入 provenance 或无法复现的系统只能放附录。
