# 多层 Special Block 目标完成审计

本文件把用户目标逐项映射到可检查的代码、实验与文档。只有所有必需项都有当前 Amazon x1 的直接证据时，任务才可标记完成。

## 目标拆解与证据

| 显式要求 | 验收标准 | 当前证据 | 状态 / 缺口 |
|---|---|---|---|
| 独立文件夹开发 | 不修改原始脏工作树；独立分支可审阅 | shared clone `/home/sunyahui/worktrees/FilterVectorCode_multilevel_special`；分支 `codex/multilevel-special-block-20260905` | 完成 |
| 多层 block 构建 | T1 中层保留，并叠加独立 T2 上层；双 ownership、父子关系、持久化可用 | `uni_nav_graph_special_blocks.cpp`、`ung_special_block_trie.*`；提交 `1f0c1e6`、`368227e` | 完成 |
| 多层查询 | 候选按普通 0 -> 中层 1 -> 上层 2 单调激活；不能越级；同 point 去重升级 | `ung_special_block_activation.h`、`ung_special_candidate_queue.*`、`uni_nav_graph_search_backend.cpp` | 完成；focused C++ 4/4 |
| 数据集测试 | 只使用 Amazon 原始 100% x1，query、labels、GT 和主图 provenance 一致 | runner 的 SHA/fingerprint gate；30,723 labels、602,453 points | 25/50/75% 完成；0.5/0.9/9.9% 正在补跑 |
| 参数调优 | 至少扫描 T2=4k/10k/25k/50k 和足够密的 Lsearch；只选实测点 | `config.amazon_x1_build_sweep.json`、dense/final configs、source CSV | 高选择率完成；低选择率正在补跑 |
| 普通/单层/多层比较 | 同主图、ELS、query/GT、线程数与 timing 口径，按同 Recall 比较 | `run_selection_sweep.py` 与正式 source CSV | 25/50/75% 有 single/multi；plain 尚未进入可提交正式主表，低选择率待补 |
| 各种选择率 | 覆盖现有六档真实 workload：0.499%、0.903%、9.907%、24.915%、49.971%、74.994% | query/GT 文件及 SHA；`EXPERIMENT_STATE.md` workload 表 | 后三档完成；前三档旧结果来自错误主图，正在重测 |
| 外部方法比较 | FAVOR、NaviX、Curator、ACORN 使用相同 Amazon x1 query/GT、K=10、1000 queries、100 threads，并明确 timing 边界 | 各 baseline runner、source CSV 和 ACORN correctness patch | 25/50/75% 完成；低选择率待补；跨系统仅作离散 Pareto 系统位置对照 |
| 完整数据表和分析 | 可提交完整 sweep、主表、构建表、负结果、适用边界；数字可由脚本重算 | `generate_paper_results.py`、`paper_results.csv`、多层报告 | 当前主表仅三档且构建字段口径需修正 |
| 可复现性 | 记录 commit、命令、配置、数据 checksum、硬件、repeats、raw/aggregate 路径 | config/manifest/command/environment/source CSV | 部分完成；需增加统一复现清单和六档 source CSV |
| 审阅就绪 | 两轮结构审阅、一轮有限上下文交付审阅，阻塞问题处理或明确降级 | review-ready 审阅流程 | 第一轮进行中 |

## 当前不能使用的证据

- `runs/multilevel_selection/` 的六档结果绑定 21,834-label hybrid 主图，只能作历史机制诊断。
- `runs/quarantine/` 内所有并发污染、运行中换 binary、instrumentation bug 或 label mismatch 结果均不得进入论文表。
- 固定 L 的耗时不能作为加速结论；主结果必须在共同 Recall 门槛下选择离散实测点。
- Special Block 的 stage/overlay 时间与其完整 builder wall time必须分列，不能混称。

## 当前进行中的最小补充实验

在正确的 30,723-label Amazon x1 主图上，以同一不可变 search binary、100 threads、1000 queries，运行 plain、single T1=1k、multi T2=4k/10k/25k/50k 的 0.5%/0.9%/9.9% 宽 L 网格。先用 3 repeats 筛选，再对共同 Recall/Pareto 候选做 7 repeats。
