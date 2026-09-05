# Agent 看板

最后更新：`2026-09-05 22:31 +0800`
分支：`codex/multilevel-special-block-20260905`
检查点：`a1cd432`（push：`未推送；origin 指向用户的脏工作树，不直接 push`）

## 目标

在隔离 checkout 中完善多层 Special Block 构建与查询；在 Amazon 100% x1 数据上调优层级阈值，并按真实查询选择率和相同 Recall 公平比较普通 UNG、单层与多层方案，形成完整数据表和分析。

## 当前状态

- 总体：`进行中`
- 摘要：两层实现正确，但 provenance 审计发现此前 overlay 基于 21,834-label hybrid 主图，而六档 query/GT 属于 30,723-label Amazon x1，导致 Recall 人为在约 0.52 饱和。该轮结果已停止并隔离；正在切换到数据一致的 482,387-group 主图重建 overlay。

## 进行中

- 在固定 Amazon x1 主图、query/GT、CPU brute-force ELS、100 threads 和搜索二进制快照下运行 single/T2 的 25%/50%/75% 查询初筛。

## 完成历史

- 创建隔离 checkout `/home/sunyahui/worktrees/FilterVectorCode_multilevel_special`；旧 Git 不支持 worktree，使用 shared clone。
- 实现中层 T1=1000、上层 T2=10000 的独立分区、双 ownership、分层边授权和候选 level upgrade — 证据：`1f0c1e6`。
- 修复新格式 loader 白名单和同一 point 多 level 重复占槽导致的 Recall 崩溃 — 证据：`368227e`。
- Amazon advantage workload 端到端 A/B：Recall 约 0.902 时 1.124x；Recall 约 0.934 时 1.331x — 证据：`WORKTREE_HANDOFF.md` 和 `runs/query_current_repeat5/`。
- 六档 workload 数据审计：实际平均选择率 0.499%、0.903%、9.907%、24.915%、49.971%、74.994%，每档 1000 queries，均已有精确 GT。
- 已新增断点续跑 runner、manifest、warm/cold 分离、Pareto 与等 Recall 汇总器；单元测试 3/3 通过 — 证据：`bfdd685`。
- 修复 plain UNG 的 CPU ELS warmup 条件并通过真实 sel_0p5 smoke；旧 7.29 s 首点已判为无效 — 证据：`e0c8c63`。
- 核心 sweep 已完成 plain 六档及 single 前四档；`single_1k/sel_25` 的 CSV 完整。后续检查发现旧 driver 实际仍存活，曾与新 tmux runner 短暂并发；受影响的 single sel_50/sel_75 已隔离并将重测。
- 单实例重测及全部 multi case 已完成：18 case、每 case 20 个 L、每 L 3 repeats；结构审计通过。plain Recall 跨 repeat 最大 spread 0.0029，single/multi 为 0。
- 汇总改为 workload 自适应质量目标：使用 baseline 在代表性 L 的实测 Recall，不插值、不外推；同时输出扫描内最大 Recall。
- 新增可恢复的 T2 build runner，显式清理继承的 `UNG_*`、锁定输出目录、使用 staging 原子发布并验证 Amazon x1/fingerprint/metadata；T2=10k 现有产物验证通过。
- 完成 T2=4k/25k/50k 构建与验证；upper block 数分别为 45/8/4，总 special edges 分别为 63,580,487/58,990,803/56,574,549。
- 完成 ELS/主图交叉审计：当前 hybrid 主图上 L=20k 的 `cpu_min_super_sets`/`cpu_bruteforce_els` Recall 为 0.398/0.436；旧 Trie_block 主图上为 0.724/0.800。历史高 Recall 来自主图/分组版本差异，不是旧 ELS 更好。
- 增加中层/上层覆盖、展开、边扫描和激活机制统计；轻量性能路径不执行额外计数循环 — 证据：`ab4a94a`，smoke CSV 97 列、1000 行可解析。
- 发现 hybrid 主图与六档 query/GT 数据版本错配：query `{1}` 在原始 Amazon labels 匹配 582,582 点，在 hybrid index labels 仅匹配 290,684 点；此前约 0.52 Recall 饱和不能作为论文质量结论。
- Amazon x1 正确主图上的 T2=4k/10k overlay 已通过标签重排语义与 fingerprint 校验；构建分别为 94.43 s/89.05 s，upper blocks 为 46/22。
- T2=25k/50k 也已完成并通过相同校验；构建分别为 92.02 s/94.40 s，upper blocks 为 8/5；四档复用验证全部通过。
- 查询 sweep 新增输入 provenance 校验和 content-addressed 只读搜索二进制快照，防止运行中重编译污染整轮结果；11 项 Python 单测通过 — 证据：`2ffcf3d`。

## 下一步

监控 `t2_query_amazon_x1`，完成后验证 15/15 case、固定 binary hash 和三次 repeats，再生成同 Recall 初筛表。

## 阻塞与问题

- 原始 checkout 有大量未提交/未跟踪文件；禁止直接 merge 或覆盖。
- 搜索进程第一个 L 的 repeat 0 仍有线程池/工作区冷启动，初筛主指标使用 repeats 1--2 的 warm mean；最终复测增加重复数。
- 远端磁盘使用率 94%，尚余约 835 GB；实验产物必须限制在必要矩阵，不复制主向量数据。
- `sunyahuia600-sunyahui` 未在本机配置；使用 `ssh -l sunyahui sunyahuia6000`。
- 旧系统 `ps -o pid=,etime=,stat=,cmd=` 输出异常曾造成 PID 10039 已退出的误判；已停止两个 runner，并给 runner 增加 output-root 独占锁。受资源竞争影响的结果位于 `runs/quarantine/20260905T2119_contention/`，禁止进入汇总。
- plain UNG 使用随机搜索路径，3 repeats 的 Recall 最大 spread 为 0.0029；所有等 Recall 结论需保留质量余量或在最终候选上增加 repeats，不能按 1e-4 差异排序。
- 两次旧 T2 初筛分别因运行中重编译、instrumentation 条件误放而隔离在 `runs/quarantine/20260905T2218_binary_rebuild/` 和 `runs/quarantine/20260905T2225_instrumentation_bug/`；禁止用于结论。
- `runs/quarantine/20260905T_current_label_mismatch/` 使用了错误的 21,834-label hybrid 主图；仅可作机制诊断，禁止用于 Recall/QPS 主张。

## 验证

- `ctest -R 'special_block_trie|special_block_free_state|special_candidate_queue|special_edge_io'` — `通过`：4/4。
- `cmake --build build_ung_rel -j16 --target build_special_block_index search_UNG_index` — `通过`。
- `git diff --check` — `通过`。
- `python3 experiments/multilevel_special/validate_selection_sweep.py ...` — `通过`：18/18 case，L 网格和 3 repeats 完整。
- `cd experiments/multilevel_special && python3 -m unittest -v test_multilevel_selection.py` — `通过`：11/11，含查询 provenance 与不可变二进制快照。

## 仅在需要时阅读的细节

- `WORKTREE_HANDOFF.md` — 需要理解实现语义、已有构建/查询结果和 merge-back 边界时阅读。
- `experiments/multilevel_special/EXPERIMENT_STATE.md` — 需要继续参数扫描、解释实验或恢复运行时阅读。

## 恢复说明

1. 先读本看板，再读当前进行中的实验状态文档。
2. 运行 `git status --short`，不要暂存 `runs/`。
3. 检查 `tmux` 会话 `t2_query_amazon_x1`、`runs/t2_query_screen_amazon_x1/manifest.json` 和 `driver.log`。
4. 从“下一步”继续；新实测结果必须先写实验账本，再更新结论。

## 清理提示

- `runs/` 是未提交实验 artifact；最终只提交配置、汇总表和必要报告。
- 不再适用的旧 review-ready 看板已由本看板替代。
