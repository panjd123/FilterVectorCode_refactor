# Agent 看板

最后更新：`2026-09-06 01:36 +0800`
分支：`codex/multilevel-special-block-20260905`
检查点：`f53784b`（push：`未推送；origin 指向用户的脏工作树，不直接 push`）

## 目标

在隔离 checkout 中完善多层 Special Block 构建与查询；在 Amazon 100% x1 数据上调优层级阈值，并按真实查询选择率和相同 Recall 公平比较普通 UNG、单层与多层方案，形成完整数据表和分析。

## 当前状态

- 总体：`进行中`
- 摘要：两层实现和 Amazon x1 内部正式矩阵已完成。多层在高质量 50%/75% 查询上最高达到 4.936x/6.274x，25% 仅在最高质量点达到 1.220x、其余多为持平或减速。NaviX/FAVOR 粗网格已完成；Curator 已从 x1 隔离重建并通过 filter/Recall smoke，当前正在补 warmup 与逐 repeat 稳健计时。

## 进行中

- 修复 Curator runner 的稳健计时并复跑三档：每个 `search_ef` 先 warmup 1 批，再保存 5 个原始 batch wall times；随后拆分 filter preparation、临时树构建和 ANN search 时间。

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
- 正确 Amazon x1 初筛完成 18/18 case（plain/single/4k/10k/25k/50k × 25%/50%/75%），每点 3 repeats；统一验证和保守等 Recall 汇总通过 — 证据：`166fd06` 与 `runs/t2_query_screen_amazon_x1/summary/`。
- gate probe 表明上层完整覆盖 query 比例为 26.0%/51.3%/77.3%（25%/50%/75% workload）；绝对 covered-points threshold 区分力有限。
- upper on/off 消融、三档 dense-L、7-repeat 正式候选和 15-repeat 低 L 稳定性复测全部完成并通过 validator；所有主要 sweep 使用同一 immutable binary SHA-256 `83c0444f...35e3`。
- 正式等 Recall 结果：25% 高质量点最高 1.220x，但中档存在减速；50% 在代表性中高质量目标为 1.620x--4.936x；75% 为 2.349x--6.274x。低 L Recall 完全稳定，wall time 有 100-thread 短任务调度长尾，主表使用 warm median/mean 并报告 CV。
- fresh single 与多层中层 partition/trie 完全一致；中层 intra edges 相差 10,602/约 0.038%，来自并行建图非确定性，因此同索引 upper on/off 用于机制，fresh single vs multi 用于系统等 Recall。
- NaviX/FAVOR 三档 x1、100 threads、5-repeat 粗网格已完成；FAVOR 50% 的 L200/L500 存在 15--18 秒调度长尾，后续统一报告 median/CV 与 raw repeats。
- Curator 使用独立 `thirdparty/curator-v2` clone 和 Python 3.10/FAISS 1.7.4 环境从 602,453-point x1 重建：build 102.207 s、磁盘 2.046 GB、内存估计 2.184 GB；16-query smoke 的 320 个返回 ID 无 containment 违规。

## 下一步

等待 Curator warmup/raw-repeat 三档复测完成，检查 Recall 稳定性与 per-budget median/CV；再做 3 个代表 `search_ef` 的阶段 profile，完成 ACORN provenance 审计和跨系统同 Recall 主表。

## 阻塞与问题

- 原始 checkout 有大量未提交/未跟踪文件；禁止直接 merge 或覆盖。
- 搜索进程第一个 L 的 repeat 0 仍有线程池/工作区冷启动，初筛主指标使用 repeats 1--2 的 warm mean；最终复测增加重复数。
- 远端磁盘使用率 94%，尚余约 835 GB；实验产物必须限制在必要矩阵，不复制主向量数据。
- `sunyahuia600-sunyahui` 本机别名未配置且跳板无法解析该名字；当前可用路由为 `ssh -J W300-pub sunyahui@10.77.110.170`。
- 旧系统 `ps -o pid=,etime=,stat=,cmd=` 输出异常曾造成 PID 10039 已退出的误判；已停止两个 runner，并给 runner 增加 output-root 独占锁。受资源竞争影响的结果位于 `runs/quarantine/20260905T2119_contention/`，禁止进入汇总。
- plain UNG 使用随机搜索路径，3 repeats 的 Recall 最大 spread 为 0.0029；所有等 Recall 结论需保留质量余量或在最终候选上增加 repeats，不能按 1e-4 差异排序。
- 两次旧 T2 初筛分别因运行中重编译、instrumentation 条件误放而隔离在 `runs/quarantine/20260905T2218_binary_rebuild/` 和 `runs/quarantine/20260905T2225_instrumentation_bug/`；禁止用于结论。
- `runs/quarantine/20260905T_current_label_mismatch/` 使用了错误的 21,834-label hybrid 主图；仅可作机制诊断，禁止用于 Recall/QPS 主张。
- Curator 首轮只保存 5-repeat mean，出现 `search_ef` 增大但 batch time 下降；该时间结论已降级，必须使用新增 raw-repeat 文件和阶段 profile 解释。
- ACORN 当前隔离仓库内只有 Curator-v2 自带 wrapper/源码线索，尚未确认 x1 index、输入 hash 和可直接运行的独立入口，未满足主表 provenance。

## 验证

- `ctest -R 'special_block_trie|special_block_free_state|special_candidate_queue|special_edge_io'` — `通过`：4/4。
- `cmake --build build_ung_rel -j16 --target build_special_block_index search_UNG_index` — `通过`。
- `git diff --check` — `通过`。
- `python3 experiments/multilevel_special/validate_selection_sweep.py ...` — `通过`：dense/final/stability 所有 case 完整。
- `cd experiments/multilevel_special && python3 -m unittest -v test_multilevel_selection.py` — `通过`：13/13，含 method-specific L、查询 provenance 与不可变二进制快照。
- `.venv_curator/bin/python -m unittest -v experiments.curator_baseline.test_curator_*` — `通过`：12/12，含持久化加载、ID 映射、optimized bitmap filter 和 raw-repeat 输出。
- Curator x1 smoke — `通过`：build/save/load 成功，16 queries、2 个 ef，320 个返回 ID 的 containment 违规数为 0。

## 仅在需要时阅读的细节

- `WORKTREE_HANDOFF.md` — 需要理解实现语义、已有构建/查询结果和 merge-back 边界时阅读。
- `experiments/multilevel_special/EXPERIMENT_STATE.md` — 需要继续参数扫描、解释实验或恢复运行时阅读。

## 恢复说明

1. 先读本看板，再读当前进行中的实验状态文档。
2. 运行 `git status --short`，不要暂存 `runs/`。
3. 检查 `runs/external_baselines/`、外部 baseline 配置和实际运行进程；不要覆盖 `/home/graphdb/FilterVectorResult` 的共享索引或历史结果。
4. 从“下一步”继续；新实测结果必须先写实验账本，再更新结论。

## 清理提示

- `runs/` 是未提交实验 artifact；最终只提交配置、汇总表和必要报告。
- 不再适用的旧 review-ready 看板已由本看板替代。
