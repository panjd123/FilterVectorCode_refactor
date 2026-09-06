# Agent 看板

最后更新：`2026-09-06`
分支：`codex/multilevel-special-block-20260905`
实现与结果检查点：`7a2bf4635eac43d174100770838daf1b3a10fa58`；当前源码回归收尾基于 `1906381f26c3f728bf8d428169f9fe992ca95da2`（origin 指向用户脏工作树，不直接 push）

## 目标

在隔离 shared clone 中实现并验证真正的多层 Special Block：保留中层，再叠加更大上层；在 Amazon 100% x1 六档真实选择率上按相同 Recall 比较 plain、单层、原始多层、调优多层和外部系统，形成可复现、可展示的论文级结果。

## 当前状态

- 总体：`review-ready`
- 摘要：固定 two-level 构建/逐级查询、六档内部正式实验、T1/T2 调优和四个外部系统比较均已完成。高选择率使用统一 immutable binary；两轮结构审阅和有限上下文交付审阅完成，最终回归与不可变实现/结果 checkpoint 已通过。

## 进行中

- 无。实现、论文结果闭包和当前源码 24 点版本漂移审计均已完成。

## 完成历史

- 建立隔离 shared clone `/home/sunyahui/worktrees/FilterVectorCode_multilevel_special`；原始脏仓库未修改。
- 实现两次独立 bottom-up partition、双 ownership、`activation_level=0/1/2`、逐级 edge transition、同点原位升级与去重 — 核心提交 `1f0c1e6`、`368227e`、`7008424`。
- 修复 T1/T2 配置所有权与最终值校验、显式 sidecar fail-closed — 提交 `05a4381`。
- 六档 Amazon x1 内部正式矩阵完成：0.499%、0.903%、9.907%、24.915%、49.971%、74.994%，每档 1000 queries、K=10、100 threads；所有 sweep validator 通过。
- T1=500/1000/2000、T2=4k/10k/25k/50k 已调参；最终当前配置为 T1=2k,T2=25k。
- 低选择率 crossing 网格补齐：0.903% 下 single 与原始多层按 L=3500--6000、7 repeats 公平选点。
- 外部低选择率完成：NaviX/FAVOR 各 3 workloads x 11 budgets x 5 repeats；Curator 各 12 budgets x 5 repeats；ACORN gamma=1/2/4/8/12 共 180 个粗筛点，选中质量上限点 5-repeat 复测，全部零过滤违规。
- 重写 `generate_paper_results.py`：生成六档内部/外部主表、构建表、209 个内部和 447 个外部 canonical 测量点、source SHA-256 manifest；11 项一致性/provenance 测试要求内部 CSV 与 manifest 的 method/workload/L/repeats 完全一致。Canonical 点按 workload/method/variant/budget 去重，paired rerun 覆盖同键旧统计。
- 重写 `MULTILEVEL_SPECIAL_BLOCK_PAPER_REPORT_CN.md`：分开第二层结构收益、T1 调优收益和系统最终结果，完整呈现负结果与计时边界。
- 修复第二轮结构审阅问题：持久化边界 topology validation、explicit sidecar format/fingerprint fail-closed、upper ownership map 内存计数、固定 middle/upper 两层命名；focused C++ tests 5/5。
- 加强 graph-aware metadata validation：同层 child 唯一父节点且无环；root/child/upper label 必须满足 trie 包含关系；entry point、direct members、point_count 与源 UNG 必须一致；显式 bundle 缺 regular/special edge sidecar 时 fail closed。
- 构建结果已迁入受 source manifest 哈希保护的 `build_results_source.csv`；生成器拒绝 CSV/manifest 网格或 repeat 不一致。
- 统一高选择率 binary：current crossing 与 paired formal 均使用 `f078e174...287b11`；三档 paired validator 全部通过，upper-off/on 因果证据已进入完整测量池。
- 第二轮结构审阅为 clean follow-up（无 High/Medium/Low）；有限上下文交付审阅确认唯一阻断是不可变 checkpoint。
- 最终回归通过：生产/测试目标构建成功，focused C++ 5/5，Python 26/26，结果生成器重建 60 行主结果、209 个内部 canonical 点和 7 行构建结果，`git diff --check` 通过。
- 实现、配置、compact evidence 与报告已提交为 `7a2bf4635eac43d174100770838daf1b3a10fa58`；`runs/` 和 nested third-party clones 未提交。
- 最终 limited-context 交付复审 verdict 为 `review-ready`，不可变 checkpoint/provenance 阻断闭环。
- 当前源码 fresh regression 完成六档 24/24 点：Special Recall 最大漂移 0，timing ratio 中位数 1.0195、范围 0.9781--1.2188；0.499% 使用 21 repeats 并显式保留长尾/CV。

## 下一步

由用户先为原始 checkout 形成 clean checkpoint，再 cherry-pick 本分支提交或按 handoff 逐项集成。

## 阻塞与问题

- 原始 checkout 有大量用户改动，禁止直接 merge；最终仅声明分支/patch merge-ready。
- 服务器 Git 1.8.3.1 不支持 native worktree，当前隔离环境是 `git clone --shared`。
- jump host 偶发断连；只做短时串行重试，避免并发 SSH。
- `runs/`、`thirdparty/acorn-official/`、`thirdparty/curator-v2/` 是未跟踪运行/第三方产物，不得提交。
- 多层查询发现 upper blocks 时禁用语义不完整的 GPU batch path，当前正确性优先，仍有性能优化空间。

## 验证

- `ctest -R 'ung_build_config|special_block_trie|special_block_free_state|special_candidate_queue|special_edge_io'` — `通过`：5/5。
- `cmake --build build_ung_rel -j16 --target build_special_block_index search_UNG_index` — `通过`。
- `python3 experiments/multilevel_special/validate_selection_sweep.py ...` — `通过`：全部正式内部 sweep。
- `python3 -m unittest -v experiments.multilevel_special.test_multilevel_selection` — `通过`：13/13。
- `python3 -m unittest -v test_generate_paper_results.py` — `通过`：13/13；包含统一 binary、CSV/manifest 网格/repeats、build source、upper-off 消融、current-source 24 点回归和 LF-only CSV 输出检查。
- `validate_selection_sweep.py config.amazon_x1_paired_formal_sel{25,50,75}.json` — `通过`：3/3，每个方法 7 repeats、Recall 无漂移。
- Curator 低选择率产物 — `通过`：3 workloads x 12 budgets x 5 measured。
- ACORN 低选择率产物 — `通过`：180 个 screen 点 + 3 个 formal 点，filter violations=0。

## 仅在需要时阅读的细节

- `WORKTREE_HANDOFF.md` — 理解实现语义、隔离来源与 merge-back 边界时阅读。
- `experiments/multilevel_special/EXPERIMENT_STATE.md` — 恢复实验、查询 raw artifact 或解释历史隔离结果时阅读。
- `docs/reports/MULTILEVEL_SPECIAL_BLOCK_PAPER_REPORT_CN.md` — 审阅方法与论文结论。
- `docs/reports/MULTILEVEL_SPECIAL_BLOCK_REPRODUCE_CN.md` — 从 compact evidence 审计或重新运行 raw benchmark 时阅读。

## 恢复说明

1. 先读本看板。
2. 运行 `git status --short`，不要暂存 `runs/` 或 `thirdparty/` nested clones。
3. 核对实现/结果检查点 `7a2bf4635eac43d174100770838daf1b3a10fa58`；其后的提交只更新交付状态。
4. 任何新数值必须先进入 source CSV 并由生成器输出。

## 清理提示

- 保留 compact aggregate CSV；不提交 raw `runs/` 和大型索引。
- 完成审阅后压缩 `EXPERIMENT_STATE.md` 中已经失效的 hybrid 主图记录。
