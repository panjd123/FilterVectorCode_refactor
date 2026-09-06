# Handoff: Multi-level Special Block

## 用户目标

在不修改原始脏仓库的前提下，独立实现真正的多层 Special Block：保留中层 block，叠加更大的上层 block；查询从普通图逐级进入中层和上层。随后在 Amazon 原始 100% x1 上调参，以端到端 Recall 为质量标准，和 plain、单层及外部 filtered-ANN 系统做公平比较，并形成论文级文档。

## 隔离环境

- 原始 checkout：`/home/graphdb/FilterVectorCode_refactor`，分支 `shopai8/special-block-e2e-opt`，创建隔离目录时 HEAD `dda63bd`。
- 隔离 checkout：`/home/sunyahui/worktrees/FilterVectorCode_multilevel_special`。
- 实现分支：`codex/multilevel-special-block-20260905`。
- 实现与结果检查点：`7a2bf4635eac43d174100770838daf1b3a10fa58`；当前代码检查点：`5793e04`，增加 upper activation reachability 校验。
- 前一审计 HEAD：`77e98eb`；多层 `favor_blocks` 已 fail closed，正常查询使用 `free_state`。
- 服务器 Git 1.8.3.1 不支持 native worktree，因此使用 `git clone --shared`；branch/index 独立、对象库共享。
- 原始 checkout 有大量用户改动。不得在那里 reset/checkout/merge；应先由用户形成 clean checkpoint，再 cherry-pick 本分支意图提交。

## 当前实现

构建器对同一 group trie 做两次独立 bottom-up uncovered partition。中层和上层分别维护 ownership；同一 point 可同时属于两层。普通图、中层 special graph、上层 special graph 作为 overlay 共存。持久化格式为 `special_block_trie_multilevel_v1`，显式 sidecar 缺失/损坏时 fail closed。

查询候选持有 `activation_level in {0,1,2}`：level 0 只能走普通边，level 1 可走中层边，level 2 才能走上层边；一条 transition 最多提升一级。同一点只占一个候选槽，高层路径到达时原位升级并允许重新扩展。主扩展与 pre-expand 共用 `special_block_edge_transition`。多层 metadata 存在时禁用尚未携带 level 的 GPU batch bypass，以正确性优先。

核心提交：

- `1f0c1e6`：多层 partition/ownership/overlay/activation 主实现。
- `368227e`：point 去重、level 原位升级、loader 修复。
- `7008424`：统一 edge transition helper，禁用语义不完整的多层 GPU batch path。
- `05a4381`：T1/T2 配置统一所有权和最终校验，显式 sidecar fail-closed。
- `7a2bf46`：graph-aware 语义校验、完整正式结果闭包、论文报告与最终回归基线。

## 已验证结果

数据为 Amazon 原始 x1：602,453 points、768D、482,387 groups、30,723 labels。六档真实 workload 平均选择率为 0.499%、0.903%、9.907%、24.915%、49.971%、74.994%；每档 1000 queries、K=10、100 threads。唯一质量标准是对 exact filtered GT 的 Recall。

固定 T1=1k 时，统一 binary 的方法族最佳结果表明：新增上层相对单层在 24.915%/49.971%/74.994% 同 Recall 门槛下为 1.289x/1.709x/2.320x；固定 T2=25k 的同轮 paired 对照为 1.289x/1.709x/2.229x。联合调优后当前配置为 T1=2k,T2=25k；相对 plain 在六档分别为 1.044x、0.560x、0.377x、0.552x、14.484x、57.377x。负结果必须保留：0.903%--24.915% plain 更快；0.499% 的差异小于波动尺度，只能称为基本持平。

高选择率主表不再使用旧 binary `83c0444...` 的 aggregate。所有内部正式 run manifest 都必须绑定搜索 binary `f078e174...287b11`；`generate_paper_results.py` 会 fail closed。最新 upper-off/on 同索引、同 L 消融分别为：25% `.8946 -> .9012`，50% `.7633 -> .8584`，75% `.7678 -> .8759`。

外部低选择率也已补齐：

- FAVOR/NaviX：3 workloads x 11 budgets x 5 repeats。
- Curator：3 workloads x 12 `search_ef` x 5 measured；旧服务器需使用 `/lib64/libopenblas.so.0`，不能使用要求 GLIBC 2.27 的 `/home/graphdb/OpenBLAS`。
- ACORN：gamma=1/2/4/8/12，3 workloads x 12 ef x 3 measured，共 180 个 screen 点；选中质量上限点再做 5-repeat formal，全部 `filter_violations=0`。三档最高 Recall 为 .7391/.7907/.8968，均未达到 .90。

完整方法、构建、查询与外部表在 `docs/reports/MULTILEVEL_SPECIAL_BLOCK_PAPER_REPORT_CN.md`。结果生成入口为 `experiments/multilevel_special/generate_paper_results.py`；源 CSV 和 SHA-256 manifest 位于 `experiments/multilevel_special/results_summary/`。

## 当前未决事项

没有实现或论文主表阻断项。第二轮结构审阅为 clean follow-up；有限上下文交付审阅确认交付内容可理解。最终回归通过生产/测试目标构建、focused C++ 7/7、Python 29/29、结果重建和 `git diff --check`。旧 loaded-byte 实测值来自计数修复前，约低估 4.1 MiB，报告已显式降级，不能当精确峰值。

当前源码 binary `88d7dba1...f37f` 另完成六档 24 点版本漂移审计；它与冻结论文 binary 不同，因此只进入 `results_summary/current_source_regression.csv`，不改写主表。18 个 Special 点 Recall 最大漂移为 0；24 点耗时比中位数 1.0195、范围 0.9781--1.2188。0.499% 采用 21 repeats 后仍有调度长尾，不能用单次或严格 timing gate 判断回归。

此外，当前 builder `6ff471a8...50692` 对 tuned 配置独立重建三次，wall time 为 60.250 / 56.285 / 56.916 s。结构 sidecar 稳定，但 GPU FastGrnnd approximate intra edges 因原子并行更新不保证 bitwise deterministic；50% 的冻结 L=500 在一份 bundle 上为 R=.8495，因此跨重建推荐 L=550（三次 R=.8540--.8582）。这组结果单列于 `current_source_rebuild*.csv`，不能与冻结主表混写。

当前源码还增加了默认关闭、batch timer 外的 `UNG_VALIDATE_FILTER_RESULTS=1`。在同一 fresh-built T1=2k/T2=25k bundle 上覆盖六档 workload，共 60,000 个结果槽、59,991 个实际返回点、9 个 missing 槽，过滤违规为 0；六档 Recall 全部达标。该审计 binary 为 `378e71d7...0313`，只用于正确性，不更新冻结性能表；逐档 compact 记录为 `results_summary/filter_validation_audit.csv`。
sidecar loader 进一步加入逐 edge 的 owner/direct-child 语义校验，将 legacy binary 改为验证后再发布的 staged load，并修正 upper inter edge 的 light/heavy 分流与查询 per-target-block cap，使其按 owner level 选择 ownership map。converter/runtime fallback 共用严格 CSV parser；损坏 requested heavy binary 可用合法 CSV 回退、无 fallback 时 fail closed。真实 fresh bundle 的 62,938,887 条 special edges 全量通过；最新 binary `550f04c4...204c` 在 50% workload、L=550 得到 Recall=.8575、10,000 个返回点、0 个过滤违规。单次 load/query 为 1.893/0.973 s，只作正确性证据，不替换冻结性能表。

最后一轮结构审计进一步要求：同层 child 必须指向最近的同层 Trie block 祖先，middle `parent_block_id` 必须指向根路径上最近的 upper block。真实 tuned metadata 为 103 个 middle、8 个 upper、108 条同层 child edge，全部通过重推导。随机树反例同时证明独立 T1/T2 partition 不保证 upper 是 middle 的严格粗化，因此没有错误地要求 middle 全体成员共享 upper owner；cross-partition membership activation 已成为可单测共享 helper。最终 binary `b62d6e5...da2e` 全量加载 62,938,887 条边，并在 50%/L550 上保持 Recall=.8575、10,000 个结果、0 违规；单次 2.096 s load/1.518 s query 只作正确性证据。

最新 reachability 校验还要求每个 upper block 至少有一个 middle-owned direct member，保证存在合法 `1 -> 2` 激活位置；这不要求两个 partition 严格嵌套。focused C++ tests 7/7 通过。binary `f0a978c5...a0aae` 全量加载同一 62,938,887-edge sidecar，50%/L550 Recall=.8575、10,000 个结果、0 违规。关闭 light stats 的机制审计显示 533/1000 query 使用 special graph、512/1000 搜索 upper block、57,897 次 upper activation；诊断耗时不进入主性能表。
三次 build 早于 immutable builder snapshot 功能，旧 manifest 未原生携带 builder hash；compact CSV 将审计记录的 hash 明确标为 `historical_audit_record`。当前 builder 文件 mtime 早于三次 manifest 且 hash 一致，但这仍不是逐份 manifest 的密码学绑定。新运行会记录 `manifest_snapshot`。
CPU Vamana large-block 对照为 1131.863 s，GPU 路径按完整 builder wall 快 19.89x；该对照图本身不同，只用于说明确定性 CPU 回退的工程代价。

`runs/`、`thirdparty/acorn-official/`、`thirdparty/curator-v2/` 是未跟踪实验/第三方产物，不得提交。compact aggregate、runner、patch、报告和 manifest 应提交。

原始 checkout 仍含大量用户改动，故本分支只声明 merge-ready，不自动修改原始 checkout。建议先保存原始工作树，再 cherry-pick 本分支从 `1f0c1e6` 到当前 HEAD 的任务提交；若只审阅最终增量，先从 `7a2bf46` 开始阅读。

## 恢复入口

1. 先读 `AGENT_KANBAN.md`，再运行 `git status --short` 和 `git rev-parse HEAD`。
2. 只在需要定位 raw 产物时读 `experiments/multilevel_special/EXPERIMENT_STATE.md`。
3. 重跑 `python3 experiments/multilevel_special/generate_paper_results.py` 和对应 unittest，确认表可完全重建。
4. 运行 focused C++ tests、multilevel Python tests 和 `git diff --check`。
5. 最终检查 implementation branch clean、原始 target 是否前进、nested clones 未进入提交；由于 target 脏，只声明 cherry-pick ready，不自动合并。

完整命令、环境、raw replay 与当前源码 fresh rerun 的边界见 `docs/reports/MULTILEVEL_SPECIAL_BLOCK_REPRODUCE_CN.md`。历史论文 binary 位于未提交的 `runs/`；fresh rerun 必须使用新 output root 和新 manifest，不能覆盖历史 evidence。
