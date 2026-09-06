# Handoff: Multi-level Special Block

## 用户目标

在不修改原始脏仓库的前提下，独立实现真正的多层 Special Block：保留中层 block，叠加更大的上层 block；查询从普通图逐级进入中层和上层。随后在 Amazon 原始 100% x1 上调参，以端到端 Recall 为质量标准，和 plain、单层及外部 filtered-ANN 系统做公平比较，并形成论文级文档。

## 隔离环境

- 原始 checkout：`/home/graphdb/FilterVectorCode_refactor`，分支 `shopai8/special-block-e2e-opt`，创建隔离目录时 HEAD `dda63bd`。
- 隔离 checkout：`/home/sunyahui/worktrees/FilterVectorCode_multilevel_special`。
- 实现分支：`codex/multilevel-special-block-20260905`。
- 当前已提交检查点：`05a4381`；本轮实验配置、compact results、报告和看板尚待最终审阅后提交。
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

第二轮结构审阅发现的拓扑持久化、显式 sidecar fail-closed、upper ownership 内存计数和固定两层命名问题已经修复并通过 focused tests。当前只剩最终 limited-context 交付审阅、全量回归、checkpoint commit 和 merge-readiness 审计。旧 loaded-byte 实测值来自计数修复前，约低估 4.1 MiB，报告已显式降级，不能当精确峰值。

`runs/`、`thirdparty/acorn-official/`、`thirdparty/curator-v2/` 是未跟踪实验/第三方产物，不得提交。compact aggregate、runner、patch、报告和 manifest 应提交。

## 恢复入口

1. 先读 `AGENT_KANBAN.md`，再运行 `git status --short` 和 `git rev-parse HEAD`。
2. 只在需要定位 raw 产物时读 `experiments/multilevel_special/EXPERIMENT_STATE.md`。
3. 重跑 `python3 experiments/multilevel_special/generate_paper_results.py` 和对应 unittest，确认表可完全重建。
4. 运行 focused C++ tests、multilevel Python tests 和 `git diff --check`。
5. 最终检查 implementation branch clean、原始 target 是否前进、nested clones 未进入提交；由于 target 脏，只声明 cherry-pick ready，不自动合并。
