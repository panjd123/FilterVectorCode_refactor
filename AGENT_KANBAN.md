# Agent 看板

最后更新：`2026-09-06`
分支：`codex/multilevel-special-block-20260905`
实现与结果检查点：`7a2bf4635eac43d174100770838daf1b3a10fa58`；当前代码检查点：`104986c`（origin 指向用户脏工作树，不直接 push）

## 目标

在隔离 shared clone 中实现并验证真正的多层 Special Block：保留中层，再叠加更大上层；在 Amazon 100% x1 六档真实选择率上按相同 Recall 比较 plain、单层、原始多层、调优多层和外部系统，形成可复现、可展示的论文级结果。

## 当前状态

- 总体：`验证中`
- 摘要：固定 two-level 构建/逐级查询、正式实验与论文交付已完成；最新审计补足 upper activation reachability 校验，真实 sidecar、端到端过滤正确性与 upper 路径 telemetry 均通过。

## 进行中

- 同步 reachability 审计文档与 manifest，重跑结果闭包测试并形成独立 checkpoint；冻结论文主性能数值不变。

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
- 最终回归通过：生产/测试目标构建成功，focused C++ 5/5，Python 28/28，结果生成器重建 60 行主结果、209 个内部 canonical 点和 7 行构建结果，`git diff --check` 通过。
- 实现、配置、compact evidence 与报告已提交为 `7a2bf4635eac43d174100770838daf1b3a10fa58`；`runs/` 和 nested third-party clones 未提交。
- 最终 limited-context 交付复审 verdict 为 `review-ready`，不可变 checkpoint/provenance 阻断闭环。
- 当前源码 fresh regression 完成六档 24/24 点：Special Recall 最大漂移 0，timing ratio 中位数 1.0195、范围 0.9781--1.2188；0.499% 使用 21 repeats 并显式保留长尾/CV。
- 当前源码 tuned 配置完成三次独立 fresh build：builder wall 中位数 56.916 s；block/trie/regular edge bitwise stable，GPU approximate special edges 非 bitwise deterministic。50% 跨重建稳健点为 L=550，三次 Recall .8540--.8582。
- CPU Vamana large-block 对照总 builder wall 1131.863 s；GPU tuned 构建中位 56.916 s，完整构建快 19.89x，故不以 CPU 回退换取字节级确定性。
- 最终 provenance 审计把三次旧 fresh build 的 builder hash 来源显式写入 CSV：旧 manifest 没有原生 hash，故标为 `historical_audit_record`；后续 fresh run 才是 `manifest_snapshot`。同时说明日志 `gpu_intra_enabled=0` 是旧全局开关，实际 routed path 有 24 个大 block 使用 FastGrnnd CUDA。主性能表未变化；提交 `dba4696`。
- 新增默认关闭、计时外的 filtered-result validator 与独立单元测试；六档真实 workload 共 60,000 个结果槽、59,991 个实际返回点、9 个 missing 槽，所有返回点违规为 0，六档 Recall 均达标。审计只作正确性证据，不进入性能表。
- 六档 filtered-result 合法性数据与说明已提交为 `d06d0fd`。
- sidecar loader 已补强逐 edge owner/direct-child 校验和 legacy staged load，并修复 upper inter edge 的 light/heavy 分流层级；真实 fresh bundle 的 62,938,887 条 special edges 全量通过，50%/L=550 查询为 Recall=.8575、10,000 个结果、0 过滤违规。
- converter/runtime fallback 已统一严格 CSV parser；upper inter per-target-block cap 使用 upper ownership；损坏 requested heavy binary 可用合法 CSV 回退、无 fallback 时 fail closed。测试还修复固定临时目录残留导致的不可重入问题，并明确空图/空 block bundle 是合法退化输入。最新 binary `550f04c4...204c` 再次全量加载 62,938,887 条边，50%/L=550 为 Recall=.8575、10,000 个结果、0 过滤违规；单次耗时不进入性能表。
- 第三轮结构审计从 root-label path 重推导 nearest same-layer parent 与 nearest upper ancestor；真实 tuned metadata 为 103 个 middle、8 个 upper、108 条同层 child edge，全部一致。随机树反例证明独立 partition 不保证严格 refinement，故保留双 ownership 且不施加错误的全成员同 upper-owner 约束；cross-partition membership activation 已提取为共享 helper 并单测。最终 binary `b62d6e5...da2e` 全量加载 62,938,887 条边，50%/L550 Recall=.8575、10,000 个结果、0 违规。
- upper activation reachability 校验要求每个 upper block 至少包含一个由其子树内部 middle block 拥有的非空 direct member；真实构造夹具、零 ownership 与仅 ancestor ownership 两类不可达负例均已覆盖，代码检查点 `104986c`。最终 binary `a54518b8...7cd98` 全量加载 62,938,887 条边；50%/L550 Recall=.8575、10,000 个结果、0 违规。详细统计显示 512/1000 query 搜索 upper block、57,897 次 upper activation。

## 下一步

更新 artifact hash，重跑 Python/result closure，提交文档 checkpoint；随后检查 merge-readiness 并继续非破坏性审计至目标时长。

## 阻塞与问题

- 原始 checkout 有大量用户改动，禁止直接 merge；最终仅声明分支/patch merge-ready。
- 服务器 Git 1.8.3.1 不支持 native worktree，当前隔离环境是 `git clone --shared`。
- jump host 偶发断连；只做短时串行重试，避免并发 SSH。
- `runs/`、`thirdparty/acorn-official/`、`thirdparty/curator-v2/` 是未跟踪运行/第三方产物，不得提交。
- 多层查询发现 upper blocks 时禁用语义不完整的 GPU batch path，当前正确性优先，仍有性能优化空间。

## 验证

- `ctest -R 'ung_build_config|special_block_trie|special_block_free_state|special_candidate_queue|lng_block_partition|special_edge_io|filter_validation'` — `通过`：7/7。
- `cmake --build build_ung_rel -j16 --target build_special_block_index search_UNG_index convert_special_edges` — `通过`。
- 真实 `T1=2k,T2=25k` sidecar + 50%/L550 — `通过`：62,938,887 edges，Recall=.8575，10,000 results，0 violations；detail stats 记录 57,897 upper activations。
- `python3 experiments/multilevel_special/validate_selection_sweep.py ...` — `通过`：全部正式内部 sweep。
- `python3 -m unittest -v experiments.multilevel_special.test_multilevel_selection` — `通过`：14/14；新增 builder binary immutable snapshot 回归。
- `python3 -m unittest -v test_generate_paper_results.py` — `通过`：15/15；包含统一 binary、CSV/manifest 网格/repeats、build source、upper-off 消融、current-source 24 点回归、三次 rebuild 鲁棒性、过滤合法性审计和 LF-only CSV 输出检查。
- `cd experiments/multilevel_special && python3 -m unittest -v test_multilevel_selection.py test_generate_paper_results.py` — `通过`：29/29；生成 60 行主结果、209 个内部点、7 行构建结果。
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
3. 核对实现/结果检查点 `7a2bf4635eac43d174100770838daf1b3a10fa58` 和当前 reachability code checkpoint `104986c`；后者不改主性能数值。
4. 任何新数值必须先进入 source CSV 并由生成器输出。

## 清理提示

- 保留 compact aggregate CSV；不提交 raw `runs/` 和大型索引。
- 完成审阅后压缩 `EXPERIMENT_STATE.md` 中已经失效的 hybrid 主图记录。
