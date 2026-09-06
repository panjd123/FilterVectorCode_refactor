# Agent 看板

最后更新：`2026-09-06`
分支：`codex/multilevel-special-block-20260905`
实现与结果检查点：`7a2bf4635eac43d174100770838daf1b3a10fa58`；当前代码检查点：`3827ff6c9031684726d879926b3c86b16fddd471`（origin 指向用户脏工作树，不直接 push）

## 目标

在隔离 shared clone 中实现并验证真正的多层 Special Block：保留中层，再叠加更大上层；在 Amazon 100% x1 六档真实选择率上按相同 Recall 比较 plain、单层、原始多层、调优多层和外部系统，形成可复现、可展示的论文级结果。

## 当前状态

- 总体：`验证中`
- 摘要：固定 two-level 构建/逐级查询、正式实验与论文交付已完成；最终 source-layout gate、fresh build、two-level/legacy load-query 和完整回归均已通过，正在固化最终 checkpoint 与 merge-back 风险清单。

## 进行中

- 更新最终 checkpoint 与 merge-back 数字后，运行最后一次 clean-tree/manifest 检查。

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
- upper activation reachability 校验要求每个 upper block 至少包含一个由其子树内部 middle block 拥有的非空 direct member；真实构造夹具、零 ownership 与仅 ancestor ownership 两类不可达负例均已覆盖。公开 graph validator 也已改为自包含基础格式校验，代码检查点 `3fa0616`。最终 binary `3947f438...11ba3` 全量加载 62,938,887 条边；50%/L550 Recall=.8575、10,000 个结果、0 违规。详细统计显示 512/1000 query 搜索 upper block、57,897 次 upper activation。
- build-preflight 首次 immutable-snapshot 运行在 7.179 s 后按预期暴露生命周期错误：partition 阶段尚未生成 `entry_point_id`，完整 graph validator 报 `entry point is not in a direct member group`。此前 61.157 s 成功构建发生在该调用进入 binary 之前，不能作为新 preflight 的通过证据；旧判断已失效。失败 manifest 原生绑定 builder `df3015b5038095adcea83fc33fec713c69e3bf2f40bc01f9ac98bed5f8626be6`。
- validator 已按生命周期拆分：partition preflight 验证拓扑、ownership、point count 与 upper reachability，但不要求尚未生成的 entry point；intra graph 完成后执行完整 gate，再构造 regular overlay/保存。新增单测验证无 entry point 的合法 partition 只通过前者。
- 最终 immutable builder `c5cee68dc2dab7c409270eb37bbd73e903a425fbedd6a7bf943a4c52c0653a8f` fresh 构建成功：57.313 s，111 blocks（8 upper）、62,941,206 edges。search binary `af73a74de3cac02be1b4e9ae44c2d6eaa68cf73d463d43ddc1f3dc721064990b` 在该 bundle 上 50%/L550 Recall=.8593、10,000 results、0 violations；较旧 rebuild 上界 .8582 高 .0011，符合 approximate graph 波动并超过 .85 门槛。无 `gpulock`，单次时间不更新冻结性能表。
- fresh build/query 证据已压缩为 `results_summary/source/build_preflight_audit.csv`，由 source/artifact manifest 闭包保护，并由新增 provenance test 锁定 hash、x1/T1/T2、Recall 与 0 violation。完整 build、CTest 14/14、Python 30/30 和结果重建均通过；主结果仍为 60 selected rows、209 internal points、7 build rows。
- 两阶段 validator、单测、fresh provenance、报告与 artifact closure 已提交为 `394facda1f912a6710d189f44a34f66d53a4190d`；`runs/` 与 nested third-party clone 未暂存。
- 完整 graph gate 已移动到 intra 完成、inter 开始前，并在 overlay 后防御性复核；最终 fresh build/query 证据已更新，代码 checkpoint `088c872`。preflight 单测进一步证明它仍拒绝错误 point count 与不可达 upper。current search binary 对旧单层 sidecar 的 50%/L1800 回归为 Recall=.8523、10,000 results、0 violations。
- 最终 fresh bundle 六档 correctness audit 全部达标：Recall=.9101/.9165/.9009/.9023/.8593/.8745；60,000 slots、59,991 returned、9 missing、0 filter violations。compact 证据进入 `final_fresh_six_workload_audit.csv`，不使用单次耗时更新性能表。
- 最终六档 audit、边界单测和报告闭包已提交为 `fbac94bae94ced0551bf2627c376991c8bfdf665`。
- 最终 source-layout gate 覆盖 sentinel、label path 规范/唯一性、全局 range 分区、point ownership、每层最近-root direct ownership、common labels 与 subtree point count；系统性损坏负例、完整 CTest 14/14、Python 32/32 均通过。
- 构建阶段只做一次完整 source/partition preflight，intra 后只检查 entry-point delta；最终 builder `c305f487...e0b295b` fresh 构建 T1=2k/T2=25k 成功：runner wall 59.038 s、metadata/partition 2.659 s、111 blocks/8 upper、62,941,289 edges。最终 search `052e4cc3...88be2a` 在 fresh two-level 的 50%/L550 为 Recall=.8587，在旧 single-level 的 50%/L1800 为 Recall=.8523；各 10,000 results、0 violations。单次 timing 仅作 correctness/provenance audit。
- merge-back 审计：原始 checkout HEAD 仍为共同基线 `dda63bd`，隔离分支领先 70 commits；原始 checkout 有 150 项脏改动。任务分支现有 409 个改动路径，其中 128 个与原始脏路径重叠：101 个当前结果相同（含双方都删除的一个路径）、27 个内容不同。分支可独立审阅，但必须先保存原始改动并人工整合 27 个分叉路径，禁止直接自动 merge/cherry-pick；清单见 `docs/reports/MULTILEVEL_SPECIAL_BLOCK_MERGE_BACK_CN.md`。

## 下一步

提交更新后的 checkpoint/handoff，确认 tracked tree clean；等待达到用户要求的 18 小时后关闭 goal。

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
- `cd experiments/multilevel_special && python3 -m unittest -v test_multilevel_selection.py test_generate_paper_results.py` — `通过`：31/31；生成 60 行主结果、209 个内部点、7 行构建结果。
- source-layout 最终回归 — `通过`：完整 CTest 14/14；Python 32/32；fresh two-level 与旧 single-level 各 10,000 个结果、0 filter violations。
- `validate_selection_sweep.py config.amazon_x1_paired_formal_sel{25,50,75}.json` — `通过`：3/3，每个方法 7 repeats、Recall 无漂移。
- Curator 低选择率产物 — `通过`：3 workloads x 12 budgets x 5 measured。
- ACORN 低选择率产物 — `通过`：180 个 screen 点 + 3 个 formal 点，filter violations=0。

## 仅在需要时阅读的细节

- `WORKTREE_HANDOFF.md` — 理解实现语义、隔离来源与 merge-back 边界时阅读。
- `docs/reports/MULTILEVEL_SPECIAL_BLOCK_MERGE_BACK_CN.md` — 准备把隔离分支整合回原始脏 checkout 时阅读。
- `experiments/multilevel_special/EXPERIMENT_STATE.md` — 恢复实验、查询 raw artifact 或解释历史隔离结果时阅读。
- `docs/reports/MULTILEVEL_SPECIAL_BLOCK_PAPER_REPORT_CN.md` — 审阅方法与论文结论。
- `docs/reports/MULTILEVEL_SPECIAL_BLOCK_REPRODUCE_CN.md` — 从 compact evidence 审计或重新运行 raw benchmark 时阅读。

## 恢复说明

1. 先读本看板。
2. 运行 `git status --short`，不要暂存 `runs/` 或 `thirdparty/` nested clones。
3. 核对实现/结果检查点 `7a2bf4635eac43d174100770838daf1b3a10fa58` 和当前 validator checkpoint `394facda1f912a6710d189f44a34f66d53a4190d`；后者不改主性能数值。
4. 任何新数值必须先进入 source CSV 并由生成器输出。

## 清理提示

- 保留 compact aggregate CSV；不提交 raw `runs/` 和大型索引。
- 完成审阅后压缩 `EXPERIMENT_STATE.md` 中已经失效的 hybrid 主图记录。
