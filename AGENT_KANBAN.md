# Agent 看板

最后更新：`2026-09-10 Asia/Shanghai`
分支：`codex/multilevel-special-block-20260905`
实现与结果检查点：`7a2bf4635eac43d174100770838daf1b3a10fa58`；高选择率文档与证据检查点：`42c21c10696638dbf83a54bd00e37bb88dc474ac`（origin 指向用户脏工作树，不直接 push）

## 目标

在隔离 shared clone 中实现并验证真正的多层 Special Block；本轮新增核验 `FilterVectorData/Amazon` 中历史查询集，并在当前 Amazon x1 base/labels、exact GT 与统一 binary 下重测可兼容集合。

## 当前状态

- 总体：`进行中`
- 摘要：既有 Amazon x1 高选择率结论保持冻结。本轮发现 `old_query_1000/query_minlen1_cov10k` 与当前 x1 基本兼容（当前重算平均选择率 28.610%）；hybrid 普通集需重标定，hybrid Zipf 与当前 label 语义不兼容，selected 集存在后验筛选偏差。正在为兼容集重算 exact GT 并做 0/1/2 层重测。

## 进行中

- 为 `old_query_1000/query_minlen1_cov10k` 生成当前 Amazon x1 exact GT，随后用统一 binary 粗筛 0 层、T1=1k 单层、T1=1k 多个 T2，以及已有共享调优单层/双层。

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
- merge-back 审计：原始 checkout HEAD 仍为共同基线 `dda63bd`，隔离分支领先 70+ commits；原始 checkout 有 150 项脏改动。任务分支现有 410 个改动路径，其中 128 个与原始脏路径重叠：101 个当前结果相同（含双方都删除的一个路径）、27 个内容不同、0 个未解释单边路径。分支可独立审阅，但必须先保存原始改动并人工整合 27 个分叉路径，禁止直接自动 merge/cherry-pick；清单见 `docs/reports/MULTILEVEL_SPECIAL_BLOCK_MERGE_BACK_CN.md`。
- 新的互斥阶段计时 smoke 已通过：0/1/2 层逐 query 的 `ELS + EntryPointSetup + BlockAuthorization + GraphSearch + Residual = Total`，最大 closure error 约 `3e-12 ms/query`；0 层 authorization=0。禁用 query-result reuse 后首轮 ELS lazy initialization 明显，故正式统计丢弃 repeat 0，后续 warm repeat 仍不复用查询结果。
- 公平调优规则已锁定：0 层只调 L；1 层独立调 `T1={500,1k,2k,4k,8k} × L`；2 层调同一 T1 网格与合法 `T2={4k,10k,25k,50k} × L`，边界赢家必须继续向外扩展或局部加密。正式只声明预先声明离散网格内的实测最优，同时区分跨六档共享阈值（论文主结论）和逐 workload oracle（能力上界）。
- 阶段计时、runner、测试与公平调优协议已提交为 `4bd1fd50b7a046ba4f7f02817a320ddc3710bfb7`；`runs/` 与 nested third-party clone 未暂存。
- 统一结构网格完成 `23/23`：5 个单层、18 个合法两层，全部使用 builder `c305f487...e0b295b`；累计 wall 1660.278 s，索引声明磁盘总量 15.033 GiB。
- 查询粗筛配置已 dry-run 覆盖 `24 methods x 6 workloads = 144 cases`、851 个 L 点；支持 workload-specific L 网格，严格检查每个 L/repeat 的 stage closure、0 层 authorization=0 和 ELS reuse disabled。
- 旧单层结果曾缺少显式 root-label coverage，已隔离且不参与统计；修正后的所有 Special case 统一使用 `UNG_SPECIAL_BLOCK_ROOT_LABEL_COVERAGE=1`。plain 结果不受该开关影响并按 provenance 复用。
- 单层已完成结构的中间证据显示除 `sel_005` 外，5 档 workload 的当前赢家落在 T1=8000 扫描上界；已准备 T1=16000/32000 构建扩展，待 coarse sweep 完成后补建与补测，当前不得宣称单层阈值最优。
- formal shortlist 生成器已使用独立输出目录，并记录选择规则、coarse config 与 `all_points.csv` 的 SHA-256 来源，防止旧 formal 产物或被修改的 coarse 结果静默混入；提交 `3057b1a`，本地及远端单测均为 32/32。
- 原 coarse 144/144 case 通过 validator；共享赢家为单层 T1=8k（相对 0 层 geomean speedup 2.01x）与两层 T1=8k/T2=25k（2.67x），两者均触 T1 上边界，尚不能作为最终最优结论。
- 新增第一轮边界 guard：单层 T1=16k/32k；两层 (2k,100k)、(8k,9k/100k)、(16k,25k/50k/100k)、(32k,50k/100k)。总网格 34 个结构，其中 26 个两层结构；提交 `848949b`。
- 10 个 guard 结构全部构建并验证；34 个结构 x 6 workloads 的 extended coarse sweep 已完成 `204/204` case，旧网格严格复用 144 case，新增实测 60 case；等待完整 validator/selector 输出后判断是否闭合边界。
- 完整 204-case validator 通过并汇总 1209 个离散点。共享配置更新为单层 T1=32k（coarse geomean 2.8435x）和两层 T1=16k/T2=100k（2.8862x）；二者仍分别触 T1/T2 上边界，不能作为最终最优结论。第二轮 focused guard 新增单层 T1=64k，以及两层 (16k,200k)、(32k,40k/200k)、(64k,100k/200k)，共 6 个结构、36 个 query cases。
- 第二轮 6 个 guard 结构和 36 个 query cases 完成，完整 240-case validator 通过，汇总 1424 个离散点。共享赢家保持单层 T1=32k（2.8435x）与两层 T1=16k/T2=100k（2.8862x），因此共享部署阈值已不触边。oracle 尚有 6 个轴触边；最终 guard 只为 sel_25 单层、sel_25 两层和 sel_005/sel_01 两层补测 8 个可行结构、11 case，另保留 4 个退化端点作为结构上界，不参与共享配置选择。
- 高选择率 7-repeat formal sweep 已完成并通过完整性验证：95/95 cases、570 points；repeat 0 为 cold，主指标为后 6 个 warm batch median，每个 repeat Recall 均须 >=0.87，固定 `cpu_bruteforce_els` 且 `UNG_DISABLE_ELS_REUSE=1`。
- 正式主 sweep 与 boundary extension 合并为 719 个唯一 7-repeat 测点。最终共享结构结果：0 层总 warm median 853173.4 ms；1 层 T1=128k 为 2176.467 ms、相对 0 层 geomean 326.2761x；2 层 (T1=16k,T2=400k) 为 1960.075 ms、相对 0 层 361.8935x，因此两层相对单层约 1.1092x。共享结构轴与 Lsearch 轴均已闭合。
- 100% 独立 unfiltered control 的最终共享点：0 层 L=256250、356311.0 ms；1 层 T1=128k/L=125、438.012 ms、相对 0 层 813.4731x；2 层 (16k,400k)/L=15、366.174 ms、相对 0 层 973.0647x。两层下侧已测到合法端点 L=K=10；该 control 不能与 filtered workload 连续外推。
- 首轮专用 selector 输出 18 个逐 workload oracle、3 个共享配置，并暴露 13 个结构轴和 16 个 Lsearch 边界命中；补测后 shared 边界命中归零，oracle 仅保留 2 个结构轴提示。通用 equal-recall 汇总为空不表示无可行点，本任务以逐 repeat Recall 门槛的专用 selector 为准。
- 三份报告已按证据域收紧：原六档不支持“两层整体优于单层”；新增 80.024%--96.702% 仅支持五个离散 filtered 测点的一致方向，不宣称连续区间或统计显著性；100% 单列为空 predicate control。
- 高选择率 compact evidence 已完成哈希复核：10/10 artifact 与 `high_selectivity_evidence_manifest.csv` 一致；总量约 0.5 MiB。manifest 中绝对路径是原始远端执行 provenance，复现命令和提交内 evidence 入口均使用仓库相对路径。
- 高选择率配置、脚本、compact evidence 与三份报告已提交为 `42c21c10696638dbf83a54bd00e37bb88dc474ac`；`runs/` 与 nested third-party clones 未提交。

## 下一步

完成 exact GT 后 dry-run 新配置，先跑最小正确性点，再执行宽 L 粗筛；按预声明 Recall 门槛从实测点比较，而不使用旧 GT、旧 coverage 或插值。

## 阻塞与问题

- `hybrid_query/query_minlen1_cov10k_zipf` 在当前 x1 上每 query 仅 0--1 个匹配点，不能作为 K=10 的同口径 Recall workload；`query_selected_recall_advantage` 是按历史方法表现后验筛选的集合，只能作诊断。
- 原始 checkout 有大量用户改动，禁止直接 merge；最终仅声明分支/patch merge-ready。
- 服务器 Git 1.8.3.1 不支持 native worktree，当前隔离环境是 `git clone --shared`。
- jump host 偶发断连；只做短时串行重试，避免并发 SSH。
- `runs/`、`thirdparty/acorn-official/`、`thirdparty/curator-v2/` 是未跟踪运行/第三方产物，不得提交。
- 多层查询发现 upper blocks 时禁用语义不完整的 GPU batch path，当前正确性优先，仍有性能优化空间。
- 75% 以上共享边界已闭合，但仍只声称预声明离散网格内的实测最优；未做连续参数优化或统计显著性检验。
- 100% 是空 containment predicate 的独立 unfiltered control；不得据此把 96.7% 与 100% 连成区间，也不得外推 97%--99%。
- build manifest 对复用构建不记录 returncode 与外层 wall；报告生成器已按 `reused_existing=true + validate_case 成功` 接受该状态，并把未知 wall 明确显示为 N/A，仍不把 builder 内部 total_time 冒充 runner wall。
- `sunyahuia600-sunyahui` 当前并非有效 SSH alias；使用 `ssh -l sunyahui sunyahuia6000-jump`。网络失败时仍只串行短重试三次。

## 验证

- `ctest -R 'ung_build_config|special_block_trie|special_block_free_state|special_candidate_queue|lng_block_partition|special_edge_io|filter_validation'` — `通过`：7/7。
- `cmake --build build_ung_rel -j16 --target build_special_block_index search_UNG_index convert_special_edges` — `通过`。
- 真实 `T1=2k,T2=25k` sidecar + 50%/L550 — `通过`：62,938,887 edges，Recall=.8575，10,000 results，0 violations；detail stats 记录 57,897 upper activations。
- `python3 experiments/multilevel_special/validate_selection_sweep.py ...` — `通过`：全部正式内部 sweep。
- `python3 -m unittest -v experiments.multilevel_special.test_multilevel_selection` — `通过`：14/14；新增 builder binary immutable snapshot 回归。
- `python3 -m unittest -v test_generate_paper_results.py` — `通过`：18/18；包含统一 binary、CSV/manifest 网格/repeats、build source、upper-off 消融、current-source 24 点回归、三次 rebuild 鲁棒性、过滤合法性、source-layout audit 和 LF-only CSV 输出检查。
- `cd experiments/multilevel_special && python3 -m unittest -v test_multilevel_selection.py test_generate_paper_results.py` — `通过`：32/32；生成 60 行主结果、209 个内部点、7 行构建结果。
- `source_manifest.csv` 与 `artifact_manifest.csv` 逐文件 SHA-256 核验 — `通过`。
- source-layout 最终回归 — `通过`：完整 CTest 14/14；Python 32/32；fresh two-level 与旧 single-level 各 10,000 个结果、0 filter violations。
- `validate_selection_sweep.py config.amazon_x1_paired_formal_sel{25,50,75}.json` — `通过`：3/3，每个方法 7 repeats、Recall 无漂移。
- `config.amazon_x1_layer_breakdown_smoke.json` — `通过`：三种层数 closure error 绝对值最大约 `3e-12 ms/query`，0 层 authorization=0；仅验证测量协议，不作为最优性能点。
- `cd experiments/multilevel_special && python3 -m unittest -v test_multilevel_selection.py` — `通过`：32/32；包含 formal 输出隔离、选择策略与 coarse source SHA-256 provenance。
- 首轮 7-repeat formal validator — `通过`：78/78 case、468 points；selector 为 18 oracle rows、3 shared configurations，并正确暴露 10 条结构轴边界。
- 最终 7-repeat formal validator — `通过`：92/92 case、552 points；18 oracle rows、3 shared configurations、0 个开放边界命中。
- `cd experiments/multilevel_special && python3 -m unittest -q test_multilevel_selection.py` — `通过`：38/38；覆盖复用 build manifest、formal boundary guard 和严格整数合法域端点。
- `python3 -m unittest -q experiments.multilevel_special.test_generate_paper_results` — `通过`：18/18；旧论文结果闭包仍可重建。
- `python3 -m unittest -q test_multilevel_selection.py test_generate_paper_results.py` — `通过`：56/56。
- `validate_selection_sweep.py config.amazon_x1_high_selectivity_formal.json` — `通过`：95/95 cases，每 case 6 个 L 点、42 条 repeat 明细，stage closure 完整。
- `select_layer_tuning.py ... --boundary-source config` — `通过`：18 个 oracle rows、3 个 shared configurations；审计暴露 13 个结构轴和 16 个 Lsearch 边界命中。
- `cd experiments/multilevel_special && python3 -m unittest -v test_multilevel_selection.py` — `通过`：44/44；覆盖高选择率 extension、互斥点合并、Lsearch>=K 和边界审计。
- 高选择率 boundary extension — `通过`：20/20 cases、149 points；与主 sweep 合并后 719 个唯一点，3 个 shared 配置，shared 结构/Lsearch 边界均为 0。
- `python3 -m unittest -v test_generate_paper_results.py test_multilevel_selection.py` — `通过`：62/62；结果生成器重建 60 行主结果、209 个内部点和 7 行构建结果。
- `high_selectivity_evidence_manifest.csv` 逐文件 SHA-256 核验 — `通过`：10/10；719 点 aggregate 与两份 execution manifest 均匹配。
- `artifact_manifest.csv` 与 `layer_tuning_evidence_manifest.csv` — `通过`：分别核验 43/43 与 12/12 个 SHA-256；文档本地路径 0 缺失；`git diff --check` 通过。
- Curator 低选择率产物 — `通过`：3 workloads x 12 budgets x 5 measured。
- ACORN 低选择率产物 — `通过`：180 个 screen 点 + 3 个 formal 点，filter violations=0。

## 仅在需要时阅读的细节

- `WORKTREE_HANDOFF.md` — 理解实现语义、隔离来源与 merge-back 边界时阅读。
- `docs/reports/MULTILEVEL_SPECIAL_BLOCK_MERGE_BACK_CN.md` — 准备把隔离分支整合回原始脏 checkout 时阅读。
- `experiments/multilevel_special/EXPERIMENT_STATE.md` — 恢复实验、查询 raw artifact 或解释历史隔离结果时阅读。
- `docs/reports/MULTILEVEL_SPECIAL_BLOCK_PAPER_REPORT_CN.md` — 审阅方法与论文结论。
- `docs/reports/MULTILEVEL_SPECIAL_BLOCK_REPRODUCE_CN.md` — 从 compact evidence 审计或重新运行 raw benchmark 时阅读。
- `docs/reports/MULTILEVEL_SPECIAL_BLOCK_HIGH_SELECTIVITY_PLAN_CN.md` — 恢复 75% 以上 workload 生成、调优与结果审计时阅读。

## 恢复说明

1. 先读本看板。
2. 运行 `git status --short`，不要暂存 `runs/` 或 `thirdparty/` nested clones。
3. 核对实现/结果检查点 `7a2bf4635eac43d174100770838daf1b3a10fa58`、最终源码/证据检查点 `cb72797af5af615ca196d3618903417a54b0da2b` 与高选择率文档/证据检查点 `42c21c10696638dbf83a54bd00e37bb88dc474ac`。
4. 任何新数值必须先进入 source CSV 并由生成器输出。

## 清理提示

- 保留 compact aggregate CSV；不提交 raw `runs/` 和大型索引。
- 完成审阅后压缩 `EXPERIMENT_STATE.md` 中已经失效的 hybrid 主图记录。
