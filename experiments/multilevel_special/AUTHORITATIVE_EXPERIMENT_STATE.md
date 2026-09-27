# 权威多层消融实验状态

本文档用于维护 `fcb74ad` 之后的实验协议、问题状态、假设和证据。只有本轮固定
binary、固定数据 provenance、完整 manifest 产生的结果可以进入最终主表。

## 2026-09-27 18:12 State Sync

1. 当前事实：旧 396-case Amazon campaign 已停止，保留 88 个完成 case；新的
   deadline-bounded 协议使用 6 点 `Lsearch`、1 cold + 2 warm 和 3300 秒 case
   超时，优先取得三个 held-out 数据集上的 DRH 与 5 个冻结人工替代方案。
2. 已过时判断：仅把 `benchmark_profile` 改为 `cpu` 并设置旧
   `UNG_SPECIAL_BLOCK_GPU_INTRA=0`，不能保证纯 CPU 构建。
3. 已解决问题 M27：size-routed intra path 由 `UNG_SPECIAL_INTRA_ROUTE` 独立控制；
   Reviews/VariousImg 的失败日志分别记录 19/20 个 GPU intra blocks，validator
   正确拒绝了 profile 与实际 backend 不一致的 artifact。
4. 当前根因：deadline 生成器继承了 frozen held-out grid 的
   `UNG_SPECIAL_INTRA_ROUTE=1` 和 `jasper_style` large backend，却把 profile 改成
   CPU，形成 provenance 矛盾。
5. 验证结果：生成器同时关闭 legacy intra、inter 和 size-routed intra；新 root
   中 Reviews/VariousImg automatic DRH 分别用 120.44/633.16 秒构建完成，metadata
   和日志均记录 GPU intra blocks/inter use 为 0。下一步恢复 bounded query campaign。
6. 新 measurement issue M28：performance pass 默认启用 light stats，layered 路径
   的逐类 edge counters 因此保持 0；这不是“扫描零条边”。deadline 汇总器将这些
   值改报 NA 并显式标记 counter mode，最终边数只引用独立 detailed profile。

## 当前状态

目标指标：在相同 Recall 门槛下比较 batch wall-time/QPS，并独立解释 entry-group、
entry-point setup、block authorization、graph search 以及点/边/距离计算工作量。

当前主要问题：修复后 Amazon broad screen 尚未覆盖全部 44 个方法；完成后还需要
bounded crossing、正式重复和独立 profile。修复前 campaign 使用的 Trie entry
错误地把空标签集解释为“无结果”，其 99% 结果只保留作缺陷诊断。详细计数会给主
吞吐测量增加分支和计数开销，因此不能把 profile wall time 混入主 QPS。
截至 2026-09-27 14:48，manifest 已完成 87/396，另有 1 个 case 运行中；
当前为 `l0_trie_entry_optimized_lng / sel_80`，仍使用 SHA-256 为
`4c99a51c...a81` 的冻结 binary snapshot。
零层 LNG 95% 的
conservative crossing 为 $L=320000$，warm Recall 最小值 0.9134，
warm-median QPS 1.50268；99% crossing 为 $L=400000$，两次 warm Recall
均为 0.9408，warm-median QPS 0.78694，该 case 子进程 wall time 为
5984.49 s。99% crossing 的 warm-median 每查询阶段时间为 entry-group
9820.05 ms、entry-point setup 123.902 ms、graph search 114615.5 ms；中位工作量
为 508269.5 个 visited points、5216405 条 scanned edges 和 535069.5 次
distance calculations，stage closure 最大绝对误差为
$2.68\times10^{-10}$ ms。binary hash 与 campaign snapshot 一致。
零层 Trie 80% 仅在 $L=N=602453$ 达标，两次 warm Recall 均为
0.9699，batch wall time 为 1710.64/1701.88 s，warm-median QPS 为
0.58608。该点 warm-median 每查询 entry-group、entry-point setup 和 graph
search 时间分别为 12.3657、83.3588 和 167524 ms；工作量为 464337 个
visited points、3158390 条 scanned edges 和 485108 次 distance calculations。
零层 Trie 95% 也仅在 $L=N$ 达标，两次 warm Recall 均为
0.9666，batch wall time 为 2044.47/2048.38 s，warm-median QPS 为
0.48866。该点 warm-median 每查询 entry-group、entry-point setup 和 graph
search 时间分别为 13.6413、90.7394 和 201565 ms；工作量为 552827 个
visited points、3759560 条 scanned edges 和 575137 次 distance calculations。
零层 Trie 99% 也仅在 $L=N$ 达标，两次 warm Recall 均为
0.9656，batch wall time 为 2262.49/2263.96 s，warm-median QPS 为
0.44185。该点 warm-median 每查询 entry-group、entry-point setup 和 graph
search 时间分别为 4.4373、187.277 和 222941 ms；工作量为 566231 个
visited points、3858770 条 scanned edges 和 600306 次 distance calculations。

首个两层设置 `1024:lng,16384:trie + optimized_lng entry` 的无路由版和
routed control 九档已完成。无路由版在 0.5%--30% 的最大 Recall 为
0.8878/0.8567/0.8928/0.8492/0.8393，均未达 0.90；在 60%/80%/95%/99%
的 conservative crossing QPS 为 61.403/49.699/39.754/16.791，相对 LNG-0
加速为 15.09x/27.57x/26.46x/21.34x。

routed control 在 0.5%/1%/5%/10% 的 crossing QPS 为
2135.274/4151.919/462.431/287.579，相对 LNG-0 为
0.862x/0.991x/0.849x/0.639x；30% 在 $L\le45000$ 未 crossing。其
60%/80%/95%/99% QPS 为 58.579/43.669/40.125/14.775，相对 LNG-0
为 14.40x/24.22x/26.70x/18.78x。因此 router 能恢复部分低档 Recall，
但额外授权与路由并非免费，不能把中低选择率宣称为严格不变。
据此，held-out 最终自动方案定义为“DRH 静态层级 + 无参数精确
upper-authorization gate”：层数、阈值和 topology 仅由 $N/R/C$ 生成，
gate 仅比较当前谓词与持久化 block-root label，不读取 selectivity、延迟或
Recall。所有 35 个人工 hierarchy 候选使用同一 gate，使 oracle 只调整
层数、阈值和逐层 topology；无 gate DRH 仅作消融。

第二个 query-free 静态 comparator `8192:lng,131072:trie` 的 gated 九档也已
完成。其 0.5%/1%/5%/10% crossing QPS 为
2241.587/4262.756/459.137/420.190，相对 LNG-0 为
0.905x/1.017x/0.843x/0.933x；30% 在 $L\le45000$ 未 crossing，最大 Recall
为 0.8578；60%/80%/95%/99% QPS 为
61.368/39.531/38.279/14.802，相对 LNG-0 为
15.08x/21.93x/25.47x/18.81x。该结构只有 24 个 materialized blocks 和
54,719,258 条 sidecar edges，而 DRH 有 212 个 blocks 和 73,527,844 条
sidecar edges。更粗 comparator 在 10% 更接近 plain，在 80%/95% 则低于
较细 DRH，表明粒度存在可解释的 workload trade-off。

`SpecialBlockSearchUsed` 给出的 DRH gate 启用率依次为
0%/0%/4.3%/35.7%/32.5%/61.8%/82.5%/98.3%/100%；mass-ladder 对应为
0%/0%/4.3%/0%/29.6%/61.4%/82.2%/98.2%/99.3%。该非单调性来自谓词与
block-root label 的精确包含关系，而非用平均选择率阈值路由。0.5% 和 1% 完全
回退 base path；screen 中剩余 QPS 差异主要伴随 entry-group 时间波动，须由
正式重复判断，不能归因于层级搜索。

零层 screen 的 conservative crossing 对比如下。`Trie/LNG` 小于 1 表示
Trie 较慢；这些是 1 cold + 2 warm 的 screen 数据，不替代 formal pass。

| 选择率 | LNG-0 L | LNG-0 QPS | Trie-0 L | Trie-0 QPS | Trie/LNG |
|---:|---:|---:|---:|---:|---:|
| 0.5% | 2000 | 2476.351 | 1000 | 14921.900 | 6.026x |
| 1% | 2200 | 4191.045 | 1200 | 11591.420 | 2.766x |
| 5% | 3200 | 544.932 | 800 | 10044.230 | 18.432x |
| 10% | 26000 | 450.171 | 64000 | 270.630 | 0.601x |
| 30% | 120000 | 52.769 | no crossing ($L\le150000$) | -- | -- |
| 60% | 260000 | 4.069 | 320000 | 2.511 | 0.617x |
| 80% | 320000 | 1.803 | 602453 | 0.586 | 0.325x |
| 95% | 320000 | 1.503 | 602453 | 0.489 | 0.325x |
| 99% | 400000 | 0.787 | 602453 | 0.442 | 0.561x |

交叉控制 `Trie topology + optimized-LNG entry` 在 0.5%/1%/5%/10%/30%
的最高 warm-min Recall 仅为 0.4546/0.4270/0.4370/0.4669/0.4040，均未
crossing。相同 Trie topology 配合 Trie entry 在前四档可以 crossing；在各档
最大已测 $L$，optimized-LNG entry 的平均入口数为
160.5/123.4/789.0/345.3/1578.1，而 Trie entry 为
1277.4/1510.0/2355.4/4714.9/8835.4。这支持 reachability-interaction 解释：
LNG 的最小超集入口依赖非 prefix 的 containment 边继续扩散，而 Trie topology
只保留 prefix 分支；较少入口虽然减少点、边和距离计算，却不能覆盖所有可达分支。
该结果是 screen 机制证据，最终效应大小仍以后续 formal pass 为准。

当前最佳方法：每个结构使用同一个不可变 search binary；先做轻量 performance
pass，再对选中的 Recall crossing 做独立 profile pass。主结论使用 performance
pass 的 wall time，profile pass 只用于机制解释。profile 同时显式标记
`measurement_pass=profile` 与 `protocol.phase=profile`，不再复用 formal 相位名。

截至 2026-09-27 19:07，修复后的 held-out campaign 已完成 8 条 supervisor
记录，正在 Reviews baseline 的 `query_minlen2_cov1k` 上运行。独立 Amazon
构建 deadline 子集已经冻结为 40 个 case：base 与
`auto_drh_v1` hierarchy 分别比较 5 个 backend profile，每个 timing profile 使用
1 cold + 2 measured，resource profile 独立运行 1 次。每个 case 有 3300 秒硬超时；
query supervisor 未写入完成标记前，构建 runner fail closed，因而两类测量不会并发。
第三个独立 profile runner 已实现：它必须等到 query 和 build supervisor
均有完成标记，并确认实验进程退出后，才重编译 gate-timing binary。它仅对
baseline/DRH 在 performance pass 中的 conservative crossing（无 crossing 时为明示
best-measured point）收集 detailed counters，不使用 profile wall time 替代主 QPS。

截至 19:25，baseline/DRH priority screen 的 5 个 held-out workload 已完成。
DRH/plain QPS 比在 Genome 3.365%/6.292% 为 `0.9536x/0.9817x`，在
Reviews 0.200%/4.115% 为 `1.0142x/1.0044x`，在 VariousImg 10.252%
为 `0.2046x`；所有点均按 all-warm Recall crossing 比较。VariousImg 的分解显示
DRH graph-search 约为 167.0 ms/query，plain 约为 34.1 ms/query，而 ELS 反而从
2.21 ms/query 降至 1.58 ms/query；因而该负结果来自被合法启用的层内图搜索，
不是入口组查找。这些仍是 1 cold + 2 warm 的 deadline screen，结论不使用
显著性语言。

截至 20:13，三个 held-out 数据集的 3 个 automatic build、10 个 priority
query 和 15 个 frozen manual build 均已完成，manual query 正在串行运行。
Amazon build 与 detailed-profile supervisor 仍分别等待 query 与 build completion
marker，没有并发污染。

在查看任何新路由结果前，DRH-v2 已冻结为独立后续消融。它不改变 DRH-v1 的
层数、阈值或 topology，而是在查询可授权的最高 upper layer 上计算 direct-point
mass $M_h$，并仅当 $M_h\ge T_{L+1}=\rho T_L$ 时进入 multilevel backend；否则在
创建 overlay seeds 前走原 zero-layer backend。该规则只读 query labels 与持久化
block metadata，不读取 query distribution、latency、Recall 或 selectivity。
协议、源码和 same-binary plain/DRH-v1/DRH-v2 runner 均已在首个 DRH-v2
measurement 前提交；四个数据集的阈值分别为 262144、65536、131072 和 262144。

## 公平性协议

- 数据、query、GT、K、entry-point 数、graph backend 和 Recall 判定规则在同一
  对比中保持一致；QPS 是 `num_threads=100` 的 query-level parallel batch
  throughput，不是单查询、单线程延迟。
- 零层 LNG：base topology=`lng`，entry strategy=`optimized_lng`。
- 零层 Trie：base topology=`trie`，entry strategy=`trie`。
- topology 消融按完整方法比较，也追加 crossed combinations 以分离 topology 与
  entry strategy 的贡献。
- 每个方法先用 coarse/fine Lsearch 找到满足 Recall 的最小实测 crossing；正式
  结果不得用插值代替实测点。
- warm-up 与 measured repeats 分离；正式结果至少 1 次 cold + 9 次 measured，
  关键点采用 1 + 15 次，并报告 median、CV、p95 与 bootstrap 95% CI。
- 主 QPS pass 设置轻量统计；profile pass 开启详细计数和 timing，不把 profile
  pass 的 wall time 混入主 QPS。
- 12 小时门槛按 manifest 中成功完成的 build/search 子进程 active seconds 求和，
  不计等待、排队或人为 sleep。
- 原始 CSV、命令、环境、binary hash、index meta/hash、主机与 GPU 快照全部保留。

## Issue Ledger

| ID | 类型 | 状态 | 描述 | 判定/下一步 |
|---|---|---|---|---|
| M1 | measurement | RESOLVED | 旧 runner 使用五种历史 provider 名且限制两层 | 已迁移到 original/optimized_lng/trie 与任意 hierarchy plan；118 个 Python tests 通过 |
| M2 | measurement | RESOLVED | detail counters 会污染主吞吐 | 已分离 performance/profile pass，并在 manifest 标注用途 |
| M3 | measurement | ACTIVE | 旧 QF-SSL 数据与新 binary/语义不一致 | 所有主表继续只采用 `fcb74ad` 后重测结果 |
| M4 | correctness | RESOLVED | Trie entry 将合法空 containment 谓词返回为空，污染含空谓词的 99% workload | build/load 时预计算 root terminal frontier；C++ 15/15、Python 167/167，真实空谓词 `L=N` Recall@10=1.0；修复前 campaign 标记无效并以新 hash 重跑 |
| M5 | measurement | RESOLVED | 通用汇总器的 `recall_min` 曾包含 cold repeat，与 crossing 协议不一致 | `recall`/`recall_min`/`recall_max` 和 warm timing 统一按声明的 `cold_repeats` 切分；增加冷启动 Recall 不影响 crossing 的回归测试 |
| M6 | measurement | RESOLVED | 旧辅助汇总在多个达标点中按最快时间选点，可能受噪声影响而偏离 crossing 定义 | 改为每个方法选择所有 warm repeat 达标的最小实测 `Lsearch`；禁止插值或按延迟回选更大 L |
| M7 | measurement | RESOLVED | formal performance 与 profile 共享 output root 时，旧汇总路径可能互相覆盖 | `pass_subdirs` 配置下将汇总隔离到 `summary/performance` 与 `summary/profile` |
| M8 | output schema | CONTAINED | 当前 snapshot 的 `search_time_summary.csv` 数据行含 `AverageNodesVisited`，但表头漏写该列，导致后续工作量列错位 | 主 Recall/latency 前四列不受影响；论文工作量只读取列宽正确的 `search_work_details.csv`。源码表头已修复，但为保持 screen/crossing/formal binary hash 一致，在本轮 campaign 完成前不重编译 |
| M9 | reporting | RESOLVED | `results.md` 曾使用 mean-Recall crossing，而严格协议要求所有 warm repeat 均达标 | Markdown 主表改用 conservative crossing，同时显示 warm mean、warm min 和 `min-target` margin；CSV 仍同时保留 mean 与 conservative 版本 |
| M10 | reporting | RESOLVED | 通用 Markdown 只展示 Recall/QPS，用户要求的阶段和工作量证据需另行解析 CSV | 每个 conservative crossing 现自动报告五段 timing、visited points、base/special edges、entry/graph/total distance calculations 和 entry 数；profile pass 使用同一格式 |
| M11 | provenance | RESOLVED | profile 的 measurement pass 曾被记为 formal protocol phase，证据语义含混 | 新增显式 `profile` phase，并强制 measurement pass 与 protocol phase 成对；167/167 Python tests 通过 |
| M12 | measurement | RESOLVED | GPU build 无外部锁时只有一次手工空闲检查，不足以支撑权威 timing | runner 优先在 `gpulock perf` 下重执行；当前主机无 `gpulock` 时，每个 GPU case 强制三次连续空闲预检并在 manifest 明示 no-lock 限制，缺证据的 artifact 校验失败 |
| M13 | measurement | ACTIVE | 当前 immutable query snapshot 在 exact upper-authorization gate 之前开始总计时，但没有把 gate 本身计入 `BlockAuthorizationTime_ms`；该时间落入 residual，主 QPS/Recall 不受影响 | 源码已把 gate 与后续 coverage 时间累加为完整 authorization stage；`run_authoritative_instrumented_profile.py` 已实现独立 build、活跃进程拒绝、formal-L 复用及 manifest hash 校验，待当前 query/build 流水线退出后执行 |
| M14 | provenance | RESOLVED | held-out policy 的 `manual_hierarchy_cases=36` 会把包含 DRH 的总候选数误读成 36 个额外人工方案 | policy 改为 `frozen_hierarchy_candidates=36` 与 `manual_alternatives=35`，validator 同时核对字段、角色计数和闭合关系并拒绝旧字段；六份冻结 build/search config hash 未变 |
| M15 | provenance | RESOLVED | 最终 query validator 曾只核对少数命令选项，无法完整证明 query/GT/index 路径、环境和 executable snapshot 均与配置一致 | validator 现逐 case 重建完整命令与 `UNG_*` 环境，校验 content-addressed snapshot 路径、文件名/内容 hash，并绑定 manifest；86/86 完成项通过严格回扫 |
| M16 | reporting | RESOLVED | 中文报告曾缺少 LaTeX 已有的全局 depth 与完整零层组合表，且单独调用 Markdown renderer 会绕过部分结果校验 | Markdown 现补齐九档全局配置、零层 2x3 topology/entry crossing 和 GPU lock/idle 证据，并自行执行 depth、held-out、build fail-closed 校验；executable hash 使用文件状态键缓存，mutation 回归仍通过 |
| M17 | measurement | RESOLVED | build 汇总曾将独立 base/hierarchy phase 的同编号 repeat 相加并称为 paired end-to-end measurement | 完整索引点估计改为两个阶段中位数之和，CI 对 original base、accelerated base、hierarchy 独立 bootstrap，并拒绝样本数不平衡；文档统一标为 stage-composed full-index cost |
| M18 | reporting | RESOLVED | held-out oracle 的两个旧计数字段名称含 manual，但实际候选集合包含 DRH 本身 | 汇总器删除歧义字段，改报 feasible/complete oracle candidates，并硬性要求候选角色计数为 1 DRH + 35 manual = 36 |
| M19 | reporting | RESOLVED | 主表和七类图只显式覆盖 44 个 Amazon screen 方法中的代表性子集，完整方法结果只能从 raw points 手工恢复 | 中文报告和论文单栏附录新增 fail-closed 44x9 crossing 矩阵：crossing 报 L/QPS/warm-min Recall，NC 报最高实测 Recall@L，并明确 2-repeat screen 不替代 15-repeat formal 结论 |
| M20 | reporting | RESOLVED | profile 报告曾只要求 breakdown 字段存在、有限且 stage timing 闭合，仍可能接受负值或总计小于分项的物理矛盾证据 | 生成器新增非负约束，并要求 total edges/distances 不小于任一组成分项；三项负向回归使专项测试增至 30 项、全量测试增至 179 项 |
| M21 | reporting | RESOLVED | 构建报告曾只要求至少一个 base 与一个 hierarchy 行，部分 backend profile 或整个结构缺失时仍可能生成 | 生成器现在精确要求 5 个 base profile、2x5 个 hierarchy profile 和对应 10 个 stage-composed 行；两项缺行回归使专项测试增至 32 项、全量测试增至 181 项 |
| M22 | provenance | RESOLVED | build manifest 虽记录 binary hash，但最终报告验证未核对 command 实际执行的 snapshot，也未禁止 timing/resource pass 混用不同 builder binary | 逐 case 解析 command、重算 snapshot SHA256 并与 manifest 绑定；同一 component 两种 pass 的 binary hash 必须唯一；两项回归使专项测试增至 34 项、全量测试增至 183 项 |
| M23 | provenance | RESOLVED | build runner 断点续跑时只验证 artifact 结构，可能用当前 provenance 覆盖由旧 binary 生成的既有产物 | 自产 build artifact 只有在旧 manifest 存在且原始 builder hash 与当前 snapshot 相同时才允许复用；缺失 provenance 或 binary drift 均 fail closed，三项回归使全量测试增至 186 项 |
| M24 | reporting | RESOLVED | 论文只写“相同线程数”，可能把权威 QPS 误读为单查询单线程延迟 | 论文与中文报告明确标注 100 个查询工作线程的 query-level parallel batch throughput；同源生成器要求 screen/formal/profile/held-out/build-quality 的 `num_threads` 完全一致，一项回归使全量测试增至 187 项 |
| M25 | methodology | RESOLVED | screen 的逐方法 Lsearch 网格上限不同，旧 crossing 仅将未过线方法末点扩展 1.5x，可能在低于同 workload 其他方法已测预算处过早宣告 NC | crossing 现在为每个 workload 计算共同预算：默认取所有方法已测最大 Lsearch，显式 `max_lsearch` 则作为统一上限；未过线方法一次细化到该共同预算，formal 与最终报告均拒绝未实际测到共同预算的 NC，并核对 disabled cell 与 NC 记录一一对应；相关回归使全量测试增至 189 项 |
| M26 | reporting | RESOLVED | 共同 Recall 搜索预算仅存在于 formal provenance，论文和中文报告中的 NC 无法直接对应具体 workload 上限 | LaTeX 与 Markdown 生成入口现在都直接校验 formal selection provenance，并从同一个 `shared_max_lsearch_by_workload` 动态生成九档预算表；缺失、不完整或与 disabled/NC cell 不闭合时拒绝生成。一项双入口回归使全量测试增至 190 项，含真实动态表的 Tectonic fixture 编译通过 |
| M27 | measurement | RESOLVED | deadline held-out case 声明为 CPU profile，但继承的 size-routed intra path 仍调用 Jasper/Tagore CUDA | 保留 fail-closed validator；生成器显式把 legacy intra/inter 与 `UNG_SPECIAL_INTRA_ROUTE` 全部置 0；Reviews/VariousImg 真实 automatic build 均通过且 verified GPU counters 为 0 |
| M28 | measurement | ACTIVE | performance pass 的 light stats 会让 layered special-edge counters 为 0，旧汇总表若不标注会误读为未扫描边 | deadline 汇总器已把 light-pass edge counters 改报 NA；待新 binary detailed profile 提供 base/special intra/inter 边数并同时解决 M13 authorization 归因 |
| M29 | measurement | RESOLVED | 原 authoritative build campaign 的 5 measured x 15 profile 不能在 deadline 内稳定完成，且缺少逐 case 硬超时与 query/build 隔离门 | 从 frozen configs 生成 1 cold + 2 measured 的 5 base + 5 DRH backend 子集，资源测量独立运行；每 case 独立进程组、3300 秒硬超时、原子 supervisor manifest 和 query completion gate；8 项专项及全量 205/205 测试通过。该结果只标为 deadline screen，不冒充正式 5-repeat 证据 |
| M30 | measurement | ACTIVE | M13/M28 需要新 binary 的真实 gate timing 与 detailed edge/work counters，但 profile 不得污染主 QPS 或 GPU build timing | 独立 prepare/runner 已于 `5117e76` 提交，两级 completion gate + process-idle gate 保证串行；全量 213/213 测试通过，待 query/build 完成后实测 |
| M31 | methodology | RESOLVED | DRH-v1 的 nonempty exact gate 无法阻止合法但过小的 upper region 放大 bounded-search work，且直接累计所有 upper layers 会在三层以上重复计点 | DRH-v2 只累计最高可授权层的互斥 direct members，以 DRH 下一未物化尺度作 gate；不足时在 outer router 真正回退 base。协议在测量前冻结，同 binary ablation 使用 plain/DRH-v1 冻结网格的并集 |
| H1 | hypothesis | PARTIAL | 无条件多层在高选择率降低 graph work，但低选择率未必保持零层 Recall；结构授权 router 可能恢复零层路径 | Amazon routed control 九档验证 |
| H2 | hypothesis | PARTIAL | Trie 与 LNG topology 的优劣由标签包含结构及 entry frontier 的 reachability 交互决定，而非选择率单独决定 | crossed `Trie topology + optimized-LNG entry` 在 0.5%--30% 均未 crossing，但原生 Trie entry 在前四档 crossing；待其余 topology/entry combinations 和 formal pass |
| H3 | hypothesis | PARTIAL | gated DRH-v1 可由 `N/R/C` 决定层数、阈值和逐层 topology，但 exact gate 只保证无合法 upper 时回到 base，不保证所有中低选择率无损 | held-out priority screen 在 Genome/Reviews 四点为 0.954--1.014x，VariousImg 10.252% 为 0.205x 且由 graph-search 放大解释；待同 gate 的 5 个 deadline manual alternatives 量化自动方案 regret。四个数据集都导出两层，本轮不宣称实证了自动深度变化 |
| H4 | hypothesis | ACTIVE | 下一尺度 direct-mass gate 可在不做 query/performance 校准的前提下拒绝规模不足的 upper route，恢复中低选择率接近 plain 的路径，同时保留高选择率可用 overlay | 协议已冻结；待同 binary plain/DRH-v1/DRH-v2 在 Amazon 九档及三个 held-out 数据集上的 screen。无论正负均保留，且不宣称单调加速定理 |

## 实验阶段

| 阶段 | 目的 | 完成条件 | 状态 |
|---|---|---|---|
| E0 | runner 与统计 smoke | provenance、Recall、stage closure、work counters 全通过 | 完成 |
| E1 | Amazon broad screen | 九档、零/一/多层、Trie/LNG、coarse L 完整 | 进行中 |
| E2 | Amazon crossing/formal | 每方法每档最小实测 crossing，正式重复完成 | 未开始 |
| E3 | 多数据集验证 | Reviews/Genome/VariousImg 至少各一个低档和一个较高档或可用代表档 | DRH/plain priority screen 完成；5 个 frozen manual alternatives 正在运行 |
| E4 | 自动策略 | query-free 输出层数、阈值、逐层 topology，并与声明网格 oracle 比较 | DRH-v1 oracle comparison 正在运行；DRH-v2 next-scale gate 及 same-binary ablation 已预注册并排队 |
| E5 | 报告 | 原始证据可追溯，表格/曲线/限制完整 | 同源 fail-closed LaTeX/中文报告生成器及 37 项专项测试完成；中文报告与论文附录均含完整 44x9 screen crossing 矩阵和同源九档共同搜索预算，等待权威输入 |

## 已知数据

- Amazon：602,453 points，九档实测平均选择率约 0.499%、0.903%、5.038%、
  9.907%、30.027%、60.047%、80.024%、95.020%、99.001%，每档 1,000 queries。
- Genome：108,077 points，DRH-v1=`256:lng,4096:trie`，3.365% 和 6.292%。
- Reviews：288,065 points，DRH-v1=`512:lng,8192:trie`，0.200% 和 4.115%。
- VariousImg：758,935 points，DRH-v1=`1024:lng,16384:trie`，10.252%。
- 三个 held-out 数据集均固定 36 个 hierarchy candidates（1 DRH + 35 manual
  alternatives）和 38 个查询方法；其余两个方法是零层 baseline 与 DRH ungated
  ablation。该网格在读取 held-out Recall/latency 前生成，自动方案不读取 query
  distribution。
- 机器：2 x Intel Xeon Platinum 8360Y，144 logical CPUs；NVIDIA L20 46,068 MiB。

## 修复前仅作诊断的 screen 观测

以下数字来自修复前 binary，只能说明候选结构和运行成本，不能进入最终主表或与
修复后结果组成同一公平比较。尤其是所有 `entry_strategy=trie` 且 workload 含空
谓词的结果无效。

- 固定 binary SHA256：
  `3ae4fe9afe7bea7dd7ee382e86c5de6dd77abc5a31e5e569c1126fe8cc63ce4a`。
- DRH-v1 `1024:lng,16384:trie` 在 60%、80%、95%、99% 的最小已测
  Recall@10>=0.90 点分别为 `L=5000/5000/2500/2500`；对应 warm batch 中位数
  约为 `17.390/22.087/25.102/59.646 s`。这些是 3-repeat screen 数据，不替代
  后续正式重复。
- 同一无 router 结构在 0.5%、1%、5%、10%、30% 当前网格上的最大 Recall 分别为
  `0.8878/0.8567/0.8928/0.8492/0.8393`，因此尚不能在 Recall=0.90 下比较速度。
  这反驳了“增加层后低选择率天然不变”的强假设，但不能预判结构授权 router 的结果。
- LNG-zero 的 Recall=0.90 screen crossing 在 60%、80%、95%、99% 的 warm batch
  中位数约为 `246.812/551.695/672.675/1272.810 s`；相对该 screen baseline，
  DRH-v1 暂时对应 `14.19x/24.98x/26.80x/21.34x`。最终结论必须使用 formal pass
  和 bootstrap 区间。

## 下一最小实验

继续轮询已在 tmux 中运行的 held-out query supervisor，不重启现有进程；
保持 Amazon build waiter 和 detailed-profile waiter 按 query -> build -> profile 顺序串行。
query 完成后生成 strict conservative crossing，build 完成后汇总独立 GPU 构建证据，
profile 完成后解决 M13/M28/M30 并回填报告。旧 44-method full screen 不再恢复。
