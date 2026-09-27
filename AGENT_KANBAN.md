# Agent 看板

最后更新：`2026-09-27 16:43 Asia/Shanghai`
分支：`codex/multilevel-special-block-20260905`
检查点：`cd1fda1`（push：`不执行，origin 指向用户工作树`）

## 目标

在统一的任意层 hierarchy / 逐层 LNG-Trie topology / 三种 entry strategy
结构上完成至少 12 小时的公平消融，给出不同选择率的 Recall-QPS 曲线、阶段与
搜索工作量分解，并在多数据集上验证无需 query 校准的自动分层方法及其相对人工
网格最优的差距。

## 当前状态

- 总体：`进行中`
- 摘要：新结构已在 `fcb74ad` 完成；Trie entry 对空 containment 谓词的历史错误
  已修复，并通过 15/15 CTest、187/187 Python tests 和 Amazon 单查询端到端验证。实验框架已
  迁移到 `orthogonal_v2`，明确分离 hierarchy、逐层 topology 和三种 entry
  strategy，并采用轻量 performance pass 与独立 profile pass。旧 QF-SSL 数据仅作
  候选假设，不能作为本轮结论。
- 当前主要风险：零层方法在 80%--99% 选择率需要极大的 `Lsearch`。不得为赶时间
  缩短 query 数或把未过 Recall 的点纳入主表；修复前 binary 的结果不得与修复后
  binary 混入同一比较。
- 当前 campaign 使用 hash `4c99a5...` 的不可变 snapshot；源码已补上 summary CSV
  漏失的 `AverageNodesVisited` 表头，但在 screen/crossing/formal/profile 全部结束前
  不得重编译 `build_ung_rel/apps/search_UNG_index`。工作量主表读取未错位的
  `search_work_details.csv`。
- 截至 `2026-09-27 16:43`，manifest 已完成 88/396，另有 1 个 case 运行中；
  当前为 `l0_trie_entry_optimized_lng / sel_95`，仍使用冻结的
  content-addressed binary。LNG-0 的 95%
  conservative crossing 为 `L=320000`，warm Recall min/max 为
  `0.9134/0.9136`，warm-median QPS 为 `1.50268`；99% crossing 为
  `L=400000`，warm Recall min/max 均为 `0.9408`，warm-median QPS 为
  `0.78694`。
- Trie-0 的 80% 只在 `L=N=602453` 达标，两次 warm Recall 均为
  `0.9699`，batch wall time 为 `1710.64/1701.88 s`，warm-median QPS
  为 `0.58608`。该点 warm-median 每查询 entry-group/entry-point/graph-search
  时间为 `12.3657/83.3588/167524 ms`，工作量为 `464337` 个 visited
  points、`3158390` 条 scanned edges 和 `485108` 次 distance calculations。
- Trie-0 的 95% 同样只在 `L=N` 达标，两次 warm Recall 均为
  `0.9666`，batch wall time 为 `2044.47/2048.38 s`，warm-median QPS
  为 `0.48866`。该点 warm-median 每查询 entry-group/entry-point/graph-search
  时间为 `13.6413/90.7394/201565 ms`，工作量为 `552827` 个 visited
  points、`3759560` 条 scanned edges 和 `575137` 次 distance calculations。
- Trie-0 的 99% 也只在 `L=N` 达标，两次 warm Recall 均为
  `0.9656`，batch wall time 为 `2262.49/2263.96 s`，warm-median QPS
  为 `0.44185`。该点 warm-median 每查询 entry-group/entry-point/graph-search
  时间为 `4.4373/187.277/222941 ms`，工作量为 `566231` 个 visited
  points、`3858770` 条 scanned edges 和 `600306` 次 distance calculations。
- 两层 `1024:lng,16384:trie + optimized_lng entry` 的无路由与 routed
  control 九档均已完成。无路由版在 0.5%--30% 未达 Recall 0.90，
  但在 60%/80%/95%/99% 为 `61.40/49.70/39.75/16.79 QPS`；
  routed 版在 0.5%--10% 恢复 crossing，但相对 LNG-0 仍有约
  `14%/1%/15%/36%` 开销，30% 在 `L<=45000` 未 crossing。这些数据
  反驳“增层天然不损低选择率”，也说明 router 成本必须显式报告。
- gated `8192:lng,131072:trie` comparator 九档已完成：0.5%/1%/5%/10%
  相对 LNG-0 为 `0.905x/1.017x/0.843x/0.933x`，30% 未 crossing，
  60%/80%/95%/99% 为 `15.08x/21.93x/25.47x/18.81x`。它比 DRH 粗，
  在 10% 更接近 plain、在 80%/95% 更慢，支持“粒度权衡”而非单调层数结论。
- exact gate 的分层启用率在 DRH 九档上为
  `0%/0%/4.3%/35.7%/32.5%/61.8%/82.5%/98.3%/100%`；它由谓词与
  block-root label 的包含关系决定，不是按平均选择率阈值路由。
- 当前 immutable binary 未单独计时 exact gate，故 gate 成本被归入 residual；
  主 QPS/Recall 不受影响。源码 `7f723b6` 已修正 authorization 归因，但不在
  query campaign 中重编译，结束后用独立 instrumented profile 验证。
- 根据这一 development-set 证据，held-out 的最终自动方案已严格定义为
  “DRH 静态层级 + 无参数精确 upper-authorization gate”。为公平比较，
  35 个人工 hierarchy 候选也全部使用相同 gate；无 gate DRH 仅作消融。
  Genome/Reviews/VariousImg 配置已重新生成，每个仍为 38 个方法；每套包含
  1 个零层 baseline、1 个 DRH routed、35 个 manual routed alternatives 和
  1 个 DRH ungated ablation。168/168 Python tests、三套 build dry-run、
  三套零层 search dry-run 和本地 Tectonic 论文编译通过。
- held-out policy 元数据现明确记录
  `frozen_hierarchy_candidates=36` 与 `manual_alternatives=35`，并拒绝旧的
  `manual_hierarchy_cases` 歧义字段。重生成前后三套 build/search config 的
  SHA-256 完全一致；仅三份 policy 的元数据及 hash 变化。
- held-out 汇总器现在从方法角色硬性核验 oracle 集合恰为 1 个 DRH 加 35 个
  manual alternatives，并将结果字段命名为 `feasible_oracle_candidates` 和
  `complete_oracle_candidates`；不再保留会把 DRH 误称为 manual 的旧列名。
- 最终中文报告与论文单栏附录新增完整 44-method x 9-workload screen crossing 矩阵；每格报告
  最小实测 crossing 的 L/QPS/warm-min Recall，或未 crossing 时的最高实测
  Recall 与 L。该矩阵明确标为 2-warm-repeat 候选筛选，并链接逐 L 原始点；
  15-repeat formal 表仍是性能结论的唯一依据。
- Amazon 与三个正式 held-out 数据集的 DRH 都实例化为两层；报告和论文将其明确
  列为证据边界。Music/Tiktok/Laion 按静态规则会产生三层，但当前缺少本轮协议
  可直接使用的 exact GT，因此只作为结构性观察，不冒充性能验证。
- 最终 query validator 现在逐 case 重建完整命令和 `UNG_*` 环境，要求执行文件
  位于该 phase 的 content-addressed snapshot 目录、文件名 hash 与实际内容一致，
  并将实测 executable hash 与 manifest 逐项绑定。当前 86 个完成项全部通过；
  相同文件状态的 executable digest 只计算一次，文件 size/mtime/ctime 变化会使
  缓存失效，保留 binary mutation 检测。
- 中文权威报告现在与 LaTeX 使用同一组已验证输入，并补齐九档全局 depth 汇总、
  零层 2 topology x 3 entry strategy 的完整 crossing 表和 GPU lock/idle 证据。
  Markdown 入口自身也重复执行 depth、held-out 与 build fail-closed 校验。
- 完整索引构建对比不再把独立 base/hierarchy phase 中相同 repeat 编号视为统计
  配对。点估计是两个阶段 wall-time 中位数之和，95% CI 对 original base、
  accelerated base 和 hierarchy 三组样本独立 bootstrap；论文明确称为
  stage-composed full-index cost，不冒充一次连续 child process 的 wall time。
- crossed-entry screen 显示 `Trie topology + optimized-LNG entry` 在
  0.5%--30% 的最高 warm-min Recall 仅为 0.404--0.467，而原生 Trie entry
  在前四档 crossing；较少入口降低搜索工作但无法覆盖 Trie 的全部 prefix 分支。
  这说明模块接口正交不等于效果无交互，最终差距仍由 formal pass 确认。
- profile 证据现显式使用 `measurement_pass=profile` 和
  `protocol.phase=profile`；两者不一致时配置校验直接失败。
- screen 接续 watcher PID 为 `25189`；query 完成后串行启动
  held-out 与 build 的 watcher PID 为 `46105`，目标 tmux 为
  `fv_heldout_then_build_20260926`。

## 进行中

- 修复前 `fv_auth_campaign_20260925` 已停止并保留为 invalidated diagnostic。
  该 binary 对空谓词返回零个 Trie entry；99% workload 含 697/1000 个空谓词，
  因而 Trie-entry 结果不能用于算法比较。
- 正在以新的内容寻址 binary 和独立 output root 重启完整 Amazon 44-method x
  9-workload screen，确保所有对比共享同一修复后 binary。
- Genome、Reviews、VariousImg 的 DRH-v1 与 36-candidate frozen hierarchy set
  （1 DRH + 35 manual alternatives）配置已
  生成并通过 source/build dry-run、零层 query/GT dry-run 和静态一致性检查；待
  Amazon campaign 释放机器后由 `run_heldout_campaign.py` 继续。

## 完成历史

- exact-level 多层搜索初版与旧 QF-SSL 实验完成 — 证据：`4585099` 至 `cda3c75`；
  这些结果均标记为 pre-refactor 历史证据。
- hierarchy、逐层 topology、entry strategy 已解耦，搜索不混扫层级且无隐式晋级 —
  证据：`fcb74ad`。
- `fcb74ad` clean build、CTest 15/15、三个 benchmark 脚本 `bash -n`、diff check
  均通过。
- Amazon authoritative hierarchy/base 索引已生成；主 screen 包含 42 个正交
  method 和 2 个 query-independent upper-authorization controls，共
  44-method x 9-workload；
  搜索 binary snapshot SHA256 为
  `4c99a51c0f74b187d37f7070f73017b6230935126275560fff1061fa90830a81`。
- DRH-v1 已在观察本轮 layered 结果前冻结：`T1=nearestPow2(sqrt(N))`、
  `rho=max(2,round(R/C))`、`N/T<C` 停止、按 `N/T>R` 选择 LNG/Trie；Amazon
  输出为 `1024:lng,16384:trie`。层数已在论文中写成闭式
  `max(0, 1 + floor(log_rho(N/(C*T1))))`，并明确该代价仅是静态代理而非最优性定理。
- 查询/建图 runner 的 provenance、binary snapshot、resume elapsed ledger、GPU
  backend/fallback 校验和 build-quality 再搜索链已实现；汇总表显式记录完整阈值、
  逐层 topology、entry strategy 与 routing policy；Python 实验测试 168/168 通过。
- GPU build runner 已加入 fail-closed 隔离证据：若有 `gpulock` 则整个
  campaign 在 `perf` 锁下运行；当前主机无该工具，因此每个 GPU case 需
  通过三次连续空闲快照并记录 `idle_preflight_no_lock` 限制。
- held-out 汇总器严格使用 warm-repeat Recall 最小值的首个实测 crossing，并输出
  DRH/plain、DRH/per-workload oracle 和 DRH/global-configuration oracle；顺序运行
  的方法采用独立样本 bootstrap。
- 通用 `results.md` 现在使用 conservative crossing，并分别报告 warm mean Recall、
  warm min Recall 和 min-target margin；避免将均值达标误解为所有重复均达标。
- `results.md` 的每个达标点自动输出 entry-group、entry-point setup、
  authorization、graph search 阶段耗时，以及 visited points、base/special edges、
  entry/graph distance calculations 和 entry 数；选择率按数值升序展示。
- Recall/QPS 六类消融图、plain/单层最优/双层最优/DRH 统一汇总器和
  campaign 结束后的 fail-closed 自动产物链已实现；论文补充了 hybrid GPU sidecar
  构建伪代码，Tectonic 编译成功。
- Trie empty-predicate root terminal frontier 改为 build/load 时预计算；Amazon
  真实空谓词在 `L=N=602453` 的单查询 Recall@10 为 1.0，而旧实现为 0。
- 当前九档 campaign 已有独立的 fail-closed 论文结果生成器
  `generate_authoritative_paper_results.py`：它会重跑 Amazon formal/profile、
  三个 held-out formal 和 build-quality validator，核对四个 build manifest、
  conservative crossing、profile L、分层路径启用率与重复数，并只在全部通过后
  原子替换 `generated_results.tex`。合成 fixture 的 7 个回归测试已通过；历史六档
  `generate_paper_results.py` 保持不变。
- `run_authoritative_instrumented_profile.py` 已把 M13 的重测路径固化：等待所有
  query/build 进程退出后，在独立 `build_ung_profile_instrumented` 中构建，不改动
  performance/held-out 使用的 immutable binary；profile config 从已验证 formal
  crossing 派生，最终强制核对 manifest binary SHA256。4 项 fixture 测试及当前
  活跃实验拒绝启动检查均通过。

## 下一步

完成 Amazon screen/crossing/formal/profile，再串行运行
Reviews/Genome/VariousImg 留出验证和 GPU 建图 timing/resource/quality 矩阵，
最后生成自动方案、人工 oracle、零/一/二层、逐阶段查询与建图对比。

## 阻塞与问题

- 无外部阻塞；Amazon 修复后 screen 使用 CPU，L20 不用于该查询。
- 不修改 `/home/graphdb/FilterVectorCode_refactor`。
- 不提交 `runs/`、两个 nested `thirdparty/` 目录或误生成的长文件名。

## 验证

- `cmake --build build_ung_rel --clean-first -j16` — `通过`：`fcb74ad` 前最终构建。
- `env LC_ALL=C LANG=C ctest --output-on-failure` — `通过`：15/15。
- 新实验框架测试 — `通过`：Python unittest 187/187，`git diff --check` 通过；
  含实际生成 screen 附录的完整 `main.tex` 已通过本地 Tectonic 编译。
- Breakdown 与报告校验 — `通过`：34/34 专项测试；生成器拒绝负耗时/负工作量、
  总边数小于任一边类别、总距离计算小于任一距离分项，以及不闭合的 stage timing。
  构建表还必须完整覆盖 5 个 base profile、2 种 hierarchy 结构乘 5 个 profile，
  以及对应 10 个 stage-composed 行，缺失或额外组合均拒绝生成。
  每个 build case 的实际 command binary snapshot 还必须与 manifest hash 一致，
  且同一 component 的 timing/resource pass 不得混用不同 builder binary。
  Authoritative build resume 只复用具有旧 manifest 且 builder hash 不变的自产物；
  无 provenance 或跨 binary 的产物必须显式重建。
  论文和中文报告现在明确把 QPS 定义为 100 个查询工作线程的 query-level
  parallel batch throughput；生成器拒绝各查询阶段 `num_threads` 不一致。
- Trie 空谓词回归 — `通过`：root frontier 单元测试覆盖 build/load 两条路径；Amazon
  空谓词单查询 `L=N` 得到 Recall@10=1.0。
- 12 小时有效 build/search 采集 — `进行中`；manifest 会分别标记单调时钟实测值
  和由旧 artifact mtime 恢复的近似值，并在验证时强制总量不少于 43,200 秒。

## 仅在需要时阅读的细节

- `experiments/multilevel_special/AUTHORITATIVE_EXPERIMENT_STATE.md` — 实验协议、
  issue ledger、假设与证据清单；恢复实验或解释结论时阅读。
- `docs/reports/QF_SSL_AUTO_POLICY_REPORT_CN.md` — 旧 QF-SSL 方法与历史结果；仅作
  候选启发式来源，不得与新 binary 数据混合。

## 恢复说明

1. 先读本看板和 `AUTHORITATIVE_EXPERIMENT_STATE.md`。
2. 运行 `git status --short`，确认未暂存 `runs/`、`thirdparty/` 和误生成文件。
3. 核对 HEAD、实验 runner 的 PID/manifest 和 binary SHA256。
4. 若长任务仍在运行，只轮询现有进程；不要重新启动同一矩阵。
5. 从“下一步”继续，任何结论只引用新结构 binary 生成且 provenance 完整的数据。

## 清理提示

- `runs/` 保存原始证据，不纳入 Git。
- 旧 config/result summary 保留用于历史追溯，但最终报告必须分开标注。
