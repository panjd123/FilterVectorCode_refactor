# Agent 看板

最后更新：`2026-09-26 19:56 Asia/Shanghai`
分支：`codex/multilevel-special-block-20260905`
检查点：`a083232`（push：`不执行，origin 指向用户工作树`）

## 目标

在统一的任意层 hierarchy / 逐层 LNG-Trie topology / 三种 entry strategy
结构上完成至少 12 小时的公平消融，给出不同选择率的 Recall-QPS 曲线、阶段与
搜索工作量分解，并在多数据集上验证无需 query 校准的自动分层方法及其相对人工
网格最优的差距。

## 当前状态

- 总体：`进行中`
- 摘要：新结构已在 `fcb74ad` 完成；Trie entry 对空 containment 谓词的历史错误
  已修复，并通过 15/15 CTest、103/103 Python tests 和 Amazon 单查询端到端验证。实验框架已
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
- 修复后 screen 已完成 6/396 个 method-workload case：零层 LNG 的
  0.5%/1%/5%/10%/30%/60%；当前在运行 80% 高 L 网格。

## 进行中

- 修复前 `fv_auth_campaign_20260925` 已停止并保留为 invalidated diagnostic。
  该 binary 对空谓词返回零个 Trie entry；99% workload 含 697/1000 个空谓词，
  因而 Trie-entry 结果不能用于算法比较。
- 正在以新的内容寻址 binary 和独立 output root 重启完整 Amazon 44-method x
  9-workload screen，确保所有对比共享同一修复后 binary。
- Genome、Reviews、VariousImg 的 DRH-v1 与 36-case frozen manual grid 配置已
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
  输出为 `1024:lng,16384:trie`。
- 查询/建图 runner 的 provenance、binary snapshot、resume elapsed ledger、GPU
  backend/fallback 校验和 build-quality 再搜索链已实现；汇总表显式记录完整阈值、
  逐层 topology、entry strategy 与 routing policy；Python 实验测试 103/103 通过。
- held-out 汇总器严格使用 warm-repeat Recall 最小值的首个实测 crossing，并输出
  DRH/plain、DRH/per-workload oracle 和 DRH/global-configuration oracle；顺序运行
  的方法采用独立样本 bootstrap。
- 通用 `results.md` 现在使用 conservative crossing，并分别报告 warm mean Recall、
  warm min Recall 和 min-target margin；避免将均值达标误解为所有重复均达标。
- `results.md` 的每个达标点自动输出 entry-group、entry-point setup、
  authorization、graph search 阶段耗时，以及 visited points、base/special edges、
  entry/graph distance calculations 和 entry 数；选择率按数值升序展示。
- Trie empty-predicate root terminal frontier 改为 build/load 时预计算；Amazon
  真实空谓词在 `L=N=602453` 的单查询 Recall@10 为 1.0，而旧实现为 0。

## 下一步

完成 Amazon screen/crossing/formal/profile，再运行 Reviews/Genome/VariousImg 留出
验证，最后生成自动方案、人工 oracle、零/一/二层及逐阶段查询对比。本轮先闭合
用户当前要求的查询证据；建图 timing/resource 不作为本轮完成门槛。

## 阻塞与问题

- 无外部阻塞；Amazon 修复后 screen 使用 CPU，L20 不用于该查询。
- 不修改 `/home/graphdb/FilterVectorCode_refactor`。
- 不提交 `runs/`、两个 nested `thirdparty/` 目录或误生成的长文件名。

## 验证

- `cmake --build build_ung_rel --clean-first -j16` — `通过`：`fcb74ad` 前最终构建。
- `env LC_ALL=C LANG=C ctest --output-on-failure` — `通过`：15/15。
- 新实验框架测试 — `通过`：Python unittest 103/103，`git diff --check` 通过。
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
