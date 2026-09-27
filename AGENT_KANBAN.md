# Agent 看板

最后更新：`2026-09-27 19:07 Asia/Shanghai`
分支：`codex/multilevel-special-block-20260905`
检查点：`5117e76`（push：`不执行，origin 指向用户工作树`）

## 目标

在统一任意层框架下公平比较 hierarchy、逐层 LNG/Trie topology、三种 entry
strategy 与 routing；报告 Recall-QPS、阶段耗时、点/边/距离计算，验证 query-free
自动 DRH 与人工候选，并完成 GPU 构建实验、权威中文报告和 SIGMOD/VLDB 风格论文。

## 当前状态

- 总体：`进行中`
- 当前优先级：先完成 deadline-bounded held-out 证据，再生成论文方法与结果。
- 旧 396-case Amazon full campaign 已停止并保留 88 个完成 case；失败点
  `l0_trie_entry_optimized_lng/sel_95` 因单个 warm repeat 约 64.6 分钟被终止，
  不再恢复该 campaign。
- 新协议固定 `Lsearch=[40,100,500,2500,10000,40000]`、1 cold + 2 warm、
  每 case 3300 秒硬超时。每个 repeat 执行同一冻结的完整查询集，不重新随机采样；
  cold 仅预热，warm 用于估计 throughput 与波动，Recall crossing 要求所有 warm
  repeats 达标。若仍超时，可在明确标注的 screen 中降为 1+1，
  不把它冒充正式重复结果。
- Amazon 已有 9 个代表方法覆盖九档。routed DRH `1024:lng,16384:trie`
  相对 LNG-0 在 60/80/95/99% 为约 `14.40/24.22/26.70/18.78x`，在
  0.5/1/5/10% 为 `0.86/0.99/0.85/0.64x`；30% 未达到 Recall 0.90。
  因而当前证据只支持高选择率优势，不能声称全区间无损。
- deadline held-out 子集包含 baseline、DRH 和 5 个 query-independent frozen
  manual structural alternatives；它不是原 35-manual full oracle。

## 进行中

- 三段串行证据流水线正在运行：query supervisor 已完成 8 条记录，当前为 Reviews
  baseline 的 `query_minlen2_cov1k`；Amazon build runner 仅等待 query completion；
  detailed-profile runner 将等待 query 和 build 都完成且相关进程退出后才启动。

## 完成历史

- 任意 hierarchy、逐层 LNG/Trie、三种 entry strategy 已正交化；搜索只扫描
  candidate 当前层边，无 edge fallthrough、隐式晋级或跨层混扫 — `fcb74ad`。
- Trie 空谓词 root frontier 错误已修复；真实 Amazon 空谓词 Recall@10=1.0。
- DRH-v1 已在 held-out query 前冻结：`T1=nearestPow2(sqrt(N))`、
  `rho=max(2,round(R/C))`、`N/T<C` 停止、按 `N/T>R` 选择 LNG/Trie；
  exact authorization gate 不读取 query distribution、latency 或 Recall。
- 查询/build runner 已记录 binary hash、命令、环境、manifest、超时和断点恢复；
  汇总器使用 conservative crossing 并分离 performance/profile pass。
- 权威生成器已能输出 Recall/QPS、五段 timing、visited points、base/special
  edges、entry/graph/total distance calculations 和动态共同 Recall 预算。
- deadline campaign 工具和配置已提交 — `7c20af4`；CPU profile 初步修正 —
  `a0efa26`；Python tests 当时为 190/190。
- 错误配置下的 `fv_deadline_evidence_20260927` supervisor、查询进程组和 tmux
  已于 18:07 安全停止；Genome automatic build 完成 19.76 秒，Reviews/VariousImg
  因 validator 正确发现 routed GPU work 而失败。
- deadline CPU sidecar 修复已完成：生成器同时关闭 legacy intra、inter 和
  size-routed intra，并新增两项回归测试。新 root 的 Reviews/VariousImg automatic
  build 分别用 120.44/633.16 秒完成；metadata 与日志均证明 GPU intra blocks 和
  inter use 为 0。
- deadline 专用汇总器已实现：原子输出全方法 conservative crossing、DRH/plain、
  DRH/5-manual、单一 global manual、阶段耗时、点/边/距离计算、CPU sidecar 构建
  和完整性 manifest；partial 模式会逐 case 明示缺失，严格模式拒绝不完整网格。
- 已确认 performance pass 默认 `UNG_SPECIAL_LIGHT_STATS=1`，因此 layered 方法的
  special-edge 0 是“计数关闭”而非“未扫描边”。deadline 汇总器现把这类边计数
  输出为 NA 并标记 light；真实 base/special intra/inter 边数必须来自后续独立
  detailed profile。当前 snapshot 的 exact-gate 时间也仍在 residual 中。
- deadline Amazon 构建 prepare/runner 已实现并生成 40 个 case（15 base timing、
  15 DRH hierarchy timing、5 base resource、5 hierarchy resource）；8 项专项测试
  与全量 `205/205` Python tests 通过，查询未结束时的真实启动尝试被隔离门拒绝。
- 独立 detailed-profile prepare/runner 已提交 — `5117e76`；从 performance
  pass 的 conservative crossing 选择 baseline/DRH 点，无 crossing 时明示标记
  best measured point，且 profile wall time 不进入主 QPS。全量 Python tests 为 `213/213`。

## 下一步

继续轮询现有 query campaign，不重启 PID；保持 build/profile waiter 串行。
query 完成后生成 held-out strict 汇总，build 完成后汇总 GPU 构建证据，profile
完成后合并真实边/点/距离计数，再回填中文报告和论文。

## 阻塞与问题

- 无外部阻塞。
- 不修改 `/home/graphdb/FilterVectorCode_refactor`。
- 不提交或删除 `runs/`、两个 nested `thirdparty/` 目录和误生成的长文件名。
- GPU 构建性能只引用独立 Amazon build study；held-out CPU sidecar 时间不作为
  GPU 加速证据。

## 验证

- `213/213` Python tests — `通过`；profile prepare/runner 与现有全套回归均通过。
- 当前 deadline Reviews/VariousImg build — `失败且已解释`：环境虽有
  `GPU_INTRA=0`，但日志记录 `gpu_intra_blocks=19/20`，来自独立 size route。
- 新 CPU route 修复 — `通过`：18/18 case 静态为 `cpu/0/0/0`；两项单元测试及
  Reviews/VariousImg 真实 automatic build 通过，metadata backend 证据为 0。
- 构建/query 隔离门 — `通过`：query supervisor 未完成时，deadline build runner
  在启动任何 case 前以非零状态拒绝执行。

## 仅在需要时阅读的细节

- `experiments/multilevel_special/AUTHORITATIVE_EXPERIMENT_STATE.md` — 实验协议、
  issue ledger、假设和原始证据边界。
- `runs/deadline_evidence_20260927/` — 已停止的首轮 bounded campaign 原始日志；
  仅用于诊断，不与修复后 run root 混合。
- `runs/deadline_evidence_20260927_cpufix/` — 修复后的权威 bounded campaign root。
- `runs/deadline_build_20260927/` — 查询结束后使用的 Amazon bounded 构建 root；
  当前只有已生成配置，尚无 timing 运行。
- `runs/deadline_profile_20260927/` — query 与 build 都完成后用的独立 detailed
  profile root；只能用于机制 breakdown，不能替代 performance-pass QPS。

## 恢复说明

1. 先读本看板及权威实验状态。
2. 运行 `git status --short`，保留三项已知 untracked 内容。
3. 核对 HEAD、tmux/PID、manifest 和 binary SHA-256。
4. 若长任务仍在运行，只轮询现有 handle；不要重复启动。
5. 从“下一步”继续，结论只引用协议一致且 provenance 完整的数据。

## 清理提示

- `runs/` 保留原始证据，不纳入 Git。
- 旧 full campaign 只保留作历史证据，不再恢复。
