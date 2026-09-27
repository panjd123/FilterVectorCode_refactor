# Agent 看板

最后更新：`2026-09-27 18:36 Asia/Shanghai`
分支：`codex/multilevel-special-block-20260905`
检查点：`ed22c63`（push：`不执行，origin 指向用户工作树`）

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
  每 case 3300 秒硬超时。重复用于预热后估计 warm throughput；Recall crossing
  要求所有 warm repeats 达标。若仍超时，可在明确标注的 screen 中降为 1+1，
  不把它冒充正式重复结果。
- Amazon 已有 9 个代表方法覆盖九档。routed DRH `1024:lng,16384:trie`
  相对 LNG-0 在 60/80/95/99% 为约 `14.40/24.22/26.70/18.78x`，在
  0.5/1/5/10% 为 `0.86/0.99/0.85/0.64x`；30% 未达到 Recall 0.90。
  因而当前证据只支持高选择率优势，不能声称全区间无损。
- deadline held-out 子集包含 baseline、DRH 和 5 个 query-independent frozen
  manual structural alternatives；它不是原 35-manual full oracle。

## 进行中

- 从独立 root `runs/deadline_evidence_20260927_cpufix/` 恢复 bounded campaign；
  automatic build 会复用已验证 Reviews/VariousImg artifact，并先补建 Genome，随后
  优先运行三个数据集的 baseline/DRH query，再运行 5 个 manual alternatives。

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

## 下一步

跑完整 Python 套件并提交 CPU route 修复；随后在 tmux 中从新 root 启动 bounded
campaign，持续检查超时、Recall crossing 和 provenance。查询完成后汇总 held-out、
breakdown 和 GPU build 数据并生成报告/论文。

## 阻塞与问题

- 无外部阻塞。
- 不修改 `/home/graphdb/FilterVectorCode_refactor`。
- 不提交或删除 `runs/`、两个 nested `thirdparty/` 目录和误生成的长文件名。
- GPU 构建性能只引用独立 Amazon build study；held-out CPU sidecar 时间不作为
  GPU 加速证据。

## 验证

- `197/197` Python tests — `待全量复跑`：新增 light/detail counter 语义回归后
  专项 5/5 已通过。
- 当前 deadline Reviews/VariousImg build — `失败且已解释`：环境虽有
  `GPU_INTRA=0`，但日志记录 `gpu_intra_blocks=19/20`，来自独立 size route。
- 新 CPU route 修复 — `通过`：18/18 case 静态为 `cpu/0/0/0`；两项单元测试及
  Reviews/VariousImg 真实 automatic build 通过，metadata backend 证据为 0。

## 仅在需要时阅读的细节

- `experiments/multilevel_special/AUTHORITATIVE_EXPERIMENT_STATE.md` — 实验协议、
  issue ledger、假设和原始证据边界。
- `runs/deadline_evidence_20260927/` — 已停止的首轮 bounded campaign 原始日志；
  仅用于诊断，不与修复后 run root 混合。
- `runs/deadline_evidence_20260927_cpufix/` — 修复后的权威 bounded campaign root。

## 恢复说明

1. 先读本看板及权威实验状态。
2. 运行 `git status --short`，保留三项已知 untracked 内容。
3. 核对 HEAD、tmux/PID、manifest 和 binary SHA-256。
4. 若长任务仍在运行，只轮询现有 handle；不要重复启动。
5. 从“下一步”继续，结论只引用协议一致且 provenance 完整的数据。

## 清理提示

- `runs/` 保留原始证据，不纳入 Git。
- 旧 full campaign 只保留作历史证据，不再恢复。
