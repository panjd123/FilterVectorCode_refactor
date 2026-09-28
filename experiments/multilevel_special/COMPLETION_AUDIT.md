# 多层 Filtered-ANN Deadline Evidence 完成审计

本审计区分两种证据契约：本次已执行的 deadline-bounded campaign，以及未执行的
历史 396-case formal campaign。前者可以支持当前论文中的 screen-level 结论，
不能被表述为后者已经完成。

## 固定口径

- 开发数据集：Amazon，602,453 个 768 维向量、482,387 个 groups。
- held-out 数据集：Genome、Reviews、VariousImg。
- Amazon 选择率：0.5%、1%、5%、10%、30%、60%、80%、95%、99%。
- 查询：每档固定 1,000 queries，K=10，100 query threads，exact containment GT。
- 主 QPS：同一冻结 query set，1 次 cold + 2 次 warm，取 warm batch time 中位数。
- crossing：所有 warm repeats 的 Recall@10 均不低于 0.90 的最小实测 Lsearch；
  不插值、不外推。
- candidate 只扫描 activation level 自身边；无 edge fallthrough、隐式晋级或
  跨层混扫。
- detailed profile 与主 QPS、GPU timing、resource profile 分离；profile 缺失
  计数不得解释为 0。
- 自动 DRH 只使用 N、R、C 决定层数、阈值和逐层 topology，不读取 query
  distribution、Recall 或 latency。

## Deadline Campaign 状态

| 证据项 | 状态 | 边界 |
|---|---:|---|
| Amazon 原 LNG-base 两层 LL/TL/TT topology | **27/27 complete** | 3 topology x 9 workloads；每 case 最多 3300 s |
| Amazon 完整 L0 x overlay topology factorial | **126/126 complete** | 2 个 0L、4 个 1L、8 个 2L topology x 9 workloads；两半使用同一查询与构建二进制 |
| Amazon representative detailed profile | **24/25 complete** | 0L-Trie / 95% 在 3300.0 s 超时，论文与报告记为 NC；未使用部分计数 |
| GPU/CPU hierarchy build campaign | **41/41 complete** | timing 与 resource 分离；5 个 hierarchy backend，各 2 个 measured timing repeats |
| Held-out DRH-v1 vs manual | **complete** | manual 仅为 5 个预注册 alternatives，不称为 35-case oracle |
| DRH-v2 routing ablation | **complete** | 与冻结 DRH-v1 manual comparison 分开报告 |
| 报告与论文生成 | **complete_with_declared_timeout** | finalization manifest 显式记录上述唯一 profile timeout |
| 回归测试 | **256/256 passed** | `python3 -m unittest discover -s experiments/multilevel_special -p "test_*.py"` |
| LaTeX | **compiled** | Tectonic 0.15.0 musl；无 undefined reference/citation 或 fatal error |

## 构建结果门禁

- CPU hierarchy median：1000.64 s。
- hybrid GPU intra：90.49 s，hierarchy stage 相对 CPU 为 11.06x。
- full GPU：105.90 s，hierarchy stage 相对 CPU 为 9.45x。
- hybrid GPU intra+inter：132.94 s，hierarchy stage 相对 CPU 为 7.53x。
- full GPU WMMA：1011.13 s，0.99x，是保留的负结果。
- 最佳 composed base+hierarchy 路径为 hybrid GPU intra：141.27 s，
  相对 original CPU base 的 1.35x，bootstrap 95% CI [1.33, 1.36]。
- composed 数值是独立 stage median 的和，不冒充单次联合 wall-clock。

## 可核验产物

- `runs/remaining_evidence_20260928/supervisor_manifest.json`：topology/profile
  case 状态与 3300 s timeout provenance。
- `runs/deadline_build_20260927/deadline_build_supervisor_manifest.json`：41/41
  build case 状态。
- `runs/deadline_build_20260927/build_study/summary/build_summary.csv`：重复
  timing 与独立 resource profile。
- `runs/deadline_build_20260927/build_study/summary/build_end_to_end.csv`：composed
  结果与 bootstrap CI。
- `runs/complete_evidence_20260928/finalization_manifest.json`：生成器命令、
  Tectonic hash、输出 hash 和声明式缺口。
- `docs/reports/MULTILEVEL_SPECIAL_BLOCK_AUTHORITATIVE_RESULTS_CN.md`：中文权威报告。
- `docs/papers/multilevel_ung/main.pdf`：当前论文 PDF。
- GitHub `main`：包含本审计版本；精确提交以远端 `refs/heads/main` 为权威，
  避免在提交内容中写入会立即失效的自引用 commit ID。
- ShareLaTeX：`xm.jarden@gmail.com` 所有的项目
  `ML-UNG Paper - 2026-09-28` 已同步全部 18 个论文源、数据和 PDF 文件；文本与
  二进制内容均逐文件哈希核对，服务端可编译为 20 页 PDF。

## 未完成边界

- 历史 396-case formal campaign 没有恢复或重跑；不得把本次 screen-level QPS
  描述为 formal confidence-interval result。
- 查询 QPS 没有正式置信区间；构建 composed speedup 有 bootstrap 95% CI。
- `0L-Trie / 95%` detailed profile 在固定上限内无法完成，保持 NC；它不影响
  对应主 QPS crossing，但该组合没有阶段耗时或边工作量分解。
- VariousImg 的显著负结果和 WMMA 构建负结果必须保留。

## 完成判定

本次 deadline-bounded 代码、实验、报告和论文交付已完成，带一个明确声明的
profile timeout。完整 396-case formal evidence 仍未完成，不能由本交付替代。
