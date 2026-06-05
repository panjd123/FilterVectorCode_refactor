# FilterVectorCode_refactor 文档索引

本文档说明当前仓库文档和优化资产的组织方式。根目录只保留项目入口和运行脚本，实验报告、运行手册、历史来源说明统一放在 `docs/` 下。

## 当前主线

当前论文/实验主线的简短定位：

```text
cross-edge: UNG_UNIVERSAL_GPU=1 universal flat double-buffer 是当前主工程路线；
group graph: workload-aware router，小组 CPU/bounded fallback，中组 packed exact-anchor，大组 FastGrnndCuda/reverse-tail；
output boundary: NeighborList64/reserve/direct-H2D 已降低 CPU-compatible 图物化开销，但 flat/CSR 仍是未完成的下一步。
```

| 文档 | 作用 |
| --- | --- |
| `papers/UNG_GPU_OPTIMIZATION_PAPER_DRAFT_CN.md` | 中文论文草稿，包含方法、实验和限制 |
| `papers/UNG_GPU_OPTIMIZATION_PAPER_DRAFT_EN.md` | 英文论文草稿 |
| `papers/SUBMISSION_READINESS_STATUS_CN.md` | 投稿就绪状态快照，列出当前 artifact gate、阻塞项和下一步命令 |
| `papers/REBUTTAL_CHECKLIST_CN.md` | 投稿前 reviewer rebuttal 检查表，逐条列攻击点、当前回答、证据和边界 |
| `papers/REVIEWER_AUTHOR_ITERATION_CN.md` | 审稿人攻击点、作者回应和实验迭代记录 |
| `papers/EVIDENCE_MATRIX_CN.md` | claim-level 投稿证据矩阵，说明哪些结论可写、哪些还缺实验 |
| `papers/PAPER_SUBMISSION_TODO_CN.md` | 投稿前补实验清单 |
| `reports/TECHNICAL_REPORT_OPTIMIZATION_SPEEDUP_CN.md` | 当前最终技术报告，汇总性能、recall、结构诊断和边界 |
| `reports/END_TO_END_RECALL_STATUS_CN.md` | 端到端 search recall 当前证据与缺口 |
| `reports/FINAL_GROUP_GRAPH_METHOD_CN.md` | FastGrnndCuda / group graph 方法设计与当前结论 |
| `reports/FASTGRNNDCUDA_AMAZON_SCALE_RESULTS_CN.md` | Amazon scale 上的 FastGrnndCuda 结果 |
| `reports/CURRENT_OPTIMIZATION_STATUS_CN.md` | 历史当前优化状态、实测耗时、适用条件和未解决瓶颈 |
| `reports/CROSS_GROUP_CELEBA_RESULTS_CN.md` | 历史 CelebA、SIFT30、Amazon、cross-edge 和 fused topK 实验报告 |
| `runbooks/RUNBOOK_OPTIMIZED_FEATURES_CN.md` | 编译、运行、环境变量、benchmark 复现命令 |
| `runbooks/UNG_BUILD_MODE_SWITCHES_CN.md` | CPU/GPU/group graph/cross-edge 各实现开关；包含 `UNG_UNIVERSAL_GPU=1` 当前推荐 cross-edge 口径 |
| `runbooks/X400_REVERSE_TAIL_AB_CN.md` | x400 light reverse-tail A/B 复现实验流程 |
| `runbooks/TESTING_GUIDE.md` | CPU/GPU A/B、正确性测试和性能回归测试 |
| `REFACTOR_DEEP_DIVE_CN.md` | 代码职责、端到端流程、关键实现和接手顺序 |

## 设计与规格

| 文档 | 作用 |
| --- | --- |
| `specs/` | UNG 组内 PG GPU 并行化相关 PDF 规格文档 |

## 历史归档

| 路径 | 内容 |
| --- | --- |
| `archive/BASELINE_PROVENANCE.md` | baseline 来源 |
| `archive/COMPLETE_LAYOUT.md` | 早期完整布局说明 |
| `archive/HISTORY_MAP.md` | 分支、tag、历史迁移记录 |
| `archive/OPTIMIZATION_PROVENANCE.md` | 优化来源和迁移审计 |
| `archive/patches/` | 历史 patch 导出 |

## 脚本与工具

| 路径 | 内容 |
| --- | --- |
| `../scripts/run_ung_tests.sh` | UNG 快速测试入口 |
| `../scripts/benchmarks/` | cross-edge / SIFT30 / Tagore / x400 reverse-tail 等 benchmark 入口脚本 |
| `../tools/benchmarks/` | CUDA benchmark 源码、coverage query 生成、图结构诊断、artifact audit 和论文 claim 语言检查工具 |
| `../tools/datasets/generate_powerzipf_labels.py` | power-Zipf 标签生成工具 |

投稿前默认 gate：

```bash
python3 tools/benchmarks/run_submission_gate.py
```

当前已登记 artifact 的最终 gate：

```bash
python3 tools/benchmarks/run_submission_gate.py --final
```

若后续重跑 x400 或 router A/B，可传入新的输出目录：

```bash
python3 tools/benchmarks/run_submission_gate.py \
  --x400-root <x400_ab_root> \
  --router-root <router_ab_root> \
  --final
```

## 清理规则

本地 CMake build 目录属于可再生临时产物，统一由 `.gitignore` 忽略，例如 `build_*`、`CMakeFiles/`、`CMakeCache.txt`。实验结果不放在仓库内，继续写到 `/home/graphdb/FilterVectorResultsRefactor/...`。
