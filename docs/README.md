# FilterVectorCode_refactor 文档索引

本文档说明当前仓库文档和优化资产的组织方式。根目录 `README.md` 只保留项目入口、当前主线和原始 UNG/ACORN 背景；实验报告、论文证据、运行手册、历史来源说明统一放在 `docs/` 下。

## 0. 当前主线

当前论文/实验主线的简短定位：

```text
cross-edge: UNG_UNIVERSAL_GPU=1 universal flat double-buffer 是当前主工程路线；
group graph: workload-aware router，小组 CPU/bounded fallback，中组 packed exact-anchor，大组 FastGrnndCuda/reverse-tail；
output boundary: NeighborList64/reserve/direct-H2D 已降低 CPU-compatible 图物化开销，但 flat/CSR 仍是未完成的下一步；
query entry group: gpu_cover_frontier correct-cover 是当前查询入口组选择优化，保证覆盖实际候选 group，允许入口组冗余。
```

查询入口组优化的详细记录在 `reports/TECHNICAL_REPORT_OPTIMIZATION_SPEEDUP_CN.md` 第 19 节：

- 语义：输出入口组必须覆盖所有满足 `query_labels subset labels(group)` 的实际 group。
- 方法：`F_raw -> cap -> descendants OR -> F union (C - Covered)`。
- 正确性边界：不要求所有 label 组合都有对应 group；只要求 descendant 表能覆盖真实 label-superset 关系。
- 实测：Amazon 100%x40、`nq=10240`、`delta=1 cap=8192`，完整接口边界 `59.83 ms`，相对 CPU scan 128T `19.85x`。
- 端到端影响：入口组阶段加速不等于图查询整体同等加速；10%x40 现有 query CSV 中 CPU entry 占 `48~51%`，替换 entry stage 后理论上界约 `1.9~2.0x`，但 `gpu_cover_frontier` 仍需接入正式 `search_UNG_index` 后复测。
- 建图 100%x40：已生成 `23,284,680` 点数据；此前 `prepare_group_storages_graphs` 崩溃已定位为 `Storage` 里 `IdxType` 乘法溢出并修复。旧 GPU cross-edge 需要 `71.5GB` 全量向量 device cache，48GB A6000 OOM；新增 `UNG_GPU_X_STREAMING=1` 后按 X-side chunk 流式上传，strict skip-additional build 已完成。当前 v1 streaming cross-edge `697.1s`，其中 `pack_q=450.3s`，说明它解决显存边界但还不是加速主结果。
- 阶段展示摘要：`reports/TECHNICAL_REPORT_OPTIMIZATION_SPEEDUP_CN.md` 第 21 节汇总了这段时间的工作、关键发现、当前边界和未来方向；其中单独记录了 Amazon 100%x40 查询 graph 耗时异常，指出 `DistCalcs/NumVisited/NumEntries` 不能解释 `core_search_time_ms` 暴涨，下一步需要细粒度计数器、硬件计数和 cache-friendly graph search 重构。

## 1. 给人看的最短路径

如果目标是快速理解“我们现在做到了什么、哪些能写进论文、哪些不能夸大”，按这个顺序读：

| 顺序 | 文档 | 作用 |
|---:|---|---|
| 1 | `papers/SUBMISSION_READINESS_STATUS_CN.md` | 投稿就绪状态快照，先看当前 gate、阻塞项和主 claim 状态 |
| 2 | `papers/EVIDENCE_MATRIX_CN.md` | claim-level 证据矩阵，说明哪些结论可写、哪些还缺实验 |
| 3 | `reports/TECHNICAL_REPORT_OPTIMIZATION_SPEEDUP_CN.md` | 当前最完整技术报告；第 21 节可作为阶段展示摘要，汇总工作、发现、现状和未来研究方向 |
| 4 | `papers/REBUTTAL_CHECKLIST_CN.md` | reviewer 可能攻击点、当前回答和还需补的证据 |
| 5 | `papers/UNG_GPU_OPTIMIZATION_PAPER_DRAFT_CN.md` / `papers/UNG_GPU_OPTIMIZATION_PAPER_DRAFT_EN.md` | 中文/英文论文草稿 |

## 2. 给 AI / 后续 Agent 看的入口

这些文档更适合让后续编码代理或新接手的人快速恢复上下文。它们不是论文正文素材，而是任务状态、代码地图和实现约束。

| 文档 | 作用 |
|---|---|
| `REFACTOR_DEEP_DIVE_CN.md` | 代码职责、端到端流程、关键实现和接手顺序 |
| `papers/PAPER_SUBMISSION_TODO_CN.md` | 投稿前补实验清单；包含很多“下一步该跑什么”的工作队列 |
| `papers/EVIDENCE_MATRIX_CN.md` | 既给人看，也给 AI 看；后续写作和实验必须以这个矩阵约束 claim |
| `reports/CURRENT_OPTIMIZATION_STATUS_CN.md` | 历史当前优化状态、实测耗时、适用条件和未解决瓶颈 |
| `reports/OPTIMIZATION_STEP_BY_STEP.md` | 优化过程流水记录，适合追溯为什么尝试过某条路线 |
| `runbooks/UNG_BUILD_MODE_SWITCHES_CN.md` | CPU/GPU/group graph/cross-edge 各实现开关和环境变量 |
| `runbooks/TESTING_GUIDE.md` | CPU/GPU A/B、正确性测试和性能回归测试 |

建议 AI 接手顺序：

1. 先读 `papers/EVIDENCE_MATRIX_CN.md`，确认哪些结果能当主 claim。
2. 再读 `reports/TECHNICAL_REPORT_OPTIMIZATION_SPEEDUP_CN.md` 的最新章节，尤其是第 18、19 节。
3. 如果要改代码，再读 `REFACTOR_DEEP_DIVE_CN.md` 和对应 runbook。
4. 如果要跑实验，优先使用 `scripts/benchmarks/` 和 `tools/benchmarks/` 中已有脚本，不要临时拼不可复现命令。

## 3. 论文与投稿文档

| 文档 | 作用 |
|---|---|
| `papers/UNG_GPU_OPTIMIZATION_PAPER_DRAFT_CN.md` | 中文论文草稿，包含方法、实验和限制 |
| `papers/UNG_GPU_OPTIMIZATION_PAPER_DRAFT_EN.md` | 英文论文草稿 |
| `papers/SUBMISSION_READINESS_STATUS_CN.md` | 投稿就绪状态快照，列出当前 artifact gate、阻塞项和下一步命令 |
| `papers/REBUTTAL_CHECKLIST_CN.md` | 投稿前 reviewer rebuttal 检查表，逐条列攻击点、当前回答、证据和边界 |
| `papers/REVIEWER_AUTHOR_ITERATION_CN.md` | 审稿人攻击点、作者回应和实验迭代记录 |
| `papers/EVIDENCE_MATRIX_CN.md` | claim-level 投稿证据矩阵，说明哪些结论可写、哪些还缺实验 |
| `papers/PAPER_SUBMISSION_TODO_CN.md` | 投稿前补实验清单 |
| `papers/GRAPH_OUTPUT_BOUNDARY_OPTIMIZATION_CN.md` | Graph 输出边界优化专题记录 |

## 4. 技术报告

| 文档 | 作用 |
|---|---|
| `reports/TECHNICAL_REPORT_OPTIMIZATION_SPEEDUP_CN.md` | 当前最终技术报告，汇总性能、recall、结构诊断、查询优化和边界 |
| `reports/UNIVERSAL_GPU_REPLACEMENT_ANALYSIS_CN.md` | universal GPU replacement 的设计分析和限制 |
| `reports/END_TO_END_RECALL_STATUS_CN.md` | 端到端 search recall 当前证据与缺口 |
| `reports/FINAL_GROUP_GRAPH_METHOD_CN.md` | FastGrnndCuda / group graph 方法设计与当前结论 |
| `reports/FASTGRNNDCUDA_AMAZON_SCALE_RESULTS_CN.md` | Amazon scale 上的 FastGrnndCuda 结果 |
| `reports/CURRENT_OPTIMIZATION_STATUS_CN.md` | 历史当前优化状态、实测耗时、适用条件和未解决瓶颈 |
| `reports/CROSS_GROUP_CELEBA_RESULTS_CN.md` | 历史 CelebA、SIFT30、Amazon、cross-edge 和 fused topK 实验报告 |
| `reports/OPTIMIZATION_STEP_BY_STEP.md` | 优化尝试过程记录 |

## 5. Runbook 与复现实验

| 文档 | 作用 |
|---|---|
| `runbooks/RUNBOOK_OPTIMIZED_FEATURES_CN.md` | 编译、运行、环境变量、benchmark 复现命令 |
| `runbooks/UNG_BUILD_MODE_SWITCHES_CN.md` | CPU/GPU/group graph/cross-edge 各实现开关；包含 `UNG_UNIVERSAL_GPU=1` 当前推荐 cross-edge 口径 |
| `runbooks/X400_REVERSE_TAIL_AB_CN.md` | x400 light reverse-tail A/B 复现实验流程 |
| `runbooks/TESTING_GUIDE.md` | CPU/GPU A/B、正确性测试和性能回归测试 |

常用 gate：

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

## 6. 脚本与工具

| 路径 | 内容 |
|---|---|
| `../scripts/run_ung_tests.sh` | UNG 快速测试入口 |
| `../scripts/benchmarks/` | cross-edge / query entry group / SIFT30 / Tagore / x400 reverse-tail 等 benchmark 入口脚本 |
| `../tools/benchmarks/` | CUDA benchmark 源码、coverage query 生成、图结构诊断、artifact audit 和论文 claim 语言检查工具 |
| `../tools/datasets/generate_powerzipf_labels.py` | power-Zipf 标签生成工具 |

查询入口组 benchmark 相关入口：

| 路径 | 内容 |
|---|---|
| `../UNG/codes/tools/query_entry_group_bench.cu` | CPU scan、CPU exact minimal、GPU correct-cover 查询入口组 benchmark |
| `../scripts/benchmarks/run_query_entry_group_bench.sh` | 查询入口组 benchmark 脚本入口 |

## 7. 设计规格与历史归档

| 路径 | 内容 |
|---|---|
| `specs/` | UNG 组内 PG GPU 并行化相关 PDF 规格文档 |
| `archive/BASELINE_PROVENANCE.md` | baseline 来源 |
| `archive/COMPLETE_LAYOUT.md` | 早期完整布局说明 |
| `archive/HISTORY_MAP.md` | 分支、tag、历史迁移记录 |
| `archive/OPTIMIZATION_PROVENANCE.md` | 优化来源和迁移审计 |
| `archive/patches/` | 历史 patch 导出 |

## 8. 清理规则

本地 CMake build 目录属于可再生临时产物，统一由 `.gitignore` 忽略，例如 `build_*`、`CMakeFiles/`、`CMakeCache.txt`。实验结果不放在仓库内，继续写到 `/home/graphdb/FilterVectorResultsRefactor/...` 或明确的 `/home/graphdb/fv_runs/...` artifact 目录。
