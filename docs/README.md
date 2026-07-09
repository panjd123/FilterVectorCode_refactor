# FilterVectorCode_refactor 文档索引

本文档说明当前仓库文档和优化资产的组织方式。根目录 `README.md` 只保留项目入口、当前主线和原始 UNG/ACORN 背景；实验报告、论文证据、运行手册、历史来源说明统一放在 `docs/` 下。

## 0. 当前主线与第一阅读入口

当前推荐先读：

```text
docs/reports/REVIEW_READY_HANDOFF_CN.md
```

这份文档是当前 review-ready 的唯一推荐入口，包含阅读顺序、最终性能表、复现证据和剩余风险。

当前论文/实验主线的简短定位如下：

```text
cross-edge: 当前 full-quality best 使用 GPU grouped fused topK + SearchQueue lazy reserve；universal flat double-buffer 是重要工程路线但不是当前 best full-quality 口径；
group graph: workload-aware router，小组 CPU/bounded fallback，中组 packed exact-anchor，大组 FastGrnndCuda/reverse-tail；
output boundary: NeighborList64/reserve/direct-H2D 已降低 CPU-compatible 图物化开销，CSR search backend 已接入但仍需端到端复测；
query entry group: gpu_cover_frontier correct-cover 是当前查询入口组选择优化，保证覆盖实际候选 group，允许入口组冗余。
```

注意：`gpu_cover_frontier` 已接入 production search route，但当前实现是 per-query 串行 CUDA provider；`19.85x` 仍只应写成 entry provider microbenchmark 结果，不能写成端到端 query 加速。

## 1. 给人看的最短路径

如果目标是快速理解“我们现在做到了什么、哪些能写进论文、哪些不能夸大”，按这个顺序读：

| 顺序 | 文档 | 作用 |
|---:|---|---|
| 1 | `reports/WORK_PRESENTATION_DEEPRESEARCH_CN.md` | 工作展示主文档：方法、baseline、数据集特点、主性能表和展示建议 |
| 2 | `reports/REVIEW_READY_HANDOFF_CN.md` | 当前 review-ready 唯一入口：阅读顺序、最终结果、复现证据、风险 |
| 3 | `reports/THREE_MAINLINES_METHOD_BASELINE_DATA_SPEEDUP_CN.md` | 五条主线最终性能大表、full-quality 输出边界、复现 checklist |
| 4 | `reports/SPECIAL_BLOCK_BUILD_QUERY_AB_CN.md` | special block 构建/查询、GPU intra/inter、save/load 与近似路线证据 |
| 3 | `papers/EVIDENCE_MATRIX_CN.md` | claim-level 证据矩阵，说明哪些结论可写、哪些还缺实验 |
| 4 | `runbooks/UNG_METHOD_REGISTRY_CN.md` | 方法注册表：每条实现路径的分类、选择方式、入口函数、语义边界 |
| 5 | `reports/CURRENT_METHODS_EFFECTS_AND_GAPS_CN.md` | 更详细的当前总览：方法、效果、负结果、边界和下一步缺口 |
| 6 | `reports/TECHNICAL_REPORT_OPTIMIZATION_SPEEDUP_CN.md` | 长技术报告和证据链；适合查细节，不再作为第一次阅读入口 |
| 7 | `papers/SUBMISSION_READINESS_STATUS_CN.md` | 投稿就绪状态快照，先看当前 gate、阻塞项和主 claim 状态 |
| 8 | `papers/REBUTTAL_CHECKLIST_CN.md` | reviewer 可能攻击点、当前回答和还需补的证据 |

## 2. 给 AI / 后续 Agent 看的入口

这些文档更适合让后续编码代理或新接手的人快速恢复上下文。它们不是论文正文素材，而是任务状态、代码地图和实现约束。

| 文档 | 作用 |
|---|---|
| `reports/SYSTEM_SCOPE_PERFORMANCE_AND_LIMITS_CN.md` | 背景/范围说明：系统组成、性能优势、适用环境、学术 claim 边界；当前 review 优先读 `REVIEW_READY_HANDOFF_CN.md` |
| `REFACTOR_DEEP_DIVE_CN.md` | 代码职责、端到端流程、关键实现和接手顺序 |
| `reports/CODE_STRUCTURE_REVIEW_20260625_CN.md` | 当前代码结构审计：哪些接口清楚、哪些实现过度嵌套、下一轮怎么拆 |
| `papers/PAPER_SUBMISSION_TODO_CN.md` | 投稿前补实验清单；包含很多“下一步该跑什么”的工作队列 |
| `papers/EVIDENCE_MATRIX_CN.md` | 既给人看，也给 AI 看；后续写作和实验必须以这个矩阵约束 claim |
| `reports/CURRENT_OPTIMIZATION_STATUS_CN.md` | 历史当前优化状态、实测耗时、适用条件和未解决瓶颈 |
| `reports/OPTIMIZATION_STEP_BY_STEP.md` | 优化过程流水记录，适合追溯为什么尝试过某条路线 |
| `runbooks/UNG_METHOD_REGISTRY_CN.md` | 当前所有 CPU/GPU/main/baseline/diagnostic/boundary 方法注册表 |
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
| `reports/SYSTEM_SCOPE_PERFORMANCE_AND_LIMITS_CN.md` | 背景/范围说明，精简说明系统组成、性能证据、适用环境和 claim 边界 |
| `reports/THREE_MAINLINES_METHOD_BASELINE_DATA_SPEEDUP_CN.md` | 按 cross-edge、ELS/入口组、PG/group graph、additional_edges/output boundary 整理方法、baseline、数据集/构造形态、加速比和不能夸大的边界 |
| `reports/BEST_FULL_QUALITY_VERSION_CN.md` | 当前 best_full_quality 可复现配置、最新 CPU baseline 对比、端到端构建加速比和剩余瓶颈 |
| `reports/SPECIAL_BLOCK_GRAPH_QUERY_DESIGN_CN.md` | 特异块构造、特异边 sidecar、查询 free 状态、x1-only 数据口径、threshold sweep 和 smoke 验证结果 |
| `reports/SPECIAL_BLOCK_BUILD_QUERY_AB_CN.md` | 特异块 x1-restored full-quality build/query A/B、构建负结果、查询收益边界和下一步 |
| `reports/GPU_GROUP_GRAPH_RELATED_WORK_COLLISION_CN.md` | GPU group graph / PG 构建相关工作与撞车风险分析，比较 CAGRA、Tagore、Vamana/NSG、Filtered-DiskANN 和当前 group-aware reverse exact-anchor |
| `reports/CURRENT_METHODS_EFFECTS_AND_GAPS_CN.md` | 更详细的专家总览，按方法/效果/负结果/缺口展开 |
| `reports/CODE_STRUCTURE_REVIEW_20260625_CN.md` | 代码结构审计，回答当前方法是否接口清晰、文件结构是否易维护 |
| `reports/TECHNICAL_REPORT_OPTIMIZATION_SPEEDUP_CN.md` | 长技术报告，汇总性能、recall、结构诊断、查询优化和边界 |
| `reports/UNIVERSAL_GPU_REPLACEMENT_ANALYSIS_CN.md` | universal GPU replacement 的设计分析和限制 |
| `reports/END_TO_END_RECALL_STATUS_CN.md` | 端到端 search recall 当前证据与缺口 |
| `reports/FINAL_GROUP_GRAPH_METHOD_CN.md` | FastGrnndCuda / group graph 方法设计与当前结论 |
| `reports/FASTGRNNDCUDA_AMAZON_SCALE_RESULTS_CN.md` | Amazon scale 上的 FastGrnndCuda 结果 |
| `reports/CURRENT_OPTIMIZATION_STATUS_CN.md` | 历史当前优化状态，保留作证据来源；新读者优先看 `CURRENT_METHODS_EFFECTS_AND_GAPS_CN.md` |
| `reports/CROSS_GROUP_CELEBA_RESULTS_CN.md` | 历史 CelebA、SIFT30、Amazon、cross-edge 和 fused topK 实验报告 |
| `reports/OPTIMIZATION_STEP_BY_STEP.md` | 优化尝试过程记录，适合追溯，不适合作为主结论入口 |

## 5. Runbook 与复现实验

| 文档 | 作用 |
|---|---|
| `runbooks/RUNBOOK_OPTIMIZED_FEATURES_CN.md` | 编译、运行、环境变量、benchmark 复现命令 |
| `runbooks/UNG_METHOD_REGISTRY_CN.md` | 方法注册表：每条实现路径的分类、选择方式、入口函数、语义边界 |
| `runbooks/UNG_BUILD_MODE_SWITCHES_CN.md` | CPU/GPU/group graph/cross-edge 各实现开关；注意 cross-edge 当前 best 口径以 `THREE_MAINLINES...` 为准 |
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
| `../scripts/benchmarks/run_query_entry_group_sweep.sh` | 查询入口组超参 sweep，汇总 `frontier_delta` / `frontier_cover_cap` 的性能和输出组数质量 |

端到端建图质量检查入口：

| 路径 | 内容 |
|---|---|
| `../scripts/benchmarks/run_end_to_end_recall_ab.sh` | 构建索引后跑 filtered-search recall A/B；这是 PG/group graph、cross-edge、additional_edges、output boundary 变更的质量 gate |
| `../scripts/benchmarks/run_special_block_build_query_ab.sh` | special block threshold/skip A/B 编排，内部调用 full-quality recall 流水线 |
| `../tools/benchmarks/summarize_special_block_ab.py` | 汇总 special block build/query/bucket 指标 |
| `../tools/benchmarks/extract_stride_vecs_bin.py` | 从 repeat 数据中按 stride 构造 x1-restored 向量文件 |

注意：局部 topK overlap 或单 group kNN recall 只能作为诊断，不能替代 `run_end_to_end_recall_ab.sh` 的查询结果质量检查。

## 7. 设计规格与历史归档

| 路径 | 内容 |
|---|---|
| `specs/` | UNG 组内 PG GPU 并行化相关 PDF 规格文档 |
| `archive/BASELINE_PROVENANCE.md` | baseline 来源 |
| `archive/COMPLETE_LAYOUT.md` | 早期完整布局说明 |
| `archive/HISTORY_MAP.md` | 分支、tag、历史迁移记录 |
| `archive/OPTIMIZATION_PROVENANCE.md` | 优化来源和迁移审计 |
| `archive/patches/` | 历史 patch 导出 |

## 7.1 展示材料

| 路径 | 内容 |
|---|---|
| `presentations/ADVISOR_BRIEFING_20260625_CN.md` | 当前最适合给导师汇报的材料，按工作线解释进展、效果和缺口 |
| `presentations/0605.md` | 早期展示文档，包含更细的建图、cross-edge 和 query-entry scan 表格 |

## 8. 清理规则

本地 CMake build 目录属于可再生临时产物，统一由 `.gitignore` 忽略，例如 `build_*`、`CMakeFiles/`、`CMakeCache.txt`。实验结果不放在仓库内，继续写到 `/home/graphdb/FilterVectorResultsRefactor/...` 或明确的 `/home/graphdb/fv_runs/...` artifact 目录。
