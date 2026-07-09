# Review-ready Handoff: FilterVectorCode_refactor 当前审阅入口

本文是给人类 reviewer 的入口文档。目标是先读本文和少量链接文档，就能理解当前项目做了什么、哪些结果可以引用、哪些结果只是诊断或近似候选。

## 1. 建议阅读顺序

| 顺序 | 文档 | 用途 |
|---:|---|---|
| 1 | `docs/reports/WORK_PRESENTATION_DEEPRESEARCH_CN.md` | 直接用于工作展示的主文档：问题、方法、baseline、数据集、性能表、claim 边界和展示顺序。 |
| 2 | `docs/reports/THREE_MAINLINES_METHOD_BASELINE_DATA_SPEEDUP_CN.md` | 当前总入口；从 `## 0. 最终性能大表` 开始读，先看所有主线的最终口径和 claim 边界。 |
| 3 | `docs/reports/SPECIAL_BLOCK_BUILD_QUERY_AB_CN.md` | special block 构建/查询主证据；包含 CPU/GPU intra/inter 2x2、GPU inter kernel tuning、two-tier heavy sidecar、保存阶段优化。 |
| 4 | `docs/reports/SPECIAL_BLOCK_GRAPH_QUERY_DESIGN_CN.md` | special block 定义、sidecar special edges、free-state 查询、x1-only 数据口径和默认参数。 |
| 5 | `docs/reports/REVIEW_READY_REQUIREMENTS_MATRIX_CN.md` | 用户历史要求到证据的覆盖矩阵；适合 reviewer 快速查“有没有回答这个问题”。 |
| 6 | `AGENT_KANBAN.md` | 当前 agent 工作状态和恢复说明；只在继续接手修改时需要读。 |

## 2. 当前代码与实验范围

当前 review 范围聚焦 `UNG` 构建与查询实验中的几条主线：

- cross group edge GPU fused topK 与 CPU baseline 对比；
- ELS / entry group provider GPU correct-cover；
- PG / group graph GPU route；
- full-quality additional/output materialization 边界；
- special block graph/query，包括 GPU intra、GPU inter、two-tier sidecar、save/load 优化。

不建议 reviewer 把当前仓库当作“干净 release branch”审查：工作树有大量历史改动和新拆分文件，当前目标是研究 artifact review-ready，不是发布版本代码冻结。

## 3. 当前最应该引用的结果

| 结果 | 当前引用口径 | 证据位置 |
|---|---|---|
| Cross group edge | Amazon 1%x200 full-quality：GPU best lazy-reserve cross `2842.36 ms`，相对最快 CPU hybrid `37338.0 ms` 为 `13.14x`。 | `THREE_MAINLINES...` 的 `0.1` 和 `1.4`。 |
| ELS provider | 中高质量 `~1300-1500` entry groups 区间，GPU fused compact 相对同质量 CPU 约 `2.7-3.0x`。 | `THREE_MAINLINES...` 的 `0.1` 和 `2.x`。 |
| PG/group graph | Amazon 1%x200 full-quality group `4.42x`；Amazon 10%x40 full-quality group `3.54x`、Index `1.53x`。 | `THREE_MAINLINES...` 的 `0.1` 和 `3.x`。 |
| additional/output boundary | CPU exact materialized additional 相对 CPU Vamana additional `7.16x`；SearchQueue lazy reserve `1909.9 -> 15.8 ms`。 | `THREE_MAINLINES...` 的 `0.1` 和 `4.x`。 |
| Special block GPU intra/inter | 当前 default：CPU intra exact-topK `th=2048`，GPU intra split `th=128`，GPU inter `2` warps/query。GPU/GPU overlay `~7.87 s`。 | `THREE_MAINLINES...` 的 `0.2/0.3`; `SPECIAL_BLOCK_BUILD...`。 |
| Special save/load | binary-only + skip reordered + CPU-provider-only skip LNG text: save `40.18 -> 19.10 s`。 | `THREE_MAINLINES...` 的 `0.1`; `SPECIAL_BLOCK_BUILD...` 保存阶段段落。 |

## 4. 最重要的 claim 边界

1. `full-quality` 只对保留完整 UNG 语义并通过 filtered-search recall 检查的结果成立。
2. 局部 topK overlap 不是图质量黄金标准；最终质量只能看 filtered-search recall / latency。
3. special block 的 `cap10/iter4` 和 two-tier heavy sidecar 仍属于近似/候选路径，不能写成默认 full-quality 端到端加速结论。
4. `UNG_SKIP_LNG_TEXT_SETS=1` 只适合 CPU entry-provider / UNG special search；GPU cover-frontier provider 仍需要 `_lng_descendants`。
5. TF32 WMMA/shared tile special inter 是负结果；当前默认是 cuda-core source-exact kernel，`2` warps/query。

## 5. 复现与验证入口

| 类型 | 命令或路径 | 说明 |
|---|---|---|
| 编译验证 | `cmake --build /home/graphdb/FilterVectorCode_refactor/build_mode_switch -j 16 --target build_UNG_index search_UNG_index` | 最近 special inter warps=2 后通过。 |
| Claim 语言检查 | `python3 tools/benchmarks/check_paper_claim_language.py docs/reports/THREE_MAINLINES_METHOD_BASELINE_DATA_SPEEDUP_CN.md docs/reports/SPECIAL_BLOCK_BUILD_QUERY_AB_CN.md docs/reports/SPECIAL_BLOCK_GRAPH_QUERY_DESIGN_CN.md` | 防止 overclaim。 |
| Special GPU inter warps sweep | `/home/graphdb/fv_runs/special_blocks_20260705/special_gpu_inter_warps_sweep_20260707_114325` | 证明 `2` warps/query 最优。 |
| Special GPU inter default smoke | `/home/graphdb/fv_runs/special_blocks_20260705/special_gpu_inter_warps2_default_smoke_20260707_114920` | 包含 build breakdown 和 coverage L50/L200 recall。 |
| Special intra 2x2 | `/home/graphdb/fv_runs/special_blocks_20260705/intra_inter_2x2_current_20260707_111233` and `/home/graphdb/fv_runs/special_blocks_20260705/gpu_cpu_cell_current_20260707_112007` | 2x2 当前表来源；第一次 `gpu_cpu` 有 env 泄漏，使用单独重跑目录。 |
| CPU/GPU intra split sweeps | `/home/graphdb/fv_runs/special_blocks_20260705/special_intra_exact_topk_sweep_20260706_131012`, `/home/graphdb/fv_runs/special_blocks_20260705/special_gpu_intra_split_sweep_20260706_222042` | CPU default `2048`、GPU default `128` 的依据。 |

## 6. 当前已知 review 风险

| 风险 | 影响 | 处理状态 |
|---|---|---|
| 工作树很脏，包含大量历史改动和 untracked 文件 | reviewer 很难从 `git status` 判断本轮变更边界 | 本 handoff 明确范围；后续应单独清理/提交。 |
| untracked `._*` AppleDouble 文件 | 干扰 review 和 grep | 已移动到 `/tmp/fv_appledouble_20260708`，`.gitignore` 已加入忽略规则。 |
| 部分实验是 smoke 或单次 run | 不能作为稳定 latency claim | 文档中标注 smoke/diagnostic/近似候选。 |
| SSH 跳板偶发 reset | 影响复现体验 | 短间隔重试通常恢复；看板记录该问题。 |
| special block 默认 full-quality 端到端加速未完全闭环 | 不能作为论文主 claim | 当前只写结构收益、stage 优化和候选路径。 |

## 7. 本轮 review-ready 状态

- 已完成：主性能总表、special block 细报告、设计文档、看板、本 handoff、需求矩阵、主 README/docs README 入口修正、method registry special block 登记。
- 已验证：编译、claim-language 检查、AppleDouble 噪声清理。
- 审查状态：交付物第一轮为 `nearly review-ready`，主要 blocker 已修；结构 follow-up 为 `nearly review-ready`，剩余风险是工作树仍脏、不是 release-style review。
