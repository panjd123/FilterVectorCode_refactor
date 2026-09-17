# Review-ready Requirements Matrix

本表把本轮对话中的主要用户要求映射到当前代码、文档和实验数据。状态含义：`done` 已有证据闭环；`partial` 有实现/实验但仍有质量或复现边界；`deferred` 明确不作为当前主线；`blocked` 当前环境或数据阻塞；`unknown` 证据不足。

| 用户要求 / 审阅问题 | 状态 | 证据 | 剩余缺口 |
|---|---|---|---|
| 精简并整理大文档，明确系统包含什么、优势和适用环境 | done | `docs/reports/THREE_MAINLINES_METHOD_BASELINE_DATA_SPEEDUP_CN.md` 的 `## 0. 最终性能大表`；`SYSTEM_SCOPE_PERFORMANCE_AND_LIMITS_CN.md` | 后续若论文结构变化需同步更新。 |
| 拆分 cross group edge、ELS、PG/group graph、additional/output、special block 五条主线，按方法/baseline/数据集/加速比整理 | done | `THREE_MAINLINES...` 全文，尤其 `0.1` 总表和各章节。 | 表中旧实验仍多，reviewer 需先读 `0.x`。 |
| full-quality、additional_edges、CPU-compatible materialization 的含义和成本解释 | done | `THREE_MAINLINES...` 第 4 章和 `0.1` 表；`check_paper_claim_language.py` 通过。 | GPU-native flat/CSR search backend 未实现。 |
| topK overlap 不能作为质量黄金标准，必须用端到端 filtered-search recall | done | `THREE_MAINLINES...` 开头质量标准；`END_TO_END_RECALL_STATUS_CN.md`；各实验表引用 recall。 | 部分新 special 路径仍只有 smoke，需要更多 workload。 |
| ELS CPU/GPU 公平比较、质量用 entry group 数量 | done | `THREE_MAINLINES...` 第 2 章；GPU fused compact `2.7-3.0x` 同质量区间。 | production batch/end-to-end A/B 仍缺。 |
| special block 只用于 x1/original 数据，区分平凡/非平凡 special | done | `SPECIAL_BLOCK_GRAPH_QUERY_DESIGN_CN.md`；`THREE_MAINLINES...` special 覆盖率表。 | 原始 x1 provenance 仍需 reviewer 注意 restored-from-x40 口径。 |
| special block 构建复用 group/cross 语义，避免旧 per-source Vamana/CPU 慢路径误导 | partial | `SPECIAL_BLOCK_BUILD_QUERY_AB_CN.md` 记录 special-as-group、threadfix、GPU intra/inter。 | 当前 cap10/iter4 仍是近似候选，不是默认 full-quality 主 claim。 |
| CPU/GPU intra 公平口径：CPU 不应全 Vamana，应该有合理分流 | done | CPU exact-topK `th=2048`，GPU split `th=128`；`SPECIAL_BLOCK_BUILD...` `CPU intra 分流修正`。 | CPU exact-topK 只做 coverage smoke，更多 workloads 可补。 |
| GPU inter 加速不明显，需要 kernel 效率优化 | done | `UNG_SPECIAL_GPU_INTER_WARPS=2`；warps sweep `special_gpu_inter_warps_sweep_20260707_114325`；default smoke `special_gpu_inter_warps2_default_smoke_20260707_114920`。 | 进一步 shared-memory 分桶 kernel 可作为后续优化，不是当前默认。 |
| shared memory / WMMA 是否可优化 GPU inter | done | TF32 WMMA tile prototype 负结果，kernel `33.53s`；文档记录原因 tiles `3.61M`。 | 可进一步研究大 segment 专用 shared-memory route。 |
| 给出 CPU/GPU intra/inter 2x2 表 | done | `THREE_MAINLINES...` `0.2`；artifact `intra_inter_2x2_current_20260707_111233` + valid `gpu_cpu_cell_current_20260707_112007`。 | 2x2 是单次构建 timing，存在机器噪声。 |
| 保存阶段优化：binary-only、skip reordered、skip LNG text | done | `SPECIAL_BLOCK_BUILD...` 保存阶段段落；`skip_export_binary_lngskip_build_20260706_121730`。 | `UNG_SKIP_LNG_TEXT_SETS=1` 不能用于 GPU cover-frontier provider。 |
| two-tier heavy sidecar 查询恢复 broad query recall | partial | `twotier_adaptive_four_20260706_093816`、`twotier_alternating_broad2_20260706_094843`；文档记录 recall 恢复和 timing 噪声。 | 仍是候选/近似路线，需更多 workload 与同进程 A/B。 |
| 和 Tagore/CAGRA/FastGrnnd 相关工作比较 | partial | `GPU_GROUP_GRAPH_RELATED_WORK_COLLISION_CN.md`、`THREE_MAINLINES...` PG/group graph 章节。 | 若写论文，需要补更正式 related-work 表和未开源方法的论文数字对齐。 |
| exact-level 语义：候选只扫描当前 activation level 拥有的边 | done | `ung_special_block_activation.h` 的 owner-level gate；`test_special_block_free_state.cpp` | 当前只物化至两层。 |
| 自动同时确定层数和阈值，且不依赖 latency/Recall/query calibration | done | QF-SSL：`analyze_auto_layer_policy.py`；`QF_SSL_AUTO_POLICY_REPORT_CN.md` 第 1--2 节 | 是结构启发式，不是最优性定理。 |
| 仅有一个候选尺度时自动退化 plain | done | singleton stop rule；Genome 最终 `layer_count=0`；单元测试覆盖 | CelebA 最终静态工件未单独生成，旧一层结果仅作开发动机。 |
| Amazon 0.5/1/5/10/30/60/80/95/99 九档等 Recall 重测 | done | router formal、low-critical compact CSV；报告第 4 节 | 99% 是 303 label-1 + 697 empty 的 synthetic mixture。 |
| 低选择率保持不变、高选择率获得二层收益 | done | Amazon 15-repeat 低档 router/plain 总体持平；统一 current binary 的 30--99% router/同 T1 单层 geomean 1.992x | “持平”是统计结论，不是逐次 wall-time 恒等。 |
| 自动分层与手工调优比较 | done | 主报告 4.1 节；统一 current binary 九档自动/prior-tuned geomean 1.010x，范围 0.932x--1.212x | prior-tuned 是一个历史固定结构，不是全局 oracle。 |
| 双层最优、单层最优与 plain 比较 | done | 主报告 4.2 节；`layer_tuning_oracle_summary.csv` 的六档历史候选 sweep | “最优”严格限定为预定义已测候选网格；历史 binary 与最终实现分表呈现。 |
| 跨数据集验证与数据集来源说明 | done | Amazon、Genome、Reviews、VariousImg；CelebA development negative；报告第 2、3、5 节 | Reviews 0.200% wall-time 因短任务调度不稳定，标为不可判定。 |
| 区分 measured/static heuristic/prior-tuned/historical/inference | done | 报告第 4--8 节显式标注 | singleton rule 是观察负例后的开发修正，不称 held-out。 |
| exact-level structural router：无 upper 授权时走 plain query path | done | `UNG_SPECIAL_REQUIRE_UPPER_AUTHORIZATION=1`；`SpecialBlockSearchUsed` 路由证据 | sidecar 仍在进程初始化时加载，内存代价不消失。 |
| 项目达到 review-ready：结构清楚、证据可复现、需求明确回答 | done | 本文件、`AGENT_KANBAN.md`、QF-SSL 主报告；两轮技术审阅和有限上下文读者审阅已完成；Python 66/66、C++ 1/1 通过。 | 当前实现最多物化两层；论文仍需将方法称为结构启发式而非最优性定理。 |
