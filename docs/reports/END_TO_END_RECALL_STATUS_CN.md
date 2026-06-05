# End-to-End Recall 当前证据与缺口

日期: 2026-06-01

本文档记录当前已有的端到端查询质量证据，并明确它能回答哪些审稿问题、不能回答哪些问题。

## 1. Reviewer 问题

**Reviewer attack:** 你们的 GPU 构建优化是否会破坏最终 filtered-search recall？

这个问题需要拆成两层：

1. **cross-edge 后端切换是否影响 recall？**  
   在 group graph 保持 CPU Vamana 的情况下，对比 CPU existing / naive GPU / paper fused。

2. **FastGrnndCuda group graph 是否影响 recall？**  
   需要对比 CPU Vamana group graph 和 FastGrnndCuda/packed exact/adaptive group graph。这个问题已在原始 Amazon PF、SIFT30、Amazon 1% x100 full-quality、Amazon 1% x200 coverage-query、Amazon 10% x40 full-quality 和 Amazon 1% x400 gather-Q 上有端到端证据。x100 最新 conservative `adaptive_cuda` route 在 full-quality 下给出 Index `5450 ms`、group `2773.35 ms`、L100/L500/L1000 `0.829/0.870/0.895`，相对历史 FastGrnnd full-quality `6370.67 ms`、`3216.21 ms`、`0.828/0.871/0.896` 改善 build 且 recall 基本一致。x200 第一轮结果暴露旧重 prune 的性能失败；diversified light prune 修复后 build time 已快于 CPU Vamana，L5000 recall gap 缩到 `0.0024`。最新 packed exact-anchor full-quality sweep 进一步把 x200 group 降到 `3827.28 ms`，repeat=3 L5000 `0.908`，与 CPU 对齐；x400 packed exact-anchor L5000 `0.967`，接近/略高历史 CPU `0.966`。10%x40 full-quality repeat=3 显示 Index `1.53x`，新增 L50/100/200/500/1000/2000/5000 sweep 显示 FastGrnnd recall 相对 CPU 为 `+0.0019/+0.0018/-0.0010/+0.0000/+0.0020/+0.0020/+0.0010`。x400 light prune 暴露 `0.0081` recall gap，但 reverse-tail+repair 和 packed exact-anchor 都给出更强候选。因此目前应写成 workload-aware GPU backend，而不是无条件无损替代。

## 2. 已有 Amazon PF 查询结果

已有日志位于：

```text
/home/graphdb/fv_runs/amazon_cpu_existing_search_pf_r3
/home/graphdb/fv_runs/amazon_naive_gpu_search_pf_r3
/home/graphdb/fv_runs/amazon_paper_fused_search_pf_r3
```

查询配置从日志可见：

| 项目 | 值 |
|---|---|
| query file | `/home/fengxiaoyao/FilterVector/FilterVectorData/Amazon/query_select_pf_A_B_C-sub-base-123456789/Amazon_query.bin` |
| query labels | `/home/fengxiaoyao/FilterVector/FilterVectorData/Amazon/query_select_pf_A_B_C-sub-base-123456789/Amazon_query_labels.txt` |
| query count | `1000` |
| query label count | `303` |
| GT | `/tmp/fv_amazon_pf_gt_K10/Amazon_gt_labels_containment.bin` |
| repeats | `3` |
| Lsearch | `1000, 2000, 5000` |

注意：这些 run 的 `query_group_id_file` 缺失，日志显示 `警告：未找到查询来源组ID文件: /tmp/fv_missing_query_group_ids.txt`。代码审查显示，`true_query_group_ids` 只在 `is_ung_more_entry == true` 时用于向 entry group 集合注入 oracle group；当前 `search_UNG_index` 没有暴露这个命令行选项，局部变量默认是 `false`，因此缺失该文件不会影响当前实验路径的 recall。它仍应在正式 artifact 中补齐，以避免未来打开 `is_ung_more_entry` 时改变语义。

## 3. Search Recall / Latency 汇总

| Variant | Lsearch | Avg recall | Batch time ms | Query P50 ms | Query P95 ms | Query P99 ms | Mean visited nodes | Mean dist calcs |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| CPU existing | 1000 | 0.908600 | 1479.140 | 23.108 | 246.578 | 795.942 | 681.9 | 1015.1 |
| CPU existing | 2000 | 0.928800 | 1106.310 | 20.309 | 141.951 | 344.915 | 894.7 | 1227.8 |
| CPU existing | 5000 | 0.969700 | 1035.900 | 18.646 | 139.046 | 330.722 | 1311.5 | 1644.6 |
| naive GPU cross | 1000 | 0.908367 | 1407.490 | 23.286 | 290.114 | 576.137 | 681.8 | 1014.9 |
| naive GPU cross | 2000 | 0.928800 | 1108.530 | 21.224 | 147.861 | 340.064 | 894.8 | 1227.9 |
| naive GPU cross | 5000 | 0.969733 | 1023.950 | 17.630 | 143.354 | 327.929 | 1311.8 | 1644.9 |
| paper fused cross | 1000 | 0.908600 | 1306.650 | 24.327 | 207.819 | 490.114 | 681.9 | 1015.1 |
| paper fused cross | 2000 | 0.928767 | 1109.220 | 20.953 | 145.568 | 351.329 | 894.7 | 1227.8 |
| paper fused cross | 5000 | 0.969767 | 1107.740 | 19.867 | 148.865 | 341.398 | 1311.8 | 1644.9 |

## 4. 作者回应

这组结果可以支撑一个较窄的结论：

> 在 Amazon PF 查询集上，当 group graph 仍使用 CPU Vamana 时，cross-edge 后端从 existing/CPU 路径切换到 naive GPU 或 paper fused GPU 后，最终 filtered-search recall 基本不变；Lsearch=1000/2000/5000 的 avg recall 差异在约 `3e-4` 量级内。

这有助于回答 reviewer 对 cross-edge 实现正确性的质疑，但不能回答 FastGrnndCuda 的质量问题。

## 5. 仍缺的关键实验

必须补：

| 缺口 | 为什么必须补 |
|---|---|
| CPU Vamana group vs FastGrnndCuda/adaptive group 的同 query / 同 GT recall A/B | 已有 Amazon PF、SIFT30、Amazon 1% x100、Amazon 1% x200、Amazon 10% x40 和 Amazon 1% x400；x100 conservative adaptive 已补 full-quality 正结果，仍需更多参数点和真实多标签来确定 router 边界 |
| Amazon 10% x40 search recall | 已有 full-quality repeat=3 L50/100/200/500/1000/2000/5000；efs 在当前 UNG-only path 为 `0` |
| Amazon 1% x400 gather-Q search recall | 已有 full-quality gather-Q A/B；仍需 reverse-tail / budgeted occlusion / quality router 缩小 light512 recall gap |
| 带 query source groups 的 rerun | 当前代码路径不使用它，但正式 artifact 应补齐，避免 future flag 改变行为 |

推荐入口：

```bash
scripts/benchmarks/run_end_to_end_recall_ab.sh
```

## 6. 新增 A/B: 原始 Amazon PF，CPU Vamana Group vs FastGrnndCuda

为了直接回答 FastGrnndCuda 是否破坏端到端查询质量，我们补跑了一组原始 Amazon PF A/B。该实验使用同一份 query、GT、search 参数和 fused cross-edge；区别只在 group graph 后端。

日志：

```text
/home/graphdb/fv_runs/reviewer_e2e_amazon_pf_group_ab_addedges_20260601_221044
```

配置要点：

| 项目 | 值 |
|---|---|
| base | `/home/fengxiaoyao/FilterVector/FilterVectorData/Amazon/Amazon_base.bin` |
| labels | `/home/fengxiaoyao/FilterVector/FilterVectorData/Amazon/Amazon_base_labels.txt` |
| points | `602,453` |
| groups | `482,388` |
| query | `query_select_pf_A_B_C-sub-base-123456789`, `1000` queries |
| GT | `/tmp/fv_amazon_pf_gt_K10/Amazon_gt_labels_containment.bin` |
| cross-edge | `UNG_CROSS_EDGE_IMPL=1`, `UNG_GPU_TOPK_IMPL=3` |
| additional edges | `UNG_ADDITIONAL_EDGES_IMPL=0`，即 CPU Vamana 补边 |
| FastGrnndCuda threshold | `UNG_TAGORE_MIN_GROUP_SIZE=128` |
| query source groups | 缺失；当前 `is_ung_more_entry=false`，不影响本实验路径 |

### 6.1 Build Time

| Variant | Index ms | group ms | cross ms | tagore groups | tagore points | tagore GNN | tagore prune |
|---|---:|---:|---:|---:|---:|---:|---:|
| CPU Vamana group | 43376.6 | 5776.25 | 20194.7 | 0 | 0 | - | - |
| FastGrnndCuda + CPU fallback | 36732.0 | 1049.92 | 18847.4 | 99 | 57001 | 103.955 | 284.333 |

Speedup:

| Metric | Speedup |
|---|---:|
| Index time | `1.18x` |
| group graph | `5.50x` |

FastGrnndCuda 只覆盖 99 个较大 group；大量小 group 走 complete fallback，少量 group 走 CPU fallback。因此这组结果说明：在原始 Amazon PF 的长尾 group 分布上，hybrid group graph 可以显著降低 group 构建时间，但端到端 Index speedup 被 LNG 和补边 cross-edge 限制。

### 6.1.1 Reviewer 追问: 为什么 group 5.50x 但 Index 只有 1.18x？

这是 Amdahl 限制，不是 group graph 加速失效。启用 CPU Vamana additional edges 后，FastGrnndCuda 已经把 group graph 从 `13.3%` 的 Index 占比降到 `2.9%`；剩余主耗时转移到 cross-edge 和 LNG。

| Variant | Graph % | LNG % | Cross % | Label % | Other ms |
|---|---:|---:|---:|---:|---:|
| CPU Vamana group + add edges | 13.32% | 27.69% | 46.56% | 9.80% | 1146.66 |
| FastGrnndCuda + add edges | 2.86% | 32.71% | 51.31% | 10.44% | 985.85 |

进一步对比 skip/additional_edges：

| Config | Variant | Index ms | Group ms | LNG ms | Cross ms | L5000 recall |
|---|---|---:|---:|---:|---:|---:|
| skip additional edges | CPU Vamana group | 30753.2 | 5189.51 | 12616.2 | 9057.34 | 0.616900 |
| skip additional edges | FastGrnndCuda | 28935.6 | 1046.64 | 14058.2 | 9393.44 | 0.615633 |
| CPU additional edges | CPU Vamana group | 43376.6 | 5776.25 | 12009.1 | 20194.7 | 0.971767 |
| CPU additional edges | FastGrnndCuda | 36732.0 | 1049.92 | 12014.6 | 18847.4 | 0.971133 |

`additional_edges` 带来的 cross 时间增量约为：

| Variant | Cross add-edge overhead |
|---|---:|
| CPU Vamana group | `11137.36 ms` |
| FastGrnndCuda | `9453.96 ms` |

作者回应：

> FastGrnndCuda 解决了 group graph 瓶颈，但完整质量配置中 additional_edges 是必要质量路径，并把 cross-edge 推高到约半数 Index time。因此端到端进一步加速需要优化 additional_edges 或把它并入 GPU exact/fused 路径，而不是继续只优化 group graph。

### 6.2 Search Recall / Latency

| Variant | Lsearch | Avg recall | Batch time ms | Query P50 ms | Query P95 ms | Query P99 ms | Mean visited nodes |
|---|---:|---:|---:|---:|---:|---:|---:|
| CPU Vamana group | 1000 | 0.912333 | 1599.83 | 34.746 | 458.906 | 787.489 | 680.2 |
| FastGrnndCuda + CPU fallback | 1000 | 0.911700 | 1429.83 | 31.749 | 452.615 | 758.292 | 691.3 |
| CPU Vamana group | 2000 | 0.932067 | 1493.11 | 27.864 | 426.262 | 754.491 | 897.6 |
| FastGrnndCuda + CPU fallback | 2000 | 0.931500 | 1385.74 | 25.819 | 405.332 | 734.896 | 913.7 |
| CPU Vamana group | 5000 | 0.971767 | 1539.42 | 29.908 | 453.851 | 796.574 | 1334.8 |
| FastGrnndCuda + CPU fallback | 5000 | 0.971133 | 1328.07 | 27.449 | 419.463 | 737.447 | 1355.0 |

Recall delta:

| Lsearch | FastGrnndCuda - CPU Vamana |
|---:|---:|
| 1000 | `-0.000633` |
| 2000 | `-0.000567` |
| 5000 | `-0.000634` |

作者回应：

> 在原始 Amazon PF 查询集上，FastGrnndCuda hybrid group graph 将 group graph 构建时间降低 `5.50x`，端到端 Index time 提升 `1.18x`，同时 filtered-search avg recall 下降小于 `0.001`。这提供了第一条 group graph 替换的端到端质量证据。

限制：

- 这不是 Amazon 1% x100 / x400 repeat 数据集。
- query source groups 缺失；当前代码路径不使用它，但正式 artifact 仍应补齐。
- 该结果依赖 CPU Vamana additional edges；跳过 additional edges 会显著降低 recall，见下一节。

## 7. 反向发现: additional_edges 不能随便跳过

同一原始 Amazon PF A/B 中，如果设置 `UNG_ADDITIONAL_EDGES_IMPL=1` 跳过补边，recall 明显下降：

| Variant | additional edges | Lsearch | Avg recall |
|---|---|---:|---:|
| CPU Vamana group | skip | 1000 | 0.613100 |
| CPU Vamana group | skip | 2000 | 0.614467 |
| CPU Vamana group | skip | 5000 | 0.616900 |
| FastGrnndCuda + CPU fallback | skip | 1000 | 0.612167 |
| FastGrnndCuda + CPU fallback | skip | 2000 | 0.613233 |
| FastGrnndCuda + CPU fallback | skip | 5000 | 0.615633 |

日志：

```text
/home/graphdb/fv_runs/reviewer_e2e_amazon_pf_group_ab_20260601_220540
```

结论：

- 对原始 Amazon PF 查询，`additional_edges` 是质量关键路径。
- 任何只用 `UNG_ADDITIONAL_EDGES_IMPL=1` 的 build speedup 表，都不能单独作为最终端到端质量结论。
- 论文中必须拆开报告：`skip additional_edges` 是 stage ablation，不是完整质量配置。

## 8. 新增证据: Amazon 1% x200 Coverage Query 与 Light Prune

这组实验回答 Amazon repeat-scale 上的端到端质量和构建性能。query 不是单组 query，而是按 coverage profile 从 group hierarchy 中采样：

| Metric | Value |
|---|---:|
| queries | `1000` |
| broad / medium / narrow | `316 / 420 / 264` |
| matched groups avg / p50 / p95 | `258.97 / 32 / 1087.4` |
| matched points avg / p50 / p95 | `55556.6 / 6800 / 244500` |
| GT time | `8733 ms` |

日志：

```text
/home/graphdb/fv_runs/reviewer_e2e_amazon_1pct_x200_coverage_20260601_230224
```

配置要点：

| 项目 | 值 |
|---|---|
| base | `/home/graphdb/FilterVectorBenchData/fv_amazon_1pct_x200_jitter_nonempty/Amazon_1pct_x200_base.bin` |
| labels | `/home/graphdb/FilterVectorBenchData/fv_amazon_1pct_x200_jitter_nonempty/Amazon_1pct_x200_labels.txt` |
| query | `query_coverage_1000/query.bin` |
| GT | `query_coverage_1000_gt/gt_K10_containment.bin` |
| cross-edge | `UNG_CROSS_EDGE_IMPL=1`, `UNG_GPU_TOPK_IMPL=3` |
| additional edges | `UNG_ADDITIONAL_EDGES_IMPL=0`，CPU Vamana |
| repeats | `1` |

### 8.1 Build Time

| Variant | Index ms | group ms | cross ms | save-index ms |
|---|---:|---:|---:|---:|
| CPU Vamana group | `17596.8` | `8511.72` | `4601.91` | `21533.9` |
| FastGrnndCuda, old heavy prune | `38273.7` | `26620.6` | `6055.72` | `19581.3` |
| FastGrnndCuda, top32 light prune | `14211.8` | `6016.04` | `4926.37` | `20577.7` |
| FastGrnndCuda, diversified light prune | `16478.5` | `6706.36` | `4857.33` | `23396.3` |
| old batched exact kNN, confounded | `12595.9` | `4356.18` | `4120.75` | - |
| packed exact-anchor router, nx4096 | `11043.5` | `3827.28` | `4868.15` | - |

FastGrnndCuda group graph 细分：

| Variant | direct build | H2D | GNN | prune | D2H | fill |
|---|---:|---:|---:|---:|---:|---:|
| old heavy prune | `26282 ms` | `1538.32` | `3551.54` | `19218.3` | `413.047` | `318.071` |
| top32 light prune | `5746.81 ms` | `1180.47` | `3198.35` | `712.392` | `134.631` | `247.246` |
| diversified light prune | `6365.69 ms` | `1448.82` | `3229.56` | `759.66` | `161.408` | `317.508` |
| old batched exact kNN, confounded | `3950.64 ms` | `524.173` | `674.693` | `0` | `29.8523` | `381.630` |
| packed exact-anchor router, nx4096 | `2899.6 ms` | `347.710` | `925.721` | `0` | `29.742` | `902.017` |

### 8.2 Search Recall

| Variant | Lsearch | Avg recall | Avg time ms |
|---|---:|---:|---:|
| CPU Vamana group | 100 | `0.8230` | `586.593` |
| FastGrnndCuda | 100 | `0.8125` | `480.749` |
| CPU Vamana group | 200 | `0.8250` | `426.670` |
| FastGrnndCuda | 200 | `0.8273` | `410.823` |
| CPU Vamana group | 500 | `0.8490` | `454.759` |
| FastGrnndCuda | 500 | `0.8472` | `428.388` |
| CPU Vamana group | 1000 | `0.8690` | `445.490` |
| FastGrnndCuda, old heavy prune | 1000 | `0.8662` | `427.785` |
| FastGrnndCuda, top32 light prune | 100 | `0.8135` | `514.950` |
| FastGrnndCuda, top32 light prune | 200 | `0.8239` | `431.594` |
| FastGrnndCuda, top32 light prune | 500 | `0.8421` | `432.069` |
| FastGrnndCuda, top32 light prune | 1000 | `0.8618` | `431.559` |
| FastGrnndCuda, diversified light prune | 1000 | `0.8656` | `533.818` |
| CPU Vamana group | 2000 | `0.8820` | `449.970` |
| FastGrnndCuda, top32 light prune | 2000 | `0.8769` | `467.524` |
| CPU Vamana group | 5000 | `0.9080` | `486.791` |
| FastGrnndCuda, top32 light prune | 5000 | `0.9029` | `492.195` |
| FastGrnndCuda, diversified light prune | 5000 | `0.9056` | `523.592` |
| old batched exact kNN, confounded | 1000 | `0.815867` | `481.520` |
| old batched exact kNN, confounded | 5000 | `0.819867` | `488.171` |
| packed exact-anchor router, nx4096 | 1000 | `0.869` | `515.379` |
| packed exact-anchor router, nx4096 | 5000 | `0.908` | `468.162` |

结论：

- 旧 heavy prune 是明确失败案例：group graph 比 CPU Vamana 慢 `3.13x`，Index time 慢 `2.18x`；原因主要是 prune `19218.3 ms`。
- diversified light prune 把 prune 降到 `759.66 ms`，group graph 快于 CPU Vamana `1.27x`，Index 快 `1.07x`。
- 质量方面，diversified light prune L1000 低 CPU `0.0034`，L5000 低 `0.0024`。这说明性能问题已经解决，但系统级替代路线还需要补一个轻量质量增强。
- 旧 batched exact kNN 是 confounded 结果：group graph 更快，但 L5000 recall 只有 `0.819867`；后续结构诊断显示它与新 packed exact 的组内图一致，差异来自 cross edges 缺失。
- packed exact-anchor router 在 full-quality 口径下 x200 L5000 `0.908`，已经不能再用旧 exact 负结果说明 exact 组内图不可行。
- 当前写法应是：中等组使用 diversified light prune，大组使用 strong prune；下一步补低成本 reverse augmentation 或 budgeted occlusion。

## 9. 可写入论文的谨慎表述

## 8. 新增 A/B: SIFT30，修复后 Fused Cross 固定下的 Group Graph 端到端 A/B

为了回答“结果是否只对 Amazon 有效”的 reviewer 质疑，我们补跑了一组 SIFT30 端到端查询实验。该实验固定 query、containment GT、search 参数、修复后的 fused cross-edge 和 CPU Vamana additional edges，只切换 group graph 后端。

日志：

```text
/home/graphdb/fv_runs/debug_sift30_fused_cross_exactdist_cpu_20260601_224812
/home/graphdb/fv_runs/debug_sift30_fused_cross_exactdist_fast_20260601_225006
```

修复前有两类失败/错误结果：

1. `/home/graphdb/fv_runs/reviewer_e2e_sift30_group_ab_20260601_222438`：`DIRECT_QID_ALL_FUSED=1` 跳过 Q materialization，但 `nx>4096` 的大组回落旧 tiled path 仍读取 `g_d_Q`，触发 `naive dot kernel launch failed in full-tile path`。
2. `/home/graphdb/fv_runs/debug_sift30_fused_cross_singletonfix_cpu_20260601_224505`：修掉 crash 后 recall 仍低，原因是 host merge 路径启用 id-only writeback，用 rank 代替真实距离，破坏跨 target-group 的 `SearchQueue` 排序。

代码修复：

- 当存在 singleton 或 direct-qid fused 未覆盖的大组时，自动关闭 `direct_qid_all_effective`，materialize `g_d_Q` 和 `q_norm`。
- singleton fastpath 不再因为 `direct_qid_fused=1` 被跳过。
- `id-only writeback` 只允许在 GPU global merge 路径下生效；host merge 路径必须 D2H 真实距离。

配置要点：

| 项目 | 值 |
|---|---|
| base | `/home/graphdb/aaaGPU/FilterVectorData/sift/sift_base.bin` |
| base labels | `sift_base_30_labels_zipf_origstyle.txt` |
| points | `1,000,000` |
| groups | `69,176` |
| query | `sift_query.bin`, `10,000` queries |
| query labels | `sift_query_12_labels_zipf_containment.txt` |
| GT | `/home/graphdb/fv_runs/reviewer_e2e_sift30_group_ab_20260601_222438/GroundTruth/GT_._K10/sift_gt_labels_containment.bin` |
| cross-edge | `UNG_CROSS_EDGE_IMPL=1`, `UNG_GPU_TOPK_IMPL=3`, strict fused |
| additional edges | `UNG_ADDITIONAL_EDGES_IMPL=0`，CPU Vamana |
| FastGrnndCuda threshold | `UNG_TAGORE_MIN_GROUP_SIZE=128` |

### 8.1 Build Time

| Variant | Index ms | Group ms | Cross ms | Tagore groups | Tagore points |
|---|---:|---:|---:|---:|---:|
| CPU Vamana group | 31740.4 | 11245.3 | 17886.2 | 0 | 0 |
| FastGrnndCuda + CPU fallback | 24989.9 | 1889.79 | 17868.9 | 844 | 693196 |

Speedup:

| Metric | Speedup |
|---|---:|
| Index time | `1.27x` |
| group graph | `5.95x` |
| cross-edge vs CPU Vamana cross | `3.90x` |

FastGrnndCuda 直接处理 844 个大 group / 693,196 个点；67,629 个小 group 走 complete fallback，703 个 group / 62,611 个点走 CPU fallback。修复后的 fused cross-edge 在 CPU Vamana group 下为 `17886.2 ms`，相比同配置 CPU Vamana cross 的 `69834.5 ms` 为 `3.90x`。

### 8.2 Search Recall / Latency

| Variant | Lsearch | Avg recall | Batch time ms | Query P50 ms | Query P95 ms | Query P99 ms | Mean visited nodes |
|---|---:|---:|---:|---:|---:|---:|---:|
| CPU Vamana group | 100 | 0.609423 | 1116.36 | 8.440 | 37.918 | 44.364 | 2184.2 |
| FastGrnndCuda + CPU fallback | 100 | 0.607997 | 1068.88 | 8.110 | 37.886 | 43.041 | 2030.0 |
| CPU Vamana group | 200 | 0.706580 | 1089.39 | 9.267 | 39.890 | 44.604 | 3850.9 |
| FastGrnndCuda + CPU fallback | 200 | 0.706287 | 1046.82 | 8.882 | 38.442 | 43.306 | 3579.0 |
| CPU Vamana group | 500 | 0.814600 | 1289.81 | 11.655 | 43.940 | 50.763 | 7924.7 |
| FastGrnndCuda + CPU fallback | 500 | 0.812680 | 1248.73 | 11.056 | 43.280 | 50.586 | 7350.7 |
| CPU Vamana group | 1000 | 0.877587 | 1579.17 | 14.804 | 52.273 | 62.739 | 13214.6 |
| FastGrnndCuda + CPU fallback | 1000 | 0.876297 | 1505.12 | 14.071 | 49.754 | 59.882 | 12285.0 |

Recall delta:

| Lsearch | FastGrnndCuda - CPU Vamana |
|---:|---:|
| 100 | `-0.001426` |
| 200 | `-0.000293` |
| 500 | `-0.001920` |
| 1000 | `-0.001290` |

作者回应：

> SIFT30 的端到端结果支持 FastGrnndCuda 在 Amazon 之外仍能保持较小 recall 损失：group graph 加速 `5.95x`、Index 加速 `1.27x`，Lsearch=1000 recall 下降约 `0.00129`。修复后的 fused cross-edge 与 CPU cross 的 recall 基本一致，并将 SIFT30 cross-edge 从 `69834.5 ms` 降到 `17886.2 ms`。这次 reviewer 迭代也暴露并修复了两个容易被攻击的语义问题：direct-qid-all 对 fallback path 的覆盖不足，以及 id-only writeback 在 host merge 路径下破坏跨组排序。

## 9. 可写入论文的谨慎表述

可以写：

- Existing Amazon PF search runs show that replacing the cross-edge backend with the paper fused implementation preserves filtered-search recall when the intra-group graph is fixed to CPU Vamana.
- This validates the cross-edge backend at query level, but does not yet validate FastGrnndCuda as a drop-in replacement for CPU Vamana group graphs.
- On the original Amazon PF query workload with CPU Vamana additional edges enabled, FastGrnndCuda hybrid group graph reduces group construction time by `5.50x` and index time by `1.18x`, with average recall loss below `0.001`.
- On SIFT30 with fixed fused cross-edge and CPU additional edges, FastGrnndCuda hybrid reduces group construction time by `5.95x` and index time by `1.27x`, with Lsearch=1000 average recall loss of about `0.00129`.
- On Amazon 1% x200 coverage queries, old heavy-prune FastGrnndCuda fails as a performance optimization, but diversified light prune makes it faster than CPU Vamana with a much smaller remaining recall gap.

不能写：

- FastGrnndCuda preserves end-to-end recall on all workloads.
- FastGrnndCuda always speeds up group graph construction.
- The full optimized pipeline is recall-equivalent to CPU Vamana.
- `UNG_ADDITIONAL_EDGES_IMPL=1` 跳过补边的结果不能支撑最终端到端质量结论。
