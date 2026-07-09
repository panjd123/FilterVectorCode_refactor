# Special Block Build/Query A/B 初步结论

本文记录当前 special block graph/query 的 full-quality filtered-search A/B。结论先行：

## 0. Reviewer 当前口径速览

这一节只回答 reviewer 最容易混淆的问题：当前 special block 哪些数字能引用，哪些只是历史负结果或诊断。

### 0.1 当前推荐/候选配置

| 配置项 | 当前口径 | 说明 |
|---|---|---|
| 数据口径 | Amazon 100% x1 restored | special block 只面向 x1/original 语义；restored 数据用于当前诊断实验。 |
| special threshold | `UNG_SPECIAL_BLOCK_MIN_POINTS=100` | T=100。 |
| intra | CPU default: exact-topK `th=2048`; GPU default: exact-topK `th=128` + Tagore/FastGrnnd CUDA | GPU intra 是当前推荐。 |
| inter | GPU source-exact cuda-core, `UNG_SPECIAL_GPU_INTER_WARPS=2` | `8` warps/query 是旧慢默认。 |
| cap / iter | `UNG_SPECIAL_INTER_MAX_PAIR_WORK=10000000`, `UNG_TAGORE_ITER=4` | 这是近似候选配置，不是无条件 full-quality 默认。 |
| query | `UNG_SPECIAL_BLOCK_SEARCH=1`, free-state search | 普通搜索不能评价 special index 质量。 |
| quality gate | filtered-search recall / latency | 局部 topK overlap 不作为质量黄金标准。 |

### 0.2 当前能引用的 special block 数字

| case | quality class | index | overlay | query evidence | claim status | main artifact |
|---|---|---:|---:|---|---|---|
| GPU intra + GPU inter tuned | approximate candidate | `~28.6 s` | `~7.87 s` | coverage L50/L200 `0.344/0.403` smoke | 可引用为 tuned special overlay 性能；不能写成默认 full-quality 端到端 speedup | `/home/graphdb/fv_runs/special_blocks_20260705/special_gpu_inter_warps2_default_smoke_20260707_114920/summary.txt` |
| CPU intra + CPU inter corrected | diagnostic baseline | `105.78 s` | `85.50 s` | 未单独完整 query sweep | 可用于 CPU/GPU stage 对照 | `/home/graphdb/fv_runs/special_blocks_20260705/intra_inter_2x2_current_20260707_111233` |
| GPU intra + CPU inter corrected | diagnostic baseline | `38.60 s` | `18.07 s` | 未单独完整 query sweep | 用于说明 GPU inter 的边际收益 | `/home/graphdb/fv_runs/special_blocks_20260705/gpu_cpu_cell_current_20260707_112007` |
| no-special baseline | full-quality baseline | `17.21 s` | N/A | 旧同环境 baseline | 当前 special 不能宣称 Index speedup 的主要原因 | `/home/graphdb/fv_runs/special_blocks_20260705/threadfix_normal_ab_20260706_025123/threadfix_normal_summary.md` |

### 0.2b Special Block Recall 成绩总表

| 结果类型 | workload | Lsearch | recall | 质量含义 | artifact |
|---|---|---:|---:|---|---|
| tuned GPU/GPU build smoke | `query_coverage_100` | `50` | `0.344` | 当前 GPU intra/inter 调参后 coverage smoke | `/home/graphdb/fv_runs/special_blocks_20260705/special_gpu_inter_warps2_default_smoke_20260707_114920/summary.txt` |
| tuned GPU/GPU build smoke | `query_coverage_100` | `200` | `0.403` | 同上 | 同上 |
| corrected CPU exact-topK intra | `query_coverage_100` | `50` | `0.341` | 接近旧 CPU Vamana `0.344`，说明 exact-topK 分流可作为 corrected CPU baseline | `/home/graphdb/fv_runs/special_blocks_20260705/exact_topk_th2048_search_smoke_20260706_131808` |
| corrected CPU exact-topK intra | `query_coverage_100` | `200` | `0.400` | 接近旧 CPU Vamana `0.401` | 同上 |
| bounded ring negative | `query_coverage_100` | `50` | `0.323` | 快但质量下降，不可默认 | `/home/graphdb/fv_runs/special_blocks_20260705/bounded_th2048_search_smoke_20260706_130755` |
| bounded ring negative | `query_coverage_100` | `200` | `0.370` | 同上 | 同上 |
| adaptive-heavy | `query_coverage_100` | `800` | `0.464` | 四 workload 验证中与 no-cap 可对齐 L recall delta 为 0 | `/home/graphdb/fv_runs/special_blocks_20260705/twotier_adaptive_four_20260706_093816` |
| adaptive-heavy | `query_mix_100_seed456` | `800` | `0.702` | 同上 | 同上 |
| adaptive-heavy | `query_broad2_100_seed901` | `800` | `0.488` | broad2 L800 恢复 no-cap recall；light-only L800 为 `0.485` | 同上 |
| adaptive-heavy | `query_len1_broad_100_seed902` | `800` | `0.360` | query-size gate 避免 len1 broad 无效 heavy 触发，recall 不变 | 同上 |

这张表只汇总 recall，不直接声称 query latency 加速。query wall time 在多次实验中有 batch/load/outlier 噪声，不能单独从这些 smoke 数字推出稳定 query speedup。

### 0.3 如何阅读旧表

- `special T100 skip`、`special-as-group T100 skip` 和旧 `parent-child exact all-pairs` 是历史负结果，用于说明为什么旧 overlay 不可接受。
- `CPU intra + GPU inter = 155.92 s -> 5.16 s` 是未修正 CPU Vamana baseline；公平 CPU baseline 已改为 exact-topK `th=2048`，GPU intra 加速约 `14.4x`。
- `bounded ring` 是负结果：速度快但 coverage L50/L200 recall 降到 `0.323/0.370`。
- `TF32 WMMA tile` 是负结果：tiles 太碎，kernel `33.53 s`。
- two-tier heavy sidecar / cap10 / iter4 是候选近似路线，必须继续用端到端 filtered-search recall 验证。

### 0.4 最小复现命令

以下命令用于验证当前代码可以构建，并检查文档 claim 语言；完整性能复现使用下表 artifact 中已保存的脚本和日志。

```bash
cd /home/graphdb/FilterVectorCode_refactor
cmake --build /home/graphdb/FilterVectorCode_refactor/build_mode_switch -j 16 --target build_UNG_index search_UNG_index
python3 tools/benchmarks/check_paper_claim_language.py \
  docs/reports/THREE_MAINLINES_METHOD_BASELINE_DATA_SPEEDUP_CN.md \
  docs/reports/SPECIAL_BLOCK_BUILD_QUERY_AB_CN.md \
  docs/reports/SPECIAL_BLOCK_GRAPH_QUERY_DESIGN_CN.md \
  docs/reports/REVIEW_READY_HANDOFF_CN.md \
  docs/reports/REVIEW_READY_REQUIREMENTS_MATRIX_CN.md
```

### 0.5 当前不能写的结论

- 不能写 special block 已经带来默认 full-quality 端到端 Index speedup。
- 不能写 cap10/iter4 是无损默认。
- 不能用局部 topK overlap 替代 filtered-search recall。
- 不能把 `UNG_SKIP_LNG_TEXT_SETS=1` 当作 GPU cover-frontier provider 可用的通用格式。


- special block coverage 很高，但不能直接等同 speedup。
- 当前 T=100 实现能在部分查询设置下降低搜索工作量：L50 per-query mean latency 从 `556.56 ms` 降到 `408.58 ms`，约 `1.36x`；L200 从 `347.84 ms` 到 `369.35 ms`，没有加速。
- 构建成本目前明显不可接受：Index time 从 `119.94 s` 增到 `564.86 s`，主要来自 special sidecar edge overlay `444.90 s`。
- 平凡 special 的普通图/跨组 source skip 只省了很小一部分工作：group skip `36` groups / `7317` points，cross skip `169` source-target group pairs / `40933` query-vector visits，additional skip `36` groups / `7317` points。
- T=1600 skip-only 在 15 分钟 timeout 内未完成，日志显示普通 cross 已完成但 special overlay 仍未产出最终 build_time；因此 CPU special overlay 没有可写成论文正结果的 build-time Pareto 点。
- 后续 root-cause audit 发现上述 CPU heuristic `413.49 s` overlay 含有 OpenMP 线程数污染：CPU special intra 调用 `Vamana::build(..., 1)` 后把全局 OpenMP 线程数留在 `1`，special inter 没有显式恢复 `_num_threads`，导致 inter 近似单线程。修复后 CPU/CPU T=100 overlay 为 `165.19 s`，inter 为 `12.99 s`；GPU intra 的主要价值应重新解释为把 special intra 从约 `152 s` 降到约 `8 s`，而不是让 CPU inter 因 child graph 变快。
- 最新 special intra route 复核进一步确认：旧 CPU intra baseline 的阈值确实不合理。T=100 时普通 group `complete_threshold=64` 小于 special block 构造阈值，因此旧 CPU intra 把全部 `1974` 个 special blocks / `577822` 点送入 CPU Vamana，耗时 `155.92 s`；这只是历史慢 baseline。已新增 special-intra 独立分流：CPU default 使用 distance exact-topK 处理 `n<=2048` 的 special blocks，GPU default 调到 `th=128`，小块 exact-topK、大块 Tagore/FastGrnnd CUDA。修正后 CPU default intra 为 `75.88 s`，GPU default intra 为 `5.11-5.28 s`，GPU 相对更公平 CPU baseline 仍有约 `14.4x` stage speedup；旧 `30.24x` 不能再作为公平 CPU baseline 口径。
- 当前仍不能写 special block 已带来端到端 index speedup。同环境 no-special baseline 为 `17.21 s`（ordinary GPU cross active, cross `2.03 s`）；threadfix 后正常 A/B 中 CPU special 为 `326.50 s`，GPU intra + CPU inter 为 `211.67 s`，GPU intra + GPU inter 为 `187.95 s`。恢复 GPU source-skip 后，cap10 + `UNG_TAGORE_ITER=4` + GPU intra/inter 的最新 controlled run 为 index `29.59 s`、overlay `11.21 s`、ordinary cross `1.59 s`，但 cap10 仍是近似 knob，不能直接写成默认 full-quality speedup。

实现状态更新：上述 full-quality A/B 中 `special T100 skip` 是重构前的 `parent-child exact all-pairs` special inter overlay 结果，保留为负结果基线。当前代码已改为把 special block 视作 special group，inter special edges 复用普通 CPU cross 的 target-search helper：小 pair exact scan，大 pair 在 child special group 的 Vamana/complete graph 上搜索。该新实现已通过小型 x1 filtered-search smoke。最新 profiling 表明，CPU special inter 必须显式设置 OpenMP 线程数；修复前的 CPU inter 慢主要是线程数泄漏，不是 target-search helper 语义本身。

## 1. 实验口径

数据集：

```text
/home/graphdb/fv_runs/special_blocks_20260705/amazon_100pct_x1_restored_from_x40
```

构造方式：

- base vectors 从 Amazon 100% x40 的 `Amazon_100pct_x40_base.bin` 每 40 行取 1 行，得到 `582117` 点、`768` 维。
- base labels 使用 `/home/graphdb/fv_runs/special_tiny_subtrees_20260702/amazon_100pct_x1_from_x40_labels.txt`。
- 这是 `x1-restored-from-x40` 数据，不应写成独立官方原始 x1 数据；可以作为 x1 口径诊断证据。

Query/GT：

```text
query: /home/graphdb/fv_runs/special_blocks_20260705/amazon_100pct_x1_restored_from_x40/query_coverage_100
gt:    /home/graphdb/fv_runs/special_blocks_20260705/amazon_100pct_x1_restored_from_x40/query_coverage_100_gt/gt_K10_containment.bin
```

Query generator 生成 `100` 条 coverage-style 查询，query length=2。该 workload 偏 broad：

| metric | value |
|---|---:|
| broad queries | 91 |
| medium queries | 9 |
| matched groups avg / p50 / p95 | `24164.8 / 4992 / 73643.8` |
| matched points avg / p50 / p95 | `29466.0 / 5426 / 86058.3` |

Build/search 统一使用：

```text
UNG_CROSS_EDGE_IMPL=3          # cpu_exact_scan
UNG_ADDITIONAL_EDGES_IMPL=2    # cpu_exact_scan
NUM_THREADS=128
LSEARCH_VALUES="50 200"
K=10
NUM_REPEATS=1
```

质量标准是 build 后 filtered-search recall，不使用局部 topK overlap。

## 2. Threshold / Pareto 状态

覆盖率 sweep 使用已有 dense T artifact：

```text
/home/graphdb/fv_runs/special_blocks_20260704/bottom_up_special_blocks_dense_T.md
```

覆盖率不能当 speedup，但它决定哪些 T 值值得实测。选取小/中/大阈值如下：

| T | blocks | trivial | nontrivial | query special ratio | query nontrivial ratio | full-quality performance status |
|---:|---:|---:|---:|---:|---:|---|
| 25 | 7308 | 137 | 7171 | `0.8623` | `0.8336` | coverage only，block 数过多，未做 full-quality build |
| 100 | 1974 | 36 | 1938 | `0.8328` | `0.8162` | 完成 full-quality A/B；查询有条件收益，构建显著变差 |
| 400 | 516 | 4 | 512 | `0.7964` | `0.7919` | 尝试 skip-only，构建过慢后中断，未作为完整性能点 |
| 1600 | 131 | 0 | 131 | `0.7519` | `0.7519` | skip-only `timeout 900s`，未完成 special index |
| 6400 | 30 | 0 | 30 | `0.7051` | `0.7051` | coverage only，未做 full-quality build |
| 12800 | 16 | 0 | 16 | `0.6842` | `0.6842` | coverage only，未做 full-quality build |

当前 Pareto 判断：

- T=100 是唯一完成 full-quality build/search A/B 的点；重构前 exact-overlay 和重构后 special-as-group 版本都完成了，但 build cost 都不可接受。
- T=1600 覆盖率仍高，但重构前构建未能在 900s 内完成，说明仅增大阈值不足以消除旧 sidecar 构建问题。
- 当前 special-as-group 重构后，T=100 per-source Vamana 版本构建从旧 exact-overlay 的 `564.86 s` 进一步变为 `941.25 s`，说明直接逐点调用 Vamana 不是可接受的构建方案。加入 `target_nx <= 1000` brute-force 分流后，T=100 降到 `548.87 s`，比旧 exact-overlay 略快，但仍远慢于 baseline `119.94 s`。

## 3. 代码与脚本改动

新增/更新的 instrumentation：

- `build_time.csv` / `meta` 写出 special metadata time、intra/inter overlay time、special edge count、trivial skip 计数。
- `query_details_repeat*.csv` 写出 special coverage、nontrivial coverage、free/regular nodes、distance calcs、special/regular edge scans。
- `UNG_SPECIAL_BLOCK_SKIP_TRIVIAL=0/1` 支持保留或关闭平凡 special 的普通图/跨组 source skip，默认仍为 `1`。
- `query_covers_special_block` 修正为 containment 语义：query labels 必须被 block root label-set 包含。

新增工具：

| 文件 | 用途 |
|---|---|
| `tools/benchmarks/extract_stride_vecs_bin.py` | 从 repeat x40 base 中按 stride 恢复 x1-restored base |
| `scripts/benchmarks/run_special_block_build_query_ab.sh` | 对多个 T 编排 baseline/special/ablation |
| `tools/benchmarks/summarize_special_block_ab.py` | 汇总 build/query/bucket 表 |

GPU 接入状态：

- `UNG_SPECIAL_BLOCK_GPU_INTRA=1` 已接入 special block 组内图构建：把 special block 的非连续 member points pack 成连续 float buffer，调用现有 Tagore/FastGrnnd CUDA batch builder，再把 local graph 映射回 special sidecar edges。
- 小 block 仍走 complete/bounded-complete；GPU intra 只处理超过 complete threshold 的 special block。注意这里的 complete 是组内图构建分流，和 special inter/cross 的 exact topK scan 不是同一层逻辑。当前 T=100 special block 在默认 `UNG_GROUP_GRAPH_COMPLETE_NX=64` 下全部超过 complete threshold，因此 CPU-only special intra 全部走 Vamana，而不会自动走 exact/complete。
- special inter/cross 已接入一个 special 专用 packed source-exact GPU path：source 使用 parent special block member points，target 使用 child special block member group ranges 作为多 segment descriptor，复用现有 source-exact topK CUDA kernel。普通 GPU cross backend 仍然不能直接复用，因为它强依赖普通 group 的连续 `_group_id_to_range` 和 `_label_nav_graph` in/out-neighbors。


当前实现重构：

- 普通 CPU cross 的 `CpuVamana`、`CpuExactScan`、`CpuHybridScanVamana` 已收敛到同一套 target-search helper。
- helper 的语义是：`target_nx <= UNG_CPU_TARGET_EXACT_MAX_NX` 或 `nq * nx <= UNG_CPU_HYBRID_SCAN_MAX_WORK` 时 exact scan；否则复用 target group 的 Vamana/complete graph 搜索；最终按 `_num_cross_edges` 保留 topK。当前 CPU heuristic baseline 使用 `UNG_CPU_TARGET_EXACT_MAX_NX=1000`。已完成 T=100 和 T=400 full-quality A/B；T=1600 沿用此前 timeout/coverage-only 结论。
- special inter edges 也调用同一 helper，把 child special block 当成特殊 target group，而不是再写一套 all-pairs exact topK。
- 小型 x1 smoke artifact：`/home/graphdb/fv_runs/special_blocks_20260705/special_group_refactor_smoke`，recall `1.0`。
- 大数据重跑 artifact：`/home/graphdb/fv_runs/special_blocks_20260705/amazon_x1_restored_special_group_T100_rerun`。

## 4. Build-Time Breakdown

完成的 full-quality A/B：

```text
/home/graphdb/fv_runs/special_blocks_20260705/amazon_x1_restored_ab_T100_400_1600
```

| case | T | skip trivial | index ms | group ms | cross ms | special metadata ms | special intra ms | special inter ms | special edge overlay ms |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| baseline plain | - | - | `119942` | `2326.52` | `103882` | `0` | `0` | `0` | `0` |
| special T100 skip | 100 | 1 | `564856` | `2376.43` | `100780` | `1545.34` | `154825` | `290070` | `444895` |
| special-as-group T100 skip | 100 | 1 | `941250` | `2412.57` | `122032` | `1545.34` | `152599` | `647846` | `800445` |
| target-nx<=1000 heuristic T100 skip | 100 | 1 | `548867` | `2228.65` | `117047` | `1695.13` | `151841` | `261644` | `413485` |
| GPU intra T100 skip | 100 | 1 | `149437` | `2191.24` | `111586` | `1655.37` | `7808.03` | `12543.9` | `20352` |
| GPU intra+inter T100 skip | 100 | 1 | `159614` | `1519.53` | `122187` | `1684.03` | `7811.03` | `11255.8` | `19066.9` |
| target-nx<=1000 heuristic T400 skip | 400 | 1 | `943617` | `2330.55` | `101832` | - | `232274` | `591847` | `824121` |

T=100 的旧 special overlay 数据需要分两层读。修复前，special-as-group per-source Vamana 重构后 inter 从旧 exact-overlay 的 `290.07 s` 变为 `647.85 s`；加入 `target_nx<=1000` 分流后 inter 降到 `261.64 s`、overlay `413.49 s`。后续 route profile 证明这里存在 OpenMP 线程数污染：CPU special intra 的 `Vamana::build(..., 1)` 把全局 OpenMP 线程数留在 `1`，special inter 的 parallel loop 未显式 `num_threads(_num_threads)`。修复后 CPU/CPU T=100 profile 中 inter 降到 `12.99 s`、overlay 降到 `165.19 s`；正常 A/B 中 CPU/CPU 为 index `326.50 s`、inter `12.18 s`、overlay `162.42 s`，GPU intra + CPU inter 为 index `211.67 s`、inter `11.96 s`、overlay `19.72 s`，GPU intra + GPU inter 为 index `187.95 s`、inter `11.12 s`、overlay `18.82 s`；同环境 no-special baseline 为 index `17.21 s`、cross `2.03 s`。因此，GPU intra 的可信构建收益主要是 special intra 本身 `~150 s -> ~8 s`，不是 CPU inter 依赖 GPU child graph 才变快。GPU inter 在 inter 已正确并行后的边际收益较小（normal A/B 中 inter `11.96 s -> 11.12 s`）。T=400 查询 recall 更高，但旧 CPU overlay 升到 `824.12 s`、index `943.62 s`，不是构建 Pareto 点；需要在 threadfix 后重跑再重新判断。

Threadfix/profile 证据目录：`/home/graphdb/fv_runs/special_blocks_20260705/clean_ab2_20260705_222431/inter_route_profile_note.md`。

| case | exact loop | graph loop | special inter | special overlay | index |
|---|---:|---:|---:|---:|---:|
| CPU/CPU before threadfix | `131.92 s` | `149.54 s` | `281.66 s` | `436.66 s` | `611.33 s` |
| CPU/CPU after threadfix | `8.06 s` | `4.74 s` | `12.99 s` | `165.19 s` | `352.74 s` |
| GPU intra + CPU inter | `7.64 s` | `4.38 s` | `12.23 s` | `20.11 s` | `210.26 s` |

Threadfix 后正常 A/B（无 profile/no-output/reserve 诊断开关）结果：

| case | special intra | special inter | special overlay | index |
|---|---:|---:|---:|---:|
| CPU intra + CPU inter | `150.24 s` | `12.18 s` | `162.42 s` | `326.50 s` |
| GPU intra + CPU inter | `7.76 s` | `11.96 s` | `19.72 s` | `211.67 s` |
| GPU intra + GPU inter | `7.69 s` | `11.12 s` | `18.82 s` | `187.95 s` |

正常 A/B 证据目录：`/home/graphdb/fv_runs/special_blocks_20260705/threadfix_normal_ab_20260706_025123/threadfix_normal_summary.md`。查询时间在该 run 中仍主要受 `MinSupersetT_ms` 和 batch wall time 波动影响；对齐到 query/repeat 后，recall、DistCalcs 和 edge-scan median delta 都很小，mean core-time 差异主要由少数 outlier 拉动。因此 query 侧结论需要单独复核，不应从本表直接推出稳定 query speedup。

CPU intra 路径复核：为了回答“CPU intra 为什么不会自动 exact/complete”，补充了 route counter，并在 cap10 + `UNG_TAGORE_ITER=4` + GPU inter 固定配置下对比 CPU intra 与 GPU intra。结果表明这不是 33-64 点小块漏分流造成的：在 T=100 和默认 `complete_threshold=64` 下，所有 special blocks 都超过 complete cutoff。

证据目录：`/home/graphdb/fv_runs/special_blocks_20260705/special_intra_threshold_ab_20260706_075906`。

| case | index | special intra | special inter | overlay | intra edges | L50/L200 recall | route |
|---|---:|---:|---:|---:|---:|---|---|
| CPU intra + GPU inter | `182.91 s` | `155.92 s` | `6.13 s` | `162.06 s` | `14,757,710` | `0.344 / 0.401` | `complete=0 blocks, cpu_vamana=1974 blocks / 577822 points` |
| GPU intra + GPU inter | `29.59 s` | `5.16 s` | `6.05 s` | `11.21 s` | `16,122,116` | `0.345 / 0.403` | `gpu_intra=1974 blocks / 577822 points, fallback=0` |

解释：这张表是旧 CPU Vamana baseline，不是最终公平 CPU baseline。旧 CPU intra 是 special block 内部导航图构建，但没有单独的 small-block exact-topK 分流；由于 T=100 而普通 group complete threshold 只有 `64`，所有 special blocks 都进入 CPU Vamana，因此 `155.92 / 5.16 = 30.24x` 会高估 GPU 相对合理 CPU 分流的收益。两条路径的 intra edge count 不同，query recall smoke 接近但不完全相同；因此质量仍以 filtered-search recall 为准，不用局部 topK 重合度下结论。

CPU intra 分流修正：新增 `UNG_SPECIAL_INTRA_COMPLETE_NX`、`UNG_SPECIAL_INTRA_EXACT_TOPK` 和 `UNG_SPECIAL_INTRA_BOUNDED_COMPLETE`。正确方向不是简单提高 complete threshold。全连接 complete 在 `th=1024` 时把 sidecar edges 从 `39.17M` 放大到 `176.48M`；bounded ring 虽然在 `th=2048` 时把 CPU intra 降到 `41.94 s`、edges 控制在 `42.80M`，但 coverage L50/L200 recall 下降到 `0.323 / 0.370`，不可作为默认。distance exact-topK 小块分流在 `th=2048` 时 CPU intra 为 `75.64 s`、overlay `81.73 s`、edges `42.80M`，coverage L50/L200 为 `0.341 / 0.400`，基本接近旧 CPU Vamana `0.344 / 0.401`。当前代码因此采用 CPU default `exact-topK, th=2048`；GPU default 单独调到 `th=128`，即 `530` 个小 block / `59,847` 点走 exact-topK，其余 `1444` blocks / `517,975` 点走 CUDA。

| CPU special intra route | threshold | intra ms | overlay ms | sidecar edges | coverage L50/L200 | 结论 |
|---|---:|---:|---:|---:|---:|---|
| old CPU Vamana | `64` | `160.30 s` | `166.33 s` | `39.17M` | `0.344 / 0.401` | 历史慢 baseline |
| full complete | `1024` | `72.79 s` | `79.08 s` | `176.48M` | 未跑 | 边数爆炸，不宜默认 |
| bounded ring | `2048` | `41.94 s` | `48.00 s` | `42.80M` | `0.323 / 0.370` | 快但质量明显下降 |
| exact-topK | `2048` | `75.64 s` | `81.73 s` | `42.80M` | `0.341 / 0.400` | 当前 CPU 默认候选 |
| GPU Tagore/FastGrnnd | old default `100` | `5.23 s` | `11.36 s` | `40.54M` | 旧 smoke `0.345 / 0.403` | 全部走 CUDA |
| GPU split exact-topK/CUDA | new default `128` | `5.11 s` | `11.24 s` | `40.54M` | `0.344 / 0.403` | 当前推荐 GPU 路径 |

证据目录：`/home/graphdb/fv_runs/special_blocks_20260705/special_intra_complete_sweep_20260706_123142`、`special_intra_bounded_complete_sweep_20260706_125305`、`bounded_th2048_search_smoke_20260706_130755`、`special_intra_exact_topk_sweep_20260706_131012`、`exact_topk_th2048_search_smoke_20260706_131808`、`special_intra_default_route_check_20260706_132011`。

GPU intra 分界线 sweep：在 `UNG_SPECIAL_BLOCK_GPU_INTRA=1` 下显式扫描 `UNG_SPECIAL_INTRA_COMPLETE_NX`。`th=128` 略优于全 CUDA，且 coverage L50/L200 recall 为 `0.344 / 0.403`；`th>=256` 会把过多 block 交给 CPU exact-topK，intra 和 index 均变慢。因此 GPU default 设为 `128`，不是沿用 CPU default 的 `2048`。

GPU inter kernel 效率优化：原 special GPU inter 使用 source-exact CUDA kernel，每个 source query 一个 CTA，默认 `8` warps/query。breakdown 显示 pack/H2D/D2H 都不是瓶颈，旧默认 `kernel_ms≈5.34 s` 占 GPU inter 总时间 `~88%`。这说明性能差距主要来自 kernel 形态和每 query 的 warp 配置，而不是 CPU-GPU 传输。尝试复用现有 TF32 WMMA tile kernel 是负结果：tiles 达到 `3.61M`，kernel `33.53 s`，远慢于 cuda-core `5.33 s`，说明当前 source/segment 太碎，直接套 16-query WMMA tile 不合适。随后扫描 `UNG_SPECIAL_GPU_INTER_WARPS`，发现 `8` warps/query 明显过配；`2` warps/query 最优，kernel 从 `5.35 s` 降到 `1.87 s`，GPU inter 总时间从 `~6.1 s` 降到 `~2.7 s`，coverage L50/L200 recall 仍为 `0.344 / 0.403`。当前默认已改为 `2` warps/query，保留 env override。

| special GPU inter variant | kernel ms | inter total ms | overlay ms | index ms | coverage L50/L200 | 结论 |
|---|---:|---:|---:|---:|---:|---|
| cuda-core, 8 warps old default | `5.33 s` | `6.10 s` | `11.95 s` | `32.37 s` | 旧 smoke `0.344 / 0.403` | warp 过配 |
| TF32 WMMA tile prototype | `33.53 s` | `34.49 s` | `39.70 s` | `60.30 s` | 未继续 | tile 过碎，负结果 |
| cuda-core, 1 warp | `1.81 s` | `2.67 s` | `7.88 s` | `27.83 s` | 未继续 | 接近最优 |
| cuda-core, 2 warps new default | `1.87 s` | `2.71 s` | `7.87 s` | `28.62 s` | `0.344 / 0.403` | 当前默认 |
| cuda-core, 4 warps | `2.71 s` | `3.58 s` | `8.79 s` | `28.40 s` | 未继续 | 较慢 |
| cuda-core, 16 warps | `13.20 s` | `14.09 s` | `19.34 s` | `39.99 s` | 未继续 | 明显过配 |

证据目录：`/home/graphdb/fv_runs/special_blocks_20260705/special_gpu_inter_wmma_ab_20260707_113944`、`special_gpu_inter_warps_sweep_20260707_114325`、`special_gpu_inter_warps2_default_smoke_20260707_114920`.

| GPU split threshold | exact-topK blocks / points | CUDA blocks / points | intra ms | overlay ms | index ms | sidecar edges |
|---:|---:|---:|---:|---:|---:|---:|
| `100` | `0 / 0` | `1974 / 577822` | `5.23 s` | `11.36 s` | `31.17 s` | `40.54M` |
| `128` | `530 / 59847` | `1444 / 517975` | `5.11 s` | `11.24 s` | `29.45 s` | `40.54M` |
| `256` | `1404 / 214846` | `570 / 362976` | `9.52 s` | `15.68 s` | `36.02 s` | `40.54M` |
| `512` | `1769 / 344557` | `205 / 233265` | `10.89 s` | `17.15 s` | `36.03 s` | `41.79M` |
| `1024` | `1906 / 437149` | `68 / 140673` | `19.37 s` | `25.51 s` | `45.45 s` | `42.38M` |
| `2048` | `1956 / 506519` | `18 / 71303` | `31.46 s` | `37.74 s` | `57.36 s` | `42.71M` |

证据目录：`/home/graphdb/fv_runs/special_blocks_20260705/special_gpu_intra_split_sweep_20260706_222042`、`gpu_split_th128_search_smoke_20260706_222800`、`gpu_default_th128_check_20260706_222948`.


No-skip GPU-cross 上界 probe：设置 `UNG_SPECIAL_BLOCK_SKIP_TRIVIAL=0` 后，ordinary cross 恢复 GPU backend，special GPU/GPU index 为 `38.52 s`、cross `1.80 s`、overlay `18.63 s`。该结果不代表默认 skip 语义，但证明当前最大剩余构建差距是 ordinary GPU cross 与 trivial-source skip 语义不兼容，而不是 special inter。证据目录：`/home/graphdb/fv_runs/special_blocks_20260705/special_noskip_gpu_cross_20260706_032906`。

GPU cross source-skip 正式 smoke：在 host-side GPU cross query planning/packing 中跳过 trivial special source groups 后，默认 `skip_trivial=1` 也能走普通 GPU cross。当前 guarded code 的 full build/search 结果为 index `38.65 s`、ordinary cross `~1.78 s`、overlay `18.59 s`、L50/L200 recall `0.345/0.403`；此前 prototype smoke 为 index `37.93 s`、cross `1.67 s`、overlay `18.59 s`；GPU query 数从 no-skip 的 `21,704,410` 降到 `21,663,477`，差值 `40,933` 与旧 CPU skip 统计一致。该结果已通过 ordinary graph source-mask verifier；若要作为论文级最终证据，还可补每 target topK 的 sampled verifier。后续 counter probe 已让 GPU skip path 写出 `special_cross_trivial_skipped_pairs=169` 和 `special_cross_trivial_skipped_query_vectors=40933`，证据目录：`/home/graphdb/fv_runs/special_blocks_20260705/special_gpu_cross_skip_counter_20260706_035327`。初始 build/search smoke 证据目录：`/home/graphdb/fv_runs/special_blocks_20260705/special_gpu_cross_skip_probe_20260706_034209`。query-level smoke 对齐 CPU-skip 与 GPU-source-skip 后，L50/L200 的 `300/300` 个 `(repeat, QueryID)` recall 完全一致，DistCalcs 和 special edge scan 的 median delta 均为 `0`。进一步的 ordinary graph verifier 对比 GPU no-skip 与 GPU source-skip：非 skipped source points 的邻接完全一致（`non_skipped_diff=0`），全部 `7317` 个 skipped source points 发生差异且 skip 图清空普通边；这验证了 GPU source mask 的普通图输出语义。 route guard 已补充：special skip 下仅允许已验证的 regular batched/double-buffer GPU cross route；`source_exact`、`x_streaming`、`cpu_tiny_groups`、`universal` 和 cuVS 组合会 fail closed，避免静默绕过 source mask。valid guard smoke index `37.40 s`，invalid x-streaming smoke 打出明确错误。

Pair-work cap 语义实验：新增 `UNG_SPECIAL_INTER_MAX_PAIR_WORK` 后，cap=5M 跳过 `87` 个高成本 parent-child pair，inter `10.98 s -> 4.69 s`、overlay `18.59 s -> 12.19 s`、index `38.65 s -> 31.27 s`，L50/L200 recall 仍为 `0.345/0.403`；cap=1M 跳过 `359` 个 pair，inter `1.59 s`、overlay `9.07 s`、index `29.23 s`，L50/L200 recall `0.347/0.405`。第一套 100-query coverage workload 的 Lsearch sweep 显示 cap=1M 在 L20-L800 的 mean recall 未下降、median per-query delta 为 `0`，但少数 query/repeat 会退化，最差到 `-0.5`。第二套 query_mix_100_seed456 workload 显示 cap=1M 出现一致小幅 mean recall 损失：L20/L50/L100/L200/L400/L800 delta 分别为 `-0.005/-0.008/-0.007/-0.004/-0.004/-0.004`，且低 L 没有 better cases。因此 cap=1M 不是无损优化，也不是安全默认。补跑 cap=5M 后，第二套 workload 在 L20/L50/L100/L200 recall 完全一致，L400/L800 仅 `-0.001`，最差 per-query delta `-0.1`；cap=5M 对应 index `31.27 s`、inter `4.69 s`、overlay `12.19 s`，是更保守的近似候选。进一步扫描 cap=10M/20M 后，两套 workload 在 L20-L800 均与 no-cap 逐 query recall 完全一致；cap=10M 对应 index `33.36 s`、inter `5.98 s`、overlay `13.47 s`，cap=20M 对应 index `34.63 s`、inter `7.34 s`、overlay `14.83 s`。

后续专门补了 broad-heavy 质量反例，并进一步拆开 cap 与 Tagore iter 的混杂因素。新增 `query_broad2_100_seed901` 是 100 条 query length=2、profile 全 broad 的 workload，matched groups avg/p50/p95 为 `23130.20/5124.0/53875.6`，matched points avg/p50/p95 为 `28026.22/5332.5/73164.8`。在 no-cap index 与 cap10+iter4 index 上复用既有索引做 search-only A/B，cap10+iter4 - no-cap 的逐 `(repeat,Lsearch,QueryID)` recall delta 为：L20 `0.000`，L50 `-0.003`，L100 `-0.003`，L200 `-0.004`，L400 `-0.003`，L800 `-0.012`；最差单点 delta 到 `-0.5`。但这不能全部归因于 cap10，因为 cap10 default-iter index 在同一 workload 上只有 L800 `-0.003`，L20/L200/L400 为 `+0.001`，L50/L100 为 `0`；cap20 也几乎相同。进一步补 no-cap+iter4 index 后，no-cap+iter4 与 no-cap default 在 broad2 上逐 `(repeat,Lsearch,QueryID)` recall 完全一致，所有 L 的 mean/min/max delta 都是 `0`。因此，更准确的结论是：cap10 本身在该 workload 上有轻微高 L 风险；iter4 单独没有暴露风险；cap10+iter4 的额外退化来自 cap 与较低 Tagore iter/图构建近似的交互。同一轮新增 `query_len1_broad_100_seed902` 是单标签 broad-heavy workload，matched points p95 `562880.0`，cap10+iter4 未出现 recall 下降（L20/L50/L100/L800 delta `0`，L200/L400 `+0.001`）。因此风险主要集中在多标签 broad/shallow 组合，而不是所有大候选 query。

进一步把 cap20 也放到同一 broad-heavy workload 上验证。cap20 build 为 index `34.63 s`、inter `7.34 s`、overlay `14.83 s`，比 no-cap `38.65 s` 仍快，但比 cap10 慢。cap20 与 cap10 default 在 broad2 上的质量边界接近：`query_broad2_100_seed901` 上 L20/L50/L100 delta 为 `0`，L200/L400 为 `+0.001`，L800 为 `-0.003`，最差单点 `-0.4`；`query_len1_broad_100_seed902` 上仅 L200 出现 `-0.001`、最差 `-0.1`。因此 cap20 是比 cap10+iter4 更保守的近似点，但并没有证明全局 pair-work cap 可成为 full-quality 默认。全局 cap 的根本问题没有消失：它会裁掉部分 broad/shallow query 可能需要的 heavy parent-child special inter edges；低 Tagore iter 本身单独未在 broad2 上退化，但会和 cap 交互放大质量风险。后续更合理方向是大范围 workload 验证、两层 sidecar 或 query-adaptive heavy-edge 使用，而不是继续把单一 cap 写成安全优化。补充 high-L 对照显示，单纯给 cap=1M 的 broad query 增大 Lsearch 不能恢复主要 regression：qid 87 在 L1200-L3200 仍为 no-cap `1.0`、cap `0.7`，说明缺失的 heavy shallow-parent inter edges 对该类 query 是结构性需要。证据目录：`/home/graphdb/fv_runs/special_blocks_20260705/inter_cap_probe_20260706_043900`、`/home/graphdb/fv_runs/special_blocks_20260705/cap_lsearch_sweep_20260706_045326`、`/home/graphdb/fv_runs/special_blocks_20260705/cap_lsearch_sweep_mix_20260706_050722`、`/home/graphdb/fv_runs/special_blocks_20260705/cap10_extra_quality_20260706_081618`、`/home/graphdb/fv_runs/special_blocks_20260705/nocap_iter4_broad2_20260706_084114`。

离线 query-adaptive heavy-edge 模拟：在 `query_broad2_100_seed901` 上，用已有 no-cap/cap10/cap20 的 query_details 做混合模拟：被规则触发的 query 使用 no-cap recall，其余 query 使用 cap variant recall。这个模拟不代表代码已实现 query-adaptive route，只用于评估方向是否值得实现。结果显示，cap10+iter4 的 L800 退化可以通过 `matched_groups>=20000` 规则完全恢复，该规则只触发 `17/100` 条 query；`matched_points>=20000` 触发 `22/100` 条 query 也能完全恢复。对 cap10 default 和 cap20，`matched_points>=50000` 只触发 `8/100` 条 query 就能恢复 L800 退化。这个结果支持下一步做两层 sidecar：默认使用 capped light sidecar；当 query 的 matched groups/points 表明它是 broad/shallow 重查询时，再启用 heavy sidecar 或 no-cap special inter。证据文件：`/home/graphdb/fv_runs/special_blocks_20260705/cap10_extra_quality_20260706_081618/adaptive_heavy_edge_sim.md`。

两层 sidecar 的成本估算也支持继续实现原型。no-cap special edges 为 `43,826,311` 条，cap10 default 为 `40,568,512` 条，cap20 为 `42,098,996` 条；也就是说 cap10 light sidecar 额外 heavy tier 约 `3.26M` 条边，cap20 heavy tier 约 `1.73M` 条边。CSV artifact 体积从 no-cap 约 `1.1GB` 降到 cap10 约 `950MB`、cap20 约 `986MB`。在 broad2 L800 查询侧，no-cap 与 cap10/cap20 的平均实际扫描差异很小：no-cap 平均 `10,549.9` 条 special edges/query，cap10 default `10,444.5`，cap20 `10,475.3`。如果按 `matched_groups>=20000` 只对 `17%` query 启用 heavy tier，cap10+iter4 的质量恢复只需要摊到所有 query 平均额外扫描约 `194` 条 special edges；对 cap10 default/cap20，`matched_points>=50000` 触发 `8%` query，平均额外扫描约 `103`/`101` 条 special edges。这个量级说明 query-adaptive heavy-edge 的 runtime 成本可能远小于全量 no-cap 构建/存储成本，但仍需真实代码实现后用 wall time 验证。

subset runtime 估算：进一步把 `query_broad2_100_seed901` 按 `matched_groups>=20000` 拆成 `17` 条 triggered query 和 `83` 条 rest query；triggered 子集用 no-cap index 搜索，rest 子集用 cap10+iter4 index 搜索，然后按 query 数加权。该实验仍不是单进程 two-tier 实现，因为两次 search 都独立加载 index，batch 组成也不同；但它给出一个比纯扫描数更接近真实搜索的估计。加权结果在 L20/L50/L100/L200/L400/L800 的 recall 为 `0.328/0.351/0.361/0.400/0.431/0.488`，与 no-cap 全量 broad2 完全一致；对应 weighted time 为 `1128.4/1076.9/1046.9/1032.2/1008.3/1157.6 ms`，没有显示明显 runtime 代价。全量对照中 no-cap L800 为 `1055.9 ms / 0.488`，cap10+iter4 为 `1226.4 ms / 0.476`。因此 two-tier 的质量收益在这个 workload 上明确，真实 wall time 需要代码内单进程 route 实现后再验证。证据目录：`/home/graphdb/fv_runs/special_blocks_20260705/twotier_subset_runtime_20260706_085639`。

单进程 two-tier 原型 smoke：新增 env-gated 原型，不改变默认行为。加载 no-cap `special_edges.csv` 时，如果设置 `UNG_SPECIAL_HEAVY_EDGE_PAIR_WORK=10000000`，会按 parent-child special block pair work 把高成本 inter edges 移入 heavy sidecar；搜索时只有设置 `UNG_SPECIAL_HEAVY_EDGE_SEARCH=1` 且 query 满足 `UNG_SPECIAL_HEAVY_EDGE_MIN_MATCHED_POINTS` 或 `UNG_SPECIAL_HEAVY_EDGE_MIN_ENTRIES` 才扫描 heavy sidecar。该原型目前是 load-time split，不是构建时直接输出两层 sidecar，但已经能验证单进程 query route。`query_broad2_100_seed901` L800 smoke 结果：light-only 为 `1291.38 ms / 0.485 recall`，adaptive-heavy（`MIN_MATCHED_POINTS=20000`）为 `1103.19 ms / 0.488 recall`，恢复到 no-cap recall；`SpecialHeavyEdgesEnabled=66` 行，正好对应 `22` 条触发 query × `3` repeats，`SpecialHeavyEdgesScanned=2052`。证据目录：`/home/graphdb/fv_runs/special_blocks_20260705/twotier_proto_smoke_20260706_091151`。

扩展到 L20-L800 和第二个 workload 后，原型仍然支持这个方向，但也暴露出要继续调规则。证据目录：`/home/graphdb/fv_runs/special_blocks_20260705/twotier_proto_full_20260706_091716`。在 `query_broad2_100_seed901` 上，light-only recall 为 L20/L50/L100/L200/L400/L800 `0.328/0.351/0.361/0.401/0.432/0.485`；adaptive-heavy 为 `0.328/0.351/0.361/0.400/0.431/0.488`，L800 恢复到 no-cap，L200/L400 有 `-0.001` 级别波动。对应 time 分别为 light-only `1325.78/1101.86/1043.08/1097.48/1109.28/1121.09 ms`，adaptive-heavy `1181.76/1141.77/1079.87/1195.09/1067.83/1061.40 ms`，仍主要受 batch 噪声影响。`SpecialHeavyEdgesEnabled=396` 行，对应 `22` 条 query × `6` 个 L × `3` repeats，`SpecialHeavyEdgesScanned=12312`。在 `query_len1_broad_100_seed902` 上，adaptive-heavy 与 light-only recall 完全一致（`0.209/0.232/0.256/0.307/0.327/0.360`），但 `SpecialHeavyEdgesEnabled=144` 行且 `SpecialHeavyEdgesScanned=0`，说明 `MIN_MATCHED_POINTS=20000` 会误触发一些单标签 broad query，不过这些 query 实际没有扫到 heavy sidecar。下一步需要更精细的触发条件，例如结合 `QuerySize>=2` 或 `matched_groups`，避免无效触发。

触发规则修正：新增 `UNG_SPECIAL_HEAVY_EDGE_MIN_QUERY_SIZE`，可以要求 query label 数达到阈值后才允许 heavy sidecar。设置 `MIN_QUERY_SIZE=2`、`MIN_MATCHED_POINTS=20000` 后，L800 smoke 结果为：`query_broad2_100_seed901` `1308.89 ms / 0.488 recall`，`SpecialHeavyEdgesEnabled=66`、`SpecialHeavyEdgesScanned=2052`；`query_len1_broad_100_seed902` `5707.63 ms / 0.360 recall`，`SpecialHeavyEdgesEnabled=0`、`SpecialHeavyEdgesScanned=0`。因此 query-size gate 能保留 broad2 的质量恢复，同时消除 len1 broad 的无效触发。证据目录：`/home/graphdb/fv_runs/special_blocks_20260705/twotier_querysize_gate_20260706_093114`。

四套 workload 扩展验证：使用同一 no-cap index，load-time split `pair_work>10M`，adaptive-heavy 规则为 `MIN_QUERY_SIZE=2`、`MIN_MATCHED_POINTS=20000`，跑 `query_coverage_100`、`query_mix_100_seed456`、`query_broad2_100_seed901`、`query_len1_broad_100_seed902` 的 L20/L50/L100/L200/L400/L800。与已有 no-cap 对照相比，所有可对齐 L 上 recall delta 均为 `0`。触发行为也符合预期：coverage 触发 `20` 条 query（`60` rows），query_mix 触发 `14` 条 query（`42` rows），broad2 触发 `22` 条 query（`66` rows），len1 触发 `0`。heavy scanned 在 coverage/query_mix/broad2 中均为 `2052`，len1 为 `0`。时间仍有明显运行噪声：broad2 adaptive-heavy L800 `1063.29 ms / 0.488` 接近 no-cap `1055.88 ms / 0.488`；query_mix 和 len1 的 adaptive-heavy 时间比旧 no-cap run 更慢，不能把这解释成算法代价，后续需要同进程/交替重复 A/B。证据目录：`/home/graphdb/fv_runs/special_blocks_20260705/twotier_adaptive_four_20260706_093816`。

交替重复 A/B：为降低跨 run 机器状态影响，补充 `query_broad2_100_seed901` L800 的 light/adaptive 交替搜索，每个 case 内部 `num_repeats=1`，外层交替 3 轮。light mean 为 `1839.87 ms / 0.485 recall`，adaptive mean 为 `1349.07 ms / 0.488 recall`；mean query time 为 `743.52 ms` vs `608.32 ms`，mean core time 为 `120.81 ms` vs `98.15 ms`。adaptive 每轮触发 `22` 条 query，扫描 `684` 条 heavy edges。进一步做 paired query-level 分析后，adaptive-light 的 mean delta 为：all queries `-135.20 ms` query time、heavy-triggered queries `-185.79 ms`、non-heavy queries `-120.93 ms`；non-heavy queries 也整体更快，说明这组 timing 仍主要受 run-order/系统噪声影响，不能写成 adaptive 加速。但 paired 分析没有显示 heavy-triggered queries 存在系统性 overhead，同时 recall 从 light 的 `0.485` 恢复到 `0.488`。因此当前可写成“未观察到明显 overhead 并恢复 recall”，不能写成稳定加速。证据目录：`/home/graphdb/fv_runs/special_blocks_20260705/twotier_alternating_broad2_20260706_094843`。

持久化 split-file smoke：新增 `UNG_SPECIAL_HEAVY_EDGE_SAVE_PAIR_WORK`，保存 index 时可直接把 high pair-work inter edges 写入 `special_heavy_edges.csv`，`special_edges.csv` 保留 light sidecar。用 no-cap+iter4/GPU intra/GPU inter 构建 split index 后，light 文件 `special_edges.csv` 为 `949MB`、`40,537,573` 行（含 header），heavy 文件 `special_heavy_edges.csv` 为 `77MB`、`3,257,545` 行（含 header），合计与 no-cap edge count 对齐。构建 index `39.05 s`、overlay `16.77 s`，保存阶段 `47.13 s`，说明 CSV 分文件写入仍有较大 I/O 成本。使用该 split index 搜索 broad2 L800：light-only `1260.70 ms / 0.481 recall`，adaptive-heavy `1245.14 ms / 0.484 recall`，`SpecialHeavyEdgesEnabled=66`、`SpecialHeavyEdgesScanned=2052`。同一 index 的 merged no-split 控制实验把 `special_edges.csv + special_heavy_edges.csv` 重新合并成单文件，其 broad2 L800 也是 `1032.05 ms / 0.484 recall`。因此，持久化 split 文件和 heavy 加载机制没有造成主要质量损失；这次新建 index 与旧 no-cap/load-time-split `0.488` 的差异主要来自重新构建的图质量波动，而不是 split 保存/加载语义。证据目录：`/home/graphdb/fv_runs/special_blocks_20260705/twotier_split_index_20260706_101849` 和 `.../twotier_split_index_20260706_101849_merged_control`。

保存阶段进一步 profiling：补充 `full_save_timing_20260706_104933` 后，完整 `UniNavGraph::save()` 阶段拆分如下：meta/build_time `2.36 s`，base storage `1.83 s`，group metadata `2.29 s`，special blocks `7.79 s`，graph/trie `16.04 s`，LNG sets `4.11 s`，LNG out-neighbors `0.67 s`，vector_attr_graph `0.11 s`，roaring bitmaps `0.28 s`，reordered export `4.70 s`，总 index save `40.18 s`。其中 `save_special_blocks()` 内部为 metadata `2.0 ms`、members `39.5 ms`、children `0.2 ms`、special edge CSV 写入 `7.75 s`，light/heavy 行数分别为 `40,538,086` 和 `3,257,544`。因此，split special edge CSV 是一个明确的 `~7.8 s` 成本，但不是最大项。

进一步细拆 `graph_and_trie` 后，最大保存瓶颈定位到 trie：`fine_save_timing_20260706_105852` 中 `new_to_old_vec_ids=0.58 s`、`trie=14.04 s`、`graph=0.74 s`、`global_graph=0.65 s`、`global_entry_point=0.002 s`，合计 `16.01 s`。文件大小也说明不能只按字节数判断：`trie` 约 `149MB` 却花 `14.04 s`，而 `reordered_vecs.fvecs` 为 `1.7GB` 只花 `4.18~4.70 s`。这说明 trie 的保存格式/递归写出/小对象写出方式很可能是保存阶段最大的单点问题。后续若优化保存阶段，优先级应是：1) 优化或二进制化 trie 保存；2) 二进制/CSR special sidecar；3) LNG text sets；4) 评估是否跳过/延迟 ACORN reordered export。

Trie 保存根因确认与最小修复：`TrieIndex::save()` 原来在每个节点、parent 和 children 行都使用 `std::endl`，这会每行 flush。把这些 `std::endl` 改成 `'\n'` 后，`fine_save_timing_20260706_110910` 中 trie 保存从 `14.04 s` 降到 `6.12 s`，`graph_and_trie` 从 `16.01 s` 降到 `8.06 s`；同次总 save 为 `33.24 s`。这是一个语义不变的低风险修复。该 run 中 special edge CSV 写入波动到 `10.17 s`，说明 special CSV 仍是后续可优化项，但 trie 的 per-line flush 已经不再是最大瓶颈。证据目录：`/home/graphdb/fv_runs/special_blocks_20260705/fine_save_timing_20260706_110910`。

special sidecar 二进制格式收益估计：用临时 C++ converter 把 split index 的 CSV sidecar 转成固定宽度二进制记录（source/target/block/kind）。light sidecar 从 `949MB` 降到 `619MB`，heavy sidecar 从 `77MB` 降到 `50MB`。从 CSV 转二进制因为要解析文本，light/heavy 分别花 `11.55 s` / `1.30 s`；但二进制直接读入只需 `0.67 s` / `0.05 s`。这不是构建时直接写 binary 的最终性能，但说明 binary sidecar 能显著降低加载时间和约三分之一体积。若正式实现，应在内存 edge vector 上直接写 binary/CSR，而不是先写 CSV 再转换。证据文件位于 `/home/graphdb/fv_runs/special_blocks_20260705/twotier_split_index_20260706_101849/index_files/special_edges.bin` 和 `special_heavy_edges.bin`，临时工具为 `/tmp/special_edges_bin_bench.cpp`。

env-gated binary sidecar 原型：新增 `UNG_SPECIAL_EDGE_BINARY=1`，保存时额外写 `special_edges.bin` / `special_heavy_edges.bin`，加载时优先读取 binary，CSV 仍保留用于兼容。`binary_sidecar_build_20260706_112952` 验证了 repo 内直接写 binary 的路径：light CSV/bin 为 `949MB/619MB`，heavy CSV/bin 为 `77MB/50MB`，binary 记录数为 `40,537,951` 和 `3,257,544`。用该 index 以 binary load 搜索 broad2 L800，日志显示 `binary_light_edges=40537951`、`heavy_edges=3257544`，结果为 `1258.61 ms / 0.488 recall`，`SpecialHeavyEdgesEnabled=66`、`SpecialHeavyEdgesScanned=2052`，说明 build-written binary sidecar 可正确加载并恢复 recall。需要注意：CSV+binary 双写时 special block 保存时间为 `11.12 s`，比只写 CSV 的 `~7.8-10.2 s` 不更快。

binary-only 保存模式：新增 `UNG_SPECIAL_EDGE_BINARY_ONLY=1`，只写 `special_edges.bin` / `special_heavy_edges.bin`，不写 special edge CSV。`binary_only_sidecar_build_20260706_113959` 中 special block 保存降到 `1.88 s`，其中 edge 写入 `1.82 s`；二进制 light/heavy 文件为 `619MB/50MB`，记录数 `40,538,062/3,257,544`。这相对 CSV split 的 `~7.8-10.2 s` 是明确收益。用 binary-only index 以 binary load 搜索 broad2 L800，日志显示 `binary_light_edges=40538062`、`heavy_edges=3257544`，结果 `1484.06 ms / 0.484 recall`，与该新建 index 的 split/merged 质量边界一致；质量差异仍来自重建图波动，不是 binary-only 语义。正式方向应转为 binary-only 或 CSR-only，CSV 只保留调试/兼容路径。

binary-only 加载健壮性：为避免 binary-only index 因用户忘记设置 `UNG_SPECIAL_EDGE_BINARY=1` 而静默加载空 sidecar，加载逻辑已改为：如果 `special_edges.csv` 不存在但 `special_edges.bin` 存在，则自动读取 binary light sidecar；heavy sidecar 仍由 `UNG_SPECIAL_HEAVY_EDGE_SEARCH=1` 控制。`binary_only_sidecar_build_20260706_113959` 上不设置 `UNG_SPECIAL_EDGE_BINARY`、只设置 heavy search 规则时，broad2 L800 仍能加载 `binary_light_edges=40538062`、`heavy_edges=3257544`，结果 `1369.34 ms / 0.484 recall`，`SpecialHeavyEdgesEnabled=66`、`SpecialHeavyEdgesScanned=2052`。这消除了 binary-only index 的一个易错运行口径。

跳过 reordered export：新增 `UNG_SKIP_REORDERED_EXPORT=1`，用于不构建/不使用 ACORN index 的 UNG special-block 实验。与 binary-only sidecar 同时启用后，`skip_export_binary_build_20260706_120018` 的保存阶段为：meta/build_time `2.21 s`，base storage `1.78 s`，group metadata `2.35 s`，special blocks `1.49 s`，graph/trie `7.97 s`，LNG sets `4.20 s`，LNG out-neighbors `0.67 s`，vector_attr_graph `0.10 s`，roaring bitmaps `0.28 s`，reordered export `0.0 s`，总 index save `21.06 s`。这相对未优化的 `40.18 s` 和 trie-fix 后的 `33.24 s` 又少了一段。该 index 不生成 `reordered_vecs.fvecs`，因此 ACORN 相关 search route 不可用；但普通 UNG special search 可用，broad2 L800 smoke 为 `1292.28 ms / 0.484 recall`，并正确加载 `binary_light_edges=40538023`、`heavy_edges=3257544`。结论：对当前不使用 ACORN 的 special-block pipeline，skip reordered export 是明确有效的保存优化；若论文或 artifact 需要 ACORN 对照，应关闭该开关。

跳过 LNG 文本集合：新增 `UNG_SKIP_LNG_TEXT_SETS=1`，只跳过文本格式的 `covered_sets`、`lng_descendants_num`、`lng_descendants`，仍保存 `lng_coverage_ratio`、`lng_out_neighbors.dat` 和 roaring bitmap 缓存。加载侧已改为：三个文本文件都存在时走旧路径；若文本文件缺失，则打印提示并依赖 roaring 缓存和 out-neighbors。与 binary-only sidecar 和 skip reordered export 同时启用后，`skip_export_binary_lngskip_build_20260706_121730` 的 `lng_sets_ms` 从上一轮 `4204.6 ms` 降到 `637.6 ms`，总 index save 从 `21.06 s` 降到 `19.10 s`；本轮 meta/build_time 等阶段有噪声变慢，因此不能把 stage 差值简单等同为总 save speedup。文件检查确认未生成 `covered_sets`、`lng_descendants_num`、`lng_descendants`。CPU entry-provider special search smoke 通过：`query_broad2_100_seed901` L800 为 `2068.40 ms / 0.488 recall`，`query_coverage_100` L200 为 `1711.44 ms / 0.403 recall`，日志显示 `LNG text covered_sets/lng_descendants are absent; relying on roaring caches and out_neighbors.`，并正确加载 `binary_light_edges=40538041`、`heavy_edges=3257544`。这些 search 数字只证明加载和 recall 路径可用，不作为查询加速 A/B。重要 caveat：GPU cover-frontier provider 仍直接依赖 `_lng_descendants` 文本结构；在没有重建或替代该结构前，`UNG_SKIP_LNG_TEXT_SETS=1` 只应标为 CPU entry-provider / UNG special search 的保存优化，不能作为默认 index 格式。
GPU/CPU special inter backend 复核：旧 controlled run 中 GPU inter 约 `6.05 s`，CPU inter 约 `8.93-11.0 s`，收益不明显。最新 kernel tuning 后，GPU inter 默认 `2` warps/query，inter total 降到 `~2.7 s`，相对当前 CPU inter `~10-11 s` 为约 `4x`。breakdown 显示 H2D/D2H 只有十几毫秒，主要优化来自 kernel warps 配置：旧 `8` warps/query 过配，`2` warps/query 把 kernel `~5.34 s -> ~1.87 s`。证据目录：`/home/graphdb/fv_runs/special_blocks_20260705/cap10_iter4_cpu_inter_20260706_072252`、`special_gpu_inter_warps_sweep_20260707_114325`、`special_gpu_inter_warps2_default_smoke_20260707_114920`。

Tagore iteration knob：`UNG_TAGORE_ITER=4` 与 cap=10M 组合后，index `33.36 s -> 31.64 s`，special intra `7.48 s -> 5.22 s`，overlay `13.47 s -> 11.28 s`。在 original coverage 和 query_mix 上 L20-L800 逐 query recall 与 iter=10 完全一致；在 single-label workload 上相对 no-cap 也无 per-query recall 下降，但少了 iter=10 在 L400 的少量偶然提升。继续测试 iter=2/3 后发现二者虽略快（iter3 index `31.03 s`、iter2 `31.25 s`），但均出现少量 `-0.1` per-query regression。因此 iter=4 是当前更稳的构建 knob，低于 4 不宜默认。证据目录：`/home/graphdb/fv_runs/special_blocks_20260705/tagore_iter4_cap10_20260706_062128`、`/home/graphdb/fv_runs/special_blocks_20260705/tagore_iter2_cap10_20260706_064227`、`/home/graphdb/fv_runs/special_blocks_20260705/tagore_iter3_cap10_20260706_070059`。 另一个 K 维度负结果：`UNG_TAGORE_K=32` 在 max_degree=32 下会导致 special GPU intra 几乎全部 fallback 并崩溃；当前已加 guard，要求 special GPU intra 下 `UNG_TAGORE_K > max_degree`，因此当前 best 仍使用 `UNG_TAGORE_K=64`。K32 负结果目录：`/home/graphdb/fv_runs/special_blocks_20260705/tagore_k32_iter4_cap10_20260706_073348`；guard smoke：`/home/graphdb/fv_runs/special_blocks_20260705/tagore_k32_guard_invalid2_20260706_074603`。


| item | value |
|---|---:|
| special blocks | `1974` |
| trivial blocks | `36` |
| member groups | `478162` |
| member points | `577822` |
| child block edges | `1957` |
| sidecar special edges | `42431228` |
| intra special edges | `14757710` |
| inter special edges | `27673518` |

special-as-group T=100 的 edge count 未变，仍为 `42,431,228` sidecar edges，其中 intra `14,757,710`、inter `27,673,518`。它改变的是 inter 生成方式，不改变最终每点 topK 数量。

普通图/跨组 source skip 的节省很有限：

| skipped work | value |
|---|---:|
| group graph skipped groups | `36` |
| group graph skipped points | `7317` |
| cross skipped pairs | `169` |
| cross skipped query-vector visits | `40933` |
| additional skipped groups | `36` |
| additional skipped points | `7317` |

对照 baseline cross work：

| metric | baseline | special T100 skip |
|---|---:|---:|
| cross pairs | `3993149` | `3992980` |
| cross query visits | `21704952` | `21664019` |
| cross ms | `103882` | `100780` |

解释：T=100 下平凡 special 数量太少，跳过普通 source work 几乎不能抵消 special overlay 的构建成本。

## 5. Query-Time Breakdown

`search_time_summary.csv` 的 batch wall time：

| case | Lsearch | batch avg ms | recall |
|---|---:|---:|---:|
| baseline plain | 50 | `1508.58` | `0.373` |
| special T100 skip | 50 | `1431.64` | `0.393` |
| baseline plain | 200 | `1560.39` | `0.511` |
| special T100 skip | 200 | `1288.54` | `0.524` |

`query_details` per-query mean：

| case | Lsearch | mean query ms | recall | dist calcs | visited | weighted special | weighted nontrivial |
|---|---:|---:|---:|---:|---:|---:|---:|
| baseline plain | 50 | `556.56` | `0.373` | `1212.65` | `487.70` | `0` | `0` |
| special T100 skip | 50 | `408.58` | `0.393` | `642.59` | `600.73` | `0.8106` | `0.7967` |
| special-as-group T100 skip | 50 | `413.02` | `0.393` | `642.60` | `600.74` | `0.8106` | `0.7967` |
| target-nx<=1000 heuristic T100 skip | 50 | `578.49` | `0.393` | `642.59` | `600.73` | `0.8106` | `0.7967` |
| GPU intra T100 skip | 50 | `509.04` | `0.386` | `635.89` | `594.03` | - | - |
| GPU intra+inter T100 skip | 50 | `625.55` | `0.386` | `635.92` | `594.06` | - | - |
| target-nx<=1000 heuristic T400 skip | 50 | `320.96` | `0.398` | `539.30` | `497.44` | `0.7711` | `0.7673` |
| baseline plain | 200 | `347.84` | `0.511` | `1576.53` | `851.51` | `0` | `0` |
| special T100 skip | 200 | `369.35` | `0.524` | `1231.48` | `1100.07` | `0.8106` | `0.7967` |
| special-as-group T100 skip | 200 | `504.64` | `0.524` | `1231.48` | `1100.07` | `0.8106` | `0.7967` |
| target-nx<=1000 heuristic T100 skip | 200 | `458.85` | `0.524` | `1231.48` | `1100.07` | `0.8106` | `0.7967` |
| GPU intra T100 skip | 200 | `414.81` | `0.517` | `1232.65` | `1101.24` | - | - |
| GPU intra+inter T100 skip | 200 | `449.29` | `0.518` | `1231.37` | `1099.96` | - | - |
| target-nx<=1000 heuristic T400 skip | 200 | `451.28` | `0.528` | `1055.77` | `924.36` | `0.7711` | `0.7673` |

Special free-state counters：

| Lsearch | special edge scans | regular edge scans | free dist calcs | regular dist calcs |
|---:|---:|---:|---:|---:|
| 50 | `243.89` | `562.87` | `380.40` | `945.83` |
| 200 | `1022.51` | `1135.12` | `686.28` | `1139.29` |

解释：

- L50 查询中，special search 降低了 distance calcs，per-query mean latency 有改善。
- L200 查询中，special search 仍减少 distance calcs，但 visited nodes 增加，special/regular edge scan 增多，per-query mean 没有稳定改善。
- special-as-group 重构后，查询语义基本保持；L50 与旧 special 接近，L200 更慢，说明构建方式重构没有自动改善查询路径。target-nx heuristic 改善了构建，但查询 L50/L200 分别为 `578.49 ms` / `458.85 ms`，不是查询正结果。
- recall 略高不能写成质量优势；这是同一 Lsearch 下搜索路径差异带来的结果，需要更大 query/repeat 复验。

## 6. Lsearch / Recall 曲线

固定 Lsearch 的单点 latency 不是 special block 查询价值的主要判断标准；更关键的是 recall 曲线是否左移，也就是达到同等 recall 需要的 Lsearch 是否更小。

Search-only sweep artifact：

```text
/home/graphdb/fv_runs/special_blocks_20260705/query_lsearch_sweep_equal_recall_fine
```

同一 x1-restored index/query/GT，Lsearch sweep 如下：

| Lsearch | baseline recall | special recall | baseline batch ms | special batch ms |
|---:|---:|---:|---:|---:|
| 20 | `0.337` | `0.349` | `1457.53` | `1837.11` |
| 30 | `0.357` | `0.376` | `1495.88` | `1624.37` |
| 40 | `0.359` | `0.378` | `1399.13` | `1418.01` |
| 50 | `0.373` | `0.393` | `1373.03` | `1487.39` |
| 75 | `0.416` | `0.434` | `1454.31` | `1283.80` |
| 100 | `0.425` | `0.445` | `1685.05` | `1545.51` |
| 150 | `0.479` | `0.495` | `1240.72` | `1163.94` |
| 200 | `0.511` | `0.524` | `1505.25` | `1691.58` |
| 300 | `0.532` | `0.546` | `1343.71` | `1466.09` |
| 400 | `0.575` | `0.589` | `1527.13` | `1416.66` |
| 500 | `0.601` | `0.608` | `1201.20` | `1503.99` |
| 750 | `0.637` | `0.643` | `1357.21` | `1264.05` |
| 1000 | `0.683` | `0.674` | `1264.67` | `1506.21` |

按最小网格 Lsearch 看：

| target recall | baseline min L | special min L | 解释 |
|---:|---:|---:|---|
| `0.37` | 50 | 30 | special 低 recall 段左移 |
| `0.425` | 100 | 75 | special 中低 recall 段左移 |
| `0.50` | 200 | 200 | 同一档位，special recall 略高 |
| `0.60` | 500 | 500 | 同一档位，special recall 略高 |
| `0.65` | 1000 | 1000 | 高 recall 段收益基本消失 |

线性插值估计：

| target recall | baseline L | special L | baseline/special |
|---:|---:|---:|---:|
| `0.35` | `26.5` | `20.4` | `1.30x` |
| `0.37` | `47.9` | `27.8` | `1.72x` |
| `0.40` | `65.7` | `54.3` | `1.21x` |
| `0.425` | `100.0` | `69.5` | `1.44x` |
| `0.45` | `123.1` | `105.0` | `1.17x` |
| `0.50` | `182.8` | `158.6` | `1.15x` |
| `0.55` | `341.9` | `309.3` | `1.10x` |
| `0.60` | `496.2` | `457.9` | `1.08x` |
| `0.65` | `820.7` | `806.5` | `1.02x` |

结论：special block 的查询价值更适合写成 recall-curve 左移，而不是固定 Lsearch 下 latency 必然更低。低/中 recall 目标下，special 可以用更小 Lsearch 达到相近 recall；高 recall 目标下收益基本消失。

## 7. Coverage Bucket 观察

T=100 的 weighted special coverage 为 `0.8106`，nontrivial 为 `0.7967`。按 bucket 看，coverage 高不必然更快：

| Lsearch | bucket type | bucket | rows | mean ms | recall | dist calcs | special edges | regular edges |
|---:|---|---|---:|---:|---:|---:|---:|---:|
| 50 | special | [0.00,0.25) | 77 | `425.96` | `0.475` | `467.74` | `100.31` | `441.42` |
| 50 | special | [0.90,1.00] | 12 | `483.62` | `0.150` | `1777.92` | `968.08` | `1187.33` |
| 200 | special | [0.00,0.25) | 77 | `394.63` | `0.609` | `909.68` | `523.44` | `898.96` |
| 200 | special | [0.90,1.00] | 12 | `366.06` | `0.233` | `3163.83` | `3127.67` | `2166.75` |

当前现象：

- 高 coverage query 往往也是 broad query，候选量和边扫描本身更大；不能只看 coverage 推断 speedup。
- nontrivial coverage 与 special coverage 高度接近，但高 nontrivial bucket 仍可能更慢或 recall 更低。
- 后续应按 matched points、entry count、special edge fanout 与 Lsearch 联合分桶，而不是只按 coverage 分桶。

## 8. Timeout / Negative Results

T=1600 skip-only：

```text
/home/graphdb/fv_runs/special_blocks_20260705/amazon_x1_restored_T1600_skip_timeout
```

结果：`timeout 900s`，未完成 build，未产出完整 `build_time.csv`。

已知日志：

- special block metadata 成功：`blocks=131`、`trivial_blocks=0`、`member_points=574760`、`child_block_edges=129`、metadata `2020 ms`。
- 普通 group graph 完成：`2321.63 ms`。
- 普通 cross 完成：`103859.0 ms`。
- timeout 发生在后续阶段，未看到 `[special_edges]` 完成行。

解释：即使阈值升到 1600，当前实现仍未能在 15 分钟内形成完整 special index。考虑到普通 baseline index 约 `120 s`，这已经是构建侧强负结果。

T=100 noskip 因构建时间过长被中断；T=400 target-nx heuristic 已完成，但构建成本更高，不作为正向 Pareto 点。

## 9. 当前论文价值判断

可以写进论文的方向：

- special block 是一种 x1/original 口径下的 label trie 块级查询机制，coverage 高，且需要区分 trivial/nontrivial。
- 当前 full-quality A/B 显示：free-state search 可以减少 distance calcs，并让低/中 recall 段的 recall curve 左移；例如 target recall `0.425` 的插值 Lsearch 从 baseline `100.0` 降到 special `69.5`。
- 当前实现的 special edge overlay 构建成本曾是该方法走向论文主结果的最大障碍。threadfix、GPU source-skip、cap10、Tagore iter4 和 GPU intra/inter 后，T=100 最新 controlled run 的 special overlay 为 `11.21 s`、index 为 `29.59 s`，但仍慢于同环境 no-special `17.21 s`，且 cap10 是近似构建 knob，需要更多 workload 或 query-adaptive heavy-edge 设计支撑后才能进入默认方法。

不能写进论文的正结果：

- 不能写 special block 已带来端到端 index speedup。
- 不能把 coverage 表写成 query speedup 表。
- 不能把 synthetic smoke 时间写成性能证据。
- 不能把 x1-restored-from-x40 说成独立原始数据，除非之后补充真实原始 x1 provenance。

下一步如果继续推进，优先级应是：

1. 保留 `target_nx <= UNG_CPU_TARGET_EXACT_MAX_NX` 作为 CPU heuristic baseline；默认 `1000` 比 per-source Vamana 更合理。
2. 保留 special inter 的 OpenMP threadfix：CPU fallback 前 `omp_set_num_threads(_num_threads)`，parallel loop 显式 `num_threads(_num_threads)`，避免 CPU special intra 把后续 inter 污染成单线程。
3. 保留 `UNG_SPECIAL_BLOCK_GPU_INTRA=1` 作为当前 special build 推荐开关；旧 route counter 证明未修正 CPU baseline 会把 T=100 special blocks 全部送 CPU Vamana，已改为 CPU default exact-topK `th=2048`。当前公平口径下 CPU default intra `75.88 s`，GPU default intra `5.11-5.28 s`，GPU inter default `~2.7 s`，intra stage speedup 约 `14.4x`；旧 `30.24x` 仅作为未修正 CPU Vamana 历史 baseline。no-skip 上界 probe 恢复 GPU cross 后 index 为 `38.52 s`；随后 GPU cross source-skip 在默认 `skip_trivial=1` 下达到 index `37.93 s`，guarded full smoke 为 `38.65 s`；叠加 cap10+iter4 后最新 controlled run 为 `29.59 s`。已完成 ordinary graph source-mask verifier 和 route guard；下一步应围绕 cap10/cap20 的质量边界做更广 workload 验证，或设计两层 sidecar / query-adaptive heavy-edge 使用，确认能否形成正式近似路径。
4. special inter/cross 已有 packed source/target descriptor GPU path；threadfix 后 normal A/B 已量化其边际较小（inter `11.96 s -> 11.12 s`，overlay `19.72 s -> 18.82 s`），后续若普通 cross/backend 或 sidecar fanout 改动需重新量化。
5. 限制或采样 block-local sidecar edge fanout，并对 free-state search 做 query-adaptive route。
