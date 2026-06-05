# FastGrnndCuda Amazon Scale Results

日期: 2026-05-28  
更新: 2026-06-02，补充 Amazon 1% x200 coverage-query diversified light prune，并修正 batched exact kNN 结论：早期低 recall 结果被结构诊断归因为 cross/additional 口径混杂；packed exact-anchor full-quality sweep 已补 L5000。

> 注意：本文前半部分记录 2026-05-28 的 build-scale 历史结果，当时 FastGrnndCuda 对 `nx≈200` 使用旧 heavy prune，因此 x200 结论偏负面。2026-06-01 新增 `UNG_FAST_GRNND_LIGHT_PRUNE_NX=256` 后，x200 端到端结果已经转为性能可用但有 recall gap。旧表不删除，用作“为什么需要自适应剪枝强度”的证据。

## 测试配置

统一使用 `build_mode_switch/apps/build_UNG_index`，线程数 `--num_threads 128`，Amazon float L2，`max_degree=32`，`Lbuild=100`，`alpha=1.2`，`num_cross_edges=6`。

优化版配置:

```bash
UNG_BUILD_PROFILE=custom
UNG_GROUP_GRAPH_IMPL=3
UNG_TAGORE_FALLBACK_IMPL=0
UNG_TAGORE_K=64
UNG_TAGORE_ITER=4
UNG_TAGORE_M=64
UNG_GROUP_GRAPH_COMPLETE_NX=64
UNG_GET_MIN_SUPER_SETS_IMPL=0
UNG_LNG_IMPL=0
UNG_DESCENDANTS_IMPL=0
UNG_COVERAGE_IMPL=1
UNG_COVERAGE_THREADS=128
UNG_CROSS_EDGE_IMPL=1
UNG_ADDITIONAL_EDGES_IMPL=1
UNG_GPU_TOPK_IMPL=3
UNG_FORCE_CUSTOM_KERNEL=1
UNG_GEMM_IMPL=2
UNG_GPU_GLOBAL_MERGE=1
UNG_GPU_GLOBAL_MERGE_DIRECT=1
UNG_GEMM_VERIFY_SAMPLES=0
UNG_COUNT_VALID_PAIRS=0
```

主表默认 `UNG_TAGORE_MIN_GROUP_SIZE=256`，另补测 `128` 作为阈值对照。

注意：上述历史 build-scale 配置中 `UNG_ADDITIONAL_EDGES_IMPL=1` 表示跳过 additional edges，只能用于阶段性能拆分和早期工程对比。本文后续标为 full-quality 或涉及 end-to-end recall 的结论，必须使用 `UNG_ADDITIONAL_EDGES_IMPL=0`，即 CPU Vamana additional edges。

说明:

- `group ms` 是组内图构建阶段。
- `cross ms` 是 cross-group edges 阶段。
- baseline 采用当前可比的 CPU Vamana 组内图 + GPU fused cross-edge 配置；`1%x100 CPU exact` 单独列出，用来对比 CPU 暴力 cross-edge 版本。
- 最快 FastGrnndCuda fallback 使用 complete graph 替代小组 CPU Vamana，速度结果可比，但 recall 仍需要端到端查询验证。
- 本机没有 `gpulock`，本轮在 GPU0 空闲时顺序运行，`CUDA_VISIBLE_DEVICES=0`。

## 按 x 放大测试

| 数据集 | 点数 | 组数 | 方法 | min group | group ms | cross ms | Index ms |
|---|---:|---:|---|---:|---:|---:|---:|
| Amazon 1% x10 | 60,240 | 5,666 | FastGrnndCuda + complete fallback | 256 | 50.2 | 565.6 | 891 |
| Amazon 1% x40 | 240,960 | 5,666 | FastGrnndCuda + complete fallback | 256 | 849.6 | 505.3 | 2,442 |
| Amazon 1% x100 | 602,400 | 5,666 | FastGrnndCuda + complete fallback | 256 | 4,152.7 | 1,351.0 | 8,323 |
| Amazon 1% x100 | 602,400 | 5,666 | FastGrnndCuda + complete fallback | 128 | 4,533.4 | 1,257.4 | 8,179 |
| Amazon 1% x200 | 1,204,800 | 5,666 | FastGrnndCuda + complete fallback | 256 | 20,021.6 | 2,788.9 | 26,642 |
| Amazon 1% x200 | 1,204,800 | 5,666 | FastGrnndCuda + complete fallback | 128 | 26,068.5 | 2,431.9 | 32,441 |
| Amazon 1% x400 | 2,409,600 | 5,666 | FastGrnndCuda + complete fallback | 256 | 31,074.6 | 6,722.0 | 44,713 |
| Amazon 1% x400 | 2,409,600 | 5,666 | FastGrnndCuda + complete fallback | 128 | 31,963.4 | 5,253.6 | 45,288 |

历史结论:

- `x10` 基本没有大组，组内图全走 complete fallback，Index time 主要由 cross-edge GPU 调用和固定开销决定。
- `x40` 只有 16 个组进入 FastGrnndCuda，整体和 CPU Vamana 组内图差异很小。
- `x100` 用 `min=128` 的整体 Index time 略优，但组内图本身仍慢于 CPU exact 测试里的 CPU Vamana 组内图。
- `x200` 不适合把阈值降到 128；所有组进入旧 FastGrnndCuda heavy prune 后 prune 太慢，整体变差。该问题已被 2026-06-01 的 light prune 部分缓解，见下文。
- `x400` 所有组都足够大，阈值 128/256 都会走 FastGrnndCuda；`min=256` 本轮略快。

## 按采样比例测试

| 数据集 | 点数 | 组数 | 方法 | group ms | cross ms | Index ms |
|---|---:|---:|---|---:|---:|---:|
| Amazon 1% x40 | 240,960 | 5,666 | FastGrnndCuda + complete fallback | 849.6 | 505.3 | 2,442 |
| Amazon 10% x40 | 2,409,800 | 53,840 | FastGrnndCuda + complete fallback | 7,225.2 | 6,201.2 | 23,207 |

`10%x40` 比 `1%x40` 点数约 10 倍、组数约 9.5 倍。FastGrnndCuda 直接处理的大组数从 16 增至 219，小组 complete fallback 仍然占组内图主要时间。

## Speedup

| 数据集 | baseline | baseline Index ms | 最优优化版 | 优化 Index ms | Index speedup | group speedup |
|---|---|---:|---|---:|---:|---:|
| Amazon 1% x10 | old CPU sanity | 3,626 | opt256 | 891 | 4.07x | 4.94x |
| Amazon 1% x40 | CPU Vamana group + GPU fused cross | 2,484 | opt256 | 2,442 | 1.02x | 1.05x |
| Amazon 1% x100 | CPU Vamana group + GPU fused cross | 16,824 | opt128 | 8,179 | 2.06x | 2.55x |
| Amazon 1% x200 | CPU Vamana group + GPU fused cross | 34,603 | opt256 | 26,642 | 1.30x | 1.31x |
| Amazon 1% x400 | CPU Vamana group + GPU fused cross | 70,616 | opt256 | 44,713 | 1.58x | 1.74x |
| Amazon 10% x40 | CPU Vamana group + GPU fused cross | 49,144 | opt256 | 23,207 | 2.12x | 4.52x |

补充对比:

| 数据集 | 方法 | group ms | cross ms | Index ms | 相对优化版 |
|---|---|---:|---:|---:|---:|
| Amazon 1% x100 | CPU exact cross-edge | 3,266.2 | 2,389.8 | 6,995 | opt128 慢 1.17x |
| Amazon 1% x100 | CPU Vamana cross-edge | 3,266.3 | 18,735.8 | 23,150 | opt128 快 2.83x |

因此，若 baseline 是 CPU Vamana cross-edge，GPU fused cross-edge 明显有效；若 baseline 是 CPU exact cross-edge，`1%x100` 上当前 FastGrnndCuda 组内图还不占优。

## FastGrnndCuda 组成

| 数据集 | min group | Tagore groups | complete fallback groups | complete fallback points | direct build ms | fallback ms | GNN ms | prune ms |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Amazon 1% x10 | 256 | 0 | 5,666 | 60,240 | 0.0 | 50.1 | 0.0 | 0.0 |
| Amazon 1% x40 | 256 | 16 | 5,650 | 234,360 | 279.2 | 566.6 | 14.8 | 39.1 |
| Amazon 1% x100 | 256 | 68 | 5,598 | 565,800 | 544.8 | 3,595.2 | 74.0 | 156.5 |
| Amazon 1% x100 | 128 | 128 | 5,538 | 553,800 | 689.4 | 3,828.3 | 105.5 | 264.1 |

## Amazon 1% x100 Full-Quality Recall Check

2026-06-02 补了 x100 coverage-query repeat=3 A/B，用于回答“x100 build-scale 加速是否破坏最终 search recall”。该实验使用新生成的 `/tmp/fv_amazon_1pct_x100_jitter_nonempty/query_coverage_1000`，GT 为 `/tmp/fv_amazon_1pct_x100_jitter_nonempty/query_coverage_1000_gt/gt_K10_containment.bin`。Query profile 为 medium/broad/narrow `477/331/192`，matched groups avg/p50/p95 为 `311.50/39.5/1095.0`。

| Variant | Index | Group | Cross | L100 | L500 | L1000 |
|---|---:|---:|---:|---:|---:|---:|
| CPU Vamana group + GPU fused cross + CPU additional_edges | `6863.47` | `3303.26` | `1732.42` | `0.825` | `0.868` | `0.890667` |
| FastGrnndCuda + CPU fallback + GPU fused cross + CPU additional_edges | `6370.67` | `3216.21` | `1871.51` | `0.828` | `0.871` | `0.896` |

结论：x100 full-quality repeat=3 下 FastGrnndCuda 没有降低 recall，Index 约 `1.08x`，但 group graph 只约 `1.03x`。因此 x100 应写成质量闭环证据，而不是强 group graph 加速证据；历史 `6630 ms` build-scale 点仍主要用于 cross-edge / end-to-end build 性能展示。
| Amazon 1% x200 | 256 | 128 | 5,538 | 1,107,600 | 1,208.7 | 18,780.2 | 175.9 | 555.4 |
| Amazon 1% x200 | 128 | 5,666 | 0 | 0 | 25,741.2 | 9.4 | 3,258.9 | 18,484.9 |
| Amazon 1% x400 | 256 | 5,666 | 0 | 0 | 30,612.7 | 11.6 | 4,940.4 | 19,943.7 |
| Amazon 1% x400 | 128 | 5,666 | 0 | 0 | 31,482.6 | 18.3 | 5,010.2 | 20,204.4 |
| Amazon 10% x40 | 256 | 219 | 53,621 | 2,213,600 | 1,577.4 | 5,579.9 | 346.9 | 631.5 |

历史关键瓶颈:

- complete fallback 对 `x200` 这种大量 128-255 小组的规模不友好，`min=256` 下 fallback 花 18.8s。
- 降到 `min=128` 会让所有 `x200` 组进 FastGrnndCuda，但 GPU prune 直接变成 18.5s，仍然不划算。
- `x400` 的组内图全部由 FastGrnndCuda 处理，prune 是最大单项开销，约 19-20s。

## 2026-06-01 补充: x200 Light Prune

为了解决 `nx≈200` 上旧 heavy prune 无法摊销的问题，新增：

```bash
UNG_FAST_GRNND_LIGHT_PRUNE_NX=256
```

该路径对中等组直接保留 GNN-Descent 近邻并做轻量去重，跳过 sampled reverse + RNG occlusion 重剪枝。

Amazon 1% x200 coverage-query 端到端 A/B：

| Variant | Index ms | group ms | cross ms | L1000 recall | L5000 recall |
|---|---:|---:|---:|---:|---:|
| CPU Vamana group | 17,596.8 | 8,511.72 | 4,601.91 | 0.8690 | 0.9080 |
| FastGrnndCuda old heavy prune | 38,273.7 | 26,620.6 | 6,055.72 | 0.8662 | - |
| FastGrnndCuda top32 light prune | 14,211.8 | 6,016.04 | 4,926.37 | 0.8619 | 0.9029 |
| FastGrnndCuda diversified light prune | 16,478.5 | 6,706.36 | 4,857.33 | 0.8656 | 0.9056 |
| old batched exact kNN, confounded | 12,595.9 | 4,356.18 | 4,120.75 | 0.8159 | 0.8199 |
| packed exact-anchor router, nx4096 | 11,043.5 | 3,827.28 | 4,868.15 | 0.8690 | 0.9080 |

FastGrnndCuda 组成：

| Variant | direct build | GNN | prune | D2H |
|---|---:|---:|---:|---:|
| old heavy prune | 26,282 | 3,551.54 | 19,218.3 | 413.047 |
| top32 light prune | 5,746.81 | 3,198.35 | 712.392 | 134.631 |
| diversified light prune | 6,365.69 | 3,229.56 | 759.66 | 161.408 |
| batched exact kNN, pure local | 3,950.64 | 674.693 | 0 | 29.8523 |

结论：

- top32 light prune 将 x200 prune 加速约 `27.0x`，group graph 加速约 `4.42x`，Index 加速约 `2.69x`，相对旧 heavy prune。
- diversified light prune 相对 CPU Vamana group graph `1.27x`、Index `1.07x` 更快，L5000 recall gap 缩到 `0.0024`。
- old batched exact kNN 的低 recall 不能再作为组内 exact 图不可行证据：后续诊断显示它与新 packed exact 的组内结构一致，差异来自 cross edges 缺失。packed exact-anchor 在 full-quality 口径下 x200 L5000 已达 `0.9080`，因此下一步是验证该 router 在 x100、10%x40 和真实多标签上的泛化边界。

## 日志位置

```text
/home/graphdb/FilterVectorBenchResults/onchip_rewrite_20260527/x1p10_fastgrnnd_hybrid256_complete_fallback_dimfix/run.log
/home/graphdb/FilterVectorBenchResults/onchip_rewrite_20260527/x1p40_fastgrnnd_hybrid256_complete_fallback_dimfix/run.log
/home/graphdb/FilterVectorBenchResults/onchip_rewrite_20260527/x1p100_fastgrnnd_hybrid256_complete_fallback_dimfix/run.log
/home/graphdb/FilterVectorBenchResults/onchip_rewrite_20260527/x1p100_fastgrnnd_hybrid128_complete_fallback_dimfix/run.log
/home/graphdb/FilterVectorBenchResults/onchip_rewrite_20260527/x1p200_fastgrnnd_hybrid256_complete_fallback_dimfix/run.log
/home/graphdb/FilterVectorBenchResults/onchip_rewrite_20260527/x1p200_fastgrnnd_hybrid128_complete_fallback_dimfix/run.log
/home/graphdb/FilterVectorBenchResults/onchip_rewrite_20260527/x1p400_fastgrnnd_hybrid256_complete_fallback_dimfix/run.log
/home/graphdb/FilterVectorBenchResults/onchip_rewrite_20260527/x1p400_fastgrnnd_hybrid128_complete_fallback_dimfix/run.log
/home/graphdb/FilterVectorBenchResults/onchip_rewrite_20260527/x10p40_fastgrnnd_hybrid256_complete_fallback_dimfix/run.log
```
