# Best Full-Quality 版本记录

本文维护当前建议作为“最佳版本”的可复现实验配置。它不是单个 kernel 的 microbenchmark，而是完整 `build_UNG_index + search_UNG_index` 的 full-quality A/B：构建后用 filtered-search recall 检查查询质量。

## 当前 Best Variant

脚本入口：

```bash
scripts/benchmarks/run_end_to_end_recall_ab.sh
```

变体名：

```bash
VARIANTS="cpu_vamana_group best_full_quality"
```

`best_full_quality` 已登记在脚本 `variant_env()` 中，当前包含：

```bash
UNG_GROUP_GRAPH_IMPL=3
UNG_TAGORE_MIN_GROUP_SIZE=128
UNG_TAGORE_K=64
UNG_TAGORE_ITER=4
UNG_TAGORE_M=64
UNG_FAST_GRNND_LIGHT_PRUNE_NX=256
UNG_FAST_GRNND_LIGHT_HEAD=16
UNG_FAST_GRNND_REPAIR_DEGREE=1
UNG_FAST_GRNND_BATCH_EXACT_NX=4096
UNG_FAST_EXACT_DIRECT_H2D=1
UNG_FAST_EXACT_DEVICE_LOOKUP=1
UNG_FAST_EXACT_ANCHOR_TAIL=1
UNG_FAST_EXACT_ANCHOR_SLOTS=4
UNG_FAST_EXACT_BIDIR_ANCHOR=1
UNG_FAST_EXACT_REVERSE_CAP=4
UNG_FAST_EXACT_REVERSE_SLOTS=4
UNG_FAST_EXACT_REVERSE_FORWARD_CAP=16
UNG_GET_MIN_SUPER_SETS_IMPL=0
UNG_LNG_IMPL=0
UNG_DESCENDANTS_IMPL=0
UNG_COVERAGE_IMPL=1
UNG_CROSS_EDGE_IMPL=1
UNG_ADDITIONAL_EDGES_IMPL=2
UNG_ADDITIONAL_DIRECT_APPEND=0
UNG_GPU_TOPK_IMPL=3
UNG_CROSS_EDGE_GPU_STRICT=1
UNG_TAGORE_COMPACT_D2H=1
UNG_TAGORE_FILL_THREADS=16
```

解释：

- group graph 使用当前 exact-anchor + bidirectional anchor + reverse slots 路线。
- cross-edge 使用 GPU batched fused topK。
- cross-edge SearchQueue 输出使用 lazy reserve：只对实际写回的 query id 分配 topK 容量，避免为全部 120 万点预先 `reserve()`。
- additional_edges 使用 CPU exact scan 的 materialized 路径；不要打开 `UNG_ADDITIONAL_DIRECT_APPEND=1`，该路径实测会因 per-edge lock 变慢。
- query entry group 本轮仍使用 `cpu_min_super_sets`，没有把 ELS GPU provider 计入 best 端到端 build 结果。

## 最新 Full-Quality 结果

当前包含 lazy SearchQueue reserve 的最新 artifact：

```text
/home/graphdb/fv_runs/cross_lazy_reserve_best_20260702_181842
```

| variant | index ms | group ms | cross ms | L20 recall | L1000 recall | L5000 recall |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `best_full_quality` lazy reserve | 13444.2 | 4844.47 | 2842.36 | 0.814200 | 0.871 | 0.911 |

说明：该单次复测的 group graph 时间有明显波动，因此不应用它更新 group graph claim；它用于验证 cross 阶段 lazy reserve 优化。cross 阶段从前一版 `4186.91 ms` 降到 `2842.36 ms`。其中 `output_storage(ms)` 从诊断复测中的 `1909.9 ms` 降到 `15.8 ms`。

前一版完整 CPU-vs-best A/B artifact：

```text
/home/graphdb/fv_runs/best_full_quality_e2e_20260702_163334
```

前一版完整 A/B 结果：

```text
GPU: NVIDIA RTX A6000
dataset: /home/graphdb/FilterVectorBenchData/fv_amazon_1pct_x200_jitter_nonempty
query: query_coverage_1000
K=10
Lsearch=20 50 100 200 500 1000 2000 5000
```

硬件和数据：

```text
GPU: NVIDIA RTX A6000
dataset: /home/graphdb/FilterVectorBenchData/fv_amazon_1pct_x200_jitter_nonempty
query: query_coverage_1000
K=10
Lsearch=20 50 100 200 500 1000 2000 5000
```

| variant | index ms | group ms | cross ms | L20 recall | L1000 recall | L5000 recall |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `cpu_vamana_group` | 14077.4 | 8119.73 | 2950.86 | 0.816000 | 0.871 | 0.911 |
| `best_full_quality` | 8624.82 | 1837.34 | 4186.91 | 0.814467 | 0.871 | 0.911 |

端到端构建加速比：

```text
14077.4 / 8624.82 = 1.6328x
```

组内图构建加速比：

```text
8119.73 / 1837.34 = 4.4193x
```

## Cross 阶段 CPU Baseline 与当前结果

容易混淆的口径：

- `cpu_vamana_group` 在 `best_full_quality_e2e_20260702_163334` 中只表示 group graph 用 CPU Vamana；它的 `cross_ms=2950.86` 仍然是 GPU cross-edge，不是 CPU cross baseline。
- 真正 CPU Vamana cross baseline 需要 `UNG_CROSS_EDGE_IMPL=0`。
- 强 CPU exact-scan cross baseline 需要 `UNG_CROSS_EDGE_IMPL=3`。

实测：

| cross 口径 | artifact | cross ms | 说明 |
| --- | --- | ---: | --- |
| CPU Vamana cross | `/home/graphdb/fv_runs/cpu_vamana_cross_baseline_20260702_182210` | 43365.3 | `UNG_CROSS_EDGE_IMPL=0` |
| CPU exact-scan cross | `/home/graphdb/fv_runs/cross_writeback_probe_fixed2_20260702_174442/cpu_vamana_cross_exact` | 37911.8 | `UNG_CROSS_EDGE_IMPL=3` |
| GPU best before lazy reserve | `/home/graphdb/fv_runs/best_full_quality_e2e_20260702_163334` | 4186.91 | SearchQueue eager reserve |
| GPU best after lazy reserve | `/home/graphdb/fv_runs/cross_lazy_reserve_best_20260702_181842` | 2842.36 | SearchQueue lazy reserve |

相对真正 CPU Vamana cross，当前 GPU cross 加速比：

```text
43365.3 / 2842.36 = 15.256x
```

相对强 CPU exact-scan cross，当前 GPU cross 加速比：

```text
37911.8 / 2842.36 = 13.338x
```

因此，按“CPU cross 阶段”定义，当前已超过 2x 目标。若拿 `cpu_vamana_group` A/B 中的 `2950.86 ms` 做比较，那不是 CPU cross baseline，而是另一条 GPU cross 配置的同阶段结果；当前 lazy reserve 版本为 `2842.36 ms`，只比它略快。

## 质量边界

`best_full_quality` 在 L1000/L5000 与 CPU baseline 对齐；L20 低 `0.001533`。因此当前可以写：

```text
在 Amazon 1% x200 coverage-query full-quality 构建口径下，best_full_quality 将 index build 从 14.08s 降到 8.62s，端到端构建加速 1.63x；高 L recall 与 CPU baseline 对齐，低 L 有约 0.0015 的小幅差距。
```

不应写：

```text
best_full_quality 与 CPU Vamana 完全等价。
所有 Lsearch 下 recall 完全一致。
端到端 4.4x。
```

`4.42x` 只对应 group graph 阶段，不是完整 Index time。

## 当前剩余瓶颈

lazy reserve 后 best 中 `build_cross_edges_time=2842.36 ms`。下一步优化重点应从 SearchQueue 输出边界转向：

- `prepare_all` H2D / host-register 波动；
- GPU kernel 路径本身；
- additional_edges exact scan 的进一步批处理或 GPU 化；
- query entry group provider 的 production 端到端启用和评估。
