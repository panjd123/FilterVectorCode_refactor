# Amazon 1% x400 Reverse-Tail A/B Runbook

本文记录 x400 上验证 `light reverse-tail prune` 的可复现实验流程。该实验专门回答 reviewer 对 group graph 普遍替代的质疑：

> light prune 能加速但 recall drop 达 `0.0081`；是否能用低成本 reverse-tail 多样性把质量补回来，而不退回 heavy prune？

## 当前证据

| Variant | Index | Group | Prune | L1000 | L5000 | 结论 |
|---|---:|---:|---:|---:|---:|---|
| CPU Vamana | `37807.6 ms` | `17537.3 ms` | - | `0.9460` | `0.9660` | 质量 baseline |
| heavy prune | `52552.2 ms` | `29192.3 ms` | `19977.7 ms` | `0.9443` | `0.9643` | 质量接近但太慢 |
| light512 head24 | `30826.5 ms` | `10362.9 ms` | `1022.45 ms` | `0.9379` | `0.9579` | 快，但 drop `0.0081` |
| light512 head28 | `30897.7 ms` | `10276.6 ms` | `1001.65 ms` | `0.9323` | `0.9523` | 更差，说明不能简单增大近邻头部 |

代码已新增三类待验证路径：

```bash
UNG_FAST_GRNND_LIGHT_REVERSE_CAP
UNG_FAST_GRNND_LIGHT_REVERSE_SLOTS
UNG_FAST_GRNND_LIGHT_REVERSE_FORWARD_CAP
UNG_FAST_GRNND_REPAIR_DEGREE
UNG_TAGORE_COMPACT_D2H
```

默认 `UNG_FAST_GRNND_LIGHT_REVERSE_CAP=0`，保持原 light prune 行为。
默认 `UNG_FAST_GRNND_REPAIR_DEGREE=1`，在 light prune 末尾用确定性环形边补足低出度点；默认 `UNG_TAGORE_COMPACT_D2H=1`，把 GPU 图从 `k` 宽压缩为 `final_degree+1` 宽再写回 host。

## 运行前检查

当前机器没有 `gpulock`。为了避免被其他 GPU 任务污染，脚本默认要求 GPU utilization 为 `0`、没有 compute app，且 `memory.used <= GPU_IDLE_MAX_MEMORY_MB`。默认 `GPU_IDLE_MAX_MEMORY_MB=1024` MiB。

```bash
nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader
```

如果 GPU 忙，不要跑性能表。等待空闲后再运行。

注意：只看 utilization 不够。当前曾出现过 `utilization.gpu=0%` 但显存几乎满、仍有 `./build/bin/Test` compute app 的状态，这种情况下脚本会拒绝运行，避免 OOM 或污染结果。若只是 dry-run 检查命令，可设置 `DRY_RUN=1`；若明确要忽略 guard，设置 `WAIT_GPU_IDLE=0`。

## 推荐命令

默认跑三个 reviewer-facing group-graph 变体：

```bash
scripts/benchmarks/run_x400_reverse_tail_ab.sh
```

默认变体如下：

| 目录 | 目的 | 关键环境变量 |
|---|---|---|
| `light512_norepair` | 复现历史 light512 行为，作为 repair 的公平对照 | `UNG_FAST_GRNND_REPAIR_DEGREE=0` |
| `light512_repair` | 测 degree repair 是否修复 low-degree / weak-component 问题 | `UNG_FAST_GRNND_REPAIR_DEGREE=1` |
| `light512_reverse_repair` | 测 reverse-tail + repair 是否进一步补回 recall | `UNG_FAST_GRNND_LIGHT_REVERSE_CAP/SLOTS/FORWARD_CAP`, `UNG_FAST_GRNND_REPAIR_DEGREE=1` |

同时重跑 CPU Vamana baseline：

```bash
RUN_CPU=1 scripts/benchmarks/run_x400_reverse_tail_ab.sh
```

修改 reverse-tail 参数：

```bash
LIGHT_REVERSE_CAP=8 \
LIGHT_REVERSE_SLOTS=4 \
LIGHT_REVERSE_FORWARD_CAP=24 \
scripts/benchmarks/run_x400_reverse_tail_ab.sh
```

只跑当前默认 repair 路径，不重跑 no-repair / reverse-tail：

```bash
RUN_REPAIR_ABLATION=0 RUN_REVERSE_TAIL=0 \
scripts/benchmarks/run_x400_reverse_tail_ab.sh
```

增加 compact-D2H ablation，量化 D2H/fill 外围收益：

```bash
RUN_COMPACT_ABLATION=1 \
scripts/benchmarks/run_x400_reverse_tail_ab.sh
```

当前已完成一次 compact-D2H ablation：`/home/graphdb/fv_runs/x400_compact_ab_20260602_080045`。结论是负/不稳定：compact-on `Index=39759.1 ms`、`D2H=294.133 ms`、`fill=249.144 ms`，compact-off `Index=33019.7 ms`、`D2H=292.299 ms`、`fill=70.4478 ms`；另一次 compact-on build-only rerun `/home/graphdb/fv_runs/x400_compact_on_buildonly_20260602_080513` 仍慢于 compact-off。因此 compact-D2H 只能作为 overhead ablation，不应写成 x400 主加速来源或图质量贡献。

构建完成后自动扫描保存图并生成综合表：

```bash
RUN_GRAPH_DIAG=1 \
scripts/benchmarks/run_x400_reverse_tail_ab.sh
```

Dry-run 检查路径和环境变量：

```bash
WAIT_GPU_IDLE=0 DRY_RUN=1 \
OUTDIR=/tmp/fv_x400_reverse_tail_ab_dryrun \
scripts/benchmarks/run_x400_reverse_tail_ab.sh
```

## 输出

脚本会在输出目录生成：

```text
light512/summary.csv
light512_norepair/summary.csv
light512_repair/summary.csv
light512_reverse_repair/summary.csv
search_recall_summary.csv
x400_ab_summary.csv
x400_ab_summary.md
```

每个 variant 的 build env 位于：

```text
<variant>/fastgrnnd_cpu_fallback/others/env
```

必须确认 `light512_repair` 中包含：

```text
UNG_FAST_GRNND_REPAIR_DEGREE=1
UNG_TAGORE_COMPACT_D2H=1
```

必须确认 `light512_reverse_repair` 中包含：

```text
UNG_FAST_GRNND_LIGHT_REVERSE_CAP=8
UNG_FAST_GRNND_LIGHT_REVERSE_SLOTS=4
UNG_FAST_GRNND_LIGHT_REVERSE_FORWARD_CAP=24
UNG_FAST_GRNND_REPAIR_DEGREE=1
UNG_TAGORE_COMPACT_D2H=1
```

## 通过标准

论文主张不能只看 Index time。建议判据：

| 判据 | 目标 |
|---|---:|
| Index speedup vs CPU | `>= 1.15x` |
| Group graph speedup vs CPU | `>= 1.4x` |
| L5000 recall drop vs CPU | `<= 0.003` |
| L1000 recall drop vs CPU | `<= 0.004` |
| P95/P99 latency | 不显著恶化 |

额外结构判据：

| 判据 | 目标 |
|---|---:|
| zero intra-degree ratio | 接近 CPU Vamana，不能高于历史 light512 |
| low intra-degree `<=4` ratio | 明显低于历史 light512 的 `0.00929781` |
| largest WCC < 0.9 的 group 数 | 明显低于历史 light512 的 `39` |

建议在 build/search 结束后运行：

```bash
python3 tools/benchmarks/diagnose_ung_graph_structure.py \
  <variant>/fastgrnnd_cpu_fallback/index_files \
  --out <variant>/graph_diag \
  --reciprocal-limit 0
```

如果已经设置 `RUN_GRAPH_DIAG=1`，脚本会自动对每个完成的 case 生成 `graph_diag`，并调用：

```bash
python3 tools/benchmarks/summarize_x400_ab.py <out_root> \
  --csv <out_root>/x400_ab_summary.csv \
  --md <out_root>/x400_ab_summary.md
```

`x400_ab_summary.md` 是推荐贴进论文草稿或 reviewer 记录的第一版综合表，它同时包含 build time、L1000/L5000 recall、P95/P99 query latency，以及 zero/low intra-degree 和 WCC 指标。

如果 repair 显著改善结构但 recall 改善有限，应写成“结构修复不足以恢复导航质量”，下一步需要 budgeted occlusion。

如果 reverse-tail 达不到 recall 目标，但比 `light512_repair` 有改善，应写成“partial positive”：低成本 reverse-tail 有帮助，但还需要 budgeted occlusion 或 router。

如果 reverse-tail 质量不改善，应写成“negative ablation”：只加入入边候选不足以恢复 Vamana-style navigability，需要真正的低预算 occlusion 或混合 strong/light 路由。

## 论文写法

成功时可写：

> Light reverse-tail pruning recovers most of the x400 recall loss while retaining the light-prune build-time advantage, indicating that the missing component is low-cost tail diversity rather than more nearest neighbors.

失败时应写：

> The x400 head and reverse-tail ablations show that pruning quality is not recovered by simply reallocating edge slots. This motivates a quality-aware router or budgeted occlusion path, and we do not claim a lossless universal replacement for CPU Vamana.
