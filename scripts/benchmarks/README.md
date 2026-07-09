# Benchmark scripts for UNG optimization evidence

This directory provides runnable entrypoints for the current UNG optimization
reports and paper evidence matrix. The main reviewer-facing documents are:

```text
docs/reports/CURRENT_METHODS_EFFECTS_AND_GAPS_CN.md
docs/papers/SUBMISSION_READINESS_STATUS_CN.md
docs/papers/EVIDENCE_MATRIX_CN.md
docs/reports/TECHNICAL_REPORT_OPTIMIZATION_SPEEDUP_CN.md
```

Older CelebA / SIFT30 / Tagore scripts are kept for provenance and ablation,
but the current paper route is:

```text
cross-edge: UNG_UNIVERSAL_GPU=1 universal flat double-buffer
group graph: workload-aware router with CPU/bounded fallback, packed exact-anchor, and FastGrnndCuda/reverse-tail
full-quality search: UNG_ADDITIONAL_EDGES_IMPL=0
```

Here `full-quality` means the built index keeps the current complete UNG
semantics, especially CPU Vamana `additional_edges`, and is then evaluated by
filtered-search recall against brute-force GT. Local topK overlap is only a
diagnostic for kernels or group-local graph shape; it is not a substitute for
end-to-end query quality.

## Common outputs

By default scripts write to:

```text
/home/graphdb/FilterVectorResultsRefactor/organized_benchmarks/<case>_<timestamp>
```

Override with:

```bash
RESULTS_ROOT=/path/to/results scripts/benchmarks/run_existing_summary.sh
OUTDIR=/path/to/one-run scripts/benchmarks/run_nx256_exact_benchmark.sh
```

GPU timing scripts try to use `gpulock perf --wait-gpu-idle` when `gpulock` is
installed. If not found, they print a warning and run directly.

## Existing-result summary

```bash
scripts/benchmarks/run_existing_summary.sh
```

Produces `existing_results_summary.csv` from already available CelebA, SIFT30,
Tagore, and fused-topK logs. This is a provenance helper, not the final paper
gate.

## UNG full cross-edge A/B

```bash
scripts/benchmarks/run_ung_cross_edge_ab.sh cpu
scripts/benchmarks/run_ung_cross_edge_ab.sh gpu
scripts/benchmarks/run_ung_cross_edge_ab.sh fused
```

These run the full `build_UNG_index` path on `sift30_zipf_origstyle`.

## End-to-end search recall A/B

```bash
DATASET=amazon \
DATA_DIR=/path/to/amazon_data \
QUERY_DIR_NAME=query_xxx \
GT_FILE=/path/to/amazon_gt_labels_containment.bin \
scripts/benchmarks/run_end_to_end_recall_ab.sh
```

Runs the reviewer-critical A/B: CPU Vamana group graph vs FastGrnndCuda /
AdaptiveCuda variants under the same query set, GT, cross-edge implementation,
additional-edge setting, and `Lsearch` values.
The output `summary.csv` joins build time, group/cross breakdown, average recall,
and average query time from `search_time_summary.csv`.

This is the required quality gate for build changes. PG/group graph, cross-edge,
additional_edges, output-boundary, and router changes should be judged by this
filtered-search recall pipeline, not by standalone topK edge overlap.

The default cross-edge setting is the paper fused path:

```text
UNG_CROSS_EDGE_IMPL=1
UNG_GPU_TOPK_IMPL=3
UNG_CROSS_EDGE_GPU_STRICT=1
```

For current universal-route experiments, set:

```bash
UNG_UNIVERSAL_GPU=1 \
UNG_ADDITIONAL_EDGES_IMPL=0 \
scripts/benchmarks/run_end_to_end_recall_ab.sh
```

`UNG_ADDITIONAL_EDGES_IMPL=0` is the full-quality setting because it keeps CPU
Vamana additional edges. `UNG_ADDITIONAL_EDGES_IMPL=1` skips additional edges
and is only valid for stage timing, kernel/output-boundary ablations, or smoke
tests.

## Query entry group hyperparameter sweep

```bash
BASE_LABEL_FILE=/path/to/base_labels.txt \
QUERY_LABEL_FILE=/path/to/query_labels.txt \
MAX_QUERY=10240 \
FRONTIER_DELTAS="1 2 3" \
FRONTIER_COVER_CAPS="64 128 256 512 1024 2048 8192" \
scripts/benchmarks/run_query_entry_group_sweep.sh
```

This sweeps `gpu_cover_frontier` hyperparameters and writes:

```text
query_entry_group_sweep.csv
query_entry_group_sweep.md
```

For this benchmark, correctness means coverage-correct entry groups: all real
candidate groups satisfying `query_labels subset labels(group)` must be covered.
Quality means fewer returned entry groups under that correctness constraint,
because fewer groups usually means the provider starts higher in the label/LNG
hierarchy and gives less work to graph expansion. The sweep includes CPU scan
and CPU exact-minimal baselines. It does not by itself prove end-to-end query
speedup; that still requires `run_end_to_end_recall_ab.sh` with
`ENTRY_GROUP_PROVIDER=gpu_cover_frontier`.

The single-case helper also accepts `FRONTIER_DELTA`:

```bash
FRONTIER_DELTA=1 FRONTIER_COVER_CAP=8192 \
scripts/benchmarks/run_query_entry_group_bench.sh
```

For diagnostic A/B runs where fused cross-edge is not the variable under test,
these can now be overridden. For example, SIFT30 group-graph recall A/B used
CPU Vamana cross-edge and CPU additional edges:

```bash
UNG_CROSS_EDGE_IMPL=0 \
UNG_GPU_TOPK_IMPL=0 \
UNG_CROSS_EDGE_GPU_STRICT=0 \
UNG_ADDITIONAL_EDGES_IMPL=0 \
scripts/benchmarks/run_end_to_end_recall_ab.sh
```

Use `DRY_RUN=1` to validate paths and commands without launching the builds.

To summarize any completed search runs:

```bash
scripts/benchmarks/summarize_search_recall.py \
  cpu=/path/to/cpu_search_run \
  fused=/path/to/fused_search_run \
  -o /path/to/search_recall_summary.csv
```

## Amazon 1% x400 light reverse-tail A/B

```bash
scripts/benchmarks/run_x400_reverse_tail_ab.sh
```

This runs the x400 full-quality gather-Q experiment needed by the reviewer
iteration:

- `light512_norepair`: historical light512 behavior with
  `UNG_FAST_GRNND_REPAIR_DEGREE=0`
- `light512_repair`: default repair path with
  `UNG_FAST_GRNND_REPAIR_DEGREE=1`
- `light512_reverse_repair`: same route plus
  `UNG_FAST_GRNND_LIGHT_REVERSE_CAP/SLOTS/FORWARD_CAP`

Set `RUN_CPU=1` to also rebuild the CPU Vamana group baseline. By default the
script refuses to run unless GPU utilization is zero, used GPU memory is below
`GPU_IDLE_MAX_MEMORY_MB` (default `1024` MiB), and no compute app is present.
Set `WAIT_GPU_IDLE=0` to override this guard. Use `DRY_RUN=1` to validate paths
and per-variant env files without launching builds. Set `RUN_COMPACT_ABLATION=1` to add a
`UNG_TAGORE_COMPACT_D2H=0` ablation for measuring graph writeback and host-fill
overhead. Set `RUN_GRAPH_DIAG=1` to scan each saved index after build/search and
write a combined reviewer table:

```text
<out_root>/x400_ab_summary.md
<out_root>/x400_ab_summary.csv
```

To summarize an existing x400 A/B root:

```bash
python3 tools/benchmarks/summarize_x400_ab.py /path/to/x400_ab_root \
  --csv /path/to/x400_ab_root/x400_ab_summary.csv \
  --md /path/to/x400_ab_root/x400_ab_summary.md
```

## Paper Artifact Audit

Before moving claims from the reviewer-audit section into the main paper tables,
run:

```bash
python3 tools/benchmarks/audit_paper_artifacts.py \
  --csv /tmp/paper_artifact_audit.csv \
  --md /tmp/paper_artifact_audit.md \
  --fail-required
```

This checks whether the local files named by the evidence matrix are present and
labels remaining items as `PENDING_EXPERIMENT` or `MISSING_EXPERIMENT`.
After running the x400 A/B, pass its output root:

```bash
python3 tools/benchmarks/audit_paper_artifacts.py \
  --x400-root /path/to/x400_ab_root \
  --fail-submission
```

Use `--fail-required` while drafting; use `--fail-submission` only as the final
submission gate because it fails on every pending or missing experiment.

Detailed reviewer-facing protocol and pass/fail criteria are documented in:

```text
docs/runbooks/X400_REVERSE_TAIL_AB_CN.md
```

## FastGrnndCuda / AdaptiveCuda mixed exact-GNN router A/B

```bash
scripts/benchmarks/run_group_graph_router_ab.sh
```

This sweeps `UNG_FAST_GRNND_BATCH_EXACT_NX` and compares against an optional CPU
Vamana group baseline. It is intended to answer whether the mixed router can
use batched exact GPU construction for small/medium groups without reproducing
the old pure-exact recall regression.

Important current interpretation: the old batched-exact low-recall result is now
treated as a confounded ablation because later graph diagnostics showed missing
cross/additional edges, not a different intra-group exact graph. Packed
exact-anchor is therefore a current positive candidate, but only in
full-quality A/B runs with fixed cross/additional settings.

Default target:

```text
/home/graphdb/FilterVectorBenchData/fv_amazon_1pct_x200_jitter_nonempty
query_coverage_1000
ROUTER_THRESHOLDS="0 128 256 512"
```

Useful invocations:

```bash
DRY_RUN=1 scripts/benchmarks/run_group_graph_router_ab.sh

ROUTER_THRESHOLDS="0 64 128 192 256" \
LSEARCH_VALUES="100 200 500 1000 5000" \
scripts/benchmarks/run_group_graph_router_ab.sh
```

The pass/fail criterion is end-to-end, not just build time: a threshold is only
paper-usable if `group_ms`/`index_ms` improves while recall remains close to the
non-exact FastGrnndCuda route and CPU Vamana baseline.

To summarize a completed run:

```bash
python3 tools/benchmarks/summarize_group_graph_router_ab.py <out_root> \
  --csv <out_root>/router_ab_summary.csv \
  --md <out_root>/router_ab_summary.md
```

The paper artifact audit accepts the same root:

```bash
python3 tools/benchmarks/audit_paper_artifacts.py \
  --router-root <out_root> \
  --fail-required
```

The audit deliberately keeps `DRY_RUN=1` outputs in `PENDING_EXPERIMENT` state
because those summaries contain only `NA` timing/recall rows.

## Stored UNG graph structure diagnostics

```bash
python3 tools/benchmarks/diagnose_ung_graph_structure.py \
  /path/to/index_files \
  --out /path/to/graph_diag \
  --reciprocal-limit 0
```

This reads a saved UNG `graph` and `group_id_to_range` without rebuilding the
index. It reports out-degree, intra/cross edge split, low-degree ratios, and
per-group weak connectivity. Use it to explain recall regressions such as the
x400 light-prune gap. Set a positive `--reciprocal-limit` only for small graphs
or sampled indexes; reciprocal coverage stores intra edges in memory.

To compare multiple diagnostic outputs:

```bash
python3 tools/benchmarks/summarize_graph_diagnostics.py \
  cpu=/path/to/cpu_diag \
  light512=/path/to/light512_diag \
  -o /path/to/summary_table.md
```

## Cross-edge baseline fairness summary

```bash
python3 tools/benchmarks/summarize_cross_baselines.py \
  cpu_vamana=/path/to/cpu_vamana_run \
  cpu_exact_128t=/path/to/cpu_exact_run \
  cuvs_per_group=/path/to/cuvs_run \
  sgemm_topk=/path/to/sgemm_topk_run \
  final_fused=/path/to/final_fused_run \
  --baseline cpu_vamana \
  --csv /path/to/cross_baselines.csv \
  --md /path/to/cross_baselines.md
```

This extracts `build_time.csv`, index `meta`, and `[GPU GEMM] H2D/Kernel/D2H`
events from saved runs. It is intended for paper tables that answer whether
the `22x` cross-edge speedup only compares against a weak CPU Vamana baseline.

## CelebA singleton cross-edge split, refactor GPU path

```bash
scripts/benchmarks/run_celeba_singleton_cross_edge.sh
```

Runs three CelebA builds with `UNG_BENCH_MIN_NX/MAX_NX` on the current
refactor/GPU cross-edge path. This is not the old CPU baseline; the old CPU
singleton split in `docs/reports/CROSS_GROUP_CELEBA_RESULTS_CN.md` was measured separately
under `/home/graphdb/FilterVectorCode_old_rerun_results/celeba_singleton_cpu_20260521_153552`.

- `nx1`: only singleton target groups.
- `nx_ge2`: non-singleton target groups.
- `all`: no `nx` filter.

The summary is written to `summary.csv` under the run directory.

## `100 x nx=256` exact topK benchmark

```bash
scripts/benchmarks/run_nx256_exact_benchmark.sh
```

Defaults:

```text
NUM_GROUPS=100
NX=256
NQ_LIST="1024 2560"
DIM=128
TOPK=6
PAYLOADS=2
REPLAYS=4
```

It reports:

- `nvidia_sgemm_plus_topk`: cuBLAS strided-batched SGEMM + separate topK.
- `our_fused_l2_topk`: one-warp/query fused L2 + topK kernel.
- `our_hybrid_sgemm_for_heavy`: current recommended heavy-workload routing.

## `100 x nx=256` Tagore Vamana build + graph query

```bash
scripts/benchmarks/run_tagore_nx256_benchmark.sh
```

This generates 100 synthetic `.fvecs` groups, builds Tagore Vamana indexes, then
runs the GPU graph-query benchmark for `NQ_LIST="1024 2560"`.

This is a standalone microbenchmark. It should not be reported as the current
UNG group graph method unless it is connected to an end-to-end
`build_UNG_index` + filtered-search recall A/B. Current mainline group graph
experiments should use `UNG_GROUP_GRAPH_IMPL=3` or `UNG_GROUP_GRAPH_IMPL=4`.

## Run all lightweight/default pieces

```bash
scripts/benchmarks/run_all_benchmarks.sh
```

Set `RUN_HEAVY_UNG=1` to also run full UNG builds:

```bash
RUN_HEAVY_UNG=1 scripts/benchmarks/run_all_benchmarks.sh
```
