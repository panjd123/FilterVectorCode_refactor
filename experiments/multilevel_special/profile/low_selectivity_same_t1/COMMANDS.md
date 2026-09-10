# Reproduction commands

All paths below are on `sunyahuia6000` as user `sunyahui`. The commands were
launched from `/home/sunyahui/worktrees/FilterVectorCode_multilevel_special`.

## Preflight

```bash
sha256sum build_ung_rel/apps/search_UNG_index
git rev-parse HEAD
nsys --version
ncu --version
command -v gpulock
nvidia-smi --query-gpu=timestamp,index,name,utilization.gpu,memory.used,memory.total \
  --format=csv,noheader
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader
nsys status -e
```

The preflight output is under `raw/`. `gpulock` was absent, so every measured
invocation is preceded by the two `nvidia-smi` checks embedded in `run_case.sh`.

## Correctness and detailed stage/counter runs

The wrapper sets `UNG_VALIDATE_FILTER_RESULTS=1`,
`UNG_SPECIAL_LIGHT_STATS=0`, and `UNG_SPECIAL_PROFILE_TIMING=1`. It explicitly
unsets the optional GPU free-distance switch.

```bash
base=experiments/multilevel_special/profile/low_selectivity_same_t1

# 0.903%, same-L and historical same-Recall point.
$base/run_case.sh $base/runs/sel1_single_l10000 \
  query_minlen5_avgsel1pct \
  runs/fresh_single_build_amazon_x1/single_1000/block_index 10000 single \
  > $base/raw/sel1_single_l10000.log 2>&1
$base/run_case.sh $base/runs/sel1_multi10k_l10000 \
  query_minlen5_avgsel1pct \
  runs/multilevel_builds_amazon_x1/t2_10000/block_index 10000 multi \
  > $base/raw/sel1_multi10k_l10000.log 2>&1

# 9.907%, same-L causal point.
$base/run_case.sh $base/runs/sel10_single_l15000 \
  query_minlen3_avgsel10pct \
  runs/fresh_single_build_amazon_x1/single_1000/block_index 15000 single \
  > $base/raw/sel10_single_l15000.log 2>&1
$base/run_case.sh $base/runs/sel10_multi10k_l15000 \
  query_minlen3_avgsel10pct \
  runs/multilevel_builds_amazon_x1/t2_10000/block_index 15000 multi \
  > $base/raw/sel10_multi10k_l15000.log 2>&1
```

Formal stage/filter-validation results produced by the sibling experiment use
the same current binary, query and GT and are preferred over rerunning these
commands when their exact points are present.

## Nsight Systems

Systems is run before any decision to use Compute. The profile records process
tree CPU samples/context switches plus CUDA API/kernel activity.

```bash
base=experiments/multilevel_special/profile/low_selectivity_same_t1

$base/run_nsys_case.sh $base/raw/nsys_sel1_single_l4500 \
  $base/runs/nsys_sel1_single_l4500 query_minlen5_avgsel1pct \
  runs/fresh_single_build_amazon_x1/single_1000/block_index 4500 single \
  > $base/raw/nsys_sel1_single_l4500.log 2>&1

$base/run_nsys_case.sh $base/raw/nsys_sel1_multi25k_l4500 \
  $base/runs/nsys_sel1_multi25k_l4500 query_minlen5_avgsel1pct \
  runs/multilevel_builds_amazon_x1/t2_25000/block_index 4500 multi \
  > $base/raw/nsys_sel1_multi25k_l4500.log 2>&1
```

The coordinated window was deliberately restricted to these two sel1 cases. No
additional sel10 profile was run. Reports were queried without GUI using:

```bash
nsys stats --report cuda_api_sum,cuda_gpu_kern_sum,osrt_sum,cuda_gpu_trace \
  $base/raw/nsys_sel1_single_l4500.nsys-rep
nsys stats --report cuda_api_sum,cuda_gpu_kern_sum,osrt_sum,cuda_gpu_trace \
  $base/raw/nsys_sel1_multi25k_l4500.nsys-rep
```

Nsight Systems 2023.4 does not provide a built-in `cpustats` report. CPU leaf
samples for the search path were extracted from each exported SQLite database:

```sql
WITH search_samples AS (
  SELECT DISTINCT c.id
  FROM SAMPLING_CALLCHAINS AS c
  JOIN StringIds AS frame ON frame.id = c.symbol
  WHERE frame.value LIKE 'ANNS::UniNavGraph::execute_special_block_ung_query%'
), totals AS (
  SELECT count(*) AS n
  FROM COMPOSITE_EVENTS AS event
  JOIN search_samples USING (id)
)
SELECT leaf_name.value AS leaf, count(*) AS samples,
       100.0 * count(*) / (SELECT n FROM totals) AS pct
FROM COMPOSITE_EVENTS AS event
JOIN search_samples USING (id)
JOIN SAMPLING_CALLCHAINS AS leaf_frame
  ON leaf_frame.id = event.id AND leaf_frame.stackDepth = 0
JOIN StringIds AS leaf_name ON leaf_name.id = leaf_frame.symbol
GROUP BY leaf_frame.symbol
ORDER BY samples DESC;
```

## Nsight Compute decision

NCU is run only if Systems identifies a search-relevant GPU kernel. If both
paths have zero CUDA API calls and zero CUDA kernels during the query run, NCU
is intentionally skipped and that negative result is part of the evidence.
