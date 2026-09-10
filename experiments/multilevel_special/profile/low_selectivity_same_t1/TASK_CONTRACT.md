# Task contract: low-selectivity same-T1 profile

## Question

Why can a fixed two-level Special Block index be slower than the single-level
index when both have `T1=1000`, for Amazon x1 containment workloads below 10%
selectivity?

## Controlled comparison

- Repository: `/home/sunyahui/worktrees/FilterVectorCode_multilevel_special`
- Search binary: one immutable SHA-256 for every compared case (recorded at run time)
- Dataset/index: Amazon x1, the same base vectors and main UNG index
- Queries/GT: identical within each workload pair
- Search settings: K=10, 100 query threads, 16 entry points, CPU brute-force ELS,
  neighbor-list backend, containment semantics
- Structural variable: single-level `T1=1000` versus two-level `T1=1000` and
  a recorded `T2`
- Representative workloads: 0.903% and 9.907% mean selectivity
- Correctness: `UNG_VALIDATE_FILTER_RESULTS=1`; require zero filter violations and
  record Recall before accepting any timing/profile result
- Timing: exclude cold load/repeat where summaries permit; preserve per-stage and
  per-query counters; profile the same executable/query/GT/L pairs

## Comparison lenses

1. Same-L causal lens: holds search budget fixed and attributes runtime/counter deltas.
2. Equal/near-equal-Recall lens: prevents a quality increase from being mislabeled
   as overhead. Exact equality is preferred; if unavailable on the measured grid,
   report the closest bracketing points and the Recall gap explicitly.

## Hypotheses and falsifiers

- H1 GPU-kernel regression: two-level is slower because it launches more/slower CUDA
  kernels. Falsified if the timed search has no CUDA kernels/API calls in either case.
- H2 authorization overhead: two-level scans more block metadata before graph search.
  Supported if authorization time rises materially and source attribution points to
  the block-coverage/hierarchy loop; weakened if authorization is negligible.
- H3 traversal-work increase: upper-level transitions expand/scan more CPU graph work.
  Supported if detailed counters (upper activations/nodes/edges, distance calculations,
  queue operations) and graph-stage time increase together.
- H4 launch/synchronization overhead: supported only by an Nsys CUDA API/kernel timeline.
- H5 measurement/quality confound: same-L slowdown disappears at equal Recall, or Recall
  differs enough that the pair is not performance-comparable.

## Tool decision rule

Use Nsight Systems first for CPU sampling, OS runtime, CUDA API, and CUDA kernels.
Run Nsight Compute only if Systems shows a search-relevant GPU kernel whose behavior
differs between the pair. The source currently gates the optional GPU free-distance
batch to `upper_blocks == 0` and the formal environments do not enable it; therefore
an empty search-time CUDA lane is an expected, reportable result, not a reason to force NCU.

## Resource isolation

`gpulock` is not installed on the remote host as of 2026-09-10. Before every timing
or profile command, record `nvidia-smi` utilization and process state and proceed only
when GPU utilization is 0%. This is weaker than an exclusive lock and is a declared
measurement limitation.

## Write boundary

Only `experiments/multilevel_special/profile/low_selectivity_same_t1/` is created or
modified. No source, existing configuration, `AGENT_KANBAN.md`, paper document, or
`/home/graphdb/FilterVectorCode_refactor` content is changed.
