# Authoritative hierarchy-build protocol

## Question and claim boundary

The build study asks whether a multilevel index, with GPU assistance, can be
built no slower than the original CPU method while preserving its search
quality.  It does **not** assume that adding an independently materialized
hierarchy can have zero cost relative to the same accelerated base-only build.
For an independent sidecar the latter total is necessarily

`T(total multilevel) = T(base) + T(hierarchy sidecar)`.

Accordingly, the paper reports two distinct comparisons:

1. stage-composed full-index construction cost versus the original CPU baseline; and
2. hierarchy overhead relative to the same accelerated base build.

"No slower" is accepted only when the upper bound of the 95% confidence
interval for the multilevel/original wall-time ratio is at most 1.0.  A point
estimate below 1.0 alone is not sufficient.

## Frozen build objects

- `B0-original-cpu`: zero-layer LNG base index using `original_cpu`.
- `B0-current-cpu`: zero-layer LNG base index using `current_cpu`.
- `B0-paper-gpu`: zero-layer LNG base index using the frozen accelerated GPU
  profile used by the search comparison.
- `H1-cpu`: the representative one-layer prefix of DRH-v1
  (`1024:lng` on Amazon), built with CPU layer-local and CPU inter-block edge
  construction.
- `H1-gpu`: the same representative one-layer structure with GPU-assisted
  construction.
- `Hauto-cpu`: the query-independent DRH-v1 hierarchy with CPU construction.
- `Hauto-hybrid`: the same DRH-v1 hierarchy using CPU small blocks, sampled
  middle blocks, and GPU large blocks.
- `Hauto-full-gpu`: the same DRH-v1 hierarchy with GPU layer-local and GPU
  inter-block construction enabled.

The manual query oracle is deliberately not a separate construction row. Its
purpose is to quantify query-time tuning regret; building its winning structure
would not change the CPU/GPU backend comparison and could silently make the
construction study conditional on query measurements. The automatic and
manual-oracle query results remain reported together in the query section.

The hierarchy thresholds and per-layer topology must be identical between the
CPU/GPU rows.  Backend changes may change graph edges, so their output is not
treated as byte-identical and must pass the downstream quality checks below.

## Timing protocol

GPU timing uses an external `gpulock perf 0 -- ...` lock when that host tool is
available. The current host does not provide `gpulock`; in that environment,
every GPU case must instead pass three consecutive `nvidia-smi` samples with
no compute process, 0% utilization, and at most 16 MiB of driver memory. The
accepted snapshots are stored in `gpu_isolation.json` and copied into the
manifest. This fallback is fail closed but is reported as
`idle_preflight_no_lock`: it verifies pre-run idleness and does not claim the
race-free exclusivity of a lock.

The isolation check is outside the timed region. Each component wall time starts
immediately before its builder child process and includes that component's
complete CPU/GPU pipeline. The reported full-index total is the sum of the base
and hierarchy median wall times; it is a stage-composed estimate, not a claim
that both stages ran inside one timed child process. CUDA kernel microbenchmarks
remain separate evidence and are never substituted for construction wall time.

- Use one immutable snapshot of each build executable and record SHA-256.
- Run builds serially on the same host with the same thread count, graph degree,
  `Lbuild`, alpha, input data, and source index.
- Build hierarchy sidecars from measured repeat 0 of the same campaign's
  accelerated base profile. This keeps timing and quality provenance aligned.
- Run one unreported cold build followed by at least five measured builds per
  primary row.  Interleave CPU and GPU rows by repeat to reduce time drift.
- Primary time is process wall time from invocation through validated files on
  disk. Internal metadata, layer-local edge, inter-block edge, serialization,
  and compact-reload timers are explanatory and must close against wall time.
- Report median, p95, coefficient of variation, and a 95% confidence interval.
  For the full-index ratio, resample original base, accelerated base, and
  hierarchy stage measurements independently; repeat identifiers from separate
  phases are not statistical pairs.
- Resource profiling is a separate pass. Sample the direct process at 200 ms
  for peak resident CPU memory and peak GPU memory; do not mix this monitored
  pass into the primary wall-time estimate.
- Record final hierarchy bytes and total base-plus-hierarchy bytes. Cache-drop
  privileges and filesystem state must be reported; do not label a run cold-I/O
  unless page cache was actually controlled.
- Validate effective backend evidence, not only requested environment values.
  A GPU-labelled row is invalid if the persisted base profile differs, a
  requested GPU stage reports no work, or the log reports CPU fallback.

## Search-quality equivalence

Every measured build must pass structural validation before timing is accepted.
One representative output from each backend is then searched with the same
binary, queries, GT, entry strategy, and measured Recall-crossing protocol used
by the query study. Report:

- Recall@10 at the frozen comparison `Lsearch` values;
- minimum measured `Lsearch` reaching Recall@10 >= 0.90;
- QPS at that crossing;
- filter violations, which must remain zero; and
- graph edge counts and index sizes.

The GPU build is quality-equivalent only if it reaches the target Recall on
every declared workload. Performance claims compare QPS at each backend's own
smallest measured crossing; an additional shared-`Lsearch` table reveals graph
quality changes directly.

## Required reported ratios

- `T(B0-original-cpu) / T(B0-paper-gpu + Hauto-full-gpu)` answers the user's
  original-algorithm question.
- `T(Hauto-cpu) / T(Hauto-full-gpu)` isolates hierarchy acceleration.
- `T(B0-paper-gpu + Hauto-full-gpu) / T(B0-paper-gpu)` reports unavoidable
  multilevel overhead against the same modern base builder.
- `T(Hauto-full-gpu) / T(H1-gpu)` reports the marginal cost of automatic depth.

No kernel-only number may be presented as full-index construction speedup.
