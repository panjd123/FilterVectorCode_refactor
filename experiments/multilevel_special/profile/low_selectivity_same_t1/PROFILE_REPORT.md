# Same-T1 low-selectivity profile report

## Executive conclusion

For Amazon x1 at 0.903% mean selectivity, the formal seven-repeat run contains
an exact-quality slower point: single-level `T1=1000,L=4500` and two-level
`T1=1000,T2=25000,L=4500` both reach Recall 0.9043 with zero filter
violations, while warm-batch medians are 154.999 and 168.684 ms. The two-level
point is 13.686 ms, or 8.83%, slower in that run.

The evidence does **not** support the intuitive explanation that the second
level is doing extra search work:

- every observed middle/upper block, node, edge, activation and queue counter is
  zero in both cases; total distance calculations are effectively identical
  (3640.081 versus 3639.962 per query);
- mutually exclusive per-query stage medians are slightly lower for two-level
  (11.425 versus 11.587 ms), including graph search (9.822 versus 9.936 ms);
- both Nsight Systems reports contain no CUDA API trace, no CUDA kernel data and
  no GPU trace; the runtime path is CPU-only;
- the profiled rerun did not reproduce the slowdown: its warm medians were
  162.282 ms single and 154.769 ms two-level, so two-level was 4.63% faster.
  CPU sample composition was also nearly the same.

Therefore the defensible result is: a fixed two-level overlay *can measure slower*
at a low-selectivity same-T1 point, but this run does not establish a deterministic
second-level execution cost. The remaining candidates are CPU-side run-to-run
variation and indirect memory-layout/cache effects from the larger overlay. The
latter is plausible but unproven: the selected two-level overlay is 737 MiB rather
than 515 MiB, while its `special_edges.bin` is 559,565,016 rather than
327,980,552 bytes.

## Measurement contract and provenance

- Repository: `/home/sunyahui/worktrees/FilterVectorCode_multilevel_special`
- Branch / commit: `codex/multilevel-special-block-20260905` /
  `e71079a536a5175a22d4b74546e7d9b1a49e12c0`
- Search binary SHA-256:
  `6fa4082dae77e99b6c1b3c84c75d324d57cc25e0da32149011379453cfa40087`
- Dataset: Amazon x1; identical base vectors, main UNG index, query set, exact
  ground truth, K=10, 100 query threads, 16 entries, containment semantics,
  CPU brute-force ELS and `L=4500` for the selected pair.
- Structural variable only: single `T1=1000` versus fixed two-level
  `T1=1000,T2=25000`.
- Correctness: `UNG_VALIDATE_FILTER_RESULTS=1`; all formal selected-method rows
  had zero violations (210,000 checked results per method for sel1 and 490,000
  per method for sel10). Both profile cases checked 10,000 results in every
  repeat with zero violations.
- Timing: formal comparisons use six warm repeats after excluding repeat 0.
- Tools: Nsight Systems 2023.4.4.54; Nsight Compute 2024.1.0.
- Isolation limitation: `gpulock` is absent. The two profiles were coordinated
  with the sibling experiment and launched only after `nvidia-smi` reported 0%
  utilization and no compute process. This is weaker than an exclusive lock.

The exact task contract, commands, preflight, binary hash, source excerpts and
compact text/JSON exports are preserved beside this document. The large
`.nsys-rep`/SQLite files and per-query profile runs remain untracked on the
measurement host. No source or configuration was changed for profiling.

## Formal current-binary evidence

### Selected low-selectivity pair (0.903%, same L and exact same Recall)

| metric | single T1=1k | two-level T1=1k,T2=25k | two-level delta |
|---|---:|---:|---:|
| Recall | 0.9043 | 0.9043 | 0 |
| warm batch mean (ms) | 156.780 | 164.571 | +4.97% |
| warm batch median (ms) | 154.999 | 168.684 | +8.83% |
| warm CV | 5.02% | 5.72% | +0.70 pp |
| per-query total median (ms) | 11.587 | 11.425 | -1.39% |
| ELS median (ms/query) | 0.900 | 0.868 | -3.56% |
| entry setup median (ms/query) | 0.671 | 0.650 | -3.07% |
| block authorization median (ms/query) | 0.01138 | 0.01095 | -3.76% |
| graph search median (ms/query) | 9.936 | 9.822 | -1.14% |
| residual median (ms/query) | 0.0877 | 0.0617 | -29.66% |
| distance calculations/query | 3640.081 | 3639.962 | -0.003% |
| free blocks/query | 1.140 | 1.140 | 0 |
| special blocks searched/query | 0.839 | 0.839 | 0 |
| middle/upper blocks, nodes, edges, activations | 0 | 0 | 0 |
| special queue counters | 0 | 0 | 0 |
| filter violations | 0 | 0 | 0 |

The apparent tension between batch wall time and per-query stage medians is real,
not a closure bug: each query row closes to its five stages (maximum closure error
is below 4e-14 ms), but 100 concurrent query timings do not sum one-to-one to the
batch critical path. In the two-level run, the stage row selected by the
batch-median repeat had total 12.333 ms/query and graph 10.612 ms/query, whereas
the median across warm stage rows was 11.425 and 9.822 ms/query. This prevents a
claim that any measured stage is a stable 8.83% culprit.

### Boundary near 10% selectivity

The completed sel10 extension changes the interpretation of the earlier
same-`L=15000` slowdown. At that L, the two-level index has substantially higher
Recall, so it is not an equal-quality overhead comparison. On the final discrete
grid, single-level first reaches target Recall 0.900 at `L=35000` (Recall 0.9002,
warm median 6417.535 ms); two-level T2=25k reaches it at `L=13500` (Recall 0.9023,
warm median 2386.735 ms), a 2.689x median speedup. Thus the near-10% result is a
quality/work crossing, not evidence that two levels are inherently slower.

This report focuses its profiler comparison on the user-selected two sel1 cases;
no additional sel10 profile was run.

## Nsight Systems evidence

Both reports were captured with process-tree CPU sampling/context switches and
`cuda,nvtx,osrt` tracing. The profile wrapper used the identical binary, sel1
query/GT, `L=4500` and seven repeats.

### Correctness and profile timing

| case | Recall | warm times (ms) | warm mean | warm median |
|---|---:|---|---:|---:|
| single T1=1k | 0.9043 | 189.832, 163.564, 204.341, 156.746, 161.000, 152.589 | 171.345 | 162.282 |
| two-level T1=1k,T2=25k | 0.9043 | 174.883, 155.195, 150.825, 154.342, 156.147, 153.620 | 157.502 | 154.769 |

The profiled rerun reverses the formal ordering: two-level is 4.63% faster by
median and 8.08% faster by mean. Profiling overhead means these values do not
replace formal timing; the reversal demonstrates that the formal slowdown is not
stable enough to assign to a deterministic code path from this evidence.

### CPU sampling

Nsight Systems 2023.4 has no built-in `cpustats` report, so the exported SQLite
`COMPOSITE_EVENTS`, `SAMPLING_CALLCHAINS` and `StringIds` tables were queried
directly. Samples whose call chain contains
`UniNavGraph::execute_special_block_ung_query` isolate the search path from load
and parsing work. There are 120,962 such samples for single (76.46% of all
158,193 samples) and 115,464 for two-level (75.18% of all 153,590 samples).

| search-path leaf | single | two-level |
|---|---:|---:|
| `SpecialCandidateQueue::insert` | 44.07% | 43.53% |
| `FloatL2DistanceHandler::compute_avx2` | 16.77% | 16.96% |
| inlined/body `execute_special_block_ung_query` symbol A | 12.18% | 12.23% |
| `memmove` | 9.41% | 9.34% |
| inlined/body `execute_special_block_ung_query` symbol B | 7.74% | 8.65% |
| `GraphSearchBackend::neighbors` | 2.58% | 2.65% |
| `SpecialCandidateQueue::has_unexpanded` | 1.15% | 1.15% |
| `Storage<float>::get_vector` | 0.75% | 0.81% |
| `special_edges_for_point` | 0.30% | 0.33% |

The dominant work in both cases is the same CPU candidate-queue, AVX2 distance,
copy/move and neighbor-access path. The sample proportions do not reveal a new
two-level hotspot. Since no NVTX query ranges were added, samples cannot be split
exactly by repeat; they are statistical attribution over the process run.

### OS runtime and CUDA absence

The OS-runtime summaries are dominated by thread-pool waits and locks. Across
all threads, `pthread_cond_wait` totals 58.485 s / 1185 calls for single and
54.955 s / 981 calls for two-level; `pthread_mutex_lock` totals 9.283 s / 1156
calls and 8.233 s / 724 calls. These are aggregate overlapping thread times, not
wall times, and do not identify a two-level penalty.

For **both** reports, `nsys stats` states:

- `does not contain CUDA trace data`;
- `does not contain CUDA kernel data`;
- `does not contain GPU trace data`.

This falsifies H1 (GPU-kernel regression) and H4 (CUDA launch/synchronization
overhead) for the measured execution.

## Nsight Compute decision

NCU was intentionally skipped. Systems found no kernel to profile, and source
inspection agrees: the optional GPU distance batch requires
`UNG_SPECIAL_FREE_GPU_DISTANCE` and `_special_block_summary.upper_blocks == 0`;
the formal/profile environment did not enable it. Running NCU would therefore
produce no evidence relevant to this two-level CPU path and would violate the
predeclared Nsys-to-NCU decision rule.

## Source attribution

- `UNG/codes/src/uni_nav_graph_search_backend.cpp:337-338`: profile timing and
  detailed-counter switches.
- `UNG/codes/src/uni_nav_graph_search_backend.cpp:456-570`: block authorization,
  hierarchical free-state propagation and middle/upper counting.
- `UNG/codes/src/uni_nav_graph_search_backend.cpp:967-1022`: candidate insertion
  and middle/upper activation accounting.
- `UNG/codes/src/uni_nav_graph_search_backend.cpp:1278-1445`: special-edge scan,
  level transitions, scalar distance path and optional GPU gate.
- `UNG/codes/src/uni_nav_graph_search_backend.cpp:1460-1532`: regular-neighbor
  scan and activation propagation.
- `UNG/codes/src/uni_nav_graph_search_backend.cpp:1541-1577`: result extraction
  and final search timing.
- `UNG/codes/src/ung_gpu_l2_batch.cu:205-237`: optional CUDA batch path.
- `UNG/codes/apps/search_UNG_index.cpp:659-760`: filter validation and batch wall
  timing.
- `UNG/codes/apps/search_UNG_index.cpp:789-916`: stage summary emission.

Exact numbered excerpts are in `raw/source_attribution.txt`.

## Hypothesis disposition and falsifier

| hypothesis | disposition | evidence |
|---|---|---|
| H1 GPU-kernel regression | rejected | no CUDA trace or kernel data in either report |
| H2 authorization overhead | rejected for selected pair | authorization is 0.01138 vs 0.01095 ms/query |
| H3 extra upper traversal | rejected for selected pair | all upper/middle execution counters are zero; work counts match |
| H4 CUDA launch/sync overhead | rejected | no CUDA API or GPU timeline |
| H5 quality/measurement confound | supported | sel10 same-L Recall differs; sel1 slowdown reverses under Nsys |
| memory-layout/cache effect | plausible, unproven | overlay is 222 MiB larger, but no cache-counter experiment was run |

The minimal next experiment that could establish or falsify the remaining CPU
layout explanation is an interleaved randomized A/B benchmark of the exact sel1
pair under an exclusive CPU/GPU lock, with many process-level repetitions, CPU
affinity/frequency controls and hardware counters for cycles, instructions, LLC
misses and memory bandwidth. A repeatable positive two-level delta that tracks
LLC/memory pressure while distance/upper counters remain equal would support the
layout hypothesis. If the confidence interval includes zero or ordering continues
to flip, the observed 8.83% formal point should be treated as measurement variance.
