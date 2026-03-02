# Optimization Provenance

- Optimization reference repo: `/home/graphdb/aaaGPU/FilterVectorCode`
- Source branch/commit: `gpu-opt-correctness-loop` (see `.optimization_source_aaagpu_commit`)
- Imported optimization files:
  - `UNG/codes/include/uni_nav_graph.h`
  - `UNG/codes/src/uni_nav_graph.cpp`
  - `UNG/codes/src/gpu_gemm_topk.cu`
  - `UNG/codes/src/CMakeLists.txt`
- Baseline CUDA file is preserved as `UNG/codes/src/uni_nav_graph_cuda.baseline.cu` for diff/audit.

## Incremental Migration (2026-03-02)

- Target commit in this repo: `076d990` (`optimized-clean-v2`)
- Additional source reference (aaaGPU): `eb6d892`
- Incrementally reintroduced in `UNG/codes/src/uni_nav_graph.cpp`:
  - `get_min_super_sets` container reuse + bucketized candidate traversal
  - `get_descendants_info` thread-local BFS buffers
  - `cal_f_coverage_ratio` threaded and `descendants_direct` path with env toggles

## Reverted Experiment (2026-03-03)

- Tested commit: `ff47638`
  - attempted to stabilize `descendants_direct` by capping coverage threads and changing OMP schedule
- Reverted by: `14a0fa9`
- Reason: reproducible regression in `coverage` and end-to-end index time (see `HISTORY_MAP.md` benchmark links)

## High-impact LNG/coverage Optimization (2026-03-03)

- Commit: `216cd6c` (current `optimized-clean-v2`)
- Scope:
  - `UNG/codes/include/label_nav_graph.h`
  - `UNG/codes/src/uni_nav_graph.cpp`
- Summary:
  - converted descendants/coverage containers from hash-set layout to contiguous vectors
  - rewired descendants and coverage construction to avoid hash insertion hot paths
  - kept legacy coverage branch correctness via post-merge dedup (`sort + unique`)
  - switched Roaring bitmap init to batched `addMany` + OMP parallel over groups
- Expected effect:
  - remove large hash-table construction overhead and reduce memory traffic
  - significantly reduce volatility in `descendants/coverage` stage timing
