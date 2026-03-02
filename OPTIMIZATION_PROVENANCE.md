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
