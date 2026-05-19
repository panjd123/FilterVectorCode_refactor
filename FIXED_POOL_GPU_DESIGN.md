# FixedPoolGPU Experimental Builder

## Goal

Replace Vamana's dynamic graph construction with an experimental fixed-size neighbor-pool builder that is easier to map to GPU execution.

The builder is selected with:

```bash
--index_type FixedPoolGPU
```

It is intentionally experimental. The default `Vamana` path is unchanged.

## Algorithm

For each label group independently:

1. Allocate a fixed pool of `L` neighbor candidates for every point.
2. Initialize each pool with deterministic random neighbors.
3. Repeat for `T` iterations:
   - for each point `v`, inspect `neighbor(v)` and `neighbor(neighbor(v))` candidates;
   - compute distances from `v` to those candidates on GPU;
   - keep the best `L` candidates in the fixed pool.
4. Robust-prune the final pool to at most `R=max_degree` graph edges.
5. Write edges into the existing `Graph::neighbors` layout so downstream LNG, cross-edge, search, and save code can be reused.

## Current Implementation

Files:

- `UNG/codes/include/fixed_pool_graph.h`
- `UNG/codes/src/fixed_pool_graph.cu`
- `UNG/codes/src/uni_nav_graph.cpp`
- `UNG/codes/src/CMakeLists.txt`

Environment knobs:

- `UNG_FIXED_POOL_L`: pool size, clamped to `[max_degree, 128]`.
- `UNG_FIXED_POOL_ITERS`: neighbor-of-neighbor refinement iterations.
- `UNG_FIXED_POOL_SEED`: deterministic random initialization seed.
- `UNG_FIXED_POOL_THREADS`: CUDA threads per point block.

Constraints:

- Supports `float` vectors only.
- Supports `max_degree <= 128`.
- Designed as a correctness/profiling prototype, not the final optimized kernel.

## Smoke Test

Dataset:

- 1024 SIFT vectors
- one label group
- `R=16`, `L=32`, `iters=2`

Results:

| Builder | build_graph_time | graph_num_edges | total index_time |
|---|---:|---:|---:|
| FixedPoolGPU | 197.50 ms | 11182 | 201.13 ms |
| Vamana | 65.18 ms | 13456 | 68.63 ms |

Logs:

- FixedPoolGPU: `/home/graphdb/FilterVectorResultsRefactor/fixedpool_smoke_general_20260519_131334/others/ung_build.log`
- Vamana: `/home/graphdb/FilterVectorResultsRefactor/vamana_smoke_general_20260519_131344/others/ung_build.log`

## Why It Is Slower Now

This first version still has several prototype costs:

1. Per group `cudaMalloc/cudaFree` and H2D copy.
2. One group is launched at a time instead of batching many groups.
3. Candidate merge is serial inside each block after parallel distance computation.
4. Robust prune is serial per node.
5. No persistent device-resident reordered base vector buffer is reused across groups.

## Next Optimization Plan

1. Batch multiple groups in one launch and reuse device allocations.
2. Keep all reordered base vectors resident on GPU during group graph construction.
3. Replace serial per-node merge with warp/block top-L selection.
4. Replace serial robust prune with staged approximate prune:
   - top-L sort/select in shared memory;
   - warp-level occlusion checks;
   - compact selected edges.
5. Add NCU profiles for the refine and prune kernels.
6. Compare recall/search quality against Vamana, not just build time.
