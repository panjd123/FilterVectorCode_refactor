# History Map

## Branches
- `baseline-clean`: GPU baseline + dependencies + portability fixes
- `optimized-clean-v2`: baseline-clean + aaaGPU optimization + compatibility fixes

## Key tags
- `baseline-gpu`: initial imported UNG/codes baseline
- `baseline-gpu-complete`: baseline plus ACORN/scripts
- `baseline-gpu-complete-v2`: runnable baseline with dependency/path fixes
- `optimized-from-aaagpu-v2`: core optimization apply point (reference transplant)
- `optimized-from-aaagpu-v2-buildable`: latest buildable optimized head

## Patch exports
- `patches/aaagpu_core_optimization.patch`
- `patches/aaagpu_optimization_buildable_full.patch`

## Recent Migration Log

### 2026-03-02 - `076d990`
- Commit: `perf(ung): migrate trie/descendants/coverage optimizations from aaagpu`
- File changed:
  - `UNG/codes/src/uni_nav_graph.cpp`
- Migrated logic:
  - `get_min_super_sets`: thread-local container reuse + size-bucket traversal path (small span) + large-span sort fallback.
  - `get_descendants_info`: thread-local BFS buffers (`visited_epoch`, queue, discovered vectors) to reduce repeated allocation.
  - `cal_f_coverage_ratio`: optional `descendants_direct` implementation, threaded coverage pass, env controls:
    - `UNG_COVERAGE_IMPL` (`0=legacy_topological_merge`, `1=descendants_direct`)
    - `UNG_COVERAGE_THREADS`

### Validation & Benchmark (A/B, 3 runs each)
- Benchmark CSV:
  - `/home/graphdb/FilterVectorResultsRefactor/ab_benchmark_20260302_235803.csv`
- Compared commits:
  - `before`: `87892ae` (`076d990^`)
  - `after`: `076d990`
- Fixed settings:
  - dataset `celeba`, `--max_degree 64 --Lbuild 100 --alpha 1.2 --num_cross_edges 6`
  - `UNG_CROSS_EDGE_BACKEND=1`
  - `UNG_COVERAGE_IMPL=1`
  - `UNG_COVERAGE_THREADS=32`

Median results (ms):
- `Finished building LNG`: `44578.7 -> 42171.5` (`-5.4%`)
- `descendants`: `2359.1 -> 2032.0` (`-13.9%`)
- `cross-group edges`: `1711.4 -> 1541.0` (`-10.0%`)
- `coverage`: `13965.8 -> 26666.0` (`+90.9%`, high variance)
- `Index time`: `72437 -> 84878` (`+17.2%`, regressed)

Conclusion:
- Cross-edge stage improved.
- Full index time regressed under `UNG_COVERAGE_IMPL=1`, dominated by unstable coverage time.
- This optimization point is kept for audit and follow-up tuning; use A/B logs above for reproduction.

### 2026-03-03 - Stability attempt (reverted)
- Attempted commit: `ff47638`
  - change: in `descendants_direct` coverage path, auto-cap threads to 16 and switch to `guided` schedule.
- A/B benchmark:
  - CSV: `/home/graphdb/FilterVectorResultsRefactor/ab_covcap_20260303_001148.csv`
  - baseline code commit: `5985bb1` (same core code as `076d990`)
  - tested commit: `ff47638`
- Result: regression, so reverted.
  - median `coverage`: `4738.65 -> 21245.55 ms`
  - median `Index time`: `58044.5 -> 83180 ms`
- Revert commit: `14a0fa9`

### Correctness Smoke
- GPU backend smoke:
  - `/home/graphdb/FilterVectorResultsRefactor/refactor_mig_smoke_gpu_20260302_235333/others/ung_build.log`
- CPU backend smoke:
  - `/home/graphdb/FilterVectorResultsRefactor/refactor_mig_smoke_cpu_20260302_235447/others/ung_build.log`
- Structural consistency checks:
  - `lng_descendants_rb.bin`, `covered_sets_rb.bin`, `vector_attr_graph` md5 are identical between CPU/GPU smokes.
  - `graph` differs (expected candidate: cross-edge tie-order / backend ordering differences), no crash or format mismatch.
