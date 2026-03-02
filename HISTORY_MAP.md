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

### 2026-03-03 - `216cd6c` (high-impact)
- Commit: `perf(lng): replace hash sets with vectors for descendants/coverage`
- Files changed:
  - `UNG/codes/include/label_nav_graph.h`
  - `UNG/codes/src/uni_nav_graph.cpp`
- Core changes:
  - `LabelNavGraph::covered_sets`:
    - `std::vector<std::unordered_set<IdxType>>` -> `std::vector<std::vector<IdxType>>`
  - `LabelNavGraph::_lng_descendants`:
    - `std::vector<std::unordered_set<IdxType>>` -> `std::vector<std::vector<IdxType>>`
  - `get_descendants_info`:
    - remove hash insert path; write BFS discovered ids directly to vector container
  - `cal_f_coverage_ratio`:
    - descendants-direct path writes contiguous vectors (no hash build)
    - legacy path appends vectors then `sort + unique` for correctness under DAG multi-path overlap
  - `initialize_roaring_bitsets`:
    - parallelized over groups
    - switched from per-element `add` loop to batched `addMany`
    - removed verbose debug prints

Validation:
- GPU/CPU structural checks after change:
  - `lng_descendants_rb.bin`, `covered_sets_rb.bin`, `vector_attr_graph`, `lng_descendants_num`, `lng_coverage_ratio` are md5-identical between:
    - `/home/graphdb/FilterVectorResultsRefactor/opt_vecset_gpu_20260303_003040/index_files`
    - `/home/graphdb/FilterVectorResultsRefactor/opt_vecset_cpu_20260303_003205/index_files`

Performance:
- Single-run reference:
  - before: `/home/graphdb/FilterVectorResultsRefactor/baseline_head_20260303_002418/others/ung_build.log`
    - `Index time: 49982 ms`
  - after: `/home/graphdb/FilterVectorResultsRefactor/opt_vecset_gpu_20260303_003040/others/ung_build.log`
    - `Index time: 42214 ms`

- A/B (2 runs each, same params/env):
  - CSV: `/home/graphdb/FilterVectorResultsRefactor/ab_vecset_20260303_003348.csv`
  - compared commits:
    - `before`: `dcd5dee`
    - `after`: `216cd6c`
  - median results (ms):
    - `Finished building LNG`: `39002.6 -> 32308.1` (`-17.2%`)
    - `descendants`: `3059.7 -> 104.6` (`-96.6%`)
    - `coverage`: `29665.9 -> 219.1` (`-99.3%`)
    - `cross-group edges`: `1540.2 -> 1414.0` (`-8.2%`)
    - `Index time`: `84115.0 -> 35314.5` (`-58.0%`)

### 2026-03-03 - `d68f3a0` (safe trie/phase1 follow-up)
- Commit: `perf(lng): cut trie and Phase1 allocation overhead`
- Files changed:
  - `UNG/codes/src/trie.cpp`
  - `UNG/codes/src/uni_nav_graph.cpp`
- Core changes (semantics-preserving):
  - `build_label_nav_graph` Phase1:
    - use thread-local reusable `min_super_set_ids` buffer
    - write by `swap` to avoid repeated allocation/copy
  - `TrieIndex::insert`:
    - avoid redundant `_label_to_nodes.resize(...)` on every new node; resize only when needed
  - `TrieIndex::get_super_set_entrances`:
    - replace `std::queue` with vector+head-index traversal
    - replace `std::set` dedup with `std::unordered_set` dedup
    - add label bound checks before indexing `_label_to_nodes`

Validation (graph semantics stayed aligned):
- for before/after A/B logs:
  - `Average number of descendants per group` remained `275.4`
  - `LNG edges` remained `946138`
  - `target_groups` remained `115073`

Performance A/B (2 runs each):
- CSV:
  - `/home/graphdb/FilterVectorResultsRefactor/ab_safe_trie_20260303_005332.csv`
- compared commits:
  - `before`: `8d1a549`
  - `after`: `d68f3a0`
- median results (ms):
  - `Finished building LNG`: `45746.45 -> 33878.15` (`-25.9%`)
  - `descendants`: `131.75 -> 117.6` (`-10.7%`)
  - `coverage`: `261.2 -> 329.75` (`+26.2%`, small absolute +68.6ms)
  - `cross-group edges`: `1518.25 -> 1354.85` (`-10.8%`)
  - `Index time`: `49100.5 -> 36969.5` (`-24.7%`)

### Correctness Smoke
- GPU backend smoke:
  - `/home/graphdb/FilterVectorResultsRefactor/refactor_mig_smoke_gpu_20260302_235333/others/ung_build.log`
- CPU backend smoke:
  - `/home/graphdb/FilterVectorResultsRefactor/refactor_mig_smoke_cpu_20260302_235447/others/ung_build.log`
- Structural consistency checks:
  - `lng_descendants_rb.bin`, `covered_sets_rb.bin`, `vector_attr_graph` md5 are identical between CPU/GPU smokes.
  - `graph` differs (expected candidate: cross-edge tie-order / backend ordering differences), no crash or format mismatch.
