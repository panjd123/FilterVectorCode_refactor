# Optimization Step-by-Step (UNG GPU Build Path)

This document summarizes each optimization step, what changed in code, and measured performance impact.

## Measurement Setup

- Dataset: `celeba`
- Build params: `--max_degree 64 --Lbuild 100 --alpha 1.2 --num_cross_edges 6`
- Env: `UNG_CROSS_EDGE_BACKEND=1`, `UNG_COVERAGE_IMPL=1`, `UNG_COVERAGE_THREADS=32`
- Main metric: `Index time` from `others/ung_build.log`
- Stage metrics: `Finished building LNG`, `descendants`, `coverage`, `cross-group edges`

## Step 1: aaagpu migration (regression in end-to-end)

- Commit: `076d990`
- Goal: migrate aaagpu CPU-side optimizations (`get_min_super_sets`, descendants BFS reuse, coverage mode controls)
- Files:
  - `UNG/codes/src/uni_nav_graph.cpp`
- Benchmark:
  - CSV: `/home/graphdb/FilterVectorResultsRefactor/ab_benchmark_20260302_235803.csv`
  - Before `87892ae` vs After `076d990` (median):
    - `Index time`: `72437 -> 84878 ms` (`+17.2%`, regression)
    - `LNG`: `44578.7 -> 42171.5 ms`
    - `descendants`: `2359.1 -> 2032.0 ms`
    - `coverage`: `13965.8 -> 26666.0 ms` (large variance/regression)
    - `cross-group edges`: `1711.4 -> 1541.0 ms`
- Conclusion:
  - Cross-edge improved, but overall regressed due to unstable/slower coverage stage.

## Step 2: coverage thread-cap attempt (reverted)

- Tried commit: `ff47638`
- Revert commit: `14a0fa9`
- Goal: stabilize coverage by capping threads and using guided scheduling
- Benchmark:
  - CSV: `/home/graphdb/FilterVectorResultsRefactor/ab_covcap_20260303_001148.csv`
  - Before `5985bb1` vs After `ff47638` (median):
    - `Index time`: `58044.5 -> 83180.0 ms` (`+43.3%`, regression)
    - `coverage`: `4738.65 -> 21245.55 ms` (strong regression)
- Conclusion:
  - Rejected and reverted.

## Step 3: high-impact container/layout rewrite

- Commit: `216cd6c`
- Goal: remove hash-heavy descendants/coverage construction overhead
- Files:
  - `UNG/codes/include/label_nav_graph.h`
  - `UNG/codes/src/uni_nav_graph.cpp`
- Core code changes:
  - `covered_sets` and `_lng_descendants`: `unordered_set` containers -> contiguous `vector`
  - descendants path writes BFS-discovered ids directly
  - coverage descendants-direct path uses contiguous appends
  - legacy coverage path keeps correctness via `sort + unique`
  - roaring init parallelized with `addMany`
- Correctness checks:
  - CPU/GPU md5 matches for `lng_descendants_rb.bin`, `covered_sets_rb.bin`, `vector_attr_graph`, `lng_descendants_num`, `lng_coverage_ratio`
  - Paths:
    - `/home/graphdb/FilterVectorResultsRefactor/opt_vecset_gpu_20260303_003040/index_files`
    - `/home/graphdb/FilterVectorResultsRefactor/opt_vecset_cpu_20260303_003205/index_files`
- Benchmark:
  - CSV: `/home/graphdb/FilterVectorResultsRefactor/ab_vecset_20260303_003348.csv`
  - Before `dcd5dee` vs After `216cd6c` (median):
    - `Index time`: `84115.0 -> 35314.5 ms` (`-58.0%`)
    - `LNG`: `39002.6 -> 32308.1 ms` (`-17.2%`)
    - `descendants`: `3059.7 -> 104.55 ms` (`-96.6%`)
    - `coverage`: `29665.95 -> 219.1 ms` (`-99.3%`)
    - `cross-group edges`: `1540.2 -> 1413.95 ms` (`-8.2%`)

## Step 4: safe trie/phase1 follow-up

- Commit: `d68f3a0`
- Goal: reduce LNG Phase1 overhead without changing graph semantics
- Files:
  - `UNG/codes/src/trie.cpp`
  - `UNG/codes/src/uni_nav_graph.cpp`
- Core code changes:
  - Phase1 out-neighbor build uses thread-local reusable `min_super_set_ids` + `swap`
  - trie insertion avoids redundant `_label_to_nodes.resize(...)`
  - trie superset traversal uses vector-head queue and hash dedup (`unordered_set`)
  - added bounds guards for `_label_to_nodes` indexing
- Semantic validation from logs:
  - `Average descendants per group`: unchanged (`275.4`)
  - `LNG edges`: unchanged (`946138`)
  - `target_groups`: unchanged (`115073`)
- Benchmark:
  - CSV: `/home/graphdb/FilterVectorResultsRefactor/ab_safe_trie_20260303_005332.csv`
  - Before `8d1a549` vs After `d68f3a0` (median):
    - `Index time`: `49100.5 -> 36969.5 ms` (`-24.7%`)
    - `LNG`: `45746.45 -> 33878.15 ms` (`-25.9%`)
    - `descendants`: `131.75 -> 117.6 ms` (`-10.7%`)
    - `coverage`: `261.2 -> 329.75 ms` (`+26.2%`, small absolute +68.55ms)
    - `cross-group edges`: `1518.25 -> 1354.85 ms` (`-10.8%`)

## Discarded Fast Path (invalid semantics)

- Attempt (not committed): exact-node downward BFS shortcut in `get_min_super_sets`
- Observed issue:
  - `LNG edges` dropped from `946138` to `20007`
  - `Average descendants` dropped from `275.4` to `0.3`
  - `target_groups` dropped from `115073` to `20007`
- Evidence log:
  - `/home/graphdb/FilterVectorResultsRefactor/opt_exactbfs_gpu_20260303_004843/others/ung_build.log`
- Action:
  - fully rolled back before commit; not part of mainline history.

## Current Best Known State

- Branch head performance commit: `d68f3a0`
- Latest doc state commit: see `git log` (current docs commit after `d68f3a0`)
- Best stable end-to-end range observed on CelebA in this round:
  - ~`35s` to `40s` index time with correct LNG semantics
