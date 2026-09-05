# UNG Entry Route Performance Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Remove repeated entry-route Roaring unions and CPU ELS cold work from the measured Amazon query sweep, reuse search resources across Lsearch values, and reduce special-edge startup parsing without changing default search results or recall.

**Architecture:** Add a single-flight cache whose immutable value contains both entry group IDs and their derived route statistics. Replace the all-label dense ELS allocation with workload-prepared/lazy per-label rows, warm all unique workload keys before timing, and pass one reusable execution context through the Lsearch sweep. Extract special-edge binary I/O into a shared module and add a non-destructive CSV-to-binary converter.

**Tech Stack:** C++17, Roaring bitmaps, OpenMP/std::thread, CMake/CTest, Python 3 benchmark analysis.

## Global Constraints

- Preserve all pre-existing dirty-worktree edits; stage only files and hunks belonging to the current task.
- Default results, distances, recall, and route-stat CSV semantics must remain unchanged.
- Cold-start/warm-up time must be reported separately from measured search time.
- Do not retain an optimization that lacks reproducible benefit on `query_selected_recall_advantage`.
- Phase 2 entry-width and candidate-queue experiments are out of scope until this Phase 1 plan is benchmarked.

---

### Task 1: Single-flight complete entry-result cache

**Files:**
- Create: `UNG/codes/include/ung_entry_group_cache.h`
- Create: `UNG/codes/src/ung_entry_group_cache.cpp`
- Create: `UNG/codes/test/test_entry_group_cache.cpp`
- Modify: `UNG/codes/include/ung_entry_group.h`
- Modify: `UNG/codes/src/CMakeLists.txt`
- Modify: `UNG/codes/test/CMakeLists.txt`

**Interfaces:**
- Produces: `EntryGroupRouteStats`, `CachedEntryGroupResult`, and `EntryGroupResultCache::get_or_compute(const std::string &, ComputeFn)`.
- Cache values are immutable snapshots; one producer runs per key while concurrent callers wait on its shared future.

- [ ] **Step 1: Write the failing single-flight cache test**

Add a test that launches eight real threads against one key. The producer increments an atomic counter and returns the hand-derived value below:

```cpp
ANNS::EntryGroupResultCache cache;
std::atomic<int> computes{0};
std::vector<ANNS::CachedEntryGroupResult> observed(8);
std::vector<std::thread> workers;
for (size_t i = 0; i < observed.size(); ++i) {
   workers.emplace_back([&, i] {
      observed[i] = cache.get_or_compute("cpu:1,2", [&] {
         computes.fetch_add(1);
         return ANNS::CachedEntryGroupResult{
             {3, 7}, ANNS::EntryGroupRouteStats{true, 2, 11, 29, 0.125f}};
      });
   });
}
for (auto &worker : workers) worker.join();
expect(computes.load() == 1, "concurrent identical keys must compute once");
for (const auto &value : observed) {
   expect(value.group_ids == std::vector<ANNS::IdxType>({3, 7}), "cached IDs must be exact");
   expect(value.route_stats.entry_group_matched_points == 29, "cached stats must be exact");
}
```

Also test that an exception is observed by current waiters and that a later call retries successfully rather than retaining a poisoned cache entry.

- [ ] **Step 2: Run the test target and verify RED**

Run:

```bash
cmake -S UNG/codes -B build_ung_rel -DCMAKE_BUILD_TYPE=Release
cmake --build build_ung_rel -j16 --target test_entry_group_cache
```

Expected: compilation fails because `ung_entry_group_cache.h` and its public types do not exist.

- [ ] **Step 3: Implement the minimal cache**

Define:

```cpp
struct EntryGroupRouteStats {
   bool valid = false;
   size_t num_entry_points = 0;
   size_t num_lng_descendants = 0;
   size_t entry_group_matched_points = 0;
   float entry_group_total_coverage = 0.0f;
};

struct CachedEntryGroupResult {
   std::vector<IdxType> group_ids;
   EntryGroupRouteStats route_stats;
};

class EntryGroupResultCache {
public:
   using ComputeFn = std::function<CachedEntryGroupResult()>;
   CachedEntryGroupResult get_or_compute(const std::string &key, const ComputeFn &compute);
   size_t size() const;
private:
   mutable std::mutex mutex_;
   std::unordered_map<std::string, std::shared_future<CachedEntryGroupResult>> values_;
};
```

The producer inserts a promise/future under the lock, computes after releasing the lock, and publishes the value. On failure it sets the promise exception and erases the key so a later request retries.

- [ ] **Step 4: Build and verify GREEN**

Run:

```bash
cmake --build build_ung_rel -j16 --target test_entry_group_cache
ctest --test-dir build_ung_rel --output-on-failure -R '^entry_group_cache$'
```

Expected: one test passes, zero failures.

- [ ] **Step 5: Commit Task 1 only**

```bash
git add UNG/codes/include/ung_entry_group.h UNG/codes/include/ung_entry_group_cache.h UNG/codes/src/ung_entry_group_cache.cpp UNG/codes/src/CMakeLists.txt UNG/codes/test/test_entry_group_cache.cpp UNG/codes/test/CMakeLists.txt
git commit -m "perf: add single-flight entry group result cache"
```

---

### Task 2: Cache and restore route statistics

**Files:**
- Modify: `UNG/codes/include/ung_entry_group.h`
- Modify: `UNG/codes/include/uni_nav_graph.h`
- Modify: `UNG/codes/src/uni_nav_graph_entry_provider.cpp`
- Modify: `UNG/codes/src/uni_nav_graph_query_features.cpp`
- Modify: `UNG/codes/test/test_entry_group_cache.cpp`

**Interfaces:**
- Consumes: `EntryGroupResultCache` and `EntryGroupRouteStats` from Task 1.
- Produces: `UniNavGraph::compute_entry_group_route_stats(...) const` and provider results with `route_stats.valid == true` on cached CPU ELS paths.

- [ ] **Step 1: Write failing tests for exact stat restoration**

Extend the cache test with a literal `QueryStats` application check:

```cpp
ANNS::QueryStats stats;
const ANNS::EntryGroupRouteStats snapshot{true, 2, 11, 29, 0.125f};
ANNS::apply_entry_group_route_stats(snapshot, stats);
expect(stats.num_entry_points == 2, "cached entry count must be restored");
expect(stats.num_lng_descendants == 11, "cached descendant count must be restored");
expect(stats.entry_group_matched_points == 29, "cached matched points must be restored");
expect(stats.entry_group_total_coverage == 0.125f, "cached coverage must be restored");
```

Name the break: a cache hit that restores IDs but silently leaves route statistics at zero.

- [ ] **Step 2: Run and verify RED**

Run `cmake --build build_ung_rel -j16 --target test_entry_group_cache`.

Expected: compilation fails because `apply_entry_group_route_stats` does not exist.

- [ ] **Step 3: Refactor route-stat computation and integrate the cache**

Add:

```cpp
EntryGroupRouteStats UniNavGraph::compute_entry_group_route_stats(
    const std::vector<IdxType> &entry_group_ids) const;
```

Move the two Roaring unions from `populate_entry_group_route_stats` into this function. Make `populate_entry_group_route_stats` call the new function and apply the snapshot, preserving non-cached providers.

Change `_cpu_bruteforce_els_query_cache` from `unordered_map<string, vector<IdxType>>` plus mutex to `EntryGroupResultCache`. In `compute_cpu_bruteforce_entry_groups_for_execution`, compute IDs and route stats inside `get_or_compute`; on hits copy both into `EntryGroupProviderResult`. Update `prepare_entry_groups_for_execution`:

```cpp
if (result.route_stats.valid)
   apply_entry_group_route_stats(result.route_stats, stats);
else
   populate_entry_group_route_stats(entry_group_ids, stats);
```

- [ ] **Step 4: Verify focused and existing ELS tests**

```bash
cmake --build build_ung_rel -j16 --target test_entry_group_cache test_cpu_bruteforce_els
ctest --test-dir build_ung_rel --output-on-failure -R '^(entry_group_cache|cpu_bruteforce_els)$'
```

Expected: both tests pass; no compile warnings.

- [ ] **Step 5: Commit Task 2 only**

```bash
git add UNG/codes/include/ung_entry_group.h UNG/codes/include/uni_nav_graph.h UNG/codes/src/uni_nav_graph_entry_provider.cpp UNG/codes/src/uni_nav_graph_query_features.cpp UNG/codes/test/test_entry_group_cache.cpp
git commit -m "perf: reuse cached entry route statistics"
```

---

### Task 3: Workload-aware CPU ELS rows and pre-timing warm-up

**Files:**
- Modify: `UNG/codes/include/ung_cpu_bruteforce_els.h`
- Modify: `UNG/codes/src/ung_cpu_bruteforce_els.cpp`
- Modify: `UNG/codes/include/uni_nav_graph.h`
- Modify: `UNG/codes/src/uni_nav_graph_entry_provider.cpp`
- Modify: `UNG/codes/apps/search_UNG_index.cpp`
- Modify: `UNG/codes/test/test_cpu_bruteforce_els.cpp`

**Interfaces:**
- Produces: `build_cpu_bruteforce_els_rows(...)`, lazy row preparation, and `CpuElsWarmupStats UniNavGraph::warmup_cpu_bruteforce_els(...)`.
- The query path calls the same row-preparation primitive as warm-up, so unseen labels remain correct.

- [ ] **Step 1: Write failing workload-row tests**

Use the literal group-label fixture below:

```cpp
const std::vector<std::vector<ANNS::LabelType>> labels{
   {}, {1}, {1, 2}, {2, 3}, {4}
};
const auto rows = ANNS::build_cpu_bruteforce_els_rows(labels, 4, {1, 3});
expect(rows.label_group_bits.size() == 2, "only requested label rows must be built");
expect(bit_is_set(rows.label_group_bits.at(1), 1), "label 1 must contain group 1");
expect(bit_is_set(rows.label_group_bits.at(1), 2), "label 1 must contain group 2");
expect(bit_is_set(rows.label_group_bits.at(3), 3), "label 3 must contain group 3");
expect(rows.label_group_bits.count(2) == 0, "unrequested label 2 must not be allocated");
```

Add a second test that lazily builds label 2 and obtains groups 2 and 3. Expected values must be literals, not computed with the production selector.

- [ ] **Step 2: Run and verify RED**

Run `cmake --build build_ung_rel -j16 --target test_cpu_bruteforce_els`.

Expected: compilation fails because workload-row APIs do not exist.

- [ ] **Step 3: Replace the all-label dense allocation**

Change `CpuBruteForceElsCache::label_group_bits` to:

```cpp
std::unordered_map<LabelType, std::vector<uint64_t>> label_group_bits;
```

Keep `size_group_bits` dense because it has only `max_group_label_size + 1` rows. Implement one group scan that fills only requested missing labels. The normal query path calls the same function with its labels before intersecting rows. Protect preparation with the existing cache mutex and never interpret a not-yet-built row as a nonexistent index label.

- [ ] **Step 4: Add the explicit warm-up API and application logging**

Add:

```cpp
struct CpuElsWarmupStats {
   size_t unique_query_keys = 0;
   size_t prepared_labels = 0;
   double elapsed_ms = 0.0;
};
CpuElsWarmupStats warmup_cpu_bruteforce_els(
    const std::shared_ptr<IStorage> &query_storage,
    bool recursive_more_start,
    bool ung_more_entry,
    size_t scalar_els_cap);
```

Collect unique canonical label sets and their union, prepare rows once, then resolve every unique key through the production provider/cache. In `search_UNG_index`, invoke it during the existing warm-up phase whenever `entry_group_provider == CpuBruteForceEls`. Log exactly one summary line with elapsed time, unique key count, and prepared label count.

- [ ] **Step 5: Verify tests and a format-check search**

```bash
cmake --build build_ung_rel -j16 --target search_UNG_index test_cpu_bruteforce_els test_entry_group_cache
ctest --test-dir build_ung_rel --output-on-failure -R '^(cpu_bruteforce_els|entry_group_cache)$'
```

Then run the existing small `query_selected_recall_advantage_format_check` command pattern with `UNG_SPECIAL_BLOCK_SEARCH=1`; write results to a new `/tmp/ung-entry-route-format-check` directory. Expected: warm-up log reports 79 prepared labels for the full workload (or the exact smaller fixture count for format-check), and first measured ELS time no longer contains cache construction.

- [ ] **Step 6: Commit Task 3 only**

```bash
git add UNG/codes/include/ung_cpu_bruteforce_els.h UNG/codes/src/ung_cpu_bruteforce_els.cpp UNG/codes/include/uni_nav_graph.h UNG/codes/src/uni_nav_graph_entry_provider.cpp UNG/codes/apps/search_UNG_index.cpp UNG/codes/test/test_cpu_bruteforce_els.cpp
git commit -m "perf: warm workload-aware CPU ELS rows"
```

---

### Task 4: Reuse search threads and caches across Lsearch values

**Files:**
- Create: `UNG/codes/include/ung_search_execution_context.h`
- Modify: `UNG/codes/include/search_cache.h`
- Modify: `UNG/codes/include/uni_nav_graph.h`
- Modify: `UNG/codes/src/uni_nav_graph_search.cpp`
- Modify: `UNG/codes/apps/search_UNG_index.cpp`
- Modify: `UNG/codes/test/test_navix_search.cpp` or create `UNG/codes/test/test_search_execution_context.cpp`
- Modify: `UNG/codes/test/CMakeLists.txt`

**Interfaces:**
- Produces: `SearchExecutionContext(uint32_t num_threads, IdxType num_points, IdxType max_lsearch)` and a `search_hybrid(..., SearchExecutionContext &context, ...)` overload.
- Existing callers keep the compatibility overload, which creates a one-call context.

- [ ] **Step 1: Write a failing lifecycle/equivalence test**

Create a deterministic small graph/query fixture and run two Lsearch values through one context. Run the same values through the compatibility overload. Assert literal query result IDs and equality between both paths. Also assert the second call does not retain visited/free state from the first.

Name the break: reused cache state contaminates the next Lsearch or a max-capacity queue silently changes the requested Lsearch breadth.

- [ ] **Step 2: Run and verify RED**

Build the focused test target. Expected: compilation fails because `SearchExecutionContext` and the overload do not exist.

- [ ] **Step 3: Implement reusable resources without changing logical capacity**

The context owns one `SearchCacheList` built at `max_lsearch` and one `ThreadPool`. Before a cache is used for a query, call:

```cpp
search_cache->search_queue.reserve(runtime.Lsearch);
```

`SearchQueue::reserve` already grows storage only when required while always resetting logical `_capacity`, so a max-sized context does not change search breadth. Refactor the enqueue/join block to use `context.pool` and `context.search_cache_list`.

In `search_UNG_index`, construct the context once using the maximum requested Lsearch before the repeat/Lsearch loops and pass it to every call.

- [ ] **Step 4: Verify focused, full CTest, and format-check results**

```bash
cmake --build build_ung_rel -j16 --target search_UNG_index test_search_execution_context
ctest --test-dir build_ung_rel --output-on-failure
```

Expected: zero failures; reusable and compatibility results match.

- [ ] **Step 5: Commit Task 4 only**

```bash
git add UNG/codes/include/ung_search_execution_context.h UNG/codes/include/search_cache.h UNG/codes/include/uni_nav_graph.h UNG/codes/src/uni_nav_graph_search.cpp UNG/codes/apps/search_UNG_index.cpp UNG/codes/test/test_search_execution_context.cpp UNG/codes/test/CMakeLists.txt
git commit -m "perf: reuse UNG search execution resources"
```

---

### Task 5: Shared binary special-edge I/O and non-destructive converter

**Files:**
- Create: `UNG/codes/include/ung_special_edge_io.h`
- Create: `UNG/codes/src/ung_special_edge_io.cpp`
- Create: `UNG/codes/tools/convert_special_edges.cpp`
- Create: `UNG/codes/test/test_special_edge_io.cpp`
- Modify: `UNG/codes/src/uni_nav_graph_special_blocks.cpp`
- Modify: `UNG/codes/src/CMakeLists.txt`
- Modify: `UNG/codes/tools/CMakeLists.txt`
- Modify: `UNG/codes/test/CMakeLists.txt`

**Interfaces:**
- Produces: packed 16-byte `SpecialEdgeBinaryRecord`, validated binary reader/writer, and `convert_special_edges <csv> <bin>`.
- Binary format: `uint64_t count`, followed by `{uint32_t source,target,block; uint8_t kind; uint8_t pad[3];}` records.

- [ ] **Step 1: Write failing round-trip and invalid-file tests**

Use a three-edge CSV fixture with literal intra/inter records. Convert it, read it through the shared reader, and assert all fields and count. Add truncated-header and truncated-record cases; they must return a validation error rather than a partial success.

- [ ] **Step 2: Run and verify RED**

Build `test_special_edge_io`. Expected: missing header/API compilation failure.

- [ ] **Step 3: Extract validated I/O and implement atomic conversion**

Move the existing binary record/read/write logic out of the anonymous namespace. Validate file size against `sizeof(uint64_t) + count * sizeof(SpecialEdgeBinaryRecord)`. The converter writes `<output>.tmp`, flushes/closes, validates it, and renames it to the requested output only on success. It never deletes the CSV.

Change index loading to prefer a valid binary sidecar whenever it exists; on validation failure log the reason and fall back to CSV.

- [ ] **Step 4: Verify tests and convert the acceptance index**

```bash
cmake --build build_ung_rel -j16 --target convert_special_edges test_special_edge_io search_UNG_index
ctest --test-dir build_ung_rel --output-on-failure -R '^special_edge_io$'
build_ung_rel/tools/convert_special_edges \
  /home/dev/graphdb/FilterVectorResult/Amazon/index/UNG_special_blocks_hybrid_bdeg64_bcross4_iroute1_small2048_large8192/index_files/special_edges.csv \
  /home/dev/graphdb/FilterVectorResult/Amazon/index/UNG_special_blocks_hybrid_bdeg64_bcross4_iroute1_small2048_large8192/index_files/special_edges.bin
```

Expected: converted count is exactly 35,430,136; CSV remains present; a load log reports binary selection and the same edge count.

- [ ] **Step 5: Commit Task 5 only**

```bash
git add UNG/codes/include/ung_special_edge_io.h UNG/codes/src/ung_special_edge_io.cpp UNG/codes/tools/convert_special_edges.cpp UNG/codes/test/test_special_edge_io.cpp UNG/codes/src/uni_nav_graph_special_blocks.cpp UNG/codes/src/CMakeLists.txt UNG/codes/tools/CMakeLists.txt UNG/codes/test/CMakeLists.txt
git commit -m "perf: load special edges from validated binary sidecar"
```

---

### Task 6: Acceptance benchmark and keep/revert decisions

**Files:**
- Create: `tools/benchmarks/compare_ung_query_sweeps.py`
- Create: `tools/tests/test_compare_ung_query_sweeps.py`
- Create: `docs/reports/UNG_ENTRY_ROUTE_PERFORMANCE_AMAZON_CN.md`
- Modify only if required by benchmark defects: files from Tasks 1-5.

**Interfaces:**
- Produces a CSV/Markdown comparison keyed by Lsearch with wall time, QPS, recall, mean query/core/ELS/Other time, and relative speedup.

- [ ] **Step 1: Write the failing comparison-tool test**

Create temporary baseline/optimized CSV fixtures with two Lsearch rows and literal values. Assert that the tool:

- rejects mismatched Lsearch sets;
- rejects any recall mismatch by default;
- computes `other_speedup = baseline_other / optimized_other` correctly;
- reports cold Lsearch separately from steady-state rows.

- [ ] **Step 2: Run and verify RED**

```bash
python3 -m unittest tools.tests.test_compare_ung_query_sweeps -v
```

Expected: import/file failure because the comparison tool does not exist.

- [ ] **Step 3: Implement the minimal comparison tool**

Inputs are `--baseline-dir`, `--optimized-dir`, and `--out-dir`. Read `search_time_summary.csv` and `query_details_repeat1.csv`. Match rows by Lsearch and QueryID, compare recall exactly at parsed float precision, and write both machine-readable CSV and a concise Markdown report.

- [ ] **Step 4: Build and run all tests before benchmarking**

```bash
cmake --build build_ung_rel -j16 --target search_UNG_index convert_special_edges
ctest --test-dir build_ung_rel --output-on-failure
python3 -m unittest tools.tests.test_compare_ung_query_sweeps -v
```

Expected: all commands exit 0 with zero test failures.

- [ ] **Step 5: Run the full optimized Amazon sweep into a new result directory**

Use the exact command recorded in the baseline `Amazon_search_output.txt`, changing only `--result_path_prefix` to a new `query_selected_recall_advantage_1000_1000_20000_optimized/results/` path. Set `UNG_SPECIAL_BLOCK_SEARCH=1`. Do not overwrite the baseline result directory.

Run at least three full repeats in separate output directories or with `--num_repeats 3`; report repeat 1 separately and use repeats 2-3 for steady-state variance. Preserve the same 100-thread, 16-entry-point, neighbor-list configuration for the primary comparison.

- [ ] **Step 6: Compare correctness and performance**

```bash
python3 tools/benchmarks/compare_ung_query_sweeps.py \
  --baseline-dir /home/dev/graphdb/FilterVectorResult/Amazon/results/UNG_special_blocks_hybrid_bdeg64_bcross4_iroute1_small2048_large8192/query_selected_recall_advantage_1000_1000_20000/results \
  --optimized-dir /home/dev/graphdb/FilterVectorResult/Amazon/results/UNG_special_blocks_hybrid_bdeg64_bcross4_iroute1_small2048_large8192/query_selected_recall_advantage_1000_1000_20000_optimized/results \
  --out-dir /home/dev/graphdb/FilterVectorCode_refactor/docs/reports/ung_entry_route_amazon_data
```

Acceptance requires identical recall for every Lsearch, identical sampled result IDs where available, materially lower steady-state `OtherT_ms`, and no measured-search ELS cold spike. Record index-load and warm-up times separately.

- [ ] **Step 7: Keep only demonstrated optimizations and write the report**

If a task has no reproducible wall-time/QPS benefit, revert only that task's commit after preserving its measurements in the report. Write final before/after tables and explicit retained/rejected decisions in `UNG_ENTRY_ROUTE_PERFORMANCE_AMAZON_CN.md`.

- [ ] **Step 8: Final verification and commit**

```bash
git diff --check
ctest --test-dir build_ung_rel --output-on-failure
python3 -m unittest tools.tests.test_compare_ung_query_sweeps -v
git status --short
```

Stage only the comparison tool, its test, the report, and any verified benchmark-driven fixes:

```bash
git add tools/benchmarks/compare_ung_query_sweeps.py tools/tests/test_compare_ung_query_sweeps.py docs/reports/UNG_ENTRY_ROUTE_PERFORMANCE_AMAZON_CN.md
git commit -m "perf: validate UNG entry route optimization on Amazon"
```
