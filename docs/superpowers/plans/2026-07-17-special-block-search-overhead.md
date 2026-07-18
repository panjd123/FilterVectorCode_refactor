# Special Block Search Overhead Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reduce `gpu_bruteforce_els_special_blocks` core-outside search overhead on `query_minlen1_cov100k` so its `core_search_time_ms` advantage can show up in `search_time_ms` and batch time.

**Architecture:** Reuse per-thread search scratch state instead of allocating and clearing `_num_points`-sized special visited arrays for every query. Keep the special block algorithm semantics unchanged and verify recall plus time breakdowns before and after.

**Tech Stack:** C++17 UNG search backend, existing `VisitedSet` epoch marker, existing CSV benchmark outputs, Python 3 verification script using standard library only.

## Global Constraints

- Do not change special block graph semantics or recall intentionally.
- Do not overwrite existing dirty worktree changes.
- Use the existing `SearchCache` scratch object to hold reusable query-local state.
- Validate with `query_minlen1_cov100k` CSV metrics before and after.

---

### Task 1: Add Baseline Metric Script

**Files:**
- Create: `tools/benchmarks/compare_special_overhead.py`

**Interfaces:**
- Consumes: two result directories with `query_minlen1_cov100k/results/query_details_repeat1.csv` and `search_time_summary.csv`.
- Produces: printed metrics for `Time_ms`, `search_time_ms`, `core_search_time_ms`, non-core search overhead, recall, and batch time.

- [ ] **Step 1: Write the script**

Create a script that accepts `--ung-dir`, `--special-dir`, `--query`, and `--min-lsearch`.

- [ ] **Step 2: Run baseline**

Run it on existing `FilterVectorResult/Amazon/results` directories and record current `query_minlen1_cov100k` overhead.

### Task 2: Reuse Special Search Scratch State

**Files:**
- Modify: `UNG/codes/include/search_cache.h`
- Modify: `UNG/codes/include/uni_nav_graph.h`
- Modify: `UNG/codes/src/uni_nav_graph_search.cpp`
- Modify: `UNG/codes/src/uni_nav_graph_search_backend.cpp`

**Interfaces:**
- Consumes: existing `SearchCache` from `thread_function`.
- Produces: `execute_special_block_ung_query(..., std::shared_ptr<SearchCache> search_cache, ...)`.

- [ ] **Step 1: Extend `SearchCache`**

Add two `VisitedSet` members for special regular/free states and initialize them with `_num_points`.

- [ ] **Step 2: Pass `SearchCache` to special search**

Update the private method declaration, call site, and definition.

- [ ] **Step 3: Replace per-query `std::vector<uint8_t>` visited arrays**

Use `special_visited_regular.clear()` and `special_visited_free.clear()` at query start, then `check()`/`set()` for membership.

### Task 3: Verify Build And Metrics

**Files:**
- No additional files unless the existing build requires generated outputs.

**Interfaces:**
- Consumes: modified C++ sources and existing benchmark command if available.
- Produces: compile result and metric comparison.

- [ ] **Step 1: Build the affected target**

Use the existing local build if present. If dependencies or GPU runtime block execution, report the blocker precisely.

- [ ] **Step 2: Run or prepare `query_minlen1_cov100k` benchmark**

If the benchmark can run locally, generate fresh special results and compare against baseline. If it cannot run, at least verify compilation and provide the exact command to run.

- [ ] **Step 3: Compare metrics**

Use `tools/benchmarks/compare_special_overhead.py` to check whether non-core search overhead decreases and recall stays close.
