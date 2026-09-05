# NaviX HNSW UNG Integration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an HNSW/NaviX graph build path to UNG and expose NaviX adaptive-local filtered search through `search_hybrid` with `force_use_alg=5`.

**Architecture:** Reuse UNG storage, label grouping, persistence, and experiment apps. Add a focused HNSW builder that writes into the existing `Graph` adjacency representation, then add a NaviX search backend that consumes UNG entry groups as the prefilter set and adaptively chooses one-hop, directed two-hop, or full two-hop expansion from the NaviX reference algorithm.

**Tech Stack:** C++17, OpenMP, existing UNG `Graph`, `SearchCache`, `DistanceHandler`, Boost program options, CMake, GoogleTest.

## Global Constraints

- Keep user and existing uncommitted changes intact.
- Do not vendor the full Kuzu codebase; port only the algorithmic pieces needed by UNG.
- `index_type=HNSW` and `index_type=NaviX` must both build an HNSW-style graph, with `NaviX` naming available for benchmark clarity.
- `force_use_alg=5` must select NaviX search without requiring ACORN indexes.
- Search results must preserve the existing old-id writeback contract.

---

### Task 1: HNSW Builder

**Files:**
- Create: `UNG/codes/include/navix_hnsw.h`
- Create: `UNG/codes/src/navix_hnsw.cpp`
- Modify: `UNG/codes/src/CMakeLists.txt`
- Test: `UNG/codes/test/test_navix_hnsw.cpp`
- Modify: `UNG/codes/test/CMakeLists.txt`

**Interfaces:**
- Produces: `class ANNS::NavixHnsw { void build(...); IdxType get_entry_point() const; }`
- Consumes: existing `IStorage`, `DistanceHandler`, `Graph`, `SearchCacheList`.

- [ ] **Step 1: Write the failing test**

```cpp
TEST(NavixHnswTest, BuildsBoundedNeighborGraph) {
  auto storage = make_small_float_storage({{0, 0}, {1, 0}, {2, 0}, {10, 0}, {11, 0}});
  auto distance = ANNS::get_distance_handler("float", "L2");
  auto graph = std::make_shared<ANNS::Graph>(storage->get_num_points());
  ANNS::NavixHnsw index(false);

  index.build(storage, distance, graph, 2, 8, 8, 1);

  ASSERT_LT(index.get_entry_point(), storage->get_num_points());
  for (ANNS::IdxType id = 0; id < storage->get_num_points(); ++id) {
    EXPECT_LE(graph->neighbors[id].size(), 2);
  }
  EXPECT_FALSE(graph->neighbors[0].empty());
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cmake --build build_ung_rel -j 8 --target test_navix_hnsw`
Expected: FAIL because `navix_hnsw.h` and target do not exist.

- [ ] **Step 3: Write minimal implementation**

Implement deterministic HNSW-like insertion with one lower layer:
`search_layer`, `select_neighbors`, reverse edge insertion, and max-degree pruning. Use existing distance handler and graph storage.

- [ ] **Step 4: Run test to verify it passes**

Run: `cmake --build build_ung_rel -j 8 --target test_navix_hnsw && ./build_ung_rel/test/test_navix_hnsw`
Expected: PASS.

### Task 2: UNG Build Route

**Files:**
- Modify: `UNG/codes/apps/build_UNG_index.cpp`
- Modify: `UNG/codes/src/uni_nav_graph_group_graph.cpp`
- Modify: `UNG/codes/include/uni_nav_graph.h`
- Test: `UNG/codes/test/test_navix_hnsw.cpp`

**Interfaces:**
- Consumes: `ANNS::NavixHnsw::build`.
- Produces: `index_type` accepts `Vamana`, `HNSW`, and `NaviX`.

- [ ] **Step 1: Write the failing test**

Add a test that calls `UniNavGraph::build(..., "NaviX", ...)` on a small labeled dataset and expects no exit and a nonempty graph after save/load.

- [ ] **Step 2: Run test to verify it fails**

Run: `cmake --build build_ung_rel -j 8 --target test_navix_hnsw && ./build_ung_rel/test/test_navix_hnsw`
Expected: FAIL with invalid index name.

- [ ] **Step 3: Write minimal implementation**

Branch in `build_graph_for_all_groups()` so `Vamana` keeps existing behavior while `HNSW` and `NaviX` use `NavixHnsw` per group. Keep small-group complete graph fast path.

- [ ] **Step 4: Run test to verify it passes**

Run: `cmake --build build_ung_rel -j 8 --target test_navix_hnsw && ./build_ung_rel/test/test_navix_hnsw`
Expected: PASS.

### Task 3: NaviX Search Backend

**Files:**
- Create: `UNG/codes/include/ung_navix_search.h`
- Create: `UNG/codes/src/ung_navix_search.cpp`
- Modify: `UNG/codes/include/uni_nav_graph.h`
- Modify: `UNG/codes/src/uni_nav_graph_search.cpp`
- Modify: `UNG/codes/src/uni_nav_graph_search_backend.cpp`
- Modify: `UNG/codes/include/ung_query_route.h`
- Modify: `UNG/codes/src/uni_nav_graph_query_route.cpp`
- Modify: `UNG/codes/src/CMakeLists.txt`
- Test: `UNG/codes/test/test_navix_search.cpp`
- Modify: `UNG/codes/test/CMakeLists.txt`

**Interfaces:**
- Produces: `bool UniNavGraph::execute_navix_query(...)`.
- Produces: `QueryRouteDecision::uses_navix()` and `force_use_alg=5`.
- Consumes: `compute_bitmap_from_groups`, `GraphSearchBackend`, `SearchCache`, `DistanceHandler`.

- [ ] **Step 1: Write the failing test**

Build a tiny graph and filter set where one-hop alone misses a valid filtered second-hop node; assert `force_use_alg=5` returns filtered ids only and visits at least one second-hop candidate.

- [ ] **Step 2: Run test to verify it fails**

Run: `cmake --build build_ung_rel -j 8 --target test_navix_search`
Expected: FAIL because NaviX route/backend do not exist.

- [ ] **Step 3: Write minimal implementation**

Implement NaviX adaptive-local:
one-hop if local selectivity `>= 0.4`; otherwise compare `(totalNbrs * filteredNbrs + filteredNbrs) * 0.4` with `totalNbrs + (totalNbrs - filteredNbrs)` and choose directed two-hop or full two-hop. Seed up to `K` filtered nodes into candidates to handle negative correlation.

- [ ] **Step 4: Run test to verify it passes**

Run: `cmake --build build_ung_rel -j 8 --target test_navix_search && ./build_ung_rel/test/test_navix_search`
Expected: PASS.

### Task 4: App and Experiment Wiring

**Files:**
- Modify: `UNG/codes/apps/search_UNG_index.cpp`
- Modify: `experiments/search_comparison/run_search_comparison.py`
- Test: `experiments/search_comparison/test_run_search_comparison.py`

**Interfaces:**
- Consumes: `force_use_alg=5`.
- Produces: config-driven methods can choose NaviX without ACORN index paths.

- [ ] **Step 1: Write the failing test**

Add a Python config fixture with a method named `NaviX`, `force_use_alg: 5`, and no ACORN index path. Assert the generated command contains `--force_use_alg 5` and does not require ACORN paths.

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest experiments/search_comparison/test_run_search_comparison.py -q`
Expected: FAIL on missing expected NaviX behavior if current script enforces ACORN assumptions.

- [ ] **Step 3: Write minimal implementation**

Document `force_use_alg=5` in the option help and keep runner ACORN paths optional for NaviX.

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest experiments/search_comparison/test_run_search_comparison.py -q`
Expected: PASS.

### Task 5: Verification

**Files:**
- All modified C++ and Python files.

**Interfaces:**
- Consumes: all previous tasks.
- Produces: verified build/test status.

- [ ] **Step 1: Build focused targets**

Run: `cmake --build build_ung_rel -j 8 --target test_navix_hnsw test_navix_search search_UNG_index build_UNG_index`

- [ ] **Step 2: Run focused C++ tests**

Run: `./build_ung_rel/test/test_navix_hnsw && ./build_ung_rel/test/test_navix_search`

- [ ] **Step 3: Run Python runner tests**

Run: `pytest experiments/search_comparison/test_run_search_comparison.py -q`

- [ ] **Step 4: Inspect diff**

Run: `git status --short && git diff --stat`

