# FAVOR Block Search Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an experimental UNG search mode that uses ELS-derived approximate block space plus FAVOR-style selectivity-aware exclusion distances, while returning only label-valid TD vectors.

**Architecture:** Keep existing `special_free` behavior intact. Add a runtime search mode `UNG_SPECIAL_SEARCH_MODE=favor_blocks` that builds a TD bitmap from ELS coverage, maps ELS entry groups to their special blocks and child blocks, searches those block/sidecar regions with adjusted distances for NTD points, and filters final results to TD only.

**Tech Stack:** C++17, existing UNG search backend, CRoaring, CMake direct test executables.

## Global Constraints

- Do not change existing `special_free` default behavior.
- Experimental mode must be enabled explicitly by `UNG_SPECIAL_SEARCH_MODE=favor_blocks`.
- TD/NTD membership must come from existing croaring coverage of ELS groups.
- Final result queue must contain TD points only.
- The first implementation may use exact current distance for both TD and NTD, then add penalty to NTD.

---

### Task 1: FAVOR Distance Helper

**Files:**
- Create: `UNG/codes/include/ung_favor_block_search.h`
- Modify: `UNG/codes/test/CMakeLists.txt`
- Create: `UNG/codes/test/test_favor_block_search.cpp`

**Interfaces:**
- Produces: `float favor_exclusion_distance(float selectivity, float lsearch, float delta_d, float alpha)`.
- Produces: `float favor_adjusted_distance(float base_distance, bool is_target, float exclusion_distance)`.

- [ ] Write failing test for NTD penalty and selectivity monotonicity.
- [ ] Build `test_favor_block_search` and verify it fails because the header is absent.
- [ ] Add the helper header.
- [ ] Register and run CTest.

### Task 2: Runtime Mode Wiring

**Files:**
- Modify: `UNG/codes/include/ung_query_route.h`
- Modify: `UNG/codes/src/uni_nav_graph_query_route.cpp`

**Interfaces:**
- Produces: `SpecialSearchMode special_search_mode` in `SearchRuntimeConfig`.
- Consumes env `UNG_SPECIAL_SEARCH_MODE`, values `free_state` and `favor_blocks`.

- [ ] Add enum and parser.
- [ ] Default to `free_state` for compatibility.
- [ ] Compile `search_UNG_index`.

### Task 3: FAVOR Block Search Backend

**Files:**
- Modify: `UNG/codes/include/uni_nav_graph.h`
- Modify: `UNG/codes/include/ung_query_stats.h`
- Modify: `UNG/codes/src/uni_nav_graph_search_backend.cpp`
- Modify: `UNG/codes/apps/search_UNG_index.cpp`

**Interfaces:**
- Produces: `execute_favor_block_ung_query(...)`.
- Uses `compute_bitmap_from_groups(entry_group_ids)` for TD bitmap.
- Uses `_group_id_to_special_block` and `child_block_ids` for approximate block space.

- [ ] Build `target_bitmap` and `filter_map` once per query.
- [ ] Select approximate blocks from ELS groups and child blocks up to `UNG_FAVOR_BLOCK_DEPTH`.
- [ ] Seed entries from ELS groups, marking no candidate as automatically free.
- [ ] Use adjusted distance for queue ordering.
