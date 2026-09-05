# Special-block Free-state Activation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Activate special-block free-state traversal for containment queries whose block coverage root is a virtual trie prefix.

**Architecture:** Keep `root_labels` as the coverage predicate and use the existing persisted group/point-to-block maps to decide whether an entry or regular neighbour is free.  Persist a real member-group anchor for newly built blocks, but never use a single anchor as the runtime activation gate.

**Tech Stack:** C++17, CMake, UNG `UniNavGraph`, CTest, existing Genome benchmark runner.

## Global Constraints

- Apply membership-based free activation only to `containment` queries.
- A point is free only if its mapped block is covered by the query's `root_labels`.
- Keep child-block member ownership exclusive; do not activate a parent block through a child member.
- Do not overwrite the existing dirty worktree files.
- Do not accept recall degradation without investigating it.

---

### Task 1: Add a focused failing free-state regression executable

**Files:**
- Create: `UNG/codes/test/test_special_block_free_state.cpp`
- Modify: `UNG/codes/test/CMakeLists.txt`

**Interfaces:**
- Consumes: `UniNavGraph::execute_special_block_ung_query`, `QueryStats`, and a tiny temporary containment index.
- Produces: CTest target `test_special_block_free_state` with two assertions: covered member activates free/sidecar traversal; uncovered member does not.

- [ ] **Step 1: Write the failing test**

Create a four-point containment fixture whose block has virtual `root_labels={1}`, real member groups `{1,2}`, and a sidecar edge from member-point 0 to member-point 1.  Execute one query with labels `{1}` and assert:

```cpp
EXPECT_GT(stats.special_entry_free_points, 0U);
EXPECT_GT(stats.special_free_nodes_expanded, 0U);
EXPECT_GT(stats.special_edges_scanned, 0U);
```

Execute `{9}` against the same fixture and assert:

```cpp
EXPECT_EQ(stats.special_entry_free_points, 0U);
EXPECT_EQ(stats.special_free_nodes_expanded, 0U);
EXPECT_EQ(stats.special_edges_scanned, 0U);
```

The fixture must use a block metadata row with legacy `root_group_id=0`; this proves old Genome artifacts are repaired at runtime rather than only after a rebuild.

- [ ] **Step 2: Register and run the test red**

Add:

```cmake
add_executable(test_special_block_free_state test_special_block_free_state.cpp)
target_link_libraries(test_special_block_free_state PRIVATE -Wl,--whole-archive ${PROJECT_NAME} Vamana -Wl,--no-whole-archive Boost::program_options Boost::filesystem OpenMP::OpenMP_CXX ${ROARING_LIB} onnxruntime faiss)
add_test(NAME special_block_free_state COMMAND test_special_block_free_state)
```

Run:

```bash
cmake --build build_ung_rel --target test_special_block_free_state -j8
ctest --test-dir build_ung_rel --output-on-failure -R special_block_free_state
```

Expected: the covered-query assertion fails because the current root-only gate leaves all special counters at zero.

### Task 2: Make containment activation membership-based

**Files:**
- Modify: `UNG/codes/src/uni_nav_graph_search_backend.cpp`

**Interfaces:**
- Consumes: `_group_id_to_special_block`, `_point_to_special_block`, `query_covers_block`, and `runtime.scenario`.
- Produces: covered entry members and covered regular neighbours as `free` candidates.

- [ ] **Step 1: Add a local coverage predicate**

Immediately after `query_covers_block` is built, add a lambda with bounds checks:

```cpp
auto block_is_covered = [&](IdxType block_id) {
    return runtime.scenario == "containment" && block_id > 0 &&
           block_id < query_covers_block.size() && query_covers_block[block_id] != 0;
};
```

- [ ] **Step 2: Replace the entry root-only gate**

Replace:

```cpp
const IdxType root_block_id = is_special_block_root_group(group_id) ? _group_id_to_special_block[group_id] : 0;
const bool covered_root = root_block_id > 0 && root_block_id < query_covers_block.size() && query_covers_block[root_block_id] != 0;
```

with:

```cpp
const IdxType block_id = group_id < _group_id_to_special_block.size()
                               ? _group_id_to_special_block[group_id]
                               : 0;
const bool covered_root = block_is_covered(block_id);
```

- [ ] **Step 3: Replace the neighbour root-point gate**

Replace the `is_special_block_root_point(neighbor)` condition with a lookup of
`_point_to_special_block[neighbor]`, then set `next_free = true` only when
`block_is_covered(block_id)` is true.

- [ ] **Step 4: Run the regression green**

Run the Task 1 build and CTest command.  Expected: both assertions pass; the uncovered query still has zero special counters.

- [ ] **Step 5: Commit**

```bash
git add UNG/codes/src/uni_nav_graph_search_backend.cpp UNG/codes/test/CMakeLists.txt UNG/codes/test/test_special_block_free_state.cpp
git commit -m "fix: activate special blocks through covered members"
```

### Task 3: Persist and validate a real member anchor

**Files:**
- Modify: `UNG/codes/include/ung_special_blocks.h`
- Modify: `UNG/codes/src/uni_nav_graph_special_blocks.cpp`
- Modify: `UNG/codes/test/test_special_block_free_state.cpp`

**Interfaces:**
- Consumes: each block's sorted `member_group_ids`.
- Produces: nonzero persisted `anchor_group_id` for newly built blocks; legacy root-zero files remain loadable.

- [ ] **Step 1: Extend the test with save/load checks**

Build a virtual-prefix block, save it, reload it, and assert its persisted
anchor is nonzero and belongs to `member_group_ids`; repeat the covered-query
assertions after reload.

- [ ] **Step 2: Run the new test red**

Run the Task 1 CTest command.  Expected: it fails because the current CSV
stores `root_group_id=0` for virtual-prefix blocks.

- [ ] **Step 3: Add the anchor field and write it**

Add `IdxType anchor_group_id = 0;` to `SpecialBlock`.  After member collection
and sorting, set it to `member_group_ids.front()`; reject a block with no
members.  Write `anchor_group_id` as the second `special_blocks.csv` column
with header `anchor_group_id`.

- [ ] **Step 4: Load both schemas compatibly**

When loading, accept `root_group_id` or `anchor_group_id` headers.  Treat a
legacy zero value as allowed; after members load, derive `anchor_group_id` from
the first member when absent.  Runtime activation remains map-based, so this
compatibility path cannot reintroduce the root-only failure.

- [ ] **Step 5: Run green and commit**

```bash
cmake --build build_ung_rel --target test_special_block_free_state -j8
ctest --test-dir build_ung_rel --output-on-failure -R special_block_free_state
git add UNG/codes/include/ung_special_blocks.h UNG/codes/src/uni_nav_graph_special_blocks.cpp UNG/codes/test/test_special_block_free_state.cpp
git commit -m "fix: persist real special-block anchors"
```

### Task 4: Rebuild and validate Genome end-to-end

**Files:**
- No source changes.
- Create: a timestamped directory beneath `FilterVectorResult/Genome/index/` and `FilterVectorResult/Genome/results/`.

**Interfaces:**
- Consumes: rebuilt special-block index and `query_minlen1_cov10k` workload.
- Produces: new `query_details_repeat1.csv` and a comparison to the existing baseline.

- [ ] **Step 1: Build affected binary and unit suite**

```bash
cmake --build build_ung_rel --target build_UNG_index search_UNG_index test_special_block_free_state -j8
ctest --test-dir build_ung_rel --output-on-failure -R special_block_free_state
```

- [ ] **Step 2: Rebuild the special-block Genome index**

Reuse the existing Genome build command from
`FilterVectorResult/Genome/index/UNG_special_blocks/others/build.log`, changing
only output prefixes to a timestamped directory and retaining
`UNG_SPECIAL_BLOCKS=1`, `UNG_SPECIAL_BLOCK_DATA_MODE=x1`, and
`UNG_SPECIAL_BLOCK_MIN_POINTS=1000`.

- [ ] **Step 3: Run the recorded query workload**

Reuse the command in
`FilterVectorResult/Genome/results/gpu_bruteforce_els_special_blocks/query_minlen1_cov10k/others/Genome_search_output.txt`, set `UNG_SPECIAL_BLOCK_SEARCH=1`, and write to a timestamped results directory.

- [ ] **Step 4: Check activation and recall**

For every `Lsearch`, compare new versus baseline average recall.  Also require:

```bash
awk -F, 'NR==1 {for(i=1;i<=NF;i++) h[$i]=i; next} $(h["SpecialEntryFreePoints"])>0 && $(h["SpecialEdgesScanned"])>0 {ok=1} END {exit ok?0:1}' results/query_details_repeat1.csv
```

Expected: exit 0, demonstrating actual free-state sidecar traversal.  If any
recall delta is material, stop and investigate the affected queries before
claiming success.
