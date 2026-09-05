# CPU ELS Cache Roaring Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Cache CPU brute-force ELS label/size bit indexes and use CRoaring descendant unions when available.

**Architecture:** Add a lazy CPU cache to `UniNavGraph` that stores dense label and label-size group bitsets derived from `_group_id_to_label_set`. Rewrite `compute_cpu_bruteforce_entry_groups_for_execution()` to reuse that cache and prefer `_lng_descendants_rb` for descendant coverage while preserving the existing vector fallback.

**Tech Stack:** C++17, OpenMP search runtime, CRoaring C++ wrapper, existing UNG provider interfaces.

## Global Constraints

- Keep the public CLI and provider enum unchanged.
- Do not add new dependencies; CRoaring is already included by the project.
- Avoid dense CPU descendant bitsets because memory is `O(num_groups^2 / 8)`.
- Preserve vector descendant fallback when `_lng_descendants_rb` is unavailable.

---

### Task 1: Add CPU ELS Cache State

**Files:**
- Modify: `UNG/codes/include/uni_nav_graph.h`
- Modify: `UNG/codes/src/uni_nav_graph_entry_provider.cpp`

**Interfaces:**
- Produces: `struct CpuBruteForceElsCache` with `valid`, `num_groups_including_zero`, `words_per_query`, `max_group_label_size`, `label_to_dense`, `label_group_bits`, and `size_group_bits`.
- Produces: `void ensure_cpu_bruteforce_els_cache() const`.
- Consumes: `_group_id_to_label_set`, `_num_groups`.

- [ ] **Step 1: Add declarations**

Add a private cache struct and mutable fields in `UniNavGraph`.

- [ ] **Step 2: Implement cache initialization**

Build dense flat rows:

```cpp
label_group_bits[dense_label * words_per_query + word]
size_group_bits[label_size * words_per_query + word]
```

- [ ] **Step 3: Guard initialization**

Use a mutex and a `valid` flag so parallel search threads build the cache once.

### Task 2: Use Cache And CRoaring In CPU ELS

**Files:**
- Modify: `UNG/codes/src/uni_nav_graph_entry_provider.cpp`

**Interfaces:**
- Consumes: `ensure_cpu_bruteforce_els_cache() const`.
- Consumes: `_lng_descendants_rb`.
- Produces: unchanged `EntryGroupProviderResult compute_cpu_bruteforce_entry_groups_for_execution(...)`.

- [ ] **Step 1: Replace per-query index construction**

Use cached `label_group_bits`, `size_group_bits`, and `max_group_label_size`.

- [ ] **Step 2: Add Roaring conversion helpers**

Convert dense `std::vector<uint64_t>` query bitsets into `roaring::Roaring` by iterating set bits.

- [ ] **Step 3: Prefer Roaring descendant coverage**

When `_lng_descendants_rb.size() > _num_groups`, compute:

```cpp
covered_rb |= _lng_descendants_rb[gid];
result_rb = selected_frontier_rb;
result_rb |= (candidate_rb - covered_rb);
```

- [ ] **Step 4: Preserve vector fallback**

Keep existing descendant vector logic for indexes where Roaring descendants are not initialized.

### Task 3: Verify Build And Behavior

**Files:**
- Test existing build target under `UNG/codes`.

**Interfaces:**
- Consumes: existing CMake project.
- Produces: compile verification for changed C++ code.

- [ ] **Step 1: Configure or reuse build directory**

Use the existing project build workflow if present.

- [ ] **Step 2: Build affected target**

Build the search app or library target that compiles `uni_nav_graph_entry_provider.cpp`.

- [ ] **Step 3: Report residual risk**

If a full runtime ELS equivalence test cannot be run without datasets, report that clearly.
