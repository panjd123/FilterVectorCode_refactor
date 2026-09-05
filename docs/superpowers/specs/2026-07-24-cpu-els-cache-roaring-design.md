# CPU ELS Cache And Roaring Design

## Goal

Optimize the `cpu_bruteforce_els` entry-group provider in UNG search by removing per-query reconstruction of label and label-size bit indexes, and use existing CRoaring descendant bitmaps when available.

## Current Behavior

`UniNavGraph::compute_cpu_bruteforce_entry_groups_for_execution()` rebuilds these structures for every query:

- `label_group_bits`: groups containing each label.
- `size_group_bits`: groups with each label-set size.
- `max_group_label_size`.

It then computes candidate groups, selects a small frontier, computes descendants covered by that frontier, and emits:

```cpp
selected_frontier | (candidates - covered)
```

The GPU `GpuCoverFrontierProvider` performs the same high-level algorithm, but builds label, size, and descendant bitsets once in its provider constructor and reuses them for all queries.

## Design

Add a CPU-side lazy cache owned by `UniNavGraph`:

- Flat dense `std::vector<uint64_t>` for `label_group_bits`.
- Flat dense `std::vector<uint64_t>` for `size_group_bits`.
- `label_to_dense` mapping for compact label rows.
- `words_per_query`, `num_groups_including_zero`, and `max_group_label_size`.
- A mutex and validity flag so parallel query threads initialize the cache once.

The CPU brute-force ELS provider will:

1. Ensure the cache is initialized.
2. Compute `candidate_bits` by ANDing cached dense label rows.
3. Compute `frontier_bits` by ORing cached size rows for `[query_len, query_len + 2]`.
4. Compact up to `8192` frontier ids.
5. Prefer `_lng_descendants_rb` to compute `covered` with CRoaring OR operations.
6. Fall back to the existing vector-descendant bit-setting path if Roaring descendants are unavailable.

## Roaring Path

When `_lng_descendants_rb.size() > _num_groups`, convert per-query dense `candidate_bits` and `selected_frontier_bits` to `roaring::Roaring`, OR the selected frontier descendants, subtract covered descendants from candidates, and emit the result Roaring contents.

This avoids a dense `O(num_groups^2 / 8)` descendant matrix on CPU while still accelerating descendant union work.

## Compatibility

The public CLI and provider enum stay unchanged. Existing saved indexes continue to work because the CPU cache is derived from loaded `_group_id_to_label_set` and existing `_lng_descendants_rb` if present.

## Testing

Add or update focused tests for CPU brute-force ELS equivalence on a small synthetic graph:

- Cached CPU brute-force output matches the previous formula.
- The Roaring path and vector fallback produce identical group ids.
- Queries containing an unknown label produce no groups.
