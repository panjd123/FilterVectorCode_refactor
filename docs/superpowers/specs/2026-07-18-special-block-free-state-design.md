# Special-block free-state activation for virtual trie roots

## Goal

Make a loaded special-block sidecar participate in containment search when a
block is rooted at a virtual label-trie prefix.  A virtual prefix may not have
an exact vector group, so the current `root_group_id == 0` representation never
creates a free entry or free neighbour and consequently never scans a special
edge.

## Scope and invariants

- The change applies only to containment semantics.  Equality and overlap
  searches must not inherit this rule without their own proof of validity.
- A point becomes free only when its owning group belongs to a block whose
  `root_labels` contain every query label.
- A group/point belongs to at most one active block: the existing recursive
  block collector stops at child blocks, so the persisted member mapping is the
  innermost-block mapping.
- A free point may scan the sidecar edges already associated with it.  It does
  not make unrelated regular points or unrelated blocks free.
- Existing indexes with `root_group_id=0` remain loadable and gain the repaired
  activation behaviour; rebuilding produces an explicit nonzero anchor.

## Data model

`root_labels` remains the possibly virtual trie prefix used for coverage.
`root_group_id` is replaced in newly written metadata by `anchor_group_id`: a
deterministic real member group (the smallest member ID) used for integrity and
compatibility diagnostics, not for free-state activation.  A block with no
member group is invalid and must be rejected during construction/load.

The in-memory group-to-block and point-to-block maps remain the authoritative
membership relation.  They must be used at both activation sites.

## Search flow

For each query, derive `query_covers_block[block_id]` from `root_labels`.

1. When adding an entry group, resolve its block through
   `_group_id_to_special_block`.  Mark its entry points free only when that
   block is covered.
2. When examining a regular graph neighbour, resolve its block through
   `_point_to_special_block`.  Promote it to free only when that block is
   covered.
3. A free candidate scans its sidecar edges.  With
   `UNG_SPECIAL_BLOCK_FREE_USE_REGULAR` unset it does not also scan regular
   edges, preserving the current free-state policy.

This makes virtual coverage prefixes usable without treating a nonexistent
group as a search entry.

## Tests

Add a focused C++ regression test using a small containment index with a
virtual-prefix block and real member groups.  It must demonstrate:

1. A covered query produces free entries/expansions and scans a sidecar edge.
2. A non-covered query has zero free and special-edge counters.
3. Nested blocks map a member only to its innermost block and do not cross-scan
   a parent sidecar.
4. Save/load preserves a real anchor for newly built indexes and accepts old
   root-zero metadata without disabling membership-based activation.

## End-to-end acceptance

Rebuild the Genome special-block index and run `query_minlen1_cov10k` with
special search enabled.  At least one query that covers a block must have
nonzero `SpecialEntryFreePoints`, `SpecialFreeDistCalcs`, and
`SpecialEdgesScanned`.  Compare recall to the current run at each `Lsearch`;
any material degradation is a failure requiring investigation rather than a
silent acceptance.
