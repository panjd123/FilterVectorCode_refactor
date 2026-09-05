# Special Block Trie Entry Provider Design

## Goal

Persist the trie used by `special_block_partition=trie` and query it directly for
special-block entry groups. This path replaces the existing LNG/group ELS provider
without changing how graph search decides whether an entry or neighbor is free.

## Scope

The new provider is selected explicitly with:

```text
UNG_SPECIAL_BLOCK_SEARCH=1
--entry_group_provider special_block_trie
```

It is valid only for indexes built with the trie special-block partition. Ordinary
UNG, LNG-partitioned special blocks, ACORN, and the existing ELS providers retain
their current behavior.

## Persistent Index

`SpecialBlockTrieIndex` owns a read-only flat representation of the construction
trie. Each node stores:

```cpp
struct SpecialBlockTrieNode {
   LabelType label;
   uint32_t parent_id;
   uint32_t first_child;
   uint32_t child_count;
   IdxType terminal_group_id;
   IdxType block_id;
};
```

Children are stored in a separate flat node-ID array and sorted by label. An
in-memory `label -> node IDs` posting table is built when the index is constructed
or loaded. The root is node zero and has no terminal group or block ID.

The special-block partitioner uses this same index rather than constructing a
second temporary trie. Point counts, subtree counts, and uncovered counts remain
transient partition state because they are not needed during query execution.
When a trie node becomes a block root, its persisted `block_id` is set.

Each `SpecialBlock` also persists `entry_point_id`, the global point ID of the
entry returned by the block-local Vamana/Tagore graph. The field is appended to
`special_blocks.csv`; loaders accept the old eight-column schema and leave the
entry invalid, while newly built Trie indexes require a valid entry before the
Trie regular path is used.

The index is saved as `special_block_trie.bin`. The header contains a fixed magic,
format version, node count, child-reference count, group count, and a checksum of
the serialized node and child arrays. Loads reject a missing file, wrong version,
invalid parent/child references, duplicate or out-of-range terminal groups, invalid
block IDs, and checksum mismatches.

## Query Semantics

Query labels are expected to be strictly increasing, matching storage input. The
query path uses them directly and asserts this invariant in debug builds; it does
not sort or allocate a normalized copy per query.

For a non-empty query:

1. Select `query_labels.back()` as the pivot.
2. Read every trie node in that label posting.
3. Walk from the pivot node toward the root and verify that all earlier query
   labels occur on the path; extra path labels are allowed.
4. Start a downward traversal from every verified pivot node.
5. When a terminal node is reached, emit its group ID and stop traversing that
   branch.
6. Continue through non-terminal children.

Because a strictly increasing root-to-node path can contain the pivot label only
once, nodes in one pivot posting are incomparable and their descendant subtrees do
not overlap. The validated index also contains each terminal group exactly once,
so the hot path does not allocate a full-index visited array.

There is deliberately no cross-branch minimal-superset filter. If `{1,3}` and
`{1,2,3}` are the first terminals on different structural branches for `Q={1}`,
both are returned. Descendants of either terminal are not returned.

The query provider itself does not inspect block free state. Its output remains an
ordinary `entry_group_ids` vector. During special graph search, the Trie provider
uses the terminal-to-nearest-block-ancestor relation and the persisted
`entry_point_id` as a conditional portal: an initial covered terminal or the first
regular neighbor entering a covered block inserts that block entry as an
additional free candidate. The portal is disabled for the legacy LNG A/B path and
can be disabled explicitly with `UNG_SPECIAL_TRIE_DISABLE_BLOCK_PORTALS=1`.
For low-Lsearch workloads, `UNG_SPECIAL_TRIE_BLOCK_PORTAL_MIN_LSEARCH` can defer
the transition until the search budget is large enough to benefit from another
block entry.
Uncovered blocks never receive this transition, so the portal cannot introduce
points that fail containment.

Empty queries return no entries. Unknown pivot labels also return no entries.

Search applications precompute the result for each unique query-label key before
measured search and retain the group IDs, route statistics, and Trie traversal
statistics in a single-flight cache. This matches the existing CPU ELS benchmark
policy and avoids repeating the same Trie traversal for every Lsearch value.

## Provider Integration

Add `SpecialBlockTrie` to both public provider enums. The provider records elapsed
time in the existing ELS timing field and maps trie counters to the existing query
statistics:

- pivot posting size -> candidate count;
- successful upward checks -> successful checks;
- upward and downward visits -> trie nodes traversed;
- emitted terminal count -> entry group count through existing route statistics.

Selecting the provider without special-block search, without a loaded trie index,
or against an LNG-partitioned index is a configuration error. There is no silent
fallback to `_trie_index`, LNG descendants, `cpu_bruteforce_els`, or another
provider.

## Metrics

Build and load logs report:

- node and child-reference counts;
- serialized index bytes;
- trie construction time;
- trie save and load time.
- persisted block-entry count and implicit Trie-to-block portal relations.

Per-query statistics report pivot postings, upward checks, nodes visited, terminal
candidates, final entries, and elapsed time through the provider result and the
existing query-detail CSV fields.

## Testing

Focused tests cover:

- first-terminal pruning on one branch;
- retaining structurally distinct terminals without global minimal filtering;
- multi-label upward containment with extra labels on the path;
- empty and unknown-label queries;
- block-root metadata round trip;
- binary save/load equivalence;
- rejection of truncated or corrupt files;
- provider parsing and runtime configuration validation.

After focused tests, build a trie-partitioned Amazon index and compare
`cpu_bruteforce_els` with `special_block_trie` on
`query_selected_recall_advantage`, recording build/load size and time, ELS time,
entry counts, free/regular entries, total search time, QPS, and recall. Include a
same-topology `UNG_SPECIAL_TRIE_DISABLE_BLOCK_PORTALS=1` ablation so portal recall
gains and costs are measured independently of the legacy LNG topology.
