# Composable Group Topology and Hierarchy

The implementation separates three choices that were previously coupled:

1. **Hierarchy plan**: zero or more materialized layers and the minimum point
   threshold of each layer.
2. **Group topology**: `lng` or `trie`, selected independently for the base
   graph and for every materialized layer.
3. **Entry-group strategy**: exactly one of `original`, `optimized_lng`, or
   `trie` at query time.

An experiment must report all three choices. Naming only the entry strategy is
not enough to identify the graph being measured.

## Configuration

The base topology is selected while building the UNG index:

```bash
UNG_BASE_GROUP_TOPOLOGY=trie build_UNG_index ...
```

The default is `lng`. A zero-layer experiment leaves `UNG_HIERARCHY_LAYERS`
empty and does not load a block sidecar.

Additional layers use the grammar
`<min_points>:<topology>[,<min_points>:<topology>...]`. Thresholds must be
positive and strictly increasing from fine to coarse. For example:

```bash
UNG_HIERARCHY_LAYERS='1000:trie,16000:lng,400000:trie' \
build_special_block_index \
  --ung_index_path_prefix /path/to/ung/index_files/ \
  --block_index_path_prefix /path/to/hierarchy/index_files/ \
  --result_path_prefix /path/to/hierarchy/results/ \
  --data_type float \
  --dist_fn L2 \
  --num_threads 60 \
  --min_points 1000 \
  --max_degree 32 \
  --num_cross_edges 4 \
  --Lbuild 100 \
  --alpha 1.2
```

`--min_points` remains required as the legacy first-threshold input. When
`UNG_HIERARCHY_LAYERS` is set, the explicit plan is authoritative. The older
`UNG_SPECIAL_BLOCK_MIN_POINTS`, `UNG_SPECIAL_BLOCK_UPPER_MIN_POINTS`, and
`UNG_SPECIAL_BLOCK_PARTITION` settings are translated to a one- or two-layer
plan for compatibility.

At search time, select entry discovery independently:

```bash
search_UNG_index \
  --index_path_prefix /path/to/ung/index_files/ \
  --block_index_path_prefix /path/to/hierarchy/index_files/ \
  --entry_group_strategy trie \
  ...
```

`--entry_group_provider` is a deprecated spelling of
`--entry_group_strategy`. Historical provider values are accepted only as
aliases for one of the three strategies.

## Search Invariant

Each candidate carries an activation level. A level-zero candidate scans only
the persisted base graph. A candidate at materialized level `i` scans only
edges whose owner is level `i`; it never scans the base graph, falls through
to a lower layer, or promotes itself to a higher layer. Every query-authorized
layer is seeded independently, and the candidates share one bounded queue.

Visited state is separate per activation level. Therefore the same vector may
legitimately be explored once in the base graph and once in each independently
seeded materialized layer without suppressing another layer's traversal.

## Fair Zero-Layer Comparison

Use no block sidecar and compare at the same Recall target:

| Name | Base topology | Entry strategy |
|---|---|---|
| LNG-0 | `lng` | `optimized_lng` |
| Trie-0 | `trie` | `trie` |

Keep the dataset, query set, ground truth, thread count, graph build settings,
search backend, and binary identical. For each method report the smallest
measured `Lsearch` that reaches the Recall target and the complete batch wall
time. This comparison changes both group connectivity and entry discovery by
design; a component-level ablation should vary only one column at a time.

## Persistence

The UNG `meta` file records `base_group_topology` and `hierarchy_layers`. The
independent hierarchy sidecar records the same plan, its source-index
fingerprint, block metadata, and per-layer edges. Loading rebuilds the terminal
group Trie from group labels and reconstructs the selected base topology, so
freshly built and reloaded indexes expose the same three entry strategies.

Special edges use CSR v2. Source IDs are represented by the offset table;
edges store a 32-bit target plus a packed block/kind word. The loader remains
compatible with legacy edge files. CSV diagnostics are disabled by default;
set `UNG_SPECIAL_EDGE_BINARY_ONLY=0` for edge CSV files and
`UNG_SPECIAL_BLOCK_METADATA_CSV=1` for block metadata CSV files.
