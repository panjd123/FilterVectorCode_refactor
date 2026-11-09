# CPU Special Blocks Build Experiment

This experiment builds UNG indexes with:

- `UNG_GROUP_GRAPH_IMPL=0`
- CPU small-group complete-graph acceleration
- special blocks enabled
- special block minimum uncovered points threshold: `UNG_SPECIAL_BLOCK_MIN_POINTS=1000`
- CPU Vamana cross edges and CPU Vamana additional edges

Run from the repository root:

```bash
./run_cpu_special_blocks_experiment.sh
```

Use a different config:

```bash
./run_cpu_special_blocks_experiment.sh experiments/cpu_special_blocks/config.json
```

Outputs are written under `result_root/run_name_timestamp/` for the default run-root layout, or under `result_root/<Dataset>/index/<index_name>/` when `output_layout` is `index_by_dataset`.
Each dataset directory contains `index_files/`, `results/`, `others/build.log`, and `others/time.txt`. The runner does not write config-directory run logs or summary CSV files.

## Trie Block Without UNG Rebuild

Set `"build_ung": false` to skip `build_UNG_index`. The block builder reuses
the required UNG files when available. When any required source file or
metadata is unavailable, it automatically rebuilds its block-only input from
`<data_root>/<Dataset>/<Dataset>_base.bin` and `<data_root>/<Dataset>/<Dataset>_base_labels.txt`.
This fallback builds only the independent Trie block sidecar; it does not save
a full UNG index under `index_files/`.

To reuse a specific existing base index instead of the output directory's
`index_files/`, set:

```json
"build_ung": false,
"ung_index_path_prefix": "/path/to/base_trie/index_files",
"require_ung_index": true,
"required_base_topology": "trie"
```

`require_ung_index=true` disables the silent dataset fallback by requiring the
source `meta`, vectors, labels, group labels, and group ranges to exist.
`required_base_topology=trie` additionally rejects an LNG base index. The
optional `{dataset}` token in `ung_index_path_prefix` is replaced with the
normalized dataset name.

## Self-Built All-Trie Multilevel Blocks With GPU Vamana

`Amazon_config_Multilevel_Trie_block.json` builds a fresh Amazon base index
from the dataset files and then builds its independent multilevel sidecar. It
does not reuse a prebuilt index. When the base topology is `trie`,
`build_UNG_index` now uses the terminal-group Trie topology directly and skips
`build_label_nav_graph`, so no LNG topology is constructed. The base `meta`
records `lng_constructed=0` and `build_LNG_time(ms)=0`. Its hierarchy is:

```text
base topology: trie
level 0:       1000:trie
level 1:       16000:trie
```

The corresponding environment is:

```json
"UNG_BASE_GROUP_TOPOLOGY": "trie",
"UNG_HIERARCHY_LAYERS": "1000:trie,16000:trie",
"UNG_SPECIAL_BLOCK_PARTITION": "trie"
```

The runner executes these stages in order:

1. Configure CUDA 12.4 for SM 89 and compile the UNG/base, block, and search applications.
2. Run `build_UNG_index` from the Amazon vectors and labels into this run's `index_files/`.
3. Verify that the new base index metadata says `base_group_topology=trie`.
4. Run `build_special_block_index` against that newly built base index.

The base Trie uses CPU group graphs and does not build an LNG. Descendants,
coverage sets, selector statistics, and the vector-attribute bipartite graph
were removed on 2026-10-19. Cross-group edges use the CPU hybrid route:

```json
"num_threads": 60,
"UNG_GROUP_GRAPH_COMPLETE_NX": "32",
"UNG_CROSS_EDGE_IMPL": "4",
"UNG_CPU_TARGET_EXACT_MAX_NX": "1000",
"UNG_CPU_HYBRID_SCAN_MAX_WORK": "10000",
"UNG_TRIE_CROSS_SOURCE_EXACT": "1",
"UNG_ADDITIONAL_EDGES_IMPL": "2"
```

Groups with at most 32 points build a complete intra-group graph rather than a
Vamana graph, matching the base `max_degree=32`. The two hybrid thresholds are
retained as the fallback policy when `UNG_TRIE_CROSS_SOURCE_EXACT=0`; they do
not split work in the active source-exact Trie route.

For a Trie base, `UNG_TRIE_CROSS_SOURCE_EXACT=1` replaces the legacy
target-pair loop with one source-centric exact pass. Each source point is
scheduled once, scans the union of all child-group ranges, and writes exact
global top-k cross edges directly into its preallocated output queue. This is
equivalent to merging exact per-child top-k results, but removes per-query
temporary queues, Vamana/SearchCache lookup, and per-relation OpenMP team
creation. Missing topology relations are repaired with exact CPU additional
edges (`impl=2`); singleton targets are appended directly without a distance
calculation.

CUDA is mandatory for this config:

```json
"cuda": {
  "enabled": true,
  "required": true,
  "compiler": "/usr/local/cuda-12.4/bin/nvcc",
  "architectures": "89"
}
```

The block-local Vamana route is also fail-closed:

```json
"UNG_SPECIAL_BLOCK_GPU_INTRA": "1",
"UNG_SPECIAL_BLOCK_GPU_INTRA_STRICT": "1",
"UNG_SPECIAL_BLOCK_GPU_INTRA_REQUIRE_USED": "1",
"UNG_SPECIAL_INTRA_COMPLETE_NX": "128",
"UNG_SPECIAL_INTRA_ROUTE": "0",
"UNG_TAGORE_K": "128"
```

Blocks with at most 128 points use the exact small-block path. Larger blocks
must use the Tagore CUDA Vamana builder. A CUDA exception, CPU fallback, or a
run in which no block actually uses CUDA fails the build. The output block
`meta` records `gpu_intra_blocks`, `gpu_intra_points`,
`gpu_intra_fallback_blocks`, and the CPU Vamana counts for auditing.

The single-level `Amazon_config_Trie_block.json` also enables balanced block-local
GPU batches with `UNG_TAGORE_BATCH_STREAMS=4`. Eligible blocks are packed into
windows of at most `UNG_SPECIAL_BLOCK_GPU_INTRA_BATCH_POINTS=65536` points; a
single larger block runs by itself. Within a window, the builder sorts blocks by
point count and assigns each to the worker with the fewest accumulated points.
Each worker uses its own CUDA stream and reuses its workspace across blocks.
`[GPU_INTRA_BATCH]` and `[TagoreCuda][batch_balance]` log the actual batch size,
stream count, and worker point totals. Set `UNG_TAGORE_BATCH_STREAMS=1` to retain
serial block builds. Strict GPU failures still terminate the build.

Run the small GPU integration check (serial, balanced, bounded-window, and routed
large-block cases) with an available GPU ID, for example:

```bash
gpulock check 0 -- env \
  UNG_GPU_INTRA_TEST_BINARY="$PWD/build_ung_multilevel_trie_gpu/apps/build_special_block_index" \
  python3 tools/tests/test_special_block_gpu_intra.py
```

Run it from the repository root:

```bash
./run_cpu_special_blocks_experiment.sh \
  experiments/cpu_special_blocks/Amazon_config_Multilevel_Trie_block.json
```

The output is written below
`/home/dev/graphdb/FilterVectorResult/Amazon/index/` with the cross-block edge
count, total hierarchy level count (including the base Trie), and minimum point
threshold of every materialized block layer included in the directory name:

```text
Trie_multilevel_gpu_block_sourceexact_complete32_cross_edges-4-LAYERS-3-8192-13072/
  index_files/        # freshly built Trie base index
  block_index_files/  # multilevel block sidecar
  results/
  block_results/
  others/
```

To add a third layer, append another strictly larger threshold, for example:

```json
"UNG_HIERARCHY_LAYERS": "1000:trie,16000:trie,128000:trie"
```

`UNG_SPECIAL_BLOCK_MIN_POINTS` must remain equal to the first threshold because
the builder still accepts that legacy command-line value; when
`UNG_HIERARCHY_LAYERS` is present, the explicit hierarchy is authoritative.

## Index Name Parameter Suffix

`run_cpu_special_blocks_experiment.sh` supports an optional `index_name_params` list in the config. Each item appends one build/env value to the final index directory name, so different block settings do not overwrite each other under `FilterVectorResult/<Dataset>/index/`.

Example:

```json
"index_name": "UNG_special_blocks",
"index_name_params": [
  {"source": "env", "key": "UNG_SPECIAL_BLOCK_MAX_DEGREE", "label": "bdeg"},
  {"source": "env", "key": "UNG_SPECIAL_BLOCK_NUM_CROSS_EDGES", "label": "bcross"}
]
```

With the default config this produces an index name like:

```text
UNG_special_blocks_bdeg32_bcross4
```

## Block Degree Sweep

Use `special_block_max_degree_values` to build several independent block
indexes while keeping the other settings fixed. The runner appends
`_<min_points>_M_<degree>` to each index name. For example:

```json
"index_name": "Trie_block",
"special_block_max_degree_values": [16, 32, 64],
"ung_env": {
  "UNG_SPECIAL_BLOCK_MIN_POINTS": "100",
  "UNG_SPECIAL_BLOCK_MAX_DEGREE": "64"
}
```

This produces `Trie_block_100_M_16`, `Trie_block_100_M_32`, and
`Trie_block_100_M_64`. The scalar `UNG_SPECIAL_BLOCK_MAX_DEGREE` remains the
fallback when no sweep array is present. A ready-to-run Amazon example is
`Amazon_config_Trie_block_M_sweep.json`:

```bash
./run_cpu_special_blocks_experiment.sh experiments/cpu_special_blocks/Amazon_config_Trie_block_M_sweep.json
```

## Cross-Block Edge Sweep

Use `special_block_num_cross_edges_values` to vary the number of edges per
same-layer block-topology relation while keeping the block threshold and
block-local maximum degree fixed. The runner appends
`_<min_points>_M_<degree>_cross_block_<edges>` to each index name.

For example:

```json
"index_name": "Trie_block",
"special_block_num_cross_edges_values": [2, 4, 6],
"ung_env": {
  "UNG_SPECIAL_BLOCK_MIN_POINTS": "100",
  "UNG_SPECIAL_BLOCK_MAX_DEGREE": "64"
}
```

This produces `Trie_block_100_M_64_cross_block_2`,
`Trie_block_100_M_64_cross_block_4`, and
`Trie_block_100_M_64_cross_block_6`. Ready-to-run configurations are
