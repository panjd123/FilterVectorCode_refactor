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

