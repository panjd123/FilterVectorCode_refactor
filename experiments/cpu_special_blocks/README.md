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

Outputs are written under `result_root/run_name_timestamp/`.
Each dataset directory contains `index_files/`, `results/`, `others/build.log`, `others/time.txt`, and `summary.csv`.
The top-level run directory also contains an aggregated `summary.csv`.
