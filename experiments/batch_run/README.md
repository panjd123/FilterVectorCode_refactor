# Batch Experiment Runner

Use this runner when you want one JSON file to run the same methods across several datasets.

```bash
cd /home/dev/graphdb/FilterVectorCode_refactor
python3 experiments/batch_run/run_batch_experiments.py experiments/batch_run/config.json --dry-run
python3 experiments/batch_run/run_batch_experiments.py experiments/batch_run/config.json
```

Edit `config.json`:

- `datasets`: add objects with `dataset` and `query_task`.
- `methods`: choose from `favor`, `Navix`, `curator`, `UNG__hybrid`, `cpu_bruteforce_els_special_blocks`, `cpu_bruteforce_els_ung`.
- `build`: shared build hyperparameters. The batch runner filters these per child runner so UNG/FAVOR/Navix/Curator-only keys do not leak into unrelated configs.
- `search`: shared search hyperparameters copied into generated child configs.
- `method_search`: per-runner search hyperparameters. Use `sweeps` to build explicit search lists, for example `100..1000` by `100`, then `2000..20000` by `1000`.
- `overrides`: final per-runner overrides for any generated child config.

For `favor`, `method_search.favor.sweeps` becomes top-level `ef_values`. For `Navix` and `search_comparison`, `sweeps` becomes `search.lsearch_values`, and those runners expand the list only when building the search command.

The runner writes generated child configs under `generated_config_dir` and calls existing experiment runners without changing their build/search internals. For the shell index-build steps, the batch runner filters out datasets whose target index already exists. Effective special-block names include `index_name_params`, for example `UNG_special_blocks_hybrid_bdeg128_bcross10`.

Current call order for the default `methods` is:

1. `run_cpu_special_blocks_experiment.sh` for missing `UNG__hybrid` indexes only
2. `run_cpu_special_blocks_experiment.sh` for missing special-block indexes only
3. `experiments/search_comparison/run_search_comparison.py` for the CPU ELS methods
4. `experiments/favor/run_favor_experiment.py`
5. `experiments/navix_baseline/run_navix_baseline.py`
6. `experiments/curator_baseline/run_curator_baseline.py`
