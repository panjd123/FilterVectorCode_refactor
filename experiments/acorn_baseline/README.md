# ACORN Baseline Experiment

从项目根目录运行：

```bash
cd /home/dev/graphdb/FilterVectorCode_refactor
python3 experiments/acorn_baseline/run_acorn_baseline.py experiments/acorn_baseline/config.json
```

脚本会自动执行：

1. `cmake -S ACORN -B <build_dir> ...`
2. 编译 `faiss`、`utils`、`test_acorn`
3. 编译或复用 UNG 的 `compute_groundtruth`、`fvecs_to_bin`
4. 若 query bin 不存在但 fvecs 存在，自动转换
5. 若 containment GT 不存在，自动计算
6. 若 `FilterVectorResult/<Dataset>/index/ACORN/index_files/acorn.index` 等文件不存在，执行 `test_acorn build`
7. 执行 `test_acorn search`，并把 ACORN 原生 `avg_*.csv` 归一化为通用的 `search_time_summary.csv` 和 `search_time_summary_qps.csv`

默认输出：

```text
/home/dev/graphdb/FilterVectorResult/<Dataset>/index/ACORN/index_files/
/home/dev/graphdb/FilterVectorResult/<Dataset>/results/ACORN/<query_task>_<efs_start>_<efs_step>_<efs_end>/
```

ACORN 的运行方式：

- 构建阶段调用 `ACORN/build.../demos/test_acorn build ...`，写出 `acorn.index`、`acorn1.index`、`acorn.index.meta` 和 `acorn.index.inverted_index`。
- 搜索阶段调用 `test_acorn search ...`，读取上述索引、加载 query fvecs 和 GT bin，并按 `efs` 列表 sweep。
- `efs` 在归一化结果中映射为 `Lsearch/Average_Efs`，方便和 NaviX/UNG 结果画图或汇总。

常用参数在 `config.json` 中修改：

- `build.M`
- `build.M_beta`
- `build.gamma`
- `search.K`
- `search.num_threads`
- `search.num_repeats`
- `search.efs_start/search.efs_end/search.efs_step`
- `search.efs_values`
- `datasets[].dataset`
- `datasets[].query_task`

也可以在单个 dataset 配置里覆盖 `build`、`search` 或显式文件路径，例如 `base_fvecs`、`base_bin_file`、`query_fvecs`、`query_bin_file`、`groundtruth_file`、`index_path_prefix`、`result_path_prefix`。
