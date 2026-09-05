# NaviX Baseline Experiment

从项目根目录运行：

```bash
cd /home/dev/graphdb/FilterVectorCode_refactor
python3 experiments/navix_baseline/run_navix_baseline.py experiments/navix_baseline/config.json
```

脚本会自动执行：

1. `cmake -S UNG/codes -B <build_dir> -DCMAKE_BUILD_TYPE=Release`
2. 编译 `build_UNG_index`、`search_UNG_index`、`compute_groundtruth`、`fvecs_to_bin`
3. 若 query bin 不存在但 fvecs 存在，自动转换
4. 若 containment GT 不存在，自动计算
5. 若 `FilterVectorResult/<Dataset>/index/NaviX/index_files/meta` 不存在，执行 `build_UNG_index --index_type NaviX`
6. 执行 `search_UNG_index --force_use_alg 5`

默认输出：

```text
/home/dev/graphdb/FilterVectorResult/<Dataset>/index/NaviX/index_files/
/home/dev/graphdb/FilterVectorResult/<Dataset>/results/NaviX/<query_task>_<lsearch_start>_<lsearch_step>_<lsearch_end>/
```

常用参数在 `config.json` 中修改：

- `build.num_threads`
- `build.max_degree`
- `build.Lbuild`
- `search.K`
- `search.num_threads`
- `search.num_repeats`
- `search.lsearch_start/search.lsearch_end/search.lsearch_step`
- `datasets[].dataset`
- `datasets[].query_task`

也可以在单个 dataset 配置里覆盖 `build`、`search` 或显式文件路径，例如 `base_bin_file`、`query_bin_file`、`groundtruth_file`、`index_path_prefix`、`result_path_prefix`。
