# UNG Search Comparison

这个实验用于在已有的 `UNG` / `UNG_special_blocks` 索引上跑查询任务，并比较不同 search route 的结果。当前配置主要覆盖 Amazon、Genome、Reviews、VariousImg 等数据集。

## 如何跑查询任务

推荐从 `FilterVectorCode_refactor` 目录启动：

```bash
cd /home/dev/graphdb/FilterVectorCode_refactor
./run_search_comparison_experiment.sh experiments/search_comparison/config_genome.json
```

如果不传配置文件，脚本默认使用：

```text
/home/dev/graphdb/FilterVectorCode_refactor/experiments/search_comparison/config_genome.json
```

脚本会先执行 `build_ung_local.sh`，确保 `build_ung_rel/apps/search_UNG_index` 等二进制存在，然后调用：

```bash
python3 experiments/search_comparison/run_search_comparison.py experiments/search_comparison/config_genome.json
```

如果已经确认代码编译完成，也可以直接跑 Python 脚本：

```bash
cd /home/dev/graphdb/FilterVectorCode_refactor
python3 experiments/search_comparison/run_search_comparison.py experiments/search_comparison/config_genome.json
```

## 选择要跑的查询任务

查询任务由配置文件里的 `datasets[].query_task` 决定。例如：

```json
{
  "dataset": "Genome",
  "query_task": "query_minlen1_cov10k"
}
```

这表示会读取：

```text
/home/dev/graphdb/FilterVectorData/Genome/query_minlen1_cov10k/Genome_query.bin
/home/dev/graphdb/FilterVectorData/Genome/query_minlen1_cov10k/Genome_query_labels.txt
```

如果 `<Dataset>_query.bin` 不存在，但同目录下有 `<Dataset>_query.fvecs`，脚本会自动调用 `build_ung_rel/tools/fvecs_to_bin` 转成 bin 文件。

要换查询任务，复制一个现有配置文件，然后修改：

- `datasets`: 只保留要跑的数据集和 `query_task`
- `methods`: 只保留要比较的方法
- `search`: 修改 `K`、`num_threads`、`num_repeats`、`lsearch_start/end/step` 等搜索参数

已有示例：

```bash
cd /home/dev/graphdb/FilterVectorCode_refactor
./run_search_comparison_experiment.sh experiments/search_comparison/config_amazon.json
./run_search_comparison_experiment.sh experiments/search_comparison/config_genome.json
./run_search_comparison_experiment.sh experiments/search_comparison/config.json
```

## Ground Truth 和输出目录

脚本会先检查当前查询任务是否已有 containment ground truth：

```text
/home/dev/graphdb/FilterVectorResult/<Dataset>/GroundTruth/<QueryTaskName>/<Dataset>_gt_labels_containment.bin
```

如果 GT 不存在，会自动调用 `build_ung_rel/tools/compute_groundtruth` 生成。

搜索结果写到：

```text
/home/dev/graphdb/FilterVectorResult/<Dataset>/results/<Method>/<QueryTaskName>_<lsearch_start>_<lsearch_step>_<lsearch_end>/
```

例如 `query_minlen1_cov10k` 配合 `lsearch_start=100`、`lsearch_step=100`、`lsearch_end=1000` 时，目录名是：

```text
query_minlen1_cov10k_100_100_1000
```

每个方法目录下会包含原始 `search_UNG_index` 输出 CSV，以及额外生成的：

```text
results/search_time_summary_qps.csv
```

其中 `QPS = num_queries * 1000 / Average_Time_ms`。

本次实验整体日志会写在配置文件同目录下：

```text
experiments/search_comparison/search_comparison_<YYYYMMDD_HHMMSS>.log
```
