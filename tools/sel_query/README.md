# 查询任务筛选工具说明

`select_query_task_by_time.py` 用于从已有搜索结果中挑选“某个方法明显快于另一个方法”的 query，并把这些 query 重新物化成一个新的查询任务目录。

脚本会读取每个方法结果目录下的：

```text
<results_root>/<method>/<query_result_task>/results/query_details_repeat1.csv
```

并按相同的 `QueryID + Lsearch` 比较耗时字段，默认比较 `Time_ms`。如果某条 query 在除第一个 `Lsearch` 外的任一步长上满足规则，就会被选入新查询任务。

## 文件

- `select_query_task_by_time.py`：主脚本。
- `select_query_config.json`：默认配置文件。

## 快速运行

只统计会选出多少条，不写输出文件：

```bash
cd /home/dev/graphdb/FilterVectorCode_refactor
python3 tools/sel_query/select_query_task_by_time.py \
  --config tools/sel_query/select_query_config.json \
  --dry-run
```

生成新的查询任务：

```bash
cd /home/dev/graphdb/FilterVectorCode_refactor
python3 tools/sel_query/select_query_task_by_time.py \
  --config tools/sel_query/select_query_config.json
```

## 配置说明

顶层配置示例：

```json
{
  "dataset": "Amazon",
  "results_root": "/home/dev/graphdb/FilterVectorResult/Amazon/results",
  "query_data_root": "/home/dev/graphdb/FilterVectorData",
  "output_task": "query_selected_time_advantage",
  "result_task_suffix": "_1000_1000_20000",
  "skip_first_lsearch": true,
  "time_column": "Time_ms",
  "min_recall": 0.0,
  "overwrite": false,
  "max_queries": 0,
  "selections": []
}
```

字段含义：

- `dataset`：数据集名，例如 `Amazon`、`Reviews`。
- `results_root`：该数据集的搜索结果根目录。
- `query_data_root`：查询任务数据根目录，脚本会读取 `<query_data_root>/<dataset>/<source_task>/`。
- `output_task`：输出的新查询任务目录名。
- `result_task_suffix`：默认结果任务后缀。若 `source_task` 是 `query_minlen1_cov10k`，后缀是 `_1000_1000_20000`，则结果任务名默认为 `query_minlen1_cov10k_1000_1000_20000`。
- `skip_first_lsearch`：是否跳过第一个 `Lsearch`。默认建议为 `true`。
- `time_column`：用于比较的耗时字段，默认 `Time_ms`。
- `min_recall`：可选召回率下限。设为 `0.0` 表示不按 Recall 过滤。
- `overwrite`：输出目录已有文件时是否允许覆盖。
- `max_queries`：全局默认数量限制。`0` 表示不限数量。
- `selections`：筛选规则列表。

## 单条规则

规则示例：

```json
{
  "name": "gpu_els_ung_faster_than_hybrid",
  "source_task": "query_minlen1_cov10k",
  "faster_method": "gpu_bruteforce_els_ung",
  "slower_method": "UNG__hybrid",
  "min_speedup": 2.0,
  "result_task_suffix": "_100_100_1000",
  "max_queries": 100
}
```

字段含义：

- `name`：规则名，会写入 `selected_queries.csv`。
- `source_task`：原始查询任务目录名，例如 `query_minlen1_cov10k`。
- `faster_method`：预期更快的方法目录名。
- `slower_method`：被比较的较慢方法目录名。
- `min_speedup`：速度比阈值，计算方式为 `slower_time_ms / faster_time_ms`。例如 `2.0` 表示快方法至少快 2 倍。
- `min_delta_ms`：可选绝对耗时差阈值，默认 `0.0`。
- `min_recall`：可选规则级 Recall 下限，会覆盖全局 `min_recall`。
- `skip_first_lsearch`：可选规则级设置，会覆盖全局 `skip_first_lsearch`。
- `lsearch_alignment`：Lsearch 配对方式。`exact`（默认）要求两侧 Lsearch 数值相等；`ordinal` 按排序后的第 1、第 2、... 个 Lsearch 配对，适用于两个方法的步长不同。`skip_first_lsearch` 在两种模式下都跳过第一对。
- `time_column`：可选规则级耗时字段，会覆盖全局 `time_column`。
- `faster_time_column`：可选，`faster_method` 使用的耗时字段；未设置时使用规则级 `time_column` 或全局 `time_column`。
- `slower_time_column`：可选，`slower_method` 使用的耗时字段；未设置时使用规则级 `time_column` 或全局 `time_column`。
- `max_queries`：可选规则级数量限制。`0` 表示不限量；大于 0 时按 `speedup` 从高到低取前 N 条。
- `result_task_suffix`：可选规则级结果任务后缀，会覆盖全局 `result_task_suffix`。
- `faster_result_task_suffix`：可选，`faster_method` 的结果任务后缀；未设置时使用 `result_task_suffix` 或全局后缀。
- `slower_result_task_suffix`：可选，`slower_method` 的结果任务后缀；未设置时使用 `result_task_suffix` 或全局后缀。
- `query_result_task`：可选完整结果任务名。如果设置了它，脚本会直接使用该值，不再拼接 `source_task + result_task_suffix`。

## 当前两类筛选

第一类查询任务，例如 `query_minlen1_cov10k`：

```json
{
  "name": "gpu_els_ung_faster_than_hybrid",
  "source_task": "query_minlen1_cov10k",
  "faster_method": "gpu_bruteforce_els_ung",
  "slower_method": "UNG__hybrid",
  "min_speedup": 2.0,
  "result_task_suffix": "_1000_1000_20000",
  "max_queries": 0
}
```

表示挑选 `gpu_bruteforce_els_ung` 的 `Time_ms` 明显快于 `UNG__hybrid` 的 query。

第二类 Zipf 查询任务，例如 `query_minlen1_cov10k_zipf`：

```json
{
  "name": "special_blocks_faster_than_gpu_els_ung_zipf",
  "source_task": "query_minlen1_cov10k_zipf",
  "faster_method": "gpu_bruteforce_els_special_blocks",
  "slower_method": "gpu_bruteforce_els_ung",
  "min_speedup": 2.0,
  "result_task_suffix": "_1000_1000_20000",
  "max_queries": 0
}
```

表示挑选 `gpu_bruteforce_els_special_blocks` 的 `Time_ms` 明显快于 `gpu_bruteforce_els_ung` 的 query。

## 输出

输出目录默认是：

```text
<query_data_root>/<dataset>/<output_task>/
```

会生成：

- `<dataset>_query_labels.txt`：新查询任务的 labels。
- `<dataset>_query.fvecs`：新查询向量，按选中 query 从源任务复制。
- `<dataset>_query.bin`：新查询向量 bin 文件。
- `selected_queries.csv`：筛选明细，包括新 query id、来源任务、原 query id、快方法与慢方法各自命中的 Lsearch、两个方法耗时和 speedup。
- `selection_manifest.json`：本次输出的摘要信息。

## 注意事项

- 脚本按同一规则内每个 `QueryID` 只保留一个最佳 `Lsearch`，即 speedup 最大的那一步。
- 如果多个规则选中了同一个 `source_task + QueryID`，最终只保留 speedup 最大的一条。
- `max_queries` 是先在每条规则内按 speedup 截断，然后再做跨规则去重。
- 如果结果目录后缀不一致，优先在对应规则里设置 `result_task_suffix`。
- 如果结果任务名无法用 `source_task + suffix` 表达，直接设置 `query_result_task`。
- 输出文件已存在且 `overwrite` 为 `false` 时，脚本会停止，避免覆盖旧结果。
