# 查询任务生成说明

本目录用于通过 JSON 配置批量生成过滤查询。当前示例配置是：

```text
custom_query_tasks.example.json
```

推荐使用项目里的流程脚本运行，它会自动完成三件事：

1. 调用 `generate_mixed_queries` 生成查询标签和查询向量。
2. 把生成的 `.fvecs` 查询向量转换成搜索脚本常用的 `.bin`。
3. 如果配置中开启 `analysis_params.analyze`，额外生成 coverage 分析 CSV。

## 运行命令

在项目根目录执行：

```bash
cd /home/dev/graphdb/FilterVectorCode_refactor

python3 scripts/run_custom_query_tasks.py \
  experiments/query_task/custom_query_tasks.example.json
```

也可以用 shell 包装脚本：

```bash
cd /home/dev/graphdb/FilterVectorCode_refactor

bash scripts/run_custom_query_tasks.sh \
  experiments/query_task/custom_query_tasks.example.json
```

构建目录由 JSON 顶层 `build_dir` 指定；如果命令行仍传第二个参数，它会覆盖 JSON 中的值。脚本会从这里寻找：

```text
build_ung_rel/tools/generate_mixed_queries
build_ung_rel/tools/fvecs_to_bin
```

如果这两个工具不存在，需要先编译：

```bash
cmake --build build_ung_rel --target generate_mixed_queries fvecs_to_bin -j 16
```

## 当前配置会生成什么

当前 `custom_query_tasks.example.json` 的核心配置如下：

```json
{
  "build_dir": "/home/dev/graphdb/FilterVectorCode_refactor/build_ung_rel",
  "query_tasks": [
    {
      "task_name": "minlen1_cov20k",
      "mode": "variable_sub_base",
      "dataset": "Amazon",
      "data_dir": "/home/dev/graphdb/FilterVectorData/Amazon",
      "sub_base_params": {
        "num_points": 3000,
        "min_query_length": 1,
        "max_query_length": 0,
        "K": 20001,
        "max_coverage": 602453,
        "min_children": 0,
        "parent_range_start": 0.0,
        "parent_range_end": 0.7
      }
    }
  ]
}
```

含义是：在 Amazon 数据集上生成 3000 条 query；每条 query 的标签长度至少为 1，不固定长度；每条 query 至少覆盖 20001 个向量；覆盖数量最多不超过 602453。

输出目录由 `data_dir` 和 `task_name` 决定：

```text
/home/dev/graphdb/FilterVectorData/Amazon/query_minlen1_cov20k/
```

输出文件包括：

```text
Amazon_query_labels.txt
Amazon_query.fvecs
Amazon_query.bin
profiled_minlen1_cov20k.csv
```

其中：

- `Amazon_query_labels.txt`：每条 query 的过滤标签集合。
- `Amazon_query.fvecs`：查询向量，fvecs 格式。
- `Amazon_query.bin`：查询向量，搜索脚本常用的 bin 格式。
- `profiled_minlen1_cov20k.csv`：每条 query 的 coverage 统计，开启 `analysis_params.analyze=true` 时生成。

## 配置字段说明

顶层字段：

- `build_dir`：构建目录，需包含 `tools/generate_mixed_queries` 和 `tools/fvecs_to_bin`。可写绝对路径，也可写相对项目根目录的路径如 `build_ung_rel`。
- `enabled`：是否启用该任务。设为 `false` 会跳过。
- `task_name`：查询任务名。最终目录名固定为 `query_${task_name}`。
- `mode`：生成模式。当前推荐 `variable_sub_base`。
- `dataset`：数据集名前缀。脚本会读取 `${dataset}_base_labels.txt` 和 `${dataset}_base.fvecs`。
- `data_dir`：数据集目录。
- `overwrite`：如果输出已存在，是否覆盖重跑。

`sub_base_params` 字段：

- `num_points`：要生成的 query 数量。
- `min_query_length`：query 标签长度下限。
- `max_query_length`：query 标签长度上限。设为 `0` 表示不额外限制，最多到 parent label set 的长度。
- `K`：最小覆盖向量数。比如要覆盖大于 10 万个向量，写 `100001`。
- `max_coverage`：最大覆盖向量数。通常可以写数据集总向量数。
- `min_groundtruth`：每条 query 必须拥有的最小有效真值数量，默认 `20`。生成器会在所有 `sub_base` 模式（包括批次平均选择率模式）中强制执行该下限。
- `min_children`：对 query label set 的 superset group 数量要求。`variable_sub_base` 常用 `0`，表示只按 coverage 过滤。
- `target_average_selectivity`：可选，范围 `[0, 1]`。设置后启用批次平均选择率模式，使整批 query 的 `平均 coverage / base 向量数` 接近该值；此模式不再对每条 query 应用 `K` 和 `max_coverage`，但仍强制 `min_groundtruth`。
- `average_selectivity_tolerance`：批次实际平均选择率与目标值允许的绝对误差，默认 `0.001`。
- `average_candidate_pool_size`：单标签候选无法达到目标容差时，最多额外采样的候选数，默认 `20000`。
- `parent_range_start`：parent label set 来源区间起点比例，范围 `[0.0, 1.0)`。比如前 70% 写 `0.0`。
- `parent_range_end`：parent label set 来源区间终点比例，范围 `(0.0, 1.0]`。比如前 70% 写 `0.7`，后 30% 写 `1.0` 且 start 写 `0.7`。
- `cache-file`：superset 统计缓存。存在时可以加速生成。

`analysis_params` 字段：

- `analyze`：设为 `true` 时，生成后会重新分析每条 query 的 coverage，并输出 `profiled_${task_name}.csv`。

## 自定义示例

如果想生成“标签长度至少为 3，覆盖向量大于 10 万”的查询，可以改成：

```json
{
  "task_name": "minlen3_cov100k",
  "mode": "variable_sub_base",
  "dataset": "Amazon",
  "data_dir": "/home/dev/graphdb/FilterVectorData/Amazon",
  "overwrite": false,
  "sub_base_params": {
    "num_points": 1000,
    "min_query_length": 3,
    "max_query_length": 0,
    "K": 100001,
    "max_coverage": 602453,
    "min_children": 0,
    "parent_range_start": 0.0,
    "parent_range_end": 0.7,
    "cache-file": "/home/dev/graphdb/FilterVectorData/Amazon/sub_base_cache"
  },
  "analysis_params": {
    "analyze": true
  }
}
```

运行后输出目录是：

```text
/home/dev/graphdb/FilterVectorData/Amazon/query_minlen3_cov100k/
```

## 注意事项

- `variable_sub_base` 不保证固定标签长度，只保证长度不小于 `min_query_length`。
- `K` 是 coverage 下限，不是搜索时返回的 TopK。
- 批次平均选择率模式优先混合目标两侧的可达 coverage。例如目标 `0.5` 不要求每条 query 都有 50% 的选择率，只保证整批平均值在容差内。
- 在满足平均选择率容差的前提下，生成器优先选择尚未使用过的 label set；候选种类不足或继续使用新标签会破坏平均选择率时才会重复。日志中的 `unique_label_sets` 和 `duplicate_queries` 可用于检查多样性。
- 若采样到的 coverage 无法包围目标，或者离散 coverage 在给定 query 数量下无法满足容差，生成器会直接失败并打印可达范围，不会进行逐 query 的百万级无效重试。
- 如果生成速度很慢，通常说明满足 `min_query_length` 和 coverage 条件的 query 很稀疏，可以适当降低 `K`、提高 `max_coverage`，或增加 `num_points` 之外的候选尝试逻辑。
- 如果 `overwrite=false` 且目标 `.fvecs` 已存在，脚本会跳过生成，但仍会补 `.bin` 和 profile。
