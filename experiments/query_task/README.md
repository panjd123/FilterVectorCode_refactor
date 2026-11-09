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
cd /home/dev/graphdb/FilterVectorCode_multilevel_special

python3 scripts/run_custom_query_tasks.py \
  experiments/query_task/custom_query_tasks.example.json
```

也可以用 shell 包装脚本：

```bash
cd /home/dev/graphdb/FilterVectorCode_multilevel_special

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
  "build_dir": "/home/dev/graphdb/FilterVectorCode_multilevel_special/build_ung_rel",
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
- `min_selectivity`：每条 query 的 containment 选择率下界（包含），必须与 `max_selectivity` 同时设置。
- `max_selectivity`：每条 query 的 containment 选择率上界；小于 `1.0` 时不包含上界，等于 `1.0` 时包含。因此三个连续区间可配置为 `[0,0.1)`、`[0.1,0.4)`、`[0.4,1.0]`。
- `selection_target_selectivity`：最终选择候选时优先选择最接近该目标选择率的不同 label set；相同距离随机打散。
- `preferred_selectivity_min` / `preferred_selectivity_max`：最终选择优先使用的子区间。只要子区间内存在候选，初次选择和普通重复补足都从该子池抽取。
- `duplicate_selectivity_min` / `duplicate_selectivity_max`：候选不足、必须重复补足时使用的优先子区间，不影响初次选择不同 label set。
- `include_selectivity_extremes`：设为 `true` 时预留两个位置，分别放入完整候选池中选择率最低和最高的查询；两者仍满足标签长度、选择率区间和 `min_groundtruth` 约束。
- `selectivity_candidate_pool_size`：额外随机生成并计算 coverage 的候选数量。低选择率标签空间很大时应提高；中高选择率会对达到下界的少量高频标签进行完整子集枚举。
- `random_seed`：候选抽样、最终标签选择和查询向量选择的随机种子；`0` 表示使用系统随机源。
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
- 固定区间模式按每个候选查询标签自身的 `coverage / base_count` 分桶，不再使用目标值上下浮动或软容差。
- 中高选择率区间利用反单调性：达到下界的多标签集合，其每个单标签也必须达到下界；高频标签不超过 20 个时会完整枚举其子集。
- 候选数不少于 `num_points` 时优先无放回随机选择；候选不足时随机复用，日志中的 `selected_unique_label_sets` 和 `duplicate_queries` 会报告多样性。
- 固定区间模式仍强制 `min_groundtruth`、标签长度和 `min_children`，并忽略旧的目标值/容差配置字段。
- 如果生成速度很慢，通常说明满足 `min_query_length` 和 coverage 条件的 query 很稀疏，可以适当降低 `K`、提高 `max_coverage`，或增加 `num_points` 之外的候选尝试逻辑。
- 如果 `overwrite=false` 且目标 `.fvecs` 已存在，脚本会跳过生成，但仍会补 `.bin` 和 profile。
