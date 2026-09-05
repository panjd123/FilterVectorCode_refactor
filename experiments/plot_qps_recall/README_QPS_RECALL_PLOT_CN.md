# QPS-Recall 绘图脚本说明

这个目录下新增了两个文件，用于从三个 UNG 方法的
`search_time_summary.csv` 生成 `search_time_summary_draw.csv`，并绘制
QPS-Recall 折线图。

## 文件

- `plot_qps_recall_from_summary.py`
  - 纯 Python 标准库实现，不依赖 `matplotlib`。
  - 输出 SVG 图，因此在没有绘图库的环境中也能运行。
- `plot_qps_recall_config.sh`
  - 可直接修改的配置脚本。
- 默认配置示例为 Genome 的多个查询任务；修改 JSON 中的 `query_tasks` 即可切换或扩展任务。

## 默认绘图口径

默认使用：

```bash
SORT_MODE="time-column-only"
```

含义是：

1. 读取每个方法的 `results/search_time_summary.csv`。
2. 只把 `Average_Time_ms` 这一列单独升序排序。
3. `Lsearch`、`Average_Efs`、`Average_Recall` 保持原始行顺序，不跟随时间列移动。
4. 根据重排后的 `Average_Time_ms` 计算 QPS。
5. 写出 `results/search_time_summary_draw.csv`。
6. 使用 `search_time_summary_draw.csv` 绘制 QPS-Recall 折线图。

这个口径对应你当前要求的“只有 Average_Time_ms 位置升序，其他列不要跟随变化”。

## QPS 计算

`Average_Time_ms` 是一批 query 在当前 Lsearch 下的 batch wall time，不是单 query 平均延迟。

脚本会从：

```text
FilterVectorData/<Dataset>/<QueryTask>/<Dataset>_query.bin
```

读取 query 数量 `num_queries`，然后计算：

```text
QPS = num_queries * 1000 / Average_Time_ms
```

例如 1000 个 query 时：

```text
QPS = 1000 * 1000 / Average_Time_ms
```

## 运行方法

直接运行：

```bash
bash /home/dev/graphdb/FilterVectorCode_refactor/experiments/plot_qps_recall/plot_qps_recall_config.sh
```

默认会处理：

```text
/home/dev/graphdb/FilterVectorResult/Amazon/results/{method}/query_special_core_all_then_els_1000/results/search_time_summary.csv
```

其中 `{method}` 为：

- `UNG`
- `gpu_bruteforce_els_ung`
- `gpu_bruteforce_els_special_blocks`

## 输出文件

每个方法的结果目录中会生成：

```text
search_time_summary_draw.csv
```

总输出目录默认为：

```text
/home/dev/graphdb/FilterVectorResult/Amazon/results/plot/query_special_core_all_then_els_1000
```

其中包含：

- `qps_recall_draw_linear.svg`
- `qps_recall_draw_log.svg`
- `combined_qps_recall_draw.csv`
- `recall_threshold_summary_draw.csv`
- `summary_draw.md`

图例默认放在绘图区外侧右边。

配置脚本默认带 `--skip-missing`：如果某个方法的 `search_time_summary.csv` 还没生成，会先跳过该方法并绘制已有方法。若希望强制三个方法都必须存在，可以从 `plot_qps_recall_config.sh` 中删除 `--skip-missing`。


## 带 Lsearch 步长后缀的结果目录

如果结果目录名包含步长，例如：

```text
query_minlen1_cov20k_1000_1000_20000
```

但数据目录中的 query bin 仍然是：

```text
query_minlen1_cov20k/Amazon_query.bin
```

则在 JSON 中分开配置：

```json
"query_task": "query_minlen1_cov20k",
"result_task": "query_minlen1_cov20k_1000_1000_20000",
"lsearch": {"start": 1000, "step": 1000, "end": 20000}
```

脚本会用 `query_task` 查找 query bin 以计算 query 数，用 `result_task` 查找各方法的 `search_time_summary.csv` 并作为 plot 子目录名。

## 修改查询任务

编辑 `plot_qps_recall_config.json` 的 `query_tasks`。列表中的每个任务都会完整绘制一组图，输出到各自的 `plot/<result_task>` 目录：

```json
{
  "dataset": "Genome",
  "query_tasks": [
    "query_selected_mixed_advantage",
    "query_minlen2_cov1k",
    "query_minlen5_cov0.1k"
  ]
}
```

因此查询任务只需在顶层配置一次，不需要在每个 `methods` 项中重复填写。也支持为某个任务单独指定结果目录和 Lsearch 范围：

```json
"query_tasks": [
  {
    "query_task": "query_minlen2_cov1k",
    "result_task": "query_minlen2_cov1k_1000_1000_20000",
    "lsearch": {"start": 1000, "step": 1000, "end": 20000}
  }
]
```

单任务时也可以继续使用旧格式的顶层 `query_task`；它与 `query_tasks: ["..."]` 等价。

如果某个方法的结果目录带有查询任务前缀，可在 `methods[].result_task` 中使用 `{query_task}`（或 `<query_task>`）占位符，例如 `"{query_task}_search_ef64_search_ef10240"`。这样批量绘图时会自动替换成当前任务。

## Recall 阈值

默认使用自动阈值：脚本会读取当前已加载结果中的 `Average_Recall` 最小值和最大值，按 `--threshold-step` 生成阈值，并额外包含实际最小/最大 recall。这样换查询任务后不需要手动改 `0.6/0.7/...` 这类固定范围。

如果要手动指定阈值，可以重复传入 `--threshold`，例如：

```bash
--threshold 0.8 --threshold 0.9 --threshold 0.95
```

## 其它排序模式

脚本还支持另外两个模式：

```bash
SORT_MODE="row-by-time"
```

整行按 `Average_Time_ms` 升序排序，即 `Lsearch`、`Recall` 等字段跟着时间一起移动。

```bash
SORT_MODE="none"
```

完全不排序，保持原始 `search_time_summary.csv` 行顺序，只增加 QPS 列。

## 单独调用 Python 脚本

```bash
python3 /home/dev/graphdb/FilterVectorCode_refactor/experiments/plot_qps_recall/plot_qps_recall_from_summary.py \
  --results-root /home/dev/graphdb/FilterVectorResult/Amazon/results \
  --data-root /home/dev/graphdb/FilterVectorData/Amazon \
  --dataset Amazon \
  --query-task query_special_core_all_then_els_1000 \
  --sort-mode time-column-only \
  --legend-outside
```
