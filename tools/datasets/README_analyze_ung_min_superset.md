# UNG MinSuperset 查询筛选与任务生成

这个脚本用于比较 `gpu_bruteforce_els_ung` 和 `UNG` 两个算法的 per-query 结果：

- 按 `(repeat, Lsearch, efs, QueryID)` 对齐两个 `query_details_repeat1.csv`。
- 统计每个 `Lsearch` 下 `gpu_bruteforce_els_ung.MinSupersetT_ms < UNG.MinSupersetT_ms` 的 query 数量。
- 进一步筛选 `MinSupersetT_ms / Time_ms` 占比较大的 query。默认口径是看 UNG 侧，占比阈值为 `30%`。
- 可选把某个 `Lsearch` 下满足条件的 query 从原始 query task 中抽出来，生成新的 query task。

## 脚本路径

```bash
/home/dev/graphdb/FilterVectorCode_refactor/tools/datasets/analyze_ung_min_superset.py
```

## 默认输入

脚本默认比较下面两个文件：

```text
/home/dev/graphdb/FilterVectorResult/Amazon/results/gpu_bruteforce_els_ung/zipf/query_minlen1_cov20k_1000_1000_20000/results/query_details_repeat1.csv
/home/dev/graphdb/FilterVectorResult/Amazon/results/UNG/zipf/query_minlen1_cov20k_1000_1000_20000/results/query_details_repeat1.csv
```

默认原始 query task 目录是：

```text
/home/dev/graphdb/FilterVectorData/Amazon/zipf_query/query_minlen1_cov20k
```

该目录需要包含：

```text
Amazon_query.bin
Amazon_query.fvecs
Amazon_query_labels.txt
```

## 只做统计

```bash
cd /home/dev/graphdb/FilterVectorCode_refactor

python3 tools/datasets/analyze_ung_min_superset.py
```

默认筛选口径：

```text
gpu.MinSupersetT_ms < ung.MinSupersetT_ms
UNG.MinSupersetT_ms / UNG.Time_ms >= 0.30
```

## 导出统计 CSV 和匹配明细

```bash
cd /home/dev/graphdb/FilterVectorCode_refactor

python3 tools/datasets/analyze_ung_min_superset.py \
  --summary-out /home/dev/graphdb/FilterVectorResult/Amazon/results/ung_min_superset_gpu_smaller_large_share_summary.csv \
  --details-out /home/dev/graphdb/FilterVectorResult/Amazon/results/ung_min_superset_gpu_smaller_large_share_details.csv
```

输出含义：

- `summary.csv`：每个 `Lsearch` 的汇总计数、占比、平均/中位数 share、平均/中位数 speedup。
- `details.csv`：所有满足条件的 per-query 明细。

## 生成 Lsearch=1000 的新 query task

这会把默认口径下 `Lsearch=1000` 的 505 个 query 抽出来：

```bash
cd /home/dev/graphdb/FilterVectorCode_refactor

python3 tools/datasets/analyze_ung_min_superset.py \
  --materialize-lsearch 1000 \
  --output-dir /home/dev/graphdb/FilterVectorData/Amazon/zipf_query/query_minlen1_cov20k_gpu_els_min_superset_lsearch1000_share30 \
  --summary-out /home/dev/graphdb/FilterVectorResult/Amazon/results/ung_min_superset_gpu_smaller_large_share_summary.csv \
  --details-out /home/dev/graphdb/FilterVectorResult/Amazon/results/ung_min_superset_gpu_smaller_large_share_details.csv
```

生成目录：

```text
/home/dev/graphdb/FilterVectorData/Amazon/zipf_query/query_minlen1_cov20k_gpu_els_min_superset_lsearch1000_share30
```

生成文件：

```text
Amazon_query.bin
Amazon_query.fvecs
Amazon_query_labels.txt
selected_queries.csv
selected_sources.csv
selection_manifest.json
README.md
```

其中：

- `Amazon_query.bin/.fvecs/_labels.txt` 是新的 query task，可作为后续搜索输入。
- `selected_queries.csv` 记录新 query id 到原始 `QueryID` 的映射，以及两算法的时间、share、speedup。
- `selected_sources.csv` 记录每条新 query 来自哪个原始文件和原始行。
- `selection_manifest.json` 记录输入文件、筛选条件、输出文件和 query 数量。

如果输出目录已有文件，需要显式加：

```bash
--overwrite
```

## 常用参数

调整“占比很大”的阈值：

```bash
--min-share 0.50
```

改成看 GPU 侧占比、两侧同时满足、任一侧满足：

```bash
--share-side gpu
--share-side both
--share-side either
```

抽取其他 `Lsearch`：

```bash
--materialize-lsearch 2000
```

换输入 CSV：

```bash
--gpu-csv /path/to/gpu/query_details_repeat1.csv \
--ung-csv /path/to/ung/query_details_repeat1.csv
```

换原始 query task：

```bash
--source-query-dir /path/to/source/query_task \
--dataset Amazon
```

## 本次已生成的结果

使用默认阈值 `0.30`、默认 `share-side=ung`、`Lsearch=1000`：

```text
materialized queries: 505
```

新 query task 目录：

```text
/home/dev/graphdb/FilterVectorData/Amazon/zipf_query/query_minlen1_cov20k_gpu_els_min_superset_lsearch1000_share30
```
