# Curator-v2 Python/FAISS Baseline 运行说明

本文档说明如何在 `/home/dev/graphdb/FilterVectorCode_refactor` 中运行 Curator-v2 baseline，并保持当前项目的数据格式、过滤语义和评测输出口径。

## 1. 目录位置

Curator baseline 相关文件位于：

```bash
/home/dev/graphdb/FilterVectorCode_refactor/experiments/curator_baseline
```

Curator-v2 原始实现放在：

```bash
/home/dev/graphdb/FilterVectorCode_refactor/thirdparty/curator-v2
```

默认配置文件：

```bash
experiments/curator_baseline/config.json
```

## 2. 首次环境构建

首次运行前，需要构建 Curator-v2 自带的 FAISS Python 扩展：

```bash
cd /home/dev/graphdb/FilterVectorCode_refactor
JOBS=16 experiments/curator_baseline/setup_curator_env.sh
```

说明：

- 默认虚拟环境路径是 `.venv_curator/`。
- 默认使用 `/usr/bin/python3.10` 创建环境。
- 默认使用 `/usr/bin/swig`。
- 默认使用 `/home/dev/graphdb/OpenBLAS/libopenblas.so`，如果路径不同，可以通过 `OPENBLAS_LIB=...` 覆盖。
- `JOBS=16` 控制编译 FAISS/Curator 扩展时的并行线程数。

构建成功时会看到：

```text
Curator-v2 FAISS import OK
```

## 3. 运行 Curator baseline

使用默认配置运行：

```bash
cd /home/dev/graphdb/FilterVectorCode_refactor
experiments/curator_baseline/run_curator_baseline_with_env.sh experiments/curator_baseline/config.json
```

也可以直接用 Python 运行：

```bash
cd /home/dev/graphdb/FilterVectorCode_refactor
PYTHONPATH=thirdparty/curator-v2 .venv_curator/bin/python \
  experiments/curator_baseline/run_curator_baseline.py \
  experiments/curator_baseline/config.json
```

推荐使用 `run_curator_baseline_with_env.sh`，它会自动设置 `PYTHONPATH` 和 `LD_LIBRARY_PATH`。

## 4. 当前论文口径参数

当前默认参数按论文 Curator 的搜索参数空间设置：

```json
"build": {
  "nlist": 32,
  "max_sl_size": 128,
  "max_leaf_size": 128,
  "search_ef": 64,
  "beam_size": 4,
  "num_threads": 1
},
"search": {
  "K": 10,
  "num_repeats": 1,
  "num_threads": 1,
  "search_parameter": "search_ef",
  "ef_values": [64, 128, 256, 512, 1024]
}
```

对应论文中的 Curator 参数：

- `nlist ∈ {16, 32}`
- `Bmax ∈ {64, 128, 256}`，在实现里对应 `max_sl_size` 和 `max_leaf_size`
- `ef ∈ {64, 128, 256, 512, 1024}`，在实现里对应 `search_ef`
- `beam_size = 4`
- 论文默认单线程实验，因此示例 config 里 `num_threads=1`

如果需要多线程评测，可以显式修改：

```json
"build": {
  "num_threads": 32
},
"search": {
  "num_threads": 32
}
```

注意：修改 build 参数后，已有 Curator 索引会因为元数据不匹配而重新构建。

## 5. 索引复用逻辑

Curator 索引会保存到：

```bash
<result_root>/<dataset>/index/Curator/index_files/curator.index
```

对应元数据保存到：

```bash
<result_root>/<dataset>/index/Curator/index_files/curator_meta.json
```

再次运行时，如果满足以下条件，会直接复用已有索引，不重新构建：

- `curator.index` 存在
- `curator_meta.json` 存在
- `persistent=true`
- `num_points` 一致
- `dim` 一致
- `build` 参数完全一致

构建统计保存在：

```bash
<result_root>/<dataset>/index/Curator/build/build_time.csv
<result_root>/<dataset>/index/Curator/build/index_size.csv
```

`build_time.csv` 中的 `Reused Existing Index` 为 `true` 时，表示本次复用了已有索引。

## 6. 输出路径

搜索结果路径与 UNG/NaviX 的目录风格保持一致：

```bash
<result_root>/<dataset>/results/Curator/<query_task>_search_ef64_search_ef1024/results/
```

主要输出文件包括：

```bash
search_time_summary.csv
search_time_summary_qps.csv
curator_results.csv
```

`curator_results.csv` 每行包含：

- `search_ef`：本行结果对应的 Curator 搜索预算
- `QueryID`：查询编号
- `Recall`：该 query 在当前 `search_ef` 下的 recall@K
- `Search_Time_ms`：该 query 的在线搜索耗时，包含过滤集合计算、过滤 ID 转换和 Curator 搜索
- `VisitedPoints`：实际计算数据向量距离的点数
- `VisitedEdges`：搜索中实际扫描的临时树边数
- `DistanceComputations`：数据向量距离与树节点 centroid 距离的总计算次数
- `ResultIDs`：该 query 返回的 topK 向量 ID，用分号分隔

summary CSV 中保留精简字段：

- `Lsearch`：等于 Curator 的 `ef/search_ef`
- `Recall`：当前 `search_ef` 下的 recall
- `Average_Time_ms`：当前 `search_ef` 下的平均总耗时
- `Average_VisitedPoints`：当前批次每 query 的平均访问点数
- `Average_VisitedEdges`：当前批次每 query 的平均扫描边数
- `Average_DistanceComputations`：当前批次每 query 的平均距离计算次数

配置多次 repeat 时，三个 `Average_*` 计数字段以所有 repeat 中的全部 query 执行为分母。

`search_time_summary_qps.csv` 额外包含 `QPS`。

## 7. 常用命令

重新构建环境：

```bash
cd /home/dev/graphdb/FilterVectorCode_refactor
JOBS=16 experiments/curator_baseline/setup_curator_env.sh
```

运行默认 Reviews/Amazon 配置：

```bash
cd /home/dev/graphdb/FilterVectorCode_refactor
experiments/curator_baseline/run_curator_baseline_with_env.sh experiments/curator_baseline/config.json
```

运行测试：

```bash
cd /home/dev/graphdb/FilterVectorCode_refactor
PYTHONPATH=thirdparty/curator-v2 .venv_curator/bin/python \
  -m unittest discover -s experiments/curator_baseline -p 'test_*.py' -v
```

只想强制重建 Curator 索引时，可以删除对应数据集的索引目录：

```bash
rm -rf /home/dev/graphdb/FilterVectorResult/<dataset>/index/Curator/index_files
```

删除后再次运行会重新构建索引。

## 8. 搜索过滤计算口径

当前 runner 默认使用中间态在线查询口径：

- 每个 query 在每个 `search_ef` 下都会实时计算 containment 过滤集合。
- 默认 `search.filter_mode=indexed`：base labels 构建一次倒排 posting list；query 到来时实时做 posting-list 交集。
- 如果要完全朴素扫描，可以设置 `search.filter_mode=scan`，会遍历全部 base label sets 判断 query labels 是否为子集。
- 过滤集合计算、external id 到 internal vid 的转换/排序和 Curator 搜索都计入 `Average_Time_ms`。
- 不会跨 `search_ef` 提前预计算或复用 query filter。
- 如果 Curator-v2 FAISS 扩展支持 `get_label_to_vid_mapping` 和 `search_with_bitmap_filter_optimized`，单次 query 内会用 sorted internal vids 走 optimized search。

## 9. 注意事项

- 当前 baseline 使用项目已有的 `.bin` 向量、label txt、groundtruth bin 格式。
- 当前过滤语义是 containment，查询 label 集合必须被 base label 集合包含。
- Curator-v2 的真实核心搜索参数是 `search_ef`，不是 UNG 的 `Lsearch` 或 IVF 的 `nprobe`。
- `search_time_summary.csv` 保留耗时、召回率和三个批次平均计数字段，其中 `Lsearch` 表示 Curator 的 `ef/search_ef`。
- 如果修改 `build` 配置，索引复用会自动失效并重新构建。
