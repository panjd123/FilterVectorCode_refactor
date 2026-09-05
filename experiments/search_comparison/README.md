# UNG Search Comparison

这个实验用于在已有的 `UNG` / `UNG_special_blocks` 索引上跑查询任务，并比较不同 search route 的结果。当前配置主要覆盖 Amazon、Genome、Reviews、VariousImg 等数据集。

Amazon SpecialBlockTrie 当前最优 `ordered + intra cap16` 的索引构建、搜索命令和输出说明见：

```text
experiments/search_comparison/README_SPECIAL_BLOCK_TRIE_CAP16_CN.md
```

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
- 顶层 `search`: 修改默认的 `K`、`num_threads`、`num_repeats`、`lsearch_start/end/step` 等搜索参数

`methods[]` 可以设置 `search` 覆盖该算法的搜索参数。合并优先级为顶层
`search`、`datasets[].search`、`methods[].search`（后者优先）。例如让两个算法使用不同的
`Lsearch` 步长：

```json
"methods": [
  {
    "name": "Trie_block_hybrid",
    "search": {
      "lsearch_step": 500
    }
  },
  {
    "name": "UNG__hybrid",
    "search": {
      "lsearch_step": 1000
    }
  }
]
```

算法级 `search` 也可覆盖 `lsearch_start`、`lsearch_end` 和 `lsearch_values`；相应的结果目录会使用
该算法实际生效的 `lsearch_start/lsearch_step/lsearch_end`。设置 `lsearch_values` 时，显式列表优先于步长。

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

## SpecialBlockTrie 入口组缓存时机

如果需要测量每次搜索都重新计算 ELS、不复用 query-label 对应的入口组和
route statistics，可在 method 中设置：

```json
{
  "name": "Trie_block_hybrid_no_els_reuse",
  "entry_group_provider": "special_block_trie",
  "reuse_els": false
}
```

`reuse_els` 默认为 `true`。设为 `false` 后，每个 Lsearch 和每个 repeat 都会为
每条 query 重新计算 ELS；脚本也会自动禁用 CPU ELS 和 SpecialBlockTrie 预热。
该模式同时适用于 `cpu_bruteforce_els` 和 `special_block_trie`。用于 ELS 计算的
静态 label bitmap/Trie 索引仍会保留；关闭的只是跨 query、Lsearch 和 repeat 的
ELS 结果复用。

如果只需要入口 group 和后续图搜索、不需要 descendants/coverage route statistics，
可在 method 的 `env` 中设置 `UNG_DISABLE_ENTRY_ROUTE_STATS=1`。该开关不会跳过
入口 group 计算；它只跳过对 LNG descendants 和 coverage Roaring bitmap 的合并，
对应的候选覆盖统计字段会输出为 0。

之前的 Block-HNSW 上层路由实验已经移除。当前高选择率优化保留完整的
free-state block 搜索，不再硬裁剪候选 block。对于父 block 到子 block 的高扇出，
可以用 `UNG_SPECIAL_FREE_INTER_EDGE_SCAN_CAP` 限制每个 free 源点扫描的
inter-block 边总数，也可以用 `UNG_SPECIAL_FREE_INTER_EDGE_PER_BLOCK_CAP` 限制
每个目标子 block 的 inter-block 边数；未设置或设为 `null` 时不限制。后者适合
进行每个父子 block 访问 1/2/4 条的 Recall 和耗时对比。
如果只希望限制高扇出的父 block，可配合
`UNG_SPECIAL_FREE_INTER_EDGE_CAP_MIN_CHILD_BLOCKS`；例如设置为 `8` 时，只有
拥有至少 8 个子 block 的父 block 才应用上述 cap，其他 block 保持原始搜索路径。

对于 `entry_group_provider: "special_block_trie"`，可在 method 中设置：

```json
"warmup_special_block_trie": false
```

默认值为 `true`，会在计时前为所有不重复的 query-label 集合计算并缓存入口组。设为
`false` 时跳过该阶段；第一个 `Lsearch` 的 query 会按需计算入口组并写入同一缓存，因此
这部分耗时计入第一个 `Lsearch`，后续 `Lsearch` 和 repeat 复用缓存。

本次实验整体日志会写在配置文件同目录下：

```text
experiments/search_comparison/search_comparison_<YYYYMMDD_HHMMSS>.log
```
