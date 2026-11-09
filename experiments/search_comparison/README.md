# UNG Search Comparison

这个实验用于在已有的 `UNG` / `UNG_special_blocks` 索引上跑查询任务，并比较不同 search route 的结果。当前配置主要覆盖 Amazon、Genome、Reviews、VariousImg 等数据集。

Amazon SpecialBlockTrie 当前最优 `ordered + intra cap16` 的索引构建、搜索命令和输出说明见：

```text
experiments/search_comparison/README_SPECIAL_BLOCK_TRIE_CAP16_CN.md
```

## 如何跑查询任务

从当前代码目录启动：

```bash
cd /home/dev/graphdb/FilterVectorCode_multilevel_special
./run_search_comparison_experiment.sh experiments/search_comparison/config_genome.json
```

如果不传配置文件，脚本默认使用：

```text
/home/dev/graphdb/FilterVectorCode_multilevel_special/experiments/search_comparison/config_genome.json
```

脚本从 JSON 读取 `build_dir`，验证对应的 `apps/search_UNG_index`，然后调用：

```bash
python3 experiments/search_comparison/run_search_comparison.py experiments/search_comparison/config_genome.json
```

如果已经确认代码编译完成，也可以直接跑 Python 脚本：

```bash
cd /home/dev/graphdb/FilterVectorCode_multilevel_special
python3 experiments/search_comparison/run_search_comparison.py experiments/search_comparison/config_genome.json
```

对于原生 UNG 方法，可以设置顶层 `"auto_build_ung": true`。runner 会在搜索前检查
`<result_root>/<Dataset>/index/<index_name>/index_files/` 是否包含完整、可搜索的 UNG
索引；缺失时调用当前 `build_dir/apps/build_UNG_index` 自动构建。构建配置读取顶层
`build` 和 `ung_env`。原生 UNG 自动构建强制使用 `UNG_BASE_GROUP_TOPOLOGY=lng`，不允许
`UNG_HIERARCHY_LAYERS` 或任何启用的 `UNG_SPECIAL_*` 选项。原生 UNG 搜索也不会加载
`block_index_files`，即使索引目录旁存在遗留的 block sidecar。

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
该算法实际生效的搜索范围。设置 `lsearch_values` 时，显式列表优先于步长。

需要在主搜索区间之前增加一段更密的低预算搜索时，可配置：

```json
"lsearch_prefix_start": 100,
"lsearch_prefix_end": 1000,
"lsearch_prefix_step": 100
```

例如主区间为 `1000..20000/1000` 时，实际搜索点为
`100,200,...,1000,2000,...,20000`，边界 `1000` 不会重复。结果目录使用实际起点、
前置步长和实际终点（`_<query>_100_100_20000`）。前置区间只加入严格小于当前 method
主区间起点的值，因此 method 若单独覆盖为 `10..100/10`，不会被继承的前置区间改变。

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

### 多层 Trie block 搜索

`config_amazon_multilevel_trie_block.json` 启用 `1000:trie,16000:trie`
两层 block。搜索状态由 `(point_id, activation_level)` 标识：同一个点可以在多个
block 层分别展开，但其向量距离通过 query-local 全局缓存只计算一次。

入口与后续发现的节点会保留当前层，并生成所有更高的 query-free 层状态，按最高层
优先展开；因此可以从基础图或第一层直接提升到第二层，但上层状态不会反向生成较细
层状态。层间移动严格单调向上。

`Lsearch` 只限制唯一 `point_id` 的结果 beam；同一点的多个
`(point_id, activation_level)` 状态进入独立路径 frontier，不重复占用 beam 容量。
每个授权 block 默认生成 16 个结构化 landmark，普通 block 保留最近 1 个，
query-free frontier block 最多保留 8 个。入口按层级从高到低处理，并在普通入口加入
全局竞争前，于各自所属层预展开每 block 最多 2 个 landmark。
多层模式默认使用 heap，配置仍显式设置：

```json
"UNG_SPECIAL_CANDIDATE_HEAP": "1",
"UNG_SPECIAL_MAX_ACTIVE_LEVEL": "2",
"UNG_SPECIAL_TRIE_BLOCK_SEEDS_PER_BLOCK": "16",
"UNG_SPECIAL_TRIE_BLOCK_SEEDS_RETAIN_PER_BLOCK": "1",
"UNG_SPECIAL_TRIE_FRONTIER_BLOCK_SEEDS_RETAIN_PER_BLOCK": "8",
"UNG_SPECIAL_TRIE_BLOCK_SEEDS_PREEXPAND_PER_BLOCK": "2"
```

`query_details_repeat*.csv` 中的 `SpecialCrossLayerStatesInserted` 和
`SpecialDistanceCacheHits` 可用于确认跨层跳跃和距离复用是否实际发生。

如果需要测量每次搜索都重新计算 ELS、不复用 query-label 对应的入口组，可在 method 中设置：

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

2026-10-19 起已移除 Idea1/Idea2 自适应选择器、UNG 内的 ACORN 位图搜索和
对应的全量后代/覆盖集合预计算。固定图搜索使用 `force_use_alg=1/2`，入口由
`entry_group_strategy` 选择；独立 NaviX 仍使用 `5`。`0/3/4` 会明确报错。
`UNG_DISABLE_ENTRY_ROUTE_STATS`、`UNG_DESCENDANTS_IMPL`、`UNG_COVERAGE_IMPL` 等
旧开关不再需要，相关统计及 CSV 列已删除。`skip_query_features`、
`skip_bitmap_comparison` 和模型路径仅作为旧命令兼容参数解析，不执行任何处理。

新构建只保存核心图、Trie、标签和块索引，不创建全局后代闭包、覆盖位图、属性
二分图或选择器特征。旧索引的这些文件会被加载器忽略；重建同一路径时清理其
过期文件。块激活所需的根标签、公共标签、成员和层次信息仍保留。
`favor_blocks`/`hnsw_blocks` 图后端的目标点集合改为按查询遍历组邻接关系生成，
不依赖全量预计算。独立 ACORN 基线不受此次删除影响。

CPU 集成回归（不运行 GPU kernel）：

```bash
UNG_GRAPH_ONLY_BIN_DIR=build_ung_multilevel_trie_gpu/apps \
python3 tools/tests/test_graph_only_index.py
```

可设置 `UNG_GRAPH_ONLY_REFERENCE_BIN_DIR` 指向修改前保存的应用目录，比较同一
数据的核心图文件和逐查询 Recall、距离计算数、访问节点数、入口数及边扫描数。

之前的 Block-HNSW 上层路由实验已经移除。当前高选择率优化保留完整的
free-state block 搜索，不再硬裁剪候选 block。对于父 block 到子 block 的高扇出，
可以用 `UNG_SPECIAL_FREE_INTER_EDGE_SCAN_CAP` 限制每个 free 源点扫描的
inter-block 边总数，也可以用 `UNG_SPECIAL_FREE_INTER_EDGE_PER_BLOCK_CAP` 限制
每个目标子 block 的 inter-block 边数；未设置或设为 `null` 时不限制。后者适合
进行每个父子 block 访问 1/2/4 条的 Recall 和耗时对比。
如果只希望限制高扇出的父 block，可配合
`UNG_SPECIAL_FREE_INTER_EDGE_CAP_MIN_CHILD_BLOCKS`；例如设置为 `8` 时，只有
拥有至少 8 个子 block 的父 block 才应用上述 cap，其他 block 保持原始搜索路径。

渐进式 query-time block router 可通过以下环境变量显式启用：

```json
"UNG_SPECIAL_BLOCK_ROUTER_TOP_R": "4",
"UNG_SPECIAL_BLOCK_ROUTER_PORTALS_PER_BLOCK": "8",
"UNG_SPECIAL_BLOCK_ROUTER_LB_SHORTLIST_MULTIPLIER": "4",
"UNG_SPECIAL_BLOCK_ROUTER_RECHECK_INTERVAL": "32"
```

启用后，搜索只从 query-free block tree 的 frontier roots 开始。父 block 首次展开时，
先按 child 完整子树的 centroid/radius 计算 L2 lower bound，再对一个 lower-bound
window 中的 child 使用共享 portal（block entry point 加结构化 group landmarks）
重排。`TOP_R` 表示每轮批量激活数量，而不是永久保留数量：未激活 child 留在
pending 队列中；每搜索 `RECHECK_INTERVAL` 个候选后，以当前 top-K 距离作为 τ，
继续激活满足 `lower_bound < τ` 的下一批。候选队列耗尽或 early-stop 前也会强制
recheck，因此 R 之外的 child 不会因首轮排名而被永久丢弃。

centroid/radius 自动保存在 block index 的 `special_block_router_bounds.bin`；文件头
校验维度、点数、block 数和 topology hash，拓扑变化后自动重建，不需要重新构造
special edges。设置 `UNG_SPECIAL_BLOCK_ROUTER_DISABLE_LOWER_BOUND=1` 可做
portal-only A/B。FLOAT/L2 以外的查询不会启用 lower-bound router。

router 模式仍检查逐点 inter-block edges，但只对已由当前批次激活的目标 child
保留原有导航边；指向 pending/已由 lower bound 剪枝 child 的边会被跳过。block 内
special graph 始终保留。
未设置 `UNG_SPECIAL_BLOCK_ROUTER_TOP_R` 或设为 `0` 时保持原路径，便于 A/B。
`query_details_repeat*.csv` 中的 `SpecialBlockRouter*` 字段记录 lower-bound、
shortlist、portal 打分/激活、阶段耗时和省略的 inter-edge 数量。
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


## Trie block 有无 block 消融（只补跑无 block）

使用同一份 Trie 基础图，对比启用 block overlay 与普通图搜索的 Recall/QPS。
有 block 组复用 `Trie_block_500_M_32_cross_block_4` 的已有结果；本轮只运行
`Trie_no_block`，不重新构建索引。

| 数据集 | 无 block 配置 | 查询数量 |
| --- | --- | --- |
| Amazon | `config_amazon_no_block.json` | 700 |
| VImg | `config_VImg_no_block.json` | 594 |
| Music | `config_music_no_block.json` | 537 |

查询任务统一为 `query_minlen1_sel10to40pct`，沿用对应 `config_amazon.json`、
`config_VImg.json`、`config_music.json` 的搜索参数：float/L2、containment、K=10、
100 线程、16 个入口、1 次 repeat。无 block 组当前补测 Lsearch 为
`10000..100000/10000`，共 10 个点，不包含低预算前置区间。
有 block 组复用已有 `100..1000/100` 加 `2000..20000/1000` 的 29 个点。
查询和 GT 复用原文件。

无 block 配置仍读取
`<result_root>/<Dataset>/index/Trie_block_500_M_32_cross_block_4/index_files/`，
保留相同的向量顺序、group、Trie 拓扑与基础图边。`entry_group_provider: "trie"`
与有 block 配置的兼容名称 `"special_block_trie"` 在当前代码中映射到同一入口策略；
使用 `trie` 可避免 runner 加载 `block_index_files`。
`special_block_search: false` 使 runner 清除继承的 `UNG_SPECIAL_*` 环境变量，
搜索走普通 `execute_ung_query`，不使用 block 授权、block 入口扩展及 special edges。
两组均关闭 ELS 结果复用（`reuse_els: false`）和入口 route statistics
（`UNG_DISABLE_ENTRY_ROUTE_STATS=1`）。不要开启 `auto_build_ung`；该功能构建 LNG
拓扑，与本实验要保留的 Trie 基础图不同。

在项目根目录一键顺序运行三个数据集：

```bash
./run_block_presence_ablation.sh
```

也可只运行一个数据集：

```bash
./run_search_comparison_experiment.sh experiments/search_comparison/config_amazon_no_block.json
```

只检查配置指定的搜索二进制，不开始查询：

```bash
UNG_SEARCH_PREFLIGHT_ONLY=1 ./run_block_presence_ablation.sh
```

结果分别位于：

```text
<result_root>/<Dataset>/results/Trie_no_block/query_minlen1_sel10to40pct_10000_10000_100000/results/
<result_root>/<Dataset>/results/Trie_block_500_M_32_cross_block_4/query_minlen1_sel10to40pct_100_100_20000/results/
```

每组用 `search_time_summary_qps.csv` 比较 Recall/QPS 曲线，并在相近 Recall 下比较
QPS；相同 Lsearch 的 Recall 和耗时可作为补充。无 block 结果独立保存，不覆盖已有
有 block 结果。之前无 block 的 `query_minlen1_sel10to40pct_100_100_20000` 结果保留在
原目录；本次大预算结果写入新目录。runner 默认跳过已覆盖当前全部 10 个点的
完整无 block 结果，失败后可重跑脚本。

配置布置时已核对三个数据集的有 block summary 均覆盖全部 29 个点，查询 bin、
标签和 GT 均存在。复用历史性能结果时，应核对原运行日志中的代码/二进制版本和
机器负载；若其与本次运行不同，该对比只能作为历史参照，严格的 block 消融需在
相同版本和环境下补跑有 block 组。本次启动脚本只包含无 block 组。
