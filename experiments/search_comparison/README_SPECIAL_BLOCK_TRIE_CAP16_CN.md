# Amazon 独立 SpecialBlockTrie 索引构建与搜索

本文说明如何从零运行一轮 Amazon 的两阶段索引构建和搜索：

1. 构建纯 UNG 索引；
2. 读取纯 UNG，构建独立的 Trie block sidecar；
3. 同时加载两套索引运行 `query_selected_recall_advantage` 搜索。

block sidecar 不保存向量内容，只保存 block/trie 元数据、向量 ID 和边。搜索时的向量数据仍由
UNG/基础数据提供。

这里的“纯 UNG”是指 UNG 目录中不再混入 block sidecar，并不表示当前通用 UNG 搜索加载器
已经裁掉 LNG。推荐构建配置仍保留 `UNG_LNG_IMPL=1`，`index_files/` 中的
`lng_out_neighbors.dat`、descendants/coverage 等仍属于当前 UNG 自身。Trie block 分区不依赖
LNG 来组织 block，正常 Trie 搜索也使用 sidecar 中的 `special_trie_regular_edges.bin`，不会在
block sidecar 中重复保存一份 LNG。当前这一轮不要手工删除 UNG 的 LNG 文件；进一步做“无
LNG 的专用最小加载器”需要单独修改并验证通用 UNG 加载流程。

## 推荐配置

```text
index: UNG_special_block_trie_regular
entry provider: special_block_trie
candidate queue: ordered（heap 关闭）
lazy block activation: 关闭
free intra-edge scan cap: 16
Lsearch: 2500
K: 10
search threads: 100
repeats: 3
```

历史实测结果为 `Recall=0.952714`、汇总 `QPS=2100.59`。cap16 是搜索参数，不改变索引
格式，因此不需要为 cap16 单独构建索引。

## 目录与输入

所有命令从仓库根目录运行：

```bash
cd /home/dev/graphdb/FilterVectorCode_refactor
```

主要输入为：

```text
/home/dev/graphdb/FilterVectorData/Amazon/Amazon_base.bin
/home/dev/graphdb/FilterVectorData/Amazon/Amazon_base_labels.txt
/home/dev/graphdb/FilterVectorData/Amazon/query_selected_recall_advantage/Amazon_query.bin
/home/dev/graphdb/FilterVectorData/Amazon/query_selected_recall_advantage/Amazon_query_labels.txt
```

构建配置和搜索配置中的 `index_name` 必须一致。当前两份推荐配置均使用：

```text
UNG_special_block_trie_regular
```

## 1. 首次干净复跑前处理旧索引

当前结果目录可能是解耦前生成的旧混合索引，其中 block 文件位于 `index_files/`。构建脚本会
复用固定目录，但不会替你删除所有历史文件。为了得到能够明确验证的新目录，先将旧目录改名
备份：

```bash
old=/home/dev/graphdb/FilterVectorResult/Amazon/index/UNG_special_block_trie_regular
if [ -e "$old" ]; then
  mv "$old" "${old}_legacy_$(date +%Y%m%d_%H%M%S)"
fi
```

只需在需要从零重建时执行这一步。不要在另一个构建或搜索进程正在使用该目录时执行。

如果希望保留固定目录并用新名字构建，则同时修改下面两份 JSON 中的 `index_name`：

```text
experiments/cpu_special_blocks/config_special_block_trie_regular.json
experiments/search_comparison/config_amazon_special_block_cap16_best.json
```

## 2. 构建纯 UNG 和独立 block 索引

运行：

```bash
gpulock perf --wait-gpu-idle 0 -- \
  ./run_cpu_special_blocks_experiment.sh \
  experiments/cpu_special_blocks/config_special_block_trie_regular.json
```

脚本会编译：

```bash
cmake -S UNG/codes -B build_ung_rel -DCMAKE_BUILD_TYPE=Release
cmake --build build_ung_rel -j16 \
  --target build_UNG_index build_special_block_index search_UNG_index
```

然后顺序执行两个独立构建器：

```text
build_UNG_index              -> index_files/、results/
build_special_block_index    -> block_index_files/、block_results/
```

第二阶段通过 `--ung_index_path_prefix` 读取第一阶段的 UNG 索引，通过
`--block_index_path_prefix` 将 block sidecar 写到另一个目录。完整输出布局为：

```text
/home/dev/graphdb/FilterVectorResult/Amazon/index/UNG_special_block_trie_regular/
├── index_files/             # 纯 UNG 索引
├── block_index_files/       # 独立 Trie block sidecar
├── results/                 # UNG build_time.csv
├── block_results/           # block build_time.csv
└── others/
    ├── build.log            # 两阶段完整构建日志
    └── time.txt             # wall time、最大 RSS、pipeline wall time
```

block sidecar 默认必须存在以下 5 个文件：

```text
meta
special_blocks.bin
special_block_trie.bin
special_edges.bin
special_trie_regular_edges.bin
```

`special_blocks.csv`、`special_block_members.csv`、`special_block_children.csv` 只用于人工诊断，
默认不写出。需要它们时，在构建前设置 `UNG_SPECIAL_BLOCK_METADATA_CSV=1`；搜索不依赖这些
CSV。边文件默认使用紧凑 CSR v2 格式。

构建结束后验证两套索引：

```bash
root=/home/dev/graphdb/FilterVectorResult/Amazon/index/UNG_special_block_trie_regular

test -f "$root/index_files/meta"
test -f "$root/block_index_files/meta"
test -f "$root/block_index_files/special_blocks.bin"
test -f "$root/block_index_files/special_block_trie.bin"
test -f "$root/block_index_files/special_edges.bin"
test -f "$root/block_index_files/special_trie_regular_edges.bin"

# 干净的新布局中，block 文件不应再写入纯 UNG 目录。
test ! -e "$root/index_files/special_block_trie.bin"
test ! -e "$root/index_files/special_edges.bin"
```

任何一条 `test` 失败时，先查看：

```bash
tail -n 100 "$root/others/build.log"
cat "$root/others/time.txt"
```

## 3. 查看 block 构建时间、磁盘和内存统计

```bash
root=/home/dev/graphdb/FilterVectorResult/Amazon/index/UNG_special_block_trie_regular

cat "$root/block_index_files/meta"
cat "$root/block_results/build_time.csv"
du -h "$root/block_index_files"/*
```

重点字段为：

```text
build_time(ms)                  # block sidecar 总构建时间
disk_bytes                      # 5 个默认持久化文件的总字节数
build_memory_logical_bytes      # 构建态有效元素的逻辑字节数
build_memory_allocated_bytes    # 构建态容器实际分配容量的估算字节数
loaded_memory_logical_bytes     # 保存后重新加载的有效元素字节数
loaded_memory_allocated_bytes   # 搜索加载态容器实际分配容量的估算字节数
compact_reload_time(ms)         # 保存后重新加载紧凑索引的耗时
```

这里的内存统计是各运行时容器按 `sizeof(element) * count/capacity` 累加的索引内存估算，不是
进程 RSS。`others/time.txt` 中的 `max_rss_kb` 是纯 UNG 构建进程的外部峰值 RSS；当前脚本
追加运行 block 构建器，所以它不能替代上述 block 自身的加载态统计。

## 4. 运行 Amazon 搜索

构建成功后运行：

```bash
python3 experiments/search_comparison/run_search_comparison.py \
  experiments/search_comparison/config_amazon_special_block_cap16_best.json
```

搜索 runner 会加载：

```text
--index_path_prefix       .../index_files/
--block_index_path_prefix .../block_index_files/
```

只要 `block_index_files/meta` 存在，第二个参数会自动添加。旧混合索引仍可兼容加载，但本次
干净构建应走独立 sidecar 路径。

搜索配置还会设置：

```text
UNG_SPECIAL_BLOCK_SEARCH=1
UNG_SPECIAL_FREE_INTRA_EDGE_SCAN_CAP=16
UNG_SPECIAL_CANDIDATE_HEAP unset
UNG_SPECIAL_TRIE_LAZY_BLOCK_ACTIVATION unset
UNG_SPECIAL_SEARCH_MODE=free_state
--entry_group_provider special_block_trie
--Lsearch 2500
```

如果只改了搜索代码，可先单独重编译后再运行，无需重建索引：

```bash
cmake -S UNG/codes -B build_ung_rel -DCMAKE_BUILD_TYPE=Release
cmake --build build_ung_rel -j16 --target search_UNG_index

python3 experiments/search_comparison/run_search_comparison.py \
  experiments/search_comparison/config_amazon_special_block_cap16_best.json
```

也可用封装脚本编译后搜索：

```bash
./run_search_comparison_experiment.sh \
  experiments/search_comparison/config_amazon_special_block_cap16_best.json
```

## 5. 查看搜索结果

输出目录为：

```text
/home/dev/graphdb/FilterVectorResult/Amazon/results/
  special_trie_ordered_intra_cap16_best_repeat3/
  query_selected_recall_advantage_2500_250_2500/
```

主要文件为：

```text
results/search_time_summary.csv       # 平均耗时和 Recall
results/search_time_summary_qps.csv   # 额外包含 QPS
results/search_time_details.csv       # 每次 repeat 的批次耗时
results/query_details_repeat3.csv     # 每条查询的距离、扫边和队列统计
others/Amazon_search_output.txt       # 完整日志和实际搜索命令
```

读取最终汇总：

```bash
cat /home/dev/graphdb/FilterVectorResult/Amazon/results/\
special_trie_ordered_intra_cap16_best_repeat3/\
query_selected_recall_advantage_2500_250_2500/\
results/search_time_summary_qps.csv
```

如果 ground truth 不存在，搜索 runner 会自动调用 `build_ung_rel/tools/compute_groundtruth`，并
写入：

```text
/home/dev/graphdb/FilterVectorResult/Amazon/GroundTruth/
query_selected_recall_advantage/Amazon_gt_labels_containment.bin
```

用于复现 cap16 与原路径 A/B 对比的配置仍为：

```text
experiments/search_comparison/config_amazon_special_block_cap16_reverse_ab.json
```
