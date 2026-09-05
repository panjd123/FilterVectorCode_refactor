# LNG Special Blocks 对照实验

这个目录用于对比两种 UNG special block 构建方式：

- `trie_special_blocks`：原有 trie 前缀树切 block。
- `lng_special_blocks_bfs`：新增 LNG 生成树切 block，生成树模式为 BFS。
- `lng_special_blocks_random_seed1`：新增 LNG 生成树切 block，生成树模式为随机候选边，随机种子为 `1`。

默认配置以 `experiments/cpu_special_blocks/config.json` 为基准。除了索引名和下面几个 block 构建方式相关环境变量外，构建参数保持一致：

```text
UNG_SPECIAL_BLOCK_PARTITION=trie|lng
UNG_SPECIAL_BLOCK_TREE_MODE=bfs|random
UNG_SPECIAL_BLOCK_TREE_SEED=1
```

## 文件说明

```text
experiments/lng_special_blocks/
├── config.json                  # 实验配置
├── run_lng_special_blocks.py    # 构建索引 + 搜索的 Python runner
├── run_lng_special_blocks.sh    # 从仓库根目录调用的 shell 包装脚本
└── README_CN.md                 # 中文说明
```

## 默认路径约定

默认配置使用：

```text
data_root   = /home/dev/graphdb/FilterVectorData
result_root = /home/dev/graphdb/FilterVectorResult
build_dir   = /home/dev/graphdb/FilterVectorCode_refactor/build_ung_rel
```

索引输出：

```text
FilterVectorResult/<Dataset>/index/<index_name>/index_files/
```

搜索输出：

```text
FilterVectorResult/<Dataset>/results/<method_name>/<query_task>_<lsearch_start>_<lsearch_step>_<lsearch_end>/
```

搜索日志在：

```text
.../others/search.log
```

构建日志在：

```text
FilterVectorResult/<Dataset>/index/<index_name>/others/build.log
```

## 运行方式

从仓库根目录运行：

```bash
./experiments/lng_special_blocks/run_lng_special_blocks.sh
```

指定配置文件：

```bash
./experiments/lng_special_blocks/run_lng_special_blocks.sh \
  experiments/lng_special_blocks/config.json
```

只跑某一个方法：

```bash
./experiments/lng_special_blocks/run_lng_special_blocks.sh \
  experiments/lng_special_blocks/config.json \
  --only lng_special_blocks_bfs
```

只构建索引，不搜索：

```bash
./experiments/lng_special_blocks/run_lng_special_blocks.sh \
  experiments/lng_special_blocks/config.json \
  --build-only
```

只搜索已有索引：

```bash
./experiments/lng_special_blocks/run_lng_special_blocks.sh \
  experiments/lng_special_blocks/config.json \
  --search-only
```

检查配置和将要运行的方法，不真正执行：

```bash
./experiments/lng_special_blocks/run_lng_special_blocks.sh \
  experiments/lng_special_blocks/config.json \
  --dry-run
```

## 配置重点

`build_tools` 控制 runner 是否先编译 C++ 目标。默认策略是 `if_stale`：如果目标不存在，或 `UNG/codes` 下的源码/CMake 文件比目标二进制更新，就自动执行 CMake 编译。

```json
"build_tools": {
  "policy": "if_stale",
  "jobs_env": "BUILD_JOBS"
}
```

`policy` 可选值：

- `if_stale`：默认值，缺失或源码更新时编译。
- `if_missing`：只在目标不存在时编译，等同旧行为。
- `always`：每次运行前都编译相关目标。

`common_ung_env` 中的参数与 `cpu_special_blocks/config.json` 对齐，默认包括：

```json
"UNG_BUILD_PROFILE": "custom",
"UNG_GROUP_GRAPH_IMPL": "0",
"UNG_GET_MIN_SUPER_SETS_IMPL": "0",
"UNG_LNG_IMPL": "1",
"UNG_DESCENDANTS_IMPL": "1",
"UNG_COVERAGE_IMPL": "0",
"UNG_CROSS_EDGE_IMPL": "0",
"UNG_ADDITIONAL_EDGES_IMPL": "0",
"UNG_SPECIAL_BLOCKS": "1",
"UNG_SPECIAL_BLOCK_DATA_MODE": "x1",
"UNG_SPECIAL_BLOCK_SKIP_TRIVIAL": "1",
"UNG_SPECIAL_BLOCK_MIN_POINTS": "1000",
"UNG_SPECIAL_BLOCK_MAX_DEGREE": "128",
"UNG_SPECIAL_BLOCK_NUM_CROSS_EDGES": "10"
```

每个 `methods[]` 只覆盖 block 构建方式和索引名。例如 BFS LNG block：

```json
{
  "name": "lng_special_blocks_bfs",
  "index_name": "UNG_lng_special_blocks_hybrid_bfs",
  "ung_env": {
    "UNG_SPECIAL_BLOCK_PARTITION": "lng",
    "UNG_SPECIAL_BLOCK_TREE_MODE": "bfs",
    "UNG_SPECIAL_BLOCK_TREE_SEED": "1"
  },
  "search_env": {
    "UNG_SPECIAL_BLOCK_SEARCH": "1"
  }
}
```

## 数据集配置

默认示例使用 `Amazon`，并假设查询任务目录存在：

```json
"datasets": [
  {
    "dataset": "Amazon",
    "query_task": "query_A_B_C-sub-base-123456789"
  }
]
```

如果你的查询任务名字不同，直接修改 `query_task`。搜索时需要：

```text
FilterVectorData/<Dataset>/<query_task>/<Dataset>_query.bin
FilterVectorData/<Dataset>/<query_task>/<Dataset>_query_labels.txt
```

如果只有 `.fvecs`，runner 会在允许时调用 `tools/fvecs_to_bin` 自动转换。

如果 groundtruth 不存在，runner 会调用 `tools/compute_groundtruth` 自动生成：

```text
FilterVectorResult/<Dataset>/GroundTruth/<query_task>/<Dataset>_gt_labels_containment.bin
```

## 对照关系

这组实验的核心控制变量是 block 的切分来源：

```text
trie_special_blocks:
  UNG_SPECIAL_BLOCK_PARTITION=trie

lng_special_blocks_bfs:
  UNG_SPECIAL_BLOCK_PARTITION=lng
  UNG_SPECIAL_BLOCK_TREE_MODE=bfs

lng_special_blocks_random_seed1:
  UNG_SPECIAL_BLOCK_PARTITION=lng
  UNG_SPECIAL_BLOCK_TREE_MODE=random
  UNG_SPECIAL_BLOCK_TREE_SEED=1
```

其他 group graph、cross edge、special edge 和搜索参数默认保持一致，用于观察 LNG 生成树 block 与原 trie block 的构建时间、索引规模、Recall/QPS 差异。

## 常见问题

runner 默认会在目标缺失或源码比二进制更新时执行：

```bash
cmake -S UNG/codes -B <build_dir> -DCMAKE_BUILD_TYPE=Release
cmake --build <build_dir> -j${BUILD_JOBS:-16} --target <target>
```

如果需要完全复用已有二进制，可以把 `build_tools.policy` 改为 `if_missing`；如果希望每次实验前都重新编译，可以改为 `always`。

如果提示缺少 base 文件，请检查：

```text
FilterVectorData/<Dataset>/<Dataset>_base.bin
FilterVectorData/<Dataset>/<Dataset>_base_labels.txt
```

如果提示缺少 query task，请检查 `config.json` 里的 `query_task` 是否和数据目录一致。
