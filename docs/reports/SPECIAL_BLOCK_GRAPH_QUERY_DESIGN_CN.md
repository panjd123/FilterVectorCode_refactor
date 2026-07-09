# 特异块图构建与自由状态查询设计

本文记录新的特异块构造、建图和查询语义。旧的 `tiny_group_size` / `tiny_group_ratio` 定义全部废弃，不再作为主设计依据。

## 1. 特异块定义

输入是一棵 label-set trie/group tree。每个 terminal group 有若干数据点。每个 trie node 的子树点数为该 node 自身 terminal group 点数加所有后代 terminal group 点数。

新的特异块自底向上构造：

```text
initial uncovered_points(node) = subtree_points(node)
postorder 自底向上遍历 trie node
当 uncovered_points(node) > SPECIAL_BLOCK_MIN_POINTS 时，在该 node 建立一个特异块
这个特异块吸收 node 子树中尚未被更低层特异块吸收的 terminal groups
然后将 node 对父亲贡献的 uncovered_points 置为 0
```

默认阈值先用：

```text
SPECIAL_BLOCK_MIN_POINTS = 100
```

适用数据口径：

```text
特异块只面向正常原始数据集，例如 Amazon 100% x1。
不要直接在 x40/x200/x400 repeat 数据上构造特异块。
如果输入来自 repeat 数据，必须先还原 x1 标签/点数口径，或者明确按 repeat 倍数缩放阈值后只做诊断。
```

代码保护：

```text
启用 UNG_SPECIAL_BLOCKS=1 时，必须同时设置：
UNG_SPECIAL_BLOCK_DATA_MODE=x1

否则 build 会直接报错，避免误在 repeat 数据上构造特异块。
```

注意：

- 这里不再检查 tiny group size。
- 这里不再检查 tiny group ratio。
- 特异块可以只包含根 node 自身的一个 terminal group；这种块叫平凡特异块。
- 特异块之间形成一棵/一个 DAG：高层特异块在向下收集成员时，如果遇到低层特异块就停止，并记录一条从高层特异块到低层特异块的特异块边。

## 2. 特异块成员与边界

对每个特异块维护：

```text
block_id
root trie node
root_group_id，如果 root trie node 本身是 terminal group，否则为 0
member_group_ids：该 block 吸收的 terminal groups，不穿过子特异块
child_block_ids：向下遇到的直接子特异块
point_count：member groups 的点数之和
subtree_point_count：root trie node 完整子树点数
is_trivial：member_group_ids 只有 root_group_id 且没有 child_block_ids
```

这里 `point_count` 是这个块真正吸收的点；`subtree_point_count` 可能包含子特异块里的点。

## 3. 查询相关术语

查询阶段需要区分“块存在”和“当前 query 是否完整覆盖该块”：

```text
查询特异块：
  如果 query labels 被某个特异块的 root label-set 完整包含，
  则该 query 完整覆盖这个特异块；该块是查询特异块。

特异点：
  查询特异块中的点。

平凡特异点：
  如果查询特异块是平凡特异块，则其中的点叫平凡特异点。

普通点：
  如果 query 只覆盖某个特异块中的一部分 group，
  例如 query 对应的是该特异块内部的一个或多个更小子树，
  但没有完整覆盖该特异块 root，则这些候选点仍按普通点处理。
```

覆盖率指标：

```text
特异组大小临界线：
  `SPECIAL_BLOCK_MIN_POINTS`，即默认的 100。

非平凡特异组覆盖率：
  原始数据点中，属于非平凡特异块 member groups 的点数比例。

查询特异组覆盖率：
  在真实 query 的 matched points 加权总和中，属于查询特异块的点数比例。

非平凡查询特异组覆盖率：
  在真实 query 的 matched points 加权总和中，属于非平凡查询特异块的点数比例。
```

注意：查询覆盖率按点数直接加和，不对每个 query 的比例取平均。这样与查询工作量近似成正比。

## 4. 普通建图如何跳过平凡特异块

普通 UNG 建图仍然保留原有 group graph 和 cross-edge 语义，但启用特异块优化后：

```text
跳过平凡特异块内部的普通组内图构建。
跳过从平凡特异块出发的普通跨组边构建。
不跳过从其他组到平凡特异块内点的普通跨组边。
```

这保证普通图不会为“将由特异块图覆盖的平凡根”重复建边，但仍允许普通搜索从外部进入平凡特异块。

## 5. 特异边

在普通图上叠加特异块图：

```text
特异块内部构建块内组内图。
特异块之间构建跨块边。
这些边带 edge metadata，称为特异边。
```

特异边需要记录：

```text
edge target point id
special_block_id
edge kind: intra_special_block 或 inter_special_block
```

跨块边保存父特异块 block_id。后续查询遇到自由节点时，可以优先沿特异边扩展。

当前现有 `Graph::neighbors` 只保存 neighbor id，没有边类型或 block id。因此特异边不应直接混入普通 `Graph::neighbors` 后丢失类型；需要新增 parallel sidecar adjacency，例如：

```text
_special_edges_by_point[point_id] = vector<SpecialEdge>
```

## 6. 查询自由状态

查询时每个候选点维护状态：

```text
free = true / false
```

规则：

1. 入口组如果是某个特异块根组，则所有初始化候选点都是 free。
2. 非 free 节点扩展普通边时，如果扩展到一个新的特异块根节点，则这个新节点标记为 free。
3. free 节点扩展出来的节点都保持 free。
4. free 节点可以走特异边，也可以走普通边；默认只走特异边，以获得更快查询。
5. 非 free 节点只能走普通边。

这要求 search queue 的候选状态从单纯 `(id, distance, expanded)` 扩展为至少：

```text
id
distance
expanded
free
```

如果需要严格避免不同状态下同一点被错误去重，visited set 也需要考虑 `(point_id, free_state)` 或保守地让 free 状态升级已有候选。

## 7. 当前实现状态

已实现的 feature flags：

```text
UNG_SPECIAL_BLOCKS=1
UNG_SPECIAL_BLOCK_DATA_MODE=x1
UNG_SPECIAL_BLOCK_MIN_POINTS=100   # 可改为 200/400/800/1600
UNG_SPECIAL_BLOCK_SEARCH=1         # search 阶段启用 free-state 查询
UNG_SPECIAL_BLOCK_FREE_USE_REGULAR=1 # A/B：free 节点同时走普通边
```

实现内容：

- `UNG_SPECIAL_BLOCKS=1` 时按 bottom-up `uncovered_points > threshold` 构造特异块。
- 构造结果保存为 `special_blocks.csv`、`special_block_members.csv`、`special_block_children.csv`。
- 特异边保存为 sidecar adjacency，不写入普通 `Graph::neighbors`。默认兼容格式是 `special_edges.csv`；当前也支持 `UNG_SPECIAL_EDGE_BINARY=1` 双写二进制 sidecar，以及 `UNG_SPECIAL_EDGE_BINARY_ONLY=1` 只写 `special_edges.bin` / `special_heavy_edges.bin`。
- load 阶段会恢复 block metadata、point/group 映射和 sidecar special edges。
- 平凡特异块根组的普通组内图不写入最终普通图；但会用 scratch graph 保留 Vamana target index，保证其他 group 到平凡块的普通 cross-edge 仍可构建。
- 普通 cross-edge 和 additional_edges 的 source work 会跳过平凡特异块根组；target 侧不跳过。
- 启用特异块时，普通 group graph / cross-edge 不再简单强制 CPU。GPU ordinary cross 支持 trivial special source-mask，并对未验证 route fail closed；special intra/inter 也有独立 CPU/GPU 分流。
- `UNG_SPECIAL_BLOCK_SEARCH=1` 时，查询候选维护 regular/free 双状态 visited；同一个 point 被 regular 访问过后仍允许以 free 状态再次入队。
- free 节点默认只走 sidecar special edges；设置 `UNG_SPECIAL_BLOCK_FREE_USE_REGULAR=1` 可同时走普通边做保守 A/B。

需要严格说明的边界：

- 当前已把特异块视为一种 special group：块内图使用 special-intra 独立分流，而不是直接复用普通 group complete threshold。CPU 默认对 `n<=UNG_SPECIAL_INTRA_COMPLETE_NX` 的 special block 走 distance exact-topK，默认阈值为 `2048`，每点保留最多 `max_degree` 个最近邻；更大的 block 仍通过非连续点集 storage view 调用 CPU Vamana。GPU 路径默认不采用 CPU 的 `2048` 阈值，而是使用独立的 `128` 阈值：T=100 下 `530` 个小 block / `59,847` 点走 exact-topK，其余 `1444` blocks / `517,975` 点 pack 后送 Tagore/FastGrnnd CUDA。`UNG_SPECIAL_INTRA_BOUNDED_COMPLETE` 仅保留为诊断/负结果开关；bounded ring 会伤 recall，不应作为默认。
- 父特异块到子特异块的 inter special edges 复用普通 CPU cross 的 target-search helper：`target_nx <= UNG_CPU_TARGET_EXACT_MAX_NX` 或 `nq * nx <= UNG_CPU_HYBRID_SCAN_MAX_WORK` 时走 exact scan，否则在 child special group 的 Vamana/complete 图上搜索，再按 `_num_cross_edges` 保留 topK。edge metadata 保存父 block id。
- 性能边界：CPU special intra 会调用 `Vamana::build(..., 1)`，而 Vamana 内部会设置全局 OpenMP 线程数；special inter CPU fallback 必须在进入 parallel loop 前恢复 `_num_threads`，并在 pragma 上显式 `num_threads(_num_threads)`，否则会被前序 CPU intra 污染成近似单线程。
- GPU special intra 已做大数据性能接入。旧 CPU baseline 因 `complete_threshold=64 < T=100` 把 `1974` blocks / `577822` points 全部送入 CPU Vamana，耗时 `155.92 s`；修正 CPU default 后，distance exact-topK `th=2048` 的 CPU intra 为 `75.88 s`，coverage L50/L200 smoke 为 `0.341 / 0.400`，接近旧 CPU Vamana `0.344 / 0.401`。GPU default 使用 `th=128` split 后，intra 为 `5.11-5.28 s`，coverage L50/L200 为 `0.344 / 0.403`；相对修正 CPU baseline 仍有约 `14.4x` stage speedup。GPU inter 使用 source-exact cuda-core kernel，当前默认 `UNG_SPECIAL_GPU_INTER_WARPS=2`，把 kernel 从旧 `~5.34 s` 降到 `~1.87 s`、inter total 降到 `~2.7 s`；TF32 WMMA tile 原型因 tiles 过碎是负结果。`th>=256` 会把过多 blocks 交给 CPU exact-topK，构建变慢；旧 `30.24x` 只能作为未修正 CPU Vamana baseline。
- 默认普通搜索不应直接用于 special index 的质量结论，因为普通图已经按设计跳过了平凡块出边；必须用 `UNG_SPECIAL_BLOCK_SEARCH=1` 的 filtered-search recall 验证。
- `UNG_SKIP_REORDERED_EXPORT=1` 可在不需要 ACORN route 的 UNG special-block 实验中跳过 reordered vector/label 导出；生成的 index 不适合 ACORN 对照。
- `UNG_SKIP_LNG_TEXT_SETS=1` 只跳过 `covered_sets`、`lng_descendants_num`、`lng_descendants` 三个文本缓存，仍保存 roaring bitmaps 和 `lng_out_neighbors.dat`。加载侧可在文本缺失时依赖 roaring 缓存完成 CPU entry-provider / UNG special search；但 GPU cover-frontier provider 当前仍直接依赖 `_lng_descendants`，因此该开关不是通用默认格式，除非后续为 GPU provider 增加重建或替代路径。

## 8. Threshold Sweep 统计

统计脚本：

```text
python3 tools/benchmarks/analyze_special_block_threshold_sweep.py \
  --base-label-file /home/graphdb/fv_runs/special_tiny_subtrees_20260702/amazon_100pct_x1_from_x40_labels.txt \
  --query-label-file /home/graphdb/FilterVectorBenchData/fv_amazon_1pct_x200_jitter_nonempty/query_coverage_1000/query_labels.txt \
  --thresholds 100 200 400 800 1600 \
  --out-prefix /home/graphdb/fv_runs/special_blocks_20260704/bottom_up_special_blocks
```

数据口径：

- base labels 是 Amazon 100% x1 口径，由 x40 repeat labels 每 40 行取 1 行恢复，`total_points=582117`，`total_groups=482387`。
- query labels 使用已有 Amazon 1% x200 `query_coverage_1000` workload，只用于真实 query 形态的加权覆盖统计；没有把 x200 repeat group size 当作原始 x1 group size。
- 查询覆盖率按所有 query 的 matched points 直接加和，不平均每个 query 的比例。

结果文件：

```text
/home/graphdb/fv_runs/special_blocks_20260704/bottom_up_special_blocks.md
```

| threshold | blocks | trivial | nontrivial | tree special points | tree special ratio | tree nontrivial ratio | query special ratio | query nontrivial ratio | query outside ratio | max block points |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 100 | 1974 | 36 | 1938 | 577822 | 0.992622 | 0.980052 | 0.832844 | 0.816222 | 0.167156 | 13236 |
| 200 | 1015 | 12 | 1003 | 576210 | 0.989853 | 0.982820 | 0.815744 | 0.806518 | 0.184256 | 20273 |
| 400 | 516 | 4 | 512 | 575639 | 0.988872 | 0.985398 | 0.796395 | 0.791865 | 0.203605 | 29173 |
| 800 | 246 | 0 | 246 | 574760 | 0.987362 | 0.987362 | 0.773756 | 0.773756 | 0.226244 | 49459 |
| 1600 | 131 | 0 | 131 | 574760 | 0.987362 | 0.987362 | 0.751926 | 0.751926 | 0.248074 | 58034 |

解释：

- 新 bottom-up uncovered 定义和早期 frontier 定义不同；它会形成层级块，并让父块吸收尚未被子块覆盖的点，因此全树覆盖率显著更高。
- 阈值越高，block 数越少，query special coverage 下降；这符合“更粗粒度块更少，但 query 完整覆盖块根的机会减少”的预期。

## 9. 编译与端到端验证

编译命令：

```text
cmake --build build_mode_switch -j 16 --target build_UNG_index search_UNG_index
```

结果：通过。

小型 x1 smoke artifact：

```text
/home/graphdb/fv_runs/special_blocks_20260704/smoke_x1
```

验证点：

- `UNG_SPECIAL_BLOCKS=1` 且没有 `UNG_SPECIAL_BLOCK_DATA_MODE=x1` 时拒绝构建；日志包含 x1/original 数据口径错误。
- special build 保存 `special_blocks.csv`、`special_block_members.csv`、`special_block_children.csv`、`special_edges.csv`。
- block-local direct smoke 中 special index：`special_block_count=6`，`special_edge_count=106`，其中 intra=70、inter=36。
- `search_UNG_index` filtered-search smoke：
  - default index 普通搜索：`Lsearch=6, recall=1.0`
  - special index 普通搜索：`Lsearch=6, recall=0.8`，说明普通图跳过后不能用默认搜索评价质量
  - special index + `UNG_SPECIAL_BLOCK_SEARCH=1`：`Lsearch=6, recall=1.0`

`run_end_to_end_recall_ab.sh` wrapper smoke：

```text
/home/graphdb/fv_runs/special_blocks_20260704/smoke_x1/wrapper_default
/home/graphdb/fv_runs/special_blocks_20260704/smoke_x1/wrapper_special
```

| run | index_ms | group_ms | cross_ms | Lsearch | avg_time_ms | avg_recall |
|---|---:|---:|---:|---:|---:|---:|
| wrapper default | 55.7923 | 10.5862 | 2.91178 | 6 | 1.94723 | 1 |
| wrapper special + free-state | 41.7041 | 8.43006 | 0.697471 | 6 | 1.82924 | 1 |

这些 smoke 结果只证明功能链路和 filtered-search recall 路径可执行；数据太小，不能作为性能结论。
