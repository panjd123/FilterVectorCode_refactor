# 当前优化状态、适用条件与挑战

本文总结 `FilterVectorCode_refactor` 当前围绕 UNG 构建阶段做过的主要优化、实测效果、生效条件、失败路径和后续挑战。重点覆盖最近围绕 SIFT1M/SIFT30、GPU cross-edge、Tagore GPU 建图、ANN batch query 和数据集构造的工作。

配套运行手册：

```text
docs/runbooks/RUNBOOK_OPTIMIZED_FEATURES_CN.md
```

阅读方式：

```text
本文回答：现在项目进展如何、做了哪些优化、什么条件下生效、还有哪些挑战。
运行手册回答：这些功能怎么编译、怎么跑、环境变量怎么配、结果文件怎么看。
```

## 0. 实测结果总览

这一节是目前做过的关键实验台账，避免只保留泛泛结论。更详细的运行命令见 `docs/runbooks/RUNBOOK_OPTIMIZED_FEATURES_CN.md`。

### 0.0 2026-06-02 最新状态快照

本文前半部分保留了 SIFT30、Tagore 接入和早期 GPU cross-edge 的历史台账。最新论文主线已经进一步收敛，读本文时应先按下面的状态理解：

| 模块 | 当前状态 | 可写结论 | 不能写成 |
|---|---|---|---|
| cross-edge | 当前主工程路线是 `UNG_UNIVERSAL_GPU=1`：target-centric descriptor batching + double-buffer + GPU global merge + flat-id output | SIFT30 skip-additional cross `4918.9 -> 3180.63 ms`；Amazon 1% x200 full-quality cross `2494.21 ms`，L1000/L5000 `0.871/0.911`；Amazon 1% x100 full-quality cross `1689.40 ms`，L100/L500/L1000 `0.826/0.868/0.891` | 不能写成所有 workload 无条件端到端加速；x100 Index 变慢，是 boundary result |
| source-centric no-lock | 修复 id-only null-distance 写入和 stream error check 后降级为 future direction | source-centric 遍历方向仍合理，后续应做 two-stage source grouped GEMM + per-source reduce | 不能把 legacy WMMA smoke 当成当前主表；可信 CUDA-core id-only SIFT30 cross `5926.67 ms`，慢于 target/universal |
| group graph | 当前最强候选是 workload-aware router：小组 CPU/bounded fallback，中组 packed exact-anchor，大组 FastGrnndCuda/reverse-tail | x200 packed exact-anchor group `3827.28 ms`，L1000/L5000 `0.869/0.908`；x400 packed exact-anchor L1000/L5000 `0.945/0.967`；x400 reverse-tail+repair Index `33044.9 ms`，接近同脚本 CPU recall | 不能写成 FastGrnndCuda 或 exact-anchor 已无损普遍替代 CPU Vamana |
| output boundary | 已用 reserve 和 `NeighborList64` 降低 CPU-compatible graph allocator 成本，但最终仍是 host graph/search 语义 | x200 NeighborList64 auto-reserve 把 reserve `2856.2 -> 25.4 ms`，tagore fill `224.8 -> 14.7 ms`；10%x40 reserve `6487.3 -> 21.2 ms` | 不能写成 flat/CSR 已完成；下一步仍是 GraphView/CSR + additional_edges flat backend |
| artifact gate | `python3 tools/benchmarks/run_submission_gate.py --final` 已通过 | 已登记 claim 均有本机 artifact 和关键数值校验 | gate 通过不等于科学上全部投稿完成，x400 参数扫、真实多标签和 flat/CSR 仍是缺口 |

因此当前总判断是：**cross-edge fused/universal route 是稳定贡献；组内图是 workload-aware speed/quality router；CPU-compatible output boundary 是下一阶段最大系统瓶颈。**

### 0.1 CPU 原始版本基线：各阶段耗时

这组数据来自旧 CPU 原始版本重新运行得到的完整构建结果，用来回答“最原始 CPU 版本每个部分到底耗时多少”。它不是 refactor 初始版本，也不是 SIFT30 实验，而是旧仓库 CPU 路径在 CelebA 上的基线。因此它适合用来解释早期 CPU 优化为什么优先针对 LNG / coverage / cross-edge，不能直接和后面的 SIFT30 GPU cross-edge 数字横向比较。

数据集与规模：

| 项 | 数值 |
| --- | ---: |
| 数据集 | `celeba` |
| base vectors | `202599` |
| 维度 | `512` |
| labels | `40` |
| groups | `115114` |
| vector-attribute edges | `1830201` |
| average descendants/group | `275.4` |

结果文件：

```text
/home/graphdb/FilterVectorCode_old_rerun_results/celeba_c2_20260519_111420/results/build_time.csv
/home/graphdb/FilterVectorCode_old_rerun_results/celeba_c2_20260519_111420/others/ung_build.log
```

完整阶段耗时：

| 阶段 | `build_time.csv` 字段 | 耗时 |
| --- | --- | ---: |
| 总构建 | `index_time` | `85028.9 ms` |
| 标签处理、分组、trie 准备 | `label_processing_time` | `557.93 ms` |
| 组内 PG 构建 | `build_graph_time` | `169.522 ms` |
| vector-attribute bipartite graph | `build_vector_attr_graph_time` | `105.21 ms` |
| LNG 构建 | `build_LNG_time` | `28241.5 ms` |
| descendants 计算 | `cal_descendants_time` | `3273.92 ms` |
| coverage ratio 计算 | `cal_coverage_ratio_time` | `11075.4 ms` |
| cross-group edges | `build_cross_edges_time` | `30139 ms` |

占总 `index_time` 的比例：

| 阶段 | 占比 |
| --- | ---: |
| `build_cross_edges_time` | `35.4%` |
| `build_LNG_time` | `33.2%` |
| `cal_coverage_ratio_time` | `13.0%` |
| `cal_descendants_time` | `3.9%` |
| `label_processing_time` | `0.7%` |
| `build_graph_time` | `0.2%` |
| `build_vector_attr_graph_time` | `0.1%` |

解释：

```text
CPU 原始版本/CelebA 基线里，最大两块是 cross-group edges 和 LNG 构建。
coverage 和 descendants 也明显占时间，因此早期 CPU 优化重点放在 LNG Phase1、descendants/coverage 数据结构和 cross-edge 上。
build_time.csv 里的 cross_edge_step1_time 等子字段是未初始化垃圾值，不能使用；只看 build_cross_edges_time。
```

### 0.2 稳定主线：SIFT30 CPU/GPU cross-edge A/B

数据集：`sift30_zipf_origstyle`

| 指标 | CPU baseline | GPU cross-edge | 加速/变化 |
| --- | ---: | ---: | ---: |
| `build_cross_edges_time` | `44560 ms` | `5002 ms` | `8.9x` |
| `index_time` | `64104.6 ms` | `23555.1 ms` | `2.7x` |
| `build_graph_time` | 同量级 CPU Vamana | `10001.6 ms` | 仍是 CPU group PG |

结果文件：

```text
CPU: /home/graphdb/FilterVectorResultsRefactor/sift30_zipf_origstyle_current_cpu_20260519_133655/results/build_time.csv
GPU: /home/graphdb/FilterVectorResultsRefactor/sift30_gpu_default_heavy_sgemm_20260519_135430/results/build_time.csv
```

`index_time` 阶段拆解：

| 阶段 | CPU baseline ms | CPU 占比 | GPU cross-edge ms | GPU 占比 |
| --- | ---: | ---: | ---: | ---: |
| `label_processing` | `626.10` | `0.98%` | `616.57` | `2.62%` |
| `build_graph` | `11,322.50` | `17.66%` | `10,001.60` | `42.46%` |
| `build_vector_attr_graph` | `232.53` | `0.36%` | `193.91` | `0.82%` |
| `build_LNG` | `1,732.00` | `2.70%` | `1,716.85` | `7.29%` |
| `descendants` | `41.74` | `0.07%` | `84.38` | `0.36%` |
| `coverage` | `5,259.98` | `8.21%` | `5,656.04` | `24.01%` |
| `cross_edges` | `44,560.00` | `69.51%` | `5,002.32` | `21.24%` |
| `unaccounted residual` | `329.75` | `0.51%` | `283.43` | `1.20%` |
| `index_time` | `64,104.60` | `100%` | `23,555.10` | `100%` |

说明：`unaccounted residual = index_time - build_time.csv 已显式记录阶段之和`。它不是写盘时间，也不是单独算法阶段，而是 `index.build(...)` 内未插桩的 glue code、容器初始化/收尾和阶段间调度开销。`save()` 写盘在 `Index time` 打印之后单独发生，不计入 `index_time`。

结论：

```text
当前最稳定、最可信的加速来自 cross-edge 精确 topK 的 GPU 化。
端到端加速没有 cross-edge 加速高，因为 group 内 PG / coverage / LNG 等阶段仍占时间。
```

注意：这次 SIFT30 A/B 的 coverage 日志是 `coverage impl: legacy_topological_merge`，没有设置 `UNG_COVERAGE_IMPL=1`。因此 `coverage=5656.04 ms` 反映的是 legacy coverage 在该口径下的剩余开销，不代表我们后续加入的 `descendants_direct` 最快路径。

### 0.3 历史稳定优化：descendants / coverage

来源：`docs/reports/OPTIMIZATION_STEP_BY_STEP.md`

| 指标 | 优化前 | 优化后 | 变化 |
| --- | ---: | ---: | ---: |
| `index_time` | `84115.0 ms` | `35314.5 ms` | `-58.0%` |
| `descendants` | `3059.7 ms` | `104.55 ms` | `-96.6%` |
| `coverage` | `29665.95 ms` | `219.1 ms` | `-99.3%` |

结论：

```text
这是 CPU 侧最明确的一轮优化，收益来自把 hash set 式 descendants/coverage 构造换成连续 vector + 批量 Roaring 写入。
```

其中 `descendants_direct` 是我们引入的 coverage 优化路径，不是原始默认实现。它直接使用已经算好的 `_lng_descendants` 拼出 coverage：

```text
coverage(group) = group 自身 vectors + 所有 descendant groups 的 vectors
```

运行时需要显式打开：

```bash
export UNG_COVERAGE_IMPL=1
```

### 0.4 Trie / LNG Phase1 优化

来源：历史 A/B 记录。

| 指标 | 优化前 | 优化后 | 变化 |
| --- | ---: | ---: | ---: |
| `index_time` | `49100.5 ms` | `36969.5 ms` | `-24.7%` |
| `LNG` | `45746.45 ms` | `33878.15 ms` | `-25.9%` |

结论：

```text
这部分优化有效，但 LNG 语义敏感，不能用破坏 edge 语义的 aggressive shortcut。
之前 exact BFS 快路径会把 LNG edges 从 946138 降到 20007，判定为错误路径。
```

### 0.5 Tagore GPU 建图：中大 group 实验

结果目录：

```text
/home/graphdb/FilterVectorResultsRefactor/tagore_build_bench_20260519_172638
```

实验范围：

```text
SIFT30 中 nx >= 1024 的 group
groups = 119
n_total = 465622
```

整体结果：

| 指标 | 数值 |
| --- | ---: |
| `wall_total` | `3.434 s` |
| `sum_group_total` | `3.246 s` |
| `mean` | `27.3 ms/group` |
| `p50` | `13.97 ms/group` |
| `max` | `427.15 ms` |

按 group size 分桶：

| `nx` 区间 | group 数 | 平均建图时间 |
| --- | ---: | ---: |
| `1024-2047` | `63` | `13.94 ms` |
| `2048-4095` | `32` | `18.73 ms` |
| `4096-8191` | `15` | `35.62 ms` |
| `8192-16383` | `6` | `73.46 ms` |
| `>=16384` | `3` | `264.45 ms` |

结论：

```text
Tagore 对中大 group 的 GPU 建图有潜力。
但这个实验不是完整替代 CPU Vamana 的等价 A/B：只测了 nx>=1024，且图结构/剪枝语义不保证和 UNG CPU Vamana 完全一致。
```

### 0.6 Tagore 图 + GPU ANN batch query

benchmark 源码：

```text
/home/graphdb/FilterVectorResultsRefactor/tagore_build_bench_20260519_172638/tagore_index_gpu_query_bench.cu
```

日志：

```text
/home/graphdb/FilterVectorResultsRefactor/tagore_build_bench_20260519_172638/tagore_index_gpu_query_10x1k_4096q.log
/home/graphdb/FilterVectorResultsRefactor/tagore_build_bench_20260519_172638/tagore_index_gpu_query_10x1k_32768q.log
```

实验设定：

```text
10 个 nx≈1k 的 group
exact baseline = 精确 GPU topK
ANN = 复用 Tagore 建好的 graph 后做 GPU graph search
```

关键结果：

| query/group | 配置 | `recall@6` | exact e2e | ANN graph reuse e2e | ANN build+search |
| ---: | --- | ---: | ---: | ---: | ---: |
| `4096` | 高 recall 配置 | `0.958` | `16.55 ms` | `13.33 ms` | `302.60 ms` |
| `32768` | `degree64 entries256 etop8` | `0.958` | `115.49 ms` | `84.55 ms` | `373.82 ms` |

结论：

```text
当 graph 可复用且 query 很多时，ANN search 有机会比 exact topK 快。
如果 build 时间必须计入单次请求，当前还不划算。
```

### 0.8 数据分布实验

已看过的代表性分布：

| 数据集 | 结论 |
| --- | --- |
| `sift_hiermedium` | `nq≈nx≈250`，更适合测中等均衡 GPU batch 负载 |
| SIFT30 Zipf | `nx p50=1`，`nq p50=55`，`nq/nx p50=36`，很多 tiny/singleton group |
| SIFT12 Zipf | `nx p50=10`，`nq p50=406`，`nq/nx p50=32.85` |
| 新 `sift_powerzipf30` | `num_groups=145017`，`singleton_groups=92521`，`p50=1`，`p90=7`，`p95=14`，`p99=80`，`max=26664` |

新 power-Zipf 文件：

```text
/home/graphdb/aaaGPU/FilterVectorData/sift/sift_base_30_labels_powerzipf_a0.85_c0.75_p0.9_seed20260519.txt
/home/graphdb/aaaGPU/FilterVectorData/sift/sift_base_30_labels_powerzipf_a0.85_c0.75_p0.9_seed20260519.manifest.json
```

结论：

```text
独立 Bernoulli / Zipf 式标签采样很容易制造大量唯一 label-set，从而产生过多 singleton group。
如果目标是研究 GPU 中等 group batch 负载，应该显式控制 label-set template / group size 分布，而不是只调 Zipf 陡峭程度。
```

## 1. 项目当前在优化什么

这个项目实现的是带标签过滤的近邻检索，核心构建流程是：

```text
base vectors + base labels
  -> 按完整 label set 分 group
  -> 每个 group 内建 proximity graph
  -> 对 group label set 建 LNG
  -> 计算 descendants / coverage
  -> 构建 cross-group edges
  -> 保存 UNG index
```

当前主要性能问题分为三类：

```text
1. LNG / descendants / coverage 的组合结构开销
2. group 内 PG 构建，原来主要是 CPU Vamana
3. cross-group topK，形状是很多 (nq, nx, dim) -> topK
```

其中 `nq/nx` 的定义是：

```text
nx = target group 的点数
nq = 指向该 target group 的 in-neighbor groups 的点数总和
work = nq * nx
```

这个指标本质上描述 cross-edge 构建中的 GPU 矩阵/ANN 查询负载。

## 2. 已经生效的稳定优化

### 2.1 descendants / coverage 容器从 hash set 改为连续 vector

历史上最重要的稳定优化是把 `_lng_descendants` / `covered_sets` 从哈希集合式写法迁移到连续向量布局。

核心收益：

```text
减少 unordered_set 插入热点
减少小对象分配
提升 cache locality
Roaring 初始化改为 addMany 批量写入
```

实测历史结果见 `docs/reports/OPTIMIZATION_STEP_BY_STEP.md`：

```text
Index time: 84115.0 -> 35314.5 ms (-58.0%)
descendants: 3059.7 -> 104.55 ms (-96.6%)
coverage: 29665.95 -> 219.1 ms (-99.3%)
```

生效条件：

```text
descendants / coverage 总量大
哈希插入和去重成为瓶颈
输出语义允许最终 sort/unique 或顺序 append 后批量写入
```

挑战：

```text
如果 LNG 结构产生重复覆盖，legacy 路径仍需 sort+unique 保证语义
vector 布局降低构建成本，但 coverage 总量过大时仍会占内存
```

### 2.2 Trie / LNG Phase1 的低风险优化

已做优化包括：

```text
get_min_super_sets 使用线程本地容器复用
候选 label-set size 跨度小时走 bucket 路径
trie 插入去掉冗余 resize
get_super_set_entrances 从 queue 改 vector+head
去重结构改进
```

历史 A/B：

```text
Index time: 49100.5 -> 36969.5 ms (-24.7%)
LNG: 45746.45 -> 33878.15 ms (-25.9%)
```

生效条件：

```text
group 数很多
label-set 包含关系复杂
get_min_super_sets 是主瓶颈
```

挑战：

```text
LNG 语义非常敏感，不能用过度激进的 exact-node 快路径
曾经的 exact BFS 快路径会把 LNG edges 从 946138 降到 20007，语义错误
```

## 3. GPU cross-edge 精确 topK 优化

### 3.1 当前 GPU cross-edge 在做什么

cross-edge 的核心任务是对每个 target group 做：

```text
Q: 来自 in-neighbor groups 的点，数量 nq
X: target group 内点，数量 nx
计算每个 q 在 X 中的 topK
```

这里的 LNG 是 label-set 包含关系 DAG，不是严格树。一个 child group 可以有多个 parent groups。cross-edge 会对每条 `parent_group -> child_group` 分别展开：父组点作为 query `q`，孩子组点作为候选 `x`，在每个孩子组内单独取 topK 并加边 `q -> topK(X)`。它不是把所有孩子合并后取一个全局 topK。

当前主要实现位于：

```text
UNG/codes/src/gpu_gemm_topk.cu
```

现有策略包括：

```text
小 group / 中 group fused topK kernel
较大 group 走 SGEMM / grouped tile + topK update
支持 bucket / batched descriptors
对 singleton 有专门 fastpath
```

### 3.2 SIFT30 上的 CPU/GPU cross-edge 加速

SIFT30 原始 Zipf 风格数据：

```text
CPU build_cross_edges_time: 44560 ms
GPU build_cross_edges_time: 5002 ms
```

对应结果：

```text
CPU: /home/graphdb/FilterVectorResultsRefactor/sift30_zipf_origstyle_current_cpu_20260519_133655/results/build_time.csv
GPU: /home/graphdb/FilterVectorResultsRefactor/sift30_gpu_default_heavy_sgemm_20260519_135430/results/build_time.csv
```

粗略加速：

```text
cross-edge 约 8.9x
```

端到端：

```text
CPU index_time: 64104.6 ms
GPU index_time: 23555.1 ms
端到端约 2.7x
```

阶段拆解：

| 阶段 | CPU ms | GPU ms | 说明 |
| --- | ---: | ---: | --- |
| `build_graph` | `11322.50` | `10001.60` | 仍是 CPU group 内 PG/Vamana |
| `coverage` | `5259.98` | `5656.04` | 未被 cross-edge GPU 化覆盖 |
| `build_LNG` | `1732.00` | `1716.85` | 基本同量级 |
| `cross_edges` | `44560.00` | `5002.32` | 主要加速来源，`8.91x` |
| `unaccounted residual` | `329.75` | `283.43` | `index_time` 减去显式阶段合计 |
| `index_time` | `64104.60` | `23555.10` | 端到端 `2.72x` |

GPU cross-edge 内部日志：

```text
[GPU GEMM] H2D(ms)=67.9  Kernel(ms)=2294.8  D2H(ms)=46.4
build_cross_edges_time = 5002.32 ms
```

解释：GPU 显式 kernel/H2D/D2H 约 `2409.1 ms`，cross-edge 总时间还有约 `2593.2 ms` 是 host 侧准备、分组调度、结果回写、additional edges、add offset 和 merge 等 CPU 侧开销。

生效条件：

```text
cross-edge 是主瓶颈
存在足够多 target groups 或足够大的 nq*nx
GPU 可以批量处理多个 group，摊薄 launch 开销
```

不生效或收益差的条件：

```text
大量 nx=1 或极小 group，算力利用率低
单个 group 太小且逐 group launch
数据分布被极少数超大 group 主导，尾部任务拖慢
```

### 3.3 groupGEMM / TF32 / WMMA 的结论

我们测试过多种精确 topK 路径：

```text
SGEMM + topK update
TF32 batched / WMMA 变体
small / medium fused kernels
```

经验结论：

```text
cuBLAS/cuVS brute force 对足够大矩阵很强
自研 fused kernel 对小中 group 有机会减少 launch 和中间矩阵开销
TF32/WMMA 并不自动赢，必须匹配足够大的 tile 和合适的数据分布
```

典型挑战：

```text
64x64 或 128x128 这种单 group 工作量太小
每个 group 单独 launch 不划算
fusion/batching 比单个 kernel 的微优化更重要
```

## 4. group 内 PG 构建：CPU Vamana、Tagore

### 4.1 当前 CPU Vamana 的角色

UNG 传统 group 内图使用 CPU Vamana。它既是构建方法，也是查询时可导航的 group 内图结构。

在 SIFT30 上：

```text
CPU Vamana build_graph_time 约 10-11.3 s
```

例子：

```text
/home/graphdb/FilterVectorResultsRefactor/sift30_gpu_default_heavy_sgemm_20260519_135430/results/build_time.csv
build_graph_time = 10001.6 ms
```

### 4.2 Tagore GPU 建图结果

Tagore 是 GPU 建图库，不是 CPU Vamana。它的流程是：

```text
GPU GNN-Descent 生成 kNN 候选图
GPU pruning 生成 Vamana/NSG/CAGRA/DPG 风格图
```

对 SIFT30 中 `nx >= 1024` 的 119 个 group 测得：

```text
groups = 119
n_total = 465622
wall_total = 3.434 s
sum_group_total = 3.246 s
mean = 27.3 ms/group
p50 = 14.0 ms/group
max = 427.1 ms
```

按规模：

```text
1024-2047: 63 groups, mean 13.94 ms
2048-4095: 32 groups, mean 18.73 ms
4096-8191: 15 groups, mean 35.62 ms
8192-16383: 6 groups, mean 73.46 ms
>=16384: 3 groups, mean 264.45 ms
```

结果目录：

```text
/home/graphdb/FilterVectorResultsRefactor/tagore_build_bench_20260519_172638
```

和当前 CPU Vamana 粗略比较：

```text
Tagore GPU 中大 group: 3.43 s
UNG CPU Vamana 全量: 10-11.3 s
显示约 3x 级别潜力
```

注意：这个比较不是完全等价，因为 Tagore 只测了 `nx>=1024`，且输出图不保证和 CPU Vamana 边完全一致。

### 4.3 Tagore 的 block 粒度并行能力

Tagore 当前核心 kernel 基本是：

```text
1 block / point
```

例如：

```cpp
dim3 grid(points_num, 1, 1);
initialize_graph<<<points_num, 32>>>(...)
nn_descent_opt_cal<<<grid, block2>>>(...)
select_path<<<grid_s, block_s>>>(...)
```

每个 block 负责一个点的邻居池：

```text
初始化随机邻居
从邻居的邻居采样候选
计算候选距离
merge topK 邻居
pruning 保留 FINAL_DEGREE
```

生效条件：

```text
nx 至少几百到上千，block 数足够多
每点候选池大小固定，适合 shared memory + warp reduction / WMMA
```

挑战：

```text
原始 Python API 逐 group 串行，不能自然让多个 group 并发占 SM
有文件读写和 index 落盘
没有 stream 参数
大量 cudaMalloc/cudaMemcpy/cudaDeviceSynchronize
```

建议路线：

```text
第一步：改 C++ in-memory single-group API + workspace 复用 + cudaStream_t
第二步：多 stream 并发多个中小 group
第三步：真正 batched kernel，把 block 映射到 (group_id, local_point_id)
```

## 5. GPU ANN batch query 的实验结论

### 5.1 问题设定

我们测试了：

```text
Tagore 建好的 group graph
GPU batch graph ANN query
和 exact GPU topK 对比
```

测试数据：

```text
10 个 nx≈1028-1108 的 group
每组 nq=4096 或 32768
dim=128
topK=6
```

benchmark 文件：

```text
/home/graphdb/FilterVectorResultsRefactor/tagore_build_bench_20260519_172638/tagore_index_gpu_query_bench.cu
```

### 5.2 4096 query/group

高 recall 配置：

```text
degree64, entries256, entry_top8
recall@6 = 0.958
exact e2e = 16.55 ms
ANN reuse graph e2e = 13.33 ms
ANN build+search = 302.60 ms
reuse speedup = 1.24x
```

中等 recall 配置：

```text
degree32, entries256, entry_top4
recall@6 = 0.875
exact e2e = 16.80 ms
ANN reuse graph e2e = 11.06 ms
reuse speedup = 1.52x
```

### 5.3 32768 query/group

高 recall 配置：

```text
degree64, entries256, entry_top8
recall@6 = 0.958
exact e2e = 115.49 ms
ANN reuse graph e2e = 84.55 ms
ANN build+search = 373.82 ms
reuse speedup = 1.37x
```

中等 recall 配置：

```text
degree32, entries256, entry_top4
recall@6 = 0.875
exact e2e = 119.27 ms
ANN reuse graph e2e = 69.94 ms
reuse speedup = 1.71x
```

结论：

```text
如果图只建一次、query 很多，ANN 有潜力
如果每个新 group 只查一次，build+search 当前不如 exact topK
```

生效条件：

```text
graph 可复用
nq 足够大，能摊薄建图开销
目标 recall 不要求接近 1.0
nx 至少几百到几千，图搜索才有意义
```

挑战：

```text
当前 query kernel 只是原型，候选收集策略简单
高 recall 下 ANN 搜索成本上升明显
Tagore 图输出需要 in-memory 接入，否则文件 I/O 会吞掉收益
```

## 6. 数据分布与 nq/nx 结论

### 6.1 SIFT 12-label Zipf

结果目录：

```text
/home/graphdb/FilterVectorResultsRefactor/sift_readme_zipf12_current_20260519_115616
```

统计：

```text
groups = 2386
target_groups = 2373
nx mean = 312.5, p50 = 10, p95 = 947.8, max = 77840
nq mean = 4075.6, p50 = 406, p95 = 18641.8, max = 177610
nq/nx mean = 38.97, p50 = 32.85, p95 = 85.5, max = 387
```

特点：

```text
nq 通常远大于 nx
少数超大 group 支配 work
```

### 6.2 SIFT 30-label orig-style Zipf

结果目录：

```text
/home/graphdb/FilterVectorResultsRefactor/sift30_zipf_origstyle_current_gpu_20260519_133252
```

统计：

```text
groups = 69177
target_groups = 69146
nx mean = 12.59, p50 = 1, p95 = 23, max = 39528
nq mean = 345.67, p50 = 55, p95 = 1075, max = 80649
nq/nx mean = 44.72, p50 = 36, p95 = 112.2, max = 554
```

特点：

```text
大量 singleton/tiny group
中位数 nx=1，不适合验证中大 group ANN
但 cross-edge GPU batching 仍有显著收益
```

### 6.3 自构造 hiermedium

结果目录：

```text
/home/graphdb/FilterVectorResultsRefactor/sift_hiermedium_tf32_batched_full_20260519_155829
```

统计：

```text
groups = 4001
target_groups = 3000
nx mean = 250.71, p50 = 251, p95 = 332, max = 420
nq mean = 247.87, p50 = 249, p95 = 331, max = 420
nq/nx mean = 1.03, p50 = 0.99, p95 = 1.64, max = 2.50
```

特点：

```text
非常均衡，适合测 small/medium exact topK kernel
不适合证明 nq 很大时 ANN graph query 的优势
```

### 6.4 新 power-Zipf 标签数据集

新生成数据：

```text
/home/graphdb/aaaGPU/FilterVectorData/sift/sift_base_30_labels_powerzipf_a0.85_c0.75_p0.9_seed20260519.txt
```

生成公式：

```text
P(label i) = min(0.9, 0.75 / i^0.85)
```

静态 label-set 统计：

```text
num_groups = 145017
singleton_groups = 92521
group_size p50 = 1
p90 = 7
p95 = 14
p99 = 80
max = 26664
top1_share = 2.67%
top10_share = 10.03%
```

它达成了：

```text
组合更多：69176 -> 145017 groups
头部更小：max 66978 -> 26664
```

但问题是：

```text
singleton group 仍然太多，p50=1
如果目标是中等 group ANN，这个版本还不理想
```

### 6.5 Amazon root 扇出数据集

详细报告见：

```text
docs/reports/CROSS_GROUP_CELEBA_RESULTS_CN.md
```

结果与统计文件：

```text
/home/graphdb/FilterVectorResultsRefactor/amazon_m32_lb100_cpu/others/ung_build_cpu.log
/home/graphdb/FilterVectorResultsRefactor/amazon_m32_lb100_cpu/analysis/amazon_nq_stats_by_nx_bin.csv
/home/graphdb/FilterVectorResultsRefactor/amazon_m32_lb100_cpu/analysis/amazon_cross_edge_nq_nx_by_group.csv
/home/graphdb/FilterVectorResultsRefactor/amazon_m32_lb100_cpu/analysis/amazon_depth_vector_share.csv
/home/graphdb/FilterVectorResultsRefactor/amazon_m32_lb100_cpu/analysis/amazon_depth_outdegree_stats.csv
```

静态结构：

```text
groups = 482388
LNG edges = 4022490
roots = 1
leaves = 377039
root group = 44
root nx = 20336
root out_degree = 29341
```

逐层结构摘要：

```text
max depth = 153
depth 0: vector_pct = 3.38%, avg_out_degree = 29341.00
depth 2: avg_out_degree = 105.06
depth 3: avg_out_degree = 84.82
depth 4: avg_out_degree = 37.19, sum_out_degree = 1192397
depth 7: vector_pct = 10.62%, avg_out_degree = 5.84
depth >= 13: avg_out_degree 基本低于 1，逐渐进入叶子层
```

完整 CPU 构建：

```text
cross-edge CPU = 694401.1 ms
explicit measured build stages = 934031.3 ms
unaccounted residual = 375.7 ms
Index time = 934407 ms
wall time = 16:08.27
peak RSS = 47610484 KB
```

说明：Amazon 的 `load data = 3021 ms` 发生在 `Index time` 计时之前，不计入 `index_time`；`Index saved in 29564.6 ms` 发生在 `Index time` 打印之后，也不计入 `index_time`。这里的 residual 只表示 `index.build(...)` 内未单独插桩的少量 glue code 和阶段间开销。

按 `nx` 分桶的关键结论：

| nx bin | group_count | group_pct | avg_nq | sum_nq_pct | work_pct |
| --- | ---: | ---: | ---: | ---: | ---: |
| `1` | `458017` | `94.95%` | `1326.78` | `98.27%` | `89.77%` |
| `2-3` | `20411` | `4.23%` | `451.32` | `1.49%` | `2.86%` |
| `512-1023` | `17` | `0.004%` | `1595.06` | `0.00%` | `3.55%` |

结论：

```text
Amazon 的主耗时不是少量大 nx group，而是 root 大组扇出造成的海量 nx=1/tiny target group。
root group 44 的 20336 个向量扇出到 29341 个目标组，导致 nq=20336 的目标组贡献 96.49% 的 sum(nq)。
优化重点应优先处理“大源组 -> 大量 singleton/tiny 目标组”的批处理/合并。
```

## 7. 当前主要挑战

### 7.1 数据集目标不一致

不同实验目标需要不同数据分布：

```text
测 LNG scalability：需要大量 groups 和复杂包含关系
测 exact GPU topK：需要可控的 nq/nx bucket
测 ANN graph query：需要 nx 几百到几千，且 nq 足够大
测 Tagore build：需要中大 group，避免 singleton 主导
```

一个数据集很难同时满足所有目标。

### 7.2 Zipf 调参容易产生大量 singleton

增加 labels、降低 Zipf 头部集中度、提高 label cardinality 会增加 label-set 组合数，但也容易让组合过度稀疏：

```text
num_groups 接近 N
大量 group size = 1
PG build 和 ANN search 失去意义
```

所以后续应固定报告：

```text
group_size p50/p90/p95/p99/max
singleton ratio
nq/nx p50/p95/p99
work top1/top10 share
```

### 7.3 Tagore 还不是可直接生产接入的库

当前 Tagore 的问题：

```text
Python API
文件输入输出
逐 group 默认流串行
per-call cudaMalloc/cudaMemcpy/synchronize
输出 index 文件而不是直接写 UNG graph
```

要用于主线，需要改成：

```text
C++ in-memory API
workspace pool
cudaStream_t
输出 fixed-degree graph 到 device/host buffer
小 group fallback exact/complete graph
```

### 7.4 ANN query 的 recall/performance tradeoff 未完全解决

当前原型显示：

```text
recall 0.87-0.90 时 reuse graph 可有 1.5-1.7x
recall 0.95 左右时约 1.2-1.4x
build+single-query-batch 不划算
```

后续需要：

```text
更好的 entry selection
真正 beam search / frontier 管理
warp/CTA 粒度 topM
避免动态分支和 visited 大结构
多 group batched query
```

## 8. 后续建议路线

### 8.1 短期

```text
1. 固定一套数据分布报告脚本，自动输出 group/nq/nx/work 指标
2. 对新 power-Zipf 跑一次完整 UNG build，确认 LNG 和 cross-edge 真实形状
3. 保留 hiermedium 作为均衡 small/medium exact topK benchmark
4. 保留 SIFT30 Zipf 作为长尾 stress benchmark
```

### 8.2 中期

```text
1. 做 Tagore C++ in-memory single-group wrapper
2. 去掉 Python/file I/O/per-call malloc
3. 对 nx>=1024 的 group 替换 CPU Vamana
4. 小 group 继续 complete graph 或 exact topK
5. 测 graph quality：recall、degree、连通性、最终 search 效果
```

### 8.3 长期

```text
1. Batched Tagore kernels：block 映射到 (group, local_point)
2. GPU resident group graph + batch ANN query
3. 根据 group size 和 query volume 自动选择 exact topK 或 ANN graph search
4. 目标不是所有场景替代 exact，而是在 graph 可复用且 nq 大时获得端到端收益
```

## 9. 当前判断

目前已经可靠生效的是：

```text
LNG/coverage 数据结构优化
GPU exact/grouped fused cross-edge topK
universal flat double-buffer cross-edge route 的 x200 full-quality 正结果
packed exact-anchor / FastGrnndCuda / reverse-tail 的 workload-aware group graph router
NeighborList64 / reserve / direct-H2D 等 CPU-compatible output-boundary 减压优化
```

仍处于研究原型的是：

```text
source-centric no-lock 的 two-stage grouped GEMM + per-source reduce
flat adjacency / CSR / GraphView 输出后端
additional_edges 的 GPU/flat full-quality backend
真实多标签和更多 x400 参数扫
GPU ANN batch query 替代 exact topK
```

最重要的工程判断：

```text
不要把所有 group 都交给同一种算法。
小 group 用 CPU/bounded fallback；中组可用 packed exact-anchor；大组用 FastGrnndCuda/reverse-tail 或更强 prune。
cross-edge 当前以 universal flat double-buffer 为主线；source-centric 只能作为后续 no-lock 设计方向。
只要最终 search 仍消费 CPU-compatible Graph::neighbors/SearchQueue/Vamana 语义，端到端收益就会受 output boundary 限制。
```
