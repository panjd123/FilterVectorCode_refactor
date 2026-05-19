# 当前优化状态、适用条件与挑战

本文总结 `FilterVectorCode_refactor` 当前围绕 UNG 构建阶段做过的主要优化、实测效果、生效条件、失败路径和后续挑战。重点覆盖最近围绕 SIFT1M/SIFT30、GPU cross-edge、Tagore GPU 建图、ANN batch query 和数据集构造的工作。

配套运行手册：

```text
RUNBOOK_OPTIMIZED_FEATURES_CN.md
```

阅读方式：

```text
本文回答：现在项目进展如何、做了哪些优化、什么条件下生效、还有哪些挑战。
运行手册回答：这些功能怎么编译、怎么跑、环境变量怎么配、结果文件怎么看。
```

## 0. 实测结果总览

这一节是目前做过的关键实验台账，避免只保留泛泛结论。更详细的运行命令见 `RUNBOOK_OPTIMIZED_FEATURES_CN.md`。

### 0.1 稳定主线：SIFT30 CPU/GPU cross-edge A/B

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

结论：

```text
当前最稳定、最可信的加速来自 cross-edge 精确 topK 的 GPU 化。
端到端加速没有 cross-edge 加速高，因为 group 内 PG / LNG / I/O 等阶段仍占时间。
```

### 0.2 历史稳定优化：descendants / coverage

来源：`OPTIMIZATION_STEP_BY_STEP.md`

| 指标 | 优化前 | 优化后 | 变化 |
| --- | ---: | ---: | ---: |
| `index_time` | `84115.0 ms` | `35314.5 ms` | `-58.0%` |
| `descendants` | `3059.7 ms` | `104.55 ms` | `-96.6%` |
| `coverage` | `29665.95 ms` | `219.1 ms` | `-99.3%` |

结论：

```text
这是 CPU 侧最明确的一轮优化，收益来自把 hash set 式 descendants/coverage 构造换成连续 vector + 批量 Roaring 写入。
```

### 0.3 Trie / LNG Phase1 优化

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

### 0.4 FixedPoolGPU：失败实验

结果目录：

```text
/home/graphdb/FilterVectorResultsRefactor/pg_build_compare_20260519_172008
```

| 方法 | `build_graph_time` | 结论 |
| --- | ---: | --- |
| CPU Vamana | `2773.15 ms` | baseline |
| FixedPoolGPU | `129447 ms` | 远慢于 CPU |

结论：

```text
naive 固定邻居池 GPU builder 没有解决 per-group malloc/copy/launch 和访存问题，当前不应作为主线。
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

### 0.7 数据分布实验

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

实测历史结果见 `OPTIMIZATION_STEP_BY_STEP.md`：

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

## 4. group 内 PG 构建：CPU Vamana、FixedPoolGPU、Tagore

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

### 4.2 FixedPoolGPU 实验失败

我们实现过一个 fixed neighbor pool GPU builder：

```text
随机固定邻居池
通过邻居的邻居多轮 refine
再 prune 成边
```

它的思想是 GPU 友好的定长数组，但当前实现不成熟。

实测在 hiermedium：

```text
CPU Vamana build_graph_time: 2773.15 ms
FixedPoolGPU build_graph_time: 129447 ms
```

结果目录：

```text
/home/graphdb/FilterVectorResultsRefactor/pg_build_compare_20260519_172008
```

失败原因：

```text
per-group cudaMalloc/cudaMemcpy/kernel/D2H 太重
OpenMP 线程独立发 GPU 工作，调度混乱
算法输出不等价于 CPU Vamana
大量小 group 时 GPU launch 开销支配
```

结论：

```text
FixedPoolGPU 当前不应作为主线
```

### 4.3 Tagore GPU 建图结果

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

### 4.4 Tagore 的 block 粒度并行能力

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
GPU exact cross-edge topK
部分 small/medium group batching/fusion
```

仍处于研究原型的是：

```text
GPU group PG build 替代 CPU Vamana
Tagore C++ 接入
GPU ANN batch query 替代 exact topK
新数据分布设计
```

最重要的工程判断：

```text
不要把所有 group 都交给同一种算法。
小 group 用 complete/exact；中大 group 用 GPU builder；query 足够多且 graph 可复用时才用 ANN graph search。
```
