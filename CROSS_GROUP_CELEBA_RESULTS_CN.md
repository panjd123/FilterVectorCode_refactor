# CelebA 与跨组边实验结果整理

本文整理两类结果：

1. `celeba` 从最初 CPU 版本到优化版本的构建耗时变化，以及每个阶段采用的优化办法。
2. 跨组边单独实验的对比口径：CPU/ANN 图、GPU 建图+搜索、NVIDIA exact brute force、以及我们自己的 `groupGEMM + fused topK` kernel。

> 说明：本文只写入当前仓库和本机结果目录中能追溯到原始日志/CSV 的数据。`100 个 nx=256, nq=10*nx` 和 `100 个 nx=256, nq=4*nx` 专项结果已在本轮补测，并在第 5 节给出脚本、输出目录和实测表格。

---

## 1. CelebA 原始 CPU 版本基线

### 1.1 数据集与运行口径

数据来源：旧 CPU 原始版本在 CelebA 上重新运行的完整构建结果。

原始结果文件：

```text
/home/graphdb/FilterVectorCode_old_rerun_results/celeba_c2_20260519_111420/results/build_time.csv
/home/graphdb/FilterVectorCode_old_rerun_results/celeba_c2_20260519_111420/others/ung_build.log
```

数据规模：

| 项 | 数值 |
| --- | ---: |
| 数据集 | `celeba` |
| base vectors | `202,599` |
| 维度 | `512` |
| labels | `40` |
| groups | `115,114` |
| vector-attribute edges | `1,830,201` |
| average descendants/group | `275.4` |

CelebA 组分布特征（本机统计）：

| 指标 | 数值 |
| --- | ---: |
| 每向量标签数均值 | `9.03` |
| 每向量标签数中位数 | `9` |
| 每向量标签数 p95 | `14` |
| 组大小中位数 | `1` |
| 单点组占比 | `78.0%` |
| 最大组大小 | `358` |

结论：CelebA 是典型的“大量 tiny group + 少量中大 group”分布，所以 CPU 侧主要瓶颈不是组内 PG，而是 LNG、coverage、descendants 和 cross-group edges。

### 1.2 常见问题：单点组也建立跨组边

初始 CelebA 数据集里有一个非常典型的问题：**大量 group 只有 1 个点，但 cross-group edges 仍然会按 group 逐个处理**。

按本机统计：

| 指标 | 数值 |
| --- | ---: |
| 总 group 数 | `115,114` |
| 单点组占比 | `78.0%` |
| 单点组数量 | `89,826` |
| 单点组点数占比 | `44.34%` |
| 最大组大小 | `358` |

更细的 group size 分布：

| group size | group 数 | group 占比 | 点数 | 点数占比 |
| --- | ---: | ---: | ---: | ---: |
| `nx=1` | `89,826` | `78.03%` | `89,826` | `44.34%` |
| `2<=nx<=8` | `23,077` | `20.05%` | `68,499` | `33.81%` |
| `9<=nx<=64` | `2,144` | `1.86%` | `37,219` | `18.37%` |
| `65<=nx<=255` | `66` | `0.06%` | `6,697` | `3.31%` |
| `nx>=256` | `1` | `0.001%` | `358` | `0.18%` |

这会带来几个问题：

1. **计算量小但任务数极多**  
   单点组的 `nx=1`，真正距离计算很少；但它仍然需要进入 cross-edge 流程，包括候选 query 收集、数据 gather、topK 初始化/更新、结果回写、图合并等固定开销。

2. **GPU 路径容易被 launch / 调度 / H2D-D2H 开销支配**  
   对 `nx=1` 的 group，单个 group 几乎无法吃满 GPU。若逐 group 启动 kernel，即使每个 tiny task 只有几十微秒固定开销，`≈9 万` 个单点组也会累计到秒级甚至十几秒级。

3. **CPU 路径也会被容器与图合并开销支配**  
   在 CPU 版本中，单点组会造成大量小 vector / set / heap / adjacency 更新，距离计算本身反而不是主要成本。

4. **跨组边语义收益低**  
   单点组没有有意义的组内 PG；为每个单点组都建立精确跨组边，更多是在维护导航连通性，而不是提升组内近邻质量。

以原始 CelebA CPU 结果为例：

| 阶段 | 耗时 | 占总构建 |
| --- | ---: | ---: |
| `build_cross_edges_time` | `30139 ms` | `35.4%` |
| `build_LNG_time` | `28241.5 ms` | `33.2%` |
| `index_time` | `85028.9 ms` | `100.0%` |

这里 cross-edge 已经是最大单项耗时。结合 `78%` 单点组分布，可以判断：初始数据集的 cross-edge 慢，很大一部分不是来自大矩阵距离计算，而是来自“**海量 tiny/singleton group 的固定开销**”。

为了量化这个问题，本轮用**原始 CPU 版本**做了 `nx` 过滤实验。由于旧版原本没有 `UNG_BENCH_MIN_NX/MAX_NX` 开关，本次只在旧源码中临时加入 target group size 过滤和日志，不改 Vamana 搜索、additional edges、merge 等核心逻辑；测试后已恢复旧源码。

输出目录：

```text
/home/graphdb/FilterVectorCode_old_rerun_results/celeba_singleton_cpu_20260521_153552
```

实验通过临时 `UNG_BENCH_MIN_NX/MAX_NX` 控制 target group size：

| 过滤条件 | 含义 | active target groups | active queries | `index_time` | `build_cross_edges_time` | wall time |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `UNG_BENCH_MIN_NX=1`, `UNG_BENCH_MAX_NX=1` | 只处理单点 target group | `89,788` | `1,297,147` | `80424.2 ms` | `10863.4 ms` | `96.72 s` |
| `UNG_BENCH_MIN_NX=2`, `UNG_BENCH_MAX_NX=1048576` | 只处理非单点 target group | `25,285` | `793,541` | `69490.1 ms` | `3964.05 ms` | `86.63 s` |
| `UNG_BENCH_MIN_NX=0`, `UNG_BENCH_MAX_NX=1048576` | 同一插桩二进制下全量 target group | `115,073` | `2,090,688` | `80617.8 ms` | `14498.7 ms` | `99.86 s` |
| 历史未插桩原始 CPU 全量 | 完整原始 rerun 结果 | - | - | `85028.9 ms` | `30139 ms` | - |

读数说明：

- 在原始 CPU 版本中，`nx=1` 单点 target groups 单独耗时 `10863.4 ms`，也就是 **10.86 秒**。
- 同一插桩二进制下，单点组占 cross-edge 时间约 `10863.4 / 14498.7 = 74.9%`；按 `nx=1` 与 `nx>=2` 两个拆分实验求和，单点组占 `10863.4 / (10863.4 + 3964.05) = 73.3%`。
- 单点组只有 `44.34%` 的点数，却贡献了约 `73%~75%` 的 CPU cross-edge 拆分耗时，说明旧版 CPU 的瓶颈主要不是距离计算量，而是大量 singleton group 带来的 per-group 固定开销、搜索缓存调度、容器插入和图合并。
- 历史未插桩原始 CPU 全量 `build_cross_edges_time=30139 ms` 高于本次插桩 all run 的 `14498.7 ms`，说明旧 CPU 路径运行间波动/代码状态有差异；但在同一插桩二进制的 `nx=1 / nx>=2 / all` 对比中，singleton 占比非常明确。若按 `73%~75%` 的拆分比例外推到历史全量 `30139 ms`，单点组对应约 `22.1~22.6 s` 的 cross-edge 成本。

可以用一个很粗的线性模型解释 singleton 固定开销：

```text
cross_edge_time ≈ a * active_target_groups + b * active_queries
```

用本次原始 CPU 插桩结果的 `nx=1` 和 `nx>=2` 两个点反推：

| 参数 | 估算值 | 含义 |
| --- | ---: | --- |
| `a` | `0.0905 ms/group` | 每个 target group 的固定开销 |
| `b` | `0.00211 ms/query` | 每个 in-neighbor query 的搜索/插入开销 |
| `a / b` | `≈42.8 queries/group` | query 成本超过 group 固定开销的临界点 |

对 singleton 来说：

| 项 | 数值 |
| --- | ---: |
| active singleton target groups | `89,788` |
| active queries | `1,297,147` |
| 平均 `nq/group` | `14.45` |
| 估算固定开销 `a*groups` | `8122.7 ms` |
| 估算 query 开销 `b*queries` | `2740.7 ms` |
| 固定开销占 singleton 时间 | `≈74.8%` |

因此，单点组慢的核心不是 `nx=1` 的距离计算，而是 `≈9 万` 个 target group 每个都要付一次固定成本。

什么时候会进入“和点数/查询数相关”的增长范围：

1. **对 `nx=1` singleton**  
   当平均 `nq/group` 超过约 `43` 时，`b*nq` 才会超过 `a`，query/search 成本开始主导。CelebA singleton 当前平均 `nq/group≈14.45`，还明显处在固定开销主导区。

2. **对小 group：`2<=nx<=Lbuild`**  
   旧 CPU Vamana 查询在很小 group 上很容易访问到大部分局部图节点，因此单 query 成本会随 `nx` 增长。此时总成本近似从：

   ```text
   group 固定成本 + nq * small_search_cost(nx)
   ```

   过渡到和 `nq*nx` 更相关。

3. **对中大 group：`nx` 接近或超过 `Lbuild=100`**  
   单 query 不一定线性扫完整 group，而是受 `Lbuild`、graph degree、搜索收敛轮数限制；增长更接近 `nq * visited_nodes`，其中 `visited_nodes` 通常被 `Lbuild` 上界约束。

4. **对 additional edges 阶段**  
   旧 CPU 代码还会为没有连通的 out-neighbor group 补边，这部分会扫描当前 group 的前若干点，最多到 `_num_cross_edges`，因此对 `nx=1` 是固定小成本，对 `nx>=num_cross_edges` 后更接近按 missing out-neighbor 数增长。

经验判断：

| 区间 | 主导因素 | 现象 |
| --- | --- | --- |
| `nx=1` 且 `nq/group < ~43` | per-group 固定开销 | singleton 数越多越慢 |
| tiny group 且 `nq` 小 | 固定开销 + 容器开销 | GPU/CPU 都容易调度不划算 |
| `nq/group` 很大 | query/search 开销 | 开始随 in-neighbor 点数增长 |
| `nx` 增大到几十/上百 | Vamana search / additional edges | 开始随局部图规模和访问节点数增长 |
| `nx>=Lbuild` | `Lbuild`/degree 限制的 graph search | 不再严格线性随 `nx` 增长 |

对极端数据集的影响会更明显：

| 极端分布 | 影响 |
| --- | --- |
| `N` 个点几乎各自形成唯一 label-set | group 数接近 `N`，单点组接近 `100%`，组内 PG 基本失去意义 |
| 每个单点组仍走完整 cross-edge | 固定开销从 `O(groups)` 放大到接近 `O(N)` 个 tiny task |
| GPU exact topK 逐 group 处理 | 大量 kernel launch / gather / 回写开销吞掉 GPU 并行收益 |
| LNG 包含关系复杂 | 单点 target group 的 `nq` 仍可能很大，形成“计算不大但调度很碎”的长尾 |

因此后续优化需要把 singleton/tiny group 当作单独路径处理，而不是和中大 group 共用同一套 cross-edge pipeline。可选策略包括：

- **tiny group CPU fast path**：`nx=1` 或很小 `nx` 直接在 CPU 批量处理，避免 GPU launch。
- **singleton 批处理**：把多个 `nx=1` target group 合并成一个 batched descriptor，一次 kernel 处理多个 group。
- **按 work 而不是只按 `nx` 分桶**：用 `work = nq * nx` 判断是否值得走 GPU。
- **跳过低收益边或延迟构建**：对只影响连通性、对 recall 贡献小的 singleton 边，可考虑惰性构建或搜索时补偿。
- **heavy/tiny 分流**：大 `nq*nx` 走 SGEMM，tiny/small 走 fused/batched 或 CPU fast path。

### 1.3 原始 CPU 版本分阶段耗时

| 阶段 | `build_time.csv` 字段 | 耗时 | 占 `index_time` 比例 |
| --- | --- | ---: | ---: |
| 总构建 | `index_time` | `85028.9 ms` | `100.0%` |
| 标签处理、分组、trie 准备 | `label_processing_time` | `557.93 ms` | `0.7%` |
| 组内 PG 构建 | `build_graph_time` | `169.522 ms` | `0.2%` |
| vector-attribute bipartite graph | `build_vector_attr_graph_time` | `105.21 ms` | `0.1%` |
| LNG 构建 | `build_LNG_time` | `28241.5 ms` | `33.2%` |
| descendants 计算 | `cal_descendants_time` | `3273.92 ms` | `3.9%` |
| coverage ratio 计算 | `cal_coverage_ratio_time` | `11075.4 ms` | `13.0%` |
| cross-group edges | `build_cross_edges_time` | `30139 ms` | `35.4%` |

注意：旧 CSV 里的 `cross_edge_step1_time` 等子字段是未初始化垃圾值，不能用于分析；只看 `build_cross_edges_time`。

---

## 2. CelebA/CPU 主线优化过程

### 2.1 优化总览

| 优化阶段 | 主要目标 | 优化前 | 优化后 | 变化 | 证据来源 |
| --- | --- | ---: | ---: | ---: | --- |
| 原始 CPU 基线 | 建立阶段耗时画像 | `85028.9 ms` | - | - | old rerun CelebA |
| descendants / coverage 容器重构 | 降低集合构造和 Roaring 写入开销 | `84115.0 ms` | `35314.5 ms` | `-58.0%` | `OPTIMIZATION_STEP_BY_STEP.md` |
| Trie / LNG Phase1 优化 | 压缩超集枚举和容器分配开销 | `49100.5 ms` | `36969.5 ms` | `-24.7%` | `CURRENT_OPTIMIZATION_STATUS_CN.md` |
| cross-edge GPU 化（稳定主线，SIFT30 A/B） | 将精确 topK 距离计算搬到 GPU | `64104.6 ms` | `23555.1 ms` | `2.7x` 端到端 | SIFT30 CPU/GPU A/B |

> 说明：前三项可用于解释 CelebA CPU 路径的优化逻辑；第四项是当前最稳定的 GPU cross-edge 证据，但数据集是 `sift30_zipf_origstyle`，不能和 CelebA 原始 CPU 基线直接横向相除。

SIFT30 CPU/GPU A/B 的 `index_time` 拆解如下。这里的 `unaccounted residual` 是 `index_time` 减去 `build_time.csv` 已显式记录阶段之和，不包含 `index.save(...)` 写盘。

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

结论：`8.9x` 是 `cross_edges` 单项加速，端到端 `index_time` 是 `2.72x`。GPU 版里瓶颈从 cross-edge 转移到 CPU group 内 PG (`build_graph`) 和 coverage。

### 2.2 descendants / coverage：哈希集合改连续布局

结果：

| 指标 | 优化前 | 优化后 | 变化 |
| --- | ---: | ---: | ---: |
| `index_time` | `84115.0 ms` | `35314.5 ms` | `-58.0%` |
| `descendants` | `3059.7 ms` | `104.55 ms` | `-96.6%` |
| `coverage` | `29665.95 ms` | `219.1 ms` | `-99.3%` |
| `cross-group edges` | `1540.2 ms` | `1413.95 ms` | `-8.2%` |

采用的方法：

- `_lng_descendants` 与 `covered_sets` 从 `unordered_set` 改为连续 `vector` 布局。
- descendants 构建直接写 BFS 发现序列，避免大量 hash insert。
- coverage 的 descendants-direct 路径改为连续 append。
- legacy 路径保留 `sort + unique`，保证 DAG 合并语义。
- Roaring 初始化改成并行 + `addMany` 批量写入。

对应代码范围：

```text
UNG/codes/include/label_nav_graph.h
UNG/codes/src/uni_nav_graph.cpp
```

### 2.3 Trie / LNG Phase1：复用容器并降低候选枚举开销

结果：

| 指标 | 优化前 | 优化后 | 变化 |
| --- | ---: | ---: | ---: |
| `index_time` | `49100.5 ms` | `36969.5 ms` | `-24.7%` |
| `LNG` | `45746.45 ms` | `33878.15 ms` | `-25.9%` |

采用的方法：

- `build_label_nav_graph` Phase1 使用线程本地可复用 `min_super_set_ids`，避免每个 group 重复分配。
- 结果写回时使用 `swap`，减少拷贝。
- trie 插入去掉冗余 `resize`。
- superset 候选遍历从零散容器改成 `vector + head` 队列式遍历。
- 去重逻辑集中到 `unordered_set`，但避免在热路径频繁构造大对象。

注意：LNG 语义敏感，之前尝试过 aggressive shortcut / exact BFS 快路径，会把 LNG edges 从 `946138` 降到 `20007`，属于错误路径，不能作为优化结果使用。

### 2.4 cross-edge：GPU exact topK 主线

稳定 A/B 数据集：`sift30_zipf_origstyle`。

| 指标 | CPU baseline | GPU cross-edge | 加速/变化 |
| --- | ---: | ---: | ---: |
| `build_cross_edges_time` | `44560 ms` | `5002 ms` | `8.9x` |
| `index_time` | `64104.6 ms` | `23555.1 ms` | `2.7x` |
| `build_graph_time` | CPU Vamana | `10001.6 ms` | 仍是 CPU group PG |

结果文件：

```text
CPU: /home/graphdb/FilterVectorResultsRefactor/sift30_zipf_origstyle_current_cpu_20260519_133655/results/build_time.csv
GPU: /home/graphdb/FilterVectorResultsRefactor/sift30_gpu_default_heavy_sgemm_20260519_135430/results/build_time.csv
```

采用的方法：

- 将跨组边精确 topK 的距离计算从 CPU 搬到 GPU。
- group 批次化：按 target group 组织 `Q x X`，其中 `X` 是目标组内点，`Q` 是指向该 target group 的 in-neighbor groups 的点。
- 预先 flatten/gather query，减少每组重复准备开销。
- 对足够大的矩阵走 cuBLAS / cuBLASLt / SGEMM 路径。
- 对小中 group 引入 fused kernel，减少中间 dot 矩阵写出和单独 topK kernel launch。
- 重负载分流：`UNG_NAIVE_HEAVY_NX=256`、`UNG_NAIVE_HEAVY_NQ=256`、`UNG_NAIVE_HEAVY_SGEMM=1` 时，大任务优先交给 SGEMM。

---

## 3. 跨组边单独实验：已有实测结果

### 3.1 Tagore GPU 建图：中大 group

结果目录：

```text
/home/graphdb/FilterVectorResultsRefactor/tagore_build_bench_20260519_172638
```

实验范围：SIFT30 中 `nx >= 1024` 的 group。

| 指标 | 数值 |
| --- | ---: |
| groups | `119` |
| n_total | `465,622` |
| wall_total | `3.434 s` |
| sum_group_total | `3.246 s` |
| mean | `27.3 ms/group` |
| p50 | `13.97 ms/group` |
| max | `427.15 ms` |

按 group size 分桶：

| `nx` 区间 | group 数 | 平均建图时间 |
| --- | ---: | ---: |
| `1024-2047` | `63` | `13.94 ms` |
| `2048-4095` | `32` | `18.73 ms` |
| `4096-8191` | `15` | `35.62 ms` |
| `8192-16383` | `6` | `73.46 ms` |
| `>=16384` | `3` | `264.45 ms` |

结论：Tagore 对中大 group 的 GPU 建图有潜力；但它不是 CPU Vamana 的严格等价 A/B，因为只测了 `nx>=1024`，且图构造/剪枝语义不保证和 UNG CPU Vamana 完全一致。

### 3.2 Tagore 图 + GPU ANN batch query vs exact GPU topK

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

| 项 | 数值 |
| --- | ---: |
| group 数 | `10` |
| group size | `nx≈1028-1108` |
| dim | `128` |
| topK | `6` |
| exact baseline | GPU exact topK |
| ANN | 复用 Tagore 建好的 graph 后做 GPU graph search |

关键结果：

| query/group | 配置 | `recall@6` | exact e2e | ANN graph reuse e2e | ANN build+search | reuse speedup |
| ---: | --- | ---: | ---: | ---: | ---: | ---: |
| `4096` | degree64, entries256, entry_top8 | `0.958` | `16.55 ms` | `13.33 ms` | `302.60 ms` | `1.24x` |
| `32768` | degree64, entries256, entry_top8 | `0.958` | `115.49 ms` | `84.55 ms` | `373.82 ms` | `1.37x` |

补充中等 recall 配置：

| query/group | 配置 | `recall@6` | exact e2e | ANN graph reuse e2e | reuse speedup |
| ---: | --- | ---: | ---: | ---: | ---: |
| `4096` | degree32, entries256, entry_top4 | `0.875` | `16.80 ms` | `11.06 ms` | `1.52x` |
| `32768` | degree32, entries256, entry_top4 | `0.875` | `119.27 ms` | `69.94 ms` | `1.71x` |

结论：当 graph 可复用且 query 很多时，ANN graph search 有机会比 exact topK 快；如果 build 时间必须计入单次请求，当前仍不划算。

---

## 4. 我们的 `groupGEMM + fused topK` kernel 优化过程

代码位置：

```text
UNG/codes/src/gpu_gemm_topk.cu
```

### 4.1 优化路径

| 阶段 | 方法 | 解决的问题 | 适用场景 |
| --- | --- | --- | --- |
| CPU exact topK | CPU 逐 group 算距离并维护 topK | 原始正确基线 | 小规模/无 GPU |
| GPU cuBLAS exact | 每个 group/tile 做 GEMM，随后单独 topK update | 利用 Tensor Core/SGEMM 吞吐 | 大矩阵 `nq*nx` 足够大 |
| grouped tile GEMM | 多个 tile/group 批处理，减少 launch 和 descriptor 开销 | 长尾 group 过多导致 launch 开销大 | 中等 group / 多 group |
| small/medium fused topK | dot + L2 距离 + topK 在一个 kernel 内完成 | 避免写出中间 dot 矩阵，减少 topK kernel | 小中 group |
| large group fused | 对 `nx>=256` 的 group 做 fused topK | 面向 `nx=256~4096` 的跨组边热点 | 中大 group |
| heavy SGEMM fallback | 大任务分流回 SGEMM | 自研 fused kernel 不一定打得过 cuBLAS/cuVS brute force | 大 `nq`、大 `nx` |

### 4.2 关键 kernel / 路径

| 名称 | 作用 |
| --- | --- |
| `ung_dot_batched_naive_global_kernel` | 自研 CUDA-core dot kernel |
| `update_topk_from_dot_tile_kernel` | 单 tile dot 结果更新 running topK |
| `update_topk_from_dot_grouped_tiles_kernel` | grouped tiles 更新 topK，减少 launch |
| `ung_small_group_topk_fused_global_kernel` | 小 group fused topK |
| `ung_medium_group_topk_fused_global_kernel` | 中 group fused topK |
| `ung_group_desc_topk_fused_global_kernel` | bucket/descriptor 后按 group 描述符批处理 |
| `large_group_fused_mode=2` | TF32 Tensor Core 16x16 WMMA fused topK 路径 |

### 4.3 SIFT30 上按 `nx` 分桶的 fused kernel 实测

结果目录：

```text
/home/graphdb/FilterVectorResultsRefactor/fused_topk_sift30_20260519_1530
```

共同口径：

```text
UNG_CROSS_EDGE_BACKEND=1
UNG_LARGE_GROUP_FUSED=1
UNG_LARGE_GROUP_WARPS=8
数据集：sift30_zipf_origstyle
base vectors：SIFT1M，dim=128
num_cross_edges=6
```

结果：

| 实验目录 | `nx` 范围 | fused groups | active queries | `build_cross_edges_time` | 备注 |
| --- | ---: | ---: | ---: | ---: | --- |
| `nx256_511_w8` | `256-511` | `208` | `2,064,856` | `1515.07 ms` | large-group fused 成功 |
| `nx512_1023_w8` | `512-1023` | `103` | `1,620,871` | `2244.91 ms` | large-group fused 成功 |
| `nx1024_4096_w8` | `1024-4096` | `81` | `2,982,891` | `2398.04 ms` | large-group fused 成功 |
| `verify_nx256_511` | `256-511` | `208` | `2,064,856` | `50609.7 ms` | 开启严格抽样校验，不能当性能值 |
| `debug_launch_nx256_511_v2` | `256-511` | - | `2,064,856` | `46583.4 ms` | 历史失败：symbol not found 后 fallback/失败路径 |

对 `nx=256-511` 桶的优化变化：

| 阶段 | 状态 | `build_cross_edges_time` |
| --- | --- | ---: |
| 历史 debug/fallback 路径 | large fused kernel launch 失败，日志含 `named symbol not found` | `46583.4 ms` |
| 正常 large-group fused | `208` groups / `2,064,856` queries 进入 fused path | `1515.07 ms` |
| 严格校验运行 | `UNG_GEMM_VERIFY_SAMPLES=2000`, `UNG_GEMM_VERIFY_STRICT=1` | `50609.7 ms` |

说明：`verify_nx256_511` 用于正确性抽样校验，校验逻辑会显著放大耗时，不能作为性能对比点。

### 4.4 推荐开关

```bash
export UNG_CROSS_EDGE_BACKEND=1
export UNG_CROSS_EDGE_GPU_STRICT=1
export UNG_COVERAGE_IMPL=1
export UNG_COVERAGE_THREADS=32
export UNG_SMALL_GROUP_FUSED=1
export UNG_MEDIUM_GROUP_FUSED=1
export UNG_BUCKET_GROUP_FUSED=1
export UNG_LARGE_GROUP_FUSED=1
export UNG_LARGE_GROUP_FUSED_MODE=2
export UNG_NAIVE_HEAVY_SGEMM=1
export UNG_NAIVE_HEAVY_NX=256
export UNG_NAIVE_HEAVY_NQ=256
```

查看日志：

```bash
rg -n "\[PROF\] cross_edges|heavy_sgemm|large_group_fused|topk" "$OUT/others/ung_build.log" "$OUT/others/ung_prof.log"
```

---

## 5. 用户指定的 `100 x nx=256` 跨组边实验

用户希望整理两组单独实验：

1. `100` 个 group，每组 `nx=256`，`nq=10*nx=2560`。
2. `100` 个 group，每组 `nx=256`，`nq=4*nx=1024`。

本轮已补测并整理为可复现脚本。输出目录：

```text
Exact / NVIDIA brute / our fused:
/home/graphdb/FilterVectorResultsRefactor/organized_benchmarks/nx256_exact_20260521_manual

Tagore Vamana build + GPU graph search:
/home/graphdb/FilterVectorResultsRefactor/organized_benchmarks/tagore_nx256_20260521_manual
```

可复现入口：

```bash
# 已有 CelebA / SIFT30 / Tagore / fused-topK 结果汇总
scripts/benchmarks/run_existing_summary.sh

# UNG 完整构建 cross-edge A/B：CPU / GPU / fused
scripts/benchmarks/run_ung_cross_edge_ab.sh cpu
scripts/benchmarks/run_ung_cross_edge_ab.sh gpu
scripts/benchmarks/run_ung_cross_edge_ab.sh fused

# 100 x nx=256 exact topK：NVIDIA SGEMM + topK vs our fused topK
scripts/benchmarks/run_nx256_exact_benchmark.sh

# 100 x nx=256 Tagore Vamana build + GPU graph query
scripts/benchmarks/run_tagore_nx256_benchmark.sh
```

### 5.1 实验口径

| 参数 | 值 |
| --- | ---: |
| groups | `100` |
| `nx` | `256` |
| `nq/group` | `1024` 和 `2560` |
| dim | `128` |
| topK | `6` |
| metric | L2 |
| GPU | `NVIDIA RTX A6000` |
| GPU lock | 本机未找到 `gpulock`，脚本记录 warning 后直接运行 |

### 5.2 Exact brute force / 我们的 fused topK

脚本：

```bash
scripts/benchmarks/run_nx256_exact_benchmark.sh
```

源码：

```text
tools/benchmarks/nx256_cross_edge_exact_benchmark.cu
```

结果 CSV：

```text
/home/graphdb/FilterVectorResultsRefactor/organized_benchmarks/nx256_exact_20260521_manual/summary.csv
```

结果：

| `nq/group` | 方法 | H2D | search/topK | reuse e2e | exact / 对齐 |
| ---: | --- | ---: | ---: | ---: | ---: |
| `1024` | NVIDIA `cuBLAS SGEMM + separate topK` | `12.0595 ms` | `4.6403 ms` | `16.6998 ms` | `1.000000` baseline |
| `1024` | 我们 `fused L2 + topK` | `12.0595 ms` | `6.5953 ms` | `18.6549 ms` | `0.999989` vs SGEMM |
| `1024` | 我们 `hybrid heavy -> SGEMM` | `12.0595 ms` | `4.6403 ms` | `16.6998 ms` | `1.000000` |
| `2560` | NVIDIA `cuBLAS SGEMM + separate topK` | `45.4684 ms` | `12.9544 ms` | `58.4228 ms` | `1.000000` baseline |
| `2560` | 我们 `fused L2 + topK` | `45.4684 ms` | `17.8034 ms` | `63.2718 ms` | `0.999996` vs SGEMM |
| `2560` | 我们 `hybrid heavy -> SGEMM` | `45.4684 ms` | `12.9544 ms` | `58.4228 ms` | `1.000000` |

结论：在 `nx=256` 且 `nq=1024/2560` 的固定 dense workload 上，cuBLAS SGEMM + topK 更快；我们的推荐 hybrid 路径应该把这类 heavy workload 分流到 SGEMM。单纯 fused L2 topK 的价值主要在更小/更碎 group，或避免中间矩阵写出占主导的场景。

### 5.3 Tagore Vamana build + GPU graph search

脚本：

```bash
scripts/benchmarks/run_tagore_nx256_benchmark.sh
```

相关源码：

```text
scripts/benchmarks/generate_and_build_tagore_nx256.py
tools/benchmarks/tagore_index_gpu_query_bench.cu
```

结果文件：

```text
/home/graphdb/FilterVectorResultsRefactor/organized_benchmarks/tagore_nx256_20260521_manual/tagore_nx256_build_times.csv
/home/graphdb/FilterVectorResultsRefactor/organized_benchmarks/tagore_nx256_20260521_manual/tagore_query_summary.csv
```

Tagore Vamana 建图结果：

| 指标 | 数值 |
| --- | ---: |
| groups | `100` |
| n_total | `25,600` |
| wall_total | `1.8997 s` |
| sum_group_total | `0.6215 s` |
| mean build | `6.2145 ms/group` |
| max build | `170.7027 ms` |

Tagore graph query 结果（列出代表配置）：

| `nq/group` | 配置 | build | exact e2e | ANN reuse e2e | ANN once e2e | recall@6 | reuse vs exact |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `1024` | entries64, entry_top4 | `621.45 ms` | `21.86 ms` | `23.25 ms` | `644.70 ms` | `0.9280` | `0.94x` |
| `1024` | entries128, entry_top4 | `621.45 ms` | `21.90 ms` | `25.67 ms` | `647.12 ms` | `0.9724` | `0.85x` |
| `1024` | entries256, entry_top8 | `621.45 ms` | `21.97 ms` | `34.87 ms` | `656.33 ms` | `1.0000` | `0.63x` |
| `2560` | entries64, entry_top4 | `621.45 ms` | `57.21 ms` | `60.85 ms` | `682.31 ms` | `0.9285` | `0.94x` |
| `2560` | entries128, entry_top4 | `621.45 ms` | `57.23 ms` | `63.81 ms` | `685.26 ms` | `0.9725` | `0.90x` |
| `2560` | entries256, entry_top8 | `621.45 ms` | `57.22 ms` | `89.37 ms` | `710.83 ms` | `1.0000` | `0.64x` |

结论：对 `nx=256` 这种小 group，Tagore Vamana 的 build 成本和 graph search 开销都不占优；即使复用 graph，ANN reuse e2e 也没有超过 exact brute force。它更适合之前 `nx>=1024` 的中大 group 或更高复用次数场景。

### 5.4 CPU Vamana / UNG 完整路径

可复现脚本：

```bash
scripts/benchmarks/run_ung_cross_edge_ab.sh cpu
scripts/benchmarks/run_ung_cross_edge_ab.sh gpu
scripts/benchmarks/run_ung_cross_edge_ab.sh fused
```

说明：这个脚本跑完整 UNG 构建，包含 CPU Vamana 组内 PG、LNG、coverage、cross-edge。它不是合成 `100 x nx=256` 的微基准，而是端到端主线 A/B，用来验证优化是否真正改善完整索引构建。

---

## 6. Amazon 案例：`nq/nx` 分布与 CPU cross-edge 慢因

本节补充 Amazon 数据集在当前 refactor 版本上的一次完整 CPU 构建和 GPU strict 尝试。它用于解释一种和 CelebA/SIFT30 不同、但对 cross-edge 很致命的结构：**一个超大 root group 扇出到大量 singleton target group**。

### 6.1 数据位置、运行口径与输出文件

原始用户给出的旧日志：

```text
/home/fengxiaoyao/FilterVector/FilterVectorResults/Amazon/Index/M32_LB100_alpha1.2_C6_EP16_AN602453_AM32_AMB64_AG80/others/ung_build.log
```

旧日志中记录的数据路径是 `/mnt/disk1/syh/ljk/FilterVector/...`，但当前机器上该路径不存在；本机可用数据在：

```text
/home/fengxiaoyao/FilterVector/FilterVectorData/Amazon/
```

关键数据文件：

```text
/home/fengxiaoyao/FilterVector/FilterVectorData/Amazon/Amazon_base.bin
/home/fengxiaoyao/FilterVector/FilterVectorData/Amazon/Amazon_base_labels.txt
/home/fengxiaoyao/FilterVector/FilterVectorData/Amazon/encoding_map.txt
/home/fengxiaoyao/FilterVector/FilterVectorData/Amazon/log
```

复现实验参数：

| 参数 | 值 |
| --- | ---: |
| dataset | `Amazon` |
| base vectors | `602,453` |
| dimension | `768` |
| labels | `30,723` |
| max_degree | `32` |
| Lbuild | `100` |
| alpha | `1.2` |
| num_cross_edges | `6` |
| num_threads | `16` |

结果目录：

```text
/home/graphdb/FilterVectorResultsRefactor/amazon_m32_lb100_cpu
/home/graphdb/FilterVectorResultsRefactor/amazon_m32_lb100_gpu
/home/graphdb/FilterVectorResultsRefactor/amazon_m32_lb100_gpu_sgemm
```

生成的分布统计表：

```text
/home/graphdb/FilterVectorResultsRefactor/amazon_m32_lb100_cpu/analysis/amazon_cross_edge_nq_nx_by_group.csv
/home/graphdb/FilterVectorResultsRefactor/amazon_m32_lb100_cpu/analysis/amazon_cross_edge_distribution_summary.csv
/home/graphdb/FilterVectorResultsRefactor/amazon_m32_lb100_cpu/analysis/amazon_nq_stats_by_nx_bin.csv
/home/graphdb/FilterVectorResultsRefactor/amazon_m32_lb100_cpu/analysis/amazon_nq_nx_2d_count.csv
/home/graphdb/FilterVectorResultsRefactor/amazon_m32_lb100_cpu/analysis/amazon_nq_nx_2d_work.csv
/home/graphdb/FilterVectorResultsRefactor/amazon_m32_lb100_cpu/analysis/amazon_top_work_groups.csv
```

### 6.2 CPU 构建耗时

当前 refactor CPU 后端完整构建日志：

```text
/home/graphdb/FilterVectorResultsRefactor/amazon_m32_lb100_cpu/others/ung_build_cpu.log
```

实测耗时：

| 阶段 | 耗时 |
| --- | ---: |
| load data | `3.021 s` |
| prepare group storages | `2.172 s` |
| build graph for each group | `3.737 s` |
| vector-attribute bipartite graph | `246.37 ms` |
| LNG | `26.548 s` |
| descendants | `37.0 ms` |
| coverage | `206.889 s` |
| cross-edge CPU | `694.401 s` |
| explicit measured build stages | `934.031 s` |
| unaccounted residual | `0.376 s` |
| Index time | `934.407 s` |
| `/usr/bin/time` wall time | `16:08.27` |
| peak RSS | `47,610,484 KB` |

说明：`load data` 发生在 `Index time` 计时之前；`index.save(...)` 写盘发生在 `Index time` 打印之后。Amazon 这次 `Index saved in 29.565 s`，不计入 `Index time`。`unaccounted residual` 只表示 `index.build(...)` 内部未单独插桩的少量 glue code 和阶段间开销。

与旧日志相比：

| 指标 | 旧日志 | 当前 refactor CPU |
| --- | ---: | ---: |
| cross-edge | `1,426.998 s` | `694.401 s` |
| Index time | `1,464.088 s` | `934.407 s` |

说明：当前 refactor 在部分阶段已经比旧日志快，但 Amazon 的 cross-edge 仍然是最大耗时项。

### 6.3 GPU strict 尝试结果

GPU 测试期间机器上的 RTX A6000 不是空闲状态，`nvidia-smi` 显示 GPU 利用率为 `100%`，且有外部进程 `./build/bin/Test` 占用 GPU。因此 GPU 性能不能作为稳定结论；但 strict 模式暴露了当前 GPU path 在 Amazon 形状上的失败点。

默认 GPU/custom naive kernel：

```text
/home/graphdb/FilterVectorResultsRefactor/amazon_m32_lb100_gpu/others/ung_build_gpu.log
```

失败信息：

```text
[cross_edges][GPU] failed: naive dot kernel launch failed in full-tile path.
Command terminated by signal 6
```

强制 cuBLAS SGEMM：

```text
UNG_FORCE_CUSTOM_KERNEL=0 UNG_GEMM_IMPL=1
```

日志：

```text
/home/graphdb/FilterVectorResultsRefactor/amazon_m32_lb100_gpu_sgemm/others/ung_build_gpu_sgemm.log
```

失败信息：

```text
[cross_edges][GPU] failed: cublasSgemm failed in tail-tile path.
Command terminated by signal 6
```

结论：这次没有得到有效 GPU 完整耗时。当前 GPU cross-edge 对 Amazon 这种 root 扇出 + 海量 singleton 的形状还需要修复 launch/tail-tile 路径，不能用 fallback 后的 CPU 结果冒充 GPU 成绩。

### 6.4 树/LNG 结构

Amazon 的 label group 结构更准确地说是 Trie/LNG DAG，而不是严格树。统计结果：

| 指标 | 值 |
| --- | ---: |
| groups | `482,388` |
| LNG edges | `4,022,490` |
| roots | `1` |
| leaves | `377,039` |
| avg out-degree | `8.34` |
| max out-degree | `29,341` |
| max in-degree | `755` |
| max label depth | `153` |

唯一 root：

| group_id | label_depth | nx | in_degree | out_degree |
| ---: | ---: | ---: | ---: | ---: |
| `44` | `0` | `20,336` | `0` | `29,341` |

高扇出节点：

| group_id | depth | nx | in_degree | out_degree |
| ---: | ---: | ---: | ---: | ---: |
| `44` | `0` | `20,336` | `0` | `29,341` |
| `427706` | `3` | `1` | `1` | `10,199` |
| `23479` | `3` | `2` | `1` | `9,916` |
| `156` | `2` | `961` | `1` | `8,371` |
| `1813` | `3` | `5` | `2` | `8,129` |

按 label depth 的主要分布：

| depth | groups | vectors | out_edges |
| ---: | ---: | ---: | ---: |
| `0` | `1` | `20,336` | `29,341` |
| `2` | `408` | `2,002` | `42,863` |
| `3` | `8,599` | `11,568` | `729,403` |
| `4` | `32,062` | `37,176` | `1,192,397` |
| `5` | `32,075` | `42,285` | `789,441` |
| `6` | `11,884` | `19,707` | `361,222` |
| `7` | `52,382` | `63,960` | `305,849` |
| `8` | `33,969` | `45,920` | `174,200` |
| `9` | `39,301` | `46,914` | `129,757` |
| `10` | `35,429` | `43,081` | `86,529` |

逐层点数占比和逐层平均出度的完整表：

```text
/home/graphdb/FilterVectorResultsRefactor/amazon_m32_lb100_cpu/analysis/amazon_depth_vector_share.csv
/home/graphdb/FilterVectorResultsRefactor/amazon_m32_lb100_cpu/analysis/amazon_depth_outdegree_stats.csv
```

按 depth 的点数占比：

| depth | group_count | vector_count | vector_pct |
| ---: | ---: | ---: | ---: |
| `0` | `1` | `20,336` | `3.38%` |
| `2` | `408` | `2,002` | `0.33%` |
| `3` | `8,599` | `11,568` | `1.92%` |
| `4` | `32,062` | `37,176` | `6.17%` |
| `5` | `32,075` | `42,285` | `7.02%` |
| `6` | `11,884` | `19,707` | `3.27%` |
| `7` | `52,382` | `63,960` | `10.62%` |
| `8` | `33,969` | `45,920` | `7.62%` |
| `9` | `39,301` | `46,914` | `7.79%` |
| `10` | `35,429` | `43,081` | `7.15%` |
| `11` | `33,789` | `40,514` | `6.72%` |
| `12` | `29,476` | `35,394` | `5.87%` |
| `13` | `27,176` | `31,667` | `5.26%` |
| `14` | `23,024` | `26,345` | `4.37%` |
| `15` | `19,967` | `22,758` | `3.78%` |
| `16-20` | `62,027` | `69,588` | `11.55%` |
| `21-30` | `37,627` | `40,031` | `6.64%` |
| `31-50` | `6,759` | `7,095` | `1.18%` |
| `51+` | `297` | `313` | `0.05%` |

按 depth 的平均每组出度：

| depth | group_count | sum_out_degree | avg_out_degree | max_out_degree |
| ---: | ---: | ---: | ---: | ---: |
| `0` | `1` | `29,341` | `29,341.00` | `29,341` |
| `2` | `408` | `42,863` | `105.06` | `8,371` |
| `3` | `8,599` | `729,403` | `84.82` | `10,199` |
| `4` | `32,062` | `1,192,397` | `37.19` | `6,272` |
| `5` | `32,075` | `789,441` | `24.61` | `3,504` |
| `6` | `11,884` | `361,222` | `30.40` | `2,170` |
| `7` | `52,382` | `305,849` | `5.84` | `1,127` |
| `8` | `33,969` | `174,200` | `5.13` | `887` |
| `9` | `39,301` | `129,757` | `3.30` | `492` |
| `10` | `35,429` | `86,529` | `2.44` | `367` |
| `11` | `33,789` | `61,205` | `1.81` | `290` |
| `12` | `29,476` | `39,751` | `1.35` | `139` |
| `13` | `27,176` | `26,611` | `0.98` | `144` |
| `14` | `23,024` | `17,971` | `0.78` | `131` |
| `15` | `19,967` | `12,579` | `0.63` | `60` |
| `16-20` | `62,027` | `21,559` | `0.35` | `161` |
| `21-30` | `37,627` | `1,973` | `0.05` | `29` |
| `31-50` | `6,759` | `36` | `0.005` | `3` |
| `51+` | `297` | `3` | `0.01` | `1` |

关键观察：root group `44` 自身有 `20,336` 个向量，并向 `29,341` 个目标 group 扇出。这会把 `nq=20,336` 传给大量目标组，是 Amazon CPU 慢的第一原因。

### 6.5 `nq/nx` 定义与总体分布

在 cross-edge 构建中，对每个目标组 `tgt`：

```text
nx(tgt) = target group 自己的向量数
nq(tgt) = sum(size(in_group)) for in_group in in_neighbors[tgt]
work(tgt) = nq(tgt) * nx(tgt)
```

也就是说，`nq` 不是 query 文件里的查询数，而是这个目标组需要处理的跨组源向量数。

总体分布：

| metric | mean | p50 | p90 | p95 | p99 | p99.9 | max |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| nx | `1.25` | `1` | `1` | `2` | `3` | `22` | `20,336` |
| in_degree | `8.34` | `4` | `19` | `29` | `64` | `152` | `755` |
| out_degree | `8.34` | `0` | `5` | `17` | `157` | `1,099` | `29,341` |
| nq | `1,281.92` | `12` | `148` | `20,336` | `20,336` | `20,336` | `20,336` |
| nq*nx | `1,403.25` | `13` | `197` | `20,336` | `20,336` | `40,672` | `19,542,896` |

`nx` 分桶：

| nx bin | group_count | group_pct |
| --- | ---: | ---: |
| `1` | `458,017` | `94.95%` |
| `2-3` | `20,411` | `4.23%` |
| `4-7` | `2,407` | `0.50%` |
| `8-15` | `865` | `0.18%` |
| `16-31` | `317` | `0.07%` |
| `>=1024` | `5` | `0.001%` |

`nq` 分桶：

| nq bin | group_count | group_pct | nq_sum_pct |
| --- | ---: | ---: | ---: |
| `1-15` | `280,365` | `58.12%` | `0.31%` |
| `16-255` | `157,259` | `32.60%` | `1.01%` |
| `512-1023` | `10,419` | `2.16%` | `1.56%` |
| `16384-32767` | `29,341` | `6.08%` | `96.49%` |

关键结论：`nq=20336` 的那 `29,341` 个目标组贡献了 `96.49%` 的 `sum(nq)`。这正是 root group `44` 的扇出。

### 6.6 按 `nx` 分桶统计 `nq`

下表回答“`nx=1` 的 `nq` 平均值是多少，`nx=2-3` 的 `nq` 平均值是多少”。

完整 CSV：

```text
/home/graphdb/FilterVectorResultsRefactor/amazon_m32_lb100_cpu/analysis/amazon_nq_stats_by_nx_bin.csv
```

| nx bin | group_count | group_pct | avg_nx | avg_nq | p50_nq | p90_nq | p95_nq | p99_nq | max_nq | sum_nq_pct | avg(nq*nx) | work_pct |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `1` | `458,017` | `94.95%` | `1.00` | `1,326.78` | `12` | `158` | `20,336` | `20,336` | `20,336` | `98.27%` | `1,326.78` | `89.77%` |
| `2-3` | `20,411` | `4.23%` | `2.12` | `451.32` | `12` | `70` | `345` | `20,336` | `20,336` | `1.49%` | `946.94` | `2.86%` |
| `4-7` | `2,407` | `0.50%` | `4.93` | `327.25` | `16` | `138` | `608` | `20,336` | `20,336` | `0.13%` | `1,581.88` | `0.56%` |
| `8-15` | `865` | `0.18%` | `10.38` | `437.82` | `22` | `329` | `962` | `20,336` | `20,336` | `0.06%` | `4,603.21` | `0.59%` |
| `16-31` | `317` | `0.07%` | `21.29` | `394.28` | `39` | `446` | `831` | `17,518` | `20,336` | `0.02%` | `8,950.56` | `0.42%` |
| `32-63` | `164` | `0.03%` | `44.41` | `278.31` | `58.5` | `585` | `933` | `1,094` | `20,336` | `0.01%` | `13,782.25` | `0.33%` |
| `64-127` | `108` | `0.02%` | `85.74` | `927.33` | `76.5` | `734` | `1,154` | `20,336` | `20,336` | `0.02%` | `66,693.41` | `1.06%` |
| `128-255` | `47` | `0.01%` | `178.79` | `180.53` | `74` | `493` | `788` | `1,031` | `1,033` | `0.00%` | `31,883.62` | `0.22%` |
| `256-511` | `30` | `0.006%` | `350.03` | `316.20` | `138` | `1,000` | `1,046` | `1,801` | `2,102` | `0.00%` | `117,798.47` | `0.52%` |
| `512-1023` | `17` | `0.004%` | `702.41` | `1,595.06` | `134` | `2,016` | `5,738` | `17,416` | `20,336` | `0.00%` | `1,412,428.29` | `3.55%` |
| `>=1024` | `5` | `0.001%` | `5,231.20` | `108.40` | `54` | `257` | `301` | `337` | `346` | `0.00%` | `152,255.80` | `0.11%` |

结论：Amazon 的主耗时不是少量大 `nx` group，而是海量 `nx=1` group。`nx=1` 占 `94.95%` 的 group，贡献 `98.27%` 的 `sum(nq)` 和 `89.77%` 的 `sum(nq*nx)`。也就是说优化方向应优先处理“大源组 -> 大量 singleton/tiny 目标组”的批处理/合并，而不是只优化少量中大 `nx` group。

### 6.7 Top work group

按 `nq * nx` 排名前 10：

| rank | group_id | nx | nq | in_degree | out_degree | nq*nx |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `1` | `156` | `961` | `20,336` | `1` | `8,371` | `19,542,896` |
| `2` | `7120` | `71` | `20,336` | `1` | `332` | `1,443,856` |
| `3` | `20155` | `71` | `20,336` | `1` | `134` | `1,443,856` |
| `4` | `1452` | `638` | `2,089` | `4` | `114` | `1,332,782` |
| `5` | `7406` | `65` | `20,336` | `1` | `95` | `1,321,840` |
| `6` | `19957` | `64` | `20,336` | `1` | `257` | `1,301,504` |
| `7` | `825` | `590` | `1,967` | `2` | `62` | `1,160,530` |
| `8` | `28363` | `55` | `20,336` | `1` | `23` | `1,118,480` |
| `9` | `918` | `382` | `2,102` | `3` | `142` | `802,964` |
| `10` | `767` | `1005` | `612` | `3` | `586` | `615,060` |

单个 top group 的 `nq*nx` 很大，但总量上 `nx=1` 桶仍然支配 CPU 工作量。因此 Amazon 的优化重点应分两层：

1. 对 `nx=1/tiny` 的目标组，避免逐 group、逐 query 的 CPU Vamana fixed-point 调用，考虑按共同父源组批处理。
2. 对少量 `nx>=512` 且 `nq` 较大的 heavy group，用 SGEMM/exact topK 或修复后的 GPU path 做批处理。

---

## 7. 结论

1. CelebA 原始 CPU 基线中，最大瓶颈是 cross-group edges (`35.4%`) 和 LNG (`33.2%`)；coverage (`13.0%`) 和 descendants (`3.9%`) 也明显。
2. CPU 侧最有效的优化是 descendants / coverage 容器重构：`index_time` 从 `84115.0 ms` 降到 `35314.5 ms`，descendants 降 `96.6%`，coverage 降 `99.3%`。
3. Trie / LNG Phase1 优化进一步让 LNG 降低约 `25.9%`，但 LNG 语义敏感，错误 shortcut 不能使用。
4. GPU cross-edge exact topK 是当前最稳定的 GPU 主线：SIFT30 A/B 中 cross-edge 从 `44560 ms` 降到 `5002 ms`，约 `8.9x`。
5. 我们的 fused topK kernel 在 SIFT30 `nx=256-511` 桶中把可比成功路径做到 `1515.07 ms`；历史 debug/fallback 路径约 `46583.4 ms`，严格校验路径约 `50609.7 ms`，后两者不能当最终性能。
6. `100 x nx=256, nq=4*nx/10*nx` 已补测：dense workload 下 NVIDIA/cuBLAS SGEMM + topK 分别为 `4.64 ms` 和 `12.95 ms` search-only；我们的 hybrid 路径应将此类 heavy workload 分流到 SGEMM。
7. `100 x nx=256` 的 Tagore Vamana 建图总和约 `621.45 ms`，graph reuse e2e 未超过 exact brute force，说明 GPU graph 路线更适合更大 `nx` 或更高复用次数。
8. Amazon 案例显示另一类主要瓶颈：`nx=1` 的 singleton 目标组占 `94.95%`，贡献 `98.27%` 的 `sum(nq)` 和 `89.77%` 的 `sum(nq*nx)`；root group `44` 的 `20,336` 个向量扇出到 `29,341` 个目标组，是 CPU cross-edge 慢的主因。
