# 多层 Special Block：方法、构建代价与同 Recall 查询性能

更新时间：2026-09-06

## TL;DR

我们在单层 Special Block 上新增了真正的第二层图 overlay：保留 `T1=1000` 的中层 block，并以 `T2=4k/10k/25k/50k` 构建更大尺度的上层 block。查询候选不再只有普通/自由两个状态，而是按 `普通图(0) -> 中层(1) -> 上层(2)` 单调激活；普通候选不能直接使用上层边。它的作用不是让一次相同 `Lsearch` 的扩展更便宜，而是提供跨越更大合法子树的导航边，以更小的 `Lsearch` 达到相同端到端 Recall。

在 Amazon 原始 100% x1、每档 1000 条真实 query、100 search threads、K=10 上，相对单层 `T1=1k`：

| 平均选择率 | 共同 Recall 门槛 | 单层最快实测点 | 多层最快实测点 | 1000-query batch 加速 |
|---:|---:|---|---|---:|
| 24.915% | >=0.90 | L20000, R=0.9057, 6520.180 ms | T2=25k, L16000, R=0.9062, 5335.580 ms | **1.222x** |
| 49.971% | >=0.85 | L2000, R=0.8640, 673.019 ms | T2=50k, L700, R=0.8661, 439.856 ms | **1.530x** |
| 74.994% | >=0.87 | L5000, R=0.8865, 2508.740 ms | T2=25k, L1600, R=0.8871, 1069.745 ms | **2.345x** |

在单层扫描可达的最高质量附近，多层分别为 `1.222x / 4.925x / 6.290x`。这证明第二层在宽过滤条件下能显著压低达到同质量所需的搜索预算；25% workload 的收益较小，是明确的适用边界。

这不是“全面优于所有 filtered-ANN 系统”的结果。共同 Recall 门槛下，FAVOR 在三档 workload 都更快；多层方法在 50% 档快于 NaviX、Curator 和已测 ACORN，在 75% 档快于 NaviX/Curator、略快于 ACORN-gamma12，但仍慢于 FAVOR。论文应把主要创新 claim 放在单层到多层的结构增益，并把跨系统结果作为诚实的系统位置对照。

## 1. 问题与创新点

### 1.1 单层结构的限制

单层 Special Block 把 label trie 上满足阈值的合法子树组织为 sidecar graph。查询完整覆盖某个 block 后，候选可以使用该 block 的 special edges。但是只有 `T1=1000` 一种尺度时，搜索可以在一个中等子树内快速移动，却缺少更大合法区域之间的长程导航。继续增大单层阈值会丢失原有中层结构，也不是多层。

### 1.2 我们的方法

构建器在同一 group trie 上执行两次独立的 bottom-up uncovered partition：

- 中层：`T1=1000`，保持原单层 block 的 ID 和成员不变；
- 上层：独立使用 `T2>T1` 构造更大 block；
- 同一 group/point 可以同时属于一个中层和一个上层 block，分别维护 ownership；
- 每个中层 block 记录最近的上层容器 `parent_block_id`；
- 普通图、中层 special graph、上层 special graph 作为三个 overlay 共存。

这里“独立分区”很重要：上层不是把中层 block 简单聚合后替换原图，而是在保留中层可达性的同时增加更长尺度的边。

查询候选携带 `activation_level in {0,1,2}`：

```text
level 0: 只能走普通边
   | 完整进入合法中层 block
   v
level 1: 可走普通边 + 中层 special edge
   | 到达合法上层 block 根/成员
   v
level 2: 可走普通边 + 中层 edge + 上层 edge
```

状态只升不降，且一条边最多把候选提升一级，所以 level 0 不能绕过中层直接使用 level 2。若同一个 point 通过更高层路径再次到达，候选队列原位升级它的 level 并允许重新扩展，但不重复占用 `Lsearch` 或 top-K 槽。这个去重/升级约束是正确性的必要条件。

## 2. Baseline 与公平比较

### 2.1 内部结构 baseline

| 方法 | 图结构 | 查询状态 | 用途 |
|---|---|---|---|
| Single-level | 普通图 + `T1=1k` overlay | 0 -> 1 | 直接回答第二层是否有价值，是核心 baseline |
| Multi-level | 普通图 + `T1=1k` + `T2` overlay | 0 -> 1 -> 2 | 我们的方法 |
| Upper-off | 与 Multi-level 使用同一个索引，但查询禁用 level 2 | 0 -> 1 | 排除“只是重新构图波动”的机制消融，不与 fresh single 的系统时间混用 |

内部比较固定同一主图、ELS provider、query/GT、K 和线程数。由于多层在固定 L 下通常做更多有效探索、Recall 更高，性能主张必须按相同 Recall 比较；固定 L 只用于说明机制。

### 2.2 外部系统 baseline

| 系统 | 本实验中的角色 | 调节参数 | 计时边界 |
|---|---|---|---|
| FAVOR | HNSW-based filtered search；用 exclusion mechanism 跳过不可能产生合法结果的区域，并动态选择搜索策略 | `Lsearch` | 1000-query batch total |
| NaviX | 在 UNG runner 中接入的 NaviX navigation route；过滤入口和邻接扩展由其既有图路径处理 | `Lsearch` | 1000-query batch total |
| Curator | clustering tree 上嵌入轻量 per-label sub-index；复杂谓词通过多个 label index 组合 | `search_ef` | 1000-query batch total |
| Official ACORN | predicate-agnostic graph，把谓词合法 bitmap 传给官方 ACORN search；扫描 gamma 与 efSearch | `gamma`, `efSearch` | 同时报 total 与 core ANN search |

ACORN adapter 发现官方 hybrid search 的初始 candidate 未同时检查 `filter_map`，会返回不合法点；修复后所有纳入结果的 filter violations 均为 0，Recall 不变。外部主表对每种方法仅选择达到预先声明 Recall 门槛的最快实测点，不插值、不外推。

## 3. 数据集与度量

| 项目 | 数值/口径 |
|---|---|
| 数据集 | Amazon 原始 100% x1，不使用 xN repeat/hybrid 数据 |
| 向量数 / 维度 | 602,453 / 768 |
| group 数 / label 数 | 482,387 / 30,723 |
| query | 每档 1000 条真实 query |
| 平均选择率 | 24.915%、49.971%、74.994% |
| 输出 | K=10 |
| 并行度 | 100 search threads |
| 质量 | 对同一 exact filtered GT 的端到端 Recall |
| 时间主统计 | 排除 cold repeat 后的 1000-query batch median；同时保存 mean/CV |

Block 数、覆盖率、入口数、边数和局部 top-K overlap 都只是诊断指标，不能代替最终 filtered-search Recall。

## 4. 构建代价

以下时间是在已经存在的 base UNG index 上构建 Special Block overlay 的时间，不是从原始向量开始的完整索引总时间。

这批正式索引的实际构建 dispatch（由 `build.log` 核对）为：block 内部 `n<=2048` 用 CPU exact-topK，`2048<n<8192` 用 CPU sampled-Vamana（256 candidates），`n>=8192` 用 `jasper_style`/FastGrnnd CUDA；block 间 200--220 个 parent-child pairs 全部因 pair-work 超过 10000 而走 CPU graph search（`ef=32`），没有启用 GPU-inter。因此下表是“hybrid intra + CPU inter”的公平 single/multi 结构实验，不应误标为 GPU/GPU 构建结果。

| 方法 | upper blocks | overlay build | 相对单层 | special edges | sidecar bytes |
|---|---:|---:|---:|---:|---:|
| Single T1=1k | 0 | 72.927 s | 1.000x | 40,395,111 | 539,055,889 |
| Multi T2=4k | 46 | 94.425 s | 1.295x | 72,733,732 | 799,669,854 |
| Multi T2=10k | 22 | 89.051 s | 1.221x | 71,753,752 | 791,828,031 |
| Multi T2=25k | 8 | 92.024 s | 1.262x | 69,343,169 | 772,497,914 |
| Multi T2=50k | 5 | 94.396 s | 1.294x | 66,543,978 | 749,982,894 |

第二层使 overlay build 增加约 22.1%--29.5%，sidecar 增加约 39.1%--48.3%。阈值更大时上层 block 更少、总边数和磁盘占用下降，但查询最优 T2 随 workload/质量变化，不能仅按最小索引选择。作为参照，base UNG 的历史 metadata 为 index build 211.819 s、index+additional 215.406 s；由于它与 overlay runner 的计时边界不同，不能把 `72--94 s` 直接与外部系统的 from-scratch build time 做严格速度比。

外部系统的 from-scratch 构建记录如下。它们的索引语义、并行实现和文件边界不同，因此用于说明系统成本量级，不计算跨系统 build speedup。

| 系统/配置 | build 时间 | 索引/内存 | 备注 |
|---|---:|---:|---|
| Base UNG metadata | 211.819 s；含 additional 为 215.406 s | 未与本表统一重测 | Special overlay 尚未计入 |
| FAVOR | 76.174 s | 约 4.490 GB file / 4.494 GB memory | 完整 FAVOR index |
| Curator | 102.207 s | 2.046 GB disk / 2.180 GB memory | Amazon x1 隔离重建 |
| NaviX | 1316.957 s | graph 260.8 MB；vectors 1850.7 MB；labels 21.4 MB | 完整 NaviX build |
| ACORN-1, gamma=1 | 17.268 s core（20.44 s wall） | 2.130 GB | `M=32, M_beta=64, efConstruction=40` |
| ACORN gamma=2/4/8/12 | 53.413 / 80.696 / 135.577 / 212.311 s core | 约 2.06--2.08 GB | `M=32, M_beta=32`；efConstruction 随 gamma 为 64/128/256/384 |

## 5. 查询主结果

### 5.1 相对单层：共同 Recall 门槛

每行先固定一个双方可达的 Recall 门槛，再从各自实测点中选最短时间。

| 选择率 | 门槛 | Single T1=1k | Multi-level | L 缩减 | batch speedup |
|---:|---:|---|---|---:|---:|
| 24.915% | R>=0.90 | L20000, R=0.9057, 6520.180 ms | T2=25k, L16000, R=0.9062, 5335.580 ms | 20.0% | **1.222x** |
| 49.971% | R>=0.85 | L2000, R=0.8640, 673.019 ms | T2=50k, L700, R=0.8661, 439.856 ms | 65.0% | **1.530x** |
| 74.994% | R>=0.87 | L5000, R=0.8865, 2508.740 ms | T2=25k, L1600, R=0.8871, 1069.745 ms | 68.0% | **2.345x** |

对应 warm mean/CV 分别为：单层 `6511.293/0.007、674.929/0.011、2510.332/0.006 ms`；多层 `5336.662/0.009、441.970/0.016、1068.780/0.008 ms`。主表中的差异远大于稳态波动。

### 5.2 单层最高质量附近

| 选择率 | Single 最高质量点 | 达到不低于该 Recall 的最快 Multi | batch speedup |
|---:|---|---|---:|
| 24.915% | L20000, R=0.9057, 6520.180 ms | T2=25k, L16000, R=0.9062, 5335.580 ms | **1.222x** |
| 49.971% | L20000, R=0.9506, 11518.550 ms | T2=50k, L5500, R=0.9511, 2338.770 ms | **4.925x** |
| 74.994% | L20000, R=0.9301, 16614.450 ms | T2=50k, L4500, R=0.9311, 2641.455 ms | **6.290x** |

这组结果直接展示了多层结构在高选择率、高质量区域的价值：其收益主要来自把所需 L 降到单层的 22.5%--27.5%，而不是同 L kernel 更快。

### 5.3 外部系统位置

时间均为 1000-query batch median。`ACORN core` 仅是 ANN search；跨系统排序使用包含 filter lookup/materialization 的 `total`。不同系统的 Recall 高于门槛幅度不同，因此此表是“达到门槛的最快实测点”，不是完全相同 Recall 的连续插值。

| 选择率 / 门槛 | Multi-level | FAVOR | NaviX | Curator | Official ACORN |
|---|---|---|---|---|---|
| 24.915% / R>=0.90 | T2=25k L16000, R=.9062, **5335.580 ms** | L5000, R=.9133, **1754.680 ms** | L2000, R=.9153, **2413.740 ms** | ef10240, R=.9744, **1838.554 ms** | ACORN-1 ef16384, R=.9011, **8207.948 ms total / 7943.332 ms core** |
| 49.971% / R>=0.85 | T2=50k L700, R=.8661, **439.856 ms** | L100, R=.8644, **200.929 ms** | L100, R=.8695, **2612.825 ms** | ef10240, R=.9742, **1590.117 ms** | ACORN-1 ef8192, R=.8531, **3106.804 ms total / 2864.419 ms core** |
| 74.994% / R>=0.87 | T2=25k L1600, R=.8871, **1069.745 ms** | L200, R=.8920, **265.841 ms** | L200, R=.9179, **2065.955 ms** | ef10240, R=.9629, **1942.862 ms** | gamma12 ef2048, R=.8710, **1148.216 ms total / 843.998 ms core** |

解释边界：

- 多层相对单层的因果性最强，因为两者共享主图、ELS 和搜索框架。
- FAVOR 是当前三档均领先的系统 baseline，说明多层 overlay 还没有成为整体最优系统。
- 50%/75% 下，多层比 NaviX/Curator 更快；25% 下则更慢。
- 75% 下多层 total 比 ACORN-gamma12 快约 `1.073x`，但 ACORN core search 更快；差距来自 ACORN 的 filter lookup/materialization 边界，必须同时呈现。
- Curator 在此网格的最优点 Recall 明显高于门槛，不能据此声称其在精确同 Recall 下必然更慢；需要更密的低预算点才是严格连续 Pareto 比较。

## 6. 为什么宽查询收益更大

完整覆盖上层 block 的 query 比例随选择率增加：25%/50%/75% 分别约为 26.0%/51.3%/77.3%。只有完整覆盖时，上层边才可被授权。因此：

1. 25% 查询中，大多数 query 不能使用上层或只能短暂使用，新增边的维护/扫描成本更难摊薄；
2. 50% 和 75% 查询更常完整覆盖大子树，上层边能跨越中层局部结构；
3. Recall-L 曲线明显左移，最终以更小 L 达到相同质量。

同一多层索引的 upper-on/off 消融显示，在所有已测 workload/L 上启用 level 2 都提高 Recall；详细计数也观测到上层激活、节点扩展和边扫描。因此收益来自真实的上层路径，而不是单纯的构建随机波动。

## 7. 正确性、局限与论文 claim

可以写入论文的结论：

- 提出层级化 Special Block overlay，在保留中层图的同时增加更大尺度合法子树导航；
- 通过逐级 activation、per-point 去重和原位 level upgrade 保证普通 -> 中层 -> 上层语义；
- 在 Amazon x1 的 50%/75% 高选择率 workload 中，相对单层显著减少同 Recall 所需 L，并在最高质量附近达到 `4.925x/6.290x` batch speedup；
- 代价是 overlay 构建增加约 22%--30%，sidecar 增加约 39%--48%。

不能写的结论：

- 不能把固定 L 下的耗时变化写成加速；
- 不能把 block coverage、边 overlap 或入口数写成准确率；
- 不能把 overlay build time 写成完整 from-scratch index build time；
- 不能声称当前方法全面优于 FAVOR/Curator/NaviX/ACORN；
- 不能把 21,834-label hybrid 主图上的旧结果与当前 30,723-label Amazon x1 结果混用。

当前实现还有一个明确性能边界：GPU free-distance batch scratch 尚未携带 per-edge activation level，因此多层模式暂时走标量路径，避免把 level 2 错误降为 level 1。未来若把 level metadata 纳入 batch scratch，可以继续优化查询 kernel，但必须重新验证 Recall。

## 8. 复现与证据

- 实现分支：`codex/multilevel-special-block-20260905`，隔离 shared clone：`/home/sunyahui/worktrees/FilterVectorCode_multilevel_special`。
- 内部正式结果：`runs/final_candidates_sel25_amazon_x1/summary/`、`runs/final_candidates_sel50_amazon_x1/summary/`、`runs/final_candidates_sel75_amazon_x1/summary/`。
- 外部 baseline 汇总由 `experiments/external_baselines/`、`experiments/curator_baseline/` 和 `experiments/acorn_baseline/` 的 runner 生成；大索引和 raw runs 不提交。
- focused C++ tests：4/4；多层 Python tests：13/13；Curator tests：12/12；ACORN 所有纳入结果为 0 filter violations。
- 当前主表只使用实测点；所有 Recall 由同一 query/GT 计算，不使用局部 top-K overlap 替代。
