# GPU 组内图构建相关工作与撞车风险分析

更新时间：2026-07-02

本文聚焦当前 GPU group graph / PG 构建路线，特别是：

```text
packed exact-anchor / direct-H2D exact-anchor
+ group-aware reciprocal / reverse-tail augmentation
```

目标是回答三个问题：

1. 相关 GPU 图构建方法有哪些。
2. 我们的方法是否和现有方法撞车。
3. 相同点、不同点和论文写法边界是什么。

## 1. 我们当前方法的准确表述

当前方法不是通用 GPU kNN graph builder，也不是完整复现 CPU Vamana RobustPrune。它服务于 UNG filtered vector search 的组内 PG 构建，输入已经按 label-set group 切分。

当前主线可拆成：

| 部分 | 当前做法 | 作用 |
|---|---|---|
| group route | small group CPU/bounded fallback；medium group packed exact-anchor；large group FastGrnnd/reverse-tail | 避免单一 GPU backend 在小组/中组/大组上同时失效 |
| packed exact-anchor | 对 medium group 做 exact in-group topK；输出固定 degree graph | 保证中组质量和速度 |
| direct-H2D / GPU lookup | 避免 host pack，GPU 侧查 point->group/local id | 降低 exact-anchor 外围开销 |
| anchor/tail output | nearest head + anchor/tail diversity | 避免纯 nearest-neighbor clique |
| group-aware reverse augmentation | 从 exact-anchor 输出构造同组 reverse candidates，用少量 reverse slots 替换尾部边 | 改善低 Lsearch 下的可达性和导航质量 |

当前最佳质量增强点：

```text
head12_anchor4_bidir_rev4
```

效果：

| 指标 | default direct-H2D | head12_anchor4_bidir_rev4 |
|---|---:|---:|
| group build | `3204.88 ms` | `1822.89 ms` |
| L20 recall | `0.808867` | `0.814533` |
| L500 recall | `0.845000` | `0.850000` |

CPU Vamana 对照：

| 指标 | CPU Vamana |
|---|---:|
| group build | `8511.72 ms` |
| L20 recall | `0.819000` |
| L500 recall | `0.849000` |

结论：该增强把 L20 gap 缩小超过一半，并把 L500 修到略高于 CPU，同时 group build 仍显著快于 CPU 和 default direct-H2D。

## 2. 相关工作总览

| 类别 | 代表工作 / 系统 | 相关点 | 与我们关系 |
|---|---|---|---|
| CPU proximity graph | HNSW、NSG、DiskANN/Vamana | 使用邻居选择、RNG/MRNG/RobustPrune、图多样性来提升导航 | 我们的质量目标一致，但没有复现完整 CPU pruning；我们只做 GPU exact-anchor 输出增强 |
| Filtered ANN graph | Filtered-DiskANN / StitchedVamana、UNG | 过滤条件下需要保证连通和候选覆盖；StitchedVamana 按 label/filter 构图再合并 | 我们是 UNG label-set group 内 PG 构建，不是为每个 filter 单独建完整 Vamana |
| GPU kNN graph construction | GNND / GPU NN-Descent、GNN-Descent | GPU 上并行构建近邻图初始候选 | 我们复用/借鉴 GPU batch 构图思想，但 medium groups 走 exact-anchor，不是近似 NN-Descent 初始化 |
| GPU graph index construction | Tagore | GPU 加速 NSG/Vamana-like graph indexing，GNN-Descent + prune | 我们代码里使用 Tagore/FastGrnnd 作为大组路线；本次 exact-anchor/reverse augmentation 是 UNG medium-group 的定制扩展 |
| GPU ANN graph search/index | CAGRA/cuVS | GPU-native graph，固定度，图优化，reverse edge addition，提高 reachability | 与我们最接近的是 reverse edge addition 思想；但 CAGRA 是全局 GPU graph index，我们是 label group-local exact-anchor 增强 |
| Relative/RNG graph methods | RNN-Descent、GRNND、RNG pruning | 通过相对邻居/occlusion 保留多样性边 | 我们只做低成本 reciprocal/reverse-tail，不做完整 RNG/RobustPrune；后续可借鉴做局部 diversity pruning |

## 3. 是否撞车

### 3.1 与 CAGRA 的关系

CAGRA 是 GPU-native graph-based ANN index。公开材料中明确包含：

```text
kNN graph construction
graph optimization / pruning
reverse edge addition
fixed out-degree GPU-friendly graph
```

撞车风险：**中等**。因为我们也使用了 reverse/reciprocal edge idea，并且同样追求固定出度和 GPU-friendly 构图。

关键差异：

| 维度 | CAGRA | 我们 |
---|---|---|
| 图对象 | 全局 ANN graph | UNG label-set group 内 PG |
| 构图输入 | 全数据集 kNN graph / global graph | 已按 label-set group 切分后的 group-local points |
| reverse 边 | 全局图 reverse edge addition | group-aware reverse，必须处理 local id -> global packed id -> local id |
| 质量目标 | GPU batch ANN search throughput / recall | filtered search recall under UNG query route |
| 输出边界 | GPU-native graph/search | 当前仍回填 CPU-compatible `Graph::neighbors` |

论文写法：

```text
We adopt the general insight that reciprocal/reverse edges improve graph reachability, as also used in GPU graph indexes such as CAGRA. Our contribution is not reverse edges in isolation, but a group-aware reciprocal augmentation for UNG's packed exact-anchor builder, where edges are group-local and must preserve filtered-search semantics.
```

### 3.2 与 Tagore / GNN-Descent 的关系

Tagore 目标是 GPU 加速 NSG/Vamana-style graph indexing。它包含 GNN-Descent 初始化和多种 prune/refine。

撞车风险：**中等偏低**。原因是我们代码确实使用 Tagore/FastGrnnd 作为大组路线，但本次方法修改的是 medium group 的 packed exact-anchor 输出策略。

| 维度 | Tagore/FastGrnnd | 我们这次方法 |
---|---|---|
| 候选生成 | GNN-Descent approximate candidates | exact in-group scan/topK |
| 剪枝 | light prune / reverse-tail / heavy prune | fixed-degree exact-anchor + group-aware reverse slots |
| 成本 | 大组可摊销；中组 heavy prune 可能太慢 | 中组保持 exact kernel，不增加主距离计算 |
| 目标 | 通用 graph index construction | UNG group-local PG speed/quality tradeoff |

论文写法：

```text
For large groups we rely on Tagore/FastGrnnd-style GPU graph construction. For medium groups, however, heavy pruning is not cost-effective. We therefore use an exact-anchor CUDA builder and add only a lightweight group-aware reverse augmentation.
```

### 3.3 与 Vamana / DiskANN RobustPrune 的关系

Vamana 的 RobustPrune 用几何条件控制冗余边，目标是低直径、强导航性和 bounded degree。

撞车风险：**低到中**。我们是受它启发，但没有实现完整 RobustPrune。

| 维度 | Vamana RobustPrune | 我们 |
---|---|---|
| 候选来源 | Greedy search 访问集 + existing neighbors | exact in-group topK + anchor/reverse candidates |
| pruning | alpha-based occlusion / robust prune | 低成本 slot replacement |
| 计算成本 | CPU 上可接受，GPU medium groups 上未必划算 | 不增加 exact distance compute |
| 质量 | CPU baseline | GPU quality-enhanced candidate |

边界：

```text
不能写“我们实现了 Vamana pruning 的 GPU 版本”。
只能写“借鉴 Vamana/RNG 的多样性与互惠可达性思想，以轻量 slot replacement 形式用于 exact-anchor 输出”。
```

### 3.4 与 NSG / RNG / RNN-Descent / GRNND 的关系

NSG 近似 MRNG，RNG/RNN-Descent/GRNND 都强调 relative-neighbor 或 occlusion 类边选择。

撞车风险：**低**。我们没有做完整 relative-neighbor descent，也没有做全局 NSG。

相同点：

- 都认为纯 nearest neighbors 容易形成局部 clique。
- 都需要长边/多样性/互惠边改善低 Lsearch 导航。

不同点：

- 我们只替换少量 tail slots，不做全局 occlusion。
- 我们限定在 UNG group-local exact-anchor 输出阶段。
- 我们的评价是 filtered query recall，而不是普通 ANN recall。

### 3.5 与 Filtered-DiskANN / StitchedVamana 的关系

Filtered-DiskANN 的 StitchedVamana 会为 filter/label 子集构图并合并，目标是 filtered ANNS。

撞车风险：**低到中**。因为二者都处理 filtered ANN 和 label/filter 语义，但图构建粒度不同。

| 维度 | StitchedVamana | UNG + 我们的方法 |
---|---|---|
| filter model | 单/多 filter label 对应点集 | label-set containment / LNG / group |
| graph construction | per-filter Vamana overlay | group-local PG + LNG/cross-edge |
| 本次增强 | 不适用 | group-local exact-anchor reverse augmentation |

论文写法：

```text
Filtered-DiskANN addresses filtered search by building and stitching filter-specific Vamana graphs. UNG instead organizes label-set containment into groups and a label navigation graph. Our contribution lies inside the group-local proximity graph construction, not in replacing the filtered index framework.
```

## 4. 我们方法的可写定位

建议定位：

```text
A lightweight group-aware reciprocal augmentation for GPU exact-anchor group graph construction in filtered vector search.
```

贡献点应写成组合：

1. `workload-aware group graph router`：small / medium / large group 分治。
2. `packed exact-anchor CUDA builder`：medium group exact topK，避免 heavy prune。
3. `direct-H2D / GPU lookup`：消除 host pack。
4. `group-aware reciprocal/reverse-tail augmentation`：低成本提升低 Lsearch 导航质量。
5. `full-quality filtered-search evaluation`：用端到端 recall，而不是局部 topK overlap。

不建议定位：

```text
新的通用 GPU ANN graph construction algorithm
GPU Vamana / GPU NSG 的完整替代
CAGRA 的替代
完整 RobustPrune GPU 化
```

## 5. 当前与现有方法的差异总结表

| 方法 | 是否撞车 | 相同点 | 关键差异 | 我们应如何引用 |
|---|---|---|---|---|
| CAGRA | 中 | GPU graph、fixed degree、reverse edges | CAGRA 是全局 GPU ANN graph；我们是 UNG group-local exact-anchor | 引用 reverse edge / GPU graph indexing 思想，强调 filtered group-local 差异 |
| Tagore | 中偏低 | GPU graph construction、GNN-Descent、prune | 我们对 medium groups 用 exact-anchor，不走 GNN prune | 引用 GPU graph indexing 基础，说明我们是 workload-aware router 的 medium path |
| DiskANN/Vamana | 中 | bounded degree、robust navigation | 我们不是 RobustPrune；只做轻量 reverse slots | 引用 CPU quality baseline 和 diversity motivation |
| NSG/RNG | 低 | 多样性/相对邻居思想 | 不构建全局 NSG/MRNG | 引用作为导航多样性理论背景 |
| Filtered-DiskANN | 低到中 | filtered ANN | filter-specific stitched graph vs UNG label-set group | 引用 filtered ANN baseline，强调框架不同 |
| cuVS Vamana | 中 | GPU Vamana/DiskANN construction | 普通 Vamana/DiskANN vs UNG group-local PG | 引用为 GPU Vamana library baseline，强调 filtered group 约束 |
| BANG | 低 | GPU/CPU 协同、减少 PCIe 数据流 | 主要是 search system，不是 group graph construction | 引用为 future GPU search/memory hierarchy 方向 |
| HNSW | 低 | 多样性邻居选择、避免 clique | hierarchical NSW，与 exact-anchor 不同 | 引用多样性边选择动机 |
| UNG | 高相关 | label-set containment, group graph, LNG/cross-edge | 我们是在 UNG group graph backend 上做 GPU 化 | 作为直接基础工作 |

## 6. 论文风险与防守口径

| Reviewer 可能质疑 | 回答 |
|---|---|
| reverse edges 不是新东西，CAGRA 已经有 | 承认 reverse/reciprocal 是已有思想；创新点是 group-aware local/global id handling + exact-anchor medium-group builder + filtered-search full-quality 证据 |
| 这是不是 GPU Vamana / NSG | 不是。我们没有实现完整 RobustPrune / NSG pruning，只做轻量 slot-level augmentation |
| 为什么不用 CAGRA / cuVS | CAGRA 是全局 ANN graph；我们的图是 UNG group-local PG，受 label-set containment、LNG、cross-edge 和 filtered search route 约束 |
| 只看 group graph 局部 topK 是否足够 | 不够。我们用 filtered-search recall sweep，尤其低 Lsearch，作为质量证据 |
| 低 L 仍有 gap | 是。当前 L20 gap 缩小超过一半但未完全闭合；后续方向是更强的 local diversity pruning |

## 7. 后续可查的论文方向

| 方向 | 为什么值得继续查 |
|---|---|
| CAGRA reverse edge addition / graph optimization | 可借鉴 rank-based reverse merge，可能比当前 fixed reverse slots 更稳 |
| Vamana RobustPrune / alpha pruning | 可设计 GPU-friendly local occlusion，只作用于最后几个 slots |
| NSG / MRNG pruning | 提供多样性边选择理论背景 |
| RNN-Descent / GRNND | 可借鉴 relative-neighbor GPU pruning，但需控制成本 |
| Filtered-DiskANN / StitchedVamana | 作为 filtered ANN 框架对照，避免误称独创 filtered graph idea |
| cuVS Vamana / GPU DiskANN | 作为通用 GPU Vamana/DiskANN 构建对照，避免误称我们是 GPU Vamana |
| HNSW neighbor selection | 作为多样性邻居选择动机 |
| BANG / hybrid GPU search | 作为未来 search-side CPU/GPU 协同与 PCIe 控制参考 |

## 8. 创新性与性能对比：Tagore / CAGRA / cuVS Vamana vs 我们

本节单独从创新性和性能两个角度比较。不同系统的论文数字通常来自不同数据集、不同 recall 目标、不同硬件和不同任务定义，因此不能直接把绝对 speedup 当作同表同口径结论。下表区分“论文/系统报告数字”和“本仓库实测数字”。

### 8.1 创新性对比

| 方法 | 核心创新 | 和我们相同点 | 和我们不同点 | 撞车判断 |
|---|---|---|---|---|
| Tagore / GNN-Descent | GPU GNN-Descent 初始化；GPU 支持 NSG/Vamana-like graph indexing；面向通用 graph index construction | 都是 GPU graph construction；都关注候选生成和 prune/refine | Tagore 面向通用 graph indexing；我们的 medium-group 方法走 exact-anchor，不走 GNN-Descent；本次新增的是 group-aware reverse augmentation | 不直接撞车；Tagore 是大组/通用构图基础，我们是 UNG medium-group 特化 |
| CAGRA | GPU-native fixed-degree graph；NN-Descent/kNN graph；图优化；reverse edge addition；面向 GPU batch search | 都使用 fixed degree 和 reverse/reciprocal reachability 思想 | CAGRA 是全局 ANN graph；我们是 filtered UNG group-local PG；需要 local/global id 映射和 filtered-search recall | 思想相近但问题不同；需要引用 CAGRA 说明 reverse edge 是已有技术 |
| cuVS Vamana | GPU 加速 Vamana/DiskANN index construction | 都涉及 GPU 上构建 Vamana/graph index | cuVS Vamana 构建普通 Vamana/DiskANN 图；我们不实现完整 RobustPrune，只做 exact-anchor + reverse slots | 中等风险；要明确我们不是 GPU Vamana |
| HNSW/NSG/Vamana/RNG | 多样性邻居选择、RobustPrune/RNG pruning、防止局部 clique | 都强调多样性边改善导航 | 我们是低成本 slot-level augmentation，不是完整 pruning algorithm | 低风险；作为理论动机引用 |
| Filtered-DiskANN / UNG | filter/label-aware graph ANN | 都服务 filtered search | Filtered-DiskANN 是 stitched filter-specific Vamana；UNG 是 label-set containment graph；我们改的是 UNG group-local PG backend | 框架相关但不撞车 |

### 8.2 性能对比边界

| 方法 | 论文/系统报告性能 | 我们能否直接复现/同口径比较 | 我们当前实测相关数字 | 结论 |
|---|---|---|---|---|
| CAGRA | CAGRA 论文报告 graph construction 相对 HNSW `2.2x~27x`；NVIDIA/cuVS 集成材料报告 CAGRA/HNSW build 可到 `8x~15x` 或 `12.3x` 等量级 | 不能直接同口径比较：CAGRA 是全局 ANN graph，不是 UNG group-local filtered PG | 我们 x200 group graph：CPU `8511.72 ms`，default direct-H2D `3204.88 ms`，reverse-enhanced `1822.89/2063.79 ms` | CAGRA 的通用 GPU graph build speedup更大；我们贡献在 filtered UNG group-local medium groups 和 recall gap 修复 |
| Tagore / FastGrnnd | Tagore 论文目标是 GPU 加速 NSG/Vamana-like graph indexing，使用 GNN-Descent + prune/refine | 代码里有 Tagore/FastGrnnd 路线，可在部分 workload 同仓库比较 | x200 旧 heavy prune group `26620.6 ms`，慢于 CPU；light/diversified prune group `6706.36 ms`；packed exact-anchor `3827.28 ms`；direct-H2D exact-anchor `3204.88 ms`；reverse-enhanced `1822.89/2063.79 ms` | 在 x200 medium-group workload，我们的 exact-anchor route 明显优于直接套 FastGrnnd/Tagore prune；但大组仍需要 Tagore/FastGrnnd |
| cuVS Vamana | NVIDIA 报告 GPU DiskANN/Vamana build 可有很高 CPU 加速（公开材料有 `40x or greater` 等说法） | 不能直接同口径比较：普通 Vamana/DiskANN vs UNG group-local PG | 我们没有运行 cuVS Vamana baseline；仅以 CPU Vamana group graph 作为质量/速度 baseline | 只能作为相关系统，不可宣称超过 cuVS Vamana |
| CPU Vamana | 本仓库质量 baseline | 可直接同口径比较 | x200 CPU group `8511.72 ms`；reverse-enhanced exact-anchor group `1822.89~2063.79 ms`，约 `4.1x~4.7x` group speedup；低 L recall gap 缩小但未完全消失 | 我们在该 filtered group workload 上优于 CPU build speed，质量有 tradeoff |

### 8.3 我们相对 Tagore 的性能判断

在本仓库 x200 medium-group workload 中，直接使用 Tagore/FastGrnnd prune 并不是最优：

| 路线 | group / index | recall 口径 | 判断 |
|---|---:|---|---|
| old FastGrnnd heavy prune | group `26620.6 ms`，Index `38273.7 ms` | L1000 `0.8662` | 慢于 CPU，不适合 x200 中组 |
| diversified light prune | group `6706.36 ms`，Index `16478.5 ms` | L1000/L5000 `0.8656/0.9056` | 性能转正但仍有 recall gap |
| packed exact-anchor | group `3827.28 ms`，Index `11043.5 ms` | L1000/L5000 `0.869/0.908` | 中组强候选 |
| direct-H2D exact-anchor | group `3204.88 ms` | Lsearch sweep：L20 gap `-0.0101`，L500 gap `-0.004`，L2000/L5000 对齐 | 快，但低 L 有 gap |
| group-aware reverse exact-anchor | group `1822.89~2063.79 ms` | L20 gap 缩小超过一半，L500 对齐/略高 CPU | 当前 medium-group 最优 speed-quality 点 |

因此，和 Tagore 的关系应写成：

```text
Tagore/FastGrnnd is a strong general GPU graph indexing baseline, especially useful for large groups. However, for UNG medium-sized label groups, heavy or even light GNN-based pruning is not the best tradeoff. A packed exact-anchor builder with direct-H2D and group-aware reverse augmentation gives better construction time and competitive filtered-search recall.
```

### 8.4 我们相对 CAGRA 的性能判断

CAGRA 的论文/系统数字通常比我们的 group-local speedup 更大，例如相对 HNSW 的 build speedup 可到 `2.2x~27x`，系统集成里也常报告 `8x~15x` 或更高量级。但这不是同一任务：

| 维度 | CAGRA | 我们 |
|---|---|---|
| 数据组织 | 全局 ANN graph | label-set group 内 PG + LNG/cross-edge |
| 目标 | GPU-native ANN graph construction/search | filtered UNG group graph construction |
| 质量评估 | ordinary ANN recall / throughput | filtered-search recall under query labels |
| reverse edge | graph-level reverse edge addition | group-aware local reverse augmentation |
| 性能可比性 | 不能直接用 CAGRA speedup 对比 | 只能说明相关思路和系统定位 |

建议写法：

```text
CAGRA demonstrates that reverse edges and GPU-native fixed-degree graph construction are effective for global ANN graphs. Our work applies the reciprocal-edge idea to a different setting: group-local exact-anchor construction for filtered search, where preserving label-containment semantics and local/global id correctness is essential.
```

### 8.5 当前创新性边界

可以主张：

- workload-aware GPU group graph route；
- medium group 使用 packed exact-anchor 而非重 GNN prune；
- direct-H2D / GPU lookup 消除 host pack；
- group-aware reciprocal/reverse-tail augmentation 缩小低 Lsearch filtered recall gap；
- full-quality filtered-search L sweep 作为质量证据。

不能主张：

- 发明 reverse edges；
- 超过 CAGRA/cuVS Vamana 的通用 GPU graph construction；
- 实现完整 GPU Vamana/NSG/RobustPrune；
- 所有 Lsearch 无损；
- 所有 group size 上都优于 Tagore/FastGrnnd。

## 9. 当前结论

我们的方法与 CAGRA、Vamana、NSG、Tagore 等有明确相关性，但不构成直接撞车。最接近的已有思想是 **reverse edge addition / reciprocal diversity**。我们的差异在于：

1. 场景是 UNG filtered vector search 的 group-local PG。
2. 基底是 packed exact-anchor CUDA builder，不是全局 kNN graph refinement。
3. reverse augmentation 必须 group-aware 处理 local/global id。
4. 目标是在不增加 exact distance compute 的情况下缩小低 Lsearch recall gap。
5. 评价使用 full-quality filtered-search recall sweep，而不是单组 topK 或普通 ANN recall。

因此论文写法应主动承认：

```text
Reverse/reciprocal edges are a known graph ANN technique.
Our contribution is adapting this idea to GPU exact-anchor group graph construction under UNG's filtered-search setting, with a workload-aware router and end-to-end filtered recall evidence.
```
