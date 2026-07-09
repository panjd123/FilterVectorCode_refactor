# UNG 组内图最终 GPU 方法总结

本文记录当前最终推荐的组内 PG 构建方法，以及它和 Tagore / GRNND 的边界。

## 方法名称

当前推荐后端：

```bash
UNG_GROUP_GRAPH_IMPL=3
```

代码名是 `FastGrnndCuda`。论文表述上建议称为：

**Adaptive Reverse-augmented Local Pruning**

中文可以写作：

**自适应反向候选增强局部剪枝**

它不是严格复现 GRNND，也不是 CPU Vamana 的逐边等价实现。它的目标是服务 UNG 组内图：在保持 GPU 构建速度的前提下，提高最终 UNG search recall。

## 核心设计

该方法把组内图构建拆成两步：

1. 使用 Tagore 的 GPU GNN-Descent 产生初始候选池。
2. 跳过 Tagore 的重 `select_path + filter_reverse` prune，改用我们自己的自适应局部剪枝。

更准确地说，当前工程实现已经从“单一 FastGrnndCuda 后端”扩展成 **workload-aware hybrid builder**。`AdaptiveCuda` 入口会先按 `nx` 对 group 分桶：

```text
for each group:
    if nx <= complete_threshold or nx < tagore_min_group_size:
        fallback group
    else:
        GPU group

for each GPU group:
    if nx <= exact_batch_threshold:
        packed exact-anchor batch
    else:
        GNN/FastGrnnd batch
```

这段逻辑现在集中在 `UNG/codes/src/uni_nav_graph_group_graph.cpp` 的 `build_graph_for_all_groups_tagore_cuda()`、`partition_tagore_groups()` 和 `build_tagore_batch_artifacts()`。它不是运行时根据 recall 动态选择，而是在 build 前按 group size 做静态路由。fallback 与 GPU batch 可以并发执行，避免 CPU 小组构图完全串行挡住 GPU batch。

## 代码级算法流程

### A. fallback small-group path

fallback path 处理两类组：`nx <= complete_threshold` 的很小组，以及小于 `tagore_min_group_size`、不值得上 GPU 的组。代码在 `UNG/codes/src/uni_nav_graph_group_graph.cpp` 的 `build_tagore_fallback_groups()`。

如果 `nx <= complete_threshold` 或 `UNG_TAGORE_FALLBACK_IMPL=0`，直接构造 complete 或 bounded-complete graph：

```text
if nx <= max_degree + 1:
    每点连接同组内所有其它点
else:
    每点连接 (i+1) % nx, (i+2) % nx, ... , (i+max_degree) % nx
```

bounded-complete 的实现位于 `UNG/codes/src/uni_nav_graph.cpp` 的 `build_bounded_complete_graph()`。它牺牲了 complete graph 的逐边等价性，但把小组 fallback 的出度固定到 `max_degree`，避免大量小组物化 `nx * (nx-1)` 边。

如果不走 complete/bounded-complete，则回退到 CPU Vamana。这一路径保留 CPU Vamana 语义，但只用于 GPU 不划算的小组或兼容口径。

### B. packed exact-anchor path

packed exact-anchor 面向 small-to-medium groups。入口是 `build_fast_exact_cuda_batch()`，代码在 `UNG/codes/src/tagore_graph_builder.cu:969-1280`。

它的算法步骤是：

1. 将多个 group 拼成一个 packed point array，同时保存 `offsets[group]` 和 `sizes[group]`。
2. 如果 group 的 base vectors 在 host 内存中连续，则合并成 direct-H2D run，直接拷到 GPU，避免 CPU 临时 pack。
3. 在 GPU 上生成每个 packed point 对应的 `group_id` 和 `local_id`。
4. 对每个 source point，扫描同 group 的所有 destination point，计算 exact L2 距离，并维护 topK。
5. 输出邻接行时不直接写纯最近邻，而是写 **nearest head + anchor tail + sampled tail**。

exact kernel 在 `UNG/codes/src/tagore_graph_builder.cu:657-790`。每个 block 处理一个 source point，128 个线程跨维度累加 L2 距离，block reduce 后由 thread 0 更新共享内存里的 topK。

输出规则如下：

```text
near_limit = min(out_degree, head_keep, final_degree)
先写 top_idx[0 : near_limit]

如果 anchor_tail 打开:
    对剩余名额，按 hop = ((t+1) * nx) / (remaining+1)
    写入 candidate = (local_src + hop) % nx

然后从 topK 尾部按 step 等距采样
最后线性扫描尾部补满并去重
```

这里的 anchor tail 是一个导航性设计：它让 exact graph 不只是局部最近邻图，而是在每个点的边表尾部保留少量分散连接，降低局部团簇过强导致的导航退化风险。

### C. GNN candidate generation

大组进入 GNN/FastGrnnd batch。候选生成仍复用 Tagore 的 GPU GNN-Descent kernel，batch path 位于 `UNG/codes/src/tagore_graph_builder.cu:1848-1900`。

主要阶段是：

```text
initialize_graph
cal_power
nn_descent_opt_sample
nn_descent_opt_reverse_sample
nn_descent_opt_cal
nn_descent_opt_merge
do_reverse_graph
sample_kernel6
merge_reverse_plus
```

这一步输出每个点的候选邻居表 `graph_dev` 和候选距离 `nei_distance`。我们没有重写 NN-Descent 主循环，主要改的是后续 prune 和系统级 batching/fill。

### D. light prune

light prune 用于 `nx <= UNG_FAST_GRNND_LIGHT_PRUNE_NX` 的中等组，代码在 `UNG/codes/src/tagore_graph_builder.cu:396-500`。

它不重新计算候选距离，也不做 alpha occlusion。它假设 GNN-Descent 的候选已经大致按距离排序，然后按以下规则剪枝：

```text
1. 保留前 head_keep 个有效近邻。
2. 如果还没满，从候选尾部按 step 等距采样，增加多样性。
3. 再从尾部线性扫描补满，去重。
4. 如果 degree 仍不足并启用 repair，则按 (src + hop) % nx 环形补边。
```

这个分支的目标是解决 `nx≈200` 上 heavy prune 无法摊销的问题。它的代价是质量可能下降，所以只适合中等组或作为 router 的 fast path。

### E. light reverse-tail prune

reverse-tail 是 light prune 的质量增强版本，代码在 `UNG/codes/src/tagore_graph_builder.cu:502-632`。

它先调用 `build_sampled_reverse_kernel()`，对候选图里的前若干 forward edges 做反向采样：

```text
for each edge src -> dst in forward candidates:
    sampled_reverse[dst].push(src)
```

反向采样代码在 `UNG/codes/src/tagore_graph_builder.cu:148-170`。随后 light reverse prune 的输出顺序是：

```text
1. nearest head
2. sampled reverse candidates，最多 reverse_slots 条
3. GNN tail 等距采样
4. GNN tail 线性补满
5. degree repair 环形补边
```

这个设计针对 x400 light prune 的结构问题：light prune 会减少低出度点和弱连通分量附近的入边覆盖，reverse-tail 用很小的额外预算把“别人认为我近”的点补进来，而不做完整 RNG occlusion。

### F. heavy FastGrnnd prune

heavy prune 用于大组或质量敏感组，代码在 `UNG/codes/src/tagore_graph_builder.cu:286-394`。它更接近 Vamana/RNG-style pruning：

```text
candidate_pool = forward GNN candidates[0:forward_cap]
               + sampled_reverse candidates[0:reverse_cap]

对 candidate_pool 中每个 dst:
    计算 dist(src, dst)
按 dist(src, dst) 排序

for dst in sorted candidates:
    reject = false
    for pivot in accepted_neighbors:
        if alpha * dist(pivot, dst) < dist(src, dst):
            reject = true
            break
    if not reject:
        accept dst
```

这保留了相对邻域 / occlusion 的导航图思想，但每个 candidate 都可能要和已接受邻居计算距离，因此比 light prune 明显更慢。当前策略是：中等组尽量不用 heavy prune；x400 这类质量敏感大组用 reverse-tail 或更强 prune 兜底。

### G. packed output and host graph fill

GPU 输出格式是每行：

```text
row[0] = degree
row[1 : degree+1] = local neighbor ids
```

对于 exact batch，D2H 后结果保留在连续 `packed_graph` 里，并用 `packed_offsets` 定位每个 group。`UniNavGraph` fill 阶段可以直接从 packed buffer 读取，不再先 scatter 到每个 group 的临时 `result.graph`。代码在 `UNG/codes/src/uni_nav_graph.cpp:1685-1715`。

但最终仍要写回 CPU-compatible `Graph::neighbors`，并创建 Vamana-compatible graph object / entry point。因此当前算法贡献和输出边界要分开写：GPU 侧已经生成了邻接边，系统仍保留 host graph materialization。

最终剪枝阶段包含三个关键技巧：

| 技巧 | 作用 |
| --- | --- |
| diversified light prune for medium groups | 对 `nx<=UNG_FAST_GRNND_LIGHT_PRUNE_NX` 的中等组保留 GNN-Descent 近邻头部，并从候选尾部采样少量多样性邻居，避免 reverse + RNG occlusion 在小负载上无法摊销 |
| degree repair for light prune | 默认开启的待验证质量修复：在 light prune 候选不足时用确定性环形边补足出度，直接针对 x400 中 zero/low intra-degree 和 weak-component 分裂 |
| light reverse-tail prune | 可选增强路径：在 light prune 中先构造少量 sampled reverse candidates，只把它们填入尾部预算，不做全量 RNG occlusion，用于验证 x400 上“低成本反向尾部多样性”是否能缩小 recall gap |
| compact D2H graph writeback | 默认开启的外围优化：prune 后把 `k` 宽 GPU 图压缩为 `final_degree+1` 宽再 D2H，减少 graph writeback 和 host fill 的无效数据 |
| mixed exact/GNN batch router | 新增调度优化：当 `UNG_FAST_GRNND_BATCH_EXACT_NX>0` 时，把符合阈值的小/中组单独合成一个 batched exact GPU 批次，大组继续走 Tagore/GNN 批次，避免一个大组把整批请求拖回逐组 GNN 路径 |
| packed exact graph buffer | batched exact D2H 后保留连续 graph buffer 和 group offsets，由 `UniNavGraph` fill 直接读取，避免先 scatter 到每组 `result.graph` 再写入 `_group_graphs` 的重复 host copy |
| bounded-complete small-group fallback | 新增系统 route：对很小组不盲目 GPU 化，也不物化完整 `nx-1` 出边；`nx<=max_degree+1` 保持 complete，否则写 `max_degree` 条环形强连通边，降低大量小组的 CPU 图物化成本 |
| old batched exact kNN ablation | 早期低 recall 结果后来被结构诊断归因为 cross/additional 口径混杂；保留为实验教训，不再作为 exact 组内图不可行证据 |
| 固定预算候选池 | 每个点只处理固定数量候选，避免 Tagore `SELECT_CAND=480` 带来的重后处理成本 |
| reverse-augmented candidates | 除 forward GNN 邻居外，采样“指向当前点”的反向候选，补充局部连通性和图多样性 |
| local RNG-style occlusion | 用相对邻域 / alpha occlusion 的轻量规则剪枝，保留更适合导航的邻居 |

## 和 GRNND 的区别

GRNND / RNN-Descent 的核心是把 RNG pruning 融入 NN-Descent 迭代过程，用 double-buffer neighbor pool 和 disordered propagation 持续更新候选集。

当前方法借鉴了它的方向，但实现上有明显区别：

| 对比项 | GRNND / RNN-Descent | 我们的最终方法 |
| --- | --- | --- |
| 候选生成 | 自己的 RNN-Descent 迭代框架 | 复用 Tagore GNN-Descent，降低接入风险 |
| 剪枝位置 | 融入迭代过程 | 作为轻量后端替换 Tagore 重 prune |
| 反向边 | 迭代中传播和更新 | prune 前显式 sampled reverse augmentation |
| 优化目标 | 通用高质量 GPU graph construction | 面向 UNG group graph 的端到端 build/search tradeoff |
| 工程边界 | 独立图构建算法 | 直接接入 `UniNavGraph`，支持环境变量 A/B |

因此，论文里不建议说“复现 GRNND”。更准确的说法是：

> Inspired by RNG-style pruning and reverse-neighbor propagation, we design a reverse-augmented local pruning backend tailored for UNG group graphs.

## 和 Tagore 的区别

Tagore 的 GPU Vamana pruning 质量较稳，但在当前 UNG group 场景里太重。

当前 `g1024_nx2048` 测试中：

| 方法 | build_graph | prune/refine | L20 | L50 | L100 | L200 | L500 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 原始 Tagore | 9348.87 ms | prune 4531.38 ms | 0.9355 | 0.9765 | 0.9860 | 0.9915 | 0.9945 |
| FastGrnndCuda 旧版 | 3790.43 ms | prune 657.63 ms | 0.9375 | 0.9790 | 0.9870 | 0.9920 | 0.9955 |
| 最终版：Reverse-augmented Local RNG | 4014.91 ms | prune 810.08 ms | 0.9515 | 0.9880 | 0.9930 | 0.9955 | 0.9965 |
| Method3 iter=4 | 8248.08 ms | prune 4492 + refine 944 ms | 0.9500 | 0.9880 | 0.9930 | 0.9965 | 0.9985 |
| CPU Vamana | 9265.03 ms | - | 0.9655 | 0.9910 | 0.9990 | 1.0000 | 1.0000 |

结论：

- 相比原始 Tagore，最终版同时提升 recall 和构建速度。
- 相比 Method3 iter=4，最终版 L20/L50/L100 基本持平，但 build_graph 从 `8248.08 ms` 降到 `4014.91 ms`。
- 相比 CPU Vamana，最终版 build_graph 约 `2.31x` 更快，但高 Lsearch recall 仍有差距。

## 为什么这个方法更适合写成我们的贡献

可以总结成三点：

1. **面向 UNG 的剪枝目标**  
   我们不追求构造一个和 CPU Vamana 逐边等价的通用图，而是优化最终 filtered search 的 recall / build time tradeoff。

2. **反向候选增强而非重 refine**  
   旧 Method3 在 Tagore 重 prune 后再做 refine，速度损失明显。最终版把 reverse 信息前移到轻量 prune 的候选池中，只增加约 `152 ms` prune 成本，却把 L20 从 `0.9375` 提到 `0.9515`。

3. **固定预算局部剪枝**  
   每个点处理固定上限候选，避免 Tagore Vamana prune 的复杂路径依赖和大候选扫描。因此在 `nx=2048` 这种中等 group 上也能获得明显加速。

## 当前推荐配置

```bash
UNG_BUILD_PROFILE=custom \
UNG_GROUP_GRAPH_IMPL=3 \
UNG_TAGORE_K=64 \
UNG_TAGORE_ITER=4 \
UNG_TAGORE_M=64 \
UNG_FAST_GRNND_LIGHT_PRUNE_NX=256 \
build_UNG_index ...
```

其中 `UNG_TAGORE_ITER=4` 是当前性价比较好的点。继续增加迭代数会提高 GNN-Descent 时间，但 recall 增益有限。`UNG_FAST_GRNND_LIGHT_PRUNE_NX=256` 是为 Amazon 1% x200 这类 `nx≈200` workload 增加的中等组 fast path。

为 x400 质量缺口新增的推荐实验配置：

```bash
UNG_FAST_GRNND_LIGHT_PRUNE_NX=512 \
UNG_FAST_GRNND_REPAIR_DEGREE=1 \
UNG_TAGORE_COMPACT_D2H=1 \
UNG_FAST_GRNND_LIGHT_REVERSE_CAP=8 \
UNG_FAST_GRNND_LIGHT_REVERSE_SLOTS=4 \
UNG_FAST_GRNND_LIGHT_REVERSE_FORWARD_CAP=24 \
build_UNG_index ...
```

默认 `UNG_FAST_GRNND_LIGHT_REVERSE_CAP=0`，即不启用 reverse-tail。默认 `UNG_FAST_GRNND_REPAIR_DEGREE=1` 会在 light prune 末尾补足低出度点；默认 `UNG_TAGORE_COMPACT_D2H=1` 会减少 GPU 图写回宽度。打开 reverse-tail 后，light path 会复用 sampled reverse 构建 kernel，从每个点的反向入边候选中取固定预算填入尾部名额。该路径不计算距离、不做 occlusion，目标是回答 x400 的 reviewer 问题：是否可以用接近 light prune 的成本补回一部分尾部多样性。

x400 A/B 结果显示：

| Case | Index | Group | Prune | L1000 | L5000 | low<=4 |
|---|---:|---:|---:|---:|---:|---:|
| light512_norepair | `30128.1` | `12360.1` | `1005.94` | `0.937067` | `0.9574` | `0.00937` |
| light512_repair | `36310.5` | `16545.7` | `1018.17` | `0.9378` | `0.957767` | `9.67e-5` |
| light512_reverse_repair | `33044.9` | `13996.7` | `1093.37` | `0.945567` | `0.9659` | `8.67e-5` |

同脚本 CPU Vamana baseline 为 Index `36257.3`、L1000/L5000 `0.946/0.965`，历史 CPU Vamana gather-Q full-quality 为 Index `37807.6`、L1000/L5000 `0.9460/0.9660`。因此 repair-only 是 partial/negative：结构改善明显，但速度变慢且 recall 只小幅提升。reverse-tail+repair 是当前 x400 推荐质量增强点：相对同脚本 CPU，Index 约 `1.10x`，L5000 略高，L1000 低约 `0.00043`。compact-D2H 已补 on/off ablation，结论是负/不稳定：在当前 x400+reserve 设置下 compact-on 没有稳定 build-time 收益，只能作为外围开销拆分，不能写成 x400 主加速来源。

新增的 mixed exact/GNN batch router 用下面的开关启用：

```bash
UNG_FAST_GRNND_BATCH_EXACT_NX=256
```

它只在 `UNG_GROUP_GRAPH_IMPL=3` 时生效。旧实现只有当整个 Tagore batch 的所有 group 都小于阈值时，才会进入 batched exact；一旦混入大组，所有小/中组也会被迫走逐组 GNN kernel launch。新实现会把同一批请求拆成 exact batch 和 GNN batch，再按原 group 顺序合并结果。最新实现还把 exact batch 的 host 结果保留为 packed graph buffer，fill 阶段按 packed offset 直接读，减少 D2H 后 scatter + fill 的重复 host copy。这个优化主要针对 many-group workload 的外围摊销，不改变下游 `Graph/Vamana` 接口。x200/x400 coverage-query 的 L5000 repeat 已经显示 packed exact-anchor 可以达到 CPU 级 recall；但它仍必须作为 workload-aware router，而不能未经 x100/10%x40/真实多标签确认就写成所有 workload 的无条件替代。

Amazon 1% x200 coverage-query repeat=1 初步 A/B：

| Case | Index | Group | L100 | L500 | L1000 |
|---|---:|---:|---:|---:|---:|
| CPU Vamana | `13473.7` | `7812.0` | `0.8230` | `0.8490` | `0.8690` |
| router exact nx0 | `16179.7` | `8606.52` | `0.8180` | `0.8463` | `0.8661` |
| router exact nx256 | `14098.0` | `5770.88` | `0.8229` | `0.8510` | `0.8690` |
| router exact nx512 | `13937.2` | `5840.47` | `0.8230` | `0.8500` | `0.8700` |
| packed exact-anchor nx4096 | `11043.5` | `3827.28` | - | `0.9080` | `0.8690` |

该结果说明 `nx256/512` 作为调度分支没有复现低质量问题，并能把 group graph 相对 non-exact router 加速约 `1.49x/1.47x`。packed exact-anchor `nx4096` 进一步说明外围 copy/fill 是实质瓶颈：group graph 对 CPU Vamana `8511.72 ms` 为 `2.22x`，对旧 scatter exact run `4929.54 ms` 为 `1.29x`；repeat=3 L5000 `0.9080` 与 CPU Vamana 对齐。仍需跨数据集验证。

Amazon 10% x40 的新实验进一步收紧了 router 边界。该数据集有 `53,840` 个 group 和大量 `nx<=64` 小组。把 `UNG_GROUP_GRAPH_COMPLETE_NX` 从默认 `64` 降到 `32`，让 `53,833` 个小/中组进入 packed exact 后，group graph 从 `6097.03 ms` 变慢到 `7329.12 ms`，主要时间在 `pack=2728.81 ms`、`h2d=1014.8 ms`、`fill=1968.77 ms`，L100 recall 仍为 `0.867`。这说明“中组 exact-anchor 有效”不能外推为“所有小组都上 GPU”。新增 `UNG_GROUP_GRAPH_BOUNDED_COMPLETE=1` 后，仍保持 `complete_threshold=64`，但小组 fallback 改为有界强连通图，10%x40 Index 从 `25035.9 ms` 降到 `22230.2 ms`，group 从 `6097.03 ms` 降到 `5240.6 ms`，fallback_wall 从 `4225.67 ms` 降到 `3088.13 ms`。复用该 index 的 repeat=3 search sweep 给出 L100/L500/L1000 recall `0.867/0.900/0.909`，与 CPU `0.8647/0.898/0.908` 同量级；但 query time `1083.34/951.434/839.479 ms` 比历史 CPU/FastGrnnd run 慢，需要同脚本复测和路径诊断。因此该路线目前应写成 build-side system route 和 recall sanity，而不是 search-latency 优化或 CPU Vamana 的逐边等价替换。

当前推荐的三段式 group graph router 是：

| Group bucket | 推荐实现 | 理由 |
|---|---|---|
| very small / fallback groups | `UNG_GROUP_GRAPH_BOUNDED_COMPLETE=1` | 避免完整小组图物化；强连通且出度有界；GPU pack/H2D/fill 对这类组不划算 |
| small-to-medium groups | `UNG_FAST_GRNND_BATCH_EXACT_NX` packed exact-anchor | x200/x400 coverage-query 已显示 CPU 级 L5000 recall 和更低 group build time |
| large / quality-sensitive groups | FastGrnndCuda light/reverse-tail 或更强 prune | x400 表明简单 light prune 有 recall gap，需要 reverse-tail / quality route 兜底 |

### Graph 输出边界优化

2026-06-02 新增 `UNG_GRAPH_RESERVE_CAPACITY`。未显式设置时走 auto 策略：点数大、group 数多、或小组 fallback 点数占比高时自动开启；显式设为 `0/1` 可覆盖 auto。它不改变组内图、cross-edge 或 search 语义，只是在 group graph 构建前按 `min(max_degree, group_size-1) + num_cross_edges + slack` 给每个点的 `Graph::neighbors` 预留容量，避免 GPU 输出、bounded complete fallback 和 cross-edge merge 在 `std::vector[point]` 上反复小分配。

Amazon 10%x40 的 reserve on/off A/B 显示这是实质瓶颈，而不是微小优化：

| 配置 | Index ms | Build graph ms | Cross edges ms | tagore fill ms | fallback wall ms | cross merge ms |
|---|---:|---:|---:|---:|---:|---:|
| reserve=0 | 39746.1 | 14238.1 | 14761.0 | 823.5 | 11844.6 | 1677.1 |
| reserve=1 | 24704.5 | 1662.7 | 9994.0 | 23.9 | 520.2 | 2.3 |

该优化的代价是 reserve 本身仍需 `6487.3 ms`，预留 `106,031,200` 条边容量，因此端到端收益为 `1.61x`，不是 build graph 局部的 `8.6x`。Amazon 1%x200 上，auto 关闭时 Index `16895.4 ms`，强制 reserve 时 Index `15436.9 ms`；虽然 reserve 花 `2958.7 ms`，但 `tagore_fill_time` 从 `2497.0 ms` 降到 `150.9 ms`，cross merge 从 `572.0 ms` 降到 `13.8 ms`。因此 auto 策略现在也按 `UNG_GRAPH_RESERVE_AUTO_MIN_POINTS=500000` 覆盖大点数中组 workload。

在此基础上，当前代码把 `Graph::neighbors` 从 `std::vector<IdxType>*` 改成 64-inline small-buffer `NeighborList*`，保持 `begin/end/size/resize/reserve/emplace_back/insert` 等调用接口。它把 reserve 作为 stopgap 的主要 allocator 成本吸收到邻接表对象内：x200 auto-reserve 的 reserve 从旧 `2856.2 ms` 降到 `25.4 ms`，`tagore_fill` 从 `224.8 ms` 降到 `14.7 ms`；10%x40 reserve-on 的 reserve 从 `6487.3 ms` 降到 `21.2 ms`，`tagore_fill` 从 `23.9 ms` 降到 `7.6 ms`。48-inline 版本是负例：x200 `tagore_fill` 回退到 `202.5 ms`，因此最终保留 64-inline。10%x40 的总 Index 本轮因 cross-edge `prepare_all H2D=4974.9 ms` 抖动而未稳定改善，所以该结果应写成输出边界阶段优化，而不是端到端主表加速。

这支持一个更强的系统结论：当前 GPU backend 的剩余主要阻力是 Graph 输出边界。最终普适替代应把 `Graph::neighbors` 改为 GraphView/CSR 或 implicit small-group view，使 GPU group graph 与 cross-edge 直接产出连续边缓冲，而不是先写回 per-node object array。详细设计见 `docs/papers/GRAPH_OUTPUT_BOUNDARY_OPTIMIZATION_CN.md`。

对应 x400 reviewer 实验入口：

```bash
scripts/benchmarks/run_x400_reverse_tail_ab.sh
```

默认输出三个变体：

| 变体 | 目的 |
|---|---|
| `light512_norepair` | 复现历史 light512，作为 repair 对照 |
| `light512_repair` | 单独评估 degree repair 是否降低 low-degree / weak-component 问题 |
| `light512_reverse_repair` | 评估 reverse-tail 在 repair 基础上是否补回 recall |

`RUN_COMPACT_ABLATION=1` 已补 `light512_repair_compact_off`。结果显示 compact-on `Index=39759.1 ms`、`D2H=294.133 ms`、`fill=249.144 ms`，compact-off `Index=33019.7 ms`、`D2H=292.299 ms`、`fill=70.4478 ms`；另一次 compact-on build-only rerun 为 `Index=37845.6 ms`、`D2H=254.135 ms`、`fill=163.966 ms`。因此 compact-D2H 在这组 x400+reserve 设置下不是稳定收益来源，只保留为 negative/ambiguous overhead ablation。

Amazon 1% x200 coverage-query 上：

| Variant | group graph | prune | Index | L1000 | L5000 |
|---|---:|---:|---:|---:|---:|
| CPU Vamana | 8511.72 | - | 17596.8 | 0.8690 | 0.9080 |
| FastGrnndCuda old heavy prune | 26620.6 | 19218.3 | 38273.7 | 0.8662 | - |
| FastGrnndCuda top32 light prune | 6016.04 | 712.392 | 14211.8 | 0.8619 | 0.9029 |
| FastGrnndCuda diversified light prune | 6706.36 | 759.66 | 16478.5 | 0.8656 | 0.9056 |
| old batched exact kNN, confounded | 4356.18 | 0 | 12595.9 | 0.8159 | 0.8199 |
| packed exact-anchor router, nx4096 | 3827.28 | 0 | 11043.5 | 0.8690 | 0.9080 |

这个结果说明：`nx≈200` 的失败原因不是 GPU GNN 候选生成，而是重 prune 无法摊销；同时，旧 batched exact 低 recall 不是组内 exact 图本身的问题，而是 cross/additional 口径混杂。packed exact-anchor router 在 full-quality 口径下同时给出最快 group graph 和 CPU 级 L5000 recall，是当前最强候选。

## 后续可继续优化点

新增调度优化：

```bash
UNG_TAGORE_BATCH_STREAMS=4
```

原先 `build_tagore_vamana_cuda_batch` 的 GNN 路径虽然复用了一套 workspace，但仍然按 group 串行执行整串 Tagore/GNN-Descent kernel。对 `x40/x100/x200` 这类 many-group workload，每个 group 的 kernel grid 很小，单个 group 很难填满 GPU，串行 launch 还会把 H2D、convert、GNN、prune、D2H 的短阶段全部排队。现在新增可选多 stream 调度：把 group 按点数做负载均衡拆分，每个 worker 使用独立 non-blocking CUDA stream 和独立 workspace 并发处理，最后按原 group 顺序合并结果。该优化不改变 graph/prune 语义，主要面向“保留 FastGrnndCuda 质量、减少 many-group 外围和小 kernel under-utilization”的普适替代方向。

注意事项：

- 默认 `UNG_TAGORE_BATCH_STREAMS=1`，保证历史配置不变。
- stream 数不是越多越好；每个 stream 都会持有一份 `max_points` workspace，显存压力随 stream 数增加。
- 该优化预期对 many small/medium groups 更有效，对单个大 group 或已进入 batched exact 路径的场景收益有限。
- 该优化仍需在 GPU 空闲后重测 `10%x40`、`1%x100/x200/x400`，重点看 `direct build wall` 是否下降，而 per-group 阶段时间求和可能因并发大于 wall time。

Amazon 1% x200 coverage-query 的快速 A/B 结果：

| 配置 | Index | group graph | direct build wall | cross | L100 recall |
|---|---:|---:|---:|---:|---:|
| `streams=1` baseline | 16408.5 ms | 9097.11 ms | 5519.41 ms | 4670.05 ms | 0.8172 |
| `streams=4` | 16542.8 ms | 8751.60 ms | 3388.89 ms | 4919.28 ms | 0.8176 |
| `streams=4, fast_fill=1, fill_threads=32` | 15552.3 ms | 8131.67 ms | 3148.55 ms | 4753.44 ms | 0.8180 |
| `streams=2, fast_fill=1, fill_threads=16` | 13905.5 ms | 6441.52 ms | 4308.56 ms | 4837.63 ms | 0.8192 |
| `streams=8, fast_fill=1, fill_threads=16` | 15125.8 ms | 6320.87 ms | 3089.78 ms | 4989.27 ms | 0.8180 |

当前最好 group graph 时间来自 `streams=8, fast_fill=1, fill_threads=16`，相对单 stream baseline 为 `1.44x`；当前最好 Index 来自 `streams=2, fast_fill=1, fill_threads=16`，相对 baseline 为 `1.18x`。这组结果说明 GPU direct build 已能通过并发 stream 明显加速，但端到端 Index 仍受 CPU fill、cross-edge 波动以及后续串行阶段限制。下一步若继续追求数量级收益，应该把“GPU batch 完成后再统一 CPU fill”的结构改成分块流水：每个 worker stream 完成一批 group 后立即在 CPU 侧 fill，同时其他 stream 继续 GNN/prune/D2H。

剩余优化点：

1. compact-D2H 已完成负/不稳定 ablation：不能把它写成 x400 主收益或算法质量贡献，只能作为外围开销拆分和反例。
2. x400 同脚本 CPU baseline 已补：Index `36257.3 ms`、L1000/L5000 `0.946/0.965`；后续重点转为 reverse-tail 参数扫和真实多标签。
3. 做 budgeted occlusion：只对 top few candidates 做 RNG-style 检查，避免旧 heavy prune 的 `候选数 * R * dim` 成本。
4. 将 sampled reverse buffer 纳入 batch workspace，避免 prune 内部 `cudaMalloc/cudaFree`。
5. 对 forward / reverse 候选比例做自适应，例如按 group size 或候选重复率调整。
6. 增加轻量入口点选择，避免 FastGrnndCuda 当前默认 entry point 过于简单。
