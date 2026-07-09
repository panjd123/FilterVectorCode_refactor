# 面向分层过滤向量搜索的 UNG 构建加速：Grouped Fused TopK 与反向候选增强局部剪枝

> 论文草稿，中文技术版。本文以投稿论文为目标组织现有方法、实验和边界；后续可翻译成英文并按目标会议模板裁剪。

## 摘要

过滤向量搜索系统通常需要同时处理向量近邻关系和标签/属性约束。Unified Navigating Graph (UNG) 通过标签层级结构组织数据，并在构建阶段生成组内图、标签导航图和跨组边，以支持带过滤条件的近似最近邻查询。然而，在真实标签分布下，UNG 构建不是单个大规模密集线性代数问题，而是由大量大小不均的 group workload 组成：每个目标组需要从若干父/邻接组收集查询向量，并在目标组内执行 topK 连接；同时，组内图构建在大组和长尾小组上呈现完全不同的性能特征。现有 CPU Vamana 构建路径在这些阶段存在明显瓶颈，直接逐 group 调用 GPU 库也会被数据准备、API 调用和写回成本吞噬。

本文提出一套面向 UNG 构建负载的 GPU 加速方案。对于跨组边构建，我们设计 grouped fused topK，将 group-level `(nq, nx, dim) -> topK` 批处理到少量 GPU kernel 中，并引入 direct-qid 常驻向量访问、TF32 WMMA group kernel、安全写回策略和分组描述符调度，减少 query 展开、H2D/D2H 和逐 group 调用开销。对于组内图构建，我们提出 FastGrnndCuda：复用 GPU GNN-Descent 生成候选，然后用反向候选增强的局部 RNG-style 剪枝替代重后处理，在中大组上降低构建时间并改善相对 Tagore 的图质量。

在 Amazon sampled + jitter repeat 系列上，本文方法在 Amazon 1% x100 上将 cross-edge 从 CPU Vamana 的 `18735.8 ms` 降到 `848.9 ms`，相对 CPU exact scan达到 `2.81x`，相对 cuVS per-group 达到 `4.39x`，相对 SGEMM+topK 达到 `1.64x`。端到端构建中，当前混合方案在 Amazon 1% x100 上将 Index time 从 CPU Vamana group + GPU fused baseline 的 `16824 ms` 降到 `6630 ms`，达到 `2.54x`，但该历史点是 skip-additional 口径。新的 x100 full-quality conservative `adaptive_cuda` route 进一步把小组 CPU fallback、中组 packed exact-anchor、大组 FastGrnndCuda 收敛到一个后端：Index `5450 ms`、group graph `2773.35 ms`、L100/L500/L1000 `0.829/0.870/0.895`，相对历史 FastGrnnd full-quality `6370.67 ms`、`3216.21 ms`、`0.828/0.871/0.896` 保持 recall 基本一致并改善 build time。新的 Amazon 1% x200 coverage-query 端到端实验显示，旧 FastGrnndCuda 在 `nx≈200` bucket 上被重 prune 拖慢：group graph `26620.6 ms`，慢于 CPU Vamana 的 `8511.72 ms`。加入 diversified light prune 后，group graph 降到 `6706.36 ms`，Index time 从 CPU 的 `17596.8 ms` 降到 `16478.5 ms`；L1000 recall 为 `0.8656`，低于 CPU 的 `0.8690`，L5000 recall 为 `0.9056`，接近 CPU 的 `0.9080`。进一步去掉 batched exact path 的 D2H 后 host scatter 后，packed exact-anchor router 在 x200 上把 group graph 降到 `3827.28 ms`、Index 降到 `11043.5 ms`，repeat=3 L1000/L5000 为 `0.869/0.908`；x400 同路径 group graph 为 `11588.4 ms`、Index 为 `28436.3 ms`，repeat=3 L1000/L5000 为 `0.945/0.967`。新的 direct-H2D A/B 进一步表明，旧 x200 exact-anchor 性能主要受外围限制：旧路径 group `8999.26 ms` 中 `pack=4140.61 ms`、GPU exact kernel `931.72 ms`、fill `2849.50 ms`；direct-H2D + GPU lookup + fill16 后 group 降到 `3204.88 ms`、`pack=0.04 ms`、kernel `928.67 ms`、fill `1383.53 ms`，L1000 recall 仍为 `0.867`。结构诊断还显示，早期 exact 低 recall run 与新 packed exact 的组内图完全一致，差异来自 cross edges 缺失，因此旧 exact 负结果应视为实验口径混杂。针对 cross-edge 的调度边界，我们进一步修复了 double-buffer direct-qid path 的 all-or-nothing 回退：Amazon 1% x200 full-quality 中，partial routing 将 `5173/5195` 个 target groups 送入 double-buffer，仅 `22` 个大组回退，使 cross-edge 从旧 target-current `3900.81 ms` 降到 `2657.23 ms`，repeat=3 L1000/L5000 为 `0.866/0.907`。最新 universal flat double-buffer route 将 target-centric descriptor batching、double-buffer execution、GPU global merge 和 flat-id output 收敛到 `UNG_UNIVERSAL_GPU=1`：SIFT30 skip-additional cross-edge `3180.63 ms`，Amazon 1% x200 full-quality cross-edge `2494.21 ms` 且 L1000/L5000 recall `0.871/0.911`；Amazon 1% x100 full-quality cross-edge `1689.40 ms` 且 L100/L500/L1000 recall `0.826/0.868/0.891`，但该 x100 点没有带来端到端 Index 加速。Amazon 10% x40 many-group 压力测试进一步显示，修复 direct-pageable 和 direct-qid chunk path 后，full-quality A/B 中 FastGrnndCuda + GPU cross 把 Index 从 CPU Vamana group 的 `36053.3 ms` 降到 `23497.4 ms`，达到 `1.53x`；repeat=3 下 L100/L500/L1000 recall 从 `0.8647/0.898/0.908` 到 `0.8668/0.898/0.9103`，且 P95/P99 query latency 没有恶化。查询阶段，我们进一步把 CPU `get_min_super_sets_debug()` 改写为 GPU correct-cover 批处理入口组选择：Amazon 100%x40、`nq=10240` 上 resident 完整接口时间为 `59.83 ms`，相对 CPU scan 128T `1187.56 ms` 为 `19.85x`；在 10%x40 已有 search artifact 中，CPU 入口组查找占 query latency 约 `48~51%`，若只替换入口组阶段且后续图搜索不变，端到端 query latency 上界约 `1.9~2.0x`。本轮还生成并构建 Amazon 100%x40：`23,284,680` 点、`482,387` groups。此前 `prepare_group_storages_graphs` 崩溃已定位为 `Storage` 中 `IdxType` 乘法溢出并修复；旧 GPU cross-edge 的 `71.5GB` resident-cache OOM 已由 `UNG_GPU_X_STREAMING=1` 绕过，strict skip-additional build 完成，Index `945683 ms`、cross-edge generate `697114.7 ms`。但 v1 streaming 的 `pack_q=450338.9 ms` 占主导，因此它只能作为显存可扩展性结果，不能作为建图加速主结果。Amazon 1% x400 gather-Q full-quality A/B 暴露了更强的 pruning tradeoff：heavy prune 的 L5000 recall 只低 `0.0017`，但 Index 比 CPU 慢；light512 将 Index 从历史 CPU `37807.6 ms` 降到 `30826.5 ms`，但 L5000 recall 从 `0.9660` 降到 `0.9579`。新的 reverse-tail + repair A/B 将 x400 L5000 recall 提升到 `0.9659`，接近同脚本 CPU `0.965`，Index `33044.9 ms` 相对同脚本 CPU `36257.3 ms` 为约 `1.10x`。因此本文最终主张是 cross-edge fused/universal route 作为稳定贡献，group graph 采用 conservative adaptive route、full-quality packed exact-anchor 与 reverse-tail/GNN 的自适应 GPU router；查询阶段的 correct-cover 是互补优化，但仍需接入正式 `search_UNG_index` 后复测冗余入口组对图搜索的影响，而不是无条件替代所有 workload。

## 1. 引言

向量数据库和推荐系统中的过滤近邻搜索通常需要在向量相似度之外满足标签、属性或层级条件。与无过滤 ANN 不同，过滤查询会动态约束候选集合，导致索引结构既要保留局部向量可导航性，又要在不同标签组合之间建立可达路径。UNG 的构建过程正是围绕这个目标展开：它首先按照标签集合构造 group，再为每个 group 构建局部 proximity graph，随后构建标签导航图和覆盖关系，最后为相邻/父子 group 生成 cross-group edges。

这类构建任务在 GPU 上并不天然容易。第一，跨组边不是单个大矩阵乘法，而是数千个不同 `(nq, nx)` 组合的小到中等规模 topK 搜索。逐组调用 cuBLAS 或 cuVS 会产生大量 pack、H2D、kernel launch、D2H 和 host merge 开销。第二，组内图构建存在显著长尾。小组数量多但单组计算量小，直接 GPU 化可能得不偿失；中大组适合 GPU 构建，但图质量又受到候选生成和剪枝策略影响。第三，UNG 构建是端到端系统任务，局部 kernel 更快不一定转化为 Index time 更快，`prepare_all`、fallback 策略和写回路径都必须纳入度量。

本文的核心观点是：过滤向量索引构建和查询都需要 workload-aware 的 GPU 化，而不是把每个局部任务简单映射到现有 GPU 库。我们围绕 UNG 的构建瓶颈和查询入口组瓶颈设计了对应方法：

- 对 cross-edge，使用 grouped fused topK 把大量 group topK 搜索组织成批处理 GPU workload，并让 kernel 直接通过全局 query id 访问常驻 base vectors，避免重复展开 query。
- 对 group graph，把 FastGrnndCuda 作为可选 backend，而不是无条件替代 CPU Vamana；通过 bucket 实验决定哪些 group size 进入 GPU，哪些回退 CPU。
- 对 query entry group selection，把 CPU exact-minimal 的入口组剪枝改写为 GPU correct-cover 集合运算，在保证不漏候选 group 的前提下允许少量冗余入口。
- 对端到端系统，保留 CPU fallback、complete fallback、GPU fused 和 gather-Q/direct-qid 等配置边界，使不同 workload scale 下可以选择稳定且高收益的路径。

本文贡献如下：

1. **Grouped fused cross-edge topK。** 我们将 UNG 跨组边构建抽象为大量 group-level exact topK，并设计常驻向量、direct-qid、TF32 WMMA、安全写回和 group descriptor 调度，使 Amazon 1% x100 cross-edge 相对 CPU Vamana 达到 `22.07x`。
2. **反向候选增强局部 RNG 剪枝与自适应剪枝强度。** 我们提出 FastGrnndCuda，用轻量 reverse-augmented local pruning 替换 Tagore 的重 prune/refine 路径；同时针对 `nx≈200` 暴露出的重 prune 失败，加入 diversified light prune 中等组路径，将 x200 group graph 从 `26620.6 ms` 降到 `6706.36 ms`，并保留接近 CPU Vamana 的高 `Lsearch` recall。
3. **面向过滤索引构建的系统级实验分析。** 我们系统比较 CPU Vamana、CPU exact scan、cuVS per-group、SGEMM+topK、old fused、final fused、complete fallback 和 CPU fallback，明确给出不同 scale 下的收益、瓶颈和不能夸大的边界。
4. **覆盖正确的 GPU 查询入口组选择。** 我们将 `get_min_super_sets_debug()` 的高成本候选 scan/minimal prune 拆成 bitset intersection、frontier compaction、descendant OR 和 uncovered fallback，在 `nq=10240` 上达到 `19.85x` 入口组阶段加速，并给出后续图查询不变时的端到端 latency 上界分析。

## 2. 背景与问题定义

### 2.1 UNG 构建流程

给定向量集合 `X`、标签集合和标签层级，UNG 构建大致包含以下阶段：

1. **group partitioning**：按标签组合划分 group，并建立 group 到向量区间的映射。
2. **group graph construction**：为每个 group 内向量构建局部 proximity graph。
3. **vector-attribute bipartite graph**：建立向量和属性/标签之间的二分图。
4. **label navigation graph (LNG)**：在标签组合之间建立导航边。
5. **descendants / coverage**：计算层级覆盖和后续查询所需的覆盖关系。
6. **cross-edge construction**：在 group 之间生成跨组边，补充过滤查询中跨标签区域的可达性。

在当前实现中，cross-edge 的主要模式是：对每个目标 group `G_x`，从其 LNG 入邻居或父侧相关 group 中收集 query 向量集合 `Q`，在目标 group 的向量集合 `X_g` 中为每个 query 执行 topK，生成跨组连接。因此单个任务形态可表示为：

```text
Input:  Q_g in R^{nq_g x d}, X_g in R^{nx_g x d}
Output: topK nearest neighbors in X_g for each q in Q_g
```

真实 Amazon 标签分布下，`nq_g` 和 `nx_g` 高度不均。大量小组不适合逐个 GPU 调用；中大组需要足够并行度；超大 scale 又会暴露内存容量和写回路径限制。

### 2.2 Baseline 方法

本文比较的基线包括：

| 类别 | 方法 | 角色 |
|---|---|---|
| 原始 CPU | CPU Vamana cross-edge | 原始慢路径，近似搜索生成跨组边 |
| 强 CPU | CPU exact scan | 对每个 group pair 精确扫描 topK |
| GPU 库 | cuVS per group | 逐 group 调用 GPU brute-force |
| GPU 库 | SGEMM + separate topK | GEMM 与 topK 分离的强基线 |
| 早期自研 | old fused | direct-qid 前的 grouped fused topK |
| 本文 | final fused | direct-qid + TF32 group + safe writeback |
| 组内图 | CPU Vamana group graph | 质量 baseline |
| 组内图 | Tagore / FastGrnndCuda | GPU group graph 构建路径 |

## 3. 方法

### 3.1 Workload-Aware Grouped Fused TopK

直接使用 GPU 库的一个自然方案是对每个 group 调用一次 brute-force search 或 GEMM，然后再执行 topK。但在 UNG 中，group 数量可达数千到数万，且每个 group 的 `(nq, nx)` 分布高度不均。逐 group 调库会产生大量重复成本：

- query vectors 需要反复 pack 到连续内存。
- 每个 group 都会触发 API 调用和 kernel launch。
- topK id/dist 需要逐 group D2H。
- host 端还需要将结果合并回 `SearchQueue` 和 `_graph->neighbors`。

本文的 cross-edge 路径首先把 target groups 批处理，构造 compact descriptors：

```text
desc = (qi, x_offset, nx, group_id)
```

其中 `qi` 指向 flatten 后 query row，`x_offset` 指向目标 group 在全量 base vectors 中的起始位置。kernel 根据 descriptor 完成 dot/L2 计算与 topK 更新。对中小 group，使用 group-desc fused kernel；对满足形状条件的 group，使用 TF32 WMMA group kernel；对超出 direct-qid 稳定范围的规模，回退到 gather-Q fused 路径。

除了当前 target-centric 主路径，我们还实现了一个 source-centric no-lock 设计作为下一阶段候选。原始 cross-edge 语义可以等价地从 target group 的 `in_neighbors[target]` 展开，也可以从 source/query group 的 `out_neighbors[source]` 展开。target-centric 写法会让同一个 qid 在多个 target group 的局部 topK 中重复出现，最终需要 host `SearchQueue` merge 或 GPU global merge；后者在 qid-lock 实验中已被证明会引入 GPU lock contention。source-centric 写法让每个 qid 只由所属 source group 处理一次，kernel 在该 source group 的所有目标 segment 上维护最终 topK，因此天然无锁。当前实现提供 `UNG_GPU_SOURCE_EXACT=1` 和 `UNG_GPU_SOURCE_EXACT_MODE={0,1}`，其中 `0` 是 CUDA-core exact scan，`1` 是实验性 TF32 WMMA + fused topK。最新错误检查和 id-only 输出修复后，可信 CUDA-core source path 在 SIFT30 skip-additional 上 cross `5926.67 ms`、kernel `5525.5 ms`，慢于 target-centric fused `4918.9 ms`；因此本文不把 source-centric 写成 current best，而把它作为 two-stage source grouped GEMM + per-source reduce 的后续方向。

最新主工程路线改为 universal flat double-buffer。它仍保持 target-centric descriptor batching，但默认让所有可覆盖 target group 进入 double-buffer all-DB 路径，并用 flat-id output 和 GPU global merge 降低 `SearchQueue`/per-group container 写回成本。SIFT30 skip-additional 中该路径 cross `3180.63 ms`；Amazon 1% x200 full-quality 中保留 CPU Vamana `additional_edges` 后 cross `2494.21 ms`，L1000/L5000 recall `0.871/0.911`；Amazon 1% x100 full-quality 中 cross `1689.40 ms`，L100/L500/L1000 recall `0.826/0.868/0.891`，但 Index time 变慢。因此它在本文中被写成 x200 正结果和 x100 边界证据，而不是无条件端到端加速。

### 3.2 Direct-QID 常驻数据路径

传统 GPU brute-force 路径会先把所有 query vectors 展开到 `Q` buffer，再计算 `Q * X^T`。这对 UNG cross-edge 是低效的，因为 query 本身来自同一个 base vector 集合。本文将全量 base vectors 一次性上传到 `g_d_all_X`，并计算全量 `g_d_all_norm`。之后每个 query 只需要传输全局 qid：

```text
q = g_d_all_X[qid]
q_norm = g_d_all_norm[qid]
```

这个设计减少了 query H2D 和 query norm kernel，并降低了大批量 group search 的 host pack 成本。代价是每次 `build_UNG_index` 进程需要一次 `prepare_all`：

```text
prepare_all = host registration + full-data H2D + full-data norm
```

该成本不是每个 group 重复发生，但单次构建必须计入端到端时间。实验中 x100 的 `prepare_all` 为 `218.714 ms`，cross total 为 `848.9 ms`，resident cross 为 `630.186 ms`。

### 3.3 TF32 WMMA Group Kernel 与 Fused TopK

对适合 Tensor Core 的 group shape，本文实现 TF32 WMMA fused topK kernel。kernel 使用 16x16 MMA tile 计算 query tile 与目标 group tile 的 dot product，并在同一 kernel 内完成 topK 更新，避免写出完整 dense distance matrix。

相比 SGEMM + separate topK，该设计的目标不是在所有 dense GEMM 形状上超过 cuBLAS，而是减少 UNG workload 中的中间结果写出、独立 topK kernel 和 per-group 调度开销。当前实现还包括：

- `UNG_TF32_GROUP_2D=1`：以 `grid.y=group`、`grid.x=query tile` 的方式调度。
- `UNG_GPU_ID_ONLY_WRITEBACK=1`：只 D2H topK id，因为最终 graph merge 只需要 id。
- `UNG_BUCKET_GROUP_FUSED=1`：按 group shape 路由到 small/medium/TF32 路径。

### 3.4 FastGrnndCuda: 反向候选增强局部剪枝

CPU Vamana 是 UNG group graph 的质量 baseline，但在大组上构建慢。Tagore 提供了 GPU GNN-Descent 和 GPU pruning 能力，但原始 prune 路径在 UNG group 场景中较重。本文提出 FastGrnndCuda，核心是把 GPU group graph 构建拆成：

1. 使用 Tagore/GNN-Descent 生成 forward 候选。
2. 从“指向当前点”的 reverse candidates 中采样补充候选池。
3. 使用固定预算 local RNG-style occlusion 剪枝，输出每点邻居。

与 GRNND/RNN-Descent 的区别是：本文没有复现完整 RNN-Descent 迭代框架，也没有把 RNG pruning 融入整个 NN-Descent 迭代过程；本文复用已有 GNN 候选生成，将 reverse augmentation 和 local RNG pruning 作为面向 UNG group graph 的轻量后端。这降低了工程接入风险，也让方法能直接在 `UniNavGraph` 构建流程中通过环境变量 A/B。

与 Tagore 的区别是：本文不使用 Tagore 的重 `select_path + filter_reverse` prune，而是用固定预算候选池控制 prune 时间，并显式引入 reverse candidates 改善低 search budget 下的 recall。

针对 x400 light prune 的质量缺口，我们进一步加入一个默认关闭的 light reverse-tail 路径。该路径不恢复 heavy RNG occlusion，而是在 light prune 内构造固定预算 sampled reverse candidates，并只把少量反向候选填入尾部名额。它由 `UNG_FAST_GRNND_LIGHT_REVERSE_CAP/SLOTS/FORWARD_CAP` 控制，用于验证“低成本反向尾部多样性”是否能在保持 light prune 速度的同时缩小 x400 recall gap。x400 A/B 显示，degree repair 单独能显著降低 low-degree 比例，但 L5000 只从 `0.9574` 提到 `0.957767` 且 Index 变慢；reverse-tail + repair 将 L5000 提到 `0.9659`，接近同脚本 CPU `0.965`，同时 Index `33044.9 ms` 相对同脚本 CPU `36257.3 ms` 为约 `1.10x`。

2026-06-02 新增两个面向 group graph 泛化性的实现改动。第一，light prune 默认开启 `UNG_FAST_GRNND_REPAIR_DEGREE=1`，在候选不足时用确定性环形边补足出度，直接针对 x400 结构诊断中的 zero/low intra-degree 和 weak-component 分裂。第二，Tagore/FastGrnnd batch 默认开启 `UNG_TAGORE_COMPACT_D2H=1`，prune 后把 `k` 宽候选表压缩为 `final_degree+1` 宽再写回 host；同时 batch graph fill 改成 OpenMP 并行回填。这些改动不改变 UNG 查询接口，目标是降低 D2H/host fill 外围开销并修复 light prune 的低出度结构问题。当前实验证明 repair 需要与 reverse-tail 配合使用，compact-D2H 在 x400+reserve 设置下是负/不稳定 ablation：compact-on 没有稳定 build-time 收益，因此不能写成主贡献。

### 3.5 Hybrid Fallback 策略

不是所有 group 都适合 GPU group graph。小组 GPU kernel 启动和 fallback 成本可能超过 CPU Vamana；部分中间范围 group 的 complete fallback 或 GPU prune 也可能不划算。因此当前最快端到端配置采用混合策略：

- `nx <= complete_threshold`：complete graph 或轻量 fallback。
- `nx < UNG_TAGORE_MIN_GROUP_SIZE`：默认 CPU Vamana fallback。
- `nx >= UNG_TAGORE_MIN_GROUP_SIZE`：FastGrnndCuda。

这是一项工程策略，不应和 FastGrnndCuda 方法本身混为一个算法贡献。论文中可将它作为 system policy 或 workload-aware routing。

### 3.5.1 AdaptiveCuda Conservative Route

为了避免把多个环境变量拼接成不可复现的手工策略，我们新增 `UNG_GROUP_GRAPH_IMPL=4` 的 `adaptive_cuda` 后端。该后端默认使用三段式 route：

```text
nx < 128        -> CPU fallback
128 <= nx <=512 -> packed exact-anchor
nx > 512        -> FastGrnndCuda
```

同时，CPU fallback 与 GPU batch 并发执行，exact packed 结果在 host fill 时直接从 packed graph buffer 按 offset 读取，减少 D2H 后按组拆分和逐边去重的开销。这个 route 的目标不是完全去掉 CPU，而是在 full-quality 口径下先取得质量安全的 build 改善。

Amazon 1% x100 full-quality 结果显示，conservative adaptive route 的 Index 为 `5450 ms`、group graph 为 `2773.35 ms`，L100/L500/L1000 recall 为 `0.829/0.870/0.895`；历史 FastGrnnd full-quality 为 `6370.67 ms`、`3216.21 ms`、`0.828/0.871/0.896`。这说明 conservative route 可以作为 x100 上的正向替代候选。

我们也测试了更激进的小组 bounded fallback：skip-additional build 中 group graph 可降到 `1953.07 ms`，但 L100/L500/L1000 recall 只有 `0.816/0.823/0.827`。因此，小组 CPU Vamana 仍是重要瓶颈，但简单环形强连通图不能作为 full-quality 主线；它只能作为负结果 ablation，说明未来需要设计保持小组可导航性的轻量构图。

### 3.6 Mixed Exact/GNN Router：初步验证的调度优化

早期 batched exact kNN 在 x200 上曾表现为低 recall：pure local L5000 `0.8199`，anchor-tail 约 `0.829`。后续结构诊断表明，这个结论不能归因于组内 exact 图本身：旧 exact-anchor 和新 packed exact 的组内图完全一致，都是每点 32 条组内边、低出度为 `0`、所有 group 的最大 WCC 为 `1`；差异集中在 cross edges，旧 run 为 `1,038,000`，CPU/new packed 都是 `1,093,224`。因此旧结果应作为 cross/additional 口径混杂的 ablation，而不是“exact graph 不可行”的证明。

更可靠的结论来自 full-quality packed exact-anchor router。旧实现只有在“整个 batch 的所有 group 都小于阈值”时才会启用 batched exact；一旦混入大组，小/中组也被迫走逐组 GNN 路径。当前实现已改为按 `UNG_FAST_GRNND_BATCH_EXACT_NX` 将请求拆成 exact batch 和 GNN batch，再按原 group 顺序合并结果。

我们已在 Amazon 1% x200 coverage-query 上完成一轮初步 A/B。该实验使用 `NUM_REPEATS=1`，并启用 `UNG_TAGORE_BATCH_STREAMS=2`、`UNG_TAGORE_FILL_FAST=1` 和 `UNG_TAGORE_FILL_THREADS=16` 以减少 many-group 外围开销。结果显示，`UNG_FAST_GRNND_BATCH_EXACT_NX=256/512` 没有复现 pure exact kNN 的低 recall 问题：`router_exact_nx256` 的 L100/L500/L1000 recall 为 `0.8229/0.8510/0.8690`，CPU Vamana 为 `0.8230/0.8490/0.8690`，non-exact router 为 `0.8180/0.8463/0.8661`。同时 `router_exact_nx256` 的 group graph 为 `5770.88 ms`，相对 non-exact router `8606.52 ms` 为 `1.49x`。

随后我们把 exact batch 的 host 结果改为 packed graph buffer，由 `UniNavGraph` fill 阶段按 offset 直接读取，避免 D2H 后先拆成每组 `result.graph` 再复制到 `_group_graphs`。这个改动不改变 exact graph 语义，但显著降低外围成本。x200 `UNG_FAST_GRNND_BATCH_EXACT_NX=4096` 从旧 scatter run 的 group `4929.54 ms` 降到 `3827.28 ms`，Index `11043.5 ms`，repeat=3 L1000/L5000 `0.869/0.908`；x400 同配置 group `11588.4 ms`，Index `28436.3 ms`，repeat=3 L1000/L5000 `0.945/0.967`。

进一步的 x200 direct-H2D A/B 表明，packed exact-anchor 的性能上限仍被输入 pack 和输出图物化限制。旧 exact batch 会把已经按 group 连续重排的 base vectors 再复制到临时 packed host buffer，并在 CPU 上生成 point-to-group lookup；该路径 group `8999.26 ms`，其中 `pack=4140.61 ms`、GPU exact kernel `931.72 ms`、fill `2849.50 ms`。当前默认路径在连续 host run 上直接 H2D，并在 GPU 上生成 lookup；手动 `fill16` A/B 将 group 降到 `3204.88 ms`，`pack` 降到 `0.04 ms`，GPU kernel 仍约 `928.67 ms`，L1000 recall 不变。这个实验说明旧性能问题不是 exact kernel 本身，而是 CPU-compatible output boundary。审稿角度需要强调：这是当前最强的 group-graph 替代候选，但仍需要 x100、10%x40 和真实多标签数据确认泛化边界；进一步大幅优化需要 flat adjacency/CSR，不能把当前 `std::vector` 图回填写成已经解决。

我们也验证了一个更激进但最终无效的 CUDA kernel 改写：`UNG_FAST_EXACT_WARP_KERNEL=1` 让每个 warp 负责一个 source 点，试图减少原 block-per-source exact kernel 中每个候选 dst 的 block-wide reduce 和同步。Amazon 1% x200 router exact256 full-quality A/B 显示，旧 block kernel 的 group graph 为 `4031.36 ms`、exact kernel `841.136 ms`，而 warp kernel 变为 `4334.28 ms`、`987.27 ms`，L1000 recall 同为 `0.867`。原因是当前维度和组大小下，距离计算从 128 threads 降到 32 lanes 后并行度损失大于同步减少。`UNG_TAGORE_FILL_THREADS` 扫描也显示，默认 16 线程 fill `1682.11 ms` 优于 8 线程 `2113.86 ms` 和 64 线程 `2321.96 ms`，说明 fill 受 host allocator 和内存写竞争限制。这个负结果进一步支持本文的系统判断：下一步大幅优化应重构图输出边界，而不是继续微调 exact kernel 或 fill 线程。

该结果说明 exact 可以作为小/中组的调度分支，而不是独立替代全部组内图；但是它仍不是已完成的最终质量 router。投稿前仍应运行 repeat=3，并在更多 group-size 分布上确认：

```bash
scripts/benchmarks/run_group_graph_router_ab.sh
python3 tools/benchmarks/summarize_group_graph_router_ab.py <out_root>
```

只有当 `UNG_FAST_GRNND_BATCH_EXACT_NX=128/256/512` 等阈值在降低 `group_ms/index_ms` 的同时，保持 recall 接近 `router_exact_nx0` 和 CPU Vamana，才可将该 router 写入主方法；否则它只能作为 negative 或 partial ablation。当前 x200 结果支持把它写成初步正向 ablation，而不能写成普遍替代已经解决。

### 3.7 GPU Correct-Cover 查询入口组选择

UNG 查询在进入图搜索前，需要根据 query labels 找到一批入口 group。原始 CPU 路径调用 `get_min_super_sets_debug()`，先找出所有满足 `query_labels subset labels(group)` 的候选 group，再做 minimal prune，尽量去掉父子冗余。这个 exact-minimal 输出有利于减少后续图搜索入口，但在 many-group workload 中会成为 query latency 的大头。

本文把入口组选择改写为一个 coverage-correct 的 GPU 批处理集合运算。对一个 query，定义：

```text
C       = {g | query_labels subset labels(g)}
F_raw   = {g in C | |query_labels| <= |labels(g)| <= |query_labels| + delta}
F       = first_cap(F_raw)
Covered = union(descendants(f) for f in F)
Output  = F union (C - Covered)
```

其中 `delta` 控制 frontier 的 label-size 窗口，`cap` 控制每个 query 最多拿多少个 frontier group 做 descendant OR。`cap` 不限制最终输出 group 数；如果某个候选没有被 frontier descendants 覆盖，它会落入 `C - Covered` 并被直接输出。因此该方法保证覆盖所有实际存在的候选 group，但不保证和 CPU exact minimal 输出完全一致。

这个正确性不依赖“query label 的精确组合是否存在对应 group”。如果 `{a,b}` 没有 group，但 `{a,b,c}`、`{a,b,d}` 等 superset group 实际存在，它们仍属于 `C`；当 frontier 为空或被 `cap` 截断时，最坏输出就是更多 `C - Covered`，不会漏掉候选。真正的前提是 descendant 表必须完整覆盖 label-superset 关系。benchmark 中 `build_descendant_bitsets_from_labels()` 按 label 包含关系直接构造 descendants；生产 LNG 路径则需要保证 BFS descendants 至少覆盖所有真实 label-superset group。

实现上，`query_entry_group_bench.cu` 使用 dense bitset 表示 label 到 group、group size bucket、candidate、frontier 和 output。GPU pipeline 为：

```text
candidate_frontier_kernel:
  label bitset intersection 得到 C，并按 size window 得到 F_raw

compact_frontier_ids_kernel:
  把 F_raw 截断到 cap，compact 为 frontier id list F

descendant_cover_list_kernel:
  对 F 中 group 的 descendant bitset 做 OR，得到 Covered

cover_frontier_select_kernel:
  输出 F union (C - Covered)
```

关键优化是 list-driven coverage：先把 frontier bitset compact 成 id list，再只对这些 id 做 descendant OR。旧实现对每个 `(query, bitset word)` 扫完整 frontier bitset，`delta=1` 路径约 `762.92 ms`；list-driven 后在 Amazon 100%x40、`nq=10240` 上降到 `60.72 ms`。

该方法和后续图查询的关系需要保守表述。它只替换 `query labels -> entry group ids/bitset`，后续仍要执行：

```text
entry group ids -> get_entry_points_given_group_id() -> iterate_to_fixed_point()
```

如果 GPU correct-cover 输出的入口组比 CPU exact minimal 多，后续 entry point materialization 和图搜索可能变慢。因此本文将现有结果写成入口组阶段加速和端到端上界分析；正式端到端 query speedup 仍需把 `gpu_cover_frontier` 接入 `search_UNG_index` 后，在同一 index/query/Lsearch 上复测。

## 4. 实验设置

### 4.1 硬件与数据集

实验环境：

| 项目 | 配置 |
|---|---|
| GPU | NVIDIA RTX A6000 |
| 线程 | `--num_threads 128` |
| 距离 | L2 |
| UNG 参数 | `max_degree=32`, `Lbuild=100`, `alpha=1.2`, `num_cross_edges=6` |

主要数据集为 Amazon sampled + jitter repeat 系列：

| 数据集 | 点数 | 组数 |
|---|---:|---:|
| Amazon 1% x40 | 240,960 | 5,666 |
| Amazon 1% x100 | 602,400 | 5,666 |
| Amazon 1% x200 | 1,204,800 | 5,666 |
| Amazon 1% x400 | 2,409,600 | 5,666 |
| Amazon 10% x40 | 2,409,800 | 53,840 |

其中 x40/x100/x200/x400 表示在保留标签结构的前提下复制并 jitter 向量坐标，用于控制 group size 和计算负载。

### 4.2 当前优化配置

cross-edge 使用：

```bash
UNG_CROSS_EDGE_IMPL=1
UNG_ADDITIONAL_EDGES_IMPL=0
UNG_GPU_TOPK_IMPL=3
```

其中 `UNG_ADDITIONAL_EDGES_IMPL=0` 表示保留 CPU Vamana additional edges，是本文端到端 recall 表使用的 full-quality 口径。`UNG_ADDITIONAL_EDGES_IMPL=1` 表示跳过 additional edges，只能用于阶段性能 ablation；原始 Amazon PF 已显示 skip 后 recall 约 `0.61`，不能作为完整质量配置。

当前 `FusedGroupTopk` 默认启用：

```bash
UNG_DIRECT_QID_FUSED=1
UNG_DIRECT_QID_ALL_FUSED=1
UNG_GPU_ID_ONLY_WRITEBACK=1
UNG_TF32_GROUP_2D=1
UNG_BUCKET_GROUP_FUSED=1
```

group graph 使用：

```bash
UNG_GROUP_GRAPH_IMPL=3
UNG_TAGORE_K=64
UNG_TAGORE_ITER=4
UNG_TAGORE_M=64
```

阈值为：x40/x200/x400 使用 `UNG_TAGORE_MIN_GROUP_SIZE=256`，x100 使用 `128`。

当报告 `adaptive_cuda` 结果时，组内图使用 `UNG_GROUP_GRAPH_IMPL=4`，默认 `nx<128` 保留 CPU fallback，`128<=nx<=512` 走 packed exact-anchor，`nx>512` 走 FastGrnndCuda，并开启 CPU fallback 与 GPU batch 并发。该配置是 workload-aware route，不是 CPU Vamana 的逐边等价替代。

## 5. 实验结果

### 5.1 Cross-Edge 方法对比

Amazon 1% x100 cross-edge 结果如下：

该表可由 `tools/benchmarks/summarize_cross_baselines.py` 从保存的 run 目录复现，当前输出在 `/home/graphdb/fv_runs/baseline_fairness_x100_20260602/cross_baselines.md`。CPU exact baseline 使用 `128` build threads；测试机 CPU 为 Intel Xeon Platinum 8360Y，2 sockets，36 cores/socket，144 logical CPUs。

| 方法 | cross total | H2D | kernel/build | D2H | Index | 说明 |
|---|---:|---:|---:|---:|---:|---|
| CPU Vamana cross | `18735.8 ms` | - | - | - | `23150 ms` | 原始慢路径 |
| CPU exact scan | `2389.8 ms` | - | - | - | `6995 ms` | 强 CPU baseline |
| cuVS per group | `3730.5 ms` | `592.1` | `821.8` | `43.5` | `14353 ms` | 逐 group 调库 |
| SGEMM + separate topK | `1395.9 ms` | `108.4` | `541.6` | `9.9` | `7452 ms` | 强 GPU library baseline |
| old fused baseline | `1268.5 ms` | `103.9` | `528.7` | `7.8` | `7388 ms` | direct-qid 前 |
| final direct-qid fused, CPU group | `1313.1 ms` | `76.0` | `227.0` | `5.8` | `14053 ms` | 只替换 cross-edge |
| final direct-qid fused, mixed group | `848.9 ms` | `74.1` | `232.8` | `3.6` | `6630 ms` | 最新 full optimized |

相对 `848.9 ms`：

| baseline | speedup |
|---|---:|
| CPU Vamana cross | `22.07x` |
| CPU exact scan | `2.81x` |
| cuVS per group | `4.39x` |
| SGEMM + topK | `1.64x` |

结果说明了两个事实。第一，相对原始 CPU Vamana cross-edge，GPU fused topK 收益非常明显。第二，相对 SGEMM + separate topK 的收益不是数量级，而是来自 workload-level fusion 和外围开销降低；因此不能声称本文 kernel 在所有 dense GEMM 上超过 cuBLAS。

### 5.1.1 Source-Centric No-Lock Cross-Edge Smoke Test

为回答“输出边界能否比 qid-lock global merge 更系统地解决”，我们新增 source-centric no-lock cross-edge 路径。它不在 target group 上生成多个局部 topK 再合并，而是按 source group 的 `out_neighbors` 一次性维护每个 qid 的最终 topK。该设计避免 per-qid lock，也避免把同一个 qid 的多个 target group 局部结果交给 host `SearchQueue` 再归并。

SIFT30 smoke test 使用 `additional_edges=skip`，因此只能作为 cross-edge 后端性能和工程可行性证据，不能替代 full-quality 主表。

| 路径 | artifact | cross total | prepare/H2D | kernel | D2H | writeback | 备注 |
|---|---|---:|---:|---:|---:|---:|---|
| source-centric CUDA-core exact, legacy | `/home/graphdb/FilterVectorResultsRefactor/organized_benchmarks/ung_cross_edge_fused_20260602_052124` | `5981.0 ms` | `20.3 ms` | `5524.0 ms` | `1.9 ms` | `43.0 ms` | 输出边界低，但 kernel 利用率差 |
| target-centric fused 当前路径 | `/home/graphdb/FilterVectorResultsRefactor/organized_benchmarks/ung_cross_edge_fused_20260602_052209` | `4918.9 ms` | `77.4 ms` | `2189.9 ms` | `35.7 ms` | chunk writeback 约 `43.8 ms` | 当前对照 |
| source-centric TF32 WMMA, legacy | `/home/graphdb/FilterVectorResultsRefactor/organized_benchmarks/ung_cross_edge_fused_20260602_052750` | `3021.9 ms` | `23.2 ms` | `1402.9 ms` | `2.9 ms` | `63.1 ms` | 旧 smoke；缺少当前错误检查边界 |
| source-centric TF32 WMMA + source-block padding, legacy | `/home/graphdb/fv_runs/source_wmma_padded_sift30_skipadd_20260602_053546` | `1927.7 ms` | `23.2 ms` | `1418.5 ms` | `1.9 ms` | `26.8 ms` | 旧 smoke；只能作为设计线索 |
| source-centric CUDA-core id-only, current checked | `/home/graphdb/FilterVectorResultsRefactor/organized_benchmarks/ung_cross_edge_fused_20260602_091046` | `5926.7 ms` | `23.5 ms` | `5525.5 ms` | `1.0 ms` | `13.3 ms` | 修复 null-distance 写和 stream error check 后的可信负结果 |

该结果说明：source-centric 重排本身能消除 qid 冲突和大部分输出边界，但当前单阶段 source kernel 的候选扫描并行度不足。旧 TF32 WMMA smoke 显示 Tensor Core 化可能降低 kernel time，但在补齐同等错误检查、id-only 输出边界和 TF32/FP32 topK 一致性前，不能作为主表加速证据。后续若要把 source-centric 推成普遍替代，需要 two-stage source grouped GEMM + per-source reduce：先并行生成 `(source query tile, target segment tile)` partial topK，再按 source query 在 GPU 上归并。

随后补的 Amazon 1% x200 full-quality 最小闭环给出更克制的结论。CPU Vamana group、`additional_edges=cpu_vamana` 下，source-centric w8 cross 为 `4693.4 ms`、kernel `3224.6 ms`、L1000/L5000 repeat=3 为 `0.871/0.911`；source-centric w4 cross 为 `4803.07 ms`、kernel `3091.9 ms`、repeat=3 同为 `0.871/0.911`；同机 target-centric fused cross 为 `3900.81 ms`、kernel `2095.8 ms`、repeat=3 为 `0.858/0.9068`。这些历史 run 显示 source-centric 可以接入 Amazon full-quality build/search，但在新错误检查复核前不能作为性能主张；即使按历史口径，它也不是 current best。进一步的结构诊断显示，target-current 主要问题不是组内图，而是少了 `4404` 条 cross edges。我们随后修复 target-centric double-buffer 的 all-or-nothing 回退：当 batch 中只有少数超大 target group 不满足 `DB_LARGE_MAX_NX` 时，supported groups 仍走 double-buffer，unsupported groups 单独回退。x200 full-quality 中该 partial routing 让 `5173/5195` 个 target groups 进入 double-buffer，cross-edge 降到 `2657.23 ms`、kernel 降到 `1636.0 ms`，结构诊断 cross edges 为 `1,093,032`，只比 source/CPU 对齐口径 `1,093,224` 少 `192`；repeat=3 L1000/L5000 为 `0.866/0.907`。这说明当前最有效的近期路线是提高 target-centric fast-path 覆盖率，而 source-centric 仍应作为下一阶段 no-lock 替代候选；投稿前还需要更好的 source WMMA tiling、additional_edges GPU/full-quality A/B 和 TF32/FP32 topK 一致性检查。

### 5.2 Cross-Edge 时间组成

Amazon 1% x100 最新 full optimized：

| 项目 | 时间 |
|---|---:|
| cross total | `848.9 ms` |
| prepare_all | `218.714 ms` |
| resident cross | `630.186 ms` |
| GPU H2D events | `74.1 ms` |
| GPU kernel events | `232.8 ms` |
| GPU D2H events | `3.6 ms` |

`prepare_all` 在不同运行中波动较大，尤其 host registration。此前同路径 CPU-group 重测中 x100 `prepare_all=540.843 ms`，本轮 full optimized 为 `218.714 ms`。因此本文报告 cross total 和 resident cross 两个口径。

### 5.3 End-to-End Scale

当前最快工程配置：FastGrnndCuda 处理大组，小组默认 CPU Vamana fallback；x400 因 direct-qid 不稳定，使用稳定 gather-Q fused cross-edge。

| 数据集 | group graph | vector_attr | LNG | desc | coverage | cross | Index |
|---|---:|---:|---:|---:|---:|---:|---:|
| 1% x40 | `868.5` | `126.4` | `98.3` | `10.5` | `57.9` | `389.8` | `2375` |
| 1% x100 | `3373.2` | `209.9` | `43.7` | `3.9` | `12.4` | `848.9` | `6630` |
| 1% x200 | `8630.8` | `410.2` | `37.4` | `2.0` | `18.6` | `3679.5` | `16426` |
| 1% x400 | `31245.6` | `777.0` | `65.0` | `19.3` | `211.0` | `12462.8` | `50562` |
| 100% x40 | `17785.4` | `9170.56` | `10585.3` | `31.3` | `3673.6` | `697114.7` | `X-streaming v1 completed, not speedup result` |

相对旧 CPU Vamana group + GPU fused cross-edge baseline：

| 数据集 | baseline Index | 当前 Index | Index speedup | baseline group | 当前 group | group speedup |
|---|---:|---:|---:|---:|---:|---:|
| 1% x40 | `2484` | `2375` | `1.05x` | `891.0` | `868.5` | `1.03x` |
| 1% x100 | `16824` | `6630` | `2.54x` | `11549.9` | `3373.2` | `3.42x` |
| 1% x200 | `34603` | `16426` | `2.11x` | `26231.4` | `8630.8` | `3.04x` |
| 1% x400 | `70616` | `50562` | `1.40x` | `54043.2` | `31245.6` | `1.73x` |
| 100% x40 | `NA` | `NA` | `NA` | `NA` | `NA` | `NA` |

这些 build-scale 结果显示，x100 的混合策略收益较明显；x200 在该表中看起来有收益，但后续 coverage-query 端到端 A/B 证明该结论不能直接外推到完整质量配置。x40 中可 GPU 化的大组太少；x400 中 group graph prune 和稳定 gather-Q cross-edge 成为主要瓶颈。新增 `100%x40` 构建口径已经生成数据并完成 optimized skip-additional build。此前 `prepare_group_storages_graphs` 崩溃已修复；旧 cross-edge full resident vector cache 需要 `71,530,536,960` bytes，在 48GB A6000 上 OOM。新增 `UNG_GPU_X_STREAMING=1` 后，`23,284,680` 点、`482,387` groups 的 strict build 可以完成：Index `945683 ms`，cross-edge generate `697114.7 ms`，其中 `pack_q=450338.9 ms`、GPU kernel `72749.8 ms`。因此 `100%x40` 必须写成 scalability/boundary result：显存边界已被 X-side streaming 绕过，但 v1 不能进入建图加速主表，不能用 10%x40 线性外推。

### 5.3.1 Amazon 1% x100 Coverage-Query End-to-End A/B

为补齐 x100 repeat-scale 质量闭环，我们从 `/tmp/fv_amazon_1pct_x100_jitter_nonempty` 生成 `1000` 个 coverage query，profile 为 medium/broad/narrow `477/331/192`，matched groups avg/p50/p95 为 `311.50/39.5/1095.0`，matched points avg/p50/p95 为 `33211.90/4100.0/123200.0`。GT exact compute 为 `5275 ms`。

full-quality A/B 固定 GPU fused cross-edge 和 CPU additional_edges，只切换 group graph：

```text
/home/graphdb/fv_runs/end_to_end_recall_ab_20260602_041311
```

| Variant | Index | Group | Cross | L100 | L500 | L1000 | P95 L100/L500/L1000 |
|---|---:|---:|---:|---:|---:|---:|---:|
| CPU Vamana group | `6863.47` | `3303.26` | `1732.42` | `0.825` | `0.868` | `0.890667` | `162.1 / 163.1 / 162.4` |
| FastGrnndCuda + CPU fallback | `6370.67` | `3216.21` | `1871.51` | `0.828` | `0.871` | `0.896` | `174.5 / 159.3 / 163.5` |

该实验回答了 x100 的质量问题：FastGrnndCuda 没有降低 recall，Index 约 `1.08x`，group graph 约 `1.03x`。但它不是强 group speedup 证据，因为 `5538/5666` 个组仍走 CPU fallback，只有 `128` 个组进入 FastGrnndCuda；x100 的主要论文价值是 repeat-scale recall 闭环，而不是证明组内图 kernel 大幅加速。

### 5.4 Group Graph 组成

当前最快 fallback 策略下：

| 数据集 | Tagore groups | CPU fallback groups | complete fallback groups | direct build | fallback wall | GNN | prune | group total |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1% x40 | `16` | `112` | `5538` | `229.4` | `636.6` | `14.5` | `38.8` | `868.5` |
| 1% x100 | `128` | `5538` | `0` | `540.0` | `2822.2` | `106.3` | `265.8` | `3373.2` |
| 1% x200 | `128` | `5538` | `0` | `961.1` | `7648.4` | `174.4` | `557.4` | `8630.8` |
| 1% x400 | `5666` | `0` | `0` | `30812.5` | `11.0` | `4940.1` | `20021.9` | `31245.6` |

x400 中所有 group 都进入 FastGrnndCuda，`prune=20021.9 ms` 是最大单项瓶颈。x100/x200 的最快结果依赖 CPU fallback，因此论文中应将 fallback 策略作为系统工程策略，而不是把所有收益归因给 GPU group graph 算法本身。

### 5.5 单组图质量与构建时间

在 `g1024_nx2048` 单组图质量实验中：

| 方法 | build graph | prune/refine | L20 | L50 | L100 | L200 | L500 |
|---|---:|---:|---:|---:|---:|---:|---:|
| 原始 Tagore | `9348.87 ms` | prune `4531.38 ms` | `0.9355` | `0.9765` | `0.9860` | `0.9915` | `0.9945` |
| FastGrnndCuda 旧版 | `3790.43 ms` | prune `657.63 ms` | `0.9375` | `0.9790` | `0.9870` | `0.9920` | `0.9955` |
| Reverse-augmented Local RNG | `4014.91 ms` | prune `810.08 ms` | `0.9515` | `0.9880` | `0.9930` | `0.9955` | `0.9965` |
| Method3 iter=4 | `8248.08 ms` | prune `4492` + refine `944 ms` | `0.9500` | `0.9880` | `0.9930` | `0.9965` | `0.9985` |
| CPU Vamana | `9265.03 ms` | - | `0.9655` | `0.9910` | `0.9990` | `1.0000` | `1.0000` |

最终方法相对原始 Tagore 同时提升低 budget recall 和构建速度；相对 Method3，在 L20/L50/L100 基本持平的情况下显著降低 build graph 时间；相对 CPU Vamana，速度约 `2.31x`，但高 budget recall 仍有差距。

### 5.6 Complete Fallback Ablation

此前使用 `UNG_TAGORE_FALLBACK_IMPL=0` 的 complete fallback 结果：

| 数据集 | group | cross | Index | 判断 |
|---|---:|---:|---:|---|
| 1% x40 complete fallback | `849.6` | `505.3` | `2442` | 与 CPU fallback 接近 |
| 1% x100 complete fallback | `4533.4` | `1257.4` | `8179` | 慢于 CPU fallback |
| 1% x200 complete fallback | `20021.6` | `2788.9` | `26642` | fallback complete 图过重 |
| 1% x400 complete fallback | `31074.6` | `6722.0` | `44713` | 无 fallback，差异主要来自 cross 配置 |
| 10% x40 complete fallback | `7225.2` | `6201.2` | `23207` | group 数足够多时收益明显 |

这组结果说明 fallback 策略会显著影响端到端表现。complete fallback 对某些中间 group size 不一定合适，CPU Vamana fallback 在 x100/x200 上更快。

### 5.7 投稿前 Reviewer Audit

当前证据已经能支撑 cross-edge fused topK 的性能结论，但要形成投稿级论文，还必须回答下面这些审稿问题。这里把问题保留在草稿中，是为了避免把尚未闭环的证据写成已经证明的结论。

| 审稿问题 | 当前回答 | 投稿前必须补的实验 |
|---|---|---|
| FastGrnndCuda 是否保持最终 filtered-search recall，而不只是单组 graph recall？ | 已在原始 Amazon PF、SIFT30、Amazon 1% x100、Amazon 1% x200、Amazon 10% x40 和 Amazon 1% x400 gather-Q 上部分回答：x100 adaptive full-quality 中 L100/L500/L1000 为 `0.829/0.870/0.895`；10%x40 full-quality 下 Index `1.53x`，且 L50/100/200/500/1000/2000/5000 recall 基本不降；x400 light512 能加速但有 `0.0081` recall gap，reverse-tail+repair 已基本补回；compact-D2H 已补负/不稳定 ablation。 | x400 更多参数扫；补真实多标签。 |
| `22.07x` 是否只是在打弱 CPU baseline？ | 已经列 CPU exact scan、cuVS per-group、SGEMM+topK，但还不够完整。 | 补 CPU exact scan 线程 scaling，并注明 CPU 型号、线程数和 OpenMP 配置。 |
| grouped fused topK 是系统贡献还是普通工程清理？ | 最稳妥的主张是 irregular group topK 的 workload-level fusion。 | 补 resident vectors、direct-qid、id-only writeback、final fused vs SGEMM+topK 的 ablation。 |
| 方法在大规模是否稳定？ | x400 direct-qid 当前不稳定，x400 主表使用 gather-Q fused。 | 修复 x400 strict direct-qid，或正式给出 gather-Q/direct-qid routing policy。 |
| x400 degree repair 是否只是人为补边？ | 已完成 A/B：repair-only 几乎修掉 low-degree 但 Index 变慢且 recall 只小幅提升；reverse-tail+repair 把 L5000 从 `0.9574` 提到 `0.9659`，接近同脚本 CPU `0.965`。compact-D2H 已补负/不稳定 ablation。 | 仍需更多参数扫。 |
| compact-D2H 是否只是工程优化？ | 是系统外围优化，不是图质量贡献；当前 x400 A/B 中 compact-on `Index=39759.1 ms` 慢于 compact-off `33019.7 ms`，D2H 也没有稳定下降。 | 只作为 overhead negative ablation 报告。 |
| mixed exact/GNN router 是否会把低质量 exact 图混进最终方法？ | x200 repeat=1 初步 A/B 显示 `nx256/512` 未复现 pure exact 质量崩塌，L1000 接近 CPU Vamana，并且 group 相对 non-exact router 约 `1.49x/1.47x`。 | 补 repeat=3 和更多数据集；区分 exact router 与多 stream / fast fill 的各自贡献。 |
| Amazon 10% x40 大量 group 场景是否还能扩展？ | 已修复 direct-pageable 和 direct-qid chunk path；full-quality repeat=3 A/B 中 FastGrnndCuda Index `23497.4 ms` vs CPU Vamana `36053.3 ms`，`1.53x`。新增 L50/100/200/500/1000/2000/5000 sweep 中，FastGrnnd recall 相对 CPU 为 `+0.0019/+0.0018/-0.0010/+0.0000/+0.0020/+0.0020/+0.0010`。 | latency 非单调，不能写成稳定 search latency 加速；仍需真实多标签。 |
| 结果是否泛化到 Amazon repeat 之外？ | SIFT30 已有修复后 fused cross + group graph 端到端 A/B。 | 再补 CelebA 或真实多标签数据。 |

对应实验入口是 `scripts/benchmarks/run_end_to_end_recall_ab.sh`。该脚本固定 query set、ground truth、cross-edge 实现和搜索参数，只切换 group graph backend，用于回答第一优先级的端到端 recall 问题。

### 5.8 原始 Amazon PF 的 Group Graph 端到端 A/B

我们补跑了一组原始 Amazon PF 查询 workload。两个 variant 使用同一 query set、GT、fused cross-edge 和 CPU Vamana additional edges；主要差别是 group graph 后端。

| Variant | Index ms | Group ms | Cross ms | L1000 recall | L2000 recall | L5000 recall |
|---|---:|---:|---:|---:|---:|---:|
| CPU Vamana group | 43376.6 | 5776.25 | 20194.7 | 0.912333 | 0.932067 | 0.971767 |
| FastGrnndCuda + CPU fallback | 36732.0 | 1049.92 | 18847.4 | 0.911700 | 0.931500 | 0.971133 |

这是目前第一条 FastGrnndCuda 的真实端到端查询质量证据。它显示 group graph 构建加速 `5.50x`，总 Index time 加速 `1.18x`，三个 Lsearch 上的平均 recall 下降都小于 `0.001`。该实验仍不能替代 Amazon repeat-scale 的正式实验，但它比单组 graph recall 更直接地回应了 reviewer 对 group graph 替换质量的质疑。虽然该 run 缺少 query source-group 文件，代码审查显示该文件只在 `is_ung_more_entry=true` 时用于 oracle group 注入；当前 `search_UNG_index` CLI 已暴露 `--is_ung_more_entry`，但默认仍是 `false`。

同时我们发现 additional edges 对该 workload 的质量非常关键。若设置 `UNG_ADDITIONAL_EDGES_IMPL=1` 跳过补边，CPU Vamana group 和 FastGrnndCuda group 的 avg recall 都只有约 `0.61`。因此跳过 additional edges 的结果只能作为阶段拆分 ablation，不能作为完整质量配置。

端到端只有 `1.18x` 的原因是剩余阶段占比已经成为主导。启用 CPU additional edges 后，FastGrnndCuda 将 group graph 的 Index 占比从 `13.32%` 降到 `2.86%`，但 cross-edge 占比上升到 `51.31%`，LNG 占比为 `32.71%`。这是 Amdahl 限制：group graph 被加速后，进一步端到端收益需要优化 additional-edge 构建和 LNG，而不是只继续优化组内图 kernel。

### 5.9 SIFT30 的 Group Graph 端到端 A/B

为了检查结论是否只依赖 Amazon，我们补跑 SIFT30 查询 workload。该实验固定修复后的 fused cross-edge 和 CPU Vamana additional edges，只切换 group graph 后端。SIFT30 使用 `1,000,000` base vectors、`69,176` groups 和 `10,000` containment queries。

| Variant | Index ms | Group ms | Cross ms | L100 recall | L200 recall | L500 recall | L1000 recall |
|---|---:|---:|---:|---:|---:|---:|---:|
| CPU Vamana group | 31740.4 | 11245.3 | 17886.2 | 0.609423 | 0.706580 | 0.814600 | 0.877587 |
| FastGrnndCuda + CPU fallback | 24989.9 | 1889.79 | 17868.9 | 0.607997 | 0.706287 | 0.812680 | 0.876297 |

这组结果显示，SIFT30 上 group graph 构建加速 `5.95x`，总 Index time 加速 `1.27x`，L1000 recall 下降约 `0.00129`。修复后的 fused cross-edge 将 SIFT30 cross 从 CPU Vamana cross 的 `69834.5 ms` 降到 `17886.2 ms`，约 `3.90x`，并且 CPU Vamana group 的 fused-cross recall 与 CPU-cross recall 基本一致。

这轮实验也暴露并修复了两个 reviewer 会抓的语义问题。第一，`DIRECT_QID_ALL_FUSED=1` 不能覆盖 `nx>4096` 大组 fallback path 时，必须 materialize `g_d_Q` 和 `q_norm`。第二，host merge 路径下不能启用 id-only writeback，因为跨 target-group 的 `SearchQueue` 排序需要真实距离；id-only 只适合 GPU global merge 已完成全局排序的路径。

### 5.10 Amazon 1% x200 Coverage Query 与 Light Prune

为了避免只在原始 Amazon PF 和 SIFT30 上报告正向结果，我们补跑了 Amazon 1% x200 repeat-scale 的 coverage-query A/B。第一轮实验暴露旧重 prune 在 `nx≈200` 上不划算；随后我们加入 light prune 中等组路径，验证是否能把该 bucket 变成 GPU 可替代。query 由 group hierarchy 采样生成，覆盖范围如下：

| Metric | Value |
|---|---:|
| queries | `1000` |
| broad / medium / narrow | `316 / 420 / 264` |
| matched groups avg / p50 / p95 | `258.97 / 32 / 1087.4` |
| matched points avg / p50 / p95 | `55556.6 / 6800 / 244500` |

两个 variant 使用同一 query、同一 containment GT、同一 fused cross-edge 和 CPU Vamana additional edges，只切换 group graph backend：

| Variant | Index ms | Group ms | Cross ms | L100 recall | L200 recall | L500 recall | L1000 recall |
|---|---:|---:|---:|---:|---:|---:|---:|
| CPU Vamana group | `17596.8` | `8511.72` | `4601.91` | `0.8230` | `0.8250` | `0.8490` | `0.8690` |
| FastGrnndCuda, old heavy prune | `38273.7` | `26620.6` | `6055.72` | `0.8125` | `0.8273` | `0.8472` | `0.8662` |
| FastGrnndCuda, top32 light prune | `14211.8` | `6016.04` | `4926.37` | `0.8135` | `0.8239` | `0.8421` | `0.8618` |
| FastGrnndCuda, diversified light prune | `16478.5` | `6706.36` | `4857.33` | - | - | - | `0.8656` |
| batched exact kNN, pure local | `12595.9` | `4356.18` | `4120.75` | - | - | - | `0.8159` |

FastGrnndCuda 的主要差异来自 prune：

| Variant | H2D | GNN | prune | D2H | group total |
|---|---:|---:|---:|---:|---:|
| old heavy prune | `1538.32` | `3551.54` | `19218.3` | `413.047` | `26620.6` |
| top32 light prune | `1180.47` | `3198.35` | `712.392` | `134.631` | `6016.04` |
| diversified light prune | `1448.82` | `3229.56` | `759.66` | `161.408` | `6706.36` |
| batched exact kNN, pure local | `524.173` | `674.693` | `0` | `29.8523` | `4356.18` |

高 Lsearch 下，light prune 可以继续提高 recall，但仍低于 CPU Vamana：

| Variant | L1000 | L2000 | L5000 |
|---|---:|---:|---:|
| CPU Vamana group | `0.8690` | `0.8820` | `0.9080` |
| FastGrnndCuda, top32 light prune | `0.8619` | `0.8769` | `0.9029` |
| FastGrnndCuda, diversified light prune | `0.8656` | - | `0.9056` |
| batched exact kNN, pure local | `0.8159` | - | `0.8199` |

这个结果改变了论文写法：FastGrnndCuda 的失败不是 GPU group graph 本身不可行，而是重 RNG prune 在中等组上无法摊销。diversified light prune 后，x200 Index 快于 CPU Vamana `1.07x`，group graph 快 `1.27x`；packed exact-anchor 在 full-quality cross/additional 口径下进一步把 group graph 降到 `3827.28 ms`，L5000 达到 CPU baseline 的 `0.9080`。direct-H2D A/B 又说明 exact-anchor 旧实现仍被 host pack 和 `std::vector` fill 掩盖，手动 fill16 可把同一路径 group 降到 `3204.88 ms` 且 L1000 recall 不变。早期 exact 低 recall run 由于 cross-edge 数不同被重新归因为 confounded ablation。最终系统应写成 workload-aware router：能用 packed exact-anchor 的组走低外围 exact path，质量或规模更敏感的分布再由 reverse-tail/GNN 路径兜底。

### 5.11 Amazon 10% x40 Many-Group Stress Test

`Amazon 10% x40` 有 `2,409,800` 个点和 `53,840` 个 group，是对大量 target groups 和较小 group 的压力测试。balanced coverage query 已生成，`1000` 个 query 的 profile 为 medium/narrow/broad `540/238/222`，matched groups avg/p50/p95 为 `883.09 / 25.5 / 3297.0`，matched points avg/p50/p95 为 `39910.08 / 1060.0 / 138738.0`。

第一轮 full-quality build 在 `target_groups=49374` 的 cross-edge 阶段超过 10 分钟没有进入 GPU event 输出。profile 发现旧 path 虽然打印 `direct_pageable=1`，但因为连续性探测被 `hostreg_size_allowed` 短路，实际走 `pinned_staging`，导致 7.4GB base vectors 在 CPU 侧 repack 一次：`host_pack_ms=6205.6`，`prepare_all=6500.7 ms`。修复后 direct pageable path 生效，并且 double-buffer chunk 不再 materialize `d_Q/q_norm`，kernel 通过 `qid` 直接读常驻 `g_d_all_X/g_d_all_norm`。

skip-additional 构建结果如下，用于隔离 cross-edge 和 group graph 性能：

| Variant | Index | Group graph | Cross-edge | prepare_all | GPU kernel events | L100 sanity |
|---|---:|---:|---:|---:|---:|---:|
| CPU exact cross, CPU Vamana group | `89529.3 ms` | `26083.3 ms` | `54879.3 ms` | - | - | - |
| GPU direct-qid cross, CPU Vamana group | `42651.1 ms` | `22747.6 ms` | `8475.4 ms` | `2596.3` | `2669.4` | `0.800` |
| GPU direct-qid cross, FastGrnndCuda + complete fallback | `26422.1 ms` | `7689.0 ms` | `8869.1 ms` | `3222.4` | `2691.5` | `0.800` |

这给了两个新的写作点。第一，direct-qid fused cross-edge 在大量 group 场景下相对 CPU exact cross 仍有 `6.47x` 加速，说明贡献不是只适用于 1%x100。第二，FastGrnndCuda + complete fallback 把 Index 从 `89.5s` 降到 `26.4s`，但这只是在 skip-additional 口径下的性能证据；已有 Amazon PF 实验证明 additional edges 对 recall 很关键，因此 10%x40 的正式质量表必须用 `UNG_ADDITIONAL_EDGES_IMPL=0`。

full-quality A/B 结果如下：

```text
/home/graphdb/fv_runs/reviewer_e2e_amazon_10pct_x40_cpu_group_directqid_fulladd_20260602_005156
/home/graphdb/fv_runs/reviewer_e2e_amazon_10pct_x40_fastgrnnd_directqid_fulladd_20260602_005446
```

| Variant | Index | Group graph | Cross-edge | L100 | L500 | L1000 |
|---|---:|---:|---:|---:|---:|---:|
| CPU Vamana group + GPU direct-qid cross + CPU additional_edges | `36053.3 ms` | `18218.5 ms` | `9777.3 ms` | `0.8647` | `0.8980` | `0.9080` |
| FastGrnndCuda + complete fallback + GPU direct-qid cross + CPU additional_edges | `23497.4 ms` | `5147.2 ms` | `9451.1 ms` | `0.8668` | `0.8980` | `0.9103` |

full-quality 下，Index speedup 为 `1.53x`，group graph speedup 为 `3.54x`，L100/L500/L1000 recall 没有下降。这是目前 10%x40 上最强的 group graph 替换证据。repeat=3 的 per-query `Time_ms` 分位数也没有恶化：L100 P95/P99 从 `254.8/499.3 ms` 降到 `226.4/458.3 ms`，L500 从 `244.5/515.3 ms` 降到 `201.2/430.7 ms`，L1000 从 `244.2/503.6 ms` 降到 `231.3/494.0 ms`。随后补跑 L50/100/200/500/1000/2000/5000 sweep，FastGrnnd recall 相对 CPU 分别为 `+0.0019/+0.0018/-0.0010/+0.0000/+0.0020/+0.0020/+0.0010`；这说明 `1.53x` build speedup 不依赖单个 Lsearch 点。该 sweep 的 query latency 仍非单调，因此不能写成稳定 search latency 加速。

随后我们围绕“能否提出更普遍替代”补了一个小组 router 反例和一个系统优化。第一，若把 `UNG_GROUP_GRAPH_COMPLETE_NX` 从默认 `64` 降到 `32`，让几乎所有小/中组进入 packed exact-anchor，`TagoreCuda groups` 从 `1297` 增到 `53840`，group graph 反而从 `6097.03 ms` 变为 `7329.12 ms`。分项显示问题在外围而不是 exact kernel：`pack=2728.81 ms`、`h2d=1014.8 ms`、`fill=1968.77 ms`，L100 recall 仍为 `0.867`。这说明 x200/x400 上 exact-anchor 有效，不能直接外推为“所有小组都应上 GPU”。

第二，我们新增 `UNG_GROUP_GRAPH_BOUNDED_COMPLETE=1`：`nx<=max_degree+1` 仍保持 complete graph；更大但仍走 small fallback 的组，每点只写 `max_degree` 条环形强连通边。这个路径不是 complete graph 的逐边等价实现，而是 small-group fast path。10%x40 full-quality build 复测中，Index 从 `25035.9 ms` 降到 `22230.2 ms`，group graph 从 `6097.03 ms` 降到 `5240.6 ms`，fallback_wall 从 `4225.67 ms` 降到 `3088.13 ms`。复用该 index 的 repeat=3 search sweep 给出 L100/L500/L1000 recall `0.867/0.900/0.909`，与 CPU `0.8647/0.898/0.908` 和 FastGrnnd `0.8668/0.898/0.9103` 同量级。因此当前更合理的普遍替代设计是三段式 router：very-small groups 走 bounded-complete，small/medium groups 走 packed exact-anchor，large/quality-sensitive groups 走 FastGrnndCuda light/reverse-tail 或更强 prune。不过 bounded sweep 的 avg query time 为 `1083.34/951.434/839.479 ms`，慢于历史 CPU/FastGrnnd run；该点需要同脚本复测和路径诊断，不能写成 search-latency 改进。

### 5.12 Query Entry Group 加速与端到端查询上界

本节回答一个容易被误写的问题：入口组阶段 `19.85x` 加速不等于端到端查询 `19.85x`。UNG query 的计时边界可以拆成：

```text
Time_ms = MinSupersetT_ms + search_time_ms + 其他轻量收尾
search_time_ms = entry point materialization + core graph search
core_search_time_ms = iterate_to_fixed_point()
```

当前 GPU correct-cover 仍是独立 benchmark，还没有接入 `search_UNG_index` 的正式图查询路径。因此下表是“将 CPU entry stage 替换为 resident GPU entry stage、后续图搜索保持不变”的上界估算。GPU entry 时间使用 Amazon 100%x40、`nq=10240` 重跑 artifact 的 `59.8294 ms / 10240 = 0.00584 ms/query`。

| workload / Lsearch | old total ms/query | CPU entry ms/query | 后续 search ms/query | CPU avg NumEntries | entry 占比 | 替换 entry 后估算 total | 估算 speedup |
|---|---:|---:|---:|---:|---:|---:|---:|
| Amazon 10%x40 full-quality, L100 | `70.242` | `35.569` | `34.534` | `107.94` | `50.6%` | `34.679` | `2.03x` |
| Amazon 10%x40 full-quality, L500 | `62.420` | `31.801` | `30.615` | `107.94` | `50.9%` | `30.625` | `2.04x` |
| Amazon 10%x40 full-quality, L1000 | `66.440` | `31.825` | `34.610` | `107.94` | `47.9%` | `34.621` | `1.92x` |
| Amazon 1%x200 coverage, L1000 | `50.988` | `3.406` | `47.071` | `66.67` | `6.7%` | `47.588` | `1.07x` |
| Amazon 1%x200 coverage, L5000 | `48.166` | `3.764` | `44.241` | `66.67` | `7.8%` | `44.408` | `1.08x` |
| Amazon 1%x400 CPU baseline, L1000 | `41.229` | `3.492` | `37.280` | `29.08` | `8.5%` | `37.743` | `1.09x` |
| Amazon 1%x400 CPU baseline, L5000 | `36.188` | `3.918` | `31.953` | `29.08` | `10.8%` | `32.276` | `1.12x` |

结论是：many-group 查询中，CPU minimal-superset 查找本身已经占 `48~51%`，因此入口组 GPU 化有可能带来约 `2x` query latency 上界收益；但在 x200/x400 coverage query 中，后续图搜索已经主导，入口组加速只能给出 `1.07~1.12x` 上界。

这个上界还有一个重要 caveat：GPU correct-cover 的输出不是 CPU exact minimal。Amazon 100%x40 上 `delta=1, cap=8192` 平均输出 `2184.26` 个 group，而 CPU exact minimal 为 `616.24` 个 group。冗余入口组不会破坏 coverage，但可能增加 `get_entry_points_given_group_id()` 和 `iterate_to_fixed_point()` 的工作量。因此论文中应把当前结果写成：入口组阶段已经被显著加速；端到端查询收益在 many-group 场景上有明确上界潜力；正式主表仍需要接入 production search path 后同时报告 `NumEntries`、core search time、recall 和 latency。

100%x40 查询入口组重跑结果如下，artifact 为 `/home/graphdb/FilterVectorResultsRefactor/organized_benchmarks/query_entry_group_100pct_x40_20260605_105039`：

| 方法 | total ms | avg output groups | QPS | speedup vs CPU scan |
|---|---:|---:|---:|---:|
| CPU scan 128T | `1187.56` | `23332.2` | `8623` | `1.00x` |
| CPU exact minimal | `4569.83` | `616.24` | `2241` | `0.26x` |
| GPU correct-cover d1 cap8192 | `59.8294` | `2184.26` | `171153` | `19.85x` |

### 5.13 已有 Cross-Edge Query-Level Sanity Check

目前还有一组 Amazon PF 查询结果，可以在 group graph 固定为 CPU Vamana 的条件下，对比 existing、naive GPU cross 和 paper fused cross。它不能验证 FastGrnndCuda，但可以检查 cross-edge 后端切换是否改变最终 filtered-search recall。

| Variant | Lsearch | Avg recall | Batch time ms | Query P95 ms |
|---|---:|---:|---:|---:|
| CPU existing | 1000 | 0.908600 | 1479.140 | 246.578 |
| naive GPU cross | 1000 | 0.908367 | 1407.490 | 290.114 |
| paper fused cross | 1000 | 0.908600 | 1306.650 | 207.819 |
| CPU existing | 2000 | 0.928800 | 1106.310 | 141.951 |
| naive GPU cross | 2000 | 0.928800 | 1108.530 | 147.861 |
| paper fused cross | 2000 | 0.928767 | 1109.220 | 145.568 |
| CPU existing | 5000 | 0.969700 | 1035.900 | 139.046 |
| naive GPU cross | 5000 | 0.969733 | 1023.950 | 143.354 |
| paper fused cross | 5000 | 0.969767 | 1107.740 | 148.865 |

这组结果中，三种 cross-edge 后端的平均 recall 差异约在 `3e-4` 量级内，说明在 group graph 不变时，paper fused cross-edge 没有明显破坏查询质量。不过该 run 没有启用 FastGrnndCuda，因此只能作为 cross-edge sanity check，不能作为最终 group graph 替换的质量证明。

### 5.14 Amazon 1% x400 Gather-Q Full-Quality A/B

x400 是当前最强的 group graph 泛化反例：它的单组规模足以让 GPU 候选生成有并行度，但 x400 direct-qid 路径仍不稳定，所以主表使用稳定 gather-Q fused cross-edge。我们为 x400 生成 coverage query：`1000` 个 query，medium/narrow/broad 为 `672/258/70`，matched groups avg/p50/p95 为 `73.37/21.0/134.2`，matched points avg/p50/p95 为 `31016.0/8400.0/53700.0`。GT exact compute 为 `14039 ms`。

三个 variant 使用同一 query、同一 GT、同一 gather-Q fused cross-edge 和 CPU Vamana additional edges，只切换 group graph：

```text
/home/graphdb/fv_runs/reviewer_e2e_amazon_1pct_x400_gatherq_fulladd_20260602_010839
/home/graphdb/fv_runs/reviewer_e2e_amazon_1pct_x400_light512_gatherq_fulladd_20260602_011323
```

| Variant | Index ms | Group ms | Cross ms | Prune ms | L1000 | L5000 | P95 L1000 | P95 L5000 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| CPU Vamana group | `37807.6` | `17537.3` | `12296.6` | - | `0.9460` | `0.9660` | `121.4` | `75.6` |
| FastGrnndCuda heavy prune | `52552.2` | `29192.3` | `12815.3` | `19977.7` | `0.9443` | `0.9643` | `96.2` | `64.1` |
| FastGrnndCuda light512 | `30826.5` | `10362.9` | `12681.6` | `1022.45` | `0.9379` | `0.9579` | `102.4` | `62.5` |
| FastGrnndCuda light512 head28 | `30897.7` | `10276.6` | `12692.3` | `1001.65` | `0.9323` | `0.9523` | `101.6` | `66.1` |

x400 的结论比 10%x40 更苛刻。Heavy prune 保留了接近 CPU 的质量，L1000/L5000 只低 `0.0017`，但 prune 占 `19977.7 ms`，导致 Index 变慢到 `52552.2 ms`。把 light-prune 阈值提高到 `512` 后，prune 降到 `1022.45 ms`，group graph 相对 CPU 加速 `1.69x`，Index 加速 `1.23x`，但 L1000/L5000 recall 均下降 `0.0081`。进一步把 `LIGHT_HEAD` 从默认 `24` 提到 `28` 没有补回质量，反而把 L1000/L5000 降到 `0.9323/0.9523`。这与 light kernel 的机制一致：它先保留近邻头部，再用剩余名额从 GNN 候选尾部等距抽样；增大 head 会挤掉尾部多样性。

为了确认 recall gap 的结构原因，我们对保存的 x400 CPU Vamana 和 light512 index 运行了 `tools/benchmarks/diagnose_ung_graph_structure.py`，并用 `tools/benchmarks/summarize_graph_diagnostics.py` 生成比较表。两者 cross edges 完全相同，都是 `2,131,224` 条；差异集中在组内图：light512 的 intra edges 从 `75,767,742` 降到 `73,538,978`，少 `2,228,764` 条，平均组内出度从 `31.4441` 降到 `30.5192`，零组内出度比例从 `0` 增到 `0.00199162`，低组内出度 `<=4` 比例从 `0.000000415` 增到 `0.00929781`，并出现 `39` 个最大 weak component 小于 `0.9` 的 group。因此 x400 不能支持“FastGrnndCuda 无损替代 CPU Vamana”的写法；它要求一个更细的剪枝预算或质量 router。后续 x400 A/B 显示，degree repair 单独几乎消除 zero/low-degree，但 Index 变慢且 L5000 只提升 `0.000367`；reverse-tail+repair 则把 L5000 从 `0.9574` 提到 `0.9659`，接近历史 CPU `0.9660`，同时仍比历史 CPU Index 快。因此 x400 的有效方向是低成本 tail/reverse diversity，而不是简单补边或一味增加 nearest-head。

## 6. 分析与讨论

### 6.1 为什么 cuVS per-group 不够好

cuVS 的 brute-force kernel 本身适合 dense search，但 UNG 的 workload 是大量小到中等 group 组合。逐 group 调用导致以下成本无法摊销：

- 每组 query/target pack。
- 每组 H2D/D2H。
- 每组 API 和 kernel launch。
- 每组结果从库格式转换到 UNG graph 结构。

因此 x100 上 cuVS per-group cross 为 `3730.5 ms`，而 final fused 为 `848.9 ms`。主要收益来自 workload-level batching 和外围开销减少，而不是单个 dot product kernel 的绝对算力优势。

### 6.2 为什么 SGEMM+topK 仍是强 baseline

SGEMM + separate topK 利用了成熟 cuBLAS/cuBLASLt，kernel 性能强。因此本文 final fused 相对 SGEMM+topK 的 x100 加速为 `1.64x`，不是数量级。本文方法的价值在于：

- 不写出完整距离矩阵。
- 减少 separate topK 和中间 buffer。
- 在 GPU global merge 已完成全局排序时利用 id-only writeback 降低 D2H；host merge 路径保留真实距离以保证跨组排序正确。
- 通过 direct-qid 避免 query 展开。

### 6.3 `prepare_all` 的解释

`prepare_all` 是 data residency cost。它在一次构建进程中只发生一次，但无法跨进程复用。论文中应同时报告：

- **single-build total**：包含 `prepare_all`，代表实际 Index time。
- **resident compute**：不包含 `prepare_all`，代表数据常驻后 cross-edge 算法本体。

忽略 `prepare_all` 会夸大 GPU 算法收益；把 `prepare_all` 当作每 group 成本又会低估方法。

### 6.4 x400 direct-qid 稳定性

x400 direct-qid fused 触发 medium group-desc kernel illegal memory access；将 `UNG_GPU_FLAT_Q_CAP_MB` 降到 512 MB 后仍失败，说明问题不是单次 chunk 过大。当前稳定方案是关闭 direct-qid，使用 gather-Q fused。论文主结果中不应把 fallback CPU 的 `86359.0 ms` 当作 GPU 性能，也不应声称 direct-qid 已覆盖所有 scale。

## 7. 相关工作定位

### 7.1 GPU brute-force 与 library baselines

cuBLAS/cuBLASLt 和 cuVS 提供了强大的 GPU dense search 基础能力。本文不试图替代这些库的通用场景，而是针对 UNG 的大量 group-level topK workload 做调度、融合和写回优化。

### 7.2 GPU 图构建

Tagore 类系统使用 GPU GNN-Descent 和 pruning 构建 ANN 图。GRNND/RNN-Descent 方向进一步将 RNG-style pruning 和邻居传播融入迭代过程，提升 GPU 构图质量。本文方法受 reverse-neighbor propagation 和 RNG-style pruning 启发，但没有复现完整 GRNND；我们复用 GNN-Descent 候选，将 reverse augmentation 与 local RNG pruning 作为 UNG group graph 的后端替换。

### 7.3 过滤近邻搜索索引

过滤 ANN 索引需要处理属性选择性、标签层级和局部图可达性。UNG 的 cross-edge 设计正是为了补充 group 之间的可达路径。本文关注构建阶段加速，与查询算法优化互补。

## 8. 局限与后续工作

本文当前仍有以下限制：

1. **端到端 search recall 尚需更多真实数据集。** 原始 Amazon PF、SIFT30、Amazon 1% x100、Amazon 1% x200 coverage、Amazon 10% x40 和 Amazon 1% x400 gather-Q 已有 group graph A/B。x100 已补 full-quality repeat=3 质量闭环；x200 已从旧重 prune 反例变成 light prune 的 speed/quality tradeoff；10%x40 full-quality repeat=3 A/B 已显示 FastGrnndCuda `1.53x` Index 加速且 L100/L500/L1000 recall/P95/P99 不降；x400 reverse-tail+repair 将 L5000 从 light512 的 `0.9574` 提到 `0.9659`，接近同脚本 CPU `0.965`，Index 约 `1.10x`；compact-D2H 已补负/不稳定 ablation。最终论文还需要更多 x400 参数扫，并补 CelebA 或真实多标签端到端数据。
2. **x400 direct-qid 不稳定。** 需要修复 medium group-desc kernel 的大规模 qid 路径，或设计更稳健的 gather/direct 混合策略。
3. **CPU-compatible 输出边界仍是瓶颈。** 当前 GPU group graph 和 cross-edge 结果最终仍回填到 host `Graph::neighbors` / `SearchQueue` / Vamana-compatible structures。direct-H2D exact-anchor A/B 已经把输入侧 host pack 从 `4140.61 ms` 降到 `0.04 ms`，但输出侧 `Graph::neighbors` fill 曾长期为秒级；这说明剩余边界是 host 图物化，而不是 exact kernel。Graph reserve A/B 进一步证明 per-node adjacency 分配/扩容是实质瓶颈：Amazon 10%x40 中 reserve 把 Index 从 `39746.1 ms` 降到 `24704.5 ms`，`tagore_fill` 从 `823.5 ms` 降到 `23.9 ms`，cross merge 从 `1677.1 ms` 降到 `2.3 ms`；但旧 reserve 自身预留 `106,031,200` 条容量并花 `6487.3 ms`，只是把 allocator 成本前置。最新 64-inline `NeighborList` small-buffer 把 x200 reserve 从 `2856.2 ms` 降到 `25.4 ms`、fill 从 `224.8 ms` 降到 `14.7 ms`；10%x40 reserve 从 `6487.3 ms` 降到 `21.2 ms`、fill 从 `23.9 ms` 降到 `7.6 ms`。48-inline 是负例，x200 fill 回退到 `202.5 ms`。因此 small-buffer 是当前保留的低风险输出边界优化，但它仍是 CPU-compatible per-node object，不是 CSR。我们在 Amazon 10%x40 上测试了更激进的 `UNG_GPU_DB_NOSPLIT=1`、`UNG_GPU_DB_GLOBAL_MERGE=1`、`UNG_GPU_ID_VECTOR_WRITEBACK=1` 路径：D2H/writeback 降到 `3.5/33.2 ms`，但 qid-lock 竞争使 kernel 从 `2952.2 ms` 升到 `4834.3 ms`，cross 总时间从 `9451.1 ms` 变慢到 `11598.0 ms`。因此简单带锁 global merge 是负结果。较低风险的方向是 partial double-buffer routing：在 x200 中它把 cross-edge 降到 `2657.23 ms`，但仍保留 host `SearchQueue` 语义。进一步的 direct-global/flat-id smoke 说明：direct-global 只能把 x100 skip-additional 的 `add_offset` 从 `2.5 ms` 降到 `0.0 ms`，不是主要收益；flat-id writeback 将同口径 cross 从 `1650.95 ms` 降到 `707.83 ms`，D2H 从 `11.1 ms` 降到 `0.6 ms`，merge 从 `17.2 ms` 降到 `0.5 ms`，recall 保持 `0.816`。但该实验跳过 additional_edges，且 direct-global 与 CPU Vamana additional 不兼容。additional_edges direct append 的 x400 full-quality 探索也没有稳定完成，即使用 per-node lock 仍停在 GPU `batched_search end` 后，因此不能把“直接并行 append 到 host graph”作为解决方案。因此更完整的替换路径仍需要 qid-sharded/segmented no-lock merge 和 flat adjacency/CSR graph backend，并补 full-quality recall A/B。
4. **FastGrnndCuda prune 是瓶颈，且需要自适应强度。** x400 heavy prune 为 `19977.7 ms`，x200 old heavy prune 为 `19218.3 ms`。diversified light prune 将 x200 prune 降到 `759.66 ms`，light512 将 x400 prune 降到约 `1006 ms`，但原始 light512 质量 gap 为 `0.0081`。reverse-tail+repair 只把 prune 增到 `1093.37 ms`，却将 L1000/L5000 各提升 `0.0085`。packed exact-anchor 的新结果说明局部 exact 图在部分 coverage-query 上可以达到 CPU 级 recall；后续问题转为何时 exact、何时 reverse-tail/GNN，而不是把 exact 作为已失败路线排除。
5. **fallback/router 策略仍偏工程化。** 当前最快端到端结果依赖 CPU fallback、bounded-complete、packed exact-anchor 或 light prune 的组合。10%x40 证明 all-small-groups GPU exact 会被 pack/H2D/fill 吞掉收益，bounded-complete 则能减少小组图物化成本并保持 recall；但 bounded-complete 的 search latency 单独复测偏慢，后续需要把三段式 group-size/quality router 做成 principled policy，并补同脚本 repeat-scale 质量/延迟表。
6. **100%x40 建图暴露 resident dataset 和 Q 展开边界。** 本轮已生成 Amazon 100%x40 数据并尝试 optimized skip-additional build。旧的 `prepare_group_storages_graphs` segfault 已定位为 `Storage` 中 `uint32_t` 乘法溢出并修复；旧 cross-edge universal route 还要求把全量 `23,284,680 x 768` float 向量常驻 GPU，`g_d_all_X` 需要 `71.5GB`，在 48GB A6000 上 OOM。新增 X-side streaming 后，strict skip-additional build 已完成：Index `945683 ms`，cross-edge generate `697114.7 ms`。但 v1 streaming 的主要开销是重复 host 打包 Q，`pack_q=450338.9 ms`，GPU kernel 为 `72749.8 ms`。因此 100%x40 可以作为“显存边界已可绕过”的 scalability 结果，但不能进入建图 speedup 主表；后续需要 Q-side device cache/gather、source-centric two-stage 或 flat/CSR 输出边界。
7. **查询入口组 GPU 化还缺正式端到端接入。** `gpu_cover_frontier` 已在独立 benchmark 中证明入口组阶段可加速，但 `search_UNG_index` 当前正式路径仍调用 CPU `get_min_super_sets_debug()`。由于 correct-cover 输出可能比 CPU exact minimal 更多入口组，接入后必须同时测 `NumEntries`、entry point materialization、`core_search_time_ms`、recall 和 total `Time_ms`；不能只用入口组 microbenchmark 声称端到端 query latency 已经加速。
8. **实验数据集仍需扩展。** SIFT30 已补修复后端到端 A/B，但当前主性能结果仍集中在 Amazon sampled+jitter 系列，后续需要补 CelebA/真实多标签数据集。

## 9. 结论

本文展示了过滤向量索引构建中“系统级 GPU 化”的必要性。UNG 的 cross-edge 构建不是单个大 GEMM，而是由大量 group-level topK search 组成；组内图构建也不是统一适合 GPU 或 CPU，而是受 group size、外围搬运和图质量要求共同影响。通过 grouped fused topK、direct-qid 常驻访问、TF32 WMMA fused topK、安全写回和 FastGrnndCuda/packed exact-anchor/ bounded-complete 组内图路由，本文在 Amazon 1% x100 上将 cross-edge 降到 `848.9 ms`，并将端到端 Index time 降到 `6630 ms`。x200 coverage-query 结果进一步表明，自适应剪枝能把 heavy-prune 反例变成 faster-than-CPU 的 speed/quality tradeoff，而 packed exact-anchor 在 full-quality 口径下可以进一步达到 CPU 级 L5000 recall 和更低 group build time。cross-edge 的 partial double-buffer routing 还说明，性能不佳常来自 fast-path 覆盖边界而不是单个 fused kernel 算力：x200 中把 supported/unsupported target groups 拆开后，cross-edge 从 `3900.81 ms` 降到 `2657.23 ms`。10%x40 many-group 压力测试进一步说明，普遍替代必须把数据常驻、direct-qid、descriptor scheduling、fallback policy 和 additional edges 统一考虑：full-quality 下我们已得到 `1.53x` Index 加速且 L100/L500/L1000 recall 不降；进一步的 bounded-complete、Graph reserve 和 NeighborList small-buffer 实验证明 very-small groups 不应盲目 GPU 化，而应减少 CPU 图物化和 host adjacency 分配。查询阶段的 correct-cover 入口组选择显示，CPU minimal-superset 查找在 many-group workload 中可占近半 query latency，GPU bitset/frontier/descendant pipeline 能把该阶段降到 `0.00584 ms/query` 量级，并给出约 `2x` 的端到端查询上界潜力；但它必须和后续图搜索一起评估，因为冗余入口组可能增加 `iterate_to_fixed_point()` 的工作。Graph reserve 证明 allocator 是一阶瓶颈，NeighborList 把 reserve/fill 的主要 allocator 成本降到毫秒级，但仍保留 CPU-compatible per-node graph；qid-lock global merge 负结果也说明，输出边界不能靠简单 device lock merge 消除，后续需要 no-lock segmented merge 和 flat/CSR graph backend。x400 gather-Q full-quality 结果则给出当前最强反例：light prune 可把 Index 加速到 `1.23x`，但 recall gap 达 `0.0081`；heavy prune 质量接近却更慢。因此最终论文应把方法定位为 workload-aware GPU backend 和质量/性能 router，而不是无条件替代 CPU Vamana。实验同时表明，`prepare_all`、fallback 策略、x400 direct-qid 稳定性、CPU-compatible 输出边界、query entry group 接入和 group graph router 设计是进一步提升的关键方向。

## 附录 A. 关键实验日志

最终技术报告：

```text
docs/reports/TECHNICAL_REPORT_OPTIMIZATION_SPEEDUP_CN.md
```

最新 full optimized 重测：

```text
/home/graphdb/FilterVectorBenchResults/final_report_retest_20260528_prof/x1p40_final_opt256/run.log
/home/graphdb/FilterVectorBenchResults/final_report_retest_20260528_prof/x1p40_final_opt256/gpu_prof.log
/home/graphdb/FilterVectorBenchResults/final_report_retest_20260528_prof/x1p100_final_opt128/run.log
/home/graphdb/FilterVectorBenchResults/final_report_retest_20260528_prof/x1p100_final_opt128/gpu_prof.log
/home/graphdb/FilterVectorBenchResults/final_report_retest_20260528_prof/x1p200_final_opt256/run.log
/home/graphdb/FilterVectorBenchResults/final_report_retest_20260528_prof/x1p200_final_opt256/gpu_prof.log
/home/graphdb/FilterVectorBenchResults/final_report_retest_20260528/x1p400_final_opt256_gatherq/run.log
/home/graphdb/FilterVectorBenchResults/final_report_retest_20260528/x1p400_final_opt256_gatherq/gpu_prof.log
```

主要 baseline：

```text
/tmp/fv_amazon_1pct_x100_cpu_vamana/run.log
/tmp/fv_amazon_1pct_x100_cpu_exact/run.log
/home/graphdb/FilterVectorBenchResults/onchip_rewrite_20260527/x1p100_cpu_group_sgemm_topk_baseline/run.log
/home/graphdb/FilterVectorBenchResults/onchip_rewrite_20260527/x1p100_cpu_group_cuvs_pergroup_baseline/run.log
/home/graphdb/FilterVectorBenchResults/onchip_rewrite_20260527/x100_hybrid_default/run.log
```

组内图方法总结：

```text
docs/reports/FINAL_GROUP_GRAPH_METHOD_CN.md
docs/reports/FASTGRNNDCUDA_AMAZON_SCALE_RESULTS_CN.md
```

## 附录 B. Artifact Reproducibility Map

| 论文 claim / 审稿问题 | 复现入口 | 主要输出 artifact | 当前状态 |
|---|---|---|---|
| x100 cross-edge 是否只打弱 CPU baseline | `tools/benchmarks/summarize_cross_baselines.py` | `/home/graphdb/fv_runs/baseline_fairness_x100_20260602/cross_baselines.md` | 已生成；包含 CPU Vamana、CPU exact 128T、cuVS per-group、SGEMM+topK、final fused |
| FastGrnndCuda 是否保持端到端 search recall | `scripts/benchmarks/run_end_to_end_recall_ab.sh` | 各 run 的 `results/search_time_summary.csv`、`results/query_details_repeat*.csv` | Amazon PF、SIFT30、x100、x200、10%x40、x400 已有部分证据；真实多标签仍缺 |
| x400 light prune recall gap 是否来自组内结构 | `tools/benchmarks/diagnose_ung_graph_structure.py` 和 `tools/benchmarks/summarize_graph_diagnostics.py` | `/home/graphdb/fv_runs/graph_diagnostics_x400_20260602/summary_table.md` | 已完成历史 CPU vs light512 诊断 |
| x400 repair / reverse-tail 是否能补回质量 | `scripts/benchmarks/run_x400_reverse_tail_ab.sh` | `/home/graphdb/fv_runs/x400_reverse_tail_ab_20260602_030908/x400_ab_summary.md`、`x400_ab_summary.csv` | 已实测；reverse-tail+repair 基本补回 light512 recall gap，repair-only 是 partial/negative |
| compact-D2H 是否只是外围优化 | `RUN_COMPACT_ABLATION=1 scripts/benchmarks/run_x400_reverse_tail_ab.sh` | `/home/graphdb/fv_runs/x400_compact_ab_20260602_080045/x400_ab_summary.md` 中 `tagore_d2h_time`、`tagore_fill_time`、Index 对比 | 已实测；负/不稳定，只能写成 overhead ablation，不能写成算法质量贡献 |
| mixed exact/GNN router 是否能低风险加速小/中组 | `scripts/benchmarks/run_group_graph_router_ab.sh` 和 `tools/benchmarks/summarize_group_graph_router_ab.py` | `/home/graphdb/fv_runs/group_graph_router_ab_streamfill_20260602_030037/router_ab_summary.md`、`router_ab_summary.csv` | x200 repeat=1 已实测；投稿前仍需 repeat=3 / 跨数据集确认 |
| partial double-buffer routing 是否修复 cross-edge all-or-nothing 回退 | `scripts/benchmarks/run_end_to_end_recall_ab.sh` + `UNG_GPU_DB_NOSPLIT=1`、`UNG_GPU_DB_SPLIT_UNSUPPORTED=1` | `/home/graphdb/fv_runs/target_fused_x200_db_split_20260602_060034/end_to_end_recall_ab_20260602_060034`；结构诊断 `/home/graphdb/fv_runs/graph_diag_x200_db_split_20260602` | 已实测；x200 cross `3900.81 -> 2657.23 ms`，repeat=3 L1000/L5000 `0.866/0.907`，但质量仍低于 source-centric |
| x400 direct-qid 是否稳定 | direct-qid strict rerun 或 formal gather-Q/direct-qid router | direct-qid 成功日志或 limitation 记录 | 当前仍作为 limitation；x400 主表使用 gather-Q |

上述表的作用是限制论文表述边界：只有已经有 artifact 的 claim 才能进入主表；待验证实现只能写成实验计划、ablation 或 limitation。

## 附录 C. 可直接转成英文论文的标题候选

1. **Workload-Aware GPU Construction for Filtered Navigating Graphs**
2. **Grouped Fused TopK and Reverse-Augmented Pruning for Fast Filtered Vector Index Construction**
3. **Accelerating Unified Navigating Graph Construction with Grouped GPU TopK and Local RNG Pruning**
