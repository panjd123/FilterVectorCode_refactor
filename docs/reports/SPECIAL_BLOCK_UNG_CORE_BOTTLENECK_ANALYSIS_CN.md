# Special Block UNG Core Time 瓶颈与 3-5 倍提速分析

## 结论

在 Amazon `query_selected_recall_advantage` 上，当前 SpecialBlockTrie Router 首次达到
`Recall >= 0.95` 的工作点是 `Lsearch=2000`：

| 方法 | 参数 | Recall | QPS |
|---|---:|---:|---:|
| 当前 SpecialBlockTrie Router | Lsearch=2,000 | 0.950786 | 1,632.1 |
| 当前方法 100 线程稳态复测 | Lsearch=2,000 | 0.950786 | 1,644.6 |
| 当前目录最强 baseline | CPU ELS special blocks, Lsearch=12,000 | 0.954744 | 1,644.5 |

因此当前方法与最强 baseline 基本持平。3 倍和 5 倍目标分别是约 `4,933 QPS` 和
`8,222 QPS`；1,971 条 query 的 batch wall time 必须从约 `1.20 s` 降至
`400 ms` 和 `240 ms`。

主要结论如下：

1. `99.7%` 查询时间在 core search，ELS 已不是瓶颈。
2. `84.1%` core time 用在 block special graph：预展开占 `8.7%`，主 special-edge
   搜索占 `75.4%`。
3. 每 query 平均扫描约 `9.70 万` 条 free special edges，其中 `75.5%` 只命中已访问
   节点。
4. 每 query 平均做 `2.76 万` 次 768 维精确距离计算。邻居候选中 `80.4%` 在算完完整
   距离后立刻被 beam 最差距离拒绝。
5. 当前有序 `std::vector` beam 每 query 搬移约 `413 万` 个 candidate，约等于
   `47 MiB/query` 的候选数组移动。
6. 当前最大的问题不是 Trie 入口查找，而是把“block 对 query 合法”直接变成“立即激活
   并预展开这个 block”。平均每 query 有 `49.5` 个 free block，当前 Router 几乎将其
   全部预展开，使搜索过早进入高出度 block graph。
7. 只替换距离 kernel、关闭统计或调整线程数无法得到 3 倍。3 倍需要同时减少 block
   激活数、special-edge 扫描、完整距离计算和 beam 搬移；5 倍需要引入压缩距离或新的
   block-level 路由结构。

## 实验口径

- Dataset: Amazon，602,453 vectors，dimension 768。
- Query task: `query_selected_recall_advantage`，1,971 queries，225 种 label set。
- `K=10`，默认 100 search threads。
- 当前索引：`UNG_special_block_trie_regular`。
- 当前索引包含 170 blocks、35,430,825 条 special edges：
  25,535,489 条 intra-block edges 和 9,895,336 条 inter-block edges。
- Core phase profiling 使用 `UNG_SPECIAL_PROFILE_TIMING=1`，它在 adjacency scan 中增加
  计时开销。因此 phase 百分比可信，profiled QPS 不作为最终吞吐值。
- 候选计数 profiling 使用 detail stats；搜索决策与非 profiling 路径相同。

结果目录：

- Recall/QPS threshold sweep:
  `FilterVectorResult/Amazon/results/special_trie_final_threshold_sweep/`
- Core phase profile:
  `FilterVectorResult/Amazon/results/special_trie_final_core_profile/`
- Candidate diagnostics:
  `FilterVectorResult/Amazon/results/special_trie_core_diag_profile/`
- 控制变量复测:
  `FilterVectorResult/Amazon/results/special_trie_core_control_*/`

## 目标差距

| 目标 | QPS | 1,971 queries wall time | 相对当前稳态提速 |
|---|---:|---:|---:|
| 当前 | 1,644.6 | 1,198.5 ms | 1.00x |
| baseline 3x | 4,933.4 | 399.5 ms | 3.00x |
| baseline 5x | 8,222.4 | 239.7 ms | 5.00x |

Profile 中平均 core time 为 `65.39 ms/query`。在并行度和负载均衡不变的近似下，
3 倍和 5 倍对应约 `21.80 ms/query` 和 `13.08 ms/query` 的 core 预算。

## Core Time 分解

`Lsearch=2000` 的 phase-only profile：

| 阶段 | 平均时间/query | Core 占比 | 内容 |
|---|---:|---:|---|
| Block coverage | 0.014 ms | 0.02% | 170 blocks 的 common-label coverage 和 free 传播 |
| Entry scoring | 8.703 ms | 13.31% | group points、block landmarks 的精确距离和入口保留 |
| Block seed preexpand | 5.664 ms | 8.66% | 强制展开 routed block landmarks |
| Main special edges | 49.331 ms | 75.44% | free block graph 扫边、visited、距离和 beam 插入 |
| Regular edges | 1.002 ms | 1.53% | terminal group regular graph 搜索 |
| Result materialization | 0.007 ms | 0.01% | 输出 top-K |
| Core 未单独归因 | 0.690 ms | 1.06% | loop/control 和少量临时结构开销 |
| **Core total** | **65.391 ms** | **100%** | |

Core 之外：

| 部分 | 平均时间/query | 总查询时间占比 |
|---|---:|---:|
| ELS / Special Trie provider | 0.069 ms | 0.11% |
| Other | 0.127 ms | 0.19% |
| Core | 65.391 ms | 99.70% |

所以继续优化 Trie posting、terminal group 数或 ELS cache，最多只能获得接近噪声水平的
收益。Entry scoring 有优化价值，但单独完全消除它也只有约 `1.15x` 的理论上限。

## Hot Path 工作量

`Lsearch=2000` 每 query 平均值：

| 指标 | 当前值 | 解释 |
|---|---:|---|
| Distance calculations | 27,588.95 | 每次读取并计算 768-D float vector |
| Free distance calculations | 24,946.12 | 90.4% 的距离计算发生在 free/block 路径 |
| Regular distance calculations | 2,642.83 | regular 路径只占 9.6% |
| Free special edges scanned | 96,968.33 | 主要扫边负担 |
| Regular edges scanned | 1,538.72 | 只占约 1.56% scanned edges |
| Special edges accepted | 23,805.38 | 24.5% special edges 首次命中节点 |
| Preexpand edges scanned | 2,933.12 | 约占 special scans 的 3.0% |
| Preexpand edges accepted | 2,847.01 | 接受率 97.1%，几乎每条都触发距离计算 |
| Queue insert attempts | 23,991.83 | 已计算距离后尝试进入 beam |
| Queue bound rejections | 19,278.41 | 80.35% 尝试立即被最差距离拒绝 |
| Actual queue insertions | 4,713.42 | 只有 19.65% 真正进入 beam |
| Shifted candidates | 4,134,308 | 有序 vector 插入造成的元素搬移 |
| Free nodes expanded | 1,674.90 | 主循环展开的 free nodes |
| Regular nodes expanded | 376.18 | 主循环展开的 regular nodes |
| Free blocks searched | 49.49 | 基本等于所有 eligible free blocks |

### 1. Special graph 访问过宽

当前 free special graph 每个被展开节点平均带来约 58 条边。加上预展开的 block seed，
每 query 扫描约 9.70 万条 special edges。只有约 2.38 万条边首次命中节点，另外
`75.5%` 只是 visited check。

这不是简单的重复 edge record。更主要的原因是多个已展开节点汇聚到相同邻居，以及
每个 parent point 都携带到 child block 的 cross-block edges。当前 free 路径扫描时只用
`target_point_id`，并没有利用 `SpecialEdge` 中的 block/kind 信息跳过已经激活的 child
block。

### 2. 完整距离算完才知道候选无用

`visit_neighbor` 先调用完整 768-D `score_distance`，然后才在 `insert_candidate` 中检查
`queue.back()`。因此 19,278 个 bound-rejected candidates 已经支付了完整距离成本。

仅 base-vector payload 就是：

```text
27,588.95 * 768 * 4 bytes = 80.83 MiB/query
```

按当前约 1,645 QPS 计算，base-vector 有效读取量约 `139 GB/s`，还不包括 adjacency、
visited arrays、queue 和其他元数据。若不减少距离工作，3 倍和 5 倍分别要求约
`418 GB/s` 和 `697 GB/s` 的 vector payload，已不现实。

### 3. Beam 使用 O(L) 插入

当前 beam 是按 `(distance, id, free)` 排序的 `std::vector`。每次成功插入用二分查找
定位，但 `vector::insert` 仍需移动后半段。在 Lsearch=2,000 时：

```text
4,134,308 shifted candidates/query
约 47.3 MiB/query（按 12-byte SpecialCandidate）
```

这些移动大多发生在 cache hierarchy，但仍是高频写流量。已有设计文档
`docs/superpowers/plans/2026-08-08-ung-special-candidate-heaps.md` 的 heap 方案值得实施，
但它单独不可能提供 3 倍。

### 4. Free eligibility 与 block activation 混在一起

父 block free 后 child block free 是正确的 Trie 单调性质。但 free 的含义应当是：
“若搜索到该 block，可以不再做标签过滤并自由走 block graph”，而不是“查询开始时必须
给所有 free descendants 注入 seed 并强制预展开”。

当前平均状态：

```text
49.49 free blocks/query
62.37 retained block landmarks/query
51.33 forcibly preexpanded landmarks/query
49.49 blocks actually searched/query
```

也就是 eligibility 几乎直接变成 activation。对宽标签 `{1}` 和 `{2}`，分别有 97 和
95 个 free blocks，约 98/97 个 landmarks 被强制预展开。此时搜索退化为同时搜索大量
block-local graphs。

正确的提速方向是保留 free 继承语义，但让 activation 变成 lazy：先在 block/group
router 上找到少数有希望的 blocks，再进入其 free graph；其他合法 block 只保留为可达
状态，不在 query 开始时全部预展开。

## 与最强 Baseline 的 Core 工作量对比

虽然两者索引和入口 provider 不同，但相同 query task 上的工作量差异能解释为什么当前
方法尚未变快：

| 指标/query | 当前 Trie Router L=2,000 | CPU ELS baseline L=12,000 |
|---|---:|---:|
| Recall | 0.950786 | 0.954744 |
| QPS | 约 1,633-1,645 | 1,644.5 |
| Distance calculations | 27,589 | 18,667 |
| Special edges scanned | 96,968 | 65,550 |
| Regular edges scanned | 1,539 | 9,119 |
| Free nodes expanded | 1,675 | 1,072 |
| Regular nodes expanded | 376 | 1,669 |

baseline 虽然展开更多 total nodes，但其中大量是低出度 regular nodes；它少展开约 36%
的高出度 free nodes，因此总 special scans 和距离计算分别少约 32%。当前方法更早进入
free block graph，把便宜的 group-level 导航替换成了昂贵的 block-local 搜索。

这说明新的 block router 不应继续增加全局 block seeds。应建立真正的 block-level
导航层，用便宜的 regular/block routing 减少进入 free graph 的 block 数量。

## 查询分布与长尾

1,971 条 query 中有 1,602 条是单标签查询。它们贡献约 `87.4%` 的总 core time。
最重的四个 label set 为 `{1}`、`{2}`、`{20835}`、`{20836}`，合计贡献约 `74.2%`
core time。

| Label set | Query 数 | Mean core | Mean special-edge time | Recall |
|---|---:|---:|---:|---:|
| `{1}` | 422 | 114.7 ms | 86.0 ms | 0.9540 |
| `{2}` | 298 | 105.4 ms | 79.8 ms | 0.9198 |
| `{20835}` | 295 | 77.1 ms | 68.4 ms | 0.9390 |
| `{20836}` | 173 | 61.5 ms | 56.4 ms | 0.9838 |

单标签路径既是吞吐瓶颈，也是 recall 风险集中处。只优化四个固定标签不够通用，而且即使
这四类快 5 倍，Amdahl 总提速也只有约 2.46 倍。更合理的通用策略是：为高频或高覆盖
单标签建立 filter-specific coarse graph/direct graph，其他查询继续走通用 Trie/block
路径。

慢查询也会限制 batch wall time。Profile 中 p95 core 是约 `200 ms`，最大约
`420 ms`；当前最大单 query 已超过 3 倍目标的整个 `400 ms` batch 预算。因此优化必须
同时压低单标签长尾，不能只改善平均小查询。

## 控制变量结果

所有设置固定 `Lsearch=2000`，取 repeat 2/3 的稳态平均：

| 设置 | Recall | Steady time | Steady QPS | 结论 |
|---|---:|---:|---:|---|
| Detail stats | 0.950786 | 1,242.2 ms | 1,586.8 | 控制组 |
| Light stats | 0.950786 | 1,244.1 ms | 1,584.2 | 统计计数不是瓶颈 |
| AVX2 FMA 4 accumulators | 0.950786 | 1,293.7 ms | 1,523.5 | 比默认 AVX2 慢约 4% |
| Existing early-stop | 0.466464 | 256.7 ms | 7,677.3 | 速度达到目标但 recall 完全不可用 |
| Prefetch disabled | 0.950786 | 1,246.7 ms | 1,581.0 | 当前 prefetch 无稳定收益，也无明显损失 |

简单维度抽样也不可用：

| Distance score | Recall | Wall time | 结论 |
|---|---:|---:|---|
| Exact 768-D | 0.950786 | 1,523 ms | 对照 |
| Strided 384-D | 0.948300 | 1,783 ms | 更慢且低于目标 |
| Strided 192-D | 0.946372 | 1,698 ms | 更慢且低于目标 |
| Strided 96-D | 0.935515 | 1,757 ms | 更慢且 recall 明显下降 |

原因是 approximate loop 是 scalar strided access，而 exact L2 已使用连续 AVX2。该结果
否定的是“简单抽维”，不否定 SIMD scalar quantization、PQ/OPQ 或 exact bounded L2。

### Thread scaling

相同代码、相同 Lsearch 的稳态结果：

| Threads | Steady QPS | 相对 100 threads |
|---:|---:|---:|
| 36 | 1,296 | 0.79x |
| 72 | 1,483 | 0.90x |
| 100 | 1,645 | 1.00x |
| 144 | 1,563 | 0.95x |

100 threads 已是本机当前路径的最佳点，144 threads 因 SMT/带宽竞争反而下降。线程数
调优不能提供目标所需的数量级收益。

## Amdahl 上限

如果 entry、preexpand、regular 和其他 core 时间完全不变，只优化 49.33 ms 的 main
special-edge 阶段：

- 达到 3 倍总提速，main special-edge 必须从 49.33 ms 降到约 5.74 ms，即 `8.6x`。
- 达到 5 倍总提速不可能，因为其他阶段合计约 16.06 ms，已经超过 13.08 ms 的总预算。

即使 entry 和 preexpand 都先优化 2 倍：

- 3 倍目标仍要求 main special-edge 约 `3.8x`。
- 5 倍目标仍要求 main special-edge 约 `11.7x`。

所以必须组合多类优化，不能期待一个 kernel 或一个参数完成目标。

## 优化优先级

### P0：先消除确定性的软件浪费

#### 1. 用 reusable heap 替换 ordered-vector beam

- Expansion min-heap 取最近未展开节点。
- Bounded result max-heap 维护当前 top-L 上界。
- 用 active/generation 标记处理 heap 中的 stale entries。
- scratch 放进 `SearchCache`，跨 query 复用分配。
- 验证逐 query Recall 与现有路径一致，并单独统计 heap push/pop/stale 数量。

目标：将 `SpecialQueueShiftedCandidates` 从 413 万降到 0。预期是重要的 10%-30% 级
优化，但不应把它单独估计为 3 倍。

#### 2. 增加 exact bounded L2 / early abandon

当 beam 已满时，`queue.back().distance` 是单调不增的精确上界。按 32 或 64 维 AVX2
chunk 累加 L2；若 partial sum 已严格大于上界，可立即返回 reject，不读取剩余维度。
这不改变搜索结果，因为 L2 partial sum 是完整距离的严格下界。

必须先增加 `dimensions_evaluated` 统计，测量 19,278 个 rejected candidates 平均在多少
维退出。若大多数到 700 维才退出，收益有限；若在 200-400 维退出，收益会明显。

#### 3. 复用 query scratch，并缓存 label-route skeleton

当前每 query 会构造多个 vectors，以及 `vector<vector<SpecialCandidate>>` 的 per-block
临时结构。225 种 label set 的以下内容可以缓存：

- covered/free/frontier block IDs；
- block parent 和 lazy successor 列表；
- group entry point IDs 及其 state；
- query-independent block landmark IDs。

query 时只重新计算向量距离和 query-dependent block ranking。该项主要压缩 8.7 ms
entry phase 和分配长尾。

### P1：减少 special graph 的宽度

#### 4. 分离 free eligibility 与 activation

新增一个真正的 block-level router：

1. Trie 只生成合法 block frontier 和 terminal regular entries。
2. 在 block graph/landmark graph 上用低成本距离搜索 top-B promising blocks。
3. 只对 active blocks 注入并预展开 local entry。
4. child block 仍自动继承 free eligibility，但仅在 router/cross-edge 到达时 activation。
5. 未激活 blocks 保留 fallback，不能被语义上删除。

需要 sweep `B=4/8/16/32/64`，同时观察整体 recall 以及 `{1}`、`{2}`、`{20835}` 的
per-label recall。目标不是硬截断 block，而是用更好的 coarse navigation 让
`Recall >= 0.95` 在更小的 vector-level Lsearch 下实现。

#### 5. 降低 block graph degree 和 cross-block 冗余

当前 special graph 全局平均 degree 约 61，其中约 28% 是 inter-block edges。建议重建
`R=64/48/32/24` 的 intra graph，并对 cross edges 做以下 A/B：

- 每 source point 的 cross edges 改为少量 portal/bridge nodes；
- target child 已直接 seeded/activated 时，跳过指向该 child 的重复 inter edges；
- 给 intra/inter 分开预算，优先保留 Vamana robust-pruned intra neighbors；
- 统计 query-time intra/inter scanned、unique targets 和 recall contribution。

目标是在 Recall 不降的条件下把 special scans 从 9.70 万降到 4-6 万/query。

#### 6. 使用 compact CSR target-only adjacency

当前 `SpecialEdge` 在内存中包含 target、block ID 和 kind；free hot path 实际只需要 target。
可建立 query-optimized CSR：

- `uint32_t offsets[num_points+1]`；
- `uint32_t target_ids[num_edges]`；
- 如需 inter-edge 策略，单独保存 boundary 或 bitset。

这会把 hot adjacency payload 从约 12 bytes/edge 降到约 4 bytes/edge，并消除
`vector<vector<...>>` 的大量小 allocation 和 pointer chasing。它不是主向量流量，但有助于
cache/TLB 和 visited scan。

### P2：把完整向量计算降到 1 万次/query 以下

#### 7. SIMD SQ8/PQ coarse score + exact rerank

简单抽维失败后，应使用连续、紧凑、SIMD-friendly 的 representation：

- 每点 64-192 bytes SQ8/PQ code 作为 graph traversal score；
- 保留比当前更宽的 approximate candidate pool；
- 对最终候选或小的 rerank pool 计算 exact 768-D L2；
- 分别测 traversal recall、rerank recall 和 end-to-end QPS。

若要求完全保持现有搜索决策，可先做带保守误差界的 quantized lower bound，仅对可能进入
beam 的候选计算 exact L2。若允许只保证最终 `Recall >= 0.95`，PQ traversal + exact
rerank 的提速空间更大。

#### 8. 为高覆盖单标签建立 direct/coarse graph

单标签 query 占 81.3% 的 query 数和 87.4% 的 core time。可以按通用阈值选择高频或高
覆盖标签，为其构建 filter-specific Vamana/HNSW 或共享的 label landmark graph，避免在
几十个 block-local graphs 之间平铺搜索。该策略不应硬编码 `{1}` 等标签，而应由训练/构建
数据的 query frequency 和 posting size 决定。

此项最可能显著改善 `{1}`、`{2}` 等长尾，但还必须同时优化剩余通用路径，才能突破
Amdahl 限制。

### P3：系统层优化

#### 9. NUMA-aware storage 和线程 affinity

当前 1.8 GiB base vectors 由主线程 `aligned_alloc + ifstream::read` 首次触页，双路机器上
可能集中在一个 NUMA node；ThreadPool 没有 affinity。应测试：

- vectors/adjacency page interleave；
- 每 NUMA node 一份只读 base vectors；
- per-node query queues 和 SearchCache；
- 72 physical cores 与 SMT 的固定 affinity。

该项可能改善带宽利用和 p95，但当前 thread sweep 已表明它不会单独提供 3 倍。

## 建议的实施顺序与验收指标

### 第一阶段：不改索引语义

1. Heap candidate queue。
2. Exact bounded L2，并记录平均 evaluated dimensions。
3. SearchCache scratch reuse 和 label-route skeleton cache。
4. Compact CSR adjacency。

验收：Recall 与当前逐 query 一致；目标 `1.4-2.0x`，不承诺简单相乘。

### 第二阶段：改 block 路由与图拓扑

1. Lazy block activation / block-level router。
2. Intra degree sweep。
3. Sparse cross-block portals 和已激活 child 去重。
4. 在 Lsearch 500-2,000 上重新画 Recall-QPS Pareto。

验收目标：

- `SpecialFreeEdgesScanned <= 40,000-60,000/query`；
- `DistCalcs <= 10,000-15,000/query`；
- `{1}`、`{2}`、`{20835}` 不成为 recall 回退来源；
- 总 Recall `>= 0.95`。

### 第三阶段：压缩距离和单标签 fast path

1. SQ8/PQ traversal + exact rerank。
2. 高覆盖单标签 direct/coarse graph。
3. NUMA-aware deployment。

最终验收：

| 指标 | 3x 目标 | 5x 目标 |
|---|---:|---:|
| Recall | >= 0.95 | >= 0.95 |
| QPS | >= 4,933 | >= 8,222 |
| Batch wall time | <= 400 ms | <= 240 ms |
| Mean core budget | 约 <= 21.8 ms/query | 约 <= 13.1 ms/query |
| Ordered-vector shifts | 0 | 0 |

## 最终判断

当前方案已经解决了入口覆盖不足导致的 recall 问题，但代价是把几乎所有 free blocks 都
激活并推入高出度 special graph。现在的瓶颈不是“入口组仍然太多”这一单点，而是：

```text
全 free-block 预展开
  -> 约 9.7 万 special-edge scans/query
  -> 约 2.4 万新邻居
  -> 完整 768-D 距离
  -> 80% 被 beam 拒绝
  -> 成功插入仍触发 O(L) vector 搬移
```

要达到 3 倍，应先用 heap、exact bounded L2 和 compact scratch 消除软件浪费，再用 lazy
block activation 和更稀疏的 block graph 把高成本 free 搜索缩小。要稳定达到 5 倍，仅靠
现有 float32 graph traversal 很难，需要 SQ8/PQ 或 filter-specific coarse/direct graph
减少 768-D base-vector 流量。

报告日期：2026-08-11。
