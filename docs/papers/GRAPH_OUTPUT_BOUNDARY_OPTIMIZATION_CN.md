# Graph 输出边界优化记录

本文记录一个新的系统瓶颈判断：当前 GPU 组内图和 GPU cross-edge 已经能把核心计算压到毫秒/秒级，但最终仍要落回 `Graph::neighbors` 这一 CPU-compatible 图对象。在旧实现中它是 `std::vector<IdxType>[N]`，在 many-group workload 上会触发大量 per-node 小分配、逐点填充、offset 转换和追加边 merge，成为“普遍替代 CPU Vamana”的主要阻力。最新实现把它替换为 small-buffer `NeighborList`，在不改变大多数调用接口的前提下显著降低 allocator 开销。

## 当前问题

当前主流程有三个耦合点：

1. GPU group graph 输出 compact rows 后，CPU 逐点写入 `graph->neighbors[local_id]`。
2. `add_offset_for_uni_nav_graph()` 再逐点把局部 id 转成全局 id。
3. cross-edge 和 additional-edge 继续向同一个 `std::vector` 追加边。

这让 GPU kernel 的收益被 host 图物化抵消。直接 H2D/packed exact 已经证明输入侧搬运可以压低；warp exact/fill-thread tuning 的负面结果说明，继续微调小 kernel 或填充线程数不是根本解。

## 低风险优化一: reserve 预分配

新增 `reserve_graph_neighbor_capacity()`。未显式设置 `UNG_GRAPH_RESERVE_CAPACITY` 时走 auto 策略：点数大、group 数多、或小组 fallback 点数占比高时自动打开；显式设置 `UNG_GRAPH_RESERVE_CAPACITY=0/1` 可覆盖 auto。它在 group graph 构建前给每个点的邻接表预留容量：

- 组内边容量约为 `min(max_degree, group_size-1)`；
- cross-edge 预留 `_num_cross_edges`；
- additional slack 默认为 `_num_cross_edges`，可由 `UNG_GRAPH_RESERVE_ADDITIONAL_SLACK` 调整；
- `UNG_GRAPH_RESERVE_HARD_CAP` 提供上限保护。

该优化不改变图语义，只改变 `std::vector` 分配时机。`UNG_GRAPH_RESERVE_CAPACITY=0` 可强制回退。

## reserve A/B

配置：`AdaptiveCuda`，`UNG_GROUP_GRAPH_BOUNDED_COMPLETE=1`，`UNG_ADAPTIVE_EXACT_MAX_NX=512`，`num_threads=128`，`num_cross_edges=6`。

运行目录：`/home/graphdb/fv_runs/graph_reserve_ab_20260602_074459`

| 配置 | Index ms | Build graph ms | Cross edges ms | tagore fill ms | fallback wall ms | cross merge ms |
|---|---:|---:|---:|---:|---:|---:|
| reserve=0 | 39746.1 | 14238.1 | 14761.0 | 823.5 | 11844.6 | 1677.1 |
| reserve=1 | 24704.5 | 1662.7 | 9994.0 | 23.9 | 520.2 | 2.3 |

额外代价：reserve 阶段预留 `106,031,200` 条边容量，耗时 `6487.3 ms`，计入 label processing 阶段。因此总收益不是 `build_graph_time` 的 8.6x，而是端到端 Index 的 `1.61x`。

Amazon 1%x200 的补充 A/B 说明 reserve 对中组 exact-anchor 也不是纯负面：auto 关闭时 Index `16895.4 ms`，`tagore_fill_time=2497.0 ms`，`cross merge=572.0 ms`；强制开启时 Index `15436.9 ms`，reserve 自身 `2958.7 ms`，但 `tagore_fill_time` 降到 `150.9 ms`，`cross merge` 降到 `13.8 ms`。因此 auto 策略加入 `UNG_GRAPH_RESERVE_AUTO_MIN_POINTS=500000`，避免 x200 这类大点数 workload 默认漏开 reserve。

## 低风险优化二: small-buffer NeighborList

reserve A/B 证明问题来自 per-node `std::vector` 分配和扩容，但 reserve 本身仍会把 allocator 成本前置到 label processing。为进一步降低 CPU-compatible 输出边界成本，`Graph::neighbors` 已从 `std::vector<IdxType>*` 改成接口兼容的 `NeighborList*`。`NeighborList` 在对象内保留 64 个 `IdxType` 的 inline buffer；只有超过 64 条邻居时才走堆分配。当前论文主参数通常是 `max_degree=32`、`num_cross_edges=6`，加少量 additional/slack 后多数点可以完全留在对象内。

这不是 CSR/GraphView 的最终形态，但它有两个优点：

- 旧 Vamana、search、cross-edge merge 大多数代码仍通过 `begin/end/size/resize/reserve/emplace_back/insert` 访问邻居，不需要改搜索语义；
- 对 small/medium group、bounded-complete fallback、GPU graph fill 和 cross merge 都同时生效，因此比只优化某一个 kernel 更接近普遍系统优化。

运行目录：

```text
/home/graphdb/fv_runs/neighborlist_x200_autoreserve_20260602_081303
/home/graphdb/fv_runs/neighborlist_x200_noreserve_20260602_081201
/home/graphdb/fv_runs/neighborlist_10px40_autoreserve_20260602_081403
/home/graphdb/fv_runs/neighborlist48_x200_autoreserve_20260602_081636
```

| Workload / Variant | Index ms | Build graph ms | Cross ms | reserve ms | tagore fill ms | fallback wall ms | cross merge ms |
|---|---:|---:|---:|---:|---:|---:|---:|
| x200 old auto-reserve (`std::vector`) | 16395.0 | 5940.5 | 3685.4 | 2856.2 | 224.8 | 16.7 | 0.9 |
| x200 NeighborList64 auto-reserve | 15716.6 | 6984.6 | 3905.6 | 25.4 | 14.7 | 6.5 | 1.5 |
| x200 NeighborList64 reserve=0 | 15891.0 | 7518.4 | 4150.1 | 0.0 | 142.5 | 18.5 | 1.0 |
| 10%x40 old reserve=1 (`std::vector`) | 24704.5 | 1662.7 | 9994.0 | 6487.3 | 23.9 | 520.2 | 2.3 |
| 10%x40 NeighborList64 auto-reserve | 26167.8 | 1981.3 | 14230.6 | 21.2 | 7.6 | 435.9 | 12.0 |
| x200 NeighborList48 auto-reserve | 16015.4 | 7726.6 | 3975.9 | 25.1 | 202.5 | 5.6 | 35.4 |

解读必须保守：

- x200 的 `reserve_ms` 从 `2856.2 ms` 降到 `25.4 ms`，`tagore_fill` 从 `224.8 ms` 降到 `14.7 ms`，说明 old reserve 的秒级代价确实主要来自堆分配；
- 10%x40 的 `reserve_ms` 从 `6487.3 ms` 降到 `21.2 ms`，`tagore_fill` 从 `23.9 ms` 进一步降到 `7.6 ms`，fallback wall 也从 `520.2 ms` 降到 `435.9 ms`；
- 10%x40 的总 Index 没有赢，主要因为该 run 的 cross-edge `prepare_all H2D` 抖动到 `4974.9 ms`，旧 reserve-on run 是 `810.4 ms`，因此不能把这组写成稳定端到端加速；
- 48-inline 是负例：虽然 reserve 仍只有 `25.1 ms`，但 x200 `tagore_fill` 退化到 `202.5 ms`，`merge_cross` 退化到 `35.4 ms`。当前保留 64-inline，不把容量选择写成算法贡献。

## 结论

reserve 和 NeighborList A/B 共同证明当前性能不佳的关键原因之一是 Graph 输出边界，而不是 GPU 算力不足：

- `tagore_fill_time` 从 `823.5 ms` 降到 `23.9 ms`，说明大量时间来自 per-node vector 分配和扩容；
- bounded complete fallback 从 `11844.6 ms` 降到 `520.2 ms`，说明小组图物化也主要被分配拖慢；
- cross merge 从 `1677.1 ms` 降到 `2.3 ms`，说明后续追加边同样受容量不足影响；
- small-buffer 后 reserve 本身从秒级降到几十毫秒，说明它吸收了 reserve 作为 stopgap 的主要 allocator 成本；
- 但最终图仍是 CPU-compatible per-node object array，不是 GPU-native flat adjacency，因此仍不是普遍替代的最终形态。

## 本轮结构性快路径: direct-global + flat-id writeback

为了逼近“普遍替代”而不是继续微调单个 kernel，新增一条更明确的输出边界快路径：

- `UNG_INTRA_GLOBAL_IDS=-1/0/1`：auto/关闭/开启。auto 仅在 GPU/Adaptive 组内图、cross-edge 不依赖 CPU Vamana、additional-edge 不依赖 CPU Vamana 时启用。
- 启用后，组内图填充阶段直接写全局邻居 id，`add_offset_for_uni_nav_graph()` 变成 no-op。
- 如果用户强行把 direct-global 与 CPU Vamana cross/additional 混用，构建阶段直接报错。原因是 CPU Vamana 的组内搜索要求邻居为 local id。
- benchmark 脚本补充透传 `UNG_GPU_FLAT_ID_WRITEBACK`，允许 GPU cross-edge 直接输出平铺 `N * topk` id 数组，跳过 `SearchQueue[N]` 对象物化。

smoke 配置：Amazon 1%x100 coverage query，`AdaptiveCuda`，`additional_edges=skip`，`bounded_complete=1`，`fallback_impl=0`，`reserve=0`，`Lsearch=100`，repeat=1。该实验只用于验证输出边界优化，不作为 full-quality recall 主结果。

| Variant | Index ms | Group ms | Cross ms | GPU kernel ms | D2H ms | add_offset ms | merge_cross ms | Recall |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| local-id + SearchQueue | 4693.59 | 493.05 | 1650.95 | 656.9 | 11.1 | 2.5 | 17.2 | 0.816 |
| direct-global + SearchQueue | 6682.97 | 521.68 | 3530.31 | 658.5 | 11.3 | 0.0 | 21.0 | 0.816 |
| direct-global + flat-id | 3986.41 | 498.14 | 707.83 | 255.1 | 0.6 | 0.0 | 0.5 | 0.816 |

解释要保守：

- direct-global 本身只稳定消掉 offset pass；在 NeighborList64 后，x100 上 offset 只有约 2.5ms，因此不能把它写成主要加速来源。
- flat-id writeback 是本轮更有效的结构性优化：它避免 GPU topK 结果落到 `SearchQueue` 对象，并使 merge 变成顺序读取平铺 id。这个结果支持“输出边界比单个 GPU kernel 更关键”的判断。
- direct-global + SearchQueue 那组出现较大 cross generate 抖动，不能作为负面算法结论；同一配置下 kernel event 基本一致，差异主要来自外围调度/host 路径。
- 该路径与 CPU Vamana additional 不兼容，full-quality 版本若仍保留 CPU Vamana repair，就不能默认启用 direct-global。更普遍的方案仍需要 GraphView/CSR，让构建和查询都能读取 flat segment，而不是依赖 per-node mutable object。

## 负结果: additional_edges direct append

一个自然的 reviewer 建议是：既然 `additional_edges` 最后只是追加到 `_graph->neighbors[from_id]`，能否跳过 `vector<vector<pair>> additional_edges` 的中间容器，直接在生成阶段 append 到最终图。我们实现了 `UNG_ADDITIONAL_DIRECT_APPEND=1` 做探索，并在 x400 full-quality 口径上测试。

运行目录：

```text
/home/graphdb/fv_runs/additional_direct_x400_on_20260602_083520
/home/graphdb/fv_runs/additional_direct_x400_lock_on_20260602_083735
```

结果是负面的：两个 run 的 build log 都停在 GPU cross-edge `batched_search end` 之后，没有打印 `[cross_edges] breakdown`、`Index time`，`summary.csv` 也只有表头。lock-guarded run 的 env 明确包含 `UNG_ADDITIONAL_DIRECT_APPEND=1`，说明即使给每个点的邻接表追加加锁，直接在 OpenMP additional 阶段 mutate CPU-compatible `Graph::neighbors` 仍不是可靠路径。

因此当前代码把该 shortcut 限制为只允许 `additional_edges_impl=CpuExactScan` 的实验路径；如果用户在 CPU Vamana additional 下请求它，会打印：

```text
[cross_edges] additional_direct_append disabled: supported only for cpu_exact additional_edges
```

这个负结果支持下一版设计：additional/cross/group graph 不应在生成阶段随机修改 per-node mutable object，而应先写入独立 flat segment 或 staged edge buffer，再由 GraphView/CSR 统一暴露给 search。否则锁、迭代器生命周期、Vamana local/global id 语义和并行追加顺序都会继续污染性能和稳定性结论。

## 下一版结构性方案

要把方法写成更普遍的 GPU backend，而不是 workload-specific patch，需要把输出边界从 `std::vector[N]` 改成 GraphView：

| 层 | 设计 |
|---|---|
| Build 输出 | GPU group graph / cross-edge 直接写 `offsets + edges` 或分段 packed edge buffer |
| Search 读取 | `neighbors(point_id) -> span`，旧 `std::vector` 和新 CSR 都实现同一 accessor |
| 小组 fallback | 对 bounded complete 使用 implicit ring neighbor view，避免物化 `nx * degree` |
| 追加边 | cross/additional edges 作为独立 segment，查询时按 segment 拼接；最终保存时可选择 materialize |
| 兼容边界 | CPU Vamana 和旧保存格式保留 fallback，论文实现走 CSR/GraphView 快路径 |

需要补的验证：

1. CPU graph import 到 CSR 后，edge count、degree distribution、recall 与旧图一致。
2. optimized route 用 CSR/implicit small-group 后复测 Amazon 10%x40、1%x200、1%x400。
3. 单独汇报 Graph materialization / reserve / CSR build / search latency，避免只报 Index time。

当前不能声称“无损普遍替代 CPU Vamana”。更准确的说法是：我们已经证明主要剩余瓶颈在 Graph 输出边界，并给出 reserve 和 NeighborList small-buffer 两级低风险优化；下一步需要 GraphView/CSR 才能把 GPU backend 做成更普遍的替代。
