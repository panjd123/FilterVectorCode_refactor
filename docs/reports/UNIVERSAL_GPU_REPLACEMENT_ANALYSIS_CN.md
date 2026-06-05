# UNG GPU 普遍替代方案分析

日期：2026-06-02

## 结论

当前最稳的 cross-edge 主线仍是 target-centric grouped fused topK。我们尝试把 source-centric no-lock exact search 推成更普遍的替代，但实验显示它还不能直接替代现有主线：

| 路径 | 数据/口径 | cross-edge | GPU kernel | 结论 |
|---|---:|---:|---:|---|
| target-centric fused | SIFT30 skip-additional 历史点 | 4918.9 ms | 2189.9 ms | 当前更稳主线 |
| source-centric TF32 WMMA | SIFT30 skip-additional 历史点 | 3021.9 ms | 1402.9 ms | 曾显示潜力，但需要错误检查复核 |
| source-centric CUDA-core id-only | SIFT30 skip-additional 新复测 | 5926.7 ms | 5525.5 ms | no-lock 但并行度不足 |

新复测 artifact：

`/home/graphdb/FilterVectorResultsRefactor/organized_benchmarks/ung_cross_edge_fused_20260602_091046`

关键日志：

```text
[cross_edges][source_exact] source_groups=54553 queries=984443 segments=381768
tiles=105649 padded_tiles=0 mode=cuda_core warps=8
prepare_h2d(ms)=21.4 desc_h2d(ms)=2.1 kernel(ms)=5525.5
d2h(ms)=1.0 writeback(ms)=13.3 id_only=1 total(ms)=5858.0
build_cross_edges_time,5926.67
```

## 已修复的问题

这轮修复了 source-centric id-only 输出边界的一个真实 bug：当调用方只需要 neighbor id 时，host 侧会把 `d_dis/h_dis` 置空，但 source CUDA kernel 仍然无条件写 `out_dist[...]`。这会触发 illegal memory access，之前缺少 stream 错误检查时容易被误判为 `kernel=0.0 ms` 的假加速。

修复内容：

- source CUDA-core kernel 写 `out_dist` 前检查非空。
- source TF32 WMMA kernel 写 `out_dist` 前检查非空。
- source path 增加 kernel launch 与 stream synchronize 错误检查。
- source id-only 路径不再分配和回传距离输出。
- `UNG_GPU_SOURCE_EXACT_MODE` 默认改为 `0`，即 CUDA-core exact；TF32 WMMA 仍可显式设为 `1`，但不作为默认安全路径。

## 为什么当前性能不佳

source-centric 的优势是每个 query 只输出一次 topK，天然避免 target-centric 的 qid lock/global merge 问题。但当前实现是单阶段 kernel：一个 query 或 16-query tile 在 kernel 内串行扫完该 source group 的所有 target segment。对于真实标签图，一个 source group 可能有多个 child/target segment，候选总量大时只有少量 CTA/warp 在做长循环，GPU 并行度被压低。

target-centric fused 虽然要处理重复 qid 和 merge，但它天然按 target group 展开，有更多 group/tile 级并行度。当前 Amazon/SIFT 这类 workload 下，增加并行度比减少输出 merge 更重要，所以 source CUDA-core no-lock 反而慢。

另一个问题是 source TF32 WMMA 为避免 CTA 内不同 source group 共享 B tile，需要 group padding；这会在小 source group 很多时制造大量 inactive tile。SIFT30 上 padding 曾把 tile 数量膨胀到约 47.7 万。我们尝试 per-warp shared B 取消 padding，但触发 illegal memory access，已回退，不作为当前结果。

## 真正可行的大幅优化方向

要成为普遍替代，source-centric 不能是一阶段串行 scan，而应重构为 two-stage source grouped GEMM + reduce：

1. **Stage A: parallel partial topK**
   把 workload 切成 `(source_query_tile, target_segment_tile)`，每个 CTA 只负责一个较小 target tile，在 GPU 上生成 partial topK。这样候选维度完全并行化，避免一个 warp 串行扫多个 segment。

2. **Stage B: per-source reduce topK**
   同一个 source query 的所有 partial topK 在 GPU 上二次归并。因为归并范围只在同一个 source group/query 内，不需要全局 qid lock。

3. **Flat output first**
   输出保持 `[num_points, topk]` flat id buffer，不直接写 `_graph->neighbors`。最后再由 CPU 一次性 append，或进一步把 `_graph` 后端改成 CSR/flat adjacency。

4. **Shape router**
   小 `nx`/小候选量继续走 target fused 或 CPU exact；中等候选量走 source two-stage；大 group graph 仍由 adaptive FastGrnnd/exact-anchor router 负责。所谓“普遍替代”应该是统一 router，而不是一个 kernel 覆盖所有形状。

## 当前主张边界

可以主张：

- target-centric grouped fused topK 是当前稳定 cross-edge 加速贡献。
- source-centric no-lock 是下一代候选设计，但当前单阶段实现不能替代 target fused。
- 已修复 source id-only 输出边界和错误检查，防止隐藏 kernel failure 进入实验表。
- 更现实的“普遍替代”不是单一 source-centric kernel，而是 target-centric batched descriptor + GPU global merge + flat-id output 的统一路由；它保持现有 cross-edge 语义，同时把输出边界从 `SearchQueue` 对象插入改成批量数组写回。

不能主张：

- source-centric 已经是普遍替代。
- source TF32 WMMA 已经可靠。
- 不能把 `kernel=0.0 ms` 的历史 smoke 当作有效加速结果。

## 2026-06-02 universal route 更新

这轮进一步确认：当前性能不佳的主因不是 H2D/D2H，而是**形状覆盖不完整导致大量 group 走旧 fallback**，以及 cross-edge 输出仍然经过 CPU `SearchQueue` 插入边界。source-centric 单阶段虽然去掉了 qid merge，但把一个 source group 的多个 target segment 串行扫完，kernel 本体反而慢；因此它不是当前可落地的普遍替代。

新的可落地路线是：

1. `UNG_UNIVERSAL_GPU=1` 默认启用 `UNG_GPU_DB_NOSPLIT=1` 和 flat-id writeback。
2. flat/id 输出请求会自动启用 double-buffer 路径内部的 GPU global merge，不再要求用户额外设置 `UNG_GPU_DB_GLOBAL_MERGE=1`。
3. universal 模式下 `UNG_GPU_DB_LARGE_MAX_NX` 默认覆盖到上限，使所有 target group 都优先走 double-buffer descriptor batch；超大组仍可通过环境变量重新切回 fallback。
4. 输出先形成 `[num_points, num_cross_edges]` flat id buffer，然后 CPU 只做一次线性 append 到 `_graph->neighbors`。

SIFT30 skip-additional smoke：

| 路径 | 关键设置 | cross-edge | generate wall | GPU kernel 累计 | D2H | merge_cross | 备注 |
|---|---|---:|---:|---:|---:|---:|---|
| 历史 target fused | grouped fused, SearchQueue 输出 | 4918.9 ms | - | 2189.9 ms | 35.7 ms | - | 当前稳定旧主线 |
| source exact CUDA-core | source-centric id-only | 5926.7 ms | 5890.3 ms | 5525.5 ms | 1.0 ms | 14.1 ms | no-lock 但 kernel 串行扫描过重 |
| universal flat fallback | `UNG_UNIVERSAL_GPU=1`, `DB_LARGE_MAX_NX=4096` | 3864.6 ms | 3839.0 ms | 2384.0 ms | 1.9 ms | 6.1 ms | flat 输出生效，但 20 个大组触发 fallback |
| universal all-DB | `UNG_UNIVERSAL_GPU=1`, `DB_LARGE_MAX_NX=1048576` | 3180.6 ms | 3142.9 ms | 3870.4 ms | 2.1 ms | 15.0 ms | 94 chunks 全部走 double-buffer；kernel 为跨流累计，墙钟更低 |

artifact：

`/home/graphdb/FilterVectorResultsRefactor/organized_benchmarks/universal_all_db_sift30_20260602_092318`

关键日志：

```text
[cross_edges] double_buffer chunks=94 groups=69146 queries=23901768 chunk_q=262144
group_desc=13634576 tile_desc=642082 H2D(ms)=30.8 Kernel(ms)=3870.4
D2H(ms)=2.1 writeback(ms)=15.5
[cross_edges] breakdown generate(ms)=3142.9 additional(ms)=0.0 add_offset(ms)=22.3
merge_cross(ms)=15.0 merge_add(ms)=0.1 id_vector_active=0 flat_id_active=1
build_cross_edges_time,3180.63
```

相对历史 target fused，SIFT30 cross-edge 从 `4918.9 ms` 降到 `3180.6 ms`，约 `1.55x`；相对 source exact negative path 约 `1.86x`。这不是最终上限，因为 kernel 累计时间仍高，说明 DB tile kernel 的大组路径还有继续优化空间；但它已经比单阶段 source-centric 更符合“普遍替代”的工程形态。

## 2026-06-02 Amazon x200 full-quality smoke

Reviewer 继续攻击：SIFT30 skip-additional 不能说明 full-quality 系统成立，因为真实 build 还要跑 CPU Vamana `additional_edges`，并且 search recall 需要闭环验证。我们补了 Amazon 1% x200 coverage query 的最小 full-quality smoke：

- group graph 固定为 CPU Vamana，避免把 group graph 质量变化混入 cross-edge 对比。
- `additional_edges_impl=cpu_vamana`，不跳过 additional edges。
- cross-edge 使用 `UNG_UNIVERSAL_GPU=1`，即 double-buffer descriptor batch + GPU global merge + flat-id output。
- build 后复用 index 补跑 L1000/L5000 repeat=3。

artifact：

`/home/graphdb/fv_runs/end_to_end_recall_ab_20260602_092928`

| 路径 | additional_edges | cross-edge | generate | additional | GPU kernel 累计 | D2H | merge_cross | L1000/L5000 recall |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| x200 partial DB target route | CPU Vamana | 2657.23 ms | - | - | - | - | - | 0.866 / 0.907 |
| x200 universal flat all-DB | CPU Vamana | 2494.21 ms | 1636.2 ms | 843.5 ms | 1887.5 ms | 1.2 ms | 0.7 ms | 0.871 / 0.911 |

关键日志：

```text
[cross_edges] double_buffer chunks=28 groups=5195 queries=7292200 chunk_q=262144
group_desc=0 tile_desc=456807 H2D(ms)=2.9 Kernel(ms)=1887.5
D2H(ms)=1.2 writeback(ms)=19.2
[cross_edges] breakdown generate(ms)=1636.2 additional(ms)=843.5 add_offset(ms)=13.1
merge_cross(ms)=0.7 merge_add(ms)=0.5 id_vector_active=0 flat_id_active=1
build_cross_edges_time,2494.21
L1000/L5000 repeat=3 recall: 0.871 / 0.911
```

这组结果回答了一个关键审稿问题：universal route 不只是 skip-additional smoke，它能在 CPU Vamana additional_edges 的 full-quality 口径下跑通，并且 cross-edge 低于已有 x200 partial double-buffer 对照。仍然不能把它写成全局完成的普遍替代，因为：

- 当前只补了 Amazon 1% x200，尚未覆盖 x100、10%x40、x400 和真实多标签。
- `additional_edges` 仍在 CPU 侧执行，`additional(ms)=843.5` 已经是 x200 cross-edge 的显著部分。
- 最终图仍回填 CPU-compatible `_graph->neighbors`，还不是 GraphView/CSR 后端。

## 2026-06-02 Amazon x100 full-quality boundary

Reviewer 再追问：x200 正结果是否只是某个 group-size 分布的偶然点？我们补了 Amazon 1% x100 coverage query 的同口径 full-quality smoke：

- group graph 固定为 CPU Vamana。
- `additional_edges_impl=cpu_vamana`。
- cross-edge 使用 `UNG_UNIVERSAL_GPU=1`。
- build 后复用 index 跑 L100/L500/L1000 repeat=3。

artifact：

`/home/graphdb/fv_runs/end_to_end_recall_ab_20260602_093357`

| 路径 | additional_edges | Index | cross-edge | generate | additional | GPU kernel 累计 | D2H | merge_cross | L100/L500/L1000 recall |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| x100 historical CPU-group target route | CPU Vamana | 6863.47 ms | 1732.42 ms | - | - | 470.4 ms | 9.0 ms | - | 0.825 / 0.868 / 0.890667 |
| x100 universal flat all-DB | CPU Vamana | 7461.27 ms | 1689.40 ms | 1005.2 ms | 675.3 ms | 872.0 ms | 0.6 ms | 0.7 ms | 0.826 / 0.868 / 0.891 |

关键日志：

```text
[cross_edges] double_buffer chunks=14 groups=5195 queries=3646100 chunk_q=262144
group_desc=3568700 tile_desc=4888 H2D(ms)=3.6 Kernel(ms)=872.0
D2H(ms)=0.6 writeback(ms)=6.8
[cross_edges] breakdown generate(ms)=1005.2 additional(ms)=675.3 add_offset(ms)=7.8
merge_cross(ms)=0.7 merge_add(ms)=0.2 id_vector_active=0 flat_id_active=1
build_cross_edges_time,1689.4
L100/L500/L1000 repeat=3 recall: 0.826 / 0.868 / 0.891
```

这组结果是边界而不是强正结果：

- cross-edge 小幅降低 `1732.42 -> 1689.40 ms`，flat-id 使 D2H/merge 明显下降。
- recall 没有下降，说明 full-quality 兼容性成立。
- 但 Index `6863.47 -> 7461.27 ms` 变慢，主要来自 label processing / host registration 波动和 universal all-DB kernel 累计更高；search latency 也从约 `390 ms` 变到约 `700 ms`，不能写成端到端更快。

因此 x100 支持的写法是：universal route 在第二个 full-quality workload 上跑通且保持 recall，但收益依赖 group-size 分布和外围波动；它补强“不是 x200 孤例”的兼容性证据，同时限制我们不能把 universal flat route 写成无条件端到端加速。换句话说，不能写成无条件端到端加速，只能写成 full-quality 兼容性和 cross-edge/output-boundary 的边界证据。
