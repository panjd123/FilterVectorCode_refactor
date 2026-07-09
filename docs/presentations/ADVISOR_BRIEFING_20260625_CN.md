# 导师汇报：UNG GPU 优化工作进展与现状

汇报日期：2026-06-26

## 0. 我现在会怎么汇报这项工作

我们这段时间不是做出了一个“GPU 版本全面替代 CPU Vamana”的单点优化，而是逐步把 UNG 的构建和查询拆成几个真实瓶颈，并分别找到了适合 GPU 化或不适合 GPU 化的边界：

```text
1. cross-edge：可以 GPU 化，核心贡献是 grouped fused topK / universal flat double-buffer。
2. group graph：不能简单 GPU 化，需要按 group size 和图质量风险做 router。
3. query entry scan：可以批处理 GPU 化，但目前还是独立 benchmark，尚未正式接入 search。
4. output boundary：CPU-compatible Graph/NeighborList 仍是 Amdahl 瓶颈，flat/CSR 还没完成。
5. 100%x40：已经能跑通关键路径，但大图 build/query 暴露了新的系统瓶颈。
```

我认为目前最成熟、最能写成论文贡献的是 **cross-edge fused topK + workload-aware group graph router + query-entry correct-cover scan 的系统化分析**。但论文主张必须保守：现在还不能说“普遍替代 CPU”，只能说“在特定 group workload 上取得可解释的 speed/quality Pareto，并明确了剩余系统瓶颈”。

代码结构方面，我会明确说明：目前配置层已经收敛，`UngBuildConfig` 能表达 CPU、naive GPU、paper fused、adaptive group graph 等主口径；cross-edge 已拆出 descriptor/planning/dispatch/output helper，C++ cross-edge orchestration 已拆到 `uni_nav_graph_cross_edges.cpp`，CPU/GPU group graph adapter 已拆到 `uni_nav_graph_group_graph.cpp`，search/query 执行路径已拆到 `uni_nav_graph_search.cpp`，LNG/trie/coverage 已拆到 `uni_nav_graph_label_graph.cpp`，save/load/statistics 已拆到 `uni_nav_graph_io.cpp`，vector-attribute graph 已拆到 `uni_nav_graph_vector_attr.cpp`，legacy ACORN augment 已拆到 `uni_nav_graph_acorn_augment.cpp`，profiling helper 已拆到 `ung_prof_log.{h,cpp}`。但实现层还没完全清爽：`uni_nav_graph.h` 仍是较大的 facade header，`gpu_gemm_topk.cu` 仍保留 cross-edge 顶层 CUDA route，search/group/cross backend 还没有完全对象化。详细审计见 `docs/reports/CODE_STRUCTURE_REVIEW_20260625_CN.md`，方法注册表见 `docs/runbooks/UNG_METHOD_REGISTRY_CN.md`。

## 1. 总体进度表

| 工作线 | 当前状态 | 代表结果 | 我对结果的判断 |
|---|---|---|---|
| cross-edge GPU fused topK | 比较成熟 | Amazon 1%x100：相对 CPU exact `2.82x`，相对 cuVS per-group `4.39x`，相对 SGEMM+topK `1.64x` | 可以作为核心正结果 |
| universal flat double-buffer | 可作为当前 cross-edge 工程主线 | Amazon 1%x200 full-quality cross `2494.21 ms`，L1000/L5000 `0.871/0.911` | x200 是正结果，x100 是 boundary |
| group graph router | 有正结果，但不是无条件替代 | x200 packed exact-anchor group `2.22x`；10%x40 full-quality Index `1.53x`、group `3.54x` | 可以写 workload-aware router，不能写无损替代 |
| x400 quality route | 有 Pareto 点 | reverse-tail+repair Index `33044.9 ms` vs CPU `36257.3 ms`，L5000 `0.9659` vs CPU `0.965` | 说明质量修复方向有效，但还需参数扫 |
| query entry scan | 阶段加速强，但未端到端接入 | 100%x40、`nq=10240`：GPU correct-cover `59.83 ms`，相对 CPU scan `19.85x` | 可以写入口阶段加速，不能写端到端 query 加速 |
| output boundary | 已定位，部分止血 | NeighborList64 把 x200 reserve `2856.2 -> 25.4 ms`、fill `224.8 -> 14.7 ms` | 证明 CPU graph materialization 是瓶颈，但 CSR 还未完成 |
| 100%x40 scale | 可完成但慢 | strict skip-add build Index `945683 ms`，cross generate `697.1s`，`pack_q=450.3s` | 只能写 scalability/boundary，不能写加速 |

## 2. 第一条工作线：cross-edge 构建

### 2.1 做了什么

UNG 的 cross-edge 不是一个大 GEMM，而是大量不规则 group-pair topK：

```text
for each target group:
  for each source/parent group:
    source points as Q
    target points as X
    compute topK edges
```

逐组调用 CPU、cuVS 或 cuBLAS 都会被外围开销拖慢：group API 调用、host pack、H2D/D2H、距离矩阵物化、host merge、SearchQueue 写回。

我们做的核心是：

```text
descriptor batching
+ fused distance/topK
+ double-buffer
+ flat-id / universal route
+ partial split，避免少数大组拖累整批
```

### 2.2 效果

| baseline | Cross ms | 我们 Cross ms | 加速比 | 说明 |
|---|---:|---:|---:|---|
| CPU Vamana cross | `18735.8` | `848.9` | `22.07x` | 原始慢 baseline |
| CPU exact scan 128T | `2389.8` | `848.9` | `2.82x` | 强 CPU baseline |
| cuVS per-group brute force | `3730.5` | `848.9` | `4.39x` | 证明逐组调用库外围重 |
| SGEMM + topK | `1395.9` | `848.9` | `1.64x` | 证明 fused topK 减少中间结果和独立 topK pass |

Amazon 1%x200 full-quality 中，universal flat all-DB 可以接上 CPU Vamana `additional_edges`，cross 为 `2494.21 ms`，L1000/L5000 recall 为 `0.871/0.911`。这说明它不是只在跳过补边的 smoke test 中有效。

### 2.3 我会怎么解释

这条线最像论文贡献：我们不是声称自研 GEMM 普遍快于 cuBLAS/cuVS，而是说 **UNG cross-edge 的单位任务太碎，库调用边界和输出边界主导了成本；把 group-pair topK 合成 fused batch 后可以明显降低系统开销**。

### 2.4 还缺什么

- `additional_edges` 仍多由 CPU Vamana 完成，full-quality 下可能成为新瓶颈。
- x400/100%x40 还不能说已经解决，大图上的 Q 展开和输出边界仍很重。
- source-centric no-lock 方向合理，但当前 single-stage kernel 是负结果，需要 two-stage source grouped GEMM + reduce。

## 3. 第二条工作线：组内 group graph 构建

### 3.1 做了什么

最初想法是用 Tagore / GPU NN-Descent 替代 CPU Vamana。实验后发现不成立：如果照搬 Tagore 重 prune，会慢；如果只保留轻量 topK，又可能损伤 UNG search 的可导航性。

最终收敛成 workload-aware router：

```text
small group  -> CPU Vamana 或 bounded-complete fallback
medium group -> packed exact-anchor
large group  -> FastGrnndCuda + reverse-tail / repair
```

### 3.2 为什么比 Tagore 更适合 UNG

| 问题 | Tagore / naive GPU 的表现 | 我们的处理 |
|---|---|---|
| 重 prune 太慢 | x200 heavy prune `prune=19218.3 ms`，group graph 慢于 CPU | 中组不用重 prune，改 packed exact-anchor / light prune |
| pure topK 导航性弱 | 早期 exact 低 recall 曾误导判断，但后续发现是 cross 口径混杂 | 不再把 exact-anchor 判死；用 anchor/tail 保留导航边 |
| 大组低出度/弱连通 | light prune x400 recall 掉约 `0.008` | reverse-tail + repair 补反向和尾部多样性 |
| 小组 GPU 不划算 | 10%x40 小组全进 GPU exact 后被 pack/H2D/fill 吃掉收益 | 小组 fallback 或 bounded-complete，减少图物化 |

### 3.3 效果

| workload | 方法 | Before | After | 加速比 | 质量 |
|---|---|---:|---:|---:|---|
| Amazon 1%x100 full-quality | adaptive CUDA | Index `6863.47` | `5450.00` | `1.26x` | L100/L500/L1000 `0.829/0.870/0.895` |
| Amazon 1%x200 full-quality | packed exact-anchor | Index `17596.80` | `11043.50` | `1.59x` | L1000/L5000 `0.869/0.908` |
| Amazon 1%x400 coverage | packed exact / quality route | Index `36257.30` | `28436.30` | `1.27x` | L1000/L5000 `0.945/0.967` |
| Amazon 10%x40 full-quality | FastGrnndCuda + fallback | Index `36053.30` | `23497.40` | `1.53x` | L100/L500/L1000 不降 |

### 3.4 我会怎么解释

这条线的核心不是“我们有一个更快的 GPU 图构建算法”，而是：

```text
不同 group size 的最优策略不同。
UNG group graph 需要在 build time、导航性和 CPU-compatible output 之间折中。
```

packed exact-anchor 的价值是中组质量稳定且构建快；FastGrnndCuda/reverse-tail 的价值是大组避免 exact all-pairs，同时用 tail/reverse 修复 light prune 的图结构问题；小组则不值得强行 GPU 化。

### 3.5 还缺什么

- x400 只证明 reverse-tail+repair 是 Pareto 点，还需要更多参数扫。
- packed exact-anchor 主要在 Amazon repeat workload 上证据最强，需要真实多标签或 CelebA full-quality A/B。
- router 的阈值目前偏工程经验，需要用 group distribution 和成本模型解释。

## 4. 第三条工作线：query entry scan

### 4.1 做了什么

原始 CPU 查询入口组查找要找所有满足：

```text
query_labels subset labels(group)
```

的 group，并尽量做 exact minimal 剪枝。这个过程在 many-group workload 上很慢。

我们提出 GPU correct-cover：

```text
C = all matching groups
F = capped frontier groups within label-size window
Covered = descendants(F)
Output = F union (C - Covered)
```

语义是：保证覆盖所有真实候选 group，但允许冗余入口组。

### 4.2 效果

Amazon 100%x40、`nq=10240`：

| 方法 | total | avg output groups | QPS | 相对 CPU scan |
|---|---:|---:|---:|---:|
| CPU scan 128T | `1187.56 ms` | `23332.2` | `8623` | `1.00x` |
| CPU exact minimal | `4569.83 ms` | `616.24` | `2241` | `0.26x` |
| GPU correct-cover d1 cap8192 | `59.8294 ms` | `2184.26` | `171153` | `19.85x` |

10%x40 的已有 query CSV 显示，CPU entry scan 占 query latency 约 `48~51%`。如果只替换 entry 阶段且后续 search 不变，端到端上界约 `1.9~2.0x`。

### 4.3 当前限制

这还不是正式端到端 query 加速，因为 `gpu_cover_frontier` 尚未接入 `search_UNG_index`。而且 GPU 输出的入口组比 CPU exact minimal 多，可能拖慢后续：

```text
entry groups -> entry points -> iterate_to_fixed_point()
```

所以这条线现在应写成“入口组阶段加速”和“端到端上界潜力”，不能写成正式端到端 query speedup。

## 5. 第四条工作线：output boundary

### 5.1 发现

很多时候 GPU kernel 不是主瓶颈。GPU 结果最后仍要回填到 CPU-compatible graph：

```text
Graph::neighbors / NeighborList / SearchQueue / additional_edges
```

这会带来大量 per-node object 写入、allocator、merge 和 D2H/host materialization 开销。

### 5.2 已做的止血优化

| 优化 | 效果 | 判断 |
|---|---|---|
| direct-H2D | x200 exact-anchor host pack `4140.61 -> 0.04 ms` | 输入侧 pack 已解决 |
| compact D2H | 降低写回 padding | 有帮助，但不是主瓶颈 |
| graph reserve | 10%x40 fill/fallback/merge 大幅下降，但 reserve 自身重 | 证明 allocator 是瓶颈 |
| NeighborList64 | x200 reserve `2856.2 -> 25.4 ms`，fill `224.8 -> 14.7 ms` | 当前低风险保留方案 |
| flat-id smoke | x100 跳过补边口径 cross `1650.95 -> 707.83 ms` | 证明 flat 输出方向正确，但不是 full-quality |

### 5.3 还缺什么

真正要继续提升，需要从 CPU-compatible per-node graph 过渡到：

```text
flat adjacency / CSR / GraphView
```

并让 search 和 additional_edges 能直接消费这个布局。否则 GPU 构建越快，CPU 输出边界越会成为 Amdahl 限制。

## 6. 100%x40 大规模实验现状

### 6.1 已解决

- `Storage` 中 32-bit offset / byte-count 溢出已修。
- resident all-X GPU cache 需要约 `71.5GB`，48GB A6000 OOM；用 `UNG_GPU_X_STREAMING=1` 可以绕过。
- Amazon 100%x40 strict skip-additional build 已完成。

### 6.2 结果

| 指标 | 数值 |
|---|---:|
| points | `23,284,680` |
| groups | `482,387` |
| Index | `945683 ms` |
| cross generate | `697114.7 ms` |
| pack_q | `450338.9 ms` |
| GPU kernel | `72749.8 ms` |

### 6.3 判断

100%x40 目前是 scalability / boundary result，不是加速结果。它说明我们已经能越过显存 OOM 和数据规模边界，但下一步必须解决 Q 侧重复 host pack、source-centric streaming 或 Q-side device cache。

## 7. 我认为现在论文/汇报应采用的主张

可以讲：

```text
UNG GPU optimization should be treated as an irregular group workload.
Grouped fused topK gives robust cross-edge speedups.
Intra-group graph construction requires workload-aware routing and quality-aware pruning.
Query-entry scan can be accelerated by coverage-correct GPU batching.
The remaining bottleneck is CPU-compatible graph output and large-graph search memory behavior.
```

中文表述：

> UNG 的 GPU 化不是把某一个 Vamana 或 GEMM kernel 替换掉，而是要把不规则 group topK、组内图质量、入口组选择和 CPU 输出边界统一考虑。我们已经在 cross-edge 和部分 group graph workload 上取得了可解释的加速，并明确了大规模场景下的下一批系统瓶颈。

不能讲：

- “GPU group graph 已经在所有场景保持 CPU Vamana 质量并完全取代它。”
- “我们的 GEMM / topK kernel 在任意形状上都比 cuBLAS 或 cuVS 更快。”
- “100%x40 已经是加速结果。”
- “query entry scan 19.85x 等于端到端 query 19.85x。”

## 8. 代码结构和可维护性现状

目前代码不是“已经完全清爽”，但已经从早期的临时实验状态往可维护结构收敛了一步：

本轮代码审计后的可汇报结论：

```text
外部使用层：已经清楚。主要方法都能通过 UngBuildConfig / env profile 选择，runbook 有登记。
中间接口层：部分清楚。Tagore group graph、query route/provider、cross-edge result/config 已有 request/result/config 契约。
底层实现层：仍偏复杂。group graph、cross-edge、query search 还没有完全变成独立 backend object，尤其 cross-edge CUDA route 仍由 gpu_gemm_topk.cu 顶层调度。
```

因此我不会说“代码已经完全简洁”，而会说：**当前代码已经足够支撑可复现实验和论文证据审计；如果要长期维护或继续接新方法，还需要把 group graph、cross-edge、query entry 三条线继续提升成 router + backend + output writer 的对象接口。**

本轮静态检查的规模信号也支持这个判断：`uni_nav_graph.cpp` 已收敛到约 `0.34k` 行，search/query、label graph、group graph、cross-edge、I/O、vector-attribute 和 legacy ACORN augment 都已分文件；但 `uni_nav_graph_group_graph.cpp`、`uni_nav_graph_cross_edges.cpp`、`uni_nav_graph_label_graph.cpp` 和 `gpu_gemm_topk.cu` 仍在 `0.86k~1.07k` 行量级，说明它们已经离开主文件，但还不是完全插件化 backend。

| 层面 | 当前状态 | 判断 |
|---|---|---|
| 顶层方法选择 | `UngBuildConfig` 已能表达 `original_cpu/current_cpu/naive_gpu/paper_fused/custom` | 主口径清楚，应保留 |
| 方法注册 | `docs/runbooks/UNG_METHOD_REGISTRY_CN.md` 已登记 main/baseline/diagnostic/boundary/negative | 后续新增方法有统一入口 |
| group graph C++ API | `tagore_graph_builder.h` 以 request/result 形式封装 Tagore/FastGrnnd batch | 目前最清晰的 backend 接口样板 |
| cross-edge config | 新增 `ung_cross_edge_config.{h,cpp}`，把 legacy env bridge 和 runtime route 从 `uni_nav_graph.cpp` 拆出 | 已完成第一轮低风险收敛 |
| cross-edge CUDA | `gpu_cross_groups_search_all_batched()` / `gpu_cross_groups_search_x_streaming()` 已显式接收 `const CrossEdgeGpuRuntimeConfig&` 和 `CrossEdgeBuildTiming*`；公共 descriptor、planning、regular route、double-buffer helper、X-streaming helper、source-exact experimental path、cuVS baseline 已分文件 | 主入口必须显式传 route/timing/writeback 指针；planning、regular route、DB route 和 benchmark filter 都已改成 reference config；workload planning、DB 资源/调度、boundary route、negative path 和 library baseline 不再堆在主执行函数里 |
| cross-edge C++ orchestration | `uni_nav_graph_cross_edges.cpp` 承载 backend resolve、CPU/GPU route adapter；additional_edges 和 merge/writeback 已拆成 helper | cross-edge 主体已离开 `uni_nav_graph.cpp`；下一步把 helper 提升为 output writer/backend adapter |
| cross-edge result | `CrossEdgeBuildResult` 汇总 timing、writeback、fallback、flat-id 状态；backend 私有函数接收 `CrossEdgeBuildTiming&` | 减少 `build_cross_group_edges()`、prepare 和 CUDA helper 里的散变量和三指针散传，为 backend adapter 做准备 |
| cross-edge workload | `gpu_cross_edge_planning.cuh` 中的 `plan_cross_edge_target_workload()` 返回 `CrossEdgeTargetWorkload`，统一 batched helper 的 target counts、nx、CPU tiny、bench-filter 和 count timing | planning 已从 kernel execution 和 DB helper 中拆出；后续仍需继续拆 CUDA execution 文件 |
| cross-edge DB route | `CrossEdgeDoubleBufferRoute` / `CrossEdgeDoubleBufferEligibility` 表达 double-buffer 设置和进入条件 | request / no-split / device-gather / unsupported split / global merge / chunk / nx 阈值不再散落在执行函数开头 |
| cross-edge DB slot | `CrossEdgeDoubleBufferSlot` 管理 stream/event、pinned/device buffer 和 descriptor buffer；`finish_cross_edge_double_buffer_slot()` 处理 completion timing/writeback | slot 生命周期、扩容和 finish 逻辑已从主执行函数移出 |
| cross-edge DB pack | `pack_cross_edge_double_buffer_chunk()` 返回 `CrossEdgeDoubleBufferChunkPack` | query packing、target offset、query-id 映射和 descriptor 构造已从 kernel enqueue 中拆出 |
| cross-edge DB enqueue | `enqueue_cross_edge_double_buffer_chunk()` 执行 H2D、topK 初始化、kernel launch、global merge、D2H 和 done event | stream 操作从主执行函数移出，主函数只串接 pack/stats/enqueue |
| cross-edge DB global output | `finalize_cross_edge_double_buffer_global_merge()` 处理最终 D2H 和 flat-id / id-vector / SearchQueue 写回 | global-merge 输出边界从主执行函数移出 |
| cross-edge DB schedule | `run_cross_edge_double_buffer_chunks()` 负责 chunk 切分和双 slot 轮转 | finish-before-enqueue 的调度顺序从主执行函数移出 |
| query entry | GPU correct-cover 还在 benchmark 工具内 | 语义清楚，但未正式 backend 化 |

### 8.1 接口清晰度矩阵

如果导师追问“现在代码是否足够清楚，后续同学能不能接上”，我会按下面这张表回答。结论是：**主要方法已经有统一选择入口，组内图和查询入口组已经有较清楚的数据契约；cross-edge 的 CUDA 内部仍偏复杂，但已经从一个巨型文件拆成 route/config/planning/dispatch/output helper；真正还没完成的是正式 backend 对象化和 query-entry GPU 接入 production search。**

| 工作线 | 方法/实现 | 当前接入入口 | 接口状态 | 是否适合后续直接扩展 |
|---|---|---|---|---|
| build profile | `original_cpu/current_cpu/naive_gpu/paper_fused/custom` | `UngBuildConfig::from_env()` | 清楚：顶层 profile 和 enum 已统一 | 是。新增主口径应先加 enum/profile，再更新 runbook |
| group graph | CPU Vamana / complete / bounded fallback | `build_graph_for_all_groups()` + `CpuGroupGraphSettings` | 较清楚：baseline/fallback 语义集中 | 是。仍建议下一步抽 `GroupGraphBackend` |
| group graph | Tagore / GrnndLike / FastGrnnd / Adaptive | `TagoreGroupRequest` / `TagoreBuildResult` / `build_tagore_vamana_cuda_batch()` | 清楚：这是当前最像 backend API 的接口 | 是。新增 GPU group builder 应仿照 request/result |
| cross-edge | CPU Vamana / CPU exact / CPU hybrid / original CPU | `UngCrossEdgeImpl` + `uni_nav_graph_cross_edges.cpp` | 较清楚：CPU baselines 和 GPU route adapter 已集中 | 可以扩展，但不要放到 `uni_nav_graph.cpp` |
| cross-edge | fused / SGEMM+topK / cuVS / source exact / X-streaming | `CrossEdgeGpuRuntimeConfig` + `gpu_cross_edge_*.cuh` | 中等：外层 route 清楚，内部 kernel/helper 较多 | 可以扩展，但应先建新的 helper/backend adapter，不要继续增大 `gpu_gemm_topk.cu` |
| cross-edge output | SearchQueue / id-vector / flat-id writeback | `CrossEdgeGpuWritebackMode` / `CrossEdgeBuildResult` | 半清楚：mode 已有，但 output writer 还不是独立对象 | 暂时谨慎。下一步应抽 `CrossEdgeOutputWriter` 或 CSR/flat graph view |
| query route | UNG / ACORN / selector / force route | `SearchRuntimeConfig` + `QueryRouteDecision` | 较清楚：旧长签名已变成 wrapper | 可以扩展，但 route policy 还没对象化 |
| query entry group | CPU min-super-set provider | `EntryGroupProviderRequest` / `EntryGroupProviderResult` | 清楚：输入 query labels，输出 group ids 和语义标志 | 是。新增 provider 应走同一 request/result |
| query entry group | GPU correct-cover | `query_entry_group_bench.cu`，production selector 暂 fallback | 半完成：算法语义清楚，但未接入正式 search | 不能直接当 production 结果。下一步要接 compact output |
| public header dependency | ACORN / ONNX selector / Vamana backend / Tagore build settings | `uni_nav_graph.h` 前置声明，search/io/group-graph `.cpp` 包含重头文件；`ung_build_settings.h` 只前置声明 `TagorePruneMode` | 较清楚：重依赖不再从 public header 泄露 | 是。继续减少 `uni_nav_graph.h` 中非必要 backend 头 |
| legacy augment | ACORN distance-oriented edge augmentation | `AcornInUng` / `uni_nav_graph_acorn_augment.cpp` | 清楚地隔离为 legacy/experimental | 不建议扩展。新 cross-edge 不应走这里 |

这张表也说明了当前“简洁清晰”的边界：**从外部跑实验已经清楚，从内部加一个全新 backend 还不够清楚**。最需要继续拆的是：

```text
GroupGraphBackend：把 CPU fallback、packed exact、FastGrnnd/adaptive route 从 UniNavGraph helper 提升为对象接口。
CrossEdgeBackend + CrossEdgeOutputWriter：让 fused、SGEMM、cuVS、X-streaming 共享同一个 workload/result/output 契约。
SearchEntryProvider：把 CPU min-super-set 和 GPU correct-cover 接入同一 production provider 层。
```

`CrossEdgeGpuRuntimeConfig` 目前已经统一管理：

```text
source_exact / universal_route / db_nosplit / double_buffer / x_streaming
source_exact mode / warps / padding
x_stream X/Q/output chunk sizes
query_upload_device_gather
global_merge / global_merge_direct / global_merge_direct_max_nx
db_global_merge / db_split_unsupported
cpu_tiny fallback / singleton fastpath
direct_qid fused / direct_qid_all_fused
GEMM/cuBLAS fallback / naive CUDA tuning
gather_threads / gather_blocks
flat_q_cap_mb / flat_out_cap_mb
db_chunk_queries / db_medium_max_nx / db_large_max_nx
small / medium / large fused kernel thresholds and warps
id-vector / flat-id / SearchQueue writeback
```

这一步的意义是：cross-edge 的主路径不再依赖 CUDA 深处隐藏 env，而是在 `build_cross_group_edges()` 创建一次 runtime route，并把它显式传给 GPU backend。X-streaming 仍然是 100%x40/OOM 的 boundary route，不是主性能路线，但它的 chunk 配置也已经进入同一个 route。最新的小整理还把 `UngGroupQueryDesc`、`UngGroupTileDesc`、`UngSourceQueryDesc`、`UngSourceTileDesc`、`UngTargetSegmentDesc` 抽到 `gpu_cross_edge_common.cuh`，新增 `CrossEdgeBuildResult` 收拢主流程分阶段耗时、active writeback、fallback 和 flat-id 状态，并把 GPU optimized、cuVS baseline、source-exact backend 私有函数改为接收 `CrossEdgeBuildTiming&`，把 batched / X-stream CUDA helper 改为显式接收 `const CrossEdgeGpuRuntimeConfig&` 和 `CrossEdgeBuildTiming*`，全量向量上传也通过 `gpu_prepare_all_vectors_for_cross_edge()` 接入 timing 对象；`plan_cross_edge_target_workload()` 和 `CrossEdgeBatchCapacity` 已进一步移到 route-neutral 的 `gpu_cross_edge_planning.cuh`，并已改成显式消费 reference config；Q/output/resident-X/norm/dot/descriptor/global-merge/cuBLAS workspace 等复用缓存位于 `gpu_cross_edge_buffers.cuh`，topK device primitive 位于 `gpu_cross_edge_topk_device.cuh`，norm/topK update kernels 位于 `gpu_cross_edge_update_kernels.cuh`，resident all-X upload/release 位于 `gpu_cross_edge_resident_vectors.cuh`，CPU tiny exact fallback 位于 `gpu_cross_edge_cpu_tiny.cuh`，query packing/upload 位于 `gpu_cross_edge_query_pack.cuh`，bucket descriptor assembly 位于 `gpu_cross_edge_bucket_descriptors.cuh`，per-group dispatch 位于 `gpu_cross_edge_group_dispatch.cuh`，并且 regular dispatch 已经通过 `CrossEdgeRegularWorkloadView` / `CrossEdgeRegularDispatchConfig` / `CrossEdgeRegularDeviceBuffers` / `CrossEdgeRegularDispatchCounters` 收敛接口，不再散传几十个参数。regular all-batched 的 query gather、singleton、direct-qid fused、global merge/writeback、GEMM backend 和 fused-kernel 阈值已经进入 `gpu_cross_edge_regular_route.cuh`，并显式消费 reference config。singleton fastpath kernels 位于 `gpu_cross_edge_singleton_kernels.cuh`，large-group fused kernels 位于 `gpu_cross_edge_large_group_kernels.cuh`，descriptor-batched TF32 kernels 位于 `gpu_cross_edge_descriptor_kernels.cuh`，small/medium/group-desc fused kernels 位于 `gpu_cross_edge_group_fused_kernels.cuh`，`CrossEdgeDoubleBufferRoute` / `CrossEdgeDoubleBufferEligibility`、`CrossEdgeDoubleBufferSlot`、`pack/enqueue/finalize/schedule` 等 DB helper 位于 `gpu_cross_edge_double_buffer.cuh`，其中 DB route 设置也已改成显式 reference config；SGEMM/cuBLASLt/naive-dot + separate-topK baseline 的 tile 执行位于 `gpu_cross_edge_sgemm_baseline.cuh`，bucket fused descriptor launch 位于 `gpu_cross_edge_bucket_fused.cuh`，immediate per-group fused launch 位于 `gpu_cross_edge_per_group_fused.cuh`，普通 batched 路径的 D2H/writeback 位于 `gpu_cross_edge_output.cuh`，profiling summary 位于 `gpu_cross_edge_profile.cuh`，optional CPU exact verify 位于 `gpu_cross_edge_verify.cuh`，X-streaming boundary route 位于 `gpu_cross_edge_x_streaming.cuh`，source-exact negative path 位于 `gpu_cross_edge_source_exact*.cuh`，cuVS library baseline 位于 `gpu_cross_edge_cuvs_baseline.cuh`。主 `.cu` 文件现在保留顶层 route setup 和高层执行分支，为后续拆分 CUDA 文件和 backend adapter 降低耦合。

同时，route 的名字、分类、writeback mode 和基础合法性检查也收进 config：主流程现在打印 `route=<name> class=<class> writeback=<mode>`，而不是到处手写 `source_exact/x_streaming/id_vector/flat_id` 的布尔组合。

但我不会把现在的代码汇报成“已经足够简洁”。静态扫描显示 `gpu_gemm_topk.cu` 的主入口不再直接读取 `UNG_UNIVERSAL_GPU` 或 diagnostics env，planning、regular route、DB route 和 benchmark filter helper 已显式消费 reference config；graph reserve、CPU group graph、Tagore/FastGrnnd group graph、CPU hybrid cross、GPU cross lifecycle 和 additional_edges 的 env fallback 已经移到 `ung_build_settings.{h,cpp}`，group graph impl 到 CUDA/adaptive/prune-mode 的映射也收进 `GroupGraphRoute`，CPU Vamana group graph、complete/bounded-complete fallback、Tagore/FastGrnnd group partition、fallback group build、exact/GNN batch split 和 batch result fill/writeback 都已集中到 `uni_nav_graph_group_graph.cpp`。剩余问题不是“开关散落”，而是 `CrossEdgeGpuRuntimeConfig` 仍然过宽，`uni_nav_graph.cpp` 仍是一个过重的 orchestration 文件，这些 settings 还没提升成正式的 `GroupGraphRouterConfig` / build orchestrator config。

还没有完成的结构问题也很明确：

```text
uni_nav_graph.cpp 仍然偏重：group graph、cross-edge 和 search/query 已经拆出，但 build/load/save、trie/LNG/coverage、query generation 与若干旧 experimental helper 仍在主文件中。
gpu_gemm_topk.cu 仍然过重：顶层 route setup 和高层执行分支还在主文件；GPU 复用缓存/handle、topK device primitive、norm/topK update kernels、resident all-X upload/release、CPU tiny exact fallback、query packing/upload、bucket descriptor assembly、per-group dispatch、singleton fastpath、large-group fused kernels、descriptor-batched TF32 kernels、small/medium/group-desc fused kernels、SGEMM/separate-topK baseline、bucket fused descriptor launch、immediate per-group fused launch、普通 batched output finalization、profiling summary、optional verify、X-streaming boundary、source-centric negative path 和 cuVS baseline 已经拆到独立 helper。
query-entry GPU provider 还没接入正式 search path。
flat/CSR graph output writer 还没完成，GPU 构建结果仍被 CPU-compatible Graph/NeighborList 限制。
```

所以我会向导师汇报：我们已经能清楚区分“哪些方法要保留、为什么保留、怎样切换”，但下一步如果要长期推进，应该优先拆 `router + backend + output writer`，而不是继续往大文件里追加开关。

## 9. 我希望导师给意见的问题

1. 论文主贡献是聚焦 **cross-edge fused topK + workload-aware graph builder**，还是把 query-entry correct-cover 也放进主贡献？
2. 对 group graph，是否接受“router + Pareto”作为方法主张，而不是“单一 GPU 图构建算法”？
3. 下一步优先补哪条证据：
   - 真实多标签 full-quality A/B；
   - query-entry 接入正式 search；
   - flat/CSR graph output；
   - 100%x40 graph search 细粒度计数器。
4. x400 目前是 `1.10x` 左右且 recall 接近 CPU，是否足够作为大组质量修复证据，还是必须继续做参数扫？

## 10. 下一步计划

| 优先级 | 工作 | 目标 |
|---:|---|---|
| P0 | query-entry GPU 接入 `search_UNG_index` | 从阶段加速变成正式端到端 query A/B |
| P0 | 给 `iterate_to_fixed_point()` 加细粒度计数器 | 解释 100%x40 graph search 为什么不按 dist calcs 比例增长 |
| P0 | 统一 current-best router 复测表 | 避免从多个历史 artifact 拼主表 |
| P1 | 真实多标签 / CelebA full-quality GPU group/cross A/B | 支撑泛化 |
| P1 | additional_edges / output boundary flat staged buffer | 降低 full-quality 剩余瓶颈 |
| P2 | Q-side device cache / source two-stage streaming | 让 100%x40 从可完成变成有性能希望 |
