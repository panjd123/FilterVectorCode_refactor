# Workload-Aware GPU Construction for Filtered Navigating Graphs

> Paper draft. This document is written as a conference-style technical manuscript and is grounded in the current implementation and measurement logs. It intentionally separates algorithmic contributions from engineering routing policies and known limitations.

## Abstract

Filtered vector search indexes must optimize both geometric proximity and attribute-constrained reachability. Unified Navigating Graph (UNG) addresses this problem by partitioning vectors into label groups, building intra-group proximity graphs, constructing a label navigation graph, and adding cross-group edges. However, UNG construction is not a single dense linear algebra problem. Its dominant stages consist of thousands of irregular group workloads: each target group receives queries from related groups and performs a small-to-medium topK search, while intra-group graph construction exhibits a long-tailed group-size distribution. CPU Vamana is robust but expensive, and naively invoking GPU libraries per group loses much of the potential speedup to packing, transfers, API overhead, and host-side writeback.

We present a workload-aware GPU construction pipeline for UNG. For cross-group edge construction, we introduce grouped fused topK, which batches many group-level `(nq, nx, dim) -> topK` tasks into GPU kernels and eliminates repeated query materialization through direct query-id access to resident base vectors. The implementation further uses TF32 WMMA group kernels, descriptor-based scheduling, and a safe writeback policy to reduce intermediate data movement without breaking cross-group ordering. For intra-group graph construction, we introduce FastGrnndCuda, a reverse-augmented local pruning backend that reuses GPU GNN-Descent candidates and replaces a heavy pruning path with fixed-budget local RNG-style occlusion.

On the Amazon sampled + jitter-repeat workload, the optimized cross-edge path reduces Amazon 1% x100 cross-edge time from `18735.8 ms` with CPU Vamana to `848.9 ms`, a `22.07x` speedup. It is also `2.81x` faster than CPU exact scan, `4.39x` faster than per-group cuVS brute force, and `1.64x` faster than SGEMM plus a separate topK stage. End-to-end, the current hybrid construction pipeline reduces Amazon 1% x100 build-scale index time from `16824 ms` to `6630 ms` (`2.54x`) relative to the CPU-Vamana-group + GPU-fused-cross baseline; this historical point is a skip-additional configuration. A new full-quality x100 coverage-query A/B fixes GPU fused cross and CPU additional edges: CPU Vamana groups give index `6863.47 ms` and L100/L500/L1000 `0.825/0.868/0.890667`, while FastGrnndCuda + CPU fallback gives index `6370.67 ms` and `0.828/0.871/0.896`. We further add a conservative `adaptive_cuda` route that keeps CPU fallback for small groups, uses packed exact-anchor for medium groups, and uses FastGrnndCuda for large groups while overlapping CPU fallback with the GPU batch. On the same x100 full-quality setup, this route gives index `5450 ms`, group time `2773.35 ms`, and L100/L500/L1000 `0.829/0.870/0.895`, improving build time without a meaningful recall loss relative to the FastGrnndCuda full-quality run. A new Amazon 1% x200 coverage-query experiment exposes the cost of using one pruning strategy for all groups: the old heavy-prune FastGrnndCuda path changes group construction from CPU Vamana's `8511.72 ms` to `26620.6 ms`. After adding diversified light pruning for medium groups, x200 group construction drops to `6706.36 ms` and index time drops from CPU's `17596.8 ms` to `16478.5 ms`. The remaining gap is quality: L1000 recall is `0.8656` versus CPU's `0.8690`, and L5000 is `0.9056` versus `0.9080`. We also removed a host-side scatter in the batched exact path by returning one packed graph buffer and filling UNG graph rows directly from offsets. This packed exact-anchor router reduces x200 group construction to `3827.28 ms` and index time to `11043.5 ms`, with repeat=3 L1000/L5000 recall `0.869/0.908`; on x400 it gives `11588.4 ms` group time, `28436.3 ms` index time, and repeat=3 L1000/L5000 `0.945/0.967`. A follow-up direct-H2D A/B shows that the older x200 exact-anchor path was mainly limited by overhead rather than by the GPU exact kernel: the old path has group time `8999.26 ms`, with `4140.61 ms` in host pack, `931.72 ms` in the GPU exact kernel, and `2849.50 ms` in graph fill; direct-H2D plus GPU lookup and fill16 reduces group time to `3204.88 ms`, with `0.04 ms` pack time and a similar `928.67 ms` kernel time, while L1000 recall remains `0.867`. A structural audit shows that the earlier low-recall exact run had the same intra-group graph as the new packed exact run, but fewer cross edges (`1,038,000` versus `1,093,224`), so that negative result was confounded by cross/additional-edge configuration. For the cross-edge scheduling boundary, we further fixed an all-or-nothing fallback in the double-buffer direct-qid path: on Amazon 1% x200 full-quality, partial routing sends `5173/5195` target groups through double-buffer and only `22` large groups through fallback, reducing cross-edge time from the old target-current `3900.81 ms` to `2657.23 ms`, with repeat=3 L1000/L5000 recall `0.866/0.907`. The latest universal flat double-buffer route combines target-centric descriptor batching, double-buffer execution, GPU global merge, and flat-id output under `UNG_UNIVERSAL_GPU=1`: it gives SIFT30 skip-additional cross-edge `3180.63 ms`, Amazon 1% x200 full-quality cross-edge `2494.21 ms` with L1000/L5000 recall `0.871/0.911`, and Amazon 1% x100 full-quality cross-edge `1689.40 ms` with L100/L500/L1000 recall `0.826/0.868/0.891`; the x100 point is a compatibility and cross-edge boundary result, not an end-to-end index-time speedup. On the Amazon 10% x40 many-group stress test, the full-quality additional-edge A/B reduces index time from `36053.3 ms` with CPU Vamana groups to `23497.4 ms` with FastGrnndCuda (`1.53x`), while repeat=3 L100/L500/L1000 recall changes from `0.8647/0.898/0.908` to `0.8668/0.898/0.9103`, without worsening P95/P99 query latency. The Amazon 1% x400 gather-Q full-quality run exposes a stronger pruning tradeoff: heavy pruning preserves quality but is slower than CPU, while light512 reduces index time from `37807.6 ms` to `30826.5 ms` but drops L5000 recall from `0.9660` to `0.9579`. A new reverse-tail + repair A/B raises x400 L5000 recall to `0.9659`, close to the same-script CPU baseline `0.965`, while keeping index time at `33044.9 ms`, about `1.10x` faster than the same-script CPU baseline `36257.3 ms`.

## 1. Introduction

Approximate nearest-neighbor search with filters is a central operation in modern vector databases and recommender systems. A query may need to satisfy a combination of labels, categories, user constraints, or hierarchical attributes in addition to vector similarity. This makes filtered ANN indexing fundamentally different from unfiltered ANN: the index must preserve local geometric navigability while also maintaining reachability across attribute-defined partitions.

Unified Navigating Graph (UNG) follows this principle. It partitions vectors into label groups, builds a proximity graph within each group, constructs a label navigation graph, computes hierarchical coverage metadata, and inserts cross-group edges. These cross-group edges are critical: they allow filtered search to move between related label groups rather than relying only on local within-group navigation.

The construction bottlenecks are irregular. Cross-edge construction is not one large matrix multiplication; it is a set of many group-level topK searches with different `(nq, nx)` shapes. Intra-group graph construction is similarly heterogeneous: tiny groups are too small to amortize GPU overhead, medium groups are sensitive to fallback policy, and large groups expose graph-pruning costs. A direct GPU port of each subtask is therefore insufficient. Per-group cuVS or cuBLAS calls repeatedly pay packing, H2D/D2H, API, and writeback overheads. Conversely, a single CPU graph builder such as Vamana remains expensive on large or numerous groups.

This paper argues that filtered graph construction requires workload-aware GPU execution. We make three contributions:

1. **Grouped fused cross-edge topK.** We formulate UNG cross-edge construction as many group-level exact topK tasks and implement a fused GPU pipeline with resident base vectors, direct query-id access, TF32 WMMA group kernels, safe writeback, and descriptor-based scheduling.
2. **Reverse-augmented local graph pruning with adaptive pruning strength.** We introduce FastGrnndCuda, which reuses GPU GNN-Descent candidates and applies fixed-budget reverse-augmented local RNG-style pruning. We also add a light-prune medium-group path that turns the x200 failure case into a faster-than-CPU speed/quality tradeoff.
3. **A system-level evaluation of filtered graph construction.** We compare CPU Vamana, CPU exact scan, per-group cuVS, SGEMM+topK, old fused kernels, final fused kernels, complete fallback, and CPU fallback. We explicitly report data residency costs and document the scale where direct-qid execution is currently unstable.

## 2. Background and Problem Setting

### 2.1 UNG Construction

Given base vectors `X`, vector labels, and a label hierarchy, UNG construction contains six major stages:

1. **Group partitioning.** Vectors are partitioned by label combinations, and each group is mapped to a contiguous vector-id range.
2. **Intra-group graph construction.** A local proximity graph is built for each group.
3. **Vector-attribute bipartite graph construction.** Vector-to-label relations are materialized.
4. **Label navigation graph construction.** Label groups are linked according to hierarchy and coverage.
5. **Descendant and coverage computation.** Hierarchical metadata is computed for later filtered search.
6. **Cross-group edge construction.** Edges are added from related source groups into target groups.

The cross-edge stage can be described as a collection of group-level topK tasks. For a target group `g`, UNG collects query vectors from incoming or parent-side related groups and searches within the target group:

```text
Input:  Q_g in R^{nq_g x d}, X_g in R^{nx_g x d}
Output: topK nearest neighbors in X_g for each q in Q_g
```

The Amazon workloads used in this paper contain thousands of such groups. Their `(nq_g, nx_g)` distribution is irregular, which is the main reason why per-group GPU library calls are inefficient.

### 2.2 Baselines

We evaluate against the following construction alternatives:

| Category | Method | Role |
|---|---|---|
| CPU | CPU Vamana cross-edge | Original approximate cross-edge path |
| CPU | CPU exact scan | Strong exact CPU topK baseline |
| GPU library | cuVS per group | Per-group brute-force GPU search |
| GPU library | SGEMM + separate topK | Strong dense GEMM baseline |
| Previous in-house | old fused | Grouped fused topK before direct-qid |
| Ours | final fused | Direct-qid, TF32 group kernel, safe writeback |
| Group graph | CPU Vamana | Quality baseline for intra-group graphs |
| Group graph | Tagore / FastGrnndCuda | GPU intra-group graph construction |

## 3. Method

### 3.1 Grouped Fused TopK for Cross-Group Edges

The straightforward GPU approach is to invoke brute-force search or GEMM for each target group and then run topK. This approach is dominated by repeated overheads on UNG workloads. Instead, our cross-edge pipeline batches target groups and constructs compact descriptors:

```text
desc = (query_row, target_x_offset, target_nx, group_id)
```

Each descriptor maps one query row to its target group range in the resident base-vector array. GPU kernels use these descriptors to compute distances and update topK in the same execution path. Groups are routed by shape into small, medium, and TF32 group kernels. This avoids materializing a full distance matrix and reduces the number of per-group host interactions.

The fused design targets the actual UNG workload rather than dense GEMM in isolation. Its main benefit comes from reducing intermediate writes, separate topK launches, per-group API overhead, and host-side result conversion.

### 3.2 Direct Query-ID Access and Data Residency

Most cross-edge queries are themselves base vectors. Instead of expanding query vectors into a dense `Q` buffer, we prepare all base vectors once on the GPU:

```text
g_d_all_X    = resident base vectors
g_d_all_norm = resident L2 norms
```

Then each query is represented by its global vector id:

```text
q      = g_d_all_X[qid]
q_norm = g_d_all_norm[qid]
```

This removes query-vector H2D traffic and query-norm kernels from the repeated group search path. The one-time per-process cost is `prepare_all`, which includes host registration, full-data H2D, and norm computation. We report both single-build time including `prepare_all` and resident compute time excluding it.

### 3.3 TF32 WMMA and Id-Only Writeback

For group shapes that benefit from Tensor Cores, we use TF32 WMMA kernels. The current group kernel uses a 16x16 MMA tile and fuses distance computation with topK update. The implementation uses:

- `UNG_TF32_GROUP_2D=1`, which maps query tiles and groups to a 2D grid.
- `UNG_GPU_ID_ONLY_WRITEBACK=1`, which copies only topK ids back to host because graph insertion only needs ids.
- `UNG_BUCKET_GROUP_FUSED=1`, which routes groups into the appropriate fused path.

This path does not claim to outperform cuBLAS for all dense GEMM shapes. It is designed to reduce end-to-end overhead for many irregular group-level topK tasks.

We also implemented a source-centric no-lock cross-edge prototype as the next replacement direction. The original cross-edge semantics can be traversed either by target groups via `in_neighbors[target]` or by source/query groups via `out_neighbors[source]`. The target-centric implementation emits multiple local topK rows for the same qid and therefore requires host `SearchQueue` merging or a GPU global merge. A simple qid-lock global merge was measured as a negative ablation because it reduced D2H/writeback but increased GPU lock contention. The source-centric prototype assigns each qid to its owning source group and maintains the final topK across all target segments inside the kernel, eliminating per-qid locks. It is enabled by `UNG_GPU_SOURCE_EXACT=1` with `UNG_GPU_SOURCE_EXACT_MODE=0` for CUDA-core exact scan or `1` for experimental TF32 WMMA fused topK. After adding stream error checks and fixing the id-only output boundary, the checked CUDA-core source path gives SIFT30 skip-additional cross `5926.67 ms` and kernel `5525.5 ms`, slower than the target-centric fused cross `4918.9 ms`. We therefore do not use source-centric as the current best result; instead, it motivates a future two-stage source grouped GEMM plus per-source reduce design.

The current engineering route is universal flat double-buffer. It keeps target-centric descriptor batching, routes all supported target groups through the double-buffer all-DB path by default, and uses flat-id output plus GPU global merge to reduce `SearchQueue` and per-group-container writeback overhead. This route gives SIFT30 skip-additional cross `3180.63 ms`; on Amazon 1% x200 full-quality with CPU Vamana `additional_edges`, it gives cross `2494.21 ms` and L1000/L5000 recall `0.871/0.911`; on Amazon 1% x100 full-quality it gives cross `1689.40 ms` and L100/L500/L1000 recall `0.826/0.868/0.891`, but the index time is slower. We therefore present it as an x200 positive result and an x100 boundary result, not as an unconditional end-to-end speedup.

### 3.4 FastGrnndCuda for Intra-Group Graphs

CPU Vamana provides a strong group graph quality baseline but is slow for large groups. Tagore-like GPU graph construction can generate candidates quickly, but its original pruning path is expensive in our setting. FastGrnndCuda decomposes graph construction into:

1. GPU GNN-Descent candidate generation.
2. Reverse candidate sampling, where nodes that point to the current node are added to its candidate pool.
3. Fixed-budget local RNG-style occlusion pruning.

The method is inspired by reverse-neighbor propagation and RNG-style pruning, but it is not a reproduction of GRNND/RNN-Descent. We do not integrate RNG pruning into the full NN-Descent iteration. Instead, we reuse the candidate-generation stage and replace the heavy post-processing path with a lightweight pruning backend tailored to UNG group graphs.

For the x400 quality gap, we also added a disabled-by-default light reverse-tail path. It constructs a small sampled-reverse pool inside the light-prune branch and uses only a fixed number of reverse candidates to fill tail slots, without restoring heavy RNG occlusion. The path is controlled by `UNG_FAST_GRNND_LIGHT_REVERSE_CAP/SLOTS/FORWARD_CAP`. In the x400 A/B, degree repair alone mostly fixes low-degree structure but only changes L5000 from `0.9574` to `0.957767` and increases index time. Reverse-tail + repair raises L5000 to `0.9659`, close to CPU's `0.9660`, while keeping index time below the historical CPU baseline.

### 3.5 Workload-Aware Routing

The fastest end-to-end configuration is a hybrid policy:

- Very small groups use complete graph or lightweight fallback.
- Groups below `UNG_TAGORE_MIN_GROUP_SIZE` use CPU Vamana fallback by default.
- Larger groups use FastGrnndCuda.

This routing policy is an engineering component of the system, not the core graph-pruning algorithm. We report it explicitly because it materially affects end-to-end construction time.

### 3.5.1 Conservative Adaptive Route

To make the routing policy reproducible rather than a collection of manually composed environment variables, we add `UNG_GROUP_GRAPH_IMPL=4`, named `adaptive_cuda`. Its default route is:

```text
nx < 128        -> CPU fallback
128 <= nx <=512 -> packed exact-anchor
nx > 512        -> FastGrnndCuda
```

The CPU fallback path runs concurrently with the GPU batch, and packed exact results are filled directly from a single packed graph buffer. On Amazon 1% x100 with full additional edges, this conservative route gives index `5450 ms`, group time `2773.35 ms`, and L100/L500/L1000 `0.829/0.870/0.895`, compared with the FastGrnndCuda full-quality run's `6370.67 ms`, `3216.21 ms`, and `0.828/0.871/0.896`.

We also tested an aggressive bounded fallback that replaces small-group CPU Vamana. It reduces skip-additional group construction to `1953.07 ms`, but L100/L500/L1000 fall to `0.816/0.823/0.827`. We therefore treat it as a negative ablation: small-group CPU Vamana is a remaining bottleneck, but a simple ring-like bounded graph is not a full-quality replacement.

### 3.6 Mixed Exact/GNN Router as a Pending Scheduling Optimization

An earlier batched exact-kNN run appeared negative on x200: pure local L5000 was `0.8199`, and deterministic anchor-tail was about `0.829`. We no longer use that run as evidence against the intra-group exact graph itself. A structural audit shows that the old exact-anchor index and the new packed exact index have identical intra-group structure: `38,553,600` intra edges, zero low-degree nodes, and largest weakly connected component ratio `1` for all groups. The difference is in cross edges: the old run has `1,038,000`, while both CPU Vamana and the new packed exact run have `1,093,224`. The old low-recall result is therefore a confounded cross/additional-edge ablation.

The current scheduling question is whether small and medium groups can be routed to one batched exact CUDA call while larger groups keep Tagore/GNN candidate generation. The previous implementation only used the batched exact path when every group in the batch satisfied the threshold. If one large group was present, all small and medium groups also fell back to the GNN path. The current implementation splits requests by `UNG_FAST_GRNND_BATCH_EXACT_NX`, runs an exact batch and a GNN batch, and merges results back in the original group order.

We have completed one preliminary A/B run on the Amazon 1% x200 coverage-query workload. The run used `NUM_REPEATS=1` and enabled `UNG_TAGORE_BATCH_STREAMS=2`, `UNG_TAGORE_FILL_FAST=1`, and `UNG_TAGORE_FILL_THREADS=16` to reduce many-group overhead. In this setting, `UNG_FAST_GRNND_BATCH_EXACT_NX=256/512` did not reproduce the large recall loss of the pure exact-kNN graph. `router_exact_nx256` achieved L100/L500/L1000 recall `0.8229/0.8510/0.8690`, compared with CPU Vamana `0.8230/0.8490/0.8690` and the non-exact router `0.8180/0.8463/0.8661`. Its group-graph time was `5770.88 ms`, a `1.49x` speedup over the non-exact router's `8606.52 ms`.

We then optimized the exact-batch writeback path. The previous implementation copied the D2H graph into per-group `result.graph` vectors and the UNG fill stage copied those rows again into `_group_graphs`. The current implementation returns a packed graph buffer plus offsets and lets `UniNavGraph` fill directly from that packed buffer. This does not change exact-graph semantics, but it removes one host scatter pass. With `UNG_FAST_GRNND_BATCH_EXACT_NX=4096`, x200 group time drops from the old scatter run's `4929.54 ms` to `3827.28 ms`, with index time `11043.5 ms` and repeat=3 L1000/L5000 `0.869/0.908`. The same path on x400 gives group time `11588.4 ms`, index time `28436.3 ms`, and repeat=3 L1000/L5000 `0.945/0.967`.

A later x200 direct-H2D A/B separates the remaining exact-anchor overhead. The old exact batch copied already group-contiguous base vectors into a temporary host packed buffer and generated point-to-group lookup arrays on the CPU. That path has group time `8999.26 ms`, including `4140.61 ms` of host pack, `931.72 ms` in the GPU exact kernel, and `2849.50 ms` of graph fill. With direct H2D from contiguous host runs, GPU-side lookup generation, and a lower graph-fill thread count, group time drops to `3204.88 ms`; pack time becomes `0.04 ms`, the GPU exact kernel remains `928.67 ms`, and L1000 recall remains `0.867`. We treat this as the strongest current group-graph replacement candidate, while still requiring x100, 10%x40, and real multi-label validation before making a broad replacement claim. It also clarifies the next bottleneck: `Graph::neighbors` materialization remains a CPU-compatible output boundary, so further large gains require a flat-adjacency/CSR backend rather than only more CUDA kernel tuning.

This result suggests that exact search can be useful as a small/medium-group scheduling branch, but not as a standalone replacement for all intra-group graph construction. Before submission, it should still be repeated with `NUM_REPEATS=3` and tested on additional group-size distributions:

```bash
scripts/benchmarks/run_group_graph_router_ab.sh
python3 tools/benchmarks/summarize_group_graph_router_ab.py <out_root>
```

The router is paper-usable only if thresholds such as `UNG_FAST_GRNND_BATCH_EXACT_NX=128/256/512` reduce `group_ms/index_ms` while recall remains close to both `router_exact_nx0` and CPU Vamana. The current x200 result supports it as a positive preliminary ablation, not as proof that the universal replacement problem is solved.

## 4. Experimental Setup

### 4.1 Hardware and Parameters

| Item | Configuration |
|---|---|
| GPU | NVIDIA RTX A6000 |
| CPU threads | `--num_threads 128` |
| Distance | L2 |
| UNG parameters | `max_degree=32`, `Lbuild=100`, `alpha=1.2`, `num_cross_edges=6` |

The main workload is Amazon sampled + jitter repeat. The repeat factor increases vector count while preserving label structure.

| Dataset | Points | Groups |
|---|---:|---:|
| Amazon 1% x40 | 240,960 | 5,666 |
| Amazon 1% x100 | 602,400 | 5,666 |
| Amazon 1% x200 | 1,204,800 | 5,666 |
| Amazon 1% x400 | 2,409,600 | 5,666 |
| Amazon 10% x40 | 2,409,800 | 53,840 |

### 4.2 Optimized Configuration

Cross-edge:

```bash
UNG_CROSS_EDGE_IMPL=1
UNG_ADDITIONAL_EDGES_IMPL=0
UNG_GPU_TOPK_IMPL=3
```

`UNG_ADDITIONAL_EDGES_IMPL=0` keeps the CPU Vamana additional-edge path and is the full-quality configuration used for end-to-end recall claims. `UNG_ADDITIONAL_EDGES_IMPL=1` skips additional edges and is only a stage ablation.

Default fused path:

```bash
UNG_DIRECT_QID_FUSED=1
UNG_DIRECT_QID_ALL_FUSED=1
UNG_GPU_ID_ONLY_WRITEBACK=1
UNG_TF32_GROUP_2D=1
UNG_BUCKET_GROUP_FUSED=1
```

Group graph:

```bash
UNG_GROUP_GRAPH_IMPL=3
UNG_TAGORE_K=64
UNG_TAGORE_ITER=4
UNG_TAGORE_M=64
```

For the scale experiments, `UNG_TAGORE_MIN_GROUP_SIZE=256` is used for x40, x200, and x400; `128` is used for x100.

## 5. Evaluation

### 5.1 Cross-Edge Performance

Table 1 compares cross-edge methods on Amazon 1% x100.

The table is reproducible with `tools/benchmarks/summarize_cross_baselines.py` from saved run directories; the current generated artifact is `/home/graphdb/fv_runs/baseline_fairness_x100_20260602/cross_baselines.md`. The CPU exact baseline uses `128` build threads on an Intel Xeon Platinum 8360Y machine with two sockets, 36 cores per socket, and 144 logical CPUs.

| Method | Cross total | H2D | Kernel/build | D2H | Index | Notes |
|---|---:|---:|---:|---:|---:|---|
| CPU Vamana cross | `18735.8 ms` | - | - | - | `23150 ms` | Original slow path |
| CPU exact scan | `2389.8 ms` | - | - | - | `6995 ms` | Strong CPU baseline |
| cuVS per group | `3730.5 ms` | `592.1` | `821.8` | `43.5` | `14353 ms` | Per-group library calls |
| SGEMM + separate topK | `1395.9 ms` | `108.4` | `541.6` | `9.9` | `7452 ms` | Strong GPU library baseline |
| old fused baseline | `1268.5 ms` | `103.9` | `528.7` | `7.8` | `7388 ms` | Before direct-qid |
| final direct-qid fused, CPU group | `1313.1 ms` | `76.0` | `227.0` | `5.8` | `14053 ms` | Cross-edge replacement only |
| final direct-qid fused, mixed group | `848.9 ms` | `74.1` | `232.8` | `3.6` | `6630 ms` | Current full optimized run |

Using `848.9 ms` as the optimized cross-edge time:

| Baseline | Speedup |
|---|---:|
| CPU Vamana cross | `22.07x` |
| CPU exact scan | `2.81x` |
| cuVS per group | `4.39x` |
| SGEMM + topK | `1.64x` |

The speedup over cuVS is primarily due to workload-level batching and reduced per-group overhead. The speedup over SGEMM+topK is smaller, which is expected because cuBLAS/cuBLASLt are strong compute baselines.

### 5.1.1 Source-Centric No-Lock Cross-Edge Smoke Test

To test whether the output boundary can be addressed more directly than with qid-lock global merge, we added a source-centric no-lock prototype. It traverses `out_neighbors[source]` and computes the final topK for each qid across all target segments inside one kernel path. This avoids per-qid locks and avoids emitting multiple target-local topK rows for the same qid.

This is a SIFT30 smoke test with `additional_edges=skip`; it is not a full-quality result and does not replace the Amazon x100 main cross-edge table.

| Path | Artifact | Cross total | Prepare/H2D | Kernel | D2H | Writeback | Note |
|---|---|---:|---:|---:|---:|---:|---|
| source-centric CUDA-core exact, legacy | `/home/graphdb/FilterVectorResultsRefactor/organized_benchmarks/ung_cross_edge_fused_20260602_052124` | `5981.0 ms` | `20.3 ms` | `5524.0 ms` | `1.9 ms` | `43.0 ms` | Low output overhead but poor kernel efficiency |
| current target-centric fused | `/home/graphdb/FilterVectorResultsRefactor/organized_benchmarks/ung_cross_edge_fused_20260602_052209` | `4918.9 ms` | `77.4 ms` | `2189.9 ms` | `35.7 ms` | chunk writeback about `43.8 ms` | Current comparison point |
| source-centric TF32 WMMA, legacy | `/home/graphdb/FilterVectorResultsRefactor/organized_benchmarks/ung_cross_edge_fused_20260602_052750` | `3021.9 ms` | `23.2 ms` | `1402.9 ms` | `2.9 ms` | `63.1 ms` | Old smoke; not checked under the current error-boundary instrumentation |
| source-centric TF32 WMMA + source-block padding, legacy | `/home/graphdb/fv_runs/source_wmma_padded_sift30_skipadd_20260602_053546` | `1927.7 ms` | `23.2 ms` | `1418.5 ms` | `1.9 ms` | `26.8 ms` | Old smoke; design signal only |
| source-centric CUDA-core id-only, current checked | `/home/graphdb/FilterVectorResultsRefactor/organized_benchmarks/ung_cross_edge_fused_20260602_091046` | `5926.7 ms` | `23.5 ms` | `5525.5 ms` | `1.0 ms` | `13.3 ms` | Checked after null-distance output and stream-error fixes; negative result |

The result supports the design direction but not the current implementation as a replacement. Source-centric traversal removes qid conflicts and reduces D2H/writeback, but the current one-stage kernel serializes too much candidate scanning within a source query/tile. The legacy TF32 WMMA smoke suggests that Tensor Core source tiling may be promising, but it cannot be used as a main result until it is rerun with the same error checks, id-only output boundary, and TF32/FP32 topK consistency tests. A credible source-centric replacement should be two-stage: generate partial topK for `(source query tile, target segment tile)` in parallel, then reduce partial results per source query on GPU.

The first Amazon full-quality check is more conservative. On Amazon 1% x200 with CPU Vamana intra-group graphs and `additional_edges=cpu_vamana`, historical source-centric w8 gives cross `4693.4 ms`, kernel `3224.6 ms`, and L1000/L5000 repeat=3 recall `0.871/0.911`; source-centric w4 gives cross `4803.07 ms`, kernel `3091.9 ms`, and the same repeat=3 recall `0.871/0.911`; the same-machine target-centric fused path gives cross `3900.81 ms`, kernel `2095.8 ms`, and repeat=3 recall `0.858/0.9068`. These historical runs show that source-centric can be connected to the full-quality Amazon build/search pipeline, but they are not a current performance claim until rerun under the checked source path; even under the historical measurement, source-centric is not the current performance winner. A follow-up structural audit showed that the target-current gap was not caused by the intra-group graph but by missing cross edges. We then fixed the all-or-nothing fallback in the target-centric double-buffer path: when only a few target groups exceed `DB_LARGE_MAX_NX`, supported groups still use double-buffer and unsupported groups fall back separately. On x200 full-quality this sends `5173/5195` target groups through double-buffer, reduces cross-edge time to `2657.23 ms` and kernel time to `1636.0 ms`, and gives `1,093,032` cross edges, only `192` fewer than the source/CPU-aligned `1,093,224`. Its repeat=3 L1000/L5000 recall is `0.866/0.907`. This makes target-route coverage the stronger near-term optimization, while source-centric remains a candidate for a later no-lock replacement. Before source-centric can become a main result, it needs better source WMMA tiling, additional-edge-enabled A/B, and TF32/FP32 topK consistency checks.

### 5.2 Cross-Edge Breakdown

Amazon 1% x100 full optimized:

| Component | Time |
|---|---:|
| Cross total | `848.9 ms` |
| `prepare_all` | `218.714 ms` |
| Resident cross | `630.186 ms` |
| GPU H2D events | `74.1 ms` |
| GPU kernel events | `232.8 ms` |
| GPU D2H events | `3.6 ms` |

`prepare_all` is reported separately because it is a one-time data residency cost per build process, not a per-group cost. Host registration can also vary between runs; one previous x100 run observed `540.843 ms` for the same conceptual stage.

### 5.3 End-to-End Construction

Table 3 reports the current fastest engineering configuration from the build-scale retest: FastGrnndCuda for selected large groups, CPU Vamana fallback for smaller groups, and fused cross-edge construction. A later end-to-end x200 coverage-query run shows that medium groups need a lighter pruning path; the old heavy-prune negative result and the light-prune fix are reported in Section 5.10. For x400, direct-qid is unstable, so the stable gather-Q fused path is used.

| Dataset | Group graph | Vector-attr | LNG | Desc | Coverage | Cross | Index |
|---|---:|---:|---:|---:|---:|---:|---:|
| 1% x40 | `868.5` | `126.4` | `98.3` | `10.5` | `57.9` | `389.8` | `2375` |
| 1% x100 | `3373.2` | `209.9` | `43.7` | `3.9` | `12.4` | `848.9` | `6630` |
| 1% x200 | `8630.8` | `410.2` | `37.4` | `2.0` | `18.6` | `3679.5` | `16426` |
| 1% x400 | `31245.6` | `777.0` | `65.0` | `19.3` | `211.0` | `12462.8` | `50562` |

Relative to the CPU-Vamana-group + GPU-fused-cross baseline:

| Dataset | Baseline Index | Current Index | Index speedup | Baseline group | Current group | Group speedup |
|---|---:|---:|---:|---:|---:|---:|
| 1% x40 | `2484` | `2375` | `1.05x` | `891.0` | `868.5` | `1.03x` |
| 1% x100 | `16824` | `6630` | `2.54x` | `11549.9` | `3373.2` | `3.42x` |
| 1% x200 | `34603` | `16426` | `2.11x` | `26231.4` | `8630.8` | `3.04x` |
| 1% x400 | `70616` | `50562` | `1.40x` | `54043.2` | `31245.6` | `1.73x` |

The build-scale table makes x100 look like the clearest sweet spot. x200 appears favorable in that older build-only table, but the later coverage-query end-to-end A/B shows that this cannot be generalized to the full-quality configuration. x40 has too few large groups to benefit much from GPU graph construction. x400 is limited by graph pruning and by the need to use the more stable gather-Q cross-edge path.

### 5.4 Intra-Group Graph Breakdown

| Dataset | Tagore groups | CPU fallback groups | Complete fallback groups | Direct build | Fallback wall | GNN | Prune | Group total |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1% x40 | `16` | `112` | `5538` | `229.4` | `636.6` | `14.5` | `38.8` | `868.5` |
| 1% x100 | `128` | `5538` | `0` | `540.0` | `2822.2` | `106.3` | `265.8` | `3373.2` |
| 1% x200 | `128` | `5538` | `0` | `961.1` | `7648.4` | `174.4` | `557.4` | `8630.8` |
| 1% x400 | `5666` | `0` | `0` | `30812.5` | `11.0` | `4940.1` | `20021.9` | `31245.6` |

The x400 case shows that pruning is the dominant intra-group graph bottleneck (`20021.9 ms`). In x100 and x200, the fastest end-to-end configuration relies on CPU fallback for many smaller groups; we therefore distinguish the FastGrnndCuda algorithm from the hybrid routing policy.

### 5.5 Graph Quality on a Single Group

For `g1024_nx2048`, we compare group graph quality using recall at different search budgets:

| Method | Build graph | Prune/refine | L20 | L50 | L100 | L200 | L500 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Original Tagore | `9348.87 ms` | prune `4531.38 ms` | `0.9355` | `0.9765` | `0.9860` | `0.9915` | `0.9945` |
| FastGrnndCuda old | `3790.43 ms` | prune `657.63 ms` | `0.9375` | `0.9790` | `0.9870` | `0.9920` | `0.9955` |
| Reverse-augmented Local RNG | `4014.91 ms` | prune `810.08 ms` | `0.9515` | `0.9880` | `0.9930` | `0.9955` | `0.9965` |
| Method3 iter=4 | `8248.08 ms` | prune `4492` + refine `944 ms` | `0.9500` | `0.9880` | `0.9930` | `0.9965` | `0.9985` |
| CPU Vamana | `9265.03 ms` | - | `0.9655` | `0.9910` | `0.9990` | `1.0000` | `1.0000` |

The final reverse-augmented pruning path improves both build time and recall over the original Tagore path. It is much faster than Method3 while matching L20/L50/L100. CPU Vamana still has the best high-budget recall, so FastGrnndCuda should be described as a speed-quality tradeoff rather than a strict semantic replacement.

### 5.6 Fallback Ablation

Complete fallback results from a previous run:

| Dataset | Group | Cross | Index | Interpretation |
|---|---:|---:|---:|---|
| 1% x40 complete fallback | `849.6` | `505.3` | `2442` | Similar to CPU fallback |
| 1% x100 complete fallback | `4533.4` | `1257.4` | `8179` | Slower than CPU fallback |
| 1% x200 complete fallback | `20021.6` | `2788.9` | `26642` | Complete fallback too heavy |
| 1% x400 complete fallback | `31074.6` | `6722.0` | `44713` | No fallback; difference mainly from cross-edge config |
| 10% x40 complete fallback | `7225.2` | `6201.2` | `23207` | More groups amortize GPU work better |

This confirms that fallback policy materially affects end-to-end results.

### 5.7 Submission-Readiness Reviewer Audit

The current evidence supports the cross-edge performance claim, but a submission-ready paper still needs to answer the following reviewer questions. We keep these questions in the draft to avoid overstating the current state of evidence.

| Reviewer question | Current answer | Required experiment before submission |
|---|---|---|
| Does FastGrnndCuda preserve filtered-search recall, not only single-group graph recall? | Partially answered on original Amazon PF, SIFT30, Amazon 1% x100, Amazon 1% x200, Amazon 10% x40, and Amazon 1% x400 gather-Q. The x100 full-quality repeat=3 run does not reduce L100/L500/L1000 recall; the 10%x40 full-quality run gives `1.53x` index speedup, and the L50/100/200/500/1000/2000/5000 sweep keeps recall within `[-0.001,+0.002]` of CPU; x400 light512 is faster but has a `0.0081` recall gap. | Add more x400 parameter sweeps and a real multi-label dataset. |
| Is the `22.07x` speedup only against a weak CPU baseline? | The paper also reports CPU exact scan, cuVS per-group, and SGEMM+topK. | Add CPU exact scan thread scaling and explicitly report CPU model, thread count, and OpenMP configuration. |
| Is grouped fused topK an algorithmic/system contribution or just engineering cleanup? | The strongest defensible claim is workload-level fusion for irregular group topK. | Add ablations for resident vectors, direct-qid, id-only writeback, and final fused vs SGEMM+topK. |
| Is the method stable at large scale? | x400 direct-qid is currently unstable; x400 uses gather-Q fused. | Either fix direct-qid x400 strict mode or present a formal gather-Q/direct-qid routing policy. |
| Is x400 degree repair just artificial edge padding? | It is currently only an implemented hypothesis, not a reported result. | Run `light512_norepair/light512_repair/light512_reverse_repair` and report both search recall and stored-graph diagnostics. |
| Is compact-D2H only an engineering optimization? | Yes, and the current x400 ablation is negative/ambiguous: compact-on is slower than compact-off and does not show a stable D2H win. It should not be reported as graph-quality improvement or as the x400 speedup source. | Report it only as an overhead negative ablation. |
| Does the mixed exact/GNN router reintroduce the low-quality pure-exact graph? | The x200 repeat=1 A/B shows that `nx256/512` does not reproduce the pure-exact recall collapse; L1000 is close to CPU Vamana and group time is about `1.49x/1.47x` faster than the non-exact router. | Add repeat=3 and more datasets; isolate the exact-router effect from multi-stream and fast-fill effects. |
| Does the method scale to many small/medium groups? | Amazon 10% x40 full-quality repeat=3 A/B now runs: FastGrnndCuda index time is `23497.4 ms` vs CPU Vamana `36053.3 ms`, `1.53x`. A wider L50/100/200/500/1000/2000/5000 sweep shows recall deltas `+0.0019/+0.0018/-0.0010/+0.0000/+0.0020/+0.0020/+0.0010` relative to CPU. | Query latency is noisy/non-monotonic, so this is recall robustness evidence rather than a latency-speedup claim. |
| Does the result generalize beyond Amazon sampled+jitter-repeat? | SIFT30 now has a fixed fused-cross + group-graph end-to-end A/B. | Add CelebA or another real multi-label dataset. |

The corresponding experiment driver is `scripts/benchmarks/run_end_to_end_recall_ab.sh`, which fixes query set, ground truth, cross-edge implementation, and search parameters while varying the group-graph backend.

### 5.8 End-to-End Group-Graph A/B on Original Amazon PF

We ran an additional end-to-end A/B on the original Amazon PF query workload. Both variants use the same query set, ground truth, fused cross-edge backend, and CPU Vamana additional edges. The only intended difference is the intra-group graph backend.

| Variant | Index ms | Group ms | Cross ms | L1000 recall | L2000 recall | L5000 recall |
|---|---:|---:|---:|---:|---:|---:|
| CPU Vamana group | 43376.6 | 5776.25 | 20194.7 | 0.912333 | 0.932067 | 0.971767 |
| FastGrnndCuda + CPU fallback | 36732.0 | 1049.92 | 18847.4 | 0.911700 | 0.931500 | 0.971133 |

This gives the first end-to-end evidence that FastGrnndCuda can reduce group construction time while preserving filtered-search quality on a real query workload. Group construction improves by `5.50x`, and total index time improves by `1.18x`; the average recall loss is below `0.001` for all three `Lsearch` values. This experiment does not replace the required Amazon repeat-scale evaluation, but it closes the most immediate reviewer concern better than single-group recall alone. Although the query source-group file is missing in this run, code inspection shows that it is only used when `is_ung_more_entry` is enabled; the current `search_UNG_index` CLI does not expose that flag and leaves it false.

We also found that additional edges are quality-critical on this workload. With `UNG_ADDITIONAL_EDGES_IMPL=1` (skip), average recall drops to roughly `0.61` for both CPU Vamana group and FastGrnndCuda group. Therefore, skip-additional-edges results should be treated as a stage ablation, not as a full-quality end-to-end configuration.

The modest `1.18x` end-to-end speedup is explained by the remaining stage mix. With CPU additional edges enabled, FastGrnndCuda reduces the group-graph share from `13.32%` to `2.86%` of index time, but cross-edge construction becomes `51.31%` and LNG construction `32.71%` of the optimized build. This is an Amdahl limitation: after group graph acceleration, further end-to-end gains require accelerating additional-edge construction and LNG rather than only improving intra-group graph kernels.

### 5.9 End-to-End Group-Graph A/B on SIFT30

To test whether the group-graph result is specific to Amazon, we ran a SIFT30 query workload. This experiment fixes the repaired fused cross-edge backend and CPU Vamana additional edges, and only switches the intra-group graph backend. The SIFT30 run uses `1,000,000` base vectors, `69,176` groups, and `10,000` containment queries.

| Variant | Index ms | Group ms | Cross ms | L100 recall | L200 recall | L500 recall | L1000 recall |
|---|---:|---:|---:|---:|---:|---:|---:|
| CPU Vamana group | 31740.4 | 11245.3 | 17886.2 | 0.609423 | 0.706580 | 0.814600 | 0.877587 |
| FastGrnndCuda + CPU fallback | 24989.9 | 1889.79 | 17868.9 | 0.607997 | 0.706287 | 0.812680 | 0.876297 |

On SIFT30, FastGrnndCuda reduces group construction time by `5.95x` and total index time by `1.27x`; the L1000 average recall drops by about `0.00129`. The repaired fused cross-edge backend reduces SIFT30 cross-edge time from the CPU Vamana cross baseline of `69834.5 ms` to `17886.2 ms`, or `3.90x`, while preserving the CPU-Vamana-group recall.

This reviewer iteration exposed and fixed two semantic bugs. First, when `DIRECT_QID_ALL_FUSED=1` cannot cover `nx>4096` large-group fallback paths, the pipeline must materialize `g_d_Q` and `q_norm`. Second, id-only writeback is not safe for the host-merge path because the per-query `SearchQueue` still needs true distances to rank candidates across target groups; id-only writeback is only safe after GPU global merge has already performed the global ordering.

### 5.10 Amazon 1% x200 Coverage Query and Light Prune

We added an Amazon 1% x200 repeat-scale A/B with coverage queries generated from the index hierarchy. These queries are not single-group probes: 1000 queries cover `258.97` matched groups on average, with p50 `32` and p95 `1087.4`; matched points average `55556.6`, with p50 `6800` and p95 `244500`. This experiment first exposed a heavy-prune failure and then evaluated a new light-prune path for medium groups.

| Variant | Index ms | Group ms | Cross ms | L100 recall | L200 recall | L500 recall | L1000 recall |
|---|---:|---:|---:|---:|---:|---:|---:|
| CPU Vamana group | `17596.8` | `8511.72` | `4601.91` | `0.8230` | `0.8250` | `0.8490` | `0.8690` |
| FastGrnndCuda, old heavy prune | `38273.7` | `26620.6` | `6055.72` | `0.8125` | `0.8273` | `0.8472` | `0.8662` |
| FastGrnndCuda, top32 light prune | `14211.8` | `6016.04` | `4926.37` | `0.8135` | `0.8239` | `0.8421` | `0.8618` |
| FastGrnndCuda, diversified light prune | `16478.5` | `6706.36` | `4857.33` | - | - | - | `0.8656` |
| batched exact kNN, pure local | `12595.9` | `4356.18` | `4120.75` | - | - | - | `0.8159` |

The difference is dominated by pruning:

| Variant | H2D | GNN | Prune | D2H | Group total |
|---|---:|---:|---:|---:|---:|
| old heavy prune | `1538.32` | `3551.54` | `19218.3` | `413.047` | `26620.6` |
| top32 light prune | `1180.47` | `3198.35` | `712.392` | `134.631` | `6016.04` |
| diversified light prune | `1448.82` | `3229.56` | `759.66` | `161.408` | `6706.36` |
| batched exact kNN, pure local | `524.173` | `674.693` | `0` | `29.8523` | `4356.18` |

At higher search budgets, light prune keeps improving but remains below CPU Vamana:

| Variant | L1000 | L2000 | L5000 |
|---|---:|---:|---:|
| CPU Vamana group | `0.8690` | `0.8820` | `0.9080` |
| FastGrnndCuda, top32 light prune | `0.8619` | `0.8769` | `0.9029` |
| FastGrnndCuda, diversified light prune | `0.8656` | - | `0.9056` |
| batched exact kNN, pure local | `0.8159` | - | `0.8199` |

The conclusion is more useful than a simple speed table. Heavy RNG-style pruning is too expensive for the `nx≈200` bucket, but diversified light pruning makes GPU group construction faster than CPU Vamana (`1.27x` for group graph and `1.07x` for index time). The earlier batched exact-kNN ablation is now treated as confounded because its cross-edge count does not match the CPU or new packed-exact indexes. With full-quality cross/additional edges and packed writeback, exact-anchor reaches CPU-level L5000 recall on x200 while reducing group construction to `3827.28 ms`; after direct-H2D and fill-thread tuning, the same x200 exact-anchor route reaches `3204.88 ms` group time in the best measured run without changing recall. We also tested a more warp-centric exact-anchor kernel (`UNG_FAST_EXACT_WARP_KERNEL=1`) and found it to be a negative ablation on x200: the exact kernel worsened from `841.136 ms` to `987.27 ms`, and group time worsened from `4031.36 ms` to `4334.28 ms` with unchanged L1000 recall. Fill-thread scaling was also negative: the default 16-thread fill (`1682.11 ms`) beat both 8 threads (`2113.86 ms`) and 64 threads (`2321.96 ms`). The remaining question is no longer whether exact-anchor can work on this coverage workload, but how to remove the CPU-compatible graph materialization boundary and where the router should switch to GNN/reverse-tail paths on other group-size and label distributions.

### 5.11 Existing Query-Level Sanity Check for Cross-Edge Backends

We also have an existing Amazon PF query run that compares three indexes while keeping the intra-group graph backend fixed to CPU Vamana. This does not validate FastGrnndCuda, but it checks whether changing the cross-edge backend changes final filtered-search recall.

| Variant | Lsearch | Average recall | Batch time ms | Query P95 ms |
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

The average recall differences are within roughly `3e-4`, which is consistent with the cross-edge backend preserving query quality when the group graph is unchanged. This run does not exercise FastGrnndCuda, so we treat it as a cross-edge sanity check rather than the final recall evaluation.

## 6. Discussion

### 6.1 Why Per-Group cuVS Underperforms

cuVS provides efficient brute-force GPU kernels, but UNG creates many irregular group tasks. Per-group invocation repeatedly pays packing, transfer, API, launch, and conversion costs. This is why per-group cuVS takes `3730.5 ms` on x100 while the fused path takes `848.9 ms`.

### 6.2 Why SGEMM+TopK Remains Competitive

SGEMM+topK is a strong baseline because cuBLAS/cuBLASLt provide highly optimized Tensor Core execution. The fused path improves end-to-end time by avoiding full distance-matrix materialization, separate topK, unnecessary distance writeback, and query expansion. The `1.64x` speedup over SGEMM+topK should be interpreted as a workload-level fusion result, not as a claim that the custom kernel dominates cuBLAS on dense GEMM.

### 6.3 Data Residency Cost

`prepare_all` is a one-time cost per build process. It should be included in single-build timing but can be separated when analyzing resident cross-edge compute. This distinction is important because host registration varies across runs and can otherwise obscure kernel-level improvements.

### 6.4 Direct-QID Stability at x400

The x400 direct-qid path currently triggers illegal memory access in the medium group-desc kernel. Reducing `UNG_GPU_FLAT_Q_CAP_MB` to 512 MB still fails, which suggests that the problem is not merely a chunk-size issue. We therefore use the stable gather-Q fused path for x400 and list direct-qid stability as future work.

## 7. Related Work Positioning

### 7.1 GPU Brute Force and Dense Linear Algebra

cuBLAS, cuBLASLt, and cuVS provide strong dense GPU primitives. Our work is complementary: we target the irregular multi-group topK workload created by filtered graph construction, where orchestration and writeback overheads are first-order costs.

### 7.2 GPU Graph Construction

Tagore-style GPU graph construction uses GPU GNN-Descent and pruning to build ANN graphs. GRNND/RNN-Descent integrates RNG-style pruning and neighbor propagation more deeply into graph construction. FastGrnndCuda borrows the insight that reverse-neighbor information and RNG-style occlusion improve navigability, but implements them as a lightweight pruning backend on top of GNN-Descent candidates for UNG group graphs.

### 7.3 Filtered ANN Indexes

Filtered ANN indexes must manage both vector proximity and attribute reachability. UNG addresses this by adding label-graph structure and cross-group edges. This paper focuses on accelerating construction; query-time optimization and end-to-end recall evaluation are complementary directions.

## 8. Limitations and Future Work

1. **End-to-end search recall on repeat-scale workloads.** We now have group-graph A/B evidence on original Amazon PF, SIFT30, Amazon 1% x100, Amazon 1% x200 coverage, Amazon 10% x40, and Amazon 1% x400 gather-Q. The x100 result closes the recall question but shows only small group-build speedup because most groups still use CPU fallback. The x200 result shows that adaptive pruning strength is necessary: light prune fixes performance but leaves a recall gap. The 10% x40 full-quality repeat=3 run shows a `1.53x` index-time speedup without L100/L500/L1000 recall or P95/P99 latency loss. The x400 reverse-tail + repair run raises L5000 from light512's `0.9574` to `0.9659`, close to the same-script CPU baseline `0.965`, with about `1.10x` index-time speedup; compact-D2H has been measured as a negative/ambiguous overhead ablation. A publishable evaluation still needs more Lsearch/efs points, more x400 parameter sweeps, and a stronger real multi-label CPU/GPU method A/B. The current CelebA artifact is a regenerated-GT sanity check, not yet a main-table GPU-method result.
2. **Direct-qid stability at larger scale.** x400 currently requires gather-Q fused execution.
3. **CPU-compatible output boundary.** The current implementation still writes GPU-produced graph and cross-edge results back into host `Graph::neighbors`/`SearchQueue`/Vamana-compatible structures. The direct-H2D exact-anchor A/B removes the input-side host pack (`4140.61 ms` to about `0.04 ms`), but the output-side graph materialization was still seconds in earlier runs; this shows that the remaining boundary is host graph materialization, not the exact kernel. A Graph reserve A/B further confirms that per-node host adjacency allocation is a first-order cost: on Amazon 10% x40, reserve reduces index time from `39746.1 ms` to `24704.5 ms`, `tagore_fill` from `823.5 ms` to `23.9 ms`, and cross merge from `1677.1 ms` to `2.3 ms`. However, the old reserve path itself allocates capacity for `106,031,200` edges and costs `6487.3 ms`; it moves allocator cost earlier rather than replacing the host graph representation. The retained 64-inline `NeighborList` small-buffer implementation absorbs most of that allocator cost while preserving the CPU-compatible API: on x200, reserve falls from `2856.2 ms` to `25.4 ms` and fill from `224.8 ms` to `14.7 ms`; on 10% x40, reserve falls from `6487.3 ms` to `21.2 ms` and fill from `23.9 ms` to `7.6 ms`. A 48-inline variant is a negative ablation, with x200 fill regressing to `202.5 ms`. Thus small-buffer adjacency is a retained low-risk system optimization, but it is still a per-node host object rather than a GPU-native CSR backend. A warp-per-source exact kernel and wider/narrower fill-thread settings were negative ablations, so further microtuning of this compatibility layer is unlikely to produce a general replacement. We tested a more aggressive Amazon 10% x40 path with `UNG_GPU_DB_NOSPLIT=1`, `UNG_GPU_DB_GLOBAL_MERGE=1`, and `UNG_GPU_ID_VECTOR_WRITEBACK=1`: D2H/writeback fell to `3.5/33.2 ms`, but qid-lock contention increased kernel time from `2952.2 ms` to `4834.3 ms`, and cross-edge time worsened from `9451.1 ms` to `11598.0 ms`. Thus simple locked global merge is a negative ablation. Partial double-buffer routing is a lower-risk improvement: it reduces x200 cross-edge time to `2657.23 ms` while keeping host `SearchQueue` semantics. A newer direct-global/flat-id smoke test further separates the boundary: direct-global only removes a small x100 skip-additional offset pass (`2.5 ms` to `0.0 ms`), whereas flat-id writeback reduces the same cross-edge path from `1650.95 ms` to `707.83 ms`, D2H from `11.1 ms` to `0.6 ms`, and merge from `17.2 ms` to `0.5 ms` with unchanged recall `0.816`. This is still a skip-additional smoke test and is incompatible with CPU Vamana additional repair, so it is not a full-quality main-table result. We also tried directly appending full-quality additional edges into host `Graph::neighbors`; the x400 runs did not complete, including a lock-guarded attempt that stopped after GPU `batched_search end`. This negative ablation argues against ad hoc host-graph mutation and in favor of staged flat segments. A general replacement still needs qid-sharded or segmented no-lock merge plus a flat-adjacency/CSR graph backend.
4. **Pruning cost and quality gap.** FastGrnndCuda heavy prune takes `19977.7 ms` on x400 and `19218.3 ms` in the old x200 heavy-prune run. Diversified light prune reduces x200 prune to `759.66 ms`, and light512 reduces x400 prune to about `1006 ms`, but the original x400 light path loses `0.0081` recall. A stored-graph diagnostic on x400 shows that the light512 gap is structural: cross edges are unchanged, but intra-group edges drop from `75,767,700` to `73,539,000`, zero intra-degree points appear (`0.00199162`), the low intra-degree `<=4` ratio rises to `0.00929781`, and `39` groups have largest weak component ratio below `0.9`. Degree repair alone mostly fixes low-degree structure but is too costly and gives little recall gain; reverse-tail + repair adds only modest prune cost (`1093.37 ms`) and recovers most of the recall gap. The old exact-kNN low-recall result is now treated as confounded by cross-edge/additional-edge differences, while packed exact-anchor is a strong candidate on x200/x400 coverage queries.
5. **Adaptive pruning policy.** The fastest pipeline uses CPU fallback for many smaller groups and light prune for medium groups. A more principled light/strong-prune router, or a quality-enhanced medium-group GPU builder, would strengthen the system contribution.
6. **Dataset coverage.** SIFT30 now has a repaired end-to-end A/B, and CelebA has a regenerated-GT sanity artifact. However, the strongest performance results are still on Amazon sampled + jitter-repeat workloads. A stronger CelebA or other real multi-label CPU/GPU method A/B is still needed for a complete paper.

## 9. Conclusion

UNG construction exposes a GPU workload that is irregular, group-oriented, and sensitive to host-side overheads. Treating cross-edge construction as independent per-group library calls leaves substantial performance on the table. By batching group-level topK searches, using direct query-id access to resident vectors, fusing TF32 distance computation with topK, and reducing writeback, the proposed cross-edge path reaches `848.9 ms` on Amazon 1% x100 and provides `22.07x` speedup over CPU Vamana cross-edge construction. Combined with FastGrnndCuda and a workload-aware fallback policy, the current pipeline reduces Amazon 1% x100 end-to-end index build time to `6630 ms`. The x200 coverage-query result shows why the group-graph backend needs workload-aware routing: diversified light prune turns a heavy-prune failure into a faster-than-CPU build, and packed exact-anchor further reaches CPU-level L5000 recall with lower group construction time once full-quality cross/additional edges are fixed. Partial double-buffer routing also shows that cross-edge slowdowns can come from fast-path coverage boundaries rather than from the fused kernel alone: splitting supported and unsupported target groups reduces x200 cross-edge time from `3900.81 ms` to `2657.23 ms`. The 10% x40 many-group test further shows that a general replacement must optimize data residency, direct-qid execution, descriptor scheduling, fallback policy, and additional edges together; in the full-quality setting it provides `1.53x` index-time speedup without L100/L500/L1000 recall or P95/P99 latency loss. The bounded-complete, Graph reserve, and NeighborList experiments show that very small groups should not be blindly moved to GPU exact construction; the system must reduce CPU graph materialization and host adjacency allocation. Reserve proves allocator cost is first-order, and the small-buffer NeighborList removes most of that allocator cost, but the graph is still a CPU-compatible per-node representation rather than a flat GPU-native backend. The failed qid-lock global-merge ablation shows that the remaining output boundary cannot be removed by a simple locked device merge; it needs a no-lock segmented/CSR design. The x400 gather-Q run gives the strongest current counterexample for one-size-fits-all pruning: light pruning makes the build faster but costs `0.0081` recall, while heavy pruning preserves quality but is slower than CPU. The remaining bottlenecks are clear: cross-dataset packed/router validation, x400 direct-qid stability, CPU-compatible output boundaries, and graph-quality routing design.

## Artifact and Log References

Technical report:

```text
docs/reports/TECHNICAL_REPORT_OPTIMIZATION_SPEEDUP_CN.md
```

Latest full optimized logs:

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

Main baselines:

```text
/tmp/fv_amazon_1pct_x100_cpu_vamana/run.log
/tmp/fv_amazon_1pct_x100_cpu_exact/run.log
/home/graphdb/FilterVectorBenchResults/onchip_rewrite_20260527/x1p100_cpu_group_sgemm_topk_baseline/run.log
/home/graphdb/FilterVectorBenchResults/onchip_rewrite_20260527/x1p100_cpu_group_cuvs_pergroup_baseline/run.log
/home/graphdb/FilterVectorBenchResults/onchip_rewrite_20260527/x100_hybrid_default/run.log
```

Group graph summaries:

```text
docs/reports/FINAL_GROUP_GRAPH_METHOD_CN.md
docs/reports/FASTGRNNDCUDA_AMAZON_SCALE_RESULTS_CN.md
```

## Artifact Reproducibility Map

| Paper claim / reviewer question | Reproduction entry point | Main artifact | Current status |
|---|---|---|---|
| Is the x100 cross-edge speedup only against a weak CPU baseline? | `tools/benchmarks/summarize_cross_baselines.py` | `/home/graphdb/fv_runs/baseline_fairness_x100_20260602/cross_baselines.md` | Done; includes CPU Vamana, 128-thread CPU exact, cuVS per-group, SGEMM+topK, and final fused |
| Does FastGrnndCuda preserve end-to-end search recall? | `scripts/benchmarks/run_end_to_end_recall_ab.sh` | Per-run `results/search_time_summary.csv` and `results/query_details_repeat*.csv` | Partial evidence on Amazon PF, SIFT30, x100, x200, 10%x40, and x400; real multi-label remains missing |
| Does the x400 light-prune recall gap come from intra-group graph structure? | `tools/benchmarks/diagnose_ung_graph_structure.py` and `tools/benchmarks/summarize_graph_diagnostics.py` | `/home/graphdb/fv_runs/graph_diagnostics_x400_20260602/summary_table.md` | Done for historical CPU vs light512 |
| Can x400 repair / reverse-tail recover quality? | `scripts/benchmarks/run_x400_reverse_tail_ab.sh` | `/home/graphdb/fv_runs/x400_reverse_tail_ab_20260602_030908/x400_ab_summary.md` and `x400_ab_summary.csv` | Measured; reverse-tail + repair recovers most of the light512 recall gap, while repair-only is partial/negative |
| Is compact-D2H only an overhead optimization? | `RUN_COMPACT_ABLATION=1 scripts/benchmarks/run_x400_reverse_tail_ab.sh` | `/home/graphdb/fv_runs/x400_compact_ab_20260602_080045/x400_ab_summary.md` | Completed negative/ambiguous ablation; should not be described as a graph-quality contribution or x400 speedup source |
| Can the mixed exact/GNN router accelerate small/medium groups without recall loss? | `scripts/benchmarks/run_group_graph_router_ab.sh` and `tools/benchmarks/summarize_group_graph_router_ab.py` | `/home/graphdb/fv_runs/group_graph_router_ab_streamfill_20260602_030037/router_ab_summary.md` and `router_ab_summary.csv` | Measured on x200 repeat=1; repeat=3 and cross-dataset confirmation are still needed |
| Does partial double-buffer routing fix cross-edge all-or-nothing fallback? | `scripts/benchmarks/run_end_to_end_recall_ab.sh` with `UNG_GPU_DB_NOSPLIT=1` and `UNG_GPU_DB_SPLIT_UNSUPPORTED=1` | `/home/graphdb/fv_runs/target_fused_x200_db_split_20260602_060034/end_to_end_recall_ab_20260602_060034`; graph diagnosis `/home/graphdb/fv_runs/graph_diag_x200_db_split_20260602` | Measured; x200 cross-edge `3900.81 -> 2657.23 ms`, repeat=3 L1000/L5000 `0.866/0.907`, but quality remains below source-centric |
| Is x400 direct-qid stable? | Direct-qid strict rerun or formal gather-Q/direct-qid routing policy | Successful strict log or explicit limitation | Currently a limitation; x400 main tables use gather-Q |

This table is intentionally conservative: claims without an artifact stay in the reviewer-audit or limitation sections rather than the main result tables.
