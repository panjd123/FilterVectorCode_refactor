# FilterVectorCode 工作展示稿：按创新点组织的方法、Baseline、数据集与性能

## 0. 展示口径

这份文档按“创新点/计算步骤”组织，而不是按时间线组织。每个创新点都回答：

- 这个计算步骤是什么，为什么慢；
- 我们的方法是什么，有哪些关键超参；
- baseline 是什么，有哪些关键超参；
- 数据集有什么特点；
- 耗时、加速比和质量边界是什么。

最终质量标准：建图后的 filtered-search recall / latency。局部 topK overlap 只作为诊断，不能替代最终查询质量。

## 1. 总表：当前最应该展示的结果

| 创新点 / 计算步骤 | 我们的方法 | 关键超参 | baseline | 数据集特点 | 当前耗时 | 加速比 / 收益 | 质量边界 |
|---|---|---|---|---|---:|---:|---|
| Cross group edge | GPU grouped fused topK + SearchQueue lazy reserve | `num_cross_edges=6`, `max_degree=32`, GPU batched route | CPU hybrid scan/Vamana 128T | Amazon `1%x200`, full-quality | GPU cross `2842.36 ms` | `13.14x` vs `37338.0 ms` | 保留 additional/search recall；stage 加速不等同完整 Index |
| Cross-edge fairness | GPU grouped fused topK | same topK, same grouped workload | CPU exact 128T / cuVS per-group / SGEMM+topK | Amazon `1%x100`, stage A/B | GPU cross `848.9 ms` | `2.82x` vs CPU exact; `4.39x` vs cuVS; `1.64x` vs SGEMM | 证明不是只打弱 baseline |
| ELS / 入口组 | GPU correct-cover + fused compact | `frontier_delta=2`, `cap=1024` 示例 | CPU cover-frontier same quality / CPU exact minimal | Amazon `100%x40`, `nq=10240` | `97.80 ms / 1337 groups` | `2.97x` vs CPU `290.70 ms / 1337 groups` | provider microbenchmark；输出 groups 越少越好且不能漏 |
| PG / group graph | Workload-aware GPU route / FastGrnndCuda | group-size route, `max_degree=32`, FastGrnndCuda | CPU Vamana | Amazon `1%x200`, full-quality | group `1837.34 ms` | `4.42x` vs `8119.73 ms` | 高 L recall 对齐，低 L 小差距 |
| PG stress | FastGrnndCuda route | 同上 | CPU Vamana | Amazon `10%x40`, full-quality stress | group `5147.2 ms`, Index `23497.4 ms` | group `3.54x`, Index `1.53x` | L100/L500/L1000 recall 不低于 CPU |
| Additional/output boundary | CPU exact materialized additional + lazy reserve | exact materialized, SearchQueue lazy reserve | CPU Vamana additional / eager reserve | Amazon `1%x200` / `10%x40` | additional `831.7 ms`; output `15.8 ms` | `7.16x`; `1909.9 -> 15.8 ms` | 系统边界优化，不是 GPU-native search |
| Special block intra/inter | special sidecar + free-state search + GPU intra/inter | T=100, CPU intra th=2048, GPU intra th=128, GPU inter warps=2, cap10/iter4 | corrected CPU exact-topK/Vamana intra + CPU inter | Amazon `100% x1 restored` | GPU/GPU overlay `~7.87 s`, Index `27.8-28.6 s` | intra `~14.4x`; inter `~4x` | 结构候选；cap10/iter4 仍是近似 knob |
| Special save/load | binary-only sidecar + skip reordered + CPU-provider-only skip LNG text | binary-only, skip reordered, skip LNG text | CSV sidecar / full save | Amazon `100% x1 restored` | save `19.10 s` | `40.18 -> 19.10 s` | skip LNG text 不适用于 GPU cover-frontier |

## 2. 创新点一：Cross Group Edge GPU grouped fused topK

### 2.1 这个计算步骤是什么

UNG 的 label navigation graph 会要求不同 group 之间连接 cross edges。朴素做法是对每个 source group / target group 单独做 topK 搜索。慢点在于：

- group pair 数量多；
- 每组单独 launch 或单独调用库函数有调度开销；
- SGEMM+topK 会物化中间矩阵；
- CPU Vamana / exact scan 在大规模 group pair 上很慢；
- 最终还要写回 CPU-compatible graph。

### 2.2 我们的方法

我们把 group-to-group workload 合并成 descriptor batch，在 GPU kernel 内直接做距离计算和 topK。这样避免逐组 launch、逐组 cuVS 调用和中间矩阵物化。SearchQueue lazy reserve 进一步减少 CPU 写回阶段的 eager 分配。

关键超参 / 配置：

| 配置 | 值 / 说明 |
|---|---|
| topK | `num_cross_edges=6` |
| graph degree | `max_degree=32` |
| route | GPU batched grouped fused topK |
| output | SearchQueue lazy reserve |
| quality | full-quality 时保留 additional_edges，并用 filtered-search recall 检查 |

### 2.3 Baseline

| Baseline | 关键超参 | 作用 |
|---|---|---|
| CPU exact scan 128T | 128 threads | 强 CPU brute-force baseline |
| CPU hybrid scan/Vamana 128T | threshold=1e6 in x200 sweep | 当前最快 CPU cross baseline |
| cuVS per-group | per-group library call | GPU library baseline |
| SGEMM+topK | materialized GEMM + separate topK | 常见 dense baseline |
| CPU Vamana cross | Vamana search | 历史弱 baseline，只作参考 |

### 2.4 数据集与结果

| 数据集 | 特点 | 方法耗时 | baseline 耗时 | 加速比 |
|---|---|---:|---:|---:|
| Amazon `1%x100` | fairness stage A/B，比较多种实现 | `848.9 ms` | CPU exact `2389.8 ms` | `2.82x` |
| Amazon `1%x100` | 同上 | `848.9 ms` | cuVS per-group `3730.5 ms` | `4.39x` |
| Amazon `1%x100` | 同上 | `848.9 ms` | SGEMM+topK `1395.9 ms` | `1.64x` |
| Amazon `1%x200` | full-quality，保留 additional/search recall | `2842.36 ms` | CPU hybrid `37338.0 ms` | `13.14x` |

展示时说：我们不是只和慢 CPU Vamana 比，而是同时打了 CPU exact、CPU hybrid、cuVS 和 SGEMM+topK。

## 3. 创新点二：ELS / 入口组 GPU correct-cover provider

### 3.1 这个计算步骤是什么

查询时需要从 query labels 找到 entry groups。CPU exact minimal 可以输出最少入口组，但代价高。GPU provider 的目标不是直接提升最终 recall，而是更快地产生 coverage-correct 的入口组。

质量标准：

- coverage-correct：不能漏掉任何合法候选 group；
- output groups 越少越好；
- provider 质量不等同最终 ANN topK recall。

### 3.2 我们的方法

把 group labels、LNG descendants、coverage 关系放到 GPU bitset/frontier 表中，对 batch query 并行计算入口组，并用 fused compact 减少输出压缩开销。

关键超参：

| 配置 | 含义 |
|---|---|
| `frontier_delta` | 允许在 query label size 附近扩展 frontier，影响覆盖质量和计算量 |
| `cap` | 控制输出候选规模和 GPU compact 成本 |
| fused compact | 减少中间输出和 D2H 开销 |

### 3.3 Baseline

| Baseline | 作用 |
|---|---|
| CPU exact minimal | 最少入口组质量上界，但慢 |
| CPU cover-frontier same quality | 公平性能 baseline |
| CPU scan weak baseline | 只能作诊断，不能作为主 claim |

### 3.4 数据集与结果

| 数据集 | 特点 | 我们的方法 | baseline | 加速比 |
|---|---|---:|---:|---:|
| Amazon `100%x40`, `nq=10240` | provider microbenchmark，放大 batch throughput | `97.80 ms / 1337 groups` | CPU same-quality `290.70 ms / 1337 groups` | `2.97x` |
| Amazon `100%x40`, `nq=10240` | 更重 OR / memory-bound 配置 | `385.50 ms / 1280 groups` | CPU `1053 ms / 1280 groups` | `2.73x` |

展示时说：GPU 在中高质量区间有 2.7-3.0x；但 production 当前仍是 per-query provider，端到端 batch query 还要补。

## 4. 创新点三：PG / Group Graph workload-aware GPU route

### 4.1 这个计算步骤是什么

每个 group 内要构建 proximity graph。不同 group size 的最优策略不同：小 group 上 GPU packing/fill 开销可能大于收益，大 group 上 CPU Vamana 很慢。

### 4.2 我们的方法

按 group size 和质量风险分流：

- small group：CPU / bounded fallback；
- medium / large group：GPU Tagore/FastGrnndCuda；
- 对低 L recall 做 reverse-tail / diversity / repair 类质量修复。

### 4.3 Baseline

CPU Vamana group graph 是 full-quality baseline。质量必须看建图后 filtered-search recall。

### 4.4 数据集与结果

| 数据集 | 特点 | 我们的方法 | CPU baseline | 加速比 | 质量 |
|---|---|---:|---:|---:|---|
| Amazon `1%x200` | full-quality | group `1837.34 ms` | CPU Vamana `8119.73 ms` | `4.42x` | 高 L recall 对齐，低 L 小差距 |
| Amazon `10%x40` | stress / many groups | group `5147.2 ms` | CPU Vamana `18218.5 ms` | `3.54x` | L100/L500/L1000 不低于 CPU |
| Amazon `10%x40` | same run, full Index | Index `23497.4 ms` | CPU Index `36053.3 ms` | `1.53x` | full-quality 正结果 |

展示时说：这条线的创新是 router，而不是单个 GPU kernel。

## 5. 创新点四：Full-quality additional / output boundary

### 5.1 这个计算步骤是什么

full-quality 构建不只包含 GPU cross-edge kernel。还要补 additional_edges，并把结果写回 CPU-compatible `Graph::neighbors` / SearchQueue / NeighborList 结构。

这一步是典型 Amdahl 边界：kernel 快了，但输出物化慢，端到端加速仍会被吃掉。

### 5.2 我们的方法

- additional_edges：从 CPU Vamana additional 改成 CPU exact materialized；
- output storage：SearchQueue lazy reserve，避免对所有点 eager reserve。

### 5.3 Baseline 与结果

| 子步骤 | 我们的方法 | baseline | 耗时 | 加速 / 收益 |
|---|---|---|---:|---:|
| additional_edges | CPU exact materialized | CPU Vamana additional | `831.7 ms` vs `5954.2 ms` | `7.16x` |
| output storage | SearchQueue lazy reserve | eager reserve | `15.8 ms` vs `1909.9 ms` | `~121x` |

展示时说：这解释了为什么我们不能只报 GPU kernel 时间，full-quality 的系统边界同样重要。

## 6. 创新点五：Special Block sidecar graph + free-state query

### 6.1 这个计算步骤是什么

Amazon x1 label tree 很稀疏，很多 query 会完整覆盖一些 label subtree。Special block 把这些 subtree 当成特殊 group，单独构建 sidecar graph，并在查询时允许 free-state 节点优先走 special edges。

它解决的是稀疏 x1 label tree 下的结构问题，不适合直接套在 x40/x200 repeat 数据上。

### 6.2 我们的方法

| 子步骤 | 方法 | 关键超参 |
|---|---|---|
| block 构造 | bottom-up uncovered-points frontier | `UNG_SPECIAL_BLOCK_MIN_POINTS=100` |
| intra | CPU exact-topK / GPU Tagore-FastGrnnd split | CPU `th=2048`; GPU `th=128` |
| inter | source-exact cuda-core special inter | `UNG_SPECIAL_GPU_INTER_WARPS=2` |
| query | free-state search | `UNG_SPECIAL_BLOCK_SEARCH=1`, optional `FREE_USE_REGULAR=1` |
| save/load | binary-only sidecar | `UNG_SPECIAL_EDGE_BINARY_ONLY=1` |

### 6.3 Baseline

| Baseline | 作用 |
|---|---|
| old CPU Vamana intra | 历史慢 baseline，不再作为公平口径 |
| corrected CPU exact-topK/Vamana intra | 当前 CPU intra baseline |
| CPU inter | 当前 CPU inter baseline |
| normal no-special UNG | 判断 special 是否能形成端到端 Index speedup的基准 |

### 6.4 2x2 构建表

单位秒。

| intra / inter | CPU inter | GPU inter |
|---|---:|---:|
| CPU intra | index `105.78`; intra `75.15`; inter `10.36`; overlay `85.50` | index `98.95`; intra `73.79`; inter `6.07`; overlay `79.86` |
| GPU intra | index `38.60`; intra `7.07`; inter `11.00`; overlay `18.07` | index `28.62`; intra `5.16`; inter `2.72`; overlay `7.87` |

### 6.5 Intra 调参

| 路径 | 耗时 | quality smoke | 结论 |
|---|---:|---:|---|
| CPU 全 Vamana | `155.92-160.30 s` | `0.344 / 0.401` | 旧慢 baseline |
| CPU exact-topK `th=2048` | `75.64-75.88 s` | `0.341 / 0.400` | corrected CPU baseline |
| GPU split `th=128` | `5.11-5.28 s` | `0.344 / 0.403` | 当前 GPU 默认 |
| bounded ring `th=2048` | `41.94 s` | `0.323 / 0.370` | 负结果 |

公平加速：GPU intra 约 `14.4x` vs corrected CPU。

### 6.6 Inter kernel 调参

| variant | kernel | inter total | 结论 |
|---|---:|---:|---|
| cuda-core 8 warps/query | `~5.34 s` | `~6.1 s` | 旧默认，过配 |
| cuda-core 2 warps/query | `~1.87 s` | `~2.7 s` | 当前默认 |
| TF32 WMMA/shared tile | `33.53 s` | `34.49 s` | 负结果，tiles 太碎 |

公平加速：GPU inter 约 `4x` vs CPU inter。

### 6.7 Special block 的展示边界

能说：special block 是一个很有潜力的结构候选，GPU intra/inter stage 已经明显优化。

不能说：special block 已经是默认 full-quality 端到端加速方法。

原因：cap10 / iter4 / two-tier heavy sidecar 仍是候选近似路线，需要更多 workload 和更严格同进程 A/B。

## 7. 数据集特点总结

| 数据集 | 用途 | 需要说明的特点 |
|---|---|---|
| Amazon `1%x100` | cross-edge fairness | stage-level；比较 CPU exact、cuVS、SGEMM+topK 等 baseline |
| Amazon `1%x200` | cross-edge / PG full-quality | 当前 full-quality 主证据之一 |
| Amazon `10%x40` | group graph stress | many-group / group-size route 压力更明显 |
| Amazon `100%x40` | ELS provider | provider microbenchmark，不等同端到端 query |
| Amazon `100% x1 restored` | special block | x1/original 语义诊断；不要包装成官方原始 x1 |

## 8. 最后怎么收束

展示最后建议强调三句话：

1. 我们不是单点 kernel 优化，而是在 UNG 的多个瓶颈阶段做分解优化。
2. 最清楚的主结果是 cross-edge `13.14x`、PG group `4.42x`、special intra/inter 的强 stage 优化。
3. 真正端到端还受 full-quality output boundary、entry-provider 接入、special 近似质量验证限制；所有最终 claim 必须用 filtered-search recall 说话。
