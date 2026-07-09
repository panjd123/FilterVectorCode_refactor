# 五条优化主线与 full-quality 输出边界：方法、baseline、数据集与加速比

更新时间：2026-07-07

本文把当前工作拆成五条相互独立但会在端到端系统中相互影响的主线，并把 full-quality 下的 `additional_edges` / CPU-compatible graph materialization 单独列为系统边界章节：

1. cross group edge 加速。
2. GPU 加速提取 ELS，即查询入口组。
3. 加速构建 PG/group graph。
4. full-quality additional_edges 与 CPU-compatible graph materialization。
5. 特异块图构建与自由状态查询。

每条线都按“方法、baseline、数据集/构造形态、加速比”整理。表中 `full-quality` 表示：构建时保留当前完整 UNG 语义，尤其是 `additional_edges`；构建后用 filtered brute-force ground truth 跑 `search_UNG_index`，用最终查询结果的 recall / latency 检查索引质量。`microbenchmark` 只说明局部算子或 provider；`diagnostic` 用于解释瓶颈或规模边界，不能直接写成端到端主结论。

本文里的质量标准有一个硬约束：PG/group graph 或 cross-edge 的局部 topK overlap 只能作为诊断，不是索引质量的黄金标准。过滤向量搜索的最终质量只能由“建图后端到端查询结果”验证，即固定 query、GT、`Lsearch` 和 search route 后比较 filtered search recall、latency 和查询细节。简单保留最近邻 topK 甚至可能损害导航图需要的多样性和连通性，因此不能用局部 topK 准确率替代端到端查询质量。

## 0. 最终性能大表

本节只放当前最应该引用的最终口径。旧实验、负结果和诊断数据仍保留在后续章节，用于解释为什么不能 overclaim。

### 0.1 五条主线最终结果

| 主线 | 当前推荐方法 | 公平 baseline | 数据集/口径 | 当前性能 | 加速比/收益 | 质量与边界 |
|---|---|---|---|---:|---:|---|
| Cross group edge | GPU grouped fused topK + SearchQueue lazy reserve | CPU hybrid scan/Vamana 128T | Amazon 1%x200 full-quality | cross `2842.36 ms` | `13.14x` vs CPU hybrid `37338.0 ms` | 保留 full-quality additional/search recall；仍受 prepare_all/H2D 与 CPU-compatible output boundary 限制 |
| Cross group edge fairness | GPU grouped fused topK | CPU exact 128T / cuVS / SGEMM+topK | Amazon 1%x100 stage A/B | cross `848.9 ms` | `2.82x` vs CPU exact, `4.39x` vs cuVS, `1.64x` vs SGEMM+topK | stage-level fairness，不等同完整 Index speedup |
| ELS / 入口组 | GPU correct-cover + fused compact | CPU cover-frontier same quality / CPU exact minimal | Amazon 100%x40 provider microbenchmark | 中高质量 `~1300-1500` groups | `2.7-3.0x` vs same-quality CPU | 质量是 coverage-correct 且 entry groups 越少越好；缺 batch/end-to-end query A/B |
| PG/group graph | workload-aware GPU route / FastGrnndCuda | CPU Vamana group graph | Amazon 1%x200 full-quality | group `1837.34 ms` | `4.42x` vs `8119.73 ms` | 高 L recall 对齐，低 L 有小幅差距；不能写 GPU 无条件替代 CPU Vamana |
| PG/group graph stress | FastGrnndCuda route | CPU Vamana group graph | Amazon 10%x40 full-quality | group `5147.2 ms`, Index `23497.4 ms` | group `3.54x`, Index `1.53x` | L100/L500/L1000 recall 不低于 CPU baseline |
| additional/output boundary | CPU exact materialized additional + lazy reserve | CPU Vamana additional / eager SearchQueue reserve | Amazon 1%x200 / 10%x40 boundary A/B | additional `831.7 ms`; output storage `15.8 ms` | additional `7.16x`; output storage `1909.9 -> 15.8 ms` | 这是 CPU-compatible graph 边界优化，不是 GPU-native flat/CSR search backend |
| special block graph/query | special block + free-state search + GPU intra/inter tuned | corrected CPU exact-topK/Vamana intra + CPU inter | Amazon 100% x1 restored, T=100, cap10/iter4 | GPU/GPU overlay `7.87-7.88 s`, Index `27.8-28.6 s` | intra `~14.4x` vs corrected CPU; inter `~4x` vs CPU inter | cap10/iter4 仍是近似构建 knob；必须用 filtered-search recall 验证，不能直接宣称 full-quality 默认 |
| special save/load | binary-only sidecar + skip reordered export + CPU-provider-only skip LNG text | CSV sidecar / full save | Amazon 100% x1 restored | save `19.10 s` | full save `40.18 -> 19.10 s` | `UNG_SKIP_LNG_TEXT_SETS=1` 只适用于 CPU entry-provider/UNG special search，GPU cover-frontier 仍需要 `_lng_descendants` |

### 0.2 Special Block 当前 2x2 构建表

当前默认分流：CPU intra 使用 distance exact-topK `th=2048` + 少量 CPU Vamana；GPU intra 使用 exact-topK `th=128` + Tagore/FastGrnnd CUDA；GPU inter 使用 source-exact cuda-core `2` warps/query。下表单位是秒。

| intra / inter | CPU inter | GPU inter |
|---|---:|---:|
| CPU intra | index `105.78`; intra `75.15`; inter `10.36`; overlay `85.50` | index `98.95`; intra `73.79`; inter `6.07`; overlay `79.86` |
| GPU intra | index `38.60`; intra `7.07`; inter `11.00`; overlay `18.07` | index `28.62`; intra `5.16`; inter `2.72`; overlay `7.87` |

注意：`GPU intra + CPU inter` 这一格来自单独重跑的有效结果；第一次 2x2 脚本有 `UNG_SPECIAL_BLOCK_GPU_INTER` 环境变量残留，已丢弃。

### 0.3 Special Intra / Inter 调参结论

| 项目 | 旧口径 | 修正后 CPU | 当前 GPU | 当前结论 |
|---|---:|---:|---:|---|
| special intra | CPU 全 Vamana `155.92-160.30 s`; GPU all CUDA `5.16 s` | exact-topK `th=2048`, `75.64-75.88 s`, recall `0.341/0.400` | split `th=128`, `5.11-5.28 s`, recall `0.344/0.403` | 旧 `30.24x` 是未修正 CPU 口径；当前公平口径约 `14.4x` |
| special inter | GPU old `8` warps/query: kernel `~5.34 s`, total `~6.1 s` | CPU inter `~10-11 s` | GPU `2` warps/query: kernel `~1.87 s`, total `~2.7 s` | GPU inter 从不明显提升变成约 `4x` vs CPU inter；H2D/D2H 不是瓶颈 |
| WMMA/shared tile | 未接入 | 不适用 | TF32 WMMA tile kernel `33.53 s` | 负结果：tiles `3.61M`，source/segment 太碎，shared-memory tile 直接套用更慢 |

### 0.4 当前不能写成主结论的内容

| 项目 | 原因 |
|---|---|
| special block 已带来默认 full-quality 端到端 Index speedup | special cap/iter/two-tier 仍含近似语义，且同环境 no-special baseline 仍需严格对齐 |
| 用 topK overlap 证明 PG/group graph 或 cross-edge 质量 | 最终质量只能由 filtered-search recall / latency 验证 |
| `UNG_SKIP_LNG_TEXT_SETS=1` 作为通用默认格式 | GPU cover-frontier provider 当前仍依赖 `_lng_descendants` |
| WMMA/shared-memory special inter 直接替代 cuda-core | 当前 tile 形态过碎，实测显著变慢 |

## 1. Cross Group Edge 加速

### 1.1 方法

cross group edge 的当前主方法是 target-centric grouped fused topK。它把大量 group-to-group topK workload 合并成 descriptor batch，减少逐组 launch、重复 packing、中间相似度矩阵物化和 CPU-compatible graph 写回成本。历史上 `UNG_UNIVERSAL_GPU=1` universal flat double-buffer 是重要工程路线；当前 x200 best full-quality 结果则采用普通 GPU batched route 加 SearchQueue lazy reserve，因为该组合在完整质量口径下更稳。

2026-07-02 更新：当前 best full-quality 仍使用 `UNG_CROSS_EDGE_IMPL=1` 的 GPU batched route，但在 SearchQueue 输出边界上加入 lazy reserve。这个优化不改变 cross-edge topK 语义，只把原来为全部 `num_points` 预先 `reserve(_num_cross_edges)` 的 eager 分配改成“实际写回某个 query id 时才 reserve”。该改动把诊断中的 `output_storage(ms)` 从约 `1909.9 ms` 降到 `15.8 ms`，使 x200 best cross 从前一版 `4186.91 ms` 降到 `2842.36 ms`。

当前应区分三类实现：

| 方法 | 定位 | 说明 |
|---|---|---|
| grouped fused topK | main | 按 group descriptor 批处理，kernel 内完成距离计算与 topK，避免 SGEMM 后再单独 topK |
| universal flat double-buffer | main engineering route | 在 target-centric 语义下做 double-buffer execution、GPU global merge 和 flat-id output，是当前推荐工程路线 |
| X-side streaming | boundary | 解决 resident all-X 超显存时的可完成性，当前不作为性能主结果 |

### 1.2 Baseline

cross-edge 必须同时报告强 baseline，不能只和旧 CPU Vamana cross 比。当前至少保留以下对照：

| Baseline | 作用 | 备注 |
|---|---|---|
| CPU exact 128T | 强 CPU 精确扫描 baseline | 用于回应“只是打弱 CPU baseline”的质疑 |
| CPU hybrid scan/Vamana 128T | 当前最快 CPU cross baseline | 小 pair 走 exact scan，大 pair 走 Vamana；x200 上 threshold=1e6 最快 |
| cuVS per-group | 库函数 GPU baseline | 逐组调用 cuVS，体现 per-group 调度和 packing 开销 |
| SGEMM+topK | 常规密集线性代数 baseline | 用 GEMM 生成相似度，再 separate topK |
| CPU Vamana cross | 历史/弱 baseline | 可以作为历史加速参考，但不能作为唯一 baseline |

### 1.3 数据集/构造形态

| Workload | 证据等级 | 构造形态 | 用途 |
|---|---|---|---|
| Amazon 1%x100 | strong-baseline fairness | 主要比较 cross-edge stage，包含 CPU exact 128T、cuVS per-group、SGEMM+topK、CPU Vamana cross | 证明 fused topK 相对强 baseline 的 stage-level 加速 |
| Amazon 1%x200 | full-quality | GPU batched / universal / lazy-reserve variants + 当前完整 recall 口径 | 证明 cross-edge route 可以接入 full-quality 构建，并比较输出边界开销 |
| Amazon 1%x200 | CPU baseline sweep | CPU exact scan、CPU Vamana、CPU hybrid scan/Vamana threshold sweep | 确认最快 CPU cross baseline，并避免把 GPU cross 误当 CPU baseline |
| Amazon 10%x40 | diagnostic skip-additional | direct-qid cross 诊断，跳过 additional_edges | 解释 many-group stress 下 cross 算子潜力，不能作为 full-quality 主表 |
| Amazon 100%x40 | boundary | resident all-X 约需超过 A6000 48GB，X-side streaming 可完成 strict skip-additional build | 说明显存边界和可完成性 |

### 1.4 加速比

| Workload | 方法结果 | Baseline | 加速比 | 可写结论 |
|---|---:|---:|---:|---|
| Amazon 1%x100 fairness | final fused cross `848.9 ms` | CPU exact 128T `2389.8 ms` | `2.82x` | fused topK 优于强 CPU exact baseline |
| Amazon 1%x100 fairness | final fused cross `848.9 ms` | cuVS per-group `3730.5 ms` | `4.39x` | 优势来自跨 group batching 和减少逐组库调用开销 |
| Amazon 1%x100 fairness | final fused cross `848.9 ms` | SGEMM+topK `1395.9 ms` | `1.64x` | fused topK 避免 materialized GEMM + separate topK 的额外成本 |
| Amazon 1%x100 fairness | final fused cross `848.9 ms` | CPU Vamana cross `18735.8 ms` | `22.07x` | 只能作为历史弱 baseline 参考，不应单独用于主 claim |
| Amazon 1%x200 full-quality | universal cross `2494.21 ms` | old partial DB route `2657.23 ms` | `1.07x` | universal flat all-DB 是主工程收敛方向；加速不应夸大 |
| Amazon 1%x200 full-quality | GPU best lazy-reserve cross `2842.36 ms` | CPU Vamana cross `43365.3 ms` | `15.26x` | 真正 CPU Vamana cross baseline，`UNG_CROSS_EDGE_IMPL=0` |
| Amazon 1%x200 full-quality | GPU best lazy-reserve cross `2842.36 ms` | CPU exact-scan cross `40089.8 ms` | `14.10x` | 强 CPU brute-force baseline，本轮 sweep |
| Amazon 1%x200 full-quality | GPU best lazy-reserve cross `2842.36 ms` | CPU hybrid threshold=1e6 `37338.0 ms` | `13.14x` | 当前最快 CPU cross baseline |
| Amazon 10%x40 diagnostic | GPU direct-qid cross `8475.4 ms` | CPU exact cross `54879.3 ms` | `6.47x` | 说明 many-group cross 算子有明显 GPU 潜力，但该行是 skip-additional diagnostic |

CPU baseline sweep artifact：

```text
/home/graphdb/fv_runs/cpu_cross_baseline_sweep_20260702_184801
```

current best lazy-reserve cross artifact：

```text
/home/graphdb/fv_runs/cross_lazy_reserve_best_20260702_181842
```

注意：`/home/graphdb/fv_runs/best_full_quality_e2e_20260702_163334` 中的 `cpu_vamana_group,cross_ms=2950.86` 不是 CPU cross baseline；该 case 只是 group graph 使用 CPU Vamana，cross-edge 仍然使用 GPU batched route。因此不能用 `2842.36 / 2950.86` 解释 GPU-vs-CPU。

### 1.5 Claim 边界

可以写：cross-edge 是当前证据最清楚的 stage-level 加速点；在 Amazon 1%x100 fairness 中，fused topK 同时优于 CPU exact 128T、cuVS per-group 和 SGEMM+topK。在 Amazon 1%x200 full-quality 口径下，SearchQueue lazy reserve 后的 GPU best cross 相对当前最快 CPU cross baseline 约 `13.14x`。

不能写：不能只报 `22.07x`；不能把 skip-additional 结果写成 full-quality；不能把 X-side streaming 写成主要性能加速；不能把 `cpu_vamana_group` 中仍然启用 GPU cross 的 `cross_ms` 误写成 CPU cross baseline；不能把 x200 的 `1.07x` universal route 写成端到端大幅加速。

## 2. GPU 加速提取 ELS/入口组

### 2.1 方法

ELS 这里指 search/query 阶段的 entry label-set / entry groups。当前方法是 `gpu_cover_frontier` correct-cover provider：把 group labels、LNG descendants 和 coverage 关系组织为 CUDA 侧 bitset/frontier 表，按 query labels 输出 coverage-correct entry group ids。

正确性标准是 coverage correctness：

```text
C = {g | query_labels subset labels(g)}
输出 entry groups 必须覆盖 C 中所有真实候选 group。
```

质量标准不是 topK recall，而是在 coverage-correct 的前提下输出组数尽量少。入口组越少，越说明 provider 尽可能从 label/LNG 的上层开始，后续 `entry group ids -> entry points -> iterate_to_fixed_point()` 的负担也越小。CPU exact minimal 是组数最少的质量 baseline，但它很慢；GPU correct-cover 允许比 exact minimal 更多的入口组，以换取更低 provider latency。

`gpu_cover_frontier` 的两个主要超参是：

| 超参 | 含义 | 性能/质量影响 |
|---|---|---|
| `frontier_delta` | 只把 label-size 在 `[|query|, |query| + delta]` 范围内的候选作为 frontier | 增大后通常能覆盖更多后代、减少最终输出组数，但 coverage OR 更贵 |
| `frontier_cover_cap` | 每个 query 最多拿多少个 frontier group 做 descendant coverage OR | 增大后通常输出组数减少，但 kernel 时间增加；截断不会破坏 coverage correctness，只会让 `Uncovered` 变多 |

当前 production 状态：

| 项 | 状态 |
|---|---|
| production route | 已接入 `SearchEntryProvider` 路径，可通过 `gpu_cover_frontier` provider 使用 |
| CUDA provider | 已有常驻表构建和 per-query CUDA 调用 |
| 当前限制 | per-query 串行 CUDA provider，缺 batch/end-to-end A/B |
| production 超参 | 当前代码默认 `frontier_delta=2`、`frontier_cover_cap=8192`，尚未暴露为 search CLI/env |
| 必须补的指标 | total latency、entry ms、NumEntries、core search ms、recall |

### 2.2 Baseline

| Baseline | 语义 | 作用 |
|---|---|---|
| CPU scan 128T | 扫描候选 group，输出 raw candidate groups | 弱/诊断 baseline；不能作为公平 GPU 加速主对照 |
| CPU cover-frontier 128T | 和 GPU cover-frontier 同算法语义的 OpenMP parallel implementation | 公平性能 baseline；必须和 GPU 使用相同 `delta/cap`、相同输出 group id materialization 后再比较 |
| CPU exact minimal | 求更小的 exact minimal entry groups | 质量/冗余对照，不一定更快 |
| production CPU provider | 端到端 search route 中的当前 CPU 路径 | 未来 batch/end-to-end A/B 必须对照 |

### 2.3 数据集/构造形态

| Workload | 证据等级 | 构造形态 | 用途 |
|---|---|---|---|
| Amazon 100%x40，`nq=10240` | microbenchmark | many-group / high coverage pressure 的 query entry provider benchmark | 测 entry provider 局部耗时和输出组数 |
| Amazon 10%x40 full-quality query | estimate / pending A/B | paper draft 中基于 CPU entry 占比的上界估算 | 说明潜在端到端收益上限，不能当实测 |
| Amazon 1%x200 / 1%x400 full-quality query | estimate / pending A/B | CPU entry 占比较低的构造 | 说明在 entry 不是瓶颈时端到端收益有限 |

### 2.4 加速比

Amazon 100%x40、`nq=10240` 的修正后 pair sweep 如下。这里的“质量”只指 entry provider 输出组数，组数越少越接近 CPU exact minimal；最终 search 质量仍必须看端到端 recall。

关键公平性说明：`19.85x` 只说明 GPU cover-frontier 优于 raw CPU scan 诊断 baseline，不能作为公平 GPU-vs-CPU 主结论。修正后的 benchmark 在同一进程中只构建一次 label/descendant bitset，并对每个 `delta/cap` 成对运行 CPU cover-frontier 和 GPU cover-frontier，二者都 materialize group id 输出。artifact：

```text
/home/graphdb/FilterVectorResultsRefactor/organized_benchmarks/query_entry_group_pair_sweep_20260701_nocheck
```

基线：

| 方法 | total ms | avg output groups | quality vs exact |
|---|---:|---:|---:|
| CPU scan 128T | `1166.43` | `23332.20` | `37.862x` |
| CPU exact minimal | `4628.71` | `616.24` | `1.000x` |

同配置 CPU/GPU 成对结果的代表点：

| 配置 | CPU ms | CPU groups | GPU ms | GPU groups | GPU vs CPU | 质量关系 |
|---|---:|---:|---:|---:|---:|---|
| `d1 cap64` | `16.06` | `2172.57` | `47.37` | `2172.57` | `0.34x` | same |
| `d2 cap1024` | `296.67` | `1336.57` | `127.32` | `1336.57` | `2.33x` | same |
| `d2 cap8192` | `457.66` | `1435.14` | `218.83` | `1435.14` | `2.09x` | same |
| `d3 cap8192` | `1032.90` | `1279.98` | `415.13` | `1279.98` | `2.49x` | same |

注意：`d2/d3` 的小 cap 配置下，CPU/GPU 虽然都 coverage-correct，但 frontier compact 顺序不同，实际输出组数可能差很多。例如 `d3 cap512` 的 CPU 输出 `1182.24` groups，而 GPU 输出 `9247.91` groups；这种配置不能按同名超参直接说同质量，必须按实际 `avg_output_groups` 比较。

按质量阈值选最优方法：

| max avg output groups | CPU best | CPU ms/groups | GPU best | GPU ms/groups | GPU vs CPU | 更好 |
|---:|---|---:|---|---:|---:|---|
| `1200` | `cpu d3 cap512` | `468.08 / 1182.24` | 无 | - | - | CPU only |
| `1300` | `cpu d3 cap512` | `468.08 / 1182.24` | `gpu d3 cap8192` | `415.13 / 1279.98` | `1.13x` | GPU |
| `1400` | `cpu d2 cap256` | `195.68 / 1384.20` | `gpu d2 cap1024` | `127.32 / 1336.57` | `1.54x` | GPU |
| `1500` | `cpu d2 cap256` | `195.68 / 1384.20` | `gpu d2 cap1024` | `127.32 / 1336.57` | `1.54x` | GPU |
| `1700` | `cpu d2 cap128` | `134.28 / 1674.73` | `gpu d2 cap1024` | `127.32 / 1336.57` | `1.06x` | GPU |
| `2200` | `cpu d1 cap64` | `16.06 / 2172.57` | `gpu d1 cap64` | `47.37 / 2172.57` | `0.34x` | CPU |

读表方式：

- 如果目标是弱 baseline 对比，`delta=1 cap=8192` 相对 CPU scan 128T 为 `19.85x`。但这不是公平主结论，因为 CPU scan 输出 raw candidate groups，和 cover-frontier 的同语义输出不同。
- 如果允许约 `2170` 个入口组，CPU `d1 cap64` 是最快点：`16.06 ms`，同质量下 GPU `47.37 ms`，CPU 更好。
- 如果要求约 `1300~1500` 个入口组，GPU 是更好的 Pareto 点：例如 `<=1400` groups 时 GPU `d2 cap1024` 为 `127.32 ms / 1336.57 groups`，CPU 最优为 `195.68 ms / 1384.20 groups`，GPU 约 `1.54x` 更快且组数更少。
- 如果要求 `<1200` 个入口组，当前 grid 里只有 CPU 达到；GPU 最少到 `1279.98` groups。
- 如果目标是入口组尽量少，CPU exact minimal 仍是最强质量 baseline，但速度只有 CPU scan 的 `0.26x`。
- GPU 仍有约 `25 ms` 完整 bitset D2H 成本；若改成 GPU-side compact `(offsets, group_ids)`，中高质量区间的 GPU 优势应该进一步扩大。
- production `gpu_cover_frontier` 当前默认 `delta=2 cap=8192`，并且是 per-query 串行 CUDA provider；上表来自批处理 microbenchmark，不等于 production 端到端 search speedup。

### 2.4.1 GPU compact 输出优化与瓶颈归因

为了验证 D2H 是否是主要瓶颈，我们在 benchmark 中新增 `gpu_cover_frontier_compact` 路径：GPU 仍先计算 coverage-correct `out_bits`，但随后在 GPU 上 count/compact 成 group ids，只把 compact ids 拷回 CPU，而不是 D2H 完整 `nq * words` bitset。

当前 Amazon 100%x40 的输出 bitset 尺寸为：

```text
nq=10240, groups=482388, words_per_query=7538
full output bitset = 10240 * 7538 * 8 ~= 589 MiB
```

硬件上，A6000 显存带宽约 `768 GB/s`，PCIe 4.0 x16 单向理论约 `31.5 GB/s`，双路 Xeon 8360Y 内存理论峰值约 `409.6 GB/s`。实测 full-bitset GPU 的 D2H 大约 `25 ms`，折合约 `23 GiB/s`，接近 PCIe 实际可达范围。因此完整 bitset D2H 确实是瓶颈之一。

但是 compact 输出不是唯一瓶颈。ELS 的核心 coverage OR 是 memory-bound bitset workload，不是 dense compute。按 descendant OR 的最低读流量估计：

| 配置 | avg frontier groups | descendant 读流量下界 | GPU kernel | 等效读带宽下界 |
|---|---:|---:|---:|---:|
| `gpu d1 cap64` | `6.07` | `3.8 GB` | `14.19 ms` | `264 GB/s` |
| `gpu d2 cap1024` | `93.87` | `58.0 GB` | `94.09 ms` | `616 GB/s` |
| `gpu d3 cap8192` | `408.95` | `252.5 GB` | `382.3 ms` | `661 GB/s` |

这说明中高质量配置下，GPU kernel 已经接近 A6000 显存带宽上限。优化 D2H 能改善总时间，但不会带来数量级提升；真正的大头逐渐变成 descendant bitset OR 本身。

compact 输出关键点实测：

```text
artifact: /home/graphdb/FilterVectorResultsRefactor/organized_benchmarks/query_entry_group_compact_keypoints_20260701_v2
```

| 配置 | full-bitset GPU | fused compact GPU | D2H 变化 | 总时间变化 | 质量 |
|---|---:|---:|---:|---:|---:|
| `d1 cap1024` | `57.34 ms` | `29.42 ms` | `25.03 -> 3.72 ms` | `1.95x` | `2184 groups` |
| `d2 cap1024` | `127.20 ms` | `97.80 ms` | `25.04 -> ~2-4 ms` | `1.30x` | `1337 groups` |
| `d2 cap8192` | `216.90 ms` | `187.40 ms` | `25.08 -> ~2-4 ms` | `1.16x` | `1435 groups` |
| `d3 cap8192` | `415.00 ms` | `385.50 ms` | `25.06 -> ~2-4 ms` | `1.08x` | `1280 groups` |

解释：

- fused compact 输出显著降低 D2H：约 `25 ms` 降到 `2~4 ms`。
- 在宽松质量 `d1` 区间，fused compact 也明显改善 GPU，但仍未超过 CPU：`d1 cap64` 下 CPU `16.08 ms / 2173 groups`，GPU fused compact `19.65 ms / 2173 groups`，GPU 为 `0.82x`。也就是说，宽松质量下 CPU 仍是最佳。
- 在 `d1 cap1024` 这类低 frontier 配置，fused compact 把 total 从 `57.34 ms` 降到 `29.42 ms`，接近 `1.95x`。
- 在 `d2 cap1024` 这类中高质量配置，fused compact 后 GPU 为 `97.80 ms / 1337 groups`，相对同质量 CPU `290.70 ms / 1337 groups` 约 `2.97x`，达到 `2x~3x` 目标。
- 在 `d3 cap8192` 这类重 OR 配置，kernel 已经 memory-bound，D2H 降低后 full GPU 到 fused compact 只提升约 `1.08x`；但相对同质量 CPU `1053 ms / 1280 groups` 仍约 `2.73x`。

并行度 scaling 进一步说明：这个 benchmark 测的是批量查询吞吐，不是单 query 串行路径。固定中高质量配置 `d2 cap1024`，CPU 使用 OpenMP 128 线程，GPU 使用 batch kernel + fused compact 输出。随着 `nq` 增大，GPU 优势更稳定：

```text
artifact: /home/graphdb/FilterVectorResultsRefactor/organized_benchmarks/query_entry_group_parallel_scaling_20260701
```

| nq | CPU cover-frontier | GPU fused compact | output groups | GPU vs CPU |
|---:|---:|---:|---:|---:|
| `1` | `0.358 ms` | `0.138 ms` | `5350` | `2.60x` |
| `16` | `2.30 ms` | `0.185 ms` | `1132` | `12.41x` |
| `64` | `10.84 ms` | `0.517 ms` | `1041` | `20.96x` |
| `256` | `19.62 ms` | `2.578 ms` | `1295` | `7.61x` |
| `1024` | `40.99 ms` | `9.715 ms` | `1330` | `4.22x` |
| `4096` | `119.6 ms` | `38.87 ms` | `1330` | `3.08x` |
| `10240` | `294.4 ms` | `97.82 ms` | `1337` | `3.01x` |

这里的 `nq=1` 行不是 production 单 query 路径，而是同一个 batch benchmark 在极小 query 数下的下界；真正论文应优先报告 `nq>=1024` 的批量吞吐。该表支持你的判断：并行查询越充分，GPU 越容易体现优势；在目标质量区间，`nq=4096/10240` 时 GPU fused compact 稳定达到约 `3x`。

因此，当前瓶颈排序应写成：

1. 中高质量配置：descendant bitset OR 的显存带宽是主瓶颈。
2. 所有 full-bitset GPU 配置：完整 output bitset D2H 是固定瓶颈；fused compact 已把这项显著降低。
3. compact 输出后：中高质量 `1300~1500` groups 区间达到 `2x~3x` CPU 加速；宽松质量约 `2170` groups 区间 CPU 仍略优；重 OR 场景的主瓶颈仍是 descendant bitset OR 的显存带宽。

端到端上界估算仍需保守：

| Workload | GPU entry 的潜在端到端收益 | CPU entry 占比 | 估算上限 | 备注 |
|---|---:|---:|---:|---|
| Amazon 10%x40 query | 替换 entry provider | `48~51%` | `1.9~2.0x` | 不是实测端到端 speedup |
| Amazon 1%x200 / 1%x400 query | 替换 entry provider | `6~11%` | `1.07~1.12x` | entry 占比低时 Amdahl 上限较小 |

### 2.5 Claim 边界

可以写：GPU correct-cover provider 在 Amazon 100%x40 的 entry provider microbenchmark 中相对 CPU scan 128T 达到 `19.85x`，但这是相对弱/诊断 scan baseline。修正后的 pair sweep 表明，GPU 不是全区间占优：宽松质量约 `2170` groups 时 CPU 更快。进一步的 fused compact 输出优化后，中高质量区间达到目标：`d2 cap1024` 为 `97.80 ms / 1337 groups`，相对同质量 CPU `290.70 ms / 1337 groups` 约 `2.97x`；`d3 cap8192` 为 `385.50 ms / 1280 groups`，相对同质量 CPU `1053 ms / 1280 groups` 约 `2.73x`。最严格 `<1200` groups 当前只有 CPU 达到。

不能写：不能把 `19.85x` 写成公平 GPU-vs-CPU parallel 加速；不能把 `19.85x` 写成端到端 query 加速；不能忽略 GPU provider 输出入口组比 CPU exact minimal 多；不能在缺少 batch/end-to-end A/B 前声称 query 主流程已经获得稳定加速。

当前距离端到端最大加速比的主要障碍是 Amdahl 边界和接口粒度：如果 entry provider 只占 total query 的 `6~11%`，即使 provider 自身很快，端到端也只能得到约 `1.07~1.12x`；若在 10%x40 这类 entry 占比 `48~51%` 的场景，潜在上限约 `1.9~2.0x`，但必须用 batch provider 降低 per-query launch/同步开销，并证明 NumEntries 增加不会把 graph expansion 成本抵消掉。

## 3. 加速构建 PG/Group Graph

### 3.1 方法

PG/group graph 当前不是单一 GPU backend，而是 workload-aware router。推荐语义是按 group size 和质量风险分流：

| Group 类型 | 推荐路径 | 原因 |
|---|---|---|
| small group | CPU Vamana 或 bounded-complete fallback | GPU packing/H2D/fill 容易吞掉计算收益 |
| medium group | packed exact-anchor | exact-anchor 可以获得较稳定质量，direct-H2D/device lookup 降低 pack 成本 |
| large group | FastGrnndCuda + reverse-tail / repair | 用近似候选和质量增强控制大组构图成本与 recall |

这条主线的核心不是“GPU 替代 CPU Vamana”，而是把不同 group size 的代价结构显式路由，避免 small-group 负收益，并在 medium/large group 上用 GPU 算子降低构图时间。

PG/group graph 的质量标准必须是建图后的 filtered search 结果，而不是局部 topK overlap。原因是导航图需要可达性、连通性、多样性和跨区域跳转能力；简单保留距离最近的 topK 边可能导致局部团簇过强、尾部/反向多样性不足，最终查询 recall 反而下降。当前 x400 light512 反例正说明这一点：保留 head nearest neighbors 并不充分，reverse-tail+repair 才更接近 CPU recall。

### 3.2 Baseline

| Baseline | 作用 | 备注 |
|---|---|---|
| CPU Vamana group graph | full-quality baseline | 必须作为质量和速度对照 |
| old exact-anchor host-pack path | exact-anchor 内部 A/B baseline | 用于证明 direct-H2D / GPU lookup / fill16 消除 pack/fill 开销 |
| heavy prune / all-small GPU exact | negative baseline | 解释为什么不能盲目 GPU 化 |
| x100 CPU Vamana group + same cross/additional | small-scale full-quality baseline | 用于质量闭环，不是强 group 加速场景 |

### 3.3 数据集/构造形态

| Workload | 证据等级 | 构造形态 | 用途 |
|---|---|---|---|
| Amazon 1%x200 | full-quality | packed exact-anchor + repeat=3 search | 证明 medium group route 的正结果 |
| Amazon 10%x40 | full-quality stress | FastGrnndCuda + complete fallback + GPU direct-qid cross + CPU additional_edges | 证明 many-group stress 下 adaptive route 有 index-level 收益 |
| Amazon 1%x100 | full-quality sanity | CPU/FastGrnndCuda 同 cross/additional 口径 | 证明质量不降，但 group speedup 很小 |
| Amazon 1%x400 | pressure / Pareto | packed exact x400、reverse-tail+repair | 说明大规模压力下质量增强 route 的边界 |
| heavy prune / all-small exact | negative | prune 或小组全 GPU 化 | 解释负结果和 router 必要性 |

### 3.4 加速比

| Workload | 方法结果 | Baseline | 加速比 | 质量口径 |
|---|---:|---:|---:|---|
| Amazon 1%x200 full-quality latest best | best_full_quality group `1837.34 ms` | CPU Vamana group `8119.73 ms` | `4.42x` | L1000/L5000 `0.871/0.911` 对齐；L20 `0.814467` vs CPU `0.816` |
| Amazon 1%x200 full-quality | packed exact-anchor group `3827.28 ms` | CPU group `8511.72 ms` | `2.22x` | repeat=3 L1000/L5000 `0.869/0.908`，对齐 CPU |
| x200 exact-anchor internal A/B | direct-H2D + GPU lookup + fill16 group `3204.88 ms` | old exact path group `8999.26 ms` | `2.81x` | L1000 recall `0.867`；主要消除 pack/fill 开销 |
| Amazon 10%x40 full-quality | FastGrnndCuda route group `5147.2 ms` | CPU Vamana group `18218.5 ms` | `3.54x` | L100/L500/L1000 `0.8668/0.8980/0.9103`，不低于 CPU `0.8647/0.8980/0.9080` |
| Amazon 10%x40 full-quality | FastGrnndCuda route Index `23497.4 ms` | CPU Vamana Index `36053.3 ms` | `1.53x` | 同上，端到端 index 级正结果 |
| Amazon 1%x100 full-quality | FastGrnndCuda group `3216.21 ms` | CPU Vamana group `3303.26 ms` | `1.03x` | L100/L500/L1000 `0.828/0.871/0.896` vs CPU `0.825/0.868/0.890667` |
| Amazon 1%x400 packed exact | group `11588.4 ms` | CPU `17537.3 ms` | `1.51x` | repeat=3 L1000/L5000 `0.945/0.967` |
| Amazon 1%x400 reverse-tail+repair | Index `33044.9 ms` | same-script CPU `36257.3 ms` | `1.10x` | L1000/L5000 `0.945567/0.9659`，接近 CPU `0.946/0.965` |

最新同批次 CPU/Tagore/our 接入对比：

```text
/home/graphdb/fv_runs/graph_builder_cpu_tagore_our_20260702_152249
```

| 方法 | group graph ms | 相对 CPU Vamana | 质量说明 |
|---|---:|---:|---|
| CPU Vamana | `8223.67` | `1.00x` | L5000 `0.911` |
| TagoreCuda, `nx>=1024` | `8298.99` | `0.99x` | 只覆盖 `23` 个大组、`41400` 点；大部分仍是 CPU fallback |
| TagoreCuda, `nx>=128` | `32327.7` | `0.25x` | 覆盖 `5666` 组、`1.2048M` 点；原始 Tagore prune `19599.8 ms`，不适合该 group workload |
| our exact-anchor/reverse | `1895.75` | `4.34x` | L5000 `0.911`；L20 `0.8142~0.8144`，略低 CPU |

当前推荐作为最稳 group-graph speedup 的数字来自完整 CPU-vs-best A/B：

```text
/home/graphdb/fv_runs/best_full_quality_e2e_20260702_163334
CPU group = 8119.73 ms
best group = 1837.34 ms
speedup = 4.42x
```

注意：`/home/graphdb/fv_runs/cross_lazy_reserve_best_20260702_181842` 中 best group `4844.47 ms` 是单次复测波动样本，用来验证 cross lazy reserve，不应替代 group graph 主结论。


### 3.4.1 x200 direct-H2D Lsearch sweep

为避免只报告超大 `Lsearch`，我们复用 x200 direct-H2D exact-anchor index，补跑了从小 L 到大 L 的 full-quality search sweep。artifact：

```text
/home/graphdb/fv_runs/x200_direct_h2d_lsearch_sweep_20260702/x200_lsearch_sweep_summary.md
```

| Lsearch | CPU recall | GPU direct-H2D recall | Delta | CPU batch ms | GPU batch ms |
|---:|---:|---:|---:|---:|---:|
| `20` | `0.819000` | `0.808867` | `-0.010133` | `499.140` | `468.917` |
| `50` | `0.819000` | `0.816600` | `-0.002400` | `427.601` | `429.117` |
| `100` | `0.823000` | `0.822000` | `-0.001000` | `432.525` | `428.761` |
| `200` | `0.825000` | `0.823333` | `-0.001667` | `430.689` | `431.430` |
| `500` | `0.849000` | `0.845000` | `-0.004000` | `440.811` | `438.016` |
| `1000` | `0.869000` | `0.867000` | `-0.002000` | `443.942` | `446.626` |
| `2000` | `0.882000` | `0.882000` | `+0.000000` | `452.236` | `454.777` |
| `5000` | `0.908000` | `0.908000` | `+0.000000` | `478.259` | `483.520` |

结论：`Lsearch=1000/5000` 不是唯一质量证据。低 L 下确实有小幅质量代价，尤其 L20 下降约 `0.010`、L500 下降 `0.004`；L50-L200 下降约 `0.001~0.0024`，L2000/L5000 对齐。因此 x200 direct-H2D 应写成“2.66x group-build speedup with a small low-L recall tradeoff”，而不是“所有 L 下无损”。


### 3.4.2 x200 exact-anchor low-L quality tuning

我们进一步尝试在不增加 exact 距离计算的前提下，通过 exact-anchor 输出策略提升低 `Lsearch` recall。调参只改变最终 32 条组内边中 nearest head、anchor diversity 和 tail-neighbor 的比例，理论上不应显著增加 GPU kernel 成本。

```text
artifact: /home/graphdb/fv_runs/x200_quality_tune_exact_anchor_20260702/quality_tuning_summary.md
```

| Lsearch | CPU | default direct-H2D | head16_anchor1 | head8_anchor1 | head16_anchor4 |
|---:|---:|---:|---:|---:|---:|
| `20` | `0.819000` | `0.808867 (-0.010133)` | `0.811600 (-0.007400)` | `0.808733 (-0.010267)` | `0.810400 (-0.008600)` |
| `50` | `0.819000` | `0.816600 (-0.002400)` | `0.817133 (-0.001867)` | `0.817200 (-0.001800)` | `0.815400 (-0.003600)` |
| `100` | `0.823000` | `0.822000 (-0.001000)` | `0.821000 (-0.002000)` | `0.821667 (-0.001333)` | `0.820600 (-0.002400)` |
| `200` | `0.825000` | `0.823333 (-0.001667)` | `0.823000 (-0.002000)` | `0.823000 (-0.002000)` | `0.824000 (-0.001000)` |
| `500` | `0.849000` | `0.845000 (-0.004000)` | `0.845000 (-0.004000)` | `0.845000 (-0.004000)` | `0.850000 (+0.001000)` |

Build times:

| case | group ms | index ms | 结论 |
|---|---:|---:|---|
| `head16_anchor1` | `1586.98` | `13740.2` | L20 小幅改善，但不足以缩小主要 gap |
| `head24_anchor0` | `2678.88` | `15227.0` | 多数 L recall 变差，不保留 |
| `head8_anchor1` | `1604.28` | `10505.8` | 基本中性，未明显改善 |
| `head12_anchor4` | `1710.05` | `10534.4` | L20 `0.8118`，比 default `+0.0029`；L500 `0.850`，略高 CPU；L50 仍低 `0.0032` |
| `head12_anchor4_bidir` | `2238.07` | `15277.9` | L20 `0.812067`，比 default `+0.0032`；L500 `0.850`，略高 CPU；相对 default direct-H2D group 仍更快 |
| `head12_anchor4_bidir_rev4` | `1822.89` | `16637.8` | 最强 L20 点：L20 `0.814533`，把 L20 gap 从 `-0.0101` 缩到 `-0.0045`；L500 `0.850` |
| `head16_anchor4_bidir_rev4` | `2063.79` | `12237.7` | 当前推荐性价比点：L20 `0.813933`，L50 `0.8164`，L500 `0.850`；group 比 default direct-H2D 快，index/cross 波动较小 |
| `head16_anchor4` | `1613.47` | `13970.1` | L500 提升到 `0.850`，略高 CPU；L20/L50 仍有 gap |

结构诊断：

```text
artifact: /home/graphdb/fv_runs/x200_quality_tune_exact_anchor_20260702/graph_diag/summary_table.md
```

CPU、default direct-H2D 和 `head16_anchor4` 的出度、low-degree、WCC 都几乎一致：intra out-degree 约 `32`，zero/low-degree 约 `0`，所有 group 最大 WCC 为 `1`。因此低 L gap 不是缺边或断连，而是边选择/导航质量差异。

结论：简单 head/tail/anchor slot 调参属于低成本但收益有限的质量增强。group-aware reciprocal/reverse-tail augmentation 是有效方向。`head12_anchor4_bidir_rev4` 是最强 L20 点：L20 从 default `0.808867` 提到 `0.814533`，把 L20 gap 从 `-0.0101` 缩到 `-0.0045`，缩小超过一半；L500 从 `0.845` 提到 `0.850`，略高 CPU `0.849`。`head16_anchor4_bidir_rev4` 是当前推荐性价比点：group `2063.79 ms`，仍快于 default direct-H2D `3204.88 ms` 和 CPU `8511.72 ms`；L20 `0.813933`、L50 `0.8164`、L500 `0.850`。二者仍未完全对齐 L20/L50，但已经显著缩小低 L gap，且没有损失 GPU 建图性能。

### 3.5 Claim 边界

可以写：PG/group graph 需要 workload-aware router；在 Amazon 10%x40 full-quality stress test 中，FastGrnndCuda route 带来 group `3.54x`、Index `1.53x` 的加速且 recall 不降；在 Amazon 1%x200 最新 best full-quality A/B 中 group graph 相对 CPU Vamana 为 `4.42x`，高 L recall 对齐，低 L 有小幅差距。

不能写：不能写 GPU 无条件替代 CPU Vamana；不能把 x100 的质量闭环写成强加速；不能把 heavy prune 或 all-small exact 的负结果隐藏；不能把 direct-H2D/fill 优化解释成算法质量贡献，它主要是输出/packing 边界优化。

## 4. Full-Quality Additional Edges 与 CPU-Compatible Graph Materialization

### 4.1 这一步是什么

full-quality 构建指当前完整 UNG 构建语义，而不是只跑某个 GPU kernel。对 cross-edge 阶段来说，完整流程是：

```text
1. 组内 PG/group graph 已经构好。
2. cross-edge backend 为 label graph 中的 parent/child 或相关 group 关系生成 topK 跨组边。
3. additional_edges 检查每个 group 是否已经通过 cross-edge 覆盖到必要的 out-neighbor groups。
4. 如果某些 out-neighbor group 没有被连接，则用 CPU Vamana 或 CPU exact scan 生成补边。
5. add_offset_for_uni_nav_graph() 把组内 local id 转成全局 id。
6. merge_cross_edges_to_graph() 和 merge_additional_edges_to_graph() 把 GPU/CPU 结果写回 `_graph->neighbors`。
7. 保存 index 后，用 filtered-search query + brute-force GT 跑端到端 recall / latency。
```

`additional_edges` 的作用是补齐 label/LNG 语义要求的跨组连通性。它不是可随意省略的附属项；跳过后 cross-edge stage 可能更快，但搜索 recall 会明显下降。历史 Amazon PF 已显示 skip additional 后 L5000 recall 约 `0.616`，full-quality 才回到约 `0.91~0.97`。

CPU-compatible graph materialization 指：即使 GPU 已经生成了边，当前 search 和 additional_edges 仍主要消费 host 侧 `Graph::neighbors` / `NeighborList` / SearchQueue / Vamana-compatible structures。因此 GPU 输出必须 D2H，并被回填到每个点的 CPU 邻接对象。这一步会产生 allocator、per-node object、merge、local/global id 转换和并行写入成本，是 full-quality 下吞掉 stage speedup 的主要 Amdahl 边界之一。

### 4.2 方法

| 方法 | 定位 | 说明 |
|---|---|---|
| CPU Vamana additional | full-quality compatibility baseline | 对缺失连接的 out-neighbor group 调 Vamana search 生成补边，是当前 full-quality 主口径 |
| CPU exact additional | diagnostic / candidate baseline | 用 exact scan 生成补边，可降低部分 additional 本体时间，但外围/总时间未必更快 |
| CPU exact materialized additional | current best candidate | 先 materialize `additional_edges`，再统一 merge，避免 per-edge lock |
| Graph reserve | output-boundary mitigation | 预留 host adjacency 容量，减少 per-node vector 扩容 |
| NeighborList64 | retained low-risk optimization | 用 64-inline small-buffer adjacency 吸收 allocator 成本，仍保持 CPU-compatible graph |
| flat-id writeback / direct-global | diagnostic / partial output change | 证明小对象输出有成本，但多为 skip-additional smoke，不能当 full-quality 主结果 |
| direct append additional | negative | 直接在并行 additional 阶段 mutate `_graph->neighbors` 不稳定，不是可靠方案 |
| SearchQueue lazy reserve | current retained optimization | GPU SearchQueue writeback 只对实际写回的 query id 分配 topK 容量，避免全量点预分配 |

### 4.3 Baseline

| Baseline | 作用 | 边界 |
|---|---|---|
| skip additional_edges | stage timing / ablation | 只能隔离 cross-edge 或 output boundary；不能代表完整质量 |
| CPU Vamana additional | 当前 full-quality 质量 baseline | 语义稳，但可能比 GPU kernel 还慢 |
| old `std::vector<IdxType>*` graph output | 输出边界 baseline | 用于证明 allocator/per-node object 是实质瓶颈 |
| NeighborList48 | negative baseline | 说明 inline capacity 不能随意缩小 |

### 4.4 数据集/构造形态

| Workload | 证据等级 | 构造形态 | 用途 |
|---|---|---|---|
| Amazon 1%x100 additional/output-boundary smoke | diagnostic | 比较 CPU Vamana additional、CPU exact additional 与 GPU kernel event | 证明 full-quality additional 可超过 fused kernel 本体 |
| Amazon 10%x40 graph reserve A/B | output-boundary A/B | reserve off/on，同一类 full-quality 构建路径 | 证明 host adjacency 预留能显著降低 fill/merge，但 reserve 不是最终方案 |
| Amazon 1%x200 auto-reserve / NeighborList64 | output-boundary A/B | old vector + reserve vs NeighborList64 | 证明 small-buffer adjacency 大幅降低 reserve/fill 成本 |
| Amazon 1%x200 best additional probe | full-quality A/B | CPU Vamana additional、CPU exact materialized、CPU exact direct append | 确认 current best additional 路径 |
| Amazon 1%x200 cross lazy reserve | full-quality A/B | eager SearchQueue reserve vs lazy reserve | 确认 cross output storage 初始化是实质瓶颈 |
| Amazon 1%x200 / 1%x100 universal full-quality | full-quality smoke/boundary | 保留 CPU Vamana additional，替换 cross-edge route | 验证 cross-edge route 能接上 additional 和 search recall |

### 4.5 加速比

| Workload | 方法结果 | Baseline | 加速比/变化 | 结论 |
|---|---:|---:|---:|---|
| Amazon 1%x100 smoke | CPU Vamana additional `1269.257 ms` | GPU kernel event `655.969 ms` | additional 是 kernel 的 `1.94x` | full-quality 加速上限受 additional 限制 |
| Amazon 1%x100 smoke | CPU exact additional `207.893 ms` | CPU Vamana additional `1269.257 ms` | additional 本体 `6.11x` 更快 | 但 total cross `2568.729 ms` 仍略慢，不能当简单替代 |
| Amazon 10%x40 reserve A/B | Index `24704.5 ms` | reserve-off `39746.1 ms` | `1.61x` | reserve 证明 output boundary 是实质瓶颈 |
| Amazon 10%x40 reserve A/B | tagore fill `23.9 ms` | reserve-off `823.5 ms` | `34.45x` | fill/allocator 成本显著下降 |
| Amazon 10%x40 reserve A/B | cross merge `2.3 ms` | reserve-off `1677.1 ms` | `729x` | cross merge 的 per-node 扩容被消除 |
| Amazon 1%x200 auto reserve | tagore fill `224.8 ms` | auto-off `2497.0 ms` | `11.11x` | auto reserve 降低 fill，但仍是 CPU graph |
| Amazon 1%x200 NeighborList64 | reserve `25.4 ms` | old vector reserve `2856.2 ms` | `112x` | small-buffer 吸收大部分 reserve allocator 成本 |
| Amazon 1%x200 NeighborList64 | tagore fill `14.7 ms` | old vector fill `224.8 ms` | `15.3x` | 保留的低风险系统优化 |
| Amazon 10%x40 NeighborList64 | reserve `21.2 ms` | old reserve `6487.3 ms` | `306x` | 大幅降低预留成本 |
| Amazon 10%x40 NeighborList64 | tagore fill `7.6 ms` | old reserve fill `23.9 ms` | `3.14x` | 仍不是 GPU-native CSR/GraphView |
| Amazon 1%x200 best additional probe | CPU exact materialized additional `831.7 ms` | CPU Vamana additional `5954.2 ms` | `7.16x` | L20/L1000/L5000 recall `0.8144/0.871/0.911`，对齐 Vamana additional |
| Amazon 1%x200 best additional probe | CPU exact direct append `2682.9 ms` | CPU exact materialized `831.7 ms` | `0.31x` | direct append 因 per-edge lock 变慢，是负结果 |
| Amazon 1%x200 cross lazy reserve | output storage `15.8 ms` | eager reserve diagnostic `1909.9 ms` | `120.9x` | SearchQueue 输出初始化瓶颈被基本消除 |
| Amazon 1%x200 cross lazy reserve | cross `2842.36 ms` | previous best cross `4186.91 ms` | `1.47x` | recall L1000/L5000 `0.871/0.911` 对齐 |

additional probe artifact：

```text
/home/graphdb/fv_runs/additional_edges_opt_probe_20260702_155417
```

cross lazy reserve artifact：

```text
/home/graphdb/fv_runs/cross_lazy_reserve_best_20260702_181842
```

诊断事实：

```text
additional_work missing_group_edges=9204
additional_work query_vectors=18408
additional_work exact_dim_ops=2950963200
additional_work appended_edges=55224
```

这说明 CPU exact materialized additional 的工作量是稳定的，之前 full-quality cross 波动主要不是 merge_add，而是 SearchQueue 输出初始化和 prepare_all/H2D 的波动。

### 4.6 Claim 边界

可以写：full-quality 系统仍有 CPU-compatible output boundary；`additional_edges` 和 host graph materialization 可以成为比 GPU fused topK kernel 更大的端到端瓶颈。Graph reserve、NeighborList64 和 SearchQueue lazy reserve 是有效的低风险系统优化，证明 allocator/per-node object churn 是真实瓶颈。当前 best additional 路径应使用 CPU exact materialized，而不是 CPU Vamana additional 或 direct append。

不能写：不能把 skip additional 当 full-quality；不能把 reserve、NeighborList64 或 SearchQueue lazy reserve 写成 GPU-native graph backend 已完成；不能把 direct append additional 写成正结果；如果把 additional_edges 改成 GPU/flat backend，必须重新做 full-quality filtered-search recall A/B。

## 5. 特异块图构建与自由状态查询

### 5.1 方法

特异块是面向正常原始 x1 数据集的 label trie / group tree 压缩方法。旧的 `tiny_group_size` / `tiny_group_ratio` 口径已经废弃；当前定义只看 bottom-up `uncovered_points`：

```text
初始 uncovered_points(node) = subtree_points(node)
自底向上遍历 trie
当 uncovered_points(node) > UNG_SPECIAL_BLOCK_MIN_POINTS 时建立 special block
该 block 吸收当前子树中尚未被子 block 吸收的 terminal groups
遇到子 special block 停止，并记录 parent block -> child block
```

当前实现包含四个部分：

| 组件 | 方法 | 说明 |
|---|---|---|
| block 构造 | bottom-up uncovered frontier | 支持平凡 block 和非平凡 block；`special_blocks.csv` 保存 block metadata |
| 普通图跳过 | 只跳过平凡 special root source | 跳过平凡 special 内部普通 group graph、从平凡 special 出发的普通 cross/additional source work；保留其他 group 到平凡 special 点的普通边 |
| special sidecar graph | special group graph + shared CPU target-search helper | 不混入 `Graph::neighbors`；块内图复用 complete/Vamana，inter edges 复用普通 CPU cross 的 exact/Vamana heuristic；`special_edges.csv` 保存 target point、parent block id、`intra/inter` kind |
| free-state query | regular/free 双状态 visited | `UNG_SPECIAL_BLOCK_SEARCH=1` 启用；free 节点默认只走 special edges，`UNG_SPECIAL_BLOCK_FREE_USE_REGULAR=1` 可做 A/B |

重要口径：必须区分总体 special、平凡 special 和非平凡 special。

| 口径 | 定义 | 用途 |
|---|---|---|
| overall special | query 完整覆盖某个 special block root label-set，则该 block member points 算 special points | 衡量特异块机制理论上能接管多少 query candidate work |
| trivial special | block 只有 root terminal group，且无 child blocks | 解释普通图跳过的主要对象；平凡 block 更像“可被特殊边接管的普通 group root” |
| nontrivial special | 非平凡 block，包含多个 member groups 或 child-block 边界 | 更接近块级抽象和跨 group 结构收益，论文 claim 应优先看这个口径 |

### 5.2 Baseline

| Baseline / 对照 | 作用 | 边界 |
|---|---|---|
| 默认 UNG 普通图 + 普通搜索 | full-quality 行为对照 | 不启用特异块时的默认语义 |
| special index + 普通搜索 | 负/诊断对照 | 因普通图已跳过平凡 special 出边，不能作为质量结论 |
| special index + free-state search | 当前正确查询路径 | 必须用 filtered-search recall 验证 |
| old tiny-ratio / frontier 特异点定义 | 废弃口径 | 只能用于说明为什么旧定义过宽，不进入主表 |

### 5.3 数据集/构造形态

| Workload | 证据等级 | 构造形态 | 用途 |
|---|---|---|---|
| Amazon 100% x1 restored labels | coverage statistics | 从 x40 repeat labels 每 40 行取 1 行恢复 x1 label/group size 口径 | 统计 block 覆盖率；不能直接用 x40/x200 repeat group size |
| Amazon 100% x1 restored vectors+labels | preliminary full-quality A/B | 从 x40 repeat vectors/labels 每 40 行取 1 行恢复 x1-restored 数据，重新生成 query/GT | 初步测 build cost 与 free-state query latency；不能写成独立官方原始 x1 |
| Amazon 1% x200 query labels | query-shape statistics | 使用已有 `query_coverage_1000/query_labels.txt`，只作为真实 query label 形态 | 统计 query 加权覆盖率；不把 x200 base group size 当原始数据 |
| synthetic x1 smoke | correctness / integration | 小型 x1 数据，低阈值构造 special blocks | 验证 metadata、sidecar、free-state search、wrapper recall |

### 5.4 覆盖率与验证表

统计 artifact：

```text
/home/graphdb/fv_runs/special_blocks_20260704/bottom_up_special_blocks_dense_T.md
```

数据说明：

```text
base labels: /home/graphdb/fv_runs/special_tiny_subtrees_20260702/amazon_100pct_x1_from_x40_labels.txt
total_points=582117
total_groups=482387
query labels: /home/graphdb/FilterVectorBenchData/fv_amazon_1pct_x200_jitter_nonempty/query_coverage_1000/query_labels.txt
queries=1000
```

完整覆盖率表。这里主表保留覆盖率与最大 block size；每个 T 的 per-query 细节和 block size 分布在同目录的 `bottom_up_special_blocks_dense_T_T*.per_query.csv` / `.blocks.csv` 中。

| threshold | blocks | trivial blocks | nontrivial blocks | tree special points | tree special ratio | tree nontrivial points | tree nontrivial ratio | query matched points | query special points | query special ratio | query nontrivial points | query nontrivial ratio | outside points | outside ratio | max block points |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 25 | 7308 | 137 | 7171 | 579729 | 0.995898 | 567185 | 0.974349 | 26544059 | 22889007 | 0.862302 | 22127790 | 0.833625 | 3655052 | 0.137698 | 6174 |
| 50 | 3817 | 83 | 3734 | 578526 | 0.993831 | 567914 | 0.975601 | 26544059 | 22528776 | 0.848731 | 21893310 | 0.824791 | 4015283 | 0.151269 | 8605 |
| 75 | 2587 | 54 | 2533 | 578081 | 0.993067 | 569190 | 0.977793 | 26544059 | 22286307 | 0.839597 | 21765071 | 0.819960 | 4257752 | 0.160403 | 10724 |
| 100 | 1974 | 36 | 1938 | 577822 | 0.992622 | 570505 | 0.980052 | 26544059 | 22107073 | 0.832844 | 21665853 | 0.816222 | 4436986 | 0.167156 | 13236 |
| 150 | 1322 | 19 | 1303 | 577213 | 0.991576 | 571887 | 0.982426 | 26544059 | 21828777 | 0.822360 | 21506402 | 0.810215 | 4715282 | 0.177640 | 17151 |
| 200 | 1015 | 12 | 1003 | 576210 | 0.989853 | 572116 | 0.982820 | 26544059 | 21653155 | 0.815744 | 21408268 | 0.806518 | 4890904 | 0.184256 | 20273 |
| 300 | 680 | 5 | 675 | 575940 | 0.989389 | 573529 | 0.985247 | 26544059 | 21343821 | 0.804090 | 21203337 | 0.798798 | 5200238 | 0.195910 | 26259 |
| 400 | 516 | 4 | 512 | 575639 | 0.988872 | 573617 | 0.985398 | 26544059 | 21139561 | 0.796395 | 21019305 | 0.791865 | 5404498 | 0.203605 | 29173 |
| 600 | 334 | 0 | 334 | 574760 | 0.987362 | 574760 | 0.987362 | 26544059 | 20784702 | 0.783027 | 20784702 | 0.783027 | 5759357 | 0.216973 | 39121 |
| 800 | 246 | 0 | 246 | 574760 | 0.987362 | 574760 | 0.987362 | 26544059 | 20538635 | 0.773756 | 20538635 | 0.773756 | 6005424 | 0.226244 | 49459 |
| 1200 | 161 | 0 | 161 | 574760 | 0.987362 | 574760 | 0.987362 | 26544059 | 20185007 | 0.760434 | 20185007 | 0.760434 | 6359052 | 0.239566 | 54876 |
| 1600 | 131 | 0 | 131 | 574760 | 0.987362 | 574760 | 0.987362 | 26544059 | 19959167 | 0.751926 | 19959167 | 0.751926 | 6584892 | 0.248074 | 58034 |
| 2400 | 71 | 0 | 71 | 574760 | 0.987362 | 574760 | 0.987362 | 26544059 | 19459294 | 0.733094 | 19459294 | 0.733094 | 7084765 | 0.266906 | 78111 |
| 3200 | 58 | 0 | 58 | 574760 | 0.987362 | 574760 | 0.987362 | 26544059 | 19273181 | 0.726083 | 19273181 | 0.726083 | 7270878 | 0.273917 | 83762 |
| 4800 | 41 | 0 | 41 | 574760 | 0.987362 | 574760 | 0.987362 | 26544059 | 18990017 | 0.715415 | 18990017 | 0.715415 | 7554042 | 0.284585 | 97953 |
| 6400 | 30 | 0 | 30 | 574760 | 0.987362 | 574760 | 0.987362 | 26544059 | 18716830 | 0.705123 | 18716830 | 0.705123 | 7827229 | 0.294877 | 115970 |
| 9600 | 22 | 0 | 22 | 574760 | 0.987362 | 574760 | 0.987362 | 26544059 | 18425675 | 0.694154 | 18425675 | 0.694154 | 8118384 | 0.305846 | 123690 |
| 12800 | 16 | 0 | 16 | 562880 | 0.966953 | 562880 | 0.966953 | 26544059 | 18162469 | 0.684239 | 18162469 | 0.684239 | 8381590 | 0.315761 | 157476 |

读表方式：

- `tree special ratio` 是原始 x1 数据点覆盖率；`tree nontrivial ratio` 排除了平凡 special。
- `query special ratio` 是真实 query 加权覆盖率；按 matched points 加总，不平均 per-query ratio。
- `query nontrivial ratio` 是更严格、更适合论文 claim 的口径。
- threshold 越大，block 数越少，query 完整覆盖 block root 的机会下降，因此 query special ratio 下降。

功能/质量 smoke：

```text
/home/graphdb/fv_runs/special_blocks_20260704/smoke_x1
```

| 验证项 | 结果 | 说明 |
|---|---:|---|
| x1 guard | pass | `UNG_SPECIAL_BLOCKS=1` 但未设置 `UNG_SPECIAL_BLOCK_DATA_MODE=x1` 时拒绝构建 |
| special metadata | `special_block_count=6` | 小型 x1 smoke，低阈值构造 blocks |
| special sidecar | `special_edge_count=106` | intra=70，inter=36；边保存到 `special_edges.csv` |
| direct `search_UNG_index` | recall `1.0` | special index + `UNG_SPECIAL_BLOCK_SEARCH=1` |
| wrapper `run_end_to_end_recall_ab.sh` | recall `1.0` | filtered-search recall gate，artifact: `/home/graphdb/fv_runs/special_blocks_20260704/smoke_x1/wrapper_special2` |

Wrapper smoke 表：

| run | index ms | group ms | cross ms | Lsearch | avg time ms | avg recall |
|---|---:|---:|---:|---:|---:|---:|
| special + free-state | 41.7041 | 8.43006 | 0.697471 | 6 | 1.82924 | 1 |

注意：该 smoke 只说明功能链路和 filtered-search recall 可执行，不能作为性能 claim。

### 5.5 Full-Quality Build/Query A/B

新增初步真实性能 A/B：

```text
report: /home/graphdb/FilterVectorCode_refactor/docs/reports/SPECIAL_BLOCK_BUILD_QUERY_AB_CN.md
artifact: /home/graphdb/fv_runs/special_blocks_20260705/amazon_x1_restored_ab_T100_400_1600
dataset: /home/graphdb/fv_runs/special_blocks_20260705/amazon_100pct_x1_restored_from_x40
```

数据口径：

- base vectors 从 Amazon 100% x40 每 40 行取 1 行，base labels 使用 x1-restored label 文件；这是 `x1-restored-from-x40`，不是独立官方原始 x1。
- query set 为重新生成的 100 条 x1-restored coverage-style filtered queries，GT 由 `compute_groundtruth` 生成。
- build 使用 full-quality filtered-search 路径：`UNG_CROSS_EDGE_IMPL=3`、`UNG_ADDITIONAL_EDGES_IMPL=2`，即 CPU exact cross/additional。

Build-time 结果：

| case | T | skip trivial | index ms | group ms | cross ms | special metadata ms | special intra ms | special inter ms | special overlay ms |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| baseline plain | - | - | `119942` | `2326.52` | `103882` | `0` | `0` | `0` | `0` |
| special T100 skip | 100 | 1 | `564856` | `2376.43` | `100780` | `1545.34` | `154825` | `290070` | `444895` |
| special-as-group T100 skip | 100 | 1 | `941250` | `2412.57` | `122032` | `1545.34` | `152599` | `647846` | `800445` |
| target-nx<=1000 heuristic T100 skip | 100 | 1 | `548867` | `2228.65` | `117047` | `1695.13` | `151841` | `261644` | `413485` |
| GPU intra T100 skip | 100 | 1 | `149437` | `2191.24` | `111586` | `1655.37` | `7808.03` | `12543.9` | `20352` |
| GPU intra+inter T100 skip | 100 | 1 | `159614` | `1519.53` | `122187` | `1684.03` | `7811.03` | `11255.8` | `19066.9` |
| target-nx<=1000 heuristic T400 skip | 400 | 1 | `943617` | `2330.55` | `101832` | - | `232274` | `591847` | `824121` |

Query-time 结果，按 `query_details` per-query mean 统计：

| case | Lsearch | mean query ms | recall | dist calcs | visited | weighted special | weighted nontrivial |
|---|---:|---:|---:|---:|---:|---:|---:|
| baseline plain | 50 | `556.56` | `0.373` | `1212.65` | `487.70` | `0` | `0` |
| special T100 skip | 50 | `408.58` | `0.393` | `642.59` | `600.73` | `0.8106` | `0.7967` |
| special-as-group T100 skip | 50 | `413.02` | `0.393` | `642.60` | `600.74` | `0.8106` | `0.7967` |
| target-nx<=1000 heuristic T100 skip | 50 | `578.49` | `0.393` | `642.59` | `600.73` | `0.8106` | `0.7967` |
| target-nx<=1000 heuristic T400 skip | 50 | `320.96` | `0.398` | `539.30` | `497.44` | `0.7711` | `0.7673` |
| baseline plain | 200 | `347.84` | `0.511` | `1576.53` | `851.51` | `0` | `0` |
| special T100 skip | 200 | `369.35` | `0.524` | `1231.48` | `1100.07` | `0.8106` | `0.7967` |
| special-as-group T100 skip | 200 | `504.64` | `0.524` | `1231.48` | `1100.07` | `0.8106` | `0.7967` |
| target-nx<=1000 heuristic T100 skip | 200 | `458.85` | `0.524` | `1231.48` | `1100.07` | `0.8106` | `0.7967` |
| target-nx<=1000 heuristic T400 skip | 200 | `451.28` | `0.528` | `1055.77` | `924.36` | `0.7711` | `0.7673` |

解释：

- T=100 free-state search 在 L50 上有 per-query latency 改善，约 `1.36x`；L200 没有稳定改善。special-as-group 重构后查询语义基本保持，L50 接近旧 special，L200 更慢。
- T=100 重构前构建成本明显变差，Index time 从 `119.94 s` 增到 `564.86 s`；special-as-group per-source Vamana 进一步变为 `941.25 s`；加入 `target_nx<=1000` brute-force 分流后降到 `548.87 s`，略好于旧 exact-overlay，但仍远慢于 baseline。
- 平凡 special skip 省下的普通 source work 很小：group skip `36` groups / `7317` points；cross skip `169` pairs / `40933` query-vector visits；additional skip `36` groups / `7317` points。
- T=1600 skip-only 在 `timeout 900s` 内未完成完整 build；日志显示 ordinary cross 已完成，但未产出 `[special_edges]` 完成行，因此当前实现还没有可接受的 build-time Pareto 点。

### 5.6 Claim 边界

可以写：在 Amazon 100% x1 restored label 口径下，bottom-up special block 覆盖率很高；以 threshold=100 为例，原始点 overall special coverage 为 `99.26%`，nontrivial special coverage 为 `98.01%`；在真实 query label workload 上，query 加权 overall special coverage 为 `83.28%`，nontrivial 为 `81.62%`。当前代码已实现 sidecar special edges 和 free-state 查询，并通过 `run_end_to_end_recall_ab.sh` 的 filtered-search smoke。x1-restored full-quality A/B 显示，free-state query 可以减少 distance calcs 并在 L50 上改善 per-query latency，但 exact-overlay 和 special-as-group 两种 CPU 构建方式都是明确负结果。

不能写：不能把 x40/x200 repeat 后 group size 当作 x1 原始数据；不能把 special coverage 写成端到端加速；不能用 special index 的普通搜索 recall 评价质量；不能把 synthetic smoke 时间写成性能优势；不能声称当前 special block 带来端到端 Index speedup。special-as-group 的 per-source Vamana search 路径不能作为正向构建方案，需要批处理/GPU 或更严格 fanout 控制。

## 6. 端到端质量流水线

### 6.1 当前是否已有支持

已有。当前 reviewer-facing 流水线是：

```text
scripts/benchmarks/run_end_to_end_recall_ab.sh
```

它执行：

```text
build_UNG_index -> save index -> search_UNG_index -> search_time_summary.csv / query_details_repeat*.csv -> summary.csv
```

如果 `GT_FILE` 不存在且设置 `GENERATE_GT=1`，脚本会调用：

```text
build_dir/tools/compute_groundtruth
```

用 filter-then-bruteforce 生成 filtered ANN ground truth。`search_UNG_index` 输出 `Average_Recall`、`Average_Time_ms`，并在 query detail 中输出 `Time_ms`、`search_time_ms`、`core_search_time_ms`、`MinSupersetT_ms`、`DistCalcs`、`NumNodeVisited`、`NumEntries` 等字段。

### 6.2 唯一有效的建图质量验证标准

对 PG/group graph、cross-edge、additional_edges 或 output boundary 的质量判断，必须使用该端到端 filtered-search recall 流水线，固定：

| 固定项 | 原因 |
|---|---|
| query set 和 query labels | 避免换 query 后 recall 不可比 |
| brute-force GT | 以最终 filtered ANN 目标为标准 |
| `Lsearch` / `K` / repeats | 避免单点参数偶然性 |
| cross-edge 和 additional_edges 口径 | 隔离被测变量 |
| entry provider 和 search backend | 避免入口组或邻接读取改变混入建图质量 |

局部 topK overlap、exact-anchor topK 命中、单 group kNN recall 只能作为诊断指标，不能作为论文里的建图质量标准。原因是导航图质量不是“每个点的最近邻边越多越好”，而是最终能否在受过滤条件约束下高效找到正确查询结果。

### 6.3 当前已有脚本入口

| 脚本/工具 | 用途 |
|---|---|
| `scripts/benchmarks/run_end_to_end_recall_ab.sh` | 通用 full-quality build + search recall A/B |
| `scripts/benchmarks/run_group_graph_router_ab.sh` | sweep group graph router 阈值，并调用端到端 recall A/B |
| `scripts/benchmarks/run_x400_reverse_tail_ab.sh` | x400 full-quality reverse-tail/repair A/B |
| `tools/benchmarks/summarize_group_graph_router_ab.py` | 汇总 router A/B 的 build/search/recall 表 |
| `tools/benchmarks/summarize_x400_ab.py` | 汇总 x400 build、search recall 和结构诊断 |
| `tools/benchmarks/diagnose_ung_graph_structure.py` | 结构诊断，只解释原因，不替代端到端 recall |

## 7. 五条线的端到端关系

五条线对应不同 Amdahl 瓶颈：

| 主线 | 最明确收益 | 最主要端到端障碍 |
|---|---|---|
| cross group edge | x200 full-quality 中相对当前最快 CPU cross baseline 约 `13.14x`；x100 fairness 已比较 CPU exact 128T、cuVS per-group、SGEMM+topK | prepare_all/H2D、GPU kernel 本体和 remaining CPU-compatible output boundary 仍限制 Index speedup |
| ELS/入口组 | coverage-correct GPU path 已有 microbenchmark 和 production 接入；fused compact 后中高质量约 `1300~1500` output groups 区间 GPU 约 `2.7~3.0x` 优于同质量 CPU；`19.85x` 仅相对 CPU scan 弱 baseline | GPU 不是全区间占优；production 仍是 per-query 串行 CUDA provider；缺 batch/end-to-end A/B |
| PG/group graph | x200 best full-quality group `4.42x`；10%x40 full-quality 有 Index `1.53x`、group `3.54x` | small groups、packing/H2D/fill、质量修复成本决定 router 边界，不能单一 GPU 化 |
| additional/output boundary | CPU exact materialized additional 相对 CPU Vamana additional `7.16x`；SearchQueue lazy reserve 将 output storage 从 `1909.9 ms` 降到 `15.8 ms` | 仍是 CPU-compatible graph，不是 flat/CSR GPU-native search backend |
| special block graph/query | x1 restored labels 上 coverage 高；T=100 时 query nontrivial special coverage `81.62%`；已有 sidecar + free-state filtered-search smoke；x1-restored search-only sweep 中低/中 recall 段 Lsearch 需求下降，例如 target recall `0.425` 插值 Lsearch `100.0 -> 69.5`；当前保存路径已从未优化 `40.18 s` 降到 binary-only + skip reordered 的 `21.06 s`，CPU-provider-only 再跳过 LNG 文本集合后为 `19.10 s` | 旧 CPU special overlay 结果含 OpenMP 线程数污染；threadfix 后正常 A/B 中 T100 CPU/CPU 为 index `326.50 s`、overlay `162.42 s`、inter `12.18 s`，GPU intra + GPU inter 为 index `187.95 s`、overlay `18.82 s`、inter `11.12 s`；同环境 no-special baseline 为 index `17.21 s`、cross `2.03 s`。GPU source-skip 已恢复默认 `skip_trivial=1` 下的 ordinary GPU cross，guarded full smoke index `38.65 s`、skip 计数器 `169/40933`、ordinary graph source-mask verifier 通过。cap10 + `UNG_TAGORE_ITER=4` + GPU intra/inter 最新 controlled run 为 index `29.59 s`、overlay `11.21 s`、ordinary cross `1.59 s`。special intra route 复核显示旧 CPU baseline 因 T=100/default complete threshold 64 把全部 `1974` blocks / `577822` points 送入 CPU Vamana，形成 `155.92 s` 的偏慢 baseline；修正后 CPU default 用 distance exact-topK `th=2048`，intra `75.88 s`、coverage L50/L200 `0.341/0.400`，GPU default 调到 `th=128` 后，小块 exact-topK、大块 Tagore/FastGrnnd CUDA，intra `5.11-5.28 s`、相对修正 CPU 约 `14.4x`；GPU inter 通过 `2` warps/query 把 kernel `~5.34 s -> ~1.87 s`、inter total `~6.1 s -> ~2.7 s`。bounded ring 虽快但 recall 降到 `0.323/0.370`，不能作为默认；cap10 仍是近似 knob，query timing 仍受 MinSupersetT、index load 和 outlier 影响，不能直接作为稳定 query speedup 或默认 full-quality 结论；`UNG_SKIP_LNG_TEXT_SETS=1` 仅适用于 CPU entry-provider / UNG special search，GPU cover-frontier provider 仍需要 `_lng_descendants` |

因此，若问题是“距离端到端有最大加速比的最大障碍是什么”，当前答案不是某个单独 kernel，而是三个接口边界：

1. cross group 目前相对真正 CPU baseline 已经超过 `13x`，但若看完整 Index，prepare_all/H2D、kernel 本体和 CPU-compatible 输出边界仍是主要 Amdahl 项。
2. 查询侧 `gpu_cover_frontier` 需要 batch/end-to-end route，证明 provider 加速不会被更多 entry groups 和 graph expansion 抵消。
3. group graph 已有 x200 `4.42x`，但必须继续按 workload 路由；小组和输出填充边界会限制 GPU 算子加速转化为 Index speedup。
4. special block 目前是结构和查询语义的新主线；覆盖率高且低/中 recall 段 recall curve 左移。CPU target-size heuristic 比 per-source Vamana 更合理；root-cause audit 发现旧 CPU inter 慢主要是 OpenMP 线程数泄漏，threadfix 后 CPU inter 与 GPU-intra CPU inter 同量级。GPU intra 的当前解释已经更新：旧 CPU baseline 的 `155.92 s -> 5.16 s`、`30.24x` 是未修正 CPU Vamana 口径；修正后的 CPU default 使用 distance exact-topK `th=2048`，CPU intra `75.88 s`，coverage L50/L200 `0.341/0.400` 接近旧 CPU Vamana `0.344/0.401`；GPU default 调到 `th=128`，intra `5.11-5.28 s`，相对修正 CPU baseline 约 `14.4x`；`th>=256` 变慢。GPU inter 已把 warps/query 从 8 调到 2，kernel `~5.34 s -> ~1.87 s`，inter total `~6.1 s -> ~2.7 s`。cap=1M probe 可把 index 降到 `29.23 s`，但第二套 query_mix workload 出现一致小幅 mean recall 损失；cap=10M default-iter index 为 `33.36 s`、overlay `13.47 s`，cap20 为 `34.63 s`、overlay `14.83 s`，二者在新增 broad-heavy length-2 workload 上主要只剩 L800 mean `-0.003`、最差 `-0.4`。cap10+`UNG_TAGORE_ITER=4` + GPU intra/inter controlled run index 可到 `29.59 s`、overlay `11.21 s`，但同一 broad2 workload 出现更明显退化：L50/L100/L200/L400/L800 mean recall delta 为 `-0.003/-0.003/-0.004/-0.003/-0.012`，最差单点 `-0.5`。进一步补 no-cap+iter4 index 后，broad2 上逐 query/repeat recall 与 no-cap default 完全一致；因此 extra regression 不是 cap10 或 iter4 单独造成，而是 cap 与低 iter 图近似的交互风险。离线 adaptive heavy-edge 模拟显示：如果 broad2 中 `matched_groups>=20000` 的 `17%` query 使用 no-cap/heavy sidecar，其余 query 使用 cap10+iter4，就能完全恢复 L800 退化；subset search 估算中该 two-tier 组合的 broad2 L800 加权结果为 `1157.6 ms / 0.488 recall`，恢复到 no-cap recall。进一步的单进程 env-gated two-tier 原型 smoke 中，load-time split + adaptive heavy（`MIN_MATCHED_POINTS=20000`）在 broad2 L800 达到 `1103.19 ms / 0.488 recall`，light-only 为 `1291.38 ms / 0.485 recall`；扩展 L20-L800 后 broad2 的 adaptive-heavy recall 为 `0.328/0.351/0.361/0.400/0.431/0.488`，基本恢复 no-cap，len1 broad recall 不变。新增 `MIN_QUERY_SIZE=2` gate 后，broad2 L800 仍为 `0.488`，而 len1 broad 的 heavy 触发从 `144` 行降为 `0`。四套 workload（coverage、query_mix、broad2、len1）扩展验证中，adaptive-heavy 与已有 no-cap 对照在所有可对齐 L 上 recall delta 均为 `0`；触发 query 数分别为 `20/14/22/0`。交替重复 broad2 L800 A/B 中，adaptive 恢复 recall且未观察到 heavy-triggered query 的系统性 overhead；但 non-heavy queries 同样变快，说明 timing 仍受 run-order/系统噪声影响，不能写成稳定加速。成本估算也支持该方向：cap10 light sidecar 相对 no-cap 少约 `3.26M` 条 special edges。持久化 split-file smoke 已能写出 light `949MB` 和 heavy `77MB` 两个文件；同一 index 的 merged no-split 控制实验与 split adaptive 都是 broad2 L800 `0.484`，说明 split 保存/加载不是质量差异主因，旧 no-cap/load-time split `0.488` 的差异主要来自重新构建图波动。保存 profiling 显示 special edge CSV 写入约 `7.8 s`，但完整 save 最大项进一步定位到 trie：原本 graph/trie stage `16.0 s` 中，trie 保存 `14.0 s`。修复 `TrieIndex::save()` 每行 `std::endl` flush 后，trie 保存降到 `6.1 s`，graph/trie 降到 `8.1 s`。special sidecar binary 原型显示 light/heavy 体积可从 `949/77MB` 降到 `619/50MB`，二进制读入约 `0.67/0.05 s`；repo 内 binary-only sidecar 可正确加载并在 broad2 L800 达到 `0.488 recall`，special block save 降到 `1.88 s`、edge 写入 `1.82 s`，且 binary-only index 现在可在 CSV 缺失时自动读取 `.bin`，避免空 sidecar。进一步跳过 ACORN reordered export 后，保存阶段降到 `21.06 s`，但 ACORN 路径不可用；当前 special-block UNG search smoke 仍可运行。再跳过 LNG 文本集合后，CPU-provider-only 保存阶段降到 `19.10 s`，其中 `lng_sets_ms` 从 `4204.6 ms` 到 `637.6 ms`，并通过 broad2/coverage CPU entry-provider search smoke；但 GPU cover-frontier provider 仍需要 `_lng_descendants`，所以该开关不能作为通用默认 index 格式。所有 cap 都是近似语义，不能直接作为默认方法；后续方向应是更多 workload 验证和正式两层 sidecar / query-adaptive heavy-edge，而不是单一全局 cap。

## 8. Reviewer Reproduction Checklist

| 主线 | 主 artifact / summary | 复现或验证入口 | 质量等级 | 备注 |
|---|---|---|---|---|
| Cross group edge | `/home/graphdb/fv_runs/cpu_cross_baseline_sweep_20260702_184801`; `/home/graphdb/fv_runs/cross_lazy_reserve_best_20260702_181842` | 文档第 1.4 节记录 commands/artifacts；必要时重跑对应 benchmark scripts | full-quality / stage fairness | x200 full-quality 与 x100 fairness 分开读。 |
| ELS / entry provider | `tools/benchmarks/query_entry_group_bench*` outputs; 文档第 2 章 pair sweep 表 | `scripts/benchmarks/run_query_entry_group_bench.sh` / sweep scripts | microbenchmark | 输出 group 数是质量指标；缺 end-to-end batch query。 |
| PG/group graph | `/home/graphdb/fv_runs/x200_direct_h2d_lsearch_sweep_20260702/x200_lsearch_sweep_summary.md`; `FASTGRNNDCUDA_AMAZON_SCALE_RESULTS_CN.md` | group graph build scripts and `search_UNG_index` recall sweep | full-quality | 高 L recall 对齐；低 L 小差距需保留。 |
| additional/output boundary | `best additional probe` and graph reserve artifacts in 第 4 章 | build logs + `build_time.csv` | full-quality boundary | CPU-compatible graph 边界，不是 GPU-native search。 |
| Special block intra/inter | `/home/graphdb/fv_runs/special_blocks_20260705/special_gpu_inter_warps2_default_smoke_20260707_114920/summary.txt`; 2x2 artifacts | `SPECIAL_BLOCK_BUILD_QUERY_AB_CN.md` 0.4 命令 + artifact scripts | approximate candidate / smoke | cap10/iter4 不是无损默认；使用 filtered-search recall。 |
| Special save/load | `/home/graphdb/fv_runs/special_blocks_20260705/skip_export_binary_lngskip_build_20260706_121730/save_extract.txt` | build env in artifact `others/env.log` | CPU-provider-only save optimization | skip LNG text 不适用于 GPU cover-frontier provider。 |

所有论文/报告 claim 修改后都应运行：

```bash
cd /home/graphdb/FilterVectorCode_refactor
python3 tools/benchmarks/check_paper_claim_language.py docs/reports/THREE_MAINLINES_METHOD_BASELINE_DATA_SPEEDUP_CN.md docs/reports/SPECIAL_BLOCK_BUILD_QUERY_AB_CN.md docs/reports/SPECIAL_BLOCK_GRAPH_QUERY_DESIGN_CN.md docs/reports/REVIEW_READY_HANDOFF_CN.md docs/reports/REVIEW_READY_REQUIREMENTS_MATRIX_CN.md
```

## 8. 证据入口

| 文档 | 用途 |
|---|---|
| `docs/reports/SYSTEM_SCOPE_PERFORMANCE_AND_LIMITS_CN.md` | 当前第一入口，说明系统组成、性能优势、适用环境和 claim 边界 |
| `docs/reports/CURRENT_METHODS_EFFECTS_AND_GAPS_CN.md` | 更详细的方法、效果、负结果和缺口总览 |
| `docs/runbooks/UNG_METHOD_REGISTRY_CN.md` | 方法注册表，确认每条方法的入口、开关和语义边界 |
| `docs/papers/EVIDENCE_MATRIX_CN.md` | claim-level 证据矩阵，防止 overclaim |
| `docs/reports/TECHNICAL_REPORT_OPTIMIZATION_SPEEDUP_CN.md` | 长技术报告和原始证据链 |
| `docs/reports/FASTGRNNDCUDA_AMAZON_SCALE_RESULTS_CN.md` | PG/group graph 的 Amazon scale 结果来源 |
| `docs/reports/SPECIAL_BLOCK_GRAPH_QUERY_DESIGN_CN.md` | 特异块定义、sidecar 图、free-state 查询、x1-only 口径和 smoke 验证 |
| `docs/reports/SPECIAL_BLOCK_BUILD_QUERY_AB_CN.md` | special block x1-restored full-quality build/query A/B、负结果和下一步 |
| `scripts/benchmarks/run_end_to_end_recall_ab.sh` | 建图质量唯一有效验证入口：build 后跑 filtered-search recall |
| `scripts/benchmarks/run_special_block_build_query_ab.sh` | special block threshold/skip A/B 编排 |
| `scripts/benchmarks/run_query_entry_group_sweep.sh` | ELS/入口组超参 sweep：性能 vs avg output groups |
