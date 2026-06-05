# UNG 构建优化最终技术报告

日期: 2026-06-01
代码路径: `/home/graphdb/FilterVectorCode_refactor`
主要数据集: Amazon sampled + jitter repeat 系列
GPU: NVIDIA RTX A6000

## TLDR

一句话结论：我们已经把 UNG 的 GPU 化从“单个 kernel 加速”推进到“按 workload 分流的系统设计”，其中 cross-edge 是当前最稳定的加速贡献，group graph 需要 size/quality-aware router，查询入口组 GPU 化已经证明阶段性可行，但 Amazon 100%x40 暴露出后续 graph search 的 CPU 随机访存和队列维护瓶颈。

| 主题 | 当前做到的结果 | 现在不能夸大的边界 |
|---|---|---|
| cross-edge | Amazon 1% x100 上 fused topK cross-edge `848.9 ms`，相对 CPU Vamana `22.07x`，相对 CPU exact scan `2.81x`，相对 cuVS per-group `4.39x`，相对 SGEMM+topK `1.64x` | 100%x40 的 X-streaming 已能完成但 `pack_q=450.3s`，还不是大规模加速结果 |
| group graph / PG | packed exact-anchor、FastGrnndCuda、reverse-tail+repair 和 bounded fallback 已组成 workload-aware router；10%x40 full-quality Index `36053.3 -> 23497.4 ms`，约 `1.53x` | 不能写成 GPU 无条件替代 CPU Vamana；x200/x400 说明 prune 强度和图质量必须权衡 |
| 查询入口组 | GPU correct-cover 在 Amazon 100%x40、`nq=10240` 上 `59.83 ms`，相对 CPU scan 128T `19.85x`；CPU exact minimal 为 `4569.83 ms` | 这是入口组阶段独立 benchmark，尚未接入正式 `search_UNG_index` 端到端路径 |
| Amazon 100%x40 | 修复了 32-bit offset 溢出和 71.5GB resident-cache OOM，完成 `23,284,680` 点、`482,387` groups 的 strict build | 当前用于暴露扩展性瓶颈，不用于主加速表 |
| 查询 graph 异常 | 100%x40 L1000 中 `DistCalcs` 只比 10%x40 高约 `2.15x`，但 `core_search_time_ms` 从 `1.69` 到 `549.67`，说明现有指标低估真实 CPU 工作量 | 目前是强问题定位，不是最终定论；下一步需要细粒度计数器和硬件计数验证 |

最重要的发现：

1. **GPU 加速有效，但必须按负载形态分流。** 小组、多组、中组、大组的瓶颈不同，单一路径会在某些 workload 上变成反例。
2. **真正的系统瓶颈正在从 GPU kernel 转向外围。** host pack、H2D/D2H、Graph fill、SearchQueue、VisitedSet、邻接表随机访问开始主导大规模场景。
3. **入口组 exact minimal 不是唯一可接受语义。** GPU correct-cover 保证不漏候选 group，允许少量冗余，是更适合批处理的查询入口策略。
4. **100%x40 的查询 graph search 是下一轮核心研究问题。** 仅看 `DistCalcs/NumVisited/NumEntries` 已经解释不了耗时，下一步要测 `edge_scans`、`visited_hits/misses`、`queue_memmove_bytes`、cache/TLB miss 和多线程带宽争用。

## 0. 最终结论

本轮优化后的结论需要分成 cross-edge 和 group graph 两条线报告，不能只看单个 kernel。

1. **cross-edge exact topK 已经有明确收益。**
   Amazon 1% x100 最新重测中，当前 direct-qid fused 路径 cross-edge 为 `848.9 ms`。相对 CPU Vamana cross-edge `18735.8 ms` 是 `22.07x`，相对 CPU exact scan `2389.8 ms` 是 `2.81x`，相对 cuVS per-group `3730.5 ms` 是 `4.39x`，相对 SGEMM+topK `1395.9 ms` 是 `1.64x`。

2. **端到端收益主要来自“cross-edge fused + 自适应剪枝强度”，这比简单 CPU/GPU route 更接近系统级替代设计。**
   当前最快工程口径是：cross-edge 用 fused topK；group graph 不再按 `nx≈200` 直接回退 CPU，而是对中等组使用 diversified light prune，对大组保留更强剪枝。Amazon 1% x100 的历史最快 build-scale Index time 从 CPU Vamana group + 旧 GPU fused baseline `16824 ms` 降到 `6630 ms`，为 `2.54x`，但该历史点是 skip-additional 口径。新补的 x100 full-quality coverage-query repeat=3 A/B 固定 GPU fused cross 和 CPU additional_edges，Index `6863.47 -> 6370.67 ms`，L100/L500/L1000 recall `0.825/0.868/0.890667 -> 0.828/0.871/0.896`，说明 x100 质量闭环通过但 group speedup 只有 `1.03x`。新补的 Amazon 1% x200 coverage-query 暴露出旧 FastGrnndCuda 重 prune 的问题：group graph `26620.6 ms`，慢于 CPU Vamana 的 `8511.72 ms`。加入 diversified light prune 后，x200 group graph 降到 `6706.36 ms`，Index 降到 `16478.5 ms`，相对 CPU Vamana 分别为 `1.27x` 和 `1.07x`。

3. **`prepare_all` 必须单独列。**
   它是一次 `build_UNG_index` 进程内的全量 base vectors GPU 常驻准备，包含 host registration、全量 H2D、全量 norm。它不是每个 group 重复发生，也不会跨进程保留。单次 build 必须计入 cross time；评估算法本体时可以另报 `resident cross = cross total - prepare_all`。

4. **x400 暴露了 direct-qid 大规模稳定性问题。**
   x400 direct-qid fused 在 medium group-desc kernel 触发 illegal memory access；减小 chunk cap 仍失败。稳定可复现的大规模结果需要关闭 direct-qid，回到 gather-Q fused 路径。报告主表对 x400 使用稳定 gather-Q 结果，并把 direct-qid 失败作为限制列出。

5. **端到端质量已有 Amazon PF、SIFT30 和 Amazon 1% x200 coverage 三组证据；x200 已从反例变成 speed/quality tradeoff。**
   原始 Amazon PF 上，FastGrnndCuda + CPU fallback 相对 CPU Vamana group 的 recall drop 小于 `0.001`；SIFT30 上，修复 fused cross-edge 语义后 Lsearch=1000 recall 从 `0.877587` 到 `0.876297`，下降约 `0.00129`。Amazon 1% x200 coverage-query 中，旧重 prune 版本 L1000 recall `0.8662` 但 build 慢；diversified light prune 版本 Index `16478.5 ms` 快于 CPU `17596.8 ms`，L1000 recall `0.8656`，L5000 recall `0.9056`，相对 CPU L5000 `0.9080` 低 `0.0024`。因此下一步重点是提升 light prune 的质量，而不是回退 CPU。

6. **早期 batched exact kNN 负结果被重新归因，packed exact-anchor 成为强候选。**
   早期 x200 exact run 的 L5000 只有 `0.819867/0.829`，但结构诊断显示旧 exact-anchor 与新 packed exact 的组内图完全一致，差异来自 cross edges：旧 run `1,038,000`，CPU/new packed 都是 `1,093,224`。因此旧结果不能作为“exact 组内图不可行”的证据。修正后，packed exact-anchor 在 full-quality 口径下 x200 group graph `3827.28 ms`、Index `11043.5 ms`、repeat=3 L1000/L5000 `0.869/0.908`；x400 group graph `11588.4 ms`、Index `28436.3 ms`、repeat=3 L1000/L5000 `0.945/0.967`。

7. **Amazon 10% x40 的最新结果把替代路线推向系统级设计。**
   `10%x40` 有 `2,409,800` 点和 `53,840` 个 group，是大量小/中组压力场景。第一轮 build 暴露 `target_groups=49374` 时 cross-edge 在 GPU event 前卡住；修复 direct-pageable upload 和 direct-qid chunk kernel 后，skip-additional 口径下 CPU exact cross 为 `54879.3 ms`，新 GPU direct-qid cross 为 `8475.4 ms`，cross 加速 `6.47x`。最新 full-quality A/B 使用 CPU additional_edges：CPU Vamana group Index `36053.3 ms`，FastGrnndCuda + complete fallback Index `23497.4 ms`，Index speedup `1.53x`；repeat=3 下 L100/L500/L1000 recall 从 `0.8647/0.898/0.908` 到 `0.8668/0.898/0.9103`，P95/P99 latency 未恶化。

8. **进一步的 10%x40 反例说明，小组不应盲目 GPU 化。**
   新 packed exact-anchor 在 x200/x400 上成立后，我们尝试把 10%x40 的 complete threshold 从 `64` 降到 `32`，让 `53,833` 个小/中组进入 GPU exact。结果 group graph 从 `6097.03 ms` 变慢到 `7329.12 ms`：`pack=2728.81 ms`、`h2d=1014.8 ms`、`fill=1968.77 ms` 成为主因，L100 recall 仍为 `0.867`。因此小组场景的瓶颈不是 exact kernel 算不动，而是 host pack、PCIe、Graph fill 和小组图物化。我们新增 `UNG_GROUP_GRAPH_BOUNDED_COMPLETE=1`，在 `nx<=max_degree+1` 保持 complete graph，在更大小组中每点写 `max_degree` 条环形强连通边。10%x40 full-quality build 复测中，Index 从 `25035.9 ms` 降到 `22230.2 ms`，group 从 `6097.03 ms` 降到 `5240.6 ms`，fallback_wall 从 `4225.67 ms` 降到 `3088.13 ms`。复用该 index 的 repeat=3 L100/L500/L1000 search sweep 为 `0.867/0.900/0.909`，与 CPU `0.8647/0.898/0.908` 同量级。该结果支持的路线是 size-aware router：很小组 bounded-complete，中组 packed exact-anchor，大组 GNN/reverse-tail。但 bounded sweep 的 query time `1083.34/951.434/839.479 ms` 比历史 CPU/FastGrnnd run 慢，当前只能写成 build-side 初步正结果和 recall sanity，不能写成 search latency 改进。

9. **Amazon 1% x400 说明替代路线仍受质量/剪枝预算限制。**
   x400 full-quality gather-Q A/B 中，历史 CPU Vamana group Index 为 `37807.6 ms`；同脚本 CPU baseline 后续复测为 `36257.3 ms`、L1000/L5000 `0.946/0.965`。FastGrnndCuda heavy prune 的 L5000 recall 只从 `0.9660` 降到 `0.9643`，但 prune 为 `19977.7 ms`，Index 变慢到 `52552.2 ms`。将 light-prune 阈值提高到 `512` 后，prune 降到 `1022.45 ms`，Index 降到 `30826.5 ms`，相对 CPU `1.23x`；但 L1000/L5000 recall 均低 `0.0081`。`LIGHT_HEAD=28` 进一步说明简单多保留近邻不是解法：Index 仍为 `30897.7 ms`，但 L5000 recall 降到 `0.9523`。因此 x400 支持“自适应剪枝可以换性能”，不支持“无损 GPU group graph 普遍替代”。

## 0.5 审稿人攻击面与补实验路线

下面按 reviewer 可能提出的问题组织。后续实验应优先回答这些问题，而不是只继续整理已有结果。

### Q1: 端到端查询质量是否被 GPU group graph 换掉了？

早期报告最弱的一点是 group graph 主要用单组 graph recall / 构建时间支撑，而不是最终 filtered search recall。现在已经补了原始 Amazon PF 和 SIFT30 两组端到端 A/B，但审稿人仍会追问：FastGrnndCuda 不是 CPU Vamana 的语义等价实现，是否在 Amazon repeat-scale 或大规模 x400 上质量下降？

必须补：

| 实验 | 对照 | 指标 | 最低要求 |
|---|---|---|---|
| Amazon 1% x100 end-to-end filtered search | 已补 CPU Vamana group vs FastGrnndCuda + CPU fallback，固定 GPU fused cross 和 CPU additional_edges | full-quality repeat=3 下 recall 不降；Index `1.08x`，group `1.03x` | 作为质量闭环证据，不作为强 group speedup 证据 |
| Amazon 10% x40 end-to-end filtered search | 同上 | recall、latency、build time | 已补 full-quality repeat=3 A/B 和 L50/100/200/500/1000/2000/5000 sweep；latency 非单调，不能写成稳定 search-latency 加速 |
| x400 稳定 gather-Q end-to-end search | CPU Vamana group baseline, FastGrnndCuda | recall、latency、build time | 如果 direct-qid 不稳定，主结论必须只基于 gather-Q |

已补证据：

| 数据集 | 固定变量 | Group speedup | Index speedup | Recall 变化 |
|---|---|---:|---:|---:|
| 原始 Amazon PF | fused cross-edge + CPU additional edges | `5.50x` | `1.18x` | L5000 `0.971767 -> 0.971133`，`-0.000634` |
| SIFT30 | fixed fused cross-edge + CPU additional edges | `5.95x` | `1.27x` | L1000 `0.877587 -> 0.876297`，`-0.001290` |
| Amazon 1% x200 coverage, old heavy prune | fused cross-edge + CPU additional edges | `0.32x` | `0.46x` | L1000 `0.8690 -> 0.8662`，`-0.0028` |
| Amazon 1% x200 coverage, diversified light prune | fused cross-edge + CPU additional edges | `1.27x` | `1.07x` | L1000 `0.8690 -> 0.8656`，`-0.0034`；L5000 `0.9080 -> 0.9056`，`-0.0024` |
| Amazon 1% x200 coverage, mixed exact/GNN router nx256, stream/fill optimized | fused cross-edge + CPU additional edges | `1.35x` vs CPU, `1.49x` vs router0 | `0.96x` vs CPU, `1.15x` vs router0 | repeat=1 L100/L500/L1000 `0.823/0.849/0.869 -> 0.8229/0.851/0.869` |
| Amazon 1% x200 coverage, old batched exact kNN | mixed / confounded | `1.95x` | `1.40x` | L5000 `0.8199`，但 cross_edges 少 `55,224`，不能隔离组内图质量 |
| Amazon 1% x200 coverage, packed exact-anchor router | fused cross-edge + CPU additional edges | `2.22x` vs CPU group, `1.29x` vs旧 scatter exact | `1.59x` vs historical CPU index | repeat=3 L1000/L5000 `0.869/0.908`，与 CPU `0.869/0.908` 对齐 |
| Amazon 10% x40 coverage, old prepare path | fused cross-edge + CPU additional edges | - | - | query/GT 已生成；第一轮 build 在 cross-edge `target_groups=49374` 阶段超过 10 分钟未进入 GPU event |
| Amazon 10% x40 coverage, fixed direct-qid, skip additional | GPU direct-qid cross + skip additional | group `3.39x` vs CPU exact run | Index `3.39x` vs CPU exact run | L100 sanity `0.800`；由于跳过 additional edges，不能作为完整质量结果 |
| Amazon 10% x40 coverage, fixed direct-qid, full additional | GPU direct-qid cross + CPU additional_edges | group `3.54x` | Index `1.53x` | repeat=3 L100/L500/L1000 `0.8647/0.898/0.908 -> 0.8668/0.898/0.9103` |
| Amazon 10% x40 coverage, packed exact for almost all groups | GPU direct-qid cross + CPU additional_edges | group `0.83x` vs complete64 exact run | Index `1.02x` vs complete64 exact run | L100 `0.867` 不变，但 `pack/h2d/fill` 过高，证明 all-small-groups GPU exact 是负结果 |
| Amazon 10% x40 coverage, bounded-complete fallback | GPU direct-qid cross + CPU additional_edges | group `1.16x` vs complete64 exact run | Index `1.13x` vs complete64 exact run | L100/L500/L1000 `0.867/0.900/0.909`，recall 不降；search latency 比历史 run 慢，需复测诊断 |
| Amazon 1% x400 gather-Q, heavy prune | gather-Q cross + CPU additional_edges | `0.60x` | `0.72x` | L1000/L5000 `0.9460/0.9660 -> 0.9443/0.9643`，质量接近但更慢 |
| Amazon 1% x400 gather-Q, light512 | gather-Q cross + CPU additional_edges | `1.69x` | `1.23x` | L1000/L5000 `0.9460/0.9660 -> 0.9379/0.9579`，build 更快但 recall drop `0.0081` |
| Amazon 1% x400 gather-Q, light512 head28 | gather-Q cross + CPU additional_edges | `1.71x` | `1.22x` | L1000/L5000 `0.9460/0.9660 -> 0.9323/0.9523`，说明减少尾部多样性会继续降质 |
| Amazon 1% x400 gather-Q, light512 reverse-tail+repair | gather-Q cross + CPU additional_edges | `1.23x` vs same-script CPU group | `1.10x` vs same-script CPU index | L1000/L5000 `0.946/0.965 -> 0.945567/0.9659`，基本补回 light512 gap |
| Amazon 1% x400 gather-Q, packed exact-anchor router | gather-Q cross + CPU additional_edges | `1.51x` vs historical CPU group | `1.33x` vs historical CPU index | repeat=3 L1000/L5000 `0.945/0.967`，接近/略高历史 CPU `0.946/0.966` |

Amazon 1% x200 coverage 最初是反向证据，不应被隐藏。该 workload 的组大小集中在 `nx=200`，旧 FastGrnndCuda 的 GPU prune 成本无法摊销，导致 group graph 比 CPU Vamana 更慢。diversified light prune 说明问题可以通过剪枝强度自适应缓解：中等组不做重 RNG occlusion，而保留近邻头部并从尾部候选采样少量多样性邻居，性能转正；但质量仍有约 `0.2%~0.3%` recall gap，需要继续做轻量 reverse augmentation 或低成本 occlusion。

Amazon 10% x40 是新的系统压力点。balanced coverage query 已生成，matched groups avg/p50/p95 为 `883.09 / 25.5 / 3297.0`，matched points avg/p50/p95 为 `39910.08 / 1060.0 / 138738.0`，GT exact compute `16076 ms`。第一轮 full-quality CPU Vamana group + fused cross build 在 `target_groups=49374` 后超过 10 分钟没有进入 `[GPU GEMM] H2D/Kernel/D2H` 输出，说明大量小组场景下 cross-edge 的 host prepare、descriptor scheduling 或 launch 前路径已经成为 reviewer 级 blocker。后续修复显示主要问题不是 GPU compute 本身，而是外围：旧路径误走 pinned staging，7.4GB host repack 花 `6205.6 ms`，`prepare_all total=6500.7 ms`；修复后 direct pageable 路径 `prepare_all≈2.2-2.7 s`，`host_pack=0`。full-quality A/B 已经跑通，FastGrnndCuda 在该 workload 上给出 `1.53x` Index speedup 且 recall 不降。

Amazon 1% x400 是新的质量/剪枝预算压力点。它有 `2,409,600` 点、`5,666` 个 group，coverage query 的 matched groups avg/p50/p95 为 `73.37 / 21.0 / 134.2`，matched points avg/p50/p95 为 `31016.0 / 8400.0 / 53700.0`，GT exact compute `14039 ms`。在 stable gather-Q + CPU additional_edges 口径下，heavy prune 证明图质量可以接近 CPU，但 prune `19977.7 ms` 使 Index 变慢；light512 证明性能可以超过 CPU，但 L1000/L5000 recall 各低 `0.0081`。`LIGHT_HEAD=28` 比默认 head24 更差，因为当前 light kernel 会用剩余名额从 GNN 候选尾部等距抽样，增大 head 会挤掉多样性边。新的 x400 repair/reverse-tail A/B 显示，degree repair 单独几乎消除 zero/low-degree，但 Index 从 `30128.1` 变慢到 `36310.5`，L5000 只从 `0.9574` 到 `0.957767`；reverse-tail+repair 则以 `33044.9 ms` Index 达到 L1000/L5000 `0.945567/0.9659`，基本贴近同脚本 CPU `0.946/0.965`，Index 约 `1.10x`。因此 x400 的有效方向不是单纯补边或多保留 head nearest neighbors，而是低成本 tail/reverse diversity。

2026-06-02 进一步做了四个不改变搜索接口的外围/调度优化。第一，Tagore/FastGrnnd GPU batch默认开启 `UNG_TAGORE_COMPACT_D2H=1`，prune 后先把 `k` 宽的候选表压缩成 `final_degree+1` 宽再 D2H；典型 `tagore_k=64, max_degree=32` 时，每点图写回从 64 个 `uint32` 降到 33 个，D2H、host graph buffer 和 CPU fill 的无效数据约减半。第二，新增 `UNG_TAGORE_BATCH_STREAMS`，把 many-group workload 按点数拆到多个 non-blocking CUDA stream，避免单 group 小 kernel 串行填不满 GPU。第三，新增 `UNG_TAGORE_FILL_FAST` 和 `UNG_TAGORE_FILL_THREADS`，减少 CPU 端 graph fill 的重复检查和过度线程竞争。第四，batched exact path 新增 packed graph buffer：D2H 后不再把连续 graph 拆成每个 group 的 `result.graph` 再由 fill 阶段读一遍，而是在 `UniNavGraph` fill 中按 packed offset 直接读。packed exact-anchor 把 x200 exact-router group 从旧 scatter run 的 `4929.54 ms` 降到 `3827.28 ms`，Index 降到 `11043.5 ms`，repeat=3 L1000/L5000 `0.869/0.908`；x400 同路径 group 为 `11588.4 ms`，Index `28436.3 ms`，repeat=3 L1000/L5000 `0.945/0.967`。同时，结构诊断显示旧 exact 低 recall 是 cross/additional 口径混杂，而不是组内 exact 图本身的失败。

新增结构诊断证据来自 `tools/benchmarks/diagnose_ung_graph_structure.py`，输出保存在 `/home/graphdb/fv_runs/graph_diagnostics_x400_20260602`，比较表由 `tools/benchmarks/summarize_graph_diagnostics.py` 生成到 `/home/graphdb/fv_runs/graph_diagnostics_x400_20260602/summary_table.md`。该脚本直接扫描保存的 x400 index graph，不重建索引。CPU Vamana 与 light512 的 cross edges 相同，都是 `2,131,224` 条；差异集中在组内图：light512 的 intra edges 从 `75,767,742` 降到 `73,538,978`，少 `2,228,764` 条，平均组内出度从 `31.4441` 降到 `30.5192`，零组内出度比例从 `0` 增到 `0.00199162`，低组内出度 `<=4` 比例从 `0.000000415` 增到 `0.00929781`，并出现 `39` 个最大 weak component 小于 `0.9` 的 group。这个诊断把 x400 recall gap 归因到组内图结构稀疏/局部分裂，而不是 cross-edge 或 additional_edges 差异。

注意：SIFT30 fused strict 在 reviewer 迭代中暴露过两个问题并已修复。第一，`DIRECT_QID_ALL_FUSED=1` 会让超大组 fallback path 读未 materialize 的 `g_d_Q`；第二，host merge 路径下 id-only writeback 用 rank 代替真实距离，破坏跨 target-group 排序。修复后 SIFT30 fused cross 的 CPU Vamana group L1000 recall 为 `0.877587`，与 CPU cross `0.877523` 基本一致。

### Q2: baseline 是否公平，还是挑了一个很弱的 CPU Vamana cross？

cross-edge 现在已经有 CPU exact scan、cuVS per-group、SGEMM+topK 和 fused 的表，但论文写作时不能只强调相对 CPU Vamana 的 `22.07x`。审稿人会要求强 baseline。

baseline fairness 表已固化为脚本输出：

```text
python3 tools/benchmarks/summarize_cross_baselines.py ...
/home/graphdb/fv_runs/baseline_fairness_x100_20260602/cross_baselines.md
```

实验机器 CPU 为 Intel Xeon Platinum 8360Y，2 sockets，36 cores/socket，144 logical CPUs；x100 CPU exact baseline 的 `build_num_threads=128`。该表显示 final fused 相对 CPU Vamana cross 为 `22.07x`，相对 CPU exact 128 线程为 `2.82x`，相对 cuVS per-group 为 `4.39x`，相对 SGEMM+topK 为 `1.64x`。

必须补：

| 问题 | 需要补的内容 |
|---|---|
| CPU exact scan 是否充分并行？ | 已有 128 线程结果和 CPU 型号；若时间允许，继续补 1/32/64/128 线程 scaling |
| cuVS per-group 是否被不公平外围开销拖慢？ | 明确 cuVS baseline 是“按 UNG group 逐组调用”的 workload-level baseline；同时保留 SGEMM+topK 作为 library kernel baseline |
| SGEMM+topK 是否启用 TF32？ | 报告 cuBLAS math mode；如果是 TF32，应在表中标注；如果不是，必须补 TF32 baseline |
| fused 的加速来自 kernel 还是外围？ | 分开列 `prepare_all`、resident cross、H2D event、kernel event、D2H event、host/writeback |

### Q3: 自研 groupgemm+topK 的贡献是否只是工程优化？

审稿人会攻击：如果 SGEMM+topK 只慢 `1.64x`，创新是否足够？我们的回答不能是“kernel 更快”，而应该是“面向 irregular group workload 的系统级 fusion”。

必须补：

| Ablation | 目的 |
|---|---|
| cuVS per-group vs SGEMM+topK vs old fused vs final fused | 证明逐步减少 group API、pack、query 展开、writeback 后的收益 |
| final fused 关闭 direct-qid / id-only writeback / resident base vectors | 证明每个设计点贡献 |
| 按 `(nq,nx)` bucket 汇总耗时和 FLOPs | 说明甜点区在哪，哪些 bucket 应该 route 到 cuBLAS |
| x40/x100/x200/x400 + 10%x40 | 证明小组多/中组多/大组多时的边界 |

### Q4: x400 direct-qid 失败是否影响论文可信度？

会被问：为什么一个核心优化在 x400 失败？是否有越界 bug？因此投稿版有两种选择：

| 路线 | 要求 |
|---|---|
| 修复 direct-qid | x400 direct-qid strict 模式通过，并补 cuda-memcheck / sanitizer 或最小复现说明 |
| 作为 limitation | 主表 x400 只用 gather-Q，direct-qid 只在稳定规模报告；正文明确 direct-qid large-scale kernel 是 future work |

### Q5: 结果是否只对 Amazon sampled+jitter-repeat 成立？

当前主结果过度依赖 Amazon 1% repeat 系列。这个数据集有价值，因为它制造了可控 group size，但论文必须证明不是只针对这个构造。

必须补：

| 数据集 | 目的 | 最少实验 |
|---|---|---|
| SIFT/SIFT30 | 通用向量分布，较弱标签结构 | 已补 group graph end-to-end A/B；还需修 fused cross strict 或明确 fallback |
| CelebA 或真实多标签 | 多标签过滤结构 | cross-edge 和 end-to-end recall |
| Amazon 10% x40 | 组数足够多的小/中组压力 | build breakdown + recall |

### Q6: group graph 的 FastGrnndCuda 是否真能替代 Vamana？

单组 graph recall 不能完全说明 UNG search 质量，因为 UNG 查询还受 label tree、entry points、cross edges 和 search beam 影响。

必须补：

| 实验 | 解释力 |
|---|---|
| 单组 graph recall vs end-to-end recall 相关性 | 证明单组 benchmark 是否可作为代理指标 |
| FastGrnndCuda 参数扫描 `iter/K/M/min_group_size` | 给出速度/质量 Pareto，而不是只报一个点 |
| group degree/connectivity 分布 | 防止局部剪枝造成孤立点或连通性退化 |
| fallback policy ablation: CPU fallback vs complete fallback vs all FastGrnndCuda | 说明 hybrid 策略不是偷偷换了更容易的图 |
| batched exact kNN negative ablation | 证明 GPU core 快不等于 Vamana-style 可导航图；限制论文主张边界 |

## 1. 方法边界

### 1.1 CPU Baselines

| 方法 | 说明 | 论文用途 |
|---|---|---|
| Original/CPU Vamana cross | 原始 cross-edge 用 Vamana 近似搜索 | 原始慢路径 baseline |
| CPU exact scan | 对每个 `(query group, target group)` 精确扫描 topK | 强 CPU baseline |
| CPU Vamana group graph | 组内图原始/当前 CPU Vamana | 质量 baseline |

### 1.2 Naive GPU Baselines

| 方法 | 说明 | 主要问题 |
|---|---|---|
| cuVS per group | 每个 group 调一次 cuVS brute-force | 逐 group API、pack、H2D、D2H 外围成本高 |
| SGEMM + separate topK | cuBLAS/cuBLASLt GEMM 后单独 topK | kernel 强，但不是 workload-level fused |
| old fused | 早期 fused topK | 减少部分外围，但仍有 query 展开和写回成本 |

### 1.3 当前优化方法

| 部分 | 当前实现 | 是否论文主贡献 |
|---|---|---|
| cross-edge | grouped fused topK, direct-qid, TF32 WMMA group kernel, safe id-only writeback after GPU global merge | 是 |
| group graph | FastGrnndCuda: Tagore GNN candidates + reverse-augmented local RNG-style prune | 是 |
| small-group fallback | CPU Vamana fallback 或 complete fallback | 工程策略，需要和主方法分开写 |

### 1.4 Group Graph 算法实现细节

当前组内 PG 构建不是单一 GPU kernel，而是 **workload-aware hybrid builder**。代码入口是 `UniNavGraph::build_graph_for_all_groups_tagore_cuda()`。它先按每个 group 的 `nx` 静态分桶，再分别调用 fallback、packed exact-anchor 或 FastGrnndCuda。关键实现位置：

```text
UNG/codes/src/uni_nav_graph.cpp:1436-1465   判断是否 AdaptiveCuda，并把 group 划入 fallback/GPU batch
UNG/codes/src/uni_nav_graph.cpp:1476-1503   fallback: complete/bounded-complete 或 CPU Vamana
UNG/codes/src/uni_nav_graph.cpp:1511-1532   CPU fallback 与 GPU batch 并发
UNG/codes/src/uni_nav_graph.cpp:1561-1624   GPU batch 再拆成 exact batch 与 GNN batch
```

顶层路由语义是：

| group 类型 | 触发条件 | 构建算法 | 算法含义 |
|---|---|---|---|
| very small / fallback | `nx <= complete_threshold` 或 `nx < tagore_min_group_size` | complete、bounded-complete 或 CPU Vamana | 小组 GPU 固定开销不划算；bounded-complete 在 `nx > max_degree + 1` 时每点只连环上后 `max_degree` 个点，保持有界出度和基本强连通 |
| small-to-medium | `nx <= exact_batch_threshold` | packed exact-anchor | 对每个点 exact 扫同组所有点维护 topK；输出不是纯最近邻，而是 nearest head + anchor tail + sampled tail |
| large / quality-sensitive | 其余 GPU group | GNN-Descent + FastGrnnd prune | 复用 Tagore GPU GNN-Descent 产生候选，跳过 Tagore 重 `select_path + filter_reverse`，改用轻量/反向/RNG-style 本地剪枝 |

packed exact-anchor 的核心实现是 `build_fast_exact_cuda_batch()` 和 `batched_exact_knn_graph_kernel()`：

```text
UNG/codes/src/tagore_graph_builder.cu:969-1086   多 group packed offsets/sizes，direct-H2D run 合并，device lookup 配置
UNG/codes/src/tagore_graph_builder.cu:1153-1197  H2D 与 GPU point->group/local lookup
UNG/codes/src/tagore_graph_builder.cu:1199-1218  exact graph kernel launch
UNG/codes/src/tagore_graph_builder.cu:699-731    每个 source 扫同 group dst，block reduce L2，维护 topK
UNG/codes/src/tagore_graph_builder.cu:735-790    输出 nearest head、anchor tail、sampled tail
```

exact-anchor 的“anchor”不是 Vamana entry point，而是对每个 source 在 topK 近邻之外按环形 hop 选择少量分散点：

```text
candidate = (local_src + hop) % n
```

因此 exact-anchor 图不是纯 kNN 图。它先保留 `head_keep` 个最近邻，再用环形 anchor 和 topK 尾部等距采样补充多样性，目标是避免图过度局部、改善导航性。

FastGrnndCuda 的候选生成仍来自 Tagore GNN-Descent。batch path 中依次执行初始化、正/反向 sample、距离计算、merge、reverse graph 和二阶段采样合并；随后进入我们自己的 prune：

```text
UNG/codes/src/tagore_graph_builder.cu:1848-1900  GNN-Descent 候选生成
UNG/codes/src/tagore_graph_builder.cu:1350-1444  根据 nx 选择 light prune / reverse-tail / heavy prune
```

FastGrnndCuda 有三种剪枝形态：

| 剪枝形态 | 代码位置 | 算法 |
|---|---|---|
| light prune | `tagore_graph_builder.cu:396-500` | 直接从 GNN 候选保留 nearest head；再从候选尾部等距采样补多样性；不足时用环形 repair 补足低出度 |
| light reverse-tail | `tagore_graph_builder.cu:502-632` | 先用 sampled reverse 收集“指向当前点”的反向候选；输出顺序为 nearest head、少量 reverse slots、GNN tail、repair |
| heavy FastGrnnd prune | `tagore_graph_builder.cu:286-394` | 合并 forward GNN 候选与 sampled reverse 候选，重算距离并排序，然后用 alpha occlusion 逐个接受/拒绝 |

heavy prune 的 occlusion 规则是：

```text
若已有邻居 pivot 满足 alpha * dist(pivot, dst) < dist(src, dst)，则拒绝 dst。
```

这比 light prune 更接近 Vamana/RNG 风格剪枝，但在 `nx≈200` 这类中等组上成本无法摊销；因此中等组优先 exact-anchor 或 light prune，大组/质量敏感组才使用 reverse-tail 或 heavy prune。

最后，GPU 生成的边仍要回填到 CPU-compatible `Graph::neighbors`。exact batch 支持 packed graph buffer：D2H 后不再先 scatter 到每组临时 `result.graph`，而是在 `UniNavGraph` fill 阶段按 packed offset 直接读：

```text
UNG/codes/src/uni_nav_graph.cpp:1685-1715
```

这减少了一次 host copy，但仍保留 `Graph::neighbors` / Vamana-compatible 输出边界。因此本文不能把当前实现写成 flat/CSR 或 CPU Vamana 的无条件替代。

当前 `UNG_GPU_TOPK_IMPL=3` 的 `FusedGroupTopk` 会默认打开：

```bash
UNG_FORCE_CUSTOM_KERNEL=1
UNG_SMALL_GROUP_FUSED=1
UNG_MEDIUM_GROUP_FUSED=1
UNG_MEDIUM_GROUP_MAX_NX=4096
UNG_DIRECT_QID_FUSED=1
UNG_DIRECT_QID_ALL_FUSED=1
UNG_GPU_ID_ONLY_WRITEBACK=1
UNG_NAIVE_HEAVY_SGEMM=0
UNG_BUCKET_GROUP_FUSED=1
UNG_LARGE_GROUP_FUSED=1
UNG_LARGE_GROUP_FUSED_MODE=2
UNG_LARGE_GROUP_MIN_NX=32
UNG_LARGE_GROUP_MAX_NX=128
UNG_LARGE_GROUP_WARPS=4
UNG_TF32_GROUP_2D=1
```

## 2. 最新重测配置

build-scale 性能重测默认使用：

```bash
build_mode_switch/apps/build_UNG_index
--data_type float --dist_fn L2 --num_threads 128
--max_degree 32 --Lbuild 100 --alpha 1.2 --num_cross_edges 6
UNG_BUILD_PROFILE=custom
UNG_GROUP_GRAPH_IMPL=3
UNG_TAGORE_K=64
UNG_TAGORE_ITER=4
UNG_TAGORE_M=64
UNG_GET_MIN_SUPER_SETS_IMPL=0
UNG_LNG_IMPL=0
UNG_DESCENDANTS_IMPL=0
UNG_COVERAGE_IMPL=1
UNG_COVERAGE_THREADS=128
UNG_CROSS_EDGE_IMPL=1
UNG_ADDITIONAL_EDGES_IMPL=1
UNG_GPU_TOPK_IMPL=3
UNG_CROSS_EDGE_GPU_STRICT=1
```

注意：`UNG_ADDITIONAL_EDGES_IMPL=1` 在当前代码中表示 **skip additional edges**，只能用于阶段性能拆分。所有端到端 recall A/B 的完整质量配置使用 `UNG_ADDITIONAL_EDGES_IMPL=0`，即 CPU Vamana additional edges。报告中凡是讨论查询质量的表，都必须明确 additional edges 口径。

`UNG_TAGORE_MIN_GROUP_SIZE`：

| 数据集 | min group |
|---|---:|
| Amazon 1% x40 | `256` |
| Amazon 1% x100 | `128` |
| Amazon 1% x200 | `256` |
| Amazon 1% x400 | `256` |

说明：本机没有 `gpulock`，重测前确认 GPU0 空闲，使用 `CUDA_VISIBLE_DEVICES=0` 顺序运行。

## 3. Cross-Edge 主对比

Amazon 1% x100，除特别说明外只比较 cross-edge 阶段。

| 方法 | cross total | H2D | kernel/build | D2H | Index | 说明 |
|---|---:|---:|---:|---:|---:|---|
| CPU Vamana cross | `18735.8 ms` | - | - | - | `23150 ms` | 原始慢路径 |
| CPU exact scan | `2389.8 ms` | - | - | - | `6995 ms` | 强 CPU baseline |
| cuVS per group | `3730.5 ms` | `592.1` | `821.8` | `43.5` | `14353 ms` | 逐 group 调库 |
| SGEMM + separate topK | `1395.9 ms` | `108.4` | `541.6` | `9.9` | `7452 ms` | 强 GPU library baseline |
| old fused baseline | `1268.5 ms` | `103.9` | `528.7` | `7.8` | `7388 ms` | direct-qid 前 |
| final direct-qid fused, CPU group | `1313.1 ms` | `76.0` | `227.0` | `5.8` | `14053 ms` | 只重测 cross 默认 |
| final direct-qid fused, mixed group | `848.9 ms` | `74.1` | `232.8` | `3.6` | `6630 ms` | 最新 full optimized run |

使用最新 full optimized x100 `848.9 ms` 计算：

| baseline | 加速比 |
|---|---:|
| vs CPU Vamana cross | `22.07x` |
| vs CPU exact scan | `2.81x` |
| vs cuVS per group | `4.39x` |
| vs SGEMM + topK | `1.64x` |

`848.9 ms` 的细分：

| 项目 | 时间 |
|---|---:|
| prepare_all | `218.714 ms` |
| resident cross | `630.186 ms` |
| GPU H2D events | `74.1 ms` |
| GPU kernel events | `232.8 ms` |
| GPU D2H events | `3.6 ms` |

### 3.5 Amazon 10% x40 Direct-QID 修复与 Full-Quality A/B

`Amazon 10% x40` 用于检验大量 target groups 场景：`2,409,800` 个点，`53,840` 个 group，cross-edge target groups 为 `49,374`。第一轮 build 在 cross-edge 阶段超过 10 分钟没有进入 GPU event 输出。修复后，skip-additional 口径下得到下面结果：

| Variant | Index | Group graph | Cross-edge | prepare_all | H2D events | kernel events | D2H events | L100 sanity |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| CPU exact cross, CPU Vamana group | `89529.3 ms` | `26083.3 ms` | `54879.3 ms` | - | - | - | - | - |
| GPU direct-qid cross, CPU Vamana group | `42651.1 ms` | `22747.6 ms` | `8475.4 ms` | `2596.3` | `2534.0` | `2669.4` | `83.1` | `0.800` |
| GPU direct-qid cross, FastGrnndCuda + complete fallback | `26422.1 ms` | `7689.0 ms` | `8869.1 ms` | `3222.4` | `3297.0` | `2691.5` | `84.7` | `0.800` |

加速比：

| 对比 | Speedup |
|---|---:|
| GPU direct-qid cross vs CPU exact cross | `6.47x` |
| CPU Vamana group + GPU cross vs CPU exact baseline Index | `2.10x` |
| FastGrnndCuda + GPU cross vs CPU exact baseline Index | `3.39x` |
| FastGrnndCuda group graph vs CPU Vamana group graph | `3.39x` |

这组结果说明 10%x40 的核心问题不是 fused kernel 算不动，而是系统外围。旧 path 的 `prepare_all` 实际走 `pinned_staging`，`host_pack_ms=6205.6`，总 `6500.7 ms`；修复后直接从连续 base storage 做 pageable H2D，`host_pack_ms=0`，总 `2596.3 ms`。此外，double-buffer path 不再为每个 chunk materialize `d_Q` 和 `q_norm`，而是只上传 `qid` 与 group descriptors，由 medium/TF32 fused kernel 直接读取全量常驻 `g_d_all_X/g_d_all_norm`。

边界：这三组都设置 `UNG_ADDITIONAL_EDGES_IMPL=1`，即跳过 additional edges。L100 `0.800` 只能说明 search 没有明显崩溃，不能作为完整质量结果。正式论文必须补 `UNG_ADDITIONAL_EDGES_IMPL=0` 或等价 additional-edge backend 的 full-quality run。

随后补跑 full-quality A/B，固定 `UNG_ADDITIONAL_EDGES_IMPL=0`，也就是 CPU Vamana additional edges：

```text
/home/graphdb/fv_runs/reviewer_e2e_amazon_10pct_x40_cpu_group_directqid_fulladd_20260602_005156
/home/graphdb/fv_runs/reviewer_e2e_amazon_10pct_x40_fastgrnnd_directqid_fulladd_20260602_005446
```

| Variant | Index | Group graph | Cross-edge | prepare_all | GPU kernel events | L100 | L500 | L1000 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| CPU Vamana group + GPU direct-qid cross + CPU additional_edges | `36053.3 ms` | `18218.5 ms` | `9777.3 ms` | `2740.8` | `2925.7` | `0.8647` | `0.8980` | `0.9080` |
| FastGrnndCuda + complete fallback + GPU direct-qid cross + CPU additional_edges | `23497.4 ms` | `5147.2 ms` | `9451.1 ms` | `2206.5` | `2952.2` | `0.8668` | `0.8980` | `0.9103` |

full-quality 结论：

- Index speedup: `1.53x`。
- Group graph speedup: `3.54x`。
- Cross-edge 基本相同，`1.03x`，因此该 A/B 主要检验 group graph 替换。
- L100/L500/L1000 recall 没有下降，反而分别为 `+0.0021 / 0 / +0.0023`。这比 skip-additional L100 `0.800` 更能说明完整系统质量。

repeat=3 per-query latency 分位数来自 `query_details_repeat3.csv` 的 `Time_ms`，不同于整批 1000 query 的 `Average_Time_ms`：

| Variant | Lsearch | Avg batch ms | P50 query ms | P95 query ms | P99 query ms | recall=0 ratio |
|---|---:|---:|---:|---:|---:|---:|
| CPU Vamana group | 100 | `791.6` | `48.6` | `254.8` | `499.3` | `13.53%` |
| FastGrnndCuda | 100 | `728.6` | `45.5` | `226.4` | `458.3` | `13.30%` |
| CPU Vamana group | 500 | `705.2` | `41.1` | `244.5` | `515.3` | `10.20%` |
| FastGrnndCuda | 500 | `656.9` | `41.9` | `201.2` | `430.7` | `10.20%` |
| CPU Vamana group | 1000 | `735.9` | `41.8` | `244.2` | `503.6` | `9.20%` |
| FastGrnndCuda | 1000 | `694.7` | `41.5` | `231.3` | `494.0` | `8.97%` |

## 4. End-to-End Scale

主表使用当前最快工程配置：FastGrnndCuda 处理大组，默认 CPU Vamana fallback 处理小于 `UNG_TAGORE_MIN_GROUP_SIZE` 的组。x400 由于 direct-qid 失败，使用稳定 gather-Q fused 路径。

| 数据集 | 点数 | group graph | vector_attr | LNG | desc | coverage | cross | Index |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1% x40 | 240,960 | `868.5` | `126.4` | `98.3` | `10.5` | `57.9` | `389.8` | `2375` |
| 1% x100 | 602,400 | `3373.2` | `209.9` | `43.7` | `3.9` | `12.4` | `848.9` | `6630` |
| 1% x200 | 1,204,800 | `8630.8` | `410.2` | `37.4` | `2.0` | `18.6` | `3679.5` | `16426` |
| 1% x400 | 2,409,600 | `31245.6` | `777.0` | `65.0` | `19.3` | `211.0` | `12462.8` | `50562` |

相对旧 CPU Vamana group + GPU fused cross-edge baseline：

| 数据集 | baseline Index | 当前 Index | Speedup | baseline group | 当前 group | group speedup |
|---|---:|---:|---:|---:|---:|---:|
| 1% x40 | `2484` | `2375` | `1.05x` | `891.0` | `868.5` | `1.03x` |
| 1% x100 | `16824` | `6630` | `2.54x` | `11549.9` | `3373.2` | `3.42x` |
| 1% x200 | `34603` | `16426` | `2.11x` | `26231.4` | `8630.8` | `3.04x` |
| 1% x400 | `70616` | `50562` | `1.40x` | `54043.2` | `31245.6` | `1.73x` |

## 4.5 End-to-End Recall A/B

这部分专门回答 reviewer 对“只建得快、查询质量是否下降”的质疑。所有结果都使用同一 query、同一 GT、同一 search 参数，只切换 group graph 后端。

### 4.5.1 原始 Amazon PF

日志：

```text
/home/graphdb/fv_runs/reviewer_e2e_amazon_pf_group_ab_addedges_20260601_221044
```

| Variant | Index ms | Group ms | Cross ms | L1000 recall | L2000 recall | L5000 recall |
|---|---:|---:|---:|---:|---:|---:|
| CPU Vamana group | `43376.6` | `5776.25` | `20194.7` | `0.912333` | `0.932067` | `0.971767` |
| FastGrnndCuda + CPU fallback | `36732.0` | `1049.92` | `18847.4` | `0.911700` | `0.931500` | `0.971133` |

结论：group graph 加速 `5.50x`，Index 加速 `1.18x`，三个 Lsearch 的 recall drop 均小于 `0.001`。同一数据集上若跳过 additional_edges，L5000 recall 只有约 `0.616`，因此 `UNG_ADDITIONAL_EDGES_IMPL=1` 只能作为 stage ablation，不能作为完整质量配置。

### 4.5.2 SIFT30

日志：

```text
/home/graphdb/fv_runs/debug_sift30_fused_cross_exactdist_cpu_20260601_224812
/home/graphdb/fv_runs/debug_sift30_fused_cross_exactdist_fast_20260601_225006
```

配置：`1,000,000` base vectors，`69,176` groups，`10,000` containment queries；固定修复后的 `UNG_CROSS_EDGE_IMPL=1` / `UNG_GPU_TOPK_IMPL=3` 和 `UNG_ADDITIONAL_EDGES_IMPL=0`，只切换 group graph。

| Variant | Index ms | Group ms | Cross ms | L100 recall | L200 recall | L500 recall | L1000 recall |
|---|---:|---:|---:|---:|---:|---:|---:|
| CPU Vamana group | `31740.4` | `11245.3` | `17886.2` | `0.609423` | `0.706580` | `0.814600` | `0.877587` |
| FastGrnndCuda + CPU fallback | `24989.9` | `1889.79` | `17868.9` | `0.607997` | `0.706287` | `0.812680` | `0.876297` |

结论：group graph 加速 `5.95x`，Index 加速 `1.27x`，L1000 recall drop 约 `0.00129`。修复后的 fused cross-edge 将 SIFT30 cross 从 CPU Vamana cross 的 `69834.5 ms` 降到 `17886.2 ms`，为 `3.90x`，且 CPU Vamana group 的 recall 与 CPU cross 基本一致。

### 4.5.3 Amazon 1% x200 Coverage Query

这组实验是 reviewer 要求的 repeat-scale 端到端验证。它先暴露了旧 FastGrnndCuda 的失败点，随后验证了 light prune 能把 `nx≈200` 从性能反例拉回到可用区间。query 由 `generate_coverage_queries.py` 从 x200 索引结构中生成，不是单组 query：

| Query profile | 值 |
|---|---:|
| queries | `1000` |
| profile counts | broad `316`, medium `420`, narrow `264` |
| matched groups avg / p50 / p95 | `258.97 / 32 / 1087.4` |
| matched points avg / p50 / p95 | `55556.6 / 6800 / 244500` |
| GT compute time | `8733 ms` |

日志：

```text
/home/graphdb/fv_runs/reviewer_e2e_amazon_1pct_x200_coverage_20260601_230224
/home/graphdb/FilterVectorBenchData/fv_amazon_1pct_x200_jitter_nonempty/query_coverage_1000
/home/graphdb/FilterVectorBenchData/fv_amazon_1pct_x200_jitter_nonempty/query_coverage_1000_gt/gt_K10_containment.bin
```

固定 `UNG_CROSS_EDGE_IMPL=1`、`UNG_GPU_TOPK_IMPL=3`、`UNG_ADDITIONAL_EDGES_IMPL=0`，只切换 group graph：

| Variant | Index ms | Group ms | Cross ms | L100 recall | L200 recall | L500 recall | L1000 recall |
|---|---:|---:|---:|---:|---:|---:|---:|
| CPU Vamana group | `17596.8` | `8511.72` | `4601.91` | `0.8230` | `0.8250` | `0.8490` | `0.8690` |
| FastGrnndCuda, old heavy prune | `38273.7` | `26620.6` | `6055.72` | `0.8125` | `0.8273` | `0.8472` | `0.8662` |
| FastGrnndCuda, top32 light prune | `14211.8` | `6016.04` | `4926.37` | `0.8135` | `0.8239` | `0.8421` | `0.8618` |
| FastGrnndCuda, diversified light prune | `16478.5` | `6706.36` | `4857.33` | - | - | - | `0.8656` |
| old batched exact kNN, confounded | `12595.9` | `4356.18` | `4120.75` | - | - | - | `0.8159` |
| old batched exact + anchor, confounded | `12539.3` | `5534.10` | - | - | - | - | `0.8240` |
| packed exact-anchor router, nx4096 | `11043.5` | `3827.28` | `4868.15` | - | - | - | `0.8690` |

FastGrnndCuda 的详细组成：

| Variant | direct build | H2D | GNN | prune | D2H | fill |
|---|---:|---:|---:|---:|---:|---:|
| old heavy prune | `26282 ms` | `1538.32` | `3551.54` | `19218.3` | `413.047` | `318.071` |
| top32 light prune | `5746.81 ms` | `1180.47` | `3198.35` | `712.392` | `134.631` | `247.246` |
| diversified light prune | `6365.69 ms` | `1448.82` | `3229.56` | `759.66` | `161.408` | `317.508` |
| old batched exact kNN, confounded | `3950.64 ms` | `524.173` | `674.693` | `0` | `29.8523` | `381.630` |
| old batched exact + anchor, confounded | - | - | `935.249` | `0` | - | - |
| packed exact-anchor router, nx4096 | `2899.6 ms` | `347.710` | `925.721` | `0` | `29.742` | `902.017` |

高 Lsearch 复测：

| Variant | L1000 | L2000 | L5000 |
|---|---:|---:|---:|
| CPU Vamana group | `0.8690` | `0.8820` | `0.9080` |
| FastGrnndCuda, top32 light prune | `0.8619` | `0.8769` | `0.9029` |
| FastGrnndCuda, diversified light prune | `0.8656` | - | `0.9056` |
| old batched exact kNN, confounded | `0.8159` | - | `0.8199` |
| old batched exact + anchor, confounded | `0.8240` | - | `0.8290` |
| packed exact-anchor router, nx4096 | `0.8690` | - | `0.9080` |

结论：

- `nx≈200` 的 Amazon repeat-scale bucket 上，问题不是 GNN 候选生成，而是旧重 prune：`19218.3 ms`，占旧 FastGrnndCuda group graph 约 `72%`。
- top32 light prune 将 prune 降到 `712.392 ms`，约 `27.0x` 加速；diversified light prune 为 `759.66 ms`，几乎不增加 prune 成本，但 L5000 recall 从 `0.9029` 提到 `0.9056`。
- 与 CPU Vamana 相比，当前默认 diversified light prune 版本 group graph `1.27x`、Index `1.07x` 更快，L1000 recall 低 `0.0034`，L5000 低 `0.0024`。
- 旧 batched exact kNN 低 recall 是重要实验教训，但不是组内 exact 图不可行的证据。新结构诊断显示旧 exact-anchor 与新 packed exact 的组内图完全一致，差异来自 cross edges：旧 `1,038,000`，CPU/new packed `1,093,224`。
- packed exact-anchor router 是当前最强正结果：通过去掉 D2H 后按组 scatter 再 fill 的重复 host copy，x200 group graph 从旧 scatter exact run `4929.54 ms` 降到 `3827.28 ms`，对 CPU Vamana group `8511.72 ms` 为 `2.22x`；repeat=3 L5000 为 `0.9080`，与 CPU baseline 对齐。
- 因此更接近“普遍替代”的设计从“只做 GNN/reverse pruning”转为“full-quality packed exact-anchor 处理可覆盖的中小组，大组或真实复杂分布再由 reverse-tail/GNN 路由兜底”。后续重点是 x100、10%x40 和真实多标签上的泛化边界。
- 这组实验的保存索引时间另计：CPU run `Index saved in 21533.9 ms`，old Fast run `19581.3 ms`，top32 light prune run `20577.7 ms`，diversified light prune run `23396.3 ms`。报告 Index time 不含落盘，artifact 复现实验若测 wall time 必须把保存时间单独列出。

## 5. Group Graph 结果

### 5.1 当前最快 fallback

| 数据集 | Tagore groups | CPU fallback groups | complete fallback groups | direct build | fallback wall | GNN | prune | group total |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1% x40 | `16` | `112` | `5538` | `229.4` | `636.6` | `14.5` | `38.8` | `868.5` |
| 1% x100 | `128` | `5538` | `0` | `540.0` | `2822.2` | `106.3` | `265.8` | `3373.2` |
| 1% x200 | `128` | `5538` | `0` | `961.1` | `7648.4` | `174.4` | `557.4` | `8630.8` |
| 1% x400 | `5666` | `0` | `0` | `30812.5` | `11.0` | `4940.1` | `20021.9` | `31245.6` |

观察：

- x100/x200 的最快端到端结果依赖 CPU Vamana fallback。它是工程上最快的混合方案，但论文写法里要和 FastGrnndCuda 主方法分开。
- x400 所有组都进入 FastGrnndCuda，`prune` 是最大瓶颈，约 `20.0 s`。
- x40 太小，FastGrnndCuda 只覆盖 16 个大组，整体收益有限。

### 5.2 Complete Fallback Ablation

此前 `UNG_TAGORE_FALLBACK_IMPL=0` 的 complete fallback 结果如下，保留作为 ablation：

| 数据集 | group | cross | Index | 判断 |
|---|---:|---:|---:|---|
| 1% x40 complete fallback | `849.6` | `505.3` | `2442` | 与 CPU fallback 接近 |
| 1% x100 complete fallback | `4533.4` | `1257.4` | `8179` | 慢于 CPU fallback |
| 1% x200 complete fallback | `20021.6` | `2788.9` | `26642` | fallback complete 图过重 |
| 1% x400 complete fallback | `31074.6` | `6722.0` | `44713` | 无 fallback，差异主要来自 cross 配置 |
| 10% x40 complete fallback | `7225.2` | `6201.2` | `23207` | group 数足够多时收益明显 |

## 6. x400 Direct-Qid 失败

补测结果：

| 配置 | 结果 |
|---|---|
| direct-qid fused, default cap | `medium group_desc launch failed`, fallback CPU，cross `86359.0 ms` |
| direct-qid fused, cap 512 MB | strict 模式失败，`illegal memory access` |
| gather-Q fused | 成功，cross `12462.8 ms` |

对应日志：

```text
/home/graphdb/FilterVectorBenchResults/final_report_retest_20260528/x1p400_final_opt256/run.log
/home/graphdb/FilterVectorBenchResults/final_report_retest_20260528/x1p400_final_opt256_cap512/run.log
/home/graphdb/FilterVectorBenchResults/final_report_retest_20260528/x1p400_final_opt256_gatherq/run.log
```

判断：x400 失败不是 chunk 太大导致，因为 `UNG_GPU_FLAT_Q_CAP_MB=512` 后每 chunk 约 `172K` queries 仍失败。更可能是 direct-qid medium group-desc kernel 在大规模 qid 范围下存在越界或状态复用问题。最终报告中 x400 不应使用 direct-qid fallback CPU 的 `86359.0 ms` 作为 GPU 性能。

## 7. prepare_all 口径

`gpu_prepare_all_vectors_on_device()` 做三件事：

1. 上传全量 base vectors 到 `g_d_all_X`。
2. 计算全量 `g_d_all_norm`。
3. 对连续 host base vectors 默认使用 `cudaHostRegister` 后 H2D。

最新重测：

| 数据集 | prepare_all | host register | full-data H2D | norm | cross total | resident cross |
|---|---:|---:|---:|---:|---:|---:|
| 1% x40 | `102.805` | `72.472` | `29.851` | `0.062` | `389.8` | `286.995` |
| 1% x100 | `218.714` | `145.269` | `73.085` | `0.053` | `848.9` | `630.186` |
| 1% x200 | `486.287` | `339.767` | `146.104` | `0.058` | `3679.5` | `3193.213` |
| 1% x400 gather-Q | `702.898` | `409.846` | `292.518` | `0.085` | `12462.8` | `11759.902` |

`prepare_all` 波动较大，尤其 host registration。此前 x100 CPU-group 重测中 `prepare_all=540.843 ms`，同一路径本轮为 `218.714 ms`。因此报告中必须把它单独列出来，避免把 host registration 抖动误解释为 kernel 算法差异。

## 8. 当前瓶颈

### 8.1 cross-edge

x100 最新 full optimized：

```text
cross total       = 848.9 ms
prepare_all       = 218.7 ms
resident cross    = 630.2 ms
GPU H2D events    = 74.1 ms
GPU kernel events = 232.8 ms
GPU D2H events    = 3.6 ms
```

剩余非 GPU event 成本包括：

- host registration
- target group / query id 映射
- descriptor 构造和 H2D
- `SearchQueue::insert`
- cross_group_neighbors 到 `_graph->neighbors` 的 merge
- OpenMP scheduling 和 run-to-run 波动

### 8.2 group graph

FastGrnndCuda 的主要瓶颈是 prune：

| 数据集 | GNN | prune | 结论 |
|---|---:|---:|---|
| 1% x100 | `106.3` | `265.8` | 尚可 |
| 1% x200 | `174.4` | `557.4` | GPU direct 部分不慢，fallback 决定总时间 |
| 1% x400 | `4940.1` | `20021.9` | prune 是最大单项瓶颈 |

最新代码改造补充：

- FastGrnnd prune/reverse buffer 从“每个 group 内 `cudaMalloc/cudaFree`”改成 batch 入口按最大 group 一次分配，避免大量 group 场景下 allocator 固定成本被重复放大。
- Tagore batch 主路径从阶段间 `cudaDeviceSynchronize` 改成 CUDA event 计时：H2D、convert、memset、GNN、prune、D2H 仍可分别计时，但 convert/GNN/prune 之间不再强制 host 等待。
- Batch kernel launch 显式绑定同一个 stream，语义仍由 stream 顺序保证；这属于外围调度优化，不改变 GNN-descent / FastGrnnd prune 的候选生成与剪枝规则。
- H2D 带宽统计修正为实际传输的 float bytes；之前按 half bytes 估算会低估有效带宽、误导外围开销分析。

这次改造针对的是“普遍替代”最核心的工程问题：UNG group graph 不是一个单大图，而是大量中小 group。原实现即使 kernel 本身可加速，也会被每组同步、每组分配、每组单独计时拖慢。该改造预计对 group 数多、单 group `nx` 中小的 workload 收益最大；对少量超大 group，收益主要来自去掉 prune workspace 分配，核心 GNN/prune 算法耗时仍需靠 segmented/bucketed multi-group GNN 或新的剪枝策略进一步优化。

## 9. 实验日志索引

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

主要 baselines：

```text
/tmp/fv_amazon_1pct_x100_cpu_vamana/run.log
/tmp/fv_amazon_1pct_x100_cpu_exact/run.log
/home/graphdb/FilterVectorBenchResults/onchip_rewrite_20260527/x1p100_cpu_group_sgemm_topk_baseline/run.log
/home/graphdb/FilterVectorBenchResults/onchip_rewrite_20260527/x1p100_cpu_group_cuvs_pergroup_baseline/run.log
/home/graphdb/FilterVectorBenchResults/onchip_rewrite_20260527/x100_hybrid_default/run.log
```

Complete fallback ablation：

```text
/home/graphdb/FilterVectorBenchResults/onchip_rewrite_20260527/x1p40_fastgrnnd_hybrid256_complete_fallback_dimfix/run.log
/home/graphdb/FilterVectorBenchResults/onchip_rewrite_20260527/x1p100_fastgrnnd_hybrid128_complete_fallback_dimfix/run.log
/home/graphdb/FilterVectorBenchResults/onchip_rewrite_20260527/x1p200_fastgrnnd_hybrid256_complete_fallback_dimfix/run.log
/home/graphdb/FilterVectorBenchResults/onchip_rewrite_20260527/x1p400_fastgrnnd_hybrid256_complete_fallback_dimfix/run.log
/home/graphdb/FilterVectorBenchResults/onchip_rewrite_20260527/x10p40_fastgrnnd_hybrid256_complete_fallback_dimfix/run.log
```

Reviewer-facing 汇总工具：

```text
tools/benchmarks/summarize_cross_baselines.py
tools/benchmarks/diagnose_ung_graph_structure.py
tools/benchmarks/summarize_graph_diagnostics.py
tools/benchmarks/summarize_x400_ab.py
```

当前已生成的汇总 artifact：

```text
/home/graphdb/fv_runs/baseline_fairness_x100_20260602/cross_baselines.md
/home/graphdb/fv_runs/graph_diagnostics_x400_20260602/summary_table.md
```

x400 repair / reverse-tail 下一轮实验会生成：

```text
/home/graphdb/fv_runs/x400_reverse_tail_ab_20260602_030908/x400_ab_summary.md
/home/graphdb/fv_runs/x400_reverse_tail_ab_20260602_030908/x400_ab_summary.csv
```

其中 `x400_ab_summary.md` 同时包含 build time、L1000/L5000 recall、P95/P99 query latency 和 graph diagnostics。当前结果支持把 reverse-tail+repair 写成 x400 质量增强 Pareto 点；`UNG_TAGORE_COMPACT_D2H=1` 仍需要 `RUN_COMPACT_ABLATION=1` 单独证明其外围收益，不能写成质量贡献。

## 10. 最终可写表述

可以写：

- UNG cross-edge 是大量 group-level `(nq, nx, dim) -> topK` workload，不是一个单一大 GEMM。逐 group 调 cuVS 会被 API、pack、H2D、D2H 和 writeback 外围吞掉。
- 我们用 direct-qid 常驻数据、group batching、TF32 WMMA fused topK 和安全的写回策略，把 Amazon 1% x100 cross-edge 降到 `848.9 ms`。
- 相对 CPU Vamana cross-edge 为 `22.07x`，相对 CPU exact scan 为 `2.81x`，相对 cuVS per-group 为 `4.39x`，相对 SGEMM+topK 为 `1.64x`。
- FastGrnndCuda 在大组上替换 CPU Vamana；配合 CPU fallback 后，Amazon 1% x100 Index time 从 `16824 ms` 降到 `6630 ms`。
- 2026-06-02 新增 `adaptive_cuda` group-graph backend：小组保留 CPU fallback，中组 packed exact-anchor，大组 FastGrnndCuda，并让 CPU fallback 与 GPU batch 并发。Amazon 1% x100 full-quality 配置下，Index time 从历史 FastGrnnd full-quality `6370.67 ms` 降到 `5450 ms`，group graph 从 `3216.21 ms` 降到 `2773.35 ms`；L100/L500/L1000 recall 为 `0.829/0.870/0.895`，与历史 `0.828/0.871/0.896` 基本一致。

不能夸大：

- 不能说自研 fused kernel 在所有 dense GEMM 形状上都比 cuBLAS/cuVS 快。
- 不能忽略 `prepare_all`；单次 build 必须计入，常驻后性能只能作为补充口径。
- FastGrnndCuda 不是 CPU Vamana 的严格等价替代。当前已有原始 Amazon PF、SIFT30、Amazon 1% x100、Amazon 1% x200、Amazon 10% x40 和 Amazon 1% x400 gather-Q 的端到端 search recall 表；10%x40 宽 Lsearch sweep 已补，最终论文还需要补真实多标签以及更多 x400 参数扫；compact-D2H 和同脚本 CPU baseline 已补。
- x400 direct-qid 目前不稳定；大规模主表要用 gather-Q fused 或先修复 direct-qid kernel。
- `UNG_TAGORE_FALLBACK_IMPL=0` 的 bounded-complete 小组替代只能作为 ablation：Amazon 1% x100 skip-additional build 中 group graph 可降到 `1953.07 ms`，但 skip-additional L100/L500/L1000 recall 只有 `0.816/0.823/0.827`，不能作为 full-quality 主线。
- 当前实现仍保留 CPU-compatible output boundary：GPU group graph 和 cross-edge 结果最终要写回 host `std::vector<IdxType>` / `SearchQueue` / Vamana-compatible structures。它是下一轮大幅优化的主要目标，不能写成已经被 GPU 完整替代。

## 11. 2026-06-02 adaptive_cuda 普遍替代尝试

代码开关：

```text
UNG_GROUP_GRAPH_IMPL=4              # adaptive_cuda
UNG_TAGORE_MIN_GROUP_SIZE=128       # adaptive 默认值；小于该值走 fallback
UNG_ADAPTIVE_EXACT_MAX_NX=512       # 128..512 走 packed exact-anchor
UNG_TAGORE_OVERLAP_FALLBACK=1       # adaptive 默认开启；CPU fallback 与 GPU batch 并发
UNG_TAGORE_FALLBACK_THREADS=<N>     # 可选，默认使用 build num_threads
UNG_TAGORE_FALLBACK_IMPL=0          # 激进 ablation：fallback 改 bounded-complete
```

Amazon 1% x100 full-quality conservative adaptive：

```text
/home/graphdb/fv_runs/adaptive_cuda_x100_fulladd_min128_exact512_20260602_042853/build.log
/home/graphdb/fv_runs/adaptive_cuda_x100_fulladd_min128_exact512_20260602_042853/search_eval/search_time_summary.csv
```

| 配置 | Index time | group graph | cross/additional | L100 | L500 | L1000 |
|---|---:|---:|---:|---:|---:|---:|
| 历史 FastGrnnd full-quality | `6370.67 ms` | `3216.21 ms` | `1871.51 ms` | `0.828` | `0.871` | `0.896` |
| adaptive conservative full-quality | `5450 ms` | `2773.35 ms` | `1513.7 ms` | `0.829` | `0.870` | `0.895` |

本轮结论：

- 当前性能不佳的根因不是单纯 GPU core 慢，而是大量小组造成的 host object fill、fallback Vamana、per-stage 串行调度和 additional_edges 口径混杂。
- 对 x100 coverage-query，保留 CPU fallback 语义并 overlap GPU batch 是质量安全的；它给出 `1.17x` Index 加速和 `1.16x` group-graph 加速。
- 把小组也替换成 bounded-complete 可以把 group graph 进一步降到 `1953.07 ms`，但 recall 明显不足。它说明“小组 CPU Vamana”确实是剩余大头，也说明不能为了普遍替代直接牺牲小组图质量。

## 12. 2026-06-02 输出适配层瓶颈与下一步大幅优化

最近一次针对“普遍替代”的代码审查和小实验给出一个更清晰的系统结论：当前 GPU 路径的核心 kernel 不是唯一瓶颈，CPU-compatible output boundary 已经成为主要 Amdahl 限制之一。

现状：

- group graph：FastGrnndCuda / packed exact-anchor 在 GPU 上生成邻接候选，但最终仍要回填到 `_graph->neighbors` 这样的 `std::vector<IdxType>` 数组，并构造 Vamana-compatible entry point / graph object。
- cross-edge：GPU fused topK 生成 topK id/dist 后，还要经过 host D2H、`cross_group_neighbors[qid].insert(...)`、additional_edges 检查、再 merge 到 `_graph->neighbors`。
- search：主循环本质只需要“给定点 id 取得邻接 span”，但当前存储是 host vector array；这使 GPU 端 flat 输出必须先适配回 CPU 对象。

保留的低风险改动：

- `UniNavGraph::iterate_to_fixed_point` 和 `Vamana::iterate_to_fixed_point` 已改为按引用遍历邻接表，避免每扩展一个点都复制 `_graph->neighbors[cur.id]` 到临时 `std::vector`。这不改变搜索语义，主要减少 query/additional_edges 中的 CPU 拷贝开销。
- GPU cross-edge 成功路径默认延迟释放全量向量 cache：`UNG_GPU_RELEASE_AFTER_CROSS=0` 时不在 cross-edge 计时关键路径里立即 `cudaFree/cudaHostUnregister`，异常路径仍释放；需要还原旧行为可设 `UNG_GPU_RELEASE_AFTER_CROSS=1`。
- additional_edges 的 connected-group 检查保留线程本地 epoch marker，替代每个 group 构造 `unordered_set`。该改动不改变“哪些 out group 已被 cross-edge 连接”的判断，只减少哈希表分配和随机访问；它是外围优化，不作为论文主贡献。
- 严格 GPU + id-vector writeback 实验路径不再无条件初始化 `_num_points` 个 `SearchQueue`。该优化只在 `UNG_GPU_ID_VECTOR_WRITEBACK=1 && UNG_GPU_DB_NOSPLIT=1 && UNG_CROSS_EDGE_GPU_STRICT=1` 下生效，避免影响默认 split double-buffer 语义。

已否定并回退的微优化：

- 给 GPU graph fill 提前预留 cross-edge 容量：x100 上 `tagore_fill` 从历史约 `28.3 ms` 增到 `120 ms+`，原因是每点过度分配 capacity 的成本超过后续 merge 收益。

不应过度解读的微优化：

- epoch marker 和延迟释放能减少适配层开销，但不能改变 Amdahl 主项。x100 full-quality smoke test 显示，CPU Vamana additional_edges 下 `additional_edges_ms=1269.257 ms`，已经大于 GPU kernel event `655.969 ms`。因此下一轮大幅优化必须处理 additional_edges / flat graph 输出，而不是只继续调 fused topK kernel。

### 12.1 x200 exact-router 输出边界 A/B

为了进一步判断“普遍替代”为什么没有出现大幅端到端收益，我们对 x200 `router_exact_nx256` 做了一个小范围反向 A/B。代码改动是把 batched exact 的候选宽度和输出图宽度拆开：计算仍保留 `tagore_k=64` 候选，但默认只 D2H `final_degree+1` 宽度的 compact graph。这样不会改变 exact-anchor 的候选语义，只减少写回的 padding。

实验入口均使用 `scripts/benchmarks/run_group_graph_router_ab.sh`，`RUN_CPU=0 ROUTER_THRESHOLDS=256 NUM_REPEATS=1 LSEARCH_VALUES=1000 WAIT_GPU_IDLE=0`。因为这些 run 不是同一空闲状态下的长重复实验，结论只用于定位瓶颈，不作为主表速度 claim。

| Run | artifact | Index | Group | Cross | pack | H2D | GNN | prune | D2H | fill | L1000 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| old artifact | `/home/graphdb/fv_runs/group_graph_router_ab_streamfill_20260602_030037/router_exact_nx256` | `14098` | `5770.88` | `5002.54` | `1343.56` | `534.903` | `1001` | `652.547` | `91.7672` | `2121.2` | `0.869` |
| compact exact output | `/home/graphdb/fv_runs/router_exact_nx256_compact_20260602_061437` | `12865.6` | `4823.44` | `5547.16` | `1184.44` | `373.025` | `838.748` | `550.644` | `18.9667` | `1576.9` | `0.867` |
| no compact, same code | `/home/graphdb/fv_runs/router_exact_nx256_nocompact_20260602_061534` | `13188.6` | `4693.92` | `5877.52` | `1188.87` | `380.54` | `835.101` | `550.426` | `31.7758` | `1388.15` | `0.867` |
| compact + fill16 + assign | `/home/graphdb/fv_runs/router_exact_nx256_assign_fill16_20260602_062042` | `13127.7` | `4922.44` | `5741.9` | `1176.1` | `515.104` | `842.378` | `552.551` | `27.3735` | `1501.8` | `0.867` |

本轮结论：

1. compact exact output 是低风险内存路径清理：D2H 从旧 artifact 的 `91.77 ms` 降到 `18.97~31.78 ms` 量级。但这部分在 group graph 中占比已经不高，单独做不到端到端大幅加速。
2. `tagore_fill` 仍然在 `1.1~2.3s` 范围内波动，且 `fill_cpu_sum` 可达数十到数百秒级线程累计时间。主要问题不是 GPU exact kernel，而是把 GPU 结果回填到每点一个 `std::vector<IdxType>` 的 CPU 图对象：大量小 vector resize/assign、allocator 争用和 cache miss 会吞掉收益。
3. 降低 `UNG_TAGORE_FILL_THREADS` 到 `16` 能在某些 run 中降低 fill，但会影响 pack/H2D/调度波动；这说明线程数只是权衡，不是根治方案。
4. 因此更大的优化应转向 flat adjacency / CSR：group graph 和 cross-edge 直接输出全局 id 的 flat edge buffer，search 通过 span view 消费，而不是继续把 GPU 结果物化回 Vamana 兼容 `std::vector` 图。

因此，下一轮真正可能带来大幅收益的方向不是继续微调这些 host object，而是引入 GPU-native flat adjacency / CSR 输出路径：

```text
offsets[N + 1] + edges[E] + optional edge_type/source metadata
```

目标设计：

1. group graph GPU backend 直接输出全局 id 的 flat adjacency，避免先写 local graph 再 `add_offset_for_uni_nav_graph`。
2. cross-edge GPU backend 直接 append 到 flat edge buffer，绕过 `SearchQueue -> cross_group_neighbors -> _graph->neighbors` 的二次 host merge。
3. search 增加 `GraphView` / span-like adapter，使旧 `std::vector` graph 和新 flat graph 都能被同一查询循环消费。
4. additional_edges 如果继续保留 CPU Vamana 语义，必须明确这是 full-quality 兼容层；如果改为 GPU exact/additional backend，必须重新做 full-quality recall A/B。

论文写作边界：

- 当前版本可以写成 “workload-aware GPU backend under a CPU-compatible output boundary”。
- 不能写成 “GPU 已完整替代 CPU/Vamana graph construction stack”。
- flat/CSR 是明确 future work 或下一版系统贡献，除非补齐实现、build A/B 和端到端 recall A/B。

### 12.2 additional_edges / release 边界 smoke test

这组实验只用于定位瓶颈，不进入主表：临时 index 输出已清理以释放根分区空间，保留证据是 profiler 日志。它回答 reviewer 会追问的一个问题：为什么 GPU kernel 已经很快，但 full-quality cross-edge 仍然不够快？

| 数据集 / 变体 | profiler | total cross | generate | additional_edges | GPU kernel event | 结论 |
|---|---|---:|---:|---:|---:|---|
| Amazon 1% x10, 默认延迟释放 | `/home/graphdb/Codes/FilterVectorResultsCUDA/prof/ung_prof_20260602_064021_103101.log` | `298.470 ms` | `195.400 ms` | `35.154 ms` | `8.172 ms` | 小负载下 kernel 已很小，主要是 prepare / host 边界 |
| Amazon 1% x10, `UNG_GPU_RELEASE_AFTER_CROSS=1` | `/home/graphdb/Codes/FilterVectorResultsCUDA/prof/ung_prof_20260602_064045_103351.log` | `370.218 ms` | `274.950 ms` | `52.735 ms` | `7.342 ms` | 立刻释放 cache 会把 allocator / unregister 同步放回关键路径 |
| Amazon 1% x100, CPU Vamana additional | `/home/graphdb/Codes/FilterVectorResultsCUDA/prof/ung_prof_20260602_064111_103566.log` | `2439.062 ms` | `876.695 ms` | `1269.257 ms` | `655.969 ms` | full-quality 下 additional_edges 已超过 GPU kernel |
| Amazon 1% x100, CPU exact additional | `/home/graphdb/Codes/FilterVectorResultsCUDA/prof/ung_prof_20260602_064200_103859.log` | `2568.729 ms` | `1155.114 ms` | `207.893 ms` | `658.020 ms` | CPU exact 可降低 additional 本体，但 generate/prepare 波动和全扫成本抵消收益 |

本轮 reviewer 结论：

1. `additional_edges` 不是可以忽略的附属阶段。skip 会破坏 recall，CPU Vamana full-quality 又可能成为比 GPU kernel 更大的单项开销。
2. CPU exact additional 不是简单替代：它降低了 `additional_edges_ms`，但会引入更多 exact scan / prepare / run-to-run 波动，当前 x100 smoke 中 total cross 反而略慢。
3. 延迟释放 GPU cache 是合理的工程默认值，因为 `build_UNG_index` 是单次构建进程，cross-edge 后没有后续必须立即释放显存的阶段；若作为库嵌入长进程，可以用 `UNG_GPU_RELEASE_AFTER_CROSS=1` 恢复旧行为。
4. 更完整的系统级替代路线应把 additional_edges 纳入统一设计：要么实现 GPU exact/additional backend 并做 full-quality recall A/B，要么改变图输出为 flat adjacency / CSR，使 cross-edge 和 additional_edges 都能直接追加到统一 edge buffer，减少 `SearchQueue` 和 `std::vector` 适配层。

## 13. 2026-06-02 qid-lock GPU global merge 负结果

为了进一步回答“能否做一个更普遍的 GPU 替代”，我们尝试把 cross-edge 的输出边界从 host `SearchQueue` 往 GPU 内部推进。直觉上，10%x40 的 fused cross-edge 已经有明显的 D2H/writeback/host merge 成本，似乎可以让每个 chunk 先在 GPU 上合并到全局 qid topK，最后只 D2H 一次全局 id buffer，再直接写 `_graph->neighbors`。

实验路径：

```text
/home/graphdb/fv_runs/db_global_merge_nosplit2_10pct_x40_20260602_050915
```

关键开关：

```bash
UNG_GPU_DB_NOSPLIT=1
UNG_GPU_DB_GLOBAL_MERGE=1
UNG_GPU_ID_VECTOR_WRITEBACK=1
UNG_GPU_DB_LARGE_MAX_NX=16384
```

对照是 10%x40 full-quality 历史有效点：

```text
/home/graphdb/fv_runs/reviewer_e2e_amazon_10pct_x40_fastgrnnd_directqid_fulladd_20260602_005446
```

结果：

| 路径 | cross total | prepare/H2D sum | kernel | D2H | writeback | 备注 |
|---|---:|---:|---:|---:|---:|---|
| split double-buffer, host `SearchQueue` merge | `9451.1 ms` | `2262.2 ms` | `2952.2 ms` | `87.1 ms` | 每 split chunk 约 `30~47 ms` | 历史 full-quality 有效点 |
| nosplit + qid-lock GPU global merge + id-vector writeback | `11598.0 ms` | `888.3 ms` | `4834.3 ms` | `3.5 ms` | `33.2 ms` | 负结果 |

这个实验有两个重要结论：

1. 输出边界确实是瓶颈。nosplit qid-lock 版本把 D2H 和 host writeback 从“每个 split chunk 都发生”压到几毫秒级，说明原系统在 `SearchQueue -> cross_group_neighbors -> _graph->neighbors` 这一段存在可优化空间。
2. 简单 global merge 不是正确解。每个 qid 一个 device lock 的做法把 host 开销转成 GPU atomic/lock contention，kernel 从 `2952.2 ms` 升到 `4834.3 ms`，最终 cross total 反而从 `9451.1 ms` 变慢到 `11598.0 ms`。

因此，当前代码把该路径保留为显式实验开关，而不是默认优化：

```bash
UNG_GPU_DB_NOSPLIT=1
UNG_GPU_ID_VECTOR_WRITEBACK=1
```

默认仍保持 split double-buffer，避免性能倒退。论文中不能把 qid-lock global merge 写成最终贡献；它应该作为 negative ablation，支撑下一版设计：**qid-sharded / segmented no-lock merge + flat adjacency/CSR graph backend**。具体来说，下一版应该按 qid range 或 segment 让每个 merge worker 独占输出区间，避免 per-qid spin lock；或者先输出 `(qid, gid, dist)` tuples，按 qid 做 segmented topK，再一次性生成 flat edge buffer。

### 13.1 flat-id writeback: 输出容器替换的部分正结果

在 qid-lock global merge 负结果之后，我们进一步拆分“输出边界”问题：不再试图改变 GPU merge 语义，只把成功的 GPU global topK 输出从 host `std::vector<std::vector<IdxType>>` 改成固定宽度扁平数组：

```bash
UNG_GPU_FLAT_ID_WRITEBACK=1
UNG_GPU_DB_NOSPLIT=1
UNG_GPU_GLOBAL_MERGE=1
UNG_GPU_ID_ONLY_WRITEBACK=1
```

实现上，GPU 仍然 D2H 最终 topK ids，但 host 侧不再为 `_num_points` 个小 vector 做 `resize/reserve/emplace_back`。`additional_edges` 的 connected-group 检查和最终 `_graph->neighbors` merge 直接扫描 `flat_ids[num_points * topk]`，无效 slot 用 `UINT32_MAX` 哨兵表示。这个改动不改变 cross-edge topK 语义；它只测量 host 输出容器成本。

Amazon 1% x200 skip-additional A/B：

| 路径 | artifact | Index | Group | Cross | GPU kernel | D2H | merge_cross | 备注 |
|---|---|---:|---:|---:|---:|---:|---:|---|
| 旧 id-vector writeback | `/home/graphdb/fv_runs/flat_id_probe_x200_20260602_065652/vector` | `13291.9 ms` | `6072.02 ms` | `4241.0 ms` | `2569.6 ms` | `2.3 ms` | 未单独打印 | `vector<vector<IdxType>>` 输出 |
| flat-id writeback | `/home/graphdb/fv_runs/flat_id_probe_x200_20260602_065652/flat` | `12791.7 ms` | `6145.07 ms` | `3914.0 ms` | `2581.3 ms` | `2.3 ms` | 未单独打印 | 固定宽度 flat ids |
| flat-id writeback + breakdown | `/home/graphdb/fv_runs/flat_id_probe_x200_breakdown_20260602_065923/flat` | `11050.2 ms` | `4805.19 ms` | `3573.7 ms` | `2578.5 ms` | `2.3 ms` | `203.2 ms` | 新增 stdout breakdown |

breakdown run 的关键输出：

```text
[GPU GEMM] H2D(ms)=152.7  Kernel(ms)=2578.5  D2H(ms)=2.3
[cross_edges] breakdown generate(ms)=3324.6 additional(ms)=0.0 add_offset(ms)=45.0 merge_cross(ms)=203.2 merge_add(ms)=0.1 id_vector_active=0 flat_id_active=1 flat_items=7228800
```

结论：

1. flat-id 是部分正结果：相对同口径旧 id-vector，cross-edge 从 `4241.0 ms` 到 `3914.0 ms`，约 `1.08x`；重跑 breakdown 由于 group graph / host registration 波动，cross 为 `3573.7 ms`，但 GPU kernel 仍稳定在约 `2.58s`。
2. 这个结果证明 host 小对象输出有成本，但也证明“只换输出容器”不够。flat-id 下 `merge_cross=203.2 ms`，D2H 只有 `2.3 ms`，剩余大头在 `generate=3324.6 ms`，其中 GPU kernel 是 `2578.5 ms`，prepare/host plan 仍有数百毫秒。
3. 因为这组是 `UNG_ADDITIONAL_EDGES_IMPL=1` skip-additional，它只能作为输出边界性能拆分，不能作为 full-quality 端到端结论。若要把 flat 输出写成主贡献，下一步必须接入 full-quality additional_edges，并补 search recall A/B。
4. 论文写法应是：flat-id 是 flat adjacency/CSR 的低风险前置步骤，支持“CPU-compatible output boundary 是瓶颈”的论点；不能写成“flat/CSR backend 已完成”。

### 13.2 direct-global + flat-id smoke: offset 不是大头，输出容器才是更有效切入点

为了进一步靠近系统级替代设计，我们新增 `UNG_INTRA_GLOBAL_IDS`。这条路径让 GPU/Adaptive 组内图在填充阶段直接写 global neighbor id，从而跳过 `add_offset_for_uni_nav_graph()`。但该开关只在 cross-edge 和 additional_edges 不依赖 CPU Vamana 时自动启用；如果用户强行与 CPU Vamana cross/additional 混用，构建阶段直接报错。原因是 CPU Vamana 的组内搜索期望邻居为 local id。

Amazon 1%x100 skip-additional smoke：

| 路径 | artifact | Index | Group | Cross | GPU kernel | D2H | add_offset | merge_cross | L100 recall |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| local-id + SearchQueue | `/home/graphdb/fv_runs/direct_global_off_smoke_20260602_084831` | `4693.59 ms` | `493.05 ms` | `1650.95 ms` | `656.9 ms` | `11.1 ms` | `2.5 ms` | `17.2 ms` | `0.816` |
| direct-global + SearchQueue | `/home/graphdb/fv_runs/direct_global_smoke_20260602_084744` | `6682.97 ms` | `521.68 ms` | `3530.31 ms` | `658.5 ms` | `11.3 ms` | `0.0 ms` | `21.0 ms` | `0.816` |
| direct-global + flat-id | `/home/graphdb/fv_runs/flat_id_direct_global_smoke_20260602_084939` | `3986.41 ms` | `498.14 ms` | `707.83 ms` | `255.1 ms` | `0.6 ms` | `0.0 ms` | `0.5 ms` | `0.816` |

解释：

1. direct-global 单独只消掉 `add_offset`，而 NeighborList64 后这个 pass 在 x100 上只有约 `2.5 ms`，不是大幅加速来源。
2. direct-global + SearchQueue 的 cross 抖动到 `3530.31 ms`，但 GPU kernel event 与 local-id 基本一致，不能把这组写成算法负结果；它说明外围路径仍有较大噪声。
3. flat-id writeback 是更有效的结构性切入点：它避免把 GPU topK 输出物化为 `SearchQueue`/小对象容器，使 D2H 从 `11.1` 降到 `0.6 ms`，merge 从 `17.2` 降到 `0.5 ms`。
4. 该实验仍是 `additional_edges=skip`，不能作为 full-quality 主表。若 full-quality 路径仍保留 CPU Vamana additional repair，就不能默认启用 direct-global。下一步必须把 additional_edges 也纳入 flat segment/GraphView 设计，或者实现 GPU exact/additional backend 并做 search recall A/B。

### 13.3 additional_edges direct append: 显然 shortcut 的负结果

另一个自然想法是：不再先构造 `std::vector<std::vector<std::pair<from,to>>> additional_edges`，而是在检查到缺失 parent->child coverage 时直接把补边 append 到 `_graph->neighbors[from_id]`。我们实现了 `UNG_ADDITIONAL_DIRECT_APPEND=1` 做探索。

x400 full-quality 探索：

| 路径 | artifact | 结果 |
|---|---|---|
| direct append 初版 | `/home/graphdb/fv_runs/additional_direct_x400_on_20260602_083520` | build log 停在 GPU `batched_search end` 后，没有 cross breakdown / Index time，summary 只有表头 |
| direct append + per-node lock | `/home/graphdb/fv_runs/additional_direct_x400_lock_on_20260602_083735` | env 含 `UNG_ADDITIONAL_DIRECT_APPEND=1`，但同样停在 `batched_search end` 后，没有完成 build |

这个结果说明，问题不是简单“少一个中间 vector”就能解决。full-quality additional_edges 仍依赖 CPU Vamana search cache、local/global id 语义、并行 connected-group 检查和最终 graph mutation。直接在 OpenMP 阶段修改 per-node mutable adjacency，即使加锁，也没有给出稳定路径。

当前代码把这个开关限制为只支持 `additional_edges_impl=CpuExactScan` 的实验路径；如果在 CPU Vamana additional 下请求，会打印 disabled 信息并回到 materialized additional_edges 路径。论文中应把它写成负面 ablation：它支持 staged flat segment / GraphView / CSR，而不是支持更多锁或直接 append。

## 14. 2026-06-02 source-centric no-lock WMMA cross-edge

针对 qid-lock 负结果，我们实现了一条新的实验路径：

```bash
UNG_GPU_SOURCE_EXACT=1
UNG_GPU_SOURCE_EXACT_MODE=1   # 1=tf32_wmma, 0=cuda_core
```

设计变化：

- 旧 target-centric：按 target group 遍历 `in_neighbors[target]`，同一个 qid 会在多个 target group 中重复出现，最后需要 host `SearchQueue` merge 或 GPU qid-lock merge。
- 新 source-centric：按 source/query group 遍历 `out_neighbors[source]`，每个 qid 只由自己所属 source group 处理一次；kernel 在该 source group 的所有目标 segment 上维护最终 topK，因此天然无锁。
- `mode=0` 是 CUDA-core exact scan，用来验证语义和输出边界。
- `mode=1` 是 TF32 WMMA groupGEMM + fused topK，每个 16-row query tile 在寄存器里维护最终 topK，D2H 只回最终 topK id/dist。

SIFT30 smoke test，`additional_edges=skip`，同一脚本 `scripts/benchmarks/run_ung_cross_edge_ab.sh fused`：

| 路径 | artifact | cross total | prepare/H2D | kernel | D2H | writeback | 备注 |
|---|---|---:|---:|---:|---:|---:|---|
| source-centric CUDA-core exact | `/home/graphdb/FilterVectorResultsRefactor/organized_benchmarks/ung_cross_edge_fused_20260602_052124` | `5981.0 ms` | `20.3 ms` | `5524.0 ms` | `1.9 ms` | `43.0 ms` | 输出边界低，但 kernel 利用率差 |
| target-centric fused 当前路径 | `/home/graphdb/FilterVectorResultsRefactor/organized_benchmarks/ung_cross_edge_fused_20260602_052209` | `4918.9 ms` | `77.4 ms` | `2189.9 ms` | `35.7 ms` | chunk writeback 约 `43.8 ms` | 仍有重复 query visits 和 split/fallback |
| source-centric TF32 WMMA, legacy | `/home/graphdb/FilterVectorResultsRefactor/organized_benchmarks/ung_cross_edge_fused_20260602_052750` | `3021.9 ms` | `23.2 ms` | `1402.9 ms` | `2.9 ms` | `63.1 ms` | 旧 smoke；缺少当前错误检查边界 |
| source-centric TF32 WMMA + source-block padding + id-vector writeback, legacy | `/home/graphdb/fv_runs/source_wmma_padded_sift30_skipadd_20260602_053546` | `1927.7 ms` | `23.2 ms` | `1418.5 ms` | `1.9 ms` | `26.8 ms` | 旧 smoke；只能作为 source tiling 设计线索 |
| source-centric CUDA-core id-only, current checked | `/home/graphdb/FilterVectorResultsRefactor/organized_benchmarks/ung_cross_edge_fused_20260602_091046` | `5926.7 ms` | `23.5 ms` | `5525.5 ms` | `1.0 ms` | `13.3 ms` | 修复 null-distance 写和 stream error check 后的可信负结果 |

从这组三个点可以看出：

1. source-centric 重排本身解决了输出边界和 qid 冲突问题，但可信 CUDA-core id-only 复测显示 kernel 仍是主瓶颈：`5525.5 ms`，完整 cross `5926.7 ms`，慢于 target-centric fused。
2. 旧 TF32 WMMA/padding smoke 曾显示 kernel 可降到约 `1.4s`，说明 Tensor Core source tiling 方向可能有价值；但这些 run 缺少当前的 null-distance 输出防护和 stream error check，不能作为主表加速证据。
3. 旧 WMMA kernel 的一个 reviewer-risk 是：一个 block 内多个 warp 可能处理不同 source group，而 B tile 是 block 级 shared memory。旧 padding 方案是设计线索，但如果要写成主方法，必须重新在当前 checked 代码上验证 TF32/FP32 topK 一致性和 full-quality recall。
4. 当前可信结论是负/边界结果：source-centric 单阶段 kernel 有 no-lock 和低 D2H/writeback 优势，但候选扫描并行度不足。下一步应改成 two-stage source grouped GEMM + per-source reduce，而不是继续声称已有 source path 更快。
5. `prepare_all` 的 host-register/pageable 策略会造成几十到百毫秒级波动；因此任何 source smoke 都不能替代 full-quality 主结果。

这个结果比 qid-lock global merge 更接近无锁替代方向：

- 不依赖全局 per-qid lock。
- 不需要把同一个 qid 的多个 target group 结果交给 host `SearchQueue` 再归并。
- 候选集合仍然来自 LNG `out_neighbors`，语义上对应原来的 cross-edge 候选边集合。

仍需补齐的实验边界：

- 这只是 SIFT30 smoke test，且 `additional_edges=skip`；不能直接当 full-quality 论文主结果。
- 需要在 Amazon 1%x40/x100/x200/x400 和 full additional_edges 上重测 build time 与 end-to-end recall。
- 需要检查 TF32 对 topK 次序的影响；如果 recall 或图结构敏感，需要提供 FP32 CUDA-core exact 对照或 tie-break 修正。
- 最终若要写成系统贡献，还需要把输出进一步接到 flat adjacency/CSR，而不是再回填 CPU `std::vector` graph。

## 15. 2026-06-02 Amazon x200 full-quality source-centric A/B

Reviewer 追问：SIFT30 skip-additional smoke test 不够，source-centric WMMA 能否在 Amazon full-quality 下跑通，并且是否真的比当前 target-centric fused 更快？

这轮先跑最小闭环：Amazon 1% x200 coverage query，CPU Vamana group graph，`additional_edges=cpu_vamana`，L1000 repeat=1。为了让 benchmark 脚本能真实传递 source-centric 开关，已补 `scripts/benchmarks/run_end_to_end_recall_ab.sh` 的可选环境变量白名单：

```text
UNG_GPU_SOURCE_EXACT
UNG_GPU_SOURCE_EXACT_MODE
UNG_GPU_SOURCE_EXACT_WARPS
UNG_GPU_ID_VECTOR_WRITEBACK
UNG_GPU_DB_NOSPLIT
UNG_GPU_DB_GLOBAL_MERGE
UNG_GPU_DB_SPLIT_UNSUPPORTED
UNG_GPU_DB_LARGE_MAX_NX / DB_MEDIUM_MAX_NX / DB_CHUNK_QUERIES
UNG_GPU_PREPARE_DIRECT_HOSTREG / DIRECT_PAGEABLE / HOSTREG_MAX_MB
```

同时修复了一个 full additional_edges 稳定性问题：此前 `SearchCacheList` 被改成懒创建，但 `additional_edges=cpu_vamana` 会在 OpenMP parallel 区域内首次调用 `ensure_search_cache_list()`，导致多个线程竞争初始化同一个 `unique_ptr` 并在 x200 full-quality 中 segfault。现在在进入 additional_edges 并行循环前，如果实现是 CPU Vamana，会单线程预创建 cache。

| 路径 | artifact | Index | Group | Cross | prepare/H2D | kernel | D2H | L1000/L5000 recall repeat=3 | L1000/L5000 time |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| source-centric WMMA w8 | `/home/graphdb/fv_runs/source_wmma_amazon_x200_fulladd_fix_20260602_054536` | `15947.7 ms` | `8046.64 ms` | `4693.4 ms` | `419.5 ms` | `3224.6 ms` | `0.2 ms` | `0.871 / 0.911` | `505.605 / 474.941 ms` |
| source-centric WMMA w4 | `/home/graphdb/fv_runs/source_wmma_amazon_x200_fulladd_w4_20260602_054745` | `14170.9 ms` | `7553.84 ms` | `4803.07 ms` | `418.8 ms` | `3091.9 ms` | `0.2 ms` | `0.871 / 0.911` | `526.791 / 459.486 ms` |
| target-centric fused current | `/home/graphdb/fv_runs/target_fused_amazon_x200_fulladd_current_20260602_054639` | `13250.2 ms` | `7528.36 ms` | `3900.81 ms` | `437.5 ms` | `2095.8 ms` | `9.1 ms` | `0.858 / 0.9068` | `498.369 / 473.352 ms` |

后续复核修正：2026-06-02 新增 source path stream error check 和 id-only null-distance 输出修复后，SIFT30 source CUDA-core id-only 可信复测为 cross `5926.67 ms`、kernel `5525.5 ms`，慢于 target-centric fused 历史对照 cross `4918.9 ms`、kernel `2189.9 ms`。因此本节的 source WMMA 数字保留为历史 full-quality/探索记录，不能作为论文主表加速证据；source-centric 的当前定位应是 no-lock 设计方向和 two-stage source grouped GEMM + per-source reduce 的后续路线。

本轮结论：

1. 历史 source-centric WMMA run 曾在 Amazon x200 full-quality 下跑通，并接上 CPU Vamana additional_edges 和 search。复用已建 index 补跑 L1000/L5000 repeat=3 后，source w8/w4 都是 `0.871/0.911`，没有出现明显质量崩塌。但新错误检查后，这些数字只能支持“方向可行/质量未崩”的探索记录，不能证明当前实现已经是可靠加速主线。
2. 性能上，source-centric 还不能替代当前 target-centric 主路径。w8 cross `4693.4 ms`、w4 cross `4803.07 ms`，都慢于同机 target-centric `3900.81 ms`。核心原因是 source WMMA kernel 为 `3.1~3.2s`，高于 target-centric fused 的 `2.1s`。
3. w4 把 source kernel 从 `3224.6 ms` 降到 `3091.9 ms`，但 full cross total 没降，说明除了 warp/block padding，additional_edges 和外围阶段仍有波动；不能只看 kernel 调参。
4. target current 的 repeat=3 recall 为 `0.858/0.9068`，低于历史 target/CPU x200 结果，说明 current worktree、cross-edge 路径和 additional_edges 组合仍需结构诊断；不能把 source 的更高 recall 直接解释成算法质量贡献。
5. 这改变论文写法：source-centric 是正确的无锁方向和质量可行候选，但当前单阶段实现还不是性能主结果。主文仍应保留 x100 fused cross-edge strong-baseline 表和 x200 partial double-buffer routing 作为 cross-edge 主证据，把 source-centric 写成下一阶段 two-stage no-lock 替代路线和负/部分 A/B。

## 16. 2026-06-02 x200 partial double-buffer routing

继续追问“能否做更普遍替代”后，我们发现 target-centric cross-edge 的一个系统性性能问题：double-buffer direct-qid 路径原先是 **all-or-nothing**。只要一个 batch 中混入 `nx > UNG_GPU_DB_LARGE_MAX_NX` 的 target group，整个 batch 都会回退老路径。Amazon 1% x200 中只有少数大组超阈值，但它们会拖累大量本来适合 double-buffer 的中小组。

修复方式：在 `gpu_cross_groups_search_all_batched()` 中加入 partial routing。对同一个 target group batch，先把 `nx <= DB_LARGE_MAX_NX` 的 supported groups 单独送入 double-buffer direct-qid fused path，再把 unsupported groups 单独回退原路径。当前只在 `SearchQueue` 写回路径启用，避免 id-vector writeback 下跨 split 失去全局 topK merge 语义。

| 路径 | artifact | Index | Group | Cross | prepare/H2D | kernel | D2H | L1000 recall |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| target-current before split | `/home/graphdb/fv_runs/target_fused_amazon_x200_fulladd_current_20260602_054639` | `13250.2 ms` | `7528.36 ms` | `3900.81 ms` | `437.5 ms` | `2095.8 ms` | `9.1 ms` | `0.858` repeat=3 |
| naive `DB_NOSPLIT=1` all-or-nothing | `/home/graphdb/fv_runs/target_fused_x200_db_nosplit_global_20260602_055827/end_to_end_recall_ab_20260602_055827` | `14581.8 ms` | `7683.42 ms` | `4332.46 ms` | `430.1 ms` | `2569.0 ms` | `1.1 ms` | `0.869` repeat=1 |
| partial split routing | `/home/graphdb/fv_runs/target_fused_x200_db_split_20260602_060034/end_to_end_recall_ab_20260602_060034` | `12988.9 ms` | `7676.06 ms` | `2657.23 ms` | `425.3 ms` | `1636.0 ms` | `14.8 ms` | `0.866 / 0.907` repeat=3 |

新 run 的 build log 明确显示：

```text
double_buffer split supported_groups=5173 fallback_groups=22 max_supported_nx=1023
double_buffer chunks=28 groups=5173 queries=7254400
Kernel(ms)=1531.9 D2H(ms)=13.7 writeback(ms)=178.4
```

结论：

1. 这是当前最明确的 cross-edge 大幅优化：相对同机 target-current，Cross `3900.81 -> 2657.23 ms`，为 `1.47x`；kernel `2095.8 -> 1636.0 ms`，为 `1.28x`。
2. 单纯打开 `DB_NOSPLIT=1` 是负结果，因为 all-or-nothing 会让全部 5195 个 target groups 因少数超大组回退老路径。partial split 后 5173 个组进入 double-buffer，只剩 22 个大组回退。
3. 该优化比 source-centric WMMA 更适合作为近期 cross-edge 主线：它保留 target-centric 的分组复用优势，同时减少老路径覆盖面。复用该 index 补跑 L1000/L5000 repeat=3 后，recall 为 `0.866/0.907`，相对旧 target-current 的 `0.858/0.9068` 修复了 L1000，但仍低于 source-centric 的 `0.871/0.911`。因此它是性能强、质量部分修复的 target-route 优化，不应写成质量最优。
4. 结构诊断已补：partial split 的组内结构与 source/CPU 对齐口径一致，`intra_edges=38,546,837`、`zero_intra_ratio=0`、`groups_with_largest_wcc_lt_0_9=0`；cross edges 为 `1,093,032`，只比 source/source-w4 的 `1,093,224` 少 `192`，明显好于旧 target-current 的 `1,088,820`。诊断目录：`/home/graphdb/fv_runs/graph_diag_x200_db_split_20260602`。
5. 若要进一步同时提高速度和质量，需要 no-lock segmented merge 或 source/target hybrid route，而不是 qid-lock global merge。

## 17. 2026-06-02 x200 packed exact-anchor 外围优化

继续追问“更通用的替代路线”后，group graph 的最强候选从纯 FastGrnndCuda 转向 **size-aware router**：小组保留 complete/CPU fallback，中组走 packed exact-anchor，大组走 FastGrnnd/reverse-tail。x200 packed exact-anchor 原先已经有较好 recall，但性能仍被 host 外围拖住。

这次定位到两个具体问题：

1. `build_fast_exact_cuda_batch()` 虽然处理的是已经按 group 重排过的 base vectors，但仍把所有 exact groups 重新打包到临时 host buffer；同时默认在 CPU 上生成 `point_group_ids/point_local_ids`。在 Amazon 1% x200 router_exact_nx4096 上，旧路径 `tagore_pack_time=4140.61 ms`，高于 GPU exact kernel 本身的 `931.72 ms`。
2. exact packed 图回填到 `Graph::neighbors` 时，默认用 128 个 OpenMP 线程给 120 万个 `std::vector` 做 resize/assign，allocator 竞争严重。把 fill 线程降到 16 后，fill 从约 `2.85s` 降到约 `1.38~1.55s`；8 线程可把 fill 降到 `0.63s`，但该轮 H2D 波动较大，因此默认采用更稳的 16。

代码改动：

- 新增 `UNG_FAST_EXACT_DIRECT_H2D=1` 默认：如果 exact batch 的源 group 在 host 上形成连续段，直接按 run 发起 H2D，跳过临时 packed_data。
- 新增 `UNG_FAST_EXACT_DIRECT_H2D_MAX_RUNS=4096`：避免碎片过多时发起大量小 H2D。
- `UNG_FAST_EXACT_DEVICE_LOOKUP` 默认从 `0` 改为 `1`：point lookup 改由 GPU kernel 生成。
- exact packed fill 热路径改成 `resize + memcpy`。
- `UNG_TAGORE_FILL_THREADS` 默认从 `num_threads` 改为 `min(num_threads,16)`，仍可手动覆盖。

Amazon 1% x200 coverage，`router_exact_nx4096`，`fastgrnnd_cpu_fallback`，full additional_edges，L1000 repeat=1：

| 路径 | Index | Group graph | direct build wall | pack | H2D | GPU exact kernel | D2H | fill | Cross | L1000 recall |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 旧 packed + CPU lookup | `19135.1 ms` | `8999.26 ms` | `6078.11 ms` | `4140.61 ms` | `415.45 ms` | `931.72 ms` | `25.27 ms` | `2849.50 ms` | `5901.89 ms` | `0.867` |
| direct H2D + GPU lookup + fill128 | `13008.9 ms` | `4614.02 ms` | `1732.96 ms` | `0.05 ms` | `495.90 ms` | `925.69 ms` | `23.50 ms` | `2865.77 ms` | `5700.82 ms` | `0.867` |
| direct H2D + GPU lookup + fill32 | `11801.0 ms` | `3760.88 ms` | `1835.78 ms` | `0.04 ms` | `519.12 ms` | `926.68 ms` | `26.37 ms` | `1865.30 ms` | `5636.06 ms` | `0.867` |
| direct H2D + GPU lookup + fill16 | `11441.3 ms` | `3204.88 ms` | `1784.17 ms` | `0.04 ms` | `510.17 ms` | `928.67 ms` | `27.78 ms` | `1383.53 ms` | `5646.73 ms` | `0.867` |
| 新默认 direct H2D + GPU lookup + fill<=16 | `13839.9 ms` | `3579.27 ms` | `1964.15 ms` | `0.05 ms` | `749.10 ms` | `934.37 ms` | `18.94 ms` | `1553.28 ms` | `6307.25 ms` | `0.867` |

结论：

1. packed exact-anchor 的旧性能不佳主要不是 GPU exact kernel，而是 host pack 和 `std::vector` 图物化。去掉 pack 后，group graph 从 `8999.26 ms` 降到 `4614.02 ms`；再限制 fill 线程后，手动最优降到 `3204.88 ms`，相对旧路径 `2.81x`。
2. recall 不变，说明这是外围实现优化，不改变 exact-anchor 图语义。
3. 新默认 run 的 H2D/cross 有波动，但 group graph 仍为 `3579.27 ms`，相对旧路径 `2.51x`。报告表中应优先列 group graph breakdown，避免把 cross-edge 同机波动误解释成 group graph 退化。
4. 下一层大幅优化必须改 Graph 物化边界：当前 `Graph` 是每点一个 `std::vector<IdxType>`，120 万点会触发大量小分配。若要继续推进系统级替代，需要在保存和搜索路径支持 flat adjacency/CSR，或者让 GPU group graph 直接输出最终 index 文件格式，减少 CPU container 回填。

## 18. 2026-06-02 universal flat double-buffer cross-edge 最终定位

继续从 reviewer 角度追问“能否提出一个更通用的 cross-edge 替代”后，结论从 source-centric 单阶段 kernel 转向 **universal flat double-buffer**。原因是：

1. source-centric 的遍历方向正确，能天然避免 qid-lock，但当前单阶段 kernel 把一个 source query/tile 的多个 target segment 串行扫完，候选扫描并行度不足。修复 id-only null-distance 写入和 stream error check 后，可信 SIFT30 CUDA-core source-centric cross 为 `5926.67 ms`、kernel `5525.5 ms`，慢于 target fused。
2. target-centric partial double-buffer 已证明旧路径的主要问题之一是 coverage/fallback 边界，而不是一定要改变遍历方向。
3. 输出边界的有效改法不是 qid-lock global merge，而是 flat-id output + GPU global merge，减少 `SearchQueue`/per-group container 的 host 写回成本，同时保留 target-centric 语义。

当前实现通过 `UNG_UNIVERSAL_GPU=1` 启用：

```text
target-centric descriptor batching
double-buffer execution
GPU global merge
flat-id output
default all-DB coverage via UNG_GPU_DB_LARGE_MAX_NX=1048576
```

### 18.1 SIFT30 skip-additional smoke

artifact：

```text
/home/graphdb/FilterVectorResultsRefactor/organized_benchmarks/universal_all_db_sift30_20260602_092318
```

| 路径 | Cross | Resident/prepare后 | Kernel cumulative | D2H | writeback/merge | 备注 |
|---|---:|---:|---:|---:|---:|---|
| target fused 历史对照 | `4918.9 ms` | - | `2189.9 ms` | - | - | 旧 target-centric fused |
| source-centric checked negative | `5926.67 ms` | `5890.3 ms` | `5525.5 ms` | `1.0 ms` | `13.3 ms` | 修复错误检查后的可信负结果 |
| universal all-DB | `3180.63 ms` | `3142.9 ms` | `3870.4 ms` | `2.1 ms` | `15.0 ms` | 94 chunks 全部走 double-buffer；kernel 为跨流累计，墙钟更低 |

结论：SIFT30 上 universal route 相对 target fused cross `4918.9 -> 3180.63 ms`，约 `1.55x`；相对修复后的 source-centric 为约 `1.86x`。但该实验是 skip-additional cross-edge smoke，不是 full-quality recall 结论。

### 18.2 Amazon 1% x200 full-quality positive

artifact：

```text
/home/graphdb/fv_runs/end_to_end_recall_ab_20260602_092928
```

配置：CPU Vamana group graph，CPU Vamana `additional_edges`，只把 cross-edge 换成 `UNG_UNIVERSAL_GPU=1`。

| 路径 | Cross | generate | additional_edges | GPU kernel cumulative | D2H | merge_cross | L1000/L5000 recall |
|---|---:|---:|---:|---:|---:|---:|---:|
| x200 partial DB target route | `2657.23 ms` | - | - | - | - | - | `0.866 / 0.907` |
| x200 universal flat all-DB | `2494.21 ms` | `1636.2 ms` | `843.5 ms` | `1887.5 ms` | `1.2 ms` | `0.7 ms` | `0.871 / 0.911` |

结论：这是 universal route 从 skip-additional 推进到 full-quality 的正结果。它证明该路径能接上 CPU Vamana additional_edges 和 search recall，并且 cross-edge 低于已有 x200 partial double-buffer 对照。

### 18.3 Amazon 1% x100 full-quality boundary

artifact：

```text
/home/graphdb/fv_runs/end_to_end_recall_ab_20260602_093357
```

同样固定 CPU Vamana group graph 和 CPU Vamana `additional_edges`，只替换 cross-edge。

| 路径 | Index | Cross | generate | additional_edges | GPU kernel cumulative | D2H | merge_cross | L100/L500/L1000 recall |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| x100 historical CPU-group target route | `6863.47 ms` | `1732.42 ms` | - | - | `470.4 ms` | `9.0 ms` | - | `0.825 / 0.868 / 0.890667` |
| x100 universal flat all-DB | `7461.27 ms` | `1689.40 ms` | `1005.2 ms` | `675.3 ms` | `872.0 ms` | `0.6 ms` | `0.7 ms` | `0.826 / 0.868 / 0.891` |

结论：x100 证明 universal route 的 full-quality 兼容性和 cross-edge 小幅收益，但不支持端到端加速主张。Index time 变慢，search latency 也有波动，因此这一点必须写成 boundary result。

### 18.4 当前论文写法

可以写：

```text
Universal flat double-buffer is the current cross-edge engineering route.
It provides a positive full-quality x200 result and a compatible x100 boundary result.
```

不能写：

```text
Universal flat double-buffer solves all workloads or provides unconditional end-to-end speedup.
```

下一步若继续推进，应优先做：

1. x400、10%x40、真实多标签 full-quality A/B；
2. additional_edges 的 GPU/flat backend；
3. GraphView/CSR 输出路径，让 group graph、cross-edge 和 additional_edges 都直接写 flat segment，减少 CPU-compatible per-node object 物化。

## 19. 2026-06-04 GPU 覆盖正确的查询入口组选择

本节记录查询阶段的新优化：把原来 CPU 上的入口组查找拆成可批处理的 GPU 覆盖选择。它对应的 benchmark 工具是 `UNG/codes/tools/query_entry_group_bench.cu`，脚本入口是 `scripts/benchmarks/run_query_entry_group_bench.sh`。

### 19.1 查询入口组语义

UNG 查询不是直接从所有满足 label 条件的叶子/细粒度 group 开始搜，而是先在 label tree / LNG 上找到一批入口组。理想 CPU exact minimal 的目标是：

```text
C = {g | query_labels subset labels(g)}
在 C 中选择最上层的一批组，使这些组的所有后代恰好覆盖 C，
并尽量去掉彼此之间存在祖先/后代关系的冗余入口。
```

其中 `C` 是所有满足过滤条件的候选组。CPU exact minimal 的输出组数少，但代价高：先全量 scan 找候选，再做最小化/剪枝。很多查询下，后续 UNG search 可以接受少量入口组冗余，只要不漏掉任何满足过滤条件的 group。因此 GPU 路径采用“保证覆盖、允许冗余”的策略。

### 19.2 GPU correct-cover 算法

当前保留的 GPU provider 是 `gpu_cover_frontier`。旧的只取 frontier 的 `gpu_frontier` 路径因为不能保证覆盖全部候选，已经不作为有效实现考虑。

对每个 query：

```text
C = {g | query_labels subset labels(g)}
F_raw = {g in C | |query_labels| <= |labels(g)| <= |query_labels| + delta}
F = first_cap(F_raw)
Covered = union(descendants(f) for f in F)
Uncovered = C - Covered
Output = F union Uncovered
```

变量含义：

- `delta`：frontier 的 label-size 窗口。`delta` 越大，越可能选到更浅/更少的入口组，但 coverage OR 的工作量也更大。
- `cap` / `--frontier-cover-cap`：每个 query 最多拿多少个 `F_raw` 里的 frontier group 去做 descendant coverage OR。它只限制参与 coverage 的 frontier 数，不限制最终输出组数。
- `avg_frontier_groups`：实际参与 coverage OR 的平均 frontier group 数。
- `avg_output_groups`：最终返回给查询阶段的平均入口组数。

正确性来自两个集合关系：

1. `Covered` 只用于减少冗余；如果某个候选组没有被 `F` 的后代覆盖，它会进入 `Uncovered`。
2. `Output = F union (C - Covered)`，因此对任意 `g in C`，要么被某个 `f in F` 的后代覆盖，要么直接出现在输出里，不会漏掉满足过滤条件的候选组。

所以即使 `cap` 截断了 `F_raw`，也只会减少 `Covered`、增加 `Uncovered`，导致输出组更多，但不会破坏覆盖语义。它和 CPU exact minimal 的区别是：CPU 输出尽量 minimal，GPU 输出 coverage-correct 但可能冗余。

### 19.3 实现细节

CPU exact minimal 的慢点主要在两步：

1. 多线程 scan 所有 group label，找出 `C`。
2. 对候选组做最小化/覆盖剪枝，去掉父子冗余。

GPU correct-cover 把查询批处理后，以 dense bitset 表示候选、frontier、descendant coverage 和输出。最终实现里最关键的优化是把旧的全 frontier bitset 扫描改成 list-driven coverage：

- `compact_frontier_ids_kernel`：把每个 query 的 `frontier_bits` compact 成连续的 `frontier_ids`，同时生成 `selected_frontier_bits`。
- `descendant_cover_list_kernel`：只遍历 compact 后的 frontier id 列表，对这些 frontier 的 descendant bitset 做 OR。
- `cover_frontier_select_kernel`：输出 `selected_frontier_bits | (candidate_bits & ~covered_bits)`。

这个改法避免了旧 `descendant_cover_kernel` 对每个 `(query, bitset word)` 再扫描完整 frontier bitset。对 `delta=1` 的 100%x40 测试，旧路径约 `762.92 ms`，list-driven 后降到 `60.72 ms`，内部提升约 `12.2x`。

工程上还做了两点计时/基线修正：

- D2H 输出 bitset 使用 pinned host buffer；当前仍然传回完整输出 bitset。
- `UNG/codes/tools/CMakeLists.txt` 中 CUDA benchmark target 链接 `OpenMP::OpenMP_CUDA`，否则 CPU “128T” 基线实际可能没有正确启用 OpenMP。

### 19.4 Amazon 100%x40 nq=10240 结果

测试口径：

- base labels：`/home/fengxiaoyao/FilterVector/FilterVectorData/Amazon/Amazon_base_labels.txt`
- query labels：复用 10%x40 coverage query 并重复到 `nq=10240`
- full Amazon unique groups：`482388`
- dense descendant bitset：约 `27.1 GiB`
- 表中 GPU 时间是 resident/steady-state 查询时间，不包含 descendant bitset 构建与 H2D 初始化；冷启动构建约 `6~7 s`，不适合单次查询摊销。

| 方法 | delta | cap | total ms | kernel ms | D2H ms | CPU materialize ms | avg_frontier_groups | avg output groups | QPS | 相对 CPU scan |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| CPU scan 128T | - | - | `1187.56` | `1187.56` | `0` | `0` | - | `23332.2` | `8623` | `1.00x` |
| CPU exact minimal | - | - | `4569.83` | `1104.21` | `0` | `3465.62` | - | `616.24` | `2241` | `0.26x` |
| GPU cover | `1` | `8192` | `59.8294` | `24.1543` | `24.9913` | `10.6587` | `17.7699` | `2184.26` | `171153` | `19.85x` |
| GPU cover | `2` | `8192` | `219.009` | `183.273` | `25.0302` | `10.68` | `192.437` | `1435.14` | `46756` | `5.39x` |
| GPU cover | `3` | `64` | `80.4164` | `33.8344` | `25.9203` | `20.6239` | `27.6805` | `14574.0` | `127337` | `14.68x` |
| GPU cover | `3` | `128` | `96.684` | `52.634` | `25.0181` | `19.0011` | `48.2824` | `12197.1` | `105914` | `12.21x` |
| GPU cover | `3` | `256` | `128.945` | `83.125` | `26.158` | `19.6277` | `81.5456` | `10337.8` | `79411` | `9.15x` |
| GPU cover | `3` | `512` | `167.385` | `124.141` | `26.554` | `16.6608` | `126.288` | `9216.31` | `61176` | `7.05x` |
| GPU cover | `3` | `1024` | `218.899` | `175.06` | `26.6624` | `17.1505` | `182.396` | `5658.99` | `46780` | `5.39x` |
| GPU cover | `3` | `2048` | `258.799` | `222.969` | `26.0766` | `9.72493` | `235.098` | `1415.50` | `39568` | `4.56x` |
| GPU cover | `3` | `8192` | `415.523` | `380.322` | `26.0607` | `9.10423` | `408.948` | `1279.98` | `24644` | `2.84x` |

结论：

1. 如果以完整接口为边界，即输入 query label、输出 CPU-side group ids/bitset，`delta=1, cap=8192` 当前是最快点：`59.83 ms / 10240 queries`，约 `171k QPS`，相对 CPU scan 128T 为 `19.85x`。
2. CPU exact minimal 输出最少，平均 `616.24` 个入口组，但总时间 `4569.83 ms`，其中 prune/materialize 为 `3465.62 ms`，明显不适合作为高吞吐批处理路径。
3. `delta=3` 通过调小 `cap` 可以保持较高 QPS，但输出组数会显著增加。例如 `cap=64` 时 `80.42 ms`，但平均输出 `14574` 个入口组；`cap=8192` 输出降到 `1279.98`，但时间升到 `415.52 ms`。
4. 当前主要剩余开销是 D2H 完整 bitset 回传：100%x40、`nq=10240` 时输出 bitset 约 `617 MiB`，即使用 pinned memory 仍约 `25 ms`。下一步应在 GPU 上 compact 成 `(offsets, group_ids)`，只回传真实输出组 id。
5. 复现实验 artifact：`/home/graphdb/FilterVectorResultsRefactor/organized_benchmarks/query_entry_group_100pct_x40_20260605_105039`。

### 19.5 正确性验证与使用边界

已经做过两类 sanity：

- 10%x40、`2048` queries、`delta=3 cap=8192`：GPU checksum 与 CPU reference 一致。
- 100%x40 full groups、`100` queries、`delta=1 cap=8192`：GPU 与 CPU reference 的 `avg_candidates=2065.11`，checksum 同为 `6016792615221813030`。

使用边界：

- 这是查询阶段入口组选择优化，不是 PG 构建或 cross-edge 构建优化。
- 它保证 coverage，不保证输出和 CPU exact minimal 完全一致。
- resident 查询吞吐很高，但需要把 descendant bitset 常驻或至少批量摊销。若每次冷启动都重新构建并上传 `27.1 GiB` descendant bitset，则单次查询不划算。
- 当前返回完整 bitset 到 CPU，适合 benchmark 和验证；生产路径应继续做 GPU-side compact output，减少 D2H 和 CPU materialize。

### 19.6 对端到端图查询 latency 的影响

需要区分两个层次：

1. **入口组选择本身**：`query labels -> entry group ids/bitset`。这是 `gpu_cover_frontier` 已经 benchmark 的部分。
2. **后续图查询**：`entry group ids -> entry points -> iterate_to_fixed_point()`。这部分仍在 CPU 图结构上执行，耗时取决于入口组数量、每组取多少 entry point、visited set 初始化、搜索队列扩展和图质量。

当前 `search_UNG_index` 的 per-query 计时字段对应关系如下：

```text
Time_ms              = 单 query 总时间
MinSupersetT_ms      = CPU get_min_super_sets_debug() 入口组查找
search_time_ms       = entry points materialization + 后续搜索阶段
core_search_time_ms  = iterate_to_fixed_point() 或 ACORN core search
NumEntries           = 输出入口组数量
```

因此，入口组 GPU 化后的端到端收益不能直接等于 `19.85x`。如果只把 `MinSupersetT_ms` 替换成 resident GPU correct-cover 的平均 `59.8294 / 10240 = 0.00584 ms/query`，并假设后续图查询耗时不变，则端到端上界为：

```text
T_new_upper_bound = T_old - MinSupersetT_cpu + T_entry_gpu
```

代表性已有 artifact 的分解如下。注意这些是 **替换入口组阶段的估算上界**，不是已经接入生产 search path 后的实测，因为当前 `gpu_cover_frontier` 仍是独立 benchmark 工具，`search_UNG_index` 正式路径仍调用 CPU `get_min_super_sets_debug()`。

| workload / Lsearch | old total ms/query | CPU entry ms/query | 后续 search ms/query | CPU avg NumEntries | entry 占比 | 替换 entry 后估算 total | 估算 speedup |
|---|---:|---:|---:|---:|---:|---:|---:|
| Amazon 10%x40 full-quality, L100 | `70.242` | `35.569` | `34.534` | `107.94` | `50.6%` | `34.679` | `2.03x` |
| Amazon 10%x40 full-quality, L500 | `62.420` | `31.801` | `30.615` | `107.94` | `50.9%` | `30.625` | `2.04x` |
| Amazon 10%x40 full-quality, L1000 | `66.440` | `31.825` | `34.610` | `107.94` | `47.9%` | `34.621` | `1.92x` |
| Amazon 1%x200 coverage, L1000 | `50.988` | `3.406` | `47.071` | `66.67` | `6.7%` | `47.588` | `1.07x` |
| Amazon 1%x200 coverage, L5000 | `48.166` | `3.764` | `44.241` | `66.67` | `7.8%` | `44.408` | `1.08x` |
| Amazon 1%x400 CPU baseline, L1000 | `41.229` | `3.492` | `37.280` | `29.08` | `8.5%` | `37.743` | `1.09x` |
| Amazon 1%x400 CPU baseline, L5000 | `36.188` | `3.918` | `31.953` | `29.08` | `10.8%` | `32.276` | `1.12x` |

解读：

1. **many-group 查询最受益。** 10%x40 的 `MinSupersetT_ms` 占总 query latency 约 `48~51%`，所以入口组 GPU 化即使不动后续图搜索，也可能把端到端查询压到约 `1.9~2.0x`。
2. **x200/x400 coverage query 受益有限。** 这些 workload 的 CPU entry 已经只有 `3~4 ms/query`，后续图查询占主导，因此入口组加速的端到端上界只有 `1.07~1.12x`。
3. **后续图搜索可能被入口组冗余放大。** `gpu_cover_frontier` 的 `delta=1, cap=8192` 在 full Amazon 100%x40 上平均输出 `2184.26` 个 group，而 CPU exact minimal 为 `616.24` 个 group。它保证 coverage，但如果直接把这些冗余 group 转成 entry points，`get_entry_points_given_group_id()` 和 `iterate_to_fixed_point()` 的输入会变大，后续图查询可能变慢。上表中的估算隐含了“后续 search 不变”的上界假设。
4. **真正生产路径应输出 compact group ids，并控制冗余。** 当前完整 bitset D2H 约 `25 ms / 10240 queries`；下一步应该在 GPU 上 compact `(offsets, group_ids)`，并用 `delta/cap` 或 GPU-side lightweight pruning 控制 `NumEntries`，再做同一 index、同一 query、同一 Lsearch 的端到端 A/B。

### 19.7 方法总结：查询入口组 GPU correct-cover

这个方法不是把 CPU exact minimal 原样搬到 GPU，而是把 UNG 查询入口组问题改写成一个 coverage-correct 的批处理集合运算：

```text
Input:
  query label batch Q
  group label bitsets B_label
  group-size bitsets B_size
  descendant bitsets D_group

GPU pipeline:
  1. candidate_frontier_kernel:
       对每个 query 做 label bitset intersection，得到 C；
       再按 label size window 得到 F_raw。
  2. compact_frontier_ids_kernel:
       把 F_raw 截断到 cap 并 compact 为 frontier id list F；
       同时写 selected_frontier_bits。
  3. descendant_cover_list_kernel:
       对 F 中的 group 做 descendant bitset OR，得到 Covered。
  4. cover_frontier_select_kernel:
       输出 F union (C - Covered)。

Output:
  coverage-correct entry group bitset / ids
```

算法设计点：

- **不依赖精确 label 组合存在。** 如果 query label 的精确组合没有对应 group，`C` 仍然由所有实际存在的 superset group 构成；`F` 为空时最坏输出就是 `C`，不会漏。
- **`cap` 不影响 correctness。** 截断 frontier 只会减少 `Covered`，从而让更多 group 落入 `Uncovered = C - Covered` 并被直接输出。
- **descendants 表是关键前提。** benchmark 中 `build_descendant_bitsets_from_labels()` 按 label 包含关系直接构造 descendants，因此覆盖证明成立；生产 LNG 路径必须保证 BFS descendants 至少覆盖所有真实 label-superset group。
- **性能核心是 list-driven coverage。** 旧做法按 `(query, word)` 扫完整 frontier bitset；新做法先 compact frontier id，再只对这些 id 的 descendant bitset 做 OR，把 `delta=1` 路径从约 `762.92 ms` 降到 `60.72 ms`。

## 20. 2026-06-05 统一加入 Amazon 100%x40 对比口径

为避免只在 1%/10% sampled workload 上报告加速比，本轮补了 `Amazon 100%x40` 口径。数据集由 full Amazon 非空 base rows 复制 `x40` 并加同类 jitter 生成：

```text
dataset: /home/graphdb/FilterVectorBenchData/fv_amazon_100pct_x40_jitter_nonempty
source_points: 602453
nonempty_source_points: 582117
repeat: 40
points: 23284680
dim: 768
labels: 23284680 lines
```

### 20.1 建图阶段：初始化崩溃和 GPU resident OOM 已修复，X-streaming v1 可完成但不是加速结果

建图 smoke 使用当前优化路线的保守 build-only 口径：

```text
UNG_GROUP_GRAPH_IMPL=4              # adaptive_cuda
UNG_UNIVERSAL_GPU=1                 # universal flat double-buffer cross-edge
UNG_ADDITIONAL_EDGES_IMPL=1         # skip additional_edges, isolate build scalability
UNG_GROUP_GRAPH_BOUNDED_COMPLETE=1
UNG_GPU_FLAT_ID_WRITEBACK=1
```

第一次 smoke 在 `prepare_group_storages_graphs` 附近 segfault。gdb 定位到 `Storage<float>::reorder_data()` 的 `memcpy`，根因不是 group id 损坏，而是 `Storage` 多处用 `IdxType(uint32_t)` 做 `num_points * dim` 和 `idx * dim` 乘法。`23,284,680 * 768` 超过 32-bit，导致 load 阶段实际分配/读取了溢出后的短 buffer，reorder 时访问越界。

修复内容：

- `Storage::load_from_file()` / `write_to_file()` / `reorder_data()` 的向量 byte count 改用 `size_t` checked multiplication。
- `Storage::get_vector()` / `prefetch_vec_by_id()` / `choose_medoid()` 的 `idx * dim` 偏移改用 `size_t`。
- `aligned_alloc` 对应释放改为 `std::free`。
- GPU cross-edge `prepare_all` 对 `cudaMalloc(g_d_all_X)`、H2D、norm kernel 做显式错误检查，避免 OOM 后继续 kernel 并留下 sticky illegal access。
- GPU cross-edge 失败且组内图写 global ids 时，fallback 不再走 CPU Vamana；CPU Vamana fallback 需要 local ids，语义不兼容，现在改为 CPU exact scan fallback。
- 新增 `UNG_GPU_X_STREAMING=1`：按 target group 的 X range 分块上传到 GPU，descriptor 的 `x_off` 改为 chunk-local offset，host writeback 时再加回原始 target group global base。这样不再需要一次性常驻 `g_d_all_X`。

resident-cache OOM 复现 artifact：

```text
/home/graphdb/fv_runs/amazon_100pct_x40_build_strict_20260605_114201/optimized_skipadd_strict
```

X-streaming strict build artifact：

```text
/home/graphdb/fv_runs/amazon_100pct_x40_xstream_strict_20260605_120230
```

| 数据集 | 点数 | groups | 方法 | 状态 | load ms | Index | Group | Cross | speedup |
|---|---:|---:|---|---|---:|---:|---:|---:|---:|
| Amazon 1%x100 | `602,400` | `5,666` | adaptive/full-quality | completed | - | `5450 ms` | `2773.35 ms` | - | 正结果，见正文 |
| Amazon 1%x200 | `1,204,800` | `5,666` | packed exact-anchor/full-quality | completed | - | `11043.5 ms` | `3827.28 ms` | - | 正结果，见正文 |
| Amazon 10%x40 | `2,409,800` | `53,840` | FastGrnndCuda/full-quality | completed | - | `23497.4 ms` | `5147.2 ms` | `9451.1 ms` | `1.53x` vs CPU Vamana group |
| Amazon 1%x400 | `2,409,600` | `5,666` | reverse-tail+repair/full-quality | completed | - | `33044.9 ms` | - | - | `1.10x` vs same-script CPU |
| Amazon 100%x40 | `23,284,680` | `482,387` | optimized skip-add strict + X streaming | completed | `53789 ms` | `945683 ms` | `17785.4 ms` | `697114.7 ms` generate | 不作为加速结果 |

X-streaming 完整 build 阶段分解：

| 阶段 | 时间 ms | 说明 |
|---|---:|---|
| load | `53789` | 真实读取 71.5GB float base + labels |
| label/init/reorder | `44490.4` | 包含 trie/group divide、全量 reorder、group storage/graph view、reserve |
| group graph | `17785.4` | `3959` 个 GPU groups，`458017` 个 bounded complete fallback groups，`20411` 个 CPU fallback groups |
| vector_attr_graph | `9170.56` | 总边 `261,375,800` |
| LNG | `10585.3` | `482,387` groups，`3,993,149` LNG edges，平均出度 `8.3` |
| descendants | `31.3` | optimized epoch BFS |
| coverage | `3673.6` | descendants_direct |
| cross-edge generate | `697114.7` | X-streaming exact cross-edge，`453046` target groups |
| total cross-edge section | `762837.6` | 包含 roaring/init、SearchQueue 物化和 cross-edge 生成 |
| Index time | `945683` | 不含最终 index save |
| index save | `497689.1` | 写出 vector_attr graph、roaring bitsets、reordered vectors/labels |

X-streaming v1 的 cross-edge 内部分解：

| 项 | 时间 ms | 说明 |
|---|---:|---|
| chunks / subchunks | `8 / 1270` | X 按 `UNG_GPU_X_CHUNK_MB=8192` 切成 8 个 chunk，Q/output 再切 subchunk |
| streamed x_vecs | `22,028,560` | 实际有 in-neighbor 的 target group X 总量 |
| streamed queries | `868,198,080` | target-centric 语义下重复展开的 query visits |
| pack_x | `31,765.5` | host 端打包 target X chunk |
| pack_q | `450,338.9` | 主要瓶颈：每个 target group/subchunk 重复从 CPU 打包 Q |
| H2D + Xnorm | `108,868.0` | X/Q H2D 和 norm kernel，显存峰值约 10.6GB |
| GPU kernel | `72,749.8` | medium descriptor + TF32 WMMA tile fused topK |
| D2H | `1,596.6` | topK idx/dist 回传 |
| writeback | `27,550.5` | 写回 host `SearchQueue` |

因此，`100%x40` 的状态已经从“无法进入/无法完成”推进到“可完成但性能不合格”。这行仍不能用于论文主加速表；它支持的结论是：resident full-vector cache 不是不可逾越的正确性/显存边界，但要成为主结果，下一轮必须消除 Q 侧重复 host pack，例如改为 source-centric two-stage、Q-side device gather/cache、或者 flat/CSR 输出边界下的 device-side merge。

### 20.2 查询阶段：100%x40 已有完整入口组 benchmark

查询阶段不需要 70GB 向量，只依赖 label/group 结构。本轮重跑 full Amazon labels + repeated 10%x40 coverage query 到 `nq=10240`：

```text
artifact: /home/graphdb/FilterVectorResultsRefactor/organized_benchmarks/query_entry_group_100pct_x40_20260605_105039
```

| 方法 | total ms | avg output groups | QPS | speedup vs CPU scan |
|---|---:|---:|---:|---:|
| CPU scan 128T | `1187.56` | `23332.2` | `8623` | `1.00x` |
| CPU exact minimal | `4569.83` | `616.24` | `2241` | `0.26x` |
| GPU correct-cover d1 cap8192 | `59.8294` | `2184.26` | `171153` | `19.85x` |

注意：`CPU exact minimal` 输出最少，但总时间更慢；`GPU correct-cover` 保证 coverage，但输出 group 多于 exact minimal。端到端图查询仍需要接入正式 search path 后测 `NumEntries` 和 `core_search_time_ms`。

## 21. 2026-06-05 阶段性展示摘要：工作、发现、现状和下一步

本节用于对外展示这段时间的工作进展。它不是新增单点实验，而是把构建阶段、查询入口组、100%x40 放大实验和当前暴露的问题整理成一个可讨论的技术面。

### 21.1 这段时间完成的工作

围绕 UNG 的 GPU 化，我们实际推进了四条线：

| 方向 | 已完成内容 | 当前结论 |
|---|---|---|
| cross-edge 构建 | grouped fused topK、direct-qid、TF32 WMMA path、universal flat double-buffer、X-side streaming | 在 1%/10% repeat workload 上是最稳定的正结果；100%x40 已能完成，但 v1 streaming 被 Q 侧重复 pack 拖慢 |
| 组内 PG/group graph | FastGrnndCuda、light prune、reverse-tail+repair、packed exact-anchor、bounded fallback、workload-aware router | 中大组有加速空间，但不能无条件替代 CPU Vamana；需要按 `nx` 和质量风险 route |
| 查询入口组 | CPU exact minimal 语义梳理；GPU correct-cover 批处理入口组选择；`delta/cap` 控制覆盖与冗余 | 入口组阶段本身已在 Amazon 100%x40、`nq=10240` 上达到 `19.85x` vs CPU scan 128T，但尚未接入正式 graph search |
| 大规模可扩展性 | 生成并构建 Amazon 100%x40；修复 32-bit offset 溢出；解决 71.5GB resident-cache OOM | 证明系统可以跑到 2328 万点、48 万组，但性能瓶颈从 GPU kernel 转移到 host pack、图查询随机访存和输出边界 |

### 21.2 当前能站住的技术发现

1. **UNG GPU 化必须 workload-aware。**
   小组、多组、中等组、大组的瓶颈完全不同。小组主要卡在 host pack、D2H、Graph fill 和 per-group 调度；中等组适合 packed exact-anchor；大组需要 GNN/reverse-tail 这类近似构图；cross-edge 则需要把大量 irregular group topK 合成少数批处理 kernel。

2. **cross-edge 的核心收益来自减少外围和融合 topK，而不只是更快 GEMM。**
   相对 CPU Vamana cross 的大加速容易被认为 baseline 太弱，所以报告里必须同时列 CPU exact scan、cuVS per-group、SGEMM+topK 和 fused。当前 fused 的价值是把 group API 调用、query 展开、H2D/D2H 和 topK 写回一起压缩。

3. **组内图不能只看构建速度。**
   Tagore/FastGrnndCuda 类方法如果 prune 过轻会损伤可导航性，prune 过重又会慢于 CPU Vamana。x400 的 reverse-tail+repair 说明低成本多样性边比简单保留更多 nearest-neighbor head 更有效。

4. **查询入口组 exact minimal 很贵，但 exact minimal 不是唯一可接受目标。**
   CPU `get_min_super_sets_debug()` 当前是 exact minimal：先找所有满足 `query_labels subset labels(group)` 的候选，再过滤掉非最小 superset。GPU correct-cover 改成保证覆盖所有真实候选 group，允许入口组冗余。这个语义更适合批处理 GPU，但必须控制冗余，否则后续 graph search 可能变慢。

5. **100%x40 说明“可扩展”和“加速”是两个阶段。**
   现在已经从崩溃/OOM 推进到可完成 build 和 query-entry benchmark，但 100%x40 还不能作为加速主结果。它暴露了下一阶段真正需要解决的系统瓶颈。

### 21.3 Amazon 100%x40 查询 graph 耗时异常：问题描述

最近最需要继续研究的是：`Amazon100%x40` 的后续 graph search 耗时相对 `10%x40` 不成比例地变大，而常规统计项并没有同步放大。

已有 formal query timing 如下。注意 `100%x40` 这里复用 10%x40 coverage query 去测时间，GT/recall 不作为质量结论；表中耗时分解可以用于分析 CPU 查询路径。

| workload | Lsearch | total ms/query | scan ms/query | graph ms/query | core search ms/query | avg entry groups | avg dist calcs | avg visited |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Amazon10%x40 | `1000` | `63.57` | `30.53` | `32.99` | `1.69` | `107.94` | `2301.95` | `862.49` |
| Amazon100%x40 | `1000` | `1263.70` | `115.41` | `1145.28` | `549.67` | `292.49` | `4952.30` | `1044.05` |
| Amazon100%x1 | `1000` | `151.03` | `148.62` | `2.35` | `2.09` | `1372.54` | `3378.34` | `1954.05` |

直观看，`dist calcs` 只增加约 `2.15x`，`visited` 只增加约 `1.21x`，`entry groups` 只增加约 `2.71x`，但 `core_search_time_ms` 从 `1.69` 到 `549.67`。这说明 `DistCalcs/NumVisited/NumEntries` 不是解释 graph search CPU 时间的充分指标。

### 21.4 初步归因：DistCalcs 低估了真实 CPU 工作量

当前 `core_search_time_ms` 覆盖的是 `iterate_to_fixed_point()`，核心循环不是单纯距离计算：

```text
for entry_point:
  SearchQueue::insert(entry_point, distance(query, entry_point))

while search_queue has unexpanded node:
  cur = closest unexpanded
  for neighbor in graph.neighbors[cur.id]:
    visited_set.check(neighbor)
    visited_set.set(neighbor)
    SearchQueue::insert(neighbor, distance(query, neighbor))
```

其中 `DistCalcs` 只统计 distance 计算次数，没有统计以下工作：

| 隐藏工作 | 100%x40 为什么更敏感 |
|---|---|
| 邻接表扫描 | 访问过的 neighbor 不产生新的 distance calc，但仍要读 `_graph->neighbors[cur.id]` 并查 visited |
| `VisitedSet` 随机访问 | 100%x40 有 `23,284,680` 点，visited marks 大约几十 MB，随机访问容易超出 cache/TLB 有效范围 |
| `SearchQueue::insert` | 当前是有序数组，插入会二分查找并 `memmove`；`Lsearch=1000` 时队列维护不是常数成本 |
| 向量随机读 | Amazon dim=768，一次距离计算要读约 `3KB` float；100%x40 向量工作集远大于 cache |
| 多线程带宽争用 | `search_hybrid()` 用线程池并发跑 query；大图随机访存下 per-query wall time 会被内存带宽和 LLC/TLB 争用放大 |

额外证据是保存的主图规模：

| workload | graph 文件大小 | 前 100 万行平均出度 |
|---|---:|---:|
| Amazon10%x40 | `709 MB` | `39.15` |
| Amazon100%x40 | `6.5 GB` | `37.67` |

平均出度没有变大，图文件和向量/visited 工作集却变大约一个数量级。因此这更像是内存层级、随机访问和队列维护成本问题，而不是简单的 distance-compute 数量问题。

还有一个计时口径容易误读：`query_details_repeat*.csv` 里的 `core_search_time_ms` 是每个 query 在线程内部记录的 wall time；而 `search_time_summary.csv` 的 `Average_Time_ms` 是整批 query 的 batch wall time。`Amazon100%x40 L1000` 的 batch time 为 `10949.3 ms / 1000 queries`，吞吐视角约 `10.95 ms/query`。per-query wall time 会在线程并发下重叠，同时也更敏感地反映内存争用。因此后续展示时应同时报告 batch QPS 和 per-query 分位数，不能只看一个字段。

### 21.5 当前现状和不能夸大的地方

| 问题 | 当前状态 | 展示时的表述边界 |
|---|---|---|
| cross-edge GPU 加速 | 1%/10% repeat workload 有强证据；100%x40 streaming 可完成但慢 | 可以写成主要贡献，但 100%x40 只能写可扩展性和瓶颈分析 |
| group graph GPU 替代 | packed exact-anchor 和 reverse-tail/GNN 在部分 workload 有质量/速度 Pareto | 不能写成无条件替代 CPU Vamana；必须强调 router |
| query entry GPU | 独立 benchmark 中入口组阶段 `19.85x` vs CPU scan 128T | 还不是正式端到端 search speedup；需要接入 `search_UNG_index` |
| 100%x40 graph query 异常 | 已发现 `DistCalcs` 无法解释 core time；初步指向随机访存/队列/多线程争用 | 这是重要发现和下一步研究方向，不应现在下定论为某一个单点 bug |

### 21.6 下一步研究方向

下一阶段不应只继续调单个 kernel，而要补齐观测指标并改查询执行结构：

1. **给 `iterate_to_fixed_point()` 加更细计数器。**
   至少记录 `edge_scans`、`visited_checks`、`visited_hits`、`visited_misses`、`queue_insert_calls`、`queue_rejects`、`queue_memmove_bytes`、entry point 数量、每 query 最大队列长度。这样才能判断 `core_search_time` 到底是边扫描、visited、queue 还是 distance。

2. **做单线程/多线程对比和硬件计数。**
   对 10%x40 和 100%x40 用同一 query 跑 `1/16/64/128` 线程，配合 `perf stat` 看 cache misses、LLC miss、TLB miss、memory bandwidth。当前怀疑是大工作集随机访问和线程间带宽争用，但需要硬件计数证明。

3. **重构 CPU search queue。**
   当前 `SearchQueue::insert` 是有序数组插入，`Lsearch` 大时 `memmove` 可能成为主要成本。可以测试 heap + lazy expansion、bounded candidate heap、分层队列或 SIMD-friendly topL buffer。

4. **把主图改成更 cache-friendly 的布局。**
   当前 `Graph` 是每点一个 `NeighborList`，大图上指针和邻接内存分散。后续应测试 CSR/flat adjacency、按 group reorder、prefetch 更准确的邻接和向量布局。

5. **把 GPU correct-cover 接入正式 search path，并控制冗余。**
   必须输出 compact `(offsets, group_ids)`，并在 GPU 上做轻量 minimal/near-minimal prune，避免 correct-cover 输出过多 entry groups 后拖慢 graph search。

6. **探索批处理图查询 GPU 化。**
   如果 100%x40 的主要瓶颈确实是 CPU 大图随机访存，那么只加速入口组不够。需要考虑 batch query 下的 GPU visited bitset、frontier expansion、或将 entry group 到候选点的搜索改成分组 batch exact/approx 混合。

### 21.7 展示版一句话总结

目前我们已经把 UNG 构建 GPU 化从单点 kernel 优化推进到 workload-aware 系统设计：cross-edge 是最稳定的加速贡献，group graph 需要按组大小和质量风险自适应 route，查询入口组阶段已经具备高吞吐 GPU correct-cover 替代；但 Amazon 100%x40 暴露出新的核心问题，即后续 graph search 的 CPU 时间由随机访存、队列维护和大图工作集主导，不能再用 `DistCalcs` 解释，下一步必须围绕批处理查询和 cache-friendly 图执行结构继续推进。
