# Reviewer-Author 实验迭代记录

本文档用于交替记录“审稿人攻击点”和“作者应补的证据”。原则是：每个新增实验都必须回答一个明确质疑；每个论文主张都必须能追溯到日志、脚本或表格。

投稿时的 claim-level 证据索引见 `docs/papers/EVIDENCE_MATRIX_CN.md`。本文档保留过程性 reviewer attack / author response；证据矩阵用于决定主文哪些话已经可写、哪些只能作为 limitation 或 ablation。

## 当前总体判断

当前论文最强的部分是 **cross-edge fused topK**：已有 CPU Vamana、CPU exact、cuVS per-group、SGEMM+topK、old fused、final fused 的主对照，并且有 `prepare_all` / H2D / kernel / D2H 拆分。

当前论文最需要继续补强的部分仍是 **FastGrnndCuda 是否能替代 CPU Vamana group graph**：现在已有原始 Amazon PF、SIFT30、Amazon 1% x200 coverage、Amazon 10% x40 和 Amazon 1% x400 gather-Q 多组真实 UNG filtered-search recall A/B。x200 证明 light prune 可以把中等组从性能反例拉回可用区；10%x40 证明 many-group 场景 full-quality 下可加速且 recall/P95/P99 不降；x400 则证明 heavy prune 质量接近但慢、light prune 快但有 `0.0081` recall gap。因此仍不能写成“无损替代”，而应写成“自适应剪枝强度 + 质量路由的 GPU backend”。

2026-06-02 修正 exact 结论：早期 “batched exact kNN group graph” 后端的 GPU core 很快但 recall 下降，后来被结构诊断重新归因。旧 exact-anchor 和新 packed exact 的组内图完全一致，差异来自 cross edges 缺失；因此旧结果不能证明 exact 组内图不可行。新 packed exact-anchor full-quality sweep 在 x200/x400 上给出 CPU 级 L5000 recall 和更低 group build time，使它成为当前最强 group-graph 替代候选之一。

2026-06-02 继续补强“普遍替代”路线：新增 `adaptive_cuda` conservative route，把小组 CPU fallback、中组 packed exact-anchor、大组 FastGrnndCuda 收敛到一个后端，并让 CPU fallback 与 GPU batch 并发。Amazon 1% x100 full-quality 下 Index `5450 ms`、group `2773.35 ms`、L100/L500/L1000 `0.829/0.870/0.895`，相对历史 FastGrnnd full-quality `6370.67 ms`、`3216.21 ms`、`0.828/0.871/0.896` 有 build 改善且 recall 基本一致。对应的激进 bounded fallback 虽把 skip-additional group 降到 `1953.07 ms`，但 L100/L500/L1000 只有 `0.816/0.823/0.827`，因此只能作为 negative ablation。

2026-06-02 进一步追问“输出边界能否成为普遍替代突破口”。qid-lock GPU global merge 是负结果：它降低 D2H/writeback，但把瓶颈转移到 GPU lock contention。随后新增 `UNG_GPU_FLAT_ID_WRITEBACK=1`，把 cross-edge 输出从 `vector<vector<IdxType>>` 改为固定宽度 flat ids。Amazon 1% x200 skip-additional 中，旧 id-vector cross `4241.0 ms`，flat-id cross `3914.0 ms`；breakdown run 为 `3573.7 ms`，其中 `generate=3324.6 ms`、GPU kernel `2578.5 ms`、D2H `2.3 ms`、`merge_cross=203.2 ms`。Reviewer 结论：flat-id 支持“CPU-compatible output boundary 有成本”，但不是完整解决；full-quality additional_edges 和最终 search 图仍需要 flat adjacency/CSR + recall A/B。

同一天继续追问 packed exact-anchor 为什么仍不能成为“普遍替代”。新 A/B 显示，旧 exact batch 的最大问题不是 GPU exact kernel，而是 host pack 和 `Graph::neighbors` fill：x200 old path group `8999.26 ms`，其中 `pack=4140.61 ms`、kernel `931.72 ms`、fill `2849.50 ms`；direct-H2D + GPU lookup + fill16 后 group `3204.88 ms`，`pack=0.04 ms`、kernel `928.67 ms`、fill `1383.53 ms`，L1000 recall 仍为 `0.867`。Reviewer 结论：direct-H2D 是强系统优化，证明 exact-anchor 旧性能被外围掩盖；但剩余 `std::vector` 图物化仍是 CPU-compatible output boundary，下一步必须是 flat adjacency/CSR，而不是继续声称 Vamana/host graph stack 已被替代。

2026-06-02 继续从 CUDA kernel 角度攻击“为什么不直接把 exact-anchor kernel 写得更 GPU 化”。我们实现了实验性 `UNG_FAST_EXACT_WARP_KERNEL=1`：每个 warp 处理一个 source 点，减少原 block-per-source exact kernel 在每个 dst 上的 block-wide reduce 和 `__syncthreads()`。Amazon 1% x200 router exact256 full-quality A/B 证明该想法在当前维度/组大小下是负结果：旧 block kernel group `4031.36 ms`、exact kernel `841.136 ms`、L1000 recall `0.867`；warp kernel group `4334.28 ms`、exact kernel `987.27 ms`、recall 不变。原因是同步减少抵不过距离计算并行度从 128 threads 降到 32 lanes。随后扫 `UNG_TAGORE_FILL_THREADS` 也显示 fill 不是简单线程不足：默认 16 线程 group `4031.36 ms`、fill `1682.11 ms`；64 线程 group `4857.36 ms`、fill `2321.96 ms`；8 线程 group `4475.57 ms`、fill `2113.86 ms`。Reviewer 结论：继续微调 exact kernel 或 fill 线程不能带来普遍替代所需的大幅收益；瓶颈已经明确转移到 `std::vector` 图物化、host allocator 和 CPU-compatible adjacency boundary。实验性 warp kernel 保留为默认关闭的开关，不能写进主方法。

同日继续追问“既然输出边界是瓶颈，能不能用最小改动大幅优化”。新增 `reserve_graph_neighbor_capacity()`，在不改变图语义的前提下为 `Graph::neighbors` 预留容量。Amazon 10%x40 A/B 证明 per-node vector 扩容确实是瓶颈：reserve=0 时 Index `39746.1 ms`、build graph `14238.1 ms`、tagore fill `823.5 ms`、fallback wall `11844.6 ms`、cross merge `1677.1 ms`；reserve=1 时 Index `24704.5 ms`、build graph `1662.7 ms`、tagore fill `23.9 ms`、fallback wall `520.2 ms`、cross merge `2.3 ms`。但 reviewer 继续攻击成立：reserve 自身预留 `106,031,200` 条容量并花 `6487.3 ms`，只是把部分成本前置，不是 GPU-native 图表示。Amazon 1%x200 补充 A/B 显示 auto 策略也必须考虑大点数中组 workload：旧 auto-off fill `2497.0 ms`，最终按 `UNG_GRAPH_RESERVE_AUTO_MIN_POINTS=500000` 开启后 fill `224.8 ms`、cross merge `0.9 ms`。结论：reserve 是低风险止血和强证据，但论文仍必须把最终方向写成 GraphView/CSR/implicit small-group view，而不是声称输出边界已解决。

同日进一步攻击“source-centric no-lock 能否成为普遍替代”。审稿角度认为：如果 source route 真的让每个 qid 只处理一次，应该能同时解决 qid-lock 和 SearchQueue merge。作者复查后发现一个重要实现风险：id-only source path 会让 host 不分配 `d_dis/h_dis`，但 source CUDA-core/WMMA kernel 仍无条件写 `out_dist[...]`；旧代码还缺少 source kernel stream error check，因此可能把 illegal memory access 隐藏成 `kernel=0.0 ms` 或不可靠 smoke。修复后，source kernel 写 `out_dist` 前检查非空，host 不再分配/回传 distance，且 source path 增加 launch/sync 错误检查；`UNG_GPU_SOURCE_EXACT_MODE` 默认改为 CUDA-core exact。SIFT30 skip-additional 可信复测 `/home/graphdb/FilterVectorResultsRefactor/organized_benchmarks/ung_cross_edge_fused_20260602_091046` 显示：source CUDA-core id-only cross `5926.67 ms`、kernel `5525.5 ms`、D2H `1.0 ms`、writeback `13.3 ms`，慢于 target-centric fused 历史对照 cross `4918.9 ms`、kernel `2189.9 ms`。Reviewer 结论：source-centric 方向仍合理，但当前单阶段 kernel 把一个 source query/tile 的所有 target segment 串行扫完，候选扫描并行度不足；旧 TF32 WMMA/padding smoke 只能作为 design signal，不能作为主表加速证据。下一步若继续追求普遍替代，应实现 two-stage source grouped GEMM + per-source reduce，而不是把当前 source path 写成已完成替代。设计记录见 `docs/reports/UNIVERSAL_GPU_REPLACEMENT_ANALYSIS_CN.md`。

同日继续攻击“universal flat double-buffer 只是 SIFT30 skip-additional smoke，不能证明 full-quality”。作者补了 Amazon 1% x200 coverage query 的最小闭环：固定 CPU Vamana group graph，`additional_edges=cpu_vamana`，只把 cross-edge 换成 `UNG_UNIVERSAL_GPU=1` 的 all-DB flat-id route。结果 `/home/graphdb/fv_runs/end_to_end_recall_ab_20260602_092928` 显示 cross-edge `2494.21 ms`，其中 `generate=1636.2 ms`、`additional=843.5 ms`、GPU kernel 累计 `1887.5 ms`、D2H `1.2 ms`、`merge_cross=0.7 ms`、`flat_id_active=1`。复用 index 跑 L1000/L5000 repeat=3，recall 为 `0.871/0.911`。对照已有 x200 partial double-buffer artifact，cross 为 `2657.23 ms`、L1000/L5000 `0.866/0.907`。Reviewer 结论：universal route 已经不只是 skip-additional smoke，可以写成 x200 full-quality 正结果；但 additional_edges 仍是 CPU Vamana，且只覆盖 x200，不能写成所有 workload 的完成态普遍替代。

继续追问“x200 是否孤例”后，作者补了 Amazon 1% x100 同口径 full-quality smoke：固定 CPU Vamana group graph、CPU Vamana additional_edges，只替换 `UNG_UNIVERSAL_GPU=1` cross-edge。结果 `/home/graphdb/fv_runs/end_to_end_recall_ab_20260602_093357` 显示 cross-edge `1689.40 ms`，略低于历史 CPU-group target route `1732.42 ms`；breakdown 为 `generate=1005.2 ms`、`additional=675.3 ms`、GPU kernel 累计 `872.0 ms`、D2H `0.6 ms`、`merge_cross=0.7 ms`、`flat_id_active=1`。repeat=3 L100/L500/L1000 recall `0.826/0.868/0.891`，与历史 `0.825/0.868/0.890667` 持平。但 Index `6863.47 -> 7461.27 ms` 变慢，search latency 也明显变慢。Reviewer 结论：x100 证明 universal route 的 full-quality 兼容性和 cross-edge 小幅收益，但也给出端到端边界；论文不能把 universal flat all-DB 写成无条件端到端加速。

## Round 1: 核心攻击与作者回应

| Reviewer attack | Author response needed | Current evidence | Missing experiment | Status |
|---|---|---|---|---|
| 你们只证明 build 快，没有证明 search recall 不掉。 | 固定 cross-edge 和搜索参数，对比 CPU Vamana group vs FastGrnndCuda/adaptive group 的 filtered-search recall/latency。 | 原始 Amazon PF：group `5.50x`，Index `1.18x`，recall drop < `0.001`；SIFT30：group `5.95x`，Index `1.27x`，L1000 drop `0.00129`；Amazon 1% x100 adaptive full-quality：Index `5450 ms`，L100/L500/L1000 `0.829/0.870/0.895`；Amazon 1% x200 diversified light prune：Index `1.07x`，L5000 drop `0.0024`；Amazon 10% x40 full-quality repeat=3：Index `1.53x`，L50/100/200/500/1000/2000/5000 recall 基本不降；Amazon 1% x400 reverse-tail+repair：Index `33044.9`，L1000/L5000 `0.945567/0.9659`，接近同脚本 CPU `0.946/0.965` 且相对同脚本 CPU Index `36257.3` 约 `1.10x`。 | 真实多标签；x400 更多参数扫。 | 部分回答 |
| `22x` 是不是只打了很弱的 CPU Vamana cross baseline？ | 主文同时报告 CPU exact scan、cuVS per-group、SGEMM+topK；摘要里避免只强调 `22x`。 | x100 cross table 已有强 baseline；`/home/graphdb/fv_runs/baseline_fairness_x100_20260602/cross_baselines.md` 可复现生成。CPU exact 为 128 线程，机器为 Intel Xeon Platinum 8360Y 双路 36C/socket，144 logical CPUs。 | 若时间允许补 CPU exact 1/32/64/128 scaling。 | 部分回答 |
| 你们相对 SGEMM+topK 只有 `1.64x`，贡献是否足够？ | 把贡献表述为 irregular group workload 的系统级 fusion，而不是 dense GEMM kernel 更强。 | x100 SGEMM+topK vs fused table；prepare_all breakdown。 | direct-qid / id-only / resident vectors ablation。 | 待补 |
| cuVS per-group 慢是不是因为你们 baseline 写得差？ | 明确 cuVS per-group 是 workload-level baseline；同时提供 SGEMM+topK 作为 library kernel baseline。 | x100 cuVS per-group 和 SGEMM+topK 已列。 | 记录 cuVS 调用方式、是否每组 H2D/D2H、是否复用资源。 | 待补 |
| x400 direct-qid crash 是否说明方法不稳定？ | 修复前把 x400 主表改为 gather-Q；direct-qid 只作为中等规模优化。 | x400 gather-Q 成功日志；direct-qid illegal memory access 日志。 | strict x400 direct-qid 修复或 hybrid router 规则。 | limitation |
| Amazon sampled+jitter-repeat 是否过拟合？ | 补 SIFT/SIFT30、CelebA 或真实多标签数据。 | SIFT30 已补修复后 fused cross + group graph end-to-end A/B；CelebA 仍只有历史 cross-edge 结果。 | CelebA 或真实多标签端到端 recall。 | 部分回答 |
| cross-edge GPU 化是否会影响查询正确性？ | 固定 CPU Vamana group graph，对比 existing / naive GPU / paper fused 的 filtered-search recall。 | Amazon PF r3 已有三组 search summary，recall 基本一致；`query_source_groups` 当前代码路径不使用。 | repeat-scale rerun。 | 部分回答 |
| Amazon 10% x40 是否能端到端跑通？ | 需要证明大量小组/大量 group 场景下 full-quality build 和 search 都能跑通。 | 已生成 balanced coverage query 和 GT；修复 direct-pageable 和 direct-qid chunk path 后，full-quality A/B 已跑通：CPU Vamana group Index `36053.3 ms`，FastGrnndCuda Index `23497.4 ms`，Index speedup `1.53x`。宽 Lsearch sweep L50/100/200/500/1000/2000/5000 下，FastGrnnd recall 相对 CPU 为 `+0.0019/+0.0018/-0.0010/+0.0000/+0.0020/+0.0020/+0.0010`。 | skip-additional 只能作为性能 ablation；真实多标签仍缺。 | 已补主证据 |
| 你们为了 build 快是不是跳过了必要的 additional edges？ | 同 query/GT 对比 skip vs CPU additional_edges。 | 原始 Amazon PF 上 skip additional_edges recall 约 0.61，CPU additional_edges recall 约 0.91-0.97。 | 论文主表必须把 skip 作为 ablation，不作为完整质量配置。 | 已发现风险 |
| group graph 明明 5.50x，为什么端到端只有 1.18x？ | 做 stage 占比和 Amdahl 分析。 | 带 additional edges 时 FastGrnndCuda 后 group 只占 2.86%，cross 占 51.31%，LNG 占 32.71%。 | 优化 additional_edges 或 LNG；主文解释端到端受剩余阶段限制。 | 已回答 |
| FastGrnndCuda 是否在所有中大组都更快？ | 用 repeat-scale bucket 找失败点，再做自适应剪枝。 | Amazon 1% x200 old heavy prune：group `26620.6 ms` 慢于 CPU `8511.72 ms`，prune `19218.3 ms`；diversified light prune 后 group `6706.36 ms`，prune `759.66 ms`。Amazon 1% x400 heavy prune：group `29192.3 ms` 慢于 CPU `17537.3 ms`，prune `19977.7 ms`；light512 后 group `10362.9 ms`，prune `1022.45 ms`，但 L5000 recall drop `0.0081`。 | 继续补 x100，确定 light/strong prune 阈值和质量增强策略。 | 已发现 tradeoff |
| 如果目标是普遍替代，为什么当前加速仍不稳定？ | 把问题拆成算法质量和系统固定开销两层：旧重 prune 在中等组不划算；旧 exact 负结果又被 cross/additional 口径混杂。系统上不能每个 group 反复分配、同步、写回。 | 代码已把 FastGrnnd prune/reverse workspace 从 per-group 分配改为 batch 一次分配，把 Tagore batch 主路径从阶段间 `cudaDeviceSynchronize` 改为 CUDA event 计时，并把 batched exact D2H 结果改成 packed buffer 直读。x200 packed exact-anchor group `3827.28 ms`、L5000 `0.908`；x400 group `11588.4 ms`、L5000 `0.967`。 | 仍需 x100、10%x40、真实多标签和阈值扫，证明 packed/router 不只是 coverage-query 上成立。 | 部分实测 |
| 能否把 route 收敛成一个可调用后端，而不是靠手工拼环境变量？ | 新增 `UNG_GROUP_GRAPH_IMPL=4` adaptive_cuda，默认 `nx<128` CPU fallback、`128<=nx<=512` packed exact-anchor、`nx>512` FastGrnndCuda，并 overlap CPU fallback/GPU batch。 | Amazon 1% x100 full-quality：Index `5450 ms`、group `2773.35 ms`、L100/L500/L1000 `0.829/0.870/0.895`；历史 FastGrnnd full-quality为 `6370.67 ms`、`3216.21 ms`、`0.828/0.871/0.896`。 | 需要在 10%x40/x200/x400/真实多标签上复测；需要自动阈值选择和阈值 sweep。 | x100 正结果 / 普遍性待补 |
| 如果小组 CPU fallback 仍是瓶颈，为什么不直接替换？ | 必须实测 quality，而不是只看 build。bounded-complete 小组替代确实降低 build，但伤 recall。 | x100 skip-additional bounded fallback group `1953.07 ms`，但 L100/L500/L1000 `0.816/0.823/0.827`；conservative route 保留 CPU fallback 后 full-quality recall 基本不降。 | 需要新的轻量小组构图，而不是环形强连通。 | negative ablation |
| x400 repair 是否只是人为补边，可能提高结构指标但不提升真实 search recall？ | 必须同时报告 search recall 和保存图结构诊断；如果只改善 degree/WCC 而 recall 不升，应作为 negative/partial，而不是主方法。 | x400 A/B 已完成。repair-only 把 low-degree 修掉但 Index 变慢、recall 只小幅提升；reverse-tail+repair 把 L5000 从 `0.9574` 提到 `0.9659`，同时 zero/low-degree 基本消失。 | 更多参数扫。 | 已回答为 partial/positive |
| compact-D2H 是否只是工程优化，不能算算法贡献？ | 单独做 `UNG_TAGORE_COMPACT_D2H=0/1` ablation，只报告对 `tagore_d2h_time/tagore_fill_time/Index` 的影响，不把它写成 recall 改进。 | 已完成 x400 compact A/B：compact-on 没有稳定 D2H/fill/Index 收益，当前是负/不稳定 overhead ablation。 | 如主文提及，只能放反例/外围开销。 | 已回答为 negative |
| SIFT30 上 fused cross-edge strict 会失败，是否说明系统不稳定？ | 已修两个语义问题：direct-qid-all 对 fallback path 覆盖不足；host merge 下 id-only writeback 破坏跨组排序。 | 修复后 SIFT30 CPU Vamana group + fused cross L1000 recall `0.877587`，与 CPU cross `0.877523` 基本一致。 | 仍需 x400 direct-qid strict 修复。 | 已回答 SIFT30 |
| 能否用更简单的 batched exact kNN 普遍替代 Vamana？ | 需要端到端 search recall、full-quality cross/additional_edges 和结构诊断，而不是只看 GPU exact kernel 时间。 | 旧 x200 exact L5000 `0.819867/0.829` 已被诊断为 cross-edge 口径混杂；新 packed exact-anchor x200 L5000 `0.908`、x400 L5000 `0.967`。 | 需要补 x100、10%x40、真实多标签和更多参数扫；当前可作为强候选，不能无条件泛化。 | 正向候选 |
| mixed exact/GNN router 会不会把低质量 exact 图混进最终方法？ | 不能只报告 router build 变快，必须扫 `UNG_FAST_GRNND_BATCH_EXACT_NX` 并比较 end-to-end recall。 | Amazon 1% x200 coverage repeat=1 已扫 `0/128/256/512`；`nx256/512` group 明显快于 non-exact router，L1000 recall 接近 CPU Vamana。 | 还需 repeat=3 和区分 exact router vs multi-stream/fast-fill 的 ablation。 | 初步回答 |

## Round 2: 普遍替代路线的反例和修正

| Reviewer attack | Author response needed | Current evidence | Missing experiment | Status |
|---|---|---|---|---|
| 如果目标是“普遍替代”，为什么不把所有小/中组都搬到 GPU exact-anchor？ | 必须把 GPU kernel 时间和 pack/H2D/fill/fallback 物化分开；如果外围吞掉收益，应设计 size-aware router，而不是盲目 GPU 化。 | Amazon 10%x40 `complete_threshold=32` 让 `53,833` 个小/中组进入 packed exact，`tagore_points=2.4098M`，但 `pack=2728.81 ms`、`h2d=1014.8 ms`、`fill=1968.77 ms`，group 从普通 complete64 exact 的 `6097.03 ms` 变成 `7329.12 ms`，recall L100 仍为 `0.867`。 | 需要把该负结果整理成 router policy：小组不盲目上 GPU；中组 exact；大组 GNN/reverse-tail。 | 已回答为负结果 |
| 10%x40 还有什么大幅优化空间？ | 检查小组 complete graph 是否过度物化。若 `Lsearch/Lbuild` 足够覆盖小组，可用有界强连通小图替代完整 `nx-1` 出边。 | 新增 `UNG_GROUP_GRAPH_BOUNDED_COMPLETE=1`。在 10%x40 full-quality build 中，`fallback_wall` 从 `4225.67 ms` 降到 `3088.13 ms`，Index 从 `25035.9 ms` 降到 `22230.2 ms`，group 从 `6097.03 ms` 降到 `5240.6 ms`。复用 bounded index 的 repeat=3 search sweep 给出 L100/L500/L1000 recall `0.867/0.900/0.909`，与 CPU `0.8647/0.898/0.908` 同量级。 | 新的攻击点是 latency：bounded sweep 的 avg time `1083.34/951.434/839.479 ms` 高于历史 CPU/FastGrnnd run，需要同脚本 rerun、缓存状态控制和路径诊断；也需在 x40/x100/x200 其他比例上测。 | build 正结果 / latency 待解释 |
| 这个 bounded-complete 是否改变了原始语义？ | 必须承认它不是完整图逐边等价，而是小组 fast path 的质量/性能 router。 | 实现中 `nx<=max_degree+1` 仍保留 complete graph；更大但仍属于 fallback 的小组每点连 `max_degree` 条环形边，保持强连通并限制出度。当前只把它写成系统优化和候选 route，不写成 CPU Vamana 的无损替代。 | 需要 graph structure 诊断和多 Lsearch 搜索复测。 | 已限定 claim |
| exact batch 的 point lookup 能否用 GPU 端生成来减少 CPU overhead？ | 早期 10%x40 小 exact 点数实验显示收益不稳；但 x200 `router_exact_nx4096` 证明当 exact 覆盖 120 万点时，CPU lookup + host pack 已经成为主瓶颈。当前结论改为：GPU lookup 与 direct-H2D 必须绑定使用，不能单独评估。 | 旧 x200 path：`direct_h2d=0, device_lookup=0`，group `8999.26 ms`、`pack=4140.61 ms`；新 fill16 path：`direct_h2d=1, device_lookup=1`，group `3204.88 ms`、`pack=0.04 ms`、L1000 recall `0.867` 不变；新默认 path：group `3579.27 ms`、fill `1553.28 ms`。 | 仍需 10%x40/真实多标签在新 direct-H2D 默认下复测；碎片 run 过多时 `UNG_FAST_EXACT_DIRECT_H2D_MAX_RUNS` 可能触发回退。 | 负结论被修正为规模相关；当前默认开启 direct-H2D + GPU lookup |

## Round 3: 普遍替代的系统边界

| Reviewer attack | Author response needed | Current evidence | Missing experiment / implementation | Status |
|---|---|---|---|---|
| 你们一直说 workload-aware route，但它仍保留 CPU fallback 和 Vamana/SearchQueue/std::vector 图结构，这算什么普遍替代？ | 承认当前替代只发生在计算后端和部分 router 上，尚未替代 UNG 的 CPU 输出适配层。把“普遍替代”的下一步定义为 GPU-native flat adjacency / CSR 输出，而不是继续微调单个 kernel。 | x100 adaptive 中只有 `128/5666` 个 group 进入 GPU batch，`5538` 个 fallback CPU group 仍耗约 `2.7~3.0s`；x400 中 `tagore_fill` 可达 `4~8s`，说明 GPU 图结果回填到 `std::vector<IdxType>` 和 Vamana 兼容对象本身是大头。 | 设计 flat adjacency/CSR graph 后端：GPU group graph 和 cross-edge 直接写全局 id 的 `offsets+edges`，search 直接遍历 span，跳过 `SearchQueue -> cross_group_neighbors -> _graph->neighbors` 的二次 host merge。 | 新增设计面 / 待实现 |
| 能不能靠继续优化现有适配层获得大幅性能？ | 必须用负结果约束主张：小优化可以减少开销，但不太可能带来数量级收益；错误的预留/数据结构替换还可能变慢。 | 2026-06-02 临时 A/B：给 GPU graph fill 预留 cross-edge 容量使 x100 `tagore_fill` 从历史 `28.3 ms` 增至 `120 ms+`，已回退。当前保留的低风险改动是 search / Vamana 按引用遍历邻接表、GPU cross-edge 成功路径延迟释放 cache、additional_edges connected-groups epoch marker、严格 id-vector 路径跳过无用 `SearchQueue` 初始化。 | 需要对 flat/CSR 后端做 build+search A/B；现有微优化只能作为代码卫生和适配层减压，不作为论文主贡献。 | 负结果/边界已记录 |
| 如果 x100 full-quality 中 additional_edges 已经比 GPU kernel 慢，是否说明 fused topK 贡献不再重要？ | 不是。fused topK 解决主 cross-edge topK 生成；新的证据说明 full-quality 系统下一步必须把 additional_edges 和输出边界纳入统一优化。 | Amazon 1% x100 smoke：CPU Vamana additional 下 total cross `2439.062 ms`，`additional_edges_ms=1269.257 ms`，GPU kernel event `655.969 ms`；CPU exact additional 把 `additional_edges_ms` 降到 `207.893 ms`，但 total cross 为 `2568.729 ms`，说明简单 CPU exact 不是普遍替代。profiler：`/home/graphdb/Codes/FilterVectorResultsCUDA/prof/ung_prof_20260602_064111_103566.log`、`ung_prof_20260602_064200_103859.log`。 | 实现 GPU exact/additional backend 或 flat/CSR edge buffer，并做 full-quality recall A/B。 | 新增定位结果 / 待实现 |
| compact exact output 已经把 D2H 降下来了，为什么仍不能算普遍替代？ | 把 D2H、fill 和 total group 分开报告。compact-D2H 是必要清理，但 GPU 结果仍要物化到每点一个 CPU-compatible `Graph::neighbors` 对象；NeighborList small-buffer 已降低 allocator 成本，但仍不是 CSR/GraphView。 | x200 `router_exact_nx256` 小 A/B：旧 artifact D2H `91.77 ms`、fill `2121.2 ms`；compact run D2H `18.97 ms`、fill `1576.9 ms`；same-code no-compact D2H `31.78 ms`、fill `1388.15 ms`；compact+fill16+assign D2H `27.37 ms`、fill `1501.8 ms`。后续 NeighborList64 把 x200 auto-reserve fill 降到 `14.7 ms`，但保留 per-node host object。L1000 recall 均约 `0.867~0.869`。 | 需要实现 flat adjacency/CSR graph backend，避免 `_graph->neighbors` object fill；然后做 same-script CPU / packed exact / CSR A/B 和 end-to-end recall。 | 新负/定位结果 |
| direct-H2D 已经把 pack 降到 0，为什么还不能算普遍替代？ | 因为它只消除了 exact batch 输入侧 pack；输出侧仍要把 GPU graph 物化为每点一个 `std::vector`。fill16 的 `1383.53 ms` 和默认 `1553.28 ms` 就是剩余边界。 | x200 direct-H2D A/B：old group `8999.26 ms`，direct-fill16 group `3204.88 ms`，new default group `3579.27 ms`；GPU exact kernel `~930 ms` 基本不变。 | flat adjacency/CSR graph backend；search span view；additional_edges 与 cross-edge 能直接消费 flat graph。 | 新正结果 + 新瓶颈 |
| 既然 offset pass 和 `SearchQueue` 是输出边界，能不能用最小改动就大幅解决？ | 需要分开回答。direct-global 可以去掉 offset pass，但在 NeighborList64 后 offset 只有毫秒级；flat-id writeback 才明显减少 D2H/merge，但当前仍是 skip-additional smoke。 | Amazon 1%x100 skip-additional smoke：local-id + SearchQueue cross `1650.95 ms`、`add_offset=2.5 ms`、`merge_cross=17.2 ms`、D2H `11.1 ms`、recall `0.816`；direct-global + SearchQueue `add_offset=0.0 ms` 但 cross 抖动到 `3530.31 ms`，不能当负/正算法结论；direct-global + flat-id cross `707.83 ms`、D2H `0.6 ms`、`merge_cross=0.5 ms`、recall `0.816`。artifacts：`/home/graphdb/fv_runs/direct_global_off_smoke_20260602_084831`、`/home/graphdb/fv_runs/direct_global_smoke_20260602_084744`、`/home/graphdb/fv_runs/flat_id_direct_global_smoke_20260602_084939`。 | 必须做 full-quality additional_edges A/B，或者实现 GraphView/CSR 后让 additional/cross/group graph 都写 flat segment；direct-global 与 CPU Vamana additional 不兼容，不能默认用于 full-quality CPU repair 路径。 | 新部分正结果 / 边界明确 |
| 那 additional_edges 能否直接边生成边 append 到 `_graph->neighbors`，不用中间容器和 CSR？ | 这个 shortcut 已试过，是负结果。它把问题从中间容器转移到并行修改 CPU-compatible per-node graph 的稳定性和语义边界。 | x400 full-quality direct append 探索：`/home/graphdb/fv_runs/additional_direct_x400_on_20260602_083520` 和 lock-guarded `/home/graphdb/fv_runs/additional_direct_x400_lock_on_20260602_083735` 都在 GPU `batched_search end` 后停止，没有 cross breakdown/Index time，summary 只有表头；lock-guarded env 含 `UNG_ADDITIONAL_DIRECT_APPEND=1`。当前代码已限制 `additional_direct_append` 只在 `additional_edges_impl=CpuExactScan` 下启用，CPU Vamana additional 下会打印 disabled 信息。 | 下一步不是更多锁，而是 staged output：additional/cross/group graph 写独立 flat segment 或 edge buffer，由 GraphView/CSR 在 search 侧统一暴露；若要继续 direct append，必须先用 sanitizer/root-cause 证明稳定性。 | 新负结果 / 支持 CSR |
| 如果未来改成 CSR，会不会改变查询语义？ | 作者需要给出等价边界：只改变邻接存储和写回路径，不改变 entry points、visited set、distance 计算、Lsearch 和 additional_edges 语义；如果 additional_edges 改成 GPU exact，则必须单独做 full-quality recall A/B。 | 当前 search 主循环只需要“给定点 id 取得邻接列表”这一抽象；`_graph->neighbors[cur.id]` 已经可以替换为 span-like view。 | 引入 GraphView 接口或 flat graph adapter，并同时支持旧 `std::vector` 图用于回归。 | 设计待落地 |
| 能不能靠更细的 CUDA kernel 改写解决 packed exact-anchor 的性能问题？ | 必须以同机 A/B 验证，而不是假设 warp 级实现一定更快。 | 新增 `UNG_FAST_EXACT_WARP_KERNEL=1` 负结果。Amazon 1% x200 router exact256：旧 block exact group `4031.36 ms`、kernel `841.136 ms`；warp exact group `4334.28 ms`、kernel `987.27 ms`，recall 同为 `0.867`。fill 线程扫也为负：16 线程 fill `1682.11 ms` 优于 8 线程 `2113.86 ms` 和 64 线程 `2321.96 ms`。 | 下一步不是继续 kernel/线程微调，而是 flat adjacency/CSR 输出、减少 host allocator、把 cross/additional/group graph 写回统一到 flat buffer。 | 新负结果 / 支持架构重构 |
| reserve 已经把很多 fill/merge 时间降下来了，能否把它作为输出边界最终解？ | 不能。reserve 是证明 `std::vector` 输出边界真实存在的低风险止血，但它仍保留 per-node host vector 图，且 reserve 本身有秒级成本。 | 10%x40 reserve A/B：Index `39746.1 -> 24704.5 ms`，tagore fill `823.5 -> 23.9 ms`，fallback wall `11844.6 -> 520.2 ms`，cross merge `1677.1 -> 2.3 ms`，但 reserve 花 `6487.3 ms`。x200 final auto-reserve：`enabled=1 mode=auto`，capacity `53,011,200`，fill `224.8 ms`，cross merge `0.9 ms`。 | GraphView/CSR；小组 implicit view；additional_edges flat segment；search span-view recall A/B。 | 新正结果 / 但仍支持 CSR 重构 |
| 既然 `SearchQueue` / host merge 是瓶颈，为什么不直接在 GPU 上 global merge，一次 D2H 后写回？ | 必须实测，而不是按直觉宣称“全 GPU merge 必然更快”。简单 qid lock global merge 会把 host 写回变少，但可能把锁竞争转移到 kernel。 | Amazon 10%x40 full-quality 负结果：历史 split double-buffer cross `9451.1 ms`，GPU event 为 H2D `2262.2 ms`、kernel `2952.2 ms`、D2H `87.1 ms`。实验性 `UNG_GPU_DB_NOSPLIT=1 + UNG_GPU_DB_GLOBAL_MERGE=1 + UNG_GPU_ID_VECTOR_WRITEBACK=1 + UNG_GPU_DB_LARGE_MAX_NX=16384` 只保留一条全量 double-buffer 汇总，D2H/writeback 降到 `3.5/33.2 ms`，但 kernel 升到 `4834.3 ms`，cross 总时间变成 `11598.0 ms`，比历史更慢。 | 下一版应做 qid-sharded/segmented topK merge，避免 per-qid atomic lock；当前 qid-lock global merge 只能作为 negative ablation，不进默认路径。 | 已回答为负结果 |
| target-centric fused 为什么还有大幅优化空间？是不是很多组根本没走快路径？ | 这个攻击成立。旧 double-buffer direct-qid path 是 all-or-nothing：一个 batch 内只要混入少数 `nx > DB_LARGE_MAX_NX` 大组，就会让大量本可 double-buffer 的组一起回退老路径。 | 新实现加入 partial double-buffer routing：supported groups 先走 direct-qid double-buffer，unsupported groups 单独回退。Amazon 1% x200 full-quality 中，`5173/5195` 个 target groups 进入 double-buffer，仅 `22` 个回退；cross `3900.81 -> 2657.23 ms`，kernel `2095.8 -> 1636.0 ms`。repeat=3 L1000/L5000 为 `0.866/0.907`，旧 target-current 为 `0.858/0.9068`，source-centric 为 `0.871/0.911`。artifact：`/home/graphdb/fv_runs/target_fused_x200_db_split_20260602_060034/end_to_end_recall_ab_20260602_060034`。负对照 `DB_NOSPLIT=1` all-or-nothing cross `4332.46 ms`。 | 仍需 x100/10%x40/x400 复测；当前只在 SearchQueue 写回路径启用，id-vector/no-lock merge 仍需单独设计。 | 新正结果 / 调度修复 |
| qid-lock global merge 是负结果，那你们有什么真正更普遍的 cross-edge 替代路线？ | 需要从遍历方向上消除 qid 冲突，而不是在 target-centric 输出上加锁。source-centric 按 `out_neighbors[source]` 遍历，让每个 qid 只由所属 source group 处理一次，并在 kernel 内跨所有目标 segment 维护最终 topK。 | 新增 `UNG_GPU_SOURCE_EXACT=1`，`UNG_GPU_SOURCE_EXACT_MODE=1`。SIFT30 skip-additional smoke test：source-centric CUDA-core exact cross `5981.0 ms`、kernel `5524.0 ms`；当前 target-centric fused cross `4918.9 ms`、kernel `2189.9 ms`；source-centric TF32 WMMA 同步修复后 cross `3021.9 ms`、kernel `1402.9 ms`、source path 自身 `1763.9 ms`。随后发现 WMMA B tile 是 block-shared，不能让同一 block 的 warp 混跑不同 source group；加入 source-block padding 后 cross `1927.7 ms`、kernel `1418.5 ms`、source path 自身 `1710.2 ms`，`padded_tiles=371199`。 | 仍需 Amazon 1%x40/x100/x200/x400 full-quality A/B 和 end-to-end recall；TF32 topK 次序也需和 FP32 exact 对照。当前只能写成新设计方向和初步 smoke test，不是最终主表。 | 新正向候选 / 待 full-quality 验证 |
| 审稿人追问：source-centric WMMA 旧 smoke 是否存在 block 内 shared memory 混排 bug？ | 这个质疑成立，旧路径的调度允许同一 block 的不同 warp 处理不同 source group，而 B tile 是 block 级 shared memory。修复方式不是加锁，而是在 host descriptor 生成时对每个 source group 的 16-row tile 数按 `warps_per_block` 补齐，让一个 block 只覆盖一个 source group。 | 修复后 SIFT30 skip-additional：`/home/graphdb/fv_runs/source_wmma_padded_sift30_skipadd_20260602_053546`，cross `1927.7 ms`、kernel `1418.5 ms`、D2H `1.9 ms`、writeback `26.8 ms`；pageable prepare A/B `/home/graphdb/fv_runs/source_wmma_pageable_sift30_skipadd_20260602_053837`，cross `1848.4 ms`、kernel `1420.1 ms`。 | 这只是修复调度正确性和 smoke 性能；仍需 full-quality recall 和 TF32/FP32 consistency。 | 已修复实现风险 / 仍非主表 |
| 审稿人追问：SIFT30 skip-additional 快，不代表 Amazon full-quality 快；有没有反例？ | 有。我们补了 Amazon 1% x200 coverage 的 full-quality 最小闭环。source-centric 可以跑通 CPU Vamana additional_edges 和 search，但性能不是 current best。 | source w8：`/home/graphdb/fv_runs/source_wmma_amazon_x200_fulladd_fix_20260602_054536`，cross `4693.4 ms`、kernel `3224.6 ms`、L1000/L5000 repeat=3 `0.871/0.911`；source w4：`/home/graphdb/fv_runs/source_wmma_amazon_x200_fulladd_w4_20260602_054745`，cross `4803.07 ms`、kernel `3091.9 ms`、repeat=3 `0.871/0.911`；同机 target fused：`/home/graphdb/fv_runs/target_fused_amazon_x200_fulladd_current_20260602_054639`，cross `3900.81 ms`、kernel `2095.8 ms`、repeat=3 `0.858/0.9068`。 | 性能方向已经说明 source-centric 不能作为当前主表。target current recall 低于历史 x200，需要结构诊断；不能把 source 的更高 recall 直接当成质量贡献。需要 source kernel tiling、full additional GPU 化和 repeat=3/更多数据集。 | 负/部分 A/B，防 overclaim |
| 审稿人追问：这轮 x200 source full-quality 曾经 segfault，是否稳定？ | 崩溃原因不是 WMMA kernel，而是我们把 `SearchCacheList` 懒创建后，在 `additional_edges=cpu_vamana` 的 OpenMP parallel 区域中首次初始化，造成多线程竞争同一个 `unique_ptr`。已改为进入 additional_edges 并行循环前单线程预创建 cache。 | 崩溃 run `/home/graphdb/fv_runs/source_wmma_amazon_x200_fulladd_20260602_054350` 在 source_exact 完成后 segfault；修复后 `/home/graphdb/fv_runs/source_wmma_amazon_x200_fulladd_fix_20260602_054536` 完成 build+search。 | 仍需 sanitizers 或更多数据集压力测试；但当前 crash root cause 已定位并修复。 | 实现修复，不作为论文贡献 |

这轮 reviewer 结论是：当前论文仍应把“普遍替代”写成 **workload-aware GPU backend with explicit CPU-compatible output boundary**。source-centric no-lock WMMA 说明 cross-edge 的遍历方向可以进一步减少 qid 冲突和 host merge，但完整系统仍需要 full-quality recall A/B 和 flat adjacency/CSR 输出路径；否则 GPU kernel 再快也会被 CPU 适配层 Amdahl 限制。

## 第一优先级实验: End-to-End Search Recall A/B

这个实验优先级最高，因为它决定论文能否把 FastGrnndCuda 放进主贡献，而不是只作为附录工程尝试。

### 固定变量

| 变量 | 要求 |
|---|---|
| Query set | 同一份 query vectors / query labels；如果启用 `is_ung_more_entry`，还必须固定 query source groups |
| Ground truth | 同一份 containment GT |
| Search params | 相同 `K`, `Lsearch`, `num_entry_points`, `efs` 参数 |
| Cross-edge | 建议先固定 `UNG_GPU_TOPK_IMPL=3`，避免 group graph 和 cross-edge 同时变化 |
| Threads | build 和 search 均记录线程数 |

### A/B 组

| Variant | Purpose | Environment |
|---|---|---|
| `cpu_vamana_group` | group graph 质量 baseline | `UNG_GROUP_GRAPH_IMPL=0`, `UNG_CROSS_EDGE_IMPL=1`, `UNG_GPU_TOPK_IMPL=3` |
| `fastgrnnd_cpu_fallback` | 当前最快工程配置 | `UNG_GROUP_GRAPH_IMPL=3`, `UNG_TAGORE_MIN_GROUP_SIZE={128/256}`, `UNG_CROSS_EDGE_IMPL=1`, `UNG_GPU_TOPK_IMPL=3` |
| `fastgrnnd_complete_fallback` | fallback policy ablation | `UNG_GROUP_GRAPH_IMPL=3`, `UNG_TAGORE_FALLBACK_IMPL=0` |

### 输出表

| Dataset | Variant | Build Index ms | Group ms | Cross ms | Lsearch | Avg recall | Avg time ms | P95 query ms | Notes |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| Amazon 1% x100 | CPU Vamana group + full additional | `6863.47` | `3303.26` | `1732.42` | `100/500/1000` | `0.825/0.868/0.890667` | `405.3/388.1/390.4` | `162.1/163.1/162.4` | full-quality, repeat=3 |
| Amazon 1% x100 | FastGrnndCuda + CPU fallback + full additional | `6370.67` | `3216.21` | `1871.51` | `100/500/1000` | `0.828/0.871/0.896` | `422.3/381.0/392.5` | `174.5/159.3/163.5` | full-quality, repeat=3；质量闭环正结果，group speedup 仅 `1.03x` |
| Amazon 1% x100 | adaptive_cuda conservative + full additional | `5450` | `2773.35` | `1513.7` | `100/500/1000` | `0.829/0.870/0.895` | `716.9/705.8/709.4` | - | full-quality, repeat=3；build 更快，search latency 需要同脚本重测/解释 |
| Amazon 10% x40 | CPU Vamana group + full additional | `36053.3` | `18218.5` | `9777.3` | `100/500/1000` | `0.8647/0.898/0.908` | `791.6/705.2/735.9` | `254.8/244.5/244.2` | full-quality, repeat=3 |
| Amazon 10% x40 | FastGrnndCuda + complete fallback + full additional | `23497.4` | `5147.2` | `9451.1` | `100/500/1000` | `0.8668/0.898/0.9103` | `728.6/656.9/694.7` | `226.4/201.2/231.3` | full-quality, repeat=3 |
| Amazon 1% x400 gather-Q | CPU Vamana group + full additional | `37807.6` | `17537.3` | `12296.6` | `1000/5000` | `0.9460/0.9660` | `306.5/285.3` | `121.4/75.6` | full-quality, repeat=3 |
| Amazon 1% x400 gather-Q | FastGrnndCuda heavy prune + full additional | `52552.2` | `29192.3` | `12815.3` | `1000/5000` | `0.9443/0.9643` | `269.8/279.1` | `96.2/64.1` | 质量接近但 build 更慢 |
| Amazon 1% x400 gather-Q | FastGrnndCuda light512 + full additional | `30826.5` | `10362.9` | `12681.6` | `1000/5000` | `0.9379/0.9579` | `278.6/277.6` | `102.4/62.5` | build 更快但 recall drop `0.0081` |

`search_UNG_index` 已输出：

```text
search_time_summary.csv
query_details_repeat<N>.csv
```

其中 `search_time_summary.csv` 给出每个 `Lsearch` 的平均 recall 和平均耗时；`query_details` 可计算 P50/P95/P99 和每 query recall 分布。

推荐入口脚本：

```bash
scripts/benchmarks/run_end_to_end_recall_ab.sh
```

## 已有部分证据: Amazon PF Cross-Edge Search Sanity

已有 `fv_runs/amazon_*_search_pf_r3` 结果可先回答 cross-edge 后端切换是否破坏 query recall。结论是：在 group graph 都保持 CPU Vamana 的条件下，existing、naive GPU cross、paper fused cross 的 recall 基本一致。

| Variant | Lsearch | Avg recall | Batch time ms | P95 query ms |
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

限制：这些 run 的 group graph 没有切到 FastGrnndCuda，因此它们只能支持 cross-edge 后端的 query-level sanity，不能证明 FastGrnndCuda 的端到端 recall。`query_source_groups` 虽然缺失，但默认 `search_UNG_index --is_ung_more_entry=false`，该文件不会参与当前搜索路径。

详细整理见：

```text
docs/reports/END_TO_END_RECALL_STATUS_CN.md
```

## 新增证据: 原始 Amazon PF Group Graph A/B

带 CPU Vamana additional edges 的 A/B：

| Variant | Index ms | Group ms | Cross ms | L1000 recall | L2000 recall | L5000 recall |
|---|---:|---:|---:|---:|---:|---:|
| CPU Vamana group | 43376.6 | 5776.25 | 20194.7 | 0.912333 | 0.932067 | 0.971767 |
| FastGrnndCuda + CPU fallback | 36732.0 | 1049.92 | 18847.4 | 0.911700 | 0.931500 | 0.971133 |

结论：

- FastGrnndCuda hybrid 在原始 Amazon PF 上首次拿到了端到端 search recall 证据。
- group graph speedup `5.50x`，Index speedup `1.18x`。
- 三个 Lsearch 的 recall drop 都小于 `0.001`。
- 该实验不是 Amazon repeat x100/x400，因此仍需正式补测；`query_source_groups` 当前不参与搜索路径，但 artifact 复现时最好补齐。
- Index speedup 只有 `1.18x` 的原因是剩余阶段占比很高：FastGrnndCuda 后 group graph 只占 `2.86%`，cross 占 `51.31%`，LNG 占 `32.71%`。

## 新增证据: SIFT30 Group Graph 端到端 A/B

这组实验回答“结果是否只对 Amazon 有效”。它固定修复后的 fused cross-edge 和 CPU Vamana additional edges，只切换 group graph 后端。

日志：

```text
/home/graphdb/fv_runs/debug_sift30_fused_cross_exactdist_cpu_20260601_224812
/home/graphdb/fv_runs/debug_sift30_fused_cross_exactdist_fast_20260601_225006
```

| Variant | Index ms | Group ms | Cross ms | L100 recall | L200 recall | L500 recall | L1000 recall |
|---|---:|---:|---:|---:|---:|---:|---:|
| CPU Vamana group | 31740.4 | 11245.3 | 17886.2 | 0.609423 | 0.706580 | 0.814600 | 0.877587 |
| FastGrnndCuda + CPU fallback | 24989.9 | 1889.79 | 17868.9 | 0.607997 | 0.706287 | 0.812680 | 0.876297 |

结论：

- SIFT30 上 group graph speedup `5.95x`，Index speedup `1.27x`。
- L100/L200/L500/L1000 recall drop 分别为 `0.001426` / `0.000293` / `0.001920` / `0.001290`。
- 修复后的 fused cross-edge 将 SIFT30 cross 从 CPU Vamana cross 的 `69834.5 ms` 降到 `17886.2 ms`，约 `3.90x`。
- 这轮 reviewer 迭代发现并修复了两个关键语义 bug：`DIRECT_QID_ALL_FUSED` 不能覆盖大组 fallback path 时必须 materialize Q；host merge 路径不能使用 id-only writeback，否则跨 target-group 排序错误。

## 新增证据: Amazon 1% x200 Coverage Query 与 Light Prune 修复

这组实验专门回答“Amazon repeat-scale 上 FastGrnndCuda 是否仍然成立”。第一轮结论是否定的：旧重 prune 在 `nx≈200` 这个 bucket 上性能明显输给 CPU Vamana。随后我们加入 `UNG_FAST_GRNND_LIGHT_PRUNE_NX=256` 和默认 `UNG_FAST_GRNND_LIGHT_HEAD=24`，对中等组保留近邻头部，并从候选尾部采样少量多样性邻居，避免 reverse candidate + RNG occlusion 的重剪枝成本。

日志：

```text
/home/graphdb/fv_runs/reviewer_e2e_amazon_1pct_x200_coverage_20260601_230224
```

query/GT：

```text
/home/graphdb/FilterVectorBenchData/fv_amazon_1pct_x200_jitter_nonempty/query_coverage_1000
/home/graphdb/FilterVectorBenchData/fv_amazon_1pct_x200_jitter_nonempty/query_coverage_1000_gt/gt_K10_containment.bin
```

coverage query 分布：

| Metric | Value |
|---|---:|
| queries | `1000` |
| broad / medium / narrow | `316 / 420 / 264` |
| matched groups avg / p50 / p95 | `258.97 / 32 / 1087.4` |
| matched points avg / p50 / p95 | `55556.6 / 6800 / 244500` |
| GT time | `8733 ms` |

A/B 结果：

| Variant | Index ms | Group ms | Cross ms | L100 | L200 | L500 | L1000 |
|---|---:|---:|---:|---:|---:|---:|---:|
| CPU Vamana group | `17596.8` | `8511.72` | `4601.91` | `0.8230` | `0.8250` | `0.8490` | `0.8690` |
| FastGrnndCuda, old heavy prune | `38273.7` | `26620.6` | `6055.72` | `0.8125` | `0.8273` | `0.8472` | `0.8662` |
| FastGrnndCuda, top32 light prune | `14211.8` | `6016.04` | `4926.37` | `0.8135` | `0.8239` | `0.8421` | `0.8618` |
| FastGrnndCuda, diversified light prune | `16478.5` | `6706.36` | `4857.33` | - | - | - | `0.8656` |
| old batched exact kNN, confounded | `12595.9` | `4356.18` | `4120.75` | - | - | - | `0.8159` |
| packed exact-anchor router, nx4096 | `11043.5` | `3827.28` | `4868.15` | - | - | - | `0.8690` |

FastGrnndCuda build breakdown：

| Variant | direct build | H2D | GNN | prune | D2H |
|---|---:|---:|---:|---:|---:|
| old heavy prune | `26282 ms` | `1538.32` | `3551.54` | `19218.3` | `413.047` |
| top32 light prune | `5746.81 ms` | `1180.47` | `3198.35` | `712.392` | `134.631` |
| diversified light prune | `6365.69 ms` | `1448.82` | `3229.56` | `759.66` | `161.408` |
| old batched exact kNN, confounded | `3950.64 ms` | `524.173` | `674.693` | `0` | `29.8523` |
| packed exact-anchor router, nx4096 | `2899.6 ms` | `347.710` | `925.721` | `0` | `29.742` |

高 Lsearch 复测：

| Variant | L1000 | L2000 | L5000 |
|---|---:|---:|---:|
| CPU Vamana group | `0.8690` | `0.8820` | `0.9080` |
| FastGrnndCuda, top32 light prune | `0.8619` | `0.8769` | `0.9029` |
| FastGrnndCuda, diversified light prune | `0.8656` | - | `0.9056` |
| old batched exact kNN, confounded | `0.8159` | - | `0.8199` |
| packed exact-anchor router, nx4096 | `0.8690` | - | `0.9080` |

Reviewer 结论：

- 旧重 prune 的坏点不能忽略。x200 repeat-scale 是我们自己构造来放大组内计算的目标 workload，它说明 reverse candidate + RNG occlusion 在 `nx≈200` 上不划算。
- diversified light prune 把 prune 从 `19218.3 ms` 降到 `759.66 ms`，约 `25.3x`；group graph 从 `26620.6 ms` 降到 `6706.36 ms`，约 `3.97x`；Index 从 `38273.7 ms` 降到 `16478.5 ms`，约 `2.32x`。
- 与 CPU Vamana 相比，diversified light prune 已经实现 group `1.27x`、Index `1.07x` 加速，L5000 只低 `0.0024`。这说明“普遍替代”的下一步不是回退 CPU，而是增强 light prune 的质量和降低外围抖动。
- old batched exact kNN 是 confounded 负结果：它提醒我们必须固定 full-quality cross/additional_edges 口径；新 packed exact-anchor 已证明 exact-anchor 图在 x200/x400 coverage-query 上可以达到 CPU 级 L5000 recall。

## 新增实验: Mixed Exact/GNN Router A/B

这个实验专门回答 reviewer 对“普遍替代”的两个攻击：

1. 旧 pure exact kNN 低 recall 已被重新归因，为什么还需要 GNN/reverse-tail router？
2. 如果 FastGrnndCuda 性能不稳定，是否只是因为系统层面没有把小/中组正确 batch 起来？

新增脚本：

```bash
scripts/benchmarks/run_group_graph_router_ab.sh
```

默认配置：

```bash
DATA_DIR=/home/graphdb/FilterVectorBenchData/fv_amazon_1pct_x200_jitter_nonempty
QUERY_DIR_NAME=query_coverage_1000
ROUTER_THRESHOLDS="0 128 256 512"
VARIANT=fastgrnnd_cpu_fallback
UNG_ADDITIONAL_EDGES_IMPL=0
UNG_CROSS_EDGE_IMPL=1
UNG_GPU_TOPK_IMPL=3
```

输出：

```text
<out_root>/summary.csv
<out_root>/cpu_vamana/summary.csv
<out_root>/router_exact_nx*/summary.csv
```

判据：

| 结果 | 论文解释 |
|---|---|
| build/index 明显下降，recall 接近 `router_exact_nx0` 和 CPU Vamana | mixed router 可以作为质量感知系统优化写入主方法或 ablation |
| build 下降但 recall 低于 CPU | 只能写成 partial/conditional；exact 只能用于更小阈值或由 GNN/reverse-tail 兜底 |
| build 无明显改善 | 说明瓶颈不在小/中组 batch exact，应继续优化 Tagore/GNN batch 或 host fill |

实测结果目录：

```text
/home/graphdb/fv_runs/group_graph_router_ab_streamfill_20260602_030037
```

运行配置：Amazon 1% x200 coverage-query，`NUM_REPEATS=1`，`LSEARCH_VALUES="100 500 1000"`，`UNG_TAGORE_BATCH_STREAMS=2`，`UNG_TAGORE_FILL_FAST=1`，`UNG_TAGORE_FILL_THREADS=16`。

关键结果：

| Case | Index | Group | Cross | L100 | L500 | L1000 |
|---|---:|---:|---:|---:|---:|---:|
| CPU Vamana | `13473.7` | `7812.0` | `3590.44` | `0.8230` | `0.8490` | `0.8690` |
| router exact nx0 | `16179.7` | `8606.52` | `4935.45` | `0.8180` | `0.8463` | `0.8661` |
| router exact nx128 | `15731.2` | `8073.15` | `4946.96` | `0.8178` | `0.8464` | `0.8662` |
| router exact nx256 | `14098.0` | `5770.88` | `5002.54` | `0.8229` | `0.8510` | `0.8690` |
| router exact nx512 | `13937.2` | `5840.47` | `5057.78` | `0.8230` | `0.8500` | `0.8700` |

解释：`nx256/512` 没有质量崩塌，packed exact-anchor `nx4096` 进一步在 repeat=3 下达到 x200 L5000 `0.908`。论文中可以写“packed exact-anchor 是当前最强候选”，但不能写“router 已在所有分布上解决普遍替代”。

新增 packed-buffer 快速点：

| Case | Index | Group | Cross | L1000 |
|---|---:|---:|---:|---:|
| x200 packed exact-buffer nx4096 | `11043.5` | `3827.28` | `4868.15` | `0.869` |
| x400 packed exact-buffer nx4096 | `28436.3` | `11588.4` | `12110.9` | `0.945` |

解释：这组结果已经补到 L5000/repeat，回答了 x200/x400 coverage-query 上的质量问题。它支持把 packed exact-anchor 写成强候选，但还需要 x100、10%x40 和真实多标签来证明泛化。

## 新增进展: Amazon 10% x40 Coverage Query 与 Direct-QID Cross-Edge 修复

Reviewer 原本会要求补 `Amazon 10% x40`，因为它代表“组数很多、单组较小”的压力场景。当前已经准备好 balanced coverage query 和 GT；第一轮 full-quality build 暴露了 `target_groups≈5万` 时 cross-edge prepare/scheduling 的外围瓶颈，随后通过 direct-pageable upload 和 direct-qid kernel path 做了第一轮修复。

数据与 query：

```text
/home/graphdb/FilterVectorBenchData/fv_amazon_10pct_x40_jitter_nonempty
/home/graphdb/FilterVectorBenchData/fv_amazon_10pct_x40_jitter_nonempty/query_coverage_1000_balanced
/home/graphdb/FilterVectorBenchData/fv_amazon_10pct_x40_jitter_nonempty/query_coverage_1000_balanced_gt/gt_K10_containment.bin
```

query 生成方式：

```bash
python3 tools/benchmarks/generate_coverage_queries_from_labels.py \
  --base-bin .../Amazon_10pct_x40_base.bin \
  --label-file .../Amazon_10pct_x40_labels.txt \
  --out-dir .../query_coverage_1000_balanced \
  --num-queries 1000 --query-length 2 --seed 456 \
  --profile-mix 'narrow:8,medium:12,broad:1'
```

query 分布：

| Metric | Value |
|---|---:|
| queries | `1000` |
| medium / narrow / broad | `540 / 238 / 222` |
| matched groups avg / p50 / p95 | `883.09 / 25.5 / 3297.0` |
| matched points avg / p50 / p95 | `39910.08 / 1060.0 / 138738.0` |
| GT compute | `16076 ms` exact compute, `26.24 s` wall including data load |

第一轮 full-quality CPU Vamana group + fused cross build 尝试：

```text
/home/graphdb/fv_runs/reviewer_e2e_amazon_10pct_x40_coverage_20260602_000502
```

已完成阶段：

| Stage | Result |
|---|---:|
| data load | `9206 ms` |
| groups | `53840` |
| prepare_group_storages_graphs | `7808.48 ms` |
| group graph | `26772 ms` |
| complete groups / points | `52543 / 2101720` |
| CPU Vamana groups / points | `1297 / 308080` |
| vector-attribute graph | `1497.72 ms` |
| LNG | `639.8 ms` |
| descendants | `4.7 ms` |
| coverage | `378.9 ms` |
| cross-edge target groups | `49374` |

停止原因：进入 `Building cross-group edges ... [cross_edges] backend=GPU, target_groups=49374` 后超过 10 分钟没有打印 `[GPU GEMM] H2D/Kernel/D2H` event，也没有生成 `build_time.csv`。后续 profile 发现至少有两个问题：

- `prepare_all` 虽然打印 `direct_pageable=1`，实际仍走 `pinned_staging`，导致 7.4GB base vectors 在 CPU 侧又 repack 一遍，旧 probe 中 `host_pack_ms=6205.6`、`prepare_all total=6500.7 ms`。
- double-buffer 路径内每个 chunk 仍 gather query vectors 到 `d_Q` 并重新计算 `q_norm`，没有充分利用全量 `g_d_all_X/g_d_all_norm` 常驻缓存。

修复内容：

- 对大于 `UNG_GPU_PREPARE_HOSTREG_MAX_MB` 的连续 base storage，真正走 direct pageable H2D，避免全量 CPU repack。
- double-buffer 默认开启。
- medium/TF32 tile fused kernel 增加 direct-qid 模式，chunk 内只上传 `qid` 和 descriptors，kernel 直接从 `g_d_all_X/g_d_all_norm` 读取 query 和 norm，去掉 `d_Q` gather 与 per-chunk qnorm。

修复后最新 run：

```text
/home/graphdb/fv_runs/reviewer_e2e_amazon_10pct_x40_cross_directqid_probe_20260602_003614
/home/graphdb/fv_runs/reviewer_e2e_amazon_10pct_x40_cpu_exact_skipadd_20260602_003822
/home/graphdb/fv_runs/reviewer_e2e_amazon_10pct_x40_fastgrnnd_directqid_skipadd_20260602_004109
```

先前为了验证 cross-edge 和 group graph 替换能力，三组使用 `UNG_ADDITIONAL_EDGES_IMPL=1` 跳过 additional edges；这些只能作为构建性能和 search sanity，不是完整质量结果。

| Variant | Index | Group graph | Cross-edge | Cross H2D | Cross kernel | Cross D2H | L100 sanity |
|---|---:|---:|---:|---:|---:|---:|---:|
| CPU exact cross, CPU Vamana group | `89529.3 ms` | `26083.3 ms` | `54879.3 ms` | - | - | - | - |
| GPU direct-qid cross, CPU Vamana group | `42651.1 ms` | `22747.6 ms` | `8475.4 ms` | `2534.0` | `2669.4` | `83.1` | `0.800` |
| GPU direct-qid cross, FastGrnndCuda + complete fallback | `26422.1 ms` | `7689.0 ms` | `8869.1 ms` | `3297.0` | `2691.5` | `84.7` | `0.800` |

关键加速：

- cross-edge: `54879.3 / 8475.4 = 6.47x`，相对 128 线程 CPU exact scan。
- CPU Vamana group + GPU cross: `89529.3 / 42651.1 = 2.10x` end-to-end index。
- FastGrnndCuda + GPU cross: `89529.3 / 26422.1 = 3.39x` end-to-end index。
- FastGrnndCuda group graph: `26083.3 / 7689.0 = 3.39x`，其中 GPU 直接构建 `2318.57 ms`，complete fallback `5271.25 ms`。

prepare 修复效果：

| Path | prepare_all | host_pack | full H2D |
|---|---:|---:|---:|
| old probe, incorrectly pinned staging | `6500.7 ms` | `6205.6 ms` | `292.2 ms` |
| fixed direct pageable, CPU group run | `2596.3 ms` | `0.0 ms` | `2454.8 ms` |
| fixed direct pageable, FastGrnndCuda run | `3222.4 ms` | `0.0 ms` | `3221.3 ms` |

Reviewer 结论：

- `10%x40` 证明了“普遍替代”不能只优化单个 GPU kernel；必须同时解决全量数据常驻、query id 展开、descriptor scheduling、结果写回和 group graph fallback。
- direct-qid fused 对大量小/中组场景已经能相对 CPU exact scan 获得 `6.47x` cross-edge 加速，但 `prepare_all` 仍是 2.6-3.2s 级别，必须在表中单独列出。
- FastGrnndCuda + complete fallback 在 `10%x40` 上把端到端 Index 从 CPU exact baseline 的 `89.5s` 降到 `26.4s`，但该结果跳过 additional edges；它证明性能潜力，不证明完整质量。
- full-quality `UNG_ADDITIONAL_EDGES_IMPL=0` A/B 已补，见下表；它证明 additional edges 能把 L100 从 skip 的 `0.800` 提到约 `0.865+`，因此论文主表必须使用 full-quality 口径。
- 还需要补 x100/x400，因为 x100 历史 build 表显示混合策略有收益，x400 历史表显示 prune 也很重；这两个点决定 light/strong prune 阈值。

full-quality 10%x40 A/B：

```text
/home/graphdb/fv_runs/reviewer_e2e_amazon_10pct_x40_cpu_group_directqid_fulladd_20260602_005156
/home/graphdb/fv_runs/reviewer_e2e_amazon_10pct_x40_fastgrnnd_directqid_fulladd_20260602_005446
```

| Variant | Index | Group graph | Cross-edge | prepare_all | GPU kernel events | L100 | L500 | L1000 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| CPU Vamana group + GPU direct-qid cross + CPU additional_edges | `36053.3 ms` | `18218.5 ms` | `9777.3 ms` | `2740.8` | `2925.7` | `0.8647` | `0.8980` | `0.9080` |
| FastGrnndCuda + complete fallback + GPU direct-qid cross + CPU additional_edges | `23497.4 ms` | `5147.2 ms` | `9451.1 ms` | `2206.5` | `2952.2` | `0.8668` | `0.8980` | `0.9103` |

full-quality 加速与质量结论：

- Index speedup: `36053.3 / 23497.4 = 1.53x`。
- Group graph speedup: `18218.5 / 5147.2 = 3.54x`。
- Cross-edge 基本相同：`9777.3 / 9451.1 = 1.03x`，说明 A/B 主要检验 group graph 替换。
- Recall 没有下降：L100 `+0.0021`，L500 持平，L1000 `+0.0023`。这是目前 10%x40 上最强的“可替代”证据。

repeat=3 per-query latency 分布，口径来自 `query_details_repeat3.csv` 的 `Time_ms`，不同于整批 1000 query 的 `Average_Time_ms`：

| Variant | Lsearch | P50 query ms | P95 query ms | P99 query ms | recall=0 ratio |
|---|---:|---:|---:|---:|---:|
| CPU Vamana group | 100 | `48.6` | `254.8` | `499.3` | `13.53%` |
| FastGrnndCuda | 100 | `45.5` | `226.4` | `458.3` | `13.30%` |
| CPU Vamana group | 500 | `41.1` | `244.5` | `515.3` | `10.20%` |
| FastGrnndCuda | 500 | `41.9` | `201.2` | `430.7` | `10.20%` |
| CPU Vamana group | 1000 | `41.8` | `244.2` | `503.6` | `9.20%` |
| FastGrnndCuda | 1000 | `41.5` | `231.3` | `494.0` | `8.97%` |

作者当前回应：

> Amazon 1% x200 coverage-query A/B 表明，FastGrnndCuda 的失败主要来自中等组上的重 prune；Amazon 10% x40 则说明大量 group 场景下 cross-edge 外围会成为新瓶颈。修复 direct-pageable 和 direct-qid 后，10%x40 full-quality A/B 已把 Index 从 `36.1s` 降到 `23.5s`，且 L100/L500/L1000 recall 没有下降。我们将最终系统设计调整为 adaptive-pruning GPU backend + resident direct-qid cross-edge：中等组使用 diversified light prune，大组使用 strong prune，cross-edge 直接消费全局 qid，同时保留 additional_edges 作为完整质量路径。

## 新增证据: Amazon 1% x400 Gather-Q Full-Quality A/B

Reviewer 会追问 x400，因为它同时覆盖“大组计算量足够大”和“direct-qid 大规模不稳定”两个问题。本轮补了稳定 gather-Q fused cross-edge 的 full-quality A/B，固定 `UNG_ADDITIONAL_EDGES_IMPL=0`，即 CPU Vamana additional edges；search 为 repeat=3，Lsearch=`1000/5000`。

数据与 query：

```text
/home/graphdb/FilterVectorBenchData/fv_amazon_1pct_x400_jitter_nonempty
/home/graphdb/FilterVectorBenchData/fv_amazon_1pct_x400_jitter_nonempty/query_coverage_1000
/home/graphdb/FilterVectorBenchData/fv_amazon_1pct_x400_jitter_nonempty/query_coverage_1000_gt/gt_K10_containment.bin
```

query 分布：

| Metric | Value |
|---|---:|
| queries | `1000` |
| medium / narrow / broad | `672 / 258 / 70` |
| matched groups avg / p50 / p95 | `73.37 / 21.0 / 134.2` |
| matched points avg / p50 / p95 | `31016.0 / 8400.0 / 53700.0` |
| GT exact compute | `14039 ms` |

日志：

```text
/home/graphdb/fv_runs/reviewer_e2e_amazon_1pct_x400_gatherq_fulladd_20260602_010839
/home/graphdb/fv_runs/reviewer_e2e_amazon_1pct_x400_light512_gatherq_fulladd_20260602_011323
```

构建与查询结果：

| Variant | Index | Group graph | Cross-edge | Prune | L1000 | L5000 | P95 L1000 | P95 L5000 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| CPU Vamana group + gather-Q cross + CPU additional_edges | `37807.6 ms` | `17537.3 ms` | `12296.6 ms` | - | `0.9460` | `0.9660` | `121.4 ms` | `75.6 ms` |
| FastGrnndCuda heavy prune + gather-Q cross + CPU additional_edges | `52552.2 ms` | `29192.3 ms` | `12815.3 ms` | `19977.7 ms` | `0.9443` | `0.9643` | `96.2 ms` | `64.1 ms` |
| FastGrnndCuda light512 + gather-Q cross + CPU additional_edges | `30826.5 ms` | `10362.9 ms` | `12681.6 ms` | `1022.45 ms` | `0.9379` | `0.9579` | `102.4 ms` | `62.5 ms` |
| FastGrnndCuda light512 head28 + gather-Q cross + CPU additional_edges | `30897.7 ms` | `10276.6 ms` | `12692.3 ms` | `1001.65 ms` | `0.9323` | `0.9523` | `101.6 ms` | `66.1 ms` |

结构诊断补充，使用 `tools/benchmarks/diagnose_ung_graph_structure.py` 直接扫描保存的 x400 `graph` 和 `group_id_to_range`，不重建 index：

```text
/home/graphdb/fv_runs/graph_diagnostics_x400_20260602
/home/graphdb/fv_runs/graph_diagnostics_x400_20260602/summary_table.md
```

| Metric | CPU Vamana | light512 | light512 - CPU |
|---|---:|---:|---:|
| total_edges | `77898966` | `75670202` | `-2228764` |
| intra_edges | `75767742` | `73538978` | `-2228764` |
| cross_edges | `2131224` | `2131224` | `0` |
| intra_out_degree_avg | `31.4441` | `30.5192` | `-0.9250` |
| zero_intra_ratio | `0` | `0.00199162` | `+0.00199162` |
| low_intra_le4_ratio | `0.000000415` | `0.00929781` | `+0.00929739` |
| group_largest_wcc_ratio_avg | `1.0` | `0.997002` | `-0.002998` |
| groups_with_largest_wcc_lt_0_9 | `0` | `39` | `+39` |

解释：x400 light512 的 cross edges 完全相同，recall gap 主要来自组内图结构变化。light512 通过轻剪枝少写了约 `222.9` 万条组内边，产生少量零组内出度点和低组内出度点，并让 `39` 个 group 的最大 weak component 小于 `0.9`。这支持作者的设计方向：不能只增加 head nearest neighbors，而要补低成本 tail/reverse diversity 或做质量感知 router。

结论：

- Heavy prune 质量接近 CPU：L1000/L5000 只低 `0.0017`，但 build 明显更慢，Index 只有 `0.72x`，不能作为 x400 替代。
- 将 light-prune 阈值提到 `512` 后，prune 从 `19977.7 ms` 降到 `1022.45 ms`，group graph 相对 CPU 加速 `1.69x`，Index 加速 `1.23x`。
- light512 的质量代价偏大：L1000/L5000 均低 `0.0081`。这说明 x400 需要“介于 heavy 和 light 之间”的 budgeted occlusion / reverse augmentation，而不是简单把所有大组都 strong prune 或 light prune。
- `LIGHT_HEAD=28` 反而更差：L1000/L5000 从 head24 的 `0.9379/0.9579` 降到 `0.9323/0.9523`。原因是当前 light kernel 会先保留 `head_keep` 个近邻，再用剩余名额从 GNN 候选尾部等距抽样；head 越大，尾部多样性越少。因此简单增加近邻头部不是质量修复方向。
- `LIGHT_HEAD=32` 本轮未完成：运行到 group graph 阶段后外部进程占满 GPU（`./build/bin/Test`, SM 99%），为避免污染结果已中止，不计入表。
- 作者实现回应：已加入默认关闭的 `light reverse-tail prune`，通过 `UNG_FAST_GRNND_LIGHT_REVERSE_CAP/SLOTS/FORWARD_CAP` 开启。它只在 light prune 内构造少量 sampled reverse candidates，并把它们填入尾部预算，不做 heavy RNG occlusion。该实现已完成 x400 A/B。

## 新增证据: x400 Repair / Reverse-Tail A/B

实验目录：

```text
/home/graphdb/fv_runs/x400_reverse_tail_ab_20260602_030908
```

运行配置：Amazon 1% x400 coverage-query，stable gather-Q cross，CPU additional_edges，`NUM_REPEATS=3`，`LSEARCH_VALUES="1000 5000"`，并对三个变体都运行保存图结构诊断。

| Case | Index | Group | Prune | Fill | L1000 | L5000 | zero intra | low<=4 | WCC<0.9 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| light512_norepair | `30128.1` | `12360.1` | `1005.94` | `4036.13` | `0.937067` | `0.9574` | `0.001912` | `0.00937` | `39` |
| light512_repair | `36310.5` | `16545.7` | `1018.17` | `8711.65` | `0.9378` | `0.957767` | `1.66e-6` | `9.67e-5` | `38` |
| light512_reverse_repair | `33044.9` | `13996.7` | `1093.37` | `5550.73` | `0.945567` | `0.9659` | `0` | `8.67e-5` | `39` |

同脚本 CPU Vamana baseline：Index `36257.3`，group `17180.5`，cross `11013.3`，L1000/L5000 `0.946/0.965`；历史 CPU Vamana gather-Q full-quality为 Index `37807.6`，group `17537.3`，L1000/L5000 `0.9460/0.9660`。

Reviewer 解释：

- repair-only 验证了结构诊断：低出度确实可以被补边修复，但这几乎不提升 recall，并显著增加 host fill / Index time。因此不能把 `degree repair` 单独写成算法贡献。
- reverse-tail+repair 是正向结果：L1000/L5000 相对 norepair 都提升约 `0.0085`，L5000 `0.9659` 基本贴近同脚本 CPU `0.965`，同时 Index `33044.9 ms` 比同脚本 CPU `36257.3 ms` 快约 `1.10x`。
- 该结果支持“轻量反向尾部多样性”这个方法点，比简单 head 增大或单纯补出度更合理。
- compact-D2H A/B 已补且为负/不稳定；同脚本 CPU baseline 已补，后续应做更多参数扫和真实多标签。
- x400 目前不能支持“无损普遍替代”表述；它支持的表述是：自适应剪枝能控制性能，但质量路由和剪枝预算仍是论文必须正面承认的问题。

## 第二优先级实验: Cross-Edge Ablation

目标是回答 fused speedup 到底来自哪里。

| Ablation | Expected answer |
|---|---|
| final fused | 当前最优 |
| no direct-qid | query 展开/pack 成本增加多少 |
| no id-only writeback | D2H 和 host merge 成本增加多少 |
| no resident vectors | `prepare_all` 与 repeated H2D 的权衡 |
| SGEMM+topK TF32 | cuBLAS 强 baseline |

输出必须同时包含：

```text
cross total
prepare_all
resident cross
GPU H2D events
GPU kernel events
GPU D2H events
host/writeback residual
```

## 第三优先级实验: Graph Structure Diagnostics

目标是防止 reviewer 认为 local RNG pruning 破坏图连通性。

需要统计：

| Metric | Reason |
|---|---|
| out-degree distribution | 验证剪枝后度数是否异常 |
| low-degree / zero-degree ratio | 防止孤立点 |
| weak/strong connected component size | 验证导航连通性 |
| reverse edge coverage | 验证 reverse augmentation 是否真的补了结构 |
| query visited nodes | 连接结构变化和 search latency |

这些结果可以先在代表 group 上做，再扩展到 Amazon x100/x400 全部 group 的抽样统计。

## 写作约束

1. 摘要和结论可以保留 x100 build 加速，但必须同步说明 group graph 质量证据是分 workload 的：原始 Amazon PF、SIFT30、x200、10%x40、x400 已有端到端 recall 表；x100 repeat=3 和 x400 质量增强仍是投稿前缺口。
2. 不写“FastGrnndCuda 无损替代 CPU Vamana”；写“speed-quality tradeoff”。
3. 不写“custom kernel 普遍快于 cuBLAS/cuVS”；写“对 irregular group workload 的系统级 fusion 更快”。
4. x400 direct-qid 修复前，只把 gather-Q 作为稳定大规模结果。
5. fallback policy 的收益单独归因，不能全部算作 graph algorithm 的收益。
