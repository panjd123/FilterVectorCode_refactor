# 投稿前补强清单

当前已有：

- 中文论文草稿：`docs/papers/UNG_GPU_OPTIMIZATION_PAPER_DRAFT_CN.md`
- 英文论文草稿：`docs/papers/UNG_GPU_OPTIMIZATION_PAPER_DRAFT_EN.md`
- 最终技术报告：`docs/reports/TECHNICAL_REPORT_OPTIMIZATION_SPEEDUP_CN.md`
- 投稿证据矩阵：`docs/papers/EVIDENCE_MATRIX_CN.md`

## 必补实验

这里按“审稿人会怎么攻击”来排优先级。每个实验都要能回答一个明确质疑，否则不要先做。

1. **端到端 filtered search recall**
   - 攻击点：FastGrnndCuda 不是 CPU Vamana 等价替代，可能建得快但 query 质量下降。
   - 对照组：CPU Vamana group graph、FastGrnndCuda + CPU fallback、FastGrnndCuda + complete fallback；cross-edge 固定为同一实现，避免混淆。
   - 指标：recall@10/100、P50/P95/P99 query latency、build time、index size。
   - 最少覆盖状态：Amazon 1% x100 repeat=3、Amazon 1% x200 coverage、Amazon 10% x40 full-quality 和 Amazon 1% x400 gather-Q 已补主证据。10%x40 已补 L50/100/200/500/1000/2000/5000 repeat=3 sweep。x400 compact-D2H 已补负/不稳定 ablation；x400 same-script CPU baseline 已补；剩余缺口是真实多标签和更多参数扫。
   - 当前进展：原始 Amazon PF 已补一组带 CPU additional edges 的 A/B。FastGrnndCuda group graph `1049.92 ms` vs CPU Vamana group `5776.25 ms`，Index `36732 ms` vs `43376.6 ms`，Lsearch=1000/2000/5000 recall drop 均小于 `0.001`。SIFT30 已补修复 fused cross 后的 A/B，L1000 recall drop `0.00129`。Amazon 1% x200 coverage 旧重 prune 版本 Index `17596.8 -> 38273.7 ms` 失败；diversified light prune 后 Index `16478.5 ms`，快于 CPU `1.07x`，L5000 recall `0.9056`，接近 CPU `0.9080`。Amazon 1% x400 gather-Q full-quality 已补：heavy prune 质量接近但慢；light512 快但 L5000 drop `0.0081`；reverse-tail+repair Index `33044.9 ms`，相对同脚本 CPU `36257.3 ms` 为约 `1.10x`，L1000/L5000 `0.945567/0.9659` 接近同脚本 CPU `0.946/0.965`。
   - 修正后的 exact 结论：早期 batched exact kNN 的 L5000 `0.819867/0.829` 不能再作为组内 exact 图不可行证据。新结构诊断显示旧 exact-anchor 和新 packed exact 的组内图完全一致，差异在 cross edges：旧 `1,038,000`，CPU/new packed `1,093,224`。因此旧负结果应标为 cross/additional 口径混杂。
   - 新外围与质量结果：已把 batched exact 的 D2H 结果改成 packed graph buffer 直读，避免先 scatter 到每组 `result.graph` 再在 fill 阶段读一遍。x200 `UNG_FAST_GRNND_BATCH_EXACT_NX=4096` 为 Index `11043.5 ms`、group `3827.28 ms`、repeat=3 L1000/L5000 `0.869/0.908`；x400 同配置为 Index `28436.3 ms`、group `11588.4 ms`、repeat=3 L1000/L5000 `0.945/0.967`。进一步 direct-H2D A/B 证明 x200 exact-anchor 旧性能主要来自外围：旧 path group `8999.26 ms`、`pack=4140.61 ms`、kernel `931.72 ms`、fill `2849.50 ms`；direct-H2D + GPU lookup + fill16 后 group `3204.88 ms`、`pack=0.04 ms`、kernel `928.67 ms`、fill `1383.53 ms`，L1000 recall 不变。这使 packed exact-anchor 成为当前最强 group-graph 替代候选，但仍需 10%x40/真实多标签确认，且下一步应优先做 flat adjacency/CSR 以消除剩余 fill。
   - x100 repeat=3 进展：已生成 coverage query 和 GT，full-quality A/B 固定 GPU fused cross 与 CPU additional_edges。CPU Vamana group Index `6863.47 ms`、L100/L500/L1000 `0.825/0.868/0.890667`；FastGrnndCuda + CPU fallback Index `6370.67 ms`、L100/L500/L1000 `0.828/0.871/0.896`。该结果是质量闭环正证据，但 group graph 只从 `3303.26 ms` 到 `3216.21 ms`，不是强加速点。
   - 新增 router 进展：mixed exact/GNN batch router 已完成 x200 repeat=1 A/B。`UNG_FAST_GRNND_BATCH_EXACT_NX=256/512` 没有复现 pure exact 的质量崩塌；`nx256` L100/L500/L1000 为 `0.8229/0.851/0.869`，CPU Vamana 为 `0.823/0.849/0.869`，group 为 `5770.88 ms`，相对 non-exact router `8606.52 ms` 为 `1.49x`。复现实验入口为 `scripts/benchmarks/run_group_graph_router_ab.sh`，后续需要 repeat=3 和更多数据集。
   - Amazon 10% x40 进展：balanced coverage query 和 GT 已生成，query 分布为 medium/narrow/broad `540/238/222`，matched groups avg/p50/p95 `883.09/25.5/3297.0`。第一轮 full-quality CPU Vamana + fused cross build 在 `target_groups=49374` 的 cross-edge 阶段超过 10 分钟没有进入 GPU event 输出；随后修复了 direct-pageable 未生效和 chunk 内重复 `Q gather/qnorm` 的问题。skip-additional 构建中，CPU exact cross 为 `54879.3 ms`，新 GPU direct-qid cross 为 `8475.4 ms`，cross 加速 `6.47x`。最新 full-quality A/B 使用 CPU additional_edges：CPU Vamana group Index `36053.3 ms`，FastGrnndCuda + complete fallback Index `23497.4 ms`，Index speedup `1.53x`；repeat=3 下 L100/L500/L1000 recall 从 `0.8647/0.898/0.908` 到 `0.8668/0.898/0.9103`，P95/P99 latency 也未恶化。
   - 通过标准：若 repeat-scale recall 下降，需要给出 Lsearch/efs 参数扫描，证明可用少量 query latency 换回 recall；否则不能把 FastGrnndCuda 写成无损替代。当前 10%x40 Lsearch sweep 已补，efs 在 UNG-only path 中为 `0`。
   - 注意：原始 Amazon PF 实验同时证明 `additional_edges` 不能随便跳过；skip 后 recall 约 `0.61`，带 CPU additional edges 才回到 `0.91-0.97`。

2. **强 baseline 公平性**
   - 攻击点：`22.07x` 主要来自弱 CPU Vamana cross baseline，不代表相对强 baseline 的贡献。
   - 必须列：CPU Vamana cross、CPU exact scan 128 线程、cuVS per-group、cuBLAS/SGEMM+topK、old fused、final fused。
   - 当前进展：`tools/benchmarks/summarize_cross_baselines.py` 已生成 `/home/graphdb/fv_runs/baseline_fairness_x100_20260602/cross_baselines.md`。x100 CPU exact baseline 使用 `128` build threads；机器为 Intel Xeon Platinum 8360Y，2 sockets，36 cores/socket，144 logical CPUs。final fused 相对 CPU exact 为 `2.82x`，相对 SGEMM+topK 为 `1.64x`。
   - 仍需注明：GPU 型号、cuBLAS math mode、是否 TF32、是否含 `prepare_all`。若时间允许，补 CPU exact scan 1/32/64/128 线程 scaling，用于说明 CPU baseline 已经充分并行。

3. **fused groupgemm+topK ablation**
   - 攻击点：创新可能只是“少调库/少拷贝”的工程优化，不是算法/系统贡献。
   - Ablation：关闭 resident base vectors、关闭 direct-qid、关闭 id-only writeback、关闭 custom fused 改走 SGEMM+topK。10%x40 的新增 direct-qid 修复还需要补一组 `pinned_staging/gather-Q` 对照，量化 `prepare_all 6500.7 -> 2596.3 ms` 和去掉 per-chunk `Q gather/qnorm` 的独立贡献。packed exact-anchor 已补 direct-H2D A/B，下一步不是继续调 exact kernel，而是 flat adjacency/CSR 图输出 A/B。
   - 新增 universal route 进展：`UNG_UNIVERSAL_GPU=1` 已把 target-centric descriptor batching、double-buffer execution、GPU global merge 和 flat-id output 收拢成当前 cross-edge 主工程路线。SIFT30 skip-additional cross `3180.63 ms`，相对旧 target fused `4918.9 ms` 为约 `1.55x`；Amazon 1% x200 full-quality 保留 CPU Vamana additional_edges 后 cross `2494.21 ms`，L1000/L5000 `0.871/0.911`；Amazon 1% x100 full-quality cross `1689.40 ms`，L100/L500/L1000 `0.826/0.868/0.891`，但 Index `6863.47 -> 7461.27 ms` 变慢。因此它是 x200 正结果和 x100 边界证据，不能写成无条件端到端加速。
   - source-centric 进展修正：修复 id-only null-distance 写入和 stream error check 后，SIFT30 CUDA-core source-centric cross `5926.67 ms`、kernel `5525.5 ms`，慢于 target fused 和 universal route。source-centric 只能作为 future two-stage source grouped GEMM + per-source reduce 方向，不再作为当前主线。
   - 指标：cross total、prepare_all、resident cross、H2D/kernel/D2H event、host/writeback residual。
   - 输出：按 `(nq,nx)` bucket 的耗时和占比，说明哪些 cross-edge 负载 route 到 fused，哪些 route 到 cuBLAS 更合理；同时按 `nx` bucket 说明 group graph 负载哪些 route 到 CPU Vamana，哪些 route 到 FastGrnndCuda。
   - 剩余缺口：universal route 还需要 x400、10%x40 和真实多标签 full-quality 复测；additional_edges 仍是 CPU Vamana，flat/CSR 输出还没有完成。

4. **direct-qid x400 稳定性**
   - 攻击点：核心优化在 x400 illegal memory access，说明实现不成熟。
   - 路线 A：修复 direct-qid x400，strict 模式通过，补最小复现和 sanitizer/cuda-memcheck 证据。
   - 路线 B：正式采用 gather-Q/direct-qid hybrid router；x400 主表只用 gather-Q，direct-qid 作为中等规模优化。
   - 投稿要求：未修复前不能把 x400 direct-qid 失败结果隐藏起来。

4.5. **x400 group graph 质量/性能路由**
   - 攻击点：x400 已证明 strong prune 太慢、light prune 降质，说明当前 group graph 还不是普遍替代。
   - 已有证据：heavy prune 的 L1000/L5000 recall drop 都只有 `0.0017`，但 prune `19977.7 ms`；light512 prune `1022.45 ms`，Index `1.23x`，但 L1000/L5000 recall drop `0.0081`。`LIGHT_HEAD=28` 没有改善，L5000 进一步降到 `0.9523`，说明简单多保留近邻会减少尾部多样性。
   - 新增结构诊断：`tools/benchmarks/diagnose_ung_graph_structure.py` 已对 x400 CPU Vamana 与 light512 的保存图完成扫描，输出在 `/home/graphdb/fv_runs/graph_diagnostics_x400_20260602`，比较表由 `tools/benchmarks/summarize_graph_diagnostics.py` 生成。cross edges 相同；light512 少 `2,228,764` 条组内边，低组内出度 `<=4` 比例升到 `0.00929781`，并出现 `39` 个最大 weak component 小于 `0.9` 的 group。这支持“质量问题来自组内结构稀疏/局部分裂”的解释。
   - 新增实现：已加入默认关闭的 light reverse-tail prune，环境变量为 `UNG_FAST_GRNND_LIGHT_REVERSE_CAP/SLOTS/FORWARD_CAP`。它在 light prune 中只把少量 sampled reverse candidates 填入尾部预算，不做距离计算和 occlusion，编译已通过。另新增默认开启的 `UNG_FAST_GRNND_REPAIR_DEGREE=1`，用于修复 light prune 的低出度点；新增默认开启的 `UNG_TAGORE_COMPACT_D2H=1`，用于减少 GPU graph D2H 和 host fill 外围开销。
   - 复现实验入口：`scripts/benchmarks/run_x400_reverse_tail_ab.sh`。默认跑 `light512_norepair`、`light512_repair`、`light512_reverse_repair`；设置 `RUN_CPU=1` 可重跑 CPU Vamana baseline；`RUN_COMPACT_ABLATION=1` 已用于补 compact-D2H 关闭对照，结论是负/不稳定；设置 `RUN_GRAPH_DIAG=1` 可在 build/search 后扫描保存图并生成 `x400_ab_summary.md/csv`；设置 `DRY_RUN=1` 可只检查路径和 env 文件。详细 runbook 见 `docs/runbooks/X400_REVERSE_TAIL_AB_CN.md`。
   - 必补方向：GPU 空闲后在 x400 上跑 reverse-tail、budgeted occlusion、light/strong mixed router 扫描，目标是在 group graph 仍快于 CPU 的同时把 L5000 drop 压到 `0.003` 以内；或者明确给出质量感知 router，把 x400 heavy/light 作为可调 Pareto。

5. **additional_edges 完整质量闭环**
   - 攻击点：skip additional edges 会显著降低 recall，性能表不能偷换质量口径。
   - 当前进展：10%x40 已补 `UNG_ADDITIONAL_EDGES_IMPL=0` full-quality A/B，L100/L500/L1000 recall 没有下降；skip-additional L100 只有 `0.800`，full-quality 提升到约 `0.865+`。
   - 仍需补：如果 additional_edges 在其他数据集上变慢，应补 CPU exact / GPU exact additional-edge backend，并保持语义说明清楚。

6. **跨数据集泛化**
   - 攻击点：Amazon sampled+jitter-repeat 是人为构造，可能只对这个 group size 分布有效。
   - 最少补：SIFT/SIFT30、CelebA 或真实多标签数据、Amazon 10% x40。
   - 每个数据集至少报告：group size 分布、cross-edge breakdown、end-to-end build time；能跑 search 的必须报告 recall/latency。

7. **group graph quality 结构性证据**
   - 攻击点：单组 recall 不能代表图适合导航，局部 RNG 剪枝可能破坏连通性。
   - 必须补：degree 分布、孤立点/低出度点比例、最大连通分量、query 访问节点数、单组 graph recall 与 end-to-end recall 的相关性。
   - 参数扫描：`UNG_TAGORE_ITER`、`UNG_TAGORE_K`、`UNG_TAGORE_MIN_GROUP_SIZE`，形成 speed/quality Pareto。
   - 额外要求：把旧 batched exact kNN 低 recall 标为 confounded ablation，明确 full-quality cross/additional_edges 必须固定；不能再用它证明 exact-anchor 组内图不可行。
   - 新增要求：对 packed exact-anchor / mixed exact-GNN router 补 repeat=3、阈值 sweep，并区分 exact-router、multi-stream batch、fast fill、packed-buffer 四个因素的单独贡献；如果其他数据集上 recall 下降，router 只能写成 partial/conditional。

## 建议新增图表

1. **Cross-edge 方法柱状图**
   - CPU Vamana、CPU exact、cuVS per-group、SGEMM+topK、old fused、final fused。

2. **End-to-end scale 图**
   - x40/x100/x200/x400 Index time。
   - 同时画 group graph 和 cross-edge 占比。

3. **Group graph quality-speed tradeoff**
   - CPU Vamana、Tagore、FastGrnndCuda old、Reverse-augmented Local RNG、Method3。

4. **prepare_all breakdown**
   - host register、full-data H2D、norm、resident cross。

## 写作注意

- 不要声称自研 kernel 普遍快于 cuBLAS/cuVS；应表述为对 UNG irregular group workload 的系统级 fusion 优势。
- 不要把 CPU fallback 的收益完全归因于 FastGrnndCuda；它是 hybrid routing policy。
- x400 direct-qid 当前不稳定，必须作为 limitation 或 future work。
- FastGrnndCuda 不是 CPU Vamana 的严格语义等价替代，需要以 speed/quality tradeoff 表述。
