# 投稿就绪状态快照

更新时间：2026-06-02

本文档是审稿人/作者迭代的单页状态表。它不替代 `EVIDENCE_MATRIX_CN.md`，而是给出当前能否投稿、哪些 claim 可写、哪些实验仍会阻塞投稿。

投稿前逐条 rebuttal 自查见 `docs/papers/REBUTTAL_CHECKLIST_CN.md`。该 checklist 用于防止把 pending 实验写成主文结论。

## 当前结论

**还不能按“完整投稿版”提交。**

当前可以支撑的主线是：

- cross-edge grouped fused topK 有强 baseline 证据：CPU Vamana、128 线程 CPU exact、cuVS per-group、SGEMM+topK、final fused。
- FastGrnndCuda 组内图有多组端到端 recall A/B 证据，但仍只能写成 workload-aware speed/quality tradeoff，不能写成 CPU Vamana 的无损普遍替代。
- x400 是当前最强压力测试：heavy prune 质量接近但慢，light prune 快但 L1000/L5000 recall drop 为 `0.0081`；reverse-tail+repair 已把 L5000 补到 `0.9659`，接近同脚本 CPU `0.965`，Index `33044.9 ms` 相对同脚本 CPU `36257.3 ms` 为约 `1.10x`，但仍只能写成 quality-enhanced Pareto 点。
- packed exact-anchor router 已补 x200/x400 L5000 repeat=3 search sweep，证明 host scatter/fill 外围优化不牺牲这两组 coverage-query 的高 Lsearch recall；旧 exact 低 recall 已被结构诊断归因为 cross/additional 口径混杂。新的 x200 direct-H2D A/B 进一步把旧 exact batch 的 `pack=4140.61 ms` 降到 `0.04 ms`，group 从 `8999.26 ms` 降到手动 fill16 的 `3204.88 ms`，同时 L1000 recall 保持 `0.867`。
- `adaptive_cuda` conservative route 已在 Amazon 1% x100 full-quality 上补证据：小组 CPU fallback、中组 packed exact-anchor、大组 FastGrnndCuda 的统一后端把 Index `6370.67 -> 5450 ms`、group `3216.21 -> 2773.35 ms`，L100/L500/L1000 recall `0.828/0.871/0.896 -> 0.829/0.870/0.895`。它是 x100 上的正向 route 证据，不是跨所有 workload 的完成证明。
- cross-edge 的当前收敛方向是 `UNG_UNIVERSAL_GPU=1` 的 universal flat double-buffer route：target-centric descriptor batching、double-buffer execution、GPU global merge 和 flat-id output。SIFT30 skip-additional 中 cross-edge `4918.9 -> 3180.63 ms`；Amazon 1% x200 full-quality 中 cross-edge `2494.21 ms`，L1000/L5000 recall `0.871/0.911`；Amazon 1% x100 full-quality 中 cross-edge `1689.40 ms`，L100/L500/L1000 recall `0.826/0.868/0.891`。x200 是正结果，x100 是兼容性/边界结果，不能写成无条件端到端加速。

当前不能写成主结论的是：

- `UNG_FAST_GRNND_REPAIR_DEGREE=1` 与 reverse-tail 已完成 x400 A/B；结论是 repair 单独主要改善结构但代价高、recall收益小，reverse-tail+repair 才给出质量增强 Pareto 点。
- `UNG_TAGORE_COMPACT_D2H=1` 是外围写回优化，不能写成图质量贡献。
- x400 direct-qid 仍是 limitation，x400 主表必须使用 gather-Q 路径或等待 strict 修复。
- 当前 GPU backend 仍有 CPU-compatible output boundary：图结果最终回填 host `Graph::neighbors` / `SearchQueue` / Vamana-compatible structures。direct-H2D 已消除 exact-anchor 输入侧 pack；Graph reserve A/B 证明 per-node adjacency 分配/扩容是实质瓶颈：10%x40 reserve 把 Index `39746.1 -> 24704.5 ms`、tagore fill `823.5 -> 23.9 ms`、cross merge `1677.1 -> 2.3 ms`，但旧 reserve 自身仍花 `6487.3 ms`。最新 small-buffer `NeighborList` 把 x200 reserve 从 `2856.2 ms` 降到 `25.4 ms`、fill 从 `224.8 ms` 降到 `14.7 ms`；10%x40 reserve 从 `6487.3 ms` 降到 `21.2 ms`、fill 从 `23.9 ms` 降到 `7.6 ms`。direct-global/flat-id smoke 进一步说明，offset pass 本身只剩毫秒级，而 flat-id writeback 能把 x100 skip-additional cross 从 `1650.95` 降到 `707.83 ms`、D2H 从 `11.1` 降到 `0.6 ms`、merge 从 `17.2` 降到 `0.5 ms`，但它跳过 additional_edges，不能作为 full-quality 主结果。additional direct-append x400 负例还说明，直接在 additional 阶段并行 append 到 `_graph->neighbors`，即使用 per-node lock，也没有稳定完成 full-quality build。这些结果共同支持 CSR/GraphView staged output，而不是证明 flat adjacency/CSR 已完成。
- source-centric no-lock 不能作为当前主线。修复 id-only null-distance 写入和 stream error check 后，SIFT30 source-centric CUDA-core id-only cross `5926.67 ms`，慢于 target-centric fused `4918.9 ms` 和 universal flat `3180.63 ms`。legacy source WMMA smoke 只能作为 future two-stage source grouped GEMM + reduce 的线索。

## Artifact Gate

最新命令：

```bash
python3 tools/benchmarks/run_submission_gate.py \
  --router-root /home/graphdb/fv_runs/group_graph_router_ab_streamfill_20260602_030037 \
  --x400-root /home/graphdb/fv_runs/x400_reverse_tail_ab_20260602_030908
```

当前结果：

| Claim | Evidence | Verdict |
|---|---|---|
| x100 cross-edge strong-baseline fairness | present | OK |
| x400 light512 structural diagnosis | present | OK |
| x400 repair/reverse-tail A/B | `/home/graphdb/fv_runs/x400_reverse_tail_ab_20260602_030908/x400_ab_summary.md` | OK |
| x400 same-script CPU baseline | `/home/graphdb/fv_runs/x400_cpu_only_neighborlist_20260602_082814/x400_ab_summary.md` | OK, CPU Index `36257.3 ms`, L1000/L5000 `0.946/0.965` |
| mixed exact/GNN router A/B | `/home/graphdb/fv_runs/group_graph_router_ab_streamfill_20260602_030037/router_ab_summary.md` | OK, preliminary repeat=1 |
| packed exact-anchor router L5000 sweep | `/home/graphdb/fv_runs/packed_exact_search_sweep_20260602_033900_x200/results/search_time_summary.csv`；`/home/graphdb/fv_runs/packed_exact_search_sweep_20260602_033926_x400/results/search_time_summary.csv` | OK, x200/x400 repeat=3 |
| x200 packed exact direct-H2D overhead A/B | `/home/graphdb/fv_runs/direct_h2d_exact_ab_20260602_old`；`/home/graphdb/fv_runs/direct_h2d_exact_ab_20260602_fill16`；`/home/graphdb/fv_runs/direct_h2d_exact_ab_20260602_default` | OK, pack bottleneck removed; fill remains |
| old/new exact structural diagnosis | `/home/graphdb/fv_runs/packed_exact_graph_diag_20260602_034031/summary_with_cpu.md` | OK, old exact low recall is confounded |
| Amazon 1% x100 repeat=3 group-graph recall A/B | `/home/graphdb/fv_runs/end_to_end_recall_ab_20260602_041311/summary.csv` | OK, full-quality recall closed; speedup small |
| Amazon 1% x100 adaptive_cuda conservative full-quality A/B | `/home/graphdb/fv_runs/adaptive_cuda_x100_fulladd_min128_exact512_20260602_042853/search_eval/search_time_summary.csv` | OK, x100 route improves build with matching recall |
| 10%x40 additional Lsearch sweep | `/home/graphdb/fv_runs/tenx40_lsearch_sweep_20260602_043657/tenx40_lsearch_sweep_summary.md` | OK, L50/100/200/500/1000/2000/5000 repeat=3 |
| Amazon 1% x200 partial double-buffer routing | `/home/graphdb/fv_runs/target_fused_x200_db_split_20260602_060034`；`/home/graphdb/fv_runs/graph_diag_x200_db_split_20260602` | OK, partial fast-path routing and structure evidence |
| x100 additional_edges/output-boundary smoke | `/home/graphdb/Codes/FilterVectorResultsCUDA/prof/ung_prof_20260602_064111_103566.log`；`/home/graphdb/Codes/FilterVectorResultsCUDA/prof/ung_prof_20260602_064200_103859.log` | OK, additional_edges/output-boundary bottleneck evidence |
| CelebA or real multi-label end-to-end recall | `/home/graphdb/fv_runs/celeba_current_sanity_20260602_070322` | OK as sanity; not yet a strong GPU method main-table result |
| x200 warp exact / fill-thread negative ablation | `/home/graphdb/fv_runs/warp_exact_x200_on_20260602_073034`；`/home/graphdb/fv_runs/warp_exact_x200_off_20260602_073130`；`/home/graphdb/fv_runs/fill8_exact_x200_20260602_073346`；`/home/graphdb/fv_runs/fill64_exact_x200_20260602_073241` | OK, now value-checked by artifact gate; supports output-boundary rather than kernel-microtuning diagnosis |
| Graph output-boundary reserve A/B | `/home/graphdb/fv_runs/graph_reserve_ab_20260602_074459`；`/home/graphdb/fv_runs/graph_reserve_x200_auto2_20260602_075406` | OK, value-checked; reserve reduces fill/fallback/merge but does not replace CSR/GraphView need |
| Graph output-boundary NeighborList small-buffer A/B | `/home/graphdb/fv_runs/neighborlist_x200_autoreserve_20260602_081303`；`/home/graphdb/fv_runs/neighborlist_10px40_autoreserve_20260602_081403`；负例 `/home/graphdb/fv_runs/neighborlist48_x200_autoreserve_20260602_081636` | OK, value-checked; 64-inline removes most allocator cost, 48-inline is negative; still CPU-compatible, not CSR |
| Graph output-boundary direct-global/flat-id smoke | `/home/graphdb/fv_runs/direct_global_off_smoke_20260602_084831`；`/home/graphdb/fv_runs/direct_global_smoke_20260602_084744`；`/home/graphdb/fv_runs/flat_id_direct_global_smoke_20260602_084939` | OK, value-checked; direct-global only removes offset, flat-id reduces D2H/merge on skip-additional, still not full-quality CSR |
| Graph output-boundary additional direct-append negative ablation | `/home/graphdb/fv_runs/additional_direct_x400_on_20260602_083520`；`/home/graphdb/fv_runs/additional_direct_x400_lock_on_20260602_083735` | OK, value-checked; direct append did not complete x400 full-quality, current code guards against CPU Vamana misuse |
| source-centric checked negative ablation | `/home/graphdb/FilterVectorResultsRefactor/organized_benchmarks/ung_cross_edge_fused_20260602_091046` | OK, value-checked; source-centric CUDA-core id-only is slower after correctness checks |
| SIFT30 universal flat double-buffer cross-edge | `/home/graphdb/FilterVectorResultsRefactor/organized_benchmarks/universal_all_db_sift30_20260602_092318` | OK, value-checked; skip-additional cross-edge smoke, not full-quality recall |
| Amazon 1% x200 universal flat full-quality smoke | `/home/graphdb/fv_runs/end_to_end_recall_ab_20260602_092928` | OK, value-checked; positive x200 full-quality cross-edge result with CPU Vamana additional_edges |
| Amazon 1% x100 universal flat full-quality boundary | `/home/graphdb/fv_runs/end_to_end_recall_ab_20260602_093357` | OK, value-checked; compatible and slight cross-edge win, but not end-to-end speedup |

`run_submission_gate.py` 默认模式和 `--final` 模式当前都通过，说明 Python helper 和关键 benchmark shell 入口语法正确，artifact audit 已登记的 claim 均有本机证据且关键数值通过校验；自动发现的 reviewer-facing Markdown 文档没有出现未加否定语境的“无损替代 / x400 已解决 / 自研 kernel 普遍超过 cuBLAS/cuVS”等危险表述。

注意：当前 `--final` 只表示 **已登记到 artifact audit 的证据闭环通过**，不等价于“论文科学上已经完整投稿就绪”。下方阻塞项仍然是 reviewer 层面的缺口；如果要把这些缺口写成主文结论，必须先把对应实验加入 artifact audit 并通过数值校验。

## 投稿前阻塞项

| 优先级 | 缺口 | 为什么阻塞 | 推荐入口 |
|---:|---|---|---|
| P1 | x400 更多参数扫 | 同脚本 CPU baseline 已补；reverse-tail+repair 对同脚本 CPU 为约 `1.10x`，L5000 略高但 L1000 低约 `0.00043`，需要更多参数扫证明不是单点 | `scripts/benchmarks/run_x400_reverse_tail_ab.sh` |
| P1 | packed exact-anchor / mixed exact-GNN router cross-dataset A/B | x200/x400 repeat=3 已补 L5000，但仍需确认 x100、10%x40 和真实多标签是否同样成立 | `scripts/benchmarks/run_group_graph_router_ab.sh` 或复用现有 index 做 search sweep |
| P1 | 真实多标签数据集强 A/B | CelebA regenerated-GT sanity 已存在，但还不足以作为 GPU group/cross 主表；仍需同脚本 CPU/GPU 方法对照 | CelebA 或其他真实多标签数据 |
| P1 | CPU-compatible output boundary 说明与 future-work 定位 | small-buffer 已降低 allocator 成本，但仍不是 CSR/GraphView；需防止 reviewer 把“普遍替代”理解为已经完全替换 Vamana/host graph stack | 主文 limitation + flat adjacency/CSR 设计草图 |
| P2 | CPU exact 线程扩展 | 强化 `22x` 不是弱 CPU baseline 的质疑回应 | CPU exact 1/32/64/128 线程 sweep |
| P2 | x400 direct-qid strict 修复 | 把 x400 从 limitation 改成稳定 route | direct-qid strict rerun / sanitizer |

## 下一次 GPU 空闲后的建议顺序

1. 跑 x400 repair/reverse-tail A/B：

```bash
RUN_GRAPH_DIAG=1 \
scripts/benchmarks/run_x400_reverse_tail_ab.sh
```

compact-D2H ablation 已完成，当前不再作为下一轮优先实验；若复测，只作为 overhead negative ablation。

2. 生成综合表：

```bash
python3 tools/benchmarks/summarize_x400_ab.py <out_root> \
  --csv <out_root>/x400_ab_summary.csv \
  --md <out_root>/x400_ab_summary.md
```

3. 重新跑 artifact gate：

```bash
python3 tools/benchmarks/audit_paper_artifacts.py \
  --x400-root <out_root> \
  --fail-required
```

如果准备进入投稿前最终检查，再跑：

```bash
python3 tools/benchmarks/run_submission_gate.py \
  --x400-root <out_root> \
  --router-root <router_out_root> \
  --final
```

或者只跑 artifact audit：

```bash
python3 tools/benchmarks/audit_paper_artifacts.py \
  --x400-root <out_root> \
  --router-root <router_out_root> \
  --fail-submission
```

## 写作边界

可以写：

> Grouped fused topK is a robust cross-edge construction accelerator for irregular UNG group workloads. FastGrnndCuda provides a workload-aware group-graph backend with speed/quality routing, and current evidence supports it on several workloads but not as a lossless universal replacement.

不能写：

> FastGrnndCuda losslessly replaces CPU Vamana on all group sizes.

不能写：

> x400 is solved by repair / compact-D2H.

在 x400 A/B 实测前，只能写：

> The x400 structural diagnosis motivates degree repair and reverse-tail augmentation. The A/B shows that degree repair alone mostly fixes low-degree structure but is too expensive and has little recall gain, while reverse-tail + repair recovers most of the light512 recall gap with lower cost than heavy pruning.
