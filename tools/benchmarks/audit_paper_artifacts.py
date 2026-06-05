#!/usr/bin/env python3
"""Audit paper claims against local reproducibility artifacts.

This is a lightweight reviewer-facing checklist. It does not decide scientific
validity; it only verifies whether the local files named by the paper evidence
map exist and records which claims are still intentionally pending.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from dataclasses import dataclass, replace
from pathlib import Path


@dataclass(frozen=True)
class ArtifactItem:
    claim: str
    status: str
    required: bool
    paths: tuple[str, ...]
    note: str


DEFAULT_ARTIFACTS: tuple[ArtifactItem, ...] = (
    ArtifactItem(
        claim="x100 cross-edge strong-baseline fairness",
        status="done",
        required=True,
        paths=(
            "/home/graphdb/fv_runs/baseline_fairness_x100_20260602/cross_baselines.md",
            "/home/graphdb/fv_runs/baseline_fairness_x100_20260602/cross_baselines.csv",
        ),
        note="Supports CPU Vamana / CPU exact / cuVS / SGEMM+topK / final fused comparison.",
    ),
    ArtifactItem(
        claim="x400 light512 structural diagnosis",
        status="done",
        required=True,
        paths=(
            "/home/graphdb/fv_runs/graph_diagnostics_x400_20260602/summary_table.md",
        ),
        note="Supports claim that x400 recall gap is intra-group structural, not cross-edge.",
    ),
    ArtifactItem(
        claim="x400 repair/reverse-tail A/B",
        status="done",
        required=True,
        paths=(
            "/home/graphdb/fv_runs/x400_reverse_tail_ab_20260602_030908/x400_ab_summary.md",
            "/home/graphdb/fv_runs/x400_reverse_tail_ab_20260602_030908/x400_ab_summary.csv",
        ),
        note="Supports the x400 reverse-tail + repair quality-recovery claim.",
    ),
    ArtifactItem(
        claim="x400 same-script CPU baseline",
        status="done",
        required=True,
        paths=(
            "/home/graphdb/fv_runs/x400_cpu_only_neighborlist_20260602_082814/x400_ab_summary.csv",
            "/home/graphdb/fv_runs/x400_cpu_only_neighborlist_20260602_082814/x400_ab_summary.md",
            "/home/graphdb/fv_runs/x400_cpu_only_neighborlist_20260602_082814/cpu_vamana/cpu_vamana_group/results/build_time.csv",
            "/home/graphdb/fv_runs/x400_cpu_only_neighborlist_20260602_082814/cpu_vamana/cpu_vamana_group/results/search_time_summary.csv",
            "/home/graphdb/fv_runs/x400_cpu_only_neighborlist_20260602_082814/cpu_vamana/graph_diag/graph_structure_summary.json",
        ),
        note=(
            "Same-script x400 CPU Vamana baseline used to compare reverse-tail+repair without "
            "relying on the older historical CPU run."
        ),
    ),
    ArtifactItem(
        claim="x400 compact-D2H ablation",
        status="done",
        required=True,
        paths=(
            "/home/graphdb/fv_runs/x400_compact_ab_20260602_080045/x400_ab_summary.csv",
            "/home/graphdb/fv_runs/x400_compact_ab_20260602_080045/x400_ab_summary.md",
            "/home/graphdb/fv_runs/x400_compact_ab_20260602_080045/light512_repair/fastgrnnd_cpu_fallback/results/build_time.csv",
            "/home/graphdb/fv_runs/x400_compact_ab_20260602_080045/light512_repair/fastgrnnd_cpu_fallback/others/build.log",
            "/home/graphdb/fv_runs/x400_compact_ab_20260602_080045/light512_repair/fastgrnnd_cpu_fallback/results/search_time_summary.csv",
            "/home/graphdb/fv_runs/x400_compact_ab_20260602_080045/light512_repair_compact_off/fastgrnnd_cpu_fallback/results/build_time.csv",
            "/home/graphdb/fv_runs/x400_compact_ab_20260602_080045/light512_repair_compact_off/fastgrnnd_cpu_fallback/others/build.log",
            "/home/graphdb/fv_runs/x400_compact_ab_20260602_080045/light512_repair_compact_off/fastgrnnd_cpu_fallback/results/search_time_summary.csv",
            "/home/graphdb/fv_runs/x400_compact_on_buildonly_20260602_080513/results/build_time.csv",
            "/home/graphdb/fv_runs/x400_compact_on_buildonly_20260602_080513/others/build.log",
        ),
        note=(
            "Negative/ambiguous x400 light512_repair compact-D2H ablation: compact-on does not provide "
            "a stable build-time win once graph reserve is enabled, so compact-D2H should remain an "
            "overhead ablation rather than a main x400 claim."
        ),
    ),
    ArtifactItem(
        claim="mixed exact/GNN router A/B",
        status="done",
        required=True,
        paths=(
            "/home/graphdb/fv_runs/group_graph_router_ab_streamfill_20260602_030037/router_ab_summary.md",
            "/home/graphdb/fv_runs/group_graph_router_ab_streamfill_20260602_030037/router_ab_summary.csv",
        ),
        note="Supports the preliminary x200 mixed exact/GNN router claim; repeat=3 and more datasets remain pending.",
    ),
    ArtifactItem(
        claim="packed exact-anchor L5000 sweep",
        status="done",
        required=True,
        paths=(
            "/home/graphdb/fv_runs/packed_exact_search_sweep_20260602_033900_x200/results/search_time_summary.csv",
            "/home/graphdb/fv_runs/packed_exact_search_sweep_20260602_033926_x400/results/search_time_summary.csv",
            "/home/graphdb/fv_runs/packed_exact_search_sweep_summary_20260602.csv",
        ),
        note="Supports x200/x400 repeat=3 L1000/L5000 packed exact-anchor recall claims.",
    ),
    ArtifactItem(
        claim="x200 packed exact-anchor direct-H2D overhead A/B",
        status="done",
        required=True,
        paths=(
            "/home/graphdb/fv_runs/direct_h2d_exact_ab_20260602_old/router_exact_nx4096/fastgrnnd_cpu_fallback/results/build_time.csv",
            "/home/graphdb/fv_runs/direct_h2d_exact_ab_20260602_old/router_exact_nx4096/fastgrnnd_cpu_fallback/others/build.log",
            "/home/graphdb/fv_runs/direct_h2d_exact_ab_20260602_fill16/router_exact_nx4096/fastgrnnd_cpu_fallback/results/build_time.csv",
            "/home/graphdb/fv_runs/direct_h2d_exact_ab_20260602_fill16/router_exact_nx4096/fastgrnnd_cpu_fallback/others/build.log",
            "/home/graphdb/fv_runs/direct_h2d_exact_ab_20260602_default/router_exact_nx4096/fastgrnnd_cpu_fallback/results/build_time.csv",
            "/home/graphdb/fv_runs/direct_h2d_exact_ab_20260602_default/router_exact_nx4096/fastgrnnd_cpu_fallback/others/build.log",
            "/home/graphdb/fv_runs/direct_h2d_exact_ab_20260602_default/router_exact_nx4096/fastgrnnd_cpu_fallback/results/search_time_summary.csv",
        ),
        note=(
            "Supports the claim that x200 packed exact-anchor was bottlenecked by host pack and "
            "Graph::neighbors fill, not the GPU exact kernel; direct-H2D removes pack without changing recall."
        ),
    ),
    ArtifactItem(
        claim="x200 exact-anchor warp-kernel/fill-thread negative ablation",
        status="done",
        required=True,
        paths=(
            "/home/graphdb/fv_runs/warp_exact_x200_off_20260602_073130/router_exact_nx256/fastgrnnd_cpu_fallback/results/build_time.csv",
            "/home/graphdb/fv_runs/warp_exact_x200_off_20260602_073130/router_exact_nx256/fastgrnnd_cpu_fallback/others/build.log",
            "/home/graphdb/fv_runs/warp_exact_x200_off_20260602_073130/router_exact_nx256/fastgrnnd_cpu_fallback/results/search_time_summary.csv",
            "/home/graphdb/fv_runs/warp_exact_x200_on_20260602_073034/router_exact_nx256/fastgrnnd_cpu_fallback/results/build_time.csv",
            "/home/graphdb/fv_runs/warp_exact_x200_on_20260602_073034/router_exact_nx256/fastgrnnd_cpu_fallback/others/build.log",
            "/home/graphdb/fv_runs/warp_exact_x200_on_20260602_073034/router_exact_nx256/fastgrnnd_cpu_fallback/results/search_time_summary.csv",
            "/home/graphdb/fv_runs/fill8_exact_x200_20260602_073346/router_exact_nx256/fastgrnnd_cpu_fallback/results/build_time.csv",
            "/home/graphdb/fv_runs/fill8_exact_x200_20260602_073346/router_exact_nx256/fastgrnnd_cpu_fallback/results/search_time_summary.csv",
            "/home/graphdb/fv_runs/fill64_exact_x200_20260602_073241/router_exact_nx256/fastgrnnd_cpu_fallback/results/build_time.csv",
            "/home/graphdb/fv_runs/fill64_exact_x200_20260602_073241/router_exact_nx256/fastgrnnd_cpu_fallback/results/search_time_summary.csv",
        ),
        note=(
            "Negative ablation supporting the CPU-compatible output-boundary diagnosis: "
            "warp-per-source exact is slower than the block kernel, and 8/64 fill threads are slower than 16."
        ),
    ),
    ArtifactItem(
        claim="Graph output-boundary reserve A/B",
        status="done",
        required=True,
        paths=(
            "/home/graphdb/fv_runs/graph_reserve_ab_20260602_074459/off/results/build_time.csv",
            "/home/graphdb/fv_runs/graph_reserve_ab_20260602_074459/off/others/build.log",
            "/home/graphdb/fv_runs/graph_reserve_ab_20260602_074459/on/results/build_time.csv",
            "/home/graphdb/fv_runs/graph_reserve_ab_20260602_074459/on/others/build.log",
            "/home/graphdb/fv_runs/graph_reserve_x200_on_20260602_074935/results/build_time.csv",
            "/home/graphdb/fv_runs/graph_reserve_x200_on_20260602_074935/others/build.log",
            "/home/graphdb/fv_runs/graph_reserve_x200_auto_20260602_075133/results/build_time.csv",
            "/home/graphdb/fv_runs/graph_reserve_x200_auto_20260602_075133/others/build.log",
            "/home/graphdb/fv_runs/graph_reserve_x200_auto2_20260602_075406/results/build_time.csv",
            "/home/graphdb/fv_runs/graph_reserve_x200_auto2_20260602_075406/others/build.log",
        ),
        note=(
            "Supports the Graph::neighbors output-boundary diagnosis: pre-reserving host adjacency "
            "capacity sharply reduces fill/fallback/merge time on 10%x40, and x200 auto mode now "
            "enables reserve for large point-count exact-anchor workloads."
        ),
    ),
    ArtifactItem(
        claim="Graph output-boundary NeighborList small-buffer A/B",
        status="done",
        required=True,
        paths=(
            "UNG/codes/include/graph.h",
            "/home/graphdb/fv_runs/graph_reserve_x200_auto2_20260602_075406/results/build_time.csv",
            "/home/graphdb/fv_runs/graph_reserve_x200_auto2_20260602_075406/others/build.log",
            "/home/graphdb/fv_runs/graph_reserve_ab_20260602_074459/on/results/build_time.csv",
            "/home/graphdb/fv_runs/graph_reserve_ab_20260602_074459/on/others/build.log",
            "/home/graphdb/fv_runs/neighborlist_x200_autoreserve_20260602_081303/results/build_time.csv",
            "/home/graphdb/fv_runs/neighborlist_x200_autoreserve_20260602_081303/others/build.log",
            "/home/graphdb/fv_runs/neighborlist_x200_noreserve_20260602_081201/results/build_time.csv",
            "/home/graphdb/fv_runs/neighborlist_x200_noreserve_20260602_081201/others/build.log",
            "/home/graphdb/fv_runs/neighborlist_10px40_autoreserve_20260602_081403/results/build_time.csv",
            "/home/graphdb/fv_runs/neighborlist_10px40_autoreserve_20260602_081403/others/build.log",
            "/home/graphdb/fv_runs/neighborlist48_x200_autoreserve_20260602_081636/results/build_time.csv",
            "/home/graphdb/fv_runs/neighborlist48_x200_autoreserve_20260602_081636/others/build.log",
        ),
        note=(
            "Supports replacing per-node std::vector adjacency with a CPU-compatible small-buffer "
            "NeighborList. The retained 64-inline version removes most reserve/fill allocation cost, "
            "while the 48-inline probe is a negative layout/capacity ablation."
        ),
    ),
    ArtifactItem(
        claim="Graph output-boundary direct-global/flat-id smoke",
        status="done",
        required=True,
        paths=(
            "/home/graphdb/fv_runs/direct_global_off_smoke_20260602_084831/adaptive_cuda/results/build_time.csv",
            "/home/graphdb/fv_runs/direct_global_off_smoke_20260602_084831/adaptive_cuda/others/build.log",
            "/home/graphdb/fv_runs/direct_global_off_smoke_20260602_084831/adaptive_cuda/results/search_time_summary.csv",
            "/home/graphdb/fv_runs/direct_global_smoke_20260602_084744/adaptive_cuda/results/build_time.csv",
            "/home/graphdb/fv_runs/direct_global_smoke_20260602_084744/adaptive_cuda/others/build.log",
            "/home/graphdb/fv_runs/direct_global_smoke_20260602_084744/adaptive_cuda/results/search_time_summary.csv",
            "/home/graphdb/fv_runs/flat_id_direct_global_smoke_20260602_084939/adaptive_cuda/results/build_time.csv",
            "/home/graphdb/fv_runs/flat_id_direct_global_smoke_20260602_084939/adaptive_cuda/others/build.log",
            "/home/graphdb/fv_runs/flat_id_direct_global_smoke_20260602_084939/adaptive_cuda/results/search_time_summary.csv",
        ),
        note=(
            "Smoke evidence for output-boundary restructuring: direct-global removes the offset pass, "
            "while flat-id writeback avoids SearchQueue object materialization and sharply reduces D2H/merge "
            "on a skip-additional x100 run. This is not a full-quality main-table result."
        ),
    ),
    ArtifactItem(
        claim="Graph output-boundary additional direct-append negative ablation",
        status="done",
        required=True,
        paths=(
            "/home/graphdb/fv_runs/additional_direct_x400_on_20260602_083520/summary.csv",
            "/home/graphdb/fv_runs/additional_direct_x400_on_20260602_083520/fastgrnnd_cpu_fallback/others/build.log",
            "/home/graphdb/fv_runs/additional_direct_x400_lock_on_20260602_083735/summary.csv",
            "/home/graphdb/fv_runs/additional_direct_x400_lock_on_20260602_083735/fastgrnnd_cpu_fallback/others/build.log",
            "/home/graphdb/fv_runs/additional_direct_x400_lock_on_20260602_083735/fastgrnnd_cpu_fallback/others/env",
            "UNG/codes/src/uni_nav_graph.cpp",
        ),
        note=(
            "Negative ablation: directly appending additional_edges into Graph::neighbors after GPU cross-edge "
            "generation did not complete on x400, including a lock-guarded attempt. Current code disables the "
            "requested shortcut for CPU Vamana additional_edges; staged flat/CSR output is still needed."
        ),
    ),
    ArtifactItem(
        claim="old/new exact structural diagnosis",
        status="done",
        required=True,
        paths=(
            "/home/graphdb/fv_runs/packed_exact_graph_diag_20260602_034031/summary_with_cpu.md",
        ),
        note="Supports reinterpretation of old exact low recall as cross/additional-edge confounding.",
    ),
    ArtifactItem(
        claim="10%x40 bounded-complete fallback A/B",
        status="done",
        required=True,
        paths=(
            "/home/graphdb/fv_runs/end_to_end_recall_ab_20260602_035311/summary.csv",
            "/home/graphdb/fv_runs/end_to_end_recall_ab_20260602_035311/fastgrnnd_cpu_fallback/results/build_time.csv",
            "/home/graphdb/fv_runs/end_to_end_recall_ab_20260602_035728/summary.csv",
            "/home/graphdb/fv_runs/end_to_end_recall_ab_20260602_035728/fastgrnnd_cpu_fallback/results/build_time.csv",
            "/home/graphdb/fv_runs/end_to_end_recall_ab_20260602_040110/summary.csv",
            "/home/graphdb/fv_runs/end_to_end_recall_ab_20260602_040110/fastgrnnd_cpu_fallback/results/build_time.csv",
            "/home/graphdb/fv_runs/bounded_complete10x40_search_sweep_20260602_040727/results/search_time_summary.csv",
            "/home/graphdb/fv_runs/bounded_complete10x40_search_sweep_20260602_040727/results/query_details_repeat3.csv",
        ),
        note=(
            "Supports the 10%x40 claim that bounded-complete small-group fallback reduces graph "
            "materialization cost, while moving all small groups to GPU exact is a negative result; "
            "the search sweep checks L100/L500/L1000 recall and latency."
        ),
    ),
    ArtifactItem(
        claim="Amazon 1% x100 repeat=3 group-graph recall A/B",
        status="done",
        required=True,
        paths=(
            "/home/graphdb/fv_runs/end_to_end_recall_ab_20260602_041311/summary.csv",
            "/home/graphdb/fv_runs/end_to_end_recall_ab_20260602_041311/cpu_vamana_group/results/build_time.csv",
            "/home/graphdb/fv_runs/end_to_end_recall_ab_20260602_041311/cpu_vamana_group/results/search_time_summary.csv",
            "/home/graphdb/fv_runs/end_to_end_recall_ab_20260602_041311/cpu_vamana_group/results/query_details_repeat3.csv",
            "/home/graphdb/fv_runs/end_to_end_recall_ab_20260602_041311/fastgrnnd_cpu_fallback/results/build_time.csv",
            "/home/graphdb/fv_runs/end_to_end_recall_ab_20260602_041311/fastgrnnd_cpu_fallback/results/search_time_summary.csv",
            "/home/graphdb/fv_runs/end_to_end_recall_ab_20260602_041311/fastgrnnd_cpu_fallback/results/query_details_repeat3.csv",
        ),
        note="Full-quality CPU Vamana group vs FastGrnndCuda group A/B on generated x100 coverage queries.",
    ),
    ArtifactItem(
        claim="Amazon 1% x100 adaptive_cuda conservative full-quality A/B",
        status="done",
        required=True,
        paths=(
            "/home/graphdb/fv_runs/adaptive_cuda_x100_fulladd_min128_exact512_20260602_042853/build.log",
            "/home/graphdb/fv_runs/adaptive_cuda_x100_fulladd_min128_exact512_20260602_042853/search_eval/search_time_summary.csv",
        ),
        note=(
            "Supports the conservative adaptive_cuda route: CPU fallback for small groups, packed exact-anchor "
            "for medium groups, FastGrnndCuda for large groups, with full additional edges."
        ),
    ),
    ArtifactItem(
        claim="10%x40 additional Lsearch sweep",
        status="done",
        required=True,
        paths=(
            "/home/graphdb/fv_runs/tenx40_lsearch_sweep_20260602_043657/tenx40_lsearch_sweep_summary.md",
            "/home/graphdb/fv_runs/tenx40_lsearch_sweep_20260602_043657/tenx40_lsearch_sweep_summary.csv",
            "/home/graphdb/fv_runs/tenx40_lsearch_sweep_20260602_043657/cpu/results/query_details_repeat3.csv",
            "/home/graphdb/fv_runs/tenx40_lsearch_sweep_20260602_043657/fast/results/query_details_repeat3.csv",
        ),
        note=(
            "Wider full-quality Lsearch sweep for 10%x40, reusing the CPU Vamana and FastGrnndCuda "
            "full-additional indexes, covers L=50/100/200/500/1000/2000/5000."
        ),
    ),
    ArtifactItem(
        claim="Amazon 1% x200 partial double-buffer routing",
        status="done",
        required=True,
        paths=(
            "/home/graphdb/fv_runs/target_fused_x200_db_split_20260602_060034/end_to_end_recall_ab_20260602_060034/cpu_vamana_group/others/build.log",
            "/home/graphdb/fv_runs/target_fused_x200_db_split_20260602_060034/end_to_end_recall_ab_20260602_060034/cpu_vamana_group/results/build_time.csv",
            "/home/graphdb/fv_runs/target_fused_x200_db_split_20260602_060034/end_to_end_recall_ab_20260602_060034/cpu_vamana_group/results_r3_l1000_5000/search_time_summary.csv",
            "/home/graphdb/fv_runs/graph_diag_x200_db_split_20260602/graph_structure_summary.md",
            "/home/graphdb/fv_runs/graph_diag_x200_db_split_20260602/graph_structure_summary.json",
        ),
        note=(
            "Supports the all-or-nothing double-buffer fallback fix: x200 cross-edge 3900.81 -> "
            "2657.23 ms, repeat=3 L1000/L5000 0.866/0.907, with structural diagnosis."
        ),
    ),
    ArtifactItem(
        claim="source-centric checked negative ablation",
        status="done",
        required=True,
        paths=(
            "/home/graphdb/FilterVectorResultsRefactor/organized_benchmarks/ung_cross_edge_fused_20260602_091046/others/ung_build.log",
            "/home/graphdb/FilterVectorResultsRefactor/organized_benchmarks/ung_cross_edge_fused_20260602_091046/results/build_time.csv",
            "docs/reports/UNIVERSAL_GPU_REPLACEMENT_ANALYSIS_CN.md",
            "UNG/codes/src/gpu_gemm_topk.cu",
        ),
        note=(
            "Checked source-centric id-only negative ablation after fixing null-distance output and stream "
            "error checks. Supports downgrading source-centric to a future two-stage design direction rather "
            "than a current replacement."
        ),
    ),
    ArtifactItem(
        claim="SIFT30 universal flat double-buffer cross-edge",
        status="done",
        required=True,
        paths=(
            "/home/graphdb/FilterVectorResultsRefactor/organized_benchmarks/universal_all_db_sift30_20260602_092318/others/ung_build.log",
            "/home/graphdb/FilterVectorResultsRefactor/organized_benchmarks/universal_all_db_sift30_20260602_092318/results/build_time.csv",
            "docs/reports/UNIVERSAL_GPU_REPLACEMENT_ANALYSIS_CN.md",
            "UNG/codes/src/gpu_gemm_topk.cu",
            "UNG/codes/src/uni_nav_graph.cpp",
        ),
        note=(
            "Supports the updated universal cross-edge route: target-centric descriptor batching, "
            "double-buffer execution, GPU global merge, and flat-id output. This is a skip-additional "
            "SIFT30 smoke result, not a full-quality end-to-end recall claim."
        ),
    ),
    ArtifactItem(
        claim="Amazon 1% x200 universal flat full-quality smoke",
        status="done",
        required=True,
        paths=(
            "/home/graphdb/fv_runs/end_to_end_recall_ab_20260602_092928/cpu_vamana_group/others/build.log",
            "/home/graphdb/fv_runs/end_to_end_recall_ab_20260602_092928/cpu_vamana_group/results/build_time.csv",
            "/home/graphdb/fv_runs/end_to_end_recall_ab_20260602_092928/cpu_vamana_group/results_r3_l1000_5000/search_time_summary.csv",
            "/home/graphdb/fv_runs/end_to_end_recall_ab_20260602_092928/cpu_vamana_group/others/env",
            "docs/reports/UNIVERSAL_GPU_REPLACEMENT_ANALYSIS_CN.md",
        ),
        note=(
            "Full-quality x200 smoke for the universal flat double-buffer cross-edge route with CPU Vamana "
            "additional_edges enabled. Supports compatibility and a positive x200 cross-edge result, not a "
            "claim that all workloads are solved."
        ),
    ),
    ArtifactItem(
        claim="Amazon 1% x100 universal flat full-quality boundary",
        status="done",
        required=True,
        paths=(
            "/home/graphdb/fv_runs/end_to_end_recall_ab_20260602_093357/cpu_vamana_group/others/build.log",
            "/home/graphdb/fv_runs/end_to_end_recall_ab_20260602_093357/cpu_vamana_group/results/build_time.csv",
            "/home/graphdb/fv_runs/end_to_end_recall_ab_20260602_093357/cpu_vamana_group/results_r3_l100_500_1000/search_time_summary.csv",
            "/home/graphdb/fv_runs/end_to_end_recall_ab_20260602_093357/cpu_vamana_group/others/env",
            "docs/reports/UNIVERSAL_GPU_REPLACEMENT_ANALYSIS_CN.md",
        ),
        note=(
            "Full-quality x100 boundary result for the universal flat route: compatible and slightly lower "
            "cross-edge time, but not an end-to-end speedup. This prevents overclaiming the x200 positive."
        ),
    ),
    ArtifactItem(
        claim="x100 additional_edges/output-boundary smoke",
        status="done",
        required=True,
        paths=(
            "/home/graphdb/Codes/FilterVectorResultsCUDA/prof/ung_prof_20260602_064021_103101.log",
            "/home/graphdb/Codes/FilterVectorResultsCUDA/prof/ung_prof_20260602_064045_103351.log",
            "/home/graphdb/Codes/FilterVectorResultsCUDA/prof/ung_prof_20260602_064111_103566.log",
            "/home/graphdb/Codes/FilterVectorResultsCUDA/prof/ung_prof_20260602_064200_103859.log",
        ),
        note=(
            "Supports the reviewer-facing bottleneck claim that full-quality additional_edges and "
            "CPU-compatible output boundaries can dominate the GPU fused topK kernel."
        ),
    ),
    ArtifactItem(
        claim="CelebA or real multi-label end-to-end recall",
        status="done",
        required=False,
        paths=(
            "/home/graphdb/fv_runs/celeba_current_sanity_20260602_070322/end_to_end_recall_ab_20260602_070322/cpu_vamana_group/results/build_time.csv",
            "/home/graphdb/fv_runs/celeba_current_sanity_20260602_070322/regenerated_gt/celeba_gt_labels_containment.bin",
            "/home/graphdb/fv_runs/celeba_current_sanity_20260602_070322/end_to_end_recall_ab_20260602_070322/cpu_vamana_group/results_regt_L100_1000_5000/search_time_summary.csv",
            "/home/graphdb/fv_runs/celeba_current_sanity_20260602_070322/end_to_end_recall_ab_20260602_070322/cpu_vamana_group/results_regt_L100_1000_5000/search.log",
        ),
        note=(
            "Real multi-label CelebA sanity. The bundled query_7 GT was K=5/40000 bytes; "
            "this artifact uses regenerated K=10 GT and reports L100/L1000/L5000 recall."
        ),
    ),
)


def artifacts_with_overrides(x400_root: Path | None, router_root: Path | None) -> tuple[ArtifactItem, ...]:
    artifacts = list(DEFAULT_ARTIFACTS)
    if x400_root is not None:
        x400_paths = (
            str(x400_root / "x400_ab_summary.md"),
            str(x400_root / "x400_ab_summary.csv"),
        )
        for i, item in enumerate(artifacts):
            if item.claim == "x400 repair/reverse-tail A/B":
                status = "done" if exists_all(x400_paths) else item.status
                artifacts[i] = replace(
                    item,
                    status=status,
                    paths=x400_paths,
                    note=f"Overridden by --x400-root={x400_root}; expects x400_ab_summary.md/csv.",
                )
                break
    if router_root is not None:
        router_paths = (
            str(router_root / "router_ab_summary.md"),
            str(router_root / "router_ab_summary.csv"),
        )
        for i, item in enumerate(artifacts):
            if item.claim == "mixed exact/GNN router A/B":
                status = "done" if exists_all(router_paths) and router_summary_has_measurements(router_root) else item.status
                artifacts[i] = replace(
                    item,
                    status=status,
                    paths=router_paths,
                    note=(
                        f"Overridden by --router-root={router_root}; expects router_ab_summary.md/csv "
                        "with measured index_ms and avg_recall, not DRY_RUN-only NA rows."
                    ),
                )
                break
    return tuple(artifacts)


def exists_all(paths: tuple[str, ...]) -> bool:
    return bool(paths) and all(Path(p).exists() for p in paths)


def router_summary_has_measurements(root: Path) -> bool:
    path = root / "router_ab_summary.csv"
    if not path.exists():
        return False
    with path.open() as f:
        for row in csv.DictReader(f):
            index_ms = row.get("index_ms", "").strip().upper()
            avg_recall = row.get("avg_recall", "").strip().upper()
            if index_ms not in ("", "NA") and avg_recall not in ("", "NA"):
                return True
    return False


def read_metric_csv(path: Path) -> dict[str, float]:
    values: dict[str, float] = {}
    with path.open() as f:
        for key, value, *_ in csv.reader(f):
            if key == "Index Name":
                continue
            try:
                values[key] = float(value)
            except ValueError:
                pass
    return values


def read_search_recall_csv(path: Path) -> dict[int, float]:
    recalls: dict[int, float] = {}
    with path.open() as f:
        for row in csv.DictReader(f):
            try:
                recalls[int(float(row["Lsearch"]))] = float(row["Average_Recall"])
            except (KeyError, TypeError, ValueError):
                continue
    return recalls


def expect_close(errors: list[str], name: str, got: float | None, want: float, tol: float) -> None:
    if got is None:
        errors.append(f"{name}: missing, expected {want}")
    elif abs(got - want) > tol:
        errors.append(f"{name}: got {got}, expected {want}±{tol}")


def expect_equal(errors: list[str], name: str, got: int | None, want: int) -> None:
    if got != want:
        errors.append(f"{name}: got {got}, expected {want}")


def validate_partial_double_buffer_x200(item: ArtifactItem) -> list[str]:
    errors: list[str] = []
    build_log = Path(item.paths[0])
    build_csv = Path(item.paths[1])
    search_csv = Path(item.paths[2])
    graph_json = Path(item.paths[4])

    build = read_metric_csv(build_csv)
    expect_close(errors, "build_cross_edges_time", build.get("build_cross_edges_time"), 2657.23, 0.02)
    expect_close(errors, "build_graph_time", build.get("build_graph_time"), 7676.06, 0.02)

    log_text = build_log.read_text(errors="replace")
    split_match = re.search(r"double_buffer split supported_groups=(\d+) fallback_groups=(\d+)", log_text)
    if not split_match:
        errors.append("double_buffer split line missing")
    else:
        expect_equal(errors, "supported_groups", int(split_match.group(1)), 5173)
        expect_equal(errors, "fallback_groups", int(split_match.group(2)), 22)
    kernel_match = re.search(r"\[GPU GEMM\].*Kernel\(ms\)=([0-9.]+)", log_text)
    if not kernel_match:
        errors.append("[GPU GEMM] kernel line missing")
    else:
        expect_close(errors, "gpu_kernel_ms", float(kernel_match.group(1)), 1636.0, 0.05)

    recalls = read_search_recall_csv(search_csv)
    expect_close(errors, "L1000 recall", recalls.get(1000), 0.866, 1e-6)
    expect_close(errors, "L5000 recall", recalls.get(5000), 0.907, 1e-6)

    graph = json.loads(graph_json.read_text())
    expect_equal(errors, "cross_edges", int(graph.get("cross_edges", -1)), 1093032)
    expect_equal(errors, "intra_edges", int(graph.get("intra_edges", -1)), 38546837)
    expect_close(errors, "zero_intra_ratio", float(graph.get("zero_intra_ratio", -1.0)), 0.0, 0.0)
    return errors


def validate_source_checked_negative(item: ArtifactItem) -> list[str]:
    errors: list[str] = []
    build_log = Path(item.paths[0])
    build_csv = Path(item.paths[1])
    report = Path(item.paths[2])
    source = Path(item.paths[3])

    build = read_metric_csv(build_csv)
    expect_close(errors, "build_cross_edges_time", build.get("build_cross_edges_time"), 5926.67, 0.05)

    log_text = build_log.read_text(errors="replace")
    source_match = re.search(
        r"\[cross_edges\]\[source_exact\].*mode=(\w+).*kernel\(ms\)=([0-9.]+).*"
        r"d2h\(ms\)=([0-9.]+).*writeback\(ms\)=([0-9.]+).*id_only=(\d+)",
        log_text,
    )
    if not source_match:
        errors.append("source_exact checked line missing")
    else:
        if source_match.group(1) != "cuda_core":
            errors.append(f"source mode: got {source_match.group(1)}, expected cuda_core")
        expect_close(errors, "source kernel_ms", float(source_match.group(2)), 5525.5, 0.2)
        expect_close(errors, "source d2h_ms", float(source_match.group(3)), 1.0, 0.2)
        expect_close(errors, "source writeback_ms", float(source_match.group(4)), 13.3, 0.3)
        expect_equal(errors, "source id_only", int(source_match.group(5)), 1)
    if "kernel(ms)=0.0" in log_text:
        errors.append("source checked log still contains kernel(ms)=0.0")

    report_text = report.read_text(errors="replace")
    for phrase in (
        "two-stage source grouped GEMM",
        "不能直接替代现有主线",
        "kernel(ms)=5525.5",
    ):
        if phrase not in report_text:
            errors.append(f"source report missing phrase: {phrase}")

    source_text = source.read_text(errors="replace")
    for phrase in (
        "if (out_dist) out_dist",
        'read_env_int("UNG_GPU_SOURCE_EXACT_MODE", 0, 0, 1)',
        "source_exact stream failed",
    ):
        if phrase not in source_text:
            errors.append(f"source code missing safety phrase: {phrase}")

    return errors


def validate_sift30_universal_flat_db(item: ArtifactItem) -> list[str]:
    errors: list[str] = []
    build_log = Path(item.paths[0])
    build_csv = Path(item.paths[1])
    report = Path(item.paths[2])
    gpu_source = Path(item.paths[3])
    ung_source = Path(item.paths[4])

    build = read_metric_csv(build_csv)
    expect_close(errors, "universal build_cross_edges_time", build.get("build_cross_edges_time"), 3180.63, 0.05)

    log_text = build_log.read_text(errors="replace")
    db_match = re.search(
        r"\[cross_edges\] double_buffer chunks=(\d+) groups=(\d+) queries=(\d+) chunk_q=(\d+) "
        r"group_desc=(\d+) tile_desc=(\d+) H2D\(ms\)=([0-9.]+) Kernel\(ms\)=([0-9.]+) "
        r"D2H\(ms\)=([0-9.]+) writeback\(ms\)=([0-9.]+)",
        log_text,
    )
    if not db_match:
        errors.append("universal double_buffer line missing")
    else:
        expect_equal(errors, "universal chunks", int(db_match.group(1)), 94)
        expect_equal(errors, "universal groups", int(db_match.group(2)), 69146)
        expect_equal(errors, "universal queries", int(db_match.group(3)), 23901768)
        expect_equal(errors, "universal group_desc", int(db_match.group(5)), 13634576)
        expect_equal(errors, "universal tile_desc", int(db_match.group(6)), 642082)
        expect_close(errors, "universal h2d_ms", float(db_match.group(7)), 30.8, 0.2)
        expect_close(errors, "universal kernel_ms", float(db_match.group(8)), 3870.4, 0.3)
        expect_close(errors, "universal d2h_ms", float(db_match.group(9)), 2.1, 0.2)
        expect_close(errors, "universal writeback_ms", float(db_match.group(10)), 15.5, 0.3)

    for phrase in (
        "flat_id_active=1",
    ):
        if phrase not in log_text:
            errors.append(f"universal build log missing phrase: {phrase}")
    if "build_cross_edges_time,3180.63" not in build_csv.read_text(errors="replace"):
        errors.append("universal build csv missing exact cross-edge value")

    report_text = report.read_text(errors="replace")
    for phrase in (
        "universal all-DB",
        "3180.6 ms",
        "flat_id_active=1",
        "不是最终上限",
    ):
        if phrase not in report_text:
            errors.append(f"universal report missing phrase: {phrase}")

    gpu_text = gpu_source.read_text(errors="replace")
    for phrase in (
        "universal_gpu_route",
        'read_env_int("UNG_GPU_DB_LARGE_MAX_NX", universal_gpu_route ? 1048576 : 1023',
        "cross_group_neighbor_flat_ids != nullptr",
    ):
        if phrase not in gpu_text:
            errors.append(f"universal gpu source missing phrase: {phrase}")

    ung_text = ung_source.read_text(errors="replace")
    for phrase in (
        'read_env_int_local("UNG_UNIVERSAL_GPU", 0, 0, 1)',
        'read_env_int_local("UNG_GPU_FLAT_ID_WRITEBACK", universal_gpu_route ? 1 : 0',
        "skip_searchqueue_storage",
    ):
        if phrase not in ung_text:
            errors.append(f"universal ung source missing phrase: {phrase}")

    return errors


def validate_x200_universal_flat_full_quality(item: ArtifactItem) -> list[str]:
    errors: list[str] = []
    build_log = Path(item.paths[0])
    build_csv = Path(item.paths[1])
    search_csv = Path(item.paths[2])
    env_file = Path(item.paths[3])
    report = Path(item.paths[4])

    build = read_metric_csv(build_csv)
    expect_close(errors, "x200 universal index_time", build.get("index_time"), 14233.8, 0.1)
    expect_close(errors, "x200 universal build_graph_time", build.get("build_graph_time"), 8049.75, 0.1)
    expect_close(errors, "x200 universal build_cross_edges_time", build.get("build_cross_edges_time"), 2494.21, 0.05)

    log_text = build_log.read_text(errors="replace")
    for phrase in (
        "[UNG config] additional_edges_impl=cpu_vamana",
        "flat_id_active=1",
        "additional(ms)=843.5",
        "merge_cross(ms)=0.7",
    ):
        if phrase not in log_text:
            errors.append(f"x200 universal build log missing phrase: {phrase}")

    db_match = re.search(
        r"\[cross_edges\] double_buffer chunks=(\d+) groups=(\d+) queries=(\d+) chunk_q=(\d+) "
        r"group_desc=(\d+) tile_desc=(\d+) H2D\(ms\)=([0-9.]+) Kernel\(ms\)=([0-9.]+) "
        r"D2H\(ms\)=([0-9.]+) writeback\(ms\)=([0-9.]+)",
        log_text,
    )
    if not db_match:
        errors.append("x200 universal double_buffer line missing")
    else:
        expect_equal(errors, "x200 universal chunks", int(db_match.group(1)), 28)
        expect_equal(errors, "x200 universal groups", int(db_match.group(2)), 5195)
        expect_equal(errors, "x200 universal queries", int(db_match.group(3)), 7292200)
        expect_equal(errors, "x200 universal tile_desc", int(db_match.group(6)), 456807)
        expect_close(errors, "x200 universal h2d_ms", float(db_match.group(7)), 2.9, 0.2)
        expect_close(errors, "x200 universal kernel_ms", float(db_match.group(8)), 1887.5, 0.3)
        expect_close(errors, "x200 universal d2h_ms", float(db_match.group(9)), 1.2, 0.2)
        expect_close(errors, "x200 universal writeback_ms", float(db_match.group(10)), 19.2, 0.3)

    rows = list(csv.DictReader(search_csv.open()))
    recall_by_l = {row["Lsearch"]: row for row in rows}
    if "1000" not in recall_by_l or "5000" not in recall_by_l:
        errors.append("x200 universal search summary missing L1000 or L5000")
    else:
        expect_close(errors, "x200 universal L1000 recall", float(recall_by_l["1000"]["Average_Recall"]), 0.871, 0.0005)
        expect_close(errors, "x200 universal L5000 recall", float(recall_by_l["5000"]["Average_Recall"]), 0.911, 0.0005)

    env_text = env_file.read_text(errors="replace")
    for phrase in (
        "UNG_UNIVERSAL_GPU=1",
        "UNG_ADDITIONAL_EDGES_IMPL=0",
        "UNG_GROUP_GRAPH_IMPL=0",
    ):
        if phrase not in env_text:
            errors.append(f"x200 universal env missing phrase: {phrase}")

    report_text = report.read_text(errors="replace")
    for phrase in (
        "Amazon x200 full-quality smoke",
        "2494.21 ms",
        "0.871 / 0.911",
        "不能把它写成全局完成的普遍替代",
    ):
        if phrase not in report_text:
            errors.append(f"x200 universal report missing phrase: {phrase}")

    return errors


def validate_x100_universal_flat_full_quality_boundary(item: ArtifactItem) -> list[str]:
    errors: list[str] = []
    build_log = Path(item.paths[0])
    build_csv = Path(item.paths[1])
    search_csv = Path(item.paths[2])
    env_file = Path(item.paths[3])
    report = Path(item.paths[4])

    build = read_metric_csv(build_csv)
    expect_close(errors, "x100 universal index_time", build.get("index_time"), 7461.27, 0.1)
    expect_close(errors, "x100 universal build_graph_time", build.get("build_graph_time"), 3314.27, 0.1)
    expect_close(errors, "x100 universal build_cross_edges_time", build.get("build_cross_edges_time"), 1689.4, 0.05)

    log_text = build_log.read_text(errors="replace")
    for phrase in (
        "[UNG config] additional_edges_impl=cpu_vamana",
        "flat_id_active=1",
        "additional(ms)=675.3",
        "merge_cross(ms)=0.7",
    ):
        if phrase not in log_text:
            errors.append(f"x100 universal build log missing phrase: {phrase}")

    db_match = re.search(
        r"\[cross_edges\] double_buffer chunks=(\d+) groups=(\d+) queries=(\d+) chunk_q=(\d+) "
        r"group_desc=(\d+) tile_desc=(\d+) H2D\(ms\)=([0-9.]+) Kernel\(ms\)=([0-9.]+) "
        r"D2H\(ms\)=([0-9.]+) writeback\(ms\)=([0-9.]+)",
        log_text,
    )
    if not db_match:
        errors.append("x100 universal double_buffer line missing")
    else:
        expect_equal(errors, "x100 universal chunks", int(db_match.group(1)), 14)
        expect_equal(errors, "x100 universal groups", int(db_match.group(2)), 5195)
        expect_equal(errors, "x100 universal queries", int(db_match.group(3)), 3646100)
        expect_equal(errors, "x100 universal group_desc", int(db_match.group(5)), 3568700)
        expect_equal(errors, "x100 universal tile_desc", int(db_match.group(6)), 4888)
        expect_close(errors, "x100 universal h2d_ms", float(db_match.group(7)), 3.6, 0.2)
        expect_close(errors, "x100 universal kernel_ms", float(db_match.group(8)), 872.0, 0.3)
        expect_close(errors, "x100 universal d2h_ms", float(db_match.group(9)), 0.6, 0.2)
        expect_close(errors, "x100 universal writeback_ms", float(db_match.group(10)), 6.8, 0.3)

    rows = list(csv.DictReader(search_csv.open()))
    recall_by_l = {row["Lsearch"]: row for row in rows}
    for lsearch, expected in (("100", 0.826), ("500", 0.868), ("1000", 0.891)):
        if lsearch not in recall_by_l:
            errors.append(f"x100 universal search summary missing L{lsearch}")
        else:
            expect_close(
                errors,
                f"x100 universal L{lsearch} recall",
                float(recall_by_l[lsearch]["Average_Recall"]),
                expected,
                0.0005,
            )

    env_text = env_file.read_text(errors="replace")
    for phrase in (
        "UNG_UNIVERSAL_GPU=1",
        "UNG_ADDITIONAL_EDGES_IMPL=0",
        "UNG_GROUP_GRAPH_IMPL=0",
    ):
        if phrase not in env_text:
            errors.append(f"x100 universal env missing phrase: {phrase}")

    report_text = report.read_text(errors="replace")
    for phrase in (
        "Amazon x100 full-quality boundary",
        "1689.40 ms",
        "0.826 / 0.868 / 0.891",
        "不能写成无条件端到端加速",
    ):
        if phrase not in report_text:
            errors.append(f"x100 universal report missing phrase: {phrase}")

    return errors


def validate_x100_cross_fairness(item: ArtifactItem) -> list[str]:
    errors: list[str] = []
    csv_path = Path(item.paths[1])
    rows = read_rows_by_key(csv_path, "variant")
    expected = {
        "cpu_vamana": {
            "index_ms": 23150.0,
            "group_ms": 3266.33,
            "cross_ms": 18735.8,
            "threads": "128",
            "cross_edge_impl": "cpu_vamana",
            "gpu_topk_impl": "auto",
        },
        "cpu_exact_128t": {
            "index_ms": 6995.92,
            "group_ms": 3266.23,
            "cross_ms": 2389.81,
            "threads": "128",
            "cross_edge_impl": "cpu_exact_scan",
            "gpu_topk_impl": "auto",
        },
        "cuvs_per_group": {
            "index_ms": 14353.4,
            "group_ms": 7983.65,
            "cross_ms": 3730.49,
            "threads": "128",
            "cross_edge_impl": "cuvs_brute_force",
            "gpu_topk_impl": "auto",
            "gpu_kernel_ms": 821.8,
        },
        "sgemm_topk": {
            "index_ms": 7452.41,
            "group_ms": 3399.42,
            "cross_ms": 1395.88,
            "threads": "128",
            "cross_edge_impl": "gpu_batched",
            "gpu_topk_impl": "sgemm_topk",
            "gpu_kernel_ms": 541.6,
        },
        "final_fused": {
            "index_ms": 6630.79,
            "group_ms": 3373.19,
            "cross_ms": 848.873,
            "threads": "128",
            "cross_edge_impl": "gpu_batched",
            "gpu_topk_impl": "fused_group_topk",
            "gpu_h2d_ms": 74.1,
            "gpu_kernel_ms": 232.8,
            "gpu_d2h_ms": 3.6,
        },
    }
    for variant, fields in expected.items():
        row = rows.get(variant)
        if not row:
            errors.append(f"x100 fairness missing variant {variant}")
            continue
        for field, want in fields.items():
            got = row.get(field)
            if isinstance(want, str):
                if got != want:
                    errors.append(f"{variant}.{field}: got {got}, expected {want}")
            else:
                expect_close(errors, f"{variant}.{field}", float(got or "nan"), want, 1e-3)

    try:
        cpu_vamana = float(rows["cpu_vamana"]["cross_ms"])
        cpu_exact = float(rows["cpu_exact_128t"]["cross_ms"])
        cuvs = float(rows["cuvs_per_group"]["cross_ms"])
        sgemm = float(rows["sgemm_topk"]["cross_ms"])
        final = float(rows["final_fused"]["cross_ms"])
    except (KeyError, ValueError):
        errors.append("x100 fairness cannot compute speedups")
        return errors
    expect_close(errors, "final vs cpu_vamana speedup", round(cpu_vamana / final, 2), 22.07, 0.0)
    expect_close(errors, "final vs cpu_exact speedup", round(cpu_exact / final, 2), 2.82, 0.0)
    expect_close(errors, "final vs cuvs speedup", round(cuvs / final, 2), 4.39, 0.0)
    expect_close(errors, "final vs sgemm speedup", round(sgemm / final, 2), 1.64, 0.0)
    return errors


def validate_mixed_router_x200(item: ArtifactItem) -> list[str]:
    errors: list[str] = []
    summary = Path(item.paths[1])
    seen: dict[tuple[str, int], dict[str, str]] = {}
    with summary.open() as f:
        for row in csv.DictReader(f):
            try:
                lsearch = int(float(row["lsearch"]))
            except (KeyError, TypeError, ValueError):
                continue
            seen[(row.get("case", ""), lsearch)] = row

    cpu = seen.get(("cpu_vamana", 1000))
    router = seen.get(("router_exact_nx256", 1000))
    if not cpu or not router:
        errors.append("missing cpu_vamana/router_exact_nx256 L1000 rows")
        return errors
    expect_close(errors, "cpu group_ms", float(cpu.get("group_ms", "nan")), 7812.0, 0.05)
    expect_close(errors, "cpu L1000 recall", float(cpu.get("avg_recall", "nan")), 0.869, 1e-6)
    expect_close(errors, "router256 group_ms", float(router.get("group_ms", "nan")), 5770.88, 0.05)
    expect_close(errors, "router256 L1000 recall", float(router.get("avg_recall", "nan")), 0.869, 1e-6)
    expect_close(errors, "router256 group speedup", float(router.get("group_speedup_vs_cpu", "nan")), 1.353693024287457, 1e-9)
    return errors


def read_rows_by_key(path: Path, key_name: str) -> dict[str, dict[str, str]]:
    rows: dict[str, dict[str, str]] = {}
    with path.open() as f:
        for row in csv.DictReader(f):
            key = row.get(key_name, "")
            if key:
                rows[key] = row
    return rows


def validate_packed_exact_sweep(item: ArtifactItem) -> list[str]:
    errors: list[str] = []
    x200_recalls = read_search_recall_csv(Path(item.paths[0]))
    x400_recalls = read_search_recall_csv(Path(item.paths[1]))
    expect_close(errors, "x200 L1000 recall", x200_recalls.get(1000), 0.869, 1e-6)
    expect_close(errors, "x200 L5000 recall", x200_recalls.get(5000), 0.908, 1e-6)
    expect_close(errors, "x400 L1000 recall", x400_recalls.get(1000), 0.945, 1e-6)
    expect_close(errors, "x400 L5000 recall", x400_recalls.get(5000), 0.967, 1e-6)

    expected = {
        ("x200_packed", "1000"): (515.379, 0.869),
        ("x200_packed", "5000"): (468.162, 0.908),
        ("x400_packed", "1000"): (287.988, 0.945),
        ("x400_packed", "5000"): (289.639, 0.967),
    }
    keyed: dict[tuple[str, str], dict[str, str]] = {}
    with Path(item.paths[2]).open() as f:
        for row in csv.DictReader(f):
            keyed[(row.get("variant", ""), row.get("lsearch", ""))] = row
    for (variant, lsearch), (time_ms, recall) in expected.items():
        row = keyed.get((variant, lsearch))
        if not row:
            errors.append(f"packed summary missing {variant} L{lsearch}")
            continue
        expect_close(errors, f"{variant} L{lsearch} batch_time_ms",
                     float(row.get("batch_time_ms", "nan")), time_ms, 1e-3)
        expect_close(errors, f"{variant} L{lsearch} avg_recall",
                     float(row.get("avg_recall", "nan")), recall, 1e-6)
    return errors


def validate_x200_packed_exact_direct_h2d(item: ArtifactItem) -> list[str]:
    errors: list[str] = []
    old_build = read_metric_csv(Path(item.paths[0]))
    old_log = Path(item.paths[1]).read_text(errors="replace")
    fill16_build = read_metric_csv(Path(item.paths[2]))
    fill16_log = Path(item.paths[3]).read_text(errors="replace")
    default_build = read_metric_csv(Path(item.paths[4]))
    default_log = Path(item.paths[5]).read_text(errors="replace")
    default_recalls = read_search_recall_csv(Path(item.paths[6]))

    expect_close(errors, "old index_time", old_build.get("index_time"), 19135.1, 0.05)
    expect_close(errors, "old build_graph_time", old_build.get("build_graph_time"), 8999.26, 0.05)
    expect_close(errors, "old tagore_pack_time", old_build.get("tagore_pack_time"), 4140.61, 0.01)
    expect_close(errors, "old tagore_gnn_time", old_build.get("tagore_gnn_time"), 931.718, 0.001)
    expect_close(errors, "old tagore_fill_time", old_build.get("tagore_fill_time"), 2849.5, 0.01)
    if "[FastExact] direct_h2d=0" not in old_log or "device_lookup=0" not in old_log:
        errors.append("old direct-H2D baseline log should show direct_h2d=0 and device_lookup=0")

    expect_close(errors, "fill16 index_time", fill16_build.get("index_time"), 11441.3, 0.05)
    expect_close(errors, "fill16 build_graph_time", fill16_build.get("build_graph_time"), 3204.88, 0.05)
    expect_close(errors, "fill16 tagore_pack_time", fill16_build.get("tagore_pack_time"), 0.043387, 1e-6)
    expect_close(errors, "fill16 tagore_gnn_time", fill16_build.get("tagore_gnn_time"), 928.669, 0.001)
    expect_close(errors, "fill16 tagore_fill_time", fill16_build.get("tagore_fill_time"), 1383.53, 0.01)
    if "[FastExact] direct_h2d=1" not in fill16_log or "device_lookup=1" not in fill16_log:
        errors.append("fill16 optimized log should show direct_h2d=1 and device_lookup=1")

    expect_close(errors, "default build_graph_time", default_build.get("build_graph_time"), 3579.27, 0.05)
    expect_close(errors, "default tagore_pack_time", default_build.get("tagore_pack_time"), 0.047983, 1e-6)
    expect_close(errors, "default tagore_fill_time", default_build.get("tagore_fill_time"), 1553.28, 0.01)
    expect_close(errors, "default L1000 recall", default_recalls.get(1000), 0.867, 1e-6)
    if "[FastExact] direct_h2d=1" not in default_log or "device_lookup=1" not in default_log:
        errors.append("default optimized log should show direct_h2d=1 and device_lookup=1")

    old_group = old_build.get("build_graph_time")
    fill16_group = fill16_build.get("build_graph_time")
    if old_group is not None and fill16_group is not None and not (old_group / fill16_group > 2.8):
        errors.append(f"expected fill16 group speedup >2.8x, got {old_group / fill16_group:.3f}x")
    return errors


def validate_x200_exact_warp_fill_negative(item: ArtifactItem) -> list[str]:
    errors: list[str] = []
    block_build = read_metric_csv(Path(item.paths[0]))
    block_log = Path(item.paths[1]).read_text(errors="replace")
    block_recalls = read_search_recall_csv(Path(item.paths[2]))
    warp_build = read_metric_csv(Path(item.paths[3]))
    warp_log = Path(item.paths[4]).read_text(errors="replace")
    warp_recalls = read_search_recall_csv(Path(item.paths[5]))
    fill8_build = read_metric_csv(Path(item.paths[6]))
    fill8_recalls = read_search_recall_csv(Path(item.paths[7]))
    fill64_build = read_metric_csv(Path(item.paths[8]))
    fill64_recalls = read_search_recall_csv(Path(item.paths[9]))

    expect_close(errors, "block index_time", block_build.get("index_time"), 12744.2, 0.05)
    expect_close(errors, "block build_graph_time", block_build.get("build_graph_time"), 4031.36, 0.05)
    expect_close(errors, "block tagore_gnn_time", block_build.get("tagore_gnn_time"), 841.136, 0.001)
    expect_close(errors, "block tagore_fill_time", block_build.get("tagore_fill_time"), 1682.11, 0.01)
    expect_close(errors, "block L1000 recall", block_recalls.get(1000), 0.867, 1e-6)
    if "warp_kernel=0" not in block_log:
        errors.append("block baseline log should show warp_kernel=0")

    expect_close(errors, "warp index_time", warp_build.get("index_time"), 14773.8, 0.05)
    expect_close(errors, "warp build_graph_time", warp_build.get("build_graph_time"), 4334.28, 0.05)
    expect_close(errors, "warp tagore_gnn_time", warp_build.get("tagore_gnn_time"), 987.27, 0.001)
    expect_close(errors, "warp tagore_fill_time", warp_build.get("tagore_fill_time"), 1855.18, 0.01)
    expect_close(errors, "warp L1000 recall", warp_recalls.get(1000), 0.867, 1e-6)
    if "warp_kernel=1" not in warp_log:
        errors.append("warp ablation log should show warp_kernel=1")

    expect_close(errors, "fill8 build_graph_time", fill8_build.get("build_graph_time"), 4475.57, 0.05)
    expect_close(errors, "fill8 tagore_gnn_time", fill8_build.get("tagore_gnn_time"), 844.526, 0.001)
    expect_close(errors, "fill8 tagore_fill_time", fill8_build.get("tagore_fill_time"), 2113.86, 0.01)
    expect_close(errors, "fill8 L1000 recall", fill8_recalls.get(1000), 0.867, 1e-6)

    expect_close(errors, "fill64 build_graph_time", fill64_build.get("build_graph_time"), 4857.36, 0.05)
    expect_close(errors, "fill64 tagore_gnn_time", fill64_build.get("tagore_gnn_time"), 839.703, 0.001)
    expect_close(errors, "fill64 tagore_fill_time", fill64_build.get("tagore_fill_time"), 2321.96, 0.01)
    expect_close(errors, "fill64 L1000 recall", fill64_recalls.get(1000), 0.867, 1e-6)

    block_gnn = block_build.get("tagore_gnn_time")
    warp_gnn = warp_build.get("tagore_gnn_time")
    if block_gnn is not None and warp_gnn is not None and not (warp_gnn > block_gnn):
        errors.append(f"warp kernel should be slower than block kernel in this ablation, got {warp_gnn} <= {block_gnn}")
    block_fill = block_build.get("tagore_fill_time")
    fill8 = fill8_build.get("tagore_fill_time")
    fill64 = fill64_build.get("tagore_fill_time")
    if block_fill is not None and fill8 is not None and not (fill8 > block_fill):
        errors.append(f"fill8 should be slower than default fill16, got {fill8} <= {block_fill}")
    if block_fill is not None and fill64 is not None and not (fill64 > block_fill):
        errors.append(f"fill64 should be slower than default fill16, got {fill64} <= {block_fill}")
    return errors


def parse_graph_reserve_line(text: str) -> dict[str, str]:
    match = re.search(r"\[graph_reserve\] ([^\n]+)", text)
    if not match:
        return {}
    values: dict[str, str] = {}
    for key, value in re.findall(r"([A-Za-z0-9_]+)=([^ ]+)", match.group(1)):
        values[key] = value
    return values


def parse_cross_breakdown_line(text: str) -> dict[str, float]:
    match = re.search(r"\[cross_edges\] breakdown ([^\n]+)", text)
    if not match:
        return {}
    values: dict[str, float] = {}
    for key, value in re.findall(r"([A-Za-z0-9_]+)\(ms\)=([0-9.]+)", match.group(1)):
        values[key] = float(value)
    for key, value in re.findall(r"([A-Za-z0-9_]+)=([0-9.]+)", match.group(1)):
        values.setdefault(key, float(value))
    return values


def parse_tagore_summary_line(text: str) -> dict[str, float]:
    match = re.search(r"- TagoreCuda groups:[^\n]+", text)
    if not match:
        return {}
    values: dict[str, float] = {}
    for key, value in re.findall(r"([A-Za-z0-9_]+):\s*([0-9.]+)", match.group(0)):
        try:
            values[key] = float(value)
        except ValueError:
            continue
    return values


def validate_graph_output_boundary_reserve(item: ArtifactItem) -> list[str]:
    errors: list[str] = []
    off_build = read_metric_csv(Path(item.paths[0]))
    off_log = Path(item.paths[1]).read_text(errors="replace")
    on_build = read_metric_csv(Path(item.paths[2]))
    on_log = Path(item.paths[3]).read_text(errors="replace")
    x200_forced_build = read_metric_csv(Path(item.paths[4]))
    x200_forced_log = Path(item.paths[5]).read_text(errors="replace")
    x200_auto_off_build = read_metric_csv(Path(item.paths[6]))
    x200_auto_off_log = Path(item.paths[7]).read_text(errors="replace")
    x200_auto_build = read_metric_csv(Path(item.paths[8]))
    x200_auto_log = Path(item.paths[9]).read_text(errors="replace")

    expect_close(errors, "10x40 reserve-off index", off_build.get("index_time"), 39746.1, 0.05)
    expect_close(errors, "10x40 reserve-off build_graph", off_build.get("build_graph_time"), 14238.1, 0.05)
    expect_close(errors, "10x40 reserve-off cross", off_build.get("build_cross_edges_time"), 14761.0, 0.05)
    expect_close(errors, "10x40 reserve-off tagore_fill", off_build.get("tagore_fill_time"), 823.488, 0.001)
    off_tagore = parse_tagore_summary_line(off_log)
    off_cross = parse_cross_breakdown_line(off_log)
    expect_close(errors, "10x40 reserve-off fallback_wall", off_tagore.get("fallback_wall"), 11844.6, 0.01)
    expect_close(errors, "10x40 reserve-off merge_cross", off_cross.get("merge_cross"), 1677.1, 0.01)

    expect_close(errors, "10x40 reserve-on index", on_build.get("index_time"), 24704.5, 0.05)
    expect_close(errors, "10x40 reserve-on build_graph", on_build.get("build_graph_time"), 1662.69, 0.01)
    expect_close(errors, "10x40 reserve-on cross", on_build.get("build_cross_edges_time"), 9993.95, 0.01)
    expect_close(errors, "10x40 reserve-on tagore_fill", on_build.get("tagore_fill_time"), 23.924, 0.001)
    on_reserve = parse_graph_reserve_line(on_log)
    if not on_reserve:
        errors.append("10x40 reserve-on graph_reserve line missing")
    else:
        expect_equal(errors, "10x40 reserve-on enabled", int(on_reserve.get("enabled", "-1")), 1)
        expect_equal(errors, "10x40 reserve-on capacity_edges", int(on_reserve.get("capacity_edges", "-1")), 106031200)
        expect_close(errors, "10x40 reserve-on reserve_ms", float(on_reserve.get("ms", "nan")), 6487.28, 0.01)
    on_tagore = parse_tagore_summary_line(on_log)
    on_cross = parse_cross_breakdown_line(on_log)
    expect_close(errors, "10x40 reserve-on fallback_wall", on_tagore.get("fallback_wall"), 520.2, 0.01)
    expect_close(errors, "10x40 reserve-on merge_cross", on_cross.get("merge_cross"), 2.3, 0.01)

    off_index = off_build.get("index_time")
    on_index = on_build.get("index_time")
    if off_index is not None and on_index is not None and not (off_index / on_index > 1.6):
        errors.append(f"10x40 reserve should improve index by >1.6x, got {off_index / on_index:.3f}x")
    off_fill = off_build.get("tagore_fill_time")
    on_fill = on_build.get("tagore_fill_time")
    if off_fill is not None and on_fill is not None and not (off_fill / on_fill > 30.0):
        errors.append(f"10x40 reserve should reduce tagore fill by >30x, got {off_fill / on_fill:.3f}x")
    if off_cross and on_cross and not (off_cross.get("merge_cross", 0.0) / on_cross.get("merge_cross", 1e30) > 700.0):
        errors.append("10x40 reserve should reduce cross merge by >700x")

    expect_close(errors, "x200 forced-reserve index", x200_forced_build.get("index_time"), 15436.9, 0.05)
    expect_close(errors, "x200 forced-reserve tagore_fill", x200_forced_build.get("tagore_fill_time"), 150.85, 0.01)
    forced_reserve = parse_graph_reserve_line(x200_forced_log)
    if not forced_reserve:
        errors.append("x200 forced-reserve graph_reserve line missing")
    else:
        expect_equal(errors, "x200 forced-reserve enabled", int(forced_reserve.get("enabled", "-1")), 1)
        expect_equal(errors, "x200 forced-reserve capacity_edges", int(forced_reserve.get("capacity_edges", "-1")), 53011200)

    expect_close(errors, "x200 pre-threshold-auto-off index", x200_auto_off_build.get("index_time"), 16895.4, 0.05)
    expect_close(errors, "x200 pre-threshold-auto-off tagore_fill", x200_auto_off_build.get("tagore_fill_time"), 2496.96, 0.01)
    auto_off_reserve = parse_graph_reserve_line(x200_auto_off_log)
    if not auto_off_reserve:
        errors.append("x200 pre-threshold auto-off graph_reserve line missing")
    else:
        expect_equal(errors, "x200 pre-threshold auto-off enabled", int(auto_off_reserve.get("enabled", "-1")), 0)
        if auto_off_reserve.get("mode") != "auto":
            errors.append(f"x200 pre-threshold auto-off mode: got {auto_off_reserve.get('mode')}, expected auto")

    expect_close(errors, "x200 final-auto index", x200_auto_build.get("index_time"), 16395.0, 0.05)
    expect_close(errors, "x200 final-auto tagore_fill", x200_auto_build.get("tagore_fill_time"), 224.794, 0.001)
    final_auto_reserve = parse_graph_reserve_line(x200_auto_log)
    if not final_auto_reserve:
        errors.append("x200 final-auto graph_reserve line missing")
    else:
        expect_equal(errors, "x200 final-auto enabled", int(final_auto_reserve.get("enabled", "-1")), 1)
        if final_auto_reserve.get("mode") != "auto":
            errors.append(f"x200 final-auto mode: got {final_auto_reserve.get('mode')}, expected auto")
        expect_equal(errors, "x200 final-auto capacity_edges", int(final_auto_reserve.get("capacity_edges", "-1")), 53011200)
    auto_cross = parse_cross_breakdown_line(x200_auto_log)
    expect_close(errors, "x200 final-auto merge_cross", auto_cross.get("merge_cross"), 0.9, 0.01)

    forced_fill = x200_forced_build.get("tagore_fill_time")
    auto_off_fill = x200_auto_off_build.get("tagore_fill_time")
    final_auto_fill = x200_auto_build.get("tagore_fill_time")
    if auto_off_fill is not None and final_auto_fill is not None and not (auto_off_fill / final_auto_fill > 10.0):
        errors.append(f"x200 auto reserve should reduce fill by >10x, got {auto_off_fill / final_auto_fill:.3f}x")
    if forced_fill is not None and final_auto_fill is not None and not (final_auto_fill < 2.0 * forced_fill):
        errors.append(
            f"x200 final auto fill should stay within 2x of forced reserve fill, got {final_auto_fill} vs {forced_fill}"
        )
    return errors


def validate_graph_output_boundary_neighborlist(item: ArtifactItem) -> list[str]:
    errors: list[str] = []
    graph_h = Path(item.paths[0]).read_text(errors="replace")
    if "class NeighborList" not in graph_h:
        errors.append("graph.h should define NeighborList")
    if "static constexpr size_t INLINE_CAPACITY = 64;" not in graph_h:
        errors.append("NeighborList retained implementation should use INLINE_CAPACITY=64")
    if "NeighborList* neighbors;" not in graph_h:
        errors.append("Graph::neighbors should use NeighborList*, not std::vector<IdxType>*")

    old_x200_build = read_metric_csv(Path(item.paths[1]))
    old_x200_log = Path(item.paths[2]).read_text(errors="replace")
    old_10x40_build = read_metric_csv(Path(item.paths[3]))
    old_10x40_log = Path(item.paths[4]).read_text(errors="replace")
    x200_64_build = read_metric_csv(Path(item.paths[5]))
    x200_64_log = Path(item.paths[6]).read_text(errors="replace")
    x200_64_noreserve_build = read_metric_csv(Path(item.paths[7]))
    x200_64_noreserve_log = Path(item.paths[8]).read_text(errors="replace")
    tenx40_64_build = read_metric_csv(Path(item.paths[9]))
    tenx40_64_log = Path(item.paths[10]).read_text(errors="replace")
    x200_48_build = read_metric_csv(Path(item.paths[11]))
    x200_48_log = Path(item.paths[12]).read_text(errors="replace")

    old_x200_reserve = parse_graph_reserve_line(old_x200_log)
    old_10x40_reserve = parse_graph_reserve_line(old_10x40_log)
    x200_64_reserve = parse_graph_reserve_line(x200_64_log)
    x200_64_noreserve = parse_graph_reserve_line(x200_64_noreserve_log)
    tenx40_64_reserve = parse_graph_reserve_line(tenx40_64_log)
    x200_48_reserve = parse_graph_reserve_line(x200_48_log)
    x200_64_tagore = parse_tagore_summary_line(x200_64_log)
    tenx40_64_tagore = parse_tagore_summary_line(tenx40_64_log)
    x200_64_cross = parse_cross_breakdown_line(x200_64_log)
    tenx40_64_cross = parse_cross_breakdown_line(tenx40_64_log)
    x200_48_cross = parse_cross_breakdown_line(x200_48_log)

    expect_close(errors, "x200 NeighborList64 index", x200_64_build.get("index_time"), 15716.6, 0.05)
    expect_close(errors, "x200 NeighborList64 build_graph", x200_64_build.get("build_graph_time"), 6984.64, 0.01)
    expect_close(errors, "x200 NeighborList64 cross", x200_64_build.get("build_cross_edges_time"), 3905.56, 0.01)
    expect_close(errors, "x200 NeighborList64 tagore_fill", x200_64_build.get("tagore_fill_time"), 14.7462, 0.001)
    expect_close(errors, "x200 NeighborList64 reserve_ms", float(x200_64_reserve.get("ms", "nan")), 25.4343, 0.001)
    expect_equal(errors, "x200 NeighborList64 reserve enabled", int(x200_64_reserve.get("enabled", "-1")), 1)
    expect_close(errors, "x200 NeighborList64 fallback_wall", x200_64_tagore.get("fallback_wall"), 6.47247, 0.001)
    expect_close(errors, "x200 NeighborList64 merge_cross", x200_64_cross.get("merge_cross"), 1.5, 0.01)

    expect_close(errors, "x200 NeighborList64 noreserve index", x200_64_noreserve_build.get("index_time"), 15891.0, 0.05)
    expect_close(errors, "x200 NeighborList64 noreserve tagore_fill", x200_64_noreserve_build.get("tagore_fill_time"), 142.453, 0.001)
    expect_equal(errors, "x200 NeighborList64 noreserve disabled", int(x200_64_noreserve.get("enabled", "-1")), 0)

    expect_close(errors, "10x40 NeighborList64 index", tenx40_64_build.get("index_time"), 26167.8, 0.05)
    expect_close(errors, "10x40 NeighborList64 build_graph", tenx40_64_build.get("build_graph_time"), 1981.32, 0.01)
    expect_close(errors, "10x40 NeighborList64 tagore_fill", tenx40_64_build.get("tagore_fill_time"), 7.62981, 0.001)
    expect_close(errors, "10x40 NeighborList64 reserve_ms", float(tenx40_64_reserve.get("ms", "nan")), 21.1545, 0.001)
    expect_close(errors, "10x40 NeighborList64 fallback_wall", tenx40_64_tagore.get("fallback_wall"), 435.938, 0.001)
    expect_close(errors, "10x40 NeighborList64 merge_cross", tenx40_64_cross.get("merge_cross"), 12.0, 0.01)

    expect_close(errors, "x200 NeighborList48 tagore_fill", x200_48_build.get("tagore_fill_time"), 202.546, 0.001)
    expect_close(errors, "x200 NeighborList48 merge_cross", x200_48_cross.get("merge_cross"), 35.4, 0.01)
    expect_close(errors, "x200 NeighborList48 reserve_ms", float(x200_48_reserve.get("ms", "nan")), 25.1182, 0.001)

    old_x200_fill = old_x200_build.get("tagore_fill_time")
    new_x200_fill = x200_64_build.get("tagore_fill_time")
    if old_x200_fill is not None and new_x200_fill is not None and not (old_x200_fill / new_x200_fill > 10.0):
        errors.append(f"NeighborList64 should reduce x200 fill by >10x vs old reserve, got {old_x200_fill / new_x200_fill:.3f}x")

    old_x200_reserve_ms = float(old_x200_reserve.get("ms", "nan"))
    new_x200_reserve_ms = float(x200_64_reserve.get("ms", "nan"))
    if not (old_x200_reserve_ms / new_x200_reserve_ms > 100.0):
        errors.append(f"NeighborList64 should reduce x200 reserve by >100x, got {old_x200_reserve_ms / new_x200_reserve_ms:.3f}x")

    old_10x40_reserve_ms = float(old_10x40_reserve.get("ms", "nan"))
    new_10x40_reserve_ms = float(tenx40_64_reserve.get("ms", "nan"))
    if not (old_10x40_reserve_ms / new_10x40_reserve_ms > 100.0):
        errors.append(
            f"NeighborList64 should reduce 10x40 reserve by >100x, got {old_10x40_reserve_ms / new_10x40_reserve_ms:.3f}x"
        )

    old_10x40_fill = old_10x40_build.get("tagore_fill_time")
    new_10x40_fill = tenx40_64_build.get("tagore_fill_time")
    if old_10x40_fill is not None and new_10x40_fill is not None and not (old_10x40_fill / new_10x40_fill > 3.0):
        errors.append(f"NeighborList64 should reduce 10x40 fill by >3x vs old reserve, got {old_10x40_fill / new_10x40_fill:.3f}x")

    x200_48_fill = x200_48_build.get("tagore_fill_time")
    if new_x200_fill is not None and x200_48_fill is not None and not (x200_48_fill / new_x200_fill > 10.0):
        errors.append(f"48-inline negative ablation should be >10x slower in fill than 64-inline, got {x200_48_fill / new_x200_fill:.3f}x")
    return errors


def validate_x400_reverse_tail(item: ArtifactItem) -> list[str]:
    errors: list[str] = []
    rows = read_rows_by_key(Path(item.paths[1]), "case")
    norepair = rows.get("light512_norepair")
    repair = rows.get("light512_repair")
    reverse = rows.get("light512_reverse_repair")
    if not norepair or not repair or not reverse:
        errors.append("missing one or more x400 reverse-tail cases")
        return errors

    expect_close(errors, "norepair index_time", float(norepair.get("index_time", "nan")), 30128.1, 0.05)
    expect_close(errors, "norepair L5000", float(norepair.get("L5000_recall", "nan")), 0.9574, 1e-6)
    expect_close(errors, "repair low_intra_le4", float(repair.get("low_intra_le4_ratio", "nan")),
                 9.669654714475432e-05, 1e-12)
    expect_close(errors, "reverse index_time", float(reverse.get("index_time", "nan")), 33044.9, 0.05)
    expect_close(errors, "reverse L1000", float(reverse.get("L1000_recall", "nan")), 0.945567, 1e-6)
    expect_close(errors, "reverse L5000", float(reverse.get("L5000_recall", "nan")), 0.9659, 1e-6)
    expect_close(errors, "reverse zero_intra_ratio", float(reverse.get("zero_intra_ratio", "nan")), 0.0, 0.0)
    expect_equal(errors, "reverse cross_edges", int(float(reverse.get("cross_edges", "-1"))), 2131224)
    return errors


def validate_x400_same_script_cpu_baseline(item: ArtifactItem) -> list[str]:
    errors: list[str] = []
    rows = read_rows_by_key(Path(item.paths[0]), "case")
    cpu = rows.get("cpu_vamana")
    if not cpu:
        errors.append("missing cpu_vamana row in x400 same-script summary")
        return errors

    expect_close(errors, "same-script CPU index_time", float(cpu.get("index_time", "nan")), 36257.3, 0.05)
    expect_close(errors, "same-script CPU build_graph", float(cpu.get("build_graph_time", "nan")), 17180.5, 0.05)
    expect_close(errors, "same-script CPU cross", float(cpu.get("build_cross_edges_time", "nan")), 11013.3, 0.05)
    expect_close(errors, "same-script CPU L1000", float(cpu.get("L1000_recall", "nan")), 0.946, 1e-6)
    expect_close(errors, "same-script CPU L5000", float(cpu.get("L5000_recall", "nan")), 0.965, 1e-6)
    expect_equal(errors, "same-script CPU intra_edges", int(float(cpu.get("intra_edges", "-1"))), 75767742)
    expect_equal(errors, "same-script CPU cross_edges", int(float(cpu.get("cross_edges", "-1"))), 2130576)
    expect_close(errors, "same-script CPU zero_intra_ratio", float(cpu.get("zero_intra_ratio", "nan")), 0.0, 0.0)
    expect_close(errors, "same-script CPU low_intra_le4_ratio",
                 float(cpu.get("low_intra_le4_ratio", "nan")), 4.150066401062417e-07, 1e-18)

    build = read_metric_csv(Path(item.paths[2]))
    expect_close(errors, "same-script build csv index", build.get("index_time"), 36257.3, 0.05)
    expect_close(errors, "same-script build csv build_graph", build.get("build_graph_time"), 17180.5, 0.05)
    expect_close(errors, "same-script build csv cross", build.get("build_cross_edges_time"), 11013.3, 0.05)

    recalls = read_search_recall_csv(Path(item.paths[3]))
    expect_close(errors, "same-script search L1000", recalls.get(1000), 0.946, 1e-6)
    expect_close(errors, "same-script search L5000", recalls.get(5000), 0.965, 1e-6)

    graph = json.loads(Path(item.paths[4]).read_text())
    expect_equal(errors, "same-script graph intra_edges", int(graph.get("intra_edges", -1)), 75767742)
    expect_equal(errors, "same-script graph cross_edges", int(graph.get("cross_edges", -1)), 2130576)
    expect_close(errors, "same-script graph zero_intra_ratio", float(graph.get("zero_intra_ratio", -1.0)), 0.0, 0.0)
    expect_close(errors, "same-script graph low_intra_le4_ratio",
                 float(graph.get("low_intra_le4_ratio", -1.0)), 4.150066401062417e-07, 1e-18)
    return errors


def validate_x400_compact_d2h_ablation(item: ArtifactItem) -> list[str]:
    errors: list[str] = []
    rows = read_rows_by_key(Path(item.paths[0]), "case")
    compact_on_row = rows.get("light512_repair")
    compact_off_row = rows.get("light512_repair_compact_off")
    if not compact_on_row or not compact_off_row:
        errors.append("missing compact on/off rows in x400 compact summary")
        return errors

    expect_close(errors, "compact-on summary index", float(compact_on_row.get("index_time", "nan")), 39759.1, 0.05)
    expect_close(errors, "compact-on summary group", float(compact_on_row.get("build_graph_time", "nan")), 10899.5, 0.05)
    expect_close(errors, "compact-on summary d2h", float(compact_on_row.get("tagore_d2h_time", "nan")), 294.133, 0.001)
    expect_close(errors, "compact-on summary fill", float(compact_on_row.get("tagore_fill_time", "nan")), 249.144, 0.001)
    expect_close(errors, "compact-on L5000", float(compact_on_row.get("L5000_recall", "nan")), 0.9584, 1e-6)

    expect_close(errors, "compact-off summary index", float(compact_off_row.get("index_time", "nan")), 33019.7, 0.05)
    expect_close(errors, "compact-off summary group", float(compact_off_row.get("build_graph_time", "nan")), 8878.46, 0.01)
    expect_close(errors, "compact-off summary d2h", float(compact_off_row.get("tagore_d2h_time", "nan")), 292.299, 0.001)
    expect_close(errors, "compact-off summary fill", float(compact_off_row.get("tagore_fill_time", "nan")), 70.4478, 0.001)
    expect_close(errors, "compact-off L5000", float(compact_off_row.get("L5000_recall", "nan")), 0.958633, 1e-6)

    compact_on_build = read_metric_csv(Path(item.paths[2]))
    compact_on_log = Path(item.paths[3]).read_text(errors="replace")
    compact_on_recalls = read_search_recall_csv(Path(item.paths[4]))
    compact_off_build = read_metric_csv(Path(item.paths[5]))
    compact_off_log = Path(item.paths[6]).read_text(errors="replace")
    compact_off_recalls = read_search_recall_csv(Path(item.paths[7]))
    compact_on_rerun = read_metric_csv(Path(item.paths[8]))
    compact_on_rerun_log = Path(item.paths[9]).read_text(errors="replace")

    compact_on_env = Path(str(item.paths[3]).replace("/others/build.log", "/others/env")).read_text(errors="replace")
    compact_off_env = Path(str(item.paths[6]).replace("/others/build.log", "/others/env")).read_text(errors="replace")

    if "UNG_TAGORE_COMPACT_D2H=1" not in compact_on_env:
        errors.append("compact-on log/env should show UNG_TAGORE_COMPACT_D2H=1")
    if "UNG_TAGORE_COMPACT_D2H=0" not in compact_off_env:
        errors.append("compact-off log/env should show UNG_TAGORE_COMPACT_D2H=0")
    # The build-only rerun was launched directly and does not emit the benchmark env file.
    # Its path name and exact metrics are used only to check that the compact-on result was repeatable.

    expect_close(errors, "compact-on build index", compact_on_build.get("index_time"), 39759.1, 0.05)
    expect_close(errors, "compact-on build h2d", compact_on_build.get("tagore_h2d_time"), 3669.21, 0.01)
    expect_close(errors, "compact-on build d2h", compact_on_build.get("tagore_d2h_time"), 294.133, 0.001)
    expect_close(errors, "compact-on build fill", compact_on_build.get("tagore_fill_time"), 249.144, 0.001)
    expect_close(errors, "compact-on L1000", compact_on_recalls.get(1000), 0.9394, 1e-6)
    expect_close(errors, "compact-on L5000", compact_on_recalls.get(5000), 0.9584, 1e-6)

    expect_close(errors, "compact-off build index", compact_off_build.get("index_time"), 33019.7, 0.05)
    expect_close(errors, "compact-off build h2d", compact_off_build.get("tagore_h2d_time"), 1825.54, 0.01)
    expect_close(errors, "compact-off build d2h", compact_off_build.get("tagore_d2h_time"), 292.299, 0.001)
    expect_close(errors, "compact-off build fill", compact_off_build.get("tagore_fill_time"), 70.4478, 0.001)
    expect_close(errors, "compact-off L1000", compact_off_recalls.get(1000), 0.9397, 1e-6)
    expect_close(errors, "compact-off L5000", compact_off_recalls.get(5000), 0.958633, 1e-6)

    expect_close(errors, "compact-on rerun index", compact_on_rerun.get("index_time"), 37845.6, 0.05)
    expect_close(errors, "compact-on rerun h2d", compact_on_rerun.get("tagore_h2d_time"), 3354.54, 0.01)
    expect_close(errors, "compact-on rerun d2h", compact_on_rerun.get("tagore_d2h_time"), 254.135, 0.001)
    expect_close(errors, "compact-on rerun fill", compact_on_rerun.get("tagore_fill_time"), 163.966, 0.001)

    on_d2h = compact_on_build.get("tagore_d2h_time")
    off_d2h = compact_off_build.get("tagore_d2h_time")
    if on_d2h is not None and off_d2h is not None and not (abs(on_d2h - off_d2h) < 5.0):
        errors.append(f"compact-D2H should not show a stable D2H win here; got on/off {on_d2h}/{off_d2h}")
    on_index = compact_on_build.get("index_time")
    off_index = compact_off_build.get("index_time")
    if on_index is not None and off_index is not None and not (on_index > off_index):
        errors.append(f"compact-on should be slower than compact-off in this negative ablation, got {on_index} <= {off_index}")
    rerun_index = compact_on_rerun.get("index_time")
    if rerun_index is not None and off_index is not None and not (rerun_index > off_index):
        errors.append(f"compact-on rerun should remain slower than compact-off, got {rerun_index} <= {off_index}")
    return errors


def validate_10x40_bounded_complete(item: ArtifactItem) -> list[str]:
    errors: list[str] = []
    complete64_summary = read_rows_by_key(Path(item.paths[0]), "variant").get("fastgrnnd_cpu_fallback")
    all_exact_summary = read_rows_by_key(Path(item.paths[2]), "variant").get("fastgrnnd_cpu_fallback")
    bounded_summary = read_rows_by_key(Path(item.paths[4]), "variant").get("fastgrnnd_cpu_fallback")
    if not complete64_summary or not all_exact_summary or not bounded_summary:
        errors.append("missing one or more 10%x40 summary rows")
        return errors

    expect_close(errors, "10x40 complete64 index", float(complete64_summary.get("index_ms", "nan")), 25035.9, 0.05)
    expect_close(errors, "10x40 complete64 group", float(complete64_summary.get("group_ms", "nan")), 6097.03, 0.05)
    expect_close(errors, "10x40 all-exact group", float(all_exact_summary.get("group_ms", "nan")), 7329.12, 0.05)
    expect_close(errors, "10x40 bounded index", float(bounded_summary.get("index_ms", "nan")), 22230.2, 0.05)
    expect_close(errors, "10x40 bounded group", float(bounded_summary.get("group_ms", "nan")), 5240.6, 0.05)
    expect_close(errors, "10x40 bounded L100", float(bounded_summary.get("avg_recall", "nan")), 0.867, 1e-6)

    complete64_build = read_metric_csv(Path(item.paths[1]))
    all_exact_build = read_metric_csv(Path(item.paths[3]))
    bounded_build = read_metric_csv(Path(item.paths[5]))
    expect_close(errors, "complete64 tagore_groups", complete64_build.get("tagore_groups"), 1297, 0.0)
    expect_close(errors, "complete64 tagore_points", complete64_build.get("tagore_points"), 308080, 0.5)
    expect_close(errors, "all-exact tagore_groups", all_exact_build.get("tagore_groups"), 53840, 0.0)
    expect_close(errors, "all-exact tagore_points", all_exact_build.get("tagore_points"), 2409800, 0.5)
    expect_close(errors, "all-exact tagore_pack", all_exact_build.get("tagore_pack_time"), 2728.81, 0.01)
    expect_close(errors, "all-exact tagore_h2d", all_exact_build.get("tagore_h2d_time"), 1014.8, 0.01)
    expect_close(errors, "all-exact tagore_fill", all_exact_build.get("tagore_fill_time"), 1968.77, 0.01)
    expect_close(errors, "bounded tagore_fill", bounded_build.get("tagore_fill_time"), 739.673, 0.001)

    bounded_recalls = read_search_recall_csv(Path(item.paths[6]))
    expect_close(errors, "bounded L100 recall", bounded_recalls.get(100), 0.867, 1e-6)
    expect_close(errors, "bounded L500 recall", bounded_recalls.get(500), 0.9, 1e-6)
    expect_close(errors, "bounded L1000 recall", bounded_recalls.get(1000), 0.909, 1e-6)
    return errors


def validate_10x40_lsearch_sweep(item: ArtifactItem) -> list[str]:
    errors: list[str] = []
    sweep_rows: dict[tuple[str, str], dict[str, str]] = {}
    with Path(item.paths[1]).open() as f:
        for row in csv.DictReader(f):
            sweep_rows[(row.get("variant", ""), row.get("lsearch", ""))] = row
    expected_recalls = {
        "50": (0.8457, 0.8476),
        "100": (0.8650, 0.8668),
        "200": (0.8840, 0.8830),
        "500": (0.8980, 0.8980),
        "1000": (0.9080, 0.9100),
        "2000": (0.9250, 0.9270),
        "5000": (0.9420, 0.9430),
    }
    for lsearch, (cpu_recall, fast_recall) in expected_recalls.items():
        cpu = sweep_rows.get(("cpu_vamana", lsearch))
        fast = sweep_rows.get(("fastgrnnd", lsearch))
        if not cpu or not fast:
            errors.append(f"missing 10%x40 L{lsearch} sweep rows")
            continue
        expect_close(errors, f"10x40 cpu L{lsearch}", float(cpu.get("avg_recall", "nan")), cpu_recall, 1e-6)
        expect_close(errors, f"10x40 fast L{lsearch}", float(fast.get("avg_recall", "nan")), fast_recall, 1e-6)
    cpu_l1000 = sweep_rows.get(("cpu_vamana", "1000"))
    fast_l1000 = sweep_rows.get(("fastgrnnd", "1000"))
    if cpu_l1000 and fast_l1000:
        expect_close(errors, "10x40 cpu index", float(cpu_l1000.get("index_ms", "nan")), 36053.3, 0.05)
        expect_close(errors, "10x40 fast index", float(fast_l1000.get("index_ms", "nan")), 23497.4, 0.05)
        expect_close(errors, "10x40 fast group", float(fast_l1000.get("group_ms", "nan")), 5147.17, 0.05)
    return errors


def validate_x100_repeat_group_ab(item: ArtifactItem) -> list[str]:
    errors: list[str] = []
    summary_rows: dict[tuple[str, str], dict[str, str]] = {}
    with Path(item.paths[0]).open() as f:
        for row in csv.DictReader(f):
            summary_rows[(row.get("variant", ""), row.get("lsearch", ""))] = row

    expected_summary = {
        ("cpu_vamana_group", "1000"): (6863.47, 3303.26, 1732.42, 0.890667),
        ("fastgrnnd_cpu_fallback", "1000"): (6370.67, 3216.21, 1871.51, 0.896),
    }
    for key, (index_ms, group_ms, cross_ms, recall) in expected_summary.items():
        row = summary_rows.get(key)
        if not row:
            errors.append(f"x100 repeat missing summary row {key}")
            continue
        prefix = f"x100 {key[0]} L{key[1]}"
        expect_close(errors, f"{prefix} index", float(row.get("index_ms", "nan")), index_ms, 0.01)
        expect_close(errors, f"{prefix} group", float(row.get("group_ms", "nan")), group_ms, 0.01)
        expect_close(errors, f"{prefix} cross", float(row.get("cross_ms", "nan")), cross_ms, 0.01)
        expect_close(errors, f"{prefix} recall", float(row.get("avg_recall", "nan")), recall, 1e-6)

    cpu_build = read_metric_csv(Path(item.paths[1]))
    fast_build = read_metric_csv(Path(item.paths[4]))
    expect_close(errors, "x100 cpu build_graph", cpu_build.get("build_graph_time"), 3303.26, 0.01)
    expect_close(errors, "x100 fast build_graph", fast_build.get("build_graph_time"), 3216.21, 0.01)
    expect_close(errors, "x100 fast tagore_groups", fast_build.get("tagore_groups"), 128, 0.0)
    expect_close(errors, "x100 fast tagore_points", fast_build.get("tagore_points"), 48600, 0.5)
    expect_close(errors, "x100 fast tagore_fill", fast_build.get("tagore_fill_time"), 35.223, 0.001)

    cpu_recalls = read_search_recall_csv(Path(item.paths[2]))
    fast_recalls = read_search_recall_csv(Path(item.paths[5]))
    for lsearch, cpu_recall, fast_recall in [(100, 0.825, 0.828), (500, 0.868, 0.871), (1000, 0.890667, 0.896)]:
        expect_close(errors, f"x100 cpu L{lsearch}", cpu_recalls.get(lsearch), cpu_recall, 1e-6)
        expect_close(errors, f"x100 fast L{lsearch}", fast_recalls.get(lsearch), fast_recall, 1e-6)
    return errors


def validate_x100_adaptive_cuda(item: ArtifactItem) -> list[str]:
    errors: list[str] = []
    log_text = Path(item.paths[0]).read_text(errors="replace")
    if "[UNG config] group_graph_impl=adaptive_cuda" not in log_text:
        errors.append("adaptive_cuda config line missing")
    if "[UNG config] additional_edges_impl=cpu_vamana" not in log_text:
        errors.append("adaptive additional_edges cpu_vamana line missing")

    index_match = re.search(r"Index time:\s*([0-9.]+)ms", log_text)
    if not index_match:
        errors.append("adaptive Index time line missing")
    else:
        expect_close(errors, "adaptive index", float(index_match.group(1)), 5450.0, 0.5)
    group_matches = re.findall(r"- Finished in ([0-9.]+) ms", log_text)
    if len(group_matches) < 2:
        errors.append("adaptive finished-in lines missing")
    else:
        expect_close(errors, "adaptive group", float(group_matches[1]), 2773.35, 0.01)

    tagore_match = re.search(
        r"TagoreCuda groups:\s*(\d+).*fallback_cpu_groups:\s*(\d+).*fallback_cpu_points:\s*(\d+).*fallback_wall:\s*([0-9.]+).*fill:\s*([0-9.]+)",
        log_text,
    )
    if not tagore_match:
        errors.append("adaptive TagoreCuda summary line missing")
    else:
        expect_equal(errors, "adaptive tagore_groups", int(tagore_match.group(1)), 128)
        expect_equal(errors, "adaptive fallback_cpu_groups", int(tagore_match.group(2)), 5538)
        expect_equal(errors, "adaptive fallback_cpu_points", int(tagore_match.group(3)), 553800)
        expect_close(errors, "adaptive fallback_wall", float(tagore_match.group(4)), 2743.93, 0.01)
        expect_close(errors, "adaptive tagore_fill", float(tagore_match.group(5)), 28.277, 0.001)

    recalls = read_search_recall_csv(Path(item.paths[1]))
    expect_close(errors, "adaptive L100", recalls.get(100), 0.829, 1e-6)
    expect_close(errors, "adaptive L500", recalls.get(500), 0.870, 1e-6)
    expect_close(errors, "adaptive L1000", recalls.get(1000), 0.895, 1e-6)
    return errors


def parse_prof_line_values(text: str, tag: str) -> dict[str, float]:
    match = re.search(rf"\[PROF\] {re.escape(tag)} ([^\n]+)", text)
    if not match:
        return {}
    values: dict[str, float] = {}
    for key, value in re.findall(r"([A-Za-z0-9_]+)=([0-9.]+)", match.group(1)):
        try:
            values[key] = float(value)
        except ValueError:
            continue
    return values


def validate_x100_additional_output_boundary_smoke(item: ArtifactItem) -> list[str]:
    errors: list[str] = []
    expected = [
        {
            "label": "x10 release skipped",
            "path_index": 0,
            "release_skipped": True,
            "summary": {
                "total_ms": 298.470,
                "gen_ms": 195.400,
                "add_ms": 35.154,
                "num_points": 60240.0,
                "num_groups": 5666.0,
            },
            "gpu": {"kernel_ms": 8.172},
        },
        {
            "label": "x10 release forced",
            "path_index": 1,
            "release_skipped": False,
            "summary": {
                "total_ms": 370.218,
                "gen_ms": 274.950,
                "add_ms": 52.735,
                "num_points": 60240.0,
                "num_groups": 5666.0,
            },
            "gpu": {"kernel_ms": 7.342},
        },
        {
            "label": "x100 cpu_vamana additional",
            "path_index": 2,
            "release_skipped": True,
            "summary": {
                "total_ms": 2439.062,
                "gen_ms": 876.695,
                "add_ms": 1269.257,
                "num_points": 602400.0,
                "num_groups": 5666.0,
            },
            "gpu": {"kernel_ms": 655.969},
        },
        {
            "label": "x100 cpu_exact additional",
            "path_index": 3,
            "release_skipped": True,
            "summary": {
                "total_ms": 2568.729,
                "gen_ms": 1155.114,
                "add_ms": 207.893,
                "num_points": 602400.0,
                "num_groups": 5666.0,
            },
            "gpu": {"kernel_ms": 658.020},
        },
    ]
    for case in expected:
        label = str(case["label"])
        text = Path(item.paths[int(case["path_index"])]).read_text(errors="replace")
        if bool(case["release_skipped"]) and "cross_edges.gpu_release_after_cross skipped=1" not in text:
            errors.append(f"{label}: missing release skipped marker")
        summary = parse_prof_line_values(text, "cross_edges.summary")
        if not summary:
            errors.append(f"{label}: missing cross_edges.summary")
        else:
            for key, want in case["summary"].items():
                tol = 0.001 if key.endswith("_ms") or key in ("gen_ms", "add_ms", "total_ms") else 0.0
                expect_close(errors, f"{label} summary.{key}", summary.get(key), float(want), tol)
        gpu = parse_prof_line_values(text, "cross_edges.gpu_breakdown_sum")
        if not gpu:
            errors.append(f"{label}: missing gpu_breakdown_sum")
        else:
            for key, want in case["gpu"].items():
                expect_close(errors, f"{label} gpu.{key}", gpu.get(key), float(want), 0.001)

    x100_vamana = parse_prof_line_values(Path(item.paths[2]).read_text(errors="replace"), "cross_edges.summary")
    x100_exact = parse_prof_line_values(Path(item.paths[3]).read_text(errors="replace"), "cross_edges.summary")
    x100_gpu = parse_prof_line_values(Path(item.paths[2]).read_text(errors="replace"), "cross_edges.gpu_breakdown_sum")
    if x100_vamana and x100_gpu and not (x100_vamana.get("add_ms", 0.0) > x100_gpu.get("kernel_ms", 1e30)):
        errors.append("x100 cpu_vamana additional should exceed GPU kernel event")
    if x100_vamana and x100_exact and not (x100_exact.get("add_ms", 1e30) < x100_vamana.get("add_ms", 0.0)):
        errors.append("x100 cpu_exact additional should reduce add_ms vs cpu_vamana additional")
    if x100_vamana and x100_exact and not (x100_exact.get("total_ms", 0.0) > x100_vamana.get("total_ms", 1e30)):
        errors.append("x100 cpu_exact total should remain slower in this smoke test")
    return errors


def parse_stdout_cross_breakdown(text: str) -> dict[str, float]:
    match = re.search(
        r"\[cross_edges\] breakdown generate\(ms\)=([0-9.]+) "
        r"additional\(ms\)=([0-9.]+) add_offset\(ms\)=([0-9.]+) "
        r"merge_cross\(ms\)=([0-9.]+) merge_add\(ms\)=([0-9.]+).*"
        r"flat_id_active=([0-9]+)",
        text,
    )
    if not match:
        return {}
    return {
        "gen_ms": float(match.group(1)),
        "add_ms": float(match.group(2)),
        "add_offset_ms": float(match.group(3)),
        "merge_cross_ms": float(match.group(4)),
        "merge_add_ms": float(match.group(5)),
        "flat_id_active": float(match.group(6)),
    }


def parse_stdout_gpu_gemm(text: str) -> dict[str, float]:
    match = re.search(
        r"\[GPU GEMM\] H2D\(ms\)=([0-9.]+)\s+Kernel\(ms\)=([0-9.]+)\s+D2H\(ms\)=([0-9.]+)",
        text,
    )
    if not match:
        return {}
    return {
        "h2d_ms": float(match.group(1)),
        "kernel_ms": float(match.group(2)),
        "d2h_ms": float(match.group(3)),
    }


def validate_direct_global_flat_id_smoke(item: ArtifactItem) -> list[str]:
    errors: list[str] = []
    cases = [
        {
            "label": "local-id SearchQueue",
            "build": 0,
            "log": 1,
            "search": 2,
            "build_vals": {"index_time": 4693.59, "build_cross_edges_time": 1650.95},
            "cross_vals": {"add_offset_ms": 2.5, "merge_cross_ms": 17.2, "flat_id_active": 0.0},
            "gpu_vals": {"kernel_ms": 656.9, "d2h_ms": 11.1},
            "recall": 0.816,
            "marker": "[group_graph] intra_global_ids=0",
        },
        {
            "label": "direct-global SearchQueue",
            "build": 3,
            "log": 4,
            "search": 5,
            "build_vals": {"index_time": 6682.97, "build_cross_edges_time": 3530.31},
            "cross_vals": {"add_offset_ms": 0.0, "merge_cross_ms": 21.0, "flat_id_active": 0.0},
            "gpu_vals": {"kernel_ms": 658.5, "d2h_ms": 11.3},
            "recall": 0.816,
            "marker": "[group_graph] intra_global_ids=1",
        },
        {
            "label": "direct-global flat-id",
            "build": 6,
            "log": 7,
            "search": 8,
            "build_vals": {"index_time": 3986.41, "build_cross_edges_time": 707.826},
            "cross_vals": {"add_offset_ms": 0.0, "merge_cross_ms": 0.5, "flat_id_active": 1.0},
            "gpu_vals": {"kernel_ms": 255.1, "d2h_ms": 0.6},
            "recall": 0.816,
            "marker": "[group_graph] intra_global_ids=1",
        },
    ]
    for case in cases:
        label = str(case["label"])
        build = read_metric_csv(Path(item.paths[int(case["build"])]))
        log_text = Path(item.paths[int(case["log"])]).read_text(errors="replace")
        if str(case["marker"]) not in log_text:
            errors.append(f"{label}: missing marker {case['marker']}")
        for key, want in case["build_vals"].items():
            expect_close(errors, f"{label} {key}", build.get(key), float(want), 0.05)

        cross = parse_stdout_cross_breakdown(log_text)
        if not cross:
            errors.append(f"{label}: missing stdout cross breakdown")
        else:
            for key, want in case["cross_vals"].items():
                expect_close(errors, f"{label} cross.{key}", cross.get(key), float(want), 0.1)

        gpu = parse_stdout_gpu_gemm(log_text)
        if not gpu:
            errors.append(f"{label}: missing GPU GEMM stdout breakdown")
        else:
            for key, want in case["gpu_vals"].items():
                expect_close(errors, f"{label} gpu.{key}", gpu.get(key), float(want), 0.1)

        recalls = read_search_recall_csv(Path(item.paths[int(case["search"])]))
        expect_close(errors, f"{label} L100 recall", recalls.get(100), float(case["recall"]), 1e-6)

    local_log = Path(item.paths[1]).read_text(errors="replace")
    flat_log = Path(item.paths[7]).read_text(errors="replace")
    local_cross = parse_stdout_cross_breakdown(local_log)
    flat_cross = parse_stdout_cross_breakdown(flat_log)
    local_gpu = parse_stdout_gpu_gemm(local_log)
    flat_gpu = parse_stdout_gpu_gemm(flat_log)
    if local_cross and flat_cross and not (flat_cross["merge_cross_ms"] < local_cross["merge_cross_ms"] / 10.0):
        errors.append("flat-id merge_cross should be >10x lower than local SearchQueue path")
    if local_gpu and flat_gpu and not (flat_gpu["d2h_ms"] < local_gpu["d2h_ms"] / 10.0):
        errors.append("flat-id D2H should be >10x lower than local SearchQueue path")
    return errors


def validate_additional_direct_append_negative(item: ArtifactItem) -> list[str]:
    errors: list[str] = []
    for idx, label in ((0, "unlocked"), (2, "lock-guarded")):
        summary_path = Path(item.paths[idx])
        rows = list(csv.reader(summary_path.open()))
        if len(rows) != 1:
            errors.append(f"{label}: summary should contain only the header after failed run, got {len(rows)} rows")
        log_text = Path(item.paths[idx + 1]).read_text(errors="replace")
        if "[cross_edges] batched_search end" not in log_text:
            errors.append(f"{label}: missing batched_search end marker")
        if "[cross_edges] breakdown" in log_text:
            errors.append(f"{label}: unexpected successful cross-edge breakdown")
        if "Index time:" in log_text:
            errors.append(f"{label}: unexpected successful Index time")

    env_text = Path(item.paths[4]).read_text(errors="replace")
    if "UNG_ADDITIONAL_DIRECT_APPEND=1" not in env_text:
        errors.append("lock-guarded env missing UNG_ADDITIONAL_DIRECT_APPEND=1")

    src = Path(item.paths[5]).read_text(errors="replace")
    if "additional_direct_append disabled: supported only for cpu_exact additional_edges" not in src:
        errors.append("source missing cpu_exact-only guard message for additional_direct_append")
    if "_build_config.additional_edges_impl == UngAdditionalEdgesImpl::CpuExactScan" not in src:
        errors.append("source missing CpuExactScan guard for additional_direct_append")
    return errors


def validate_celeba_real_multilabel_sanity(item: ArtifactItem) -> list[str]:
    errors: list[str] = []
    build = read_metric_csv(Path(item.paths[0]))
    expect_close(errors, "celeba index_time", build.get("index_time"), 21771.9, 0.05)
    expect_close(errors, "celeba build_LNG_time", build.get("build_LNG_time"), 18086.7, 0.05)
    expect_close(errors, "celeba build_cross_edges_time", build.get("build_cross_edges_time"), 1567.03, 0.05)
    expect_close(errors, "celeba build_graph_time", build.get("build_graph_time"), 234.168, 0.001)

    gt_size = Path(item.paths[1]).stat().st_size
    expect_equal(errors, "celeba regenerated K10 GT bytes", gt_size, 80000)

    recalls = read_search_recall_csv(Path(item.paths[2]))
    expect_close(errors, "celeba L100 recall", recalls.get(100), 0.0714, 1e-6)
    expect_close(errors, "celeba L1000 recall", recalls.get(1000), 0.2692, 1e-6)
    expect_close(errors, "celeba L5000 recall", recalls.get(5000), 0.557, 1e-6)

    log_text = Path(item.paths[3]).read_text(errors="replace")
    if "Ground truth loaded from /home/graphdb/fv_runs/celeba_current_sanity_20260602_070322/regenerated_gt/celeba_gt_labels_containment.bin" not in log_text:
        errors.append("celeba search did not use regenerated K10 GT")
    if "Total time for finding all entry groups (Entry Cost): 25385.3 ms" not in log_text:
        errors.append("celeba entry-cost marker missing")
    return errors


def validate_artifact_values(item: ArtifactItem) -> list[str]:
    if item.claim == "x100 cross-edge strong-baseline fairness":
        return validate_x100_cross_fairness(item)
    if item.claim == "x400 repair/reverse-tail A/B":
        return validate_x400_reverse_tail(item)
    if item.claim == "x400 same-script CPU baseline":
        return validate_x400_same_script_cpu_baseline(item)
    if item.claim == "x400 compact-D2H ablation":
        return validate_x400_compact_d2h_ablation(item)
    if item.claim == "Amazon 1% x200 partial double-buffer routing":
        return validate_partial_double_buffer_x200(item)
    if item.claim == "source-centric checked negative ablation":
        return validate_source_checked_negative(item)
    if item.claim == "SIFT30 universal flat double-buffer cross-edge":
        return validate_sift30_universal_flat_db(item)
    if item.claim == "Amazon 1% x200 universal flat full-quality smoke":
        return validate_x200_universal_flat_full_quality(item)
    if item.claim == "Amazon 1% x100 universal flat full-quality boundary":
        return validate_x100_universal_flat_full_quality_boundary(item)
    if item.claim == "mixed exact/GNN router A/B":
        return validate_mixed_router_x200(item)
    if item.claim == "packed exact-anchor L5000 sweep":
        return validate_packed_exact_sweep(item)
    if item.claim == "x200 packed exact-anchor direct-H2D overhead A/B":
        return validate_x200_packed_exact_direct_h2d(item)
    if item.claim == "x200 exact-anchor warp-kernel/fill-thread negative ablation":
        return validate_x200_exact_warp_fill_negative(item)
    if item.claim == "Graph output-boundary reserve A/B":
        return validate_graph_output_boundary_reserve(item)
    if item.claim == "Graph output-boundary NeighborList small-buffer A/B":
        return validate_graph_output_boundary_neighborlist(item)
    if item.claim == "Graph output-boundary direct-global/flat-id smoke":
        return validate_direct_global_flat_id_smoke(item)
    if item.claim == "Graph output-boundary additional direct-append negative ablation":
        return validate_additional_direct_append_negative(item)
    if item.claim == "10%x40 bounded-complete fallback A/B":
        return validate_10x40_bounded_complete(item)
    if item.claim == "10%x40 additional Lsearch sweep":
        return validate_10x40_lsearch_sweep(item)
    if item.claim == "Amazon 1% x100 repeat=3 group-graph recall A/B":
        return validate_x100_repeat_group_ab(item)
    if item.claim == "Amazon 1% x100 adaptive_cuda conservative full-quality A/B":
        return validate_x100_adaptive_cuda(item)
    if item.claim == "x100 additional_edges/output-boundary smoke":
        return validate_x100_additional_output_boundary_smoke(item)
    if item.claim == "CelebA or real multi-label end-to-end recall":
        return validate_celeba_real_multilabel_sanity(item)
    return []


def audit_row(item: ArtifactItem) -> dict[str, str]:
    found = exists_all(item.paths)
    value_errors: list[str] = []
    if found:
        evidence = "present"
        value_errors = validate_artifact_values(item)
        if value_errors:
            evidence = "present_but_bad_values"
    elif item.paths:
        evidence = "missing_path"
    else:
        evidence = "no_path"
    if item.required and not found:
        verdict = "BLOCK_MAIN_CLAIM"
    elif item.required and value_errors:
        verdict = "BROKEN_VALUE"
    elif item.status == "done" and found:
        verdict = "OK"
    elif item.status == "done":
        verdict = "BROKEN_ARTIFACT"
    elif item.status == "pending":
        verdict = "PENDING_EXPERIMENT"
    else:
        verdict = "MISSING_EXPERIMENT"
    return {
        "claim": item.claim,
        "declared_status": item.status,
        "required_for_current_claim": "yes" if item.required else "no",
        "evidence": evidence,
        "verdict": verdict,
        "paths": ";".join(item.paths),
        "note": item.note if not value_errors else item.note + " VALUE_ERRORS: " + "; ".join(value_errors),
    }


def make_markdown(rows: list[dict[str, str]]) -> str:
    fields = ["claim", "declared_status", "evidence", "verdict", "note"]
    lines = ["| " + " | ".join(fields) + " |", "|---" * len(fields) + "|"]
    for row in rows:
        lines.append("| " + " | ".join(row[field].replace("|", "\\|") for field in fields) + " |")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, help="Optional CSV output path.")
    parser.add_argument("--md", type=Path, help="Optional Markdown output path.")
    parser.add_argument("--x400-root", type=Path, help="Optional output root from run_x400_reverse_tail_ab.sh.")
    parser.add_argument("--router-root", type=Path, help="Optional output root from run_group_graph_router_ab.sh.")
    parser.add_argument(
        "--fail-required",
        action="store_true",
        help="Exit nonzero if a required current-claim artifact is missing.",
    )
    parser.add_argument(
        "--fail-submission",
        action="store_true",
        help="Exit nonzero if any artifact is pending or missing; use only for final submission gating.",
    )
    args = parser.parse_args()

    rows = [audit_row(item) for item in artifacts_with_overrides(args.x400_root, args.router_root)]
    if args.csv:
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        with args.csv.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
    md = make_markdown(rows)
    if args.md:
        args.md.parent.mkdir(parents=True, exist_ok=True)
        args.md.write_text(md)
    print(md, end="")

    if args.fail_required and any(row["verdict"] in ("BLOCK_MAIN_CLAIM", "BROKEN_VALUE") for row in rows):
        raise SystemExit(2)
    if args.fail_submission and any(row["verdict"] != "OK" for row in rows):
        raise SystemExit(3)


if __name__ == "__main__":
    main()
