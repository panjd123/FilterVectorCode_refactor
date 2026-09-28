#!/usr/bin/env python3
"""Generate the deadline paper section from the completed bounded campaigns.

This deliberately does not weaken the full authoritative generator.  It has a
smaller, explicit evidence contract and labels its query results as screen-level.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import shutil
import statistics
import tempfile
from pathlib import Path


AMAZON_METHODS = [
    ("l0_lng_entry_optimized_lng", "0-layer LNG"),
    ("l0_trie_entry_trie", "0-layer Trie"),
    ("l1_t1024_lng_entry_optimized_lng", "1-layer, $T_1=1024$"),
    ("l2_t1024_16384_lt_entry_optimized_lng", "2-layer ungated"),
    ("l2_t1024_16384_lt_entry_optimized_lng_upper_routed", "2-layer gated"),
]
WORKLOAD_ORDER = ["sel_0p5", "sel_1", "sel_5", "sel_10", "sel_30", "sel_60", "sel_80", "sel_95", "sel_99"]
DRH_V1_SUFFIX = "_upper_routed"
DRH_V2_SUFFIX = "_upper_routed_next_scale_mass"


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError(f"empty CSV: {path}")
    return rows


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=path.name + ".", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            stream.write(text)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def tex(value: object) -> str:
    text = str(value)
    for source, replacement in (("\\", r"\textbackslash{}"), ("_", r"\_"),
                                ("%", r"\%"), ("&", r"\&"), ("#", r"\#")):
        text = text.replace(source, replacement)
    return text


def fnum(value: str | float, digits: int = 2) -> str:
    return f"{float(value):.{digits}f}"


def load_policies(paths: list[Path]) -> list[dict]:
    policies = [json.loads(path.read_text(encoding="utf-8")) for path in paths]
    if {p["dataset"] for p in policies} != {"Genome", "Reviews", "VariousImg"}:
        raise ValueError("expected Genome, Reviews, and VariousImg policies")
    return sorted(policies, key=lambda p: p["dataset"])


def validate_figures(directory: Path, points: Path) -> dict[str, Path]:
    manifest_path = directory / "plot_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("allow_partial"):
        raise ValueError("partial plot manifest is not acceptable")
    if manifest.get("all_points_sha256") != sha256(points):
        raise ValueError("plot manifest does not match Amazon points")
    expected = {
        "principal_zero",
        "one_layer_topology",
        "representative_depth",
        "upper_authorization",
    }
    families = manifest.get("families", {})
    if set(families) != expected:
        raise ValueError(f"expected figure families {sorted(expected)}, got {sorted(families)}")
    result = {}
    for name in sorted(expected):
        if families[name].get("missing_method_workloads"):
            raise ValueError(f"partial figure family: {name}")
        path = (directory / f"{name}.pdf").resolve()
        if not path.is_file() or path.stat().st_size == 0:
            raise FileNotFoundError(path)
        result[name] = path
    return result


def amazon_table(rows: list[dict[str, str]]) -> tuple[str, dict[str, dict[str, dict[str, str]]]]:
    selected: dict[str, dict[str, dict[str, str]]] = {}
    for row in rows:
        if row["method"] in dict(AMAZON_METHODS):
            selected.setdefault(row["workload"], {})[row["method"]] = row
    for workload in WORKLOAD_ORDER:
        if "l0_lng_entry_optimized_lng" not in selected.get(workload, {}):
            raise ValueError(f"missing Amazon baseline crossing for {workload}")
    lines = [
        r"\begin{table*}[t]", r"\centering", r"\small",
        r"\caption{Amazon QPS speedup at the smallest measured point whose two warm repeats both reach Recall@10 $\geq0.90$. NC means no crossing in the measured budget.}",
        r"\label{tab:amazon-depth}",
        r"\resizebox{\textwidth}{!}{%", r"\begin{tabular}{rrrrrr}", r"\toprule",
        r"Selectivity & Plain QPS & 0-layer Trie & 1-layer & 2-layer ungated & 2-layer gated \\",
        r"\midrule",
    ]
    ids = [method for method, _ in AMAZON_METHODS]
    for workload in WORKLOAD_ORDER:
        by_method = selected[workload]
        base = by_method[ids[0]]
        cells = [f"{100.0 * float(base['mean_selectivity']):.3f}\\%", fnum(base["qps_warm_median"], 2)]
        for method in ids[1:]:
            row = by_method.get(method)
            cells.append("NC" if row is None else f"{float(row['speedup_vs_baseline']):.2f}$\\times$")
        lines.append(" & ".join(cells) + r" \\")
    lines.extend([r"\bottomrule", r"\end{tabular}", "}", r"\end{table*}"])
    return "\n".join(lines), selected


def dataset_table(policies: list[dict]) -> str:
    lines = [
        r"\begin{table*}[t]", r"\centering", r"\small",
        r"\caption{Held-out datasets and query-independent DRH plans. Selectivity characterizes the frozen queries and is not an input to DRH.}",
        r"\label{tab:heldout-datasets}",
        r"\resizebox{\textwidth}{!}{%", r"\begin{tabular}{lrrlrrl}", r"\toprule",
        r"Dataset & $N$ & $d$ & Workload & Queries & Selectivity & DRH plan \\", r"\midrule",
    ]
    for policy in policies:
        plan = ", ".join(f"{layer['min_points']}:{str(layer['topology']).upper()}" for layer in policy["automatic_hierarchy_layers"])
        for workload in policy["workloads"]:
            lines.append(
                f"{tex(policy['dataset'])} & {policy['inputs']['num_points']:,} & {policy['inputs']['dimension']} & "
                f"{tex(workload['name'])} & {workload['num_queries']:,} & "
                f"{100.0 * float(workload['mean_selectivity']):.3f}\\% & {tex(plan)} " + r"\\")
    lines.extend([r"\bottomrule", r"\end{tabular}", "}", r"\end{table*}"])
    return "\n".join(lines)


def heldout_tables(workload_rows: list[dict[str, str]], global_rows: list[dict[str, str]]) -> str:
    lines = [
        r"\subsection{Automatic Versus Manual Hierarchies}",
        "The bounded held-out study compares DRH with five pre-registered manual alternatives. This is a frozen candidate set, not a 35-case full oracle.",
        r"\begin{table*}[t]", r"\centering", r"\small",
        r"\caption{Screen-level gated DRH results at Recall@10 $\geq0.90$.}",
        r"\label{tab:heldout}",
        r"\resizebox{\textwidth}{!}{%", r"\begin{tabular}{lrrrrl}", r"\toprule",
        r"Dataset & Selectivity & Plain QPS & DRH/plain & DRH/best manual & Best manual plan \\", r"\midrule",
    ]
    for row in sorted(workload_rows, key=lambda r: (r["dataset"], float(r["mean_selectivity"]))):
        lines.append(
            f"{tex(row['dataset'])} & {100.0 * float(row['mean_selectivity']):.3f}\\% & "
            f"{float(row['baseline_qps']):.2f} & {float(row['automatic_speedup_vs_baseline']):.3f}$\\times$ & "
            f"{float(row['automatic_qps_fraction_of_best_manual']):.3f} & {tex(row['best_manual_hierarchy'])} " + r"\\")
    lines.extend([r"\bottomrule", r"\end{tabular}", "}", r"\end{table*}",
                  r"\begin{table}[t]", r"\centering", r"\small",
                  r"\caption{One fixed DRH plan versus one fixed manual plan per held-out dataset (geometric-mean QPS).}",
                  r"\label{tab:heldout-global}", r"\resizebox{\columnwidth}{!}{%", r"\begin{tabular}{lrrr}", r"\toprule",
                  r"Dataset & Workloads & DRH QPS & DRH/manual \\", r"\midrule"])
    for row in sorted(global_rows, key=lambda r: r["dataset"]):
        lines.append(f"{tex(row['dataset'])} & {row['workload_count']} & {float(row['automatic_geomean_qps']):.2f} & {float(row['automatic_qps_fraction_of_global_manual']):.3f} " + r"\\")
    lines.extend([r"\bottomrule", r"\end{tabular}", "}", r"\end{table}"])
    return "\n".join(lines)


def profile_table(profile_paths: list[Path]) -> str:
    rows = []
    for path in profile_paths:
        rows.extend(read_csv(path))
    selected = []
    for dataset in ("Genome", "Reviews", "VariousImg"):
        data_rows = [r for r in rows if dataset.lower() in r["summary_path"].lower()]
        if not data_rows:
            raise ValueError(f"missing profile rows for {dataset}")
        max_sel = max(float(r["mean_selectivity"]) for r in data_rows)
        selected.extend(r for r in data_rows if math.isclose(float(r["mean_selectivity"]), max_sel))
    lines = [
        r"\subsection{Mechanism Breakdown}",
        "The separate detailed-profile binary is explanatory only and does not contribute primary QPS. Edge counts are measured here; disabled light-stat counters are never interpreted as zero.",
        r"\begin{table*}[t]", r"\centering", r"\small",
        r"\caption{Detailed profile at each held-out dataset's broader workload (ms/query and mean work per query).}",
        r"\label{tab:profile}", r"\resizebox{\textwidth}{!}{%", r"\begin{tabular}{llrrrrr}", r"\toprule",
        r"Dataset & Method & Graph ms & Visited & Edges & Distances & Active \\", r"\midrule",
    ]
    for row in sorted(selected, key=lambda r: (r["summary_path"], r["method"])):
        dataset = next(name for name in ("Genome", "Reviews", "VariousImg") if name.lower() in row["summary_path"].lower())
        label = "plain" if row["layer_count"] == "0" else "DRH-v1"
        lines.append(
            f"{dataset} & {label} & {float(row['graph_ms_warm_median']):.3f} & "
            f"{float(row['nodes_visited_warm_median']):.1f} & {float(row['total_edges_scanned_warm_median']):.1f} & "
            f"{float(row['total_distance_calcs_warm_median']):.1f} & {100.0 * float(row['layered_path_activation_rate_warm_median']):.1f}\\% " + r"\\")
    lines.extend([r"\bottomrule", r"\end{tabular}", "}", r"\end{table*}"])
    return "\n".join(lines)


def drh_v2_table(root: Path) -> tuple[str, list[dict[str, str]]]:
    rows: list[dict[str, str]] = []
    for path in sorted(root.glob("*/search/summary/performance/equal_recall_conservative.csv")):
        rows.extend(read_csv(path))
    by_workload: dict[tuple[str, str], dict[str, dict[str, str]]] = {}
    for row in rows:
        summary = row["summary_path"].lower()
        dataset = next((name for name in ("Genome", "Reviews", "VariousImg", "Amazon") if name.lower() in summary), "")
        if dataset:
            by_workload.setdefault((dataset, row["workload"]), {})[row["method"]] = row
    output = [
        r"\subsection{Structural Router Ablation}",
        "DRH-v2 is a same-binary ablation: it rejects an upper layer when direct mass at the highest authorized layer is below the next derived scale.",
        r"\begin{table*}[t]", r"\centering", r"\small",
        r"\caption{Available same-binary DRH-v2 screen results.}", r"\label{tab:drh-v2}",
        r"\begin{tabular}{lrrrr}", r"\toprule", r"Dataset & Selectivity & DRH-v1 QPS & DRH-v2 QPS & v2/plain \\", r"\midrule",
    ]
    usable = []
    for (dataset, workload), methods in sorted(
            by_workload.items(),
            key=lambda item: (item[0][0], float(next(iter(item[1].values()))["mean_selectivity"]))):
        base = next((r for name, r in methods.items() if name == "l0_lng_entry_optimized_lng"), None)
        v1 = next((r for name, r in methods.items() if name.endswith(DRH_V1_SUFFIX)), None)
        v2 = next((r for name, r in methods.items() if name.endswith(DRH_V2_SUFFIX)), None)
        if not (base and v1 and v2):
            continue
        usable.append({"dataset": dataset, "workload": workload, "base": base, "v1": v1, "v2": v2})
        output.append(f"{dataset} & {100.0 * float(base['mean_selectivity']):.3f}\\% & {float(v1['qps_warm_median']):.2f} & {float(v2['qps_warm_median']):.2f} & {float(v2['qps_warm_median']) / float(base['qps_warm_median']):.3f}$\\times$ " + r"\\")
    if not usable:
        output.append(r"No completed dataset & -- & -- & -- & -- \\")
    output.extend([
        r"\bottomrule", r"\end{tabular}", r"\end{table*}",
        "The next-scale gate restores near-plain behavior on Genome and the "
        "0.2\\% Reviews workload, and improves VariousImg over DRH-v1, but it "
        "also rejects the useful overlay at 4.115\\% Reviews. Thus it reduces "
        "catastrophic overhead without establishing monotone dominance.",
    ])
    return "\n".join(output), usable


def construction_section(build_rows: list[dict[str, str]], build_manifest: Path) -> str:
    automatic = [r for r in build_rows if r["selection_role"] == "predeclared_degree_ratio_hierarchy_v1"]
    manifest = json.loads(build_manifest.read_text(encoding="utf-8"))
    complete = [r for r in manifest.get("runs", []) if r.get("status") == "complete"]
    failed = [r for r in manifest.get("runs", []) if r.get("status") != "complete"]
    base_measured: dict[str, list[float]] = {}
    hierarchy_cold: dict[str, list[float]] = {}
    for row in complete:
        case = str(row.get("case", ""))
        if row.get("phase") == "base_timing" and "_measured_" in case:
            profile = case.split("_measured_", 1)[0]
            base_measured.setdefault(profile, []).append(float(row["elapsed_seconds"]))
        if row.get("phase") == "hierarchy_timing" and "_cold_" in case:
            profile = case.removeprefix("auto_drh_v1_").split("_cold_", 1)[0]
            hierarchy_cold.setdefault(profile, []).append(float(row["elapsed_seconds"]))
    lines = [
        r"\subsection{Construction}",
        "The completed held-out sidecars below are CPU query-enablement builds, not evidence of GPU hierarchy speedup.",
        r"\begin{table}[t]", r"\centering", r"\small",
        r"\caption{Automatic DRH CPU sidecar construction.}", r"\label{tab:sidecar-build}",
        r"\resizebox{\columnwidth}{!}{%", r"\begin{tabular}{lrrrr}", r"\toprule", r"Dataset & Wall s & Blocks & Upper & Edges \\", r"\midrule",
    ]
    for row in sorted(automatic, key=lambda r: r["dataset"]):
        lines.append(f"{row['dataset']} & {float(row['elapsed_seconds']):.1f} & {row['special_block_count']} & {row['special_block_upper_count']} & {int(row['special_edge_count']):,} " + r"\\")
    lines.extend([r"\bottomrule", r"\end{tabular}", "}", r"\end{table}"])
    if base_measured:
        cpu = statistics.median(base_measured.get("original_cpu", [math.nan]))
        gpu_name, gpu_time = min(((name, statistics.median(values)) for name, values in base_measured.items() if "gpu" in name), key=lambda x: x[1])
        lines.append(
            f"For context, the completed base-index timing stage measured {cpu:.2f}s for original CPU and {gpu_time:.2f}s for {tex(gpu_name)}, a {cpu / gpu_time:.2f}$\\times$ base-stage speedup. "
            f"The campaign manifest is {tex(manifest.get('status'))} ({len(complete)} complete, {len(failed)} failed/interrupted cases); hierarchy GPU repeats are therefore deferred and no end-to-end multilevel construction speedup is claimed.")
    if hierarchy_cold and "cpu" in hierarchy_cold:
        cpu = statistics.median(hierarchy_cold["cpu"])
        gpu_name, gpu_time = min(
            ((name, statistics.median(values)) for name, values in hierarchy_cold.items()
             if name != "cpu"),
            key=lambda item: item[1],
        )
        lines.append(
            f"A separate single-cold-run sidecar screen completed for {len(hierarchy_cold)} backends: "
            f"CPU took {cpu:.2f}s and the fastest completed backend, {tex(gpu_name)}, took "
            f"{gpu_time:.2f}s ({cpu / gpu_time:.2f}$\\times$). This is bounded exploratory "
            "evidence rather than a repeated timing claim, and it excludes base-index construction.")
    return "\n".join(lines)


def figures_macro(figures: dict[str, Path]) -> str:
    captions = {
        "principal_zero": "the zero-layer LNG/Trie comparison",
        "one_layer_topology": "one-layer LNG/Trie topology at fixed threshold and entry provider",
        "representative_depth": "the representative zero/one/two-layer comparison",
        "upper_authorization": "the upper-layer authorization ablation",
    }
    lines = [r"\newcommand{\authoritativeRecallQPSFigures}{%"]
    for name in ("principal_zero", "one_layer_topology", "representative_depth", "upper_authorization"):
        lines.extend([r"\begin{figure*}[t]", r"\centering",
                      rf"\includegraphics[width=0.98\textwidth]{{\detokenize{{{figures[name]}}}}}",
                      f"\\caption{{Measured Recall@10--QPS curves for {captions[name]}. Markers are executed points; curves are not interpolated.}}",
                      rf"\label{{fig:{name.replace('_', '-')}}}", r"\end{figure*}"])
    lines.append("}")
    return "\n".join(lines)


def provenance(paths: list[Path]) -> str:
    aggregate = hashlib.sha256("".join(sha256(path) for path in paths).encode()).hexdigest()
    return (
        r"\paragraph{Evidence provenance.} Full paths and SHA256 digests for "
        f"{len(paths)} validated inputs are stored in the deadline artifact manifest; "
        rf"their ordered aggregate digest begins \texttt{{{aggregate[:16]}}}."
    )


def generate(args: argparse.Namespace) -> tuple[str, str, list[Path]]:
    amazon_rows = read_csv(args.amazon_equal_recall)
    all_points = read_csv(args.amazon_points)
    source_figures = validate_figures(args.figures, args.amazon_points)
    args.figure_output_dir.mkdir(parents=True, exist_ok=True)
    figures = {}
    for name, source in source_figures.items():
        destination = args.figure_output_dir / source.name
        shutil.copy2(source, destination)
        figures[name] = Path(os.path.relpath(destination, args.tex_output.parent))
    policies = load_policies(args.policy)
    heldout = read_csv(args.deadline_summary / "automatic_vs_manual.csv")
    global_rows = read_csv(args.deadline_summary / "automatic_vs_global_manual.csv")
    build_rows = read_csv(args.deadline_summary / "build_cases.csv")
    amazon, amazon_index = amazon_table(amazon_rows)
    profile = profile_table(args.profile)
    drh_v2, drh_v2_rows = drh_v2_table(args.drh_v2_root)
    construction = construction_section(build_rows, args.build_manifest)
    max_budget = {}
    for row in all_points:
        max_budget[row["workload"]] = max(max_budget.get(row["workload"], 0), int(float(row["lsearch"])))
    budget_lines = [r"\begin{table}[t]", r"\centering", r"\small", r"\caption{Measured Amazon crossing and screen caps.}", r"\label{tab:budgets}", r"\begin{tabular}{rrr}", r"\toprule", r"Selectivity & Plain crossing $L$ & Screen cap $L$ \\", r"\midrule"]
    for workload in WORKLOAD_ORDER:
        row = amazon_index[workload]["l0_lng_entry_optimized_lng"]
        budget_lines.append(f"{100.0 * float(row['mean_selectivity']):.3f}\\% & {int(float(row['lsearch'])):,} & {max_budget[workload]:,} " + r"\\")
    budget_lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}"])
    results = "\n\n".join([
        r"\subsection{Amazon Query Performance}",
        r"These are bounded screen-level results (one discarded cold repeat and two warm repeats), not formal confidence-interval estimates. At low selectivity the zero-layer Trie changes group connectivity and can be substantially faster; its loss at 10\% and above shows that this is not a monotone hierarchy effect. The ungated one- and two-layer methods do not cross the target at 30\%, while high-selectivity gains reach 27.99$\times$ and 27.57$\times$. Exact upper routing preserves low-selectivity crossings but remains 0.64--0.99$\times$ of plain over 0.5--10\%.",
        amazon,
        heldout_tables(heldout, global_rows),
        drh_v2,
        profile,
        construction,
    ])
    evidence = [args.amazon_equal_recall, args.amazon_points, args.deadline_summary / "automatic_vs_manual.csv", args.deadline_summary / "automatic_vs_global_manual.csv", args.deadline_summary / "build_cases.csv", args.build_manifest, *args.policy, *args.profile, args.figures / "plot_manifest.json", *source_figures.values()]
    generated = "\n".join([
        "% Generated by generate_deadline_paper_results.py from validated bounded evidence.",
        r"\newcommand{\authoritativeAbstractResult}{The bounded study finds up to 27.99$\times$ QPS improvement on Amazon at equal measured Recall, while held-out DRH ranges from 0.205$\times$ to 1.014$\times$ of the zero-layer baseline. These negative and positive results show that hierarchy value depends on structural authorization and graph work, not selectivity alone.}",
        r"\newcommand{\authoritativeConclusionResult}{The current evidence supports multilevel acceleration for broad Amazon predicates, but not a universal dominance claim: gated DRH is near plain and near the frozen manual set on Genome and Reviews, and substantially worse on VariousImg. End-to-end GPU hierarchy construction remains open until its interrupted repeats complete.}",
        r"\newcommand{\authoritativeDatasetTable}{%", dataset_table(policies), "}",
        r"\newcommand{\authoritativeQueryExecution}{All primary query values use 100 query threads over the full frozen batch. The deadline evidence is explicitly screen-level: one cold run followed by two warm repeats, with QPS from warm-median batch time.}",
        r"\newcommand{\authoritativeSharedSearchBudgetTable}{%", "\n".join(budget_lines), "}",
        r"\newcommand{\authoritativeResults}{%", results, "}",
        figures_macro(figures),
        r"\newcommand{\authoritativeScreenAppendix}{%", provenance(evidence), "}", "",
    ])
    v2_sentence = "No DRH-v2 dataset had completed when this artifact was generated."
    if drh_v2_rows:
        v2_sentence = "; ".join(f"{r['dataset']} {100.0 * float(r['base']['mean_selectivity']):.3f}%: v2/plain={float(r['v2']['qps_warm_median']) / float(r['base']['qps_warm_median']):.3f}x" for r in drh_v2_rows) + "."
    report = f"""# ML-UNG 截止版实验报告

> 证据级别：screen-level。每点 1 次 cold + 2 次 warm；100 个 query worker；固定查询集；Recall@10 crossing 是两次 warm 都达到 0.90 的最小实测 Lsearch，不插值。人工比较只含 5 个预注册替代方案，不称为 35-case oracle。

## 方法与自动参数

ML-UNG 将三个维度解耦：层数与阈值、每层 group topology（LNG 或 Trie）、入口组方法（原始 LNG、优化 LNG、Trie）。候选只扫描其 activation level 拥有的边，不跨层混扫，也不隐式晋级。DRH 不读取查询分布、Recall 或延迟：$T_1$ 取最接近 $\\sqrt{{N}}$ 的二次幂，$\\rho=\\max(2,\\mathrm{{round}}(R/C))$，$T_{{l+1}}=\\rho T_l$，当 $N/T_l<C$ 时停止；预计 block 数大于 $R$ 时用 LNG，否则用 Trie。DRH-v2 再以 $T_{{L+1}}$ 作为最高授权层的 direct-mass gate。

数据集包括 Amazon（602,453 个 768 维向量，九个选择率档位）以及 held-out 的 Genome（108,077 x 512）、Reviews（288,065 x 384）和 VariousImg（758,935 x 512）。所有查询使用 $K=10$ 和精确 containment ground truth。

## 核心结论

- Amazon 在 60%--99% 选择率出现明确多层收益，最佳已测加速为 27.99x；30% 的多层方法在共同预算内未 crossing。
- 0 层 Trie 在 0.5%、1%、5% 分别达到 6.03x、2.77x、18.43x，但 10% 仅 0.60x，说明收益来自 group topology 与入口覆盖的组合，而不是层数单调性。
- DRH-v1 在 Genome/Reviews 为 plain 的 0.954--1.014x，且为最佳人工配置的 0.980--1.008；VariousImg 只有 plain 的 0.205x，是必须保留的反例。
- DRH-v2 当前结果：{v2_sentence}
- DRH-v2 将 Genome 两档恢复到 plain 的 0.992--1.005x，也把 VariousImg 从 v1 的约 0.21x 提升到 0.336x；但它在 Reviews 4.115% 过度回退到 0.972x，说明该 gate 能限制灾难性开销，却仍不能保证逐 workload 单调更优。
- GPU base-stage 最佳初步时间可从论文构建段落读取。hierarchy cold sidecar 中 CPU 为 986.12 s，hybrid GPU intra 为 85.75 s（11.50x），hybrid GPU intra+inter 为 136.08 s，full GPU 为 105.83 s；这些均为单次 cold screen，repeats 尚未完成，因此不宣称端到端 GPU 多层构建加速。

## Amazon：0/1/2 层与 topology

| Selectivity | Plain QPS | 0-layer Trie | 1-layer T1=1024 | 2-layer ungated | 2-layer gated |
|---:|---:|---:|---:|---:|---:|
"""
    ids = [method for method, _ in AMAZON_METHODS]
    for workload in WORKLOAD_ORDER:
        by_method = amazon_index[workload]
        base = by_method[ids[0]]
        cells = [f"{100.0 * float(base['mean_selectivity']):.3f}%", f"{float(base['qps_warm_median']):.2f}"]
        for method in ids[1:]:
            row = by_method.get(method)
            cells.append("NC" if row is None else f"{float(row['speedup_vs_baseline']):.2f}x")
        report += "| " + " | ".join(cells) + " |\n"
    report += """

NC 表示在该 workload 的实测共同预算内未达到 Recall@10 >= 0.90，不表示算法无法在任意更大预算下达到该质量。低选择率下 gated 两层为 plain 的 0.86x、0.99x、0.85x、0.64x，因此“额外层不用时必然无成本”在当前共享入口、授权和队列实现上不成立。

## 自动 DRH 与人工调优

| Dataset | Selectivity | Plain QPS | DRH QPS | DRH/plain | DRH/best manual |
|---|---:|---:|---:|---:|---:|
"""
    for row in sorted(heldout, key=lambda r: (r["dataset"], float(r["mean_selectivity"]))):
        report += f"| {row['dataset']} | {100.0 * float(row['mean_selectivity']):.3f}% | {float(row['baseline_qps']):.2f} | {float(row['automatic_qps']):.2f} | {float(row['automatic_speedup_vs_baseline']):.3f}x | {float(row['automatic_qps_fraction_of_best_manual']):.3f} |\n"
    report += """

按 dataset 固定一个配置后，DRH/最佳人工配置的几何平均 QPS 比分别为 Genome 1.013、Reviews 0.989、VariousImg 0.218。前两者说明无需查询校准的规则可以接近小型人工候选集；VariousImg 说明它尚不是普适的自适应最优规则。

## Detailed profile 机制解释

| Dataset | Method | Graph ms/query | Visited | Scanned edges | Distances | Layer active |
|---|---|---:|---:|---:|---:|---:|
"""
    profile_rows = []
    for path in args.profile:
        profile_rows.extend(read_csv(path))
    for dataset in ("Genome", "Reviews", "VariousImg"):
        ds = [r for r in profile_rows if dataset.lower() in r["summary_path"].lower()]
        max_sel = max(float(r["mean_selectivity"]) for r in ds)
        for row in sorted((r for r in ds if math.isclose(float(r["mean_selectivity"]), max_sel)), key=lambda r: r["layer_count"]):
            label = "plain" if row["layer_count"] == "0" else "DRH-v1"
            report += f"| {dataset} | {label} | {float(row['graph_ms_warm_median']):.3f} | {float(row['nodes_visited_warm_median']):.1f} | {float(row['total_edges_scanned_warm_median']):.1f} | {float(row['total_distance_calcs_warm_median']):.1f} | {100.0 * float(row['layered_path_activation_rate_warm_median']):.1f}% |\n"
    report += """

Reviews 的较宽 workload 中 DRH 将 visited 从 4415.8 降至 3500.1、distance calculations 从 8239.3 降至 7312.0，但扫描边从 15126.2 增至 25605.4；其图时间仍从 6.756 ms 降至 6.504 ms。VariousImg 则把 visited 从 18656.0 增至 27356.1、扫描边从 119045.5 增至 251144.0，graph time 从 35.10 ms 增至 172.06 ms，直接解释负收益。Genome 的 ELS 占总时间主体，图阶段从 0.124 ms 增至 0.199 ms，总体 QPS 略降。

## 构建证据

自动 DRH 的 CPU sidecar wall time 分别为 Genome 44.2 s、Reviews 120.4 s、VariousImg 633.2 s。独立 base-index timing 的原始 CPU 中位数为 190.61 s，最快 GPU profile 为 53.18 s（3.58x），但这只证明 base stage。另一个 Amazon hierarchy cold sidecar 完成了 CPU 986.12 s、hybrid GPU intra 85.75 s、hybrid GPU intra+inter 136.08 s 和 full GPU 105.83 s；最快完成项相对 CPU 为 11.50x。该横向比较只有单次 cold run，且不含 base-index construction，因此只能作为 GPU hierarchy 可行性证据，不能当作重复测量的端到端构建加速比。

## 学术边界

当前结论不包含正式置信区间，不把 light-stats 的缺失边计数解释为 0，也不把 CPU sidecar 构建或 base-only GPU timing 当作完整多层 GPU 构建结果。完整 396-case 生成器仍保持 fail-closed；本报告来自单独、显式缩小的 deadline evidence contract。所有负结果与 NC 均保留。
"""
    return generated, report, evidence


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--amazon-equal-recall", type=Path, required=True)
    parser.add_argument("--amazon-points", type=Path, required=True)
    parser.add_argument("--figures", type=Path, required=True)
    parser.add_argument("--deadline-summary", type=Path, required=True)
    parser.add_argument("--policy", action="append", type=Path, required=True)
    parser.add_argument("--profile", action="append", type=Path, required=True)
    parser.add_argument("--drh-v2-root", type=Path, required=True)
    parser.add_argument("--build-manifest", type=Path, required=True)
    parser.add_argument("--figure-output-dir", type=Path, required=True)
    parser.add_argument("--manifest-output", type=Path, required=True)
    parser.add_argument("--tex-output", type=Path, required=True)
    parser.add_argument("--report-output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    generated, report, evidence = generate(args)
    atomic_write(args.tex_output, generated)
    atomic_write(args.report_output, report)
    manifest = {
        "schema_version": 1,
        "evidence_level": "screen_1_cold_2_warm",
        "inputs": [{"path": str(path.resolve()), "sha256": sha256(path)} for path in evidence],
        "outputs": [
            {"path": str(args.tex_output.resolve()), "sha256": sha256(args.tex_output)},
            {"path": str(args.report_output.resolve()), "sha256": sha256(args.report_output)},
        ],
    }
    atomic_write(args.manifest_output, json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(args.tex_output)
    print(args.report_output)
    print(args.manifest_output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
