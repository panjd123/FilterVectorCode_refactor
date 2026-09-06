#!/usr/bin/env python3
"""Generate the six-workload Multi-level Special Block paper tables.

The script consumes checked aggregate CSVs copied from immutable raw runs.  It
never interpolates Recall: every selected row is an actually measured point.
Formal repeats supersede coarse screens for the same method/budget.
"""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Iterable


HERE = Path(__file__).resolve().parent
OUTPUT_DIR = HERE / "results_summary"
SOURCE = OUTPUT_DIR / "source"

WORKLOADS = {
    "sel_0p5": ("0.499%", 0.0049924907, 0.90),
    "sel_1": ("0.903%", 0.0090305484, 0.90),
    "sel_10": ("9.907%", 0.0990661529, 0.90),
    "sel_25": ("24.915%", 0.2491539473, 0.90),
    "sel_50": ("49.971%", 0.4997135478, 0.85),
    "sel_75": ("74.994%", 0.7499381960, 0.87),
}
EXPECTED_SEARCH_BINARY_SHA256 = (
    "f078e1744775a3aefab6cc670b4a72e02b8d7d7118a6d34a6df76cb591287b11"
)
HIGH_INTERNAL = {
    "sel_25": SOURCE / "internal_current_binary_sel25.csv",
    "sel_50": SOURCE / "internal_current_binary_sel50.csv",
    "sel_75": SOURCE / "internal_current_binary_sel75.csv",
}
HIGH_PLAIN = {
    "sel_25": SOURCE / "internal_plain_sel25_formal.csv",
    "sel_50": SOURCE / "internal_plain_sel50_formal.csv",
    "sel_75": SOURCE / "internal_plain_sel75_formal.csv",
}
HIGH_T1 = {
    "sel_25": SOURCE / "internal_t1_sel25_formal.csv",
    "sel_50": SOURCE / "internal_t1_sel50_formal.csv",
    "sel_75": SOURCE / "internal_t1_sel75_formal.csv",
}
HIGH_PAIRED = {
    "sel_25": SOURCE / "internal_paired_formal_sel25.csv",
    "sel_50": SOURCE / "internal_paired_formal_sel50.csv",
    "sel_75": SOURCE / "internal_paired_formal_sel75.csv",
}
HIGH_CURRENT_MANIFEST = {
    workload: path.with_name(f"{path.stem}_manifest.json")
    for workload, path in HIGH_INTERNAL.items()
}
HIGH_PAIRED_MANIFEST = {
    workload: path.with_name(f"{path.stem}_manifest.json")
    for workload, path in HIGH_PAIRED.items()
}
FORMAL_INTERNAL_MANIFESTS = (
    SOURCE / "internal_low_formal_manifest.json",
    SOURCE / "internal_sel1_crossing_formal_manifest.json",
    SOURCE / "internal_t1_2000_low_formal_manifest.json",
    *(SOURCE / f"internal_plain_sel{suffix}_formal_manifest.json" for suffix in (25, 50, 75)),
    *(SOURCE / f"internal_t1_sel{suffix}_formal_manifest.json" for suffix in (25, 50, 75)),
    *HIGH_CURRENT_MANIFEST.values(),
    *HIGH_PAIRED_MANIFEST.values(),
)
INTERNAL_SOURCE_SPECS = (
    (SOURCE / "internal_low_formal_all_points.csv", SOURCE / "internal_low_formal_manifest.json"),
    (SOURCE / "internal_sel1_crossing_formal_all_points.csv", SOURCE / "internal_sel1_crossing_formal_manifest.json"),
    (SOURCE / "internal_t1_2000_low_formal_all_points.csv", SOURCE / "internal_t1_2000_low_formal_manifest.json"),
    *((HIGH_PLAIN[workload], HIGH_PLAIN[workload].with_name(f"{HIGH_PLAIN[workload].stem}_manifest.json")) for workload in HIGH_PLAIN),
    *((HIGH_T1[workload], HIGH_T1[workload].with_name(f"{HIGH_T1[workload].stem}_manifest.json")) for workload in HIGH_T1),
    *((HIGH_INTERNAL[workload], HIGH_CURRENT_MANIFEST[workload]) for workload in HIGH_INTERNAL),
    *((HIGH_PAIRED[workload], HIGH_PAIRED_MANIFEST[workload]) for workload in HIGH_PAIRED),
)
BUILD_SOURCE = SOURCE / "build_results_source.csv"

RESULT_FIELDS = [
    "comparison", "workload", "mean_selectivity", "recall_threshold",
    "method", "variant", "budget", "status", "recall",
    "recall_gap_to_threshold", "batch_median_ms", "batch_mean_ms",
    "cv", "core_median_ms", "repeats", "timing_scope",
    "speedup_vs_plain", "speedup_vs_tuned_multi", "source",
]


def read(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def select_measured(rows: Iterable[dict[str, object]], threshold: float) -> tuple[dict[str, object], str]:
    rows = list(rows)
    if not rows:
        raise RuntimeError("no measured points supplied")
    passing = [row for row in rows if float(row["recall"]) >= threshold]
    if passing:
        return min(passing, key=lambda row: float(row["batch_median_ms"])), "pass"
    max_recall = max(float(row["recall"]) for row in rows)
    highest = [row for row in rows if float(row["recall"]) == max_recall]
    return min(highest, key=lambda row: float(row["batch_median_ms"])), "quality_limit"


def validate_binary_manifest(path: Path) -> None:
    """Fail closed if a formal internal run came from another search binary."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    runs = payload.get("runs", [])
    if not runs:
        raise RuntimeError(f"manifest has no runs: {path}")
    bad = {run.get("search_binary_sha256") for run in runs} - {EXPECTED_SEARCH_BINARY_SHA256}
    if bad:
        raise RuntimeError(f"mixed search binaries in {path}: {sorted(bad)}")
    incomplete = [run for run in runs if run.get("status") != "complete" or run.get("returncode") != 0]
    if incomplete:
        raise RuntimeError(f"incomplete runs in {path}: {len(incomplete)}")


def validate_internal_source(csv_path: Path, manifest_path: Path) -> dict[tuple[str, str], int]:
    """Bind every aggregate point to the exact grid declared by its manifest.

    A binary hash alone is not enough provenance: a stale aggregate could still
    omit a method, add an undeclared L, or come from another workload.  This
    check makes the CSV and manifest an exact one-to-one description of the
    measured grid and returns the warm-repeat count (cold repeat excluded).
    """
    validate_binary_manifest(manifest_path)
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    declared: dict[tuple[str, str], dict[str, object]] = {}
    expected_points: set[tuple[str, str, int]] = set()
    repeat_counts: dict[tuple[str, str], int] = {}
    for run in payload["runs"]:
        key = (str(run["method"]), str(run["workload"]))
        if key in declared:
            raise RuntimeError(f"duplicate run in {manifest_path}: {key}")
        declared[key] = run
        repeats = int(run["num_repeats"])
        if repeats < 2:
            raise RuntimeError(f"formal run needs a cold and warm repeat: {manifest_path}: {key}")
        repeat_counts[key] = repeats - 1
        for budget in run["lsearch_values"]:
            expected_points.add((key[0], key[1], int(budget)))

    actual_points: set[tuple[str, str, int]] = set()
    for row in read(csv_path):
        key = (row["method"], row["workload"])
        run = declared.get(key)
        if run is None:
            raise RuntimeError(f"aggregate row is absent from manifest {manifest_path}: {key}")
        point = (key[0], key[1], int(row["lsearch"]))
        if point in actual_points:
            raise RuntimeError(f"duplicate aggregate point in {csv_path}: {point}")
        actual_points.add(point)
        if row["query_dir"] != run["query_dir"]:
            raise RuntimeError(f"query mismatch between {csv_path} and {manifest_path}: {point}")
        if abs(float(row["mean_selectivity"]) - float(run["mean_selectivity"])) > 1e-12:
            raise RuntimeError(f"selectivity mismatch between {csv_path} and {manifest_path}: {point}")
    if actual_points != expected_points:
        missing = sorted(expected_points - actual_points)
        extra = sorted(actual_points - expected_points)
        raise RuntimeError(
            f"aggregate grid mismatch for {csv_path}; missing={missing[:5]} extra={extra[:5]}"
        )
    return repeat_counts


def validate_all_internal_sources() -> dict[tuple[str, str, str], int]:
    repeat_counts: dict[tuple[str, str, str], int] = {}
    for csv_path, manifest_path in INTERNAL_SOURCE_SPECS:
        for (method, workload), repeats in validate_internal_source(csv_path, manifest_path).items():
            repeat_counts[(csv_path.name, method, workload)] = repeats
    return repeat_counts


def normalize_internal(
    row: dict[str, str], method: str, variant: str, source_file: str,
    repeat_counts: dict[tuple[str, str, str], int], priority: int = 0
) -> dict[str, object]:
    return {
        "workload": row["workload"], "method": method, "variant": variant,
        "budget": f"L{row['lsearch']}", "recall": float(row["recall"]),
        "batch_median_ms": float(row["batch_ms_warm_median"]),
        "batch_mean_ms": float(row["batch_ms_warm"]),
        "cv": float(row["batch_ms_warm_cv"]), "core_median_ms": "",
        "repeats": repeat_counts[(source_file, row["method"], row["workload"])],
        "timing_scope": "warm 1000-query batch; cold repeat excluded",
        "source": f"results_summary/source/{source_file}",
        "priority": priority,
    }


def deduplicate_internal(rows: Iterable[dict[str, object]]) -> list[dict[str, object]]:
    """Canonicalize duplicate points; a paired rerun supersedes an older run.

    The key is ``(workload, method family, variant, budget)``.  This means the
    generated aggregate contains one canonical statistic per experimental
    point, not one row per historical execution of that point.
    """
    selected: dict[tuple[object, ...], dict[str, object]] = {}
    for row in rows:
        key = (row["workload"], row["method"], row["variant"], row["budget"])
        previous = selected.get(key)
        if previous is None or int(row.get("priority", 0)) > int(previous.get("priority", 0)):
            selected[key] = row
    return list(selected.values())


def internal_candidates() -> list[dict[str, object]]:
    output: list[dict[str, object]] = []
    repeat_counts = validate_all_internal_sources()
    low = read(SOURCE / "internal_low_formal_all_points.csv")
    sel1_crossing = read(SOURCE / "internal_sel1_crossing_formal_all_points.csv")
    tuned_low = read(SOURCE / "internal_t1_2000_low_formal_all_points.csv")

    # Low-selectivity plain is taken from the broad formal run.  For sel_1,
    # crossing-formal replaces sparse L=7500 special points with the denser
    # formal grid around Recall 0.90.
    for row in low:
        raw = row["method"]
        if raw == "plain":
            output.append(normalize_internal(row, "Plain UNG", "no special overlay", "internal_low_formal_all_points.csv", repeat_counts))
        elif row["workload"] != "sel_1":
            if raw == "single_1k":
                output.append(normalize_internal(row, "Single-level", "T1=1k", "internal_low_formal_all_points.csv", repeat_counts))
            elif raw.startswith("multi_1k_"):
                output.append(normalize_internal(row, "Original Multi-level", raw.replace("multi_1k_", "T1=1k,T2="), "internal_low_formal_all_points.csv", repeat_counts))
    for row in sel1_crossing:
        raw = row["method"]
        if raw == "single_1k":
            output.append(normalize_internal(row, "Single-level", "T1=1k", "internal_sel1_crossing_formal_all_points.csv", repeat_counts))
        elif raw.startswith("multi_1k_"):
            output.append(normalize_internal(row, "Original Multi-level", raw.replace("multi_1k_", "T1=1k,T2="), "internal_sel1_crossing_formal_all_points.csv", repeat_counts))
    for row in tuned_low:
        output.append(normalize_internal(row, "Tuned Multi-level", "T1=2k,T2=25k", "internal_t1_2000_low_formal_all_points.csv", repeat_counts))

    for workload in ("sel_25", "sel_50", "sel_75"):
        validate_binary_manifest(HIGH_CURRENT_MANIFEST[workload])
        validate_binary_manifest(HIGH_PAIRED_MANIFEST[workload])
        for row in read(HIGH_PLAIN[workload]):
            output.append(normalize_internal(row, "Plain UNG", "no special overlay", HIGH_PLAIN[workload].name, repeat_counts))
        for row in read(HIGH_INTERNAL[workload]):
            raw = row["method"]
            if raw == "single_1k":
                output.append(normalize_internal(row, "Single-level", "T1=1k", HIGH_INTERNAL[workload].name, repeat_counts, 10))
            elif raw.startswith("multi_1k_"):
                output.append(normalize_internal(row, "Original Multi-level", raw.replace("multi_1k_", "T1=1k,T2="), HIGH_INTERNAL[workload].name, repeat_counts, 10))
        for row in read(HIGH_T1[workload]):
            raw = row["method"]
            if raw == "multi_t1_500_t2_25k":
                output.append(normalize_internal(row, "T1 sensitivity", "T1=500,T2=25k", HIGH_T1[workload].name, repeat_counts, 10))
            elif raw == "multi_t1_1000_t2_25k":
                output.append(normalize_internal(row, "Original Multi-level", "T1=1k,T2=25k", HIGH_T1[workload].name, repeat_counts, 10))
            elif raw == "multi_t1_2000_t2_25k":
                output.append(normalize_internal(row, "Tuned Multi-level", "T1=2k,T2=25k", HIGH_T1[workload].name, repeat_counts, 10))
        for row in read(HIGH_PAIRED[workload]):
            raw = row["method"]
            if raw == "single_1k":
                method, variant = "Single-level", "T1=1k"
            elif raw == "multi_1k_25k_upper_off":
                method, variant = "Upper-level ablation", "T1=1k,T2=25k,upper=off"
            elif raw == "multi_1k_25k_upper_on":
                method, variant = "Original Multi-level", "T1=1k,T2=25k"
            elif raw == "multi_t1_2000_t2_25k":
                method, variant = "Tuned Multi-level", "T1=2k,T2=25k"
            else:
                raise RuntimeError(f"unknown paired method: {raw}")
            output.append(normalize_internal(row, method, variant, HIGH_PAIRED[workload].name, repeat_counts, 20))
    return deduplicate_internal(output)


def normalize_external_low(row: dict[str, str]) -> dict[str, object]:
    budget_prefix = "ef" if row["method"] in {"Curator", "ACORN"} else "L"
    return {
        "workload": row["workload"], "method": row["method"],
        "variant": row["variant"], "budget": f"{budget_prefix}{row['budget']}",
        "recall": float(row["recall"]),
        "batch_median_ms": float(row["total_ms_median"]),
        "batch_mean_ms": float(row["total_ms_mean"]),
        "cv": float(row["total_ms_cv"]),
        "core_median_ms": row["core_ms_median"],
        "repeats": int(row["num_measured_repeats"]),
        "timing_scope": row["timing_scope"],
        "source": "results_summary/source/external_low_robust.csv",
    }


def external_candidates() -> list[dict[str, object]]:
    output = [normalize_external_low(row) for row in read(SOURCE / "external_low_robust.csv")]
    graph_rows = read(SOURCE / "navix_favor_robust.csv")
    for row in graph_rows:
        output.append({
            "workload": row["workload"], "method": row["method"],
            "variant": "official" if row["method"] == "FAVOR" else "project route",
            "budget": f"L{row['budget']}", "recall": float(row["recall"]),
            "batch_median_ms": float(row["batch_ms_warm_median"]),
            "batch_mean_ms": float(row["batch_ms_warm_mean"]),
            "cv": float(row["warm_cv"]), "core_median_ms": "",
            "repeats": 4, "timing_scope": "warm total batch; cold repeat excluded",
            "source": "results_summary/source/navix_favor_robust.csv",
        })
    for row in read(SOURCE / "curator_robust.csv"):
        output.append({
            "workload": row["workload"], "method": "Curator",
            "variant": "official v2 adapter", "budget": f"ef{row['budget']}",
            "recall": float(row["recall"]),
            "batch_median_ms": float(row["batch_ms_median"]),
            "batch_mean_ms": float(row["batch_ms_mean"]), "cv": float(row["cv"]),
            "core_median_ms": "", "repeats": int(row["num_repeats"]),
            "timing_scope": "batch total",
            "source": "results_summary/source/curator_robust.csv",
        })
    acorn_rows = read(SOURCE / "acorn_robust.csv")
    formal_keys = {(row["variant"], row["workload"], row["ef_search"]) for row in acorn_rows if row["source"] == "selected_formal"}
    for row in acorn_rows:
        key = (row["variant"], row["workload"], row["ef_search"])
        if row["source"] != "selected_formal" and key in formal_keys:
            continue
        workload = {"25%": "sel_25", "50%": "sel_50", "75%": "sel_75"}[row["workload"]]
        output.append({
            "workload": workload, "method": "ACORN", "variant": row["variant"],
            "budget": f"ef{row['ef_search']}", "recall": float(row["recall"]),
            "batch_median_ms": float(row["total_ms_median"]),
            "batch_mean_ms": float(row["total_ms_mean"]),
            "cv": float(row["total_ms_cv"]),
            "core_median_ms": row["search_ms_median"],
            "repeats": int(row["num_measured_repeats"]),
            "timing_scope": "lookup + materialize + ANN search",
            "source": "results_summary/source/acorn_robust.csv",
        })
    return output


def output_row(comparison: str, workload: str, threshold: float, row: dict[str, object], status: str) -> dict[str, object]:
    label, mean_selectivity, _ = WORKLOADS[workload]
    recall = float(row["recall"]); gap = recall - threshold
    return {
        "comparison": comparison, "workload": workload,
        "mean_selectivity": f"{mean_selectivity:.10f}", "recall_threshold": f"{threshold:.4f}",
        "method": row["method"], "variant": row["variant"], "budget": row["budget"],
        "status": status, "recall": f"{recall:.6f}",
        "recall_gap_to_threshold": f"{gap:.6f}",
        "batch_median_ms": f"{float(row['batch_median_ms']):.6f}",
        "batch_mean_ms": f"{float(row['batch_mean_ms']):.6f}",
        "cv": f"{float(row['cv']):.6f}", "core_median_ms": row["core_median_ms"],
        "repeats": row["repeats"], "timing_scope": row["timing_scope"],
        "speedup_vs_plain": "", "speedup_vs_tuned_multi": "", "source": row["source"],
    }


def build_results() -> list[dict[str, object]]:
    # Blank cells mean that the historical builder did not emit that metric;
    # the source table records this explicitly rather than inferring values.
    return read(BUILD_SOURCE)


def write_markdown_summary(selected: list[dict[str, object]], build: list[dict[str, object]]) -> None:
    internal = [row for row in selected if row["comparison"] == "internal"]
    external = [row for row in selected if row["comparison"] == "external"]
    lines = [
        "# Generated Multi-level Special Block Results",
        "",
        "All rows are measured points. `quality_limit` means that no scanned point reached the declared Recall threshold.",
        "",
        "## Internal comparison",
        "",
        "| Selectivity | Recall target | Method | Configuration | Budget | Status | Recall | Batch median (ms) | Speedup vs plain |",
        "|---:|---:|---|---|---:|---|---:|---:|---:|",
    ]
    for workload in WORKLOADS:
        label = WORKLOADS[workload][0]
        for row in (item for item in internal if item["workload"] == workload):
            speedup = f"{float(row['speedup_vs_plain']):.3f}x" if row["speedup_vs_plain"] else "--"
            lines.append(
                f"| {label} | {row['recall_threshold']} | {row['method']} | {row['variant']} | "
                f"{row['budget']} | {row['status']} | {float(row['recall']):.4f} | "
                f"{float(row['batch_median_ms']):.3f} | {speedup} |"
            )
    lines += ["", "## External system position", "",
        "| Selectivity | Recall target | Method | Configuration | Budget | Status | Recall | Total median (ms) | Core median (ms) |",
        "|---:|---:|---|---|---:|---|---:|---:|---:|"]
    for workload in WORKLOADS:
        label = WORKLOADS[workload][0]
        for row in (item for item in external if item["workload"] == workload):
            core = f"{float(row['core_median_ms']):.3f}" if row["core_median_ms"] else "--"
            lines.append(
                f"| {label} | {row['recall_threshold']} | {row['method']} | {row['variant']} | "
                f"{row['budget']} | {row['status']} | {float(row['recall']):.4f} | "
                f"{float(row['batch_median_ms']):.3f} | {core} |"
            )
    lines += ["", "## Build cost", "",
        "| Method | T1 | T2 | Middle blocks | Upper blocks | Builder wall (s) | Edge stage (s) | Special edges | Sidecar bytes | Loaded bytes |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for row in build:
        lines.append(
            "| {method} | {T1} | {T2} | {middle_blocks} | {upper_blocks} | {total_builder_wall_s} | "
            "{special_edge_stage_s} | {special_edges} | {sidecar_bytes} | {loaded_allocated_bytes} |".format(**row)
        )
    (OUTPUT_DIR / "paper_results.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_csv(path: Path, fields: list[str], rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader(); writer.writerows(rows)


def write_source_manifest() -> None:
    rows = []
    source_paths = {
        SOURCE / "internal_low_formal_all_points.csv",
        SOURCE / "internal_sel1_crossing_formal_all_points.csv",
        SOURCE / "internal_t1_2000_low_formal_all_points.csv",
        SOURCE / "external_low_robust.csv",
        SOURCE / "navix_favor_robust.csv",
        SOURCE / "curator_robust.csv",
        SOURCE / "acorn_robust.csv",
        BUILD_SOURCE,
        *HIGH_INTERNAL.values(),
        *HIGH_PLAIN.values(),
        *HIGH_T1.values(),
        *HIGH_PAIRED.values(),
        *FORMAL_INTERNAL_MANIFESTS,
    }
    for path in sorted(source_paths):
        payload = path.read_bytes()
        row_count: object = ""
        if path.suffix == ".csv":
            with path.open(newline="", encoding="utf-8") as stream:
                row_count = sum(1 for _ in csv.DictReader(stream))
        rows.append({
            "path": str(path.relative_to(HERE)),
            "sha256": hashlib.sha256(payload).hexdigest(),
            "data_rows": row_count,
        })
    write_csv(OUTPUT_DIR / "source_manifest.csv", ["path", "sha256", "data_rows"], rows)


def write_artifact_manifest() -> None:
    """Hash the immutable paper-results closure.

    Operational status files such as AGENT_KANBAN.md and WORKTREE_HANDOFF.md
    intentionally stay outside this closure because they are updated after a
    checkpoint is created.
    """
    report = HERE.parents[1] / "docs/reports/MULTILEVEL_SPECIAL_BLOCK_PAPER_REPORT_CN.md"
    if not report.is_file():  # Supports the compact local editing mirror.
        report = HERE / "MULTILEVEL_SPECIAL_BLOCK_PAPER_REPORT_CN.md"
    overview = HERE.parents[1] / "docs/reports/THREE_MAINLINES_METHOD_BASELINE_DATA_SPEEDUP_CN.md"
    if not overview.is_file():
        overview = HERE / "THREE_MAINLINES_METHOD_BASELINE_DATA_SPEEDUP_CN.md"
    runbook = HERE.parents[1] / "docs/reports/MULTILEVEL_SPECIAL_BLOCK_REPRODUCE_CN.md"
    if not runbook.is_file():
        runbook = HERE / "MULTILEVEL_SPECIAL_BLOCK_REPRODUCE_CN.md"
    paths = [
        Path(__file__).resolve(),
        HERE / "audit_current_source_regression.py",
        HERE / "test_generate_paper_results.py",
        *(HERE / f"config.amazon_x1_paired_formal_sel{suffix}.json" for suffix in (25, 50, 75)),
        *(HERE / f"config.amazon_x1_current_source_main_sel{suffix}.json"
          for suffix in (1, 10, 25, 50, 75)),
        HERE / "config.amazon_x1_current_source_main_sel0p5_repeat21.json",
        OUTPUT_DIR / "source_manifest.csv",
        OUTPUT_DIR / "paper_results.csv",
        OUTPUT_DIR / "paper_results.md",
        OUTPUT_DIR / "build_results.csv",
        OUTPUT_DIR / "internal_canonical_measured_points.csv",
        OUTPUT_DIR / "external_canonical_measured_points.csv",
        OUTPUT_DIR / "current_source_regression.csv",
        report,
        overview,
        runbook,
    ]
    rows = []
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(path)
        try:
            label = str(path.relative_to(HERE.parents[1]))
        except ValueError:
            label = path.name
        rows.append({"path": label, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
    write_csv(OUTPUT_DIR / "artifact_manifest.csv", ["path", "sha256"], rows)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for manifest in FORMAL_INTERNAL_MANIFESTS:
        validate_binary_manifest(manifest)
    internal = internal_candidates(); external = external_candidates()
    selected: list[dict[str, object]] = []
    for workload, (_, _, threshold) in WORKLOADS.items():
        chosen_internal: dict[str, dict[str, object]] = {}
        for method in ("Plain UNG", "Single-level", "Original Multi-level", "Tuned Multi-level"):
            point, status = select_measured((row for row in internal if row["workload"] == workload and row["method"] == method), threshold)
            chosen_internal[method] = point
            selected.append(output_row("internal", workload, threshold, point, status))
        plain_ms = float(chosen_internal["Plain UNG"]["batch_median_ms"])
        tuned_ms = float(chosen_internal["Tuned Multi-level"]["batch_median_ms"])
        for row in selected[-4:]:
            if row["status"] == "pass":
                row["speedup_vs_plain"] = f"{plain_ms / float(row['batch_median_ms']):.6f}"
                row["speedup_vs_tuned_multi"] = f"{tuned_ms / float(row['batch_median_ms']):.6f}"

        # Cross-system table includes both the plain and tuned internal anchors.
        for method in ("Plain UNG", "Tuned Multi-level"):
            point = chosen_internal[method]
            status = "pass" if float(point["recall"]) >= threshold else "quality_limit"
            selected.append(output_row("external", workload, threshold, point, status))
        for method in ("FAVOR", "NaviX", "Curator", "ACORN"):
            point, status = select_measured((row for row in external if row["workload"] == workload and row["method"] == method), threshold)
            selected.append(output_row("external", workload, threshold, point, status))
        for row in selected[-6:]:
            if row["status"] == "pass":
                row["speedup_vs_plain"] = f"{plain_ms / float(row['batch_median_ms']):.6f}"
                row["speedup_vs_tuned_multi"] = f"{tuned_ms / float(row['batch_median_ms']):.6f}"

    write_csv(OUTPUT_DIR / "paper_results.csv", RESULT_FIELDS, selected)
    build = build_results()
    write_csv(OUTPUT_DIR / "build_results.csv", list(build[0]), build)
    all_fields = ["workload", "method", "variant", "budget", "recall", "batch_median_ms", "batch_mean_ms", "cv", "core_median_ms", "repeats", "timing_scope", "source"]
    write_csv(OUTPUT_DIR / "internal_canonical_measured_points.csv", all_fields, [{key: row.get(key, "") for key in all_fields} for row in internal])
    write_csv(OUTPUT_DIR / "external_canonical_measured_points.csv", all_fields, [{key: row.get(key, "") for key in all_fields} for row in external])
    write_markdown_summary(selected, build)
    write_source_manifest()
    write_artifact_manifest()
    print(f"wrote {len(selected)} selected rows, {len(internal)} internal points, {len(build)} build rows")


if __name__ == "__main__":
    main()
