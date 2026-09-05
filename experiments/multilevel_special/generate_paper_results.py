#!/usr/bin/env python3
"""Regenerate the Multi-level Special Block paper table.

Inputs are compact aggregate summaries, not raw query logs.  Selection uses the
fastest measured point at or above a pre-declared Recall threshold; there is no
interpolation or extrapolation.
"""

from __future__ import annotations

import csv
from pathlib import Path


HERE = Path(__file__).resolve().parent
OUTPUT_DIR = HERE / "results_summary"
SOURCE = OUTPUT_DIR / "source"
THRESHOLDS = {"sel_25": 0.90, "sel_50": 0.85, "sel_75": 0.87}
PERCENT = {"sel_25": "25%", "sel_50": "50%", "sel_75": "75%"}
INTERNAL = {
    "sel_25": SOURCE / "internal_sel25_all_points.csv",
    "sel_50": SOURCE / "internal_sel50_all_points.csv",
    "sel_75": SOURCE / "internal_sel75_all_points.csv",
}
FIELDS = [
    "comparison", "workload", "recall_threshold", "method",
    "variant", "budget", "recall", "batch_median_ms",
    "batch_mean_ms", "cv", "core_median_ms", "repeats",
    "timing_scope",
]


def read(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def fastest(rows, threshold: float, key: str):
    candidates = [row for row in rows if float(row["recall"]) >= threshold]
    if not candidates:
        raise RuntimeError(f"no measured point reaches Recall {threshold}")
    return min(candidates, key=lambda row: float(row[key]))


def internal_row(comparison, workload, threshold, row):
    return {
        "comparison": comparison,
        "workload": workload,
        "recall_threshold": f"{threshold:.4f}",
        "method": "Single" if row["method"] == "single_1k" else "Multi",
        "variant": row["method"].replace("single_1k", "T1=1k").replace("multi_1k_", "T1=1k_T2="),
        "budget": f"L{row['lsearch']}",
        "recall": row["recall"],
        "batch_median_ms": row["batch_ms_warm_median"],
        "batch_mean_ms": row["batch_ms_warm"],
        "cv": row["batch_ms_warm_cv"],
        "core_median_ms": "",
        "repeats": "6",
        "timing_scope": "warm 1000-query batch excluding repeat 0",
    }


def main() -> None:
    output = []
    internal_cache = {workload: read(path) for workload, path in INTERNAL.items()}
    external = read(SOURCE / "navix_favor_robust.csv")
    curator = read(SOURCE / "curator_robust.csv")
    acorn = read(SOURCE / "acorn_robust.csv")

    for workload, threshold in THRESHOLDS.items():
        rows = internal_cache[workload]
        single = fastest([r for r in rows if r["method"] == "single_1k"], threshold, "batch_ms_warm_median")
        multi = fastest([r for r in rows if r["method"].startswith("multi_")], threshold, "batch_ms_warm_median")
        output.extend([internal_row("internal", workload, threshold, single), internal_row("internal", workload, threshold, multi)])

        highest_single = max(
            (r for r in rows if r["method"] == "single_1k"),
            key=lambda r: float(r["recall"]),
        )
        high_threshold = float(highest_single["recall"])
        highest_multi = fastest(
            [r for r in rows if r["method"].startswith("multi_")],
            high_threshold,
            "batch_ms_warm_median",
        )
        output.extend([internal_row("high_quality", workload, high_threshold, highest_single), internal_row("high_quality", workload, high_threshold, highest_multi)])

        output.append(internal_row("external", workload, threshold, multi))
        for method in ("FAVOR", "NaviX"):
            row = fastest([r for r in external if r["workload"] == workload and r["method"] == method], threshold, "batch_ms_warm_median")
            output.append({
                "comparison": "external", "workload": workload,
                "recall_threshold": f"{threshold:.4f}", "method": method,
                "variant": "official" if method == "FAVOR" else "project route",
                "budget": f"L{row['budget']}", "recall": row["recall"],
                "batch_median_ms": row["batch_ms_warm_median"],
                "batch_mean_ms": row["batch_ms_warm_mean"], "cv": row["warm_cv"],
                "core_median_ms": "", "repeats": "4",
                "timing_scope": "warm total batch excluding repeat 0",
            })
        row = fastest([r for r in curator if r["workload"] == workload], threshold, "batch_ms_median")
        output.append({
            "comparison": "external", "workload": workload,
            "recall_threshold": f"{threshold:.4f}", "method": "Curator",
            "variant": "official v2 adapter", "budget": f"ef{row['budget']}",
            "recall": row["recall"], "batch_median_ms": row["batch_ms_median"],
            "batch_mean_ms": row["batch_ms_mean"], "cv": row["cv"],
            "core_median_ms": "", "repeats": row["num_repeats"],
            "timing_scope": "total batch",
        })

        acorn_rows = [r for r in acorn if r["workload"] == PERCENT[workload]]
        formal_keys = {(r["variant"], r["ef_search"]) for r in acorn_rows if r["source"] == "selected_formal"}
        acorn_rows = [r for r in acorn_rows if r["source"] == "selected_formal" or (r["variant"], r["ef_search"]) not in formal_keys]
        row = fastest(acorn_rows, threshold, "total_ms_median")
        output.append({
            "comparison": "external", "workload": workload,
            "recall_threshold": f"{threshold:.4f}", "method": "ACORN",
            "variant": row["variant"], "budget": f"ef{row['ef_search']}",
            "recall": row["recall"], "batch_median_ms": row["total_ms_median"],
            "batch_mean_ms": row["total_ms_mean"], "cv": row["total_ms_cv"],
            "core_median_ms": row["search_ms_median"],
            "repeats": row["num_measured_repeats"],
            "timing_scope": "total includes lookup/materialization/search",
        })

    with (OUTPUT_DIR / "paper_results.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(output)


if __name__ == "__main__":
    main()
