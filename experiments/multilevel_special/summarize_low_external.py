#!/usr/bin/env python3
"""Aggregate low-selectivity external baselines with robust timing statistics."""

from __future__ import annotations

import argparse
import csv
import statistics
from pathlib import Path


WORKLOADS = {
    "query_minlen5_avgsel05pct": "sel_0p5",
    "query_minlen5_avgsel1pct": "sel_1",
    "query_minlen3_avgsel10pct": "sel_10",
}


def cv(values: list[float]) -> float:
    mean = statistics.fmean(values)
    return statistics.pstdev(values) / mean if len(values) > 1 and mean else 0.0


def workload(path: Path) -> str:
    for part in path.parts:
        if part in WORKLOADS:
            return WORKLOADS[part]
        for name, short in WORKLOADS.items():
            if name in part:
                return short
    raise ValueError(f"cannot infer workload from {path}")


def summarize_graph(root: Path, method: str) -> list[dict[str, object]]:
    output = []
    for path in sorted(root.rglob("search_time_details.csv")):
        by_budget: dict[int, list[dict[str, str]]] = {}
        with path.open(newline="", encoding="utf-8") as stream:
            for row in csv.DictReader(stream):
                by_budget.setdefault(int(row["Lsearch"]), []).append(row)
        for budget, rows in sorted(by_budget.items()):
            times = [float(row["Time_ms"]) for row in rows]
            recalls = [float(row["Avg_Recall"]) for row in rows]
            output.append({
                "workload": workload(path), "method": method,
                "variant": "project_route" if method == "NaviX" else "official",
                "budget": budget, "recall": statistics.fmean(recalls),
                "total_ms_mean": statistics.fmean(times),
                "total_ms_median": statistics.median(times),
                "total_ms_cv": cv(times), "total_ms_min": min(times),
                "total_ms_max": max(times), "core_ms_median": "",
                "num_measured_repeats": len(times), "filter_violations": 0,
                "timing_scope": "batch total", "source": str(path),
            })
    return output


def summarize_curator(root: Path) -> list[dict[str, object]]:
    output = []
    for detail_path in sorted(root.rglob("search_time_details.csv")):
        summary_path = detail_path.with_name("search_time_summary.csv")
        with summary_path.open(newline="", encoding="utf-8") as stream:
            recalls = {int(row["Lsearch"]): float(row["Recall"]) for row in csv.DictReader(stream)}
        by_budget: dict[int, list[float]] = {}
        with detail_path.open(newline="", encoding="utf-8") as stream:
            for row in csv.DictReader(stream):
                by_budget.setdefault(int(row["Lsearch"]), []).append(float(row["Total_Time_ms"]))
        for budget, times in sorted(by_budget.items()):
            output.append({
                "workload": workload(detail_path), "method": "Curator",
                "variant": "official_v2_adapter", "budget": budget,
                "recall": recalls[budget], "total_ms_mean": statistics.fmean(times),
                "total_ms_median": statistics.median(times), "total_ms_cv": cv(times),
                "total_ms_min": min(times), "total_ms_max": max(times),
                "core_ms_median": "", "num_measured_repeats": len(times),
                "filter_violations": 0, "timing_scope": "batch total",
                "source": str(detail_path),
            })
    return output


def summarize_acorn(root: Path, source_name: str) -> list[dict[str, object]]:
    output = []
    for path in sorted(root.rglob("*.csv")):
        with path.open(newline="", encoding="utf-8") as stream:
            rows = [row for row in csv.DictReader(stream) if row["warmup"] == "0"]
        if not rows:
            raise RuntimeError(f"no measured rows: {path}")
        violations = sum(int(row["filter_violations"]) for row in rows)
        if violations:
            raise RuntimeError(f"filter violations in {path}: {violations}")
        recalls = {float(row["recall"]) for row in rows}
        if len(recalls) != 1:
            raise RuntimeError(f"Recall changed across repeats: {path}: {recalls}")
        total = [float(row["total_ms"]) for row in rows]
        core = [float(row["search_ms"]) for row in rows]
        output.append({
            "workload": workload(path), "method": "ACORN",
            "variant": next(part for part in path.parts if part.startswith("ACORN-")),
            "budget": int(rows[0]["ef_search"]), "recall": recalls.pop(),
            "total_ms_mean": statistics.fmean(total),
            "total_ms_median": statistics.median(total), "total_ms_cv": cv(total),
            "total_ms_min": min(total), "total_ms_max": max(total),
            "core_ms_median": statistics.median(core),
            "num_measured_repeats": len(rows), "filter_violations": violations,
            "timing_scope": "lookup + materialize + ANN search",
            "source_stage": source_name, "source": str(path),
        })
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--snapshot-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = []
    rows += summarize_graph(args.snapshot_root / "navix_low_selectivity", "NaviX")
    rows += summarize_graph(args.snapshot_root / "favor_low_selectivity", "FAVOR")
    rows += summarize_curator(args.snapshot_root / "curator_low_selectivity")
    rows += summarize_acorn(args.snapshot_root / "acorn_low_selectivity_screen", "screen")
    rows += summarize_acorn(args.snapshot_root / "acorn_low_selectivity_formal", "formal")
    # If a formally repeated point exists, suppress its coarse-screen duplicate.
    formal_keys = {
        (row["workload"], row["variant"], row["budget"])
        for row in rows if row.get("source_stage") == "formal"
    }
    rows = [
        row for row in rows
        if row.get("source_stage") != "screen"
        or (row["workload"], row["variant"], row["budget"]) not in formal_keys
    ]
    for row in rows:
        row.setdefault("source_stage", "formal")
    rows.sort(key=lambda row: (str(row["workload"]), str(row["method"]), str(row["variant"]), int(row["budget"])))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as stream:
        fieldnames = list(rows[0])
        if "source_stage" not in fieldnames:
            fieldnames.insert(-1, "source_stage")
        writer = csv.DictWriter(stream, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader(); writer.writerows(rows)
    for name in WORKLOADS.values():
        print(name)
        for method in ("FAVOR", "NaviX", "Curator", "ACORN"):
            candidates = [row for row in rows if row["workload"] == name and row["method"] == method]
            passing = [row for row in candidates if float(row["recall"]) >= 0.90]
            if passing:
                best = min(passing, key=lambda row: float(row["total_ms_median"]))
                print(method, "PASS", best["variant"], best["budget"], best["recall"], best["total_ms_median"])
            elif candidates:
                best = max(candidates, key=lambda row: float(row["recall"]))
                print(method, "MISS", best["variant"], best["budget"], best["recall"], best["total_ms_median"])


if __name__ == "__main__":
    main()
