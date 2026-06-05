#!/usr/bin/env python3
"""Summarize x400 repair/reverse-tail A/B outputs.

The script expects the output layout produced by
scripts/benchmarks/run_x400_reverse_tail_ab.sh:

  <root>/<case>/<variant>/results/build_time.csv
  <root>/<case>/<variant>/results/search_time_summary.csv
  <root>/<case>/<variant>/results/query_details_repeat*.csv

Graph diagnostics are optional. If present under either
<root>/<case>/graph_diag or <root>/<case>/<variant>/graph_diag, the script
also folds graph-structure metrics into the same reviewer-facing table.
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path
from typing import Any


DEFAULT_CASE_ORDER = [
    "cpu_vamana",
    "light512_norepair",
    "light512_repair",
    "light512_reverse_repair",
    "light512_repair_compact_off",
]

BUILD_FIELDS = [
    "index_time",
    "build_graph_time",
    "build_cross_edges_time",
    "tagore_direct_build_wall_time",
    "tagore_h2d_time",
    "tagore_gnn_time",
    "tagore_prune_time",
    "tagore_d2h_time",
    "tagore_fill_time",
]

GRAPH_FIELDS = [
    "intra_edges",
    "cross_edges",
    "intra_out_degree_avg",
    "zero_intra_ratio",
    "low_intra_le4_ratio",
    "group_largest_wcc_ratio_avg",
    "group_largest_wcc_ratio_p05",
    "groups_with_largest_wcc_lt_0_9",
]


def fmt(value: Any) -> str:
    if value is None or value == "":
        return ""
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        if value == 0:
            return "0"
        if abs(value) >= 10000 or abs(value) < 0.001:
            return f"{value:.6g}"
        return f"{value:.6f}".rstrip("0").rstrip(".")
    return str(value)


def to_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def percentile(values: list[float], p: float) -> float | None:
    if not values:
        return None
    xs = sorted(values)
    pos = (len(xs) - 1) * p / 100.0
    lo = int(pos)
    hi = min(lo + 1, len(xs) - 1)
    frac = pos - lo
    return xs[lo] * (1.0 - frac) + xs[hi] * frac


def read_key_value_csv(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    out: dict[str, str] = {}
    with path.open() as f:
        next(f, None)
        for line in f:
            line = line.strip()
            if not line or "," not in line:
                continue
            key, value = line.split(",", 1)
            out[key] = value
    return out


def read_search_summary(result_dir: Path) -> dict[str, dict[str, str]]:
    path = result_dir / "search_time_summary.csv"
    if not path.exists():
        return {}
    rows: dict[str, dict[str, str]] = {}
    with path.open() as f:
        for row in csv.DictReader(f):
            lsearch = row.get("Lsearch", "")
            if lsearch:
                rows[lsearch] = row
    return rows


def read_query_details(result_dir: Path) -> dict[str, dict[str, float]]:
    paths = sorted(result_dir.glob("query_details_repeat*.csv"))
    if not paths:
        return {}
    latest = paths[-1]
    times_by_l: dict[str, list[float]] = {}
    recalls_by_l: dict[str, list[float]] = {}
    with latest.open() as f:
        for row in csv.DictReader(f):
            lsearch = row.get("Lsearch", "")
            time_ms = to_float(row.get("Time_ms"))
            recall = to_float(row.get("Recall"))
            if not lsearch:
                continue
            if time_ms is not None:
                times_by_l.setdefault(lsearch, []).append(time_ms)
            if recall is not None:
                recalls_by_l.setdefault(lsearch, []).append(recall)
    out: dict[str, dict[str, float]] = {}
    for lsearch in sorted(set(times_by_l) | set(recalls_by_l), key=lambda x: int(float(x))):
        times = times_by_l.get(lsearch, [])
        recalls = recalls_by_l.get(lsearch, [])
        out[lsearch] = {
            "query_p50_ms": percentile(times, 50) or 0.0,
            "query_p95_ms": percentile(times, 95) or 0.0,
            "query_p99_ms": percentile(times, 99) or 0.0,
            "mean_query_recall": statistics.mean(recalls) if recalls else 0.0,
        }
    return out


def load_graph_diag(case_dir: Path, run_dir: Path) -> dict[str, Any]:
    candidates = [
        case_dir / "graph_diag" / "graph_structure_summary.json",
        run_dir / "graph_diag" / "graph_structure_summary.json",
        case_dir / "graph_diagnostics" / "graph_structure_summary.json",
        run_dir / "graph_diagnostics" / "graph_structure_summary.json",
    ]
    for path in candidates:
        if path.exists():
            return json.loads(path.read_text())
    return {}


def find_run_dir(case_dir: Path) -> Path | None:
    for variant in ("cpu_vamana_group", "fastgrnnd_cpu_fallback", "fastgrnnd_complete_fallback"):
        run_dir = case_dir / variant
        if (run_dir / "results").exists():
            return run_dir
    return None


def discover_cases(root: Path, explicit: list[str]) -> list[tuple[str, Path, Path]]:
    if explicit:
        case_names = explicit
    else:
        case_names = [name for name in DEFAULT_CASE_ORDER if (root / name).exists()]
        extras = sorted(p.name for p in root.iterdir() if p.is_dir() and p.name not in set(case_names))
        case_names.extend(extras)
    out: list[tuple[str, Path, Path]] = []
    for name in case_names:
        case_dir = root / name
        run_dir = find_run_dir(case_dir)
        if run_dir is not None:
            out.append((name, case_dir, run_dir))
    return out


def summarize_case(name: str, case_dir: Path, run_dir: Path, lsearch_values: list[str]) -> dict[str, Any]:
    result_dir = run_dir / "results"
    build = read_key_value_csv(result_dir / "build_time.csv")
    search = read_search_summary(result_dir)
    details = read_query_details(result_dir)
    graph = load_graph_diag(case_dir, run_dir)

    row: dict[str, Any] = {
        "case": name,
        "run_dir": str(run_dir),
    }
    for field in BUILD_FIELDS:
        row[field] = build.get(field, "")
    for lsearch in lsearch_values:
        srow = search.get(lsearch, {})
        drow = details.get(lsearch, {})
        row[f"L{lsearch}_recall"] = srow.get("Average_Recall", "")
        row[f"L{lsearch}_avg_time_ms"] = srow.get("Average_Time_ms", "")
        row[f"L{lsearch}_avg_efs"] = srow.get("Average_Efs", "")
        row[f"L{lsearch}_p95_ms"] = drow.get("query_p95_ms", "")
        row[f"L{lsearch}_p99_ms"] = drow.get("query_p99_ms", "")
    for field in GRAPH_FIELDS:
        row[field] = graph.get(field, "")
    return row


def add_deltas(rows: list[dict[str, Any]], lsearch_values: list[str]) -> None:
    base = rows[0] if rows else {}
    base_index = to_float(base.get("index_time"))
    base_group = to_float(base.get("build_graph_time"))
    for row in rows:
        index = to_float(row.get("index_time"))
        group = to_float(row.get("build_graph_time"))
        row["index_speedup_vs_base"] = (base_index / index) if base_index and index else ""
        row["group_speedup_vs_base"] = (base_group / group) if base_group and group else ""
        for lsearch in lsearch_values:
            recall = to_float(row.get(f"L{lsearch}_recall"))
            base_recall = to_float(base.get(f"L{lsearch}_recall"))
            row[f"L{lsearch}_recall_delta_vs_base"] = (
                recall - base_recall if recall is not None and base_recall is not None else ""
            )
        low = to_float(row.get("low_intra_le4_ratio"))
        base_low = to_float(base.get("low_intra_le4_ratio"))
        row["low_intra_le4_delta_vs_base"] = low - base_low if low is not None and base_low is not None else ""


def write_csv(rows: list[dict[str, Any]], fields: list[str], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def make_markdown(rows: list[dict[str, Any]], fields: list[str], lsearch_values: list[str]) -> str:
    cols = [
        "case",
        "index_time",
        "build_graph_time",
        "tagore_prune_time",
        "tagore_d2h_time",
        "tagore_fill_time",
        "index_speedup_vs_base",
        "group_speedup_vs_base",
    ]
    for lsearch in lsearch_values:
        cols.extend([f"L{lsearch}_recall", f"L{lsearch}_recall_delta_vs_base", f"L{lsearch}_p95_ms"])
    cols.extend([
        "zero_intra_ratio",
        "low_intra_le4_ratio",
        "groups_with_largest_wcc_lt_0_9",
    ])
    cols = [c for c in cols if c in fields]
    lines = ["| " + " | ".join(cols) + " |", "|---" + "|---:" * (len(cols) - 1) + "|"]
    for row in rows:
        lines.append("| " + " | ".join(fmt(row.get(c, "")) for c in cols) + " |")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path, help="Output root from run_x400_reverse_tail_ab.sh.")
    parser.add_argument("--cases", nargs="*", default=[], help="Optional case names under root.")
    parser.add_argument("--lsearch", default="1000,5000", help="Comma-separated Lsearch values to report.")
    parser.add_argument("--csv", type=Path, help="CSV output path.")
    parser.add_argument("--md", type=Path, help="Markdown output path.")
    args = parser.parse_args()

    lsearch_values = [x.strip() for x in args.lsearch.replace(" ", ",").split(",") if x.strip()]
    cases = discover_cases(args.root, args.cases)
    rows = [summarize_case(name, case_dir, run_dir, lsearch_values) for name, case_dir, run_dir in cases]
    add_deltas(rows, lsearch_values)

    fields = [
        "case",
        *BUILD_FIELDS,
        "index_speedup_vs_base",
        "group_speedup_vs_base",
    ]
    for lsearch in lsearch_values:
        fields.extend([
            f"L{lsearch}_recall",
            f"L{lsearch}_recall_delta_vs_base",
            f"L{lsearch}_avg_time_ms",
            f"L{lsearch}_avg_efs",
            f"L{lsearch}_p95_ms",
            f"L{lsearch}_p99_ms",
        ])
    fields.extend([*GRAPH_FIELDS, "low_intra_le4_delta_vs_base", "run_dir"])

    if args.csv:
        write_csv(rows, fields, args.csv)
    md = make_markdown(rows, fields, lsearch_values)
    if args.md:
        args.md.parent.mkdir(parents=True, exist_ok=True)
        args.md.write_text(md)
    print(md, end="")


if __name__ == "__main__":
    main()
