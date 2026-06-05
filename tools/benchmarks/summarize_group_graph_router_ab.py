#!/usr/bin/env python3
"""Summarize FastGrnndCuda mixed exact/GNN router A/B outputs.

Expected input is the output root from:

  scripts/benchmarks/run_group_graph_router_ab.sh

The root-level summary.csv already joins build and search outputs from each
child end-to-end run. This script adds reviewer-facing deltas against both CPU
Vamana and the non-exact FastGrnndCuda route, then writes compact CSV/Markdown
tables for paper evidence tracking.
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path
from typing import Any


BASE_FIELDS = [
    "case",
    "router_exact_nx",
    "variant",
    "index_ms",
    "group_ms",
    "cross_ms",
    "lsearch",
    "avg_efs",
    "avg_time_ms",
    "avg_recall",
]


def to_float(value: Any) -> float | None:
    if value is None:
        return None
    text = str(value).strip()
    if text == "" or text.upper() == "NA":
        return None
    try:
        return float(text)
    except ValueError:
        return None


def fmt(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        if value == 0:
            return "0"
        if abs(value) >= 10000 or abs(value) < 0.001:
            return f"{value:.6g}"
        return f"{value:.6f}".rstrip("0").rstrip(".")
    return str(value)


def read_rows(root: Path) -> list[dict[str, str]]:
    path = root / "summary.csv"
    if not path.exists():
        raise FileNotFoundError(f"missing router summary: {path}")
    with path.open() as f:
        return list(csv.DictReader(f))


def choose_baselines_by_lsearch(
    rows: list[dict[str, str]],
) -> tuple[dict[str, dict[str, str]], dict[str, dict[str, str]]]:
    cpu = {r.get("lsearch", ""): r for r in rows if r.get("case") == "cpu_vamana"}
    router0 = {r.get("lsearch", ""): r for r in rows if r.get("case") == "router_exact_nx0"}
    return cpu, router0


def add_deltas(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    cpu_by_lsearch, router0_by_lsearch = choose_baselines_by_lsearch(rows)
    out: list[dict[str, Any]] = []
    for row in rows:
        new: dict[str, Any] = dict(row)
        lsearch = row.get("lsearch", "")
        cpu = cpu_by_lsearch.get(lsearch)
        router0 = router0_by_lsearch.get(lsearch)
        for prefix, base in (("cpu", cpu), ("router0", router0)):
            if base is None:
                new[f"index_speedup_vs_{prefix}"] = ""
                new[f"group_speedup_vs_{prefix}"] = ""
                new[f"recall_delta_vs_{prefix}"] = ""
                continue
            index = to_float(row.get("index_ms"))
            group = to_float(row.get("group_ms"))
            recall = to_float(row.get("avg_recall"))
            base_index = to_float(base.get("index_ms"))
            base_group = to_float(base.get("group_ms"))
            base_recall = to_float(base.get("avg_recall"))
            new[f"index_speedup_vs_{prefix}"] = base_index / index if base_index and index else ""
            new[f"group_speedup_vs_{prefix}"] = base_group / group if base_group and group else ""
            new[f"recall_delta_vs_{prefix}"] = (
                recall - base_recall if recall is not None and base_recall is not None else ""
            )
        out.append(new)
    return out


def write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    fields = BASE_FIELDS + [
        "index_speedup_vs_cpu",
        "group_speedup_vs_cpu",
        "recall_delta_vs_cpu",
        "index_speedup_vs_router0",
        "group_speedup_vs_router0",
        "recall_delta_vs_router0",
        "search_summary",
        "index_log",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def make_markdown(rows: list[dict[str, Any]]) -> str:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[row.get("lsearch", "")].append(row)

    lines = ["# Mixed Exact/GNN Router A/B Summary", ""]
    if not rows:
        lines.append("No rows found.")
        return "\n".join(lines) + "\n"

    for lsearch in sorted(grouped, key=lambda x: (x == "NA", float(x) if x not in ("", "NA") else 1e30)):
        lines.append(f"## Lsearch={lsearch}")
        cols = [
            "case",
            "router_exact_nx",
            "index_ms",
            "group_ms",
            "avg_recall",
            "index_speedup_vs_cpu",
            "group_speedup_vs_cpu",
            "recall_delta_vs_cpu",
            "index_speedup_vs_router0",
            "group_speedup_vs_router0",
            "recall_delta_vs_router0",
        ]
        lines.append("| " + " | ".join(cols) + " |")
        lines.append("|" + "|".join(["---"] * len(cols)) + "|")
        for row in grouped[lsearch]:
            lines.append("| " + " | ".join(fmt(row.get(c, "")) for c in cols) + " |")
        lines.append("")

    lines.extend(
        [
            "Reviewer interpretation:",
            "",
            "- A router threshold is paper-usable only if it improves build/index time while recall remains close to both CPU Vamana and `router_exact_nx0`.",
            "- If recall drops toward the pure exact-kNN negative result, the threshold should be reported as a negative or partial ablation.",
            "- Empty/NA rows usually mean the input root came from `DRY_RUN=1`; they are useful for command validation but not for paper claims.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path, help="Output root from run_group_graph_router_ab.sh.")
    parser.add_argument("--csv", type=Path, help="CSV output path.")
    parser.add_argument("--md", type=Path, help="Markdown output path.")
    args = parser.parse_args()

    rows = add_deltas(read_rows(args.root))
    csv_path = args.csv or (args.root / "router_ab_summary.csv")
    md_path = args.md or (args.root / "router_ab_summary.md")
    write_csv(rows, csv_path)
    md = make_markdown(rows)
    md_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.write_text(md)
    print(md, end="")


if __name__ == "__main__":
    main()
