#!/usr/bin/env python3
"""Summarize multiple diagnose_ung_graph_structure.py outputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


DEFAULT_METRICS = [
    "total_edges",
    "intra_edges",
    "cross_edges",
    "intra_out_degree_avg",
    "intra_out_degree_p50",
    "intra_out_degree_p95",
    "zero_intra_ratio",
    "low_intra_le4_ratio",
    "group_largest_wcc_ratio_avg",
    "group_largest_wcc_ratio_p05",
    "groups_with_largest_wcc_lt_0_9",
]


def load_summary(path: Path) -> dict[str, Any]:
    if path.is_dir():
        path = path / "graph_structure_summary.json"
    return json.loads(path.read_text())


def fmt(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        if value == 0:
            return "0"
        if abs(value) >= 10000 or abs(value) < 0.001:
            return f"{value:.6g}"
        return f"{value:.6f}".rstrip("0").rstrip(".")
    return str(value)


def numeric_delta(value: Any, base: Any) -> str:
    if not isinstance(value, (int, float)) or not isinstance(base, (int, float)):
        return "-"
    return fmt(float(value) - float(base))


def parse_variant(spec: str) -> tuple[str, Path]:
    if "=" not in spec:
        path = Path(spec)
        return path.name, path
    name, raw_path = spec.split("=", 1)
    return name, Path(raw_path)


def make_markdown(rows: list[tuple[str, dict[str, Any]]], metrics: list[str]) -> str:
    base_name, base = rows[0]
    lines = []
    lines.append("| Metric | " + " | ".join(name for name, _ in rows) + " | " + " | ".join(f"{name}-{base_name}" for name, _ in rows[1:]) + " |")
    lines.append("|---" + "|---:" * (len(rows) + max(0, len(rows) - 1)) + "|")
    for metric in metrics:
        values = [summary.get(metric) for _, summary in rows]
        deltas = [numeric_delta(value, base.get(metric)) for value in values[1:]]
        lines.append("| " + metric + " | " + " | ".join(fmt(v) for v in values) + " | " + " | ".join(deltas) + " |")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("variants", nargs="+", help="Variant specs as name=/path/to/diag_dir or /path/to/diag_dir.")
    parser.add_argument("--metrics", default=",".join(DEFAULT_METRICS), help="Comma-separated metric names.")
    parser.add_argument("-o", "--output", type=Path, help="Optional Markdown output file.")
    args = parser.parse_args()

    metrics = [m.strip() for m in args.metrics.split(",") if m.strip()]
    rows = [(name, load_summary(path)) for name, path in map(parse_variant, args.variants)]
    md = make_markdown(rows, metrics)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(md)
    print(md, end="")


if __name__ == "__main__":
    main()
