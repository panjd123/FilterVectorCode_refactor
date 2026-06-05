#!/usr/bin/env python3
"""Summarize cross-edge baseline runs for paper fairness tables."""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path
from typing import Any


GPU_EVENT_RE = re.compile(
    r"\[GPU GEMM\]\s+H2D\(ms\)=(?P<h2d>[0-9.]+)\s+Kernel\(ms\)=(?P<kernel>[0-9.]+)\s+D2H\(ms\)=(?P<d2h>[0-9.]+)"
)
PREPARE_RE = re.compile(r"prepare_all[^0-9]*(?P<value>[0-9.]+)\s*ms", re.IGNORECASE)
TARGET_RE = re.compile(r"target_groups=(?P<value>[0-9]+)")


def read_csv_kv(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    with path.open() as f:
        return {row[0]: row[1] for row in csv.reader(f) if len(row) >= 2 and row[0] != "Index Name"}


def read_meta(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    out: dict[str, str] = {}
    for line in path.read_text(errors="ignore").splitlines():
        if "=" in line:
            k, v = line.split("=", 1)
            out[k] = v
    return out


def parse_log(path: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    if not path.exists():
        return out
    text = path.read_text(errors="ignore")
    match = GPU_EVENT_RE.search(text)
    if match:
        out["gpu_h2d_ms"] = match.group("h2d")
        out["gpu_kernel_ms"] = match.group("kernel")
        out["gpu_d2h_ms"] = match.group("d2h")
    target = TARGET_RE.search(text)
    if target:
        out["target_groups"] = target.group("value")
    prepare_values = [float(m.group("value")) for m in PREPARE_RE.finditer(text)]
    if prepare_values:
        out["prepare_all_ms"] = f"{max(prepare_values):.6g}"
    return out


def fmt_num(value: str) -> str:
    if value == "":
        return ""
    try:
        v = float(value)
    except ValueError:
        return value
    if v == 0:
        return "0"
    if abs(v) >= 10000:
        return f"{v:.1f}"
    if abs(v) >= 1000:
        return f"{v:.2f}"
    return f"{v:.3f}".rstrip("0").rstrip(".")


def parse_variant(spec: str) -> tuple[str, Path]:
    if "=" not in spec:
        path = Path(spec)
        return path.name, path
    name, raw = spec.split("=", 1)
    return name, Path(raw)


def summarize_variant(name: str, run_dir: Path) -> dict[str, Any]:
    build = read_csv_kv(run_dir / "results" / "build_time.csv")
    meta = read_meta(run_dir / "index" / "meta")
    if not meta:
        meta = read_meta(run_dir / "index_files" / "meta")
    log = parse_log(run_dir / "run.log")
    row: dict[str, Any] = {
        "variant": name,
        "run_dir": str(run_dir),
        "index_ms": build.get("index_time", meta.get("index_time_add_rb(ms)", "")),
        "group_ms": build.get("build_graph_time", meta.get("build_graph_time(ms)", "")),
        "cross_ms": build.get("build_cross_edges_time", meta.get("build_cross_edges_time(ms)", "")),
        "threads": meta.get("build_num_threads", ""),
        "group_graph_impl": meta.get("group_graph_impl", ""),
        "cross_edge_impl": meta.get("cross_edge_impl", ""),
        "gpu_topk_impl": meta.get("gpu_topk_impl", ""),
        "target_groups": log.get("target_groups", ""),
        "prepare_all_ms": log.get("prepare_all_ms", ""),
        "gpu_h2d_ms": log.get("gpu_h2d_ms", ""),
        "gpu_kernel_ms": log.get("gpu_kernel_ms", ""),
        "gpu_d2h_ms": log.get("gpu_d2h_ms", ""),
        "source": str(run_dir / "run.log"),
    }
    return row


def write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    fields = [
        "variant",
        "index_ms",
        "group_ms",
        "cross_ms",
        "threads",
        "group_graph_impl",
        "cross_edge_impl",
        "gpu_topk_impl",
        "target_groups",
        "prepare_all_ms",
        "gpu_h2d_ms",
        "gpu_kernel_ms",
        "gpu_d2h_ms",
        "run_dir",
        "source",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def markdown(rows: list[dict[str, Any]], baseline: str | None) -> str:
    baseline_cross = None
    if baseline:
        for row in rows:
            if row["variant"] == baseline and row.get("cross_ms"):
                baseline_cross = float(row["cross_ms"])
                break
    lines = []
    header = [
        "Variant",
        "Index ms",
        "Group ms",
        "Cross ms",
        "Speedup vs baseline",
        "Threads",
        "Cross impl",
        "TopK impl",
        "GPU H2D",
        "GPU kernel",
        "GPU D2H",
    ]
    lines.append("| " + " | ".join(header) + " |")
    lines.append("|---|---:|---:|---:|---:|---:|---|---|---:|---:|---:|")
    for row in rows:
        speedup = ""
        if baseline_cross and row.get("cross_ms"):
            speedup = f"{baseline_cross / float(row['cross_ms']):.2f}x"
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["variant"]),
                    fmt_num(str(row.get("index_ms", ""))),
                    fmt_num(str(row.get("group_ms", ""))),
                    fmt_num(str(row.get("cross_ms", ""))),
                    speedup,
                    str(row.get("threads", "")),
                    str(row.get("cross_edge_impl", "")),
                    str(row.get("gpu_topk_impl", "")),
                    fmt_num(str(row.get("gpu_h2d_ms", ""))),
                    fmt_num(str(row.get("gpu_kernel_ms", ""))),
                    fmt_num(str(row.get("gpu_d2h_ms", ""))),
                ]
            )
            + " |"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("variants", nargs="+", help="name=/path/to/run_dir entries")
    parser.add_argument("--baseline", default="", help="Variant name used for speedup denominator")
    parser.add_argument("--csv", type=Path, help="Optional CSV output path")
    parser.add_argument("--md", type=Path, help="Optional Markdown output path")
    args = parser.parse_args()

    rows = [summarize_variant(name, path) for name, path in map(parse_variant, args.variants)]
    if args.csv:
        write_csv(rows, args.csv)
    md = markdown(rows, args.baseline or None)
    if args.md:
        args.md.parent.mkdir(parents=True, exist_ok=True)
        args.md.write_text(md)
    print(md, end="")


if __name__ == "__main__":
    main()
