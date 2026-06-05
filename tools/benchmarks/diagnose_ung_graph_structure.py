#!/usr/bin/env python3
"""Diagnose stored UNG graph structure.

The tool reads an index directory produced by build_UNG_index. It is designed
for reviewer-facing graph-quality checks: degree distributions, intra-group
edge coverage, weak connectivity, and optional reciprocal-edge coverage.
"""

from __future__ import annotations

import argparse
import csv
import json
from array import array
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean


class DSU:
    def __init__(self, n: int) -> None:
        self.parent = array("I", range(n))
        self.size = array("I", [1]) * n

    def find(self, x: int) -> int:
        parent = self.parent
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra = self.find(a)
        rb = self.find(b)
        if ra == rb:
            return
        if self.size[ra] < self.size[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        self.size[ra] += self.size[rb]


def percentile(values: list[int] | list[float], q: float) -> float:
    if not values:
        return 0.0
    xs = sorted(values)
    pos = (len(xs) - 1) * q
    lo = int(pos)
    hi = min(lo + 1, len(xs) - 1)
    frac = pos - lo
    return float(xs[lo] * (1.0 - frac) + xs[hi] * frac)


def read_meta(index_dir: Path) -> dict[str, str]:
    meta = {}
    path = index_dir / "meta"
    if not path.exists():
        return meta
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        if "=" in line:
            key, value = line.strip().split("=", 1)
            meta[key] = value
            continue
        parts = line.strip().split(maxsplit=1)
        if len(parts) == 2:
            meta[parts[0]] = parts[1]
    return meta


def read_ranges(path: Path) -> list[tuple[int, int]]:
    ranges: list[tuple[int, int]] = []
    with path.open() as f:
        for line in f:
            parts = line.split()
            if len(parts) >= 2:
                ranges.append((int(parts[0]), int(parts[1])))
            else:
                ranges.append((0, 0))
    return ranges


def build_group_lookup(ranges: list[tuple[int, int]], n: int) -> array:
    lookup = array("I", [0]) * n
    for gid, (begin, end) in enumerate(ranges):
        if end <= begin:
            continue
        for vid in range(begin, min(end, n)):
            lookup[vid] = gid
    return lookup


def parse_graph_line(line: str) -> tuple[int, list[int]]:
    parts = line.split()
    if not parts:
        return -1, []
    return int(parts[0]), [int(x) for x in parts[1:]]


def summarize_degrees(values: list[int], prefix: str) -> dict[str, float]:
    if not values:
        return {
            f"{prefix}_avg": 0.0,
            f"{prefix}_p50": 0.0,
            f"{prefix}_p95": 0.0,
            f"{prefix}_p99": 0.0,
            f"{prefix}_max": 0.0,
        }
    return {
        f"{prefix}_avg": float(mean(values)),
        f"{prefix}_p50": percentile(values, 0.50),
        f"{prefix}_p95": percentile(values, 0.95),
        f"{prefix}_p99": percentile(values, 0.99),
        f"{prefix}_max": float(max(values)),
    }


def diagnose(index_dir: Path, reciprocal_limit: int) -> tuple[dict[str, object], list[dict[str, object]]]:
    graph_path = index_dir / "graph"
    range_path = index_dir / "group_id_to_range"
    if not graph_path.exists():
        raise FileNotFoundError(f"missing graph file: {graph_path}")
    if not range_path.exists():
        raise FileNotFoundError(f"missing group_id_to_range file: {range_path}")

    meta = read_meta(index_dir)
    ranges = read_ranges(range_path)
    num_points = int(meta.get("num_points", "0"))
    if num_points <= 0:
        num_points = max((end for _, end in ranges), default=0)
    num_groups = int(meta.get("num_groups", str(max(0, len(ranges) - 1))))
    group_of = build_group_lookup(ranges, num_points)

    dsu = DSU(num_points)
    out_degree = [0] * num_points
    intra_degree = [0] * num_points
    cross_degree = [0] * num_points
    invalid_edges = 0
    self_edges = 0
    total_edges = 0
    intra_edges = 0
    cross_edges = 0
    reciprocal_edges: set[int] | None = set() if reciprocal_limit > 0 else None
    reciprocal_exact = reciprocal_limit > 0

    with graph_path.open() as f:
        for line in f:
            src, neigh = parse_graph_line(line)
            if src < 0 or src >= num_points:
                continue
            src_group = group_of[src]
            out_degree[src] = len(neigh)
            for dst in neigh:
                total_edges += 1
                if dst < 0 or dst >= num_points:
                    invalid_edges += 1
                    continue
                if dst == src:
                    self_edges += 1
                    continue
                if group_of[dst] == src_group:
                    intra_edges += 1
                    intra_degree[src] += 1
                    dsu.union(src, dst)
                    if reciprocal_edges is not None:
                        if len(reciprocal_edges) < reciprocal_limit:
                            reciprocal_edges.add(src * num_points + dst)
                        else:
                            reciprocal_edges = None
                            reciprocal_exact = False
                else:
                    cross_edges += 1
                    cross_degree[src] += 1

    reciprocal_pairs = 0
    reciprocal_ratio = None
    if reciprocal_edges is not None:
        for edge in reciprocal_edges:
            src = edge // num_points
            dst = edge - src * num_points
            if dst * num_points + src in reciprocal_edges:
                reciprocal_pairs += 1
        reciprocal_ratio = reciprocal_pairs / max(1, len(reciprocal_edges))

    group_component_counts: dict[int, Counter[int]] = defaultdict(Counter)
    for gid, (begin, end) in enumerate(ranges[: num_groups + 1]):
        if end <= begin:
            continue
        for vid in range(begin, min(end, num_points)):
            group_component_counts[gid][dsu.find(vid)] += 1

    group_rows: list[dict[str, object]] = []
    for gid, (begin, end) in enumerate(ranges[: num_groups + 1]):
        nx = max(0, min(end, num_points) - min(begin, num_points))
        if nx == 0:
            continue
        comps = group_component_counts.get(gid, Counter())
        largest = max(comps.values()) if comps else 0
        intra_vals = intra_degree[begin:end]
        cross_vals = cross_degree[begin:end]
        group_rows.append(
            {
                "group_id": gid,
                "nx": nx,
                "avg_intra_out_degree": mean(intra_vals) if intra_vals else 0.0,
                "avg_cross_out_degree": mean(cross_vals) if cross_vals else 0.0,
                "zero_intra_ratio": sum(1 for v in intra_vals if v == 0) / nx,
                "low_intra_le4_ratio": sum(1 for v in intra_vals if v <= 4) / nx,
                "weak_components": len(comps),
                "largest_weak_component_ratio": largest / nx,
            }
        )

    low_degree = sum(1 for v in intra_degree if v <= 4)
    zero_intra = sum(1 for v in intra_degree if v == 0)
    summary: dict[str, object] = {
        "index_dir": str(index_dir),
        "num_points": num_points,
        "num_groups": num_groups,
        "total_edges": total_edges,
        "intra_edges": intra_edges,
        "cross_edges": cross_edges,
        "invalid_edges": invalid_edges,
        "self_edges": self_edges,
        "intra_edge_ratio": intra_edges / max(1, total_edges),
        "cross_edge_ratio": cross_edges / max(1, total_edges),
        "zero_intra_ratio": zero_intra / max(1, num_points),
        "low_intra_le4_ratio": low_degree / max(1, num_points),
        "reciprocal_exact": reciprocal_exact,
        "reciprocal_intra_ratio": reciprocal_ratio,
    }
    summary.update(summarize_degrees(out_degree, "out_degree"))
    summary.update(summarize_degrees(intra_degree, "intra_out_degree"))
    summary.update(summarize_degrees(cross_degree, "cross_out_degree"))
    summary["group_largest_wcc_ratio_avg"] = (
        mean(float(r["largest_weak_component_ratio"]) for r in group_rows) if group_rows else 0.0
    )
    summary["group_largest_wcc_ratio_p05"] = percentile(
        [float(r["largest_weak_component_ratio"]) for r in group_rows], 0.05
    )
    summary["groups_with_largest_wcc_lt_0_9"] = sum(
        1 for r in group_rows if float(r["largest_weak_component_ratio"]) < 0.9
    )
    return summary, group_rows


def write_outputs(summary: dict[str, object], group_rows: list[dict[str, object]], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "graph_structure_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    if group_rows:
        with (out_dir / "graph_structure_groups.csv").open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(group_rows[0].keys()))
            writer.writeheader()
            writer.writerows(group_rows)
    with (out_dir / "graph_structure_summary.md").open("w") as f:
        f.write("# UNG Graph Structure Diagnostics\n\n")
        f.write(f"- index: `{summary['index_dir']}`\n")
        f.write(f"- points/groups: `{summary['num_points']}` / `{summary['num_groups']}`\n")
        f.write(f"- edges intra/cross/total: `{summary['intra_edges']}` / `{summary['cross_edges']}` / `{summary['total_edges']}`\n")
        f.write(f"- intra edge ratio: `{float(summary['intra_edge_ratio']):.6f}`\n")
        f.write(f"- zero intra-degree ratio: `{float(summary['zero_intra_ratio']):.6f}`\n")
        f.write(f"- low intra-degree <=4 ratio: `{float(summary['low_intra_le4_ratio']):.6f}`\n")
        rec = summary["reciprocal_intra_ratio"]
        rec_text = "skipped" if rec is None else f"{float(rec):.6f}"
        f.write(f"- reciprocal intra-edge ratio: `{rec_text}`\n")
        f.write(f"- avg largest weak component ratio per nonempty group: `{float(summary['group_largest_wcc_ratio_avg']):.6f}`\n")
        f.write(f"- groups with largest WCC < 0.9: `{summary['groups_with_largest_wcc_lt_0_9']}`\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("index_dir", type=Path, help="Index directory containing graph and group_id_to_range.")
    parser.add_argument("--out", type=Path, required=True, help="Output directory for JSON/CSV/Markdown diagnostics.")
    parser.add_argument(
        "--reciprocal-limit",
        type=int,
        default=20_000_000,
        help="Maximum intra edges to keep for exact reciprocal coverage; set 0 to skip.",
    )
    args = parser.parse_args()
    summary, group_rows = diagnose(args.index_dir, args.reciprocal_limit)
    write_outputs(summary, group_rows, args.out)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
