#!/usr/bin/env python3
"""Derive query-free Special Block layer candidates from static trie mass."""

from __future__ import annotations

import argparse
import csv
import json
import math
import struct
from collections import Counter
from pathlib import Path


HEADER = struct.Struct("<8sIIQQIIQ")
NODE = struct.Struct("<IIIIII")


def load_trie(path: Path):
    with path.open("rb") as stream:
        raw = stream.read(HEADER.size)
        magic, version, header_bytes, node_count, child_count, groups, _, _ = HEADER.unpack(raw)
        if magic != b"SBTRIE1\0" or version != 1 or header_bytes != HEADER.size:
            raise ValueError("unsupported trie file")
        nodes = [NODE.unpack(stream.read(NODE.size)) for _ in range(node_count)]
        children = list(struct.unpack(f"<{child_count}I", stream.read(child_count * 4)))
        if stream.read(1):
            raise ValueError("trie has trailing bytes")
    return nodes, children, groups


def group_counts(labels: Path) -> list[int]:
    counts = Counter(line.strip() for line in labels.open())
    return [0] + list(counts.values())


def partition(nodes, children, points, threshold: int):
    uncovered = [0] * len(nodes)
    block_roots = []
    for node_id in range(len(nodes) - 1, -1, -1):
        _, _, first_child, child_count, terminal_group_id, _ = nodes[node_id]
        mass = points[terminal_group_id] if terminal_group_id else 0
        for child_id in children[first_child:first_child + child_count]:
            mass += uncovered[child_id]
        uncovered[node_id] = mass
        if node_id != 0 and mass > threshold:
            block_roots.append((node_id, mass))
            uncovered[node_id] = 0
    return block_roots, uncovered[0]


def partition_row(nodes, children, points, total: int, threshold: int):
    roots, residual = partition(nodes, children, points, threshold)
    masses = sorted(mass for _, mass in roots)
    covered = sum(masses)
    quantile = lambda q: masses[round(q * (len(masses) - 1))] if masses else 0
    return {
        "threshold": threshold, "threshold_fraction": threshold / total,
        "block_count": len(roots), "covered_points": covered,
        "covered_fraction": covered / total, "root_residual_points": residual,
        "min_block_mass": masses[0] if masses else 0,
        "p50_block_mass": quantile(.5), "p90_block_mass": quantile(.9),
        "max_block_mass": masses[-1] if masses else 0,
    }


def next_power_of_two(value: float) -> int:
    return 1 << math.ceil(math.log2(max(1.0, value)))


def derive_mass_ladder(nodes, children, points, total: int, max_degree: int,
                       build_width: int, cross_edges: int, max_levels: int) -> dict:
    """Choose 0..max_levels without query traces or latency measurements.

    The base scale is the first dyadic mass at least M*Lbuild.  The scale ratio
    is the next power of two at least M/C, where C is the per-block cross-edge
    budget.  Levels are emitted until the deterministic trie partition is empty.
    """
    if max_degree <= 1:
        raise ValueError("max degree must exceed one")
    if build_width <= 0:
        raise ValueError("build width must be positive")
    if cross_edges <= 0:
        raise ValueError("cross-edge budget must be positive")
    base = next_power_of_two(max_degree * build_width)
    scale_ratio = next_power_of_two(max(2.0, max_degree / cross_edges))
    levels = []
    decisions = []
    threshold = base
    for level in range(1, max_levels + 1):
        row = partition_row(nodes, children, points, total, threshold)
        if row["block_count"] == 0:
            decisions.append({"level": level, "threshold": threshold,
                              "outcome": "stop",
                              "reason": "trie partition is empty"})
            break
        row["level"] = level
        levels.append(row)
        decisions.append({"level": level, "threshold": threshold,
                          "outcome": "add_level",
                          "block_count": row["block_count"]})
        threshold *= scale_ratio
    return {
        "name": "query_free_mass_ladder",
        "max_degree": max_degree,
        "build_width": build_width,
        "cross_edges": cross_edges,
        "base_rule": "T1 = next_power_of_two(M * Lbuild)",
        "scale_ratio_rule": "rho = next_power_of_two(max(2, M / C))",
        "stop_rule": "stop before the first threshold whose trie partition has zero blocks",
        "scale_ratio": scale_ratio,
        "layer_count": len(levels),
        "thresholds": [row["threshold"] for row in levels],
        "levels": levels,
        "decisions": decisions,
    }


def knees(rows):
    """Return maximum-curvature points in log threshold/log block-count space."""
    usable = [row for row in rows if row["block_count"] > 0]
    result = []
    for index in range(1, len(usable) - 1):
        a, b, c = usable[index - 1:index + 2]
        x1, y1 = math.log(a["threshold"]), math.log(a["block_count"])
        x2, y2 = math.log(b["threshold"]), math.log(b["block_count"])
        x3, y3 = math.log(c["threshold"]), math.log(c["block_count"])
        left = (y2 - y1) / (x2 - x1)
        right = (y3 - y2) / (x3 - x2)
        result.append((abs(right - left), b["threshold"], left, right))
    return sorted(result, reverse=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trie", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--thresholds", default="250,500,1000,2000,4000,8000,16000,32000,64000,128000,256000,400000,600000")
    parser.add_argument("--max-degree", type=int, default=64)
    parser.add_argument("--build-width", type=int, default=100)
    parser.add_argument("--cross-edges", type=int, default=4)
    parser.add_argument("--max-levels", type=int, default=64,
                        help="safety cap for static analysis; current index format materializes at most two levels")
    args = parser.parse_args()
    nodes, children, groups = load_trie(args.trie)
    points = group_counts(args.labels)
    if len(points) != groups + 1:
        raise ValueError(f"group count mismatch: {len(points)-1} vs {groups}")
    total = sum(points)
    rows = []
    for threshold in map(int, args.thresholds.split(",")):
        rows.append(partition_row(nodes, children, points, total, threshold))
    result = {
        "schema_version": 1, "policy_inputs": "static trie and group cardinalities only",
        "num_points": total, "num_groups": groups, "trie_nodes": len(nodes),
        "thresholds": rows,
        "curvature_knees": [
            {"score": score, "threshold": threshold, "left_slope": left,
             "right_slope": right}
            for score, threshold, left, right in knees(rows)
        ],
        "automatic_policy": derive_mass_ladder(
            nodes, children, points, total, args.max_degree, args.build_width,
            args.cross_edges, args.max_levels),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    with args.output.with_suffix(".csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0], lineterminator="\n")
        writer.writeheader(); writer.writerows(rows)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
