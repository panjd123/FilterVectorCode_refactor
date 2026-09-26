#!/usr/bin/env python3
"""Derive a query-independent hierarchy plan from index scale parameters."""

from __future__ import annotations

import argparse
import json
import math
import struct
from pathlib import Path


def nearest_power_of_two(value: float) -> int:
    if value <= 1:
        return 1
    exponent = math.floor(math.log2(value) + 0.5)
    return 1 << exponent


def derive_plan(num_points: int, max_degree: int,
                num_cross_edges: int) -> list[dict[str, int | str]]:
    """Return the Degree-Ratio Hierarchy (DRH) plan.

    sqrt(N) balances a simple within-block plus between-block cost proxy,
    T + N/T. Successive scales use the already configured ratio between
    intra-layer degree and cross-block degree. A scale is materialized only
    while its estimated block population N/T can support at least C choices.
    """
    if num_points <= 0 or max_degree <= 0 or num_cross_edges <= 0:
        raise ValueError("num_points, max_degree, and num_cross_edges must be positive")
    ratio = max(2, round(max_degree / num_cross_edges))
    threshold = nearest_power_of_two(math.sqrt(num_points))
    layers: list[dict[str, int | str]] = []
    while num_points / threshold >= num_cross_edges:
        estimated_blocks = num_points / threshold
        topology = "lng" if estimated_blocks > max_degree else "trie"
        layers.append({
            "min_points": threshold,
            "topology": topology,
            "estimated_blocks": round(estimated_blocks),
        })
        if threshold > (2 ** 30) // ratio:
            break
        threshold *= ratio
        if len(layers) >= 254:
            break
    return layers


def num_points_from_meta(path: Path) -> int:
    for line in path.read_text().splitlines():
        if line.startswith("num_points="):
            return int(line.split("=", 1)[1])
    raise ValueError(f"num_points is absent from {path}")


def num_points_from_bin(path: Path) -> int:
    with path.open("rb") as stream:
        header = stream.read(4)
    if len(header) != 4:
        raise ValueError(f"invalid vector binary: {path}")
    return struct.unpack("<I", header)[0]


def main() -> int:
    parser = argparse.ArgumentParser()
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--num-points", type=int)
    source.add_argument("--meta", type=Path)
    source.add_argument("--base-bin", type=Path)
    parser.add_argument("--max-degree", type=int, default=64)
    parser.add_argument("--num-cross-edges", type=int, default=4)
    parser.add_argument("--encoded", action="store_true")
    args = parser.parse_args()
    if args.num_points is not None:
        num_points = args.num_points
    elif args.meta is not None:
        num_points = num_points_from_meta(args.meta)
    else:
        num_points = num_points_from_bin(args.base_bin)
    layers = derive_plan(num_points, args.max_degree, args.num_cross_edges)
    if args.encoded:
        print(",".join(f"{row['min_points']}:{row['topology']}" for row in layers))
    else:
        print(json.dumps({
            "method": "degree_ratio_hierarchy_v1",
            "query_calibrated": False,
            "num_points": num_points,
            "max_degree": args.max_degree,
            "num_cross_edges": args.num_cross_edges,
            "scale_ratio": max(2, round(args.max_degree / args.num_cross_edges)),
            "hierarchy_layers": layers,
        }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
