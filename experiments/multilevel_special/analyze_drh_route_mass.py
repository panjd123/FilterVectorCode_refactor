#!/usr/bin/env python3
"""Audit DRH routing mass directly from immutable block/query inputs."""

from __future__ import annotations

import argparse
import csv
import json
import math
import struct
from pathlib import Path
from typing import Any


SPECIAL_BLOCK_MAGIC_V3 = b"SBLK003\0"


def read_u32_array(stream: Any, count: int) -> tuple[int, ...]:
    payload = stream.read(4 * count)
    if len(payload) != 4 * count:
        raise ValueError("truncated special-block metadata array")
    return struct.unpack(f"<{count}I", payload) if count else ()


def read_upper_blocks(path: Path) -> list[dict[str, Any]]:
    with path.open("rb") as stream:
        magic = stream.read(8)
        header = stream.read(16)
        if len(magic) != 8 or len(header) != 16:
            raise ValueError("truncated special-block metadata header")
        version, reserved, count = struct.unpack("<IIQ", header)
        if (magic != SPECIAL_BLOCK_MAGIC_V3 or version != 3 or reserved != 0 or
                count > 2 ** 32 - 1):
            raise ValueError(
                f"unsupported special-block metadata: magic={magic!r}, version={version}")
        blocks = []
        for expected_id in range(1, count + 1):
            fixed_payload = stream.read(44)
            if len(fixed_payload) != 44:
                raise ValueError("truncated special-block metadata record")
            fixed = struct.unpack("<11I", fixed_payload)
            if fixed[0] != expected_id:
                raise ValueError("non-contiguous special-block identifiers")
            root_labels = read_u32_array(stream, fixed[7])
            if tuple(sorted(root_labels)) != root_labels:
                raise ValueError("special-block root labels are not sorted")
            read_u32_array(stream, fixed[8])
            read_u32_array(stream, fixed[9])
            read_u32_array(stream, fixed[10])
            if fixed[1] > 0:
                blocks.append({
                    "block_id": fixed[0],
                    "level": fixed[1],
                    "point_count": fixed[5],
                    "root_labels": root_labels,
                })
        if stream.read(1):
            raise ValueError("trailing bytes in special-block metadata")
    return blocks


def parse_query_label_row(line: str) -> tuple[int, ...]:
    """Parse the canonical comma format used by Storage::load_from_file."""
    if not line:
        return ()
    return tuple(sorted(int(value) for value in line.split(",")))


def parse_profile_label_row(line: str) -> tuple[int, ...]:
    """Parse the profiler's human-readable space-separated label column."""
    return tuple(sorted(int(value) for value in line.replace(",", " ").split()))


def read_queries(path: Path) -> list[tuple[int, ...]]:
    return [parse_query_label_row(line)
            for line in path.read_text(encoding="utf-8").splitlines()]


def read_coverage_profile(path: Path) -> list[tuple[tuple[int, ...], int]]:
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    required = {"coverage_count", "labels"}
    if not rows or not required <= set(rows[0]):
        raise ValueError(f"coverage profile must contain {sorted(required)}: {path}")
    result = []
    for index, row in enumerate(rows):
        coverage = int(row["coverage_count"])
        if coverage < 0:
            raise ValueError(f"negative coverage at row {index}: {path}")
        result.append((parse_profile_label_row(row["labels"]), coverage))
    return result


def sorted_includes(superset: tuple[int, ...], subset: tuple[int, ...]) -> bool:
    """Match C++ std::includes, including duplicate-label multiplicity."""
    outer = 0
    for value in subset:
        while outer < len(superset) and superset[outer] < value:
            outer += 1
        if outer == len(superset) or superset[outer] != value:
            return False
        outer += 1
    return True


def nearest_rank(values: list[int], fraction: float) -> int:
    if not values:
        return 0
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, math.ceil(fraction * len(ordered)) - 1))
    return ordered[index]


def route_masses(
    blocks: list[dict[str, Any]], queries: list[tuple[int, ...]],
) -> list[dict[str, int]]:
    results = []
    for query in queries:
        authorized = [
            block for block in blocks
            if sorted_includes(block["root_labels"], query)
        ]
        if not authorized:
            results.append({"level": 0, "block_count": 0, "direct_points": 0})
            continue
        highest = max(int(block["level"]) for block in authorized)
        highest_blocks = [block for block in authorized if int(block["level"]) == highest]
        results.append({
            "level": highest,
            "block_count": len(highest_blocks),
            "direct_points": sum(int(block["point_count"]) for block in highest_blocks),
        })
    return results


def summarize(
    masses: list[dict[str, int]], threshold: int,
    queries: list[tuple[int, ...]] | None = None,
    coverage_profile: list[tuple[tuple[int, ...], int]] | None = None,
) -> dict[str, Any]:
    if threshold <= 0:
        raise ValueError("threshold must be positive")
    if not masses:
        raise ValueError("at least one query is required")
    direct = [row["direct_points"] for row in masses]
    authorized = [row for row in masses if row["block_count"] > 0]
    enabled = [row for row in masses if row["direct_points"] >= threshold]
    result = {
        "query_count": len(masses),
        "threshold": threshold,
        "nonempty_authorization_count": len(authorized),
        "nonempty_authorization_rate": len(authorized) / len(masses),
        "mass_gate_enabled_count": len(enabled),
        "mass_gate_enabled_rate": len(enabled) / len(masses),
        "direct_mass_p50": nearest_rank(direct, 0.50),
        "direct_mass_p95": nearest_rank(direct, 0.95),
        "direct_mass_max": max(direct),
    }
    if coverage_profile is None:
        result["eligible_mass_invariant"] = "not_checked"
        return result
    if queries is None or len(queries) != len(masses) or len(coverage_profile) != len(masses):
        raise ValueError("query, route-mass, and coverage-profile row counts differ")
    coverages = []
    fractions = []
    for index, (query, mass, (profile_labels, coverage)) in enumerate(
            zip(queries, masses, coverage_profile)):
        if profile_labels != query:
            raise ValueError(
                f"query/profile label mismatch at row {index}: "
                f"{query!r} != {profile_labels!r}")
        if mass["direct_points"] > coverage:
            raise ValueError(
                f"authorized direct mass exceeds eligible mass at row {index}: "
                f"{mass['direct_points']} > {coverage}")
        coverages.append(coverage)
        fractions.append(mass["direct_points"] / coverage if coverage else 0.0)
    result.update({
        "eligible_mass_invariant": "checked_and_holds",
        "eligible_mass_checked_queries": len(coverages),
        "eligible_mass_p50": nearest_rank(coverages, 0.50),
        "eligible_mass_p95": nearest_rank(coverages, 0.95),
        "eligible_mass_max": max(coverages),
        "direct_to_eligible_fraction_max": max(fractions),
    })
    return result


def unique_coverage_profile(query_root: Path) -> Path | None:
    profiles = sorted(query_root.glob("profiled_*.csv"))
    if len(profiles) > 1:
        raise ValueError(f"expected at most one coverage profile in {query_root}")
    return profiles[0] if profiles else None


def analyze_config(config_path: Path, method_name: str,
                   threshold: int) -> list[dict[str, Any]]:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    methods = [method for method in config["methods"] if method["name"] == method_name]
    if len(methods) != 1:
        raise ValueError(f"expected one method named {method_name}, got {len(methods)}")
    method = methods[0]
    block_metadata = Path(method["block_index"]) / "special_blocks.bin"
    blocks = read_upper_blocks(block_metadata)
    rows = []
    for workload in config["workloads"]:
        query_root = Path(config["data_root"]) / str(workload["query_dir"])
        query_labels = query_root / f"{config['dataset']}_query_labels.txt"
        queries = read_queries(query_labels)
        coverage_path = unique_coverage_profile(query_root)
        coverage = read_coverage_profile(coverage_path) if coverage_path else None
        row = summarize(route_masses(blocks, queries), threshold, queries, coverage)
        row.update({
            "dataset": config["dataset"],
            "workload": workload["name"],
            "mean_selectivity": workload["mean_selectivity"],
            "block_metadata": str(block_metadata.resolve()),
            "query_label_file": str(query_labels.resolve()),
            "coverage_profile": str(coverage_path.resolve()) if coverage_path else None,
            "upper_block_count": len(blocks),
        })
        rows.append(row)
    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--block-metadata", type=Path)
    parser.add_argument("--query-label-file", type=Path)
    parser.add_argument("--coverage-profile", type=Path)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--method")
    parser.add_argument("--threshold", type=int, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.config:
        if (not args.method or args.block_metadata or args.query_label_file or
                args.coverage_profile):
            parser.error("--config requires --method and excludes direct input paths")
        result: Any = analyze_config(args.config, args.method, args.threshold)
    else:
        if not args.block_metadata or not args.query_label_file or args.method:
            parser.error("direct mode requires --block-metadata and --query-label-file")
        blocks = read_upper_blocks(args.block_metadata)
        queries = read_queries(args.query_label_file)
        coverage = (read_coverage_profile(args.coverage_profile)
                    if args.coverage_profile else None)
        result = summarize(
            route_masses(blocks, queries), args.threshold, queries, coverage)
        result.update({
            "block_metadata": str(args.block_metadata.resolve()),
            "query_label_file": str(args.query_label_file.resolve()),
            "coverage_profile": (str(args.coverage_profile.resolve())
                                 if args.coverage_profile else None),
            "upper_block_count": len(blocks),
        })
    payload = json.dumps(result, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload, encoding="utf-8")
    print(payload, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
