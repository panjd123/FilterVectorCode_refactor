#!/usr/bin/env python3
"""Summarize authoritative base and hierarchy build measurements."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path

import experiment_core


HERE = Path(__file__).resolve().parent


def directory_bytes(path: Path) -> int:
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def percentile(values: list[float], fraction: float) -> float:
    return experiment_core.percentile(values, fraction)


def read_metric_csv(path: Path) -> dict[str, float]:
    with path.open(newline="") as stream:
        rows = list(csv.reader(stream))
    return {row[0]: float(row[1]) for row in rows[1:] if len(row) >= 2}


def load_rows(config_path: Path, component: str) -> list[dict]:
    config = json.loads(config_path.read_text())
    root = Path(config["output_root"])
    manifest = json.loads((root / "manifest.json").read_text())
    records = {row["name"]: row for row in manifest["runs"]}
    rows = []
    for case in config["cases"]:
        record = records.get(case["name"])
        if record is None or record.get("status") != "complete":
            continue
        case_root = Path(record["case_root"])
        index_dir = case_root / ("index_files" if component == "base" else "block_index")
        timing = read_metric_csv(case_root / "results/build_time.csv")
        internal_key = "index_time" if component == "base" else "total_time"
        row = {
            "component": component,
            "case": case["name"],
            "structure": case.get("structure", "zero_layer"),
            "profile": case["benchmark_profile"],
            "timing_role": case["timing_role"],
            "repeat": int(case["repeat"]),
            "wall_seconds": float(record["elapsed_seconds"]),
            "internal_seconds": timing[internal_key] / 1000.0,
            "index_bytes": directory_bytes(index_dir),
            "case_root": str(case_root),
        }
        for key, value in timing.items():
            row[f"stage_{key}"] = value
        resource_path = case_root / "resource_usage.json"
        if resource_path.is_file():
            resource = json.loads(resource_path.read_text())
            row["peak_rss_mib"] = float(
                resource["peak_process_tree_rss_kib"]) / 1024.0
            row["peak_gpu_memory_mib"] = float(resource["peak_gpu_memory_mib"])
        rows.append(row)
    return rows


def summarize(rows: list[dict]) -> list[dict]:
    grouped = defaultdict(list)
    resources = defaultdict(list)
    for row in rows:
        key = (row["component"], row["structure"], row["profile"])
        if row["timing_role"] == "measured":
            grouped[key].append(row)
        if "peak_rss_mib" in row:
            resources[key].append(row)

    result = []
    for key, values in sorted(grouped.items()):
        wall = [row["wall_seconds"] for row in values]
        internal = [row["internal_seconds"] for row in values]
        resource = resources.get(key, [])
        mean = statistics.mean(wall)
        result.append({
            "component": key[0],
            "structure": key[1],
            "profile": key[2],
            "measured_repeats": len(values),
            "wall_median_seconds": statistics.median(wall),
            "wall_p95_seconds": percentile(wall, 0.95),
            "wall_cv": statistics.stdev(wall) / mean if len(wall) > 1 else 0.0,
            "internal_median_seconds": statistics.median(internal),
            "wall_minus_internal_median_seconds": statistics.median(
                row["wall_seconds"] - row["internal_seconds"] for row in values),
            "index_median_mib": statistics.median(
                row["index_bytes"] / (1024.0 * 1024.0) for row in values),
            "peak_rss_mib": statistics.median(
                row["peak_rss_mib"] for row in resource) if resource else "",
            "peak_gpu_memory_mib": statistics.median(
                row["peak_gpu_memory_mib"] for row in resource) if resource else "",
        })

    by_key = {(row["component"], row["structure"], row["profile"]): row
              for row in result}
    for row in result:
        baseline_profile = "original_cpu" if row["component"] == "base" else "cpu"
        baseline = by_key.get((row["component"], row["structure"], baseline_profile))
        if baseline:
            row["speedup_vs_component_cpu"] = (
                baseline["wall_median_seconds"] / row["wall_median_seconds"])
    return result


def indexed_measured(rows: list[dict], component: str, structure: str,
                     profile: str) -> dict[int, float]:
    return {
        row["repeat"]: row["wall_seconds"] for row in rows
        if row["component"] == component and row["structure"] == structure
        and row["profile"] == profile and row["timing_role"] == "measured"
    }


def summarize_end_to_end(rows: list[dict]) -> list[dict]:
    original = indexed_measured(rows, "base", "zero_layer", "original_cpu")
    accelerated = indexed_measured(rows, "base", "zero_layer", "accelerated_gpu")
    hierarchy_keys = sorted({(row["structure"], row["profile"]) for row in rows
                             if row["component"] == "hierarchy"
                             and row["timing_role"] == "measured"})
    result = []
    for structure, profile in hierarchy_keys:
        hierarchy = indexed_measured(rows, "hierarchy", structure, profile)
        repeats = sorted(set(original) & set(accelerated) & set(hierarchy))
        if not repeats:
            continue
        original_values = [original[index] for index in repeats]
        total_values = [accelerated[index] + hierarchy[index] for index in repeats]
        overhead_values = [total_values[pos] / accelerated[index]
                           for pos, index in enumerate(repeats)]
        ratio = statistics.median(original_values) / statistics.median(total_values)
        lo, hi = experiment_core.bootstrap_median_ratio(
            original_values, total_values,
            seed=20260925 + sum(ord(ch) for ch in structure + profile),
            samples=20000, paired=True)
        result.append({
            "structure": structure,
            "hierarchy_profile": profile,
            "paired_repeats": len(repeats),
            "original_cpu_base_median_seconds": statistics.median(original_values),
            "accelerated_base_plus_hierarchy_median_seconds": statistics.median(total_values),
            "speedup_vs_original_cpu": ratio,
            "speedup_ci95_low": lo,
            "speedup_ci95_high": hi,
            "overhead_vs_accelerated_base_median": statistics.median(overhead_values),
            "no_slower_supported": lo >= 1.0,
        })
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-timing", type=Path, default=HERE / "config.authoritative_amazon_base_build_timing.json")
    parser.add_argument("--base-resource", type=Path, default=HERE / "config.authoritative_amazon_base_build_resource.json")
    parser.add_argument("--hierarchy-timing", type=Path, default=HERE / "config.authoritative_amazon_hierarchy_build_timing.json")
    parser.add_argument("--hierarchy-resource", type=Path, default=HERE / "config.authoritative_amazon_hierarchy_build_resource.json")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    rows = []
    for path, component in ((args.base_timing, "base"),
                            (args.base_resource, "base"),
                            (args.hierarchy_timing, "hierarchy"),
                            (args.hierarchy_resource, "hierarchy")):
        rows.extend(load_rows(path, component))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    experiment_core.write_csv(args.output_dir / "build_measurements.csv", rows)
    experiment_core.write_csv(args.output_dir / "build_summary.csv", summarize(rows))
    experiment_core.write_csv(
        args.output_dir / "build_end_to_end.csv", summarize_end_to_end(rows))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
