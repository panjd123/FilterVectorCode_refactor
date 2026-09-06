#!/usr/bin/env python3
"""Generate 7-repeat formal reruns from the completed coarse sweep."""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

import select_layer_tuning


def dense_crossing_grid(rows: list[dict], threshold: float) -> list[int]:
    """Bracket the measured crossing and add real integer L points nearby."""
    ordered = sorted(rows, key=lambda row: row["lsearch"])
    feasible = [row for row in ordered if row["recall_min"] >= threshold]
    if not feasible:
        return sorted({row["lsearch"] for row in ordered[-2:]})
    high = min(feasible, key=lambda row: row["lsearch"])["lsearch"]
    lower = [row["lsearch"] for row in ordered if row["lsearch"] < high]
    low = max(lower) if lower else max(1, high // 2)
    if low == high:
        return [high]
    # Include endpoints plus quartiles; all are measured in the formal run.
    return sorted({low, high,
                   int(round(low + (high - low) * 0.25)),
                   int(round(low + (high - low) * 0.50)),
                   int(round(low + (high - low) * 0.75))})


def top_structures(points: list[dict], thresholds: dict[str, float],
                   shared_top_k: int = 3, oracle_top_k: int = 2) -> set[tuple[int, str]]:
    workloads = sorted(thresholds)
    selected: set[tuple[int, str]] = set()
    layers = sorted({row["layer_count"] for row in points})
    for layer in layers:
        methods = sorted({row["method"] for row in points if row["layer_count"] == layer})
        ranked_shared = []
        for method in methods:
            ratios = []
            feasible = True
            for workload in workloads:
                own = select_layer_tuning.fastest_feasible(
                    [row for row in points if row["method"] == method and row["workload"] == workload],
                    thresholds[workload])
                base = select_layer_tuning.fastest_feasible(
                    [row for row in points if row["layer_count"] == 0 and row["workload"] == workload],
                    thresholds[workload])
                if own is None or base is None:
                    feasible = False
                    break
                ratios.append(own["batch_ms_warm_median"] / base["batch_ms_warm_median"])
            if feasible:
                score = math.exp(sum(math.log(value) for value in ratios) / len(ratios))
                ranked_shared.append((score, method))
        selected.update((layer, method) for _, method in sorted(ranked_shared)[:shared_top_k])
        for workload in workloads:
            ranked_oracle = []
            for method in methods:
                best = select_layer_tuning.fastest_feasible(
                    [row for row in points if row["method"] == method and row["workload"] == workload],
                    thresholds[workload])
                if best is not None:
                    ranked_oracle.append((best["batch_ms_warm_median"], method))
            selected.update((layer, method) for _, method in sorted(ranked_oracle)[:oracle_top_k])
    return selected


def make_formal_config(coarse: dict, points: list[dict]) -> dict:
    thresholds = {key: float(value) for key, value in coarse["recall_thresholds"].items()}
    selected = top_structures(points, thresholds)
    rows_by_key = defaultdict(list)
    for row in points:
        rows_by_key[(row["method"], row["workload"])].append(row)
    methods = []
    for source in coarse["methods"]:
        key = (int(source.get("layer_count", 0)), source["name"])
        if key not in selected:
            continue
        method = {key: value for key, value in source.items()
                  if key != "lsearch_values_by_workload"}
        by_workload = {}
        enabled = []
        for workload in coarse["workloads"]:
            rows = rows_by_key[(source["name"], workload["name"])]
            if not rows:
                continue
            grid = dense_crossing_grid(rows, thresholds[workload["name"]])
            by_workload[workload["name"]] = grid
            enabled.append(workload["name"])
        method["lsearch_values_by_workload"] = by_workload
        method["enabled_workloads"] = enabled
        methods.append(method)
    formal = dict(coarse)
    formal["num_repeats"] = 7
    formal["output_root"] = str(
        Path(coarse["output_root"]).with_name("layer_tuning_query_formal_amazon_x1"))
    formal["methods"] = methods
    return formal


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("coarse_config", type=Path)
    parser.add_argument("--points", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    coarse = json.loads(args.coarse_config.read_text())
    points_path = args.points or Path(coarse["output_root"]) / "summary/all_points.csv"
    points = select_layer_tuning.read_points(points_path)
    formal = make_formal_config(coarse, points)
    output = args.output or args.coarse_config.with_name(
        "config.amazon_x1_layer_tuning_query_formal.json")
    output.write_text(json.dumps(formal, indent=2) + "\n")
    print(f"wrote {len(formal['methods'])} formal structures to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
