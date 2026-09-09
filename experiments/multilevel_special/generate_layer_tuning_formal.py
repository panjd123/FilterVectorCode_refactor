#!/usr/bin/env python3
"""Generate 7-repeat formal reruns from the completed coarse sweep."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path

import select_layer_tuning


# A formal rerun can expose a winner on a structural axis that appeared
# closed under the noisier three-repeat coarse sweep.  These controls are
# declared before execution and inherit only the L-search bracket of the
# listed adjacent coarse structure; their thresholds remain independent.
FORMAL_BOUNDARY_GUARDS = {
    "layer2_t1_8000_t2_200000": {
        "reference": "layer2_t1_8000_t2_100000", "t1": 8000,
        "t2": 200000, "workloads": None},
    "layer2_t1_16000_t2_400000": {
        "reference": "layer2_t1_16000_t2_200000", "t1": 16000,
        "t2": 400000, "workloads": None},
    "layer2_t1_32000_t2_400000": {
        "reference": "layer2_t1_32000_t2_200000", "t1": 32000,
        "t2": 400000, "workloads": ["sel_005"]},
    "layer2_t1_32000_t2_64001": {
        "reference": "layer2_t1_64000_t2_64001", "t1": 32000,
        "t2": 64001, "workloads": ["sel_25"]},
}
FORMAL_BOUNDARY_ENDPOINTS = (
    {"name": "layer2_t1_16000_t2_600000", "layer_count": 2,
     "t1": 16000, "t2": 600000},
    {"name": "layer2_t1_32000_t2_600000", "layer_count": 2,
     "t1": 32000, "t2": 600000},
)


def dense_crossing_grid(rows: list[dict], threshold: float) -> list[int]:
    """Bracket the measured crossing and add real integer L points nearby."""
    ordered = sorted(rows, key=lambda row: row["lsearch"])
    feasible = [row for row in ordered if row["recall_min"] >= threshold]
    if not feasible:
        return sorted({row["lsearch"] for row in ordered[-2:]})
    high = min(feasible, key=lambda row: row["lsearch"])["lsearch"]
    lower = [row["lsearch"] for row in ordered if row["lsearch"] < high]
    low = max(lower) if lower else max(1, high // 2)
    higher = [row["lsearch"] for row in ordered if row["lsearch"] > high]
    guard = min(higher) if higher else int(math.ceil(high * 1.25))
    if low == high:
        return sorted({high, guard})
    # Include the bracket, interior points, and one guard point above the
    # coarse crossing. Every value is subsequently measured in the formal run.
    return sorted({low, high,
                   int(round(low + (high - low) * 0.25)),
                   int(round(low + (high - low) * 0.50)),
                   int(round(low + (high - low) * 0.75)), guard})


def selected_structure_workloads(
    points: list[dict], thresholds: dict[str, float],
    shared_top_k: int = 3, oracle_top_k: int = 2,
    near_best_ratio: float = 1.05,
) -> dict[tuple[int, str], set[str]]:
    """Map each shortlisted structure to workloads needing a formal rerun.

    Shared candidates must be measured on all workloads.  A structure selected
    only for a per-workload oracle is rerun only on the workload(s) for which
    it was competitive; this preserves the selection evidence while avoiding
    an unnecessary Cartesian product in the seven-repeat stage.
    """
    workloads = sorted(thresholds)
    selected: dict[tuple[int, str], set[str]] = defaultdict(set)
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
        ranked_shared.sort()
        shared_limit = (ranked_shared[0][0] * near_best_ratio
                        if ranked_shared else float("inf"))
        for rank, (score, method) in enumerate(ranked_shared):
            if rank >= shared_top_k and score > shared_limit:
                continue
            selected[(layer, method)].update(workloads)
        for workload in workloads:
            ranked_oracle = []
            for method in methods:
                best = select_layer_tuning.fastest_feasible(
                    [row for row in points if row["method"] == method and row["workload"] == workload],
                    thresholds[workload])
                if best is not None:
                    ranked_oracle.append((best["batch_ms_warm_median"], method))
            ranked_oracle.sort()
            oracle_limit = (ranked_oracle[0][0] * near_best_ratio
                            if ranked_oracle else float("inf"))
            for rank, (latency, method) in enumerate(ranked_oracle):
                if rank >= oracle_top_k and latency > oracle_limit:
                    continue
                selected[(layer, method)].add(workload)
    return selected


def top_structures(points: list[dict], thresholds: dict[str, float],
                   shared_top_k: int = 3, oracle_top_k: int = 2) -> set[tuple[int, str]]:
    """Compatibility wrapper returning only shortlisted structure keys."""
    return set(selected_structure_workloads(
        points, thresholds, shared_top_k, oracle_top_k))


def make_formal_config(coarse: dict, points: list[dict],
                       method_configs: list[dict] | None = None,
                       boundary_guards: dict | None = None,
                       output_root_name: str =
                       "layer_tuning_query_formal_fair_amazon_x1") -> dict:
    thresholds = {key: float(value) for key, value in coarse["recall_thresholds"].items()}
    selected = selected_structure_workloads(points, thresholds)
    source_by_name = {}
    for config in method_configs or [coarse]:
        for source in config["methods"]:
            existing = source_by_name.get(source["name"])
            identity = tuple(source.get(key) for key in
                             ("layer_count", "t1", "t2", "block_index"))
            if existing is not None:
                existing_identity = tuple(existing.get(key) for key in
                                          ("layer_count", "t1", "t2", "block_index"))
                if identity != existing_identity:
                    raise ValueError(
                        f"conflicting method definition: {source['name']}")
                continue
            source_by_name[source["name"]] = source
    active_guards = (FORMAL_BOUNDARY_GUARDS if boundary_guards is None else
                     boundary_guards)
    for target, guard in active_guards.items():
        reference = source_by_name.get(guard["reference"])
        if reference is None:
            continue
        if target not in source_by_name:
            source = dict(reference)
            source.update({"name": target, "t1": guard["t1"], "t2": guard["t2"]})
            if source.get("block_index"):
                source["block_index"] = str(
                    Path(source["block_index"]).parents[1] / target / "block_index")
            source.pop("enabled_workloads", None)
            source_by_name[target] = source
        source = source_by_name[target]
        selected[(int(source["layer_count"]), target)].update(
            guard["workloads"] or thresholds)
    rows_by_key = defaultdict(list)
    for row in points:
        rows_by_key[(row["method"], row["workload"])].append(row)
    methods = []
    for source in source_by_name.values():
        key = (int(source.get("layer_count", 0)), source["name"])
        if key not in selected:
            continue
        method = {key: value for key, value in source.items()
                  if key != "lsearch_values_by_workload"}
        by_workload = {}
        enabled = []
        selected_workloads = selected[key]
        for workload in coarse["workloads"]:
            if workload["name"] not in selected_workloads:
                continue
            guard = active_guards.get(source["name"])
            reference = guard["reference"] if guard else source["name"]
            rows = rows_by_key[(reference, workload["name"])]
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
        Path(coarse["output_root"]).with_name(output_root_name))
    formal["formal_selection"] = {
        "shared_top_k_per_layer": 3,
        "oracle_top_k_per_layer_workload": 2,
        "coarse_near_best_ratio": 1.05,
        "shortlist_rule": "retain top-k plus every structure within 5% of the coarse best",
        "boundary_guards": active_guards,
        "quality_rule": "minimum repeat Recall meets the declared threshold",
        "timing_rule": "warm-repeat batch median; cold repeat 0 excluded",
    }
    # Boundary auditing after shortlist reruns must retain the full coarse
    # structure space rather than treating the shortlist as the entire grid.
    formal["boundary_reference_methods"] = [
        {key: source.get(key) for key in ("name", "layer_count", "t1", "t2")}
        for source in source_by_name.values()
    ]
    if active_guards:
        formal["boundary_reference_methods"].extend(
            {"name": name, "layer_count": 2, "t1": guard["t1"], "t2": guard["t2"]}
            for name, guard in active_guards.items())
        formal["boundary_reference_methods"].extend(FORMAL_BOUNDARY_ENDPOINTS)
    formal["methods"] = methods
    return formal


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("coarse_config", type=Path)
    parser.add_argument("--points", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--method-configs", nargs="*", type=Path)
    parser.add_argument("--disable-boundary-guards", action="store_true")
    parser.add_argument("--output-root-name",
                        default="layer_tuning_query_formal_fair_amazon_x1")
    args = parser.parse_args()
    coarse = json.loads(args.coarse_config.read_text())
    points_path = args.points or Path(coarse["output_root"]) / "summary/all_points.csv"
    points = select_layer_tuning.read_points(points_path)
    method_config_paths = args.method_configs or [args.coarse_config]
    method_configs = [json.loads(path.read_text())
                      for path in method_config_paths]
    formal = make_formal_config(
        coarse, points, method_configs=method_configs,
        boundary_guards={} if args.disable_boundary_guards else None,
        output_root_name=args.output_root_name)
    formal["selection_provenance"] = {
        "coarse_config": str(args.coarse_config.resolve()),
        "coarse_config_sha256": hashlib.sha256(
            args.coarse_config.read_bytes()).hexdigest(),
        "coarse_points": str(points_path.resolve()),
        "coarse_points_sha256": hashlib.sha256(points_path.read_bytes()).hexdigest(),
        "method_configs": [str(path.resolve()) for path in method_config_paths],
        "method_config_sha256": {
            str(path.resolve()): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in method_config_paths
        },
    }
    output = args.output or args.coarse_config.with_name(
        "config.amazon_x1_layer_tuning_query_formal.json")
    output.write_text(json.dumps(formal, indent=2) + "\n")
    print(f"wrote {len(formal['methods'])} formal structures to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
