#!/usr/bin/env python3
"""Compare query-free DRH with a predeclared held-out manual oracle.

For each method and workload, the operating point is the smallest measured
Lsearch whose every measured (non-cold) repeat reaches the Recall threshold.
The per-workload oracle is then the lowest-latency crossing among the frozen
manual hierarchy grid. No interpolation or extrapolation is performed.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
from pathlib import Path
from typing import Any

import experiment_core
import summarize_selection_sweep


MANUAL_ROLES = {
    "predeclared_manual_oracle_grid",
    "predeclared_degree_ratio_hierarchy_v1",
}


def stable_seed(*parts: str) -> int:
    payload = "/".join(parts).encode("utf-8")
    return int(hashlib.sha256(payload).hexdigest()[:8], 16)


def first_crossings(
    rows: list[dict[str, Any]], thresholds: dict[str, float]
) -> dict[tuple[str, str], dict[str, Any]]:
    """Select the first conservative Recall crossing for every method/workload."""
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault((row["method"], row["workload"]), []).append(row)
    selected: dict[tuple[str, str], dict[str, Any]] = {}
    for key, candidates in grouped.items():
        target = float(thresholds[key[1]])
        feasible = [row for row in candidates if float(row["recall_min"]) >= target]
        if feasible:
            selected[key] = min(feasible, key=lambda row: int(row["lsearch"]))
    return selected


def geometric_mean(values: list[float]) -> float:
    if not values or any(value <= 0 for value in values):
        raise ValueError("geometric mean requires positive values")
    return math.exp(statistics.mean(math.log(value) for value in values))


def warm_times(
    config: dict[str, Any], method: dict[str, Any], workload: dict[str, Any],
    lsearch: int,
) -> list[float]:
    root = summarize_selection_sweep.measurement_root(config)
    path = root / method["name"] / workload["name"] / "search_time_details.csv"
    rows = experiment_core.read_csv(path)
    protocol = experiment_core.protocol_for(config)
    values = [
        float(row["Time_ms"])
        for row in rows
        if int(row["Lsearch"]) == lsearch
        and int(row["Repeat"]) >= protocol.cold_repeats
    ]
    if len(values) != protocol.measured_repeats:
        raise ValueError(
            f"{method['name']}/{workload['name']} L={lsearch}: expected "
            f"{protocol.measured_repeats} warm repeats, found {len(values)}")
    return values


def layer_spec(row: dict[str, Any]) -> str:
    if int(row["layer_count"]) == 0:
        return "none"
    return f"{row['thresholds']}:{row['layer_topologies']}"


def evaluate_config(
    config_path: Path, bootstrap_samples: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    config = experiment_core.load_config(config_path)
    rows = summarize_selection_sweep.read_rows(config)
    crossings = first_crossings(rows, config["recall_thresholds"])
    methods = {method["name"]: method for method in config["methods"]}
    workloads = {workload["name"]: workload for workload in config["workloads"]}
    baseline_names = [
        name for name, method in methods.items()
        if method.get("selection_role") == "zero_layer_baseline"
    ]
    automatic_names = [
        name for name, method in methods.items()
        if method.get("selection_role") == "predeclared_degree_ratio_hierarchy_v1"
    ]
    manual_names = [
        name for name, method in methods.items()
        if method.get("selection_role") in MANUAL_ROLES
    ]
    routed_names = [
        name for name, method in methods.items()
        if method.get("selection_role") == "automatic_upper_authorization_control"
    ]
    if len(baseline_names) != 1 or len(automatic_names) != 1:
        raise ValueError("expected exactly one zero-layer baseline and one DRH method")
    if not manual_names:
        raise ValueError("manual oracle grid is empty")
    baseline_name = baseline_names[0]
    automatic_name = automatic_names[0]
    dataset = str(config["dataset"])
    output: list[dict[str, Any]] = []

    for workload_name, workload in workloads.items():
        baseline = crossings.get((baseline_name, workload_name))
        automatic = crossings.get((automatic_name, workload_name))
        manual = [
            crossings[(name, workload_name)]
            for name in manual_names
            if (name, workload_name) in crossings
        ]
        if baseline is None or not manual:
            raise ValueError(
                f"{dataset}/{workload_name}: missing baseline or oracle crossing")
        oracle = min(manual, key=lambda row: float(row["batch_ms_warm_median"]))
        baseline_times = warm_times(
            config, methods[baseline_name], workload, int(baseline["lsearch"]))
        oracle_times = warm_times(
            config, methods[oracle["method"]], workload, int(oracle["lsearch"]))
        baseline_ms = float(baseline["batch_ms_warm_median"])
        oracle_ms = float(oracle["batch_ms_warm_median"])
        result = {
            "status": ("complete" if automatic is not None
                       else "automatic_no_crossing"),
            "dataset": dataset,
            "workload": workload_name,
            "mean_selectivity": float(workload["mean_selectivity"]),
            "target_recall": float(config["recall_thresholds"][workload_name]),
            "baseline_method": baseline_name,
            "baseline_lsearch": int(baseline["lsearch"]),
            "baseline_recall_min": float(baseline["recall_min"]),
            "baseline_qps": float(baseline["qps_warm_median"]),
            "automatic_method": automatic_name,
            "automatic_hierarchy": (
                layer_spec(automatic) if automatic is not None else
                experiment_core.encode_hierarchy_layers(methods[automatic_name])),
            "automatic_lsearch": "",
            "automatic_recall_min": "",
            "automatic_qps": "",
            "oracle_method": oracle["method"],
            "oracle_hierarchy": layer_spec(oracle),
            "oracle_lsearch": int(oracle["lsearch"]),
            "oracle_recall_min": float(oracle["recall_min"]),
            "oracle_qps": float(oracle["qps_warm_median"]),
            "automatic_speedup_vs_baseline": "",
            "automatic_speedup_ci95_low": "",
            "automatic_speedup_ci95_high": "",
            "oracle_speedup_vs_baseline": baseline_ms / oracle_ms,
            "automatic_qps_fraction_of_oracle": "",
            "automatic_oracle_fraction_ci95_low": "",
            "automatic_oracle_fraction_ci95_high": "",
            "manual_feasible_methods": len(manual),
        }
        if automatic is not None:
            automatic_times = warm_times(
                config, methods[automatic_name], workload,
                int(automatic["lsearch"]))
            auto_ci = experiment_core.bootstrap_median_ratio(
                baseline_times, automatic_times,
                seed=stable_seed(dataset, workload_name, "auto_vs_baseline"),
                samples=bootstrap_samples, paired=False)
            regret_ci = experiment_core.bootstrap_median_ratio(
                oracle_times, automatic_times,
                seed=stable_seed(dataset, workload_name, "auto_vs_oracle"),
                samples=bootstrap_samples, paired=False)
            automatic_ms = float(automatic["batch_ms_warm_median"])
            result.update({
                "automatic_lsearch": int(automatic["lsearch"]),
                "automatic_recall_min": float(automatic["recall_min"]),
                "automatic_qps": float(automatic["qps_warm_median"]),
                "automatic_speedup_vs_baseline": baseline_ms / automatic_ms,
                "automatic_speedup_ci95_low": auto_ci[0],
                "automatic_speedup_ci95_high": auto_ci[1],
                "automatic_qps_fraction_of_oracle": oracle_ms / automatic_ms,
                "automatic_oracle_fraction_ci95_low": regret_ci[0],
                "automatic_oracle_fraction_ci95_high": regret_ci[1],
            })
        if routed_names:
            routed = crossings.get((routed_names[0], workload_name))
            result["routed_control_method"] = routed_names[0]
            result["routed_control_qps"] = (
                float(routed["qps_warm_median"]) if routed else "")
            result["routed_control_lsearch"] = int(routed["lsearch"]) if routed else ""
        output.append(result)

    complete_manual = []
    for name in manual_names:
        selected = [crossings.get((name, workload)) for workload in workloads]
        if all(row is not None for row in selected):
            complete_manual.append((
                name,
                geometric_mean([float(row["qps_warm_median"]) for row in selected]),
            ))
    if not complete_manual:
        raise ValueError(f"{dataset}: no manual method crosses on every workload")
    global_oracle_name, global_oracle_qps = max(complete_manual, key=lambda item: item[1])
    automatic_rows = [crossings.get((automatic_name, workload))
                      for workload in workloads]
    auto_feasible = all(row is not None for row in automatic_rows)
    auto_qps = (geometric_mean([
        float(row["qps_warm_median"]) for row in automatic_rows
    ]) if auto_feasible else "")
    baseline_qps = geometric_mean([
        float(crossings[(baseline_name, workload)]["qps_warm_median"])
        for workload in workloads
    ])
    global_row = {
        "dataset": dataset,
        "status": "complete" if auto_feasible else "automatic_no_crossing",
        "workload_count": len(workloads),
        "automatic_method": automatic_name,
        "automatic_geomean_qps": auto_qps,
        "global_oracle_method": global_oracle_name,
        "global_oracle_geomean_qps": global_oracle_qps,
        "automatic_qps_fraction_of_global_oracle": (
            auto_qps / global_oracle_qps if auto_feasible else ""),
        "baseline_geomean_qps": baseline_qps,
        "automatic_speedup_vs_baseline": (
            auto_qps / baseline_qps if auto_feasible else ""),
        "complete_manual_methods": len(complete_manual),
    }
    return output, global_row


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(
    path: Path, workload_rows: list[dict[str, Any]], global_rows: list[dict[str, Any]],
) -> None:
    def ratio(value: Any, suffix: str = "") -> str:
        return f"{float(value):.3f}{suffix}" if value != "" else "NA"

    lines = [
        "# Query-free hierarchy versus held-out oracle", "",
        "Each operating point is the smallest measured L whose every warm repeat "
        "meets the declared Recall threshold. The oracle grid was frozen before "
        "held-out search results were read; no interpolation is used.", "",
        "| Dataset | Selectivity | DRH hierarchy | Oracle hierarchy | DRH/plain | DRH/oracle |",
        "|---|---:|---|---|---:|---:|",
    ]
    for row in workload_rows:
        lines.append(
            f"| {row['dataset']} | {row['mean_selectivity']:.3%} | "
            f"{row['automatic_hierarchy']} | {row['oracle_hierarchy']} | "
            f"{ratio(row['automatic_speedup_vs_baseline'], 'x')} | "
            f"{ratio(row['automatic_qps_fraction_of_oracle'])} |")
    lines.extend(["", "## Global configuration", "",
                  "| Dataset | DRH method | Manual oracle | DRH/oracle | DRH/plain |",
                  "|---|---|---|---:|---:|"])
    for row in global_rows:
        lines.append(
            f"| {row['dataset']} | {row['automatic_method']} | "
            f"{row['global_oracle_method']} | "
            f"{ratio(row['automatic_qps_fraction_of_global_oracle'])} | "
            f"{ratio(row['automatic_speedup_vs_baseline'], 'x')} |")
    path.write_text("\n".join(lines) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("configs", nargs="+", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=20000)
    args = parser.parse_args()
    workload_rows = []
    global_rows = []
    for config in args.configs:
        local, global_row = evaluate_config(config, args.bootstrap_samples)
        workload_rows.extend(local)
        global_rows.append(global_row)
    write_csv(args.output_dir / "heldout_oracle_by_workload.csv", workload_rows)
    write_csv(args.output_dir / "heldout_oracle_global.csv", global_rows)
    write_markdown(args.output_dir / "heldout_oracle.md", workload_rows, global_rows)
    print(args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
