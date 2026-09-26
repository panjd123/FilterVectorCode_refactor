#!/usr/bin/env python3
"""Compare plain, best one-layer, best two-layer, and DRH at equal Recall."""

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


def stable_seed(*parts: str) -> int:
    return int(hashlib.sha256("/".join(parts).encode()).hexdigest()[:8], 16)


def first_crossings(
    rows: list[dict[str, Any]], thresholds: dict[str, float],
) -> dict[tuple[str, str], dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault((row["method"], row["workload"]), []).append(row)
    selected = {}
    for key, candidates in grouped.items():
        feasible = [row for row in candidates
                    if float(row["recall_min"]) >= float(thresholds[key[1]])]
        if feasible:
            selected[key] = min(feasible, key=lambda row: int(row["lsearch"]))
    return selected


def method_groups(config: dict[str, Any], baseline: str) -> dict[str, list[str]]:
    groups: dict[str, list[str]] = {
        "plain": [baseline], "best_one_layer": [], "best_two_layer": [],
        "automatic_drh": [], "automatic_routed": [],
    }
    for method in config["methods"]:
        name = method["name"]
        layers = experiment_core.hierarchy_layers(method)
        routing = method.get("routing_policy", "always_layered")
        if routing == "always_layered" and len(layers) == 1:
            groups["best_one_layer"].append(name)
        if routing == "always_layered" and len(layers) == 2:
            groups["best_two_layer"].append(name)
        role = str(method.get("selection_role", ""))
        entry = method.get("entry_strategy", method.get("entry_group_provider"))
        if (role == "predeclared_degree_ratio_hierarchy_v1"
                and entry == "optimized_lng" and routing == "always_layered"):
            groups["automatic_drh"].append(name)
        if ("upper_authorization_control" in role
                and entry == "optimized_lng"
                and name.startswith("l2_t1024_16384_")):
            groups["automatic_routed"].append(name)
    if len(groups["plain"]) != 1:
        raise ValueError("expected one plain baseline")
    if len(groups["automatic_drh"]) != 1:
        raise ValueError("expected one optimized-LNG DRH method")
    if len(groups["automatic_routed"]) != 1:
        raise ValueError("expected one routed optimized-LNG DRH method")
    return groups


def fastest_crossing(
    crossings: dict[tuple[str, str], dict[str, Any]],
    methods: list[str], workload: str,
) -> dict[str, Any] | None:
    candidates = [crossings[(method, workload)] for method in methods
                  if (method, workload) in crossings]
    return min(candidates, key=lambda row: float(row["batch_ms_warm_median"])) \
        if candidates else None


def geometric_mean(values: list[float]) -> float:
    if not values or any(value <= 0 for value in values):
        raise ValueError("geometric mean requires positive values")
    return math.exp(statistics.mean(math.log(value) for value in values))


def global_method(
    crossings: dict[tuple[str, str], dict[str, Any]], methods: list[str],
    workloads: list[str],
) -> tuple[str, float] | None:
    candidates = []
    for method in methods:
        rows = [crossings.get((method, workload)) for workload in workloads]
        if all(row is not None for row in rows):
            candidates.append((method, geometric_mean([
                float(row["qps_warm_median"]) for row in rows
            ])))
    return max(candidates, key=lambda item: item[1]) if candidates else None


def warm_times(
    config: dict[str, Any], method: str, workload: str, lsearch: int,
) -> list[float]:
    root = summarize_selection_sweep.measurement_root(config)
    path = root / method / workload / "search_time_details.csv"
    protocol = experiment_core.protocol_for(config)
    values = [
        float(row["Time_ms"]) for row in experiment_core.read_csv(path)
        if int(row["Lsearch"]) == lsearch
        and int(row["Repeat"]) >= protocol.cold_repeats
    ]
    if len(values) != protocol.measured_repeats:
        raise ValueError(
            f"{method}/{workload}: expected {protocol.measured_repeats} warm repeats, "
            f"found {len(values)}")
    return values


def hierarchy(row: dict[str, Any]) -> str:
    return "none" if int(row["layer_count"]) == 0 else \
        f"{row['thresholds']}:{row['layer_topologies']}"


def evaluate(
    config: dict[str, Any], baseline: str, bootstrap_samples: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows = summarize_selection_sweep.read_rows(config)
    crossings = first_crossings(rows, config["recall_thresholds"])
    groups = method_groups(config, baseline)
    workloads = [item["name"] for item in sorted(
        config["workloads"], key=lambda item: float(item["mean_selectivity"]))]
    workload_meta = {item["name"]: item for item in config["workloads"]}
    output = []
    for workload in workloads:
        plain = fastest_crossing(crossings, groups["plain"], workload)
        if plain is None:
            raise ValueError(f"plain baseline has no crossing for {workload}")
        plain_times = warm_times(config, plain["method"], workload,
                                 int(plain["lsearch"]))
        plain_ms = float(plain["batch_ms_warm_median"])
        for category, methods in groups.items():
            selected = fastest_crossing(crossings, methods, workload)
            record = {
                "status": "complete" if selected is not None else "no_crossing",
                "dataset": config["dataset"], "workload": workload,
                "mean_selectivity": float(workload_meta[workload]["mean_selectivity"]),
                "target_recall": float(config["recall_thresholds"][workload]),
                "category": category, "method": "", "hierarchy": "",
                "entry_strategy": "", "lsearch": "", "recall_min": "",
                "qps": "", "speedup_vs_plain": "",
                "speedup_ci95_low": "", "speedup_ci95_high": "",
            }
            if selected is not None:
                selected_times = warm_times(
                    config, selected["method"], workload, int(selected["lsearch"]))
                if (selected["method"] == plain["method"]
                        and int(selected["lsearch"]) == int(plain["lsearch"])):
                    ci = (1.0, 1.0)
                else:
                    ci = experiment_core.bootstrap_median_ratio(
                        plain_times, selected_times,
                        seed=stable_seed(config["dataset"], workload, category),
                        samples=bootstrap_samples, paired=False)
                record.update({
                    "method": selected["method"], "hierarchy": hierarchy(selected),
                    "entry_strategy": selected["entry_strategy"],
                    "lsearch": int(selected["lsearch"]),
                    "recall_min": float(selected["recall_min"]),
                    "qps": float(selected["qps_warm_median"]),
                    "speedup_vs_plain": plain_ms / float(selected["batch_ms_warm_median"]),
                    "speedup_ci95_low": ci[0], "speedup_ci95_high": ci[1],
                })
            output.append(record)
    global_rows = []
    plain_global = global_method(crossings, groups["plain"], workloads)
    if plain_global is None:
        raise ValueError("plain baseline is not globally feasible")
    for category, methods in groups.items():
        selected = global_method(crossings, methods, workloads)
        global_rows.append({
            "dataset": config["dataset"], "category": category,
            "status": "complete" if selected else "no_global_crossing",
            "method": selected[0] if selected else "",
            "geomean_qps": selected[1] if selected else "",
            "speedup_vs_plain": (selected[1] / plain_global[1] if selected else ""),
            "workload_count": len(workloads),
        })
    return output, global_rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(path: Path, rows: list[dict[str, Any]], global_rows: list[dict[str, Any]]) -> None:
    lines = [
        "# Depth ablation at conservative Recall crossing", "",
        "Every row uses the smallest measured Lsearch whose every warm repeat "
        "reaches the workload target. Best-one and best-two are per-workload "
        "measured oracles; global rows use one unchanged method on every workload.", "",
        "| Selectivity | Category | Method | Hierarchy | Recall min | L | QPS | vs plain | 95% CI |",
        "|---:|---|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        if row["status"] != "complete":
            lines.append(
                f"| {row['mean_selectivity']:.3%} | {row['category']} | no crossing | "
                "NA | NA | NA | NA | NA | NA |")
            continue
        lines.append(
            f"| {row['mean_selectivity']:.3%} | {row['category']} | {row['method']} | "
            f"{row['hierarchy']} | {row['recall_min']:.4f} | {row['lsearch']} | "
            f"{row['qps']:.3f} | {row['speedup_vs_plain']:.3f}x | "
            f"[{row['speedup_ci95_low']:.3f}, {row['speedup_ci95_high']:.3f}] |")
    lines.extend(["", "## One global configuration", "",
                  "| Category | Method | Geomean QPS | vs plain |",
                  "|---|---|---:|---:|"])
    for row in global_rows:
        if row["status"] != "complete":
            lines.append(f"| {row['category']} | no global crossing | NA | NA |")
        else:
            lines.append(
                f"| {row['category']} | {row['method']} | {row['geomean_qps']:.3f} | "
                f"{row['speedup_vs_plain']:.3f}x |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=Path)
    parser.add_argument("--baseline", default="l0_lng_entry_optimized_lng")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=20000)
    args = parser.parse_args()
    config = experiment_core.load_config(args.config)
    rows, global_rows = evaluate(config, args.baseline, args.bootstrap_samples)
    write_csv(args.output_dir / "depth_by_workload.csv", rows)
    write_csv(args.output_dir / "depth_global.csv", global_rows)
    write_markdown(args.output_dir / "depth_ablation.md", rows, global_rows)
    print(args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
