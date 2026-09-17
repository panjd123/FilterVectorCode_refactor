#!/usr/bin/env python3
"""Summarize query-free auto-policy runs at matched Recall.

The first repeat is treated as cold.  Reported latency statistics use the six
remaining warm repeats.  Confidence intervals are paired bootstraps over the
warm-repeat median ratio, preserving the repeat pairing within each run.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import statistics
from pathlib import Path


def percentile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * q
    low, high = math.floor(position), math.ceil(position)
    if low == high:
        return ordered[low]
    return ordered[low] * (high - position) + ordered[high] * (position - low)


def paired_bootstrap_ratio(
    baseline: list[float], candidate: list[float], seed: int, samples: int
) -> tuple[float, float]:
    if len(baseline) != len(candidate):
        raise ValueError("paired samples must have equal length")
    rng = random.Random(seed)
    ratios = []
    for _ in range(samples):
        indices = [rng.randrange(len(baseline)) for _ in baseline]
        ratios.append(
            statistics.median(baseline[i] for i in indices)
            / statistics.median(candidate[i] for i in indices)
        )
    return percentile(ratios, 0.025), percentile(ratios, 0.975)


def stable_seed(*parts: str) -> int:
    return int(hashlib.sha256("/".join(parts).encode()).hexdigest()[:8], 16)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("configs", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=100_000)
    args = parser.parse_args()

    rows: list[dict] = []
    timings: dict[tuple[str, str, str], list[float]] = {}
    method_layers: dict[tuple[str, str], int] = {}

    for config_path in args.configs:
        config = json.loads(config_path.read_text())
        dataset = config.get("dataset") or config_path.name.split("_")[-2].title()
        root = Path(config["output_root"])
        thresholds = config["recall_thresholds"]
        for method in config["methods"]:
            method_name = method["name"]
            layer_count = int(method["layer_count"])
            method_layers[(dataset, method_name)] = layer_count
            for workload in config["workloads"]:
                workload_name = workload["name"]
                path = root / method_name / workload_name / "search_time_details.csv"
                with path.open(newline="") as stream:
                    raw = list(csv.DictReader(stream))
                repeats = sorted(int(row["Repeat"]) for row in raw)
                if repeats != list(range(7)):
                    raise ValueError(f"expected repeats 0..6: {path}")
                recalls = [float(row["Avg_Recall"]) for row in raw]
                target = float(thresholds[workload_name])
                if min(recalls) < target:
                    raise ValueError(f"Recall below {target}: {method_name}/{workload_name}")
                warm = [float(row["Time_ms"]) for row in raw if int(row["Repeat"]) > 0]
                timings[(dataset, method_name, workload_name)] = warm
                mean = statistics.mean(warm)
                rows.append({
                    "dataset": dataset,
                    "workload": workload_name,
                    "mean_selectivity": float(workload["mean_selectivity"]),
                    "method": method_name,
                    "layer_count": layer_count,
                    "t1": method.get("t1", ""),
                    "t2": method.get("t2", ""),
                    "lsearch": int(raw[0]["Lsearch"]),
                    "target_recall": target,
                    "recall_min": min(recalls),
                    "recall_mean": statistics.mean(recalls),
                    "warm_median_ms": statistics.median(warm),
                    "warm_mean_ms": mean,
                    "warm_cv": statistics.stdev(warm) / mean,
                    "warm_p95_ms": percentile(warm, 0.95),
                })

    row_index = {(row["dataset"], row["method"], row["workload"]): row for row in rows}
    for row in rows:
        dataset, workload = row["dataset"], row["workload"]
        methods = [
            method for (ds, method), _ in method_layers.items()
            if ds == dataset
        ]
        plain = next(method for method in methods if method_layers[(dataset, method)] == 0)
        plain_row = row_index[(dataset, plain, workload)]
        plain_times = timings[(dataset, plain, workload)]
        candidate_times = timings[(dataset, row["method"], workload)]
        row["speedup_vs_plain"] = plain_row["warm_median_ms"] / row["warm_median_ms"]
        low, high = paired_bootstrap_ratio(
            plain_times, candidate_times, stable_seed(dataset, workload, row["method"], "plain"),
            args.bootstrap_samples,
        )
        row["speedup_vs_plain_ci95_low"] = low
        row["speedup_vs_plain_ci95_high"] = high
        row["speedup_vs_same_t1_one_layer"] = ""
        row["same_t1_speedup_ci95_low"] = ""
        row["same_t1_speedup_ci95_high"] = ""
        if row["layer_count"] == 2:
            one = next(
                method for method in methods
                if method_layers[(dataset, method)] == 1
                and row_index[(dataset, method, workload)]["t1"] == row["t1"]
            )
            one_row = row_index[(dataset, one, workload)]
            one_times = timings[(dataset, one, workload)]
            row["speedup_vs_same_t1_one_layer"] = (
                one_row["warm_median_ms"] / row["warm_median_ms"]
            )
            low, high = paired_bootstrap_ratio(
                one_times, candidate_times, stable_seed(dataset, workload, row["method"], "one"),
                args.bootstrap_samples,
            )
            row["same_t1_speedup_ci95_low"] = low
            row["same_t1_speedup_ci95_high"] = high

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
