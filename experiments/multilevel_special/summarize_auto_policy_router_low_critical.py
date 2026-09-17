#!/usr/bin/env python3
"""Summarize 15-repeat low-selectivity plain/router confirmation."""

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
    values = sorted(values)
    position = (len(values) - 1) * q
    low, high = math.floor(position), math.ceil(position)
    if low == high:
        return values[low]
    return values[low] * (high - position) + values[high] * (position - low)


def bootstrap_ratio(
    baseline: list[float], candidate: list[float], seed: int, samples: int
) -> tuple[float, float]:
    rng = random.Random(seed)
    ratios = []
    for _ in range(samples):
        indices = [rng.randrange(len(baseline)) for _ in baseline]
        ratios.append(
            statistics.median(baseline[i] for i in indices)
            / statistics.median(candidate[i] for i in indices)
        )
    return percentile(ratios, 0.025), percentile(ratios, 0.975)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=100_000)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    root = Path(config["output_root"])
    expected_repeats = int(config["num_repeats"])
    methods = {int(m["layer_count"]): m["name"] for m in config["methods"]}
    if set(methods) != {0, 2}:
        raise ValueError("expected one plain and one two-layer router method")
    output_rows = []
    for workload in config["workloads"]:
        name = workload["name"]
        target = float(config["recall_thresholds"][name])
        samples_by_layer = {}
        recalls_by_layer = {}
        lsearch_by_layer = {}
        for layer, method in methods.items():
            path = root / method / name / "search_time_details.csv"
            with path.open(newline="") as stream:
                rows = list(csv.DictReader(stream))
            repeats = sorted(int(row["Repeat"]) for row in rows)
            if repeats != list(range(expected_repeats)):
                raise ValueError(f"unexpected repeats: {path}: {repeats}")
            recalls = [float(row["Avg_Recall"]) for row in rows]
            if min(recalls) < target:
                raise ValueError(f"Recall below {target}: {path}: {min(recalls)}")
            samples_by_layer[layer] = [
                float(row["Time_ms"]) for row in rows if int(row["Repeat"]) > 0
            ]
            recalls_by_layer[layer] = recalls
            lsearch_by_layer[layer] = int(rows[0]["Lsearch"])
        plain, router = samples_by_layer[0], samples_by_layer[2]
        seed = int(hashlib.sha256(name.encode()).hexdigest()[:8], 16)
        low, high = bootstrap_ratio(plain, router, seed, args.bootstrap_samples)
        output_rows.append({
            "workload": name,
            "mean_selectivity": workload["mean_selectivity"],
            "target_recall": target,
            "plain_lsearch": lsearch_by_layer[0],
            "router_lsearch": lsearch_by_layer[2],
            "plain_recall_min": min(recalls_by_layer[0]),
            "router_recall_min": min(recalls_by_layer[2]),
            "plain_warm_median_ms": statistics.median(plain),
            "router_warm_median_ms": statistics.median(router),
            "router_speedup_vs_plain": statistics.median(plain) / statistics.median(router),
            "speedup_ci95_low": low,
            "speedup_ci95_high": high,
            "plain_warm_cv": statistics.stdev(plain) / statistics.mean(plain),
            "router_warm_cv": statistics.stdev(router) / statistics.mean(router),
            "warm_repeats": len(plain),
        })
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(output_rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(output_rows)
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
