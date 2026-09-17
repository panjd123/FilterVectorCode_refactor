#!/usr/bin/env python3
"""Summarize the 15-repeat same-T1 Amazon confirmation experiment."""

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
    methods = sorted(config["methods"], key=lambda method: int(method["layer_count"]))
    if [int(method["layer_count"]) for method in methods] != [1, 2]:
        raise ValueError("critical experiment must contain one- and two-layer methods")
    if methods[0].get("t1") != methods[1].get("t1"):
        raise ValueError("critical experiment must hold T1 constant")

    output_rows = []
    for workload in config["workloads"]:
        name = workload["name"]
        warm_by_method = []
        recall_min = []
        lsearch = []
        for method in methods:
            path = root / method["name"] / name / "search_time_details.csv"
            with path.open(newline="") as stream:
                rows = list(csv.DictReader(stream))
            repeats = sorted(int(row["Repeat"]) for row in rows)
            if repeats != list(range(expected_repeats)):
                raise ValueError(f"expected repeats 0..{expected_repeats - 1}: {path}")
            recalls = [float(row["Avg_Recall"]) for row in rows]
            target = float(config["recall_thresholds"][name])
            if min(recalls) < target:
                raise ValueError(f"Recall below {target}: {method['name']}/{name}")
            warm_by_method.append(
                [float(row["Time_ms"]) for row in rows if int(row["Repeat"]) > 0]
            )
            recall_min.append(min(recalls))
            lsearch.append(int(rows[0]["Lsearch"]))
        one, two = warm_by_method
        seed = int(hashlib.sha256(name.encode()).hexdigest()[:8], 16)
        low, high = bootstrap_ratio(one, two, seed, args.bootstrap_samples)
        output_rows.append({
            "workload": name,
            "mean_selectivity": workload["mean_selectivity"],
            "target_recall": config["recall_thresholds"][name],
            "one_layer_lsearch": lsearch[0],
            "two_layer_lsearch": lsearch[1],
            "one_layer_recall_min": recall_min[0],
            "two_layer_recall_min": recall_min[1],
            "one_layer_warm_median_ms": statistics.median(one),
            "two_layer_warm_median_ms": statistics.median(two),
            "two_layer_speedup": statistics.median(one) / statistics.median(two),
            "speedup_ci95_low": low,
            "speedup_ci95_high": high,
            "warm_repeats": len(one),
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
