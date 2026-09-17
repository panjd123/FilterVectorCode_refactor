#!/usr/bin/env python3
"""Summarize structural-router runs against frozen cross-dataset baselines."""

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


def bootstrap_ratio(a: list[float], b: list[float], seed: int, n: int) -> tuple[float, float]:
    rng = random.Random(seed)
    ratios = []
    for _ in range(n):
        indices = [rng.randrange(len(a)) for _ in a]
        ratios.append(statistics.median(a[i] for i in indices) / statistics.median(b[i] for i in indices))
    return percentile(ratios, 0.025), percentile(ratios, 0.975)


def load_warm(root: Path, method: str, workload: str, target: float) -> tuple[list[float], list[float], int]:
    path = root / method / workload / "search_time_details.csv"
    with path.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    repeats = sorted(int(row["Repeat"]) for row in rows)
    if repeats != list(range(7)):
        raise ValueError(f"expected repeats 0..6: {path}: {repeats}")
    recalls = [float(row["Avg_Recall"]) for row in rows]
    if min(recalls) < target:
        raise ValueError(f"Recall below {target}: {path}: {min(recalls)}")
    warm = [float(row["Time_ms"]) for row in rows if int(row["Repeat"]) > 0]
    return warm, recalls, int(rows[0]["Lsearch"])


def route_fraction(root: Path, method: str, workload: str) -> tuple[int, int]:
    path = root / method / workload / "query_details_repeat7.csv"
    with path.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) % 7:
        raise ValueError(f"query rows not divisible by 7: {path}")
    n = len(rows) // 7
    counts = []
    for repeat in range(7):
        batch = rows[repeat * n : (repeat + 1) * n]
        counts.append(sum(int(float(row["SpecialBlockSearchUsed"])) for row in batch))
    if len(set(counts)) != 1:
        raise ValueError(f"nondeterministic route count: {path}: {counts}")
    return counts[0], n


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pair", nargs=2, action="append", required=True, metavar=("ROUTER", "BASELINE"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=100_000)
    args = parser.parse_args()
    output_rows = []
    for router_path_text, baseline_path_text in args.pair:
        router_config = json.loads(Path(router_path_text).read_text())
        baseline_config = json.loads(Path(baseline_path_text).read_text())
        dataset = router_config["dataset"]
        if baseline_config["dataset"] != dataset:
            raise ValueError("router/baseline dataset mismatch")
        router_root = Path(router_config["output_root"])
        baseline_root = Path(baseline_config["output_root"])
        router_method = router_config["methods"][0]["name"]
        baseline_methods = {int(m["layer_count"]): m["name"] for m in baseline_config["methods"]}
        if set(baseline_methods) != {0, 1, 2}:
            raise ValueError(f"expected 0/1/2-layer baselines for {dataset}")
        for workload in router_config["workloads"]:
            name = workload["name"]
            target = float(router_config["recall_thresholds"][name])
            router, recalls, router_l = load_warm(router_root, router_method, name, target)
            used, num_queries = route_fraction(router_root, router_method, name)
            row = {
                "dataset": dataset,
                "workload": name,
                "mean_selectivity": workload["mean_selectivity"],
                "target_recall": target,
                "router_lsearch": router_l,
                "router_recall_min": min(recalls),
                "router_recall_mean": statistics.mean(recalls),
                "router_warm_median_ms": statistics.median(router),
                "router_warm_mean_ms": statistics.mean(router),
                "router_warm_cv": statistics.stdev(router) / statistics.mean(router),
                "router_warm_p95_ms": percentile(router, 0.95),
                "special_route_queries": used,
                "num_queries": num_queries,
                "special_route_fraction": used / num_queries,
            }
            for layer, label in ((0, "plain"), (1, "one_layer"), (2, "ungated_two_layer")):
                baseline, _, baseline_l = load_warm(
                    baseline_root, baseline_methods[layer], name, target
                )
                row[f"{label}_lsearch"] = baseline_l
                row[f"speedup_vs_{label}"] = statistics.median(baseline) / statistics.median(router)
                seed = int(hashlib.sha256(f"{dataset}:{name}:{label}".encode()).hexdigest()[:8], 16)
                lo, hi = bootstrap_ratio(baseline, router, seed, args.bootstrap_samples)
                row[f"speedup_vs_{label}_ci95_low"] = lo
                row[f"speedup_vs_{label}_ci95_high"] = hi
            output_rows.append(row)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(output_rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(output_rows)
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
