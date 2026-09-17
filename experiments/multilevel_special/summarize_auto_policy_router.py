#!/usr/bin/env python3
"""Summarize calibration-free structural-router experiments.

The router run is compared with the previously recorded exact-level formal
experiment.  Repeat zero is cold; all reported latency statistics use warm
repeats only.  Recall is a hard per-repeat gate.
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


def load_run(
    root: Path, method: str, workload: str, expected_repeats: int, target: float
) -> tuple[list[float], list[float], int]:
    path = root / method / workload / "search_time_details.csv"
    with path.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    repeats = sorted(int(row["Repeat"]) for row in rows)
    if repeats != list(range(expected_repeats)):
        raise ValueError(
            f"expected repeats 0..{expected_repeats - 1}, got {repeats}: {path}"
        )
    recalls = [float(row["Avg_Recall"]) for row in rows]
    if min(recalls) < target:
        raise ValueError(
            f"Recall below {target}: {method}/{workload}, min={min(recalls)}"
        )
    warm = [float(row["Time_ms"]) for row in rows if int(row["Repeat"]) > 0]
    return warm, recalls, int(rows[0]["Lsearch"])


def route_count(
    root: Path, method: str, workload: str, expected_repeats: int, num_queries: int
) -> int:
    path = root / method / workload / f"query_details_repeat{expected_repeats}.csv"
    with path.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    expected_rows = expected_repeats * num_queries
    if len(rows) != expected_rows:
        raise ValueError(f"expected {expected_rows} query rows, got {len(rows)}: {path}")
    # Routing is deterministic, so inspect one complete repeat and verify all
    # repeats report the same counts.
    counts = []
    for repeat in range(expected_repeats):
        batch = rows[repeat * num_queries : (repeat + 1) * num_queries]
        used = sum(int(float(row["SpecialBlockSearchUsed"])) for row in batch)
        counts.append(used)
    if len(set(counts)) != 1:
        raise ValueError(f"nondeterministic routing counts: {workload}: {counts}")
    return counts[0]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("router_config", type=Path)
    parser.add_argument("baseline_config", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=100_000)
    args = parser.parse_args()

    router_config = json.loads(args.router_config.read_text())
    baseline_config = json.loads(args.baseline_config.read_text())
    router_root = Path(router_config["output_root"])
    baseline_root = Path(baseline_config["output_root"])
    router_method = router_config["methods"][0]["name"]
    router_repeats = int(router_config["num_repeats"])
    baseline_repeats = int(baseline_config["num_repeats"])
    num_queries = int(router_config["expected_num_queries"])
    baseline_methods = {int(method["layer_count"]): method["name"] for method in baseline_config["methods"] if method["name"] != "oracle_prior_t1_32000_t2_200000"}
    required_layers = {0, 1, 2}
    if set(baseline_methods) != required_layers:
        raise ValueError(f"baseline must provide layers {sorted(required_layers)}")

    output_rows = []
    for workload in router_config["workloads"]:
        name = workload["name"]
        target = float(router_config["recall_thresholds"][name])
        router_warm, router_recalls, router_lsearch = load_run(
            router_root, router_method, name, router_repeats, target
        )
        used = route_count(
            router_root, router_method, name, router_repeats, num_queries
        )
        baselines = {}
        for layer, method in baseline_methods.items():
            baselines[layer] = load_run(
                baseline_root, method, name, baseline_repeats, target
            )

        row = {
            "workload": name,
            "mean_selectivity": workload["mean_selectivity"],
            "target_recall": target,
            "router_lsearch": router_lsearch,
            "router_recall_min": min(router_recalls),
            "router_recall_mean": statistics.mean(router_recalls),
            "router_warm_median_ms": statistics.median(router_warm),
            "router_warm_mean_ms": statistics.mean(router_warm),
            "router_warm_cv": statistics.stdev(router_warm) / statistics.mean(router_warm),
            "router_warm_p95_ms": percentile(router_warm, 0.95),
            "special_route_queries": used,
            "special_route_fraction": used / num_queries,
            "warm_repeats": len(router_warm),
        }
        for layer, label in ((0, "plain"), (1, "one_layer"), (2, "ungated_two_layer")):
            warm, _, lsearch = baselines[layer]
            row[f"{label}_lsearch"] = lsearch
            row[f"speedup_vs_{label}"] = (
                statistics.median(warm) / statistics.median(router_warm)
            )
            seed_text = f"{name}:{label}:router"
            seed = int(hashlib.sha256(seed_text.encode()).hexdigest()[:8], 16)
            low, high = paired_bootstrap_ratio(
                warm, router_warm, seed, args.bootstrap_samples
            )
            row[f"speedup_vs_{label}_ci95_low"] = low
            row[f"speedup_vs_{label}_ci95_high"] = high
        output_rows.append(row)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=list(output_rows[0]), lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(output_rows)
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
