#!/usr/bin/env python3
"""Summarize formal exact-level runs and paired bootstrap speedups."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import random
import statistics
from pathlib import Path


def percentile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    pos = (len(ordered) - 1) * q
    lo, hi = math.floor(pos), math.ceil(pos)
    if lo == hi:
        return ordered[lo]
    return ordered[lo] * (hi - pos) + ordered[hi] * (pos - lo)


def bootstrap_ratio(a: list[float], b: list[float], seed: int, n: int = 100000) -> tuple[float, float]:
    """Paired bootstrap CI for median(a) / median(b)."""
    if len(a) != len(b):
        raise ValueError("paired samples must have equal length")
    rng = random.Random(seed)
    ratios = []
    for _ in range(n):
        indices = [rng.randrange(len(a)) for _ in a]
        ratios.append(statistics.median(a[i] for i in indices) / statistics.median(b[i] for i in indices))
    return percentile(ratios, .025), percentile(ratios, .975)


def main() -> int:
    repo = Path("/home/sunyahui/worktrees/FilterVectorCode_multilevel_special")
    config_path = repo / "experiments/multilevel_special/config.auto_policy_formal_exact_level.json"
    config = json.loads(config_path.read_text())
    root = Path(config["output_root"])
    methods = {m["name"]: m for m in config["methods"]}
    workloads = {w["name"]: w for w in config["workloads"]}
    threshold = config["recall_thresholds"]
    samples: dict[tuple[str, str], list[float]] = {}
    rows = []
    for method_name, method in methods.items():
        for workload_name, workload in workloads.items():
            path = root / method_name / workload_name / "search_time_details.csv"
            with path.open(newline="") as stream:
                raw = list(csv.DictReader(stream))
            if len(raw) != 7 or sorted(int(r["Repeat"]) for r in raw) != list(range(7)):
                raise ValueError(f"expected repeats 0..6: {path}")
            recalls = [float(r["Avg_Recall"]) for r in raw]
            warm = [float(r["Time_ms"]) for r in raw if int(r["Repeat"]) > 0]
            if min(recalls) < float(threshold[workload_name]):
                raise ValueError(f"formal Recall failure: {method_name}/{workload_name}")
            samples[(method_name, workload_name)] = warm
            mean = statistics.mean(warm)
            rows.append({
                "method": method_name, "layer_count": method["layer_count"],
                "t1": method.get("t1", ""), "t2": method.get("t2", ""),
                "workload": workload_name, "mean_selectivity": workload["mean_selectivity"],
                "target_recall": threshold[workload_name], "lsearch": int(raw[0]["Lsearch"]),
                "recall_min": min(recalls), "recall_mean": statistics.mean(recalls),
                "warm_median_ms": statistics.median(warm), "warm_mean_ms": mean,
                "warm_cv": statistics.stdev(warm) / mean,
                "warm_p50_ms": percentile(warm, .5), "warm_p95_ms": percentile(warm, .95),
            })
    by_key = {(r["method"], r["workload"]): r for r in rows}
    plain = "layer0_plain"
    one = "auto_mass_ladder_layer1_t1_8192"
    auto = "auto_mass_ladder_t1_8192_t2_131072"
    oracle = "oracle_prior_t1_32000_t2_200000"
    for row in rows:
        workload = row["workload"]
        base = by_key[(plain, workload)]["warm_median_ms"]
        row["speedup_vs_plain"] = base / row["warm_median_ms"]
        if row["method"] == auto:
            one_times = samples[(one, workload)]
            auto_times = samples[(auto, workload)]
            row["speedup_vs_same_t1_one_layer"] = statistics.median(one_times) / statistics.median(auto_times)
            lo, hi = bootstrap_ratio(one_times, auto_times, int(hashlib.sha256(workload.encode()).hexdigest()[:8], 16))
            row["same_t1_speedup_ci95_low"] = lo
            row["same_t1_speedup_ci95_high"] = hi
            oracle_ms = by_key[(oracle, workload)]["warm_median_ms"]
            row["latency_regret_vs_tuned_control"] = row["warm_median_ms"] / oracle_ms - 1.0
    out = root / "summary"
    out.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0])
    for extra in ("speedup_vs_plain", "speedup_vs_same_t1_one_layer",
                  "same_t1_speedup_ci95_low", "same_t1_speedup_ci95_high",
                  "latency_regret_vs_tuned_control"):
        if extra not in fields: fields.append(extra)
    with (out / "formal_results.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader(); writer.writerows(rows)
    print(out / "formal_results.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
