#!/usr/bin/env python3
"""Compact the Reviews 0.2% order-sensitivity probes into auditable evidence."""

from __future__ import annotations

import argparse
import csv
import statistics
from pathlib import Path


WORKLOAD = "query_minlen5_cov0.1k"


def warm_metrics(root: Path, method: str) -> tuple[int, float, float, float, float]:
    run_dir = root / method / WORKLOAD
    with (run_dir / "search_time_details.csv").open(newline="") as stream:
        timing = list(csv.DictReader(stream))
    with (run_dir / "search_stage_details.csv").open(newline="") as stream:
        stages = list(csv.DictReader(stream))
    warm = [float(row["Time_ms"]) for row in timing if int(row["Repeat"]) > 0]
    query = [
        float(row["AverageQueryTotal_ms"])
        for row in stages
        if int(row["Repeat"]) > 0
    ]
    return (
        len(warm),
        statistics.median(warm),
        statistics.mean(warm),
        statistics.stdev(warm) / statistics.mean(warm) if len(warm) > 1 else 0.0,
        statistics.median(query),
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs-root", type=Path, default=Path("runs"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    cases = [
        ("historical_formal", "auto_policy_cross_dataset_reviews/formal",
         "layer0_plain", "auto_layer2_t1_8192_t2_131072",
         "plain then one-layer then ungated two-layer; not router"),
        ("router_formal", "auto_policy_structural_router_reviews_formal",
         None, "auto_qf_ssl_structural_router",
         "router only; compare with historical formal plain only as a noisy reference"),
        ("plain_then_router", "auto_policy_structural_router_reviews_critical",
         "layer0_plain_current_binary", "auto_qf_ssl_structural_router",
         "15 repeats, plain process executed before router process"),
        ("router_then_plain", "auto_policy_structural_router_reviews_reverse_probe",
         "layer0_plain_current_binary", "auto_qf_ssl_structural_router",
         "7 repeats, process order reversed"),
    ]
    rows = []
    for experiment, relative_root, plain, router, note in cases:
        root = args.runs_root / relative_root
        metrics = {}
        for role, method in (("plain", plain), ("router", router)):
            if method is not None:
                metrics[role] = warm_metrics(root, method)
        row = {"experiment": experiment, "note": note}
        for role in ("plain", "router"):
            values = metrics.get(role)
            for field, value in zip(
                ("warm_repeats", "batch_median_ms", "batch_mean_ms",
                 "batch_cv", "per_query_total_median_ms"),
                values or ("", "", "", "", ""),
            ):
                row[f"{role}_{field}"] = value
        if plain is not None:
            row["batch_speedup_plain_over_router"] = (
                metrics["plain"][1] / metrics["router"][1]
            )
            row["per_query_speedup_plain_over_router"] = (
                metrics["plain"][4] / metrics["router"][4]
            )
        else:
            row["batch_speedup_plain_over_router"] = ""
            row["per_query_speedup_plain_over_router"] = ""
        rows.append(row)

    root = args.runs_root / "auto_policy_structural_router_reviews_interleaved_probe"
    samples = {"plain": [], "router": []}
    query_samples = {"plain": [], "router": []}
    for role in samples:
        for index in range(1, 13):
            values = warm_metrics(root, f"{role}_{index:02d}")
            samples[role].append(values[1])
            query_samples[role].append(values[4])
    rows.append({
        "experiment": "interleaved_independent_processes",
        "note": "12 pairs, alternating process order; each process has one cold and one warm batch",
        "plain_warm_repeats": 12,
        "plain_batch_median_ms": statistics.median(samples["plain"]),
        "plain_batch_mean_ms": statistics.mean(samples["plain"]),
        "plain_batch_cv": statistics.stdev(samples["plain"]) / statistics.mean(samples["plain"]),
        "plain_per_query_total_median_ms": statistics.median(query_samples["plain"]),
        "router_warm_repeats": 12,
        "router_batch_median_ms": statistics.median(samples["router"]),
        "router_batch_mean_ms": statistics.mean(samples["router"]),
        "router_batch_cv": statistics.stdev(samples["router"]) / statistics.mean(samples["router"]),
        "router_per_query_total_median_ms": statistics.median(query_samples["router"]),
        "batch_speedup_plain_over_router": statistics.median(samples["plain"]) / statistics.median(samples["router"]),
        "per_query_speedup_plain_over_router": statistics.median(query_samples["plain"]) / statistics.median(query_samples["router"]),
    })

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
