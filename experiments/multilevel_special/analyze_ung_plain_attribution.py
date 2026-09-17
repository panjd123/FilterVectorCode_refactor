#!/usr/bin/env python3
"""Build a stage- and work-level attribution table for UNG versus plain."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import experiment_core


def safe_ratio(numerator, denominator):
    if numerator is None or denominator in (None, 0):
        return None
    return float(numerator) / float(denominator)


def attribution_rows(config: dict, baseline: str, candidate: str) -> list[dict]:
    summaries = experiment_core.summarize_experiment(config, baseline)
    indexed = {(row["method"], row["workload"]): row for row in summaries}
    result = []
    for workload in config["workloads"]:
        name = workload["name"]
        old = indexed[(baseline, name)]
        new = indexed[(candidate, name)]
        result.append({
            "workload": name,
            "mean_selectivity": workload.get("mean_selectivity"),
            "target_recall": new["target_recall"],
            "baseline_recall": old["recall_mean"],
            "candidate_recall": new["recall_mean"],
            "baseline_lsearch": old["lsearch"],
            "candidate_lsearch": new["lsearch"],
            "end_to_end_speedup": safe_ratio(old["warm_median_ms"], new["warm_median_ms"]),
            "els_speedup": safe_ratio(old.get("els_median_ms_per_query"), new.get("els_median_ms_per_query")),
            "graph_speedup": safe_ratio(old.get("graph_median_ms_per_query"), new.get("graph_median_ms_per_query")),
            "entry_count_ratio_candidate_over_baseline": safe_ratio(new.get("mean_num_entries"), old.get("mean_num_entries")),
            "distance_calc_reduction": 1.0 - safe_ratio(new.get("mean_dist_calcs"), old.get("mean_dist_calcs")),
            "node_visit_reduction": 1.0 - safe_ratio(new.get("mean_nodes_visited"), old.get("mean_nodes_visited")),
        })
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument("--baseline", default="ung_original_entry")
    parser.add_argument("--candidate", default="plain_bitset_lng_entry")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    config = experiment_core.load_config(args.config)
    rows = attribution_rows(config, args.baseline, args.candidate)
    output = args.output or Path(config["output_root"]) / "summary" / "attribution.csv"
    experiment_core.write_csv(output, rows)
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
