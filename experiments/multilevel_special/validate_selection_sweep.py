#!/usr/bin/env python3
"""Validate structural completeness and repeat consistency of a sweep."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=Path)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    root = Path(config["output_root"])
    expected_l = {int(value) for value in config["lsearch_values"]}
    expected_repeats = int(config["num_repeats"])
    problems: list[str] = []
    expected_case_keys = {
        (method["name"], workload["name"])
        for method in config["methods"] for workload in config["workloads"]
    }

    for method in config["methods"]:
        for workload in config["workloads"]:
            name = f"{method['name']}/{workload['name']}"
            run_dir = root / method["name"] / workload["name"]
            summary_path = run_dir / "search_time_summary.csv"
            detail_path = run_dir / "search_time_details.csv"
            if not summary_path.is_file() or not detail_path.is_file():
                problems.append(f"{name}: missing summary or details")
                continue
            with summary_path.open(newline="") as stream:
                summary = list(csv.DictReader(stream))
            with detail_path.open(newline="") as stream:
                details = list(csv.DictReader(stream))
            seen_l = {int(row["Lsearch"]) for row in summary}
            if seen_l != expected_l:
                problems.append(f"{name}: L grid mismatch")
            repeat_counts = {
                value: sum(int(row["Lsearch"]) == value for row in details)
                for value in expected_l
            }
            bad_counts = {key: value for key, value in repeat_counts.items()
                          if value != expected_repeats}
            if bad_counts:
                problems.append(f"{name}: repeat counts {bad_counts}")
            recall_spreads = []
            for value in expected_l:
                recalls = [float(row["Avg_Recall"]) for row in details
                           if int(row["Lsearch"]) == value]
                if recalls:
                    recall_spreads.append((max(recalls) - min(recalls), value, recalls))
            relative_deltas = []
            for value in expected_l:
                warm = [float(row["Time_ms"]) for row in details
                        if int(row["Lsearch"]) == value and int(row["Repeat"]) > 0]
                if len(warm) > 1 and statistics.mean(warm) > 0:
                    relative_deltas.append(
                        (max(warm) - min(warm)) / statistics.mean(warm)
                    )
            maximum_recall = max(float(row["Average_Recall"]) for row in summary)
            maximum_delta = max(relative_deltas) if relative_deltas else 0.0
            maximum_recall_spread, spread_l, spread_values = max(recall_spreads)
            print(f"{name:28s} summary={len(summary):2d} details={len(details):3d} "
                  f"max_recall={maximum_recall:.6f} warm_max_delta={maximum_delta:.3f} "
                  f"recall_max_spread={maximum_recall_spread:.6f}@L{spread_l}")
            # Search may contain randomized tie-breaking.  Treat small recall
            # variation as a measured distribution, but reject instability
            # large enough to invalidate a 0.01-quality operating point.
            if maximum_recall_spread > 0.01:
                problems.append(
                    f"{name}: recall spread {maximum_recall_spread:.6f} at L={spread_l}: "
                    f"{spread_values}"
                )

    manifest_path = root / "manifest.json"
    if not manifest_path.is_file():
        problems.append("missing manifest")
    else:
        manifest = json.loads(manifest_path.read_text())
        expected_cases = len(expected_case_keys)
        runs = manifest.get("runs", [])
        current = {
            (row.get("method"), row.get("workload")): row for row in runs
            if (row.get("method"), row.get("workload")) in expected_case_keys
        }
        if len(current) != expected_cases or any(
                row.get("status") != "complete" for row in current.values()):
            problems.append(f"manifest incomplete: {len(runs)}/{expected_cases}")
        binary_hashes = {row.get("search_binary_sha256") for row in current.values()}
        if len(binary_hashes) != 1 or None in binary_hashes:
            problems.append(f"search binary hash mismatch: {sorted(str(x) for x in binary_hashes)}")
        for key, row in current.items():
            provenance = row.get("provenance", {})
            if provenance.get("base_labels_sha256") != config.get("expected_base_labels_sha256"):
                problems.append(f"{key}: base labels hash mismatch")
            if provenance.get("main_index_labels_sha256") != config.get("expected_main_index_labels_sha256"):
                problems.append(f"{key}: main-index labels hash mismatch")
            if provenance.get("expected_source_fingerprint") != config.get("expected_source_fingerprint"):
                problems.append(f"{key}: source fingerprint mismatch")

    if problems:
        print("VALIDATION FAILED")
        for problem in problems:
            print(f"- {problem}")
        return 1
    print("VALIDATION PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
