#!/usr/bin/env python3
"""Validate structural completeness and repeat consistency of a sweep."""

from __future__ import annotations

import argparse
import csv
import json
import shlex
import statistics
from pathlib import Path

import experiment_core
import run_selection_sweep


def expected_lsearch_values(config: dict, method: dict, workload: dict) -> set[int]:
    """Return the exact L grid used by the runner for one method."""
    return set(run_selection_sweep.lsearch_values_for(config, method, workload))


def command_options(path: Path) -> dict[str, list[str]]:
    tokens = shlex.split(path.read_text())
    options: dict[str, list[str]] = {}
    index = 1  # executable
    while index < len(tokens):
        token = tokens[index]
        if not token.startswith("--"):
            index += 1
            continue
        values = []
        index += 1
        while index < len(tokens) and not tokens[index].startswith("--"):
            values.append(tokens[index])
            index += 1
        options[token] = values
    return options


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=Path)
    args = parser.parse_args()
    config = experiment_core.load_config(args.config)
    protocol = experiment_core.protocol_for(config)
    root = Path(config["output_root"])
    expected_repeats = int(config["num_repeats"])
    problems: list[str] = []
    expected_case_keys = {
        (method["name"], workload["name"])
        for method in config["methods"] for workload in config["workloads"]
        if run_selection_sweep.method_enabled_for_workload(method, workload)
    }

    for method in config["methods"]:
        for workload in config["workloads"]:
            if not run_selection_sweep.method_enabled_for_workload(method, workload):
                continue
            expected_l = expected_lsearch_values(config, method, workload)
            name = f"{method['name']}/{workload['name']}"
            run_dir = root / method["name"] / workload["name"]
            summary_path = run_dir / "search_time_summary.csv"
            detail_path = run_dir / "search_time_details.csv"
            if not summary_path.is_file() or not detail_path.is_file():
                problems.append(f"{name}: missing summary or details")
                continue
            command_path = run_dir / "command.txt"
            environment_path = run_dir / "environment.json"
            if not command_path.is_file() or not environment_path.is_file():
                problems.append(f"{name}: missing executed command or environment")
                continue
            options = command_options(command_path)
            expected_options = {
                "--num_threads": [str(int(config["num_threads"]))],
                "--K": [str(int(config["K"]))],
                "--num_repeats": [str(expected_repeats)],
                "--entry_group_provider": [method["entry_group_provider"]],
                "--Lsearch": [str(value) for value in
                              run_selection_sweep.lsearch_values_for(
                                  config, method, workload)],
            }
            for option, expected in expected_options.items():
                if options.get(option) != expected:
                    problems.append(
                        f"{name}: executed {option}={options.get(option)} != {expected}")
            environment = json.loads(environment_path.read_text())
            if environment.get("UNG_DISABLE_ELS_REUSE") != "1":
                problems.append(f"{name}: executed with ELS query-result reuse enabled")
            if int(method.get("layer_count", 0)) > 0 and \
                    environment.get("UNG_SPECIAL_BLOCK_ROOT_LABEL_COVERAGE") != "1":
                problems.append(f"{name}: Special search did not use root-label coverage")
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
            if config.get("require_stage_breakdown", False):
                stage_path = run_dir / "search_stage_details.csv"
                if not stage_path.is_file():
                    problems.append(f"{name}: missing stage details")
                    continue
                with stage_path.open(newline="") as stream:
                    stages = list(csv.DictReader(stream))
                stage_counts = {
                    value: sum(int(row["Lsearch"]) == value for row in stages)
                    for value in expected_l
                }
                bad_stage_counts = {key: value for key, value in stage_counts.items()
                                    if value != expected_repeats}
                if bad_stage_counts:
                    problems.append(f"{name}: stage repeat counts {bad_stage_counts}")
                max_closure = max((abs(float(row["ClosureError_ms"])) for row in stages),
                                  default=float("inf"))
                if max_closure > 1e-6:
                    problems.append(f"{name}: stage closure error {max_closure} ms/query")
                if int(method.get("layer_count", 0)) == 0:
                    max_authorization = max(
                        (abs(float(row["AverageBlockAuthorization_ms"])) for row in stages),
                        default=float("inf"),
                    )
                    if max_authorization > 1e-9:
                        problems.append(
                            f"{name}: layer-0 authorization time is {max_authorization}")

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
            method = next(item for item in config["methods"]
                          if item["name"] == key[0])
            workload = next(item for item in config["workloads"]
                            if item["name"] == key[1])
            expected_manifest_l = sorted(expected_lsearch_values(
                config, method, workload))
            actual_manifest_l = sorted(int(value) for value in row.get("lsearch_values", []))
            if actual_manifest_l != expected_manifest_l:
                problems.append(f"{key}: manifest L grid mismatch")
            provenance = row.get("provenance", {})
            if provenance.get("base_labels_sha256") != config.get("expected_base_labels_sha256"):
                problems.append(f"{key}: base labels hash mismatch")
            if provenance.get("main_index_labels_sha256") != config.get("expected_main_index_labels_sha256"):
                problems.append(f"{key}: main-index labels hash mismatch")
            if provenance.get("expected_source_fingerprint") != config.get("expected_source_fingerprint"):
                problems.append(f"{key}: source fingerprint mismatch")
            if config.get("require_stage_breakdown", False):
                if not row.get("els_reuse_disabled"):
                    problems.append(f"{key}: ELS query-result reuse was not disabled")
                if not row.get("require_stage_breakdown"):
                    problems.append(f"{key}: stage breakdown was not required by runner")
            if "protocol" not in config:
                continue
            if row.get("protocol_phase") != protocol.phase:
                problems.append(f"{key}: protocol phase mismatch")
            if row.get("cold_repeats") != protocol.cold_repeats:
                problems.append(f"{key}: cold repeat count mismatch")
            if row.get("measured_repeats") != protocol.measured_repeats:
                problems.append(f"{key}: measured repeat count mismatch")

    if problems:
        print("VALIDATION FAILED")
        for problem in problems:
            print(f"- {problem}")
        return 1
    print("VALIDATION PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
