#!/usr/bin/env python3
"""Validate structural completeness and repeat consistency of a sweep."""

from __future__ import annotations

import argparse
import csv
import functools
import json
import os
import shlex
import statistics
from pathlib import Path

import experiment_core
import run_selection_sweep


@functools.lru_cache(maxsize=16)
def sha256_file_at_state(
    path: str, size: int, mtime_ns: int, ctime_ns: int,
) -> str:
    """Hash an executable once per observed filesystem state."""
    del size, mtime_ns, ctime_ns
    return run_selection_sweep.sha256_file(Path(path))


def executable_sha256(path: Path) -> str:
    state = path.stat()
    return sha256_file_at_state(
        str(path), state.st_size, state.st_mtime_ns, state.st_ctime_ns)


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


def result_evidence_is_complete(
    config: dict, method: dict, workload: dict, run_dir: Path,
) -> bool:
    """Apply the runner's full repeat and breakdown integrity contract."""
    pass_name = str(config.get("measurement_pass", "performance"))
    num_queries = workload.get(
        "num_queries", config.get("expected_num_queries"))
    return run_selection_sweep.result_is_complete(
        run_dir,
        run_selection_sweep.lsearch_values_for(config, method, workload),
        require_stage_breakdown=bool(config.get("require_stage_breakdown", False)),
        require_work_breakdown=bool(
            config.get("require_work_breakdown", False) or pass_name == "profile"),
        expected_repeats=int(config["num_repeats"]),
        expected_num_queries=(int(num_queries) if num_queries is not None else None),
    )


def expected_execution_contract(
    config: dict, method: dict, workload: dict, run_dir: Path,
) -> tuple[list[str], dict[str, str], str] | None:
    """Reconstruct the exact command/environment around an immutable snapshot."""
    command_path = run_dir / "command.txt"
    environment_path = run_dir / "environment.json"
    if not command_path.is_file() or not environment_path.is_file():
        return None
    try:
        executed = shlex.split(command_path.read_text())
        if not executed:
            return None
        binary = Path(executed[0]).resolve()
        snapshot_dir = (Path(config["output_root"]) / ".binary_snapshots").resolve()
        if not binary.is_file() or binary.parent != snapshot_dir:
            return None
        digest = executable_sha256(binary)
        if binary.name != f"search_UNG_index.{digest}":
            return None
        pinned = config.get("expected_search_binary_sha256")
        if pinned is not None and digest != pinned:
            return None
        expected_config = {**config, "search_app": str(binary)}
        expected_command = run_selection_sweep.build_command(
            expected_config, method, workload, run_dir)
        expected_environment = run_selection_sweep.clean_method_env(
            os.environ, expected_config, method)
    except (KeyError, OSError, TypeError, ValueError):
        return None
    return expected_command, expected_environment, digest


def result_execution_is_complete(
    config: dict, method: dict, workload: dict, run_dir: Path,
    contract: tuple[list[str], dict[str, str], str] | None = None,
) -> bool:
    """Bind complete result evidence to the exact executed binary and inputs."""
    contract = contract or expected_execution_contract(
        config, method, workload, run_dir)
    if contract is None:
        return False
    expected_command, expected_environment, binary_sha256 = contract
    pass_name = str(config.get("measurement_pass", "performance"))
    num_queries = workload.get(
        "num_queries", config.get("expected_num_queries"))
    return run_selection_sweep.result_is_complete(
        run_dir,
        run_selection_sweep.lsearch_values_for(config, method, workload),
        require_stage_breakdown=bool(config.get("require_stage_breakdown", False)),
        require_work_breakdown=bool(
            config.get("require_work_breakdown", False) or pass_name == "profile"),
        expected_repeats=int(config["num_repeats"]),
        expected_command=expected_command,
        expected_environment=expected_environment,
        expected_binary_sha256=binary_sha256,
        expected_num_queries=(int(num_queries) if num_queries is not None else None),
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=Path)
    args = parser.parse_args()
    config = experiment_core.load_config(args.config)
    protocol = experiment_core.protocol_for(config)
    root = Path(config["output_root"])
    pass_name = str(config.get("measurement_pass", "performance"))
    if config.get("pass_subdirs", False):
        root /= pass_name
    expected_repeats = int(config["num_repeats"])
    problems: list[str] = []
    executed_binary_hashes: dict[tuple[str, str], str] = {}
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
            if not result_evidence_is_complete(config, method, workload, run_dir):
                problems.append(
                    f"{name}: incomplete repeat grid or invalid stage/work evidence")
            command_path = run_dir / "command.txt"
            environment_path = run_dir / "environment.json"
            if not command_path.is_file() or not environment_path.is_file():
                problems.append(f"{name}: missing executed command or environment")
                continue
            contract = expected_execution_contract(
                config, method, workload, run_dir)
            if contract is None:
                problems.append(
                    f"{name}: invalid content-addressed executable contract")
            else:
                executed_binary_hashes[(method["name"], workload["name"])] = contract[2]
                if not result_execution_is_complete(
                        config, method, workload, run_dir, contract):
                    problems.append(
                        f"{name}: command, environment, or executable mismatch")
            options = command_options(command_path)
            expected_options = {
                "--num_threads": [str(int(config["num_threads"]))],
                "--K": [str(int(config["K"]))],
                "--num_repeats": [str(expected_repeats)],
                "--entry_group_strategy": [str(method.get(
                    "entry_strategy", method.get("entry_group_provider", "cpu_bruteforce_els")))],
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
            if len(experiment_core.hierarchy_layers(method)) > 0 and \
                    environment.get("UNG_SPECIAL_BLOCK_ROOT_LABEL_COVERAGE") != "1":
                problems.append(f"{name}: Special search did not use root-label coverage")
            if experiment_core.uses_orthogonal_method_schema(config):
                if environment.get("UNG_BASE_GROUP_TOPOLOGY") != method["base_topology"]:
                    problems.append(f"{name}: base topology environment mismatch")
                if environment.get("UNG_HIERARCHY_LAYERS", "") != \
                        experiment_core.encode_hierarchy_layers(method):
                    problems.append(f"{name}: hierarchy environment mismatch")
            expected_light = "0" if pass_name == "profile" else "1"
            if environment.get("UNG_SPECIAL_LIGHT_STATS") != expected_light:
                problems.append(f"{name}: statistics mode does not match {pass_name} pass")
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
                if len(experiment_core.hierarchy_layers(method)) == 0:
                    max_authorization = max(
                        (abs(float(row["AverageBlockAuthorization_ms"])) for row in stages),
                        default=float("inf"),
                    )
                    if max_authorization > 1e-9:
                        problems.append(
                            f"{name}: layer-0 authorization time is {max_authorization}")
            if config.get("require_work_breakdown", False) or pass_name == "profile":
                work_path = run_dir / "search_work_details.csv"
                if not work_path.is_file():
                    problems.append(f"{name}: missing work details")
                else:
                    with work_path.open(newline="") as stream:
                        work = list(csv.DictReader(stream))
                    work_counts = {
                        value: sum(int(row["Lsearch"]) == value for row in work)
                        for value in expected_l
                    }
                    bad_work_counts = {key: value for key, value in work_counts.items()
                                       if value != expected_repeats}
                    if bad_work_counts:
                        problems.append(f"{name}: work repeat counts {bad_work_counts}")

    manifest_path = Path(config["output_root"]) / (
        f"manifest_{pass_name}.json" if config.get("pass_subdirs", False)
        else "manifest.json")
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
        expected_binary_hash = config.get("expected_search_binary_sha256")
        if expected_binary_hash and binary_hashes != {expected_binary_hash}:
            problems.append(
                "search binary differs from campaign-pinned SHA256: "
                f"{sorted(str(x) for x in binary_hashes)} != "
                f"{expected_binary_hash}")
        elapsed_rows = [row for row in current.values()
                        if row.get("status") == "complete"]
        missing_elapsed = [key for key, row in current.items()
                           if row.get("status") == "complete"
                           and float(row.get("elapsed_seconds", 0)) <= 0]
        if missing_elapsed:
            problems.append(f"completed cases missing elapsed evidence: {missing_elapsed}")
        exact_seconds = sum(float(row.get("elapsed_seconds", 0))
                            for row in elapsed_rows
                            if row.get("elapsed_source") == "monotonic_child_wall")
        approximate_seconds = sum(float(row.get("elapsed_seconds", 0))
                                  for row in elapsed_rows
                                  if row.get("elapsed_source") ==
                                  "artifact_mtime_approximation")
        successful_seconds = exact_seconds + approximate_seconds
        print("successful_child_wall_seconds="
              f"{successful_seconds:.3f} exact={exact_seconds:.3f} "
              f"recovered_approximate={approximate_seconds:.3f}")
        required_seconds = float(config.get("minimum_successful_child_seconds", 0))
        if successful_seconds < required_seconds:
            problems.append(
                "successful child wall-time budget not met: "
                f"{successful_seconds:.3f} < {required_seconds:.3f} seconds")
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
            executed_binary_hash = executed_binary_hashes.get(key)
            if (executed_binary_hash is not None and
                    row.get("search_binary_sha256") != executed_binary_hash):
                problems.append(f"{key}: manifest/executable binary hash mismatch")
            provenance = row.get("provenance", {})
            if provenance.get("base_labels_sha256") != config.get("expected_base_labels_sha256"):
                problems.append(f"{key}: base labels hash mismatch")
            expected_main_hash = method.get(
                "expected_main_index_labels_sha256",
                config.get("expected_main_index_labels_sha256"))
            if expected_main_hash and provenance.get("main_index_labels_sha256") != expected_main_hash:
                problems.append(f"{key}: main-index labels hash mismatch")
            if provenance.get("expected_source_fingerprint") != config.get("expected_source_fingerprint"):
                problems.append(f"{key}: source fingerprint mismatch")
            if config.get("require_stage_breakdown", False):
                if not row.get("els_reuse_disabled"):
                    problems.append(f"{key}: ELS query-result reuse was not disabled")
                if not row.get("require_stage_breakdown"):
                    problems.append(f"{key}: stage breakdown was not required by runner")
            if row.get("measurement_pass", "performance") != pass_name:
                problems.append(f"{key}: measurement pass mismatch")
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
