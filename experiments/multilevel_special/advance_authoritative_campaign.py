#!/usr/bin/env python3
"""Advance an authoritative screen into crossing, formal, and profile configs."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
from pathlib import Path

import experiment_core


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def manifest_path(config: dict) -> Path:
    root = Path(config["output_root"])
    pass_name = str(config.get("measurement_pass", "performance"))
    return root / (f"manifest_{pass_name}.json"
                   if config.get("pass_subdirs", False) else "manifest.json")


def pin_source_binary(source: dict, result: dict) -> tuple[Path, str]:
    """Bind a derived phase to the source phase's immutable executable."""
    path = manifest_path(source)
    if not path.is_file():
        raise FileNotFoundError(path)
    manifest = json.loads(path.read_text(encoding="utf-8"))
    runs = manifest.get("runs", [])
    if not runs or any(row.get("status") != "complete" for row in runs):
        raise ValueError(f"{path}: source phase is not complete")
    hashes = {row.get("search_binary_sha256") for row in runs}
    if len(hashes) != 1 or None in hashes:
        raise ValueError(f"{path}: source phase has inconsistent binary hashes")
    digest = str(next(iter(hashes)))
    declared = source.get("expected_search_binary_sha256")
    if declared is not None and digest != declared:
        raise ValueError(f"{path}: source manifest differs from its pinned binary")
    snapshot = (Path(source["output_root"]) / ".binary_snapshots" /
                f"search_UNG_index.{digest}").resolve()
    if not snapshot.is_file() or sha256_file(snapshot) != digest:
        raise ValueError(f"{path}: source binary snapshot is missing or corrupt")
    result["search_app"] = str(snapshot)
    result["expected_search_binary_sha256"] = digest
    result.setdefault("selection_provenance", {})[
        "search_binary_sha256"] = digest
    return snapshot, digest


def source_run_dir(config: dict, method: dict, workload: dict) -> Path:
    root = Path(config["output_root"])
    if config.get("pass_subdirs", False):
        root /= str(config.get("measurement_pass", "performance"))
    return root / method["name"] / workload["name"]


def measured_recall_by_l(config: dict, method: dict, workload: dict) -> dict[int, float]:
    protocol = experiment_core.protocol_for(config)
    rows = experiment_core.read_csv(
        source_run_dir(config, method, workload) / "search_time_details.csv")
    grouped: dict[int, list[float]] = {}
    for row in rows:
        if int(row["Repeat"]) < protocol.cold_repeats:
            continue
        grouped.setdefault(int(row["Lsearch"]), []).append(float(row["Avg_Recall"]))
    expected = protocol.measured_repeats
    for value, recalls in grouped.items():
        if len(recalls) != expected:
            raise ValueError(
                f"{method['name']}/{workload['name']} L={value}: "
                f"expected {expected} measured repeats, found {len(recalls)}")
    return {value: min(recalls) for value, recalls in grouped.items()}


def crossing_grid(values: dict[int, float], threshold: float, k: int,
                  max_lsearch: int) -> list[int] | None:
    ordered = sorted(values)
    passing = [value for value in ordered if values[value] >= threshold]
    if passing:
        upper = passing[0]
        failing = [value for value in ordered if value < upper and values[value] < threshold]
        lower = failing[-1] if failing else max(k, upper // 4)
    else:
        lower = ordered[-1]
        if lower >= max_lsearch:
            return None
        upper = min(max_lsearch, max(lower + 1, math.ceil(lower * 1.5)))
        if upper == lower:
            raise ValueError(f"no Recall crossing before max_lsearch={max_lsearch}")
    width = upper - lower
    step = max(1, math.ceil(width / 6))
    grid = list(range(lower, upper + 1, step))
    if grid[-1] != upper:
        grid.append(upper)
    return sorted(set(grid))


def selected_lsearch(config: dict, method: dict, workload: dict) -> int:
    threshold = float(config["recall_thresholds"][workload["name"]])
    values = measured_recall_by_l(config, method, workload)
    passing = [value for value, recall in sorted(values.items()) if recall >= threshold]
    if not passing:
        raise ValueError(
            f"{method['name']}/{workload['name']} has no measured Recall crossing")
    return passing[0]


def selected_lsearch_or_none(config: dict, method: dict,
                             workload: dict) -> list[int] | None:
    threshold = float(config["recall_thresholds"][workload["name"]])
    values = measured_recall_by_l(config, method, workload)
    passing = [value for value, recall in sorted(values.items())
               if recall >= threshold]
    return [passing[0]] if passing else None


def with_method_grids(config: dict, resolver) -> list[dict]:
    methods = copy.deepcopy(config["methods"])
    for method in methods:
        grids = {}
        eligible = []
        previously_enabled = method.get("enabled_workloads")
        for workload in config["workloads"]:
            if (previously_enabled is not None and
                    workload["name"] not in previously_enabled):
                continue
            values = resolver(method, workload)
            if values is None:
                continue
            grids[workload["name"]] = values
            eligible.append(workload["name"])
        method["lsearch_values_by_workload"] = grids
        if (previously_enabled is not None or
                len(eligible) != len(config["workloads"])):
            method["enabled_workloads"] = eligible
    return methods


def exhausted_budget_cases(config: dict, max_lsearch: int) -> list[dict]:
    result = []
    for method in config["methods"]:
        enabled = method.get("enabled_workloads")
        for workload in config["workloads"]:
            if enabled is not None and workload["name"] not in enabled:
                continue
            values = measured_recall_by_l(config, method, workload)
            threshold = float(config["recall_thresholds"][workload["name"]])
            if max(values) >= max_lsearch and max(values.values()) < threshold:
                result.append({
                    "method": method["name"],
                    "workload": workload["name"],
                    "max_measured_lsearch": max(values),
                    "max_measured_recall": max(values.values()),
                    "target_recall": threshold,
                    "reason": "no measured Recall crossing at the declared Lsearch budget",
                })
    return result


def no_crossing_cases(config: dict) -> list[dict]:
    result = []
    for method in config["methods"]:
        enabled = method.get("enabled_workloads")
        for workload in config["workloads"]:
            if enabled is not None and workload["name"] not in enabled:
                continue
            values = measured_recall_by_l(config, method, workload)
            threshold = float(config["recall_thresholds"][workload["name"]])
            passing = [value for value, recall in values.items()
                       if recall >= threshold]
            if passing:
                continue
            max_lsearch = max(values)
            result.append({
                "method": method["name"],
                "workload": workload["name"],
                "max_measured_lsearch": max_lsearch,
                "recall_at_max_lsearch": values[max_lsearch],
                "max_measured_recall": max(values.values()),
                "target_recall": threshold,
                "reason": "no Recall crossing in the declared screen/crossing grid",
            })
    return result


def phase_output_root(source: dict, phase: str) -> str:
    root = Path(source["output_root"])
    for suffix in ("screen", "crossing", "formal", "profile"):
        marker = "_" + suffix
        if root.name.endswith(marker):
            return str(root.with_name(root.name[:-len(marker)] + "_" + phase))
    return str(root.with_name(root.name + "_" + phase))


def make_crossing(source: dict) -> dict:
    config = copy.deepcopy(source)
    max_lsearch = int(source.get("max_lsearch", source.get("expected_num_points", 1 << 30)))
    config["output_root"] = phase_output_root(source, "crossing")
    config["num_repeats"] = 3
    config["campaign_minimum_successful_child_seconds"] = int(
        source.get("campaign_minimum_successful_child_seconds",
                   source.get("minimum_successful_child_seconds", 0)))
    config["minimum_successful_child_seconds"] = 0
    config["protocol"] = {
        "phase": "crossing", "cold_repeats": 1, "measured_repeats": 2,
        "recall_rule": "all_repeats", "bootstrap_samples": 10000,
        "paired_repeats": False,
    }
    config["methods"] = with_method_grids(
        source,
        lambda method, workload: crossing_grid(
            measured_recall_by_l(source, method, workload),
            float(source["recall_thresholds"][workload["name"]]),
            int(source["K"]), max_lsearch),
    )
    config["selection_provenance"] = {
        "source_output_root": source["output_root"],
        "rule": "six-interval measured refinement between the last failing and first passing screen point",
        "excluded_no_crossing": exhausted_budget_cases(source, max_lsearch),
    }
    return config


def make_formal(source: dict) -> dict:
    config = copy.deepcopy(source)
    config["output_root"] = phase_output_root(source, "formal")
    config["num_repeats"] = 16
    config["campaign_minimum_successful_child_seconds"] = int(
        source.get("campaign_minimum_successful_child_seconds",
                   source.get("minimum_successful_child_seconds", 0)))
    config["minimum_successful_child_seconds"] = 0
    config["protocol"] = {
        "phase": "formal", "cold_repeats": 1, "measured_repeats": 15,
        "recall_rule": "all_repeats", "bootstrap_samples": 20000,
        "paired_repeats": False,
    }
    config["methods"] = with_method_grids(
        source, lambda method, workload: selected_lsearch_or_none(
            source, method, workload))
    config["selection_provenance"] = {
        "source_output_root": source["output_root"],
        "rule": "smallest explicitly measured Lsearch whose every warm repeat reaches the Recall threshold",
        "excluded_no_crossing": no_crossing_cases(source),
    }
    return config


def make_profile(formal: dict) -> dict:
    config = copy.deepcopy(formal)
    config["measurement_pass"] = "profile"
    config["num_repeats"] = 4
    config["campaign_minimum_successful_child_seconds"] = int(
        formal.get("campaign_minimum_successful_child_seconds",
                   formal.get("minimum_successful_child_seconds", 0)))
    config["minimum_successful_child_seconds"] = 0
    config["require_work_breakdown"] = True
    config["protocol"] = {
        "phase": "profile", "cold_repeats": 1, "measured_repeats": 3,
        "recall_rule": "all_repeats", "bootstrap_samples": 10000,
        "paired_repeats": False,
    }
    config["selection_provenance"] = {
        "source_output_root": formal["output_root"],
        "rule": "profile pass reuses each formal method/workload Lsearch and is excluded from primary QPS",
    }
    return config


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=("crossing", "formal", "profile"))
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    source = experiment_core.load_config(args.source)
    if args.phase == "crossing":
        result = make_crossing(source)
    elif args.phase == "formal":
        result = make_formal(source)
    else:
        result = make_profile(source)
    pin_source_binary(source, result)
    experiment_core.validate_config(result)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
