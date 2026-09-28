#!/usr/bin/env python3
"""Derive held-out detailed-profile cases from measured performance crossings."""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
DEFAULT_CAMPAIGN = HERE / "config.deadline_evidence_manifest.json"
DEFAULT_RUN_ROOT = REPO / "runs/deadline_profile_20260927"
DEFAULT_MANIFEST = HERE / "config.deadline_profile_manifest.json"
SELECTED_ROLES = {
    "zero_layer_baseline",
    "predeclared_degree_ratio_hierarchy_v1",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", dir=path.parent, text=True)
    temporary_path = Path(temporary)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2)
            stream.write("\n")
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)


def repeat_recalls(
    path: Path, cold_repeats: int, measured_repeats: int,
) -> dict[int, list[float]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    grouped: dict[int, dict[int, float]] = {}
    with path.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            repeat = int(row["Repeat"])
            lsearch = int(row["Lsearch"])
            if repeat in grouped.setdefault(lsearch, {}):
                raise ValueError(f"duplicate repeat/Lsearch in {path}: {repeat}/{lsearch}")
            grouped[lsearch][repeat] = float(row["Avg_Recall"])
    expected = set(range(cold_repeats + measured_repeats))
    warm_by_lsearch = {}
    for lsearch, recalls in grouped.items():
        if set(recalls) != expected:
            raise ValueError(
                f"incomplete repeat grid in {path} at L={lsearch}: "
                f"{sorted(recalls)} != {sorted(expected)}")
        warm_by_lsearch[lsearch] = [recalls[repeat] for repeat in range(
            cold_repeats, cold_repeats + measured_repeats)]
    return warm_by_lsearch


def profile_operating_point(
    path: Path, cold_repeats: int, measured_repeats: int,
    target_recall: float,
) -> tuple[str, int]:
    warm_by_lsearch = repeat_recalls(path, cold_repeats, measured_repeats)
    feasible = [
        lsearch for lsearch, recalls in warm_by_lsearch.items()
        if all(recall >= target_recall for recall in recalls)
    ]
    if feasible:
        return "crossing", min(feasible)
    if not warm_by_lsearch:
        raise ValueError(f"no measured points in {path}")
    best = max(
        warm_by_lsearch,
        key=lambda lsearch: (
            min(warm_by_lsearch[lsearch]),
            sum(warm_by_lsearch[lsearch]) / len(warm_by_lsearch[lsearch]),
            -lsearch,
        ),
    )
    return "no_crossing_best_measured", best


def conservative_crossing(
    path: Path, cold_repeats: int, measured_repeats: int,
    target_recall: float,
) -> int:
    status, lsearch = profile_operating_point(
        path, cold_repeats, measured_repeats, target_recall)
    if status != "crossing":
        raise ValueError(f"no conservative Recall crossing in {path}")
    return lsearch


def measurement_root(config: dict[str, Any]) -> Path:
    root = Path(config["output_root"])
    if config.get("pass_subdirs", False):
        root /= str(config.get("measurement_pass", "performance"))
    return root


def make_profile_config(
    source: dict[str, Any], source_path: Path, binary: Path,
    binary_hash: str, source_commit: str, output_root: Path,
    selected_method_names: set[str] | None = None,
    selected_workload_names: set[str] | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    protocol = source["protocol"]
    cold = int(protocol["cold_repeats"])
    measured = int(protocol["measured_repeats"])
    source_root = measurement_root(source)
    if selected_method_names is None:
        methods = [
            copy.deepcopy(method) for method in source["methods"]
            if method.get("selection_role") in SELECTED_ROLES
        ]
        roles = {str(method.get("selection_role")) for method in methods}
        if roles != SELECTED_ROLES or len(methods) != len(SELECTED_ROLES):
            raise ValueError(
                f"expected exactly one method for each selected role, got {roles}")
    else:
        methods = [
            copy.deepcopy(method) for method in source["methods"]
            if str(method["name"]) in selected_method_names
        ]
        found = {str(method["name"]) for method in methods}
        if found != selected_method_names:
            raise ValueError(
                f"selected methods are missing: {sorted(selected_method_names - found)}")

    workloads = [
        copy.deepcopy(workload) for workload in source["workloads"]
        if selected_workload_names is None
        or str(workload["name"]) in selected_workload_names
    ]
    found_workloads = {str(workload["name"]) for workload in workloads}
    if selected_workload_names is not None and found_workloads != selected_workload_names:
        raise ValueError(
            "selected workloads are missing: "
            f"{sorted(selected_workload_names - found_workloads)}")

    cases = []
    lsearch_union = set()
    for method in methods:
        by_workload = {}
        for workload in workloads:
            workload_name = str(workload["name"])
            target = float(source["recall_thresholds"][workload_name])
            detail_path = (
                source_root / str(method["name"]) / workload_name /
                "search_time_details.csv"
            )
            operating_status, lsearch = profile_operating_point(
                detail_path, cold, measured, target)
            by_workload[workload_name] = [lsearch]
            lsearch_union.add(lsearch)
            cases.append({
                "dataset": str(source["dataset"]),
                "method": str(method["name"]),
                "workload": workload_name,
                "lsearch": lsearch,
                "target_recall": target,
                "performance_status": operating_status,
                "source_details": str(detail_path),
                "source_details_sha256": sha256_file(detail_path),
            })
        method["lsearch_values_by_workload"] = by_workload

    result = copy.deepcopy(source)
    result["purpose"] = (
        "Detailed counters at deadline performance-pass conservative crossings; "
        "excluded from primary QPS."
    )
    result["measurement_pass"] = "profile"
    result["output_root"] = str(output_root)
    result["search_app"] = str(binary)
    result["expected_search_binary_sha256"] = binary_hash
    result["num_repeats"] = 3
    result["lsearch_values"] = sorted(lsearch_union)
    result["methods"] = methods
    result["workloads"] = workloads
    result["recall_thresholds"] = {
        name: value for name, value in source["recall_thresholds"].items()
        if name in found_workloads
    }
    result["require_stage_breakdown"] = True
    result["require_work_breakdown"] = True
    result["minimum_successful_child_seconds"] = 0
    result["protocol"] = {
        "phase": "profile",
        "cold_repeats": 1,
        "measured_repeats": 2,
        "recall_rule": "all_repeats",
        "bootstrap_samples": 10000,
        "paired_repeats": False,
    }
    result["selection_provenance"] = {
        "rule": (
            "reuse the smallest measured performance Lsearch whose every warm "
            "repeat reaches the configured Recall target"
        ),
        "source_config": str(source_path),
        "source_config_sha256": sha256_file(source_path),
        "source_measurement_pass": str(source.get("measurement_pass", "performance")),
    }
    result["instrumentation_provenance"] = {
        "purpose": "complete exact-gate and detailed search-work attribution",
        "source_commit": source_commit,
        "search_binary_sha256": binary_hash,
        "performance_binary_is_intentionally_unchanged": True,
    }
    return result, cases


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--campaign", type=Path, default=DEFAULT_CAMPAIGN)
    parser.add_argument("--search-binary", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    args = parser.parse_args()
    binary = args.search_binary.resolve()
    if not binary.is_file():
        raise FileNotFoundError(binary)
    binary_hash = sha256_file(binary)
    campaign_path = args.campaign.resolve()
    campaign = json.loads(campaign_path.read_text(encoding="utf-8"))
    run_root = args.run_root if args.run_root.is_absolute() else Path.cwd() / args.run_root
    configs = []
    all_cases = []
    for dataset in campaign["datasets"]:
        source_path = Path(dataset["search_config"])
        source = json.loads(source_path.read_text(encoding="utf-8"))
        slug = str(dataset["dataset"]).lower()
        output_path = HERE / f"config.deadline_profile_{slug}.json"
        profile, cases = make_profile_config(
            source, source_path, binary, binary_hash, args.source_commit,
            run_root / "heldout" / slug / "search",
        )
        atomic_json(output_path, profile)
        configs.append({
            "dataset": str(dataset["dataset"]),
            "config": str(output_path),
            "baseline_method": str(dataset["baseline_method"]),
            "cases": cases,
        })
        all_cases.extend(cases)
    manifest = {
        "schema_version": 1,
        "purpose": "Held-out detailed counters at measured performance crossings.",
        "run_root": str(run_root),
        "performance_campaign": str(campaign_path),
        "search_binary": str(binary),
        "search_binary_sha256": binary_hash,
        "source_commit": args.source_commit,
        "protocol": {
            "measurement_pass": "profile",
            "cold_repeats": 1,
            "measured_repeats": 2,
            "case_timeout_seconds": 3300,
            "primary_qps_source": "separate performance pass",
        },
        "configs": configs,
        "case_count": len(all_cases),
    }
    manifest_path = args.manifest if args.manifest.is_absolute() else Path.cwd() / args.manifest
    atomic_json(manifest_path, manifest)
    print(manifest_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
