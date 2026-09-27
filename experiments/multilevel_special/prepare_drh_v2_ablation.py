#!/usr/bin/env python3
"""Prepare a same-binary plain/DRH-v1/DRH-v2 performance ablation."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
DEFAULT_RUN_ROOT = REPO / "runs/drh_v2_20260927"
DEFAULT_MANIFEST = HERE / "config.drh_v2_manifest.json"

DATASETS = (
    {
        "dataset": "Amazon",
        "source": HERE / "config.authoritative_amazon_screen_emptyfix.json",
        "baseline": "l0_lng_entry_optimized_lng",
        "drh_v1": "l2_t1024_16384_lt_entry_optimized_lng_upper_routed",
        "threshold": 262144,
    },
    {
        "dataset": "Genome",
        "source": HERE / "config.deadline_heldout_genome_search.json",
        "baseline": "l0_lng_entry_optimized_lng",
        "drh_v1": "l2_t256_4096_lt_entry_optimized_lng_upper_routed",
        "threshold": 65536,
    },
    {
        "dataset": "Reviews",
        "source": HERE / "config.deadline_heldout_reviews_search.json",
        "baseline": "l0_lng_entry_optimized_lng",
        "drh_v1": "l2_t512_8192_lt_entry_optimized_lng_upper_routed",
        "threshold": 131072,
    },
    {
        "dataset": "VariousImg",
        "source": HERE / "config.deadline_heldout_variousimg_search.json",
        "baseline": "l0_lng_entry_optimized_lng",
        "drh_v1": "l2_t1024_16384_lt_entry_optimized_lng_upper_routed",
        "threshold": 262144,
    },
)


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


def unique_method(source: dict[str, Any], name: str) -> dict[str, Any]:
    matches = [method for method in source["methods"] if method["name"] == name]
    if len(matches) != 1:
        raise ValueError(f"expected one method named {name}, got {len(matches)}")
    return copy.deepcopy(matches[0])


def make_ablation_config(
    source: dict[str, Any], source_path: Path, binary: Path,
    binary_hash: str, source_commit: str, output_root: Path,
    baseline_name: str, drh_v1_name: str, threshold: int,
) -> dict[str, Any]:
    if threshold <= 0:
        raise ValueError("next-scale threshold must be positive")
    baseline = unique_method(source, baseline_name)
    drh_v1 = unique_method(source, drh_v1_name)
    if len(drh_v1.get("hierarchy_layers", [])) < 2:
        raise ValueError("DRH-v2 ablation requires a multilevel DRH-v1 method")

    drh_v1["selection_role"] = "degree_ratio_hierarchy_v1_exact_gate"
    drh_v1.setdefault("env", {})["UNG_SPECIAL_REQUIRE_UPPER_AUTHORIZATION"] = "1"

    drh_v2 = copy.deepcopy(drh_v1)
    drh_v2["name"] = drh_v1["name"] + "_next_scale_mass"
    drh_v2["selection_role"] = "degree_ratio_hierarchy_v2_next_scale_mass_gate"
    drh_v2["routing_policy"] = "highest_authorized_layer_direct_mass_at_least_next_scale"
    drh_v2["env"]["UNG_SPECIAL_UPPER_MIN_COVERED_POINTS"] = str(threshold)
    baseline_by_workload = baseline.get("lsearch_values_by_workload", {})
    drh_v1_by_workload = drh_v1.get("lsearch_values_by_workload", {})
    if baseline_by_workload or drh_v1_by_workload:
        drh_v2["lsearch_values_by_workload"] = {
            str(workload["name"]): sorted(set(
                baseline_by_workload.get(str(workload["name"]), []) +
                drh_v1_by_workload.get(str(workload["name"]), [])
            ))
            for workload in source["workloads"]
        }

    result = copy.deepcopy(source)
    result["purpose"] = (
        "Same-binary screen ablation of plain, exact-gated DRH-v1, and "
        "calibration-free next-scale-mass-gated DRH-v2."
    )
    result["measurement_pass"] = "performance"
    result["output_root"] = str(output_root)
    result["search_app"] = str(binary)
    result["expected_search_binary_sha256"] = binary_hash
    result["methods"] = [baseline, drh_v1, drh_v2]
    result["require_stage_breakdown"] = True
    result["require_work_breakdown"] = False
    result["minimum_successful_child_seconds"] = 0
    result["protocol"] = {
        "phase": "screen",
        "cold_repeats": 1,
        "measured_repeats": 2,
        "recall_rule": "all_repeats",
        "bootstrap_samples": 10000,
        "paired_repeats": False,
    }
    result["drh_v2_provenance"] = {
        "source_config": str(source_path),
        "source_config_sha256": sha256_file(source_path),
        "source_commit": source_commit,
        "search_binary_sha256": binary_hash,
        "rule": "T_next = (R / C) * T_last",
        "R": 64,
        "C": 4,
        "rho": 16,
        "next_scale_threshold": threshold,
        "query_calibrated": False,
        "latency_calibrated": False,
        "lsearch_grid": "union of the frozen plain and DRH-v1 grids",
    }
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--search-binary", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    args = parser.parse_args()

    binary = args.search_binary.resolve()
    if not binary.is_file():
        raise FileNotFoundError(binary)
    binary_hash = sha256_file(binary)
    run_root = args.run_root.resolve()
    configs = []
    for spec in DATASETS:
        source_path = Path(spec["source"]).resolve()
        source = json.loads(source_path.read_text(encoding="utf-8"))
        slug = str(spec["dataset"]).lower()
        config_path = HERE / f"config.drh_v2_{slug}.json"
        config = make_ablation_config(
            source, source_path, binary, binary_hash, args.source_commit,
            run_root / slug / "search", str(spec["baseline"]),
            str(spec["drh_v1"]), int(spec["threshold"]),
        )
        atomic_json(config_path, config)
        configs.append({
            "dataset": spec["dataset"],
            "config": str(config_path),
            "baseline_method": config["methods"][0]["name"],
            "drh_v1_method": config["methods"][1]["name"],
            "drh_v2_method": config["methods"][2]["name"],
            "next_scale_threshold": spec["threshold"],
            "workloads": [workload["name"] for workload in config["workloads"]],
        })

    manifest = {
        "schema_version": 1,
        "purpose": "Calibration-free DRH-v2 routing ablation.",
        "run_root": str(run_root),
        "source_commit": args.source_commit,
        "search_binary": str(binary),
        "search_binary_sha256": binary_hash,
        "configs": configs,
    }
    manifest_path = args.manifest.resolve()
    atomic_json(manifest_path, manifest)
    print(manifest_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
