#!/usr/bin/env python3
"""Generate search-quality checks for representative build outputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import experiment_core
import generate_authoritative_campaign as search_campaign
import generate_authoritative_build_configs as build_campaign


HERE = Path(__file__).resolve().parent
REPO = build_campaign.REPO
RUN_ROOT = build_campaign.RUN_ROOT
DATA_ROOT = build_campaign.DATA_ROOT
GT_ROOT = Path("/home/graphdb/FilterVectorResult/Amazon/GroundTruth")
QUERY_PROFILE_CONFIG = HERE / "config.authoritative_amazon_profile_emptyfix.json"


def labels_hash(index_dir: Path) -> str:
    return experiment_core.sha256_file(index_dir / "labels.txt")


def base_method(profile: str) -> dict:
    index = RUN_ROOT / "base_timing" / f"{profile}_measured_r0" / "index_files"
    return {
        "name": f"quality_base_{profile}",
        "main_index": str(index),
        "expected_main_index_labels_sha256": labels_hash(index),
        "base_topology": "lng",
        "entry_strategy": "optimized_lng",
        "hierarchy_layers": [],
        "special_block_search": False,
        "lsearch_values_by_workload": search_campaign.ZERO_LAYER_LSEARCH_BY_WORKLOAD,
        "env": search_campaign.common_method_env(),
    }


def hierarchy_method(structure: str, profile: str) -> dict:
    main = build_campaign.SOURCE_INDEX
    block = (RUN_ROOT / "hierarchy_timing" /
             f"{structure}_{profile}_measured_r0" / "block_index")
    layers = build_campaign.PREDECLARED_STRUCTURES[structure]
    return {
        "name": f"quality_{structure}_{profile}",
        "main_index": str(main),
        "expected_main_index_labels_sha256": labels_hash(main),
        "base_topology": "lng",
        "entry_strategy": "optimized_lng",
        "routing_policy": "require_upper_authorization",
        "hierarchy_layers": layers,
        "special_block_search": True,
        "block_index": str(block),
        "lsearch_values_by_workload": search_campaign.LAYERED_LSEARCH_BY_WORKLOAD,
        "env": search_campaign.common_method_env(),
    }


def make_config(search_app: Path, expected_search_binary_sha256: str) -> dict:
    if experiment_core.sha256_file(search_app) != expected_search_binary_sha256:
        raise ValueError("query-profile search binary does not match its pinned hash")
    methods = [base_method("original_cpu"), base_method("accelerated_gpu")]
    for structure in ("single_t1024_lng", "auto_drh_v1"):
        for profile in ("cpu", "full_gpu", "full_gpu_wmma"):
            methods.append(hierarchy_method(structure, profile))
    return {
        "schema_version": 2,
        "method_schema": "orthogonal_v2",
        "purpose": "Search-quality equivalence for representative CPU/GPU build outputs.",
        "measurement_pass": "performance",
        "pass_subdirs": True,
        "search_app": str(search_app),
        "expected_search_binary_sha256": expected_search_binary_sha256,
        "data_root": str(DATA_ROOT),
        "gt_root": str(GT_ROOT),
        "output_root": str(RUN_ROOT / "quality_screen"),
        "dataset": "Amazon",
        "expected_num_points": 602453,
        "expected_source_fingerprint": "91d78580ae29f468",
        "expected_base_labels_sha256":
            "aec768bba7092af445252835be3f1ef7f708305ea646af19ae72738df3dd2f96",
        "K": 10,
        "expected_num_queries": 1000,
        "num_threads": 100,
        "num_entry_points": 16,
        "num_repeats": 3,
        "lsearch_values": [100],
        "require_stage_breakdown": True,
        "require_work_breakdown": False,
        "protocol": {
            "phase": "screen", "cold_repeats": 1,
            "measured_repeats": 2, "recall_rule": "all_repeats",
            "bootstrap_samples": 10000, "paired_repeats": False,
        },
        "recall_thresholds": {
            name: 0.90 for name, *_ in search_campaign.WORKLOADS},
        "workloads": [
            {"name": name, "query_dir": query_dir,
             "mean_selectivity": selectivity, "num_queries": 1000}
            for name, query_dir, selectivity in search_campaign.WORKLOADS
        ],
        "methods": methods,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output", type=Path,
        default=HERE / "config.authoritative_amazon_build_quality_screen.json")
    parser.add_argument(
        "--query-profile-config", type=Path, default=QUERY_PROFILE_CONFIG)
    args = parser.parse_args()
    profile = json.loads(args.query_profile_config.read_text(encoding="utf-8"))
    expected_hash = profile.get("expected_search_binary_sha256")
    if not expected_hash:
        raise ValueError("query profile does not declare a pinned search binary")
    config = make_config(Path(profile["search_app"]), str(expected_hash))
    experiment_core.validate_config(config)
    args.output.write_text(json.dumps(config, indent=2) + "\n")
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
