#!/usr/bin/env python3
"""Generate the predeclared Amazon-x1 tuning matrix above 75%."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


WORKLOADS = (
    ("sel_80", "query_minlen1_nested_avgsel80pct", 0.8002369928),
    ("sel_85", "query_minlen1_nested_avgsel85pct", 0.8496712723),
    ("sel_90", "query_minlen1_nested_avgsel90pct", 0.9004633639),
    ("sel_95", "query_minlen1_nested_avgsel95pct", 0.9501978063),
    ("sel_967", "query_minlen1_label1_sel967pct", 0.9670165142),
    ("sel_100", "query_empty_full100pct", 1.0),
)
LAYER1 = (16000, 32000, 64000)
LAYER2 = ((8000, 100000), (8000, 200000),
          (16000, 100000), (16000, 200000), (16000, 400000),
          (32000, 100000), (32000, 200000), (32000, 400000))
FILTERED_L = {
    0: [80000, 95000, 102500, 110000, 125000],
    1: [500, 750, 1000, 1250, 1500, 2000, 3000],
    2: [250, 375, 500, 625, 750, 1000, 1500],
}
FULL_L = [50, 100, 200, 400, 800, 1600, 3200, 6400]


def grids(layer: int) -> dict[str, list[int]]:
    return {name: (FULL_L if name == "sel_100" else FILTERED_L[layer])
            for name, _, _ in WORKLOADS}


def method(repo: Path, name: str, layer: int, t1=None, t2=None) -> dict:
    item = {
        "name": name, "layer_count": layer,
        "special_block_search": layer > 0,
        "entry_group_provider": "cpu_bruteforce_els",
        "env": {"UNG_DISABLE_ELS_REUSE": "1",
                "UNG_DISABLE_CPU_ELS_WARMUP": "1"},
        "lsearch_values_by_workload": grids(layer),
    }
    if t1 is not None:
        item["t1"] = t1
    if t2 is not None:
        item["t2"] = t2
    if layer:
        item["block_index"] = str(
            repo / "runs/layer_tuning_build_amazon_x1" / name / "block_index")
        item["env"]["UNG_SPECIAL_BLOCK_ROOT_LABEL_COVERAGE"] = "1"
    return item


def make_config(repo: Path, repeats: int) -> dict:
    methods = [method(repo, "layer0_plain", 0)]
    methods += [method(repo, f"layer1_t1_{t1}", 1, t1=t1) for t1 in LAYER1]
    methods += [method(repo, f"layer2_t1_{t1}_t2_{t2}", 2, t1=t1, t2=t2)
                for t1, t2 in LAYER2]
    return {
        "search_app": str(repo / "build_ung_rel/apps/search_UNG_index"),
        "main_index": "/home/graphdb/FilterVectorResult/Amazon/index/Trie_block/index_files",
        "data_root": "/home/graphdb/FilterVectorData/Amazon",
        "gt_root": "/home/graphdb/FilterVectorResult/Amazon/GroundTruth",
        "output_root": str(repo / "runs/high_selectivity_tuning_coarse_amazon_x1"),
        "expected_source_fingerprint": "91d78580ae29f468",
        "expected_base_labels_sha256": "aec768bba7092af445252835be3f1ef7f708305ea646af19ae72738df3dd2f96",
        "expected_main_index_labels_sha256": "ddb3f616c27626afe6b20bf633aca5e9a1efd31d82bd505b163f59fc79f4dd56",
        "dataset": "Amazon", "scenario": "containment", "K": 10,
        "expected_num_queries": 1000, "num_threads": 100,
        "num_entry_points": 16, "num_repeats": repeats,
        "lsearch_values": [100], "require_stage_breakdown": True,
        "recall_thresholds": {name: 0.87 for name, _, _ in WORKLOADS},
        "workloads": [{"name": name, "query_dir": directory,
                       "mean_selectivity": selectivity}
                      for name, directory, selectivity in WORKLOADS],
        "boundary_reference_methods": [
            {key: item.get(key) for key in ("name", "layer_count", "t1", "t2")}
            for item in methods],
        "methods": methods,
        "study_design": {
            "scope": "focused predeclared grid above the measured 75% anchor",
            "shared_reference": ["layer1_t1_32000", "layer2_t1_16000_t2_200000"],
            "quality_rule": "minimum repeat Recall >= 0.87",
            "timing_rule": "warm-repeat batch median; repeat 0 excluded",
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--output", type=Path)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    output = args.output or Path(__file__).with_name(
        "config.amazon_x1_high_selectivity_tuning_coarse.json")
    output.write_text(json.dumps(make_config(args.repo, args.repeats), indent=2) + "\n")
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
