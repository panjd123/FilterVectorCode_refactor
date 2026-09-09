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
PLAIN_EXTENSION_L = {
    "sel_85": [130000, 135000],
    "sel_90": [140000, 155000],
    "sel_95": [145000, 165000],
    "sel_967": [150000, 175000],
    "sel_100": [12500, 25000, 50000, 75000, 100000, 125000, 150000, 175000],
}
FULL_PLAIN_EXTENSION_L = [200000, 225000, 250000, 275000, 300000]
STRUCTURE_EXTENSION = (
    ("layer1_t1_8000", 1, 8000, None),
    ("layer1_t1_128000", 1, 128000, None),
    ("layer2_t1_16000_t2_25000", 2, 16000, 25000),
    ("layer2_t1_16000_t2_50000", 2, 16000, 50000),
    ("layer2_t1_32000_t2_50000", 2, 32000, 50000),
    ("layer2_t1_32000_t2_80000", 2, 32000, 80000),
    ("layer2_t1_64000_t2_100000", 2, 64000, 100000),
    ("layer2_t1_64000_t2_200000", 2, 64000, 200000),
    ("layer2_t1_64000_t2_400000", 2, 64000, 400000),
)
STRUCTURE_BOUNDARY_EXTENSION = (
    ("layer1_t1_256000", 1, 256000, None),
)
STRUCTURE_ENDPOINT_EXTENSION = (
    ("layer1_t1_700000", 1, 700000, None),
)
FORMAL_BOUNDARY_L = {
    "layer1_t1_128000": {
        "sel_90": [125, 156, 188, 219],
        "sel_95": [10, 25, 50, 75, 100, 125, 156, 188, 219],
        "sel_967": [10, 25, 50, 75, 100, 125, 156, 188, 219],
    },
    "layer2_t1_16000_t2_400000": {
        "sel_85": [25, 50, 75, 100],
        "sel_90": [25, 50, 75, 100],
        "sel_95": [10, 15, 20, 25, 50, 75, 100],
        "sel_967": [10, 15, 20, 25, 50, 75, 100],
        # Search requires Lsearch >= K; K=10 is the natural lower endpoint.
        "sel_100": [10, 15, 20],
    },
    # New lower-T1 structural guard. It is measured on every workload so it
    # can compete fairly for the shared two-level deployment configuration.
    "layer2_t1_8000_t2_400000": {
        "sel_80": [250, 281, 312, 344, 375, 500],
        "sel_85": [25, 50, 75, 100, 125, 156, 188, 219, 250],
        "sel_90": [25, 50, 75, 100, 125, 156, 188, 219, 250],
        "sel_95": [25, 50, 75, 100, 125, 156, 188, 219, 250],
        "sel_967": [25, 50, 75, 100, 125, 156, 188, 219, 250],
        "sel_100": [10, 15, 20, 25, 31, 38, 44, 50, 100],
    },
    # Upper-T2 structural guard between the selected 400k threshold and the
    # above-cardinality natural endpoint.
    "layer2_t1_16000_t2_500000": {
        "sel_80": [250, 281, 312, 344, 375, 500],
        "sel_85": [25, 50, 75, 100, 125, 156, 188, 219, 250],
        "sel_90": [25, 50, 75, 100, 125, 156, 188, 219, 250],
        "sel_95": [25, 50, 75, 100, 125, 156, 188, 219, 250],
        "sel_967": [25, 50, 75, 100, 125, 156, 188, 219, 250],
        "sel_100": [10, 15, 20, 25, 31, 38, 44, 50, 100],
    },
}


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
        "structure_threshold_domain_max": 602453,
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


def make_plain_extension_config(repo: Path, repeats: int) -> dict:
    config = make_config(repo, repeats)
    config["output_root"] = str(
        repo / "runs/high_selectivity_tuning_plain_extension_amazon_x1")
    plain = method(repo, "layer0_plain", 0)
    plain["enabled_workloads"] = list(PLAIN_EXTENSION_L)
    plain["lsearch_values_by_workload"] = PLAIN_EXTENSION_L
    config["methods"] = [plain]
    config["boundary_reference_methods"] = [
        {"name": "layer0_plain", "layer_count": 0, "t1": None, "t2": None}]
    config["study_design"]["scope"] = (
        "plain-only Lsearch boundary closure after the predeclared coarse sweep")
    return config


def make_structure_extension_config(repo: Path, repeats: int) -> dict:
    config = make_config(repo, repeats)
    config["output_root"] = str(
        repo / "runs/high_selectivity_tuning_structure_extension_amazon_x1")
    config["methods"] = [
        method(repo, name, layer, t1=t1, t2=t2)
        for name, layer, t1, t2 in STRUCTURE_EXTENSION
    ]
    all_structures = make_config(repo, repeats)["boundary_reference_methods"]
    all_structures.extend(
        {"name": name, "layer_count": layer, "t1": t1, "t2": t2}
        for name, layer, t1, t2 in STRUCTURE_EXTENSION)
    config["boundary_reference_methods"] = all_structures
    config["study_design"]["scope"] = (
        "existing-index structural boundary closure after the predeclared coarse sweep")
    return config


def make_structure_boundary_extension_config(repo: Path, repeats: int) -> dict:
    config = make_config(repo, repeats)
    config["output_root"] = str(
        repo / "runs/high_selectivity_tuning_structure_boundary_extension_amazon_x1")
    config["methods"] = [
        method(repo, name, layer, t1=t1, t2=t2)
        for name, layer, t1, t2 in STRUCTURE_BOUNDARY_EXTENSION
    ]
    all_structures = make_structure_extension_config(
        repo, repeats)["boundary_reference_methods"]
    all_structures.extend(
        {"name": name, "layer_count": layer, "t1": t1, "t2": t2}
        for name, layer, t1, t2 in STRUCTURE_BOUNDARY_EXTENSION)
    config["boundary_reference_methods"] = all_structures
    config["study_design"]["scope"] = (
        "single-level upper-T1 boundary closure after the first structural extension")
    return config


def make_structure_endpoint_extension_config(repo: Path, repeats: int) -> dict:
    config = make_config(repo, repeats)
    config["output_root"] = str(
        repo / "runs/high_selectivity_tuning_structure_endpoint_extension_amazon_x1")
    config["methods"] = [
        method(repo, name, layer, t1=t1, t2=t2)
        for name, layer, t1, t2 in STRUCTURE_ENDPOINT_EXTENSION
    ]
    all_structures = make_structure_boundary_extension_config(
        repo, repeats)["boundary_reference_methods"]
    all_structures.extend(
        {"name": name, "layer_count": layer, "t1": t1, "t2": t2}
        for name, layer, t1, t2 in STRUCTURE_ENDPOINT_EXTENSION)
    config["boundary_reference_methods"] = all_structures
    config["study_design"]["scope"] = (
        "single-level natural endpoint closure above the dataset cardinality")
    return config


def make_full_plain_extension_config(repo: Path, repeats: int) -> dict:
    config = make_config(repo, repeats)
    config["output_root"] = str(
        repo / "runs/high_selectivity_tuning_full_plain_extension_amazon_x1")
    plain = method(repo, "layer0_plain", 0)
    plain["enabled_workloads"] = ["sel_100"]
    plain["lsearch_values_by_workload"] = {"sel_100": FULL_PLAIN_EXTENSION_L}
    config["methods"] = [plain]
    config["boundary_reference_methods"] = [
        {"name": "layer0_plain", "layer_count": 0, "t1": None, "t2": None}]
    config["study_design"]["scope"] = (
        "100-percent unfiltered plain Lsearch boundary closure")
    return config


def make_formal_boundary_extension_config(repo: Path, repeats: int) -> dict:
    """Close only boundaries that can change the shared formal result."""
    config = make_config(repo, repeats)
    config["output_root"] = str(
        repo / "runs/high_selectivity_formal_boundary_extension_amazon_x1")
    identities = {
        "layer1_t1_128000": (1, 128000, None),
        "layer2_t1_16000_t2_400000": (2, 16000, 400000),
        "layer2_t1_8000_t2_400000": (2, 8000, 400000),
        "layer2_t1_16000_t2_500000": (2, 16000, 500000),
    }
    methods = []
    for name, grids_by_workload in FORMAL_BOUNDARY_L.items():
        layer, t1, t2 = identities[name]
        item = method(repo, name, layer, t1=t1, t2=t2)
        item["enabled_workloads"] = list(grids_by_workload)
        item["lsearch_values_by_workload"] = grids_by_workload
        methods.append(item)
    config["methods"] = methods
    reference = list(make_structure_endpoint_extension_config(
        repo, repeats)["boundary_reference_methods"] )
    reference.append({
        "name": "layer2_t1_8000_t2_400000", "layer_count": 2,
        "t1": 8000, "t2": 400000,
    })
    reference.append({
        "name": "layer2_t1_16000_t2_500000", "layer_count": 2,
        "t1": 16000, "t2": 500000,
    })
    # A threshold above the 602,453-point cardinality cannot create an upper
    # block. It closes the upper axis but is not a performance candidate.
    reference.append({
        "name": "layer2_t1_16000_t2_700000", "layer_count": 2,
        "t1": 16000, "t2": 700000,
    })
    config["boundary_reference_methods"] = reference
    config["study_design"]["scope"] = (
        "seven-repeat closure of shared formal Lsearch and structural boundaries")
    config["study_design"]["boundary_policy"] = (
        "measure the lower-T1 guard; use the above-cardinality T2=700000 "
        "natural endpoint only as a structural boundary reference")
    return config


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--output", type=Path)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--phase", choices=("coarse", "plain-extension",
                                               "full-plain-extension",
                                               "structure-extension",
                                               "structure-boundary-extension",
                                               "structure-endpoint-extension",
                                               "formal-boundary-extension"),
                        default="coarse")
    args = parser.parse_args()
    builders = {
        "coarse": make_config,
        "plain-extension": make_plain_extension_config,
        "full-plain-extension": make_full_plain_extension_config,
        "structure-extension": make_structure_extension_config,
        "structure-boundary-extension": make_structure_boundary_extension_config,
        "structure-endpoint-extension": make_structure_endpoint_extension_config,
        "formal-boundary-extension": make_formal_boundary_extension_config,
    }
    default_names = {
        "coarse": "config.amazon_x1_high_selectivity_tuning_coarse.json",
        "plain-extension": "config.amazon_x1_high_selectivity_tuning_plain_extension.json",
        "full-plain-extension": "config.amazon_x1_high_selectivity_tuning_full_plain_extension.json",
        "structure-extension": "config.amazon_x1_high_selectivity_tuning_structure_extension.json",
        "structure-boundary-extension": "config.amazon_x1_high_selectivity_tuning_structure_boundary_extension.json",
        "structure-endpoint-extension": "config.amazon_x1_high_selectivity_tuning_structure_endpoint_extension.json",
        "formal-boundary-extension": "config.amazon_x1_high_selectivity_formal_boundary_extension.json",
    }
    output = args.output or Path(__file__).with_name(default_names[args.phase])
    output.write_text(json.dumps(builders[args.phase](args.repo, args.repeats), indent=2) + "\n")
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
