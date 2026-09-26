#!/usr/bin/env python3
"""Generate post-refactor Amazon hierarchy/topology experiment configs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


REPO = Path("/home/sunyahui/worktrees/FilterVectorCode_multilevel_special")
RUN_ROOT = REPO / "runs/authoritative_multilevel_20260925"
DATA_ROOT = Path("/home/graphdb/FilterVectorData/Amazon")
GT_ROOT = Path("/home/graphdb/FilterVectorResult/Amazon/GroundTruth")
BASE_ROOT = RUN_ROOT / "base/amazon"
HIERARCHY_ROOT = RUN_ROOT / "hierarchy/amazon_base_lng"

WORKLOADS = [
    ("sel_0p5", "query_minlen5_avgsel05pct", 0.0049924907),
    ("sel_1", "query_minlen5_avgsel1pct", 0.0090305484),
    ("sel_5", "query_nested_avgsel5pct", 0.0503811667),
    ("sel_10", "query_minlen3_avgsel10pct", 0.0990661529),
    ("sel_30", "query_nested_avgsel30pct", 0.3002724229),
    ("sel_60", "query_nested_avgsel60pct", 0.6004749068),
    ("sel_80", "query_minlen1_nested_avgsel80pct", 0.8002369928),
    ("sel_95", "query_minlen1_nested_avgsel95pct", 0.9501978063),
    ("sel_99", "query_label1_empty_avgsel99pct", 0.9900060038),
]

# The zero-layer graph needs a much larger beam once restrictive label filters
# disconnect the eligible subgraph.  Layered indexes are screened densely in
# the small-beam regime where their crossings are expected, with guard points
# retained to expose configurations that do not provide useful routing.
ZERO_LAYER_LSEARCH_BY_WORKLOAD = {
    "sel_0p5": [1000, 1200, 1400, 1600, 2000],
    "sel_1": [1200, 1400, 1600, 1800, 2200],
    "sel_5": [800, 1200, 1600, 2000, 2500, 3200],
    "sel_10": [4000, 6000, 8000, 10000, 12000, 15000, 18000,
               22000, 26000, 32000],
    "sel_30": [1000, 3000, 5000, 7000, 9000, 13000, 17000, 22000,
               32000, 45000, 60000, 90000, 120000, 150000],
    "sel_60": [250, 500, 1000, 2000, 5000, 10000, 30000, 60000,
               90000, 120000, 180000, 260000],
    "sel_80": [120000, 220000, 320000],
    "sel_95": [150000, 220000, 320000],
    "sel_99": [150000, 280000, 400000],
}

TRIE_ZERO_LAYER_LSEARCH_BY_WORKLOAD = {
    "sel_0p5": [1000, 1200, 1400, 1600, 2000],
    "sel_1": [1200, 1400, 1600, 1800, 2200],
    "sel_5": [400, 800, 1200, 1600, 2000, 2500, 3200, 5000],
    "sel_10": [500, 1000, 2500, 5000, 10000, 18000, 32000, 64000],
    "sel_30": [250, 500, 1000, 2500, 5000, 10000, 30000, 150000],
    "sel_60": [100, 250, 500, 1000, 2500, 5000, 10000, 30000,
               320000],
    "sel_80": [100, 250, 500, 1000, 2500, 5000, 10000, 30000,
               602453],
    "sel_95": [50, 100, 250, 500, 1000, 2500, 5000, 10000, 30000,
               602453],
    "sel_99": [40, 60, 80, 120, 250, 500, 1000, 5000, 25000, 30000,
               602453],
}

PRIMARY_LAYERED_LSEARCH_BY_WORKLOAD = {
    "sel_0p5": [1000, 1200, 1400, 1600, 2000],
    "sel_1": [1200, 1400, 1600, 1800, 2200],
    "sel_5": [400, 800, 1200, 1600, 2000, 2500, 3200, 5000],
    "sel_10": [100, 250, 500, 1000, 2500, 5000, 10000, 18000, 32000],
    "sel_30": [50, 100, 250, 500, 1000, 2500, 5000, 10000, 22000,
               45000],
    "sel_60": [25, 50, 100, 250, 500, 1000, 2500, 5000, 10000,
               30000, 90000],
    "sel_80": [20, 40, 80, 160, 320, 640, 1250, 2500, 5000, 10000,
               30000],
    "sel_95": [20, 40, 60, 80, 120, 250, 500, 1000, 2500, 5000,
               10000, 30000],
    "sel_99": [20, 40, 60, 80, 120, 250, 500, 1000, 2500, 5000,
               10000, 30000],
}

# The full DRH-v1 pilot above established the broad curve shape.  Remaining
# factorial and routing controls use a compact, predeclared screen followed by
# the same measured crossing refinement, avoiding repeated dense high-L scans.
LAYERED_LSEARCH_BY_WORKLOAD = {
    "sel_0p5": [1000, 1400, 2000],
    "sel_1": [1200, 1600, 2200],
    "sel_5": [400, 800, 1600, 2500, 5000],
    "sel_10": [500, 2500, 10000, 18000, 32000],
    "sel_30": [100, 500, 2500, 5000, 10000, 22000, 45000],
    "sel_60": [100, 500, 1000, 2500, 5000, 10000, 30000, 90000],
    "sel_80": [40, 160, 640, 1250, 2500, 5000, 10000, 30000],
    "sel_95": [40, 80, 250, 1000, 2500, 5000, 10000, 30000],
    "sel_99": [40, 80, 250, 1000, 2500, 5000, 10000, 30000],
}


def common_method_env() -> dict[str, str]:
    return {
        "UNG_DISABLE_ELS_REUSE": "1",
        "UNG_DISABLE_CPU_ELS_WARMUP": "1",
        "UNG_SPECIAL_BLOCK_ROOT_LABEL_COVERAGE": "1",
    }


def zero_layer_methods() -> list[dict]:
    lng = str(BASE_ROOT / "base_lng/index_files")
    trie = str(BASE_ROOT / "base_trie/index_files")
    methods = []
    for topology, main in (("lng", lng), ("trie", trie)):
        for entry in ("original", "optimized_lng", "trie"):
            methods.append({
                "name": f"l0_{topology}_entry_{entry}",
                "main_index": main,
                "base_topology": topology,
                "entry_strategy": entry,
                "hierarchy_layers": [],
                "special_block_search": False,
                "lsearch_values_by_workload": (
                    ZERO_LAYER_LSEARCH_BY_WORKLOAD if topology == "lng"
                    else TRIE_ZERO_LAYER_LSEARCH_BY_WORKLOAD),
                "env": common_method_env(),
            })
    return methods


def layered_methods(hierarchy_config: Path) -> list[dict]:
    source = json.loads(hierarchy_config.read_text())
    methods = []
    for case in source["cases"]:
        for entry in ("original", "optimized_lng", "trie"):
            is_full_grid_pilot = (
                case["name"] == "l2_t1024_16384_lt" and
                entry == "optimized_lng")
            methods.append({
                "name": case["name"] + f"_entry_{entry}",
                "main_index": source["main_index"],
                "base_topology": case["base_topology"],
                "entry_strategy": entry,
                "hierarchy_layers": case["hierarchy_layers"],
                "special_block_search": True,
                "block_index": str(HIERARCHY_ROOT / case["name"] / "block_index"),
                "selection_role": case.get("selection_role", "manual_grid"),
                "lsearch_values_by_workload": (
                    PRIMARY_LAYERED_LSEARCH_BY_WORKLOAD
                    if is_full_grid_pilot else LAYERED_LSEARCH_BY_WORKLOAD),
                "env": common_method_env(),
            })
    return methods


def ordered_methods(hierarchy_config: Path) -> list[dict]:
    """Put paper-critical comparisons first without changing case semantics."""
    methods = zero_layer_methods() + layered_methods(hierarchy_config)
    by_name = {method["name"]: method for method in methods}
    for source_name in (
            "l2_t1024_16384_lt_entry_optimized_lng",
            "l2_t8192_131072_lt_entry_optimized_lng"):
        routed = dict(by_name[source_name])
        routed["name"] = source_name + "_upper_routed"
        routed["selection_role"] = (
            str(routed["selection_role"]) + "_upper_authorization_control")
        routed["routing_policy"] = "require_upper_authorization"
        routed["lsearch_values_by_workload"] = LAYERED_LSEARCH_BY_WORKLOAD
        routed["env"] = dict(routed["env"])
        routed["env"]["UNG_SPECIAL_REQUIRE_UPPER_AUTHORIZATION"] = "1"
        by_name[routed["name"]] = routed
    priority = [
        "l0_lng_entry_optimized_lng",
        "l0_trie_entry_trie",
        "l2_t1024_16384_lt_entry_optimized_lng",
        "l2_t1024_16384_lt_entry_optimized_lng_upper_routed",
        "l2_t8192_131072_lt_entry_optimized_lng_upper_routed",
        "l1_t1024_lng_entry_optimized_lng",
        "l1_t1024_trie_entry_optimized_lng",
        "l1_t8192_lng_entry_optimized_lng",
        "l1_t8192_trie_entry_optimized_lng",
    ]
    ordered = [by_name.pop(name) for name in priority]
    ordered.extend(
        by_name.pop(name) for name in list(by_name)
        if name.endswith("_entry_optimized_lng"))
    ordered.extend(by_name.values())
    return ordered


def make_screen(hierarchy_config: Path, output_root: Path | None = None) -> dict:
    workloads = [
        {"name": name, "query_dir": query_dir, "mean_selectivity": selectivity,
         "num_queries": 1000}
        for name, query_dir, selectivity in WORKLOADS
    ]
    return {
        "schema_version": 2,
        "method_schema": "orthogonal_v2",
        "purpose": "Post-fcb74ad broad Recall-QPS screen across topology and hierarchy dimensions.",
        "measurement_pass": "performance",
        "pass_subdirs": True,
        "search_app": str(REPO / "build_ung_rel/apps/search_UNG_index"),
        "data_root": str(DATA_ROOT),
        "gt_root": str(GT_ROOT),
        "output_root": str(output_root or RUN_ROOT / "search/amazon_screen"),
        "dataset": "Amazon",
        "expected_num_points": 602453,
        "expected_source_fingerprint": "91d78580ae29f468",
        "expected_base_labels_sha256": "aec768bba7092af445252835be3f1ef7f708305ea646af19ae72738df3dd2f96",
        "expected_main_index_labels_sha256": "ddb3f616c27626afe6b20bf633aca5e9a1efd31d82bd505b163f59fc79f4dd56",
        "K": 10,
        "expected_num_queries": 1000,
        "num_threads": 100,
        "num_entry_points": 16,
        "num_repeats": 3,
        "minimum_successful_child_seconds": 43200,
        "lsearch_values": [100],
        "require_stage_breakdown": True,
        "require_work_breakdown": False,
        "protocol": {
            "phase": "screen", "cold_repeats": 1,
            "measured_repeats": 2, "recall_rule": "all_repeats",
            "bootstrap_samples": 10000, "paired_repeats": False,
        },
        "recall_thresholds": {name: 0.90 for name, *_ in WORKLOADS},
        "workloads": workloads,
        "methods": ordered_methods(hierarchy_config),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--hierarchy-config", type=Path,
        default=REPO / "experiments/multilevel_special/config.authoritative_amazon_hierarchy_grid.json")
    parser.add_argument(
        "--output", type=Path,
        default=REPO / "experiments/multilevel_special/config.authoritative_amazon_screen.json")
    parser.add_argument(
        "--output-root", type=Path,
        help="override only the search result root while reusing the declared index roots")
    args = parser.parse_args()
    config = make_screen(args.hierarchy_config, args.output_root)
    args.output.write_text(json.dumps(config, indent=2) + "\n")
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
