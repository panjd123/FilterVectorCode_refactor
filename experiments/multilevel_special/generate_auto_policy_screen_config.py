#!/usr/bin/env python3
"""Generate the nine-workload exact-level automatic-policy screen."""

from __future__ import annotations

import json
from pathlib import Path


REPO = Path("/home/sunyahui/worktrees/FilterVectorCode_multilevel_special")
OLD_BUILD = REPO / "runs/layer_tuning_build_amazon_x1"
NEW_BUILD = REPO / "runs/auto_policy_build_amazon_x1"

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

L0 = {
    "sel_0p5": [1000, 1500, 2000], "sel_1": [1500, 2000, 2500],
    "sel_5": [4000, 8000, 12000, 16000], "sel_10": [8000, 12000, 16000, 20000],
    "sel_30": [18000, 24000, 32000, 40000], "sel_60": [40000, 60000, 80000, 100000],
    "sel_80": [80000, 110000, 140000], "sel_95": [120000, 160000, 200000],
    "sel_99": [180000, 240000, 300000],
}
L1 = {
    "sel_0p5": [1000, 1500, 2500], "sel_1": [1500, 2500, 4000],
    "sel_5": [3000, 6000, 10000, 16000], "sel_10": [5000, 10000, 15000, 20000],
    "sel_30": [500, 1000, 2000, 4000, 8000], "sel_60": [250, 500, 1000, 2000, 4000],
    "sel_80": [250, 500, 750, 1000], "sel_95": [125, 250, 500, 750],
    "sel_99": [50, 100, 200, 400],
}
L2 = {
    "sel_0p5": [750, 1250, 2000], "sel_1": [1000, 2000, 3500],
    "sel_5": [2000, 4000, 8000, 12000], "sel_10": [3000, 6000, 10000, 15000],
    "sel_30": [250, 500, 1000, 2000, 4000], "sel_60": [125, 250, 500, 1000, 2000],
    "sel_80": [125, 250, 500, 750], "sel_95": [50, 125, 250, 500],
    "sel_99": [25, 50, 100, 200],
}


def method(name, layers, grid, root=None, t1=None, t2=None):
    value = {
        "name": name, "layer_count": layers, "special_block_search": layers > 0,
        "entry_group_provider": "cpu_bruteforce_els",
        "env": {"UNG_DISABLE_ELS_REUSE": "1", "UNG_DISABLE_CPU_ELS_WARMUP": "1",
                "UNG_SPECIAL_BLOCK_ROOT_LABEL_COVERAGE": "1"},
        "lsearch_values_by_workload": grid,
    }
    if root is not None:
        value["block_index"] = str(root / name / "block_index")
    if t1 is not None: value["t1"] = t1
    if t2 is not None: value["t2"] = t2
    return value


def main():
    methods = [method("layer0_plain", 0, L0)]
    methods += [
        method("balanced_layer1_t1_4096", 1, L1, NEW_BUILD, 4096),
        method("balanced_t1_4096_t2_32768", 2, L2, NEW_BUILD, 4096, 32768),
        method("degree_power_t1_4096_t2_262144", 2, L2, NEW_BUILD, 4096, 262144),
        method("block_count_t1_4000_t2_25000", 2, L2, NEW_BUILD, 4000, 25000),
        method("coverage_plateau_t1_8000_t2_32000", 2, L2, NEW_BUILD, 8000, 32000),
        method("static_plateau_t1_8000_t2_128000", 2, L2, NEW_BUILD, 8000, 128000),
        method("fanout_ladder_t1_32000_t2_128000", 2, L2, NEW_BUILD, 32000, 128000),
        method("m_budget_ladder_t1_4096_t2_131072", 2, L2, NEW_BUILD, 4096, 131072),
        method("auto_mass_ladder_layer1_t1_8192", 1, L1, NEW_BUILD, 8192),
        method("auto_mass_ladder_t1_8192_t2_131072", 2, L2, NEW_BUILD, 8192, 131072),
        method("oracle_prior_t1_32000_t2_200000", 2, L2,
               OLD_BUILD, 32000, 200000),
    ]
    # The historical directory name differs from the display name.
    methods[-1]["block_index"] = str(OLD_BUILD / "layer2_t1_32000_t2_200000" / "block_index")
    config = {
        "search_app": str(REPO / "build_ung_rel/apps/search_UNG_index"),
        "main_index": "/home/graphdb/FilterVectorResult/Amazon/index/Trie_block/index_files",
        "data_root": "/home/graphdb/FilterVectorData/Amazon",
        "gt_root": "/home/graphdb/FilterVectorResult/Amazon/GroundTruth",
        "output_root": str(REPO / "runs/auto_policy_screen_exact_level_amazon_x1"),
        "expected_source_fingerprint": "91d78580ae29f468",
        "expected_base_labels_sha256": "aec768bba7092af445252835be3f1ef7f708305ea646af19ae72738df3dd2f96",
        "expected_main_index_labels_sha256": "ddb3f616c27626afe6b20bf633aca5e9a1efd31d82bd505b163f59fc79f4dd56",
        "dataset": "Amazon", "K": 10, "expected_num_queries": 1000,
        "num_threads": 100, "num_entry_points": 16, "num_repeats": 3,
        "lsearch_values": [100], "require_stage_breakdown": True,
        "recall_thresholds": {name: (0.90 if sel <= .30 else 0.87)
                              for name, _, sel in WORKLOADS},
        "workloads": [{"name": name, "query_dir": directory,
                       "mean_selectivity": selectivity}
                      for name, directory, selectivity in WORKLOADS],
        "methods": methods,
    }
    output = Path(__file__).with_name("config.auto_policy_screen_exact_level.json")
    output.write_text(json.dumps(config, indent=2) + "\n")
    print(output)


if __name__ == "__main__":
    main()
