#!/usr/bin/env python3
"""Generate targeted exact-level ablations after the static-policy screen."""

import json
from pathlib import Path

REPO = Path("/home/sunyahui/worktrees/FilterVectorCode_multilevel_special")
BASE = json.loads((REPO / "experiments/multilevel_special/config.auto_policy_screen_exact_level.json").read_text())
BASE["output_root"] = str(REPO / "runs/auto_policy_ablation_exact_level_amazon_x1")
BASE["num_repeats"] = 3
BASE["workloads"] = [w for w in BASE["workloads"]]

COMMON = {
    "special_block_search": True,
    "entry_group_provider": "cpu_bruteforce_els",
    "env": {
        "UNG_DISABLE_ELS_REUSE": "1",
        "UNG_DISABLE_CPU_ELS_WARMUP": "1",
        "UNG_SPECIAL_BLOCK_ROOT_LABEL_COVERAGE": "1",
    },
}
L1 = {
    "sel_0p5": [1000,1500,2000], "sel_1": [1500,2000,2500],
    "sel_5": [2000,3000,4000], "sel_10": [6000,8000,10000,12000],
    "sel_30": [4000,6000,8000,10000], "sel_60": [1000,1500,2000,3000],
    "sel_80": [1000,1500,2000,3000], "sel_95": [750,1250,2000,3000],
    "sel_99": [400,800,1200,2000],
}
L2 = {
    "sel_0p5": [1250,1500,2000], "sel_1": [1500,2000,2500],
    "sel_5": [1500,2000,3000], "sel_10": [6000,8000,10000],
    "sel_30": [4000,6000,8000,10000], "sel_60": [250,500,750,1000],
    "sel_80": [250,375,500,750], "sel_95": [50,100,125,175],
    "sel_99": [50,75,100,150],
}
def m(name, layers, path, t1, t2=None, grid=L2):
    x = dict(COMMON)
    x.update(name=name, layer_count=layers, block_index=str(path), t1=t1,
             lsearch_values_by_workload=grid)
    if t2 is not None: x["t2"] = t2
    return x

BASE["methods"] = [
    m("layer1_t1_8000", 1, REPO / "runs/layer_tuning_build_amazon_x1/layer1_t1_8000/block_index", 8000, grid=L1),
    m("static_plateau_t1_8000_t2_128000", 2, REPO / "runs/auto_policy_build_amazon_x1/static_plateau_t1_8000_t2_128000/block_index", 8000, 128000),
    m("m_budget_ladder_t1_4096_t2_131072", 2, REPO / "runs/auto_policy_build_amazon_x1/m_budget_ladder_t1_4096_t2_131072/block_index", 4096, 131072),
    m("oracle_prior_t1_32000_t2_200000", 2, REPO / "runs/layer_tuning_build_amazon_x1/layer2_t1_32000_t2_200000/block_index", 32000, 200000),
]
OUT = REPO / "experiments/multilevel_special/config.auto_policy_ablation_exact_level.json"
OUT.write_text(json.dumps(BASE, indent=2) + "\n")
print(OUT)
