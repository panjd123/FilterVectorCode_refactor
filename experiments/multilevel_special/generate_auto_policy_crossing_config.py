#!/usr/bin/env python3
"""Generate a narrow exact-level sweep around each Recall crossing.

This stage does not select a deployment structure. It only measures real
integer L values between the coarse brackets so the formal experiment can use
the smallest observed L that reaches its pre-declared Recall threshold without
interpolation.
"""

from __future__ import annotations

import json
from pathlib import Path


REPO = Path("/home/sunyahui/worktrees/FilterVectorCode_multilevel_special")
BASE = json.loads((REPO / "experiments/multilevel_special/config.auto_policy_screen_exact_level.json").read_text())
BASE["output_root"] = str(REPO / "runs/auto_policy_crossing_exact_level_amazon_x1")
BASE["num_repeats"] = 3


GRIDS = {
    "layer0_plain": {
        "sel_0p5": [1100, 1200, 1300, 1400, 1500],
        "sel_1": [1600, 1700, 1800, 1900, 2000],
        "sel_5": [2500, 3000, 3500, 4000],
        "sel_10": [12000, 13000, 14000, 15000, 16000],
        "sel_30": [14000, 15000, 16000, 17000, 18000],
        "sel_60": [60000, 65000, 70000, 75000, 80000],
        "sel_80": [100000, 110000, 120000, 130000, 140000],
        "sel_95": [140000, 150000, 160000, 170000, 180000],
        "sel_99": [180000, 210000, 240000, 270000, 300000],
    },
    "auto_mass_ladder_layer1_t1_8192": {
        "sel_0p5": [1100, 1200, 1300, 1400, 1500],
        "sel_1": [1500, 1600, 1700, 1800, 1900, 2000],
        "sel_5": [1500, 1750, 2000, 2250, 2500],
        "sel_10": [8000, 8500, 9000, 9500, 10000],
        "sel_30": [6000, 6500, 7000, 7500, 8000],
        "sel_60": [1400, 1500, 1600, 1700, 1800, 2000],
        "sel_80": [1800, 2200, 2600, 3000],
        "sel_95": [800, 900, 1000, 1100, 1200, 1250],
        "sel_99": [600, 800, 1000, 1200],
    },
    "auto_mass_ladder_t1_8192_t2_131072": {
        "sel_0p5": [1300, 1400, 1500, 1600, 1750, 2000],
        "sel_1": [1500, 1600, 1700, 1800, 1900, 2000],
        "sel_5": [1200, 1400, 1600, 1800, 2000],
        "sel_10": [7500, 8000, 8500, 9000, 9500, 10000],
        "sel_30": [4500, 5000, 5500, 6000, 6500],
        "sel_60": [400, 450, 500, 550],
        "sel_80": [400, 450, 500, 550],
        "sel_95": [60, 75, 90, 100, 125],
        "sel_99": [55, 60, 70, 80, 90, 100],
    },
    "oracle_prior_t1_32000_t2_200000": {
        "sel_0p5": [1300, 1400, 1500, 1600, 1750, 2000],
        "sel_1": [1500, 1600, 1700, 1800, 1900, 2000],
        "sel_5": [1200, 1400, 1600, 1800, 2000],
        "sel_10": [7000, 7500, 8000, 8500, 9000, 10000],
        "sel_30": [4500, 5000, 5500, 6000, 6500],
        "sel_60": [400, 450, 500, 550],
        "sel_80": [400, 450, 500, 550],
        "sel_95": [60, 75, 90, 100, 125],
        "sel_99": [55, 60, 70, 80, 90, 100],
    },
}


methods = []
for source in BASE["methods"]:
    if source["name"] not in GRIDS:
        continue
    method = dict(source)
    method["lsearch_values_by_workload"] = GRIDS[source["name"]]
    methods.append(method)
BASE["methods"] = methods
BASE["crossing_protocol"] = {
    "purpose": "measure discrete Recall crossings; no interpolation",
    "selection": "smallest measured L whose deterministic Recall reaches the declared threshold",
    "timing": "repeat 0 is cold; repeats 1-2 are warm coarse-screen replicates",
}
OUT = REPO / "experiments/multilevel_special/config.auto_policy_crossing_exact_level.json"
OUT.write_text(json.dumps(BASE, indent=2) + "\n")
print(OUT)
