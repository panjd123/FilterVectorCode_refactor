#!/usr/bin/env python3
"""Freeze measured Recall crossings into the formal exact-level run."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path


REPO = Path("/home/sunyahui/worktrees/FilterVectorCode_multilevel_special")
CROSSING_CONFIG = REPO / "experiments/multilevel_special/config.auto_policy_crossing_exact_level.json"
CROSSING_ROOT = REPO / "runs/auto_policy_crossing_exact_level_amazon_x1"
METHODS = {
    "layer0_plain",
    "auto_mass_ladder_layer1_t1_8192",
    "auto_mass_ladder_t1_8192_t2_131072",
    "oracle_prior_t1_32000_t2_200000",
}


def choose_l(method: str, workload: str, threshold: float) -> int:
    path = CROSSING_ROOT / method / workload / "search_time_details.csv"
    with path.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    by_l: dict[int, list[dict[str, str]]] = {}
    for row in rows:
        by_l.setdefault(int(row["Lsearch"]), []).append(row)
    for lsearch in sorted(by_l):
        repeats = by_l[lsearch]
        if len(repeats) != 3:
            raise ValueError(f"incomplete crossing point: {method}/{workload}/L={lsearch}")
        if min(float(row["Avg_Recall"]) for row in repeats) >= threshold:
            return lsearch
    raise ValueError(f"no feasible crossing: {method}/{workload}")


base = json.loads(CROSSING_CONFIG.read_text())
thresholds = {key: float(value) for key, value in base["recall_thresholds"].items()}
methods = []
selected = {}
for source in base["methods"]:
    if source["name"] not in METHODS:
        continue
    method = dict(source)
    grid = {}
    for workload in base["workloads"]:
        name = workload["name"]
        lsearch = choose_l(source["name"], name, thresholds[name])
        grid[name] = [lsearch]
        selected[f"{source['name']}/{name}"] = lsearch
    method["lsearch_values_by_workload"] = grid
    methods.append(method)
base["methods"] = methods
base["num_repeats"] = 7
base["output_root"] = str(REPO / "runs/auto_policy_formal_exact_level_amazon_x1")
base["formal_protocol"] = {
    "selection_source": str(CROSSING_ROOT),
    "selection_config_sha256": hashlib.sha256(CROSSING_CONFIG.read_bytes()).hexdigest(),
    "selected_lsearch": selected,
    "quality_rule": "all three crossing-screen repeats meet the pre-declared Recall threshold",
    "timing_rule": "repeat 0 cold, repeats 1-6 warm; report warm median/mean/CV/p50/p95",
    "comparison_rule": "paired bootstrap over same-index warm repeats; no interpolation or extrapolation",
}
out = REPO / "experiments/multilevel_special/config.auto_policy_formal_exact_level.json"
out.write_text(json.dumps(base, indent=2) + "\n")
print(out)
