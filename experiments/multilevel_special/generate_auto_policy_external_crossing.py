#!/usr/bin/env python3
"""Create narrow external-dataset Recall crossing grids."""

from __future__ import annotations

import json
from pathlib import Path


REPO = Path("/home/sunyahui/worktrees/FilterVectorCode_multilevel_special")
GRIDS = {
    "genome": {
        "query_minlen5_cov0.1k": [10],
        "query_minlen2_cov1k": [10],
    },
    "reviews": {
        "query_minlen5_cov0.1k": [100, 125, 150, 175, 200, 225, 250],
        "query_minlen2_cov1k": [75, 100, 125, 150, 175, 200, 225, 250],
    },
    "variousimg": {
        "query_minlen2_cov5k": [2500, 2750, 3000, 3250, 3500, 3750, 4000,
                                    4250, 4500, 5000, 5500, 6000, 6500, 7000, 7500, 8000],
    },
}


def main() -> None:
    for dataset, grids in GRIDS.items():
        source = REPO / f"experiments/multilevel_special/config.auto_policy_cross_dataset_{dataset}_screen.json"
        config = json.loads(source.read_text())
        for method in config["methods"]:
            method.pop("lsearch_values", None)
            method["lsearch_values_by_workload"] = grids
        config["output_root"] = str(
            REPO / f"runs/auto_policy_cross_dataset_{dataset}/crossing")
        output = source.with_name(source.name.replace("_screen.json", "_crossing.json"))
        output.write_text(json.dumps(config, indent=2) + "\n")
        print(output)


if __name__ == "__main__":
    main()
