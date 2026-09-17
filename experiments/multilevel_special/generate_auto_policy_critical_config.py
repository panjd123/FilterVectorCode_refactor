#!/usr/bin/env python3
"""Create the pre-declared 15-repeat same-T1 confirmation run."""

from __future__ import annotations

import json
from pathlib import Path


REPO = Path("/home/sunyahui/worktrees/FilterVectorCode_multilevel_special")
SOURCE = REPO / "experiments/multilevel_special/config.auto_policy_formal_exact_level.json"
OUTPUT = REPO / "experiments/multilevel_special/config.auto_policy_critical_exact_level.json"
KEEP = {
    "auto_mass_ladder_layer1_t1_8192",
    "auto_mass_ladder_t1_8192_t2_131072",
}


def main() -> None:
    config = json.loads(SOURCE.read_text())
    config["methods"] = [m for m in config["methods"] if m["name"] in KEEP]
    config["num_repeats"] = 15
    config["output_root"] = str(
        REPO / "runs/auto_policy_critical_exact_level_amazon_x1"
    )
    config["critical_protocol"] = {
        "purpose": "same-T1 one-layer versus automatic two-layer confirmation",
        "timing_rule": "repeat 0 cold, repeats 1-14 warm",
        "quality_rule": "every repeat must meet the pre-declared Recall threshold",
        "comparison_rule": "paired bootstrap over same-index warm repeats",
    }
    OUTPUT.write_text(json.dumps(config, indent=2) + "\n")
    print(OUTPUT)


if __name__ == "__main__":
    main()
