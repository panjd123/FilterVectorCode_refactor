#!/usr/bin/env python3
"""Create the missing Trie-base half of the Amazon topology factorial."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path


SOURCE_METHODS = (
    "l1_t1024_lng_entry_optimized_lng",
    "l1_t1024_trie_entry_optimized_lng",
    "l2_t1024_16384_ll_entry_optimized_lng",
    "l2_t1024_16384_lt_entry_optimized_lng",
    "l2_t1024_16384_tl_entry_optimized_lng",
    "l2_t1024_16384_tt_entry_optimized_lng",
)
TARGET_LSEARCH = {
    "sel_0p5": [1000, 1400, 2000],
    "sel_1": [1200, 1600, 2200],
    "sel_5": [400, 800, 1600, 2500, 5000],
    "sel_10": [500, 2500, 10000, 18000, 32000],
    "sel_30": [45000, 60000, 90000],
    "sel_60": [500, 1000, 2500, 5000],
    "sel_80": [320, 1250, 2500, 5000, 10000],
    "sel_95": [120, 500, 1000, 2500, 5000],
    "sel_99": [5000, 10000, 30000],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("template", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument(
        "--run-root",
        type=Path,
        default=Path(
            "/home/sunyahui/worktrees/FilterVectorCode_multilevel_special/"
            "runs/base_topology_factorial_20260929"
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = json.loads(args.template.read_text())
    by_name = {method["name"]: method for method in config["methods"]}

    # Reuse the immutable binary snapshot from the LNG-base half of the
    # factorial so executable drift cannot be mistaken for a topology effect.
    config["search_app"] = (
        "/home/sunyahui/worktrees/FilterVectorCode_multilevel_special/"
        "runs/authoritative_multilevel_20260926_emptyfix/search/amazon_screen/"
        ".binary_snapshots/search_UNG_index."
        "4c99a51c0f74b187d37f7070f73017b6230935126275560fff1061fa90830a81"
    )

    base_index = (
        "/home/sunyahui/worktrees/FilterVectorCode_multilevel_special/"
        "runs/authoritative_multilevel_20260925/base/amazon/base_trie/index_files"
    )
    hierarchy_root = args.run_root / "hierarchy" / "amazon_base_trie"
    methods = []
    for source_name in SOURCE_METHODS:
        method = copy.deepcopy(by_name[source_name])
        topology_suffix = source_name.removeprefix("l").split("_entry_", 1)[0]
        build_case = "l" + topology_suffix.replace("t1024", "base_trie_t1024", 1)
        method["name"] = f"{build_case}_entry_trie"
        method["main_index"] = base_index
        method["base_topology"] = "trie"
        method["entry_strategy"] = "trie"
        method["block_index"] = str(hierarchy_root / build_case / "block_index")
        method["selection_role"] = "base_topology_factorial_matched_entry"
        method["lsearch_values_by_workload"] = TARGET_LSEARCH
        methods.append(method)

    config["purpose"] = (
        "Trie-base half of the controlled Amazon base-by-overlay topology "
        "factorial, matched to the 20260926 LNG-base executable and using "
        "the Trie entry provider."
    )
    config["output_root"] = str(args.run_root / "search" / "amazon_base_trie_4c99")
    config["minimum_successful_child_seconds"] = 1800
    config["methods"] = methods
    args.output.write_text(json.dumps(config, indent=2) + "\n")


if __name__ == "__main__":
    main()
