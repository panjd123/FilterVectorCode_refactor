#!/usr/bin/env python3
"""Generate the auditable 0/1/2-layer coarse query-tuning matrix."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


# Successive complete coarse sweeps exposed upper T1/T2 boundaries and one
# lower local T2 boundary. Keep the original Cartesian grid intact and add
# focused guard points for only those exposed axes instead of paying for an
# unrelated full Cartesian expansion.
LAYER1_T1_VALUES = (
    500, 1000, 2000, 4000, 8000, 16000, 32000, 64000,
    128000, 256000, 700000,
)
LAYER2_T1_VALUES = (500, 1000, 2000, 4000, 8000)
T2_VALUES = (4000, 10000, 25000, 50000)
LAYER2_GUARD_PAIRS = (
    (2000, 100000),
    (8000, 9000),
    (8000, 100000),
    (16000, 25000),
    (16000, 50000),
    (16000, 100000),
    (32000, 50000),
    (32000, 100000),
    # Second boundary guard: the first extension selected T1=32000 for the
    # shared one-level configuration and T2=100000 for the shared two-level
    # configuration.  Add only the neighboring points needed to distinguish
    # continued upper-bound improvement.  The 32k/40k point guards the lower
    # T2 side exposed by the sel_25 oracle at 32k/50k.
    (16000, 200000),
    (32000, 40000),
    (32000, 200000),
    (64000, 100000),
    (64000, 200000),
    # Final oracle guard.  The 700k one-level endpoint is above the 602,453
    # point dataset and therefore explicitly measures the no-block limiting
    # case.  Two-level points follow only the exposed 64k/200k and 32k/50k
    # trajectories up to a near-dataset-size upper threshold.
    (64000, 80000),
    (64000, 400000),
    (64000, 600000),
    (128000, 200000),
    (128000, 400000),
    (128000, 600000),
    (256000, 400000),
    (256000, 600000),
    # sel_25 selected 64k/80k after the final upper-threshold guard.  Close
    # that local legal slice with one lower-T1 control and the exact integer
    # endpoints imposed by T2 > T1.  These are workload-local oracle guards,
    # not candidates for the six-workload shared deployment configuration.
    (32000, 80000),
    (64000, 64001),
    (79999, 80000),
)

# These endpoint guards close only the still-open per-workload oracle axes.
# They intentionally do not enter shared-threshold selection, which requires a
# structure to be measured on all six workloads.
ORACLE_GUARD_WORKLOADS = {
    "layer1_t1_128000": ["sel_25"],
    "layer1_t1_256000": ["sel_25"],
    # Structural endpoint only: 700k exceeds the 602,453-point dataset and
    # produces zero middle blocks.  Keep it in the boundary reference grid
    # but never report it as a measured one-level method.
    "layer1_t1_700000": [],
    "layer2_t1_64000_t2_80000": ["sel_25"],
    "layer2_t1_64000_t2_400000": ["sel_005", "sel_01"],
    # T2=600k produces no upper block on Amazon x1.  These are structural
    # endpoints used only to prove that T2=400k is not an open search edge.
    "layer2_t1_64000_t2_600000": [],
    "layer2_t1_128000_t2_200000": ["sel_005", "sel_01"],
    "layer2_t1_128000_t2_400000": ["sel_005", "sel_01"],
    "layer2_t1_128000_t2_600000": [],
    "layer2_t1_256000_t2_400000": ["sel_005", "sel_01"],
    "layer2_t1_256000_t2_600000": [],
    "layer2_t1_32000_t2_80000": ["sel_25"],
    "layer2_t1_64000_t2_64001": ["sel_25"],
    "layer2_t1_79999_t2_80000": ["sel_25"],
}

WORKLOADS = (
    {"name": "sel_005", "query_dir": "query_minlen5_avgsel05pct",
     "mean_selectivity": 0.0049924643},
    {"name": "sel_01", "query_dir": "query_minlen5_avgsel1pct",
     "mean_selectivity": 0.0090305985},
    {"name": "sel_10", "query_dir": "query_minlen3_avgsel10pct",
     "mean_selectivity": 0.0990661565},
    {"name": "sel_25", "query_dir": "query_minlen2_avgsel25pct",
     "mean_selectivity": 0.2491539473},
    {"name": "sel_50", "query_dir": "query_minlen1_avgsel50pct",
     "mean_selectivity": 0.4997135478},
    {"name": "sel_75", "query_dir": "query_minlen1_avgsel75pct",
     "mean_selectivity": 0.7499381960},
)

LAYER0_L = {
    "sel_005": [1000, 1250, 1500, 1750, 2000],
    "sel_01": [1500, 1750, 2000, 2250, 2500],
    "sel_10": [12000, 14000, 15000, 16000, 18000],
    "sel_25": [18000, 20000, 22000, 24000, 26000],
    "sel_50": [30000, 35000, 40000, 45000],
    "sel_75": [80000, 95000, 110000, 125000],
}

LAYER1_L = {
    "sel_005": [1000, 1500, 2000, 3000, 5000],
    "sel_01": [1500, 2500, 3500, 4500, 6000, 9000],
    "sel_10": [8000, 12000, 16000, 20000, 28000, 40000],
    "sel_25": [8000, 12000, 16000, 20000, 28000],
    "sel_50": [400, 700, 1000, 1400, 1800, 2400, 3500],
    "sel_75": [700, 1000, 1600, 2400, 3500, 5000],
}

LAYER2_L = {
    "sel_005": [750, 1000, 1250, 1500, 1800, 2200],
    "sel_01": [1500, 2000, 2500, 3000, 4000, 5000],
    "sel_10": [5000, 7500, 10000, 12500, 15000, 20000],
    "sel_25": [5000, 7500, 10000, 12500, 15000, 20000],
    "sel_50": [250, 400, 500, 700, 1000, 1500],
    "sel_75": [500, 750, 1000, 1500, 2000, 3000],
}


def method(name: str, layer_count: int, l_grid: dict[str, list[int]],
           build_root: Path | None = None, t1: int | None = None,
           t2: int | None = None) -> dict:
    result = {
        "name": name,
        "layer_count": layer_count,
        "special_block_search": layer_count > 0,
        "entry_group_provider": "cpu_bruteforce_els",
        "env": {
            "UNG_DISABLE_ELS_REUSE": "1",
            "UNG_DISABLE_CPU_ELS_WARMUP": "1",
        },
        "lsearch_values_by_workload": l_grid,
    }
    if t1 is not None:
        result["t1"] = t1
    if t2 is not None:
        result["t2"] = t2
    if build_root is not None:
        result["block_index"] = str(build_root / name / "block_index")
        # The paper definition authorizes a block exactly when the query
        # contains its root-label prefix.  Multilevel search enables this
        # automatically, but declare it for every Special method so the
        # one-level control uses identical semantics.
        result["env"]["UNG_SPECIAL_BLOCK_ROOT_LABEL_COVERAGE"] = "1"
    if name in ORACLE_GUARD_WORKLOADS:
        result["enabled_workloads"] = ORACLE_GUARD_WORKLOADS[name]
    return result


def make_config(repo: Path) -> dict:
    build_root = repo / "runs/layer_tuning_build_amazon_x1"
    methods = [method("layer0_plain", 0, LAYER0_L)]
    methods.extend(method(f"layer1_t1_{t1}", 1, LAYER1_L, build_root, t1=t1)
                   for t1 in LAYER1_T1_VALUES)
    original_pairs = {
        (t1, t2) for t1 in LAYER2_T1_VALUES for t2 in T2_VALUES if t2 > t1
    }
    layer2_pairs = sorted(original_pairs | set(LAYER2_GUARD_PAIRS))
    methods.extend(
        method(f"layer2_t1_{t1}_t2_{t2}", 2, LAYER2_L, build_root,
               t1=t1, t2=t2)
        for t1, t2 in layer2_pairs
    )
    # Keep non-executable structural endpoints in the boundary reference.
    # They close a threshold axis when crossing it removes the corresponding
    # block layer, but must never become a measured performance candidate.
    boundary_reference_methods = [
        {key: item.get(key) for key in ("name", "layer_count", "t1", "t2")}
        for item in methods
    ]
    return {
        "search_app": str(repo / "build_ung_rel/apps/search_UNG_index"),
        "main_index": "/home/graphdb/FilterVectorResult/Amazon/index/Trie_block/index_files",
        "data_root": "/home/graphdb/FilterVectorData/Amazon",
        "gt_root": "/home/graphdb/FilterVectorResult/Amazon/GroundTruth",
        "output_root": str(repo / "runs/layer_tuning_query_coarse_amazon_x1"),
        "expected_source_fingerprint": "91d78580ae29f468",
        "expected_base_labels_sha256": "aec768bba7092af445252835be3f1ef7f708305ea646af19ae72738df3dd2f96",
        "expected_main_index_labels_sha256": "ddb3f616c27626afe6b20bf633aca5e9a1efd31d82bd505b163f59fc79f4dd56",
        "dataset": "Amazon",
        "K": 10,
        "expected_num_queries": 1000,
        "num_threads": 100,
        "num_entry_points": 16,
        "num_repeats": 3,
        "lsearch_values": [100],
        "recall_thresholds": {
            "sel_005": 0.90, "sel_01": 0.90, "sel_10": 0.90,
            "sel_25": 0.90, "sel_50": 0.85, "sel_75": 0.87
        },
        "require_stage_breakdown": True,
        "boundary_reference_methods": boundary_reference_methods,
        "workloads": list(WORKLOADS),
        "methods": methods,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    output = args.output or Path(__file__).with_name(
        "config.amazon_x1_layer_tuning_query_coarse.json")
    output.write_text(json.dumps(make_config(args.repo), indent=2) + "\n")
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
