#!/usr/bin/env python3
"""Generate the same-index, same-binary UNG entry-provider experiment."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import experiment_core


DEFAULT_REPO = Path("/home/sunyahui/worktrees/FilterVectorCode_multilevel_special")
SELECTIONS = ("sel_0p5", "sel_1", "sel_5", "sel_10", "sel_30",
              "sel_60", "sel_80", "sel_95", "sel_99")


def _screen_crossing(path: Path, threshold: float, repeats: int) -> int:
    if not path.is_file():
        raise FileNotFoundError(f"missing screen result: {path}")
    with path.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    crossing, _ = experiment_core.choose_recall_crossing(
        rows, threshold, repeats, "all_repeats", cold_repeats=1)
    return crossing


def _refinement_grid(path: Path, threshold: float, repeats: int, points: int = 5) -> list[int]:
    """Bracket a Recall crossing and insert integer-spaced L values."""
    with path.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    crossing, _ = experiment_core.choose_recall_crossing(
        rows, threshold, repeats, "all_repeats", cold_repeats=1)
    grouped = {}
    for row in rows:
        grouped.setdefault(int(row["Lsearch"]), []).append(float(row["Avg_Recall"]))
    lower = max((value for value, recalls in grouped.items()
                 if value < crossing and min(recalls) < threshold), default=None)
    if lower is None:
        return [crossing]
    step = max(1, (crossing - lower) // points)
    values = list(range(lower + step, crossing, step))
    return sorted(set(values[-(points - 1):] + [crossing]))


def make_config(repo: Path, phase: str, allow_partial: bool = False) -> dict:
    source_path = repo / "experiments/multilevel_special/config.auto_policy_formal_exact_level.json"
    source = json.loads(source_path.read_text())
    workloads = [item for item in source["workloads"] if item["name"] in SELECTIONS]
    plain = next(item for item in source["methods"] if item["name"] == "layer0_plain")
    grids = plain["lsearch_values_by_workload"]

    screen_repeats = 3
    if phase == "screen":
        repeats = 3
        # The old provider can have a different Recall/L curve.  A geometric
        # guard around and above the current plain crossing avoids assuming
        # that the two routes reach quality at the same L.
        grids = {name: sorted({
            max(10, int(values[0] * .5)), int(values[0]),
            int(values[0] * 2), int(values[0] * 4),
        }) for name, values in grids.items()}
        method_grids = {
            "ung_original_entry": grids,
            "plain_bitset_lng_entry": grids,
        }
    else:
        repeats = 3 if phase == "crossing" else 7
        source_phase = "screen" if phase == "crossing" else "crossing"
        source_repeats = 3
        screen_root = repo / "runs" / f"ung_plain_{source_phase}_amazon_x1"
        method_grids = {}
        method_names = ("ung_original_entry", "plain_bitset_lng_entry")
        for method_name in method_names:
            method_grids[method_name] = {}
            for workload in workloads:
                workload_name = workload["name"]
                result_path = screen_root / method_name / workload_name / "search_time_details.csv"
                try:
                    threshold = float(source["recall_thresholds"][workload_name])
                    if phase == "crossing":
                        values = _refinement_grid(
                            result_path, threshold, source_repeats)
                    else:
                        values = [_screen_crossing(
                            result_path, threshold, source_repeats)]
                except (FileNotFoundError, ValueError):
                    if not allow_partial:
                        raise
                    continue
                method_grids[method_name][workload_name] = values
        if allow_partial:
            common = set.intersection(*(set(method_grids[name]) for name in method_names))
            workloads = [item for item in workloads if item["name"] in common]
            if not workloads:
                raise ValueError("no workload has a Recall crossing for both methods")
            method_grids = {
                name: {workload["name"]: grids[workload["name"]]
                       for workload in workloads}
                for name, grids in method_grids.items()
            }

    common_env = {
        "UNG_DISABLE_ELS_REUSE": "1",
        "UNG_DISABLE_CPU_ELS_WARMUP": "1",
    }
    methods = []
    for name, provider in (("ung_original_entry", "cpu_min_super_sets"),
                           ("plain_bitset_lng_entry", "cpu_bruteforce_els")):
        methods.append({
            "name": name, "layer_count": 0, "special_block_search": False,
            "entry_group_provider": provider, "graph_search_backend": "neighbor_list",
            "main_graph": "shared UNG vector graph",
            "block_partition": "none", "env": common_env,
            "lsearch_values_by_workload": method_grids[name],
            "enabled_workloads": [item["name"] for item in workloads],
        })

    result = {key: source[key] for key in (
        "search_app", "main_index", "data_root", "gt_root",
        "expected_source_fingerprint", "expected_base_labels_sha256",
        "expected_main_index_labels_sha256", "dataset", "K",
        "expected_num_queries", "num_threads", "num_entry_points",
        "lsearch_values", "require_stage_breakdown", "recall_thresholds")}
    result.update({
        "output_root": str(repo / "runs" / f"ung_plain_{phase}_amazon_x1"),
        "num_repeats": repeats, "workloads": workloads, "methods": methods,
        "protocol": {
            "phase": phase, "cold_repeats": 1,
            "measured_repeats": repeats - 1, "recall_rule": "all_repeats",
            "bootstrap_samples": 100000 if phase == "formal" else 10000,
            "paired_repeats": False,
            "comparison": "same binary, index, query, GT, threads, graph backend; provider only",
        },
    })
    result["recall_thresholds"] = {
        workload["name"]: source["recall_thresholds"][workload["name"]]
        for workload in workloads
    }
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=DEFAULT_REPO)
    parser.add_argument(
        "--phase", choices=("screen", "crossing", "formal"),
        default="screen")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--allow-partial", action="store_true",
                        help="formal only: keep workloads complete for both methods")
    args = parser.parse_args()
    output = args.output or args.repo / "experiments/multilevel_special" / f"config.ung_plain_{args.phase}.json"
    output.write_text(json.dumps(make_config(
        args.repo, args.phase, allow_partial=args.allow_partial), indent=2) + "\n")
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
