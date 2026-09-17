#!/usr/bin/env python3
"""Generate query-free policy build and search configs for external datasets."""

from __future__ import annotations

import hashlib
import json
import struct
from pathlib import Path


REPO = Path("/home/sunyahui/worktrees/FilterVectorCode_multilevel_special")
DATA = Path("/home/graphdb/FilterVectorData")
RESULT = Path("/home/graphdb/FilterVectorResult")
POLICY = REPO / "runs/auto_layer_policy_20260917/policy_cross_dataset"
TASKS = {
    "Genome": ["query_minlen5_cov0.1k", "query_minlen2_cov1k"],
    "Reviews": ["query_minlen5_cov0.1k", "query_minlen2_cov1k"],
    "VariousImg": ["query_minlen2_cov5k"],
}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def meta(path: Path) -> dict[str, str]:
    return dict(line.split("=", 1) for line in path.read_text().splitlines() if "=" in line)


def shape(path: Path) -> tuple[int, int]:
    with path.open("rb") as stream:
        return struct.unpack("<II", stream.read(8))


def profile_mean(dataset: str, task: str, num_points: int) -> float | None:
    profiles = list((DATA / dataset / task).glob("profiled*.csv"))
    if not profiles:
        return None
    import csv
    with profiles[0].open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    key = next((name for name in rows[0] if "coverage" in name.lower()), None)
    return (sum(float(row[key]) for row in rows) / len(rows) / num_points
            if key else None)


def main() -> None:
    build_template = json.loads((REPO / "experiments/multilevel_special/config.auto_policy_build.json").read_text())
    for dataset, tasks in TASKS.items():
        source = RESULT / dataset / "index/Trie_block_hybrid/index_files"
        source_meta = meta(source / "meta")
        legacy_block_meta = meta(RESULT / dataset / "index/Trie_block_hybrid/block_index_files/meta")
        fingerprint = legacy_block_meta["source_ung_fingerprint"]
        points, _ = shape(DATA / dataset / f"{dataset}_base.bin")
        policy = json.loads((POLICY / f"{dataset}.json").read_text())["automatic_policy"]
        thresholds = policy["thresholds"]
        root = REPO / f"runs/auto_policy_cross_dataset_{dataset.lower()}"
        if not thresholds:
            raise ValueError(f"policy selected no layer for {dataset}")
        cases = [{"name": f"auto_layer1_t1_{thresholds[0]}", "min_points": thresholds[0]}]
        if len(thresholds) >= 2:
            cases.append({"name": f"auto_layer2_t1_{thresholds[0]}_t2_{thresholds[1]}",
                          "min_points": thresholds[0], "upper_min_points": thresholds[1]})
        build = dict(build_template)
        build.update({
            "main_index": str(source),
            "base_bin_file": str(DATA / dataset / f"{dataset}_base.bin"),
            "base_label_file": str(DATA / dataset / f"{dataset}_base_labels.txt"),
            "output_root": str(root / "build"),
            "expected_num_points": points,
            "expected_num_groups": int(source_meta["num_groups"]),
            "expected_source_fingerprint": fingerprint,
            "expected_base_labels_sha256": sha(DATA / dataset / f"{dataset}_base_labels.txt"),
            "cases": cases,
        })
        build_path = REPO / f"experiments/multilevel_special/config.auto_policy_cross_dataset_{dataset.lower()}_build.json"
        build_path.write_text(json.dumps(build, indent=2) + "\n")

        methods = [{
            "name": "layer0_plain", "layer_count": 0,
            "special_block_search": False, "entry_group_provider": "cpu_bruteforce_els",
            "env": {"UNG_DISABLE_ELS_REUSE": "1",
                    "UNG_DISABLE_CPU_ELS_WARMUP": "1",
                    "UNG_SPECIAL_BLOCK_ROOT_LABEL_COVERAGE": "1"},
            "lsearch_values": [50, 100, 250, 500, 1000, 2000, 4000, 8000, 16000, 32000],
        }]
        for case in cases:
            methods.append({
                "name": case["name"], "layer_count": 2 if "upper_min_points" in case else 1,
                "t1": case["min_points"], **({"t2": case["upper_min_points"]} if "upper_min_points" in case else {}),
                "special_block_search": True, "entry_group_provider": "cpu_bruteforce_els",
                "block_index": str(root / "build" / case["name"] / "block_index"),
                "lsearch_values": [25, 50, 100, 250, 500, 1000, 2000, 4000, 8000, 16000],
                "env": {"UNG_DISABLE_ELS_REUSE": "1", "UNG_DISABLE_CPU_ELS_WARMUP": "1",
                        "UNG_SPECIAL_BLOCK_ROOT_LABEL_COVERAGE": "1"},
            })
        workloads = []
        for task in tasks:
            query = DATA / dataset / task / f"{dataset}_query.bin"
            workloads.append({"name": task, "query_dir": task,
                              "mean_selectivity": profile_mean(dataset, task, points)})
        search = {
            "search_app": str(REPO / "build_ung_rel/apps/search_UNG_index"),
            "main_index": str(source), "data_root": str(DATA / dataset),
            "gt_root": str(RESULT / dataset / "GroundTruth"),
            "output_root": str(root / "screen"), "dataset": dataset,
            "expected_source_fingerprint": fingerprint,
            "expected_base_labels_sha256": sha(DATA / dataset / f"{dataset}_base_labels.txt"),
            "expected_main_index_labels_sha256": sha(source / "labels.txt"),
            "K": 10, "num_threads": 100, "num_entry_points": 16,
            "num_repeats": 3, "lsearch_values": [100], "require_stage_breakdown": True,
            "recall_thresholds": {task: .87 for task in tasks},
            "workloads": workloads, "methods": methods,
        }
        # Query counts differ across legacy datasets; the runner validates the
        # individual binary shape and label/GT row counts when no global count
        # is declared.
        search_path = REPO / f"experiments/multilevel_special/config.auto_policy_cross_dataset_{dataset.lower()}_screen.json"
        search_path.write_text(json.dumps(search, indent=2) + "\n")
        print(build_path)
        print(search_path)


if __name__ == "__main__":
    main()
