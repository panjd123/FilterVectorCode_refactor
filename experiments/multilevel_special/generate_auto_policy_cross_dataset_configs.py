#!/usr/bin/env python3
"""Generate gated DRH and fair manual-oracle configs for held-out datasets.

The automatic hierarchy is derived only from N, R, and C. Every hierarchy uses
the same parameter-free exact authorization gate, and the surrounding manual
grid is declared before any held-out query result is read.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import struct
from pathlib import Path
from typing import Any

import experiment_core
import gpu_isolation
from derive_static_hierarchy import derive_plan


REPO = Path("/home/sunyahui/worktrees/FilterVectorCode_multilevel_special")
DATA = Path("/home/graphdb/FilterVectorData")
RESULT = Path("/home/graphdb/FilterVectorResult")
DEFAULT_RUN_ROOT = REPO / "runs/authoritative_multilevel_20260926_emptyfix/heldout"
DATASETS = {
    "Genome": ("query_minlen5_cov0.1k", "query_minlen2_cov1k"),
    "Reviews": ("query_minlen5_cov0.1k", "query_minlen2_cov1k"),
    "VariousImg": ("query_minlen2_cov5k",),
}
SCREEN_LSEARCH = [40, 100, 250, 500, 1000, 2500, 5000, 10000, 25000, 40000]
AUTOMATIC_ROLE = "predeclared_degree_ratio_hierarchy_v1"
MANUAL_ROLE = "predeclared_manual_oracle_grid"
UNROUTED_ABLATION_ROLE = "degree_ratio_hierarchy_v1_unrouted_ablation"
ROUTING_POLICY = "require_upper_authorization"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_meta(path: Path) -> dict[str, str]:
    return dict(
        line.split("=", 1)
        for line in path.read_text().splitlines()
        if "=" in line
    )


def binary_shape(path: Path) -> tuple[int, int]:
    with path.open("rb") as stream:
        header = stream.read(8)
    if len(header) != 8:
        raise ValueError(f"invalid vector binary: {path}")
    return struct.unpack("<II", header)


def profile_mean(dataset: str, task: str, num_points: int) -> float:
    profiles = sorted((DATA / dataset / task).glob("profiled*.csv"))
    if not profiles:
        raise FileNotFoundError(f"missing selectivity profile: {dataset}/{task}")
    with profiles[0].open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError(f"empty selectivity profile: {profiles[0]}")
    key = next((name for name in rows[0] if "coverage" in name.lower()), None)
    if key is None:
        raise ValueError(f"coverage column is absent from {profiles[0]}")
    return sum(float(row[key]) for row in rows) / len(rows) / num_points


def plan_name(layers: list[dict[str, int | str]]) -> str:
    thresholds = "_".join(str(layer["min_points"]) for layer in layers)
    topologies = "".join(str(layer["topology"])[0] for layer in layers)
    return f"l{len(layers)}_t{thresholds}_{topologies}"


def declared_candidate_plans(
    automatic: list[dict[str, int | str]], num_points: int,
    scale_ratio: int,
) -> list[dict[str, Any]]:
    """Return a query-independent 1/2/3-level neighborhood around DRH."""
    if len(automatic) < 2:
        raise ValueError("the held-out oracle protocol requires a two-level DRH plan")
    t1 = int(automatic[0]["min_points"])
    t2 = int(automatic[1]["min_points"])
    threshold_sequences: list[tuple[int, ...]] = []
    for threshold in (max(1, t1 // 2), t1, 2 * t1, t2):
        if threshold < num_points:
            threshold_sequences.append((threshold,))
    for pair in (
        (max(1, t1 // 2), max(2, t2 // 2)),
        (t1, max(2, t2 // 2)),
        (t1, t2),
        (2 * t1, t2),
        (2 * t1, 2 * t2),
    ):
        if 0 < pair[0] < pair[1] < num_points:
            threshold_sequences.append(pair)
    t3 = t2 * scale_ratio
    if t3 < num_points:
        threshold_sequences.append((t1, t2, t3))

    automatic_signature = tuple(
        (int(layer["min_points"]), str(layer["topology"]))
        for layer in automatic
    )
    result: list[dict[str, Any]] = []
    seen: set[tuple[tuple[int, str], ...]] = set()
    for thresholds in dict.fromkeys(threshold_sequences):
        for topology_tuple in itertools.product(("lng", "trie"), repeat=len(thresholds)):
            signature = tuple(zip(thresholds, topology_tuple))
            if signature in seen:
                continue
            seen.add(signature)
            layers = [
                {"min_points": threshold, "topology": topology}
                for threshold, topology in signature
            ]
            result.append({
                "name": plan_name(layers),
                "base_topology": "lng",
                "hierarchy_layers": layers,
                "selection_role": (
                    AUTOMATIC_ROLE
                    if signature == automatic_signature
                    else MANUAL_ROLE
                ),
                "benchmark_profile": "hybrid_gpu_intra",
                "env": {
                    "UNG_SPECIAL_INTRA_ROUTE": "1",
                    "UNG_SPECIAL_INTRA_LARGE_BACKEND": "jasper_style",
                    "UNG_SPECIAL_BLOCK_GPU_INTRA": "0",
                    "UNG_SPECIAL_BLOCK_GPU_INTER": "0",
                },
            })
    if sum(case["selection_role"] == AUTOMATIC_ROLE
           for case in result) != 1:
        raise AssertionError("manual grid must contain the automatic plan exactly once")
    return result


def common_build_env() -> dict[str, str]:
    return {
        "UNG_SPECIAL_BLOCK_DATA_MODE": "x1",
        "UNG_SPECIAL_BLOCK_PARTITION": "trie",
        "UNG_SPECIAL_INTRA_SMALL_NX": "2048",
        "UNG_SPECIAL_INTRA_LARGE_NX": "8192",
        "UNG_SPECIAL_INTRA_MID_SAMPLE_CANDIDATES": "256",
        "UNG_SPECIAL_INTRA_SAMPLE_MODE": "vamana_prune",
        "UNG_SPECIAL_INTRA_SAMPLE_CANDIDATES": "256",
        "UNG_SPECIAL_INTRA_SAMPLE_MIN_N": "257",
        "UNG_TAGORE_K": "128",
        "UNG_TAGORE_ITER": "4",
        "UNG_TAGORE_M": "64",
        "UNG_SPECIAL_INTER_FORCE_GRAPH_PAIR_WORK": "10000",
        "UNG_SPECIAL_INTER_SEARCH_EF": "32",
        "UNG_SPECIAL_INTER_ROUTE_PROFILE": "1",
        "UNG_SPECIAL_EDGE_BINARY_ONLY": "1",
    }


def common_search_env() -> dict[str, str]:
    return {
        "UNG_DISABLE_ELS_REUSE": "1",
        "UNG_DISABLE_CPU_ELS_WARMUP": "1",
        "UNG_SPECIAL_BLOCK_ROOT_LABEL_COVERAGE": "1",
    }


def make_search_methods(
    source: Path, dataset_run_root: Path, cases: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Use one exact authorization gate for DRH and every manual candidate."""
    methods: list[dict[str, Any]] = [{
        "name": "l0_lng_entry_optimized_lng",
        "main_index": str(source),
        "base_topology": "lng",
        "entry_strategy": "optimized_lng",
        "hierarchy_layers": [],
        "special_block_search": False,
        "lsearch_values": SCREEN_LSEARCH,
        "selection_role": "zero_layer_baseline",
        "env": common_search_env(),
    }]
    for case in cases:
        common = {
            "main_index": str(source),
            "base_topology": "lng",
            "entry_strategy": "optimized_lng",
            "hierarchy_layers": case["hierarchy_layers"],
            "special_block_search": True,
            "block_index": str(dataset_run_root / "hierarchy" /
                               case["name"] / "block_index"),
            "lsearch_values": SCREEN_LSEARCH,
        }
        routed = {
            **common,
            "name": case["name"] + "_entry_optimized_lng_upper_routed",
            "selection_role": case["selection_role"],
            "routing_policy": ROUTING_POLICY,
            "env": {
                **common_search_env(),
                "UNG_SPECIAL_REQUIRE_UPPER_AUTHORIZATION": "1",
            },
        }
        methods.append(routed)
        if case["selection_role"] == AUTOMATIC_ROLE:
            methods.append({
                **common,
                "name": case["name"] + "_entry_optimized_lng",
                "selection_role": UNROUTED_ABLATION_ROLE,
                "env": common_search_env(),
            })
    return methods


def make_dataset_configs(
    dataset: str, tasks: tuple[str, ...], run_root: Path,
    max_degree: int, cross_edges: int, recall_threshold: float,
    search_app: Path,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    dataset_root = DATA / dataset
    result_root = RESULT / dataset
    source = result_root / "index/Trie_block_hybrid/index_files"
    source_meta = read_meta(source / "meta")
    legacy_block_meta = read_meta(
        result_root / "index/Trie_block_hybrid/block_index_files/meta")
    num_points, dimension = binary_shape(dataset_root / f"{dataset}_base.bin")
    automatic = derive_plan(num_points, max_degree, cross_edges)
    scale_ratio = max(2, round(max_degree / cross_edges))
    cases = declared_candidate_plans(automatic, num_points, scale_ratio)
    dataset_run_root = run_root / dataset.lower()

    build = {
        "schema_version": 2,
        "purpose": "Held-out DRH-v1 versus predeclared manual hierarchy grid.",
        "build_app": str(REPO / "build_ung_rel/apps/build_special_block_index"),
        "main_index": str(source),
        "base_bin_file": str(dataset_root / f"{dataset}_base.bin"),
        "base_label_file": str(dataset_root / f"{dataset}_base_labels.txt"),
        "output_root": str(dataset_run_root / "hierarchy"),
        "expected_num_points": num_points,
        "expected_num_groups": int(source_meta["num_groups"]),
        "expected_source_fingerprint": legacy_block_meta["source_ung_fingerprint"],
        "expected_base_labels_sha256": sha256_file(
            dataset_root / f"{dataset}_base_labels.txt"),
        "base_topology": "lng",
        "num_threads": 100,
        "min_points": int(automatic[0]["min_points"]),
        "max_degree": max_degree,
        "num_cross_edges": cross_edges,
        "Lbuild": 100,
        "alpha": 1.2,
        "gpu_isolation": dict(gpu_isolation.DEFAULT_POLICY),
        "env": common_build_env(),
        "cases": cases,
    }

    methods = make_search_methods(source, dataset_run_root, cases)

    workloads = []
    for task in tasks:
        query_path = dataset_root / task / f"{dataset}_query.bin"
        num_queries, query_dimension = binary_shape(query_path)
        if query_dimension != dimension:
            raise ValueError(f"query dimension mismatch: {query_path}")
        workloads.append({
            "name": task,
            "query_dir": task,
            "mean_selectivity": profile_mean(dataset, task, num_points),
            "num_queries": num_queries,
        })

    search = {
        "schema_version": 2,
        "method_schema": "orthogonal_v2",
        "purpose": "Held-out gated DRH-v1 versus a gated manual hierarchy oracle.",
        "measurement_pass": "performance",
        "pass_subdirs": True,
        "search_app": str(search_app),
        "expected_search_binary_sha256": sha256_file(search_app),
        "main_index": str(source),
        "data_root": str(dataset_root),
        "gt_root": str(result_root / "GroundTruth"),
        "output_root": str(dataset_run_root / "search_screen"),
        "dataset": dataset,
        "expected_num_points": num_points,
        "expected_source_fingerprint": legacy_block_meta["source_ung_fingerprint"],
        "expected_base_labels_sha256": sha256_file(
            dataset_root / f"{dataset}_base_labels.txt"),
        "expected_main_index_labels_sha256": sha256_file(source / "labels.txt"),
        "K": 10,
        "num_threads": 100,
        "num_entry_points": 16,
        "num_repeats": 3,
        "lsearch_values": [100],
        "max_lsearch": num_points,
        "require_stage_breakdown": True,
        "require_work_breakdown": False,
        "protocol": {
            "phase": "screen",
            "cold_repeats": 1,
            "measured_repeats": 2,
            "recall_rule": "all_repeats",
            "bootstrap_samples": 10000,
            "paired_repeats": False,
        },
        "recall_thresholds": {task: recall_threshold for task in tasks},
        "workloads": workloads,
        "methods": methods,
    }
    experiment_core.validate_config(search)
    policy = {
        "schema_version": 2,
        "dataset": dataset,
        "policy": "gated_degree_ratio_hierarchy_v1",
        "query_calibrated": False,
        "inputs": {
            "num_points": num_points,
            "dimension": dimension,
            "max_degree": max_degree,
            "num_cross_edges": cross_edges,
            "scale_ratio": scale_ratio,
        },
        "automatic_hierarchy_layers": automatic,
        "automatic_method": (
            plan_name(automatic) + "_entry_optimized_lng_upper_routed"),
        "automatic_routing_policy": ROUTING_POLICY,
        "unrouted_ablation_method": (
            plan_name(automatic) + "_entry_optimized_lng"),
        "manual_grid_frozen_before_search": True,
        "manual_grid_uses_automatic_routing_policy": True,
        "manual_hierarchy_cases": len(cases),
        "manual_depths": sorted({len(case["hierarchy_layers"]) for case in cases}),
        "workloads": workloads,
    }
    return build, search, policy


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    parser.add_argument("--dataset", action="append", choices=sorted(DATASETS))
    parser.add_argument("--max-degree", type=int, default=64)
    parser.add_argument("--num-cross-edges", type=int, default=4)
    parser.add_argument("--recall-threshold", type=float, default=0.90)
    parser.add_argument(
        "--search-app", type=Path,
        default=REPO / "build_ung_rel/apps/search_UNG_index")
    args = parser.parse_args()
    selected = args.dataset or list(DATASETS)
    config_root = REPO / "experiments/multilevel_special"
    for dataset in selected:
        build, search, policy = make_dataset_configs(
            dataset, DATASETS[dataset], args.run_root.resolve(),
            args.max_degree, args.num_cross_edges, args.recall_threshold,
            args.search_app.resolve())
        stem = f"config.authoritative_heldout_{dataset.lower()}"
        build_path = config_root / f"{stem}_build.json"
        screen_path = config_root / f"{stem}_screen.json"
        policy_path = args.run_root.resolve() / dataset.lower() / "policy.json"
        policy_path.parent.mkdir(parents=True, exist_ok=True)
        build_path.write_text(json.dumps(build, indent=2) + "\n")
        screen_path.write_text(json.dumps(search, indent=2) + "\n")
        policy_path.write_text(json.dumps(policy, indent=2) + "\n")
        print(build_path)
        print(screen_path)
        print(policy_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
