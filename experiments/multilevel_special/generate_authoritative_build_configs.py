#!/usr/bin/env python3
"""Generate frozen base and hierarchy build-time experiment matrices."""

from __future__ import annotations

import argparse
import json
from copy import deepcopy
from pathlib import Path


REPO = Path("/home/sunyahui/worktrees/FilterVectorCode_multilevel_special")
RUN_ROOT = REPO / "runs/authoritative_multilevel_20260925/build_study"
DATA_ROOT = Path("/home/graphdb/FilterVectorData/Amazon")
SOURCE_INDEX = (
    RUN_ROOT / "base_timing/accelerated_gpu_measured_r0/index_files")

ACCELERATED_BASE_ENV = {
    "UNG_BUILD_PROFILE": "custom",
    "UNG_GROUP_GRAPH_IMPL": "3",
    "UNG_TAGORE_MIN_GROUP_SIZE": "128",
    "UNG_TAGORE_K": "64",
    "UNG_TAGORE_ITER": "4",
    "UNG_TAGORE_M": "64",
    "UNG_FAST_GRNND_LIGHT_PRUNE_NX": "256",
    "UNG_FAST_GRNND_LIGHT_HEAD": "16",
    "UNG_FAST_GRNND_REPAIR_DEGREE": "1",
    "UNG_FAST_GRNND_BATCH_EXACT_NX": "4096",
    "UNG_FAST_EXACT_DIRECT_H2D": "1",
    "UNG_FAST_EXACT_DEVICE_LOOKUP": "1",
    "UNG_FAST_EXACT_ANCHOR_TAIL": "1",
    "UNG_FAST_EXACT_ANCHOR_SLOTS": "4",
    "UNG_FAST_EXACT_BIDIR_ANCHOR": "1",
    "UNG_FAST_EXACT_REVERSE_CAP": "4",
    "UNG_FAST_EXACT_REVERSE_SLOTS": "4",
    "UNG_FAST_EXACT_REVERSE_FORWARD_CAP": "16",
    "UNG_GET_MIN_SUPER_SETS_IMPL": "0",
    "UNG_LNG_IMPL": "0",
    "UNG_DESCENDANTS_IMPL": "0",
    "UNG_COVERAGE_IMPL": "1",
    "UNG_COVERAGE_THREADS": "128",
    "UNG_CROSS_EDGE_IMPL": "1",
    "UNG_ADDITIONAL_EDGES_IMPL": "2",
    "UNG_ADDITIONAL_DIRECT_APPEND": "0",
    "UNG_GPU_TOPK_IMPL": "3",
    "UNG_CROSS_EDGE_GPU_STRICT": "1",
    "UNG_TAGORE_COMPACT_D2H": "1",
    "UNG_TAGORE_FILL_THREADS": "16",
}

COMMON_HIERARCHY_ENV = {
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

BASE_PROFILES = {
    "original_cpu": {"UNG_BUILD_PROFILE": "original_cpu"},
    "current_cpu": {"UNG_BUILD_PROFILE": "current_cpu"},
    "naive_gpu": {
        "UNG_BUILD_PROFILE": "naive_gpu",
        "UNG_CROSS_EDGE_GPU_STRICT": "1",
    },
    "paper_fused": {
        "UNG_BUILD_PROFILE": "paper_fused",
        "UNG_CROSS_EDGE_GPU_STRICT": "1",
    },
    "accelerated_gpu": ACCELERATED_BASE_ENV,
}

HIERARCHY_PROFILES = {
    "cpu": {
        "UNG_SPECIAL_INTRA_ROUTE": "1",
        "UNG_SPECIAL_INTRA_LARGE_BACKEND": "cpu_vamana",
        "UNG_SPECIAL_BLOCK_GPU_INTRA": "0",
        "UNG_SPECIAL_BLOCK_GPU_INTER": "0",
    },
    "hybrid_gpu_intra": {
        "UNG_SPECIAL_INTRA_ROUTE": "1",
        "UNG_SPECIAL_INTRA_LARGE_BACKEND": "jasper_style",
        "UNG_SPECIAL_BLOCK_GPU_INTRA": "0",
        "UNG_SPECIAL_BLOCK_GPU_INTER": "0",
    },
    "hybrid_gpu_intra_inter": {
        "UNG_SPECIAL_INTRA_ROUTE": "1",
        "UNG_SPECIAL_INTRA_LARGE_BACKEND": "jasper_style",
        "UNG_SPECIAL_BLOCK_GPU_INTRA": "0",
        "UNG_SPECIAL_BLOCK_GPU_INTER": "1",
        "UNG_SPECIAL_GPU_INTER_WMMA": "0",
    },
    "full_gpu": {
        "UNG_SPECIAL_INTRA_ROUTE": "0",
        "UNG_SPECIAL_BLOCK_GPU_INTRA": "1",
        "UNG_SPECIAL_BLOCK_GPU_INTER": "1",
        "UNG_SPECIAL_GPU_INTER_WMMA": "0",
    },
    "full_gpu_wmma": {
        "UNG_SPECIAL_INTRA_ROUTE": "0",
        "UNG_SPECIAL_BLOCK_GPU_INTRA": "1",
        "UNG_SPECIAL_BLOCK_GPU_INTER": "1",
        "UNG_SPECIAL_GPU_INTER_WMMA": "1",
    },
}

PREDECLARED_STRUCTURES = {
    "single_t1024_lng": [{"min_points": 1024, "topology": "lng"}],
    "auto_drh_v1": [
        {"min_points": 1024, "topology": "lng"},
        {"min_points": 16384, "topology": "trie"},
    ],
}


def repetitions(count: int, include_cold: bool):
    if include_cold:
        yield "cold", 0
    for repeat in range(count):
        yield "measured", repeat


def make_base_config(repeats: int, resource_profile: bool) -> dict:
    cases = []
    for role, repeat in repetitions(1 if resource_profile else repeats,
                                    not resource_profile):
        for profile, env in BASE_PROFILES.items():
            cases.append({
                "name": f"{profile}_{role}_r{repeat}",
                "base_topology": "lng",
                "benchmark_profile": profile,
                "timing_role": role,
                "repeat": repeat,
                "env": deepcopy(env),
            })
    suffix = "resource" if resource_profile else "timing"
    return {
        "schema_version": 1,
        "purpose": "End-to-end zero-layer base-build backend comparison.",
        "build_app": str(REPO / "build_ung_rel/apps/build_UNG_index"),
        "data_root": str(DATA_ROOT),
        "output_root": str(RUN_ROOT / f"base_{suffix}"),
        "dataset": "Amazon",
        "expected_num_points": 602453,
        "expected_base_labels_sha256":
            "aec768bba7092af445252835be3f1ef7f708305ea646af19ae72738df3dd2f96",
        "num_threads": 100,
        "max_degree": 32,
        "Lbuild": 100,
        "alpha": 1.2,
        "num_cross_edges": 4,
        "scenario": "general",
        "resource_profile": resource_profile,
        "resource_sample_interval_seconds": 0.2,
        "cases": cases,
    }


def make_hierarchy_config(repeats: int, resource_profile: bool) -> dict:
    cases = []
    for role, repeat in repetitions(1 if resource_profile else repeats,
                                    not resource_profile):
        for structure, layers in PREDECLARED_STRUCTURES.items():
            for profile, env in HIERARCHY_PROFILES.items():
                cases.append({
                    "name": f"{structure}_{profile}_{role}_r{repeat}",
                    "base_topology": "lng",
                    "hierarchy_layers": deepcopy(layers),
                    "structure": structure,
                    "benchmark_profile": profile,
                    "timing_role": role,
                    "repeat": repeat,
                    "env": deepcopy(env),
                })
    suffix = "resource" if resource_profile else "timing"
    return {
        "schema_version": 2,
        "purpose": "CPU/GPU hierarchy construction with fixed structure semantics.",
        "build_app": str(REPO / "build_ung_rel/apps/build_special_block_index"),
        "main_index": str(SOURCE_INDEX),
        "base_bin_file": str(DATA_ROOT / "Amazon_base.bin"),
        "base_label_file": str(DATA_ROOT / "Amazon_base_labels.txt"),
        "output_root": str(RUN_ROOT / f"hierarchy_{suffix}"),
        "expected_num_points": 602453,
        "expected_num_groups": 482387,
        "expected_source_fingerprint": "91d78580ae29f468",
        "expected_base_labels_sha256":
            "aec768bba7092af445252835be3f1ef7f708305ea646af19ae72738df3dd2f96",
        "base_topology": "lng",
        "num_threads": 100,
        "min_points": 1024,
        "max_degree": 64,
        "num_cross_edges": 4,
        "Lbuild": 100,
        "alpha": 1.2,
        "resource_profile": resource_profile,
        "resource_sample_interval_seconds": 0.2,
        "env": COMMON_HIERARCHY_ENV,
        "cases": cases,
    }


def write(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2) + "\n")
    print(path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args()
    if args.repeats <= 0:
        parser.error("--repeats must be positive")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write(args.output_dir / "config.authoritative_amazon_base_build_timing.json",
          make_base_config(args.repeats, False))
    write(args.output_dir / "config.authoritative_amazon_base_build_resource.json",
          make_base_config(args.repeats, True))
    write(args.output_dir / "config.authoritative_amazon_hierarchy_build_timing.json",
          make_hierarchy_config(args.repeats, False))
    write(args.output_dir / "config.authoritative_amazon_hierarchy_build_resource.json",
          make_hierarchy_config(args.repeats, True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
