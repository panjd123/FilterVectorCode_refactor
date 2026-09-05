#!/usr/bin/env python3
"""Run a resumable Special Block search matrix on fixed queries and GT.

The core comparison deliberately keeps the main UNG index and ELS provider
constant.  A method changes only the optional Special Block overlay and its
runtime settings, so the measured delta can be attributed to the overlay.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shlex
import struct
import subprocess
import time
from pathlib import Path
from typing import Any


def read_bin_shape(path: Path) -> tuple[int, int]:
    with path.open("rb") as stream:
        header = stream.read(8)
    if len(header) != 8:
        raise ValueError(f"invalid vector binary header: {path}")
    return tuple(int(value) for value in struct.unpack("<II", header))


def clean_method_env(base: dict[str, str], method: dict[str, Any]) -> dict[str, str]:
    env = dict(base)
    for key in list(env):
        # The search binary exposes many behavior switches through UNG_*.  A
        # shell left over from another experiment must not silently change a
        # case in this matrix, including ELS reuse/warmup controls.
        if key.startswith("UNG_"):
            env.pop(key, None)
    env["UNG_DISABLE_ENTRY_ROUTE_STATS"] = "1"
    env["UNG_SPECIAL_LIGHT_STATS"] = "1"
    if method.get("special_block_search", False):
        env["UNG_SPECIAL_BLOCK_SEARCH"] = "1"
    for key, value in method.get("env", {}).items():
        if value is None:
            env.pop(key, None)
        else:
            env[str(key)] = str(value)
    return env


def output_dir(config: dict[str, Any], method: dict[str, Any], workload: dict[str, Any]) -> Path:
    return Path(config["output_root"]) / method["name"] / workload["name"]


def result_is_complete(path: Path, expected_lsearch: list[int]) -> bool:
    summary = path / "search_time_summary.csv"
    if not summary.exists():
        return False
    with summary.open(newline="") as stream:
        seen = {int(row["Lsearch"]) for row in csv.DictReader(stream)}
    return seen == set(expected_lsearch)


def build_command(config: dict[str, Any], method: dict[str, Any], workload: dict[str, Any],
                  run_dir: Path) -> list[str]:
    data_root = Path(config["data_root"])
    query_root = data_root / workload["query_dir"]
    dataset = config.get("dataset", "Amazon")
    values = [int(value) for value in method.get("lsearch_values", config["lsearch_values"])]
    query_group_file = query_root / f"{dataset}_query_source_groups.txt"
    if not query_group_file.exists():
        query_group_file = run_dir / "missing_query_source_groups.txt"
    cmd = [
        str(config["search_app"]),
        "--data_type", "float",
        "--dataset", dataset,
        "--dist_fn", "L2",
        "--num_threads", str(config["num_threads"]),
        "--K", str(config["K"]),
        "--num_repeats", str(config["num_repeats"]),
        "--is_new_method", "true",
        "--force_use_alg", str(method.get("force_use_alg", 1)),
        "--is_idea2_available", "false",
        "--is_new_trie_method", "false",
        "--is_rec_more_start", "false",
        "--is_ung_more_entry", "false",
        "--base_bin_file", str(data_root / f"{dataset}_base.bin"),
        "--query_bin_file", str(query_root / f"{dataset}_query.bin"),
        "--query_label_file", str(query_root / f"{dataset}_query_labels.txt"),
        "--query_group_id_file", str(query_group_file),
        "--gt_file", str(Path(config["gt_root"]) / workload["query_dir"] / f"{dataset}_gt_labels_containment.bin"),
        "--index_path_prefix", str(Path(config["main_index"])) + "/",
        "--result_path_prefix", str(run_dir) + "/",
        "--selector_model_prefix", str(config.get("selector_model_prefix", "/nonexistent")),
        "--scenario", "containment",
        "--num_entry_points", str(config["num_entry_points"]),
        "--Lsearch", *[str(value) for value in values],
        "--lsearch_start", str(min(values)),
        "--lsearch_step", "1",
        "--efs_start", "10",
        "--efs_step_slow", "10",
        "--efs_step_fast", "10",
        "--lsearch_threshold", str(max(values)),
        "--entry_group_provider", str(method.get("entry_group_provider", "cpu_bruteforce_els")),
        "--graph_search_backend", str(method.get("graph_search_backend", "neighbor_list")),
        "--skip_query_features", "true",
        "--skip_bitmap_comparison", "true",
    ]
    block_index = method.get("block_index")
    if block_index:
        cmd.extend(["--block_index_path_prefix", str(Path(block_index)) + "/"] )
    return cmd


def validate_case(config: dict[str, Any], method: dict[str, Any], workload: dict[str, Any]) -> int:
    dataset = config.get("dataset", "Amazon")
    query_root = Path(config["data_root"]) / workload["query_dir"]
    query_bin = query_root / f"{dataset}_query.bin"
    query_labels = query_root / f"{dataset}_query_labels.txt"
    gt_file = Path(config["gt_root"]) / workload["query_dir"] / f"{dataset}_gt_labels_containment.bin"
    main_meta = Path(config["main_index"]) / "meta"
    required = [Path(config["search_app"]), query_bin, query_labels, gt_file, main_meta]
    if method.get("block_index"):
        required.append(Path(method["block_index"]) / "meta")
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("missing required artifacts:\n" + "\n".join(missing))
    num_queries, dimension = read_bin_shape(query_bin)
    if dimension <= 0 or num_queries <= 0:
        raise ValueError(f"invalid query shape {num_queries}x{dimension}: {query_bin}")
    with query_labels.open() as stream:
        label_rows = sum(1 for line in stream if line.strip())
    if label_rows != num_queries:
        raise ValueError(f"query/label row mismatch: {num_queries} != {label_rows} for {query_root}")
    # compute_groundtruth stores K uint32 IDs followed by K float distances
    # for every query, with no global header.
    expected_gt_bytes = num_queries * int(config["K"]) * 8
    if gt_file.stat().st_size != expected_gt_bytes:
        raise ValueError(
            f"GT size mismatch: {gt_file.stat().st_size} != {expected_gt_bytes} for {gt_file}"
        )
    return num_queries


def update_manifest(path: Path, record: dict[str, Any]) -> None:
    state: dict[str, Any] = {"schema_version": 1, "runs": []}
    if path.exists():
        state = json.loads(path.read_text())
    runs = [item for item in state.get("runs", [])
            if (item.get("method"), item.get("workload")) != (record["method"], record["workload"])]
    runs.append(record)
    state["runs"] = sorted(runs, key=lambda item: (item["method"], item["workload"]))
    state["updated_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(state, indent=2) + "\n")
    temporary.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=Path)
    parser.add_argument("--method", action="append", default=[])
    parser.add_argument("--workload", action="append", default=[])
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    output_root = Path(config["output_root"])
    output_root.mkdir(parents=True, exist_ok=True)
    manifest_path = output_root / "manifest.json"
    selected_methods = set(args.method)
    selected_workloads = set(args.workload)

    for method in config["methods"]:
        if selected_methods and method["name"] not in selected_methods:
            continue
        for workload in config["workloads"]:
            if selected_workloads and workload["name"] not in selected_workloads:
                continue
            run_dir = output_dir(config, method, workload)
            values = [int(value) for value in method.get("lsearch_values", config["lsearch_values"])]
            num_queries = validate_case(config, method, workload)
            if result_is_complete(run_dir, values) and not args.force:
                print(f"[SKIP] {method['name']}/{workload['name']} complete", flush=True)
                continue
            run_dir.mkdir(parents=True, exist_ok=True)
            cmd = build_command(config, method, workload, run_dir)
            (run_dir / "command.txt").write_text(shlex.join(cmd) + "\n")
            effective_env = clean_method_env(os.environ, method)
            (run_dir / "environment.json").write_text(
                json.dumps({key: value for key, value in effective_env.items() if key.startswith("UNG_")},
                           indent=2, sort_keys=True) + "\n"
            )
            run_record = {
                "method": method["name"],
                "workload": workload["name"],
                "query_dir": workload["query_dir"],
                "mean_selectivity": workload.get("mean_selectivity"),
                "num_queries": num_queries,
                "lsearch_values": values,
                "num_repeats": config["num_repeats"],
                "status": "dry_run" if args.dry_run else "running",
                "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                "run_dir": str(run_dir),
            }
            update_manifest(manifest_path, run_record)
            print(f"[RUN] {method['name']}/{workload['name']}", flush=True)
            print(shlex.join(cmd), flush=True)
            if args.dry_run:
                continue
            start = time.monotonic()
            with (run_dir / "search.log").open("w") as log_file:
                completed = subprocess.run(
                    cmd, env=effective_env,
                    stdout=log_file, stderr=subprocess.STDOUT, text=True,
                )
            run_record["elapsed_seconds"] = time.monotonic() - start
            run_record["returncode"] = completed.returncode
            run_record["finished_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
            run_record["status"] = "complete" if completed.returncode == 0 else "failed"
            update_manifest(manifest_path, run_record)
            if completed.returncode != 0:
                raise RuntimeError(f"search failed for {method['name']}/{workload['name']}; see {run_dir / 'search.log'}")
            if not result_is_complete(run_dir, values):
                raise RuntimeError(f"incomplete summary for {method['name']}/{workload['name']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
