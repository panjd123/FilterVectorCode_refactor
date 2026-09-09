#!/usr/bin/env python3
"""Run a resumable Special Block search matrix on fixed queries and GT.

The core comparison deliberately keeps the main UNG index and ELS provider
constant.  A method changes only the optional Special Block overlay and its
runtime settings, so the measured delta can be attributed to the overlay.
"""

from __future__ import annotations

import argparse
import csv
import fcntl
import hashlib
import json
import os
import shlex
import shutil
import struct
import subprocess
import time
from pathlib import Path
from typing import Any


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_meta(path: Path) -> dict[str, str]:
    result: dict[str, str] = {}
    for line in path.read_text().splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            result[key.strip()] = value.strip()
    return result


def snapshot_search_app(search_app: Path, output_root: Path) -> tuple[Path, str]:
    """Use one immutable executable for the whole sweep.

    Rebuilding the normal build-tree target while a sweep is active otherwise
    changes semantics midway through an apparently valid manifest.
    """
    digest = sha256_file(search_app)
    snapshot_dir = output_root / ".binary_snapshots"
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    snapshot = snapshot_dir / f"search_UNG_index.{digest}"
    if not snapshot.exists():
        temporary = snapshot.with_suffix(".tmp")
        shutil.copy2(search_app, temporary)
        if sha256_file(temporary) != digest:
            temporary.unlink(missing_ok=True)
            raise RuntimeError("search binary changed while it was being snapshotted")
        temporary.chmod(0o555)
        temporary.replace(snapshot)
    elif sha256_file(snapshot) != digest:
        raise RuntimeError(f"corrupt search binary snapshot: {snapshot}")
    return snapshot, digest


def validate_provenance(config: dict[str, Any], method: dict[str, Any]) -> dict[str, str]:
    main_dir = Path(config["main_index"])
    base_labels = Path(config["data_root"]) / f"{config.get('dataset', 'Amazon')}_base_labels.txt"
    required = [main_dir / "meta", main_dir / "labels.txt", base_labels]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("missing provenance inputs:\n" + "\n".join(missing))
    expected_hash = config.get("expected_base_labels_sha256")
    base_hash = sha256_file(base_labels)
    main_labels_hash = sha256_file(main_dir / "labels.txt")
    if expected_hash and base_hash != expected_hash:
        raise ValueError("configured base labels hash does not match the query dataset")
    expected_main_hash = config.get("expected_main_index_labels_sha256")
    if expected_main_hash and main_labels_hash != expected_main_hash:
        raise ValueError("configured main-index labels hash does not match the source graph")
    expected_fingerprint = config.get("expected_source_fingerprint")
    block_index = method.get("block_index")
    if block_index:
        block_meta = parse_meta(Path(block_index) / "meta")
        actual_fingerprint = block_meta.get("source_ung_fingerprint")
        if expected_fingerprint and actual_fingerprint != expected_fingerprint:
            raise ValueError(
                f"block/main source fingerprint mismatch for {method['name']}: "
                f"{actual_fingerprint} != {expected_fingerprint}"
            )
        main_meta = parse_meta(main_dir / "meta")
        for key in ("num_points", "num_groups"):
            if block_meta.get(key) != main_meta.get(key):
                raise ValueError(
                    f"block/main {key} mismatch for {method['name']}: "
                    f"{block_meta.get(key)} != {main_meta.get(key)}"
                )
        layer_count = int(method.get("layer_count", 0))
        expected_t1 = method.get("t1")
        actual_t1 = int(block_meta.get("special_block_min_points", "-1"))
        if expected_t1 is None or actual_t1 != int(expected_t1):
            raise ValueError(
                f"block T1 mismatch for {method['name']}: {actual_t1} != {expected_t1}")
        actual_t2 = int(block_meta.get("special_block_upper_min_points", "0"))
        upper_count = int(block_meta.get("special_block_upper_count", "0"))
        if layer_count == 1 and (actual_t2 != 0 or upper_count != 0):
            raise ValueError(
                f"one-level method {method['name']} points to a multilevel block index")
        if layer_count == 2:
            expected_t2 = method.get("t2")
            if expected_t2 is None or actual_t2 != int(expected_t2) or upper_count <= 0:
                raise ValueError(
                    f"two-level block mismatch for {method['name']}: "
                    f"T2={actual_t2}, upper_count={upper_count}, expected T2={expected_t2}")
    return {
        "base_labels_sha256": base_hash,
        "main_index_labels_sha256": main_labels_hash,
        "expected_source_fingerprint": str(expected_fingerprint or ""),
    }


def acquire_run_lock(output_root: Path):
    """Hold an exclusive lock for one sweep output tree.

    Two runners writing the same case can overwrite CSVs and, more subtly,
    contend for all search threads while still producing plausible-looking
    timings.  Keep the returned file object alive for the process lifetime.
    """
    lock_path = output_root / ".runner.lock"
    lock_file = lock_path.open("a+")
    try:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as error:
        lock_file.seek(0)
        owner = lock_file.read().strip() or "unknown"
        lock_file.close()
        raise RuntimeError(
            f"another selection sweep holds {lock_path} (pid={owner})"
        ) from error
    lock_file.seek(0)
    lock_file.truncate()
    lock_file.write(str(os.getpid()) + "\n")
    lock_file.flush()
    return lock_file


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


def method_enabled_for_workload(method: dict[str, Any], workload: dict[str, Any]) -> bool:
    enabled = method.get("enabled_workloads")
    return enabled is None or workload["name"] in enabled


def lsearch_values_for(config: dict[str, Any], method: dict[str, Any],
                       workload: dict[str, Any]) -> list[int]:
    """Resolve an explicit L grid, allowing workload-specific tuning.

    Search difficulty differs by two orders of magnitude across the Amazon x1
    workloads.  A single global L grid either misses a Recall crossing or
    wastes most of the tuning budget, so the most specific declared grid wins.
    """
    workload_name = str(workload["name"])
    by_workload = method.get("lsearch_values_by_workload", {})
    values = by_workload.get(workload_name)
    if values is None:
        values = workload.get("lsearch_values")
    if values is None:
        values = method.get("lsearch_values", config["lsearch_values"])
    resolved = [int(value) for value in values]
    minimum_lsearch = int(config.get("K", 1))
    if (not resolved or len(set(resolved)) != len(resolved)
            or any(value < minimum_lsearch for value in resolved)):
        raise ValueError(
            f"invalid Lsearch grid for {method['name']}/{workload_name}: "
            f"{resolved}; every value must be >= K={minimum_lsearch}")
    return resolved


def result_is_complete(path: Path, expected_lsearch: list[int],
                       require_stage_breakdown: bool = False,
                       expected_repeats: int | None = None,
                       expected_command: list[str] | None = None,
                       expected_environment: dict[str, str] | None = None,
                       expected_binary_sha256: str | None = None) -> bool:
    summary = path / "search_time_summary.csv"
    if not summary.exists():
        return False
    with summary.open(newline="") as stream:
        seen = {int(row["Lsearch"]) for row in csv.DictReader(stream)}
    if seen != set(expected_lsearch):
        return False
    details = path / "search_time_details.csv"
    if expected_repeats is not None:
        if not details.exists():
            return False
        with details.open(newline="") as stream:
            detail_rows = list(csv.DictReader(stream))
        repeat_counts = {
            value: sum(int(row["Lsearch"]) == value for row in detail_rows)
            for value in expected_lsearch
        }
        if any(count != expected_repeats for count in repeat_counts.values()):
            return False
    command_path = path / "command.txt"
    if expected_command is not None:
        if not command_path.exists() or shlex.split(command_path.read_text()) != expected_command:
            return False
    environment_path = path / "environment.json"
    if expected_environment is not None:
        if not environment_path.exists():
            return False
        recorded = json.loads(environment_path.read_text())
        expected_ung = {key: value for key, value in expected_environment.items()
                        if key.startswith("UNG_")}
        if recorded != expected_ung:
            return False
    if expected_binary_sha256 is not None:
        if expected_command is None or not expected_command:
            return False
        binary = Path(expected_command[0])
        if not binary.is_file() or sha256_file(binary) != expected_binary_sha256:
            return False
    if not require_stage_breakdown:
        return True
    stage = path / "search_stage_details.csv"
    if not stage.exists():
        return False
    with stage.open(newline="") as stream:
        stage_rows = list(csv.DictReader(stream))
    stage_lsearch = {int(row["Lsearch"]) for row in stage_rows}
    required_columns = {
        "AverageQueryTotal_ms", "AverageELS_ms",
        "AverageEntryPointSetup_ms", "AverageBlockAuthorization_ms",
        "AverageGraphSearch_ms", "AverageResidual_ms", "ClosureError_ms",
    }
    return stage_lsearch == set(expected_lsearch) and bool(stage_rows) and \
        required_columns.issubset(stage_rows[0])


def build_command(config: dict[str, Any], method: dict[str, Any], workload: dict[str, Any],
                  run_dir: Path) -> list[str]:
    data_root = Path(config["data_root"])
    query_root = data_root / workload["query_dir"]
    dataset = config.get("dataset", "Amazon")
    values = lsearch_values_for(config, method, workload)
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
        "--scenario", str(workload.get("scenario", config.get("scenario", "containment"))),
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
    expected_num_queries = config.get("expected_num_queries")
    if expected_num_queries is not None and num_queries != int(expected_num_queries):
        raise ValueError(
            f"query count mismatch: {num_queries} != {expected_num_queries} for {query_bin}")
    with query_labels.open() as stream:
        # An empty label set is a valid containment query matching the whole
        # dataset. Count physical rows rather than non-empty rows so a 100%
        # control remains distinguishable from a truncated label file.
        label_rows = sum(1 for _ in stream)
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
    # Retain this handle until main returns; closing it releases the advisory
    # lock even after an exception or normal process exit.
    run_lock = acquire_run_lock(output_root)
    source_search_app = Path(config["search_app"])
    search_snapshot, search_binary_sha256 = snapshot_search_app(
        source_search_app, output_root)
    config["search_app"] = str(search_snapshot)
    manifest_path = output_root / "manifest.json"
    selected_methods = set(args.method)
    selected_workloads = set(args.workload)

    for method in config["methods"]:
        if selected_methods and method["name"] not in selected_methods:
            continue
        for workload in config["workloads"]:
            if not method_enabled_for_workload(method, workload):
                continue
            if selected_workloads and workload["name"] not in selected_workloads:
                continue
            run_dir = output_dir(config, method, workload)
            values = lsearch_values_for(config, method, workload)
            num_queries = validate_case(config, method, workload)
            provenance = validate_provenance(config, method)
            require_stage_breakdown = bool(config.get("require_stage_breakdown", False))
            expected_command = build_command(config, method, workload, run_dir)
            effective_env = clean_method_env(os.environ, method)
            if result_is_complete(
                    run_dir, values, require_stage_breakdown,
                    expected_repeats=int(config["num_repeats"]),
                    expected_command=expected_command,
                    expected_environment=effective_env,
                    expected_binary_sha256=search_binary_sha256) and not args.force:
                update_manifest(manifest_path, {
                    "method": method["name"],
                    "workload": workload["name"],
                    "query_dir": workload["query_dir"],
                    "mean_selectivity": workload.get("mean_selectivity"),
                    "num_queries": num_queries,
                    "lsearch_values": values,
                    "num_repeats": config["num_repeats"],
                    "layer_count": int(method.get("layer_count", 0)),
                    "t1": method.get("t1"),
                    "t2": method.get("t2"),
                    "status": "complete",
                    "reused_existing": True,
                    "run_dir": str(run_dir),
                    "provenance": provenance,
                    "source_search_app": str(source_search_app),
                    "search_binary_sha256": search_binary_sha256,
                    "els_reuse_disabled": effective_env.get("UNG_DISABLE_ELS_REUSE") == "1",
                    "require_stage_breakdown": require_stage_breakdown,
                })
                print(f"[SKIP] {method['name']}/{workload['name']} complete", flush=True)
                continue
            run_dir.mkdir(parents=True, exist_ok=True)
            cmd = expected_command
            (run_dir / "command.txt").write_text(shlex.join(cmd) + "\n")
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
                "layer_count": int(method.get("layer_count", 0)),
                "t1": method.get("t1"),
                "t2": method.get("t2"),
                "num_repeats": config["num_repeats"],
                "status": "dry_run" if args.dry_run else "running",
                "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                "run_dir": str(run_dir),
                "provenance": provenance,
                "source_search_app": str(source_search_app),
                "search_binary_sha256": search_binary_sha256,
                "els_reuse_disabled": effective_env.get("UNG_DISABLE_ELS_REUSE") == "1",
                "require_stage_breakdown": require_stage_breakdown,
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
            if not result_is_complete(run_dir, values, require_stage_breakdown):
                raise RuntimeError(f"incomplete summary for {method['name']}/{workload['name']}")
    run_lock.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
