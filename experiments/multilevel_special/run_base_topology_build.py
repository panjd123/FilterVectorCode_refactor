#!/usr/bin/env python3
"""Build resumable UNG base indexes for controlled LNG/Trie topology A/B tests."""

from __future__ import annotations

import argparse
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

import experiment_core
import gpu_isolation
import process_resource_probe


REQUIRED_INDEX_FILES = (
    "meta", "graph", "global_graph", "group_entry_points",
    "group_id_to_label_set", "group_id_to_range", "labels.txt",
    "new_to_old_vec_ids", "vecs.bin",
)

BASE_PROFILE_META = {
    "original_cpu": {
        "build_profile": "original_cpu", "group_graph_impl": "vamana_cpu",
        "cross_edge_impl": "original_cpu", "gpu_topk_impl": "auto",
    },
    "current_cpu": {
        "build_profile": "current_cpu", "group_graph_impl": "vamana_cpu",
        "cross_edge_impl": "cpu_vamana", "gpu_topk_impl": "auto",
    },
    "naive_gpu": {
        "build_profile": "naive_gpu", "group_graph_impl": "vamana_cpu",
        "cross_edge_impl": "gpu_batched", "gpu_topk_impl": "sgemm_topk",
    },
    "paper_fused": {
        "build_profile": "paper_fused", "group_graph_impl": "vamana_cpu",
        "cross_edge_impl": "gpu_batched", "gpu_topk_impl": "fused_group_topk",
    },
    "accelerated_gpu": {
        "build_profile": "custom", "group_graph_impl": "fast_grnnd_cuda",
        "cross_edge_impl": "gpu_batched", "gpu_topk_impl": "fused_group_topk",
    },
}


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


def acquire_lock(output_root: Path):
    path = output_root / ".base_builder.lock"
    handle = path.open("a+")
    try:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as error:
        handle.seek(0)
        owner = handle.read().strip() or "unknown"
        handle.close()
        raise RuntimeError(f"another base build holds {path} (pid={owner})") from error
    handle.seek(0)
    handle.truncate()
    handle.write(f"{os.getpid()}\n")
    handle.flush()
    return handle


def snapshot_binary(path: Path, output_root: Path) -> tuple[Path, str]:
    digest = sha256_file(path)
    directory = output_root / ".binary_snapshots"
    directory.mkdir(parents=True, exist_ok=True)
    snapshot = directory / f"build_UNG_index.{digest}"
    if not snapshot.exists():
        temporary = snapshot.with_suffix(".tmp")
        shutil.copy2(path, temporary)
        if sha256_file(temporary) != digest:
            temporary.unlink(missing_ok=True)
            raise RuntimeError("build binary changed while being snapshotted")
        temporary.chmod(0o555)
        temporary.replace(snapshot)
    return snapshot, digest


def update_manifest(path: Path, record: dict[str, Any]) -> None:
    state: dict[str, Any] = {"schema_version": 1, "runs": []}
    if path.exists():
        state = json.loads(path.read_text())
    runs = [item for item in state.get("runs", []) if item.get("name") != record["name"]]
    runs.append(record)
    state["runs"] = sorted(runs, key=lambda item: item["name"])
    state["updated_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(state, indent=2) + "\n")
    temporary.replace(path)


def validate_inputs(config: dict[str, Any]) -> dict[str, Any]:
    dataset = str(config["dataset"])
    data_root = Path(config["data_root"])
    base_bin = data_root / f"{dataset}_base.bin"
    base_labels = data_root / f"{dataset}_base_labels.txt"
    required = [Path(config["build_app"]), base_bin, base_labels]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("missing base build inputs:\n" + "\n".join(missing))
    with base_bin.open("rb") as stream:
        header = stream.read(8)
    if len(header) != 8:
        raise ValueError(f"invalid vector binary header: {base_bin}")
    points, dimension = struct.unpack("<II", header)
    expected = config.get("expected_num_points")
    if expected is not None and points != int(expected):
        raise ValueError(f"base point count mismatch: {points} != {expected}")
    labels_hash = sha256_file(base_labels)
    expected_hash = config.get("expected_base_labels_sha256")
    if expected_hash and labels_hash != expected_hash:
        raise ValueError(f"base-label hash mismatch: {labels_hash} != {expected_hash}")
    return {
        "num_points": points,
        "dimension": dimension,
        "base_bin_bytes": base_bin.stat().st_size,
        "base_labels_sha256": labels_hash,
    }


def clean_env(config: dict[str, Any], case: dict[str, Any], topology: str) -> dict[str, str]:
    env = {key: value for key, value in os.environ.items() if not key.startswith("UNG_")}
    env.update({str(key): str(value) for key, value in config.get("env", {}).items()})
    env.update({str(key): str(value) for key, value in case.get("env", {}).items()})
    env["UNG_BASE_GROUP_TOPOLOGY"] = topology
    env["UNG_HIERARCHY_LAYERS"] = ""
    env["UNG_SPECIAL_BLOCKS"] = "0"
    return env


def command(config: dict[str, Any], executable: Path, staging: Path) -> list[str]:
    dataset = str(config["dataset"])
    data_root = Path(config["data_root"])
    placeholders = Path(config["output_root"]) / ".placeholders"
    placeholders.mkdir(parents=True, exist_ok=True)
    info = placeholders / "base_labels_info.log"
    roots = placeholders / "tree_roots.txt"
    info.touch(exist_ok=True)
    roots.touch(exist_ok=True)
    return [
        str(executable), "--data_type", "float", "--dataset", dataset,
        "--dist_fn", "L2", "--num_threads", str(int(config["num_threads"])),
        "--max_degree", str(int(config["max_degree"])),
        "--Lbuild", str(int(config["Lbuild"])),
        "--alpha", str(float(config["alpha"])),
        "--num_cross_edges", str(int(config["num_cross_edges"])),
        "--base_bin_file", str(data_root / f"{dataset}_base.bin"),
        "--base_label_file", str(data_root / f"{dataset}_base_labels.txt"),
        "--base_label_info_file", str(info),
        "--base_label_tree_roots", str(roots),
        "--index_path_prefix", str(staging / "index_files") + "/",
        "--result_path_prefix", str(staging / "results") + "/",
        "--scenario", str(config.get("scenario", "general")),
    ]


def validate_case(config: dict[str, Any], case: dict[str, Any], root: Path) -> dict[str, str]:
    index = root / "index_files"
    missing = [str(index / name) for name in REQUIRED_INDEX_FILES
               if not (index / name).is_file() or (index / name).stat().st_size == 0]
    if missing:
        raise ValueError("missing or empty base-index files:\n" + "\n".join(missing))
    if config.get("resource_profile", False):
        resource_path = root / "resource_usage.json"
        if not resource_path.is_file() or resource_path.stat().st_size == 0:
            raise ValueError(f"missing resource profile: {resource_path}")
        resource = json.loads(resource_path.read_text())
        if int(resource.get("num_samples", 0)) <= 0:
            raise ValueError(f"empty resource profile: {resource_path}")
    meta = parse_meta(index / "meta")
    expected = {
        "base_group_topology": str(case["base_topology"]),
        "hierarchy_layers": "",
        "num_points": str(int(config["expected_num_points"])),
        "max_degree": str(int(config["max_degree"])),
        "num_cross_edges": str(int(config["num_cross_edges"])),
    }
    profile_value = case.get("benchmark_profile")
    profile = str(profile_value) if profile_value is not None else ""
    if profile:
        if profile not in BASE_PROFILE_META:
            raise ValueError(f"unknown benchmark profile for {case['name']}: {profile}")
        expected.update(BASE_PROFILE_META[profile])
    mismatches = {key: (meta.get(key), value) for key, value in expected.items()
                  if meta.get(key) != value}
    if mismatches:
        raise ValueError(f"metadata mismatch for {case['name']}: {mismatches}")
    if profile.endswith("gpu") or profile in {"naive_gpu", "paper_fused"}:
        build_log = root / "build.log"
        if not build_log.is_file():
            raise ValueError(f"missing GPU backend evidence: {build_log}")
        log_text = build_log.read_text(errors="replace")
        if "fallback to CPU" in log_text:
            raise ValueError(f"GPU build fell back to CPU for {case['name']}")
        if "[GPU GEMM]" not in log_text:
            raise ValueError(f"GPU cross-edge execution was not observed for {case['name']}")
    gpu_isolation.validate_case_evidence(case, root)
    return meta


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=Path)
    parser.add_argument("--case", action="append", default=[])
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    selected = set(args.case)
    selected_cases = [
        case for case in config["cases"]
        if not selected or case["name"] in selected
    ]
    if not args.dry_run and any(
            gpu_isolation.profile_uses_gpu(case) for case in selected_cases):
        device = int(gpu_isolation.policy_for(config)["device"])
        gpu_isolation.reexec_under_perf_lock(device)
    output_root = Path(config["output_root"])
    output_root.mkdir(parents=True, exist_ok=True)
    lock = acquire_lock(output_root)
    provenance = validate_inputs(config)
    executable, binary_hash = snapshot_binary(Path(config["build_app"]), output_root)
    provenance["build_binary_sha256"] = binary_hash
    manifest = output_root / "manifest.json"

    for case in config["cases"]:
        if selected and case["name"] not in selected:
            continue
        topology = str(case["base_topology"])
        if topology not in {"lng", "trie"}:
            raise ValueError(f"invalid base topology: {topology}")
        final = output_root / case["name"]
        if final.exists() and not args.force:
            try:
                meta = validate_case(config, case, final)
            except ValueError:
                pass
            else:
                previous = experiment_core.existing_manifest_record(
                    manifest, {"name": case["name"]})
                experiment_core.require_matching_build_binary_for_reuse(
                    previous, binary_hash)
                print(f"[SKIP] {case['name']} validated at {final}", flush=True)
                reused_record = {
                    "name": case["name"], "base_topology": topology,
                    "status": "complete", "reused_existing": True,
                    "case_root": str(final), "metadata": meta,
                    "provenance": provenance,
                }
                experiment_core.retain_elapsed_evidence(
                    reused_record, previous,
                    final, "command.txt", "build.log")
                update_manifest(manifest, reused_record)
                continue
        if final.exists():
            quarantine = output_root / "quarantine" / (
                f"{case['name']}_{time.strftime('%Y%m%dT%H%M%S')}")
            quarantine.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(final), str(quarantine))
        staging = output_root / ".staging" / f"{case['name']}_{os.getpid()}"
        if staging.exists():
            shutil.rmtree(staging)
        (staging / "index_files").mkdir(parents=True)
        (staging / "results").mkdir(parents=True)
        cmd = command(config, executable, staging)
        env = clean_env(config, case, topology)
        (staging / "command.txt").write_text(shlex.join(cmd) + "\n")
        (staging / "environment.json").write_text(json.dumps(
            {key: value for key, value in env.items() if key.startswith("UNG_")},
            indent=2, sort_keys=True) + "\n")
        record = {
            "name": case["name"], "base_topology": topology,
            "status": "dry_run" if args.dry_run else "running",
            "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "staging_root": str(staging), "provenance": provenance,
        }
        update_manifest(manifest, record)
        print(f"[BUILD] {case['name']} topology={topology}", flush=True)
        print(shlex.join(cmd), flush=True)
        if args.dry_run:
            continue
        try:
            isolation = gpu_isolation.prepare_case(config, case)
        except RuntimeError as error:
            record.update({
                "status": "blocked_gpu_isolation",
                "gpu_isolation_error": str(error),
                "finished_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            })
            update_manifest(manifest, record)
            raise
        (staging / "gpu_isolation.json").write_text(
            json.dumps(isolation, indent=2) + "\n")
        record["gpu_isolation"] = isolation
        update_manifest(manifest, record)
        with (staging / "build.log").open("w") as log:
            if config.get("resource_profile", False):
                resource_usage = process_resource_probe.run_profiled(
                    cmd, env, log,
                    float(config.get("resource_sample_interval_seconds", 0.2)))
                (staging / "resource_usage.json").write_text(
                    json.dumps(resource_usage, indent=2) + "\n")
                elapsed_seconds = float(resource_usage["elapsed_seconds"])
                returncode = int(resource_usage["returncode"])
            else:
                start = time.monotonic()
                result = subprocess.run(
                    cmd, env=env, stdout=log,
                    stderr=subprocess.STDOUT, text=True)
                elapsed_seconds = time.monotonic() - start
                returncode = result.returncode
        record.update({
            "elapsed_seconds": elapsed_seconds,
            "elapsed_source": "monotonic_child_wall",
            "returncode": returncode,
            "finished_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        })
        if returncode != 0:
            record["status"] = "failed"
            update_manifest(manifest, record)
            raise RuntimeError(f"build failed for {case['name']}; see {staging / 'build.log'}")
        meta = validate_case(config, case, staging)
        final.parent.mkdir(parents=True, exist_ok=True)
        staging.rename(final)
        record.update({"status": "complete", "case_root": str(final),
                       "metadata": meta, "reused_existing": False})
        record.pop("staging_root", None)
        update_manifest(manifest, record)

    lock.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
