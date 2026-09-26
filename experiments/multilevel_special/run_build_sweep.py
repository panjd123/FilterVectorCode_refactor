#!/usr/bin/env python3
"""Build and validate a resumable upper-threshold Special Block matrix.

Every newly built case uses the same T1, graph parameters, source UNG index,
and explicit environment.  A case becomes visible at its final path only after
its metadata and required files pass validation.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import re
import shlex
import shutil
import struct
import subprocess
import time
from pathlib import Path
from typing import Any, Optional

import experiment_core
import process_resource_probe


REQUIRED_INDEX_FILES = (
    "meta",
    "special_blocks.bin",
    "special_block_trie.bin",
    "special_edges.bin",
)


def parse_meta(path: Path) -> dict[str, str]:
    result: dict[str, str] = {}
    for line in path.read_text().splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            result[key.strip()] = value.strip()
    return result


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def snapshot_build_app(build_app: Path, output_root: Path) -> tuple[Path, str]:
    """Pin one immutable builder executable for the whole build sweep."""
    digest = sha256_file(build_app)
    snapshot_dir = output_root / ".binary_snapshots"
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    snapshot = snapshot_dir / f"build_special_block_index.{digest}"
    if not snapshot.exists():
        temporary = snapshot.with_suffix(".tmp")
        shutil.copy2(build_app, temporary)
        if sha256_file(temporary) != digest:
            temporary.unlink(missing_ok=True)
            raise RuntimeError("build binary changed while it was being snapshotted")
        temporary.chmod(0o555)
        temporary.replace(snapshot)
    elif sha256_file(snapshot) != digest:
        raise RuntimeError(f"corrupt build binary snapshot: {snapshot}")
    return snapshot, digest


def validate_reordered_labels(base_labels: Path, index_dir: Path) -> dict[str, str]:
    """Prove that index labels equal base labels under new-to-old IDs.

    UNG stores points in group-contiguous order, so byte-identical label files
    are sufficient but not necessary. The persisted new_to_old_vec_ids mapping
    defines the semantic comparison required for an independently built
    overlay to use the source graph safely.
    """
    index_labels = index_dir / "labels.txt"
    mapping = index_dir / "new_to_old_vec_ids"
    if not index_labels.is_file() or not mapping.is_file():
        raise FileNotFoundError(
            f"source index is missing label provenance files: {index_labels}, {mapping}"
        )
    base_hash = sha256_file(base_labels)
    index_hash = sha256_file(index_labels)
    if index_hash == base_hash:
        return {"base_labels_sha256": base_hash,
                "index_labels_sha256": index_hash,
                "label_alignment": "identity"}

    with base_labels.open() as stream:
        base_rows = [line.strip() for line in stream]
    with index_labels.open() as stream:
        index_rows = [line.strip() for line in stream]
    with mapping.open() as stream:
        mapping_rows = [int(line.strip()) for line in stream if line.strip()]
    if len(index_rows) != len(base_rows) or len(mapping_rows) != len(base_rows):
        raise ValueError(
            "source index label/mapping row count does not match the requested x1 base labels: "
            f"base={len(base_rows)}, index={len(index_rows)}, mapping={len(mapping_rows)}"
        )
    seen = bytearray(len(base_rows))
    for new_id, old_id in enumerate(mapping_rows):
        if old_id < 0 or old_id >= len(base_rows) or seen[old_id]:
            raise ValueError(f"invalid new-to-old permutation at new id {new_id}: {old_id}")
        seen[old_id] = 1
        if index_rows[new_id] != base_rows[old_id]:
            raise ValueError(
                "source index labels do not match x1 base labels under new-to-old mapping: "
                f"new_id={new_id}, old_id={old_id}"
            )
    return {"base_labels_sha256": base_hash,
            "index_labels_sha256": index_hash,
            "label_alignment": "new_to_old_permutation"}


def acquire_lock(output_root: Path):
    lock_path = output_root / ".builder.lock"
    lock_file = lock_path.open("a+")
    try:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as error:
        lock_file.seek(0)
        owner = lock_file.read().strip() or "unknown"
        lock_file.close()
        raise RuntimeError(f"another build sweep holds {lock_path} (pid={owner})") from error
    lock_file.seek(0)
    lock_file.truncate()
    lock_file.write(str(os.getpid()) + "\n")
    lock_file.flush()
    return lock_file


def case_min_points(config: dict[str, Any], case: dict[str, Any]) -> int:
    layers = experiment_core.hierarchy_layers(case)
    if layers:
        return int(layers[0]["min_points"])
    return int(case.get("min_points", config["min_points"]))


def clean_build_env(base: dict[str, str], config: dict[str, Any], case: dict[str, Any],
                    upper: Optional[int]) -> dict[str, str]:
    env = {key: value for key, value in base.items() if not key.startswith("UNG_")}
    env.update({str(key): str(value) for key, value in config.get("env", {}).items()})
    env.update({str(key): str(value) for key, value in case.get("env", {}).items()})
    layers = experiment_core.hierarchy_layers(case)
    if layers:
        env["UNG_SPECIAL_BLOCKS"] = "1"
        env["UNG_BASE_GROUP_TOPOLOGY"] = str(
            case.get("base_topology", config.get("base_topology", "lng")))
        env["UNG_HIERARCHY_LAYERS"] = experiment_core.encode_hierarchy_layers(case)
    else:
        env["UNG_SPECIAL_BLOCK_MIN_POINTS"] = str(case_min_points(config, case))
        if upper is not None:
            env["UNG_SPECIAL_BLOCK_UPPER_MIN_POINTS"] = str(upper)
    return env


def validate_source(config: dict[str, Any]) -> dict[str, str]:
    required = [Path(config["build_app"]), Path(config["main_index"]) / "meta",
                Path(config["base_bin_file"]), Path(config["base_label_file"])]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("missing build inputs:\n" + "\n".join(missing))
    with Path(config["base_bin_file"]).open("rb") as stream:
        header = stream.read(8)
    if len(header) != 8:
        raise ValueError("invalid base vector header")
    num_points, dimension = struct.unpack("<II", header)
    if num_points != int(config["expected_num_points"]) or dimension <= 0:
        raise ValueError(f"base data is not the required Amazon x1 input: {num_points}x{dimension}")
    base_labels = Path(config["base_label_file"])
    provenance = validate_reordered_labels(base_labels, Path(config["main_index"]))
    base_hash = provenance["base_labels_sha256"]
    expected_hash = config.get("expected_base_labels_sha256")
    if expected_hash and base_hash != expected_hash:
        raise ValueError(f"base-label hash mismatch: {base_hash} != {expected_hash}")
    provenance["build_binary_sha256"] = sha256_file(Path(config["build_app"]))
    return provenance


def validate_case(config: dict[str, Any], case: dict[str, Any], case_root: Path) -> dict[str, str]:
    block_dir = case_root / "block_index"
    missing = [str(block_dir / name) for name in REQUIRED_INDEX_FILES
               if not (block_dir / name).is_file() or (block_dir / name).stat().st_size == 0]
    if missing:
        raise ValueError("missing or empty block-index files:\n" + "\n".join(missing))
    timing = case_root / "results" / "build_time.csv"
    if not timing.is_file() or timing.stat().st_size == 0:
        raise ValueError(f"missing build timing: {timing}")
    if config.get("resource_profile", False):
        resource_path = case_root / "resource_usage.json"
        if not resource_path.is_file() or resource_path.stat().st_size == 0:
            raise ValueError(f"missing resource profile: {resource_path}")
        resource = json.loads(resource_path.read_text())
        if int(resource.get("num_samples", 0)) <= 0:
            raise ValueError(f"empty resource profile: {resource_path}")
    meta = parse_meta(block_dir / "meta")
    upper_value = case.get("upper_min_points")
    upper = int(upper_value) if upper_value is not None else None
    layers = experiment_core.hierarchy_layers(case)
    is_multilevel = len(layers) > 1 or upper is not None
    expected = {
        "index_format": "special_block_trie_multilevel_v1" if is_multilevel
                        else "special_block_trie_v2",
        "source_input": "ung_index",
        "source_ung_fingerprint": str(config["expected_source_fingerprint"]),
        "num_points": str(int(config["expected_num_points"])),
        "num_groups": str(int(config["expected_num_groups"])),
        "special_block_min_points": str(case_min_points(config, case)),
        "special_block_max_degree": str(int(config["max_degree"])),
        "special_block_num_cross_edges": str(int(config["num_cross_edges"])),
    }
    if layers:
        expected["hierarchy_layers"] = experiment_core.encode_hierarchy_layers(case)
        expected["base_group_topology"] = str(
            case.get("base_topology", config.get("base_topology", "lng")))
    if upper is not None:
        expected["special_block_upper_min_points"] = str(upper)
    mismatches = {key: (meta.get(key), value) for key, value in expected.items()
                  if meta.get(key) != value}
    if mismatches:
        raise ValueError(f"metadata mismatch for {case['name']}: {mismatches}")
    if layers and len(layers) > 1 and int(meta.get("special_block_upper_count", "0")) <= 0:
        raise ValueError(f"case {case['name']} produced no blocks above level 1")
    if not layers and upper is not None and int(meta.get("special_block_upper_count", "0")) <= 0:
        raise ValueError(f"case {case['name']} produced no upper blocks")
    if not layers and upper is None and int(meta.get("special_block_upper_count", "0")) != 0:
        raise ValueError(f"single-level case {case['name']} unexpectedly produced upper blocks")
    validate_backend_evidence(case, case_root, meta)
    return meta


def validate_backend_evidence(case: dict[str, Any], case_root: Path,
                              meta: dict[str, str]) -> None:
    log_path = case_root / "build.log"
    if not log_path.is_file():
        raise ValueError(f"missing hierarchy backend evidence: {log_path}")
    text = log_path.read_text(errors="replace")
    intra_matches = re.findall(
        r"\[special_edges\] gpu_intra_enabled=(\d+) gpu_intra_blocks=(\d+) "
        r"gpu_intra_points=(\d+) gpu_intra_fallback_blocks=(\d+)", text)
    inter_matches = re.findall(
        r"\[special_edges\] gpu_inter_enabled=(\d+) gpu_inter_used=(\d+) "
        r"gpu_inter_ms=([0-9.eE+-]+)", text)
    if not intra_matches or not inter_matches:
        raise ValueError(f"incomplete hierarchy backend evidence: {log_path}")
    intra_enabled, intra_blocks, intra_points, intra_fallbacks = (
        int(value) for value in intra_matches[-1])
    inter_enabled = int(inter_matches[-1][0])
    inter_used = int(inter_matches[-1][1])
    profile = str(case["benchmark_profile"])
    requires_intra = profile != "cpu"
    requires_inter = profile in {
        "hybrid_gpu_intra_inter", "full_gpu", "full_gpu_wmma"}
    if profile == "cpu" and any((intra_enabled, intra_blocks, inter_enabled, inter_used)):
        raise ValueError(f"CPU hierarchy case used a GPU backend: {case['name']}")
    if requires_intra and (intra_blocks <= 0 or intra_points <= 0):
        raise ValueError(f"GPU intra-block work was not observed for {case['name']}")
    if requires_intra and intra_fallbacks != 0:
        raise ValueError(f"GPU intra-block fallback was observed for {case['name']}")
    if requires_inter and (inter_enabled != 1 or inter_used != 1):
        raise ValueError(f"GPU inter-block work was not observed for {case['name']}")
    if not requires_inter and inter_used != 0:
        raise ValueError(f"unexpected GPU inter-block work for {case['name']}")
    if profile == "full_gpu_wmma" and "mode=tf32_wmma" not in text:
        raise ValueError(f"WMMA inter-block execution was not observed for {case['name']}")
    meta.update({
        "verified_gpu_intra_blocks": str(intra_blocks),
        "verified_gpu_intra_points": str(intra_points),
        "verified_gpu_intra_fallback_blocks": str(intra_fallbacks),
        "verified_gpu_inter_used": str(inter_used),
        "verified_wmma_inter": str(int("mode=tf32_wmma" in text)),
    })


def build_command(config: dict[str, Any], case: dict[str, Any], case_root: Path,
                  build_app: Optional[Path] = None) -> list[str]:
    return [
        str(build_app or config["build_app"]),
        "--ung_index_path_prefix", str(Path(config["main_index"])) + "/",
        "--base_bin_file", str(config["base_bin_file"]),
        "--base_label_file", str(config["base_label_file"]),
        "--block_index_path_prefix", str(case_root / "block_index") + "/",
        "--result_path_prefix", str(case_root / "results") + "/",
        "--data_type", "float",
        "--dist_fn", "L2",
        "--num_threads", str(int(config["num_threads"])),
        "--min_points", str(case_min_points(config, case)),
        "--max_degree", str(int(config["max_degree"])),
        "--num_cross_edges", str(int(config["num_cross_edges"])),
        "--Lbuild", str(int(config["Lbuild"])),
        "--alpha", str(float(config["alpha"])),
    ]


def update_manifest(path: Path, record: dict[str, Any]) -> None:
    state: dict[str, Any] = {"schema_version": 1, "runs": []}
    if path.exists():
        state = json.loads(path.read_text())
    runs = [item for item in state.get("runs", []) if item.get("name") != record["name"]]
    runs.append(record)
    state["runs"] = sorted(
        runs, key=lambda item: int(item["upper_min_points"])
        if item.get("upper_min_points") is not None else -1)
    state["updated_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(state, indent=2) + "\n")
    temporary.replace(path)


def quarantine(path: Path, output_root: Path, reason: str) -> None:
    if not path.exists():
        return
    stamp = time.strftime("%Y%m%dT%H%M%S")
    destination = output_root / "quarantine" / f"{path.name}_{stamp}_{reason}"
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(path), str(destination))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=Path)
    parser.add_argument("--case", action="append", default=[])
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    output_root = Path(config["output_root"])
    output_root.mkdir(parents=True, exist_ok=True)
    lock = acquire_lock(output_root)
    source_provenance = validate_source(config)
    build_app, build_binary_sha256 = snapshot_build_app(Path(config["build_app"]), output_root)
    source_provenance["build_binary_sha256"] = build_binary_sha256
    manifest = output_root / "manifest.json"
    selected = set(args.case)

    for case in config["cases"]:
        if selected and case["name"] not in selected:
            continue
        layers = experiment_core.hierarchy_layers(case)
        upper_value = case.get("upper_min_points") if not layers else None
        upper = int(upper_value) if upper_value is not None else None
        minimum = case_min_points(config, case)
        if upper is not None and upper <= minimum:
            raise ValueError(f"upper threshold must exceed T1: {upper}")
        final_root = Path(case.get("existing_path", output_root / case["name"]))
        try:
            meta = validate_case(config, case, final_root)
            if not args.force:
                print(f"[SKIP] {case['name']} validated at {final_root}", flush=True)
                reused_record = {
                    "name": case["name"], "min_points": minimum,
                    "upper_min_points": upper,
                    "status": "complete", "case_root": str(final_root),
                    "reused_existing": True, "metadata": meta,
                    "source_provenance": source_provenance,
                }
                experiment_core.retain_elapsed_evidence(
                    reused_record,
                    experiment_core.existing_manifest_record(
                        manifest, {"name": case["name"]}),
                    final_root, "command.txt", "build.log")
                update_manifest(manifest, reused_record)
                continue
        except (FileNotFoundError, ValueError):
            if final_root.exists() and not case.get("existing_path"):
                quarantine(final_root, output_root, "invalid")
        if case.get("existing_path"):
            raise RuntimeError(f"existing case failed validation and will not be overwritten: {final_root}")
        if final_root.exists():
            quarantine(final_root, output_root, "forced")

        staging = output_root / ".staging" / f"{case['name']}_{os.getpid()}"
        if staging.exists():
            quarantine(staging, output_root, "stale_staging")
        staging.mkdir(parents=True)
        cmd = build_command(config, case, staging, build_app)
        env = clean_build_env(os.environ, config, case, upper)
        (staging / "command.txt").write_text(shlex.join(cmd) + "\n")
        (staging / "environment.json").write_text(json.dumps(
            {key: value for key, value in env.items() if key.startswith("UNG_")},
            indent=2, sort_keys=True) + "\n")
        record = {"name": case["name"], "min_points": minimum,
                  "upper_min_points": upper,
                  "status": "dry_run" if args.dry_run else "running",
                  "staging_root": str(staging),
                  "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                  "source_provenance": source_provenance}
        update_manifest(manifest, record)
        level_text = (experiment_core.encode_hierarchy_layers(case)
                      if layers else (f"T2={upper}" if upper is not None else "single-level"))
        print(f"[BUILD] {case['name']} T1={minimum} {level_text}", flush=True)
        print(shlex.join(cmd), flush=True)
        if args.dry_run:
            continue
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
        record.update({"elapsed_seconds": elapsed_seconds,
                       "elapsed_source": "monotonic_child_wall",
                       "returncode": returncode,
                       "finished_at": time.strftime("%Y-%m-%dT%H:%M:%S%z")})
        if returncode != 0:
            record["status"] = "failed"
            update_manifest(manifest, record)
            raise RuntimeError(f"build failed for {case['name']}; see {staging / 'build.log'}")
        meta = validate_case(config, case, staging)
        final_root.parent.mkdir(parents=True, exist_ok=True)
        staging.rename(final_root)
        record.update({"status": "complete", "case_root": str(final_root),
                       "reused_existing": False, "metadata": meta})
        record.pop("staging_root", None)
        update_manifest(manifest, record)

    lock.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
