#!/usr/bin/env python3
"""Build and validate a resumable upper-threshold Special Block matrix.

Every newly built case uses the same T1, graph parameters, source UNG index,
and explicit environment.  A case becomes visible at its final path only after
its metadata and required files pass validation.
"""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import shlex
import shutil
import struct
import subprocess
import time
from pathlib import Path
from typing import Any


REQUIRED_INDEX_FILES = (
    "meta",
    "special_blocks.bin",
    "special_block_trie.bin",
    "special_edges.bin",
    "special_trie_regular_edges.bin",
)


def parse_meta(path: Path) -> dict[str, str]:
    result: dict[str, str] = {}
    for line in path.read_text().splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            result[key.strip()] = value.strip()
    return result


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


def clean_build_env(base: dict[str, str], config: dict[str, Any], upper: int) -> dict[str, str]:
    env = {key: value for key, value in base.items() if not key.startswith("UNG_")}
    env.update({str(key): str(value) for key, value in config.get("env", {}).items()})
    env["UNG_SPECIAL_BLOCK_MIN_POINTS"] = str(int(config["min_points"]))
    env["UNG_SPECIAL_BLOCK_UPPER_MIN_POINTS"] = str(upper)
    return env


def validate_source(config: dict[str, Any]) -> None:
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


def validate_case(config: dict[str, Any], case: dict[str, Any], case_root: Path) -> dict[str, str]:
    block_dir = case_root / "block_index"
    missing = [str(block_dir / name) for name in REQUIRED_INDEX_FILES
               if not (block_dir / name).is_file() or (block_dir / name).stat().st_size == 0]
    if missing:
        raise ValueError("missing or empty block-index files:\n" + "\n".join(missing))
    timing = case_root / "results" / "build_time.csv"
    if not timing.is_file() or timing.stat().st_size == 0:
        raise ValueError(f"missing build timing: {timing}")
    meta = parse_meta(block_dir / "meta")
    expected = {
        "index_format": "special_block_trie_multilevel_v1",
        "source_input": "ung_index",
        "source_ung_fingerprint": str(config["expected_source_fingerprint"]),
        "num_points": str(int(config["expected_num_points"])),
        "num_groups": str(int(config["expected_num_groups"])),
        "special_block_min_points": str(int(config["min_points"])),
        "special_block_upper_min_points": str(int(case["upper_min_points"])),
        "special_block_max_degree": str(int(config["max_degree"])),
        "special_block_num_cross_edges": str(int(config["num_cross_edges"])),
    }
    mismatches = {key: (meta.get(key), value) for key, value in expected.items()
                  if meta.get(key) != value}
    if mismatches:
        raise ValueError(f"metadata mismatch for {case['name']}: {mismatches}")
    if int(meta.get("special_block_upper_count", "0")) <= 0:
        raise ValueError(f"case {case['name']} produced no upper blocks")
    return meta


def build_command(config: dict[str, Any], case_root: Path) -> list[str]:
    return [
        str(config["build_app"]),
        "--ung_index_path_prefix", str(Path(config["main_index"])) + "/",
        "--base_bin_file", str(config["base_bin_file"]),
        "--base_label_file", str(config["base_label_file"]),
        "--block_index_path_prefix", str(case_root / "block_index") + "/",
        "--result_path_prefix", str(case_root / "results") + "/",
        "--data_type", "float",
        "--dist_fn", "L2",
        "--num_threads", str(int(config["num_threads"])),
        "--min_points", str(int(config["min_points"])),
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
    state["runs"] = sorted(runs, key=lambda item: int(item["upper_min_points"]))
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
    validate_source(config)
    manifest = output_root / "manifest.json"
    selected = set(args.case)

    for case in config["cases"]:
        if selected and case["name"] not in selected:
            continue
        upper = int(case["upper_min_points"])
        if upper <= int(config["min_points"]):
            raise ValueError(f"upper threshold must exceed T1: {upper}")
        final_root = Path(case.get("existing_path", output_root / case["name"]))
        try:
            meta = validate_case(config, case, final_root)
            if not args.force:
                print(f"[SKIP] {case['name']} validated at {final_root}", flush=True)
                update_manifest(manifest, {
                    "name": case["name"], "upper_min_points": upper,
                    "status": "complete", "case_root": str(final_root),
                    "reused_existing": True, "metadata": meta,
                })
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
        cmd = build_command(config, staging)
        env = clean_build_env(os.environ, config, upper)
        (staging / "command.txt").write_text(shlex.join(cmd) + "\n")
        (staging / "environment.json").write_text(json.dumps(
            {key: value for key, value in env.items() if key.startswith("UNG_")},
            indent=2, sort_keys=True) + "\n")
        record = {"name": case["name"], "upper_min_points": upper,
                  "status": "dry_run" if args.dry_run else "running",
                  "staging_root": str(staging),
                  "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z")}
        update_manifest(manifest, record)
        print(f"[BUILD] {case['name']} T1={config['min_points']} T2={upper}", flush=True)
        print(shlex.join(cmd), flush=True)
        if args.dry_run:
            continue
        start = time.monotonic()
        with (staging / "build.log").open("w") as log:
            result = subprocess.run(cmd, env=env, stdout=log, stderr=subprocess.STDOUT, text=True)
        record.update({"elapsed_seconds": time.monotonic() - start,
                       "returncode": result.returncode,
                       "finished_at": time.strftime("%Y-%m-%dT%H:%M:%S%z")})
        if result.returncode != 0:
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
