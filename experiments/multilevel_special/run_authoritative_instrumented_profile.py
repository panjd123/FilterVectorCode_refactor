#!/usr/bin/env python3
"""Rebuild and rerun the Amazon profile with complete authorization timing.

The authoritative performance and held-out campaigns intentionally keep their
immutable search binary. This runner is used after those campaigns: it builds
the timing-only source correction in a separate directory, derives a fresh
profile config from the validated formal config, and verifies that the result
manifest used exactly the newly built binary.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

import advance_authoritative_campaign


HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
REQUIRED_GATE_TIMING_COMMIT = "7f723b6"
ACTIVE_PATTERN = (
    "run_authoritative_campaign.py|run_heldout_campaign.py|"
    "run_authoritative_build_campaign.py|run_selection_sweep.py|"
    "search_UNG_index|build_UNG_index|build_special_block_index"
)


def run(command: list[str], cwd: Path = HERE) -> None:
    print("+", subprocess.list2cmdline(command), flush=True)
    subprocess.run(command, cwd=cwd, check=True)


def capture(command: list[str], cwd: Path = HERE) -> str:
    return subprocess.run(
        command, cwd=cwd, check=True, text=True,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    ).stdout.strip()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def active_experiment_processes() -> list[str]:
    result = subprocess.run(
        ["pgrep", "-af", ACTIVE_PATTERN], text=True,
        stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, check=False)
    return [
        line for line in result.stdout.splitlines()
        if "pgrep -af" not in line
        and "run_authoritative_instrumented_profile.py" not in line
    ]


def ensure_source_ready(repo: Path, required_commit: str) -> str:
    ancestor = subprocess.run(
        ["git", "merge-base", "--is-ancestor", required_commit, "HEAD"],
        cwd=repo, check=False)
    if ancestor.returncode != 0:
        raise RuntimeError(
            f"required authorization-timing commit is absent: {required_commit}")
    tracked = capture(
        ["git", "status", "--porcelain", "--untracked-files=no", "--", "UNG/codes"],
        cwd=repo)
    if tracked:
        raise RuntimeError(
            "refusing to build from modified query sources:\n" + tracked)
    return capture(["git", "rev-parse", "HEAD"], cwd=repo)


def cmake_cache_value(path: Path, key: str) -> str:
    if not path.is_file():
        raise FileNotFoundError(path)
    prefix = key + ":"
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if line.startswith(prefix):
            _, value = line.split("=", 1)
            if value:
                return value
    raise ValueError(f"{path}: missing CMake cache key {key}")


def build_instrumented_binary(
    repo: Path, build_dir: Path, reference_build_dir: Path, jobs: int,
) -> tuple[Path, list[list[str]]]:
    cache = reference_build_dir / "CMakeCache.txt"
    configure = [
        "cmake", "-S", str(repo / "UNG/codes"), "-B", str(build_dir),
        "-DCMAKE_BUILD_TYPE=Release", "-DBUILD_TESTING=OFF",
        "-DCMAKE_C_COMPILER=" + cmake_cache_value(cache, "CMAKE_C_COMPILER"),
        "-DCMAKE_CXX_COMPILER=" + cmake_cache_value(cache, "CMAKE_CXX_COMPILER"),
        "-DCMAKE_CUDA_COMPILER=" + cmake_cache_value(cache, "CMAKE_CUDA_COMPILER"),
    ]
    build = [
        "cmake", "--build", str(build_dir), "--target", "search_UNG_index",
        "-j", str(jobs),
    ]
    run(configure, cwd=repo)
    run(build, cwd=repo)
    binary = build_dir / "apps/search_UNG_index"
    if not binary.is_file() or not os.access(binary, os.X_OK):
        raise RuntimeError(f"instrumented search binary is missing: {binary}")
    return binary, [configure, build]


def instrumented_output_root(formal_root: Path) -> Path:
    name = formal_root.name
    if name.endswith("_formal"):
        name = name[:-len("_formal")]
    return formal_root.with_name(name + "_profile_instrumented")


def make_profile_config(
    formal: dict[str, Any], binary: Path, source_commit: str,
    binary_hash: str,
) -> dict[str, Any]:
    profile = advance_authoritative_campaign.make_profile(copy.deepcopy(formal))
    profile["output_root"] = str(instrumented_output_root(
        Path(formal["output_root"])))
    profile["search_app"] = str(binary)
    profile["instrumentation_provenance"] = {
        "purpose": "complete exact-gate plus block-coverage authorization timing",
        "source_commit": source_commit,
        "required_gate_timing_commit": REQUIRED_GATE_TIMING_COMMIT,
        "search_binary_sha256": binary_hash,
        "performance_binary_is_intentionally_unchanged": True,
    }
    return profile


def write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", dir=path.parent, text=True)
    temporary_path = Path(temporary)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
            json.dump(payload, stream, indent=2)
            stream.write("\n")
        os.replace(temporary_path, path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


def profile_manifest(config: dict[str, Any]) -> Path:
    root = Path(config["output_root"])
    pass_name = str(config.get("measurement_pass", "profile"))
    return root / (f"manifest_{pass_name}.json"
                   if config.get("pass_subdirs", False) else "manifest.json")


def validate_profile_binary(config: dict[str, Any], expected_hash: str) -> None:
    path = profile_manifest(config)
    if not path.is_file():
        raise FileNotFoundError(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    expected_pairs = {
        (str(method["name"]), str(workload["name"]))
        for method in config["methods"]
        for workload in config["workloads"]
        if method.get("enabled_workloads") is None
        or workload["name"] in method["enabled_workloads"]
    }
    current = {
        (str(row.get("method")), str(row.get("workload"))): row
        for row in payload.get("runs", [])
        if (str(row.get("method")), str(row.get("workload"))) in expected_pairs
    }
    if set(current) != expected_pairs:
        raise RuntimeError("instrumented profile manifest is incomplete")
    hashes = {row.get("search_binary_sha256") for row in current.values()}
    if hashes != {expected_hash}:
        raise RuntimeError(
            f"instrumented profile binary mismatch: {sorted(str(x) for x in hashes)}")
    if any(row.get("status") != "complete" or int(row.get("returncode", 0)) != 0
           for row in current.values()):
        raise RuntimeError("instrumented profile manifest contains failed runs")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--formal-config", type=Path,
        default=HERE / "config.authoritative_amazon_formal_emptyfix.json")
    parser.add_argument(
        "--profile-config", type=Path,
        default=HERE / "config.authoritative_amazon_profile_instrumented.json")
    parser.add_argument("--build-dir", type=Path,
                        default=REPO / "build_ung_profile_instrumented")
    parser.add_argument("--reference-build-dir", type=Path,
                        default=REPO / "build_ung_rel")
    parser.add_argument("--jobs", type=int, default=16)
    args = parser.parse_args()
    if args.jobs <= 0:
        parser.error("--jobs must be positive")

    active = active_experiment_processes()
    if active:
        raise RuntimeError(
            "refusing to perturb an active experiment:\n" + "\n".join(active))
    formal_path = args.formal_config.resolve()
    if not formal_path.is_file():
        raise FileNotFoundError(formal_path)
    run([sys.executable, str(HERE / "validate_selection_sweep.py"),
         str(formal_path)])
    source_commit = ensure_source_ready(REPO, REQUIRED_GATE_TIMING_COMMIT)
    binary, build_commands = build_instrumented_binary(
        REPO, args.build_dir.resolve(), args.reference_build_dir.resolve(),
        args.jobs)
    binary_hash = sha256(binary)
    formal = json.loads(formal_path.read_text(encoding="utf-8"))
    profile = make_profile_config(
        formal, binary.resolve(), source_commit, binary_hash)
    profile["instrumentation_provenance"]["build_commands"] = build_commands
    profile_path = args.profile_config.resolve()
    write_json_atomic(profile_path, profile)

    run([sys.executable, str(HERE / "run_selection_sweep.py"),
         str(profile_path)])
    run([sys.executable, str(HERE / "validate_selection_sweep.py"),
         str(profile_path)])
    run([
        sys.executable, str(HERE / "summarize_selection_sweep.py"),
        str(profile_path), "--baseline", "l0_lng_entry_optimized_lng",
        "--targets", "0.9",
    ])
    validate_profile_binary(profile, binary_hash)
    print(profile["output_root"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
