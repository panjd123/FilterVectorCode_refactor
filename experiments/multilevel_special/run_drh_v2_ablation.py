#!/usr/bin/env python3
"""Run DRH-v2 only after the frozen deadline campaigns have quiesced."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import prepare_drh_v2_ablation as prepare
import run_authoritative_instrumented_profile as build_helper
import run_deadline_build_campaign as bounded


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
PROFILE_SUPERVISOR = (
    REPO / "runs/deadline_profile_20260927" /
    "deadline_profile_supervisor_manifest.json"
)
MANIFEST = HERE / "config.drh_v2_manifest.json"
SEARCH_RUNNER = HERE / "run_selection_sweep.py"
VALIDATOR = HERE / "validate_selection_sweep.py"
SUMMARIZER = HERE / "summarize_selection_sweep.py"
CAMPAIGN_SUMMARIZER = HERE / "summarize_drh_v2_ablation.py"
DATASET_PRIORITY = {"VariousImg": 0, "Amazon": 1, "Genome": 2, "Reviews": 3}


def manifest_finished(path: Path) -> bool:
    if not path.is_file():
        return False
    return bool(json.loads(path.read_text(encoding="utf-8")).get("finished_at_utc"))


def wait_for_manifest(path: Path, poll_seconds: float,
                      timeout_seconds: float) -> None:
    start = time.monotonic()
    while not manifest_finished(path):
        elapsed = time.monotonic() - start
        if elapsed >= timeout_seconds:
            raise TimeoutError(
                f"completion marker did not appear within {timeout_seconds:.0f}s: {path}")
        print(
            f"[WAIT] {path.name} incomplete after {elapsed:.0f}s; "
            f"polling in {poll_seconds:.0f}s",
            flush=True,
        )
        time.sleep(min(poll_seconds, timeout_seconds - elapsed))


def append_record(path: Path, state: dict[str, Any], record: dict[str, Any]) -> None:
    state["runs"].append(record)
    state["updated_at_utc"] = datetime.now(timezone.utc).isoformat()
    bounded.atomic_json(path, state)


def run_checked(command: list[str], cwd: Path) -> None:
    print("+ " + subprocess.list2cmdline(command), flush=True)
    subprocess.run(command, cwd=cwd, check=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case-timeout-seconds", type=float, default=3300.0)
    parser.add_argument("--poll-seconds", type=float, default=30.0)
    parser.add_argument("--wait-timeout-seconds", type=float, default=172800.0)
    parser.add_argument("--jobs", type=int, default=16)
    parser.add_argument("--build-dir", type=Path,
                        default=REPO / "build_ung_drh_v2_release")
    parser.add_argument("--reference-build-dir", type=Path,
                        default=REPO / "build_ung_rel")
    args = parser.parse_args()
    if min(args.case_timeout_seconds, args.poll_seconds,
           args.wait_timeout_seconds) <= 0 or args.jobs <= 0:
        parser.error("timeouts, poll interval, and jobs must be positive")

    wait_for_manifest(PROFILE_SUPERVISOR, args.poll_seconds, args.wait_timeout_seconds)
    source_commit = build_helper.ensure_source_ready(
        REPO, build_helper.REQUIRED_GATE_TIMING_COMMIT)
    binary, build_commands = build_helper.build_instrumented_binary(
        REPO, args.build_dir.resolve(), args.reference_build_dir.resolve(), args.jobs)

    test_build = [
        "cmake", "--build", str(args.build_dir.resolve()), "--target",
        "test_special_block_free_state", "-j", str(args.jobs),
    ]
    run_checked(test_build, REPO)
    test_binary = args.build_dir.resolve() / "test/test_special_block_free_state"
    if not test_binary.is_file() or not os.access(test_binary, os.X_OK):
        raise RuntimeError(f"missing C++ gate test binary: {test_binary}")
    run_checked([str(test_binary)], REPO)

    run_checked([
        sys.executable, str(HERE / "prepare_drh_v2_ablation.py"),
        "--search-binary", str(binary),
        "--source-commit", source_commit,
    ], HERE)
    campaign = json.loads(MANIFEST.read_text(encoding="utf-8"))
    run_root = Path(campaign["run_root"])
    supervisor = run_root / "drh_v2_supervisor_manifest.json"
    log_path = run_root / "drh_v2_campaign.log"
    state: dict[str, Any] = {
        "schema_version": 1,
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "case_timeout_seconds": args.case_timeout_seconds,
        "campaign_config": str(MANIFEST),
        "source_commit": source_commit,
        "search_binary": str(binary),
        "search_binary_sha256": campaign["search_binary_sha256"],
        "build_commands": build_commands + [test_build],
        "runs": [],
    }
    bounded.atomic_json(supervisor, state)

    datasets = sorted(
        campaign["configs"],
        key=lambda row: DATASET_PRIORITY.get(str(row["dataset"]), 99),
    )
    for dataset in datasets:
        config = Path(dataset["config"])
        status, returncode, elapsed = bounded.run_bounded(
            [sys.executable, str(VALIDATOR), str(config)], log_path,
            min(args.case_timeout_seconds, 600.0))
        append_record(supervisor, state, {
            "stage": "validate", "dataset": dataset["dataset"],
            "status": status, "returncode": returncode,
            "elapsed_seconds": elapsed,
        })
        if status != "complete":
            continue

        methods = [
            dataset["baseline_method"], dataset["drh_v1_method"],
            dataset["drh_v2_method"],
        ]
        for workload in dataset["workloads"]:
            for method in methods:
                status, returncode, elapsed = bounded.run_bounded(
                    [sys.executable, str(SEARCH_RUNNER), str(config),
                     "--method", str(method), "--workload", str(workload)],
                    log_path, args.case_timeout_seconds)
                append_record(supervisor, state, {
                    "stage": "query", "dataset": dataset["dataset"],
                    "method": method, "workload": workload,
                    "status": status, "returncode": returncode,
                    "elapsed_seconds": elapsed,
                })

        status, returncode, elapsed = bounded.run_bounded(
            [sys.executable, str(SUMMARIZER), str(config), "--baseline",
             str(dataset["baseline_method"]), "--targets", "0.9"],
            log_path, min(args.case_timeout_seconds, 600.0))
        append_record(supervisor, state, {
            "stage": "summarize", "dataset": dataset["dataset"],
            "status": status, "returncode": returncode,
            "elapsed_seconds": elapsed,
        })

    state["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
    state["status"] = (
        "complete" if all(row["status"] == "complete" for row in state["runs"])
        else "complete_with_failures"
    )
    bounded.atomic_json(supervisor, state)
    if state["status"] == "complete":
        run_checked([
            sys.executable, str(CAMPAIGN_SUMMARIZER),
            "--manifest", str(MANIFEST),
        ], HERE)
    print(supervisor)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
