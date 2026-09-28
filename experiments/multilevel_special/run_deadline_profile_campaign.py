#!/usr/bin/env python3
"""Run detailed held-out profiles after query and build timing are quiescent."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import prepare_deadline_profile_campaign as prepare_profile
import run_deadline_build_campaign as bounded


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
PERFORMANCE_CAMPAIGN = HERE / "config.deadline_evidence_manifest.json"
PROFILE_MANIFEST = HERE / "config.deadline_profile_manifest.json"
PREPARE = HERE / "prepare_deadline_profile_campaign.py"
SEARCH_RUNNER = HERE / "run_selection_sweep.py"
VALIDATOR = HERE / "validate_selection_sweep.py"
SUMMARIZER = HERE / "summarize_selection_sweep.py"
QUERY_SUPERVISOR = (
    REPO / "runs/deadline_evidence_20260927_cpufix" /
    "deadline_supervisor_manifest.json"
)
BUILD_SUPERVISOR = (
    REPO / "runs/deadline_build_20260927" /
    "deadline_build_supervisor_manifest.json"
)
ACTIVE_PATTERN = (
    "run_deadline_evidence_campaign.py|run_deadline_build_campaign.py|"
    "run_selection_sweep.py|search_UNG_index|build_UNG_index|"
    "build_special_block_index"
)


def manifest_finished(path: Path) -> bool:
    if not path.is_file():
        return False
    state = json.loads(path.read_text(encoding="utf-8"))
    if not state.get("finished_at_utc"):
        return False
    # The query supervisor predates the explicit status field. Newer
    # supervisors publish it, and a terminal failure must not open the gate.
    return state.get("status", "complete") == "complete"


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


def active_experiment_processes() -> list[str]:
    result = subprocess.run(
        ["pgrep", "-af", ACTIVE_PATTERN], text=True,
        stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, check=False)
    return [
        line for line in result.stdout.splitlines()
        if "pgrep -af" not in line
        and "run_deadline_profile_campaign.py" not in line
    ]


def wait_for_idle(poll_seconds: float, timeout_seconds: float) -> None:
    start = time.monotonic()
    while True:
        active = active_experiment_processes()
        if not active:
            return
        elapsed = time.monotonic() - start
        if elapsed >= timeout_seconds:
            raise TimeoutError(
                "experiment processes remained active after upstream completion:\n" +
                "\n".join(active))
        print(
            f"[WAIT] {len(active)} experiment process(es) still exiting; "
            f"polling in {poll_seconds:.0f}s",
            flush=True,
        )
        time.sleep(min(poll_seconds, timeout_seconds - elapsed))


def append_record(path: Path, state: dict[str, Any],
                  record: dict[str, Any]) -> None:
    key = (record.get("stage"), record.get("dataset"),
           record.get("method"), record.get("workload"))
    state["runs"] = [row for row in state["runs"] if (
        row.get("stage"), row.get("dataset"), row.get("method"),
        row.get("workload")) != key]
    state["runs"].append(record)
    state["updated_at_utc"] = datetime.now(timezone.utc).isoformat()
    bounded.atomic_json(path, state)


def main() -> int:
    import run_authoritative_instrumented_profile as instrumented

    parser = argparse.ArgumentParser()
    parser.add_argument("--case-timeout-seconds", type=float, default=3300.0)
    parser.add_argument("--poll-seconds", type=float, default=30.0)
    parser.add_argument("--wait-timeout-seconds", type=float, default=86400.0)
    parser.add_argument(
        "--skip-build-gate", action="store_true",
        help=("Run after the frozen query campaign is complete and the host is "
              "idle, without requiring the independent construction campaign."),
    )
    parser.add_argument("--jobs", type=int, default=16)
    parser.add_argument("--build-dir", type=Path,
                        default=REPO / "build_ung_profile_instrumented_deadline")
    parser.add_argument("--reference-build-dir", type=Path,
                        default=REPO / "build_ung_rel")
    args = parser.parse_args()
    if min(args.case_timeout_seconds, args.poll_seconds,
           args.wait_timeout_seconds) <= 0 or args.jobs <= 0:
        parser.error("timeouts, poll interval, and jobs must be positive")

    wait_for_manifest(QUERY_SUPERVISOR, args.poll_seconds, args.wait_timeout_seconds)
    if not args.skip_build_gate:
        wait_for_manifest(BUILD_SUPERVISOR, args.poll_seconds, args.wait_timeout_seconds)
    wait_for_idle(args.poll_seconds, min(args.wait_timeout_seconds, 600.0))

    performance = json.loads(PERFORMANCE_CAMPAIGN.read_text(encoding="utf-8"))
    for dataset in performance["datasets"]:
        command = [sys.executable, str(VALIDATOR), str(dataset["search_config"])]
        status, returncode, _ = bounded.run_bounded(
            command, REPO / "runs/deadline_profile_20260927/preflight.log",
            min(args.case_timeout_seconds, 600.0))
        if status != "complete":
            raise RuntimeError(
                f"performance config failed validation: {dataset['search_config']} "
                f"(status={status}, returncode={returncode})")

    source_commit = instrumented.ensure_source_ready(
        REPO, instrumented.REQUIRED_GATE_TIMING_COMMIT)
    binary, build_commands = instrumented.build_instrumented_binary(
        REPO, args.build_dir.resolve(), args.reference_build_dir.resolve(), args.jobs)
    binary_hash = prepare_profile.sha256_file(binary)
    subprocess.run([
        sys.executable, str(PREPARE),
        "--campaign", str(PERFORMANCE_CAMPAIGN),
        "--search-binary", str(binary),
        "--source-commit", source_commit,
    ], cwd=HERE, check=True)

    campaign = json.loads(PROFILE_MANIFEST.read_text(encoding="utf-8"))
    run_root = Path(campaign["run_root"])
    supervisor = run_root / "deadline_profile_supervisor_manifest.json"
    log_path = run_root / "deadline_profile_campaign.log"
    state: dict[str, Any] = {
        "schema_version": 1,
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "case_timeout_seconds": args.case_timeout_seconds,
        "campaign_config": str(PROFILE_MANIFEST),
        "source_commit": source_commit,
        "search_binary_sha256": binary_hash,
        "build_commands": build_commands,
        "build_gate_skipped": args.skip_build_gate,
        "runs": [],
    }
    bounded.atomic_json(supervisor, state)

    for dataset in campaign["configs"]:
        config = Path(dataset["config"])
        for case in dataset["cases"]:
            command = [
                sys.executable, str(SEARCH_RUNNER), str(config),
                "--method", str(case["method"]),
                "--workload", str(case["workload"]),
            ]
            status, returncode, elapsed = bounded.run_bounded(
                command, log_path, args.case_timeout_seconds)
            append_record(supervisor, state, {
                "stage": "profile_query",
                "dataset": str(dataset["dataset"]),
                "method": str(case["method"]),
                "workload": str(case["workload"]),
                "lsearch": int(case["lsearch"]),
                "status": status,
                "returncode": returncode,
                "elapsed_seconds": elapsed,
            })
        for stage, command in (
            ("validate", [sys.executable, str(VALIDATOR), str(config)]),
            ("summarize", [
                sys.executable, str(SUMMARIZER), str(config),
                "--baseline", str(dataset["baseline_method"]),
                "--targets", "0.9",
            ]),
        ):
            status, returncode, elapsed = bounded.run_bounded(
                command, log_path, min(args.case_timeout_seconds, 600.0))
            append_record(supervisor, state, {
                "stage": stage,
                "dataset": str(dataset["dataset"]),
                "method": "",
                "workload": "",
                "status": status,
                "returncode": returncode,
                "elapsed_seconds": elapsed,
            })

    state["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
    state["status"] = (
        "complete" if all(row["status"] == "complete" for row in state["runs"])
        else "complete_with_failures"
    )
    bounded.atomic_json(supervisor, state)
    print(supervisor)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
