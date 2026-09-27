#!/usr/bin/env python3
"""Run the deadline evidence subset with a hard wall-time cap per case."""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve().parent
PREPARE = HERE / "prepare_deadline_evidence_campaign.py"
BUILD_RUNNER = HERE / "run_build_sweep.py"
SEARCH_RUNNER = HERE / "run_selection_sweep.py"
SUMMARIZER = HERE / "summarize_selection_sweep.py"
CAMPAIGN_CONFIG = HERE / "config.deadline_evidence_manifest.json"


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", dir=path.parent, text=True)
    temporary_path = Path(temporary)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2)
            stream.write("\n")
        os.replace(temporary_path, path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


def run_bounded(
    command: list[str], log_path: Path, timeout_seconds: float,
) -> tuple[str, int, float]:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    start = time.monotonic()
    with log_path.open("a", encoding="utf-8") as log:
        log.write("+ " + subprocess.list2cmdline(command) + "\n")
        log.flush()
        process = subprocess.Popen(
            command, cwd=HERE, stdout=log, stderr=subprocess.STDOUT,
            text=True, start_new_session=True)
        try:
            returncode = process.wait(timeout=timeout_seconds)
            status = "complete" if returncode == 0 else "failed"
        except subprocess.TimeoutExpired:
            status, returncode = "timeout", 124
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
            log.write(
                f"[DEADLINE TIMEOUT] exceeded {timeout_seconds:.0f} seconds\n")
    return status, returncode, time.monotonic() - start


def append_record(path: Path, state: dict[str, Any], record: dict[str, Any]) -> None:
    state["runs"].append(record)
    state["updated_at_utc"] = datetime.now(timezone.utc).isoformat()
    atomic_json(path, state)


def completed_builds(config_path: Path) -> set[str]:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    path = Path(config["output_root"]) / "manifest.json"
    if not path.is_file():
        return set()
    rows = json.loads(path.read_text(encoding="utf-8")).get("runs", [])
    return {str(row["name"]) for row in rows if row.get("status") == "complete"}


def structure_name(method: dict[str, Any]) -> str:
    return str(method["name"]).split("_entry_", 1)[0]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case-timeout-seconds", type=float, default=3300.0)
    parser.add_argument("--skip-prepare", action="store_true")
    args = parser.parse_args()
    if args.case_timeout_seconds <= 0:
        parser.error("--case-timeout-seconds must be positive")
    if not args.skip_prepare:
        subprocess.run([sys.executable, str(PREPARE)], cwd=HERE, check=True)
    campaign = json.loads(CAMPAIGN_CONFIG.read_text(encoding="utf-8"))
    run_root = Path(json.loads(Path(
        campaign["datasets"][0]["search_config"]).read_text())["output_root"])
    run_root = run_root.parents[2]
    supervisor_path = run_root / "deadline_supervisor_manifest.json"
    log_path = run_root / "deadline_campaign.log"
    state: dict[str, Any] = {
        "schema_version": 1,
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "case_timeout_seconds": args.case_timeout_seconds,
        "campaign_config": str(CAMPAIGN_CONFIG),
        "runs": [],
    }
    atomic_json(supervisor_path, state)

    datasets = []
    for row in campaign["datasets"]:
        build_path = Path(row["build_config"])
        search_path = Path(row["search_config"])
        build = json.loads(build_path.read_text(encoding="utf-8"))
        search = json.loads(search_path.read_text(encoding="utf-8"))
        datasets.append((row, build_path, search_path, build, search))

    # Establish the cross-dataset DRH comparison before spending time on the
    # manual structural oracle.
    for row, build_path, _, build, _ in datasets:
        automatic_structure = str(row["automatic_method"]).split("_entry_", 1)[0]
        command = [sys.executable, str(BUILD_RUNNER), str(build_path),
                   "--case", automatic_structure]
        status, returncode, elapsed = run_bounded(
            command, log_path, args.case_timeout_seconds)
        append_record(supervisor_path, state, {
            "stage": "automatic_build", "dataset": row["dataset"],
            "case": automatic_structure, "status": status,
            "returncode": returncode, "elapsed_seconds": elapsed,
        })

    for row, build_path, search_path, _, search in datasets:
        available = completed_builds(build_path)
        priority = [row["baseline_method"], row["automatic_method"]]
        for method_name in priority:
            method = next(item for item in search["methods"]
                          if item["name"] == method_name)
            if method.get("hierarchy_layers") and structure_name(method) not in available:
                append_record(supervisor_path, state, {
                    "stage": "priority_query", "dataset": row["dataset"],
                    "method": method_name, "status": "skipped_missing_build",
                })
                continue
            for workload in row["workloads"]:
                command = [sys.executable, str(SEARCH_RUNNER), str(search_path),
                           "--method", method_name, "--workload", workload]
                status, returncode, elapsed = run_bounded(
                    command, log_path, args.case_timeout_seconds)
                append_record(supervisor_path, state, {
                    "stage": "priority_query", "dataset": row["dataset"],
                    "method": method_name, "workload": workload,
                    "status": status, "returncode": returncode,
                    "elapsed_seconds": elapsed,
                })

    for row, build_path, _, build, _ in datasets:
        automatic_structure = str(row["automatic_method"]).split("_entry_", 1)[0]
        for case in build["cases"]:
            name = str(case["name"])
            if name == automatic_structure:
                continue
            command = [sys.executable, str(BUILD_RUNNER), str(build_path),
                       "--case", name]
            status, returncode, elapsed = run_bounded(
                command, log_path, args.case_timeout_seconds)
            append_record(supervisor_path, state, {
                "stage": "manual_build", "dataset": row["dataset"],
                "case": name, "status": status,
                "returncode": returncode, "elapsed_seconds": elapsed,
            })

    for row, build_path, search_path, _, search in datasets:
        available = completed_builds(build_path)
        for method_name in row["manual_methods"]:
            method = next(item for item in search["methods"]
                          if item["name"] == method_name)
            if structure_name(method) not in available:
                append_record(supervisor_path, state, {
                    "stage": "manual_query", "dataset": row["dataset"],
                    "method": method_name, "status": "skipped_missing_build",
                })
                continue
            for workload in row["workloads"]:
                command = [sys.executable, str(SEARCH_RUNNER), str(search_path),
                           "--method", method_name, "--workload", workload]
                status, returncode, elapsed = run_bounded(
                    command, log_path, args.case_timeout_seconds)
                append_record(supervisor_path, state, {
                    "stage": "manual_query", "dataset": row["dataset"],
                    "method": method_name, "workload": workload,
                    "status": status, "returncode": returncode,
                    "elapsed_seconds": elapsed,
                })
        run_bounded(
            [sys.executable, str(SUMMARIZER), str(search_path),
             "--baseline", row["baseline_method"], "--targets", "0.9"],
            log_path, min(args.case_timeout_seconds, 600.0))

    state["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
    atomic_json(supervisor_path, state)
    print(supervisor_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
