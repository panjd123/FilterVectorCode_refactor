#!/usr/bin/env python3
"""Run the bounded Amazon construction campaign one isolated case at a time."""

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
PREPARE = HERE / "prepare_deadline_build_campaign.py"
CAMPAIGN_CONFIG = HERE / "config.deadline_amazon_build_manifest.json"
DEFAULT_QUERY_SUPERVISOR = (
    HERE.parents[1] / "runs/deadline_evidence_20260927_cpufix" /
    "deadline_supervisor_manifest.json"
)
PHASES = ("base_timing", "hierarchy_timing", "base_resource",
          "hierarchy_resource")


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
        temporary_path.unlink(missing_ok=True)


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


def require_finished_query_campaign(path: Path) -> None:
    if not path.is_file():
        raise RuntimeError(f"query campaign manifest is missing: {path}")
    state = json.loads(path.read_text(encoding="utf-8"))
    if not state.get("finished_at_utc"):
        raise RuntimeError(
            "query campaign is still active or incomplete; construction timing "
            "must not overlap the 100-thread query campaign")


def load_or_initialize(path: Path, campaign_path: Path,
                       timeout_seconds: float) -> dict[str, Any]:
    if path.is_file():
        state = json.loads(path.read_text(encoding="utf-8"))
        if Path(state["campaign_config"]).resolve() != campaign_path.resolve():
            raise RuntimeError("existing supervisor manifest belongs to another campaign")
        return state
    return {
        "schema_version": 1,
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "case_timeout_seconds": timeout_seconds,
        "campaign_config": str(campaign_path.resolve()),
        "runs": [],
    }


def case_key(record: dict[str, Any]) -> tuple[str, str]:
    return str(record["phase"]), str(record["case"])


def update_record(path: Path, state: dict[str, Any],
                  record: dict[str, Any]) -> None:
    key = case_key(record)
    prior = [row for row in state["runs"] if case_key(row) == key]
    record["attempt"] = 1 + max(
        (int(row.get("attempt", 1)) for row in prior), default=0)
    state["runs"] = [row for row in state["runs"] if case_key(row) != key]
    state["runs"].append(record)
    state["updated_at_utc"] = datetime.now(timezone.utc).isoformat()
    state.pop("finished_at_utc", None)
    atomic_json(path, state)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case-timeout-seconds", type=float, default=3300.0)
    parser.add_argument("--start-at", choices=PHASES, default=PHASES[0])
    parser.add_argument("--stop-after", choices=PHASES, default=PHASES[-1])
    parser.add_argument("--skip-prepare", action="store_true")
    parser.add_argument("--skip-query-gate", action="store_true")
    parser.add_argument("--retry-failed", action="store_true")
    parser.add_argument("--query-supervisor", type=Path,
                        default=DEFAULT_QUERY_SUPERVISOR)
    args = parser.parse_args()
    if args.case_timeout_seconds <= 0:
        parser.error("--case-timeout-seconds must be positive")
    start = PHASES.index(args.start_at)
    stop = PHASES.index(args.stop_after)
    if stop < start:
        parser.error("--stop-after must not precede --start-at")
    if not args.skip_prepare:
        subprocess.run([sys.executable, str(PREPARE)], cwd=HERE, check=True)
    if not args.skip_query_gate:
        require_finished_query_campaign(args.query_supervisor.resolve())

    campaign_path = CAMPAIGN_CONFIG.resolve()
    campaign = json.loads(campaign_path.read_text(encoding="utf-8"))
    run_root = Path(campaign["run_root"])
    supervisor_path = run_root / "deadline_build_supervisor_manifest.json"
    log_path = run_root / "deadline_build_campaign.log"
    state = load_or_initialize(
        supervisor_path, campaign_path, args.case_timeout_seconds)
    existing = {case_key(row): str(row.get("status")) for row in state["runs"]}

    stages = {str(row["phase"]): row for row in campaign["stages"]}
    for phase in PHASES[start:stop + 1]:
        stage = stages[phase]
        for case in stage["cases"]:
            key = (phase, str(case))
            status = existing.get(key)
            if status == "complete" or (
                    status in {"failed", "timeout"} and not args.retry_failed):
                print(f"[SKIP] {phase}/{case}: prior status={status}", flush=True)
                continue
            command = [sys.executable, str(stage["runner"]),
                       str(stage["config"]), "--case", str(case)]
            status, returncode, elapsed = run_bounded(
                command, log_path, args.case_timeout_seconds)
            record = {
                "phase": phase,
                "case": str(case),
                "status": status,
                "returncode": returncode,
                "elapsed_seconds": elapsed,
            }
            update_record(supervisor_path, state, record)
            existing[key] = status

    state["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
    terminal_counts: dict[str, int] = {}
    for row in state["runs"]:
        status = str(row.get("status", "unknown"))
        terminal_counts[status] = terminal_counts.get(status, 0) + 1
    state["terminal_counts"] = terminal_counts
    state["status"] = (
        "complete" if set(terminal_counts) <= {"complete"}
        else "complete_with_failures"
    )
    atomic_json(supervisor_path, state)
    print(supervisor_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
