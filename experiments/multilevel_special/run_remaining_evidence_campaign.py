#!/usr/bin/env python3
"""Finish bounded topology and detailed-profile evidence after build timing."""

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

import run_deadline_build_campaign as bounded


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
SEARCH_RUNNER = HERE / "run_selection_sweep.py"
VALIDATOR = HERE / "validate_selection_sweep.py"
SUMMARIZER = HERE / "summarize_selection_sweep.py"
PLOTTER = HERE / "plot_authoritative_recall_qps.py"
TOPOLOGY_METHODS = (
    "l2_t1024_16384_ll_entry_optimized_lng",
    "l2_t1024_16384_tl_entry_optimized_lng",
    "l2_t1024_16384_tt_entry_optimized_lng",
)


def active_build_processes() -> list[str]:
    result = subprocess.run(
        ["pgrep", "-af", "run_deadline_build_campaign.py|run_build_sweep.py|build_special_block_index"],
        text=True, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, check=False)
    return [
        line for line in result.stdout.splitlines()
        if str(os.getpid()) not in line
        and "run_remaining_evidence_campaign.py" not in line
        and "pgrep -af" not in line
    ]


def wait_for_builds(poll_seconds: float, timeout_seconds: float) -> None:
    start = time.monotonic()
    while True:
        active = active_build_processes()
        if not active:
            return
        elapsed = time.monotonic() - start
        if elapsed >= timeout_seconds:
            raise TimeoutError(
                f"build processes did not quiesce within {timeout_seconds:.0f}s")
        print(
            f"[WAIT] {len(active)} build processes active after {elapsed:.0f}s",
            flush=True,
        )
        time.sleep(min(poll_seconds, timeout_seconds - elapsed))


def load_state(path: Path, case_timeout_seconds: float) -> dict[str, Any]:
    if path.is_file():
        return json.loads(path.read_text(encoding="utf-8"))
    return {
        "schema_version": 1,
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "case_timeout_seconds": case_timeout_seconds,
        "runs": [],
    }


def is_complete(state: dict[str, Any], stage: str, method: str = "",
                workload: str = "") -> bool:
    return any(
        row.get("stage") == stage
        and row.get("method", "") == method
        and row.get("workload", "") == workload
        and row.get("status") == "complete"
        for row in state["runs"]
    )


def run_stage(
    state: dict[str, Any], manifest: Path, log: Path, timeout: float,
    stage: str, command: list[str], method: str = "", workload: str = "",
) -> bool:
    if is_complete(state, stage, method, workload):
        print(f"[SKIP] {stage}/{method}/{workload}", flush=True)
        return True
    status, returncode, elapsed = bounded.run_bounded(command, log, timeout)
    state["runs"].append({
        "stage": stage,
        "method": method,
        "workload": workload,
        "status": status,
        "returncode": returncode,
        "elapsed_seconds": elapsed,
        "finished_at_utc": datetime.now(timezone.utc).isoformat(),
    })
    state["updated_at_utc"] = datetime.now(timezone.utc).isoformat()
    bounded.atomic_json(manifest, state)
    return status == "complete"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--screen-config", type=Path,
        default=HERE / "config.authoritative_amazon_screen_emptyfix.json")
    parser.add_argument(
        "--profile-config", type=Path,
        default=HERE / "config.amazon_representative_profile.json")
    parser.add_argument(
        "--run-root", type=Path,
        default=REPO / "runs/remaining_evidence_20260928")
    parser.add_argument("--case-timeout-seconds", type=float, default=3300.0)
    parser.add_argument("--wait-timeout-seconds", type=float, default=43200.0)
    parser.add_argument("--poll-seconds", type=float, default=30.0)
    args = parser.parse_args()
    if min(args.case_timeout_seconds, args.wait_timeout_seconds,
           args.poll_seconds) <= 0:
        parser.error("timeouts and poll interval must be positive")

    screen_config = args.screen_config.resolve()
    profile_config = args.profile_config.resolve()
    run_root = args.run_root.resolve()
    run_root.mkdir(parents=True, exist_ok=True)
    manifest = run_root / "supervisor_manifest.json"
    log = run_root / "campaign.log"
    state = load_state(manifest, args.case_timeout_seconds)
    state.update({
        "screen_config": str(screen_config),
        "profile_config": str(profile_config),
    })
    bounded.atomic_json(manifest, state)
    wait_for_builds(args.poll_seconds, args.wait_timeout_seconds)

    screen = json.loads(screen_config.read_text(encoding="utf-8"))
    workloads = [str(row["name"]) for row in screen["workloads"]]
    available = {str(row["name"]) for row in screen["methods"]}
    missing = set(TOPOLOGY_METHODS) - available
    if missing:
        raise ValueError(f"screen config is missing topology methods: {sorted(missing)}")
    for workload in workloads:
        for method in TOPOLOGY_METHODS:
            run_stage(
                state, manifest, log, args.case_timeout_seconds,
                "topology_query",
                [sys.executable, str(SEARCH_RUNNER), str(screen_config),
                 "--method", method, "--workload", workload],
                method, workload,
            )

    summary_timeout = min(args.case_timeout_seconds, 600.0)
    run_stage(
        state, manifest, log, summary_timeout, "topology_summarize",
        [sys.executable, str(SUMMARIZER), str(screen_config),
         "--baseline", "l0_lng_entry_optimized_lng", "--targets", "0.9"])
    points = (
        Path(screen["output_root"]) / "summary/performance/all_points.csv")
    figures = (
        Path(screen["output_root"]) / "summary/performance/figures_two_layer_topology")
    run_stage(
        state, manifest, log, summary_timeout, "topology_plot",
        [sys.executable, str(PLOTTER), str(screen_config), str(points),
         str(figures), "--family", "two_layer_topology"])
    paper_figures = (
        Path(screen["output_root"]) / "summary/performance/figures")
    run_stage(
        state, manifest, log, summary_timeout, "paper_plots",
        [sys.executable, str(PLOTTER), str(screen_config), str(points),
         str(paper_figures)])

    profile = json.loads(profile_config.read_text(encoding="utf-8"))
    for workload_row in profile["workloads"]:
        workload = str(workload_row["name"])
        for method_row in profile["methods"]:
            method = str(method_row["name"])
            run_stage(
                state, manifest, log, args.case_timeout_seconds,
                "profile_query",
                [sys.executable, str(SEARCH_RUNNER), str(profile_config),
                 "--method", method, "--workload", workload],
                method, workload,
            )
    if run_stage(
        state, manifest, log, summary_timeout, "profile_validate",
        [sys.executable, str(VALIDATOR), str(profile_config)],
    ):
        run_stage(
            state, manifest, log, summary_timeout, "profile_summarize",
            [sys.executable, str(SUMMARIZER), str(profile_config),
             "--baseline", "l0_lng_entry_optimized_lng", "--targets", "0.9"])

    state["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
    state["status"] = (
        "complete" if all(row["status"] == "complete" for row in state["runs"])
        else "complete_with_failures")
    bounded.atomic_json(manifest, state)
    print(manifest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
