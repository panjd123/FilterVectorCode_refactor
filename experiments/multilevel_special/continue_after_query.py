#!/usr/bin/env python3
"""Start held-out queries, then builds, after Amazon query validation passes."""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
import time
from pathlib import Path


HERE = Path(__file__).resolve().parent
VALIDATOR = HERE / "validate_selection_sweep.py"
HELDOUT_RUNNER = HERE / "run_heldout_campaign.py"
BUILD_RUNNER = HERE / "run_authoritative_build_campaign.py"


def profile_manifest_path(config_path: Path) -> Path | None:
    if not config_path.is_file():
        return None
    config = json.loads(config_path.read_text())
    root = Path(config["output_root"])
    pass_name = str(config.get("measurement_pass", "performance"))
    return root / (f"manifest_{pass_name}.json"
                   if config.get("pass_subdirs", False) else "manifest.json")


def profile_valid(config_path: Path) -> bool:
    manifest = profile_manifest_path(config_path)
    if manifest is None or not manifest.is_file():
        return False
    return subprocess.run(
        [sys.executable, str(VALIDATOR), str(config_path)], cwd=HERE,
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        check=False,
    ).returncode == 0


def active_query_processes() -> list[str]:
    result = subprocess.run(
        ["pgrep", "-af",
         "run_authoritative_campaign.py|run_selection_sweep.py|search_UNG_index"],
        text=True, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
        check=False,
    )
    return [line for line in result.stdout.splitlines()
            if "continue_after_query.py" not in line]


def session_exists(name: str) -> bool:
    return subprocess.run(
        ["tmux", "has-session", "-t", name], stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL, check=False,
    ).returncode == 0


def pipeline_command(
    heldout_log: Path, build_log: Path, run_build: bool,
) -> str:
    heldout = (
        f"{shlex.quote(sys.executable)} -u {shlex.quote(str(HELDOUT_RUNNER))} "
        "--skip-generate")
    command = (
        "set -o pipefail; "
        f"cd {shlex.quote(str(HERE))} && "
        f"( {heldout} 2>&1 | tee -a {shlex.quote(str(heldout_log))} )")
    if run_build:
        build = f"{shlex.quote(sys.executable)} -u {shlex.quote(str(BUILD_RUNNER))}"
        command += (
            f" && ( {build} 2>&1 | tee -a {shlex.quote(str(build_log))} )")
    return command


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--profile-config", type=Path,
        default=HERE / "config.authoritative_amazon_profile_emptyfix.json")
    parser.add_argument("--session", default="fv_heldout_then_build_20260926")
    parser.add_argument("--poll-seconds", type=float, default=60.0)
    parser.add_argument(
        "--heldout-log", type=Path,
        default=HERE.parent.parent /
        "runs/authoritative_multilevel_20260926_emptyfix/heldout.log")
    parser.add_argument(
        "--build-log", type=Path,
        default=HERE.parent.parent /
        "runs/authoritative_multilevel_20260926_emptyfix/build.log")
    parser.add_argument("--run-build-after-heldout", action="store_true")
    args = parser.parse_args()
    args.profile_config = args.profile_config.resolve()
    args.heldout_log = args.heldout_log.resolve()
    args.build_log = args.build_log.resolve()

    while not profile_valid(args.profile_config):
        time.sleep(args.poll_seconds)
    while active_query_processes():
        time.sleep(args.poll_seconds)
    if session_exists(args.session):
        raise RuntimeError(f"target tmux session already exists: {args.session}")
    args.heldout_log.parent.mkdir(parents=True, exist_ok=True)
    args.build_log.parent.mkdir(parents=True, exist_ok=True)
    command = pipeline_command(
        args.heldout_log, args.build_log, args.run_build_after_heldout)
    subprocess.run(
        ["tmux", "new-session", "-d", "-s", args.session,
         "bash", "-lc", command], check=True)
    print(f"started {args.session}: held-out"
          f"{' -> build' if args.run_build_after_heldout else ''}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
