#!/usr/bin/env python3
"""Validate a completed screen and resume the remaining campaign phases."""

from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
import time
from pathlib import Path


HERE = Path(__file__).resolve().parent
SCREEN_CONFIG = HERE / "config.authoritative_amazon_screen.json"
RUNNER = HERE / "run_authoritative_campaign.py"
VALIDATOR = HERE / "validate_selection_sweep.py"
CAMPAIGN_LOG = (
    HERE.parent.parent / "runs/authoritative_multilevel_20260925/campaign.log")


def session_exists(name: str) -> bool:
    return subprocess.run(
        ["tmux", "has-session", "-t", name],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        check=False,
    ).returncode == 0


def campaign_processes() -> list[str]:
    result = subprocess.run(
        ["pgrep", "-af",
         "run_authoritative_campaign.py|run_selection_sweep.py|search_UNG_index"],
        text=True, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
        check=False,
    )
    return [line for line in result.stdout.splitlines()
            if "continue_after_screen.py" not in line]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--session", default="fv_auth_campaign_20260925")
    parser.add_argument("--next-session")
    parser.add_argument("--poll-seconds", type=float, default=60.0)
    parser.add_argument("--screen-config", type=Path, default=SCREEN_CONFIG)
    parser.add_argument("--crossing-config", type=Path,
                        default=HERE / "config.authoritative_amazon_crossing.json")
    parser.add_argument("--formal-config", type=Path,
                        default=HERE / "config.authoritative_amazon_formal.json")
    parser.add_argument("--profile-config", type=Path,
                        default=HERE / "config.authoritative_amazon_profile.json")
    parser.add_argument("--campaign-log", type=Path, default=CAMPAIGN_LOG)
    args = parser.parse_args()
    args.screen_config = args.screen_config.resolve()
    args.crossing_config = args.crossing_config.resolve()
    args.formal_config = args.formal_config.resolve()
    args.profile_config = args.profile_config.resolve()
    args.campaign_log = args.campaign_log.resolve()

    while session_exists(args.session):
        time.sleep(args.poll_seconds)

    leftovers = campaign_processes()
    if leftovers:
        raise RuntimeError(
            "refusing to start a concurrent campaign:\n" + "\n".join(leftovers))

    subprocess.run(
        [sys.executable, str(VALIDATOR), str(args.screen_config)],
        cwd=HERE, check=True,
    )
    next_session = args.next_session or args.session
    command = (
        "set -o pipefail; "
        f"cd {shlex.quote(str(HERE))} && "
        f"{shlex.quote(sys.executable)} -u {shlex.quote(str(RUNNER))} "
        "--start-at crossing "
        f"--screen-config {shlex.quote(str(args.screen_config))} "
        f"--crossing-config {shlex.quote(str(args.crossing_config))} "
        f"--formal-config {shlex.quote(str(args.formal_config))} "
        f"--profile-config {shlex.quote(str(args.profile_config))} "
        "2>&1 | "
        f"tee -a {shlex.quote(str(args.campaign_log))}"
    )
    subprocess.run(
        ["tmux", "new-session", "-d", "-s", next_session,
         "bash", "-lc", command],
        check=True,
    )
    print(f"started {next_session}: crossing -> formal -> profile", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
