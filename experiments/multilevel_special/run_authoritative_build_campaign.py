#!/usr/bin/env python3
"""Run the resumable build-time and build-resource campaign."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


HERE = Path(__file__).resolve().parent
QUERY_PROFILE_CONFIG = HERE / "config.authoritative_amazon_profile_emptyfix.json"
PHASES = ("base_timing", "hierarchy_timing", "base_resource",
          "hierarchy_resource", "quality_screen", "quality_crossing",
          "quality_formal", "summarize")
CONFIGS = {
    "base_timing": ("run_base_topology_build.py",
                    "config.authoritative_amazon_base_build_timing.json"),
    "hierarchy_timing": ("run_build_sweep.py",
                         "config.authoritative_amazon_hierarchy_build_timing.json"),
    "base_resource": ("run_base_topology_build.py",
                      "config.authoritative_amazon_base_build_resource.json"),
    "hierarchy_resource": ("run_build_sweep.py",
                           "config.authoritative_amazon_hierarchy_build_resource.json"),
}


def run(*args: str) -> None:
    command = [sys.executable, *args]
    print("+", subprocess.list2cmdline(command), flush=True)
    subprocess.run(command, cwd=HERE, check=True)


def validate_query_gate(profile_config: Path) -> None:
    if not profile_config.is_file():
        raise RuntimeError(
            "authoritative query profile is not complete: missing config "
            f"{profile_config}")
    run("validate_selection_sweep.py", str(profile_config))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--start-at", choices=PHASES, default="base_timing")
    parser.add_argument("--stop-after", choices=PHASES, default="summarize")
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--query-profile-config", type=Path,
                        default=QUERY_PROFILE_CONFIG)
    parser.add_argument(
        "--skip-query-gate", action="store_true",
        help="development-only override; never use for authoritative build results")
    args = parser.parse_args()
    start = PHASES.index(args.start_at)
    stop = PHASES.index(args.stop_after)
    if stop < start:
        parser.error("--stop-after must not precede --start-at")
    if not args.skip_query_gate:
        validate_query_gate(args.query_profile_config.resolve())
    run("generate_authoritative_build_configs.py", "--repeats", str(args.repeats))
    for phase in PHASES[start:stop + 1]:
        print(f"\n=== authoritative build phase: {phase} ===", flush=True)
        if phase == "quality_screen":
            run("generate_authoritative_build_quality.py")
            run("run_selection_sweep.py",
                str(HERE / "config.authoritative_amazon_build_quality_screen.json"))
            run("validate_selection_sweep.py",
                str(HERE / "config.authoritative_amazon_build_quality_screen.json"))
        elif phase == "quality_crossing":
            run("advance_authoritative_campaign.py", "crossing",
                str(HERE / "config.authoritative_amazon_build_quality_screen.json"),
                str(HERE / "config.authoritative_amazon_build_quality_crossing.json"))
            run("run_selection_sweep.py",
                str(HERE / "config.authoritative_amazon_build_quality_crossing.json"))
            run("validate_selection_sweep.py",
                str(HERE / "config.authoritative_amazon_build_quality_crossing.json"))
        elif phase == "quality_formal":
            run("advance_authoritative_campaign.py", "formal",
                str(HERE / "config.authoritative_amazon_build_quality_crossing.json"),
                str(HERE / "config.authoritative_amazon_build_quality_formal.json"))
            run("run_selection_sweep.py",
                str(HERE / "config.authoritative_amazon_build_quality_formal.json"))
            run("validate_selection_sweep.py",
                str(HERE / "config.authoritative_amazon_build_quality_formal.json"))
        elif phase == "summarize":
            run("summarize_authoritative_build.py", "--output-dir",
                str(HERE / "results_summary/authoritative_build"))
        else:
            script, config = CONFIGS[phase]
            run(script, str(HERE / config))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
