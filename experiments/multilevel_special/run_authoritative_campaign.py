#!/usr/bin/env python3
"""Run the resumable authoritative Amazon experiment pipeline."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


HERE = Path(__file__).resolve().parent
HIERARCHY_CONFIG = HERE / "config.authoritative_amazon_hierarchy_grid.json"
SCREEN_CONFIG = HERE / "config.authoritative_amazon_screen.json"
CROSSING_CONFIG = HERE / "config.authoritative_amazon_crossing.json"
FORMAL_CONFIG = HERE / "config.authoritative_amazon_formal.json"
PROFILE_CONFIG = HERE / "config.authoritative_amazon_profile.json"
PHASES = ("screen", "crossing", "formal", "profile")


def run(*args: str) -> None:
    command = [sys.executable, *args]
    print("+", subprocess.list2cmdline(command), flush=True)
    subprocess.run(command, cwd=HERE, check=True)


def pass_summary_root(config_path: Path) -> Path:
    config = json.loads(config_path.read_text())
    root = Path(config["output_root"]) / "summary"
    if config.get("pass_subdirs", False):
        root /= str(config.get("measurement_pass", "performance"))
    return root


def finalize_outputs(screen_config: Path, formal_config: Path) -> None:
    """Generate final tables and figures only after all query phases pass."""
    screen_summary = pass_summary_root(screen_config)
    formal_summary = pass_summary_root(formal_config)
    run(
        "plot_authoritative_recall_qps.py", str(screen_config),
        str(screen_summary / "all_points.csv"),
        str(screen_summary / "figures"),
    )
    run(
        "summarize_depth_ablation.py", str(formal_config),
        "--output-dir", str(formal_summary / "depth_ablation"),
    )


def validate_hierarchy(hierarchy_config: Path) -> None:
    config = json.loads(hierarchy_config.read_text())
    manifest_path = Path(config["output_root"]) / "manifest.json"
    if not manifest_path.is_file():
        raise RuntimeError(f"hierarchy manifest is missing: {manifest_path}")
    manifest = json.loads(manifest_path.read_text())
    latest = {row.get("name"): row for row in manifest.get("runs", [])}
    problems = []
    for case in config["cases"]:
        row = latest.get(case["name"])
        if row is None or row.get("status") != "complete":
            problems.append(f"{case['name']}: {None if row is None else row.get('status')}")
            continue
        expected = ",".join(
            f"{layer['min_points']}:{layer['topology']}"
            for layer in case["hierarchy_layers"])
        metadata = row.get("metadata", {})
        if metadata.get("hierarchy_layers") != expected:
            problems.append(f"{case['name']}: hierarchy metadata mismatch")
        if metadata.get("base_group_topology") != case["base_topology"]:
            problems.append(f"{case['name']}: base topology metadata mismatch")
    if problems:
        raise RuntimeError("hierarchy build is incomplete:\n" + "\n".join(problems))


def execute_phase(phase: str, hierarchy_config: Path,
                  configs: dict[str, Path],
                  screen_output_root: Path | None) -> None:
    if phase == "screen":
        validate_hierarchy(hierarchy_config)
        generate_args = [
            "generate_authoritative_campaign.py",
            "--hierarchy-config", str(hierarchy_config),
            "--output", str(configs["screen"]),
        ]
        if screen_output_root is not None:
            generate_args.extend(["--output-root", str(screen_output_root)])
        run(*generate_args)
        config = configs["screen"]
    elif phase == "crossing":
        run("advance_authoritative_campaign.py", "crossing",
            str(configs["screen"]), str(configs["crossing"]))
        config = configs["crossing"]
    elif phase == "formal":
        run("advance_authoritative_campaign.py", "formal",
            str(configs["crossing"]), str(configs["formal"]))
        config = configs["formal"]
    else:
        run("advance_authoritative_campaign.py", "profile",
            str(configs["formal"]), str(configs["profile"]))
        config = configs["profile"]
    run("run_selection_sweep.py", str(config))
    run("validate_selection_sweep.py", str(config))
    run("summarize_selection_sweep.py", str(config),
        "--baseline", "l0_lng_entry_optimized_lng", "--targets", "0.9")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--start-at", choices=PHASES, default="screen")
    parser.add_argument("--stop-after", choices=PHASES, default="profile")
    parser.add_argument("--hierarchy-config", type=Path, default=HIERARCHY_CONFIG)
    parser.add_argument("--screen-config", type=Path, default=SCREEN_CONFIG)
    parser.add_argument("--crossing-config", type=Path, default=CROSSING_CONFIG)
    parser.add_argument("--formal-config", type=Path, default=FORMAL_CONFIG)
    parser.add_argument("--profile-config", type=Path, default=PROFILE_CONFIG)
    parser.add_argument(
        "--screen-output-root", type=Path,
        help="override the generated screen output root without changing index roots")
    args = parser.parse_args()
    args.hierarchy_config = args.hierarchy_config.resolve()
    args.screen_config = args.screen_config.resolve()
    args.crossing_config = args.crossing_config.resolve()
    args.formal_config = args.formal_config.resolve()
    args.profile_config = args.profile_config.resolve()
    if args.screen_output_root is not None:
        args.screen_output_root = args.screen_output_root.resolve()
    start = PHASES.index(args.start_at)
    stop = PHASES.index(args.stop_after)
    if stop < start:
        parser.error("--stop-after must not precede --start-at")
    configs = {
        "screen": args.screen_config,
        "crossing": args.crossing_config,
        "formal": args.formal_config,
        "profile": args.profile_config,
    }
    for phase in PHASES[start:stop + 1]:
        print(f"\n=== authoritative phase: {phase} ===", flush=True)
        execute_phase(phase, args.hierarchy_config, configs,
                      args.screen_output_root)
    if args.stop_after == "profile":
        finalize_outputs(args.screen_config, args.formal_config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
