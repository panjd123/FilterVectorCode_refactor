#!/usr/bin/env python3
"""Run the held-out DRH versus frozen manual-oracle campaign."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


HERE = Path(__file__).resolve().parent
DATASETS = ("genome", "reviews", "variousimg")
DISPLAY_NAMES = {
    "genome": "Genome",
    "reviews": "Reviews",
    "variousimg": "VariousImg",
}
PHASES = ("build", "screen", "crossing", "formal", "profile", "summarize")


def run(*args: str) -> None:
    command = [sys.executable, *args]
    print("+", subprocess.list2cmdline(command), flush=True)
    subprocess.run(command, cwd=HERE, check=True)


def config_path(dataset: str, phase: str) -> Path:
    return HERE / f"config.authoritative_heldout_{dataset}_{phase}.json"


def run_search_phase(dataset: str, phase: str) -> None:
    config = config_path(dataset, phase)
    run("run_selection_sweep.py", str(config))
    run("validate_selection_sweep.py", str(config))
    run("summarize_selection_sweep.py", str(config),
        "--baseline", "l0_lng_entry_optimized_lng", "--targets", "0.9")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", action="append", choices=DATASETS)
    parser.add_argument("--start-at", choices=PHASES, default="build")
    parser.add_argument("--stop-after", choices=PHASES, default="summarize")
    parser.add_argument(
        "--skip-generate", action="store_true",
        help="reuse already frozen build/screen configs instead of regenerating them")
    parser.add_argument(
        "--summary-output", type=Path,
        default=HERE / "results_summary/heldout_oracle")
    args = parser.parse_args()
    selected = args.dataset or list(DATASETS)
    start = PHASES.index(args.start_at)
    stop = PHASES.index(args.stop_after)
    if stop < start:
        parser.error("--stop-after must not precede --start-at")

    if not args.skip_generate and start == 0:
        generator_args = ["generate_auto_policy_cross_dataset_configs.py"]
        for dataset in selected:
            generator_args.extend(["--dataset", DISPLAY_NAMES[dataset]])
        run(*generator_args)

    for phase in PHASES[start:stop + 1]:
        print(f"\n=== held-out phase: {phase} ===", flush=True)
        if phase == "summarize":
            formal_configs = [str(config_path(dataset, "formal"))
                              for dataset in selected]
            run("summarize_heldout_oracle.py", *formal_configs,
                "--output-dir", str(args.summary_output.resolve()))
            continue
        for dataset in selected:
            print(f"--- {dataset} ---", flush=True)
            if phase == "build":
                run("run_build_sweep.py", str(config_path(dataset, "build")))
            elif phase == "screen":
                run_search_phase(dataset, "screen")
            elif phase == "crossing":
                run("advance_authoritative_campaign.py", "crossing",
                    str(config_path(dataset, "screen")),
                    str(config_path(dataset, "crossing")))
                run_search_phase(dataset, "crossing")
            elif phase == "formal":
                run("advance_authoritative_campaign.py", "formal",
                    str(config_path(dataset, "crossing")),
                    str(config_path(dataset, "formal")))
                run_search_phase(dataset, "formal")
            elif phase == "profile":
                run("advance_authoritative_campaign.py", "profile",
                    str(config_path(dataset, "formal")),
                    str(config_path(dataset, "profile")))
                run_search_phase(dataset, "profile")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
