#!/usr/bin/env python3
"""One entry point for declaring, running, validating, and summarizing sweeps."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import experiment_core


HERE = Path(__file__).resolve().parent


def print_matrix(config: dict) -> None:
    fields = ["method", "main_graph", "entry_group_provider", "entry_structure",
              "coverage_structure", "special_overlay", "layer_count",
              "block_partition", "t1", "t2"]
    print("| " + " | ".join(fields) + " |")
    print("|" + "|".join("---" for _ in fields) + "|")
    for method in config["methods"]:
        row = experiment_core.method_semantics(config, method)
        print("| " + " | ".join(str(row.get(key, "")) for key in fields) + " |")


def delegate(script: str, args: list[str]) -> int:
    return subprocess.run([sys.executable, str(HERE / script), *args]).returncode


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in ("check", "matrix", "validate"):
        child = subparsers.add_parser(command)
        child.add_argument("config", type=Path)
    run = subparsers.add_parser("run")
    run.add_argument("config", type=Path)
    run.add_argument("--method", action="append", default=[])
    run.add_argument("--workload", action="append", default=[])
    run.add_argument("--force", action="store_true")
    run.add_argument("--dry-run", action="store_true")
    summarize = subparsers.add_parser("summarize")
    summarize.add_argument("config", type=Path)
    summarize.add_argument("--baseline", required=True)
    summarize.add_argument("--output", type=Path)
    summarize.add_argument("--bootstrap-samples", type=int)
    summarize.add_argument(
        "--allow-partial", action="store_true",
        help="emit unavailable rows instead of failing on incomplete cases")
    args = parser.parse_args(argv)

    config = experiment_core.load_config(args.config)
    if args.command == "check":
        protocol = experiment_core.protocol_for(config)
        print(json.dumps({
            "status": "valid", "phase": protocol.phase,
            "cases": sum(1 for _ in experiment_core.iter_cases(config)),
            "cold_repeats": protocol.cold_repeats,
            "measured_repeats": protocol.measured_repeats,
        }, indent=2))
        return 0
    if args.command == "matrix":
        print_matrix(config)
        return 0
    if args.command == "validate":
        return delegate("validate_selection_sweep.py", [str(args.config)])
    if args.command == "run":
        forwarded = [str(args.config)]
        for method in args.method:
            forwarded.extend(["--method", method])
        for workload in args.workload:
            forwarded.extend(["--workload", workload])
        if args.force:
            forwarded.append("--force")
        if args.dry_run:
            forwarded.append("--dry-run")
        return delegate("run_selection_sweep.py", forwarded)
    rows = experiment_core.summarize_experiment(
        config, args.baseline, bootstrap_samples=args.bootstrap_samples,
        allow_partial=args.allow_partial)
    output = args.output or Path(config["output_root"]) / "summary" / "results.csv"
    experiment_core.write_csv(output, rows)
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
