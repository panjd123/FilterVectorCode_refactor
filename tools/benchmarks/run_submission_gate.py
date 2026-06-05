#!/usr/bin/env python3
"""Run the reviewer-facing submission gates in one command.

This combines four checks that were previously run manually:

1. Python syntax checks for the paper/benchmark helper scripts.
2. Shell syntax checks for key benchmark entrypoints.
3. Claim-language lint for paper drafts, reports, README, and runbooks.
4. Artifact audit for required and pending paper evidence.

Use the default mode during drafting. Use --final only when checking whether the
paper is truly submission-ready; it makes pending/missing experiments fail.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


DEFAULT_PYTHON_FILES = (
    "tools/benchmarks/audit_paper_artifacts.py",
    "tools/benchmarks/check_paper_claim_language.py",
    "tools/benchmarks/summarize_cross_baselines.py",
    "tools/benchmarks/summarize_graph_diagnostics.py",
    "tools/benchmarks/summarize_group_graph_router_ab.py",
    "tools/benchmarks/summarize_x400_ab.py",
)


DEFAULT_SHELL_FILES = (
    "scripts/benchmarks/common.sh",
    "scripts/benchmarks/run_end_to_end_recall_ab.sh",
    "scripts/benchmarks/run_group_graph_router_ab.sh",
    "scripts/benchmarks/run_x400_reverse_tail_ab.sh",
)


def run_step(name: str, cmd: list[str]) -> int:
    print(f"\n[gate] {name}", flush=True)
    print("[cmd] " + " ".join(cmd), flush=True)
    proc = subprocess.run(cmd)
    if proc.returncode == 0:
        print(f"[ok] {name}", flush=True)
    else:
        print(f"[fail] {name}: exit={proc.returncode}", flush=True)
    return proc.returncode


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--x400-root", type=Path, help="Output root from run_x400_reverse_tail_ab.sh.")
    parser.add_argument("--router-root", type=Path, help="Output root from run_group_graph_router_ab.sh.")
    parser.add_argument(
        "--final",
        action="store_true",
        help="Use final submission gating: fail on any pending/missing artifact.",
    )
    parser.add_argument(
        "--skip-py-compile",
        action="store_true",
        help="Skip Python syntax checks.",
    )
    parser.add_argument(
        "--skip-shell-check",
        action="store_true",
        help="Skip shell syntax checks for key benchmark entrypoints.",
    )
    parser.add_argument(
        "--skip-language",
        action="store_true",
        help="Skip paper claim-language lint.",
    )
    args = parser.parse_args()

    failures = 0
    py = sys.executable or "python3"

    if not args.skip_py_compile:
        failures += run_step(
            "python syntax",
            [py, "-m", "py_compile", *DEFAULT_PYTHON_FILES],
        )

    if not args.skip_shell_check:
        failures += run_step(
            "shell syntax",
            ["bash", "-n", *DEFAULT_SHELL_FILES],
        )

    if not args.skip_language:
        failures += run_step(
            "paper claim language",
            [py, "tools/benchmarks/check_paper_claim_language.py"],
        )

    audit_cmd = [py, "tools/benchmarks/audit_paper_artifacts.py"]
    if args.x400_root:
        audit_cmd.extend(["--x400-root", str(args.x400_root)])
    if args.router_root:
        audit_cmd.extend(["--router-root", str(args.router_root)])
    audit_cmd.append("--fail-submission" if args.final else "--fail-required")
    failures += run_step("paper artifact audit", audit_cmd)

    if failures:
        print(f"\n[gate] FAILED with aggregate exit sum {failures}.", flush=True)
        raise SystemExit(1)
    print("\n[gate] PASSED.", flush=True)


if __name__ == "__main__":
    main()
