#!/usr/bin/env python3
"""Fail-closed generation and compilation for the completed evidence set."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import run_deadline_build_campaign as bounded


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
PAPER = REPO / "docs/papers/multilevel_ung"
RUNS = REPO / "runs"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run(command: list[str], cwd: Path) -> None:
    print("+ " + subprocess.list2cmdline(command), flush=True)
    subprocess.run(command, cwd=cwd, check=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--tectonic", type=Path,
        default=Path("/home/sunyahui/.local/opt/tectonic-0.15.0-musl/tectonic"))
    parser.add_argument(
        "--output-manifest", type=Path,
        default=RUNS / "complete_evidence_20260928/finalization_manifest.json")
    args = parser.parse_args()
    if not args.tectonic.is_file():
        raise FileNotFoundError(args.tectonic)

    amazon_summary = (
        RUNS / "authoritative_multilevel_20260926_emptyfix/search/amazon_screen/summary/performance")
    figures = amazon_summary / "figures"
    heldout_root = RUNS / "authoritative_multilevel_20260926_emptyfix/heldout"
    deadline_summary = RUNS / "deadline_evidence_20260927_cpufix/deadline_summary"
    profile_root = RUNS / "deadline_profile_20260927/heldout"
    build_root = RUNS / "deadline_build_20260927"
    amazon_profile_root = RUNS / "amazon_representative_profile_20260928"
    remaining_manifest = RUNS / "remaining_evidence_20260928/supervisor_manifest.json"
    generator_manifest = RUNS / "deadline_paper_20260928/manifest.json"
    generator = HERE / "generate_deadline_paper_results.py"
    command = [
        sys.executable, str(generator),
        "--amazon-equal-recall", str(amazon_summary / "equal_recall_conservative.csv"),
        "--amazon-points", str(amazon_summary / "all_points.csv"),
        "--figures", str(figures),
        "--deadline-summary", str(deadline_summary),
        "--policy", str(heldout_root / "genome/policy.json"),
        "--policy", str(heldout_root / "reviews/policy.json"),
        "--policy", str(heldout_root / "variousimg/policy.json"),
        "--profile", str(profile_root / "genome/search/summary/profile/equal_recall_conservative.csv"),
        "--profile", str(profile_root / "reviews/search/summary/profile/equal_recall_conservative.csv"),
        "--profile", str(profile_root / "variousimg/search/summary/profile/equal_recall_conservative.csv"),
        "--amazon-profile-points", str(amazon_profile_root / "summary/profile/all_points.csv"),
        "--amazon-profile-selection", str(amazon_profile_root / "selection_manifest.json"),
        "--amazon-profile-supervisor-manifest", str(remaining_manifest),
        "--drh-v2-root", str(RUNS / "drh_v2_20260927"),
        "--build-manifest", str(build_root / "deadline_build_supervisor_manifest.json"),
        "--build-summary", str(build_root / "build_study/summary/build_summary.csv"),
        "--build-end-to-end", str(build_root / "build_study/summary/build_end_to_end.csv"),
        "--require-two-layer-topology",
        "--figure-output-dir", str(PAPER / "generated_figures"),
        "--manifest-output", str(generator_manifest),
        "--tex-output", str(PAPER / "generated_results.tex"),
        "--report-output", str(REPO / "docs/reports/MULTILEVEL_SPECIAL_BLOCK_AUTHORITATIVE_RESULTS_CN.md"),
    ]
    run(command, HERE)
    run(["git", "diff", "--check"], REPO)
    run([
        sys.executable, "-m", "unittest", "discover",
        "-s", str(HERE), "-p", "test_*.py",
    ], REPO)
    run([str(args.tectonic), "main.tex", "--keep-logs"], PAPER)
    log = (PAPER / "main.log").read_text(encoding="utf-8", errors="replace")
    forbidden = re.compile(
        r"undefined (reference|citation)|there were undefined|Emergency stop|Fatal error",
        re.IGNORECASE)
    match = forbidden.search(log)
    if match:
        raise RuntimeError(f"fatal or unresolved TeX diagnostic: {match.group(0)}")
    outputs = [
        PAPER / "generated_results.tex",
        PAPER / "main.pdf",
        REPO / "docs/reports/MULTILEVEL_SPECIAL_BLOCK_AUTHORITATIVE_RESULTS_CN.md",
        generator_manifest,
    ]
    for path in outputs:
        if not path.is_file() or path.stat().st_size == 0:
            raise FileNotFoundError(path)
    payload = {
        "schema_version": 1,
        "finished_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "complete_with_declared_timeout",
        "declared_incomplete_evidence": [{
            "stage": "profile_query",
            "method": "l0_trie_entry_trie",
            "workload": "sel_95",
            "reason": "exceeded the fixed 3300-second per-case cap",
            "supervisor_manifest": str(remaining_manifest.resolve()),
        }],
        "generator_command": command,
        "tectonic": str(args.tectonic.resolve()),
        "tectonic_sha256": sha256(args.tectonic),
        "outputs": [
            {"path": str(path.resolve()), "sha256": sha256(path),
             "bytes": path.stat().st_size}
            for path in outputs
        ],
    }
    bounded.atomic_json(args.output_manifest.resolve(), payload)
    print(args.output_manifest.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
