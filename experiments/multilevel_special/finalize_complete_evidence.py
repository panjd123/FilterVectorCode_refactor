#!/usr/bin/env python3
"""Fail-closed generation and compilation for the completed evidence set."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
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
    topology_config = HERE / "config.authoritative_amazon_base_trie_query_grid.json"
    topology_run = RUNS / "base_topology_factorial_20260929"
    topology_summary = topology_run / "summary"
    trie_summary = topology_run / "search/amazon_base_trie_4c99/summary/performance"
    run([
        sys.executable, str(HERE / "experiment_cli.py"), "validate",
        str(topology_config),
    ], HERE)
    run([
        sys.executable, str(HERE / "summarize_selection_sweep.py"),
        str(topology_config),
        "--baseline", "l1_base_trie_t1024_lng_entry_trie",
        "--targets", "0.9",
    ], HERE)
    run([
        sys.executable, str(HERE / "summarize_base_topology_factorial.py"),
        str(amazon_summary / "equal_recall_conservative.csv"),
        str(trie_summary / "equal_recall_conservative.csv"),
        str(topology_summary),
        "--lng-all-points", str(amazon_summary / "all_points.csv"),
        "--trie-all-points", str(trie_summary / "all_points.csv"),
        "--lng-manifest", str(
            RUNS / "authoritative_multilevel_20260926_emptyfix/search/amazon_screen/manifest_performance.json"
        ),
        "--trie-manifest", str(
            topology_run / "search/amazon_base_trie_4c99/manifest_performance.json"
        ),
        "--lng-hierarchy-manifest", str(
            RUNS / "authoritative_multilevel_20260925/hierarchy/amazon_base_lng/manifest.json"
        ),
        "--trie-hierarchy-manifest", str(
            topology_run / "hierarchy/amazon_base_trie/manifest.json"
        ),
    ], HERE)
    shutil.copy2(
        topology_summary / "generated_topology_factorial.tex",
        PAPER / "generated_topology_factorial.tex",
    )
    paper_data = PAPER / "generated_data"
    paper_data.mkdir(parents=True, exist_ok=True)
    for name in (
        "factorial_equal_recall.csv",
        "best_by_selectivity.csv",
        "upper_trie_pairwise.csv",
        "base_matched_pairwise.csv",
        "trie_effect_summary.csv",
        "global_configuration_summary.csv",
        "manifest.json",
    ):
        shutil.copy2(topology_summary / name, paper_data / name)
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
        "--topology-factorial-dir", str(topology_summary),
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
    run([str(args.tectonic), "main.tex", "--keep-logs", "--keep-intermediates"], PAPER)
    log = (PAPER / "main.log").read_text(encoding="utf-8", errors="replace")
    forbidden = re.compile(
        r"undefined (reference|citation)|there were undefined|Emergency stop|Fatal error",
        re.IGNORECASE)
    match = forbidden.search(log)
    if match:
        raise RuntimeError(f"fatal or unresolved TeX diagnostic: {match.group(0)}")
    outputs = [
        PAPER / "generated_results.tex",
        PAPER / "generated_topology_factorial.tex",
        paper_data / "factorial_equal_recall.csv",
        paper_data / "best_by_selectivity.csv",
        paper_data / "upper_trie_pairwise.csv",
        paper_data / "base_matched_pairwise.csv",
        paper_data / "trie_effect_summary.csv",
        paper_data / "global_configuration_summary.csv",
        paper_data / "manifest.json",
        PAPER / "main.pdf",
        REPO / "docs/reports/MULTILEVEL_SPECIAL_BLOCK_AUTHORITATIVE_RESULTS_CN.md",
        topology_summary / "README.md",
        topology_summary / "factorial_equal_recall.csv",
        topology_summary / "upper_trie_pairwise.csv",
        topology_summary / "base_matched_pairwise.csv",
        topology_summary / "trie_effect_summary.csv",
        topology_summary / "global_configuration_summary.csv",
        generator_manifest,
    ]
    # Include nested section sources: the manuscript is no longer one TeX file.
    outputs.extend(sorted(PAPER.rglob("*.tex")))
    outputs.extend(sorted(PAPER.glob("*.bib")))
    outputs.extend([PAPER / "ADVISOR_NOTE_CN.md", PAPER / "PAPER_OUTLINE_CN.md"])
    outputs = list(dict.fromkeys(outputs))
    for path in outputs:
        if not path.is_file() or path.stat().st_size == 0:
            raise FileNotFoundError(path)
    payload = {
        "schema_version": 1,
        "finished_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "complete_with_declared_timeout",
        "scope": "bounded evidence generation; not the formal publication gate",
        "unestablished_claims": [
            "matched-Recall quality across accelerated construction profiles",
            "current-protocol external-system superiority",
            "depth-change generalization of the automatic policy",
        ],
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
