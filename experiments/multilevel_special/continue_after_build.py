#!/usr/bin/env python3
"""Finalize the authoritative study after held-out and build work completes."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path


HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
RUN_ROOT = REPO / "runs/authoritative_multilevel_20260926_emptyfix"
PAPER_DIR = REPO / "docs/papers/multilevel_ung"
RESULTS = HERE / "results_summary"
DEFAULT_TECTONIC = Path("/home/sunyahui/.local/opt/tectonic-0.15.0-musl/tectonic")
ACTIVE_PATTERN = (
    "run_authoritative_campaign.py|run_heldout_campaign.py|"
    "run_authoritative_build_campaign.py|run_selection_sweep.py|"
    "search_UNG_index|build_UNG_index|build_special_block_index"
)


def run(command: list[str], cwd: Path = HERE) -> None:
    print("+", subprocess.list2cmdline(command), flush=True)
    subprocess.run(command, cwd=cwd, check=True)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def active_experiment_processes() -> list[str]:
    result = subprocess.run(
        ["pgrep", "-af", ACTIVE_PATTERN], text=True,
        stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, check=False)
    return [
        line for line in result.stdout.splitlines()
        if "pgrep -af" not in line and "continue_after_build.py" not in line
    ]


def required_artifacts(run_root: Path, results: Path) -> list[Path]:
    return [
        results / "heldout_oracle/heldout_oracle_by_workload.csv",
        results / "heldout_oracle/heldout_oracle_global.csv",
        results / "authoritative_build/build_summary.csv",
        results / "authoritative_build/build_end_to_end.csv",
        run_root / "search/amazon_formal/summary/performance/equal_recall_conservative.csv",
    ]


def paper_generation_command(run_root: Path, results: Path) -> list[str]:
    command = [
        sys.executable, str(HERE / "generate_authoritative_paper_results.py"),
        "--amazon-formal-config",
        str(HERE / "config.authoritative_amazon_formal_emptyfix.json"),
        "--amazon-profile-config",
        str(HERE / "config.authoritative_amazon_profile_instrumented.json"),
    ]
    for dataset in ("genome", "reviews", "variousimg"):
        command.extend([
            "--heldout-formal-config",
            str(HERE / f"config.authoritative_heldout_{dataset}_formal.json"),
        ])
    for dataset in ("genome", "reviews", "variousimg"):
        command.extend([
            "--heldout-policy", str(run_root / f"heldout/{dataset}/policy.json"),
        ])
    command.extend([
        "--build-quality-formal-config",
        str(HERE / "config.authoritative_amazon_build_quality_formal.json"),
        "--build-config",
        str(HERE / "config.authoritative_amazon_base_build_timing.json"),
        "--build-config",
        str(HERE / "config.authoritative_amazon_base_build_resource.json"),
        "--build-config",
        str(HERE / "config.authoritative_amazon_hierarchy_build_timing.json"),
        "--build-config",
        str(HERE / "config.authoritative_amazon_hierarchy_build_resource.json"),
        "--amazon-formal",
        str(run_root / "search/amazon_formal/summary/performance/equal_recall_conservative.csv"),
        "--amazon-depth",
        str(run_root / "search/amazon_formal/summary/performance/depth_ablation/depth_by_workload.csv"),
        "--amazon-depth-global",
        str(run_root / "search/amazon_formal/summary/performance/depth_ablation/depth_global.csv"),
        "--amazon-profile",
        str(run_root / "search/amazon_profile_instrumented/summary/profile/equal_recall_conservative.csv"),
        "--heldout-workload",
        str(results / "heldout_oracle/heldout_oracle_by_workload.csv"),
        "--heldout-global",
        str(results / "heldout_oracle/heldout_oracle_global.csv"),
        "--build-summary",
        str(results / "authoritative_build/build_summary.csv"),
        "--build-end-to-end",
        str(results / "authoritative_build/build_end_to_end.csv"),
    ])
    return command


def write_manifest(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", dir=path.parent, text=True)
    temporary_path = Path(temporary)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
            json.dump(payload, stream, indent=2)
            stream.write("\n")
        os.replace(temporary_path, path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--poll-seconds", type=float, default=60.0)
    parser.add_argument("--run-root", type=Path, default=RUN_ROOT)
    parser.add_argument("--results-dir", type=Path, default=RESULTS)
    parser.add_argument("--tectonic", type=Path, default=DEFAULT_TECTONIC)
    args = parser.parse_args()
    if args.poll_seconds <= 0:
        parser.error("--poll-seconds must be positive")
    run_root = args.run_root.resolve()
    results = args.results_dir.resolve()
    needed = required_artifacts(run_root, results)

    while not all(path.is_file() and path.stat().st_size > 0 for path in needed):
        time.sleep(args.poll_seconds)
    while active_experiment_processes():
        time.sleep(args.poll_seconds)

    run([sys.executable, str(HERE / "run_authoritative_instrumented_profile.py")])
    run(paper_generation_command(run_root, results))

    generated = PAPER_DIR / "generated_results.tex"
    if "\\pending" in generated.read_text(encoding="utf-8"):
        raise RuntimeError("generated paper results still contain pending markers")
    paper_output = run_root / "paper"
    paper_output.mkdir(parents=True, exist_ok=True)
    run([
        str(args.tectonic.resolve()), "-X", "compile", "--keep-logs",
        "--keep-intermediates", "--outdir", str(paper_output), "main.tex",
    ], cwd=PAPER_DIR)
    pdf = paper_output / "main.pdf"
    if not pdf.is_file() or pdf.stat().st_size <= 0:
        raise RuntimeError(f"final paper PDF is missing: {pdf}")
    run(["git", "diff", "--check"], cwd=REPO)
    write_manifest(run_root / "finalization_manifest.json", {
        "schema_version": 1,
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_commit": subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=REPO, check=True,
            text=True, stdout=subprocess.PIPE).stdout.strip(),
        "generated_results": str(generated),
        "generated_results_sha256": sha256(generated),
        "paper_pdf": str(pdf),
        "paper_pdf_sha256": sha256(pdf),
    })
    print(run_root / "finalization_manifest.json", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
