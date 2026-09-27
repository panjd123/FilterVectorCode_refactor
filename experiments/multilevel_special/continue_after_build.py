#!/usr/bin/env python3
"""Finalize the authoritative study after held-out and build work completes."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
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
QUERY_CONFIG_INPUT_FLAGS = (
    "--amazon-formal-config", "--amazon-profile-config",
    "--heldout-formal-config", "--build-quality-formal-config",
)
BUILD_CONFIG_INPUT_FLAGS = ("--build-config",)
CONFIG_INPUT_FLAGS = QUERY_CONFIG_INPUT_FLAGS + BUILD_CONFIG_INPUT_FLAGS
RESULT_INPUT_FLAGS = (
    "--amazon-formal", "--amazon-depth", "--amazon-depth-global",
    "--amazon-profile", "--heldout-workload", "--heldout-global",
    "--build-summary", "--build-end-to-end",
)
PROVENANCE_PATTERN = re.compile(
    r"^% (source-sha256|config-sha256|heldout-policy-sha256|"
    r"validator-sha256|manifest-sha256) "
    r"(\S+) ([0-9a-f]{64})$")
TECTONIC_FAILURE_PATTERNS = (
    re.compile(r"^!", re.MULTILINE),
    re.compile(
        r"\bundefined (?:citation|citations|reference|references|control sequence)\b",
        re.IGNORECASE),
    re.compile(
        r"\b(?:citation|reference)\b[^\n]{0,200}\bundefined\b",
        re.IGNORECASE),
    re.compile(r"\bfatal error\b", re.IGNORECASE),
    re.compile(r"\bemergency stop\b", re.IGNORECASE),
    re.compile(r"\bno output pdf file produced\b", re.IGNORECASE),
    re.compile(r"^warning--", re.IGNORECASE | re.MULTILINE),
    re.compile(r"\bi couldn't open (?:database|style) file\b", re.IGNORECASE),
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
        "--validator", str(HERE / "validate_selection_sweep.py"),
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


def flag_values(command: list[str], flags: tuple[str, ...]) -> list[tuple[str, Path]]:
    """Return every path-valued occurrence of the requested command flags."""
    values = []
    for index, token in enumerate(command):
        if token in flags:
            if index + 1 >= len(command):
                raise ValueError(f"missing value after {token}")
            values.append((token, Path(command[index + 1]).resolve()))
    return values


def paper_input_snapshot(command: list[str]) -> list[dict[str, str]]:
    """Hash every direct evidence/configuration input consumed by the generator."""
    inputs: list[tuple[str, str, Path]] = []
    for _, path in flag_values(command, CONFIG_INPUT_FLAGS):
        inputs.append(("config-sha256", path.name, path))
    for _, path in flag_values(command, RESULT_INPUT_FLAGS):
        inputs.append(("source-sha256", path.name, path))
    for _, path in flag_values(command, ("--validator",)):
        inputs.append(("validator-sha256", path.name, path))
    for _, path in flag_values(command, ("--heldout-policy",)):
        policy = json.loads(path.read_text(encoding="utf-8"))
        dataset = policy.get("dataset")
        if not isinstance(dataset, str) or not dataset:
            raise ValueError(f"held-out policy lacks dataset: {path}")
        inputs.append(("heldout-policy-sha256", dataset, path))
    for _, path in flag_values(command, QUERY_CONFIG_INPUT_FLAGS):
        config = json.loads(path.read_text(encoding="utf-8"))
        root = Path(config["output_root"])
        if not root.is_absolute():
            root = HERE / root
        pass_name = str(config.get("measurement_pass", "performance"))
        manifest = root / (f"manifest_{pass_name}.json"
                           if config.get("pass_subdirs", False)
                           else "manifest.json")
        inputs.append(("manifest-sha256", path.name, manifest.resolve()))
    for _, path in flag_values(command, BUILD_CONFIG_INPUT_FLAGS):
        config = json.loads(path.read_text(encoding="utf-8"))
        root = Path(config["output_root"])
        if not root.is_absolute():
            root = HERE / root
        inputs.append((
            "manifest-sha256", path.name, (root / "manifest.json").resolve()))

    snapshot = []
    identities = set()
    for kind, label, path in inputs:
        identity = (kind, label)
        if identity in identities:
            raise ValueError(f"duplicate paper input provenance identity: {identity}")
        identities.add(identity)
        if not path.is_file() or path.stat().st_size <= 0:
            raise FileNotFoundError(f"paper input is missing or empty: {path}")
        snapshot.append({
            "kind": kind, "label": label, "path": str(path),
            "sha256": sha256(path),
        })
    return snapshot


def validate_generated_provenance(
    generated: Path, snapshot: list[dict[str, str]],
) -> None:
    """Require generated LaTeX provenance to match every current direct input."""
    observed: dict[tuple[str, str], str] = {}
    for line in generated.read_text(encoding="utf-8").splitlines():
        match = PROVENANCE_PATTERN.fullmatch(line)
        if not match:
            continue
        identity = (match.group(1), match.group(2))
        if identity in observed:
            raise RuntimeError(f"duplicate generated provenance: {identity}")
        observed[identity] = match.group(3)
    expected = {
        (row["kind"], row["label"]): row["sha256"] for row in snapshot
    }
    if observed != expected:
        missing = sorted(set(expected) - set(observed))
        unexpected = sorted(set(observed) - set(expected))
        stale = sorted(
            identity for identity in set(expected) & set(observed)
            if expected[identity] != observed[identity])
        raise RuntimeError(
            "generated paper provenance mismatch: "
            f"missing={missing}, unexpected={unexpected}, stale={stale}")


def verify_input_snapshot(snapshot: list[dict[str, str]]) -> None:
    """Close the generation/compilation time-of-check-to-time-of-use window."""
    changed = []
    for row in snapshot:
        path = Path(row["path"])
        if not path.is_file() or sha256(path) != row["sha256"]:
            changed.append(str(path))
    if changed:
        raise RuntimeError(f"paper inputs changed during finalization: {changed}")


def validate_tectonic_logs(output_dir: Path) -> list[dict[str, str]]:
    """Reject incomplete references, BibTeX warnings, and TeX fatal diagnostics."""
    records = []
    for name in ("main.log", "main.blg"):
        path = output_dir / name
        if not path.is_file() or path.stat().st_size <= 0:
            raise RuntimeError(f"required Tectonic log is missing or empty: {path}")
        contents = path.read_text(encoding="utf-8", errors="replace")
        matches = [pattern.pattern for pattern in TECTONIC_FAILURE_PATTERNS
                   if pattern.search(contents)]
        if matches:
            raise RuntimeError(
                f"Tectonic log failed publication checks: {path}: {matches}")
        records.append({"path": str(path.resolve()), "sha256": sha256(path)})
    return records


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
    generation_command = paper_generation_command(run_root, results)
    run(generation_command)

    generated = PAPER_DIR / "generated_results.tex"
    if "\\pending" in generated.read_text(encoding="utf-8"):
        raise RuntimeError("generated paper results still contain pending markers")
    paper_inputs = paper_input_snapshot(generation_command)
    validate_generated_provenance(generated, paper_inputs)
    paper_output = run_root / "paper"
    paper_output.mkdir(parents=True, exist_ok=True)
    paper_sources = [{
        "path": str(path.resolve()), "sha256": sha256(path),
    } for path in (
        PAPER_DIR / "main.tex", PAPER_DIR / "references.bib", generated)]
    tectonic = args.tectonic.resolve()
    tectonic_sha256 = sha256(tectonic)
    run([
        str(tectonic), "-X", "compile", "--keep-logs",
        "--keep-intermediates", "--outdir", str(paper_output), "main.tex",
    ], cwd=PAPER_DIR)
    pdf = paper_output / "main.pdf"
    if not pdf.is_file() or pdf.stat().st_size <= 0:
        raise RuntimeError(f"final paper PDF is missing: {pdf}")
    tectonic_logs = validate_tectonic_logs(paper_output)
    verify_input_snapshot(paper_inputs)
    verify_input_snapshot(paper_sources)
    if sha256(tectonic) != tectonic_sha256:
        raise RuntimeError("Tectonic binary changed during finalization")
    run(["git", "diff", "--check"], cwd=REPO)
    write_manifest(run_root / "finalization_manifest.json", {
        "schema_version": 2,
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_commit": subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=REPO, check=True,
            text=True, stdout=subprocess.PIPE).stdout.strip(),
        "generated_results": str(generated),
        "generated_results_sha256": sha256(generated),
        "paper_inputs": paper_inputs,
        "paper_generator_sha256": sha256(
            HERE / "generate_authoritative_paper_results.py"),
        "paper_sources": paper_sources,
        "tectonic": str(tectonic),
        "tectonic_sha256": tectonic_sha256,
        "tectonic_logs": tectonic_logs,
        "paper_pdf": str(pdf),
        "paper_pdf_sha256": sha256(pdf),
    })
    print(run_root / "finalization_manifest.json", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
