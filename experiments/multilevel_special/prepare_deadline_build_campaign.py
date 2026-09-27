#!/usr/bin/env python3
"""Prepare the bounded Amazon construction campaign from frozen configs."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
DEFAULT_RUN_ROOT = REPO / "runs/deadline_build_20260927"
DEFAULT_MANIFEST = HERE / "config.deadline_amazon_build_manifest.json"

BASE_PROFILES = (
    "original_cpu",
    "current_cpu",
    "naive_gpu",
    "paper_fused",
    "accelerated_gpu",
)
HIERARCHY_STRUCTURE = "auto_drh_v1"
HIERARCHY_PROFILES = (
    "cpu",
    "hybrid_gpu_intra",
    "hybrid_gpu_intra_inter",
    "full_gpu",
    "full_gpu_wmma",
)

SPECS = (
    {
        "phase": "base_timing",
        "runner": "run_base_topology_build.py",
        "source": "config.authoritative_amazon_base_build_timing.json",
        "output": "config.deadline_amazon_base_build_timing.json",
        "profiles": BASE_PROFILES,
        "structure": None,
        "resource_profile": False,
    },
    {
        "phase": "hierarchy_timing",
        "runner": "run_build_sweep.py",
        "source": "config.authoritative_amazon_hierarchy_build_timing.json",
        "output": "config.deadline_amazon_hierarchy_build_timing.json",
        "profiles": HIERARCHY_PROFILES,
        "structure": HIERARCHY_STRUCTURE,
        "resource_profile": False,
    },
    {
        "phase": "base_resource",
        "runner": "run_base_topology_build.py",
        "source": "config.authoritative_amazon_base_build_resource.json",
        "output": "config.deadline_amazon_base_build_resource.json",
        "profiles": BASE_PROFILES,
        "structure": None,
        "resource_profile": True,
    },
    {
        "phase": "hierarchy_resource",
        "runner": "run_build_sweep.py",
        "source": "config.authoritative_amazon_hierarchy_build_resource.json",
        "output": "config.deadline_amazon_hierarchy_build_resource.json",
        "profiles": HIERARCHY_PROFILES,
        "structure": HIERARCHY_STRUCTURE,
        "resource_profile": True,
    },
)


def absolute_no_resolve(path: Path) -> Path:
    path = path.expanduser()
    return path if path.is_absolute() else Path.cwd() / path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", dir=path.parent, text=True)
    temporary_path = Path(temporary)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2)
            stream.write("\n")
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)


def timing_case_selected(case: dict[str, Any], measured_repeats: int) -> bool:
    role = str(case.get("timing_role", ""))
    repeat = int(case.get("repeat", -1))
    return (role == "cold" and repeat == 0) or (
        role == "measured" and 0 <= repeat < measured_repeats)


def select_cases(
    config: dict[str, Any], profiles: tuple[str, ...],
    structure: str | None, measured_repeats: int,
    resource_profile: bool,
) -> list[dict[str, Any]]:
    selected = []
    for source_case in config["cases"]:
        case = copy.deepcopy(source_case)
        profile = str(case.get("benchmark_profile", ""))
        if profile not in profiles:
            continue
        if structure is not None and not str(case["name"]).startswith(
                f"{structure}_{profile}_"):
            continue
        if resource_profile:
            if case.get("timing_role") != "measured" or int(
                    case.get("repeat", -1)) != 0:
                continue
        elif not timing_case_selected(case, measured_repeats):
            continue
        selected.append(case)
    return selected


def prepare_config(
    source: dict[str, Any], output_root: Path, profiles: tuple[str, ...],
    structure: str | None, measured_repeats: int, resource_profile: bool,
    source_config_sha256: str | None = None,
) -> dict[str, Any]:
    result = copy.deepcopy(source)
    result["output_root"] = str(output_root)
    result["purpose"] = (
        "Deadline-bounded Amazon construction evidence; generated from the "
        "frozen authoritative configuration."
    )
    result["campaign_protocol"] = {
        "evidence_tier": "deadline_screen",
        "cold_repeats": 0 if resource_profile else 1,
        "measured_repeats": 1 if resource_profile else measured_repeats,
        "resource_profile": resource_profile,
        "query_independent": True,
    }
    if source_config_sha256 is not None:
        result["campaign_protocol"]["source_config_sha256"] = source_config_sha256
    result["cases"] = select_cases(
        source, profiles, structure, measured_repeats, resource_profile)
    expected = len(profiles) * (1 if resource_profile else 1 + measured_repeats)
    if len(result["cases"]) != expected:
        raise ValueError(
            f"selected {len(result['cases'])} cases, expected {expected} for "
            f"profiles={profiles}, structure={structure}")
    names = [str(case["name"]) for case in result["cases"]]
    if len(names) != len(set(names)):
        raise ValueError("selected build case names are not unique")
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--measured-repeats", type=int, default=2)
    args = parser.parse_args()
    if args.measured_repeats <= 0:
        parser.error("--measured-repeats must be positive")
    run_root = absolute_no_resolve(args.run_root)
    stages = []
    for spec in SPECS:
        source_path = HERE / str(spec["source"])
        output_path = HERE / str(spec["output"])
        source = json.loads(source_path.read_text(encoding="utf-8"))
        config = prepare_config(
            source=source,
            output_root=run_root / "build_study" / str(spec["phase"]),
            profiles=spec["profiles"],
            structure=spec["structure"],
            measured_repeats=args.measured_repeats,
            resource_profile=bool(spec["resource_profile"]),
            source_config_sha256=sha256_file(source_path),
        )
        atomic_json(output_path, config)
        stages.append({
            "phase": spec["phase"],
            "runner": str(HERE / str(spec["runner"])),
            "config": str(output_path),
            "source_config": str(source_path),
            "source_config_sha256": sha256_file(source_path),
            "cases": [case["name"] for case in config["cases"]],
        })
    manifest = {
        "schema_version": 1,
        "purpose": "Bounded end-to-end Amazon construction evidence.",
        "run_root": str(run_root),
        "evidence_tier": "deadline_screen",
        "protocol": {
            "cold_repeats": 1,
            "measured_repeats": args.measured_repeats,
            "resource_repeats": 1,
            "case_timeout_seconds": 3300,
            "timing_statistic": "median of measured wall-clock repeats",
            "kernel_time_is_not_end_to_end_time": True,
        },
        "stages": stages,
        "summarizer": str(HERE / "summarize_authoritative_build.py"),
        "summary_output_dir": str(run_root / "build_study" / "summary"),
    }
    manifest_path = absolute_no_resolve(args.manifest)
    atomic_json(manifest_path, manifest)
    print(manifest_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
