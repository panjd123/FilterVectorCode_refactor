#!/usr/bin/env python3
"""Derive a reproducible detailed-profile subset from a performance sweep."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from prepare_deadline_profile_campaign import (
    atomic_json,
    make_profile_config,
    sha256_file,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-config", type=Path, required=True)
    parser.add_argument("--search-binary", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--output-config", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--method", action="append", required=True)
    parser.add_argument("--workload", action="append", required=True)
    args = parser.parse_args()

    source_path = args.source_config.resolve()
    binary = args.search_binary.resolve()
    if not source_path.is_file():
        raise FileNotFoundError(source_path)
    if not binary.is_file():
        raise FileNotFoundError(binary)
    source = json.loads(source_path.read_text(encoding="utf-8"))
    binary_hash = sha256_file(binary)
    profile, cases = make_profile_config(
        source,
        source_path,
        binary,
        binary_hash,
        args.source_commit,
        args.output_root.resolve(),
        selected_method_names=set(args.method),
        selected_workload_names=set(args.workload),
    )
    output_config = args.output_config.resolve()
    atomic_json(output_config, profile)
    manifest = {
        "schema_version": 1,
        "purpose": "Selected detailed counters at measured performance operating points.",
        "source_config": str(source_path),
        "profile_config": str(output_config),
        "output_root": str(args.output_root.resolve()),
        "search_binary": str(binary),
        "search_binary_sha256": binary_hash,
        "source_commit": args.source_commit,
        "protocol": profile["protocol"],
        "cases": cases,
    }
    atomic_json(args.manifest.resolve(), manifest)
    print(output_config)
    print(args.manifest.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
