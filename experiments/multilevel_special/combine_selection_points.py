#!/usr/bin/env python3
"""Combine disjoint selection sweeps without choosing among duplicate runs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import summarize_selection_sweep


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("configs", nargs="+", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    rows = []
    seen = set()
    for config_path in args.configs:
        config = json.loads(config_path.read_text())
        for row in summarize_selection_sweep.read_rows(config):
            key = (row["workload"], row["method"], row["lsearch"])
            if key in seen:
                raise ValueError(f"duplicate measured point across sweeps: {key}")
            seen.add(key)
            rows.append(row)
    rows.sort(key=lambda row: (row["layer_count"], row["method"],
                               row["workload"], row["lsearch"]))
    summarize_selection_sweep.write_csv(args.output, rows)
    print(f"wrote {len(rows)} unique measured points to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
