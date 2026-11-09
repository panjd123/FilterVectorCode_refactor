#!/usr/bin/env python3
"""Prepare non-overwriting filtered query tasks from Trie per-query results."""

from __future__ import annotations

import argparse
import csv
import subprocess
import sys
from pathlib import Path


DATASETS = ("Genome", "Laion", "Music", "Tiktok", "VImg", "Reviews")
BANDS = ("sel10to40pct", "sel40to100pct")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--workspace-root", type=Path, default=Path("/home/dev/graphdb")
    )
    return parser.parse_args()


def percentile_nearest(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    return ordered[round((len(ordered) - 1) * fraction)]


def main() -> int:
    args = parse_args()
    workspace = args.workspace_root.resolve()
    data_root = workspace / "FilterVectorData"
    result_root = workspace / "FilterVectorResult"
    filter_script = Path(__file__).with_name("filter_query_task.py")

    for dataset in DATASETS:
        for band in BANDS:
            task = f"query_minlen1_{band}"
            details_path = (
                result_root
                / dataset
                / "results"
                / "Trie_block_500_M_32_cross_block_4"
                / f"{task}_100_100_20000"
                / "results"
                / "query_details_repeat1.csv"
            )
            with details_path.open(newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            l1000 = [row for row in rows if int(row["Lsearch"]) == 1000]
            l5000 = [row for row in rows if int(row["Lsearch"]) == 5000]
            if not l1000 or len(l1000) != len(l5000):
                raise ValueError(f"incomplete L=1000/L=5000 rows: {details_path}")

            p90_l1000 = percentile_nearest(
                [float(row["Time_ms"]) for row in l1000], 0.90
            )
            p75_l5000 = percentile_nearest(
                [float(row["Time_ms"]) for row in l5000], 0.75
            )
            bad_l1000 = {
                int(row["QueryID"])
                for row in l1000
                if float(row["Recall"]) <= 0.5
                or float(row["Time_ms"]) >= p90_l1000
            }
            bad_l5000 = {
                int(row["QueryID"])
                for row in l5000
                if float(row["Recall"]) < 0.9
                and float(row["Time_ms"]) >= p75_l5000
            }
            excluded = sorted(bad_l1000 | bad_l5000)
            destination_task = (
                f"{task}_exclude_trie_bad_l5000_l1000_r05_p90_{len(excluded)}"
            )
            source_dir = data_root / dataset / task
            destination_dir = data_root / dataset / destination_task
            if destination_dir.exists():
                raise FileExistsError(
                    f"refusing to overwrite existing task: {destination_dir}"
                )
            reason = (
                "Union of Lsearch=5000 slow/low-recall queries "
                f"(Recall < 0.9 and Time_ms >= P75={p75_l5000:.6f}) and "
                "Lsearch=1000 anomalies "
                f"(Recall <= 0.5 or Time_ms >= P90={p90_l1000:.6f})"
            )
            subprocess.run(
                [
                    sys.executable,
                    str(filter_script),
                    "--source-dir",
                    str(source_dir),
                    "--destination-dir",
                    str(destination_dir),
                    "--dataset",
                    dataset,
                    "--exclude-ids",
                    ",".join(str(query_id) for query_id in excluded),
                    "--reason",
                    reason,
                ],
                check=True,
            )
            print(
                f"{dataset},{band},source={len(l1000)},excluded={len(excluded)},"
                f"retained={len(l1000) - len(excluded)},task={destination_task}"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
