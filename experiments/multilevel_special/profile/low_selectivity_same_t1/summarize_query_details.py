#!/usr/bin/env python3
"""Summarize the low-selectivity profile cases without third-party packages."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path


FIELDS = (
    "Time_ms",
    "EntryGroupSearchTime_ms",
    "EntryPointSetupTime_ms",
    "BlockAuthorizationTime_ms",
    "GraphSearchTime_ms",
    "ResidualTime_ms",
    "Recall",
    "TotalDistanceCalcs",
    "EntryPointDistanceCalcs",
    "GraphSearchDistanceCalcs",
    "SpecialFreeBlockCount",
    "SpecialQueryMiddleBlockCount",
    "SpecialQueryUpperBlockCount",
    "SpecialQueryUpperCoveredPoints",
    "SpecialQueryUpperEnabled",
    "SpecialBlocksSearched",
    "SpecialMiddleBlocksSearched",
    "SpecialUpperBlocksSearched",
    "SpecialRegularNodesExpanded",
    "SpecialFreeNodesExpanded",
    "SpecialMiddleNodesExpanded",
    "SpecialUpperNodesExpanded",
    "SpecialRegularEdgesScanned",
    "SpecialFreeEdgesScanned",
    "SpecialIntraEdgesScanned",
    "SpecialInterEdgesScanned",
    "SpecialMiddleEdgesScanned",
    "SpecialUpperEdgesScanned",
    "SpecialMiddleActivations",
    "SpecialUpperActivations",
    "SpecialQueueInsertAttempts",
    "SpecialQueueBoundRejections",
    "SpecialQueueInsertions",
    "SpecialQueueShiftedCandidates",
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    parser.add_argument("cases", nargs="+", help="NAME=RUN_DIRECTORY")
    args = parser.parse_args()
    result = {}
    for item in args.cases:
        name, raw_path = item.split("=", 1)
        path = Path(raw_path)
        detail = next(path.glob("query_details_repeat*.csv"))
        with detail.open(newline="") as stream:
            rows = list(csv.DictReader(stream))
        case = {}
        for field in FIELDS:
            if field not in rows[0]:
                continue
            values = [float(row[field]) for row in rows]
            case[field] = {
                "mean": statistics.fmean(values),
                "median": statistics.median(values),
                "nonzero": sum(value != 0 for value in values),
                "max": max(values),
            }
        with (path / "search_time_details.csv").open(newline="") as stream:
            repeats = list(csv.DictReader(stream))
        warm = [float(row["Time_ms"]) for row in repeats[1:]]
        case["BatchWarm_ms"] = {
            "mean": statistics.fmean(warm),
            "median": statistics.median(warm),
            "min": min(warm),
            "max": max(warm),
            "n": len(warm),
        }
        violations_path = path / "filter_validation.csv"
        if violations_path.exists():
            with violations_path.open(newline="") as stream:
                validations = list(csv.DictReader(stream))
            case["FilterViolations"] = sum(
                int(row["FilterViolations"]) for row in validations
            )
        result[name] = case
    args.output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
