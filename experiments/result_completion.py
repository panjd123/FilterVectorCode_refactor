#!/usr/bin/env python3

"""Shared checks for reusing completed search sweeps."""

import csv
import math
from pathlib import Path
from typing import Iterable, Mapping


def search_summary_complete(summary_path: Path, expected_values: Iterable[int]) -> bool:
    """Return true only when every configured search value has valid summary metrics."""
    expected = {int(value) for value in expected_values}
    if not expected or not summary_path.is_file() or summary_path.stat().st_size == 0:
        return False

    try:
        with summary_path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            fieldnames = set(reader.fieldnames or [])
            required_columns = {"Lsearch", "Average_Time_ms"}
            if not required_columns.issubset(fieldnames):
                return False
            recall_column = "Average_Recall" if "Average_Recall" in fieldnames else "Recall"
            if recall_column not in fieldnames:
                return False
            completed: set[int] = set()
            for row in reader:
                try:
                    raw_value = float(row["Lsearch"])
                    metrics = [float(row["Average_Time_ms"]), float(row[recall_column])]
                except (KeyError, TypeError, ValueError):
                    return False
                if not math.isfinite(raw_value) or not raw_value.is_integer():
                    return False
                if not all(math.isfinite(metric) for metric in metrics):
                    return False
                value = int(raw_value)
                if value in expected:
                    completed.add(value)
    except (OSError, csv.Error, UnicodeError):
        return False

    return completed == expected


def skip_completed_enabled(*scopes: Mapping | None) -> bool:
    """Resolve the opt-out flag from broad to narrow configuration scopes."""
    enabled = True
    for scope in scopes:
        if scope and "skip_completed_searches" in scope:
            enabled = bool(scope["skip_completed_searches"])
    return enabled


def should_skip_completed_search(
    summary_path: Path,
    expected_values: Iterable[int],
    *scopes: Mapping | None,
) -> bool:
    return skip_completed_enabled(*scopes) and search_summary_complete(summary_path, expected_values)


def describe_search_values(values: Iterable[int]) -> str:
    normalized = [int(value) for value in values]
    if not normalized:
        return "empty"
    if len(normalized) == 1:
        return str(normalized[0])
    return f"{normalized[0]}..{normalized[-1]} ({len(normalized)} values)"
