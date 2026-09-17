#!/usr/bin/env python3
"""Freeze external-dataset Recall crossings into seven-repeat runs."""

from __future__ import annotations

import csv
import json
from pathlib import Path


REPO = Path("/home/sunyahui/worktrees/FilterVectorCode_multilevel_special")
DATASETS = ("genome", "reviews", "variousimg")


def feasible_values(path: Path, threshold: float) -> list[int]:
    with path.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    grouped: dict[int, list[dict[str, str]]] = {}
    for row in rows:
        grouped.setdefault(int(row["Lsearch"]), []).append(row)
    result = []
    for value in sorted(grouped):
        repeats = grouped[value]
        if len(repeats) != 3:
            raise ValueError(f"incomplete screen point: {path} L={value}")
        if min(float(row["Avg_Recall"]) for row in repeats) >= threshold:
            result.append(value)
    if not result:
        raise ValueError(f"no measured Recall crossing in {path}")
    return result


def formal_quality_checked_value(
    screen_path: Path, formal_path: Path, threshold: float
) -> tuple[int, bool]:
    """Use the first screen crossing, advancing only after a formal failure."""
    candidates = feasible_values(screen_path, threshold)
    selected = candidates[0]
    if not formal_path.exists():
        return selected, False
    with formal_path.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != 7:
        return selected, False
    measured = {int(row["Lsearch"]) for row in rows}
    if len(measured) != 1:
        return selected, False
    measured_l = measured.pop()
    if measured_l >= selected and min(float(row["Avg_Recall"]) for row in rows) >= threshold:
        return measured_l, measured_l != selected
    if measured_l == selected:
        later = [value for value in candidates if value > selected]
        if not later:
            raise ValueError(f"formal Recall failed and no higher crossing exists: {formal_path}")
        return later[0], True
    return selected, False


def main() -> None:
    for dataset in DATASETS:
        source = REPO / f"experiments/multilevel_special/config.auto_policy_cross_dataset_{dataset}_crossing.json"
        config = json.loads(source.read_text())
        root = Path(config["output_root"])
        formal_root = root.parent / "formal"
        quality_advances = {}
        for method in config["methods"]:
            values = {}
            for workload in config["workloads"]:
                name = workload["name"]
                value, advanced = formal_quality_checked_value(
                    root / method["name"] / name / "search_time_details.csv",
                    formal_root / method["name"] / name / "search_time_details.csv",
                    .90,
                )
                values[name] = [value]
                if advanced:
                    quality_advances[f"{method['name']}/{name}"] = value
            method.pop("lsearch_values", None)
            method["lsearch_values_by_workload"] = values
        config["num_repeats"] = 7
        config["recall_thresholds"] = {w["name"]: .90 for w in config["workloads"]}
        config["output_root"] = str(formal_root)
        config["formal_protocol"] = {
            "selection_source": str(root),
            "selection_rule": "smallest measured L whose three screen repeats all reach Recall@10 >= 0.90",
            "timing_rule": "repeat 0 cold, repeats 1-6 warm",
            "formal_quality_advances": quality_advances,
        }
        output = source.with_name(source.name.replace("_crossing.json", "_formal.json"))
        output.write_text(json.dumps(config, indent=2) + "\n")
        print(output)


if __name__ == "__main__":
    main()
