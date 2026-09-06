#!/usr/bin/env python3
"""Select fair per-layer oracle and shared-threshold configurations.

Every selected row is an actually measured point whose minimum repeat Recall
meets the workload threshold.  No interpolation is used.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path


def read_points(path: Path) -> list[dict]:
    with path.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    for row in rows:
        for key in ("layer_count", "lsearch"):
            row[key] = int(row[key])
        for key in ("t1", "t2"):
            row[key] = int(row[key]) if row[key] else None
        for key in ("recall", "recall_min", "batch_ms_warm",
                    "batch_ms_warm_median"):
            row[key] = float(row[key])
    return rows


def fastest_feasible(rows: list[dict], threshold: float) -> dict | None:
    feasible = [row for row in rows if row["recall_min"] >= threshold]
    return min(feasible, key=lambda row: (row["batch_ms_warm_median"],
                                          row["batch_ms_warm"], row["lsearch"])) \
        if feasible else None


def select_oracle(points: list[dict], thresholds: dict[str, float]) -> list[dict]:
    result = []
    workloads = sorted(thresholds)
    layers = sorted({row["layer_count"] for row in points})
    for layer in layers:
        for workload in workloads:
            selected = fastest_feasible(
                [row for row in points
                 if row["layer_count"] == layer and row["workload"] == workload],
                thresholds[workload],
            )
            if selected is not None:
                row = dict(selected)
                row["target_recall"] = thresholds[workload]
                row["selection_scope"] = "per_workload_oracle"
                result.append(row)
    return result


def select_shared(points: list[dict], thresholds: dict[str, float]) -> tuple[list[dict], list[dict]]:
    workloads = sorted(thresholds)
    configs = sorted({(row["layer_count"], row["method"], row["t1"], row["t2"])
                      for row in points})
    baseline = {}
    for workload in workloads:
        baseline[workload] = fastest_feasible(
            [row for row in points
             if row["layer_count"] == 0 and row["workload"] == workload],
            thresholds[workload],
        )
    winners = []
    summaries = []
    for layer in sorted({item[0] for item in configs}):
        candidates = []
        for config_key in [item for item in configs if item[0] == layer]:
            _, method, t1, t2 = config_key
            selected = []
            for workload in workloads:
                row = fastest_feasible(
                    [point for point in points
                     if point["method"] == method and point["workload"] == workload],
                    thresholds[workload],
                )
                if row is None:
                    break
                selected.append(row)
            if len(selected) != len(workloads) or any(baseline[w] is None for w in workloads):
                continue
            normalized = [row["batch_ms_warm_median"] /
                          baseline[row["workload"]]["batch_ms_warm_median"]
                          for row in selected]
            score = math.exp(sum(math.log(value) for value in normalized) / len(normalized))
            candidates.append((score, sum(row["batch_ms_warm_median"] for row in selected),
                               method, t1, t2, selected))
        if not candidates:
            continue
        score, total, method, t1, t2, selected = min(candidates)
        summaries.append({
            "layer_count": layer, "method": method, "t1": t1, "t2": t2,
            "geomean_latency_vs_layer0": score,
            "geomean_speedup_vs_layer0": 1.0 / score,
            "sum_batch_ms_warm_median": total,
        })
        for source in selected:
            row = dict(source)
            row["target_recall"] = thresholds[row["workload"]]
            row["selection_scope"] = "shared_thresholds"
            row["speedup_vs_layer0"] = (
                baseline[row["workload"]]["batch_ms_warm_median"] /
                row["batch_ms_warm_median"]
            )
            winners.append(row)
    return summaries, winners


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=Path)
    parser.add_argument("--points", type=Path)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    root = Path(config["output_root"]) / "summary"
    points = read_points(args.points or root / "all_points.csv")
    thresholds = {key: float(value) for key, value in config["recall_thresholds"].items()}
    oracle = select_oracle(points, thresholds)
    shared, selected = select_shared(points, thresholds)
    write_csv(root / "layer_oracle.csv", oracle)
    write_csv(root / "shared_configurations.csv", shared)
    write_csv(root / "shared_selected_points.csv", selected)
    print(f"selected {len(oracle)} oracle rows and {len(shared)} shared configurations")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
