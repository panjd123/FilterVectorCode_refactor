#!/usr/bin/env python3
"""Validate and compare the non-Amazon filtered Trie block-500 campaign."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path


KEY_L = (1000, 5000, 20000)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def mean(rows: list[dict[str, str]], column: str) -> float:
    return statistics.fmean(float(row[column]) for row in rows)


def percent_change(after: float, before: float) -> float:
    return 100.0 * (after / before - 1.0) if before else float("nan")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--validation-output", type=Path, required=True)
    args = parser.parse_args()

    config = json.loads(args.config.read_text())
    result_root = Path(config["result_root"])
    data_root = Path(config["data_root"])
    method = config["methods"][0]["name"]
    original_method = config["methods"][0]["index_name"]

    output_rows: list[dict[str, object]] = []
    validations: list[dict[str, object]] = []

    for item in config["datasets"]:
        dataset = item["dataset"]
        task = item["query_task"]
        band = "10to40" if "sel10to40pct" in task else "40to100"
        original_task = f"query_minlen1_sel{band}pct"
        suffix = "_100_100_20000"
        new_results = result_root / dataset / "results" / method / f"{task}{suffix}" / "results"
        old_results = result_root / dataset / "results" / original_method / f"{original_task}{suffix}" / "results"
        map_path = data_root / dataset / task / "query_id_map.csv"

        id_map = read_csv(map_path)
        retained = len(id_map)
        original = 700 if band == "10to40" else 300
        new_summary = read_csv(new_results / "search_time_summary.csv")
        new_qps = read_csv(new_results / "search_time_summary_qps.csv")
        new_details = read_csv(new_results / "query_details_repeat1.csv")
        old_summary = read_csv(old_results / "search_time_summary.csv")
        old_qps = read_csv(old_results / "search_time_summary_qps.csv")
        old_details = read_csv(old_results / "query_details_repeat1.csv")

        validations.append(
            {
                "dataset": dataset,
                "band": band,
                "retained": retained,
                "excluded": original - retained,
                "summary_rows": len(new_summary),
                "qps_rows": len(new_qps),
                "detail_rows": len(new_details),
                "expected_detail_rows": retained * 29,
                "valid": len(new_summary) == 29
                and len(new_qps) == 29
                and len(new_details) == retained * 29,
            }
        )

        old_summary_by_l = {int(float(row["Lsearch"])): row for row in old_summary}
        new_summary_by_l = {int(float(row["Lsearch"])): row for row in new_summary}
        old_qps_by_l = {int(float(row["Lsearch"])): row for row in old_qps}
        new_qps_by_l = {int(float(row["Lsearch"])): row for row in new_qps}
        old_detail_by_l: dict[int, dict[int, dict[str, str]]] = {}
        new_detail_by_l: dict[int, dict[int, dict[str, str]]] = {}
        for row in old_details:
            old_detail_by_l.setdefault(int(float(row["Lsearch"])), {})[int(row["QueryID"])] = row
        for row in new_details:
            new_detail_by_l.setdefault(int(float(row["Lsearch"])), {})[int(row["QueryID"])] = row

        for lsearch in KEY_L:
            old_retained_rows = [
                old_detail_by_l[lsearch][int(pair["OriginalQueryID"])] for pair in id_map
            ]
            new_retained_rows = [
                new_detail_by_l[lsearch][int(pair["QueryID"])] for pair in id_map
            ]
            old_all = old_summary_by_l[lsearch]
            new_all = new_summary_by_l[lsearch]
            old_retained_time = mean(old_retained_rows, "Time_ms")
            new_retained_time = mean(new_retained_rows, "Time_ms")
            old_retained_recall = mean(old_retained_rows, "Recall")
            new_retained_recall = mean(new_retained_rows, "Recall")
            output_rows.append(
                {
                    "Dataset": dataset,
                    "Selectivity": band,
                    "OriginalQueries": original,
                    "RetainedQueries": retained,
                    "ExcludedQueries": original - retained,
                    "RetainedPct": 100.0 * retained / original,
                    "Lsearch": lsearch,
                    "OldAllRecall": float(old_all["Average_Recall"]),
                    "OldRetainedRecall": old_retained_recall,
                    "NewRetainedRecall": new_retained_recall,
                    "RecallDeltaVsOldAll": new_retained_recall - float(old_all["Average_Recall"]),
                    "RecallDeltaSameQueries": new_retained_recall - old_retained_recall,
                    "OldRetainedMeanQueryTime_ms": old_retained_time,
                    "NewRetainedMeanQueryTime_ms": new_retained_time,
                    "SameQueryTimeDeltaPct": percent_change(new_retained_time, old_retained_time),
                    "OldBatchTime_ms": float(old_all["Average_Time_ms"]),
                    "NewBatchTime_ms": float(new_all["Average_Time_ms"]),
                    "OldQPS": float(old_qps_by_l[lsearch]["QPS"]),
                    "NewQPS": float(new_qps_by_l[lsearch]["QPS"]),
                    "QPSDeltaPct": percent_change(
                        float(new_qps_by_l[lsearch]["QPS"]), float(old_qps_by_l[lsearch]["QPS"])
                    ),
                }
            )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(output_rows[0]))
        writer.writeheader()
        writer.writerows(output_rows)
    args.validation_output.write_text(json.dumps(validations, indent=2) + "\n")

    print(f"comparison={args.output}")
    print(f"validation={args.validation_output}")
    for row in validations:
        print(
            f"{row['dataset']:8s} {row['band']:7s} retained={row['retained']:3d} "
            f"summary={row['summary_rows']:2d} qps={row['qps_rows']:2d} "
            f"details={row['detail_rows']:5d}/{row['expected_detail_rows']:5d} valid={row['valid']}"
        )


if __name__ == "__main__":
    main()
