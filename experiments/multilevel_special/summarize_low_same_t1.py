#!/usr/bin/env python3
"""Create compact, fail-closed evidence for the low-selectivity same-T1 study."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import statistics
from pathlib import Path
from typing import Any


FIELDS = [
    "workload", "mean_selectivity", "recall_threshold", "method",
    "layer_count", "t1", "t2", "lsearch", "recall_mean",
    "recall_min", "recall_max", "warm_repeats", "warm_mean_ms",
    "warm_median_ms", "warm_stddev_ms", "warm_cv", "warm_min_ms",
    "warm_max_ms", "speedup_vs_layer1_mean",
    "speedup_vs_layer1_median", "stage_closure_max_abs_ms_per_query",
    "checked_results", "filter_violations",
]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def select_point(config: dict[str, Any], method: dict[str, Any],
                 workload: dict[str, Any]) -> dict[str, Any]:
    run_dir = Path(config["output_root"]) / method["name"] / workload["name"]
    detail = read_csv(run_dir / "search_time_details.csv")
    stages = read_csv(run_dir / "search_stage_details.csv")
    filters = read_csv(run_dir / "filter_validation.csv")
    expected_repeats = int(config["num_repeats"])
    expected_l = {int(value) for value in method["lsearch_values_by_workload"][workload["name"]]}
    by_l: dict[int, list[dict[str, str]]] = {}
    stage_by_l: dict[int, list[dict[str, str]]] = {}
    filter_by_l: dict[int, list[dict[str, str]]] = {}
    for row in detail:
        by_l.setdefault(int(row["Lsearch"]), []).append(row)
    for row in stages:
        stage_by_l.setdefault(int(row["Lsearch"]), []).append(row)
    for row in filters:
        filter_by_l.setdefault(int(row["Lsearch"]), []).append(row)
    if set(by_l) != expected_l or set(stage_by_l) != expected_l or set(filter_by_l) != expected_l:
        raise RuntimeError(f"incomplete L grid for {method['name']}/{workload['name']}")
    candidates = []
    for lsearch in sorted(expected_l):
        drows, srows, frows = by_l[lsearch], stage_by_l[lsearch], filter_by_l[lsearch]
        if not all(len(rows) == expected_repeats for rows in (drows, srows, frows)):
            raise RuntimeError(f"repeat mismatch for {method['name']}/{workload['name']}/L{lsearch}")
        warm = [float(row["Time_ms"]) for row in drows if int(row["Repeat"]) > 0]
        recalls = [float(row["Avg_Recall"]) for row in drows]
        closure = max(abs(float(row["ClosureError_ms"])) for row in srows)
        checked = sum(int(row["CheckedResults"]) for row in frows)
        violations = sum(int(row["FilterViolations"]) for row in frows)
        if len(warm) != expected_repeats - 1 or closure > 1e-6 or violations:
            raise RuntimeError(
                f"invalid evidence for {method['name']}/{workload['name']}/L{lsearch}: "
                f"warm={len(warm)} closure={closure} violations={violations}")
        mean = statistics.mean(warm)
        candidate = {
            "workload": workload["name"],
            "mean_selectivity": float(workload["mean_selectivity"]),
            "recall_threshold": float(config["recall_thresholds"][workload["name"]]),
            "method": method["name"], "layer_count": int(method["layer_count"]),
            "t1": method.get("t1") if method.get("t1") is not None else "",
            "t2": method.get("t2") if method.get("t2") is not None else "",
            "lsearch": lsearch, "recall_mean": statistics.mean(recalls),
            "recall_min": min(recalls), "recall_max": max(recalls),
            "warm_repeats": len(warm), "warm_mean_ms": mean,
            "warm_median_ms": statistics.median(warm),
            "warm_stddev_ms": statistics.stdev(warm),
            "warm_cv": statistics.stdev(warm) / mean,
            "warm_min_ms": min(warm), "warm_max_ms": max(warm),
            "speedup_vs_layer1_mean": "", "speedup_vs_layer1_median": "",
            "stage_closure_max_abs_ms_per_query": closure,
            "checked_results": checked, "filter_violations": violations,
        }
        candidates.append(candidate)
    passing = [row for row in candidates if row["recall_min"] >= row["recall_threshold"]]
    if not passing:
        maximum = max(row["recall_min"] for row in candidates)
        raise RuntimeError(
            f"no conservative Recall crossing for {method['name']}/{workload['name']}; "
            f"max recall_min={maximum}")
    return min(passing, key=lambda row: row["warm_mean_ms"])


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    manifest = json.loads((Path(config["output_root"]) / "manifest.json").read_text())
    expected_cases = len(config["methods"]) * len(config["workloads"])
    if len(manifest.get("runs", [])) != expected_cases:
        raise RuntimeError(f"manifest has {len(manifest.get('runs', []))}/{expected_cases} cases")
    if any(row.get("status") != "complete" or row.get("returncode") != 0
           for row in manifest["runs"]):
        raise RuntimeError("manifest contains incomplete cases")
    binary_hashes = {row["search_binary_sha256"] for row in manifest["runs"]}
    if len(binary_hashes) != 1:
        raise RuntimeError(f"mixed binaries: {binary_hashes}")

    selected = [select_point(config, method, workload)
                for workload in config["workloads"] for method in config["methods"]]
    for workload in config["workloads"]:
        subset = [row for row in selected if row["workload"] == workload["name"]]
        baseline = next(row for row in subset if row["method"] == "layer1_t1_1000")
        for row in subset:
            row["speedup_vs_layer1_mean"] = baseline["warm_mean_ms"] / row["warm_mean_ms"]
            row["speedup_vs_layer1_median"] = baseline["warm_median_ms"] / row["warm_median_ms"]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_dir / "low_same_t1_equal_recall.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS, lineterminator="\n")
        writer.writeheader(); writer.writerows(selected)

    source_root = Path(config["data_root"]); gt_root = Path(config["gt_root"])
    hash_rows = []
    for workload in config["workloads"]:
        query_root = source_root / workload["query_dir"]
        for kind, path in (
            ("query_vectors", query_root / "Amazon_query.bin"),
            ("query_labels", query_root / "Amazon_query_labels.txt"),
            ("exact_gt", gt_root / workload["query_dir"] / "Amazon_gt_labels_containment.bin"),
        ):
            hash_rows.append({"artifact": f"{workload['name']}:{kind}",
                              "path": str(path), "sha256": sha256(path)})
    hash_rows.append({"artifact": "search_binary",
                      "path": manifest["runs"][0]["source_search_app"],
                      "sha256": next(iter(binary_hashes))})
    for method in config["methods"]:
        index = Path(method.get("block_index", config["main_index"]))
        hash_rows.append({"artifact": f"{method['name']}:index_meta",
                          "path": str(index / "meta"), "sha256": sha256(index / "meta")})
    hash_path = args.output_dir / "low_same_t1_hashes.csv"
    with hash_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["artifact", "path", "sha256"], lineterminator="\n")
        writer.writeheader(); writer.writerows(hash_rows)
    print(f"wrote {csv_path} and {hash_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
