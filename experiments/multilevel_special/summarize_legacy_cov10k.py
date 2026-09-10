#!/usr/bin/env python3
"""Build compact, conservative evidence for the legacy Amazon cov10k rerun."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import statistics
from pathlib import Path
from typing import Any


FIELDS = [
    "workload", "query_dir", "mean_selectivity", "target_recall",
    "method", "layer_count", "t1", "t2", "lsearch",
    "recall_mean", "recall_min", "recall_max", "warm_repeats",
    "warm_mean_ms", "warm_median_ms", "warm_stddev_ms", "warm_cv",
    "speedup_vs_plain_mean", "speedup_vs_fixed_t1_layer1_mean",
    "stage_closure_max_abs_ms_per_query",
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
                 workload: dict[str, Any], target: float) -> dict[str, Any]:
    run_dir = Path(config["output_root"]) / method["name"] / workload["name"]
    details = read_csv(run_dir / "search_time_details.csv")
    stages = read_csv(run_dir / "search_stage_details.csv")
    expected_repeats = int(config["num_repeats"])
    expected_l = {int(value) for value in method["lsearch_values"]}
    candidates: list[dict[str, Any]] = []
    for lsearch in sorted(expected_l):
        drows = [row for row in details if int(row["Lsearch"]) == lsearch]
        srows = [row for row in stages if int(row["Lsearch"]) == lsearch]
        if len(drows) != expected_repeats or len(srows) != expected_repeats:
            raise RuntimeError(f"repeat mismatch: {method['name']}/L{lsearch}")
        warm = [float(row["Time_ms"]) for row in drows if int(row["Repeat"]) > 0]
        recalls = [float(row["Avg_Recall"]) for row in drows]
        closure = max(abs(float(row["ClosureError_ms"])) for row in srows)
        if len(warm) != expected_repeats - 1 or closure > 1e-6:
            raise RuntimeError(f"invalid evidence: {method['name']}/L{lsearch}")
        mean = statistics.mean(warm)
        candidates.append({
            "workload": workload["name"],
            "query_dir": workload["query_dir"],
            "mean_selectivity": float(workload["mean_selectivity"]),
            "target_recall": target,
            "method": method["name"],
            "layer_count": int(method["layer_count"]),
            "t1": method.get("t1") if method.get("t1") is not None else "",
            "t2": method.get("t2") if method.get("t2") is not None else "",
            "lsearch": lsearch,
            "recall_mean": statistics.mean(recalls),
            "recall_min": min(recalls),
            "recall_max": max(recalls),
            "warm_repeats": len(warm),
            "warm_mean_ms": mean,
            "warm_median_ms": statistics.median(warm),
            "warm_stddev_ms": statistics.stdev(warm),
            "warm_cv": statistics.stdev(warm) / mean,
            "speedup_vs_plain_mean": "",
            "speedup_vs_fixed_t1_layer1_mean": "",
            "stage_closure_max_abs_ms_per_query": closure,
        })
    passing = [row for row in candidates if row["recall_min"] >= target]
    if not passing:
        raise RuntimeError(f"no conservative crossing: {method['name']}")
    return min(passing, key=lambda row: row["warm_mean_ms"])


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--target", type=float, default=0.90)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    root = Path(config["output_root"])
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected_cases = len(config["methods"]) * len(config["workloads"])
    runs = manifest.get("runs", [])
    if len(runs) != expected_cases or any(
            row.get("status") != "complete" or row.get("returncode") != 0
            for row in runs):
        raise RuntimeError(f"manifest is not complete: {len(runs)}/{expected_cases}")
    binary_hashes = {row["search_binary_sha256"] for row in runs}
    if len(binary_hashes) != 1:
        raise RuntimeError(f"mixed search binaries: {binary_hashes}")

    selected = [
        select_point(config, method, workload, args.target)
        for workload in config["workloads"] for method in config["methods"]
    ]
    for workload in config["workloads"]:
        subset = [row for row in selected if row["workload"] == workload["name"]]
        plain = next(row for row in subset if row["method"] == "layer0_plain")
        fixed = next(row for row in subset if row["method"] == "layer1_t1_1000")
        for row in subset:
            row["speedup_vs_plain_mean"] = plain["warm_mean_ms"] / row["warm_mean_ms"]
            row["speedup_vs_fixed_t1_layer1_mean"] = fixed["warm_mean_ms"] / row["warm_mean_ms"]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    result_path = args.output_dir / "legacy_cov10k_equal_recall.csv"
    with result_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(selected)

    workload = config["workloads"][0]
    query_root = Path(config["data_root"]) / workload["query_dir"]
    gt_path = Path(config["gt_root"]) / workload["query_dir"] / "Amazon_gt_labels_containment.bin"
    manifest_snapshot = args.output_dir / "legacy_cov10k_execution_manifest.json"
    shutil.copyfile(manifest_path, manifest_snapshot)
    hash_rows = [
        {"artifact": "config", "digest_kind": "sha256", "path": str(args.config), "digest": sha256(args.config)},
        {"artifact": "execution_manifest", "digest_kind": "sha256", "path": str(manifest_snapshot), "digest": sha256(manifest_snapshot)},
        {"artifact": "query_vectors", "digest_kind": "sha256", "path": str(query_root / "Amazon_query.bin"), "digest": sha256(query_root / "Amazon_query.bin")},
        {"artifact": "query_labels", "digest_kind": "sha256", "path": str(query_root / "Amazon_query_labels.txt"), "digest": sha256(query_root / "Amazon_query_labels.txt")},
        {"artifact": "exact_gt", "digest_kind": "sha256", "path": str(gt_path), "digest": sha256(gt_path)},
        {"artifact": "search_binary", "digest_kind": "sha256", "path": runs[0]["source_search_app"], "digest": next(iter(binary_hashes))},
    ]
    for name, digest in runs[0]["provenance"].items():
        kind = "sha256" if name.endswith("sha256") else "source_fingerprint"
        hash_rows.append({"artifact": name, "digest_kind": kind,
                          "path": "manifest provenance", "digest": digest})
    hash_path = args.output_dir / "legacy_cov10k_hashes.csv"
    with hash_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=["artifact", "digest_kind", "path", "digest"],
            lineterminator="\n")
        writer.writeheader()
        writer.writerows(hash_rows)
    print(f"wrote {result_path} and {hash_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
