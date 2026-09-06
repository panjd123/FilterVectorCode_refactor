#!/usr/bin/env python3
"""Compare fresh current-source operating points with the frozen paper table.

The fresh runs are deliberately kept outside the frozen paper evidence. This
tool reads their manifests and warm-repeat details, checks that all six
workloads and four internal methods are present under one binary hash, and
writes a compact drift audit without changing the paper result selection.
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path


WORKLOAD_SUFFIXES = ("0p5", "1", "10", "25", "50", "75")
FRESH_CONFIGS = {
    "0p5": "config.amazon_x1_current_source_main_sel0p5_repeat21.json",
    **{suffix: f"config.amazon_x1_current_source_main_sel{suffix}.json"
       for suffix in WORKLOAD_SUFFIXES[1:]},
}
METHOD_NAMES = {
    "Plain UNG": "plain",
    "Single-level": "single_1k",
    "Tuned Multi-level": "multi_t1_2000_t2_25k",
}
FIELDS = (
    "workload", "method", "variant", "budget", "frozen_recall",
    "fresh_recall_mean", "recall_delta", "frozen_median_ms",
    "fresh_median_ms", "fresh_over_frozen", "fresh_min_ms",
    "fresh_max_ms", "fresh_cv", "measured_repeats",
    "fresh_binary_sha256", "raw_run_dir",
)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def fresh_method_name(row: dict[str, str]) -> str:
    if row["method"] in METHOD_NAMES:
        return METHOD_NAMES[row["method"]]
    if row["method"] != "Original Multi-level":
        raise ValueError(f"unknown frozen method: {row['method']}")
    try:
        threshold = row["variant"].split("T2=", 1)[1]
    except IndexError as error:
        raise ValueError(f"missing T2 in original-multilevel variant: {row['variant']}") from error
    return f"multi_1k_{threshold}"


def load_fresh_configs(config_dir: Path) -> dict[str, dict[str, object]]:
    configs = {}
    for suffix, filename in FRESH_CONFIGS.items():
        path = config_dir / filename
        configs[suffix] = json.loads(path.read_text(encoding="utf-8"))
    return configs


def audit(paper_results: Path, config_dir: Path) -> list[dict[str, object]]:
    frozen = [row for row in read_csv(paper_results) if row["comparison"] == "internal"]
    if len(frozen) != 24:
        raise RuntimeError(f"expected 24 frozen internal points, found {len(frozen)}")
    output: list[dict[str, object]] = []
    hashes: set[str] = set()
    configs = load_fresh_configs(config_dir)

    for row in frozen:
        suffix = row["workload"].removeprefix("sel_")
        if suffix not in WORKLOAD_SUFFIXES:
            raise RuntimeError(f"unexpected workload: {row['workload']}")
        config = configs[suffix]
        run_root = Path(str(config["output_root"]))
        manifest_path = run_root / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        method = fresh_method_name(row)
        records = [
            record for record in manifest.get("runs", [])
            if record.get("method") == method and record.get("workload") == row["workload"]
        ]
        if len(records) != 1 or records[0].get("status") != "complete":
            raise RuntimeError(f"missing complete fresh manifest record: {row['workload']}/{method}")
        record = records[0]
        hashes.add(str(record["search_binary_sha256"]))
        expected_budget = int(str(row["budget"]).removeprefix("L"))
        if [int(value) for value in record["lsearch_values"]] != [expected_budget]:
            raise RuntimeError(f"fresh budget mismatch: {row['workload']}/{method}")

        detail_path = run_root / method / row["workload"] / "search_time_details.csv"
        details = read_csv(detail_path)
        expected_repeats = int(record["num_repeats"])
        if len(details) != expected_repeats:
            raise RuntimeError(f"repeat count mismatch: {detail_path}")
        if {int(item["Lsearch"]) for item in details} != {expected_budget}:
            raise RuntimeError(f"detail budget mismatch: {detail_path}")
        warm = [float(item["Time_ms"]) for item in details if int(item["Repeat"]) > 0]
        recalls = [float(item["Avg_Recall"]) for item in details]
        fresh_median = statistics.median(warm)
        fresh_mean = statistics.fmean(warm)
        fresh_cv = statistics.stdev(warm) / fresh_mean if len(warm) > 1 else 0.0
        frozen_recall = float(row["recall"])
        frozen_ms = float(row["batch_median_ms"])
        output.append({
            "workload": row["workload"], "method": row["method"],
            "variant": row["variant"], "budget": row["budget"],
            "frozen_recall": f"{frozen_recall:.6f}",
            "fresh_recall_mean": f"{statistics.fmean(recalls):.6f}",
            "recall_delta": f"{statistics.fmean(recalls) - frozen_recall:.6f}",
            "frozen_median_ms": f"{frozen_ms:.6f}",
            "fresh_median_ms": f"{fresh_median:.6f}",
            "fresh_over_frozen": f"{fresh_median / frozen_ms:.6f}",
            "fresh_min_ms": f"{min(warm):.6f}",
            "fresh_max_ms": f"{max(warm):.6f}",
            "fresh_cv": f"{fresh_cv:.6f}",
            "measured_repeats": len(warm),
            "fresh_binary_sha256": record["search_binary_sha256"],
            "raw_run_dir": str(run_root / method / row["workload"]),
        })

    if len(hashes) != 1:
        raise RuntimeError(f"fresh runs use mixed binaries: {sorted(hashes)}")
    output.sort(key=lambda item: (WORKLOAD_SUFFIXES.index(str(item["workload"]).removeprefix("sel_")), str(item["method"])))
    return output


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    here = Path(__file__).resolve().parent
    parser.add_argument("--paper-results", type=Path,
                        default=here / "results_summary/paper_results.csv")
    parser.add_argument("--config-dir", type=Path, default=here)
    parser.add_argument("--output", type=Path,
                        default=here / "results_summary/current_source_regression.csv")
    args = parser.parse_args()
    rows = audit(args.paper_results, args.config_dir)
    write_csv(args.output, rows)
    ratios = [float(row["fresh_over_frozen"]) for row in rows]
    special_recall_deltas = [
        abs(float(row["recall_delta"])) for row in rows if row["method"] != "Plain UNG"
    ]
    print(
        f"wrote {len(rows)} points; median timing ratio={statistics.median(ratios):.4f}; "
        f"range=[{min(ratios):.4f}, {max(ratios):.4f}]; "
        f"max special Recall drift={max(special_recall_deltas):.6f}"
    )


if __name__ == "__main__":
    main()
