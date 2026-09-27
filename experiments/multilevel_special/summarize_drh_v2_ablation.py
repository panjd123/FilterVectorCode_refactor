#!/usr/bin/env python3
"""Consolidate the same-binary DRH-v2 ablation with explicit failures."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path
from typing import Any

import prepare_drh_v2_ablation as prepare


HERE = Path(__file__).resolve().parent
DEFAULT_MANIFEST = HERE / "config.drh_v2_manifest.json"


def summary_root(config: dict[str, Any]) -> Path:
    root = Path(config["output_root"]) / "summary"
    if config.get("pass_subdirs", False):
        root /= str(config.get("measurement_pass", "performance"))
    return root


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def operating_point(
    rows: list[dict[str, str]], method: str, workload: str, target: float,
) -> tuple[str, dict[str, str] | None]:
    candidates = [
        row for row in rows
        if row["method"] == method and row["workload"] == workload
    ]
    if not candidates:
        return "missing", None
    feasible = [row for row in candidates if float(row["recall_min"]) >= target]
    if feasible:
        return "crossing", min(feasible, key=lambda row: int(row["lsearch"]))
    return "no_crossing", max(
        candidates,
        key=lambda row: (
            float(row["recall_min"]), float(row["recall"]), -int(row["lsearch"])
        ),
    )


def percentile(values: list[float], fraction: float) -> float | str:
    if not values:
        return ""
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, int(fraction * len(ordered) + 0.999999) - 1))
    return ordered[index]


def route_stats(
    config: dict[str, Any], method: str, workload: str, lsearch: int,
) -> dict[str, float | str]:
    root = Path(config["output_root"])
    if config.get("pass_subdirs", False):
        root /= str(config.get("measurement_pass", "performance"))
    path = root / method / workload / f"query_details_repeat{config['num_repeats']}.csv"
    rows = read_csv(path)
    cold = int(config["protocol"]["cold_repeats"])
    selected = [
        row for row in rows
        if int(row["Repeat"]) >= cold and int(row["Lsearch"]) == lsearch
    ]
    if not selected:
        return {
            "route_activation_rate": "", "upper_enabled_rate": "",
            "upper_direct_mass_p50": "", "upper_direct_mass_p95": "",
        }
    masses = [float(row["SpecialQueryUpperCoveredPoints"]) for row in selected]
    return {
        "route_activation_rate": statistics.mean(
            float(row["SpecialBlockSearchUsed"]) for row in selected),
        "upper_enabled_rate": statistics.mean(
            float(row["SpecialQueryUpperEnabled"]) for row in selected),
        "upper_direct_mass_p50": statistics.median(masses),
        "upper_direct_mass_p95": percentile(masses, 0.95),
    }


def summarize_dataset(spec: dict[str, Any]) -> list[dict[str, Any]]:
    config_path = Path(spec["config"])
    config = json.loads(config_path.read_text(encoding="utf-8"))
    points = read_csv(summary_root(config) / "all_points.csv")
    methods = (
        ("plain", str(spec["baseline_method"])),
        ("drh_v1", str(spec["drh_v1_method"])),
        ("drh_v2", str(spec["drh_v2_method"])),
    )
    output = []
    for workload in config["workloads"]:
        workload_name = str(workload["name"])
        target = float(config["recall_thresholds"][workload_name])
        selected = {
            role: operating_point(points, method, workload_name, target)
            for role, method in methods
        }
        plain = selected["plain"][1] if selected["plain"][0] == "crossing" else None
        v1 = selected["drh_v1"][1] if selected["drh_v1"][0] == "crossing" else None
        for role, method in methods:
            status, row = selected[role]
            result: dict[str, Any] = {
                "dataset": spec["dataset"],
                "workload": workload_name,
                "mean_selectivity": workload["mean_selectivity"],
                "target_recall": target,
                "role": role,
                "method": method,
                "status": status,
                "next_scale_threshold": spec["next_scale_threshold"] if role == "drh_v2" else "",
            }
            if row is not None:
                result.update({
                    "lsearch": row["lsearch"],
                    "recall": row["recall"],
                    "recall_min": row["recall_min"],
                    "batch_ms_warm_median": row["batch_ms_warm_median"],
                    "qps_warm_median": row["qps_warm_median"],
                    "els_ms_warm_median": row["els_ms_warm_median"],
                    "entry_ms_warm_median": row["entry_ms_warm_median"],
                    "block_authorization_ms_warm_median": row["block_authorization_ms_warm_median"],
                    "graph_ms_warm_median": row["graph_ms_warm_median"],
                    "nodes_visited_warm_median": row["nodes_visited_warm_median"],
                    "total_distance_calcs_warm_median": row["total_distance_calcs_warm_median"],
                })
                if role != "plain":
                    result.update(route_stats(
                        config, method, workload_name, int(row["lsearch"])))
                result["speedup_vs_plain"] = (
                    float(plain["batch_ms_warm_median"]) /
                    float(row["batch_ms_warm_median"])
                    if plain is not None and status == "crossing" else ""
                )
                result["speedup_vs_drh_v1"] = (
                    float(v1["batch_ms_warm_median"]) /
                    float(row["batch_ms_warm_median"])
                    if role == "drh_v2" and v1 is not None and
                    status == "crossing" else ""
                )
            output.append(result)
    return output


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = list(dict.fromkeys(key for row in rows for key in row))
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def fmt(row: dict[str, Any], field: str, digits: int = 3) -> str:
    value = row.get(field, "")
    return "NA" if value in (None, "") else f"{float(value):.{digits}f}"


def write_markdown(path: Path, rows: list[dict[str, Any]]) -> None:
    lines = [
        "# DRH-v2 Same-Binary Ablation",
        "",
        "All crossings are the minimum measured Lsearch for which every warm repeat "
        "meets the configured Recall target. One cold repeat is discarded; primary "
        "QPS uses the median of two complete warm batches.",
        "",
        "| Dataset | Selectivity | Method | Status | L | Recall min | QPS | vs plain | vs DRH-v1 | route rate | upper mass p50/p95 |",
        "|---|---:|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['dataset']} | {100 * float(row['mean_selectivity']):.3f}% | "
            f"{row['role']} | {row['status']} | {row.get('lsearch', 'NA')} | "
            f"{fmt(row, 'recall_min', 4)} | {fmt(row, 'qps_warm_median', 2)} | "
            f"{fmt(row, 'speedup_vs_plain', 3)} | {fmt(row, 'speedup_vs_drh_v1', 3)} | "
            f"{fmt(row, 'route_activation_rate', 3)} | "
            f"{fmt(row, 'upper_direct_mass_p50', 0)}/{fmt(row, 'upper_direct_mass_p95', 0)} |"
        )
    lines.extend([
        "",
        "The mass gate is a structural eligibility test, not a speedup guarantee. "
        "Rows with `no_crossing` are retained and receive no speedup claim.",
        "",
    ])
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    rows = []
    for spec in manifest["configs"]:
        rows.extend(summarize_dataset(spec))
    run_root = Path(manifest["run_root"])
    write_csv(run_root / "drh_v2_summary.csv", rows)
    write_markdown(run_root / "drh_v2_summary.md", rows)
    evidence = {
        "schema_version": 1,
        "source_manifest": str(args.manifest.resolve()),
        "source_manifest_sha256": prepare.sha256_file(args.manifest.resolve()),
        "row_count": len(rows),
        "summary_csv": str(run_root / "drh_v2_summary.csv"),
        "summary_csv_sha256": prepare.sha256_file(run_root / "drh_v2_summary.csv"),
        "summary_markdown": str(run_root / "drh_v2_summary.md"),
        "summary_markdown_sha256": prepare.sha256_file(run_root / "drh_v2_summary.md"),
    }
    prepare.atomic_json(run_root / "drh_v2_summary_manifest.json", evidence)
    print(run_root / "drh_v2_summary.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
