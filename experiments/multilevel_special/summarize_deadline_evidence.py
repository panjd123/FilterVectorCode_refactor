#!/usr/bin/env python3
"""Summarize the bounded held-out campaign without overstating its evidence.

The deadline campaign is a screen-level study: one cold repeat and two warm
repeats over six predeclared Lsearch values.  This script keeps no-crossing and
missing cases explicit, compares DRH only with the five frozen manual
alternatives, and exports the timing/work counters at each conservative
crossing.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import experiment_core
import run_selection_sweep
import summarize_selection_sweep


BASELINE_ROLE = "zero_layer_baseline"
AUTOMATIC_ROLE = "predeclared_degree_ratio_hierarchy_v1"
MANUAL_ROLE = "predeclared_manual_oracle_grid"
ROUTING_POLICY = "require_upper_authorization"
EXPECTED_MANUAL_ALTERNATIVES = 5

STAGE_FIELDS = (
    "query_total_ms_warm_median",
    "els_ms_warm_median",
    "entry_ms_warm_median",
    "block_authorization_ms_warm_median",
    "graph_ms_warm_median",
    "residual_ms_warm_median",
)
WORK_FIELDS = (
    "nodes_visited_warm_median",
    "regular_edges_scanned_warm_median",
    "special_intra_edges_scanned_warm_median",
    "special_inter_edges_scanned_warm_median",
    "total_edges_scanned_warm_median",
    "entry_point_distance_calcs_warm_median",
    "graph_search_distance_calcs_warm_median",
    "total_distance_calcs_warm_median",
    "num_entries_warm_median",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent,
        prefix=f".{path.name}.", delete=False,
    ) as stream:
        stream.write(text)
        temporary = Path(stream.name)
    temporary.replace(path)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", newline="", dir=path.parent,
        prefix=f".{path.name}.", delete=False,
    ) as stream:
        if fields:
            writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)
        temporary = Path(stream.name)
    temporary.replace(path)


def atomic_json(path: Path, payload: Any) -> None:
    atomic_text(path, json.dumps(payload, indent=2) + "\n")


def only(items: list[str], description: str) -> str:
    if len(items) != 1:
        raise ValueError(f"expected one {description}, got {items}")
    return items[0]


def hierarchy_spec(row: dict[str, Any]) -> str:
    if int(row["layer_count"]) == 0:
        return "none"
    return f"{row['thresholds']}:{row['layer_topologies']}"


def operating_point(
    rows: list[dict[str, Any]], target_recall: float,
) -> tuple[str, dict[str, Any] | None]:
    if not rows:
        return "missing", None
    feasible = [row for row in rows if float(row["recall_min"]) >= target_recall]
    if feasible:
        return "crossing", min(feasible, key=lambda row: int(row["lsearch"]))
    return "no_crossing", max(
        rows,
        key=lambda row: (
            float(row["recall_min"]), float(row["recall"]), -int(row["lsearch"])
        ),
    )


def validate_crossing_breakdown(row: dict[str, Any]) -> None:
    for field in STAGE_FIELDS + WORK_FIELDS:
        value = row.get(field, "")
        if value in (None, ""):
            raise ValueError(
                f"missing breakdown {field}: {row['method']}/{row['workload']}")
        if not math.isfinite(float(value)) or float(value) < 0:
            raise ValueError(
                f"invalid breakdown {field}={value}: "
                f"{row['method']}/{row['workload']}")
    closure = row.get("closure_error_ms_max_abs", "")
    if closure in (None, "") or abs(float(closure)) > 1e-6:
        raise ValueError(
            f"stage closure exceeds tolerance: {row['method']}/{row['workload']}")
    if float(row["total_edges_scanned_warm_median"]) < max(
        float(row["regular_edges_scanned_warm_median"]),
        float(row["special_intra_edges_scanned_warm_median"]),
        float(row["special_inter_edges_scanned_warm_median"]),
    ):
        raise ValueError(f"edge counters do not close: {row['method']}/{row['workload']}")
    if float(row["total_distance_calcs_warm_median"]) < max(
        float(row["entry_point_distance_calcs_warm_median"]),
        float(row["graph_search_distance_calcs_warm_median"]),
    ):
        raise ValueError(
            f"distance counters do not close: {row['method']}/{row['workload']}")


def method_roles(config: dict[str, Any]) -> tuple[str, str, list[str]]:
    methods = config["methods"]
    baseline = only([
        row["name"] for row in methods if row.get("selection_role") == BASELINE_ROLE
    ], "zero-layer baseline")
    automatic = only([
        row["name"] for row in methods if row.get("selection_role") == AUTOMATIC_ROLE
    ], "automatic DRH")
    manual = [
        row["name"] for row in methods if row.get("selection_role") == MANUAL_ROLE
    ]
    if len(manual) != EXPECTED_MANUAL_ALTERNATIVES:
        raise ValueError(
            f"expected {EXPECTED_MANUAL_ALTERNATIVES} frozen manual alternatives, "
            f"got {manual}")
    by_name = {row["name"]: row for row in methods}
    for name in [automatic, *manual]:
        if by_name[name].get("routing_policy") != ROUTING_POLICY:
            raise ValueError(f"oracle candidate lacks exact routing gate: {name}")
    return baseline, automatic, manual


def summarize_search_config(
    config_path: Path, require_complete: bool,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    config = experiment_core.load_config(config_path)
    protocol = experiment_core.protocol_for(config)
    if protocol.cold_repeats != 1 or protocol.measured_repeats != 2:
        raise ValueError(f"unexpected deadline repeat protocol: {config_path}")
    baseline, automatic, manual = method_roles(config)
    methods = {row["name"]: row for row in config["methods"]}
    workloads = {row["name"]: row for row in config["workloads"]}
    rows = summarize_selection_sweep.read_rows(config)
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault((str(row["method"]), str(row["workload"])), []).append(row)

    expected_l = {
        int(value) for value in config.get("lsearch_values", [])
    }
    all_rows: list[dict[str, Any]] = []
    breakdown_rows: list[dict[str, Any]] = []
    selected: dict[tuple[str, str], tuple[str, dict[str, Any] | None]] = {}
    incomplete: list[str] = []
    for workload_name, workload in workloads.items():
        target = float(config["recall_thresholds"][workload_name])
        for method_name, method in methods.items():
            if not run_selection_sweep.method_enabled_for_workload(method, workload):
                continue
            points = grouped.get((method_name, workload_name), [])
            measured_l = {int(row["lsearch"]) for row in points}
            if measured_l != expected_l:
                incomplete.append(
                    f"{config['dataset']}/{method_name}/{workload_name}: "
                    f"L={sorted(measured_l)} expected={sorted(expected_l)}")
            status, point = operating_point(points, target)
            selected[(method_name, workload_name)] = (status, point)
            record: dict[str, Any] = {
                "dataset": config["dataset"],
                "workload": workload_name,
                "mean_selectivity": float(workload["mean_selectivity"]),
                "method": method_name,
                "selection_role": method.get("selection_role", "unspecified"),
                "status": status,
                "target_recall": target,
                "expected_lsearch_count": len(expected_l),
                "measured_lsearch_count": len(measured_l),
            }
            if point is not None:
                record.update({
                    "hierarchy": hierarchy_spec(point),
                    "entry_strategy": point["entry_strategy"],
                    "routing_policy": point["routing_policy"],
                    "lsearch": int(point["lsearch"]),
                    "recall_mean": float(point["recall"]),
                    "recall_min": float(point["recall_min"]),
                    "recall_max": float(point["recall_max"]),
                    "qps_warm_median": float(point["qps_warm_median"]),
                    "batch_ms_warm_median": float(point["batch_ms_warm_median"]),
                    "batch_ms_warm_cv": float(point["batch_ms_warm_cv"]),
                })
                if status == "crossing":
                    validate_crossing_breakdown(point)
                    breakdown_rows.append({
                        **record,
                        **{field: point[field] for field in STAGE_FIELDS + WORK_FIELDS},
                        "layered_path_activation_rate_warm_median": point.get(
                            "layered_path_activation_rate_warm_median", ""),
                        "closure_error_ms_max_abs": point["closure_error_ms_max_abs"],
                    })
            all_rows.append(record)

    if require_complete and incomplete:
        raise ValueError("incomplete deadline campaign:\n" + "\n".join(incomplete))

    comparison_rows: list[dict[str, Any]] = []
    for workload_name, workload in workloads.items():
        baseline_status, baseline_point = selected[(baseline, workload_name)]
        auto_status, auto_point = selected[(automatic, workload_name)]
        manual_points = [
            (name, selected[(name, workload_name)][1])
            for name in manual
            if selected[(name, workload_name)][0] == "crossing"
        ]
        best_manual = (
            max(manual_points, key=lambda item: float(item[1]["qps_warm_median"]))
            if manual_points else None
        )
        row = {
            "dataset": config["dataset"],
            "workload": workload_name,
            "mean_selectivity": float(workload["mean_selectivity"]),
            "target_recall": float(config["recall_thresholds"][workload_name]),
            "baseline_status": baseline_status,
            "automatic_status": auto_status,
            "manual_crossing_count": len(manual_points),
            "manual_candidate_count": len(manual),
            "automatic_method": automatic,
            "best_manual_method": best_manual[0] if best_manual else "",
        }
        if baseline_status == "crossing" and baseline_point is not None:
            row.update({
                "baseline_lsearch": int(baseline_point["lsearch"]),
                "baseline_qps": float(baseline_point["qps_warm_median"]),
                "baseline_recall_min": float(baseline_point["recall_min"]),
            })
        if auto_status == "crossing" and auto_point is not None:
            row.update({
                "automatic_hierarchy": hierarchy_spec(auto_point),
                "automatic_lsearch": int(auto_point["lsearch"]),
                "automatic_qps": float(auto_point["qps_warm_median"]),
                "automatic_recall_min": float(auto_point["recall_min"]),
            })
            if baseline_status == "crossing" and baseline_point is not None:
                row["automatic_speedup_vs_baseline"] = (
                    float(auto_point["qps_warm_median"])
                    / float(baseline_point["qps_warm_median"])
                )
        if best_manual:
            manual_name, manual_point = best_manual
            row.update({
                "best_manual_hierarchy": hierarchy_spec(manual_point),
                "best_manual_lsearch": int(manual_point["lsearch"]),
                "best_manual_qps": float(manual_point["qps_warm_median"]),
                "best_manual_recall_min": float(manual_point["recall_min"]),
            })
            if auto_status == "crossing" and auto_point is not None:
                row["automatic_qps_fraction_of_best_manual"] = (
                    float(auto_point["qps_warm_median"])
                    / float(manual_point["qps_warm_median"])
                )
        comparison_rows.append(row)

    complete_manual: list[tuple[str, float]] = []
    for name in manual:
        points = [selected[(name, workload)][1] for workload in workloads]
        statuses = [selected[(name, workload)][0] for workload in workloads]
        if all(status == "crossing" for status in statuses):
            qps = [float(point["qps_warm_median"]) for point in points]
            complete_manual.append((name, math.exp(statistics.mean(map(math.log, qps)))))
    global_row: dict[str, Any] = {
        "dataset": config["dataset"],
        "workload_count": len(workloads),
        "complete_manual_candidates": len(complete_manual),
        "manual_candidate_count": len(manual),
        "automatic_method": automatic,
    }
    auto_points = [selected[(automatic, workload)][1] for workload in workloads]
    auto_statuses = [selected[(automatic, workload)][0] for workload in workloads]
    if all(status == "crossing" for status in auto_statuses):
        auto_geomean = math.exp(statistics.mean(
            math.log(float(point["qps_warm_median"])) for point in auto_points))
        global_row["automatic_geomean_qps"] = auto_geomean
        global_row["status"] = "complete"
    else:
        global_row["status"] = "automatic_not_complete"
    if complete_manual:
        name, manual_geomean = max(complete_manual, key=lambda item: item[1])
        global_row["best_global_manual_method"] = name
        global_row["best_global_manual_geomean_qps"] = manual_geomean
        if "automatic_geomean_qps" in global_row:
            global_row["automatic_qps_fraction_of_global_manual"] = (
                float(global_row["automatic_geomean_qps"]) / manual_geomean)
    return all_rows, comparison_rows, breakdown_rows, {
        "global": global_row,
        "incomplete_cases": incomplete,
        "config_sha256": sha256_file(config_path),
        "config": str(config_path),
    }


def summarize_builds(campaign: dict[str, Any]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for dataset in campaign["datasets"]:
        config_path = Path(dataset["build_config"])
        config = json.loads(config_path.read_text(encoding="utf-8"))
        root = Path(config["output_root"])
        manifest_path = root / "manifest.json"
        manifest = (json.loads(manifest_path.read_text(encoding="utf-8"))
                    if manifest_path.is_file() else {"runs": []})
        records = {row["name"]: row for row in manifest.get("runs", [])}
        for case in config["cases"]:
            record = records.get(case["name"], {})
            metadata = record.get("metadata", {})
            output.append({
                "dataset": dataset["dataset"],
                "case": case["name"],
                "selection_role": case.get("selection_role", "unspecified"),
                "hierarchy_layers": experiment_core.encode_hierarchy_layers(case),
                "benchmark_profile": case["benchmark_profile"],
                "status": record.get("status", "missing"),
                "elapsed_seconds": record.get("elapsed_seconds", ""),
                "build_time_ms": metadata.get("build_time(ms)", ""),
                "special_block_count": metadata.get("special_block_count", ""),
                "special_block_upper_count": metadata.get(
                    "special_block_upper_count", ""),
                "special_edge_count": metadata.get("special_edge_count", ""),
                "disk_bytes": metadata.get("disk_bytes", ""),
                "verified_gpu_intra_blocks": metadata.get(
                    "verified_gpu_intra_blocks", ""),
                "verified_gpu_inter_used": metadata.get(
                    "verified_gpu_inter_used", ""),
                "build_binary_sha256": record.get(
                    "source_provenance", {}).get("build_binary_sha256", ""),
            })
    return output


def value(row: dict[str, Any], field: str, digits: int = 3) -> str:
    item = row.get(field, "")
    return "NA" if item in (None, "") else f"{float(item):.{digits}f}"


def ratio(row: dict[str, Any], field: str) -> str:
    rendered = value(row, field)
    return rendered if rendered == "NA" else rendered + "x"


def render_markdown(
    run_root: Path, all_rows: list[dict[str, Any]],
    comparisons: list[dict[str, Any]], breakdown: list[dict[str, Any]],
    global_rows: list[dict[str, Any]], builds: list[dict[str, Any]],
    incomplete: list[str],
) -> str:
    lines = [
        "# Deadline-bounded 多数据集证据报告", "",
        "> 证据级别：screen-level。每个点为 1 次 cold + 2 次 warm，QPS 是 100 个",
        "> query worker 的 batch throughput。crossing 只取所有 warm repeats 均达到",
        "> Recall@10 >= 0.90 的最小实测 Lsearch，不插值。人工对照仅含 5 个在读取",
        "> held-out query 结果前冻结的结构替代方案，不等同于 35-case full oracle。", "",
        f"生成时间（UTC）：`{datetime.now(timezone.utc).isoformat()}`", "",
        "## 自动 DRH 与人工替代", "",
        "| Dataset | Sel. | DRH status | DRH QPS | DRH/plain | Best manual | Manual QPS | DRH/manual |",
        "|---|---:|---|---:|---:|---|---:|---:|",
    ]
    for row in sorted(comparisons, key=lambda item: (
            item["dataset"], float(item["mean_selectivity"]))):
        lines.append(
            f"| {row['dataset']} | {100.0 * row['mean_selectivity']:.3f}% | "
            f"{row['automatic_status']} | {value(row, 'automatic_qps')} | "
            f"{ratio(row, 'automatic_speedup_vs_baseline')} | "
            f"{row.get('best_manual_method') or 'NA'} | "
            f"{value(row, 'best_manual_qps')} | "
            f"{value(row, 'automatic_qps_fraction_of_best_manual')} |")
    lines.extend(["", "## 单一人工配置对所有 workload", "",
                  "| Dataset | Status | DRH geomean QPS | Best global manual | Manual geomean QPS | DRH/manual |",
                  "|---|---|---:|---|---:|---:|"])
    for row in global_rows:
        lines.append(
            f"| {row['dataset']} | {row['status']} | "
            f"{value(row, 'automatic_geomean_qps')} | "
            f"{row.get('best_global_manual_method') or 'NA'} | "
            f"{value(row, 'best_global_manual_geomean_qps')} | "
            f"{value(row, 'automatic_qps_fraction_of_global_manual')} |")
    lines.extend(["", "## 全方法保守 crossing", "",
                  "| Dataset | Sel. | Role | Method | Status | L | Recall min | QPS | CV |",
                  "|---|---:|---|---|---|---:|---:|---:|---:|"])
    for row in sorted(all_rows, key=lambda item: (
            item["dataset"], float(item["mean_selectivity"]), item["method"])):
        lines.append(
            f"| {row['dataset']} | {100.0 * row['mean_selectivity']:.3f}% | "
            f"{row['selection_role']} | {row['method']} | {row['status']} | "
            f"{row.get('lsearch', 'NA')} | {value(row, 'recall_min', 4)} | "
            f"{value(row, 'qps_warm_median')} | {value(row, 'batch_ms_warm_cv')} |")
    lines.extend(["", "## Crossing 阶段耗时", "",
                  "单位为 ms/query；均为 warm-repeat 中位数。", "",
                  "| Dataset | Workload | Method | ELS | Entry point | Authorization | Graph | Residual |",
                  "|---|---|---|---:|---:|---:|---:|---:|"])
    for row in sorted(breakdown, key=lambda item: (
            item["dataset"], float(item["mean_selectivity"]), item["method"])):
        lines.append(
            f"| {row['dataset']} | {row['workload']} | {row['method']} | "
            f"{value(row, 'els_ms_warm_median')} | "
            f"{value(row, 'entry_ms_warm_median')} | "
            f"{value(row, 'block_authorization_ms_warm_median')} | "
            f"{value(row, 'graph_ms_warm_median')} | "
            f"{value(row, 'residual_ms_warm_median')} |")
    lines.extend(["", "## Crossing 搜索工作量", "",
                  "| Dataset | Workload | Method | Visited | Base edges | Special edges | Distances | Entries |",
                  "|---|---|---|---:|---:|---:|---:|---:|"])
    for row in sorted(breakdown, key=lambda item: (
            item["dataset"], float(item["mean_selectivity"]), item["method"])):
        special_edges = (
            float(row["special_intra_edges_scanned_warm_median"])
            + float(row["special_inter_edges_scanned_warm_median"])
        )
        lines.append(
            f"| {row['dataset']} | {row['workload']} | {row['method']} | "
            f"{value(row, 'nodes_visited_warm_median', 1)} | "
            f"{value(row, 'regular_edges_scanned_warm_median', 1)} | "
            f"{special_edges:.1f} | {value(row, 'total_distance_calcs_warm_median', 1)} | "
            f"{value(row, 'num_entries_warm_median', 1)} |")
    lines.extend(["", "## Held-out sidecar 构建", "",
                  "这些 CPU sidecar 只服务查询比较，不作为 GPU 构建加速证据。", "",
                  "| Dataset | Case | Role | Status | Wall s | Blocks | Upper | Edges | GPU intra | GPU inter |",
                  "|---|---|---|---|---:|---:|---:|---:|---:|---:|"])
    for row in builds:
        lines.append(
            f"| {row['dataset']} | {row['case']} | {row['selection_role']} | "
            f"{row['status']} | {value(row, 'elapsed_seconds')} | "
            f"{row.get('special_block_count') or 'NA'} | "
            f"{row.get('special_block_upper_count') or 'NA'} | "
            f"{row.get('special_edge_count') or 'NA'} | "
            f"{row.get('verified_gpu_intra_blocks') or 'NA'} | "
            f"{row.get('verified_gpu_inter_used') or 'NA'} |")
    lines.extend(["", "## 完整性", ""])
    if incomplete:
        lines.append("以下 case 尚未覆盖全部六个 Lsearch 点，因此报告保持 partial：")
        lines.extend(f"- `{item}`" for item in incomplete)
    else:
        lines.append("所有声明的 dataset/method/workload case 均覆盖完整六点网格。")
    lines.extend(["", "原始结果、命令、环境、binary snapshot 和 manifest 均保留在：",
                  f"`{run_root}`。", ""])
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("campaign", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--allow-partial", action="store_true")
    args = parser.parse_args()
    campaign = json.loads(args.campaign.read_text(encoding="utf-8"))
    first_search = json.loads(Path(
        campaign["datasets"][0]["search_config"]).read_text(encoding="utf-8"))
    run_root = Path(first_search["output_root"]).parents[2]
    output_dir = args.output_dir or run_root / "deadline_summary"

    all_rows: list[dict[str, Any]] = []
    comparisons: list[dict[str, Any]] = []
    breakdown: list[dict[str, Any]] = []
    global_rows: list[dict[str, Any]] = []
    incomplete: list[str] = []
    sources = []
    for dataset in campaign["datasets"]:
        local_all, local_comparisons, local_breakdown, metadata = (
            summarize_search_config(
                Path(dataset["search_config"]),
                require_complete=not args.allow_partial,
            ))
        all_rows.extend(local_all)
        comparisons.extend(local_comparisons)
        breakdown.extend(local_breakdown)
        global_rows.append(metadata["global"])
        incomplete.extend(metadata["incomplete_cases"])
        sources.append(metadata)
    builds = summarize_builds(campaign)

    write_csv(output_dir / "all_method_crossings.csv", all_rows)
    write_csv(output_dir / "automatic_vs_manual.csv", comparisons)
    write_csv(output_dir / "automatic_vs_global_manual.csv", global_rows)
    write_csv(output_dir / "crossing_breakdown.csv", breakdown)
    write_csv(output_dir / "build_cases.csv", builds)
    atomic_text(output_dir / "deadline_evidence_report.md", render_markdown(
        run_root, all_rows, comparisons, breakdown, global_rows, builds, incomplete))
    atomic_json(output_dir / "summary_manifest.json", {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "evidence_level": "deadline_bounded_screen",
        "campaign_sha256": sha256_file(args.campaign),
        "campaign": str(args.campaign),
        "complete": not incomplete,
        "incomplete_cases": incomplete,
        "source_configs": sources,
        "outputs": [
            "all_method_crossings.csv", "automatic_vs_manual.csv",
            "automatic_vs_global_manual.csv", "crossing_breakdown.csv",
            "build_cases.csv", "deadline_evidence_report.md",
        ],
    })
    print(output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
