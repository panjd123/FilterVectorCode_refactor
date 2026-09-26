#!/usr/bin/env python3
"""Summarize fixed-L and equal-Recall results for a selection sweep."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path
from typing import Any

import experiment_core
import run_selection_sweep


WORK_FIELDS = {
    "nodes_visited_warm_median": "AverageNodesVisited",
    "regular_nodes_expanded_warm_median": "AverageRegularNodesExpanded",
    "special_nodes_expanded_warm_median": "AverageSpecialNodesExpanded",
    "regular_edges_scanned_warm_median": "AverageRegularEdgesScanned",
    "special_edges_scanned_warm_median": "AverageSpecialEdgesScanned",
    "special_intra_edges_scanned_warm_median": "AverageSpecialIntraEdgesScanned",
    "special_inter_edges_scanned_warm_median": "AverageSpecialInterEdgesScanned",
    "total_edges_scanned_warm_median": "AverageTotalEdgesScanned",
    "total_distance_calcs_warm_median": "AverageTotalDistanceCalcs",
    "entry_point_distance_calcs_warm_median": "AverageEntryPointDistanceCalcs",
    "graph_search_distance_calcs_warm_median": "AverageGraphSearchDistanceCalcs",
    "num_entries_warm_median": "AverageNumEntries",
    "entry_group_matched_points_warm_median": "AverageEntryGroupMatchedPoints",
}


def measurement_root(config: dict[str, Any]) -> Path:
    root = Path(config["output_root"])
    if config.get("pass_subdirs", False):
        root /= str(config.get("measurement_pass", "performance"))
    return root


def summary_root(config: dict[str, Any]) -> Path:
    root = Path(config["output_root"]) / "summary"
    if config.get("pass_subdirs", False):
        root /= str(config.get("measurement_pass", "performance"))
    return root


def method_summary_fields(config: dict[str, Any],
                          method: dict[str, Any]) -> dict[str, Any]:
    """Expose every independent method dimension in result tables."""
    layers = experiment_core.hierarchy_layers(method)
    if experiment_core.uses_orthogonal_method_schema(config):
        semantics = experiment_core.method_semantics(config, method)
    else:
        semantics = {
            "base_topology": method.get("base_topology", "unspecified"),
            "layer_topologies": (
                ",".join(layer["topology"] for layer in layers) or "none"),
            "entry_strategy": method.get(
                "entry_strategy",
                method.get("entry_group_provider", "unspecified")),
            "routing_policy": method.get("routing_policy", "legacy"),
        }
    thresholds = [layer["min_points"] for layer in layers]
    return {
        "layer_count": len(layers),
        "thresholds": ",".join(str(value) for value in thresholds) or "none",
        "t1": thresholds[0] if thresholds else "",
        "t2": thresholds[1] if len(thresholds) > 1 else "",
        "base_topology": semantics["base_topology"],
        "layer_topologies": semantics["layer_topologies"],
        "entry_strategy": semantics["entry_strategy"],
        "routing_policy": semantics.get("routing_policy", "always_layered"),
        "selection_role": method.get("selection_role", "unspecified"),
    }


def read_rows(config: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    root = measurement_root(config)
    protocol = experiment_core.protocol_for(config)
    workload_by_name = {item["name"]: item for item in config["workloads"]}
    for method in config["methods"]:
        for workload_name, workload in workload_by_name.items():
            if not run_selection_sweep.method_enabled_for_workload(method, workload):
                continue
            run_dir = root / method["name"] / workload_name
            num_queries = int(workload.get(
                "num_queries", config.get("expected_num_queries", 0)))
            summary_path = run_dir / "search_time_summary.csv"
            detail_path = run_dir / "search_time_details.csv"
            if not summary_path.exists():
                continue
            detail_by_l: dict[int, list[dict[str, str]]] = {}
            if detail_path.exists():
                with detail_path.open(newline="") as stream:
                    for detail in csv.DictReader(stream):
                        detail_by_l.setdefault(int(detail["Lsearch"]), []).append(detail)
            stage_by_l: dict[int, list[dict[str, str]]] = {}
            stage_path = run_dir / "search_stage_details.csv"
            if stage_path.exists():
                with stage_path.open(newline="") as stream:
                    for stage in csv.DictReader(stream):
                        stage_by_l.setdefault(int(stage["Lsearch"]), []).append(stage)
            work_by_l: dict[int, list[dict[str, str]]] = {}
            work_path = run_dir / "search_work_details.csv"
            if work_path.exists():
                with work_path.open(newline="") as stream:
                    for work in csv.DictReader(stream):
                        work_by_l.setdefault(int(work["Lsearch"]), []).append(work)
            with summary_path.open(newline="") as stream:
                for source in csv.DictReader(stream):
                    lsearch = int(source["Lsearch"])
                    method_fields = method_summary_fields(config, method)
                    detail = detail_by_l.get(lsearch, [])
                    warm_detail = [
                        item for item in detail
                        if int(item["Repeat"]) >= protocol.cold_repeats
                    ]
                    warm = [float(item["Time_ms"]) for item in warm_detail]
                    all_times = [float(item["Time_ms"]) for item in detail]
                    warm_recall = [float(item["Avg_Recall"]) for item in warm_detail]
                    warm_stage = [item for item in stage_by_l.get(lsearch, [])
                                  if int(item["Repeat"]) >= protocol.cold_repeats]
                    warm_work = [item for item in work_by_l.get(lsearch, [])
                                 if int(item["Repeat"]) >= protocol.cold_repeats]
                    def stage_median(field: str) -> float | str:
                        values = [float(item[field]) for item in warm_stage]
                        return statistics.median(values) if values else ""
                    ordered_warm = sorted(
                        warm_detail,
                        key=lambda item: float(item["Time_ms"]))
                    middle_repeats: set[int] = set()
                    if ordered_warm:
                        middle_repeats.add(int(ordered_warm[(len(ordered_warm) - 1) // 2]["Repeat"]))
                        middle_repeats.add(int(ordered_warm[len(ordered_warm) // 2]["Repeat"]))
                    aligned_stage = [item for item in warm_stage
                                     if int(item["Repeat"]) in middle_repeats]
                    def aligned_stage_mean(field: str) -> float | str:
                        values = [float(item[field]) for item in aligned_stage]
                        return statistics.mean(values) if values else ""
                    def work_median(field: str) -> float | str:
                        values = [float(item[field]) for item in warm_work
                                  if item.get(field) not in (None, "")]
                        return statistics.median(values) if values else ""
                    warm_mean = (statistics.mean(warm) if warm
                                 else float(source["Average_Time_ms"]))
                    warm_median = statistics.median(warm) if warm else warm_mean
                    rows.append({
                        "workload": workload_name,
                        "query_dir": workload["query_dir"],
                        "mean_selectivity": float(workload["mean_selectivity"]),
                        "method": method["name"],
                        **method_fields,
                        "lsearch": lsearch,
                        "recall": (statistics.mean(warm_recall) if warm_recall
                                   else float(source["Average_Recall"])),
                        "num_queries": num_queries or "",
                        "batch_ms_all": float(source["Average_Time_ms"]),
                        "batch_ms_warm": warm_mean,
                        "batch_ms_warm_median": warm_median,
                        "qps_warm_median": (
                            1000.0 * num_queries / warm_median
                            if num_queries > 0 and warm_median > 0 else ""),
                        "batch_ms_warm_cv": (statistics.stdev(warm) / warm_mean
                                             if len(warm) > 1 and warm_mean > 0 else 0.0),
                        "batch_ms_min": min(all_times) if all_times else float(source["Average_Time_ms"]),
                        "batch_ms_max": max(all_times) if all_times else float(source["Average_Time_ms"]),
                        "recall_min": (min(warm_recall) if warm_recall
                                       else float(source["Average_Recall"])),
                        "recall_max": (max(warm_recall) if warm_recall
                                       else float(source["Average_Recall"])),
                        "query_total_ms_warm_median": stage_median("AverageQueryTotal_ms"),
                        "els_ms_warm_median": stage_median("AverageELS_ms"),
                        "entry_ms_warm_median": stage_median("AverageEntryPointSetup_ms"),
                        "block_authorization_ms_warm_median": stage_median("AverageBlockAuthorization_ms"),
                        "graph_ms_warm_median": stage_median("AverageGraphSearch_ms"),
                        "residual_ms_warm_median": stage_median("AverageResidual_ms"),
                        "query_total_ms_at_batch_median": aligned_stage_mean("AverageQueryTotal_ms"),
                        "els_ms_at_batch_median": aligned_stage_mean("AverageELS_ms"),
                        "entry_ms_at_batch_median": aligned_stage_mean("AverageEntryPointSetup_ms"),
                        "block_authorization_ms_at_batch_median": aligned_stage_mean("AverageBlockAuthorization_ms"),
                        "graph_ms_at_batch_median": aligned_stage_mean("AverageGraphSearch_ms"),
                        "residual_ms_at_batch_median": aligned_stage_mean("AverageResidual_ms"),
                        "stage_closure_ms_at_batch_median": aligned_stage_mean("ClosureError_ms"),
                        "closure_error_ms_max_abs": (
                            max((abs(float(item["ClosureError_ms"])) for item in warm_stage),
                                default="")
                        ),
                        **{output: work_median(source_field)
                           for output, source_field in WORK_FIELDS.items()},
                        "summary_path": str(summary_path),
                    })
    return rows


def pareto_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result = []
    for row in rows:
        dominated = any(
            other["workload"] == row["workload"]
            and other["method"] == row["method"]
            and other["recall"] >= row["recall"]
            and other["batch_ms_warm"] <= row["batch_ms_warm"]
            and (other["recall"] > row["recall"] or other["batch_ms_warm"] < row["batch_ms_warm"])
            for other in rows
        )
        if not dominated:
            result.append(dict(row))
    return result


def equal_recall_rows(rows: list[dict[str, Any]], baseline: str, targets: list[float],
                      recall_field: str = "recall") -> list[dict[str, Any]]:
    output = []
    workloads = sorted({row["workload"] for row in rows})
    methods = sorted({row["method"] for row in rows})
    for workload in workloads:
        candidates = [row for row in rows if row["workload"] == workload]
        for target in targets:
            selected: dict[str, dict[str, Any]] = {}
            for method in methods:
                feasible = [row for row in candidates
                            if row["method"] == method and row[recall_field] >= target]
                if feasible:
                    selected[method] = min(feasible, key=lambda row: row["lsearch"])
            base = selected.get(baseline)
            for method, row in selected.items():
                enriched = dict(row)
                enriched["target_recall"] = target
                enriched["recall_margin"] = row[recall_field] - target
                enriched["baseline_method"] = baseline
                enriched["speedup_vs_baseline"] = (
                    base["batch_ms_warm"] / row["batch_ms_warm"] if base else ""
                )
                enriched["speedup_vs_baseline_median"] = (
                    base["batch_ms_warm_median"] / row["batch_ms_warm_median"]
                    if base else ""
                )
                enriched["lsearch_reduction_vs_baseline"] = (
                    1.0 - row["lsearch"] / base["lsearch"] if base else ""
                )
                output.append(enriched)
    return output


def baseline_l_targets(rows: list[dict[str, Any]], baseline: str,
                       target_lsearch: list[int],
                       recall_field: str = "recall") -> dict[str, list[float]]:
    """Use measured baseline recalls as workload-specific quality targets.

    A fixed target such as 0.95 is not reachable for every selectivity.  The
    baseline's recall at representative L values gives comparable low/mid/high
    operating points without interpolation or extrapolation.
    """
    targets: dict[str, list[float]] = {}
    for workload in sorted({row["workload"] for row in rows}):
        candidates = [row for row in rows
                      if row["workload"] == workload and row["method"] == baseline]
        selected = []
        for lsearch in target_lsearch:
            matches = [row for row in candidates if row["lsearch"] == lsearch]
            if matches:
                selected.append(float(matches[0][recall_field]))
        # Recall can saturate, so avoid duplicated targets after rounding noise.
        targets[workload] = sorted(set(selected))
    return targets


def equal_recall_rows_by_workload(
    rows: list[dict[str, Any]], baseline: str, targets: dict[str, list[float]],
    recall_field: str = "recall",
) -> list[dict[str, Any]]:
    output = []
    for workload, workload_targets in targets.items():
        subset = [row for row in rows if row["workload"] == workload]
        output.extend(equal_recall_rows(
            subset, baseline, workload_targets, recall_field=recall_field))
    return output


def max_recall_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result = []
    keys = sorted({(row["workload"], row["method"]) for row in rows})
    for workload, method in keys:
        candidates = [row for row in rows
                      if row["workload"] == workload and row["method"] == method]
        if candidates:
            result.append(max(candidates, key=lambda row: (row["recall"], -row["batch_ms_warm"])))
    return result


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(path: Path, equal_rows: list[dict[str, Any]], baseline: str,
                   maximum_rows: list[dict[str, Any]]) -> None:
    def metric(row: dict[str, Any], field: str, digits: int = 3) -> str:
        value = row.get(field, "")
        return f"{float(value):.{digits}f}" if value not in (None, "") else "NA"

    lines = ["# 多层 Special Block 选择率实验", "",
             "主表使用 warm repeats 的离散实测点；每种方法选择所有 warm repeats "
             "均达到目标 Recall 的最小实测 Lsearch，不做插值。", ""]
    workload_order = sorted(
        {row["workload"] for row in equal_rows},
        key=lambda name: min(float(row["mean_selectivity"])
                             for row in equal_rows if row["workload"] == name),
    )
    for workload in workload_order:
        subset = [row for row in equal_rows if row["workload"] == workload]
        if not subset:
            continue
        lines.extend([f"## {workload}（平均选择率 {subset[0]['mean_selectivity']:.3%}）", "",
                      f"Baseline: `{baseline}`", "",
                      "| Recall target | 方法 | hierarchy | entry | routing | warm mean Recall | warm min Recall | min-target margin | L | QPS | warm median ms | CV | median 加速 |",
                      "|---:|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|"] )
        for row in sorted(subset, key=lambda item: (item["target_recall"], item["method"])):
            median_speedup = row["speedup_vs_baseline_median"]
            median_speedup_text = (f"{median_speedup:.3f}x"
                                   if isinstance(median_speedup, float) else "NA")
            qps = row.get("qps_warm_median")
            qps_text = f"{qps:.3f}" if isinstance(qps, float) else "NA"
            hierarchy = (
                "none" if row["layer_count"] == 0 else
                f"{row['thresholds']}:{row['layer_topologies']}")
            lines.append(
                f"| {row['target_recall']:.3f} | {row['method']} | {hierarchy} | "
                f"{row['entry_strategy']} | {row['routing_policy']} | {row['recall']:.6f} | "
                f"{row['recall_min']:.6f} | {row['recall_margin']:+.6f} | {row['lsearch']} | "
                f"{qps_text} | {row['batch_ms_warm_median']:.3f} | "
                f"{row['batch_ms_warm_cv']:.3f} | {median_speedup_text} |"
            )
        lines.append("")
    breakdown_rows = [
        row for row in equal_rows
        if row.get("query_total_ms_warm_median", "") not in (None, "")
        or row.get("nodes_visited_warm_median", "") not in (None, "")
    ]
    if breakdown_rows:
        lines.extend([
            "## 达标点阶段耗时", "",
            "每个数值是先在一次 warm repeat 内对 query 取平均，再在 warm repeats "
            "间取中位数；单位为 ms/query。", "",
            "| workload | 方法 | L | total | entry-group | entry-point setup | authorization | graph search | residual |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|",
        ])
        for row in sorted(breakdown_rows, key=lambda item: (
                float(item["mean_selectivity"]), item["target_recall"], item["method"])):
            lines.append(
                f"| {row['workload']} | {row['method']} | {row['lsearch']} | "
                f"{metric(row, 'query_total_ms_warm_median')} | "
                f"{metric(row, 'els_ms_warm_median')} | "
                f"{metric(row, 'entry_ms_warm_median')} | "
                f"{metric(row, 'block_authorization_ms_warm_median')} | "
                f"{metric(row, 'graph_ms_warm_median')} | "
                f"{metric(row, 'residual_ms_warm_median')} |"
            )
        lines.extend([
            "", "## 达标点搜索工作量", "",
            "工作量同样是 warm-repeat 中位数；base/special 边计数与总边数分开报告，"
            "距离计算分为入口点与图搜索两部分。", "",
            "| workload | 方法 | L | visited points | base edges | special intra edges | special inter edges | total edges | entry distances | graph distances | total distances | entries |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ])
        for row in sorted(breakdown_rows, key=lambda item: (
                float(item["mean_selectivity"]), item["target_recall"], item["method"])):
            lines.append(
                f"| {row['workload']} | {row['method']} | {row['lsearch']} | "
                f"{metric(row, 'nodes_visited_warm_median', 1)} | "
                f"{metric(row, 'regular_edges_scanned_warm_median', 1)} | "
                f"{metric(row, 'special_intra_edges_scanned_warm_median', 1)} | "
                f"{metric(row, 'special_inter_edges_scanned_warm_median', 1)} | "
                f"{metric(row, 'total_edges_scanned_warm_median', 1)} | "
                f"{metric(row, 'entry_point_distance_calcs_warm_median', 1)} | "
                f"{metric(row, 'graph_search_distance_calcs_warm_median', 1)} | "
                f"{metric(row, 'total_distance_calcs_warm_median', 1)} | "
                f"{metric(row, 'num_entries_warm_median', 1)} |"
            )
        lines.append("")
    lines.extend(["## 扫描范围内最大 Recall", "",
                  "此表用于识别共同可达质量上限，不代表最大 L 是性能最优点。", "",
                  "| workload | 方法 | hierarchy | 最大 Recall | L | QPS | warm batch ms |",
                  "|---|---|---|---:|---:|---:|---:|"])
    for row in sorted(maximum_rows, key=lambda item: (
            float(item["mean_selectivity"]), item["method"])):
        qps = row.get("qps_warm_median")
        qps_text = f"{qps:.3f}" if isinstance(qps, float) else "NA"
        hierarchy = (
            "none" if row["layer_count"] == 0 else
            f"{row['thresholds']}:{row['layer_topologies']}")
        lines.append(
            f"| {row['workload']} | {row['method']} | {hierarchy} | "
            f"{row['recall']:.6f} | {row['lsearch']} | {qps_text} | "
            f"{row['batch_ms_warm_median']:.3f} |"
        )
    lines.append("")
    path.write_text("\n".join(lines) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=Path)
    parser.add_argument("--baseline", default="single_1k")
    parser.add_argument("--targets", nargs="*", type=float)
    parser.add_argument("--target-lsearch", nargs="*", type=int,
                        default=[500, 1000, 2000, 5000, 10000, 20000])
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    root = summary_root(config)
    rows = read_rows(config)
    write_csv(root / "all_points.csv", rows)
    write_csv(root / "pareto_points.csv", pareto_rows(rows))
    if args.targets is None:
        targets = baseline_l_targets(rows, args.baseline, args.target_lsearch)
        equal = equal_recall_rows_by_workload(rows, args.baseline, targets)
    else:
        equal = equal_recall_rows(rows, args.baseline, args.targets)
    write_csv(root / "equal_recall.csv", equal)
    if args.targets is None:
        conservative_targets = baseline_l_targets(
            rows, args.baseline, args.target_lsearch, recall_field="recall_min")
        conservative_equal = equal_recall_rows_by_workload(
            rows, args.baseline, conservative_targets, recall_field="recall_min")
    else:
        conservative_equal = equal_recall_rows(
            rows, args.baseline, args.targets, recall_field="recall_min")
    write_csv(root / "equal_recall_conservative.csv", conservative_equal)
    maximum = max_recall_rows(rows)
    write_csv(root / "max_recall.csv", maximum)
    write_markdown(root / "results.md", conservative_equal, args.baseline, maximum)
    print(f"wrote {len(rows)} points to {root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
