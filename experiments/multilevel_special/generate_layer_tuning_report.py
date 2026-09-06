#!/usr/bin/env python3
"""Render the validated 0/1/2-layer formal tuning result.

End-to-end latency is the wall time of a 1000-query, 100-thread batch.  The
five stage columns are medians of per-query instrumented time across warm
repeats. They close against per-query total, not parallel batch wall time.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


STAGES = (
    ("els_ms_warm_median", "ELS"),
    ("entry_ms_warm_median", "Entry"),
    ("block_authorization_ms_warm_median", "Auth"),
    ("graph_ms_warm_median", "Graph"),
    ("residual_ms_warm_median", "Residual"),
)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def structure(row: dict[str, str]) -> str:
    layer = int(row["layer_count"])
    if layer == 0:
        return "Plain (0 layer)"
    if layer == 1:
        return f"T1={int(row['t1']):,}"
    return f"T1={int(row['t1']):,}, T2={int(row['t2']):,}"


def qps(batch_ms: float, query_count: int) -> float:
    return query_count * 1000.0 / batch_ms


def oracle_with_speedup(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    baselines = {row["workload"]: float(row["batch_ms_warm_median"])
                 for row in rows if int(row["layer_count"]) == 0}
    output = []
    for row in rows:
        enriched = dict(row)
        base = baselines.get(row["workload"])
        latency = float(row["batch_ms_warm_median"])
        enriched["speedup_vs_layer0"] = str(base / latency) if base else ""
        output.append(enriched)
    return output


def write_table_csv(path: Path, rows: list[dict[str, str]], query_count: int) -> None:
    fields = ["selection_scope", "workload", "mean_selectivity",
              "target_recall", "layer_count", "method", "t1", "t2",
              "lsearch", "recall", "recall_min",
              "batch_ms_warm_median", "qps", "speedup_vs_layer0",
              "query_total_ms_warm_median",
              *(field for field, _ in STAGES)]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for source in rows:
            row = {field: source.get(field, "") for field in fields}
            row["qps"] = f"{qps(float(source['batch_ms_warm_median']), query_count):.6f}"
            writer.writerow(row)


def render_rows(rows: list[dict[str, str]], query_count: int) -> list[str]:
    output = []
    for row in sorted(rows, key=lambda item: (item["workload"], int(item["layer_count"]))):
        stages = [float(row[field]) for field, _ in STAGES]
        output.append(
            f"| {row['workload']} | {float(row['mean_selectivity']):.3%} | "
            f"{int(row['layer_count'])} | {structure(row)} | {int(row['lsearch'])} | "
            f"{float(row['recall_min']):.4f} / {float(row['recall']):.4f} | "
            f"{float(row['batch_ms_warm_median']):.3f} | "
            f"{qps(float(row['batch_ms_warm_median']), query_count):.1f} | "
            f"{float(row['speedup_vs_layer0']):.3f}x | "
            + " | ".join(f"{value:.4f}" for value in stages) + " |"
        )
    return output


def selected_build_rows(shared_summary: list[dict[str, str]],
                        oracle_rows: list[dict[str, str]],
                        build_manifest: dict | None) -> list[dict]:
    if build_manifest is None:
        return []
    selected_methods = {row["method"] for row in shared_summary}
    selected_methods.update(row["method"] for row in oracle_rows)
    by_name = {row["name"]: row for row in build_manifest.get("runs", [])}
    output = []
    for method in sorted(selected_methods):
        source = by_name.get(method)
        if source is None:
            continue  # layer 0 has no Special overlay build.
        if source.get("status") != "complete" or source.get("returncode") != 0:
            raise RuntimeError(f"selected build is incomplete: {method}")
        meta = source.get("metadata", {})
        output.append({
            "method": method,
            "layer_count": 2 if source.get("upper_min_points") is not None else 1,
            "t1": int(source["min_points"]),
            "t2": source.get("upper_min_points"),
            "wall_s": float(source["elapsed_seconds"]),
            "metadata_ms": float(meta["special_block_metadata_time(ms)"]),
            "trie_ms": float(meta["special_block_trie_build_time(ms)"]),
            "intra_ms": float(meta["special_edge_intra_build_time(ms)"]),
            "inter_ms": float(meta["special_edge_inter_build_time(ms)"]),
            "regular_ms": float(meta["special_trie_regular_edge_build_time(ms)"]),
            "save_ms": float(meta["special_blocks_save_time(ms)"]),
            "blocks": int(meta["special_block_count"]),
            "upper_blocks": int(meta["special_block_upper_count"]),
            "edges": int(meta["special_edge_count"]),
            "disk_bytes": int(meta["disk_bytes"]),
        })
    return output


def render_report(config: dict, shared_summary: list[dict[str, str]],
                  shared_rows: list[dict[str, str]],
                  oracle_rows: list[dict[str, str]],
                  boundary_rows: list[dict[str, str]],
                  build_rows: list[dict] | None = None) -> str:
    query_count = int(config.get("expected_num_queries", 1000))
    lines = [
        "# 0/1/2 层 Special Block 公平调优结果", "",
        "所有性能点均为离散实测；以每个 repeat 的最低 Recall 达到预声明门槛为可行条件，不插值。",
        "端到端列是 1000-query、100-thread batch 的 warm-repeat 中位墙钟；阶段列是 warm-repeat 中位的平均单查询工作时间。阶段五项彼此互斥并闭合到单查询总时间，但由于查询并行，不能与 batch 墙钟直接相加或换算。",
        "", "## 跨六档共享阈值：部署主结论", "",
        "| workload | 选择率 | 层数 | 共享结构 | L | Recall min / mean | batch ms | QPS | vs 0层 | ELS ms/q | Entry ms/q | Auth ms/q | Graph ms/q | Residual ms/q |",
        "|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        *render_rows(shared_rows, query_count), "",
        "共享结构按六档相对 0 层延迟比的几何平均最小化；同一层只能选择一套 T1/T2，L 仍按 workload 调整。", "",
        "| 层数 | 结构 | 六档几何平均加速 | 六档 warm batch ms 之和 |",
        "|---:|---|---:|---:|",
    ]
    for row in sorted(shared_summary, key=lambda item: int(item["layer_count"])):
        lines.append(
            f"| {int(row['layer_count'])} | {structure(row)} | "
            f"{float(row['geomean_speedup_vs_layer0']):.3f}x | "
            f"{float(row['sum_batch_ms_warm_median']):.3f} |"
        )
    lines += ["", "## 逐 workload oracle 上界", "",
              "该表允许每个 workload 独立更换 T1/T2，仅表示调参上界，不是单一可部署配置。", "",
              "| workload | 选择率 | 层数 | oracle 结构 | L | Recall min / mean | batch ms | QPS | vs 0层 | ELS ms/q | Entry ms/q | Auth ms/q | Graph ms/q | Residual ms/q |",
              "|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
              *render_rows(oracle_rows, query_count), ""]
    lines += ["## 边界审计", "",
              "若下表非空，所列赢家仍触及离散扫描边界；在对应方向扩展前，只能称为当前网格内最优。", "",
              "| 范围 | workload | 层数 | 方法 | T1 | T2 | 轴 | 方向 | 已测轴值 |",
              "|---|---|---:|---|---:|---:|---|---|---|"]
    for row in boundary_rows:
        lines.append(
            f"| {row['selection_scope']} | {row['workload']} | {row['layer_count']} | "
            f"{row['method']} | {row['t1']} | {row['t2']} | {row['axis']} | "
            f"{row['direction']} | {row['measured_axis_values']} |"
        )
    if not boundary_rows:
        lines.append("| -- | -- | -- | 无触边赢家 | -- | -- | -- | -- | -- |")
    lines += ["", "## 所选结构的构建成本", "",
              "这里计量的是在既有 UNG 主图之上生成 Special Block overlay 的增量成本，不包含基础 UNG/Vamana 主图构建；因此 0 层记为 N/A，而不是 0。各阶段是 builder 原生计时，wall 是 runner 外层墙钟。", "",
              "| 层数 | 结构 | wall s | metadata ms | trie ms | intra ms | inter ms | regular-overlay ms | save ms | blocks | upper | special edges | disk GiB |",
              "|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for row in sorted(build_rows or [], key=lambda item: (item["layer_count"], item["t1"], item["t2"] or 0)):
        t2 = "" if row["t2"] is None else f", T2={int(row['t2']):,}"
        lines.append(
            f"| {row['layer_count']} | T1={row['t1']:,}{t2} | {row['wall_s']:.3f} | "
            f"{row['metadata_ms']:.3f} | {row['trie_ms']:.3f} | {row['intra_ms']:.3f} | "
            f"{row['inter_ms']:.3f} | {row['regular_ms']:.3f} | {row['save_ms']:.3f} | "
            f"{row['blocks']:,} | {row['upper_blocks']:,} | {row['edges']:,} | "
            f"{row['disk_bytes'] / (1024 ** 3):.3f} |"
        )
    if not build_rows:
        lines.append("| -- | 构建 manifest 未提供 | -- | -- | -- | -- | -- | -- | -- | -- | -- | -- | -- |")
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=Path)
    parser.add_argument("--summary-dir", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--build-manifest", type=Path)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    root = args.summary_dir or Path(config["output_root"]) / "summary"
    shared_summary = read_csv(root / "shared_configurations.csv")
    shared = read_csv(root / "shared_selected_points.csv")
    oracle = oracle_with_speedup(read_csv(root / "layer_oracle.csv"))
    boundaries = read_csv(root / "selected_boundary_audit.csv")
    build_manifest = (json.loads(args.build_manifest.read_text())
                      if args.build_manifest else None)
    build_rows = selected_build_rows(shared_summary, oracle, build_manifest)
    query_count = int(config.get("expected_num_queries", 1000))
    write_table_csv(root / "layer_tuning_shared_table.csv", shared, query_count)
    write_table_csv(root / "layer_tuning_oracle_table.csv", oracle, query_count)
    output = args.output or root / "layer_tuning_report.md"
    output.write_text(render_report(
        config, shared_summary, shared, oracle, boundaries, build_rows),
                      encoding="utf-8")
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
