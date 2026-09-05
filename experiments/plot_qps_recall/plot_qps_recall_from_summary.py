#!/usr/bin/env python3
"""Draw QPS-Recall curves from UNG search_time_summary.csv files.

This script intentionally uses only Python's standard library.  It writes SVG
figures, so it can run in environments without matplotlib.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import html
import math
import struct
from pathlib import Path


@dataclass(frozen=True)
class LsearchSpec:
    start: int
    step: int
    end: int

    def suffix(self) -> str:
        return f"{self.start}_{self.step}_{self.end}"

    def contains(self, value: int) -> bool:
        return self.start <= value <= self.end and (value - self.start) % self.step == 0


@dataclass(frozen=True)
class MethodSpec:
    method_dir: str
    label: str
    color: str
    marker: str
    query_task: str | None = None
    lsearch: LsearchSpec | None = None
    explicit_result_task: str | None = None

    def result_task(self, fallback_result_task: str) -> str:
        if self.explicit_result_task:
            return self.explicit_result_task
        if self.query_task and self.lsearch:
            return f"{self.query_task}_{self.lsearch.suffix()}"
        return fallback_result_task


DEFAULT_METHODS = [
    MethodSpec("UNG", "Original UNG", "#4c78a8", "circle"),
    MethodSpec("gpu_bruteforce_els_ung", "GPU brute-force ELS UNG", "#f58518", "square"),
    MethodSpec("gpu_bruteforce_els_special_blocks", "GPU ELS + special_blocks", "#54a24b", "triangle"),
    MethodSpec("cpu_bruteforce_els_ung", "CPU brute-force ELS UNG", "#e45756", "diamond"),
    MethodSpec("cpu_bruteforce_els_special_blocks", "CPU ELS + special_blocks", "#72b7b2", "cross"),
    MethodSpec("FAVOR", "FAVOR", "#b279a2", "plus"),
    MethodSpec("NaviX", "NaviX", "#ff9da6", "x"),
]


def parse_method_spec(spec: str) -> MethodSpec:
    parts = spec.split(":")
    if len(parts) not in (4, 6, 8):
        raise ValueError(
            "--method must be METHOD:LABEL:COLOR:MARKER, "
            "METHOD:LABEL:COLOR:MARKER:QUERY_TASK:RESULT_TASK, or "
            "METHOD:LABEL:COLOR:MARKER:QUERY_TASK:START:STEP:END"
        )
    method = MethodSpec(parts[0], parts[1], parts[2], parts[3])
    if len(parts) == 4:
        return method
    if len(parts) == 6:
        return MethodSpec(parts[0], parts[1], parts[2], parts[3], parts[4], None, parts[5])
    lsearch = LsearchSpec(start=int(parts[5]), step=int(parts[6]), end=int(parts[7]))
    if lsearch.step <= 0:
        raise ValueError("method lsearch step must be positive")
    return MethodSpec(parts[0], parts[1], parts[2], parts[3], parts[4], lsearch)


def compose_result_task(query_task: str, lsearch: LsearchSpec | None) -> str:
    if not lsearch:
        return query_task
    return f"{query_task}_{lsearch.suffix()}"


def filter_lsearch_rows(rows: list[dict[str, float | int]], lsearch: LsearchSpec | None):
    if not lsearch:
        return rows
    return [row for row in rows if lsearch.contains(int(row["Lsearch"]))]


def skip_first_lsearch_row(rows: list[dict[str, float | int]]):
    """Drop the point with the smallest Lsearch value for a selected method."""
    if not rows:
        return rows
    first_lsearch = min(int(row["Lsearch"]) for row in rows)
    return [row for row in rows if int(row["Lsearch"]) != first_lsearch]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Draw QPS-Recall SVG curves from UNG summaries.")
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--query-task", required=True, help="Query task used to find query bin/labels under data-root.")
    parser.add_argument("--result-task", default=None, help="Result directory task name under results-root. Defaults to --query-task.")
    parser.add_argument("--dataset", default="Amazon")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument(
        "--method",
        action="append",
        default=None,
        help="METHOD:LABEL:COLOR:MARKER or METHOD:LABEL:COLOR:MARKER:QUERY_TASK:START:STEP:END.",
    )
    parser.add_argument(
        "--skip-first-lsearch-method",
        action="append",
        default=None,
        metavar="METHOD",
        help="Skip the first (smallest) Lsearch point when drawing this method.",
    )
    parser.add_argument(
        "--sort-mode",
        choices=("time-column-only", "row-by-time", "none"),
        default="time-column-only",
        help=(
            "time-column-only sorts only Average_Time_ms and keeps other columns in original order; "
            "row-by-time sorts whole rows by Average_Time_ms; none keeps the summary unchanged."
        ),
    )
    parser.add_argument("--legend-outside", action="store_true", default=True)
    parser.add_argument("--skip-missing", action="store_true", help="Skip methods whose search_time_summary.csv is missing.")
    parser.add_argument("--no-log", action="store_true", help="Only write the linear-scale SVG.")
    parser.add_argument(
        "--threshold",
        action="append",
        type=float,
        default=None,
        help="Recall threshold for summary table. Can be repeated. Overrides auto thresholds.",
    )
    parser.add_argument(
        "--threshold-mode",
        choices=("auto", "fixed"),
        default="auto",
        help="auto derives recall thresholds from loaded result range; fixed uses --threshold values or defaults.",
    )
    parser.add_argument("--threshold-step", type=float, default=0.05)
    parser.add_argument("--lsearch-start", type=int, default=None)
    parser.add_argument("--lsearch-step", type=int, default=None)
    parser.add_argument("--lsearch-end", type=int, default=None)
    return parser.parse_args()


def find_query_bin(data_root: Path, query_task: str, dataset: str) -> Path:
    direct = data_root / query_task / f"{dataset}_query.bin"
    if direct.exists():
        return direct
    matches = sorted(data_root.glob(f"**/{query_task}/{dataset}_query.bin"))
    if matches:
        return matches[0]
    raise FileNotFoundError(
        f"could not find {dataset}_query.bin for {query_task} under {data_root}"
    )


def query_count(data_root: Path, query_task: str, dataset: str) -> tuple[int, int]:
    bin_path = find_query_bin(data_root, query_task, dataset)
    with bin_path.open("rb") as f:
        header = f.read(8)
    if len(header) != 8:
        raise ValueError(f"bad query bin header: {bin_path}")
    return struct.unpack("<II", header)


def read_summary(path: Path) -> list[dict[str, float | int]]:
    with path.open(newline="") as f:
        rows = []
        for row in csv.DictReader(f):
            lsearch = int(float(row["Lsearch"]))
            average_efs = float(row.get("Average_Efs") or lsearch)
            average_recall = float(row.get("Average_Recall") or row["Recall"])
            rows.append(
                {
                    "Lsearch": lsearch,
                    "Average_Efs": average_efs,
                    "Average_Time_ms": float(row["Average_Time_ms"]),
                    "Average_Recall": average_recall,
                }
            )
    return rows


def build_draw_rows(rows: list[dict[str, float | int]], num_queries: int, sort_mode: str):
    if sort_mode == "row-by-time":
        draw_rows = [dict(row) for row in sorted(rows, key=lambda row: float(row["Average_Time_ms"]))]
    else:
        draw_rows = [dict(row) for row in rows]
        if sort_mode == "time-column-only":
            sorted_times = sorted(float(row["Average_Time_ms"]) for row in rows)
            for row, avg_ms in zip(draw_rows, sorted_times):
                row["Average_Time_ms"] = avg_ms
    for row in draw_rows:
        avg_ms = float(row["Average_Time_ms"])
        row["QPS"] = (num_queries * 1000.0 / avg_ms) if avg_ms > 0 else 0.0
    return draw_rows


def write_draw_csv(path: Path, rows: list[dict[str, float | int]]) -> None:
    with path.open("w", newline="") as f:
        fields = ["Lsearch", "Average_Efs", "Average_Time_ms", "Average_Recall", "QPS"]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row[field] for field in fields})


def marker_svg(kind: str, x: float, y: float, color: str) -> str:
    if kind == "circle":
        return f'<circle cx="{x:.2f}" cy="{y:.2f}" r="4" fill="white" stroke="{color}" stroke-width="2"/>'
    if kind == "square":
        return (
            f'<rect x="{x - 4:.2f}" y="{y - 4:.2f}" width="8" height="8" '
            f'fill="white" stroke="{color}" stroke-width="2"/>'
        )
    if kind == "triangle":
        points = [(x, y - 5), (x - 4.7, y + 3.5), (x + 4.7, y + 3.5)]
        return '<polygon points="{}" fill="white" stroke="{}" stroke-width="2"/>'.format(
            " ".join(f"{px:.2f},{py:.2f}" for px, py in points),
            color,
        )
    if kind == "diamond":
        points = [(x, y - 5), (x - 5, y), (x, y + 5), (x + 5, y)]
        return '<polygon points="{}" fill="white" stroke="{}" stroke-width="2"/>'.format(
            " ".join(f"{px:.2f},{py:.2f}" for px, py in points),
            color,
        )
    if kind == "cross":
        return (
            f'<g stroke="{color}" stroke-width="2.4" stroke-linecap="round">'
            f'<line x1="{x - 5:.2f}" y1="{y:.2f}" x2="{x + 5:.2f}" y2="{y:.2f}"/>'
            f'<line x1="{x:.2f}" y1="{y - 5:.2f}" x2="{x:.2f}" y2="{y + 5:.2f}"/>'
            f'</g>'
        )
    if kind == "plus":
        return (
            f'<g stroke="{color}" stroke-width="2" stroke-linecap="round">'
            f'<line x1="{x - 5:.2f}" y1="{y:.2f}" x2="{x + 5:.2f}" y2="{y:.2f}"/>'
            f'<line x1="{x:.2f}" y1="{y - 5:.2f}" x2="{x:.2f}" y2="{y + 5:.2f}"/>'
            f'</g>'
        )
    return (
        f'<g stroke="{color}" stroke-width="2" stroke-linecap="round">'
        f'<line x1="{x - 4:.2f}" y1="{y - 4:.2f}" x2="{x + 4:.2f}" y2="{y + 4:.2f}"/>'
        f'<line x1="{x - 4:.2f}" y1="{y + 4:.2f}" x2="{x + 4:.2f}" y2="{y - 4:.2f}"/>'
        f'</g>'
    )


def auto_thresholds(series, step: float) -> list[float]:
    recalls = [float(row["Average_Recall"]) for _, _, _, _, rows in series for row in rows]
    if not recalls:
        return []
    if step <= 0:
        raise ValueError("--threshold-step must be positive")
    lo = min(recalls)
    hi = max(recalls)
    start = math.ceil(lo / step) * step
    end = math.floor(hi / step) * step
    values = []
    cur = start
    while cur <= end + 1e-12:
        values.append(round(cur, 4))
        cur += step
    for value in (round(lo, 4), round(hi, 4)):
        if value not in values:
            values.append(value)
    return sorted(values)


def make_svg(series, query_task: str, result_task: str, dataset: str, num_queries: int, sort_mode: str, logy: bool, legend_outside: bool) -> str:
    width = 1180 if legend_outside else 1000
    height = 660
    margin_left = 96
    margin_right = 300 if legend_outside else 36
    margin_top = 66
    margin_bottom = 90
    plot_width = width - margin_left - margin_right
    plot_height = height - margin_top - margin_bottom
    all_recalls = [row["Average_Recall"] for _, _, _, _, rows in series for row in rows]
    all_qps = [row["QPS"] for _, _, _, _, rows in series for row in rows]
    xmin = max(0.0, min(all_recalls) - 0.03)
    xmax = min(1.0, max(all_recalls) + 0.03)
    if logy:
        ymin = max(1.0, min(all_qps) * 0.75)
        ymax = max(all_qps) * 1.25
    else:
        ymin = 0.0
        ymax = max(all_qps) * 1.12

    def sx(x: float) -> float:
        return margin_left + (x - xmin) / (xmax - xmin) * plot_width

    def sy(y: float) -> float:
        if logy:
            ly = (math.log10(max(y, 1e-9)) - math.log10(ymin)) / (math.log10(ymax) - math.log10(ymin))
            return margin_top + (1 - ly) * plot_height
        return margin_top + (1 - (y - ymin) / (ymax - ymin)) * plot_height

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<style>text{font-family:Arial,Helvetica,sans-serif;fill:#222}.axis{stroke:#333;stroke-width:1.4}.grid{stroke:#d8d8d8;stroke-width:1}.line{fill:none;stroke-width:3}.tick{font-size:13px}.title{font-size:20px;font-weight:700}.label{font-size:16px}.legend{font-size:14px}.note{font-size:12px;fill:#555}</style>',
        '<rect width="100%" height="100%" fill="white"/>',
    ]
    scale_text = " (log scale)" if logy else ""
    title = f"{dataset} {result_task}: QPS vs Recall{scale_text}"
    parts.append(f'<text x="{margin_left + plot_width / 2}" y="32" text-anchor="middle" class="title">{html.escape(title)}</text>')
    note = f"data query={query_task}; sort-mode={sort_mode}; QPS = {num_queries} * 1000 / Average_Time_ms"
    parts.append(f'<text x="{margin_left + plot_width / 2}" y="52" text-anchor="middle" class="note">{html.escape(note)}</text>')

    xticks = []
    tick = math.ceil(xmin * 20) / 20
    while tick <= xmax + 1e-9:
        xticks.append(round(tick, 2))
        tick += 0.05
    for tick in xticks:
        x = sx(tick)
        parts.append(f'<line x1="{x:.2f}" y1="{margin_top}" x2="{x:.2f}" y2="{margin_top + plot_height}" class="grid"/>')
        parts.append(f'<text x="{x:.2f}" y="{margin_top + plot_height + 24}" text-anchor="middle" class="tick">{tick:.2f}</text>')

    if logy:
        yticks = []
        for base in [1, 2, 5]:
            for exp in range(1, 5):
                value = base * (10**exp)
                if ymin <= value <= ymax:
                    yticks.append(value)
        yticks = sorted(set(yticks))
    else:
        step = 250 if ymax < 2500 else 500
        yticks = list(range(0, int(math.ceil(ymax / step)) * step + 1, step))
    for tick in yticks:
        if tick < ymin or tick > ymax:
            continue
        y = sy(max(tick, 1e-9))
        parts.append(f'<line x1="{margin_left}" y1="{y:.2f}" x2="{margin_left + plot_width}" y2="{y:.2f}" class="grid"/>')
        parts.append(f'<text x="{margin_left - 12}" y="{y + 4:.2f}" text-anchor="end" class="tick">{tick:g}</text>')

    parts.append(f'<line x1="{margin_left}" y1="{margin_top + plot_height}" x2="{margin_left + plot_width}" y2="{margin_top + plot_height}" class="axis"/>')
    parts.append(f'<line x1="{margin_left}" y1="{margin_top}" x2="{margin_left}" y2="{margin_top + plot_height}" class="axis"/>')
    parts.append(f'<text x="{margin_left + plot_width / 2}" y="{height - 30}" text-anchor="middle" class="label">Recall</text>')
    parts.append(f'<text transform="translate(30,{margin_top + plot_height / 2}) rotate(-90)" text-anchor="middle" class="label">QPS (queries / second)</text>')

    for _, label, color, marker, rows in series:
        points = " ".join(f'{sx(row["Average_Recall"]):.2f},{sy(row["QPS"]):.2f}' for row in rows)
        parts.append(f'<polyline points="{points}" class="line" stroke="{color}"/>')
        for row in rows:
            x = sx(row["Average_Recall"])
            y = sy(row["QPS"])
            parts.append(marker_svg(marker, x, y, color))
            tooltip = (
                f'{label} L={row["Lsearch"]} Recall={row["Average_Recall"]:.4f} '
                f'QPS={row["QPS"]:.1f} DrawTime={row["Average_Time_ms"]:.2f}ms'
            )
            parts.append(f"<title>{html.escape(tooltip)}</title>")

    if legend_outside:
        legend_x = margin_left + plot_width + 38
        legend_y = margin_top + 20
    else:
        legend_x = margin_left + plot_width - 330
        legend_y = margin_top + 20
        parts.append(f'<rect x="{legend_x - 16}" y="{legend_y - 24}" width="330" height="86" fill="white" stroke="#ccc"/>')
    parts.append(f'<text x="{legend_x}" y="{legend_y - 20}" class="legend" font-weight="700">Methods</text>')
    for idx, (_, label, color, marker, _) in enumerate(series):
        y = legend_y + idx * 30
        parts.append(f'<line x1="{legend_x}" y1="{y}" x2="{legend_x + 34}" y2="{y}" stroke="{color}" stroke-width="3"/>')
        parts.append(marker_svg(marker, legend_x + 17, y, color))
        parts.append(f'<text x="{legend_x + 46}" y="{y + 5}" class="legend">{html.escape(label)}</text>')

    parts.append("</svg>")
    return "\n".join(parts)


def main() -> None:
    args = parse_args()
    methods = [parse_method_spec(spec) for spec in args.method] if args.method else DEFAULT_METHODS
    skip_first_methods = set(args.skip_first_lsearch_method or ())
    num_queries, _ = query_count(args.data_root, args.query_task, args.dataset)
    query_count_by_task = {args.query_task: num_queries}
    global_lsearch = None
    if args.lsearch_start is not None or args.lsearch_step is not None or args.lsearch_end is not None:
        if args.lsearch_start is None or args.lsearch_step is None or args.lsearch_end is None:
            raise ValueError("--lsearch-start, --lsearch-step, and --lsearch-end must be set together")
        if args.lsearch_step <= 0:
            raise ValueError("--lsearch-step must be positive")
        global_lsearch = LsearchSpec(args.lsearch_start, args.lsearch_step, args.lsearch_end)
    result_task = args.result_task or compose_result_task(args.query_task, global_lsearch)
    output_dir = args.output_dir or args.results_root / "plot" / result_task
    output_dir.mkdir(parents=True, exist_ok=True)

    combined = []
    series = []
    for method in methods:
        method_query_task = method.query_task or args.query_task
        if method_query_task not in query_count_by_task:
            query_count_by_task[method_query_task], _ = query_count(args.data_root, method_query_task, args.dataset)
        method_num_queries = query_count_by_task[method_query_task]
        method_result_task = method.result_task(result_task)
        method_lsearch = method.lsearch or (None if method.explicit_result_task else global_lsearch)
        result_dir = args.results_root / method.method_dir / method_result_task / "results"
        summary_path = result_dir / "search_time_summary.csv"
        if not summary_path.exists():
            if args.skip_missing:
                print(f"warning: skip missing summary: {summary_path}")
                continue
            raise FileNotFoundError(summary_path)
        rows = filter_lsearch_rows(read_summary(summary_path), method_lsearch)
        if method.method_dir in skip_first_methods:
            rows = skip_first_lsearch_row(rows)
        draw_rows = build_draw_rows(rows, method_num_queries, args.sort_mode)
        write_draw_csv(result_dir / "search_time_summary_draw.csv", draw_rows)
        for row in draw_rows:
            combined.append(
                {
                    "method": method.method_dir,
                    "label": method.label,
                    "query_task": method_query_task,
                    "result_task": method_result_task,
                    "num_queries": method_num_queries,
                    **row,
                }
            )
        series.append((method.method_dir, method.label, method.color, method.marker, draw_rows))

    if not series:
        raise ValueError("no method summaries were loaded")

    with (output_dir / "combined_qps_recall_draw.csv").open("w", newline="") as f:
        fields = ["method", "label", "query_task", "result_task", "num_queries", "Lsearch", "Average_Efs", "Average_Time_ms", "Average_Recall", "QPS"]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(combined)

    if args.threshold:
        thresholds = sorted(set(args.threshold))
    elif args.threshold_mode == "auto":
        thresholds = auto_thresholds(series, args.threshold_step)
    else:
        thresholds = [0.6, 0.7, 0.8, 0.85, 0.9, 0.95]
    summary_rows = []
    with (output_dir / "recall_threshold_summary_draw.csv").open("w", newline="") as f:
        fields = ["threshold", "method", "label", "best_Lsearch", "best_recall", "best_qps", "best_time_ms"]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for threshold in thresholds:
            for method, label, _, _, rows in series:
                eligible = [row for row in rows if row["Average_Recall"] >= threshold]
                record = {
                    "threshold": threshold,
                    "method": method,
                    "label": label,
                    "best_Lsearch": "",
                    "best_recall": "",
                    "best_qps": "",
                    "best_time_ms": "",
                }
                if eligible:
                    best = max(eligible, key=lambda row: row["QPS"])
                    record.update(
                        {
                            "best_Lsearch": best["Lsearch"],
                            "best_recall": best["Average_Recall"],
                            "best_qps": best["QPS"],
                            "best_time_ms": best["Average_Time_ms"],
                        }
                    )
                writer.writerow(record)
                summary_rows.append(record)

    for logy, name in [(False, "linear"), (True, "log")]:
        if logy and args.no_log:
            continue
        query_task_note = args.query_task if len(query_count_by_task) == 1 else "mixed; see combined CSV"
        svg = make_svg(series, query_task_note, result_task, args.dataset, num_queries, args.sort_mode, logy, args.legend_outside)
        (output_dir / f"qps_recall_draw_{name}.svg").write_text(svg)

    lines = [
        "# QPS-Recall Draw Summary",
        "",
        f"Data query task: `{args.query_task if len(query_count_by_task) == 1 else 'mixed; see combined CSV'}`",
        f"Result task: `{result_task}`",
        "",
        f"QPS formula: `{num_queries} * 1000 / Average_Time_ms`",
        "",
        f"Draw CSV sort mode: `{args.sort_mode}`",
        f"Recall threshold mode: `{args.threshold_mode}`",
        "",
        "## Best QPS at Recall Thresholds",
        "",
        "| Recall >= | Method | Lsearch | Recall | QPS | Draw Time ms |",
        "| ---: | --- | ---: | ---: | ---: | ---: |",
    ]
    for row in summary_rows:
        if row["best_Lsearch"] == "":
            lines.append(f"| {row['threshold']} | {row['label']} | - | - | - | - |")
        else:
            lines.append(
                f"| {row['threshold']} | {row['label']} | {row['best_Lsearch']} | "
                f"{float(row['best_recall']):.4f} | {float(row['best_qps']):.1f} | "
                f"{float(row['best_time_ms']):.2f} |"
            )
    (output_dir / "summary_draw.md").write_text("\n".join(lines) + "\n")
    print(f"wrote draw CSVs and SVGs to {output_dir}")


if __name__ == "__main__":
    main()
