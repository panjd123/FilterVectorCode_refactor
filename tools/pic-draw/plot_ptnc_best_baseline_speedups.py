#!/usr/bin/env python3
"""Draw option C: PTNC speedups over each task's fastest qualifying baseline.

By default, read current drawing CSVs through plot_recall_qps_all_datasets.py.
Use --input to plot an existing best_baseline_details.csv snapshot instead.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


# =============================================================================
# 常用配置：颜色放在最前面；修改这里即可调整图片，无需修改下方绘制函数。
# =============================================================================

# ------------------------------- 颜色 -------------------------------------
SELECTIVITY_STYLES = {
    "low": {"label": "Low", "color": "#2e7d32"},       # C 图低选择率：绿色。
    "medium": {"label": "Medium", "color": "#0aaabd"}, # C 图中选择率：青色。
    "high": {"label": "High", "color": "#ff8200"},     # C 图高选择率：橙色。
}
MARKER_EDGE_COLOR = "#000000"       # C 图数据点的白色外框。
LEGEND_MARKER_EDGE_COLOR = "#000000" # 图例中点和色块的白色外框。
LEGEND_RECALL_COLOR = "#333333"      # 图例中表示 recall 的点颜色。
CANVAS_BACKGROUND = "#ffffff"       # 整张图的背景色。
PLOT_BACKGROUND = "#fafafa"         # C 图坐标区背景色。
AXIS_COLOR = "#222222"              # C 图坐标轴和文字颜色。
GRID_COLOR = "#c9c9c9"              # 网格线颜色。
SEPARATOR_COLOR = "#dcdcdc"         # C 图数据集之间的竖向分隔线。
BASELINE_COLOR = "#a04329"          # C 图 1× 持平虚线颜色。
LEGEND_BACKGROUND_COLOR = "#ffffff" # 图例背景色。
LEGEND_BORDER_COLOR = "#b8b8b8"     # 图例外框颜色。

# ------------------------ C 图：点大小与线粗细 ----------------------------
MARKER_SIZE = 15.0           # 点直径，单位 pt；增大后圆点和三角形同时变大。
MARKER_EDGE_WIDTH = 1.2      # 数据点白框线宽，单位 pt；加粗以便缩放后仍清楚，设为 0 可取消。
MARKER_ALPHA = 1.0           # 数据点透明度，0 为透明、1 为不透明。
RECALL_MARKERS = ("o", "^") # 两个 recall 的点形：圆形、三角形。
LINE_WIDTH = 5.0            # 同一任务的两个 recall 点之间的连线粗细，单位 pt。
LINE_STYLE = "-"            # "-" 实线，"--" 虚线。
LINE_ALPHA = 0.8            # 连线透明度。
AXIS_WIDTH = 0.9            # C 图坐标轴线粗细。
GRID_WIDTH = 0.6            # 网格线粗细。
SEPARATOR_WIDTH = 0.9       # C 图数据集竖向分隔线粗细。
BASELINE_WIDTH = 2.0        # C 图 1× 持平虚线粗细。

# -------------------------- C 图：图例位置和大小 --------------------------
LEGEND_LOCATION = "upper center" # 图例自身的对齐方式。
LEGEND_ANCHOR = (0.5, 1)       # 整张图归一化坐标；x 越大越右，y 越大越上。
LEGEND_FONT_SIZE = 35             # 图例文字字号，单位 pt。
LEGEND_FONT_WEIGHT = 600          # 图例字重：400 常规，700 粗体。
LEGEND_COLUMNS = 5               # 每行图例数量；减小可换行。
LEGEND_MARKER_SIZE = 15.0         # 图例中点和色块的直径，单位 pt。
LEGEND_MARKER_EDGE_WIDTH = 1.2    # 图例点和色块的白框线宽，与数据点一致。
LEGEND_COLUMN_SPACING = 1.2       # 图例列间距，单位为图例字号。
LEGEND_LABEL_SPACING = 0.5        # 多行图例的行间距。
LEGEND_LINE_LENGTH = 1.0          # 图例符号区域的长度。
LEGEND_GAP_AFTER_LINE = 0.4       # 图例符号与文字的间距。
LEGEND_BORDER_PADDING = 0.3       # 图例内容与外框之间的留白。
LEGEND_SHOW_FRAME = False        # True：显示图例外框和背景。
LEGEND_BORDER_WIDTH = 0.8        # 图例外框粗细。

# -------------------------- C 图：画布、坐标和间距 ------------------------
FONT_FAMILY = "Liberation Serif" # 与 all_datasets 绘图脚本一致的 Times 风格字体。
FIGURE_WIDTH = 1800              # C 图画布宽度，单位为布局像素。
FIGURE_HEIGHT = 510              # C 图画布高度，单位为布局像素。
LAYOUT_DPI = 96                  # 布局像素到画布英寸的换算基准。
PANEL_MARGIN_LEFT = 155          # C 图左侧留白，单位为布局像素。
PANEL_MARGIN_RIGHT = 28
PANEL_MARGIN_TOP = 50           # 顶部留给图例的空间。
PANEL_MARGIN_BOTTOM = 125
DATASET_GAP = 4.0                # 相邻数据集中心距离，单位为横轴数据坐标。
SELECTIVITY_GAP = 1.0            # 同一数据集内三种选择率的水平间距。
RECALL_POINT_SPACING = 0.60      # 同一任务的两个 recall 点之间的水平间距。
X_AXIS_TITLE = ""                # 保留数据集名称，不显示 Dataset 标题。
Y_AXIS_TITLE = "Speedup"
X_AXIS_TITLE_FONT_SIZE = 30      # 横轴标题字号。
Y_AXIS_TITLE_FONT_SIZE = 35      # 纵轴标题字号。
Y_AXIS_TITLE_PAD = 3            # 纵轴标题距刻度文字的间距，单位 pt；越大越靠左。
Y_AXIS_TITLE_OFFSET_X = 10.0      # 标题水平偏移，单位为布局像素；正数向右，负数向左。
Y_AXIS_TITLE_OFFSET_Y = 0.0      # 标题垂直偏移，单位为布局像素；正数向下，负数向上。
# 例如：OFFSET_X=-20、OFFSET_Y=-10，会在默认位置基础上左移 20、上移 10。
X_TICK_FONT_SIZE = 35           # 横轴数据集名称字号。
Y_TICK_FONT_SIZE = 35            # 纵轴数字字号。
X_TICK_ROTATION = 0              # 横轴数据集名称的旋转角度。
Y_LIMITS = (0.65, 16.0)          # C 图纵轴范围；采用以 2 为底的对数轴。
Y_TICKS = (1, 2, 4, 8, 16)
Y_SHOW_MIN_TICK = True           # 自动标出纵轴下限。
Y_MIN_TICK_FORMAT = "{value:.1f}×" # 下限刻度保留一位小数。

# ----------------------------- 数据与输出 ----------------------------------
DATASETS = ("Genome", "Reviews", "Amazon", "VImg", "Music", "BReviews", "Tiktok", "Laion")
SELECTIVITIES = ("low", "medium", "high")
THRESHOLDS = (0.9, 0.95)
INPUT_CSV = None  # None：每次读取最新绘图 CSV；也可设为已有统计 CSV 的 Path。
OUTPUT_DIR = Path(__file__).resolve().parent / "pic" # 所有图文件保存在此目录。
OUTPUT_NAME = "speedup_paired_dots"                  # 输出文件名，不含扩展名。
OUTPUT_FORMATS = ("pdf", "svg", "png")
PNG_DPI = 400                    # PNG 分辨率；PDF/SVG 保持矢量。
SAVE_PADDING = 0.04             # 导出图片周围留白，单位英寸。

# ---------------------- 其他坐标、文字和图例配置 ---------------------------
QUERY_TASKS = {
    "low": "query_minlen1_sel0to10pct",
    "medium": "query_minlen1_sel10to40pct",
    "high": "query_minlen1_sel40to100pct",
}
X_DATA_PADDING = 0.55         # 首尾数据点与坐标区边缘的距离。
X_AXIS_TITLE_FONT_FAMILY = FONT_FAMILY
X_AXIS_TITLE_FONT_WEIGHT = 700
Y_AXIS_TITLE_FONT_FAMILY = FONT_FAMILY
Y_AXIS_TITLE_FONT_WEIGHT = 700
X_AXIS_TITLE_PAD = 14         # 轴标题与刻度文字的间距，单位 pt。
X_TICK_FONT_FAMILY = FONT_FAMILY
X_TICK_FONT_WEIGHT = 700
X_TICK_ALIGNMENT = "center"   # 旋转文字时可改为 "right"。
X_TICK_PAD = 10               # 刻度文字距坐标轴的间距，单位 pt。
Y_TICK_FONT_FAMILY = FONT_FAMILY
Y_TICK_FONT_WEIGHT = 700
Y_TICK_PAD = 8
TICK_LENGTH = 6
TICK_WIDTH = 0.9
Y_SCALE = "log"              # "log" 或 "linear"。
Y_LOG_BASE = 2
Y_TICK_FORMAT = "{value:g}×"  # 如需纯数字可改为 "{value:g}"。
X_TICK_LABEL_OVERRIDES = {}   # 如 {"BReviews": "BRev."}，只改显示名称。
RECALL_LEGEND_FORMAT = "Recall ≥ {threshold:g}"
SHOW_TOP_SPINE = False
SHOW_RIGHT_SPINE = False
SHOW_Y_GRID = True
GRID_DASH = (2, 2)           # 网格线：2 pt 实线 + 2 pt 间隔。
GRID_ALPHA = 0.65
SHOW_DATASET_SEPARATORS = True
SEPARATOR_STYLE = "-"
SEPARATOR_ALPHA = 0.8
SHOW_BASELINE = True
BASELINE_VALUE = 1.0
BASELINE_STYLE = "--"
LEGEND_FONT_FAMILY = FONT_FAMILY
LEGEND_BORDER_AXES_PAD = 0.0
LEGEND_COLOR_MARKER = "s"    # 选择率图例统一使用色块，点形图例单独表示 recall。
LEGEND_BACKGROUND_ALPHA = 1.0
LEGEND_ROUNDED_CORNERS = True


from matplotlib.font_manager import FontProperties
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator
from matplotlib.transforms import ScaledTranslation

import plot_recall_qps_all_datasets as source_plot


def load_current_speedups() -> tuple[list[dict], dict]:
    """Reuse the source plot's path resolution, QPS conversion and filters."""
    panels, missing = source_plot.load_all_series()
    by_task = {
        (str(dataset["dataset"]), str(dataset["query_task"])): {
            series.label: series.points for series in panel
        }
        for dataset, panel in zip(source_plot.DATASETS, panels)
    }
    records = []
    for dataset in DATASETS:
        for selectivity in SELECTIVITIES:
            task = QUERY_TASKS[selectivity]
            curves = by_task.get((dataset, task), {})
            for threshold in THRESHOLDS:
                qualified = {}
                for algorithm, points in curves.items():
                    point = max(
                        (point for point in points if point.recall >= threshold),
                        key=lambda point: (point.qps, point.recall), default=None,
                    )
                    if point is not None:
                        qualified[algorithm] = point
                ptnc = qualified.get("PTNC")
                baseline = max(
                    (label for label in qualified if label != "PTNC"),
                    key=lambda label: qualified[label].qps, default=None,
                )
                best = qualified[baseline] if baseline is not None else None
                valid = ptnc is not None and best is not None
                records.append({
                    "dataset": dataset, "query_task": task, "selectivity": selectivity,
                    "recall_threshold": threshold,
                    "status": "ok" if valid else ("PTNC_below_threshold_or_missing" if ptnc is None else "no_qualifying_baseline"),
                    "speedup": ptnc.qps / best.qps if valid else "NA",
                    "best_baseline": baseline if baseline is not None else "NA",
                    "ptnc_qps": ptnc.qps if ptnc else "NA",
                    "best_baseline_qps": best.qps if best else "NA",
                    "ptnc_selected_recall": ptnc.recall if ptnc else "NA",
                    "best_baseline_selected_recall": best.recall if best else "NA",
                })
    provenance = []
    for dataset in source_plot.DATASETS:
        if str(dataset["dataset"]) not in DATASETS or str(dataset["query_task"]) not in QUERY_TASKS.values():
            continue
        for algorithm in source_plot.ALGORITHM_RESULTS:
            source = source_plot.resolve_source(dataset, algorithm)
            drawing = source_plot.processed_path(source) if source is not None else None
            if drawing is not None and drawing.is_file():
                provenance.append({"drawing_csv": str(drawing), "sha256": hashlib.sha256(drawing.read_bytes()).hexdigest()})
    return records, {"source_plot": str(Path(source_plot.__file__).resolve()), "missing_curves": missing, "inputs": provenance}


def make_figure(records: list[dict]):
    plt.rcParams.update({
        "font.family": FONT_FAMILY, "text.color": AXIS_COLOR,
        "axes.labelcolor": AXIS_COLOR, "xtick.color": AXIS_COLOR,
        "ytick.color": AXIS_COLOR, "pdf.fonttype": 42, "ps.fonttype": 42,
        "svg.fonttype": "path", "savefig.bbox": None,
    })
    figure = plt.figure(figsize=(FIGURE_WIDTH / LAYOUT_DPI, FIGURE_HEIGHT / LAYOUT_DPI), facecolor=CANVAS_BACKGROUND)
    axis = figure.add_axes((
        PANEL_MARGIN_LEFT / FIGURE_WIDTH, PANEL_MARGIN_BOTTOM / FIGURE_HEIGHT,
        (FIGURE_WIDTH - PANEL_MARGIN_LEFT - PANEL_MARGIN_RIGHT) / FIGURE_WIDTH,
        (FIGURE_HEIGHT - PANEL_MARGIN_BOTTOM - PANEL_MARGIN_TOP) / FIGURE_HEIGHT,
    ), facecolor=PLOT_BACKGROUND)
    values = {(row["dataset"], row["selectivity"], row["recall_threshold"]): row["speedup"] for row in records}
    for i, dataset in enumerate(DATASETS):
        for j, selectivity in enumerate(SELECTIVITIES):
            center = i * DATASET_GAP + (j - (len(SELECTIVITIES) - 1) / 2) * SELECTIVITY_GAP
            xs = [center + (k - (len(THRESHOLDS) - 1) / 2) * RECALL_POINT_SPACING for k in range(len(THRESHOLDS))]
            ys = [values[dataset, selectivity, threshold] for threshold in THRESHOLDS]
            numeric_ys = [float(value) if value != "NA" else math.nan for value in ys]
            color = SELECTIVITY_STYLES[selectivity]["color"]
            axis.plot(xs, numeric_ys, color=color, linewidth=LINE_WIDTH, linestyle=LINE_STYLE, alpha=LINE_ALPHA, zorder=3)
            for k, (x, y) in enumerate(zip(xs, numeric_ys)):
                axis.plot([x], [y], linestyle="none", marker=RECALL_MARKERS[k],
                          markersize=MARKER_SIZE, markerfacecolor=color,
                          markeredgewidth=MARKER_EDGE_WIDTH, markeredgecolor=MARKER_EDGE_COLOR,
                          alpha=MARKER_ALPHA, zorder=4)
        if SHOW_DATASET_SEPARATORS and i < len(DATASETS) - 1:
            axis.axvline((i + 0.5) * DATASET_GAP, color=SEPARATOR_COLOR,
                         linewidth=SEPARATOR_WIDTH, linestyle=SEPARATOR_STYLE, alpha=SEPARATOR_ALPHA, zorder=0)
    if Y_SCALE == "log":
        axis.set_yscale("log", base=Y_LOG_BASE)
    else:
        axis.set_yscale(Y_SCALE)
    axis.set_ylim(*Y_LIMITS)
    half_group = (len(SELECTIVITIES) - 1) * SELECTIVITY_GAP / 2 + (len(THRESHOLDS) - 1) * RECALL_POINT_SPACING / 2
    axis.set_xlim(-half_group - X_DATA_PADDING, (len(DATASETS) - 1) * DATASET_GAP + half_group + X_DATA_PADDING)
    axis.set_xticks([i * DATASET_GAP for i in range(len(DATASETS))], [X_TICK_LABEL_OVERRIDES.get(name, name) for name in DATASETS])
    y_ticks = sorted(set((*Y_TICKS, Y_LIMITS[0]))) if Y_SHOW_MIN_TICK else Y_TICKS
    axis.yaxis.set_major_locator(FixedLocator(y_ticks))
    axis.yaxis.set_major_formatter(FuncFormatter(
        lambda value, _: (Y_MIN_TICK_FORMAT if Y_SHOW_MIN_TICK and math.isclose(value, Y_LIMITS[0])
                          else Y_TICK_FORMAT).format(value=value)
    ))
    axis.yaxis.set_minor_locator(NullLocator())
    axis.tick_params(axis="both", length=TICK_LENGTH, width=TICK_WIDTH)
    axis.tick_params(axis="x", pad=X_TICK_PAD)
    axis.tick_params(axis="y", pad=Y_TICK_PAD)
    for label in axis.get_xticklabels():
        label.set_fontproperties(FontProperties(family=X_TICK_FONT_FAMILY, size=X_TICK_FONT_SIZE, weight=X_TICK_FONT_WEIGHT))
        label.set_rotation(X_TICK_ROTATION)
        label.set_ha(X_TICK_ALIGNMENT)
    for label in axis.get_yticklabels():
        label.set_fontproperties(FontProperties(family=Y_TICK_FONT_FAMILY, size=Y_TICK_FONT_SIZE, weight=Y_TICK_FONT_WEIGHT))
    axis.set_xlabel(X_AXIS_TITLE, fontfamily=X_AXIS_TITLE_FONT_FAMILY, fontsize=X_AXIS_TITLE_FONT_SIZE, fontweight=X_AXIS_TITLE_FONT_WEIGHT, labelpad=X_AXIS_TITLE_PAD)
    axis.set_ylabel(Y_AXIS_TITLE, fontfamily=Y_AXIS_TITLE_FONT_FAMILY, fontsize=Y_AXIS_TITLE_FONT_SIZE, fontweight=Y_AXIS_TITLE_FONT_WEIGHT, labelpad=Y_AXIS_TITLE_PAD)
    # 在自动排版的位置上追加偏移；按布局 DPI 换算，确保 PNG/PDF/SVG 位移一致。
    axis.yaxis.label.set_transform(
        axis.yaxis.label.get_transform()
        + ScaledTranslation(Y_AXIS_TITLE_OFFSET_X / LAYOUT_DPI,
                            -Y_AXIS_TITLE_OFFSET_Y / LAYOUT_DPI,
                            figure.dpi_scale_trans)
    )
    for spine in axis.spines.values():
        spine.set_color(AXIS_COLOR)
        spine.set_linewidth(AXIS_WIDTH)
    axis.spines["top"].set_visible(SHOW_TOP_SPINE)
    axis.spines["right"].set_visible(SHOW_RIGHT_SPINE)
    if SHOW_Y_GRID:
        axis.grid(axis="y", color=GRID_COLOR, linewidth=GRID_WIDTH, linestyle=(0, GRID_DASH), alpha=GRID_ALPHA, zorder=0)
    if SHOW_BASELINE:
        axis.axhline(BASELINE_VALUE, color=BASELINE_COLOR, linewidth=BASELINE_WIDTH, linestyle=BASELINE_STYLE, zorder=2)
    handles = [Line2D([], [], linestyle="none", marker=LEGEND_COLOR_MARKER,
                      markerfacecolor=SELECTIVITY_STYLES[sel]["color"], markeredgecolor=LEGEND_MARKER_EDGE_COLOR,
                      markersize=LEGEND_MARKER_SIZE, markeredgewidth=LEGEND_MARKER_EDGE_WIDTH,
                      label=SELECTIVITY_STYLES[sel]["label"]) for sel in SELECTIVITIES]
    handles += [Line2D([], [], linestyle="none", marker=marker, color=LEGEND_RECALL_COLOR, markeredgecolor=LEGEND_MARKER_EDGE_COLOR,
                       markersize=LEGEND_MARKER_SIZE, markeredgewidth=LEGEND_MARKER_EDGE_WIDTH,
                       label=RECALL_LEGEND_FORMAT.format(threshold=threshold))
                for marker, threshold in zip(RECALL_MARKERS, THRESHOLDS)]
    legend = figure.legend(
        handles=handles, loc=LEGEND_LOCATION, bbox_to_anchor=LEGEND_ANCHOR,
        ncol=LEGEND_COLUMNS, prop=FontProperties(family=LEGEND_FONT_FAMILY, size=LEGEND_FONT_SIZE, weight=LEGEND_FONT_WEIGHT),
        columnspacing=LEGEND_COLUMN_SPACING, labelspacing=LEGEND_LABEL_SPACING,
        handlelength=LEGEND_LINE_LENGTH, handletextpad=LEGEND_GAP_AFTER_LINE,
        borderpad=LEGEND_BORDER_PADDING, borderaxespad=LEGEND_BORDER_AXES_PAD,
        frameon=LEGEND_SHOW_FRAME, fancybox=LEGEND_ROUNDED_CORNERS,
        facecolor=LEGEND_BACKGROUND_COLOR, edgecolor=LEGEND_BORDER_COLOR, framealpha=LEGEND_BACKGROUND_ALPHA,
    )
    legend.get_frame().set_linewidth(LEGEND_BORDER_WIDTH)
    return figure


def read_speedup_csv(path: Path) -> tuple[list[dict], dict]:
    """Optionally use an existing statistics snapshot instead of live inputs."""
    with path.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    records = [{**row, "recall_threshold": float(row["recall_threshold"]),
                "speedup": float(row["speedup"]) if row["status"] == "ok" else "NA"}
               for row in rows]
    return records, {"source_csv": str(path.resolve()),
                     "source_sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=INPUT_CSV,
                        help="Optional statistics CSV; default: recompute from current drawing CSVs.")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--output", type=Path,
                        help="Optional output filename; overrides --output-dir and OUTPUT_NAME.")
    args = parser.parse_args()
    if len(RECALL_MARKERS) != len(THRESHOLDS):
        raise ValueError("RECALL_MARKERS must contain one marker per recall threshold")
    if FIGURE_WIDTH <= PANEL_MARGIN_LEFT + PANEL_MARGIN_RIGHT or FIGURE_HEIGHT <= PANEL_MARGIN_TOP + PANEL_MARGIN_BOTTOM:
        raise ValueError("Panel margins must leave a positive plotting area")
    records, provenance = (load_current_speedups() if args.input is None
                           else read_speedup_csv(args.input))
    expected = {(dataset, sel, threshold) for dataset in DATASETS
                for sel in SELECTIVITIES for threshold in THRESHOLDS}
    actual = {(row["dataset"], row["selectivity"], row["recall_threshold"]) for row in records}
    if actual != expected or len(records) != len(expected):
        raise ValueError("Expected one comparison per configured dataset/selectivity/recall")
    if any(not math.isfinite(row["speedup"]) or row["speedup"] <= 0
           for row in records if row["status"] == "ok"):
        raise ValueError("Valid speedups must be finite and positive")
    figure = make_figure(records)
    output = (args.output if args.output is not None else args.output_dir / OUTPUT_NAME).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    for extension in OUTPUT_FORMATS:
        path = output.with_suffix(f".{extension}")
        figure.savefig(path, dpi=PNG_DPI, bbox_inches="tight", pad_inches=SAVE_PADDING,
                       facecolor=CANVAS_BACKGROUND)
        print(f"Wrote {path}")
    plt.close(figure)
    with output.with_suffix(".csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    provenance.update({
        "method": "PTNC max QPS with recall >= threshold / fastest qualifying baseline QPS; no interpolation",
        "comparisons": len(records), "figure_width_layout_px": FIGURE_WIDTH,
        "figure_height_layout_px": FIGURE_HEIGHT, "layout_dpi": LAYOUT_DPI, "png_dpi": PNG_DPI,
    })
    output.with_suffix(".json").write_text(json.dumps(provenance, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Compared {len(records)} tasks; {sum(row['status'] != 'ok' for row in records)} NA.")


if __name__ == "__main__":
    main()
