#!/usr/bin/env python3
"""Draw separate three-panel comparisons for threads=10 and K=20.

Examples:
  python3 tools/pic-draw/plot_recall_qps_query_thread_k.py
  python3 tools/pic-draw/plot_recall_qps_query_thread_k.py --group threads10
  python3 tools/pic-draw/plot_recall_qps_query_thread_k.py --group k20

Layout, fonts, widths, markers, DPI, ticks and preprocessing follow
plot_recall_qps_block.py. The nine-algorithm labels/colors follow the existing
query-label-length comparison, as does the expanded legend header. Axis limits
adapt to the data. Actual query counts are read from query.bin.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
from pathlib import Path
from types import SimpleNamespace

import plot_recall_qps_block as reference
from plot_recall_qps_query_label_length import (
    ALGORITHM_RESULTS, ALGORITHM_STYLES, CURVE_DRAW_ORDER,
    preprocess_source, query_count, series_in_draw_order,
)
from recall_qps_png_renderer import render_png


# 同时控制 th=10 和 K=20 两张图：False 隐藏图例，True 恢复显示。
SHOW_LEGEND = False

# th=10 和 K=20 两张图的最小 Y 轴刻度文字上移量，单位为布局像素。
# 正数越大，文字越靠上；0 表示不移动，负数表示下移。
# 只移动最小刻度的文字，不改变刻度数值、坐标范围或曲线。
Y_MIN_TICK_SHIFT_UP = 8.0
# 两张图均隐藏这些 Y 轴刻度；10 为 Y 轴最小值时保留，1000 为最大值时保留。
Y_TICK_HIDE_VALUES = (10.0, 20.0, 100.0, 1000.0)


# =============================================================================
# 常用绘图参数：图例高度、位置、留白和标题位置
# =============================================================================
# 布局尺寸单位为布局像素；字号为 pt。默认值保留当前图形效果。
FIGURE_HEIGHT = None       # 整图最终高度；None 随面板和图例区域自动调整。
PLOT_HEIGHT = 216          # 曲线绘图区高度；调整标题留白时保持此值。
PANEL_MARGIN_TOP = 76      # 标题区域高度；减小使子图靠近数据集名/实验标签。
PANEL_MARGIN_BOTTOM = 62   # 子图底部留白；为 X 轴刻度和标题预留。
SUBPLOT_TITLE_OFFSET_Y = 36.0  # 数据集名和实验标签：正数下移，负数上移。

# 图例框本身的高度：主要由字号、列数、行间距和框内留白共同决定。
# LEGEND_AREA_HEIGHT 控制的是顶部预留区域，不会直接拉伸图例框。
LEGEND_AREA_HEIGHT = None  # 顶部预留高度；None 自动适配，也可指定如 160。
LEGEND_TOP = 10.0          # 图例整体位置；增大向下移动，减小向上移动。
LEGEND_X_OFFSET = 0.0      # 图例整体水平偏移；正数向右，负数向左。
LEGEND_COLUMNS = 5        # 每行列数；9 个算法设为 5 时分成两行。
LEGEND_FONT_SIZE = 25     # 图例字号；减小可同时缩小框的高度和宽度。
LEGEND_LABEL_SPACING = 0.5   # 图例行间距（字号倍数）；减小会压缩图例框高度。
LEGEND_COLUMN_SPACING = 1.5  # 图例列间距（字号倍数）；减小会压缩图例框宽度。
LEGEND_BORDER_PADDING = 0.45 # 图例框内留白（字号倍数）；减小使框更紧凑。
LEGEND_LINE_LENGTH = 2.0     # 图例示例线长度（字号倍数）。
LEGEND_GAP_AFTER_LINE = 0.5  # 示例线与文字间距（字号倍数）。
LEGEND_TITLE_GAP = 0.0       # 自动布局中，图例与数据集标题间的额外留白。

# 坐标轴标题位置：水平正数向右、负数向左；垂直正数向下、负数向上。
X_AXIS_TITLE_OFFSET_X = 0.0
X_AXIS_TITLE_OFFSET_Y = 10.0
Y_AXIS_TITLE_OFFSET_X = -30.0
Y_AXIS_TITLE_OFFSET_Y = 0.0


# =============================================================================
# 数据源配置
# =============================================================================
DATASET_ORDER = ("Amazon", "VImg", "Music")
QUERY_TASK = "query_minlen1_sel10to40pct"
GROUPS = {
    "threads10": dict(K=10, num_threads=10, suffix="_threads10_k10"),
    "k20": dict(K=20, num_threads=100, suffix="_threads100_k20"),
}
OUTPUT_NAMES = {
    "threads10": "amazon_vimg_music_threads10_k10_recall_qps.png",
    "k20": "amazon_vimg_music_threads100_k20_recall_qps.png",
}
# Automatic ranges and a nine-entry legend header were requested for these plots.
# The original reference ranges/layout remain available as explicit CLI options.
AXIS_LIMITS = "auto"
LEGEND_LAYOUT = "expanded"


def padded_qps_limits(values):
    """Enclose the panel's QPS with nearby 1/2/5 log ticks, rather than decades."""
    low, high = min(values) / 1.05, max(values) * 1.05
    first = math.floor(math.log(low, reference.Y_LOG_BASE)) - 1
    last = math.ceil(math.log(high, reference.Y_LOG_BASE)) + 1
    ticks = sorted(multiplier * reference.Y_LOG_BASE ** power
                   for power in range(first, last + 1)
                   for multiplier in reference.Y_TICK_MULTIPLIERS)
    return max(tick for tick in ticks if tick <= low), min(tick for tick in ticks if tick >= high)


def settings_for(group, axis_limits=AXIS_LIMITS, legend_layout=LEGEND_LAYOUT):
    settings = SimpleNamespace(**vars(reference))
    settings.SHOW_LEGEND = SHOW_LEGEND
    settings.DATASETS = []
    experiment_label = "th=10" if group == "threads10" else "K=20"
    for name in DATASET_ORDER:
        row = copy.deepcopy(next(d for d in reference.DATASETS if d["name"] == name))
        row.update(name=f"{name}\n{experiment_label}", dataset=name,
                   query_task=QUERY_TASK, num_queries=query_count(name, QUERY_TASK))
        if group == "k20" and name == "Music":
            row['show_y_min_tick'] = True  # 此子图始终显示 Y 轴最小刻度，包括 20。
        if axis_limits == "auto":
            row["x_limits"] = None
            row["y_limits"] = None
        settings.DATASETS.append(row)
    settings.ALGORITHM_RESULTS = [
        dict(algorithm, directory=algorithm["directory"] + GROUPS[group]["suffix"])
        for algorithm in ALGORITHM_RESULTS]
    settings.CURVE_STYLES = [ALGORITHM_STYLES[a["label"]] for a in settings.ALGORITHM_RESULTS]
    settings.CURVE_DRAW_ORDER = CURVE_DRAW_ORDER
    settings.series_in_draw_order = series_in_draw_order
    settings.X_AXIS_TITLE = f"Recall@{GROUPS[group]['K']}"
    settings.padded_log_limits = padded_qps_limits
    settings.SUBPLOT_TITLE_OFFSET_Y = SUBPLOT_TITLE_OFFSET_Y
    settings.PANEL_MARGIN_TOP = PANEL_MARGIN_TOP
    settings.PANEL_MARGIN_BOTTOM = PANEL_MARGIN_BOTTOM
    settings.PANEL_HEIGHT = PLOT_HEIGHT + PANEL_MARGIN_TOP + PANEL_MARGIN_BOTTOM
    settings.X_AXIS_TITLE_OFFSET_X = X_AXIS_TITLE_OFFSET_X
    settings.X_AXIS_TITLE_OFFSET_Y = X_AXIS_TITLE_OFFSET_Y
    settings.Y_AXIS_TITLE_OFFSET_X = Y_AXIS_TITLE_OFFSET_X
    settings.Y_AXIS_TITLE_OFFSET_Y = Y_AXIS_TITLE_OFFSET_Y
    if not SHOW_LEGEND:
        settings.LEGEND_AREA_HEIGHT = 0
    elif legend_layout == "expanded":
        for key in ("LEGEND_TOP", "LEGEND_X_OFFSET", "LEGEND_COLUMNS", "LEGEND_FONT_SIZE",
                    "LEGEND_LABEL_SPACING", "LEGEND_COLUMN_SPACING", "LEGEND_BORDER_PADDING",
                    "LEGEND_LINE_LENGTH", "LEGEND_GAP_AFTER_LINE"):
            setattr(settings, key, globals()[key])
        rows = math.ceil(len(settings.ALGORITHM_RESULTS) / settings.LEGEND_COLUMNS)
        font_pixels = settings.LEGEND_FONT_SIZE * settings.LAYOUT_DPI / 72
        height = font_pixels * (rows + (rows - 1) * settings.LEGEND_LABEL_SPACING
                                + 2 * settings.LEGEND_BORDER_PADDING + 0.5)
        title_pixels = settings.SUBPLOT_TITLE_FONT_SIZE * settings.LAYOUT_DPI / 72
        automatic_header = max(settings.LEGEND_AREA_HEIGHT,
            math.ceil(height + max(0, title_pixels - 27) + max(0, settings.LEGEND_TOP)
                      + LEGEND_TITLE_GAP))
        settings.LEGEND_AREA_HEIGHT = automatic_header if LEGEND_AREA_HEIGHT is None else LEGEND_AREA_HEIGHT
    automatic_height = reference.FIGURE_HEIGHT
    if automatic_height is not None:
        automatic_height += (settings.PANEL_HEIGHT - reference.PANEL_HEIGHT
                             + settings.LEGEND_AREA_HEIGHT - reference.LEGEND_AREA_HEIGHT)
    settings.FIGURE_HEIGHT = automatic_height if FIGURE_HEIGHT is None else FIGURE_HEIGHT
    return settings


def summary_paths(dataset, algorithm):
    folder = (Path(dataset["result_root"]) / algorithm["directory"]
              / f"{dataset['query_task']}_{algorithm['step_info']}" / "results")
    return folder / reference.SOURCE_SUMMARY_NAME, folder / reference.PROCESSED_SUMMARY_NAME


def configure_y_ticks(settings, dataset, y_limits):
    """Hide selected ticks, retain 10/1000 at axis bounds, and raise minimum text."""
    excluded = [value for value in dataset.get('y_tick_exclude', ())
                if not any(math.isclose(value, bound) for bound in (10.0, 1000.0))]
    for value in Y_TICK_HIDE_VALUES:
        if math.isclose(value, 10.0) and math.isclose(y_limits[0], 10.0):
            continue
        if math.isclose(value, 1000.0) and math.isclose(y_limits[1], 1000.0):
            continue
        excluded.append(value)
    if dataset.get('show_y_min_tick', False):
        excluded = [value for value in excluded if not math.isclose(value, y_limits[0])]
    dataset['y_tick_exclude'] = tuple(dict.fromkeys(excluded))
    ticks = sorted(tick for tick in settings.log_ticks(*y_limits)
                   if not any(math.isclose(tick, value) for value in excluded))
    if ticks:
        offsets = dict(dataset.get('y_tick_label_offsets', {}))
        # The shared renderer uses negative layout offsets for upward motion.
        offsets[ticks[0]] = -Y_MIN_TICK_SHIFT_UP
        dataset['y_tick_label_offsets'] = offsets


def draw_group(group, settings, args):
    should_preprocess = reference.PREPROCESS_BEFORE_PLOTTING and not args.skip_preprocess
    generated = reused = row_count = 0
    if should_preprocess:
        for dataset in settings.DATASETS:
            for algorithm in settings.ALGORITHM_RESULTS:
                source, destination = summary_paths(dataset, algorithm)
                count = preprocess_source(source, destination,
                                          reference.FORCE_PREPROCESS or args.force_preprocess)
                generated += count > 0
                reused += count == 0
                row_count += count
        print(f"[{group}] Preprocessing: generated {generated}, reused {reused}, wrote {row_count} rows")
    if args.preprocess_only:
        return
    panels, inputs, panel_ranges = [], [], []
    for dataset in settings.DATASETS:
        curves = []
        for algorithm, style in zip(settings.ALGORITHM_RESULTS, settings.CURVE_STYLES):
            source, processed = summary_paths(dataset, algorithm)
            points = reference.read_points(processed, dataset["num_queries"])
            curves.append(reference.Series(algorithm["label"], points, style))
            inputs.append(dict(dataset=dataset["dataset"], query_task=QUERY_TASK,
                               num_queries=dataset["num_queries"], algorithm=algorithm["label"],
                               source=str(source), processed=str(processed), point_count=len(points)))
        panels.append(curves)
        qps = [point.qps for curve in curves for point in curve.points]
        y_limits = dataset.get('y_limits') or settings.padded_log_limits(qps)
        configure_y_ticks(settings, dataset, y_limits)
        panel_ranges.append(dict(dataset=dataset['dataset'], qps_min=min(qps), qps_max=max(qps),
                                 y_limits=y_limits))
    output = args.output_dir.expanduser().resolve() / OUTPUT_NAMES[group]
    width, height = render_png(settings, panels, output)
    output.with_suffix('.inputs.json').write_text(json.dumps(dict(
        group=group, K=GROUPS[group]['K'], num_threads=GROUPS[group]['num_threads'],
        axis_limits=args.axis_limits, legend_layout=args.legend_layout,
        panels=panel_ranges, inputs=inputs), indent=2) + '\n')
    print(f"[{group}] Wrote {output}: {width}x{height}, {settings.PNG_DPI} DPI, "
          f"{len(panels)} panels, {len(inputs)} curves, {sum(row['point_count'] for row in inputs)} points")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--group', choices=('both', *GROUPS), default='both')
    parser.add_argument('--output-dir', type=Path, default=reference.OUTPUT_DIR)
    parser.add_argument('--axis-limits', choices=('reference', 'auto'), default=AXIS_LIMITS)
    parser.add_argument('--legend-layout', choices=('reference', 'expanded'), default=LEGEND_LAYOUT)
    parser.add_argument('--skip-preprocess', action='store_true')
    parser.add_argument('--force-preprocess', action='store_true')
    parser.add_argument('--preprocess-only', action='store_true')
    args = parser.parse_args()
    if args.preprocess_only and (args.skip_preprocess or not reference.PREPROCESS_BEFORE_PLOTTING):
        raise ValueError('--preprocess-only requires preprocessing')
    groups = list(GROUPS) if args.group == 'both' else [args.group]
    settings = {g: settings_for(g, args.axis_limits, args.legend_layout) for g in groups}
    # Check every selected source before writing any processed file or image.
    for config in settings.values():
        for dataset in config.DATASETS:
            for algorithm in config.ALGORITHM_RESULTS:
                source, processed = summary_paths(dataset, algorithm)
                required = processed if args.skip_preprocess else source
                if not required.is_file():
                    raise FileNotFoundError(f'Missing experiment summary: {required}')
    for group, config in settings.items():
        draw_group(group, config, args)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
