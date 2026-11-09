#!/usr/bin/env python3
"""Plot short-label queries above long-label queries in a single 2x3 PNG.

Usage:
  python3 tools/pic-draw/plot_recall_qps_query_label_length.py
  python3 tools/pic-draw/plot_recall_qps_query_label_length.py --task short
  python3 tools/pic-draw/plot_recall_qps_query_label_length.py --skip-preprocess

Columns are Amazon, VImg, Music. Titles show the query-label count interval.
Recall limits are padded from the data; logarithmic QPS limits and ticks use
powers of ten, matching plot_recall_qps_all_datasets.py. Fonts and curve sizes
are inherited from plot_recall_qps_block.py.
"""

from __future__ import annotations

import argparse
import csv
import math
import struct
from pathlib import Path
from types import SimpleNamespace

import plot_recall_qps_block as reference
from preprocess_search_results import OUTPUT_COLUMNS
from recall_qps_png_renderer import render_png


# =============================================================================
# 常用绘图参数：整图高度、图例高度及 X/Y 轴标题位置
# =============================================================================
# 位置、留白使用布局像素；水平正数向右、负数向左，垂直正数向下、负数向上。
FIGURE_HEIGHT = None     # 整图高度；None 根据面板自动计算，也可手动指定。
PLOT_HEIGHT = 216        # 曲线绘图区高度，恢复缩小前的高度；调整留白时保持此值。
PANEL_MARGIN_TOP = 80    # 面板顶部标题区域；减小只压缩标题附近留白。
PANEL_MARGIN_BOTTOM = 58 # 面板底部坐标文字区域；减小只压缩底部留白。
SUBPLOT_GAP_Y = 6        # 两排面板的垂直间距。

LEGEND_AREA_HEIGHT = None  # 顶部图例区域高度；None 自动计算，也可指定如 160。
LEGEND_TOP = 10.0          # 图例框垂直位置；增大向下移动，减小向上移动。
LEGEND_X_OFFSET = 0.0     # 图例框水平偏移；正数右移，负数左移。
LEGEND_LABEL_SPACING = 0.5  # 图例行间距，单位为字号倍数；增大可增加图例框高度。
LEGEND_TITLE_GAP = 0.0    # 自动布局中，图例与首排标题之间的额外留白。

X_AXIS_TITLE_OFFSET_X = 0.0   # Recall@10 标题：正数右移，负数左移。
X_AXIS_TITLE_OFFSET_Y = 15.0  # Recall@10 标题：正数下移，负数上移。
Y_AXIS_TITLE_OFFSET_X = -30.0 # QPS 标题：正数右移，负数左移。
Y_AXIS_TITLE_OFFSET_Y = 0.0   # QPS 标题：正数下移，负数上移。

# 已有 draw 文件默认直接使用，保留手动修改；缺失时才从 summary 生成。
PREPROCESS_BEFORE_PLOTTING = True  # False 时完全跳过预处理，也可用 --skip-preprocess。
FORCE_PREPROCESS = False          # True 或 --force-preprocess 会覆盖已有 draw 文件。


# =============================================================================
# 数据源配置；字体、线宽、点大小及 DPI 沿用参考脚本。
# =============================================================================
DATA_ROOT = Path("/home/dev/graphdb/FilterVectorData")
DATASET_ORDER = ("Amazon", "VImg", "Music")
QUERY_TASKS = {
    "short": "query_len1to3_sel0to40pct",
    "long": "query_len4to6_sel0to40pct",
}
OUTPUT_NAMES = {
    "both": "amazon_vimg_music_query_label_length_recall_qps.png",
    "short": "amazon_vimg_music_short_labels_recall_qps.png",
    "long": "amazon_vimg_music_long_labels_recall_qps.png",
}
LABEL_RANGES = {"short": (1, 3), "long": (4, 6)}
# 与 plot_recall_qps_all_datasets.py 一致：每个子图按自己的数据自动定范围。
X_LIMITS = None
Y_LIMITS = None
# 为数据集名和标签范围两行标题预留空间。
TITLE_EXTRA_SPACE = 36

# step_info 必须精确匹配本次实验的结果目录，不回退到旧查询任务。
# UNG 使用与现有九算法对比图一致的 10000..100000 搜索范围。
# 图例顺序和配色沿用现有九算法对比图，ACORN 使用其算法名称。
ALGORITHM_RESULTS = [
    {"label": "Milvus", "directory": "milvus_hnsw", "step_info": "1000_1000_5000"},
    {"label": "ACORN", "directory": "ACORN", "step_info": "1000_1000_20000"},
    {"label": "NaviX", "directory": "NaviX", "step_info": "1000_1000_20000"},
    {"label": "RWalks", "directory": "rwalks", "step_info": "1000_1000_20000"},
    {"label": "PathSeer", "directory": "pathseer", "step_info": "1000_1000_20000"},
    {"label": "Curator", "directory": "Curator", "step_info": "search_ef64_search_ef10240"},
    {"label": "UNG", "directory": "UNG", "step_info": "10000_10000_100000"},
    {"label": "FAVOR", "directory": "favor", "step_info": "1000_1000_20000"},
    {"label": "PTNC", "directory": "Trie_block_500_M_32_cross_block_4", "step_info": "100_100_20000"},
]
ALGORITHM_STYLES = {
    "PTNC": {"color": "#f57c00", "dash": "", "marker": "square"},
    "UNG": {"color": "#2e7d32", "dash": "", "marker": "triangle"},
    "ACORN": {"color": "#c6bf3a", "dash": "", "marker": "diamond"},
    "Curator": {"color": "#00acc1", "dash": "", "marker": "circle"},
    "FAVOR": {"color": "#d32f2f", "dash": "", "marker": "pentagon"},
    "Milvus": {"color": "#1565c0", "dash": "", "marker": "star"},
    "NaviX": {"color": "#8d6e63", "dash": "", "marker": "hexagon"},
    "PathSeer": {"color": "#5118c2f7", "dash": "", "marker": "cross"},
    "RWalks": {"color": "#455a64", "dash": "", "marker": "plus"},
}
CURVE_DRAW_ORDER = ("ACORN", "NaviX", "RWalks", "PathSeer", "Curator", "Milvus", "UNG", "FAVOR", "PTNC")


def series_in_draw_order(series):
    by_label = {curve.label: curve for curve in series}
    if len(by_label) != len(series) or set(by_label) != set(CURVE_DRAW_ORDER):
        raise ValueError("Each panel must contain all nine algorithms exactly once")
    return [by_label[label] for label in CURVE_DRAW_ORDER]


def query_count(dataset, task):
    path = DATA_ROOT / dataset / task / f"{dataset}_query.bin"
    with path.open("rb") as handle:
        header = handle.read(8)
    if len(header) != 8:
        raise ValueError(f"Invalid query bin header: {path}")
    count, dimension = struct.unpack("<II", header)
    if count <= 0 or path.stat().st_size != 8 + count * dimension * 4:
        raise ValueError(f"Invalid query bin size: {path}")
    return count


def log_ticks(low, high):
    """Use integral powers of ten, as in the all-datasets plotting script."""
    first = math.ceil(math.log(low, reference.Y_LOG_BASE) - 1e-12)
    last = math.floor(math.log(high, reference.Y_LOG_BASE) + 1e-12)
    return [reference.Y_LOG_BASE ** power for power in range(first, last + 1)]


def fmt_log_tick(value):
    power = round(math.log(value, reference.Y_LOG_BASE))
    return rf"$\mathbf{{10^{{{power}}}}}$"


def settings_for(groups):
    # The shared renderer lays panels out in list order: short row, then long row.
    settings = SimpleNamespace(**vars(reference))
    settings.DATASETS = []
    for group in groups:
        task = QUERY_TASKS[group]
        low, high = LABEL_RANGES[group]
        for dataset in DATASET_ORDER:
            settings.DATASETS.append(dict(
                name=dataset + "\n" + rf"$|L_q|\in[{low},{high}]$",
                dataset=dataset, group=group, query_task=task,
                result_root=reference.RESULT_ROOT / dataset / "results",
                num_queries=query_count(dataset, task)))
    settings.X_LIMITS = X_LIMITS
    settings.Y_LIMITS = Y_LIMITS
    settings.log_ticks = log_ticks
    settings.fmt_log_tick = fmt_log_tick
    settings.SUBPLOT_COLUMNS = len(DATASET_ORDER)
    settings.FIGURE_HEIGHT = FIGURE_HEIGHT
    # Derive total panel height from its fixed plotting area and adjustable
    # whitespace so reducing margins never squashes the actual curves.
    settings.PANEL_MARGIN_TOP = PANEL_MARGIN_TOP
    settings.PANEL_MARGIN_BOTTOM = PANEL_MARGIN_BOTTOM
    settings.PANEL_HEIGHT = PLOT_HEIGHT + PANEL_MARGIN_TOP + PANEL_MARGIN_BOTTOM
    settings.SUBPLOT_GAP_Y = SUBPLOT_GAP_Y
    settings.SUBPLOT_TITLE_OFFSET_Y += TITLE_EXTRA_SPACE
    settings.CANVAS_MARGIN_BOTTOM = max(settings.CANVAS_MARGIN_BOTTOM, 10)
    settings.TITLE_FORMAT = "({letter}) {dataset}"
    settings.ALGORITHM_RESULTS = ALGORITHM_RESULTS
    settings.CURVE_STYLES = [ALGORITHM_STYLES[row["label"]] for row in ALGORITHM_RESULTS]
    settings.CURVE_DRAW_ORDER = CURVE_DRAW_ORDER
    settings.series_in_draw_order = series_in_draw_order
    settings.LEGEND_TOP = LEGEND_TOP
    settings.LEGEND_X_OFFSET = LEGEND_X_OFFSET
    settings.LEGEND_LABEL_SPACING = LEGEND_LABEL_SPACING
    settings.X_AXIS_TITLE_OFFSET_X = X_AXIS_TITLE_OFFSET_X
    settings.X_AXIS_TITLE_OFFSET_Y = X_AXIS_TITLE_OFFSET_Y
    settings.Y_AXIS_TITLE_OFFSET_X = Y_AXIS_TITLE_OFFSET_X
    settings.Y_AXIS_TITLE_OFFSET_Y = Y_AXIS_TITLE_OFFSET_Y
    # Keep enough room for the shared nine-entry legend above the first row.
    rows = math.ceil(len(ALGORITHM_RESULTS) / settings.LEGEND_COLUMNS)
    font_pixels = settings.LEGEND_FONT_SIZE * settings.LAYOUT_DPI / 72
    title_pixels = settings.SUBPLOT_TITLE_FONT_SIZE * settings.LAYOUT_DPI / 72
    legend_height = font_pixels * (
        rows + (rows - 1) * settings.LEGEND_LABEL_SPACING
        + 2 * settings.LEGEND_BORDER_PADDING + 0.5
    )
    automatic_header = max(settings.LEGEND_AREA_HEIGHT,
                           math.ceil(legend_height + max(0, title_pixels - 27)
                                     + max(0, LEGEND_TOP) + LEGEND_TITLE_GAP))
    settings.LEGEND_AREA_HEIGHT = (automatic_header if LEGEND_AREA_HEIGHT is None
                                   else LEGEND_AREA_HEIGHT)
    return settings


def summary_paths(dataset, algorithm):
    folder = (Path(dataset["result_root"]) / algorithm["directory"]
              / f"{dataset['query_task']}_{algorithm['step_info']}" / "results")
    return folder / reference.SOURCE_SUMMARY_NAME, folder / reference.PROCESSED_SUMMARY_NAME


def preprocess_source(source, destination, force):
    """Keep existing draw files unless explicitly forced; generate missing ones."""
    if destination.is_file() and not force:
        return 0
    if not source.is_file():
        raise FileNotFoundError(f"Missing campaign summary: {source}")
    with source.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        fields = reader.fieldnames or []
        rows = list(reader)
    recall_column = "Average_Recall" if "Average_Recall" in fields else "Recall"
    required = {"Lsearch", reference.TIME_COLUMN, recall_column}
    if not required.issubset(fields):
        raise ValueError(f"{source}: missing columns {sorted(required.difference(fields))}")
    if len(rows) <= reference.DROP_FIRST_ROWS:
        raise ValueError(f"{source}: insufficient rows after dropping initial rows")
    retained = rows[reference.DROP_FIRST_ROWS:]
    times = []
    for row in retained:
        raw = row[reference.TIME_COLUMN]
        value = float(raw)
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"{source}: invalid positive time {raw!r}")
        times.append((value, raw))
    times.sort(key=lambda entry: entry[0])
    output = [dict(Lsearch=row["Lsearch"], Average_Efs=row.get("Average_Efs") or row["Lsearch"],
                   Average_Time_ms=raw_time, Average_Recall=row[recall_column])
              for row, (_, raw_time) in zip(retained, times)]
    with destination.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=OUTPUT_COLUMNS)
        writer.writeheader()
        writer.writerows(output)
    return len(output)


def load_series(settings):
    panels = []
    for dataset in settings.DATASETS:
        curves = []
        for algorithm in ALGORITHM_RESULTS:
            _, processed = summary_paths(dataset, algorithm)
            points = reference.read_points(processed, dataset["num_queries"])
            curves.append(reference.Series(algorithm["label"], points, ALGORITHM_STYLES[algorithm["label"]]))
        panels.append(curves)
    return panels


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--task", choices=("short", "long", "both"), default="both")
    parser.add_argument("--output-dir", type=Path, default=reference.OUTPUT_DIR)
    parser.add_argument("--skip-preprocess", action="store_true")
    parser.add_argument("--force-preprocess", action="store_true")
    parser.add_argument("--preprocess-only", action="store_true")
    args = parser.parse_args()
    should_preprocess = PREPROCESS_BEFORE_PLOTTING and not args.skip_preprocess
    if args.preprocess_only and not should_preprocess:
        raise ValueError("--preprocess-only requires preprocessing")
    groups = list(QUERY_TASKS) if args.task == "both" else [args.task]
    # Check every source before generating any processed CSV or image.
    settings = settings_for(groups)
    for dataset in settings.DATASETS:
        for algorithm in ALGORITHM_RESULTS:
            source, _ = summary_paths(dataset, algorithm)
            if not source.is_file():
                raise FileNotFoundError(f"Missing campaign summary: {source}")
    if should_preprocess:
        generated = reused = row_count = 0
        for dataset in settings.DATASETS:
            for algorithm in ALGORITHM_RESULTS:
                source, destination = summary_paths(dataset, algorithm)
                count = preprocess_source(source, destination, FORCE_PREPROCESS or args.force_preprocess)
                generated += count > 0
                reused += count == 0
                row_count += count
        print(f"Generated {generated}, reused {reused} processed CSVs; wrote {row_count} rows")
    if args.preprocess_only:
        return 0
    all_series = load_series(settings)
    output = args.output_dir.expanduser().resolve() / OUTPUT_NAMES[args.task]
    width, height = render_png(settings, all_series, output)
    counts = ", ".join(f"{row['dataset']}/{row['group']}={row['num_queries']}" for row in settings.DATASETS)
    print(f"Wrote {output} ({width}x{height}, {settings.PNG_DPI} DPI; query counts: {counts})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
