#!/usr/bin/env python3
"""Draw a 4x6 Recall@10-QPS comparison with Matplotlib.

The 24 panels cover eight datasets and three selectivity tasks. Missing result
CSVs are skipped, allowing partially completed experiments to be plotted.
"""

from __future__ import annotations

import argparse
import csv
import math
import struct
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

from preprocess_search_results import OUTPUT_COLUMNS, processed_output_is_valid
from recall_qps_png_renderer import render_png


# =============================================================================
# User configuration: datasets, tasks, and algorithm result directories
# =============================================================================

# ---------------------------- 顺序配置（常用） -----------------------------
# DATASET_ORDER：子图按此顺序从左到右、从上到下排列。
# 每个数据集占连续三张图（对应 QUERY_TASK_ORDER），每行放两个数据集。
DATASET_ORDER = (
    "Genome",
    "Reviews",
    "Amazon",
    "VImg",
    "Music",
    "BReviews",
    "Tiktok",
    "Laion",
)
# QUERY_TASK_ORDER：同一数据集内三个查询任务从左到右的排列顺序。
QUERY_TASK_ORDER = (
    "query_minlen1_sel0to10pct",
    "query_minlen1_sel10to40pct",
    "query_minlen1_sel40to100pct",
)

# LEGEND_ORDER：顶部图例从左到右的顺序，也决定算法结果的读取顺序。
# 它不控制曲线的上下覆盖关系；覆盖关系由 CURVE_DRAW_ORDER 单独设置。
LEGEND_ORDER = (
    
    "Milvus",
    "AOCRN",
    
    "NaviX",
    
    "RWalks","PathSeer",
    "Curator","UNG",
    "FAVOR",
    "PTNC",
)

# CURVE_DRAW_ORDER：曲线绘制顺序。越靠前越早画、位于底层；越靠后越晚画、
# 位于上层。曲线重叠严重时，可把希望突出显示的算法移到列表末尾。
CURVE_DRAW_ORDER = (
    
    "AOCRN",
    
    "NaviX",
    
    "RWalks","PathSeer",
    "Curator","Milvus","UNG",
    "FAVOR",
    "PTNC",
)

# ------------------------------ 数据路径 -----------------------------------
RESULT_ROOT = Path("/home/dev/graphdb/FilterVectorResult")  # 搜索结果根目录。
DATA_ROOT = Path("/home/dev/graphdb/FilterVectorData")      # 原始数据根目录。

# 标题选择率范围覆盖：(数据集, 查询任务) -> (最小值, 最大值)，单位为百分比。
# None 表示该端继续使用 profiled_*.csv 计算出的实际值。
SELECTION_RANGE_OVERRIDES = {
    ("Amazon", "query_minlen1_sel10to40pct"): (None, None),
    ("BReviews", "query_minlen1_sel10to40pct"): (None, 39.7),
    ("Genome", "query_minlen1_sel10to40pct"): (None, 39.9),
    ("Laion", "query_minlen1_sel10to40pct"): (None, 39.2),
    ("Music", "query_minlen1_sel10to40pct"): (None, 39.5),
    ("Reviews", "query_minlen1_sel10to40pct"): (None, 39.8),
    ("Tiktok", "query_minlen1_sel10to40pct"): (None, 40.0),
    ("Genome", "query_minlen1_sel40to100pct"): (None, 83.5),
    ("Music", "query_minlen1_sel40to100pct"): (None, 80.0),
    ("Reviews", "query_minlen1_sel40to100pct"): (None, 81.2),
    ("Tiktok", "query_minlen1_sel40to100pct"): (None, 85.1),
    ("VImg", "query_minlen1_sel40to100pct"): (None, 83.4),
}


def selection_range_label(dataset: str, query_task: str) -> str:
    """Return the actual query selectivity range as a one-decimal math label."""
    base_path = DATA_ROOT / dataset / f"{dataset}_base.fvecs"
    profile_path = (
        DATA_ROOT
        / dataset
        / query_task
        / f"profiled_{query_task.removeprefix('query_')}.csv"
    )
    if not base_path.is_file() or not profile_path.is_file():
        return r"$\phi\ \mathrm{unavailable}$"

    with base_path.open("rb") as handle:
        raw_dimension = handle.read(4)
    if len(raw_dimension) != 4:
        raise ValueError(f"Invalid fvecs header: {base_path}")
    dimension = struct.unpack("<i", raw_dimension)[0]
    record_bytes = 4 * (dimension + 1)
    file_bytes = base_path.stat().st_size
    if dimension <= 0 or file_bytes % record_bytes:
        raise ValueError(f"Invalid fvecs size or dimension: {base_path}")
    num_base_vectors = file_bytes // record_bytes

    with profile_path.open(newline="", encoding="utf-8-sig") as handle:
        coverages = [
            int(row["coverage_count"])
            for row in csv.DictReader(handle)
        ]
    if not coverages:
        raise ValueError(f"Selection profile contains no rows: {profile_path}")
    low = min(coverages) * 100.0 / num_base_vectors
    high = max(coverages) * 100.0 / num_base_vectors
    override = SELECTION_RANGE_OVERRIDES.get((dataset, query_task))
    if override is not None:
        override_low, override_high = override
        low = low if override_low is None else override_low
        high = high if override_high is None else override_high
    left_bracket = "(" if query_task == "query_minlen1_sel0to10pct" else "["
    return rf"$\phi \in {left_bracket}{low:.1f}\%, {high:.1f}\%]$"


# Dataset-major order plus six columns puts two datasets (three tasks each) on
# every row: 8 datasets x 3 tasks = 24 panels = 4 rows x 6 columns.
DATASETS = [
    {
        "name": f"{dataset}\n{selection_range_label(dataset, query_task)}",
        "dataset": dataset,
        "query_task": query_task,
        "result_root": RESULT_ROOT / dataset / "results",
        "num_queries": 1000,
    }
    for dataset in DATASET_ORDER
    for query_task in QUERY_TASK_ORDER
]

# ---------------------------- 算法结果路径 ---------------------------------
# label：图例名称；directories：算法结果目录候选；suffixes：任务目录后缀候选，
# 按从前到后的优先级查找。allow_fallback=False 表示精确后缀不存在时直接跳过，
# 不再通配选择其他实验结果。PTNC 当前严格使用 100_100_20000。
_ALGORITHM_CONFIGS = [
    {
        "label": "PTNC",
        "directories": ("Trie_block_500_M_32_cross_block_4",),
        "suffixes": ("100_100_20000",),
        "allow_fallback": False,
    },
    {
        "label": "UNG",
        "directories": ("UNG",),
        "suffixes": ("10000_10000_100000", "1000_5000_31000"),
    },
    {
        # Preserve the requested legend spelling and accept existing ACORN names.
        "label": "AOCRN",
        "directories": ("ACORN",),
        "suffixes": ("1000_1000_20000",),
    },
    {
        "label": "Curator",
        "directories": ("Curator",),
        "suffixes": (
            "search_ef64_search_ef10240",
            "search_ef64_search_ef8192",
            "search_ef64_search_ef1024",
        ),
    },
    {
        "label": "FAVOR",
        "directories": ("favor", "FAVOR"),
        "suffixes": ("1000_1000_20000",),
    },
    {
        "label": "Milvus",
        "directories": ("milvus_hnsw", "Milvus"),
        "suffixes": ("1000_1000_5000", "1000_1000_20000"),
    },
    {
        "label": "NaviX",
        "directories": ("NaviX", "navix"),
        "suffixes": ("1000_1000_20000",),
    },
    {
        "label": "PathSeer",
        "directories": ("pathseer", "PathSeer"),
        "suffixes": ("1000_1000_20000",),
    },
    {
        "label": "RWalks",
        "directories": ("rwalks", "RWalks"),
        "suffixes": ("1000_1000_10000", "1000_1000_20000", "10_50_500"),
    },
]

_ALGORITHM_BY_LABEL = {
    str(algorithm["label"]): algorithm for algorithm in _ALGORITHM_CONFIGS
}
if set(LEGEND_ORDER) != set(_ALGORITHM_BY_LABEL):
    raise ValueError("LEGEND_ORDER must contain every configured algorithm exactly once")
if set(CURVE_DRAW_ORDER) != set(_ALGORITHM_BY_LABEL):
    raise ValueError("CURVE_DRAW_ORDER must contain every configured algorithm exactly once")
ALGORITHM_RESULTS = [_ALGORITHM_BY_LABEL[label] for label in LEGEND_ORDER]
# ------------------------------ 数据处理 -----------------------------------
SOURCE_SUMMARY_NAME = "search_time_summary.csv"          # 原始实验汇总文件。
PROCESSED_SUMMARY_NAME = "search_time_summary_draw.csv"  # 绘图专用文件。
RECALL_COLUMN = "Average_Recall"  # 横轴所用 Recall 字段。
TIME_COLUMN = "Average_Time_ms"   # 用于换算 QPS 的总耗时字段。
DROP_FIRST_ROWS = 1               # 预处理时丢弃开头的数据行数。
PREPROCESS_BEFORE_PLOTTING = True # True：绘图前自动生成绘图专用 CSV。
FORCE_PREPROCESS = False          # True：强制覆盖已存在的有效预处理文件。
SORT_POINTS_BY_RECALL = True      # True：按 Recall 从小到大连接曲线点。
# 单条曲线的最小 Recall（保留大于等于阈值的点）：
# (数据集, 查询任务, 算法) -> 最小 Recall。
MIN_RECALL_FILTERS = {
    ("Music", "query_minlen1_sel10to40pct", "PTNC"): 0.8,
    ("VImg", "query_minlen1_sel10to40pct", "PTNC"): 0.8,
}

# ------------------------------ 输出设置 -----------------------------------
OUTPUT_DIR = Path(__file__).resolve().parent / "pic"  # 默认图片目录。
OUTPUT_PATH = OUTPUT_DIR / "all_datasets_three_selectivities_recall_qps.png"
PNG_DPI = 400    # 输出分辨率；论文图片通常使用 300 或 600 DPI。
LAYOUT_DPI = 96  # 布局基准 DPI；通常不需要修改。


# =============================================================================
# Matplotlib layout and visual style
# =============================================================================

# FIGURE_WIDTH/HEIGHT=None 时根据面板尺寸自动计算整图大小。
FIGURE_WIDTH = 3100
FIGURE_HEIGHT = 1800
PANEL_WIDTH = 430     # 与 plot_recall_qps_block.py 保持一致。
PANEL_HEIGHT = 356    # 参考高度 330；额外 26 px 容纳第二行 phi 标题。
SUBPLOT_COLUMNS = 6   # 每行子图数；24 张图对应 4 行 6 列。
CANVAS_BACKGROUND = "#ffffff"  # 整张图片的背景色。
PLOT_BACKGROUND = "#fafafa"    # 每个坐标区内部的背景色。

# 面板内边距决定实际坐标区大小；顶部需容纳数据集名和 phi 范围两行标题。
PANEL_MARGIN_LEFT = 75    # 与参考脚本一致，给 25 pt 的 Y 刻度和 QPS 留空间。
PANEL_MARGIN_RIGHT = 6    # 与参考脚本一致。
PANEL_MARGIN_TOP = 78     # 参考值 52；额外 26 px 容纳第二行 phi 标题。
PANEL_MARGIN_BOTTOM = 62  # 与参考脚本一致，容纳 25 pt 的 X 轴文字。
CANVAS_MARGIN_LEFT = 30   # 与参考脚本一致。
CANVAS_MARGIN_RIGHT = 12  # 与参考脚本一致。
CANVAS_MARGIN_BOTTOM = 10   # 整图底部留白，避免最后一行 Recall@10 被裁切。
LEGEND_AREA_HEIGHT = 110  # 九项 25 pt 图例需额外空间，避免遮挡第一行双行标题。
SUBPLOT_GAP_X = 18        # 与参考脚本一致。
SUBPLOT_GAP_Y = 22        # 与参考脚本一致。

# 字号单位均为 pt；字体配置分别控制标题、轴标题、刻度和图例。
# 字重常用值：400=常规，600=半粗，700=粗体。
FONT_FAMILY = "Liberation Serif"              # 全局默认字体，兼容 Times 风格。
SUBPLOT_TITLE_FONT_FAMILY = FONT_FAMILY       # 数据集名、小图序号和 phi 范围字体。
SUBPLOT_TITLE_FONT_SIZE = 30                  # 双行子图标题字号，单位 pt。
SUBPLOT_TITLE_FONT_WEIGHT = 700               # 双行子图标题字重。
X_AXIS_TITLE_FONT_FAMILY = FONT_FAMILY        # X 轴标题 Recall@10 的字体。
X_AXIS_TITLE_FONT_SIZE = 30                   # X 轴标题字号，单位 pt。
X_AXIS_TITLE_FONT_WEIGHT = 700                # X 轴标题字重。
Y_AXIS_TITLE_FONT_FAMILY = FONT_FAMILY        # Y 轴标题 QPS 的字体。
Y_AXIS_TITLE_FONT_SIZE = 30                   # Y 轴标题字号，单位 pt。
Y_AXIS_TITLE_FONT_WEIGHT = 700                # Y 轴标题字重。
Y_AXIS_TITLE_FIRST_COLUMN_ONLY = True         # True：QPS 只显示在每行最左侧子图。
X_TICK_FONT_FAMILY = FONT_FAMILY              # X 轴刻度数字字体。
X_TICK_FONT_SIZE = 30                         # X 轴刻度字号，单位 pt。
X_TICK_FONT_WEIGHT = 700                      # X 轴刻度使用粗体。
Y_TICK_FONT_FAMILY = FONT_FAMILY              # Y 轴刻度数字字体。
Y_TICK_FONT_SIZE = 30                         # Y 轴刻度字号，单位 pt。
Y_TICK_FONT_WEIGHT = 700                      # Y 轴刻度使用粗体。
LEGEND_FONT_FAMILY = FONT_FAMILY              # 顶部共享图例字体。
LEGEND_FONT_SIZE = 30                         # 图例字号，单位 pt。
LEGEND_FONT_WEIGHT = 600                      # 图例字重。
TITLE_FORMAT = "({letter}) {dataset}"         # 标题格式；dataset 内含换行和 phi 范围。

# 文字偏移单位为布局像素：X 正数向右；Y 正数向下、负数向上。
SUBPLOT_TITLE_OFFSET_Y = 30.0   # 数据集名和 phi 范围整体下移；继续增大可进一步下移。
X_AXIS_TITLE_OFFSET_X = 0.0     # 与参考脚本一致。
X_AXIS_TITLE_OFFSET_Y = 20.0     # 与参考脚本一致。
Y_AXIS_TITLE_OFFSET_X = -30.0   # 与参考脚本一致。
Y_AXIS_TITLE_OFFSET_Y = 0.0     # 与参考脚本一致。

# None 表示有数据的面板自动确定范围；EMPTY_* 仅用于完全无曲线的面板。
X_LIMITS = None
Y_LIMITS = None
# 单个子图的 Y 轴范围覆盖：(数据集, 查询任务) -> (下限, 上限)。
# None 表示该端继续使用数据自动计算出的 10 的整数次幂边界。
Y_LIMIT_OVERRIDES = {
    ("Genome", "query_minlen1_sel40to100pct"): (None, 1.2e4),
}
EMPTY_X_LIMITS = (0.5, 1.0)      # 无数据面板的 Recall 范围。
EMPTY_Y_LIMITS = (100.0, 1000.0) # 无数据面板的 QPS 范围。
X_TICK_COUNT = 5   # 横轴期望刻度数量。
Y_LOG_BASE = 10    # QPS 纵轴固定为以 10 为底，只标注 10 的整数次方。
AXIS_COLOR = "#222222"  # 坐标轴、刻度和文字颜色。
GRID_COLOR = "#c9c9c9"  # 主网格线颜色。
GRID_WIDTH = 0.6        # 网格线宽，单位 pt。
GRID_DASH = "2,2"       # 网格虚线：2 pt 线段 + 2 pt 间隔。
AXIS_WIDTH = 0.9        # 左、下坐标轴线宽，单位 pt。

# 图例内容顺序由脚本顶部 LEGEND_ORDER 控制；这里控制排版和边框。
LEGEND_COLUMNS = 9             # 图例列数；减小后会自动换行。
LEGEND_X_OFFSET = 0.0          # 与参考脚本一致。
LEGEND_TOP = 10.0               # 图例距整图顶部的位置；增大该值可继续下移。
LEGEND_COLUMN_SPACING = 1.5    # 与参考脚本一致。
LEGEND_LABEL_SPACING = 0.5     # 与参考脚本一致。
LEGEND_LINE_LENGTH = 2.0       # 与参考脚本一致。
LEGEND_GAP_AFTER_LINE = 0.5    # 与参考脚本一致。
LEGEND_BORDER_PADDING = 0.45   # 与参考脚本一致。
LEGEND_BACKGROUND_COLOR = "#ffffff" # 图例背景色。
LEGEND_BORDER_COLOR = "#b8b8b8"     # 图例边框颜色。
LEGEND_BORDER_WIDTH = 0.8      # 图例边框线宽；设为 0 可隐藏边框。
LEGEND_ROUNDED_CORNERS = True  # True 使用圆角边框。

# ---------------------------- 曲线与数据点 ---------------------------------
# 线宽和 marker 大小单位为 pt。dash="" 表示实线；例如 "6,2" 表示 6 pt
# 实线加 2 pt 间隔。九个算法默认使用不同 marker。
LINE_WIDTH = 6.0
MARKER_SIZE = 15.0
MARKER_EDGE_WIDTH = 0.8
MARKER_EDGE_COLOR = "#f2f2f2"
# MARKER_EDGE_COLOR = "auto"
# marker 可选：square/triangle/diamond/circle/pentagon/star/hexagon/cross/plus。
ALGORITHM_STYLES = {
    "PTNC": {"color": "#f57c00", "dash": "", "marker": "square"},
    "UNG": {"color": "#2e7d32", "dash": "", "marker": "triangle"},
    "AOCRN": {"color": "#c6bf3a", "dash": "", "marker": "diamond"},
    "Curator": {"color": "#00acc1", "dash": "", "marker": "circle"},
    "FAVOR": {"color": "#d32f2f", "dash": "", "marker": "pentagon"},
    "Milvus": {"color": "#1565c0", "dash": "", "marker": "star"},
    "NaviX": {"color": "#8d6e63", "dash": "", "marker": "hexagon"},
    "PathSeer": {"color": "#5118c2f7", "dash": "", "marker": "cross"},
    "RWalks": {"color": "#455a64", "dash": "", "marker": "plus"},
}
CURVE_STYLES = [ALGORITHM_STYLES[label] for label in LEGEND_ORDER]

@dataclass(frozen=True)
class Point:
    recall: float
    qps: float


@dataclass(frozen=True)
class Series:
    label: str
    points: tuple[Point, ...]
    style: dict[str, object]


def series_in_draw_order(series_list: Sequence[Series]) -> list[Series]:
    """Order only the curves available in this panel."""
    by_label = {series.label: series for series in series_list}
    if len(by_label) != len(series_list):
        raise ValueError("Series labels must be unique within each panel")
    return [by_label[label] for label in CURVE_DRAW_ORDER if label in by_label]


def resolve_source(
    dataset: dict[str, object], algorithm: dict[str, object]
) -> Path | None:
    """Resolve one source CSV, preferring configured run suffixes."""
    result_root = Path(dataset["result_root"])
    query_task = str(dataset["query_task"])
    directories = tuple(str(value) for value in algorithm["directories"])
    suffixes = tuple(str(value) for value in algorithm["suffixes"])

    for directory in directories:
        algorithm_root = result_root / directory
        for suffix in suffixes:
            source = (
                algorithm_root
                / f"{query_task}_{suffix}"
                / "results"
                / SOURCE_SUMMARY_NAME
            )
            if source.is_file():
                return source

    if not bool(algorithm.get("allow_fallback", True)):
        return None
    # Tolerate new sweep ranges while preferring explicit paths above.
    for directory in directories:
        matches = sorted(
            (result_root / directory).glob(
                f"{query_task}_*/results/{SOURCE_SUMMARY_NAME}"
            )
        )
        if matches:
            return matches[0]
    return None


def processed_path(source_path: Path) -> Path:
    return source_path.with_name(PROCESSED_SUMMARY_NAME)


def preprocess_source(input_path: Path, output_path: Path, *, force: bool) -> int:
    """Normalize UNG-style and baseline-style summaries for plotting."""
    if not force and processed_output_is_valid(
        output_path, time_column=TIME_COLUMN
    ):
        return 0

    with input_path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames or []
        rows = list(reader)

    recall_column = "Average_Recall" if "Average_Recall" in fieldnames else "Recall"
    missing = {"Lsearch", TIME_COLUMN, recall_column}.difference(fieldnames)
    if missing:
        names = ", ".join(sorted(missing))
        raise ValueError(f"{input_path}: missing required column(s): {names}")
    if len(rows) <= DROP_FIRST_ROWS:
        raise ValueError(f"{input_path}: too few data rows")

    retained = rows[DROP_FIRST_ROWS:]
    times: list[tuple[float, str]] = []
    for line_number, row in enumerate(retained, start=DROP_FIRST_ROWS + 2):
        raw_time = str(row[TIME_COLUMN])
        try:
            numeric_time = float(raw_time)
        except ValueError as exc:
            raise ValueError(
                f"{input_path}:{line_number}: invalid {TIME_COLUMN}: {raw_time!r}"
            ) from exc
        if not math.isfinite(numeric_time) or numeric_time <= 0:
            raise ValueError(f"{input_path}:{line_number}: invalid positive time")
        times.append((numeric_time, raw_time))
    times.sort(key=lambda item: item[0])

    output_rows = []
    for row, (_, raw_time) in zip(retained, times):
        output_rows.append(
            {
                "Lsearch": row["Lsearch"],
                "Average_Efs": row.get("Average_Efs") or row["Lsearch"],
                TIME_COLUMN: raw_time,
                RECALL_COLUMN: row[recall_column],
            }
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=OUTPUT_COLUMNS)
        writer.writeheader()
        writer.writerows(output_rows)
    return len(output_rows)


def preprocess_all(force: bool = FORCE_PREPROCESS) -> tuple[int, int, int, int]:
    """Preprocess every available source and count missing curves."""
    generated = reused = rows = missing = 0
    for dataset in DATASETS:
        for algorithm in ALGORITHM_RESULTS:
            source = resolve_source(dataset, algorithm)
            if source is None:
                missing += 1
                continue
            written_rows = preprocess_source(
                source,
                processed_path(source),
                force=force,
            )
            if written_rows:
                generated += 1
                rows += written_rows
            else:
                reused += 1
    return generated, reused, rows, missing


def read_points(csv_path: Path, num_queries: int) -> tuple[Point, ...]:
    """Read processed Recall/time values and calculate QPS."""
    points: list[Point] = []
    with csv_path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames or []
        missing = {RECALL_COLUMN, TIME_COLUMN}.difference(fieldnames)
        if missing:
            raise ValueError(
                f"{csv_path}: missing required column(s): {', '.join(sorted(missing))}"
            )
        for line_number, row in enumerate(reader, start=2):
            try:
                recall = float(row[RECALL_COLUMN])
                average_time_ms = float(row[TIME_COLUMN])
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"{csv_path}:{line_number}: invalid recall or time value"
                ) from exc
            if (
                not math.isfinite(recall)
                or not math.isfinite(average_time_ms)
                or average_time_ms <= 0
            ):
                raise ValueError(
                    f"{csv_path}:{line_number}: recall must be finite and time positive"
                )
            points.append(
                Point(recall=recall, qps=num_queries * 1000.0 / average_time_ms)
            )
    if not points:
        raise ValueError(f"Processed CSV contains no data rows: {csv_path}")
    if SORT_POINTS_BY_RECALL:
        points.sort(key=lambda point: point.recall)
    return tuple(points)


def load_all_series() -> tuple[list[list[Series]], list[str]]:
    """Load available curves and return readable missing entries."""
    all_series: list[list[Series]] = []
    missing_entries: list[str] = []
    for dataset in DATASETS:
        subplot: list[Series] = []
        for algorithm, style in zip(ALGORITHM_RESULTS, CURVE_STYLES):
            source = resolve_source(dataset, algorithm)
            drawing_csv = processed_path(source) if source is not None else None
            if drawing_csv is None or not drawing_csv.is_file():
                missing_entries.append(
                    f"{dataset['dataset']}/{dataset['query_task']}/{algorithm['label']}"
                )
                continue
            curve_key = (
                str(dataset["dataset"]),
                str(dataset["query_task"]),
                str(algorithm["label"]),
            )
            points = read_points(drawing_csv, int(dataset["num_queries"]))
            minimum_recall = MIN_RECALL_FILTERS.get(curve_key)
            if minimum_recall is not None:
                points = tuple(
                    point for point in points if point.recall >= minimum_recall
                )
                if not points:
                    raise ValueError(
                        f"{drawing_csv}: no points remain after applying minimum "
                        f"Recall {minimum_recall}"
                    )
            subplot.append(
                Series(
                    label=str(algorithm["label"]),
                    points=points,
                    style=style,
                )
            )
        if subplot:
            override = Y_LIMIT_OVERRIDES.get(
                (str(dataset["dataset"]), str(dataset["query_task"]))
            )
            if override is not None:
                qps_values = [
                    point.qps for series in subplot for point in series.points
                ]
                automatic_low, automatic_high = padded_log_limits(qps_values)
                override_low, override_high = override
                dataset["y_limits"] = (
                    automatic_low if override_low is None else override_low,
                    automatic_high if override_high is None else override_high,
                )
        else:
            # The shared Matplotlib renderer needs limits when no values exist.
            dataset["x_limits"] = EMPTY_X_LIMITS
            dataset["y_limits"] = EMPTY_Y_LIMITS
        all_series.append(subplot)
    return all_series, missing_entries


def padded_linear_limits(values: Sequence[float]) -> tuple[float, float]:
    low, high = min(values), max(values)
    padding = max((high - low) * 0.06, 0.005)
    return max(0.0, low - padding), min(1.0, high + padding)


def padded_log_limits(values: Sequence[float]) -> tuple[float, float]:
    low_power = math.floor(math.log(min(values), Y_LOG_BASE))
    high_power = math.ceil(math.log(max(values), Y_LOG_BASE))
    if low_power == high_power:
        high_power += 1
    return Y_LOG_BASE**low_power, Y_LOG_BASE**high_power


def nice_linear_ticks(low: float, high: float, count: int) -> list[float]:
    span = high - low
    if span <= 0:
        return [low]
    raw_step = span / max(count - 1, 1)
    magnitude = 10 ** math.floor(math.log10(raw_step))
    normalized = raw_step / magnitude
    if normalized <= 1:
        nice = 1
    elif normalized <= 2:
        nice = 2
    elif normalized <= 2.5:
        nice = 2.5
    elif normalized <= 5:
        nice = 5
    else:
        nice = 10
    step = nice * magnitude
    first = math.ceil((low - step * 1e-9) / step) * step
    ticks: list[float] = []
    value = first
    while value <= high + step * 1e-9:
        ticks.append(value)
        value += step
    return ticks


def log_ticks(low: float, high: float) -> list[float]:
    """Return only integral powers of ten within the visible log range."""
    first_power = math.ceil(math.log(low, Y_LOG_BASE) - 1e-12)
    last_power = math.floor(math.log(high, Y_LOG_BASE) + 1e-12)
    return [
        Y_LOG_BASE**power
        for power in range(first_power, last_power + 1)
    ]


def fmt_x_tick(value: float) -> str:
    return f"{value:.2f}".rstrip("0").rstrip(".")


def fmt_log_tick(value: float) -> str:
    power = round(math.log(value, Y_LOG_BASE))
    return rf"$\mathbf{{10^{{{power}}}}}$"


def describe_inputs(all_series: Iterable[Sequence[Series]]) -> str:
    panels = list(all_series)
    curve_count = sum(len(subplot) for subplot in panels)
    point_count = sum(len(series.points) for subplot in panels for series in subplot)
    return f"{len(panels)} panels, {curve_count} curves, {point_count} points"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Draw a 4x6 Matplotlib Recall@10-QPS figure for eight datasets, "
            "three selectivities, and nine algorithms."
        )
    )
    parser.add_argument(
        "--output", type=Path, default=OUTPUT_PATH, help=f"default: {OUTPUT_PATH}"
    )
    parser.add_argument(
        "--skip-preprocess",
        action="store_true",
        help="Reuse existing search_time_summary_draw.csv files.",
    )
    parser.add_argument(
        "--force-preprocess",
        action="store_true",
        help="Regenerate existing processed CSV files.",
    )
    parser.add_argument(
        "--preprocess-only",
        action="store_true",
        help="Prepare available CSV files without drawing the PNG.",
    )
    parser.add_argument(
        "--list-missing",
        action="store_true",
        help="Print every skipped dataset/task/algorithm combination.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    should_preprocess = PREPROCESS_BEFORE_PLOTTING and not args.skip_preprocess
    if should_preprocess:
        generated, reused, rows, missing = preprocess_all(
            force=FORCE_PREPROCESS or args.force_preprocess
        )
        print(
            f"Preprocessing: generated {generated}, reused {reused}, "
            f"wrote {rows} rows, skipped {missing} missing curves"
        )
    if args.preprocess_only:
        if not should_preprocess:
            raise ValueError("--preprocess-only cannot be combined with --skip-preprocess")
        return 0

    all_series, missing_entries = load_all_series()
    if args.list_missing:
        for entry in missing_entries:
            print(f"Missing: {entry}")

    output_path = args.output.expanduser().resolve()
    if output_path.suffix.lower() != ".png":
        raise ValueError(f"Output path must end with .png: {output_path}")
    pixel_width, pixel_height = render_png(
        sys.modules[__name__], all_series, output_path
    )
    print(
        f"Wrote {output_path} at {PNG_DPI} DPI, {pixel_width}x{pixel_height} px "
        f"({describe_inputs(all_series)}; skipped {len(missing_entries)} missing curves)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
