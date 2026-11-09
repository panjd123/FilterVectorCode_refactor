#!/usr/bin/env python3
"""Compare Trie with/without blocks for Amazon, VImg and Music.

Uses plot_recall_qps_block.py layout. With-block curves read existing draw CSVs;
without-block curves retain their separate preprocessing workflow. Axis limits
include both variants; QPS uses each task's actual query count.
Run: python3 tools/pic-draw/plot_recall_qps_block_presence.py
"""

from __future__ import annotations

import argparse
import csv
import html
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

from preprocess_search_results import preprocess_file
from recall_qps_png_renderer import render_png

# =============================================================================
# PNG 配置单位说明（以下所有可调参数均集中在脚本开头）
# =============================================================================
# 1. 字号、线宽、点大小、边框宽度使用 pt（磅）；提高 PNG_DPI 不需要同步放大。
# 2. 画布宽高、边距、间距和位置偏移使用布局像素；实际 PNG 像素会按
#    PNG_DPI / LAYOUT_DPI 等比例放大。
# 3. PNG_DPI 只控制输出分辨率；LAYOUT_DPI 用于把布局像素换算为物理尺寸。


# =============================================================================
# 用户配置：数据集、查询任务、步长和算法路径
# =============================================================================

RESULT_ROOT = Path("/home/dev/graphdb/FilterVectorResult")

# result_root 表示数据集结果路径；query_task 和 step_info 会组合成：
# <query_task>_<step_info>，例如 query_minlen1_sel10to40pct_100_100_20000。
# num_queries 用于通过 QPS = num_queries * 1000 / Average_Time_ms 计算 QPS。
# x_limits/y_limits 可针对每个数据集单独调整；y_tick_exclude 用于隐藏指定 Y 刻度。
DATASETS = [
    {
        "name": "Amazon",
        "result_root": RESULT_ROOT / "Amazon" / "results",
        "query_task": "query_minlen1_sel10to40pct",
        "step_info": "100_100_20000",
        "num_queries": 700,
        "x_limits": (0.0, 1.0),
        "y_limits": (300.0, 4000.0),
        "y_tick_label_offsets": {800.0: -10.0},  # 负数仅使刻度文字上移。
        "y_tick_exclude": (1000.0,),  # 保留参考脚本的刻度隐藏设置。
    },
    
    {
        "name": "VImg",
        "result_root": RESULT_ROOT / "VImg" / "results",
        "query_task": "query_minlen1_sel10to40pct",
        "step_info": "100_100_20000",
        "num_queries": 594,
        "x_limits": (0.0, 1.0),
        "y_limits": (200.0, 20000.0),
        "y_tick_exclude": (5000.0,),  # 保留参考脚本的刻度隐藏设置。
    },
    {
        "name": "Music",
        "result_root": RESULT_ROOT / "Music" / "results",
        "query_task": "query_minlen1_sel10to40pct",
        "step_info": "100_100_20000",
        "num_queries": 537,
        "x_limits": (0.0, 1.0),
        "y_limits": (100.0, 5000.0),
    },
]

# directory 是 result_root 下的算法目录；label 是图例显示名称。
# With block 复用已有 block=500、M=32、cross edges=4 的结果。
ALGORITHM_RESULTS = [
    {
        "label": "PTNC w/ Blocks",
        "directory": "Trie_block_500_M_32_cross_block_4",
        "processed_summary_name": "search_time_summary_draw.csv",
        "preprocess": False,  # 直接读取已有 draw 文件，包括强制预处理时也不改写。
    },
    {"label": "PTNC w/o Blocks", "directory": "Trie_no_block"},
]

# ---------------------------- 曲线覆盖顺序 ----------------------------
# 这里只控制绘图区的覆盖顺序，不改变 ALGORITHM_RESULTS 决定的图例顺序。
# 列表越靠前越先绘制、位于底层；越靠后越晚绘制、位于最上层。
# 每个算法名称必须与 ALGORITHM_RESULTS 中的 label 完全一致，且只能出现一次。
CURVE_DRAW_ORDER = [
    "PTNC w/o Blocks",  # 最先绘制，位于底层。
    "PTNC w/ Blocks",     # 最后绘制，位于最上层。
]

# 无 block 仍使用独立预处理文件；有 block 通过算法配置直接读取已有 draw 文件。
SOURCE_SUMMARY_NAME = "search_time_summary.csv"
PROCESSED_SUMMARY_NAME = "search_time_summary_block_presence_draw.csv"
RECALL_COLUMN = "Average_Recall"
TIME_COLUMN = "Average_Time_ms"
DROP_FIRST_ROWS = 1
PREPROCESS_BEFORE_PLOTTING = True
FORCE_PREPROCESS = False  # False 时复用已有有效文件；True 时强制重新生成。

# 所有图片默认写入 tools/pic-draw/pic/；目录不存在时会自动创建。
OUTPUT_DIR = Path(__file__).resolve().parent / "pic"
OUTPUT_PATH = OUTPUT_DIR / "amazon_vimg_music_block_presence_recall_qps.png"
PNG_DPI = 400       # PNG 输出分辨率；论文图片通常使用 300 或 600 DPI。
LAYOUT_DPI = 96     # 布局基准 DPI；通常保持 96，仅在需要整体缩放布局时修改。


# =============================================================================
# 用户配置：300 DPI PNG 的画布、字体、位置、图例和曲线样式
# =============================================================================

# 三个数据集 + SUBPLOT_COLUMNS=3 即为一行三列。
# FIGURE_WIDTH/HEIGHT 是 96 DPI 下的布局宽高；PNG 像素会按 PNG_DPI/LAYOUT_DPI 放大。
# 例如布局宽 1362 在 300 DPI 下输出约 4256 像素；宽高应按相同比例修改。
# 设为 None 时，脚本会根据子图尺寸、间距和边距自动计算。
FIGURE_WIDTH = None   # 整张图的布局宽度；None 表示根据子图和边距自动计算。
FIGURE_HEIGHT = 390   # 整张图的布局高度；减小可压缩图下方留白。

# PANEL_WIDTH/HEIGHT 使用布局像素，只控制子图内部几何尺寸。
PANEL_WIDTH = 430     # 单个子图区域的布局宽度。
PANEL_HEIGHT = 330    # 单个子图区域的布局高度。
SUBPLOT_COLUMNS = 3   # 每行子图数量；三个数据集设为 3 即一行三列。
CANVAS_BACKGROUND = "#ffffff"  # 整张图片背景色。
PLOT_BACKGROUND = "#fafafa"    # 各坐标轴绘图区背景色。

PANEL_MARGIN_LEFT = 72    # 绘图区相对单个子图左边界的距离。
PANEL_MARGIN_RIGHT = 6    # 绘图区相对单个子图右边界的距离。
PANEL_MARGIN_TOP = 52      # 绘图区相对单个子图顶部的距离。
PANEL_MARGIN_BOTTOM = 62  # 为 X 轴刻度和 Recall@10 预留的底部空间。
CANVAS_MARGIN_LEFT = 24   # 整张图左侧外边距；增大时只增加左侧留白。
CANVAS_MARGIN_RIGHT = 12  # 整张图右侧外边距；可与左侧独立调整。
CANVAS_MARGIN_BOTTOM = 0.2  # 整张图底部外边距；减小可减少 Recall@10 下方留白。
LEGEND_AREA_HEIGHT = 54  # 顶部为图例预留的总高度；增大时三个子图整体向下。
SUBPLOT_GAP_X = 18  # 相邻两列子图之间的水平间距。
SUBPLOT_GAP_Y = 22  # 相邻两行子图之间的垂直间距。

# 已安装 Liberation Serif；它与 Times 字体度量兼容，适合论文图。
# fontconfig 中 Times New Roman 和 Times 也会自动映射到该字体。
FONT_FAMILY = "Liberation Serif"

# ------------------------------ 字体控制 ------------------------------
# 字号单位为 pt（磅），不会随 PNG_DPI 改变物理大小；400=常规、600=半粗、700=粗体。
SUBPLOT_TITLE_FONT_FAMILY = FONT_FAMILY  # 数据集名称和小图序号的字体。
SUBPLOT_TITLE_FONT_SIZE = 25             # 数据集名称和小图序号字号，单位 pt。
SUBPLOT_TITLE_FONT_WEIGHT = 700          # 数据集名称和小图序号的字重。
X_AXIS_TITLE_FONT_FAMILY = FONT_FAMILY   # X 轴标题 Recall@10 的字体。
X_AXIS_TITLE_FONT_SIZE = 25              # X 轴标题 Recall@10 字号，单位 pt。
X_AXIS_TITLE_FONT_WEIGHT = 700           # X 轴标题 Recall@10 的字重。
Y_AXIS_TITLE_FONT_FAMILY = FONT_FAMILY   # Y 轴标题 QPS 的字体。
Y_AXIS_TITLE_FONT_SIZE = 25              # Y 轴标题 QPS 字号，单位 pt。
Y_AXIS_TITLE_FONT_WEIGHT = 700           # Y 轴标题 QPS 的字重。
X_TICK_FONT_FAMILY = FONT_FAMILY         # X 轴刻度数字的字体。
X_TICK_FONT_SIZE = 25                    # X 轴刻度字号，单位 pt。
X_TICK_FONT_WEIGHT = 600                 # X 轴刻度数字的字重。
Y_TICK_FONT_FAMILY = FONT_FAMILY         # Y 轴刻度数字的字体。
Y_TICK_FONT_SIZE = 25                    # Y 轴刻度字号，单位 pt。
Y_TICK_FONT_WEIGHT = 600                 # Y 轴刻度数字的字重。
LEGEND_FONT_FAMILY = FONT_FAMILY         # 顶部算法图例的字体。
LEGEND_FONT_SIZE = 25                    # 顶部图例字号，单位 pt。
LEGEND_FONT_WEIGHT = 600                 # 顶部算法图例的字重。
TITLE_FORMAT = "({letter}){dataset}"     # 小图序号和数据集名称的文本格式。

# ---------------------------- 文字位置控制 ----------------------------
# 偏移使用布局像素；水平偏移正数向右、负数向左；垂直偏移正数向下、负数向上。
SUBPLOT_TITLE_OFFSET_Y = 0.0  # 数据集名称和序号：负数上移，正数下移。
X_AXIS_TITLE_OFFSET_X = 0.0   # X 轴标题 Recall@10 的水平偏移量。
X_AXIS_TITLE_OFFSET_Y = 10.0   # X 轴标题 Recall@10 的垂直偏移量。
Y_AXIS_TITLE_OFFSET_X = -30.0  # Y 轴标题 QPS：负数向左、正数向右，单位为布局像素。
Y_AXIS_TITLE_OFFSET_Y = 0.0   # Y 轴标题 QPS 的垂直偏移量。
Y_AXIS_TITLE_FIRST_COLUMN_ONLY = True  # 仅最左列子图显示 QPS 标题。

# None 表示每个子图根据自己的数据自动计算范围；也可改为 (0.9, 1.0)、
# (1e2, 1e5) 这样的固定范围。
# DATASETS 中的 x_limits/y_limits 优先级更高，可单独控制每张子图。
X_LIMITS = None       # 全局 X 轴范围；DATASETS 中的 x_limits 会覆盖它。
Y_LIMITS = None       # 全局 Y 轴范围；DATASETS 中的 y_limits 会覆盖它。
X_TICK_COUNT = 5     # X 轴期望刻度数量（实际数量会按整洁刻度自动调整）。
Y_LOG_BASE = 10       # Y 轴对数底数。
# 紧凑对数纵轴在每个十倍区间显示 1、2、5 三个刻度。
Y_TICK_MULTIPLIERS = (1, 2, 5)
AXIS_COLOR = "#222222"  # 坐标轴、标题和刻度文字颜色。
GRID_COLOR = "#c9c9c9"  # 主网格线颜色。
GRID_WIDTH = 0.6   # 网格线宽，单位 pt。
GRID_DASH = "2,2"  # 网格虚线样式：线段长度,间隔长度，单位 pt。
AXIS_WIDTH = 0.9   # 坐标轴线宽，单位 pt。

# ------------------------------ 图例布局 ------------------------------
LEGEND_COLUMNS = 2              # 两个算法图例同一行显示。
LEGEND_X_OFFSET = 0.0           # 图例整体水平偏移：负数向左，正数向右。
LEGEND_TOP = -25.0               # 图例框的纵坐标：减小使图例上移，增大使图例下移。
LEGEND_COLUMN_SPACING = 1.5   # 图例列间距，单位为图例字号的倍数。
LEGEND_LABEL_SPACING = 0.5    # 图例多行时的行间距，单位为图例字号的倍数。
LEGEND_LINE_LENGTH = 2.0     # 图例示例线长度，单位为图例字号的倍数。
LEGEND_GAP_AFTER_LINE = 0.5  # 示例线与文字间距，单位为图例字号的倍数。
LEGEND_BORDER_PADDING = 0.45   # 图例内容与边框间距，单位为图例字号的倍数。
LEGEND_BACKGROUND_COLOR = "#ffffff"  # 图例框背景颜色。
LEGEND_BORDER_COLOR = "#b8b8b8"      # 图例边框颜色；当前为灰色。
LEGEND_BORDER_WIDTH = 0.8              # 图例边框线宽，单位 pt；设为 0 可隐藏。
LEGEND_ROUNDED_CORNERS = True      # True 使用圆角图例框；False 使用直角框。

LINE_WIDTH = 6.0              # 折线宽度，单位 pt；300 DPI 不需要按 DPI 放大此值。
MARKER_SIZE = 15.0             # 数据点直径，单位 pt；Matplotlib 中不是半径。
MARKER_EDGE_WIDTH = 0.8         # 数据点灰白色边框线宽，单位 pt。
MARKER_EDGE_COLOR = "#f2f2f2"  # 数据点边框颜色；浅灰白效果接近原 SVG 图。
# CURVE_STYLES 与 ALGORITHM_RESULTS 一一对应；dash="" 表示实线，
# 例如 dash="6,3" 表示 6 pt 线段 + 3 pt 间隔；marker 可选 square/triangle/diamond/circle。
CURVE_STYLES = [
    {"color": "#f57c00", "dash": "", "marker": "square"},
    {"color": "#2e7d32", "dash": "", "marker": "triangle"},

]

SORT_POINTS_BY_RECALL = True  # True 时按 Recall 从小到大连接数据点。


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
    """Return curves in configured bottom-to-top drawing order."""
    labels = [series.label for series in series_list]
    if len(labels) != len(set(labels)):
        raise ValueError("Series labels must be unique when using CURVE_DRAW_ORDER")
    if len(CURVE_DRAW_ORDER) != len(set(CURVE_DRAW_ORDER)):
        raise ValueError("CURVE_DRAW_ORDER contains duplicate algorithm labels")

    available = set(labels)
    configured = set(CURVE_DRAW_ORDER)
    if configured != available:
        missing = sorted(available - configured)
        unknown = sorted(configured - available)
        raise ValueError(
            "CURVE_DRAW_ORDER must list every algorithm exactly once; "
            f"missing={missing}, unknown={unknown}"
        )

    by_label = {series.label: series for series in series_list}
    return [by_label[label] for label in CURVE_DRAW_ORDER]



def result_task(dataset: dict[str, object]) -> str:
    """Combine query task and step information into the result directory name."""
    query_task = str(dataset["query_task"]).strip("_")
    step_info = str(dataset["step_info"]).strip("_")
    if not query_task or not step_info:
        raise ValueError("query_task and step_info must not be empty")
    return f"{query_task}_{step_info}"


def summary_paths(
    dataset: dict[str, object], algorithm: dict[str, object]
) -> tuple[Path, Path]:
    result_dir = (
        Path(dataset["result_root"])
        / str(algorithm["directory"])
        / result_task(dataset)
        / "results"
    )
    processed_name = str(algorithm.get("processed_summary_name", PROCESSED_SUMMARY_NAME))
    return result_dir / SOURCE_SUMMARY_NAME, result_dir / processed_name


def preprocess_all(force: bool = FORCE_PREPROCESS) -> tuple[int, int, int]:
    """Build missing/invalid files and reuse existing valid processed CSVs."""
    processed_count = 0
    skipped_count = 0
    row_count = 0
    for dataset in DATASETS:
        for algorithm in ALGORITHM_RESULTS:
            source_path, processed_path = summary_paths(dataset, algorithm)
            if not algorithm.get("preprocess", True):
                if not processed_path.is_file():
                    raise FileNotFoundError(f"Missing existing draw CSV: {processed_path}")
                skipped_count += 1
                continue
            written_rows = preprocess_file(
                source_path,
                processed_path,
                time_column=TIME_COLUMN,
                drop_first_rows=DROP_FIRST_ROWS,
                force=force,
            )
            if written_rows == 0:
                skipped_count += 1
            else:
                processed_count += 1
                row_count += written_rows
    return processed_count, skipped_count, row_count


def read_points(csv_path: Path, num_queries: int) -> tuple[Point, ...]:
    """Read a processed CSV and calculate QPS from its reordered time column."""
    if not csv_path.is_file():
        raise FileNotFoundError(
            f"Plot CSV does not exist: {csv_path}"
        )
    if num_queries <= 0:
        raise ValueError(f"num_queries must be positive for {csv_path}")

    points: list[Point] = []
    with csv_path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames or []
        missing = {RECALL_COLUMN, TIME_COLUMN}.difference(fieldnames)
        if missing:
            names = ", ".join(sorted(missing))
            raise ValueError(f"{csv_path}: missing required column(s): {names}")

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
                    f"{csv_path}:{line_number}: recall must be finite and time must be positive"
                )
            qps = num_queries * 1000.0 / average_time_ms
            points.append(Point(recall=recall, qps=qps))

    if not points:
        raise ValueError(f"Processed CSV contains no data rows: {csv_path}")
    if SORT_POINTS_BY_RECALL:
        points.sort(key=lambda point: point.recall)
    return tuple(points)


def load_all_series() -> list[list[Series]]:
    if len(CURVE_STYLES) < len(ALGORITHM_RESULTS):
        raise ValueError("CURVE_STYLES must contain at least one style per algorithm")

    result: list[list[Series]] = []
    for dataset in DATASETS:
        subplot: list[Series] = []
        num_queries = int(dataset["num_queries"])
        for algorithm, style in zip(ALGORITHM_RESULTS, CURVE_STYLES):
            _, processed_path = summary_paths(dataset, algorithm)
            subplot.append(
                Series(
                    label=algorithm["label"],
                    points=read_points(processed_path, num_queries),
                    style=style,
                )
            )
        result.append(subplot)
    return result


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
    """Return log ticks and always include the configured axis endpoints."""
    first_power = math.floor(math.log(low, Y_LOG_BASE))
    last_power = math.ceil(math.log(high, Y_LOG_BASE))
    ticks = [low, high]
    for power in range(first_power, last_power + 1):
        scale = Y_LOG_BASE**power
        for multiplier in Y_TICK_MULTIPLIERS:
            tick = multiplier * scale
            if low < tick < high:
                ticks.append(tick)
    # Remove numerically duplicate ticks while preserving ascending order.
    unique_ticks: list[float] = []
    for tick in sorted(ticks):
        if not unique_ticks or not math.isclose(tick, unique_ticks[-1]):
            unique_ticks.append(tick)
    return unique_ticks


def fmt_x_tick(value: float) -> str:
    return f"{value:.2f}".rstrip("0").rstrip(".")



def fmt_log_tick(value: float) -> str:
    """Use ordinary decimal digits on the logarithmic Y axis."""
    if math.isclose(value, round(value)):
        return str(int(round(value)))
    return f"{value:.6f}".rstrip("0").rstrip(".")


def svg_line(
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    *,
    color: str,
    width: float,
    dash: str = "",
) -> str:
    dash_attr = f' stroke-dasharray="{dash}"' if dash else ""
    return (
        f'<line x1="{x1:.2f}" y1="{y1:.2f}" x2="{x2:.2f}" y2="{y2:.2f}" '
        f'stroke="{color}" stroke-width="{width}"{dash_attr}/>'
    )


def svg_marker(x: float, y: float, style: dict[str, object]) -> str:
    color = str(style["color"])
    marker = str(style["marker"])
    size = MARKER_SIZE
    common = (
        f'fill="{color}" stroke="{MARKER_EDGE_COLOR}" '
        f'stroke-width="{MARKER_EDGE_WIDTH}"'
    )
    if marker == "square":
        return (
            f'<rect x="{x-size:.2f}" y="{y-size:.2f}" '
            f'width="{2*size:.2f}" height="{2*size:.2f}" {common}/>'
        )
    if marker == "triangle":
        points = (
            f"{x:.2f},{y-size:.2f} {x-size:.2f},{y+size:.2f} "
            f"{x+size:.2f},{y+size:.2f}"
        )
        return f'<polygon points="{points}" {common}/>'
    if marker == "diamond":
        points = (
            f"{x:.2f},{y-size:.2f} {x+size:.2f},{y:.2f} "
            f"{x:.2f},{y+size:.2f} {x-size:.2f},{y:.2f}"
        )
        return f'<polygon points="{points}" {common}/>'
    return f'<circle cx="{x:.2f}" cy="{y:.2f}" r="{size:.2f}" {common}/>'


def text_element(
    x: float,
    y: float,
    text: str,
    *,
    size: int,
    anchor: str = "start",
    weight: int = 400,
    family: str = FONT_FAMILY,
    transform: str = "",
) -> str:
    transform_attr = f' transform="{transform}"' if transform else ""
    return (
        f'<text x="{x:.2f}" y="{y:.2f}" text-anchor="{anchor}" '
        f'font-family="{html.escape(family)}" font-size="{size}" font-weight="{weight}" fill="{AXIS_COLOR}"'
        f'{transform_attr}>{html.escape(text)}</text>'
    )


def draw_legend(canvas_width: int) -> list[str]:
    """Draw a movable shared legend with a configurable gray border."""
    columns = min(LEGEND_COLUMNS, len(ALGORITHM_RESULTS))
    rows = math.ceil(len(ALGORITHM_RESULTS) / columns)
    content_width = columns * LEGEND_ITEM_WIDTH
    content_height = max(2 * MARKER_SIZE, LEGEND_FONT_SIZE + 4)
    box_width = content_width + 2 * LEGEND_PADDING_X
    box_height = (
        2 * LEGEND_PADDING_Y
        + content_height
        + (rows - 1) * LEGEND_ROW_HEIGHT
    )

    # 默认水平居中，再叠加 LEGEND_X_OFFSET，便于只移动图例。
    box_x = (canvas_width - box_width) / 2 + LEGEND_X_OFFSET
    box_y = LEGEND_TOP
    start_x = box_x + LEGEND_PADDING_X
    first_center_y = box_y + LEGEND_PADDING_Y + content_height / 2

    # 边框必须先绘制，之后的线、点和文字才能显示在它上面。
    parts: list[str] = [
        f'<rect x="{box_x:.2f}" y="{box_y:.2f}" '
        f'width="{box_width:.2f}" height="{box_height:.2f}" '
        f'fill="{LEGEND_BACKGROUND_COLOR}" stroke="{LEGEND_BORDER_COLOR}" '
        f'stroke-width="{LEGEND_BORDER_WIDTH}" rx="{LEGEND_BORDER_RADIUS}" '
        f'ry="{LEGEND_BORDER_RADIUS}"/>'
    ]

    for index, (algorithm, style) in enumerate(
        zip(ALGORITHM_RESULTS, CURVE_STYLES)
    ):
        row, column = divmod(index, columns)
        x = start_x + column * LEGEND_ITEM_WIDTH
        y = first_center_y + row * LEGEND_ROW_HEIGHT
        parts.append(
            svg_line(
                x,
                y,
                x + LEGEND_LINE_LENGTH,
                y,
                color=str(style["color"]),
                width=LINE_WIDTH,
                dash=str(style["dash"]),
            )
        )
        parts.append(svg_marker(x + LEGEND_LINE_LENGTH / 2, y, style))
        parts.append(
            text_element(
                x + LEGEND_LINE_LENGTH + LEGEND_GAP_AFTER_LINE,
                y + 5,
                algorithm["label"],
                size=LEGEND_FONT_SIZE,
                weight=LEGEND_FONT_WEIGHT,
                family=LEGEND_FONT_FAMILY,
            )
        )
    return parts


def draw_subplot(
    panel_x: float,
    panel_y: float,
    subplot_index: int,
    dataset: dict[str, object],
    series_list: Sequence[Series],
) -> list[str]:
    plot_x = panel_x + PANEL_MARGIN_LEFT
    plot_y = panel_y + PANEL_MARGIN_TOP
    plot_width = PANEL_WIDTH - PANEL_MARGIN_LEFT - PANEL_MARGIN_RIGHT
    plot_height = PANEL_HEIGHT - PANEL_MARGIN_TOP - PANEL_MARGIN_BOTTOM
    dataset_name = str(dataset["name"])

    recalls = [point.recall for series in series_list for point in series.points]
    qps_values = [point.qps for series in series_list for point in series.points]
    dataset_x_limits = dataset.get("x_limits", X_LIMITS)
    dataset_y_limits = dataset.get("y_limits", Y_LIMITS)
    x_low, x_high = dataset_x_limits or padded_linear_limits(recalls)
    y_low, y_high = dataset_y_limits or padded_log_limits(qps_values)
    if x_low >= x_high or y_low <= 0 or y_low >= y_high:
        raise ValueError("Invalid X_LIMITS or Y_LIMITS")

    def scale_x(value: float) -> float:
        return plot_x + (value - x_low) / (x_high - x_low) * plot_width

    y_low_log = math.log(y_low, Y_LOG_BASE)
    y_high_log = math.log(y_high, Y_LOG_BASE)

    def scale_y(value: float) -> float:
        position = (
            math.log(value, Y_LOG_BASE) - y_low_log
        ) / (y_high_log - y_low_log)
        return plot_y + plot_height * (1 - position)

    clip_id = f"plot-clip-{subplot_index}"

    parts = [
        f'<clipPath id="{clip_id}"><rect x="{plot_x:.2f}" y="{plot_y:.2f}" '
        f'width="{plot_width:.2f}" height="{plot_height:.2f}"/></clipPath>',
        f'<rect x="{plot_x:.2f}" y="{plot_y:.2f}" width="{plot_width:.2f}" '
        f'height="{plot_height:.2f}" fill="{PLOT_BACKGROUND}"/>',
    ]

    for tick in nice_linear_ticks(x_low, x_high, X_TICK_COUNT):
        x = scale_x(tick)
        parts.append(
            svg_line(
                x,
                plot_y,
                x,
                plot_y + plot_height,
                color=GRID_COLOR,
                width=GRID_WIDTH,
                dash=GRID_DASH,
            )
        )
        parts.append(
            text_element(
                x,
                plot_y + plot_height + 21,
                fmt_x_tick(tick),
                size=X_TICK_FONT_SIZE,
                weight=X_TICK_FONT_WEIGHT,
                family=X_TICK_FONT_FAMILY,
                anchor="middle",
            )
        )

    y_ticks = log_ticks(y_low, y_high)
    excluded_y_ticks = tuple(
        float(value) for value in dataset.get("y_tick_exclude", ())
    )
    y_ticks = [
        tick
        for tick in y_ticks
        if not any(math.isclose(tick, excluded) for excluded in excluded_y_ticks)
    ]
    tick_label_offsets = {
        float(value): float(offset)
        for value, offset in dataset.get("y_tick_label_offsets", {}).items()
    }
    for tick in y_ticks:
        y = scale_y(tick)
        label_offset_y = next(
            (offset for value, offset in tick_label_offsets.items() if math.isclose(tick, value)),
            0.0,
        )
        parts.append(
            svg_line(
                plot_x,
                y,
                plot_x + plot_width,
                y,
                color=GRID_COLOR,
                width=GRID_WIDTH,
                dash=GRID_DASH,
            )
        )
        parts.append(
            text_element(
                plot_x - 9,
                y + 5 + label_offset_y,
                fmt_log_tick(tick),
                size=Y_TICK_FONT_SIZE,
                weight=Y_TICK_FONT_WEIGHT,
                family=Y_TICK_FONT_FAMILY,
                anchor="end",
            )
        )

    parts.append(f'<g clip-path="url(#{clip_id})">')
    for series in series_in_draw_order(series_list):
        coordinates = [
            (scale_x(point.recall), scale_y(point.qps))
            for point in series.points
        ]
        points_attr = " ".join(f"{x:.2f},{y:.2f}" for x, y in coordinates)
        dash_attr = (
            f' stroke-dasharray="{series.style["dash"]}"'
            if series.style["dash"]
            else ""
        )
        parts.append(
            f'<polyline points="{points_attr}" fill="none" '
            f'stroke="{series.style["color"]}" stroke-width="{LINE_WIDTH}" '
            f'stroke-linejoin="round" stroke-linecap="round"{dash_attr}/>'
        )
        parts.extend(svg_marker(x, y, series.style) for x, y in coordinates)
    parts.append("</g>")

    parts.append(
        svg_line(
            plot_x,
            plot_y + plot_height,
            plot_x + plot_width,
            plot_y + plot_height,
            color=AXIS_COLOR,
            width=AXIS_WIDTH,
        )
    )
    parts.append(
        svg_line(
            plot_x,
            plot_y,
            plot_x,
            plot_y + plot_height,
            color=AXIS_COLOR,
            width=AXIS_WIDTH,
        )
    )

    subplot_letter = chr(ord("a") + subplot_index)
    title = TITLE_FORMAT.format(
        index=subplot_index + 1, letter=subplot_letter, dataset=dataset_name
    )
    parts.append(
        text_element(
            panel_x + PANEL_WIDTH / 2,
            panel_y + 27 + SUBPLOT_TITLE_OFFSET_Y,
            title,
            size=SUBPLOT_TITLE_FONT_SIZE,
            family=SUBPLOT_TITLE_FONT_FAMILY,
            anchor="middle",
            weight=SUBPLOT_TITLE_FONT_WEIGHT,
        )
    )
    parts.append(
        text_element(
            plot_x + plot_width / 2 + X_AXIS_TITLE_OFFSET_X,
            panel_y + PANEL_HEIGHT - 16 + X_AXIS_TITLE_OFFSET_Y,
            "Recall@10",
            size=X_AXIS_TITLE_FONT_SIZE,
            family=X_AXIS_TITLE_FONT_FAMILY,
            anchor="middle",
            weight=X_AXIS_TITLE_FONT_WEIGHT,
        )
    )
    y_label_x = panel_x + 20 + Y_AXIS_TITLE_OFFSET_X
    y_label_y = plot_y + plot_height / 2 + Y_AXIS_TITLE_OFFSET_Y
    parts.append(
        text_element(
            y_label_x,
            y_label_y,
            "QPS",
            size=Y_AXIS_TITLE_FONT_SIZE,
            family=Y_AXIS_TITLE_FONT_FAMILY,
            anchor="middle",
            weight=Y_AXIS_TITLE_FONT_WEIGHT,
            transform=f"rotate(-90 {y_label_x:.2f} {y_label_y:.2f})",
        )
    )
    return parts


def render_svg(all_series: Sequence[Sequence[Series]]) -> str:
    dataset_count = len(DATASETS)
    if dataset_count == 0:
        raise ValueError("DATASETS must not be empty")
    columns = min(SUBPLOT_COLUMNS, dataset_count)
    rows = math.ceil(dataset_count / columns)
    canvas_width = int(
        CANVAS_MARGIN_LEFT + CANVAS_MARGIN_RIGHT
        + columns * PANEL_WIDTH
        + (columns - 1) * SUBPLOT_GAP_X
    )
    canvas_height = int(
        LEGEND_AREA_HEIGHT
        + rows * PANEL_HEIGHT
        + (rows - 1) * SUBPLOT_GAP_Y
        + CANVAS_MARGIN_BOTTOM
    )

    # 最终 width/height 控制显示尺寸，viewBox 保留内部布局坐标。
    figure_width = FIGURE_WIDTH if FIGURE_WIDTH is not None else canvas_width
    figure_height = FIGURE_HEIGHT if FIGURE_HEIGHT is not None else canvas_height

    parts = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{figure_width}" '
        f'height="{figure_height}" viewBox="0 0 {canvas_width} {canvas_height}">',
        f'<rect width="100%" height="100%" fill="{CANVAS_BACKGROUND}"/>',
        f'<g font-family="{html.escape(FONT_FAMILY)}">',
    ]
    parts.extend(draw_legend(canvas_width))

    for index, (dataset, series_list) in enumerate(zip(DATASETS, all_series)):
        row, column = divmod(index, columns)
        panel_x = CANVAS_MARGIN_LEFT + column * (PANEL_WIDTH + SUBPLOT_GAP_X)
        panel_y = LEGEND_AREA_HEIGHT + row * (PANEL_HEIGHT + SUBPLOT_GAP_Y)
        parts.extend(
            draw_subplot(
                panel_x,
                panel_y,
                index,
                dataset,
                series_list,
            )
        )

    parts.extend(["</g>", "</svg>", ""])
    return "\n".join(parts)


def describe_inputs(all_series: Iterable[Sequence[Series]]) -> str:
    curve_count = sum(len(subplot) for subplot in all_series)
    point_count = sum(
        len(series.points) for subplot in all_series for series in subplot
    )
    return (
        f"{len(DATASETS)} subplot(s), {curve_count} curve(s), "
        f"{point_count} point(s)"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Preprocess summaries and draw with/without-block Recall@10-QPS curves as a 400 DPI PNG."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=OUTPUT_PATH,
        help=f"PNG output path (default: {OUTPUT_PATH})",
    )
    parser.add_argument(
        "--skip-preprocess",
        action="store_true",
        help="Reuse existing processed CSV files instead of regenerating them.",
    )
    parser.add_argument(
        "--force-preprocess",
        action="store_true",
        help="Rebuild valid processed CSV files instead of reusing them.",
    )
    parser.add_argument(
        "--preprocess-only",
        action="store_true",
        help="Create processed CSV files without drawing the PNG.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    should_preprocess = PREPROCESS_BEFORE_PLOTTING and not args.skip_preprocess
    if should_preprocess:
        processed_count, skipped_count, row_count = preprocess_all(
            force=FORCE_PREPROCESS or args.force_preprocess
        )
        print(
            f"Preprocessing: generated {processed_count} file(s), "
            f"reused {skipped_count} file(s), wrote {row_count} row(s)"
        )
    if args.preprocess_only:
        if not should_preprocess:
            raise ValueError("--preprocess-only cannot be combined with --skip-preprocess")
        return 0

    all_series = load_all_series()
    output_path = args.output.expanduser().resolve()
    if output_path.suffix.lower() != ".png":
        raise ValueError(f"Output path must end with .png: {output_path}")
    pixel_width, pixel_height = render_png(
        sys.modules[__name__], all_series, output_path
    )
    print(
        f"Wrote {output_path} at {PNG_DPI} DPI, "
        f"{pixel_width}x{pixel_height} px ({describe_inputs(all_series)})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
