#!/usr/bin/env python3
"""Plot CPU graph construction stage shares as PNG/PDF by default."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
from matplotlib.patches import FancyBboxPatch, Patch


# =============================================================================
# 数据及输出配置：直接运行本脚本仅生成占比图（PNG 和 PDF）。
# =============================================================================
SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT = SCRIPT_DIR / "trie_block_build_optimizations/current_cpu_build_breakdown.csv"
DEFAULT_OUTPUT = SCRIPT_DIR / "pic/trie_block_build_optimizations/current_cpu_build"
# ratio：100% 堆叠占比图；time：按秒的分组柱状图；both：分别输出两张图。
DEFAULT_MODE = "ratio"
PNG_DPI = 400
OUTPUT_FORMATS = (".png", ".pdf")
DATASETS = ["Genome", "Reviews", "Amazon", "VImg", "Music", "BReviews", "Tiktok", "Laion"]

# CSV 列名、图例文字、颜色；列表顺序同时控制堆叠顺序（从下到上）和分组顺序。
# Base Graph 包括基础 Trie 分组、组内图和基础组间边构建。
STAGES = [
    ("base_build_s", "Base Graph", "#2e7d32"),
    ("intra_build_s", "Intra-block Graph", "#ff8200"),
    ("inter_build_s", "Inter-block Edges", "#0aaabd"),
]

# =============================================================================
# 绘图样式配置：通常只需修改这里，不需要改动绘图函数。
# =============================================================================
PLOT_STYLE = {
    # ===== 柱子内百分比数字：颜色、大小在这里调整 =====
    "value_label_color": "#060606",  # 数字颜色：白色；黑色用 "#000000"。
    "value_label_font_size": 25,     # 数字字号（pt）：数值越大，文字越大。
    "value_label_font_weight": 700,  # 数字字重：400 为常规，700 为加粗。
    "show_ratio_labels": True,      # 是否显示柱子内的百分比数字。
    "ratio_label_decimals": 1,      # 百分比保留的小数位数。
    # 小于该占比的区段不显示数字，避免重叠；0.07 表示 7%。
    "ratio_label_min_fraction": 0.07,

    # ===== 图形、坐标轴及其他样式 =====
    "figure_size": (16.5, 4.8),
    # 与 plot_index_time_memory.py 的 Times 风格及字重保持一致。
    "font_family": "Liberation Serif",
    "mathtext_fontset": "stix",
    "base_font_size": 18,
    "x_tick_font_size": 30,
    "y_tick_font_size": 30,
    "tick_font_weight": 700,
    "y_axis_label_font_size": 30,
    "axis_label_font_weight": 700,
    "ratio_y_axis_label": "Build Time Share",
    "time_y_axis_label": "CPU Build\nTime (s)",
    # 标题位置为坐标轴相对坐标，x 越小越靠左，y=0.5 为垂直居中。
    "y_axis_label_position": (-0.07, 0.50),
    "ratio_y_axis_limits": (0.0, 1.0),
    "ratio_y_axis_ticks": (0.0, 0.25, 0.5, 0.75, 1.0),
    # 秒数跨度较大，分组图使用对数轴；各阶段柱高直接表示秒数。
    "time_y_axis_limits": (0.1, 1.0e5),
    "time_y_axis_tick_exponents": (-1, 0, 1, 2, 3, 4, 5),
    "dataset_spacing": 1.0,
    "ratio_bar_width": 0.68,
    "time_bar_width": 0.24,
    "time_bar_fill_ratio": 0.92,
    "bar_alpha": 0.96,
    "bar_edge_color": "#000000",
    "bar_edge_width": 0.5,
    # 圆角和间隙仅用于占比图；设为 0 可取消。高度单位为占比 0–1。
    "segment_rounding": 0.012,
    "segment_gap": 0.003,
    "show_time_labels": False,
    "time_label_offset": 3,
    "legend_font_size": 28,
    "legend_font_weight": 600,
    # 使用整张图的相对坐标；增大 x 向右移动，增大 y 向上移动。
    "legend_anchor": (0.53, 0.995),
    "legend_location": "upper center",
    "legend_columns": 3,
    "legend_column_spacing": 1.5,
    "legend_handle_length": 1.25,
    "legend_handle_height": 0.9,
    "legend_frame_on": True,
    "legend_frame_face_color": "#ffffff",
    "legend_frame_edge_color": "#b8b8b8",
    "legend_frame_alpha": 1.0,
    "legend_frame_line_width": 0.8,
    "legend_frame_border_pad": 0.45,
    "x_tick_padding": 7,
    "y_tick_padding": 5,
    "x_side_padding": 0.55,
    "canvas_background": "#ffffff",
    "plot_background": "#fafafa",
    "axis_color": "#222222",
    "axis_width": 0.9,
    "major_grid_color": "#c9c9c9",
    "major_grid_width": 0.6,
    "major_grid_dash": (0, (2, 2)),
    # 子图边距依次为左、右、下、上；上方空间用于图例。
    "subplot_margins": (0.085, 0.995, 0.17, 0.78),
    "save_padding": 0.04,
}


def load_times(input_csv: Path) -> np.ndarray:
    """Read unrounded stage seconds; normalize only over these three stages."""
    records = {}
    with input_csv.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"dataset", "three_stage_total_s", *(column for column, _, _ in STAGES)}
        missing = required - set(reader.fieldnames or ())
        if missing:
            raise ValueError(f"Missing CSV columns: {sorted(missing)}")
        for row in reader:
            dataset = row["dataset"].strip()
            if dataset in records:
                raise ValueError(f"Duplicate dataset: {dataset}")
            values = np.asarray([float(row[column]) for column, _, _ in STAGES])
            if not np.isfinite(values).all() or np.any(values <= 0):
                raise ValueError(f"Stage times must be finite and positive: {dataset}")
            if not np.isclose(values.sum(), float(row["three_stage_total_s"]), rtol=1e-9, atol=1e-9):
                raise ValueError(f"Stage times do not match three_stage_total_s: {dataset}")
            records[dataset] = values
    if set(records) != set(DATASETS):
        raise ValueError(
            f"Dataset mismatch: missing={sorted(set(DATASETS) - set(records))}, "
            f"extra={sorted(set(records) - set(DATASETS))}"
        )
    return np.asarray([records[dataset] for dataset in DATASETS])


def draw_ratio(ax: plt.Axes, x: np.ndarray, times: np.ndarray) -> None:
    """Stack base/intra/inter shares; file loading/saving is outside this sum."""
    fractions = times / times.sum(axis=1, keepdims=True)
    bottoms = np.zeros(len(DATASETS))
    for index, (_, _, color) in enumerate(STAGES):
        heights = fractions[:, index]
        for center, bottom, height in zip(x, bottoms, heights):
            width = PLOT_STYLE["ratio_bar_width"]
            gap = min(PLOT_STYLE["segment_gap"], height * 0.25)
            visible_height = height - gap
            rounding = min(PLOT_STYLE["segment_rounding"], visible_height / 2, width / 2)
            ax.add_patch(FancyBboxPatch(
                (center - width / 2, bottom + gap / 2), width, visible_height,
                boxstyle=f"round,pad=0,rounding_size={rounding}",
                facecolor=color, edgecolor=PLOT_STYLE["bar_edge_color"],
                linewidth=PLOT_STYLE["bar_edge_width"], alpha=PLOT_STYLE["bar_alpha"], zorder=3,
            ))
            if PLOT_STYLE["show_ratio_labels"] and height >= PLOT_STYLE["ratio_label_min_fraction"]:
                ax.text(
                    center, bottom + height / 2,
                    f"{height * 100:.{PLOT_STYLE['ratio_label_decimals']}f}%",
                    ha="center", va="center", fontsize=PLOT_STYLE["value_label_font_size"],
                    color=PLOT_STYLE["value_label_color"],
                    fontweight=PLOT_STYLE["value_label_font_weight"], zorder=5,
                )
        bottoms += heights
    ax.set_ylim(*PLOT_STYLE["ratio_y_axis_limits"])
    ax.set_yticks(PLOT_STYLE["ratio_y_axis_ticks"])
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=0))


def log_tick_label(value: float, _position: int | None = None) -> str:
    return rf"$\mathbf{{10^{{{round(np.log10(value))}}}}}$"


def draw_time(ax: plt.Axes, x: np.ndarray, times: np.ndarray) -> None:
    """Group absolute seconds on a log axis so small datasets remain readable."""
    width = PLOT_STYLE["time_bar_width"]
    for index, (_, _, color) in enumerate(STAGES):
        bars = ax.bar(
            x + (index - (len(STAGES) - 1) / 2) * width, times[:, index],
            width=width * PLOT_STYLE["time_bar_fill_ratio"], color=color,
            edgecolor=PLOT_STYLE["bar_edge_color"], linewidth=PLOT_STYLE["bar_edge_width"],
            alpha=PLOT_STYLE["bar_alpha"], zorder=3,
        )
        if PLOT_STYLE["show_time_labels"]:
            ax.bar_label(
                bars, labels=[f"{value:.1f}" for value in times[:, index]],
                padding=PLOT_STYLE["time_label_offset"],
                fontsize=PLOT_STYLE["value_label_font_size"], color=PLOT_STYLE["value_label_color"],
                fontweight=PLOT_STYLE["value_label_font_weight"],
            )
    ax.set_yscale("log")
    ax.set_ylim(*PLOT_STYLE["time_y_axis_limits"])
    ax.yaxis.set_major_locator(mticker.FixedLocator(
        10.0 ** np.asarray(PLOT_STYLE["time_y_axis_tick_exponents"])
    ))
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(log_tick_label))
    ax.minorticks_off()


def plot(times: np.ndarray, output_prefix: Path, mode: str) -> None:
    """Apply the shared style and save one figure for each selected mode."""
    rc = {
        "font.family": PLOT_STYLE["font_family"],
        "mathtext.fontset": PLOT_STYLE["mathtext_fontset"],
        "font.size": PLOT_STYLE["base_font_size"],
        "font.weight": PLOT_STYLE["tick_font_weight"],
        "xtick.labelsize": PLOT_STYLE["x_tick_font_size"],
        "ytick.labelsize": PLOT_STYLE["y_tick_font_size"],
        "axes.labelsize": PLOT_STYLE["y_axis_label_font_size"],
        "axes.labelweight": PLOT_STYLE["axis_label_font_weight"],
        "axes.edgecolor": PLOT_STYLE["axis_color"],
        "axes.linewidth": PLOT_STYLE["axis_width"],
        "text.color": PLOT_STYLE["axis_color"],
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
    modes = ("ratio", "time") if mode == "both" else (mode,)
    with plt.rc_context(rc):
        for selected in modes:
            fig, ax = plt.subplots(figsize=PLOT_STYLE["figure_size"])
            x = np.arange(len(DATASETS)) * PLOT_STYLE["dataset_spacing"]
            (draw_ratio if selected == "ratio" else draw_time)(ax, x, times)
            ax.set_xlim(x[0] - PLOT_STYLE["x_side_padding"], x[-1] + PLOT_STYLE["x_side_padding"])
            ax.set_xticks(x, DATASETS)
            ax.set_ylabel(PLOT_STYLE[f"{selected}_y_axis_label"])
            ax.yaxis.set_label_coords(*PLOT_STYLE["y_axis_label_position"])
            ax.tick_params(axis="x", length=0, pad=PLOT_STYLE["x_tick_padding"])
            ax.tick_params(axis="y", length=0, pad=PLOT_STYLE["y_tick_padding"])
            ax.set_facecolor(PLOT_STYLE["plot_background"])
            ax.set_axisbelow(True)
            ax.yaxis.grid(
                True, color=PLOT_STYLE["major_grid_color"],
                linewidth=PLOT_STYLE["major_grid_width"], linestyle=PLOT_STYLE["major_grid_dash"],
            )
            ax.spines[["top", "right"]].set_visible(False)
            handles = [Patch(
                facecolor=color, edgecolor=PLOT_STYLE["bar_edge_color"],
                linewidth=PLOT_STYLE["bar_edge_width"], alpha=PLOT_STYLE["bar_alpha"], label=label,
            ) for _, label, color in STAGES]
            legend = fig.legend(
                handles=handles, loc=PLOT_STYLE["legend_location"],
                bbox_to_anchor=PLOT_STYLE["legend_anchor"], ncol=PLOT_STYLE["legend_columns"],
                prop={"size": PLOT_STYLE["legend_font_size"], "weight": PLOT_STYLE["legend_font_weight"]},
                frameon=PLOT_STYLE["legend_frame_on"], fancybox=True,
                facecolor=PLOT_STYLE["legend_frame_face_color"],
                edgecolor=PLOT_STYLE["legend_frame_edge_color"],
                framealpha=PLOT_STYLE["legend_frame_alpha"],
                borderpad=PLOT_STYLE["legend_frame_border_pad"],
                handlelength=PLOT_STYLE["legend_handle_length"],
                handleheight=PLOT_STYLE["legend_handle_height"],
                columnspacing=PLOT_STYLE["legend_column_spacing"],
            )
            legend.get_frame().set_linewidth(PLOT_STYLE["legend_frame_line_width"])
            left, right, bottom, top = PLOT_STYLE["subplot_margins"]
            fig.subplots_adjust(left=left, right=right, bottom=bottom, top=top)
            output_prefix.parent.mkdir(parents=True, exist_ok=True)
            for suffix in OUTPUT_FORMATS:
                output = output_prefix.parent / f"{output_prefix.name}_{selected}{suffix}"
                fig.savefig(
                    output, dpi=PNG_DPI, bbox_inches="tight", pad_inches=PLOT_STYLE["save_padding"],
                    facecolor=PLOT_STYLE["canvas_background"],
                )
                print(f"Saved {output}")
            plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-csv", "--summary", dest="input_csv", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-prefix", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--mode", choices=("ratio", "time", "both"), default=DEFAULT_MODE)
    args = parser.parse_args()
    times = load_times(args.input_csv.expanduser().resolve())
    plot(times, args.output_prefix.expanduser().resolve(), args.mode)
    print("Ratio denominator: base + intra-block + inter-block stage seconds; loading/saving excluded.")


if __name__ == "__main__":
    main()
