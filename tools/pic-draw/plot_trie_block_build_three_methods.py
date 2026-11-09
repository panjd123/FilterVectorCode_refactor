#!/usr/bin/env python3
"""Plot the three intra-block graph build-time columns in current_summary.md."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

from plot_index_time_memory import DATASETS, PLOT_STYLE as INDEX_STYLE, log_tick_label


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT = SCRIPT_DIR / "trie_block_build_optimizations/current_summary.md"
DEFAULT_OUTPUT = (
    SCRIPT_DIR / "pic/trie_block_build_optimizations/current_three_methods_intra_block"
)
# 数据列名须与 Markdown 表头一致；METHODS 仅控制图例显示文字。
DATA_COLUMNS = ("CPU", "GPU-Naive", "GPU+LB+TopK")
METHODS = ("CPU", "GPU", "GPU+LB+TopK")
COLORS = ("#2e7d32", "#0aaabd", "#ff8200")
PLOT_STYLE = {
    "figure_size": (16.5, 5.4),
    "bar_width": 0.24,
    "bar_edge_color": "#000000",
    "bar_edge_width": 0.8,
    "font_size": 28,
    "y_limits": (0.1, 1.0e5),
    # 纵轴标题位置 (x, y)，相对于绘图区；x 越小越靠左，y 越大越靠上。
    "y_label_position": (-0.045, 0.5),
    # 图例锚点 (x, y)，相对于绘图区；x 越大越靠右，y 越大越靠上。
    "legend_anchor": (0.5, 0.98),
    # 将图例的哪个位置对齐到上面的锚点。
    "legend_location": "lower center",
    "subplot_margins": (0.08, 0.995, 0.16, 0.82),
}


def load_times(summary: Path) -> np.ndarray:
    """Read only the intra-block graph table, excluding end-to-end times."""
    section = summary.read_text(encoding="utf-8").split(
        "## 块内图构建时间（秒）", 1
    )[1].split("\n## ", 1)[0]
    rows = [
        [cell.strip() for cell in line.strip().strip("|").split("|")]
        for line in section.splitlines() if line.strip().startswith("|")
    ]
    expected_header = ["数据集", *DATA_COLUMNS]
    if len(rows) != len(DATASETS) + 2 or rows[0] != expected_header:
        raise ValueError(
            f"Expected {len(DATASETS)} dataset rows with header {expected_header}; "
            f"found {max(len(rows) - 2, 0)} rows with header {rows[0] if rows else None}."
        )
    if [row[0] for row in rows[2:]] != DATASETS:
        raise ValueError("Unexpected dataset order in the intra-block graph table.")
    values = np.asarray([[float(cell) for cell in row[1:]] for row in rows[2:]])
    if values.shape != (len(DATASETS), len(DATA_COLUMNS)):
        raise ValueError("Expected three times for each dataset.")
    if not np.isfinite(values).all() or np.any(values <= 0):
        raise ValueError("Construction times must be finite and positive.")
    return values


def plot(summary: Path, output_prefix: Path) -> None:
    values = load_times(summary)
    with plt.rc_context({
        "font.family": INDEX_STYLE["font_family"],
        "mathtext.fontset": INDEX_STYLE["mathtext_fontset"],
        "font.size": PLOT_STYLE["font_size"],
        "font.weight": INDEX_STYLE["tick_font_weight"],
        "axes.labelweight": INDEX_STYLE["axis_label_font_weight"],
        "axes.edgecolor": INDEX_STYLE["axis_color"],
        "axes.linewidth": INDEX_STYLE["axis_width"],
        "text.color": INDEX_STYLE["axis_color"],
        "figure.facecolor": INDEX_STYLE["canvas_background"],
    }):
        fig, ax = plt.subplots(figsize=PLOT_STYLE["figure_size"])
        x = np.arange(len(DATASETS))
        width = PLOT_STYLE["bar_width"]
        for i, (method, color) in enumerate(zip(METHODS, COLORS)):
            ax.bar(
                x + (i - 1) * width, values[:, i], width=width * 0.92,
                color=color, label=method, zorder=3,
                edgecolor=PLOT_STYLE["bar_edge_color"],
                linewidth=PLOT_STYLE["bar_edge_width"],
                alpha=INDEX_STYLE["bar_alpha"],
            )
        ax.set_yscale("log")
        ax.set_ylim(*PLOT_STYLE["y_limits"])
        ax.yaxis.set_major_locator(mticker.FixedLocator(10.0 ** np.arange(-1, 6)))
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(log_tick_label))
        ax.minorticks_off()
        ax.set_xticks(x, DATASETS)
        ax.set_ylabel("Intra-block Graph \n Build Time (s)")
        ax.yaxis.set_label_coords(*PLOT_STYLE["y_label_position"])
        ax.tick_params(axis="both", length=0, pad=7)
        ax.margins(x=0.015)
        ax.set_facecolor(INDEX_STYLE["plot_background"])
        ax.set_axisbelow(True)
        ax.yaxis.grid(
            True, color=INDEX_STYLE["major_grid_color"],
            linewidth=INDEX_STYLE["major_grid_width"],
            linestyle=INDEX_STYLE["major_grid_dash"],
        )
        ax.spines[["top", "right"]].set_visible(False)
        legend = ax.legend(
            loc=PLOT_STYLE["legend_location"],
            bbox_to_anchor=PLOT_STYLE["legend_anchor"],
            ncol=3, facecolor="white", edgecolor="#b8b8b8", framealpha=1.0,
            handlelength=1.25, columnspacing=1.5,
        )
        legend.get_frame().set_linewidth(0.8)
        left, right, bottom, top = PLOT_STYLE["subplot_margins"]
        fig.subplots_adjust(left=left, right=right, bottom=bottom, top=top)
        output_prefix.parent.mkdir(parents=True, exist_ok=True)
        for suffix in (".png", ".pdf"):
            output = output_prefix.with_suffix(suffix)
            fig.savefig(output, dpi=400, bbox_inches="tight", pad_inches=0.04)
            print(f"Saved {output}")
        plt.close(fig)
    print(f"Plotted {values.size} intra-block graph measurements from {summary}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-summary", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-prefix", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    plot(args.input_summary.expanduser().resolve(), args.output_prefix.expanduser().resolve())


if __name__ == "__main__":
    main()
