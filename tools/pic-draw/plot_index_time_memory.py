#!/usr/bin/env python3
"""Plot index build time and size from the current eight-dataset summary CSV."""

from __future__ import annotations

import argparse
import ast
import csv
import math
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
from matplotlib.patches import Patch


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT = (
    SCRIPT_DIR.parents[2]
    / "FilterVectorResult/index_build_summary/memory_and_build_summary.csv"
)
DEFAULT_COLOR_SOURCE = SCRIPT_DIR / "plot_recall_qps_all_datasets.py"
DEFAULT_OUTPUT = SCRIPT_DIR / "pic" / "index_memory_comparison"
PNG_DPI = 400

DATASETS = [
    "Genome",
    "Reviews",
    "Amazon",
    "VImg",
    "Music",
    "BReviews",
    "Tiktok",
    "Laion",
]

ALGORITHMS = (
    "Milvus", "ACORN", "NaviX", "RWalks", "PathSeer",
    "Curator", "UNG", "FAVOR", "PTNC",
)
# 参考绘图脚本将 ACORN 的样式键拼写为 AOCRN；CSV 和本图使用 ACORN。
COLOR_LABEL_ALIASES = {"ACORN": "AOCRN"}

# =============================================================================
# 绘图样式配置：通常只需修改这里，不需要改动 plot() 函数。
# =============================================================================
PLOT_STYLE = {
    # 整张图的宽度和高度，单位为英寸。
    "figure_size": (16.5, 9.0),
    # 与 plot_recall_qps_all_datasets.py 的 Times 风格、字号和字重一致。
    "font_family": "Liberation Serif",
    "mathtext_fontset": "stix",
    "base_font_size": 18,
    # 算法顺序：控制每组柱子从左到右的顺序，也是传入图例的顺序。
    # 多行图例按列填充，与 plot_recall_qps_query_thread_k.py 一致。
    # 调整时只需移动名称，必须保留全部九个算法且不能重复。
    "algorithm_order": list(ALGORITHMS),
    # 横、纵坐标刻度文字的字号。
    "x_tick_font_size": 30,
    "y_tick_font_size": 30,
    "tick_font_weight": 700,
    # 上下两个子图的纵轴标题及字号。
    "time_y_axis_label": "Build Time (s)",
    "memory_y_axis_label": "Index Size (GB)",
    "y_axis_label_font_size": 30,
    "axis_label_font_weight": 700,
    # 上方时间子图的纵轴范围；上限略高于 10^5，以完整显示 10^5 刻度。
    "time_y_axis_limits": (1.0, 1.2e5),
    # 上方时间子图固定显示的对数主刻度指数，即 10^0 到 10^5。
    "time_y_axis_tick_exponents": (0, 1, 2, 3, 4, 5),
    # 两个纵轴标题的位置 (x, y)，使用坐标轴相对坐标；增大 y 可向上移动。
    # x 越小越靠左，y=0.5 为垂直居中。
    "time_y_axis_label_position": (-0.055, 0.50),
    "memory_y_axis_label_position": (-0.055, 0.58),
    # 图例样式；九个算法和 PTNC CPU 虚框分成两行，每行五个。
    "legend_font_size": 30,
    "legend_font_weight": 600,
    "legend_column_spacing": 1.5,
    "legend_handle_length": 1.25,
    "legend_handle_height": 0.9,
    # 图例锚点 (x, y)：增大 x 向右移动，增大 y 向上移动。
    "legend_anchor": (0.53, 0.985),
    "legend_columns": 5,
    # 图例灰色圆角框样式。
    "legend_frame_face_color": "#ffffff",
    "legend_frame_edge_color": "#b8b8b8",
    "legend_frame_alpha": 1.0,
    "legend_frame_line_width": 0.8,
    "legend_frame_border_pad": 0.45,
    # 数据集组间距以及组内单根柱子的中心间距。
    "dataset_spacing": 1.18,
    "bar_width": 0.115,
    "bar_fill_ratio": 0.92,
    # 柱子外观。
    "bar_alpha": 0.96,
    "bar_edge_color": "#000000",
    # 实色柱黑边粗细，单位 pt；数值越大越粗，例如 0.5 更细、1.2 更粗。
    # 同时控制时间图、内存图及对应图例的实色边框，不影响 PTNC CPU 虚框。
    "bar_edge_width": 0.5,
    # PTNC 实色柱为最优 GPU 时间；上方白色虚框延伸至 CPU 总构建时间。
    "ptnc_cpu_face_color": "#ffffff",
    "ptnc_cpu_edge_color": "#222222",
    # PTNC CPU 虚线边框粗细，单位 pt；数值越大越粗，也同步作用于 CPU 图例。
    "ptnc_cpu_edge_width": 1.0,
    # 虚线样式：(起始偏移, (线段长度, 空隙长度))；调整粗细请修改上一项。
    "ptnc_cpu_line_style": (0, (3, 2)),
    "ptnc_cpu_legend_label": "PTNC (CPU)",
    # 是否在柱顶显示数值。
    "show_value_labels": False,
    "value_label_font_size": 7.0,
    "value_label_color": "#222222",
    "value_label_offset": 2,
    # 横轴标签与坐标轴之间的距离及图形两端留白。
    "x_tick_padding": 7,
    "x_axis_margin": 0.012,
    # 主、次网格线样式。
    "canvas_background": "#ffffff",
    "plot_background": "#fafafa",
    "axis_color": "#222222",
    "axis_width": 0.9,
    "major_grid_color": "#c9c9c9",
    "major_grid_width": 0.6,
    "major_grid_dash": (0, (2, 2)),
    # 上下子图之间的垂直间距；数值越大，两个子图距离越远。
    "subplot_vertical_spacing": 0.25,
    # 子图边距依次为左、右、下、上；上方空间用于两行图例。
    "subplot_margins": (0.078, 0.995, 0.10, 0.78),
}


def load_algorithm_colors(source_path: Path) -> dict[str, str]:
    """Read the shared palette without executing the recall/QPS script."""
    tree = ast.parse(source_path.read_text(encoding="utf-8"), filename=str(source_path))
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == "ALGORITHM_STYLES"
            for target in node.targets
        ):
            styles = ast.literal_eval(node.value)
            colors = {}
            for algorithm in ALGORITHMS:
                key = algorithm if algorithm in styles else COLOR_LABEL_ALIASES.get(algorithm, algorithm)
                if key not in styles or "color" not in styles[key]:
                    raise ValueError(f"Missing color for {algorithm} in {source_path}")
                colors[algorithm] = styles[key]["color"]
            return colors
    raise ValueError(f"ALGORITHM_STYLES was not found in {source_path}")


def load_series(
    input_csv: Path, color_source: Path = DEFAULT_COLOR_SOURCE,
) -> list[tuple[str, str, list[float], list[float]]]:
    """Use CSV sizes/times as recorded, including their existing memory scopes."""
    colors = load_algorithm_colors(color_source)
    values = {}
    with input_csv.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"dataset", "algorithm", "loaded_memory_gib", "build_time_seconds"}
        missing_columns = required - set(reader.fieldnames or ())
        if missing_columns:
            raise ValueError(f"Missing CSV columns: {sorted(missing_columns)}")
        for row in reader:
            key = row["dataset"].strip(), row["algorithm"].strip()
            if key[0] not in DATASETS or key[1] not in ALGORITHMS:
                raise ValueError(f"Unexpected dataset/algorithm {key} at {input_csv}:{reader.line_num}")
            if key in values:
                raise ValueError(f"Duplicate dataset/algorithm {key} at {input_csv}:{reader.line_num}")
            try:
                values[key] = float(row["loaded_memory_gib"]), float(row["build_time_seconds"])
            except (TypeError, ValueError) as exc:
                raise ValueError(f"Invalid size/time for {key} at {input_csv}:{reader.line_num}") from exc
    missing = [(dataset, algorithm) for dataset in DATASETS for algorithm in ALGORITHMS
               if (dataset, algorithm) not in values]
    if missing:
        raise ValueError(f"Missing dataset/algorithm records in {input_csv}: {missing}")
    series = [
        (algorithm, colors[algorithm],
         [values[dataset, algorithm][0] for dataset in DATASETS],
         [values[dataset, algorithm][1] for dataset in DATASETS])
        for algorithm in ALGORITHMS
    ]
    return ordered_series(series)


def load_ptnc_cpu_times(input_csv: Path) -> np.ndarray | None:
    """Read optional CPU end-to-end times, requiring all eight PTNC records."""
    with input_csv.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if "cpu_build_time_seconds" not in (reader.fieldnames or ()):
            return None
        times = {
            row["dataset"].strip(): float(row["cpu_build_time_seconds"])
            for row in reader if row["algorithm"].strip() == "PTNC"
        }
    values = np.asarray([times[dataset] for dataset in DATASETS], dtype=float)
    if not np.isfinite(values).all() or np.any(values <= 0):
        raise ValueError("PTNC CPU construction times must be finite and positive.")
    return values


def ordered_series(
    series: list[tuple[str, str, list[float], list[float]]],
) -> list[tuple[str, str, list[float], list[float]]]:
    """Return series in the configured order after validating all names."""
    available = {item[0]: item for item in series}
    order = PLOT_STYLE["algorithm_order"]
    duplicates = sorted({name for name in order if order.count(name) > 1})
    unknown = sorted(set(order) - set(available))
    missing = sorted(set(available) - set(order))
    if duplicates or unknown or missing or len(order) != len(series):
        raise ValueError(
            "Invalid algorithm_order; "
            f"duplicates={duplicates}, unknown={unknown}, missing={missing}."
        )
    return [available[name] for name in order]


def validate_data(
    series: list[tuple[str, str, list[float], list[float]]],
) -> tuple[np.ndarray, np.ndarray]:
    """Validate and return the memory and construction-time matrices."""
    memory_values = np.asarray([item[2] for item in series], dtype=float)
    time_values = np.asarray([item[3] for item in series], dtype=float)
    expected_shape = (len(series), len(DATASETS))
    for name, values in (("memory", memory_values), ("time", time_values)):
        if values.shape != expected_shape:
            raise ValueError(
                f"Expected {name} shape {expected_shape}, got {values.shape}."
            )
        if not np.isfinite(values).all():
            raise ValueError(f"All {name} values must be finite.")
        if np.any(values <= 0):
            raise ValueError(
                f"A logarithmic y-axis requires positive {name} values."
            )
    return memory_values, time_values


def value_label(value: float) -> str:
    """Format optional labels above the bars."""
    return f"{value:.1f}" if value < 1000 else f"{value / 1000:.1f}k"


def log_tick_label(value: float, _position: int | None = None) -> str:
    """Use the reference plot's bold STIX powers of ten."""
    return rf"$\mathbf{{10^{{{round(math.log10(value))}}}}}$"


def draw_panel(
    ax,
    x: np.ndarray,
    offsets: np.ndarray,
    values: np.ndarray,
    series: list[tuple[str, str, list[float], list[float]]],
    y_axis_label: str,
    y_axis_label_position: tuple[float, float],
    show_x_tick_labels: bool,
) -> None:
    """Draw one grouped-bar panel using the shared algorithm colors."""
    width = PLOT_STYLE["bar_width"]
    for index, ((label, color, _, _), y_values) in enumerate(zip(series, values)):
        positions = x + offsets[index]
        ax.bar(
            positions,
            y_values,
            width=width * PLOT_STYLE["bar_fill_ratio"],
            color=color,
            edgecolor=PLOT_STYLE["bar_edge_color"],
            linewidth=PLOT_STYLE["bar_edge_width"],
            alpha=PLOT_STYLE["bar_alpha"],
            label=label,
            zorder=3,
        )

        if PLOT_STYLE["show_value_labels"]:
            for xpos, value in zip(positions, y_values):
                ax.annotate(
                    value_label(float(value)),
                    (xpos, value),
                    xytext=(0, PLOT_STYLE["value_label_offset"]),
                    textcoords="offset points",
                    ha="center",
                    va="bottom",
                    rotation=90,
                    fontsize=PLOT_STYLE["value_label_font_size"],
                    color=PLOT_STYLE["value_label_color"],
                    clip_on=False,
                )

    all_values = values.ravel()
    lower_limit = 10 ** math.floor(math.log10(float(np.min(all_values))))
    upper_limit = 10 ** math.ceil(math.log10(float(np.max(all_values))))
    ax.set_yscale("log")
    ax.set_ylim(lower_limit, upper_limit)
    ax.yaxis.set_major_locator(mticker.LogLocator(base=10.0))
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(log_tick_label))
    ax.minorticks_off()

    ax.set_xticks(x, DATASETS)
    ax.tick_params(
        axis="x",
        length=0,
        pad=PLOT_STYLE["x_tick_padding"],
        labelbottom=show_x_tick_labels,
    )
    ax.tick_params(axis="y", which="major", length=0, pad=5)
    ax.tick_params(axis="y", which="minor", length=0)
    for label in (*ax.get_xticklabels(), *ax.get_yticklabels()):
        label.set_fontfamily(PLOT_STYLE["font_family"])
        label.set_fontweight(PLOT_STYLE["tick_font_weight"])
    ax.margins(x=PLOT_STYLE["x_axis_margin"])
    ax.set_ylabel(y_axis_label, fontweight=PLOT_STYLE["axis_label_font_weight"])
    ax.yaxis.set_label_coords(*y_axis_label_position)

    ax.set_axisbelow(True)
    ax.set_facecolor(PLOT_STYLE["plot_background"])
    ax.yaxis.grid(
        True,
        which="major",
        color=PLOT_STYLE["major_grid_color"],
        linestyle=PLOT_STYLE["major_grid_dash"],
        linewidth=PLOT_STYLE["major_grid_width"],
    )
    ax.xaxis.grid(False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(PLOT_STYLE["axis_color"])
        ax.spines[side].set_linewidth(PLOT_STYLE["axis_width"])


def plot(
    output_prefix: Path, input_csv: Path = DEFAULT_INPUT,
    color_source: Path = DEFAULT_COLOR_SOURCE,
) -> None:
    """Draw construction-time and index-size panels and save a 400 DPI PNG."""
    series = load_series(input_csv, color_source)
    memory_values, time_values = validate_data(series)
    cpu_times = load_ptnc_cpu_times(input_csv)
    ptnc_index = next(i for i, item in enumerate(series) if item[0] == "PTNC")
    if cpu_times is not None and np.any(cpu_times < time_values[ptnc_index]):
        raise ValueError("PTNC CPU times must be at least the GPU times for the extension bars.")
    print(f"Loaded {len(series) * len(DATASETS)} records from {input_csv}")

    rc = {
        "font.family": PLOT_STYLE["font_family"],
        "font.serif": ["Liberation Serif", "STIXGeneral", "DejaVu Serif"],
        "mathtext.fontset": PLOT_STYLE["mathtext_fontset"],
        "font.size": PLOT_STYLE["base_font_size"],
        "axes.labelsize": PLOT_STYLE["y_axis_label_font_size"],
        "xtick.labelsize": PLOT_STYLE["x_tick_font_size"],
        "ytick.labelsize": PLOT_STYLE["y_tick_font_size"],
        "axes.labelweight": PLOT_STYLE["axis_label_font_weight"],
        "axes.linewidth": PLOT_STYLE["axis_width"],
        "text.color": PLOT_STYLE["axis_color"],
        "axes.labelcolor": PLOT_STYLE["axis_color"],
        "xtick.color": PLOT_STYLE["axis_color"],
        "ytick.color": PLOT_STYLE["axis_color"],
        "figure.facecolor": PLOT_STYLE["canvas_background"],
    }

    with plt.rc_context(rc):
        fig, (time_ax, memory_ax) = plt.subplots(
            2,
            1,
            figsize=PLOT_STYLE["figure_size"],
            sharex=True,
        )
        x = np.arange(len(DATASETS), dtype=float) * PLOT_STYLE["dataset_spacing"]
        offsets = (
            np.arange(len(series), dtype=float) - (len(series) - 1) / 2
        ) * PLOT_STYLE["bar_width"]

        draw_panel(
            time_ax,
            x,
            offsets,
            time_values,
            series,
            PLOT_STYLE["time_y_axis_label"],
            PLOT_STYLE["time_y_axis_label_position"],
            show_x_tick_labels=True,
        )
        if cpu_times is not None:
            # 在线性时间上计算差值，再交由对数轴变换；虚框顶部恰好为 CPU 总时间。
            gpu_times = time_values[ptnc_index]
            time_ax.bar(
                x + offsets[ptnc_index],
                cpu_times - gpu_times,
                bottom=gpu_times,
                width=PLOT_STYLE["bar_width"] * PLOT_STYLE["bar_fill_ratio"],
                facecolor=PLOT_STYLE["ptnc_cpu_face_color"],
                edgecolor=PLOT_STYLE["ptnc_cpu_edge_color"],
                linewidth=PLOT_STYLE["ptnc_cpu_edge_width"],
                linestyle=PLOT_STYLE["ptnc_cpu_line_style"],
                label=PLOT_STYLE["ptnc_cpu_legend_label"],
                zorder=4,
            )
        time_ax.set_ylim(*PLOT_STYLE["time_y_axis_limits"])
        time_ax.yaxis.set_major_locator(
            mticker.FixedLocator(
                10.0 ** np.asarray(PLOT_STYLE["time_y_axis_tick_exponents"])
            )
        )
        time_ax.yaxis.set_major_formatter(mticker.FuncFormatter(log_tick_label))
        draw_panel(
            memory_ax,
            x,
            offsets,
            memory_values,
            series,
            PLOT_STYLE["memory_y_axis_label"],
            PLOT_STYLE["memory_y_axis_label_position"],
            show_x_tick_labels=True,
        )

        legend_handles = [
            Patch(
                facecolor=color,
                edgecolor=PLOT_STYLE["bar_edge_color"],
                linewidth=PLOT_STYLE["bar_edge_width"],
                alpha=PLOT_STYLE["bar_alpha"],
                label=label,
            )
            for label, color, _, _ in series
        ]
        if cpu_times is not None:
            legend_handles.append(Patch(
                facecolor=PLOT_STYLE["ptnc_cpu_face_color"],
                edgecolor=PLOT_STYLE["ptnc_cpu_edge_color"],
                linewidth=PLOT_STYLE["ptnc_cpu_edge_width"],
                linestyle=PLOT_STYLE["ptnc_cpu_line_style"],
                label=PLOT_STYLE["ptnc_cpu_legend_label"],
            ))
        # 与 query_thread_k 使用的共享 renderer 一样，保留算法列表顺序，
        # 让 Matplotlib 按列填充两行图例，不再额外重排 handles。
        legend = fig.legend(
            handles=legend_handles,
            loc="upper center",
            bbox_to_anchor=PLOT_STYLE["legend_anchor"],
            ncol=PLOT_STYLE["legend_columns"],
            frameon=True,
            fancybox=True,
            facecolor=PLOT_STYLE["legend_frame_face_color"],
            edgecolor=PLOT_STYLE["legend_frame_edge_color"],
            framealpha=PLOT_STYLE["legend_frame_alpha"],
            borderpad=PLOT_STYLE["legend_frame_border_pad"],
            handlelength=PLOT_STYLE["legend_handle_length"],
            handleheight=PLOT_STYLE["legend_handle_height"],
            columnspacing=PLOT_STYLE["legend_column_spacing"],
            prop={"family": PLOT_STYLE["font_family"],
                  "size": PLOT_STYLE["legend_font_size"],
                  "weight": PLOT_STYLE["legend_font_weight"]},
        )
        legend.get_frame().set_linewidth(PLOT_STYLE["legend_frame_line_width"])

        left, right, bottom, top = PLOT_STYLE["subplot_margins"]
        fig.subplots_adjust(
            left=left,
            right=right,
            bottom=bottom,
            top=top,
            hspace=PLOT_STYLE["subplot_vertical_spacing"],
        )

        output_prefix.parent.mkdir(parents=True, exist_ok=True)
        output_path = output_prefix.with_suffix(".png")
        fig.savefig(
            output_path,
            dpi=PNG_DPI,
            bbox_inches="tight",
            pad_inches=0.025,
            facecolor=PLOT_STYLE["canvas_background"],
        )
        print(f"Saved {output_path} ({PNG_DPI} DPI)")
        plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-csv", type=Path, default=DEFAULT_INPUT,
        help="Summary CSV containing loaded_memory_gib and build_time_seconds.",
    )
    parser.add_argument(
        "--color-source", type=Path, default=DEFAULT_COLOR_SOURCE,
        help="Python source whose ALGORITHM_STYLES dictionary supplies the colors.",
    )
    parser.add_argument(
        "--output-prefix",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Output path without extension; only a 400 DPI PNG is written.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    plot(
        args.output_prefix.expanduser().resolve(),
        args.input_csv.expanduser().resolve(),
        args.color_source.expanduser().resolve(),
    )


if __name__ == "__main__":
    main()
