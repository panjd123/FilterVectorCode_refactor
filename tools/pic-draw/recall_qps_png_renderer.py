"""Matplotlib PNG backend for ``plot_recall_qps_block.py``.

This module is intentionally separate from the data preparation and layout
configuration so the main plotting script can keep all user-adjustable values
in one place.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Sequence

import matplotlib

matplotlib.use("Agg")
from matplotlib import pyplot as plt  # noqa: E402
from matplotlib.font_manager import FontProperties  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.transforms import ScaledTranslation  # noqa: E402


MARKERS = {
    "square": "s",
    "triangle": "^",
    "diamond": "D",
    "circle": "o",
    "pentagon": "p",
    "star": "*",
    "hexagon": "h",
    "cross": "X",
    "plus": "P",
}


def _dash_pattern(raw: object) -> object:
    text = str(raw).strip()
    if not text:
        return "solid"
    values = tuple(float(value.strip()) for value in text.split(","))
    return (0, values)


def _font(family: str, size: float, weight: int) -> FontProperties:
    """Build a font from the user-configured fallback family list."""
    families = [name.strip() for name in family.split(",") if name.strip()]
    return FontProperties(family=families or ["serif"], size=size, weight=weight)


def _marker_edge_color(settings: Any, curve_color: str) -> str:
    """Resolve auto to the curve color so marker edges do not cut the line."""
    configured = str(settings.MARKER_EDGE_COLOR)
    return curve_color if configured.lower() == "auto" else configured


def render_png(
    settings: Any,
    all_series: Sequence[Sequence[Any]],
    output_path: Path,
    *,
    pdf_output_path: Path | None = None,
) -> tuple[int, int]:
    """Render a PNG, optionally save a vector PDF, and return PNG dimensions."""
    dataset_count = len(settings.DATASETS)
    columns = min(settings.SUBPLOT_COLUMNS, dataset_count)
    rows = math.ceil(dataset_count / columns)
    canvas_width = int(
        settings.CANVAS_MARGIN_LEFT
        + settings.CANVAS_MARGIN_RIGHT
        + columns * settings.PANEL_WIDTH
        + (columns - 1) * settings.SUBPLOT_GAP_X
    )
    canvas_height = int(
        settings.LEGEND_AREA_HEIGHT
        + rows * settings.PANEL_HEIGHT
        + (rows - 1) * settings.SUBPLOT_GAP_Y
        + settings.CANVAS_MARGIN_BOTTOM
    )
    layout_width = settings.FIGURE_WIDTH or canvas_width
    layout_height = settings.FIGURE_HEIGHT or canvas_height
    output_width = round(layout_width * settings.PNG_DPI / settings.LAYOUT_DPI)
    output_height = round(layout_height * settings.PNG_DPI / settings.LAYOUT_DPI)

    matplotlib.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": [
                "Liberation Serif",
                "STIXGeneral",
                "DejaVu Serif",
            ],
            "mathtext.fontset": "stix",
        }
    )
    figure = plt.figure(
        figsize=(output_width / settings.PNG_DPI, output_height / settings.PNG_DPI),
        dpi=settings.PNG_DPI,
        facecolor=settings.CANVAS_BACKGROUND,
    )

    for index, (dataset, series_list) in enumerate(
        zip(settings.DATASETS, all_series)
    ):
        row, column = divmod(index, columns)
        panel_x = settings.CANVAS_MARGIN_LEFT + column * (
            settings.PANEL_WIDTH + settings.SUBPLOT_GAP_X
        )
        panel_y = settings.LEGEND_AREA_HEIGHT + row * (
            settings.PANEL_HEIGHT + settings.SUBPLOT_GAP_Y
        )
        plot_x = panel_x + settings.PANEL_MARGIN_LEFT
        plot_y = panel_y + settings.PANEL_MARGIN_TOP
        plot_width = (
            settings.PANEL_WIDTH
            - settings.PANEL_MARGIN_LEFT
            - settings.PANEL_MARGIN_RIGHT
        )
        plot_height = (
            settings.PANEL_HEIGHT
            - settings.PANEL_MARGIN_TOP
            - settings.PANEL_MARGIN_BOTTOM
        )
        axes = figure.add_axes(
            [
                plot_x / canvas_width,
                1.0 - (plot_y + plot_height) / canvas_height,
                plot_width / canvas_width,
                plot_height / canvas_height,
            ],
            facecolor=settings.PLOT_BACKGROUND,
        )

        recalls = [point.recall for series in series_list for point in series.points]
        qps_values = [point.qps for series in series_list for point in series.points]
        x_limits = dataset.get("x_limits", settings.X_LIMITS)
        y_limits = dataset.get("y_limits", settings.Y_LIMITS)
        x_low, x_high = x_limits or settings.padded_linear_limits(recalls)
        y_low, y_high = y_limits or settings.padded_log_limits(qps_values)

        axes.set_xlim(x_low, x_high)
        axes.set_ylim(y_low, y_high)
        axes.set_yscale("log", base=settings.Y_LOG_BASE)
        x_ticks = settings.nice_linear_ticks(
            x_low, x_high, settings.X_TICK_COUNT
        )
        y_ticks = settings.log_ticks(y_low, y_high)
        excluded_y_ticks = tuple(float(value) for value in dataset.get("y_tick_exclude", ()))
        y_ticks = [
            tick
            for tick in y_ticks
            if not any(math.isclose(tick, excluded) for excluded in excluded_y_ticks)
        ]
        axes.set_xticks(x_ticks)
        axes.set_xticklabels(
            [settings.fmt_x_tick(value) for value in x_ticks],
            fontproperties=_font(
                settings.X_TICK_FONT_FAMILY, settings.X_TICK_FONT_SIZE, settings.X_TICK_FONT_WEIGHT
            ),
        )
        axes.set_yticks(y_ticks)
        axes.set_yticklabels(
            [settings.fmt_log_tick(value) for value in y_ticks],
            fontproperties=_font(
                settings.Y_TICK_FONT_FAMILY, settings.Y_TICK_FONT_SIZE, settings.Y_TICK_FONT_WEIGHT
            ),
        )
        axes.minorticks_off()
        axes.tick_params(axis="both", which="both", length=0, pad=3)
        tick_label_offsets = {
            float(value): float(offset)
            for value, offset in dataset.get("y_tick_label_offsets", {}).items()
        }
        for tick, label in zip(y_ticks, axes.get_yticklabels()):
            offset_y = next(
                (
                    offset
                    for value, offset in tick_label_offsets.items()
                    if math.isclose(tick, value)
                ),
                0.0,
            )
            if offset_y:
                label.set_transform(
                    label.get_transform()
                    + ScaledTranslation(
                        0,
                        -offset_y / settings.LAYOUT_DPI,
                        figure.dpi_scale_trans,
                    )
                )
        axes.grid(
            True,
            which="major",
            color=settings.GRID_COLOR,
            linewidth=settings.GRID_WIDTH,
            linestyle=(0, tuple(float(v) for v in settings.GRID_DASH.split(","))),
        )
        axes.set_axisbelow(True)
        axes.spines["top"].set_visible(False)
        axes.spines["right"].set_visible(False)
        for side in ("left", "bottom"):
            axes.spines[side].set_color(settings.AXIS_COLOR)
            axes.spines[side].set_linewidth(settings.AXIS_WIDTH)

        for series in settings.series_in_draw_order(series_list):
            axes.plot(
                [point.recall for point in series.points],
                [point.qps for point in series.points],
                color=str(series.style["color"]),
                linewidth=settings.LINE_WIDTH,
                linestyle=_dash_pattern(series.style["dash"]),
                marker=MARKERS.get(str(series.style["marker"]), "o"),
                markersize=settings.MARKER_SIZE,
                markerfacecolor=str(series.style["color"]),
                markeredgecolor=_marker_edge_color(settings, str(series.style["color"])),
                markeredgewidth=settings.MARKER_EDGE_WIDTH,
                clip_on=True,
            )

        subplot_letter = chr(ord("a") + index)
        title = settings.TITLE_FORMAT.format(
            index=index + 1,
            letter=subplot_letter,
            dataset=str(dataset["name"]),
        )
        figure.text(
            (panel_x + settings.PANEL_WIDTH / 2) / canvas_width,
            1.0
            - (
                panel_y
                + 27
                + settings.SUBPLOT_TITLE_OFFSET_Y
            )
            / canvas_height,
            title,
            ha="center",
            va="baseline",
            color=settings.AXIS_COLOR,
            fontproperties=_font(
                settings.SUBPLOT_TITLE_FONT_FAMILY,
                settings.SUBPLOT_TITLE_FONT_SIZE,
                settings.SUBPLOT_TITLE_FONT_WEIGHT,
            ),
        )
        figure.text(
            (
                plot_x
                + plot_width / 2
                + settings.X_AXIS_TITLE_OFFSET_X
            )
            / canvas_width,
            1.0
            - (
                panel_y
                + settings.PANEL_HEIGHT
                - 16
                + settings.X_AXIS_TITLE_OFFSET_Y
            )
            / canvas_height,
            getattr(settings, "X_AXIS_TITLE", "Recall@10"),
            ha="center",
            va="baseline",
            color=settings.AXIS_COLOR,
            fontproperties=_font(
                settings.X_AXIS_TITLE_FONT_FAMILY,
                settings.X_AXIS_TITLE_FONT_SIZE,
                settings.X_AXIS_TITLE_FONT_WEIGHT,
            ),
        )
        show_y_title = (
            column == 0
            or not getattr(settings, "Y_AXIS_TITLE_FIRST_COLUMN_ONLY", False)
        )
        if show_y_title:
            y_label_x = panel_x + 20 + settings.Y_AXIS_TITLE_OFFSET_X
            y_label_y = plot_y + plot_height / 2 + settings.Y_AXIS_TITLE_OFFSET_Y
            figure.text(
                y_label_x / canvas_width,
                1.0 - y_label_y / canvas_height,
                "QPS",
                ha="center",
                va="center",
                rotation=90,
                color=settings.AXIS_COLOR,
                fontproperties=_font(
                    settings.Y_AXIS_TITLE_FONT_FAMILY,
                    settings.Y_AXIS_TITLE_FONT_SIZE,
                    settings.Y_AXIS_TITLE_FONT_WEIGHT,
                ),
            )

    handles: list[Line2D] = []
    for algorithm, style in zip(settings.ALGORITHM_RESULTS, settings.CURVE_STYLES):
        handles.append(
            Line2D(
                [0],
                [0],
                label=algorithm["label"],
                color=str(style["color"]),
                linewidth=settings.LINE_WIDTH,
                linestyle=_dash_pattern(style["dash"]),
                marker=MARKERS.get(str(style["marker"]), "o"),
                markersize=settings.MARKER_SIZE,
                markerfacecolor=str(style["color"]),
                markeredgecolor=_marker_edge_color(settings, str(style["color"])),
                markeredgewidth=settings.MARKER_EDGE_WIDTH,
            )
        )
    legend = figure.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(
            0.5 + settings.LEGEND_X_OFFSET / canvas_width,
            1.0 - settings.LEGEND_TOP / canvas_height,
        ),
        ncol=min(settings.LEGEND_COLUMNS, len(handles)),
        prop=_font(settings.LEGEND_FONT_FAMILY, settings.LEGEND_FONT_SIZE, settings.LEGEND_FONT_WEIGHT),
        frameon=True,
        fancybox=settings.LEGEND_ROUNDED_CORNERS,
        framealpha=1.0,
        facecolor=settings.LEGEND_BACKGROUND_COLOR,
        edgecolor=settings.LEGEND_BORDER_COLOR,
        borderpad=max(settings.LEGEND_BORDER_PADDING, 0),
        handlelength=max(
            settings.LEGEND_LINE_LENGTH, 0.5
        ),
        handletextpad=max(
            settings.LEGEND_GAP_AFTER_LINE, 0
        ),
        columnspacing=settings.LEGEND_COLUMN_SPACING,
        labelspacing=settings.LEGEND_LABEL_SPACING,
    )
    legend.get_frame().set_linewidth(settings.LEGEND_BORDER_WIDTH)
    legend.set_visible(getattr(settings, "SHOW_LEGEND", True))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        figure.savefig(
            output_path,
            format="png",
            dpi=settings.PNG_DPI,
            facecolor=settings.CANVAS_BACKGROUND,
            edgecolor="none",
        )
        if pdf_output_path is not None:
            pdf_output_path.parent.mkdir(parents=True, exist_ok=True)
            # Embed TrueType font subsets to preserve the configured typefaces
            # in LaTeX; compress PDF streams without rasterizing the figure.
            with matplotlib.rc_context({
                "pdf.fonttype": 42,
                "pdf.use14corefonts": False,
                "pdf.compression": 9,
            }):
                figure.savefig(
                    pdf_output_path,
                    format="pdf",
                    facecolor=settings.CANVAS_BACKGROUND,
                    edgecolor="none",
                )
    finally:
        plt.close(figure)
    return output_width, output_height
