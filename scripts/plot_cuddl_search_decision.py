#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = ["jsonschema", "matplotlib", "pandas", "typer"]
# ///
"""Plot indexed-versus-exhaustive search decisions from shared result JSON."""

import math
from itertools import pairwise
from pathlib import Path
from typing import Annotated

import matplotlib.pyplot as plt
import pandas as pd
import plot_utils as pu
import typer
from benchmark_schema import flatten_measurements, load_result
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from matplotlib.patches import Patch

REQUIRED_COLUMNS = {
    "status",
    "reference_count",
    "skew",
    "query_count",
    "query_profile",
    "fill_ratio",
    "indexed_bucket_count",
    "index_mode",
    "exhaustive_p50_ms",
    "indexed_p50_ms",
    "kill_gate_outcome",
}
VALID_OUTCOMES = {"indexed_win", "exhaustive_win", "inconclusive", "recall_loss"}
REFSEQ_REFERENCE_COUNT = 200687
PAPER_INDEXED_BUCKET_COUNT = 2048
PAPER_FILL_RATIO = 1.0
PAPER_QUERY_PROFILE = "copied"


def skew_fraction(value: object) -> float:
    text = str(value)
    if text == "uniform":
        return 0.0
    if text == "hot":
        return 0.5
    return float(text.removesuffix("%")) / 100.0


def speedup_label(log2_speedup: float) -> str:
    """Format a log2 speedup as the actual exhaustive/indexed time ratio."""
    ratio = 2.0**log2_speedup
    if ratio >= 100.0:
        return f"{ratio:.0f}"
    if ratio >= 10.0:
        return f"{ratio:.1f}"
    return f"{ratio:.2g}"


def main(
    json_path: Annotated[
        Path,
        typer.Argument(exists=True, dir_okay=False, help="Search-decision result JSON"),
    ],
    output: Annotated[
        Path, typer.Option(dir_okay=False, help="Figure output path")
    ] = Path("results/cuddl-search-decision.pdf"),
) -> None:
    """Render speedup curves across the hot-bucket fraction."""
    try:
        result = load_result(json_path, "search_decision")
    except ValueError as error:
        raise typer.BadParameter(f"{json_path}: {error}") from error
    data = pd.DataFrame(flatten_measurements(result))
    missing = sorted(REQUIRED_COLUMNS - set(data.columns))
    if missing:
        raise typer.BadParameter(f"result is missing columns: {', '.join(missing)}")

    data = data.loc[data["status"] == "ok"].copy()
    # The 16-bit key is the canonical index key; the 15-bit tradeoff has its
    # own figure (plot_cuddl_search_key_bits.py).
    data["key_bits"] = (
        data["index_mode"].astype("string").str.extract(r"b(\d+)_k(\d+)")[1].astype(int)
    )
    data = data.loc[data["key_bits"] == 16].copy()
    if data.empty:
        raise typer.BadParameter("result has no successful workloads")
    numeric = (
        "reference_count",
        "query_count",
        "fill_ratio",
        "indexed_bucket_count",
        "exhaustive_p50_ms",
        "indexed_p50_ms",
    )
    try:
        data[list(numeric)] = data[list(numeric)].apply(pd.to_numeric)
    except (TypeError, ValueError) as error:
        raise typer.BadParameter(
            f"result contains non-numeric benchmark values: {error}"
        )

    timing_columns = [name for name in numeric if name.endswith("_ms")]
    if (data[timing_columns] <= 0).any().any():
        raise typer.BadParameter("result contains non-positive search timings")

    data = data.loc[
        (data["indexed_bucket_count"] == PAPER_INDEXED_BUCKET_COUNT)
        & (data["fill_ratio"] == PAPER_FILL_RATIO)
        & (data["query_profile"] == PAPER_QUERY_PROFILE)
    ].copy()
    if data.empty:
        raise typer.BadParameter("result has no canonical paper workloads")

    data["hot_fraction"] = data["skew"].map(skew_fraction)
    data["speedup"] = data["exhaustive_p50_ms"] / data["indexed_p50_ms"]
    data["log2_speedup"] = data["speedup"].map(math.log2)
    unknown_outcomes = sorted(set(data["kill_gate_outcome"]) - VALID_OUTCOMES)
    if unknown_outcomes:
        raise typer.BadParameter(
            f"result contains unknown kill-gate outcomes: {', '.join(unknown_outcomes)}"
        )

    references = sorted(data["reference_count"].unique())
    query_counts = sorted(data["query_count"].unique())
    workload_columns = [
        "query_count",
        "hot_fraction",
        "reference_count",
        "fill_ratio",
        "indexed_bucket_count",
        "query_profile",
    ]
    if data.duplicated(workload_columns).any():
        raise typer.BadParameter("result contains duplicate workloads")

    # The realistic query regime is a single query and a 128-query batch; older
    # result JSONs carried more query-count axes that no longer reflect the paper workload.
    panel_columns = [
        (query_count,) for query_count in query_counts if query_count in {1, 128}
    ]
    if not panel_columns:
        raise typer.BadParameter("result has no 1-query or 128-query workloads")
    hot_fractions = sorted(data["hot_fraction"].unique())
    log2_values = data["log2_speedup"]
    maximum_log2 = max(1.0, float(log2_values.abs().max()))
    colour_map = LinearSegmentedColormap.from_list(
        "search_decision",
        ["#f4a582", "#ffffff", "#92c5de"],
    )
    colour_norm = TwoSlopeNorm(
        vmin=-maximum_log2,
        vcenter=0.0,
        vmax=maximum_log2,
    )

    if len(hot_fractions) == 1:
        half_step = 0.05
    else:
        half_step = min(right - left for left, right in pairwise(hot_fractions)) / 2.0
    x_edges = [
        hot_fractions[0] - half_step,
        *[(left + right) / 2.0 for left, right in pairwise(hot_fractions)],
        hot_fractions[-1] + half_step,
    ]
    y_edges = [index - 0.5 for index in range(len(references) + 1)]

    # Stacked panels with square cells: the grid is 11 hot fractions by 3
    # reference counts, so a panel is 11/3 as wide as it is tall. Size the
    # figure in inches so each cell stays readable and annotated.
    cell_size = 0.42
    panel_width = len(hot_fractions) * cell_size
    panel_height = len(references) * cell_size
    left_margin = 1.55
    bottom_margin = 0.56
    top_margin = 0.85
    panel_gap = 0.35
    # Gap to panels + tick-label room + bar + rotated label + right margin.
    colorbar_space = 0.10 + 0.55 + 0.06 + 0.95
    fig_width = left_margin + panel_width + colorbar_space
    fig_height = (
        bottom_margin + len(panel_columns) * panel_height + panel_gap + top_margin
    )
    fig, axes = plt.subplots(
        len(panel_columns),
        1,
        figsize=(fig_width, fig_height),
        sharex=True,
        sharey=True,
        squeeze=False,
    )
    mesh = None
    skew_ticks = [value / 100.0 for value in range(0, 51, 5)]
    for panel_column, (query_count,) in enumerate(panel_columns):
        ax = axes[panel_column][0]
        panel = data[data["query_count"] == query_count]
        matrix = (
            panel.pivot(
                index="reference_count",
                columns="hot_fraction",
                values="log2_speedup",
            )
            .reindex(index=references, columns=hot_fractions)
            .to_numpy(dtype=float)
        )
        mesh = ax.pcolormesh(
            x_edges,
            y_edges,
            matrix,
            cmap=colour_map,
            norm=colour_norm,
            shading="flat",
            edgecolors="white",
            linewidth=0.8,
        )
        ax.set_xlim(x_edges[0], x_edges[-1])
        ax.set_ylim(len(references) - 0.5, -0.5)
        ax.set_aspect(x_edges[1] - x_edges[0], adjustable="box")
        for row_index in range(len(references)):
            for column_index, value in enumerate(matrix[row_index]):
                if not math.isfinite(value):
                    continue
                ax.text(
                    (x_edges[column_index] + x_edges[column_index + 1]) / 2.0,
                    row_index,
                    r"$\times$" + speedup_label(value),
                    ha="center",
                    va="center",
                    fontsize=10,
                    color="#1a1a1a",
                )
        ax.set_xticks(skew_ticks)
        ax.set_xticklabels(
            [f"{value:.0%}" for value in skew_ticks],
            fontsize=pu.TICK_LABEL_FONT_SIZE,
        )
        ax.set_yticks(range(len(references)))
        ax.set_yticklabels(
            [
                f"{int(reference):,} (RefSeq)"
                if reference == REFSEQ_REFERENCE_COUNT
                else f"{int(reference):,} (synthetic)"
                for reference in references
            ],
            fontsize=pu.TICK_LABEL_FONT_SIZE,
        )
        ax.set_title(
            pu.paper_text(
                f"{int(query_count):,} quer{'y' if query_count == 1 else 'ies'}",
                bold=True,
            ),
            fontsize=pu.TITLE_FONT_SIZE,
        )
        if panel_column == 0:
            ax.set_ylabel(
                pu.paper_text("References"),
                fontsize=pu.AXIS_LABEL_FONT_SIZE,
            )
        if panel_column == len(panel_columns) - 1:
            ax.set_xlabel(
                pu.paper_text("Hot buckets (score 1 for 1% of references)"),
                fontsize=pu.AXIS_LABEL_FONT_SIZE,
            )
        ax.tick_params(axis="both", length=0)
        ax.spines[:].set_visible(False)

    if mesh is None:
        raise typer.BadParameter("result has no heatmap workloads")
    fig.subplots_adjust(
        left=left_margin / fig_width,
        right=(left_margin + panel_width) / fig_width,
        bottom=bottom_margin / fig_height,
        top=(bottom_margin + len(panel_columns) * panel_height + panel_gap)
        / fig_height,
        hspace=panel_gap / panel_height,
    )
    colorbar_axes = fig.add_axes(
        (
            (left_margin + panel_width + 0.10 + 0.55) / fig_width,
            bottom_margin / fig_height,
            0.06 / fig_width,
            (len(panel_columns) * panel_height + panel_gap) / fig_height,
        )
    )
    colorbar = fig.colorbar(mesh, cax=colorbar_axes)
    # Ticks stay on the log2 scale (diverging colours), but the labels and the
    # cell annotations show the actual exhaustive / indexed time ratio.
    colour_ticks = [
        tick for tick in range(-16, 17, 4) if -maximum_log2 <= tick <= maximum_log2
    ]
    colorbar.set_ticks(colour_ticks)
    colorbar.set_ticklabels(
        [f"1/{2**-tick}" if tick < 0 else str(2**tick) for tick in colour_ticks]
    )
    colorbar.set_label(
        pu.paper_text("exhaustive / indexed median time") + r" ($\times$)",
        fontsize=pu.AXIS_LABEL_FONT_SIZE,
    )
    colorbar.ax.tick_params(labelsize=pu.TICK_LABEL_FONT_SIZE)
    fig.suptitle(
        pu.paper_text("Indexed versus exhaustive search", bold=True),
        fontsize=pu.TITLE_FONT_SIZE,
        y=1.0 - 0.06 / fig_height,
    )
    fig.legend(
        handles=[
            Patch(color="#92c5de", label=pu.paper_text("Indexed faster")),
            Patch(color="#f4a582", label=pu.paper_text("Exhaustive faster")),
        ],
        loc="upper left",
        bbox_to_anchor=(0.005, 1.0 - 0.22 / fig_height),
        frameon=False,
        fontsize=12,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    pu.save_figure(fig, output, pad_inches=0.10)


if __name__ == "__main__":
    typer.run(main)
