#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = ["matplotlib", "pandas", "typer"]
# ///
"""Plot indexed-versus-exhaustive search decisions from the summary CSV."""

import math
from itertools import pairwise
from pathlib import Path
from typing import Annotated

import matplotlib.pyplot as plt
import pandas as pd
import plot_utils as pu
import typer
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

REQUIRED_COLUMNS = {
    "status",
    "reference_count",
    "fill_ratio",
    "skew",
    "query_count",
    "exhaustive_p50_ms",
    "exhaustive_p95_ms",
    "indexed_p50_ms",
    "indexed_p95_ms",
    "kill_gate_outcome",
}
VALID_OUTCOMES = {"indexed_win", "exhaustive_win", "inconclusive"}


def skew_fraction(value: object) -> float:
    text = str(value)
    if text == "uniform":
        return 0.0
    if text == "hot":
        return 0.5
    return float(text.removesuffix("%")) / 100.0


def main(
    csv_path: Annotated[
        Path, typer.Argument(exists=True, dir_okay=False, help="Decision summary CSV")
    ],
    output: Annotated[
        Path, typer.Option(dir_okay=False, help="Figure output path")
    ] = Path("results/cuddl-search-decision.pdf"),
) -> None:
    """Render speedup curves across the hot-bucket fraction."""
    data = pu.load_csv(csv_path)
    missing = sorted(REQUIRED_COLUMNS - set(data.columns))
    if missing:
        raise typer.BadParameter(f"CSV is missing columns: {', '.join(missing)}")

    data = data.loc[data["status"] == "ok"].copy()
    if data.empty:
        raise typer.BadParameter("CSV has no successful workloads")

    numeric = (
        "reference_count",
        "fill_ratio",
        "query_count",
        "exhaustive_p50_ms",
        "exhaustive_p95_ms",
        "indexed_p50_ms",
        "indexed_p95_ms",
    )
    try:
        data[list(numeric)] = data[list(numeric)].apply(pd.to_numeric)
    except (TypeError, ValueError) as error:
        raise typer.BadParameter(f"CSV contains non-numeric benchmark values: {error}")

    timing_columns = [name for name in numeric if name.endswith("_ms")]
    if (data[timing_columns] <= 0).any().any():
        raise typer.BadParameter("CSV contains non-positive search timings")

    data["hot_fraction"] = data["skew"].map(skew_fraction)
    data["p50_speedup"] = data["exhaustive_p50_ms"] / data["indexed_p50_ms"]
    data["p95_speedup"] = data["exhaustive_p95_ms"] / data["indexed_p95_ms"]
    data["p50_log2_speedup"] = data["p50_speedup"].map(math.log2)
    data["p95_log2_speedup"] = data["p95_speedup"].map(math.log2)
    unknown_outcomes = sorted(set(data["kill_gate_outcome"]) - VALID_OUTCOMES)
    if unknown_outcomes:
        raise typer.BadParameter(
            f"CSV contains unknown kill-gate outcomes: {', '.join(unknown_outcomes)}"
        )

    references = sorted(data["reference_count"].unique())
    fill_ratios = sorted(data["fill_ratio"].unique())
    query_counts = sorted(data["query_count"].unique())
    workload_columns = ["fill_ratio", "query_count", "hot_fraction", "reference_count"]
    if data.duplicated(workload_columns).any():
        raise typer.BadParameter("CSV contains duplicate workloads")

    panel_columns = [
        (fill_ratio, query_count)
        for fill_ratio in fill_ratios
        for query_count in query_counts
    ]
    percentile_columns = (
        ("p50_log2_speedup", "p50"),
        ("p95_log2_speedup", "p95"),
    )
    hot_fractions = sorted(data["hot_fraction"].unique())
    log2_values = pd.concat(
        [data[column] for column, _ in percentile_columns], ignore_index=True
    )
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

    fig, axes = plt.subplots(
        len(percentile_columns),
        len(panel_columns),
        figsize=(3.2 * len(panel_columns), 5.8),
        sharex=True,
        sharey=True,
        squeeze=False,
    )
    mesh = None
    skew_ticks = [value / 100.0 for value in range(0, 51, 10)]
    for percentile_row, (column, percentile) in enumerate(percentile_columns):
        for panel_column, (fill_ratio, query_count) in enumerate(panel_columns):
            ax = axes[percentile_row][panel_column]
            panel = data[
                (data["fill_ratio"] == fill_ratio)
                & (data["query_count"] == query_count)
            ]
            matrix = (
                panel.pivot(
                    index="reference_count",
                    columns="hot_fraction",
                    values=column,
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
            ax.set_xlim(0.0, 0.5)
            ax.set_ylim(len(references) - 0.5, -0.5)
            ax.set_xticks(skew_ticks)
            ax.set_xticklabels(
                [f"{value:.0%}" for value in skew_ticks],
                fontsize=pu.TICK_LABEL_FONT_SIZE,
            )
            ax.set_yticks(range(len(references)))
            ax.set_yticklabels(
                [f"{int(reference):,}" for reference in references],
                fontsize=pu.TICK_LABEL_FONT_SIZE,
            )
            ax.set_title(
                pu.paper_text(
                    f"{fill_ratio:.0%} fill, {int(query_count):,} q",
                    bold=True,
                ),
                fontsize=pu.TITLE_FONT_SIZE,
            )
            if panel_column == 0:
                ax.set_ylabel(
                    pu.paper_text(f"{percentile}\nReferences"),
                    fontsize=pu.AXIS_LABEL_FONT_SIZE,
                )
            if percentile_row == len(percentile_columns) - 1:
                ax.set_xlabel(
                    pu.paper_text("Hot-bucket fraction"),
                    fontsize=pu.AXIS_LABEL_FONT_SIZE,
                )
            else:
                ax.tick_params(axis="x", labelbottom=False)
            ax.tick_params(axis="both", length=0)
            ax.spines[:].set_visible(False)

    if mesh is None:
        raise typer.BadParameter("CSV has no heatmap workloads")
    colorbar = fig.colorbar(
        mesh,
        ax=axes.ravel().tolist(),
        fraction=0.025,
        pad=0.025,
    )
    colorbar.set_label(
        pu.paper_text("log2(exhaustive / indexed time)"),
        fontsize=pu.AXIS_LABEL_FONT_SIZE,
    )
    colorbar.ax.tick_params(labelsize=pu.TICK_LABEL_FONT_SIZE)
    fig.suptitle(
        pu.paper_text("Indexed versus exhaustive search", bold=True),
        fontsize=pu.TITLE_FONT_SIZE,
        y=0.97,
    )
    fig.text(
        0.5,
        0.87,
        pu.paper_text(
            "Blue = indexed faster; orange = exhaustive faster.\n"
            "Colour is log2(exhaustive / indexed); p50 and p95 are separate rows."
        ),
        fontsize=pu.DEFAULT_FONT_SIZE,
        ha="center",
    )
    fig.subplots_adjust(
        left=0.12,
        right=0.84,
        bottom=0.14,
        top=0.70,
        hspace=0.35,
        wspace=0.18,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    pu.save_figure(fig, output, pad_inches=0.10)


if __name__ == "__main__":
    typer.run(main)
