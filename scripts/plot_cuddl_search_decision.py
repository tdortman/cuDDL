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
    "skew",
    "query_count",
    "index_mode",
    "exhaustive_median_ms",
    "indexed_median_ms",
    "kill_gate_outcome",
}
VALID_OUTCOMES = {"indexed_win", "exhaustive_win", "inconclusive", "recall_loss"}


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
    # The 16-bit key is the canonical index key; the 15-bit tradeoff has its
    # own figure (plot_cuddl_search_key_bits.py).
    data["key_bits"] = data["index_mode"].str.extract(r"b(\d+)_k(\d+)")[1].astype(int)
    data = data.loc[data["key_bits"] == 16].copy()
    if data.empty:
        raise typer.BadParameter("CSV has no successful workloads")
    numeric = (
        "reference_count",
        "query_count",
        "exhaustive_median_ms",
        "indexed_median_ms",
    )
    try:
        data[list(numeric)] = data[list(numeric)].apply(pd.to_numeric)
    except (TypeError, ValueError) as error:
        raise typer.BadParameter(f"CSV contains non-numeric benchmark values: {error}")

    timing_columns = [name for name in numeric if name.endswith("_ms")]
    if (data[timing_columns] <= 0).any().any():
        raise typer.BadParameter("CSV contains non-positive search timings")

    data["hot_fraction"] = data["skew"].map(skew_fraction)
    data["speedup"] = data["exhaustive_median_ms"] / data["indexed_median_ms"]
    data["log2_speedup"] = data["speedup"].map(math.log2)
    unknown_outcomes = sorted(set(data["kill_gate_outcome"]) - VALID_OUTCOMES)
    if unknown_outcomes:
        raise typer.BadParameter(
            f"CSV contains unknown kill-gate outcomes: {', '.join(unknown_outcomes)}"
        )

    references = sorted(data["reference_count"].unique())
    query_counts = sorted(data["query_count"].unique())
    workload_columns = [
        "query_count",
        "hot_fraction",
        "reference_count",
    ]
    if data.duplicated(workload_columns).any():
        raise typer.BadParameter("CSV contains duplicate workloads")

    panel_columns = [(query_count,) for query_count in query_counts]
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

    # One wide row: a panel per query count, slide-friendly.
    fig, axes = plt.subplots(
        1,
        len(panel_columns),
        figsize=(4.6 * len(panel_columns), 3.4),
        sharex=True,
        sharey=True,
        squeeze=False,
    )
    mesh = None
    skew_ticks = [value / 100.0 for value in range(0, 51, 10)]
    for panel_column, (query_count,) in enumerate(panel_columns):
        ax = axes[0][panel_column]
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
            pu.paper_text(f"{int(query_count):,} q", bold=True),
            fontsize=pu.TITLE_FONT_SIZE,
        )
        if panel_column == 0:
            ax.set_ylabel(
                pu.paper_text("References"),
                fontsize=pu.AXIS_LABEL_FONT_SIZE,
            )
        ax.set_xlabel(
            pu.paper_text("Hot-bucket fraction"),
            fontsize=pu.AXIS_LABEL_FONT_SIZE,
        )
        ax.tick_params(axis="both", length=0)
        ax.spines[:].set_visible(False)

    if mesh is None:
        raise typer.BadParameter("CSV has no heatmap workloads")
    colorbar = fig.colorbar(
        mesh,
        ax=axes.ravel().tolist(),
        orientation="horizontal",
        location="bottom",
        fraction=0.06,
        pad=0.13,
    )
    colorbar.set_label(
        pu.paper_text("log2(exhaustive / indexed median time)"),
        fontsize=pu.AXIS_LABEL_FONT_SIZE,
    )
    colorbar.ax.tick_params(labelsize=pu.TICK_LABEL_FONT_SIZE)
    fig.suptitle(
        pu.paper_text("Indexed versus exhaustive search", bold=True),
        fontsize=pu.TITLE_FONT_SIZE,
        y=0.995,
    )
    fig.subplots_adjust(
        left=0.12,
        right=0.96,
        bottom=0.26,
        top=0.90,
        wspace=0.22,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    pu.save_figure(fig, output, pad_inches=0.10)


if __name__ == "__main__":
    typer.run(main)
