#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = ["matplotlib", "pandas", "typer"]
# ///
"""Plot indexed-versus-exhaustive search decisions from the summary CSV."""

import math
from pathlib import Path
from typing import Annotated

import matplotlib.pyplot as plt
import pandas as pd
import plot_utils as pu
import typer
from matplotlib.lines import Line2D

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
    data["p50_log2_speedup"] = (data["exhaustive_p50_ms"] / data["indexed_p50_ms"]).map(
        math.log2
    )
    data["p95_log2_speedup"] = (data["exhaustive_p95_ms"] / data["indexed_p95_ms"]).map(
        math.log2
    )
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

    fig, axes = plt.subplots(
        len(fill_ratios),
        len(query_counts),
        figsize=(4.2 * len(query_counts), 3.0 * len(fill_ratios) + 2.4),
        sharex=True,
        sharey=True,
        squeeze=False,
    )
    colours = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    reference_colours = {
        reference: colours[index % len(colours)]
        for index, reference in enumerate(references)
    }
    skew_ticks = [value / 100.0 for value in range(0, 51, 10)]

    for panel_row, fill_ratio in enumerate(fill_ratios):
        for panel_column, query_count in enumerate(query_counts):
            ax = axes[panel_row][panel_column]
            panel = data[
                (data["fill_ratio"] == fill_ratio)
                & (data["query_count"] == query_count)
            ]
            for reference in references:
                curve = panel[panel["reference_count"] == reference].sort_values(
                    "hot_fraction"
                )
                if curve.empty:
                    continue
                markevery = max(1, len(curve) // 10)
                colour = reference_colours[reference]
                ax.plot(
                    curve["hot_fraction"],
                    curve["p50_log2_speedup"],
                    color=colour,
                    linewidth=pu.LINE_WIDTH,
                    marker="o",
                    markersize=pu.MARKER_SIZE - 1,
                    markevery=markevery,
                )
                ax.plot(
                    curve["hot_fraction"],
                    curve["p95_log2_speedup"],
                    color=colour,
                    linewidth=pu.LINE_WIDTH,
                    linestyle="--",
                    marker="o",
                    markersize=pu.MARKER_SIZE - 1,
                    markevery=markevery,
                )

            ax.axhline(
                0.0,
                color="#202020",
                linewidth=pu.REFERENCE_LINE_WIDTH,
                linestyle=":",
            )
            ax.set_xlim(0.0, 0.5)
            ax.set_xticks(skew_ticks)
            ax.set_xticklabels(
                [f"{value:.0%}" for value in skew_ticks],
                fontsize=pu.TICK_LABEL_FONT_SIZE,
            )
            ax.set_title(
                pu.paper_text(f"{int(query_count):,} queries", bold=True),
                fontsize=pu.TITLE_FONT_SIZE,
            )
            if panel_column == 0:
                ax.set_ylabel(
                    pu.paper_text(f"{fill_ratio:.0%} fill\nlog2 time ratio"),
                    fontsize=pu.AXIS_LABEL_FONT_SIZE,
                )
            if panel_row == len(fill_ratios) - 1:
                ax.set_xlabel(
                    pu.paper_text("Hot-bucket fraction"),
                    fontsize=pu.AXIS_LABEL_FONT_SIZE,
                )
            ax.tick_params(axis="y", labelsize=pu.TICK_LABEL_FONT_SIZE)
            ax.grid(axis="y", linestyle="--", alpha=pu.GRID_ALPHA)
            ax.set_axisbelow(True)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

    reference_handles = [
        Line2D(
            [0],
            [0],
            color=reference_colours[reference],
            linewidth=pu.LINE_WIDTH,
            marker="o",
            label=f"{int(reference):,} refs",
        )
        for reference in references
    ]
    percentile_handles = [
        Line2D(
            [0],
            [0],
            color="#202020",
            linewidth=pu.LINE_WIDTH,
            linestyle=linestyle,
            marker="o",
            label=percentile,
        )
        for percentile, linestyle in (("p50", "-"), ("p95", "--"))
    ]
    fig.legend(
        handles=[*reference_handles, *percentile_handles],
        loc="upper center",
        bbox_to_anchor=(0.5, 0.85),
        ncol=min(3, len(reference_handles) + len(percentile_handles)),
        fontsize=pu.DEFAULT_FONT_SIZE,
        frameon=False,
    )
    fig.suptitle(
        pu.paper_text("Indexed versus exhaustive search", bold=True),
        fontsize=pu.TITLE_FONT_SIZE,
        y=0.97,
    )
    fig.text(
        0.5,
        0.87,
        pu.paper_text(
            "Above 0 favours indexed; below 0 favours exhaustive.\n"
            "Values are log2 time ratios. Solid p50, dashed p95."
        ),
        fontsize=pu.DEFAULT_FONT_SIZE,
        ha="center",
    )
    fig.subplots_adjust(
        left=0.16,
        right=0.98,
        top=0.70,
        hspace=0.32,
        wspace=0.16,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    pu.save_figure(fig, output, pad_inches=0.10)


if __name__ == "__main__":
    typer.run(main)
