#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = ["matplotlib", "pandas", "typer"]
# ///
"""Plot indexed-versus-exhaustive search decisions from the summary CSV."""

from pathlib import Path
from typing import Annotated

import matplotlib as mpl
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
}
MARKERS = ("o", "s", "^", "D", "v", "P", "X")


def main(
    csv_path: Annotated[
        Path, typer.Argument(exists=True, dir_okay=False, help="Decision summary CSV")
    ],
    output: Annotated[
        Path, typer.Option(dir_okay=False, help="Figure output path")
    ] = Path("results/cuddl-search-decision.pdf"),
) -> None:
    """Render the p50/p95 kill-gate boundary for every workload."""
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

    data["p50_speedup"] = data["exhaustive_p50_ms"] / data["indexed_p50_ms"]
    data["p95_speedup"] = data["exhaustive_p95_ms"] / data["indexed_p95_ms"]

    references = sorted(data["reference_count"].unique())
    fill_ratios = sorted(data["fill_ratio"].unique())
    query_counts = sorted(data["query_count"].unique())
    skews = sorted(data["skew"].unique())
    colours = mpl.colormaps["viridis"].resampled(len(references))
    reference_colours = {
        reference: colours(index) for index, reference in enumerate(references)
    }
    reference_sizes = {
        reference: 50 + 30 * index for index, reference in enumerate(references)
    }
    skew_markers = {
        skew: MARKERS[index % len(MARKERS)] for index, skew in enumerate(skews)
    }

    fig, axes = plt.subplots(
        len(query_counts),
        len(fill_ratios),
        figsize=(6 * len(fill_ratios), 5 * len(query_counts)),
        sharex=True,
        sharey=True,
        squeeze=False,
    )
    for row, query_count in enumerate(query_counts):
        for column, fill_ratio in enumerate(fill_ratios):
            ax = axes[row][column]
            workloads = data.loc[
                (data["query_count"] == query_count)
                & (data["fill_ratio"] == fill_ratio)
            ]
            for workload in workloads.sort_values(
                "reference_count", ascending=False
            ).itertuples():
                ax.scatter(
                    workload.p50_speedup,
                    workload.p95_speedup,
                    color=reference_colours[workload.reference_count],
                    marker=skew_markers[workload.skew],
                    edgecolor="black",
                    linewidth=0.5,
                    s=reference_sizes[workload.reference_count],
                    alpha=0.85,
                    zorder=3,
                )
            ax.axvline(1.0, color="#333333", linestyle="--", linewidth=1)
            ax.axhline(1.0, color="#333333", linestyle="--", linewidth=1)
            pu.format_axis(
                ax,
                xlabel=(
                    "Indexed p50 speedup (exhaustive / indexed)"
                    if row == len(query_counts) - 1
                    else ""
                ),
                ylabel=(
                    "Indexed p95 speedup (exhaustive / indexed)" if column == 0 else ""
                ),
                title=f"{fill_ratio:.0%} fill, {int(query_count):,} queries",
                xscale="log",
                yscale="log",
            )

    speedups = pd.concat([data["p50_speedup"], data["p95_speedup"]])
    lower = min(1.0, speedups.min()) / 1.4
    upper = max(1.0, speedups.max()) * 1.4
    for ax in axes.flat:
        ax.set_xlim(lower, upper)
        ax.set_ylim(lower, upper)

    handles = [
        Line2D(
            [],
            [],
            color=reference_colours[reference],
            marker="o",
            linestyle="None",
            markersize=reference_sizes[reference] ** 0.5,
            label=f"References: {int(reference):,}",
        )
        for reference in references
    ]
    handles.extend(
        Line2D(
            [],
            [],
            color="black",
            marker=skew_markers[skew],
            markerfacecolor="white",
            linestyle="None",
            markersize=8,
            label=f"Skew: {skew}",
        )
        for skew in skews
    )
    fig.suptitle(
        pu.paper_text(
            "Search kill gate (top right: indexed; bottom left: exhaustive)",
            bold=True,
        ),
        fontsize=pu.TITLE_FONT_SIZE,
    )
    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=min(len(handles), 5),
        frameon=False,
        fontsize=pu.LEGEND_FONT_SIZE,
    )
    fig.subplots_adjust(bottom=0.13, top=0.92)
    output.parent.mkdir(parents=True, exist_ok=True)
    pu.save_figure(fig, output)


if __name__ == "__main__":
    typer.run(main)
