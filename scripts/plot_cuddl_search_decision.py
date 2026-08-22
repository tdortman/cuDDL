#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = ["matplotlib", "pandas", "typer"]
# ///
"""Plot indexed-versus-exhaustive search decisions from the summary CSV."""

from pathlib import Path
from typing import Annotated

import matplotlib.pyplot as plt
import pandas as pd
import plot_utils as pu
import typer
from matplotlib.patches import Rectangle

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
OUTCOMES = {
    "indexed_win": ("#92c5de", "Indexed"),
    "exhaustive_win": ("#f4a582", "Exhaustive"),
    "inconclusive": ("#d9d9d9", "Mixed"),
}


def factor_text(factor: float) -> str:
    return f"{factor:.3f}x" if factor < 1.01 else f"{factor:.3g}x"


def percentile_result(name: str, speedup: float) -> str:
    if speedup > 1.0:
        return f"{name}: I {factor_text(speedup)}"
    if speedup < 1.0:
        return f"{name}: E {factor_text(1.0 / speedup)}"
    return f"{name}: tie"


def main(
    csv_path: Annotated[
        Path, typer.Argument(exists=True, dir_okay=False, help="Decision summary CSV")
    ],
    output: Annotated[
        Path, typer.Option(dir_okay=False, help="Figure output path")
    ] = Path("results/cuddl-search-decision.pdf"),
) -> None:
    """Render the search winner and conservative speedup for every workload."""
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
    unknown_outcomes = sorted(set(data["kill_gate_outcome"]) - set(OUTCOMES))
    if unknown_outcomes:
        raise typer.BadParameter(
            f"CSV contains unknown kill-gate outcomes: {', '.join(unknown_outcomes)}"
        )

    references = sorted(data["reference_count"].unique())
    fill_ratios = sorted(data["fill_ratio"].unique())
    query_counts = sorted(data["query_count"].unique())
    skews = sorted(data["skew"].unique())
    workloads = {
        (
            workload.fill_ratio,
            workload.query_count,
            workload.skew,
            workload.reference_count,
        ): workload
        for workload in data.itertuples()
    }
    if len(workloads) != len(data):
        raise typer.BadParameter("CSV contains duplicate workloads")

    fig, axes = plt.subplots(
        len(skews),
        len(fill_ratios),
        figsize=(3.6 * len(fill_ratios), 2.3 * len(skews) + 1.6),
        sharex=True,
        sharey=True,
        squeeze=False,
    )
    for panel_row, skew in enumerate(skews):
        for panel_column, fill_ratio in enumerate(fill_ratios):
            ax = axes[panel_row][panel_column]
            for reference_row, reference in enumerate(references):
                for query_column, query_count in enumerate(query_counts):
                    workload = workloads.get((fill_ratio, query_count, skew, reference))
                    if workload is None:
                        colour, label, text_size = "#f2f2f2", "No data", 10
                    else:
                        colour, winner = OUTCOMES[workload.kill_gate_outcome]
                        text_size = 10
                        if workload.kill_gate_outcome == "indexed_win":
                            factor = min(workload.p50_speedup, workload.p95_speedup)
                            label = f"{winner}\n{factor_text(factor)}"
                        elif workload.kill_gate_outcome == "exhaustive_win":
                            factor = min(
                                1.0 / workload.p50_speedup,
                                1.0 / workload.p95_speedup,
                            )
                            label = f"{winner}\n{factor_text(factor)}"
                        else:
                            label = "\n".join(
                                (
                                    "Mixed",
                                    percentile_result("p50", workload.p50_speedup),
                                    percentile_result("p95", workload.p95_speedup),
                                )
                            )
                            text_size = 8
                    ax.add_patch(
                        Rectangle(
                            (query_column - 0.5, reference_row - 0.5),
                            1,
                            1,
                            facecolor=colour,
                            edgecolor="white",
                            linewidth=2,
                        )
                    )
                    ax.text(
                        query_column,
                        reference_row,
                        label,
                        color="#202020",
                        fontsize=text_size,
                        fontweight="bold",
                        ha="center",
                        va="center",
                    )

            ax.set_xlim(-0.5, len(query_counts) - 0.5)
            ax.set_ylim(len(references) - 0.5, -0.5)
            ax.set_xticks(range(len(query_counts)))
            ax.set_xticklabels(
                [pu.paper_text(f"{int(count):,}") for count in query_counts],
                fontsize=pu.TICK_LABEL_FONT_SIZE,
            )
            ax.set_yticks(range(len(references)))
            ax.set_yticklabels(
                [pu.paper_text(f"{int(count):,}") for count in references],
                fontsize=pu.TICK_LABEL_FONT_SIZE,
            )
            ax.tick_params(length=0, labelleft=panel_column == 0)
            for spine in ax.spines.values():
                spine.set_visible(False)
            if panel_row == 0:
                ax.set_title(
                    pu.paper_text(f"{fill_ratio:.0%} fill", bold=True),
                    fontsize=pu.TITLE_FONT_SIZE,
                )
            if panel_column == len(fill_ratios) - 1:
                ax.set_ylabel(
                    pu.paper_text(f"{str(skew).capitalize()} skew", bold=True),
                    fontsize=pu.AXIS_LABEL_FONT_SIZE,
                    rotation=-90,
                    labelpad=28,
                )
                ax.yaxis.set_label_position("right")

    fig.suptitle(
        pu.paper_text("Indexed versus exhaustive search", bold=True),
        fontsize=pu.TITLE_FONT_SIZE,
        y=0.98,
    )
    fig.text(
        0.5,
        0.91,
        pu.paper_text(
            "Winner must beat both p50 and p95; factor is the smaller speedup"
        ),
        fontsize=pu.DEFAULT_FONT_SIZE,
        ha="center",
    )
    fig.supxlabel(
        pu.paper_text("Queries per search call", bold=True),
        fontsize=pu.AXIS_LABEL_FONT_SIZE,
        y=0.08,
    )
    fig.supylabel(
        pu.paper_text("References", bold=True),
        fontsize=pu.AXIS_LABEL_FONT_SIZE,
        x=0.02,
    )
    fig.subplots_adjust(
        left=0.16,
        right=0.86,
        bottom=0.16,
        top=0.80,
        hspace=0.10,
        wspace=0.08,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    pu.save_figure(fig, output)


if __name__ == "__main__":
    typer.run(main)
