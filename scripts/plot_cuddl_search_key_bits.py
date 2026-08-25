#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = ["matplotlib", "pandas", "typer"]
# ///
"""Plot 15-bit versus 16-bit index keys from the search decision CSV.

RefSeq register scores occupy the 15-bit space (the top bit is empirically
dead, paper Section 8.2), so masking to 15 bits halves the index offset table
without changing the posting lists: same visits, same candidate quality, half
the resident bytes. One panel per relevant metric on the 200,687-sketch
RefSeq database.
"""

from pathlib import Path
from typing import Annotated

import matplotlib.pyplot as plt
import pandas as pd
import plot_utils as pu
import typer

PAPER_REFERENCE_COUNT = 200687

METRICS = (
    ("atomic_updates", "Index entries visited", "log"),
    ("indexed_median_ms", "Indexed median time (ms)", "linear"),
)

KEY_BIT_STYLES = {
    15: {"color": "#2c7fb8", "label": "15-bit key", "marker": "o"},
    16: {"color": "#d95f0e", "label": "16-bit key", "marker": "s"},
}


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
    ] = Path("results/cuddl-search-key-bits.pdf"),
) -> None:
    """Render posting work and time for both key widths."""
    data = pu.load_csv(csv_path)
    required = {
        "status",
        "reference_count",
        "query_count",
        "skew",
        "index_mode",
        *(name for name, _, _ in METRICS),
    }
    missing = sorted(required - set(data.columns))
    if missing:
        raise typer.BadParameter(f"CSV is missing columns: {', '.join(missing)}")

    data = data.loc[data["status"] == "ok"].copy()
    data = data.loc[data["reference_count"] == PAPER_REFERENCE_COUNT].copy()
    data = data.loc[data["query_count"] == 128].copy()
    if data.empty:
        raise typer.BadParameter("CSV has no paper-scale, 128-query workloads")

    data["hot_fraction"] = data["skew"].map(skew_fraction)
    data["key_bits"] = data["index_mode"].str.extract(r"b(\d+)_k(\d+)")[1].astype(int)
    for name, _, _ in METRICS:
        data[name] = pd.to_numeric(data[name])
    hot_fractions = sorted(data["hot_fraction"].unique())
    if not hot_fractions:
        raise typer.BadParameter("CSV has no hot-fraction sweep")

    fig, axes = plt.subplots(
        1,
        len(METRICS),
        figsize=(8.4, 3.7),
        squeeze=False,
    )
    for metric_column, (column, label, scale) in enumerate(METRICS):
        ax = axes[0][metric_column]
        for bits, style in KEY_BIT_STYLES.items():
            series = (
                data[data["key_bits"] == bits].groupby("hot_fraction")[column].median()
            )
            # The two widths coincide within noise on several metrics; offset the
            # points slightly so the second series does not paint over the first.
            offset = (bits - 15.5) * 0.024
            ax.plot(
                series.index + offset,
                series,
                marker=style["marker"],
                color=style["color"],
                label=style["label"],
            )
        if scale == "log":
            ax.set_yscale("log")
        ax.set_title(
            pu.paper_text(label, bold=True),
            fontsize=pu.TITLE_FONT_SIZE,
        )
        ax.set_xlabel(
            pu.paper_text("Hot-bucket fraction"),
            fontsize=pu.AXIS_LABEL_FONT_SIZE,
        )
        ax.tick_params(labelsize=pu.TICK_LABEL_FONT_SIZE)

    # The median-time curves rise toward the top-right of the right panel, so
    # its bottom-right corner hosts the legend without covering any data.
    axes[0][1].legend(
        handles=[
            plt.Line2D(
                [],
                [],
                color=style["color"],
                marker=style["marker"],
                label=style["label"],
            )
            for style in KEY_BIT_STYLES.values()
        ],
        loc="lower right",
        fontsize=pu.TICK_LABEL_FONT_SIZE,
        frameon=False,
        handlelength=2.5,
    )
    fig.suptitle(
        pu.paper_text(
            f"15-bit versus 16-bit index keys on {PAPER_REFERENCE_COUNT:,} RefSeq sketches",
            bold=True,
        ),
        fontsize=pu.TITLE_FONT_SIZE + 2,
        y=0.985,
    )
    fig.subplots_adjust(
        left=0.07,
        right=0.97,
        bottom=0.16,
        top=0.78,
        wspace=0.30,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    pu.save_figure(fig, output, pad_inches=0.10)


if __name__ == "__main__":
    typer.run(main)
