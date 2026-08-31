#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = ["matplotlib", "pandas", "typer"]
# ///
"""Plot cuDDL raw-DNA pairwise estimation error as heatmaps by size ratio."""

from pathlib import Path
from typing import Annotated

import plot_utils as pu
import typer

METRICS = (
    ("containment_absolute_error", "Containment"),
    ("completeness_absolute_error", "Completeness"),
    ("wkid_absolute_error", "WKID"),
    ("ani_absolute_error", "ANI"),
)


def main(
    csv_path: Annotated[
        Path, typer.Argument(exists=True, dir_okay=False, help="Pairwise accuracy CSV")
    ],
    output_dir: Annotated[
        Path, typer.Option(file_okay=False, help="Figure output directory")
    ] = Path("results/pairwise-accuracy"),
) -> None:
    """Render median absolute error over the raw-DNA parameter sweep."""
    data = pu.load_csv(csv_path)
    required = {
        "implementation",
        "orientation",
        "power",
        "requested_ani",
        "size_ratio",
        *(column for column, _ in METRICS),
    }
    missing = sorted(required - set(data.columns))
    if missing:
        raise typer.BadParameter(f"CSV is missing columns: {', '.join(missing)}")
    if data.empty:
        raise typer.BadParameter("CSV has no pairwise accuracy rows")
    data = data[
        (data["implementation"] == "cuddl")
        & (data["orientation"] == "query_to_reference")
    ]
    if data.empty:
        raise typer.BadParameter("CSV has no cuDDL query-to-reference rows")

    powers = sorted(int(value) for value in data["power"].unique())
    ani_levels = sorted(float(value) for value in data["requested_ani"].unique())
    size_ratios = sorted(int(value) for value in data["size_ratio"].unique())

    metric_maxima = {
        column: data.groupby(["size_ratio", "power", "requested_ani"])[column]
        .median()
        .max()
        * 100
        for column, _ in METRICS
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    for ratio in size_ratios:
        fig, axes = pu.setup_figure(nrows=2, ncols=2)
        for ax, (column, title) in zip(axes.flat, METRICS, strict=True):
            values = (
                data[data["size_ratio"] == ratio]
                .groupby(["requested_ani", "power"])[column]
                .median()
                .unstack("power")
                .reindex(index=ani_levels, columns=powers)
            )
            if values.isna().any().any():
                raise typer.BadParameter(
                    "CSV does not contain every target-ANI/query-length/"
                    "size-ratio combination"
                )
            image = ax.imshow(
                values * 100,
                aspect="auto",
                cmap="viridis",
                interpolation="nearest",
                origin="lower",
                vmin=0,
                vmax=metric_maxima[column],
            )
            ax.set_title(pu.paper_text(title, bold=True), fontsize=pu.TITLE_FONT_SIZE)
            ax.set_xticks(range(len(powers)), [rf"$2^{{{power}}}$" for power in powers])
            ax.set_yticks(
                range(len(ani_levels)),
                [pu.paper_text(f"{ani * 100:g}%") for ani in ani_levels],
            )
            ax.tick_params(axis="both", labelsize=pu.TICK_LABEL_FONT_SIZE)
            colorbar = fig.colorbar(image, ax=ax, pad=0.02)
            colorbar.ax.tick_params(labelsize=pu.TICK_LABEL_FONT_SIZE)

        fig.suptitle(
            pu.paper_text(
                f"Median absolute estimation error (%), reference/query {ratio}:1",
                bold=True,
            ),
            fontsize=pu.TITLE_FONT_SIZE,
        )
        fig.supxlabel(
            pu.paper_text("Query sequence length", bold=True),
            fontsize=pu.AXIS_LABEL_FONT_SIZE,
        )
        fig.supylabel(
            pu.paper_text("Target ANI", bold=True), fontsize=pu.AXIS_LABEL_FONT_SIZE
        )
        fig.tight_layout(rect=(0.04, 0.05, 1, 0.92), h_pad=3.2, w_pad=2.0)
        pu.save_figure(fig, output_dir / f"ratio-{ratio}.pdf")


if __name__ == "__main__":
    typer.run(main)
