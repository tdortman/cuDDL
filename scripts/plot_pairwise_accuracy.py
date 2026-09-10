#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = ["jsonschema", "matplotlib", "pandas", "typer"]
# ///
"""Plot cuDDL raw-DNA pairwise estimation error as heatmaps by size ratio."""

from pathlib import Path
from typing import Annotated

import matplotlib.pyplot as plt
import pandas as pd
import plot_utils as pu
import typer
from benchmark_schema import flatten_measurements, load_result

METRICS = (
    ("containment_absolute_error", "Containment"),
    ("completeness_absolute_error", "Completeness"),
    ("wkid_absolute_error", "WKID"),
    ("ani_absolute_error", "ANI"),
)


def plot_comparison(data: pd.DataFrame, output_dir: Path) -> None:
    """Keep each implementation separate with common scales for each metric."""
    labels = {
        "cuddl": "cuDDL",
        "bbtools": "BBTools DDL",
        "rabbitsketch": "RabbitSketch FastKMV",
        "cuco_hll": "cuco HLL",
    }
    unknown = set(data["implementation"]) - labels.keys()
    if unknown:
        raise typer.BadParameter(f"Unknown accuracy implementations: {sorted(unknown)}")
    implementations = [name for name in labels if name in set(data["implementation"])]
    metrics = (
        (("cardinality_absolute_relative_error", "Cardinality"),)
        if "cardinality_absolute_relative_error" in data
        else ()
    ) + METRICS
    powers = sorted(data["power"].unique())
    anis = sorted(data["requested_ani"].unique())
    medians = data.groupby(["implementation", "size_ratio", "requested_ani", "power"])[
        [column for column, _ in metrics]
    ].median()
    maxima = medians.max() * 100
    for ratio in sorted(data["size_ratio"].unique()):
        fig, axes = plt.subplots(
            len(implementations),
            len(metrics),
            squeeze=False,
            figsize=(3.8 * len(metrics), 3.0 * len(implementations) + 1.2),
        )
        for row, name in enumerate(implementations):
            for column_index, (column, title) in enumerate(metrics):
                ax = axes[row, column_index]
                values = (
                    medians.loc[(name, ratio), column]
                    .unstack("power")
                    .reindex(index=anis, columns=powers)
                )
                if values.isna().any().any():
                    raise typer.BadParameter(
                        f"Incomplete accuracy grid for {name}, ratio {ratio}"
                    )
                image = ax.imshow(
                    values * 100,
                    aspect="auto",
                    cmap="viridis",
                    origin="lower",
                    interpolation="nearest",
                    vmin=0,
                    vmax=max(float(maxima[column]), 1e-12),
                )
                if row == 0:
                    ax.set_title(pu.paper_text(title, bold=True))
                ax.set_xticks(
                    range(len(powers)), [rf"$2^{{{power}}}$" for power in powers]
                )
                ax.set_yticks(
                    range(len(anis)), [pu.paper_text(f"{ani * 100:g}%") for ani in anis]
                )
                if column_index == 0:
                    ax.set_ylabel(pu.paper_text(labels[name] + "\nTarget ANI"))
                if row + 1 == len(implementations):
                    ax.set_xlabel(pu.paper_text("Query bases"))
                fig.colorbar(image, ax=ax, pad=0.02)
        fig.suptitle(
            pu.paper_text(
                f"Median estimation error, reference/query {ratio}:1", bold=True
            )
        )
        fig.text(
            0.5,
            0.025,
            pu.paper_text(
                "Cardinality: absolute relative error (%). Other metrics: absolute error (percentage points).\n"
                "RabbitSketch ANI is Jaccard-derived; cuDDL and BBTools ANI are WKID-derived. Medians are over trials."
            ),
            ha="center",
            fontsize=10,
        )
        fig.tight_layout(rect=(0, 0.09, 1, 0.96))
        stem = output_dir / f"comparison-ratio-{ratio}"
        fig.savefig(stem.with_suffix(".png"), dpi=180, bbox_inches="tight")
        fig.savefig(stem.with_suffix(".svg"), bbox_inches="tight")
        pu.save_figure(fig, stem.with_suffix(".pdf"))


def main(
    result_path: Annotated[
        Path, typer.Argument(exists=True, dir_okay=False, help="Pairwise accuracy JSON")
    ],
    output_dir: Annotated[
        Path, typer.Option(file_okay=False, help="Figure output directory")
    ] = Path("results/pairwise-accuracy"),
    compare: Annotated[
        bool,
        typer.Option(
            help="Compare all implementations on shared scales, including cardinality"
        ),
    ] = False,
) -> None:
    """Render median absolute error over the raw-DNA parameter sweep."""
    try:
        result = load_result(result_path, "pairwise_accuracy")
    except ValueError as error:
        raise typer.BadParameter(f"{result_path}: {error}") from error
    data = pd.DataFrame(flatten_measurements(result))
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
        raise typer.BadParameter(f"JSON is missing fields: {', '.join(missing)}")
    if data.empty:
        raise typer.BadParameter("JSON has no pairwise accuracy measurements")
    if compare:
        selected = data[data["orientation"] == "query_to_reference"]
        if selected.empty:
            raise typer.BadParameter("JSON has no query-to-reference measurements")
        output_dir.mkdir(parents=True, exist_ok=True)
        plot_comparison(selected, output_dir)
        return
    data = data[
        (data["implementation"] == "cuddl")
        & (data["orientation"] == "query_to_reference")
    ]
    if data.empty:
        raise typer.BadParameter("JSON has no cuDDL query-to-reference measurements")

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
        fig, _ = pu.setup_figure(nrows=2, ncols=2)
        for ax, (column, title) in zip(fig.axes, METRICS, strict=True):
            values = (
                data[data["size_ratio"] == ratio]
                .groupby(["requested_ani", "power"])[column]
                .median()
                .unstack("power")
                .reindex(index=ani_levels, columns=powers)
            )
            if values.isna().any().any():
                raise typer.BadParameter(
                    "JSON does not contain every target-ANI/query-length/"
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
