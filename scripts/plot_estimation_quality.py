#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = ["jsonschema", "matplotlib", "pandas", "typer"]
# ///
"""Plot standalone pairwise estimation quality across implementations."""

from pathlib import Path
from typing import Annotated

import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
import plot_utils as pu
import typer
from benchmark_schema import flatten_measurements, load_result

METRICS = (
    ("cardinality_absolute_relative_error", "Cardinality"),
    ("containment_absolute_error", "Containment"),
    ("completeness_absolute_error", "Completeness"),
    ("wkid_absolute_error", "WKID"),
    ("ani_absolute_error", "ANI"),
)
SYMMETRIC_METRICS = {"wkid_absolute_error", "ani_absolute_error"}
ESTIMATORS = (
    ("cuddl", "min-MLE", "cardinality_absolute_relative_error"),
    ("cuddl-bbtools", "BBTools", "cardinality_bbtools_absolute_relative_error"),
    ("cuddl-paper", "paper", "cardinality_paper_absolute_relative_error"),
)
CUDDL = pu.FILTER_STYLES["cuddl"]
BBTOOLS = pu.FILTER_STYLES["cuco_hll"]
ANI_ONLY = {"skani", "hypergen"}


def require(data: pd.DataFrame, columns: set[str], name: str) -> None:
    missing = sorted(columns - set(data.columns))
    if missing:
        raise typer.BadParameter(f"{name} is missing columns: {', '.join(missing)}")


def stats(frame: pd.DataFrame, column: str) -> tuple[float, float, float] | None:
    if column not in frame or not frame[column].notna().any():
        return None
    errors = frame[column].dropna() * 100
    return (
        float(errors.quantile(0.05)),
        float(errors.quantile(0.5)),
        float(errors.quantile(0.95)),
    )


def main(
    quality_json: Annotated[
        Path,
        typer.Argument(
            exists=True, dir_okay=False, help="Pairwise accuracy benchmark result JSON"
        ),
    ],
    output_dir: Annotated[
        Path, typer.Option(file_okay=False, help="Figure output directory")
    ] = Path("results/pairwise-accuracy"),
    ani_min_exact: Annotated[
        float,
        typer.Option(
            min=0.0,
            max=1.0,
            help="Minimum exact_ani kept in the ANI panel only; other panels unfiltered.",
        ),
    ] = 0.0,
 ) -> None:
    """Render estimation quality as a 2x3 slide-sized panel figure."""
    output_dir.mkdir(parents=True, exist_ok=True)
    try:
        quality_result = load_result(quality_json, "pairwise_accuracy")
    except ValueError as error:
        raise typer.BadParameter(f"{quality_json}: {error}") from error
    quality = pd.DataFrame(flatten_measurements(quality_result))
    require(
        quality,
        {"implementation", *(column for column, _ in METRICS)},
        "quality JSON",
    )

    missing_implementations = {"cuddl", "bbtools"} - set(quality["implementation"])
    if missing_implementations:
        raise typer.BadParameter(
            "quality JSON is missing implementations: "
            + ", ".join(sorted(missing_implementations))
        )
    implementations = [
        ("cuddl", "cuDDL min-MLE", CUDDL),
        ("cuddl-bbtools", "cuDDL BBTools", pu.FILTER_STYLES["cuddl_bbtools"]),
        ("cuddl-paper", "cuDDL paper", pu.FILTER_STYLES["cuddl_paper"]),
        ("bbtools", "BBTools DDL", BBTOOLS),
    ]
    if "rabbitsketch" in set(quality["implementation"]):
        implementations.append(
            ("rabbitsketch", "RabbitSketch FastKMV", {"color": "#009E73"})
        )
    if "cuco_hll" in set(quality["implementation"]):
        implementations.append(("cuco_hll", "cuco HLL", {"color": "#E69F00"}))
    if "dashing2" in set(quality["implementation"]):
        implementations.append(
            ("dashing2", "Dashing2 FullSetSketch", {"color": "#CC79A7"})
        )
    if "skani" in set(quality["implementation"]):
        implementations.append(("skani", "skani", {"color": "#56B4E9"}))
    if "hypergen" in set(quality["implementation"]):
        implementations.append(("hypergen", "HyperGen", {"color": "#D55E00"}))
    if "cub-exact" in set(quality["implementation"]):
        implementations.append(("cub-exact", "cub-exact", {"color": "#999999"}))
    lanes = [name for name, _, _ in implementations]
    labels = [label for _, label, _ in implementations]
    styles = {name: style for name, _, style in implementations}
    def frame_for(implementation: str, column: str) -> pd.DataFrame:
        # Estimator variants live on cuDDL rows under their own columns.
        frame = quality[quality["implementation"] == (
            "cuddl" if implementation in ("cuddl-bbtools", "cuddl-paper")
            else implementation)]
        if column == "ani_absolute_error" and ani_min_exact > 0.0 and "exact_ani" in frame:
            frame = frame[frame["exact_ani"] >= ani_min_exact]
        if column in SYMMETRIC_METRICS and "orientation" in frame:
            forwarded = frame[frame["orientation"] == "query_to_reference"]
            if len(forwarded):
                return forwarded
        return frame

    def draw(ax: mpl.axes.Axes, names: list[str], tags: dict[str, str],
             column: str, title: str) -> None:
        maximum = 0.0
        rows: list[tuple[float, float, float] | None] = []
        for name in names:
            source = column
            if column == "cardinality_absolute_relative_error" and name in (
                    "cuddl-bbtools", "cuddl-paper"):
                variant = "bbtools" if name == "cuddl-bbtools" else "paper"
                source = f"cardinality_{variant}_absolute_relative_error"
            row = stats(frame_for(name, source), source)
            rows.append(row)
            if row is not None:
                maximum = max(maximum, row[2])
        limit = maximum * 1.3 if maximum else 1
        for i, (name, row) in enumerate(zip(names, rows)):
            if row is None:
                ax.text(limit * 0.03, i,
                        "ANI only" if name in ANI_ONLY else "No data",
                        va="center", fontsize=9)
                continue
            low, median, high = row
            ax.barh(i, median, height=0.6,
                    color=styles[name]["color"], edgecolor="black",
                    xerr=[[median - low], [high - median]], capsize=3)
            ax.text(high + limit * 0.025, i, f"{median:.2f}",
                    va="center", fontsize=9)
        ax.set_xlim(0, limit)
        display = {name: label for name, label, _ in implementations}
        display.update({"cuddl-bbtools": "cuDDL BBTools",
                        "cuddl-paper": "cuDDL paper"})
        pad = 0.5 if len(names) > 1 else 1.5
        ax.set_ylim(len(names) - 1 + pad, 0 - pad)
        names_shown = [tags.get(name, display.get(name, name)) for name in names]
        ax.set_yticks(range(len(names)), names_shown, fontsize=9)
        ax.set_title(pu.paper_text(title, bold=True), fontsize=12, pad=8)
        ax.set_xlabel(pu.paper_text("Median error (%)"), fontsize=10)
        ax.tick_params(axis="x", labelsize=9)
        ax.xaxis.set_major_locator(mpl.ticker.MaxNLocator(nbins=4))
        ax.xaxis.grid(True, alpha=pu.GRID_ALPHA)
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)
    fig = plt.figure(figsize=(13.33, 7.5), layout="constrained")
    gs = fig.add_gridspec(2, 6)
    axes = [fig.add_subplot(gs[0, 0:2]),
            fig.add_subplot(gs[0, 2:4]),
            fig.add_subplot(gs[0, 4:6]),
            fig.add_subplot(gs[1, 1:3]),
            fig.add_subplot(gs[1, 3:5])]
    draw(axes[0], lanes, {},
         "cardinality_absolute_relative_error", "Cardinality")
    for (column, title), ax in zip(METRICS[1:], axes[1:]):
        draw(ax, lanes, {}, column, title)
    stem = output_dir / "estimation_quality"
    fig.savefig(stem.with_suffix(".png"), dpi=180)
    pu.save_figure(fig, stem.with_suffix(".pdf"))

if __name__ == "__main__":
    typer.run(main)
