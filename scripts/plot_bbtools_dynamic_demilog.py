#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = ["jsonschema", "matplotlib", "pandas", "typer"]
# ///
"""Plot cuDDL versus original BBTools DynamicDemiLog results."""

from pathlib import Path
from typing import Annotated

import pandas as pd
import plot_utils as pu
import typer
from benchmark_schema import flatten_measurements, load_result

IMPLEMENTATIONS = (
    ("cuddl", "cuDDL", pu.FILTER_STYLES["cuddl"]),
    ("bbtools", "BBTools DynamicDemiLog", pu.FILTER_STYLES["cuco_hll"]),
)
ESTIMATORS = (
    (
        "cuddl",
        "estimate",
        "cuDDL unbiased",
        {**pu.FILTER_STYLES["cuddl"], "linestyle": "--"},
    ),
    (
        "cuddl",
        "bounded_estimate",
        "cuDDL corrected",
        pu.FILTER_STYLES["cuddl_bbtools"],
    ),
    (
        "bbtools",
        "estimate",
        "BBTools DynamicDemiLog",
        pu.FILTER_STYLES["cuco_hll"],
    ),
)


def main(
    json_path: Annotated[Path, typer.Option(help="Comparison benchmark JSON")],
    output_dir: Annotated[Path, typer.Option(help="Figure output directory")] = Path(
        "results/bbtools"
    ),
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    try:
        result = load_result(json_path, "dynamic_demilog")
    except ValueError as error:
        raise typer.BadParameter(f"{json_path}: {error}") from error
    data = pd.DataFrame(flatten_measurements(result))

    throughput = data.groupby(["implementation", "count"])["adds_per_second"].median()
    fig, ax = pu.setup_figure(figsize=(12, 8))
    for name, label, style in IMPLEMENTATIONS:
        series = throughput.loc[name] / pu.THROUGHPUT_SCALE
        ax.loglog(
            series.index,
            series,
            color=style["color"],
            marker=style["marker"],
            label=label,
        )
    pu.format_axis(ax, xlabel="Distinct inputs", ylabel="Median billion adds / s")
    pu.create_legend(ax)
    pu.save_figure(fig, output_dir / "throughput.pdf")

    speedup = throughput.loc["cuddl"] / throughput.loc["bbtools"]
    fig, ax = pu.setup_figure(figsize=(12, 8))
    ax.semilogx(
        speedup.index,
        speedup,
        color=pu.FILTER_STYLES["cuddl"]["color"],
        marker=pu.FILTER_STYLES["cuddl"]["marker"],
    )
    ax.axhline(1.0, color="#333333", linestyle="--")
    pu.format_axis(ax, xlabel="Distinct inputs", ylabel="cuDDL speedup over BBTools")
    pu.save_figure(fig, output_dir / "speedup.pdf")

    fig, ax = pu.setup_figure(figsize=(12, 8))
    for implementation, column, label, style in ESTIMATORS:
        selected = data[data["implementation"] == implementation]
        series = (
            (((selected[column] - selected["count"]) / selected["count"]).abs() * 100.0)
            .groupby(selected["count"])
            .mean()
        )
        ax.semilogx(
            series.index,
            series,
            color=style["color"],
            marker=style["marker"],
            linestyle=style.get("linestyle", "-"),
            label=label,
        )
    pu.format_axis(
        ax,
        xlabel="Exact distinct count",
        ylabel="Mean absolute estimation error (%)",
    )
    pu.create_legend(ax)
    pu.save_figure(fig, output_dir / "accuracy.pdf")

    fig, ax = pu.setup_figure(figsize=(12, 8))
    exact = data[data["implementation"] == "cuddl"]["count"].drop_duplicates()
    ax.loglog(exact, exact, color="#333333", linestyle="--", label="Exact")
    for implementation, column, label, style in ESTIMATORS:
        selected = data[data["implementation"] == implementation]
        series = selected.groupby("count")[column].mean()
        ax.loglog(
            series.index,
            series,
            color=style["color"],
            marker=style["marker"],
            linestyle=style.get("linestyle", "-"),
            label=label,
        )
    pu.format_axis(ax, xlabel="Exact distinct count", ylabel="Estimated distinct count")
    pu.create_legend(ax)
    pu.save_figure(fig, output_dir / "cardinality_estimates.pdf")

    for name in ("throughput", "speedup", "accuracy", "cardinality_estimates"):
        typer.echo(f"Wrote {output_dir / f'{name}.pdf'}")


if __name__ == "__main__":
    typer.run(main)
