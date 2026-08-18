#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = ["matplotlib", "pandas", "typer"]
# ///
"""Plot cuDDL versus original BBTools DynamicDemiLog results."""

from pathlib import Path
from typing import Annotated

import plot_utils as pu
import typer

IMPLEMENTATIONS = (
    ("cuddl", "cuDDL", pu.FILTER_STYLES["cuddl"]),
    ("bbtools", "BBTools DynamicDemiLog", pu.FILTER_STYLES["cuco_hll"]),
)


def main(
    csv_path: Annotated[Path, typer.Option(help="Comparison benchmark CSV")],
    output_dir: Annotated[Path, typer.Option(help="Figure output directory")] = Path(
        "results/bbtools"
    ),
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    data = pu.load_csv(csv_path)
    data["absolute_error"] = (
        (data["estimate"] - data["count"]) / data["count"]
    ).abs() * 100.0

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

    accuracy = data.groupby(["implementation", "count"])["absolute_error"].mean()
    fig, ax = pu.setup_figure(figsize=(12, 8))
    for name, label, style in IMPLEMENTATIONS:
        series = accuracy.loc[name]
        ax.semilogx(
            series.index,
            series,
            color=style["color"],
            marker=style["marker"],
            label=label,
        )
    pu.format_axis(
        ax,
        xlabel="Exact distinct count",
        ylabel="Mean absolute estimation error (%)",
    )
    pu.create_legend(ax)
    pu.save_figure(fig, output_dir / "accuracy.pdf")

    estimates = data.groupby(["implementation", "count"])["estimate"].mean()
    fig, ax = pu.setup_figure(figsize=(12, 8))
    exact = estimates.loc["cuddl"].index
    ax.loglog(exact, exact, color="#333333", linestyle="--", label="Exact")
    for name, label, style in IMPLEMENTATIONS:
        series = estimates.loc[name]
        ax.loglog(
            series.index,
            series,
            color=style["color"],
            marker=style["marker"],
            label=label,
        )
    pu.format_axis(ax, xlabel="Exact distinct count", ylabel="Estimated distinct count")
    pu.create_legend(ax)
    pu.save_figure(fig, output_dir / "cardinality_estimates.pdf")

    for name in ("throughput", "speedup", "accuracy", "cardinality_estimates"):
        typer.echo(f"Wrote {output_dir / f'{name}.pdf'}")


if __name__ == "__main__":
    typer.run(main)
