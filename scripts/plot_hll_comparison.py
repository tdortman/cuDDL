#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "jsonschema",
#   "matplotlib",
#   "pandas",
#   "numpy",
#   "typer",
# ]
# ///
"""Plot cuDDL vs cuCollections HyperLogLog benchmarks.

The four figures are loaded from one validated benchmark result JSON:

- throughput: construction items/s against item count.
- accuracy: relative cardinality error against item count.
- cardinality_estimates: exact, cuDDL, and cuCollections estimated counts.
- similarity: similarity-estimation time against item count.

Styling comes from ./scripts/plot_utils.py.
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import pandas as pd
import plot_utils as pu
import typer
from benchmark_schema import flatten_measurements, load_result

app = typer.Typer(help="Plot cuDDL vs cuco HLL benchmarks", no_args_is_help=True)


@app.command()
def plot(
    result_json: Annotated[
        Path,
        typer.Argument(exists=True, dir_okay=False, help="HLL comparison result JSON"),
    ],
    output_dir: Annotated[Path, typer.Option(help="Where to write the figures")] = Path(
        "results/hll"
    ),
) -> None:
    """Render throughput, cardinality, estimate-accuracy, and similarity figures."""
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        result = load_result(result_json, operation="hll_comparison")
    except ValueError as error:
        raise typer.BadParameter(f"{result_json}: {error}") from error
    bench = pd.DataFrame(flatten_measurements(result))
    if bench.empty:
        raise typer.BadParameter(f"{result_json}: result has no measurements")
    required = {
        "implementation",
        "phase",
        "items",
        "gpu_min_ms",
        "gpu_median_ms",
        "exact",
        "estimate",
        "absolute_error",
    }
    missing = sorted(required - set(bench.columns))
    if missing:
        raise typer.BadParameter(
            f"{result_json}: result is missing columns: {', '.join(missing)}"
        )

    construction = bench[bench["phase"] == "construction"].copy()
    # Minimum time is immune to the box's periodic clock stretching, unlike the median.
    construction["gigaelem_per_s"] = (
        construction["items"] / (construction["gpu_min_ms"] / 1e3) / pu.THROUGHPUT_SCALE
    )
    cardinality = bench[bench["phase"] == "cardinality"]

    construction_estimators = [
        (
            "cuddl",
            "cuDDL",
            pu.FILTER_STYLES["cuddl"]["color"],
            pu.FILTER_STYLES["cuddl"]["marker"],
        ),
        (
            "cuco_hll",
            "cuco HLL",
            pu.FILTER_STYLES["cuco_hll"]["color"],
            pu.FILTER_STYLES["cuco_hll"]["marker"],
        ),
    ]
    estimators = [
        (
            "cuddl",
            pu.FILTER_DISPLAY_NAMES["cuddl"],
            pu.FILTER_STYLES["cuddl"]["color"],
            pu.FILTER_STYLES["cuddl"]["marker"],
        ),
        (
            "cuco_hll",
            pu.FILTER_DISPLAY_NAMES["cuco_hll"],
            pu.FILTER_STYLES["cuco_hll"]["color"],
            pu.FILTER_STYLES["cuco_hll"]["marker"],
        ),
    ]
    accuracy_estimators = [
        (
            estimator,
            pu.FILTER_DISPLAY_NAMES[estimator],
            pu.FILTER_STYLES[estimator]["color"],
            pu.FILTER_STYLES[estimator]["marker"],
        )
        for estimator in ("cuddl", "cuddl_bbtools", "cuddl_paper", "cuco_hll")
    ]
    timing_estimators = [*accuracy_estimators]

    # --- Throughput figure ---------------------------------------------------
    fig, ax = pu.setup_figure(figsize=(12, 8))
    for estimator, label, color, marker in construction_estimators:
        data = construction[construction["implementation"] == estimator]
        ax.semilogx(
            data["items"],
            data["gigaelem_per_s"],
            color=color,
            marker=marker,
            label=label,
        )
    pu.format_axis(ax, xlabel="Items", ylabel="Giga elements / s")
    pu.create_legend(ax)
    pu.save_figure(fig, output_dir / "throughput.pdf", message="Wrote throughput plot")
    typer.echo(f"Wrote {output_dir / 'throughput.pdf'}")

    # --- Cardinality-estimation figure --------------------------------------
    fig, ax = pu.setup_figure(figsize=(12, 8))
    for estimator, label, color, marker in timing_estimators:
        data = cardinality[cardinality["implementation"] == estimator]
        if data.empty:
            continue
        ax.semilogx(
            data["items"],
            data["gpu_median_ms"] * 1e3,
            color=color,
            marker=marker,
            label=label,
        )
    pu.format_axis(ax, xlabel="Items", ylabel="Cardinality estimation time (µs)")
    pu.create_legend(ax)
    pu.save_figure(
        fig, output_dir / "cardinality.pdf", message="Wrote cardinality plot"
    )
    typer.echo(f"Wrote {output_dir / 'cardinality.pdf'}")

    # --- Cardinality-accuracy figure ----------------------------------------
    accuracy = bench[bench["phase"] == "accuracy"]
    fig, ax = pu.setup_figure(figsize=(12, 8))
    for estimator, label, color, marker in accuracy_estimators:
        data = accuracy[accuracy["implementation"] == estimator].copy()
        data["absolute_error"] *= 100.0
        mean = data.groupby("items")["absolute_error"].mean()
        ax.semilogx(
            mean.index,
            mean,
            color=color,
            marker=marker,
            label=label,
        )
    pu.format_axis(
        ax,
        xlabel="Exact distinct count",
        ylabel="Mean absolute estimation error (%)",
    )
    pu.create_legend(ax)
    pu.save_figure(fig, output_dir / "accuracy.pdf", message="Wrote accuracy plot")
    typer.echo(f"Wrote {output_dir / 'accuracy.pdf'}")

    # --- Cardinality-estimate figure ----------------------------------------
    fig, ax = pu.setup_figure(figsize=(12, 8))
    exact = (
        cardinality.dropna(subset=["exact"])
        .groupby("items", as_index=False)["exact"]
        .first()
    )
    ax.loglog(
        exact["items"],
        exact["exact"],
        color="#333333",
        linestyle="--",
        label="Exact",
    )
    for estimator, label, color, marker in estimators:
        data = cardinality[cardinality["implementation"] == estimator]
        ax.loglog(
            data["items"],
            data["estimate"],
            color=color,
            marker=marker,
            label=label,
        )
    pu.format_axis(ax, xlabel="Exact distinct count", ylabel="Estimated distinct count")
    pu.create_legend(ax)
    pu.save_figure(
        fig,
        output_dir / "cardinality_estimates.pdf",
        message="Wrote cardinality estimate plot",
    )
    typer.echo(f"Wrote {output_dir / 'cardinality_estimates.pdf'}")

    # --- Similarity-estimation figure ---------------------------------------
    similarity = bench[bench["phase"] == "similarity"]
    fig, ax = pu.setup_figure(figsize=(12, 8))
    for estimator, label, color, marker in estimators:
        data = similarity[similarity["implementation"] == estimator]
        ax.semilogx(
            data["items"],
            data["gpu_median_ms"] * 1e3,
            color=color,
            marker=marker,
            label=label,
        )
    pu.format_axis(ax, xlabel="Items per set", ylabel="Similarity estimation time (µs)")
    pu.create_legend(ax)
    pu.save_figure(fig, output_dir / "similarity.pdf", message="Wrote similarity plot")
    typer.echo(f"Wrote {output_dir / 'similarity.pdf'}")


if __name__ == "__main__":
    app()
