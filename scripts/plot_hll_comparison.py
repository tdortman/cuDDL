#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "matplotlib",
#   "pandas",
#   "numpy",
#   "typer",
# ]
# ///
"""Plot cuDDL vs cuCollections HyperLogLog benchmarks.

Three figures:

- `throughput`: construction items/s against item count.
- `accuracy`: relative cardinality error against item count.
- `cardinality_estimates`: exact, cuDDL, and cuCollections estimated counts.
- `similarity`: similarity-estimation time against item count.

All come from the single benchmark `--csv` output. Styling comes from
`./scripts/plot_utils.py`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import plot_utils as pu
import typer

app = typer.Typer(help="Plot cuDDL vs cuco HLL benchmarks", no_args_is_help=True)


@app.command()
def plot(
    nvbench_csv: Annotated[Path, typer.Option(help="Benchmark --csv output")],
    accuracy_csv: Annotated[
        Path | None, typer.Option(help="Independent accuracy-trial CSV")
    ] = None,
    output_dir: Annotated[Path, typer.Option(help="Where to write the figures")] = Path(
        "results/hll"
    ),
):
    """Render throughput, cardinality, estimate-accuracy, and similarity figures."""
    output_dir.mkdir(parents=True, exist_ok=True)

    bench = pu.load_csv(nvbench_csv)
    bench = bench.rename(columns={"Iters": "Items", "Benchmark": "Estimator"})
    construction = bench[bench["Estimator"].str.endswith("_construction")].copy()
    construction["Gigaelem/s"] = construction["Median Throughput"] / pu.THROUGHPUT_SCALE
    cardinality = bench[bench["Estimator"].str.endswith("_cardinality")]

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
    timing_estimators = [
        *accuracy_estimators,
    ]

    # --- Throughput figure ---------------------------------------------------
    fig, ax = pu.setup_figure(figsize=(12, 8))
    for estimator, label, color, marker in construction_estimators:
        estimator = f"{estimator}_construction"
        data = construction[construction["Estimator"] == estimator]
        ax.semilogx(  # ty: ignore[unresolved-attribute]
            data["Items"],
            data["Gigaelem/s"],
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
        data = cardinality[cardinality["Estimator"] == f"{estimator}_cardinality"]
        if data.empty:
            continue
        ax.semilogx(  # ty: ignore[unresolved-attribute]
            data["Items"],
            data["Median GPU Time"] * 1e6,
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
    accuracy = pu.load_csv(accuracy_csv) if accuracy_csv else cardinality.copy()
    if accuracy_csv is None:
        accuracy["Estimator"] = accuracy["Estimator"].str.removesuffix("_cardinality")
        accuracy["Absolute Error"] = (
            (accuracy["Estimate"] - accuracy["Exact"]) / accuracy["Exact"]
        ).abs()

    fig, ax = pu.setup_figure(figsize=(12, 8))
    for estimator, label, color, marker in accuracy_estimators:
        data = accuracy[accuracy["Estimator"] == estimator].copy()
        data["Absolute Error"] *= 100.0
        mean = data.groupby("Items")["Absolute Error"].mean()
        ax.semilogx(  # ty: ignore[unresolved-attribute]
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

    # --- Cardinality-accuracy figure ----------------------------------------
    fig, ax = pu.setup_figure(figsize=(12, 8))
    exact = cardinality.groupby("Items", as_index=False)["Exact"].first()
    ax.loglog(  # ty: ignore[unresolved-attribute]
        exact["Items"],
        exact["Exact"],
        color="#333333",
        linestyle="--",
        label="Exact",
    )
    for estimator, label, color, marker in estimators:
        estimator = f"{estimator}_cardinality"
        data = cardinality[cardinality["Estimator"] == estimator]
        ax.loglog(  # ty: ignore[unresolved-attribute]
            data["Items"],
            data["Estimate"],
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
    similarity = bench[bench["Estimator"].str.endswith("_similarity")]
    fig, ax = pu.setup_figure(figsize=(12, 8))
    for estimator, label, color, marker in estimators:
        estimator = f"{estimator}_similarity"
        data = similarity[similarity["Estimator"] == estimator]
        ax.semilogx(  # ty: ignore[unresolved-attribute]
            data["Items"],
            data["Median GPU Time"] * 1e6,
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
