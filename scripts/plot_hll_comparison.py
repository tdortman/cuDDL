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
- `cardinality`: cardinality-estimation time against item count.
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

    # --- Throughput figure ---------------------------------------------------
    fig, ax = pu.setup_figure(figsize=(12, 8))
    for estimator, label, color, marker in [
        ("cuddl_construction", "cuDDL", "#2E86AB", "o"),
        ("cuco_hll_construction", "cuco HLL", "#A23B72", "^"),
    ]:
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
    for estimator, label, color, marker in [
        ("cuddl_cardinality", "cuDDL", "#2E86AB", "o"),
        ("cuco_hll_cardinality", "cuco HLL", "#A23B72", "^"),
    ]:
        data = cardinality[cardinality["Estimator"] == estimator]
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
    fig, ax = pu.setup_figure(figsize=(12, 8))
    exact = cardinality.groupby("Items", as_index=False)["Exact"].first()
    ax.loglog(  # ty: ignore[unresolved-attribute]
        exact["Items"],
        exact["Exact"],
        color="#333333",
        linestyle="--",
        label="Exact",
    )
    for estimator, label, color, marker in [
        ("cuddl_cardinality", "cuDDL", "#2E86AB", "o"),
        ("cuco_hll_cardinality", "cuco HLL", "#A23B72", "^"),
    ]:
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
    for estimator, label, color, marker in [
        ("cuddl_similarity", "cuDDL", "#2E86AB", "o"),
        ("cuco_hll_similarity", "cuco HLL", "#A23B72", "^"),
    ]:
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
