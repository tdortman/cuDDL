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

Two figures:

- `throughput`: construction items/s against item count.
- `cardinality`: cardinality-estimation time against item count.

Both come from the single benchmark `--csv` output. Styling comes from
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
    """Render the construction throughput and cardinality-time figures."""
    output_dir.mkdir(parents=True, exist_ok=True)

    bench = pu.load_csv(nvbench_csv)
    bench = bench.rename(columns={"Iters": "Items", "Benchmark": "Estimator"})
    construction = bench[bench["Estimator"].str.endswith("_construction")].copy()
    construction["Gigaelem/s"] = construction["Elem/s (elem/sec)"] / pu.THROUGHPUT_SCALE
    cardinality = bench[bench["Estimator"].str.endswith("_cardinality")]

    # --- Throughput figure ---------------------------------------------------
    fig, ax = pu.setup_figure(figsize=(12, 8))
    for estimator, color, marker in [
        ("cuddl_construction", "#2E86AB", "o"),
        ("cuco_hll_construction", "#A23B72", "^"),
    ]:
        data = construction[construction["Estimator"] == estimator]
        ax.semilogx(  # ty: ignore[unresolved-attribute]
            data["Items"],
            data["Gigaelem/s"],
            color=color,
            marker=marker,
            label=pu.paper_text(estimator.split("_")[0]),
        )
    pu.format_axis(ax, xlabel="Items", ylabel="Giga elements / s")
    pu.create_legend(ax)
    pu.save_figure(fig, output_dir / "throughput.pdf", message="Wrote throughput plot")
    typer.echo(f"Wrote {output_dir / 'throughput.pdf'}")

    # --- Cardinality-estimation figure --------------------------------------
    fig, ax = pu.setup_figure(figsize=(12, 8))
    for estimator, color, marker in [
        ("cuddl_cardinality", "#2E86AB", "o"),
        ("cuco_hll_cardinality", "#A23B72", "^"),
    ]:
        data = cardinality[cardinality["Estimator"] == estimator]
        ax.semilogx(  # ty: ignore[unresolved-attribute]
            data["Items"],
            data["GPU Time (sec)"] * 1e6,
            color=color,
            marker=marker,
            label=pu.paper_text(estimator.split("_")[0]),
        )
    pu.format_axis(ax, xlabel="Items", ylabel="Cardinality estimation time (µs)")
    pu.create_legend(ax)
    pu.save_figure(
        fig, output_dir / "cardinality.pdf", message="Wrote cardinality plot"
    )
    typer.echo(f"Wrote {output_dir / 'cardinality.pdf'}")


if __name__ == "__main__":
    app()
