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
"""Plot cuDDL vs cuCollections HyperLogLog construction benchmarks.

Two figures:

- `throughput`: items/s (from the nvbench CSV) against item count for the two estimators.
- `accuracy`: relative cardinality error (from the nvbench CSV summaries) against item count.

Both come from the single benchmark `--csv` output. Styling comes from
`./scripts/plot_utils.py`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import plot_utils as pu
import typer

app = typer.Typer(
    help="Plot cuDDL vs cuco HLL construction benchmark", no_args_is_help=True
)


@app.command()
def plot(
    nvbench_csv: Annotated[Path, typer.Option(help="Benchmark --csv output")],
    output_dir: Annotated[Path, typer.Option(help="Where to write the figures")] = Path(
        "results/hll"
    ),
):
    """Render the throughput and accuracy figures."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Throughput from the nvbench CSV.
    bench = pu.load_csv(nvbench_csv)
    bench["Gigaelem/s"] = bench["Elem/s (elem/sec)"] / pu.THROUGHPUT_SCALE
    bench = bench.rename(columns={"Iters": "Items", "Benchmark": "Estimator"})

    # Accuracy from the same CSV, in the summary columns attached by the benchmark.
    acc = bench.rename(columns={"Distinct": "distinct", "Rel Error": "rel_error"})
    acc["rel_error"] = acc["rel_error"].astype("float64")

    # --- Throughput figure ---------------------------------------------------
    fig, ax = pu.setup_figure(figsize=(12, 8))
    for estimator, color, marker in [
        ("cuddl_construction", "#2E86AB", "o"),
        ("cuco_hll_construction", "#A23B72", "^"),
    ]:
        data = bench[bench["Estimator"] == estimator]
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

    # --- Accuracy figure -----------------------------------------------------
    fig, ax = pu.setup_figure(figsize=(12, 8))
    for estimator, color, marker in [
        ("cuddl_construction", "#2E86AB", "o"),
        ("cuco_hll_construction", "#A23B72", "^"),
    ]:
        data = acc[acc["Estimator"] == estimator]
        ax.semilogx(  # ty: ignore[unresolved-attribute]
            data["Items"],
            data["rel_error"] * 100.0,
            color=color,
            marker=marker,
            label=pu.paper_text(estimator.split("_")[0]),
        )
    pu.format_axis(ax, xlabel="Items", ylabel="Relative cardinality error (%)")
    pu.create_legend(ax)
    pu.save_figure(fig, output_dir / "accuracy.pdf", message="Wrote accuracy plot")
    typer.echo(f"Wrote {output_dir / 'accuracy.pdf'}")


if __name__ == "__main__":
    app()
