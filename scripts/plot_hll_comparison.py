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
- `accuracy`: relative cardinality error (from the benchmark's accuracy side-channel) against
  item count.

Reads the benchmark's nvbench CSV (`--nvbench-csv`) and the accuracy rows it prints to stdout
(`--accuracy`), then renders both figures. Styling comes from `./scripts/plot_utils.py`.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Annotated

import pandas as pd
import plot_utils as pu
import typer

app = typer.Typer(
    help="Plot cuDDL vs cuco HLL construction benchmark", no_args_is_help=True
)


def parse_accuracy(lines: Iterator[str]) -> pd.DataFrame:
    """Collect `cuddl,count=..,rel_error=..` and `cuco_hll,..` rows from the benchmark stdout."""
    rows: list[dict[str, str]] = []
    for line in lines:
        line = line.strip()
        if not (line.startswith(("cuddl,", "cuco_hll,"))):
            continue
        fields: dict[str, str] = {}
        head, _, body = line.partition(",")
        fields["benchmark"] = head
        for pair in body.split(","):
            key, _, value = pair.partition("=")
            fields[key] = value
        rows.append(fields)
    return pd.DataFrame(rows)


@app.command()
def plot(
    nvbench_csv: Annotated[Path, typer.Option(help="Benchmark --csv output")],
    accuracy: Annotated[Path, typer.Option(help="Benchmark stdout saved to a file")],
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

    # Accuracy from the benchmark's printed rows.
    with accuracy.open() as f:
        acc = parse_accuracy(f)
    acc["Items"] = acc["count"].astype("int64")
    acc["rel_error"] = acc["rel_error"].astype("float64")

    # --- Throughput figure ---------------------------------------------------
    fig, ax = pu.setup_figure(figsize=(12, 8))
    for estimator, color, marker in [
        ("cuddl_construction", "#2E86AB", "o"),
        ("cuco_hll_construction", "#A23B72", "^"),
    ]:
        data = bench[bench["Estimator"] == estimator]
        ax.semilogx(
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
        ("cuddl", "#2E86AB", "o"),
        ("cuco_hll", "#A23B72", "^"),
    ]:
        data = acc[acc["benchmark"] == estimator]
        ax.semilogx(
            data["Items"],
            data["rel_error"] * 100.0,
            color=color,
            marker=marker,
            label=pu.paper_text(estimator),
        )
    pu.format_axis(ax, xlabel="Items", ylabel="Relative cardinality error (%)")
    pu.create_legend(ax)
    pu.save_figure(fig, output_dir / "accuracy.pdf", message="Wrote accuracy plot")
    typer.echo(f"Wrote {output_dir / 'accuracy.pdf'}")


if __name__ == "__main__":
    app()
