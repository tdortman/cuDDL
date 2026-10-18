#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = ["matplotlib", "pandas", "tabulate", "typer"]
# ///
"""Reference-build time per input format, decompression backend and loader count on one GPU.

Reads the `RESULT [input=] rep= workers= backend= seconds=` lines a decompression run prints, one
per build. `gpu` lets the hardware decompression engine inflate what it can and `gpu-cores` runs
the same pipeline with nvCOMP on the SMs, so the gap between the two is what the engine adds; the
gap between inputs, such as gzip as distributed and the same genomes as BGZF, is what the format
costs. Prints the median per point as a Markdown table per input and draws one panel per loader
count: a cluster of bars per input, one bar per backend, labelled with the median and spanning
the range of repetitions.
"""

import re
from pathlib import Path
from typing import Annotated

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plot_utils as pu
import typer

RESULT = re.compile(
    r"^RESULT (?:input=(\S+) )?rep=(\d+) workers=(\d+) backend=(\S+) seconds=([\d.]+)$"
)
LABELS = {
    "cpu": "cpu (libdeflate)",
    "hybrid": "hybrid (CPU inflate, GPU parse)",
    "gpu-cores": "gpu, nvCOMP on SMs",
    "gpu": "gpu, decompression engine",
}


def main(
    logs: Annotated[list[Path], typer.Argument(exists=True, dir_okay=False)],
    title: Annotated[str, typer.Option(help="Figure title")] = "B300, 100,000 genomes",
    names: Annotated[
        list[str] | None,
        typer.Option("--name", help="Legend name for an input, as INPUT=NAME"),
    ] = None,
    output: Annotated[
        Path, typer.Option(help="PDF to write; a PNG goes beside it")
    ] = Path("results/decompression-engine.pdf"),
) -> None:
    rows = [
        m.groups()
        for log in logs
        for line in log.read_text().splitlines()
        if (m := RESULT.match(line))
    ]
    if not rows:
        raise typer.BadParameter(f"no RESULT lines in {', '.join(map(str, logs))}")
    runs = pd.DataFrame(
        rows, columns=["input", "rep", "workers", "backend", "seconds"]
    ).astype({"rep": int, "workers": int, "seconds": float})
    runs["input"] = runs["input"].fillna("input")
    legend = dict(name.split("=", 1) for name in names or [])
    stats = runs.groupby(["input", "workers", "backend"])["seconds"].agg(
        ["median", "min", "max"]
    )
    inputs = list(dict.fromkeys(runs["input"]))
    for input_ in inputs:
        table = stats.xs(input_, level="input")["median"].unstack("backend")
        if {"gpu", "gpu-cores"} <= set(table.columns):
            table["engine speedup"] = table["gpu-cores"] / table["gpu"]
        print(f"\n{legend.get(input_, input_)}\n")
        print(table.to_markdown(floatfmt=".2f"))

    workers = sorted(runs["workers"].unique())
    backends = [b for b in LABELS if b in set(runs["backend"])]
    width = 0.8 / len(backends)
    x = np.arange(len(inputs))
    cols = min(2, len(workers))
    rows = -(-len(workers) // cols)
    fig, axes = plt.subplots(
        rows,
        cols,
        figsize=(1.6 * len(inputs) * cols + 1, 3.6 * rows + 0.6),
        sharey=True,
        squeeze=False,
        layout="constrained",
    )
    for ax in axes.flat[len(workers) :]:
        ax.set_visible(False)
    for ax, count in zip(axes.flat, workers):
        for i, backend in enumerate(backends):
            part = stats.xs((count, backend), level=("workers", "backend")).reindex(
                inputs
            )
            bars = ax.bar(
                x + (i - (len(backends) - 1) / 2) * width,
                part["median"],
                width,
                yerr=[part["median"] - part["min"], part["max"] - part["median"]],
                capsize=2,
                color=f"C{i}",
                label=LABELS[backend],
            )
            for bar, median, top in zip(bars, part["median"], part["max"]):
                if np.isfinite(median):
                    ax.text(
                        bar.get_x() + width / 2,
                        top * 1.04,
                        f"{median:.0f}" if median >= 100 else f"{median:.1f}",
                        ha="center",
                        va="bottom",
                        fontsize=7,
                    )
        ax.set_xticks(x, [legend.get(i, i) for i in inputs], fontsize=8)
        ax.set_title(f"{count} CPU loaders", fontsize=10)
    first = axes[0][0]
    first.set_yscale("log")
    # Room above the tallest whisker for its label.
    bottom, top = stats["min"].min() * 0.7, stats["max"].max() * 1.5
    first.set_yticks(
        [
            t
            for e in range(-1, 4)
            for t in (10**e, 2 * 10**e, 5 * 10**e)
            if bottom <= t <= top
        ]
    )
    first.set_ylim(bottom, top)
    first.yaxis.set_major_formatter(plt.FormatStrFormatter("%g"))
    first.yaxis.set_minor_formatter(plt.NullFormatter())
    for row in axes:
        row[0].set_ylabel("reference build time (s)")
    fig.legend(
        *first.get_legend_handles_labels(),
        loc="outside lower center",
        ncol=len(backends),
        fontsize=8,
        frameon=False,
    )
    fig.suptitle(title)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output.with_suffix(".png"), dpi=200, bbox_inches="tight")
    pu.save_figure(fig, output)


if __name__ == "__main__":
    typer.run(main)
