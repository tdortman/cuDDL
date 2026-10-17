#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = ["matplotlib", "pandas", "tabulate", "typer"]
# ///
"""Inflation throughput on the SMs against the hardware decompression engine.

Reads the `--csv` output of `cuddl-inflate-benchmark`, where both backends inflate the same
engine-sized DEFLATE members of each input, and plots inflated GB/s per input and backend. Prints
the throughputs and the engine's speedup as a Markdown table.
"""

from pathlib import Path
from typing import Annotated

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plot_utils as pu
import typer

BACKENDS = {"sm": "nvCOMP on SMs", "engine": "decompression engine"}


def main(
    csv: Annotated[Path, typer.Argument(exists=True, dir_okay=False)],
    title: Annotated[str, typer.Option(help="Figure title")] = "B300, inflation alone",
    names: Annotated[
        list[str],
        typer.Option("--name", help="Label for an input directory's name, as NAME=LABEL"),
    ] = [],
    output: Annotated[
        Path, typer.Option(help="PDF to write; a PNG goes beside it")
    ] = Path("results/inflate-engine.pdf"),
) -> None:
    runs = pd.read_csv(csv)
    runs["input"] = runs["input"].map(lambda p: Path(p).name)
    runs["gbps"] = runs["GlobalMem BW (bytes/sec)"] / 1e9
    labels = dict(name.split("=", 1) for name in names)
    inputs = list(dict.fromkeys(runs["input"]))
    table = runs.pivot(index="input", columns="backend", values="gbps").reindex(inputs)
    table["engine speedup"] = table["engine"] / table["sm"]
    print(table.rename(index=labels).to_markdown(floatfmt=".1f"))

    width = 0.8 / len(BACKENDS)
    x = np.arange(len(inputs))
    fig, ax = plt.subplots(figsize=(1.8 * len(inputs) + 1.5, 4), layout="constrained")
    for i, (backend, label) in enumerate(BACKENDS.items()):
        values = table[backend].to_numpy()
        bars = ax.bar(
            x + (i - (len(BACKENDS) - 1) / 2) * width, values, width, color=f"C{i + 2}", label=label
        )
        ax.bar_label(bars, fmt="%.0f", fontsize=8, padding=2)
    ax.set_xticks(x, [labels.get(i, i) for i in inputs], fontsize=8)
    ax.set_ylabel("inflated GB/s")
    ax.set_ylim(top=table[list(BACKENDS)].to_numpy().max() * 1.12)
    ax.set_title(title)
    fig.legend(loc="outside lower center", ncol=len(BACKENDS), fontsize=8, frameon=False)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output.with_suffix(".png"), dpi=200, bbox_inches="tight")
    pu.save_figure(fig, output)


if __name__ == "__main__":
    typer.run(main)
