#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = ["jsonschema", "matplotlib", "pandas", "typer"]
# ///
"""Plot 15-bit versus 16-bit indexes across actual RefSeq database sizes."""

from pathlib import Path
from typing import Annotated

import matplotlib.pyplot as plt
import pandas as pd
import plot_utils as pu
import typer
from benchmark_schema import flatten_measurements, load_result

KEY_BIT_STYLES = {
    15: {
        "color": "#0072b2",
        "label": "15-bit key",
        "linestyle": "-",
        "marker": "o",
        "markerfacecolor": "#0072b2",
    },
    16: {
        "color": "#d55e00",
        "label": "16-bit key",
        "linestyle": "--",
        "marker": "s",
        "markerfacecolor": "none",
    },
}


def main(
    json_path: Annotated[
        Path,
        typer.Argument(exists=True, dir_okay=False, help="Search-decision result JSON"),
    ],
    output: Annotated[
        Path, typer.Option(dir_okay=False, help="Figure output path")
    ] = Path("results/cuddl-search-key-bits.pdf"),
) -> None:
    """Render indexed latency and search speedup for both key widths."""
    try:
        result = load_result(json_path, "search_decision")
    except ValueError as error:
        raise typer.BadParameter(f"{json_path}: {error}") from error
    data = pd.DataFrame(flatten_measurements(result))
    required = {
        "status",
        "reference_count",
        "query_count",
        "query_profile",
        "indexed_bucket_count",
        "index_mode",
        "exhaustive_p50_ms",
        "indexed_p50_ms",
    }
    missing = sorted(required - set(data.columns))
    if missing:
        raise typer.BadParameter(f"result is missing columns: {', '.join(missing)}")

    data = data.loc[
        (data["status"] == "ok")
        & (data["query_profile"] == "refseq_reference_scaling")
        & (pd.to_numeric(data["indexed_bucket_count"]) == 4096)
    ].copy()
    if data.empty:
        raise typer.BadParameter("result has no successful RefSeq scaling workloads")
    data["key_bits"] = (
        data["index_mode"].astype("string").str.extract(r"b(\d+)_k(\d+)")[1].astype(int)
    )
    for column in (
        "reference_count",
        "query_count",
        "exhaustive_p50_ms",
        "indexed_p50_ms",
    ):
        data[column] = pd.to_numeric(data[column])
    if (data["indexed_p50_ms"] <= 0).any():
        raise typer.BadParameter("result contains non-positive indexed timings")
    data["search_speedup"] = data["exhaustive_p50_ms"] / data["indexed_p50_ms"]

    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.7), layout="constrained")
    for bits, style in KEY_BIT_STYLES.items():
        series = data.loc[data["key_bits"] == bits].sort_values("reference_count")
        if series.empty:
            continue
        axes[0].plot(
            series["reference_count"],
            series["indexed_p50_ms"],
            color=style["color"],
            linestyle=style["linestyle"],
            marker=style["marker"],
            markerfacecolor=style["markerfacecolor"],
            label=style["label"],
        )
        axes[1].plot(
            series["reference_count"],
            series["search_speedup"],
            color=style["color"],
            linestyle=style["linestyle"],
            marker=style["marker"],
            markerfacecolor=style["markerfacecolor"],
            label=style["label"],
        )
    for ax in axes:
        ax.set_xscale("log", base=2)
        ax.set_xlabel(pu.paper_text("References"))
        labeled_references = [1024, 4096, 16384, 65536, 148108]
        ax.set_xticks(
            labeled_references,
            [f"{value / 1000.0:.0f}k" for value in labeled_references],
        )
        ax.grid(True, which="major", alpha=0.25)
        ax.margins(x=0.04, y=0.15)
    axes[0].set_yscale("log")
    axes[0].set_title(pu.paper_text("Indexed batch latency", bold=True))
    axes[0].set_ylabel(pu.paper_text("Median time (ms)"))
    axes[1].set_title(pu.paper_text("Indexed speedup", bold=True))
    axes[1].set_ylabel(pu.paper_text("Exhaustive / indexed time") + r" ($\times$)")
    axes[1].set_ylim(bottom=0.0)
    axes[1].axhline(1.0, color="#666666", linestyle="--", linewidth=1, label="Parity")
    axes[0].legend(frameon=False)
    axes[1].legend(frameon=False)
    fig.suptitle(
        pu.paper_text("Index key width across RefSeq database sizes", bold=True),
        fontsize=pu.TITLE_FONT_SIZE,
    )
    fig.supxlabel(pu.paper_text("Overlapping curves indicate equivalent performance"))
    output.parent.mkdir(parents=True, exist_ok=True)
    pu.save_figure(fig, output, pad_inches=0.10)


if __name__ == "__main__":
    typer.run(main)
