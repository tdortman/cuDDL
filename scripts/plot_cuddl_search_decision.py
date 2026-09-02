#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = ["jsonschema", "matplotlib", "pandas", "typer"]
# ///
"""Plot synthetic all-to-all decisions and actual RefSeq database scaling."""

import math
from itertools import pairwise
from pathlib import Path
from typing import Annotated

import matplotlib.pyplot as plt
import pandas as pd
import plot_utils as pu
import typer
from benchmark_schema import flatten_measurements, load_result
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from matplotlib.ticker import FuncFormatter, PercentFormatter

COMMON_COLUMNS = {
    "status",
    "reference_count",
    "query_count",
    "query_profile",
    "indexed_bucket_count",
    "index_mode",
    "exhaustive_p50_ms",
    "indexed_p50_ms",
}


def skew_fraction(value: object) -> float:
    return float(str(value).removesuffix("%")) / 100.0


def speedup_label(log2_speedup: float) -> str:
    ratio = 2.0**log2_speedup
    if ratio >= 100.0:
        return f"{ratio:.0f}"
    if ratio >= 10.0:
        return f"{ratio:.1f}"
    return f"{ratio:.2g}"


def reference_label(value: float) -> str:
    return f"{value / 1000.0:.0f}k"


def cell_edges(values: list[float]) -> list[float]:
    if len(values) == 1:
        return [values[0] - 0.025, values[0] + 0.025]
    return [
        values[0] - (values[1] - values[0]) / 2.0,
        *[(left + right) / 2.0 for left, right in pairwise(values)],
        values[-1] + (values[-1] - values[-2]) / 2.0,
    ]


def plot_synthetic(ax: plt.Axes, data: pd.DataFrame) -> None:
    required = {"fill_ratio", "skew"}
    missing = sorted(required - set(data.columns))
    if missing:
        raise typer.BadParameter(
            f"synthetic rows are missing columns: {', '.join(missing)}"
        )
    data = data.loc[
        (pd.to_numeric(data["indexed_bucket_count"]) == 2048)
        & (pd.to_numeric(data["fill_ratio"]) == 1.0)
    ].copy()
    if data.empty:
        raise typer.BadParameter("result has no canonical synthetic all-to-all rows")
    data["hot_fraction"] = data["skew"].map(skew_fraction)
    data["log2_speedup"] = (
        pd.to_numeric(data["exhaustive_p50_ms"]) / pd.to_numeric(data["indexed_p50_ms"])
    ).map(math.log2)
    references = sorted(data["reference_count"].astype(int).unique())
    hot_fractions = sorted(data["hot_fraction"].unique())
    matrix = (
        data.pivot(
            index="reference_count", columns="hot_fraction", values="log2_speedup"
        )
        .reindex(index=references, columns=hot_fractions)
        .to_numpy(dtype=float)
    )
    maximum = max(1.0, float(data["log2_speedup"].abs().max()))
    mesh = ax.pcolormesh(
        cell_edges(hot_fractions),
        [index - 0.5 for index in range(len(references) + 1)],
        matrix,
        cmap=LinearSegmentedColormap.from_list(
            "search_decision", ["#d55e00", "#ffffff", "#0072b2"]
        ),
        norm=TwoSlopeNorm(vmin=-maximum, vcenter=0.0, vmax=maximum),
        edgecolors="white",
        linewidth=0.8,
    )
    for row, values in enumerate(matrix):
        for column, value in enumerate(values):
            if math.isfinite(value):
                ax.text(
                    hot_fractions[column],
                    row,
                    r"$\times$" + speedup_label(value),
                    ha="center",
                    va="center",
                    fontsize=9,
                    color="white" if abs(value) > maximum * 0.55 else "#222222",
                )
    ax.set(
        title=pu.paper_text("Synthetic all-to-all", bold=True),
        xlabel=pu.paper_text("Hot buckets (score shared by 1% of references)"),
        ylabel=pu.paper_text("References"),
        yticks=range(len(references)),
        yticklabels=[f"{value:,}" for value in references],
    )
    ax.set_xticks(hot_fractions, [f"{value:.0%}" for value in hot_fractions])
    ax.set_ylim(len(references) - 0.5, -0.5)
    colorbar = ax.figure.colorbar(mesh, ax=ax, pad=0.02)
    colorbar.set_ticks(
        [
            -maximum,
            *[tick for tick in colorbar.get_ticks() if -maximum < tick < maximum],
            maximum,
        ]
    )
    colorbar.ax.yaxis.set_major_formatter(
        FuncFormatter(lambda value, _position: speedup_label(value))
    )
    colorbar.set_label(
        pu.paper_text("exhaustive / indexed median time") + r" ($\times$)"
    )


def plot_refseq(axes: tuple[plt.Axes, plt.Axes, plt.Axes], data: pd.DataFrame) -> None:
    missing = sorted({"selected_candidates"} - set(data.columns))
    if missing:
        raise typer.BadParameter(
            f"RefSeq scaling rows are missing columns: {', '.join(missing)}"
        )
    data = data.loc[pd.to_numeric(data["indexed_bucket_count"]) == 4096].copy()
    if data.empty:
        raise typer.BadParameter("result has no RefSeq reference-scaling rows")
    data["query_count"] = pd.to_numeric(data["query_count"])
    data["reference_count"] = pd.to_numeric(data["reference_count"])
    data["selected_candidates"] = pd.to_numeric(data["selected_candidates"])
    data = data.sort_values("reference_count")
    latency_ax, speedup_ax, candidates_ax = axes
    for implementation, column, color, marker in (
        ("Exhaustive", "exhaustive_p50_ms", "#d55e00", "s"),
        ("Indexed", "indexed_p50_ms", "#0072b2", "o"),
    ):
        latency_ax.plot(
            data["reference_count"],
            pd.to_numeric(data[column]),
            label=implementation,
            color=color,
            marker=marker,
            linewidth=2,
            markersize=5,
        )
    speedup_ax.plot(
        data["reference_count"],
        pd.to_numeric(data["exhaustive_p50_ms"])
        / pd.to_numeric(data["indexed_p50_ms"]),
        color="#0072b2",
        marker="o",
        linewidth=2,
    )
    candidates_ax.plot(
        data["reference_count"],
        data["selected_candidates"] / (data["query_count"] * data["reference_count"]),
        color="#009e73",
        marker="^",
        linewidth=2,
    )
    latency_ax.set(
        title=pu.paper_text("RefSeq batch latency (4,096 queries)", bold=True),
        ylabel=pu.paper_text("Median time (ms)"),
    )
    speedup_ax.set(
        title=pu.paper_text("Indexed speedup", bold=True),
        ylabel=pu.paper_text("Exhaustive / indexed time") + r" ($\times$)",
    )
    candidates_ax.set(
        title=pu.paper_text("Candidate fraction", bold=True),
        ylabel=pu.paper_text("Candidate pairs / database pairs"),
    )
    for ax in axes:
        ax.set_xscale("log", base=2)
        labeled_references = [1024, 4096, 16384, 65536, 148108]
        ax.set_xticks(
            labeled_references,
            [reference_label(value) for value in labeled_references],
        )
        ax.set_xlabel(pu.paper_text("References"))
        ax.grid(True, which="major", alpha=0.25)
        ax.margins(x=0.05, y=0.15)
    latency_ax.set_yscale("log")
    speedup_ax.set_ylim(bottom=0.0)
    speedup_ax.axhline(1.0, color="#666666", linestyle="--", linewidth=1)
    speedup_ax.text(
        data["reference_count"].max(),
        1.0,
        pu.paper_text("parity"),
        ha="right",
        va="bottom",
    )
    candidate_fraction = data["selected_candidates"] / (
        data["query_count"] * data["reference_count"]
    )
    candidates_ax.set_ylim(0.0, float(candidate_fraction.max()) * 1.15)
    candidates_ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=1))
    latency_ax.legend(frameon=False)


def main(
    json_path: Annotated[
        Path,
        typer.Argument(exists=True, dir_okay=False, help="Search-decision result JSON"),
    ],
    output: Annotated[
        Path, typer.Option(dir_okay=False, help="Figure output path")
    ] = Path("results/cuddl-search-decision.pdf"),
) -> None:
    """Render every successful 16-bit search profile."""
    try:
        result = load_result(json_path, "search_decision")
    except ValueError as error:
        raise typer.BadParameter(f"{json_path}: {error}") from error
    data = pd.DataFrame(flatten_measurements(result))
    missing = sorted(COMMON_COLUMNS - set(data.columns))
    if missing:
        raise typer.BadParameter(f"result is missing columns: {', '.join(missing)}")
    data = data.loc[data["status"] == "ok"].copy()
    data["key_bits"] = (
        data["index_mode"].astype("string").str.extract(r"b(\d+)_k(\d+)")[1].astype(int)
    )
    data = data.loc[data["key_bits"] == 16].copy()
    if data.empty:
        raise typer.BadParameter("result has no successful 16-bit workloads")
    timings = data[["exhaustive_p50_ms", "indexed_p50_ms"]].apply(pd.to_numeric)
    if (timings <= 0).any().any():
        raise typer.BadParameter("result contains non-positive search timings")

    synthetic = data.loc[data["query_profile"] == "all_to_all"].copy()
    refseq = data.loc[data["query_profile"] == "refseq_reference_scaling"].copy()
    if synthetic.empty and refseq.empty:
        raise typer.BadParameter("result has no supported search profiles")
    if not synthetic.empty and not refseq.empty:
        fig = plt.figure(figsize=(12.0, 7.2), layout="constrained")
        grid = fig.add_gridspec(2, 3)
        plot_synthetic(fig.add_subplot(grid[0, :]), synthetic)
        plot_refseq(
            tuple(fig.add_subplot(grid[1, column]) for column in range(3)), refseq
        )
    elif not synthetic.empty:
        fig, ax = plt.subplots(figsize=(9.0, 3.6), layout="constrained")
        plot_synthetic(ax, synthetic)
    else:
        fig, axes = plt.subplots(1, 3, figsize=(12.0, 3.7), layout="constrained")
        plot_refseq(tuple(axes), refseq)
    fig.suptitle(
        pu.paper_text("Indexed versus exhaustive search", bold=True),
        fontsize=pu.TITLE_FONT_SIZE,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    pu.save_figure(fig, output, pad_inches=0.10)


if __name__ == "__main__":
    typer.run(main)
