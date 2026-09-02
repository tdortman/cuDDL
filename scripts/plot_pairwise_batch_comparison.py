#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = ["jsonschema", "matplotlib", "pandas", "typer"]
# ///
"""Plot pairwise quality, throughput, runtime composition, and speedup separately."""

import math
from pathlib import Path
from typing import Annotated

import pandas as pd
import plot_utils as pu
import typer
from benchmark_schema import flatten_measurements, load_result

QUALITY_METRICS = (
    ("containment_absolute_error", "Containment"),
    ("completeness_absolute_error", "Completeness"),
    ("wkid_absolute_error", "WKID"),
    ("ani_absolute_error", "ANI"),
)
SYMMETRIC_METRICS = {"wkid_absolute_error", "ani_absolute_error"}
CUDDL = pu.FILTER_STYLES["cuddl"]
BBTOOLS = pu.FILTER_STYLES["cuco_hll"]


def require(data: pd.DataFrame, columns: set[str], name: str) -> None:
    missing = sorted(columns - set(data.columns))
    if missing:
        raise typer.BadParameter(f"{name} is missing columns: {', '.join(missing)}")


def add_top_legend(fig, ax, *, ncol: int) -> None:
    """Place this figure's legend above its single plot."""
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    axes_box = ax.get_position()
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        fontsize=pu.LEGEND_FONT_SIZE,
        loc="lower center",
        bbox_to_anchor=((axes_box.x0 + axes_box.x1) / 2, axes_box.y1 + 0.06),
        ncol=ncol,
        framealpha=pu.LEGEND_FRAME_ALPHA,
    )


def main(
    quality_json: Annotated[
        Path,
        typer.Argument(
            exists=True, dir_okay=False, help="Pairwise accuracy benchmark result JSON"
        ),
    ],
    batch_json: Annotated[
        Path,
        typer.Argument(
            exists=True, dir_okay=False, help="Pairwise batch benchmark result JSON"
        ),
    ],
    output_dir: Annotated[
        Path, typer.Option(file_okay=False, help="Figure output directory")
    ] = Path("results/pairwise-batch-comparison"),
) -> None:
    """Render four standalone comparison figures from shared JSON results."""
    output_dir.mkdir(parents=True, exist_ok=True)
    try:
        quality_result = load_result(quality_json, "pairwise_accuracy")
    except ValueError as error:
        raise typer.BadParameter(f"{quality_json}: {error}") from error
    quality = pd.DataFrame(flatten_measurements(quality_result))
    require(
        quality,
        {"implementation", "orientation", *(column for column, _ in QUALITY_METRICS)},
        "quality JSON",
    )
    try:
        batch_result = load_result(batch_json, "pairwise_batch")
    except ValueError as error:
        raise typer.BadParameter(f"{batch_json}: {error}") from error
    batch = pd.DataFrame(flatten_measurements(batch_result))
    require(
        batch,
        {
            "implementation",
            "mode",
            "sketch_pairs",
            "threads",
            "seconds_per_batch",
            "pair_comparisons_per_second",
        },
        "batch JSON",
    )

    cuddl = batch[batch["implementation"] == "cuddl"]
    bbtools = batch[batch["implementation"] == "bbtools"]
    gpu = cuddl.pivot(index="sketch_pairs", columns="mode", values="seconds_per_batch")
    pair_comparisons_per_second = cuddl.pivot(
        index="sketch_pairs", columns="mode", values="pair_comparisons_per_second"
    )
    sketch_pair_counts = sorted(int(value) for value in gpu.index)
    gpu = gpu.reindex(sketch_pair_counts)
    pair_comparisons_per_second = pair_comparisons_per_second.reindex(
        sketch_pair_counts
    )
    cpu = (
        bbtools.groupby(["mode", "sketch_pairs"])["pair_comparisons_per_second"]
        .median()
        .unstack(0)
    )
    cpu_pair_counts = sorted(int(value) for value in cpu.index)
    cpu = cpu.reindex(cpu_pair_counts)
    thread_counts: dict[str, int] = {}
    for mode in ("parallel", "sequential"):
        values = sorted(
            {int(value) for value in bbtools.loc[bbtools["mode"] == mode, "threads"]}
        )
        if len(values) != 1:
            raise typer.BadParameter(
                f"expected one BBTools {mode} thread count, got {values}"
            )
        thread_counts[mode] = values[0]
    thread_count = thread_counts["parallel"]

    missing_implementations = {"cuddl", "bbtools"} - set(quality["implementation"])
    if missing_implementations:
        raise typer.BadParameter(
            "quality JSON is missing implementations: "
            + ", ".join(sorted(missing_implementations))
        )
    positions = list(range(len(QUALITY_METRICS)))
    width = 0.36
    fig, quality_ax = pu.setup_figure()
    for offset, implementation, label, style in (
        (-0.5, "cuddl", "cuDDL", CUDDL),
        (0.5, "bbtools", "BBTools DDL", BBTOOLS),
    ):
        selected = quality[quality["implementation"] == implementation]
        values = [
            (
                selected[selected["orientation"] == "query_to_reference"]
                if column in SYMMETRIC_METRICS
                else selected
            )[column].quantile(0.95)
            * 100
            for column, _ in QUALITY_METRICS
        ]
        bars = quality_ax.bar(
            [position + offset * width for position in positions],
            values,
            width,
            color=style["color"],
            edgecolor="black",
            linewidth=pu.BAR_EDGE_WIDTH,
            label=label,
        )
        quality_ax.bar_label(
            bars,
            fmt="%.2f",
            fontsize=pu.BAR_FONT_SIZE,
            fontweight="bold",
            padding=3,
        )
    quality_ax.set_xticks(
        positions,
        [label for _, label in QUALITY_METRICS],
    )
    quality_ax.set_ylim(0, quality_ax.get_ylim()[1] * 1.15)
    pu.format_axis(
        quality_ax,
        xlabel="",
        ylabel="P95 absolute error (%)",
        title="Estimation quality",
        xscale=None,
        grid=False,
    )
    quality_ax.tick_params(axis="both", labelsize=pu.TICK_LABEL_FONT_SIZE)
    quality_ax.grid(axis="y", linestyle="--", alpha=pu.GRID_ALPHA)
    add_top_legend(fig, quality_ax, ncol=2)
    pu.save_figure(fig, output_dir / "estimation_quality.pdf")

    fig, throughput_ax = pu.setup_figure()
    throughput_ax.plot(
        sketch_pair_counts,
        pair_comparisons_per_second["device_resident"] / 1e6,
        color=CUDDL["color"],
        marker=CUDDL["marker"],
        linewidth=pu.LINE_WIDTH,
        markersize=pu.MARKER_SIZE,
        label="cuDDL, device resident",
    )
    throughput_ax.plot(
        sketch_pair_counts,
        pair_comparisons_per_second["transfer_each_batch"] / 1e6,
        color=CUDDL["color"],
        marker="^",
        linestyle="--",
        linewidth=pu.LINE_WIDTH,
        markersize=pu.MARKER_SIZE,
        label="cuDDL, transfer each batch",
    )
    throughput_ax.plot(
        cpu_pair_counts,
        cpu["parallel"] / 1e6,
        color=BBTOOLS["color"],
        marker=BBTOOLS["marker"],
        linewidth=pu.LINE_WIDTH,
        markersize=pu.MARKER_SIZE,
        label=f"BBTools, {thread_count} thread{'s' if thread_count != 1 else ''}",
    )
    throughput_ax.plot(
        cpu_pair_counts,
        cpu["sequential"] / 1e6,
        color=BBTOOLS["color"],
        linestyle=":",
        linewidth=pu.LINE_WIDTH,
        label=(
            f"BBTools, {thread_counts['sequential']} "
            f"thread{'s' if thread_counts['sequential'] != 1 else ''}"
        ),
    )
    pu.format_axis(
        throughput_ax,
        xlabel="Sketch pairs per batch",
        ylabel="Pair comparisons per second (millions)",
        title="Batched throughput",
        xscale="log",
        yscale="log",
    )
    throughput_ax.tick_params(axis="both", labelsize=pu.TICK_LABEL_FONT_SIZE)
    add_top_legend(fig, throughput_ax, ncol=2)
    pu.save_figure(fig, output_dir / "batched_throughput.pdf")

    components = pd.DataFrame(
        {
            "Sketch transfers": gpu["h2d"] + gpu["d2h"],
            "Comparison kernel": gpu["device_resident"],
        },
        index=gpu.index,
    )
    shares = components.div(components.sum(axis=1), axis=0) * 100
    stage_names = [
        "Sketch transfers",
        "Comparison kernel",
    ]
    matrix = shares[stage_names].T
    fig, stages_ax = pu.setup_figure()
    stages_ax.imshow(matrix, aspect="auto", cmap="Blues", vmin=0, vmax=100)
    stages_ax.set_xticks(
        range(len(sketch_pair_counts)),
        [rf"$2^{{{int(math.log2(count))}}}$" for count in sketch_pair_counts],
    )
    stages_ax.set_yticks(range(len(stage_names)), stage_names)
    for row, stage in enumerate(stage_names):
        for column, value in enumerate(matrix.loc[stage]):
            label = (
                r"$<0.1$"
                if value < 0.1
                else f"{value:.1f}"
                if value < 10
                else f"{value:.0f}"
            )
            stages_ax.text(
                column,
                row,
                label,
                ha="center",
                va="center",
                color="white" if value >= 50 else "black",
                fontsize=pu.BAR_FONT_SIZE,
            )
    pu.format_axis(
        stages_ax,
        xlabel="Sketch pairs per batch",
        ylabel="",
        title="Isolated phase timing share (%)",
        xscale=None,
        grid=False,
    )
    stages_ax.tick_params(axis="both", labelsize=pu.TICK_LABEL_FONT_SIZE)
    fig.tight_layout()
    pu.save_figure(fig, output_dir / "runtime_share.pdf")

    shared_pair_counts = sorted(set(sketch_pair_counts) & set(cpu_pair_counts))
    resident_speedup = (
        pair_comparisons_per_second.loc[shared_pair_counts, "device_resident"]
        / cpu.loc[shared_pair_counts, "parallel"]
    )
    transferred_speedup = (
        pair_comparisons_per_second.loc[shared_pair_counts, "transfer_each_batch"]
        / cpu.loc[shared_pair_counts, "parallel"]
    )
    fig, speedup_ax = pu.setup_figure()
    speedup_ax.plot(
        shared_pair_counts,
        resident_speedup,
        color=CUDDL["color"],
        marker="o",
        linewidth=pu.LINE_WIDTH,
        markersize=pu.MARKER_SIZE,
        label="Device resident",
    )
    speedup_ax.plot(
        shared_pair_counts,
        transferred_speedup,
        color="#E76F51",
        marker="^",
        linestyle="--",
        linewidth=pu.LINE_WIDTH,
        markersize=pu.MARKER_SIZE,
        label="Transfer each batch",
    )
    speedup_ax.axhline(
        1.0, color="#555555", linewidth=pu.REFERENCE_LINE_WIDTH, linestyle=":"
    )
    pu.format_axis(
        speedup_ax,
        xlabel="Sketch pairs per batch",
        ylabel=f"Speedup over {thread_count}-thread BBTools",
        title="Relative throughput",
        xscale="log",
        yscale="log",
    )
    speedup_ax.tick_params(axis="both", labelsize=pu.TICK_LABEL_FONT_SIZE)
    add_top_legend(fig, speedup_ax, ncol=2)
    pu.save_figure(fig, output_dir / "relative_throughput.pdf")

    typer.echo(
        f"At {shared_pair_counts[-1]} sketch pairs, resident cuDDL is "
        f"{resident_speedup.iloc[-1]:.1f}x "
        f"the {thread_count}-thread BBTools throughput; transfer-each-batch is "
        f"{transferred_speedup.iloc[-1]:.2f}x."
    )


if __name__ == "__main__":
    typer.run(main)
