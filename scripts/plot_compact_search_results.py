#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = ["jsonschema", "matplotlib", "pandas", "typer"]
# ///
"""Plot one or more cuDDL compact-search pipeline JSON reports."""

from pathlib import Path
from typing import Annotated, Any

import matplotlib.pyplot as plt
import plot_utils as pu
import typer
from benchmark_schema import load_result

STAGES = (
    ("parse_reference", "Parse ref."),
    ("parse_query", "Parse query"),
    ("host_to_device_transfer", "Host to device"),
    ("sketch_reference", "Sketch ref."),
    ("sketch_query", "Sketch query"),
    ("sketch_extract", "Extract"),
    ("database_build", "Database build"),
    ("index_build", "Index build"),
    ("search_exhaustive", "Exhaustive"),
    ("search_indexed", "Indexed"),
)
MARKERS = ("o", "s", "^", "D", "v", "P", "X", "*")


def load_report(path: Path) -> dict[str, Any]:
    try:
        report = load_result(path, "indexed_search")
    except ValueError as error:
        raise typer.BadParameter(f"{path}: {error}") from error

    statistics = pipeline_measurement(report)["timings"]
    missing = [stage for stage, _ in STAGES if stage not in statistics]
    if missing:
        raise typer.BadParameter(f"{path}: missing timing stages: {', '.join(missing)}")
    for stage, _ in STAGES:
        values = statistics[stage]
        try:
            minimum = float(values["min_ms"])
            median = float(values["median_ms"])
            maximum = float(values["max_ms"])
            samples = int(values["samples"])
        except (KeyError, TypeError, ValueError) as error:
            raise typer.BadParameter(
                f"{path}: invalid statistics for {stage}"
            ) from error
        if minimum <= 0 or not minimum <= median <= maximum or samples <= 0:
            raise typer.BadParameter(f"{path}: inconsistent statistics for {stage}")

    return report


def pipeline_measurement(report: dict[str, Any]) -> dict[str, Any]:
    try:
        return next(
            measurement
            for measurement in report["measurements"]
            if measurement["case"].get("measurement") == "pipeline"
        )
    except StopIteration as error:
        raise typer.BadParameter(
            "indexed_search result has no pipeline measurement"
        ) from error


def main(
    reports: Annotated[
        list[Path],
        typer.Argument(exists=True, dir_okay=False, help="Benchmark result JSON files"),
    ],
    output: Annotated[
        Path, typer.Option(dir_okay=False, help="Comparison figure output path")
    ] = Path("results/cuddl-compact-search-results.pdf"),
) -> None:
    """Compare timing distributions, peak memory, and indexed-search work."""
    if not reports:
        raise typer.BadParameter("provide at least one benchmark JSON file")
    loaded = [load_report(path) for path in reports]
    names = [report["name"] for report in loaded]
    if len(names) != len(set(names)):
        raise typer.BadParameter("benchmark names must be unique")

    colors = [plt.get_cmap("tab10")(index % 10) for index in range(len(loaded))]
    markers = [MARKERS[index % len(MARKERS)] for index in range(len(loaded))]
    fig = plt.figure(figsize=(11.0, 7.2))
    grid = fig.add_gridspec(2, 2, height_ratios=(1.45, 1.0), hspace=0.42, wspace=0.30)
    timing_ax = fig.add_subplot(grid[0, :])
    memory_ax = fig.add_subplot(grid[1, 0])
    work_ax = fig.add_subplot(grid[1, 1])

    stage_x = list(range(len(STAGES)))
    offset_width = min(0.12, 0.7 / len(loaded))
    for report_index, (report, color, marker) in enumerate(
        zip(loaded, colors, markers, strict=True)
    ):
        offset = (report_index - (len(loaded) - 1) / 2) * offset_width
        x = [position + offset for position in stage_x]
        statistics = pipeline_measurement(report)["timings"]
        medians = [statistics[stage]["median_ms"] for stage, _ in STAGES]
        lower = [
            median - statistics[stage]["min_ms"]
            for median, (stage, _) in zip(medians, STAGES, strict=True)
        ]
        upper = [
            statistics[stage]["max_ms"] - median
            for median, (stage, _) in zip(medians, STAGES, strict=True)
        ]
        timing_ax.errorbar(
            x,
            medians,
            yerr=[lower, upper],
            fmt=marker,
            color=color,
            capsize=2.5,
            markersize=6,
            linewidth=1.2,
            label=pu.paper_text(report["name"]),
        )

    timing_ax.set_yscale("log")
    timing_ax.set_ylabel(pu.paper_text("Time [ms]"), fontsize=pu.AXIS_LABEL_FONT_SIZE)
    timing_ax.set_xticks(
        stage_x, [pu.paper_text(label) for _, label in STAGES], rotation=24, ha="right"
    )
    timing_ax.set_title(
        pu.paper_text("Stage medians and observed ranges", bold=True),
        fontsize=pu.TITLE_FONT_SIZE,
    )
    timing_ax.grid(True, axis="y", which="both", linestyle="--", alpha=pu.GRID_ALPHA)
    timing_ax.legend(
        frameon=True,
        framealpha=0.9,
        edgecolor="#d0d0d0",
        loc="upper right",
        ncols=min(4, len(loaded)),
        fontsize=pu.TICK_LABEL_FONT_SIZE,
    )

    run_x = list(range(len(loaded)))
    bar_width = 0.36
    gib = 1024**3
    memory_ax.bar(
        [x - bar_width / 2 for x in run_x],
        [
            pipeline_measurement(report)["memory_bytes"]["host_peak"] / gib
            for report in loaded
        ],
        bar_width,
        color="#2c7fb8",
        label=pu.paper_text("Host"),
    )
    memory_ax.bar(
        [x + bar_width / 2 for x in run_x],
        [
            pipeline_measurement(report)["memory_bytes"]["device_peak"] / gib
            for report in loaded
        ],
        bar_width,
        color="#d95f0e",
        label=pu.paper_text("Device"),
    )
    memory_ax.set_ylabel(
        pu.paper_text("Peak memory [GiB]"), fontsize=pu.AXIS_LABEL_FONT_SIZE
    )
    memory_ax.set_title(
        pu.paper_text("Peak memory", bold=True), fontsize=pu.TITLE_FONT_SIZE
    )
    memory_ax.set_ylim(
        0,
        max(
            max(
                pipeline_measurement(report)["memory_bytes"]["host_peak"]
                for report in loaded
            ),
            max(
                pipeline_measurement(report)["memory_bytes"]["device_peak"]
                for report in loaded
            ),
        )
        / gib
        * 1.35,
    )
    memory_ax.legend(frameon=False, loc="upper center", ncols=2)

    posting_bars = work_ax.bar(
        [x - bar_width / 2 for x in run_x],
        [
            pipeline_measurement(report)["metrics"]["posting_visits"]
            for report in loaded
        ],
        bar_width,
        color="#2a9d8f",
        label=pu.paper_text("Posting visits"),
    )
    candidate_bars = work_ax.bar(
        [x + bar_width / 2 for x in run_x],
        [pipeline_measurement(report)["metrics"]["candidates"] for report in loaded],
        bar_width,
        color="#a23b72",
        label=pu.paper_text("Candidates"),
    )
    work_ax.bar_label(posting_bars, fmt="%g", padding=4, fontsize=pu.BAR_FONT_SIZE)
    work_ax.bar_label(candidate_bars, fmt="%g", padding=4, fontsize=pu.BAR_FONT_SIZE)
    work_ax.set_yscale("symlog", linthresh=1)
    work_ax.set_ylim(
        0,
        max(
            1,
            max(
                pipeline_measurement(report)["metrics"]["posting_visits"]
                for report in loaded
            ),
            max(
                pipeline_measurement(report)["metrics"]["candidates"]
                for report in loaded
            ),
        )
        * 10,
    )
    work_ax.set_ylabel(
        pu.paper_text("Count [symmetric log scale]"), fontsize=pu.AXIS_LABEL_FONT_SIZE
    )
    work_ax.set_title(
        pu.paper_text("Indexed-search work", bold=True), fontsize=pu.TITLE_FONT_SIZE
    )
    work_ax.legend(frameon=False)

    labels = [pu.paper_text(name) for name in names]
    for ax in (memory_ax, work_ax):
        ax.set_xticks(run_x, labels, rotation=20, ha="right")
        ax.grid(True, axis="y", linestyle="--", alpha=pu.GRID_ALPHA)
        ax.tick_params(labelsize=pu.TICK_LABEL_FONT_SIZE)
    timing_ax.tick_params(labelsize=pu.TICK_LABEL_FONT_SIZE)

    fig.suptitle(
        pu.paper_text("cuDDL compact-search pipeline comparison", bold=True),
        fontsize=pu.TITLE_FONT_SIZE + 2,
        y=0.99,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    pu.save_figure(fig, output, pad_inches=0.08)


if __name__ == "__main__":
    typer.run(main)
