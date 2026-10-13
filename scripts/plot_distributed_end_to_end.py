#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = ["matplotlib", "pandas", "typer"]
# ///
"""Figures for cuddl-distributed-all-to-all-benchmark JSON reports.

Per report: stacked per-rank phase times (list, sketch, allgather, build, index,
search, download, write) and the wall time per phase of the slowest rank. The
benchmark records the search phase including its downloads and writes; search
here is the rest of it: claiming, launching and waiting for tiles. Prints a
summary table with wall time, node-hours, pairs and bytes written for every
report.
"""

import json
from pathlib import Path
from typing import Annotated

import numpy as np
import plot_utils as pu
import typer

PHASES = [
    "list",
    "sketch",
    "allgather",
    "build",
    "index",
    "search",
    "download",
    "write",
]
LABELS = {"list": "list files"}
COLORS = {
    "list": "#8C8C8C",
    "sketch": "#8172B3",
    "allgather": "#937860",
    "build": "#55A868",
    "index": "#CCB974",
    "search": "#4C72B0",
    "download": "#DD8452",
    "write": "#C44E52",
}


def rank_seconds(rank: dict, phase: str) -> float:
    """Seconds @p rank spent in @p phase, with downloads and writes taken out of the search."""
    if phase == "search":
        return (rank["search_ms"] - rank["download_ms"] - rank["write_ms"]) / 1000
    return rank.get(f"{phase}_ms", 0.0) / 1000


def human_bytes(value: float) -> str:
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if value < 1000:
            return f"{value:.1f} {unit}"
        value /= 1000
    return f"{value:.1f} PB"


def plot_report(report: dict, output: Path) -> None:
    ranks = sorted(report["ranks_detail"], key=lambda r: r["rank"])
    phases = [p for p in PHASES if any(f"{p}_ms" in r for r in ranks)]
    fig, axes = pu.setup_figure(figsize=(12, 10), nrows=2)
    labels = [str(r["rank"]) for r in ranks]
    bottom = np.zeros(len(ranks))
    for phase in phases:
        heights = np.array([rank_seconds(r, phase) for r in ranks])
        axes[0].bar(
            labels,
            heights,
            bottom=bottom,
            color=COLORS[phase],
            label=LABELS.get(phase, phase),
            edgecolor="black",
            linewidth=0.3,
        )
        bottom += heights
    axes[0].set_ylim(0, bottom.max() * 1.3)
    axes[0].set_ylabel(
        pu.paper_text("Time [s]", bold=True), fontsize=pu.AXIS_LABEL_FONT_SIZE
    )
    axes[0].set_xlabel(
        pu.paper_text("Rank", bold=True), fontsize=pu.AXIS_LABEL_FONT_SIZE
    )
    threshold = report["minimum_matches"]
    axes[0].set_title(
        pu.paper_text(
            f"{report['genomes']} genomes on {report['ranks']} GPUs ({report['nodes']} node"
            f"{'s' if report['nodes'] > 1 else ''} × {report['ranks'] // report['nodes']}), "
            + (f"at least {threshold} index matches" if threshold else "every pair"),
            bold=True,
        ),
        fontsize=pu.TITLE_FONT_SIZE,
    )
    axes[0].legend(fontsize=pu.LEGEND_FONT_SIZE - 4, ncol=8, loc="upper center")
    axes[0].grid(True, axis="y", ls="--", alpha=pu.GRID_ALPHA)

    wall = [max(rank_seconds(r, p) for r in ranks) for p in phases]
    x = np.arange(len(phases))
    bars = axes[1].bar(
        x,
        wall,
        0.6,
        color=[COLORS[p] for p in phases],
        edgecolor="black",
        linewidth=pu.BAR_EDGE_WIDTH,
    )
    axes[1].bar_label(bars, fmt="%.1f s", fontsize=pu.BAR_FONT_SIZE, padding=3)
    axes[1].set_ylabel(
        pu.paper_text("Wall time, slowest rank [s]", bold=True),
        fontsize=pu.AXIS_LABEL_FONT_SIZE,
    )
    axes[1].set_ylim(0, max(wall) * 1.2)
    axes[1].set_xticks(x, [LABELS.get(p, p) for p in phases])
    axes[1].tick_params(labelsize=pu.TICK_LABEL_FONT_SIZE)
    axes[1].grid(True, axis="y", ls="--", alpha=pu.GRID_ALPHA)
    fig.tight_layout()
    pu.save_figure(fig, output)


def main(
    reports: Annotated[
        list[Path],
        typer.Argument(exists=True, dir_okay=False, help="End-to-end JSON reports"),
    ],
    output_dir: Annotated[
        Path | None, typer.Option("--output-dir", "-o", help="Default: build/")
    ] = None,
) -> None:
    output_dir = pu.resolve_output_dir(output_dir, __file__)
    rows = []
    for path in reports:
        report = json.loads(path.read_text())
        plot_report(report, output_dir / f"end_to_end_{path.stem}.pdf")
        wall_s = report["wall_ms"] / 1000
        rows.append(
            [
                path.stem,
                report["genomes"],
                report["nodes"],
                report["ranks"],
                report["minimum_matches"],
                f"{wall_s:.1f}",
                f"{report['nodes'] * wall_s / 3600:.4f}",
                report["pairs_total"],
                human_bytes(report["bytes_written_total"]),
                human_bytes(report["input_bytes"]),
            ]
        )
    header = [
        "report",
        "genomes",
        "nodes",
        "GPUs",
        "min",
        "wall s",
        "node-h",
        "pairs",
        "written",
        "input",
    ]
    widths = [max(len(str(c)) for c in col) for col in zip(header, *rows)]
    for row in [header, *rows]:
        typer.echo("  ".join(str(c).rjust(w) for c, w in zip(row, widths)))


if __name__ == "__main__":
    typer.run(main)
