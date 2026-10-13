#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = ["matplotlib", "numpy", "pandas", "typer"]
# ///
"""Strong and weak scaling of cuddl-distributed-all-to-all-benchmark across nodes.

Takes the reports of scripts/sketch-scaling.sh. Strong-scaling runs are those with
the most genomes; weak-scaling runs keep genomes per node equal to the
smallest 1-node run. Reports whose name ends in -r0 are skipped: a first run
reads from Lustre only what earlier runs left uncached, so it measures neither
a cold nor a cached pipeline. The remaining repeats read from the page cache
and are summarised by their median. Genomes differ in size, so weak-scaling
runs differ slightly in bytes per node; ideal times and efficiency therefore
use compressed input bytes per second per node, relative to the fewest nodes.
Writes stacked phase wall times per node count, with the ideal sketch time as
a dashed tick on each bar, and the sketch efficiency, and prints a table.
SimdSketch reports of scripts/simdsketch-nodes.sh among the inputs add a hatched
SimdSketch bar next to each cuDDL bar (sketch, gather, compare and write) and
their rows in the table.
"""

import json
import statistics
from pathlib import Path
from typing import Annotated

import numpy as np
import plot_utils as pu
import typer
from plot_distributed_end_to_end import COLORS, LABELS, PHASES, rank_seconds

BAR_WIDTH = 0.4
SIMDSKETCH_PHASES = [
    ("sketch_s", "sketch"),
    ("gather_s", "allgather"),
    ("compare_s", "compare"),
    ("write_s", "write"),
]
COMPARE_COLOR = "#DA8BC3"


def sketch_s(report: dict) -> float:
    return report["phases_wall_ms"]["sketch"] / 1000


def group(reports: list[dict]) -> dict[int, list[dict]]:
    points: dict[int, list[dict]] = {}
    for report in reports:
        points.setdefault(report["nodes"], []).append(report)
    return dict(sorted(points.items()))


def median_report(points: dict[int, list[dict]]) -> dict[int, list[dict]]:
    """One report per node count holding the median of each SimdSketch timing over repeats."""
    keys = ["sketch_s", "gather_s", "compare_s", "write_s", "wall_s"]
    return {
        n: [{**runs[0], **{k: statistics.median(r[k] for r in runs) for k in keys}}]
        for n, runs in points.items()
    }


def median_sketch_s(runs: list[dict]) -> float:
    return statistics.median(sketch_s(r) for r in runs)


def efficiency(points: dict[int, list[dict]]) -> list[float]:
    """Input bytes per second per node, relative to the fewest nodes."""
    rates = [
        runs[0]["input_bytes"] / median_sketch_s(runs) / n for n, runs in points.items()
    ]
    return [rate / rates[0] for rate in rates]


def phase_panel(
    ax,
    points: dict[int, list[dict]],
    baseline: dict[int, list[dict]],
    strong: bool,
    title: str,
    node_axis: str,
) -> None:
    x = np.arange(len(points))
    ours = x - BAR_WIDTH / 2 if baseline else x
    bottom = np.zeros(len(points))
    for phase in PHASES:
        heights = np.array(
            [
                statistics.median(
                    max(rank_seconds(rank, phase) for rank in r["ranks_detail"]) for r in runs
                )
                for runs in points.values()
            ]
        )
        ax.bar(
            ours,
            heights,
            BAR_WIDTH,
            bottom=bottom,
            color=COLORS[phase],
            label=LABELS.get(phase, phase),
            edgecolor="black",
            linewidth=pu.BAR_EDGE_WIDTH,
        )
        bottom += heights
    ideal = [
        median_sketch_s(runs) * e
        for runs, e in zip(points.values(), efficiency(points))
    ]
    ax.hlines(
        ideal,
        ours - BAR_WIDTH / 2,
        ours + BAR_WIDTH / 2,
        colors="black",
        linestyles="--",
        lw=pu.REFERENCE_LINE_WIDTH,
        label="ideal sketch",
    )
    totals = list(bottom)
    positions = list(ours)
    if baseline:
        theirs = x + BAR_WIDTH / 2
        stack = np.zeros(len(points))
        for key, phase in SIMDSKETCH_PHASES:
            heights = np.array(
                [baseline[n][0][key] if n in baseline else 0.0 for n in points]
            )
            ax.bar(
                theirs,
                heights,
                BAR_WIDTH,
                bottom=stack,
                color=COLORS.get(phase, COMPARE_COLOR),
                label=phase if phase not in COLORS else None,
                hatch="//",
                edgecolor="black",
                linewidth=pu.BAR_EDGE_WIDTH,
            )
            stack += heights
        ax.bar(
            theirs[:1],
            [0.0],
            BAR_WIDTH,
            facecolor="white",
            hatch="//",
            edgecolor="black",
            linewidth=pu.BAR_EDGE_WIDTH,
            label="SimdSketch",
        )
        totals += [s for n, s in zip(points, stack) if n in baseline]
        positions += [t for n, t in zip(points, theirs) if n in baseline]
    for position, total in zip(positions, totals):
        ax.text(
            position,
            total,
            f"{total:.0f} s" if total >= 10 else f"{total:.1f} s",
            ha="center",
            va="bottom",
            fontsize=pu.BAR_FONT_SIZE,
        )
    ticks = [
        str(n) if strong else f"{n}\n({runs[0]['genomes'] // 1000}k genomes)"
        for n, runs in points.items()
    ]
    ax.set_xticks(x, ticks)
    ax.set_ylim(0, max(totals) * 1.15)
    pu.format_axis(ax, node_axis, "Wall time, slowest rank [s]", title, xscale=None)
    ax.grid(False, axis="x")
    ax.tick_params(labelsize=pu.TICK_LABEL_FONT_SIZE)


def main(
    reports: Annotated[
        list[Path],
        typer.Argument(
            exists=True, dir_okay=False, help="Reports of scripts/sketch-scaling.sh"
        ),
    ],
    output_dir: Annotated[
        Path | None, typer.Option("--output-dir", "-o", help="Default: build/")
    ] = None,
) -> None:
    output_dir = pu.resolve_output_dir(output_dir, __file__)
    loaded = [
        json.loads(path.read_text())
        for path in reports
        if not path.stem.endswith("-r0")
    ]
    runs = [r for r in loaded if r.get("tool") != "simdsketch"]
    simdsketch = [r for r in loaded if r.get("tool") == "simdsketch"]
    thresholds = {r["minimum_matches"] for r in runs}
    if len(thresholds) > 1:
        raise typer.BadParameter(f"reports mix minimum-match thresholds {sorted(thresholds)}")
    minimum = thresholds.pop()
    output = "every pair" if minimum == 0 else f"at least {minimum} matches"
    most = max(r["genomes"] for r in runs)
    per_node = min(r["genomes"] for r in runs if r["nodes"] == 1)
    series = {
        "Strong scaling": (True, group([r for r in runs if r["genomes"] == most])),
        "Weak scaling": (
            False,
            group([r for r in runs if r["genomes"] == per_node * r["nodes"]]),
        ),
    }
    series = {name: s for name, s in series.items() if len(s[1]) > 1}
    baseline = {
        "Strong scaling": median_report(
            group([r for r in simdsketch if r["genomes"] == most])
        ),
        "Weak scaling": median_report(
            group([r for r in simdsketch if r["genomes"] == per_node * r["nodes"]])
        ),
    }
    gpus = runs[0]["ranks"] // runs[0]["nodes"]
    node_axis = f"Nodes ({gpus} GPU{'s' if gpus > 1 else ''} each)"

    fig, axes = pu.setup_figure(figsize=(12, 6.5 * len(series)), nrows=len(series))
    axes = list(np.atleast_1d(axes))
    for ax, (name, (strong, points)) in zip(axes, series.items()):
        genomes = (
            f"{most // 1000}k genomes"
            if strong
            else f"{per_node // 1000}k genomes per node"
        )
        phase_panel(ax, points, baseline[name], strong, f"{name}, {genomes}, {output}", node_axis)
    pu.add_top_legend(fig, axes, ncol=4)
    pu.save_figure(fig, output_dir / "sketch_scaling_phases.pdf")

    fig, ax = pu.setup_figure(figsize=(8, 5.5))
    for name, (strong, points) in series.items():
        ax.plot(
            list(points),
            efficiency(points),
            "-" if strong else ":",
            marker="o",
            lw=pu.LINE_WIDTH,
            ms=pu.MARKER_SIZE,
            label=name,
        )
    ax.axhline(1.0, color="black", lw=pu.REFERENCE_LINE_WIDTH, ls="--")
    ax.set_xticks(sorted({n for _, points in series.values() for n in points}))
    ax.set_ylim(0, None)
    pu.format_axis(ax, node_axis, "Sketch efficiency", None, xscale=None)
    ax.tick_params(labelsize=pu.TICK_LABEL_FONT_SIZE)
    pu.add_top_legend(fig, ax, ncol=2)
    pu.save_figure(fig, output_dir / "sketch_scaling_efficiency.pdf")

    if simdsketch:
        header = [
            "scaling",
            "nodes",
            "genomes",
            "threads",
            "sketch s",
            "gather s",
            "compare s",
            "write s",
            "wall s",
            "cuDDL speedup",
        ]
        rows = []
        for name, (_, points) in series.items():
            for n, rs in baseline[name].items():
                r = rs[0]
                ours = (
                    statistics.median(x["wall_ms"] for x in points[n]) / 1000
                    if n in points
                    else None
                )
                rows.append(
                    [
                        name.split()[0].lower(),
                        n,
                        r["genomes"],
                        r["threads"],
                        f"{r['sketch_s']:.1f}",
                        f"{r['gather_s']:.1f}",
                        f"{r['compare_s']:.1f}",
                        f"{r['write_s']:.1f}",
                        f"{r['wall_s']:.1f}",
                        f"{r['wall_s'] / ours:.1f}x" if ours else "-",
                    ]
                )
        widths = [max(len(str(c)) for c in col) for col in zip(header, *rows)]
        typer.echo("\nSimdSketch")
        for row in [header, *rows]:
            typer.echo("  ".join(str(c).rjust(w) for c, w in zip(row, widths)))

    header = [
        "scaling",
        "nodes",
        "genomes",
        "input",
        "sketch s (min-max)",
        "GB/s/node",
        "efficiency",
        "wall s",
    ]
    rows = []
    for name, (strong, points) in series.items():
        for (n, runs), eff in zip(points.items(), efficiency(points)):
            sketches = [sketch_s(r) for r in runs]
            median = statistics.median(sketches)
            rows.append(
                [
                    name.split()[0].lower(),
                    n,
                    runs[0]["genomes"],
                    f"{runs[0]['input_bytes'] / 1e9:.1f} GB",
                    f"{median:.2f} ({min(sketches):.2f}-{max(sketches):.2f})",
                    f"{runs[0]['input_bytes'] / 1e9 / median / n:.2f}",
                    f"{eff:.2f}",
                    f"{statistics.median(r['wall_ms'] for r in runs) / 1000:.2f}",
                ]
            )
    widths = [max(len(str(c)) for c in col) for col in zip(header, *rows)]
    for row in [header, *rows]:
        typer.echo("  ".join(str(c).rjust(w) for c, w in zip(row, widths)))


if __name__ == "__main__":
    typer.run(main)
