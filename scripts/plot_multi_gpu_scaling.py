#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = ["matplotlib", "pandas", "typer"]
# ///
"""Figures for cuddl-multi-gpu-all-to-all-benchmark JSON reports.

Strong-scaling reports give speedup over one GPU against GPU count within one
node, against full-node count per GPUs-per-node layout, and speedup and
parallel efficiency against total GPU count. Weak-scaling reports give
efficiency t(1 GPU) / t(P GPUs) and pairs per second per GPU. Each run is
compared with the 1-GPU run of its own report; bars compare results kept on
the GPU (device) with results downloaded per tile (host), error bars span the
5th to 95th percentile of the samples and markers show ideal scaling. When reports repeat a point,
the first one wins.

Reports from newer benchmark builds also give, for the largest rank count of
each report: a per-rank Gantt chart of tile completion times with the closing
barrier wait, max/mean rank time against rank count, per-rank setup and search
phases, communication costs, a threshold sweep (pairs, bytes and output time
against minimum index matches) and pairs per second per GPU against a
single-GPU line and an optional measured ALU ceiling. Every figure family also
prints its key numbers.
"""

import json
from pathlib import Path
from typing import Annotated

import matplotlib.pyplot as plt
import numpy as np
import plot_utils as pu
import typer

CONSUMERS = {"device": "Results on GPU", "host": "Downloaded to host"}
COLORS = {"device": "#4C72B0", "host": "#DD8452"}
PHASE_COLORS = {
    "load": "#8172B3",
    "broadcast": "#937860",
    "upload": "#64B5CD",
    "build": "#55A868",
    "index": "#CCB974",
    "search": "#4C72B0",
    "download": "#DD8452",
    "write": "#C44E52",
    "wait": "#BBBBBB",
}

GPU_AXIS = "GPUs (nodes × GPUs per node)"


def layout(ranks: int, per_node: int) -> str:
    """'nodes×GPUs per node' of a run; mpirun fills nodes slot by slot."""
    return f"{-(-ranks // per_node)}×{min(ranks, per_node)}"


def gpu_tick(ranks: int, per_node: int) -> str:
    return f"{ranks}\n({layout(ranks, per_node)})"


def gpus_text(ranks: int, per_node: int) -> str:
    nodes = -(-ranks // per_node)
    return f"{ranks} GPUs on {nodes} node{'s' if nodes > 1 else ''} × {min(ranks, per_node)}"


class Point:
    """One rank count of one report, with that report's 1-GPU run."""

    def __init__(self, run: dict, baseline: dict, report: dict):
        self.run = run
        self.baseline = baseline
        self.ranks = run["ranks"]
        self.ranks_per_node = report["ranks_per_node"]
        self.mode = report.get("mode", "strong")
        # Reports from before weak scaling hold one pair count for every run.
        self.pairs = run.get("pairs", report["pairs"])
        # mpirun fills nodes slot by slot, as the benchmark assumes.
        self.nodes = -(-self.ranks // self.ranks_per_node)


def load_reports(paths: list[Path]) -> list[dict]:
    reports = [json.loads(path.read_text()) for path in paths]
    for path, report in zip(paths, reports):
        if not any(r["ranks"] == 1 for r in report["runs"]):
            raise typer.BadParameter(f"{path} has no 1-GPU run")
    return reports


def points_of(reports: list[dict], mode: str) -> list[Point]:
    points = []
    for report in reports:
        if report.get("mode", "strong") != mode:
            continue
        baseline = next(r for r in report["runs"] if r["ranks"] == 1)
        points += [Point(r, baseline, report) for r in report["runs"]]
    return points


def ratio_chart(
    ax: plt.Axes,
    selected: list[tuple[int, Point]],
    ideal: list[float],
    value,
    ylabel: str,
    fmt: str,
) -> None:
    """Grouped device/host bars of value(baseline_ms, run_ms, point) with p5/p95 whiskers."""
    x = np.arange(len(selected))
    consumers = [c for c in CONSUMERS if all(c in p.run for _, p in selected)]
    width = 0.8 / len(consumers)
    top = max(ideal)
    for i, consumer in enumerate(consumers):
        rows = [
            (p.baseline[consumer]["median_ms"], p.run[consumer], p) for _, p in selected
        ]
        mid = np.array([value(b, r["median_ms"], p) for b, r, p in rows])
        low = mid - np.array([value(b, r["p95_ms"], p) for b, r, p in rows])
        high = np.array([value(b, r["p5_ms"], p) for b, r, p in rows]) - mid
        top = max(top, float((mid + high).max()))
        bars = ax.bar(
            x + (i - (len(consumers) - 1) / 2) * width,
            mid,
            width,
            yerr=[low, high],
            capsize=3,
            color=COLORS[consumer],
            edgecolor="black",
            linewidth=pu.BAR_EDGE_WIDTH,
            alpha=pu.SCALING_BAR_ALPHA,
            label=CONSUMERS[consumer],
        )
        ax.bar_label(bars, fmt=fmt, fontsize=pu.BAR_FONT_SIZE, padding=4)
    ax.scatter(x, ideal, marker="_", s=2000, color="black", zorder=3, label="Ideal")
    ax.set_xticks(x, [gpu_tick(p.ranks, p.ranks_per_node) for _, p in selected])
    ax.tick_params(labelsize=pu.TICK_LABEL_FONT_SIZE)
    ax.set_ylabel(pu.paper_text(ylabel, bold=True), fontsize=pu.AXIS_LABEL_FONT_SIZE)
    ax.grid(True, axis="y", ls="--", alpha=pu.GRID_ALPHA)
    ax.set_ylim(0, top * 1.12)


def speedup(base_ms: float, ms: float, _: Point) -> float:
    return base_ms / ms


def strong_efficiency(base_ms: float, ms: float, p: Point) -> float:
    return base_ms / ms / p.ranks


def speedup_chart(
    selected: list[tuple[int, Point]],
    ideal: list[float],
    xlabel: str,
    title: str,
    output: Path,
) -> None:
    fig, ax = pu.setup_figure(figsize=(10, 6))
    ratio_chart(ax, selected, ideal, speedup, "Speedup vs. 1 GPU", "%.1f×")
    pu.format_axis(ax, xlabel, "Speedup vs. 1 GPU", title, xscale=None)
    pu.add_top_legend(fig, ax, ncol=3)
    pu.save_figure(fig, output)


def two_panel_chart(
    selected: list[tuple[int, Point]],
    panels: list[tuple[list[float], object, str, str]],
    xlabel: str,
    title: str,
    output: Path,
) -> None:
    fig, axes = pu.setup_figure(figsize=(10, 9), nrows=len(panels))
    for ax, (ideal, value, ylabel, fmt) in zip(axes, panels):
        ratio_chart(ax, selected, ideal, value, ylabel, fmt)
        ax.set_xlabel(
            pu.paper_text(xlabel, bold=True), fontsize=pu.AXIS_LABEL_FONT_SIZE
        )
    axes[0].set_title(pu.paper_text(title, bold=True), fontsize=pu.TITLE_FONT_SIZE)
    pu.add_top_legend(fig, axes[0], ncol=3)
    pu.save_figure(fig, output)


def first_by(points: list[Point], key) -> list[tuple[int, Point]]:
    chosen: dict[int, Point] = {}
    for p in points:
        chosen.setdefault(key(p), p)
    return sorted(chosen.items())


def print_table(title: str, header: list[str], rows: list[list]) -> None:
    typer.secho(title, bold=True)
    widths = [max(len(str(c)) for c in col) for col in zip(header, *rows)]
    for row in [header, *rows]:
        typer.echo("  " + "  ".join(str(c).rjust(w) for c, w in zip(row, widths)))


def strong_figures(points: list[Point], output_dir: Path) -> None:
    if not points:
        return
    gpus = first_by([p for p in points if p.nodes == 1], lambda p: p.ranks)
    speedup_chart(
        gpus,
        [float(ranks) for ranks, _ in gpus],
        GPU_AXIS,
        "All-to-all scaling across GPUs",
        output_dir / "multi_gpu_scaling_gpus.pdf",
    )
    for per_node in sorted({p.ranks_per_node for p in points}):
        full = [
            p
            for p in points
            if p.ranks_per_node == per_node and p.ranks == p.nodes * per_node
        ]
        nodes = first_by(full, lambda p: p.nodes)
        if len(nodes) < 2:
            continue
        speedup_chart(
            nodes,
            [float(n * per_node) for n, _ in nodes],
            GPU_AXIS,
            "All-to-all scaling across nodes",
            output_dir / f"multi_gpu_scaling_nodes_{per_node}_per_node.pdf",
        )
    total = first_by(points, lambda p: p.ranks)
    two_panel_chart(
        total,
        [
            ([float(r) for r, _ in total], speedup, "Speedup vs. 1 GPU", "%.1f×"),
            ([1.0] * len(total), strong_efficiency, "Parallel efficiency", "%.2f"),
        ],
        GPU_AXIS,
        "Strong scaling",
        output_dir / "multi_gpu_strong_scaling.pdf",
    )
    print_table(
        "Strong scaling (device consumer)",
        ["GPUs", "nodes", "median ms", "speedup", "efficiency"],
        [
            [
                r,
                p.nodes,
                f"{p.run['device']['median_ms']:.2f}",
                f"{speedup(p.baseline['device']['median_ms'], p.run['device']['median_ms'], p):.2f}",
                f"{strong_efficiency(p.baseline['device']['median_ms'], p.run['device']['median_ms'], p):.2f}",
            ]
            for r, p in total
            if "device" in p.run
        ],
    )


def weak_figures(points: list[Point], output_dir: Path) -> None:
    if not points:
        return
    total = first_by(points, lambda p: p.ranks)

    def per_gpu_rate(base_ms: float, ms: float, p: Point) -> float:
        return p.pairs / (ms / 1000) / p.ranks / 1e9

    base_rate = [
        p.baseline["pairs"] / (p.baseline["device"]["median_ms"] / 1000) / 1e9
        for _, p in total
    ]
    two_panel_chart(
        total,
        [
            ([1.0] * len(total), speedup, "Weak-scaling efficiency", "%.2f"),
            (base_rate, per_gpu_rate, "Gpairs/s per GPU", "%.2f"),
        ],
        f"{GPU_AXIS}; references grow as sqrt(GPUs)",
        "Weak scaling",
        output_dir / "multi_gpu_weak_scaling.pdf",
    )
    print_table(
        "Weak scaling (device consumer)",
        ["GPUs", "references", "pairs/GPU", "median ms", "efficiency"],
        [
            [
                r,
                p.run["references"],
                p.pairs // r,
                f"{p.run['device']['median_ms']:.2f}",
                f"{p.baseline['device']['median_ms'] / p.run['device']['median_ms']:.2f}",
            ]
            for r, p in total
            if "device" in p.run
        ],
    )


def tag_of(report: dict) -> str:
    return f"{report.get('mode', 'strong')}_{report['references']}"


def gantt_figures(reports: list[dict], output_dir: Path) -> None:
    for report in reports:
        runs = [r for r in report["runs"] if "rank_timeline" in r]
        if not runs:
            continue
        run = max(runs, key=lambda r: r["ranks"])
        timeline = sorted(run["rank_timeline"], key=lambda t: t["rank"])
        fig, ax = pu.setup_figure(figsize=(12, max(3, 0.45 * len(timeline) + 2)))
        shades = ["#4C72B0", "#8FAADC"]
        for row, rank in enumerate(timeline):
            for i, tile in enumerate(rank["tiles"]):
                ax.barh(
                    row,
                    tile["end_ms"] - tile["start_ms"],
                    left=tile["start_ms"],
                    color=shades[i % 2],
                    edgecolor="black",
                    linewidth=0.3,
                )
            ax.barh(
                row,
                rank["wait_ms"],
                left=rank["search_ms"],
                color="white",
                edgecolor="#888888",
                hatch="//",
                linewidth=0.3,
            )
        ax.set_yticks(
            range(len(timeline)),
            [f"{t['rank']} ({t['host'].split('.')[0]})" for t in timeline],
        )
        ax.invert_yaxis()
        balance = run["load_balance"]
        pu.format_axis(
            ax,
            "Time since start barrier [ms]",
            "Rank",
            f"{gpus_text(run['ranks'], report['ranks_per_node'])}: max/mean rank time {balance['max_rank_ms']:.2f}/"
            f"{balance['mean_rank_ms']:.2f} ms = {balance['ratio']:.3f}",
            xscale=None,
        )
        handles = [
            plt.Rectangle(
                (0, 0), 1, 1, color=shades[0], label="Tile (alternating shade)"
            ),
            plt.Rectangle(
                (0, 0),
                1,
                1,
                facecolor="white",
                edgecolor="#888888",
                hatch="//",
                label="Barrier wait",
            ),
        ]
        ax.legend(handles=handles, loc="lower right", fontsize=pu.LEGEND_FONT_SIZE - 4)
        pu.save_figure(
            fig, output_dir / f"multi_gpu_gantt_{tag_of(report)}_{run['ranks']}.pdf"
        )

        counts = [r["ranks"] for r in runs]
        ratios = [r["load_balance"]["ratio"] for r in runs]
        fig, ax = pu.setup_figure(figsize=(8, 5))
        bars = ax.bar(
            [gpu_tick(c, report["ranks_per_node"]) for c in counts],
            ratios,
            color="#4C72B0",
            edgecolor="black",
            linewidth=pu.BAR_EDGE_WIDTH,
        )
        ax.bar_label(bars, fmt="%.3f", fontsize=pu.BAR_FONT_SIZE, padding=3)
        ax.axhline(1, color="black", lw=pu.REFERENCE_LINE_WIDTH, ls="--")
        ax.set_ylim(0.9, max(ratios) * 1.05 + 0.02)
        pu.format_axis(
            ax, GPU_AXIS, "Max / mean rank time", "Load balance", xscale=None
        )
        ax.tick_params(labelsize=pu.TICK_LABEL_FONT_SIZE)
        pu.save_figure(fig, output_dir / f"multi_gpu_load_balance_{tag_of(report)}.pdf")
        print_table(
            f"Load balance ({tag_of(report)})",
            ["GPUs", "max ms", "mean ms", "ratio"],
            [
                [
                    r["ranks"],
                    f"{r['load_balance']['max_rank_ms']:.2f}",
                    f"{r['load_balance']['mean_rank_ms']:.2f}",
                    f"{r['load_balance']['ratio']:.3f}",
                ]
                for r in runs
            ],
        )


def stacked_ranks(ax: plt.Axes, rows: list[dict], phases: list[str]) -> None:
    labels = [str(r["rank"]) for r in rows]
    bottom = np.zeros(len(rows))
    for phase in phases:
        values = np.array([r.get(f"{phase}_ms", 0.0) for r in rows])
        if not values.any():
            continue
        ax.bar(
            labels,
            values,
            bottom=bottom,
            color=PHASE_COLORS[phase],
            label=phase,
            edgecolor="black",
            linewidth=0.3,
        )
        bottom += values
    ax.grid(True, axis="y", ls="--", alpha=pu.GRID_ALPHA)


def phase_figures(reports: list[dict], output_dir: Path) -> None:
    for report in reports:
        runs = [r for r in report["runs"] if "phases" in r]
        if "setup" not in report or not runs:
            continue
        run = max(runs, key=lambda r: r["ranks"])
        fig, axes = pu.setup_figure(figsize=(12, 9), nrows=2)
        setup = sorted(report["setup"], key=lambda s: s["rank"])
        # Reports from before every rank loaded the file itself also hold a broadcast.
        stacked_ranks(axes[0], setup, ["load", "broadcast", "upload", "build", "index"])
        axes[0].set_title(
            pu.paper_text("Setup per rank", bold=True), fontsize=pu.TITLE_FONT_SIZE
        )
        axes[0].set_ylabel(
            pu.paper_text("Time [ms]", bold=True), fontsize=pu.AXIS_LABEL_FONT_SIZE
        )
        axes[0].set_ylim(0, axes[0].get_ylim()[1] * 1.2)
        axes[0].legend(fontsize=pu.LEGEND_FONT_SIZE - 4, ncol=5, loc="upper center")
        # Each rank's search once with results left on the GPU (when its last tile finished)
        # and once with every tile's results copied to the host, the copies overlapping the
        # search of later tiles.
        on_gpu = {t["rank"]: t["search_ms"] for t in run["rank_timeline"]}
        to_host = {s["rank"]: s["search_ms"] for s in run["phases"]}
        ranks = sorted(to_host)
        x = np.arange(len(ranks))
        width = 0.4
        axes[1].bar(
            x - width / 2,
            [on_gpu.get(r, 0.0) for r in ranks],
            width,
            color=PHASE_COLORS["search"],
            label="results kept on GPU",
            edgecolor="black",
            linewidth=0.3,
        )
        axes[1].bar(
            x + width / 2,
            [to_host[r] for r in ranks],
            width,
            color=PHASE_COLORS["download"],
            label="results copied to host",
            edgecolor="black",
            linewidth=0.3,
        )
        axes[1].set_xticks(x, [str(r) for r in ranks])
        axes[1].grid(True, axis="y", ls="--", alpha=pu.GRID_ALPHA)
        axes[1].set_title(
            pu.paper_text(
                f"Search per rank, {gpus_text(run['ranks'], report['ranks_per_node'])}",
                bold=True,
            ),
            fontsize=pu.TITLE_FONT_SIZE,
        )
        axes[1].set_ylabel(
            pu.paper_text("Time [ms]", bold=True), fontsize=pu.AXIS_LABEL_FONT_SIZE
        )
        axes[1].set_xlabel(
            pu.paper_text("Rank", bold=True), fontsize=pu.AXIS_LABEL_FONT_SIZE
        )
        axes[1].set_ylim(0, axes[1].get_ylim()[1] * 1.2)
        axes[1].legend(fontsize=pu.LEGEND_FONT_SIZE - 4, ncol=2, loc="upper center")
        fig.tight_layout()
        pu.save_figure(fig, output_dir / f"multi_gpu_phases_{tag_of(report)}.pdf")
        print_table(
            f"Setup phases, max over ranks ({tag_of(report)})",
            ["load ms", "upload ms", "build ms", "index ms"],
            [
                [
                    f"{max(s.get(f'{k}_ms', 0.0) for s in setup):.1f}"
                    for k in ["load", "upload", "build", "index"]
                ]
            ],
        )


def communication_figures(reports: list[dict], output_dir: Path) -> None:
    for report in reports:
        entries = [c for c in report.get("communication", []) if c["ranks"] > 1]
        if not entries:
            continue
        fig, axes = pu.setup_figure(figsize=(10, 9), nrows=2)
        labels = [gpu_tick(c["ranks"], report["ranks_per_node"]) for c in entries]
        times = [c["broadcast"]["median_ms"] for c in entries]
        bars = axes[0].bar(
            labels,
            times,
            color="#937860",
            edgecolor="black",
            linewidth=pu.BAR_EDGE_WIDTH,
        )
        axes[0].bar_label(
            bars,
            labels=[f"{c['broadcast_gbps']:.1f} GB/s" for c in entries],
            fontsize=pu.BAR_FONT_SIZE,
            padding=3,
        )
        axes[0].set_ylim(0, max(times) * 1.2)
        axes[0].set_ylabel(
            pu.paper_text("Database broadcast [ms]", bold=True),
            fontsize=pu.AXIS_LABEL_FONT_SIZE,
        )
        axes[0].set_title(
            pu.paper_text(
                f"Communication ({entries[0]['broadcast_bytes'] / 1e6:.0f} MB database)",
                bold=True,
            ),
            fontsize=pu.TITLE_FONT_SIZE,
        )
        barrier = [c["barrier_us"]["median_us"] for c in entries]
        bars = axes[1].bar(
            labels,
            barrier,
            color="#BBBBBB",
            edgecolor="black",
            linewidth=pu.BAR_EDGE_WIDTH,
        )
        axes[1].bar_label(bars, fmt="%.1f", fontsize=pu.BAR_FONT_SIZE, padding=3)
        axes[1].set_ylim(0, max(barrier) * 1.2)
        axes[1].set_ylabel(
            pu.paper_text("MPI barrier [us]", bold=True),
            fontsize=pu.AXIS_LABEL_FONT_SIZE,
        )
        for ax in axes:
            ax.set_xlabel(
                pu.paper_text(GPU_AXIS, bold=True), fontsize=pu.AXIS_LABEL_FONT_SIZE
            )
            ax.tick_params(labelsize=pu.TICK_LABEL_FONT_SIZE)
        for ax in axes:
            ax.grid(True, axis="y", ls="--", alpha=pu.GRID_ALPHA)
        fig.tight_layout()
        pu.save_figure(
            fig, output_dir / f"multi_gpu_communication_{tag_of(report)}.pdf"
        )
        print_table(
            f"Communication ({tag_of(report)})",
            ["GPUs", "nodes", "broadcast ms", "GB/s", "barrier µs"],
            [
                [
                    c["ranks"],
                    c["nodes"],
                    f"{c['broadcast']['median_ms']:.2f}",
                    f"{c['broadcast_gbps']:.1f}",
                    f"{c['barrier_us']['median_us']:.1f}",
                ]
                for c in entries
            ],
        )


def human_bytes(value: float) -> str:
    for unit in ["B", "KB", "MB", "GB", "TB", "PB"]:
        if value < 1000:
            return f"{value:.1f} {unit}"
        value /= 1000
    return f"{value:.1f} EB"


def sweep_figures(reports: list[dict], output_dir: Path) -> None:
    for report in reports:
        sweep = report.get("threshold_sweep")
        if not sweep:
            continue
        labels = [str(s["minimum_matches"]) if s["indexed"] else "all" for s in sweep]
        fig, axes = pu.setup_figure(figsize=(10, 9), nrows=2)
        x = np.arange(len(sweep))
        pairs = [max(s["pairs_emitted"], 1) for s in sweep]
        bars = axes[0].bar(
            x, pairs, color="#4C72B0", edgecolor="black", linewidth=pu.BAR_EDGE_WIDTH
        )
        axes[0].bar_label(
            bars,
            labels=[human_bytes(s["bytes_written"]) for s in sweep],
            fontsize=pu.BAR_FONT_SIZE,
            padding=3,
        )
        axes[0].set_yscale("log")
        axes[0].set_ylim(min(pairs) / 2, max(pairs) * 8)
        axes[0].set_ylabel(
            pu.paper_text("Pairs emitted", bold=True), fontsize=pu.AXIS_LABEL_FONT_SIZE
        )
        axes[0].set_title(
            pu.paper_text(
                f"Threshold sweep, {report['references']} genomes, {gpus_text(sweep[0]['ranks'], report['ranks_per_node'])}",
                bold=True,
            ),
            fontsize=pu.TITLE_FONT_SIZE,
        )
        bottom = np.zeros(len(sweep))
        for key, label, color in [
            ("search", "search (GPU)", PHASE_COLORS["search"]),
            ("download_ms", "download", PHASE_COLORS["download"]),
            ("write_ms", "write + flush", PHASE_COLORS["write"]),
        ]:
            values = np.array(
                [s["search"]["median_ms"] if key == "search" else s[key] for s in sweep]
            )
            axes[1].bar(
                x,
                values,
                bottom=bottom,
                color=color,
                label=label,
                edgecolor="black",
                linewidth=0.3,
            )
            bottom += values
        axes[1].set_ylim(0, bottom.max() * 1.15)
        axes[1].set_ylabel(
            pu.paper_text("Time, slowest rank [ms]", bold=True),
            fontsize=pu.AXIS_LABEL_FONT_SIZE,
        )
        for ax in axes:
            ax.set_xlabel(
                pu.paper_text("Minimum index matches", bold=True),
                fontsize=pu.AXIS_LABEL_FONT_SIZE,
            )
            ax.set_xticks(x, labels)
            ax.tick_params(labelsize=pu.TICK_LABEL_FONT_SIZE)
        axes[1].legend(fontsize=pu.LEGEND_FONT_SIZE - 4)
        for ax in axes:
            ax.grid(True, axis="y", ls="--", alpha=pu.GRID_ALPHA)
        fig.tight_layout()
        pu.save_figure(
            fig, output_dir / f"multi_gpu_threshold_sweep_{tag_of(report)}.pdf"
        )
        all_pairs = report["references"] * (report["references"] - 1) // 2
        print_table(
            f"Threshold sweep ({tag_of(report)})",
            [
                "min matches",
                "pairs",
                "fraction",
                "bytes",
                "search ms",
                "download ms",
                "write ms",
            ],
            [
                [
                    label,
                    s["pairs_emitted"],
                    f"{s['pairs_emitted'] / all_pairs:.2e}",
                    human_bytes(s["bytes_written"]),
                    f"{s['search']['median_ms']:.1f}",
                    f"{s['download_ms']:.1f}",
                    f"{s['write_ms']:.1f}",
                ]
                for label, s in zip(labels, sweep)
            ],
        )


def efficiency_figure(
    points: list[Point], output_dir: Path, roofline: float | None
) -> None:
    total = [
        (r, p) for r, p in first_by(points, lambda p: p.ranks) if "device" in p.run
    ]
    if not total:
        return
    rates = [
        p.pairs / (p.run["device"]["median_ms"] / 1000) / r / 1e9 for r, p in total
    ]
    single = next(p for r, p in total if r == 1)
    single_rate = single.pairs / (single.run["device"]["median_ms"] / 1000) / 1e9
    fig, ax = pu.setup_figure(figsize=(10, 6))
    bars = ax.bar(
        [gpu_tick(r, p.ranks_per_node) for r, p in total],
        rates,
        color="#4C72B0",
        edgecolor="black",
        linewidth=pu.BAR_EDGE_WIDTH,
    )
    ax.bar_label(bars, fmt="%.2f", fontsize=pu.BAR_FONT_SIZE, padding=3)
    ax.axhline(
        single_rate,
        color="black",
        ls="--",
        lw=pu.REFERENCE_LINE_WIDTH,
        label=f"1 GPU alone ({single_rate:.2f})",
    )
    top = max(max(rates), single_rate)
    if roofline:
        ax.axhline(
            roofline / 1e9,
            color="#C44E52",
            ls="-",
            lw=pu.REFERENCE_LINE_WIDTH,
            label=f"Measured ALU ceiling ({roofline / 1e9:.2f})",
        )
        top = max(top, roofline / 1e9)
    ax.set_ylim(0, top * 1.15)
    pu.format_axis(
        ax,
        GPU_AXIS,
        "Gpairs/s per GPU",
        "Per-GPU throughput, results on GPU",
        xscale=None,
    )
    ax.tick_params(labelsize=pu.TICK_LABEL_FONT_SIZE)
    pu.add_top_legend(fig, ax, ncol=2)
    pu.save_figure(fig, output_dir / "multi_gpu_per_gpu_throughput.pdf")
    print_table(
        "Per-GPU throughput (device consumer)",
        ["GPUs", "Gpairs/s/GPU", "vs 1 GPU"] + (["vs bound"] if roofline else []),
        [
            [r, f"{rate:.2f}", f"{rate / single_rate:.2f}"]
            + ([f"{rate / (roofline / 1e9):.2f}"] if roofline else [])
            for (r, _), rate in zip(total, rates)
        ],
    )


def main(
    reports: Annotated[
        list[Path],
        typer.Argument(exists=True, dir_okay=False, help="Benchmark JSON reports"),
    ],
    output_dir: Annotated[
        Path | None, typer.Option("--output-dir", "-o", help="Default: build/")
    ] = None,
    roofline_pairs_per_second: Annotated[
        float | None,
        typer.Option(
            help="Per-GPU ceiling to draw, in pairs/s: the alu-power-ceiling benchmark's ALU warp "
            "instructions per second divided by the search's ALU warp instructions per pair"
        ),
    ] = None,
) -> None:
    loaded = load_reports(reports)
    output_dir = pu.resolve_output_dir(output_dir, __file__)
    strong = points_of(loaded, "strong")
    strong_figures(strong, output_dir)
    weak_figures(points_of(loaded, "weak"), output_dir)
    gantt_figures(loaded, output_dir)
    phase_figures(loaded, output_dir)
    communication_figures(loaded, output_dir)
    sweep_figures(loaded, output_dir)
    efficiency_figure(strong, output_dir, roofline_pairs_per_second)


if __name__ == "__main__":
    typer.run(main)
