#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = ["matplotlib", "pandas", "typer"]
# ///
"""Scaling over genome counts and GPUs, from the jobs logs/size-grid.sh, logs/simdsketch-grid.sh and
logs/ceilings.sh run. Writes three panels into OUTPUT_DIR:

- grid_saturation.pdf: on one GPU, each stage's throughput as a share of the hardware ceiling
  that bounds it, against the genome count. Compare: device-resident pairs per second against
  the ALU ceiling (alu-power-ceiling over the search's ALU warp instructions per pair). Download:
  the 8-byte packed counts of every pair copied to the host, against one GPU's device-to-host
  bandwidth (nvbandwidth). End-to-end results: the 4-byte similarities of the m0 pipeline's
  search phase, against the same bandwidth. Cold read: the pipeline's gzip input over its first
  sketch phase after an eviction (r0, m20), against buffered reads from the file system;
- grid_strong.pdf: device-resident speedup over one GPU for every genome count;
- grid_e2e.pdf: FASTA-to-results wall time against the genome count on one node and on the
  most nodes, for cuDDL at --minimum-matches 20 and 0 (warm, median of r1-r3) and SimdSketch.
"""

import json
import re
import statistics
from pathlib import Path
from typing import Annotated

import matplotlib as mpl
import matplotlib.pyplot as plt
import plot_utils as pu
import typer

# acmart sets text in Linux Libertine. plot_utils renders text with LaTeX, whose Computer
# Modern switches to its wide small-size designs at these sizes.
mpl.rcParams.update(
    {
        "font.size": 7,
        "text.latex.preamble": r"\usepackage{libertine}\usepackage[libertine]{newtxmath}",
    }
)
SIZE = re.compile(r"-(\d+)k-")
HOST_BYTES_PER_PAIR = 8  # packed_pairwise_counts
SIMILARITY_BYTES = 4


def size_of(path: Path) -> int:
    return int(SIZE.search(path.name).group(1)) * 1000


def size_label(n: int) -> str:
    return f"{n // 1000}k"


def median_by(paths: list[Path], key) -> dict[tuple, float]:
    """Median of key(report) over the files that share (size, nodes) in their names."""
    groups: dict[tuple, list[float]] = {}
    for path in paths:
        nodes = int(re.search(r"-(\d+)x1-", path.name).group(1))
        groups.setdefault((size_of(path), nodes), []).append(
            key(json.loads(path.read_text()))
        )
    return {k: statistics.median(v) for k, v in groups.items()}


def finish(ax: plt.Axes, fig: plt.Figure, path: Path) -> None:
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(True, which="major", ls=":", lw=0.4, alpha=0.6)
    fig.tight_layout(pad=0.2)
    pu.save_figure(fig, path)


def main(
    results: Annotated[Path, typer.Option(exists=True, file_okay=False)] = Path(
        "results"
    ),
    gpu: Annotated[str, typer.Option(help="GPU tag in the result names")] = "a100",
    alu_per_pair: Annotated[
        float, typer.Option(help="ALU warp instructions per pair (Nsight Compute)")
    ] = 50.4,
    output_dir: Annotated[Path | None, typer.Option()] = None,
) -> None:
    out = pu.resolve_output_dir(output_dir, __file__)
    ceilings = {
        row["phase"]: row["gbps"] * 1e9
        for line in (results / f"mogon-{gpu}-ceilings-refseq-2026-10.jsonl")
        .read_text()
        .splitlines()
        if line.strip()
        for row in [json.loads(line)]
    }
    # A40 ceilings were measured per GPU (-gpu0, -gpu7), A100 for the node: take the slowest GPU.
    alu_files = sorted(results.glob(f"mogon-{gpu}-alu-power-ceiling*.json"))
    if not alu_files:
        raise typer.BadParameter(f"no mogon-{gpu}-alu-power-ceiling*.json in {results}")
    pairs_ceiling = (
        min(
            json.loads(p.read_text())["alu_warp_instructions_per_second"]
            for p in alu_files
        )
        / alu_per_pair
    )
    searches = {
        size_of(p): json.loads(p.read_text())
        for p in results.glob(f"mogon-{gpu}-grid-search-*.json")
    }
    e2e = sorted(results.glob(f"mogon-{gpu}-grid-e2e-*.json"))
    if not searches or not e2e:
        raise typer.BadParameter(f"no grid results for {gpu} in {results}")
    sizes = sorted(searches)
    colors = plt.cm.viridis([i / max(len(sizes) - 1, 1) for i in range(len(sizes))])

    def runs(m: int, reps: str) -> list[Path]:
        return [p for p in e2e if re.search(rf"-m{m}-r[{reps}]\.json$", p.name)]

    def phase_s(report: dict, phase: str) -> float:
        return report["phases_wall_ms"][phase] / 1e3

    # (a) Share of the bounding ceiling on one GPU.
    single = {n: next(r for r in searches[n]["runs"] if r["ranks"] == 1) for n in sizes}
    d2h = ceilings["device_to_host_per_gpu"]
    cold = median_by(runs(20, "0"), lambda r: r["input_bytes"] / phase_s(r, "sketch"))
    results_rate = median_by(
        runs(0, "123"),
        lambda r: r["pairs_total"] * SIMILARITY_BYTES / phase_s(r, "search"),
    )
    stages = [
        (
            "Compare (GPU ALU)",
            {n: single[n]["device"]["pairs_per_second"] / pairs_ceiling for n in sizes},
        ),
        (
            "Download (PCIe)",
            {
                n: single[n]["pairs"]
                * HOST_BYTES_PER_PAIR
                / (single[n]["host"]["median_ms"] / 1e3)
                / d2h
                for n in sizes
            },
        ),
        (
            "E2E results (PCIe)",
            {n: v / d2h for (n, nodes), v in results_rate.items() if nodes == 1},
        ),
        (
            "Cold read (Lustre)",
            {
                n: v / ceilings["filesystem_read_buffered"]
                for (n, nodes), v in cold.items()
                if nodes == 1
            },
        ),
    ]
    fig, ax = plt.subplots(figsize=(2.35, 1.6))
    for (label, shares), marker in zip(stages, "osD^"):
        xs = sorted(shares)
        ax.plot(
            xs,
            [100 * shares[x] for x in xs],
            marker=marker,
            ms=2.5,
            lw=0.9,
            label=label,
        )
    ax.axhline(100, color="#666666", lw=0.6, ls="--")
    ax.set_xscale("log")
    ax.set_ylim(0, 115)
    ax.set_xlabel("Genomes")
    pct = r"\%" if mpl.rcParams["text.usetex"] else "%"
    ax.set_ylabel(f"Share of ceiling [{pct}]")
    ax.legend(
        frameon=False,
        fontsize=6.2,
        loc="lower left",
        bbox_to_anchor=(0.0, 1.0),
        ncol=2,
        handlelength=1.2,
        columnspacing=0.8,
        borderpad=0.1,
    )
    finish(ax, fig, out / "grid_saturation.pdf")
    for label, shares in stages:
        typer.echo(
            f"{label:20} "
            + "  ".join(
                f"{size_label(n)} {100 * shares[n]:5.1f}" for n in sorted(shares)
            )
        )

    # (b) Device-resident strong scaling for every genome count.
    fig, ax = plt.subplots(figsize=(2.35, 1.6))
    most = max(r["ranks"] for r in searches[sizes[-1]]["runs"])
    ax.plot([1, most], [1, most], color="#999999", lw=0.6, ls="--", label="Ideal")
    for n, color in zip(sizes, colors):
        points = sorted(
            (r["ranks"], r["device"]["speedup_vs_first_run"])
            for r in searches[n]["runs"]
        )
        ax.plot(
            *zip(*points), marker="o", ms=2, lw=0.8, color=color, label=size_label(n)
        )
    ax.set_xscale("log", base=2)
    ax.set_yscale("log", base=2)
    ticks = sorted({r for r in (1, 2, 4, 8, 16, 32, 64, most) if r <= most})
    ax.set_xticks(ticks, [str(t) for t in ticks])
    ax.set_yticks(ticks, [str(t) for t in ticks])
    ax.set_xlabel("GPUs")
    ax.set_ylabel("Speedup over 1 GPU")
    ax.legend(
        frameon=False,
        fontsize=6,
        ncol=2,
        loc="upper left",
        handlelength=1.0,
        columnspacing=0.6,
    )
    finish(ax, fig, out / "grid_strong.pdf")

    # (c) End-to-end time against the genome count, on one node and on the most nodes.
    simd = sorted(results.glob("mogon-epyc-grid-simdsketch-*.json"))
    walls = {
        "cuDDL $m_{\\min}=20$": median_by(
            runs(20, "123"), lambda r: r["wall_ms"] / 1e3
        ),
        "cuDDL $m_{\\min}=0$": median_by(runs(0, "123"), lambda r: r["wall_ms"] / 1e3),
        "SimdSketch": median_by(simd, lambda r: r["wall_s"]) if simd else {},
    }
    nodes = sorted({k[1] for k in walls["cuDDL $m_{\\min}=0$"]})
    shown = [nodes[0], nodes[-1]] if len(nodes) > 1 else nodes
    fig, ax = plt.subplots(figsize=(2.35, 1.6))
    for (label, series), color in zip(walls.items(), ["#2E86AB", "#6A4C93", "#E76F51"]):
        for count, style in zip(shown, ["-", "--"]):
            points = sorted((n, s) for (n, c), s in series.items() if c == count)
            if points:
                ax.plot(
                    *zip(*points),
                    ls=style,
                    marker="o",
                    ms=2,
                    lw=0.9,
                    color=color,
                    label=f"{label}, {count} node{'s' if count > 1 else ''}",
                )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Genomes")
    ax.set_ylabel("FASTA to results [s]")
    ax.legend(frameon=False, fontsize=6, loc="upper left", handlelength=1.6)
    finish(ax, fig, out / "grid_e2e.pdf")
    for label, series in walls.items():
        for count in shown:
            typer.echo(
                f"{label:22} {count} nodes "
                + "  ".join(
                    f"{size_label(n)} {s:7.1f}"
                    for (n, c), s in sorted(series.items())
                    if c == count
                )
            )


if __name__ == "__main__":
    typer.run(main)
