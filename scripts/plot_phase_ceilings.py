#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = ["matplotlib", "pandas", "typer"]
# ///
"""Each end-to-end pipeline phase on one node against that node's measured ceiling.

Rows follow the pipeline, each the phase's throughput per node as a share of its own ceiling,
labelled with both in the phase's unit:

- read: gzip bytes per second during the cold sketch phase (scripts/mgpu_eval.py, first read
  from the shared filesystem), against O_DIRECT reads of the same files (scripts/hw_ceilings.py);
- sketch: gzip bytes per second during a warm sketch phase, with the files in the page cache,
  against libdeflate inflating them on every core;
- compare: one GPU's exhaustive all-to-all pairs per second (strong scaling, one rank), against
  the alu-power-ceiling rate of ALU warp instructions divided by the search's ALU warp
  instructions per pair (Nsight Compute, scripts/all_to_all_roofline.py);
- write: result bytes per second during the write phase, against buffered writes to node-local
  scratch.
"""

import json
from pathlib import Path
from typing import Annotated

import matplotlib as mpl
import matplotlib.pyplot as plt
import plot_utils as pu
import typer

mpl.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["Linux Libertine O", "Linux Libertine", "DejaVu Serif"],
        "font.size": 8,
        "pdf.fonttype": 42,
    }
)


def ceiling(rows: list[dict], phase: str) -> dict:
    (row,) = (r for r in rows if r["phase"] == phase)
    return row


def main(
    ceilings: Annotated[
        Path, typer.Option(exists=True, help="scripts/hw_ceilings.py JSON lines")
    ] = Path("results/mogon-a100-ceilings-100k.jsonl"),
    cold: Annotated[
        Path,
        typer.Option(exists=True, help="End-to-end run reading from the filesystem"),
    ] = Path("results/mogon-a100-eval-e2e-100k-1x6-m20-cold.json"),
    warm: Annotated[
        Path,
        typer.Option(
            exists=True,
            help="End-to-end run from the page cache that writes every pair",
        ),
    ] = Path("results/mogon-a100-eval-e2e-100k-1x6-m0-warm.json"),
    strong: Annotated[
        Path, typer.Option(exists=True, help="Strong-scaling search run with one rank")
    ] = Path("results/mogon-a100-eval-100k-1x6-strong.json"),
    alu: Annotated[
        Path,
        typer.Option(
            exists=True, help="alu-power-ceiling result of the same GPU model"
        ),
    ] = Path("results/mogon-a100-alu-power-ceiling.json"),
    alu_per_pair: Annotated[
        float,
        typer.Option(
            min=0.0,
            help="ALU warp instructions per pair of the exhaustive search, from "
            "scripts/all_to_all_roofline.py (59.9 on the 100k snapshot)",
        ),
    ] = 59.9,
    output_dir: Annotated[Path | None, typer.Option()] = None,
) -> None:
    rows = [
        json.loads(line) for line in ceilings.read_text().splitlines() if line.strip()
    ]
    cold_run, warm_run = json.loads(cold.read_text()), json.loads(warm.read_text())
    (single,) = (r for r in json.loads(strong.read_text())["runs"] if r["ranks"] == 1)
    alu_rate = json.loads(alu.read_text())["alu_warp_instructions_per_second"]

    def per_node(run: dict, size: float, phase: str) -> float:
        return size / (run["phases_wall_ms"][phase] / 1e3) / run["nodes"]

    # (label, achieved, ceiling, unit scale, unit)
    phases = [
        (
            "Read\n(cold)",
            per_node(cold_run, cold_run["input_bytes"], "sketch"),
            ceiling(rows, "filesystem_read_direct")["gbps"] * 1e9,
            1e9,
            "GB/s",
        ),
        (
            "Inflate +\nsketch (warm)",
            per_node(warm_run, warm_run["input_bytes"], "sketch"),
            max(r["gbps"] for r in rows if r["phase"].startswith("libdeflate_inflate_"))
            * 1e9,
            1e9,
            "GB/s",
        ),
        (
            "Compare\n(1 GPU)",
            single["device"]["pairs_per_second"],
            alu_rate / alu_per_pair,
            1e9,
            "Gpairs/s",
        ),
        (
            "Write\n(local)",
            per_node(warm_run, warm_run["bytes_written_total"], "write"),
            ceiling(rows, "local_write")["gbps"] * 1e9,
            1e9,
            "GB/s",
        ),
    ]

    fig, ax = plt.subplots(figsize=(3.4, 1.7))
    y = range(len(phases))
    shares = [100 * achieved / limit for _, achieved, limit, _, _ in phases]
    # plot_utils renders text with LaTeX, where a bare % starts a comment.
    pct = r"\%" if mpl.rcParams["text.usetex"] else "%"
    ax.barh(
        y, [100] * len(phases), height=0.62, color="#e6e6e6", label="Measured ceiling"
    )
    ax.barh(y, shares, height=0.62, color="#1f77b4", label="cuDDL")
    ax.axvline(100, color="#666666", lw=0.7, ls="--")
    for yi, share, (_, achieved, limit, scale, unit) in zip(y, shares, phases):
        ax.text(
            max(share, 100) + 2,
            yi,
            f"{share:.0f}{pct}  ({achieved / scale:.2f} / {limit / scale:.2f} {unit})",
            va="center",
            ha="left",
            fontsize=6.8,
        )
    ax.set_yticks(list(y), [p[0] for p in phases], fontsize=7)
    ax.invert_yaxis()
    ax.set_xlim(0, 175)
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.set_xlabel(f"Share of the node's measured ceiling [{pct}]", fontsize=7.5)
    ax.tick_params(axis="x", labelsize=7)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(False)
    ax.legend(
        frameon=False,
        fontsize=6.5,
        loc="lower left",
        bbox_to_anchor=(0.0, 1.0),
        ncol=2,
        handlelength=1.0,
        borderpad=0.1,
    )
    fig.tight_layout(pad=0.3)
    out = pu.resolve_output_dir(output_dir, __file__) / "phase_ceilings.pdf"
    pu.save_figure(fig, out)
    for (label, achieved, limit, scale, unit), share in zip(phases, shares):
        typer.echo(
            f"{label.replace(chr(10), ' '):24} {achieved / scale:8.3f} / {limit / scale:8.3f} "
            f"{unit:9} {share:5.1f}%"
        )


if __name__ == "__main__":
    typer.run(main)
