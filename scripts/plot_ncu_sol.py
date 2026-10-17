#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = ["matplotlib", "pandas", "typer"]
# ///
"""Speed-of-light throughputs from scripts/run_ncu_sol.py, one figure per corpus size.

ncu_sol_<genomes>: every profiled kernel of every tool, labelled with its share of the run's GPU
time. The left panel is SM throughput, the right one memory throughput, with the L1, L2 and DRAM
throughputs it is the maximum of marked on the bar. All values are percentages of the peak.
"""

import json
from pathlib import Path
from typing import Annotated

import matplotlib.pyplot as plt
import plot_utils as pu
import typer

app = typer.Typer(add_completion=False)
UNITS = (("l1", "L1", "o"), ("l2", "L2", "s"), ("dram", "DRAM", "D"))


@app.command()
def main(
    results: Annotated[Path, typer.Argument(exists=True, file_okay=False, help="Output directory")],
    output_dir: Annotated[Path | None, typer.Option(help="Defaults to RESULTS/plots")] = None,
) -> None:
    output_dir = output_dir or results / "plots"
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = [json.loads(line) for line in (results / "kernels.jsonl").read_text().splitlines()]
    tools = list(dict.fromkeys(row["tool"] for row in rows))
    colors = {tool: plt.cm.tab10(i % 10) for i, tool in enumerate(tools)}
    for genomes in sorted({row["genomes"] for row in rows}):
        kernels = sorted(
            (row for row in rows if row["genomes"] == genomes),
            key=lambda row: (tools.index(row["tool"]), -row["time_share"]),
        )
        labels = [
            pu.paper_text(f"{row['tool']}: {row['kernel'][:40]} ({row['time_share']:.0%})")
            for row in kernels
        ]
        positions = range(len(kernels))[::-1]
        fig, (sm_ax, memory_ax) = plt.subplots(
            1, 2, figsize=(12, 0.45 * len(kernels) + 1.5), sharey=True
        )
        bar_colors = [colors[row["tool"]] for row in kernels]
        sm_ax.barh(positions, [row["sm"] for row in kernels], color=bar_colors)
        memory_ax.barh(positions, [row["memory"] for row in kernels], color=bar_colors, alpha=0.6)
        for key, name, marker in UNITS:
            memory_ax.scatter([row[key] for row in kernels], positions, marker=marker,
                              color="black", s=pu.MARKER_SIZE * 4, label=name, zorder=3)
        sm_ax.set_yticks(list(positions), labels)
        for ax, title in ((sm_ax, "SM throughput"), (memory_ax, "Memory throughput")):
            ax.set_xlim(0, 100)
            ax.set_xlabel(pu.paper_text("% of peak"))
            ax.set_title(f"{title}, {genomes:,} genomes", fontsize=pu.TITLE_FONT_SIZE)
            ax.grid(True, axis="x", ls="--", alpha=pu.GRID_ALPHA)
        memory_ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=len(UNITS),
                         fontsize=pu.LEGEND_FONT_SIZE * 0.7, framealpha=pu.LEGEND_FRAME_ALPHA)
        fig.tight_layout()
        fig.savefig(output_dir / f"ncu_sol_{genomes}.png", dpi=200, bbox_inches="tight")
        pu.save_figure(fig, output_dir / f"ncu_sol_{genomes}.pdf")


if __name__ == "__main__":
    app()
