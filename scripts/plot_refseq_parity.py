#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "typer",
#   "matplotlib",
# ]
# ///
"""Plot the cuDDL versus BBTools decoded-row RefSeq parity comparison.

Reads the machine-readable evidence produced by `cuddl-refseq-parity`
(`results/refseq-parity/evidence.json`) and renders one slide-oriented
two-panel figure:

- speed (index build and query batch for cuDDL, BBTools CSR, BBTools CSR2),
- memory footprint (query-time resident and peak during the run for both
  systems).

The figure is written as PDF and PNG. The speed bars are medians over the
measured runs reported by `cuddl-refseq-parity`; the title and subtitle state
the correctness outcome.
"""

import json
import statistics
from pathlib import Path
from typing import Annotated

import matplotlib.pyplot as plt
import typer
from plot_utils import FILTER_STYLES, paper_text

CUDDL_COLOR = FILTER_STYLES["cuddl"]["color"]
BBTOOLS_COLOR = FILTER_STYLES["cuco_hll"]["color"]
CSR2_COLOR = "#C58BB0"

GIB = 1 << 30
PHASES = (("index", "build index"), ("query", "query batch"))


def _bytes(value: float) -> str:
    return f"{value / GIB:.2f} GiB"


def _latency(value_ms: float) -> str:
    if value_ms >= 100:
        return f"{value_ms / 1000:.1f} s"
    if value_ms >= 1:
        return f"{value_ms:.1f} ms"
    return f"{value_ms * 1000:.0f} us"


def main(
    evidence: Annotated[
        Path,
        typer.Option(
            exists=True, dir_okay=False, help="cuddl-refseq-parity evidence JSON"
        ),
    ] = Path("results/refseq-parity/evidence.json"),
    output: Annotated[
        Path,
        typer.Option(
            dir_okay=False, help="Figure path prefix (.pdf and .png are written)"
        ),
    ] = Path("results/refseq-parity/comparison"),
) -> None:
    data = json.loads(evidence.read_text())

    if not data.get("ok"):
        typer.echo(
            f"evidence at {evidence} is not a passing run (ok: {data.get('ok')})", err=True
        )
        raise typer.Exit(code=1)
    if not data.get("bbtools"):
        typer.echo("evidence has no BBTools section; run without --skip-bbtools", err=True)
        raise typer.Exit(code=1)
    if not data.get("gpu", {}).get("available"):
        typer.echo("evidence has no GPU results; run without --no-gpu", err=True)
        raise typer.Exit(code=1)

    phases = data["phases_ms"]
    bbtools = data["bbtools"]
    gpu = data["gpu"]
    queries = data["queries"]

    cuddl_phases = {
        "index": phases["index_build"],
        "query": phases["query_total"],
    }
    bbtools_seconds = bbtools["timings_seconds"]
    bbtools_phases = {
        "index": statistics.median(bbtools_seconds["index_build_csr2_runs"]) * 1000.0,
        "query": statistics.median(bbtools_seconds["query_batch_runs"]) * 1000.0,
    }
    bbtools_csr_phases = {
        "index": statistics.median(bbtools_seconds["index_build_csr_runs"]) * 1000.0,
        "query": statistics.median(bbtools_seconds["query_batch_csr_runs"]) * 1000.0,
    }

    queries_ok = sum(
        all(
            q[key]
            for key in (
                "decode_matches_bbtools",
                "counts_match_bbtools",
                "candidates_match_bbtools",
                "summaries_match_bbtools",
                "candidates_match_oracle",
                "summaries_match_oracle",
                "summaries_match_exhaustive",
            )
        )
        for q in queries
    )

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    fig.subplots_adjust(top=0.70, wspace=0.45)

    legend_handles = [
        plt.Line2D(
            [0], [0], marker="s", linestyle="", color=CUDDL_COLOR,
            label=paper_text("cuDDL (GPU)"),
        ),
        plt.Line2D(
            [0], [0], marker="s", linestyle="", color=BBTOOLS_COLOR,
            label=paper_text("BBTools CSR (32-bit)"),
        ),
        plt.Line2D(
            [0], [0], marker="s", linestyle="", color=CSR2_COLOR,
            label=paper_text("BBTools CSR2 (21-bit)"),
        ),
    ]
    fig.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.87),
        ncol=3,
        frameon=False,
        fontsize=11,
    )

    fig.suptitle(
        paper_text("Decoded-row RefSeq parity: cuDDL vs BBTools DDLIndex/CSR2"),
        fontsize=15,
        y=0.995,
    )
    fig.text(
        0.5,
        0.930,
        paper_text(
            f"{queries_ok}/{len(queries)} selected queries: identical index match counts, "
            "candidate IDs, and exact summaries across cuDDL, BBTools, and the exhaustive "
            "decoded-row oracle"
        ),
        ha="center",
        va="center",
        fontsize=11,
    )

    # Panel 1: speed. Index build and query batch share the same log-latency axis.
    speed_ax = axes[0]
    speed_groups = ["build index", "query batch"]
    speed_systems = [
        ("cuDDL", [cuddl_phases["index"], cuddl_phases["query"]], CUDDL_COLOR),
        (
            "BBTools CSR",
            [bbtools_csr_phases["index"], bbtools_csr_phases["query"]],
            BBTOOLS_COLOR,
        ),
        (
            "BBTools CSR2",
            [bbtools_phases["index"], bbtools_phases["query"]],
            CSR2_COLOR,
        ),
    ]
    width = 0.24
    for system_index, (_, values, color) in enumerate(speed_systems):
        offset = (system_index - 1) * width
        bars = speed_ax.bar(
            [group + offset for group in range(len(speed_groups))],
            [max(value, 1e-6) for value in values],
            width,
            color=color,
        )
        speed_ax.bar_label(bars, fmt=lambda v: _latency(v), fontsize=7, padding=2)
    speed_ax.set_yscale("log")
    speed_ax.set_ylim(top=max(value for _, values, _ in speed_systems for value in values) * 2.0)
    speed_ax.set_xticks(range(len(speed_groups)))
    speed_ax.set_xticklabels([paper_text(label) for label in speed_groups], fontsize=11)
    speed_ax.set_ylabel(paper_text("wall-clock (log)"), fontsize=12)
    speed_ax.set_title(paper_text("Speed"), fontsize=13)
    speed_ax.grid(axis="y", linestyle=":", alpha=0.4)
    speed_ax.set_axisbelow(True)

    # Panel 2: memory footprint.
    memory_ax = axes[1]
    cuddl_rows_bytes = gpu.get("persistent_rows_bytes", 0)
    cuddl_index_bytes = gpu.get("persistent_index_bytes", 0)
    bbtools_memory = bbtools["memory_bytes"]
    bbtools_rows_bytes = bbtools_memory.get("heap_used_before_index", 0)
    memory_groups = [
        (
            "query-time resident",
            (
                ("cuDDL rows + dense index", cuddl_rows_bytes + cuddl_index_bytes, CUDDL_COLOR),
                (
                    "BBTools rows + CSR index",
                    bbtools_rows_bytes + bbtools_memory.get("index_csr_bytes", 0),
                    BBTOOLS_COLOR,
                ),
                (
                    "BBTools rows + CSR2 index",
                    bbtools_rows_bytes + bbtools_memory.get("index_csr2_bytes", 0),
                    CSR2_COLOR,
                ),
            ),
        ),
        (
            "peak during run",
            (
                ("cuDDL peak device", gpu.get("peak_allocated_bytes", 0), CUDDL_COLOR),
                (
                    "BBTools JVM heap peak",
                    bbtools_memory.get("heap_used_peak", 0),
                    BBTOOLS_COLOR,
                ),
            ),
        ),
    ]
    memory_width = 0.28
    for group, (_, entries) in enumerate(memory_groups):
        count = len(entries)
        for i, (_, value, color) in enumerate(entries):
            bars = memory_ax.bar(
                [group + (i - (count - 1) / 2) * memory_width],
                [max(value, 1)],
                memory_width,
                color=color,
            )
            memory_ax.bar_label(bars, fmt=lambda v: _bytes(v), fontsize=7, padding=2)
    top_value = max(value for _, entries in memory_groups for _, value, _ in entries)
    memory_ax.set_ylim(0, top_value * 1.18)
    memory_ax.set_xticks([0, 1])
    memory_ax.set_xticklabels([paper_text(label) for label, _ in memory_groups], fontsize=11)
    memory_ax.set_ylabel(paper_text("GiB"), fontsize=12)
    memory_ax.set_title(paper_text("Memory footprint"), fontsize=13)
    memory_ax.grid(axis="y", linestyle=":", alpha=0.4)
    memory_ax.set_axisbelow(True)

    output = output.with_suffix("")
    fig.savefig(f"{output}.pdf", bbox_inches="tight", pad_inches=0.15)
    fig.savefig(f"{output}.png", bbox_inches="tight", dpi=220, pad_inches=0.15)
    print(f"wrote {output}.pdf")
    print(f"wrote {output}.png")
    print("phases (ms):")
    for key, label in PHASES:
        print(
            f"  {label:>12}: cuDDL {cuddl_phases[key]:12.3f} | "
            f"BBTools CSR {bbtools_csr_phases[key]:12.3f} | "
            f"BBTools CSR2 {bbtools_phases[key]:12.3f}"
        )


if __name__ == "__main__":
    typer.run(main)
