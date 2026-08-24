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
four-panel figure:

- asset load (one-shot, both systems),
- index build (cuDDL, BBTools CSR, BBTools CSR2),
- answering the twenty queries (cuDDL batched, both BBTools backends),
- memory footprint (query-time resident and peak during the run for both
  systems),

The figure is written as PDF and PNG. The index-build and query phases are
medians over the measured runs reported by `cuddl-refseq-parity` (after its
warm-up runs); the title and subtitle state the correctness outcome. Asset
load is reported as a one-shot phase.
"""

import json
import statistics
from pathlib import Path
from typing import Annotated

import matplotlib.pyplot as plt
import typer
from plot_utils import FILTER_STYLES, paper_text

# Reuse the repository's existing cuDDL / BBTools palette (see plot_utils.py and
# plot_bbtools_dynamic_demilog.py): cuDDL is the primary `cuddl` blue, BBTools is the
# `cuco_hll` magenta used opposite cuDDL in the BBTools comparison figures. CSR2's packed
# backend gets a lighter shade of the same magenta.
CUDDL_COLOR = FILTER_STYLES["cuddl"]["color"]
BBTOOLS_COLOR = FILTER_STYLES["cuco_hll"]["color"]
CSR2_COLOR = "#C58BB0"

GIB = 1 << 30
PHASES = (("load", "load asset"), ("index", "build index"), ("query", "20 queries"))


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
        "load": phases["asset_read"] + phases["a48_parse"],
        "index": phases["index_build"],
        "query": phases["resident_query_total"],
    }
    bbtools_seconds = bbtools["timings_seconds"]
    bbtools_load = bbtools_seconds["load"] * 1000.0
    bbtools_phases = {
        "load": bbtools_load,
        "index": statistics.median(bbtools_seconds["index_build_csr2_runs"]) * 1000.0,
        "query": statistics.median(bbtools_seconds["query_batch_runs"]) * 1000.0,
    }
    bbtools_csr_phases = {
        "load": bbtools_load,
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

    fig, axes = plt.subplots(1, 4, figsize=(18, 4.8))
    fig.subplots_adjust(top=0.70, wspace=0.45)

    # ---- Shared figure legend (kept out of the panels so it never overlaps the bars). ----
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

    # ---- Title and subtitle (replaces the unreadable bottom footer). ----
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

    def bar_panel(ax, title, entries, fmt, ylabel):
        """One panel of labelled bars with its own log range, so each panel's bars stay wide."""
        bars = ax.bar(
            range(len(entries)),
            [max(value, 1e-6) for _, value, _ in entries],
            0.5,
            color=[color for _, _, color in entries],
        )
        ax.bar_label(bars, fmt=fmt, fontsize=8, padding=2)
        ax.set_yscale("log")
        ax.set_ylim(top=max(value for _, value, _ in entries) * 2.0)
        ax.set_xticks(range(len(entries)))
        ax.set_xticklabels(
            [paper_text(label) for label, _, _ in entries], fontsize=9, rotation=12, ha="right"
        )
        ax.set_ylabel(paper_text(ylabel), fontsize=12)
        ax.set_title(paper_text(title), fontsize=13)
        ax.grid(axis="y", linestyle=":", alpha=0.4)
        ax.set_axisbelow(True)

    # ---- Panel 1: asset load (one-shot, backend-independent for BBTools). ----
    bar_panel(
        axes[0],
        "Asset load (one-shot)",
        [
            ("cuDDL", cuddl_phases["load"], CUDDL_COLOR),
            ("BBTools", bbtools_phases["load"], BBTOOLS_COLOR),
        ],
        lambda v: _latency(v),
        "wall-clock (log)",
    )

    # ---- Panel 2: index build, both BBTools storage backends. ----
    bar_panel(
        axes[1],
        "Index build",
        [
            ("cuDDL", cuddl_phases["index"], CUDDL_COLOR),
            ("BBTools CSR", bbtools_csr_phases["index"], BBTOOLS_COLOR),
            ("BBTools CSR2", bbtools_phases["index"], CSR2_COLOR),
        ],
        lambda v: _latency(v),
        "wall-clock (log)",
    )

    # ---- Panel 3: answering the twenty queries. ----
    bar_panel(
        axes[2],
        "20-query batch",
        [
            ("cuDDL batched", cuddl_phases["query"], CUDDL_COLOR),
            ("BBTools CSR", bbtools_csr_phases["query"], BBTOOLS_COLOR),
            ("BBTools CSR2", bbtools_phases["query"], CSR2_COLOR),
        ],
        lambda v: _latency(v),
        "wall-clock (log)",
    )

    # ---- Panel 4: memory footprint. ----
    # The bars compare the same quantity per system: "query-time resident" is everything each
    # system holds once the index is built (cuDDL rows + dense offsets + postings on the
    # device; for BBTools the JVM heap right after its index build, which includes the decoded
    # rows plus that backend's index). "Peak" is the high-water mark over the whole run; the
    # JVM peak keeps both backends' indexes alive because the harness rebuilds them in
    # alternating order each iteration, so it is a protocol artifact, not a query-time cost.
    ax = axes[3]
    cuddl_rows_bytes = gpu.get("persistent_rows_bytes", 0)
    cuddl_index_bytes = gpu.get("persistent_index_bytes", 0)
    bbtools_memory = bbtools["memory_bytes"]
    memory_groups = [
        (
            "query-time resident",
            (
                ("cuDDL rows + dense index", cuddl_rows_bytes + cuddl_index_bytes, CUDDL_COLOR),
                (
                    "BBTools CSR heap",
                    bbtools_memory.get("heap_used_after_csr", 0),
                    BBTOOLS_COLOR,
                ),
                (
                    "BBTools CSR2 heap",
                    bbtools_memory.get("heap_used_after_csr2", 0),
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
    x = range(len(memory_groups))
    width = 0.32
    for group, (_, entries) in enumerate(memory_groups):
        count = len(entries)
        for i, (label, value, color) in enumerate(entries):
            bars = ax.bar(
                [group + (i - (count - 1) / 2) * width],
                [max(value, 1)],
                width,
                color=color,
            )
            ax.bar_label(bars, fmt=lambda v: _bytes(v), fontsize=7, padding=2)
    top_value = max(value for _, entries in memory_groups for _, value, _ in entries)
    ax.set_ylim(0, top_value * 1.18)
    ax.set_xticks(list(x))
    ax.set_xticklabels([paper_text(label) for label, _ in memory_groups], fontsize=11)
    ax.set_ylabel(paper_text("GiB"), fontsize=12)
    ax.set_title(paper_text("Memory footprint"), fontsize=13)
    ax.text(
        0.5,
        -0.30,
        paper_text(
            "cuDDL: GPU device memory (rows + dense index). BBTools: JVM heap (decoded rows "
            "+ index); the JVM peak keeps both backends alive per alternating iteration."
        ),
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=7.5,
        wrap=True,
    )
    ax.grid(axis="y", linestyle=":", alpha=0.4)
    ax.set_axisbelow(True)

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
    print(
        "queries (batch): "
        f"cuDDL {cuddl_phases['query']:.2f} ms, "
        f"BBTools CSR {bbtools_csr_phases['query']:.2f} ms, "
        f"BBTools CSR2 {bbtools_phases['query']:.2f} ms"
    )


if __name__ == "__main__":
    typer.run(main)
