#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = ["jsonschema", "matplotlib", "pandas", "typer"]
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

import math
from pathlib import Path
from typing import Annotated

import matplotlib.pyplot as plt
import pandas as pd
import typer
from benchmark_schema import flatten_measurements, load_result
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


def _rows(frame: pd.DataFrame, implementation: str, phase: str) -> pd.DataFrame:
    rows = frame[frame["implementation"].eq(implementation) & frame["phase"].eq(phase)]
    if rows.empty:
        raise typer.BadParameter(
            f"benchmark result has no {implementation!r} {phase!r} measurement"
        )
    return rows


def _timing(frame: pd.DataFrame, implementation: str, phase: str) -> float:
    rows = _rows(frame, implementation, phase)
    if len(rows) != 1 or "wall_clock_median_ms" not in rows:
        raise typer.BadParameter(
            f"benchmark result has ambiguous timing for {implementation!r} {phase!r}"
        )
    value = pd.to_numeric(rows["wall_clock_median_ms"], errors="coerce").iloc[0]
    if pd.isna(value):
        raise typer.BadParameter(
            f"benchmark result has invalid timing for {implementation!r} {phase!r}"
        )
    value = float(value)
    if not math.isfinite(value) or value < 0:
        raise typer.BadParameter(
            f"benchmark result has invalid timing for {implementation!r} {phase!r}"
        )
    return value


def _memory(frame: pd.DataFrame, implementation: str, phase: str, name: str) -> int:
    rows = _rows(frame, implementation, phase)
    if len(rows) != 1 or name not in rows:
        raise typer.BadParameter(
            f"benchmark result has ambiguous memory for {implementation!r} {phase!r}"
        )
    value = pd.to_numeric(rows[name], errors="coerce").iloc[0]
    if pd.isna(value):
        raise typer.BadParameter(
            f"benchmark result has invalid memory for {implementation!r} {phase!r}"
        )
    value = float(value)
    if not math.isfinite(value) or value < 0 or not value.is_integer():
        raise typer.BadParameter(
            f"benchmark result has invalid memory for {implementation!r} {phase!r}"
        )
    return int(value)


def main(
    evidence: Annotated[
        Path,
        typer.Option(
            exists=True, dir_okay=False, help="cuddl-refseq-parity benchmark JSON"
        ),
    ] = Path("results/refseq-parity/evidence.json"),
    output: Annotated[
        Path,
        typer.Option(
            dir_okay=False, help="Figure path prefix (.pdf and .png are written)"
        ),
    ] = Path("results/refseq-parity/comparison"),
) -> None:
    try:
        result = load_result(evidence, "refseq_parity")
    except ValueError as error:
        raise typer.BadParameter(f"{evidence}: {error}") from error
    frame = pd.DataFrame(flatten_measurements(result))
    if frame.empty:
        raise typer.BadParameter("provide at least one benchmark measurement")

    status_rows = _rows(frame, "RefSeq parity", "correctness")
    if len(status_rows) != 1:
        raise typer.BadParameter("benchmark result has ambiguous correctness status")
    status = status_rows.iloc[0]
    if not bool(status["ok"]):
        typer.echo(f"evidence at {evidence} is not a passing run (ok: false)", err=True)
        raise typer.Exit(code=1)
    queries_ok = int(status["queries_ok"])
    queries_total = int(status["queries_total"])
    if queries_total < 1 or queries_ok != queries_total:
        typer.echo(
            f"evidence at {evidence} has incomplete query validation "
            f"({queries_ok}/{queries_total})",
            err=True,
        )
        raise typer.Exit(code=1)

    implementations = set(frame["implementation"])
    if not {"BBTools CSR", "BBTools CSR2"}.issubset(implementations):
        typer.echo(
            "evidence has no BBTools measurements; run without --skip-bbtools", err=True
        )
        raise typer.Exit(code=1)
    if not (
        frame["implementation"].eq("cuDDL")
        & frame["phase"].isin(("index_build", "query_batch"))
    ).any():
        typer.echo("evidence has no GPU measurements; run without --no-gpu", err=True)
        raise typer.Exit(code=1)

    cuddl_phases = {
        "index": _timing(frame, "cuDDL", "index_build"),
        "query": _timing(frame, "cuDDL", "query_batch"),
    }
    bbtools_phases = {
        "index": _timing(frame, "BBTools CSR2", "index_build"),
        "query": _timing(frame, "BBTools CSR2", "query_batch"),
    }
    bbtools_csr_phases = {
        "index": _timing(frame, "BBTools CSR", "index_build"),
        "query": _timing(frame, "BBTools CSR", "query_batch"),
    }

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    fig.subplots_adjust(top=0.70, wspace=0.45)

    legend_handles = [
        plt.Line2D(
            [0],
            [0],
            marker="s",
            linestyle="",
            color=CUDDL_COLOR,
            label=paper_text("cuDDL (GPU)"),
        ),
        plt.Line2D(
            [0],
            [0],
            marker="s",
            linestyle="",
            color=BBTOOLS_COLOR,
            label=paper_text("BBTools CSR (32-bit)"),
        ),
        plt.Line2D(
            [0],
            [0],
            marker="s",
            linestyle="",
            color=CSR2_COLOR,
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
            f"{queries_ok}/{queries_total} selected queries: identical index match counts, "
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
    speed_ax.set_ylim(
        top=max(value for _, values, _ in speed_systems for value in values) * 2.0
    )
    speed_ax.set_xticks(range(len(speed_groups)))
    speed_ax.set_xticklabels([paper_text(label) for label in speed_groups], fontsize=11)
    speed_ax.set_ylabel(paper_text("wall-clock (log)"), fontsize=12)
    speed_ax.set_title(paper_text("Speed"), fontsize=13)
    speed_ax.grid(axis="y", linestyle=":", alpha=0.4)
    speed_ax.set_axisbelow(True)

    # Panel 2: memory footprint.
    memory_ax = axes[1]
    cuddl_rows_bytes = _memory(frame, "cuDDL", "query_batch", "persistent_rows_bytes")
    cuddl_index_bytes = _memory(frame, "cuDDL", "query_batch", "persistent_index_bytes")
    cuddl_peak_bytes = _memory(frame, "cuDDL", "query_batch", "peak_allocated_bytes")
    bbtools_rows_bytes = _memory(
        frame, "BBTools CSR2", "query_batch", "heap_used_before_index"
    )
    bbtools_csr_bytes = _memory(frame, "BBTools CSR", "query_batch", "index_csr_bytes")
    bbtools_csr2_bytes = _memory(
        frame, "BBTools CSR2", "query_batch", "index_csr2_bytes"
    )
    bbtools_peak_bytes = _memory(frame, "BBTools CSR2", "query_batch", "heap_used_peak")
    memory_groups = [
        (
            "query-time resident",
            (
                (
                    "cuDDL rows + dense index",
                    cuddl_rows_bytes + cuddl_index_bytes,
                    CUDDL_COLOR,
                ),
                (
                    "BBTools rows + CSR index",
                    bbtools_rows_bytes + bbtools_csr_bytes,
                    BBTOOLS_COLOR,
                ),
                (
                    "BBTools rows + CSR2 index",
                    bbtools_rows_bytes + bbtools_csr2_bytes,
                    CSR2_COLOR,
                ),
            ),
        ),
        (
            "peak during run",
            (
                ("cuDDL peak device", cuddl_peak_bytes, CUDDL_COLOR),
                (
                    "BBTools JVM heap peak",
                    bbtools_peak_bytes,
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
    memory_ax.set_xticklabels(
        [paper_text(label) for label, _ in memory_groups], fontsize=11
    )
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
