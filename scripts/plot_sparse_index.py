#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = ["matplotlib", "pandas", "typer"]
# ///
"""Plot dense/sparse NVBench measurements as a 16:9 slide (PDF, SVG, PNG).

Generate compatible input with:
  build/benchmarks/cuddl-compact-search-benchmark \
    --stopping-criterion sample-count --min-samples 30 --target-samples 30 \
    --no-batch --csv results/index-layouts.csv \
    -b compact_indexed_build -b compact_indexed_batch_search

Then run:
  nix develop -c uv run scripts/plot_sparse_index.py results/index-layouts.csv --queries 1024

The devShell supplies LaTeX for the shared plot_utils styling.
This uses the benchmark's compact 2,048-bucket, 16-bit-key fixture. Memory
excludes original rows, temporary scratch, and unused pool reservations.
Timings include the complete indexed build or query-batch search operation;
input upload is outside both timed regions. CSV medians have no associated
confidence intervals. Repeated CSV states are rejected, not averaged.
"""

import csv
import math
from pathlib import Path
from typing import Annotated

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import plot_utils as pu
import typer
from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator

BUCKETS = 2048
KEYS = 65536
STYLES = {
    "dense": {**pu.FILTER_STYLES["cuddl"], "linestyle": "-"},
    "sparse": {**pu.FILTER_STYLES["cuddl_paper"], "linestyle": "--"},
}


def load_measurements(path: Path, queries: int = 1) -> tuple[list[dict], str]:
    """Require matched layouts/workloads and check the memory accounting."""
    query_benchmark = (
        "compact_indexed_search" if queries == 1 else "compact_indexed_batch_search"
    )
    states = {}
    devices = set()
    with path.open(newline="") as stream:
        for row in csv.DictReader(stream):
            benchmark = row.get("Benchmark")
            if benchmark not in {"compact_indexed_build", query_benchmark}:
                continue
            if (
                benchmark == "compact_indexed_batch_search"
                and int(row["Queries"]) != queries
            ):
                continue
            layout = row["Index"]
            references = int(row["References"])
            key = (references, layout, benchmark)
            if layout not in STYLES or row["Skipped"] != "No" or references <= 0:
                raise ValueError(f"invalid or skipped configuration: {key}")
            if key in states:
                raise ValueError(
                    f"duplicate configuration: {key}; supply one benchmark run"
                )
            seconds = float(row["Median GPU Time"])
            samples = int(row["Samples"])
            resident = float(row["Resident Bytes"])
            index_bytes = resident - references * BUCKETS * 2
            expected = (
                4 * (BUCKETS * KEYS + 1) + 4 * BUCKETS * references
                if layout == "dense"
                else 6 * BUCKETS * references
            )
            if not math.isfinite(seconds) or seconds <= 0 or samples < 2:
                raise ValueError(f"invalid timing/sample count: {key}")
            if index_bytes != expected:
                raise ValueError(
                    f"memory does not match the compact 2,048-bucket fixture: {key}"
                )
            if (
                benchmark == "compact_indexed_build"
                and int(row["Scores Indexed"]) != references * BUCKETS
            ):
                raise ValueError(f"unexpected indexed bucket count: {key}")
            devices.add((row["Device"], row["Device Name"]))
            states[key] = (seconds, samples, index_bytes / 2**20)
    if len(devices) != 1:
        raise ValueError("expected measurements from exactly one GPU")
    records = []
    for references in sorted({key[0] for key in states}):
        for layout in STYLES:
            try:
                build = states[references, layout, "compact_indexed_build"]
                query = states[references, layout, query_benchmark]
            except KeyError as error:
                raise ValueError(
                    f"missing dense/sparse build/query counterpart at R={references}"
                ) from error
            records.append(
                {
                    "references": references,
                    "layout": layout,
                    "build_ms": build[0] * 1e3,
                    "query_ms": query[0] * 1e3,
                    "queries": queries,
                    "index_mib": build[2],
                    "build_samples": build[1],
                    "query_samples": query[1],
                }
            )
    return records, next(iter(devices))[1]


def main(
    csv_path: Annotated[
        Path, typer.Argument(exists=True, dir_okay=False, help="NVBench layout CSV")
    ],
    output: Annotated[
        Path, typer.Option(help="Output basename, without extension")
    ] = Path("results/sparse-index"),
    queries: Annotated[int, typer.Option(min=1, help="Query count to plot")] = 1024,
    note: Annotated[
        str, typer.Option(help="Acquisition caveat saved in the companion caption")
    ] = "",
) -> None:
    """Export a slide, its plotted data, and a caption with measurement definitions."""
    try:
        records, device = load_measurements(csv_path, queries)
    except (ValueError, KeyError) as error:
        raise typer.BadParameter(str(error), param_hint="csv_path") from error
    references = sorted({row["references"] for row in records})
    # Keep wide sweeps readable without dropping any measured points.
    ticks = (
        references
        if len(references) <= 5
        else [
            2**power
            for power in range(
                math.floor(math.log2(references[0])),
                math.ceil(math.log2(references[-1])) + 1,
                3,
            )
            if references[0] / 1.1 <= 2**power <= references[-1]
        ]
    )
    sample_counts = sorted(
        {row[key] for row in records for key in ("build_samples", "query_samples")}
    )
    sample_label = (
        str(sample_counts[0])
        if len(sample_counts) == 1
        else f"{sample_counts[0]} to {sample_counts[-1]}"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    panels = [
        ("build_ms", "Build time", "Median time (ms)", True),
        (
            "query_ms",
            f"Query time ({queries:,} {'query' if queries == 1 else 'queries'})",
            "Median batch time (ms)",
            True,
        ),
        ("index_mib", "Index memory", "Persistent index (MiB)", True),
    ]
    with plt.rc_context(
        {
            "font.size": pu.DEFAULT_FONT_SIZE,
            "xtick.labelsize": pu.TICK_LABEL_FONT_SIZE,
            "ytick.labelsize": pu.TICK_LABEL_FONT_SIZE,
        }
    ):
        fig, axes = pu.setup_figure(figsize=(16, 9), ncols=3)
        fig.set_facecolor("white")
        fig.subplots_adjust(left=0.075, right=0.97, bottom=0.13, top=0.75, wspace=0.35)
        fig.text(
            0.065,
            0.92,
            pu.paper_text("Dense vs. sparse index", bold=True),
            fontsize=29,
        )
        for ax, (metric, title, ylabel, logarithmic) in zip(axes, panels, strict=True):
            for layout, style in STYLES.items():
                series = [row for row in records if row["layout"] == layout]
                ax.plot(
                    references,
                    [row[metric] for row in series],
                    label=layout.capitalize(),
                    linewidth=pu.LINE_WIDTH,
                    markersize=pu.MARKER_SIZE,
                    **style,
                )
            pu.format_axis(
                ax,
                xlabel="References",
                ylabel=ylabel,
                title=title,
                xscale="log",
                yscale="log" if logarithmic else "linear",
                grid=False,
            )
            ax.xaxis.set_major_locator(FixedLocator(ticks))
            ax.xaxis.set_major_formatter(
                FuncFormatter(lambda value, _: f"{value:,.0f}")
            )
            ax.xaxis.set_minor_locator(NullLocator())
            if logarithmic:
                ax.set_yscale("log", base=10)
                ax.yaxis.set_major_formatter(
                    FuncFormatter(lambda value, _: f"{value:g}")
                )
            else:
                ax.set_ylim(bottom=0)
            ax.grid(
                axis="y",
                which="major",
                alpha=pu.GRID_ALPHA,
                linestyle="--",
                linewidth=0.8,
            )
            ax.set_axisbelow(True)
            ax.margins(x=0.08, y=0.2)
        fig.legend(
            *axes[0].get_legend_handles_labels(),
            loc="upper left",
            bbox_to_anchor=(0.058, 0.87),
            ncols=2,
            frameon=False,
            fontsize=pu.LEGEND_FONT_SIZE,
        )
        for extension in ("pdf", "svg", "png"):
            target = Path(f"{output}.{extension}")
            fig.savefig(target, dpi=180, facecolor="white")
            typer.echo(target)
        plt.close(fig)
    with Path(f"{output}.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    Path(f"{output}.txt").write_text(
        f"Source: {csv_path.resolve()}\nGPU: {device}\n"
        f"Timed samples per configuration: {sample_label}\n"
        "Dense and sparse indexes on the same synthetic compact-score fixture.\n"
        "Timings: NVBench median GPU time, converted from seconds to milliseconds.\n"
        f"Query count: {queries}; time covers the full batch, not an amortized per-query value.\n"
        "Build includes database creation/destruction and index construction on resident input.\n"
        "Query includes posting lookup, candidate selection, and exact refinement.\n"
        + (
            "Batch queries sample evenly spaced reference rows; rows repeat when query count exceeds reference count.\n"
            "Batch output storage is reused in 128-query tiles; host validation and transfers are not timed.\n"
            if queries > 1
            else "The single-query benchmark uses its synthetic probe query.\n"
        )
        + "Index MiB = (Resident Bytes - references * 2048 * sizeof(uint16_t)) / 2**20.\n"
        "This is persistent index storage, not peak allocation or pool-reserved GPU memory.\n"
        "All dense/sparse configurations of the two named benchmarks are plotted; other benchmarks are ignored.\n"
        "Log10 build, query, and memory axes.\n"
        "Log2 reference axis. Source-data CSV retains absolute values.\n"
        "One run; no uncertainty intervals available from these CSV summaries.\n"
        f"Acquisition note: {note or 'not supplied'}\n"
    )


if __name__ == "__main__":
    typer.run(main)
