#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = ["jsonschema", "matplotlib", "pandas", "typer"]
# ///
"""Compare resident packed-input pipelines, native stage costs, and application overhead."""

import csv
import math
from pathlib import Path
from typing import Annotated

import matplotlib as mpl
import numpy as np
import plot_utils as pu
import typer
from benchmark_schema import load_result

STAGES = {
    "cuddl": (
        ("Reset sketches", "resident_reset_wall"),
        ("Hash + construct", "resident_construct_wall"),
        ("Cardinality + winner statistics", "resident_statistics_wall"),
        ("Extract selected rows", "resident_rows_wall"),
        ("Build database + index", "resident_index_wall"),
        ("Search + retain device results", "resident_search_wall"),
    ),
    "rabbitsketch": (
        ("Reset sketches", "resident_reset_wall"),
        ("Hash + construct", "resident_construct_wall"),
        ("Finalize bottom-k", "resident_finalize_wall"),
        ("Cardinality", "resident_cardinality_wall"),
        ("Compare + retain host results", "resident_search_wall"),
    ),
}
WALL_CLOCKS = {"nvbench_cpu_wall", "steady_clock_cpu_wall"}


def read_pipeline(path: Path) -> dict:
    report = load_result(path, "pipeline")
    pipelines = [
        row
        for row in report["measurements"]
        if row["case"].get("measurement") == "pipeline"
    ]
    if len(pipelines) != 1:
        raise ValueError(f"{path}: expected exactly one pipeline measurement")
    pipeline = pipelines[0]
    implementation = pipeline["implementation"]["name"]
    if implementation not in {"cuddl", "rabbitsketch"}:
        raise ValueError(f"{path}: unsupported pipeline {implementation}")
    case = pipeline["case"]
    if not {"k", "references", "queries", "input_cache"} <= case.keys():
        raise ValueError(f"{path}: missing workload settings")
    if case.get("topology") not in {"batch", "all-to-all"}:
        raise ValueError(f"{path}: missing or unsupported topology")
    if implementation == "cuddl":
        if case.get("rows") not in {"compact", "packed"} or case.get("index") not in {
            "sparse",
            "dense",
        }:
            raise ValueError(f"{path}: missing row/index configuration")
        label = f"cuDDL\n{case['rows']} / {case['index']}"
    else:
        label = f"RabbitSketch\n{pipeline['implementation'].get('variant', 'CPU')}"
    return {
        "path": path,
        "report": report,
        "pipeline": pipeline,
        "label": label,
        "implementation": implementation,
    }


def timing(row: dict, key: str) -> dict | None:
    value = row["pipeline"].get("timings", {}).get(key)
    if value is None:
        return None
    if value.get("source") not in WALL_CLOCKS:
        raise ValueError(f"{row['path']}: {key} must be CPU wall time, not GPU events")
    bounds = [value.get(field) for field in ("min_ms", "median_ms", "max_ms")]
    if not all(isinstance(v, (int, float)) and math.isfinite(v) for v in bounds):
        raise ValueError(f"{row['path']}: {key} requires finite min/median/max timings")
    if not 0 <= bounds[0] <= bounds[1] <= bounds[2]:
        raise ValueError(f"{row['path']}: {key} has invalid timing bounds")
    return value


def workload(row: dict) -> tuple:
    report, case = row["report"], row["pipeline"]["case"]
    datasets = report["datasets"]
    if not datasets or not any(name.startswith("reference_") for name in datasets):
        raise ValueError(
            f"{row['path']}: dataset fingerprints are required for comparison"
        )
    return (
        tuple(
            sorted(
                (name, value["sha256"])
                for name, value in datasets.items()
                if case["topology"] == "batch" or name.startswith("reference_")
            )
        ),
        *(
            (
                0
                if key == "queries" and case["topology"] == "all-to-all"
                else case.get(key)
            )
            for key in ("k", "references", "queries", "topology", "input_cache")
        ),
        *(
            report["system"].get(key)
            for key in ("cpu", "architecture", "logical_cpu_count", "ram_bytes")
        ),
    )


def save(fig, output: Path) -> None:
    fig.savefig(
        output.with_suffix(".png"), dpi=200, bbox_inches="tight", facecolor="white"
    )
    pu.save_figure(fig, output.with_suffix(".pdf"))


def main(
    reports: Annotated[list[Path], typer.Argument(exists=True)],
    output_dir: Annotated[Path, typer.Option(file_okay=False)] = Path(
        "results/pipeline-comparison/plots"
    ),
) -> None:
    """Plot resident totals and native stages; keep files-to-JSON totals separate."""
    try:
        paths = []
        for path in reports:
            if path.is_dir():
                discovered = sorted(path.glob("*.json"))
                if not discovered:
                    raise ValueError(f"{path}: no JSON reports found")
                paths.extend(discovered)
            else:
                paths.append(path)
        rows = [read_pipeline(path) for path in paths]
        if len(rows) < 2:
            raise ValueError("provide at least two pipeline reports")
        if any(workload(row) != workload(rows[0]) for row in rows[1:]):
            raise ValueError(
                "reports must match inputs, k, topology, cache policy and host hardware"
            )
        if len({row["label"] for row in rows}) != len(rows):
            raise ValueError("duplicate pipeline labels")
        capacities = {
            row["pipeline"]["case"].get(
                "buckets" if row["implementation"] == "cuddl" else "sketch_size"
            )
            for row in rows
        }
        if None in capacities or len(capacities) != 1:
            raise ValueError("compare equal cuDDL bucket and RabbitSketch entry counts")
        gpu_rows = [row for row in rows if row["implementation"] == "cuddl"]
        settings = {
            tuple(
                row["pipeline"]["case"].get(key)
                for key in ("indexed_buckets", "key_bits", "hash_seed")
            )
            + (row["report"]["system"].get("gpu"),)
            for row in gpu_rows
        }
        if len(settings) > 1:
            raise ValueError("cuDDL reports must match index parameters, seed and GPU")
        for row in rows:
            case = row["pipeline"]["case"]
            if (
                case.get("resident_input") != "packed_u64_actg_max"
                or case.get("resident_minimum_matches") != 0
            ):
                raise ValueError(
                    "rerun benchmarks with resident packed-input support and all-pair output"
                )
            for key in (
                "resident_total_wall",
                "end_to_end_wall",
                *(key for _, key in STAGES[row["implementation"]]),
            ):
                if timing(row, key) is None:
                    raise ValueError(f"{row['path']}: missing {key}; rerun benchmark")
    except ValueError as error:
        raise typer.BadParameter(str(error)) from error

    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "timings.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            [
                "report",
                "pipeline",
                "stage",
                "timer",
                "source",
                "samples",
                "min_ms",
                "median_ms",
                "max_ms",
            ]
        )
        for row in rows:
            for label, key in (
                ("Resident total", "resident_total_wall"),
                ("Files to JSON", "end_to_end_wall"),
                *STAGES[row["implementation"]],
            ):
                value = timing(row, key)
                writer.writerow(
                    [
                        str(row["path"]),
                        row["label"].replace("\n", " / "),
                        label,
                        key,
                        *(
                            value.get(field, "")
                            for field in (
                                "source",
                                "samples",
                                "min_ms",
                                "median_ms",
                                "max_ms",
                            )
                        ),
                    ]
                )
            if row["implementation"] == "cuddl":
                for label, key in (
                    ("Resident total, GPU events", "resident_total_wall"),
                    *STAGES["cuddl"],
                ):
                    key = key.removesuffix("_wall")
                    value = row["pipeline"]["timings"][key]
                    writer.writerow(
                        [
                            str(row["path"]),
                            row["label"].replace("\n", " / "),
                            label,
                            key,
                            *(
                                value.get(field, "")
                                for field in (
                                    "source",
                                    "samples",
                                    "min_ms",
                                    "median_ms",
                                    "max_ms",
                                )
                            ),
                        ]
                    )

    case = rows[0]["pipeline"]["case"]
    q = 0 if case["topology"] == "all-to-all" else case["queries"]
    subtitle = f"{case['topology']} | {case['references']} references, {q} queries | k={case['k']} | {next(iter(capacities))} buckets / entries"
    labels = [
        pu.paper_text(
            row["label"]
            + (
                f"\n{row['pipeline']['case']['orchestration_threads']} threads"
                if row["implementation"] == "rabbitsketch"
                else ""
            )
        )
        for row in rows
    ]
    colors = [
        pu.FILTER_STYLES["cuddl" if row["implementation"] == "cuddl" else "cuco_hll"][
            "color"
        ]
        for row in rows
    ]
    with mpl.rc_context({"font.size": pu.DEFAULT_FONT_SIZE}):
        for key, stem, title, note in (
            (
                "resident_total_wall",
                "resident_total",
                "Resident packed-input pipeline time",
                "Same packed values, resident in native memory. No parsing, input transfers, downloads or JSON.\nReusable input/sketch/workspace/result buffers are preallocated; database/index construction is timed.",
            ),
            (
                "end_to_end_wall",
                "end_to_end",
                "Secondary: files-to-JSON application time",
                "Includes input processing, transfers, native metrics, JSON and teardown.\nThis is application cost, not the resident-compute comparison.",
            ),
        ):
            values = [timing(row, key) for row in rows]
            medians = [value["median_ms"] for value in values]
            maximum = max(value["max_ms"] for value in values)
            fig, ax = pu.setup_figure(figsize=(max(10, 2 * len(rows)), 6))
            bars = ax.bar(
                range(len(rows)),
                medians,
                color=colors,
                edgecolor="black",
                yerr=[
                    [v["median_ms"] - v["min_ms"] for v in values],
                    [v["max_ms"] - v["median_ms"] for v in values],
                ],
                capsize=4,
            )
            for i, (bar, row, value) in enumerate(zip(bars, rows, values, strict=True)):
                bar.set_hatch("//" if row["implementation"] == "rabbitsketch" else "")
                ax.text(
                    i,
                    value["max_ms"] + maximum * 0.025,
                    pu.paper_text(f"{value['median_ms']:.3g} ms\nn={value['samples']}"),
                    ha="center",
                    va="bottom",
                    fontsize=pu.BAR_FONT_SIZE,
                )
            ax.set_xticks(range(len(rows)), labels)
            ax.set_ylim(0, maximum * 1.22 if maximum else 1)
            pu.format_axis(
                ax,
                xlabel="",
                ylabel="Wall time (ms)",
                title=title,
                xscale=None,
                grid=False,
            )
            ax.yaxis.grid(True, alpha=pu.GRID_ALPHA)
            ax.set_axisbelow(True)
            fig.text(
                0.5,
                0.96,
                pu.paper_text(subtitle),
                ha="center",
                fontsize=pu.BAR_FONT_SIZE,
            )
            fig.text(
                0.5,
                0.015,
                pu.paper_text(
                    "Bars: medians; whiskers: observed min-max, not confidence intervals.\n"
                    + note
                ),
                ha="center",
                fontsize=10,
            )
            fig.tight_layout(rect=(0, 0.13, 1, 0.94))
            save(fig, output_dir / stem)

        stage_labels = (
            "Reset sketches",
            "Hash + construct",
            "Cardinality / winner statistics",
            "Prepare comparisons",
            "Build database + index",
            "Search + retain results",
        )
        stage_index = {
            "resident_reset_wall": 0,
            "resident_construct_wall": 1,
            "resident_statistics_wall": 2,
            "resident_cardinality_wall": 2,
            "resident_rows_wall": 3,
            "resident_finalize_wall": 3,
            "resident_index_wall": 4,
            "resident_search_wall": 5,
        }
        values = np.full((len(stage_labels), len(rows)), np.nan)
        for column, row in enumerate(rows):
            for _, key in STAGES[row["implementation"]]:
                values[stage_index[key], column] = timing(row, key)["median_ms"]
        positive = values[values > 0]
        low = float(positive.min()) if positive.size else 1e-6
        high = max(float(positive.max()), low * 10) if positive.size else 1e-5
        norm = mpl.colors.LogNorm(vmin=low, vmax=high, clip=True)
        cmap = mpl.colormaps["viridis"].with_extremes(bad="#eeeeee")
        fig, ax = pu.setup_figure(figsize=(13, 6.6))
        image = ax.imshow(
            np.ma.masked_invalid(values), norm=norm, cmap=cmap, aspect="auto"
        )
        for (row, column), value in np.ndenumerate(values):
            if np.isnan(value):
                text, color = "Not required", "#555555"
            else:
                text = f"{value:.3g}"
                rgb = cmap(norm(value))[:3]
                linear = [
                    c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4
                    for c in rgb
                ]
                luminance = sum(c * w for c, w in zip(linear, (0.2126, 0.7152, 0.0722)))
                color = "black" if luminance > 0.179 else "white"
            ax.text(
                column,
                row,
                pu.paper_text(text),
                ha="center",
                va="center",
                color=color,
                fontsize=12,
            )
        ax.set_xticks(range(len(rows)), labels)
        ax.set_yticks(
            range(len(stage_labels)), [pu.paper_text(label) for label in stage_labels]
        )
        ax.set_xticks(np.arange(len(rows) + 1) - 0.5, minor=True)
        ax.set_yticks(np.arange(len(stage_labels) + 1) - 0.5, minor=True)
        ax.grid(False)
        ax.grid(which="minor", color="white", linewidth=2)
        ax.tick_params(which="both", length=0)
        ax.set_title(pu.paper_text("Resident pipeline stages", bold=True), pad=18)
        colorbar = fig.colorbar(image, ax=ax, pad=0.025, fraction=0.04)
        colorbar.set_label(pu.paper_text("Median wall time (ms, log scale)"))
        fig.text(
            0.5, 0.97, pu.paper_text(subtitle), ha="center", fontsize=pu.BAR_FONT_SIZE
        )
        note = "Cells: isolated-stage medians in ms, not additive. Min-max ranges are in timings.csv.\nPreparation: GPU row extraction; CPU bottom-k finalization. Winner statistics and indexing are GPU-only."
        if np.any(values == 0):
            note += "\nZero timings use the lowest colour, with an explicit 0 label."
        fig.text(0.5, 0.025, pu.paper_text(note), ha="center", fontsize=10)
        fig.tight_layout(rect=(0, 0.11, 1, 0.94))
        save(fig, output_dir / "stages")
    typer.echo(
        f"Plotted {len(rows)} resident pipelines; exact values and clocks: {output_dir / 'timings.csv'}"
    )


if __name__ == "__main__":
    app = typer.Typer(pretty_exceptions_enable=False)
    app.command()(main)
    app()
