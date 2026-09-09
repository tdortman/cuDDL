#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = ["jsonschema", "matplotlib", "pandas", "typer"]
# ///
"""Compare packed/sequence resident pipelines or legacy sequence application timings."""

import csv
import math
from pathlib import Path
from typing import Annotated

import matplotlib as mpl
import matplotlib.pyplot as plt
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
SEQUENCE_STAGES = {
    "cuddl": (
        ("Reset sketches", "resident_reset_wall"),
        ("Encode + hash + construct", "resident_construct_wall"),
        ("Cardinality + winner statistics", "resident_statistics_wall"),
        ("Extract selected rows", "resident_rows_wall"),
        ("Build database + index", "resident_index_wall"),
        ("Search + retain device results", "resident_search_wall"),
    ),
    "rabbitsketch": (
        ("Reset sketches", "resident_reset_wall"),
        ("Encode + hash + construct", "resident_construct_wall"),
        ("Finalize bottom-k", "resident_finalize_wall"),
        ("Cardinality", "resident_cardinality_wall"),
        ("Compare + retain host results", "resident_search_wall"),
    ),
}
WALL_CLOCKS = {"nvbench_cpu_wall", "steady_clock_cpu_wall"}
APPLICATION_STAGES = (
    ("Prepare sketches / database", "prepare_wall"),
    ("Search + metrics + JSON", "query_output_wall"),
    ("Teardown", "teardown_wall"),
)
SEQUENCE_RESIDENT_INPUT = "sequence_ascii"
BATCHED_SCOPE = "batched_resident_segments"
SINGLE_SCOPE = "resident_pipeline"
RESIDENT_SCOPES = {BATCHED_SCOPE, SINGLE_SCOPE}


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
        case.get("ingest", "packed"),
        case.get("resident_batch_bytes"),
        case.get("resident_batches"),
        case.get("resident_timing_scope"),
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
    """Plot packed/sequence resident totals and stages, or legacy sequence phases."""
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
                "reports must match inputs, k, topology, cache policy, ingest mode, resident input, batch contract and host hardware"
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
        sequence = rows[0]["pipeline"]["case"].get("ingest", "packed") == "sequence"
        has_resident = any(
            row["pipeline"].get("timings", {}).get("resident_total_wall") is not None
            for row in rows
        )
        sequence_resident = sequence and has_resident
        legacy_sequence = sequence and not has_resident
        if sequence and any(
            (row["pipeline"].get("timings", {}).get("resident_total_wall") is not None)
            != has_resident
            for row in rows
        ):
            raise ValueError("mixed resident and legacy sequence reports; rerun benchmarks")
        scope = rows[0]["pipeline"]["case"].get("resident_timing_scope")
        batched = sequence_resident and scope == BATCHED_SCOPE
        if sequence_resident:
            stages = SEQUENCE_STAGES
            totals = [
                (
                    "Resident total (summed segments)"
                    if batched
                    else "Resident total",
                    "resident_total_wall",
                ),
                ("Files to JSON", "end_to_end_wall"),
            ]
        elif legacy_sequence:
            stages = {implementation: APPLICATION_STAGES for implementation in STAGES}
            totals = [("Files to JSON", "end_to_end_wall")]
        else:
            stages = STAGES
            totals = [
                ("Resident total", "resident_total_wall"),
                ("Files to JSON", "end_to_end_wall"),
            ]
        for row in rows:
            case = row["pipeline"]["case"]
            if sequence_resident:
                if case.get("resident_input") != SEQUENCE_RESIDENT_INPUT:
                    raise ValueError(
                        f"{row['path']}: expected {SEQUENCE_RESIDENT_INPUT} input and all-pair output"
                    )
                batch_bytes = case.get("resident_batch_bytes")
                batches = case.get("resident_batches")
                if (
                    not isinstance(batch_bytes, int)
                    or batch_bytes <= 0
                    or not isinstance(batches, int)
                    or batches < 0
                    or case.get("resident_timing_scope") not in RESIDENT_SCOPES
                ):
                    raise ValueError(
                        f"{row['path']}: missing resident batch contract; rerun benchmark"
                    )
            else:
                expected_input = (
                    {"cuddl": "sequence_tiles", "rabbitsketch": "fastx_files"}[
                        row["implementation"]
                    ]
                    if legacy_sequence
                    else "packed_u64_actg_max"
                )
                if case.get("resident_input") != expected_input:
                    raise ValueError(
                        f"{row['path']}: expected {expected_input} input and all-pair output"
                    )
            if case.get("resident_minimum_matches") != 0:
                raise ValueError(
                    f"{row['path']}: expected all-pair output with minimum matches 0"
                )
            for key in (
                *(key for _, key in totals),
                *(key for _, key in stages[row["implementation"]]),
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
                *totals,
                *stages[row["implementation"]],
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
            if row["implementation"] == "cuddl" and not legacy_sequence:
                for label, key in (
                    (totals[0][0] + ", GPU events", "resident_total_wall"),
                    *stages["cuddl"],
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
    if sequence_resident:
        subtitle += f" | {case['resident_batches']} batches x {case['resident_batch_bytes']} B cap | {case['resident_timing_scope']}"
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
                "Resident sequence-input pipeline time (summed segments)"
                if batched
                else "Resident sequence-input pipeline time"
                if sequence_resident
                else "Resident packed-input pipeline time",
                "Same ASCII bases and record boundaries in native memory, staged in batches. "
                "Total sums aligned raw per-sample segment times, then summarizes; not a contiguous E2E.\n"
                "Parsing, decompression, uploads, downloads and JSON are outside timing; "
                "sketches are retained for all genomes and every pair is represented."
                if sequence_resident
                else "Same packed values, resident in native memory. No parsing, input transfers, downloads or JSON.\nReusable input/sketch/workspace/result buffers are preallocated; database/index construction is timed.",
            ),
            (
                "end_to_end_wall",
                "end_to_end",
                "Sequence-input files-to-JSON application time"
                if legacy_sequence
                else "Secondary: files-to-JSON application time",
                "Includes input processing, transfers, native metrics, JSON and teardown.\nThis is application cost, not a resident-compute comparison.",
            ),
        ):
            if legacy_sequence and key == "resident_total_wall":
                continue
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
            "Encode + hash + construct" if sequence_resident else "Hash + construct",
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
        if legacy_sequence:
            stage_labels = tuple(label for label, _ in APPLICATION_STAGES)
            stage_index = {key: i for i, (_, key) in enumerate(APPLICATION_STAGES)}
        values = np.full((len(stage_labels), len(rows), 3), np.nan)
        for column, row in enumerate(rows):
            for _, key in stages[row["implementation"]]:
                value = timing(row, key)
                values[stage_index[key], column] = [
                    value[field] for field in ("min_ms", "median_ms", "max_ms")
                ]
        panel_rows = math.ceil(len(stage_labels) / 3)
        fig, axes = plt.subplots(
            panel_rows,
            3,
            squeeze=False,
            sharey=True,
            figsize=(15, max(4.5, len(rows) * 0.8) * panel_rows + 1.5),
        )
        for stage, ax in enumerate(axes.flat):
            maximum = max(
                (high for high in values[stage, :, 2] if np.isfinite(high)), default=0
            )
            scale, unit = (1000, "s") if maximum >= 1000 else (1, "ms")
            limit = maximum / scale * 1.3 if maximum else 1
            for i, (low, median, high) in enumerate(values[stage] / scale):
                if np.isnan(median):
                    ax.text(limit * 0.03, i, "Not required", va="center", fontsize=11)
                    continue
                ax.barh(
                    i,
                    median,
                    height=0.55,
                    color=colors[i],
                    edgecolor="black",
                    hatch="//" if rows[i]["implementation"] == "rabbitsketch" else "",
                    xerr=[[median - low], [high - median]],
                    capsize=3,
                )
                ax.text(
                    high + limit * 0.025,
                    i,
                    pu.paper_text(f"{median:.3g}"),
                    va="center",
                    fontsize=pu.BAR_FONT_SIZE,
                )
            ax.set_xlim(0, limit)
            ax.set_ylim(len(rows) - 0.5, -0.5)
            ax.set_yticks(range(len(rows)), labels)
            ax.set_title(
                pu.paper_text(stage_labels[stage], bold=True), fontsize=14, pad=14
            )
            ax.set_xlabel(pu.paper_text(f"Wall time ({unit})"))
            ax.xaxis.set_major_locator(mpl.ticker.MaxNLocator(nbins=4))
            ax.xaxis.grid(True, alpha=pu.GRID_ALPHA)
            ax.set_axisbelow(True)
            ax.spines[["top", "right"]].set_visible(False)
        fig.suptitle(
            pu.paper_text(
                "Sequence-input application phases"
                if legacy_sequence
                else "Resident sequence pipeline stages (summed segments)"
                if batched
                else "Resident sequence pipeline stages"
                if sequence_resident
                else "Resident pipeline stages",
                bold=True,
            ),
            fontsize=18,
            y=0.99,
        )
        fig.text(
            0.5, 0.91, pu.paper_text(subtitle), ha="center", fontsize=pu.BAR_FONT_SIZE
        )
        if batched:
            note = (
                "Stage medians need not sum to the median total. "
                "Construct includes ASCII encoding and k-mer generation. "
                "Batched segments are summed per sample, then summarized; not a contiguous E2E. "
                + "Preparation: GPU row extraction; CPU bottom-k finalization."
            )
        elif sequence_resident:
            note = (
                "Isolated stages are not additive. Construct includes ASCII encoding and k-mer generation. "
                + "Preparation: GPU row extraction; CPU bottom-k finalization."
            )
        elif legacy_sequence:
            note = "Preparation includes file ingest and sketching; cuDDL also builds the database/index. Search includes output processing."
        else:
            note = "Isolated stages are not additive. Preparation: GPU row extraction; CPU bottom-k finalization."
        note = (
            "Compare pipelines within each panel: time scales differ between stages. Lower is faster.\n"
            "Bars: medians; whiskers: observed min-max, not confidence intervals. "
            "Exact timings in timings.csv.\n" + note
        )
        fig.text(0.5, 0.025, pu.paper_text(note), ha="center", fontsize=10)
        fig.tight_layout(rect=(0, 0.14, 1, 0.91), w_pad=2.5, h_pad=3)
        save(fig, output_dir / "stages")
    typer.echo(
        f"Plotted {len(rows)} {'sequence-input' if legacy_sequence else 'resident'} pipelines; exact values and clocks: {output_dir / 'timings.csv'}"
    )


if __name__ == "__main__":
    app = typer.Typer(pretty_exceptions_enable=False)
    app.command()(main)
    app()
