#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = ["jsonschema", "typer"]
# ///
"""Combine HLL NVBench timings with direct accuracy JSON."""

from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import Annotated, Any

import typer
from benchmark_schema import load_result, make_result, write_result

NVBENCH_REQUIRED = {"Benchmark", "Iters", "Median GPU Time"}
METRIC_FIELDS = {
    "Exact": "exact",
    "Estimate": "estimate",
    "Exact Similarity": "exact_similarity",
    "Similarity": "similarity",
}

app = typer.Typer(
    help="Normalize HLL comparison CSVs into a validated benchmark JSON.",
    no_args_is_help=True,
)


def read_rows(path: Path, required: set[str], label: str) -> list[dict[str, str]]:
    try:
        with path.open(newline="") as source:
            reader = csv.DictReader(source)
            fields = set(reader.fieldnames or ())
            missing = sorted(required - fields)
            if missing:
                raise typer.BadParameter(
                    f"{label} is missing columns: {', '.join(missing)}"
                )
            return [
                row
                for row in reader
                if any(
                    value and value.strip()
                    for value in row.values()
                    if value is not None
                )
            ]
    except OSError as error:
        raise typer.BadParameter(f"cannot read {label} {path}: {error}") from error


def number(
    row: dict[str, str], field: str, label: str, *, integer: bool = False
) -> int | float:
    raw = row.get(field, "").strip()
    try:
        value = float(raw)
    except ValueError as error:
        raise typer.BadParameter(f"{label} has invalid {field}: {raw!r}") from error
    if not math.isfinite(value):
        raise typer.BadParameter(f"{label} has non-finite {field}: {raw!r}")
    if integer:
        if not value.is_integer():
            raise typer.BadParameter(f"{label} has non-integer {field}: {raw!r}")
        return int(value)
    return value


def timing(row: dict[str, str], label: str) -> dict[str, Any]:
    median = number(row, "Median GPU Time", label)
    minimum = (
        number(row, "Min GPU Time", label)
        if row.get("Min GPU Time", "").strip()
        else median
    )
    maximum = (
        number(row, "Max GPU Time", label)
        if row.get("Max GPU Time", "").strip()
        else median
    )
    if minimum < 0 or median < 0 or maximum < 0 or minimum > median or median > maximum:
        raise typer.BadParameter(f"{label} has inconsistent GPU time statistics")
    samples = 1
    if row.get("Samples", "").strip():
        samples = int(number(row, "Samples", label, integer=True))
        if samples < 1:
            raise typer.BadParameter(f"{label} has invalid Samples: {samples}")
    return {
        "samples": samples,
        "min_ms": minimum * 1e3,
        "median_ms": median * 1e3,
        "max_ms": maximum * 1e3,
        "source": "nvbench",
    }


def normalize_nvbench(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    measurements: list[dict[str, Any]] = []
    for index, row in enumerate(rows, start=2):
        label = f"NVBench row {index}"
        benchmark = row["Benchmark"].strip()
        try:
            estimator, phase = benchmark.rsplit("_", 1)
        except ValueError as error:
            raise typer.BadParameter(
                f"{label} has invalid Benchmark: {benchmark!r}"
            ) from error
        if phase not in {"construction", "cardinality", "similarity"} or not estimator:
            raise typer.BadParameter(f"{label} has invalid Benchmark: {benchmark!r}")
        items = number(row, "Iters", label, integer=True)
        if items < 1:
            raise typer.BadParameter(f"{label} has invalid Iters: {items}")
        metrics = {
            output: number(row, source, label)
            for source, output in METRIC_FIELDS.items()
            if row.get(source, "").strip()
        }
        measurement: dict[str, Any] = {
            "implementation": {"name": estimator},
            "case": {"phase": phase, "items": items, "benchmark": benchmark},
            "timings": {"gpu": timing(row, label)},
        }
        if metrics:
            measurement["metrics"] = metrics
        measurements.append(measurement)
    return measurements


def normalize(
    nvbench_csv: Annotated[
        Path, typer.Argument(exists=True, dir_okay=False, help="HLL NVBench CSV")
    ],
    accuracy_json: Annotated[
        Path, typer.Argument(exists=True, dir_okay=False, help="HLL accuracy JSON")
    ],
    output: Annotated[
        Path, typer.Option("--output", "-o", dir_okay=False, help="Result JSON path")
    ] = Path("results/hll-comparison.json"),
) -> None:
    """Combine raw HLL timings and direct accuracy measurements."""
    nvbench_rows = read_rows(nvbench_csv, NVBENCH_REQUIRED, "NVBench CSV")
    if not nvbench_rows:
        raise typer.BadParameter("NVBench CSV has no rows")
    accuracy = load_result(accuracy_json, operation="hll_comparison")
    result = make_result(
        name="cuDDL vs cuco HLL",
        operation="hll_comparison",
        scope="kernel",
        system=accuracy["system"],
        measurements=[
            *normalize_nvbench(nvbench_rows),
            *accuracy["measurements"],
        ],
    )
    write_result(output, result)
    typer.echo(f"Wrote {output}")


if __name__ == "__main__":
    typer.run(normalize)
