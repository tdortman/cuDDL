#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = ["jsonschema", "typer"]
# ///
"""Benchmark cuDDL against parallel BBTools DynamicDemiLog construction."""

import csv
import json
import os
import statistics
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Annotated, Any

import typer
from benchmark_schema import load_result, make_result, write_result

FIELDS = (
    "implementation",
    "count",
    "buckets",
    "threads",
    "trial",
    "seconds",
    "adds_per_second",
    "estimate",
    "bounded_estimate",
)
RESULT_NAME = "cuDDL versus BBTools DynamicDemiLog"


def run_csv(command: list[str]) -> list[dict[str, str]]:
    try:
        result = subprocess.run(command, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as error:
        message = error.stderr.strip() or error.stdout.strip() or str(error)
        raise RuntimeError(message) from error
    return list(csv.DictReader(result.stdout.splitlines(), fieldnames=FIELDS))


def java_heap(count: int) -> str:
    gib = 1 << 30
    # Each input occupies eight bytes; leave half again for GC and JVM objects.
    return f"-Xmx{max(1, (count * 12 + gib - 1) // gib)}g"


def powers(first: int, last: int) -> list[int]:
    return [1 << exponent for exponent in range(first, last + 1)]


def measurement(row: dict[str, str]) -> dict[str, object]:
    seconds = float(row["seconds"])
    case = {
        "count": int(row["count"]),
        "buckets": int(row["buckets"]),
        "trial": int(row["trial"]),
    }
    if row["threads"]:
        case["threads"] = int(row["threads"])
    return {
        "implementation": {"name": row["implementation"]},
        "case": case,
        "metrics": {
            "seconds": seconds,
            "adds_per_second": float(row["adds_per_second"]),
            "estimate": int(float(row["estimate"])),
            "bounded_estimate": int(float(row["bounded_estimate"])),
        },
        "timings": {
            "construction": {
                "samples": 1,
                "median_ms": seconds * 1000.0,
                "source": "per-trial benchmark output",
            }
        },
    }


def main(
    min_power: Annotated[int, typer.Option(min=1)] = 8,
    max_power: Annotated[int, typer.Option(min=1)] = 28,
    threads: Annotated[int, typer.Option(min=1, help="BBTools worker count")] = (
        os.cpu_count() or 1
    ),
    warmup: Annotated[int, typer.Option(min=0)] = 3,
    trials: Annotated[int, typer.Option(min=1)] = 10,
    build_dir: Annotated[Path, typer.Option()] = Path("build"),
    output: Annotated[Path | None, typer.Option()] = None,
) -> None:
    if min_power > max_power:
        raise typer.BadParameter("min-power must not exceed max-power")

    root = Path(__file__).resolve().parent.parent
    build_dir = build_dir if build_dir.is_absolute() else root / build_dir
    jar = root / "subprojects/bbmap/bbtools.jar"
    source = root / "benchmarks/BBToolsDynamicDemiLogBenchmark.java"
    gpu = build_dir / "benchmarks/cuddl-dynamic-demilog-benchmark"
    for path in (jar, source, gpu):
        if not path.exists():
            raise typer.BadParameter(
                f"missing {path}; run `meson compile -C {build_dir}`"
            )

    measurements: list[dict[str, Any]] = []
    with tempfile.TemporaryDirectory() as classes:
        subprocess.run(
            ["javac", "-cp", str(jar), "-d", classes, str(source)], check=True
        )
        for count in powers(min_power, max_power):
            java_rows = run_csv(
                [
                    "java",
                    java_heap(count),
                    "-cp",
                    f"{classes}:{jar}",
                    "BBToolsDynamicDemiLogBenchmark",
                    str(count),
                    "2048",
                    str(threads),
                    str(warmup),
                    str(trials),
                ]
            )
            measurements.extend(measurement(row) for row in java_rows)
            gpu_output = Path(classes) / f"cuddl-{count}.json"
            subprocess.run(
                [
                    str(gpu),
                    "--count",
                    str(count),
                    "--warmup",
                    str(warmup),
                    "--runs",
                    str(trials),
                    "--output",
                    str(gpu_output),
                ],
                check=True,
            )
            gpu_result = load_result(gpu_output, operation="dynamic_demilog")
            measurements.extend(gpu_result["measurements"])

    result = make_result(
        name=RESULT_NAME,
        operation="dynamic_demilog",
        scope="end_to_end",
        datasets={},
        system=gpu_result["system"],
        measurements=measurements,
    )
    if output is not None:
        write_result(output, result)
    else:
        json.dump(result, sys.stdout, indent=2, allow_nan=False)
        sys.stdout.write("\n")

    largest = 1 << max_power
    medians = {
        name: statistics.median(
            float(item["metrics"]["adds_per_second"])
            for item in measurements
            if item["implementation"]["name"] == name
            and item["case"]["count"] == largest
        )
        for name in ("bbtools", "cuddl")
    }
    typer.echo(
        f"Median speedup at 2^{max_power}: "
        f"{medians['cuddl'] / medians['bbtools']:.2f}x",
        err=True,
    )


if __name__ == "__main__":
    typer.run(main)
