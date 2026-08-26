#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = ["typer"]
# ///
"""Benchmark cuDDL against parallel BBTools DynamicDemiLog construction."""

import csv
import os
import statistics
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Annotated

import typer

FIELDS = (
    "implementation",
    "count",
    "buckets",
    "threads",
    "trial",
    "seconds",
    "adds_per_second",
    "estimate",
)


def run(command: list[str]) -> list[dict[str, str]]:
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

    rows: list[dict[str, str]] = []
    with tempfile.TemporaryDirectory() as classes:
        subprocess.run(
            ["javac", "-cp", str(jar), "-d", classes, str(source)], check=True
        )
        for count in powers(min_power, max_power):
            rows += run(
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
            rows += run(
                [
                    str(gpu),
                    "--count",
                    str(count),
                    "--warmup",
                    str(warmup),
                    "--runs",
                    str(trials),
                ]
            )

    if output:
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("w", newline="") as destination:
            writer = csv.DictWriter(destination, fieldnames=FIELDS)
            writer.writeheader()
            writer.writerows(rows)
    else:
        writer = csv.DictWriter(sys.stdout, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    largest = str(1 << max_power)
    medians = {
        name: statistics.median(
            float(row["adds_per_second"])
            for row in rows
            if row["implementation"] == name and row["count"] == largest
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
