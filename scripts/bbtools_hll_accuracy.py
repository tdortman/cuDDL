#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = ["fastjsonschema", "typer"]
# ///
"""Cardinality accuracy of BBTools DynamicDemiLog on the seeded sets of cuddl-hll-accuracy.

Runs benchmarks/BBToolsHllAccuracy.java against subprojects/bbmap/bbtools.jar and writes its
estimates with the metrics cuddl-hll-accuracy records, under the implementation name `bbtools`.
"""

import subprocess
import tempfile
from pathlib import Path
from typing import Annotated

import typer
from benchmark_schema import make_result, write_result


def main(
    output: Annotated[Path, typer.Option()] = Path("results/hll-accuracy-bbtools.json"),
    min_power: Annotated[int, typer.Option(min=1)] = 8,
    max_power: Annotated[int, typer.Option(min=1, max=40)] = 24,
    trials: Annotated[int, typer.Option(min=1)] = 32,
    seed: Annotated[int, typer.Option(help="Root seed of cuddl-hll-accuracy")] = 42,
    buckets: Annotated[int, typer.Option(min=1)] = 2048,
) -> None:
    if min_power > max_power:
        raise typer.BadParameter("min-power must not exceed max-power")
    root = Path(__file__).resolve().parent.parent
    jar = root / "subprojects/bbmap/bbtools.jar"
    source = root / "benchmarks/BBToolsHllAccuracy.java"
    with tempfile.TemporaryDirectory() as classes:
        subprocess.run(
            ["javac", "-cp", str(jar), "-d", classes, str(source)], check=True
        )
        lines = subprocess.run(
            [
                "java",
                "-cp",
                f"{classes}:{jar}",
                "BBToolsHllAccuracy",
                str(min_power),
                str(max_power),
                str(trials),
                str(seed),
                str(buckets),
            ],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.split()
    measurements = []
    for line in lines:
        items, trial, estimate = map(int, line.split(","))
        signed_error = (estimate - items) / items
        measurements.append(
            {
                "implementation": {"name": "bbtools"},
                "case": {"phase": "accuracy", "items": items, "trial": trial},
                "metrics": {
                    "exact": items,
                    "estimate": float(estimate),
                    "signed_error": signed_error,
                    "absolute_error": abs(signed_error),
                },
            }
        )
    write_result(
        output,
        make_result(
            name="HLL cardinality accuracy",
            operation="hll_comparison",
            scope="end_to_end",
            measurements=measurements,
        ),
    )


if __name__ == "__main__":
    typer.run(main)
