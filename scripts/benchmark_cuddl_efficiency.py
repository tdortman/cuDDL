#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = ["typer"]
# ///
"""Run correctness-checked NVBench A/B experiments"""

import json
import subprocess
from enum import Enum
from pathlib import Path
from typing import Annotated

import typer


class Suite(str, Enum):
    all = "all"
    checks = "checks"
    construction = "construction"
    genome_sizes = "genome-sizes"
    size_controls = "size-controls"


def main(
    suite: Annotated[Suite, typer.Argument()] = Suite.all,
    samples: Annotated[int, typer.Option(min=1)] = 300,
    output: Annotated[Path, typer.Option()] = Path("results/efficiency"),
    binary: Annotated[Path, typer.Option()] = Path(
        "build/benchmarks/cuddl-efficiency-benchmark"
    ),
) -> None:
    output.mkdir(parents=True, exist_ok=True)

    def run(
        name: str, benchmark: str, axes: dict[str, str], count: int = samples
    ) -> None:
        command = [
            str(binary),
            "--benchmark",
            benchmark,
            "--stopping-criterion",
            "sample-count",
            "--target-samples",
            str(count),
            "--min-samples",
            str(count),
        ]
        for key, value in axes.items():
            command += ["--axis", f"{key}={value}"]
        report = output / f"{name}.json"
        command += ["--csv", str(output / f"{name}.csv"), "--json", str(report)]
        print(f"Running {name}", flush=True)
        with (output / f"{name}.log").open("w") as log:
            subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
        data = json.loads(report.read_text())
        states = [s for b in data["benchmarks"] for s in b["states"]]
        skipped = [s for s in states if s.get("is_skipped")]
        if skipped:
            raise RuntimeError(
                f"{name}: {len(skipped)} failed/skipped states; inspect {report}"
            )
        print(
            f"  {len(states)} states passed correctness checks and completed",
            flush=True,
        )

    if suite in (Suite.all, Suite.checks):
        run(
            "check-construction",
            "construction_efficiency",
            {
                "Buckets": "[2048,8192]",
                "Items": "[0,7,3073,65535,65536]",
                "Input": "[random,repeated]",
                "FloorRounds": "[0,8,32]",
                "Misaligned": "1",
            },
            1,
        )
    if suite == Suite.genome_sizes:
        for name, rounds in (("forward", "[0,32]"), ("reverse", "[32,0]")):
            run(
                f"genome-sizes-{name}",
                "construction_efficiency",
                {
                    "Input": "[worm,chr14]",
                    "Items": "[1048576,4194304,8388608,12582912,16777216,25165824,33554432,50331648,67108864]",
                    "StartPercent": "[0,50,100]",
                    "FloorRounds": rounds,
                },
            )
    if suite == Suite.size_controls:
        for name, rounds in (("forward", "[0,32]"), ("reverse", "[32,0]")):
            run(
                f"size-controls-{name}",
                "construction_efficiency",
                {
                    "Input": "[random,duplicates]",
                    "Items": "[25165824,33554432,50331648]",
                    "FloorRounds": rounds,
                },
            )
    if suite in (Suite.all, Suite.construction):
        run("construction-random", "construction_efficiency", {"Input": "random"})
        run(
            "construction-duplicates",
            "construction_efficiency",
            {
                "Input": "duplicates",
                "Items": "[1048576,16777216]",
            },
        )
        run(
            "construction-fasta",
            "construction_efficiency",
            {
                "Input": "[ecoli,worm,chr14]",
                "Items": "0",
            },
        )


if __name__ == "__main__":
    typer.run(main)
