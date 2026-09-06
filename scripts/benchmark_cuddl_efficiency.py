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
    floor_sweep = "floor-sweep"


def parse_sizes(value: str) -> list[int]:
    try:
        sizes = [int(n) for n in value.split(",")]
    except ValueError as error:
        raise typer.BadParameter("Sizes must be comma-separated integers") from error
    if any(n < 0 for n in sizes) or len(set(sizes)) != len(sizes):
        raise typer.BadParameter("Sizes must be nonnegative and unique")
    return sizes


def main(
    suite: Annotated[Suite, typer.Argument()] = Suite.all,
    samples: Annotated[int, typer.Option(min=1)] = 300,
    timeout: Annotated[
        float,
        typer.Option(
            min=0.001, help="NVBench time limit per configuration, in seconds."
        ),
    ] = 15.0,
    output: Annotated[Path, typer.Option()] = Path("results/efficiency"),
    genome: Annotated[
        list[Path] | None,
        typer.Option(
            "--fastx",
            "--genome",
            exists=True,
            dir_okay=False,
            readable=True,
            help="FASTA/FASTQ file; repeat for multiple inputs. Replaces built-in sweep inputs.",
        ),
    ] = None,
    label: Annotated[
        list[str] | None,
        typer.Option(
            help="Plot label for each --fastx, in the same order; repeat per input."
        ),
    ] = None,
    random: Annotated[
        bool,
        typer.Option(
            "--random", help="Include random input alongside the FASTX files."
        ),
    ] = False,
    items: Annotated[
        list[int] | None,
        typer.Option(
            min=0,
            help="K-mer count; repeat for a size sweep. Zero uses the full genome.",
        ),
    ] = None,
    fastx_items: Annotated[
        list[str] | None,
        typer.Option(
            help="Comma-separated sizes per --fastx, in file order; overrides --items."
        ),
    ] = None,
    random_items: Annotated[
        str | None,
        typer.Option(
            help="Comma-separated positive sizes for random input; enables --random and overrides --items."
        ),
    ] = None,
    device: Annotated[int, typer.Option(min=0, help="NVBench CUDA device index")] = 0,
    binary: Annotated[Path, typer.Option()] = Path(
        "build/benchmarks/cuddl-efficiency-benchmark"
    ),
) -> None:
    if (
        genome or items or label or random or fastx_items or random_items
    ) and suite != Suite.floor_sweep:
        raise typer.BadParameter("Input and size options require floor-sweep")
    if label is not None and (
        len(label) != len(genome or []) or any(not x.strip() for x in label)
    ):
        raise typer.BadParameter("Provide one nonempty --label per --fastx")
    paths = [p.resolve() for p in genome or []]
    if len(set(paths)) != len(paths):
        raise typer.BadParameter("Duplicate genome paths")
    if any(any(c in str(p) for c in ",[]:\n\r") for p in paths):
        raise typer.BadParameter(
            "Genome paths cannot contain commas, brackets, colons, or newlines"
        )
    sizes = (
        items
        if items is not None
        else [
            65536,
            262144,
            1048576,
            4194304,
            8388608,
            16777216,
            25165824,
            33554432,
            67108864,
        ]
    )
    if len(set(sizes)) != len(sizes):
        raise typer.BadParameter("Duplicate --items values")
    if fastx_items is not None and len(fastx_items) != len(paths):
        raise typer.BadParameter("Provide one --fastx-items list per --fastx")
    genome_sizes = (
        [parse_sizes(value) for value in fastx_items]
        if fastx_items is not None
        else [sizes] * len(paths)
    )
    random_sizes = (
        parse_sizes(random_items)
        if random_items is not None
        else [n for n in sizes if n > 0]
    )
    random = random or random_items is not None
    if random and (not random_sizes or 0 in random_sizes):
        raise typer.BadParameter("Random input requires positive sizes")
    if 0 in sizes and not paths:
        raise typer.BadParameter("--items 0 requires --genome")
    if suite == Suite.floor_sweep and any(output.glob("floor-sweep-*")):
        raise typer.BadParameter(
            "Output already contains a sweep; choose a fresh --output directory"
        )
    output.mkdir(parents=True, exist_ok=True)

    def run(
        name: str, benchmark: str, axes: dict[str, str], count: int = samples
    ) -> None:
        command = [
            str(binary),
            "--devices",
            str(device),
            "--benchmark",
            benchmark,
            "--stopping-criterion",
            "sample-count",
            "--target-samples",
            str(count),
            "--min-samples",
            str(count),
            "--timeout",
            str(timeout),
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
        if not states or skipped:
            raise RuntimeError(
                f"{name}: {len(skipped)} failed/skipped states; inspect {report}"
            )
        print(
            f"  {len(states)} states passed correctness checks and completed",
            flush=True,
        )

    if suite == Suite.floor_sweep:
        (output / "floor-sweep-labels.json").write_text(
            json.dumps(
                {
                    f"fasta={path}": name
                    for path, name in zip(paths, label or [p.name for p in paths])
                },
                indent=2,
            )
            + "\n"
        )
        rounds = [0, 1, 2, 4, 8, 12, 16, 24, 32, 48, 64, 96, 128]
        inputs_to_run = [
            (f"genome-{i}", f"fasta={path}", "[0,50,100]", input_sizes)
            for i, (path, input_sizes) in enumerate(zip(paths, genome_sizes))
        ]
        if not paths and not random:
            inputs_to_run = [
                ("random", "random", "0", random_sizes),
                ("genomes", "[worm,chr14]", "[0,50,100]", sizes),
            ]
        if random:
            inputs_to_run.append(("random", "random", "0", random_sizes))
        for order in ("forward", "reverse"):
            for name, inputs, windows, input_sizes in inputs_to_run:
                run(
                    f"floor-sweep-{order}-{name}",
                    "construction_efficiency",
                    {
                        "Buckets": "2048",
                        "Items": "[" + ",".join(map(str, input_sizes)) + "]",
                        "Input": inputs,
                        "StartPercent": windows,
                        "FloorRounds": "[" + ",".join(map(str, rounds)) + "]",
                    },
                )
            rounds.reverse()

    if suite in (Suite.all, Suite.checks):
        run(
            "check-construction",
            "construction_efficiency",
            {
                "Buckets": "[2048,8192]",
                "Items": "[0,7,3073,65535,65536]",
                "Input": "[random,repeated]",
                "FloorRounds": "[0,1,2,4,8,12,16,24,32,48,64,96,128]",
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
