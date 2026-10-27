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
    blacklist = "blacklist"


def main(
    suite: Annotated[Suite, typer.Argument()] = Suite.blacklist,
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
            help="FASTA/FASTQ file; repeat for multiple inputs. Replaces the bundled genomes.",
        ),
    ] = None,
    label: Annotated[
        list[str] | None,
        typer.Option(
            help="Plot label for each --fastx, in the same order; repeat per input."
        ),
    ] = None,
    blacklist: Annotated[
        Path | None,
        typer.Option(
            exists=True,
            dir_okay=False,
            readable=True,
            help="BBTools DDL blacklist for the blacklist suite",
        ),
    ] = None,
    device: Annotated[int, typer.Option(min=0, help="NVBench CUDA device index")] = 0,
    binary: Annotated[Path, typer.Option()] = Path(
        "build/benchmarks/cuddl-efficiency-benchmark"
    ),
) -> None:
    genome = genome or [
        Path("data/genomes") / name
        for name in ("ecoli_k12_mg1655.fna", "WBcel235.fna", "chr14.fna")
    ]
    blacklist = blacklist or Path(
        "subprojects/bbmap/resources/refseqGenomeDDLBlacklist_k25e5b65536_fused.fa.gz"
    )
    if not blacklist.is_file() or any(not p.is_file() for p in genome):
        raise typer.BadParameter(
            "Missing input: supply --fastx and --blacklist or fetch the bundled data"
        )
    if any(c in str(blacklist) for c in ",[]:\n\r"):
        raise typer.BadParameter(
            "Blacklist path cannot contain commas, brackets, colons, or newlines"
        )
    if any(output.glob("blacklist-*")):
        raise typer.BadParameter(
            "Output already contains a blacklist comparison; choose a fresh --output"
        )
    if label is not None and (
        len(label) != len(genome) or any(not x.strip() for x in label)
    ):
        raise typer.BadParameter("Provide one nonempty --label per --fastx")
    paths = [p.resolve() for p in genome]
    if len(set(paths)) != len(paths):
        raise typer.BadParameter("Duplicate genome paths")
    if any(any(c in str(p) for c in ",[]:\n\r") for p in paths):
        raise typer.BadParameter(
            "Genome paths cannot contain commas, brackets, colons, or newlines"
        )
    output.mkdir(parents=True, exist_ok=True)

    def run(name: str, benchmark: str, axes: dict[str, str]) -> None:
        command = [
            str(binary),
            "--devices",
            str(device),
            "--benchmark",
            benchmark,
            "--stopping-criterion",
            "sample-count",
            "--target-samples",
            str(samples),
            "--min-samples",
            str(samples),
            "--timeout",
            str(timeout),
            "--no-batch",
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

    variants = [("unfiltered", None), ("bbtools", str(blacklist.resolve()))]
    reports = []
    for order in ("forward", "reverse"):
        for i, (path, name) in enumerate(zip(paths, label or [p.name for p in paths])):
            for variant, source in variants:
                report_name = f"blacklist-{order}-{i}-{variant}"
                run(
                    report_name,
                    "sequence_construction",
                    {
                        "Path": str(path),
                        "K": "25",
                        "Offset": "0",
                        "BlocksPerSM": "0",
                        **({"Blacklist": source} if source is not None else {}),
                    },
                )
                reports.append(
                    {
                        "label": name,
                        "path": str(path),
                        "variant": variant,
                        "order": order,
                        "report": report_name + ".csv",
                    }
                )
        variants.reverse()
    (output / "blacklist-comparison.json").write_text(
        json.dumps(
            {
                "blacklist": str(blacklist.resolve()),
                "samples": samples,
                "scope": "resident GPU sequence construction; excludes parsing and upload",
                "reports": reports,
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    typer.run(main)
