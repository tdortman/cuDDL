#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = ["typer"]
# ///
"""Side-by-side cost of measured CPU (SimdSketch) and GPU (cuDDL) all-to-all runs.

Pairs every cuddl-distributed-all-to-all-benchmark report with the
`scripts/cpu-baseline.sh` measurement of the same genome count and prints their
measured wall times, node-hours and output bytes. Nothing is fitted or scaled:
a report without a CPU measurement of its genome count is an error. cuDDL's
sketch phase includes reading and decompressing the genomes, like SimdSketch's.
"""

import json
from pathlib import Path
from typing import Annotated

import typer


def main(
    cpu: Annotated[
        Path, typer.Argument(exists=True, help="JSON lines from scripts/cpu-baseline.sh")
    ],
    gpu_reports: Annotated[
        list[Path],
        typer.Argument(
            exists=True, help="cuddl-distributed-all-to-all-benchmark reports"
        ),
    ],
) -> None:
    measured = {}
    for line in cpu.read_text().splitlines():
        if line.strip():
            row = json.loads(line)
            measured[row["genomes"]] = row
    header = [
        "run",
        "genomes",
        "nodes",
        "output",
        "sketch s",
        "compare s",
        "total s",
        "node-h",
        "output bytes",
    ]
    table = []
    for path in gpu_reports:
        report = json.loads(path.read_text())
        row = measured.get(report["genomes"])
        if row is None:
            raise typer.BadParameter(
                f"no CPU measurement of {report['genomes']} genomes for {path}"
            )
        cpu_total = row["sketch_s"] + row["compare_s"]
        cpu_line = [
            f"SimdSketch, {row['threads']} threads",
            row["genomes"],
            1,
            "every pair",
            f"{row['sketch_s']:.1f}",
            f"{row['compare_s']:.1f}",
            f"{cpu_total:.1f}",
            f"{cpu_total / 3600:.4f}",
            f"{row['compare_bytes'] / 1e9:.2f} GB",
        ]
        if cpu_line not in table:
            table.append(cpu_line)
        wall = report["phases_wall_ms"]
        total = report["wall_ms"] / 1000
        threshold = report["minimum_matches"]
        table.append(
            [
                f"cuDDL, {path.stem}",
                report["genomes"],
                report["nodes"],
                f">= {threshold} matches" if threshold else "every pair",
                f"{wall['sketch'] / 1000:.1f}",
                f"{(total * 1000 - wall['sketch']) / 1000:.1f}",
                f"{total:.1f}",
                f"{report['nodes'] * total / 3600:.4f}",
                f"{report['bytes_written_total'] / 1e9:.2f} GB",
            ]
        )
    widths = [max(len(str(c)) for c in col) for col in zip(header, *table)]
    for line in [header, *table]:
        typer.echo("  ".join(str(c).rjust(w) for c, w in zip(line, widths)))
    typer.echo(
        "cuDDL 'compare s' is the wall time after sketching: allgather, build, index, search,"
        " downloads and writes."
    )


if __name__ == "__main__":
    typer.run(main)
