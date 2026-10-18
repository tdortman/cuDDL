#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = ["typer"]
# ///
"""SimdSketch all-to-all on one node: sketch a hashed sample of N genomes, then compare every
pair, with the settings the micro-comparison uses (k=25, 2048 buckets of 8 bits).

SimdSketch has no thresholded output; compare writes one 32-bit similarity per pair. Prints one
JSON line per genome count. scripts/simdsketch_nodes.py runs the same comparison across several
nodes. Inside a Slurm job, a missing SimdSketch binary is built on the node first.
"""

import hashlib
import json
import os
import socket
from pathlib import Path
from typing import Annotated

import typer
from simdsketch_nodes import SIMDSKETCH, build_simdsketch, timed
from slurm_utils import MGPU, genome_files


def main(
    genomes: Annotated[Path, typer.Option(exists=True, file_okay=False)] = MGPU
    / "genomes-30k",
    sizes: Annotated[
        list[int] | None,
        typer.Option(
            help="Genome counts, one run each [default: 1000 3000 10000 20000 30000]"
        ),
    ] = None,
    threads: Annotated[int, typer.Option(help="SimdSketch threads")] = len(
        os.sched_getaffinity(0)
    ),
    scratch: Annotated[
        Path | None,
        typer.Option(
            help="Working directory [default: /localscratch/$SLURM_JOB_ID/cpu-baseline]"
        ),
    ] = None,
    simdsketch: Annotated[
        Path, typer.Option(help="simdsketch-benchmark binary")
    ] = SIMDSKETCH,
) -> None:
    sizes = sizes or [1000, 3000, 10000, 20000, 30000]
    scratch = scratch or Path(
        f"/localscratch/{os.environ.get('SLURM_JOB_ID', 'local')}/cpu-baseline"
    )
    if "SLURM_JOB_ID" in os.environ and not simdsketch.exists():
        build_simdsketch()
    scratch.mkdir(parents=True, exist_ok=True)
    # The genomes whose hashed file names come first, so each size keeps the collection's mix
    # and is a subset of the larger ones.
    paths = sorted(
        genome_files(genomes), key=lambda p: hashlib.blake2b(p.name.encode()).digest()
    )
    sketch, pairs = scratch / "sketch.sk", scratch / "pairs.f32"
    for n in sizes:
        if len(paths) < n:
            raise SystemExit(f"Error: fewer than {n} genomes in {genomes}")
        listing = scratch / f"genomes-{n}.txt"
        listing.write_text("".join(f"{p}\n" for p in paths[:n]))
        sketch_s = timed(
            simdsketch,
            "-j",
            threads,
            "sketch",
            "-k",
            "25",
            "-s",
            "2048",
            "-b",
            "8",
            "--output",
            sketch,
            listing,
        )
        compare_s = timed(
            simdsketch,
            "-j",
            threads,
            "compare",
            "--references",
            sketch,
            "--output",
            pairs,
        )
        print(
            json.dumps(
                {
                    "tool": "simdsketch",
                    "genomes": n,
                    "threads": threads,
                    "input_bytes": sum(p.stat().st_size for p in paths[:n]),
                    "sketch_s": round(sketch_s, 3),
                    "compare_s": round(compare_s, 3),
                    "compare_bytes": pairs.stat().st_size,
                    "host": socket.gethostname().split(".")[0],
                }
            ),
            flush=True,
        )
        sketch.unlink()
        pairs.unlink()


if __name__ == "__main__":
    typer.run(main)
