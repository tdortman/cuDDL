#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = ["typer"]
# ///
"""Strong (100k genomes) and weak (25k genomes per node) scaling of the end-to-end pipeline, one
A40 per node.

Weak runs below 100k read genomes-<N>k-sample, a hashed sample of genomes-100k
(snapshot_genomes.py), so every size has the same mix of genomes. Each configuration
runs 4 times: run r0 reads part of its input from Lustre for the first time, r1-r3 find it in
the page cache. After each of r1-r3, SimdSketch
runs on the same nodes and cores: the nodes are shared with other jobs, and alternating the tools
exposes both to the same load. With 64 cores per GPU, the cpu backend sketched 100k genomes in
21.7 s against 29.5-30.3 s for hybrid, whose GPU-side FASTA normalisation becomes the bottleneck.
Submit from the project root:

    sbatch -M mogonki -A ki-gpu4science -p a40 -N 4 --ntasks-per-node=1 -c 64 --gres=gpu:1 \\
      --mem=400G -o logs/%x-%j.log -e logs/%x-%j.log \\
      --wrap 'source ./.envrc && uv run --offline scripts/sketch_scaling.py'
"""

import json
import os
from typing import Annotated

import typer
from simdsketch_nodes import simdsketch_nodes
from slurm_utils import MGPU, per_node, run, scratch


def main(
    minimum: Annotated[
        int,
        typer.Option(
            help="--minimum-matches; 0 reports and writes every pair, as SimdSketch does"
        ),
    ] = 20,
    strong: Annotated[bool, typer.Option(help="Strong-scaling configurations")] = True,
    simdsketch: Annotated[
        bool, typer.Option(help="SimdSketch after runs r1-r3")
    ] = True,
) -> None:
    S = scratch()
    nodes = int(os.environ["SLURM_JOB_NUM_NODES"])
    threads = int(os.environ["SLURM_CPUS_PER_TASK"])
    run("ninja", "-C", "build", "build.ninja")
    run(
        "meson",
        "compile",
        "-C",
        "build",
        "-j",
        str(threads),
        "cuddl-distributed-all-to-all-benchmark",
    )
    per_node(nodes, "mkdir", "-p", S)

    def configuration(n: int, size: str, scaling: str, snapshot: str) -> None:
        genomes = MGPU / f"genomes-{snapshot}"
        name = f"results/mogon-a40-sketch-{scaling}-{size}-{n}x1"
        for rep in range(4):
            run(
                "mpirun",
                "-n",
                str(n),
                "--map-by",
                "ppr:1:node",
                "--bind-to",
                "none",
                "-x",
                "LD_LIBRARY_PATH",
                "-x",
                "PATH",
                "./build/benchmarks/cuddl-distributed-all-to-all-benchmark",
                genomes,
                "--minimum-matches",
                str(minimum),
                "--decompression",
                "cpu",
                "--output-dir",
                S,
                "-o",
                f"{name}-m{minimum}-r{rep}.json",
                keep=r"genomes=|Error",
            )
            if simdsketch and rep > 0:
                report = simdsketch_nodes(genomes, n, threads)
                with open(f"{name}-simdsketch-r{rep}.json", "w") as f:
                    f.write(json.dumps(report) + "\n")

    for n in range(1, nodes + 1):
        if strong:
            configuration(n, "100k", "strong", "100k")
        if n < 4:
            configuration(n, f"{25 * n}k", "weak", f"{25 * n}k-sample")


if __name__ == "__main__":
    typer.run(main)
