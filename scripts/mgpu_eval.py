#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = ["typer"]
# ///
"""Multi-GPU evaluation inside a Slurm job with one rank per usable GPU.

Runs strong scaling with a threshold sweep, weak scaling, then the end-to-end pipeline (cold,
then warm at every node count), then SimdSketch at every node count. Builds the snapshot's
database if missing. Writes results/mogon-<gpu>-TAG-*.json. Submit from the project root:

    sbatch -M mogonki -A ki-gpu4science -p a40 -N 2 --ntasks-per-node=8 -c 16 --gres=gpu:8 \\
      --mem=512G -o logs/%x-%j.log -e logs/%x-%j.log \\
      --wrap 'source ./.envrc && uv run --offline scripts/mgpu_eval.py --size 30k'
"""

import json
import os
import shutil
import subprocess
from pathlib import Path
from typing import Annotated

import typer
from simdsketch_nodes import simdsketch_nodes
from slurm_utils import MGPU, OWN_GPU, gpu_tag, per_node, run, scratch, usable_gpus

BENCH = "./build/benchmarks/cuddl-multi-gpu-all-to-all-benchmark"
E2E = "./build/benchmarks/cuddl-distributed-all-to-all-benchmark"
KEEP = r"^rank 0|verification|load_balance|threshold|genomes=|Error"


def infiniband_ports() -> str:
    """Active InfiniBand ports for UCX: A100 nodes also have two Ethernet ports that are down."""
    devices = subprocess.run(
        ["ibstat", "-l"], check=True, capture_output=True, text=True
    ).stdout
    ports = []
    for device in devices.split():
        state = subprocess.run(
            ["ibstat", device, "1"], check=True, capture_output=True, text=True
        )
        if "State: Active" in state.stdout and "Link layer: InfiniBand" in state.stdout:
            ports.append(f"{device}:1")
    return ",".join(ports)


def main(
    size: Annotated[
        str, typer.Option(help="Snapshot data/refseq211/mgpu/genomes-SIZE")
    ] = "30k",
    tag: Annotated[str, typer.Option(help="Result name tag")] = "eval",
    strong: Annotated[
        bool, typer.Option(help="Strong scaling with a threshold sweep")
    ] = True,
    weak: Annotated[bool, typer.Option(help="Weak scaling")] = True,
    e2e: Annotated[bool, typer.Option(help="End-to-end pipeline runs")] = True,
    simdsketch: Annotated[
        bool, typer.Option(help="SimdSketch at every node count")
    ] = True,
    e2e_output: Annotated[
        Path | None,
        typer.Option(
            help="Directory for the end-to-end result files [default: node-local scratch]. On an "
            "A100 node, four ranks wrote 20 GB to Lustre in 5.8 s and to the NVMe in 18.7 s"
        ),
    ] = None,
) -> None:
    ranks_per_node = int(os.environ["SLURM_NTASKS_PER_NODE"])
    nodes = int(os.environ["SLURM_JOB_NUM_NODES"])
    ranks = int(os.environ["SLURM_NTASKS"])
    cores = os.environ["SLURM_CPUS_ON_NODE"]
    # One rank per usable GPU of this job; on nodes shared with other jobs that is fewer than the
    # node has, and a GPU in MIG mode without instances is allocated but unusable.
    gpus = usable_gpus()
    if ranks_per_node != len(gpus):
        raise SystemExit(
            f"Error: run one rank per usable GPU (--ntasks-per-node={len(gpus)})"
        )
    gpu = gpu_tag(gpus[0].name)
    S = scratch()
    e2e_output = e2e_output or S / "e2e"
    genomes = MGPU / f"genomes-{size}"
    database = MGPU / f"refseq211-{size}.cuddl"

    # Regenerate first: meson compile looks up targets before it notices a changed meson.build.
    run("ninja", "-C", "build", "build.ninja")
    run(
        "meson",
        "compile",
        "-C",
        "build",
        "-j",
        cores,
        "cuddl-multi-gpu-all-to-all-benchmark",
        "cuddl-distributed-all-to-all-benchmark",
        "cuddl-build-reference-db",
    )
    if not database.exists():
        S.mkdir(parents=True, exist_ok=True)
        run(
            "./build/examples/cuddl-build-reference-db",
            genomes,
            "-o",
            S / "db.cuddl",
            "--workers",
            cores,
        )
        shutil.copy(S / "db.cuddl", database)

    # UCX's registration cache took 31 s for the first 410 MB MPI_Bcast across two nodes and 0.8 s
    # without it; later broadcasts were also faster without it (165 against 220 ms).
    os.environ["UCX_RCACHE_ENABLE"] = "n"
    os.environ["UCX_NET_DEVICES"] = infiniband_ports()

    def mpirun(count: int, *command) -> None:
        run(
            "mpirun",
            "-n",
            str(count),
            "--bind-to",
            "none",
            "-x",
            "LD_LIBRARY_PATH",
            "-x",
            "PATH",
            "-x",
            "UCX_RCACHE_ENABLE",
            "-x",
            "UCX_NET_DEVICES",
            *command,
            keep=KEEP,
        )

    out = f"results/mogon-{gpu}-{tag}-{size}-{nodes}x{ranks_per_node}"
    for line in per_node(
        nodes, "sh", "-c", 'echo "$(hostname -s) GPUs $CUDA_VISIBLE_DEVICES"'
    ):
        print(line, flush=True)
    # The sweep writes every rank's results to node-local NVMe, and the first rank of every node
    # reads the database from there.
    per_node(nodes, "mkdir", "-p", S / "sweep", e2e_output)
    per_node(nodes, "cp", database, S / "db.cuddl")
    # The search benchmark keeps every GPU visible: it runs 2 or 4 ranks per node and picks GPUs
    # on separate PCIe switches, which downloads need (808 against 1526 ms for 2 ranks).
    if strong:
        mpirun(
            ranks,
            BENCH,
            S / "db.cuddl",
            "--samples",
            "10",
            "--warmups",
            "2",
            "--thresholds",
            *["0", "5", "10", "20", "50", "100", "200", "500"],
            "--write-dir",
            S / "sweep",
            "-o",
            f"{out}-strong.json",
        )
    if weak:
        mpirun(
            ranks,
            BENCH,
            S / "db.cuddl",
            "--weak",
            "--samples",
            "10",
            "--warmups",
            "2",
            "-o",
            f"{out}-weak.json",
        )
    if not e2e:
        return

    # End-to-end ranks see only their own GPU: with all 8 visible the 100k sketch took 13.4 s
    # instead of 11.8-12.3 s. With 16 cores per A40, hybrid decompression sketched 100k genomes
    # fastest (12 s; cpu 15 s, gpu 16-25 s); automatic picks gpu on discrete GPUs.
    def pipeline(count: int, minimum: int, name: str) -> None:
        mpirun(
            count,
            *OWN_GPU,
            E2E,
            "--decompression",
            "hybrid",
            genomes,
            "--minimum-matches",
            str(minimum),
            "--output-dir",
            e2e_output,
            "-o",
            f"results/mogon-{gpu}-{tag}-e2e-{size}-{name}.json",
        )

    # Earlier jobs on these nodes may have left genomes in the page cache: drop them so that the
    # cold run reads every byte from Lustre.
    per_node(
        nodes,
        "uv",
        "run",
        "--offline",
        "scripts/warm_page_cache.py",
        genomes,
        "--evict",
    )
    pipeline(ranks, 20, f"{nodes}x{ranks_per_node}-m20-cold")
    # The cold run read only each node's share; every node reads all genomes once so that runs on
    # fewer nodes find their larger shares cached too. Evicting first times that read from Lustre
    # with every node reading at once and as many reads in flight as the ranks' readers keep.
    evict = [genomes, "--evict", "--threads", "64"]
    per_node(nodes, "uv", "run", "--offline", "scripts/warm_page_cache.py", *evict)
    reads = per_node(
        nodes,
        *["uv", "run", "--offline", "scripts/warm_page_cache.py", genomes],
        *["--threads", str(64 * ranks_per_node)],
    )
    with open(
        f"results/mogon-{gpu}-{tag}-read-{size}-{nodes}x{ranks_per_node}.json", "w"
    ) as f:
        f.write(
            json.dumps([json.loads(line) for line in reads if line.startswith("{")])
            + "\n"
        )
    for n in range(1, nodes + 1):
        for minimum in [20, 0]:
            pipeline(
                n * ranks_per_node, minimum, f"{n}x{ranks_per_node}-m{minimum}-warm"
            )

    # SimdSketch with the cores the cuDDL ranks had and the page cache they warmed.
    if simdsketch:
        threads = ranks_per_node * int(os.environ["SLURM_CPUS_PER_TASK"])
        for n in range(1, nodes + 1):
            report = simdsketch_nodes(genomes, n, threads)
            path = (
                f"results/mogon-{gpu}-{tag}-simdsketch-{size}-{n}x{ranks_per_node}.json"
            )
            with open(path, "w") as f:
                f.write(json.dumps(report) + "\n")


if __name__ == "__main__":
    typer.run(main)
