#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = ["typer"]
# ///
"""SimdSketch all-to-all on the first NODES nodes of this Slurm job, split the way cuDDL's
distributed pipeline splits it.

Every node sketches a contiguous slice of the genomes with THREADS threads, the sketch parts
are exchanged through Lustre, and every node compares the genomes r with r % NODES == node
against all later genomes and writes their similarities to node-local scratch. SimdSketch has no
thresholded output, so every pair is written. Each step reports its slowest node; the compare
step reports that node's comparing (SimdSketch's own timer around the similarity loop) and the
rest of its run, mostly writing the similarities. Prints one JSON line.

The sketch, gather and compare commands are the per-node steps `run` starts through srun.
"""

import json
import os
import re
import shutil
import subprocess
import time
from pathlib import Path
from typing import Annotated, Any

import typer
from slurm_utils import genome_files, per_node, read_files, run, scratch

SIMDSKETCH = Path("build/benchmarks/simdsketch-benchmark")
SELF = ["uv", "run", "--offline", "scripts/simdsketch_nodes.py"]

app = typer.Typer(add_completion=False)


def timed(*command: str | Path | int, env: dict[str, str] | None = None) -> float:
    start = time.perf_counter()
    subprocess.run(
        [str(c) for c in command], check=True, env=env, stdout=subprocess.DEVNULL
    )
    return time.perf_counter() - start


def node() -> tuple[int, Path]:
    """This step's node index and its node-local working directory."""
    local = scratch() / "simdsketch"
    local.mkdir(parents=True, exist_ok=True)
    return int(os.environ["SLURM_PROCID"]), local


def build_simdsketch() -> None:
    """SimdSketch picks its SIMD paths for the compiling CPU, so it is built on the node. The
    crates are vendored, so cargo needs no network. The nodes have no clang, so link with gcc."""
    env = {**os.environ, "CARGO_TARGET_X86_64_UNKNOWN_LINUX_GNU_LINKER": "gcc"}
    run(
        "ninja",
        "-C",
        "build",
        "benchmarks/simdsketch-benchmark",
        env=env,
        stdout=Path("/dev/null"),
    )


def simdsketch_nodes(genomes: Path, nodes: int, threads: int) -> dict:
    build_simdsketch()
    shared = Path(f"build/scratch/simdsketch-{os.environ['SLURM_JOB_ID']}")
    shutil.rmtree(shared, ignore_errors=True)
    shared.mkdir(parents=True)
    paths = genome_files(genomes)
    (shared / "all.txt").write_text("".join(f"{p}\n" for p in paths))
    # Every step runs on the same nodes: srun -N alone may pick other nodes of the job per step.
    hosts = subprocess.run(
        ["scontrol", "show", "hostnames"], check=True, capture_output=True, text=True
    ).stdout.split()[:nodes]
    # srun checks -c against the job's SLURM_CPUS_PER_TASK; the step's tasks get the node's cores.
    env = {**os.environ, "SLURM_CPUS_PER_TASK": str(threads)}
    srun_args = ["-w", ",".join(hosts), "--ntasks-per-node=1", "-c", str(threads)]

    def step(name: str) -> dict[str, float]:
        """Run step NAME on every node; return the times of the node with the largest sum."""
        lines = per_node(
            nodes,
            *SELF,
            name,
            shared,
            str(nodes),
            str(threads),
            srun_args=srun_args,
            env=env,
        )
        times = [json.loads(line) for line in lines]
        assert len(times) == nodes, lines
        return max(times, key=lambda t: sum(t.values()))

    report: dict[str, Any] = {
        "tool": "simdsketch",
        "genomes": len(paths),
        "nodes": nodes,
        "threads": threads,
        "input_bytes": sum(p.stat().st_size for p in paths),
    }
    for name in ["sketch", "gather", "compare"]:
        report |= {k: round(v, 3) for k, v in step(name).items()}
    shutil.rmtree(shared)
    report["compare_bytes"] = len(paths) * (len(paths) - 1) // 2 * 4
    report["wall_s"] = round(
        sum(report[k] for k in ["sketch_s", "gather_s", "compare_s", "write_s"]), 3
    )
    return report


SharedArg = Annotated[Path, typer.Argument(help="Shared directory of this run")]
NodesArg = Annotated[int, typer.Argument(min=1, help="Nodes taking part")]
ThreadsArg = Annotated[int, typer.Argument(min=1, help="SimdSketch threads per node")]


@app.command("run")
def run_all(
    genomes: Annotated[Path, typer.Argument(exists=True, file_okay=False)],
    nodes: NodesArg,
    threads: ThreadsArg,
) -> None:
    """Run every step on the first NODES nodes and print the JSON report."""
    print(json.dumps(simdsketch_nodes(genomes, nodes, threads)))


@app.command()
def sketch(shared: SharedArg, nodes: NodesArg, threads: ThreadsArg) -> None:
    index, local = node()
    paths = (shared / "all.txt").read_text().splitlines()
    begin, end = index * len(paths) // nodes, (index + 1) * len(paths) // nodes
    part = local / "part.txt"
    part.write_text("".join(f"{p}\n" for p in paths[begin:end]))
    # Untimed: the node reads its slice once, so the sketch reads from the page cache like the
    # cuDDL runs it is compared with.
    read_files([Path(p) for p in paths[begin:end]], 16)
    seconds = timed(
        SIMDSKETCH,
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
        shared / f"part-{index}.sk",
        part,
    )
    print(json.dumps({"sketch_s": seconds}))


@app.command()
def gather(shared: SharedArg, nodes: NodesArg, threads: ThreadsArg) -> None:
    _, local = node()
    start = time.perf_counter()
    for part in shared.glob("part-*.sk"):
        shutil.copy(part, local)
    print(json.dumps({"gather_s": time.perf_counter() - start}))


@app.command()
def compare(shared: SharedArg, nodes: NodesArg, threads: ThreadsArg) -> None:
    index, local = node()
    timings = local / "resident.json"
    total = timed(
        SIMDSKETCH,
        "-j",
        threads,
        "compare",
        "--references",
        *(local / f"part-{j}.sk" for j in range(nodes)),
        "--shard",
        index,
        "--shards",
        nodes,
        "--output",
        local / "pairs.f32",
        env={**os.environ, "CUDDL_RESIDENT_TIMINGS": str(timings)},
    )
    resident = re.search(r'"resident_ms":\s*([0-9.eE+-]+)', timings.read_text())
    if resident is None:
        raise RuntimeError(f"{timings} holds no resident_ms")
    comparing = float(resident[1]) / 1e3
    shutil.rmtree(local)
    print(json.dumps({"compare_s": comparing, "write_s": total - comparing}))


if __name__ == "__main__":
    app()
