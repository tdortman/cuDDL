"""Shared helpers for the scripts that benchmark inside a MOGON Slurm job."""

import mmap
import os
import re
import shlex
import subprocess
import sys
import threading
import time
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import NamedTuple

MGPU = Path("data/refseq211/mgpu")
# Runs the rest of an mpirun command line with only the rank's node-local GPU visible.
OWN_GPU = ["uv", "run", "--offline", "scripts/own_gpu.py"]
BLOCK = 4 << 20  # read size; mmap buffers are page-aligned, as O_DIRECT requires


class Gpu(NamedTuple):
    uuid: str
    name: str


def scratch() -> Path:
    """The job's node-local NVMe directory, removed when the job ends."""
    return Path("/localscratch") / os.environ["SLURM_JOB_ID"]


def trace(args: list[str]) -> None:
    """Echo a command to stderr, like Bash's set -x, keeping stdout for results."""
    print(f"+ [{time.strftime('%T')}]", shlex.join(args), file=sys.stderr, flush=True)


def run(
    *command: str | Path,
    env: dict[str, str] | None = None,
    keep: str | None = None,
    stdout: Path | None = None,
) -> None:
    """Run COMMAND and fail on a nonzero exit.

    KEEP is a regular expression: only output lines matching it are printed, cut to 400
    characters. STDOUT sends the standard output to a file instead.
    """
    args = [str(c) for c in command]
    trace(args)
    if stdout is not None:
        with stdout.open("w") as f:
            subprocess.run(args, check=True, env=env, stdout=f)
        return
    if keep is None:
        subprocess.run(args, check=True, env=env)
        return
    with subprocess.Popen(
        args,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        errors="replace",
    ) as process:
        assert process.stdout is not None  # stdout=PIPE
        for line in process.stdout:
            if re.search(keep, line):
                print(line.rstrip()[:400], flush=True)
    if process.returncode != 0:
        raise subprocess.CalledProcessError(process.returncode, args)


def per_node(
    nodes: int,
    *command: str | Path,
    srun_args: Sequence[str] = (),
    env: dict[str, str] | None = None,
) -> list[str]:
    """Run COMMAND once on each of the job's first NODES nodes; returns their output lines."""
    args = [
        "srun",
        "-N",
        str(nodes),
        f"--ntasks={nodes}",
        *srun_args,
        *map(str, command),
    ]
    trace(args)
    result = subprocess.run(args, check=True, capture_output=True, text=True, env=env)
    return result.stdout.splitlines()


def usable_gpus() -> list[Gpu]:
    """This job's GPUs that CUDA can use, in nvidia-smi order, skipping a GPU in MIG mode
    without instances. UUIDs, because nvidia-smi's indices follow PCI order and CUDA's need not."""
    rows = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=uuid,name,mig.mode.current",
            "--format=csv,noheader",
        ],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.splitlines()
    fields = [row.split(", ") for row in rows]
    return [Gpu(uuid, name) for uuid, name, mig in fields if mig != "Enabled"]


def gpu_tag(name: str) -> str:
    """Model for result names: a40, a100, h100, b200, gh200, ..."""
    model = re.search(r"[A-Z]+\d+", name.removeprefix("NVIDIA "))
    return model.group().lower() if model else re.sub(r"\W+", "-", name.lower())


def genome_files(genomes: Path) -> list[Path]:
    return sorted(genomes.iterdir())


def read_files(paths: list[Path], threads: int, direct: bool = False) -> None:
    """Read every file once; with DIRECT, through O_DIRECT, bypassing the page cache."""
    flags = os.O_RDONLY | (os.O_DIRECT if direct else 0)
    local = threading.local()

    def read(path: Path) -> None:
        if not hasattr(local, "buffer"):
            local.buffer = mmap.mmap(-1, BLOCK)
        fd = os.open(path, flags)
        try:
            while os.readv(fd, [local.buffer]):
                pass
        finally:
            os.close(fd)

    with ThreadPoolExecutor(threads) as pool:
        list(pool.map(read, paths))


def evict_files(paths: list[Path], threads: int) -> None:
    """Drop every file from this node's page cache (posix_fadvise DONTNEED, which also works
    on Lustre without root), so the next read comes from storage."""

    def drop(path: Path) -> None:
        fd = os.open(path, os.O_RDONLY)
        try:
            os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
        finally:
            os.close(fd)

    with ThreadPoolExecutor(threads) as pool:
        list(pool.map(drop, paths))
