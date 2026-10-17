#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = ["typer"]
# ///
"""Run COMMAND as one MPI rank that sees only its own GPU.

The rank gets its OpenMPI node-local rank's entry among the job's GPUs that CUDA can use; a GPU
in MIG mode without instances (one per A100 node on MOGON) is skipped. With every GPU visible,
page-locked allocations are mapped into all of them, which slowed sketching on 8-GPU nodes.
Usage under mpirun: uv run --offline scripts/own_gpu.py BINARY ARGS...
"""

import os
from typing import Annotated

import typer
from slurm_utils import usable_gpus

app = typer.Typer(
    add_completion=False,
    context_settings={"allow_interspersed_args": False, "ignore_unknown_options": True},
)


@app.command()
def main(
    command: Annotated[list[str], typer.Argument(help="Command and its arguments")],
) -> None:
    rank = int(os.environ["OMPI_COMM_WORLD_LOCAL_RANK"])
    os.environ["CUDA_VISIBLE_DEVICES"] = usable_gpus()[rank].uuid
    os.execvp(command[0], command)


if __name__ == "__main__":
    app()
