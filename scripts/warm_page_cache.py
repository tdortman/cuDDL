#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = ["typer"]
# ///
"""Read every file of GENOMES once, so later runs on this node find them in the page cache."""

from pathlib import Path
from typing import Annotated

import typer
from slurm_utils import genome_files, read_files


def main(
    genomes: Annotated[Path, typer.Argument(exists=True, file_okay=False)],
    threads: Annotated[int, typer.Option(help="Files read at once")] = 16,
) -> None:
    read_files(genome_files(genomes), threads)


if __name__ == "__main__":
    typer.run(main)
