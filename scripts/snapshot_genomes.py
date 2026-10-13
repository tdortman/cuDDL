#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = ["typer"]
# ///
"""Fix a genome set while a download is still running.

Creates OUTPUT holding absolute symlinks to COUNT completed genome files of
GENOMES, in name order: the first COUNT, or with --spread COUNT files evenly
spaced over all of them. Spread subsets of an accession-ordered collection keep
its mix of species and near-duplicates; a prefix can be far more redundant.
aria2 keeps a NAME.aria2 control file beside each unfinished download, so those
files are skipped. Snapshots are meant to stay unchanged, so an existing OUTPUT
is an error.
"""

import os
from pathlib import Path
from typing import Annotated

import typer

SUFFIXES = (".fna.gz", ".fa.gz", ".fasta.gz", ".fna", ".fa", ".fasta")


def main(
    genomes: Annotated[
        Path, typer.Argument(exists=True, file_okay=False, help="Download directory")
    ],
    count: Annotated[int, typer.Argument(min=1, help="Genomes to link")],
    output: Annotated[Path, typer.Argument(help="Snapshot directory to create")],
    spread: Annotated[
        bool, typer.Option(help="Take evenly spaced genomes instead of the first ones")
    ] = False,
) -> None:
    if output.exists():
        raise typer.BadParameter(f"{output} already exists; snapshots are not modified")
    source = genomes.resolve()
    names = {entry.name for entry in os.scandir(source)}
    complete = sorted(
        n for n in names if n.endswith(SUFFIXES) and f"{n}.aria2" not in names
    )
    if len(complete) < count:
        raise typer.BadParameter(
            f"only {len(complete)} completed genomes in {genomes}, {count} requested"
        )
    staging = output.with_name(output.name + ".partial")
    staging.mkdir(parents=True)
    chosen = (
        [complete[i * len(complete) // count] for i in range(count)]
        if spread
        else complete[:count]
    )
    for name in chosen:
        (staging / name).symlink_to(source / name)
    # Rename last, so an interrupted run never leaves a short snapshot under OUTPUT.
    staging.rename(output)
    typer.echo(f"linked {count} of {len(complete)} completed genomes into {output}")


if __name__ == "__main__":
    typer.run(main)
