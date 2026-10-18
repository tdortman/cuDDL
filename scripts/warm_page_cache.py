#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = ["typer"]
# ///
"""Read every file of GENOMES once, so later runs on this node find them in the page cache.

With --evict, drop them from this node's page cache instead (posix_fadvise DONTNEED, which also
works on Lustre), so the next run reads them from storage.
"""

import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Annotated

import typer
from slurm_utils import genome_files, read_files


def main(
    genomes: Annotated[Path, typer.Argument(exists=True, file_okay=False)],
    threads: Annotated[int, typer.Option(help="Files read at once")] = 16,
    evict: Annotated[
        bool, typer.Option(help="Drop the files from the page cache")
    ] = False,
) -> None:
    if not evict:
        read_files(genome_files(genomes), threads)
        return

    def drop(path: Path) -> None:
        fd = os.open(path, os.O_RDONLY)
        try:
            os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
        finally:
            os.close(fd)

    with ThreadPoolExecutor(threads) as pool:
        list(pool.map(drop, genome_files(genomes)))


if __name__ == "__main__":
    typer.run(main)
