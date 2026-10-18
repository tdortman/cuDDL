#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = ["typer"]
# ///
"""Read every file of GENOMES once, so later runs on this node find them in the page cache.

Prints the read as one JSON line. With --evict, drops the files from this node's page cache
instead, so the next run reads them from storage.
"""

import json
import socket
import time
from pathlib import Path
from typing import Annotated

import typer
from slurm_utils import evict_files, genome_files, read_files


def main(
    genomes: Annotated[Path, typer.Argument(exists=True, file_okay=False)],
    threads: Annotated[int, typer.Option(help="Files read at once")] = 16,
    evict: Annotated[
        bool, typer.Option(help="Drop the files from the page cache")
    ] = False,
) -> None:
    paths = genome_files(genomes)
    if evict:
        evict_files(paths, threads)
        return
    start = time.perf_counter()
    read_files(paths, threads)
    seconds = time.perf_counter() - start
    size = sum(p.stat().st_size for p in paths)
    print(
        json.dumps(
            {
                "host": socket.gethostname(),
                "threads": threads,
                "bytes": size,
                "seconds": round(seconds, 3),
                "gbps": round(size / seconds / 1e9, 3),
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    typer.run(main)
