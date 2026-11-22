#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = ["typer"]
# ///
"""Standalone cold read rate of a genome list: the reference for the model's cold-ingest term.

Evicts every listed file from this host's page cache with posix_fadvise(DONTNEED), then reads all
of them whole with a pool of threads (as the pipeline's readers do) and reports GB/s. One JSON
line per thread count, appended to --output.
"""

import json
import os
import socket
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Annotated

import typer

CHUNK = 8 << 20


def cached_gb() -> float:
    for line in Path("/proc/meminfo").read_text().splitlines():
        if line.startswith("Cached:"):
            return int(line.split()[1]) / 1e6
    raise RuntimeError("no Cached: line in /proc/meminfo")


def evict(path: str) -> None:
    fd = os.open(path, os.O_RDONLY)
    try:
        os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
    finally:
        os.close(fd)


def read_whole(path: str) -> int:
    total = 0
    with open(path, "rb", buffering=0) as f:
        while chunk := f.read(CHUNK):
            total += len(chunk)
    return total


def main(
    genome_list: Annotated[Path, typer.Argument(exists=True, dir_okay=False)],
    threads: Annotated[list[int], typer.Option(help="Reader thread counts, each after a fresh eviction")] = [64, 128],
    output: Annotated[Path, typer.Option()] = Path("cold-read.jsonl"),
) -> None:
    paths = [p.strip() for p in genome_list.read_text().splitlines() if p.strip()]
    for count in threads:
        before = cached_gb()
        with ThreadPoolExecutor(64) as pool:
            list(pool.map(evict, paths, chunksize=256))
        after = cached_gb()
        start = time.perf_counter()
        with ThreadPoolExecutor(count) as pool:
            total = sum(pool.map(read_whole, paths, chunksize=64))
        seconds = time.perf_counter() - start
        row = {
            "host": socket.gethostname(),
            "list": str(genome_list),
            "files": len(paths),
            "threads": count,
            "bytes": total,
            "seconds": round(seconds, 3),
            "gb_s": round(total / seconds / 1e9, 4),
            "cached_gb_before_evict": round(before, 1),
            "cached_gb_after_evict": round(after, 1),
            "method": "posix_fadvise DONTNEED on every file, then whole-file reads in 8 MiB chunks by a thread pool",
        }
        typer.echo(json.dumps(row))
        with output.open("a") as f:
            f.write(json.dumps(row) + "\n")


if __name__ == "__main__":
    typer.run(main)
