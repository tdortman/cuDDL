#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = ["typer"]
# ///
"""Measure one node's hardware ceilings for each phase of the end-to-end pipeline.

Appends one JSON line per measurement to OUTPUT:

- reading the genome snapshot with the pipeline's pattern of many ~1 MB gzip files, first with
  O_DIRECT (bypassing the client page cache, so from the filesystem), then from the page cache;
- sequential write (buffered, as the pipeline writes its results, then fsync) and O_DIRECT read
  on the node-local scratch and on the shared filesystem;
- CPU inflation with libdeflate, the library and patches the loaders use, built from
  subprojects/libdeflate;
- the search's integer-ALU ceiling under the power limit (alu-power-ceiling);
- one rank sketching the snapshot with each --decompression backend on one GPU.

Host-to-device bandwidth is not measured: one A40 sketches about 1.3 GB/s of gzip, an order of
magnitude below a PCIe 4 x16 link.

Run on a whole node, e.g. on MOGON:

    sbatch -M mogonki -A ki-gpu4science -p a40 -N 1 --ntasks-per-node=1 -c 128 --gres=gpu:8 \\
      --mem=512G -o logs/%x-%j.log -e logs/%x-%j.log \\
      --wrap 'source ./.envrc && uv run --offline scripts/hw_ceilings.py'
"""

import ctypes
import json
import mmap
import multiprocessing
import os
import shutil
import socket
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import BrokenBarrierError
from typing import Annotated

import typer
from slurm_utils import (
    BLOCK,
    MGPU,
    OWN_GPU,
    genome_files,
    gpu_tag,
    read_files,
    run,
    scratch,
    usable_gpus,
)

LIBDEFLATE = Path("subprojects/libdeflate")
LIBDEFLATE_INSUFFICIENT_SPACE = 3


def stream_files(directory: Path, streams: int, gib: int) -> tuple[float, float]:
    """Seconds to write STREAMS files of GIB GiB each through the page cache and fsync them, then
    to read them back with O_DIRECT. O_DIRECT writes are synchronous per request: on Lustre,
    8 streams of them reached 28 MB/s."""
    directory.mkdir(parents=True, exist_ok=True)
    paths = [directory / f"stream-{i}" for i in range(streams)]
    blocks = (gib << 30) // BLOCK

    def write(path: Path) -> None:
        buffer = mmap.mmap(-1, BLOCK)
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC)
        try:
            for _ in range(blocks):
                os.write(fd, buffer)
            os.fsync(fd)
        finally:
            os.close(fd)

    with ThreadPoolExecutor(streams) as pool:
        start = time.perf_counter()
        list(pool.map(write, paths))
        written = time.perf_counter() - start
    start = time.perf_counter()
    read_files(paths, streams, direct=True)
    read = time.perf_counter() - start
    shutil.rmtree(directory)
    return written, read


def build_libdeflate(build: Path) -> Path:
    library = build / "libdeflate.so"
    sources = [
        *(LIBDEFLATE / "lib").glob("*.c"),
        *(LIBDEFLATE / "lib/x86").glob("*.c"),
        *(LIBDEFLATE / "lib/arm").glob("*.c"),
    ]
    run("cc", "-O2", "-shared", "-fPIC", f"-I{LIBDEFLATE}", *sources, "-o", library)
    return library.resolve()


def gzip_size(path: Path) -> int:
    """Inflated size from the gzip trailer: exact for single-member files under 4 GiB."""
    with path.open("rb") as f:
        f.seek(-4, os.SEEK_END)
        return int.from_bytes(f.read(4), "little")


def inflate_worker(
    library: Path, paths: list[Path], sizes: tuple[int, int], claims, barrier, spans
) -> None:
    """Inflate files claimed from CLAIMS until none are left; put (start, end) on SPANS. SIZES
    are the largest input and output, the starting sizes of the buffers."""
    try:
        lib = ctypes.CDLL(str(library))
        lib.libdeflate_alloc_decompressor.restype = ctypes.c_void_p
        lib.libdeflate_gzip_decompress_ex.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_size_t,
            ctypes.c_void_p,
            ctypes.c_size_t,
            ctypes.POINTER(ctypes.c_size_t),
            ctypes.POINTER(ctypes.c_size_t),
        ]
        decompressor = lib.libdeflate_alloc_decompressor()
        data, output = bytearray(sizes[0]), bytearray(sizes[1])
        source = ctypes.addressof((ctypes.c_char * len(data)).from_buffer(data))
        consumed, produced = ctypes.c_size_t(), ctypes.c_size_t()
        barrier.wait()  # every process has its buffers before the clock starts
        start = time.monotonic()
        while True:
            with claims.get_lock():
                index = claims.value
                claims.value += 1
            if index >= len(paths):
                break
            with paths[index].open("rb") as f:
                size = f.readinto(data)
            offset = 0
            while offset < size:
                target = (ctypes.c_char * len(output)).from_buffer(output)
                status = lib.libdeflate_gzip_decompress_ex(
                    decompressor,
                    source + offset,
                    size - offset,
                    target,
                    len(output),
                    ctypes.byref(consumed),
                    ctypes.byref(produced),
                )
                del target  # release the export so the bytearray can grow
                if (
                    status == LIBDEFLATE_INSUFFICIENT_SPACE
                ):  # an earlier member was larger
                    output.extend(bytes(len(output)))
                    continue
                if status != 0:
                    raise RuntimeError(
                        f"libdeflate failed on {paths[index]} with status {status}"
                    )
                offset += consumed.value
        spans.put((start, time.monotonic()))
    except BaseException as error:
        barrier.abort()
        spans.put(error)


def inflate_files(library: Path, paths: list[Path], processes: int) -> float:
    """Seconds PROCESSES processes take to read every file from the page cache and inflate each
    of its gzip members. Processes, not threads: 16 threads calling libdeflate through ctypes
    reached 4.5 GB/s of gzip on an A100 node and 128 threads only 3.8."""
    context = multiprocessing.get_context("fork")
    claims, barrier, spans = (
        context.Value("q", 0),
        context.Barrier(processes),
        context.Queue(),
    )
    sizes = max(p.stat().st_size for p in paths), max(map(gzip_size, paths))
    workers = [
        context.Process(
            target=inflate_worker, args=(library, paths, sizes, claims, barrier, spans)
        )
        for _ in range(processes)
    ]
    for worker in workers:
        worker.start()
    results = [spans.get() for _ in workers]
    for worker in workers:
        worker.join()
    for result in results:
        if isinstance(result, BaseException) and not isinstance(
            result, BrokenBarrierError
        ):
            raise result
    return max(end for _, end in results) - min(start for start, _ in results)


def main(
    genomes: Annotated[
        Path,
        typer.Argument(exists=True, file_okay=False, help="Genome snapshot directory"),
    ] = MGPU / "genomes-30k",
    local: Annotated[
        Path | None,
        typer.Option(
            help="Node-local directory [default: /localscratch/$SLURM_JOB_ID]"
        ),
    ] = None,
    shared: Annotated[
        Path,
        typer.Option(help="Directory on the shared filesystem for the stream test"),
    ] = Path("logs/ceilings"),
    output: Annotated[
        Path | None,
        typer.Option(
            help="JSON lines file [default: results/mogon-<gpu>-ceilings.jsonl]"
        ),
    ] = None,
    build: Annotated[Path, typer.Option(help="Meson build directory")] = Path("build"),
    threads: Annotated[int, typer.Option(help="Threads for reads and inflation")] = len(
        os.sched_getaffinity(0)
    ),
    workers: Annotated[
        int,
        typer.Option(help="Parser threads of the sketching rank: the cores per GPU"),
    ] = 16,
    streams: Annotated[int, typer.Option(help="Parallel files in the stream test")] = 8,
    stream_gib: Annotated[int, typer.Option(help="GiB per stream file")] = 4,
    alu_seconds: Annotated[int, typer.Option(help="alu-power-ceiling run time")] = 60,
) -> None:
    name = usable_gpus()[0].name  # the GPU own_gpu.py gives the only rank
    tag = gpu_tag(name)
    output = output or Path(f"results/mogon-{tag}-ceilings.jsonl")
    local = local or scratch()
    paths = genome_files(genomes)
    gzip_bytes = sum(p.stat().st_size for p in paths)

    def record(phase: str, size: int, seconds: float) -> None:
        row = {
            "host": socket.gethostname(),
            "gpu": name,
            "threads": threads,
            "phase": phase,
            "bytes": size,
            "seconds": round(seconds, 3),
            "gbps": round(size / seconds / 1e9, 3),
        }
        print(json.dumps(row), flush=True)
        with output.open("a") as f:
            f.write(json.dumps(row) + "\n")

    def timed(phase: str, size: int, work, *args) -> None:
        start = time.perf_counter()
        work(*args)
        record(phase, size, time.perf_counter() - start)

    run("ninja", "-C", build, "build.ninja")
    run(
        "meson",
        "compile",
        "-C",
        build,
        "-j",
        str(threads),
        "alu-power-ceiling",
        "cuddl-distributed-all-to-all-benchmark",
    )
    libdeflate = build_libdeflate(build)

    # Direct reads come first: the page-cache pass afterwards would otherwise warm nothing new.
    timed("filesystem_read_direct", gzip_bytes, read_files, paths, threads, True)
    read_files(paths, threads, direct=False)
    timed("page_cache_read", gzip_bytes, read_files, paths, threads, False)

    stream_bytes = streams * (stream_gib << 30)
    for label, directory in [
        ("local", local / "streams"),
        ("shared", shared / "streams"),
    ]:
        written, read = stream_files(directory, streams, stream_gib)
        record(f"{label}_write", stream_bytes, written)
        record(f"{label}_read", stream_bytes, read)

    # Rates in gzip bytes, as the sketch phase counts its input.
    for count in sorted({workers, threads}):
        seconds = inflate_files(libdeflate, paths, count)
        record(f"libdeflate_inflate_{count}_threads", gzip_bytes, seconds)

    own_gpu = {**os.environ, "OMPI_COMM_WORLD_LOCAL_RANK": "0"}  # first usable GPU only
    run(
        *OWN_GPU,
        build / "benchmarks/alu-power-ceiling",
        "--seconds",
        str(alu_seconds),
        "-o",
        f"results/mogon-{tag}-alu-power-ceiling.json",
        env=own_gpu,
    )

    # One rank per backend; --minimum-matches 20 keeps search and output small next to sketching.
    for backend in ["cpu", "gpu", "hybrid"]:
        report = Path(f"results/mogon-{tag}-ceilings-sketch-{backend}.json")
        run(
            "mpirun",
            "-n",
            "1",
            "--bind-to",
            "none",
            "-x",
            "LD_LIBRARY_PATH",
            "-x",
            "PATH",
            *OWN_GPU,
            build / "benchmarks/cuddl-distributed-all-to-all-benchmark",
            genomes,
            "--minimum-matches",
            "20",
            "--workers",
            str(workers),
            "--decompression",
            backend,
            "--output-dir",
            local,
            "-o",
            report,
        )
        sketch_ms = json.loads(report.read_text())["phases_wall_ms"]["sketch"]
        record(f"sketch_{backend}_{workers}_workers", gzip_bytes, sketch_ms / 1e3)


if __name__ == "__main__":
    typer.run(main)
