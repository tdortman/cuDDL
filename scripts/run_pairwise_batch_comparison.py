#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = ["jsonschema", "typer"]
# ///
"""Run cuDDL and BBTools batch comparisons into one normalized JSON result."""

import csv
import io
import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Annotated, cast

import typer
from benchmark_schema import make_result, measurements_from_rows, write_result

ROOT = Path(__file__).resolve().parent.parent
NVBENCH_MODES = {
    "batch_compare_kernel": "device_resident",
    "batch_compare_h2d": "h2d",
    "batch_compare_d2h": "d2h",
    "batch_compare_end_to_end": "transfer_each_batch",
}
BBTOOLS_MODES = {
    "bbtools_sequential": "sequential",
    "bbtools_parallel": "parallel",
}
SYSTEM_FIELDS = {
    "System OS": "os",
    "System Kernel": "kernel",
    "System Architecture": "architecture",
    "System CPU": "cpu",
    "System Logical CPU Count": "logical_cpu_count",
    "System RAM Bytes": "ram_bytes",
    "Device Name": "gpu",
    "Compute Capability": "compute_capability",
    "SM Count": "sm_count",
    "GPU RAM Bytes": "gpu_ram_bytes",
    "CUDA Runtime Version": "cuda_runtime_version",
    "CUDA Driver Version": "cuda_driver_version",
    "CUDA Compile Version": "cuda_compile_version",
}
SYSTEM_INTEGER_FIELDS = {
    "logical_cpu_count",
    "ram_bytes",
    "sm_count",
    "gpu_ram_bytes",
    "cuda_runtime_version",
    "cuda_driver_version",
    "cuda_compile_version",
}


def available_cpus() -> int:
    try:
        return len(os.sched_getaffinity(0))
    except AttributeError:
        return os.cpu_count() or 1


def project_path(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def run(
    command: list[str], *, capture: bool = False
) -> subprocess.CompletedProcess[str]:
    typer.echo("+ " + " ".join(command), err=True)
    return subprocess.run(
        command,
        cwd=ROOT,
        check=True,
        text=True,
        stdout=subprocess.PIPE if capture else None,
    )


def read_nvbench(path: Path) -> tuple[list[dict[str, object]], dict[str, int | str]]:
    rows: list[dict[str, object]] = []
    systems: set[tuple[str, ...]] = set()
    with path.open(newline="") as stream:
        for source in csv.DictReader(stream):
            mode = NVBENCH_MODES.get(source["Benchmark"])
            if mode is None or source["Skipped"] != "No":
                continue
            systems.add(tuple(source[field] for field in SYSTEM_FIELDS))
            batch = int(source["Batch"])
            seconds = float(source["GPU Time (sec)"])
            rows.append(
                {
                    "implementation": "cuddl",
                    "mode": mode,
                    "batch": batch,
                    "threads": 0,
                    "trial": None,
                    "samples": int(source["Samples"]),
                    "iterations": None,
                    "seconds_per_batch": seconds,
                    "comparisons_per_second": batch / seconds,
                    "device": source["Device Name"],
                }
            )
    if len(systems) != 1:
        raise RuntimeError("NVBench rows do not describe exactly one system")
    values = next(iter(systems))
    system = {
        output: int(value) if output in SYSTEM_INTEGER_FIELDS else value
        for output, value in zip(SYSTEM_FIELDS.values(), values, strict=True)
    }
    return rows, system


def read_bbtools(text: str) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for source in csv.DictReader(io.StringIO(text)):
        mode = BBTOOLS_MODES.get(source["implementation"])
        if mode is None:
            continue
        rows.append(
            {
                "implementation": "bbtools",
                "mode": mode,
                "batch": int(source["batch"]),
                "threads": int(source["threads"]),
                "trial": int(source["trial"]),
                "samples": None,
                "iterations": int(source["iterations"]),
                "seconds_per_batch": float(source["seconds_per_batch"]),
                "comparisons_per_second": float(source["comparisons_per_second"]),
                "device": None,
            }
        )
    return rows


def main(
    output: Annotated[
        Path, typer.Option(dir_okay=False, help="Normalized combined JSON output")
    ] = Path("results/pairwise-batch-comparison.json"),
    build_dir: Annotated[
        Path, typer.Option(file_okay=False, help="Configured Meson build directory")
    ] = Path("build"),
    bbtools_jar: Annotated[
        Path, typer.Option(dir_okay=False, help="Pinned BBTools jar")
    ] = Path("subprojects/bbmap/bbtools.jar"),
    threads: Annotated[
        int, typer.Option(min=1, help="BBTools parallel worker count")
    ] = available_cpus(),
    samples: Annotated[
        int, typer.Option(min=1, help="NVBench samples per batch size")
    ] = 30,
    iterations: Annotated[
        int, typer.Option(min=1, help="BBTools batches timed per trial")
    ] = 10,
    trials: Annotated[int, typer.Option(min=1, help="BBTools timing trials")] = 5,
    java_heap_gb: Annotated[
        int, typer.Option(min=1, help="Maximum JVM heap in GiB")
    ] = 8,
) -> None:
    """Build and run both implementations, then publish one JSON result."""
    build_dir = project_path(build_dir)
    bbtools_jar = project_path(bbtools_jar)
    output = project_path(output)
    java_source = ROOT / "benchmarks/BBToolsPairwiseBatchBenchmark.java"
    classes = build_dir / "bbtools-pairwise-batch-classes"

    if not bbtools_jar.is_file():
        raise typer.BadParameter(f"BBTools jar does not exist: {bbtools_jar}")
    run(["meson", "compile", "-C", str(build_dir), "cuddl-pairwise-batch-benchmark"])
    cuddl_benchmark = build_dir / "benchmarks/cuddl-pairwise-batch-benchmark"
    if not cuddl_benchmark.is_file():
        raise typer.BadParameter(f"cuDDL benchmark was not built: {cuddl_benchmark}")

    shutil.rmtree(classes, ignore_errors=True)
    classes.mkdir(parents=True)
    run(
        [
            "javac",
            "-cp",
            str(bbtools_jar),
            "-d",
            str(classes),
            str(java_source),
        ]
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="cuddl-pairwise-batch-") as temporary:
        nvbench_csv = Path(temporary) / "nvbench.csv"
        run(
            [
                str(cuddl_benchmark),
                "--stopping-criterion",
                "sample-count",
                "--target-samples",
                str(samples),
                "--min-samples",
                str(samples),
                "--no-batch",
                "--csv",
                str(nvbench_csv),
                "--quiet",
            ]
        )
        java = run(
            [
                "java",
                f"-Xmx{java_heap_gb}g",
                "-cp",
                os.pathsep.join((str(bbtools_jar), str(classes))),
                "BBToolsPairwiseBatchBenchmark",
                str(threads),
                str(iterations),
                str(trials),
            ],
            capture=True,
        )
        cuddl_rows, system = read_nvbench(nvbench_csv)
        rows = cuddl_rows + read_bbtools(java.stdout)

    expected = len(NVBENCH_MODES) + len(BBTOOLS_MODES)
    if len({(row["implementation"], row["mode"]) for row in rows}) != expected:
        raise RuntimeError("one or more benchmark modes produced no rows")
    rows.sort(
        key=lambda row: (
            row["implementation"],
            row["mode"],
            cast(int, row["batch"]),
            cast(int | None, row["trial"]) or -1,
        )
    )
    result = make_result(
        name="Pairwise batch comparison",
        operation="pairwise_batch",
        scope="end_to_end",
        system=system,
        measurements=measurements_from_rows(
            rows,
            case_fields=("mode", "batch", "threads", "trial"),
            omit_fields=("device",),
        ),
    )
    write_result(output, result)
    typer.echo(f"Saved {len(rows)} rows to {output}")


if __name__ == "__main__":
    typer.run(main)
