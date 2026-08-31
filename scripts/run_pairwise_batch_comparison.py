#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = ["typer"]
# ///
"""Run cuDDL and BBTools batch comparisons into one normalized CSV."""

import csv
import io
import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Annotated

import typer

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
FIELDS = (
    "implementation",
    "mode",
    "batch",
    "threads",
    "trial",
    "samples",
    "iterations",
    "seconds_per_batch",
    "comparisons_per_second",
    "device",
)


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


def read_nvbench(path: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    with path.open(newline="") as stream:
        for source in csv.DictReader(stream):
            mode = NVBENCH_MODES.get(source["Benchmark"])
            if mode is None or source["Skipped"] != "No":
                continue
            batch = int(source["Batch"])
            seconds = float(source["GPU Time (sec)"])
            rows.append(
                {
                    "implementation": "cuddl",
                    "mode": mode,
                    "batch": str(batch),
                    "threads": "0",
                    "trial": "",
                    "samples": source["Samples"],
                    "iterations": "",
                    "seconds_per_batch": f"{seconds:.17g}",
                    "comparisons_per_second": f"{batch / seconds:.17g}",
                    "device": source["Device Name"],
                }
            )
    return rows


def read_bbtools(text: str) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for source in csv.DictReader(io.StringIO(text)):
        mode = BBTOOLS_MODES.get(source["implementation"])
        if mode is None:
            continue
        rows.append(
            {
                "implementation": "bbtools",
                "mode": mode,
                "batch": source["batch"],
                "threads": source["threads"],
                "trial": source["trial"],
                "samples": "",
                "iterations": source["iterations"],
                "seconds_per_batch": source["seconds_per_batch"],
                "comparisons_per_second": source["comparisons_per_second"],
                "device": "",
            }
        )
    return rows


def main(
    output: Annotated[
        Path, typer.Option(dir_okay=False, help="Normalized combined CSV output")
    ] = Path("results/pairwise-batch-comparison.csv"),
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
    """Build and run both implementations, then publish one CSV."""
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
        rows = read_nvbench(nvbench_csv) + read_bbtools(java.stdout)

    expected = len(NVBENCH_MODES) + len(BBTOOLS_MODES)
    if len({(row["implementation"], row["mode"]) for row in rows}) != expected:
        raise RuntimeError("one or more benchmark modes produced no rows")
    rows.sort(
        key=lambda row: (
            row["implementation"],
            row["mode"],
            int(row["batch"]),
            int(row["trial"] or -1),
        )
    )
    with tempfile.NamedTemporaryFile(
        "w", newline="", dir=output.parent, prefix=output.name, delete=False
    ) as stream:
        temporary_output = Path(stream.name)
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    temporary_output.replace(output)
    typer.echo(f"Saved {len(rows)} rows to {output}")


if __name__ == "__main__":
    typer.run(main)
