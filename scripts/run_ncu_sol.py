#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = ["typer"]
# ///
"""How close each GPU tool's kernels come to the hardware's peak throughputs.

For every --reference-count and GPU tool, the tool's `run_micro_comparison.py --performance-only`
lane runs twice over:

1. under Nsight Systems, for every kernel's share of the run's GPU time and its launch count;
2. once per kernel that matters (the largest ones up to --coverage of GPU time, at most
   --max-kernels) under Nsight Compute, which records the speed-of-light throughputs below for
   --launches launches from the middle of the run, then ends the run.

Each kernel becomes one line of OUTPUT_DIR/kernels.jsonl: its time share and the duration-weighted
mean of each throughput, as a percentage of the unit's peak. scripts/plot_ncu_sol.py draws them.

The pairs a lane compares are capped at --max-pairs, a stride over every pair, so a 100k corpus
stays affordable: per-launch throughputs do not depend on how many pairs a run compares, but the
time shares do, so they describe the capped run.

    ./scripts/run_ncu_sol.py /localscratch/genomes --reference-count 1000 \\
      --reference-count 10000 --output-dir results/ncu-sol-gh200
"""

import csv
import io
import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Annotated

import typer

ROOT = Path(__file__).resolve().parent.parent
MICRO = ROOT / "scripts/run_micro_comparison.py"
# The throughputs of Nsight Compute's GPU Speed Of Light section, as a percentage of peak.
METRICS = {
    "sm": "sm__throughput.avg.pct_of_peak_sustained_elapsed",
    "memory": "gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed",
    "l1": "l1tex__throughput.avg.pct_of_peak_sustained_active",
    "l2": "lts__throughput.avg.pct_of_peak_sustained_elapsed",
    "dram": "gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed",
}
DURATION = "gpu__time_duration.sum"

app = typer.Typer(add_completion=False)


def number(text: str) -> float:
    return float(text.replace(",", "")) if text not in ("", "n/a") else 0.0


def csv_rows(text: str, first_column: str) -> list[dict[str, str]]:
    """Rows of a CSV report whose header starts with @p first_column, skipping the lines a tool
    prints before it."""
    lines = text.splitlines(keepends=True)
    start = next(i for i, line in enumerate(lines) if line.lstrip('"').startswith(first_column))
    return list(csv.DictReader(io.StringIO("".join(lines[start:]))))


def tool_path(name: str) -> str:
    path = shutil.which(name) or next(
        (str(p) for p in sorted(Path("/usr/local").glob(f"cuda*/bin/{name}"), reverse=True)), None
    )
    if path is None:
        raise typer.BadParameter(f"{name} is neither on PATH nor in /usr/local/cuda*/bin")
    return path


def kernel_times(nsys: str, command: list[str], scratch: Path) -> list[dict]:
    """Kernels by GPU time, largest first, under their base names as Nsight Compute matches them."""
    report = scratch / "timeline"
    subprocess.run(
        [nsys, "profile", "--trace=cuda", "--sample=none", "--cpuctxsw=none", "--force-overwrite=true",
         "-o", str(report), *command],
        check=True,
    )
    summary = subprocess.run(
        [nsys, "stats", "--report", "cuda_gpu_kern_sum:base", "--format", "csv", "--output", "-",
         "--force-export=true", f"{report}.nsys-rep"],
        check=True, capture_output=True, text=True,
    ).stdout
    kernels: dict[str, dict] = {}
    for row in csv_rows(summary, "Time (%)"):
        entry = kernels.setdefault(row["Name"], {"kernel": row["Name"], "ns": 0.0, "launches": 0})
        entry["ns"] += number(row["Total Time (ns)"])
        entry["launches"] += int(number(row["Instances"]))
    total = sum(k["ns"] for k in kernels.values())
    ranked = sorted(kernels.values(), key=lambda k: k["ns"], reverse=True)
    for entry in ranked:
        entry["time_share"] = entry["ns"] / total if total else 0.0
    return ranked


def kernel_throughputs(
    ncu: str, command: list[str], kernel: dict, launches: int, scratch: Path
) -> tuple[int, dict[str, float]]:
    """Profiled launches and the duration-weighted mean throughput per metric."""
    base = scratch / "kernel"
    for stale in scratch.glob("kernel.ncu-rep*"):
        stale.unlink()
    count = min(launches, kernel["launches"])
    # Launches from the middle of the run: the first ones often warm caches or size buffers.
    skip = min((kernel["launches"] - count) // 2, 20)
    # --kill ends the run once the launches are profiled, so the lane reports a failure; only the
    # report matters here.
    subprocess.run(
        [ncu, "--target-processes", "all", "--kernel-name", f"regex:^{kernel['kernel']}$",
         "--launch-skip", str(skip), "--launch-count", str(count), "--kill", "yes",
         "--metrics", ",".join([DURATION, *METRICS.values()]), "-o", str(base),
         *command],
        check=False,
    )
    # Recent Nsight Compute versions write compressed .ncu-repz reports.
    report = next(scratch.glob("kernel.ncu-rep*"), None)
    if report is None:
        raise RuntimeError(f"Nsight Compute profiled no launch of {kernel['kernel']}")
    rows = csv_rows(subprocess.run(
        [ncu, "--import", str(report), "--csv", "--page", "raw"],
        check=True, capture_output=True, text=True,
    ).stdout, "ID")
    rows = [row for row in rows if row["ID"].isdigit()]
    weight = sum(number(row[DURATION]) for row in rows)
    means = {
        name: sum(number(row[metric]) * number(row[DURATION]) for row in rows) / weight
        for name, metric in METRICS.items()
    }
    return len(rows), means


@app.command()
def main(
    genomes: Annotated[Path, typer.Argument(exists=True, file_okay=False, help="Genome directory")],
    output_dir: Annotated[Path, typer.Option(help="kernels.jsonl and the lanes' reports land here")],
    reference_count: Annotated[
        list[int], typer.Option(min=2, help="Hashed sample of N genomes, all-to-all")
    ] = [1000, 10000, 100000],
    tools: Annotated[str, typer.Option(help="GPU tools to profile")] = (
        "cuddl,cub-exact,pp-sketchlib,hypergen-gpu"
    ),
    max_pairs: Annotated[int, typer.Option(min=1, help="Pairs a lane compares at most")] = 10_000_000,
    coverage: Annotated[
        float, typer.Option(min=0, max=1, help="Profile kernels up to this share of GPU time")
    ] = 0.9,
    max_kernels: Annotated[int, typer.Option(min=1, help="Kernels profiled per lane at most")] = 5,
    launches: Annotated[int, typer.Option(min=1, help="Launches profiled per kernel")] = 10,
    threads: Annotated[int, typer.Option(min=1, help="Host threads per lane")] = len(
        os.sched_getaffinity(0)
    ),
) -> None:
    nsys, ncu = tool_path("nsys"), tool_path("ncu")
    output_dir.mkdir(parents=True, exist_ok=True)
    kernels_path = output_dir / "kernels.jsonl"
    for count in reference_count:
        pairs = min(max_pairs, count * (count - 1) // 2)
        for tool in filter(None, tools.split(",")):
            command = [
                str(MICRO), str(genomes), "--reference-count", str(count), "--topology",
                "all-to-all", "--performance-only", "--tools", tool, "--samples", "1",
                "--warmups", "0", "--threads", str(threads), "--idle-seconds", "0",
                "--max-pairs", str(pairs), "--output", str(output_dir / f"{tool}-{count}.json"),
            ]
            with tempfile.TemporaryDirectory(prefix="ncu-sol-") as scratch:
                ranked = kernel_times(nsys, command, Path(scratch))
                covered = 0.0
                for kernel in ranked[:max_kernels]:
                    if covered >= coverage:
                        break
                    covered += kernel["time_share"]
                    profiled, means = kernel_throughputs(ncu, command, kernel, launches,
                                                         Path(scratch))
                    with kernels_path.open("a") as stream:
                        stream.write(json.dumps({
                            "tool": tool, "genomes": count, "pairs": pairs,
                            "kernel": kernel["kernel"], "time_share": kernel["time_share"],
                            "launches": kernel["launches"], "profiled_launches": profiled,
                            **means,
                        }) + "\n")


if __name__ == "__main__":
    app()
