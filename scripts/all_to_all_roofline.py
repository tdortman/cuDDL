#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = ["typer"]
# ///
"""Hardware ceilings for the exhaustive all-to-all search from Nsight Compute counters.

Reads an `ncu --csv --page raw` export of every launch of one complete search
(the kernels of all tiles) and the pair count that search produced. Sums each
kernel's counters, derives instructions and bytes per pair, and turns them into
pairs/s ceilings for a GPU spec: the issue bound (one warp instruction per SM
sub-partition per clock), the integer ALU pipe bound (half a warp instruction
per sub-partition per clock for LOP3, the search's main instruction: 16 INT32
lanes on Ampere GA10x, and Blackwell GB20x's ALU Heavy sub-pipe, measured with
benchmarks/alu_power_ceiling.cu), the FMA pipe bound (half a
warp instruction per clock) and the DRAM bound. The ceilings assume the target
GPU executes the same instructions per pair as the profiled one; compare the
SASS of both architectures before relying on that.
"""

import csv
import io
from pathlib import Path
from typing import Annotated

import typer

GPUS = {
    # name: (SMs, sustained SM clock in GHz, ALU warp instructions per sub-partition per clock,
    # DRAM GB/s)
    "a40": (84, 1.74, 0.5, 696.0),
    "rtx5070ti": (70, 2.63, 0.5, 896.0),
}

COUNTERS = [
    "gpu__time_duration.sum",
    "smsp__inst_executed.sum",
    "smsp__thread_inst_executed.sum",
    "sm__inst_executed_pipe_alu.sum",
    "sm__inst_executed_pipe_fma.sum",
    "sm__inst_executed_pipe_lsu.sum",
    "lts__t_bytes.sum",
    "dram__bytes.sum",
]
AVERAGES = [
    "smsp__issue_active.avg.pct_of_peak_sustained_active",
    "sm__throughput.avg.pct_of_peak_sustained_elapsed",
    "sm__warps_active.avg.pct_of_peak_sustained_active",
    "sm__pipe_alu_cycles_active.avg.pct_of_peak_sustained_active",
]


def number(text: str) -> float:
    return float(text.replace(",", "")) if text not in ("", "n/a") else 0.0


def short_name(kernel: str) -> str:
    name = kernel.split("<")[0].split("(")[0]
    return name.removeprefix("void ").split("::")[-1]


def load(path: Path) -> dict[str, dict[str, float]]:
    """Counters summed per kernel, and duration-weighted averages of the percentages."""
    lines = [
        line
        for line in path.read_text().splitlines(keepends=True)
        if line.startswith('"')
    ]
    rows = list(csv.reader(io.StringIO("".join(lines))))
    header = rows[0]
    kernels: dict[str, dict[str, float]] = {}
    for row in rows[2:]:
        values = dict(zip(header, row))
        if not values.get("ID", "").isdigit():
            continue
        name = short_name(values["Kernel Name"])
        totals = kernels.setdefault(name, {"launches": 0.0})
        totals["launches"] += 1
        duration = number(values["gpu__time_duration.sum"])
        for counter in COUNTERS:
            totals[counter] = totals.get(counter, 0.0) + number(values.get(counter, ""))
        for average in AVERAGES:
            totals[average] = (
                totals.get(average, 0.0) + number(values.get(average, "")) * duration
            )
    for totals in kernels.values():
        for average in AVERAGES:
            totals[average] /= totals["gpu__time_duration.sum"]
    return kernels


def main(
    metrics: Annotated[
        Path, typer.Argument(exists=True, help="ncu --csv --page raw export")
    ],
    pairs: Annotated[int, typer.Option(help="Pairs the profiled search produced")],
    gpu: Annotated[
        list[str] | None,
        typer.Option(help=f"GPU specs: {', '.join(GPUS)} [default: all]"),
    ] = None,
    clock_ghz: Annotated[
        list[float] | None,
        typer.Option(help="Override sustained clocks, in --gpu order"),
    ] = None,
    measured_gpairs: Annotated[
        list[float] | None, typer.Option(help="Measured Gpairs/s, in --gpu order")
    ] = None,
) -> None:
    gpu = gpu or list(GPUS)
    kernels = load(metrics)
    total = {c: sum(k[c] for k in kernels.values()) for c in COUNTERS}
    typer.secho(f"Per kernel ({pairs:,} pairs)", bold=True)
    header = [
        "kernel",
        "launches",
        "ms",
        "warp inst/pair",
        "thread inst/pair",
        "ALU/pair",
        "FMA/pair",
        "LSU/pair",
        "L2 B/pair",
        "DRAM B/pair",
        "issue %",
        "SM %",
        "occupancy %",
        "ALU busy %",
    ]
    rows = []
    for name, k in [
        *kernels.items(),
        ("total", {**total, "launches": sum(k["launches"] for k in kernels.values())}),
    ]:
        rows.append(
            [
                name,
                int(k["launches"]),
                f"{k['gpu__time_duration.sum'] / 1e6:.2f}",
                f"{k['smsp__inst_executed.sum'] / pairs:.3f}",
                f"{k['smsp__thread_inst_executed.sum'] / pairs:.2f}",
                f"{k['sm__inst_executed_pipe_alu.sum'] / pairs:.3f}",
                f"{k['sm__inst_executed_pipe_fma.sum'] / pairs:.3f}",
                f"{k['sm__inst_executed_pipe_lsu.sum'] / pairs:.3f}",
                f"{k['lts__t_bytes.sum'] / pairs:.2f}",
                f"{k['dram__bytes.sum'] / pairs:.3f}",
                *(f"{k[a]:.1f}" if a in k else "" for a in AVERAGES),
            ]
        )
    widths = [max(len(str(c)) for c in col) for col in zip(header, *rows)]
    for row in [header, *rows]:
        typer.echo("  " + "  ".join(str(c).rjust(w) for c, w in zip(row, widths)))

    warp = total["smsp__inst_executed.sum"] / pairs
    alu = total["sm__inst_executed_pipe_alu.sum"] / pairs
    fma = total["sm__inst_executed_pipe_fma.sum"] / pairs
    dram = total["dram__bytes.sum"] / pairs
    typer.secho("\nCeilings, assuming the profiled instructions per pair", bold=True)
    header = [
        "GPU",
        "SMs",
        "GHz",
        "issue Gpairs/s",
        "ALU Gpairs/s",
        "FMA Gpairs/s",
        "DRAM Gpairs/s",
        "bound",
        "measured",
        "of bound",
    ]
    rows = []
    for i, name in enumerate(gpu):
        sms, clock, alu_rate, dram_gbps = GPUS[name]
        if clock_ghz:
            clock = clock_ghz[i]
        cycles = sms * clock * 1e9
        ceilings = {
            "issue": cycles * 4 / warp,
            "ALU": cycles * 4 * alu_rate / alu if alu else float("inf"),
            "FMA": cycles * 2 / fma if fma else float("inf"),
            "DRAM": dram_gbps * 1e9 / dram if dram else float("inf"),
        }
        bound = min(ceilings, key=ceilings.__getitem__)
        measured = measured_gpairs[i] if measured_gpairs else None
        rows.append(
            [
                name,
                sms,
                f"{clock:.2f}",
                *(f"{ceilings[k] / 1e9:.2f}" for k in ["issue", "ALU", "FMA", "DRAM"]),
                bound,
                f"{measured:.2f}" if measured else "",
                f"{measured * 1e9 / ceilings[bound]:.0%}" if measured else "",
            ]
        )
    widths = [max(len(str(c)) for c in col) for col in zip(header, *rows)]
    for row in [header, *rows]:
        typer.echo("  " + "  ".join(str(c).rjust(w) for c, w in zip(row, widths)))


if __name__ == "__main__":
    typer.run(main)
