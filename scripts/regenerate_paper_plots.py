#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "matplotlib",
#   "numpy",
#   "typer",
# ]
# ///
"""Redraws the paper subfigures in paper/figures from the benchmark results files."""

import json
import sys
from collections import defaultdict
from collections.abc import Sequence
from glob import glob
from pathlib import Path
from statistics import median
from typing import Annotated

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import typer

mpl.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["Linux Libertine O", "Linux Libertine", "DejaVu Serif"],
        "font.size": 9,
        "axes.labelsize": 10,
        "axes.titlesize": 10,
        "xtick.labelsize": 8.5,
        "ytick.labelsize": 8.5,
        "legend.fontsize": 8,
        "figure.titlesize": 10,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "grid.alpha": 0.4,
        "grid.linestyle": "--",
        "axes.grid": True,
        "axes.axisbelow": True,
    }
)

out_dir = Path("paper/figures")

# Paper system letter for each ablation label prefix.
SYSTEMS = {"GH200": "Sys. B", "MOGON": "Sys. C"}
# Micro comparison tools: implementation name, or name:variant for a variant drawn apart ->
# (legend label, colour).
TOOLS = {
    "cuddl": ("cuDDL", "#1f77b4"),
    "simdsketch": ("SimdSketch", "#2ca02c"),
    "rabbitsketch": ("RabbitSketch", "#8c564b"),
    "bbtools": ("BBTools", "#ff7f0e"),
    "dashing2": ("Dashing2", "#9467bd"),
    "sketchlib": ("sketchlib.rust", "#e377c2"),
    "pp-sketchlib": ("pp-sketchlib", "#17becf"),
    "cub-exact": ("cub-exact", "#7f7f7f"),
    "hypergen-cpu": ("HyperGen (CPU)", "#bcbd22"),
    "hypergen-gpu": ("HyperGen (GPU)", "#dbdb8d"),
    "skani": ("skani", "#d62728"),
}


def save_with_bar_labels(
    fig: plt.Figure,
    stem: str,
    bar_label_pairs: Sequence[tuple] = (),
    pad: float = 0.3,
) -> None:
    plt.tight_layout(pad=pad)
    if bar_label_pairs:
        fig.canvas.draw()
        min_fs = min(
            text.get_fontsize() for _, texts in bar_label_pairs for text in texts
        )
        for bars, texts in bar_label_pairs:
            for bar, text in zip(bars, texts):
                max_w = bar.get_window_extent().width * 0.90
                while (
                    text.get_window_extent().width > max_w and text.get_fontsize() > 4.5
                ):
                    text.set_fontsize(text.get_fontsize() * 0.92)
                min_fs = min(min_fs, text.get_fontsize())
        for _, texts in bar_label_pairs:
            for text in texts:
                text.set_fontsize(min_fs)
    fig.savefig(out_dir / f"{stem}.pdf")
    fig.savefig(out_dir / f"{stem}.png")
    plt.close(fig)


def accuracy_series(path: Path) -> dict[str, list[float]]:
    """Mean absolute cardinality error in percent per implementation, by ascending set size."""
    errors: dict[str, dict[int, list[float]]] = defaultdict(lambda: defaultdict(list))
    for m in json.loads(path.read_text())["measurements"]:
        if m["case"].get("phase") == "accuracy":
            errors[m["implementation"]["name"]][m["case"]["items"]].append(
                m["metrics"]["absolute_error"]
            )
    return {
        name: [100 * float(np.mean(by_items[n])) for n in sorted(by_items)]
        for name, by_items in errors.items()
    }


def cardinality_accuracy(hll_accuracy: Path, bbtools_accuracy: Path) -> None:
    """cardinality_accuracy.pdf (seeded sets of 2^8 .. 2^24 elements)."""
    ours = accuracy_series(hll_accuracy)
    bbtools_acc = accuracy_series(bbtools_accuracy)["bbtools"]
    cuddl_acc, cuco_acc = ours["cuddl"], ours["cuco_hll"]
    if not len(cuddl_acc) == len(bbtools_acc) == len(cuco_acc) == 17:
        raise typer.BadParameter("expected 17 set sizes (2^8 .. 2^24) in each series")
    fig = plt.figure(figsize=(2.3, 1.7), dpi=300)
    ax = fig.add_axes((0.19, 0.22, 0.77, 0.73))
    powers = np.arange(8, 25)
    ax.plot(
        powers, cuddl_acc, "-o", color="#1f77b4", markersize=3.2, lw=1.1, label="cuDDL"
    )
    ax.plot(
        powers,
        bbtools_acc,
        "--s",
        color="#ff7f0e",
        markersize=3.0,
        lw=1.1,
        label="BBTools DDL",
    )
    ax.plot(
        powers,
        cuco_acc,
        ":^",
        color="#d62728",
        markersize=3.2,
        lw=1.1,
        label="cuco HLL++",
    )
    ax.set_xlabel(r"Input cardinality ($\log_2 N$)", fontsize=7.8)
    ax.set_ylabel("Mean abs. error [%]", fontsize=7.8)
    ax.set_xticks([8, 12, 16, 20, 24])
    ax.tick_params(axis="both", labelsize=7.2)
    ax.set_ylim(-0.05, 3.1)
    ax.legend(
        frameon=False,
        loc="upper left",
        fontsize=6.3,
        handlelength=1.2,
        labelspacing=0.15,
    )
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.savefig(out_dir / "cardinality_accuracy.pdf")
    fig.savefig(out_dir / "cardinality_accuracy.png")
    plt.close(fig)
    print(
        "mean abs. error: "
        + ", ".join(
            f"{name} {np.mean(series):.2f}%"
            for name, series in [
                ("cuDDL", cuddl_acc),
                ("BBTools", bbtools_acc),
                ("cuco", cuco_acc),
            ]
        )
    )


def get_hll_time(data: dict, estimator: str, phase: str, items: int) -> float:
    (row,) = (
        m
        for m in data["measurements"]
        if "timings" in m
        and m["implementation"]["name"] == estimator
        and m["case"]["phase"] == phase
        and m["case"]["items"] == items
    )
    return row["timings"]["gpu"]["median_ms"] * 1e3


def cardinality_latency(hll_gh200: Path, hll_rtx5090: Path, hll_items: int) -> None:
    """cardinality_latency.pdf (speedup over cuco HLL++ across GH200 and RTX 5090)."""
    gh_result = json.loads(hll_gh200.read_text())
    m3_result = json.loads(hll_rtx5090.read_text())
    phases = ["construction", "cardinality", "similarity"]
    sp_gh = [
        get_hll_time(gh_result, "cuco_hll", p, hll_items)
        / get_hll_time(gh_result, "cuddl", p, hll_items)
        for p in phases
    ]
    sp_m3 = [
        get_hll_time(m3_result, "cuco_hll", p, hll_items)
        / get_hll_time(m3_result, "cuddl", p, hll_items)
        for p in phases
    ]

    fig = plt.figure(figsize=(2.3, 1.7), dpi=300)
    ax = fig.add_axes((0.34, 0.22, 0.63, 0.73))
    y_pos = np.arange(len(phases))
    h_bar = 0.32

    b_gh = ax.barh(y_pos - h_bar / 2, sp_gh, h_bar, label="Sys. B", color="#2ca02c")
    b_m3 = ax.barh(y_pos + h_bar / 2, sp_m3, h_bar, label="Sys. A", color="#1f77b4")

    ax.axvline(1.0, color="#666666", linestyle="--", lw=0.8, alpha=0.8)

    for bar in [*b_gh, *b_m3]:
        ax.text(
            bar.get_width() + 0.35,
            bar.get_y() + bar.get_height() / 2,
            f"{bar.get_width():.2f}x",
            va="center",
            ha="left",
            fontsize=7.2,
        )

    ax.set_xlabel("Speedup vs. cuco HLL++", fontsize=7.8)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(["Sketch", "Cardinality", "Similarity"], fontsize=7.8)
    ax.invert_yaxis()
    ax.set_xlim(0, 15.0)
    ax.tick_params(axis="both", labelsize=7.2)
    ax.legend(
        loc="upper right",
        bbox_to_anchor=(1.02, 1.02),
        frameon=False,
        fontsize=6.2,
        borderpad=0.1,
        handlelength=1.0,
    )
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.savefig(out_dir / "cardinality_latency.pdf")
    fig.savefig(out_dir / "cardinality_latency.png")
    plt.close(fig)


# Pairwise accuracy rows, top to bottom: implementation name -> label. Tools that estimate only
# Jaccard similarity, core distance or alignment ANI have no containment, completeness or WKID.
# The exact-set rows apply the two main k-mer formulas to the exact sets: their ANI error is the
# formula's alone, the floor every sketch on that formula shares.
PAIRWISE_TOOLS = {
    "cuddl": "cuDDL",
    "bbtools": "BBTools",
    "dashing2": "Dashing2",
    "rabbitsketch": "RabbitSketch",
    "cuco_hll": "cuco HLL++",
    "simdsketch": "SimdSketch",
    "sketchlib": "sketchlib.rust",
    "pp-sketchlib": "pp-sketchlib",
    "hypergen": "HyperGen",
    "skani": "skani",
    "exact-wkid-root": "Exact, WKID$^{1/k}$",
    "exact-poisson": "Exact, Mash",
}
# `ani_sketch` scores each tool's ANI against its own formula over the exact k-mer sets, the
# sketch's share of the error; `ani_ge95` scores it against ANIm on pairs of at least 95% ANI.
PAIRWISE_METRICS = {
    "containment": ("Containment", "#2ca02c"),
    "completeness": ("Completeness", "#9467bd"),
    "wkid": ("WKID", "#5b9bd5"),
    "ani_sketch": ("ANI, sketch", "#ff8c38"),
    "ani_ge95": ("ANI, ≥95%", "#d62728"),
}
# Only DDL estimates completeness from the sketch comparison itself; the benchmark derives the
# other tools' completeness from two cardinality estimates, so it is not drawn for them.
COMPLETENESS_TOOLS = {"cuddl", "bbtools"}


def pairwise_error(path: Path) -> None:
    """pairwise_error.pdf (median and p95 absolute error per tool and metric)."""
    summaries = {
        (m["implementation"]["name"], m["case"]["metric"]): m["metrics"]
        for m in json.loads(path.read_text())["measurements"]
        if m["case"].get("measurement") == "error_summary"
    }
    if not summaries:
        raise typer.BadParameter(
            f"{path} has no error summaries, rerun scripts/run_pairwise_accuracy.py"
        )

    fig, ax = plt.subplots(figsize=(2.3, 2.2), dpi=300)
    h_bar = 0.16
    offsets = (
        np.arange(len(PAIRWISE_METRICS)) - (len(PAIRWISE_METRICS) - 1) / 2
    ) * h_bar
    for offset, (metric, (label, colour)) in zip(offsets, PAIRWISE_METRICS.items()):
        rows, medians, p95s = [], [], []
        for row, (name, tool) in enumerate(PAIRWISE_TOOLS.items()):
            # Absent where the tool has no such metric. Only DDL estimates completeness.
            summary = summaries.get((name, metric))
            if summary is None or (
                metric == "completeness" and name not in COMPLETENESS_TOOLS
            ):
                continue
            rows.append(row + offset)
            medians.append(100 * summary["absolute_error_median"])
            p95s.append(100 * summary["absolute_error_p95"])
            print(
                f"{tool:15} {label:13} median {medians[-1]:.3f}%  "
                f"p95 {p95s[-1]:.3f}%  n={summary['count']}"
            )
        ax.barh(rows, p95s, h_bar, color=colour, label=label)
        ax.plot(medians, rows, "o", color="black", ms=1.6)
    ax.plot([], [], "o", color="black", ms=1.6, label="Median")
    ax.set_yticks(range(len(PAIRWISE_TOOLS)), PAIRWISE_TOOLS.values())
    ax.set_ylim(len(PAIRWISE_TOOLS) - 0.5, -0.5)
    # Errors span four decades, from sketch-only ANI below 1% to sketches that miss divergent
    # pairs entirely (ANI near 0, an 85% error).
    ax.set_xscale("log")
    ax.set_xlim(0.005, 200)
    ax.set_xticks([0.01, 0.1, 1, 10, 100], ["0.01", "0.1", "1", "10", "100"])
    ax.set_xlabel("Abs. error, p95 (bars) [%]", fontsize=7.8)
    ax.tick_params(axis="x", labelsize=7.2)
    ax.tick_params(axis="y", labelsize=7.2, length=0)
    ax.grid(True, axis="x")
    ax.grid(False, axis="y")
    # Long bars reach into every corner, so the legend sits above the axes, centred on the
    # figure because the tool labels push the axes right.
    fig.legend(
        *ax.get_legend_handles_labels(),
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.0),
        ncols=3,
        fontsize=6.0,
        handlelength=0.9,
        handletextpad=0.3,
        columnspacing=0.6,
        labelspacing=0.15,
        borderpad=0.1,
    )
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout(pad=0.2, rect=(0, 0, 1, 0.88))
    fig.savefig(out_dir / "pairwise_error.pdf")
    fig.savefig(out_dir / "pairwise_error.png")
    plt.close(fig)


def build_medians(paths: list[Path]) -> list[tuple[str, float, float]]:
    """(category, cpu, gpu) median build seconds per ablation label, warm before cold per system."""
    seconds: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    owner: dict[str, Path] = {}
    for path in paths:
        for line in path.read_text().splitlines():
            row = json.loads(line) if line.strip() else {}
            if row.get("suite") != "build" or row.get("config") not in ("cpu", "gpu"):
                continue
            label = row["label"]
            if owner.setdefault(label, path) != path:
                raise typer.BadParameter(
                    f"label {label!r} in both {owner[label]} and {path}"
                )
            seconds[label][row["config"]].append(row["seconds"])
    bars = []
    for label, by_config in seconds.items():
        host, _, medium = label.partition(", ")
        system = next(
            (s for prefix, s in SYSTEMS.items() if host.startswith(prefix)), None
        )
        if system is None or set(by_config) != {"cpu", "gpu"}:
            raise typer.BadParameter(
                f"ablation label {label!r} has no system or lacks cpu/gpu builds"
            )
        bars.append(
            (
                f"{system}\n{medium or 'warm'}",
                median(by_config["cpu"]),
                median(by_config["gpu"]),
            )
        )
    return sorted(
        bars, key=lambda bar: (bar[0].split("\n")[0], not bar[0].endswith("warm"))
    )


def b300_inflate(csv: Path) -> None:
    """b300_inflate.pdf (B300 hardware DE vs. SM nvCOMP isolated inflation throughput)."""
    import pandas as pd

    df_inf = pd.read_csv(csv)
    df_inf["gbps"] = df_inf["GlobalMem BW (bytes/sec)"] / 1e9

    fig = plt.figure(figsize=(2.45, 1.8), dpi=300)
    ax = fig.add_axes((0.18, 0.23, 0.78, 0.72))
    # The CSV names each input by its staging path on the B300 host.
    inputs_order = ["genomes-10k-spread", "genomes-10k-spread-bgzf-zlib"]
    input_labels = ["Standard gzip\n($\\leq 4$ MiB)", "BGZF\n(bgzip)"]
    x = np.arange(len(inputs_order))
    w = 0.35
    sm_gbps = [
        df_inf[df_inf["input"].str.endswith(inp) & (df_inf["backend"] == "sm")][
            "gbps"
        ].iloc[0]
        for inp in inputs_order
    ]
    eng_gbps = [
        df_inf[df_inf["input"].str.endswith(inp) & (df_inf["backend"] == "engine")][
            "gbps"
        ].iloc[0]
        for inp in inputs_order
    ]
    b1 = ax.bar(x - w / 2, sm_gbps, w, label="SMs (nvCOMP)", color="#ff7f0e")
    b2 = ax.bar(x + w / 2, eng_gbps, w, label="Hardware DE", color="#1f77b4")
    ax.bar_label(b1, fmt="%.0f", padding=2, fontsize=6.8)
    ax.bar_label(b2, fmt="%.0f", padding=2, fontsize=6.8)
    ax.set_xticks(x)
    ax.set_xticklabels(input_labels, fontsize=7.5)
    ax.set_ylabel("Inflate throughput [GB/s]", fontsize=8.2)
    ax.set_ylim(0, 270)
    ax.tick_params(axis="both", labelsize=7.5)
    ax.legend(
        loc="upper right",
        frameon=False,
        fontsize=6.5,
        handlelength=1.0,
        borderpad=0.1,
        labelspacing=0.15,
    )
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.savefig(out_dir / "b300_inflate.pdf")
    fig.savefig(out_dir / "b300_inflate.png")
    plt.close(fig)


def io_backend_build(ablation: list[Path]) -> None:
    """io_backend_build.pdf (host vs. device inflation, warm cache and cold storage)."""
    bars = build_medians(ablation)
    categories = [bar[0] for bar in bars]
    cpu_vals = [bar[1] for bar in bars]
    gpu_vals = [bar[2] for bar in bars]

    fig = plt.figure(figsize=(2.45, 1.8), dpi=300)
    ax = fig.add_axes((0.18, 0.23, 0.78, 0.72))
    x = np.arange(len(categories))
    width = 0.35

    rects1 = ax.bar(
        x - width / 2, cpu_vals, width, label="cpu (libdeflate)", color="#2ca02c"
    )
    rects2 = ax.bar(
        x + width / 2, gpu_vals, width, label="gpu (nvCOMP)", color="#1f77b4"
    )
    ax.bar_label(rects1, fmt="%.1f", padding=2, fontsize=6.8)
    ax.bar_label(rects2, fmt="%.1f", padding=2, fontsize=6.8)

    ax.set_ylabel("Build time [s]", fontsize=8.2)
    ax.set_xticks(x)
    ax.set_xticklabels(categories, fontsize=7.5)
    ax.set_ylim(0, 36)
    ax.tick_params(axis="both", labelsize=7.5)
    ax.legend(
        frameon=False,
        loc="upper left",
        fontsize=6.5,
        borderpad=0.1,
        handlelength=1.0,
        handletextpad=0.3,
        labelspacing=0.15,
    )
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.savefig(out_dir / "io_backend_build.pdf")
    fig.savefig(out_dir / "io_backend_build.png")
    plt.close(fig)


def b300_build(log: Path, inputs: list[str], workers: int) -> None:
    """b300_build_100k.pdf (B300 build time per backend on standard gzip and BGZF)."""
    seconds: dict[tuple[str, str], list[float]] = defaultdict(list)
    for line in log.read_text().splitlines():
        if line.startswith("RESULT "):
            f = dict(field.split("=", 1) for field in line.split()[1:])
            if int(f["workers"]) == workers:
                seconds[(f["input"], f["backend"])].append(float(f["seconds"]))
    backends = ["cpu", "hybrid", "gpu-cores", "gpu"]
    missing = [(i, b) for i in inputs for b in backends if (i, b) not in seconds]
    if missing:
        raise typer.BadParameter(f"{log} has no {workers}-worker RESULT for {missing}")
    cpu, hybrid, gpu_cores, gpu_de = (
        [median(seconds[(i, b)]) for i in inputs] for b in backends
    )

    fig = plt.figure(figsize=(2.45, 1.8), dpi=300)
    ax = fig.add_axes((0.18, 0.23, 0.78, 0.72))
    categories = ["Standard\ngzip", "BGZF\n(bgzip)"]
    x = np.array([0.0, 1.10])
    w = 0.22

    b1 = ax.bar(x - 1.5 * w, cpu, w, label="cpu", color="#2ca02c")
    b2 = ax.bar(x - 0.5 * w, hybrid, w, label="hybrid", color="#9467bd")
    b3 = ax.bar(x + 0.5 * w, gpu_cores, w, label="gpu-cores", color="#ff7f0e")
    b4 = ax.bar(x + 1.5 * w, gpu_de, w, label="gpu (DE)", color="#1f77b4")

    for b in (b1, b2, b3, b4):
        for bar in b:
            val = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                val + 0.8,
                f"{val:.1f}",
                ha="center",
                va="bottom",
                fontsize=5.0,
            )

    ax.set_ylabel("Build time [s]", fontsize=8.2)
    ax.set_xticks(x)
    ax.set_xticklabels(categories, fontsize=7.5)
    ax.set_ylim(0, 48)
    ax.set_xlim(-0.60, 1.70)
    ax.tick_params(axis="both", labelsize=7.5)
    ax.legend(
        ncol=2,
        frameon=False,
        loc="upper left",
        fontsize=6.5,
        borderpad=0.1,
        handlelength=1.0,
        handletextpad=0.3,
        labelspacing=0.15,
        columnspacing=0.5,
    )
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.savefig(out_dir / "b300_build_100k.pdf")
    fig.savefig(out_dir / "b300_build_100k.png")
    plt.close(fig)


def stage_runs(paths: list[Path]) -> list[dict]:
    """Per tool: label, colour, and SKETCH/COMPARE energy in kJ.

    Every run compares all pairs of its corpus. A tool run on fewer genomes than the largest
    corpus is extrapolated to it and starred: SKETCH by the genome ratio, COMPARE by the
    all-to-all pair ratio.
    """
    by_tool: dict[str, dict[str, dict]] = defaultdict(dict)
    for path in paths:
        for m in json.loads(path.read_text())["measurements"]:
            name, op = m["implementation"]["name"], m["case"]["measurement"]
            variant = f"{name}:{m['implementation'].get('variant')}"
            name = variant if variant in TOOLS else name
            if op in by_tool[name]:
                raise typer.BadParameter(
                    f"{name} {op} appears twice among --stage files"
                )
            if m["case"]["topology"] != "all-to-all":
                raise typer.BadParameter(f"{name} {op} is not an all-to-all run")
            by_tool[name][op] = m
    if unknown := set(by_tool) - set(TOOLS):
        raise typer.BadParameter(f"no label for tools {sorted(unknown)}")
    if missing := [TOOLS[t][0] for t in TOOLS if t not in by_tool]:
        print(f"warning: no stage results for {', '.join(missing)}", file=sys.stderr)
    target = max(ops["micro-sketch"]["case"]["genomes"] for ops in by_tool.values())
    runs = []
    for name, ops in sorted(
        by_tool.items(), key=lambda item: list(TOOLS).index(item[0])
    ):
        compare = ops["micro-compare"]
        pairs = compare["metrics"]
        genomes = ops["micro-sketch"]["case"]["genomes"]
        sketch_scale = target / genomes
        compare_scale = target * (target - 1) / (genomes * (genomes - 1))
        if pairs.get("pairs_evaluated") and pairs.get("pairs_total"):
            compare_scale *= pairs["pairs_total"] / pairs["pairs_evaluated"]
        label, color = TOOLS[name]
        sketch_cpu, sketch_gpu, sketch_total = stage_energy(
            ops["micro-sketch"], sketch_scale
        )
        compare_cpu, compare_gpu, compare_total = stage_energy(compare, compare_scale)
        sampled = genomes < target or (
            pairs.get("pairs_evaluated")
            and pairs.get("pairs_evaluated") != pairs.get("pairs_total")
        )
        runs.append(
            {
                "label": label + ("*" if sampled else ""),
                "color": color,
                "energy": (
                    sketch_cpu,
                    sketch_gpu,
                    compare_cpu,
                    compare_gpu,
                    sketch_total + compare_total,
                ),
            }
        )
    return runs


def stage_energy(m: dict, scale: float) -> tuple[float, float, float]:
    """Dynamic CPU, dynamic GPU and total node energy in kJ, scaled by `scale`.

    CPU-only lanes carry no GPU energy fields.
    """
    mt = m["metrics"]
    factor = scale / 1e3
    return (
        max(mt["energy_dynamic_cpu_j"], 0.0) * factor,
        max(mt.get("energy_dynamic_gpu_j", 0.0), 0.0) * factor,
        (mt["energy_cpu_j"] + mt.get("energy_gpu_j", 0.0)) * factor,
    )


def load_tool_ops(path: Path) -> dict[tuple[str, str], tuple[float, float]]:
    """Loads (tool, op) -> (resident_sec, wall_sec) from benchmark JSON."""
    if not path.exists():
        return {}
    res = {}
    for m in json.loads(path.read_text()).get("measurements", []):
        t = m["implementation"]["name"]
        var = m["implementation"].get("variant")
        op = m["case"]["measurement"]
        key = f"{t}:{var}" if var else t
        sec = m.get("timings", {}).get("resident", {}).get("median_ms", 0) / 1000.0
        wall = m.get("timings", {}).get("wall", {}).get("median_ms", 0) / 1000.0
        res[(key, op)] = (sec, wall)
    return res


def micro_ops() -> None:
    """micro_ops_10k.pdf (100k RefSeq resident and E2E times across 5090 and GH200)."""
    mon03 = {}
    for p in [
        Path("results/mon03-micro-perf-100k.json"),
        # cuDDL again after `cuddl-pipeline-benchmark` stopped requesting match counts.
        Path("results/mon03-micro-perf-100k-cuddl.json"),
        Path("results/mon03-micro-perf-10k-hypergen-skani.json"),
        Path("results/mon03-micro-perf-10k-cub-exact.json"),
    ]:
        mon03.update(load_tool_ops(p))
    gh200 = {}
    for p in [
        Path("results/gh200-micro-perf-100k-cuddl.json"),
        Path("results/gh200-micro-perf-100k-pp-sketchlib.json"),
        Path("results/gh200-micro-perf-10k-hypergen.json"),
        Path("results/gh200-micro-perf-10k-cub-exact.json"),
    ]:
        gh200.update(load_tool_ops(p))

    def from_10k(ops: dict, key: str) -> list[float | None]:
        """A tool run on every pair of 10k genomes, scaled to 100k as `stage_runs` scales its
        energy: SKETCH by the genome ratio, every pairwise operation by the pair ratio."""
        sketch = ops.get((key, "micro-sketch"))
        compare = ops.get((key, "micro-compare"))
        if sketch is None or compare is None:
            return [None] * 5
        genomes, pairs = 10.0, (100_000 * 99_999) / (10_000 * 9_999)
        search, top_k = (
            ops.get((key, op), (None,))[0] for op in ("micro-search", "micro-top-k")
        )
        return [
            sketch[0] * genomes,
            compare[0] * pairs,
            search and search * pairs,
            top_k and top_k * pairs,
            sketch[1] * genomes + compare[1] * pairs,
        ]

    tools_list = [
        (
            "cuDDL (Sys. A)",
            [
                mon03.get(("cuddl:reference-db", "micro-sketch"), (None,))[0],
                mon03.get(("cuddl:gpu", "micro-compare"), (None,))[0],
                mon03.get(("cuddl:gpu-indexed", "micro-search"), (None,))[0],
                mon03.get(("cuddl:gpu-indexed", "micro-top-k"), (None,))[0],
                mon03.get(("cuddl:reference-db", "micro-sketch"), (0, 0))[1]
                + mon03.get(("cuddl:gpu", "micro-compare"), (0, 0))[1],
            ],
        ),
        (
            "cuDDL (Sys. B)",
            [
                gh200.get(("cuddl:reference-db", "micro-sketch"), (None,))[0],
                gh200.get(("cuddl:gpu", "micro-compare"), (None,))[0],
                gh200.get(("cuddl:gpu-indexed", "micro-search"), (None,))[0],
                gh200.get(("cuddl:gpu-indexed", "micro-top-k"), (None,))[0],
                gh200.get(("cuddl:reference-db", "micro-sketch"), (0, 0))[1]
                + gh200.get(("cuddl:gpu", "micro-compare"), (0, 0))[1],
            ],
        ),
        (
            "SimdSketch",
            [
                mon03.get(("simdsketch:bucket-b8", "micro-sketch"), (None,))[0],
                mon03.get(("simdsketch:bucket-b8", "micro-compare"), (None,))[0],
                mon03.get(("simdsketch:bucket-b8", "micro-search"), (None,))[0],
                None,
                mon03.get(("simdsketch:bucket-b8", "micro-sketch"), (0, 0))[1]
                + mon03.get(("simdsketch:bucket-b8", "micro-compare"), (0, 0))[1],
            ],
        ),
        (
            "RabbitSketch",
            [
                mon03.get(("rabbitsketch:FastKMV", "micro-sketch"), (None,))[0],
                mon03.get(("rabbitsketch:FastKMV", "micro-compare"), (None,))[0],
                mon03.get(("rabbitsketch:FastKMV", "micro-search"), (None,))[0],
                None,
                mon03.get(("rabbitsketch:FastKMV", "micro-sketch"), (0, 0))[1]
                + mon03.get(("rabbitsketch:FastKMV", "micro-compare"), (0, 0))[1],
            ],
        ),
        (
            "BBTools",
            [
                mon03.get(("bbtools:ddl", "micro-sketch"), (None,))[0],
                mon03.get(("bbtools:ddl", "micro-compare"), (None,))[0],
                mon03.get(("bbtools:ddl", "micro-search"), (None,))[0],
                mon03.get(("bbtools:ddl", "micro-top-k"), (None,))[0],
                mon03.get(("bbtools:ddl", "micro-sketch"), (0, 0))[1]
                + mon03.get(("bbtools:ddl", "micro-compare"), (0, 0))[1],
            ],
        ),
        (
            "Dashing2",
            [
                mon03.get(("dashing2:SetSketch", "micro-sketch"), (None,))[0],
                mon03.get(("dashing2:SetSketch", "micro-compare"), (None,))[0],
                mon03.get(("dashing2:SetSketch", "micro-search"), (None,))[0],
                None,
                mon03.get(("dashing2:SetSketch", "micro-sketch"), (0, 0))[1]
                + mon03.get(("dashing2:SetSketch", "micro-compare"), (0, 0))[1],
            ],
        ),
        (
            "sketchlib.rust",
            [
                mon03.get(("sketchlib:one-perm-minhash", "micro-sketch"), (None,))[0],
                mon03.get(("sketchlib:one-perm-minhash", "micro-compare"), (None,))[0],
                mon03.get(("sketchlib:one-perm-minhash", "micro-search"), (None,))[0],
                mon03.get(("sketchlib:one-perm-minhash", "micro-top-k"), (None,))[0],
                mon03.get(("sketchlib:one-perm-minhash", "micro-sketch"), (0, 0))[1]
                + mon03.get(("sketchlib:one-perm-minhash", "micro-compare"), (0, 0))[1],
            ],
        ),
        (
            "pp-sketchlib (Sys. A)",
            [
                # pp-sketchlib sketches assemblies on the CPU only; its E2E keeps that time.
                None,
                mon03.get(("pp-sketchlib:core-accessory", "micro-compare"), (None,))[0],
                mon03.get(("pp-sketchlib:core-accessory", "micro-search"), (None,))[0],
                mon03.get(("pp-sketchlib:core-accessory", "micro-top-k"), (None,))[0],
                mon03.get(("pp-sketchlib:core-accessory", "micro-sketch"), (0, 0))[1]
                + mon03.get(("pp-sketchlib:core-accessory", "micro-compare"), (0, 0))[
                    1
                ],
            ],
        ),
        (
            "pp-sketchlib (Sys. B)",
            [
                None,
                gh200.get(("pp-sketchlib:core-accessory", "micro-compare"), (None,))[0],
                gh200.get(("pp-sketchlib:core-accessory", "micro-search"), (None,))[0],
                gh200.get(("pp-sketchlib:core-accessory", "micro-top-k"), (None,))[0],
                gh200.get(("pp-sketchlib:core-accessory", "micro-sketch"), (0, 0))[1]
                + gh200.get(("pp-sketchlib:core-accessory", "micro-compare"), (0, 0))[
                    1
                ],
            ],
        ),
        ("HyperGen (CPU)*", from_10k(mon03, "hypergen-cpu:hdc")),
        ("HyperGen GPU* (Sys. A)", from_10k(mon03, "hypergen-gpu:hdc")),
        ("HyperGen GPU* (Sys. B)", from_10k(gh200, "hypergen-gpu:hdc")),
        ("skani*", from_10k(mon03, "skani:cpu")),
        ("cub-exact* (Sys. A)", from_10k(mon03, "cub-exact:gpu-exact")),
        ("cub-exact* (Sys. B)", from_10k(gh200, "cub-exact:gpu-exact")),
    ]

    ops = ["SKETCH", "COMPARE", "SEARCH", "TOP-K", "E2E"]
    # E2E sums default absent stages to 0 s, so a zero marks a tool without results.
    rows = [
        (name.replace("Sys. ", ""), [v or np.nan for v in vals])
        for name, vals in tools_list
        if any(vals)
    ]
    times = np.array([vals for _, vals in rows])
    # Colour is the slowdown against the fastest tool of each operation, so every column spans
    # its own range instead of sharing one log axis from milliseconds to days.
    slowdown = np.log10(times / np.nanmin(times, axis=0))
    fig, ax = plt.subplots(figsize=(2.9, 2.2), dpi=300)
    ax.imshow(
        np.ma.masked_invalid(slowdown), cmap="YlOrRd", vmin=0, vmax=5.5, aspect="auto"
    )
    for (row, col), seconds in np.ndenumerate(times):
        if np.isnan(seconds):
            text, dark = "–", False
        else:
            text = (
                f"{seconds:.2f}"
                if seconds < 10
                else f"{seconds:.0f}"
                if seconds < 1e4
                else f"{seconds / 1e3:.0f}k"
            )
            dark = slowdown[row, col] > 3.3
        ax.text(
            col,
            row,
            text,
            ha="center",
            va="center",
            fontsize=5.4,
            color="white" if dark else "black",
        )
    ax.set_xticks(range(len(ops)), ops, fontsize=5.0)
    ax.xaxis.tick_top()
    ax.set_yticks(range(len(rows)), [name for name, _ in rows], fontsize=5.6)
    ax.tick_params(length=0, pad=1.5)
    ax.set_xticks(np.arange(len(ops) + 1) - 0.5, minor=True)
    ax.set_yticks(np.arange(len(rows) + 1) - 0.5, minor=True)
    ax.grid(which="minor", color="white", linestyle="-", linewidth=0.6)
    ax.grid(which="major", visible=False)
    ax.tick_params(which="minor", length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    plt.tight_layout(pad=0.2)
    fig.savefig(out_dir / "micro_ops_10k.pdf")
    fig.savefig(out_dir / "micro_ops_10k.png")
    plt.close(fig)
    for name, vals in tools_list:
        print(
            f"{name:22} "
            + "  ".join(f"{s:10.4g} s" if s is not None else "      None" for s in vals)
        )


def micro_energy(runs: list[dict]) -> None:
    """micro_energy_10k.pdf (100k RefSeq stage energy: above-idle dynamic + idle baseline = total)."""
    fig, ax = plt.subplots(figsize=(2.55, 2.2), dpi=300)
    # (name, dyn_sketch_cpu, dyn_sketch_gpu, dyn_comp_cpu, dyn_comp_gpu, tot_energy), highest first
    # so the lowest is at the top.
    energy_rows = sorted(
        ((run["label"], *run["energy"]) for run in runs),
        key=lambda r: r[5],
        reverse=True,
    )
    y = np.arange(len(energy_rows))
    names = [r[0] for r in energy_rows]
    cap = 490.0
    segments = [
        ("SKETCH, CPU", "#9ecae1"),
        ("SKETCH, GPU", "#fdae6b"),
        ("COMPARE, CPU", "#3182bd"),
        ("COMPARE, GPU", "#e6550d"),
        ("Idle baseline", "#d9d9d9"),
    ]
    plot_rows = []
    for r in energy_rows:
        dyn_parts = list(r[1:5])
        dyn_tot = sum(dyn_parts)
        tot = r[5]
        idle = max(tot - dyn_tot, 0.0)
        parts = dyn_parts + [idle]
        if tot <= cap:
            plot_rows.append(parts)
        else:
            scale = cap / tot
            plot_rows.append([p * scale for p in parts])
    left = np.zeros(len(energy_rows))
    for idx, (seg_label, color) in enumerate(segments):
        vals_arr = np.array([row[idx] for row in plot_rows], dtype=float)
        ax.barh(y, vals_arr, left=left, height=0.64, label=seg_label, color=color)
        left += vals_arr
    for yi, r in zip(y.tolist(), energy_rows):
        dyn_tot = sum(r[1:5])
        tot = r[5]
        if tot > cap:
            ax.text(
                cap - 6,
                yi,
                f"→ {dyn_tot / 1e3:.0f}/{tot / 1e3:.0f} MJ*",
                va="center",
                ha="right",
                fontsize=5.8,
                fontweight="bold",
                color="#222222",
            )
        else:
            ax.text(
                tot + 6,
                yi,
                f"{dyn_tot:.1f}/{tot:.1f}",
                va="center",
                ha="left",
                fontsize=6.0,
            )
    ax.set_yticks(y)
    ax.set_yticklabels(names, fontsize=7.6)
    ax.set_xlim(0, 800)
    ax.set_xlabel("Energy: dyn. / total [kJ]", fontsize=8.5)
    ax.tick_params(axis="x", labelsize=7.5)
    ax.legend(
        ncol=1,
        frameon=False,
        loc="upper right",
        bbox_to_anchor=(1.1, 1.0),
        fontsize=5.2,
        borderpad=0.0,
        handlelength=0.6,
        handletextpad=0.2,
        labelspacing=0.12,
    )
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout(pad=0.3)
    fig.savefig(out_dir / "micro_energy_10k.pdf")
    fig.savefig(out_dir / "micro_energy_10k.png")
    plt.close(fig)
    for r in energy_rows[::-1]:
        print(f"{r[0]:15} dynamic {sum(r[1:5]):12.2f} kJ  total {r[5]:12.2f} kJ")


def existing(paths: list[Path], option: str) -> list[Path]:
    if not paths or (absent := [p for p in paths if not p.is_file()]):
        raise typer.BadParameter(
            f"missing files: {absent if paths else 'none matched'}", param_hint=option
        )
    return paths


def main(
    hll_accuracy: Annotated[Path, typer.Option(exists=True, dir_okay=False)] = Path(
        "results/hll-accuracy.json"
    ),
    bbtools_accuracy: Annotated[Path, typer.Option(exists=True, dir_okay=False)] = Path(
        "results/hll-accuracy-bbtools.json"
    ),
    hll_gh200: Annotated[Path, typer.Option(exists=True, dir_okay=False)] = Path(
        "results/gh200-hll.json"
    ),
    hll_rtx5090: Annotated[Path, typer.Option(exists=True, dir_okay=False)] = Path(
        "results/monster3-hll.json"
    ),
    hll_items: Annotated[
        int, typer.Option(help="Items per sketch for the speedup panel.")
    ] = 1 << 28,
    pairwise_accuracy: Annotated[
        Path, typer.Option(exists=True, dir_okay=False)
    ] = Path("results/pairwise-accuracy-refseq1000.json"),
    ablation: Annotated[
        list[Path] | None,
        typer.Option(
            help="Ablation JSONL files [default: results/ablation-gh200.jsonl "
            "results/ablation-mogon-*.jsonl]"
        ),
    ] = None,
    b300_csv: Annotated[Path, typer.Option(exists=True, dir_okay=False)] = Path(
        "results/b300-inflate-10k.csv"
    ),
    b300_log: Annotated[Path, typer.Option(exists=True, dir_okay=False)] = Path(
        "results/b300-build-100k.log"
    ),
    b300_input: Annotated[
        list[str] | None,
        typer.Option(
            help="Standard gzip and BGZF inputs [default: genomes-100k genomes-100k-bgzf-zlib]"
        ),
    ] = None,
    b300_workers: Annotated[int, typer.Option()] = 48,
    stage: Annotated[
        list[Path] | None,
        typer.Option(
            help="Micro comparison energy JSON files [default: "
            "results/mogon-*-micro-energy-10k-*.json and the "
            "results/mogon-*-micro-energy-100k-*.json of every other tool]"
        ),
    ] = None,
) -> None:
    ablation = existing(
        ablation
        or [
            Path("results/ablation-gh200.jsonl"),
            *map(Path, sorted(glob("results/ablation-mogon-*.jsonl"))),
        ],
        "--ablation",
    )
    b300_input = b300_input or ["genomes-100k", "genomes-100k-bgzf-zlib"]
    if len(b300_input) != 2:
        raise typer.BadParameter(
            "give the standard gzip and the BGZF input", param_hint="--b300-input"
        )
    stage = existing(
        stage
        or [
            # HyperGen, skani and cub-exact run every pair of 10k genomes and are
            # extrapolated to 100k. pp-sketchlib runs at its five default k-mer lengths.
            *(
                Path(path)
                for path in sorted(glob("results/mogon-*-micro-energy-100k-*.json"))
                if not any(
                    t in path for t in ("hypergen", "skani", "cub-exact", "k17,29")
                )
            ),
            *map(Path, sorted(glob("results/mogon-*-micro-energy-10k-*.json"))),
        ],
        "--stage",
    )
    runs = stage_runs(stage)

    out_dir.mkdir(parents=True, exist_ok=True)
    cardinality_accuracy(hll_accuracy, bbtools_accuracy)
    cardinality_latency(hll_gh200, hll_rtx5090, hll_items)
    pairwise_error(pairwise_accuracy)
    b300_inflate(b300_csv)
    b300_build(b300_log, b300_input, b300_workers)
    io_backend_build(ablation)
    micro_ops()
    micro_energy(runs)
    print(f"Regenerated all paper subfigures in {out_dir}.")


if __name__ == "__main__":
    typer.run(main)
