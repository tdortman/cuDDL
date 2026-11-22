#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "matplotlib",
#   "numpy",
#   "typer",
# ]
# ///
"""Redraws the paper subfigures in paper/figures and the README's docs/images/all-to-all.png
from the benchmark results files."""

import csv
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
from matplotlib.patches import Patch

# acmart sets the paper in Linux Libertine. LaTeX renders the text with the class's own
# packages, so labels and math match the body text.
mpl.rcParams.update(
    {
        "text.usetex": True,
        "text.latex.preamble": r"\usepackage[T1]{fontenc}\usepackage{libertine}"
        r"\usepackage[libertine]{newtxmath}",
        "font.family": "serif",
        "font.size": 9,
        "axes.labelsize": 10,
        "axes.titlesize": 10,
        "xtick.labelsize": 8.5,
        "ytick.labelsize": 8.5,
        "legend.fontsize": 8,
        "figure.titlesize": 10,
        "pdf.use14corefonts": True,
        "grid.alpha": 0.4,
        "hatch.linewidth": 0.6,
        "grid.linestyle": "--",
        "axes.grid": True,
        "axes.axisbelow": True,
    }
)

out_dir = Path("paper/figures")

# Hatch the energy bars that scale 10k measurements to 100k.
SCALED_HATCH = "///"
SCALED_HATCH_COLOR = (0.0, 0.0, 0.0, 0.5)
SCALED_ENERGY_KEY = Patch(
    facecolor="#d9d9d9",
    edgecolor=SCALED_HATCH_COLOR,
    linewidth=0,
    hatch=SCALED_HATCH * 3,
    label="scaled from 10k",
)

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


def cardinality_accuracy(
    hll_accuracy: Path, bbtools_accuracy: Path, uncapped_accuracy: Path
) -> None:
    """cardinality_accuracy.pdf (seeded sets of 2^8 .. 2^24 elements)."""
    powers = np.arange(8, 25)
    trials = {(1 << int(power), trial) for power in powers for trial in range(32)}
    recorded = defaultdict(dict)
    for path in (hll_accuracy, bbtools_accuracy):
        for m in json.loads(path.read_text())["measurements"]:
            case, metrics = m["case"], m["metrics"]
            if case.get("phase") != "accuracy":
                continue
            key = (case["items"], case["trial"])
            if key in trials:
                if metrics["exact"] != case["items"]:
                    raise typer.BadParameter(f"offered count differs from truth in {path}")
                recorded[m["implementation"]["name"]][key] = metrics["signed_error"]
    with uncapped_accuracy.open(newline="") as source:
        uncapped = {
            key: float(row["Signed Error"])
            for row in csv.DictReader(source)
            if row["Estimator"] == "cuddl"
            and (key := (int(row["Items"]), int(row["Trial"]))) in trials
        }

    series = [
        ("cuDDL", uncapped, "-o", "#1f77b4"),
        ("cuco HLL++", recorded["cuco_hll"], ":^", "#d62728"),
    ]
    if uncapped.keys() != trials or any(
        recorded[name].keys() != trials for name in ("cuddl", "cuco_hll", "bbtools")
    ):
        raise typer.BadParameter("expected 32 trials at each set size from 2^8 to 2^24")
    # The archived CSV keeps the estimates before cuDDL applies the offered-count cap.
    if any(
        abs(min(0, raw) - recorded["cuddl"][key]) > 5e-7
        for key, raw in uncapped.items()
    ):
        raise typer.BadParameter("uncapped cuDDL trials do not match the capped JSON")

    fig = plt.figure(figsize=(2.05, 1.34), dpi=300)
    ax = fig.add_axes((0.19, 0.22, 0.77, 0.73))
    for label, errors, style, colour in series:
        means = [
            100 * np.mean([abs(errors[1 << int(p), t]) for t in range(32)])
            for p in powers
        ]
        ax.plot(powers, means, style, color=colour, markersize=3.2, lw=1.1, label=label)
        print(f"{label}: uncapped MARE {np.mean(means):.2f}%")
    for name in ("cuddl", "cuco_hll", "bbtools"):
        capped = [-min(0, e) for e in recorded[name].values()]
        print(f"{name}: matched-cap MARE {100 * np.mean(capped):.2f}%")
    ax.set_xlabel(r"Input cardinality ($\log_2 N$)", fontsize=7.8)
    ax.set_ylabel(r"Uncapped MARE [\%]", fontsize=7.8)
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
    fig.savefig(
        out_dir / "cardinality_accuracy.pdf", bbox_inches="tight", pad_inches=0.02
    )
    fig.savefig(
        out_dir / "cardinality_accuracy.png", bbox_inches="tight", pad_inches=0.02
    )
    plt.close(fig)


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
    """cardinality_latency.pdf (sketch & cardinality speedup over cuco HLL++ on Sys. A and B)."""
    gh_result = json.loads(hll_gh200.read_text())
    m3_result = json.loads(hll_rtx5090.read_text())
    phases = ["construction", "cardinality"]
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

    fig = plt.figure(figsize=(2.05, 1.25), dpi=300)
    ax = fig.add_axes((0.34, 0.25, 0.63, 0.70))
    y_pos = np.arange(len(phases))
    h_bar = 0.30

    b_gh = ax.barh(y_pos - h_bar / 2, sp_gh, h_bar, label="Sys. B", color="#2ca02c")
    b_m3 = ax.barh(y_pos + h_bar / 2, sp_m3, h_bar, label="Sys. A", color="#1f77b4")

    ax.axvline(1.0, color="#666666", linestyle="--", lw=0.8, alpha=0.8)

    for bar in [*b_gh, *b_m3]:
        ax.text(
            bar.get_width() + 0.08,
            bar.get_y() + bar.get_height() / 2,
            f"{bar.get_width():.2f}x",
            va="center",
            ha="left",
            fontsize=7.2,
        )

    ax.set_xlabel("Speedup vs. cuco HLL++", fontsize=7.8)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(["Sketch", "Cardinality"], fontsize=7.8)
    ax.invert_yaxis()
    ax.set_xlim(0, 3.3)
    ax.tick_params(axis="both", labelsize=7.2)
    ax.legend(
        loc="upper right",
        bbox_to_anchor=(1.02, 1.05),
        frameon=False,
        fontsize=6.4,
        borderpad=0.1,
        handlelength=1.0,
    )
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.savefig(
        out_dir / "cardinality_latency.pdf", bbox_inches="tight", pad_inches=0.02
    )
    fig.savefig(
        out_dir / "cardinality_latency.png", bbox_inches="tight", pad_inches=0.02
    )
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
    "ani_ge95": (r"ANI, $\geq$95\%", "#d62728"),
}


def pairwise_error(path: Path) -> None:
    """pairwise_error.pdf (standalone single-column median and p95 absolute error per tool)."""
    summaries = {
        (m["implementation"]["name"], m["case"]["metric"]): m["metrics"]
        for m in json.loads(path.read_text())["measurements"]
        if m["case"].get("measurement") == "error_summary"
    }
    if not summaries:
        raise typer.BadParameter(
            f"{path} has no error summaries, rerun scripts/run_pairwise_accuracy.py"
        )

    fig, ax = plt.subplots(figsize=(3.35, 2.30), dpi=300)
    h_bar = 0.16
    offsets = (
        np.arange(len(PAIRWISE_METRICS)) - (len(PAIRWISE_METRICS) - 1) / 2
    ) * h_bar
    for offset, (metric, (label, colour)) in zip(offsets, PAIRWISE_METRICS.items()):
        rows, medians, p95s = [], [], []
        for row, (name, tool) in enumerate(PAIRWISE_TOOLS.items()):
            summary = summaries.get((name, metric))
            if summary is None:
                continue
            rows.append(row + offset)
            medians.append(100 * summary["absolute_error_median"])
            p95s.append(100 * summary["absolute_error_p95"])
            print(
                f"{tool:15} {label:13} median {medians[-1]:.3f}%  "
                f"p95 {p95s[-1]:.3f}%  n={summary['count']}"
            )
        ax.barh(rows, p95s, h_bar, color=colour, label=label)
        ax.plot(medians, rows, "o", color="black", ms=2.0)
    ax.plot([], [], "o", color="black", ms=2.0, label="Median")
    ax.set_yticks(range(len(PAIRWISE_TOOLS)), PAIRWISE_TOOLS.values())
    ax.set_ylim(len(PAIRWISE_TOOLS) - 0.5, -0.5)
    ax.set_xscale("log")
    ax.set_xlim(0.005, 200)
    ax.set_xticks([0.01, 0.1, 1, 10, 100], ["0.01", "0.1", "1", "10", "100"])
    ax.set_xlabel(r"Abs. error, p95 (bars) and median (dots) [pp]", fontsize=8.2)
    ax.tick_params(axis="x", labelsize=7.6)
    ax.tick_params(axis="y", labelsize=7.6, length=0)
    ax.grid(True, axis="x")
    ax.grid(False, axis="y")
    fig.legend(
        *ax.get_legend_handles_labels(),
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.54, 1.01),
        ncols=3,
        fontsize=6.8,
        handlelength=1.0,
        handletextpad=0.35,
        columnspacing=0.8,
        labelspacing=0.2,
        borderpad=0.1,
    )
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout(pad=0.25, rect=(0, 0, 1, 0.89))
    fig.savefig(out_dir / "pairwise_error.pdf", bbox_inches="tight", pad_inches=0.02)
    fig.savefig(out_dir / "pairwise_error.png", bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def build_medians(paths: list[Path]) -> list[tuple[str, float, float, float]]:
    """(category, cpu, gpu, automatic) median build seconds per ablation label, warm before cold
    per system; automatic is NaN where the label has no automatic builds."""
    seconds: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    owner: dict[str, Path] = {}
    for path in paths:
        for line in path.read_text().splitlines():
            row = json.loads(line) if line.strip() else {}
            if row.get("suite") != "build" or row.get("config") not in ("cpu", "gpu", "automatic"):
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
        if system is None or not {"cpu", "gpu"} <= set(by_config):
            raise typer.BadParameter(
                f"ablation label {label!r} has no system or lacks cpu/gpu builds"
            )
        bars.append(
            (
                f"{system}\n{medium or 'warm'}",
                median(by_config["cpu"]),
                median(by_config["gpu"]),
                median(by_config["automatic"]) if "automatic" in by_config else float("nan"),
            )
        )
    return sorted(
        bars, key=lambda bar: (bar[0].split("\n")[0], not bar[0].endswith("warm"))
    )


def io_backend_build(ablation: list[Path]) -> None:
    """io_backend_build.pdf (host vs. device inflation, warm cache and cold storage)."""
    bars = build_medians(ablation)
    categories = [bar[0] for bar in bars]
    cpu_vals = [bar[1] for bar in bars]
    gpu_vals = [bar[2] for bar in bars]
    auto_vals = [bar[3] for bar in bars]

    fig = plt.figure(figsize=(2.40, 1.50), dpi=300)
    ax = fig.add_axes((0.18, 0.23, 0.78, 0.72))
    x = np.arange(len(categories)) * 1.2
    width = 0.34

    rects1 = ax.bar(
        x - 0.36, cpu_vals, width, label="cpu (libdeflate)", color="#2ca02c"
    )
    rects2 = ax.bar(x, gpu_vals, width, label="gpu (nvCOMP)", color="#1f77b4")
    rects3 = ax.bar(
        x + 0.36, auto_vals, width, label="zero-copy host (GH200)", color="#9467bd"
    )
    for rects, vals in ((rects1, cpu_vals), (rects2, gpu_vals), (rects3, auto_vals)):
        labels = ["" if np.isnan(v) else f"{v:.1f}" for v in vals]
        ax.bar_label(rects, labels=labels, padding=1.5, fontsize=4.6)

    ax.set_ylabel("Build time [s]", fontsize=8.2)
    ax.set_xticks(x)
    ax.set_xticklabels(categories, fontsize=7.5)
    # Headroom keeps the three-row legend clear of the tallest bar labels.
    ax.set_ylim(0, 44)
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
    fig.savefig(out_dir / "io_backend_build.pdf", bbox_inches="tight", pad_inches=0.02)
    fig.savefig(out_dir / "io_backend_build.png", bbox_inches="tight", pad_inches=0.02)
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
                "wall_s": (
                    ops["micro-sketch"]["timings"]["wall"]["median_ms"] * sketch_scale
                    + compare["timings"]["wall"]["median_ms"] * compare_scale
                )
                / 1e3,
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


def micro_ops() -> list[tuple[str, list[float | None]]]:
    """micro_ops_100k.pdf (100k RefSeq resident and E2E times across 5090 and GH200).

    Returns each row's label and its SKETCH, COMPARE, SEARCH, TOP-K, and E2E seconds."""
    mon03 = {}
    for p in [
        Path("results/mon03-micro-perf-100k.json"),
        # cuDDL again after `cuddl-pipeline-benchmark` stopped requesting match counts.
        Path("results/mon03-micro-perf-100k-cuddl.json"),
        Path("results/mon03-micro-perf-10k-hypergen-skani.json"),
        # HyperGen on every pair of 100k, replacing its rows from the 10k file above.
        Path("results/mon03-micro-perf-100k-hypergen.json"),
        Path("results/mon03-micro-perf-10k-cub-exact.json"),
    ]:
        mon03.update(load_tool_ops(p))
    gh200 = {}
    for p in [
        Path("results/gh200-micro-perf-100k-cuddl.json"),
        Path("results/gh200-micro-perf-100k-pp-sketchlib.json"),
        Path("results/gh200-micro-perf-100k-hypergen.json"),
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

    def measured(ops: dict, key: str) -> list[float | None]:
        """A tool run on every pair of the 100k corpus."""
        sketch, compare = ops[(key, "micro-sketch")], ops[(key, "micro-compare")]
        search, top_k = (
            ops.get((key, op), (None,))[0] for op in ("micro-search", "micro-top-k")
        )
        return [sketch[0], compare[0], search, top_k, sketch[1] + compare[1]]

    def mash_ops(path: Path) -> list[float | None]:
        """Mash 2.3 process wall times: the 100k sketch as measured, and `mash triangle` (each
        unordered pair once) on 10k genomes scaled to 100k by the pair ratio."""
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        sketch = next(r["wall_s"] for r in rows if r["step"] == "sketch" and r["genomes"] == "100k")
        triangle = median(
            r["wall_s"] for r in rows if r["step"] == "triangle" and r["genomes"] == "10k"
        )
        compare = triangle * (100_000 * 99_999) / (10_000 * 9_999)
        return [sketch, compare, compare, None, sketch + compare]

    tools_list = [
        (
            "cuDDL (Sys. A)",
            [
                mon03.get(("cuddl:reference-db", "micro-sketch"), (None,))[0],
                0.496,  # results/mon03-deltas-100k.jsonl (real 100k RefSeq, no-delta LSB LOP3)
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
                0.832,  # results/lop3-isolation-gh200.jsonl (real 100k RefSeq, no-delta LSB LOP3)
                0.326,  # results/m6-search-topk-lop3-gh200.log (indexed SEARCH m_min=5)
                0.327,  # results/m6-search-topk-lop3-gh200.log (indexed TOP-K K=3, m_min=5)
                gh200.get(("cuddl:reference-db", "micro-sketch"), (0, 0))[1]
                + gh200.get(("cuddl:gpu", "micro-compare"), (0, 0))[1],
            ],
        ),
        (
            "cuDDL-CPU (Sys. A)",
            [
                # All-CPU ingest from gzip FASTA (read, inflate, pack, SIMD sketch), median of
                # results/mon03-cpu-micro-e2e.jsonl.
                12.91,
                3.881,  # results/mon03-cpu-bitsliced-100k.jsonl (AVX-512 vpternlogd, 120t median)
                None,
                None,
                # Ingest plus dense search and write with deltas, O_DIRECT and fsync
                # (results/mon03-cpu-final-100k.jsonl, tile 64).
                12.91 + 3.326,
            ],
        ),
        (
            "SimdSketch (tiled)",
            [
                mon03.get(("simdsketch:bucket-b8", "micro-sketch"), (None,))[0],
                11.087,  # results/mon03-simdsketch-tiled-100k.log (tile=128, bulk 8 MiB write)
                11.087,
                None,
                46.28,  # 29.15 s sketch + 17.13 s tiled bulk-write compare
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
        ("Mash*", mash_ops(Path("results/mon03-mash.jsonl"))),
        ("HyperGen (CPU)", measured(mon03, "hypergen-cpu:hdc")),
        ("HyperGen GPU (Sys. A)", measured(mon03, "hypergen-gpu:hdc")),
        ("HyperGen GPU (Sys. B)", measured(gh200, "hypergen-gpu:hdc")),
        ("skani*", from_10k(mon03, "skani:cpu")),
        ("cub-exact* (Sys. A)", from_10k(mon03, "cub-exact:gpu-exact")),
        ("cub-exact* (Sys. B)", from_10k(gh200, "cub-exact:gpu-exact")),
    ]

    ops = ["SKETCH\n(res.)", "COMPARE\n(res.)", "SEARCH\n(res.)", "TOP-K\n(res.)", "E2E\n(file)"]
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
    fig, ax = plt.subplots(figsize=(3.35, 2.38), dpi=300)
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
                else f"{seconds:.1f}"
                if seconds < 100
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
            fontsize=6.8,
            color="white" if dark else "black",
        )
    ax.set_xticks(range(len(ops)), ops, fontsize=6.4)
    ax.xaxis.tick_top()
    ax.set_yticks(
        range(len(rows)),
        [name.replace("*", r"\textsuperscript{*}") for name, _ in rows],
        fontsize=7.0,
    )
    ax.tick_params(length=0, pad=1.5)
    ax.set_xticks(np.arange(len(ops) + 1) - 0.5, minor=True)
    ax.set_yticks(np.arange(len(rows) + 1) - 0.5, minor=True)
    ax.grid(which="minor", color="white", linestyle="-", linewidth=0.6)
    ax.grid(which="major", visible=False)
    ax.tick_params(which="minor", length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    plt.tight_layout(pad=0.2)
    fig.savefig(out_dir / "micro_ops_100k.pdf", bbox_inches="tight", pad_inches=0.02)
    fig.savefig(out_dir / "micro_ops_100k.png", bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    for name, vals in tools_list:
        print(
            f"{name:22} "
            + "  ".join(f"{s:10.4g} s" if s is not None else "      None" for s in vals)
        )
    return tools_list


def readme_all_to_all(tools: list[tuple[str, list[float | None]]]) -> None:
    """docs/images/all-to-all.png: each tool's 100k COMPARE time on System A, for the README.

    One row per tool, its fastest variant, so every bar ran on the same workstation. cub-exact
    is the exact k-mer baseline, not a released tool, so the README leaves it out."""
    best: dict[str, float] = {}
    for name, times in tools:
        if "Sys. B" in name or name.startswith("cub-exact") or times[1] is None:
            continue
        tool = name.replace("*", "").split(" (")[0].removesuffix(" GPU")
        tool += "*" if "*" in name else ""
        best[tool] = min(best.get(tool, np.inf), times[1])
    rows = sorted(best.items(), key=lambda row: row[1], reverse=True)
    reference = best["cuDDL"]
    fig, ax = plt.subplots(figsize=(6.5, 3.4))
    bars = ax.barh(
        [name for name, _ in rows],
        [seconds for _, seconds in rows],
        color=[TOOLS["cuddl"][1] if name == "cuDDL" else "#9e9e9e" for name, _ in rows],
    )
    labels = [
        (f"{seconds:.3g}" if seconds < 100 else f"{seconds:,.0f}")
        + " s"
        + ("" if name == "cuDDL" else rf", {seconds / reference:,.0f}$\times$ slower")
        for name, seconds in rows
    ]
    ax.bar_label(bars, labels=labels, padding=3)
    ax.set_xscale("log")
    ax.set_xlim(right=max(best.values()) * 60)
    ax.set_xlabel("Seconds to compare all pairs of 100,000 genomes, log scale")
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    path = Path("docs/images/all-to-all.png")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)



def micro_energy(runs: list[dict]) -> None:
    """micro_energy_100k.pdf (standalone single-column 100k RefSeq stage energy)."""
    fig, ax = plt.subplots(figsize=(3.35, 2.15), dpi=300)
    # (name, dyn_sketch_cpu, dyn_sketch_gpu, dyn_comp_cpu, dyn_comp_gpu, tot_energy, wall_s), highest first
    # so the lowest is at the top.
    energy_rows = sorted(
        ((run["label"], *run["energy"], run["wall_s"]) for run in runs),
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
        bars = ax.barh(y, vals_arr, left=left, height=0.64, label=seg_label, color=color)
        for bar, row in zip(bars, energy_rows):
            if row[0].endswith("*"):
                bar.set_hatch(SCALED_HATCH)
                bar.set_edgecolor(SCALED_HATCH_COLOR)
                bar.set_linewidth(0)
        left += vals_arr
    def fmt_time(sec: float) -> str:
        if sec >= 10_000:
            return f"{sec / 1e3:.0f}ks"
        if sec >= 1_000:
            return f"{sec / 1e3:.1f}ks"
        return f"{sec:.0f}s"

    for yi, r in zip(y.tolist(), energy_rows):
        dyn_tot = sum(r[1:5])
        tot = r[5]
        wall_s = r[6]
        t_str = fmt_time(wall_s)
        if tot > cap:
            star = "*" if r[0].endswith("*") else ""
            d_mj, t_mj = dyn_tot / 1e3, tot / 1e3
            mj_str = f"{d_mj:.0f}/{t_mj:.0f}" if t_mj >= 10 else f"{d_mj:.1f}/{t_mj:.1f}"
            ax.text(
                cap + 6,
                yi,
                rf"\textbf{{$\rightarrow$ {mj_str} MJ{star}}} ({t_str})",
                va="center",
                ha="left",
                fontsize=5.6,
                color="#222222",
            )
        else:
            ax.text(
                tot + 6,
                yi,
                f"{dyn_tot:.0f}/{tot:.0f} ({t_str})" if tot >= 100 else f"{dyn_tot:.1f}/{tot:.1f} ({t_str})",
                va="center",
                ha="left",
                fontsize=5.7,
            )
    ax.set_yticks(y)
    ax.set_yticklabels(names, fontsize=7.2)
    ax.set_xlim(0, 1020)
    ax.set_xlabel("Energy: dyn. / total [kJ] (runtime)", fontsize=8.0)
    ax.tick_params(axis="x", labelsize=7.2)
    ax.legend(
        handles=[
            *(Patch(facecolor=color, label=label) for label, color in segments),
            SCALED_ENERGY_KEY,
        ],
        ncol=1,
        frameon=False,
        loc="upper right",
        bbox_to_anchor=(1.04, 1.0),
        fontsize=5.6,
        borderpad=0.0,
        handlelength=0.6,
        handletextpad=0.2,
        labelspacing=0.10,
    )
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout(pad=0.3)
    fig.savefig(out_dir / "micro_energy_100k.pdf", bbox_inches="tight", pad_inches=0.02)
    fig.savefig(out_dir / "micro_energy_100k.png", bbox_inches="tight", pad_inches=0.02)
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
    uncapped_accuracy: Annotated[Path, typer.Option(exists=True, dir_okay=False)] = Path(
        "results/cuddl-hll-accuracy.csv"
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
    # System B's builds come from the rerun on System A's 100k list, which adds automatic;
    # System C's from the one MOGON job that timed warm and Lustre builds.
    build_ablation = existing(
        [
            Path("results/ablation-gh200-alist.jsonl"),
            Path("results/ablation-mogon-813965.jsonl"),
        ],
        "--ablation",
    )
    stage = existing(
        stage
        or [
            # skani and cub-exact run every pair of 10k genomes and are extrapolated to
            # 100k. pp-sketchlib runs at its five default k-mer lengths. The oct26 SimdSketch
            # lanes use the October 2026 corpus and are reported in the text only.
            *(
                Path(path)
                for path in sorted(glob("results/mogon-*-micro-energy-100k-*.json"))
                if not any(t in path for t in ("skani", "cub-exact", "k17,29", "oct26"))
            ),
            *(
                Path(path)
                for path in sorted(glob("results/mogon-*-micro-energy-10k-*.json"))
                if "hypergen" not in path
            ),
        ],
        "--stage",
    )
    runs = stage_runs(stage)

    out_dir.mkdir(parents=True, exist_ok=True)
    cardinality_accuracy(hll_accuracy, bbtools_accuracy, uncapped_accuracy)
    cardinality_latency(hll_gh200, hll_rtx5090, hll_items)
    pairwise_error(pairwise_accuracy)
    io_backend_build(build_ablation)
    readme_all_to_all(micro_ops())
    micro_energy(runs)
    print(f"Regenerated all paper subfigures in {out_dir}.")


if __name__ == "__main__":
    typer.run(main)
