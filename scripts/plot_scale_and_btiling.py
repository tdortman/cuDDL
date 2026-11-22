#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "matplotlib>=3.9",
#   "numpy>=2.0",
#   "typer>=0.12",
# ]
# ///
"""Plot a two-panel single-column figure for Section 5.3:
(a) B-dimension tiling linearity at N = 100k (B = 2,048, 4,096, 8,192):
    H100 untiled vs. H100 B-tiled vs. RTX 5090 B-tiled.
(b) Single-GPU resident throughput from 100k to 1.5M genomes (1.125 Tpairs)
    across RTX 5090, H100, and A100.
"""

import json
import statistics
from pathlib import Path
from typing import Annotated

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import typer

mpl.rcParams.update(
    {
        "text.usetex": True,
        "text.latex.preamble": r"\usepackage[T1]{fontenc}\usepackage{libertine}"
        r"\usepackage[libertine]{newtxmath}",
        "font.family": "serif",
        "pdf.use14corefonts": True,
        "ps.useafm": True,
    }
)


def load_bucket_medians(path: Path) -> dict[int, float]:
    by_b: dict[int, list[float]] = {}
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        by_b.setdefault(int(row["buckets"]), []).append(float(row["ms"]) / 1000.0)
    return {b: statistics.median(vals) for b, vals in sorted(by_b.items())}


def load_100k_lsb_gpairs(path: Path) -> float:
    vals = []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("variant") == "bitsliced_lop3_lsb":
            vals.append(float(row["gpairs_s"]))
    return statistics.median(vals)


def load_scale_1m(path: Path) -> dict[int, tuple[float, float]]:
    out: dict[int, tuple[float, float]] = {}
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        out[int(row["n"])] = (float(row["gpairs_s"]), float(row["ms"]) / 1000.0)
    return out


def main(
    results_dir: Annotated[
        Path,
        typer.Option("--results-dir", "-r", help="Directory with JSONL benchmark results."),
    ] = Path("results"),
    output: Annotated[
        Path,
        typer.Option("--output", "-o", help="Output PDF path."),
    ] = Path("paper/figures/scale_and_btiling.pdf"),
) -> None:
    """Generate the two-panel B-tiling and million-genome scaling figure."""
    h100_untiled = load_bucket_medians(results_dir / "gh200-bucket-sweep-100k.jsonl")
    h100_tiled = load_bucket_medians(results_dir / "gh200-bucket-sweep-v4-100k.jsonl")
    rtx_tiled = load_bucket_medians(results_dir / "mon03-bucket-sweep-v4-100k.jsonl")

    rtx_100k = load_100k_lsb_gpairs(results_dir / "mon03-cmp-ablation-100k-v5.jsonl")
    h100_100k = load_100k_lsb_gpairs(results_dir / "alist/gh200-cmp-ablation-100k-v5.jsonl")
    a100_100k = load_100k_lsb_gpairs(results_dir / "mogon-a100-cmp-ablation-100k-874821.jsonl")

    rtx_1m = load_scale_1m(results_dir / "mon03-scale-1m.jsonl")
    h100_1m = load_scale_1m(results_dir / "gh200-scale-1m.jsonl")
    a100_1m = load_scale_1m(results_dir / "mogon-a100-scale-1m.jsonl")

    # Okabe-Ito colorblind-safe palette
    c_rtx = "#0072B2"   # blue
    c_h100 = "#009E73"  # bluish green
    c_a100 = "#D55E00"  # vermilion
    c_unt = "#7F7F7F"   # neutral gray for untiled baseline

    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(3.35, 1.78), constrained_layout=True
    )

    # =========================================================================
    # Panel (a): B-dimension tiling at N = 100k (runtime [s], zero baseline)
    # =========================================================================
    bs = [2048, 4096, 8192]
    x_a = np.arange(len(bs))

    y_h_unt = [h100_untiled[b] for b in bs]
    y_h_til = [h100_tiled[b] for b in bs]
    y_r_til = [rtx_tiled[b] for b in bs]
    # Ideal O(B) reference anchored at H100 B=2048
    y_ideal_h = [y_h_til[0] * (b / 2048.0) for b in bs]

    ax1.plot(
        x_a, y_ideal_h, color="#9CA3AF", ls=":", lw=1.0, zorder=1, label=r"Ideal $O(B)$"
    )
    ax1.plot(
        x_a,
        y_h_unt,
        color=c_unt,
        ls="--",
        marker="^",
        ms=4.0,
        lw=1.3,
        label="H100 untiled",
        zorder=2,
    )
    ax1.plot(
        x_a,
        y_h_til,
        color=c_h100,
        ls="-",
        marker="s",
        ms=4.0,
        lw=1.5,
        label=r"H100 $B$-tiled",
        zorder=3,
    )
    ax1.plot(
        x_a,
        y_r_til,
        color=c_rtx,
        ls="-",
        marker="o",
        ms=4.0,
        lw=1.5,
        label=r"5090 $B$-tiled",
        zorder=4,
    )

    # Direct speedup callout at B = 8192
    ax1.annotate(
        "",
        xy=(2.0, y_h_til[2] + 0.20),
        xytext=(2.0, y_h_unt[2] - 0.22),
        arrowprops=dict(arrowstyle="<->", color="#374151", lw=0.8),
        zorder=5,
    )
    ax1.text(
        2.06,
        0.5 * (y_h_til[2] + y_h_unt[2]),
        r"2.0$\times$",
        fontsize=6.8,
        color="#111827",
        ha="left",
        va="center",
    )

    ax1.set_xticks(x_a)
    ax1.set_xticklabels(["2k", "4k", "8k"], fontsize=7.0)
    ax1.set_xlim(-0.2, 2.52)
    ax1.set_ylim(0.0, 8.2)
    ax1.set_yticks([0, 2, 4, 6, 8])
    ax1.tick_params(axis="y", labelsize=7.0)
    ax1.set_xlabel(r"Buckets $B$ ($N = 100\text{k}$)", fontsize=7.3)
    ax1.set_ylabel("Runtime [s]", fontsize=7.3)
    ax1.set_title(r"(a) $B$-dimension tiling", fontsize=7.6, pad=3)
    ax1.grid(axis="y", color="#E5E7EB", lw=0.5, zorder=0)
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)
    ax1.legend(
        loc="upper left",
        frameon=False,
        fontsize=6.1,
        handlelength=1.4,
        labelspacing=0.22,
        borderpad=0.1,
    )

    # =========================================================================
    # Panel (b): Million-genome throughput stability (Gpairs/s, zero baseline)
    # =========================================================================
    x_b = np.array([0, 1, 2])
    y_rtx_s = [rtx_100k, rtx_1m[1_000_000][0], rtx_1m[1_500_000][0]]
    y_h100_s = [h100_100k, h100_1m[1_000_000][0], h100_1m[1_500_000][0]]
    y_a100_s = [a100_100k, a100_1m[1_000_000][0], a100_1m[1_500_000][0]]

    ax2.plot(
        x_b, y_rtx_s, color=c_rtx, ls="-", marker="o", ms=4.0, lw=1.5, label="RTX 5090"
    )
    ax2.plot(
        x_b, y_h100_s, color=c_h100, ls="-", marker="s", ms=4.0, lw=1.5, label="H100"
    )
    ax2.plot(
        x_b, y_a100_s, color=c_a100, ls="-", marker="D", ms=3.8, lw=1.5, label="A100"
    )

    # Annotate 1.5M wall times cleanly next to final points
    ax2.annotate(
        f"{rtx_1m[1_500_000][1]:.0f}\\,s",
        (2.0, y_rtx_s[2]),
        xytext=(0, 5),
        textcoords="offset points",
        fontsize=6.4,
        color=c_rtx,
        ha="center",
    )
    ax2.annotate(
        f"{h100_1m[1_500_000][1]:.0f}\\,s",
        (2.0, y_h100_s[2]),
        xytext=(0, -10),
        textcoords="offset points",
        fontsize=6.4,
        color=c_h100,
        ha="center",
    )
    ax2.annotate(
        f"{a100_1m[1_500_000][1]:.0f}\\,s",
        (2.0, y_a100_s[2]),
        xytext=(0, -10),
        textcoords="offset points",
        fontsize=6.4,
        color=c_a100,
        ha="center",
    )

    ax2.set_xticks(x_b)
    ax2.set_xticklabels(["100k", "1.0M", "1.5M"], fontsize=7.0)
    ax2.set_xlim(-0.2, 2.3)
    ax2.set_ylim(0.0, 11.8)
    ax2.set_yticks([0, 3, 6, 9])
    ax2.tick_params(axis="y", labelsize=7.0)
    ax2.set_xlabel(r"Archive size $N$ ($B = 2{,}048$)", fontsize=7.3)
    ax2.set_ylabel("Gpairs/s", fontsize=7.3)
    ax2.set_title("(b) Million-genome scale", fontsize=7.6, pad=3)
    ax2.grid(axis="y", color="#E5E7EB", lw=0.5, zorder=0)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)
    ax2.legend(
        loc="lower left",
        frameon=False,
        fontsize=6.1,
        handlelength=1.4,
        labelspacing=0.22,
        borderpad=0.1,
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=300)
    plt.close(fig)
    print(f"Wrote {output}")


if __name__ == "__main__":
    typer.run(main)
