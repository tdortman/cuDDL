#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "matplotlib>=3.9",
#   "numpy>=2.0",
#   "typer>=0.12",
# ]
# ///
"""Plot a single-column two-panel bottleneck regime map for one System C A100 node:
(a) Hardware lever: durable write bandwidth vs. archive size N (fixed 4 B dense output).
(b) Software lever: effective output bytes/pair vs. archive size N (fixed 1.233 GB/s NVMe).
"""

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


def main(
    output: Annotated[
        Path,
        typer.Option("--output", "-o", help="Output PDF path for the regime map."),
    ] = Path("paper/figures/regime_map.pdf"),
) -> None:
    """Generate the single-column two-panel System C (1 A100) bottleneck regime map."""
    t_g = 158e-6   # s/genome (63.2 s / 400k warm ingest)
    r_cmp = 4.18e9 # pairs/s (1 A100 at 400k)
    n_cmp = 2.0 * t_g * r_cmp  # 1.32e6 genomes (T_cmp = T_ingest)

    c_ing = "#E6F3F8"
    c_sto = "#FCEAE6"
    c_cmp = "#EEEAF7"
    t_ing = "#1B4F72"
    t_sto = "#922B21"
    t_cmp = "#4A3F6B"

    y_min, y_max = 3.0e4, 4.5e6

    fig, (ax1, ax2) = plt.subplots(
        1, 2, sharey=True, figsize=(3.35, 2.12), constrained_layout=True
    )

    # =========================================================================
    # Panel (a): Hardware lever — BW_write vs. N at fixed b_eff = 4.0 B (dense)
    # =========================================================================
    b_dense = 4.0
    bw = np.geomspace(0.7, 50.0, 400)
    bw_cmp = (b_dense * r_cmp) / 1e9  # 16.72 GB/s
    n_wr_a = 2.0 * t_g * (bw * 1e9) / b_dense
    n_bound_a = np.minimum(n_wr_a, n_cmp)

    ax1.fill_between(bw, y_min, np.minimum(n_bound_a, y_max), color=c_ing, zorder=0)
    m_sto_a = bw <= bw_cmp
    ax1.fill_between(bw[m_sto_a], n_bound_a[m_sto_a], y_max, color=c_sto, zorder=0)
    m_cmp_a = bw >= bw_cmp
    ax1.fill_between(bw[m_cmp_a], n_bound_a[m_cmp_a], y_max, color=c_cmp, zorder=0)

    ax1.plot(bw[m_sto_a], n_bound_a[m_sto_a], color="#C0392B", lw=1.35, zorder=2)
    ax1.plot(bw[m_cmp_a], n_bound_a[m_cmp_a], color="#6B5B95", lw=1.35, zorder=2)
    ax1.plot([bw_cmp, bw_cmp], [n_cmp, y_max], color="#6B5B95", ls=":", lw=1.0, zorder=2)

    ax1.annotate(
        "",
        xy=(20.0, 4.0e5),
        xytext=(1.75, 4.0e5),
        arrowprops=dict(arrowstyle="->", color="#374151", lw=0.9, ls="--"),
        zorder=4,
    )
    ax1.scatter([1.232], [4.0e5], c="#C0392B", marker="o", s=24, edgecolors="white", lw=0.5, zorder=5)
    ax1.annotate(
        r"1$\times$ NVMe" + "\n" + r"(331\,s)",
        (1.232, 4.0e5),
        xytext=(-3, 6),
        textcoords="offset points",
        fontsize=5.8,
        color="#111827",
        ha="left",
        va="bottom",
        zorder=6,
    )
    # Measured: 400k dense to tmpfs, 8 writer streams, 20.35 GB/s standalone, 85.6 s median.
    ax1.scatter([20.35], [4.0e5], c="#1B4F72", marker="D", s=20, edgecolors="white", lw=0.5, zorder=5)
    ax1.annotate(
        r"tmpfs" + "\n" + r"(86\,s)",
        (20.35, 4.0e5),
        xytext=(0, -7),
        textcoords="offset points",
        fontsize=5.6,
        color="#1B4F72",
        ha="center",
        va="top",
        zorder=6,
    )

    ax1.text(0.85, 2.5e6, r"\textbf{Storage}", fontsize=6.4, color=t_sto, ha="left", va="center")
    ax1.text(18.5, 2.5e6, r"\textbf{Cmp.}", fontsize=6.3, color=t_cmp, ha="left", va="center")
    ax1.text(6.5, 6.2e4, r"\textbf{Ingest}", fontsize=6.4, color=t_ing, ha="center", va="center")

    ax1.set_xscale("log")
    ax1.set_yscale("log")
    ax1.set_xlim(0.7, 50.0)
    ax1.set_ylim(y_min, y_max)
    ax1.set_xticks([1, 5, 25])
    ax1.set_xticklabels(["1", "5", "25"])
    ax1.xaxis.set_minor_formatter(mpl.ticker.NullFormatter())
    ax1.set_xlabel(r"$BW_{\mathrm{write}}$ [GB/s]", fontsize=7.2)
    ax1.set_ylabel(r"Archive size $N$ [genomes]", fontsize=7.2)
    ax1.set_title(r"(a) Fixed 4\,B dense", fontsize=7.3, pad=3)
    ax1.tick_params(labelsize=6.6)
    ax1.grid(True, which="major", alpha=0.25, lw=0.5)

    # =========================================================================
    # Panel (b): Software lever — b_eff vs. N at fixed BW_write = 1.232 GB/s (standalone dd, gpu0203)
    # =========================================================================
    bw_nvme = 1.2319e9
    b_vals = np.geomspace(0.15, 6.0, 400)
    b_cmp = bw_nvme / r_cmp  # 0.295 B/pair
    n_wr_b = 2.0 * t_g * bw_nvme / b_vals
    n_bound_b = np.minimum(n_wr_b, n_cmp)

    ax2.fill_between(b_vals, y_min, np.minimum(n_bound_b, y_max), color=c_ing, zorder=0)
    m_cmp_b = b_vals <= b_cmp
    ax2.fill_between(b_vals[m_cmp_b], n_bound_b[m_cmp_b], y_max, color=c_cmp, zorder=0)
    m_sto_b = b_vals >= b_cmp
    ax2.fill_between(b_vals[m_sto_b], n_bound_b[m_sto_b], y_max, color=c_sto, zorder=0)

    ax2.plot(b_vals[m_cmp_b], n_bound_b[m_cmp_b], color="#6B5B95", lw=1.35, zorder=2)
    ax2.plot(b_vals[m_sto_b], n_bound_b[m_sto_b], color="#C0392B", lw=1.35, zorder=2)
    ax2.plot([b_cmp, b_cmp], [n_cmp, y_max], color="#6B5B95", ls=":", lw=1.0, zorder=2)

    ax2.annotate(
        "",
        xy=(1.55, 4.0e5),
        xytext=(3.40, 4.0e5),
        arrowprops=dict(arrowstyle="->", color="#374151", lw=0.9, ls="--"),
        zorder=4,
    )
    pts_b = [
        (4.00, 4.0e5, r"dense" + "\n" + r"331\,s", "#C0392B", "o", (0, 6), "center", "bottom"),
        (1.30, 4.0e5, r"sparse" + "\n" + r"156\,s", "#2A9D8F", "s", (2, 7), "center", "bottom"),
        (0.80, 4.0e5, r"$m_{\min}{=}20$" + "\n" + r"123\,s", "#2E86AB", "D", (-2, -7), "right", "top"),
    ]
    for x, y, label, col, mk, off, ha, va in pts_b:
        ax2.scatter([x], [y], c=col, marker=mk, s=24, edgecolors="white", lw=0.5, zorder=5)
        ax2.annotate(
            label,
            (x, y),
            xytext=off,
            textcoords="offset points",
            fontsize=5.7,
            color="#111827",
            ha=ha,
            va=va,
            zorder=6,
        )

    ax2.text(5.1, 2.5e6, r"\textbf{Storage}", fontsize=6.4, color=t_sto, ha="right", va="center")
    ax2.text(0.165, 2.5e6, r"\textbf{Cmp.}", fontsize=6.0, color=t_cmp, ha="left", va="center")
    ax2.text(0.55, 6.2e4, r"\textbf{Ingest}", fontsize=6.4, color=t_ing, ha="center", va="center")

    ax2.set_xscale("log")
    ax2.set_xlim(0.15, 6.0)
    ax2.set_xticks([0.2, 1.0, 4.0])
    ax2.set_xticklabels(["0.2", "1", "4"])
    ax2.xaxis.set_minor_formatter(mpl.ticker.NullFormatter())
    ax2.set_xlabel(r"$b_{\mathrm{eff}}$ [B/pair]", fontsize=7.2)
    ax2.set_title(r"(b) Fixed 1.23\,GB/s NVMe", fontsize=7.3, pad=3)
    ax2.tick_params(labelsize=6.6)
    ax2.grid(True, which="major", alpha=0.25, lw=0.5)

    for ax in (ax1, ax2):
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=300)
    fig.savefig(output.with_suffix(".png"), dpi=200)
    plt.close(fig)
    typer.echo(f"Wrote {output}")


if __name__ == "__main__":
    typer.run(main)
