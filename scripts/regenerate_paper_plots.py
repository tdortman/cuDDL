#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "matplotlib",
#   "numpy",
#   "pandas",
# ]
# ///

from pathlib import Path
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

mpl.rcParams.update({
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
})

out_dir = Path("paper/figures")
out_dir.mkdir(parents=True, exist_ok=True)

# 1. io_backend_build.pdf
fig, ax = plt.subplots(figsize=(2.3, 1.8), dpi=300)
categories = ["GH200\nwarm", "GH200\nNVMe", "A40\nwarm", "A40\nLustre"]
cpu_vals = [10.6, 26.9, 14.7, 27.2]
gpu_vals = [9.1, 27.1, 7.9, 16.0]

x = np.arange(len(categories))
width = 0.36

rects1 = ax.bar(x - width/2, cpu_vals, width, label="cpu (libdeflate)", color="#2ca02c")
rects2 = ax.bar(x + width/2, gpu_vals, width, label="gpu (nvCOMP)", color="#1f77b4")

ax.set_ylabel("Build time [s]")
ax.set_xticks(x)
ax.set_xticklabels(categories)
ax.set_ylim(0, 36)
ax.legend(frameon=False, loc="upper left", borderpad=0.2, handletextpad=0.4, labelspacing=0.2)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

for i, (c, g) in enumerate(zip(cpu_vals, gpu_vals)):
    # Stagger or place cleanly
    if abs(c - g) < 1.0:
        ax.text(x[i] - width/2, c + 0.8, f"{c:.1f}", ha="center", va="bottom", fontsize=7.5)
        ax.text(x[i] + width/2, g + 2.5, f"{g:.1f}", ha="center", va="bottom", fontsize=7.5)
    else:
        ax.text(x[i] - width/2, c + 0.8, f"{c:.1f}", ha="center", va="bottom", fontsize=7.5)
        ax.text(x[i] + width/2, g + 0.8, f"{g:.1f}", ha="center", va="bottom", fontsize=7.5)

plt.tight_layout(pad=0.3)
fig.savefig(out_dir / "io_backend_build.pdf")
fig.savefig(out_dir / "io_backend_build.png")
plt.close(fig)

# 2. b300_build_115k.pdf
fig, ax = plt.subplots(figsize=(2.3, 1.8), dpi=300)
categories = ["Standard\ngzip", "BGZF\n(bgzip)"]
x = np.arange(len(categories))
w = 0.18

cpu = [47.1, 54.0]
hybrid = [45.9, 51.8]
gpu_cores = [19.9, 19.7]
gpu_de = [20.5, 14.4]

ax.bar(x - 1.5*w, cpu, w, label="cpu", color="#2ca02c")
ax.bar(x - 0.5*w, hybrid, w, label="hybrid", color="#9467bd")
ax.bar(x + 0.5*w, gpu_cores, w, label="gpu-cores", color="#ff7f0e")
ax.bar(x + 1.5*w, gpu_de, w, label="gpu (DE)", color="#1f77b4")

ax.set_ylabel("Build time [s] (115k genomes)")
ax.set_xticks(x)
ax.set_xticklabels(categories)
ax.set_ylim(0, 85)
ax.legend(ncol=2, frameon=False, loc="upper left", borderpad=0.2, handletextpad=0.4, labelspacing=0.2, columnspacing=0.6)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

for i in range(len(categories)):
    ax.text(x[i] - 1.5*w, cpu[i] + 1.2, f"{cpu[i]:.1f}", ha="center", va="bottom", fontsize=7)
    ax.text(x[i] - 0.5*w, hybrid[i] + 1.2, f"{hybrid[i]:.1f}", ha="center", va="bottom", fontsize=7)
    # Stagger gpu-cores and gpu_de if close
    if abs(gpu_cores[i] - gpu_de[i]) < 1.5:
        ax.text(x[i] + 0.5*w, gpu_cores[i] + 1.2, f"{gpu_cores[i]:.1f}", ha="center", va="bottom", fontsize=7)
        ax.text(x[i] + 1.5*w, gpu_de[i] + 3.2, f"{gpu_de[i]:.1f}", ha="center", va="bottom", fontsize=7)
    else:
        ax.text(x[i] + 0.5*w, gpu_cores[i] + 1.2, f"{gpu_cores[i]:.1f}", ha="center", va="bottom", fontsize=7)
        ax.text(x[i] + 1.5*w, gpu_de[i] + 1.2, f"{gpu_de[i]:.1f}", ha="center", va="bottom", fontsize=7)

plt.tight_layout(pad=0.3)
fig.savefig(out_dir / "b300_build_115k.pdf")
fig.savefig(out_dir / "b300_build_115k.png")
plt.close(fig)

print("Regenerated clean paper subfigures successfully.")
