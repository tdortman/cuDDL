#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = ["jsonschema", "matplotlib", "pandas", "typer"]
# ///
"""Speed-of-light figures from scripts/run_speed_of_light.py.

The x axis is the share of the device a run had: SMs out of the GPU's for GPU tools, physical
cores out of the CPU's for CPU tools (a run with more threads than cores counts as all cores,
and a tool keeps its faster run there). Per timing scope, process wall and resident:

- sol_speedup_<scope>: SKETCH and COMPARE throughput relative to the tool's full-device run,
  against the diagonal of throughput proportional to compute;
- sol_absolute_<scope>: the same runs as genomes/s and pairs/s.
"""

import json
from pathlib import Path
from typing import Annotated

import matplotlib.pyplot as plt
import plot_utils as pu
import typer
from benchmark_schema import load_result

app = typer.Typer(add_completion=False)
OPERATIONS = (("sketch", "SKETCH", "genomes/s"), ("compare", "COMPARE", "pairs/s"))


def rate(measurement: dict, scope: str) -> float | None:
    """Genomes or pairs per second of @p measurement in @p scope, None when unmeasured."""
    timing = measurement["timings"].get(scope)
    if not timing or not timing.get("median_ms"):
        return None
    case = measurement["case"]
    count = case["genomes"] if case["measurement"] == "micro-sketch" else case["pairs"]
    return count / (timing["median_ms"] / 1000)


@app.command()
def main(
    results: Annotated[Path, typer.Argument(exists=True, file_okay=False, help="Sweep directory")],
    output_dir: Annotated[Path | None, typer.Option(help="Defaults to RESULTS/plots")] = None,
) -> None:
    output_dir = output_dir or results / "plots"
    output_dir.mkdir(parents=True, exist_ok=True)
    # (tool, kind, operation, scope) -> {share: best rate}
    curves: dict[tuple, dict[float, float]] = {}
    for line in (results / "points.jsonl").read_text().splitlines():
        point = json.loads(line)
        units = point["sms"] if point["kind"] == "gpu" else point["cores"]
        share = units / point["total"]
        for measurement in load_result(results / point["report"], "micro")["measurements"]:
            operation = measurement["case"]["measurement"].removeprefix("micro-")
            if operation not in {o for o, _, _ in OPERATIONS}:
                continue
            for scope in ("wall", "resident"):
                if value := rate(measurement, scope):
                    curve = curves.setdefault((point["tool"], point["kind"], operation, scope), {})
                    curve[share] = max(value, curve.get(share, 0.0))

    tools = sorted({(tool, kind) for tool, kind, _, _ in curves})
    colors = {tool: plt.cm.tab10(i % 10) for i, (tool, _) in enumerate(tools)}
    for scope in ("wall", "resident"):
        for stem, relative in (("sol_speedup", True), ("sol_absolute", False)):
            fig, axes = plt.subplots(1, 2, figsize=(12, 5))
            for ax, (operation, title, unit) in zip(axes, OPERATIONS):
                for tool, kind in tools:
                    curve = curves.get((tool, kind, operation, scope))
                    if not curve or len(curve) < 2:
                        continue
                    shares = sorted(curve)
                    values = [curve[s] for s in shares]
                    if relative:
                        values = [v / curve[shares[-1]] for v in values]
                    ax.plot(shares, values, marker="o", color=colors[tool],
                            ls="-" if kind == "gpu" else "--", label=f"{tool} ({kind.upper()})",
                            lw=pu.LINE_WIDTH, ms=pu.MARKER_SIZE)
                if relative:
                    ax.plot([0, 1], [0, 1], color="grey", ls=":", lw=pu.REFERENCE_LINE_WIDTH,
                            label="proportional to compute")
                    ax.set_ylim(0, 1.15)
                    ax.set_ylabel("Throughput relative to the full device")
                else:
                    ax.set_yscale("log")
                    ax.set_ylabel(f"Throughput [{unit}]")
                ax.set_xlim(0, 1.02)
                ax.set_xlabel("Share of the device's SMs or cores")
                ax.set_title(f"{title} ({scope})", fontsize=pu.TITLE_FONT_SIZE)
                ax.grid(True, ls="--", alpha=pu.GRID_ALPHA)
            # Every tool appears in the legend once, whichever panel holds it.
            labels = {}
            for ax in axes:
                for handle, label in zip(*ax.get_legend_handles_labels()):
                    labels.setdefault(label, handle)
            if not labels:
                plt.close(fig)
                continue
            fig.legend(labels.values(), labels.keys(), loc="upper center", ncol=4,
                       fontsize=pu.LEGEND_FONT_SIZE * 0.7, framealpha=pu.LEGEND_FRAME_ALPHA,
                       bbox_to_anchor=(0.5, 1.12))
            fig.tight_layout()
            fig.savefig(output_dir / f"{stem}_{scope}.png", dpi=200, bbox_inches="tight")
            pu.save_figure(fig, output_dir / f"{stem}_{scope}.pdf")


if __name__ == "__main__":
    app()
