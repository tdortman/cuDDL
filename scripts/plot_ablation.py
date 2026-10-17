#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = ["matplotlib", "pandas", "tabulate", "typer"]
# ///
"""Ranks optimisations by what turning each off costs, from the JSON lines of ablation.py.

For every optimisation and machine, the slowdown without it is the median time of the
configuration without it over the median time of the one it is compared with, measured on the
stage it targets and end to end (the `e2e` suite's wall time). The stage is the first of: the
`micro` suite (a host stage alone), the `build` suite (the build or its sketch kernels), the
`search` suite (the search on resident sketches), and the `e2e` suite's time in the phase the
optimisation targets (metrics `phase:NAME`).

Draws a figure one ACM column wide, set in Linux Libertine like acmart: a stage panel and an
end-to-end panel share one row per optimisation, ranked by the largest end-to-end slowdown,
with one marker per machine, labelled with its slowdown, and lines spanning the slowdowns of
repetitions paired by index. Slowdowns past an axis's cap sit at the cap as an arrow. Prints
every slowdown as a Markdown table, largest first.
"""

import json
from pathlib import Path
from typing import Annotated

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plot_utils as pu
import typer

SCOPES = {"micro": "stage", "build": "stage", "search": "stage", "e2e": "end to end"}
SUITE_ORDER = {"micro": 0, "build": 1, "search": 2, "e2e": 3}
COLUMN_INCHES = 3.33  # an ACM two-column layout's column width
MARKERS = "osD^v"

# acmart sets text in Linux Libertine. Computer Modern at these sizes switches to its wide
# small-size designs, so labels would run wider than the paper's text.
mpl.rcParams["text.latex.preamble"] = r"\usepackage{libertine}\usepackage[libertine]{newtxmath}"


def load(paths: list[Path]) -> tuple[pd.DataFrame, pd.DataFrame]:
    runs, ablations = [], []
    for path in paths:
        for line in path.read_text().splitlines():
            entry = json.loads(line)
            if "ablations" in entry:
                for a in entry["ablations"]:
                    ablations.append({"suite": entry["suite"], "label": entry["label"], **a})
            else:
                phases = entry.pop("phases_s", None) or {}
                runs.append(entry | {f"phase:{k}": v for k, v in phases.items()})
    return pd.DataFrame(runs), pd.DataFrame(ablations).drop_duplicates()


def slowdowns(runs: pd.DataFrame, ablations: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for a in ablations.itertuples():
        mine = runs[(runs.suite == a.suite) & (runs.label == a.label)]
        if a.metric not in mine:
            continue
        without = mine[mine.config == a.without].set_index("rep")[a.metric].dropna()
        reference = mine[mine.config == a.reference].set_index("rep")[a.metric].dropna()
        if without.empty or reference.empty:
            continue
        paired = (without / reference).dropna()
        rows.append(
            {
                "label": a.label,
                "scope": "stage" if a.metric.startswith("phase:") else SCOPES[a.suite],
                "suite": a.suite,
                "metric": a.metric,
                "optimisation": a.optimisation,
                "with_s": reference.median(),
                "without_s": without.median(),
                "slowdown": without.median() / reference.median(),
                "low": paired.min() if not paired.empty else np.nan,
                "high": paired.max() if not paired.empty else np.nan,
                "reps": len(paired),
            }
        )
    table = pd.DataFrame(rows)
    table["order"] = table.suite.map(SUITE_ORDER)
    return (
        table.sort_values("order", kind="stable")
        .drop_duplicates(["label", "scope", "optimisation"])
        .drop(columns="order")
    )


def times(value: float) -> str:
    digits = 2 if value < 2 else 1 if value < 10 else 0
    return f"{value:.{digits}f}" + r"$\times$"


def main(
    results: Annotated[list[Path], typer.Argument(exists=True, dir_okay=False)],
    output: Annotated[
        Path, typer.Option(help="PDF to write; a PNG goes beside it")
    ] = Path("results/ablation.pdf"),
    minimum: Annotated[
        float,
        typer.Option(
            help="Leave out optimisations, and machines, within this factor of 1 everywhere"
        ),
    ] = 1.15,
    top_n: Annotated[
        int,
        typer.Option("--top", help="Keep at most this many top optimisations by slowdown"),
    ] = 9,
    stage_cap: Annotated[
        float, typer.Option(help="Largest slowdown the stage axis shows")
    ] = 12.0,
    names: Annotated[
        list[str],
        typer.Option("--name", help="Legend label for a machine label, as LABEL=NAME"),
    ] = [],
) -> None:
    runs, ablations = load(results)
    table = slowdowns(runs, ablations)
    if table.empty:
        raise typer.BadParameter("no ablation has both of its configurations measured")
    print(table.sort_values("slowdown", ascending=False).to_markdown(index=False, floatfmt=".3f"))

    # Turning an optimisation off can also be faster, so the effect counts both ways.
    table = table.assign(effect=np.abs(np.log(table.slowdown)))
    noteworthy = table[table.effect >= np.log(minimum)]
    table = table[
        table.optimisation.isin(noteworthy.optimisation) & table.label.isin(noteworthy.label)
    ]
    end_to_end = table[table.scope == "end to end"].groupby("optimisation").slowdown.max()
    stage = table[table.scope == "stage"].groupby("optimisation").slowdown.max()
    # Largest end-to-end slowdown on top; stage-only optimisations follow, by their stage slowdown.
    order = sorted(
        set(table.optimisation), key=lambda o: (end_to_end.get(o, 0), stage.get(o, 0))
    )
    if top_n > 0 and len(order) > top_n:
        order = order[-top_n:]
        table = table[table.optimisation.isin(order)]
    row = {name: i for i, name in enumerate(order)}
    labels = list(dict.fromkeys(table.label))
    legend = dict(n.split("=", 1) for n in names)

    row_inches = 0.09 * max(1, len(labels))
    fig, axes = plt.subplots(
        1,
        2,
        sharey=True,
        figsize=(COLUMN_INCHES, 0.45 + row_inches * len(order)),
        layout="constrained",
        gridspec_kw={"wspace": 0.03},
    )
    spread = 0.46 if len(labels) > 1 else 0.0
    offsets = np.linspace(-spread / 2, spread / 2, len(labels))
    for ax, scope in zip(axes, ("stage", "end to end")):
        mine = table[table.scope == scope]
        values = mine[["low", "slowdown", "high"]].to_numpy()
        top = np.nanmax(values) if values.size else 2.0
        cap = min(top, stage_cap) if scope == "stage" else top
        right = cap * 1.22
        for i, label in enumerate(labels):
            points = mine[mine.label == label]
            y = points.optimisation.map(row).to_numpy() + offsets[i]
            clipped = points.slowdown > cap
            x = points.slowdown.clip(upper=cap)
            ax.hlines(y, points.low.clip(upper=cap), points.high.clip(upper=cap), color=f"C{i}", lw=0.7)
            ax.plot(
                x[~clipped],
                y[~clipped],
                MARKERS[i % len(MARKERS)],
                color=f"C{i}",
                ms=2.6,
                label=pu.paper_text(legend.get(label, label)),
            )
            ax.plot(x[clipped], y[clipped], ">", color=f"C{i}", ms=3)
            ends = np.fmax(points.high.clip(upper=cap), x)
            for value, xi, yi, past in zip(points.slowdown, ends, y, clipped):
                ax.annotate(
                    times(value),
                    (xi, yi),
                    xytext=(-3, 0) if past else (3, 0),
                    textcoords="offset points",
                    ha="right" if past else "left",
                    va="center",
                    fontsize=4.8,
                    color=f"C{i}",
                )
        ax.axvline(1, color="black", lw=0.5)
        left = min(0.8, np.nanmin(values) * 0.95) if values.size else 0.8
        ax.set_xlim(left, right)
        ticks = mpl.ticker.MaxNLocator(nbins=4, steps=[1, 2, 2.5, 5, 10]).tick_values(left, right)
        ax.set_xticks(sorted({1.0, *(t for t in ticks if left <= t <= right and t >= 1.5)}))
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:g}" + r"$\times$"))
        ax.tick_params(labelsize=6, length=2, pad=1)
        ax.grid(True, axis="x", lw=0.3, alpha=0.5)
        ax.set_title(scope, fontsize=7, pad=2)
    axes[0].set_yticks(range(len(order)), [pu.paper_text(o) for o in order], fontsize=6)
    axes[0].tick_params(axis="y", length=0)
    axes[0].set_ylim(-0.6, len(order) - 0.4)
    if len(labels) > 1:
        fig.legend(
            *axes[0].get_legend_handles_labels(),
            loc="outside upper center",
            ncol=len(labels),
            fontsize=6,
            frameon=False,
            handletextpad=0.2,
            columnspacing=0.8,
        )
    fig.canvas.draw()
    pos0 = axes[0].get_position()
    pos1 = axes[1].get_position()
    box_center = (pos0.x0 + pos1.x1) / 2
    fig.supxlabel("slowdown without the optimisation", x=box_center, fontsize=7)
    fig.savefig(output.with_suffix(".png"), dpi=300, bbox_inches="tight")
    pu.save_figure(fig, output)


if __name__ == "__main__":
    typer.run(main)
