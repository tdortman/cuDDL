#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = ["matplotlib", "pandas", "tabulate", "typer"]
# ///
"""Ranks optimisations by the speedup each brings, from the JSON lines of ablation.py.

For every optimisation and machine, its speedup is the median time of the configuration without
it over the median time of the one it is compared with, measured on the stage it targets and end
to end (the `e2e` suite's wall time). The stage is the first of: the `micro` suite (a host stage
alone), the `build` suite (the build or its sketch kernels), the `search` suite (the search on
resident sketches), and the `e2e` suite's time in the phase the optimisation targets (metrics
`phase:NAME`).

Draws a figure one ACM column wide, set in Linux Libertine like acmart, to be included at 100%
scale: a stage panel and an end-to-end panel share one row per optimisation, ranked by the
largest end-to-end speedup, with one marker per machine, labelled with its speedup, and lines
spanning the speedups of repetitions paired by index. A row's markers are centred on it, one
lane per machine measured in that panel. Speedups past an axis's cap sit at the cap as an
arrow. Prints every speedup as a Markdown table, largest first.
"""

import json
from pathlib import Path
from typing import Annotated

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker
import numpy as np
import pandas as pd
import plot_utils as pu
import typer

SCOPES = {"micro": "stage", "build": "stage", "search": "stage", "e2e": "end to end"}
SUITE_ORDER = {"micro": 0, "build": 1, "search": 2, "e2e": 3}
COLUMN_INCHES = 3.33  # an ACM two-column layout's column width
ROW_INCHES = 0.16
LANE_INCHES = 0.065  # two value labels this far apart do not touch
FONT = 6.5
VALUE_FONT = 5.2
MARKERS = "osD^v"

# acmart sets text in Linux Libertine. Computer Modern at these sizes switches to its wide
# small-size designs, so labels would run wider than the paper's text.
mpl.rcParams["text.latex.preamble"] = (
    r"\usepackage{libertine}\usepackage[libertine]{newtxmath}"
)


def load(paths: list[Path]) -> tuple[pd.DataFrame, pd.DataFrame]:
    runs, ablations = [], []
    for path in paths:
        for line in path.read_text().splitlines():
            entry = json.loads(line)
            if "ablations" in entry:
                for a in entry["ablations"]:
                    ablations.append(
                        {"suite": entry["suite"], "label": entry["label"], **a}
                    )
            else:
                phases = entry.pop("phases_s", None) or {}
                runs.append(entry | {f"phase:{k}": v for k, v in phases.items()})
    return pd.DataFrame(runs), pd.DataFrame(ablations).drop_duplicates()


def speedups(runs: pd.DataFrame, ablations: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for a in ablations.to_dict("records"):
        mine = runs[(runs.suite == a["suite"]) & (runs.label == a["label"])]
        if a["metric"] not in mine:
            continue
        without = (
            mine[mine.config == a["without"]].set_index("rep")[a["metric"]].dropna()
        )
        reference = (
            mine[mine.config == a["reference"]].set_index("rep")[a["metric"]].dropna()
        )
        if without.empty or reference.empty:
            continue
        paired = (without / reference).dropna()
        rows.append(
            {
                "label": a["label"],
                "scope": "stage"
                if a["metric"].startswith("phase:")
                else SCOPES[a["suite"]],
                "suite": a["suite"],
                "metric": a["metric"],
                "optimisation": a["optimisation"],
                "with_s": reference.median(),
                "without_s": without.median(),
                "speedup": without.median() / reference.median(),
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
        typer.Option(
            "--top", help="Keep at most this many top optimisations by speedup"
        ),
    ] = 9,
    stage_cap: Annotated[
        float, typer.Option(help="Largest speedup the stage axis shows")
    ] = 12.0,
    names: Annotated[
        list[str] | None,
        typer.Option("--name", help="Legend label for a machine label, as LABEL=NAME"),
    ] = None,
) -> None:
    runs, ablations = load(results)
    table = speedups(runs, ablations)
    if table.empty:
        raise typer.BadParameter("no ablation has both of its configurations measured")
    print(
        table.sort_values("speedup", ascending=False).to_markdown(
            index=False, floatfmt=".3f"
        )
    )

    # An optimisation can also make a run slower, so the effect counts both ways.
    table = table.assign(effect=np.abs(np.log(table.speedup)))
    noteworthy = table[table.effect >= np.log(minimum)]
    table = table[
        table.optimisation.isin(noteworthy.optimisation)
        & table.label.isin(noteworthy.label)
    ]
    end_to_end = (
        table[table.scope == "end to end"].groupby("optimisation").speedup.max()
    )
    stage = table[table.scope == "stage"].groupby("optimisation").speedup.max()
    # Largest end-to-end speedup on top; stage-only optimisations follow, by their stage speedup.
    order = sorted(
        set(table.optimisation), key=lambda o: (end_to_end.get(o, 0), stage.get(o, 0))
    )
    if top_n > 0 and len(order) > top_n:
        order = order[-top_n:]
        table = table[table.optimisation.isin(order)]
    row = {name: i for i, name in enumerate(order)}
    machines = {label: i for i, label in enumerate(dict.fromkeys(table.label))}
    legend = dict(n.split("=", 1) for n in names or [])

    rows_inches = ROW_INCHES * len(order)
    fig, axes = plt.subplots(
        1,
        2,
        sharey=True,
        figsize=(COLUMN_INCHES, 0.42 + rows_inches),
        layout="constrained",
        gridspec_kw={"wspace": 0.04},
    )
    lane = LANE_INCHES / ROW_INCHES
    for ax, scope, title in zip(
        axes, ("stage", "end to end"), ("stage speedup", "end-to-end speedup")
    ):
        mine = table[table.scope == scope].copy()
        # Lanes from the top: the first machine sits highest in a row.
        rank = mine.label.map(machines)
        index = rank.groupby(mine.optimisation).rank(method="first") - 1
        count = mine.groupby("optimisation").label.transform("size")
        mine["y"] = mine.optimisation.map(row) + ((count - 1) / 2 - index) * lane
        values = mine[["low", "speedup", "high"]].to_numpy()
        top = np.nanmax(values) if values.size else 2.0
        cap = min(top, stage_cap) if scope == "stage" else top
        right = cap * 1.25
        for label, i in machines.items():
            points = mine[mine.label == label]
            if points.empty:
                continue
            y = points.y.to_numpy()
            clipped = points.speedup > cap
            x = points.speedup.clip(upper=cap)
            ax.hlines(
                y,
                points.low.clip(upper=cap),
                points.high.clip(upper=cap),
                color=f"C{i}",
                lw=0.7,
            )
            ax.plot(
                x[~clipped],
                y[~clipped],
                MARKERS[i % len(MARKERS)],
                color=f"C{i}",
                ms=2.8,
            )
            ax.plot(x[clipped], y[clipped], ">", color=f"C{i}", ms=3)
            ends = np.fmax(points.high.clip(upper=cap), x)
            for value, xi, yi, past in zip(points.speedup, ends, y, clipped):
                ax.annotate(
                    times(value),
                    (xi, yi),
                    xytext=(-3, 0) if past else (2.5, 0),
                    textcoords="offset points",
                    ha="right" if past else "left",
                    va="center_baseline",
                    fontsize=VALUE_FONT,
                    color=f"C{i}",
                )
        ax.axvline(1, color="black", lw=0.5)
        left = min(0.8, np.nanmin(values) * 0.95) if values.size else 0.8
        ax.set_xlim(left, right)
        ticks = mpl.ticker.MaxNLocator(nbins=4, steps=[1, 2, 2.5, 5, 10]).tick_values(
            left, right
        )
        # A tick within a fifth of the axis of 1x would print over it.
        gap = 0.2 * (right - left)
        ax.set_xticks(sorted({1.0, *(t for t in ticks if 1 + gap <= t <= right)}))
        ax.xaxis.set_major_formatter(
            plt.FuncFormatter(lambda v, _: f"{v:g}" + r"$\times$")
        )
        ax.tick_params(labelsize=FONT, length=2, pad=1)
        ax.grid(True, axis="x", lw=0.3, alpha=0.5)
        ax.set_xlabel(title, fontsize=FONT + 0.5, labelpad=1)
    axes[0].set_yticks(
        range(len(order)), [pu.paper_text(o) for o in order], fontsize=FONT
    )
    axes[0].tick_params(axis="y", length=0)
    axes[0].set_ylim(-0.55, len(order) - 0.45)
    if len(machines) > 1:
        handles = [
            mpl.lines.Line2D(
                [], [], ls="", marker=MARKERS[i % len(MARKERS)], color=f"C{i}", ms=3
            )
            for i in machines.values()
        ]
        fig.legend(
            handles,
            [pu.paper_text(legend.get(label, label)) for label in machines],
            loc="outside upper center",
            ncol=len(machines),
            fontsize=FONT,
            frameon=False,
            handletextpad=0.1,
            columnspacing=0.9,
            borderpad=0,
        )
    fig.savefig(output.with_suffix(".png"), dpi=300, bbox_inches="tight")
    pu.save_figure(fig, output)


if __name__ == "__main__":
    typer.run(main)
