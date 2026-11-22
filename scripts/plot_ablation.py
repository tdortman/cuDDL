#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = ["matplotlib", "pandas", "tabulate", "typer"]
# ///
"""Ranks optimisations by the speedup each brings, from the JSON lines of ablation.py.

For every optimisation and machine, divide the time without it by the reference time within
each repetition, then take the median of the paired ratios. This applies to the target stage
and end to end (the `e2e` suite's wall time). The stage is the first of: the `micro` suite (a host stage
alone), the `build` suite (the build or its sketch kernels), the `search` suite (the search on
resident sketches), and the `e2e` suite's time in the phase the optimisation targets (metrics
`phase:NAME`).

Draws a figure one ACM column wide, set in Linux Libertine like acmart, to be included at 100%
scale: a stage panel and an end-to-end panel share one row per optimisation, ranked by the
largest end-to-end speedup. Markers show median paired speedups. Each machine has one lane per row. The stage panel
uses a log axis to show the full range. Prints every speedup as a Markdown table, largest first.
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
ROW_INCHES = 0.185
LANE_INCHES = 0.075  # two value labels this far apart do not touch
FONT = 7.2
VALUE_FONT = 6.0
MARKERS = "osD^v"
THRESHOLDED_OUTPUT = "Thresholded output"
INDEX_VS_FILTER = "Index vs. filter"

SHORT_LABELS = {
    "Similarity records on the GPU": "4 B WKID vs. 24 B",
    "Thresholded output": r"Thresholded ($m_{\min}{=}20$)",
    "Index vs. filter": "Index vs. filter",
    "Branchless k-mer encoding (gpu)": "SWAR sketch (gpu)",
    "Branchless k-mer encoding (cpu)": "SWAR sketch (cpu)",
    "Two-bit base packing": "2-bit base pack",
    "SIMD FASTX parsing": "SIMD parsing",
    "MPI progress thread": "MPI progress",
    "LOP3 plane kernel": "LOP3 vs. scalar",
    "Bit-sliced scores": "Bit-sliced vs. 16-bit",
    "GPU decompression (nvCOMP)": "GPU nvCOMP",
    "DNA-tuned libdeflate": "DNA libdeflate",
}

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


def target_stage(suite: str, metric: str, reference: str, optimisation: str) -> str:
    if metric == "phase:write" or any(
        word in optimisation for word in ("pools", "streams", "writes", "downloads")
    ):
        return "Output"
    if metric == "phase:search" or reference.startswith("threshold"):
        return "Filter"
    if suite == "search":
        return "Compare"
    if (
        metric in ("kernel_seconds", "pack_seconds")
        or "packing" in optimisation
        or "parsing" in optimisation
    ):
        return "Sketch"
    if (
        metric == "inflate_seconds"
        or "decompression" in optimisation
        or "libdeflate" in optimisation
        or "lanes" in optimisation
        or "co-scheduling" in optimisation
    ):
        return "Inflate"
    return "I/O"


def speedups(runs: pd.DataFrame, ablations: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for a in ablations.to_dict("records"):
        mine = runs[(runs.suite == a["suite"]) & (runs.label == a["label"])]
        if a["metric"] not in mine:
            continue
        without = mine.loc[mine.config == a["without"], ["rep", a["metric"]]]
        reference = mine.loc[mine.config == a["reference"], ["rep", a["metric"]]]
        pairs = without.merge(
            reference, on="rep", validate="one_to_one", suffixes=("_without", "_ref")
        ).dropna()
        if pairs.empty:
            continue
        without = pairs[f'{a["metric"]}_without']
        reference = pairs[f'{a["metric"]}_ref']
        paired = without / reference
        optimisation = a["optimisation"]
        if optimisation == "Index filtering on the GPU":
            optimisation = THRESHOLDED_OUTPUT
        elif optimisation == "Inverted index":
            optimisation = INDEX_VS_FILTER
        rows.append(
            {
                "label": a["label"],
                "scope": "stage"
                if a["metric"].startswith("phase:")
                else SCOPES[a["suite"]],
                "suite": a["suite"],
                "metric": a["metric"],
                "optimisation": optimisation,
                "stage": target_stage(
                    a["suite"], a["metric"], a["reference"], a["optimisation"]
                ),
                "with_s": reference.median(),
                "without_s": without.median(),
                "speedup": paired.median(),
                "low": paired.min(),
                "high": paired.max(),
                "reps": len(paired),
            }
        )
    table = pd.DataFrame(rows)
    if table.empty:
        return table
    table["order"] = table.suite.map(SUITE_ORDER)
    return (
        table.sort_values("order", kind="stable")
        .drop_duplicates(["label", "scope", "optimisation"])
        .drop(columns="order")
    )


def times(value: float) -> str:
    digits = 2 if value < 2 else 1
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
    ] = 1.25,
    top_n: Annotated[
        int,
        typer.Option(
            "--top", help="Keep at most this many top optimisations by speedup"
        ),
    ] = 0,
    names: Annotated[
        list[str] | None,
        typer.Option("--name", help="Legend label for a machine label, as LABEL=NAME"),
    ] = None,
    exclude: Annotated[
        list[str] | None,
        typer.Option("--exclude", help="Optimisation names to omit from the figure"),
    ] = None,
) -> None:
    if exclude is None:
        exclude = ["Device-only pinned pools"]
    runs, ablations = load(results)
    table = speedups(runs, ablations)
    if table.empty:
        raise typer.BadParameter("no ablation has both of its configurations measured")
    print(
        table.sort_values("speedup", ascending=False).to_markdown(
            index=False, floatfmt=".3f"
        )
    )

    legend = dict(n.split("=", 1) for n in names or [])
    if legend:
        table = table[table.label.isin(legend)]
    if exclude:
        table = table[~table.optimisation.isin(exclude)]

    # An optimisation can also make a run slower, so the effect counts both ways.
    table = table.assign(effect=np.abs(np.log(table.speedup)))
    e2e_labels = set(table[table.scope == "end to end"].label)
    e2e_noteworthy = set(
        table[
            (table.scope == "end to end") & (table.effect >= np.log(1.15))
        ].optimisation
    )
    stage_rows = table[table.scope == "stage"].groupby("optimisation")
    stage_all = set(
        stage_rows.filter(
            lambda g: (
                len(set(g.label) & e2e_labels) == len(e2e_labels)
                and g.effect.min() >= np.log(minimum)
            )
        ).optimisation
    )
    noteworthy = table[table.optimisation.isin(e2e_noteworthy | stage_all)]
    if THRESHOLDED_OUTPUT in set(noteworthy.optimisation) and INDEX_VS_FILTER in stage_all:
        noteworthy = noteworthy[
            noteworthy.optimisation != "Low-plane match filter"
        ]
    paired_opts = (
        set(table[table.scope == "stage"].optimisation)
        & set(table[table.scope == "end to end"].optimisation)
        if e2e_labels
        else set(table.optimisation)
    )
    table = table[
        table.optimisation.isin(set(noteworthy.optimisation) & paired_opts)
        & table.label.isin(table[table.effect >= np.log(minimum)].label)
    ]
    end_to_end = (
        table[table.scope == "end to end"].groupby("optimisation").speedup.max()
    )
    stage = table[table.scope == "stage"].groupby("optimisation").speedup.max()
    # Select top_n optimisations by (end-to-end, stage) speedup, then group by pipeline stage.
    selected = sorted(
        set(table.optimisation), key=lambda o: (end_to_end.get(o, 0), stage.get(o, 0))
    )
    if top_n > 0 and len(selected) > top_n:
        selected = selected[-top_n:]
        table = table[table.optimisation.isin(selected)]
    stage_of = (
        table[table.scope == "stage"]
        .drop_duplicates("optimisation")
        .set_index("optimisation")["stage"]
        .to_dict()
    )
    for o in selected:
        stage_of.setdefault(o, table[table.optimisation == o]["stage"].iloc[0])
    stage_rank = {
        st: max(
            (end_to_end.get(o, 0.0), stage.get(o, 0.0))
            for o in selected
            if stage_of[o] == st
        )
        for st in set(stage_of.values())
    }
    order = sorted(
        selected,
        key=lambda o: (
            stage_rank[stage_of[o]],
            end_to_end.get(o, 0.0) if end_to_end.get(o, 0.0) >= 1.15 else 0.0,
            stage.get(o, 0.0),
        ),
    )
    row = {name: i for i, name in enumerate(order)}
    machines = {label: i for i, label in enumerate(dict.fromkeys(table.label))}

    rows_inches = ROW_INCHES * len(order)
    fig, axes = plt.subplots(
        1,
        2,
        sharey=True,
        figsize=(COLUMN_INCHES, 0.42 + rows_inches),
        layout="constrained",
        gridspec_kw={"wspace": 0.08, "width_ratios": [1.12, 0.88]},
    )
    lane = LANE_INCHES / ROW_INCHES
    for ax, scope, title in zip(
        axes, ("stage", "end to end"), ("Stage (log)", "End-to-end")
    ):
        mine = table[table.scope == scope].copy()
        # Lanes from the top: the first machine sits highest in a row.
        rank = mine.label.map(machines)
        index = rank.groupby(mine.optimisation).rank(method="first") - 1
        count = mine.groupby("optimisation").label.transform("size")
        mine["y"] = mine.optimisation.map(row) + ((count - 1) / 2 - index) * lane
        values = mine[["low", "speedup", "high"]].to_numpy()
        top = np.nanmax(values) if values.size else 2.0
        right = top * (1.85 if scope == "stage" else 1.48)
        if scope == "stage":
            ax.set_xscale("log")
        for label, i in machines.items():
            points = mine[mine.label == label]
            if points.empty:
                continue
            y = points.y.to_numpy()
            xs = points.speedup.to_numpy()
            ax.hlines(y, 1.0, xs, color=f"C{i}", lw=1.25, alpha=0.85, zorder=3)
            ax.plot(
                xs,
                y,
                MARKERS[i % len(MARKERS)],
                color=f"C{i}",
                ms=3.1,
                zorder=4,
            )
            for value, xi, yi in zip(xs, xs, y):
                anchor_x = max(xi, 1.0) if scope == "stage" and xi < 1.0 else xi
                ax.annotate(
                    times(value),
                    (anchor_x, yi),
                    xytext=(2.6, 0),
                    textcoords="offset points",
                    ha="left",
                    va="center_baseline",
                    fontsize=VALUE_FONT,
                    color=f"C{i}",
                )
        ax.axvline(1, color="black", lw=0.6, zorder=2)
        left = min(0.75, np.nanmin(values) * 0.90) if values.size else 0.75
        ax.set_xlim(left, right)
        if scope == "stage":
            candidates = (1, 2, 5, 10) if top <= 12 else (1, 5, 25, 125)
            ax.set_xticks([t for t in candidates if t <= right * 0.92])
            ax.xaxis.set_minor_locator(mpl.ticker.NullLocator())
        else:
            ax.set_xticks([t for t in (1, 2, 4, 6) if t <= right * 0.95])
        ax.xaxis.set_major_formatter(
            plt.FuncFormatter(lambda v, _: f"{v:g}" + r"$\times$")
        )
        ax.tick_params(labelsize=FONT, length=2, pad=1.5)
        ax.grid(True, axis="x", lw=0.35, alpha=0.45, zorder=1)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.set_xlabel(
            title,
            fontsize=FONT + 0.3,
            labelpad=1.5,
        )
    axes[0].set_yticks(
        range(len(order)),
        [SHORT_LABELS.get(o, pu.paper_text(o)) for o in order],
        fontsize=FONT,
    )
    axes[0].tick_params(axis="y", length=0, pad=2.5)
    axes[1].tick_params(axis="y", length=0)
    axes[0].set_ylim(-0.55, len(order) - 0.45)
    spans: list[tuple[int, int, str]] = []
    start = 0
    for idx in range(len(order)):
        if idx == len(order) - 1 or stage_of[order[idx]] != stage_of[order[idx + 1]]:
            spans.append((start, idx, stage_of[order[start]]))
            if idx < len(order) - 1:
                for ax in axes:
                    ax.axhline(idx + 0.5, color="#9ca3af", lw=0.55, ls="--", zorder=2)
            start = idx + 1
    for b_idx, (s_i, e_i, _) in enumerate(spans):
        if b_idx % 2 == 1:
            for ax in axes:
                ax.axhspan(s_i - 0.5, e_i + 0.5, color="#f3f4f6", zorder=0, lw=0)
    ax_r = axes[1].twinx()
    ax_r.spines["top"].set_visible(False)
    ax_r.spines["right"].set_visible(False)
    ax_r.spines["left"].set_visible(False)
    ax_r.spines["bottom"].set_visible(False)
    ax_r.set_ylim(-0.55, len(order) - 0.45)
    ax_r.set_yticks(
        [(s_i + e_i) / 2 for s_i, e_i, _ in spans],
        [r"\textit{" + pu.paper_text(st) + r"}" for _, _, st in spans],
        fontsize=FONT - 0.2,
    )
    ax_r.tick_params(axis="y", length=0, pad=2.5)
    if len(machines) > 1:
        handles = [
            mpl.lines.Line2D(
                [], [], ls="", marker=MARKERS[i % len(MARKERS)], color=f"C{i}", ms=3
            )
            for i in machines.values()
        ]
        fig.canvas.draw()
        x_mid = 0.5 * (axes[0].get_position().x0 + axes[1].get_position().x1)
        y_top = axes[0].get_position().y1
        fig.legend(
            handles,
            [pu.paper_text(legend.get(label, label)) for label in machines],
            loc="lower center",
            bbox_to_anchor=(x_mid, y_top + 0.015),
            bbox_transform=fig.transFigure,
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
