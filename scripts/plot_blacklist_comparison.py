#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = ["matplotlib", "pandas", "typer"]
# ///
"""Plot the blacklist suite's GPU timings and paired filtered/unfiltered ratios."""

import csv
import json
import math
from pathlib import Path
from typing import Annotated

import matplotlib.pyplot as plt
import plot_utils as pu
import typer


def load(directory: Path) -> tuple[dict, list[dict]]:
    manifest = json.loads((directory / "blacklist-comparison.json").read_text())
    rows = []
    seen = set()
    for entry in manifest["reports"]:
        path = directory / entry["report"]
        with path.open(newline="") as source:
            records = list(csv.DictReader(source))
        if len(records) != 1:
            raise ValueError(f"{path}: expected one NVBench state")
        row = records[0]
        key = entry["path"], entry["order"], entry["variant"]
        if (
            key in seen
            or entry["order"] not in {"forward", "reverse"}
            or entry["variant"] not in {"unfiltered", "bbtools"}
        ):
            raise ValueError(f"{path}: duplicate or unexpected comparison state")
        seen.add(key)
        expected_blacklist = (
            manifest["blacklist"] if entry["variant"] == "bbtools" else ""
        )
        if (
            row["Benchmark"] != "sequence_construction"
            or row["Skipped"] != "No"
            or row["Path"] != entry["path"]
            or row["Blacklist"] != expected_blacklist
            or (row["K"], row["Offset"], row["Floor"], row["BlocksPerSM"])
            != ("25", "0", "1", "0")
            or int(row["Samples"]) != manifest["samples"]
        ):
            raise ValueError(
                f"{path}: incomplete or mismatched benchmark configuration"
            )
        time_ms = float(row["GPU Time (sec)"]) * 1000
        if not math.isfinite(time_ms) or time_ms <= 0 or int(row["Kmers"]) <= 0:
            raise ValueError(f"{path}: invalid GPU time or k-mer count")
        rows.append(
            {
                **entry,
                "time_ms": time_ms,
                "device": (row["Device"], row["Device Name"]),
                "kmers": int(row["Kmers"]),
            }
        )
    paths = list(dict.fromkeys(row["path"] for row in rows))
    expected = {
        (path, order, variant)
        for path in paths
        for order in ("forward", "reverse")
        for variant in ("unfiltered", "bbtools")
    }
    if not rows or seen != expected:
        raise ValueError("Every input needs both variants in forward and reverse order")
    if len({row["device"] for row in rows}) != 1:
        raise ValueError("Cannot compare timings from different GPUs")
    for path in paths:
        if (
            len({(row["kmers"], row["label"]) for row in rows if row["path"] == path})
            != 1
        ):
            raise ValueError(f"{path}: paired input counts or labels disagree")
    return manifest, rows


def main(
    directory: Annotated[Path, typer.Argument(exists=True, file_okay=False)],
    output: Annotated[
        Path | None, typer.Option(help="Output stem for PNG and PDF")
    ] = None,
) -> None:
    try:
        manifest, rows = load(directory)
    except (ValueError, KeyError, OSError) as error:
        raise typer.BadParameter(str(error)) from error
    paths = list(dict.fromkeys(row["path"] for row in rows))
    labels = [
        next(row["label"] for row in rows if row["path"] == path) for path in paths
    ]
    values = {
        (row["path"], row["order"], row["variant"]): row["time_ms"] for row in rows
    }
    output = output or directory / "blacklist-comparison"
    output.parent.mkdir(parents=True, exist_ok=True)
    with plt.rc_context({"font.size": pu.DEFAULT_FONT_SIZE, "legend.fontsize": 11}):
        fig, (times, ratios) = plt.subplots(
            1, 2, figsize=(max(10, len(paths) * 3), 4.5), layout="constrained"
        )
        for index, (variant, label, color, hatch) in enumerate(
            [
                ("unfiltered", "cuDDL, no blacklist", pu.FILTER_COLORS["cuddl"], ""),
                ("bbtools", "cuDDL, BBTools blacklist", "#D55E00", "//"),
            ]
        ):
            x = [i + (index - 0.5) * 0.36 for i in range(len(paths))]
            means = [
                sum(values[path, order, variant] for order in ("forward", "reverse"))
                / 2
                for path in paths
            ]
            times.bar(
                x,
                means,
                width=0.34,
                label=pu.paper_text(label),
                color=color,
                hatch=hatch,
                edgecolor="black",
                linewidth=0.6,
            )
            for order, marker in (("forward", "o"), ("reverse", "s")):
                times.scatter(
                    [
                        position + (-0.04 if order == "forward" else 0.04)
                        for position in x
                    ],
                    [values[path, order, variant] for path in paths],
                    marker=marker,
                    facecolors="white",
                    edgecolors="black",
                    s=28,
                    zorder=3,
                )
        for order, marker in (("forward", "o"), ("reverse", "s")):
            ratios.scatter(
                [
                    i + (-0.04 if order == "forward" else 0.04)
                    for i in range(len(paths))
                ],
                [
                    values[path, order, "bbtools"] / values[path, order, "unfiltered"]
                    for path in paths
                ],
                marker=marker,
                s=45,
                label=pu.paper_text(f"{order.capitalize()} order"),
                color="black",
                facecolors="none" if order == "reverse" else "black",
            )
        for ax in (times, ratios):
            ax.set_xticks(
                range(len(paths)),
                [pu.paper_text(label) for label in labels],
                rotation=25,
                ha="right",
            )
            ax.set_ylim(bottom=0)
            ax.grid(axis="y", alpha=pu.GRID_ALPHA)
            ax.set_axisbelow(True)
            ax.legend(frameon=False)
        times.set_ylabel(pu.paper_text("GPU construction time [ms]"))
        ratios.set_ylabel(pu.paper_text("Time ratio, blacklist / no blacklist"))
        ratios.axhline(1, color="0.4", linestyle="--", linewidth=1)
        for extension in ("png", "pdf"):
            destination = output.with_suffix(f".{extension}")
            fig.savefig(destination, dpi=300, facecolor="white")
            typer.echo(f"Saved {destination}")
        plt.close(fig)
    typer.echo(
        f"Each point is one {manifest['samples']}-sample NVBench GPU mean; bars average both orders. Parsing and upload are excluded."
    )


if __name__ == "__main__":
    typer.run(main)
