#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = ["matplotlib", "numpy", "pandas", "typer"]
# ///
"""Plot NVBench floor-sweep CSVs and validate a forward-trained size lookup."""

import json
import math
from itertools import product
from pathlib import Path
from tempfile import TemporaryDirectory
from textwrap import fill
from typing import Annotated

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plot_utils as pu
import typer
from matplotlib.colors import TwoSlopeNorm

ROUNDS = [0, 1, 2, 4, 8, 12, 16, 24, 32, 48, 64, 96, 128]
KEY = ["Order", "Buckets", "Items", "Input", "StartPercent"]


def normalize(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty or frame.duplicated(KEY + ["FloorRounds"]).any():
        raise ValueError("Empty sweep or duplicate configurations")
    if not np.isfinite(frame["Time"]).all() or (frame["Time"] <= 0).any():
        raise ValueError("GPU times must be positive and finite")
    base = frame[frame.FloorRounds == 0][KEY + ["Time"]].rename(
        columns={"Time": "Baseline"}
    )
    result = frame.merge(base, on=KEY, how="left", validate="many_to_one")
    if result.Baseline.isna().any():
        raise ValueError("Missing zero-round baseline")
    result["Speedup"] = result.Baseline / result.Time
    result["Relative"] = result.Time / result.Baseline
    return result


def load(directory: Path) -> pd.DataFrame:
    frames = []
    for path in sorted(directory.glob("floor-sweep-*.csv")):
        order = path.name.split("-")[2]
        if order not in {"forward", "reverse"}:
            raise ValueError(f"{path}: unknown sweep order")
        frame = pd.read_csv(path).rename(columns={"GPU Time (sec)": "Time"})
        report = json.loads(path.with_suffix(".json").read_text())
        (benchmark,) = report["benchmarks"]
        expected = math.prod(len(axis["values"]) for axis in benchmark["axes"]) * len(
            benchmark["devices"]
        )
        if len(frame) != expected or not frame.Skipped.eq("No").all():
            raise ValueError(f"{path}: incomplete or skipped states")
        for axis in benchmark["axes"]:
            declared = {str(v["value"]) for v in axis["values"]}
            if set(frame[axis["name"]].astype(str)) != declared:
                raise ValueError(
                    f"{path}: CSV disagrees with declared {axis['name']} grid"
                )
        if (
            not frame.Benchmark.eq("construction_efficiency").all()
            or not frame.Misaligned.eq(0).all()
        ):
            raise ValueError(f"{path}: unexpected benchmark or alignment")
        if set(frame.FloorRounds) != set(ROUNDS) or (frame.Samples < 1).any():
            raise ValueError(f"{path}: incomplete round grid or invalid sample counts")
        frame["RequestedItems"] = frame.Items
        frame["Items"] = frame.Kmers
        if (frame.Items <= 0).any():
            raise ValueError(f"{path}: no k-mers to plot")
        frames.append(frame.assign(Order=order, Source=path.name))
    if not frames:
        raise ValueError("No floor-sweep CSV reports found")
    result = normalize(pd.concat(frames, ignore_index=True))
    columns = KEY[1:] + ["FloorRounds"]
    forward = set(
        result[result.Order == "forward"][columns].itertuples(index=False, name=None)
    )
    reverse = set(
        result[result.Order == "reverse"][columns].itertuples(index=False, name=None)
    )
    if not forward or forward != reverse:
        raise ValueError("Forward and reverse sweep grids must match")
    if result["Device Name"].nunique() != 1 or result.Device.nunique() != 1:
        raise ValueError("Do not aggregate different GPUs")
    return result


def select(frame: pd.DataFrame, keys: list[str], value: str) -> pd.DataFrame:
    best = frame.groupby(keys)[value].transform("min")
    near = frame[frame[value] <= best * 1.02].sort_values("FloorRounds")
    return near.drop_duplicates(keys).copy()


def analyze(frame: pd.DataFrame):
    best = frame.sort_values(["Time", "FloorRounds"]).drop_duplicates(KEY)
    near = select(frame, KEY, "Time")
    train = frame[(frame.Order == "forward") & (frame.StartPercent != 50)].copy()
    train["LogRelative"] = np.log(train.Relative)
    costs = train.groupby(
        ["Buckets", "Items", "FloorRounds"], as_index=False
    ).LogRelative.mean()
    costs["Cost"] = np.exp(costs.LogRelative)
    lookup = select(costs, ["Buckets", "Items"], "Cost")[
        ["Buckets", "Items", "FloorRounds", "Cost"]
    ]
    reverse = frame[frame.Order == "reverse"]
    proposed = reverse.merge(
        lookup[["Buckets", "Items", "FloorRounds"]], validate="many_to_one"
    ).assign(Policy="Size lookup")
    current = reverse[reverse.FloorRounds == 1].assign(Policy="Current rule")
    oracle = best[best.Order == "reverse"].assign(Policy="Fastest tested")
    evaluation = pd.concat([proposed, current, oracle], ignore_index=True)
    return best, near, lookup, evaluation


def size_label(n: int) -> str:
    for unit, scale in (("Gi", 1 << 30), ("Mi", 1 << 20), ("Ki", 1 << 10)):
        if n >= scale:
            return f"{n / scale:.3g}{unit}"
    return str(n)


def figures(frame, best, lookup, evaluation, directory):
    labels_path = directory / "floor-sweep-labels.json"
    input_labels = json.loads(labels_path.read_text()) if labels_path.exists() else {}
    inputs, buckets = list(frame.Input.unique()), sorted(frame.Buckets.unique())

    def save(fig, name):
        for extension in ("png", "pdf"):
            path = directory / f"{name}.{extension}"
            fig.savefig(
                path, dpi=600, facecolor="white", bbox_inches="tight", pad_inches=0.2
            )
            typer.secho(f"Saved {path}", fg=typer.colors.GREEN)
        plt.close(fig)

    with plt.rc_context(
        {
            "font.size": pu.DEFAULT_FONT_SIZE,
            "xtick.labelsize": pu.TICK_LABEL_FONT_SIZE,
            "ytick.labelsize": pu.TICK_LABEL_FONT_SIZE,
            "legend.fontsize": pu.LEGEND_FONT_SIZE,
            "legend.framealpha": pu.LEGEND_FRAME_ALPHA,
            "figure.titlesize": pu.TITLE_FONT_SIZE,
            "lines.linewidth": pu.LINE_WIDTH,
            "lines.markersize": pu.MARKER_SIZE,
            "svg.fonttype": "none",
        }
    ):
        medians = frame.groupby(
            ["Input", "Buckets", "Items", "FloorRounds"]
        ).Speedup.median()
        norm = TwoSlopeNorm(
            vmin=min(0.99, medians.min()), vcenter=1, vmax=max(1.01, medians.max())
        )
        fig, axes = plt.subplots(
            len(inputs),
            len(buckets),
            figsize=(
                10 * len(buckets),
                max(
                    5,
                    1.5
                    + 0.22 * frame.groupby(["Input", "Buckets"]).Items.nunique().max(),
                )
                * len(inputs),
            ),
            layout="constrained",
            squeeze=False,
        )
        for row, input_name in enumerate(inputs):
            for col, bucket in enumerate(buckets):
                ax = axes[row, col]
                sizes = sorted(
                    frame[
                        (frame.Input == input_name) & (frame.Buckets == bucket)
                    ].Items.unique()
                )
                size_labels = [size_label(n) for n in sizes]
                label = (
                    Path(input_name[6:]).name
                    if input_name.startswith("fasta=")
                    else input_name
                )
                values = (
                    medians.loc[(input_name, bucket)]
                    .unstack("FloorRounds")
                    .reindex(index=sizes, columns=ROUNDS)
                )
                image = ax.imshow(
                    values,
                    origin="lower",
                    aspect="auto",
                    interpolation="nearest",
                    cmap="RdBu",
                    norm=norm,
                )
                ax.scatter(
                    np.argmax(values.to_numpy(), axis=1),
                    np.arange(len(sizes)),
                    marker="o",
                    s=40,
                    facecolors="none",
                    edgecolors="black",
                    linewidths=1.5,
                )
                pu.format_axis(
                    ax,
                    "Floor warmup rounds",
                    "Input size [k-mers]",
                    xscale="linear",
                    grid=False,
                )
                ax.set_title(
                    f"{fill(input_labels.get(input_name, label), width=32)}\n{bucket} buckets",
                    usetex=False,
                    parse_math=False,
                    fontsize=pu.TITLE_FONT_SIZE,
                    fontweight="bold",
                )
                ax.set_xticks(range(len(ROUNDS)), ROUNDS, rotation=45)
                ax.set_yticks(range(len(sizes)), size_labels)
        fig.colorbar(
            image,
            ax=axes,
            label="Median speedup over zero rounds",
            shrink=0.7,
        )
        fig.suptitle(
            "Input size and floor rounds\nMedians over two sweep orders and genome windows\nCircles mark row maxima"
        )
        save(fig, "floor-rounds-heatmap")

        for kind in ("rounds", "speedup"):
            fig, axes = plt.subplots(
                len(inputs),
                len(buckets),
                figsize=(10 * len(buckets), 5 * len(inputs)),
                layout="constrained",
                squeeze=False,
            )
            for row, input_name in enumerate(inputs):
                for col, bucket in enumerate(buckets):
                    ax = axes[row, col]
                    sizes = sorted(
                        frame[
                            (frame.Input == input_name) & (frame.Buckets == bucket)
                        ].Items.unique()
                    )
                    size_labels = [size_label(n) for n in sizes]
                    label = (
                        Path(input_name[6:]).name
                        if input_name.startswith("fasta=")
                        else input_name
                    )
                    if kind == "rounds":
                        data = best[
                            (best.Input == input_name) & (best.Buckets == bucket)
                        ]
                        spread = (
                            data.groupby("Items")
                            .FloorRounds.agg(["median", "min", "max"])
                            .reindex(sizes)
                        )
                        x = np.arange(len(sizes))
                        ax.errorbar(
                            x,
                            spread["median"],
                            yerr=[
                                spread["median"] - spread["min"],
                                spread["max"] - spread["median"],
                            ],
                            fmt="o-",
                            color=pu.FILTER_COLORS["cuddl"],
                            capsize=3,
                            label="Fastest tested: median + range",
                        )
                        rule = (
                            lookup[lookup.Buckets == bucket]
                            .set_index("Items")
                            .reindex(sizes)
                        )
                        ax.plot(
                            x,
                            rule.FloorRounds,
                            "v--",
                            color=pu.FILTER_COLORS["cuddl_paper"],
                            label="Forward-trained size lookup",
                        )
                        ax.plot(
                            x,
                            np.ones(len(sizes)),
                            "D:",
                            color=pu.FILTER_COLORS["cuddl_bbtools"],
                            label="Current rule",
                        )
                        ax.set_ylabel("Floor warmup rounds")
                        ax.set_ylim(-2, best.FloorRounds.max() + 4)
                    else:
                        data = evaluation[
                            (evaluation.Input == input_name)
                            & (evaluation.Buckets == bucket)
                        ]
                        for policy, style, color in [
                            ("Fastest tested", "o-", pu.FILTER_COLORS["cuddl"]),
                            ("Size lookup", "v--", pu.FILTER_COLORS["cuddl_paper"]),
                            ("Current rule", "D:", pu.FILTER_COLORS["cuddl_bbtools"]),
                        ]:
                            spread = (
                                data[data.Policy == policy]
                                .groupby("Items")
                                .Speedup.agg(["median", "min", "max"])
                                .reindex(sizes)
                            )
                            ax.errorbar(
                                np.arange(len(sizes)),
                                spread["median"],
                                yerr=[
                                    spread["median"] - spread["min"],
                                    spread["max"] - spread["median"],
                                ],
                                fmt=style,
                                color=color,
                                capsize=3,
                                label=policy,
                            )
                        ax.axhline(1, color="grey", linewidth=pu.REFERENCE_LINE_WIDTH)
                        ax.set_ylabel("Speedup over zero rounds")
                        ax.set_ylim(0.8, max(2, evaluation.Speedup.max() * 1.05))
                    pu.format_axis(
                        ax,
                        "Input size [k-mers]",
                        ax.get_ylabel(),
                        xscale="linear",
                    )
                    ax.set_title(
                        f"{fill(input_labels.get(input_name, label), width=32)}\n{bucket} buckets",
                        usetex=False,
                        parse_math=False,
                        fontsize=pu.TITLE_FONT_SIZE,
                        fontweight="bold",
                    )
                    ax.set_xticks(
                        np.arange(len(sizes)),
                        size_labels,
                        rotation=45,
                        ha="right",
                        rotation_mode="anchor",
                    )
            handles, labels = axes[0, 0].get_legend_handles_labels()
            fig.legend(
                handles,
                labels,
                loc="outside lower center",
                ncol=1 if len(buckets) == 1 else 3,
            )
            detail = (
                "Both orders: fastest-round ranges across runs/windows"
                if kind == "rounds"
                else "Reverse order only: median and min/max across genome windows\nNot confidence intervals"
            )
            fig.suptitle(f"{kind.capitalize()} versus input size\n{detail}")
            save(fig, f"floor-rounds-{kind}")


def self_check():
    assert [size_label(n) for n in (1, 1024, 65536, 1048576, 1 << 30)] == [
        "1",
        "1Ki",
        "64Ki",
        "1Mi",
        "1Gi",
    ]
    frame = pd.DataFrame(
        [
            {
                "Order": "forward",
                "Buckets": 2048,
                "Items": 1,
                "Input": "random",
                "StartPercent": 0,
                "FloorRounds": r,
                "Time": t,
            }
            for r, t in [(0, 2), (1, 1.01), (2, 1)]
        ]
    )
    normalized = normalize(frame)
    assert normalized.Speedup.tolist() == [1, 2 / 1.01, 2]
    assert select(normalized, KEY, "Time").FloorRounds.tolist() == [1]
    for bad in (
        frame[frame.FloorRounds != 0],
        pd.concat([frame, frame]),
        frame.assign(Time=np.nan),
    ):
        try:
            normalize(bad)
        except ValueError:
            pass
        else:
            raise AssertionError("Malformed sweep accepted")
    reverse = frame.assign(Order="reverse", Time=[2, 3, 0.5])
    combined = normalize(pd.concat([frame, reverse], ignore_index=True))
    _, _, lookup, evaluation = analyze(combined)
    assert lookup.FloorRounds.tolist() == [
        1
    ]  # Held-out winner must not affect training.
    assert evaluation[evaluation.Policy == "Size lookup"].Time.tolist() == [3]
    with TemporaryDirectory() as temporary:
        directory = Path(temporary)
        axes = {
            "Buckets": [2048],
            "Items": [0, 32],
            "Input": ["fasta=/tmp/custom genome.fna"],
            "FloorRounds": ROUNDS,
            "Misaligned": [0],
            "StartPercent": [0, 50, 100],
        }
        report = {
            "benchmarks": [
                {
                    "devices": [0],
                    "axes": [
                        {"name": name, "values": [{"value": v} for v in values]}
                        for name, values in axes.items()
                    ],
                }
            ]
        }
        records = [dict(zip(axes, values)) for values in product(*axes.values())]
        sample = pd.DataFrame(records).assign(
            Benchmark="construction_efficiency", Device=0, Skipped="No", Samples=2
        )
        sample["Device Name"] = "test GPU"
        sample["GPU Time (sec)"] = 1.0
        sample["Kmers"] = np.where(sample.Items == 0, 100, sample.Items)
        for order in ("forward", "reverse"):
            path = directory / f"floor-sweep-{order}-genome-0.csv"
            sample.to_csv(path, index=False)
            path.with_suffix(".json").write_text(json.dumps(report))
        loaded = load(directory)
        assert len(loaded) == 156 and set(loaded.Items) == {32, 100}
        from unittest.mock import patch

        import benchmark_cuddl_efficiency as runner

        fastx = directory / "input.fa"
        fastx.write_text(">input\nACGT\n")
        output = directory / "mixed"
        commands = []

        def fake_run(command, **kwargs):
            commands.append(command)
            report = Path(command[command.index("--json") + 1])
            report.write_text(json.dumps({"benchmarks": [{"states": [{}]}]}))

        with patch.object(runner.subprocess, "run", side_effect=fake_run):
            runner.main(
                suite=runner.Suite.floor_sweep,
                samples=1,
                timeout=3600,
                output=output,
                genome=[fastx],
                label=["Example"],
                random=True,
                items=[0, 32],
            )
        assert len(commands) == 4
        for command in commands:
            assert command[command.index("--timeout") + 1] == "3600"
            assert (
                "Items=[32]" if "Input=random" in command else "Items=[0,32]"
            ) in command
        assert json.loads((output / "floor-sweep-labels.json").read_text()) == {
            f"fasta={fastx.resolve()}": "Example"
        }
        commands.clear()
        second = directory / "second.fa"
        second.write_text(">second\nACGT\n")
        with patch.object(runner.subprocess, "run", side_effect=fake_run):
            runner.main(
                suite=runner.Suite.floor_sweep,
                samples=1,
                output=directory / "per-input",
                genome=[fastx, second],
                fastx_items=["0,32", "16"],
                random_items="8,64",
            )
        assert len(commands) == 6
        for command in commands:
            expected = (
                "Items=[8,64]"
                if "Input=random" in command
                else (
                    "Items=[0,32]"
                    if f"Input=fasta={fastx.resolve()}" in command
                    else "Items=[16]"
                )
            )
            assert expected in command
        for invalid in ("", "1,1", "-1", "abc", "1,"):
            try:
                runner.parse_sizes(invalid)
            except typer.BadParameter:
                pass
            else:
                raise AssertionError("Invalid size list accepted")
        sample.iloc[:-1].to_csv(path, index=False)
        try:
            load(directory)
        except ValueError:
            pass
        else:
            raise AssertionError("Truncated custom sweep accepted")
    print(
        "Self-check passed: normalization, selection, holdout separation, custom full-genome loading, mixed random/FASTX dispatch, truncation rejection"
    )


def main(
    directory: Annotated[Path, typer.Argument()] = Path("results/floor-rounds"),
    check: Annotated[bool, typer.Option("--self-check")] = False,
):
    if check:
        self_check()
        return
    frame = load(directory)
    frame = frame[frame.Buckets == 2048].copy()
    if frame.empty:
        raise ValueError("Sweep contains no 2048-bucket measurements")
    best, near, lookup, evaluation = analyze(frame)
    for name, data in [
        ("timings", frame),
        ("best", best),
        ("near-best", near),
        ("lookup", lookup.sort_values(["Buckets", "Items"])),
        ("validation", evaluation),
    ]:
        data.to_csv(directory / f"floor-rounds-{name}.csv", index=False)
    holdout = evaluation[evaluation.StartPercent == 50]
    summary = holdout.groupby(["Buckets", "Policy"]).Speedup.agg(
        geomean=lambda x: np.exp(np.log(x).mean()),
        worst="min",
        best="max",
        cases="size",
    )
    summary.to_csv(directory / "floor-rounds-holdout.csv")
    figures(frame, best, lookup, evaluation, directory)
    print(
        f"Validated {len(frame)} states; {len(best)} paired configurations; {len(lookup)} lookup entries"
    )
    print(summary.to_string())


if __name__ == "__main__":
    typer.run(main)
