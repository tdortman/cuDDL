#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = ["jsonschema", "matplotlib", "pandas", "typer"]
# ///
"""Figures for micro-benchmark comparisons (operation=micro).

Time figure: per-genome SKETCH wall and per-pair COMPARE wall, log scale,
one panel each. Error figure: estimate error versus compare time, one
panel per scored metric (Jaccard MAE against the exact lane, ANI MAE
against skani). Tools without a metric for a panel are absent from it,
which is itself information. Exact-zero errors plot at the floor with an
"exact" tag.
"""

from pathlib import Path
from typing import Annotated

import matplotlib.pyplot as plt
import plot_utils as pu
import typer
from benchmark_schema import load_result

ERROR_FLOOR = 1e-5


def read_micro(path: Path) -> list[dict]:
    report = load_result(path, "micro")
    rows = []
    for m in report["measurements"]:
        case = m["case"]
        if case.get("measurement") not in (
            "micro-sketch",
            "micro-compare",
            "micro-search",
        ):
            continue
        if case.get("measurement") == "micro-search":
            query = m["timings"]["query"]
            index = m["timings"]["index_build"]
            rows.append(
                {
                    "tool": m["implementation"]["name"],
                    "variant": m["implementation"].get("variant", ""),
                    "op": "search",
                    "k": case.get("k", 0),
                    "wall_ms": query["median_ms"],
                    "lo_ms": query["min_ms"],
                    "hi_ms": query["max_ms"],
                    "index_ms": index["median_ms"],
                    "index_lo_ms": index["min_ms"],
                    "index_hi_ms": index["max_ms"],
                    "metrics": m.get("metrics", {}),
                }
            )
            continue
        wall = m["timings"]["wall"]
        rows.append(
            {
                "tool": m["implementation"]["name"],
                "variant": m["implementation"].get("variant", ""),
                "op": case["measurement"].replace("micro-", ""),
                "k": case.get("k", 0),
                "wall_ms": wall["median_ms"],
                "lo_ms": wall["min_ms"],
                "hi_ms": wall["max_ms"],
                "metrics": m.get("metrics", {}),
            }
        )
    if not rows:
        raise ValueError(f"{path}: no micro measurements found")
    if not any(r["tool"] == "cuddl" for r in rows):
        raise ValueError(f"{path}: no cuDDL measurements, nothing to compare against")
    return rows


def label(row: dict) -> str:
    name = f"{row['tool']}\n{row['variant']}" if row["variant"] else row["tool"]
    return f"{name}\nk={row['k']}"


def main(
    report: Annotated[Path, typer.Argument(exists=True, dir_okay=False)],
    output_dir: Annotated[Path, typer.Option(file_okay=False)] = Path(
        "results/micro-comparison/plots"
    ),
) -> None:
    try:
        rows = read_micro(report)
        key = lambda r: (r["tool"], r["variant"])
        sketch = sorted((r for r in rows if r["op"] == "sketch"), key=key)
        compare = sorted((r for r in rows if r["op"] == "compare"), key=key)
        search = sorted((r for r in rows if r["op"] == "search"), key=key)
        if not sketch or not compare:
            raise ValueError("need both micro-sketch and micro-compare rows")
        output_dir.mkdir(parents=True, exist_ok=True)
        fig, axes = plt.subplots(1, 3, figsize=(16, 4))
        for ax, subset, title in (
            (axes[0], sketch, "SKETCH wall total"),
            (axes[1], compare, "COMPARE wall total"),
            (axes[2], search, "SEARCH query wall total"),
        ):
            if not subset:
                ax.set_title(title + " (none)")
                continue
            names = [label(r) for r in subset]
            medians = [r["wall_ms"] for r in subset]
            lower = [max(0.0, r["wall_ms"] - r["lo_ms"]) for r in subset]
            upper = [max(0.0, r["hi_ms"] - r["wall_ms"]) for r in subset]
            ax.bar(names, medians, yerr=[lower, upper], capsize=3)
            ax.set_yscale("log")
            ax.set_ylabel("Median ms, log scale")
            ax.set_title(title)
            ax.tick_params(axis="x", labelrotation=20)
        fig.tight_layout()
        fig.savefig(output_dir / "micro_time.png", dpi=200, bbox_inches="tight")
        pu.save_figure(fig, output_dir / "micro_time.pdf")
        fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
        for ax, metric, title in (
            (axes[0], "jaccard_mae_vs_exact", "Jaccard MAE vs exact"),
            (axes[1], "ani_mae_vs_skani", "ANI MAE vs skani"),
        ):
            scored = [r for r in compare if r["metrics"].get(metric) is not None]
            if not scored:
                ax.set_title(f"{title} (no tool reports it)")
                continue
            for row in scored:
                value = row["metrics"][metric]
                timed = row["metrics"].get(
                    "per_pair_ms",
                    row["wall_ms"] / max(row["metrics"].get("pairs", 1), 1),
                )
                ax.scatter(
                    [timed],
                    [max(value, ERROR_FLOOR)],
                    label=f"{row['tool']} {row['variant']}".strip(),
                    s=60,
                )
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.set_xlabel("Compare ms per pair")
            ax.set_ylabel("MAE, log scale (floor = exact)")
            ax.set_title(title)
            ax.legend(fontsize=8)
        if search:
            names = [label(r) for r in search]
            x = range(len(search))
            width = 0.35
            axes[2].bar(
                [i - width / 2 for i in x],
                [r["metrics"].get("recall_at_k", 0.0) for r in search],
                width,
                label="recall@k",
            )
            axes[2].bar(
                [i + width / 2 for i in x],
                [r["metrics"].get("top1_rate", 0.0) for r in search],
                width,
                label="top-1",
            )
            axes[2].set_xticks(list(x))
            axes[2].set_xticklabels(names, rotation=20)
            axes[2].set_ylim(0, 1.05)
            axes[2].set_title("SEARCH recall and top-1 vs exact ranking")
            axes[2].legend(fontsize=8)
        else:
            axes[2].set_title("SEARCH recall (no tool reports it)")
        fig.tight_layout()
        fig.savefig(output_dir / "micro_error.png", dpi=200, bbox_inches="tight")
        pu.save_figure(fig, output_dir / "micro_error.pdf")

        typer.echo(
            f"{'tool':<14}{'op':<9}{'wall_ms':>10}{'per_unit_ms':>13}{'/exact':>10}{'/skani':>10}"
        )
        for row in sketch + compare:
            metrics = row["metrics"]
            unit = metrics.get(
                "per_genome_ms", metrics.get("per_pair_ms", row["wall_ms"])
            )
            typer.echo(
                f"{row['tool']:<14}{row['op']:<9}{row['wall_ms']:>10.1f}{unit:>13.2f}"
                f"{metrics.get('jaccard_mae_vs_exact', float('nan')):>10.4f}"
                f"{metrics.get('ani_mae_vs_skani', float('nan')):>10.4f}"
            )
    except ValueError as error:
        raise typer.BadParameter(str(error)) from error


if __name__ == "__main__":
    typer.run(main)
