#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = ["jsonschema", "matplotlib", "pandas", "typer"]
# ///
"""Figures for micro-benchmark comparisons (operation=micro).

Twelve single-panel figures: SKETCH wall, COMPARE wall, SEARCH query wall,
resident processing time for each of those operations,
Jaccard MAE vs exact, ANI MAE vs skani, SEARCH recall and top-1, COMPARE
truth coverage, Jaccard max error, ANI max error. Tools without a metric
for a panel are absent from it, which is itself information. Exact-zero
errors plot at the floor with an "exact" tag.
Timing bars use log axes and label median milliseconds above min/max error bars.
Resident-only bars show measured CPU, GPU, or hybrid processing.
Missing resident measurements are omitted, never inferred from wall time.
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
        resident = m.get("timings", {}).get("resident")
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
                    "resident": resident,
                    "resident_device": case.get("resident_device"),
                    "resident_reuses_compare": case.get(
                        "resident_reuses_compare", False
                    ),
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
                "resident": resident,
                "resident_device": case.get("resident_device"),
                "resident_reuses_compare": case.get("resident_reuses_compare", False),
                "metrics": m.get("metrics", {}),
            }
        )
    if not rows:
        raise ValueError(f"{path}: no micro measurements found")
    return rows


def label(row: dict) -> str:
    name = f"{row['tool']}\n{row['variant']}" if row["variant"] else row["tool"]
    return name if row["op"] == "search" else f"{name}\nk={row['k']}"


def main(
    reports: Annotated[list[Path], typer.Argument(exists=True, dir_okay=False)],
    output_dir: Annotated[Path, typer.Option(file_okay=False)] = Path(
        "results/micro-comparison/plots"
    ),
) -> None:
    try:
        merged: dict[tuple[str, str], list[dict]] = {}
        for path in reports:
            by_impl: dict[tuple[str, str], list[dict]] = {}
            for row in read_micro(path):
                by_impl.setdefault((row["tool"], row["variant"]), []).append(row)
            merged.update(by_impl)
        rows = [row for impl_rows in merged.values() for row in impl_rows]
        if not any(r["tool"] == "cuddl" for r in rows):
            raise ValueError("no cuDDL measurements, nothing to compare against")
        key = lambda r: (r["tool"], r["variant"])
        sketch = sorted((r for r in rows if r["op"] == "sketch"), key=key)
        compare = sorted((r for r in rows if r["op"] == "compare"), key=key)
        search = sorted((r for r in rows if r["op"] == "search"), key=key)
        if not sketch or not compare:
            raise ValueError("need both micro-sketch and micro-compare rows")
        output_dir.mkdir(parents=True, exist_ok=True)

        def save_both(fig: plt.Figure, stem: str, runtime_labels=()) -> None:
            fig.tight_layout()
            if runtime_labels:
                fig.canvas.draw()
                renderer = fig.canvas.get_renderer()
                for bar, text in runtime_labels:
                    width = bar.get_window_extent(renderer).width * 0.95
                    while text.get_window_extent(renderer).width > width:
                        text.set_fontsize(text.get_fontsize() * 0.9)
            fig.savefig(output_dir / f"{stem}.png", dpi=200, bbox_inches="tight")
            pu.save_figure(fig, output_dir / f"{stem}.pdf")

        def time_panel(
            subset: list[dict], title: str, stem: str, *, resident: bool = False
        ) -> None:
            runtime_labels = []
            fig, ax = plt.subplots(
                figsize=(max(5.5, len(subset) * 1.15) if resident else 5.5, 4)
            )
            if not subset:
                ax.set_title(title + (" (not recorded)" if resident else " (none)"))
                if resident:
                    ax.set_axis_off()
            else:
                names = [label(r) for r in subset]
                if resident:
                    for index, row in enumerate(subset):
                        device = {
                            "cuda": "GPU",
                            "cpu": "CPU",
                            "hybrid": "CPU + GPU",
                        }.get(row["resident_device"])
                        if device:
                            names[index] += f"\n{device}"
                        if row["resident_reuses_compare"]:
                            names[index] += " (exhaustive)"
                medians = [r["wall_ms"] for r in subset]
                lower = [max(0.0, r["wall_ms"] - r["lo_ms"]) for r in subset]
                upper = [max(0.0, r["hi_ms"] - r["wall_ms"]) for r in subset]
                bars = ax.bar(names, medians, yerr=[lower, upper], capsize=3)
                texts = ax.bar_label(
                    bars, fmt="%.3g", padding=3, fontsize=pu.BAR_FONT_SIZE
                )
                runtime_labels = list(zip(bars, texts))
                ax.set_yscale("log")
                ax.margins(y=0.15)
                if resident:
                    ax.set_ylabel("Median processing time (ms), log scale")
                else:
                    ax.set_ylabel("Median ms, log scale")
                ax.set_title(title)
                ax.tick_params(axis="x", labelrotation=20)
            save_both(fig, stem, runtime_labels)

        time_panel(sketch, "SKETCH wall total", "micro_time_sketch")
        time_panel(compare, "COMPARE wall total", "micro_time_compare")
        time_panel(search, "SEARCH query wall total", "micro_time_search")
        for operation, subset in [
            ("sketch", sketch),
            ("compare", compare),
            ("search", search),
        ]:
            resident_rows = [
                {
                    **row,
                    "wall_ms": row["resident"]["median_ms"],
                    "lo_ms": row["resident"]["min_ms"],
                    "hi_ms": row["resident"]["max_ms"],
                }
                for row in subset
                if row["resident"] is not None
            ]
            time_panel(
                resident_rows,
                f"{operation.upper()} resident processing",
                f"micro_time_{operation}_resident",
                resident=True,
            )

        def scatter_panel(metric: str, title: str, ylabel: str, stem: str) -> None:
            fig, ax = plt.subplots(figsize=(5.5, 4.5))
            scored = [r for r in compare if r["metrics"].get(metric) is not None]
            if not scored:
                ax.set_title(f"{title} (no tool reports it)")
            else:
                for row in scored:
                    value = row["metrics"][metric]
                    timed = row["metrics"].get(
                        "per_pair_ms",
                        row["wall_ms"] / max(row["metrics"].get("pairs", 1), 1),
                    )
                    reported = row["metrics"].get("skani_reported_pairs")
                    count = (
                        f"n={reported}"
                        if reported is not None
                        else f"n={row['metrics'].get('pairs', 0)}"
                    )
                    ax.scatter(
                        [timed],
                        [max(value, ERROR_FLOOR)],
                        label=f"{row['tool']} {row['variant']} {count}".strip(),
                        s=60,
                    )
                ax.set_xscale("log")
                ax.set_yscale("log")
                ax.set_xlabel("Compare ms per pair")
                ax.set_ylabel(ylabel)
                ax.set_title(title)
                ax.legend(fontsize=8)
            save_both(fig, stem)

        scatter_panel(
            "jaccard_mae_vs_exact",
            "Jaccard MAE vs exact",
            "MAE, log scale (floor = exact)",
            "micro_error_jaccard",
        )
        scatter_panel(
            "ani_mae_vs_skani",
            "ANI MAE vs skani",
            "MAE, log scale (floor = exact)",
            "micro_error_ani",
        )

        fig, ax = plt.subplots(figsize=(5.5, 4.5))
        if search:
            names = [
                f"{label(r)}\nn={r['metrics'].get('queries_scored', 0)}" for r in search
            ]
            x = range(len(search))
            width = 0.35
            ax.bar(
                [i - width / 2 for i in x],
                [r["metrics"].get("recall_at_k", 0.0) for r in search],
                width,
                label="recall@k",
            )
            ax.bar(
                [i + width / 2 for i in x],
                [r["metrics"].get("top1_rate", 0.0) for r in search],
                width,
                label="top-1",
            )
            ax.set_xticks(list(x))
            ax.set_xticklabels(names, rotation=20)
            ax.set_ylim(0, 1.05)
            ax.set_title("SEARCH recall and top-1 vs exact ranking")
            ax.legend(fontsize=8)
        else:
            ax.set_title("SEARCH recall (no tool reports it)")
        save_both(fig, "micro_search_recall")

        fig, ax = plt.subplots(figsize=(5.5, 4.5))
        cover_names = [label(r) for r in compare]
        cover_x = range(len(compare))
        cover_width = 0.35
        ax.bar(
            [i - cover_width / 2 for i in cover_x],
            [
                r["metrics"].get("skani_reported_pairs", 0)
                / max(r["metrics"].get("pairs", 1), 1)
                for r in compare
            ],
            cover_width,
            label="ANI scored vs skani",
        )
        ax.bar(
            [i + cover_width / 2 for i in cover_x],
            [
                (r["metrics"].get("pairs", 0) or 0)
                / max(max(c["metrics"].get("pairs", 1) for c in compare), 1)
                for r in compare
            ],
            cover_width,
            label="pairs vs largest lane",
        )
        ax.set_xticks(list(cover_x))
        ax.set_xticklabels(cover_names, rotation=20)
        ax.set_ylim(0, 1.05)
        ax.set_title("COMPARE truth coverage")
        ax.legend(fontsize=8)
        save_both(fig, "micro_compare_coverage")

        scatter_panel(
            "jaccard_max_err_vs_exact",
            "Jaccard max error vs exact",
            "Max error, log scale (floor = exact)",
            "micro_maxerr_jaccard",
        )
        scatter_panel(
            "ani_max_err_vs_skani",
            "ANI max error vs skani",
            "Max error, log scale (floor = exact)",
            "micro_maxerr_ani",
        )

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
