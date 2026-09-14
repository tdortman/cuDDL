#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = ["jsonschema", "matplotlib", "pandas", "typer"]
# ///
"""Cross-tool pipeline comparison on shared application stages.

Every tool maps its native phases onto three stages plus the total:

- prepare: staged input becomes queryable state (cuDDL database and
  index, RabbitSketch sketches, HyperGen sketch files).
- query: queries become metrics plus materialized pairs.
- teardown: release and cleanup.
- total: contiguous end_to_end_wall over the whole run.

Stages are workload boundaries, not identical kernels. Each tool starts
from what its input_cache names: in-tree packed runs start from packed
k-mers with parsing outside timing, HyperGen starts from FASTA files
with parsing fused into prepare. The x labels name the starting point
so fused work is visible, not hidden. Resident GPU detail stays in
plot_pipeline_comparison.py, which only in-tree tools can feed.
"""

import math
from pathlib import Path
from typing import Annotated

import matplotlib.pyplot as plt
import numpy as np
import plot_utils as pu
import typer
from benchmark_schema import load_result

WALL_CLOCKS = {"nvbench_cpu_wall", "steady_clock_cpu_wall"}
STAGES = (
    ("Prepare", "prepare_wall"),
    ("Query", "query_output_wall"),
    ("Teardown", "teardown_wall"),
    ("Total", "end_to_end_wall"),
)


def read_phase(timings: dict, path: Path, key: str) -> dict:
    value = timings.get(key)
    if value is None or value.get("source") not in WALL_CLOCKS:
        raise ValueError(f"{path}: missing CPU wall {key}")
    bounds = [value.get(field) for field in ("min_ms", "median_ms", "max_ms")]
    if not all(isinstance(v, (int, float)) and math.isfinite(v) for v in bounds):
        raise ValueError(f"{path}: {key} needs finite min/median/max")
    if not 0 <= bounds[0] <= bounds[1] <= bounds[2]:
        raise ValueError(f"{path}: {key} has invalid timing bounds")
    return value


def read_pipeline(path: Path) -> dict:
    report = load_result(path, "pipeline")
    pipelines = [
        row
        for row in report["measurements"]
        if row["case"].get("measurement") == "pipeline"
    ]
    if len(pipelines) != 1:
        raise ValueError(f"{path}: expected exactly one pipeline measurement")
    pipeline = pipelines[0]
    implementation = pipeline["implementation"]["name"]
    case = pipeline["case"]
    timings = pipeline.get("timings", {})
    phases = {label: read_phase(timings, path, key) for label, key in STAGES}
    variant = pipeline["implementation"].get("variant", "")
    name = f"{implementation} {variant}".strip()
    return {
        "path": path,
        "name": name,
        "input_cache": case.get("input_cache", "?"),
        "case": case,
        "phases": phases,
        "metrics": pipeline.get("metrics", {}),
        "datasets": report["datasets"],
    }


def main(
    reports: Annotated[list[Path], typer.Argument(exists=True)],
    output_dir: Annotated[Path, typer.Option(file_okay=False)] = Path(
        "results/pipeline-comparison/plots"
    ),
) -> None:
    """Plot grouped prepare/query/teardown/total bars and print accuracy."""
    try:
        paths: list[Path] = []
        for path in reports:
            if path.is_dir():
                discovered = sorted(path.glob("*.json"))
                if not discovered:
                    raise ValueError(f"{path}: no JSON reports found")
                paths.extend(discovered)
            else:
                paths.append(path)
        rows = [read_pipeline(path) for path in paths]
        if len(rows) < 2:
            raise ValueError("provide at least two pipeline reports")
        first = (
            rows[0]["case"].get("topology"),
            rows[0]["case"].get("k"),
            tuple(sorted(v["sha256"] for v in rows[0]["datasets"].values())),
        )
        for row in rows[1:]:
            key = (
                row["case"].get("topology"),
                row["case"].get("k"),
                tuple(sorted(v["sha256"] for v in row["datasets"].values())),
            )
            if key != first:
                raise ValueError(
                    f"{row['path']}: topology, k, or input files differ; "
                    "compare reports from one runner invocation"
                )
        if len({row["name"] for row in rows}) != len(rows):
            raise ValueError("duplicate pipeline labels")
        if not any(row["name"].split(" ")[0] == "cuddl" for row in rows):
            raise ValueError("no cuDDL pipeline included, nothing to compare against")
        width = 0.18
        x = np.arange(len(rows))
        fig, ax = plt.subplots(figsize=(max(7, 2.4 * len(rows)), 4.2))
        for i, (label, _) in enumerate(STAGES):
            medians = [row["phases"][label]["median_ms"] for row in rows]
            lower = [
                m - row["phases"][label]["min_ms"] for m, row in zip(medians, rows)
            ]
            upper = [
                row["phases"][label]["max_ms"] - m for m, row in zip(medians, rows)
            ]
            ax.bar(
                x + (i - 1.5) * width,
                medians,
                width,
                yerr=[lower, upper],
                capsize=3,
                label=label,
                color=f"C{i}",
            )
        ax.set_xticks(x)
        ax.set_xticklabels(
            [f"{row['name']}\nfrom {row['input_cache']}" for row in rows]
        )
        ax.set_ylabel("Median ms, whiskers min/max")
        ax.set_title(
            f"Application stages, {rows[0]['case'].get('topology')} "
            f"topology, k={rows[0]['case'].get('k')}"
        )
        ax.legend()
        fig.tight_layout()
        fig.savefig(output_dir / "application_stages.png", dpi=200, bbox_inches="tight")
        pu.save_figure(fig, output_dir / "application_stages.pdf")
        typer.echo(f"topology={first[0]} k={first[1]}")
        for row in rows:
            metrics = row["metrics"]
            mae = metrics.get("ani_mae_vs_skani")
            accuracy = (
                f"ANI MAE vs skani {mae:.3f} "
                f"(max {metrics.get('ani_max_err_vs_skani', 0.0):.3f}, "
                f"reported {metrics.get('skani_reported_pairs', 0)})"
                if mae is not None
                else "no ANI truth"
            )
            phases = ", ".join(
                f"{label} {row['phases'][label]['median_ms']:.1f}"
                for label, _ in STAGES
            )
            typer.echo(
                f"{row['name']} [from {row['input_cache']}]: "
                f"{phases} ms; pairs {metrics.get('matches_total', '?')}, "
                f"{accuracy}"
            )
    except ValueError as error:
        raise typer.BadParameter(str(error)) from error


if __name__ == "__main__":
    typer.run(main)
