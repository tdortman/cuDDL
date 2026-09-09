#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = ["jsonschema", "typer"]
# ///
"""Run cuDDL variants and RabbitSketch sequentially into a report directory."""

import itertools
import json
import random
import shlex
import subprocess
import tempfile
from pathlib import Path
from typing import Annotated

import typer
from benchmark_schema import load_result
from typer.core import TyperCommand

ROOT = Path(__file__).resolve().parent.parent


def read_plan_cap(plan: Path) -> int:
    """Read a probed resident cap from a --resident-plan JSON document."""
    try:
        payload = json.loads(plan.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise typer.BadParameter(f"unreadable resident plan {plan}: {error}") from error
    cap = payload.get("resident_batch_bytes") if isinstance(payload, dict) else None
    if isinstance(cap, bool) or not isinstance(cap, int) or cap <= 0:
        raise typer.BadParameter(
            f"resident plan {plan} returned invalid resident_batch_bytes: {cap!r}"
        )
    return cap


def run(command: list[str]) -> None:
    typer.echo("+ " + shlex.join(command), err=True)
    subprocess.run(command, cwd=ROOT, check=True)


class InputCommand(TyperCommand):
    """Let each file-list option consume shell-expanded paths until the next option."""

    def parse_args(self, ctx, args):
        expanded = []
        group = None
        needs_value = False
        for index, argument in enumerate(args):
            if argument == "--":
                expanded.extend(args[index:])
                break
            option, separator, _ = argument.partition("=")
            if option in ("--reference", "--query"):
                group, needs_value = option, not separator
            elif argument.startswith("-"):
                group = None
            elif group:
                if not needs_value:
                    expanded.append(group)
                needs_value = False
            expanded.append(argument)
        return super().parse_args(ctx, expanded)


def discover(inputs: list[Path]) -> list[Path]:
    files = set()
    for source in inputs:
        if source.is_file():
            files.add(source.resolve())
            continue
        for path in source.rglob("*"):
            if not path.is_file():
                continue
            name = Path(path.name.lower())
            if name.suffix in (".gz", ".bgz", ".bgzf"):
                name = Path(name.stem)
            if name.suffix in (
                ".fa",
                ".fna",
                ".fasta",
                ".ffn",
                ".frn",
                ".fq",
                ".fastq",
            ):
                files.add(path.resolve())
    return sorted(files)


def main(
    inputs: Annotated[
        list[Path] | None,
        typer.Argument(
            exists=True,
            file_okay=False,
            help="One pooled directory, or reference and query directories",
        ),
    ] = None,
    reference: Annotated[
        list[Path] | None,
        typer.Option(
            "--reference",
            exists=True,
            help="Reference files/directories; repeatable or space-separated",
        ),
    ] = None,
    query: Annotated[
        list[Path] | None,
        typer.Option(
            "--query",
            exists=True,
            help="Query files/directories; repeatable or space-separated",
        ),
    ] = None,
    query_percent: Annotated[
        float,
        typer.Option(min=0, max=100, help="Percentage of files reserved for queries"),
    ] = 20,
    split_seed: Annotated[
        int, typer.Option(help="Seed for the reproducible file split")
    ] = 42,
    output_dir: Annotated[Path, typer.Option(file_okay=False)] = Path(
        "results/pipeline-comparison/reports"
    ),
    build_dir: Annotated[Path, typer.Option(file_okay=False)] = Path("build"),
    topology: Annotated[
        str | None,
        typer.Option(
            help="batch or all-to-all; default: batch with query inputs, otherwise all-to-all"
        ),
    ] = None,
    samples: Annotated[int, typer.Option(min=2, max=10000)] = 20,
    warmups: Annotated[int, typer.Option(min=0, max=1000)] = 3,
    threads: Annotated[
        int | None,
        typer.Option(min=1, help="RabbitSketch workers; default: available CPUs"),
    ] = None,
    implementations: Annotated[
        str,
        typer.Option(help="Comma-separated implementations to run: cuddl, rabbitsketch"),
    ] = "cuddl,rabbitsketch",
    ingest: Annotated[
        str,
        typer.Option(help="cuDDL ingestion: packed, or sequence for a large corpus"),
    ] = "packed",
    workers: Annotated[
        int | None,
        typer.Option(
            min=0,
            max=64,
            help="cuDDL file loading workers for --ingest sequence; default: eight",
        ),
    ] = None,
    resident_bytes: Annotated[
        int,
        typer.Option(
            "--resident-bytes",
            min=0,
            help="Bounded resident staging payload in bytes; 0 probes GPU free memory for --ingest sequence",
        ),
    ] = 0,
) -> None:
    """Build and run both implementations at k=25 and 4,096 buckets/entries."""
    inputs = inputs or []
    if inputs and (reference or query):
        raise typer.BadParameter(
            "use positional directories or --reference/--query, not both"
        )
    if len(inputs) > 2 or (not inputs and not reference):
        raise typer.BadParameter("provide one or two directories, or --reference paths")
    pooled = len(inputs) == 1
    if inputs:
        reference = inputs[:1]
        query = inputs[1:]
    topology = topology or ("batch" if query else "all-to-all")
    if topology not in ("batch", "all-to-all"):
        raise typer.BadParameter("topology must be batch or all-to-all")
    if topology == "all-to-all" and query:
        raise typer.BadParameter(
            "all-to-all accepts references only; use batch with query inputs"
        )
    references = discover(reference or [])
    queries = discover(query or [])
    if pooled and topology == "batch":
        if not 0 < query_percent < 100:
            raise typer.BadParameter(
                "query-percent must be strictly between 0 and 100 for a split"
            )
        if len(references) < 2:
            raise typer.BadParameter(
                "splitting requires at least two distinct FASTX files"
            )
        ordered = references.copy()
        random.Random(split_seed).shuffle(ordered)
        query_count = max(
            1, min(len(ordered) - 1, int(len(ordered) * query_percent / 100))
        )
        queries = sorted(ordered[:query_count])
        references = sorted(ordered[query_count:])
        typer.echo(
            f"Split {len(ordered)} files: {len(references)} references, {len(queries)} queries "
            f"({100 * len(queries) / len(ordered):.2f}%; requested {query_percent:g}%; seed {split_seed})"
        )
    elif topology == "batch":
        if not references or not queries:
            raise typer.BadParameter(
                "batch requires nonempty reference and query inputs"
            )
        typer.echo(
            f"Explicit inputs: {len(references)} references, {len(queries)} queries"
        )
    else:
        if len(references) < 2:
            raise typer.BadParameter(
                "all-to-all requires at least two distinct FASTX files"
            )
        typer.echo(f"All-to-all: {len(references)} reference files; no query split")
    output_dir = output_dir.resolve()
    if any(output_dir.glob("*.json")):
        raise typer.BadParameter(
            "output directory already contains JSON reports; use a new directory"
        )
    selected = [name.strip() for name in implementations.split(",") if name.strip()]
    unknown = [name for name in selected if name not in ("cuddl", "rabbitsketch")]
    if unknown or not selected:
        raise typer.BadParameter(f"unknown implementations: {', '.join(unknown) or 'none'}")
    if ingest not in ("packed", "sequence"):
        raise typer.BadParameter("ingest must be packed or sequence")
    build_dir = build_dir if build_dir.is_absolute() else ROOT / build_dir
    common = [
        "--topology",
        topology,
        "--samples",
        str(samples),
        "--warmups",
        str(warmups),
    ]
    targets = [
        target
        for implementation, target in (
            ("cuddl", "cuddl-pipeline-benchmark"),
            ("rabbitsketch", "rabbitsketch-pipeline-benchmark"),
        )
        if implementation in selected
    ]
    run(["meson", "compile", "-C", str(build_dir), *targets])
    output_dir.mkdir(parents=True, exist_ok=True)
    # GPU isolated-stage coverage requires a query even when the resident all-to-all path ignores it.
    gpu_queries = queries or references[:1]
    cuddl_variants = (
        list(itertools.product(("compact", "packed"), ("sparse", "dense")))
        if "cuddl" in selected
        else []
    )
    with tempfile.TemporaryDirectory(prefix=".pipeline-", dir=output_dir) as temporary:
        configs = {}
        for implementation, query_files in (
            ("cuddl", gpu_queries),
            ("rabbitsketch", queries),
        ):
            if implementation not in selected:
                continue
            config = Path(temporary) / f"{implementation}.toml"
            config.write_text(
                "reference = "
                + json.dumps([str(path) for path in references], ensure_ascii=False)
                + "\n"
                + (
                    "query = "
                    + json.dumps(
                        [str(path) for path in query_files], ensure_ascii=False
                    )
                    + "\n"
                    if query_files
                    else ""
                ),
                encoding="utf-8",
            )
            configs[implementation] = config
        effective_resident_bytes = resident_bytes
        if ingest == "sequence" and resident_bytes == 0:
            if "cuddl" not in selected:
                raise typer.BadParameter(
                    "sequence runs with the default --resident-bytes 0 need the GPU "
                    "probe; pass explicit --resident-bytes for CPU-only runs"
                )
            caps = []
            for rows, index in cuddl_variants:
                plan = Path(temporary) / f"plan-{rows}-{index}.json"
                run(
                    [
                        str(build_dir / "benchmarks/cuddl-pipeline-benchmark"),
                        "--topology",
                        topology,
                        "--ingest",
                        "sequence",
                        "--resident-bytes",
                        "0",
                        "--resident-plan",
                        *(["--workers", str(workers)] if workers is not None else []),
                        "--rows",
                        rows,
                        "--index",
                        index,
                        "--minimum-matches",
                        "0",
                        "--config",
                        str(configs["cuddl"]),
                        "--output",
                        str(plan),
                    ]
                )
                caps.append(read_plan_cap(plan))
            effective_resident_bytes = min(caps)
            typer.echo(
                f"Probed resident cap {effective_resident_bytes} bytes "
                f"(min over {len(caps)} cuDDL variants)"
            )
        commands = [
            (
                f"cuddl-{rows}-{index}.json",
                [
                    str(build_dir / "benchmarks/cuddl-pipeline-benchmark"),
                    *common,
                    "--ingest",
                    ingest,
                    "--resident-bytes",
                    str(effective_resident_bytes),
                    *(["--workers", str(workers)] if workers is not None else []),
                    "--rows",
                    rows,
                    "--index",
                    index,
                    "--minimum-matches",
                    "0",
                ],
            )
            for rows, index in cuddl_variants
        ]
        if "rabbitsketch" in selected:
            commands.append(
                (
                    "rabbitsketch.json",
                    [
                        str(build_dir / "benchmarks/rabbitsketch-pipeline-benchmark"),
                        *common,
                        "--k",
                        "25",
                        "--ingest",
                        ingest,
                        "--resident-bytes",
                        str(effective_resident_bytes),
                        "--sketch-size",
                        "4096",
                        *(["--threads", str(threads)] if threads is not None else []),
                    ],
                )
            )
        for name, command in commands:
            report = Path(temporary) / name
            implementation = "rabbitsketch" if name == "rabbitsketch.json" else "cuddl"
            run(
                [
                    *command,
                    "--config",
                    str(configs[implementation]),
                    "--output",
                    str(report),
                ]
            )
            load_result(report, "pipeline")
            report.replace(output_dir / name)
    typer.echo(f"Saved {len(commands)} reports to {output_dir}")


if __name__ == "__main__":
    app = typer.Typer()
    app.command(cls=InputCommand)(main)
    app()
