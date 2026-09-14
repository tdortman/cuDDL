#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = ["jsonschema", "typer"]
# ///
"""Micro-benchmarks over shared fused operations (SKETCH, COMPARE).

Each tool runs its native fused path and reports wall time plus native
outputs; accuracy joins against exact oracles afterwards:

- SKETCH: FASTA files to queryable sketch state. CLI tools time their
  sketch command; cuDDL times reference-database build; RabbitSketch
  times pipeline prepare; cub-exact times its device sketch phase.
- COMPARE: pairs to similarity rows in one batched invocation per tool:
  cuDDL runs the pipeline benchmark with threshold zero (full refinement),
  RabbitSketch its pipeline query, cub-exact its device phase, and the CLI
  tools their native compare commands.

Truth comes from cub-exact-pairwise (exact Jaccard and containment,
verified bit-identical against a Python oracle) and chunked skani dist
(ANI). k and sketch sizes stay native per tool and are recorded; tools
are compared in error-versus-time space, never by equalizing inputs.

Caps are automatic unless passed explicitly: threads default to the CPU
count, max-kmers derives from free VRAM, the evaluated pair set derives
from a short cub probe against a per-invocation time budget, stored rows
derive from a pairs-output size target, and the skani truth chunk derives
from host RAM. Large corpora evaluate a size-spread genome subset that
fits the budget; every lane runs the same files.
"""

import gzip
import math
import os
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Annotated

import typer
from benchmark_schema import make_result, write_result

sys.path.insert(0, str(Path(__file__).resolve().parent))
from hypergen_pipeline import dataset_entries, summarize

ROOT = Path(__file__).resolve().parent.parent
APP = typer.Typer()
DEFAULT_TOOLS = "cuddl,rabbitsketch,hypergen,skani,dashing2,cub-exact"


def run(cmd: list[str], capture: bool = False) -> str:
    proc = subprocess.run(cmd, cwd=ROOT, check=True, capture_output=capture, text=True)
    return proc.stdout if capture else ""


def wall_of(
    command: list[str] | list[list[str]], samples: int, warmups: int
) -> list[float]:
    groups = command if command and isinstance(command[0], list) else [command]
    marks: list[float] = []
    for rep in range(warmups + samples):
        tick = time.perf_counter()
        for item in groups:
            run(item)
        done = time.perf_counter()
        if rep >= warmups:
            marks.append((done - tick) * 1000)
    return marks


def discover(directory: Path) -> list[Path]:
    exts = (".fa", ".fna", ".fasta", ".ffn", ".frn")
    files = {
        path
        for ext in exts
        for path in (*directory.glob(f"*{ext}"), *directory.glob(f"*{ext}.gz"))
        if path.is_file()
    }
    return sorted(files)


def pairs_for(
    references: list[Path], queries: list[Path], topology: str
) -> list[tuple[Path, Path]]:
    if topology == "all-to-all":
        return [
            (references[i], references[j])
            for i in range(len(references))
            for j in range(i + 1, len(references))
        ]
    return [(q, r) for q in queries for r in references]


_CUB_BYTES_PER_KEY = 32.0  # device bytes per pair key, measured ~28 on RTX 5070 Ti
_VRAM_FRACTION = 0.7  # share of free VRAM cub may size its buffers against
_PAIR_BUDGET_MARGIN = 1.5  # probe-to-full-run cost safety factor
_PROBE_PAIRS = 3  # stride-spread pairs timed to size the evaluated set
_PROBE_MIN_PAIRS = 12  # at or below this, skip the probe and run everything
_CHUNK_ROW_BYTES = 200  # estimated skani dist TSV bytes per pair row
_CHUNK_BUDGET_BYTES = 256 << 20  # per-invocation truth output target
_PAIRS_TARGET_BYTES = 32 << 20  # stored-pairs output target
_PAIR_ROW_BYTES = 160  # estimated stored bytes per pair row


def _cpu_count() -> int:
    return os.cpu_count() or 8


def _host_ram_bytes() -> int:
    try:
        return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
    except (ValueError, OSError):
        return 8 << 30


def _gpu_free_bytes() -> tuple[str, int]:
    proc = subprocess.run(
        ["nvidia-smi", "--query-gpu=name,memory.free", "--format=csv,nounits,noheader"],
        check=True,
        capture_output=True,
        text=True,
    )
    name, _, free = proc.stdout.splitlines()[0].partition(",")
    return name.strip(), int(free.strip()) << 20


def _est_genome_bases(path: Path) -> int:
    # Bases from a 64KB sample; .gz uses a generic ratio and is
    # exact-counted later only when its estimate nears the VRAM cap.
    size = path.stat().st_size
    if path.suffix == ".gz":
        return int(size * 4.2)
    with open(path, "rb") as stream:
        sample = stream.read(1 << 16)
    if not sample:
        return 0
    fraction = sum(1 for byte in sample if byte in b"ACGTNacgtn") / len(sample)
    return int(size * fraction)


def _exact_gz_bases(path: Path) -> int:
    count = 0
    with gzip.open(path, "rb") as stream:
        while chunk := stream.read(1 << 20):
            count += sum(1 for byte in chunk if byte in b"ACGTNacgtn")
    return count


def _spread_pick(paths: list[Path], sizes: dict[Path, int], count: int) -> list[Path]:
    # Stride over size order with both endpoints kept: min and max sizes
    # bound the per-pair cost range instead of clustering at one end.
    ordered = sorted(paths, key=lambda p: sizes.get(p, 0))
    count = max(1, min(count, len(ordered)))
    if count == 1:
        return [ordered[len(ordered) // 2]]
    last = len(ordered) - 1
    return sorted({ordered[round(i * last / (count - 1))] for i in range(count)})


@APP.command()
def main(
    genomes: Annotated[Path, typer.Argument(exists=True, file_okay=False)],
    queries: Annotated[list[Path] | None, typer.Option("--query", exists=True)] = None,
    topology: Annotated[str, typer.Option()] = "all-to-all",
    tools: Annotated[str, typer.Option()] = DEFAULT_TOOLS,
    samples: Annotated[int, typer.Option(min=1)] = 3,
    warmups: Annotated[int, typer.Option(min=0)] = 1,
    threads: Annotated[int | None, typer.Option(min=1)] = None,
    output: Annotated[Path, typer.Option()] = Path("results/micro-comparison.json"),
    pairs_out: Annotated[Path | None, typer.Option()] = None,
    max_kmers: Annotated[int | None, typer.Option(min=1)] = None,
    recall_k: Annotated[int, typer.Option(min=1)] = 3,
    min_matches: Annotated[int, typer.Option(min=0)] = 5,
    max_pairs: Annotated[int | None, typer.Option(min=0)] = None,
    match_rows: Annotated[int | None, typer.Option(min=0)] = None,
    skani_chunk: Annotated[int | None, typer.Option(min=1)] = None,
    budget_secs: Annotated[float, typer.Option(min=1)] = 600,
) -> None:
    """Time SKETCH, COMPARE, and SEARCH for each tool and score against oracles."""
    if topology not in ("batch", "all-to-all"):
        raise typer.BadParameter("topology must be batch or all-to-all")
    references = discover(genomes)
    query_list = sorted({q for paths in (queries or []) for q in paths if q.is_file()})
    if topology == "batch" and not query_list:
        raise typer.BadParameter("batch needs --query files")
    if topology == "all-to-all" and len(references) < 2:
        raise typer.BadParameter("all-to-all needs at least two genomes")
    # No fixed corpus ceiling: the autoscale block below samples the pair
    # space against a time budget instead of refusing large inputs.
    selected = [t.strip() for t in tools.split(",") if t.strip()]
    unknown = [t for t in selected if t not in DEFAULT_TOOLS.split(",")]
    if unknown or not selected:
        raise typer.BadParameter(f"unknown tools: {', '.join(unknown) or 'none'}")

    build = ROOT / "build"
    hypergen = build / "subprojects/hypergen/hyper-gen"
    skani = build / "subprojects/skani/skani"
    dashing2 = build / "subprojects/dashing2/dashing2"
    cub = build / "benchmarks/cub-exact-pairwise"
    refbuild = build / "benchmarks/cuddl-reference-build-benchmark"
    rabbit = build / "benchmarks/rabbitsketch-pipeline-benchmark"
    for binary in (hypergen, skani, dashing2, cub, refbuild, rabbit):
        if not binary.exists():
            raise typer.BadParameter(f"missing binary, build first: {binary}")

    measurements: list[dict] = []
    pair_table: list[dict] = []
    sketch_times: dict[str, list[float]] = {}
    sketch_bytes: dict[str, int] = {}
    sketch_k: dict[str, int] = {
        "cuddl": 25,
        "rabbitsketch": 25,
        "hypergen": 25,
        "dashing2": 25,
        "cub-exact": 25,
        # skani exposes no k flag; its screened minimizers use k=15 internally.
        "skani": 15,
    }

    with tempfile.TemporaryDirectory(prefix="micro-") as tmp:
        work = Path(tmp)
        # Autoscale: every cap derives from hardware and one short probe.
        # Explicit flags skip their derivation; explicit --max-pairs also
        # skips the probe and subsetting and restores the fixed-cap path.
        sizes = {
            p: _est_genome_bases(p) for p in dict.fromkeys([*references, *query_list])
        }
        auto_flags = {
            "threads": threads is None,
            "max_kmers": max_kmers is None,
            "max_pairs": max_pairs is None,
            "match_rows": match_rows is None,
            "skani_chunk": skani_chunk is None,
        }
        threads = threads or _cpu_count()
        need_cub = "cub-exact" in selected
        gpu_name = "none"
        if need_cub:
            try:
                gpu_name, gpu_free = _gpu_free_bytes()
            except Exception as error:
                raise typer.BadParameter(
                    f"auto caps need nvidia-smi ({error}); pass caps explicitly"
                ) from error
            if max_kmers is None:
                # Worst pair holds two max-size genomes at ~32 device bytes/key.
                max_kmers = int(_VRAM_FRACTION * gpu_free / (2 * _CUB_BYTES_PER_KEY))
            for path, est in sizes.items():
                if path.suffix == ".gz" and est > max_kmers // 3:
                    sizes[path] = est = _exact_gz_bases(path)
            over = sorted(str(p) for p, est in sizes.items() if est > max_kmers)
            if over:
                raise typer.BadParameter(
                    f"genomes exceed VRAM-derived max-kmers={max_kmers}: {', '.join(over)}"
                )
        else:
            try:
                gpu_name, _ = _gpu_free_bytes()
            except Exception:
                pass
        orig_pairs = len(pairs_for(references, query_list, topology))
        probe_per_pair_ms = 0.0
        subset_note = "all pairs"
        if need_cub and max_pairs is None and orig_pairs > _PROBE_MIN_PAIRS:
            probe_cmd = [
                str(cub),
                "--topology",
                topology,
                "--samples",
                "1",
                "--warmups",
                "0",
            ]
            if topology == "batch":
                probe_cmd += ["--reference", *[str(p) for p in references]]
                probe_cmd += ["--query", *[str(p) for p in query_list]]
            else:
                probe_cmd += ["--reference", *[str(p) for p in references]]
            probe_cmd += [
                "--max-kmers",
                str(max_kmers),
                "--max-pairs",
                str(_PROBE_PAIRS),
                "--match-rows",
                "1",
                "--output",
                str(work / "probe.json"),
            ]
            tick = time.perf_counter()
            run(probe_cmd)
            probe_wall_ms = (time.perf_counter() - tick) * 1000
            import json as jsonlib_probe

            probe_eval = jsonlib_probe.loads((work / "probe.json").read_text())["case"][
                "pairs_evaluated"
            ]
            probe_per_pair_ms = probe_wall_ms / max(probe_eval, 1)
            capacity = int(
                budget_secs * 1000 / (samples * probe_per_pair_ms * _PAIR_BUDGET_MARGIN)
            )
            if capacity < orig_pairs:
                if topology == "all-to-all":
                    keep = max(2, int((1 + math.sqrt(1 + 8 * capacity)) // 2))
                    references = _spread_pick(references, sizes, keep)
                else:
                    keep = max(
                        1, min(len(query_list), capacity // max(1, len(references)))
                    )
                    query_list = _spread_pick(query_list, sizes, keep)
                subset_note = "budget subset"
            max_pairs = 0
        elif max_pairs is None:
            max_pairs = 0
        total_pairs = len(pairs_for(references, query_list, topology))
        if subset_note != "all pairs":
            subset_note = f"subset {total_pairs}/{orig_pairs}"
        if match_rows is None:
            match_rows = _PAIRS_TARGET_BYTES // _PAIR_ROW_BYTES
        if skani_chunk is None:
            skani_chunk = max(
                1, _CHUNK_BUDGET_BYTES // (max(1, len(references)) * _CHUNK_ROW_BYTES)
            )
        file_args = [str(p) for p in references] + [
            str(p) for p in query_list if p not in references
        ]

        if need_cub:
            cub_oracle = work / "oracle.json"
            oracle_cmd = [str(cub), "--topology", topology, "--samples", "1"]
            if topology == "batch":
                oracle_cmd += ["--reference", *[str(p) for p in references]]
                oracle_cmd += ["--query", *[str(p) for p in query_list]]
            else:
                oracle_cmd += ["--reference", *[str(p) for p in references]]
            oracle_cmd += [
                "--max-kmers",
                str(max_kmers),
                "--max-pairs",
                str(max_pairs),
                "--match-rows",
                str(match_rows),
                "--output",
                str(cub_oracle),
            ]
            oracle_tick = time.perf_counter()
            run(oracle_cmd)
            oracle_wall_ms = (time.perf_counter() - oracle_tick) * 1000
            import json as jsonlib

            oracle = {
                (row["query"], row["reference"]): row
                for row in jsonlib.loads(cub_oracle.read_text())["pairs"]
            }
            if not probe_per_pair_ms:
                probe_per_pair_ms = oracle_wall_ms / max(len(oracle), 1)
        else:
            oracle = {}
        typer.echo(
            f"auto: threads={threads} max_kmers={max_kmers} pairs={total_pairs}/{orig_pairs} ({subset_note}) match_rows={match_rows} skani_chunk={skani_chunk} budget_secs={budget_secs} per_pair_ms={probe_per_pair_ms:.1f} gpu={gpu_name}"
        )
        autoscale_case = {
            "threads": threads,
            "max_kmers": max_kmers,
            "pair_budget_secs": budget_secs,
            "cub_per_pair_ms": round(probe_per_pair_ms, 3),
            "pairs_evaluated": total_pairs,
            "pairs_total": orig_pairs,
            "auto_caps": ",".join(k for k, v in auto_flags.items() if v) or "none",
            "gpu": gpu_name,
        }

        # ANI oracle from skani dist, chunked per query group so memory stays
        # bounded: triangle materializes the full N x N matrix. Rows with
        # non-positive ANI are treated as unreported, as with triangle.
        skani_ani: dict[tuple[str, str], float] = {}
        truth_queries = query_list if topology == "batch" else references
        for chunk_base in range(0, len(truth_queries), skani_chunk):
            chunk = truth_queries[chunk_base : chunk_base + skani_chunk]
            chunk_tsv = work / f"skani-truth-{chunk_base}.tsv"
            run(
                [
                    str(skani),
                    "dist",
                    "-q",
                    *[str(p) for p in chunk],
                    "-r",
                    *[str(p) for p in references],
                    "-o",
                    str(chunk_tsv),
                    "-t",
                    str(threads),
                ]
            )
            for line in chunk_tsv.read_text().splitlines():
                if not line or line.startswith("#"):
                    continue
                fields = line.split()
                if len(fields) >= 3:
                    try:
                        value = float(fields[2])
                    except ValueError:
                        continue
                    if value > 0:
                        skani_ani[(fields[1], fields[0])] = value
                        skani_ani[(fields[0], fields[1])] = value

        def record_sketch(
            tool: str, variant: str, marks: list[float], extra: dict | None = None
        ) -> None:
            measurements.append(
                {
                    "implementation": {"name": tool, "variant": variant},
                    "case": {
                        "measurement": "micro-sketch",
                        "topology": topology,
                        "genomes": len(file_args),
                        "k": sketch_k.get(tool, 0),
                        "threads": threads,
                    },
                    "timings": {"wall": summarize(marks)},
                    "metrics": {
                        "per_genome_ms": statistics.median(marks) / len(file_args),
                        **(extra or {}),
                    },
                }
            )

        if "hypergen" in selected:
            staged_to_real = {}
            for role, files in (("hgrefs", references), ("hgqueries", query_list)):
                role_dir = work / role
                role_dir.mkdir(exist_ok=True)
                for path in files:
                    link = role_dir / path.name
                    if link.is_symlink() or link.exists():
                        link.unlink()
                    link.symlink_to(path.resolve())
                    staged_to_real[str(link)] = str(path)
            sketch_cmds = [
                [
                    str(hypergen),
                    "sketch",
                    "-p",
                    str(work / "hgrefs"),
                    "-o",
                    str(work / "hgr.sk"),
                    "-t",
                    str(threads),
                    "-k",
                    "25",
                ]
            ]
            if topology == "batch":
                sketch_cmds.append(
                    [
                        str(hypergen),
                        "sketch",
                        "-p",
                        str(work / "hgqueries"),
                        "-o",
                        str(work / "hgq.sk"),
                        "-t",
                        str(threads),
                        "-k",
                        "25",
                    ]
                )
            marks = wall_of(sketch_cmds, samples, warmups)
            sketch_times["hypergen"] = marks
            sketch_bytes["hypergen"] = (work / "hgr.sk").stat().st_size
            if topology == "batch":
                sketch_bytes["hypergen"] += (work / "hgq.sk").stat().st_size
            record_sketch(
                "hypergen", "cpu", marks, {"sketch_bytes": sketch_bytes["hypergen"]}
            )
        if "skani" in selected:
            marks = []
            for rep in range(warmups + samples):
                out_dir = work / f"skdb{rep}"
                tick = time.perf_counter()
                run(
                    [
                        str(skani),
                        "sketch",
                        *file_args,
                        "-o",
                        str(out_dir),
                        "-t",
                        str(threads),
                    ]
                )
                done = time.perf_counter()
                if rep >= warmups:
                    marks.append((done - tick) * 1000)
            sketch_times["skani"] = marks
            sketch_bytes["skani"] = sum(
                p.stat().st_size
                for p in (work / f"skdb{warmups + samples - 1}").rglob("*")
                if p.is_file()
            )
            record_sketch(
                "skani", "cpu", marks, {"sketch_bytes": sketch_bytes["skani"]}
            )

        if "dashing2" in selected:
            marks = []
            for rep in range(warmups + samples):
                out_dir = work / f"d2_{rep}"
                out_dir.mkdir(exist_ok=True)
                tick = time.perf_counter()
                run(
                    [
                        str(dashing2),
                        "sketch",
                        "-k25",
                        "-S4096",
                        f"-p{threads}",
                        "--cache",
                        "--outprefix",
                        str(out_dir),
                        *file_args,
                    ]
                )
                done = time.perf_counter()
                if rep >= warmups:
                    marks.append((done - tick) * 1000)
            sketch_times["dashing2"] = marks
            sketch_bytes["dashing2"] = sum(
                p.stat().st_size
                for p in (work / f"d2_{warmups + samples - 1}").rglob("*")
                if p.is_file()
            )
            record_sketch(
                "dashing2",
                "SetSketch",
                marks,
                {"sketch_bytes": sketch_bytes["dashing2"]},
            )

        if "cuddl" in selected:
            db_out = work / "ref.cuddl"
            stdout = run(
                [
                    str(refbuild),
                    "--reference",
                    *file_args,
                    "--database",
                    str(db_out),
                    "--samples",
                    str(max(samples, 2)),
                ],
                capture=True,
            )
            import json as jsonlib2

            payload = jsonlib2.loads(stdout[stdout.index("{") :])
            marks = [payload["median_seconds"] * 1000] * max(samples, 1)
            sketch_times["cuddl"] = marks
            sketch_bytes["cuddl"] = db_out.stat().st_size
            record_sketch(
                "cuddl", "reference-db", marks, {"sketch_bytes": sketch_bytes["cuddl"]}
            )

        if "rabbitsketch" in selected:
            import json as jsonlib3

            cfg = work / "rabbit.toml"
            cfg.write_text(
                "reference = "
                + jsonlib3.dumps([str(p) for p in references])
                + "\n"
                + (
                    "query = " + jsonlib3.dumps([str(p) for p in query_list]) + "\n"
                    if query_list
                    else ""
                )
            )
            rep = work / "rabbit.json"
            rabbit_samples = max(samples, 2)
            run(
                [
                    str(rabbit),
                    "--topology",
                    topology,
                    "--samples",
                    str(rabbit_samples),
                    "--warmups",
                    str(warmups),
                    "--k",
                    "25",
                    "--ingest",
                    "packed",
                    "--sketch-size",
                    "4096",
                    "--config",
                    str(cfg),
                    "--output",
                    str(rep),
                ]
            )
            payload = jsonlib3.loads(rep.read_text())
            pipe = next(
                m
                for m in payload["measurements"]
                if m["case"].get("measurement") == "pipeline"
            )
            prepare = pipe["timings"]["prepare_wall"]
            marks = [prepare["median_ms"]] * max(samples, 1)
            sketch_times["rabbitsketch"] = marks
            record_sketch("rabbitsketch", "FastKMV", marks, {})
            rabbit_report, rabbit_rows = payload, pipe

        if "cub-exact" in selected:
            cub_rep = work / "cub.json"
            cub_cmd = [
                str(cub),
                "--topology",
                topology,
                "--samples",
                str(samples),
                "--warmups",
                str(warmups),
            ]
            if topology == "batch":
                cub_cmd += ["--reference", *[str(p) for p in references]]
                cub_cmd += ["--query", *[str(p) for p in query_list]]
            else:
                cub_cmd += ["--reference", *[str(p) for p in references]]
            cub_cmd += [
                "--max-kmers",
                str(max_kmers),
                "--max-pairs",
                str(max_pairs),
                "--match-rows",
                str(match_rows),
                "--output",
                str(cub_rep),
            ]
            run(cub_cmd)
            import json as jsonlib4

            payload = jsonlib4.loads(cub_rep.read_text())
            sketch_times["cub-exact"] = [payload["phases_ms"]["sketch"]["median_ms"]]
            record_sketch(
                "cub-exact",
                "gpu-exact",
                [payload["phases_ms"]["sketch"]["median_ms"]],
                {
                    f"device_{key}_bytes": value
                    for key, value in payload["phases_ms"]
                    .get("device_buffers", {})
                    .items()
                },
            )
            cub_phases = payload["phases_ms"]

        # COMPARE op per tool; errors join cub-exact Jaccard and skani ANI.
        def record_compare(
            tool: str, variant: str, marks: list[float], rows: list[dict]
        ) -> None:
            jaccard_errors, ani_errors, reported = [], [], 0
            stride = (
                (len(rows) + match_rows - 1) // match_rows
                if match_rows and len(rows) > match_rows
                else 1
            )
            for index, row in enumerate(rows):
                # Tool ANI scales differ (fraction vs percent); skani truth is percent.
                if row.get("ani") is not None and 0 < row["ani"] <= 1.5:
                    row["ani"] = row["ani"] * 100
                key = (row["query"], row["reference"])
                truth = oracle.get(key, oracle.get((key[1], key[0])))
                if truth is not None and row.get("jaccard") is not None:
                    jaccard_errors.append(abs(row["jaccard"] - truth["jaccard"]))
                ani_truth = skani_ani.get(key, skani_ani.get((key[1], key[0])))
                if ani_truth is not None and row.get("ani") is not None:
                    reported += 1
                    ani_errors.append(abs(row["ani"] - ani_truth))
                if index % stride == 0 or index == len(rows) - 1:
                    pair_table.append({"tool": tool, **row})
            metrics: dict = {
                "pairs": len(rows),
                "pair_stride": stride,
                "per_pair_ms": statistics.median(marks) / max(len(rows), 1),
            }
            if jaccard_errors:
                metrics["jaccard_mae_vs_exact"] = sum(jaccard_errors) / len(
                    jaccard_errors
                )
                metrics["jaccard_max_err_vs_exact"] = max(jaccard_errors)
            if ani_errors:
                metrics["ani_mae_vs_skani"] = sum(ani_errors) / len(ani_errors)
                metrics["ani_max_err_vs_skani"] = max(ani_errors)
                metrics["skani_reported_pairs"] = reported
            measurements.append(
                {
                    "implementation": {"name": tool, "variant": variant},
                    "case": {
                        "measurement": "micro-compare",
                        "topology": topology,
                        "pairs": len(rows),
                        "k": sketch_k.get(tool, 0),
                        "threads": threads,
                    },
                    "timings": {"wall": summarize(marks)},
                    "metrics": metrics,
                }
            )

        if "dashing2" in selected:
            listing = work / "d2list.txt"
            sketch_files = sorted((work / f"d2_{warmups + samples - 1}").glob("*.opss"))
            listing.write_text("".join(str(p) + "\n" for p in sketch_files))
            cmp_cmd = [
                str(dashing2),
                "cmp",
                "-k25",
                "-S4096",
                f"-p{threads}",
                "--presketched",
                "-F",
                str(listing),
            ]
            marks = wall_of([cmp_cmd], samples, warmups)
            # One extra untimed run supplies parseable rows.
            stdout = run(cmp_cmd, capture=True)
            # Matrix rows list values in -F order; columns match rows.
            table = [
                line.split()
                for line in stdout.splitlines()
                if line and not line.startswith("#")
            ]
            # First column is the row label; value columns follow -F order,
            # matching the sorted sketch file order.
            rows = []
            for i, row in enumerate(table):
                for j in range(i + 1, len(sketch_files)):
                    value = row[1 + j]
                    if value == "-":
                        continue
                    left = next(
                        p
                        for p in references + query_list
                        if p.name == sketch_files[i].name.split(".rc_canon")[0]
                    )
                    right = next(
                        p
                        for p in references + query_list
                        if p.name == sketch_files[j].name.split(".rc_canon")[0]
                    )
                    rows.append(
                        {
                            "query": str(left),
                            "reference": str(right),
                            "jaccard": float(value),
                            "ani": None,
                        }
                    )
            record_compare("dashing2", "SetSketch", marks, rows)
        if "hypergen" in selected:
            dist_out = work / "hg.ani"
            query_sketch = (
                str(work / "hgq.sk") if topology == "batch" else str(work / "hgr.sk")
            )
            marks = wall_of(
                [
                    str(hypergen),
                    "dist",
                    "-r",
                    str(work / "hgr.sk"),
                    "-q",
                    query_sketch,
                    "-o",
                    str(dist_out),
                    "-a",
                    "0",
                    "-t",
                    str(threads),
                ],
                samples,
                warmups,
            )
            rows = []
            for line in dist_out.read_text().splitlines():
                fields = line.split("\t")
                if len(fields) == 3:
                    try:
                        rows.append(
                            {
                                "query": staged_to_real.get(fields[1], fields[1]),
                                "reference": staged_to_real.get(fields[0], fields[0]),
                                "jaccard": None,
                                "ani": float(fields[2]),
                            }
                        )
                    except ValueError:
                        continue
            record_compare("hypergen", "cpu", marks, rows)
        if "skani" in selected:
            dist_out = work / "skani-dist.tsv"
            marks = wall_of(
                [
                    str(skani),
                    "dist",
                    "-q",
                    *file_args,
                    "-r",
                    *file_args,
                    "-o",
                    str(dist_out),
                    "-t",
                    str(threads),
                ],
                samples,
                warmups,
            )
            rows = []
            for line in dist_out.read_text().splitlines():
                if not line or line.startswith("#"):
                    continue
                fields = line.split()
                if len(fields) >= 4:
                    try:
                        rows.append(
                            {
                                "query": fields[1],
                                "reference": fields[0],
                                "jaccard": None,
                                "ani": float(fields[2]),
                            }
                        )
                    except ValueError:
                        continue
            record_compare("skani", "cpu", marks, rows)

        if "cuddl" in selected:
            import json as jsonlib5

            pipeline_bin = build / "benchmarks/cuddl-pipeline-benchmark"
            if not pipeline_bin.exists():
                raise typer.BadParameter(f"missing binary, build first: {pipeline_bin}")

            # One batched invocation; threshold zero refines every pair, so
            # the index path returns the same summaries as exhaustive search.
            cfg = work / "cuddl-compare.toml"
            compare_queries = query_list or references[:1]
            cfg.write_text(
                "reference = "
                + jsonlib5.dumps([str(path) for path in references])
                + "\nquery = "
                + jsonlib5.dumps([str(path) for path in compare_queries])
                + "\n"
            )
            rep = work / "cuddl-compare.json"
            run(
                [
                    str(pipeline_bin),
                    "--topology",
                    topology,
                    "--samples",
                    str(max(samples, 2)),
                    "--warmups",
                    str(warmups),
                    "--ingest",
                    "packed",
                    "--resident-bytes",
                    "0",
                    "--rows",
                    "compact",
                    "--index",
                    "sparse",
                    "--minimum-matches",
                    "0",
                    "--config",
                    str(cfg),
                    "--output",
                    str(rep),
                ]
            )
            payload = jsonlib5.loads(rep.read_text())
            pipe = next(
                m
                for m in payload["measurements"]
                if m["case"].get("measurement") == "pipeline"
            )
            query_paths = query_list if topology == "batch" else references
            rows = []
            for m in payload["measurements"]:
                if m["case"].get("measurement") != "match":
                    continue
                metrics = m.get("metrics", {})
                rows.append(
                    {
                        "query": str(query_paths[m["case"]["query_id"]]),
                        "reference": str(references[m["case"]["reference_id"]]),
                        "jaccard": None,
                        "ani": metrics.get("ani"),
                    }
                )
            query_wall = pipe["timings"]["query_output_wall"]
            record_compare(
                "cuddl", "gpu", [query_wall["median_ms"]] * max(samples, 1), rows
            )

        if "rabbitsketch" in selected:
            query_paths = query_list if topology == "batch" else references
            rows = []
            for m in rabbit_report["measurements"]:
                if m["case"].get("measurement") == "match":
                    metrics = m.get("metrics", {})
                    rows.append(
                        {
                            "query": str(query_paths[m["case"]["query_id"]]),
                            "reference": str(references[m["case"]["reference_id"]]),
                            "jaccard": metrics.get("jaccard"),
                            "ani": metrics.get("ani"),
                        }
                    )
            query_wall = rabbit_rows["timings"]["query_output_wall"]
            marks = [query_wall["median_ms"]] * max(samples, 1)
            record_compare("rabbitsketch", "FastKMV", marks, rows)

        if "cub-exact" in selected:
            rows = [
                {
                    "query": row["query"],
                    "reference": row["reference"],
                    "jaccard": row["jaccard"],
                    "ani": row["mash_ani"],
                }
                for row in jsonlib.loads(cub_rep.read_text())["pairs"]
            ]
            record_compare(
                "cub-exact", "gpu-exact", [cub_phases["compare"]["median_ms"]], rows
            )
            measurements[-1]["metrics"].update(autoscale_case)

        # SEARCH op: ranked retrieval per query; recall@k and top-1 hit rate
        # against the exact Jaccard ranking. Self-pairs never rank.
        query_set = query_list if topology == "batch" else references
        candidates = {
            str(q): sorted({str(r) for r in references} - {str(q)}) for q in query_set
        }
        n_cand = min(len(v) for v in candidates.values())
        if n_cand < 1:
            raise typer.BadParameter("SEARCH needs at least one candidate per query")
        k = min(recall_k, n_cand)

        def exact_jaccard(query: str, ref: str) -> float:
            row = oracle.get((query, ref), oracle.get((ref, query)))
            return row["jaccard"] if row is not None else 0.0

        exact_rank: dict[str, list[str]] = {}
        for query, refs in candidates.items():
            ranked = sorted(refs, key=lambda r: (-exact_jaccard(query, r), r))
            exact_rank[query] = ranked
        search_index_ms: dict[str, list[float]] = {}
        search_query_ms: dict[str, list[float]] = {}
        search_scores: dict[tuple[str, str, str], float] = {}

        if "cuddl" in selected:
            pipeline_bin = build / "benchmarks/cuddl-pipeline-benchmark"
            if not pipeline_bin.exists():
                raise typer.BadParameter(f"missing binary, build first: {pipeline_bin}")
            cfg = work / "cuddl-search.toml"
            import json as jsonlib6

            # The stage suite needs a query file even for all-to-all; match
            # rows stay triangular over references.
            search_queries = query_list or references[:1]
            cfg.write_text(
                "reference = "
                + jsonlib6.dumps([str(path) for path in references])
                + "\nquery = "
                + jsonlib6.dumps([str(path) for path in search_queries])
                + "\n"
            )
            rep = work / "cuddl-search.json"
            run(
                [
                    str(pipeline_bin),
                    "--topology",
                    topology,
                    "--samples",
                    str(max(samples, 2)),
                    "--warmups",
                    str(warmups),
                    "--ingest",
                    "packed",
                    "--resident-bytes",
                    "0",
                    "--rows",
                    "compact",
                    "--index",
                    "sparse",
                    "--minimum-matches",
                    str(min_matches),
                    "--config",
                    str(cfg),
                    "--output",
                    str(rep),
                ]
            )
            payload = jsonlib6.loads(rep.read_text())
            pipe = next(
                m
                for m in payload["measurements"]
                if m["case"].get("measurement") == "pipeline"
            )
            search_index_ms["cuddl"] = [pipe["timings"]["prepare_wall"]["median_ms"]]
            search_query_ms["cuddl"] = [
                pipe["timings"]["query_output_wall"]["median_ms"]
            ]
            for m in payload["measurements"]:
                if m["case"].get("measurement") == "match":
                    metrics = m.get("metrics", {})
                    if metrics.get("wkid") is None:
                        continue
                    search_scores[
                        (
                            "cuddl",
                            str(query_set[m["case"]["query_id"]]),
                            str(references[m["case"]["reference_id"]]),
                        )
                    ] = metrics["wkid"]

        if "skani" in selected:
            skdb = work / f"skdb{warmups + samples - 1}"
            search_out = work / "skani-search.tsv"
            marks = wall_of(
                [
                    str(skani),
                    "search",
                    "-d",
                    str(skdb),
                    "-o",
                    str(search_out),
                    "--min-af",
                    "0",
                    "-t",
                    str(threads),
                    *[str(q) for q in query_set],
                ],
                samples,
                warmups,
            )
            search_index_ms["skani"] = sketch_times["skani"]
            search_query_ms["skani"] = marks
            for line in search_out.read_text().splitlines():
                if not line or line.startswith("#"):
                    continue
                fields = line.split()
                if len(fields) >= 3:
                    try:
                        search_scores[("skani", fields[1], fields[0])] = float(
                            fields[2]
                        )
                    except ValueError:
                        continue

        # Exhaustive-rank tools reuse their COMPARE rows; sketches are the index.
        for tool, score_key in (
            ("hypergen", "ani"),
            ("dashing2", "jaccard"),
            ("rabbitsketch", "jaccard"),
            ("cub-exact", "jaccard"),
        ):
            if tool in selected:
                search_index_ms[tool] = sketch_times[tool]
        search_query_ms["hypergen"] = (
            [
                m["timings"]["wall"]["median_ms"]
                for m in measurements
                if m["implementation"]["name"] == "hypergen"
                and m["case"]["measurement"] == "micro-compare"
            ]
            if "hypergen" in selected
            else []
        )
        for tool, score_key in (
            ("dashing2", "jaccard"),
            ("rabbitsketch", "jaccard"),
            ("cub-exact", "jaccard"),
        ):
            if tool in selected:
                search_query_ms[tool] = [
                    m["timings"]["wall"]["median_ms"]
                    for m in measurements
                    if m["implementation"]["name"] == tool
                    and m["case"]["measurement"] == "micro-compare"
                ]
        if "hypergen" in selected:
            for row in pair_table:
                if row["tool"] == "hypergen" and row.get("ani") is not None:
                    search_scores[("hypergen", row["query"], row["reference"])] = row[
                        "ani"
                    ]
        for tool in ("dashing2", "rabbitsketch", "cub-exact"):
            if tool in selected:
                for row in pair_table:
                    if row["tool"] == tool and row.get("jaccard") is not None:
                        search_scores[(tool, row["query"], row["reference"])] = row[
                            "jaccard"
                        ]

        def record_search(tool: str, variant: str) -> None:
            recalls, top1 = [], 0
            counted = 0
            for query, refs in candidates.items():
                scored = [
                    (search_scores[(tool, query, ref)], ref)
                    for ref in refs
                    if (tool, query, ref) in search_scores
                ]
                if not scored:
                    continue
                ranked = [ref for _, ref in sorted(scored, key=lambda t: (-t[0], t[1]))]
                exact = [r for r in exact_rank[query] if r in {x for _, x in scored}]
                kk = min(k, len(exact))
                if kk < 1:
                    continue
                recalls.append(len(set(ranked[:kk]) & set(exact[:kk])) / kk)
                top1 += ranked[0] == exact[0]
                counted += 1
            measurements.append(
                {
                    "implementation": {"name": tool, "variant": variant},
                    "case": {
                        "measurement": "micro-search",
                        "topology": topology,
                        "queries": len(candidates),
                        "k": k,
                        "threads": threads,
                    },
                    "timings": {
                        "index_build": summarize(search_index_ms[tool]),
                        "query": summarize(search_query_ms[tool]),
                    },
                    "metrics": {
                        "recall_at_k": sum(recalls) / len(recalls) if recalls else 0.0,
                        "top1_rate": top1 / counted if counted else 0.0,
                        "queries_scored": counted,
                        "per_query_ms": (
                            statistics.median(search_query_ms[tool]) / max(counted, 1)
                        ),
                    },
                }
            )

        variants = {
            "cuddl": "gpu-indexed",
            "skani": "cpu",
            "hypergen": "cpu",
            "dashing2": "SetSketch",
            "rabbitsketch": "FastKMV",
            "cub-exact": "gpu-exact",
        }
        for tool in selected:
            if tool in search_index_ms and tool in search_query_ms:
                record_search(tool, variants[tool])

    datasets = dataset_entries("reference", references, 8)
    datasets.update(dataset_entries("query", query_list, 8))
    result = make_result(
        name="micro comparison",
        operation="micro",
        scope="end_to_end",
        datasets=datasets,
        measurements=measurements,
    )
    write_result(output, result)
    if pairs_out is not None:
        with open(pairs_out, "w") as stream:
            for row in pair_table:
                stream.write(
                    "\t".join(
                        str(row.get(k, ""))
                        for k in ("tool", "query", "reference", "jaccard", "ani")
                    )
                    + "\n"
                )
    header = f"{'tool':<14}{'op':<9}{'wall_ms':>10}{'per_unit_ms':>13}{'/exact':>10}{'/skani':>10}"
    typer.echo(header)
    for m in measurements:
        if m["case"]["measurement"] == "micro-search":
            continue
        metrics = m["metrics"]
        op = m["case"]["measurement"].replace("micro-", "")
        wall = m["timings"]["wall"]["median_ms"]
        unit = metrics.get("per_genome_ms", metrics.get("per_pair_ms", 0.0))
        typer.echo(
            f"{m['implementation']['name']:<14}{op:<9}{wall:>10.1f}{unit:>13.2f}"
            f"{metrics.get('jaccard_mae_vs_exact', float('nan')):>10.4f}"
            f"{metrics.get('ani_mae_vs_skani', float('nan')):>10.4f}"
        )
    typer.echo(
        f"{'tool':<14}{'op':<9}{'index_ms':>10}{'query_ms':>10}{'recall@k':>10}{'top1':>10}"
    )
    for m in measurements:
        if m["case"]["measurement"] != "micro-search":
            continue
        metrics = m["metrics"]
        typer.echo(
            f"{m['implementation']['name']:<14}{'search':<9}"
            f"{m['timings']['index_build']['median_ms']:>10.1f}"
            f"{m['timings']['query']['median_ms']:>10.1f}"
            f"{metrics.get('recall_at_k', float('nan')):>10.3f}"
            f"{metrics.get('top1_rate', float('nan')):>10.3f}"
        )


if __name__ == "__main__":
    APP()
