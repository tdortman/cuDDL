#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = ["jsonschema", "typer"]
# ///
"""Run HyperGen sketch + dist as a pipeline baseline into a v1 report.

Same role as the in-tree pipeline benchmarks: consume the runner's TOML
file list, time native sketch and search phases over repeated samples,
and emit the shared cuddl-benchmark/v1 envelope with operation=pipeline.
HyperGen runs exactly as its own CLI defines it; this script only stages
inputs, times phases, and converts the TSV output.
"""

import hashlib
import statistics
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Annotated

import tomllib
import typer
from benchmark_schema import make_result, write_result

ROOT = Path(__file__).resolve().parent.parent
APP = typer.Typer()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def dataset_entries(role: str, paths: list[Path], limit: int) -> dict:
    """Bounded per-file digests plus a manifest digest, mirroring host_result_json.hpp."""
    entries: dict = {}
    hashed = min(limit, len(paths)) if limit else len(paths)
    for index in range(hashed):
        entries[f"{role}_{index}"] = {
            "path": str(paths[index]),
            "sha256": sha256(paths[index]),
        }
    if hashed < len(paths):
        manifest = "".join(
            f"{paths[i].stat().st_size}\t{paths[i]}\n" for i in range(len(paths))
        )
        entries[f"{role}_manifest"] = {
            "path": f"{paths[0]} and {len(paths) - 1} more",
            "sha256": hashlib.sha256(manifest.encode()).hexdigest(),
        }
    return entries


def summarize(samples: list[float]) -> dict:
    ordered = sorted(samples)
    count = len(ordered)
    mid = count // 2
    median = ordered[mid] if count % 2 else (ordered[mid - 1] + ordered[mid]) / 2
    result = {
        "samples": count,
        "median_ms": median,
        "min_ms": ordered[0],
        "max_ms": ordered[-1],
        "source": "steady_clock_cpu_wall",
    }
    if count > 1:
        mean = statistics.fmean(ordered)
        result["p95_ms"] = ordered[min(count - 1, int(0.95 * count))]
        if mean > 0:
            result["relative_stddev_percent"] = statistics.pstdev(ordered) / mean * 100
    return result


def stage(link_dir: Path, files: list[Path]) -> dict[str, str]:
    """Symlink inputs for HyperGen's folder reader; map staged back to real."""
    link_dir.mkdir(parents=True, exist_ok=True)
    mapping = {}
    for path in files:
        link = link_dir / path.name
        if link.is_symlink() or link.exists():
            link.unlink()
        link.symlink_to(path.resolve())
        mapping[str(link)] = str(path.resolve())
    return mapping


def run(cmd: list[str]) -> None:
    subprocess.run(cmd, cwd=ROOT, check=True, capture_output=True)


def skani_truth(
    skani_bin: Path | None, genomes: list[Path], work: Path
) -> dict[tuple[str, str], tuple[float, float]]:
    """Map (query, reference) path pair to (ANI, aligned fraction)."""
    if skani_bin is None:
        return {}
    out = work / "skani.tsv"
    run([str(skani_bin), "triangle", *(str(p) for p in genomes), "-o", str(out)])
    rows = out.read_text().splitlines()
    names = [rows[i].split("\t")[0] for i in range(1, len(rows))]
    table: dict[tuple[str, str], tuple[float, float]] = {}
    af_rows = (work / "skani.tsv.af").read_text().splitlines()
    for i in range(len(names)):
        ani_vals = [float(x) for x in rows[1 + i].split("\t")[1:]]
        af_vals = [float(x) for x in af_rows[1 + i].split("\t")[1:]]

        def get(vals: list[float], a: int, b: int) -> float:
            if a < len(vals):
                return vals[a]

        for j in range(i):
            left = str(Path(names[i]).resolve())
            right = str(Path(names[j]).resolve())
            table[(left, right)] = (get(ani_vals, j, i), get(af_vals, j, i))
            table[(right, left)] = table[(left, right)]
    return table


@APP.command()
def main(
    config: Annotated[Path, typer.Option(exists=True)],
    output: Annotated[Path, typer.Option()],
    topology: Annotated[str, typer.Option()] = "batch",
    samples: Annotated[int, typer.Option(min=1)] = 20,
    warmups: Annotated[int, typer.Option(min=0)] = 3,
    k: Annotated[int, typer.Option(min=1, max=31)] = 25,
    scaled: Annotated[int, typer.Option(min=1)] = 1500,
    hv_dim: Annotated[int, typer.Option(min=64)] = 4096,
    device: Annotated[str, typer.Option()] = "cpu",
    threads: Annotated[int | None, typer.Option(min=1)] = None,
    hypergen_bin: Annotated[Path | None, typer.Option()] = None,
    skani_bin: Annotated[Path | None, typer.Option(exists=True)] = None,
    match_rows: Annotated[int, typer.Option(min=0)] = 20000,
    dataset_hashes: Annotated[int, typer.Option(min=0)] = 8,
) -> None:
    """Sketch references and queries, search, and write a pipeline report."""
    if hypergen_bin is None:
        candidate = ROOT / "build/subprojects/hypergen/hyper-gen"
        if not candidate.exists():
            raise typer.BadParameter(
                "hyper-gen binary not found; run "
                "`meson compile -C build subprojects/hypergen/hyper-gen`"
            )
        hypergen_bin = candidate
    if skani_bin is None:
        for name in (
            "build/subprojects/skani/skani",
            "build/subprojects/skani/cargo-target/release/skani",
        ):
            if (ROOT / name).exists():
                skani_bin = ROOT / name
                break
    if topology not in ("batch", "all-to-all"):
        raise typer.BadParameter("topology must be batch or all-to-all")
    if device not in ("cpu", "gpu"):
        raise typer.BadParameter("device must be cpu or gpu")
    with open(config, "rb") as stream:
        file_list = tomllib.load(stream)
    references = [Path(p) for p in file_list["reference"]]
    queries = [Path(p) for p in file_list.get("query", [])]
    if topology == "batch" and (not references or not queries):
        raise typer.BadParameter("batch requires nonempty reference and query lists")
    if topology == "all-to-all" and len(references) < 2:
        raise typer.BadParameter("all-to-all requires at least two references")

    version = subprocess.run(
        [str(hypergen_bin), "--version"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    thread_args = ["-t", str(threads)] if threads is not None else []
    sketch_base = [
        str(hypergen_bin),
        "sketch",
        "-D",
        device,
        "-k",
        str(k),
        "-s",
        str(scaled),
        "-d",
        str(hv_dim),
        *thread_args,
    ]

    end_to_end: list[float] = []
    prepare: list[float] = []
    query_out: list[float] = []
    teardown: list[float] = []
    sketch_bytes = 0
    pairs: list[tuple[str, str, float]] = []
    with tempfile.TemporaryDirectory(prefix="hypergen-pipeline-") as tmp:
        work = Path(tmp)
        staged = stage(work / "refs", references)
        if topology == "batch":
            staged.update(stage(work / "queries", queries))
        genomes = references + [q for q in queries if q not in references]
        truth = skani_truth(skani_bin, genomes, work)
        for rep in range(warmups + samples):
            rep_dir = work / f"rep{rep}"
            rep_dir.mkdir()
            tick = time.perf_counter()
            run([*sketch_base, "-p", str(work / "refs"), "-o", str(rep_dir / "r.sk")])
            if topology == "batch":
                run(
                    [
                        *sketch_base,
                        "-p",
                        str(work / "queries"),
                        "-o",
                        str(rep_dir / "q.sk"),
                    ]
                )
            sketch_done = time.perf_counter()
            dist_out = rep_dir / "pairs.ani"
            if topology == "batch":
                run(
                    [
                        str(hypergen_bin),
                        "dist",
                        "-r",
                        str(rep_dir / "r.sk"),
                        "-q",
                        str(rep_dir / "q.sk"),
                        "-o",
                        str(dist_out),
                        "-a",
                        "0",
                        *thread_args,
                    ]
                )
            else:
                run(
                    [
                        str(hypergen_bin),
                        "dist",
                        "-r",
                        str(rep_dir / "r.sk"),
                        "-q",
                        str(rep_dir / "r.sk"),
                        "-o",
                        str(dist_out),
                        "-a",
                        "0",
                        *thread_args,
                    ]
                )
            search_done = time.perf_counter()
            for path in rep_dir.glob("*"):
                path.unlink()
            rep_dir.rmdir()
            done = time.perf_counter()
            if rep >= warmups:
                prepare.append((sketch_done - tick) * 1000)
                query_out.append((search_done - sketch_done) * 1000)
                teardown.append((done - search_done) * 1000)
                end_to_end.append((done - tick) * 1000)
        run([*sketch_base, "-p", str(work / "refs"), "-o", str(work / "r.sk")])
        if topology == "batch":
            run([*sketch_base, "-p", str(work / "queries"), "-o", str(work / "q.sk")])
            run(
                [
                    str(hypergen_bin),
                    "dist",
                    "-r",
                    str(work / "r.sk"),
                    "-q",
                    str(work / "q.sk"),
                    "-o",
                    str(work / "pairs.ani"),
                    "-a",
                    "0",
                    *thread_args,
                ]
            )
        else:
            run(
                [
                    str(hypergen_bin),
                    "dist",
                    "-r",
                    str(work / "r.sk"),
                    "-q",
                    str(work / "r.sk"),
                    "-o",
                    str(work / "pairs.ani"),
                    "-a",
                    "0",
                    *thread_args,
                ]
            )
        sketch_bytes = (work / "r.sk").stat().st_size
        if topology == "batch":
            sketch_bytes += (work / "q.sk").stat().st_size
        pairs = [
            (staged[fields[0]], staged[fields[1]], float(fields[2]))
            for fields in (
                line.split("\t")
                for line in (work / "pairs.ani").read_text().splitlines()
                if line
            )
        ]

    errors: list[float] = []
    reported = 0
    rows_out: list[dict] = []
    stride = max(1, len(pairs) // match_rows) if match_rows else 1
    for index, (left, right, ani) in enumerate(pairs):
        truth_pair = truth.get((left, right))
        skani_ani = truth_pair[0] if truth_pair else None
        if truth_pair and truth_pair[0] > 0:
            reported += 1
            errors.append(abs(ani - truth_pair[0]))
        if match_rows and index % stride != 0 and index != len(pairs) - 1:
            continue
        rows_out.append(
            {
                "query": Path(right).name,
                "reference": Path(left).name,
                "hypergen_ani": ani,
                "skani_ani": skani_ani,
            }
        )
    metrics: dict = {
        "matches_total": len(pairs),
        "matches_emitted": len(rows_out),
        "skani_reported_pairs": reported,
        "sketch_bytes": sketch_bytes,
    }
    if errors:
        metrics["ani_mae_vs_skani"] = sum(errors) / len(errors)
        metrics["ani_max_err_vs_skani"] = max(errors)
    datasets = dataset_entries("reference", references, dataset_hashes)
    datasets.update(dataset_entries("query", queries, dataset_hashes))
    result = make_result(
        name="hypergen pipeline",
        operation="pipeline",
        scope="end_to_end",
        datasets=datasets,
        measurements=[
            {
                "implementation": {
                    "name": "hypergen",
                    "variant": device,
                    "version": version,
                },
                "case": {
                    "measurement": "pipeline",
                    "topology": topology,
                    "k": k,
                    "references": len(references),
                    "queries": 0 if topology == "all-to-all" else len(queries),
                    "input_cache": "fastx_files",
                    "ingest": "sequence",
                    "sketch_size": hv_dim,
                    "scaled": scaled,
                    "samples": samples,
                    "warmups": warmups,
                },
                "metrics": metrics,
                "timings": {
                    "end_to_end_wall": summarize(end_to_end),
                    "prepare_wall": summarize(prepare),
                    "query_output_wall": summarize(query_out),
                    "teardown_wall": summarize(teardown),
                },
                "memory_bytes": {"sketch_files": sketch_bytes},
            }
        ],
    )
    write_result(output, result)
    typer.echo(
        f"wrote {output}: {len(pairs)} pairs, "
        f"end-to-end median {summarize(end_to_end)['median_ms']:.1f} ms"
    )


if __name__ == "__main__":
    APP()
