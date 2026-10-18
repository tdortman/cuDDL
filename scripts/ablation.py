#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = ["typer"]
# ///
"""Ablation study: what each optimisation contributes, on its own stage and end to end.

Every configuration turns off one optimisation, through `CUDDL_ABLATE`
(include/cuddl/detail/ablation.hpp), a command-line option, an environment variable or a
second build, and runs next to the configuration it is compared with. Repetitions interleave
the configurations, so load on a shared node hits them alike. Each run appends one JSON line to
OUTPUT, and every suite first writes the ablations it measures; plot_ablation.py ranks the
optimisations from these lines.

`build` times cuddl-build-reference-db as a whole process: the loading and sketching stage.
With --uncached it evicts the genomes from the page cache before every run (posix_fadvise
DONTNEED, which also works on Lustre). `search` runs cuddl-multi-gpu-all-to-all-benchmark and
records the median exhaustive search with results left on the GPUs, with every result
downloaded, with the database chunked instead of replicated, and the threshold search.
`e2e` runs cuddl-distributed-all-to-all-benchmark, from genome files to written results, and
records its wall and phase times. Optimisations no other suite isolates are also compared on
the phase they target (metric `phase:NAME`), which plot_ablation.py shows as their stage. The
switches need a build configured with -Dablations=true; every suite checks BUILD_DIR for it.

    uv run scripts/ablation.py build GENOMES --build-dir B -o results/ablation-local.jsonl
    uv run scripts/ablation.py search DB --build-dir B --launcher "mpirun -n 8" --ranks 8 -o ...
    uv run scripts/ablation.py e2e GENOMES --build-dir B --launcher "mpirun -n 8" -o ...
"""

import json
import os
import shlex
import socket
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Annotated

import typer

app = typer.Typer(add_completion=False)

OWN_GPU = "uv run --offline scripts/own_gpu.py"


@dataclass
class Config:
    """One configuration: what it turns off and how its command line or environment differs."""

    name: str
    ablate: list[str] = field(default_factory=list)
    args: list[str] = field(default_factory=list)
    env: dict[str, str] = field(default_factory=dict)
    binary: Path | None = None
    own_gpu: bool = True


@dataclass
class Ablation:
    """An optimisation, the configuration without it, the one it is compared with, and the
    recorded time that compares them."""

    optimisation: str
    without: str
    reference: str
    metric: str = "seconds"


def check_build(build_dir: Path) -> None:
    options = json.loads(
        (build_dir / "meson-info" / "intro-buildoptions.json").read_text()
    )
    if not any(o["name"] == "ablations" and o["value"] is True for o in options):
        raise typer.BadParameter(f"{build_dir} is not configured with -Dablations=true")


def evict(paths: list[Path]) -> None:
    def drop(path: Path) -> None:
        fd = os.open(path, os.O_RDONLY)
        try:
            os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
        finally:
            os.close(fd)

    with ThreadPoolExecutor(64) as pool:
        list(pool.map(drop, paths))


def warm(paths: list[Path]) -> None:
    def read(path: Path) -> None:
        with open(path, "rb") as f:
            while f.read(4 << 20):
                pass

    with ThreadPoolExecutor(32) as pool:
        list(pool.map(read, paths))


def genome_paths(genomes: Path) -> list[Path]:
    return sorted(p.resolve() for p in genomes.rglob("*") if p.is_file())


def environment(config: Config) -> dict[str, str]:
    return os.environ | config.env | {"CUDDL_ABLATE": ",".join(config.ablate)}


def launch(launcher: str, config: Config, own_gpu: str) -> list[str]:
    """The launcher, forwarding the configuration's environment to every rank, and the own-GPU
    wrapper. Open MPI binds each rank to one core unless told otherwise, which leaves a rank's
    loaders sharing that core: 10,000 genomes took 35.7 s to sketch with `cpu` instead of 6.3 s."""
    prefix = shlex.split(launcher)
    if prefix and Path(prefix[0]).name == "mpirun":
        if "--bind-to" not in prefix:
            prefix += ["--bind-to", "none"]
        for name in ["CUDDL_ABLATE", *config.env]:
            prefix += ["-x", name]
    if config.own_gpu and own_gpu:
        prefix += shlex.split(own_gpu)
    return prefix


def record(output: Path, line: dict) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "a") as f:
        f.write(json.dumps(line) + "\n")
    print(json.dumps(line), flush=True)


def select(
    configs: list[Config], ablations: list[Ablation], only: list[str] | None
) -> tuple[list[Config], list[Ablation]]:
    if only:
        unknown = set(only) - {a.optimisation for a in ablations}
        if unknown:
            raise typer.BadParameter(f"unknown optimisations: {sorted(unknown)}")
        ablations = [a for a in ablations if a.optimisation in only]
    wanted = {a.without for a in ablations} | {a.reference for a in ablations}
    return [c for c in configs if c.name in wanted], ablations


def run_suite(
    suite: str,
    label: str,
    output: Path,
    configs: list[Config],
    ablations: list[Ablation],
    reps: int,
    run,
) -> None:
    label = label or socket.gethostname()
    record(
        output,
        {"suite": suite, "label": label, "ablations": [a.__dict__ for a in ablations]},
    )
    for rep in range(reps):
        for config in configs:
            started = time.time()
            metrics = run(config)
            record(
                output,
                {
                    "suite": suite,
                    "label": label,
                    "config": config.name,
                    "ablate": config.ablate,
                    "args": config.args,
                    "env": config.env,
                    "rep": rep,
                    "started": started,
                }
                | metrics,
            )


@app.command()
def build(
    genomes: Annotated[Path, typer.Argument(exists=True, help="Genome directory")],
    build_dir: Annotated[
        Path, typer.Option(help="Build configured with -Dablations=true")
    ],
    output: Annotated[
        Path, typer.Option("-o", "--output", help="JSON lines to append to")
    ],
    workers: Annotated[int, typer.Option(help="Loaders")] = 16,
    reps: Annotated[int, typer.Option(help="Repetitions of every configuration")] = 3,
    uncached: Annotated[
        bool,
        typer.Option(help="Evict the genomes from the page cache before every run"),
    ] = False,
    engine: Annotated[
        bool,
        typer.Option(
            help="Compare gpu with gpu-cores (GPUs with a decompression engine)"
        ),
    ] = False,
    only: Annotated[list[str] | None, typer.Option(help="Optimisations to run")] = None,
    label: Annotated[str, typer.Option(help="Machine label for the results")] = "",
    scratch: Annotated[
        Path | None,
        typer.Option(help="Directory for the database [default: a temporary one]"),
    ] = None,
) -> None:
    check_build(build_dir)
    gpu, cpu = ["--decompression", "gpu"], ["--decompression", "cpu"]
    configs = [
        Config("gpu", args=gpu),
        Config("cpu", args=cpu),
        Config("gpu co-scheduling", ["co-scheduling"], gpu),
        Config("gpu read-depth", ["read-depth"], gpu),
        Config("cpu read-depth", ["read-depth"], cpu),
        Config("gpu probe-depth", ["probe-depth"], gpu),
        Config("gpu input-lanes", ["input-lanes"], gpu),
        Config("cpu pinned-reservation", ["pinned-reservation"], cpu),
        Config("cpu base-packing", ["base-packing"], cpu),
        Config("gpu kmer-encoding", ["kmer-encoding"], gpu),
        Config("cpu kmer-encoding", ["kmer-encoding"], cpu),
    ]
    # Each ablation compares the time of the stage its optimisation targets: the whole build
    # for loading and scheduling, the sketch kernels' GPU time for the k-mer encoder.
    ablations = [
        Ablation("GPU decompression (nvCOMP)", "cpu", "gpu"),
        Ablation("GPU/host co-scheduling", "gpu co-scheduling", "gpu"),
        Ablation("Read depth (gpu)", "gpu read-depth", "gpu"),
        Ablation("Read depth (cpu)", "cpu read-depth", "cpu"),
        Ablation("Probe depth", "gpu probe-depth", "gpu"),
        Ablation("Input-sized lanes", "gpu input-lanes", "gpu"),
        Ablation("Pre-pinned packed buffers", "cpu pinned-reservation", "cpu"),
        Ablation("Two-bit base packing", "cpu base-packing", "cpu"),
        Ablation(
            "Branchless k-mer encoding (gpu)",
            "gpu kmer-encoding",
            "gpu",
            "kernel_seconds",
        ),
        Ablation(
            "Branchless k-mer encoding (cpu)",
            "cpu kmer-encoding",
            "cpu",
            "kernel_seconds",
        ),
    ]
    if engine:
        configs.append(Config("gpu-cores", args=["--decompression", "gpu-cores"]))
        ablations.append(Ablation("Decompression engine", "gpu-cores", "gpu"))
    configs, ablations = select(configs, ablations, only)

    paths = genome_paths(genomes)
    input_bytes = sum(p.stat().st_size for p in paths)
    if not uncached:
        warm(paths)
    with tempfile.TemporaryDirectory(dir=scratch) as directory:
        database = Path(directory) / "db.cuddl"
        kernels = Path(directory) / "kernels.json"

        def run(config: Config) -> dict:
            if uncached:
                evict(paths)
            binary = (
                config.binary or build_dir / "examples" / "cuddl-build-reference-db"
            )
            command = [
                str(binary),
                str(genomes),
                "-o",
                str(database),
                "--workers",
                str(workers),
            ]
            # The CLI writes the GPU time of its sketch kernels here.
            env = environment(config) | {"CUDDL_RESIDENT_TIMINGS": str(kernels)}
            start = time.perf_counter()
            subprocess.run(
                command + config.args, check=True, env=env, stdout=subprocess.DEVNULL
            )
            seconds = time.perf_counter() - start
            return {
                "seconds": seconds,
                "kernel_seconds": json.loads(kernels.read_text())["resident_ms"] / 1000,
                "genomes": len(paths),
                "input_bytes": input_bytes,
                "workers": workers,
                "uncached": uncached,
            }

        run_suite("build", label, output, configs, ablations, reps, run)


def nvbench_seconds(report: Path) -> dict[str, float]:
    """Mean CPU seconds of every state in an NVBench JSON report, by benchmark name."""
    seconds = {}
    for benchmark in json.loads(report.read_text())["benchmarks"]:
        for state in benchmark["states"]:
            if state.get("is_skipped"):
                raise RuntimeError(
                    f"{benchmark['name']} skipped: {state.get('skip_reason')}"
                )
            for summary in state["summaries"]:
                if summary["tag"] == "nv/cpu_only/time/cpu/mean":
                    value = next(d for d in summary["data"] if d["name"] == "value")[
                        "value"
                    ]
                    seconds[benchmark["name"]] = float(value)
    return seconds


@app.command()
def micro(
    genomes: Annotated[Path, typer.Argument(exists=True, help="Genome directory")],
    build_dir: Annotated[
        Path, typer.Option(help="Build configured with -Dablations=true")
    ],
    output: Annotated[
        Path, typer.Option("-o", "--output", help="JSON lines to append to")
    ],
    stock_build: Annotated[
        Path | None,
        typer.Option(help="Ablation build compiled against unpatched libdeflate"),
    ] = None,
    reps: Annotated[int, typer.Option(help="Repetitions of every configuration")] = 3,
    only: Annotated[list[str] | None, typer.Option(help="Optimisations to run")] = None,
    label: Annotated[str, typer.Option(help="Machine label for the results")] = "",
) -> None:
    """Host stages alone through cuddl-ablation-micro-benchmark, one thread each."""
    check_build(build_dir)
    configs = [Config("micro"), Config("micro simd-parsing", ["simd-parsing"])]
    configs.append(Config("micro pinned-pool", ["pinned-pool"]))
    ablations = [
        Ablation("SIMD FASTX parsing", "micro simd-parsing", "micro", "pack_seconds"),
        Ablation(
            "Device-only pinned pools", "micro pinned-pool", "micro", "pinned_seconds"
        ),
    ]
    if stock_build is not None:
        check_build(stock_build)
        binary = stock_build / "benchmarks" / "cuddl-ablation-micro-benchmark"
        configs.append(Config("micro stock-libdeflate", binary=binary))
        ablations.append(
            Ablation(
                "DNA-tuned libdeflate",
                "micro stock-libdeflate",
                "micro",
                "inflate_seconds",
            )
        )
    configs, ablations = select(configs, ablations, only)
    warm(genome_paths(genomes))
    with tempfile.TemporaryDirectory() as directory:
        report = Path(directory) / "micro.json"

        def run(config: Config) -> dict:
            binary = (
                config.binary
                or build_dir / "benchmarks" / "cuddl-ablation-micro-benchmark"
            )
            env = environment(config) | {
                "CUDDL_ABLATION_GENOMES": str(genomes.resolve())
            }
            subprocess.run(
                [str(binary), "--json", str(report)],
                check=True,
                env=env,
                stdout=subprocess.DEVNULL,
            )
            return {f"{name}_seconds": s for name, s in nvbench_seconds(report).items()}

        run_suite("micro", label, output, configs, ablations, reps, run)


@app.command()
def search(
    database: Annotated[Path, typer.Argument(exists=True, help="Database file")],
    build_dir: Annotated[
        Path, typer.Option(help="Build configured with -Dablations=true")
    ],
    output: Annotated[
        Path, typer.Option("-o", "--output", help="JSON lines to append to")
    ],
    write_dir: Annotated[
        Path, typer.Option(help="Node-local directory for threshold results")
    ],
    launcher: Annotated[
        str, typer.Option(help="MPI launcher with its rank count")
    ] = "",
    ranks: Annotated[int, typer.Option(help="Ranks the launcher starts")] = 1,
    minimum: Annotated[
        int, typer.Option(help="--minimum-matches of the threshold search")
    ] = 20,
    reps: Annotated[int, typer.Option(help="Repetitions of every configuration")] = 1,
    samples: Annotated[int, typer.Option(help="Timed searches per run")] = 10,
    only: Annotated[list[str] | None, typer.Option(help="Optimisations to run")] = None,
    label: Annotated[str, typer.Option(help="Machine label for the results")] = "",
) -> None:
    check_build(build_dir)
    # The search benchmark picks GPUs by PCIe topology itself, so every rank sees every GPU.
    exhaustive = ["--consume", "device"]
    downloads = ["--consume", "host"]
    threshold = ["--consume", "device", "--thresholds", str(minimum)]
    configs = [
        Config("exhaustive", args=exhaustive, own_gpu=False),
        Config("exhaustive deltas", ["deltas"], exhaustive, own_gpu=False),
        Config("exhaustive bit-slicing", ["bit-slicing"], exhaustive, own_gpu=False),
        Config("exhaustive lop3-kernel", ["lop3-kernel"], exhaustive, own_gpu=False),
        Config("exhaustive tile-queue", ["tile-queue"], exhaustive, own_gpu=False),
        Config(
            "exhaustive progress-thread", ["progress-thread"], exhaustive, own_gpu=False
        ),
        Config(
            "exhaustive rcache",
            args=exhaustive,
            env={"UCX_RCACHE_ENABLE": "y"},
            own_gpu=False,
        ),
        Config("chunked", args=["--consume", "chunked"], own_gpu=False),
        Config("downloads", args=downloads, own_gpu=False),
        Config("downloads numa-pool", ["numa-pool"], downloads, own_gpu=False),
        Config("downloads pinned-pool", ["pinned-pool"], downloads, own_gpu=False),
        Config("threshold", args=threshold, own_gpu=False),
        Config(
            "threshold inverted-index", ["inverted-index"], threshold, own_gpu=False
        ),
        Config(
            "threshold posting-bitmaps", ["posting-bitmaps"], threshold, own_gpu=False
        ),
        Config(
            "threshold index-fallback", ["index-fallback"], threshold, own_gpu=False
        ),
        Config(
            "threshold match-filter",
            ["match-filter", "inverted-index"],
            threshold,
            own_gpu=False,
        ),
        Config("threshold tile-queue", ["tile-queue"], threshold, own_gpu=False),
    ]
    ablations = [
        Ablation("Near-duplicate deltas", "exhaustive deltas", "exhaustive"),
        Ablation("Bit-sliced scores", "exhaustive bit-slicing", "exhaustive"),
        Ablation("LOP3 plane kernel", "exhaustive lop3-kernel", "exhaustive"),
        Ablation("Shared tile queue", "exhaustive tile-queue", "exhaustive"),
        Ablation("MPI progress thread", "exhaustive progress-thread", "exhaustive"),
        Ablation("UCX registration cache off", "exhaustive rcache", "exhaustive"),
        Ablation("Replicated database", "chunked", "exhaustive"),
        Ablation("NUMA-local download pools", "downloads numa-pool", "downloads"),
        Ablation("Device-only pinned pools", "downloads pinned-pool", "downloads"),
        Ablation("Inverted index", "threshold inverted-index", "threshold"),
        Ablation("Posting bitmaps", "threshold posting-bitmaps", "threshold"),
        Ablation("Index fallback to planes", "threshold index-fallback", "threshold"),
        Ablation(
            "Low-plane match filter",
            "threshold match-filter",
            "threshold inverted-index",
        ),
        Ablation("Shared tile queue (threshold)", "threshold tile-queue", "threshold"),
    ]
    configs, ablations = select(configs, ablations, only)
    binary = build_dir / "benchmarks" / "cuddl-multi-gpu-all-to-all-benchmark"
    with tempfile.TemporaryDirectory() as directory:
        report_path = Path(directory) / "report.json"

        def run(config: Config) -> dict:
            command = [
                *launch(launcher, config, ""),
                str(binary),
                str(database),
                "--ranks",
                str(ranks),
                "--samples",
                str(samples),
                "--warmups",
                "2",
                "--skip-verify",
                "--write-dir",
                str(write_dir),
                *config.args,
                "-o",
                str(report_path),
            ]
            subprocess.run(
                command, check=True, env=environment(config), stdout=subprocess.DEVNULL
            )
            report = json.loads(report_path.read_text())
            entry = max(report["runs"], key=lambda r: r["ranks"])
            consumer = next(c for c in ("device", "host", "chunked") if c in entry)
            sweep = report.get("threshold_sweep", [])
            seconds = sweep[0]["search"]["median_ms"] / 1000 if sweep else None
            return {
                "ranks": entry["ranks"],
                "references": entry["references"],
                "pairs": entry["pairs"],
                "gpu": report["gpu"],
                "seconds": seconds or entry[consumer]["median_ms"] / 1000,
                "pairs_emitted": sweep[0]["pairs_emitted"] if sweep else None,
            }

        run_suite("search", label, output, configs, ablations, reps, run)


@app.command()
def e2e(
    genomes: Annotated[Path, typer.Argument(exists=True, help="Genome directory")],
    build_dir: Annotated[
        Path, typer.Option(help="Build configured with -Dablations=true")
    ],
    output: Annotated[
        Path, typer.Option("-o", "--output", help="JSON lines to append to")
    ],
    output_dir: Annotated[
        Path, typer.Option(help="Node-local directory for the result files")
    ],
    launcher: Annotated[
        str, typer.Option(help="MPI launcher with its rank count")
    ] = "",
    own_gpu: Annotated[
        str, typer.Option(help="Wrapper giving each rank only its GPU")
    ] = OWN_GPU,
    decompression: Annotated[
        str, typer.Option(help="Backend of the runs not about decompression")
    ] = "automatic",
    workers: Annotated[int, typer.Option(help="Loaders per rank")] = 16,
    minimum: Annotated[
        int, typer.Option(help="--minimum-matches of the threshold runs")
    ] = 20,
    reps: Annotated[int, typer.Option(help="Repetitions of every configuration")] = 3,
    stock_build: Annotated[
        Path | None,
        typer.Option(help="Ablation build compiled against unpatched libdeflate"),
    ] = None,
    only: Annotated[list[str] | None, typer.Option(help="Optimisations to run")] = None,
    label: Annotated[str, typer.Option(help="Machine label for the results")] = "",
) -> None:
    check_build(build_dir)
    threshold = ["--minimum-matches", str(minimum), "--decompression", decompression]
    every = ["--minimum-matches", "0", "--decompression", decompression]
    gpu = ["--minimum-matches", str(minimum), "--decompression", "gpu"]
    cpu = ["--minimum-matches", str(minimum), "--decompression", "cpu"]
    configs = [
        Config("threshold", args=threshold),
        Config("gpu", args=gpu),
        Config("cpu", args=cpu),
        Config("every", args=every),
        Config("all-gpus", args=threshold, own_gpu=False),
    ]
    ablations = [
        Ablation("GPU decompression (nvCOMP)", "cpu", "gpu"),
        Ablation("Index filtering on the GPU", "every", "threshold"),
        Ablation("One visible GPU per rank", "all-gpus", "threshold"),
    ]

    def add(
        optimisation: str, ablate: list[str], base: str, env: dict | None = None
    ) -> None:
        reference = next(c for c in configs if c.name == base)
        name = f"{base} {'+'.join(ablate) or 'env'}"
        configs.append(
            Config(name, ablate, reference.args, env or {}, own_gpu=reference.own_gpu)
        )
        ablations.append(Ablation(optimisation, name, base))

    for optimisation, ablate, base in [
        ("GPU/host co-scheduling", ["co-scheduling"], "gpu"),
        ("Read depth (gpu)", ["read-depth"], "gpu"),
        ("Read depth (cpu)", ["read-depth"], "cpu"),
        ("Probe depth", ["probe-depth"], "gpu"),
        ("Input-sized lanes", ["input-lanes"], "gpu"),
        ("Pre-pinned packed buffers", ["pinned-reservation"], "cpu"),
        ("SIMD FASTX parsing", ["simd-parsing"], "cpu"),
        ("Two-bit base packing", ["base-packing"], "cpu"),
        ("Branchless k-mer encoding (gpu)", ["kmer-encoding"], "gpu"),
        ("Branchless k-mer encoding (cpu)", ["kmer-encoding"], "cpu"),
        ("Device-only pinned pools", ["pinned-pool"], "all-gpus"),
        ("NUMA-local download pools", ["numa-pool"], "every"),
        ("Near-duplicate deltas", ["deltas"], "every"),
        ("Bit-sliced scores", ["bit-slicing"], "every"),
        ("LOP3 plane kernel", ["lop3-kernel"], "every"),
        ("Shared tile queue", ["tile-queue"], "every"),
        ("MPI progress thread", ["progress-thread"], "every"),
        ("Inverted index", ["inverted-index"], "threshold"),
        ("Posting bitmaps", ["posting-bitmaps"], "threshold"),
        ("Index fallback to planes", ["index-fallback"], "threshold"),
    ]:
        add(optimisation, ablate, base)
    add("UCX registration cache off", [], "threshold", {"UCX_RCACHE_ENABLE": "y"})
    configs += [
        Config("threshold match-filter", ["inverted-index"], threshold),
        Config(
            "threshold no-match-filter", ["inverted-index", "match-filter"], threshold
        ),
        Config("every counts", args=[*every, "--records", "counts"]),
        Config("every one-stream", args=[*every, "--write-streams", "1"]),
    ]
    ablations += [
        Ablation(
            "Low-plane match filter",
            "threshold no-match-filter",
            "threshold match-filter",
        ),
        Ablation("Similarity records on the GPU", "every counts", "every"),
        Ablation(
            "Similarity records on the GPU", "every counts", "every", "phase:write"
        ),
        Ablation("Index filtering on the GPU", "every", "threshold", "phase:search"),
        Ablation("Parallel result streams", "every one-stream", "every"),
        Ablation("Parallel result streams", "every one-stream", "every", "phase:write"),
    ]
    if stock_build is not None:
        check_build(stock_build)
        binary = stock_build / "benchmarks" / "cuddl-distributed-all-to-all-benchmark"
        configs.append(Config("cpu stock-libdeflate", args=cpu, binary=binary))
        ablations.append(
            Ablation("DNA-tuned libdeflate", "cpu stock-libdeflate", "cpu")
        )
    configs, ablations = select(configs, ablations, only)
    warm(genome_paths(genomes))
    with tempfile.TemporaryDirectory() as directory:
        report_path = Path(directory) / "report.json"

        def run(config: Config) -> dict:
            binary = (
                config.binary
                or build_dir / "benchmarks" / "cuddl-distributed-all-to-all-benchmark"
            )
            command = [
                *launch(launcher, config, own_gpu),
                str(binary),
                str(genomes),
                "--workers",
                str(workers),
                "--output-dir",
                str(output_dir),
                *config.args,
                "-o",
                str(report_path),
            ]
            subprocess.run(
                command, check=True, env=environment(config), stdout=subprocess.DEVNULL
            )
            report = json.loads(report_path.read_text())
            return {
                "ranks": report["ranks"],
                "nodes": report["nodes"],
                "genomes": report["genomes"],
                "gpu": report["gpu"],
                "seconds": report["wall_ms"] / 1000,
                "phases_s": {k: v / 1000 for k, v in report["phases_wall_ms"].items()},
                "pairs": report["pairs_total"],
                "bytes_written": report["bytes_written_total"],
            }

        run_suite("e2e", label, output, configs, ablations, reps, run)


if __name__ == "__main__":
    sys.exit(app())
