#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = ["jsonschema", "typer"]
# ///
"""Micro-benchmarks over shared fused operations (SKETCH, COMPARE, SEARCH).

Each tool runs its native fused path and reports wall time plus native
outputs; accuracy joins against exact oracles afterwards unless --performance-only is set:

- SKETCH: FASTA files to queryable sketch state. CLI tools time their
  sketch command; cuDDL times reference-database build; RabbitSketch
  times pipeline prepare; cub-exact times its device sketch phase.
- COMPARE: pairs to similarity rows in one batched invocation per tool:
  cuDDL's wall is its search CLI without an index (load the reference database,
  sketch FASTX queries, compare, download results to the host and discard them);
  RabbitSketch's wall sketches FASTX
  queries against resident references; cub-exact times its device phase, and the
  CLI tools their native compare commands. Batch queries are sketched from FASTX
  inside the wall: skani dist sketches both sides itself; Dashing2 loads cached
  reference sketches and evicts query cache entries before each rep; hypergen runs
  `sketch` on the queries before `dist` on the saved reference sketch. All-to-all
  compares each unordered reference pair once: cuDDL reuses
  its database rows, RabbitSketch re-sketches the reference files. Resident times
  cover the comparison alone; the pipeline benchmarks' own output phases
  (per-genome metrics and a JSON sample of match rows) are recorded beside them,
  never counted as comparison. cuDDL's pipeline runs use the database file's row
  layout, index geometry, and search calls, so resident and wall time the same work.
- SEARCH: the query wall includes FASTX query sketching and getting the index
  ready: cuDDL loads a saved index file; tools without an index file format
  (RabbitSketch, Dashing2) construct it inside the timed interval.

With --performance-only, cuDDL also skips internal validation and auxiliary
benchmark suites. Every result is downloaded through a reusable host tile;
per-genome statistics and per-pair JSON rows are not generated.
Resident processing timings are recorded alongside wall timings in the same run.
They exclude parsing and transfers. RabbitSketch native indexed search retains
its coupled text formatting, writing to /dev/null for the resident interval.

Truth comes from cub-exact-pairwise (exact Jaccard and containment,
verified bit-identical against a Python oracle) and, when --skani-truth is
passed, chunked skani dist (ANI). The ANI oracle is opt-in because it is CPU
work and the slowest lane on a full corpus: it can run on a host that is not
measuring a GPU. Fixed-size sketches use 2048 entries (cuDDL buckets with DDL's
5-bit exponent and 11-bit mantissa, Dashing2 registers, RabbitSketch slots,
hypergen dimensions); skani keeps its own sampling and k. Entries differ in width
and information per tool, so tools are compared in error-versus-time space.

Caps are automatic unless passed explicitly: threads default to the CPU
count, max-kmers derives from free VRAM, the evaluated pair set derives
from a short cub probe against a per-invocation time budget, stored rows
derive from a pairs-output size target, and the skani truth chunk derives
from host RAM. Large corpora evaluate a size-spread genome subset that
fits the budget; every lane runs the same files.
"""

import gzip
import json
import math
import os
import shutil
import statistics
import subprocess
import sys
import tempfile
import time
from collections.abc import Callable
from contextlib import ExitStack
from enum import StrEnum
from pathlib import Path
from typing import Annotated

import typer
from benchmark_schema import make_result, write_result

sys.path.insert(0, str(Path(__file__).resolve().parent))
from hypergen_pipeline import dataset_entries, summarize

ROOT = Path(__file__).resolve().parent.parent
APP = typer.Typer()
DEFAULT_TOOLS = "cuddl,rabbitsketch,hypergen,skani,dashing2,cub-exact"
TOOLS = tuple(DEFAULT_TOOLS.split(","))


class Topology(StrEnum):
    """How the lanes pair the corpus."""

    BATCH = "batch"
    ALL_TO_ALL = "all-to-all"


class HypergenDevice(StrEnum):
    """Device hypergen runs on."""

    CPU = "cpu"
    GPU = "gpu"


class CuddlTransfer(StrEnum):
    """How the cuDDL reference build gets bytes to the device."""

    AUTOMATIC = "automatic"
    PINNED = "pinned"
    STAGED = "staged"
    IN_PLACE = "in-place"


class CuddlIndex(StrEnum):
    """Reference index used by cuDDL search."""

    DENSE = "dense"
    SPARSE = "sparse"


def run(
    cmd: list[str],
    capture: bool = False,
    quiet: bool = False,
    log_tail: bool = False,
    resident: list[dict] | None = None,
) -> str:
    """Runs @p cmd and returns its output when @p capture is set.

    @p quiet discards the command's stdout instead. A timed pass needs its time and nothing
    else, and some tools print a whole similarity matrix or a per-sequence log to stdout, which
    would otherwise bury the runner's own report. With @p log_tail, quiet commands retain
    stdout and stderr in a temporary file and report the last 15 lines on failure.
    """
    with ExitStack() as files:
        log = None
        if capture:
            stdout: object = subprocess.PIPE
            stderr: object = None
        elif quiet and log_tail:
            # Retain short diagnostic logs, not potentially quadratic similarity matrices.
            log = files.enter_context(tempfile.TemporaryFile(mode="w+"))
            stdout = log
            stderr = subprocess.STDOUT
        elif quiet:
            stdout = subprocess.DEVNULL
            stderr = subprocess.DEVNULL
        else:
            stdout = None
            stderr = None
        timing_file = (
            files.enter_context(tempfile.NamedTemporaryFile(mode="w+", suffix=".json"))
            if resident is not None
            else None
        )
        try:
            proc = subprocess.run(
                cmd,
                cwd=ROOT,
                check=True,
                stdout=stdout,
                stderr=stderr,
                text=True,
                env={**os.environ, "CUDDL_RESIDENT_TIMINGS": timing_file.name}
                if timing_file
                else None,
            )
            if timing_file is not None:
                native = json.load(timing_file)
                elapsed = native["resident_ms"]
                if (
                    type(elapsed) not in (int, float)
                    or not math.isfinite(elapsed)
                    or elapsed < 0
                ):
                    raise ValueError(
                        f"invalid resident timing from {cmd[0]}: {elapsed!r}"
                    )
                if any(
                    not isinstance(native[key], str) or not native[key]
                    for key in ("source", "device", "input")
                ):
                    raise ValueError(
                        f"invalid resident timing provenance from {cmd[0]}"
                    )
                resident.append(native)
        except subprocess.CalledProcessError as error:
            captured = error.stdout if error.stdout else error.stderr
            if log is not None and not captured:
                log.seek(0)
                captured = log.read()
            detail = "\n".join(line for line in (captured or "").splitlines()[-15:])
            raise typer.BadParameter(
                f"command exited {error.returncode}: {' '.join(cmd)}\n{detail}"
            ) from error
    return proc.stdout if capture else ""


def skani_list(work: Path, name: str, paths: list[str]) -> Path:
    """Writes @p paths to a file for skani, which takes lists instead of argv.

    A full corpus does not fit on a command line, and `--ql`/`--rl` name the same files.
    """
    listing = work / f"skani-{name}.txt"
    listing.write_text("".join(f"{p}\n" for p in paths))
    return listing


def cub_inputs(
    work: Path, name: str, references: list[Path], queries: list[Path]
) -> list[str]:
    """Returns cub's reference and query arguments for @p references and @p queries.

    A full corpus does not fit on a command line, so past `_ARGV_PATH_LIMIT` paths the lists go
    through the benchmark's TOML config instead. Both forms name the same files.
    """
    if len(references) + len(queries) <= _ARGV_PATH_LIMIT:
        args = ["--reference", *[str(p) for p in references]]
        if queries:
            args += ["--query", *[str(p) for p in queries]]
        return args
    import json as jsonlib

    config = work / f"cub-{name}.toml"
    body = "reference = " + jsonlib.dumps([str(p) for p in references]) + "\n"
    if queries:
        body += "query = " + jsonlib.dumps([str(p) for p in queries]) + "\n"
    config.write_text(body)
    return ["--config", str(config)]


def cuddl_database(
    builder: Path, work: Path, name: str, references: list[Path], workers: int
) -> Path:
    """Builds a cuDDL reference database whose IDs follow the order of @p references.

    The builder takes a folder and sorts its paths, so zero-padded symlinks pin the order.
    """
    farm = work / f"{name}-refs"
    farm.mkdir()
    width = len(str(len(references)))
    for i, path in enumerate(references):
        (farm / f"{i:0{width}d}-{path.name}").symlink_to(path.resolve())
    database = work / f"{name}.cuddl"
    run_timed(
        f"cuddl database: {len(references)} references",
        [
            str(builder),
            str(farm),
            "--k",
            "25",
            "--buckets",
            "2048",
            "--exponent-bits",
            "5",
            "--output",
            str(database),
            "--workers",
            str(workers),
        ],
    )
    return database


def cuddl_search_command(
    cli: Path,
    work: Path,
    name: str,
    database: Path,
    queries: list[Path] | None,
    minimum_matches: int,
    workers: int,
    index: Path | None = None,
) -> list[str]:
    """Returns a CLI search that loads @p database (and @p index).

    FASTX @p queries are sketched and searched against every reference; None searches the
    database rows against each other, each unordered pair once. Binary results go to
    `/dev/null`: every row still reaches the host, but a full corpus would otherwise spend the
    wall writing tens of gigabytes to disk.
    """
    # --config belongs to the top-level app, --all-to-all to the search subcommand.
    before, after = [], ["--all-to-all"]
    if queries is not None:
        config = work / f"{name}-cli.toml"
        config.write_text(
            "[search]\nquery = " + json.dumps([str(q) for q in queries]) + "\n"
        )
        before, after = ["--config", str(config)], []
    return [
        str(cli),
        *before,
        "search",
        str(database),
        *after,
        *(["--index", str(index)] if index else []),
        "--output",
        os.devnull,
        "--minimum-matches",
        str(minimum_matches),
        "--workers",
        str(workers),
    ]


def run_timed(label: str, cmd: list[str], capture: bool = False) -> str:
    """Runs one benchmark invocation, reporting what started and how long it took.

    These phases run for minutes on a full corpus, so the log names the work in progress
    rather than leaving a silent pause between measurements.
    """
    typer.echo(f"  {label}: running {Path(cmd[0]).name} ...")
    tick = time.perf_counter()
    stdout = run(cmd, capture=capture, quiet=not capture, log_tail=True)
    typer.echo(f"  {label}: done in {time.perf_counter() - tick:.1f}s")
    return stdout


def wall_of(
    command: list[str] | list[list[str]],
    samples: int,
    warmups: int,
    resident: list[dict] | None = None,
    prepare: Callable[[], None] | None = None,
    resident_skip: int = 0,
) -> list[float]:
    """Times @p command, a command or a group run back to back, per rep.

    @p prepare runs untimed before each rep. Resident timings sum over the group, except its
    first @p resident_skip commands, which count toward wall time only.
    """
    groups = command if command and isinstance(command[0], list) else [command]
    marks: list[float] = []
    for rep in range(warmups + samples):
        if prepare is not None:
            prepare()
        resident_parts = [] if resident is not None else None
        tick = time.perf_counter()
        for index, item in enumerate(groups):
            run(
                item,
                quiet=True,
                resident=resident_parts if index >= resident_skip else None,
            )
        done = time.perf_counter()
        if rep >= warmups:
            marks.append((done - tick) * 1000)
            if resident is not None:
                first = resident_parts[0]
                if any(
                    part[key] != first[key]
                    for part in resident_parts
                    for key in ("source", "device", "input")
                ):
                    raise ValueError("cannot combine different resident timing scopes")
                resident.append(
                    {
                        **first,
                        "resident_ms": sum(p["resident_ms"] for p in resident_parts),
                    }
                )
    return marks


def read_dashing2_panel(
    panel: Path, queries: list[Path], stride: int
) -> tuple[list[dict], int]:
    """Reads dashing2's rectangular panel: one row per reference, one column per query.

    The file holds `len(queries)` values per reference, which at a full corpus is millions of
    numbers, so it is read a line at a time and only a stride sample is kept. The returned count
    is every pair the tool evaluated, which is what the per-pair rate divides by.
    """
    rows: list[dict] = []
    evaluated = 0
    with panel.open() as handle:
        for line in handle:
            if not line.strip() or line.startswith("#"):
                continue
            fields = line.split()
            reference = fields[0]
            for index, value in enumerate(fields[1:]):
                if index >= len(queries) or value == "-":
                    continue
                evaluated += 1
                if evaluated % stride == 0:
                    rows.append(
                        {
                            "query": str(queries[index]),
                            "reference": reference,
                            "jaccard": float(value),
                            "ani": None,
                        }
                    )
    return rows, evaluated


def discover(directory: Path) -> list[Path]:
    exts = (".fa", ".fna", ".fasta", ".ffn", ".frn")
    files = {
        path
        for ext in exts
        for path in (*directory.rglob(f"*{ext}"), *directory.rglob(f"*{ext}.gz"))
        if path.is_file()
    }
    return sorted(files)


def count_pairs(references: list[Path], queries: list[Path], topology: str) -> int:
    if topology == "all-to-all":
        return len(references) * (len(references) - 1) // 2
    return len(queries) * len(references)


def retrieval_phases(
    payload: dict, result_phase: str, device_phase: str | None = None
) -> dict[str, float]:
    """Returns a pipeline run's retrieval cost and the phases its query interval adds.

    The pipeline interval carries the benchmark's own output phase: cuDDL serializes per-genome
    metrics and a sample of match rows into a JSON DOM, RabbitSketch dumps the same shape, and
    the DOM costs milliseconds per reference whatever the pair count is. Reporting that interval
    as the compare or search time prices the benchmark's bookkeeping as if it compared sketches,
    so the lanes report the phase that produced the pairwise results and record the rest beside
    it. @p device_phase, where a tool has one, is the comparison on its own.
    """
    timings = payload["timings"]
    retrieval = timings[result_phase]["median_ms"]
    interval = timings["query_output_wall"]["median_ms"]
    phases = {
        "retrieval_ms": retrieval,
        "excluded_output_ms": (
            0.0 if payload["case"].get("performance_only") else interval - retrieval
        ),
        "end_to_end_ms": timings["end_to_end_wall"]["median_ms"],
    }
    if device_phase is not None:
        phases["device_search_ms"] = timings[device_phase]["median_ms"]
    return phases


# Database files store packed rows and index every bucket with 15-bit keys, so the pipeline runs
# that supply resident timings and accuracy use the same geometry as the timed CLI.
_CUDDL_FILE_CONFIGURATION = [
    "--rows",
    "packed",
    "--indexed-buckets",
    "2048",
    "--key-bits",
    "15",
]


# The phase that produces the pairwise results, per tool and topology. RabbitSketch's search is
# host-resident, so that phase is its whole retrieval. cuDDL's equivalent is search_and_download,
# and the phase below is its device-only part: the comparison without the result transfer.
_RABBITS_RESULT_PHASES = {
    "batch": "search_batch_exhaustive",
    "all-to-all": "search_all_to_all_exhaustive",
}
_CUDDL_DEVICE_PHASES = {
    "batch": "search_batch",
    "all-to-all": "search_all_to_all",
}


_CUB_BYTES_PER_KEY = 32.0  # device bytes per pair key, measured ~28 on RTX 5070 Ti
_VRAM_FRACTION = 0.7  # share of free VRAM cub may size its buffers against
_PAIR_BUDGET_MARGIN = 1.5  # probe-to-full-run cost safety factor
_ARGV_PATH_LIMIT = (
    8192  # genome paths cub may take on one command line; beyond it, a config
)
_PROBE_GENOMES = 24  # probe slice, sized so the pair gap below is measurable
_PROBE_PAIRS = 30  # first probe size
_PROBE_PAIRS_HIGH = 240  # second probe size; the gap has to dwarf probe-to-probe noise
_PROBE_MIN_PAIRS = 12  # at or below this, skip the probe and run everything
_CHUNK_ROW_BYTES = 200  # estimated skani dist TSV bytes per pair row
_CHUNK_BUDGET_BYTES = 1 << 30  # per-invocation truth output target
_CHUNK_MIN_ROWS = 8 << 20  # row floor, so loading the reference sketches amortises
_CHUNK_MAX_QUERIES = 50  # skani dist switches to its hash-table index path above this
_SKANI_REF_ROM_BYTES = (
    4 << 20
)  # resident per reference sketch: measured 1.07 MB, 2.2 MB at 100k
_SKANI_RAM_FRACTION = 4  # share of host RAM one skani reference set may hold
_PAIRS_TARGET_BYTES = 32 << 20  # stored-pairs output target
_PAIR_ROW_BYTES = 160  # estimated stored bytes per pair row


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
    genomes: Annotated[
        Path,
        typer.Argument(
            exists=True,
            file_okay=False,
            help="Directory of reference genomes. Discovered recursively, and reference IDs "
            "follow the sorted paths.",
        ),
    ],
    queries: Annotated[
        list[Path] | None,
        typer.Option(
            "--query",
            exists=True,
            help="Files or directories to use as the query set. Without it, queries are a "
            "size-spread slice of the references.",
        ),
    ] = None,
    query_fraction: Annotated[
        float | None,
        typer.Option(
            min=0,
            max=1,
            help="Share of the references to use as queries, taken spread over genome size. "
            "Batch topology only, and excludes --query-count.",
        ),
    ] = None,
    query_count: Annotated[
        int | None,
        typer.Option(
            min=1,
            help="Number of references to use as queries, taken spread over genome size. Batch "
            "topology only, and excludes --query-fraction.",
        ),
    ] = None,
    reference_count: Annotated[
        int | None,
        typer.Option(
            min=1,
            help="Use only the first N references, in sorted path order, so two hosts compare "
            "the same corpus. Asking for more than were found is an error.",
        ),
    ] = None,
    topology: Annotated[
        Topology,
        typer.Option(
            help="'batch' queries a reference set against a separate query set; 'all-to-all' "
            "compares the references among themselves.",
        ),
    ] = Topology.ALL_TO_ALL,
    tools: Annotated[
        str,
        typer.Option(
            help="Comma-separated tools to measure: cuddl, rabbitsketch, hypergen, skani, "
            "dashing2, cub-exact."
        ),
    ] = DEFAULT_TOOLS,
    performance_only: Annotated[
        bool,
        typer.Option(
            help="Run only the selected tools for timings; skip truth oracles and accuracy metrics. "
            "cuDDL also skips internal validation, auxiliary suites, and per-pair JSON output."
        ),
    ] = False,
    samples: Annotated[
        int,
        typer.Option(
            min=1, help="Timed repetitions per lane. Reported values are medians."
        ),
    ] = 3,
    warmups: Annotated[
        int,
        typer.Option(min=0, help="Untimed repetitions before the timed ones."),
    ] = 1,
    threads: Annotated[
        int,
        typer.Option(
            min=1, help="CPU threads one lane may use. Defaults to the core count."
        ),
    ] = os.cpu_count() or 8,
    output: Annotated[
        Path,
        typer.Option(help="Results JSON, in the schema the plot scripts read."),
    ] = Path("results/micro-comparison.json"),
    pairs_out: Annotated[
        Path | None,
        typer.Option(
            help="Also write the compared pairs, and their oracle scores, here."
        ),
    ] = None,
    max_kmers: Annotated[
        int | None,
        typer.Option(
            min=1,
            help="Cap on the k-mers one genome may contribute, across every lane. Defaults to "
            "what free VRAM allows, and a genome above the cap is an error rather than a "
            "silent truncation.",
        ),
    ] = None,
    recall_k: Annotated[
        int,
        typer.Option(min=1, help="k for the recall@k and top-1 search metrics."),
    ] = 3,
    min_matches: Annotated[
        int,
        typer.Option(
            min=0,
            help="Minimum matching k-mers before a pair is reported, in the lanes that filter.",
        ),
    ] = 5,
    max_pairs: Annotated[
        int | None,
        typer.Option(
            min=0,
            help="Cap on the pairs cub-exact evaluates, taken as an even stride. Unset derives "
            "the cap from a probe and a time budget.",
        ),
    ] = None,
    match_rows: Annotated[
        int | None,
        typer.Option(
            min=0,
            help="Cap on the match rows a lane writes, taken as an even stride. Defaults to a "
            "32 MiB row budget.",
        ),
    ] = None,
    skani_chunk: Annotated[
        int | None,
        typer.Option(
            min=1,
            help="Queries per skani dist invocation, held at or below 50 so skani stays off its "
            "hash-table index path, which screens differently from the linear path.",
        ),
    ] = None,
    skani_refs: Annotated[
        int | None,
        typer.Option(
            min=1,
            help="Reference sketches per skani dist invocation. skani holds every reference "
            "resident, so this is what bounds its memory; it defaults to a quarter of host RAM "
            "at 4 MB per reference, 16000 on a 250 GiB host.",
        ),
    ] = None,
    skani_truth: Annotated[
        bool,
        typer.Option(
            "--skani-truth/--no-skani-truth",
            help="Run the chunked skani dist ANI oracle. Off by default: it is the slowest lane "
            "on a full corpus and it is CPU work, so it can run on a host that is not measuring "
            "a GPU. Without it the ANI error metrics are omitted.",
        ),
    ] = False,
    budget_secs: Annotated[
        float,
        typer.Option(
            min=1,
            help="Wall time the slowest lane may spend per sample. A corpus that would exceed "
            "it is cut down to a size-spread subset.",
        ),
    ] = 600,
    sketch_all: Annotated[
        bool,
        typer.Option(
            help="Sketch the full discovered corpus; compare stays on subset."
        ),
    ] = True,
    hypergen_device: Annotated[
        HypergenDevice,
        typer.Option(help="Device hypergen runs on."),
    ] = HypergenDevice.CPU,
    cuddl_index: Annotated[
        CuddlIndex,
        typer.Option(help="Reference index for cuDDL SEARCH; COMPARE is exhaustive."),
    ] = CuddlIndex.DENSE,
    cuddl_workers: Annotated[
        int | None,
        typer.Option(
            min=1,
            help="Loader workers for all cuDDL stages. Unset uses the thread count, "
            "which past the machine's decompression bandwidth costs instead of saving: on a "
            "24 thread desktop 8 workers built the 5000 genome corpus 18 percent faster.",
        ),
    ] = None,
    cub_stash_mb: Annotated[
        int | None,
        typer.Option(
            min=1,
            help="Host memory cub-exact may hold in resident k-mer arrays. Defaults to half of "
            "the machine's available memory; arrays outside the budget are packed again per "
            "pair.",
        ),
    ] = None,
    cuddl_transfer: Annotated[
        CuddlTransfer,
        typer.Option(
            help="How the cuDDL reference build gets bytes to the device: 'automatic' asks the "
            "device, 'pinned' transfers page-locked memory, 'staged' copies each genome through "
            "a heap buffer, 'in-place' lets the kernels read that heap buffer, which only a "
            "device that reads pageable host memory can do."
        ),
    ] = CuddlTransfer.AUTOMATIC,
) -> None:
    """Time SKETCH, COMPARE, and SEARCH for each tool and score against oracles."""
    if performance_only:
        skani_truth = False
    if cuddl_workers is None:
        cuddl_workers = threads
    selected = [t.strip() for t in tools.split(",") if t.strip()]
    unknown = sorted({t for t in selected if t not in TOOLS})
    if unknown:
        raise typer.BadParameter(
            f"unknown tool {', '.join(unknown)}; known tools are {', '.join(TOOLS)}"
        )
    references = discover(genomes)
    if reference_count is not None:
        # A prefix of the sorted corpus, not a spread: two hosts that collected the same
        # accessions then compare the same references even if one collection stopped earlier.
        # Taking fewer than asked for would silently compare different corpora, so it is an
        # error rather than a clamp.
        if reference_count > len(references):
            raise typer.BadParameter(
                f"--reference-count {reference_count} exceeds the {len(references)} "
                "genomes found"
            )
        references = references[:reference_count]
    query_list = sorted(
        {
            q
            for p in (queries or [])
            for q in ([p] if p.is_file() else p.rglob("*"))
            if q.is_file()
        }
    )
    if query_count is not None and query_fraction is not None:
        raise typer.BadParameter("query-count and query-fraction exclude each other")
    if (query_fraction is not None or query_count is not None) and topology != "batch":
        raise typer.BadParameter("query-count and query-fraction need batch topology")
    if query_fraction is not None and not 0 < query_fraction <= 1:
        raise typer.BadParameter("query-fraction must be within (0, 1]")
    if (query_fraction is not None or query_count is not None) and query_list:
        raise typer.BadParameter("query-count and query-fraction exclude --query")
    if (
        topology == "batch"
        and not query_list
        and query_fraction is None
        and query_count is None
    ):
        raise typer.BadParameter(
            "batch needs --query files, --query-count, or --query-fraction"
        )
    if topology == "all-to-all" and len(references) < 2:
        raise typer.BadParameter("all-to-all needs at least two genomes")
    # No fixed corpus ceiling: the autoscale block below samples the pair
    # space against a time budget instead of refusing large inputs.

    unknown = [t for t in selected if t not in DEFAULT_TOOLS.split(",")]
    if unknown or not selected:
        raise typer.BadParameter(f"unknown tools: {', '.join(unknown) or 'none'}")

    build = ROOT / "build"
    hypergen = build / "subprojects/hypergen/hyper-gen"
    skani = build / "subprojects/skani/skani"
    dashing2 = build / "subprojects/dashing2/dashing2"
    cub = build / "benchmarks/cub-exact-pairwise"
    refbuild = build / "benchmarks/cuddl-reference-build-benchmark"
    cuddl_cli = build / "examples/cuddl-reference-index"
    cuddl_dbbuild = build / "examples/cuddl-build-reference-db"
    rabbit = build / "benchmarks/rabbitsketch-pipeline-benchmark"
    # Forwarded to every cub lane that evaluates pairs, so a large corpus can be told to keep
    # less resident than the default share of host memory.
    cub_stash: list[str] = []
    if cub_stash_mb is not None:
        cub_stash += ["--stash-mb", str(cub_stash_mb)]

    required = [skani] if skani_truth else []  # the ANI oracle is opt-in
    if "hypergen" in selected:
        required.append(hypergen)
    if "dashing2" in selected:
        required.append(dashing2)
    if "cub-exact" in selected:
        required.append(cub)
    if "cuddl" in selected:
        required += [refbuild, cuddl_cli, cuddl_dbbuild]
    if "rabbitsketch" in selected:
        required.append(rabbit)
    for binary in required:
        if not binary.exists():
            raise typer.BadParameter(
                f"missing binary, build first (cuDDL CLIs need -Dexamples=enabled): {binary}"
            )

    measurements: list[dict] = []
    resident_timings: dict[tuple[str, str], dict] = {}
    native_timings: dict[tuple[str, str], dict[str, dict]] = {}
    resident_metadata: dict[tuple[str, str], dict] = {}

    def record_native_resident(tool: str, operation: str, records: list[dict]) -> None:
        first = records[0]
        if any(
            record[key] != first[key]
            for record in records
            for key in ("source", "device", "input")
        ):
            raise ValueError(
                f"inconsistent resident timing scope for {tool} {operation}"
            )
        timing = summarize([record["resident_ms"] for record in records])
        timing["source"] = first["source"]
        resident_timings[(tool, operation)] = timing
        resident_metadata[(tool, operation)] = {
            "resident_device": first["device"],
            "resident_input": first["input"],
        }

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
        if query_fraction is not None or query_count is not None:
            # Batch queries are a size-spread slice of the discovered files.
            count = (
                query_count
                if query_count is not None
                else max(1, round(query_fraction * len(references)))
            )
            count = max(1, min(count, len(references)))
            query_list = _spread_pick(references, sizes, count)
        typer.echo(f"sampling {len(references) + len(query_list)} genomes...")
        auto_flags = {
            "threads": threads is None,
            "max_kmers": max_kmers is None,
            "max_pairs": max_pairs is None,
            "match_rows": match_rows is None,
            "skani_chunk": skani_chunk is None,
            "skani_refs": skani_refs is None,
        }
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
        orig_pairs = count_pairs(references, query_list, topology)
        probe_per_pair_ms = 0.0
        subset_note = "all pairs"
        # --sketch-all keeps the full discovered corpus for sketch lanes.
        # Compare, truth, and search stay on the budget subset.
        sketch_references = list(references)
        sketch_queries = list(query_list)

        def cub_probe_rate(probe_refs, probe_queries) -> float:
            # Probe only estimates per-pair cost. A size-spread handful keeps
            # argv bounded on huge corpora instead of passing every file.
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
                probe_cmd += ["--reference", *[str(p) for p in probe_refs]]
                probe_cmd += ["--query", *[str(p) for p in probe_queries]]
            else:
                probe_cmd += ["--reference", *[str(p) for p in probe_refs]]
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
            run_timed(f"cub-exact autoscale probe ({topology})", probe_cmd)
            probe_wall_ms = (time.perf_counter() - tick) * 1000
            import json as jsonlib_probe

            probe_eval = jsonlib_probe.loads((work / "probe.json").read_text())["case"][
                "pairs_evaluated"
            ]
            # Second probe at more pairs on the same files. Fixed costs
            # (startup, CUDA init, parsing) cancel in the delta, leaving
            # the true per-pair rate. A single-point rate attributes all
            # fixed cost to each pair and undersamples by orders.
            probe_cmd2 = list(probe_cmd)
            probe_cmd2[probe_cmd2.index("--max-pairs") + 1] = str(_PROBE_PAIRS_HIGH)
            probe_cmd2[probe_cmd2.index("--match-rows") + 1] = "64"
            probe_cmd2[probe_cmd2.index("--output") + 1] = str(work / "probe2.json")
            tick2 = time.perf_counter()
            run_timed("cub-exact autoscale probe (larger sample)", probe_cmd2)
            probe_wall2_ms = (time.perf_counter() - tick2) * 1000
            probe_eval2 = jsonlib_probe.loads((work / "probe2.json").read_text())[
                "case"
            ]["pairs_evaluated"]
            # The benchmark times its own pair phase, which is the number wanted: a wall-clock
            # delta still carries everything else that scales with the corpus rather than with
            # the pairs, and cub-exact sorts and deduplicates each genome before any pair runs.
            # Measured that way the rate was 8.5 ms a pair where the pair phase is 0.17 ms, which
            # inflated every estimate and shrank the oracle cap to match.
            phases = jsonlib_probe.loads((work / "probe2.json").read_text())[
                "phases_ms"
            ]
            if probe_eval2 > 0 and "compare" in phases:
                return phases["compare"]["median_ms"] / probe_eval2
            probe_per_pair_ms = 0.0
            if probe_eval2 > probe_eval:
                probe_per_pair_ms = (probe_wall2_ms - probe_wall_ms) / (
                    probe_eval2 - probe_eval
                )
            if probe_per_pair_ms <= 0:
                probe_per_pair_ms = probe_wall2_ms / max(probe_eval2, 1)
            return probe_per_pair_ms

        def skani_probe_rate() -> float:
            # No cub oracle selected: rate the skani ANI truth over a
            # size-spread slice so large corpora still subset. Self-pairs and
            # unrefined rows make this overestimate the per-pair cost, which
            # only tightens the subset.
            slice_refs = (
                _spread_pick(references, sizes, 16)
                if len(references) > 16
                else references
            )
            slice_queries = (
                _spread_pick(query_list, sizes, 16)
                if topology == "batch" and len(query_list) > 16
                else query_list
            )
            if topology == "batch":
                pairs = len(slice_queries) * len(slice_refs)
            else:
                slice_queries = slice_refs
                pairs = max(1, len(slice_refs) * (len(slice_refs) - 1) // 2)
            tick = time.perf_counter()
            run(
                [
                    str(skani),
                    "dist",
                    "-q",
                    *[str(p) for p in slice_queries],
                    "-r",
                    *[str(p) for p in slice_refs],
                    "-o",
                    str(work / "skani-probe.tsv"),
                    "-t",
                    str(threads),
                ],
                quiet=True,
            )
            return (time.perf_counter() - tick) * 1000 / pairs

        # A corpus at or below the probe floor, or an explicit --max-pairs, runs without a
        # probe. Then there is no measured per-pair rate to name or to budget against, and the
        # report says so instead of reading names the probe never bound.
        rates: dict[str, float] = {}
        slowest = "unprobed"
        probe_per_pair_ms = 0.0
        rate_note = "unprobed"
        if max_pairs is None and orig_pairs > _PROBE_MIN_PAIRS:
            probe_refs = (
                _spread_pick(references, sizes, _PROBE_GENOMES)
                if len(references) > _PROBE_GENOMES
                else references
            )
            probe_queries = (
                _spread_pick(query_list, sizes, _PROBE_GENOMES)
                if topology == "batch" and len(query_list) > _PROBE_GENOMES
                else query_list
            )
            # Size the subset by the slowest compare lane we can measure. The exact oracle is
            # fast per pair and the ANI truth is not, and a budget that only fits the oracle
            # still leaves the other lane running for hours.
            # Neither binary is touched unless it is in play, and a run whose oracles are all
            # opted out has nothing to rate and nothing to subset for.
            rates = {"skani": skani_probe_rate()} if skani_truth else {}
            if need_cub:
                rates["cub-exact"] = cub_probe_rate(probe_refs, probe_queries)
            if rates:
                slowest = max(rates, key=lambda tool: rates[tool])
                probe_per_pair_ms = rates[slowest]
                rate_note = " ".join(
                    f"{tool}={value:.2f}ms" + ("*" if tool == slowest else "")
                    for tool, value in sorted(rates.items())
                )
                capacity = int(
                    budget_secs
                    * 1000
                    / (samples * probe_per_pair_ms * _PAIR_BUDGET_MARGIN)
                )
                # An explicit query count is a decision, not a request to be second-guessed: the
                # pairs it implies are reported below instead of being trimmed away silently.
                if capacity < orig_pairs and query_count is None:
                    if topology == "all-to-all":
                        keep = max(2, int((1 + math.sqrt(1 + 8 * capacity)) // 2))
                        references = _spread_pick(references, sizes, keep)
                    else:
                        keep = max(
                            1, min(len(query_list), capacity // max(1, len(references)))
                        )
                        query_list = _spread_pick(query_list, sizes, keep)
                    subset_note = "budget subset"
                capacity_pairs = max(1, capacity)
            else:
                capacity_pairs = None
        else:
            capacity_pairs = None
        total_pairs = count_pairs(references, query_list, topology)
        if subset_note != "all pairs":
            subset_note = f"subset {total_pairs}/{orig_pairs}"
        if match_rows is None:
            match_rows = _PAIRS_TARGET_BYTES // _PAIR_ROW_BYTES
        # The oracle and the cub-exact compare lane want different things from the same knob.
        # The oracle feeds the correctness metrics from the rows that survive the emit stride, so
        # evaluating more than that buys nothing: uncapped, a 256 x 50000 corpus spends about two
        # hours to report 209715 rows. The compare lane is the performance comparison and keeps
        # the whole pair space, like every other tool.
        if max_pairs is None:
            oracle_pairs = min(match_rows or total_pairs, total_pairs)
            if capacity_pairs is not None:
                oracle_pairs = min(oracle_pairs, capacity_pairs)
            all_pairs = 0
            if need_cub and not performance_only:
                typer.echo(f"exact oracle: {oracle_pairs} of {total_pairs} pairs")
        else:
            oracle_pairs = max_pairs
            all_pairs = max_pairs
        # The packed ingest parses every genome into host memory, and its footprint runs to
        # several times the corpus: a 66 GB corpus reached 291 GB on a 384 GB host, so the
        # share-of-RAM test that used to pick it under-estimated and the kernel OOM-killed the
        # lane. Stream records through the device arena instead, which is sized from free
        # device memory and does not grow with the corpus.
        cuddl_ingest = "sequence"
        if skani_refs is None:
            # skani holds every reference sketch resident for the whole invocation, so the
            # reference side is chunked as well as the query side. Without this a full corpus
            # never fits: 100k references reached 215 GB on a 250 GB host and OOM'd it.
            skani_refs = max(
                1, (_host_ram_bytes() // _SKANI_RAM_FRACTION) // _SKANI_REF_ROM_BYTES
            )
        skani_refs = max(1, min(skani_refs, len(references)))
        if skani_chunk is None:
            # A chunk has to be big enough for one reference chunk's load to disappear into the
            # work, so take the larger of a byte target and a row target against that reference
            # chunk rather than the whole corpus.
            # Stay at or below _CHUNK_MAX_QUERIES: above it skani dist builds a marker
            # hash table over the references per invocation. On a redundant corpus the
            # screened path is both slower and a different filter, so the chunk never
            # crosses that line.
            target = max(_CHUNK_BUDGET_BYTES, _CHUNK_MIN_ROWS * _CHUNK_ROW_BYTES)
            by_output = max(1, target // (max(1, skani_refs) * _CHUNK_ROW_BYTES))
            skani_chunk = min(_CHUNK_MAX_QUERIES, by_output)
        reference_paths = set(references)
        file_args = [str(p) for p in references] + [
            str(p) for p in query_list if p not in reference_paths
        ]
        if not sketch_all:
            sketch_references = references
            sketch_queries = query_list
        sketch_reference_paths = set(sketch_references)
        sketch_file_args = [str(p) for p in sketch_references] + [
            str(p) for p in sketch_queries if p not in sketch_reference_paths
        ]

        if need_cub and not performance_only:
            cub_oracle = work / "oracle.json"
            oracle_cmd = [str(cub), "--topology", topology, "--samples", "1"]
            oracle_cmd += cub_stash
            oracle_cmd += cub_inputs(
                work, "oracle", references, query_list if topology == "batch" else []
            )
            oracle_cmd += [
                "--max-kmers",
                str(max_kmers),
                "--max-pairs",
                str(oracle_pairs),
                "--match-rows",
                str(match_rows),
                "--output",
                str(cub_oracle),
            ]
            oracle_tick = time.perf_counter()
            run_timed(f"cub-exact truth oracle ({total_pairs} pairs)", oracle_cmd)
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
        if query_count is not None and total_pairs > 0:
            if not rates:
                typer.echo(
                    f"query set: {len(query_list)} queries x {len(references)} references = "
                    f"{total_pairs} pairs, unprobed"
                )
            else:
                minutes = total_pairs * probe_per_pair_ms / 1000 / 60
                typer.echo(
                    f"query set: {len(query_list)} queries x {len(references)} references = "
                    f"{total_pairs} pairs, about {minutes:.1f} min per sample for the slowest "
                    f"lane ({slowest} at {probe_per_pair_ms:.3f}ms/pair)"
                )
        typer.echo(
            f"auto: threads={threads} max_kmers={max_kmers} pairs={total_pairs}/{orig_pairs} ({subset_note}) match_rows={match_rows} skani_chunk={skani_chunk} skani_refs={skani_refs} budget_secs={budget_secs} per_pair_ms={probe_per_pair_ms:.3f} ({rate_note}) gpu={gpu_name}"
        )
        autoscale_case = {
            "threads": threads,
            "max_kmers": max_kmers,
            "pair_budget_secs": budget_secs,
            "cub_per_pair_ms": round(probe_per_pair_ms, 3),
            "budget_tool": slowest,
            # Metric values are scalars in this schema, so each tool's rate is its own field.
            **{f"pair_ms_{tool}": round(value, 3) for tool, value in rates.items()},
            "pairs_evaluated": total_pairs,
            "pairs_total": orig_pairs,
            "auto_caps": ",".join(k for k, v in auto_flags.items() if v) or "none",
            "gpu": gpu_name,
        }

        # ANI oracle from skani dist, chunked per query group so memory stays
        # bounded: triangle materializes the full N x N matrix. Rows with
        # non-positive ANI are treated as unreported, as with triangle. Opt in with
        # --skani-truth: on a full corpus it is the slowest lane in the harness by a wide margin,
        # and it is CPU work, so it is the lane to leave off the host that is measuring a GPU.
        skani_ani: dict[tuple[str, str], float] = {}
        reference_sketches: list[str] = []
        truth_queries: list[str] = []
        if skani_truth:
            # Sketch both sides once. The truth runs per query chunk, and handing skani raw FASTAs
            # would re-sketch every reference in every chunk: on a full corpus that is the whole
            # corpus read once per chunk.
            truth_db = work / "skani-truth-db"
            run_timed(
                f"skani truth: sketch {len(references)} references",
                [
                    str(skani),
                    "sketch",
                    "--separate-sketches",
                    "-l",
                    str(skani_list(work, "truth-refs", [str(p) for p in references])),
                    "-o",
                    str(truth_db),
                    "-t",
                    str(threads),
                ],
            )
            reference_sketches = sorted(str(p) for p in truth_db.glob("*.sketch"))
            if topology == "batch":
                query_db = work / "skani-truth-queries"
                run_timed(
                    f"skani truth: sketch {len(query_list)} queries",
                    [
                        str(skani),
                        "sketch",
                        "--separate-sketches",
                        "-l",
                        str(
                            skani_list(
                                work, "truth-queries", [str(p) for p in query_list]
                            )
                        ),
                        "-o",
                        str(query_db),
                        "-t",
                        str(threads),
                    ],
                )
                truth_queries = sorted(str(p) for p in query_db.glob("*.sketch"))
            else:
                truth_queries = reference_sketches
        for ref_base in range(0, len(reference_sketches), skani_refs):
            ref_chunk = reference_sketches[ref_base : ref_base + skani_refs]
            ref_list = skani_list(work, f"truth-refs-{ref_base}", ref_chunk)
            for chunk_base in range(0, len(truth_queries), skani_chunk):
                chunk = truth_queries[chunk_base : chunk_base + skani_chunk]
                chunk_tsv = work / f"skani-truth-{ref_base}-{chunk_base}.tsv"
                run(
                    [
                        str(skani),
                        "dist",
                        "--ql",
                        str(
                            skani_list(work, f"truth-q-{ref_base}-{chunk_base}", chunk)
                        ),
                        "--rl",
                        str(ref_list),
                        "-o",
                        str(chunk_tsv),
                        "-t",
                        str(threads),
                    ],
                    quiet=True,
                    log_tail=True,
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
                        "genomes": len(sketch_file_args),
                        "k": sketch_k.get(tool, 0),
                        "threads": threads,
                    },
                    "timings": {"wall": summarize(marks)},
                    "metrics": {
                        "per_genome_ms": statistics.median(marks)
                        / len(sketch_file_args),
                        **(extra or {}),
                    },
                }
            )
            native_timings.setdefault(
                (tool, "sketch"), {"wall": measurements[-1]["timings"]["wall"]}
            )

        if "hypergen" in selected:
            staged_to_real = {}
            # (suffix, refs, queries): full-corpus sketch supplies timing in
            # sketch-all mode while the subset sketch feeds dist.
            hg_runs = [("", references, query_list)]
            timed_suffix = "-full" if sketch_all else ""
            if sketch_all:
                hg_runs.insert(0, ("-full", sketch_references, sketch_queries))
            hg_marks = {}
            hg_resident = {}
            for suffix, refs, run_queries in hg_runs:
                for role, files in (
                    (f"hgrefs{suffix}", refs),
                    (f"hgqueries{suffix}", run_queries),
                ):
                    role_dir = work / role
                    role_dir.mkdir(exist_ok=True)
                    for n, path in enumerate(files):
                        link = role_dir / f"{n}_{path.name}"
                        if link.is_symlink() or link.exists():
                            link.unlink()
                        link.symlink_to(path.resolve())
                sketch_cmds = [
                    [
                        str(hypergen),
                        "sketch",
                        "-p",
                        str(work / f"hgrefs{suffix}"),
                        "-o",
                        str(work / f"hgr{suffix}.sk"),
                        "-t",
                        str(threads),
                        "-k",
                        "25",
                        "-D",
                        hypergen_device,
                        "-d",
                        "2048",
                    ]
                ]
                if topology == "batch":
                    sketch_cmds.append(
                        [
                            str(hypergen),
                            "sketch",
                            "-p",
                            str(work / f"hgqueries{suffix}"),
                            "-o",
                            str(work / f"hgq{suffix}.sk"),
                            "-t",
                            str(threads),
                            "-k",
                            "25",
                            "-D",
                            hypergen_device,
                            "-d",
                            "2048",
                        ]
                    )
                # The subset sketch only feeds dist; its timing is the full run's. One pass is
                # enough, and repeating it four times just clutters the log.
                reps = (samples, warmups) if suffix == timed_suffix else (1, 0)
                hg_resident[suffix] = []
                hg_marks[suffix] = wall_of(
                    sketch_cmds, reps[0], reps[1], resident=hg_resident[suffix]
                )
            marks = hg_marks[timed_suffix]
            record_native_resident("hypergen", "sketch", hg_resident[timed_suffix])
            sketch_times["hypergen"] = marks
            sketch_bytes["hypergen"] = (work / f"hgr{timed_suffix}.sk").stat().st_size
            if topology == "batch":
                sketch_bytes["hypergen"] += (
                    (work / f"hgq{timed_suffix}.sk").stat().st_size
                )
            record_sketch(
                "hypergen",
                hypergen_device,
                marks,
                {"sketch_bytes": sketch_bytes["hypergen"]},
            )
        if "skani" in selected:
            marks = []
            skani_resident = []
            sketch_list = work / "skani-list.txt"
            sketch_list.write_text("".join(p + "\n" for p in sketch_file_args))
            out_dir = work / "skdb"
            for rep in range(warmups + samples):
                # One database at a time. skani sketches into this directory, and only the last
                # rep is measured for size, so a full-corpus copy per rep only grows the work
                # directory. Clearing it keeps each rep a real sketch rather than a reuse.
                shutil.rmtree(out_dir, ignore_errors=True)
                tick = time.perf_counter()
                run(
                    [
                        str(skani),
                        "sketch",
                        "-l",
                        str(sketch_list),
                        "-o",
                        str(out_dir),
                        "-t",
                        str(threads),
                    ],
                    quiet=True,
                    log_tail=True,
                    resident=skani_resident if rep >= warmups else [],
                )
                done = time.perf_counter()
                if rep >= warmups:
                    marks.append((done - tick) * 1000)
            sketch_times["skani"] = marks
            record_native_resident("skani", "sketch", skani_resident)
            sketch_bytes["skani"] = sum(
                p.stat().st_size for p in out_dir.rglob("*") if p.is_file()
            )
            record_sketch(
                "skani", "cpu", marks, {"sketch_bytes": sketch_bytes["skani"]}
            )

        if "dashing2" in selected:
            # (suffix, args): full-corpus sketch supplies timing in sketch-all
            # mode while the subset sketch feeds cmp.
            d2_runs = [("", file_args)]
            d2_timed = "-full" if sketch_all else ""
            if sketch_all:
                d2_runs.insert(0, ("-full", sketch_file_args))
            d2_marks = {}
            d2_resident = []
            for suffix, args in d2_runs:
                list_path = work / f"d2list{suffix}.txt"
                list_path.write_text("".join(p + "\n" for p in args))
                # As with hypergen: the subset sketch only feeds cmp, so it runs once.
                d2_reps = warmups + samples if suffix == d2_timed else 1
                rep_marks = []
                out_dir = work / f"d2{suffix}"
                for rep in range(d2_reps):
                    shutil.rmtree(out_dir, ignore_errors=True)
                    out_dir.mkdir()
                    tick = time.perf_counter()
                    run(
                        [
                            str(dashing2),
                            "sketch",
                            "-k25",
                            "-S2048",
                            f"-p{threads}",
                            "--cache",
                            "--outprefix",
                            str(out_dir),
                            "-F",
                            str(list_path),
                        ],
                        quiet=True,
                        log_tail=True,
                        resident=d2_resident
                        if suffix == d2_timed and rep >= warmups
                        else [],
                    )
                    done = time.perf_counter()
                    if rep >= warmups:
                        rep_marks.append((done - tick) * 1000)
                d2_marks[suffix] = rep_marks
                if suffix == d2_timed:
                    d2_sketch_bytes = sum(
                        path.stat().st_size
                        for path in out_dir.rglob("*")
                        if path.is_file()
                    )
                    if sketch_all:
                        shutil.rmtree(out_dir)
            marks = d2_marks[d2_timed]
            record_native_resident("dashing2", "sketch", d2_resident)
            sketch_times["dashing2"] = marks
            sketch_bytes["dashing2"] = d2_sketch_bytes
            record_sketch(
                "dashing2",
                "SetSketch",
                marks,
                {"sketch_bytes": sketch_bytes["dashing2"]},
            )

        if "cuddl" in selected:
            db_out = work / "ref.cuddl"
            # Full-corpus sketch-all runs pass paths by config file;
            # argv cannot hold a hundred thousand genomes.
            ref_cmd = [str(refbuild)]
            if sketch_all:
                import json as jsonlib_ref

                ref_cfg = work / "refbuild.toml"
                ref_cfg.write_text(
                    "reference = "
                    + jsonlib_ref.dumps([str(p) for p in sketch_file_args])
                    + "\n"
                )
                ref_cmd += ["--config", str(ref_cfg)]
            else:
                ref_cmd += ["--reference", *file_args]
            stdout = run_timed(
                f"cuddl sketch: reference database for {len(sketch_file_args)} genomes "
                f"({max(samples, 2)} samples, {cuddl_workers} loaders)",
                ref_cmd
                + [
                    "--database",
                    str(db_out),
                    "--samples",
                    str(max(samples, 2)),
                    # The benchmark defaults to one loader, which gunzips and
                    # parses every genome on the calling thread.
                    "--workers",
                    str(cuddl_workers),
                    "--transfer",
                    cuddl_transfer,
                ],
                capture=True,
            )
            import json as jsonlib2

            payload = jsonlib2.loads(stdout[stdout.index("{") :])
            resident_timings[("cuddl", "sketch")] = payload["resident"]
            native_timings[("cuddl", "sketch")] = {"wall": payload["wall"]}
            marks = [payload["wall"]["median_ms"]]
            sketch_times["cuddl"] = marks
            sketch_bytes["cuddl"] = db_out.stat().st_size
            record_sketch(
                "cuddl", "reference-db", marks, {"sketch_bytes": sketch_bytes["cuddl"]}
            )
            # The SKETCH database has the same configuration as the CLI's, so COMPARE and
            # SEARCH reuse it whenever it holds exactly their references, in order.
            cuddl_databases = {
                tuple(sketch_file_args if sketch_all else file_args): db_out
            }

            def cuddl_reference_database(refs: list[Path], name: str) -> Path:
                key = tuple(str(p) for p in refs)
                if key not in cuddl_databases:
                    cuddl_databases[key] = cuddl_database(
                        cuddl_dbbuild, work, name, refs, cuddl_workers
                    )
                return cuddl_databases[key]

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
            # Time canonicalisation as part of sketching, using bounded ASCII batches.
            staged_bytes = sum(sizes.get(p, 0) for p in (*references, *query_list))
            resident_cap = max(1 << 20, _host_ram_bytes() // 16)
            typer.echo(
                f"  rabbitsketch: sequence ingest, {staged_bytes >> 30} GiB of input"
            )
            run(
                [
                    str(rabbit),
                    "--threads",
                    str(threads),
                    "--topology",
                    topology,
                    "--samples",
                    str(rabbit_samples),
                    "--warmups",
                    str(warmups),
                    *(["--performance-only"] if performance_only else []),
                    "--k",
                    "25",
                    "--ingest",
                    "sequence",
                    "--resident-bytes",
                    str(resident_cap),
                    "--sketch-size",
                    "2048",
                    "--config",
                    str(cfg),
                    "--output",
                    str(rep),
                ],
                quiet=True,
            )
            payload = jsonlib3.loads(rep.read_text())
            pipe = next(
                m
                for m in payload["measurements"]
                if m["case"].get("measurement") == "pipeline"
            )
            prepare = pipe["timings"]["prepare_wall"]
            native_timings[("rabbitsketch", "sketch")] = {"wall": prepare}
            resident_timings[("rabbitsketch", "sketch")] = pipe["timings"][
                "resident_sketch"
            ]
            resident_timings[("rabbitsketch", "compare")] = pipe["timings"][
                "resident_compare"
            ]
            marks = [prepare["median_ms"]]
            sketch_times["rabbitsketch"] = marks
            record_sketch("rabbitsketch", "FastKMV", marks, {})
            rabbit_report, rabbit_rows = payload, pipe

        if "cub-exact" in selected:
            import json as jsonlib4

            cub_rep = work / "cub.json"
            if sketch_all:
                # Full-corpus sketch supplies timing; the subset run below
                # supplies compare rows. Full lists exceed argv, use config.
                run_timed(
                    f"cub-exact sketch: full corpus, {len(sketch_references)} genomes "
                    f"({samples} samples)",
                    [
                        str(cub),
                        "--topology",
                        topology,
                        "--samples",
                        str(samples),
                        "--warmups",
                        str(warmups),
                        *cub_stash,
                        *cub_inputs(
                            work,
                            "sketch",
                            sketch_references,
                            sketch_queries if topology == "batch" else [],
                        ),
                        "--max-kmers",
                        str(max_kmers),
                        "--sketch-only",
                        "--output",
                        str(work / "cub-sketch.json"),
                    ],
                )
                sketch_payload = jsonlib4.loads((work / "cub-sketch.json").read_text())
            cub_cmd = [
                str(cub),
                "--topology",
                topology,
                "--samples",
                str(samples),
                "--warmups",
                str(warmups),
            ]
            cub_cmd += cub_stash
            cub_cmd += cub_inputs(
                work, "compare", references, query_list if topology == "batch" else []
            )
            cub_cmd += [
                "--max-kmers",
                str(max_kmers),
                "--max-pairs",
                str(all_pairs),
                "--match-rows",
                str(match_rows),
                "--output",
                str(cub_rep),
            ]
            run_timed(
                f"cub-exact compare: {total_pairs} pairs over the budget subset "
                f"({samples} samples)",
                cub_cmd,
            )
            payload = jsonlib4.loads(cub_rep.read_text())
            if sketch_all:
                payload = sketch_payload
            sketch_times["cub-exact"] = [payload["phases_ms"]["sketch"]["median_ms"]]
            native_timings[("cub-exact", "sketch")] = {
                "wall": payload["phases_ms"]["sketch"]
            }
            resident_timings[("cub-exact", "sketch")] = payload["phases_ms"][
                "resident_sketch"
            ]
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
            cub_compare = jsonlib4.loads(cub_rep.read_text())
            cub_phases = cub_compare["phases_ms"]
            native_timings[("cub-exact", "compare")] = {"wall": cub_phases["compare"]}
            resident_timings[("cub-exact", "compare")] = cub_phases["resident_compare"]

        # COMPARE op per tool; errors join cub-exact Jaccard and skani ANI.
        def record_compare(
            tool: str,
            variant: str,
            marks: list[float],
            rows: list[dict],
            evaluated: int | None = None,
        ) -> None:
            jaccard_errors, ani_errors, reported = [], [], 0
            retained_before = len(pair_table)
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
            # Rows may be a sample of what the tool evaluated, so the rate divides by the
            # tool's own count when it reports one.
            pairs = evaluated if evaluated is not None else len(rows)
            metrics: dict = {
                "pairs": pairs,
                "native_pair_rows": len(rows),
                "retained_pair_rows": len(pair_table) - retained_before,
                "report_row_stride": stride,
                "per_pair_ms": statistics.median(marks) / max(pairs, 1),
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
                        "pairs": pairs,
                        "k": sketch_k.get(tool, 0),
                        "threads": threads,
                    },
                    "timings": {"wall": summarize(marks)},
                    "metrics": metrics,
                }
            )
            native_timings.setdefault(
                (tool, "compare"), {"wall": measurements[-1]["timings"]["wall"]}
            )

        if "dashing2" in selected:
            # `cmp` writes a value per pair. A square matrix over a full corpus is about ten
            # billion numbers, which no host can hold, so the queries go in by file: the
            # rectangular panel holds one row per reference and one column per query, which is
            # the same asymmetric comparison bounded by the query count. References load from
            # the sketch cache the sketch lane filled. Batch queries enter under their own
            # names, and their cache entries are removed before every rep, so each timed pass
            # sketches them from FASTX; all-to-all queries are the cached references.
            d2_dir = work / "d2"
            d2_dir.mkdir(exist_ok=True)
            panel_queries = query_list or references
            d2_query_inputs = list(panel_queries)
            d2_prepare = None
            if topology == "batch":
                d2_query_dir = work / "d2-fastx-queries"
                d2_query_dir.mkdir(exist_ok=True)
                d2_query_inputs = []
                for n, path in enumerate(panel_queries):
                    link = d2_query_dir / f"fastx-query-{n}-{path.name}"
                    link.symlink_to(path.resolve())
                    d2_query_inputs.append(link)

                def d2_prepare() -> None:
                    for cached in d2_dir.glob("fastx-query-*"):
                        cached.unlink()

            reference_list = work / "d2refs.txt"
            query_listing = work / "d2queries.txt"
            reference_list.write_text("".join(f"{p}\n" for p in references))
            query_listing.write_text("".join(f"{p}\n" for p in d2_query_inputs))
            panel = work / "d2panel.txt"
            cmp_cmd = [
                str(dashing2),
                "cmp",
                "-k25",
                "-S2048",
                f"-p{threads}",
                "--cache",
                "--outprefix",
                str(d2_dir),
                "-F",
                str(reference_list),
                "-Q",
                str(query_listing),
                "--cmpout",
                str(panel),
            ]
            d2_compare_resident = []
            marks = wall_of(
                [cmp_cmd],
                samples,
                warmups,
                resident=d2_compare_resident,
                prepare=d2_prepare,
            )
            record_native_resident("dashing2", "compare", d2_compare_resident)
            panel_pairs = len(references) * len(panel_queries)
            panel_stride = (
                max(1, (panel_pairs + match_rows - 1) // match_rows)
                if match_rows
                else 1
            )
            rows, evaluated = read_dashing2_panel(panel, panel_queries, panel_stride)
            record_compare("dashing2", "SetSketch", marks, rows, evaluated=evaluated)
            measurements[-1]["metrics"]["native_pair_row_stride"] = panel_stride

        if "hypergen" in selected:
            dist_out = work / "hg.ani"
            query_sketch = (
                str(work / "hgq.sk") if topology == "batch" else str(work / "hgr.sk")
            )
            hg_dist = [
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
            ]
            # dist only reads sketch files, so batch walls sketch the FASTX queries first;
            # resident time stays the comparison alone. All-to-all queries are the reference
            # sketch file.
            hg_query_sketch = [
                str(hypergen),
                "sketch",
                "-p",
                str(work / "hgqueries"),
                "-o",
                query_sketch,
                "-t",
                str(threads),
                "-k",
                "25",
                "-D",
                hypergen_device,
                "-d",
                "2048",
            ]
            hg_compare_resident = []
            marks = wall_of(
                [hg_query_sketch, hg_dist] if topology == "batch" else [hg_dist],
                samples,
                warmups,
                resident=hg_compare_resident,
                resident_skip=1 if topology == "batch" else 0,
            )
            record_native_resident("hypergen", "compare", hg_compare_resident)
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
            record_compare("hypergen", hypergen_device, marks, rows)
        if "skani" in selected:
            # Chunked like the truth lane: one invocation per reference chunk, so skani never
            # holds the whole corpus resident. The sample is the sum over chunks, which is what a
            # host-sized run costs.
            compare_commands: list[list[str]] = []
            compare_outs: list[Path] = []
            # skani compares each query against each reference. A batch run's query side is the
            # query set, not every file: putting file_args on both sides made this lane square the
            # corpus, 50256 x 50256 pairs across 4024 invocations instead of 256 x 50000.
            compare_queries = (
                query_list if topology == "batch" and query_list else file_args
            )
            for ref_base in range(0, len(file_args), skani_refs):
                ref_chunk = file_args[ref_base : ref_base + skani_refs]
                ref_list = skani_list(work, f"compare-r-{ref_base}", ref_chunk)
                for q_base in range(0, len(compare_queries), skani_chunk):
                    q_chunk = compare_queries[q_base : q_base + skani_chunk]
                    chunk_out = work / f"skani-dist-{ref_base}-{q_base}.tsv"
                    compare_outs.append(chunk_out)
                    compare_commands.append(
                        [
                            str(skani),
                            "dist",
                            "--ql",
                            str(
                                skani_list(
                                    work, f"compare-q-{ref_base}-{q_base}", q_chunk
                                )
                            ),
                            "--rl",
                            str(ref_list),
                            "-o",
                            str(chunk_out),
                            "-t",
                            str(threads),
                        ]
                    )
            skani_compare_resident = []
            marks = wall_of(
                compare_commands, samples, warmups, resident=skani_compare_resident
            )
            record_native_resident("skani", "compare", skani_compare_resident)
            rows = []
            for dist_out in compare_outs:
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

            # One exhaustive batched invocation, with no index construction or filtering.
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
            run_timed(
                f"cuddl compare: {len(compare_queries)} queries x "
                f"{len(references)} references ({topology})",
                [
                    str(pipeline_bin),
                    "--topology",
                    topology,
                    *(["--performance-only"] if performance_only else []),
                    "--samples",
                    str(max(samples, 2)),
                    "--warmups",
                    str(warmups),
                    "--ingest",
                    cuddl_ingest,
                    "--resident-bytes",
                    "0",
                    # Same layout and index geometry as the CLI's database file.
                    *_CUDDL_FILE_CONFIGURATION,
                    "--exhaustive",
                    "--minimum-matches",
                    "0",
                    "--workers",
                    str(cuddl_workers),
                    "--config",
                    str(cfg),
                    "--output",
                    str(rep),
                ],
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
            # Retrieval includes exhaustive comparison and every result's transfer to the host.
            phases = retrieval_phases(
                pipe,
                "search_and_download",
                f"{_CUDDL_DEVICE_PHASES[topology]}_exhaustive",
            )
            resident_timings[("cuddl", "compare")] = pipe["timings"][
                f"{_CUDDL_DEVICE_PHASES[topology]}_exhaustive"
            ]
            # Wall is the CLI: load the database, sketch FASTX queries (batch) or reuse its
            # rows (all-to-all, unique pairs only), compare, and discard the binary results.
            cuddl_db = cuddl_reference_database(references, "cuddl-compare")
            typer.echo(f"  cuddl compare: CLI wall ({topology})")
            marks = wall_of(
                cuddl_search_command(
                    cuddl_cli,
                    work,
                    "cuddl-compare",
                    cuddl_db,
                    query_list if topology == "batch" else None,
                    0,
                    cuddl_workers,
                ),
                samples,
                warmups,
            )
            native_timings[("cuddl", "compare")] = {"wall": summarize(marks)}
            record_compare(
                "cuddl",
                "gpu",
                marks,
                rows,
                evaluated=count_pairs(references, query_list, topology),
            )
            measurements[-1]["metrics"].update(phases)

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
            phases = retrieval_phases(rabbit_rows, _RABBITS_RESULT_PHASES[topology])
            # Wall sketches the query side from FASTX against resident reference sketches.
            fastx_compare = rabbit_rows["timings"]["query_fastx_compare"]
            native_timings[("rabbitsketch", "compare")] = {"wall": fastx_compare}
            evaluated = rabbit_rows["metrics"].get("match_rows_total")
            record_compare(
                "rabbitsketch",
                "FastKMV",
                [fastx_compare["median_ms"]] * max(samples, 1),
                rows,
                evaluated=evaluated,
            )
            measurements[-1]["metrics"].update(phases)

        if "cub-exact" in selected:
            rows = [
                {
                    "query": row["query"],
                    "reference": row["reference"],
                    "jaccard": row["jaccard"],
                    "ani": row["mash_ani"],
                }
                for row in cub_compare["pairs"]
            ]
            record_compare(
                "cub-exact",
                "gpu-exact",
                [cub_phases["compare"]["median_ms"]],
                rows,
                evaluated=cub_compare["case"]["pairs_evaluated"],
            )
            measurements[-1]["metrics"].update(autoscale_case)
            # Report what cub kept resident, so a bounded run says whether the budget bit.
            measurements[-1]["metrics"].update(
                {
                    key: cub_compare["case"][key]
                    for key in ("stash_mb_allowed", "stashed_mb", "reparsed_genomes")
                    if key in cub_compare["case"]
                }
            )

        # SEARCH op: ranked retrieval per query; recall@k and top-1 hit rate
        # against the exact Jaccard ranking. Self-pairs never rank.
        query_set = query_list if topology == "batch" else references
        # One shared name list: rebuilding the reference names per query allocates a fresh
        # string per reference per query, which is tens of gigabytes of identical text at a few
        # thousand queries. The list is already name-ordered, which is the tie-break order the
        # exact ranking uses.
        reference_names = sorted({str(r) for r in references})
        candidates: dict[str, list[str]] = {}
        for query in query_set if not performance_only else []:
            name = str(query)
            candidates[name] = [entry for entry in reference_names if entry != name]
        reference_set = set(references)
        n_cand = min(len(references) - (q in reference_set) for q in query_set)
        if n_cand < 1:
            raise typer.BadParameter("SEARCH needs at least one candidate per query")
        k = min(recall_k, n_cand)

        def exact_jaccard(query: str, ref: str) -> float:
            key = (query, ref) if (query, ref) in oracle else (ref, query)
            return oracle[key]["jaccard"]

        exact_rank: dict[str, list[str]] = {}
        for query, refs in candidates.items():
            known = [r for r in refs if (query, r) in oracle or (r, query) in oracle]
            ranked = sorted(known, key=lambda r: (-exact_jaccard(query, r), r))
            exact_rank[query] = ranked
        search_index_ms: dict[str, list[float]] = {}
        search_query_ms: dict[str, list[float]] = {}
        search_phases: dict[str, dict[str, float]] = {}
        search_scores: dict[tuple[str, str, str], float] = {}
        search_cases: dict[str, dict] = {}

        if "cuddl" in selected:
            pipeline_bin = build / "benchmarks/cuddl-pipeline-benchmark"
            if not pipeline_bin.exists():
                raise typer.BadParameter(f"missing binary, build first: {pipeline_bin}")
            cfg = work / "cuddl-search.toml"
            import json as jsonlib6

            # The stage suite needs a query file even for all-to-all; match
            # rows stay triangular over references.
            search_queries = query_list or references[:1]
            # Sketch-all searches the full-corpus index with the subset as queries. All-to-all
            # over that index would need N*(N-1)/2 result rows on the device, which is more than
            # a GPU holds, so the lane runs as batch: same index, queries on one side.
            search_references = sketch_references if sketch_all else references
            search_topology = (
                "batch" if sketch_all and topology == "all-to-all" else topology
            )
            cfg.write_text(
                "reference = "
                + jsonlib6.dumps([str(path) for path in search_references])
                + "\nquery = "
                + jsonlib6.dumps([str(path) for path in search_queries])
                + "\n"
            )
            rep = work / "cuddl-search.json"
            run_timed(
                f"cuddl search: index {len(search_references)} references, "
                f"query {len(search_queries)}",
                [
                    str(pipeline_bin),
                    "--topology",
                    search_topology,
                    *(["--performance-only"] if performance_only else []),
                    "--samples",
                    str(max(samples, 2)),
                    "--warmups",
                    str(warmups),
                    "--ingest",
                    cuddl_ingest,
                    "--resident-bytes",
                    "0",
                    *_CUDDL_FILE_CONFIGURATION,
                    "--index",
                    cuddl_index,
                    "--minimum-matches",
                    str(min_matches),
                    "--workers",
                    str(cuddl_workers),
                    "--config",
                    str(cfg),
                    "--output",
                    str(rep),
                ],
            )
            payload = jsonlib6.loads(rep.read_text())
            pipe = next(
                m
                for m in payload["measurements"]
                if m["case"].get("measurement") == "pipeline"
            )
            # Same counting rule as the compare lane: the query interval here also runs the
            # benchmark's output phase, which prunes nothing when the threshold drops pairs.
            phases = retrieval_phases(
                pipe,
                "search_and_download",
                f"{_CUDDL_DEVICE_PHASES[search_topology]}_indexed",
            )
            search_phases["cuddl"] = phases
            resident_timings[("cuddl", "search")] = pipe["timings"][
                f"{_CUDDL_DEVICE_PHASES[search_topology]}_indexed"
            ]
            # Wall is the CLI: load the database and the saved index, sketch FASTX queries
            # (batch) or reuse its rows (all-to-all, unique pairs only), search, and discard
            # the binary results.
            # Index construction is timed separately as index_build.
            search_db = cuddl_reference_database(search_references, "cuddl-search")
            search_index_file = search_db.with_suffix(f".{cuddl_index}.index")
            run_timed(
                f"cuddl search: save {cuddl_index} index",
                [
                    str(cuddl_cli),
                    "build",
                    str(search_db),
                    "--format",
                    str(cuddl_index),
                    "--output",
                    str(search_index_file),
                ],
            )
            typer.echo(f"  cuddl search: CLI wall ({search_topology})")
            marks = wall_of(
                cuddl_search_command(
                    cuddl_cli,
                    work,
                    "cuddl-search",
                    search_db,
                    query_set if search_topology == "batch" else None,
                    min_matches,
                    cuddl_workers,
                    search_index_file,
                ),
                samples,
                warmups,
            )
            native_timings[("cuddl", "search")] = {
                "query": summarize(marks),
                "index_build": pipe["timings"]["prepare_wall"],
            }
            search_index_ms["cuddl"] = [pipe["timings"]["prepare_wall"]["median_ms"]]
            search_query_ms["cuddl"] = marks
            for m in payload["measurements"]:
                if m["case"].get("measurement") == "match":
                    metrics = m.get("metrics", {})
                    if metrics.get("wkid") is None:
                        continue
                    search_scores[
                        (
                            "cuddl",
                            str(
                                search_references[m["case"]["query_id"]]
                                if search_topology != "batch"
                                else search_queries[m["case"]["query_id"]]
                            ),
                            str(search_references[m["case"]["reference_id"]]),
                        )
                    ] = metrics["wkid"]

        if "skani" in selected:
            skdb = work / "skdb"
            search_out = work / "skani-search.tsv"
            skani_search_resident = []
            marks = wall_of(
                [
                    str(skani),
                    "search",
                    "-d",
                    str(skdb),
                    "--ql",
                    str(skani_list(work, "search-q", [str(q) for q in query_set])),
                    "-o",
                    str(search_out),
                    "--min-af",
                    "0",
                    "-t",
                    str(threads),
                ],
                samples,
                warmups,
                resident=skani_search_resident,
            )
            record_native_resident("skani", "search", skani_search_resident)
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

        if "dashing2" in selected:
            records = []
            if topology == "all-to-all":
                neighbors = work / "d2-neighbors.tsv"
                command = [
                    str(dashing2),
                    "cmp",
                    "-k25",
                    "-S2048",
                    f"-p{threads}",
                    "--cache",
                    "--outprefix",
                    str(d2_dir),
                    "-F",
                    str(reference_list),
                    "--topk",
                    str(k),
                    "--cmpout",
                    str(neighbors),
                ]
                # No index file format: the process wall loads cached sketches, builds the
                # LSH index, and queries it.
                marks = wall_of(command, samples, warmups, resident=records)
                if any(record.get("index_build_ms", 0) <= 0 for record in records):
                    raise ValueError("Dashing2 did not execute its native LSH index")
                search_index_ms["dashing2"] = [r["index_build_ms"] for r in records]
                search_query_ms["dashing2"] = marks
                if not performance_only:
                    with neighbors.open() as handle:
                        for line in handle:
                            if not line.strip() or line.startswith("#"):
                                continue
                            query, *hits = line.rstrip("\n").split("\t")
                            for hit in hits:
                                reference, _, score = hit.rpartition(":")
                                search_scores[("dashing2", query, reference)] = float(
                                    score
                                )
                search_cases["dashing2"] = {
                    "index": "native-lsh",
                    "index_supported": True,
                }
            else:
                search_query_ms["dashing2"] = wall_of(
                    cmp_cmd, samples, warmups, resident=records, prepare=d2_prepare
                )
                search_index_ms["dashing2"] = sketch_times["dashing2"]
                search_cases["dashing2"] = {
                    "index": "none",
                    "index_supported": False,
                    "index_unavailable_reason": "native CLI LSH traversal is all-to-all only",
                    "index_build_includes_sketching": True,
                }
                native_timings[("dashing2", "search")] = (
                    {"index_build": native_timings[("dashing2", "sketch")]["wall"]}
                    if ("dashing2", "sketch") in native_timings
                    else {}
                )
                for row in pair_table:
                    if row["tool"] == "dashing2" and row.get("jaccard") is not None:
                        search_scores[("dashing2", row["query"], row["reference"])] = (
                            row["jaccard"]
                        )
            record_native_resident("dashing2", "search", records)

        if "rabbitsketch" in selected:
            timings = rabbit_rows["timings"]
            index = rabbit_rows["case"]["search_index"]
            index_timing = (
                timings["search_index_build"]
                if index != "none"
                else timings["prepare_wall"]
            )
            # No index file format: the FASTX query wall also constructs the index.
            query_timing = timings["query_fastx_search"]
            search_index_ms["rabbitsketch"] = [index_timing["median_ms"]]
            search_query_ms["rabbitsketch"] = [query_timing["median_ms"]]
            native_timings[("rabbitsketch", "search")] = {
                "index_build": index_timing,
                "query": query_timing,
            }
            resident_timings[("rabbitsketch", "search")] = timings["resident_search"]
            search_cases["rabbitsketch"] = {
                "index": index,
                "index_supported": index != "none",
                "resident_output": rabbit_rows["case"]["search_resident_output"],
                "index_build_includes_sketching": index == "none",
            }
            if index == "none":
                search_cases["rabbitsketch"]["index_unavailable_reason"] = (
                    "native CSR traversal is all-to-all only"
                )
                for row in pair_table:
                    if row["tool"] == "rabbitsketch" and row.get("jaccard") is not None:
                        search_scores[
                            ("rabbitsketch", row["query"], row["reference"])
                        ] = row["jaccard"]
            else:
                for row in rabbit_report["measurements"]:
                    if row["case"]["measurement"] != "search-match":
                        continue
                    query = str(references[row["case"]["query_id"]])
                    reference = str(references[row["case"]["reference_id"]])
                    score = row["metrics"]["jaccard"]
                    search_scores[("rabbitsketch", query, reference)] = score
                    search_scores[("rabbitsketch", reference, query)] = score

        # These lanes expose exhaustive comparison only.
        for tool, score_key in (
            ("hypergen", "ani"),
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
        for tool, score_key in (("cub-exact", "jaccard"),):
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
        for tool in ("cub-exact",):
            if tool in selected:
                for row in pair_table:
                    if row["tool"] == tool and row.get("jaccard") is not None:
                        search_scores[(tool, row["query"], row["reference"])] = row[
                            "jaccard"
                        ]

        def record_search(
            tool: str, variant: str, extra: dict[str, float] | None = None
        ) -> None:
            recalls, top1 = [], 0
            counted = 0
            for query in candidates:
                exact = exact_rank[query]
                scored = [
                    (search_scores[(tool, query, ref)], ref)
                    for ref in exact
                    if (tool, query, ref) in search_scores
                ]
                ranked = [ref for _, ref in sorted(scored, key=lambda t: (-t[0], t[1]))]
                kk = min(k, len(exact))
                if kk < 1:
                    continue
                recalls.append(len(set(ranked[:kk]) & set(exact[:kk])) / kk)
                top1 += bool(ranked) and ranked[0] == exact[0]
                counted += 1
            measurements.append(
                {
                    "implementation": {"name": tool, "variant": variant},
                    "case": {
                        "measurement": "micro-search",
                        "topology": topology,
                        "queries": len(query_set),
                        "k": k,
                        "threads": threads,
                        **search_cases.get(tool, {}),
                    },
                    "timings": {
                        "index_build": summarize(search_index_ms[tool]),
                        "query": summarize(search_query_ms[tool]),
                    },
                    "metrics": {
                        **(
                            {
                                "recall_at_k": sum(recalls) / len(recalls)
                                if recalls
                                else 0.0,
                                "top1_rate": top1 / counted if counted else 0.0,
                                "queries_scored": counted,
                                "accuracy_scope": "oracle_known_pairs",
                            }
                            if not performance_only and oracle
                            else {}
                        ),
                        "per_query_ms": (
                            statistics.median(search_query_ms[tool])
                            / max(len(query_set), 1)
                        ),
                        **(extra or {}),
                    },
                }
            )

        variants = {
            "cuddl": "gpu-indexed",
            "skani": "cpu",
            "hypergen": hypergen_device,
            "dashing2": "SetSketch",
            "rabbitsketch": "FastKMV",
            "cub-exact": "gpu-exact",
        }
        for tool in selected:
            if tool in search_index_ms and tool in search_query_ms:
                record_search(tool, variants[tool], search_phases.get(tool))
                if tool in {"hypergen", "cub-exact"}:
                    resident_timings[(tool, "search")] = resident_timings[
                        (tool, "compare")
                    ]
                    if (tool, "compare") in resident_metadata:
                        resident_metadata[(tool, "search")] = resident_metadata[
                            (tool, "compare")
                        ]
                    measurements[-1]["case"]["resident_reuses_compare"] = True
                    if (tool, "compare") in native_timings:
                        native_timings[(tool, "search")] = {
                            "query": native_timings[(tool, "compare")]["wall"],
                            "index_build": native_timings[(tool, "sketch")]["wall"],
                        }

    for measurement in measurements:
        tool = measurement["implementation"]["name"]
        operation = measurement["case"]["measurement"].removeprefix("micro-")
        measurement["timings"].update(native_timings.get((tool, operation), {}))
        measurement["timings"]["resident"] = resident_timings[(tool, operation)]
        measurement["case"]["resident_device"] = (
            "cuda" if tool in {"cuddl", "cub-exact"} else "cpu"
        )
        measurement["case"]["resident_input"] = (
            "sequence_ascii"
            if operation == "sketch"
            else "sorted_kmer_sets"
            if tool == "cub-exact"
            else "indexed_sketches"
            if operation == "search"
            and (
                tool in {"cuddl", "skani"}
                or measurement["case"].get("index", "none") != "none"
            )
            else "sketches"
        )
        measurement["case"].update(resident_metadata.get((tool, operation), {}))
        if measurement["implementation"]["name"] == "cuddl" and measurement["case"][
            "measurement"
        ] in {"micro-compare", "micro-search"}:
            measurement["case"]["index"] = (
                "none"
                if measurement["case"]["measurement"] == "micro-compare"
                else cuddl_index
            )

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
    # The pipeline lanes' query interval also runs each benchmark's own output phase. Say what
    # is not in the numbers above it, so nobody reads it as comparison time.
    excluded = [
        (m["implementation"]["name"], m["metrics"]["excluded_output_ms"])
        for m in measurements
        if m["case"]["measurement"] == "micro-compare"
        and "excluded_output_ms" in m["metrics"]
    ]
    if excluded:
        typer.echo(
            "excluded from the lanes above (per-genome metrics + match-row JSON): "
            + ", ".join(f"{tool} {value:.1f}ms" for tool, value in excluded)
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
