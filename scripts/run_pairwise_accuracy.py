#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = ["fastjsonschema", "typer"]
# ///
"""Run cuDDL, BBTools, RabbitSketch, and cuco HLL accuracy cases into one JSON result."""

import csv
import hashlib
import io
import math
import os
import shlex
import shutil
import subprocess
import tempfile
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Annotated, TextIO

import benchmark_schema
import typer
from anim import anim, plain_fasta, sequence_bases

ROOT = Path(__file__).resolve().parent.parent
UINT64_MASK = (1 << 64) - 1
DNA_ALPHABET = b"ACGT"
DNA_TRANSLATION = bytes(DNA_ALPHABET[value & 3] for value in range(256))
DEFAULT_POWERS = (16, 18, 20)
DEFAULT_ANI_LEVELS = (0.85, 0.87, 0.90, 0.95, 0.97, 0.99, 0.999, 1.0)
DEFAULT_SIZE_RATIOS = (1, 2, 10)
CASES_FIELDS = (
    "generator_seed",
    "power",
    "trial",
    "size_ratio",
    "requested_ani",
    "actual_ani",
    "mutation_count",
    "reference_bases",
    "query_bases",
    "reference_sha256",
    "query_sha256",
    "reference_path",
    "query_path",
)
CSV_FIELDS = (
    "implementation",
    "generator_seed",
    "k",
    "buckets",
    "power",
    "trial",
    "size_ratio",
    "requested_ani",
    "actual_ani",
    "mutation_count",
    "reference_bases",
    "query_bases",
    "reference_sha256",
    "query_sha256",
    "reference_path",
    "query_path",
    "orientation",
    "left_cardinality",
    "right_cardinality",
    "intersection",
    "lower",
    "equal",
    "higher",
    "both_empty",
    "exact_cardinality",
    "sketch_cardinality",
    "cardinality_signed_error",
    "cardinality_absolute_error",
    "cardinality_relative_error",
    "cardinality_absolute_relative_error",
    "sketch_cardinality_bbtools",
    "cardinality_bbtools_signed_error",
    "cardinality_bbtools_absolute_error",
    "cardinality_bbtools_relative_error",
    "cardinality_bbtools_absolute_relative_error",
    "sketch_cardinality_paper",
    "cardinality_paper_signed_error",
    "cardinality_paper_absolute_error",
    "cardinality_paper_relative_error",
    "cardinality_paper_absolute_relative_error",
    "sketch_cardinality_uncapped",
    "cardinality_uncapped_signed_error",
    "cardinality_uncapped_absolute_error",
    "cardinality_uncapped_relative_error",
    "cardinality_uncapped_absolute_relative_error",
    "exact_containment",
    "sketch_containment",
    "containment_signed_error",
    "containment_absolute_error",
    "exact_completeness",
    "sketch_completeness",
    "completeness_signed_error",
    "completeness_absolute_error",
    "exact_wkid",
    "sketch_wkid",
    "wkid_signed_error",
    "wkid_absolute_error",
    "exact_set_derived_ani",
    "exact_ani",
    "sketch_ani",
    "ani_signed_error",
    "ani_absolute_error",
    "aligned_fraction",
)
PATH_FIELDS = {"reference_path", "query_path"}
PUBLIC_FIELDS = tuple(field for field in CSV_FIELDS if field not in PATH_FIELDS)
KEY_FIELDS = (
    "generator_seed",
    "k",
    "buckets",
    "power",
    "trial",
    "requested_ani",
    "actual_ani",
    "mutation_count",
    "reference_bases",
    "query_bases",
    "reference_sha256",
    "query_sha256",
    "reference_path",
    "query_path",
    "orientation",
    "left_cardinality",
    "right_cardinality",
    "intersection",
)
CASE_FIELDS = tuple(field for field in KEY_FIELDS if field not in PATH_FIELDS)
INTEGER_FIELDS = {
    "generator_seed",
    "k",
    "buckets",
    "power",
    "trial",
    "size_ratio",
    "mutation_count",
    "reference_bases",
    "query_bases",
    "left_cardinality",
    "right_cardinality",
    "intersection",
    "lower",
    "equal",
    "higher",
    "both_empty",
}
FLOAT_FIELDS = {
    "requested_ani",
    "actual_ani",
    "exact_cardinality",
    "sketch_cardinality",
    "cardinality_signed_error",
    "cardinality_absolute_error",
    "cardinality_relative_error",
    "cardinality_absolute_relative_error",
    "sketch_cardinality_bbtools",
    "cardinality_bbtools_signed_error",
    "cardinality_bbtools_absolute_error",
    "cardinality_bbtools_relative_error",
    "cardinality_bbtools_absolute_relative_error",
    "sketch_cardinality_paper",
    "cardinality_paper_signed_error",
    "cardinality_paper_absolute_error",
    "cardinality_paper_relative_error",
    "cardinality_paper_absolute_relative_error",
    "sketch_cardinality_uncapped",
    "cardinality_uncapped_signed_error",
    "cardinality_uncapped_absolute_error",
    "cardinality_uncapped_relative_error",
    "cardinality_uncapped_absolute_relative_error",
    "exact_containment",
    "sketch_containment",
    "containment_signed_error",
    "containment_absolute_error",
    "exact_completeness",
    "sketch_completeness",
    "completeness_signed_error",
    "completeness_absolute_error",
    "exact_wkid",
    "sketch_wkid",
    "wkid_signed_error",
    "wkid_absolute_error",
    "exact_ani",
    "sketch_ani",
    "ani_signed_error",
    "ani_absolute_error",
    "aligned_fraction",
}
COUNT_FIELDS = ("lower", "equal", "higher", "both_empty")
app = typer.Typer(
    help="Compare cuDDL, BBTools, RabbitSketch, and cuco HLL on deterministic raw DNA.",
)


def available_cpus() -> int:
    try:
        return len(os.sched_getaffinity(0))
    except AttributeError:
        return os.cpu_count() or 1


def project_path(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def run(
    command: list[str], *, capture: bool = False
) -> subprocess.CompletedProcess[str]:
    typer.echo("+ " + shlex.join(command), err=True)
    return subprocess.run(
        command,
        cwd=ROOT,
        check=True,
        text=True,
        stdout=subprocess.PIPE if capture else None,
    )


def read_csv(source: TextIO) -> tuple[list[str], list[dict[str, str]]]:
    reader = csv.DictReader(source)
    return list(reader.fieldnames or []), list(reader)


def coerce_row(row: dict[str, str]) -> dict[str, object]:
    coerced: dict[str, object] = {}
    for field in PUBLIC_FIELDS:
        value = row[field]
        if value == "":
            continue
        coerced[field] = (
            int(value)
            if field in INTEGER_FIELDS
            else float(value)
            if field in FLOAT_FIELDS
            else value
        )
    return coerced


def splitmix64(value: int) -> int:
    value &= UINT64_MASK
    value ^= value >> 30
    value = (value * 0xBF58476D1CE4E5B9) & UINT64_MASK
    value ^= value >> 27
    value = (value * 0x94D049BB133111EB) & UINT64_MASK
    return (value ^ (value >> 31)) & UINT64_MASK


def generate_dna(seed: int, bases: int) -> bytes:
    raw = bytearray()
    state = seed
    for _ in range((bases + 7) // 8):
        state = splitmix64(state)
        raw.extend(state.to_bytes(8, "little"))
    return bytes(raw[:bases]).translate(DNA_TRANSLATION)


def mutation_plan(
    sequence: bytes, mutation_count: int, seed: int
) -> list[tuple[int, int]]:
    """Select an ordered random subset and a non-identity replacement for each base."""
    if mutation_count < 0 or mutation_count > len(sequence):
        raise ValueError("mutation count is outside the query")

    swaps: dict[int, int] = {}
    state = splitmix64(seed ^ 0xD1B54A32D192ED03)
    plan: list[tuple[int, int]] = []
    for index in range(mutation_count):
        state = splitmix64(state)
        selected = index + state % (len(sequence) - index)
        position = swaps.get(selected, selected)
        swaps[selected] = swaps.get(index, index)
        state = splitmix64(state)
        original = sequence[position]
        shift = 1 + state % 3
        plan.append(
            (position, DNA_ALPHABET[(DNA_ALPHABET.index(original) + shift) & 3])
        )
    return plan


def mutate_dna(
    sequence: bytes, plan: list[tuple[int, int]], mutation_count: int
) -> bytes:
    mutated = bytearray(sequence)
    for position, replacement in plan[:mutation_count]:
        mutated[position] = replacement
    return bytes(mutated)


def write_fasta(path: Path, name: str, sequence: bytes) -> None:
    path.write_bytes(b">" + name.encode("ascii") + b"\n" + sequence + b"\n")


# A pair aligned over the 15% floor at 75% identity still shares about 0.15 * 0.75**25 = 1.1e-4
# of its 25-mers, while unrelated bacteria share almost none, so pairs below this exact
# containment cannot be scored and are not aligned. On 100 RefSeq genomes it skips half of the
# pairs, and every scored pair has a containment of at least 9.7e-3.
MIN_ALIGNABLE_CONTAINMENT = 1e-4


def anim_table(
    nucmer: Path,
    delta_filter: Path,
    genomes: list[str],
    digests: dict[str, str],
    candidates: set[tuple[str, str]],
    cache_path: Path,
    work: Path,
    threads: int,
) -> dict[tuple[str, str], tuple[float, float]]:
    """Map (query, reference) path pair to (ANIm fraction, aligned fraction), one nucmer
    alignment per unordered pair in @p candidates, given as sorted path pairs. The aligned
    fraction is the smaller of the two genomes' shares inside the 1-to-1 alignments, so a floor
    on it holds for both. Pairs without alignments, or not in @p candidates, get ANI 0 and
    align nothing, so the floor leaves them unscored. Each alignment is appended to
    @p cache_path under the genomes' SHA-256 digests as it completes, and pairs found there
    are not aligned again."""
    cached: dict[tuple[str, str], tuple[float, float]] = {}
    if cache_path.is_file():
        for line in cache_path.read_text().splitlines():
            first, second, ani, aligned = line.split("\t")
            cached[(first, second)] = (float(ani), float(aligned))
    fasta = work / "anim-fasta"
    fasta.mkdir()
    table: dict[tuple[str, str], tuple[float, float]] = {
        (query, reference): (0.0, 0.0) for query in genomes for reference in genomes
    }
    pairs = []
    for i in range(len(genomes)):
        for j in range(i):
            if tuple(sorted((genomes[i], genomes[j]))) not in candidates:
                continue
            value = cached.get(
                tuple(sorted((digests[genomes[i]], digests[genomes[j]])))
            )
            if value is None:
                pairs.append((i, j))
            else:
                table[(genomes[i], genomes[j])] = table[(genomes[j], genomes[i])] = (
                    value
                )
    total = len(genomes) * (len(genomes) - 1) // 2
    typer.echo(
        f"ANIm truth: {len(candidates)} of {total} pairs share at least "
        f"{MIN_ALIGNABLE_CONTAINMENT:g} of their k-mers, {len(candidates) - len(pairs)} are in "
        f"{cache_path}, aligning {len(pairs)} on {threads} threads ..."
    )
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(threads) as pool, cache_path.open("a") as cache:
        plain = list(
            pool.map(
                lambda indexed: plain_fasta(ROOT / indexed[1], fasta, indexed[0]),
                enumerate(genomes),
            )
        )
        bases = list(pool.map(sequence_bases, plain))
        alignments = pool.map(
            lambda pair: anim(
                nucmer, delta_filter, work, plain[pair[1]], plain[pair[0]]
            ),
            pairs,
        )
        for (query, reference), alignment in zip(pairs, alignments):
            if alignment is None:
                value = (0.0, 0.0)
            else:
                aligned = min(
                    alignment.query_bases / bases[query],
                    alignment.reference_bases / bases[reference],
                )
                value = (alignment.ani / 100.0, aligned)
            table[(genomes[query], genomes[reference])] = value
            table[(genomes[reference], genomes[query])] = value
            first, second = sorted(
                (digests[genomes[query]], digests[genomes[reference]])
            )
            cache.write(f"{first}\t{second}\t{value[0]!r}\t{value[1]!r}\n")
            cache.flush()
    shutil.rmtree(fasta)
    for name in genomes:
        table[(name, name)] = (1.0, 1.0)
    return table


def discover_genomes(directory: Path) -> list[Path]:
    """Recursively collect FASTX files, sorted for deterministic sampling."""
    exts = (".fa", ".fna", ".fasta", ".ffn", ".frn")
    files = {
        path
        for ext in exts
        for path in (*directory.rglob(f"*{ext}"), *directory.rglob(f"*{ext}.gz"))
        if path.is_file()
    }
    return sorted(files)


def write_genome_cases(
    directory: Path, genomes: list[Path], seed: int, count: int
) -> Path:
    """Write one ordered-pair case per sampled genome pair; truth from tools."""
    import gzip

    def base_count(path: str) -> int:
        opener = gzip.open if path.endswith(".gz") else open
        total = 0
        with opener(path, "rt") as handle:
            for line in handle:
                if line and not line.startswith(">"):
                    total += len(line.strip())
        return total

    # The genomes whose hashed file names come first: accession order clusters species, and
    # the hash keeps the collection's mix and picks the same genomes on every host.
    ranked = sorted(genomes, key=lambda p: hashlib.blake2b(p.name.encode()).digest())
    sampled = sorted(str(p) for p in ranked[:count])
    bases = {path: base_count(path) for path in sampled}
    digests = {
        path: hashlib.sha256(Path(path).read_bytes()).hexdigest() for path in sampled
    }
    cases_csv = directory / "cases.csv"
    with cases_csv.open("w", newline="") as destination:
        writer = csv.DictWriter(destination, fieldnames=CASES_FIELDS)
        writer.writeheader()
        for reference in sampled:
            for query in sampled:
                if query == reference:
                    continue
                writer.writerow(
                    {
                        "generator_seed": seed,
                        "power": 0,
                        "trial": 0,
                        "size_ratio": 0,
                        "requested_ani": 0.0,
                        "actual_ani": 0.0,
                        "mutation_count": 0,
                        "reference_bases": bases[reference],
                        "query_bases": bases[query],
                        "reference_sha256": digests[reference],
                        "query_sha256": digests[query],
                        "reference_path": reference,
                        "query_path": query,
                    }
                )
    return cases_csv


def write_cases(
    directory: Path,
    powers: list[int],
    ani_levels: list[float],
    size_ratios: list[int],
    trials: int,
    root_seed: int,
) -> Path:
    inputs = directory / "inputs"
    inputs.mkdir()
    cases_csv = directory / "cases.csv"
    maximum_ratio = max(size_ratios)

    with cases_csv.open("w", newline="") as destination:
        writer = csv.DictWriter(destination, fieldnames=CASES_FIELDS)
        writer.writeheader()
        for power in powers:
            query_bases = 1 << power
            for trial in range(trials):
                generator_seed = splitmix64(root_seed ^ (power << 48) ^ (trial << 24))
                ancestor = generate_dna(generator_seed, maximum_ratio * query_bases)

                references: dict[int, tuple[Path, str]] = {}
                for ratio in size_ratios:
                    reference_bases = ratio * query_bases
                    if reference_bases in references:
                        continue
                    sequence = ancestor[:reference_bases]
                    path = (
                        inputs / f"p{power}_t{trial}_reference_n{reference_bases}.fna"
                    )
                    write_fasta(
                        path,
                        f"reference_p{power}_t{trial}_n{reference_bases}",
                        sequence,
                    )
                    references[reference_bases] = (
                        path,
                        hashlib.sha256(sequence).hexdigest(),
                    )

                queries: dict[int, tuple[Path, str, float]] = {}
                query = ancestor[:query_bases]
                maximum_mutations = max(
                    round((1.0 - requested_ani) * query_bases)
                    for requested_ani in ani_levels
                )
                plan = mutation_plan(query, maximum_mutations, generator_seed)
                for requested_ani in ani_levels:
                    mutation_count = round((1.0 - requested_ani) * query_bases)
                    if mutation_count in queries:
                        continue
                    sequence = mutate_dna(query, plan, mutation_count)
                    path = inputs / f"p{power}_t{trial}_query_m{mutation_count}.fna"
                    write_fasta(
                        path, f"query_p{power}_t{trial}_m{mutation_count}", sequence
                    )
                    queries[mutation_count] = (
                        path,
                        hashlib.sha256(sequence).hexdigest(),
                        (query_bases - mutation_count) / query_bases,
                    )

                for ratio in size_ratios:
                    reference_bases = ratio * query_bases
                    reference_path, reference_sha256 = references[reference_bases]
                    for requested_ani in ani_levels:
                        mutation_count = round((1.0 - requested_ani) * query_bases)
                        query_path, query_sha256, actual_ani = queries[mutation_count]
                        writer.writerow(
                            {
                                "generator_seed": generator_seed,
                                "power": power,
                                "trial": trial,
                                "size_ratio": ratio,
                                "requested_ani": requested_ani,
                                "actual_ani": actual_ani,
                                "mutation_count": mutation_count,
                                "reference_bases": reference_bases,
                                "query_bases": query_bases,
                                "reference_sha256": reference_sha256,
                                "query_sha256": query_sha256,
                                "reference_path": reference_path,
                                "query_path": query_path,
                            }
                        )
    return cases_csv


ERROR_SUFFIX = "_absolute_error"
# Summarised as `<metric>_relative`: cardinality errors also scaled by the exact count.
RELATIVE_ERROR_SUFFIX = "_absolute_relative_error"
SUMMARY_QUANTILES = {"median": 0.5, "p90": 0.9, "p95": 0.95, "p99": 0.99}


def percentile(ordered: list[float], q: float) -> float:
    """Linear interpolation between the closest ranks of sorted values, numpy's default."""
    position = q * (len(ordered) - 1)
    low = math.floor(position)
    high = min(low + 1, len(ordered) - 1)
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)


def wkid_root_ani(intersection: int, left: int, right: int, k: int) -> float:
    wkid = intersection / min(left, right)
    return wkid ** (1.0 / k) if wkid > 0.0 else 0.0


def dice_root_ani(intersection: int, left: int, right: int, k: int) -> float:
    jaccard = intersection / (left + right - intersection)
    return (2.0 * jaccard / (1.0 + jaccard)) ** (1.0 / k) if jaccard > 0.0 else 0.0


def poisson_ani(intersection: int, left: int, right: int, k: int) -> float:
    jaccard = intersection / (left + right - intersection)
    if jaccard <= 0.0:
        return 0.0
    return max(0.0, 1.0 + math.log(2.0 * jaccard / (1.0 + jaccard)) / k)


# The k-mer similarity to ANI conversion each lane applies to its sketch estimate. The same
# formula over the exact k-mer sets separates the sketch's error (`ani_sketch`) from the
# formula's own error against alignment ANI. skani aligns, so it has no exact-set counterpart.
ANI_FORMULAS = {
    **dict.fromkeys(
        (
            "cuddl",
            "cuddl-bbtools",
            "cuddl-paper",
            "cuddl-uncapped",
            "bbtools",
            "dashing2",
            "cuco_hll",
        ),
        wkid_root_ani,
    ),
    "rabbitsketch": dice_root_ani,
    **dict.fromkeys(
        ("simdsketch", "sketchlib", "pp-sketchlib", "hypergen", "cub-exact"),
        poisson_ani,
    ),
}
# Every formula over the exact k-mer sets, scored against the same ANI truth as the lanes.
EXACT_SET_LANES = {
    "exact-wkid-root": wkid_root_ani,
    "exact-dice-root": dice_root_ani,
    "exact-poisson": poisson_ani,
}
# Species-level comparisons sit at or above it, and the k-mer formulas drift below it.
ANI_BIN_EDGE = 0.95


def summary_key(case: dict, metric: str) -> tuple:
    """What one summarised value describes, so pairs seen from both sides count once. Rows read
    the pair from its left set: the query in `query_to_reference` and in the orientation-free
    rows of the ANI-only lanes, the reference otherwise."""
    reference, query = case["reference_sha256"], case["query_sha256"]
    left, right = (
        (reference, query)
        if case.get("orientation") == "reference_to_query"
        else (query, reference)
    )
    if metric.startswith("cardinality"):
        return (left,)
    if metric in ("containment", "completeness"):
        return (left, right)
    return tuple(sorted((left, right)))


def error_summaries(measurements: list[dict]) -> list[dict]:
    """One measurement per implementation and metric: the distribution of its absolute error
    with every value counted once, per genome for cardinality, per ordered pair for the
    directional containment and completeness, and per unordered pair for the rest."""
    errors: dict[tuple[str, str], list[float]] = defaultdict(list)
    seen: set[tuple] = set()
    for measurement in measurements:
        name = measurement["implementation"]["name"]
        for key, value in measurement["metrics"].items():
            if key.endswith(ERROR_SUFFIX):
                metric = key.removesuffix(ERROR_SUFFIX)
            elif key.endswith(RELATIVE_ERROR_SUFFIX):
                metric = key.removesuffix(RELATIVE_ERROR_SUFFIX) + "_relative"
            else:
                continue
            identity = (name, metric, summary_key(measurement["case"], metric))
            if identity in seen:
                continue
            seen.add(identity)
            errors[name, metric].append(value)
    summaries = []
    for (name, metric), values in errors.items():
        values.sort()
        summaries.append(
            {
                "implementation": {"name": name},
                "case": {"measurement": "error_summary", "metric": metric},
                "metrics": {
                    "count": len(values),
                    "absolute_error_mean": math.fsum(values) / len(values),
                    **{
                        f"absolute_error_{label}": percentile(values, q)
                        for label, q in SUMMARY_QUANTILES.items()
                    },
                    "absolute_error_max": values[-1],
                },
            }
        )
    return summaries


def case_matcher(original_rows: list[dict]):
    """Tests whether a lane row matches some cuDDL case on the case fields both carry, with
    one set lookup per row. cuDDL rows all carry the same fields."""
    keys: dict[tuple[str, ...], set[tuple]] = {}

    def matches(row: dict) -> bool:
        fields = tuple(f for f in CASE_FIELDS if f in row and f in original_rows[0])
        if fields not in keys:
            keys[fields] = {tuple(o[f] for f in fields) for o in original_rows}
        return tuple(row[f] for f in fields) in keys[fields]

    return matches


@app.command()
def main(
    output: Annotated[
        Path, typer.Option(dir_okay=False, help="Combined accuracy JSON output")
    ] = Path("results/pairwise-accuracy.json"),
    build_dir: Annotated[
        Path, typer.Option(file_okay=False, help="Configured Meson build directory")
    ] = Path("build"),
    bbtools_jar: Annotated[
        Path, typer.Option(dir_okay=False, help="Pinned BBTools jar")
    ] = Path("subprojects/bbmap/bbtools.jar"),
    threads: Annotated[
        int, typer.Option(min=1, help="Threads used to build reference sketches")
    ] = available_cpus(),
    powers: Annotated[
        list[int] | None,
        typer.Option("--power", min=5, help="Base-2 query length exponent"),
    ] = None,
    ani_levels: Annotated[
        list[float] | None,
        typer.Option("--ani", min=0.0, max=1.0, help="Requested nucleotide identity"),
    ] = None,
    size_ratios: Annotated[
        list[int] | None,
        typer.Option("--size-ratio", min=1, help="Reference/query length ratio"),
    ] = None,
    trials: Annotated[
        int, typer.Option(min=1, help="Independent trials per parameter point")
    ] = 8,
    seed: Annotated[int, typer.Option(min=0, help="Root generator seed")] = 42,
    genome_dir: Annotated[
        Path | None,
        typer.Option(
            help="Directory to scan recursively for FASTX genomes. When set, "
            "cases come from real genome pairs instead of synthetic DNA: "
            "genome_count files are sampled and every ordered pair becomes "
            "a case. Synthetic options (--power, --ani, --size-ratio, "
            "--trials, --seed) are rejected in this mode."
        ),
    ] = None,
    anim_cache: Annotated[
        Path,
        typer.Option(
            dir_okay=False,
            help="Genome-dir mode only: ANIm results by genome SHA-256 pair, read before "
            "aligning and appended to as alignments finish, so a rerun aligns only new pairs.",
        ),
    ] = Path("build/anim-cache.tsv"),
    genome_count: Annotated[
        int, typer.Option(min=2, help="Genomes to sample from genome-dir.")
    ] = 8,
    ani_min_aligned_fraction: Annotated[
        float,
        typer.Option(
            min=0.0,
            max=1.0,
            help="Genome-dir mode only: aligned fraction a pair needs before its ANI is "
            "scored. A pair that barely aligns carries no identity signal, so a set-derived "
            "ANI over it measures nothing: below this floor the ANI error columns are left "
            "out and the row keeps its estimates. Matches skani's default reporting floor.",
        ),
    ] = 0.15,
    implementations: Annotated[
        str,
        typer.Option(
            help="Comma-separated lanes to run: cuddl, bbtools, rabbitsketch, "
            "cuco_hll, dashing2, skani, hypergen, simdsketch, sketchlib, pp-sketchlib, cub-exact. "
            "cuDDL and BBTools always run, they define cases and the CSV all lanes "
            "validate against."
        ),
    ] = "cuddl,bbtools,rabbitsketch,cuco_hll,dashing2,skani,hypergen,simdsketch,sketchlib,pp-sketchlib,cub-exact",
    pair_rows: Annotated[
        bool,
        typer.Option(
            help="Keep one measurement per implementation, ordered pair and orientation "
            "beside the per-implementation error summaries. They number about 20 per ordered "
            "pair at 2 KB each, so a corpus of 1,000 genomes would write 40 GB of them."
        ),
    ] = True,
) -> None:
    """Build and run all implementations, then publish one JSON result."""
    selected = [name.strip() for name in implementations.split(",") if name.strip()]
    unknown = [
        name
        for name in selected
        if name
        not in (
            "cuddl",
            "bbtools",
            "rabbitsketch",
            "cuco_hll",
            "dashing2",
            "skani",
            "hypergen",
            "simdsketch",
            "sketchlib",
            "pp-sketchlib",
            "cub-exact",
        )
    ]
    if unknown or not selected:
        raise typer.BadParameter(
            f"unknown implementations: {', '.join(unknown) or 'none'}"
        )
    genome_mode = genome_dir is not None
    synthetic_touched = (
        powers is not None
        or ani_levels is not None
        or size_ratios is not None
        or trials != 8
        or seed != 42
    )
    if genome_mode and synthetic_touched:
        raise typer.BadParameter(
            "genome-dir rejects synthetic options "
            "(--power, --ani, --size-ratio, --trials, --seed)"
        )
    if not genome_mode and ani_min_aligned_fraction != 0.15:
        raise typer.BadParameter(
            "--ani-min-aligned-fraction applies to --genome-dir runs only; "
            "synthetic pairs carry their own realised ANI"
        )
    powers = powers or list(DEFAULT_POWERS)
    ani_levels = ani_levels or list(DEFAULT_ANI_LEVELS)
    size_ratios = size_ratios or list(DEFAULT_SIZE_RATIOS)
    if genome_mode:
        genome_dir = project_path(genome_dir)
        if not genome_dir.is_dir():
            raise typer.BadParameter(f"genome directory not found: {genome_dir}")
        found = discover_genomes(genome_dir)
        if len(found) < genome_count:
            raise typer.BadParameter(
                f"genome directory holds {len(found)} FASTX files, "
                f"fewer than genome-count={genome_count}"
            )
    else:
        if not powers or not ani_levels or not size_ratios:
            raise typer.BadParameter(
                "powers, ANI levels, and size ratios must not be empty"
            )
        for name, values in (
            ("power", powers),
            ("ANI", ani_levels),
            ("size ratio", size_ratios),
        ):
            if len(values) != len(set(values)):
                raise typer.BadParameter(f"{name} values must be unique")
        for power in powers:
            mutation_counts = [round((1.0 - ani) * (1 << power)) for ani in ani_levels]
            if len(mutation_counts) != len(set(mutation_counts)):
                raise typer.BadParameter(
                    f"ANI targets collapse to the same mutation count at power {power}; "
                    "use a longer sequence or fewer ANI targets"
                )

    output = project_path(output)
    build_dir = project_path(build_dir)
    bbtools_jar = project_path(bbtools_jar)
    java_source = ROOT / "benchmarks/BBToolsPairwiseAccuracy.java"
    if not bbtools_jar.is_file():
        raise typer.BadParameter(f"BBTools jar does not exist: {bbtools_jar}")

    run(
        [
            "meson",
            "compile",
            "-C",
            str(build_dir),
            "cuddl-pairwise-accuracy",
            "rabbitsketch-pairwise-accuracy",
        ]
    )
    cuddl_benchmark = build_dir / "benchmarks/cuddl-pairwise-accuracy"
    if not cuddl_benchmark.is_file():
        raise typer.BadParameter(f"cuDDL benchmark was not built: {cuddl_benchmark}")

    with tempfile.TemporaryDirectory(prefix="cuddl-pairwise-accuracy-") as temporary:
        temporary_dir = Path(temporary)
        if genome_mode:
            cases_csv = write_genome_cases(temporary_dir, found, seed, genome_count)
        else:
            cases_csv = write_cases(
                temporary_dir, powers, ani_levels, size_ratios, trials, seed
            )
        cuddl_json = temporary_dir / "cuddl.json"
        cuddl_csv = temporary_dir / "cuddl.csv"
        classes = temporary_dir / "classes"
        classes.mkdir()
        run(
            [
                str(cuddl_benchmark),
                "--cases",
                str(cases_csv),
                "--output",
                str(cuddl_json),
                "--cuco-hll",
            ]
        )
        cuddl_result = benchmark_schema.load_result(
            cuddl_json, operation="pairwise_accuracy"
        )
        aligned_fraction: dict[tuple[str, str], float] = {}
        ani_truth: dict[tuple[str, str], tuple[float, float]] = {}
        if genome_mode:
            # ANI truth is ANIm, nucmer alignment of every pair that shares enough k-mers to
            # be scored; cub-exact stays the oracle of the set metrics. Paths never reach the
            # normalised case, so rows match on SHA-256 and look their paths up in the cases CSV.
            run(["ninja", "-C", str(build_dir), "subprojects/mummer-4.0.1/nucmer"])
            mummer = build_dir / "subprojects/mummer-4.0.1"
            with cases_csv.open(newline="") as handle:
                case_rows = list(csv.DictReader(handle))
            truth_genomes = sorted(
                {r["reference_path"] for r in case_rows}
                | {r["query_path"] for r in case_rows}
            )
            reference_paths: dict[str, str] = {}
            query_paths: dict[str, str] = {}
            for r in case_rows:
                reference_paths.setdefault(r["reference_sha256"], r["reference_path"])
                query_paths.setdefault(r["query_sha256"], r["query_path"])
            # cuDDL's rows carry the exact containment of both orientations.
            candidates = {
                tuple(
                    sorted(
                        (
                            reference_paths[row["case"]["reference_sha256"]],
                            query_paths[row["case"]["query_sha256"]],
                        )
                    )
                )
                for row in cuddl_result["measurements"]
                if row["metrics"]["exact_containment"] >= MIN_ALIGNABLE_CONTAINMENT
            }
            digests = {path: sha for sha, path in reference_paths.items()}
            ani_truth = anim_table(
                mummer / "nucmer",
                mummer / "delta-filter",
                truth_genomes,
                digests,
                candidates,
                project_path(anim_cache),
                temporary_dir,
                threads,
            )
            for row in cuddl_result["measurements"]:
                task = row["case"]
                ref = reference_paths[task["reference_sha256"]]
                qry = query_paths[task["query_sha256"]]
                ani, af = ani_truth[(qry, ref)]
                aligned_fraction[(task["reference_sha256"], task["query_sha256"])] = af
                row["case"]["actual_ani"] = ani
                row["case"]["requested_ani"] = ani
                row["metrics"]["exact_ani"] = ani
                row["metrics"]["ani_signed_error"] = row["metrics"]["sketch_ani"] - ani
                row["metrics"]["ani_absolute_error"] = abs(
                    row["metrics"]["ani_signed_error"]
                )
                row["metrics"]["aligned_fraction"] = af
        hll_measurements = [
            row
            for row in cuddl_result["measurements"]
            if row["implementation"]["name"] == "cuco_hll"
        ]
        cuddl_result["measurements"] = [
            row
            for row in cuddl_result["measurements"]
            if row["implementation"]["name"] == "cuddl"
        ]
        hll_rows = benchmark_schema.flatten_measurements(
            {"measurements": hll_measurements}
        )
        original_rows = benchmark_schema.flatten_measurements(cuddl_result)
        if len(hll_rows) != len(original_rows) or any(
            any(hll[field] != original[field] for field in KEY_FIELDS)
            for hll, original in zip(hll_rows, original_rows, strict=True)
        ):
            raise RuntimeError("cuco HLL case metadata differs from cuDDL")
        benchmark_schema.write_result(cuddl_json, cuddl_result)
        rows = [
            {
                field: (
                    "1"
                    if flattened.get(field) is True
                    else "0"
                    if flattened.get(field) is False
                    else str(flattened[field])
                    if field in flattened
                    else ""
                )
                for field in CSV_FIELDS
            }
            for flattened in benchmark_schema.flatten_measurements(cuddl_result)
        ]
        if not rows:
            raise RuntimeError("cuDDL emitted no measurements")
        if {row["implementation"] for row in rows} != {"cuddl"}:
            raise RuntimeError("cuDDL emitted an unexpected implementation")
        with cuddl_csv.open("w", newline="") as destination:
            writer = csv.DictWriter(destination, fieldnames=CSV_FIELDS)
            writer.writeheader()
            writer.writerows(rows)

        run(
            [
                "javac",
                "-cp",
                str(bbtools_jar),
                "-d",
                str(classes),
                str(java_source),
            ]
        )
        java = run(
            [
                "java",
                "-cp",
                os.pathsep.join((str(classes), str(bbtools_jar))),
                "BBToolsPairwiseAccuracy",
                str(cuddl_csv),
                str(threads),
            ],
            capture=True,
        )
        reference_fields, reference = read_csv(io.StringIO(java.stdout))
        rabbit_json = temporary_dir / "rabbitsketch.json"
        run(
            [
                str(build_dir / "benchmarks/rabbitsketch-pairwise-accuracy"),
                "--cases",
                str(cuddl_json),
                "--output",
                str(rabbit_json),
            ]
        )
        rabbit_result = benchmark_schema.load_result(rabbit_json, "pairwise_accuracy")
        rabbit_rows = benchmark_schema.flatten_measurements(rabbit_result)
        original_rows = benchmark_schema.flatten_measurements(cuddl_result)
        if len(rabbit_rows) != len(original_rows):
            raise RuntimeError("RabbitSketch emitted a different number of cases")
        for index, (original, rabbit) in enumerate(
            zip(original_rows, rabbit_rows, strict=True)
        ):
            if rabbit["implementation"] != "rabbitsketch" or any(
                original[field] != rabbit[field] for field in KEY_FIELDS
            ):
                raise RuntimeError(f"RabbitSketch case metadata differs at row {index}")
        dashing2_rows: list = []
        dashing2_result = None
        skani_rows: list = []
        skani_result = None
        cub_rows: list = []
        cub_result = None
        if "dashing2" in selected:
            dashing2_bin = build_dir / "subprojects/dashing2/dashing2"
            if not dashing2_bin.is_file():
                raise typer.BadParameter(
                    f"Dashing2 binary was not built: {dashing2_bin}"
                )
            dashing2_json = temporary_dir / "dashing2.json"
            run(
                [
                    str(ROOT / "benchmarks/dashing2_pairwise_accuracy.py"),
                    str(cases_csv),
                    str(cuddl_json),
                    str(dashing2_json),
                    str(dashing2_bin),
                ]
            )
            dashing2_result = benchmark_schema.load_result(
                dashing2_json, "pairwise_accuracy"
            )
            dashing2_rows = benchmark_schema.flatten_measurements(dashing2_result)
            if len(dashing2_rows) != len(original_rows):
                raise RuntimeError("Dashing2 emitted a different number of cases")
            for index, (original, dashing2) in enumerate(
                zip(original_rows, dashing2_rows, strict=True)
            ):
                if dashing2["implementation"] != "dashing2" or any(
                    original[field] != dashing2[field] for field in KEY_FIELDS
                ):
                    raise RuntimeError(f"Dashing2 case metadata differs at row {index}")
        skani_skipped = 0
        skani_bin = build_dir / "subprojects/skani/skani"
        if "skani" in selected:
            if not skani_bin.is_file():
                raise typer.BadParameter(f"skani binary was not built: {skani_bin}")
            skani_json = temporary_dir / "skani.json"
            run(
                [
                    str(ROOT / "benchmarks/skani_pairwise_accuracy.py"),
                    str(cases_csv),
                    str(cuddl_json),
                    str(skani_json),
                    str(skani_bin),
                    str(threads),
                ]
            )
            skani_result = benchmark_schema.load_result(skani_json, "pairwise_accuracy")
            skani_rows = benchmark_schema.flatten_measurements(skani_result)
            # ANI-only lane on synthetic pairs below the detection floor:
            # subset match on KEY_FIELDS, not row-count equality.
            matches_case = case_matcher(original_rows)
            for index, skani in enumerate(skani_rows):
                if skani["implementation"] != "skani":
                    raise RuntimeError(
                        f"skani emitted an unexpected implementation at row {index}"
                    )
                if not matches_case(skani):
                    raise RuntimeError(f"skani case metadata differs at row {index}")
            skani_skipped = len(original_rows) - len(skani_rows)
            if skani_skipped:
                typer.echo(
                    f"skani skipped {skani_skipped} of {len(original_rows)} "
                    "pairs below the detection floor"
                )
            if genome_mode:
                # Rows match their ANIm truth on SHA-256, as cuDDL's do.
                with cases_csv.open(newline="") as handle:
                    sha_paths = {
                        (r["reference_sha256"], r["query_sha256"]): (
                            r["query_path"],
                            r["reference_path"],
                        )
                        for r in csv.DictReader(handle)
                    }
                for row in skani_result["measurements"]:
                    task = row["case"]
                    key = sha_paths[(task["reference_sha256"], task["query_sha256"])]
                    ani, af = ani_truth[key]
                    row["metrics"]["exact_ani"] = ani
                    row["metrics"]["ani_signed_error"] = (
                        row["metrics"]["sketch_ani"] - ani
                    )
                    row["metrics"]["ani_absolute_error"] = abs(
                        row["metrics"]["ani_signed_error"]
                    )
                    row["metrics"]["aligned_fraction"] = af
                skani_rows = benchmark_schema.flatten_measurements(skani_result)
        ani_lane_results: dict[str, dict] = {}
        ani_lane_rows: dict[str, list] = {}
        if "hypergen" in selected:
            hypergen_bin = build_dir / "subprojects/hypergen/hyper-gen"
            if not hypergen_bin.is_file():
                raise typer.BadParameter(
                    f"hyper-gen binary was not built: {hypergen_bin}"
                )
            hypergen_json = temporary_dir / "hypergen.json"
            run(
                [
                    str(ROOT / "benchmarks/hypergen_pairwise_accuracy.py"),
                    str(cases_csv),
                    str(cuddl_json),
                    str(hypergen_json),
                    str(hypergen_bin),
                    str(threads),
                ]
            )
            ani_lane_results["hypergen"] = benchmark_schema.load_result(
                hypergen_json, "pairwise_accuracy"
            )
        jaccard_lanes = {
            "simdsketch": (
                "simdsketch-benchmark",
                build_dir / "benchmarks/simdsketch-benchmark",
            ),
            "sketchlib": (
                "subprojects/sketchlib/sketchlib",
                build_dir / "subprojects/sketchlib/sketchlib",
            ),
            "pp-sketchlib": (
                "ppsketchlib-micro",
                build_dir / "benchmarks/ppsketchlib-micro",
            ),
        }
        for lane, (target, lane_bin) in jaccard_lanes.items():
            if lane not in selected:
                continue
            run(
                ["meson", "compile", "-C", str(build_dir), target]
                if lane in ("simdsketch", "pp-sketchlib")
                else ["ninja", "-C", str(build_dir), target]
            )
            lane_json = temporary_dir / f"{lane}.json"
            run(
                [
                    str(ROOT / "benchmarks/jaccard_pairwise_accuracy.py"),
                    lane,
                    str(cases_csv),
                    str(cuddl_json),
                    str(lane_json),
                    str(lane_bin),
                    str(threads),
                ]
            )
            ani_lane_results[lane] = benchmark_schema.load_result(
                lane_json, "pairwise_accuracy"
            )
        if ani_lane_results and genome_mode:
            with cases_csv.open(newline="") as handle:
                ani_sha_paths = {
                    (r["reference_sha256"], r["query_sha256"]): (
                        r["query_path"],
                        r["reference_path"],
                    )
                    for r in csv.DictReader(handle)
                }
        matches_case = case_matcher(original_rows)
        for lane, lane_result in ani_lane_results.items():
            # ANI lanes emit one row per ordered pair, not per orientation.
            # Every row must match a cuDDL case on shared fields.
            for index, lane_row in enumerate(
                benchmark_schema.flatten_measurements(lane_result)
            ):
                if lane_row["implementation"] != lane:
                    raise RuntimeError(
                        f"{lane} emitted an unexpected implementation at row {index}"
                    )
                if not matches_case(lane_row):
                    raise RuntimeError(f"{lane} case metadata differs at row {index}")
            if genome_mode:
                for row in lane_result["measurements"]:
                    task = row["case"]
                    key = ani_sha_paths[
                        (task["reference_sha256"], task["query_sha256"])
                    ]
                    ani, af = ani_truth[key]
                    row["metrics"]["exact_ani"] = ani
                    row["metrics"]["ani_signed_error"] = (
                        row["metrics"]["sketch_ani"] - ani
                    )
                    row["metrics"]["ani_absolute_error"] = abs(
                        row["metrics"]["ani_signed_error"]
                    )
                    row["metrics"]["aligned_fraction"] = af
                    if "core_ani" in row["metrics"]:
                        row["metrics"]["core_ani_absolute_error"] = abs(
                            row["metrics"]["core_ani"] - ani
                        )
            ani_lane_rows[lane] = benchmark_schema.flatten_measurements(lane_result)
        if "cub-exact" in selected:
            cub_bin = build_dir / "benchmarks/cub-exact-pairwise"
            if not cub_bin.is_file():
                raise typer.BadParameter(f"cub-exact binary was not built: {cub_bin}")
            cub_json = temporary_dir / "cub.json"
            run(
                [
                    str(ROOT / "benchmarks/cub_pairwise_accuracy.py"),
                    str(cases_csv),
                    str(cuddl_json),
                    str(cub_json),
                    str(cub_bin),
                ]
            )
            cub_result = benchmark_schema.load_result(cub_json, "pairwise_accuracy")
            cub_rows = benchmark_schema.flatten_measurements(cub_result)
            if len(cub_rows) != len(original_rows):
                raise RuntimeError("cub-exact emitted a different number of cases")
            for index, (original, cub) in enumerate(
                zip(original_rows, cub_rows, strict=True)
            ):
                if cub["implementation"] != "cub-exact" or any(
                    original[field] != cub[field]
                    for field in CASE_FIELDS
                    if field in original and field in cub
                ):
                    raise RuntimeError(
                        f"cub-exact case metadata differs at row {index}"
                    )

    if reference_fields != list(CSV_FIELDS):
        raise RuntimeError("BBTools emitted a different CSV schema")
    if len(reference) != len(rows):
        raise RuntimeError(
            f"BBTools emitted {len(reference)} rows for {len(rows)} cuDDL rows"
        )
    if {row["implementation"] for row in reference} != {"bbtools"}:
        raise RuntimeError("BBTools emitted an unexpected implementation")

    for index, (gpu, cpu) in enumerate(zip(rows, reference, strict=True), start=1):
        if any(gpu[field] != cpu[field] for field in KEY_FIELDS):
            raise RuntimeError(f"case metadata differs at row {index}")
        for implementation, row in (("cuDDL", gpu), ("BBTools", cpu)):
            count_sum = sum(int(row[field]) for field in COUNT_FIELDS)
            if count_sum != int(row["buckets"]):
                raise RuntimeError(
                    f"{implementation} counts do not sum to buckets at row {index}"
                )
    typed_rows = [coerce_row(row) for row in (*rows, *reference)]
    base_case = CASE_FIELDS
    cuddl_variants = (
        (
            "cuddl",
            "sketch_cardinality",
            "cardinality_signed_error",
            "cardinality_absolute_error",
            "cardinality_relative_error",
            "cardinality_absolute_relative_error",
        ),
        (
            "cuddl-bbtools",
            "sketch_cardinality_bbtools",
            "cardinality_bbtools_signed_error",
            "cardinality_bbtools_absolute_error",
            "cardinality_bbtools_relative_error",
            "cardinality_bbtools_absolute_relative_error",
        ),
        (
            "cuddl-paper",
            "sketch_cardinality_paper",
            "cardinality_paper_signed_error",
            "cardinality_paper_absolute_error",
            "cardinality_paper_relative_error",
            "cardinality_paper_absolute_relative_error",
        ),
        # Minimum MLE without the cap at the offered k-mer count, to show what the cap buys.
        (
            "cuddl-uncapped",
            "sketch_cardinality_uncapped",
            "cardinality_uncapped_signed_error",
            "cardinality_uncapped_absolute_error",
            "cardinality_uncapped_relative_error",
            "cardinality_uncapped_absolute_relative_error",
        ),
    )
    result = benchmark_schema.make_result(
        name="Pairwise sketch accuracy",
        operation="pairwise_accuracy",
        scope="end_to_end",
        datasets={},
        system=cuddl_result["system"],
        measurements=[],
    )
    for variant, sketch_col, signed_col, abs_col, rel_col, absrel_col in cuddl_variants:
        variant_rows = []
        for row in typed_rows:
            if row["implementation"] != "cuddl":
                continue
            renamed = dict(row)
            renamed["implementation"] = variant
            renamed["sketch_cardinality"] = row[sketch_col]
            renamed["cardinality_signed_error"] = row[signed_col]
            renamed["cardinality_absolute_error"] = row[abs_col]
            renamed["cardinality_relative_error"] = row[rel_col]
            renamed["cardinality_absolute_relative_error"] = row[absrel_col]
            variant_rows.append(renamed)
        normalized_variants = benchmark_schema.measurements_from_rows(
            variant_rows,
            case_fields=base_case,
            omit_fields=PATH_FIELDS,
        )
        for normalized, original in zip(normalized_variants, variant_rows, strict=True):
            normalized["implementation"] = {"name": original["implementation"]}
        result["measurements"].extend(normalized_variants)
    for row in typed_rows:
        if row["implementation"] == "bbtools":
            result["measurements"].extend(
                benchmark_schema.measurements_from_rows(
                    [row],
                    case_fields=base_case,
                    omit_fields=PATH_FIELDS,
                )
            )
    rabbit_measurements = benchmark_schema.measurements_from_rows(
        rabbit_rows,
        case_fields=(*CASE_FIELDS, "hash_seed", "sketch_size"),
        omit_fields=PATH_FIELDS,
    )
    for normalized, original in zip(
        rabbit_measurements, rabbit_result["measurements"], strict=True
    ):
        normalized["implementation"] = original["implementation"]
    result["measurements"].extend(rabbit_measurements)
    if "dashing2" in selected:
        assert dashing2_result is not None  # set when the dashing2 lane ran
        dashing2_measurements = benchmark_schema.measurements_from_rows(
            dashing2_rows,
            case_fields=(*CASE_FIELDS, "sketch_size"),
            omit_fields=PATH_FIELDS,
        )
        for normalized, original in zip(
            dashing2_measurements, dashing2_result["measurements"], strict=True
        ):
            normalized["implementation"] = original["implementation"]
        result["measurements"].extend(dashing2_measurements)
    for lane, lane_rows, lane_result in (
        ("skani", skani_rows, skani_result if "skani" in selected else None),
        *(
            (lane, ani_lane_rows[lane], ani_lane_results[lane])
            for lane in ani_lane_results
        ),
        ("cub-exact", cub_rows, cub_result if "cub-exact" in selected else None),
    ):
        if lane not in selected:
            continue
        assert lane_result is not None  # set when the lane ran
        lane_measurements = benchmark_schema.measurements_from_rows(
            lane_rows,
            case_fields=CASE_FIELDS,
            omit_fields=PATH_FIELDS,
        )
        for normalized, original in zip(
            lane_measurements, lane_result["measurements"], strict=True
        ):
            normalized["implementation"] = original["implementation"]
        result["measurements"].extend(lane_measurements)
    normalized_hll = benchmark_schema.measurements_from_rows(
        hll_rows,
        case_fields=(*CASE_FIELDS, "hll_precision"),
        omit_fields=PATH_FIELDS,
    )
    for normalized, original in zip(normalized_hll, hll_measurements, strict=True):
        normalized["implementation"] = original["implementation"]
    result["measurements"].extend(normalized_hll)
    # Exact k-mer set truth per pair, from cuDDL's rows, for the exact-set ANI of each formula.
    exact_sets: dict[tuple[str, str], tuple[int, int, int, int]] = {}
    for measurement in [*result["measurements"]]:
        if measurement["implementation"]["name"] != "cuddl":
            continue
        case = measurement["case"]
        exact_sets.setdefault(
            (case["reference_sha256"], case["query_sha256"]),
            (
                case["intersection"],
                case["left_cardinality"],
                case["right_cardinality"],
                case["k"],
            ),
        )
        if case["orientation"] != "query_to_reference":
            continue
        for lane, formula in EXACT_SET_LANES.items():
            ani = formula(*exact_sets[(case["reference_sha256"], case["query_sha256"])])
            exact = measurement["metrics"]["exact_ani"]
            # The case geometry and alignment that the cuDDL row carries among its metrics.
            carried = {
                key: measurement["metrics"][key]
                for key in ("size_ratio", "aligned_fraction")
                if key in measurement["metrics"]
            }
            result["measurements"].append(
                {
                    "implementation": {"name": lane},
                    "case": dict(case),
                    "metrics": {
                        "exact_ani": exact,
                        "sketch_ani": ani,
                        "ani_signed_error": ani - exact,
                        "ani_absolute_error": abs(ani - exact),
                        **carried,
                    },
                }
            )
    if genome_mode:
        # A set-derived ANI over a pair that barely aligns measures the k-th root's noise floor,
        # not identity. Those rows keep their estimates and lose their ANI error columns, so
        # every aggregate over ani_absolute_error is an aggregate over related pairs.
        for measurement in result["measurements"]:
            case = measurement["case"]
            case["case_source"] = "refseq"
            aligned = aligned_fraction[(case["reference_sha256"], case["query_sha256"])]
            metrics = measurement["metrics"]
            metrics["ani_scored"] = aligned >= ani_min_aligned_fraction
            if metrics["ani_scored"]:
                continue
            for field in (
                "ani_signed_error",
                "ani_absolute_error",
                "core_ani_absolute_error",
            ):
                metrics.pop(field, None)
        scored = sum(
            1
            for measurement in result["measurements"]
            if measurement["metrics"]["ani_scored"]
        )
        typer.echo(
            f"ANI scored on {scored} of {len(result['measurements'])} rows; the rest are "
            f"below the {ani_min_aligned_fraction:.0%} aligned-fraction floor"
        )
    # ANI errors only exist where ANI is scored. Each splits by the truth's ANI band, and each
    # lane with a k-mer formula also reports its error against that formula over exact sets.
    for measurement in result["measurements"]:
        metrics = measurement["metrics"]
        if "ani_absolute_error" not in metrics:
            continue
        band = "ge95" if metrics["exact_ani"] >= ANI_BIN_EDGE else "lt95"
        metrics[f"ani_{band}_absolute_error"] = metrics["ani_absolute_error"]
        formula = ANI_FORMULAS.get(measurement["implementation"]["name"])
        if formula is None:
            continue
        case = measurement["case"]
        exact = formula(*exact_sets[(case["reference_sha256"], case["query_sha256"])])
        metrics["exact_formula_ani"] = exact
        metrics["ani_sketch_signed_error"] = metrics["sketch_ani"] - exact
        metrics["ani_sketch_absolute_error"] = abs(metrics["sketch_ani"] - exact)
    pairs = result["measurements"]
    summaries = error_summaries(pairs)
    result["measurements"] = [*pairs, *summaries] if pair_rows else summaries
    benchmark_schema.write_result(output, result)
    typer.echo(
        f"Saved {len(summaries)} error summaries over {len(pairs)} rows to {output}"
        + ("" if pair_rows else ", without the rows")
    )


if __name__ == "__main__":
    app()
