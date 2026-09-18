#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = ["jsonschema", "typer"]
# ///
"""Run cuDDL, BBTools, RabbitSketch, and cuco HLL accuracy cases into one JSON result."""

import csv
import hashlib
import io
import os
import shlex
import subprocess
import tempfile
from pathlib import Path
from typing import Annotated, TextIO

import benchmark_schema
import typer

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
    "skani_aligned_fraction",
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
    "skani_aligned_fraction",
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


def skani_triangle(
    skani_bin: Path, genomes: list[str], work: Path, threads: int
) -> dict[tuple[str, str], tuple[float, float]]:
    """Map (query, reference) path pair to (ANI fraction, aligned fraction)."""
    out = work / "skani-tri.tsv"
    proc = subprocess.run(
        [str(skani_bin), "triangle", *genomes, "-o", str(out), "-t", str(threads)],
        cwd=ROOT,
        check=True,
        text=True,
        capture_output=True,
    )
    _ = proc
    rows = out.read_text().splitlines()
    names = [rows[i].split("\t")[0] for i in range(1, len(rows))]
    af_rows = (work / "skani-tri.tsv.af").read_text().splitlines()
    table: dict[tuple[str, str], tuple[float, float]] = {}
    for i in range(len(names)):
        ani_vals = [float(x) for x in rows[1 + i].split("\t")[1:]]
        af_vals = [float(x) for x in af_rows[1 + i].split("\t")[1:]]
        for j in range(i):
            left, right = names[i], names[j]
            table[(left, right)] = (ani_vals[j] / 100.0, af_vals[j] / 100.0)
            table[(right, left)] = table[(left, right)]
    for name in names:
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
    import random

    def base_count(path: str) -> int:
        opener = gzip.open if path.endswith(".gz") else open
        total = 0
        with opener(path, "rt") as handle:
            for line in handle:
                if line and not line.startswith(">"):
                    total += len(line.strip())
        return total

    sampled = sorted(random.Random(seed).sample([str(p) for p in genomes], count))
    bases = {path: base_count(path) for path in sampled}
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
                        "reference_sha256": hashlib.sha256(
                            Path(reference).read_bytes()).hexdigest(),
                        "query_sha256": hashlib.sha256(
                            Path(query).read_bytes()).hexdigest(),
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
            "cuco_hll, dashing2, skani, hypergen, cub-exact. cuDDL and BBTools "
            "always run, they define cases and the CSV all lanes validate against."
        ),
    ] = "cuddl,bbtools,rabbitsketch,cuco_hll,dashing2,skani,hypergen,cub-exact",
) -> None:
    """Build and run all implementations, then publish one JSON result."""
    selected = [name.strip() for name in implementations.split(",") if name.strip()]
    unknown = [name for name in selected if name not in
               ("cuddl", "bbtools", "rabbitsketch", "cuco_hll", "dashing2",
                "skani", "hypergen", "cub-exact")]
    if unknown or not selected:
        raise typer.BadParameter(
            f"unknown implementations: {', '.join(unknown) or 'none'}")
    genome_mode = genome_dir is not None
    synthetic_touched = (
        powers is not None or ani_levels is not None
        or size_ratios is not None or trials != 8 or seed != 42
    )
    if genome_mode and synthetic_touched:
        raise typer.BadParameter(
            "genome-dir rejects synthetic options "
            "(--power, --ani, --size-ratio, --trials, --seed)")
    if not genome_mode and ani_min_aligned_fraction != 0.15:
        raise typer.BadParameter(
            "--ani-min-aligned-fraction applies to --genome-dir runs only; "
            "synthetic pairs carry their own realised ANI"
        )
    if genome_mode:
        genome_dir = project_path(genome_dir)
        if not genome_dir.is_dir():
            raise typer.BadParameter(f"genome directory not found: {genome_dir}")
        found = discover_genomes(genome_dir)
        if len(found) < genome_count:
            raise typer.BadParameter(
                f"genome directory holds {len(found)} FASTX files, "
                f"fewer than genome-count={genome_count}")
    else:
        powers = powers or list(DEFAULT_POWERS)
        ani_levels = ani_levels or list(DEFAULT_ANI_LEVELS)
        size_ratios = size_ratios or list(DEFAULT_SIZE_RATIOS)
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
            cases_csv = write_genome_cases(
                temporary_dir, found, seed, genome_count)
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
        if genome_mode:
            skani_bin = build_dir / "subprojects/skani/skani"
            if not skani_bin.is_file():
                raise typer.BadParameter(
                    f"skani binary was not built: {skani_bin}")
            with cases_csv.open(newline="") as handle:
                case_rows = list(csv.DictReader(handle))
            tri_genomes = sorted(
                {r["reference_path"] for r in case_rows} |
                {r["query_path"] for r in case_rows})
            genome_tri = skani_triangle(
                skani_bin, tri_genomes, temporary_dir, threads)
            for row in cuddl_result["measurements"]:
                task = row["case"]
                ref = next(r["reference_path"] for r in case_rows
                           if r["reference_sha256"] == task["reference_sha256"])
                qry = next(r["query_path"] for r in case_rows
                           if r["query_sha256"] == task["query_sha256"])
                key = (qry, ref)
                if key not in genome_tri:
                    raise RuntimeError(f"skani triangle lacks pair {key}")
                ani, af = genome_tri[key]
                aligned_fraction[
                    (task["reference_sha256"], task["query_sha256"])] = af
                row["case"]["actual_ani"] = ani
                row["case"]["requested_ani"] = ani
                row["metrics"]["exact_ani"] = ani
                row["metrics"]["ani_signed_error"] = (
                    row["metrics"]["sketch_ani"] - ani)
                row["metrics"]["ani_absolute_error"] = abs(
                    row["metrics"]["ani_signed_error"])
                row["metrics"]["skani_aligned_fraction"] = af
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
        hypergen_rows: list = []
        hypergen_result = None
        cub_rows: list = []
        cub_result = None
        if "dashing2" in selected:
            dashing2_bin = build_dir / "subprojects/dashing2/dashing2"
            if not dashing2_bin.is_file():
                raise typer.BadParameter(
                    f"Dashing2 binary was not built: {dashing2_bin}")
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
                dashing2_json, "pairwise_accuracy")
            dashing2_rows = benchmark_schema.flatten_measurements(dashing2_result)
            if len(dashing2_rows) != len(original_rows):
                raise RuntimeError(
                    "Dashing2 emitted a different number of cases")
            for index, (original, dashing2) in enumerate(
                zip(original_rows, dashing2_rows, strict=True)
            ):
                if dashing2["implementation"] != "dashing2" or any(
                    original[field] != dashing2[field] for field in KEY_FIELDS
                ):
                    raise RuntimeError(
                        f"Dashing2 case metadata differs at row {index}")
        skani_skipped = 0
        genome_tri: dict[tuple[str, str], tuple[float, float]] = {}
        skani_bin = build_dir / "subprojects/skani/skani"
        if "skani" in selected:
            if not skani_bin.is_file():
                raise typer.BadParameter(
                    f"skani binary was not built: {skani_bin}")
            if genome_mode:
                with cases_csv.open(newline="") as handle:
                    case_rows = list(csv.DictReader(handle))
                tri_genomes = sorted(
                    {r["reference_path"] for r in case_rows} |
                    {r["query_path"] for r in case_rows})
                genome_tri = skani_triangle(
                    skani_bin, tri_genomes, temporary_dir, threads)
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
            skani_result = benchmark_schema.load_result(
                skani_json, "pairwise_accuracy")
            skani_rows = benchmark_schema.flatten_measurements(skani_result)
            # ANI-only lane on synthetic pairs below the detection floor:
            # subset match on KEY_FIELDS, not row-count equality.
            for index, skani in enumerate(skani_rows):
                if skani["implementation"] != "skani":
                    raise RuntimeError(
                        f"skani emitted an unexpected implementation at row {index}")
                match = [o for o in original_rows if all(
                    o[field] == skani[field] for field in CASE_FIELDS
                    if field in o and field in skani)]
                if not match:
                    raise RuntimeError(
                        f"skani case metadata differs at row {index}")
            skani_skipped = len(original_rows) - len(skani_rows)
            if skani_skipped:
                typer.echo(
                    f"skani skipped {skani_skipped} of {len(original_rows)} "
                    "pairs below the detection floor")
            if genome_mode:
                # Split truth: skani triangle is the ANI oracle on real
                # genomes. cub-exact stays the set-metric oracle. Paths
                # never reach the normalized case; match on sha256 and
                # look up triangle paths from the cases CSV.
                with cases_csv.open(newline="") as handle:
                    sha_paths = {
                        (r["reference_sha256"], r["query_sha256"]): (
                            r["query_path"], r["reference_path"])
                        for r in csv.DictReader(handle)}
                for row in skani_result["measurements"]:
                    task = row["case"]
                    key = sha_paths[(
                        task["reference_sha256"], task["query_sha256"])]
                    if key not in genome_tri:
                        raise RuntimeError(
                            f"skani triangle lacks pair {key}")
                    ani, af = genome_tri[key]
                    row["metrics"]["exact_ani"] = ani
                    row["metrics"]["ani_signed_error"] = (
                        row["metrics"]["sketch_ani"] - ani)
                    row["metrics"]["ani_absolute_error"] = abs(
                        row["metrics"]["ani_signed_error"])
                    row["metrics"]["skani_aligned_fraction"] = af
                skani_rows = benchmark_schema.flatten_measurements(skani_result)
        if "hypergen" in selected:
            hypergen_bin = build_dir / "subprojects/hypergen/hyper-gen"
            if not hypergen_bin.is_file():
                raise typer.BadParameter(
                    f"hyper-gen binary was not built: {hypergen_bin}")
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
            hypergen_result = benchmark_schema.load_result(
                hypergen_json, "pairwise_accuracy")
            hypergen_rows = benchmark_schema.flatten_measurements(hypergen_result)
            # ANI lanes emit one row per triangle pair, not per orientation.
            # Every row must match a cuDDL case on shared fields.
            for index, hypergen in enumerate(hypergen_rows):
                if hypergen["implementation"] != "hypergen":
                    raise RuntimeError(
                        f"hypergen emitted an unexpected implementation at row {index}")
                match = [o for o in original_rows if all(
                    o[field] == hypergen[field] for field in CASE_FIELDS
                    if field in o and field in hypergen)]
                if not match:
                    raise RuntimeError(
                        f"hypergen case metadata differs at row {index}")
            if genome_mode:
                with cases_csv.open(newline="") as handle:
                    hg_sha_paths = {
                        (r["reference_sha256"], r["query_sha256"]): (
                            r["query_path"], r["reference_path"])
                        for r in csv.DictReader(handle)}
                for row in hypergen_result["measurements"]:
                    task = row["case"]
                    key = hg_sha_paths[(
                        task["reference_sha256"], task["query_sha256"])]
                    if key not in genome_tri:
                        raise RuntimeError(
                            f"skani triangle lacks pair {key}")
                    ani, af = genome_tri[key]
                    row["metrics"]["exact_ani"] = ani
                    row["metrics"]["ani_signed_error"] = (
                        row["metrics"]["sketch_ani"] - ani)
                    row["metrics"]["ani_absolute_error"] = abs(
                        row["metrics"]["ani_signed_error"])
                    row["metrics"]["skani_aligned_fraction"] = af
                hypergen_rows = benchmark_schema.flatten_measurements(
                    hypergen_result)
        if "cub-exact" in selected:
            cub_bin = build_dir / "benchmarks/cub-exact-pairwise"
            if not cub_bin.is_file():
                raise typer.BadParameter(
                    f"cub-exact binary was not built: {cub_bin}")
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
            cub_result = benchmark_schema.load_result(
                cub_json, "pairwise_accuracy")
            cub_rows = benchmark_schema.flatten_measurements(cub_result)
            if len(cub_rows) != len(original_rows):
                raise RuntimeError(
                    "cub-exact emitted a different number of cases")
            for index, (original, cub) in enumerate(
                zip(original_rows, cub_rows, strict=True)
            ):
                if cub["implementation"] != "cub-exact" or any(
                    original[field] != cub[field] for field in CASE_FIELDS
                    if field in original and field in cub
                ):
                    raise RuntimeError(
                        f"cub-exact case metadata differs at row {index}")

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
        ("cuddl", "sketch_cardinality", "cardinality_signed_error",
         "cardinality_absolute_error", "cardinality_relative_error",
         "cardinality_absolute_relative_error"),
        ("cuddl-bbtools", "sketch_cardinality_bbtools", "cardinality_bbtools_signed_error",
         "cardinality_bbtools_absolute_error", "cardinality_bbtools_relative_error",
         "cardinality_bbtools_absolute_relative_error"),
        ("cuddl-paper", "sketch_cardinality_paper", "cardinality_paper_signed_error",
         "cardinality_paper_absolute_error", "cardinality_paper_relative_error",
         "cardinality_paper_absolute_relative_error"),
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
        for normalized, original in zip(
            normalized_variants, variant_rows, strict=True
        ):
            normalized["implementation"] = {"name": original["implementation"]}
        result["measurements"].extend(normalized_variants)
    for row in typed_rows:
        if row["implementation"] == "bbtools":
            result["measurements"].extend(benchmark_schema.measurements_from_rows(
                [row],
                case_fields=base_case,
                omit_fields=PATH_FIELDS,
            ))
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
        ("hypergen", hypergen_rows, hypergen_result if "hypergen" in selected else None),
        ("cub-exact", cub_rows, cub_result if "cub-exact" in selected else None),
    ):
        if lane not in selected:
            continue
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
    if genome_mode:
        # A set-derived ANI over a pair that barely aligns measures the k-th root's noise floor,
        # not identity. Those rows keep their estimates and lose their ANI error columns, so
        # every aggregate over ani_absolute_error is an aggregate over related pairs.
        for measurement in result["measurements"]:
            case = measurement["case"]
            case["case_source"] = "refseq"
            aligned = aligned_fraction[
                (case["reference_sha256"], case["query_sha256"])]
            metrics = measurement["metrics"]
            metrics["ani_scored"] = aligned >= ani_min_aligned_fraction
            if metrics["ani_scored"]:
                continue
            for field in ("ani_signed_error", "ani_absolute_error"):
                metrics.pop(field, None)
        scored = sum(
            1 for measurement in result["measurements"]
            if measurement["metrics"]["ani_scored"])
        typer.echo(
            f"ANI scored on {scored} of {len(result['measurements'])} rows; the rest are "
            f"below the {ani_min_aligned_fraction:.0%} aligned-fraction floor"
        )
    benchmark_schema.write_result(output, result)
    typer.echo(f"Saved {len(result['measurements'])} measurements to {output}")


if __name__ == "__main__":
    app()
