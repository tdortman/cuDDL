#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = ["typer"]
# ///
"""Run cuDDL and BBTools pairwise accuracy cases into one CSV."""

import csv
import hashlib
import io
import os
import shlex
import subprocess
import tempfile
from pathlib import Path
from typing import Annotated, TextIO

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
)
PATH_FIELDS = {"reference_path", "query_path"}
PUBLIC_FIELDS = tuple(field for field in CSV_FIELDS if field not in PATH_FIELDS)
KEY_FIELDS = (
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
)
COUNT_FIELDS = ("lower", "equal", "higher", "both_empty")
app = typer.Typer(
    help="Run both pairwise accuracy implementations on deterministic raw DNA.",
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
    return reader.fieldnames or [], list(reader)


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
        Path, typer.Option(dir_okay=False, help="Combined accuracy CSV output")
    ] = Path("results/pairwise-accuracy.csv"),
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
) -> None:
    """Build and run both implementations, then publish one CSV."""
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

    run(["meson", "compile", "-C", str(build_dir), "cuddl-pairwise-accuracy"])
    cuddl_benchmark = build_dir / "benchmarks/cuddl-pairwise-accuracy"
    if not cuddl_benchmark.is_file():
        raise typer.BadParameter(f"cuDDL benchmark was not built: {cuddl_benchmark}")

    with tempfile.TemporaryDirectory(prefix="cuddl-pairwise-accuracy-") as temporary:
        temporary_dir = Path(temporary)
        cases_csv = write_cases(
            temporary_dir, powers, ani_levels, size_ratios, trials, seed
        )
        cuddl_csv = temporary_dir / "cuddl.csv"
        classes = temporary_dir / "classes"
        classes.mkdir()
        run(
            [
                str(cuddl_benchmark),
                "--cases",
                str(cases_csv),
                "--csv",
                str(cuddl_csv),
            ]
        )
        with cuddl_csv.open(newline="") as source:
            fields, rows = read_csv(source)
        if fields != list(CSV_FIELDS):
            raise RuntimeError("cuDDL emitted an unexpected CSV schema")
        if not rows:
            raise RuntimeError("cuDDL emitted no rows")
        if {row["implementation"] for row in rows} != {"cuddl"}:
            raise RuntimeError("cuDDL emitted an unexpected implementation")

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
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", newline="", dir=output.parent, prefix=output.name, delete=False
    ) as destination:
        temporary_output = Path(destination.name)
        writer = csv.DictWriter(destination, fieldnames=PUBLIC_FIELDS)
        writer.writeheader()
        for row in (*rows, *reference):
            writer.writerow({field: row[field] for field in PUBLIC_FIELDS})
    temporary_output.replace(output)

    typer.echo(f"Verified {len(rows)} matched raw-DNA pair rows")
    typer.echo(f"Saved {len(rows) + len(reference)} rows to {output}")


if __name__ == "__main__":
    app()
