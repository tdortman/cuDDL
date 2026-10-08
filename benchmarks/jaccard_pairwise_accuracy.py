#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = []
# ///
"""Replay pairwise accuracy cases through SimdSketch or sketchlib.rust (ANI only).

Both tools estimate Jaccard similarity alone, so neither has cardinality, containment,
completeness, or WKID estimates. Each reports ANI through the Mash Poisson model,
1 + ln(2j / (1 + j)) / k, floored at zero as sketchlib's `ani_pois` does. Every genome is
sketched once and compared all-to-all in one call. Jaccard is symmetric, so each pair yields
one row, as for the other ANI-only lanes.

Usage: jaccard_pairwise_accuracy.py TOOL CASES TRUTH OUTPUT BINARY THREADS
TOOL is simdsketch (the `simdsketch-benchmark` harness, bucket sketch, s=2048, b=8) or
sketchlib (`sketchlib`, s=2048 64-bit bins).
"""

import csv
import json
import math
import struct
import subprocess
import sys
from pathlib import Path

K = 25
SKETCH_SIZE = 2048
CASE_KEYS = (
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
)
VARIANTS = {"simdsketch": "bucket-b8", "sketchlib": "one-perm-minhash"}


def run(cmd: list[str]) -> None:
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"{cmd[0]} failed ({' '.join(cmd)}):\n{proc.stderr[-2000:]}")


def simdsketch_jaccard(
    binary: str, genomes: list[str], work: Path, threads: str
) -> dict[tuple[int, int], float]:
    listing = work / "genomes.txt"
    listing.write_text("".join(f"{path}\n" for path in genomes))
    sketches, out = work / "sketches.bin", work / "jaccard.f32"
    run([binary, "-j", threads, "sketch", "--output", str(sketches), str(listing)])
    run(
        [
            binary,
            "-j",
            threads,
            "compare",
            "--references",
            str(sketches),
            "--output",
            str(out),
        ]
    )
    values = iter(struct.iter_unpack("<f", out.read_bytes()))
    # The harness writes the strict upper triangle row by row.
    return {
        (i, j): next(values)[0]
        for i in range(len(genomes))
        for j in range(i + 1, len(genomes))
    }


def sketchlib_jaccard(
    binary: str, genomes: list[str], work: Path, threads: str
) -> dict[tuple[int, int], float]:
    listing = work / "genomes.tsv"
    listing.write_text("".join(f"g{i}\t{path}\n" for i, path in enumerate(genomes)))
    prefix, out = work / "sketches", work / "dists.tsv"
    run(
        [
            binary,
            "sketch",
            "-f",
            str(listing),
            "-o",
            str(prefix),
            "-k",
            str(K),
            "-s",
            str(SKETCH_SIZE),
            "--threads",
            threads,
            "--quiet",
        ]
    )
    run(
        [
            binary,
            "dist",
            str(prefix),
            "-k",
            str(K),
            "--threads",
            threads,
            "-o",
            str(out),
            "--quiet",
        ]
    )
    similarities = {}
    for line in out.read_text().splitlines():
        if not line.strip():
            continue
        left, right, distance = line.split("\t")
        i, j = sorted((int(left[1:]), int(right[1:])))
        similarities[(i, j)] = 1.0 - float(distance)
    return similarities


def poisson_ani(jaccard: float) -> float:
    if jaccard <= 0.0:
        return 0.0
    return max(0.0, 1.0 + math.log(2.0 * jaccard / (1.0 + jaccard)) / K)


def main(
    tool: str,
    cases_path: str,
    truth_path: str,
    output_path: str,
    binary: str,
    threads: str,
) -> None:
    if tool not in VARIANTS:
        raise RuntimeError(f"unknown tool {tool!r}; expected one of {sorted(VARIANTS)}")
    with open(cases_path, newline="") as handle:
        paths = {
            (row["reference_sha256"], row["query_sha256"]): (
                row["reference_path"],
                row["query_path"],
            )
            for row in csv.DictReader(handle)
        }
    report = json.loads(Path(truth_path).read_text())
    rows = [
        m for m in report["measurements"] if m["implementation"].get("name") == "cuddl"
    ]
    if not rows:
        raise RuntimeError("no cuDDL rows to replay")

    work = Path(output_path).parent / f"{tool}-work"
    work.mkdir(parents=True, exist_ok=True)
    genomes = sorted({path for pair in paths.values() for path in pair})
    index = {path: i for i, path in enumerate(genomes)}
    measure = simdsketch_jaccard if tool == "simdsketch" else sketchlib_jaccard
    similarities = measure(binary, genomes, work, threads)
    expected = len(genomes) * (len(genomes) - 1) // 2
    if len(similarities) != expected:
        raise RuntimeError(f"{tool} returned {len(similarities)} of {expected} pairs")

    emitted: set[tuple[str, str]] = set()
    out_rows = []
    for n, row in enumerate(rows):
        task, truth = row["case"], row["metrics"]
        try:
            ref_path, qry_path = paths[(task["reference_sha256"], task["query_sha256"])]
        except KeyError:
            raise RuntimeError(f"case CSV lacks FASTA paths at row {n}")
        if (ref_path, qry_path) in emitted:
            continue
        emitted.add((ref_path, qry_path))
        if ref_path == qry_path:
            jaccard = 1.0
        else:
            jaccard = similarities[tuple(sorted((index[ref_path], index[qry_path])))]
        if not math.isfinite(jaccard):
            raise RuntimeError(f"{tool} returned a nonfinite Jaccard at row {n}")
        ani = poisson_ani(jaccard)
        exact = truth["exact_ani"]
        out_rows.append(
            {
                "implementation": {"name": tool, "variant": VARIANTS[tool]},
                "case": {key: task[key] for key in CASE_KEYS if key in task},
                "metrics": {
                    "exact_ani": exact,
                    "sketch_ani": ani,
                    "ani_signed_error": ani - exact,
                    "ani_absolute_error": abs(ani - exact),
                    "sketch_jaccard": jaccard,
                    "ani_estimator": "Mash Poisson ANI from sketch Jaccard",
                },
            }
        )

    Path(output_path).write_text(
        json.dumps(
            {
                "schema": report["schema"],
                "name": f"{tool} pairwise accuracy",
                "operation": "pairwise_accuracy",
                "scope": report.get("scope", "end_to_end"),
                "datasets": {},
                "system": report["system"],
                "measurements": out_rows,
            },
            indent=1,
        )
        + "\n"
    )
    print(f"Saved {len(out_rows)} {tool} accuracy rows to {output_path}")


if __name__ == "__main__":
    main(*sys.argv[1:])
