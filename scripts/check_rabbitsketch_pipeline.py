#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = ["jsonschema", "typer"]
# ///
"""Check the real RabbitSketch CLI against exact, unsampled k-mer sets."""

import gzip
import hashlib
import itertools
import math
import os
import random
import re
import subprocess
import tempfile
from pathlib import Path
from typing import Annotated

import typer
from benchmark_schema import load_result


def main(
    binary: Annotated[Path, typer.Option(exists=True, dir_okay=False)] = Path(
        "build/benchmarks/rabbitsketch-pipeline-benchmark"
    ),
) -> None:
    """Exercise FASTX boundaries, native comparisons, report schema, and failures."""
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    rng = random.Random(20260908)
    sequence = "".join(rng.choices("ACGT", k=512))
    complement = str.maketrans("ACGT", "TGCA")
    genomes = [
        [sequence],
        [sequence[:256], sequence[300:]],
        [sequence.translate(complement)[::-1].lower()],
        [sequence[:256] + "NNRY" + sequence[300:]],
        ["ACGT", "ACGT"],
        ["A" * 512],
    ]
    k = 25
    exact = []
    for genome in genomes:
        kmers = set()
        for record in genome:
            for run in re.findall("[ACGT]+", record.upper()):
                for i in range(len(run) - k + 1):
                    word = run[i : i + k]
                    kmers.add(min(word, word.translate(complement)[::-1]))
        exact.append(kmers)

    with tempfile.TemporaryDirectory(prefix="rabbitsketch-pipeline-") as temporary:
        root = Path(temporary)
        paths = []
        for i, genome in enumerate(genomes):
            path = root / f"genome-{i}.fa"
            path.write_text(
                "".join(f">record-{j}\n{record}\n" for j, record in enumerate(genome))
            )
            paths.append(path)
        # Reverse-complement equivalence also checks multiline FASTQ and gzip parsing.
        paths[2] = root / "reverse.fq.gz"
        reverse = genomes[2][0]
        with gzip.open(paths[2], "wt") as stream:
            stream.write(
                f"@reverse\n{reverse[:200]}\n{reverse[200:]}\n+\n{'I' * 512}\n"
            )
        for topology in ("batch", "all-to-all"):
            report = root / f"{topology}.json"
            command = [
                str(binary.resolve()),
                "--topology",
                topology,
                "--samples",
                "2",
                "--warmups",
                "0",
            ]
            for path in paths:
                command.extend(("--reference", str(path)))
                if topology == "batch":
                    command.extend(("--query", str(path)))
            subprocess.run([*command, "--output", str(report)], check=True)
            data = load_result(report, "pipeline")
            pipeline = data["measurements"][0]
            assert pipeline["case"]["orchestration_threads"] == len(
                os.sched_getaffinity(0)
            )
            for threads in (1, 3):
                subprocess.run(
                    [*command, "--threads", str(threads), "--output", str(report)],
                    check=True,
                )
                threaded = load_result(report, "pipeline")
                assert (
                    threaded["measurements"][0]["case"]["orchestration_threads"]
                    == threads
                )
                assert threaded["measurements"][1:] == data["measurements"][1:]
            expected_pairs = set(
                itertools.product(range(len(paths)), repeat=2)
                if topology == "batch"
                else itertools.combinations(range(len(paths)), 2)
            )
            matches = {
                (row["case"]["query_id"], row["case"]["reference_id"]): row["metrics"]
                for row in data["measurements"][1:]
                if row["case"]["measurement"] == "match"
            }
            assert matches.keys() == expected_pairs
            for (q, r), values in matches.items():
                union = exact[q] | exact[r]
                expected = len(exact[q] & exact[r]) / len(union) if union else 0.0
                assert math.isclose(values["jaccard"], expected, abs_tol=1e-12), (
                    q,
                    r,
                    values,
                )
                assert values["query_cardinality"] == len(exact[q])
                assert values["reference_cardinality"] == len(exact[r])
            for role in (
                ("reference", "query") if topology == "batch" else ("reference",)
            ):
                for i, path in enumerate(paths):
                    assert (
                        data["datasets"][f"{role}_{i}"]["sha256"]
                        == hashlib.sha256(path.read_bytes()).hexdigest()
                    )
            assert not any(
                key.startswith("cuda") or key == "gpu" for key in data["system"]
            )
            assert pipeline["metrics"]["resident_streaming_equal"]
            assert pipeline["case"]["resident_input"] == "packed_u64_actg_max"
            for key in (
                "resident_total_wall",
                "resident_reset_wall",
                "resident_construct_wall",
                "resident_finalize_wall",
                "resident_cardinality_wall",
                "resident_search_wall",
            ):
                assert pipeline["timings"][key]["samples"] == 2
                assert pipeline["timings"][key]["source"] == "nvbench_cpu_wall"
            phases = [
                pipeline["timings"][key]
                for key in ("prepare_wall", "query_output_wall", "teardown_wall")
            ]
            total = pipeline["timings"]["end_to_end_wall"]
            assert sum(p["min_ms"] for p in phases) <= total["min_ms"] + 1e-9
            assert total["max_ms"] <= sum(p["max_ms"] for p in phases) + 1e-9
            for timing in pipeline["timings"].values():
                assert timing["samples"] == 2
                assert 0 <= timing["min_ms"] <= timing["median_ms"] <= timing["max_ms"]
            if topology == "batch":
                # Bounded resident replay must preserve record and chunk boundaries.
                streamed_report = root / "streamed.json"
                subprocess.run(
                    [
                        *command,
                        "--ingest",
                        "sequence",
                        "--resident-bytes",
                        "256",
                        "--output",
                        str(streamed_report),
                    ],
                    check=True,
                )
                streamed = load_result(streamed_report, "pipeline")
                streamed_pipeline = streamed["measurements"][0]
                assert streamed_pipeline["case"]["ingest"] == "sequence"
                assert streamed_pipeline["case"]["resident_input"] == "sequence_ascii"
                assert streamed_pipeline["case"]["resident_batch_bytes"] == 256
                assert streamed_pipeline["case"]["resident_batches"] > 1
                assert (
                    streamed_pipeline["case"]["resident_timing_scope"]
                    == "batched_resident_segments"
                )
                assert streamed_pipeline["metrics"]["resident_streaming_equal"]
                for key in ("parse_fastx", "construct_resident"):
                    assert key not in streamed_pipeline["timings"]
                for key in (
                    "resident_total_wall",
                    "resident_reset_wall",
                    "resident_construct_wall",
                    "resident_finalize_wall",
                    "resident_cardinality_wall",
                    "resident_search_wall",
                ):
                    value = streamed_pipeline["timings"][key]
                    assert value["source"] == "nvbench_cpu_wall"
                    assert value["samples"] == 2
                    assert 0 <= value["min_ms"] <= value["median_ms"] <= value["max_ms"]
                assert streamed["measurements"][1:] == data["measurements"][1:]
                typer.echo(
                    "PASS streamed file ingest: identical sketches and pair metrics"
                )
            typer.echo(
                f"PASS {topology}: {len(matches)} exact pair results, identical with 1, 3 and default threads; SIMD {pipeline['case']['simd_u64_path']}"
            )

        # Full sketches exercise the SIMD bottom-k stopping boundary, unlike exact sets.
        subprocess.run(
            [*command, "--sketch-size", "32", "--output", str(report)], check=True
        )
        sampled = load_result(report, "pipeline")
        sampled_matches = {
            (row["case"]["query_id"], row["case"]["reference_id"]): row["metrics"]
            for row in sampled["measurements"][1:]
            if row["case"]["measurement"] == "match"
        }
        assert sampled_matches[0, 2]["jaccard"] == 1.0
        assert sampled_matches[0, 2]["ani"] == 1.0
        assert sampled_matches[0, 4]["jaccard"] == 0.0
        assert all(0 <= values["jaccard"] <= 1 for values in sampled_matches.values())
        typer.echo(
            "PASS sampled sketches: reverse-complement identity and empty comparison"
        )

        malformed = root / "truncated.fq"
        malformed.write_text("@bad\nACGT\n+\nI\n")
        base = [
            str(binary.resolve()),
            "--reference",
            str(paths[0]),
            "--samples",
            "2",
            "--warmups",
            "0",
        ]
        original = paths[0].read_bytes()
        for args in (
            ["--query", str(paths[0]), "--query", str(malformed)],
            ["--query", str(paths[0]), "--threads", "0"],
            ["--query", str(paths[0]), "--threads", "-1"],
            ["--query", str(paths[0]), "--output", str(paths[0])],
            ["--topology", "all-to-all"],
            [],
        ):
            failure = subprocess.run(
                [*base, *args], capture_output=True, text=True, check=False
            )
            assert failure.returncode != 0 and failure.stderr
        assert paths[0].read_bytes() == original
        typer.echo(
            "PASS malformed FASTQ, invalid topology, and input overwrite protection"
        )


if __name__ == "__main__":
    typer.run(main)
