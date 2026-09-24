#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = ["jsonschema", "typer"]
# ///
"""Check real CLI tool selection and exclusion of benchmark output from query timings."""

import json
import random
import subprocess
import struct
import sys
import tempfile
from pathlib import Path

import typer
from benchmark_schema import load_result
from run_micro_comparison import run_timed


def main() -> None:
    diagnostic = "benchmark failure detail"
    try:
        run_timed(
            "failure diagnostic",
            [
                sys.executable,
                "-c",
                f"import sys; sys.stderr.write({diagnostic!r}); sys.exit(1)",
            ],
        )
    except typer.BadParameter as error:
        assert diagnostic in str(error)
        assert "command exited 1" in str(error)
    else:
        raise AssertionError("failed benchmark was accepted")
    runner = Path(__file__).with_name("run_micro_comparison.py")
    with tempfile.TemporaryDirectory(prefix="micro-performance-") as tmp:
        work = Path(tmp)
        genomes = work / "genomes"
        genomes.mkdir()
        rng = random.Random(42)
        sequence = "".join(rng.choices("ACGT", k=100_000))
        for index in range(3):
            (genomes / f"{index}.fna").write_text(f">genome{index}\n{sequence}\n")
        for tool, performance_only, topology in [
            ("cub-exact", True, "all-to-all"),
            ("skani", True, "all-to-all"),
            ("dashing2", True, "all-to-all"),
            ("dashing2", False, "all-to-all"),
            ("dashing2", True, "batch"),
            ("hypergen", True, "all-to-all"),
            ("hypergen", True, "batch"),
            ("cuddl", True, "all-to-all"),
            ("cuddl", True, "batch"),
            ("rabbitsketch", True, "all-to-all"),
            ("rabbitsketch", False, "all-to-all"),
            ("rabbitsketch", True, "batch"),
            ("rabbitsketch", False, "batch"),
            ("cub-exact", False, "all-to-all"),
        ]:
            output = work / "result.json"
            samples = 3 if tool in {"cuddl", "rabbitsketch", "dashing2"} else 1
            command = [
                str(runner),
                str(genomes),
                "--tools",
                tool,
                "--samples",
                str(samples),
                "--warmups",
                "0",
                "--threads",
                "72" if tool == "cuddl" and topology == "all-to-all" else "2",
                "--max-kmers",
                "100000",
                "--output",
                str(output),
            ]
            if performance_only:
                # Performance-only must override even an explicit ANI oracle request.
                command += ["--performance-only", "--skani-truth"]
            if topology == "batch":
                command += ["--topology", "batch", "--query-count", "2"]
                if tool == "dashing2":
                    command += ["--match-rows", "2"]
                if tool == "cuddl":
                    command += ["--cuddl-workers", "72", "--cuddl-index", "sparse"]
            result = subprocess.run(
                command, capture_output=True, text=True, check=False
            )
            assert result.returncode == 0, result.stdout + result.stderr
            measurements = json.loads(output.read_text())["measurements"]
            assert {m["implementation"]["name"] for m in measurements} == {tool}
            assert {m["case"]["measurement"] for m in measurements} == {
                "micro-sketch",
                "micro-compare",
                "micro-search",
            }
            for measurement in measurements:
                operation = measurement["case"]["measurement"]
                if tool == "dashing2" and operation == "micro-sketch":
                    assert "resident" not in measurement["timings"]
                    assert "resident_input" not in measurement["case"]
                    assert "resident_device" not in measurement["case"]
                    assert measurement["timings"]["wall"]["median_ms"] > 0
                    assert measurement["timings"]["wall"]["samples"] == samples
                    continue
                resident = measurement["timings"]["resident"]
                assert resident["median_ms"] > 0, measurement
                assert resident["min_ms"] <= resident["median_ms"] <= resident["max_ms"]
                if tool in {"cuddl", "rabbitsketch", "dashing2"}:
                    for timing in measurement["timings"].values():
                        assert timing["samples"] == samples, measurement
                if operation == "micro-compare":
                    assert (
                        measurement["case"]["pairs"] == measurement["metrics"]["pairs"]
                    )
                    if tool in {"cuddl", "rabbitsketch"}:
                        assert measurement["case"]["pairs"] == (
                            6 if topology == "batch" else 3
                        ), measurement
                    if tool == "dashing2" and topology == "batch":
                        assert measurement["case"]["pairs"] == 6
                        assert measurement["metrics"]["retained_pair_rows"] == 2
                        assert measurement["metrics"]["native_pair_row_stride"] == 3
                        assert measurement["metrics"]["report_row_stride"] == 1
            search = next(
                m for m in measurements if m["case"]["measurement"] == "micro-search"
            )
            if tool in {"cuddl", "rabbitsketch"}:
                compare = next(
                    m
                    for m in measurements
                    if m["case"]["measurement"] == "micro-compare"
                )
                for measurement in measurements:
                    operation = measurement["case"]["measurement"]
                    if operation == "micro-sketch":
                        continue
                    if tool == "rabbitsketch" and operation == "micro-search":
                        continue
                    timing = "query" if operation == "micro-search" else "wall"
                    metrics = (
                        compare["metrics"]
                        if tool == "rabbitsketch"
                        else measurement["metrics"]
                    )
                    # Wall adds process start, loading, and FASTX query sketching.
                    assert (
                        measurement["timings"][timing]["median_ms"]
                        > measurement["timings"]["resident"]["median_ms"]
                    ), measurement
                    assert "excluded_output_ms" in metrics
                    if tool == "cuddl":
                        assert metrics["excluded_output_ms"] == 0
                        assert measurement["case"]["index"] == (
                            "none"
                            if operation == "micro-compare"
                            else "sparse"
                            if topology == "batch"
                            else "dense"
                        )
            if performance_only:
                if tool in {"dashing2", "rabbitsketch"}:
                    assert not search["case"].get("resident_reuses_compare", False)
                    assert search["case"]["index"] == (
                        "none"
                        if topology == "batch"
                        else "native-lsh"
                        if tool == "dashing2"
                        else "native-csr"
                    )
                for measurement in measurements:
                    metrics = measurement["metrics"]
                    assert (
                        not {
                            "jaccard_mae_vs_exact",
                            "ani_mae_vs_skani",
                            "recall_at_k",
                            "top1_rate",
                            "queries_scored",
                        }
                        & metrics.keys()
                    )
                queries = 2 if topology == "batch" else 3
                assert search["case"]["queries"] == queries
                assert (
                    search["metrics"]["per_query_ms"]
                    == search["timings"]["query"]["median_ms"] / queries
                )
            elif tool == "cub-exact":
                assert "recall_at_k" in search["metrics"]
            else:
                assert "recall_at_k" not in search["metrics"]
            print(
                f"{tool} {topology}: {'performance-only' if performance_only else 'accuracy'} passed"
            )

        examples = runner.parent.parent / "build/examples"
        database = work / "cli.cuddl"
        index = work / "cli.index"
        subprocess.run(
            [
                str(examples / "cuddl-build-reference-db"),
                str(genomes),
                "--k",
                "25",
                "--buckets",
                "2048",
                "--exponent-bits",
                "5",
                "--output",
                str(database),
            ],
            check=True,
            capture_output=True,
        )
        subprocess.run(
            [
                str(examples / "cuddl-reference-index"),
                "build",
                str(database),
                "--output",
                str(index),
            ],
            check=True,
            capture_output=True,
        )
        for extra in ([], ["--index", str(index)]):
            command = [
                str(examples / "cuddl-reference-index"),
                "search",
                str(database),
                "--all-to-all",
                "--minimum-matches",
                "1" if extra else "0",
                *extra,
            ]
            listing = subprocess.run(
                command, check=True, capture_output=True, text=True
            ).stdout.splitlines()[1:]
            tsv = sorted(
                (*map(int, row.split("\t")[:6]), float(row.split("\t")[6]))
                for row in listing
            )
            assert [row[:2] for row in tsv] == [(0, 1), (0, 2), (1, 2)], tsv
            binary = work / "cli-results.bin"
            subprocess.run(
                [*command, "--output", str(binary)], check=True, capture_output=True
            )
            assert sorted(struct.iter_unpack("<6Id", binary.read_bytes())) == tsv
        print(
            "cuddl CLI: all-to-all searches each unordered pair once, TSV and binary agree"
        )

        rabbit = (
            runner.parent.parent / "build/benchmarks/rabbitsketch-pipeline-benchmark"
        )
        replay_genome = work / "replay.fna"
        replay_genome.write_text(
            f">short\n{sequence[:24]}\n>exact\n{sequence[24:49]}\n"
            f">ambiguous\n{sequence[:7000].lower()}NNN{sequence[7000:14000]}\n"
        )
        replay_results = []
        for cap in (4096, 1 << 20):
            result = subprocess.run(
                [
                    str(rabbit),
                    "--reference",
                    str(replay_genome),
                    str(genomes / "0.fna"),
                    "--query",
                    str(genomes / "1.fna"),
                    "--topology",
                    "batch",
                    "--ingest",
                    "sequence",
                    "--resident-bytes",
                    str(cap),
                    "--k",
                    "25",
                    "--samples",
                    "3",
                    "--warmups",
                    "1",
                    "--threads",
                    "2",
                    "--output",
                    str(output),
                ],
                capture_output=True,
                text=True,
                check=False,
            )
            assert result.returncode == 0, result.stdout + result.stderr
            rows = load_result(output)["measurements"]
            report = next(m for m in rows if m["case"]["measurement"] == "pipeline")
            assert report["metrics"]["resident_sequence_chunk_oracle_equal"]
            assert report["timings"]["resident_sketch"]["samples"] == 3
            assert (report["case"]["resident_batches"] > 1) == (cap == 4096)
            replay_results.append(
                [m for m in rows if m["case"]["measurement"] != "pipeline"]
            )
        assert replay_results[0] == replay_results[1]
        print("rabbitsketch: bounded resident replay preserves multi-record results")

        search_genomes = work / "search-genomes"
        search_genomes.mkdir()
        unrelated = "".join(rng.choices("ACGT", k=100_000))
        for index, bases in enumerate((sequence, sequence, unrelated)):
            (search_genomes / f"{index}.fna").write_text(f">genome{index}\n{bases}\n")
        result = subprocess.run(
            [
                str(runner),
                str(search_genomes),
                "--tools",
                "dashing2,rabbitsketch,cub-exact",
                "--samples",
                "3",
                "--warmups",
                "0",
                "--threads",
                "2",
                "--max-kmers",
                "100000",
                "--output",
                str(output),
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        assert result.returncode == 0, result.stdout + result.stderr
        for measurement in load_result(output)["measurements"]:
            if (
                measurement["implementation"]["name"] in {"dashing2", "rabbitsketch"}
                and measurement["case"]["measurement"] == "micro-search"
            ):
                assert measurement["metrics"]["queries_scored"] == 3
                assert abs(measurement["metrics"]["recall_at_k"] - 1 / 3) < 1e-12
        print(
            "native indexes: retrieve the identical neighbor and count empty results as misses"
        )

        pipeline = runner.parent.parent / "build/benchmarks/cuddl-pipeline-benchmark"
        pipeline_genome = work / "pipeline.fna"
        pipeline_genome.write_text(f">genome\n{sequence[:1000]}\n")
        genome = str(pipeline_genome)
        for topology, ingest, performance, references, queries, exhaustive in [
            ("batch", "sequence", True, 257, 259, False),
            ("all-to-all", "sequence", True, 257, 1, False),
            ("batch", "packed", True, 3, 2, False),
            ("batch", "sequence", False, 3, 2, False),
            ("batch", "sequence", False, 10001, 1, False),
            ("batch", "sequence", True, 257, 259, True),
            ("all-to-all", "sequence", True, 257, 1, True),
            ("batch", "packed", True, 3, 2, True),
            ("batch", "sequence", False, 3, 2, True),
        ]:
            config = work / "pipeline.toml"
            config.write_text(
                "reference = "
                + json.dumps([genome] * references)
                + "\nquery = "
                + json.dumps([genome] * queries)
                + "\n"
            )
            command = [
                str(pipeline),
                "--config",
                str(config),
                "--output",
                str(output),
                "--topology",
                topology,
                "--ingest",
                ingest,
                "--minimum-matches",
                "0",
                "--samples",
                "2",
                "--warmups",
                "0",
                "--workers",
                "2",
            ]
            if performance:
                command.append("--performance-only")
            if exhaustive:
                command.append("--exhaustive")
            result = subprocess.run(
                command, capture_output=True, text=True, check=False
            )
            assert result.returncode == 0, result.stdout + result.stderr
            measurements = load_result(output)["measurements"]
            report = next(
                m for m in measurements if m["case"]["measurement"] == "pipeline"
            )
            expected = (
                references * (references - 1) // 2
                if topology == "all-to-all"
                else references * queries
            )
            assert report["metrics"]["match_rows_total"] == expected
            if performance:
                assert not report["metrics"]["validation_performed"]
                assert "oracle_passed" not in report["metrics"]
                assert not report["metrics"]["all_to_all_suite"]
                assert all(m["case"]["measurement"] == "pipeline" for m in measurements)
                phases = report["timings"]
                requested = (
                    "search_all_to_all_"
                    if topology == "all-to-all"
                    else "search_batch_"
                ) + ("exhaustive" if exhaustive else "indexed")
                assert requested in phases and "search_and_download" in phases
                assert not any(
                    ("indexed" if exhaustive else "exhaustive") in p
                    or "single" in p
                    or "resident" in p
                    for p in phases
                )
                if topology == "batch":
                    assert not any("all_to_all" in p for p in phases)
                memory = report["memory_bytes"]
                if exhaustive:
                    assert report["case"]["index"] == "none"
                    assert memory["persistent_index"] == 0
                    assert memory["search_workspace"] == 0
                assert (
                    memory["host_result_capacity"]
                    <= memory["search_result_capacity"]
                    + memory["search_match_capacity"]
                )
                if references == 257:
                    assert report["metrics"]["downloaded_tiles"] > 1
                    assert memory["host_result_capacity"] < expected * 36
            else:
                assert report["metrics"]["oracle_passed"]
                assert report["metrics"]["oracle_pairs_checked"] > 0
                if references == 10001:
                    assert not report["metrics"]["all_to_all_suite"]
                    assert not any("all_to_all" in p for p in report["timings"])
            print(
                f"pipeline {topology} {ingest}: {expected} rows, "
                f"validation={not performance}, exhaustive={exhaustive} passed"
            )


if __name__ == "__main__":
    typer.run(main)
