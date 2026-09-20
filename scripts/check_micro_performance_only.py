#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = ["jsonschema", "typer"]
# ///
"""Check real CLI tool selection and exclusion of benchmark output from query timings."""

import json
import random
import subprocess
import sys
import tempfile
from pathlib import Path

import typer
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
            ("cuddl", True, "all-to-all"),
            ("cuddl", True, "batch"),
            ("rabbitsketch", True, "all-to-all"),
            ("cub-exact", False, "all-to-all"),
        ]:
            output = work / "result.json"
            command = [
                str(runner),
                str(genomes),
                "--tools",
                tool,
                "--samples",
                "1",
                "--warmups",
                "0",
                "--threads",
                "2",
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
                    timing = "query" if operation == "micro-search" else "wall"
                    metrics = (
                        compare["metrics"]
                        if tool == "rabbitsketch"
                        else measurement["metrics"]
                    )
                    assert (
                        measurement["timings"][timing]["median_ms"]
                        == metrics["retrieval_ms"]
                    )
                    assert "excluded_output_ms" in metrics
            if performance_only:
                assert "truth oracle" not in result.stdout
                assert "skani truth:" not in result.stdout
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
            else:
                assert "truth oracle" in result.stdout
                assert "recall_at_k" in search["metrics"]
            print(
                f"{tool} {topology}: {'performance-only' if performance_only else 'accuracy'} passed"
            )


if __name__ == "__main__":
    typer.run(main)
