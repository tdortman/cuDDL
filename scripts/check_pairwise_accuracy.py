#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = ["jsonschema", "typer"]
# ///
"""Check three-way accuracy output against exact sets and pair orientation."""

import math
import subprocess
import sys
from pathlib import Path
from typing import Annotated

import typer
from benchmark_schema import flatten_measurements, load_result

ROOT = Path(__file__).resolve().parent.parent


def main(
    output: Annotated[
        Path, typer.Option(help="Retained smoke-test accuracy report")
    ] = Path("build/pairwise-accuracy-check.json"),
) -> None:
    output = output.resolve()
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/run_pairwise_accuracy.py"),
            "--output",
            str(output),
            "--power",
            "9",
            "--power",
            "12",
            "--ani",
            "0.85",
            "--ani",
            "0.95",
            "--ani",
            "1",
            "--size-ratio",
            "1",
            "--size-ratio",
            "2",
            "--size-ratio",
            "10",
            "--trials",
            "2",
            "--threads",
            "2",
        ],
        cwd=ROOT,
        check=True,
    )
    verify(output)


def verify(output: Path) -> None:
    rows = flatten_measurements(load_result(output, "pairwise_accuracy"))
    assert len(rows) == 180, len(rows)
    cases = {}
    for row in rows:
        key = tuple(
            row[field]
            for field in (
                "power",
                "trial",
                "size_ratio",
                "requested_ani",
                "orientation",
            )
        )
        implementations = cases.setdefault(key, {})
        assert row["implementation"] not in implementations
        implementations[row["implementation"]] = row
        for metric in ("cardinality", "containment", "completeness", "wkid", "ani"):
            exact, estimate = row[f"exact_{metric}"], row[f"sketch_{metric}"]
            assert math.isfinite(estimate)
            assert math.isclose(
                row[f"{metric}_signed_error"], estimate - exact, abs_tol=1e-12
            )
            assert math.isclose(
                row[f"{metric}_absolute_error"], abs(estimate - exact), abs_tol=1e-12
            )
        if row["size_ratio"] == 1 and row["actual_ani"] == 1:
            for metric in ("containment", "completeness", "wkid", "ani"):
                assert row[f"sketch_{metric}"] == 1, (row["implementation"], metric)
        if row["implementation"] == "rabbitsketch":
            assert not {"lower", "equal", "higher", "both_empty"} & row.keys()
            assert row["sketch_size"] == row["buckets"] == 2048
            # Below capacity, FastKMV retains the entire set: no sampling error.
            if max(row["left_cardinality"], row["right_cardinality"]) < 2048:
                for metric in ("cardinality", "containment", "completeness", "wkid"):
                    assert math.isclose(
                        row[f"sketch_{metric}"], row[f"exact_{metric}"], abs_tol=1e-12
                    ), (key, metric)
            if row["size_ratio"] > 1 and row["actual_ani"] == 1:
                assert (
                    row["sketch_ani"] < 1
                )  # Native Jaccard ANI includes the size imbalance.
    for implementations in cases.values():
        assert set(implementations) == {"cuddl", "bbtools", "rabbitsketch"}
        for field in (
            "left_cardinality",
            "right_cardinality",
            "intersection",
            "reference_sha256",
            "query_sha256",
        ):
            assert len({row[field] for row in implementations.values()}) == 1
    for key, implementations in cases.items():
        if key[-1] != "query_to_reference" or key[2] == 1:
            continue
        forward = implementations["rabbitsketch"]
        reverse = cases[(*key[:-1], "reference_to_query")]["rabbitsketch"]
        assert math.isclose(
            forward["sketch_containment"] * forward["sketch_cardinality"],
            reverse["sketch_containment"] * reverse["sketch_cardinality"],
            rel_tol=1e-12,
        )
        assert forward["sketch_ani"] == reverse["sketch_ani"]
    typer.echo(
        "PASS: 180 accuracy rows, exact small sets, identical inputs, and both orientations"
    )


if __name__ == "__main__":
    typer.run(main)
