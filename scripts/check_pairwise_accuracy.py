#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = ["jsonschema", "typer"]
# ///
"""Check four-way accuracy output against exact sets and pair orientation."""

import math
import subprocess
import sys
from pathlib import Path
from typing import Annotated

import typer
from benchmark_schema import flatten_measurements, load_result

ROOT = Path(__file__).resolve().parent.parent

# The runner's default --ani-min-aligned-fraction, restated so the checker can hold a result to
# the contract its rows claim: below this aligned fraction a pair's ANI is not scored.
ANI_ALIGNED_FLOOR = 0.15


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
    sources = {row.get("case_source", "synthetic") for row in rows}
    if sources == {"refseq"}:
        verify_genome(rows)
        return
    assert sources == {"synthetic"} or not any(
        "case_source" in row for row in rows), sources
    full = [r for r in rows if r["implementation"] in
            ("cuddl", "cuddl-bbtools", "cuddl-paper", "bbtools", "rabbitsketch",
             "cuco_hll", "dashing2", "cub-exact")]
    assert len(full) == 8 * 60, len(full)
    partial = [r for r in rows if r["implementation"] in ("skani", "hypergen")]
    assert partial, "ANI-only lanes missing entirely"
    cases = {}
    for row in rows:
        key = tuple(
            row[field]
            for field in (
                "power",
                "trial",
                "size_ratio",
                "requested_ani",
            )
        ) + (row.get("orientation"),)
        scored = ("ani",) if row["implementation"] in ("skani", "hypergen") else (
            "cardinality", "containment", "completeness", "wkid", "ani")
        for metric in scored:
            exact, estimate = row[f"exact_{metric}"], row[f"sketch_{metric}"]
            assert math.isfinite(estimate)
            assert math.isclose(
                row[f"{metric}_signed_error"], estimate - exact, abs_tol=1e-12
            )
            assert math.isclose(
                row[f"{metric}_absolute_error"], abs(estimate - exact), abs_tol=1e-12
            )
        if row["implementation"] in ("skani", "hypergen"):
            assert not {"lower", "equal", "higher", "both_empty",
                        "exact_cardinality", "exact_containment"} & row.keys()
            continue
        if row["size_ratio"] == 1 and row["actual_ani"] == 1 and row["implementation"] not in ("dashing2", "cub-exact"):
            for metric in ("containment", "completeness", "wkid", "ani"):
                assert row[f"sketch_{metric}"] == 1, (row["implementation"], metric)
        if row["implementation"] == "cuco_hll":
            assert not {"lower", "equal", "higher", "both_empty"} & row.keys()
            assert row["hll_precision"] == 11
            left, right = row["sketch_cardinality"], row["sketch_right_cardinality"]
            assert row["sketch_intersection_raw"] == left + right - row["sketch_union"]
            assert 0 <= row["sketch_intersection"] <= min(left, right)
            if row["actual_ani"] == 1 and row["orientation"] == "query_to_reference":
                assert (
                    row["sketch_union"] == right
                )  # The query is a subset of the reference.
        if row["implementation"] == "dashing2":
            assert not {"lower", "equal", "higher", "both_empty"} & row.keys()
            assert row["sketch_size"] == row["buckets"] == 2048
            # FullSetSketch containment is a register estimate, not exact:
            # identical sets land within ~10 percent at these sizes.
            if row["actual_ani"] == 1 and row["orientation"] == "query_to_reference":
                assert math.isclose(row["sketch_containment"], 1.0, abs_tol=0.10), (key, "containment")
        if row["implementation"] == "cub-exact":
            # Same sorted unique k-mer sets as truth: set metrics bit-zero.
            for metric in ("cardinality", "containment", "completeness", "wkid"):
                assert row[f"sketch_{metric}"] == row[f"exact_{metric}"], (key, metric)
        if row["implementation"] in ("cuddl-bbtools", "cuddl-paper"):
            cuddl = cases[key]["cuddl"] if key in cases and "cuddl" in cases[key] else None
            if cuddl is not None:
                for field in ("lower", "equal", "higher", "both_empty",
                              "sketch_containment", "sketch_completeness",
                              "sketch_wkid", "sketch_ani"):
                    assert row[field] == cuddl[field], (key, field)
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
        assert {"cuddl", "cuddl-bbtools", "cuddl-paper", "bbtools", "rabbitsketch",
                "cuco_hll", "dashing2", "cub-exact"} <= set(implementations)
        full = {k: v for k, v in implementations.items() if v["implementation"] not in ("skani", "hypergen")}
        for field in (
            "left_cardinality",
            "right_cardinality",
            "intersection",
            "reference_sha256",
            "query_sha256",
        ):
            assert len({row[field] for row in full.values()}) == 1
    for key, implementations in cases.items():
        if key[-1] != "query_to_reference" or key[2] == 1:
            continue
        hll = implementations["cuco_hll"]
        reversed_hll = cases[(*key[:-1], "reference_to_query")]["cuco_hll"]
        for field in (
            "sketch_union",
            "sketch_intersection",
            "sketch_wkid",
            "sketch_ani",
        ):
            assert hll[field] == reversed_hll[field]
        assert hll["sketch_cardinality"] == reversed_hll["sketch_right_cardinality"]
        forward = implementations["rabbitsketch"]
    typer.echo(
        f"PASS: {len(rows)} accuracy rows, exact small sets, identical inputs, and both orientations"
    )


def verify_genome(rows: list[dict]) -> None:
    """Check RefSeq-mode output: split truth, internal consistency."""
    cases: dict = {}
    seen: set = set()
    for row in rows:
        key = (row["reference_sha256"], row["query_sha256"],
               row.get("orientation"), row["implementation"])
        assert key not in seen, key
        seen.add(key)
        pair = (row["reference_sha256"], row["query_sha256"])
        implementations = cases.setdefault(pair, {})
        if row["implementation"] not in ("skani", "hypergen"):
            implementations[row["implementation"]] = row
        else:
            implementations.setdefault(row["implementation"], row)
        scored = ("ani",) if row["implementation"] in ("skani", "hypergen") else (
            "cardinality", "containment", "completeness", "wkid", "ani")
        # Unrelated pairs are not scored: their set-derived ANI measures the k-th root's noise
        # floor, so the runner drops the error columns and says so on the row.
        assert isinstance(row["ani_scored"], bool), key
        if not row["ani_scored"]:
            assert "ani_signed_error" not in row and "ani_absolute_error" not in row, key
            scored = tuple(metric for metric in scored if metric != "ani")
        if "skani_aligned_fraction" in row:
            assert row["ani_scored"] == (
                row["skani_aligned_fraction"] >= ANI_ALIGNED_FLOOR), key
        for metric in scored:
            exact, estimate = row[f"exact_{metric}"], row[f"sketch_{metric}"]
            assert math.isfinite(estimate)
            assert math.isclose(
                row[f"{metric}_signed_error"], estimate - exact, abs_tol=1e-12
            )
            assert math.isclose(
                row[f"{metric}_absolute_error"], abs(estimate - exact), abs_tol=1e-12
            )
        if row["implementation"] in ("skani", "hypergen"):
            assert math.isfinite(row.get("skani_aligned_fraction", float("nan")))
            continue
        if row["implementation"] == "cub-exact":
            for metric in ("cardinality", "containment", "completeness", "wkid"):
                assert row[f"sketch_{metric}"] == row[f"exact_{metric}"], (key, metric)
    for implementations in cases.values():
        assert {"cuddl", "cuddl-bbtools", "cuddl-paper", "cub-exact"} <= set(implementations)
        full = {k: v for k, v in implementations.items()
                if v["implementation"] not in ("skani", "hypergen")}
        for field in ("left_cardinality", "right_cardinality", "intersection",
                      "reference_sha256", "query_sha256"):
            assert len({row[field] for row in full.values()}) == 1
        if "skani" in implementations and "hypergen" in implementations:
            assert math.isclose(
                implementations["skani"]["exact_ani"],
                implementations["hypergen"]["exact_ani"], abs_tol=1e-12)

if __name__ == "__main__":
    typer.run(main)
