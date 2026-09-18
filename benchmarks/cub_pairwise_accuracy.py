#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = []
# ///
"""Replay pairwise accuracy cases through cub-exact-pairwise (exact lanes).

cub-exact computes the same sorted unique k-mer sets the cuDDL benchmark
derives truth from, so its containment/completeness/wkid errors should be
bit-zero and its ANI (mash_ani, set-derived) lands near exact_set_derived_ani.
One invocation per distinct (reference, query) pair; orientation only swaps
which side is left, and both orientations share the pair result.
"""

import csv
import json
import subprocess
import sys
from pathlib import Path


def pair_result(binary: str, reference: str, query: str, out: Path) -> dict:
    cmd = [binary, "--topology", "batch", "--reference", reference,
           "--query", query, "--samples", "1", "--output", str(out)]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"cub-exact failed ({' '.join(cmd)}):\n{proc.stderr[-2000:]}")
    pairs = json.loads(out.read_text())["pairs"]
    if len(pairs) != 1:
        raise RuntimeError(f"expected 1 cub pair row, got {len(pairs)} in {out}")
    return pairs[0]


def main(cases_path: str, truth_path: str, output_path: str, binary: str) -> None:
    with open(cases_path, newline="") as handle:
        cases = list(csv.DictReader(handle))
    paths = {
        (row["reference_sha256"], row["query_sha256"]): (
            row["reference_path"], row["query_path"])
        for row in cases
    }
    report = json.loads(Path(truth_path).read_text())
    rows = [m for m in report["measurements"] if m["implementation"].get("name") == "cuddl"]
    if not rows:
        raise RuntimeError("no cuDDL rows to replay")

    work = Path(output_path).parent / "cub-work"
    work.mkdir(parents=True, exist_ok=True)
    seen: dict[tuple[str, str], dict] = {}
    out_rows = []
    for n, row in enumerate(rows):
        task, truth = row["case"], row["metrics"]
        orientation = truth["orientation"]
        forward = orientation == "query_to_reference"
        if orientation not in ("query_to_reference", "reference_to_query"):
            raise RuntimeError(f"unknown pair orientation: {orientation}")
        try:
            ref_path, qry_path = paths[
                (task["reference_sha256"], task["query_sha256"])]
        except KeyError:
            raise RuntimeError(f"case CSV lacks FASTA paths at row {n}")
        key = (ref_path, qry_path)
        if key not in seen:
            seen[key] = pair_result(binary, ref_path, qry_path, work / f"n{n}.json")
        pair = seen[key]
        # cub names sides by CLI arg: a=query-arg, b=reference-arg (verified:
        # small-query run reports distinct_a=small, containment_a_in_b=1.0).
        left_size = truth["left_cardinality"]
        right_size = truth["right_cardinality"]
        cont = (pair["containment_a_in_b"] if forward
                else pair["containment_b_in_a"])
        comp = min(1.0, left_size / right_size)
        wkid = pair["intersection"] / min(left_size, right_size)
        ani = pair["mash_ani"] / 100.0

        metrics: dict = {}
        for extra in ("left_cardinality", "right_cardinality",
                      "intersection", "exact_set_derived_ani"):
            metrics[extra] = truth[extra]
        metrics["orientation"] = orientation

        def add(name: str, estimate: float, exact_key: str | None = None) -> None:
            exact = truth[exact_key or ("exact_" + name)]
            metrics["exact_" + name] = exact
            metrics["sketch_" + name] = estimate
            metrics[name + "_signed_error"] = estimate - exact
            metrics[name + "_absolute_error"] = abs(estimate - exact)

        add("cardinality", float(pair["distinct_a"] if forward
                                 else pair["distinct_b"]))
        add("containment", cont)
        add("completeness", comp)
        add("wkid", wkid)
        add("ani", ani)
        metrics["cardinality_relative_error"] = (
            metrics["cardinality_signed_error"] / truth["exact_cardinality"])
        metrics["cardinality_absolute_relative_error"] = abs(
            metrics["cardinality_relative_error"])
        metrics["sketch_jaccard"] = pair["jaccard"]
        metrics["ani_estimator"] = "cub mash_ani/100"
        metrics["query_method"] = "exact sorted k-mer sets"
        out_rows.append({
            "implementation": {"name": "cub-exact", "variant": "gpu-exact"},
            "case": {k: v for k, v in task.items()
                     if k in ("generator_seed", "power", "trial", "size_ratio",
                              "requested_ani", "actual_ani", "mutation_count",
                              "reference_bases", "query_bases", "reference_sha256",
                              "query_sha256", "orientation", "left_cardinality",
                              "right_cardinality", "intersection")},
            "metrics": metrics,
        })
    truth_report = json.loads(Path(truth_path).read_text())
    report_out = {
        "schema": truth_report["schema"],
        "name": "cub-exact pairwise accuracy",
        "operation": "pairwise_accuracy",
        "scope": truth_report.get("scope", "end_to_end"),
        "datasets": {},
        "system": truth_report["system"],
        "measurements": out_rows,
    }
    Path(output_path).write_text(json.dumps(report_out, indent=1) + "\n")
    print(f"Saved {len(out_rows)} cub-exact accuracy rows to {output_path}")


if __name__ == "__main__":
    _, cases_path, truth_path, output_path, binary = sys.argv
    main(cases_path, truth_path, output_path, binary)
