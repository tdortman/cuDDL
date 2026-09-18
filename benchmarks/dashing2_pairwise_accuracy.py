#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = []
# ///
"""Replay pairwise accuracy cases through Dashing2 FullSetSketch.

Takes the cases CSV (with reference_path/query_path columns) plus the cuDDL
accuracy JSON (for exact_* truth), sketches every genome once into a cache,
and reads each metric off one panel per metric. A panel takes all genomes as
`-F` and all of them again as `-Q`, so it emits one row per F entry and one
column per Q entry and carries every ordered pair in a single call. Cell
(F=right, Q=left) is the pair the row describes.

FullSetSketch, not the default one-permutation SetSketch: one-permutation has
a documented failure mode on small sets and returns flat 0 below ~1k
distinct k-mers, which the accuracy grid is full of. FullSetSketch compares
equally fast and degrades gracefully there.

Measures per pair:
- jaccard: default similarity
- containment of left in right: --containment (returns |F&Q|/|Q|)
- intersection size: --intersection-size, whose diagonal is a genome's own
  cardinality, so one panel supplies both cardinalities as well

Derived: completeness = min(1, cardL/cardR), wkid = I/min(A,B),
ani = wkid^(1/25) compared against exact_set_derived_ani.
"""

import csv
import json
import subprocess
import sys
from pathlib import Path

K = 25
SKETCH_SIZE = 2048
METRIC_FLAGS = {
    "jaccard": [],
    "containment": ["--containment"],
    "intersection": ["--intersection-size"],
}


def panel(
    binary: str, flags: list[str], genomes: list[str], work: Path, tag: str
) -> list[list[float]]:
    """One rectangular cmp panel of every genome against every genome."""
    listing = work / f"{tag}.list.txt"
    listing.write_text("".join(f"{path}\n" for path in genomes))
    out = work / f"{tag}.out.txt"
    cache = work / "cache"
    cache.mkdir(parents=True, exist_ok=True)
    cmd = [
        binary, "cmp", f"-k{K}", f"-S{SKETCH_SIZE}", "--full", *flags,
        "--cache", "--outprefix", str(cache),
        "-F", str(listing), "-Q", str(listing), "--cmpout", str(out),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"dashing2 failed ({' '.join(cmd)}):\n{proc.stderr[-2000:]}")
    table: list[list[float]] = []
    for line in out.read_text().splitlines():
        if not line or line.startswith("#"):
            continue
        cells = line.split("\t")[1:]
        if len(cells) != len(genomes):
            raise RuntimeError(
                f"dashing2 returned {len(cells)} columns for {len(genomes)} genomes in {out}"
            )
        try:
            table.append([float(cell) for cell in cells])
        except ValueError as error:
            raise RuntimeError(f"dashing2 returned a non-numeric panel cell in {out}") from error
    if len(table) != len(genomes):
        raise RuntimeError(
            f"dashing2 returned {len(table)} rows for {len(genomes)} genomes in {out}"
        )
    return table


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
    for m in rows:
        if m["case"].get("k") != K or m["case"].get("buckets") != SKETCH_SIZE:
            raise RuntimeError("expected cuDDL cases with k=25 and 2048 buckets")

    work = Path(output_path).parent / "dashing2-work"
    work.mkdir(parents=True, exist_ok=True)
    genomes = sorted({path for pair in paths.values() for path in pair})
    cells = {path: index for index, path in enumerate(genomes)}
    tables = {
        name: panel(binary, flags, genomes, work, name)
        for name, flags in METRIC_FLAGS.items()
    }

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
        left = qry_path if forward else ref_path
        right = ref_path if forward else qry_path
        left_index, right_index = cells[left], cells[right]
        jac = tables["jaccard"][right_index][left_index]
        cont = tables["containment"][right_index][left_index]
        inter = tables["intersection"][right_index][left_index]
        card_l = tables["intersection"][left_index][left_index]
        card_r = tables["intersection"][right_index][right_index]
        comp = min(1.0, card_l / card_r)
        wkid = max(0.0, min(1.0, inter / min(card_l, card_r)))
        ani = wkid ** (1.0 / K) if wkid > 0 else 0.0

        metrics: dict = {}
        for key in ("orientation", "left_cardinality", "right_cardinality",
                    "intersection", "exact_set_derived_ani"):
            metrics[key] = truth[key]

        def add(name: str, estimate: float) -> None:
            if estimate != estimate:  # NaN guard
                raise RuntimeError(f"Dashing2 returned nonfinite {name} at row {n}")
            exact = truth["exact_" + name]
            metrics["exact_" + name] = exact
            metrics["sketch_" + name] = estimate
            metrics[name + "_signed_error"] = estimate - exact
            metrics[name + "_absolute_error"] = abs(estimate - exact)

        add("cardinality", card_l)
        add("containment", cont)
        add("completeness", comp)
        add("wkid", wkid)
        add("ani", ani)
        metrics["cardinality_relative_error"] = (
            metrics["cardinality_signed_error"] / truth["exact_cardinality"])
        metrics["cardinality_absolute_relative_error"] = abs(
            metrics["cardinality_relative_error"])
        metrics["sketch_jaccard"] = jac
        metrics["ani_estimator"] = "wkid^(1/k), wkid=I/min(A,B) from intersection-size"
        metrics["completeness_estimator"] = "min(1, estimated_left/estimated_right)"
        metrics["query_method"] = "FullSetSketch rectangular cmp"
        out_rows.append({
            "implementation": {"name": "dashing2", "variant": "FullSetSketch"},
            "case": {k: v for k, v in task.items()},
            "metrics": metrics,
        })

    for m in out_rows:
        m["case"]["sketch_size"] = SKETCH_SIZE
    truth_report = json.loads(Path(truth_path).read_text())
    report_out = {
        "schema": truth_report["schema"],
        "name": "Dashing2 pairwise accuracy",
        "operation": "pairwise_accuracy",
        "scope": truth_report.get("scope", "end_to_end"),
        "datasets": {},
        "system": truth_report["system"],
        "measurements": out_rows,
    }
    Path(output_path).write_text(json.dumps(report_out, indent=1) + "\n")
    print(f"Saved {len(out_rows)} Dashing2 accuracy rows to {output_path}")


if __name__ == "__main__":
    _, cases_path, truth_path, output_path, binary = sys.argv
    main(cases_path, truth_path, output_path, binary)
