#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = []
# ///
"""Replay pairwise accuracy cases through skani dist (ANI only).

skani reports ANI, not k-mer set membership, so only the ANI error columns
are populated. Containment/completeness/wkid/cardinality stay absent; the
quality plots skip lane-metric combos with no data. ANI truth is the row's
exact_ani (realised aligned-base ANI), NOT exact_set_derived_ani.

Small synthetic pairs (< ~20 marker k-mers) are below skani's detection
floor and yield no row; those rows are skipped and reported on stderr.
Orientation-independent: one invocation covers both row orientations.
"""

import csv
import json
import subprocess
import sys
from pathlib import Path


def dist_ani(binary: str, query: str, reference: str, out: Path, threads: int) -> float | None:
    cmd = [binary, "dist", query, reference, "-o", str(out),
           "-t", str(threads), "--min-af", "0"]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"skani failed ({' '.join(cmd)}):\n{proc.stderr[-2000:]}")
    for line in out.read_text().splitlines():
        if not line or line.startswith("Ref_file"):
            continue
        fields = line.split()
        if len(fields) >= 3:
            try:
                return float(fields[2]) / 100.0
            except ValueError:
                continue
    return None


def main(cases_path: str, truth_path: str, output_path: str, binary: str, threads: str) -> None:
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

    work = Path(output_path).parent / "skani-work"
    work.mkdir(parents=True, exist_ok=True)
    seen: dict[tuple[str, str], float | None] = {}
    emitted: set[tuple[str, str]] = set()
    out_rows = []
    skipped = 0
    for n, row in enumerate(rows):
        task, truth = row["case"], row["metrics"]
        try:
            ref_path, qry_path = paths[
                (task["reference_sha256"], task["query_sha256"])]
        except KeyError:
            raise RuntimeError(f"case CSV lacks FASTA paths at row {n}")
        key = (ref_path, qry_path)
        if key not in seen:
            seen[key] = dist_ani(binary, qry_path, ref_path, work / f"n{n}.tsv", int(threads))
        ani = seen[key]
        if ani is None:
            skipped += 1
            continue
        if key in emitted:
            continue
        emitted.add(key)
        metrics: dict = {}
        exact = truth["exact_ani"]
        metrics["exact_ani"] = exact
        metrics["sketch_ani"] = ani
        metrics["ani_signed_error"] = ani - exact
        metrics["ani_absolute_error"] = abs(ani - exact)
        metrics["ani_estimator"] = "skani dist ANI/100"
        out_rows.append({
            "implementation": {"name": "skani", "variant": "cpu"},
            "case": {k: v for k, v in task.items()
                     if k in ("generator_seed", "power", "trial", "size_ratio",
                              "requested_ani", "actual_ani", "mutation_count",
                              "reference_bases", "query_bases", "reference_sha256",
                              "query_sha256", "left_cardinality",
                              "right_cardinality", "intersection")},
            "metrics": metrics,
        })
        if (n + 1) % 200 == 0:
            print(f"skani: {n + 1}/{len(rows)}", file=sys.stderr)

    truth_report = json.loads(Path(truth_path).read_text())
    report_out = {
        "schema": truth_report["schema"],
        "name": "skani pairwise accuracy",
        "operation": "pairwise_accuracy",
        "scope": truth_report.get("scope", "end_to_end"),
        "datasets": {},
        "system": truth_report["system"],
        "measurements": out_rows,
    }
    Path(output_path).write_text(json.dumps(report_out, indent=1) + "\n")
    print(f"Saved {len(out_rows)} skani accuracy rows to {output_path} "
          f"({skipped} below detection floor)")


if __name__ == "__main__":
    _, cases_path, truth_path, output_path, binary, threads = sys.argv
    main(cases_path, truth_path, output_path, binary, threads)
