#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = []
# ///
"""Replay pairwise accuracy cases through hyper-gen dist (ANI only).

hyper-gen reports ANI percent, not k-mer set membership, so only the ANI
error columns are populated. ANI truth is the row's exact_ani (realised
aligned-base ANI). Orientation-independent: one invocation covers both row
orientations. Sketch staging uses a symlink farm (hyper-gen sketches
directories, not files); dist runs with -a 0 to disable thresholding.
"""

import csv
import json
import subprocess
import sys
from pathlib import Path


def run(cmd: list[str]) -> None:
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"hyper-gen failed ({' '.join(cmd)}):\n{proc.stderr[-2000:]}")


def dist_ani(binary: str, query: str, reference: str, work: Path, tag: str, threads: int) -> float:
    farm_r = work / f"{tag}.r"
    farm_q = work / f"{tag}.q"
    for farm, src in ((farm_r, reference), (farm_q, query)):
        farm.mkdir(parents=True, exist_ok=True)
        link = farm / "0.fna"
        if link.is_symlink() or link.exists():
            link.unlink()
        link.symlink_to(Path(src).resolve())
    ref_sk, qry_sk = work / f"{tag}.r.sk", work / f"{tag}.q.sk"
    run([binary, "sketch", "-p", str(farm_r), "-o", str(ref_sk), "-t", str(threads)])
    run([binary, "sketch", "-p", str(farm_q), "-o", str(qry_sk), "-t", str(threads)])
    out = work / f"{tag}.out"
    run([binary, "dist", "-r", str(ref_sk), "-q", str(qry_sk),
         "-o", str(out), "-t", str(threads), "-a", "0"])
    for line in out.read_text().splitlines():
        if not line.strip():
            continue
        try:
            return float(line.split()[-1]) / 100.0
        except ValueError:
            continue
    raise RuntimeError(f"hyper-gen dist emitted no ANI row in {out}")


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

    work = Path(output_path).parent / "hypergen-work"
    work.mkdir(parents=True, exist_ok=True)
    seen: dict[tuple[str, str], float] = {}
    emitted: set[tuple[str, str]] = set()
    out_rows = []
    for n, row in enumerate(rows):
        task, truth = row["case"], row["metrics"]
        try:
            ref_path, qry_path = paths[
                (task["reference_sha256"], task["query_sha256"])]
        except KeyError:
            raise RuntimeError(f"case CSV lacks FASTA paths at row {n}")
        key = (ref_path, qry_path)
        if key not in seen:
            seen[key] = dist_ani(binary, qry_path, ref_path, work, f"n{n}", int(threads))
        ani = seen[key]
        if key in emitted:
            continue
        emitted.add(key)
        metrics: dict = {}
        exact = truth["exact_ani"]
        metrics["exact_ani"] = exact
        metrics["sketch_ani"] = ani
        metrics["ani_signed_error"] = ani - exact
        metrics["ani_absolute_error"] = abs(ani - exact)
        metrics["ani_estimator"] = "hyper-gen dist ANI/100"
        out_rows.append({
            "implementation": {"name": "hypergen", "variant": "gpu"},
            "case": {k: v for k, v in task.items()
                     if k in ("generator_seed", "power", "trial", "size_ratio",
                              "requested_ani", "actual_ani", "mutation_count",
                              "reference_bases", "query_bases", "reference_sha256",
                              "query_sha256", "left_cardinality",
                              "right_cardinality", "intersection")},
            "metrics": metrics,
        })
        if (n + 1) % 200 == 0:
            print(f"hypergen: {n + 1}/{len(rows)}", file=sys.stderr)

    truth_report = json.loads(Path(truth_path).read_text())
    report_out = {
        "schema": truth_report["schema"],
        "name": "hypergen pairwise accuracy",
        "operation": "pairwise_accuracy",
        "scope": truth_report.get("scope", "end_to_end"),
        "datasets": {},
        "system": truth_report["system"],
        "measurements": out_rows,
    }
    Path(output_path).write_text(json.dumps(report_out, indent=1) + "\n")
    print(f"Saved {len(out_rows)} hypergen accuracy rows to {output_path}")


if __name__ == "__main__":
    _, cases_path, truth_path, output_path, binary, threads = sys.argv
    main(cases_path, truth_path, output_path, binary, threads)
