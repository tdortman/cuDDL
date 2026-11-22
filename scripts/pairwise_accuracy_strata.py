#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy", "typer"]
# ///
"""Stratifies the pairwise containment and WKID errors of a `run_pairwise_accuracy.py` result by
true similarity. Needs per-pair rows, so the run must not have used `--no-pair-rows`."""

import json
from collections import defaultdict
from pathlib import Path
from typing import Annotated

import numpy as np
import typer

TOOLS = ("cuddl", "bbtools", "rabbitsketch", "dashing2", "cuco_hll")
METRICS = ("containment", "wkid")
# Strata are properties of the pair, so they are the same for both metrics and both directions.
WKID_FLOOR = 0.05
SPECIES_ANI = 0.95
STRATA = {
    "all": lambda wkid, ani: True,
    f"WKID>={WKID_FLOOR}": lambda wkid, ani: wkid >= WKID_FLOOR,
    f"ANI>={SPECIES_ANI:.0%}": lambda wkid, ani: ani >= SPECIES_ANI,
    f"WKID<{WKID_FLOOR}": lambda wkid, ani: wkid < WKID_FLOOR,
}


def pair_key(case: dict, metric: str) -> tuple:
    """Same identity as run_pairwise_accuracy.summary_key: containment per ordered pair, WKID
    per unordered pair."""
    reference, query = case["reference_sha256"], case["query_sha256"]
    left, right = (
        (reference, query)
        if case["orientation"] == "reference_to_query"
        else (query, reference)
    )
    return (left, right) if metric == "containment" else tuple(sorted((left, right)))


def main(
    path: Annotated[
        Path, typer.Argument(help="run_pairwise_accuracy.py JSON with per-pair rows")
    ] = Path("results/pairwise-accuracy-refseq100-equal-bytes.json"),
) -> None:
    measurements = json.loads(path.read_text())["measurements"]
    summaries = {
        (m["implementation"]["name"], m["case"]["metric"]): m["metrics"]
        for m in measurements
        if m["case"].get("measurement") == "error_summary"
    }
    # (tool, metric, stratum) -> errors, each value counted once
    errors: dict[tuple[str, str, str], list[float]] = defaultdict(list)
    seen: set[tuple] = set()
    for m in measurements:
        name, case = m["implementation"]["name"], m["case"]
        if name not in TOOLS or "orientation" not in case:
            continue
        wkid = case["intersection"] / min(
            case["left_cardinality"], case["right_cardinality"]
        )
        # Pairs below the aligned-fraction floor carry no identity truth, as in the summaries.
        ani = m["metrics"]["exact_ani"] if m["metrics"]["ani_scored"] else 0.0
        for metric in METRICS:
            key = (name, metric, pair_key(case, metric))
            if key in seen:
                continue
            seen.add(key)
            for stratum, member in STRATA.items():
                if member(wkid, ani):
                    errors[name, metric, stratum].append(
                        m["metrics"][f"{metric}_absolute_error"]
                    )
    if not errors:
        raise typer.BadParameter(
            f"{path} has no per-pair rows, rerun scripts/run_pairwise_accuracy.py "
            "without --no-pair-rows"
        )

    print("Reproduction of the stored aggregate summaries (all pairs)")
    for tool in TOOLS:
        for metric in METRICS:
            values = np.array(errors[tool, metric, "all"])
            stored = summaries[tool, metric]
            ok = len(values) == stored["count"] and all(
                np.isclose(np.percentile(values, q), stored[f"absolute_error_{label}"])
                for q, label in ((50, "median"), (95, "p95"))
            )
            print(
                f"  {tool:13} {metric:12} n={len(values):>6} "
                f"p95 {100 * np.percentile(values, 95):.3f}% "
                f"stored {100 * stored['absolute_error_p95']:.3f}%  "
                f"{'match' if ok else 'MISMATCH'}"
            )

    print("\nAbsolute error [%], median / p95 / count")
    print(f"{'tool':13} {'metric':12} " + " ".join(f"{s:>26}" for s in STRATA))
    for tool in TOOLS:
        for metric in METRICS:
            cells = []
            for stratum in STRATA:
                v = np.array(errors[tool, metric, stratum])
                cells.append(
                    f"{100 * np.median(v):7.3f} /{100 * np.percentile(v, 95):7.3f} /{len(v):>6}"
                    if len(v)
                    else "-"
                )
            print(f"{tool:13} {metric:12} " + " ".join(f"{c:>26}" for c in cells))


if __name__ == "__main__":
    typer.run(main)
