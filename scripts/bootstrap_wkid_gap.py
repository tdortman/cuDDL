#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = ["typer"]
# ///
"""Genome-level bootstrap of cuDDL's mean WKID-error gap to BBTools on related pairs.

Pairs that share a genome are dependent, so each replicate resamples genomes with replacement and
weights every pair by the product of its two genomes' multiplicities.
"""

import json
import math
import random
from collections import defaultdict
from pathlib import Path
from typing import Annotated

import typer


def p95(values: list[float]) -> float:
    ordered = sorted(values)
    position = 0.95 * (len(ordered) - 1)
    low = math.floor(position)
    high = min(low + 1, len(ordered) - 1)
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)


def main(
    accuracy: Annotated[Path, typer.Argument(exists=True, dir_okay=False)] = Path(
        "results/pairwise-accuracy-refseq100-equal-bytes.json"
    ),
    replicates: Annotated[int, typer.Option(min=100)] = 10_000,
    seed: Annotated[int, typer.Option()] = 1,
) -> None:
    rows: dict[tuple[str, str], dict[str, dict]] = defaultdict(dict)
    for m in json.loads(accuracy.read_text())["measurements"]:
        name, case = m["implementation"]["name"], m["case"]
        if name in ("cuddl", "bbtools") and "reference_sha256" in case:
            pair = tuple(sorted((case["reference_sha256"], case["query_sha256"])))
            rows[pair].setdefault(name, m["metrics"])
    pairs = {p: r for p, r in rows.items() if "cuddl" in r and "bbtools" in r}
    genomes = sorted({g for p in pairs for g in p})
    strata = {
        "wkid_ge_0.05": lambda m: m["exact_wkid"] >= 0.05,
        "ani_ge_95": lambda m: bool(m.get("ani_scored")) and m["exact_ani"] >= 0.95,
    }
    rng = random.Random(seed)
    for label, keep in strata.items():
        chosen = {p: r for p, r in pairs.items() if keep(r["cuddl"])}
        cuddl = [100 * r["cuddl"]["wkid_absolute_error"] for r in chosen.values()]
        bbtools = [100 * r["bbtools"]["wkid_absolute_error"] for r in chosen.values()]
        gaps = {p: c - b for p, c, b in zip(chosen, cuddl, bbtools)}
        means = []
        for _ in range(replicates):
            weight: dict[str, int] = defaultdict(int)
            for _ in genomes:
                weight[genomes[rng.randrange(len(genomes))]] += 1
            total = mass = 0.0
            for (a, b), gap in gaps.items():
                w = weight.get(a, 0) * weight.get(b, 0)
                total += w * gap
                mass += w
            if mass:
                means.append(total / mass)
        means.sort()
        print(
            json.dumps(
                {
                    "stratum": label,
                    "pairs": len(chosen),
                    "genomes": len(genomes),
                    "cuddl_p95_pp": round(p95(cuddl), 3),
                    "bbtools_p95_pp": round(p95(bbtools), 3),
                    "cuddl_mean_pp": round(sum(cuddl) / len(cuddl), 3),
                    "bbtools_mean_pp": round(sum(bbtools) / len(bbtools), 3),
                    "mean_gap_pp": round(sum(gaps.values()) / len(gaps), 4),
                    "ci90_pp": [
                        round(means[int(0.05 * len(means))], 3),
                        round(means[int(0.95 * len(means)) - 1], 3),
                    ],
                    # Two one-sided tests at alpha = 0.05 accept equivalence within +-margin
                    # exactly when the 90% interval lies inside it.
                    "tost_min_margin_pp": round(
                        max(abs(means[int(0.05 * len(means))]), abs(means[int(0.95 * len(means)) - 1])), 3
                    ),
                    "replicates": len(means),
                    "seed": seed,
                }
            )
        )


if __name__ == "__main__":
    typer.run(main)
