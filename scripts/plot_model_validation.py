#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "matplotlib>=3.9",
#   "numpy>=2.0",
#   "typer>=0.12",
# ]
# ///
"""Predicted vs. measured wall time for Equation (7).

Every rate comes from an isolated single-node 100k run or a standalone storage benchmark, never
from the run being predicted. Predictions extrapolate those rates to 400k and three-node runs:
setup and ingest scale with N / P, comparison, download, and writes with N^2. Sparse and
thresholded output sizes are the bytes the predicted run wrote; the script also reports the errors
when those sizes come from the 100k subset's bytes per pair instead. It prints each workload's
predictions and per-run errors, then draws the parity plot.
"""

import json
import statistics
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Annotated

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import typer

mpl.rcParams.update(
    {
        "text.usetex": True,
        "text.latex.preamble": r"\usepackage[T1]{fontenc}\usepackage{libertine}"
        r"\usepackage[libertine]{newtxmath}",
        "font.family": "serif",
        "pdf.use14corefonts": True,
        "ps.useafm": True,
    }
)

PAIRS_100K = 100_000 * 99_999 // 2
PAIRS_400K = 400_000 * 399_999 // 2
PAIRS_200K = 200_000 * 199_999 // 2
C_INPUT_BYTES_200K = 262_992_682_174


@dataclass(frozen=True)
class System:
    setup_s: float
    ingest_s: float
    compare_pairs_per_s: float
    d2h_bytes_per_s: float
    write_bytes_per_s: float


# Rates and their sources (all results/ paths):
# A, RTX 5090, RefSeq 211: setup and warm ingest from the 100k durable runs (mon03-dur-100k-*);
#   cold ingest from standalone 100k cold whole-file reads (mon03-cold-read-100k.jsonl, 10.93 GB/s);
#   resident all-to-all with near-duplicate deltas (mon03-deltas-100k.jsonl); D2H from nvbandwidth;
#   writes from fio, 8 MiB blocks, queue depth 4, O_DIRECT, end_fsync (mon03-nvme-fio.jsonl).
# B, GH200: setup (0.13 s) and warm ingest (6.32 s) from alist/gh200-dur-100k-*-warm-r{1,2,3}.json;
#   resident all-to-all with deltas (lop3-isolation-gh200.jsonl, 694.9 ms); D2H of one 134 MB tile
#   into pinned memory (gh200-d2h-bandwidth.jsonl); writes from fio, 128 KiB, queue depth 32,
#   O_DIRECT, end_fsync (gh200-nvme-fio.jsonl).
# C, A100 node, October 2026 RefSeq: 100k resident all-to-all with deltas on one A100
#   (mogon-a100-grid-search-100k-3x8.json, ranks 1); 128-core warm read + inflate at 8.322 GB/s;
#   writes from dd, 8 MiB blocks, O_DIRECT, fsync on gpu0203 (mogon-a100-standalone-write-874905.jsonl).
A_COLD_INGEST_S = 123_794_059_160 / 10.9259e9
# B reads the same 100k inputs cold from its PM9A3 at 5.429 GB/s (fio, 128 KiB, queue depth 32,
# O_DIRECT; gh200-nvme-fio.jsonl), as does a whole-file read of every input after
# posix_fadvise eviction (5.43 GB/s, gh200-cold-read-100k.jsonl).
B_COLD_INGEST_S = 123_794_059_160 / 5.429e9
C_INPUT_BYTES = 525_595_271_536
# Cold System C runs read evicted files from Lustre, which one node reads at 5.468 GB/s in a
# standalone 1,024-thread test (filesystem_read_small_files in mogon-a100-ceilings-refseq-2026-10.jsonl),
# below the 8.322 GB/s warm read + inflate rate, so the read sets cold ingest.
C_COLD_INGEST_S = C_INPUT_BYTES / 5.468e9
SYSTEMS = {
    "A": System(0.28, 7.33, PAIRS_100K / 0.4633, 56.5e9, 12.95e9),
    "B": System(0.13, 6.32, PAIRS_100K / 0.6949, 294.7e9, 1.477e9),
    "C": System(3.39, C_INPUT_BYTES / 8.322e9, 3.966e9, 26.0e9, 1.2319e9),
}
# Thresholded m_min = 20 runs search the inverted index, so their comparison term is the 100k
# one-A100 m_min = 20 search phase with output discarded (median of
# mogon-a100-grid-e2e-100k-1x1-m20-r{1,2,3}), scaled with N^2. That phase includes the edge
# download, so the download term is zero for them.
C_M20_SEARCH_100K_S = 1.069


@dataclass(frozen=True)
class Workload:
    label: str
    system: str
    pairs: int
    nodes: int
    download_bytes: float
    write_bytes: float
    runs: list[float]
    extrapolated: bool
    write_rate: float | None = None
    compare_s: float | None = None
    ingest_s: float | None = None
    setup_s: float | None = None
    group: str = "durable"


class Runs(list[float]):
    """Wall times of a workload's runs, with each run's measured search phase alongside."""

    search: list[float]

    def __init__(self, records: list[dict]) -> None:
        super().__init__(r["wall_ms"] / 1e3 for r in records)
        self.search = [r["phases_wall_ms"]["search"] / 1e3 for r in records]


def overlapped_terms(w: Workload) -> tuple[float, float, float]:
    """Predicted comparison, download and write times, which overlap in the search phase."""
    s = SYSTEMS[w.system]
    compare = w.compare_s if w.compare_s is not None else w.pairs / (s.compare_pairs_per_s * w.nodes)
    download = w.download_bytes / (s.d2h_bytes_per_s * w.nodes)
    write = w.write_bytes / ((w.write_rate or s.write_bytes_per_s) * w.nodes)
    return compare, download, write


def predict(w: Workload) -> tuple[float, float, float, float]:
    s = SYSTEMS[w.system]
    setup = w.setup_s if w.setup_s is not None else s.setup_s
    ingest = (w.ingest_s if w.ingest_s is not None else s.ingest_s) / w.nodes
    terms = overlapped_terms(w)
    overlap = setup + ingest + max(terms)
    serial = setup + ingest + sum(terms)
    all_overlapped = setup + max(ingest, *terms)
    write_only = setup + ingest + terms[2]
    return overlap, serial, all_overlapped, write_only


def upfront_valid(w: Workload) -> bool:
    """Upfront domain-of-validity predicate, evaluated before running or inspecting errors:
    (1) dedicated local storage (excludes shared Lustre), and
    (2) no NUMA-local cache eviction under zone_reclaim_mode=1 (on System C's 16x128 GiB NUMA
        nodes, tmpfs requires interleaving so combined input and output fit per node)."""
    if w.group == "lustre":
        return False
    if "tmpfs" in w.label and "interleave" not in w.label:
        return False
    return True


def validate_cpu_pipeline(results: Path) -> tuple[list[tuple[str, float, float, float, float]], float, float]:
    """Apply Equation (7) with T_download = 0 to cuDDL-CPU on Systems B (100k) and C (400k),
    using isolated resident compare and isolated warm ingest rates, against Table 6."""
    def load_jsonl(name: str) -> list[dict]:
        return [json.loads(line) for line in (results / name).read_text().splitlines() if line.strip()]

    # System B isolated CPU compare with deltas (RefSeq 211, tile 128, no write).
    b_cmp_runs: list[float] = []
    cur = ""
    for rec in load_jsonl("gh200-cpu-overlap-tiles64-128.jsonl"):
        if "compare_ms" not in rec:
            cur = rec.get("label", "")
        elif cur == "rs211-100k-A tile 128 dense --deltas" and not rec.get("o_direct") and not rec.get("fsync"):
            b_cmp_runs.append(rec["total_ms"] / 1e3)
    b_cmp_s = statistics.median(b_cmp_runs)

    # System B isolated CPU warm ingest (new SIMD sketcher, runs new-1 and new-2).
    b_ing_runs, b_list_runs = [], []
    cur = ""
    for rec in load_jsonl("gh200-cpu-sketch-ingest-100k.jsonl"):
        if "run" in rec:
            cur = rec["run"]
        elif cur in ("new-1", "new-2") and "ingest_s" in rec:
            b_ing_runs.append(rec["ingest_s"])
            b_list_runs.append(rec["list_s"])
    b_ing_s = statistics.median(b_ing_runs)
    b_list_s = statistics.median(b_list_runs)

    b_res = load_jsonl("gh200-cpu-final-100k.jsonl")
    b_e2e = load_jsonl("gh200-cpu-pipeline-final-100k.jsonl")
    b_res_dense = statistics.median(r["total_ms"] / 1e3 for r in b_res if r["format"] == "dense")
    b_res_sparse = statistics.median(r["total_ms"] / 1e3 for r in b_res if r["format"] == "sparse")
    b_e2e_dense = statistics.median(r["end_to_end_ms"] / 1e3 for r in b_e2e if r["format"] == "dense")
    b_e2e_sparse = statistics.median(r["end_to_end_ms"] / 1e3 for r in b_e2e if r["format"] == "sparse")
    b_dense_bytes = next(r["bytes"] for r in b_e2e if r["format"] == "dense")
    b_sparse_bytes = next(r["bytes"] for r in b_e2e if r["format"] == "sparse")

    # System C isolated CPU compare with deltas (400k, tile 512, no write).
    c_cmp_s = statistics.median(
        r["total_ms"] / 1e3 for r in load_jsonl("mogon-cpu-overlap-timing-875225.jsonl")
        if not r.get("o_direct") and "total_ms" in r
    )
    # System C isolated CPU warm ingest (new SIMD sketcher, runs new-1 and new-2).
    c_ing_runs, c_list_runs = [], []
    cur = ""
    for rec in load_jsonl("mogon-cpu-sketch-ingest-400k.jsonl"):
        if "run" in rec:
            cur = rec["run"]
        elif cur in ("new-1", "new-2") and "ingest_s" in rec:
            c_ing_runs.append(rec["ingest_s"])
            c_list_runs.append(rec["list_s"])
    c_ing_s = statistics.median(c_ing_runs)
    c_list_s = statistics.median(c_list_runs)

    c_res = load_jsonl("mogon-cpu-final15-400k-875327.jsonl")
    c_e2e = load_jsonl("mogon-cpu-pipeline-final15-400k-875327.jsonl")
    c_res_dense = statistics.median(r["total_ms"] / 1e3 for r in c_res if r["format"] == "dense")
    c_res_sparse = statistics.median(r["total_ms"] / 1e3 for r in c_res if r["format"] == "sparse")
    c_e2e_dense = statistics.median(r["end_to_end_ms"] / 1e3 for r in c_e2e if r["format"] == "dense")
    c_e2e_sparse = statistics.median(r["end_to_end_ms"] / 1e3 for r in c_e2e if r["format"] == "sparse")
    c_dense_bytes = next(r["bytes"] for r in c_e2e if r["format"] == "dense")
    c_sparse_bytes = next(r["bytes"] for r in c_e2e if r["format"] == "sparse")

    b_bw, c_bw = SYSTEMS["B"].write_bytes_per_s, SYSTEMS["C"].write_bytes_per_s
    b_gpu_base = SYSTEMS["B"].setup_s + SYSTEMS["B"].ingest_s
    c_gpu_base = SYSTEMS["C"].setup_s + SYSTEMS["C"].ingest_s

    cases = [
        ("B 100k dense res", max(b_cmp_s, b_dense_bytes / b_bw), b_res_dense, b_dense_bytes / b_bw, 13.49),
        ("B 100k sparse res", max(b_cmp_s, b_sparse_bytes / b_bw), b_res_sparse, b_sparse_bytes / b_bw, 4.13),
        ("B 100k dense E2E", b_list_s + b_ing_s + max(b_cmp_s, b_dense_bytes / b_bw), b_e2e_dense,
         b_gpu_base + b_dense_bytes / b_bw, 19.95),
        ("B 100k sparse E2E", b_list_s + b_ing_s + max(b_cmp_s, b_sparse_bytes / b_bw), b_e2e_sparse,
         b_gpu_base + b_sparse_bytes / b_bw, 10.58),
        ("C 400k dense res", max(c_cmp_s, c_dense_bytes / c_bw), c_res_dense, c_dense_bytes / c_bw, 261.5),
        ("C 400k sparse res", max(c_cmp_s, c_sparse_bytes / c_bw), c_res_sparse, c_sparse_bytes / c_bw, 85.0),
        ("C 400k dense E2E", c_list_s + c_ing_s + max(c_cmp_s, c_dense_bytes / c_bw), c_e2e_dense,
         c_gpu_base + c_dense_bytes / c_bw, 330.7),
        ("C 400k sparse E2E", c_list_s + c_ing_s + max(c_cmp_s, c_sparse_bytes / c_bw), c_e2e_sparse,
         c_gpu_base + c_sparse_bytes / c_bw, 156.5),
    ]
    typer.echo(
        f"CPU pipeline validation (B cmp={b_cmp_s:.2f}s ing={b_ing_s:.2f}s; "
        f"C cmp={c_cmp_s:.2f}s ing={c_ing_s:.2f}s):"
    )
    res_err, e2e_err = [], []
    for label, pred, meas, gpu_pred, gpu_meas in cases:
        err = pred / meas - 1
        (res_err if "res" in label else e2e_err).append(err)
        typer.echo(
            f"  {label:20s} pred {pred:7.2f}s vs meas {meas:7.2f}s ({100 * err:+5.1f}%)  "
            f"GPU speedup pred {pred / gpu_pred:.2f}x vs meas {meas / gpu_meas:.2f}x"
        )
    res_mape, e2e_mape = mape(res_err), mape(e2e_err)
    typer.echo(f"CPU MAPE: resident {res_mape:.1f}% (n=4), E2E {e2e_mape:.1f}% (n=4)")
    return cases, res_mape, e2e_mape


def streamed(bytes_per_pair: float, compare_pairs_per_s: float, dt: float = 0.01) -> float:
    """One System C A100 at 400k if block-triangular comparison and writes ran behind ingest:
    pairs among ingested genomes become comparable as they arrive, comparison runs at its
    resident rate, and writes drain the compared bytes at the drive's standalone rate."""
    c = SYSTEMS["C"]
    genomes, pairs = 400_000, PAIRS_400K
    per_genome = c.ingest_s / genomes
    t = compared = written = 0.0
    while written < pairs * bytes_per_pair - 1:
        n = min(genomes, t / per_genome)
        compared = min(n * (n - 1) / 2, compared + compare_pairs_per_s * dt)
        written = min(compared * bytes_per_pair, written + c.write_bytes_per_s * dt)
        t += dt
    return c.setup_s + t


def workloads(results: Path) -> list[Workload]:
    def walls(*names: str) -> Runs:
        return Runs([json.loads((results / n).read_text()) for n in names])

    def written(name: str) -> float:
        return json.loads((results / name).read_text())["bytes_written_total"]

    # Standalone tmpfs rate: eight parallel dd streams into /dev/shm in the capped-run job.
    tmpfs = next((json.loads(line)["gb_s"] * 1e9
                  for p in sorted(results.glob("mogon-a100-standalone-write-*.jsonl"))
                  for line in p.read_text().splitlines() if '"tmpfs-streams8"' in line), None)
    raw = [json.loads(line) for line in (results / "mon03-m1-records-100k.jsonl").read_text().splitlines()
           if line.strip()]
    a_sparse = [f"mon03-dur-100k-sparse-similarity-warm-r{r}.json" for r in (1, 2)]
    a_dense = [f"mon03-dur-100k-similarity-warm-r{r}.json" for r in (1, 2)]
    b_sparse = [f"alist/gh200-dur-100k-sparse-similarity-warm-r{r}.json" for r in (1, 2, 3)]
    b_dense = [f"alist/gh200-dur-100k-similarity-warm-r{r}.json" for r in (1, 2, 3)]
    b_sparse_cold = [f"alist/gh200-dur-100k-sparse-similarity-cold-r{r}.json" for r in (1, 2, 3)]
    b_dense_cold = [f"alist/gh200-dur-100k-similarity-cold-r{r}.json" for r in (1, 2, 3)]
    c_m20 = ["mogon-a100-dur-400k-1x1-m20-nvme-r2.json",
             *[f"mogon-a100-dur-400k-1x1-m20-nvme-874816-r{r}.json" for r in (1, 2, 3)]]
    # Warm runs are every run but the first of each job, which reads the input cold.
    c_sparse = ["mogon-a100-dur-400k-1x1-sparse-128c-r2.json",
                "mogon-a100-dur-400k-1x1-sparse-nvme-uncapped-r2.json",
                "mogon-a100-dur-400k-1x1-sparse-nvme-uncapped-r3.json",
                *[f"mogon-a100-dur-400k-1x1-sparse-nvme-874816-r{r}.json" for r in (1, 2, 3)]]
    c_dense = [f"mogon-a100-nvme-e2e-400k-1x1-m0-nvme-r{r}.json" for r in (1, 2, 3)]
    out = [
        Workload("A 100k sparse", "A", PAIRS_100K, 1, written(a_sparse[0]), written(a_sparse[0]),
                 walls(*a_sparse), False),
        Workload("A 100k dense", "A", PAIRS_100K, 1, PAIRS_100K * 4, written(a_dense[0]), walls(*a_dense), False),
        Workload("A 100k cold", "A", PAIRS_100K, 1, PAIRS_100K * 4, written(a_dense[0]),
                 walls("mon03-dur-100k-similarity-cold.json"), False, ingest_s=A_COLD_INGEST_S),
        Workload("A 100k sparse cold", "A", PAIRS_100K, 1, written("mon03-dur-100k-sparse-similarity-cold.json"),
                 written("mon03-dur-100k-sparse-similarity-cold.json"),
                 walls("mon03-dur-100k-sparse-similarity-cold.json"), False, ingest_s=A_COLD_INGEST_S),
        # 24 B records: the GPU downloads 8 B packed counts and the host expands them.
        Workload("A 100k 24B raw", "A", PAIRS_100K, 1, PAIRS_100K * 8, PAIRS_100K * 24,
                 Runs([r for r in raw if r["records"] == "counts"]), False),
        Workload("B 100k sparse", "B", PAIRS_100K, 1, written(b_sparse[0]), written(b_sparse[0]),
                 walls(*b_sparse), False),
        Workload("B 100k dense", "B", PAIRS_100K, 1, PAIRS_100K * 4, written(b_dense[0]), walls(*b_dense), False),
        Workload("B 100k cold", "B", PAIRS_100K, 1, PAIRS_100K * 4, written(b_dense_cold[0]),
                 walls(*b_dense_cold), False, ingest_s=B_COLD_INGEST_S),
        Workload("B 100k sparse cold", "B", PAIRS_100K, 1, written(b_sparse_cold[0]),
                 written(b_sparse_cold[0]),
                 walls(*b_sparse_cold), False, ingest_s=B_COLD_INGEST_S),
        Workload("C 3-node null", "C", PAIRS_400K, 3, PAIRS_400K * 4, 0.0,
                 walls(*[f"mogon-a100-grid-e2e-400k-3x1-m0-r{r}.json" for r in (1, 2, 3)]), True, setup_s=1.52),
        Workload("C 400k null", "C", PAIRS_400K, 1, PAIRS_400K * 4, 0.0,
                 walls(*[f"mogon-a100-nvme-e2e-400k-1x1-m0-devnull-r{r}.json" for r in (1, 2, 3)]), True),
        Workload("C 400k m20", "C", PAIRS_400K, 1, 0.0, written(c_m20[0]), walls(*c_m20), True,
                 compare_s=C_M20_SEARCH_100K_S * PAIRS_400K / PAIRS_100K),
        Workload("C 400k sparse", "C", PAIRS_400K, 1, written(c_sparse[0]), written(c_sparse[0]),
                 walls(*c_sparse), True),
        Workload("C 400k dense", "C", PAIRS_400K, 1, PAIRS_400K * 4, written(c_dense[0]), walls(*c_dense), True),
    ]
    # Cold runs evict the input from the page cache with posix_fadvise before each run.
    for kind, dl, compare_s in (("dense", PAIRS_400K * 4, None), ("sparse", None, None),
                                ("m20", 0.0, C_M20_SEARCH_100K_S * PAIRS_400K / PAIRS_100K)):
        cold = sorted(p.name for p in results.glob(f"mogon-a100-dur-400k-1x1-{kind}-cold-*-r[1-5].json"))
        if cold:
            out.append(Workload(f"C 400k {kind} cold", "C", PAIRS_400K, 1, written(cold[0]) if dl is None else dl,
                                written(cold[0]), walls(*cold), True, compare_s=compare_s,
                                ingest_s=C_COLD_INGEST_S, group="cold"))
    # 16-bit records: the GPU quantizes before download, so both transfers carry 2 B per pair. r1
    # started after a tmpfs run had evicted the cached input (below), so it reads cold from Lustre.
    dense16 = [f"mogon-a100-dur-400k-1x1-dense16-nvme-874905-r{r}.json" for r in (2, 3)]
    out.append(Workload("C 400k dense16", "C", PAIRS_400K, 1, PAIRS_400K * 2, written(dense16[0]),
                        walls(*dense16), True, group="dense16"))
    out.append(Workload("C 400k dense16 cold start", "C", PAIRS_400K, 1, PAIRS_400K * 2, written(dense16[0]),
                        walls("mogon-a100-dur-400k-1x1-dense16-nvme-874905-r1.json"), True,
                        ingest_s=C_COLD_INGEST_S, group="cold start"))
    if tmpfs is None:
        raise typer.BadParameter("no tmpfs-streams8 rate in mogon-a100-standalone-write-*.jsonl")
    # tmpfs runs, every one of jobs 874905 (default memory policy) and 874965 (default policy, then
    # pages interleaved over all NUMA nodes). gpu0203 sets zone_reclaim_mode=1 on 16 NUMA nodes, so
    # tmpfs pages filling the writers' nodes can make the kernel reclaim the 500 GB cached input.
    # Each run's cache state is classified independently of the model: from the per-node memory
    # log around every run of job 874965 (mogon-a100-tmpfs-memory-874965.jsonl), and in job 874905,
    # which has no log, from whether the next run's ingest read cold (>= 90 s against 58-68 s warm).
    #   warm: input cached at the start and still cached at the end.
    #   cold start: input evicted by the previous run, kept cached during this one; predicted with
    #     the cold Lustre ingest term.
    #   evicted: the cache was reclaimed during the run, or only part of the input was cached at its
    #     start. Reported, but outside the headline MAPE, because the model assumes a fully warm or
    #     a fully cold read and has no term for writes contending with re-reads.
    runs = [  # (stem, kind, cap GB/s or 0 for uncapped, state)
        ("dense-shm-warmup-874905", "dense", 0, "cold start"),
        *[(f"dense-shm-874905-r{r}", "dense", 0, "warm") for r in (1, 2, 3)],
        ("dense-shm-cap2-874905", "dense", 2, "evicted"),
        ("dense-shm-cap4-874905", "dense", 4, "evicted"),
        ("dense-shm-cap8-874905", "dense", 8, "evicted"),
        ("dense-shm-cap12-874905", "dense", 12, "cold start"),
        ("dense-shm-cap16-874905", "dense", 16, "warm"),
        ("dense-shm-cap20-874905", "dense", 20, "warm"),
        ("dense-shm-cap24-874905", "dense", 24, "warm"),
        ("sparse-shm-cap2-874905", "sparse", 2, "warm"),
        ("sparse-shm-cap4-874905", "sparse", 4, "warm"),
        ("sparse-shm-cap6-874905", "sparse", 6, "evicted"),
        ("dense-shm-warmup-874965", "dense", 0, "evicted"),  # 485 of 500 GB cached at its start
        ("dense-shm-cap4-874965", "dense", 4, "evicted"),
        ("dense-shm-cap8-874965", "dense", 8, "evicted"),
        ("sparse-shm-cap6-874965", "sparse", 6, "cold start"),
        ("dense-shm-interleave-cap4-874965", "dense", 4, "warm"),
        ("dense-shm-interleave-cap8-874965", "dense", 8, "warm"),
        ("sparse-shm-interleave-cap6-874965", "sparse", 6, "warm"),
        ("dense-shm-interleave-cap2-874965", "dense", 2, "warm"),
    ]
    for stem, kind, cap, state in runs:
        name = f"mogon-a100-dur-400k-1x1-{stem}.json"
        download = PAIRS_400K * 4 if kind == "dense" else written(name)
        rate = min(cap * 1e9, tmpfs) if cap else tmpfs
        cold_start = state == "cold start" or (state == "evicted" and _ingest_cold(results / name))
        out.append(Workload(f"C 400k {kind} tmpfs {stem.split('shm-')[1]} ({state})", "C", PAIRS_400K, 1, download,
                            written(name), walls(name), True, write_rate=rate,
                            ingest_s=C_COLD_INGEST_S if cold_start else None,
                            group={"warm": "tmpfs", "cold start": "cold start", "evicted": "excluded"}[state]))
    # Durable Lustre writes (jobs 875004 and 875022): buffered page-cache files with fsync, the
    # writer's Lustre path, against the standalone rate of as many concurrent buffered dd streams
    # with fsync into the same stripe-count-1 directory (mean of that stream count's tests).
    # Setup and ingest scale with N at 200k. Job 875004's only finished run followed its dd tests,
    # read its input cold (sketch phase over 90 s) and wrote eight files.
    lustre_tests = [json.loads(line) for p in sorted(results.glob("mogon-a100-lustre-write-*.jsonl"))
                    for line in p.read_text().splitlines() if line.strip()]

    def lustre_rate(streams: int) -> float:
        return statistics.fmean(t["gb_s"] * 1e9 for t in lustre_tests
                                if t["streams"] == streams and t["stripe_count"] == 1)

    for label, pattern, pairs, dl, compare_s, ingest_s, setup_s in (
        ("C 400k sparse Lustre", "400k-1x1-sparse-lustre-s3-*-r[0-9]", PAIRS_400K, None, None, None, None),
        ("C 400k m20 Lustre", "400k-1x1-m20-lustre-s3-*-r[0-9]", PAIRS_400K, 0.0,
         C_M20_SEARCH_100K_S * PAIRS_400K / PAIRS_100K, None, None),
        ("C 200k dense Lustre", "200k-1x1-dense-lustre-s3-*-r[0-9]", PAIRS_200K, PAIRS_200K * 4, None,
         C_INPUT_BYTES_200K / 8.32e9, SYSTEMS["C"].setup_s * 0.5),
        ("C 400k sparse Lustre 8 files (cold start)", "400k-1x1-sparse-lustre-warmup-875004", PAIRS_400K, None, None,
         C_COLD_INGEST_S, None),
    ):
        names = sorted(p.name for p in results.glob(f"mogon-a100-dur-{pattern}.json"))
        if not names:
            continue
        streams = json.loads((results / names[0]).read_text())["write_streams"]
        out.append(Workload(label, "C", pairs, 1, written(names[0]) if dl is None else dl, written(names[0]),
                            walls(*names), True, write_rate=lustre_rate(streams), compare_s=compare_s,
                            ingest_s=ingest_s, setup_s=setup_s, group="lustre"))
    return out


def _ingest_cold(path: Path) -> bool:
    """Whether a run's ingest read its input cold: 90 s or more on one A100 node, against 58-68 s warm."""
    return json.loads(path.read_text())["phases_wall_ms"]["sketch"] >= 90_000


def mape(errors: list[float]) -> float:
    return 100 * statistics.fmean(abs(e) for e in errors)


def main(
    results: Annotated[Path, typer.Option(exists=True, file_okay=False)] = Path("results"),
    output: Annotated[
        Path,
        typer.Option("--output", "-o", help="Output PDF path for model validation parity plot."),
    ] = Path("paper/figures/model_validation.pdf"),
) -> None:
    rows = []
    for w in workloads(results):
        runs = w.runs
        if not runs:
            raise typer.BadParameter(f"no runs for {w.label}")
        overlap, serial, all_over, write_only = predict(w)
        median = statistics.median(runs)
        rows.append((w, overlap, serial, all_over, median, min(runs), max(runs), runs, write_only))
        per_run = ", ".join(f"{100 * (overlap / r - 1):+.1f}" for r in runs)
        typer.echo(f"{w.label:30s} overlap {overlap:7.2f} serial {serial:7.2f} all {all_over:7.2f} "
                   f"median {median:7.2f} ({min(runs):.2f}-{max(runs):.2f}, n={len(runs)}) "
                   f"err {100 * (overlap / median - 1):+.1f}% per-run [{per_run}] "
                   f"search {max(overlapped_terms(w)):.2f} vs {statistics.median(runs.search):.2f} "
                   f"({100 * (max(overlapped_terms(w)) / statistics.median(runs.search) - 1):+.1f}%)")

    def errors(select, column: int) -> list[float]:
        return [r[column] / r[4] - 1 for r in rows if select(r[0])]

    # Headline: every run on local NVMe, tmpfs or /dev/null whose input cache state held through
    # the run. Runs with reclaimed cache and runs writing to shared Lustre are reported beside it.
    def headline(w: Workload) -> bool:
        return w.group not in ("excluded", "lustre")

    groups = {
        "upfront valid (predicate)": upfront_valid,
        "upfront valid extrapolated": lambda w: w.extrapolated and upfront_valid(w),
        "non-interleaved tmpfs": lambda w: "tmpfs" in w.label and "interleave" not in w.label,
        "headline (local, cache intact)": headline,
        "headline without write caps": lambda w: headline(w) and "cap" not in w.label,
        "compare/null-bound (T_cmp>T_wr)": lambda w: headline(w) and overlapped_terms(w)[0] > overlapped_terms(w)[2],
        "all runs": lambda w: True,
        "all but Lustre": lambda w: w.group != "lustre",
        "100k": lambda w: not w.extrapolated,
        "extrapolated headline": lambda w: w.extrapolated and headline(w),
        "extrapolated warm durable": lambda w: w.extrapolated and w.group == "durable",
        "cold 400k": lambda w: w.group == "cold",
        "cold start (tmpfs, dense16)": lambda w: w.group == "cold start",
        "warm tmpfs": lambda w: w.group == "tmpfs",
        "Lustre durable": lambda w: w.group == "lustre",
        "cache reclaimed": lambda w: w.group == "excluded",
    }
    for name, select in groups.items():
        if not any(select(r[0]) for r in rows):
            continue
        per_run = [r[1] / v - 1 for r in rows if select(r[0]) for v in r[7]]
        count = sum(select(r[0]) for r in rows)
        phase = [max(overlapped_terms(r[0])) / statistics.median(r[7].search) - 1 for r in rows if select(r[0])]
        wr_phase = [overlapped_terms(r[0])[2] / statistics.median(r[7].search) - 1 for r in rows if select(r[0])]
        typer.echo(f"MAPE {name:32s} n={count:2d} overlap {mape(errors(select, 1)):5.1f}% "
                   f"serial {mape(errors(select, 2)):5.1f}% all-overlapped {mape(errors(select, 3)):5.1f}% "
                   f"write-only {mape(errors(select, 8)):5.1f}% (s:{mape(wr_phase):4.1f}%)  "
                   f"per-run overlap error {100 * min(per_run):+.1f}% to {100 * max(per_run):+.1f}%  "
                   f"bias {100 * statistics.fmean(errors(select, 1)):+.1f}%  "
                   f"search phase MAPE {mape(phase):.1f}% bias {100 * statistics.fmean(phase):+.1f}% "
                   f"({100 * min(phase):+.1f}% to {100 * max(phase):+.1f}%)")

    # Output sizes from the 100k October 2026 subset instead of the predicted run: bytes per pair of
    # its m_min = 20 edges (mogon-a100-grid-e2e-100k-1x1-m20-r1.json) and of its zero-suppressed
    # sparse output (cuDDL-CPU on the same subset, gh200-cpu-overlap-tiles64-128.jsonl).
    per_pair = {"m20": 4_037_389_296 / PAIRS_100K, "sparse": 6_486_478_848 / PAIRS_100K}

    def from_100k(w: Workload) -> Workload:
        kind = next((k for k in per_pair if k in w.label), None)
        if w.system != "C" or kind is None:
            return w
        size = per_pair[kind] * w.pairs
        return replace(w, write_bytes=size, download_bytes=size if w.download_bytes else 0.0)

    select = groups["extrapolated headline"]
    total = [predict(from_100k(r[0]))[0] / r[4] - 1 for r in rows if select(r[0])]
    phase = [max(overlapped_terms(from_100k(r[0]))) / statistics.median(r[7].search) - 1
             for r in rows if select(r[0])]
    typer.echo(f"MAPE extrapolated headline, output sizes from 100k bytes per pair: overlap {mape(total):.1f}% "
               f"search phase {mape(phase):.1f}%  (m20 {per_pair['m20']:.4f} vs "
               f"{64_282_249_944 / PAIRS_400K:.4f} B/pair, sparse {per_pair['sparse']:.4f} vs "
               f"{103_717_411_244 / PAIRS_400K:.4f})")

    # Limitation 1: resident 400k rate on one A100 (mogon-a100-grid-search-400k-3x8.json, ranks 1) and
    # the 400k one-A100 m_min = 20 indexed search phase with output discarded (median of
    # mogon-a100-grid-e2e-400k-1x1-m20-r{1,2,3}).
    r_400k = 4.18e9
    m20_rate = PAIRS_400K / 15.167
    for name, b, rate in (("dense", 4.0, r_400k), ("sparse", 103_717_411_244 / PAIRS_400K, r_400k),
                          ("m20", 64_282_249_944 / PAIRS_400K, m20_rate)):
        typer.echo(f"streamed ingest, C 400k {name}: {streamed(b, rate):.1f} s")

    cpu_cases, cpu_res_mape, cpu_e2e_mape = validate_cpu_pipeline(results)

    overlap_head = mape(errors(headline, 1))
    serial_head = mape(errors(headline, 2))
    overlap_all = mape(errors(groups["all runs"], 1))
    count_all = len(rows)
    count_head = sum(headline(r[0]) for r in rows)

    fig, ax = plt.subplots(figsize=(3.35, 2.22), constrained_layout=True)
    lo, hi = 4.0, 1300.0
    xs = np.geomspace(lo, hi, 200)
    ax.fill_between(xs, xs * 0.95, xs * 1.05, color="#E5E7EB", alpha=0.75, zorder=0)
    ax.plot(xs, xs, color="#374151", lw=1.0, ls="--", zorder=1)
    colours = {
        "100k": "#0072B2",
        "400k": "#D55E00",
        "tmpfs": "#009E73",
        "cpu": "#7B2CBF",
        "excluded": "#6B7280",
        "lustre": "#CC79A7",
    }
    for w, ov, sr, _all, meas, mn, mx, _runs, _wr in rows:
        if w.group in ("excluded", "lustre"):
            ax.scatter([meas], [ov], facecolors="none", edgecolors=colours[w.group],
                       marker="D" if w.group == "excluded" else "v", s=20, lw=0.85, zorder=5)
            continue
        kind = "tmpfs" if w.group in ("tmpfs", "cold start") and "tmpfs" in w.label else (
            "400k" if w.extrapolated else "100k")
        ax.plot([meas, meas], [ov, sr], color="#9CA3AF", lw=0.9, zorder=2)
        ax.scatter([meas], [sr], facecolors="white", edgecolors="#6B7280", marker="^", s=18, lw=0.75, zorder=3)
        if mx > mn:
            ax.plot([mn, mx], [ov, ov], color="#111827", lw=1.1, zorder=4)
        ax.scatter([meas], [ov], color=colours[kind], marker={"100k": "o", "400k": "s", "tmpfs": "D"}[kind],
                   s=22 if kind == "tmpfs" else 26, edgecolors="white", lw=0.55, zorder=5)
    for _label, pred, meas, _gp, _gm in cpu_cases:
        ax.scatter([meas], [pred], color=colours["cpu"], marker="*", s=46, edgecolors="white", lw=0.4, zorder=6)
    handles = [
        mpl.lines.Line2D([], [], color=colours["100k"], marker="o", ms=4.5, lw=0, label=r"100k GPU (Sys.\,A/B)"),
        mpl.lines.Line2D([], [], color=colours["400k"], marker="s", ms=4.4, lw=0, label=r"400k or 3-node GPU (Sys.\,C)"),
        mpl.lines.Line2D([], [], color=colours["tmpfs"], marker="D", ms=3.8, lw=0, label=r"400k GPU, tmpfs (Sys.\,C)"),
        mpl.lines.Line2D([], [], color=colours["cpu"], marker="*", ms=6.2, lw=0, label=r"\texttt{cuDDL-CPU} (Sys.\,B/C)"),
        mpl.lines.Line2D([], [], color=colours["excluded"], marker="D", mfc="none", ms=3.8, lw=0,
                         label=r"Cache reclaimed in run"),
        mpl.lines.Line2D([], [], color=colours["lustre"], marker="v", mfc="none", ms=4.0, lw=0,
                         label=r"Output to Lustre"),
        mpl.lines.Line2D([], [], color="#6B7280", marker="^", mfc="white", ms=4.4, lw=0.8, label=r"Serial sum"),
    ]
    ax.legend(handles=handles, loc="upper left", frameon=False, fontsize=5.9, handlelength=1.2,
              labelspacing=0.25, borderpad=0.2)
    ax.text(0.97, 0.05,
            rf"GPU overlap ({count_head}): \textbf{{{overlap_head:.1f}\%}}" + "\n"
            + rf"CPU res.\,/ E2E (8): \textbf{{{cpu_res_mape:.1f}\% / {cpu_e2e_mape:.1f}\%}}" + "\n"
            + rf"Serial: {serial_head:.1f}\% \quad All {count_all} GPU: {overlap_all:.1f}\%",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=6.0, color="#111827",
            bbox=dict(facecolor="white", edgecolor="#D1D5DB", lw=0.6, boxstyle="round,pad=0.28"))
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ticks = [5, 10, 30, 100, 300, 1000]
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_xticklabels([str(t) for t in ticks], fontsize=7.2)
    ax.set_yticklabels([str(t) for t in ticks], fontsize=7.2)
    ax.xaxis.set_minor_formatter(mpl.ticker.NullFormatter())
    ax.yaxis.set_minor_formatter(mpl.ticker.NullFormatter())
    ax.set_xlabel("Measured wall time [s]", fontsize=7.8)
    ax.set_ylabel("Predicted wall time [s]", fontsize=7.8)
    ax.grid(True, which="major", color="#F3F4F6", lw=0.6, zorder=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=300)
    fig.savefig(output.with_suffix(".png"), dpi=200)
    plt.close(fig)
    typer.echo(f"Wrote {output}")


if __name__ == "__main__":
    typer.run(main)
