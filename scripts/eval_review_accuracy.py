#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "numpy>=2.0",
#   "sourmash>=4.8",
#   "typer>=0.12",
# ]
# ///
"""Extended accuracy experiments addressing ICS review:
1. 40-chromosome contig containment across 5, 10, 20, 50, 200, 1000 kbp (0% & 2% div)
   comparing cuDDL (4 KiB), BBTools DDL (4 KiB), sourmash (s=1k, s=10k),
   Dashing 2 (16-bit -S2048, 64-bit -S512), sketchlib.rust (4 KiB), and SimdSketch (B=4096, 4 KiB).
   Also records mean(E + H) per contig length.
2. Empirical accidental hash-collision rate on 100 random bacterial RefSeq genomes (4,950 pairs):
   separating truly disjoint pairs (exact_wkid == 0) from genuine k-mer sharing (exact_wkid > 0).
"""

import csv
import gzip
import json
import random
import struct
import subprocess
import tempfile
from pathlib import Path
from typing import Annotated

import numpy as np
import sourmash
import typer


def run_cuddl_exact_and_ddl(
    cuddl_acc_bin: Path, pairs: list[tuple[Path, Path]], tmp: Path
) -> tuple[list[dict], Path]:
    tmp.mkdir(parents=True, exist_ok=True)
    csv_path = tmp / "cases.csv"
    out_path = tmp / "cuddl_acc.json"
    out_csv = tmp / "cuddl_acc.csv"
    with csv_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "generator_seed", "power", "trial", "size_ratio",
            "requested_ani", "actual_ani", "mutation_count",
            "reference_bases", "query_bases",
            "reference_sha256", "query_sha256",
            "reference_path", "query_path",
        ])
        for idx, (ref_p, qry_p) in enumerate(pairs):
            rb = ref_p.stat().st_size
            qb = qry_p.stat().st_size
            w.writerow([
                42, 0, idx, 1, 0.0, 0.0, 0, rb, qb,
                f"ref_{idx}", f"qry_{idx}", str(ref_p.resolve()), str(qry_p.resolve()),
            ])
    proc = subprocess.run(
        [
            str(cuddl_acc_bin),
            "--cases", str(csv_path),
            "--output", str(out_path),
        ],
        capture_output=True, text=True, check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"cuddl-pairwise-accuracy failed:\n{proc.stderr[-1500:]}")
    data = json.loads(out_path.read_text())
    meas = data["measurements"]
    # Emit the CSV format expected by BBToolsPairwiseAccuracy directly from JSON measurements.
    bb_cols = [
        "implementation", "reference_path", "query_path", "reference_bases", "query_bases",
        "orientation", "exact_cardinality", "sketch_cardinality", "cardinality_signed_error",
        "cardinality_absolute_error", "cardinality_relative_error", "cardinality_absolute_relative_error",
        "sketch_cardinality_bbtools", "cardinality_bbtools_signed_error", "cardinality_bbtools_absolute_error",
        "cardinality_bbtools_relative_error", "cardinality_bbtools_absolute_relative_error",
        "sketch_cardinality_paper", "cardinality_paper_signed_error", "cardinality_paper_absolute_error",
        "cardinality_paper_relative_error", "cardinality_paper_absolute_relative_error",
        "lower", "equal", "higher", "both_empty", "exact_containment", "sketch_containment",
        "containment_signed_error", "containment_absolute_error", "exact_completeness",
        "sketch_completeness", "completeness_signed_error", "completeness_absolute_error",
        "exact_wkid", "sketch_wkid", "wkid_signed_error", "wkid_absolute_error",
        "exact_ani", "sketch_ani", "ani_signed_error", "ani_absolute_error", "aligned_fraction",
    ]
    with out_csv.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(bb_cols)
        for m in meas:
            if m["implementation"]["name"] != "cuddl":
                continue
            c = m["case"]
            met = m["metrics"]
            idx = int(c["trial"])
            ref_p, qry_p = pairs[idx]
            row = []
            for col in bb_cols:
                if col == "implementation":
                    row.append("cuddl")
                elif col == "reference_path":
                    row.append(str(ref_p.resolve()))
                elif col == "query_path":
                    row.append(str(qry_p.resolve()))
                elif col == "reference_bases":
                    row.append(str(ref_p.stat().st_size))
                elif col == "query_bases":
                    row.append(str(qry_p.stat().st_size))
                elif col == "exact_ani":
                    row.append(str(met.get("exact_set_derived_ani", 0.0)))
                else:
                    row.append(str(met.get(col, 0.0)))
            w.writerow(row)
    return meas, out_csv


def run_bbtools_from_csv(cuddl_csv: Path, bbtools_jar: Path, java_cls_dir: Path) -> list[dict]:
    cmd = [
        "java", "-Xmx8g", "-cp", f"{bbtools_jar}:{java_cls_dir}",
        "BBToolsPairwiseAccuracy", str(cuddl_csv), "8",
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if proc.returncode != 0:
        raise RuntimeError(f"BBToolsPairwiseAccuracy failed:\n{proc.stderr[-1500:]}")
    rows = list(csv.DictReader(proc.stdout.splitlines()))
    return rows


def run_dashing2_matrix(
    d2_bin: Path, extra_flags: list[str], ref_list: Path, qry_list: Path, out: Path
) -> np.ndarray:
    cmd = [
        str(d2_bin), "cmp", "-k25", *extra_flags,
        "-F", str(ref_list), "-Q", str(qry_list), "--cmpout", str(out),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if proc.returncode != 0:
        raise RuntimeError(f"dashing2 failed: {' '.join(cmd)}\n{proc.stderr[-1000:]}")
    rows = []
    for line in out.read_text().splitlines():
        if not line or line.startswith("#"):
            continue
        rows.append([float(x) for x in line.split("\t")[1:]])
    return np.array(rows, dtype=np.float64)


def run_simdsketch_diag_ani(
    simd_bin: Path, ref_paths: list[Path], qry_paths: list[Path], tmp: Path
) -> np.ndarray:
    """Run SimdSketch at B=4096, b=8 (4 KiB/sketch) and return diagonal Jaccard-derived ANI."""
    ref_list = tmp / "ss_ref.list"
    qry_list = tmp / "ss_qry.list"
    ref_list.write_text("\n".join(str(p) for p in ref_paths) + "\n")
    qry_list.write_text("\n".join(str(p) for p in qry_paths) + "\n")
    ref_sk = tmp / "ss_ref.sk"
    out_f = tmp / "ss_out.txt"
    subprocess.run(
        [str(simd_bin), "sketch", "-k", "25", "-s", "4096", "-b", "8", "--output", str(ref_sk), str(ref_list)],
        check=True, capture_output=True,
    )
    proc = subprocess.run(
        [str(simd_bin), "compare", "--references", str(ref_sk), "--queries", str(qry_list), "--output", str(out_f)],
        check=True, capture_output=True, text=True,
    )
    n = len(ref_paths)
    # Check if output is binary float32 (n*n*4 bytes) or TSV text
    raw = out_f.read_bytes()
    ani = np.zeros(n)
    if len(raw) == n * n * 4:
        mat = np.frombuffer(raw, dtype=np.float32).reshape(n, n)
        for i in range(n):
            j = float(mat[i, i])
            ani[i] = (2.0 * j / (1.0 + j)) ** (1.0 / 25.0) if j > 0.0 else 0.0
    else:
        txt = raw.decode("utf-8", errors="ignore").splitlines()
        # Parse TSV or CSV: query, ref, jaccard/dist
        for idx, line in enumerate(txt):
            parts = line.split("\t")
            if len(parts) >= 3:
                try:
                    j = float(parts[2])
                    # Check diagonal by matching names or order
                    if Path(parts[0]).stem == qry_paths[idx % n].stem or len(txt) == n:
                        ani[idx % n] = (2.0 * j / (1.0 + j)) ** (1.0 / 25.0) if j > 0.0 else 0.0
                except ValueError:
                    pass
    return ani


def ani_from_wkid(w: float, k: int = 25) -> float:
    if w <= 0.0:
        return 0.0
    if w >= 1.0:
        return 1.0
    return w ** (1.0 / k)


def main(
    genome_dir: Annotated[Path, typer.Option()] = Path("data/bacteria-refseq"),
    dashing2_bin: Annotated[Path, typer.Option()] = Path("build/subprojects/dashing2/dashing2"),
    sketchlib_bin: Annotated[Path, typer.Option()] = Path("build/subprojects/sketchlib/sketchlib"),
    simdsketch_bin: Annotated[Path, typer.Option()] = Path("build/benchmarks/simdsketch-benchmark"),
    cuddl_acc_bin: Annotated[Path, typer.Option()] = Path("build/benchmarks/cuddl-pairwise-accuracy"),
    bbtools_jar: Annotated[Path, typer.Option()] = Path("subprojects/bbmap/bbtools.jar"),
    java_cls_dir: Annotated[Path, typer.Option()] = Path("build/java-lsp"),
    out_json: Annotated[Path, typer.Option()] = Path("results/review-accuracy-extended.json"),
) -> None:
    genomes = sorted(genome_dir.rglob("*.fna.gz"))
    results: dict = {}

    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)

        # === PART 1: Empirical accidental hash collision rate on 100 random RefSeq genomes ===
        rng_g = random.Random(12345)
        sample100 = rng_g.sample(genomes, min(100, len(genomes)))
        g_dir = tmp / "rand100"
        g_dir.mkdir()
        g_fna = []
        for idx, gp in enumerate(sample100):
            p = g_dir / f"r_{idx:03d}.fna"
            with gzip.open(gp, "rb") as src, p.open("wb") as dst:
                dst.write(src.read())
            g_fna.append(p)
        pairs_100 = [(g_fna[i], g_fna[j]) for i in range(len(g_fna)) for j in range(i + 1, len(g_fna))]
        typer.echo(f"Part 1: accidental collision audit on {len(pairs_100)} random RefSeq pairs...")
        meas_100, _ = run_cuddl_exact_and_ddl(cuddl_acc_bin, pairs_100, tmp / "p1_coll")
        exact_zero_e = []
        exact_pos_e = []
        for m in meas_100:
            if m["implementation"]["name"] != "cuddl":
                continue
            met = m["metrics"]
            if met.get("orientation") != "query_to_reference":
                continue
            ex_w = float(met["exact_wkid"])
            eq = int(met["equal"])
            if ex_w == 0.0:
                exact_zero_e.append(eq)
            else:
                exact_pos_e.append(eq)
        ez = np.array(exact_zero_e, dtype=np.int64)
        ep = np.array(exact_pos_e, dtype=np.int64)
        total_p = len(ez) + len(ep)
        coll_stats = {
            "total_pairs": total_p,
            "exact_zero_pairs": int(len(ez)),
            "exact_pos_pairs": int(len(ep)),
            "exact_pos_pct": round(100.0 * len(ep) / max(total_p, 1), 2),
            "accidental_mean_E_given_exact0": round(float(ez.mean()), 4) if len(ez) else 0.0,
            "accidental_p_E_ge_1_given_exact0_pct": round(100.0 * float((ez >= 1).mean()), 2) if len(ez) else 0.0,
            "accidental_p_E_ge_2_given_exact0_pct": round(100.0 * float((ez >= 2).mean()), 2) if len(ez) else 0.0,
            "accidental_p_E_ge_6_given_exact0_pct": round(100.0 * float((ez >= 6).mean()), 4) if len(ez) else 0.0,
            "overall_E_ge_1_pct": round(100.0 * float((ez >= 1).sum() + (ep >= 1).sum()) / max(total_p, 1), 2),
            "frac_of_E_ge_1_that_are_accidental_pct": round(
                100.0 * float((ez >= 1).sum()) / max(float((ez >= 1).sum() + (ep >= 1).sum()), 1.0), 2
            ),
        }
        results["part1_accidental_collisions_100rand"] = coll_stats
        typer.echo(f"  Collision stats: {json.dumps(coll_stats)}")

        # Synthetic truly disjoint 4 Mbp random DNA pairs (20 sequences -> 190 guaranteed exact=0 pairs)
        syn_dir = tmp / "syn_disjoint"
        syn_dir.mkdir()
        rng_syn = np.random.default_rng(2026)
        bases = np.array(list("ACGT"))
        syn_paths = []
        for idx in range(20):
            seq = "".join(bases[rng_syn.integers(0, 4, size=4_000_000)])
            p = syn_dir / f"syn_{idx:02d}.fna"
            p.write_text(f">syn_{idx}\n{seq}\n")
            syn_paths.append(p)
        syn_pairs = [(syn_paths[i], syn_paths[j]) for i in range(len(syn_paths)) for j in range(i + 1, len(syn_paths))]
        typer.echo(f"Part 1b: accidental collision audit on {len(syn_pairs)} synthetic disjoint 4 Mbp pairs...")
        meas_syn, _ = run_cuddl_exact_and_ddl(cuddl_acc_bin, syn_pairs, tmp / "p1_syn")
        syn_e = []
        syn_ex = []
        for m in meas_syn:
            if m["implementation"]["name"] != "cuddl":
                continue
            met = m["metrics"]
            if met.get("orientation") != "query_to_reference":
                continue
            syn_ex.append(float(met["exact_wkid"]))
            syn_e.append(int(met["equal"]))
        se = np.array(syn_e, dtype=np.int64)
        syn_stats = {
            "pairs": int(len(se)),
            "exact_zero_pairs": int(sum(x == 0.0 for x in syn_ex)),
            "mean_E": round(float(se.mean()), 4),
            "p_E_ge_1_pct": round(100.0 * float((se >= 1).mean()), 2),
            "p_E_ge_2_pct": round(100.0 * float((se >= 2).mean()), 2),
            "p_E_ge_6_pct": round(100.0 * float((se >= 6).mean()), 4),
            "max_E": int(se.max()) if len(se) else 0,
        }
        results["part1b_synthetic_disjoint_4mbp"] = syn_stats
        typer.echo(f"  Synthetic disjoint stats: {json.dumps(syn_stats)}")

        # === PART 2: Extended 40-chromosome contig containment (5, 10, 20, 50, 200, 1000 kbp) ===
        chroms: list[tuple[str, str]] = []
        for g in genomes[:400]:
            with gzip.open(g, "rt") as f:
                cur: list[str] = []
                best = ""
                for line in f:
                    if line.startswith(">"):
                        s = "".join(cur).upper()
                        if len(s) > len(best):
                            best = s
                        cur = []
                    else:
                        cur.append(line.strip())
                s = "".join(cur).upper()
                if len(s) > len(best):
                    best = s
            if len(best) >= 2_500_000:
                chroms.append((g.stem, best))
            if len(chroms) == 40:
                break
        typer.echo(f"\nPart 2: {len(chroms)} chromosomes, mean_len={np.mean([len(s) for _, s in chroms]):.0f}")

        rng = random.Random(42)
        mut_map = {"A": "CGT", "C": "AGT", "G": "ACT", "T": "ACG"}

        def mutate(seq: str, rate: float) -> str:
            if rate == 0.0:
                return seq
            arr = list(seq)
            for idx, ch in enumerate(arr):
                if ch in mut_map and rng.random() < rate:
                    arr[idx] = mut_map[ch][rng.randrange(3)]
            return "".join(arr)

        chrom_dir = tmp / "chroms"
        chrom_dir.mkdir()
        chrom_paths = []
        sm_chr_1k, sm_chr_10k = [], []
        for idx, (name, seq) in enumerate(chroms):
            p = chrom_dir / f"chr_{idx:02d}.fna"
            p.write_text(f">{name}\n{seq}\n")
            chrom_paths.append(p)
            m1 = sourmash.MinHash(n=0, ksize=25, scaled=1000)
            m1.add_sequence(seq, force=True)
            sm_chr_1k.append(m1)
            m10 = sourmash.MinHash(n=0, ksize=25, scaled=10000)
            m10.add_sequence(seq, force=True)
            sm_chr_10k.append(m10)

        p2_rows = []
        for clen in [5_000, 10_000, 20_000, 50_000, 200_000, 1_000_000]:
            for div in [0.0, 0.02]:
                c_paths = []
                c_seqs = []
                for idx, (name, full) in enumerate(chroms):
                    start = (len(full) - clen) // 2
                    cseq = mutate(full[start : start + clen], div)
                    c_seqs.append(cseq)
                    cp = tmp / f"ctg_{clen}_{int(div*100)}_{idx:02d}.fna"
                    cp.write_text(f">{name}_ctg\n{cseq}\n")
                    c_paths.append(cp)

                sub_tmp = tmp / f"p2_{clen}_{int(div*100)}"
                sub_tmp.mkdir()
                meas, cuddl_csv = run_cuddl_exact_and_ddl(
                    cuddl_acc_bin, list(zip(chrom_paths, c_paths)), sub_tmp
                )
                ex_w = np.zeros(len(chroms))
                ex_a = np.zeros(len(chroms))
                cd_w = np.zeros(len(chroms))
                cd_a = np.zeros(len(chroms))
                cd_eh = np.zeros(len(chroms))
                for m in meas:
                    if m["implementation"]["name"] != "cuddl":
                        continue
                    c = m["case"]
                    met = m["metrics"]
                    if met.get("orientation") != "query_to_reference":
                        continue
                    t = int(c["trial"])
                    ex_w[t] = float(met["exact_wkid"])
                    ex_a[t] = float(met["exact_set_derived_ani"])
                    cd_w[t] = float(met["sketch_wkid"])
                    cd_a[t] = float(met["sketch_ani"])
                    cd_eh[t] = int(met["equal"]) + int(met["higher"])

                # BBTools DDL via BBToolsPairwiseAccuracy
                bb_rows = run_bbtools_from_csv(cuddl_csv, bbtools_jar, java_cls_dir)
                bb_w = np.zeros(len(chroms))
                bb_a = np.zeros(len(chroms))
                bb_idx = 0
                for row in bb_rows:
                    if row.get("orientation") == "query_to_reference":
                        bb_w[bb_idx] = float(row["sketch_wkid"])
                        bb_a[bb_idx] = float(row["sketch_ani"])
                        bb_idx += 1

                # sourmash
                sm1k_w, sm1k_a = np.zeros(len(chroms)), np.zeros(len(chroms))
                sm10k_w, sm10k_a = np.zeros(len(chroms)), np.zeros(len(chroms))
                empty_1k, empty_10k = 0, 0
                for idx, cseq in enumerate(c_seqs):
                    m1 = sourmash.MinHash(n=0, ksize=25, scaled=1000)
                    m1.add_sequence(cseq, force=True)
                    if len(m1) == 0:
                        empty_1k += 1
                    w1 = m1.contained_by(sm_chr_1k[idx]) if len(m1) > 0 else 0.0
                    sm1k_w[idx], sm1k_a[idx] = w1, ani_from_wkid(w1)

                    m10 = sourmash.MinHash(n=0, ksize=25, scaled=10000)
                    m10.add_sequence(cseq, force=True)
                    if len(m10) == 0:
                        empty_10k += 1
                    w10 = m10.contained_by(sm_chr_10k[idx]) if len(m10) > 0 else 0.0
                    sm10k_w[idx], sm10k_a[idx] = w10, ani_from_wkid(w10)

                # Dashing 2
                f_ref = sub_tmp / "ref.list"
                f_qry = sub_tmp / "qry.list"
                f_ref.write_text("\n".join(str(p) for p in chrom_paths) + "\n")
                f_qry.write_text("\n".join(str(p) for p in c_paths) + "\n")
                d2_16_w = np.diag(run_dashing2_matrix(
                    dashing2_bin, ["-S2048", "--full", "--regsize", "2", "--symmetric-containment"],
                    f_ref, f_qry, sub_tmp / "d2_16.tsv",
                ))
                d2_16_a = np.array([ani_from_wkid(w) for w in d2_16_w])
                d2_64_w = np.diag(run_dashing2_matrix(
                    dashing2_bin, ["-S512", "--full", "--symmetric-containment"],
                    f_ref, f_qry, sub_tmp / "d2_64.tsv",
                ))
                d2_64_a = np.array([ani_from_wkid(w) for w in d2_64_w])

                # sketchlib.rust
                sk_ref_pref = sub_tmp / "sk_ref"
                sk_qry_pref = sub_tmp / "sk_qry"
                subprocess.run(
                    [str(sketchlib_bin), "sketch", "-k", "25", "-s", "2048", "-o", str(sk_ref_pref), *[str(p) for p in chrom_paths]],
                    check=True, capture_output=True,
                )
                subprocess.run(
                    [str(sketchlib_bin), "sketch", "-k", "25", "-s", "2048", "-o", str(sk_qry_pref), *[str(p) for p in c_paths]],
                    check=True, capture_output=True,
                )
                sk_dist = subprocess.run(
                    [str(sketchlib_bin), "dist", "-k", "25", "--ani", str(sk_ref_pref), str(sk_qry_pref)],
                    check=True, capture_output=True, text=True,
                )
                sk_map = {}
                for line in sk_dist.stdout.splitlines():
                    parts = line.split("\t")
                    if len(parts) >= 3:
                        try:
                            sk_map[(Path(parts[0]).stem, Path(parts[1]).stem)] = float(parts[2])
                        except ValueError:
                            continue
                sk_a = np.zeros(len(chroms))
                for idx in range(len(chroms)):
                    val = sk_map.get((chrom_paths[idx].stem, c_paths[idx].stem), 0.0)
                    sk_a[idx] = val / 100.0 if val > 1.5 else val

                # SimdSketch (B=4096, b=8 = 4 KiB)
                ss_a = run_simdsketch_diag_ani(simdsketch_bin, chrom_paths, c_paths, sub_tmp)

                def err_stats(ew: np.ndarray, ea: np.ndarray, has_wkid: bool = True) -> dict:
                    out = {
                        "ani_med_pp": round(float(np.median(np.abs(ea - ex_a))) * 100.0, 2),
                        "ani_p95_pp": round(float(np.percentile(np.abs(ea - ex_a), 95)) * 100.0, 2),
                    }
                    if has_wkid:
                        out["wkid_med_pp"] = round(float(np.median(np.abs(ew - ex_w))) * 100.0, 2)
                        out["wkid_p95_pp"] = round(float(np.percentile(np.abs(ew - ex_w), 95)) * 100.0, 2)
                    return out

                rec = {
                    "contig_len": clen,
                    "div": div,
                    "mean_E_plus_H": round(float(np.mean(cd_eh)), 1),
                    "eh_zero_contigs": int((cd_eh == 0).sum()),
                    "cuddl_4KiB": err_stats(cd_w, cd_a),
                    "bbtools_4KiB": err_stats(bb_w, bb_a),
                    "sourmash_s1000_31KiB": {**err_stats(sm1k_w, sm1k_a), "empty_contigs": empty_1k},
                    "sourmash_s10000_3KiB": {**err_stats(sm10k_w, sm10k_a), "empty_contigs": empty_10k},
                    "dashing2_16b_S2048_16KiB": err_stats(d2_16_w, d2_16_a),
                    "dashing2_64b_S512_4KiB": err_stats(d2_64_w, d2_64_a),
                    "sketchlib_s2048_jaccard": err_stats(np.zeros_like(ex_w), sk_a, has_wkid=False),
                    "simdsketch_B4096_jaccard": err_stats(np.zeros_like(ex_w), ss_a, has_wkid=False),
                }
                p2_rows.append(rec)
                typer.echo(
                    f"len={clen:7d} div={div:.2f} E+H={rec['mean_E_plus_H']:5.1f} | "
                    f"cuDDL w={rec['cuddl_4KiB']['wkid_p95_pp']:5.2f} a={rec['cuddl_4KiB']['ani_p95_pp']:5.2f} | "
                    f"BBT w={rec['bbtools_4KiB']['wkid_p95_pp']:5.2f} a={rec['bbtools_4KiB']['ani_p95_pp']:5.2f} | "
                    f"sm1k w={rec['sourmash_s1000_31KiB']['wkid_p95_pp']:5.2f} a={rec['sourmash_s1000_31KiB']['ani_p95_pp']:5.2f} | "
                    f"d2_16 w={rec['dashing2_16b_S2048_16KiB']['wkid_p95_pp']:5.2f} a={rec['dashing2_16b_S2048_16KiB']['ani_p95_pp']:5.2f} | "
                    f"skl a={rec['sketchlib_s2048_jaccard']['ani_p95_pp']:5.2f} | "
                    f"ss a={rec['simdsketch_B4096_jaccard']['ani_p95_pp']:5.2f}"
                )

        results["part2_contig_containment_40chr"] = p2_rows

    out_json.write_text(json.dumps(results, indent=2))
    typer.echo(f"\nSaved {out_json}")


if __name__ == "__main__":
    typer.run(main)
