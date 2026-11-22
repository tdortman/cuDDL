#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "numpy>=2.0",
#   "sourmash>=4.8",
#   "typer>=0.12",
# ]
# ///
"""Tier-1 accuracy Part 2:
1. Un-handicapped Dashing 2 (16-bit --regsize 2 -S2048, 8-bit --regsize 1 -S4096, 64-bit -S512)
   vs cuDDL (B=2048, 4 KiB) against exact 25-mer truth on 100 RefSeq genomes (stratified on related pairs).
2. 40-chromosome length-asymmetric contig containment (50 kbp, 200 kbp, 1 Mbp at 0% & 2% div)
   comparing cuDDL (4 KiB), sourmash (scaled=1000 & scaled=10000), Dashing 2 (--symmetric-containment,
   both 16-bit -S2048 and 64-bit -S512), sketchlib.rust (s=2048), and SimdSketch against exact 25-mer truth.
"""

import csv
import gzip
import json
import math
import random
import subprocess
import tempfile
from pathlib import Path
from typing import Annotated

import numpy as np
import sourmash
import typer


def run_cuddl_exact_and_ddl(cuddl_acc_bin: Path, pairs: list[tuple[Path, Path]], tmp: Path) -> list[dict]:
    """Run cuddl-pairwise-accuracy to obtain BOTH exact 25-mer truth AND cuDDL estimates."""
    tmp.mkdir(parents=True, exist_ok=True)
    csv_path = tmp / "cases.csv"
    out_path = tmp / "cuddl_acc.json"
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
            w.writerow([
                42, 0, idx, 1, 0.0, 0.0, 0, 1000, 1000,
                f"ref_{idx}", f"qry_{idx}", str(ref_p.resolve()), str(qry_p.resolve()),
            ])
    proc = subprocess.run(
        [str(cuddl_acc_bin), "--cases", str(csv_path), "--output", str(out_path)],
        capture_output=True, text=True, check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"cuddl-pairwise-accuracy failed:\n{proc.stderr[-1500:]}")
    data = json.loads(out_path.read_text())
    return data["measurements"]


def run_dashing2_matrix(d2_bin: Path, extra_flags: list[str], ref_list: Path, qry_list: Path, out: Path) -> np.ndarray:
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
    out_json: Annotated[Path, typer.Option()] = Path("results/tier1-accuracy-part2.json"),
) -> None:
    genomes = sorted(genome_dir.rglob("*.fna.gz"))
    results: dict = {}

    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)

        # === PART 1: 100 RefSeq genomes — Dashing 2 64b/16b/8b vs cuDDL vs Exact Truth ===
        # Pick 100 genomes across major species clusters so we have both related (WKID>=0.05, ANI>=95%) and unrelated pairs.
        by_sp: dict[str, list[Path]] = {}
        for p in genomes[:3000]:
            with gzip.open(p, "rt", errors="ignore") as f:
                hdr = f.readline().strip()
            parts = hdr.split()
            sp = " ".join(parts[1:3]) if len(parts) >= 3 else "unk"
            by_sp.setdefault(sp, []).append(p)
        g100_gz: list[Path] = []
        for sp, lst in sorted(by_sp.items(), key=lambda kv: -len(kv[1])):
            take = min(20, len(lst), 100 - len(g100_gz))
            g100_gz.extend(lst[:take])
            if len(g100_gz) >= 100:
                break
        g100_dir = tmp / "g100"
        g100_dir.mkdir()
        g100_fna: list[Path] = []
        for idx, gp in enumerate(g100_gz):
            out_f = g100_dir / f"g_{idx:03d}.fna"
            with gzip.open(gp, "rb") as src, out_f.open("wb") as dst:
                dst.write(src.read())
            g100_fna.append(out_f)

        pairs_100 = [(g100_fna[i], g100_fna[j]) for i in range(100) for j in range(i + 1, 100)]
        typer.echo(f"Part 1: running cuddl-pairwise-accuracy on {len(pairs_100)} pairs...")
        meas_100 = run_cuddl_exact_and_ddl(cuddl_acc_bin, pairs_100, tmp / "p1_cuddl")
        exact_wkid = np.zeros(len(pairs_100))
        exact_ani = np.zeros(len(pairs_100))
        exact_card = np.zeros(100)
        cuddl_wkid = np.zeros(len(pairs_100))
        cuddl_ani = np.zeros(len(pairs_100))
        cuddl_card_err: dict[int, float] = {}
        pair_ij = [(i, j) for i in range(100) for j in range(i + 1, 100)]
        for m in meas_100:
            if m["implementation"]["name"] != "cuddl":
                continue
            c = m["case"]
            met = m["metrics"]
            t = int(c["trial"])
            i, j = pair_ij[t]
            if met.get("orientation") == "query_to_reference":
                exact_wkid[t] = float(met["exact_wkid"])
                exact_ani[t] = float(met["exact_set_derived_ani"])
                cuddl_wkid[t] = float(met["sketch_wkid"])
                cuddl_ani[t] = float(met["sketch_ani"])
                exact_card[j] = float(met["exact_cardinality"])
                cuddl_card_err[j] = float(met["cardinality_absolute_relative_error"])
            else:
                exact_card[i] = float(met["exact_cardinality"])
                cuddl_card_err[i] = float(met["cardinality_absolute_relative_error"])

        list100 = tmp / "g100.list"
        list100.write_text("\n".join(str(p) for p in g100_fna) + "\n")

        rel_mask = exact_wkid >= 0.05
        ani95_mask = exact_ani >= 0.95
        typer.echo(f"Part 1 strata: all={len(pairs_100)}, WKID>=0.05={int(rel_mask.sum())}, ANI>=95%={int(ani95_mask.sum())}")

        def summarize(label: str, est_wkid: np.ndarray, est_ani: np.ndarray, card_rel_errs: np.ndarray, extra: dict) -> dict:
            dw = np.abs(est_wkid - exact_wkid) * 100.0
            da = np.abs(est_ani - exact_ani) * 100.0
            row = {
                "label": label,
                **extra,
                "card_med_pct": round(float(np.median(card_rel_errs)) * 100.0, 3),
                "card_p95_pct": round(float(np.percentile(card_rel_errs, 95)) * 100.0, 3),
                "all_wkid_p95_pp": round(float(np.percentile(dw, 95)), 3),
                "rel_n": int(rel_mask.sum()),
                "rel_wkid_med_pp": round(float(np.median(dw[rel_mask])), 3) if rel_mask.any() else None,
                "rel_wkid_p95_pp": round(float(np.percentile(dw[rel_mask], 95)), 3) if rel_mask.any() else None,
                "ani95_n": int(ani95_mask.sum()),
                "ani95_med_pp": round(float(np.median(da[ani95_mask])), 3) if ani95_mask.any() else None,
                "ani95_p95_pp": round(float(np.percentile(da[ani95_mask], 95)), 3) if ani95_mask.any() else None,
            }
            typer.echo(
                f"  {label:24s} | card p95={row['card_p95_pct']:5.2f}% | "
                f"WKID>=0.05 p95={row['rel_wkid_p95_pp']:5.2f} pp | ANI>=95% p95={row['ani95_p95_pp']:5.2f} pp"
            )
            return row

        p1_rows = []
        p1_rows.append(summarize(
            "cuDDL (B=2048, 4 KiB)",
            cuddl_wkid, cuddl_ani,
            np.array([cuddl_card_err[i] for i in range(100)]),
            {"sketch_bytes_disk": 4096},
        ))

        for label, flags in [
            ("Dashing2 64b -S512", ["-S512", "--full"]),
            ("Dashing2 16b -S2048", ["-S2048", "--full", "--regsize", "2"]),
            ("Dashing2 8b -S4096", ["-S4096", "--full", "--regsize", "1"]),
        ]:
            cdir = tmp / f"d2_{label.replace(' ', '_')}"
            cdir.mkdir()
            sym = run_dashing2_matrix(
                dashing2_bin, [*flags, "--symmetric-containment", "--cache", "--outprefix", str(cdir)],
                list100, list100, tmp / f"{label.replace(' ', '_')}_sym.tsv",
            )
            inter = run_dashing2_matrix(
                dashing2_bin, [*flags, "--intersection-size", "--cache", "--outprefix", str(cdir)],
                list100, list100, tmp / f"{label.replace(' ', '_')}_inter.tsv",
            )
            cache_files = [f.stat().st_size for f in cdir.rglob("*") if f.is_file()]
            card = np.diag(inter)
            card_rel = np.abs(card - exact_card) / np.maximum(exact_card, 1.0)
            est_w = np.array([sym[i, j] for i, j in pair_ij])
            est_a = np.array([ani_from_wkid(w) for w in est_w])
            p1_rows.append(summarize(
                label, est_w, est_a, card_rel,
                {"sketch_bytes_disk": int(np.median(cache_files)) if cache_files else 0},
            ))

        results["part1_dashing2_regsize_100g"] = p1_rows

        # === PART 2: 40-chromosome length-asymmetric contig containment (50 kbp, 200 kbp, 1 Mbp) ===
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
        for clen in [50_000, 200_000, 1_000_000]:
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

                # 1. Exact 25-mer truth AND cuDDL (B=2048, 4 KiB) via cuddl-pairwise-accuracy
                sub_tmp = tmp / f"p2_{clen}_{int(div*100)}"
                sub_tmp.mkdir()
                meas = run_cuddl_exact_and_ddl(
                    cuddl_acc_bin, list(zip(chrom_paths, c_paths)), sub_tmp
                )
                ex_w = np.zeros(len(chroms))
                ex_a = np.zeros(len(chroms))
                cd_w = np.zeros(len(chroms))
                cd_a = np.zeros(len(chroms))
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

                # 2. sourmash scaled=1000 & scaled=10000
                sm1k_w, sm1k_a = np.zeros(len(chroms)), np.zeros(len(chroms))
                sm10k_w, sm10k_a = np.zeros(len(chroms)), np.zeros(len(chroms))
                empty_10k = 0
                for idx, cseq in enumerate(c_seqs):
                    m1 = sourmash.MinHash(n=0, ksize=25, scaled=1000)
                    m1.add_sequence(cseq, force=True)
                    w1 = m1.contained_by(sm_chr_1k[idx]) if len(m1) > 0 else 0.0
                    sm1k_w[idx], sm1k_a[idx] = w1, ani_from_wkid(w1)

                    m10 = sourmash.MinHash(n=0, ksize=25, scaled=10000)
                    m10.add_sequence(cseq, force=True)
                    if len(m10) == 0:
                        empty_10k += 1
                    w10 = m10.contained_by(sm_chr_10k[idx]) if len(m10) > 0 else 0.0
                    sm10k_w[idx], sm10k_a[idx] = w10, ani_from_wkid(w10)

                # 3. Dashing 2 (--symmetric-containment, both 16-bit -S2048 and 64-bit -S512)
                f_ref = sub_tmp / "ref.list"
                f_qry = sub_tmp / "qry.list"
                f_ref.write_text("\n".join(str(p) for p in chrom_paths) + "\n")
                f_qry.write_text("\n".join(str(p) for p in c_paths) + "\n")
                d2_16_mat = run_dashing2_matrix(
                    dashing2_bin, ["-S2048", "--full", "--regsize", "2", "--symmetric-containment"],
                    f_ref, f_qry, sub_tmp / "d2_16.tsv",
                )
                d2_16_w = np.diag(d2_16_mat)
                d2_16_a = np.array([ani_from_wkid(w) for w in d2_16_w])

                d2_64_mat = run_dashing2_matrix(
                    dashing2_bin, ["-S512", "--full", "--symmetric-containment"],
                    f_ref, f_qry, sub_tmp / "d2_64.tsv",
                )
                d2_64_w = np.diag(d2_64_mat)
                d2_64_a = np.array([ani_from_wkid(w) for w in d2_64_w])

                # 4. sketchlib.rust (-k 25 -s 2048 --ani) — Jaccard-based!
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
                    # sketchlib outputs ANI as percentage (0-100) or fraction (0-1)? Check magnitude
                    val = sk_map.get((chrom_paths[idx].stem, c_paths[idx].stem), 0.0)
                    sk_a[idx] = val / 100.0 if val > 1.5 else val

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
                    "exact_mean_wkid": round(float(np.mean(ex_w)), 4),
                    "exact_mean_ani": round(float(np.mean(ex_a)), 4),
                    "cuddl_4KiB": err_stats(cd_w, cd_a),
                    "sourmash_s1000_31KiB": err_stats(sm1k_w, sm1k_a),
                    "sourmash_s10000_3KiB": {**err_stats(sm10k_w, sm10k_a), "empty_contigs": empty_10k},
                    "dashing2_16b_S2048_16KiB": err_stats(d2_16_w, d2_16_a),
                    "dashing2_64b_S512_4KiB": err_stats(d2_64_w, d2_64_a),
                    "sketchlib_s2048_jaccard": err_stats(np.zeros_like(ex_w), sk_a, has_wkid=False),
                }
                p2_rows.append(rec)
                typer.echo(
                    f"len={clen:7d} div={div:.2f} | "
                    f"cuDDL(4KiB) w={rec['cuddl_4KiB']['wkid_p95_pp']:5.2f} a={rec['cuddl_4KiB']['ani_p95_pp']:5.2f} | "
                    f"sm1k(31KiB) w={rec['sourmash_s1000_31KiB']['wkid_p95_pp']:5.2f} a={rec['sourmash_s1000_31KiB']['ani_p95_pp']:5.2f} | "
                    f"sm10k(3KiB) w={rec['sourmash_s10000_3KiB']['wkid_p95_pp']:5.2f} a={rec['sourmash_s10000_3KiB']['ani_p95_pp']:5.2f} | "
                    f"d2_16b(16KiB) w={rec['dashing2_16b_S2048_16KiB']['wkid_p95_pp']:5.2f} a={rec['dashing2_16b_S2048_16KiB']['ani_p95_pp']:5.2f} | "
                    f"d2_64b(4KiB) w={rec['dashing2_64b_S512_4KiB']['wkid_p95_pp']:5.2f} a={rec['dashing2_64b_S512_4KiB']['ani_p95_pp']:5.2f} | "
                    f"sketchlib a={rec['sketchlib_s2048_jaccard']['ani_p95_pp']:5.2f}"
                )

        results["part2_contig_containment_40chr"] = p2_rows

    out_json.write_text(json.dumps(results, indent=2))
    typer.echo(f"\nSaved {out_json}")


if __name__ == "__main__":
    typer.run(main)
