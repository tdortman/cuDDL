#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "typer",
#   "pandas",
#   "jsonschema",
# ]
# ///

"""Acceptance runner (cuDDL).

Two commands:

- `gate`: the end-to-end go/no-go performance gate on raw FASTA, comparing GPU
  vs single-thread CPU median totals and construction times, plus BBTools metric
  context on the same pair.
- `synthetic`: the metric-equivalence acceptance at controlled ANI levels, generating
  mutation-truth genomes and applying the mean-absolute-error, standard-deviation,
  and invalid-rate gates against vendored BBTools.

The vendored BBTools wrapper is invoked through `bash` (its shebang is `/bin/bash`,
absent on NixOS). Java must be on PATH (provided by the dev shell).
"""

import hashlib
import json
import statistics
import subprocess
from collections.abc import Iterator
from pathlib import Path
from typing import Annotated

import pandas as pd
import typer
from benchmark_schema import flatten_measurements, load_result

app = typer.Typer(help="Run cuDDL acceptance gates", no_args_is_help=True)

SEED = 42
AA_LEVELS = [85, 87, 90, 95, 97, 99, 99.9, 100.0]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def splitmix64(state: int) -> int:
    state &= (1 << 64) - 1
    state ^= state >> 30
    state = (state * 0xBF58476D1CE4E5B9) & (1 << 64) - 1
    state ^= state >> 27
    state = (state * 0x94D049BB133111EB) & (1 << 64) - 1
    return (state ^ (state >> 31)) & (1 << 64) - 1


def derive_seed(
    root_seed: int, workload_tag: int, params: tuple[int, ...], replicate_index: int
) -> int:
    """seed chain: successive SplitMix64 over root seed, a fixed workload tag,
    every unsigned parameter in listed order, and the replicate index."""
    state = root_seed
    state = splitmix64(state ^ workload_tag)
    for param in params:
        state = splitmix64(state ^ (param & ((1 << 64) - 1)))
    state = splitmix64(state ^ (replicate_index & ((1 << 64) - 1)))
    return state


def rng(seed: int) -> Iterator[int]:
    state = seed
    while True:
        state = splitmix64(state)
        yield state


def generate_genome(seed: int, bases: int) -> tuple[str, Iterator[int]]:
    """Deterministic random A/C/T/G genome of `bases` characters; returns string and rng state."""
    g = rng(seed)
    alpha = "ACGT"
    chars: list[str] = [alpha[next(g) & 3] for _ in range(bases)]
    return "".join(chars), g


def mutate_genome(genome: str, ani: float, rng_state: Iterator[int]) -> str:
    """Apply independent substitutions to reach the target ANI; return the mutant."""
    out: list[str] = []
    alive = rng_state
    for base in genome:
        if next(alive) >= (1 - ani) * (1 << 64):
            out.append(base)
        else:
            others = [b for b in "ACGT" if b != base]
            out.append(others[next(alive) % 3])
    return "".join(out)


def run_ddl(
    cli: Path,
    ref: Path,
    query: Path,
    backend: str,
    runs: int,
    warmup: int,
    output: Path,
) -> Path:
    cmd = [
        str(cli),
        "--reference",
        str(ref),
        "--query",
        str(query),
        "--k",
        "25",
        "--buckets",
        "2048",
        "--backend",
        backend,
        "--runs",
        str(runs),
        "--warmup",
        str(warmup),
        "--output",
        str(output),
    ]
    res = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if res.returncode != 0:
        typer.echo(f"{backend} backend failed: {res.stderr[-500:]}", err=True)
        raise typer.Exit(code=1)
    return output


def ddl_measured_rows(result_path: Path) -> pd.DataFrame:
    result = load_result(result_path, operation="fasta_pairwise")
    df = pd.DataFrame(flatten_measurements(result))
    measured = df[df["phase"] == "measured"].copy()
    measured["total_ms"] = measured["total_median_ms"]
    measured["construction_ms"] = measured["construction_median_ms"]
    return measured


def bbtools_compare(
    bbtools_dir: Path, query: Path, ref: Path, xmx: str = "32g"
) -> dict | None:
    script = Path(bbtools_dir) / "comparesketch.sh"
    cmd = [
        "bash",
        str(script),
        f"in={query}",
        f"ref={ref}",
        "k=25",
        f"--xmx={xmx}",
        "mode=single",
        "autosize=f",
        "size=2048",
        "minkeycount=1",
        "minprob=0",
        "minentropy=0",
        "minhits=1",
        "minwkid=0",
        "records=1",
        "index=f",
        "color=f",
        "format=4",
        "out=stdout",
    ]
    res = subprocess.run(cmd, capture_output=True, text=True, check=True)
    if res.returncode != 0:
        return None
    # BBTools format=4 prints a pretty-printed JSON object across many lines; the reference
    # comparison dict lives under a key prefixed "ref_". Accumulate the outer object until braces
    # balance, then return the first reference-comparison dict.
    text = res.stdout
    start = text.find("{")
    if start == -1:
        return None
    depth = 0
    in_str = False
    end = start
    for i in range(start, len(text)):
        ch = text[i]
        if ch == '"' and not in_str:
            in_str = True
        elif ch == '"' and in_str:
            in_str = False
        elif not in_str:
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    end = i + 1
                    break
    try:
        parsed = json.loads(text[start:end])
    except json.JSONDecodeError:
        return None
    if isinstance(parsed, list):
        parsed = parsed[0] if parsed else {}
    if not isinstance(parsed, dict):
        return None
    for key, value in parsed.items():
        if str(key).startswith("ref") and isinstance(value, dict):
            return value
    return None


@app.command()
def gate(
    reference: Annotated[Path, typer.Option(help="Reference FASTA")],
    query: Annotated[Path, typer.Option(help="Query FASTA")],
    cli: Annotated[Path, typer.Option(help="cuddl-benchmark-fasta binary")],
    bbtools_dir: Annotated[Path, typer.Option(help="Vendored bbmap directory")],
    xmx: Annotated[str, typer.Option(help="BBTools Java heap (--xmx)")] = "32g",
    runs: Annotated[int, typer.Option(min=1, help="Measured runs")] = 5,
    warmup: Annotated[int, typer.Option(min=0, help="Warm-up runs")] = 1,
    workdir: Annotated[Path, typer.Option(help="Output dir")] = Path("results/gpu"),
):
    """End-to-end GPU-vs-CPU go/no-go gate plus BBTools metric context."""
    workdir.mkdir(parents=True, exist_ok=True)
    reference, query = reference.resolve(), query.resolve()

    typer.echo("=== Input checksums ===")
    for p in (reference, query):
        typer.echo(f"{p.name}: sha256 {sha256(p)}")

    typer.echo("\n=== BBTools metrics (operational context) ===")
    ref_d = bbtools_compare(bbtools_dir, query, reference, xmx=xmx)
    typer.echo(ref_d if ref_d else "no metric parsed (reporting only)")

    gpu_json = run_ddl(cli, reference, query, "gpu", runs, warmup, workdir / "gpu.json")
    cpu_json = run_ddl(cli, reference, query, "cpu", runs, warmup, workdir / "cpu.json")
    gpu = ddl_measured_rows(gpu_json)
    cpu = ddl_measured_rows(cpu_json)

    g_tot = float(gpu["total_ms"].median())
    c_tot = float(cpu["total_ms"].median())
    g_con = float(gpu["construction_ms"].median())
    c_con = float(cpu["construction_ms"].median())
    typer.echo(
        f"GPU median total {g_tot:.3f} ms, construction {g_con:.3f} ms ({len(gpu)} runs)"
    )
    typer.echo(
        f"CPU median total {c_tot:.3f} ms, construction {c_con:.3f} ms ({len(cpu)} runs)"
    )
    typer.echo(f"C/G speedup: {c_tot / g_tot:.2f}x")

    ok = True
    checks = []
    if g_tot < c_tot:
        checks.append(f"GO  : GPU total {g_tot:.3f} < CPU {c_tot:.3f}")
    else:
        ok = False
        checks.append(f"NO-GO: GPU total {g_tot:.3f} >= CPU {c_tot:.3f}")
    if g_con < c_con:
        checks.append(f"GO  : GPU construction {g_con:.3f} < CPU {c_con:.3f}")
    else:
        ok = False
        checks.append(f"NO-GO: GPU construction {g_con:.3f} >= CPU {c_con:.3f}")
    for _, row in gpu.iterrows():
        if row["status"] != "ok":
            ok = False
            checks.append(f"NO-GO: GPU run {row['run_index']} status={row['status']}")

    first = gpu.iloc[0]
    typer.echo(
        f"\nDDL WKID={first['wkid']} ANI={first['ani']} lower={first['lower']} "
        f"equal={first['equal']} higher={first['higher']}"
    )
    for c in checks:
        typer.echo(c)
    typer.echo(f"\nRESULT: {'GO' if ok else 'NO-GO'}")
    if not ok:
        raise typer.Exit(1)


@app.command()
def synthetic(
    cli: Annotated[Path, typer.Option(help="cuddl-benchmark-fasta binary")],
    bbtools_dir: Annotated[Path, typer.Option(help="Vendored bbmap directory")],
    xmx: Annotated[str, typer.Option(help="BBTools Java heap (--xmx)")] = "32g",
    replicates: Annotated[
        int,
        typer.Option(min=1, help="Replicates per ANI point (dev=5, acceptance=50)"),
    ] = 5,
    workdir: Annotated[Path, typer.Option(help="Output dir")] = Path(
        "results/gpu/synthetic"
    ),
    backend: Annotated[str, typer.Option(help="backend (gpu or cpu)")] = "gpu",
):
    """Metric-equivalence acceptance at controlled ANI points.

    Generates a deterministic 1 MiB reference per replicate, applies substitution to reach
    each ANI level, and compares the DDL and BBTools estimates against the generating truth.
    """
    workdir.mkdir(parents=True, exist_ok=True)
    if backend not in ("gpu", "cpu"):
        typer.echo("--backend must be gpu or cpu", err=True)
        raise typer.Exit(code=1)

    per_point: dict[float, dict] = {}
    for ani in AA_LEVELS:
        ddl_signed, bb_signed = [], []
        ddl_wkid_signed, bb_wkid_signed = [], []
        ddl_invalid = bb_invalid = 0
        truth_ani = ani / 100.0
        truth_wkid = truth_ani**25
        for rep in range(replicates):
            # DDL deterministic seed chain: root seed, workload tag, k, buckets, ANI, replicate.
            seed = derive_seed(SEED, 0x534B4554434830, (25, 2048, round(ani * 10)), rep)
            genome, rng_state = generate_genome(seed, 1 << 20)
            mutant = mutate_genome(genome, truth_ani, rng_state)
            ref_fa = workdir / f"a{ani}_r{rep}_ref.fa"
            que_fa = workdir / f"a{ani}_r{rep}_query.fa"
            ref_fa.write_text(f">ref_{ani}_{rep}\n{genome}\n")
            que_fa.write_text(f">query_{ani}_{rep}\n{mutant}\n")

            ddl_json = run_ddl(
                cli,
                ref_fa,
                que_fa,
                backend,
                runs=1,
                warmup=1,
                output=workdir / f"ddl_a{ani}_r{rep}.json",
            )
            row = ddl_measured_rows(ddl_json).iloc[0]
            ddl_ani = row["ani"]
            ddl_wkid = row["wkid"]
            if pd.isna(ddl_ani) or pd.isna(ddl_wkid):
                ddl_invalid += 1
            else:
                ddl_signed.append((float(ddl_ani) - truth_ani) * 100)
                ddl_wkid_signed.append((float(ddl_wkid) - truth_wkid) * 100)

            ref_d = bbtools_compare(bbtools_dir, que_fa, ref_fa, xmx=xmx)
            if ref_d is None:
                bb_invalid += 1
            else:
                bb_ani = ref_d.get("ANI")
                bb_wkid = ref_d.get("WKID")
                if bb_ani is None or bb_wkid is None:
                    bb_invalid += 1
                else:
                    bb_signed.append((float(bb_ani) - truth_ani) * 100)
                    bb_wkid_signed.append((float(bb_wkid) - truth_wkid) * 100)

        # The gate applies only where both systems produced a metric; a point with no
        # BBTools metrics cannot pass (fail closed).
        if not bb_signed:
            per_point[ani] = {
                "note": "NO BBTools metric; gate not evaluated",
                "pass": False,
            }
            typer.echo(f"ANI {ani:5.1f}%: FAIL (no BBTools metrics)")
            continue
        if not ddl_signed:
            per_point[ani] = {
                "note": "NO DDL metric; gate not evaluated",
                "pass": False,
            }
            typer.echo(f"ANI {ani:5.1f}%: FAIL (no DDL metrics)")
            continue

        ddl_ae = [abs(e) for e in ddl_signed]
        bb_ae = [abs(e) for e in bb_signed]
        ddl_mae = statistics.mean(ddl_ae)
        bb_mae = statistics.mean(bb_ae)
        # ANI mean-absolute-error excess at most 0.1pp vs BBTools.
        mae_ok = (ddl_mae - bb_mae) <= 0.1
        # WKID mean-absolute-error excess bound is the k=25 truth transform.
        wkid_mae = statistics.mean([abs(e) for e in ddl_wkid_signed])
        bb_wkid_mae = statistics.mean([abs(e) for e in bb_wkid_signed])
        wkid_excess = wkid_mae - bb_wkid_mae
        wkid_bound = 100 * (truth_wkid - max(0.0, truth_ani - 0.001) ** 25)
        wkid_ok = wkid_excess <= wkid_bound
        # Signed-error sample standard deviations within 1.2*BB + 0.05pp.
        ddl_sd = statistics.stdev(ddl_signed) if len(ddl_signed) > 1 else 0.0
        bb_sd = statistics.stdev(bb_signed) if len(bb_signed) > 1 else 0.0
        sd_ok = ddl_sd <= 1.2 * bb_sd + 0.05
        ddl_wkid_sd = (
            statistics.stdev(ddl_wkid_signed) if len(ddl_wkid_signed) > 1 else 0.0
        )
        bb_wkid_sd = (
            statistics.stdev(bb_wkid_signed) if len(bb_wkid_signed) > 1 else 0.0
        )
        wkid_sd_ok = ddl_wkid_sd <= 1.2 * bb_wkid_sd + 0.05
        # Invalid-rate excess at most 2 percentage points.
        ddl_invalid_rate = ddl_invalid / replicates
        bb_invalid_rate = bb_invalid / replicates
        invalid_ok = (ddl_invalid_rate - bb_invalid_rate) <= 0.02
        point_ok = mae_ok and wkid_ok and sd_ok and wkid_sd_ok and invalid_ok
        per_point[ani] = {
            "ddl_mae": ddl_mae,
            "bb_mae": bb_mae,
            "ani_mae_excess_pp": ddl_mae - bb_mae,
            "wkid_mae_excess_pp": wkid_excess,
            "wkid_bound_pp": wkid_bound,
            "ddl_signed_sd": ddl_sd,
            "bb_signed_sd": bb_sd,
            "ddl_wkid_signed_sd": ddl_wkid_sd,
            "bb_wkid_signed_sd": bb_wkid_sd,
            "ddl_invalid": ddl_invalid,
            "bb_invalid": bb_invalid,
            "mae_ok": mae_ok,
            "wkid_ok": wkid_ok,
            "sd_ok": sd_ok,
            "wkid_sd_ok": wkid_sd_ok,
            "invalid_ok": invalid_ok,
            "pass": point_ok,
            "n": len(ddl_signed),
        }
        typer.echo(
            f"ANI {ani:5.1f}%: mae ddl={ddl_mae:.3f} bb={bb_mae:.3f} excess={ddl_mae - bb_mae:+.3f} "
            f"{'OK' if mae_ok else 'FAIL'} | wkid excess={wkid_excess:+.3f}/{wkid_bound:.3f} "
            f"{'OK' if wkid_ok else 'FAIL'} | sd ddl={ddl_sd:.3f} bb={bb_sd:.3f} "
            f"{'OK' if sd_ok else 'FAIL'} | wkid_sd ddl={ddl_wkid_sd:.3f} bb={bb_wkid_sd:.3f} "
            f"{'OK' if wkid_sd_ok else 'FAIL'} | invalid ddl={ddl_invalid} bb={bb_invalid} "
            f"{'OK' if invalid_ok else 'FAIL'}"
        )

    (workdir / "synthetic_summary.json").write_text(json.dumps(per_point, indent=2))
    typer.echo(f"\nSummary written to {workdir / 'synthetic_summary.json'}")
    failed = any(not v.get("pass", False) for v in per_point.values())
    if failed:
        typer.echo(
            "RESULT: FAIL (some ANI points exceed the metric-equivalence margins)"
        )
        raise typer.Exit(code=1)
    typer.echo("RESULT: PASS (all ANI points within the metric-equivalence margins)")


if __name__ == "__main__":
    import json

    app()
