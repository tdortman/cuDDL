#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = ["jsonschema", "typer"]
# ///
"""Check plot_pipeline_comparison via real CLI and run_pipeline_comparison budget selection."""

import csv
import json
import subprocess
import tempfile
from pathlib import Path
from typing import Annotated
from unittest import mock

import typer
from benchmark_schema import benchmark_system, make_result, write_result

PLOT = Path(__file__).with_name("plot_pipeline_comparison.py")
APP_KEYS = ("end_to_end_wall", "prepare_wall", "query_output_wall", "teardown_wall")


def _wall(
    min_ms: float,
    median_ms: float,
    max_ms: float,
    source: str = "steady_clock_cpu_wall",
) -> dict:
    return {
        "source": source,
        "samples": 2,
        "min_ms": min_ms,
        "median_ms": median_ms,
        "max_ms": max_ms,
    }


def _gpu(min_ms: float, median_ms: float, max_ms: float) -> dict:
    return {
        "source": "nvbench_gpu_events",
        "samples": 2,
        "min_ms": min_ms,
        "median_ms": median_ms,
        "max_ms": max_ms,
    }


def _base(system: dict, datasets: dict, case: dict, timings: dict, impl: dict) -> dict:
    return make_result(
        name="synthetic",
        operation="pipeline",
        scope="end_to_end",
        datasets=datasets,
        system=system,
        measurements=[
            {
                "implementation": impl,
                "case": {"measurement": "pipeline", **case},
                "timings": timings,
            }
        ],
    )


def _reports(
    root: Path,
    system: dict,
    datasets: dict,
    cuddl_ingest: str,
    rabbit_ingest: str,
    drop: str | None = None,
    seq_resident: bool = False,
    batch_bytes: int = 64 * 1024 * 1024,
    batches: int = 3,
    scope: str = "batched_resident_segments",
    rabbit_batches: int | None = None,
) -> tuple[Path, Path, dict, dict]:
    cuddl_case = {
        "k": 25,
        "references": 1,
        "queries": 1,
        "topology": "batch",
        "input_cache": "warm_os_cache",
        "ingest": cuddl_ingest,
        "resident_input": "sequence_ascii"
        if seq_resident and cuddl_ingest == "sequence"
        else "sequence_tiles"
        if cuddl_ingest == "sequence"
        else "packed_u64_actg_max",
        "resident_minimum_matches": 0,
        "rows": "packed",
        "index": "sparse",
        "buckets": 4096,
        "indexed_buckets": 2048,
        "key_bits": 15,
        "hash_seed": 42,
    }
    if seq_resident and cuddl_ingest == "sequence":
        cuddl_case.update(
            {
                "resident_batch_bytes": batch_bytes,
                "resident_batches": batches,
                "resident_timing_scope": scope,
            }
        )
    rabbit_case = {
        "k": 25,
        "references": 1,
        "queries": 1,
        "topology": "batch",
        "input_cache": "warm_os_cache",
        "ingest": rabbit_ingest,
        "resident_input": "sequence_ascii"
        if seq_resident and rabbit_ingest == "sequence"
        else "fastx_files"
        if rabbit_ingest == "sequence"
        else "packed_u64_actg_max",
        "resident_minimum_matches": 0,
        "sketch_size": 4096,
        "orchestration_threads": 7,
    }
    if seq_resident and rabbit_ingest == "sequence":
        rabbit_case.update(
            {
                "resident_batch_bytes": batch_bytes,
                "resident_batches": batches
                if rabbit_batches is None
                else rabbit_batches,
                "resident_timing_scope": scope,
            }
        )
    if cuddl_ingest == "sequence" and not seq_resident:
        cuddl_timings = {
            "end_to_end_wall": _wall(100, 110, 120),
            "prepare_wall": _wall(80, 85, 90),
            "query_output_wall": _wall(5, 6, 7),
            "teardown_wall": _wall(10, 12, 14),
        }
    else:
        cuddl_timings = {
            "end_to_end_wall": _wall(100, 110, 120),
            "resident_total_wall": _wall(50, 55, 60, "nvbench_cpu_wall"),
            "resident_reset_wall": _wall(1, 2, 3, "nvbench_cpu_wall"),
            "resident_construct_wall": _wall(4, 5, 6, "nvbench_cpu_wall"),
            "resident_statistics_wall": _wall(7, 8, 9, "nvbench_cpu_wall"),
            "resident_rows_wall": _wall(10, 11, 12, "nvbench_cpu_wall"),
            "resident_index_wall": _wall(13, 14, 15, "nvbench_cpu_wall"),
            "resident_search_wall": _wall(16, 17, 18, "nvbench_cpu_wall"),
            "resident_total": _gpu(49, 54, 59),
            "resident_reset": _gpu(0.9, 1.9, 2.9),
            "resident_construct": _gpu(3.9, 4.9, 5.9),
            "resident_statistics": _gpu(6.9, 7.9, 8.9),
            "resident_rows": _gpu(9.9, 10.9, 11.9),
            "resident_index": _gpu(12.9, 13.9, 14.9),
            "resident_search": _gpu(15.9, 16.9, 17.9),
        }
    if rabbit_ingest == "sequence" and not seq_resident:
        rabbit_timings = {
            "end_to_end_wall": _wall(200, 210, 220),
            "prepare_wall": _wall(150, 160, 170),
            "query_output_wall": _wall(8, 9, 10),
            "teardown_wall": _wall(20, 22, 24),
        }
    else:
        rabbit_timings = {
            "end_to_end_wall": _wall(200, 210, 220),
            "resident_total_wall": _wall(90, 95, 100, "nvbench_cpu_wall"),
            "resident_reset_wall": _wall(1, 1.5, 2, "nvbench_cpu_wall"),
            "resident_construct_wall": _wall(3, 4, 5, "nvbench_cpu_wall"),
            "resident_finalize_wall": _wall(6, 7, 8, "nvbench_cpu_wall"),
            "resident_cardinality_wall": _wall(9, 10, 11, "nvbench_cpu_wall"),
            "resident_search_wall": _wall(12, 13, 14, "nvbench_cpu_wall"),
        }
    if drop is not None:
        cuddl_timings.pop(drop, None)
    cuddl_path = root / f"cuddl-{cuddl_ingest}.json"
    rabbit_path = root / f"rabbit-{rabbit_ingest}.json"
    write_result(
        cuddl_path,
        _base(
            system,
            datasets,
            cuddl_case,
            cuddl_timings,
            {"name": "cuddl", "variant": "packed_sparse"},
        ),
    )
    write_result(
        rabbit_path,
        _base(
            system,
            datasets,
            rabbit_case,
            rabbit_timings,
            {"name": "rabbitsketch", "variant": "CPU"},
        ),
    )
    return cuddl_path, rabbit_path, cuddl_timings, rabbit_timings


def _run(plot: Path, reports: list[Path], out: Path) -> subprocess.CompletedProcess:
    return subprocess.run(
        [
            "uv",
            "run",
            "--script",
            str(plot.resolve()),
            *[str(p) for p in reports],
            "--output-dir",
            str(out),
        ],
        capture_output=True,
        text=True,
        check=False,
    )


def _csv_rows(out: Path) -> list[dict]:
    with (out / "timings.csv").open() as stream:
        return list(csv.DictReader(stream))


def _check_exact(rows: list[dict], report: Path, expected: dict) -> None:
    by_timer = {row["timer"]: row for row in rows if row["report"] == str(report)}
    for key in APP_KEYS:
        row = by_timer[key]
        want = expected[key]
        assert row["source"] == want["source"]
        assert int(row["samples"]) == want["samples"]
        assert float(row["min_ms"]) == want["min_ms"]
        assert float(row["median_ms"]) == want["median_ms"]
        assert float(row["max_ms"]) == want["max_ms"]
    assert "resident_total_wall" not in by_timer
    assert not any(timer == "resident_total" for timer in by_timer)


def _check_packed(rows: list[dict], report: Path, expected: dict) -> None:
    by_timer = {row["timer"]: row for row in rows if row["report"] == str(report)}
    for key, want in expected.items():
        row = by_timer[key]
        assert row["source"] == want["source"]
        assert int(row["samples"]) == want["samples"]
        assert float(row["min_ms"]) == want["min_ms"]
        assert float(row["max_ms"]) == want["max_ms"]


def _write_runner_report(path: Path, system: dict, name: str) -> None:
    write_result(
        path,
        make_result(
            name="synthetic",
            operation="pipeline",
            scope="end_to_end",
            datasets={"reference_0": {"path": "g.fa", "sha256": "a" * 64}},
            measurements=[
                {
                    "implementation": {"name": name},
                    "case": {"measurement": "pipeline"},
                    "timings": {
                        "end_to_end_wall": {
                            "source": "steady_clock_cpu_wall",
                            "samples": 2,
                            "min_ms": 1,
                            "median_ms": 2,
                            "max_ms": 3,
                        }
                    },
                }
            ],
        ),
    )


def _run_runner(
    runner,
    calls: list,
    system: dict,
    plans: dict,
    **kwargs,
) -> None:
    """Invoke the runner with subprocess mocked; probes answer from plans."""

    def fake_run(command, **run_kwargs):
        calls.append(list(command))
        if command[0] == "meson":
            return subprocess.CompletedProcess(command, 0)
        target = Path(command[command.index("--output") + 1])
        if "--resident-plan" in command:
            rows = command[command.index("--rows") + 1]
            index = command[command.index("--index") + 1]
            target.write_text(
                json.dumps(
                    {
                        "resident_batch_bytes": plans[f"{rows}_{index}"],
                        "free_bytes": 1,
                        "reserve_bytes": 2,
                    }
                ),
                encoding="utf-8",
            )
            return subprocess.CompletedProcess(command, 0)
        name = (
            "rabbitsketch"
            if str(command[0]).endswith("rabbitsketch-pipeline-benchmark")
            else "cuddl"
        )
        _write_runner_report(target, system, name)
        return subprocess.CompletedProcess(command, 0)

    with mock.patch.object(runner.subprocess, "run", side_effect=fake_run):
        runner.main(**kwargs)


def _check_runner_probe(root: Path) -> None:
    import run_pipeline_comparison as runner

    system = benchmark_system()
    inputs = root / "runner-inputs"
    inputs.mkdir(parents=True, exist_ok=True)
    genome = inputs / "genome.fa"
    genome.write_text(">r0\nACGTACGTACGT\n", encoding="utf-8")
    query = inputs / "query.fa"
    query.write_text(">q0\nACGTACGTACGT\n", encoding="utf-8")

    calls: list = []
    plans = {
        "compact_sparse": 100000,
        "compact_dense": 80000,
        "packed_sparse": 120000,
        "packed_dense": 90000,
    }
    _run_runner(
        runner,
        calls,
        system,
        plans,
        inputs=None,
        reference=[genome],
        query=[query],
        output_dir=root / "runner-auto",
        build_dir=root / "build",
        topology="batch",
        samples=2,
        warmups=0,
        implementations="cuddl,rabbitsketch",
        ingest="sequence",
        resident_bytes=0,
    )
    probes = [call for call in calls if "--resident-plan" in call]
    assert len(probes) == 4
    for probe in probes:
        assert probe[probe.index("--ingest") + 1] == "sequence"
        assert probe[probe.index("--resident-bytes") + 1] == "0"
        assert probe[probe.index("--topology") + 1] == "batch"
        assert "--config" in probe and "--output" in probe
    actuals = [call for call in calls if call[0] != "meson" and "--resident-plan" not in call]
    assert len(actuals) == 5
    for command in actuals:
        assert command[command.index("--resident-bytes") + 1] == "80000"

    calls.clear()
    _run_runner(
        runner,
        calls,
        system,
        plans,
        inputs=None,
        reference=[genome],
        query=[query],
        output_dir=root / "runner-explicit",
        build_dir=root / "build",
        topology="batch",
        samples=2,
        warmups=0,
        implementations="cuddl,rabbitsketch",
        ingest="sequence",
        resident_bytes=1048576,
    )
    assert not any("--resident-plan" in call for call in calls)
    actuals = [call for call in calls if call[0] != "meson"]
    assert len(actuals) == 5
    for command in actuals:
        assert command[command.index("--resident-bytes") + 1] == "1048576"

    calls.clear()
    _run_runner(
        runner,
        calls,
        system,
        plans,
        inputs=None,
        reference=[genome, query],
        query=None,
        output_dir=root / "runner-packed",
        build_dir=root / "build",
        topology="all-to-all",
        samples=2,
        warmups=0,
        implementations="cuddl,rabbitsketch",
        ingest="packed",
        resident_bytes=0,
    )
    assert not any("--resident-plan" in call for call in calls)
    actuals = [call for call in calls if call[0] != "meson"]
    for command in actuals:
        assert command[command.index("--resident-bytes") + 1] == "0"

    try:
        with mock.patch.object(
            runner.subprocess, "run", return_value=subprocess.CompletedProcess([], 0)
        ):
            runner.main(
                inputs=None,
                reference=[genome],
                query=[query],
                output_dir=root / "runner-cpu-only",
                build_dir=root / "build",
                topology="batch",
                samples=2,
                warmups=0,
                implementations="rabbitsketch",
                ingest="sequence",
                resident_bytes=0,
            )
    except typer.BadParameter:
        pass
    else:
        raise AssertionError("CPU-only sequence with a zero cap must fail")

    bad_plan = root / "bad-plan.json"
    bad_plan.write_text("{}", encoding="utf-8")
    try:
        runner.read_plan_cap(bad_plan)
    except typer.BadParameter:
        pass
    else:
        raise AssertionError("invalid plan caps must fail")

def main(
    plot_script: Annotated[Path, typer.Option(exists=True, dir_okay=False)] = PLOT,
    output: Annotated[
        Path | None, typer.Option(help="Keep fixtures and plots here")
    ] = None,
) -> None:
    """Exercise plot paths, runner budget selection, plus failure modes."""
    with tempfile.TemporaryDirectory(prefix="pipeline-comparison-") as temporary:
        root = output or Path(temporary)
        root.mkdir(parents=True, exist_ok=True)
        system = benchmark_system()
        datasets = {
            "reference_0": {"path": "ref0.fa", "sha256": "a" * 64},
            "query_0": {"path": "q0.fa", "sha256": "b" * 64},
        }

        seq_dir = root / "seq"
        cuddl, rabbit, cuddl_t, rabbit_t = _reports(
            seq_dir, system, datasets, "sequence", "sequence"
        )
        result = _run(plot_script, [cuddl, rabbit], seq_dir / "plots")
        assert result.returncode == 0
        out = seq_dir / "plots"
        for stem in ("end_to_end", "stages"):
            assert (out / f"{stem}.png").is_file()
            assert (out / f"{stem}.pdf").is_file()
        assert not (out / "resident_total.png").exists()
        assert not (out / "resident_total.pdf").exists()
        rows = _csv_rows(out)
        _check_exact(rows, cuddl, cuddl_t)
        _check_exact(rows, rabbit, rabbit_t)
        typer.echo("PASS sequence ingest without resident timings")

        packed_dir = root / "packed"
        cuddl_p, rabbit_p, cuddl_pt, rabbit_pt = _reports(
            packed_dir, system, datasets, "packed", "packed"
        )
        result = _run(plot_script, [cuddl_p, rabbit_p], packed_dir / "plots")
        assert result.returncode == 0
        out = packed_dir / "plots"
        for stem in ("resident_total", "end_to_end", "stages"):
            assert (out / f"{stem}.png").is_file()
            assert (out / f"{stem}.pdf").is_file()
        rows = _csv_rows(out)
        _check_packed(rows, cuddl_p, cuddl_pt)
        _check_packed(rows, rabbit_p, rabbit_pt)
        typer.echo("PASS packed ingest retains resident outputs")

        seq_resident_dir = root / "seq-resident"
        cuddl_s, rabbit_s, cuddl_st, rabbit_st = _reports(
            seq_resident_dir, system, datasets, "sequence", "sequence", seq_resident=True
        )
        result = _run(plot_script, [cuddl_s, rabbit_s], seq_resident_dir / "plots")
        assert result.returncode == 0, result.stderr
        out = seq_resident_dir / "plots"
        for stem in ("resident_total", "end_to_end", "stages"):
            assert (out / f"{stem}.png").is_file()
            assert (out / f"{stem}.pdf").is_file()
        rows = _csv_rows(out)
        _check_packed(rows, cuddl_s, cuddl_st)
        _check_packed(rows, rabbit_s, rabbit_st)
        assert any("summed segments" in row["stage"] for row in rows)
        typer.echo("PASS sequence resident uses summed segments")

        batch_mismatch_dir = root / "batch-mismatch"
        cuddl_b, rabbit_b, _, _ = _reports(
            batch_mismatch_dir,
            system,
            datasets,
            "sequence",
            "sequence",
            seq_resident=True,
            rabbit_batches=5,
        )
        result = _run(plot_script, [cuddl_b, rabbit_b], batch_mismatch_dir / "plots")
        assert result.returncode != 0
        typer.echo("PASS batch count mismatch rejected")

        mixed_dir = root / "mixed"
        cuddl_m, rabbit_m, _, _ = _reports(
            mixed_dir, system, datasets, "sequence", "packed"
        )
        result = _run(plot_script, [cuddl_m, rabbit_m], mixed_dir / "plots")
        assert result.returncode != 0
        typer.echo("PASS mixed ingest rejected")

        mixed_resident_dir = root / "mixed-resident"
        resident_src = root / "mixed-resident-src"
        legacy_src = root / "mixed-resident-legacy-src"
        cuddl_r, _, _, _ = _reports(
            resident_src, system, datasets, "sequence", "sequence", seq_resident=True
        )
        _, rabbit_l, _, _ = _reports(
            legacy_src, system, datasets, "sequence", "sequence"
        )
        mixed_resident_dir.mkdir(parents=True, exist_ok=True)
        (mixed_resident_dir / cuddl_r.name).write_bytes(cuddl_r.read_bytes())
        (mixed_resident_dir / rabbit_l.name).write_bytes(rabbit_l.read_bytes())
        result = _run(
            plot_script,
            [mixed_resident_dir / cuddl_r.name, mixed_resident_dir / rabbit_l.name],
            mixed_resident_dir / "plots",
        )
        assert result.returncode != 0
        typer.echo("PASS mixed resident and legacy sequence rejected")

        missing_dir = root / "missing"
        cuddl_x, rabbit_x, _, _ = _reports(
            missing_dir, system, datasets, "sequence", "sequence", drop="prepare_wall"
        )
        result = _run(plot_script, [cuddl_x, rabbit_x], missing_dir / "plots")
        assert result.returncode != 0
        typer.echo("PASS missing phase timing rejected")

        _check_runner_probe(root)
        typer.echo("PASS runner probes GPU cap and shares min")


if __name__ == "__main__":
    typer.run(main)
