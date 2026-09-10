#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = ["jsonschema", "typer"]
# ///
"""Run the FASTA pipeline against deterministic boundary and storage fixtures."""

import itertools
import random
import subprocess
import tempfile
from pathlib import Path
from typing import Annotated

import typer
from benchmark_schema import load_result


def main(
    binary: Annotated[Path, typer.Option(exists=True, dir_okay=False)] = Path(
        "build/benchmarks/cuddl-pipeline-benchmark"
    ),
    output: Annotated[
        Path | None, typer.Option(help="Keep fixture inputs and reports here")
    ] = None,
) -> None:
    """Verify both row formats, both indexes, both topologies, and all public stage outputs."""
    with tempfile.TemporaryDirectory(prefix="cuddl-pipeline-") as temporary:
        root = output or Path(temporary)
        root.mkdir(parents=True, exist_ok=True)
        rng = random.Random(20260906)
        sequence = "".join(rng.choices("ACGT", k=8192))
        fixtures = {
            "genome": f">genome\n{sequence}\n",
            "partial": f">partial\n{sequence[:4096]}NNNN{sequence[5000:]}\n",
            # Several records of one genome share each batch and saturate a common winner.
            "saturated": "".join(
                f">repeats{i}\n{'A' * 1024}\n" for i in range(70)
            ),
            "empty": ">short\nACGTNN\n>another\nACGT\n",
            "reverse": f">reverse\n{sequence.translate(str.maketrans('ACGT', 'TGCA'))[::-1]}\n",
        }
        paths = {}
        for name, contents in fixtures.items():
            paths[name] = root / f"{name}.fa"
            paths[name].write_text(contents)
        required_stages = {
            "end_to_end_wall",
            "a48_decode_serial",
            "a48_decode_parallel",
            "parse_and_canonicalize",
            "host_to_device",
            "clear",
            "construct_resident",
            "clear_and_construct",
            "clear_and_incremental_construct",
            "extract_compact_rows",
            "extract_packed_rows",
            "winner_counts",
            "cardinality",
            "pairwise_summary",
            "pairwise_summary_with_cardinality",
            "compare_corresponding_rows",
            "derive_pair_metrics",
            "database_build",
            "database_and_index_build",
            "search_single_indexed",
            "search_single_exhaustive",
            "search_batch_indexed",
            "search_batch_exhaustive",
            "search_all_to_all_indexed",
            "search_all_to_all_exhaustive",
            "search_and_download",
        }
        for rows, index, topology in itertools.product(
            ("compact", "packed"),
            ("dense", "sparse"),
            ("batch", "all-to-all"),
        ):
            report = root / f"{rows}-{index}-{topology}.json"
            command = [str(binary.resolve())]
            for name in ("genome", "partial", "saturated"):
                command.extend(("--reference", str(paths[name])))
            for name in ("reverse", "empty", "saturated"):
                command.extend(("--query", str(paths[name])))
            command.extend(
                (
                    "--rows",
                    rows,
                    "--index",
                    index,
                    "--topology",
                    topology,
                    "--indexed-buckets",
                    "4096" if topology == "all-to-all" else "2048",
                    "--key-bits",
                    "16" if rows == "packed" else "15",
                    "--minimum-matches",
                    "0" if topology == "all-to-all" else "5",
                    "--samples",
                    "2",
                    "--warmups",
                    "0",
                    "--output",
                    str(report),
                )
            )
            subprocess.run(command, check=True)
            data = load_result(report, "pipeline")
            pipeline = data["measurements"][0]
            assert required_stages <= pipeline["timings"].keys()
            assert pipeline["case"]["resident_input"] == "packed_u64_actg_max"
            for key in (
                "resident_total",
                "resident_reset",
                "resident_construct",
                "resident_statistics",
                "resident_rows",
                "resident_index",
                "resident_search",
            ):
                value = pipeline["timings"][key]
                assert value["source"] == "nvbench_gpu_events"
                assert 0 <= value["min_ms"] <= value["median_ms"] <= value["max_ms"]
            phases = [
                pipeline["timings"][key]
                for key in ("prepare_wall", "query_output_wall", "teardown_wall")
            ]
            total = pipeline["timings"]["end_to_end_wall"]
            assert sum(p["min_ms"] for p in phases) <= total["min_ms"] + 1e-9
            assert total["max_ms"] <= sum(p["max_ms"] for p in phases) + 1e-9
            assert pipeline["metrics"]["oracle_passed"]
            assert pipeline["case"]["references"] == 3
            assert pipeline["case"]["queries"] == 3
            for measurement in data["measurements"][1:]:
                case, values = measurement["case"], measurement["metrics"]
                if case["measurement"] == "genome":
                    assert (
                        sum(
                            value
                            for key, value in values.items()
                            if key.startswith("q_")
                        )
                        == 4096
                    )
                    if case["genome_id"] == 2:
                        assert values["saturated"] and values["q_65535"] > 0
                    if case["role"] == "query" and case["genome_id"] == 1:
                        assert values["cardinality"] == 0 and values["q_0"] == 4096
                elif case["measurement"] == "match" and topology == "batch":
                    assert case["query_id"] != 1, (
                        "empty query passed the positive retrieval threshold"
                    )
                    if case["query_id"] == 0 and case["reference_id"] == 0:
                        assert values["wkid"] == 1 and values["ani"] == 1
            typer.echo(f"PASS {rows} / {index} / {topology}")
        # Streamed ingestion replaces the host k-mer arrays with bounded per-genome GPU tiles.
        streamed = root / "streamed.json"
        command = [str(binary.resolve())]
        for name in ("genome", "partial", "saturated"):
            command.extend(("--reference", str(paths[name])))
        for name in ("reverse", "empty", "saturated"):
            command.extend(("--query", str(paths[name])))
        command.extend(
            (
                "--ingest",
                "sequence",
                "--resident-bytes",
                "4096",
                "--rows",
                "compact",
                "--index",
                "sparse",
                "--topology",
                "batch",
                "--indexed-buckets",
                "2048",
                "--key-bits",
                "15",
                "--minimum-matches",
                "5",
                "--samples",
                "2",
                "--warmups",
                "0",
                "--output",
                str(streamed),
            )
        )
        subprocess.run(command, check=True)
        data = load_result(streamed, "pipeline")
        pipeline = data["measurements"][0]
        assert pipeline["case"]["ingest"] == "sequence"
        assert pipeline["case"]["resident_input"] == "sequence_ascii"
        assert pipeline["case"]["resident_batch_bytes"] == 4096
        assert pipeline["case"]["resident_batches"] > 1
        # Actual largest-batch device allocation stays bounded by the cap; the cap
        # itself is reported in the case, the allocation in memory_bytes.
        assert 0 < pipeline["memory_bytes"]["resident_input"] <= 4096
        assert 0 <= pipeline["memory_bytes"]["resident_input_metadata"] <= 4096
        assert pipeline["metrics"]["oracle_passed"]
        assert (
            pipeline["metrics"]["oracle_pairs_checked"]
            == pipeline["metrics"]["oracle_pairs_total"]
        )
        for key in ("host_to_device", "construct_resident", "clear_and_construct"):
            assert key not in pipeline["timings"], (
                f"packed-input stage {key} ran in streamed mode"
            )
        for key in (
            "resident_total",
            "resident_reset",
            "resident_construct",
            "resident_statistics",
            "resident_rows",
            "resident_index",
            "resident_search",
        ):
            for suffix, source in (
                ("", "nvbench_gpu_events"),
                ("_wall", "nvbench_cpu_wall"),
            ):
                value = pipeline["timings"][key + suffix]
                assert value["source"] == source
                assert value["samples"] == 2
                assert 0 <= value["min_ms"] <= value["median_ms"] <= value["max_ms"]
        # Streamed rows must reproduce the packed-input sketches, pair metrics, and match rows.
        packed = load_result(root / "compact-sparse-batch.json", "pipeline")
        assert data["measurements"][1:] == packed["measurements"][1:], (
            "streamed ingestion differs from packed-input construction"
        )
        typer.echo("PASS streamed sequence ingestion")
        automatic = root / "streamed-auto.json"
        command[command.index("--resident-bytes") + 1] = "0"
        command[command.index("--output") + 1] = str(automatic)
        subprocess.run(command, check=True)
        auto_data = load_result(automatic, "pipeline")
        auto_pipeline = auto_data["measurements"][0]
        assert auto_pipeline["case"]["resident_batch_bytes"] >= 25
        assert auto_pipeline["case"]["resident_batches"] >= 1
        assert 0 < auto_pipeline["memory_bytes"]["resident_input"] <= auto_pipeline["case"]["resident_batch_bytes"]
        assert 0 <= auto_pipeline["memory_bytes"]["resident_input_metadata"] <= auto_pipeline["case"]["resident_batch_bytes"]
        assert auto_data["measurements"][1:] == packed["measurements"][1:], (
            "automatic GPU memory sizing changed sketches or pair results"
        )
        typer.echo("PASS automatic GPU memory sizing and same-genome batch merging")
        # Single-batch corpus (empty + saturated fixtures included) must keep the cap
        # in the case but report the smaller actual allocation: no cap-sized upload.
        single_path = root / "streamed-single.json"
        single_command = [str(binary.resolve())]
        for name in ("genome", "partial", "saturated"):
            single_command.extend(("--reference", str(paths[name])))
        for name in ("reverse", "empty", "saturated"):
            single_command.extend(("--query", str(paths[name])))
        single_command.extend(
            (
                "--ingest",
                "sequence",
                "--resident-bytes",
                "1048576",
                "--rows",
                "compact",
                "--index",
                "sparse",
                "--topology",
                "batch",
                "--indexed-buckets",
                "2048",
                "--key-bits",
                "15",
                "--minimum-matches",
                "5",
                "--samples",
                "2",
                "--warmups",
                "0",
                "--output",
                str(single_path),
            )
        )
        subprocess.run(single_command, check=True)
        single_data = load_result(single_path, "pipeline")
        single_pipeline = single_data["measurements"][0]
        assert single_pipeline["case"]["resident_batch_bytes"] == 1048576
        assert single_pipeline["case"]["resident_batches"] == 1
        assert 0 < single_pipeline["memory_bytes"]["resident_input"] < 1048576
        assert 0 <= single_pipeline["memory_bytes"]["resident_input_metadata"] <= 1048576
        assert single_pipeline["memory_bytes"]["input_kmers"] == 0
        assert single_pipeline["metrics"]["oracle_passed"]
        assert single_data["measurements"][1:] == packed["measurements"][1:], (
            "single-batch streamed ingestion differs from packed-input construction"
        )
        typer.echo("PASS single-batch resident allocation without cap-sized upload")
        typer.echo(
            "All 8 pipeline configurations and streamed ingestion passed. "
            "Timing samples are smoke checks, not performance evidence."
        )


if __name__ == "__main__":
    typer.run(main)
