#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = ["typer"]
# ///
"""Convert cuDDL search-decision NVBench samples into one CSV row per workload."""

import csv
import json
import math
import shlex
import struct
from datetime import datetime, timezone
from pathlib import Path
from typing import Annotated, Any

import typer

BUILD_BENCHMARKS = ("compact_build", "indexed_build")
SEARCH_BENCHMARKS = ("exhaustive_search", "indexed_search")
FIELDS = (
    "schema_version",
    "timestamp_utc",
    "benchmark_command",
    "nvbench_version",
    "gpu_name",
    "compute_capability",
    "multiprocessor_count",
    "global_memory_bytes",
    "free_memory_before_bytes",
    "cuda_runtime_version",
    "cuda_driver_version",
    "cuda_compile_version",
    "compiler",
    "build_configuration",
    "kmer_length",
    "bucket_count",
    "indexed_bucket_count",
    "score_encoder_identity",
    "exponent_bits",
    "mantissa_bits",
    "hash_identity",
    "hash_seed",
    "canonicalisation_policy",
    "blacklist_identity",
    "blacklist_version",
    "key_mask",
    "fixture_seed",
    "reference_count",
    "fill_ratio",
    "filled_cells_per_row",
    "skew",
    "hot_fraction",
    "minimum_matches",
    "query_count",
    "warmup",
    "exhaustive_repetitions",
    "indexed_repetitions",
    "compact_build_fixture_generation_ms",
    "compact_build_host_to_device_ms",
    "indexed_build_fixture_generation_ms",
    "indexed_build_host_to_device_ms",
    "exhaustive_fixture_generation_ms",
    "exhaustive_host_to_device_ms",
    "exhaustive_validation_ms",
    "indexed_fixture_generation_ms",
    "indexed_host_to_device_ms",
    "indexed_validation_ms",
    "compact_build_p50_ms",
    "compact_build_p95_ms",
    "indexed_build_p50_ms",
    "indexed_build_p95_ms",
    "compact_resident_bytes",
    "indexed_resident_bytes",
    "compact_build_temporary_bytes",
    "indexed_build_temporary_bytes",
    "compact_search_workspace_bytes",
    "indexed_search_workspace_bytes",
    "peak_temporary_bytes",
    "populated_posting_lists",
    "posting_entries",
    "posting_visits",
    "atomic_updates",
    "selected_candidates",
    "exhaustive_exact_comparisons",
    "indexed_exact_comparisons",
    "exact_result_recall",
    "exhaustive_p50_ms",
    "exhaustive_p95_ms",
    "exhaustive_queries_per_second",
    "indexed_p50_ms",
    "indexed_p95_ms",
    "indexed_queries_per_second",
    "kill_gate_rule",
    "kill_gate_outcome",
    "status",
    "error",
)


def axis_values(state: dict[str, Any]) -> dict[str, str]:
    return {value["name"]: value["value"] for value in state["axis_values"]}


def summaries(state: dict[str, Any]) -> dict[str, dict[str, str]]:
    return {
        summary["tag"]: {value["name"]: value["value"] for value in summary["data"]}
        for summary in state.get("summaries", [])
    }


def summary_value(state: dict[str, Any], tag: str, default: str = "") -> str:
    return summaries(state).get(tag, {}).get("value", default)


def percentile(samples: list[float], quantile: float) -> float:
    values = sorted(samples)
    position = quantile * (len(values) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    fraction = position - lower
    return values[lower] * (1.0 - fraction) + values[upper] * fraction


def sample_times(path: Path, state: dict[str, Any]) -> list[float]:
    metadata = summaries(state)["nv/json/bin:nv/cold/sample_times"]
    count = int(metadata["size"])
    payload = (path.parent / metadata["filename"]).read_bytes()
    if len(payload) != count * 4:
        raise ValueError(
            f"sample sidecar has {len(payload)} bytes, expected {count * 4}"
        )
    return list(struct.unpack(f"<{count}f", payload))


def state_key(state: dict[str, Any], include_queries: bool) -> tuple[str, ...]:
    axes = axis_values(state)
    key = (axes["References"], axes["FillPermille"], axes["HotPercent"])
    return (*key, axes["Queries"]) if include_queries else key


def benchmark_states(
    document: dict[str, Any], name: str
) -> dict[tuple[str, ...], dict[str, Any]]:
    benchmark = next(item for item in document["benchmarks"] if item["name"] == name)
    include_queries = name in SEARCH_BENCHMARKS
    return {state_key(state, include_queries): state for state in benchmark["states"]}


def timing_ms(path: Path, state: dict[str, Any]) -> tuple[float, float, int]:
    samples = sample_times(path, state)
    return (
        percentile(samples, 0.50) * 1000.0,
        percentile(samples, 0.95) * 1000.0,
        len(samples),
    )


def as_number(state: dict[str, Any], tag: str) -> float:
    value = summary_value(state, tag)
    return float(value) if value else math.nan


def phase_metrics(prefix: str, state: dict[str, Any]) -> dict[str, float]:
    return {
        f"{prefix}_fixture_generation_ms": as_number(state, "fixture_generation_ms"),
        f"{prefix}_host_to_device_ms": as_number(state, "host_to_device_ms"),
    }


def main(
    input_path: Annotated[Path, typer.Argument(exists=True, dir_okay=False)],
    output: Annotated[Path, typer.Option(dir_okay=False)] = Path(
        "results/cuddl-search-decision.csv"
    ),
) -> None:
    document = json.loads(input_path.read_text())
    states = {
        name: benchmark_states(document, name)
        for name in (*BUILD_BENCHMARKS, *SEARCH_BENCHMARKS)
    }
    devices = {device["id"]: device for device in document["devices"]}
    search_keys = sorted(
        set(states["exhaustive_search"]) | set(states["indexed_search"]),
        key=lambda key: tuple(int(value) for value in key),
    )
    rows: list[dict[str, Any]] = []

    for key in search_keys:
        build_key = key[:3]
        selected = {
            "compact_build": states["compact_build"].get(build_key),
            "indexed_build": states["indexed_build"].get(build_key),
            "exhaustive_search": states["exhaustive_search"].get(key),
            "indexed_search": states["indexed_search"].get(key),
        }
        missing = [name for name, state in selected.items() if state is None]
        skipped = [
            f"{name}: {state.get('skip_reason', 'skipped')}"
            for name, state in selected.items()
            if state is not None and state.get("is_skipped")
        ]
        status = "ok" if not missing and not skipped else "skipped"
        error = "; ".join([*(f"missing {name}" for name in missing), *skipped])
        representative = selected["indexed_search"] or selected["exhaustive_search"]
        if representative is None:
            raise ValueError(f"workload {key} has no search state")
        device = devices[representative["device"]]
        axes = axis_values(representative)
        sm_version = int(device["sm_version"])
        row: dict[str, Any] = {
            "schema_version": 1,
            "timestamp_utc": datetime.fromtimestamp(
                input_path.stat().st_mtime, timezone.utc
            ).isoformat(),
            "benchmark_command": shlex.join(document["meta"]["argv"]),
            "nvbench_version": document["meta"]["version"]["nvbench"]["string"],
            "gpu_name": device["name"],
            "compute_capability": f"{sm_version // 100}.{sm_version % 100 // 10}",
            "multiprocessor_count": device["number_of_sms"],
            "global_memory_bytes": device["global_memory_size"],
            "reference_count": axes["References"],
            "fill_ratio": int(axes["FillPermille"]) / 1000.0,
            "skew": f"{int(axes['HotPercent'])}%",
            "query_count": axes["Queries"],
            "warmup": representative["cold_warmup_runs"],
            "kill_gate_rule": "indexed wins only when both p50 and p95 are strictly lower",
            "kill_gate_outcome": "inconclusive",
            "status": status,
            "error": error,
        }
        for tag in (
            "free_memory_before_bytes",
            "cuda_runtime_version",
            "cuda_driver_version",
            "cuda_compile_version",
            "compiler",
            "build_configuration",
            "kmer_length",
            "bucket_count",
            "indexed_bucket_count",
            "score_encoder_identity",
            "exponent_bits",
            "mantissa_bits",
            "hash_identity",
            "hash_seed",
            "canonicalisation_policy",
            "blacklist_identity",
            "blacklist_version",
            "key_mask",
            "fixture_seed",
            "filled_cells_per_row",
            "hot_fraction",
            "minimum_matches",
        ):
            row[tag] = summary_value(representative, tag)

        if status == "ok":
            compact_build = selected["compact_build"]
            indexed_build = selected["indexed_build"]
            exhaustive = selected["exhaustive_search"]
            indexed = selected["indexed_search"]
            assert compact_build and indexed_build and exhaustive and indexed
            compact_build_p50, compact_build_p95, _ = timing_ms(
                input_path, compact_build
            )
            indexed_build_p50, indexed_build_p95, _ = timing_ms(
                input_path, indexed_build
            )
            exhaustive_p50, exhaustive_p95, exhaustive_count = timing_ms(
                input_path, exhaustive
            )
            indexed_p50, indexed_p95, indexed_count = timing_ms(input_path, indexed)
            query_count = int(axes["Queries"])
            row.update(phase_metrics("compact_build", compact_build))
            row.update(phase_metrics("indexed_build", indexed_build))
            row.update(phase_metrics("exhaustive", exhaustive))
            row["exhaustive_validation_ms"] = as_number(exhaustive, "validation_ms")
            row.update(phase_metrics("indexed", indexed))
            row["indexed_validation_ms"] = as_number(indexed, "validation_ms")
            row.update(
                {
                    "compact_build_p50_ms": compact_build_p50,
                    "compact_build_p95_ms": compact_build_p95,
                    "indexed_build_p50_ms": indexed_build_p50,
                    "indexed_build_p95_ms": indexed_build_p95,
                    "compact_resident_bytes": as_number(
                        compact_build, "resident_bytes"
                    ),
                    "indexed_resident_bytes": as_number(
                        indexed_build, "resident_bytes"
                    ),
                    "compact_build_temporary_bytes": as_number(
                        compact_build, "build_temporary_bytes"
                    ),
                    "indexed_build_temporary_bytes": as_number(
                        indexed_build, "build_temporary_bytes"
                    ),
                    "compact_search_workspace_bytes": as_number(
                        exhaustive, "search_workspace_bytes"
                    ),
                    "indexed_search_workspace_bytes": as_number(
                        indexed, "search_workspace_bytes"
                    ),
                    "populated_posting_lists": as_number(
                        indexed, "populated_posting_lists"
                    ),
                    "posting_entries": as_number(indexed, "posting_entries"),
                    "posting_visits": as_number(indexed, "posting_visits"),
                    "atomic_updates": as_number(indexed, "atomic_updates"),
                    "selected_candidates": as_number(indexed, "selected_candidates"),
                    "exhaustive_exact_comparisons": as_number(
                        exhaustive, "exact_comparisons"
                    ),
                    "indexed_exact_comparisons": as_number(
                        indexed, "exact_comparisons"
                    ),
                    "exact_result_recall": as_number(indexed, "exact_result_recall"),
                    "exhaustive_p50_ms": exhaustive_p50,
                    "exhaustive_p95_ms": exhaustive_p95,
                    "exhaustive_queries_per_second": 1000.0
                    * query_count
                    / exhaustive_p50,
                    "indexed_p50_ms": indexed_p50,
                    "indexed_p95_ms": indexed_p95,
                    "indexed_queries_per_second": 1000.0 * query_count / indexed_p50,
                    "exhaustive_repetitions": exhaustive_count,
                    "indexed_repetitions": indexed_count,
                }
            )
            row["peak_temporary_bytes"] = max(
                row["compact_build_temporary_bytes"],
                row["indexed_build_temporary_bytes"],
                row["compact_search_workspace_bytes"],
                row["indexed_search_workspace_bytes"],
            )
            if indexed_p50 < exhaustive_p50 and indexed_p95 < exhaustive_p95:
                row["kill_gate_outcome"] = "indexed_win"
            elif exhaustive_p50 < indexed_p50 and exhaustive_p95 < indexed_p95:
                row["kill_gate_outcome"] = "exhaustive_win"
        rows.append(row)

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    typer.run(main)
