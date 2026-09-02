#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = ["jsonschema", "typer"]
# ///
"""Convert cuDDL search-decision NVBench samples into shared result JSON."""

import json
import math
import shlex
import struct
from datetime import datetime, timezone
from pathlib import Path
from typing import Annotated, Any

import typer
from benchmark_schema import make_result, measurements_from_rows, write_result

BUILD_BENCHMARKS = ("compact_build", "indexed_build")
SEARCH_BENCHMARKS = ("exhaustive_search", "indexed_search")
PAPER_REFERENCE_COUNT = 200687

FIELDS = (
    "schema_version",
    "timestamp_utc",
    "benchmark_command",
    "nvbench_version",
    "free_memory_before_bytes",
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
    "index_mode",
    "key_bits",
    "fixture_seed",
    "reference_count",
    "fill_ratio",
    "filled_cells_per_row",
    "skew",
    "hot_fraction",
    "minimum_matches",
    "query_count",
    "query_profile",
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
    "offset_bytes",
    "posting_bytes",
    "top_bit_frequency",
    "populated_posting_lists",
    "posting_entries",
    "posting_visits",
    "atomic_updates",
    "selected_candidates",
    "candidate_inflation",
    "exhaustive_exact_comparisons",
    "indexed_exact_comparisons",
    "exact_result_recall",
    "threshold_zero_recall",
    "oracle_threshold_pairs",
    "recalled_pairs",
    "exhaustive_p50_ms",
    "exhaustive_p95_ms",
    "exhaustive_queries_per_second",
    "indexed_p50_ms",
    "indexed_p95_ms",
    "indexed_queries_per_second",
    "kill_gate_rule",
    "kill_gate_outcome",
    "paper_scale_suitability",
    "status",
    "error",
)

CASE_FIELDS = (
    "schema_version",
    "timestamp_utc",
    "benchmark_command",
    "nvbench_version",
    "free_memory_before_bytes",
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
    "index_mode",
    "key_bits",
    "fixture_seed",
    "reference_count",
    "fill_ratio",
    "filled_cells_per_row",
    "skew",
    "hot_fraction",
    "minimum_matches",
    "query_count",
    "query_profile",
    "warmup",
    "kill_gate_rule",
    "kill_gate_outcome",
    "paper_scale_suitability",
    "status",
    "error",
)

INTEGER_FIELDS = {
    "schema_version",
    "multiprocessor_count",
    "global_memory_bytes",
    "free_memory_before_bytes",
    "cuda_runtime_version",
    "cuda_driver_version",
    "cuda_compile_version",
    "system_logical_cpu_count",
    "system_ram_bytes",
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
    "key_bits",
    "fixture_seed",
    "reference_count",
    "filled_cells_per_row",
    "minimum_matches",
    "query_count",
    "warmup",
    "exhaustive_repetitions",
    "indexed_repetitions",
    "compact_resident_bytes",
    "indexed_resident_bytes",
    "compact_build_temporary_bytes",
    "indexed_build_temporary_bytes",
    "compact_search_workspace_bytes",
    "indexed_search_workspace_bytes",
    "peak_temporary_bytes",
    "offset_bytes",
    "posting_bytes",
    "populated_posting_lists",
    "posting_entries",
    "posting_visits",
    "atomic_updates",
    "selected_candidates",
    "exhaustive_exact_comparisons",
    "indexed_exact_comparisons",
    "oracle_threshold_pairs",
    "recalled_pairs",
}

FLOAT_FIELDS = {
    "fill_ratio",
    "hot_fraction",
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
    "top_bit_frequency",
    "candidate_inflation",
    "exact_result_recall",
    "threshold_zero_recall",
    "exhaustive_p50_ms",
    "exhaustive_p95_ms",
    "exhaustive_queries_per_second",
    "indexed_p50_ms",
    "indexed_p95_ms",
    "indexed_queries_per_second",
}


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


def mode_key(state: dict[str, Any]) -> tuple[str, str]:
    axes = axis_values(state)
    try:
        buckets = axes.get("IndexedBuckets") or summary_value(
            state, "indexed_bucket_count"
        )
        bits = axes["KeyBits"]
    except KeyError as error:
        missing = ", ".join(
            axis for axis in ("IndexedBuckets", "KeyBits") if axis not in axes
        )
        raise ValueError(
            f"indexed benchmark state is missing required mode axis/axes: {missing}"
        ) from error
    return str(int(float(buckets))), str(int(bits))


def state_key(state: dict[str, Any], include_mode: bool = False) -> tuple[str, ...]:
    axes = axis_values(state)
    key = (
        axes["References"],
        axes.get("FillPermille", "1000"),
        axes["HotPercent"],
    )
    return (*key, *mode_key(state)) if include_mode else key


def benchmark_states(
    document: dict[str, Any], name: str
) -> dict[tuple[str, ...], dict[str, Any]]:
    try:
        benchmark = next(
            item for item in document["benchmarks"] if item["name"] == name
        )
    except StopIteration as error:
        raise ValueError(f"benchmark {name!r} is missing from input") from error
    include_mode = name in ("indexed_build", "indexed_search")
    states = {}
    for state in benchmark["states"]:
        key = state_key(state, include_mode)
        if key in states:
            raise ValueError(f"benchmark {name!r} contains duplicate state {key}")
        states[key] = state
    return states


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


def typed_value(name: str, value: Any) -> Any:
    if value is None:
        return None
    if value == "":
        return ""
    if name in INTEGER_FIELDS:
        number = float(value)
        if not math.isfinite(number):
            return None
        integer = int(number)
        if number != integer:
            raise ValueError(
                f"normalized integer field {name!r} is not integral: {value!r}"
            )
        return integer
    if name in FLOAT_FIELDS:
        number = float(value)
        return number if math.isfinite(number) else None
    return str(value)


def typed_row(row: dict[str, Any]) -> dict[str, Any]:
    return {name: typed_value(name, value) for name, value in row.items()}


def main(
    input_path: Annotated[Path, typer.Argument(exists=True, dir_okay=False)],
    output: Annotated[Path, typer.Option(dir_okay=False)] = Path(
        "results/cuddl-search-decision.json"
    ),
) -> None:
    document = json.loads(input_path.read_text())
    states = {
        name: benchmark_states(document, name)
        for name in (*BUILD_BENCHMARKS, *SEARCH_BENCHMARKS)
    }
    devices = {device["id"]: device for device in document["devices"]}
    workload_keys = sorted(
        set(states["exhaustive_search"])
        | {key[:-2] for key in states["indexed_search"]},
        key=lambda key: tuple(int(value) for value in key),
    )
    mode_keys = sorted(
        {key[-2:] for key in states["indexed_build"]}
        | {key[-2:] for key in states["indexed_search"]},
        key=lambda key: tuple(int(value) for value in key),
    )
    if not mode_keys:
        raise ValueError(
            "indexed benchmark states are required; no indexed mode states found"
        )
    rows: list[dict[str, Any]] = []

    for workload_key in workload_keys:
        build_key = workload_key[:3]
        for mode in mode_keys:
            indexed_build_key = (*build_key, *mode)
            indexed_search_key = (*workload_key, *mode)
            selected = {
                "compact_build": states["compact_build"].get(build_key),
                "indexed_build": states["indexed_build"].get(indexed_build_key),
                "exhaustive_search": states["exhaustive_search"].get(workload_key),
                "indexed_search": states["indexed_search"].get(indexed_search_key),
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
                representative = selected["indexed_build"] or selected["compact_build"]
            if representative is None:
                raise ValueError(f"workload {workload_key} has no benchmark state")
            device = devices[representative["device"]]
            axes = axis_values(representative)
            sm_version = int(device["sm_version"])
            reference_count = int(workload_key[0])
            is_paper_scale = reference_count == PAPER_REFERENCE_COUNT
            row: dict[str, Any] = {
                "schema_version": 3,
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
                "fill_ratio": int(axes.get("FillPermille", "1000")) / 1000.0,
                "skew": f"{int(axes['HotPercent'])}%",
                "query_count": reference_count,
                "query_profile": "all_to_all",
                "indexed_bucket_count": mode[0],
                "key_bits": mode[1],
                "key_mask": str((1 << int(mode[1])) - 1),
                "index_mode": f"b{mode[0]}_k{mode[1]}",
                "warmup": representative["cold_warmup_runs"],
                "kill_gate_rule": (
                    "indexed wins only when threshold-zero recall is 1.0 and "
                    "both p50 and p95 are strictly lower"
                ),
                "kill_gate_outcome": "inconclusive",
                "paper_scale_suitability": (
                    "not_paper_scale" if not is_paper_scale else "unavailable"
                ),
                "status": status,
                "error": error,
            }
            row["implementation"] = "cuddl"
            for tag in (
                "system_os",
                "system_kernel",
                "system_architecture",
                "system_cpu",
                "system_logical_cpu_count",
                "system_ram_bytes",
                "free_memory_before_bytes",
                "cuda_runtime_version",
                "cuda_driver_version",
                "cuda_compile_version",
                "compiler",
                "build_configuration",
                "kmer_length",
                "bucket_count",
                "score_encoder_identity",
                "exponent_bits",
                "mantissa_bits",
                "hash_identity",
                "hash_seed",
                "canonicalisation_policy",
                "blacklist_identity",
                "blacklist_version",
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
                query_count = reference_count
                recall = as_number(indexed, "threshold_zero_recall")
                if math.isnan(recall):
                    raise ValueError(
                        "indexed search state is missing required "
                        f"threshold_zero_recall: {indexed_search_key}"
                    )
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
                        "offset_bytes": as_number(indexed, "offset_bytes"),
                        "posting_bytes": as_number(indexed, "posting_bytes"),
                        "top_bit_frequency": as_number(indexed, "top_bit_frequency"),
                        "populated_posting_lists": as_number(
                            indexed, "populated_posting_lists"
                        ),
                        "posting_entries": as_number(indexed, "posting_entries"),
                        "posting_visits": as_number(indexed, "posting_visits"),
                        "atomic_updates": as_number(indexed, "atomic_updates"),
                        "selected_candidates": as_number(
                            indexed, "selected_candidates"
                        ),
                        "candidate_inflation": as_number(
                            indexed, "candidate_inflation"
                        ),
                        "exhaustive_exact_comparisons": as_number(
                            exhaustive, "exact_comparisons"
                        ),
                        "indexed_exact_comparisons": as_number(
                            indexed, "exact_comparisons"
                        ),
                        "exact_result_recall": as_number(
                            indexed, "exact_result_recall"
                        ),
                        "threshold_zero_recall": recall,
                        "oracle_threshold_pairs": as_number(
                            indexed, "oracle_threshold_pairs"
                        ),
                        "recalled_pairs": as_number(indexed, "recalled_pairs"),
                        "exhaustive_p50_ms": exhaustive_p50,
                        "exhaustive_p95_ms": exhaustive_p95,
                        "exhaustive_queries_per_second": 1000.0
                        * query_count
                        / exhaustive_p50,
                        "indexed_p50_ms": indexed_p50,
                        "indexed_p95_ms": indexed_p95,
                        "indexed_queries_per_second": 1000.0
                        * query_count
                        / indexed_p50,
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
                recall_loss = not math.isfinite(recall) or recall < 1.0
                indexed_win = (
                    indexed_p50 < exhaustive_p50 and indexed_p95 < exhaustive_p95
                )
                exhaustive_win = (
                    exhaustive_p50 < indexed_p50 and exhaustive_p95 < indexed_p95
                )
                if recall_loss:
                    row["kill_gate_outcome"] = "recall_loss"
                elif indexed_win:
                    row["kill_gate_outcome"] = "indexed_win"
                elif exhaustive_win:
                    row["kill_gate_outcome"] = "exhaustive_win"
                if not is_paper_scale:
                    outcome = "not_paper_scale"
                elif recall_loss:
                    outcome = "recall_loss"
                elif indexed_win:
                    outcome = "suitable"
                else:
                    outcome = "no_speed_win" if exhaustive_win else "inconclusive"
                row["paper_scale_suitability"] = outcome
            rows.append(row)

    typed_rows = [typed_row(row) for row in rows]
    system_fields = {
        "os": "system_os",
        "kernel": "system_kernel",
        "architecture": "system_architecture",
        "cpu": "system_cpu",
        "logical_cpu_count": "system_logical_cpu_count",
        "ram_bytes": "system_ram_bytes",
        "gpu": "gpu_name",
        "compute_capability": "compute_capability",
        "sm_count": "multiprocessor_count",
        "gpu_ram_bytes": "global_memory_bytes",
        "cuda_runtime_version": "cuda_runtime_version",
        "cuda_driver_version": "cuda_driver_version",
        "cuda_compile_version": "cuda_compile_version",
    }
    systems = {
        tuple(row[field] for field in system_fields.values()) for row in typed_rows
    }
    if len(systems) != 1:
        raise ValueError("benchmark states do not describe exactly one GPU system")
    system_values = next(iter(systems))
    missing_system_fields = [
        output
        for output, value in zip(system_fields, system_values, strict=True)
        if value in (None, "")
    ]
    if missing_system_fields:
        raise ValueError(
            "benchmark states are missing system metadata: "
            + ", ".join(missing_system_fields)
        )
    system = {
        output: value
        for output, value in zip(system_fields, system_values, strict=True)
    }
    measurements = measurements_from_rows(
        typed_rows,
        case_fields=CASE_FIELDS,
        omit_fields=system_fields.values(),
    )
    for row, measurement in zip(typed_rows, measurements, strict=True):
        for name in FIELDS:
            if row.get(name) == "":
                section = (
                    measurement["case"]
                    if name in CASE_FIELDS
                    else measurement["metrics"]
                )
                section[name] = ""
    result = make_result(
        name="cuDDL search decision",
        operation="search_decision",
        scope="kernel",
        system=system,
        measurements=measurements,
    )
    write_result(output, result)


if __name__ == "__main__":
    typer.run(main)
