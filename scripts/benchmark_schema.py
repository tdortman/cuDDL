from __future__ import annotations

import json
import math
import os
import platform
import tempfile
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

import jsonschema

SCHEMA_NAME = "cuddl-benchmark/v1"
SCHEMA_PATH = Path(__file__).parents[1] / "schemas/cuddl-benchmark-v1.schema.json"
VALIDATOR = jsonschema.Draft202012Validator(json.loads(SCHEMA_PATH.read_text()))


def validate_result(result: Mapping[str, Any], operation: str | None = None) -> None:
    errors = sorted(
        VALIDATOR.iter_errors(result),
        key=lambda error: tuple(map(str, error.absolute_path)),
    )
    if errors:
        error = errors[0]
        location = ".".join(map(str, error.absolute_path)) or "$"
        raise ValueError(f"{location}: {error.message}")
    if operation is not None and result["operation"] != operation:
        raise ValueError(
            f"expected operation {operation!r}, got {result['operation']!r}"
        )


def load_result(path: Path, operation: str | None = None) -> dict[str, Any]:
    try:
        result = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read benchmark result {path}: {error}") from error
    validate_result(result, operation)
    return result


def write_result(path: Path, result: Mapping[str, Any]) -> None:
    validate_result(result)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", dir=path.parent, prefix=path.name, delete=False
    ) as stream:
        temporary = Path(stream.name)
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    temporary.replace(path)


def make_result(
    *,
    name: str,
    operation: str,
    scope: str,
    measurements: Sequence[Mapping[str, Any]],
    datasets: Mapping[str, Mapping[str, str]] | None = None,
    system: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    system_info = benchmark_system()
    system_info.update(system or {})
    result: dict[str, Any] = {
        "schema": SCHEMA_NAME,
        "name": name,
        "operation": operation,
        "scope": scope,
        "datasets": dict(datasets or {}),
        "system": system_info,
        "measurements": list(measurements),
    }
    return result


def benchmark_system() -> dict[str, Any]:
    host = platform.uname()
    cpu = platform.processor()
    if not cpu:
        try:
            cpu = next(
                line.partition(":")[2].strip()
                for line in Path("/proc/cpuinfo").read_text().splitlines()
                if line.partition(":")[0].strip() == "model name"
            )
        except (OSError, StopIteration) as error:
            raise RuntimeError("cannot determine CPU model") from error
    logical_cpu_count = os.cpu_count()
    page_size = os.sysconf("SC_PAGE_SIZE")
    physical_pages = os.sysconf("SC_PHYS_PAGES")
    if logical_cpu_count is None or logical_cpu_count < 1:
        raise RuntimeError("cannot determine logical CPU count")
    if page_size < 1 or physical_pages < 1:
        raise RuntimeError("cannot determine physical memory")
    return {
        "os": host.system,
        "kernel": host.release,
        "architecture": host.machine,
        "cpu": cpu,
        "logical_cpu_count": logical_cpu_count,
        "ram_bytes": page_size * physical_pages,
    }


def _scalar(value: Any) -> str | int | float | bool | None:
    if value is None or value == "":
        return None
    if hasattr(value, "item"):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError(
        f"benchmark values must be JSON scalars, got {type(value).__name__}"
    )


def measurements_from_rows(
    rows: Iterable[Mapping[str, Any]],
    *,
    case_fields: Sequence[str],
    implementation_field: str = "implementation",
    omit_fields: Iterable[str] = (),
) -> list[dict[str, Any]]:
    cases = set(case_fields)
    omitted = set(omit_fields) | {implementation_field}
    measurements: list[dict[str, Any]] = []
    for row in rows:
        implementation = str(row[implementation_field])
        case = {
            key: scalar
            for key in case_fields
            if (scalar := _scalar(row.get(key))) is not None
        }
        metrics = {
            key: scalar
            for key, value in row.items()
            if key not in cases | omitted and (scalar := _scalar(value)) is not None
        }
        measurements.append(
            {
                "implementation": {"name": implementation},
                "case": case,
                "metrics": metrics,
            }
        )
    return measurements


def flatten_measurements(result: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for measurement in result["measurements"]:
        implementation = measurement["implementation"]
        row = {"implementation": implementation["name"]}
        for key in ("variant", "version", "revision"):
            if key in implementation:
                row[f"implementation_{key}"] = implementation[key]
        row.update(measurement["case"])
        row.update(measurement.get("metrics", {}))
        for name, timing in measurement.get("timings", {}).items():
            row.update({f"{name}_{key}": value for key, value in timing.items()})
        row.update(measurement.get("memory_bytes", {}))
        rows.append(row)
    return rows
