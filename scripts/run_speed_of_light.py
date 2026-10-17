#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = ["typer"]
# ///
"""How each tool's SKETCH and COMPARE throughput grows with the compute it is given.

GPU tools run under a private CUDA MPS daemon at each --sm-percent, which caps the SMs one
client may use (CUDA_MPS_ACTIVE_THREAD_PERCENTAGE); the SM count a cap yields is read back from
the device. CPU tools run at each --threads count. Every point is one
`run_micro_comparison.py --performance-only` report in OUTPUT_DIR, listed with its compute in
OUTPUT_DIR/points.jsonl for scripts/plot_speed_of_light.py.

    ./scripts/run_speed_of_light.py /localscratch/genomes --reference-count 10000 \\
      --output-dir results/sol-monster3
"""

import ctypes
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Annotated

import typer

ROOT = Path(__file__).resolve().parent.parent
MICRO = ROOT / "scripts/run_micro_comparison.py"
CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT = 16

app = typer.Typer(add_completion=False)


def device_sms() -> int:
    """SMs device 0 offers this process, which under MPS honours its thread percentage."""
    cuda = ctypes.CDLL("libcuda.so.1")
    device, count = ctypes.c_int(), ctypes.c_int()
    for status in (
        cuda.cuInit(0),
        cuda.cuDeviceGet(ctypes.byref(device), 0),
        cuda.cuDeviceGetAttribute(
            ctypes.byref(count), CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, device
        ),
    ):
        if status != 0:
            raise RuntimeError(f"CUDA driver call failed with status {status}")
    return count.value


def sms_under(env: dict[str, str]) -> int:
    """SMs a fresh client started with @p env gets."""
    probe = f"import sys; sys.path.insert(0, {str(ROOT / 'scripts')!r}); "
    probe += "from run_speed_of_light import device_sms; print(device_sms())"
    return int(subprocess.run([sys.executable, "-c", probe], env=env, check=True,
                              capture_output=True, text=True).stdout)


@app.command()
def main(
    genomes: Annotated[Path, typer.Argument(exists=True, file_okay=False, help="Genome directory")],
    output_dir: Annotated[Path, typer.Option(help="Reports and points.jsonl land here")],
    reference_count: Annotated[int, typer.Option(min=1, help="First N genomes, all-to-all")] = 10000,
    gpu_tools: Annotated[str, typer.Option(help="Tools run at each SM cap")] = (
        "cuddl,cub-exact,pp-sketchlib,hypergen-gpu"
    ),
    cpu_tools: Annotated[str, typer.Option(help="Tools run at each thread count")] = (
        "simdsketch,sketchlib,dashing2,rabbitsketch,bbtools,hypergen-cpu,skani"
    ),
    sm_percent: Annotated[list[int], typer.Option(min=1, max=100, help="MPS SM caps")] = [
        10, 20, 30, 50, 70, 100
    ],
    threads: Annotated[list[int], typer.Option(min=1, help="CPU thread counts")] = [
        8, 15, 30, 60, 120
    ],
    gpu_threads: Annotated[int, typer.Option(min=1, help="Host threads for GPU tools")] = 120,
    samples: Annotated[int, typer.Option(min=1)] = 3,
    warmups: Annotated[int, typer.Option(min=0)] = 1,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    points = output_dir / "points.jsonl"

    def measure(tool: str, label: str, thread_count: int, env: dict[str, str]) -> Path:
        report = output_dir / f"{tool}-{label}.json"
        subprocess.run(
            [str(MICRO), str(genomes), "--reference-count", str(reference_count),
             "--topology", "all-to-all", "--performance-only", "--tools", tool,
             "--samples", str(samples), "--warmups", str(warmups),
             "--threads", str(thread_count), "--idle-seconds", "0", "--output", str(report)],
            env=env, check=True,
        )
        return report

    def record(point: dict) -> None:
        with points.open("a") as stream:
            stream.write(json.dumps(point) + "\n")

    full_sms = device_sms()
    cores = len(os.sched_getaffinity(0)) // (2 if _smt_enabled() else 1)
    with tempfile.TemporaryDirectory(prefix="sol-mps-") as mps:
        env = os.environ | {
            "CUDA_MPS_PIPE_DIRECTORY": f"{mps}/pipe",
            "CUDA_MPS_LOG_DIRECTORY": f"{mps}/log",
        }
        for directory in ("pipe", "log"):
            Path(mps, directory).mkdir()
        subprocess.run(["nvidia-cuda-mps-control", "-d"], env=env, check=True)
        try:
            for percent in sm_percent:
                capped = env | {"CUDA_MPS_ACTIVE_THREAD_PERCENTAGE": str(percent)}
                sms = sms_under(capped)
                for tool in filter(None, gpu_tools.split(",")):
                    report = measure(tool, f"sm{percent}", gpu_threads, capped)
                    record({"tool": tool, "kind": "gpu", "sm_percent": percent, "sms": sms,
                            "total": full_sms, "report": report.name})
        finally:
            subprocess.run(["nvidia-cuda-mps-control"], input="quit\n", text=True, env=env,
                           check=True)
    for count in threads:
        for tool in filter(None, cpu_tools.split(",")):
            report = measure(tool, f"t{count}", count, dict(os.environ))
            record({"tool": tool, "kind": "cpu", "threads": count, "cores": min(count, cores),
                    "total": cores, "report": report.name})


def _smt_enabled() -> bool:
    smt = Path("/sys/devices/system/cpu/smt/active")
    return smt.exists() and smt.read_text().strip() == "1"


if __name__ == "__main__":
    app()
