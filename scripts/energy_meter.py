"""Node energy counters: RAPL package domains for the CPUs, NVML total energy for the GPUs.

Both count since boot, so the energy of a run is the difference of two readings. GPUs count only
if CUDA_VISIBLE_DEVICES lists them (UUIDs or NVML indices; unset means all, empty means none):
on MOGON a job sees every GPU of an exclusive node, allocated or not, and seven idle A100s drew
about 480 W. RAPL covers every socket, so on a shared node other jobs' CPU energy is counted
too: run on whole nodes. Neither covers DRAM on AMD, disks, NICs or fans.

Grace Hopper has no RAPL, and its NVML energy counts the whole module. There the GPU is billed its
own power, integrated by a background thread from when the meter starts, and the CPU the rest
of the module: Grace, its LPDDR5X and SysIO.
"""

import bisect
import ctypes
import os
import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import Self

RAPL = Path("/sys/class/powercap")
HWMON = Path("/sys/class/hwmon")


class PowerIntegrator:
    """Joules since construction of @p watts, read every @p period seconds."""

    def __init__(self, watts: Callable[[], float], period: float = 0.05) -> None:
        self.watts = watts
        self.period = period
        self.total = 0.0
        self._lock = threading.Lock()
        threading.Thread(target=self._run, daemon=True).start()

    def _run(self) -> None:
        last = time.monotonic()
        while True:
            time.sleep(self.period)
            now = time.monotonic()
            watts = self.watts()
            with self._lock:
                self.total += watts * (now - last)
            last = now

    def __call__(self) -> float:
        with self._lock:
            return self.total


class EnergyMeter:
    """Reads the counters this host exposes; a source that is missing or unreadable (RAPL is
    root-only on some hosts) is left out, and its joules are reported as None."""

    def __init__(self) -> None:
        self.packages = [
            domain
            for domain in sorted(RAPL.glob("intel-rapl:[0-9]*"))
            if ":"
            not in domain.name.removeprefix("intel-rapl:")  # packages, not subdomains
            and os.access(domain / "energy_uj", os.R_OK)
        ]
        # The value each CPU counter wraps at, 0 for one that never does.
        self.ranges = [
            int((d / "max_energy_range_uj").read_text()) for d in self.packages
        ]
        self.gpu_power: list[PowerIntegrator] = []
        self.gpus: list[ctypes.c_void_p] = []
        try:
            self.nvml = ctypes.CDLL("libnvidia-ml.so.1")
        except OSError:
            return
        self._check(self.nvml.nvmlInit_v2())
        visible = os.environ.get("CUDA_VISIBLE_DEVICES")
        if visible is None:
            count = ctypes.c_uint()
            self._check(self.nvml.nvmlDeviceGetCount_v2(ctypes.byref(count)))
            devices = [str(index) for index in range(count.value)]
        else:
            devices = [d.strip() for d in visible.split(",") if d.strip()]
        for device in devices:
            handle = ctypes.c_void_p()
            if device.startswith("GPU-"):
                status = self.nvml.nvmlDeviceGetHandleByUUID(
                    device.encode(), ctypes.byref(handle)
                )
            else:
                status = self.nvml.nvmlDeviceGetHandleByIndex_v2(
                    int(device), ctypes.byref(handle)
                )
            self._check(status)
            self.gpus.append(handle)
        if not self.packages and any(
            (hwmon / "power1_oem_info").read_text().startswith("Module Power Socket")
            for hwmon in HWMON.glob("hwmon*")
            if (hwmon / "power1_oem_info").exists()
        ):
            self.gpu_power = [
                PowerIntegrator(lambda handle=handle: self._milliwatts(handle) / 1e3)
                for handle in self.gpus
            ]
            # Module minus GPU can step back between two readings close together.
            self.ranges = [0] * len(self.gpus)

    @staticmethod
    def _check(status: int) -> None:
        if status != 0:
            raise RuntimeError(f"NVML call failed with status {status}")

    def _milliwatts(self, handle: ctypes.c_void_p) -> int:
        milliwatts = ctypes.c_uint()
        self._check(self.nvml.nvmlDeviceGetPowerUsage(handle, ctypes.byref(milliwatts)))
        return milliwatts.value

    def read(self) -> tuple[list[int], list[int]]:
        """Microjoules per CPU and millijoules per GPU, since boot or, on Grace Hopper, the GPU
        since the meter started."""
        cpus = [int((d / "energy_uj").read_text()) for d in self.packages]
        gpus = []
        for handle in self.gpus:
            millijoules = ctypes.c_ulonglong()
            self._check(
                self.nvml.nvmlDeviceGetTotalEnergyConsumption(
                    handle, ctypes.byref(millijoules)
                )
            )
            gpus.append(millijoules.value)
        if self.gpu_power:
            modules = gpus
            gpus = [int(power() * 1e3) for power in self.gpu_power]
            cpus = [(m - g) * 1000 for m, g in zip(modules, gpus)]
        return cpus, gpus

    def joules(
        self, start: tuple[list[int], list[int]], end: tuple[list[int], list[int]]
    ) -> dict:
        """Energy between two readings: {"cpu_j": ..., "gpu_j": ...}, None where unmeasured."""
        cpu = (
            sum(
                b - a + (wrap if b < a and wrap else 0)
                for a, b, wrap in zip(start[0], end[0], self.ranges)  # RAPL wraps
            )
            / 1e6
        )
        gpu = sum(b - a for a, b in zip(start[1], end[1])) / 1e3
        return {
            "cpu_j": cpu if self.ranges else None,
            "gpu_j": gpu if self.gpus else None,
        }

    def idle_watts(self, seconds: float) -> dict:
        """Mean power over @p seconds of sleep, {"cpu_w": ..., "gpu_w": ...}, None where
        unmeasured. On an otherwise idle node this is the power a run draws without working;
        a run's energy above idle subtracts it times the run's wall time."""
        start = self.read()
        time.sleep(seconds)
        joules = self.joules(start, self.read())
        return {
            "cpu_w": None if joules["cpu_j"] is None else joules["cpu_j"] / seconds,
            "gpu_w": None if joules["gpu_j"] is None else joules["gpu_j"] / seconds,
        }


class EnergySampler:
    """Samples @p meter on a background thread while open, so a tool that times its own phases
    can be billed for them: given a phase's wall-clock (epoch) start and end, @ref joules
    interpolates the counters between the samples around them."""

    def __init__(self, meter: EnergyMeter, period: float = 0.05) -> None:
        self.meter = meter
        self.period = period
        # Epoch seconds, and CPU and GPU joules since the first sample.
        self.times: list[float] = []
        self.cpu: list[float] = []
        self.gpu: list[float] = []
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _run(self) -> None:
        previous = self.meter.read()
        cpu = gpu = 0.0
        self.times.append(time.time())
        self.cpu.append(cpu)
        self.gpu.append(gpu)
        while True:
            stop = self._stop.wait(self.period)
            current = self.meter.read()
            # Step by step, so each RAPL difference stays within one wrap.
            step = self.meter.joules(previous, current)
            previous = current
            cpu += step["cpu_j"] or 0.0
            gpu += step["gpu_j"] or 0.0
            self.times.append(time.time())
            self.cpu.append(cpu)
            self.gpu.append(gpu)
            if stop:
                return

    def __enter__(self) -> Self:
        self._thread.start()
        return self

    def __exit__(self, *_) -> None:
        self._stop.set()
        self._thread.join()

    def _at(self, series: list[float], t: float) -> float:
        i = min(max(bisect.bisect_left(self.times, t), 1), len(self.times) - 1)
        t0, t1 = self.times[i - 1], self.times[i]
        weight = 0.0 if t1 == t0 else min(max((t - t0) / (t1 - t0), 0.0), 1.0)
        return series[i - 1] + weight * (series[i] - series[i - 1])

    def joules(self, start: float, end: float) -> dict:
        """Energy between epoch seconds @p start and @p end, as EnergyMeter.joules reports it."""
        return {
            "cpu_j": self._at(self.cpu, end) - self._at(self.cpu, start)
            if self.meter.ranges
            else None,
            "gpu_j": self._at(self.gpu, end) - self._at(self.gpu, start)
            if self.meter.gpus
            else None,
        }
