# Copyright (c) 2025 Vahab Jabrayilov (vjabrayilov@cs.columbia.edu)
# Copyright (c) 2025 DAPLab of Columbia University (https://daplab.cs.columbia.edu/)
# Copyright (c) 2025 The Trustees of Columbia University in the City of New York

from __future__ import annotations

import csv
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

try:
    from pynvml import (
        nvmlDeviceGetHandleByIndex,
        nvmlDeviceGetMemoryInfo,
        nvmlDeviceGetPowerUsage,
        nvmlDeviceGetUtilizationRates,
        nvmlInit,
        nvmlShutdown,
    )
    _NVML_OK = True
except Exception:  # pragma: no cover
    _NVML_OK = False


@dataclass
class GPUMonitorConfig:
    device_index: int = 0
    interval_s: float = 0.2


class GPUMonitor:
    """Lightweight NVML sampler writing a CSV file.

    If NVML (pynvml) isn't installed, the monitor becomes a no-op and will write only the header row.
    This keeps the rest of the pipeline runnable on non-GPU machines.
    """

    def __init__(self, out_csv: Path, cfg: Optional[GPUMonitorConfig] = None):
        self.out_csv = Path(out_csv)
        self.cfg = cfg or GPUMonitorConfig()
        self._running = False

    def run(self, duration_s: Optional[float] = None) -> None:
        self.out_csv.parent.mkdir(parents=True, exist_ok=True)
        with self.out_csv.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(
                ["t_s", "gpu_util_pct", "mem_util_pct", "mem_used_bytes", "mem_total_bytes", "power_w"]
            )

            if not _NVML_OK:
                return

            nvmlInit()
            handle = nvmlDeviceGetHandleByIndex(self.cfg.device_index)

            t0 = time.perf_counter()
            self._running = True
            while self._running:
                t = time.perf_counter() - t0
                util = nvmlDeviceGetUtilizationRates(handle)
                mem = nvmlDeviceGetMemoryInfo(handle)
                power_w = nvmlDeviceGetPowerUsage(handle) / 1000.0  # mW -> W

                mem_util_pct = (mem.used / mem.total) * 100.0 if mem.total else 0.0

                writer.writerow([t, util.gpu, util.memory, mem.used, mem.total, power_w])
                f.flush()

                if duration_s is not None and t >= duration_s:
                    break
                time.sleep(self.cfg.interval_s)

            nvmlShutdown()

    def stop(self) -> None:
        self._running = False
