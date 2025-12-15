# Copyright (c) 2025 Vahab Jabrayilov (vjabrayilov@cs.columbia.edu)
# Copyright (c) 2025 DAPLab of Columbia University (https://daplab.cs.columbia.edu/)
# Copyright (c) 2025 The Trustees of Columbia University in the City of New York

from __future__ import annotations

import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass
class MPSConfig:
    pipe_dir: Path = Path("/tmp/nvidia-mps")
    log_dir: Path = Path("/tmp/nvidia-mps-log")


def _bin(name: str) -> str:
    path = shutil.which(name)
    if not path:
        raise RuntimeError(
            f"Could not find '{name}' in PATH. Install CUDA toolkit or NVIDIA driver utilities."
        )
    return path


def start_mps(cfg: Optional[MPSConfig] = None) -> None:
    """Start the CUDA MPS control daemon.

    Notes
    -----
    This is a best-effort helper. Some managed environments may block MPS or require extra privileges.
    """
    cfg = cfg or MPSConfig()
    cfg.pipe_dir.mkdir(parents=True, exist_ok=True)
    cfg.log_dir.mkdir(parents=True, exist_ok=True)

    os.environ["CUDA_MPS_PIPE_DIRECTORY"] = str(cfg.pipe_dir)
    os.environ["CUDA_MPS_LOG_DIRECTORY"] = str(cfg.log_dir)

    subprocess.run([_bin("nvidia-cuda-mps-control"), "-d"], check=True)


def stop_mps() -> None:
    """Stop the CUDA MPS daemon (best-effort)."""
    proc = subprocess.Popen([_bin("nvidia-cuda-mps-control")], stdin=subprocess.PIPE, text=True)
    assert proc.stdin is not None
    proc.stdin.write("quit\n")
    proc.stdin.flush()
    proc.stdin.close()
    proc.wait(timeout=10)
