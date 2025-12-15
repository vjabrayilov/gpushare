# Copyright (c) 2025 Vahab Jabrayilov (vjabrayilov@cs.columbia.edu)
# Copyright (c) 2025 DAPLab of Columbia University (https://daplab.cs.columbia.edu/)
# Copyright (c) 2025 The Trustees of Columbia University in the City of New York

from __future__ import annotations

import os
import signal
import subprocess
import time
from dataclasses import dataclass
from typing import Dict, Optional, Sequence


@dataclass
class ServerProcess:
    name: str
    host: str
    port: int
    popen: subprocess.Popen

    def terminate(self, timeout_s: float = 20.0) -> None:
        if self.popen.poll() is not None:
            return
        self.popen.send_signal(signal.SIGINT)
        try:
            self.popen.wait(timeout=timeout_s)
        except subprocess.TimeoutExpired:
            self.popen.kill()
            self.popen.wait(timeout=5)


def start_vllm_server(
    *,
    model: str,
    host: str,
    port: int,
    dtype: str = "float16",
    max_model_len: int = 2048,
    gpu_memory_utilization: float = 0.9,
    enforce_eager: bool = False,
    quantization: Optional[str] = None,
    extra_args: Optional[Sequence[str]] = None,
    env: Optional[Dict[str, str]] = None,
    log_path: Optional[str] = None,
    ready_timeout_s: float = 240.0,
) -> ServerProcess:
    """Start the vLLM OpenAI-compatible server as a subprocess.

    This function is intentionally conservative: it only waits for the TCP port to open.
    """
    cmd = [
        "python",
        "-m",
        "vllm.entrypoints.openai.api_server",
        "--model",
        model,
        "--host",
        host,
        "--port",
        str(port),
        "--dtype",
        dtype,
        "--max-model-len",
        str(max_model_len),
        "--gpu-memory-utilization",
        str(gpu_memory_utilization),
    ]
    if enforce_eager:
        cmd.append("--enforce-eager")
    if quantization:
        cmd += ["--quantization", quantization]
    if extra_args:
        cmd += list(extra_args)

    merged_env = os.environ.copy()
    if env:
        merged_env.update(env)

    stdout = stderr = None
    if log_path:
        log_file = open(log_path, "w", encoding="utf-8")
        stdout = stderr = log_file

    popen = subprocess.Popen(cmd, env=merged_env, stdout=stdout, stderr=stderr)

    # Wait for port to open (best-effort).
    import socket

    deadline = time.time() + ready_timeout_s
    while time.time() < deadline:
        try:
            with socket.create_connection((host, port), timeout=1.0):
                return ServerProcess(name=f"vllm:{port}", host=host, port=port, popen=popen)
        except OSError:
            if popen.poll() is not None:
                raise RuntimeError(f"vLLM server crashed early; see log at {log_path!r}")
            time.sleep(1.0)

    raise TimeoutError(f"vLLM server did not become ready within {ready_timeout_s}s on {host}:{port}")
