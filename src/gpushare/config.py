# Copyright (c) 2025 Vahab Jabrayilov (vjabrayilov@cs.columbia.edu)
# Copyright (c) 2025 DAPLab of Columbia University (https://daplab.cs.columbia.edu/)
# Copyright (c) 2025 The Trustees of Columbia University in the City of New York


from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml


@dataclass
class ServerConfig:
    engine: str = "vllm"          # Only vLLM is implemented in this repo.
    model: str = "mistralai/Mistral-7B-Instruct-v0.2"
    host: str = "127.0.0.1"
    port: int = 8000

    # vLLM-specific knobs
    dtype: str = "float16"
    max_model_len: int = 2048
    gpu_memory_utilization: float = 0.90
    enforce_eager: bool = False
    quantization: Optional[str] = None  # e.g., 'awq' / 'gptq'

    # Extra args passed through to vLLM
    extra_args: Optional[List[str]] = None


@dataclass
class WorkloadConfig:
    prompts_path: str = "data/prompts.jsonl"
    n_requests: int = 200
    concurrency: int = 8
    max_tokens: int = 256
    temperature: float = 0.0
    top_p: float = 1.0
    seed: int = 0


@dataclass
class ExperimentConfig:
    name: str = "default"
    out_dir: str = "runs"

    # Toggle MPS around runs
    use_mps: bool = False

    # When emulating MIG, we run N servers and cap their vLLM gpu_memory_utilization
    mig_emulation: bool = False
    mig_partitions: int = 2
    mig_memory_fraction: float = 0.45
    mig_active_thread_pct: Optional[int] = None  # Maps to CUDA_MPS_ACTIVE_THREAD_PERCENTAGE if set

    # Warmup
    warmup_requests: int = 20

    # Local tracking extras
    write_env_snapshot: bool = True


def load_yaml(path: str | Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_experiment_config(path: str | Path) -> tuple[ExperimentConfig, ServerConfig, WorkloadConfig]:
    cfg = load_yaml(path)
    exp = ExperimentConfig(**cfg.get("experiment", {}))
    srv = ServerConfig(**cfg.get("server", {}))
    wrk = WorkloadConfig(**cfg.get("workload", {}))
    return exp, srv, wrk
