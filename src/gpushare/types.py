# Copyright (c) 2025 Vahab Jabrayilov (vjabrayilov@cs.columbia.edu)
# Copyright (c) 2025 DAPLab of Columbia University (https://daplab.cs.columbia.edu/)
# Copyright (c) 2025 The Trustees of Columbia University in the City of New York

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional


@dataclass
class RequestSpec:
    """A single request to an OpenAI-compatible /v1/chat/completions endpoint."""

    prompt: str
    max_tokens: int = 256
    temperature: float = 0.0
    top_p: float = 1.0


@dataclass
class LoadGenResult:
    """Aggregated metrics from one load generation run."""

    n_requests: int
    concurrency: int
    wall_time_s: float

    # Latency statistics (seconds)
    p50_s: float
    p95_s: float
    p99_s: float

    # Token accounting
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int

    # Derived throughput metrics
    req_per_s: float
    tok_per_s: float

    # Raw per-request records for debugging/analysis (optional)
    records: Optional[List[Dict[str, Any]]] = None
