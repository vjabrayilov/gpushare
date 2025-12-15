# Copyright (c) 2025 Vahab Jabrayilov (vjabrayilov@cs.columbia.edu)
# Copyright (c) 2025 DAPLab of Columbia University (https://daplab.cs.columbia.edu/)
# Copyright (c) 2025 The Trustees of Columbia University in the City of New York

from __future__ import annotations

import asyncio
import json
import os
import socket
import subprocess
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

from .config import ExperimentConfig, ServerConfig, WorkloadConfig
from .loadgen import load_prompts, make_request_set, result_to_json, run_load
from .monitor import GPUMonitor
from .mps import start_mps, stop_mps
from .servers import ServerProcess, start_vllm_server


def _write_json(path: Path, obj: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, sort_keys=True)


def _try_run(cmd: List[str]) -> Optional[str]:
    try:
        out = subprocess.check_output(cmd, stderr=subprocess.STDOUT, text=True)
        return out.strip()
    except Exception:
        return None


def _collect_env_snapshot() -> Dict[str, Any]:
    """Collect lightweight environment metadata for local experiment tracking."""
    snap: Dict[str, Any] = {}
    snap["hostname"] = socket.gethostname()
    snap["python"] = _try_run(["python", "--version"])
    snap["pip_freeze"] = _try_run(["python", "-m", "pip", "freeze"])
    snap["nvidia_smi"] = _try_run(["nvidia-smi"])
    snap["git_commit"] = _try_run(["git", "rev-parse", "HEAD"])
    snap["git_status"] = _try_run(["git", "status", "--porcelain"])
    snap["timestamp"] = time.strftime("%Y-%m-%d %H:%M:%S")
    return snap


def run_single_server_trial(
    *,
    exp: ExperimentConfig,
    srv: ServerConfig,
    wrk: WorkloadConfig,
    scenario: str,
    run_id: str,
) -> Path:
    """Run a single-server experiment (baseline or MPS) and write artifacts under runs/."""
    out = Path(exp.out_dir) / exp.name / run_id
    out.mkdir(parents=True, exist_ok=True)

    if exp.write_env_snapshot:
        _write_json(out / "env.json", _collect_env_snapshot())

    # Optional MPS toggle
    if exp.use_mps:
        start_mps()

    server_log = out / "server.log"
    server = start_vllm_server(
        model=srv.model,
        host=srv.host,
        port=srv.port,
        dtype=srv.dtype,
        max_model_len=srv.max_model_len,
        gpu_memory_utilization=srv.gpu_memory_utilization,
        enforce_eager=srv.enforce_eager,
        quantization=srv.quantization,
        extra_args=srv.extra_args,
        log_path=str(server_log),
    )

    # Prepare workload
    prompts = load_prompts(wrk.prompts_path)
    warmup = make_request_set(
        prompts,
        exp.warmup_requests,
        max_tokens=min(32, wrk.max_tokens),
        temperature=0.0,
        top_p=1.0,
        seed=wrk.seed,
    )
    reqs = make_request_set(
        prompts,
        wrk.n_requests,
        max_tokens=wrk.max_tokens,
        temperature=wrk.temperature,
        top_p=wrk.top_p,
        seed=wrk.seed,
    )

    # GPU monitor (best-effort)
    gpu_csv = out / "gpu.csv"
    monitor = GPUMonitor(gpu_csv)
    import threading

    mon_thread = threading.Thread(target=monitor.run, kwargs={"duration_s": None}, daemon=True)
    mon_thread.start()

    base_url = f"http://{srv.host}:{srv.port}"

    try:
        # Warmup
        asyncio.run(
            run_load(
                base_url=base_url,
                requests_=warmup,
                concurrency=min(4, wrk.concurrency),
                model=srv.model,
                seed=wrk.seed,
            )
        )

        # Timed run
        result = asyncio.run(
            run_load(
                base_url=base_url,
                requests_=reqs,
                concurrency=wrk.concurrency,
                model=srv.model,
                seed=wrk.seed,
            )
        )
    finally:
        monitor.stop()
        mon_thread.join(timeout=2)
        server.terminate()
        if exp.use_mps:
            stop_mps()

    summary = result_to_json(result)
    summary.update(
        {
            "experiment": exp.name,
            "scenario": scenario,
            "model": srv.model,
            "server": asdict(srv),
            "workload": asdict(wrk),
            "mps": exp.use_mps,
            "mig_emulation": False,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
    )
    _write_json(out / "summary.json", summary)
    return out


def run_mig_emulation_trial(
    *,
    exp: ExperimentConfig,
    srv: ServerConfig,
    wrk: WorkloadConfig,
    scenario: str,
    run_id: str,
) -> Path:
    """Run a MIG-like emulation by launching multiple vLLM servers on the same GPU.

    We approximate MIG by capping each server's memory footprint via vLLM's
    --gpu-memory-utilization and (optionally) capping SM time via CUDA_MPS_ACTIVE_THREAD_PERCENTAGE.

    This is NOT hardware isolation; it is a practical emulation for environments without MIG-capable GPUs.
    """
    out = Path(exp.out_dir) / exp.name / run_id
    out.mkdir(parents=True, exist_ok=True)

    if exp.write_env_snapshot:
        _write_json(out / "env.json", _collect_env_snapshot())

    # For the emulation to be stable, we enable MPS.
    start_mps()
    env_common: Dict[str, str] = {}
    if exp.mig_active_thread_pct is not None:
        env_common["CUDA_MPS_ACTIVE_THREAD_PERCENTAGE"] = str(exp.mig_active_thread_pct)

    servers: List[ServerProcess] = []
    ports = [srv.port + i for i in range(exp.mig_partitions)]
    try:
        for i, port in enumerate(ports):
            server_log = out / f"server_{i}.log"
            s = start_vllm_server(
                model=srv.model,
                host=srv.host,
                port=port,
                dtype=srv.dtype,
                max_model_len=srv.max_model_len,
                gpu_memory_utilization=exp.mig_memory_fraction,
                enforce_eager=srv.enforce_eager,
                quantization=srv.quantization,
                extra_args=srv.extra_args,
                env=env_common,
                log_path=str(server_log),
            )
            servers.append(s)

        # GPU monitor
        gpu_csv = out / "gpu.csv"
        monitor = GPUMonitor(gpu_csv)
        import threading

        mon_thread = threading.Thread(target=monitor.run, kwargs={"duration_s": None}, daemon=True)
        mon_thread.start()

        prompts = load_prompts(wrk.prompts_path)
        reqs = make_request_set(
            prompts,
            wrk.n_requests,
            max_tokens=wrk.max_tokens,
            temperature=wrk.temperature,
            top_p=wrk.top_p,
            seed=wrk.seed,
        )

        # Split requests across servers and run them simultaneously.
        split: List[List] = [[] for _ in servers]
        for idx, r in enumerate(reqs):
            split[idx % len(servers)].append(r)

        async def run_all():
            tasks = []
            for s, req_subset in zip(servers, split):
                url = f"http://{s.host}:{s.port}"
                tasks.append(
                    run_load(
                        base_url=url,
                        requests_=req_subset,
                        concurrency=max(1, wrk.concurrency // len(servers)),
                        model=srv.model,
                        seed=wrk.seed,
                    )
                )
            return await asyncio.gather(*tasks)

        t0 = time.perf_counter()
        results = asyncio.run(run_all())
        wall = time.perf_counter() - t0

        monitor.stop()
        mon_thread.join(timeout=2)

    finally:
        for s in servers:
            s.terminate()
        stop_mps()

    # Aggregate results
    n_requests = sum(r.n_requests for r in results)
    prompt_tokens = sum(r.prompt_tokens for r in results)
    completion_tokens = sum(r.completion_tokens for r in results)
    total_tokens = sum(r.total_tokens for r in results)

    # Combine latency distributions by concatenating per-request records when present.
    lat = []
    recs = []
    for r in results:
        if r.records:
            recs.extend(r.records)
            lat.extend([float(x["latency_s"]) for x in r.records])
    import numpy as np

    def pct(p: float) -> float:
        return float(np.percentile(np.array(lat, dtype=np.float64), p)) if lat else float("nan")

    summary = {
        "experiment": exp.name,
        "scenario": scenario,
        "concurrency": wrk.concurrency,
        "n_requests": int(n_requests),
        "wall_time_s": float(wall),
        "p50_s": pct(50),
        "p95_s": pct(95),
        "p99_s": pct(99),
        "prompt_tokens": int(prompt_tokens),
        "completion_tokens": int(completion_tokens),
        "total_tokens": int(total_tokens),
        "req_per_s": float(n_requests / wall) if wall > 0 else float("inf"),
        "tok_per_s": float(total_tokens / wall) if wall > 0 else float("inf"),
        "model": srv.model,
        "server": asdict(srv),
        "workload": asdict(wrk),
        "mps": True,
        "mig_emulation": True,
        "mig_partitions": exp.mig_partitions,
        "mig_memory_fraction": exp.mig_memory_fraction,
        "mig_active_thread_pct": exp.mig_active_thread_pct,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "records": recs if recs else None,
    }
    _write_json(out / "summary.json", summary)
    return out
