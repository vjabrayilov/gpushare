# Copyright (c) 2025 Vahab Jabrayilov (vjabrayilov@cs.columbia.edu)
# Copyright (c) 2025 DAPLab of Columbia University (https://daplab.cs.columbia.edu/)
# Copyright (c) 2025 The Trustees of Columbia University in the City of New York

from __future__ import annotations

import asyncio
import json
import random
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import aiohttp
import numpy as np
from tqdm import tqdm

from .types import LoadGenResult, RequestSpec


def _percentile(x: List[float], p: float) -> float:
    if not x:
        return float("nan")
    return float(np.percentile(np.array(x, dtype=np.float64), p))


def load_prompts(path: str | Path) -> List[str]:
    prompts: List[str] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            prompts.append(obj["prompt"])
    if not prompts:
        raise ValueError(f"No prompts found in {path}")
    return prompts


async def _one_request(
    session: aiohttp.ClientSession,
    url: str,
    req: RequestSpec,
    *,
    model: Optional[str] = None,
    request_id: Optional[str] = None,
    timeout_s: float = 120.0,
) -> Dict[str, Any]:
    payload = {
        "model": model or "gpt-3.5-turbo",
        "messages": [{"role": "user", "content": req.prompt}],
        "max_tokens": req.max_tokens,
        "temperature": req.temperature,
        "top_p": req.top_p,
    }
    headers = {"Content-Type": "application/json"}
    t0 = time.perf_counter()
    async with session.post(url, json=payload, headers=headers, timeout=timeout_s) as resp:
        data = await resp.json()
        t1 = time.perf_counter()
        latency_s = t1 - t0

        text = None
        if isinstance(data, dict) and "choices" in data and data["choices"]:
            choice0 = data["choices"][0]
            # OpenAI Chat Completions format
            if "message" in choice0 and "content" in choice0["message"]:
                text = choice0["message"]["content"]
            elif "text" in choice0:
                text = choice0["text"]

        usage = data.get("usage", {}) if isinstance(data, dict) else {}
        return {
            "request_id": request_id,
            "latency_s": latency_s,
            "status": resp.status,
            "prompt": req.prompt,
            "text": text,
            "usage": usage,
            "raw": data,
        }


async def run_load(
    *,
    base_url: str,
    requests_: List[RequestSpec],
    concurrency: int,
    model: Optional[str] = None,
    seed: int = 0,
) -> LoadGenResult:
    """Run a fixed request set against an OpenAI-compatible server and compute aggregate metrics."""
    random.Random(seed).shuffle(requests_)

    url = base_url.rstrip("/") + "/v1/chat/completions"
    connector = aiohttp.TCPConnector(limit=0, ttl_dns_cache=60)
    timeout = aiohttp.ClientTimeout(total=None)
    sem = asyncio.Semaphore(concurrency)

    latencies: List[float] = []
    records: List[Dict[str, Any]] = []
    prompt_tokens = 0
    completion_tokens = 0

    async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
        async def bounded(i: int, req: RequestSpec) -> Dict[str, Any]:
            async with sem:
                return await _one_request(session, url, req, model=model, request_id=str(i))

        t0 = time.perf_counter()
        tasks = [asyncio.create_task(bounded(i, r)) for i, r in enumerate(requests_)]
        for fut in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="requests"):
            rec = await fut
            records.append(rec)
            latencies.append(rec["latency_s"])

            usage = rec.get("usage") or {}
            prompt_tokens += int(usage.get("prompt_tokens") or 0)
            completion_tokens += int(usage.get("completion_tokens") or 0)

        wall = time.perf_counter() - t0

    total_tokens = prompt_tokens + completion_tokens
    req_per_s = len(records) / wall if wall > 0 else float("inf")
    tok_per_s = total_tokens / wall if wall > 0 else float("inf")

    return LoadGenResult(
        n_requests=len(records),
        concurrency=concurrency,
        wall_time_s=wall,
        p50_s=_percentile(latencies, 50),
        p95_s=_percentile(latencies, 95),
        p99_s=_percentile(latencies, 99),
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=total_tokens,
        req_per_s=req_per_s,
        tok_per_s=tok_per_s,
        records=records,
    )


def make_request_set(prompts: List[str], n_requests: int, *, max_tokens: int, temperature: float, top_p: float, seed: int = 0) -> List[RequestSpec]:
    rng = random.Random(seed)
    reqs: List[RequestSpec] = []
    for _ in range(n_requests):
        prompt = rng.choice(prompts)
        reqs.append(RequestSpec(prompt=prompt, max_tokens=max_tokens, temperature=temperature, top_p=top_p))
    return reqs


def result_to_json(result: LoadGenResult) -> Dict[str, Any]:
    d = asdict(result)
    # Don't dump raw responses by default (they can be huge). Keep per-request metrics.
    if d.get("records") is not None:
        for r in d["records"]:
            if "raw" in r:
                r["raw"] = None
    return d
