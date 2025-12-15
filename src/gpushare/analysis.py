# Copyright (c) 2025 Vahab Jabrayilov (vjabrayilov@cs.columbia.edu)
# Copyright (c) 2025 DAPLab of Columbia University (https://daplab.cs.columbia.edu/)
# Copyright (c) 2025 The Trustees of Columbia University in the City of New York

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import pandas as pd


def load_run_dir(run_dir: str | Path) -> pd.DataFrame:
    run_dir = Path(run_dir)
    rows: List[Dict] = []
    for p in sorted(run_dir.glob("**/summary.json")):
        rows.append(pd.read_json(p, typ="series").to_dict())
    if not rows:
        raise FileNotFoundError(f"No summary.json files under {run_dir}")
    return pd.DataFrame(rows)


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    keep = [
        "experiment", "scenario", "concurrency", "n_requests", "wall_time_s",
        "p50_s", "p95_s", "p99_s", "prompt_tokens", "completion_tokens", "total_tokens",
        "req_per_s", "tok_per_s",
    ]
    cols = [c for c in keep if c in df.columns]
    out = df[cols].copy()
    out.sort_values(["scenario", "concurrency"], inplace=True)
    return out


def read_gpu_csv(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    return df


def gpu_summary(df: pd.DataFrame) -> Dict[str, float]:
    return {
        "gpu_util_mean": float(df["gpu_util_pct"].mean()),
        "gpu_util_p95": float(df["gpu_util_pct"].quantile(0.95)),
        "mem_util_mean": float(df["mem_util_pct"].mean()),
        "power_mean_w": float(df["power_w"].mean()),
    }
