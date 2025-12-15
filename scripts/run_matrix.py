# Copyright (c) 2025 Vahab Jabrayilov (vjabrayilov@cs.columbia.edu)
# Copyright (c) 2025 DAPLab of Columbia University (https://daplab.cs.columbia.edu/)
# Copyright (c) 2025 The Trustees of Columbia University in the City of New York

"""Run a small experiment matrix and write results to ./runs.

Examples
--------
python scripts/run_matrix.py --model mistralai/Mistral-7B-Instruct-v0.2 --concurrency 1 2 4 8 16

Notes
-----
This script assumes you have a CUDA-capable machine and have installed the optional CUDA deps:
pip install -e '.[cuda]'

If you run it without installation from the repo root, it will add ./src to PYTHONPATH automatically.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from gpushare.config import ExperimentConfig, ServerConfig, WorkloadConfig
from gpushare.runner import run_mig_emulation_trial, run_single_server_trial


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="mistralai/Mistral-7B-Instruct-v0.2")
    ap.add_argument("--out-dir", default="runs")
    ap.add_argument("--concurrency", nargs="+", type=int, default=[1, 2, 4, 8, 16])
    ap.add_argument("--n-requests", type=int, default=200)
    ap.add_argument("--max-tokens", type=int, default=256)
    args = ap.parse_args()

    srv = ServerConfig(model=args.model)
    base_wrk = WorkloadConfig(n_requests=args.n_requests, max_tokens=args.max_tokens)

    for c in args.concurrency:
        # baseline
        exp = ExperimentConfig(
            name=f"baseline_c{c}", out_dir=args.out_dir, use_mps=False, mig_emulation=False
        )
        wrk = WorkloadConfig(**base_wrk.__dict__)
        wrk.concurrency = c
        rid = time.strftime("%Y%m%d_%H%M%S")
        run_single_server_trial(exp=exp, srv=srv, wrk=wrk, scenario="baseline", run_id=rid)

        # MPS
        exp = ExperimentConfig(
            name=f"mps_c{c}", out_dir=args.out_dir, use_mps=True, mig_emulation=False
        )
        rid = time.strftime("%Y%m%d_%H%M%S")
        run_single_server_trial(exp=exp, srv=srv, wrk=wrk, scenario="mps", run_id=rid)

        # MIG-like emulation
        exp = ExperimentConfig(
            name=f"mig_emulation_c{c}",
            out_dir=args.out_dir,
            use_mps=True,
            mig_emulation=True,
            mig_partitions=2,
            mig_memory_fraction=0.45,
            mig_active_thread_pct=50,
        )
        rid = time.strftime("%Y%m%d_%H%M%S")
        run_mig_emulation_trial(exp=exp, srv=srv, wrk=wrk, scenario="mig_emulation", run_id=rid)

    print("Done.")


if __name__ == "__main__":
    main()
