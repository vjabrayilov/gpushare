# Copyright (c) 2025 Vahab Jabrayilov (vjabrayilov@cs.columbia.edu)
# Copyright (c) 2025 DAPLab of Columbia University (https://daplab.cs.columbia.edu/)
# Copyright (c) 2025 The Trustees of Columbia University in the City of New York

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from .config import load_experiment_config
from .runner import run_mig_emulation_trial, run_single_server_trial


def main() -> None:
    p = argparse.ArgumentParser(prog="gpushare", description="LLM GPU sharing mini-benchmarks")
    sub = p.add_subparsers(dest="cmd", required=True)

    runp = sub.add_parser("run", help="Run one experiment config YAML")
    runp.add_argument("--config", required=True, help="Path to YAML config")
    runp.add_argument("--scenario", default=None, help="Override scenario label")
    runp.add_argument("--run-id", default=None, help="Override run id")
    runp.add_argument("--concurrency", type=int, default=None, help="Override workload concurrency")
    runp.add_argument("--n-requests", type=int, default=None, help="Override workload size")

    ap = sub.add_parser("print-config", help="Print the parsed config for debugging")
    ap.add_argument("--config", required=True)

    args = p.parse_args()

    if args.cmd == "print-config":
        exp, srv, wrk = load_experiment_config(args.config)
        print(json.dumps({"experiment": exp.__dict__, "server": srv.__dict__, "workload": wrk.__dict__}, indent=2))
        return

    exp, srv, wrk = load_experiment_config(args.config)
    if args.concurrency is not None:
        wrk.concurrency = args.concurrency
    if args.n_requests is not None:
        wrk.n_requests = args.n_requests

    scenario = args.scenario or exp.name
    run_id = args.run_id or time.strftime("%Y%m%d_%H%M%S")

    if exp.mig_emulation:
        out = run_mig_emulation_trial(exp=exp, srv=srv, wrk=wrk, scenario=scenario, run_id=run_id)
    else:
        out = run_single_server_trial(exp=exp, srv=srv, wrk=wrk, scenario=scenario, run_id=run_id)

    print(str(out))
