# Copyright (c) 2025 Vahab Jabrayilov (vjabrayilov@cs.columbia.edu)
# Copyright (c) 2025 DAPLab of Columbia University (https://daplab.cs.columbia.edu/)
# Copyright (c) 2025 The Trustees of Columbia University in the City of New York

"""Aggregate summaries under ./runs and generate plots.

Usage
-----
python scripts/analyze_results.py --runs runs/example --out runs/example/plots

This works for both real runs and example runs. If you did not install the package,
the script will automatically add ./src to PYTHONPATH when run from the repo root.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import matplotlib.pyplot as plt
import pandas as pd

from gpushare.analysis import load_run_dir, summarize


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", required=True, help="Directory containing run subfolders with summary.json")
    ap.add_argument("--out", required=True, help="Output directory for tables and figures")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    df = load_run_dir(args.runs)
    summ = summarize(df)

    summ.to_csv(out / "summary.csv", index=False)

    if "concurrency" in summ.columns:
        for metric, fname, ylabel in [
            ("tok_per_s", "throughput_tokens_per_s.png", "Throughput (tokens/s)"),
            ("p99_s", "p99_latency_s.png", "P99 latency (s)"),
        ]:
            plt.figure()
            for scenario, g in summ.groupby("scenario"):
                g = g.sort_values("concurrency")
                plt.plot(g["concurrency"], g[metric], marker="o", label=scenario)
            plt.xlabel("Concurrency")
            plt.ylabel(ylabel)
            plt.legend()
            plt.tight_layout()
            plt.savefig(out / fname, dpi=200)

    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
