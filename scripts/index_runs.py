# Copyright (c) 2025 Vahab Jabrayilov (vjabrayilov@cs.columbia.edu)
# Copyright (c) 2025 DAPLab of Columbia University (https://daplab.cs.columbia.edu/)
# Copyright (c) 2025 The Trustees of Columbia University in the City of New York

"""Build a simple experiment index for filesystem-based tracking.

This scans a runs/ directory for summary.json files and writes:
- runs/index.csv
- runs/index.json

Usage
-----
python scripts/index_runs.py --runs runs --out runs

You can point --runs to a subfolder such as runs/baseline.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import pandas as pd
from gpushare.analysis import load_run_dir, summarize


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="runs", help="Root runs directory to scan")
    ap.add_argument("--out", default="runs", help="Output directory for index files")
    args = ap.parse_args()

    df = load_run_dir(args.runs)
    summ = summarize(df)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    summ.to_csv(out / "index.csv", index=False)
    (out / "index.json").write_text(summ.to_json(orient="records", indent=2), encoding="utf-8")

    print(f"Wrote {out / 'index.csv'} and {out / 'index.json'}")


if __name__ == "__main__":
    main()
