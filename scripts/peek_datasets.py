#!/usr/bin/env python3
"""Preview the first 5 rows of hotels, world cities, and crime CSVs.

- Local: from project root: ``python3 scripts/peek_datasets.py`` (expects ``data/raw/`` or Kaggle layout).
- Kaggle: attach the three datasets, clone repo, run this script; paths resolve under ``/kaggle/input``.
  See ``notebooks/kaggle_bootstrap.ipynb``.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.raw_data_paths import discover_raw_paths


def peek(label: str, path: Path, *, encoding: str, n: int = 5) -> None:
    print(f"\n{'=' * 60}\n{label}: {path}\n{'=' * 60}")
    df = pd.read_csv(path, encoding=encoding, nrows=n, low_memory=False)
    print(df.to_string())
    print(f"(showing {len(df)} rows)")


def main() -> None:
    try:
        paths = discover_raw_paths()
    except FileNotFoundError as e:
        print(e, file=sys.stderr)
        sys.exit(1)
    peek("hotels", paths["hotels"], encoding="latin-1")
    peek("world_cities", paths["world"], encoding="utf-8")
    peek("crime", paths["crime"], encoding="utf-8")


if __name__ == "__main__":
    main()
