#!/usr/bin/env python3
"""Preview the first 5 rows of hotels, world cities, and crime CSVs.

- Local (Cursor): run from project root:  python scripts/peek_datasets.py
- Kaggle: clone repo to /kaggle/working/project, add your three datasets, then run this file
  (see notebooks/kaggle_bootstrap.ipynb).
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
KAGGLE_INPUT = Path("/kaggle/input")


def _find_csv_under(root: Path, names: tuple[str, ...]) -> Path | None:
    if not root.is_dir():
        return None
    for sub in root.iterdir():
        if not sub.is_dir():
            continue
        for name in names:
            p = sub / name
            if p.is_file():
                return p
        for p in sub.rglob("*.csv"):
            pl = p.name.lower()
            for name in names:
                if pl == name.lower():
                    return p
    return None


def discover_paths() -> dict[str, Path]:
    """Resolve paths on Kaggle (auto) or locally (data/raw)."""
    if KAGGLE_INPUT.is_dir():
        hotels = _find_csv_under(KAGGLE_INPUT, ("hotels.csv",))
        world = _find_csv_under(KAGGLE_INPUT, ("worldcitiespop.csv",))
        crime = _find_csv_under(
            KAGGLE_INPUT,
            ("CrimeIndex.csv", "World Crime Index .csv"),
        )
        missing = [k for k, v in [("hotels", hotels), ("world", world), ("crime", crime)] if v is None]
        if missing:
            print("Could not find:", ", ".join(missing), "under", KAGGLE_INPUT)
            print("Top-level input folders:", [p.name for p in KAGGLE_INPUT.iterdir()])
            sys.exit(1)
        return {"hotels": hotels, "world": world, "crime": crime}  # type: ignore[return-value]

    raw = PROJECT_ROOT / "data" / "raw"
    paths = {
        "hotels": raw / "hotels.csv",
        "world": raw / "worldcitiespop.csv",
        "crime": raw / "CrimeIndex.csv",
    }
    for label, p in paths.items():
        if not p.is_file():
            sys.exit(f"Missing {label}: {p}")
    return paths


def peek(label: str, path: Path, *, encoding: str, n: int = 5) -> None:
    print(f"\n{'=' * 60}\n{label}: {path}\n{'=' * 60}")
    df = pd.read_csv(path, encoding=encoding, nrows=n, low_memory=False)
    print(df.to_string())
    print(f"(showing {len(df)} rows)")


def main() -> None:
    paths = discover_paths()
    peek("hotels", paths["hotels"], encoding="latin-1")
    peek("world_cities", paths["world"], encoding="utf-8")
    peek("crime", paths["crime"], encoding="utf-8")


if __name__ == "__main__":
    main()
