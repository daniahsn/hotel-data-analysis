#!/usr/bin/env python3
"""Print the first 5 lines of each CSV under data/raw/."""

from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW = PROJECT_ROOT / "data" / "raw"
N = 5


def main() -> None:
    files = sorted(RAW.glob("*.csv"))
    if not files:
        print(f"No CSV files in {RAW}")
        return

    for path in files:
        print(f"\n{'=' * 60}\n{path.name}\n{'=' * 60}")
        with path.open(encoding="utf-8-sig", errors="replace") as f:
            for i, line in enumerate(f):
                if i >= N:
                    break
                print(line.rstrip("\n"))


if __name__ == "__main__":
    main()
