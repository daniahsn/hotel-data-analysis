#!/usr/bin/env python3
"""Print the first 5 text lines of each raw hotels / world / crime CSV."""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.raw_data_paths import DATASET_ORDER, discover_raw_paths

N = 5
ENCODING = {"hotels": "latin-1", "world": "utf-8-sig", "crime": "utf-8-sig"}


def main() -> None:
    try:
        paths = discover_raw_paths()
    except FileNotFoundError as e:
        print(e, file=sys.stderr)
        sys.exit(1)

    for key in DATASET_ORDER:
        path = paths[key]
        enc = ENCODING[key]
        print(f"\n{'=' * 60}\n{key}: {path.name}\n{path}\n{'=' * 60}")
        with path.open(encoding=enc, errors="replace") as f:
            for i, line in enumerate(f):
                if i >= N:
                    break
                print(line.rstrip("\n"))


if __name__ == "__main__":
    main()
