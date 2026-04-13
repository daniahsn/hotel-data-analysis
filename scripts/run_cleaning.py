#!/usr/bin/env python3
"""Run modular cleaning (steps 1–6) on hotels + world cities; write to ``data/processed/``."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.cleaning.hotel_world_clean import run_cleaning_pipeline


def main() -> None:
    p = argparse.ArgumentParser(description="Clean hotels + world cities (no crime).")
    p.add_argument(
        "--output-dir",
        type=Path,
        default=_ROOT / "data" / "processed",
        help="Directory for Parquet/CSV outputs",
    )
    p.add_argument(
        "--sample",
        type=int,
        default=None,
        metavar="N",
        help="Read only first N hotel rows (for local smoke tests)",
    )
    p.add_argument(
        "--format",
        choices=("parquet", "csv"),
        default="parquet",
        help="Output file format (default: parquet)",
    )
    p.add_argument("--no-join", action="store_true", help="Skip hotels ↔ cities join table")
    p.add_argument(
        "--hotels-encoding",
        default=None,
        metavar="ENC",
        help="Force hotels CSV encoding (default: try utf-8-sig then latin-1)",
    )
    args = p.parse_args()

    written = run_cleaning_pipeline(
        output_dir=args.output_dir,
        hotels_sample_rows=args.sample,
        also_join=not args.no_join,
        output_format=args.format,
        hotels_encoding=args.hotels_encoding,
    )
    for label, path in written.items():
        print(f"{label}: {path}")


if __name__ == "__main__":
    main()
