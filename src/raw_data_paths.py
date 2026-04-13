"""Resolve raw CSV paths: Kaggle `/kaggle/input` first, else local `data/raw/`."""

from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
KAGGLE_INPUT = Path("/kaggle/input")

# Order used when iterating all three datasets (e.g. head script).
DATASET_ORDER: tuple[str, ...] = ("hotels", "world", "crime")

# Filename hints when searching under each `/kaggle/input/<dataset>/` tree.
SEARCH_NAMES: dict[str, tuple[str, ...]] = {
    "hotels": ("hotels.csv",),
    "world": ("worldcitiespop.csv",),
    "crime": ("CrimeIndex.csv", "World Crime Index .csv"),
}

# Preferred local filenames under `data/raw/` (optional dev fallback).
LOCAL_NAMES: dict[str, tuple[str, ...]] = {
    "hotels": ("hotels.csv",),
    "world": ("worldcitiespop.csv",),
    "crime": ("CrimeIndex.csv", "World Crime Index .csv"),
}


def is_kaggle_runtime() -> bool:
    return KAGGLE_INPUT.is_dir()


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


def _find_in_raw_dir(raw: Path, names: tuple[str, ...]) -> Path | None:
    if not raw.is_dir():
        return None
    for name in names:
        p = raw / name
        if p.is_file():
            return p
    for p in raw.glob("*.csv"):
        if p.name.lower() in {n.lower() for n in names}:
            return p
    return None


def discover_raw_paths() -> dict[str, Path]:
    """Return paths for hotels, world cities, and crime CSVs."""
    if is_kaggle_runtime():
        out: dict[str, Path] = {}
        missing: list[str] = []
        for key in DATASET_ORDER:
            p = _find_csv_under(KAGGLE_INPUT, SEARCH_NAMES[key])
            if p is None:
                missing.append(key)
            else:
                out[key] = p
        if missing:
            tops = [p.name for p in KAGGLE_INPUT.iterdir()] if KAGGLE_INPUT.is_dir() else []
            raise FileNotFoundError(
                f"Could not find {missing} under {KAGGLE_INPUT}. "
                f"Add those datasets via **Add Data** in the notebook. Top-level inputs: {tops}"
            )
        return out

    raw = PROJECT_ROOT / "data" / "raw"
    out = {}
    missing: list[str] = []
    for key in DATASET_ORDER:
        p = _find_in_raw_dir(raw, LOCAL_NAMES[key])
        if p is None:
            missing.append(f"{key} (tried {', '.join(LOCAL_NAMES[key])} under {raw})")
        else:
            out[key] = p
    if missing:
        raise FileNotFoundError(
            "Missing raw files locally:\n- "
            + "\n- ".join(missing)
            + "\nAttach the same Kaggle datasets in a notebook, or place the CSVs in data/raw/."
        )
    return out
