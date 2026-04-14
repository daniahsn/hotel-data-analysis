"""Locate raw CSVs and the joined modeling Parquet (Kaggle + local)."""

from __future__ import annotations

from pathlib import Path

from src.name_mappings import JOINED_HOTELS_FILE_STEM

PROJECT_ROOT = Path(__file__).resolve().parent.parent
KAGGLE_INPUT = Path("/kaggle/input")

JOINED_HOTELS_PARQUET_GLOB = f"{JOINED_HOTELS_FILE_STEM}*.parquet"

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


def resolve_joined_hotels_parquet(project_root: Path | None = None) -> Path:
    """
    Path to the **joined hotels + world-cities** table used for modeling.

    Use this single helper on **Kaggle** and **locally** instead of hard-coding paths.

    - **Kaggle:** searches each dataset under ``/kaggle/input`` for
      ``hotels_with_cities*.parquet`` (prefers exact ``hotels_with_cities.parquet``).
    - **Local:** ``{project_root}/data/processed/`` with the same glob (default
      ``project_root`` = repository root).

    Name the processed file on disk or in your Kaggle Dataset using the stem
    ``hotels_with_cities`` (e.g. ``hotels_with_cities.parquet``).
    """
    if is_kaggle_runtime():
        candidates: list[Path] = []
        for top in sorted(KAGGLE_INPUT.iterdir()):
            if not top.is_dir():
                continue
            for p in top.rglob(JOINED_HOTELS_PARQUET_GLOB):
                if p.is_file():
                    candidates.append(p)
        if not candidates:
            tops = [p.name for p in KAGGLE_INPUT.iterdir()] if KAGGLE_INPUT.is_dir() else []
            raise FileNotFoundError(
                f"No {JOINED_HOTELS_PARQUET_GLOB} under {KAGGLE_INPUT}. "
                f"Add your processed joined dataset via **Add data**. Top-level inputs: {tops}"
            )

        def _sort_key(path: Path) -> tuple[int, str]:
            if path.name == f"{JOINED_HOTELS_FILE_STEM}.parquet":
                return (0, path.name)
            return (1, path.name)

        return sorted(candidates, key=_sort_key)[0]

    root = project_root or PROJECT_ROOT
    processed = root / "data" / "processed"
    direct = processed / f"{JOINED_HOTELS_FILE_STEM}.parquet"
    if direct.is_file():
        return direct
    matches = sorted(processed.glob(JOINED_HOTELS_PARQUET_GLOB))
    if not matches:
        raise FileNotFoundError(
            f"No {JOINED_HOTELS_PARQUET_GLOB} under {processed}. "
            f"Run ``scripts/pipeline/run_cleaning.py`` or place the joined Parquet there."
        )
    return matches[0]


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


def discover_raw_paths(
    keys: tuple[str, ...] | None = None,
) -> dict[str, Path]:
    """Return paths for requested datasets (default: hotels, world cities, crime)."""
    order = keys if keys is not None else DATASET_ORDER
    if is_kaggle_runtime():
        out: dict[str, Path] = {}
        missing: list[str] = []
        for key in order:
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
    for key in order:
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


# Aliases for clearer imports in notebooks (same behavior as the functions above).
joined_hotels_parquet_path = resolve_joined_hotels_parquet
raw_csv_paths = discover_raw_paths
