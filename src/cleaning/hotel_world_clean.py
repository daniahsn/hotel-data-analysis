"""
Hotels + world cities cleaning (steps 1–6, no crime).

Functions are grouped by step; each step is a small, named unit you can call or test alone.
"""

from __future__ import annotations

import functools
import re
from pathlib import Path
from typing import Any, Literal

import country_converter as coco
import pandas as pd

from src.raw_data_paths import discover_raw_paths

# ---------------------------------------------------------------------------
# Step 1 — country → ISO2
# ---------------------------------------------------------------------------


@functools.lru_cache(maxsize=1)
def _country_converter() -> coco.CountryConverter:
    return coco.CountryConverter()


def _as_clean_iso2_list(values: pd.Series) -> list[str]:
    out: list[str] = []
    for v in values.tolist():
        if pd.isna(v):
            out.append("")
            continue
        s = str(v).strip()
        if not s or s.lower() == "nan":
            out.append("")
        else:
            out.append(s.upper())
    return out


def _batch_country_convert(names: list[str], *, src: str, to: str = "ISO2") -> list[Any]:
    cc = _country_converter()
    kwargs: dict[str, Any] = {"src": src, "to": to}
    try:
        return cc.convert(names, **kwargs, not_found=None)  # type: ignore[call-arg]
    except TypeError:
        return cc.convert(names, **kwargs)


def _iso2_series_from_batch(original: pd.Series, raw: list[Any] | Any) -> pd.Series:
    if not isinstance(raw, list):
        raw = [raw]
    cleaned: list[Any] = []
    for v in raw:
        if v is None or (isinstance(v, float) and pd.isna(v)):
            cleaned.append(pd.NA)
            continue
        s = str(v).strip().upper()
        if len(s) != 2 or not s.isalpha():
            cleaned.append(pd.NA)
        else:
            cleaned.append(s)
    return pd.Series(cleaned, index=original.index, dtype="string")


def iso2_from_codes(codes: pd.Series) -> pd.Series:
    """Map 2-letter codes to validated ISO2 (uppercase). Unknown → NA."""
    names = _as_clean_iso2_list(codes)
    raw = _batch_country_convert(names, src="ISO2")
    return _iso2_series_from_batch(codes, raw)


def iso2_from_country_names(names: pd.Series) -> pd.Series:
    """Map country names (e.g. ``Albania``) to ISO2 via ``name_short``. Unknown → NA."""
    clean_list = _as_clean_iso2_list(names)
    raw = _batch_country_convert(clean_list, src="name_short")
    return _iso2_series_from_batch(names, raw)


def iso2_for_hotels(county_code: pd.Series, county_name: pd.Series) -> pd.Series:
    """Prefer ``countyCode``; fill gaps from ``countyName``."""
    return iso2_from_codes(county_code).fillna(iso2_from_country_names(county_name))


# ---------------------------------------------------------------------------
# Step 2 — city text → join key
# ---------------------------------------------------------------------------


def city_join_key(series: pd.Series) -> pd.Series:
    """Lowercase, strip, collapse spaces, strip punctuation (keep letters, digits, hyphen)."""
    s = series.fillna("").astype(str).str.strip().str.lower()
    s = s.str.replace(r"\s+", " ", regex=True)
    s = s.str.replace(r"[^a-z0-9\s-]", "", regex=True)
    s = s.str.replace(r"\s+", " ", regex=True).str.strip()
    return s.astype("string")


# ---------------------------------------------------------------------------
# Step 3 — hotel star label → numeric target
# ---------------------------------------------------------------------------

_WORD_TO_STAR: dict[str, int] = {
    "fivestar": 5,
    "fourstar": 4,
    "threestar": 3,
    "twostar": 2,
    "onestar": 1,
}


def _parse_one_star(raw: object) -> int | None:
    if raw is None or (isinstance(raw, float) and pd.isna(raw)):
        return None
    s = str(raw).strip()
    if not s or s.lower() == "nan":
        return None
    key = re.sub(r"\s+", "", s).lower()
    if key in _WORD_TO_STAR:
        return _WORD_TO_STAR[key]
    m = re.search(r"(\d)\s*star", s, flags=re.IGNORECASE)
    if m:
        n = int(m.group(1))
        if 1 <= n <= 5:
            return n
    return None


def hotel_star_rating_numeric(rating: pd.Series) -> pd.Series:
    """``FourStar`` / ``3 Star`` → 1–5; unrecognized → NA."""
    out: list[Any] = []
    for raw in rating.tolist():
        val = _parse_one_star(raw)
        out.append(val if val is not None else pd.NA)
    return pd.Series(out, index=rating.index, dtype="Int64")


# ---------------------------------------------------------------------------
# Step 4 — world cities: dedupe one row per (country_iso2, city_join_key)
# (applied inside ``clean_world_cities`` below)
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Step 5 — hotel ``Map`` column → lat/lon
# ---------------------------------------------------------------------------


def parse_hotel_map_lat_lon(map_series: pd.Series) -> pd.DataFrame:
    """Split ``lat|lon`` into ``hotel_latitude`` / ``hotel_longitude``; invalid → NA."""
    s = map_series.fillna("").astype(str).str.strip()
    parts = s.str.split("|", n=1, expand=True)
    if parts.shape[1] == 1:
        parts = parts.copy()
        parts[1] = pd.NA
    lat = pd.to_numeric(parts[0], errors="coerce")
    lon = pd.to_numeric(parts[1], errors="coerce")
    valid = lat.between(-90, 90) & lon.between(-180, 180)
    lat = lat.where(valid, pd.NA)
    lon = lon.where(valid, pd.NA)
    return pd.DataFrame({"hotel_latitude": lat, "hotel_longitude": lon}, index=map_series.index)


# ---------------------------------------------------------------------------
# Step 6 — attractions text → count
# ---------------------------------------------------------------------------


def attractions_count(series: pd.Series) -> pd.Series:
    """Comma-separated entries; empty / missing → 0; one blob without commas → 1."""
    s = series.fillna("").astype(str).str.strip()
    empty = s.eq("") | s.str.lower().eq("nan")
    parts = s.str.split(",")
    counts = parts.apply(lambda xs: sum(1 for x in xs if str(x).strip() != ""))
    return counts.where(~empty, 0).astype("Int64")


# ---------------------------------------------------------------------------
# Full-table helpers
# ---------------------------------------------------------------------------


def clean_hotels(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add: ``country_iso2``, ``city_join_key``, ``hotel_star_rating``,
    ``hotel_latitude``, ``hotel_longitude``, ``attractions_count``.
    """
    out = df.copy()
    out["country_iso2"] = iso2_for_hotels(out["countyCode"], out["countyName"])
    out["city_join_key"] = city_join_key(out["cityName"])
    out["hotel_star_rating"] = hotel_star_rating_numeric(out["HotelRating"])
    geo = parse_hotel_map_lat_lon(out["Map"])
    out["hotel_latitude"] = geo["hotel_latitude"]
    out["hotel_longitude"] = geo["hotel_longitude"]
    out["attractions_count"] = attractions_count(out["Attractions"])
    return out


def clean_world_cities(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add ``country_iso2``, ``city_join_key``; keep one row per key (max ``Population`` first).
    """
    out = df.copy()
    codes = out["Country"].astype(str).str.strip().str.upper()
    out["country_iso2"] = iso2_from_codes(codes)
    out["city_join_key"] = city_join_key(out["City"])
    pop = pd.to_numeric(out["Population"], errors="coerce")
    out = out.assign(_pop_sort=pop)
    out = out.sort_values("_pop_sort", ascending=False, na_position="last")
    out = out.drop_duplicates(subset=["country_iso2", "city_join_key"], keep="first")
    return out.drop(columns=["_pop_sort"])


def join_hotels_world_cities(hotels_clean: pd.DataFrame, world_clean: pd.DataFrame) -> pd.DataFrame:
    """Left join on ``country_iso2`` + ``city_join_key``; adds city_region / population / gazetteer lat-lon."""
    right = world_clean.rename(
        columns={
            "Region": "city_region",
            "Population": "city_population",
            "Latitude": "city_gazetteer_latitude",
            "Longitude": "city_gazetteer_longitude",
        }
    )
    keys = ["country_iso2", "city_join_key"]
    use = keys + [
        "city_region",
        "city_population",
        "city_gazetteer_latitude",
        "city_gazetteer_longitude",
    ]
    use = [c for c in use if c in right.columns]
    return hotels_clean.merge(right[use], on=keys, how="left")


HOTELS_WORLD_KEYS: tuple[str, ...] = ("hotels", "world")


def run_cleaning_pipeline(
    *,
    hotels_path: Path | None = None,
    world_path: Path | None = None,
    output_dir: Path,
    hotels_sample_rows: int | None = None,
    also_join: bool = True,
    output_format: Literal["parquet", "csv"] = "parquet",
    hotels_encoding: str = "latin-1",
    world_encoding: str = "utf-8",
) -> dict[str, Path]:
    """Load raw CSVs, run ``clean_hotels`` / ``clean_world_cities``, optional join, write outputs."""
    paths = discover_raw_paths(keys=HOTELS_WORLD_KEYS)
    hp = hotels_path or paths["hotels"]
    wp = world_path or paths["world"]
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    read_kw: dict[str, Any] = {"low_memory": False}
    if hotels_sample_rows is not None:
        read_kw["nrows"] = hotels_sample_rows

    hotels_raw = pd.read_csv(hp, encoding=hotels_encoding, **read_kw)
    world_raw = pd.read_csv(wp, encoding=world_encoding, low_memory=False)

    hotels_c = clean_hotels(hotels_raw)
    world_c = clean_world_cities(world_raw)

    written: dict[str, Path] = {}

    def _write(name: str, frame: pd.DataFrame) -> Path:
        path = output_dir / f"{name}.{output_format}"
        if output_format == "parquet":
            frame.to_parquet(path, index=False)
        else:
            frame.to_csv(path, index=False)
        return path

    written["hotels_clean"] = _write("hotels_clean", hotels_c)
    written["world_cities_clean"] = _write("world_cities_clean", world_c)
    if also_join:
        written["hotels_with_cities"] = _write("hotels_with_cities", join_hotels_world_cities(hotels_c, world_c))
    return written
