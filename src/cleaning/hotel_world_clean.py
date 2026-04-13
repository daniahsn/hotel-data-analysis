"""
Hotels + world cities cleaning (steps 1–6, no crime).

Functions are grouped by step; each step is a small, named unit you can call or test alone.
"""

from __future__ import annotations

import functools
import logging
import re
from collections.abc import Iterator
from contextlib import contextmanager
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


@contextmanager
def _silence_country_converter_warnings() -> Iterator[None]:
    """``country_converter`` logs ``X not found in name_short`` once per miss; silence during batch."""
    loggers = (
        logging.getLogger("country_converter.country_converter"),
        logging.getLogger("country_converter"),
    )
    previous = [(lg, lg.level, lg.propagate) for lg in loggers]
    try:
        for lg in loggers:
            lg.setLevel(logging.CRITICAL)
            lg.propagate = False
        yield
    finally:
        for lg, level, prop in previous:
            lg.setLevel(level)
            lg.propagate = prop


def _batch_country_convert(names: list[str], *, src: str, to: str = "ISO2") -> list[Any]:
    cc = _country_converter()
    kwargs: dict[str, Any] = {"src": src, "to": to}
    with _silence_country_converter_warnings():
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
    """
    Prefer ``countyCode``; fill gaps from ``countyName`` only where the code is missing.

    The hotels ``countyName`` field often holds city/region text (e.g. ``ANTIGUA``), not
    ISO country names. Running name matching on every row floods logs with ``not found``.
    """
    from_codes = iso2_from_codes(county_code)
    need_name = from_codes.isna()
    if not need_name.any():
        return from_codes
    from_names = iso2_from_country_names(county_name.loc[need_name])
    out = from_codes.copy()
    out.loc[need_name] = from_names.to_numpy()
    return out


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

# Kaggle / local CSVs often differ by case, spaces, or spelling vs our first sample.
_CANONICAL_HOTEL_COLUMNS: dict[str, str] = {
    "countycode": "countyCode",
    "countyname": "countyName",
    "countryname": "countyName",
    "country": "countyName",
    "citycode": "cityCode",
    "cityname": "cityName",
    "hotelcode": "HotelCode",
    "hotelname": "HotelName",
    "hotelrating": "HotelRating",
    "address": "Address",
    "attractions": "Attractions",
    "description": "Description",
    "faxnumber": "FaxNumber",
    "hotelfacilities": "HotelFacilities",
    "map": "Map",
    "phonenumber": "PhoneNumber",
    "pincode": "PinCode",
    "hotelwebsiteurl": "HotelWebsiteUrl",
}


def _normalize_hotel_header_key(name: str) -> str:
    return _strip_hotel_column_name(name).lower().replace(" ", "").replace("_", "")


def _strip_hotel_column_name(name: str) -> str:
    """
    Strip whitespace and BOM markers so the same file parses the same on Kaggle vs local.

    UTF-8 BOM read as latin-1 becomes the three characters ``\\xef\\xbb\\xbf`` prefixing
    the first header; that breaks lookups for ``countyCode`` even when the dataset is identical.
    """
    s = str(name).strip()
    while s.startswith("\ufeff"):
        s = s[1:].lstrip()
    if s.startswith("\xef\xbb\xbf"):
        s = s[3:].lstrip()
    return s.strip()


def standardize_hotel_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Strip headers, map aliases to the names used in ``clean_hotels`` (TBO / Kaggle variants).

    ``countyName`` and ``Attractions`` are optional; if absent they are added as NA so
    country codes and attraction counts still run.
    """
    h = df.copy()
    h.columns = [_strip_hotel_column_name(c) for c in h.columns]
    renames = {}
    for c in h.columns:
        k = _normalize_hotel_header_key(c)
        if k in _CANONICAL_HOTEL_COLUMNS:
            tgt = _CANONICAL_HOTEL_COLUMNS[k]
            if c != tgt:
                renames[c] = tgt
    h = h.rename(columns=renames)
    if h.columns.duplicated().any():
        h = h.loc[:, ~h.columns.duplicated(keep="first")].copy()
    if "countyName" not in h.columns:
        h["countyName"] = pd.NA
    if "Attractions" not in h.columns:
        h["Attractions"] = pd.NA
    required = ("countyCode", "cityName", "HotelRating", "Map")
    missing = [c for c in required if c not in h.columns]
    if missing:
        raise KeyError(
            f"Hotels CSV missing columns {missing} after header normalization. "
            f"Columns present: {list(h.columns)}"
        )
    return h


def clean_hotels(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add: ``country_iso2``, ``city_join_key``, ``hotel_star_rating``,
    ``hotel_latitude``, ``hotel_longitude``, ``attractions_count``.
    """
    out = standardize_hotel_columns(df)
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


def read_hotels_csv(path: Path, *, nrows: int | None = None, encoding: str | None = None) -> pd.DataFrame:
    """
    Read hotels CSV. If ``encoding`` is None, try ``utf-8-sig`` then ``latin-1`` (common Kaggle vs local).

    Same underlying file can fail in one environment if only one encoding is hard-coded.
    """
    read_kw: dict[str, Any] = {"low_memory": False, "skipinitialspace": True}
    if nrows is not None:
        read_kw["nrows"] = nrows
    if encoding is not None:
        return pd.read_csv(path, encoding=encoding, **read_kw)
    for enc in ("utf-8-sig", "latin-1"):
        try:
            return pd.read_csv(path, encoding=enc, **read_kw)
        except UnicodeDecodeError:
            continue
    return pd.read_csv(path, encoding="latin-1", **read_kw)


def run_cleaning_pipeline(
    *,
    hotels_path: Path | None = None,
    world_path: Path | None = None,
    output_dir: Path,
    hotels_sample_rows: int | None = None,
    also_join: bool = True,
    output_format: Literal["parquet", "csv"] = "parquet",
    hotels_encoding: str | None = None,
    world_encoding: str = "utf-8",
) -> dict[str, Path]:
    """Load raw CSVs, run ``clean_hotels`` / ``clean_world_cities``, optional join, write outputs."""
    paths = discover_raw_paths(keys=HOTELS_WORLD_KEYS)
    hp = hotels_path or paths["hotels"]
    wp = world_path or paths["world"]
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    hotels_raw = read_hotels_csv(hp, nrows=hotels_sample_rows, encoding=hotels_encoding)
    world_raw = pd.read_csv(wp, encoding=world_encoding, low_memory=False, skipinitialspace=True)

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
