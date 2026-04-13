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
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import pandas as pd

from src.features.hotel_text_features import (
    attractions_count,
    facilities_keyword_hits,
    facilities_token_count,
)
from src.raw_data_paths import discover_raw_paths

# ---------------------------------------------------------------------------
# Hotel CSV header normalization (shared by ``read_hotels_csv`` usecols + ``standardize_hotel_columns``)
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

# Columns required to run ``clean_hotels`` (everything else in wide exports can be dropped at read time).
_HOTEL_COLUMNS_FOR_CLEANING: frozenset[str] = frozenset(
    {
        "countyCode",
        "countyName",
        "cityCode",
        "cityName",
        "HotelCode",
        "HotelName",
        "HotelRating",
        "Map",
        "Attractions",
        "HotelFacilities",
    }
)


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


def _normalize_hotel_header_key(name: str) -> str:
    return _strip_hotel_column_name(name).lower().replace(" ", "").replace("_", "")


def _hotel_csv_usecols_keep(col: object) -> bool:
    k = _normalize_hotel_header_key(str(col))
    tgt = _CANONICAL_HOTEL_COLUMNS.get(k)
    return tgt is not None and tgt in _HOTEL_COLUMNS_FOR_CLEANING


def _world_csv_usecols_keep(col: object) -> bool:
    """Keep only gazetteer fields needed for ``clean_world_cities`` (drops Region, lat/lon, …)."""
    k = _normalize_hotel_header_key(str(col))
    return k in frozenset({"country", "city", "population"})


if TYPE_CHECKING:
    from country_converter import CountryConverter

# ---------------------------------------------------------------------------
# Step 1 — country → ISO2
# ---------------------------------------------------------------------------


@functools.lru_cache(maxsize=1)
def _country_converter() -> CountryConverter:
    """Lazy import: ``country_converter`` is heavy; avoid loading it for geo/rating-only code paths."""
    import country_converter as coco

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


def hotel_star_rating_numeric(rating: pd.Series) -> pd.Series:
    """``FourStar`` / ``3 Star`` → 1–5; unrecognized → NA."""
    mask_na = rating.isna()
    s = rating.astype(str).str.strip()
    invalid = s.str.lower().isin(("", "nan"))
    squish = s.str.replace(r"\s+", "", regex=True).str.lower()
    mapped = squish.map(_WORD_TO_STAR)
    ext = s.str.extract(r"(\d)\s*star", expand=False, flags=re.I)
    num = pd.to_numeric(ext, errors="coerce")
    from_regex = num.where(num.between(1, 5), pd.NA)
    combined = mapped.where(mapped.notna(), from_regex)
    combined = combined.where(~(mask_na | invalid), pd.NA)
    return combined.astype("Int64")


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
# Full-table helpers
# ---------------------------------------------------------------------------


def standardize_hotel_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Strip headers, map aliases to the names used in ``clean_hotels`` (TBO / Kaggle variants).

    ``countyName``, ``Attractions``, and ``HotelFacilities`` are optional; if absent they are
    added as NA so downstream features still run.
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
    if "HotelFacilities" not in h.columns:
        h["HotelFacilities"] = pd.NA
    for opt in ("cityCode", "HotelCode", "HotelName"):
        if opt not in h.columns:
            h[opt] = pd.NA
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
    ``hotel_latitude``, ``hotel_longitude``, ``attractions_count``,
    ``facilities_token_count``, ``facilities_keyword_hits`` (from ``src.features``).
    """
    out = standardize_hotel_columns(df)
    out["country_iso2"] = iso2_for_hotels(out["countyCode"], out["countyName"])
    out["city_join_key"] = city_join_key(out["cityName"])
    out["hotel_star_rating"] = hotel_star_rating_numeric(out["HotelRating"])
    geo = parse_hotel_map_lat_lon(out["Map"])
    out["hotel_latitude"] = geo["hotel_latitude"]
    out["hotel_longitude"] = geo["hotel_longitude"]
    out["attractions_count"] = attractions_count(out["Attractions"])
    out["facilities_token_count"] = facilities_token_count(out["HotelFacilities"])
    out["facilities_keyword_hits"] = facilities_keyword_hits(out["HotelFacilities"])
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
    """Left join on ``country_iso2`` + ``city_join_key``; attach ``city_population`` only."""
    left = hotels_clean.copy()
    right = world_clean.rename(columns={"Population": "city_population"})
    keys = ["country_iso2", "city_join_key"]
    use = [c for c in keys + ["city_population"] if c in right.columns]
    right_sub = right[use].copy()
    for k in keys:
        left[k] = left[k].astype("category")
        right_sub[k] = right_sub[k].astype("category")
    return left.merge(right_sub, on=keys, how="left")


# Columns written to ``hotels_with_cities`` (model-ready join). ``facilities_token_count`` → ``facilities_count``.
_FINAL_JOIN_SOURCE_COLS: tuple[str, ...] = (
    "countyCode",
    "countyName",
    "cityCode",
    "cityName",
    "HotelCode",
    "HotelName",
    "hotel_star_rating",
    "attractions_count",
    "facilities_token_count",
    "facilities_keyword_hits",
    "hotel_latitude",
    "hotel_longitude",
    "city_population",
)

_FINAL_JOIN_OUTPUT_COLS: tuple[str, ...] = (
    "countyCode",
    "countyName",
    "cityCode",
    "cityName",
    "HotelCode",
    "HotelName",
    "hotel_star_rating",
    "attractions_count",
    "facilities_count",
    "facilities_keyword_hits",
    "hotel_latitude",
    "hotel_longitude",
    "city_population",
)


def finalize_joined_hotels(joined: pd.DataFrame) -> pd.DataFrame:
    """
    Keep only the fields used for modeling from a ``join_hotels_world_cities`` frame.

    Drops join keys, raw text columns, gazetteer extras, etc. Renames
    ``facilities_token_count`` → ``facilities_count``; keeps ``facilities_keyword_hits``.
    """
    missing = [c for c in _FINAL_JOIN_SOURCE_COLS if c not in joined.columns]
    if missing:
        raise KeyError(
            f"finalize_joined_hotels: missing {missing}. Columns present: {list(joined.columns)}"
        )
    out = joined.loc[:, list(_FINAL_JOIN_SOURCE_COLS)].copy()
    out = out.rename(columns={"facilities_token_count": "facilities_count"})
    return out.reindex(columns=list(_FINAL_JOIN_OUTPUT_COLS))


HOTELS_WORLD_KEYS: tuple[str, ...] = ("hotels", "world")


def read_hotels_csv(
    path: Path,
    *,
    nrows: int | None = None,
    encoding: str | None = None,
    columns: Literal["cleaning", "all"] = "cleaning",
) -> pd.DataFrame:
    """
    Read hotels CSV. If ``encoding`` is None, try ``utf-8-sig`` then ``latin-1`` (common Kaggle vs local).

    ``columns="cleaning"`` skips wide text columns (e.g. ``Description``) not needed for ``clean_hotels``.
    Same underlying file can fail in one environment if only one encoding is hard-coded.
    """
    read_kw: dict[str, Any] = {"low_memory": False, "skipinitialspace": True}
    if nrows is not None:
        read_kw["nrows"] = nrows
    if columns == "cleaning":
        read_kw["usecols"] = _hotel_csv_usecols_keep
    if encoding is not None:
        return pd.read_csv(path, encoding=encoding, **read_kw)
    for enc in ("utf-8-sig", "latin-1"):
        try:
            return pd.read_csv(path, encoding=enc, **read_kw)
        except UnicodeDecodeError:
            continue
    return pd.read_csv(path, encoding="latin-1", **read_kw)


def read_world_cities_csv(
    path: Path,
    *,
    nrows: int | None = None,
    encoding: str = "utf-8",
    columns: Literal["cleaning", "all"] = "cleaning",
) -> pd.DataFrame:
    """Read world-cities CSV; ``columns='cleaning'`` keeps Country / City / Population only."""
    read_kw: dict[str, Any] = {"low_memory": False, "skipinitialspace": True}
    if nrows is not None:
        read_kw["nrows"] = nrows
    if columns == "cleaning":
        read_kw["usecols"] = _world_csv_usecols_keep
    return pd.read_csv(path, encoding=encoding, **read_kw)


def run_cleaning_pipeline(
    *,
    hotels_path: Path | None = None,
    world_path: Path | None = None,
    output_dir: Path,
    hotels_sample_rows: int | None = None,
    hotels_chunksize: int | None = None,
    progress_every_rows: int | None = None,
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

    world_raw = read_world_cities_csv(wp, encoding=world_encoding)
    world_c = clean_world_cities(world_raw)
    if hotels_chunksize is not None and hotels_sample_rows is not None:
        raise ValueError("Use either hotels_chunksize or hotels_sample_rows (not both).")

    if hotels_chunksize is None:
        hotels_raw = read_hotels_csv(hp, nrows=hotels_sample_rows, encoding=hotels_encoding)
        hotels_c = clean_hotels(hotels_raw)
    else:
        # Chunked mode: keep notebook output moving (Kaggle sessions disconnect less often).
        read_kw: dict[str, Any] = {
            "low_memory": False,
            "skipinitialspace": True,
            "usecols": _hotel_csv_usecols_keep,
            "chunksize": hotels_chunksize,
        }
        frames: list[pd.DataFrame] = []
        total = 0

        def _iter_chunks(enc: str) -> Iterator[pd.DataFrame]:
            yield from pd.read_csv(hp, encoding=enc, **read_kw)

        encs = (hotels_encoding,) if hotels_encoding is not None else ("utf-8-sig", "latin-1")
        last_err: Exception | None = None
        for enc in encs:
            try:
                for chunk in _iter_chunks(enc):
                    total += len(chunk)
                    frames.append(clean_hotels(chunk))
                    if progress_every_rows and (total % progress_every_rows) < hotels_chunksize:
                        print(f"[{datetime.now().isoformat(timespec='seconds')}] cleaned {total:,} hotel rows")
                last_err = None
                break
            except UnicodeDecodeError as e:
                last_err = e
                frames.clear()
                total = 0
                continue
        if last_err is not None:
            raise last_err
        hotels_c = pd.concat(frames, ignore_index=True) if frames else clean_hotels(pd.DataFrame())

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
        merged = join_hotels_world_cities(hotels_c, world_c)
        written["hotels_with_cities"] = _write("hotels_with_cities", finalize_joined_hotels(merged))
    return written
