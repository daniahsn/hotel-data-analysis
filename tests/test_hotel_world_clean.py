"""Fast local tests: tiny in-memory frames only (no large CSVs on disk)."""

from __future__ import annotations

import pandas as pd
import pytest

pytest.importorskip("country_converter")

from src.cleaning.hotel_world_clean import (
    attractions_count,
    city_join_key,
    clean_hotels,
    clean_world_cities,
    hotel_star_rating_numeric,
    iso2_for_hotels,
    iso2_from_codes,
    parse_hotel_map_lat_lon,
    standardize_hotel_columns,
)


def test_standardize_hotel_columns_strips_header_spaces() -> None:
    raw = pd.DataFrame([["AL", "Albania", " Tirana ", "FourStar", "1|2", ""]])
    raw.columns = ["countyCode", " countyName", " cityName", "HotelRating", "Map", "Attractions"]
    out = standardize_hotel_columns(raw)
    assert list(out.columns) == [
        "countyCode",
        "countyName",
        "cityName",
        "HotelRating",
        "Map",
        "Attractions",
    ]


def test_iso2_for_hotels_uses_code_only_when_present() -> None:
    """Noisy countyName must not override a valid countyCode."""
    codes = pd.Series(["AG", "AL"])
    names = pd.Series(["ANTIGUA", "Albania"])
    out = iso2_for_hotels(codes, names)
    assert out.iloc[0] == "AG"
    assert out.iloc[1] == "AL"


def test_iso2_from_codes() -> None:
    s = pd.Series(["al", "XX", pd.NA])
    out = iso2_from_codes(s)
    assert out.iloc[0] == "AL"
    assert pd.isna(out.iloc[1])
    assert pd.isna(out.iloc[2])


def test_city_join_key() -> None:
    s = pd.Series(["  New York! ", "São Paulo"])
    out = city_join_key(s)
    assert "new york" in out.iloc[0]
    assert out.iloc[1].startswith("so")


def test_hotel_star_rating_numeric() -> None:
    s = pd.Series(["FourStar", "3 Star", "nope", pd.NA])
    out = hotel_star_rating_numeric(s)
    assert out.iloc[0] == 4
    assert out.iloc[1] == 3
    assert pd.isna(out.iloc[2])
    assert pd.isna(out.iloc[3])


def test_parse_hotel_map_lat_lon() -> None:
    s = pd.Series(["41.32|19.81", "bad", pd.NA])
    out = parse_hotel_map_lat_lon(s)
    assert out.loc[0, "hotel_latitude"] == pytest.approx(41.32)
    assert out.loc[0, "hotel_longitude"] == pytest.approx(19.81)
    assert pd.isna(out.loc[1, "hotel_latitude"])
    assert pd.isna(out.loc[2, "hotel_latitude"])


def test_attractions_count() -> None:
    s = pd.Series(["a, b", "", pd.NA, "single"])
    out = attractions_count(s)
    assert out.tolist() == [2, 0, 0, 1]


def test_clean_hotels_minimal_row() -> None:
    df = pd.DataFrame(
        [
            {
                "countyCode": "AL",
                "countyName": "Albania",
                "cityName": "Tirana",
                "HotelRating": "ThreeStar",
                "Map": "41.3|19.8",
                "Attractions": "Museum, Park",
            }
        ]
    )
    out = clean_hotels(df)
    assert out["country_iso2"].iloc[0] == "AL"
    assert out["city_join_key"].iloc[0] == "tirana"
    assert out["hotel_star_rating"].iloc[0] == 3
    assert out["attractions_count"].iloc[0] == 2


def test_clean_world_cities_dedupes_by_population() -> None:
    df = pd.DataFrame(
        [
            {"Country": "ad", "City": "foo", "Region": "1", "Population": "10", "Latitude": 1.0, "Longitude": 2.0},
            {"Country": "ad", "City": "foo", "Region": "2", "Population": "99", "Latitude": 3.0, "Longitude": 4.0},
        ]
    )
    out = clean_world_cities(df)
    assert len(out) == 1
    assert int(out["Population"].iloc[0]) == 99
