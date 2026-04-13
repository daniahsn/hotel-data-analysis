"""Fast local tests: tiny in-memory frames only (no large CSVs on disk)."""

from __future__ import annotations

import pandas as pd
import pytest

from src.cleaning.hotel_world_clean import (
    city_join_key,
    clean_hotels,
    clean_world_cities,
    finalize_joined_hotels,
    hotel_star_rating_numeric,
    iso2_for_hotels,
    iso2_from_codes,
    iso2_from_country_names,
    join_hotels_world_cities,
    parse_hotel_map_lat_lon,
    read_hotels_csv,
    run_cleaning_pipeline,
    standardize_hotel_columns,
)
from src.features.hotel_text_features import (
    attractions_count,
    facilities_keyword_hits,
    facilities_token_count,
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
        "HotelFacilities",
        "cityCode",
        "HotelCode",
        "HotelName",
    ]


def test_iso2_for_hotels_uses_code_only_when_present() -> None:
    """Noisy countyName must not override a valid countyCode."""
    codes = pd.Series(["AG", "AL"])
    names = pd.Series(["ANTIGUA", "Albania"])
    out = iso2_for_hotels(codes, names)
    assert out.iloc[0] == "AG"
    assert out.iloc[1] == "AL"


def test_iso2_from_codes() -> None:
    # "XX" is two letters; country_converter may echo it and our post-check accepts
    # any A–Z pair — use a non-ISO shape instead for "invalid → NA".
    s = pd.Series(["al", "BAD", pd.NA])
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
    # Unstructured single word (no commas / distances) → 0, not a fake "1"
    assert out.tolist() == [2, 0, 0, 0]


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


def test_iso2_from_country_names_germany() -> None:
    out = iso2_from_country_names(pd.Series(["Germany"]))
    assert out.iloc[0] == "DE"


def test_iso2_for_hotels_fills_from_name_when_code_missing() -> None:
    codes = pd.Series([pd.NA, ""])
    names = pd.Series(["Germany", "France"])
    out = iso2_for_hotels(codes, names)
    assert out.iloc[0] == "DE"
    assert out.iloc[1] == "FR"


def test_standardize_hotel_columns_adds_missing_county_name() -> None:
    raw = pd.DataFrame([["US", "NYC", "FourStar", "1|1", ""]])
    raw.columns = ["countyCode", "cityName", "HotelRating", "Map", "Attractions"]
    out = standardize_hotel_columns(raw)
    assert "countyName" in out.columns
    assert pd.isna(out["countyName"].iloc[0])


def test_standardize_hotel_columns_pascal_case_aliases() -> None:
    raw = pd.DataFrame([["US", "United States", "NYC", "FourStar", "1|1", ""]])
    raw.columns = ["CountyCode", "CountyName", "CityName", "HotelRating", "Map", "Attractions"]
    out = standardize_hotel_columns(raw)
    assert out["countyCode"].iloc[0] == "US"
    assert out["countyName"].iloc[0] == "United States"


def test_parse_hotel_map_rejects_out_of_range_latitude() -> None:
    out = parse_hotel_map_lat_lon(pd.Series(["95|10"]))
    assert pd.isna(out["hotel_latitude"].iloc[0])
    assert pd.isna(out["hotel_longitude"].iloc[0])


def test_parse_hotel_map_accepts_boundary_lat_lon() -> None:
    out = parse_hotel_map_lat_lon(pd.Series(["90|180"]))
    assert out["hotel_latitude"].iloc[0] == 90
    assert out["hotel_longitude"].iloc[0] == 180


def test_hotel_star_rating_numeric_five_two_one() -> None:
    out = hotel_star_rating_numeric(pd.Series(["FiveStar", "TwoStar", "OneStar"]))
    assert out.tolist() == [5, 2, 1]


def test_city_join_key_empty_string() -> None:
    assert city_join_key(pd.Series([""])).iloc[0] == ""


def test_attractions_count_skips_empty_segments() -> None:
    assert attractions_count(pd.Series(["a,,b"])).iloc[0] == 2


def test_attractions_count_booking_style_html() -> None:
    html = (
        "Distances are displayed to the nearest 0.1 mile and kilometer. <br /> "
        "<p>Ministry of Justice - 1.8 km / 1.1 mi <br /> Palace of Culture - 1.8 km / 1.1 mi <br /> "
        "Skanderbeg Square - 1.8 km / 1.1 mi <br /> National Bank of Albania - 2.1 km / 1.3 mi <br /> "
        "</p><p>The preferred airport for Hotel Victoria is Nene Tereza Intl. Airport (TIA) - 18.3 km / 11.4 mi </p>"
    )
    assert attractions_count(pd.Series([html])).iloc[0] == 4


def test_attractions_count_within_metre_prose() -> None:
    prose = (
        "Berat : within 130000 metre  Tower of the city: within 3000 metre  "
        "Old Amfitheatre : within 3000 metre  Residence of Ex King of Albania: within 4000 metre  "
        "National Historic Museum: within 35000 metre"
    )
    assert attractions_count(pd.Series([prose])).iloc[0] == 5


def test_facilities_token_count_splits_commas_and_slashes() -> None:
    s = "Free WiFi  Parking onsite  Airport shuttle /  Laundry"
    assert facilities_token_count(pd.Series([s])).iloc[0] > 4


def test_facilities_keyword_hits_counts_amenities() -> None:
    s = "Free WiFi and private parking with 24-hour front desk and room service"
    hits = facilities_keyword_hits(pd.Series([s])).iloc[0]
    assert hits >= 4


def test_clean_hotels_adds_facility_features() -> None:
    df = pd.DataFrame(
        [
            {
                "countyCode": "AL",
                "countyName": "Albania",
                "cityName": "Tirana",
                "HotelRating": "ThreeStar",
                "Map": "41.3|19.8",
                "Attractions": "Museum, Park",
                "HotelFacilities": "Free WiFi  Private parking  Laundry facilities",
            }
        ]
    )
    out = clean_hotels(df)
    assert out["facilities_token_count"].iloc[0] >= 3
    assert out["facilities_keyword_hits"].iloc[0] >= 2


def test_clean_world_cities_keeps_distinct_city_keys() -> None:
    df = pd.DataFrame(
        [
            {"Country": "us", "City": "a", "Region": "1", "Population": "1", "Latitude": 0.0, "Longitude": 0.0},
            {"Country": "us", "City": "b", "Region": "2", "Population": "2", "Latitude": 1.0, "Longitude": 1.0},
        ]
    )
    assert len(clean_world_cities(df)) == 2


def test_join_hotels_world_cities_attaches_city_features() -> None:
    hotel = pd.DataFrame(
        [
            {
                "countyCode": "US",
                "countyName": "United States",
                "cityCode": "1",
                "cityName": "New York",
                "HotelCode": "100",
                "HotelName": "Test Inn",
                "HotelRating": "FourStar",
                "Map": "40.7|-74.0",
                "Attractions": "",
                "HotelFacilities": "WiFi",
            }
        ]
    )
    world = pd.DataFrame(
        [
            {
                "Country": "us",
                "City": "new york",
                "Region": "NY",
                "Population": "8800000",
                "Latitude": 40.71,
                "Longitude": -74.01,
            }
        ]
    )
    hc = clean_hotels(hotel)
    wc = clean_world_cities(world)
    joined = join_hotels_world_cities(hc, wc)
    assert int(joined["city_population"].iloc[0]) == 8_800_000
    final = finalize_joined_hotels(joined)
    assert list(final.columns) == [
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
    ]


def test_read_hotels_csv_strips_space_after_comma(tmp_path) -> None:
    p = tmp_path / "hotels.csv"
    p.write_text("countyCode, countyName, cityName, HotelRating, Map, Attractions\nAL,Albania,Tirana,FourStar,1|2,\n", encoding="utf-8")
    df = read_hotels_csv(p, nrows=5)
    assert "countyName" in df.columns
    assert "cityName" in df.columns
    assert clean_hotels(df)["country_iso2"].iloc[0] == "AL"


def test_run_cleaning_pipeline_writes_outputs(tmp_path) -> None:
    hp = tmp_path / "hotels.csv"
    hp.write_text(
        "countyCode,countyName,cityCode,cityName,HotelCode,HotelName,HotelRating,Map,Attractions,HotelFacilities\n"
        "DE,Germany,1,Berlin,9,Boutique,ThreeStar,52.5|13.4,Museum,WiFi\n",
        encoding="utf-8",
    )
    wp = tmp_path / "world.csv"
    wp.write_text(
        "Country,City,AccentCity,Region,Population,Latitude,Longitude\n"
        "de,berlin,Berlin,11,3700000,52.5,13.4\n",
        encoding="utf-8",
    )
    out_dir = tmp_path / "out"
    written = run_cleaning_pipeline(
        hotels_path=hp,
        world_path=wp,
        output_dir=out_dir,
        also_join=True,
        output_format="csv",
    )
    assert written["hotels_clean"].is_file()
    assert written["world_cities_clean"].is_file()
    assert written["hotels_with_cities"].is_file()
    merged = pd.read_csv(written["hotels_with_cities"])
    assert list(merged.columns) == [
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
    ]
