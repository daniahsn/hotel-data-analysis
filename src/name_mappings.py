"""
Canonical names and rename rules for this project.

Keep **all** column/header alias logic here so cleaning, Parquet I/O, and modeling share one source of truth.
Path location logic stays in ``src.raw_data_paths``.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Hotels CSV: normalized header key → canonical column name
# (normalized key = lowercase, no spaces/underscores; see ``hotel_world_clean`` helpers)
# ---------------------------------------------------------------------------
HOTEL_HEADER_TO_CANONICAL: dict[str, str] = {
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

# Subset of columns read for the cleaning pipeline (wide exports can drop the rest at read time).
HOTEL_COLUMNS_FOR_CLEANING: frozenset[str] = frozenset(
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

# World-cities CSV: normalized header keys kept for ``clean_world_cities`` minimal read.
WORLD_CSV_HEADER_KEYS: frozenset[str] = frozenset({"country", "city", "population"})

# ---------------------------------------------------------------------------
# Joined modeling Parquet: alternate column labels → canonical name
# ---------------------------------------------------------------------------
JOINED_PARQUET_COLUMN_ALIASES: dict[str, str] = {
    "facilities_keywords_count": "facilities_keyword_hits",
    "feature_count": "facilities_count",
    "Feature_keywords": "facilities_keyword_hits",
}

# Required columns after ``JOINED_PARQUET_COLUMN_ALIASES`` are applied.
JOINED_TABLE_REQUIRED_COLUMNS: tuple[str, ...] = (
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

# Output file stem for the joined table (``hotels_with_cities.parquet``, etc.).
JOINED_HOTELS_FILE_STEM: str = "hotels_with_cities"
