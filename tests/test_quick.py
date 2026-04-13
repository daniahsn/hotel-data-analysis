"""Shortest smoke tests — no ``country_converter``. Run: ``pytest tests/test_quick.py -q``"""

from __future__ import annotations

import pandas as pd
import pytest

from src.cleaning.hotel_world_clean import (
    city_join_key,
    hotel_star_rating_numeric,
    parse_hotel_map_lat_lon,
)
from src.features.hotel_text_features import attractions_count, facilities_token_count


def test_quick_helpers() -> None:
    assert attractions_count(pd.Series(["a, b"])).tolist() == [2]
    assert facilities_token_count(pd.Series(["wifi, parking"])).iloc[0] == 2
    assert hotel_star_rating_numeric(pd.Series(["FourStar"])).iloc[0] == 4
    g = parse_hotel_map_lat_lon(pd.Series(["41.0|19.0"]))
    assert g["hotel_latitude"].iloc[0] == pytest.approx(41.0)
    assert "new york" in city_join_key(pd.Series([" New York! "])).iloc[0]
