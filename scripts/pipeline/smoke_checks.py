#!/usr/bin/env python3
"""
Fast smoke checks (no pytest): helpers, cleaning pipeline on tiny CSVs, feature matrix.

Run from repo root:
  python scripts/pipeline/smoke_checks.py
"""

from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import pandas as pd

from src.cleaning.hotel_world_clean import (
    city_join_key,
    hotel_star_rating_numeric,
    iso2_for_hotels,
    parse_hotel_map_lat_lon,
    run_cleaning_pipeline,
)
from src.name_mappings import JOINED_HOTELS_FILE_STEM
from src.features.hotel_text_features import attractions_count, facilities_token_count
from src.modeling.feature_matrix import (
    build_modeling_feature_matrices,
    load_joined_hotels,
    normalize_engineered_column_names,
    save_modeling_bundle,
)


def _approx(a: float, b: float, *, tol: float = 1e-5) -> bool:
    return abs(a - b) <= tol


def run_quick_smokes() -> None:
    """No ``country_converter`` — pure helpers."""
    assert attractions_count(pd.Series(["a, b"])).tolist() == [2]
    assert facilities_token_count(pd.Series(["wifi, parking"])).iloc[0] == 2
    assert hotel_star_rating_numeric(pd.Series(["FourStar"])).iloc[0] == 4
    g = parse_hotel_map_lat_lon(pd.Series(["41.0|19.0"]))
    assert _approx(float(g["hotel_latitude"].iloc[0]), 41.0)
    assert "new york" in city_join_key(pd.Series([" New York! "])).iloc[0]
    # ISO2: code wins over noisy name
    out = iso2_for_hotels(pd.Series(["AG", "AL"]), pd.Series(["ANTIGUA", "Albania"]))
    assert str(out.iloc[0]) == "AG"
    assert str(out.iloc[1]) == "AL"


def run_cleaning_pipeline_smoke(work: Path) -> None:
    hp = work / "hotels.csv"
    hp.write_text(
        "countyCode,countyName,cityCode,cityName,HotelCode,HotelName,HotelRating,Map,Attractions,HotelFacilities\n"
        "DE,Germany,1,Berlin,9,Boutique,ThreeStar,52.5|13.4,Museum,WiFi\n",
        encoding="utf-8",
    )
    wp = work / "world.csv"
    wp.write_text(
        "Country,City,AccentCity,Region,Population,Latitude,Longitude\n"
        "de,berlin,Berlin,11,3700000,52.5,13.4\n",
        encoding="utf-8",
    )
    out_dir = work / "processed"
    written = run_cleaning_pipeline(
        hotels_path=hp,
        world_path=wp,
        output_dir=out_dir,
        also_join=True,
        output_format="csv",
    )
    assert written["hotels_clean"].is_file()
    assert written["world_cities_clean"].is_file()
    assert written[JOINED_HOTELS_FILE_STEM].is_file()
    merged = pd.read_csv(written[JOINED_HOTELS_FILE_STEM])
    assert merged["hotel_star_rating"].iloc[0] == 3
    assert "city_population" in merged.columns


def _tiny_joined_frame(*, n: int = 24) -> pd.DataFrame:
    rng = range(n)
    return pd.DataFrame(
        {
            "countyCode": ["US", "AL"] * (n // 2) + (["US"] if n % 2 else []),
            "countyName": ["United States", "Albania"] * (n // 2) + (["United States"] if n % 2 else []),
            "cityCode": [f"{i % 5}" for i in rng],
            "cityName": [f"City {i % 4}" for i in rng],
            "HotelCode": [100 + i for i in rng],
            "HotelName": [f"Inn {i}" for i in rng],
            "hotel_star_rating": pd.Series([3 + (i % 3) for i in rng], dtype="Int64"),
            "attractions_count": [i % 4 for i in rng],
            "facilities_count": [1 + (i % 5) for i in rng],
            "facilities_keyword_hits": [i % 3 for i in rng],
            "hotel_latitude": [40.0 + 0.01 * i for i in rng],
            "hotel_longitude": [-74.0 - 0.01 * i for i in rng],
            "city_population": [1_000_000 - i * 1000 for i in rng],
        }
    )


def run_modeling_smokes(work: Path) -> None:
    try:
        import sklearn  # noqa: F401
    except ImportError as e:
        raise SystemExit(
            "scikit-learn is required for modeling smokes. Install: pip install -r requirements.txt"
        ) from e

    df = _tiny_joined_frame(n=4).rename(
        columns={
            "facilities_count": "feature_count",
            "facilities_keyword_hits": "Feature_keywords",
        }
    )
    out = normalize_engineered_column_names(df)
    assert "facilities_count" in out.columns
    assert "facilities_keyword_hits" in out.columns

    p = work / "mini.parquet"
    _tiny_joined_frame().to_parquet(p, index=False)
    back = load_joined_hotels(p)
    assert len(back) == 24

    df40 = _tiny_joined_frame(n=40)
    X_train, X_test, y_train, y_test, pre = build_modeling_feature_matrices(
        df40,
        stratify=False,
        test_size=0.25,
        random_state=0,
    )
    assert X_train.shape[0] == 30
    assert X_test.shape[0] == 10
    assert X_train.shape[1] == X_test.shape[1] > 6
    assert not X_train.isna().any().any()
    assert not X_test.isna().any().any()
    assert getattr(pre, "n_features_in_", None) is not None

    df20 = _tiny_joined_frame(n=20)
    X_train, X_test, y_train, y_test, pre = build_modeling_feature_matrices(
        df20, stratify=False, test_size=0.3, random_state=1
    )
    bundle = work / "bundle"
    paths = save_modeling_bundle(
        bundle,
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        preprocessor=pre,
        meta={"rows_train": int(X_train.shape[0])},
    )
    assert paths["X_train"].is_file()
    assert paths["preprocessor"].is_file()
    assert paths["meta"].is_file()


def main() -> int:
    p = argparse.ArgumentParser(description="Run smoke checks (no pytest).")
    p.add_argument(
        "--skip-modeling",
        action="store_true",
        help="Skip feature-matrix checks (needs scikit-learn).",
    )
    args = p.parse_args()

    print("smoke: quick helpers + ISO2 …")
    run_quick_smokes()

    with tempfile.TemporaryDirectory() as td:
        work = Path(td)
        print("smoke: cleaning pipeline on tiny CSVs …")
        run_cleaning_pipeline_smoke(work)
        if not args.skip_modeling:
            print("smoke: modeling pipeline …")
            run_modeling_smokes(work)

    print("All smoke checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
