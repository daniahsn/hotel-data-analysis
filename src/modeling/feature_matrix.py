"""
Build the final numeric matrix for modeling from ``hotels_with_cities`` Parquet.

- Engineered counts: ``attractions_count``, ``facilities_count``, ``facilities_keyword_hits``
  (aliases from exports are normalized).
- Categoricals: country / city / locality (``countyCode``, ``cityCode``, ``cityName``) via
  one-hot with frequency caps (infrequent → pooled).
- Numeric: ``city_population``, ``hotel_latitude``, ``hotel_longitude`` + the three counts,
  with median imputation + :class:`sklearn.preprocessing.StandardScaler` (fit on train only).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# Columns produced by ``finalize_joined_hotels`` / cleaning pipeline.
JOINED_REQUIRED: tuple[str, ...] = (
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

# Aliases seen in notebooks / manual exports.
_COLUMN_ALIASES: dict[str, str] = {
    "facilities_keywords_count": "facilities_keyword_hits",
    "feature_count": "facilities_count",
    "Feature_keywords": "facilities_keyword_hits",
}


def normalize_engineered_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """Rename alias columns to the canonical names used in this repo."""
    out = df.copy()
    for bad, good in _COLUMN_ALIASES.items():
        if bad in out.columns and good not in out.columns:
            out = out.rename(columns={bad: good})
    return out


def load_joined_hotels(path: Path | str) -> pd.DataFrame:
    """Load joined Parquet; normalize column aliases."""
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(p)
    df = pd.read_parquet(p)
    df = normalize_engineered_column_names(df)
    missing = [c for c in JOINED_REQUIRED if c not in df.columns]
    if missing:
        raise ValueError(f"Joined table missing columns {missing}. Present: {list(df.columns)}")
    return df


def _ohe(max_categories: int | None) -> OneHotEncoder:
    kw: dict[str, Any] = {
        "handle_unknown": "ignore",
        "sparse_output": False,
    }
    if max_categories is not None:
        kw["max_categories"] = max_categories
    return OneHotEncoder(**kw)


def build_modeling_feature_matrices(
    df: pd.DataFrame,
    *,
    target_column: str = "hotel_star_rating",
    test_size: float = 0.2,
    random_state: int = 42,
    stratify: bool = True,
    drop_identifiers: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, ColumnTransformer]:
    """
    Train/test split, then fit a :class:`~sklearn.compose.ColumnTransformer` on **train only**.

    Categoricals (one-hot, infrequent pooled when ``max_categories`` is set):

    - **country**: ``countyCode``
    - **city**: ``cityCode`` (cast to string for hashing stability)
    - **region / locality** (no separate ``Region`` in join): ``cityName`` buckets local labels

    Numeric (median impute + ``StandardScaler``):

    - ``city_population``, ``hotel_latitude``, ``hotel_longitude``
    - ``attractions_count``, ``facilities_count``, ``facilities_keyword_hits``
    """
    df = normalize_engineered_column_names(df)
    if target_column not in df.columns:
        raise KeyError(f"Missing target column {target_column!r}")

    y = df[target_column]
    id_cols = ["HotelCode", "HotelName"]
    geo_cat = ["countyCode", "cityCode", "cityName"]
    numeric_cols = [
        "city_population",
        "hotel_latitude",
        "hotel_longitude",
        "attractions_count",
        "facilities_count",
        "facilities_keyword_hits",
    ]

    X = df.drop(columns=[target_column])
    if drop_identifiers:
        X = X.drop(columns=[c for c in id_cols if c in X.columns], errors="ignore")

    strat = None
    if stratify and y.notna().any():
        # Stratify only if few enough unique labels (classification-style target).
        if y.nunique(dropna=True) <= 50:
            strat = y

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=strat,
    )

    # Cast city id to string so OHE treats codes as categories, not magnitudes.
    X_train = X_train.copy()
    X_test = X_test.copy()
    if "cityCode" in X_train.columns:
        X_train["cityCode"] = X_train["cityCode"].astype("string")
        X_test["cityCode"] = X_test["cityCode"].astype("string")
    for c in ("countyCode", "cityName"):
        if c in X_train.columns:
            X_train[c] = X_train[c].astype("string")
            X_test[c] = X_test[c].astype("string")

    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    country_cols = [c for c in ("countyCode",) if c in X_train.columns]
    city_cols = [c for c in ("cityCode",) if c in X_train.columns]
    region_cols = [c for c in ("cityName",) if c in X_train.columns]

    transformers: list[tuple[str, Any, list[str]]] = [
        ("num", numeric_transformer, [c for c in numeric_cols if c in X_train.columns]),
        ("country", _ohe(250), country_cols),
        ("city", _ohe(120), city_cols),
        ("region_locality", _ohe(80), region_cols),
    ]

    # Drop empty transformer blocks (should not happen if schema correct).
    transformers = [(n, t, cols) for n, t, cols in transformers if cols]

    pre = ColumnTransformer(transformers=transformers, remainder="drop", verbose_feature_names_out=False)
    pre.set_output(transform="pandas")

    X_train_t = pre.fit_transform(X_train, y_train)
    X_test_t = pre.transform(X_test)

    return X_train_t, X_test_t, y_train, y_test, pre


def save_modeling_bundle(
    out_dir: Path,
    *,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    preprocessor: ColumnTransformer,
    meta: dict[str, Any] | None = None,
) -> dict[str, Path]:
    """Write train/test frames, preprocessor, and optional JSON metadata."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}
    paths["X_train"] = out_dir / "X_train.parquet"
    paths["X_test"] = out_dir / "X_test.parquet"
    paths["y_train"] = out_dir / "y_train.parquet"
    paths["y_test"] = out_dir / "y_test.parquet"
    paths["preprocessor"] = out_dir / "preprocessor.joblib"

    X_train.to_parquet(paths["X_train"], index=False)
    X_test.to_parquet(paths["X_test"], index=False)
    y_train.to_frame(name=y_train.name or "target").to_parquet(paths["y_train"], index=False)
    y_test.to_frame(name=y_test.name or "target").to_parquet(paths["y_test"], index=False)
    joblib.dump(preprocessor, paths["preprocessor"])

    if meta:
        meta_path = out_dir / "feature_matrix_meta.json"
        meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
        paths["meta"] = meta_path

    return paths


def default_joined_parquet(project_root: Path) -> Path:
    """Prefer ``hotels_with_cities.parquet``; else first ``hotels_with_cities*.parquet`` under ``data/processed``."""
    processed = project_root / "data" / "processed"
    direct = processed / "hotels_with_cities.parquet"
    if direct.is_file():
        return direct
    matches = sorted(processed.glob("hotels_with_cities*.parquet"))
    if not matches:
        raise FileNotFoundError(f"No hotels_with_cities*.parquet under {processed}")
    return matches[0]
