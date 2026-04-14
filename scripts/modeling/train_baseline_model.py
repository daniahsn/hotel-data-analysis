#!/usr/bin/env python3
"""
Train a baseline regressor on the joined hotels table (features from ``feature_matrix``).

Resolves the joined Parquet via :func:`src.raw_data_paths.resolve_joined_hotels_parquet`
(same path logic on Kaggle and locally).

Examples::

  python scripts/modeling/train_baseline_model.py --sample 100000 --model rf
  python scripts/modeling/train_baseline_model.py --joined-path /path/to/hotels_with_cities.parquet --model linear
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.modeling.feature_matrix import build_modeling_feature_matrices, load_joined_hotels
from src.raw_data_paths import resolve_joined_hotels_parquet


def _default_out_dir() -> Path:
    kaggle_working = Path("/kaggle/working")
    if kaggle_working.is_dir():
        return kaggle_working / "model_artifacts"
    return _ROOT / "outputs" / "model_artifacts"


def main() -> int:
    p = argparse.ArgumentParser(description="Train baseline regressor on joined hotels Parquet.")
    p.add_argument(
        "--joined-path",
        type=Path,
        default=None,
        help=f"Override path to joined Parquet (default: {resolve_joined_hotels_parquet.__name__}())",
    )
    p.add_argument(
        "--project-root",
        type=Path,
        default=_ROOT,
        help="Repository root for local joined-path resolution (default: repo containing this script)",
    )
    p.add_argument("--sample", type=int, default=None, metavar="N", help="Use random N rows after dropping NA target")
    p.add_argument(
        "--model",
        choices=("linear", "rf"),
        default="rf",
        help="linear = LinearRegression; rf = RandomForestRegressor",
    )
    p.add_argument("--test-size", type=float, default=0.2)
    p.add_argument("--random-state", type=int, default=42)
    p.add_argument("--out-dir", type=Path, default=None, help="Where to save model + metrics JSON")
    p.add_argument("--rf-estimators", type=int, default=100)
    p.add_argument("--rf-max-depth", type=int, default=20)
    args = p.parse_args()

    joined = args.joined_path or resolve_joined_hotels_parquet(args.project_root)
    out_dir = args.out_dir or _default_out_dir()
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"joined table: {joined}")
    df = load_joined_hotels(joined)
    df = df.dropna(subset=["hotel_star_rating"])
    if args.sample is not None:
        n = min(args.sample, len(df))
        df = df.sample(n=n, random_state=args.random_state)
    print(f"rows used: {len(df):,}")

    X_train, X_test, y_train, y_test, pre = build_modeling_feature_matrices(
        df,
        target_column="hotel_star_rating",
        test_size=args.test_size,
        random_state=args.random_state,
        stratify=True,
    )

    if args.model == "linear":
        model = LinearRegression()
    else:
        model = RandomForestRegressor(
            n_estimators=args.rf_estimators,
            max_depth=args.rf_max_depth,
            n_jobs=-1,
            random_state=args.random_state,
        )

    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    rmse = float(np.sqrt(mean_squared_error(y_test, pred)))
    r2 = float(r2_score(y_test, pred))

    print(f"RMSE: {rmse:.4f}")
    print(f"R2:   {r2:.4f}")

    joblib.dump(model, out_dir / "regressor.joblib")
    joblib.dump(pre, out_dir / "preprocessor.joblib")
    meta = {
        "joined_path": str(joined.resolve()),
        "rows": len(df),
        "model": args.model,
        "rmse": rmse,
        "r2": r2,
        "X_train_shape": list(X_train.shape),
        "X_test_shape": list(X_test.shape),
    }
    (out_dir / "metrics.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"wrote artifacts under {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
