#!/usr/bin/env python3
"""
Train a baseline regressor on the joined hotels table (features from ``feature_matrix``).

Resolves the joined Parquet via :func:`src.raw_data_paths.resolve_joined_hotels_parquet`
(same path logic on Kaggle and locally).

Examples::

  python scripts/modeling/train_baseline_model.py --sample 100000 --model rf
  python scripts/modeling/train_baseline_model.py --sample 100000 --model xgb
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
from sklearn.linear_model import Lasso, LinearRegression, Ridge, RidgeCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import RandomizedSearchCV

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


def _compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    abs_err = np.abs(y_pred - y_true)
    return {
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
        "within_0_5": float(np.mean(abs_err <= 0.5)),
        "within_1_0": float(np.mean(abs_err <= 1.0)),
    }


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
        choices=("linear", "ridge", "lasso", "rf", "xgb"),
        default="rf",
        help="linear = LinearRegression; ridge/lasso = regularized linear models; rf = RandomForestRegressor; xgb = XGBoost regressor",
    )
    p.add_argument("--test-size", type=float, default=0.2)
    p.add_argument("--random-state", type=int, default=42)
    p.add_argument("--out-dir", type=Path, default=None, help="Where to save model + metrics JSON")
    p.add_argument("--rf-estimators", type=int, default=100)
    p.add_argument("--rf-max-depth", type=int, default=20)
    p.add_argument("--ridge-alpha", type=float, default=1.0, help="Regularization strength for Ridge")
    p.add_argument("--lasso-alpha", type=float, default=0.001, help="Regularization strength for Lasso")
    p.add_argument("--xgb-estimators", type=int, default=300)
    p.add_argument("--xgb-max-depth", type=int, default=8)
    p.add_argument("--xgb-learning-rate", type=float, default=0.05)
    p.add_argument("--xgb-subsample", type=float, default=0.8)
    p.add_argument("--xgb-colsample-bytree", type=float, default=0.8)
    p.add_argument("--tune", action="store_true", help="Enable lightweight CV hyperparameter tuning")
    p.add_argument("--cv-folds", type=int, default=3, help="Cross-validation folds for --tune")
    p.add_argument("--tune-iters", type=int, default=12, help="Randomized search iterations for --tune")
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

    best_params: dict[str, float | int | str] = {}
    if args.model == "linear":
        model = LinearRegression()
    elif args.model == "ridge":
        if args.tune:
            alphas = np.logspace(-3, 3, 13)
            model = RidgeCV(alphas=alphas, cv=args.cv_folds)
        else:
            model = Ridge(alpha=args.ridge_alpha, random_state=args.random_state)
    elif args.model == "lasso":
        if args.tune:
            model = RandomizedSearchCV(
                estimator=Lasso(max_iter=10_000, random_state=args.random_state),
                param_distributions={"alpha": np.logspace(-4, 0, 20)},
                n_iter=args.tune_iters,
                scoring="neg_root_mean_squared_error",
                cv=args.cv_folds,
                random_state=args.random_state,
                n_jobs=-1,
            )
        else:
            model = Lasso(alpha=args.lasso_alpha, max_iter=10_000, random_state=args.random_state)
    elif args.model == "xgb":
        try:
            from xgboost import XGBRegressor
        except ImportError as e:
            raise SystemExit(
                "xgboost is required for --model xgb. Install it with: python -m pip install xgboost"
            ) from e
        if args.tune:
            model = RandomizedSearchCV(
                estimator=XGBRegressor(
                    objective="reg:squarederror",
                    n_jobs=-1,
                    random_state=args.random_state,
                ),
                param_distributions={
                    "n_estimators": [200, 300, 500, 700],
                    "max_depth": [4, 6, 8, 10],
                    "learning_rate": [0.03, 0.05, 0.08, 0.12],
                    "subsample": [0.7, 0.8, 0.9, 1.0],
                    "colsample_bytree": [0.6, 0.8, 1.0],
                    "reg_lambda": [0.5, 1.0, 3.0, 10.0],
                },
                n_iter=args.tune_iters,
                scoring="neg_root_mean_squared_error",
                cv=args.cv_folds,
                random_state=args.random_state,
                n_jobs=-1,
            )
        else:
            model = XGBRegressor(
                n_estimators=args.xgb_estimators,
                max_depth=args.xgb_max_depth,
                learning_rate=args.xgb_learning_rate,
                subsample=args.xgb_subsample,
                colsample_bytree=args.xgb_colsample_bytree,
                objective="reg:squarederror",
                n_jobs=-1,
                random_state=args.random_state,
            )
    else:
        if args.tune:
            model = RandomizedSearchCV(
                estimator=RandomForestRegressor(
                    n_jobs=-1,
                    random_state=args.random_state,
                ),
                param_distributions={
                    "n_estimators": [120, 200, 300, 500],
                    "max_depth": [10, 16, 24, None],
                    "min_samples_split": [2, 5, 10],
                    "min_samples_leaf": [1, 2, 4],
                    "max_features": ["sqrt", "log2", 0.5, None],
                },
                n_iter=args.tune_iters,
                scoring="neg_root_mean_squared_error",
                cv=args.cv_folds,
                random_state=args.random_state,
                n_jobs=-1,
            )
        else:
            model = RandomForestRegressor(
                n_estimators=args.rf_estimators,
                max_depth=args.rf_max_depth,
                n_jobs=-1,
                random_state=args.random_state,
            )

    model.fit(X_train, y_train)
    if hasattr(model, "best_params_"):
        best_params = dict(model.best_params_)  # type: ignore[assignment]
        print(f"best params: {best_params}")
        model = model.best_estimator_  # type: ignore[assignment]
    pred = model.predict(X_test)
    metrics = _compute_metrics(y_test.to_numpy(dtype=float), np.asarray(pred, dtype=float))

    print(f"RMSE: {metrics['rmse']:.4f}")
    print(f"MAE:  {metrics['mae']:.4f}")
    print(f"R2:   {metrics['r2']:.4f}")
    print(f"|err|<=0.5: {metrics['within_0_5']:.3f}")
    print(f"|err|<=1.0: {metrics['within_1_0']:.3f}")

    joblib.dump(model, out_dir / "regressor.joblib")
    joblib.dump(pre, out_dir / "preprocessor.joblib")
    meta = {
        "joined_path": str(joined.resolve()),
        "rows": len(df),
        "model": args.model,
        "tuned": bool(args.tune),
        "best_params": best_params,
        **metrics,
        "X_train_shape": list(X_train.shape),
        "X_test_shape": list(X_test.shape),
    }
    (out_dir / "metrics.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"wrote artifacts under {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
