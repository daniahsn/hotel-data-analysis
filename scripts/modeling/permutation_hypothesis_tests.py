#!/usr/bin/env python3
"""
Permutation-based hypothesis tests for coefficient significance.

Current tests target:
  - city_population coefficient
  - attractions_count coefficient
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import LinearRegression

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.modeling.feature_matrix import build_modeling_feature_matrices, load_joined_hotels
from src.raw_data_paths import resolve_joined_hotels_parquet


def _coef_for_feature(model: LinearRegression, feature_names: list[str], feature: str) -> float:
    if feature not in feature_names:
        raise KeyError(f"Feature {feature!r} not found. Present: {feature_names}")
    idx = feature_names.index(feature)
    return float(model.coef_[idx])


def _empirical_two_sided_p(obs: float, null_values: np.ndarray) -> float:
    return float((np.sum(np.abs(null_values) >= abs(obs)) + 1) / (len(null_values) + 1))


def main() -> int:
    p = argparse.ArgumentParser(description="Permutation tests for linear-regression coefficients.")
    p.add_argument("--joined-path", type=Path, default=None, help="Optional override path to joined Parquet")
    p.add_argument("--project-root", type=Path, default=_ROOT)
    p.add_argument("--sample", type=int, default=None, metavar="N", help="Randomly sample N rows")
    p.add_argument("--n-permutations", type=int, default=400, metavar="N")
    p.add_argument("--test-size", type=float, default=0.2)
    p.add_argument("--random-state", type=int, default=42)
    p.add_argument(
        "--features",
        nargs="+",
        default=["city_population", "attractions_count"],
        help="Coefficient features to test in transformed matrix",
    )
    p.add_argument(
        "--out",
        type=Path,
        default=_ROOT / "outputs" / "model_artifacts" / "hypothesis_tests.json",
        help="Output JSON path",
    )
    args = p.parse_args()

    joined = args.joined_path or resolve_joined_hotels_parquet(args.project_root)
    df = load_joined_hotels(joined).dropna(subset=["hotel_star_rating"])
    if args.sample is not None:
        df = df.sample(n=min(args.sample, len(df)), random_state=args.random_state)

    X_train, _, y_train, _, _ = build_modeling_feature_matrices(
        df,
        target_column="hotel_star_rating",
        test_size=args.test_size,
        random_state=args.random_state,
        stratify=True,
    )
    feature_names = list(X_train.columns)

    base_model = LinearRegression()
    base_model.fit(X_train, y_train)

    rng = np.random.default_rng(args.random_state)
    null_dist: dict[str, list[float]] = {f: [] for f in args.features}
    y_train_np = y_train.to_numpy()

    for _ in range(args.n_permutations):
        perm_model = LinearRegression()
        y_perm = rng.permutation(y_train_np)
        perm_model.fit(X_train, y_perm)
        for f in args.features:
            null_dist[f].append(_coef_for_feature(perm_model, feature_names, f))

    tests: dict[str, dict[str, float | str]] = {}
    for f in args.features:
        observed = _coef_for_feature(base_model, feature_names, f)
        null_values = np.asarray(null_dist[f], dtype=float)
        p_val = _empirical_two_sided_p(observed, null_values)
        tests[f] = {
            "null_hypothesis": f"Coefficient for {f} equals 0",
            "observed_coefficient": observed,
            "empirical_p_value_two_sided": p_val,
            "null_mean": float(null_values.mean()),
            "null_std": float(null_values.std(ddof=1)),
        }

    out = {
        "joined_path": str(joined.resolve()),
        "rows_used": int(len(df)),
        "n_permutations": int(args.n_permutations),
        "tests": tests,
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"Wrote hypothesis test summary to {args.out}")
    for f, r in tests.items():
        print(
            f"{f}: coef={r['observed_coefficient']:.6f}, "
            f"p={r['empirical_p_value_two_sided']:.4f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

