## Results & outputs (extracted from runs)

This document consolidates the key outputs produced by the project notebooks/scripts:

- **Processed dataset summary** from `data/processed/hotels_with_cities-2.parquet`
- **Baseline model results** (sample = 100,000; random_state = 42)
- **Hyperparameter tuning results**
- **Hypothesis testing results** (permutation-based)

---

## 1) Processed dataset outputs

**Source file**

- `data/processed/hotels_with_cities-2.parquet`

**Rows / Cols**

- **1,010,033 rows**, **13 columns**

**Top 12 columns by % missing**

- `city_population`: **65.2777%**
- `hotel_star_rating`: **31.2532%**
- `hotel_latitude`: **0.0940%**
- `hotel_longitude`: **0.0940%**
- `countyCode`: **0.0903%**
- `countyName`: **0.0000%**
- `cityCode`: **0.0000%**
- `cityName`: **0.0000%**
- `HotelCode`: **0.0000%**
- `HotelName`: **0.0000%**
- `attractions_count`: **0.0000%**
- `facilities_count`: **0.0000%**

**Hotel star rating distribution (counts)**

- **1★**: **28,734**
- **2★**: **159,754**
- **3★**: **352,613**
- **4★**: **130,862**
- **5★**: **22,402**
- **Missing (<NA>)**: **315,668**

**Example rows (first 5; selected columns)**

- `countyName=Albania`, `cityName=Albanien` appears in the first rows
- `city_population` is often missing (`NaN`) in the first rows

---

## 2) Baseline modeling outputs (local; sample = 100,000)

**Common settings**

- `--sample 100000`
- `--random-state 42`
- joined table used: `data/processed/hotels_with_cities-2.parquet`

**Saved artifacts directory (local)**

- `outputs/model_runs_local/`

### linear

- Metrics file: `outputs/model_runs_local/linear/metrics.json`
- **RMSE**: **0.7527**
- **MAE**: **0.5885**
- **R²**: **0.1993**
- **|err| ≤ 0.5**: **0.50625**
- **|err| ≤ 1.0**: **0.83805**

### ridge (alpha = 10)

- Metrics file: `outputs/model_runs_local/ridge/metrics.json`
- **RMSE**: **0.7530**
- **MAE**: **0.5882**
- **R²**: **0.1986**
- **|err| ≤ 0.5**: **0.50560**
- **|err| ≤ 1.0**: **0.83800**

### lasso (alpha = 0.001)

- Metrics file: `outputs/model_runs_local/lasso/metrics.json`
- **RMSE**: **0.7646**
- **MAE**: **0.5932**
- **R²**: **0.1738**
- **|err| ≤ 0.5**: **0.49745**
- **|err| ≤ 1.0**: **0.83350**

### rf (n_estimators = 200, max_depth = 20)

- Metrics file: `outputs/model_runs_local/rf/metrics.json`
- **RMSE**: **0.6855**
- **MAE**: **0.5199**
- **R²**: **0.3358**
- **|err| ≤ 0.5**: **0.58885**
- **|err| ≤ 1.0**: **0.86345**

> Note: Kaggle runs for the same settings (same sample size + random seed) matched these baseline metrics.

---

## 3) Hyperparameter tuning outputs (local; sample = 100,000)

### ridge (tuned; CV folds = 3)

- Metrics file: `outputs/model_runs_local/ridge_tuned/metrics.json`
- **RMSE**: **0.7528**
- **MAE**: **0.5884**
- **R²**: **0.1990**
- **|err| ≤ 0.5**: **0.5064**
- **|err| ≤ 1.0**: **0.8382**
- `best_params`: `{}` (RidgeCV-selected internally)

### rf (tuned; CV folds = 3; randomized search iters = 12)

- Metrics file: `outputs/model_runs_local/rf_tuned/metrics.json`
- **RMSE**: **0.6754**
- **MAE**: **0.5088**
- **R²**: **0.3553**
- **|err| ≤ 0.5**: **0.5912**
- **|err| ≤ 1.0**: **0.8694**
- `best_params`:
  - `n_estimators`: **120**
  - `max_depth`: **24**
  - `max_features`: **0.5**
  - `min_samples_split`: **10**
  - `min_samples_leaf`: **2**

---

## 4) Hypothesis testing outputs (local; sample = 100,000; permutations = 200)

**Output file**

- `outputs/model_runs_local/hypothesis_tests.json`

**Results**

- `city_population`
  - null: coefficient equals 0
  - observed coefficient: **0.01423**
  - two-sided empirical p-value: **0.02488**
- `attractions_count`
  - null: coefficient equals 0
  - observed coefficient: **0.05592**
  - two-sided empirical p-value: **0.00498**

