# scripts

- `scripts/data/`
  - `head_datasets.py`: print first lines of raw CSVs
  - `peek_datasets.py`: print first rows of raw CSVs
- `scripts/pipeline/`
  - `run_cleaning.py`: run cleaning + join → `data/processed/`
  - `smoke_checks.py`: fast end-to-end checks (no pytest)
- `scripts/modeling/`
  - `train_baseline_model.py`: train baseline regressor on joined Parquet

Run from repo root so imports resolve (`src/`).
