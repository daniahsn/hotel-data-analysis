# hotel-data-analysis

## Layout

- `src/`: Python package (cleaning, features, modeling)
  - `src/name_mappings.py`: **single place** for CSV header aliases, joined Parquet column aliases, required modeling columns, joined output file stem
  - `src/raw_data_paths.py`: where raw/joined **files** are on Kaggle vs local (paths only, not renames)
- `scripts/`: runnable entrypoints (grouped by purpose)
  - `scripts/data/`: quick previews of raw CSVs
  - `scripts/pipeline/`: cleaning + smoke checks
  - `scripts/modeling/`: baseline training
- `notebooks/`: Kaggle bootstrap notebook
- `docs/`: course writeups (e.g. `docs/proposal.md`)
- `data/`: local data folders (`data/raw/`, `data/processed/`)

## Common commands

```bash
pip install -r requirements.txt
python scripts/pipeline/smoke_checks.py
python scripts/pipeline/run_cleaning.py --sample 5000
python scripts/modeling/train_baseline_model.py --sample 100000 --model rf
```
