# CLAUDE.md — alpha-patent

## What This Project Is
An equity alpha signal pipeline that aggregates firm-level patent data from the Kogan-Papanikolaou-Seru-Stoffman (KPSS) dataset into monthly features, trains an expanding-window LightGBM model to predict next-month stock returns, and evaluates the resulting long-short signal via quintile portfolios and factor spanning regressions against Fama-French, momentum, and AQR Quality Minus Junk factors.

## Python Stack
- `pandas`, `numpy` — data manipulation
- `wrds` — WRDS database access (CRSP, Compustat, Fama-French)
- `lightgbm` — gradient boosting ML model
- `statsmodels` — OLS regressions with Newey-West HAC standard errors
- `scikit-learn` — cross-validation utilities
- `matplotlib` — charting (presentation-style output)
- `pyarrow` — parquet I/O

## Data Sources
Raw data lives **locally only** and is never committed to version control:
- `KPSS_2024.csv` — patent-level data (issue date, filing date, ξ value, citations, PERMNO)
- `Match_patent_cpc_2024.csv` — patent → CPC code mapping
- `Match_patent_permco_permno_2024.csv` — patent → CRSP identifier matching
- `USA.parquet` — JKP precomputed factor data (~4.65 GB)
- `compustat_with_naics.csv` — output of `compustat.ipynb` (~42 MB)
- CRSP monthly returns and Fama-French factors pulled live from WRDS

## Files to NEVER Commit
*.parquet, *.csv, *.xlsx, *.png, *.jpg, *.pdf, *.svg
Data/, data/, figures/, output/, Old/, .venv/

## Coding Conventions
- Vectorized operations only — no `iterrows()` or Python loops over DataFrames
- Cross-sectional operations use `.groupby().transform()` or `.groupby().apply()`
- All date alignment to month-end via `pd.offsets.MonthEnd(0)` before merging
- Feature names follow `snake_case`; rank-normalized versions use `_rank` suffix
- Follow existing notebook cell structure and naming conventions

## Session Start
1. Open `patent_signal.ipynb` — this is the main entry point
2. Update `DATA_DIR` in Cell 0 to your local data directory
3. Run `compustat.ipynb` separately only if you need the enriched Compustat database
4. WRDS connection is established inline via `wrds.Connection()` — requires `~/.pgpass`
