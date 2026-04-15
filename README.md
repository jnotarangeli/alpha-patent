# alpha-patent

An equity alpha signal derived from firm-level patent activity. The pipeline aggregates the KPSS patent dataset to firm-month features, trains a LightGBM model in an expanding-window out-of-sample framework, and constructs long-short quintile portfolios. Factor spanning regressions confirm the signal delivers statistically significant alpha beyond the Fama-French five-factor model, momentum, and the AQR Quality Minus Junk (QMJ) factor.

## Research Hypothesis

Firms that are more innovative — as measured by the quantity, economic value, citation impact, and technological breadth of their recent patents — earn higher future returns. This outperformance is not explained by standard risk factors, suggesting either a mispricing of innovation or compensation for a distinct innovation risk premium.

## Data Sources

All raw data files are stored locally and excluded from version control.

| File | Source | Description |
|---|---|---|
| `KPSS_2024.csv` | Kogan, Papanikolaou, Seru & Stoffman (2024) | Patent-level data: issue date, filing date, economic value (ξ), forward citations, PERMNO link |
| `Match_patent_cpc_2024.csv` | KPSS (2024) | Patent → CPC (Cooperative Patent Classification) code mapping |
| `Match_patent_permco_permno_2024.csv` | KPSS (2024) | Patent → CRSP PERMNO/PERMCO identifier matching |
| `USA.parquet` | Jensen, Kelly & Pedersen (JKP) | Precomputed factor data for the US market |
| CRSP (`crsp.msf`, `crsp.msenames`) | WRDS | Monthly stock returns, prices, shares outstanding (1970–2024) |
| Fama-French Factors (`ff.factors_monthly`) | WRDS / Ken French | MKT-RF, SMB, HML, UMD (monthly, 1990–2024) |
| QMJ | AQR Data Library | Quality Minus Junk factor (US, monthly) |
| Compustat (`comp.funda`, `comp.seg_annfund`) | WRDS | Fundamentals and segment-level NAICS codes (`compustat.ipynb`) |

## Pipeline Overview

### `compustat.ipynb` — Data Preparation
Builds a comprehensive Compustat database enriched with industry codes:
1. Pull `comp.funda` (595K obs, 46K companies, 1950–2025)
2. Pull segment-level NAICS codes from `comp.seg_annfund` (2017+)
3. Merge and export to `compustat_with_naics.csv`

### `patent_signal.ipynb` — Main Signal Notebook
1. **Feature Engineering** — Aggregate KPSS patent data to firm × month:
   - Patent count and economic value (ξ) — trailing 12-month windows
   - Forward citations (quantity and quality of innovation)
   - Filing-to-issue lag (innovation speed)
   - CPC code breadth and section diversity (technology diversification)
   - Patent acceleration (change in activity vs. prior year)
2. **CRSP Pull** — Monthly returns for common stocks (share codes 10/11), 1970–2024, via WRDS
3. **Panel Construction** — Merge patent features with CRSP; cross-sectional rank normalization each month
4. **ML Prediction** — Expanding-window LightGBM regression predicting next-month returns; OOS from 1990, retrained annually
5. **Portfolio Construction** — Quintile long-short portfolios (Q5 − Q1), equal- and value-weighted
6. **Factor Spanning** — Newey-West HAC regressions on CAPM, FF3+MOM, and FF3+MOM+QMJ to test signal independence

## Key Features

| Feature | Description |
|---|---|
| `xi_real_12m` | Trailing 12-month patent economic value (KPSS ξ) |
| `cites_12m` | Trailing 12-month forward citation count |
| `patents_12m` | Trailing 12-month patent count |
| `mean_lag` | Average filing-to-issue lag in months |
| `mean_n_cpc` | Average CPC codes per patent (breadth) |
| `mean_n_sections` | Average distinct CPC sections (technology diversity) |
| `patent_accel` | Patent activity acceleration vs. prior 12 months |

## References

- Kogan, L., Papanikolaou, D., Seru, A., & Stoffman, N. (2017). *Technological Innovation, Resource Allocation, and Growth*. Quarterly Journal of Economics.
- Jensen, T. I., Kelly, B. T., & Pedersen, L. H. (2023). *Is There a Replication Crisis in Finance?* Journal of Finance.
- Fama, E. F. & French, K. R. (1993, 2015). Three-factor and five-factor asset pricing models.
- Asness, C., Frazzini, A., & Pedersen, L. H. (2019). *Quality Minus Junk*. Review of Accounting Studies.

## Requirements

- **WRDS account** with access to CRSP, Compustat, and Fama-French tables
- Python 3.9+
- Key packages: `pandas`, `numpy`, `lightgbm`, `statsmodels`, `wrds`, `scikit-learn`, `matplotlib`, `pyarrow`

## How to Run

1. Ensure WRDS credentials are configured (`~/.pgpass` or environment variables)
2. Place the KPSS data files (`KPSS_2024.csv`, `Match_patent_cpc_2024.csv`, `Match_patent_permco_permno_2024.csv`) and `USA.parquet` in the local data directory
3. Update `DATA_DIR` in the first cell of `patent_signal.ipynb`
4. Run `compustat.ipynb` first if you need the Compustat enriched database
5. Run `patent_signal.ipynb` end-to-end; data is cached as `.parquet` files after the first WRDS pull
