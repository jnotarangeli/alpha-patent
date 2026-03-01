# Patent-Based Alpha Signal

A machine learning pipeline that constructs a cross-sectional equity return signal from U.S. patent data (KPSS 2024) and evaluates whether it provides alpha beyond standard asset pricing factors.

## Overview

The notebook engineers firm-level innovation features from patent grants—citation counts, economic value (ξ_real), filing-to-issue lag, and CPC technology breadth—then uses **LightGBM** in an expanding-window out-of-sample framework to predict next-month stock returns. Predicted returns are sorted into quintile portfolios, producing a long-short patent signal that is tested for factor spanning against MKT, SMB, HML, MOM, and QMJ.

## Pipeline

| Step | Description |
|------|-------------|
| **1 – Feature Engineering** | Aggregate KPSS patent data to firm × month; compute trailing 12-month rolling counts, values, citations, acceleration, and technology diversity metrics. |
| **2 – CRSP Returns** | Pull monthly returns and market cap for common stocks (SHRCD 10/11) from WRDS. |
| **3 – Merge** | Join patent features to CRSP on `permno × year-month`; construct the investable universe of patent-holding firms. |
| **4 – Normalization** | Cross-sectional rank-transform all features each month to [0, 1]; create intensity and log-scaled variants. |
| **5 – ML Prediction** | Expanding-window LightGBM (retrained every 12 months, OOS from 1990). Quintile-sort predicted returns into equal- and value-weighted long-short portfolios. |
| **6 – Factor Spanning** | Regress the L/S return on FF3 + Momentum + QMJ using Newey-West standard errors. |

## Data Requirements

| Source | Dataset | Access |
|--------|---------|--------|
| KPSS | `KPSS_2024.csv`, `Match_patent_cpc_2024.csv`, `Match_patent_permco_permno_2024.csv` | [KPSS patent data](https://github.com/KPSS2017/Technological-Innovation-Resource-Allocation-and-Growth-Extended-Data) |
| WRDS | CRSP monthly stock file (`crsp.msf`, `crsp.msenames`) | [WRDS](https://wrds-www.wharton.upenn.edu/) subscription |
| WRDS | Fama-French factors + Momentum (`ff.factors_monthly`) | WRDS |
| AQR | Quality Minus Junk (QMJ) monthly factors | [AQR Data Library](https://www.aqr.com/Insights/Datasets) |

> **Note:** You will need valid WRDS credentials (configured via `~/.pgpass` or entered at runtime) and should update `DATA_DIR` to point to your local copy of the KPSS files.

## Key Features (22 total, rank-normalized)

Patent counts & acceleration, economic value (ξ_real), forward citations, filing-to-issue lag, CPC code breadth, patent/citation intensity scaled by market cap, and log transforms.

## Dependencies

```
pandas, numpy, lightgbm, scikit-learn, statsmodels, matplotlib, wrds, joblib, openpyxl
```

Install with:

```bash
pip install pandas numpy lightgbm scikit-learn statsmodels matplotlib wrds joblib openpyxl
```

## Usage

1. Download the KPSS CSV files and update `DATA_DIR` in the first cell.
2. Ensure WRDS credentials are configured.
3. Run the notebook end-to-end (`patent_signal.ipynb`).

## Results (Out-of-Sample, 1990–2024)

### Long-Short Portfolio Performance

| Metric | Equal-Weighted | Value-Weighted |
|--------|---------------|----------------|
| Annualized Return | 7.23% | −1.09% |
| Annualized Volatility | 11.85% | 14.69% |
| Sharpe Ratio | 0.61 | −0.07 |
| t-statistic | 3.61 | −0.44 |
| Max Drawdown | −27.92% | −63.76% |
| Cumulative Return | 882.59% | −53.46% |

The signal is strongly concentrated in smaller firms—equal-weighted portfolios capture the effect while value-weighting dilutes it.

### Quintile Spreads (Avg Monthly Return, Equal-Weighted)

| Q1 (Low) | Q2 | Q3 | Q4 | Q5 (High) | L/S Spread |
|-----------|------|------|------|-----------|------------|
| 1.10% | 1.16% | 1.26% | 1.30% | 1.71% | 0.60% |

Returns increase monotonically from Q1 to Q5, consistent with the market underpricing patent information.

### Factor Spanning Regressions (Newey-West, EW Long-Short)

| Model | Alpha (ann.) | t-stat | R² |
|-------|-------------|--------|-----|
| CAPM | 6.38% | 3.60 | 0.016 |
| FF3 + MOM | 5.88% | 3.30 | 0.027 |
| FF3 + MOM + QMJ | 7.26% | 3.91 | 0.039 |

Alpha remains statistically significant (t > 3) across all specifications. R² stays below 4%, indicating the patent signal is nearly orthogonal to standard equity factors.

**Full factor loadings (FF3 + MOM + QMJ):**

| Factor | Coefficient | t-stat |
|--------|------------|--------|
| MKT-RF | 0.050 | 0.95 |
| SMB | 0.017 | 0.28 |
| HML | 0.020 | 0.28 |
| MOM | 0.084 | 1.27 |
| QMJ | −0.198 | −1.85 |

No individual factor loading is significant at the 5% level, confirming that the signal is complementary to value, size, momentum, and quality.

### Top Predictive Features (LightGBM Gain)

1. Total citations (patent impact)
2. Filing-to-issue lag (innovation speed)
3. CPC breadth (technology diversity)
4. Patent value (ξ_real)

## License

This project is licensed under the [MIT License](LICENSE).
