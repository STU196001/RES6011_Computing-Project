# Station-Level Passenger Demand Forecasting for the Dubai Metro

A comparative study of SARIMA and LSTM time series models for forecasting monthly
passenger demand at 53 Dubai Metro stations, with an interactive Plotly Dash dashboard.

## Key results
- SARIMA: mean MAPE 8.20%, median 7.27% across 53 stations (zero failures)
- LSTM: mean MAPE 24.39% across 51 stations (2 excluded for insufficient history)
- SARIMA was the better model at 47 of 53 stations; LSTM at 4; 2 not compared
- Modelling window: January 2018 to August 2023

## Quick start (run the dashboard)
1. Install Python 3.11 (TensorFlow does not support 3.14)
2. `py -3.11 -m venv venv`
3. `venv\Scripts\activate`
4. `pip install -r requirements.txt`
5. `python dashboard/app.py`, then open the address shown in the terminal
   (normally http://127.0.0.1:8050)

The dashboard reads from `data_processed/` and `models/`, both included, so the
raw data and notebooks are not needed to run it.

## Repository structure
| Path | Contents |
|---|---|
| `data_raw/` | Placeholder only; see `data_raw/README.md` for the source |
| `data_processed/` | Cleaned and model-ready data, station coordinates, line and map layout files |
| `notebooks/` | Pipeline notebooks 01-07 (data preparation to dashboard preparation) |
| `models/sarima/` | Fitted SARIMA model per station |
| `models/lstm/` | Trained LSTM model and scaler per station |
| `dashboard/` | Dash application (Explorer, Comparison and Insights tabs) |
| `tests/` | `Testing.ipynb`: automated tests (T01-T12) and test evidence |
| `Evidence/` | Chapter 5 and Chapter 8 evidence notebooks |
| `docs/` | Exploratory and modelling figures, results tables, design documents |
| `docs/Evidence/` | Tables and figures cited in report Chapters 5 and 8 |
| `docs/testing/` | Test plan and results tables, defect log, figures and screenshots |
| `prototype/` | Interactive HTML prototype from the design stage |

## Reproducing the full pipeline
Run notebooks 01 to 07 in order. Notebook 01 needs the raw dataset (see `data_raw/README.md`).
LSTM training runs were not seeded, so retraining gives slightly different error values.
The saved models in `models/` reproduce the reported results.

## Testing
Automated tests and functional test evidence are in `tests/Testing.ipynb`.
The test plan, results and defect log are in `docs/testing/tables/`
(for example `Table_6.1_test_plan_and_results.csv` and `Table_6.4_defect_log.csv`).

## Limitations
- The dashboard uses historical data only (no real-time feed)
- Trip counts change methodology from May 2024; a Sep-Nov 2023 data gap also exists,
  so the modelling window ends in August 2023
- LSTM was evaluated one-step-ahead; SARIMA over a 6-month forecast horizon

## Data source
Dubai Data / Dubai Pulse, "Metro Passengers Trips by Station Monthly"
(export dated 10 April 2026; downloaded 29 April 2026).
