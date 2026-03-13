# capacity-forecaster

A robust capacity planning forecasting engine for workforce and operations teams.
Uses SARIMAX with automatic model selection to produce 12-month forecasts of
**Volume**, **Hours**, **FTE**, and **Shrinkage-adjusted FTE** — with 95 %
confidence intervals — for any number of planning groups.

---

## Features

- **Automatic model selection** — grid-searches SARIMAX orders and picks the best
  fit by AICc (corrected AIC), per group, per metric
- **Shrinkage forecasting** — shrinkage is modelled as its own time series
  (logit-transformed SARIMAX) so seasonal leave patterns flow through to
  adjusted headcount
- **Data quality flagging** — groups with limited history are forecasted and
  flagged `LOW_HISTORY` rather than silently dropped, so you decide what to
  trust
- **Confidence intervals** — every metric includes `_Lower` and `_Upper` bounds
- **Simple input/output** — feed in a DataFrame with four columns, get back a
  clean forecast DataFrame ready to export or visualise

---

## Installation

```bash
pip install capacity-forecaster
```

---

## Quickstart

```python
import pandas as pd
from capacity_forecaster import CapacityForecaster

df = pd.read_csv("capacity_planning_data.csv")

forecaster = CapacityForecaster(
    weekly_hours=37.5,       # contracted hours per FTE per week
    forecast_horizon=12,     # months ahead to forecast
    default_shrinkage=0.30,  # fallback shrinkage rate (30 %)
)

results = forecaster.forecast(df)
print(results.head())
```

---

## Input format

| Column | Type | Required | Description |
|---|---|---|---|
| `Date` | date / str | ✅ | Month-start date. Any parseable format. |
| `Capacity Planning Group` | str | ✅ | Group identifier (e.g. team name). |
| `Hours` | float | ✅ | Total workload hours that month. |
| `Volume` | float | ✅ | Units of work completed that month. |
| `Shrinkage` | float | ➕ optional | Shrinkage rate in `[0.0, 1.0)` e.g. `0.25` = 25 %. When supplied, shrinkage is forecasted per group. |

Each row is one group × one month. You need at least **6 months** of history
per group to attempt a forecast; **36 months** is recommended for full accuracy.

---

## Output columns

| Column | Description |
|---|---|
| `Date` | Forecast month (month-start). |
| `Capacity Planning Group` | Group name. |
| `Data_Quality` | `"OK"` (≥ 36 months history) or `"LOW_HISTORY"` (6–35 months). |
| `Forecasted_Shrinkage` | Forecasted shrinkage rate for that month. |
| `Forecasted_Volume` | Predicted work volume. |
| `Forecasted_Volume_Lower` | Lower 95 % CI bound. |
| `Forecasted_Volume_Upper` | Upper 95 % CI bound. |
| `Forecasted_Hours` | Predicted workload hours. |
| `Forecasted_Hours_Lower` | Lower 95 % CI bound. |
| `Forecasted_Hours_Upper` | Upper 95 % CI bound. |
| `Forecasted_FTE` | Raw FTE required (Hours ÷ hours-per-FTE-per-month). |
| `Forecasted_FTE_Lower` | Lower 95 % CI bound. |
| `Forecasted_FTE_Upper` | Upper 95 % CI bound. |
| `Forecasted_FTE_Adjusted` | FTE grossed up for shrinkage. |
| `Forecasted_FTE_Adjusted_Lower` | Lower 95 % CI bound. |
| `Forecasted_FTE_Adjusted_Upper` | Upper 95 % CI bound. |

---

## Data quality

Groups are processed differently based on months of history available:

| History | Behaviour | `Data_Quality` |
|---|---|---|
| < 6 months | Skipped entirely. Warning issued. | — |
| 6–35 months | Forecast attempted. Non-seasonal ARIMA used. | `LOW_HISTORY` |
| ≥ 36 months | Full seasonal SARIMAX. | `OK` |

Filter to only reliable forecasts when needed:

```python
reliable = results[results["Data_Quality"] == "OK"]
```

---

## Parameters

### `CapacityForecaster`

| Parameter | Type | Default | Description |
|---|---|---|---|
| `weekly_hours` | float | `37.5` | Contracted hours per FTE per week. |
| `forecast_horizon` | int | `12` | Months ahead to forecast. |
| `default_shrinkage` | float | `0.30` | Fallback shrinkage rate when no `Shrinkage` column is present or values are NaN. |

### `forecast(df)`

Runs the forecast. Returns a `pd.DataFrame`.

### `diagnostic_summary(df)`

Returns a dict of per-group model metadata (chosen SARIMAX orders, AICc,
stationarity) without producing a forecast. Useful for model inspection and QA.

```python
summary = forecaster.diagnostic_summary(df)
for group, info in summary.items():
    print(group, info["data_quality"], info.get("hours_model"))
```

---

## Public API

```python
from capacity_forecaster import (
    CapacityForecaster,       # main class
    MIN_DATA_POINTS,          # 36 — recommended minimum
    ABSOLUTE_MIN_DATA_POINTS, # 6  — hard floor
    CONFIDENCE_LEVEL,         # 0.95
    DataQuality,              # DataQuality.OK / DataQuality.LOW_HISTORY
)
```

---

## Shrinkage

Shrinkage represents time lost to leave, training, breaks, and absence.
The adjusted FTE formula is:

```
Adjusted FTE = Raw FTE / (1 - Shrinkage)
```

When a `Shrinkage` column is present in your data, shrinkage is forecasted
independently per group using a logit-transformed SARIMAX model, capturing
seasonal patterns (e.g. higher shrinkage in August and December due to
holiday leave). Without a `Shrinkage` column, the `default_shrinkage` rate
is applied flat across all forecast months.

---

## Dependencies

- `pandas >= 1.5`
- `numpy >= 1.23`
- `statsmodels >= 0.14`

---

## License

MIT
