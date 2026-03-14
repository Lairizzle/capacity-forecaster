# capacity-forecaster

A robust capacity planning forecasting engine for workforce and operations teams.
Uses SARIMAX with automatic model selection to produce multi-month forecasts of
**Volume**, **Hours**, **FTE**, and **Shrinkage-adjusted FTE** — with 95%
confidence intervals — for any number of planning groups.

---

## Features

- **Automatic model selection** — grid-searches SARIMAX orders and picks the best
  fit by AICc (corrected AIC), per group, per metric
- **Shrinkage forecasting** — when shrinkage history is supplied, it is modelled
  as its own time series so seasonal leave patterns flow through to adjusted headcount
- **Gap filling** — missing months within a group's history are detected and
  linearly interpolated automatically, with the count reported in `Imputed_Months`
- **Data quality flagging** — groups with limited history are forecasted and
  flagged `LOW_HISTORY` rather than silently dropped, so you decide what to trust
- **Confidence intervals** — every metric includes `_Lower` and `_Upper` bounds at
  the 95% level
- **Flexible input** — column names are ignored; columns are matched by position,
  so your existing DataFrames work without renaming

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

df = pd.read_csv("capacity_planning_data.csv", parse_dates=["date"])

forecaster = CapacityForecaster(
    weekly_hours=37.5,       # contracted hours per FTE per week
    forecast_horizon=12,     # months ahead to forecast
    default_shrinkage=0.30,  # fallback shrinkage rate (30%)
)

results = forecaster.forecast(df)
print(results.head())
```

---

## Input format

Columns are matched **by position**, not by name. Pass your DataFrame with
columns in this order:

| Position | Role | Type | Required | Description |
|---|---|---|---|---|
| 0 | Date | datetime | ✅ | Any monthly date (month-start or month-end). Mixed formats within a column are fine. |
| 1 | Group | str | ✅ | Group identifier (e.g. team or queue name). |
| 2 | Volume | float | ✅ | Units of work completed that month. |
| 3 | Hours | float | ✅ | Total workload hours that month. |
| 4 | Shrinkage | float | ➕ optional | Shrinkage rate in `[0.0, 1.0)`, e.g. `0.25` = 25%. |

Each row is one group × one month. You need at least **6 months** of history
per group to attempt a forecast; **36 months** is recommended for full seasonal
accuracy.

Because columns are positional, you can pass a subset of your DataFrame directly:

```python
# Your columns can be named anything
results = forecaster.forecast(
    df[["period_end", "team", "contacts", "handle_hours"]]
)

# With optional shrinkage
results = forecaster.forecast(
    df[["period_end", "team", "contacts", "handle_hours", "shrinkage_rate"]]
)
```

---

## Output columns

| Column | Description |
|---|---|
| `Date` | Forecast month (month-start). |
| *(your group column name)* | Group label, using whatever name your input column had. |
| `Forecasted_Volume` | Predicted work volume. |
| `Forecasted_Volume_Lower` | Lower 95% CI bound. |
| `Forecasted_Volume_Upper` | Upper 95% CI bound. |
| `Forecasted_Hours` | Predicted workload hours. |
| `Forecasted_Hours_Lower` | Lower 95% CI bound. |
| `Forecasted_Hours_Upper` | Upper 95% CI bound. |
| `Forecasted_FTE` | Raw FTE required (hours ÷ hours-per-FTE-per-month). |
| `Forecasted_FTE_Lower` | Lower 95% CI bound. |
| `Forecasted_FTE_Upper` | Upper 95% CI bound. |
| `Forecasted_FTE_Adjusted` | FTE grossed up for shrinkage (`Raw FTE ÷ (1 − Shrinkage)`). |
| `Forecasted_FTE_Adjusted_Lower` | Lower 95% CI bound. |
| `Forecasted_FTE_Adjusted_Upper` | Upper 95% CI bound. |
| `Shrinkage_Used` | The shrinkage value applied each forecast month (forecasted or default). |
| `Imputed_Months` | Number of months gap-filled within the group's history. |
| `Data_Quality` | `"OK"` (≥ 36 months history) or `"LOW_HISTORY"` (6–35 months). |

---

## Data quality

| History | Behaviour | `Data_Quality` |
|---|---|---|
| < 6 months | Forecast attempted with simplified model. Results should be treated with caution. | `LOW_HISTORY` |
| 6–35 months | Forecast attempted. Non-seasonal ARIMA candidates used where seasonal fit is unreliable. | `LOW_HISTORY` |
| ≥ 36 months | Full seasonal SARIMAX with complete candidate grid. | `OK` |

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
| `default_shrinkage` | float | `0.30` | Fallback shrinkage when no shrinkage column is supplied or values are NaN. |

### `forecast(df)`

Runs the forecast pipeline and returns a `pd.DataFrame` with one row per
group × forecast month.

---

## Shrinkage

Shrinkage represents time lost to leave, training, breaks, and absence.
The adjusted FTE formula is:

```
Adjusted FTE = Raw FTE / (1 - Shrinkage)
```

**Without a shrinkage column** — `default_shrinkage` is applied flat across
all forecast months for all groups.

**With a shrinkage column** — shrinkage is forecasted independently per group
using SARIMAX, capturing seasonal patterns (e.g. higher shrinkage in August
and December due to holiday leave). The fallback chain if the model cannot
fit is: historical group mean → `default_shrinkage`. The forecasted value
is clipped to `[0.0, 0.9999]` to keep availability positive. The value
actually applied each month is always visible in `Shrinkage_Used`.

---

## Public API

```python
from capacity_forecaster import (
    CapacityForecaster,       # main class
    MIN_DATA_POINTS,          # 36 — recommended minimum observations
    ABSOLUTE_MIN_DATA_POINTS, # 6  — hard floor for attempting a forecast
    CONFIDENCE_LEVEL,         # 0.95
    DataQuality,              # DataQuality.OK / DataQuality.LOW_HISTORY
)
```

---

## Dependencies

- `pandas >= 2.0`
- `numpy >= 1.23`
- `statsmodels >= 0.14`

---

## License

MIT
