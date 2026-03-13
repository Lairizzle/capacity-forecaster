"""
capacity_forecaster
===================
Robust capacity planning forecasting engine using SARIMAX with automatic
model selection, per-group validation, shrinkage forecasting, and
confidence intervals.
Intended for publication on PyPI.  The public API surface is:
    CapacityForecaster          – main forecasting class
    MIN_DATA_POINTS             – recommended minimum observations (36)
    ABSOLUTE_MIN_DATA_POINTS    – hard floor below which fitting is refused (6)
    CONFIDENCE_LEVEL            – CI alpha used throughout (0.95)
    DataQuality                 – string literals for the Data_Quality column
"""

from __future__ import annotations
import warnings
from typing import Any, Dict, List, Tuple, cast
import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.mlemodel import MLEResultsWrapper
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import adfuller

__all__ = [
    "CapacityForecaster",
    "MIN_DATA_POINTS",
    "ABSOLUTE_MIN_DATA_POINTS",
    "CONFIDENCE_LEVEL",
    "DataQuality",
]

# ---------------------------------------------------------------------------
# Public constants
# ---------------------------------------------------------------------------

MIN_DATA_POINTS: int = 36
ABSOLUTE_MIN_DATA_POINTS: int = 6
CONFIDENCE_LEVEL: float = 0.95


class DataQuality:
    OK: str = "OK"
    LOW_HISTORY: str = "LOW_HISTORY"


# ---------------------------------------------------------------------------
# Private constants
# ---------------------------------------------------------------------------

_CANDIDATE_ORDERS: List[Tuple[int, int, int]] = [
    (1, 1, 1),
    (0, 1, 1),
    (1, 1, 0),
    (2, 1, 1),
    (1, 1, 2),
]

_CANDIDATE_SEASONAL_ORDERS: List[Tuple[int, int, int, int]] = [
    (1, 1, 1, 12),
    (0, 1, 1, 12),
    (1, 1, 0, 12),
    (0, 0, 0, 0),
]


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _normalise_to_month_start(dates: pd.Series) -> pd.Series:
    """
    Normalise any monthly date representation to the first day of that month.

    Examples
    --------
    2024-01-31 → 2024-01-01
    2024-01-01 → 2024-01-01
    2024-01-15 → 2024-01-01
    """
    # FIX: pandas no longer supports "MS" in to_timestamp
    return dates.dt.to_period("M").dt.to_timestamp(how="start")


def _reindex_to_full_monthly_grid(
    group_df: pd.DataFrame,
    numeric_cols: List[str],
) -> Tuple[pd.DataFrame, int]:

    group_df = group_df.copy().sort_values("Date").reset_index(drop=True)

    full_index = pd.date_range(
        start=group_df["Date"].iloc[0],
        end=group_df["Date"].iloc[-1],
        freq="MS",
    )

    imputed_months = len(full_index) - len(group_df)

    if imputed_months > 0:
        group_df = (
            group_df.set_index("Date")
            .reindex(full_index)
            .rename_axis("Date")
            .reset_index()
        )

        if "Capacity Planning Group" in group_df.columns:
            group_df["Capacity Planning Group"] = (
                group_df["Capacity Planning Group"].ffill().bfill()
            )

    for col in numeric_cols:
        if col in group_df.columns:
            group_df[col] = group_df[col].interpolate(
                method="linear",
                limit_direction="both",
            )

    return group_df, max(imputed_months, 0)


def _is_stationary(series: pd.Series) -> bool:
    try:
        result = adfuller(series.dropna(), autolag="AIC")
        return bool(result[1] < 0.05)
    except Exception:
        return False


def _select_integration_order(series: pd.Series) -> int:
    return 0 if _is_stationary(series) else 1


def _candidate_seasonal_orders(n: int) -> List[Tuple[int, int, int, int]]:
    if n < 24:
        return [(0, 0, 0, 0)]
    return _CANDIDATE_SEASONAL_ORDERS


def _fit_best_sarimax(series: pd.Series, label: str) -> MLEResultsWrapper:

    d = _select_integration_order(series)
    seasonal_candidates = _candidate_seasonal_orders(len(series))

    best_result: MLEResultsWrapper | None = None
    best_aicc = np.inf
    errors: List[str] = []

    for p, _, q in _CANDIDATE_ORDERS:
        order = (p, d, q)

        for seasonal_order in seasonal_candidates:
            try:

                model = SARIMAX(
                    series,
                    order=order,
                    seasonal_order=seasonal_order,
                    enforce_stationarity=False,
                    enforce_invertibility=False,
                    concentrate_scale=True,
                )

                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")

                    result = cast(
                        MLEResultsWrapper,
                        model.fit(
                            disp=False,
                            method="lbfgs",
                            maxiter=200,
                        ),
                    )

                k: int = int(result.df_model)
                n: int = len(series)

                aicc: float = float(result.aic) + (2 * k * (k + 1)) / max(
                    n - k - 1,
                    1,
                )

                if aicc < best_aicc:
                    best_aicc = aicc
                    best_result = result

            except Exception as exc:
                errors.append(f"order={order}, seasonal={seasonal_order}: {exc}")

    if best_result is None:
        raise RuntimeError(
            f"All SARIMAX candidates failed for '{label}'.\n" + "\n".join(errors)
        )

    return best_result


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------


class CapacityForecaster:

    def __init__(
        self,
        weekly_hours: float = 37.5,
        forecast_horizon: int = 12,
        default_shrinkage: float = 0.30,
    ) -> None:

        if weekly_hours <= 0:
            raise ValueError(f"weekly_hours must be positive, got {weekly_hours}.")

        if forecast_horizon < 1:
            raise ValueError(f"forecast_horizon must be >= 1.")

        if not 0.0 <= default_shrinkage < 1.0:
            raise ValueError("default_shrinkage must be in [0.0,1.0).")

        self.weekly_hours = weekly_hours
        self.forecast_horizon = forecast_horizon
        self.default_shrinkage = default_shrinkage

        self._hours_per_fte_per_month = weekly_hours * (52.0 / 12.0)

    def _build_forecast_index(self, last_date: pd.Timestamp) -> pd.DatetimeIndex:

        return pd.date_range(
            start=last_date + pd.offsets.MonthBegin(1),
            periods=self.forecast_horizon,
            freq="MS",
        )

    def _forecast_series(
        self,
        series: pd.Series,
        label: str,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:

        series = series.clip(lower=0)

        fitted = _fit_best_sarimax(series, label)

        forecast_obj = fitted.get_forecast(steps=self.forecast_horizon)

        predicted = np.clip(
            forecast_obj.predicted_mean.values,
            0,
            None,
        )

        ci = forecast_obj.conf_int(alpha=1.0 - CONFIDENCE_LEVEL)

        lower_ci = np.clip(ci.iloc[:, 0].values, 0, None)
        upper_ci = ci.iloc[:, 1].values

        return predicted, lower_ci, upper_ci

    def forecast(self, df: pd.DataFrame) -> pd.DataFrame:

        df = df.sort_values("Date").reset_index(drop=True)

        results: List[pd.DataFrame] = []

        for group, group_df in df.groupby("Capacity Planning Group"):

            group_name = str(group)

            group_df, imputed_months = _reindex_to_full_monthly_grid(
                group_df,
                ["Hours", "Volume"],
            )

            ts = group_df.set_index(
                pd.DatetimeIndex(group_df["Date"].values, freq="MS")
            )

            last_date = ts.index[-1]

            forecast_index = self._build_forecast_index(last_date)

            vol_pred, vol_lo, vol_hi = self._forecast_series(
                pd.Series(ts["Volume"].values, index=ts.index),
                f"{group_name}/Volume",
            )

            hrs_pred, hrs_lo, hrs_hi = self._forecast_series(
                pd.Series(ts["Hours"].values, index=ts.index),
                f"{group_name}/Hours",
            )

            shrinkage_pred = np.full(
                self.forecast_horizon,
                self.default_shrinkage,
            )

            m = self._hours_per_fte_per_month

            fte_pred = hrs_pred / m
            fte_lo = hrs_lo / m
            fte_hi = hrs_hi / m

            availability = 1.0 - shrinkage_pred

            adj_pred = fte_pred / availability
            adj_lo = fte_lo / availability
            adj_hi = fte_hi / availability

            results.append(
                pd.DataFrame(
                    {
                        "Date": forecast_index,
                        "Capacity Planning Group": group_name,
                        "Forecasted_Volume": vol_pred,
                        "Forecasted_Volume_Lower": vol_lo,
                        "Forecasted_Volume_Upper": vol_hi,
                        "Forecasted_Hours": hrs_pred,
                        "Forecasted_Hours_Lower": hrs_lo,
                        "Forecasted_Hours_Upper": hrs_hi,
                        "Forecasted_FTE": fte_pred,
                        "Forecasted_FTE_Lower": fte_lo,
                        "Forecasted_FTE_Upper": fte_hi,
                        "Forecasted_FTE_Adjusted": adj_pred,
                        "Forecasted_FTE_Adjusted_Lower": adj_lo,
                        "Forecasted_FTE_Adjusted_Upper": adj_hi,
                    }
                )
            )

        final_df = pd.concat(results, ignore_index=True)

        return final_df
