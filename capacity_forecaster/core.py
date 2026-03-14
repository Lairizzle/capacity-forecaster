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

Input DataFrame column order
-----------------------------
Columns are resolved **by position**, so names do not matter.  The caller's
original column names are preserved in the output.

    col 0  –  Date              (datetime-like; month-start or month-end)
    col 1  –  Group             (any hashable label, e.g. team / queue name)
    col 2  –  Volume            (numeric)
    col 3  –  Hours             (numeric)
    col 4  –  Shrinkage         (numeric, optional; values in [0.0, 1.0))
"""

from __future__ import annotations

import warnings
from typing import cast

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

# Canonical internal column names – used only inside this module so that the
# pipeline is fully independent of whatever names the caller uses.
_COL_DATE = "__date__"
_COL_GROUP = "__group__"
_COL_VOLUME = "__volume__"
_COL_HOURS = "__hours__"
_COL_SHRINKAGE = "__shrinkage__"

_CANDIDATE_ORDERS: list[tuple[int, int, int]] = [
    (1, 1, 1),
    (0, 1, 1),
    (1, 1, 0),
    (2, 1, 1),
    (1, 1, 2),
]
_CANDIDATE_SEASONAL_ORDERS: list[tuple[int, int, int, int]] = [
    (1, 1, 1, 12),
    (0, 1, 1, 12),
    (1, 1, 0, 12),
    (0, 0, 0, 0),
]


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _resolve_columns(
    df: pd.DataFrame,
) -> tuple[str, str, str, str, str | None]:
    """
    Return the caller's actual column names at positions 0–4.

    Position 4 (shrinkage) is optional; ``None`` is returned when absent.

    Raises
    ------
    ValueError
        If fewer than four columns are present, column 0 cannot be parsed as
        dates, or the shrinkage column contains values outside [0.0, 1.0).
    """
    if df.shape[1] < 4:
        raise ValueError(
            f"Input DataFrame must have at least 4 columns "
            f"(date, group, volume, hours) but only {df.shape[1]} were found."
        )

    col_date, col_group, col_volume, col_hours = df.columns[:4]

    if not pd.api.types.is_datetime64_any_dtype(df[col_date]):
        try:
            pd.to_datetime(df[col_date])
        except Exception as exc:
            raise ValueError(
                f"Column at position 0 ('{col_date}') could not be parsed as "
                "dates. Ensure it is a datetime column."
            ) from exc

    col_shrinkage: str | None = None
    if df.shape[1] > 4:
        col_shrinkage = df.columns[4]
        out_of_range = df[col_shrinkage].dropna()
        if ((out_of_range < 0.0) | (out_of_range >= 1.0)).any():
            raise ValueError(
                f"Shrinkage column ('{col_shrinkage}', position 4) must "
                "contain values in [0.0, 1.0). Found out-of-range entries."
            )

    return col_date, col_group, col_volume, col_hours, col_shrinkage


def _normalise_to_month_start(dates: pd.Series) -> pd.Series:
    """
    Coerce any monthly date series to the first of each month.
    Examples: 2024-01-31 → 2024-01-01, 2024-01-15 → 2024-01-01
    """
    return dates.dt.to_period("M").dt.to_timestamp()


def _make_monthly_series(values: np.ndarray, dates: pd.DatetimeIndex) -> pd.Series:
    """
    Wrap *values* in a Series with a month-start ``DatetimeIndex``.
    """
    idx = pd.date_range(start=dates[0], periods=len(dates), freq="MS")
    return pd.Series(values, index=idx)


def _reindex_to_full_monthly_grid(
    group_df: pd.DataFrame,
    numeric_cols: list[str],
    group_col: str,
) -> tuple[pd.DataFrame, int]:
    """
    Guarantee *group_df* has exactly one row per calendar month.

    Duplicate months (from mixed month-start/end inputs collapsed during
    normalisation) are aggregated by summing numeric columns and keeping the
    first group label.  Missing months are inserted and their numeric columns
    filled by linear interpolation.

    Parameters
    ----------
    group_df     : DataFrame for a single group (dates already normalised).
    numeric_cols : Canonical column names to interpolate.
    group_col    : Canonical column name carrying the group label.

    Returns
    -------
    (filled_df, imputed_months)
    """
    group_df = group_df.copy()

    agg: dict[str, str] = {col: "sum" for col in numeric_cols}
    agg[group_col] = "first"
    group_df = (
        group_df.groupby(_COL_DATE, as_index=False)
        .agg(agg)
        .sort_values(_COL_DATE)
        .reset_index(drop=True)
    )

    full_index = pd.date_range(
        start=group_df[_COL_DATE].iloc[0],
        end=group_df[_COL_DATE].iloc[-1],
        freq="MS",
    )
    imputed_months = len(full_index) - len(group_df)

    if imputed_months > 0:
        group_df = (
            group_df.set_index(_COL_DATE)
            .reindex(full_index)
            .rename_axis(_COL_DATE)
            .reset_index()
        )
        group_df[group_col] = group_df[group_col].ffill().bfill()

    for col in numeric_cols:
        group_df[col] = group_df[col].interpolate(
            method="linear", limit_direction="both"
        )

    return group_df, imputed_months


def _is_stationary(series: pd.Series) -> bool:
    try:
        return bool(adfuller(series.dropna(), autolag="AIC")[1] < 0.05)
    except Exception:
        return False


def _fit_best_sarimax(series: pd.Series, label: str) -> MLEResultsWrapper:
    """Select and fit the best SARIMAX model by AICc."""
    d = 0 if _is_stationary(series) else 1
    seasonal_candidates = (
        [(0, 0, 0, 0)] if len(series) < 24 else _CANDIDATE_SEASONAL_ORDERS
    )
    best_result: MLEResultsWrapper | None = None
    best_aicc = np.inf
    errors: list[str] = []

    for p, _, q in _CANDIDATE_ORDERS:
        for seasonal_order in seasonal_candidates:
            try:
                model = SARIMAX(
                    series,
                    order=(p, d, q),
                    seasonal_order=seasonal_order,
                    enforce_stationarity=False,
                    enforce_invertibility=False,
                    concentrate_scale=True,
                )
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    result = cast(
                        MLEResultsWrapper,
                        model.fit(disp=False, method="lbfgs", maxiter=200),
                    )
                k = int(result.df_model)
                n = len(series)
                aicc = float(result.aic) + (2 * k * (k + 1)) / max(n - k - 1, 1)
                if aicc < best_aicc:
                    best_aicc = aicc
                    best_result = result
            except Exception as exc:
                errors.append(f"order=({p},{d},{q}), seasonal={seasonal_order}: {exc}")

    if best_result is None:
        raise RuntimeError(
            f"All SARIMAX candidates failed for '{label}'.\n" + "\n".join(errors)
        )
    return best_result


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------


class CapacityForecaster:
    """
    Fit SARIMAX models per group and project volume, hours, FTE, and
    shrinkage-adjusted FTE over a configurable forecast horizon.
    """

    def __init__(
        self,
        weekly_hours: float = 37.5,
        forecast_horizon: int = 12,
        default_shrinkage: float = 0.30,
    ) -> None:
        if weekly_hours <= 0:
            raise ValueError(f"weekly_hours must be positive, got {weekly_hours}.")
        if forecast_horizon < 1:
            raise ValueError("forecast_horizon must be >= 1.")
        if not 0.0 <= default_shrinkage < 1.0:
            raise ValueError("default_shrinkage must be in [0.0, 1.0).")

        self.weekly_hours = weekly_hours
        self.forecast_horizon = forecast_horizon
        self.default_shrinkage = default_shrinkage
        self._hours_per_fte_per_month: float = weekly_hours * (52.0 / 12.0)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _forecast_series(
        self, series: pd.Series, label: str
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return (predicted, lower_ci, upper_ci) clipped to >= 0."""
        fitted = _fit_best_sarimax(series.clip(lower=0), label)
        forecast_obj = fitted.get_forecast(steps=self.forecast_horizon)
        predicted = np.clip(forecast_obj.predicted_mean.to_numpy(), 0.0, None)
        ci = forecast_obj.conf_int(alpha=1.0 - CONFIDENCE_LEVEL)
        lower_ci = np.clip(ci.iloc[:, 0].to_numpy(), 0.0, None)
        upper_ci = ci.iloc[:, 1].to_numpy()
        return predicted, lower_ci, upper_ci

    def _shrinkage_forecast(self, shrink_series: pd.Series, label: str) -> np.ndarray:
        """
        Forecast shrinkage for the horizon.

        Strategy (in order of preference):
        1. SARIMAX forecast when >= ABSOLUTE_MIN_DATA_POINTS valid points exist.
        2. Historical group mean when fewer valid points exist.
        3. ``default_shrinkage`` when no valid points exist at all.

        The result is clipped to [0.0, 0.9999] to keep availability positive.
        """
        n_valid = int(shrink_series.notna().sum())
        fallback = (
            float(shrink_series.mean()) if n_valid > 0 else self.default_shrinkage
        )
        if n_valid >= ABSOLUTE_MIN_DATA_POINTS:
            try:
                pred, _, _ = self._forecast_series(
                    shrink_series.fillna(fallback), label
                )
                return np.clip(pred, 0.0, 0.9999)
            except Exception:
                pass
        return np.full(self.forecast_horizon, fallback)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def forecast(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Produce a monthly capacity forecast for every group in *df*.

        Parameters
        ----------
        df : pd.DataFrame
            At least four columns **in this order** (names are ignored):

            =========  =====================================================
            Position   Role
            =========  =====================================================
            0          Date       – datetime (month-start or month-end)
            1          Group      – grouping label
            2          Volume     – numeric workload volume
            3          Hours      – numeric hours worked / required
            4          Shrinkage  – *optional* float in [0.0, 1.0) per row
            =========  =====================================================

            When column 4 is present its historical values are forecasted
            forward with SARIMAX.  Any gaps are linearly interpolated; the
            fallback chain is: historical mean → ``default_shrinkage``.
            When column 4 is absent, ``default_shrinkage`` is used uniformly.

        Returns
        -------
        pd.DataFrame
            One row per (group × forecast month).  The group column retains
            the caller's original name.  Output columns:

            ``Date``, ``<group col>``,
            ``Forecasted_Volume`` / ``_Lower`` / ``_Upper``,
            ``Forecasted_Hours`` / ``_Lower`` / ``_Upper``,
            ``Forecasted_FTE`` / ``_Lower`` / ``_Upper``,
            ``Forecasted_FTE_Adjusted`` / ``_Lower`` / ``_Upper``,
            ``Shrinkage_Used``, ``Imputed_Months``, ``Data_Quality``
        """
        # 1. Resolve caller column names by position.
        orig_date, orig_group, orig_vol, orig_hrs, orig_shrink = _resolve_columns(df)
        has_shrinkage = orig_shrink is not None

        # 2. Rename to canonical sentinels for the rest of the pipeline.
        rename_map = {
            orig_date: _COL_DATE,
            orig_group: _COL_GROUP,
            orig_vol: _COL_VOLUME,
            orig_hrs: _COL_HOURS,
        }
        if has_shrinkage:
            rename_map[orig_shrink] = _COL_SHRINKAGE

        df = df.rename(columns=rename_map).copy()

        # 3. Normalise all dates to month-start and sort.
        df[_COL_DATE] = _normalise_to_month_start(df[_COL_DATE])
        df = df.sort_values(_COL_DATE).reset_index(drop=True)

        # 4. Forecast per group.
        numeric_cols = [_COL_VOLUME, _COL_HOURS]
        if has_shrinkage:
            numeric_cols.append(_COL_SHRINKAGE)

        results: list[pd.DataFrame] = []

        for group, group_df in df.groupby(_COL_GROUP):
            group_name = str(group)

            group_df, imputed_months = _reindex_to_full_monthly_grid(
                group_df, numeric_cols=numeric_cols, group_col=_COL_GROUP
            )

            dates = pd.DatetimeIndex(group_df[_COL_DATE].values)
            forecast_index = pd.date_range(
                start=dates[-1] + pd.offsets.MonthBegin(1),
                periods=self.forecast_horizon,
                freq="MS",
            )

            vol_series = _make_monthly_series(group_df[_COL_VOLUME].to_numpy(), dates)
            hrs_series = _make_monthly_series(group_df[_COL_HOURS].to_numpy(), dates)

            vol_pred, vol_lo, vol_hi = self._forecast_series(
                vol_series, f"{group_name}/volume"
            )
            hrs_pred, hrs_lo, hrs_hi = self._forecast_series(
                hrs_series, f"{group_name}/hours"
            )

            if has_shrinkage:
                shrink_series = _make_monthly_series(
                    group_df[_COL_SHRINKAGE].to_numpy(), dates
                )
                shrinkage_pred = self._shrinkage_forecast(
                    shrink_series, f"{group_name}/shrinkage"
                )
            else:
                shrinkage_pred = np.full(self.forecast_horizon, self.default_shrinkage)

            m = self._hours_per_fte_per_month
            availability = 1.0 - shrinkage_pred
            fte_pred = hrs_pred / m
            fte_lo = hrs_lo / m
            fte_hi = hrs_hi / m

            results.append(
                pd.DataFrame(
                    {
                        "Date": forecast_index,
                        orig_group: group_name,
                        "Forecasted_Volume": vol_pred,
                        "Forecasted_Volume_Lower": vol_lo,
                        "Forecasted_Volume_Upper": vol_hi,
                        "Forecasted_Hours": hrs_pred,
                        "Forecasted_Hours_Lower": hrs_lo,
                        "Forecasted_Hours_Upper": hrs_hi,
                        "Forecasted_FTE": fte_pred,
                        "Forecasted_FTE_Lower": fte_lo,
                        "Forecasted_FTE_Upper": fte_hi,
                        "Forecasted_FTE_Adjusted": fte_pred / availability,
                        "Forecasted_FTE_Adjusted_Lower": fte_lo / availability,
                        "Forecasted_FTE_Adjusted_Upper": fte_hi / availability,
                        "Shrinkage_Used": shrinkage_pred,
                        "Imputed_Months": imputed_months,
                        "Data_Quality": (
                            DataQuality.LOW_HISTORY
                            if len(group_df) < MIN_DATA_POINTS
                            else DataQuality.OK
                        ),
                    }
                )
            )

        return pd.concat(results, ignore_index=True)
