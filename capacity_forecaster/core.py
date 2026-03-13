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
"""Recommended minimum monthly observations per group for reliable forecasts."""

ABSOLUTE_MIN_DATA_POINTS: int = 6
"""Hard floor — groups with fewer observations than this are skipped entirely."""

CONFIDENCE_LEVEL: float = 0.95
"""Confidence level used for all forecast intervals."""


class DataQuality:
    """
    String literals written to the ``Data_Quality`` output column.

    Attributes
    ----------
    OK : str
        Group had at least ``MIN_DATA_POINTS`` (36) months of history.
    LOW_HISTORY : str
        Group had between ``ABSOLUTE_MIN_DATA_POINTS`` (6) and
        ``MIN_DATA_POINTS`` (35) months.  The forecast was attempted but
        accuracy may be reduced — treat results with caution.
    """

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
    (0, 0, 0, 0),  # Non-seasonal fallback — used automatically for short series
]


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _is_stationary(series: pd.Series) -> bool:
    """ADF test — returns True when the series is already stationary (p < 0.05)."""
    try:
        result = adfuller(series.dropna(), autolag="AIC")
        return bool(result[1] < 0.05)
    except Exception:
        return False


def _select_integration_order(series: pd.Series) -> int:
    """Returns d=0 when the series is stationary, else d=1."""
    return 0 if _is_stationary(series) else 1


def _candidate_seasonal_orders(n: int) -> List[Tuple[int, int, int, int]]:
    """
    Restricts seasonal candidates based on series length.
    Seasonal SARIMAX requires at least two full seasonal cycles (~24 points).
    Shorter series fall back to non-seasonal orders only.
    """
    if n < 24:
        return [(0, 0, 0, 0)]
    return _CANDIDATE_SEASONAL_ORDERS


def _fit_best_sarimax(series: pd.Series, label: str) -> MLEResultsWrapper:
    """
    Grid-searches SARIMAX orders and returns the fitted result with the
    lowest AICc (corrected AIC, preferred for finite samples).

    Parameters
    ----------
    series : pd.Series
        Time series to fit.
    label : str
        Human-readable label used in error messages (e.g. ``"Group/Hours"``).

    Raises
    ------
    RuntimeError
        If every candidate order combination fails to converge.
    """
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
                        model.fit(disp=False, method="lbfgs", maxiter=200),
                    )

                k: int = int(result.df_model)
                n: int = len(series)
                aicc: float = float(result.aic) + (2 * k * (k + 1)) / max(n - k - 1, 1)

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
    """
    Forecasting engine for headcount / FTE capacity planning.

    Produces a forward forecast (default 12 months) for each
    *Capacity Planning Group* in the supplied DataFrame, independently
    forecasting:

    * **Volume**        – units of work expected each month.
    * **Hours**         – total workload hours expected each month.
    * **FTE**           – raw headcount (Hours / hours-per-FTE-per-month).
    * **FTE_Adjusted**  – FTE grossed up for shrinkage (FTE / availability).
    * **Shrinkage**     – forecasted as its own SARIMAX series when historical
                          shrinkage data is supplied; otherwise the
                          ``default_shrinkage`` rate is applied flat.

    All metrics include 95 % confidence-interval bounds.  Each group also
    receives a ``Data_Quality`` flag (``"OK"`` or ``"LOW_HISTORY"``) so
    consumers can identify and caveat less reliable forecasts without
    discarding them entirely.

    Parameters
    ----------
    weekly_hours : float
        Standard contracted weekly hours per FTE.  Defaults to ``37.5``.
    forecast_horizon : int
        Number of months to forecast ahead.  Defaults to ``12``.
    default_shrinkage : float
        Fallback shrinkage rate in ``[0.0, 1.0)`` used when no ``Shrinkage``
        column is present or a group's shrinkage values are all NaN.
        Defaults to ``0.30`` (30 %).

    Input columns
    -------------
    Required
        ``Date``                    – monthly period, any parseable format.
        ``Capacity Planning Group`` – string group identifier.
        ``Hours``                   – numeric workload hours.
        ``Volume``                  – numeric work volume.
    Optional
        ``Shrinkage``               – float in ``[0.0, 1.0)``.  When present,
                                      shrinkage is forecasted per group via
                                      SARIMAX; missing rows fall back to
                                      ``default_shrinkage``.

    Output columns
    --------------
    ``Date``, ``Capacity Planning Group``, ``Data_Quality``,
    ``Forecasted_Shrinkage``,
    ``Forecasted_Volume``, ``Forecasted_Volume_Lower``, ``Forecasted_Volume_Upper``,
    ``Forecasted_Hours``,  ``Forecasted_Hours_Lower``,  ``Forecasted_Hours_Upper``,
    ``Forecasted_FTE``,    ``Forecasted_FTE_Lower``,    ``Forecasted_FTE_Upper``,
    ``Forecasted_FTE_Adjusted``, ``Forecasted_FTE_Adjusted_Lower``,
    ``Forecasted_FTE_Adjusted_Upper``.

    Examples
    --------
    >>> from capacity_forecaster import CapacityForecaster
    >>> forecaster = CapacityForecaster(weekly_hours=37.5, default_shrinkage=0.30)
    >>> results = forecaster.forecast(df)
    >>> reliable = results[results["Data_Quality"] == "OK"]
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
            raise ValueError(f"forecast_horizon must be >= 1, got {forecast_horizon}.")
        if not 0.0 <= default_shrinkage < 1.0:
            raise ValueError(
                f"default_shrinkage must be in [0.0, 1.0), got {default_shrinkage}."
            )

        self.weekly_hours = weekly_hours
        self.forecast_horizon = forecast_horizon
        self.default_shrinkage = default_shrinkage

        # Average hours per FTE per calendar month = weekly_hours x (52 / 12)
        self._hours_per_fte_per_month: float = weekly_hours * (52.0 / 12.0)

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def _validate_dataframe(self, df: object) -> pd.DataFrame:
        """
        Validates and returns a cleaned copy of the input.

        Raises
        ------
        TypeError
            If *df* is not a :class:`pandas.DataFrame`.
        ValueError
            If required columns are missing, the DataFrame is empty, or
            ``Shrinkage`` values are out of range.
        """
        if not isinstance(df, pd.DataFrame):
            raise TypeError(f"Expected pd.DataFrame, got {type(df).__name__}.")

        required = {"Date", "Capacity Planning Group", "Hours", "Volume"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

        if df.empty:
            raise ValueError("Input DataFrame is empty.")

        df = df.copy()

        if not pd.api.types.is_datetime64_any_dtype(df["Date"]):
            df["Date"] = pd.to_datetime(df["Date"], format="mixed")

        if "Shrinkage" in df.columns:
            bad = df["Shrinkage"].dropna()
            bad = bad[(bad < 0.0) | (bad >= 1.0)]
            if not bad.empty:
                raise ValueError(
                    f"Shrinkage values must be in [0.0, 1.0). "
                    f"Out-of-range values found: {bad.unique().tolist()}"
                )

        initial = len(df)
        df = df.dropna(subset=["Hours", "Volume"]).reset_index(drop=True)
        dropped = initial - len(df)
        if dropped:
            warnings.warn(
                f"Dropped {dropped} row(s) with NaN in 'Hours' or 'Volume'.",
                UserWarning,
                stacklevel=3,
            )

        return df

    # ------------------------------------------------------------------
    # Internal forecasting helpers
    # ------------------------------------------------------------------

    def _build_forecast_index(self, last_date: pd.Timestamp) -> pd.DatetimeIndex:
        """Monthly DatetimeIndex for the forecast period."""
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
        """
        Fits SARIMAX to *series* and returns ``(predicted, lower_ci, upper_ci)``.
        All values are clipped to >= 0.
        """
        series = series.clip(lower=0)
        fitted = _fit_best_sarimax(series, label)

        forecast_obj = fitted.get_forecast(steps=self.forecast_horizon)
        predicted: np.ndarray = np.clip(forecast_obj.predicted_mean.values, 0, None)
        ci = forecast_obj.conf_int(alpha=1.0 - CONFIDENCE_LEVEL)
        lower_ci: np.ndarray = np.clip(ci.iloc[:, 0].values, 0, None)
        upper_ci: np.ndarray = ci.iloc[:, 1].values

        return predicted, lower_ci, upper_ci

    def _forecast_shrinkage(self, series: pd.Series, label: str) -> np.ndarray:
        """
        Forecasts shrinkage via logit-transformed SARIMAX.

        The logit transform maps ``(0, 1)`` to ``(-inf, +inf)`` so SARIMAX
        operates on an unbounded series, preventing out-of-range predictions.
        Results are inverse-logit transformed back to ``(0, 1)`` and clipped
        to ``[0.05, 0.95]`` for numerical safety.

        Returns point forecasts only — shrinkage CI bounds are not surfaced
        because the adjusted FTE CI already captures workload uncertainty.
        """
        clipped = series.clip(lower=0.05, upper=0.95)
        logit = pd.Series(
            np.log(clipped / (1.0 - clipped)),
            index=series.index,
            name="Shrinkage_logit",
        )
        fitted = _fit_best_sarimax(logit, label)
        logit_pred = fitted.get_forecast(
            steps=self.forecast_horizon
        ).predicted_mean.values
        return np.clip(1.0 / (1.0 + np.exp(-logit_pred)), 0.05, 0.95)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def forecast(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Produce a capacity forecast for every group in *df*.

        Groups with fewer than ``ABSOLUTE_MIN_DATA_POINTS`` (6) observations
        are skipped entirely with a warning.  Groups with between 6 and 35
        observations are forecasted but flagged ``Data_Quality = "LOW_HISTORY"``
        in the output — consumers should treat these results with caution.

        Parameters
        ----------
        df : pd.DataFrame
            See class docstring for required and optional columns.

        Returns
        -------
        pd.DataFrame
            See class docstring for output columns.

        Raises
        ------
        RuntimeError
            If no groups produced a forecast (all were skipped or all failed).
        """
        df = self._validate_dataframe(df)
        df = df.sort_values("Date").reset_index(drop=True)

        has_shrinkage_col = "Shrinkage" in df.columns
        results: List[pd.DataFrame] = []
        skipped: List[str] = []

        for group, group_df in df.groupby("Capacity Planning Group"):
            group_name: str = str(group)
            n = len(group_df)

            # Hard floor — impossible to fit any model below this
            if n < ABSOLUTE_MIN_DATA_POINTS:
                warnings.warn(
                    f"Group '{group_name}' has only {n} observation(s). "
                    f"Minimum required to attempt a forecast: {ABSOLUTE_MIN_DATA_POINTS}. "
                    "Skipping.",
                    UserWarning,
                    stacklevel=2,
                )
                skipped.append(group_name)
                continue

            # Soft threshold — forecast proceeds but confidence is reduced
            data_quality = (
                DataQuality.OK if n >= MIN_DATA_POINTS else DataQuality.LOW_HISTORY
            )
            if data_quality == DataQuality.LOW_HISTORY:
                warnings.warn(
                    f"Group '{group_name}' has only {n} observation(s) "
                    f"(recommended minimum: {MIN_DATA_POINTS}). "
                    "Forecast attempted — results flagged as "
                    f"'{DataQuality.LOW_HISTORY}'. Treat with caution.",
                    UserWarning,
                    stacklevel=2,
                )

            ts = group_df.set_index(
                pd.DatetimeIndex(group_df["Date"].values, freq="MS")
            )
            last_date: pd.Timestamp = ts.index[-1]  # type: ignore[assignment]
            forecast_index = self._build_forecast_index(last_date)

            # --- Volume ---
            vol_pred, vol_lo, vol_hi = self._forecast_series(
                pd.Series(ts["Volume"].values, index=ts.index, name="Volume"),
                f"{group_name}/Volume",
            )

            # --- Hours ---
            hrs_pred, hrs_lo, hrs_hi = self._forecast_series(
                pd.Series(ts["Hours"].values, index=ts.index, name="Hours"),
                f"{group_name}/Hours",
            )

            # --- Shrinkage ---
            if has_shrinkage_col:
                valid = group_df["Shrinkage"].dropna()
                if not valid.empty:
                    shrinkage_pred = self._forecast_shrinkage(
                        pd.Series(
                            ts["Shrinkage"].values, index=ts.index, name="Shrinkage"
                        ),
                        f"{group_name}/Shrinkage",
                    )
                else:
                    shrinkage_pred = np.full(
                        self.forecast_horizon, self.default_shrinkage
                    )
            else:
                shrinkage_pred = np.full(self.forecast_horizon, self.default_shrinkage)

            # --- Raw FTE ---
            m = self._hours_per_fte_per_month
            fte_pred = hrs_pred / m
            fte_lo = hrs_lo / m
            fte_hi = hrs_hi / m

            # --- Shrinkage-adjusted FTE ---
            availability = 1.0 - shrinkage_pred
            adj_pred = fte_pred / availability
            adj_lo = fte_lo / availability
            adj_hi = fte_hi / availability

            results.append(
                pd.DataFrame(
                    {
                        "Date": forecast_index,
                        "Capacity Planning Group": group_name,
                        "Data_Quality": data_quality,
                        "Forecasted_Shrinkage": np.round(shrinkage_pred, 4),
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

        if not results:
            raise RuntimeError(
                "No groups produced a forecast. "
                f"Skipped: {skipped}. "
                f"Hard minimum observations required: {ABSOLUTE_MIN_DATA_POINTS}."
            )

        final_df = pd.concat(results, ignore_index=True)

        for col in final_df.columns:
            if "FTE" in col:
                final_df[col] = final_df[col].round(2)
            elif "Volume" in col or "Hours" in col:
                final_df[col] = final_df[col].round(0)

        return final_df

    def diagnostic_summary(self, df: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
        """
        Returns per-group model metadata without running a full forecast.

        Useful for inspecting which SARIMAX orders were selected, AICc
        scores, and stationarity results for each series.

        Parameters
        ----------
        df : pd.DataFrame
            Same format as :meth:`forecast`.

        Returns
        -------
        dict[str, dict]
            Keys are group names.  Each value contains:

            - ``n_obs``           – number of historical observations.
            - ``data_quality``    – ``"OK"`` or ``"LOW_HISTORY"``.
            - ``hours_model``     – fitted model metadata for Hours.
            - ``volume_model``    – fitted model metadata for Volume.
            - ``shrinkage_model`` – fitted model metadata for Shrinkage
                                    (only present when a Shrinkage column exists).
        """
        df = self._validate_dataframe(df)
        df = df.sort_values("Date").reset_index(drop=True)

        has_shrinkage_col = "Shrinkage" in df.columns
        summary: Dict[str, Dict[str, Any]] = {}

        for group, group_df in df.groupby("Capacity Planning Group"):
            group_name: str = str(group)
            n = len(group_df)

            if n < ABSOLUTE_MIN_DATA_POINTS:
                summary[group_name] = {
                    "status": f"skipped — fewer than {ABSOLUTE_MIN_DATA_POINTS} observations"
                }
                continue

            ts = group_df.set_index(
                pd.DatetimeIndex(group_df["Date"].values, freq="MS")
            )
            data_quality = (
                DataQuality.OK if n >= MIN_DATA_POINTS else DataQuality.LOW_HISTORY
            )
            group_info: Dict[str, Any] = {"n_obs": n, "data_quality": data_quality}

            series_to_diagnose: List[str] = ["Hours", "Volume"]
            if has_shrinkage_col:
                series_to_diagnose.append("Shrinkage")

            for series_name in series_to_diagnose:
                if series_name == "Shrinkage":
                    raw = (
                        pd.Series(ts[series_name].values, index=ts.index)
                        .dropna()
                        .clip(lower=0.05, upper=0.95)
                    )
                    series = pd.Series(
                        np.log(raw / (1.0 - raw)),
                        index=raw.index,
                        name="Shrinkage_logit",
                    )
                else:
                    series = pd.Series(
                        ts[series_name].values, index=ts.index, name=series_name
                    ).clip(lower=0)

                try:
                    fitted: MLEResultsWrapper = _fit_best_sarimax(
                        series, f"{group_name}/{series_name}"
                    )
                    k: int = int(fitted.df_model)
                    aic: float = float(fitted.aic)
                    aicc: float = aic + (2 * k * (k + 1)) / max(n - k - 1, 1)
                    sarimax_model: SARIMAX = fitted.model  # type: ignore[assignment]
                    group_info[f"{series_name.lower()}_model"] = {
                        "order": sarimax_model.order,
                        "seasonal_order": sarimax_model.seasonal_order,
                        "aic": round(aic, 2),
                        "aicc": round(aicc, 2),
                        "stationary": _is_stationary(series),
                    }
                except RuntimeError as exc:
                    group_info[f"{series_name.lower()}_model"] = {"error": str(exc)}

            summary[group_name] = group_info

        return summary
