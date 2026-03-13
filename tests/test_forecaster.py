"""
tests/test_forecaster.py
========================
Pytest test suite for capacity_forecaster.

Run with:
    pytest tests/test_forecaster.py -v
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

from capacity_forecaster import (
    ABSOLUTE_MIN_DATA_POINTS,
    CONFIDENCE_LEVEL,
    MIN_DATA_POINTS,
    CapacityForecaster,
    DataQuality,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

EXPECTED_OUTPUT_COLUMNS = [
    "Date",
    "Capacity Planning Group",
    "Data_Quality",
    "Forecasted_Shrinkage",
    "Forecasted_Volume",
    "Forecasted_Volume_Lower",
    "Forecasted_Volume_Upper",
    "Forecasted_Hours",
    "Forecasted_Hours_Lower",
    "Forecasted_Hours_Upper",
    "Forecasted_FTE",
    "Forecasted_FTE_Lower",
    "Forecasted_FTE_Upper",
    "Forecasted_FTE_Adjusted",
    "Forecasted_FTE_Adjusted_Lower",
    "Forecasted_FTE_Adjusted_Upper",
]


def _make_df(
    n_months: int,
    groups: list[str] | None = None,
    include_shrinkage: bool = False,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Builds a minimal but realistic synthetic DataFrame for testing.
    Uses a fixed seed for reproducibility.
    """
    rng = np.random.default_rng(seed)
    groups = groups or ["Group A"]
    dates = pd.date_range("2020-01-01", periods=n_months, freq="MS")
    rows = []
    for group in groups:
        base_hours = 3000 + rng.integers(0, 500)
        base_volume = 10000 + rng.integers(0, 2000)
        for i, date in enumerate(dates):
            trend = 1.0 + (i / max(n_months, 1)) * 0.1
            row = {
                "Date": date,
                "Capacity Planning Group": group,
                "Hours": int(base_hours * trend * rng.normal(1.0, 0.03)),
                "Volume": int(base_volume * trend * rng.normal(1.0, 0.03)),
            }
            if include_shrinkage:
                row["Shrinkage"] = round(
                    float(np.clip(rng.normal(0.25, 0.02), 0.05, 0.60)), 4
                )
            rows.append(row)
    return pd.DataFrame(rows)


@pytest.fixture
def ok_df() -> pd.DataFrame:
    """Single group with MIN_DATA_POINTS months — Data_Quality = OK."""
    return _make_df(MIN_DATA_POINTS, groups=["Alpha"])


@pytest.fixture
def ok_df_with_shrinkage() -> pd.DataFrame:
    """Single group with MIN_DATA_POINTS months and a Shrinkage column."""
    return _make_df(MIN_DATA_POINTS, groups=["Alpha"], include_shrinkage=True)


@pytest.fixture
def low_history_df() -> pd.DataFrame:
    """Single group with 12 months — Data_Quality = LOW_HISTORY."""
    return _make_df(12, groups=["Beta"])


@pytest.fixture
def multi_group_df() -> pd.DataFrame:
    """Three groups: one OK, one LOW_HISTORY, one below hard floor (skipped)."""
    ok = _make_df(MIN_DATA_POINTS, groups=["OK Group"], seed=1)
    low = _make_df(12, groups=["Low Group"], seed=2)
    tiny = _make_df(3, groups=["Tiny Group"], seed=3)
    return pd.concat([ok, low, tiny], ignore_index=True)


@pytest.fixture
def forecaster() -> CapacityForecaster:
    return CapacityForecaster(
        weekly_hours=37.5, forecast_horizon=12, default_shrinkage=0.30
    )


# ---------------------------------------------------------------------------
# 1. Initialisation
# ---------------------------------------------------------------------------


class TestInit:
    def test_defaults(self) -> None:
        f = CapacityForecaster()
        assert f.weekly_hours == 37.5
        assert f.forecast_horizon == 12
        assert f.default_shrinkage == 0.30

    def test_custom_params(self) -> None:
        f = CapacityForecaster(
            weekly_hours=40.0, forecast_horizon=6, default_shrinkage=0.20
        )
        assert f.weekly_hours == 40.0
        assert f.forecast_horizon == 6
        assert f.default_shrinkage == 0.20

    def test_hours_per_fte_derived_correctly(self) -> None:
        f = CapacityForecaster(weekly_hours=37.5)
        expected = 37.5 * (52.0 / 12.0)
        assert abs(f._hours_per_fte_per_month - expected) < 1e-9

    def test_invalid_weekly_hours(self) -> None:
        with pytest.raises(ValueError, match="weekly_hours"):
            CapacityForecaster(weekly_hours=0)

    def test_invalid_forecast_horizon(self) -> None:
        with pytest.raises(ValueError, match="forecast_horizon"):
            CapacityForecaster(forecast_horizon=0)

    def test_invalid_shrinkage_above_one(self) -> None:
        with pytest.raises(ValueError, match="default_shrinkage"):
            CapacityForecaster(default_shrinkage=1.0)

    def test_invalid_shrinkage_negative(self) -> None:
        with pytest.raises(ValueError, match="default_shrinkage"):
            CapacityForecaster(default_shrinkage=-0.1)

    def test_zero_shrinkage_is_valid(self) -> None:
        f = CapacityForecaster(default_shrinkage=0.0)
        assert f.default_shrinkage == 0.0


# ---------------------------------------------------------------------------
# 2. Input validation
# ---------------------------------------------------------------------------


class TestValidation:
    def test_non_dataframe_raises_type_error(
        self, forecaster: CapacityForecaster
    ) -> None:
        with pytest.raises(TypeError, match="pd.DataFrame"):
            forecaster.forecast({"Date": [], "Capacity Planning Group": []})  # type: ignore

    def test_missing_required_column_raises(
        self, forecaster: CapacityForecaster
    ) -> None:
        df = _make_df(MIN_DATA_POINTS).drop(columns=["Hours"])
        with pytest.raises(ValueError, match="Missing required columns"):
            forecaster.forecast(df)

    def test_empty_dataframe_raises(self, forecaster: CapacityForecaster) -> None:
        df = pd.DataFrame(
            columns=["Date", "Capacity Planning Group", "Hours", "Volume"]
        )
        with pytest.raises(ValueError, match="empty"):
            forecaster.forecast(df)

    def test_string_dates_are_coerced(
        self, forecaster: CapacityForecaster, ok_df: pd.DataFrame
    ) -> None:
        ok_df["Date"] = ok_df["Date"].dt.strftime("%Y-%m-%d")
        result = forecaster.forecast(ok_df)
        assert pd.api.types.is_datetime64_any_dtype(result["Date"])

    def test_nan_rows_are_dropped_with_warning(
        self, forecaster: CapacityForecaster, ok_df: pd.DataFrame
    ) -> None:
        ok_df.loc[0, "Hours"] = np.nan
        with pytest.warns(UserWarning, match="Dropped"):
            result = forecaster.forecast(ok_df)
        assert len(result) == forecaster.forecast_horizon

    def test_invalid_shrinkage_value_raises(
        self, forecaster: CapacityForecaster, ok_df: pd.DataFrame
    ) -> None:
        ok_df["Shrinkage"] = 0.25
        ok_df.loc[0, "Shrinkage"] = 1.5  # out of range
        with pytest.raises(ValueError, match="Shrinkage values must be in"):
            forecaster.forecast(ok_df)

    def test_shrinkage_of_zero_is_valid(self) -> None:
        # Zero is a valid default_shrinkage — just validates the constructor accepts it.
        # A Shrinkage column of all-zeros would be a flat constant series that SARIMAX
        # cannot fit; that scenario is not what this test covers.
        f = CapacityForecaster(default_shrinkage=0.0)
        assert f.default_shrinkage == 0.0


# ---------------------------------------------------------------------------
# 3. Output schema
# ---------------------------------------------------------------------------


class TestOutputSchema:
    def test_all_expected_columns_present(
        self, forecaster: CapacityForecaster, ok_df: pd.DataFrame
    ) -> None:
        result = forecaster.forecast(ok_df)
        for col in EXPECTED_OUTPUT_COLUMNS:
            assert col in result.columns, f"Missing column: {col}"

    def test_row_count_equals_horizon_times_groups(
        self, forecaster: CapacityForecaster, ok_df: pd.DataFrame
    ) -> None:
        result = forecaster.forecast(ok_df)
        assert len(result) == forecaster.forecast_horizon

    def test_dates_are_sequential_month_starts(
        self, forecaster: CapacityForecaster, ok_df: pd.DataFrame
    ) -> None:
        result = forecaster.forecast(ok_df)
        dates = pd.to_datetime(result["Date"])
        diffs = dates.diff().dropna()
        # All diffs should be approximately one month (28–31 days)
        assert (diffs.dt.days >= 28).all()
        assert (diffs.dt.days <= 31).all()

    def test_forecast_starts_month_after_last_historical(
        self, forecaster: CapacityForecaster, ok_df: pd.DataFrame
    ) -> None:
        last_historical = ok_df["Date"].max()
        result = forecaster.forecast(ok_df)
        first_forecast = result["Date"].min()
        assert first_forecast == last_historical + pd.offsets.MonthBegin(1)

    def test_multi_group_row_count(self, forecaster: CapacityForecaster) -> None:
        df = _make_df(MIN_DATA_POINTS, groups=["A", "B"])
        result = forecaster.forecast(df)
        assert len(result) == forecaster.forecast_horizon * 2


# ---------------------------------------------------------------------------
# 4. Data quality flags
# ---------------------------------------------------------------------------


class TestDataQuality:
    def test_ok_flag_for_sufficient_history(
        self, forecaster: CapacityForecaster, ok_df: pd.DataFrame
    ) -> None:
        result = forecaster.forecast(ok_df)
        assert (result["Data_Quality"] == DataQuality.OK).all()

    def test_low_history_flag_for_short_series(
        self, forecaster: CapacityForecaster, low_history_df: pd.DataFrame
    ) -> None:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            result = forecaster.forecast(low_history_df)
        assert (result["Data_Quality"] == DataQuality.LOW_HISTORY).all()

    def test_low_history_emits_warning(
        self, forecaster: CapacityForecaster, low_history_df: pd.DataFrame
    ) -> None:
        with pytest.warns(UserWarning, match="LOW_HISTORY"):
            forecaster.forecast(low_history_df)

    def test_below_hard_floor_is_skipped_with_warning(
        self, forecaster: CapacityForecaster
    ) -> None:
        tiny = _make_df(ABSOLUTE_MIN_DATA_POINTS - 1, groups=["Tiny"])
        ok = _make_df(MIN_DATA_POINTS, groups=["OK"])
        df = pd.concat([tiny, ok], ignore_index=True)
        with pytest.warns(UserWarning, match="Skipping"):
            result = forecaster.forecast(df)
        assert "Tiny" not in result["Capacity Planning Group"].values
        assert "OK" in result["Capacity Planning Group"].values

    def test_all_skipped_raises_runtime_error(
        self, forecaster: CapacityForecaster
    ) -> None:
        tiny = _make_df(ABSOLUTE_MIN_DATA_POINTS - 1, groups=["Tiny"])
        with pytest.raises(RuntimeError, match="No groups produced a forecast"):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                forecaster.forecast(tiny)

    def test_data_quality_constants(self) -> None:
        assert DataQuality.OK == "OK"
        assert DataQuality.LOW_HISTORY == "LOW_HISTORY"


# ---------------------------------------------------------------------------
# 5. Numeric output sanity
# ---------------------------------------------------------------------------


class TestNumericOutput:
    def test_all_forecast_values_non_negative(
        self, forecaster: CapacityForecaster, ok_df: pd.DataFrame
    ) -> None:
        result = forecaster.forecast(ok_df)
        numeric_cols = [c for c in result.columns if c.startswith("Forecasted_")]
        for col in numeric_cols:
            assert (result[col] >= 0).all(), f"Negative values in {col}"

    def test_lower_ci_lte_predicted_lte_upper_ci(
        self, forecaster: CapacityForecaster, ok_df: pd.DataFrame
    ) -> None:
        result = forecaster.forecast(ok_df)
        for metric in ("Volume", "Hours", "FTE", "FTE_Adjusted"):
            lo = result[f"Forecasted_{metric}_Lower"]
            mid = result[f"Forecasted_{metric}"]
            hi = result[f"Forecasted_{metric}_Upper"]
            assert (lo <= mid).all(), f"{metric}: lower CI exceeds point forecast"
            assert (mid <= hi).all(), f"{metric}: point forecast exceeds upper CI"

    def test_adjusted_fte_greater_than_raw_fte_with_nonzero_shrinkage(
        self, forecaster: CapacityForecaster, ok_df: pd.DataFrame
    ) -> None:
        # default_shrinkage = 0.30, no Shrinkage column → flat default applied
        result = forecaster.forecast(ok_df)
        assert (result["Forecasted_FTE_Adjusted"] > result["Forecasted_FTE"]).all()

    def test_adjusted_fte_equals_raw_fte_when_shrinkage_zero(
        self, ok_df: pd.DataFrame
    ) -> None:
        f = CapacityForecaster(default_shrinkage=0.0)
        result = f.forecast(ok_df)
        np.testing.assert_allclose(
            result["Forecasted_FTE_Adjusted"].values,
            result["Forecasted_FTE"].values,
            rtol=1e-6,
        )

    def test_fte_derived_from_hours(
        self, forecaster: CapacityForecaster, ok_df: pd.DataFrame
    ) -> None:
        result = forecaster.forecast(ok_df)
        expected_fte = result["Forecasted_Hours"] / forecaster._hours_per_fte_per_month
        np.testing.assert_allclose(
            result["Forecasted_FTE"].values,
            expected_fte.values,
            rtol=1e-2,  # rounded to 2dp in output
        )

    def test_shrinkage_values_within_bounds(
        self, forecaster: CapacityForecaster, ok_df_with_shrinkage: pd.DataFrame
    ) -> None:
        result = forecaster.forecast(ok_df_with_shrinkage)
        assert (result["Forecasted_Shrinkage"] >= 0.05).all()
        assert (result["Forecasted_Shrinkage"] <= 0.95).all()

    def test_shrinkage_varies_across_months_when_historical_data_present(
        self, forecaster: CapacityForecaster, ok_df_with_shrinkage: pd.DataFrame
    ) -> None:
        result = forecaster.forecast(ok_df_with_shrinkage)
        # With SARIMAX forecasting, shrinkage should not be identical every month
        assert result["Forecasted_Shrinkage"].nunique() > 1

    def test_flat_shrinkage_when_no_shrinkage_column(self, ok_df: pd.DataFrame) -> None:
        f = CapacityForecaster(default_shrinkage=0.25)
        result = f.forecast(ok_df)
        # No Shrinkage column → flat default applied → all values identical
        assert result["Forecasted_Shrinkage"].nunique() == 1
        assert result["Forecasted_Shrinkage"].iloc[0] == pytest.approx(0.25)

    def test_forecast_horizon_respected(self, ok_df: pd.DataFrame) -> None:
        f = CapacityForecaster(forecast_horizon=6)
        result = f.forecast(ok_df)
        assert len(result) == 6


# ---------------------------------------------------------------------------
# 6. Shrinkage column handling
# ---------------------------------------------------------------------------


class TestShrinkageColumn:
    def test_missing_shrinkage_falls_back_to_default(self, ok_df: pd.DataFrame) -> None:
        f = CapacityForecaster(default_shrinkage=0.20)
        result = f.forecast(ok_df)
        np.testing.assert_allclose(
            result["Forecasted_Shrinkage"].values,
            0.20,
            rtol=1e-6,
        )

    def test_all_nan_shrinkage_falls_back_to_default(self, ok_df: pd.DataFrame) -> None:
        ok_df["Shrinkage"] = np.nan
        f = CapacityForecaster(default_shrinkage=0.20)
        result = f.forecast(ok_df)
        np.testing.assert_allclose(
            result["Forecasted_Shrinkage"].values,
            0.20,
            rtol=1e-6,
        )

    def test_partial_nan_shrinkage_uses_non_nan_rows(
        self, ok_df_with_shrinkage: pd.DataFrame
    ) -> None:
        # Set half the rows to NaN — should still forecast from remaining data
        ok_df_with_shrinkage.loc[:17, "Shrinkage"] = np.nan
        f = CapacityForecaster(default_shrinkage=0.30)
        result = f.forecast(ok_df_with_shrinkage)
        assert len(result) == 12


# ---------------------------------------------------------------------------
# 7. diagnostic_summary
# ---------------------------------------------------------------------------


class TestDiagnosticSummary:
    def test_returns_dict_keyed_by_group(
        self, forecaster: CapacityForecaster, ok_df: pd.DataFrame
    ) -> None:
        summary = forecaster.diagnostic_summary(ok_df)
        assert isinstance(summary, dict)
        assert "Alpha" in summary

    def test_ok_group_has_expected_keys(
        self, forecaster: CapacityForecaster, ok_df: pd.DataFrame
    ) -> None:
        summary = forecaster.diagnostic_summary(ok_df)
        info = summary["Alpha"]
        assert "n_obs" in info
        assert "data_quality" in info
        assert "hours_model" in info
        assert "volume_model" in info

    def test_shrinkage_model_present_when_column_supplied(
        self, forecaster: CapacityForecaster, ok_df_with_shrinkage: pd.DataFrame
    ) -> None:
        summary = forecaster.diagnostic_summary(ok_df_with_shrinkage)
        assert "shrinkage_model" in summary["Alpha"]

    def test_shrinkage_model_absent_without_column(
        self, forecaster: CapacityForecaster, ok_df: pd.DataFrame
    ) -> None:
        summary = forecaster.diagnostic_summary(ok_df)
        assert "shrinkage_model" not in summary["Alpha"]

    def test_skipped_group_has_status_key(self, forecaster: CapacityForecaster) -> None:
        tiny = _make_df(ABSOLUTE_MIN_DATA_POINTS - 1, groups=["Tiny"])
        ok = _make_df(MIN_DATA_POINTS, groups=["OK"])
        df = pd.concat([tiny, ok], ignore_index=True)
        summary = forecaster.diagnostic_summary(df)
        assert "status" in summary["Tiny"]
        assert "skipped" in summary["Tiny"]["status"]

    def test_model_metadata_contains_order_and_aicc(
        self, forecaster: CapacityForecaster, ok_df: pd.DataFrame
    ) -> None:
        summary = forecaster.diagnostic_summary(ok_df)
        model_info = summary["Alpha"]["hours_model"]
        assert "order" in model_info
        assert "aicc" in model_info
        assert "aic" in model_info
        assert "stationary" in model_info

    def test_data_quality_correct_in_summary(
        self, forecaster: CapacityForecaster
    ) -> None:
        ok = _make_df(MIN_DATA_POINTS, groups=["OK Group"])
        low = _make_df(12, groups=["Low Group"])
        df = pd.concat([ok, low], ignore_index=True)
        summary = forecaster.diagnostic_summary(df)
        assert summary["OK Group"]["data_quality"] == DataQuality.OK
        assert summary["Low Group"]["data_quality"] == DataQuality.LOW_HISTORY


# ---------------------------------------------------------------------------
# 8. Public constants
# ---------------------------------------------------------------------------


class TestPublicConstants:
    def test_min_data_points_value(self) -> None:
        assert MIN_DATA_POINTS == 36

    def test_absolute_min_data_points_value(self) -> None:
        assert ABSOLUTE_MIN_DATA_POINTS == 6

    def test_absolute_min_less_than_min(self) -> None:
        assert ABSOLUTE_MIN_DATA_POINTS < MIN_DATA_POINTS

    def test_confidence_level_value(self) -> None:
        assert CONFIDENCE_LEVEL == 0.95

    def test_all_importable(self) -> None:
        from capacity_forecaster import (  # noqa: F401
            ABSOLUTE_MIN_DATA_POINTS,
            CONFIDENCE_LEVEL,
            MIN_DATA_POINTS,
            CapacityForecaster,
            DataQuality,
        )
