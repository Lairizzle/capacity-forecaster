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
# Fixtures & helpers
# ---------------------------------------------------------------------------

# Exact output columns produced by forecast() regardless of shrinkage.
# Note: the group column name mirrors the caller's input name, so it is
# tested separately rather than hardcoded here.
EXPECTED_OUTPUT_COLUMNS = [
    "Date",
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
    "Shrinkage_Used",
    "Imputed_Months",
    "Data_Quality",
]


def _make_df(
    n_months: int,
    groups: list[str] | None = None,
    include_shrinkage: bool = False,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Build a minimal but realistic synthetic DataFrame for testing.

    Columns are in the positional order the API expects:
        0: date  |  1: group  |  2: volume  |  3: hours  |  4: shrinkage (opt)

    Column names use realistic-but-arbitrary strings so tests confirm the
    positional resolution works regardless of naming.
    """
    rng = np.random.default_rng(seed)
    groups = groups or ["Group A"]
    dates = pd.date_range("2020-01-01", periods=n_months, freq="MS")
    rows = []
    for group in groups:
        base_volume = 10_000 + rng.integers(0, 2_000)
        base_hours = 3_000 + rng.integers(0, 500)
        for i, date in enumerate(dates):
            trend = 1.0 + (i / max(n_months, 1)) * 0.1
            row: dict = {
                "period": date,
                "team": group,
                "contacts": int(base_volume * trend * rng.normal(1.0, 0.03)),
                "handle_hours": int(base_hours * trend * rng.normal(1.0, 0.03)),
            }
            if include_shrinkage:
                row["shrinkage_rate"] = round(
                    float(np.clip(rng.normal(0.25, 0.02), 0.05, 0.60)), 4
                )
            rows.append(row)
    return pd.DataFrame(rows)


@pytest.fixture
def ok_df() -> pd.DataFrame:
    """Single group with MIN_DATA_POINTS months → Data_Quality = OK."""
    return _make_df(MIN_DATA_POINTS, groups=["Alpha"])


@pytest.fixture
def ok_df_with_shrinkage() -> pd.DataFrame:
    """Single group with MIN_DATA_POINTS months and a shrinkage column."""
    return _make_df(MIN_DATA_POINTS, groups=["Alpha"], include_shrinkage=True)


@pytest.fixture
def low_history_df() -> pd.DataFrame:
    """Single group with 12 months → Data_Quality = LOW_HISTORY."""
    return _make_df(12, groups=["Beta"])


@pytest.fixture
def multi_group_df() -> pd.DataFrame:
    """Two groups: one OK-quality, one LOW_HISTORY."""
    ok = _make_df(MIN_DATA_POINTS, groups=["OK Group"], seed=1)
    low = _make_df(12, groups=["Low Group"], seed=2)
    return pd.concat([ok, low], ignore_index=True)


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
        assert abs(f._hours_per_fte_per_month - 37.5 * (52.0 / 12.0)) < 1e-9

    def test_invalid_weekly_hours_zero(self) -> None:
        with pytest.raises(ValueError, match="weekly_hours"):
            CapacityForecaster(weekly_hours=0)

    def test_invalid_weekly_hours_negative(self) -> None:
        with pytest.raises(ValueError, match="weekly_hours"):
            CapacityForecaster(weekly_hours=-1)

    def test_invalid_forecast_horizon(self) -> None:
        with pytest.raises(ValueError, match="forecast_horizon"):
            CapacityForecaster(forecast_horizon=0)

    def test_invalid_shrinkage_at_one(self) -> None:
        with pytest.raises(ValueError, match="default_shrinkage"):
            CapacityForecaster(default_shrinkage=1.0)

    def test_invalid_shrinkage_negative(self) -> None:
        with pytest.raises(ValueError, match="default_shrinkage"):
            CapacityForecaster(default_shrinkage=-0.1)

    def test_zero_shrinkage_is_valid(self) -> None:
        f = CapacityForecaster(default_shrinkage=0.0)
        assert f.default_shrinkage == 0.0


# ---------------------------------------------------------------------------
# 2. Input validation (_resolve_columns)
# ---------------------------------------------------------------------------


class TestValidation:
    def test_too_few_columns_raises(self, forecaster: CapacityForecaster) -> None:
        df = _make_df(MIN_DATA_POINTS)[["period", "team", "contacts"]]  # only 3 cols
        with pytest.raises(ValueError, match="at least 4 columns"):
            forecaster.forecast(df)

    def test_non_datetime_col0_raises(
        self, forecaster: CapacityForecaster, ok_df: pd.DataFrame
    ) -> None:
        # Replace the date column with strings that cannot be parsed as dates
        bad = ok_df.copy()
        bad["period"] = "not-a-date"
        with pytest.raises(ValueError):
            forecaster.forecast(bad)

    def test_shrinkage_above_one_raises(
        self, forecaster: CapacityForecaster, ok_df: pd.DataFrame
    ) -> None:
        ok_df["shrinkage_rate"] = 0.25
        ok_df.loc[0, "shrinkage_rate"] = 1.5  # invalid
        with pytest.raises(ValueError, match="Shrinkage column"):
            forecaster.forecast(ok_df)

    def test_shrinkage_negative_raises(
        self, forecaster: CapacityForecaster, ok_df: pd.DataFrame
    ) -> None:
        ok_df["shrinkage_rate"] = 0.25
        ok_df.loc[0, "shrinkage_rate"] = -0.1  # invalid
        with pytest.raises(ValueError, match="Shrinkage column"):
            forecaster.forecast(ok_df)

    def test_shrinkage_nan_is_allowed_by_validation(
        self, forecaster: CapacityForecaster, ok_df: pd.DataFrame
    ) -> None:
        # NaN shrinkage rows are valid input; they get interpolated / fallen back
        ok_df["shrinkage_rate"] = 0.25
        ok_df.loc[0, "shrinkage_rate"] = np.nan
        # Should not raise
        result = forecaster.forecast(ok_df)
        assert len(result) == forecaster.forecast_horizon

    def test_month_end_dates_accepted(self, forecaster: CapacityForecaster) -> None:
        df = _make_df(MIN_DATA_POINTS)
        # Convert to month-end
        df["period"] = df["period"] + pd.offsets.MonthEnd(0)
        result = forecaster.forecast(df)
        assert len(result) == forecaster.forecast_horizon

    def test_mixed_month_start_and_end_dates(
        self, forecaster: CapacityForecaster
    ) -> None:
        df = _make_df(MIN_DATA_POINTS)
        # Alternate between month-start and month-end within the same group
        df.loc[df.index % 2 == 0, "period"] += pd.offsets.MonthEnd(0)
        result = forecaster.forecast(df)
        assert len(result) == forecaster.forecast_horizon


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

    def test_group_column_preserves_caller_name(
        self, forecaster: CapacityForecaster, ok_df: pd.DataFrame
    ) -> None:
        # The caller's column name ("team") should appear in the output, not
        # a hardcoded string like "Capacity Planning Group".
        result = forecaster.forecast(ok_df)
        assert "team" in result.columns

    def test_row_count_single_group(
        self, forecaster: CapacityForecaster, ok_df: pd.DataFrame
    ) -> None:
        result = forecaster.forecast(ok_df)
        assert len(result) == forecaster.forecast_horizon

    def test_row_count_multi_group(
        self, forecaster: CapacityForecaster, multi_group_df: pd.DataFrame
    ) -> None:
        result = forecaster.forecast(multi_group_df)
        n_groups = multi_group_df["team"].nunique()
        assert len(result) == forecaster.forecast_horizon * n_groups

    def test_forecast_dates_are_month_start(
        self, forecaster: CapacityForecaster, ok_df: pd.DataFrame
    ) -> None:
        result = forecaster.forecast(ok_df)
        assert (result["Date"].dt.day == 1).all()

    def test_forecast_dates_are_sequential(
        self, forecaster: CapacityForecaster, ok_df: pd.DataFrame
    ) -> None:
        result = forecaster.forecast(ok_df)
        diffs = result["Date"].sort_values().diff().dropna().dt.days
        assert (diffs >= 28).all()
        assert (diffs <= 31).all()

    def test_forecast_starts_month_after_last_historical(
        self, forecaster: CapacityForecaster, ok_df: pd.DataFrame
    ) -> None:
        # Dates are normalised to month-start internally, so compare against that.
        last_historical = ok_df["period"].dt.to_period("M").dt.to_timestamp().max()
        result = forecaster.forecast(ok_df)
        first_forecast = result["Date"].min()
        assert first_forecast == last_historical + pd.offsets.MonthBegin(1)

    def test_forecast_horizon_respected(self, ok_df: pd.DataFrame) -> None:
        f = CapacityForecaster(forecast_horizon=6)
        result = f.forecast(ok_df)
        assert len(result) == 6

    def test_imputed_months_column_present_and_non_negative(
        self, forecaster: CapacityForecaster, ok_df: pd.DataFrame
    ) -> None:
        result = forecaster.forecast(ok_df)
        assert "Imputed_Months" in result.columns
        assert (result["Imputed_Months"] >= 0).all()


# ---------------------------------------------------------------------------
# 4. Gap filling
# ---------------------------------------------------------------------------


class TestGapFilling:
    def test_gap_in_history_is_filled(self, forecaster: CapacityForecaster) -> None:
        df = _make_df(MIN_DATA_POINTS)
        # Drop two months from the middle to create a gap
        df = df.drop(index=[10, 11]).reset_index(drop=True)
        result = forecaster.forecast(df)
        # Should still produce a full forecast; imputed_months reflects the gap
        assert len(result) == forecaster.forecast_horizon
        assert result["Imputed_Months"].iloc[0] == 2

    def test_no_gap_yields_zero_imputed_months(
        self, forecaster: CapacityForecaster, ok_df: pd.DataFrame
    ) -> None:
        result = forecaster.forecast(ok_df)
        assert result["Imputed_Months"].iloc[0] == 0

    def test_month_end_input_deduplicates_correctly(
        self, forecaster: CapacityForecaster
    ) -> None:
        # Month-end and month-start for the same month should collapse to one row
        df = _make_df(MIN_DATA_POINTS)
        extra = df.iloc[:1].copy()
        extra["period"] = extra["period"] + pd.offsets.MonthEnd(0)
        df = pd.concat([df, extra], ignore_index=True)
        result = forecaster.forecast(df)
        assert len(result) == forecaster.forecast_horizon


# ---------------------------------------------------------------------------
# 5. Data quality flags
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
        result = forecaster.forecast(low_history_df)
        assert (result["Data_Quality"] == DataQuality.LOW_HISTORY).all()

    def test_mixed_quality_across_groups(
        self, forecaster: CapacityForecaster, multi_group_df: pd.DataFrame
    ) -> None:
        result = forecaster.forecast(multi_group_df)
        ok_rows = result[result["team"] == "OK Group"]["Data_Quality"]
        low_rows = result[result["team"] == "Low Group"]["Data_Quality"]
        assert (ok_rows == DataQuality.OK).all()
        assert (low_rows == DataQuality.LOW_HISTORY).all()

    def test_data_quality_constants(self) -> None:
        assert DataQuality.OK == "OK"
        assert DataQuality.LOW_HISTORY == "LOW_HISTORY"
        assert DataQuality.FAILED == "FAILED"


# ---------------------------------------------------------------------------
# 6. Numeric output sanity
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

    def test_adjusted_fte_greater_than_raw_when_shrinkage_nonzero(
        self, forecaster: CapacityForecaster, ok_df: pd.DataFrame
    ) -> None:
        # default_shrinkage=0.30, no shrinkage column → flat default applied
        result = forecaster.forecast(ok_df)
        assert (result["Forecasted_FTE_Adjusted"] > result["Forecasted_FTE"]).all()

    def test_adjusted_fte_equals_raw_when_shrinkage_zero(
        self, ok_df: pd.DataFrame
    ) -> None:
        f = CapacityForecaster(default_shrinkage=0.0)
        result = f.forecast(ok_df)
        np.testing.assert_allclose(
            result["Forecasted_FTE_Adjusted"].to_numpy(),
            result["Forecasted_FTE"].to_numpy(),
            rtol=1e-6,
        )

    def test_fte_derived_from_hours(
        self, forecaster: CapacityForecaster, ok_df: pd.DataFrame
    ) -> None:
        result = forecaster.forecast(ok_df)
        expected = result["Forecasted_Hours"] / forecaster._hours_per_fte_per_month
        np.testing.assert_allclose(
            result["Forecasted_FTE"].to_numpy(),
            expected.to_numpy(),
            rtol=1e-9,
        )

    def test_adjusted_fte_derived_from_raw_and_shrinkage(
        self, forecaster: CapacityForecaster, ok_df: pd.DataFrame
    ) -> None:
        result = forecaster.forecast(ok_df)
        expected = result["Forecasted_FTE"] / (1.0 - result["Shrinkage_Used"])
        np.testing.assert_allclose(
            result["Forecasted_FTE_Adjusted"].to_numpy(),
            expected.to_numpy(),
            rtol=1e-9,
        )


# ---------------------------------------------------------------------------
# 7. Shrinkage column handling
# ---------------------------------------------------------------------------


class TestShrinkageColumn:
    def test_no_shrinkage_column_uses_default_flat(self, ok_df: pd.DataFrame) -> None:
        f = CapacityForecaster(default_shrinkage=0.20)
        result = f.forecast(ok_df)
        np.testing.assert_allclose(result["Shrinkage_Used"].to_numpy(), 0.20, rtol=1e-9)

    def test_shrinkage_used_column_always_present(
        self, forecaster: CapacityForecaster, ok_df: pd.DataFrame
    ) -> None:
        result = forecaster.forecast(ok_df)
        assert "Shrinkage_Used" in result.columns

    def test_shrinkage_used_within_valid_range(
        self, forecaster: CapacityForecaster, ok_df_with_shrinkage: pd.DataFrame
    ) -> None:
        result = forecaster.forecast(ok_df_with_shrinkage)
        assert (result["Shrinkage_Used"] >= 0.0).all()
        assert (result["Shrinkage_Used"] < 1.0).all()

    def test_shrinkage_forecast_varies_across_months(
        self, forecaster: CapacityForecaster, ok_df_with_shrinkage: pd.DataFrame
    ) -> None:
        # With SARIMAX forecasting, shrinkage should not be identical every month
        result = forecaster.forecast(ok_df_with_shrinkage)
        assert result["Shrinkage_Used"].nunique() > 1

    def test_all_nan_shrinkage_falls_back_to_default(self, ok_df: pd.DataFrame) -> None:
        ok_df["shrinkage_rate"] = np.nan
        f = CapacityForecaster(default_shrinkage=0.20)
        result = f.forecast(ok_df)
        np.testing.assert_allclose(result["Shrinkage_Used"].to_numpy(), 0.20, rtol=1e-9)

    def test_partial_nan_shrinkage_still_produces_forecast(
        self, ok_df_with_shrinkage: pd.DataFrame
    ) -> None:
        # Half the rows have NaN shrinkage; remaining rows should be enough to fit
        ok_df_with_shrinkage.loc[:17, "shrinkage_rate"] = np.nan
        f = CapacityForecaster(default_shrinkage=0.30)
        result = f.forecast(ok_df_with_shrinkage)
        assert len(result) == f.forecast_horizon

    def test_below_min_shrinkage_observations_falls_back_to_mean(
        self, ok_df: pd.DataFrame
    ) -> None:
        # Only 3 non-NaN shrinkage values — below ABSOLUTE_MIN_DATA_POINTS,
        # so the code should fall back to the historical mean rather than SARIMAX.
        ok_df["shrinkage_rate"] = np.nan
        ok_df.loc[:2, "shrinkage_rate"] = 0.22  # exactly 3 valid rows
        f = CapacityForecaster(default_shrinkage=0.30)
        result = f.forecast(ok_df)
        # All forecast months should use the mean of the 3 observed values (0.22),
        # not the default_shrinkage (0.30).
        np.testing.assert_allclose(result["Shrinkage_Used"].to_numpy(), 0.22, rtol=1e-6)


# ---------------------------------------------------------------------------
# 8. Positional column resolution
# ---------------------------------------------------------------------------


class TestPositionalColumns:
    def test_arbitrary_column_names_accepted(
        self, forecaster: CapacityForecaster
    ) -> None:
        df = _make_df(MIN_DATA_POINTS)
        # Rename to completely arbitrary names — should work fine
        df.columns = ["col_a", "col_b", "col_c", "col_d"]
        result = forecaster.forecast(df)
        assert len(result) == forecaster.forecast_horizon
        assert "col_b" in result.columns  # group column name preserved

    def test_group_column_name_preserved_in_output(
        self, forecaster: CapacityForecaster
    ) -> None:
        df = _make_df(MIN_DATA_POINTS)
        # "team" is the group column name set by _make_df
        result = forecaster.forecast(df)
        assert "team" in result.columns
        assert "Capacity Planning Group" not in result.columns

    def test_volume_and_hours_order(self, forecaster: CapacityForecaster) -> None:
        # Confirm col 2 = volume, col 3 = hours by checking FTE maths.
        # FTE = hours / hours_per_fte_per_month, so if the columns were swapped
        # the FTE values would be implausibly large (volume >> hours typically).
        df = _make_df(MIN_DATA_POINTS)
        result = forecaster.forecast(df)
        # Raw FTE should be a plausible headcount (roughly 1–50 for typical inputs)
        assert result["Forecasted_FTE"].max() < 500


# ---------------------------------------------------------------------------
# 7b. Group failure handling
# ---------------------------------------------------------------------------


class TestGroupFailureHandling:
    def test_failed_group_appears_in_results_with_failed_quality(
        self, forecaster: CapacityForecaster
    ) -> None:
        # Group below ABSOLUTE_MIN_DATA_POINTS threshold — explicitly rejected before fitting.
        tiny = _make_df(ABSOLUTE_MIN_DATA_POINTS - 1, groups=["Tiny"], seed=99)
        ok = _make_df(MIN_DATA_POINTS, groups=["OK"], seed=1)
        df = pd.concat([tiny, ok], ignore_index=True)
        with pytest.warns(UserWarning):
            result = forecaster.forecast(df)
        assert "Tiny" in result["team"].values
        tiny_rows = result[result["team"] == "Tiny"]
        assert (tiny_rows["Data_Quality"] == DataQuality.FAILED).all()

    def test_failed_group_has_nan_forecast_values(
        self, forecaster: CapacityForecaster
    ) -> None:
        tiny = _make_df(ABSOLUTE_MIN_DATA_POINTS - 1, groups=["Tiny"], seed=99)
        ok = _make_df(MIN_DATA_POINTS, groups=["OK"], seed=1)
        df = pd.concat([tiny, ok], ignore_index=True)
        with pytest.warns(UserWarning):
            result = forecaster.forecast(df)
        tiny_rows = result[result["team"] == "Tiny"]
        forecast_cols = [c for c in tiny_rows.columns if c.startswith("Forecasted_")]
        for col in forecast_cols:
            assert tiny_rows[col].isna().all(), f"{col} should be NaN for failed group"

    def test_failed_group_correct_row_count(
        self, forecaster: CapacityForecaster
    ) -> None:
        tiny = _make_df(ABSOLUTE_MIN_DATA_POINTS - 1, groups=["Tiny"], seed=99)
        ok = _make_df(MIN_DATA_POINTS, groups=["OK"], seed=1)
        df = pd.concat([tiny, ok], ignore_index=True)
        with pytest.warns(UserWarning):
            result = forecaster.forecast(df)
        tiny_rows = result[result["team"] == "Tiny"]
        assert len(tiny_rows) == forecaster.forecast_horizon

    def test_failed_group_emits_warning_with_group_name(
        self, forecaster: CapacityForecaster
    ) -> None:
        tiny = _make_df(ABSOLUTE_MIN_DATA_POINTS - 1, groups=["Tiny"], seed=99)
        ok = _make_df(MIN_DATA_POINTS, groups=["OK"], seed=1)
        df = pd.concat([tiny, ok], ignore_index=True)
        with pytest.warns(UserWarning) as record:
            forecaster.forecast(df)
        messages = " ".join(str(w.message) for w in record)
        assert "Tiny" in messages

    def test_surviving_groups_unaffected_by_peer_failure(
        self, forecaster: CapacityForecaster
    ) -> None:
        tiny = _make_df(ABSOLUTE_MIN_DATA_POINTS - 1, groups=["Tiny"], seed=99)
        ok = _make_df(MIN_DATA_POINTS, groups=["OK"], seed=1)
        df = pd.concat([tiny, ok], ignore_index=True)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            result = forecaster.forecast(df)
        ok_rows = result[result["team"] == "OK"]
        assert len(ok_rows) == forecaster.forecast_horizon
        assert (ok_rows["Data_Quality"] == DataQuality.OK).all()

    def test_all_groups_failing_still_returns_dataframe(
        self, forecaster: CapacityForecaster
    ) -> None:
        # Even if every group fails, we get a DataFrame back with FAILED rows.
        tiny = _make_df(ABSOLUTE_MIN_DATA_POINTS - 1, groups=["Tiny"], seed=99)
        with pytest.warns(UserWarning):
            result = forecaster.forecast(tiny)
        assert isinstance(result, pd.DataFrame)
        assert len(result) == forecaster.forecast_horizon
        assert (result["Data_Quality"] == DataQuality.FAILED).all()


# ---------------------------------------------------------------------------
# 9. Public constants
# ---------------------------------------------------------------------------


class TestPublicConstants:
    def test_min_data_points(self) -> None:
        assert MIN_DATA_POINTS == 36

    def test_absolute_min_data_points(self) -> None:
        assert ABSOLUTE_MIN_DATA_POINTS == 6

    def test_absolute_min_less_than_min(self) -> None:
        assert ABSOLUTE_MIN_DATA_POINTS < MIN_DATA_POINTS

    def test_confidence_level(self) -> None:
        assert CONFIDENCE_LEVEL == 0.95

    def test_all_public_names_importable(self) -> None:
        from capacity_forecaster import (  # noqa: F401
            ABSOLUTE_MIN_DATA_POINTS,
            CONFIDENCE_LEVEL,
            MIN_DATA_POINTS,
            CapacityForecaster,
            DataQuality,
        )
