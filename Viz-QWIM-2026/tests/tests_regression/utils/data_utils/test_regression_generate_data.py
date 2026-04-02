"""Regression tests for generate_data module.

These tests lock in the exact output values produced by the data generation
functions using the current deterministic random number generator
(``np.random.default_rng(seed=42)``).

If the seed, generator type, or column definitions change, these tests
will fail and must be deliberately regenerated.

Baseline values computed from the current implementation on 2026-03-01.

Author: QWIM Development Team
Version: 0.1.0
"""

from __future__ import annotations

import pytest

import polars as pl


# ==============================================================================
# Regression: generate_monthly_timeseries – fixed 2020 window
# ==============================================================================


@pytest.fixture(scope="module")
def df_2020():
    """generate_monthly_timeseries for 2020-01-01 → 2020-12-31 (12 rows)."""
    from src.utils.data_utils.generate_data import generate_monthly_timeseries

    return generate_monthly_timeseries(start_date="2020-01-01", end_date="2020-12-31")


@pytest.fixture(scope="module")
def df_default():
    """generate_monthly_timeseries with default parameters (≥279 rows)."""
    from src.utils.data_utils.generate_data import generate_monthly_timeseries

    return generate_monthly_timeseries()


class Test_Regression_Generate_Monthly_Shape:
    """Regression tests for output shape and column structure."""

    @pytest.mark.regression()
    def test_2020_row_count(self, df_2020):
        """2020 window produces exactly 12 rows."""
        assert df_2020.shape[0] == 12

    @pytest.mark.regression()
    def test_2020_column_count(self, df_2020):
        """2020 window produces exactly 8 columns (date + 7 series)."""
        assert df_2020.shape[1] == 8

    @pytest.mark.regression()
    def test_2020_column_names_locked(self, df_2020):
        """Column names are exactly ['date', 'AA', 'BB', 'CC', 'DD', 'EE', 'FF', 'GG']."""
        assert list(df_2020.columns) == ["date", "AA", "BB", "CC", "DD", "EE", "FF", "GG"]

    @pytest.mark.regression()
    def test_default_row_count_at_least_279(self, df_default):
        """Default date range produces at least 279 monthly rows."""
        assert df_default.shape[0] >= 279

    @pytest.mark.regression()
    def test_default_column_names_unchanged(self, df_default):
        """Default output column names are identical to fixed-window names."""
        assert list(df_default.columns) == ["date", "AA", "BB", "CC", "DD", "EE", "FF", "GG"]


class Test_Regression_Generate_Monthly_Values:
    """Regression tests for exact first/last values of each series column.

    Baselines verified against implementation using np.random.default_rng(seed=42).
    """

    @pytest.mark.regression()
    def test_aa_first_value_locked(self, df_2020):
        """AA[0] = 101.52 (baseline from np.random.default_rng(42))."""
        assert df_2020["AA"][0] == pytest.approx(101.52, rel=1e-6)

    @pytest.mark.regression()
    def test_aa_last_value_locked(self, df_2020):
        """AA[-1] = 101.89."""
        assert df_2020["AA"][-1] == pytest.approx(101.89, rel=1e-6)

    @pytest.mark.regression()
    def test_bb_first_value_locked(self, df_2020):
        """BB[0] = 150.07."""
        assert df_2020["BB"][0] == pytest.approx(150.07, rel=1e-6)

    @pytest.mark.regression()
    def test_bb_last_value_locked(self, df_2020):
        """BB[-1] = 179.43."""
        assert df_2020["BB"][-1] == pytest.approx(179.43, rel=1e-6)

    @pytest.mark.regression()
    def test_cc_first_value_locked(self, df_2020):
        """CC[0] = 47.86."""
        assert df_2020["CC"][0] == pytest.approx(47.86, rel=1e-6)

    @pytest.mark.regression()
    def test_cc_last_value_locked(self, df_2020):
        """CC[-1] = 58.47."""
        assert df_2020["CC"][-1] == pytest.approx(58.47, rel=1e-6)

    @pytest.mark.regression()
    def test_dd_first_value_locked(self, df_2020):
        """DD[0] = 198.86."""
        assert df_2020["DD"][0] == pytest.approx(198.86, rel=1e-6)

    @pytest.mark.regression()
    def test_dd_last_value_locked(self, df_2020):
        """DD[-1] = 202.24."""
        assert df_2020["DD"][-1] == pytest.approx(202.24, rel=1e-6)

    @pytest.mark.regression()
    def test_ee_first_value_locked(self, df_2020):
        """EE[0] = 305.43."""
        assert df_2020["EE"][0] == pytest.approx(305.43, rel=1e-6)

    @pytest.mark.regression()
    def test_ee_last_value_locked(self, df_2020):
        """EE[-1] = 294.45."""
        assert df_2020["EE"][-1] == pytest.approx(294.45, rel=1e-6)

    @pytest.mark.regression()
    def test_ff_first_value_locked(self, df_2020):
        """FF[0] = 77.93."""
        assert df_2020["FF"][0] == pytest.approx(77.93, rel=1e-6)

    @pytest.mark.regression()
    def test_ff_last_value_locked(self, df_2020):
        """FF[-1] = 93.87."""
        assert df_2020["FF"][-1] == pytest.approx(93.87, rel=1e-6)

    @pytest.mark.regression()
    def test_gg_first_value_locked(self, df_2020):
        """GG[0] = 73.56."""
        assert df_2020["GG"][0] == pytest.approx(73.56, rel=1e-6)

    @pytest.mark.regression()
    def test_gg_last_value_locked(self, df_2020):
        """GG[-1] = 77.14."""
        assert df_2020["GG"][-1] == pytest.approx(77.14, rel=1e-6)


class Test_Regression_Generate_Monthly_Reproducibility:
    """Verify that the seeded generator produces identical outputs across calls."""

    @pytest.mark.regression()
    def test_full_reproducibility_same_params(self):
        """Two calls with identical parameters produce bit-for-bit identical DataFrames."""
        from src.utils.data_utils.generate_data import generate_monthly_timeseries

        df_a = generate_monthly_timeseries(start_date="2020-01-01", end_date="2020-12-31")
        df_b = generate_monthly_timeseries(start_date="2020-01-01", end_date="2020-12-31")
        for col in df_a.columns:
            assert (df_a[col] == df_b[col]).all(), (
                f"Column {col} differs between two identical calls"
            )

    @pytest.mark.regression()
    def test_default_params_fully_reproducible(self):
        """Two default calls produce identical DataFrames."""
        from src.utils.data_utils.generate_data import generate_monthly_timeseries

        df_a = generate_monthly_timeseries()
        df_b = generate_monthly_timeseries()
        for col in df_a.columns:
            assert (df_a[col] == df_b[col]).all(), (
                f"Default column {col} differs between two calls"
            )

    @pytest.mark.regression()
    def test_default_aa_first_three_values(self):
        """Default run's first 3 AA values are [101.52, 102.80, 117.74]."""
        from src.utils.data_utils.generate_data import generate_monthly_timeseries

        df = generate_monthly_timeseries()
        aa_first_3 = df["AA"].to_list()[:3]
        expected = [101.52, 102.80, 117.74]
        for actual, exp in zip(aa_first_3, expected):
            assert actual == pytest.approx(exp, rel=1e-5)


class Test_Regression_Generate_Monthly_Statistics:
    """Regression tests for statistical properties of the 2020 window."""

    @pytest.mark.regression()
    def test_aa_mean_locked(self, df_2020):
        """AA mean ≈ 102.0425 for the 2020 window."""
        assert df_2020["AA"].mean() == pytest.approx(102.0425, rel=1e-4)

    @pytest.mark.regression()
    def test_bb_mean_locked(self, df_2020):
        """BB mean ≈ 171.2558 for the 2020 window."""
        assert df_2020["BB"].mean() == pytest.approx(171.2558, rel=1e-4)

    @pytest.mark.regression()
    def test_all_series_have_positive_values(self, df_2020):
        """All generated series contain only positive values (price-like data)."""
        series_cols = [c for c in df_2020.columns if c != "date"]
        for col in series_cols:
            assert (df_2020[col] > 0).all(), f"Column {col} has non-positive values"

    @pytest.mark.regression()
    def test_no_null_values(self, df_2020):
        """No null values in any column of the 2020 window."""
        for col in df_2020.columns:
            assert df_2020[col].null_count() == 0, (
                f"Column {col} has null values"
            )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
