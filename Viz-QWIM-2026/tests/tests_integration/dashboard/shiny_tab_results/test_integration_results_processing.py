"""Integration tests for the results tab data processing pipeline.

Verifies that process_data_for_plot_tab_results, normalize_data_tab_results,
and transform_data_tab_results work correctly in isolation and when chained
together in processing sequences.

Test Categories
---------------
- process_data_for_plot_tab_results: passthrough and downsampling behaviour
- normalize_data_tab_results: min-max and z-score normalisation correctness
- transform_data_tab_results: percent_change and cumulative transforms
- Chained pipelines: process → normalize → transform end-to-end

Author:
    QWIM Development Team

Version:
    0.1.0

Last Modified:
    2026-02-28
"""

from __future__ import annotations

from typing import Any

import polars as pl
import pytest

from src.utils.custom_exceptions_errors_loggers.logger_custom import get_logger

_logger = get_logger(__name__)

MODULE_IMPORT_AVAILABLE: bool = True

try:
    from src.dashboard.shiny_utils.utils_tab_results import (
        normalize_data_tab_results,
        process_data_for_plot_tab_results,
        transform_data_tab_results,
    )
except ImportError as exc:
    MODULE_IMPORT_AVAILABLE = False
    _logger.warning("Import failed — tests will be skipped: %s", exc)

pytestmark = pytest.mark.skipif(
    not MODULE_IMPORT_AVAILABLE,
    reason="utils_tab_results not importable in this environment",
)


# ==============================================================================
# Fixtures
# ==============================================================================


@pytest.fixture()
def small_df() -> pl.DataFrame:
    """Provide a small DataFrame with lowercase 'date' column for processing tests.

    Returns
    -------
    pl.DataFrame
        DataFrame with date and two numeric value columns, 10 rows.
    """
    return pl.DataFrame(
        {
            "date": [
                "2023-01-01", "2023-02-01", "2023-03-01", "2023-04-01", "2023-05-01",
                "2023-06-01", "2023-07-01", "2023-08-01", "2023-09-01", "2023-10-01",
            ],
            "portfolio": [100.0, 102.0, 105.0, 103.0, 108.0, 106.0, 110.0, 112.0, 109.0, 115.0],
            "benchmark": [100.0, 101.0, 103.0, 102.5, 105.0, 104.0, 107.0, 109.0, 107.5, 111.0],
        }
    )


@pytest.fixture()
def large_df() -> pl.DataFrame:
    """Provide a larger DataFrame (200 rows) to trigger downsampling in process function.

    Returns
    -------
    pl.DataFrame
        DataFrame with 200 rows of synthetic portfolio and benchmark values.
    """
    import datetime

    base = datetime.date(2020, 1, 1)
    rows = 200
    dates = [(base + datetime.timedelta(days=i)).isoformat() for i in range(rows)]
    portfolio_vals = [100.0 + i * 0.5 for i in range(rows)]
    benchmark_vals = [100.0 + i * 0.4 for i in range(rows)]

    return pl.DataFrame(
        {
            "date": dates,
            "portfolio": portfolio_vals,
            "benchmark": benchmark_vals,
        }
    )


# ==============================================================================
# process_data_for_plot_tab_results integration
# ==============================================================================


@pytest.mark.integration()
class Test_Process_Data_For_Plot_Integration:
    """Integration tests for process_data_for_plot_tab_results."""

    def test_small_df_returned_unchanged(self, small_df: pl.DataFrame) -> None:
        """DataFrames within max_points limit are returned without modification."""
        _logger.debug("Testing process_data passthrough for small DataFrames")

        result = process_data_for_plot_tab_results(small_df, date_column="date", max_points=5000)

        assert result is not None, "Result must not be None"
        assert isinstance(result, pl.DataFrame), "Result must be Polars DataFrame"
        assert result.height == small_df.height, (
            f"Small DataFrame must be returned unchanged, expected {small_df.height} rows, got {result.height}"
        )

    def test_large_df_downsampled_below_max_points(self, large_df: pl.DataFrame) -> None:
        """DataFrames exceeding max_points are downsampled to at most max_points rows."""
        _logger.debug("Testing process_data downsamples large DataFrames")

        max_points = 50
        result = process_data_for_plot_tab_results(large_df, date_column="date", max_points=max_points)

        assert result is not None, "Result must not be None"
        assert result.height <= max_points, (
            f"Downsampled DataFrame must have at most {max_points} rows, got {result.height}"
        )

    def test_none_input_returns_none(self) -> None:
        """None input is returned as-is without error."""
        _logger.debug("Testing process_data handles None input")

        result = process_data_for_plot_tab_results(None, date_column="date")

        assert result is None, "None input must be returned as None"

    def test_output_preserves_all_columns(self, small_df: pl.DataFrame) -> None:
        """Output DataFrame preserves all columns from the input."""
        _logger.debug("Testing process_data preserves column structure")

        result = process_data_for_plot_tab_results(small_df, date_column="date")

        assert set(result.columns) == set(small_df.columns), (
            f"Output columns {result.columns} must match input columns {small_df.columns}"
        )


# ==============================================================================
# normalize_data_tab_results integration
# ==============================================================================


@pytest.mark.integration()
class Test_Normalize_Data_Tab_Results_Integration:
    """Integration tests for normalize_data_tab_results."""

    def test_min_max_normalisation_range_is_zero_to_one(
        self, small_df: pl.DataFrame
    ) -> None:
        """Min-max normalised numeric columns are bounded [0, 1]."""
        _logger.debug("Testing min_max normalisation produces values in [0, 1]")

        result = normalize_data_tab_results(small_df, method="min_max")

        numeric_cols = [c for c in result.columns if c != "date"]
        for col in numeric_cols:
            col_min = result[col].min()
            col_max = result[col].max()
            assert col_min >= 0.0 - 1e-10, (
                f"After min_max norm, '{col}' min must be >= 0, got {col_min}"
            )
            assert col_max <= 1.0 + 1e-10, (
                f"After min_max norm, '{col}' max must be <= 1, got {col_max}"
            )

    def test_min_max_normalisation_minimum_is_zero(
        self, small_df: pl.DataFrame
    ) -> None:
        """Min-max normalised numeric columns have minimum value of exactly 0.0."""
        _logger.debug("Testing min_max normalisation produces min of 0")

        result = normalize_data_tab_results(small_df, method="min_max")

        numeric_cols = [c for c in result.columns if c != "date"]
        for col in numeric_cols:
            assert abs(result[col].min()) < 1e-10, (
                f"After min_max norm, '{col}' minimum must be 0.0, got {result[col].min()}"
            )

    def test_min_max_normalisation_maximum_is_one(
        self, small_df: pl.DataFrame
    ) -> None:
        """Min-max normalised numeric columns have maximum value of exactly 1.0."""
        _logger.debug("Testing min_max normalisation produces max of 1")

        result = normalize_data_tab_results(small_df, method="min_max")

        numeric_cols = [c for c in result.columns if c != "date"]
        for col in numeric_cols:
            assert abs(result[col].max() - 1.0) < 1e-10, (
                f"After min_max norm, '{col}' maximum must be 1.0, got {result[col].max()}"
            )

    def test_z_score_normalisation_mean_is_near_zero(
        self, small_df: pl.DataFrame
    ) -> None:
        """Z-score normalised numeric columns have mean approximately 0."""
        _logger.debug("Testing z_score normalisation produces mean near 0")

        result = normalize_data_tab_results(small_df, method="z_score")

        numeric_cols = [c for c in result.columns if c != "date"]
        for col in numeric_cols:
            mean_val = result[col].mean()
            assert abs(mean_val) < 1e-6, (
                f"After z_score norm, '{col}' mean must be ~0, got {mean_val}"
            )

    def test_none_method_returns_dataframe_unchanged(
        self, small_df: pl.DataFrame
    ) -> None:
        """method='none' returns the DataFrame without any transformation."""
        _logger.debug("Testing normalize with method='none' returns unchanged DataFrame")

        result = normalize_data_tab_results(small_df, method="none")

        assert result is not None, "Result must not be None"
        assert result.height == small_df.height, "Row count must be unchanged"

        numeric_cols = [c for c in small_df.columns if c != "date"]
        for col in numeric_cols:
            original = small_df[col].to_list()
            returned = result[col].to_list()
            assert original == returned, f"Column '{col}' values must be unchanged for method='none'"

    def test_normalisation_preserves_date_column(
        self, small_df: pl.DataFrame
    ) -> None:
        """Normalisation leaves the date column values unchanged."""
        _logger.debug("Testing normalisation preserves date column")

        result = normalize_data_tab_results(small_df, method="min_max")

        assert result["date"].to_list() == small_df["date"].to_list(), (
            "Date column must be unchanged after normalisation"
        )


# ==============================================================================
# transform_data_tab_results integration
# ==============================================================================


@pytest.mark.integration()
class Test_Transform_Data_Tab_Results_Integration:
    """Integration tests for transform_data_tab_results."""

    def test_percent_change_first_value_is_zero(
        self, small_df: pl.DataFrame
    ) -> None:
        """Percent change transformation yields 0.0 for the first row of each series."""
        _logger.debug("Testing percent_change first value is 0.0")

        result = transform_data_tab_results(small_df, transformation="percent_change")

        numeric_cols = [c for c in small_df.columns if c != "date"]
        for col in numeric_cols:
            pct_col = f"{col}_pct"
            if pct_col in result.columns:
                first_pct = result[pct_col][0]
                assert abs(first_pct) < 1e-10, (
                    f"First percent change for '{col}' must be 0.0, got {first_pct}"
                )

    def test_percent_change_creates_new_columns(
        self, small_df: pl.DataFrame
    ) -> None:
        """Percent change transformation adds _pct suffixed columns for each series."""
        _logger.debug("Testing percent_change adds _pct columns")

        result = transform_data_tab_results(small_df, transformation="percent_change")

        numeric_cols = [c for c in small_df.columns if c != "date"]
        for col in numeric_cols:
            pct_col = f"{col}_pct"
            assert pct_col in result.columns, (
                f"Expected column '{pct_col}' not found after percent_change transform"
            )

    def test_cumulative_creates_new_columns(
        self, small_df: pl.DataFrame
    ) -> None:
        """Cumulative transformation adds _cum suffixed columns for each series."""
        _logger.debug("Testing cumulative transform adds _cum columns")

        result = transform_data_tab_results(small_df, transformation="cumulative")

        numeric_cols = [c for c in small_df.columns if c != "date"]
        for col in numeric_cols:
            cum_col = f"{col}_cum"
            assert cum_col in result.columns, (
                f"Expected column '{cum_col}' not found after cumulative transform"
            )

    def test_cumulative_is_monotonically_non_decreasing_for_positive_series(
        self, small_df: pl.DataFrame
    ) -> None:
        """Cumulative sum of a positive series is monotonically non-decreasing."""
        _logger.debug("Testing cumulative transform monotonicity for positive data")

        result = transform_data_tab_results(small_df, transformation="cumulative")

        cum_col = "portfolio_cum"
        if cum_col in result.columns:
            values = result[cum_col].to_list()
            for i in range(1, len(values)):
                assert values[i] >= values[i - 1], (
                    f"Cumulative portfolio value at index {i} must be >= value at {i - 1}: "
                    f"{values[i]} < {values[i - 1]}"
                )

    def test_none_transformation_returns_unchanged(
        self, small_df: pl.DataFrame
    ) -> None:
        """transformation='none' returns the DataFrame without transformation."""
        _logger.debug("Testing transform with transformation='none' is a passthrough")

        result = transform_data_tab_results(small_df, transformation="none")

        assert result is not None, "Result must not be None"
        assert result.height == small_df.height, "Row count must be unchanged"


# ==============================================================================
# Chained pipeline integration
# ==============================================================================


@pytest.mark.integration()
class Test_Chained_Results_Processing_Integration:
    """Integration tests for chained processing → normalise → transform pipelines."""

    def test_process_then_normalize_min_max(
        self, small_df: pl.DataFrame
    ) -> None:
        """process_data → normalize_min_max chain produces a valid min-max normalised result."""
        _logger.debug("Testing process → normalize_min_max chain")

        processed = process_data_for_plot_tab_results(small_df, date_column="date", max_points=5000)
        normalised = normalize_data_tab_results(processed, method="min_max")

        assert isinstance(normalised, pl.DataFrame), "Chained result must be Polars DataFrame"
        assert normalised.height == small_df.height, "Row count must be preserved through chain"

        numeric_cols = [c for c in normalised.columns if c != "date"]
        for col in numeric_cols:
            assert normalised[col].min() >= 0.0 - 1e-10, f"Min-max output for '{col}' must be >= 0"
            assert normalised[col].max() <= 1.0 + 1e-10, f"Min-max output for '{col}' must be <= 1"

    def test_normalize_min_max_then_percent_change(
        self, small_df: pl.DataFrame
    ) -> None:
        """normalize_min_max → percent_change chain produces a valid output DataFrame.

        After min_max normalisation the first row equals 0.0 when the series starts at
        its minimum (as with small_df), so transform_data_tab_results skips adding _pct
        columns to avoid division-by-zero.  Use data whose minimum is NOT the first
        observation so the first normalised value is non-zero and _pct columns appear.
        """
        _logger.debug("Testing normalize_min_max → percent_change chain")

        # Build data where the minimum occurs mid-series so the first normalised value > 0
        df_rising_from_dip = pl.DataFrame(
            {
                "date": [
                    "2023-01-01", "2023-02-01", "2023-03-01", "2023-04-01", "2023-05-01",
                    "2023-06-01", "2023-07-01", "2023-08-01", "2023-09-01", "2023-10-01",
                ],
                "portfolio":  [110.0, 100.0, 107.0, 105.0, 116.0, 112.0, 118.0, 122.0, 119.0, 125.0],
                "benchmark":  [105.0, 100.0, 103.0, 102.5, 110.0, 107.0, 113.0, 116.0, 114.0, 120.0],
            }
        )

        normalised = normalize_data_tab_results(df_rising_from_dip, method="min_max")
        transformed = transform_data_tab_results(normalised, transformation="percent_change")

        assert isinstance(transformed, pl.DataFrame), "Chained result must be Polars DataFrame"
        assert "date" in transformed.columns, "Date column must survive the chain"

        numeric_cols = [c for c in df_rising_from_dip.columns if c != "date"]
        for col in numeric_cols:
            pct_col = f"{col}_pct"
            assert pct_col in transformed.columns, (
                f"Expected '{pct_col}' column after normalize→percent_change chain"
            )

    def test_full_three_stage_chain(self, large_df: pl.DataFrame) -> None:
        """Full pipeline: process → normalize_z_score → cumulative produces valid output."""
        _logger.debug("Testing full 3-stage processing chain on large DataFrame")

        processed = process_data_for_plot_tab_results(large_df, date_column="date", max_points=100)
        normalised = normalize_data_tab_results(processed, method="z_score")
        transformed = transform_data_tab_results(normalised, transformation="cumulative")

        assert isinstance(transformed, pl.DataFrame), "Full chain result must be Polars DataFrame"
        assert transformed.height > 0, "Result must have rows after full chain"
        assert "date" in transformed.columns, "Date column must survive full chain"

        # z-score normalisied data has cumulative sums:
        numeric_cols = [c for c in processed.columns if c != "date"]
        for col in numeric_cols:
            cum_col = f"{col}_cum"
            assert cum_col in transformed.columns, (
                f"Cumulative column '{cum_col}' must appear after full chain"
            )

    def test_chain_with_none_passthrough_methods(
        self, small_df: pl.DataFrame
    ) -> None:
        """Chain with method='none' and transformation='none' returns original data unchanged."""
        _logger.debug("Testing chain with passthrough (none) methods preserves data")

        processed = process_data_for_plot_tab_results(small_df, date_column="date", max_points=5000)
        normalised = normalize_data_tab_results(processed, method="none")
        transformed = transform_data_tab_results(normalised, transformation="none")

        assert transformed.height == small_df.height, "Passthrough chain must not change row count"
        assert set(transformed.columns) == set(small_df.columns), (
            "Passthrough chain must not change column structure"
        )
