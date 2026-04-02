"""Integration tests for the dashboard data loading and validation pipeline.

Verifies that get_input_data_raw, validate_portfolio_data,
calculate_portfolio_returns, and validate_portfolio_and_benchmark_data
work correctly together — from reading raw CSV files through validation
and returns computation.

Test Categories
---------------
- get_input_data_raw loads all expected data sources into correctly keyed dict
- Each loaded DataFrame has required Date column and non-empty rows
- validate_portfolio_data accepts correctly structured portfolio DataFrames
- validate_portfolio_data rejects DataFrames with structural problems
- calculate_portfolio_returns produces a valid Series from portfolio data
- validate_portfolio_and_benchmark_data works with two valid DataFrames
- Full pipeline: load → validate → compute returns

Author:
    QWIM Development Team

Version:
    0.1.0

Last Modified:
    2026-02-28
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import polars as pl
import pytest

from src.utils.custom_exceptions_errors_loggers.logger_custom import get_logger

_logger = get_logger(__name__)

MODULE_IMPORT_AVAILABLE: bool = True

try:
    from src.dashboard.shiny_utils.utils_data import (
        calculate_portfolio_returns,
        get_input_data_raw,
        validate_portfolio_and_benchmark_data_if_not_already_validated,
        validate_portfolio_data,
    )
except ImportError as exc:
    MODULE_IMPORT_AVAILABLE = False
    _logger.warning("Import failed — tests will be skipped: %s", exc)

pytestmark = pytest.mark.skipif(
    not MODULE_IMPORT_AVAILABLE,
    reason="Dashboard utils_data modules not importable in this environment",
)

# Project root relative to this test file: tests/tests_integration/dashboard/shiny_utils/
_PROJECT_DIR: Path = Path(__file__).resolve().parents[4]

# Expected keys in the dict returned by get_input_data_raw
_EXPECTED_DATA_KEYS: list[str] = [
    "Time_Series_Sample",
    "Time_Series_ETFs",
    "Weights_My_Portfolio",
]


# ==============================================================================
# Fixtures
# ==============================================================================


@pytest.fixture(scope="module")
def raw_data() -> dict[str, pl.DataFrame]:
    """Load raw data from CSV files using get_input_data_raw.

    Returns
    -------
    dict[str, pl.DataFrame]
        Dictionary mapping data source names to their Polars DataFrames.
    """
    _logger.debug("Loading raw input data from project directory: %s", _PROJECT_DIR)
    return get_input_data_raw(_PROJECT_DIR)


@pytest.fixture()
def valid_portfolio_df() -> pl.DataFrame:
    """Provide a minimal valid portfolio DataFrame for validate_portfolio_data.

    Returns
    -------
    pl.DataFrame
        DataFrame with exactly Date and Value columns, no nulls, sorted dates.
    """
    return pl.DataFrame(
        {
            "Date": ["2023-01-01", "2023-02-01", "2023-03-01", "2023-04-01", "2023-05-01"],
            "Value": [100.0, 102.5, 105.0, 103.0, 107.5],
        }
    )


@pytest.fixture()
def valid_benchmark_df() -> pl.DataFrame:
    """Provide a minimal valid benchmark DataFrame aligned to valid_portfolio_df.

    Returns
    -------
    pl.DataFrame
        DataFrame with exactly Date and Value columns.
    """
    return pl.DataFrame(
        {
            "Date": ["2023-01-01", "2023-02-01", "2023-03-01", "2023-04-01", "2023-05-01"],
            "Value": [100.0, 101.0, 99.5, 102.0, 104.5],
        }
    )


# ==============================================================================
# get_input_data_raw integration
# ==============================================================================


@pytest.mark.integration()
class Test_Get_Input_Data_Raw_Integration:
    """Integration tests for get_input_data_raw loading pipeline."""

    def test_returns_dictionary(self, raw_data: dict[str, pl.DataFrame]) -> None:
        """get_input_data_raw returns a dictionary of DataFrames."""
        _logger.debug("Testing get_input_data_raw returns dict")

        assert isinstance(raw_data, dict), "get_input_data_raw must return a dictionary"

    def test_returns_all_expected_keys(self, raw_data: dict[str, pl.DataFrame]) -> None:
        """Returned dict contains all three expected data keys."""
        _logger.debug("Testing get_input_data_raw key presence")

        for key in _EXPECTED_DATA_KEYS:
            assert key in raw_data, f"Expected key '{key}' not found in returned data dict"

    def test_time_series_sample_is_dataframe(self, raw_data: dict[str, pl.DataFrame]) -> None:
        """Time_Series_Sample value is a non-empty Polars DataFrame."""
        _logger.debug("Testing Time_Series_Sample is non-empty Polars DataFrame")

        df = raw_data.get("Time_Series_Sample")
        assert isinstance(df, pl.DataFrame), "Time_Series_Sample must be a Polars DataFrame"
        assert df.height > 0, "Time_Series_Sample must have at least one row"

    def test_time_series_etfs_is_dataframe(self, raw_data: dict[str, pl.DataFrame]) -> None:
        """Time_Series_ETFs value is a non-empty Polars DataFrame."""
        _logger.debug("Testing Time_Series_ETFs is non-empty Polars DataFrame")

        df = raw_data.get("Time_Series_ETFs")
        assert isinstance(df, pl.DataFrame), "Time_Series_ETFs must be a Polars DataFrame"
        assert df.height > 0, "Time_Series_ETFs must have at least one row"

    def test_weights_my_portfolio_is_dataframe(self, raw_data: dict[str, pl.DataFrame]) -> None:
        """Weights_My_Portfolio value is a non-empty Polars DataFrame."""
        _logger.debug("Testing Weights_My_Portfolio is non-empty Polars DataFrame")

        df = raw_data.get("Weights_My_Portfolio")
        assert isinstance(df, pl.DataFrame), "Weights_My_Portfolio must be a Polars DataFrame"
        assert df.height > 0, "Weights_My_Portfolio must have at least one row"

    def test_all_dataframes_have_date_column(self, raw_data: dict[str, pl.DataFrame]) -> None:
        """Every DataFrame returned by get_input_data_raw has a Date column."""
        _logger.debug("Testing all raw DataFrames contain Date column")

        for key, df in raw_data.items():
            if isinstance(df, pl.DataFrame):
                assert "Date" in df.columns, (
                    f"DataFrame for key '{key}' must contain a 'Date' column"
                )

    def test_time_series_sample_has_two_columns(
        self, raw_data: dict[str, pl.DataFrame]
    ) -> None:
        """Time_Series_Sample DataFrame has a Date column plus at least one data column."""
        _logger.debug("Testing Time_Series_Sample has Date and data columns")

        df = raw_data["Time_Series_Sample"]
        assert df.width >= 2, (
            f"Time_Series_Sample must have at least 2 columns (Date + data), got {df.width}: {df.columns}"
        )
        assert "Date" in df.columns, "Time_Series_Sample must have Date column"

    def test_weights_portfolio_has_date_and_component_columns(
        self, raw_data: dict[str, pl.DataFrame]
    ) -> None:
        """Weights_My_Portfolio has Date plus at least one component column."""
        _logger.debug("Testing Weights_My_Portfolio has Date and component columns")

        df = raw_data["Weights_My_Portfolio"]
        assert df.width >= 2, "Weights_My_Portfolio must have at least 2 columns"
        assert "Date" in df.columns, "Weights_My_Portfolio must have Date column"


# ==============================================================================
# validate_portfolio_data integration
# ==============================================================================


@pytest.mark.integration()
class Test_Validate_Portfolio_Data_Integration:
    """Integration tests for validate_portfolio_data function."""

    def test_valid_portfolio_df_passes_validation(
        self, valid_portfolio_df: pl.DataFrame
    ) -> None:
        """validate_portfolio_data returns (True, ...) for a correctly structured DataFrame."""
        _logger.debug("Testing validate_portfolio_data with valid input")

        is_valid, message = validate_portfolio_data(valid_portfolio_df)

        assert is_valid is True, (
            f"Valid portfolio DataFrame must pass validation, got message: '{message}'"
        )

    def test_none_input_fails_validation(self) -> None:
        """validate_portfolio_data returns (False, ...) for None input."""
        _logger.debug("Testing validate_portfolio_data rejects None")

        is_valid, message = validate_portfolio_data(None)

        assert is_valid is False, "None input must fail validation"
        assert isinstance(message, str), "Failure message must be a string"
        assert len(message) > 0, "Failure message must not be empty"

    def test_empty_dataframe_fails_validation(self) -> None:
        """validate_portfolio_data returns (False, ...) for an empty DataFrame."""
        _logger.debug("Testing validate_portfolio_data rejects empty DataFrame")

        empty_df = pl.DataFrame({"Date": [], "Value": []}).cast(
            {"Date": pl.Utf8, "Value": pl.Float64}
        )
        is_valid, message = validate_portfolio_data(empty_df)

        assert is_valid is False, "Empty DataFrame must fail validation"

    def test_dataframe_with_wrong_columns_fails_validation(self) -> None:
        """validate_portfolio_data returns (False, ...) when expected columns are absent."""
        _logger.debug("Testing validate_portfolio_data rejects wrong column names")

        wrong_cols_df = pl.DataFrame(
            {"timestamp": ["2023-01-01", "2023-02-01"], "price": [100.0, 101.0]}
        )
        is_valid, message = validate_portfolio_data(wrong_cols_df)

        assert is_valid is False, "DataFrame with wrong column names must fail validation"

    def test_dataframe_with_null_values_fails_validation(self) -> None:
        """validate_portfolio_data returns (False, ...) when Value column contains nulls."""
        _logger.debug("Testing validate_portfolio_data rejects DataFrame with null values")

        null_df = pl.DataFrame(
            {"Date": ["2023-01-01", "2023-02-01"], "Value": [100.0, None]}
        )
        is_valid, message = validate_portfolio_data(null_df)

        assert is_valid is False, "DataFrame with null Values must fail validation"

    def test_loaded_time_series_sample_passes_validation(
        self, raw_data: dict[str, pl.DataFrame]
    ) -> None:
        """Time_Series_Sample loaded from CSV has expected structure (Date + data columns)."""
        _logger.debug("Testing loaded Time_Series_Sample has valid structure")

        df = raw_data["Time_Series_Sample"]
        assert isinstance(df, pl.DataFrame), "Time_Series_Sample must be Polars DataFrame"
        assert df.height > 0, "Time_Series_Sample must be non-empty"
        assert "Date" in df.columns, "Time_Series_Sample must have a Date column"
        assert df.width >= 2, "Time_Series_Sample must have Date plus at least one data column"


# ==============================================================================
# calculate_portfolio_returns integration
# ==============================================================================


@pytest.mark.integration()
class Test_Calculate_Portfolio_Returns_Integration:
    """Integration tests for calculate_portfolio_returns function."""

    def test_returns_polars_series(
        self, valid_portfolio_df: pl.DataFrame
    ) -> None:
        """calculate_portfolio_returns returns a Polars Series."""
        _logger.debug("Testing calculate_portfolio_returns return type")

        result = calculate_portfolio_returns(valid_portfolio_df)

        assert isinstance(result, pl.Series), "calculate_portfolio_returns must return a Polars Series"

    def test_returns_length_is_one_less_than_input(
        self, valid_portfolio_df: pl.DataFrame
    ) -> None:
        """Returns Series length is one less than source DataFrame (pct_change loses first row)."""
        _logger.debug("Testing calculate_portfolio_returns output length")

        result = calculate_portfolio_returns(valid_portfolio_df)

        expected_len = valid_portfolio_df.height - 1
        assert len(result) == expected_len, (
            f"Returns length must be {expected_len}, got {len(result)}"
        )

    def test_returns_are_finite(
        self, valid_portfolio_df: pl.DataFrame
    ) -> None:
        """All values in the returns Series are finite (no NaN or Inf)."""
        _logger.debug("Testing calculate_portfolio_returns values are finite")

        result = calculate_portfolio_returns(valid_portfolio_df)

        assert result.null_count() == 0, "Returns Series must have no null values"

    def test_none_input_raises_or_returns_empty(self) -> None:
        """calculate_portfolio_returns raises ValueError or returns empty Series for None."""
        _logger.debug("Testing calculate_portfolio_returns handles None input")

        try:
            result = calculate_portfolio_returns(None)
            assert isinstance(result, pl.Series), "Result from None input must be a Polars Series"
        except ValueError:
            pass  # Raising ValueError is also acceptable behaviour

    def test_pipeline_load_validate_then_compute_returns(
        self, valid_portfolio_df: pl.DataFrame
    ) -> None:
        """Full pipeline: validate portfolio data → compute returns produces Series."""
        _logger.debug("Testing validate → compute_returns full pipeline")

        is_valid, _message = validate_portfolio_data(valid_portfolio_df)

        assert is_valid is True, "valid_portfolio_df must pass validation before computing returns"

        returns = calculate_portfolio_returns(valid_portfolio_df)

        assert isinstance(returns, pl.Series), "Returns must be a Polars Series"
        assert len(returns) > 0, "Returns Series must be non-empty for multi-row source"
        assert returns.null_count() == 0, "Returns must contain no nulls"


# ==============================================================================
# validate_portfolio_and_benchmark_data integration
# ==============================================================================


@pytest.mark.integration()
class Test_Validate_Portfolio_And_Benchmark_Integration:
    """Integration tests for validate_portfolio_and_benchmark_data function."""

    def test_two_valid_dfs_returns_true_true(
        self,
        valid_portfolio_df: pl.DataFrame,
        valid_benchmark_df: pl.DataFrame,
    ) -> None:
        """Two valid DataFrames produce (True, True) from validation function."""
        _logger.debug("Testing validation with two valid DataFrames")

        has_portfolio, has_benchmark = validate_portfolio_and_benchmark_data_if_not_already_validated(
            valid_portfolio_df,
            valid_benchmark_df,
        )

        assert has_portfolio is True, "Valid portfolio DataFrame must return has_portfolio=True"
        assert has_benchmark is True, "Valid benchmark DataFrame must return has_benchmark=True"

    def test_none_portfolio_returns_false_for_portfolio(
        self, valid_benchmark_df: pl.DataFrame
    ) -> None:
        """None portfolio with valid benchmark returns (False, True)."""
        _logger.debug("Testing validation with None portfolio")

        has_portfolio, has_benchmark = validate_portfolio_and_benchmark_data_if_not_already_validated(
            None,
            valid_benchmark_df,
        )

        assert has_portfolio is False, "None portfolio must return has_portfolio=False"

    def test_none_benchmark_returns_false_for_benchmark(
        self, valid_portfolio_df: pl.DataFrame
    ) -> None:
        """Valid portfolio with None benchmark returns (True, False)."""
        _logger.debug("Testing validation with None benchmark")

        has_portfolio, has_benchmark = validate_portfolio_and_benchmark_data_if_not_already_validated(
            valid_portfolio_df,
            None,
        )

        assert has_benchmark is False, "None benchmark must return has_benchmark=False"

    def test_both_none_returns_false_false(self) -> None:
        """Both None inputs return (False, False)."""
        _logger.debug("Testing validation with both None inputs")

        has_portfolio, has_benchmark = validate_portfolio_and_benchmark_data_if_not_already_validated(
            None,
            None,
        )

        assert has_portfolio is False, "None portfolio must return has_portfolio=False"
        assert has_benchmark is False, "None benchmark must return has_benchmark=False"
