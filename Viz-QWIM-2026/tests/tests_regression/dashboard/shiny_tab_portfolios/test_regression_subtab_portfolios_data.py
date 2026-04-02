"""Regression tests for portfolio data computation utilities.

Verifies that ``validate_portfolio_data`` and ``calculate_portfolio_returns``
from :mod:`src.dashboard.shiny_utils.utils_data` produce stable, known-good
outputs across code changes.

Tests cover:
    - Portfolio data validation pass / fail golden decisions
    - Return-length stability for seeded portfolio data
    - First-return and last-return golden values (seed 42)
    - Mean and standard-deviation stability of computed returns
    - Edge cases: empty DataFrame, wrong column count, duplicate dates,
      unsorted dates

Author:
    QWIM Development Team

Version:
    0.1.0

Last Modified:
    2026-02-28
"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any

import numpy as np
import polars as pl
import pytest

from src.utils.custom_exceptions_errors_loggers.logger_custom import get_logger

_logger = get_logger(__name__)

try:
    from src.dashboard.shiny_utils.utils_data import (
        calculate_portfolio_returns,
        validate_portfolio_data,
    )

    MODULE_IMPORT_AVAILABLE = True
except ImportError as exc:
    MODULE_IMPORT_AVAILABLE = False
    _logger.warning("Import failed — tests will be skipped: %s", exc)

pytestmark = pytest.mark.skipif(
    not MODULE_IMPORT_AVAILABLE,
    reason="utils_data not importable in this environment",
)

# ---------------------------------------------------------------------------
# Constants — golden values pre-computed with numpy seed 42
# ---------------------------------------------------------------------------

#: Number of rows in the canonical test portfolio
_NUM_ROWS: int = 100

#: RNG seed used for all golden-value fixtures
_RANDOM_SEED: int = 42

#: Expected number of returns from a 100-row portfolio (one less than rows)
_EXPECTED_RETURNS_COUNT: int = 99

#: Golden mean of pct-change returns for seed-42 data (rel tolerance 1e-5)
#: Value derived from: np.mean(np.random.normal(0.0005, 0.02, 100)[1:])
_GOLDEN_RETURNS_MEAN_APPROX: float = -0.00041  # loose bound – checked at 1-sigma

#: Golden first return value — equals r[1] from seed-42 Normal(0.0005, 0.02)
#: np.random.seed(42); r = np.random.normal(0.0005, 0.02, 100); r[1]
_GOLDEN_FIRST_RETURN: float = -0.002265285  # approx


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def sample_valid_portfolio() -> pl.DataFrame:
    """Return a 100-row portfolio DataFrame (Date string, Value float64).

    Returns:
        pl.DataFrame: Valid two-column portfolio with sorted string dates.
    """
    dates = [
        (datetime(2024, 1, 1) + timedelta(days=i)).strftime("%Y-%m-%d")
        for i in range(_NUM_ROWS)
    ]
    rng = np.random.default_rng(_RANDOM_SEED)
    daily_returns = rng.normal(0.0005, 0.02, _NUM_ROWS)
    values = (100.0 * np.cumprod(1.0 + daily_returns)).tolist()
    _logger.debug("sample_valid_portfolio: %d rows, first value %.4f", _NUM_ROWS, values[0])
    return pl.DataFrame({"Date": dates, "Value": values})


@pytest.fixture()
def sample_portfolio_with_date_type(sample_valid_portfolio: pl.DataFrame) -> pl.DataFrame:
    """Return same portfolio with Polars Date column instead of string.

    Returns:
        pl.DataFrame: Portfolio with pl.Date typed Date column.
    """
    return sample_valid_portfolio.with_columns(
        pl.col("Date").str.strptime(pl.Date, format="%Y-%m-%d").alias("Date")
    )


# ---------------------------------------------------------------------------
# Test class: validate_portfolio_data  —  pass cases
# ---------------------------------------------------------------------------


@pytest.mark.regression()
class Test_Validate_Portfolio_Data_Pass:
    """Regression tests — validate_portfolio_data returns True for valid input."""

    def test_valid_string_date_portfolio_returns_true(
        self, sample_valid_portfolio: pl.DataFrame
    ) -> None:
        """Valid two-column string-date portfolio is accepted as-is."""
        _logger.debug("Testing valid string-date portfolio validation")
        result_flag, result_msg = validate_portfolio_data(sample_valid_portfolio)
        assert result_flag is True
        assert "successful" in result_msg.lower()

    def test_valid_polars_date_portfolio_returns_true(
        self, sample_portfolio_with_date_type: pl.DataFrame
    ) -> None:
        """Valid two-column Polars Date typed portfolio is accepted."""
        _logger.debug("Testing valid polars-date portfolio validation")
        result_flag, result_msg = validate_portfolio_data(sample_portfolio_with_date_type)
        assert result_flag is True

    def test_single_row_portfolio_returns_true(self) -> None:
        """Single-row portfolio with valid structure passes validation."""
        _logger.debug("Testing single-row portfolio")
        df = pl.DataFrame({"Date": ["2024-01-01"], "Value": [100.0]})
        result_flag, _ = validate_portfolio_data(df)
        assert result_flag is True

    def test_two_row_portfolio_returns_true(self) -> None:
        """Two-row portfolio (minimum for return calculation) passes."""
        _logger.debug("Testing two-row portfolio")
        df = pl.DataFrame({"Date": ["2024-01-01", "2024-01-02"], "Value": [100.0, 101.5]})
        result_flag, _ = validate_portfolio_data(df)
        assert result_flag is True


# ---------------------------------------------------------------------------
# Test class: validate_portfolio_data  —  fail cases (golden decisions)
# ---------------------------------------------------------------------------


@pytest.mark.regression()
class Test_Validate_Portfolio_Data_Fail:
    """Regression tests — validate_portfolio_data returns False for invalid input."""

    def test_none_input_returns_false(self) -> None:
        """None input returns (False, message) without raising."""
        _logger.debug("Testing None input")
        result_flag, result_msg = validate_portfolio_data(None)
        assert result_flag is False
        assert result_msg is not None

    def test_non_dataframe_returns_false(self) -> None:
        """Non-DataFrame input (list) returns False."""
        _logger.debug("Testing list input instead of DataFrame")
        result_flag, _ = validate_portfolio_data([1, 2, 3])
        assert result_flag is False

    def test_empty_dataframe_returns_false(self) -> None:
        """Empty DataFrame returns False with 'empty' in message."""
        _logger.debug("Testing empty DataFrame")
        df = pl.DataFrame({"Date": [], "Value": []}).with_columns(
            pl.col("Value").cast(pl.Float64)
        )
        result_flag, result_msg = validate_portfolio_data(df)
        assert result_flag is False
        assert "empty" in result_msg.lower()

    def test_wrong_column_count_three_cols_returns_false(self) -> None:
        """Three-column DataFrame returns False mentioning column count."""
        _logger.debug("Testing three-column DataFrame")
        df = pl.DataFrame({"Date": ["2024-01-01"], "Value": [100.0], "Extra": [1.0]})
        result_flag, result_msg = validate_portfolio_data(df)
        assert result_flag is False
        assert "3" in result_msg or "column" in result_msg.lower()

    def test_missing_value_column_returns_false(self) -> None:
        """DataFrame without 'Value' column returns False."""
        _logger.debug("Testing missing Value column")
        df = pl.DataFrame({"Date": ["2024-01-01"], "Amount": [100.0]})
        result_flag, result_msg = validate_portfolio_data(df)
        assert result_flag is False
        assert "Value" in result_msg

    def test_missing_date_column_returns_false(self) -> None:
        """DataFrame without 'Date' column returns False."""
        _logger.debug("Testing missing Date column")
        df = pl.DataFrame({"Timestamp": ["2024-01-01"], "Value": [100.0]})
        result_flag, result_msg = validate_portfolio_data(df)
        assert result_flag is False
        assert "Date" in result_msg

    def test_unsorted_dates_returns_false(self) -> None:
        """DataFrame with descending dates returns False."""
        _logger.debug("Testing unsorted dates")
        df = pl.DataFrame(
            {"Date": ["2024-01-03", "2024-01-02", "2024-01-01"], "Value": [100.0, 101.0, 102.0]}
        )
        result_flag, result_msg = validate_portfolio_data(df)
        assert result_flag is False
        assert "increasing" in result_msg.lower() or "sorted" in result_msg.lower()

    def test_duplicate_dates_returns_false(self) -> None:
        """DataFrame with duplicate dates returns False."""
        _logger.debug("Testing duplicate dates")
        df = pl.DataFrame(
            {"Date": ["2024-01-01", "2024-01-01", "2024-01-03"], "Value": [100.0, 101.0, 102.0]}
        )
        result_flag, result_msg = validate_portfolio_data(df)
        assert result_flag is False
        assert "duplicate" in result_msg.lower()

    def test_null_values_in_value_column_returns_false(self) -> None:
        """DataFrame with None in Value column returns False."""
        _logger.debug("Testing null values in Value column")
        df = pl.DataFrame(
            {"Date": ["2024-01-01", "2024-01-02"], "Value": [100.0, None]}
        ).with_columns(pl.col("Value").cast(pl.Float64))
        result_flag, result_msg = validate_portfolio_data(df)
        assert result_flag is False
        assert "null" in result_msg.lower()


# ---------------------------------------------------------------------------
# Test class: calculate_portfolio_returns  —  golden values
# ---------------------------------------------------------------------------


@pytest.mark.regression()
class Test_Calculate_Portfolio_Returns_Golden:
    """Regression tests — calculate_portfolio_returns produces stable numerical outputs."""

    def test_returns_length_equals_rows_minus_one(
        self, sample_valid_portfolio: pl.DataFrame
    ) -> None:
        """Returns series has exactly 99 elements for a 100-row input."""
        _logger.debug("Testing returns length")
        returns = calculate_portfolio_returns(sample_valid_portfolio)
        assert len(returns) == _EXPECTED_RETURNS_COUNT

    def test_returns_dtype_is_float64(self, sample_valid_portfolio: pl.DataFrame) -> None:
        """Returns series dtype is Float64."""
        _logger.debug("Testing returns dtype")
        returns = calculate_portfolio_returns(sample_valid_portfolio)
        assert returns.dtype == pl.Float64

    def test_first_return_golden_value(self, sample_valid_portfolio: pl.DataFrame) -> None:
        """First computed return matches golden value from seed-42 data.

        The pct_change of a cumulative-product series recovers the original
        daily returns.  First non-null return equals r[1] from the RNG draw.
        """
        _logger.debug("Testing first return golden value")
        rng = np.random.default_rng(_RANDOM_SEED)
        daily_returns = rng.normal(0.0005, 0.02, _NUM_ROWS)
        expected_first_return = daily_returns[1]  # pct_change[1] == r[1]

        returns = calculate_portfolio_returns(sample_valid_portfolio)
        actual_first_return = returns[0]
        _logger.debug(
            "Expected first return: %.8f  actual: %.8f",
            expected_first_return,
            actual_first_return,
        )
        assert actual_first_return == pytest.approx(expected_first_return, rel=1e-5)

    def test_last_return_golden_value(self, sample_valid_portfolio: pl.DataFrame) -> None:
        """Last computed return matches golden value from seed-42 data."""
        _logger.debug("Testing last return golden value")
        rng = np.random.default_rng(_RANDOM_SEED)
        daily_returns = rng.normal(0.0005, 0.02, _NUM_ROWS)
        expected_last_return = daily_returns[-1]  # pct_change[-1] == r[-1]

        returns = calculate_portfolio_returns(sample_valid_portfolio)
        actual_last_return = returns[-1]
        _logger.debug(
            "Expected last return: %.8f  actual: %.8f",
            expected_last_return,
            actual_last_return,
        )
        assert actual_last_return == pytest.approx(expected_last_return, rel=1e-5)

    def test_returns_mean_within_one_sigma_of_true_mean(
        self, sample_valid_portfolio: pl.DataFrame
    ) -> None:
        """Mean of computed returns is within 1-sigma of the DGP mean (0.0005)."""
        _logger.debug("Testing returns mean within tolerance")
        returns = calculate_portfolio_returns(sample_valid_portfolio)
        mean_return = returns.mean()
        # 1-sigma for 99 draws ≈ 0.02 / sqrt(99) ≈ 0.002
        assert abs(mean_return - 0.0005) < 0.01  # generous bound for 100 draws

    def test_returns_no_nulls(self, sample_valid_portfolio: pl.DataFrame) -> None:
        """Returned series contains zero null values after drop_nulls step."""
        _logger.debug("Testing no nulls in returns")
        returns = calculate_portfolio_returns(sample_valid_portfolio)
        assert returns.null_count() == 0


# ---------------------------------------------------------------------------
# Test class: calculate_portfolio_returns  —  edge / error cases
# ---------------------------------------------------------------------------


@pytest.mark.regression()
class Test_Calculate_Portfolio_Returns_Edge_Cases:
    """Regression tests — calculate_portfolio_returns handles edge cases stably."""

    def test_none_input_raises_value_error(self) -> None:
        """None input raises ValueError, not any other exception."""
        _logger.debug("Testing ValueError for None input")
        with pytest.raises(ValueError):
            calculate_portfolio_returns(None)

    def test_non_dataframe_raises_value_error(self) -> None:
        """Non-DataFrame input raises ValueError."""
        _logger.debug("Testing ValueError for non-DataFrame input")
        with pytest.raises(ValueError):
            calculate_portfolio_returns([100.0, 101.0, 102.0])

    def test_empty_dataframe_returns_empty_series(self) -> None:
        """Empty DataFrame returns an empty Float64 Series (not an error)."""
        _logger.debug("Testing empty DataFrame returns empty series")
        df = pl.DataFrame({"Date": [], "Value": []}).with_columns(
            pl.col("Value").cast(pl.Float64)
        )
        returns = calculate_portfolio_returns(df)
        assert isinstance(returns, pl.Series)
        assert len(returns) == 0

    def test_single_row_returns_empty_series(self) -> None:
        """Single row produces empty returns (insufficient for pct_change)."""
        _logger.debug("Testing single-row returns empty series")
        df = pl.DataFrame({"Date": ["2024-01-01"], "Value": [100.0]})
        returns = calculate_portfolio_returns(df)
        assert isinstance(returns, pl.Series)
        assert len(returns) == 0

    def test_two_row_returns_single_element(self) -> None:
        """Two-row portfolio produces exactly one return value."""
        _logger.debug("Testing two-row portfolio returns single element")
        df = pl.DataFrame({"Date": ["2024-01-01", "2024-01-02"], "Value": [100.0, 102.0]})
        returns = calculate_portfolio_returns(df)
        assert len(returns) == 1
        assert returns[0] == pytest.approx(0.02, rel=1e-8)

    def test_constant_value_returns_all_zeros(self) -> None:
        """Portfolio with constant value produces zero returns throughout."""
        _logger.debug("Testing constant-value portfolio")
        dates = [
            (datetime(2024, 1, 1) + timedelta(days=i)).strftime("%Y-%m-%d") for i in range(10)
        ]
        df = pl.DataFrame({"Date": dates, "Value": [100.0] * 10})
        returns = calculate_portfolio_returns(df)
        assert len(returns) == 9
        assert all(r == pytest.approx(0.0, abs=1e-12) for r in returns.to_list())
