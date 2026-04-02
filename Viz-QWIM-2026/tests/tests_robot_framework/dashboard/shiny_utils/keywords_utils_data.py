"""
Robot Framework keyword library for dashboard shiny_utils.
===========================================================

Tests cover:
    utils_data.py
    - get_input_data_raw       -- loads raw CSV files from inputs/raw/
    - validate_portfolio_data  -- 2-column (Date, Value) DataFrame check
    - calculate_portfolio_returns -- pct-change Series from portfolio values

    utils_tab_results.py
    - process_data_for_plot_tab_results -- passthrough / downsampling
    - normalize_data_tab_results        -- min_max / z_score / none
    - transform_data_tab_results        -- percent_change / cumulative / none

Author:         QWIM Development Team
Version:        0.1.0
Last Modified:  2026-02-28
"""

from __future__ import annotations

import sys
import io
from pathlib import Path

# ---------------------------------------------------------------------------
# Project root on sys.path so src packages resolve correctly
# ---------------------------------------------------------------------------
_PROJECT_ROOT: Path = Path(__file__).resolve().parents[4]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# ---------------------------------------------------------------------------
# Robot Framework redirects sys.stderr to a StringIO object that lacks
# a .buffer attribute.  Patch it back before importing modules that
# access sys.stderr.buffer at module-load time.
# ---------------------------------------------------------------------------
if not hasattr(sys.stderr, "buffer"):
    sys.stderr = io.TextIOWrapper(io.BytesIO(), encoding="utf-8")

# ---------------------------------------------------------------------------
# Conditional imports with Module availability guard
# ---------------------------------------------------------------------------
MODULE_IMPORT_AVAILABLE: bool = True
_import_error_message: str = ""

try:
    import polars as pl
    from src.dashboard.shiny_utils.utils_data import (
        get_input_data_raw,
        validate_portfolio_data,
        calculate_portfolio_returns,
    )
    from src.dashboard.shiny_utils.utils_tab_results import (
        process_data_for_plot_tab_results,
        normalize_data_tab_results,
        transform_data_tab_results,
    )
except Exception as _exc:
    MODULE_IMPORT_AVAILABLE = False
    _import_error_message = str(_exc)


def _require_imports() -> None:
    """Raise RuntimeError when required imports are unavailable."""
    if not MODULE_IMPORT_AVAILABLE:
        raise RuntimeError(
            f"Required imports not available: {_import_error_message}"
        )


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _make_valid_portfolio_df(rows: int = 5) -> "pl.DataFrame":
    """Build a minimal valid portfolio DataFrame (Date + Value, 2 columns)."""
    import datetime as dt
    start = dt.date(2024, 1, 2)
    return pl.DataFrame({
        "Date": [start + dt.timedelta(days=i) for i in range(rows)],
        "Value": [100.0 + float(i) * 1.5 for i in range(rows)],
    })


# ===========================================================================
# Keywords — utils_data.py
# ===========================================================================


# ---------------------------------------------------------------------------
# get_input_data_raw
# ---------------------------------------------------------------------------


def load_raw_input_data() -> dict:
    """Call get_input_data_raw(_PROJECT_ROOT) and return the result dict.

    Returns: dict with keys 'Time_Series_Sample', 'Time_Series_ETFs',
             'Weights_My_Portfolio'
    """
    _require_imports()
    return get_input_data_raw(_PROJECT_ROOT)


def raw_input_data_should_contain_key(data: dict, key: str) -> None:
    """Assert that the raw input dict contains the given key.

    Arguments:
    - data -- dict returned by load_raw_input_data()
    - key  -- expected key string
    """
    _require_imports()
    if key not in data:
        raise AssertionError(
            f"Expected key '{key}' not found. Available keys: {list(data.keys())}"
        )


def raw_input_data_value_should_be_non_empty_dataframe(
    data: dict, key: str
) -> None:
    """Assert data[key] is a non-empty pl.DataFrame.

    Arguments:
    - data -- dict returned by load_raw_input_data()
    - key  -- key whose value should be a non-empty DataFrame
    """
    _require_imports()
    if key not in data:
        raise AssertionError(f"Key '{key}' not found in raw input data")
    df = data[key]
    if not isinstance(df, pl.DataFrame):
        raise AssertionError(
            f"data['{key}'] expected pl.DataFrame, got {type(df)}"
        )
    if len(df) == 0:
        raise AssertionError(f"data['{key}'] DataFrame is empty")


# ---------------------------------------------------------------------------
# validate_portfolio_data
# ---------------------------------------------------------------------------


def validate_valid_portfolio_dataframe() -> tuple:
    """Call validate_portfolio_data with a correctly shaped 2-col DataFrame.

    Returns: (is_valid, message) tuple
    """
    _require_imports()
    df = _make_valid_portfolio_df()
    return validate_portfolio_data(df)


def validate_none_portfolio_data() -> tuple:
    """Call validate_portfolio_data(None).

    Returns: (False, error_message) tuple
    """
    _require_imports()
    return validate_portfolio_data(None)


def validate_portfolio_data_with_wrong_columns() -> tuple:
    """Call validate_portfolio_data with a DataFrame missing the Value column.

    Returns: (False, error_message) tuple
    """
    _require_imports()
    import datetime as dt
    df = pl.DataFrame({
        "Date": [dt.date(2024, 1, 2)],
        "SomeOtherColumn": [100.0],
    })
    return validate_portfolio_data(df)


def validate_portfolio_data_with_extra_column() -> tuple:
    """Call validate_portfolio_data with a 3-column DataFrame (should fail).

    Returns: (False, error_message) tuple
    """
    _require_imports()
    import datetime as dt
    df = pl.DataFrame({
        "Date": [dt.date(2024, 1, 2)],
        "Value": [100.0],
        "Extra": [99.0],
    })
    return validate_portfolio_data(df)


def validation_result_should_be_valid(result: tuple) -> None:
    """Assert first element of (is_valid, message) tuple is True.

    Arguments:
    - result -- (bool, str) tuple from validate_portfolio_data
    """
    _require_imports()
    is_valid, message = result
    if not is_valid:
        raise AssertionError(f"Expected valid portfolio data but got: {message}")


def validation_result_should_be_invalid(result: tuple) -> None:
    """Assert first element of (is_valid, message) tuple is False.

    Arguments:
    - result -- (bool, str) tuple from validate_portfolio_data
    """
    _require_imports()
    is_valid, message = result
    if is_valid:
        raise AssertionError("Expected invalid portfolio data but validation returned True")


def validation_error_message_should_not_be_empty(result: tuple) -> None:
    """Assert the error message string is non-empty when is_valid is False.

    Arguments:
    - result -- (bool, str) tuple from validate_portfolio_data
    """
    _require_imports()
    is_valid, message = result
    if not is_valid and not message:
        raise AssertionError("Invalid validation result has empty error message")


# ---------------------------------------------------------------------------
# calculate_portfolio_returns
# ---------------------------------------------------------------------------


def calculate_returns_from_valid_portfolio_df() -> "pl.Series":
    """Call calculate_portfolio_returns on a valid 2-col portfolio DataFrame.

    Returns: pl.Series of percentage returns
    """
    _require_imports()
    df = _make_valid_portfolio_df(rows=10)
    return calculate_portfolio_returns(df)


def portfolio_returns_series_should_be_valid(returns: "pl.Series") -> None:
    """Assert that calculate_portfolio_returns returned a non-empty Series.

    Arguments:
    - returns -- pl.Series from calculate_portfolio_returns
    """
    _require_imports()
    if not isinstance(returns, pl.Series):
        raise AssertionError(
            f"Expected pl.Series from calculate_portfolio_returns, got {type(returns)}"
        )
    if len(returns) == 0:
        raise AssertionError("Portfolio returns Series is empty")


# ===========================================================================
# Keywords — utils_tab_results.py
# ===========================================================================

# ---------------------------------------------------------------------------
# Shared helper to create a small time-series DataFrame for tab results tests
# ---------------------------------------------------------------------------


def _make_tab_results_df(rows: int = 20) -> "pl.DataFrame":
    """Build a DataFrame with 'date' (string) and numeric columns."""
    import datetime as dt
    start = dt.date(2024, 1, 2)
    dates = [(start + dt.timedelta(days=i)).isoformat() for i in range(rows)]
    return pl.DataFrame({
        "date": dates,
        "value_a": [100.0 + float(i) * 2.0 for i in range(rows)],
        "value_b": [80.0 + float(i) * 1.5 for i in range(rows)],
    })


# ---------------------------------------------------------------------------
# process_data_for_plot_tab_results
# ---------------------------------------------------------------------------


def process_small_dataframe_for_plot() -> "pl.DataFrame":
    """Process a small DataFrame (rows < max_points=5000) — returns unchanged.

    Returns: pl.DataFrame (unchanged)
    """
    _require_imports()
    df = _make_tab_results_df(rows=20)
    return process_data_for_plot_tab_results(df, date_column="date", max_points=5000)


def processed_dataframe_should_equal_input_for_small_data(result: "pl.DataFrame") -> None:
    """Assert that small data is returned unchanged by process_data_for_plot.

    Arguments:
    - result -- return value of process_small_dataframe_for_plot()
    """
    _require_imports()
    if not isinstance(result, pl.DataFrame):
        raise AssertionError(
            f"Expected pl.DataFrame, got {type(result)}"
        )
    if len(result) != 20:
        raise AssertionError(
            f"Expected 20 rows for small data passthrough, got {len(result)}"
        )


def process_large_dataframe_returns_fewer_rows() -> None:
    """Process a DataFrame larger than max_points and verify row reduction.

    Creates a 200-row DataFrame with max_points=50 and asserts the result
    has fewer rows than the input.
    """
    _require_imports()
    rows = 200
    max_pts = 50
    import datetime as dt
    start = dt.date(2024, 1, 2)
    dates = [(start + dt.timedelta(days=i)).isoformat() for i in range(rows)]
    df = pl.DataFrame({
        "date": dates,
        "value_a": [float(i) for i in range(rows)],
    })
    result = process_data_for_plot_tab_results(df, date_column="date", max_points=max_pts)
    if len(result) >= rows:
        raise AssertionError(
            f"Expected fewer than {rows} rows after downsampling but got {len(result)}"
        )


# ---------------------------------------------------------------------------
# normalize_data_tab_results
# ---------------------------------------------------------------------------


def normalize_with_none_method_returns_unchanged() -> "pl.DataFrame":
    """normalize_data_tab_results with method='none' returns unchanged DataFrame.

    Returns: pl.DataFrame
    """
    _require_imports()
    df = _make_tab_results_df(rows=10)
    return normalize_data_tab_results(df, method="none")


def normalize_with_min_max_returns_values_in_zero_one_range(rows: int = 20) -> None:
    """Assert min-max normalisation scales all numeric values to [0.0, 1.0].

    Arguments:
    - rows -- number of rows in test DataFrame (default 20)
    """
    _require_imports()
    df = _make_tab_results_df(rows=int(rows))
    result = normalize_data_tab_results(df, method="min_max")
    numeric_cols = [c for c in result.columns if c != "date"]
    for col in numeric_cols:
        col_min = result[col].min()
        col_max = result[col].max()
        if col_min is None or col_max is None:
            continue
        if col_min < -1.0e-9 or col_max > 1.0 + 1.0e-9:
            raise AssertionError(
                f"Column '{col}' values after min_max normalisation outside [0,1]: "
                f"min={col_min}, max={col_max}"
            )


def normalize_with_z_score_returns_dataframe() -> "pl.DataFrame":
    """normalize_data_tab_results with method='z_score' returns a DataFrame.

    Returns: pl.DataFrame
    """
    _require_imports()
    df = _make_tab_results_df(rows=20)
    return normalize_data_tab_results(df, method="z_score")


def normalised_dataframe_should_be_valid(result: "pl.DataFrame", expected_rows: int | str) -> None:
    """Assert result is a non-empty DataFrame with the expected row count.

    Arguments:
    - result        -- pl.DataFrame after normalisation
    - expected_rows -- expected number of rows
    """
    _require_imports()
    if not isinstance(result, pl.DataFrame):
        raise AssertionError(
            f"Expected pl.DataFrame after normalisation, got {type(result)}"
        )
    actual = len(result)
    expected = int(expected_rows)
    if actual != expected:
        raise AssertionError(
            f"Expected {expected} rows after normalisation but got {actual}"
        )


# ---------------------------------------------------------------------------
# transform_data_tab_results
# ---------------------------------------------------------------------------


def transform_with_none_returns_dataframe_unchanged() -> "pl.DataFrame":
    """transform_data_tab_results with transformation='none' returns unchanged data.

    Returns: pl.DataFrame
    """
    _require_imports()
    df = _make_tab_results_df(rows=10)
    return transform_data_tab_results(df, transformation="none")


def transform_with_percent_change_adds_pct_columns() -> "pl.DataFrame":
    """transform_data_tab_results with 'percent_change' adds _pct suffix columns.

    Returns: pl.DataFrame with additional *_pct columns
    """
    _require_imports()
    df = _make_tab_results_df(rows=10)
    return transform_data_tab_results(df, transformation="percent_change")


def transformed_dataframe_should_contain_pct_columns(result: "pl.DataFrame") -> None:
    """Assert that percent_change transformation added at least one _pct column.

    Arguments:
    - result -- pl.DataFrame from transform_with_percent_change_adds_pct_columns()
    """
    _require_imports()
    pct_cols = [c for c in result.columns if c.endswith("_pct")]
    if not pct_cols:
        raise AssertionError(
            f"Expected _pct columns after percent_change transformation. "
            f"Columns found: {result.columns}"
        )


def transform_with_cumulative_adds_cum_columns() -> "pl.DataFrame":
    """transform_data_tab_results with 'cumulative' adds _cum suffix columns.

    Returns: pl.DataFrame with additional *_cum columns
    """
    _require_imports()
    df = _make_tab_results_df(rows=10)
    return transform_data_tab_results(df, transformation="cumulative")


def transformed_dataframe_should_contain_cum_columns(result: "pl.DataFrame") -> None:
    """Assert that cumulative transformation added at least one _cum column.

    Arguments:
    - result -- pl.DataFrame from transform_with_cumulative_adds_cum_columns()
    """
    _require_imports()
    cum_cols = [c for c in result.columns if c.endswith("_cum")]
    if not cum_cols:
        raise AssertionError(
            f"Expected _cum columns after cumulative transformation. "
            f"Columns found: {result.columns}"
        )


def cumulative_values_should_be_monotonically_increasing(result: "pl.DataFrame") -> None:
    """Assert that all _cum columns are non-decreasing (all values positive input).

    Arguments:
    - result -- pl.DataFrame with _cum columns
    """
    _require_imports()
    cum_cols = [c for c in result.columns if c.endswith("_cum")]
    for col in cum_cols:
        values = result[col].to_list()
        for idx in range(1, len(values)):
            if values[idx] < values[idx - 1]:
                raise AssertionError(
                    f"Cumulative column '{col}' not non-decreasing at index {idx}: "
                    f"{values[idx - 1]} > {values[idx]}"
                )
