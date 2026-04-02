"""Behave step definitions for dashboard utils_data and utils_tab_results.

Tests cover:
    utils_data.py
    - get_input_data_raw       — loads raw CSV files from inputs/raw/
    - validate_portfolio_data  — 2-column (Date, Value) DataFrame check
    - calculate_portfolio_returns — pct-change Series from portfolio values

    utils_tab_results.py
    - process_data_for_plot_tab_results — passthrough / downsampling
    - normalize_data_tab_results        — min_max / z_score / none
    - transform_data_tab_results        — percent_change / cumulative / none

Author:         QWIM Development Team
Version:        0.1.0
Last Modified:  2026-02-28
"""

from __future__ import annotations

import datetime as dt
import io
import sys
from pathlib import Path

from behave import given, then, when

# ---------------------------------------------------------------------------
# Project root on sys.path
# ---------------------------------------------------------------------------
_PROJECT_ROOT: Path = Path(__file__).resolve().parents[4]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# ---------------------------------------------------------------------------
# sys.stderr patch for exception_custom.py compatibility
# ---------------------------------------------------------------------------
if not hasattr(sys.stderr, "buffer"):
    sys.stderr = io.TextIOWrapper(io.BytesIO(), encoding="utf-8")

# ---------------------------------------------------------------------------
# Module-level import guard
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
    if not MODULE_IMPORT_AVAILABLE:
        raise RuntimeError(
            f"Dashboard utils source modules could not be imported: {_import_error_message}"
        )


def _make_valid_portfolio_df(rows: int = 5) -> "pl.DataFrame":
    """Build a minimal valid 2-column portfolio DataFrame (Date + Value)."""
    start = dt.date(2024, 1, 2)
    return pl.DataFrame({
        "Date": [start + dt.timedelta(days=i) for i in range(rows)],
        "Value": [100.0 + float(i) * 1.5 for i in range(rows)],
    })


def _make_tab_results_df(rows: int = 20) -> "pl.DataFrame":
    """Build a multi-column DataFrame for tab-results utilities."""
    start = dt.date(2024, 1, 2)
    dates = [(start + dt.timedelta(days=i)).isoformat() for i in range(rows)]
    return pl.DataFrame({
        "date": dates,
        "value_a": [100.0 + float(i) * 2.0 for i in range(rows)],
        "value_b": [80.0 + float(i) * 1.5 for i in range(rows)],
    })


# ===========================================================================
# When / Then — get_input_data_raw
# ===========================================================================


@when(u'I load raw input data from the project directory')
def step_load_raw_input_data(context) -> None:
    _require_imports()
    context.raw_data = get_input_data_raw(_PROJECT_ROOT)


@then(u'the raw data should be loaded without error')
def step_raw_data_loaded(context) -> None:
    assert context.raw_data is not None, "Raw input data is None"
    assert isinstance(context.raw_data, dict), (
        f"Expected dict from get_input_data_raw, got {type(context.raw_data)}"
    )


@then(u'the raw data should contain key "{key}"')
def step_raw_data_contains_key(context, key: str) -> None:
    assert key in context.raw_data, (
        f"Expected key '{key}' not found. Available keys: {list(context.raw_data.keys())}"
    )


@then(u'the value at key "{key}" should be a non-empty DataFrame')
def step_raw_data_value_non_empty(context, key: str) -> None:
    _require_imports()
    assert key in context.raw_data, (
        f"Key '{key}' not found in raw input data"
    )
    df = context.raw_data[key]
    assert isinstance(df, pl.DataFrame), (
        f"data['{key}'] expected pl.DataFrame, got {type(df)}"
    )
    assert len(df) > 0, f"data['{key}'] DataFrame is empty"


# ===========================================================================
# When / Then — validate_portfolio_data
# ===========================================================================


@when(u'I validate a valid two-column portfolio DataFrame')
def step_validate_valid_df(context) -> None:
    _require_imports()
    df = _make_valid_portfolio_df()
    context.validation_result = validate_portfolio_data(df)


@when(u'I validate None as portfolio data')
def step_validate_none(context) -> None:
    _require_imports()
    context.validation_result = validate_portfolio_data(None)


@when(u'I validate a DataFrame with wrong column names')
def step_validate_wrong_columns(context) -> None:
    _require_imports()
    df = pl.DataFrame({
        "Date": [dt.date(2024, 1, 2)],
        "SomeOtherColumn": [100.0],
    })
    context.validation_result = validate_portfolio_data(df)


@when(u'I validate a DataFrame with an extra column')
def step_validate_extra_column(context) -> None:
    _require_imports()
    df = pl.DataFrame({
        "Date": [dt.date(2024, 1, 2)],
        "Value": [100.0],
        "Extra": [99.0],
    })
    context.validation_result = validate_portfolio_data(df)


@then(u'the validation result should be valid')
def step_validation_valid(context) -> None:
    is_valid, message = context.validation_result
    assert is_valid, f"Expected valid portfolio data but got: {message}"


@then(u'the validation result first element should be True')
def step_validation_first_element_true(context) -> None:
    is_valid, _ = context.validation_result
    assert is_valid is True, (
        f"Expected True as first element of validation result but got {is_valid!r}"
    )


@then(u'the validation result should be invalid')
def step_validation_invalid(context) -> None:
    is_valid, _ = context.validation_result
    assert not is_valid, "Expected invalid portfolio data but validation returned True"


@then(u'the validation error message should not be empty')
def step_validation_error_not_empty(context) -> None:
    is_valid, message = context.validation_result
    assert not is_valid, "Validation result is valid — no error message expected"
    assert message, "Invalid validation result has an empty error message"


# ===========================================================================
# When / Then — calculate_portfolio_returns
# ===========================================================================


@when(u'I calculate portfolio returns from a valid portfolio DataFrame')
def step_calculate_returns(context) -> None:
    _require_imports()
    df = _make_valid_portfolio_df(rows=10)
    context.returns_series = calculate_portfolio_returns(df)


@then(u'the returns series should not be empty')
def step_returns_not_empty(context) -> None:
    _require_imports()
    returns = context.returns_series
    assert isinstance(returns, pl.Series), (
        f"Expected pl.Series from calculate_portfolio_returns, got {type(returns)}"
    )
    assert len(returns) > 0, "Portfolio returns Series is empty"


# ===========================================================================
# When / Then — process_data_for_plot_tab_results
# ===========================================================================


@when(u'I process a small 20-row DataFrame for plotting')
def step_process_small_df(context) -> None:
    _require_imports()
    df = _make_tab_results_df(rows=20)
    context.processed_df = process_data_for_plot_tab_results(
        df, date_column="date", max_points=5000
    )


@when(u'I process a large 200-row DataFrame with max_points {max_pts:d}')
def step_process_large_df(context, max_pts: int) -> None:
    _require_imports()
    rows = 200
    context.large_df_original_rows = rows
    start = dt.date(2024, 1, 2)
    dates = [(start + dt.timedelta(days=i)).isoformat() for i in range(rows)]
    df = pl.DataFrame({
        "date": dates,
        "value_a": [float(i) for i in range(rows)],
    })
    context.processed_df = process_data_for_plot_tab_results(
        df, date_column="date", max_points=int(max_pts)
    )


@then(u'the processed DataFrame should have {count:d} rows')
def step_processed_df_row_count(context, count: int) -> None:
    actual = len(context.processed_df)
    assert actual == count, (
        f"Expected {count} rows in processed DataFrame but got {actual}"
    )


@then(u'the processed DataFrame should have fewer than 200 rows')
def step_processed_df_fewer_rows(context) -> None:
    original = getattr(context, "large_df_original_rows", 200)
    actual = len(context.processed_df)
    assert actual < original, (
        f"Expected fewer than {original} rows after downsampling but got {actual}"
    )


# ===========================================================================
# When / Then — normalize_data_tab_results
# ===========================================================================


@when(u'I normalise a {rows:d}-row DataFrame with method "{method}"')
def step_normalise_df(context, rows: int, method: str) -> None:
    _require_imports()
    df = _make_tab_results_df(rows=int(rows))
    context.normalised_df = normalize_data_tab_results(df, method=method)


@then(u'the normalised DataFrame should have {rows:d} rows')
def step_normalised_df_row_count(context, rows: int) -> None:
    actual = len(context.normalised_df)
    assert actual == rows, (
        f"Expected {rows} rows after normalisation but got {actual}"
    )


@then(u'all numeric column values should be in the range 0.0 to 1.0')
def step_min_max_in_range(context) -> None:
    _require_imports()
    result = context.normalised_df
    numeric_cols = [c for c in result.columns if c != "date"]
    for col in numeric_cols:
        col_min = result[col].min()
        col_max = result[col].max()
        if col_min is not None:
            assert col_min >= -1e-9, (
                f"Column '{col}' min {col_min} is below 0.0 after min_max normalisation"
            )
        if col_max is not None:
            assert col_max <= 1.0 + 1e-9, (
                f"Column '{col}' max {col_max} is above 1.0 after min_max normalisation"
            )


# ===========================================================================
# When / Then — transform_data_tab_results
# ===========================================================================


@when(u'I transform a {rows:d}-row DataFrame with transformation "{transformation}"')
def step_transform_df(context, rows: int, transformation: str) -> None:
    _require_imports()
    df = _make_tab_results_df(rows=int(rows))
    context.transformed_df = transform_data_tab_results(df, transformation=transformation)


@then(u'the transformed DataFrame should have {rows:d} rows')
def step_transformed_df_row_count(context, rows: int) -> None:
    actual = len(context.transformed_df)
    assert actual == rows, (
        f"Expected {rows} rows after transformation but got {actual}"
    )


@then(u'the transformed DataFrame should contain columns with "{suffix}" suffix')
def step_transformed_has_suffix_columns(context, suffix: str) -> None:
    cols = [c for c in context.transformed_df.columns if c.endswith(suffix)]
    assert cols, (
        f"Expected columns ending with '{suffix}' but none found. "
        f"Columns: {context.transformed_df.columns}"
    )


@then(u'the transformed DataFrame should not be empty')
def step_transformed_df_not_empty(context) -> None:
    assert len(context.transformed_df) > 0, "Transformed DataFrame is empty"


@then(u'the cumulative columns should be monotonically non-decreasing')
def step_cumulative_monotonic(context) -> None:
    _require_imports()
    result = context.transformed_df
    cum_cols = [c for c in result.columns if c.endswith("_cum")]
    assert cum_cols, "No _cum columns found in transformed DataFrame"
    for col in cum_cols:
        values = result[col].to_list()
        for idx in range(1, len(values)):
            assert values[idx] >= values[idx - 1], (
                f"Cumulative column '{col}' not non-decreasing at index {idx}: "
                f"{values[idx - 1]} > {values[idx]}"
            )
