"""
Robot Framework keyword library for utils_tab_portfolios, utils_tab_clients,
and utils_errors_dashboard.
===========================================================================

Keywords cover:
    utils_tab_portfolios.validate_portfolio_data
    utils_tab_clients.format_currency_display
    utils_tab_clients.validate_financial_amount
    utils_tab_clients.validate_age_range
    utils_errors_dashboard.is_silent_exception

Author:         QWIM Development Team
Version:        0.1.0
Last Modified:  2026-05-01
"""

from __future__ import annotations

import io
import sys
import warnings

from pathlib import Path


# ---------------------------------------------------------------------------
# Project root on sys.path
# ---------------------------------------------------------------------------
_PROJECT_ROOT: Path = Path(__file__).resolve().parents[4]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# Patch sys.stderr (Robot Framework may strip .buffer)
if not hasattr(sys.stderr, "buffer"):
    sys.stderr = io.TextIOWrapper(io.BytesIO(), encoding="utf-8")

# ---------------------------------------------------------------------------
# Conditional imports
# ---------------------------------------------------------------------------
MODULE_IMPORT_AVAILABLE: bool = True
_import_error_message: str = ""

try:
    import pandas as pd

    from src.dashboard.shiny_utils.utils_errors_dashboard import (
        Error_Dashboard_Initialization,
        Error_Silent_Initialization,
        is_silent_exception,
    )
    from src.dashboard.shiny_utils.utils_tab_clients import (
        format_currency_display,
        validate_age_range,
        validate_financial_amount,
    )
    from src.dashboard.shiny_utils.utils_tab_portfolios import validate_portfolio_data
except ImportError as _exc:
    MODULE_IMPORT_AVAILABLE = False
    _import_error_message = str(_exc)


# ===========================================================================
# validate_portfolio_data keywords
# ===========================================================================


def validate_portfolio_data_with_none() -> tuple:
    """Call validate_portfolio_data(None) and return (is_valid, message)."""
    _assert_modules_available()
    return validate_portfolio_data(None)


def validate_portfolio_data_with_empty_df() -> tuple:
    """Call validate_portfolio_data with an empty DataFrame."""
    _assert_modules_available()
    return validate_portfolio_data(pd.DataFrame())


def validate_portfolio_data_without_date_column() -> tuple:
    """Call validate_portfolio_data with a DataFrame that has no Date column."""
    _assert_modules_available()
    df = pd.DataFrame({"Value": [100.0, 110.0]})
    return validate_portfolio_data(df)


def validate_portfolio_data_without_value_column() -> tuple:
    """Call validate_portfolio_data for portfolio data without Value column."""
    _assert_modules_available()
    df = pd.DataFrame({"Date": ["2024-01-01"], "Price": [100.0]})
    return validate_portfolio_data(df, dataset_name="portfolio data")


def validate_portfolio_data_with_all_non_numeric_values() -> tuple:
    """Call validate_portfolio_data with all-non-numeric Value column."""
    _assert_modules_available()
    df = pd.DataFrame({"Date": ["2024-01-01", "2024-02-01"], "Value": ["N/A", "missing"]})
    return validate_portfolio_data(df, dataset_name="portfolio data")


def validate_portfolio_data_with_valid_data() -> tuple:
    """Call validate_portfolio_data with a fully valid DataFrame."""
    _assert_modules_available()
    df = pd.DataFrame({"Date": ["2024-01-01", "2024-02-01"], "Value": [100.0, 110.0]})
    return validate_portfolio_data(df, dataset_name="portfolio data")


def validate_weights_data_with_components() -> tuple:
    """Call validate_portfolio_data with valid weights data."""
    _assert_modules_available()
    df = pd.DataFrame({"Date": ["2024-01-01"], "VTI": [0.6], "AGG": [0.4]})
    return validate_portfolio_data(df, dataset_name="weights data")


def validate_weights_data_without_components() -> tuple:
    """Call validate_portfolio_data for weights data with only Date column."""
    _assert_modules_available()
    df = pd.DataFrame({"Date": ["2024-01-01", "2024-02-01"]})
    return validate_portfolio_data(df, dataset_name="weights data")


def result_is_valid(result: tuple) -> bool:
    """Extract and return the is_valid boolean from a (bool, str) validation tuple."""
    return result[0]


def result_message(result: tuple) -> str:
    """Extract and return the error message string from a (bool, str) validation tuple."""
    return result[1]


# ===========================================================================
# format_currency_display keywords
# ===========================================================================


def format_zero_as_currency() -> str | None:
    """Return format_currency_display(0)."""
    _assert_modules_available()
    return format_currency_display(0)


def format_one_million_as_currency() -> str | None:
    """Return format_currency_display(1_000_000)."""
    _assert_modules_available()
    return format_currency_display(1_000_000)


def format_none_as_currency() -> str | None:
    """Return format_currency_display(None)."""
    _assert_modules_available()
    return format_currency_display(None)


# ===========================================================================
# validate_financial_amount keywords
# ===========================================================================


def validate_zero_financial_amount() -> float:
    """Return validate_financial_amount(0)."""
    _assert_modules_available()
    return validate_financial_amount(0)


def validate_positive_financial_amount(amount: float = 250_000.0) -> float:
    """Return validate_financial_amount(amount)."""
    _assert_modules_available()
    return validate_financial_amount(amount)


def validate_none_financial_amount() -> float:
    """Return validate_financial_amount(None)."""
    _assert_modules_available()
    return validate_financial_amount(None)


def validate_negative_financial_amount_raises() -> str:
    """Call validate_financial_amount(-100) and return the exception message."""
    _assert_modules_available()
    try:
        validate_financial_amount(-100.0)
        return ""
    except ValueError as exc:
        return str(exc)


# ===========================================================================
# validate_age_range keywords
# ===========================================================================


def validate_valid_age(age: int = 45) -> int:
    """Return validate_age_range with age=45, min=18, max=100."""
    _assert_modules_available()
    return validate_age_range(age, minimum_age=18, maximum_age=100, age_type_description="age")


def validate_below_minimum_age_raises() -> str:
    """Call validate_age_range with age=17 and return exception message."""
    _assert_modules_available()
    try:
        validate_age_range(17, minimum_age=18, maximum_age=100, age_type_description="age")
        return ""
    except ValueError as exc:
        return str(exc)


# ===========================================================================
# is_silent_exception keywords
# ===========================================================================


def check_attribute_error_is_silent() -> bool:
    """Return is_silent_exception(AttributeError(...))."""
    _assert_modules_available()
    return is_silent_exception(AttributeError("reactive not ready"))


def check_value_error_is_not_silent() -> bool:
    """Return is_silent_exception(ValueError(...))."""
    _assert_modules_available()
    return is_silent_exception(ValueError("bad value"))


def instantiate_error_silent_initialization_emits_deprecation() -> bool:
    """Return True if instantiating Error_Silent_Initialization emits DeprecationWarning."""
    _assert_modules_available()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        Error_Silent_Initialization("test")
    return any(issubclass(w.category, DeprecationWarning) for w in caught)


def instantiate_error_dashboard_initialization_emits_deprecation() -> bool:
    """Return True if instantiating Error_Dashboard_Initialization emits DeprecationWarning."""
    _assert_modules_available()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        Error_Dashboard_Initialization("test")
    return any(issubclass(w.category, DeprecationWarning) for w in caught)


# ===========================================================================
# Internal helpers
# ===========================================================================


def _assert_modules_available() -> None:
    """Raise AssertionError if required modules could not be imported."""
    if not MODULE_IMPORT_AVAILABLE:
        raise AssertionError(
            f"Required modules not importable: {_import_error_message}",
        )
