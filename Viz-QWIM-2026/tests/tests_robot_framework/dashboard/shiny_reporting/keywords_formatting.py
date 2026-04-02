"""
Robot Framework keyword library for dashboard formatting utilities.
===================================================================

Tests cover:
    utils_enhanced_formatting.py
    - format_currency_value            -- "$1,250,000" dollar-only format
    - extract_numeric_from_currency_string -- round-trip extraction
    - format_enhanced_percentage_display   -- decimal-to-percentage string
    - format_enhanced_number_display       -- locale-aware number formatting

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
    from src.dashboard.shiny_utils.utils_enhanced_formatting import (
        format_currency_value,
        extract_numeric_from_currency_string,
        format_enhanced_percentage_display,
        format_enhanced_number_display,
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


# ===========================================================================
# format_currency_value keywords
# ===========================================================================


def format_currency(amount: float | str) -> str:
    """Call format_currency_value(amount) and return the result string.

    Arguments:
    - amount -- numeric amount (float or numeric string)

    Returns: formatted string such as '$1,250,000'
    """
    _require_imports()
    return format_currency_value(float(amount))


def currency_format_should_start_with_dollar_sign(formatted: str) -> None:
    """Assert that the formatted currency string starts with '$'.

    Arguments:
    - formatted -- return value of format_currency()
    """
    _require_imports()
    if not formatted.startswith("$"):
        raise AssertionError(
            f"Expected currency string to start with '$' but got: '{formatted}'"
        )


def currency_format_should_equal(formatted: str, expected: str) -> None:
    """Assert that formatted equals expected string exactly.

    Arguments:
    - formatted -- return value of format_currency()
    - expected  -- expected string e.g. '$1,250,000'
    """
    _require_imports()
    if formatted != expected:
        raise AssertionError(
            f"Expected currency '{expected}' but got '{formatted}'"
        )


def currency_format_should_not_contain_decimal_point(formatted: str) -> None:
    """Assert that the formatted currency string has no decimal point.

    Arguments:
    - formatted -- return value of format_currency()
    """
    _require_imports()
    if "." in formatted:
        raise AssertionError(
            f"Currency format should not contain decimal point but got: '{formatted}'"
        )


def currency_format_of_zero_should_be_dollar_zero(formatted: str) -> None:
    """Assert that format_currency_value(0) returns '$0'.

    Arguments:
    - formatted -- return value of format_currency(0)
    """
    _require_imports()
    expected = "$0"
    if formatted != expected:
        raise AssertionError(
            f"Expected '{expected}' for zero amount but got '{formatted}'"
        )


# ===========================================================================
# extract_numeric_from_currency_string keywords
# ===========================================================================


def extract_numeric_from_currency(currency_string: str) -> float:
    """Call extract_numeric_from_currency_string and return the float result.

    Arguments:
    - currency_string -- formatted string such as '$1,250,000'

    Returns: float numeric value
    """
    _require_imports()
    return extract_numeric_from_currency_string(currency_string)


def extracted_value_should_equal(extracted: float | str, expected: float | str) -> None:
    """Assert that the extracted numeric value matches expected (within 0.01).

    Arguments:
    - extracted -- return value of extract_numeric_from_currency()
    - expected  -- expected float value
    """
    _require_imports()
    actual = float(extracted)
    exp = float(expected)
    if abs(actual - exp) > 0.01:
        raise AssertionError(
            f"Expected extracted value {exp} but got {actual}"
        )


def currency_round_trip_should_preserve_value(amount: float | str) -> None:
    """Format amount as currency then extract back and check equality.

    The round-trip: amount -> format_currency_value -> extract -> compare.
    Precision is to the nearest dollar (rounding applied in format).

    Arguments:
    - amount -- numeric amount to test
    """
    _require_imports()
    amount_float = float(amount)
    formatted = format_currency_value(amount_float)
    extracted = extract_numeric_from_currency_string(formatted)
    expected = float(round(amount_float))
    if abs(extracted - expected) > 0.01:
        raise AssertionError(
            f"Round-trip mismatch: amount={amount_float}, formatted='{formatted}', "
            f"extracted={extracted}, expected={expected}"
        )


def extract_from_empty_string_should_return_zero() -> None:
    """Assert that extract_numeric_from_currency_string('') returns 0.0."""
    _require_imports()
    result = extract_numeric_from_currency_string("")
    if abs(result) > 1.0e-9:
        raise AssertionError(
            f"Expected 0.0 for empty string but got {result}"
        )


def extract_from_none_alternative_should_return_zero() -> None:
    """Assert that extract_numeric_from_currency_string with non-string returns 0.0.

    Passes None-like value (0) — function guards with isinstance check.
    """
    _require_imports()
    result = extract_numeric_from_currency_string(0)  # type: ignore[arg-type]
    if abs(result) > 1.0e-9:
        raise AssertionError(
            f"Expected 0.0 for non-string input but got {result}"
        )


# ===========================================================================
# format_enhanced_percentage_display keywords
# ===========================================================================


def format_percentage(
    value: float | str,
    decimal_places: int | str = 2,
    multiply_by_hundred: bool = True,
) -> str:
    """Call format_enhanced_percentage_display and return the result string.

    Arguments:
    - value             -- decimal value (e.g. 0.05 for 5%)
    - decimal_places    -- number of decimal places in output (default 2)
    - multiply_by_hundred -- whether to multiply by 100 (default True)

    Returns: formatted percentage string such as '5.25%'
    """
    _require_imports()
    return format_enhanced_percentage_display(
        float(value),
        decimal_places=int(decimal_places),
        multiply_by_hundred=bool(multiply_by_hundred),
    )


def percentage_format_should_end_with_percent_sign(formatted: str) -> None:
    """Assert that the formatted percentage string ends with '%'.

    Arguments:
    - formatted -- return value of format_percentage()
    """
    _require_imports()
    if not formatted.endswith("%"):
        raise AssertionError(
            f"Expected percentage string to end with '%' but got: '{formatted}'"
        )


def percentage_format_should_equal(formatted: str, expected: str) -> None:
    """Assert that formatted equals expected percentage string exactly.

    Arguments:
    - formatted -- return value of format_percentage()
    - expected  -- expected string e.g. '5.25%'
    """
    _require_imports()
    if formatted != expected:
        raise AssertionError(
            f"Expected percentage '{expected}' but got '{formatted}'"
        )


# ===========================================================================
# format_enhanced_number_display keywords
# ===========================================================================


def format_number(
    number: float | int | str,
    decimal_places: int | str = 0,
) -> str:
    """Call format_enhanced_number_display and return the result string.

    Arguments:
    - number         -- numeric value to format
    - decimal_places -- number of decimal places (default 0)

    Returns: formatted number string such as '1,234,567'
    """
    _require_imports()
    return format_enhanced_number_display(
        float(number),
        decimal_places=int(decimal_places),
    )


def number_format_should_equal(formatted: str, expected: str) -> None:
    """Assert that formatted number string equals expected exactly.

    Arguments:
    - formatted -- return value of format_number()
    - expected  -- expected string e.g. '1,234,567'
    """
    _require_imports()
    if formatted != expected:
        raise AssertionError(
            f"Expected number format '{expected}' but got '{formatted}'"
        )


def number_format_should_contain_comma_separator(formatted: str) -> None:
    """Assert that the formatted number string contains at least one comma.

    Arguments:
    - formatted -- return value of format_number() for a value >= 1000
    """
    _require_imports()
    if "," not in formatted:
        raise AssertionError(
            f"Expected thousands separator comma in '{formatted}' but none found"
        )


def number_format_should_not_be_empty(formatted: str) -> None:
    """Assert that the formatted number string is not empty.

    Arguments:
    - formatted -- return value of format_number()
    """
    _require_imports()
    if not formatted:
        raise AssertionError("format_enhanced_number_display returned an empty string")
