"""Behave step definitions for dashboard formatting utilities.

Tests cover:
    utils_enhanced_formatting.py
    - format_currency_value            — "$1,250,000" dollar-only format
    - extract_numeric_from_currency_string — round-trip extraction
    - format_enhanced_percentage_display   — decimal-to-percentage string
    - format_enhanced_number_display       — locale-aware number formatting

Author:         QWIM Development Team
Version:        0.1.0
Last Modified:  2026-02-28
"""

from __future__ import annotations

import io
import sys
from pathlib import Path

from behave import given, then, use_step_matcher, when

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
    if not MODULE_IMPORT_AVAILABLE:
        raise RuntimeError(
            f"Formatting source modules could not be imported: {_import_error_message}"
        )


# ===========================================================================
# When — format_currency_value
# ===========================================================================


@when(u'I format the currency value {amount:g}')
def step_format_currency(context, amount: float) -> None:
    _require_imports()
    context.formatted_value = format_currency_value(float(amount))


# ===========================================================================
# Then — currency formatting assertions
# ===========================================================================


@then(u'the result should start with "$"')
def step_result_starts_with_dollar(context) -> None:
    assert context.formatted_value.startswith("$"), (
        f"Expected value to start with '$' but got: '{context.formatted_value}'"
    )


@then(u'the currency result should equal "{expected}"')
def step_currency_result_equals(context, expected: str) -> None:
    assert context.formatted_value == expected, (
        f"Expected currency '{expected}' but got '{context.formatted_value}'"
    )


@then(u'the result should not contain a decimal point')
def step_result_no_decimal(context) -> None:
    assert "." not in context.formatted_value, (
        f"Currency format should not contain '.' but got: '{context.formatted_value}'"
    )


# ===========================================================================
# When — extract_numeric_from_currency_string
# ===========================================================================


use_step_matcher("re")


@when(r'I extract the numeric value from "([^"]*)"')
def step_extract_numeric(context, currency_str: str) -> None:
    """Match any quoted string using regex, including the empty string ''."""
    _require_imports()
    context.extracted_value = extract_numeric_from_currency_string(currency_str)


use_step_matcher("parse")


@when(u'I extract the numeric value from a non-string input')
def step_extract_non_string(context) -> None:
    _require_imports()
    context.extracted_value = extract_numeric_from_currency_string(0)  # type: ignore[arg-type]


# ===========================================================================
# Then — extraction assertions
# ===========================================================================


@then(u'the extracted value should equal {expected:g}')
def step_extracted_equals(context, expected: float) -> None:
    actual = float(context.extracted_value)
    assert abs(actual - float(expected)) <= 0.01, (
        f"Expected extracted value {expected} but got {actual}"
    )


# ===========================================================================
# When / Then — round-trip
# ===========================================================================


@when(u'I perform a currency round-trip for amount {amount:g}')
def step_currency_round_trip(context, amount: float) -> None:
    _require_imports()
    context.round_trip_amount = float(amount)
    formatted = format_currency_value(float(amount))
    context.extracted_value = extract_numeric_from_currency_string(formatted)


@then(u'the round-trip should preserve the value')
def step_round_trip_preserved(context) -> None:
    expected = float(round(context.round_trip_amount))
    actual = float(context.extracted_value)
    assert abs(actual - expected) <= 0.01, (
        f"Round-trip mismatch: original={context.round_trip_amount}, "
        f"extracted={actual}, expected≈{expected}"
    )


# ===========================================================================
# When — format_enhanced_percentage_display
# ===========================================================================


@when(u'I format the percentage value {value:g}')
def step_format_percentage(context, value: float) -> None:
    _require_imports()
    context.formatted_value = format_enhanced_percentage_display(
        float(value),
        decimal_places=2,
        multiply_by_hundred=True,
    )


@when(u'I format percentage value {value:g} without multiplying by hundred')
def step_format_percentage_no_multiply(context, value: float) -> None:
    _require_imports()
    context.formatted_value = format_enhanced_percentage_display(
        float(value),
        decimal_places=2,
        multiply_by_hundred=False,
    )


# ===========================================================================
# Then — percentage assertions
# ===========================================================================


@then(u'the result should end with "%"')
def step_result_ends_with_percent(context) -> None:
    assert context.formatted_value.endswith("%"), (
        f"Expected value to end with '%' but got: '{context.formatted_value}'"
    )


@then(u'the percentage result should contain "{text}"')
def step_percentage_contains(context, text: str) -> None:
    assert text in context.formatted_value, (
        f"Expected '{text}' in percentage result but got: '{context.formatted_value}'"
    )


# ===========================================================================
# When — format_enhanced_number_display
# ===========================================================================


@when(u'I format the number value {value:g}')
def step_format_number(context, value: float) -> None:
    _require_imports()
    context.formatted_value = format_enhanced_number_display(
        float(value),
        decimal_places=0,
    )


@when(u'I format the number value {value:g} with {places:d} decimal places')
def step_format_number_with_places(context, value: float, places: int) -> None:
    _require_imports()
    context.formatted_value = format_enhanced_number_display(
        float(value),
        decimal_places=int(places),
    )


# ===========================================================================
# Then — number formatting assertions
# ===========================================================================


@then(u'the result should contain a comma')
def step_result_contains_comma(context) -> None:
    assert "," in context.formatted_value, (
        f"Expected comma separator in '{context.formatted_value}' but none found"
    )


@then(u'the number result should not be empty')
def step_number_result_not_empty(context) -> None:
    assert context.formatted_value, (
        "format_enhanced_number_display returned an empty string"
    )
