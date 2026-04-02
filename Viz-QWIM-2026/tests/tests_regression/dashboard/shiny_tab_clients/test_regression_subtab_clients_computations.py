"""Regression tests for client data computation utilities.

Verifies that ``format_currency_display``, ``validate_financial_amount``,
``validate_age_range``, and ``validate_string_input`` from
:mod:`src.dashboard.shiny_utils.utils_tab_clients` produce stable,
known-good outputs across code changes.

Tests cover:
    - Currency display golden-value formatting (whole-dollar amounts)
    - Financial amount validation golden decision table
    - Age range validation boundary values
    - String input validation edge cases
    - Consistency between format_currency_display and the business rules

Author:
    QWIM Development Team

Version:
    0.1.0

Last Modified:
    2026-02-28
"""

from __future__ import annotations

from typing import Any

import pytest

from src.utils.custom_exceptions_errors_loggers.logger_custom import get_logger

_logger = get_logger(__name__)

try:
    from src.dashboard.shiny_utils.utils_tab_clients import (
        format_currency_display,
        validate_age_range,
        validate_financial_amount,
        validate_string_input,
    )

    MODULE_IMPORT_AVAILABLE = True
except ImportError as exc:
    MODULE_IMPORT_AVAILABLE = False
    _logger.warning("Import failed — tests will be skipped: %s", exc)

pytestmark = pytest.mark.skipif(
    not MODULE_IMPORT_AVAILABLE,
    reason="utils_tab_clients not importable in this environment",
)


# ---------------------------------------------------------------------------
# Test class: format_currency_display  —  golden values
# ---------------------------------------------------------------------------


@pytest.mark.regression()
class Test_Format_Currency_Display_Golden:
    """Regression tests — format_currency_display produces stable string outputs."""

    @pytest.mark.parametrize(
        ("amount", "expected"),
        [
            (0, "$0"),
            (1, "$1"),
            (1_000, "$1,000"),
            (10_000, "$10,000"),
            (100_000, "$100,000"),
            (1_000_000, "$1,000,000"),
            (1_234_567, "$1,234,567"),
            (5_000_000, "$5,000,000"),
        ],
    )
    def test_whole_number_golden_values(self, amount: int, expected: str) -> None:
        """Whole-number amounts format to exact golden strings."""
        _logger.debug("format_currency_display(%s) → %s", amount, expected)
        result = format_currency_display(amount)
        assert result == expected

    @pytest.mark.parametrize(
        ("amount", "expected"),
        [
            (99.4, "$99"),
            (99.5, "$100"),
            (999.99, "$1,000"),
            (1_234.56, "$1,235"),
            (500_000.99, "$500,001"),  # round(500000.99) = 500001
        ],
    )
    def test_float_amounts_round_correctly(self, amount: float, expected: str) -> None:
        """Float amounts are rounded to nearest dollar before formatting."""
        _logger.debug("format_currency_display(%.2f) → %s", amount, expected)
        result = format_currency_display(amount)
        assert result == expected

    def test_none_returns_dollar_zero(self) -> None:
        """None input returns '$0' without raising an exception."""
        _logger.debug("Testing None → $0")
        result = format_currency_display(None)
        assert result == "$0"

    def test_zero_float_returns_dollar_zero(self) -> None:
        """Explicit 0.0 returns '$0'."""
        _logger.debug("Testing 0.0 → $0")
        result = format_currency_display(0.0)
        assert result == "$0"

    def test_string_numeric_is_converted(self) -> None:
        """Numeric string '500000' is accepted and formats correctly."""
        _logger.debug("Testing string numeric '500000'")
        result = format_currency_display("500000")
        assert result == "$500,000"

    def test_result_always_starts_with_dollar(self) -> None:
        """Every non-error return value starts with '$'."""
        _logger.debug("Testing dollar-sign prefix for various amounts")
        for amount in [0, 1, 1000, 1_000_000]:
            result = format_currency_display(amount)
            assert result.startswith("$"), f"Expected '$' prefix for {amount}, got {result!r}"

    def test_result_contains_no_decimal_point(self) -> None:
        """Formatted output never contains a decimal separator (whole dollars only)."""
        _logger.debug("Testing no decimal point in formatted output")
        for amount in [100.0, 999.99, 1_000_000.50]:
            result = format_currency_display(amount)
            assert "." not in result, f"Unexpected decimal in {result!r} for amount {amount}"


# ---------------------------------------------------------------------------
# Test class: validate_financial_amount  —  golden decisions
# ---------------------------------------------------------------------------


@pytest.mark.regression()
class Test_Validate_Financial_Amount_Golden:
    """Regression tests — validate_financial_amount golden decision table."""

    @pytest.mark.parametrize(
        ("amount", "expected"),
        [
            (0, 0.0),
            (1, 1.0),
            (1_000, 1_000.0),
            (1_000_000, 1_000_000.0),
            (0.01, 0.01),
            (999.99, 999.99),
        ],
    )
    def test_valid_positive_amounts_return_float(
        self, amount: float, expected: float
    ) -> None:
        """Valid non-negative amounts are returned as float golden values."""
        _logger.debug("validate_financial_amount(%s) → %s", amount, expected)
        result = validate_financial_amount(amount)
        assert result == pytest.approx(expected, rel=1e-9)
        assert isinstance(result, float)

    def test_none_input_returns_zero_float(self) -> None:
        """None input returns 0.0 (business rule default)."""
        _logger.debug("Testing None → 0.0")
        result = validate_financial_amount(None)
        assert result == 0.0
        assert isinstance(result, float)

    def test_zero_is_valid(self) -> None:
        """Explicit zero is a valid financial amount."""
        _logger.debug("Testing 0 is valid")
        result = validate_financial_amount(0)
        assert result == 0.0

    def test_negative_amount_raises_value_error(self) -> None:
        """Negative value raises ValueError (business rule: no negative amounts)."""
        _logger.debug("Testing -500 raises ValueError")
        with pytest.raises(ValueError, match="negative"):
            validate_financial_amount(-500)

    @pytest.mark.parametrize("negative_amount", [-0.01, -1, -1_000_000])
    def test_any_negative_raises_value_error(self, negative_amount: float) -> None:
        """Any negative amount raises ValueError regardless of magnitude."""
        _logger.debug("Testing negative amount: %s", negative_amount)
        with pytest.raises(ValueError):
            validate_financial_amount(negative_amount)

    def test_numeric_string_is_accepted(self) -> None:
        """String representation of a number is coerced and validated."""
        _logger.debug("Testing string '1000.0'")
        result = validate_financial_amount("1000.0")
        assert result == pytest.approx(1000.0)

    def test_non_numeric_string_raises_value_error(self) -> None:
        """Non-numeric string raises ValueError."""
        _logger.debug("Testing non-numeric string 'abc'")
        with pytest.raises(ValueError):
            validate_financial_amount("abc")


# ---------------------------------------------------------------------------
# Test class: validate_age_range  —  boundary golden values
# ---------------------------------------------------------------------------


@pytest.mark.regression()
class Test_Validate_Age_Range_Golden:
    """Regression tests — validate_age_range boundary and nominal golden values."""

    def test_nominal_age_returns_int(self) -> None:
        """Age of 45 within [18, 100] returns 45 as int."""
        _logger.debug("Testing age 45 in [18, 100]")
        result = validate_age_range(45, 18, 100, "investor age")
        assert result == 45
        assert isinstance(result, int)

    def test_minimum_boundary_age_is_valid(self) -> None:
        """Minimum boundary age (18) is valid and returned as-is."""
        _logger.debug("Testing minimum boundary 18")
        result = validate_age_range(18, 18, 100, "investor age")
        assert result == 18

    def test_maximum_boundary_age_is_valid(self) -> None:
        """Maximum boundary age (100) is valid and returned as-is."""
        _logger.debug("Testing maximum boundary 100")
        result = validate_age_range(100, 18, 100, "investor age")
        assert result == 100

    def test_below_minimum_raises_value_error(self) -> None:
        """Age below minimum raises ValueError with 'between' in message."""
        _logger.debug("Testing age 17 below minimum 18")
        with pytest.raises(ValueError, match="between"):
            validate_age_range(17, 18, 100, "investor age")

    def test_above_maximum_raises_value_error(self) -> None:
        """Age above maximum raises ValueError."""
        _logger.debug("Testing age 101 above maximum 100")
        with pytest.raises(ValueError, match="between"):
            validate_age_range(101, 18, 100, "investor age")

    def test_none_age_raises_value_error(self) -> None:
        """None age raises ValueError mentioning the field description."""
        _logger.debug("Testing None age")
        with pytest.raises(ValueError):
            validate_age_range(None, 18, 100, "investor age")

    def test_string_numeric_age_is_coerced(self) -> None:
        """String age '55' is coerced to int 55."""
        _logger.debug("Testing string age '55'")
        result = validate_age_range("55", 18, 100, "investor age")
        assert result == 55

    def test_retirement_age_range_golden_values(self) -> None:
        """Retirement ages 60 and 70 both valid within [50, 90] range."""
        _logger.debug("Testing retirement age range [50, 90]")
        assert validate_age_range(60, 50, 90, "retirement age") == 60
        assert validate_age_range(70, 50, 90, "retirement age") == 70

    @pytest.mark.parametrize(
        ("age", "minimum", "maximum"),
        [
            (25, 18, 100),
            (35, 18, 100),
            (45, 18, 100),
            (55, 18, 100),
            (65, 18, 100),
        ],
    )
    def test_parametrized_valid_ages(self, age: int, minimum: int, maximum: int) -> None:
        """Representative valid ages return their own value unchanged."""
        _logger.debug("Testing age %d in [%d, %d]", age, minimum, maximum)
        assert validate_age_range(age, minimum, maximum, "age") == age


# ---------------------------------------------------------------------------
# Test class: validate_string_input  —  golden decisions
# ---------------------------------------------------------------------------


@pytest.mark.regression()
class Test_Validate_String_Input_Golden:
    """Regression tests — validate_string_input produces stable pass/fail decisions."""

    def test_valid_name_returns_trimmed_string(self) -> None:
        """Non-empty string is returned as-is (stripped)."""
        _logger.debug("Testing valid name 'Jane'")
        result = validate_string_input("Jane", "first name")
        assert result == "Jane"

    def test_whitespace_only_raises_value_error(self) -> None:
        """Whitespace-only string is treated as empty and raises ValueError."""
        _logger.debug("Testing whitespace-only string")
        with pytest.raises(ValueError):
            validate_string_input("   ", "first name")

    def test_empty_string_raises_value_error(self) -> None:
        """Empty string raises ValueError."""
        _logger.debug("Testing empty string")
        with pytest.raises(ValueError):
            validate_string_input("", "last name")

    def test_none_input_raises_value_error(self) -> None:
        """None input raises ValueError."""
        _logger.debug("Testing None input")
        with pytest.raises(ValueError):
            validate_string_input(None, "first name")

    def test_leading_trailing_spaces_are_stripped(self) -> None:
        """Input with surrounding whitespace is stripped before returning."""
        _logger.debug("Testing leading/trailing whitespace stripping")
        result = validate_string_input("  Jane  ", "first name")
        assert result == "Jane"

    @pytest.mark.parametrize(
        "valid_name",
        ["John", "Mary-Jane", "O'Brien", "García", "Smith Jr.", "QWIM Investor"],
    )
    def test_various_valid_names_pass(self, valid_name: str) -> None:
        """Diverse valid name strings are accepted without modification."""
        _logger.debug("Testing valid name: %r", valid_name)
        result = validate_string_input(valid_name, "name")
        assert result == valid_name


# ---------------------------------------------------------------------------
# Test class: cross-function consistency
# ---------------------------------------------------------------------------


@pytest.mark.regression()
class Test_Client_Utils_Cross_Function_Consistency:
    """Regression tests ensuring consistency between related client utility functions."""

    def test_format_currency_of_validated_amount_is_stable(self) -> None:
        """Pipeline validate_financial_amount → format_currency_display is stable."""
        _logger.debug("Testing validate then format pipeline")
        raw_amount: Any = "750000.0"
        validated = validate_financial_amount(raw_amount)
        formatted = format_currency_display(validated)
        assert formatted == "$750,000"

    def test_zero_amount_pipeline_is_stable(self) -> None:
        """Zero amount validated then formatted gives '$0'."""
        _logger.debug("Testing zero amount pipeline")
        validated = validate_financial_amount(0)
        formatted = format_currency_display(validated)
        assert formatted == "$0"

    def test_large_amount_pipeline_is_stable(self) -> None:
        """$2,000,000 survives the validate → format pipeline unchanged."""
        _logger.debug("Testing large amount pipeline")
        validated = validate_financial_amount(2_000_000)
        formatted = format_currency_display(validated)
        assert formatted == "$2,000,000"
