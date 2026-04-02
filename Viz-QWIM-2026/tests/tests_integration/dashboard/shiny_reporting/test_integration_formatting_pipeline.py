"""Integration tests for the currency formatting pipeline.

Verifies that format_currency_value, extract_numeric_from_currency_string,
format_currency_display, and validate_financial_amount work correctly
together as a cohesive currency formatting and parsing pipeline.

Test Categories
---------------
- format_currency_value → extract_numeric_from_currency_string round-trip
- validate_financial_amount → format_currency_value pipeline
- Cross-module consistency: format_currency_display vs format_currency_value
- Edge cases: large amounts, zero, decimal truncation
- Parametrised round-trip accuracy across representative financial amounts

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

MODULE_IMPORT_AVAILABLE: bool = True

try:
    from src.dashboard.shiny_utils.utils_enhanced_formatting import (
        extract_numeric_from_currency_string,
        format_currency_value,
    )
    from src.dashboard.shiny_utils.utils_tab_clients import (
        format_currency_display,
        validate_financial_amount,
    )
except ImportError as exc:
    MODULE_IMPORT_AVAILABLE = False
    _logger.warning("Import failed — tests will be skipped: %s", exc)

pytestmark = pytest.mark.skipif(
    not MODULE_IMPORT_AVAILABLE,
    reason="Formatting pipeline modules not importable in this environment",
)


# ==============================================================================
# format_currency_value basic integration
# ==============================================================================


@pytest.mark.integration()
class Test_Format_Currency_Value_Integration:
    """Integration tests for format_currency_value standalone behaviour."""

    @pytest.mark.parametrize(
        "amount, expected",
        [
            (0, "$0"),
            (1000, "$1,000"),
            (10000, "$10,000"),
            (100000, "$100,000"),
            (1_000_000, "$1,000,000"),
            (5_000_000, "$5,000,000"),
        ],
    )
    def test_format_currency_value_integer_amounts(
        self, amount: int, expected: str
    ) -> None:
        """format_currency_value formats integer amounts correctly."""
        _logger.debug("Testing format_currency_value for amount=%s", amount)

        result = format_currency_value(amount)

        assert result == expected, (
            f"format_currency_value({amount}) must return '{expected}', got '{result}'"
        )

    @pytest.mark.parametrize(
        "amount, expected",
        [
            (100000.0, "$100,000"),
            (99999.9, "$100,000"),
            (1234.0, "$1,234"),
            (500000.4, "$500,000"),
        ],
    )
    def test_format_currency_value_float_amounts(
        self, amount: float, expected: str
    ) -> None:
        """format_currency_value truncates decimal portion and formats correctly."""
        _logger.debug("Testing format_currency_value for float amount=%s", amount)

        result = format_currency_value(amount)

        assert result == expected, (
            f"format_currency_value({amount}) must return '{expected}', got '{result}'"
        )

    def test_format_currency_value_returns_string(self) -> None:
        """format_currency_value always returns a string."""
        _logger.debug("Testing format_currency_value return type")

        result = format_currency_value(42000)

        assert isinstance(result, str), "format_currency_value must return a str"

    def test_format_currency_value_starts_with_dollar_sign(self) -> None:
        """format_currency_value output always starts with '$'."""
        _logger.debug("Testing format_currency_value output starts with $")

        result = format_currency_value(250000)

        assert result.startswith("$"), f"Output must start with '$', got '{result}'"

    def test_format_currency_value_contains_comma_separator_for_large_amounts(self) -> None:
        """format_currency_value inserts comma thousands separators for amounts >= 1000."""
        _logger.debug("Testing format_currency_value comma separator for large amount")

        result = format_currency_value(1_500_000)

        assert "," in result, f"Output for large amount must contain commas: '{result}'"


# ==============================================================================
# extract_numeric_from_currency_string integration
# ==============================================================================


@pytest.mark.integration()
class Test_Extract_Numeric_From_Currency_String_Integration:
    """Integration tests for extract_numeric_from_currency_string."""

    @pytest.mark.parametrize(
        "currency_string, expected_value",
        [
            ("$0", 0.0),
            ("$1,000", 1000.0),
            ("$100,000", 100000.0),
            ("$1,000,000", 1_000_000.0),
        ],
    )
    def test_extract_numeric_from_formatted_strings(
        self, currency_string: str, expected_value: float
    ) -> None:
        """extract_numeric_from_currency_string correctly parses formatted currency strings."""
        _logger.debug("Testing extract_numeric for string='%s'", currency_string)

        result = extract_numeric_from_currency_string(currency_string)

        assert isinstance(result, float), "extract_numeric must return float"
        assert abs(result - expected_value) < 0.01, (
            f"extract_numeric('{currency_string}') must return {expected_value}, got {result}"
        )

    def test_extract_numeric_returns_float(self) -> None:
        """extract_numeric_from_currency_string returns a float."""
        _logger.debug("Testing extract_numeric return type")

        result = extract_numeric_from_currency_string("$250,000")

        assert isinstance(result, float), "extract_numeric must return float"


# ==============================================================================
# format_currency_value → extract_numeric round-trip
# ==============================================================================


@pytest.mark.integration()
class Test_Currency_Round_Trip_Integration:
    """Integration tests for format_currency_value → extract_numeric round-trip."""

    @pytest.mark.parametrize(
        "amount",
        [0, 1000, 10000, 100000, 500000, 1_000_000, 5_000_000],
    )
    def test_format_then_extract_round_trip(self, amount: int) -> None:
        """format_currency_value → extract_numeric_from_currency_string recovers original amount."""
        _logger.debug("Testing round-trip for amount=%s", amount)

        formatted = format_currency_value(amount)
        extracted = extract_numeric_from_currency_string(formatted)

        assert abs(extracted - float(amount)) < 0.5, (
            f"Round-trip for {amount}: formatted='{formatted}', extracted={extracted}; "
            f"must recover original value within $0.50"
        )

    def test_round_trip_preserves_order_of_magnitude(self) -> None:
        """format → extract preserves relative magnitude ordering of amounts."""
        _logger.debug("Testing round-trip preserves ordering")

        amounts = [1000, 50000, 100000, 500000, 1_000_000]
        extracted = [
            extract_numeric_from_currency_string(format_currency_value(a)) for a in amounts
        ]

        for i in range(len(extracted) - 1):
            assert extracted[i] < extracted[i + 1], (
                f"Round-tripped amount {amounts[i]} must be less than {amounts[i + 1]}"
            )


# ==============================================================================
# validate_financial_amount → format_currency_value pipeline
# ==============================================================================


@pytest.mark.integration()
class Test_Validate_Then_Format_Currency_Pipeline_Integration:
    """Integration tests for validate_financial_amount → format_currency_value pipeline."""

    @pytest.mark.parametrize(
        "raw_amount, expected_formatted",
        [
            (100000.0, "$100,000"),
            (500000.0, "$500,000"),
            (0.0, "$0"),
            (1_500_000.0, "$1,500,000"),
        ],
    )
    def test_validate_then_format_pipeline(
        self, raw_amount: float, expected_formatted: str
    ) -> None:
        """validate_financial_amount → format_currency_value produces expected output."""
        _logger.debug("Testing validate → format pipeline for amount=%s", raw_amount)

        validated = validate_financial_amount(raw_amount)
        formatted = format_currency_value(validated)

        assert formatted == expected_formatted, (
            f"Pipeline for {raw_amount}: must produce '{expected_formatted}', got '{formatted}'"
        )

    def test_validated_negative_never_formats(self) -> None:
        """Negative amounts rejected by validate_financial_amount before format stage."""
        _logger.debug("Testing validate rejects negative before format")

        with pytest.raises((ValueError, Exception)):
            validated = validate_financial_amount(-1000.0)
            format_currency_value(validated)  # Must not be reached


# ==============================================================================
# Cross-module currency consistency
# ==============================================================================


@pytest.mark.integration()
class Test_Cross_Module_Currency_Consistency_Integration:
    """Integration tests verifying consistent output across formatting utilities."""

    @pytest.mark.parametrize(
        "amount",
        [0, 1000, 10000, 100000, 250000, 500000, 1_000_000],
    )
    def test_format_currency_display_matches_format_currency_value(
        self, amount: int
    ) -> None:
        """format_currency_display and format_currency_value produce identical output for whole-number amounts."""
        _logger.debug("Testing cross-module currency consistency for amount=%s", amount)

        display_result = format_currency_display(amount)
        enhanced_result = format_currency_value(amount)

        assert display_result == enhanced_result, (
            f"For amount={amount}: format_currency_display='{display_result}' "
            f"must equal format_currency_value='{enhanced_result}'"
        )

    def test_both_functions_use_dollar_sign_prefix(self) -> None:
        """Both currency formatting functions use '$' as the prefix character."""
        _logger.debug("Testing both formatters use $ prefix")

        amount = 75000

        assert format_currency_display(amount).startswith("$"), (
            "format_currency_display must prefix with '$'"
        )
        assert format_currency_value(amount).startswith("$"), (
            "format_currency_value must prefix with '$'"
        )

    def test_both_functions_produce_comma_separated_thousands(self) -> None:
        """Both currency formatting functions use comma thousands separators."""
        _logger.debug("Testing both formatters use comma separators")

        amount = 1_500_000

        assert "," in format_currency_display(amount), (
            "format_currency_display must use comma separators for large amounts"
        )
        assert "," in format_currency_value(amount), (
            "format_currency_value must use comma separators for large amounts"
        )

    def test_full_pipeline_validate_format_extract_round_trip(self) -> None:
        """Full pipeline: validate → format_currency_value → extract recovers original amount."""
        _logger.debug("Testing full validate → format → extract round-trip")

        original_amount = 350000.0

        validated = validate_financial_amount(original_amount)
        formatted = format_currency_value(validated)
        extracted = extract_numeric_from_currency_string(formatted)

        assert abs(extracted - original_amount) < 0.5, (
            f"Full round-trip must recover {original_amount}, got {extracted} "
            f"(formatted as '{formatted}')"
        )
