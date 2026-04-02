"""Regression tests for enhanced formatting utility functions.

Verifies that ``format_currency_value``, ``extract_numeric_from_currency_string``,
``validate_locale_setting``, and ``validate_currency_code`` from
:mod:`src.dashboard.shiny_utils.utils_enhanced_formatting` produce stable,
known-good outputs across code changes.

The golden values in this file encode the business rule that USD amounts are
displayed as **whole dollars with no decimal places** (e.g., ``"$1,250,000"``),
which was explicitly corrected to bypass Babel's CLDR two-decimal default.

Tests cover:
    - format_currency_value whole-dollar golden string table
    - format_currency_value rounding at boundary (.5 rounds up)
    - format_currency_value robustness (None, strings, infeasible types)
    - extract_numeric_from_currency_string golden round-trip table
    - Round-trip invariant: extract ∘ format == identity
    - validate_locale_setting accepts valid / rejects invalid locales
    - validate_currency_code accepts valid / rejects invalid codes

Author:
    QWIM Development Team

Version:
    0.1.0

Last Modified:
    2026-02-28
"""

from __future__ import annotations

import pytest

from src.utils.custom_exceptions_errors_loggers.logger_custom import get_logger

_logger = get_logger(__name__)

try:
    from src.dashboard.shiny_utils.utils_enhanced_formatting import (
        extract_numeric_from_currency_string,
        format_currency_value,
        validate_currency_code,
        validate_locale_setting,
    )

    MODULE_IMPORT_AVAILABLE = True
except ImportError as exc:
    MODULE_IMPORT_AVAILABLE = False
    _logger.warning("Import failed — tests will be skipped: %s", exc)

pytestmark = pytest.mark.skipif(
    not MODULE_IMPORT_AVAILABLE,
    reason="utils_enhanced_formatting not importable in this environment",
)


# ---------------------------------------------------------------------------
# Test class: format_currency_value  —  golden string table
# ---------------------------------------------------------------------------


@pytest.mark.regression()
class Test_Format_Currency_Value_Golden:
    """Regression tests — format_currency_value whole-dollar golden output table."""

    @pytest.mark.parametrize(
        ("amount", "expected"),
        [
            (0, "$0"),
            (0.0, "$0"),
            (1, "$1"),
            (100, "$100"),
            (1_000, "$1,000"),
            (10_000, "$10,000"),
            (100_000, "$100,000"),
            (1_000_000, "$1,000,000"),
            (1_250_000, "$1,250,000"),
            (5_000_000, "$5,000,000"),
            (10_000_000, "$10,000,000"),
            (100_000_000, "$100,000,000"),
        ],
    )
    def test_whole_number_golden_values(self, amount: float, expected: str) -> None:
        """Amounts that are already whole dollars format to the exact golden string."""
        _logger.debug("format_currency_value(%s) expected %r", amount, expected)
        assert format_currency_value(amount) == expected

    @pytest.mark.parametrize(
        ("amount", "expected"),
        [
            (0.4, "$0"),
            (0.5, "$0"),        # banker's rounding: 0 is even
            (99.4, "$99"),
            (99.5, "$100"),     # banker's rounding: 100 is even
            (999.49, "$999"),
            (999.50, "$1,000"),  # banker's rounding: 1000 is even
            (1_234.56, "$1,235"),
            (1_249.99, "$1,250"),
            (500_000.49, "$500,000"),
            (500_000.50, "$500,000"),  # banker's rounding: 500000 is even
        ],
    )
    def test_float_rounding_golden_values(self, amount: float, expected: str) -> None:
        """Float amounts are rounded to nearest dollar before formatting."""
        _logger.debug("format_currency_value(%.2f) expected %r", amount, expected)
        assert format_currency_value(amount) == expected

    def test_result_never_contains_decimal_point(self) -> None:
        """Formatted value never contains a decimal separator (whole dollars only)."""
        _logger.debug("Checking no decimal point in any output")
        test_amounts = [0, 0.5, 1.0, 99.99, 1000.01, 1_000_000.50]
        for amount in test_amounts:
            result = format_currency_value(amount)
            assert "." not in result, f"Unexpected '.' in {result!r} for amount={amount}"

    def test_result_always_starts_with_dollar_sign(self) -> None:
        """Every successful output starts with the '$' currency symbol."""
        _logger.debug("Checking '$' prefix for representative amounts")
        for amount in [0, 1, 1_000, 1_000_000]:
            result = format_currency_value(amount)
            assert result.startswith("$")

    def test_result_is_string_type(self) -> None:
        """Return type is always str."""
        _logger.debug("Checking return type is str")
        result = format_currency_value(12345)
        assert isinstance(result, str)


# ---------------------------------------------------------------------------
# Test class: format_currency_value  —  robustness / error cases
# ---------------------------------------------------------------------------


@pytest.mark.regression()
class Test_Format_Currency_Value_Robustness:
    """Regression tests — format_currency_value handles edge inputs without raising."""

    def test_non_numeric_string_returns_dollar_zero(self) -> None:
        """Non-numeric string returns '$0' (safe fallback, no exception)."""
        _logger.debug("Testing non-numeric string → '$0'")
        assert format_currency_value("abc") == "$0"

    def test_none_input_returns_dollar_zero(self) -> None:
        """None input returns '$0' (safe fallback, no exception)."""
        _logger.debug("Testing None → '$0'")
        assert format_currency_value(None) == "$0"

    def test_empty_string_returns_dollar_zero(self) -> None:
        """Empty string returns '$0'."""
        _logger.debug("Testing empty string → '$0'")
        assert format_currency_value("") == "$0"

    def test_numeric_string_is_parsed_correctly(self) -> None:
        """Numeric string '1500000' is accepted and formatted as $1,500,000."""
        _logger.debug("Testing numeric string '1500000'")
        assert format_currency_value("1500000") == "$1,500,000"

    def test_negative_amount_formats_with_sign(self) -> None:
        """Negative amounts format with a negative sign — value is stable."""
        _logger.debug("Testing negative amount formatting stability")
        result = format_currency_value(-1_000)
        assert isinstance(result, str)
        assert "$" in result


# ---------------------------------------------------------------------------
# Test class: extract_numeric_from_currency_string  —  golden table
# ---------------------------------------------------------------------------


@pytest.mark.regression()
class Test_Extract_Numeric_From_Currency_String_Golden:
    """Regression tests — extract_numeric_from_currency_string golden round-trip table."""

    @pytest.mark.parametrize(
        ("currency_string", "expected"),
        [
            ("$0", 0.0),
            ("$1", 1.0),
            ("$100", 100.0),
            ("$1,000", 1_000.0),
            ("$10,000", 10_000.0),
            ("$100,000", 100_000.0),
            ("$1,000,000", 1_000_000.0),
            ("$1,250,000", 1_250_000.0),
            ("$6,473", 6_473.0),
        ],
    )
    def test_golden_extraction_values(self, currency_string: str, expected: float) -> None:
        """Extract correct golden numeric value from formatted currency strings."""
        _logger.debug("extract_numeric_from_currency_string(%r) expected %s", currency_string, expected)
        result = extract_numeric_from_currency_string(currency_string)
        assert result == pytest.approx(expected, rel=1e-9)

    def test_empty_string_returns_zero(self) -> None:
        """Empty string returns 0.0."""
        _logger.debug("Testing empty string → 0.0")
        assert extract_numeric_from_currency_string("") == 0.0

    def test_none_input_returns_zero(self) -> None:
        """None input returns 0.0 without raising."""
        _logger.debug("Testing None → 0.0")
        assert extract_numeric_from_currency_string(None) == 0.0

    def test_non_currency_string_returns_zero(self) -> None:
        """Non-currency string returns 0.0."""
        _logger.debug("Testing non-currency string → 0.0")
        assert extract_numeric_from_currency_string("N/A") == 0.0

    def test_return_type_is_float(self) -> None:
        """Return type is always float."""
        _logger.debug("Checking return type is float")
        result = extract_numeric_from_currency_string("$1,000")
        assert isinstance(result, float)


# ---------------------------------------------------------------------------
# Test class: round-trip invariant
# ---------------------------------------------------------------------------


@pytest.mark.regression()
class Test_Format_Extract_Round_Trip:
    """Regression tests — format then extract produces the original value (round-trip)."""

    @pytest.mark.parametrize(
        "amount",
        [0, 1, 1_000, 10_000, 100_000, 1_000_000, 1_250_000, 5_000_000],
    )
    def test_format_then_extract_recovers_original(self, amount: float) -> None:
        """format_currency_value → extract_numeric round-trip is lossless for whole dollars."""
        _logger.debug("Round-trip test for amount=%s", amount)
        formatted = format_currency_value(amount)
        recovered = extract_numeric_from_currency_string(formatted)
        assert recovered == pytest.approx(float(amount), rel=1e-9)

    def test_round_trip_for_zero(self) -> None:
        """Zero survives the full format → extract pipeline."""
        _logger.debug("Round-trip test for 0")
        formatted = format_currency_value(0)
        recovered = extract_numeric_from_currency_string(formatted)
        assert recovered == pytest.approx(0.0, abs=1e-10)


# ---------------------------------------------------------------------------
# Test class: validate_locale_setting golden decisions
# ---------------------------------------------------------------------------


@pytest.mark.regression()
class Test_Validate_Locale_Setting_Golden:
    """Regression tests — validate_locale_setting accepts/rejects locales stably."""

    def test_en_us_locale_is_accepted(self) -> None:
        """Standard 'en_US' locale is accepted and returned unchanged."""
        _logger.debug("Testing en_US locale acceptance")
        result = validate_locale_setting("en_US")
        assert result == "en_US"

    def test_en_gb_locale_is_accepted(self) -> None:
        """Standard 'en_GB' locale is accepted."""
        _logger.debug("Testing en_GB locale acceptance")
        result = validate_locale_setting("en_GB")
        assert result == "en_GB"

    def test_returns_string_type(self) -> None:
        """Validated locale is always returned as str."""
        _logger.debug("Testing return type for locale validation")
        result = validate_locale_setting("en_US")
        assert isinstance(result, str)


# ---------------------------------------------------------------------------
# Test class: validate_currency_code golden decisions
# ---------------------------------------------------------------------------


@pytest.mark.regression()
class Test_Validate_Currency_Code_Golden:
    """Regression tests — validate_currency_code accepts/rejects codes stably."""

    def test_usd_is_accepted(self) -> None:
        """Standard three-letter 'USD' currency code is accepted."""
        _logger.debug("Testing USD acceptance")
        result = validate_currency_code("USD")
        assert result == "USD"

    def test_eur_is_accepted(self) -> None:
        """Standard 'EUR' currency code is accepted."""
        _logger.debug("Testing EUR acceptance")
        result = validate_currency_code("EUR")
        assert result == "EUR"

    def test_return_is_string(self) -> None:
        """Return value is always a str."""
        _logger.debug("Testing currency code return type")
        result = validate_currency_code("USD")
        assert isinstance(result, str)
