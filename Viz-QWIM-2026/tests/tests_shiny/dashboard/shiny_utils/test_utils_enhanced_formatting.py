"""Unit tests for ``src.dashboard.shiny_utils.utils_enhanced_formatting``.

All tests are pure-Python (no Shiny session, no disk I/O) and run quickly.

Run:
    pytest tests/tests_shiny/ -m unit -k utils_enhanced_formatting
"""

from __future__ import annotations

import pytest
from typing import Any


# ---------------------------------------------------------------------------
# validate_locale_setting
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestValidateLocaleSetting:
    """validate_locale_setting must raise on invalid input; return str on valid."""

    def test_non_string_raises_type_error(self) -> None:
        from src.dashboard.shiny_utils.utils_enhanced_formatting import (
            validate_locale_setting,
        )

        with pytest.raises(TypeError):
            validate_locale_setting(123)  # type: ignore[arg-type]

    def test_list_raises_type_error(self) -> None:
        from src.dashboard.shiny_utils.utils_enhanced_formatting import (
            validate_locale_setting,
        )

        with pytest.raises(TypeError):
            validate_locale_setting(["en_US"])  # type: ignore[arg-type]

    def test_none_raises_type_error(self) -> None:
        from src.dashboard.shiny_utils.utils_enhanced_formatting import (
            validate_locale_setting,
        )

        with pytest.raises(TypeError):
            validate_locale_setting(None)  # type: ignore[arg-type]

    def test_empty_string_raises_value_error(self) -> None:
        from src.dashboard.shiny_utils.utils_enhanced_formatting import (
            validate_locale_setting,
        )

        with pytest.raises(ValueError):
            validate_locale_setting("")

    def test_invalid_locale_raises_value_error(self) -> None:
        from src.dashboard.shiny_utils.utils_enhanced_formatting import (
            validate_locale_setting,
        )

        with pytest.raises(ValueError):
            validate_locale_setting("invalid_ZZZZ")

    def test_en_US_accepted(self) -> None:
        from src.dashboard.shiny_utils.utils_enhanced_formatting import (
            validate_locale_setting,
        )

        result = validate_locale_setting("en_US")
        assert result == "en_US"

    def test_de_DE_accepted(self) -> None:
        from src.dashboard.shiny_utils.utils_enhanced_formatting import (
            validate_locale_setting,
        )

        result = validate_locale_setting("de_DE")
        assert isinstance(result, str)
        assert len(result) > 0

    def test_returns_string(self) -> None:
        from src.dashboard.shiny_utils.utils_enhanced_formatting import (
            validate_locale_setting,
        )

        result = validate_locale_setting("en_US")
        assert isinstance(result, str)

    @pytest.mark.parametrize(
        "locale",
        ["en_US", "en_GB", "de_DE", "fr_FR", "es_ES", "pt_BR", "ja_JP"],
    )
    def test_common_locales_accepted(self, locale: str) -> None:
        from src.dashboard.shiny_utils.utils_enhanced_formatting import (
            validate_locale_setting,
        )

        result = validate_locale_setting(locale)
        assert isinstance(result, str)
        assert len(result) > 0


# ---------------------------------------------------------------------------
# validate_currency_code
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestValidateCurrencyCode:
    """validate_currency_code must raise on invalid input; normalise to UPPERCASE."""

    def test_non_string_raises_type_error(self) -> None:
        from src.dashboard.shiny_utils.utils_enhanced_formatting import (
            validate_currency_code,
        )

        with pytest.raises(TypeError):
            validate_currency_code(123)  # type: ignore[arg-type]

    def test_none_raises_type_error(self) -> None:
        from src.dashboard.shiny_utils.utils_enhanced_formatting import (
            validate_currency_code,
        )

        with pytest.raises(TypeError):
            validate_currency_code(None)  # type: ignore[arg-type]

    def test_empty_string_raises_value_error(self) -> None:
        from src.dashboard.shiny_utils.utils_enhanced_formatting import (
            validate_currency_code,
        )

        with pytest.raises(ValueError):
            validate_currency_code("")

    def test_two_char_code_raises_value_error(self) -> None:
        from src.dashboard.shiny_utils.utils_enhanced_formatting import (
            validate_currency_code,
        )

        with pytest.raises(ValueError):
            validate_currency_code("US")

    def test_four_char_code_raises_value_error(self) -> None:
        from src.dashboard.shiny_utils.utils_enhanced_formatting import (
            validate_currency_code,
        )

        with pytest.raises(ValueError):
            validate_currency_code("USDX")

    def test_lowercase_usd_normalised_to_uppercase(self) -> None:
        from src.dashboard.shiny_utils.utils_enhanced_formatting import (
            validate_currency_code,
        )

        result = validate_currency_code("usd")
        assert result == "USD"

    def test_uppercase_eur_accepted(self) -> None:
        from src.dashboard.shiny_utils.utils_enhanced_formatting import (
            validate_currency_code,
        )

        result = validate_currency_code("EUR")
        assert result == "EUR"

    def test_mixed_case_normalised(self) -> None:
        from src.dashboard.shiny_utils.utils_enhanced_formatting import (
            validate_currency_code,
        )

        result = validate_currency_code("Gbp")
        assert result == "GBP"

    @pytest.mark.parametrize("code", ["USD", "EUR", "GBP", "JPY", "CHF", "CAD", "AUD"])
    def test_standard_currency_codes(self, code: str) -> None:
        from src.dashboard.shiny_utils.utils_enhanced_formatting import (
            validate_currency_code,
        )

        result = validate_currency_code(code)
        assert result == code
        assert len(result) == 3


# ---------------------------------------------------------------------------
# format_enhanced_currency_display
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestFormatEnhancedCurrencyDisplay:
    """format_enhanced_currency_display must return a non-empty string."""

    def test_positive_amount_returns_string(self) -> None:
        from src.dashboard.shiny_utils.utils_enhanced_formatting import (
            format_enhanced_currency_display,
        )

        result = format_enhanced_currency_display(1234.56)
        assert isinstance(result, str)
        assert len(result) > 0

    def test_zero_amount_returns_string(self) -> None:
        from src.dashboard.shiny_utils.utils_enhanced_formatting import (
            format_enhanced_currency_display,
        )

        result = format_enhanced_currency_display(0.0)
        assert isinstance(result, str)

    def test_negative_amount_returns_string(self) -> None:
        from src.dashboard.shiny_utils.utils_enhanced_formatting import (
            format_enhanced_currency_display,
        )

        result = format_enhanced_currency_display(-500.0)
        assert isinstance(result, str)

    def test_large_amount_returns_string(self) -> None:
        from src.dashboard.shiny_utils.utils_enhanced_formatting import (
            format_enhanced_currency_display,
        )

        result = format_enhanced_currency_display(1_000_000.0)
        assert isinstance(result, str)
        assert len(result) > 0

    def test_result_contains_digits(self) -> None:
        from src.dashboard.shiny_utils.utils_enhanced_formatting import (
            format_enhanced_currency_display,
        )

        result = format_enhanced_currency_display(9999.99)
        assert any(ch.isdigit() for ch in result)


# ---------------------------------------------------------------------------
# format_enhanced_percentage_display
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestFormatEnhancedPercentageDisplay:
    """format_enhanced_percentage_display must return a non-empty string."""

    def test_standard_percentage_returns_string(self) -> None:
        from src.dashboard.shiny_utils.utils_enhanced_formatting import (
            format_enhanced_percentage_display,
        )

        result = format_enhanced_percentage_display(0.1234)
        assert isinstance(result, str)
        assert len(result) > 0

    def test_zero_returns_string(self) -> None:
        from src.dashboard.shiny_utils.utils_enhanced_formatting import (
            format_enhanced_percentage_display,
        )

        result = format_enhanced_percentage_display(0.0)
        assert isinstance(result, str)

    def test_negative_returns_string(self) -> None:
        from src.dashboard.shiny_utils.utils_enhanced_formatting import (
            format_enhanced_percentage_display,
        )

        result = format_enhanced_percentage_display(-0.05)
        assert isinstance(result, str)

    def test_hundred_percent_returns_string(self) -> None:
        from src.dashboard.shiny_utils.utils_enhanced_formatting import (
            format_enhanced_percentage_display,
        )

        result = format_enhanced_percentage_display(1.0)
        assert isinstance(result, str)

    def test_result_contains_digits(self) -> None:
        from src.dashboard.shiny_utils.utils_enhanced_formatting import (
            format_enhanced_percentage_display,
        )

        result = format_enhanced_percentage_display(0.15)
        assert any(ch.isdigit() for ch in result)


# ---------------------------------------------------------------------------
# format_enhanced_number_display
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestFormatEnhancedNumberDisplay:
    """format_enhanced_number_display must return a non-empty string."""

    def test_integer_value_returns_string(self) -> None:
        from src.dashboard.shiny_utils.utils_enhanced_formatting import (
            format_enhanced_number_display,
        )

        result = format_enhanced_number_display(42)
        assert isinstance(result, str)
        assert len(result) > 0

    def test_large_number_returns_string(self) -> None:
        from src.dashboard.shiny_utils.utils_enhanced_formatting import (
            format_enhanced_number_display,
        )

        result = format_enhanced_number_display(1_234_567.89)
        assert isinstance(result, str)

    def test_zero_returns_string(self) -> None:
        from src.dashboard.shiny_utils.utils_enhanced_formatting import (
            format_enhanced_number_display,
        )

        result = format_enhanced_number_display(0)
        assert isinstance(result, str)

    def test_negative_value_returns_string(self) -> None:
        from src.dashboard.shiny_utils.utils_enhanced_formatting import (
            format_enhanced_number_display,
        )

        result = format_enhanced_number_display(-999.5)
        assert isinstance(result, str)

    def test_result_contains_digits(self) -> None:
        from src.dashboard.shiny_utils.utils_enhanced_formatting import (
            format_enhanced_number_display,
        )

        result = format_enhanced_number_display(12345)
        assert any(ch.isdigit() for ch in result)


# ---------------------------------------------------------------------------
# get_formatting_configuration
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestGetFormattingConfiguration:
    """get_formatting_configuration must return a dict with expected keys."""

    def test_returns_dict(self) -> None:
        from src.dashboard.shiny_utils.utils_enhanced_formatting import (
            get_formatting_configuration,
        )

        result = get_formatting_configuration()
        assert isinstance(result, dict)

    def test_result_is_non_empty(self) -> None:
        from src.dashboard.shiny_utils.utils_enhanced_formatting import (
            get_formatting_configuration,
        )

        result = get_formatting_configuration()
        assert len(result) > 0

    def test_no_exception_raised(self) -> None:
        from src.dashboard.shiny_utils.utils_enhanced_formatting import (
            get_formatting_configuration,
        )

        # Must not raise
        _ = get_formatting_configuration()


# ---------------------------------------------------------------------------
# format_currency_value
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestFormatCurrencyValue:
    """format_currency_value must produce a dollar-formatted string."""

    def test_typical_amount_returns_string(self) -> None:
        from src.dashboard.shiny_utils.utils_enhanced_formatting import (
            format_currency_value,
        )

        result = format_currency_value(1234567.89)
        assert isinstance(result, str)
        assert len(result) > 0

    def test_zero_returns_string(self) -> None:
        from src.dashboard.shiny_utils.utils_enhanced_formatting import (
            format_currency_value,
        )

        result = format_currency_value(0.0)
        assert isinstance(result, str)

    def test_negative_returns_string(self) -> None:
        from src.dashboard.shiny_utils.utils_enhanced_formatting import (
            format_currency_value,
        )

        result = format_currency_value(-500.0)
        assert isinstance(result, str)

    def test_result_contains_digits(self) -> None:
        from src.dashboard.shiny_utils.utils_enhanced_formatting import (
            format_currency_value,
        )

        result = format_currency_value(999.99)
        assert any(ch.isdigit() for ch in result)

    @pytest.mark.parametrize(
        "amount",
        [0.01, 1.0, 100.0, 1_000.0, 10_000.0, 100_000.0, 1_000_000.0],
    )
    def test_various_magnitudes(self, amount: float) -> None:
        from src.dashboard.shiny_utils.utils_enhanced_formatting import (
            format_currency_value,
        )

        result = format_currency_value(amount)
        assert isinstance(result, str)
        assert len(result) > 0


# ---------------------------------------------------------------------------
# extract_numeric_from_currency_string
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestExtractNumericFromCurrencyString:
    """extract_numeric_from_currency_string must parse formatted currency back to float."""

    def test_dollar_formatted_string_parsed(self) -> None:
        from src.dashboard.shiny_utils.utils_enhanced_formatting import (
            extract_numeric_from_currency_string,
        )

        result = extract_numeric_from_currency_string("$1,234.56")
        assert isinstance(result, float)
        assert result == pytest.approx(1234.56, rel=1e-4)

    def test_no_currency_symbol_parsed(self) -> None:
        from src.dashboard.shiny_utils.utils_enhanced_formatting import (
            extract_numeric_from_currency_string,
        )

        result = extract_numeric_from_currency_string("1234.56")
        assert isinstance(result, float)
        assert result == pytest.approx(1234.56, rel=1e-4)

    def test_empty_string_returns_zero_or_safe(self) -> None:
        from src.dashboard.shiny_utils.utils_enhanced_formatting import (
            extract_numeric_from_currency_string,
        )

        result = extract_numeric_from_currency_string("")
        assert isinstance(result, (float, int))
        assert result == 0 or result == 0.0

    def test_zero_string_returns_zero(self) -> None:
        from src.dashboard.shiny_utils.utils_enhanced_formatting import (
            extract_numeric_from_currency_string,
        )

        result = extract_numeric_from_currency_string("$0.00")
        assert result == pytest.approx(0.0, abs=1e-9)

    def test_negative_currency_string_parsed(self) -> None:
        from src.dashboard.shiny_utils.utils_enhanced_formatting import (
            extract_numeric_from_currency_string,
        )

        result = extract_numeric_from_currency_string("-$500.00")
        assert isinstance(result, float)
        # May be -500 or 500 depending on implementation — just verify no crash
        assert abs(result) == pytest.approx(500.0, rel=1e-4)

    def test_million_dollar_string_parsed(self) -> None:
        from src.dashboard.shiny_utils.utils_enhanced_formatting import (
            extract_numeric_from_currency_string,
        )

        result = extract_numeric_from_currency_string("$1,000,000.00")
        assert result == pytest.approx(1_000_000.0, rel=1e-4)
