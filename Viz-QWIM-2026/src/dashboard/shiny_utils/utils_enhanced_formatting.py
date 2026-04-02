"""Enhanced Formatting Utilities Module.

This module provides comprehensive formatting functions for financial data,
currencies, percentages, dates, and numbers with locale-aware formatting
capabilities using the Babel library for internationalization support.

The module is designed to enhance the user experience in financial applications
by providing consistent, professional, and culturally-appropriate data formatting
across different locales and regions.

Features:
    - Locale-aware currency formatting with proper symbols and separators
    - Advanced percentage formatting with customizable precision
    - Number formatting with thousands separators and decimal control
    - Fallback mechanisms for robust error handling
    - Support for multiple international locales and currencies
    - Integration with Shiny applications for reactive formatting

Dependencies:
    - babel: For internationalization and locale-aware formatting
    - locale: For system locale information
    - typing: For type hints and improved code documentation

Examples
--------
    Basic currency formatting:

    ```python
    from utils_enhanced_formatting import format_enhanced_currency_display

    # Format USD currency
    formatted_usd = format_enhanced_currency_display(1234567.89)
    print(formatted_usd)  # Output: "$1,234,567.89"

    # Format EUR currency with German locale
    formatted_eur = format_enhanced_currency_display(
        1234567.89, currency_code="EUR", locale_setting="de_DE"
    )
    print(formatted_eur)  # Output: "1.234.567,89 €"
    ```

    Percentage and number formatting:

    ```python
    from utils_enhanced_formatting import (
        format_enhanced_percentage_display,
        format_enhanced_number_display,
    )

    # Format percentage with 2 decimal places
    percentage_value = format_enhanced_percentage_display(0.0525, 2)
    print(percentage_value)  # Output: "5.25%"

    # Format large number with thousands separators
    large_number = format_enhanced_number_display(1234567, 0)
    print(large_number)  # Output: "1,234,567"
    ```

Note:
    This module follows the project's coding standards with snake_case naming,
    comprehensive input validation, and defensive programming practices.
    All functions include proper error handling with meaningful fallback values
    to ensure application stability.

Attributes
----------
    SUPPORTED_LOCALES (list): List of commonly supported locale identifiers
    DEFAULT_CURRENCY_FORMATS (dict): Default formatting patterns for currencies
    FALLBACK_LOCALE (str): Default fallback locale for error scenarios

See Also
--------
    - Babel Documentation: https://babel.pocoo.org/en/latest/
    - Unicode CLDR: http://cldr.unicode.org/
    - ISO 4217 Currency Codes: https://en.wikipedia.org/wiki/ISO_4217

Version:
    0.5.1

Author:
    QWIM Dashboard Development Team

Last Modified:
    June 2025
"""

from __future__ import annotations

import logging

from decimal import Decimal, InvalidOperation
from typing import Any

from babel.core import Locale, UnknownLocaleError
from babel.dates import format_date, format_datetime
from babel.numbers import format_currency, format_decimal, format_percent


#: Module-level logger for the enhanced formatting module
logger = logging.getLogger(__name__)


# Module-level constants for configuration and supported formats
SUPPORTED_LOCALES: list[str] = [
    "en_US",  # United States English
    "en_GB",  # British English
    "de_DE",  # German (Germany)
    "fr_FR",  # French (France)
    "es_ES",  # Spanish (Spain)
    "it_IT",  # Italian (Italy)
    "ja_JP",  # Japanese (Japan)
    "zh_CN",  # Chinese (Simplified, China)
    "pt_BR",  # Portuguese (Brazil)
    "ru_RU",  # Russian (Russia)
    "ar_SA",  # Arabic (Saudi Arabia)
    "hi_IN",  # Hindi (India)
    "ko_KR",  # Korean (South Korea)
    "nl_NL",  # Dutch (Netherlands)
    "sv_SE",  # Swedish (Sweden)
    "no_NO",  # Norwegian (Norway)
    "da_DK",  # Danish (Denmark)
    "fi_FI",  # Finnish (Finland)
    "pl_PL",  # Polish (Poland)
    "tr_TR",  # Turkish (Turkey)
]


DEFAULT_CURRENCY_FORMATS: dict[str, str] = {
    "standard": "¤#,##0.00",  # Standard currency format with symbol
    "accounting": "¤#,##0.00;(¤#,##0.00)",  # Accounting format with parentheses for negatives
    "compact": "¤#,##0",  # Compact format without decimal places
    "detailed": "¤#,##0.00##",  # Detailed format with optional additional decimals
    "international": "¤¤ #,##0.00",  # International format with currency code
}

FALLBACK_LOCALE: str = "en_US"


def validate_locale_setting(locale_setting: str) -> str:
    """Validate and normalize locale setting with fallback mechanism.

    This function ensures that the provided locale setting is valid and supported
    by the Babel library. If the locale is not supported, it falls back to the
    default locale to prevent application errors.

    Args:
        locale_setting (str): The locale identifier to validate (e.g., "en_US", "de_DE")
            Expected format: language_COUNTRY (ISO 639-1 and ISO 3166-1 alpha-2)

    Returns
    -------
        str: A valid locale identifier that is supported by Babel
            Returns the original locale if valid, otherwise returns FALLBACK_LOCALE

    Raises
    ------
        TypeError: If locale_setting is not a string
        ValueError: If locale_setting validation fails

    Examples
    --------
        >>> validate_locale_setting("en_US")
        "en_US"
        >>> validate_locale_setting("invalid_locale")
        "en_US"  # Falls back to default
        >>> validate_locale_setting("de_DE")
        "de_DE"

    Note:
        This function provides warnings when falling back to the default locale
        to help with debugging locale-related issues in production environments.
    """
    # Input type validation
    if not isinstance(locale_setting, str):
        raise TypeError(f"Invalid locale type: {type(locale_setting)}. Expected string.")

    # Empty string validation
    if not locale_setting.strip():
        raise ValueError("Empty locale setting provided. Using fallback locale.")

    try:
        # Attempt to create Locale object to validate the locale
        Locale.parse(locale_setting)

        # Check if the locale is in our supported list for optimal formatting
        if locale_setting in SUPPORTED_LOCALES:
            return locale_setting
        return locale_setting

    except (UnknownLocaleError, ValueError) as exc_error:
        raise ValueError(
            f"Invalid locale '{locale_setting}': {exc_error}. Using fallback: {FALLBACK_LOCALE}",
        ) from exc_error
    except Exception as exc_error:
        raise RuntimeError(
            f"Unexpected error validating locale '{locale_setting}': {exc_error}",
        ) from exc_error


def validate_currency_code(currency_code: str) -> str:
    """Validate and normalize ISO 4217 currency code.

    This function ensures that the provided currency code follows the ISO 4217
    standard for currency codes. It performs basic validation and normalization
    to prevent formatting errors.

    Args:
        currency_code (str): The ISO 4217 currency code to validate (e.g., "USD", "EUR")
            Expected format: 3-letter uppercase alphabetic code

    Returns
    -------
        str: A valid, normalized currency code in uppercase format
            Returns "USD" as fallback if the provided code is invalid

    Raises
    ------
        TypeError: If currency_code is not a string
        ValueError: If currency_code format is invalid

    Examples
    --------
        >>> validate_currency_code("usd")
        "USD"
        >>> validate_currency_code("EUR")
        "EUR"
        >>> validate_currency_code("invalid")
        "USD"  # Falls back to default

    Note:
        This function performs basic validation but does not verify against
        the complete list of valid ISO 4217 codes to maintain performance.
        For comprehensive validation, consider integrating with a currency
        validation library.
    """
    # Input type validation
    if not isinstance(currency_code, str):
        raise TypeError(f"Invalid currency code type: {type(currency_code)}. Expected string.")

    # Normalize to uppercase and strip whitespace
    normalized_code = currency_code.strip().upper()

    # Basic format validation: must be exactly 3 alphabetic characters
    if len(normalized_code) == 3 and normalized_code.isalpha():
        return normalized_code
    raise ValueError(f"Invalid currency code format '{currency_code}'. Using fallback: USD")


def format_enhanced_currency_display(
    amount: int | float | str | Decimal,
    currency_code: str = "USD",
    locale_setting: str = "en_US",
    format_pattern: str = "standard",
    decimal_places: int | None = None,
) -> str:
    """Format currency with enhanced locale-aware display and comprehensive error handling.

    This function provides professional currency formatting with support for multiple
    locales, currencies, and formatting patterns. It includes robust error handling
    and fallback mechanisms to ensure consistent output even with invalid inputs.

    Args:
        amount (Union[int, float, str, Decimal]): The monetary amount to format
            Accepts various numeric types and string representations
            Examples: 1234.56, "1234.56", Decimal("1234.56")
        currency_code (str, optional): ISO 4217 currency code. Defaults to "USD"
            Examples: "USD", "EUR", "GBP", "JPY"
        locale_setting (str, optional): Locale for formatting. Defaults to "en_US"
            Examples: "en_US", "de_DE", "fr_FR", "ja_JP"
        format_pattern (str, optional): Formatting pattern to use. Defaults to "standard"
            Options: "standard", "accounting", "compact", "detailed", "international"
        decimal_places (Optional[int], optional): Override default decimal places
            If None, uses locale default; otherwise forces specific decimal precision

    Returns
    -------
        str: Formatted currency string with proper locale formatting
            Includes currency symbol, thousands separators, and decimal formatting
            Example outputs: "$1,234.56", "1.234,56 €", "¥1,235"

    Raises
    ------
        TypeError: If amount cannot be converted to a numeric type
        ValueError: If amount is not a valid number

    Examples
    --------
        Basic usage with defaults:

        ```python
        # Standard USD formatting
        result = format_enhanced_currency_display(1234567.89)
        # Output: "$1,234,567.89"
        ```

        International formatting:

        ```python
        # German locale with EUR currency
        result = format_enhanced_currency_display(
            1234567.89, currency_code="EUR", locale_setting="de_DE"
        )
        # Output: "1.234.567,89 €"
        ```

        Custom formatting patterns:

        ```python
        # Accounting format with parentheses for negatives
        result = format_enhanced_currency_display(-1234.56, format_pattern="accounting")
        # Output: "($1,234.56)"
        ```

        Decimal precision control:

        ```python
        # Force no decimal places for whole currency units
        result = format_enhanced_currency_display(1234.56, decimal_places=0)
        # Output: "$1,235"
        ```

    Note:
        This function automatically handles locale-specific formatting rules
        including decimal separators, thousands separators, currency symbol
        placement, and negative number representation.

        For performance-critical applications, consider caching formatted
        results when the same values are formatted repeatedly.

    See Also
    --------
        - format_enhanced_percentage_display: For percentage formatting
        - format_enhanced_number_display: For general number formatting
        - validate_currency_code: For currency code validation
        - validate_locale_setting: For locale validation
    """
    try:
        # Input validation and normalization
        try:
            validated_locale = validate_locale_setting(locale_setting)
        except (TypeError, ValueError):
            validated_locale = FALLBACK_LOCALE

        try:
            validated_currency = validate_currency_code(currency_code)
        except (TypeError, ValueError):
            validated_currency = "USD"

        # Convert amount to Decimal for precise financial calculations
        if isinstance(amount, str):
            # Clean string input by removing common currency symbols and separators
            cleaned_amount = (
                amount.replace("$", "").replace(",", "").replace("€", "").replace("£", "").strip()
            )
            try:
                decimal_amount = Decimal(cleaned_amount)
            except InvalidOperation as exc_error:
                raise ValueError(
                    f"Cannot convert '{amount}' to a valid monetary amount",
                ) from exc_error
        elif isinstance(amount, (int, float)):
            decimal_amount = Decimal(str(amount))
        elif isinstance(amount, Decimal):
            decimal_amount = amount
        else:
            raise TypeError(f"Amount must be int, float, str, or Decimal, got {type(amount)}")

        # Convert back to float for Babel formatting (Babel doesn't support Decimal directly)
        float_amount = float(decimal_amount)

        # Get the appropriate format pattern
        if format_pattern in DEFAULT_CURRENCY_FORMATS:
            babel_format = DEFAULT_CURRENCY_FORMATS[format_pattern]
        else:
            babel_format = DEFAULT_CURRENCY_FORMATS["standard"]

        # Override decimal places if specified
        if (
            decimal_places is not None
            and isinstance(decimal_places, int)
            and 0 <= decimal_places <= 10
        ):
            # Modify format pattern to force specific decimal places
            if decimal_places == 0:
                babel_format = babel_format.replace(".00", "")
            else:
                decimal_pattern = "." + "0" * decimal_places
                babel_format = babel_format.replace(".00", decimal_pattern)

        # Perform the currency formatting using Babel
        return format_currency(
            float_amount,
            validated_currency,
            locale=validated_locale,
            format=babel_format,
        )

    except (ValueError, TypeError):
        # Provide a basic fallback format
        try:
            fallback_amount = float(amount) if isinstance(amount, (int, float, str)) else 0.0
            return f"${round(fallback_amount):,}"
        except Exception:
            return "$0"
    except Exception:
        # Handle any unexpected errors with a safe fallback
        return "$0"


def format_enhanced_percentage_display(
    value: int | float | str | Decimal,
    decimal_places: int = 2,
    locale_setting: str = "en_US",
    show_positive_sign: bool = False,
    multiply_by_hundred: bool = True,
) -> str:
    """Format percentage with enhanced display options and comprehensive validation.

    This function provides professional percentage formatting with customizable
    precision, locale support, and flexible input handling. It's designed for
    financial applications where precise percentage display is critical.

    Args:
        value (Union[int, float, str, Decimal]): The percentage value to format
            If multiply_by_hundred is True: expects decimal form (0.05 = 5%)
            If multiply_by_hundred is False: expects percentage form (5 = 5%)
        decimal_places (int, optional): Number of decimal places to display. Defaults to 2
            Range: 0-10 decimal places supported
        locale_setting (str, optional): Locale for formatting. Defaults to "en_US"
            Affects decimal separator and thousands separator placement
        show_positive_sign (bool, optional): Whether to show + for positive values. Defaults to False
            Useful for displaying changes or comparisons
        multiply_by_hundred (bool, optional): Whether to multiply by 100. Defaults to True
            Set to False if input is already in percentage form

    Returns
    -------
        str: Formatted percentage string with proper locale formatting
            Examples: "5.25%", "12,34%", "+3.50%", "0%"

    Raises
    ------
        ValueError: If value cannot be converted to a valid number
        TypeError: If decimal_places is not an integer

    Examples
    --------
        Basic percentage formatting:

        ```python
        # Standard percentage from decimal
        result = format_enhanced_percentage_display(0.0525, 2)
        # Output: "5.25%"

        # Percentage with no decimals
        result = format_enhanced_percentage_display(0.15, 0)
        # Output: "15%"
        ```

        International formatting:

        ```python
        # German locale formatting
        result = format_enhanced_percentage_display(
            0.1234, decimal_places=3, locale_setting="de_DE"
        )
        # Output: "12,340%"
        ```

        Change indicators:

        ```python
        # Show positive sign for gains
        result = format_enhanced_percentage_display(0.035, show_positive_sign=True)
        # Output: "+3.50%"
        ```

        Direct percentage input:

        ```python
        # Input already in percentage form
        result = format_enhanced_percentage_display(15.5, multiply_by_hundred=False)
        # Output: "15.50%"
        ```

    Note:
        This function automatically handles locale-specific formatting rules
        for decimal separators and provides consistent percentage symbol placement.

        For financial applications, consider using multiply_by_hundred=True
        to ensure proper conversion from decimal representations.

    See Also
    --------
        - format_enhanced_currency_display: For currency formatting
        - format_enhanced_number_display: For general number formatting
    """
    try:
        # Input validation
        try:
            validated_locale = validate_locale_setting(locale_setting)
        except (TypeError, ValueError):
            validated_locale = FALLBACK_LOCALE

        # Validate decimal_places parameter
        if not isinstance(decimal_places, int) or not (0 <= decimal_places <= 10):
            decimal_places = 2

        # Convert and validate the input value
        if isinstance(value, str):
            # Clean string input by removing percentage symbols and whitespace
            cleaned_value = value.replace("%", "").strip()
            try:
                decimal_value = Decimal(cleaned_value)
            except InvalidOperation as exc_error:
                raise ValueError(f"Cannot convert '{value}' to a valid percentage") from exc_error
        elif isinstance(value, (int, float)):
            decimal_value = Decimal(str(value))
        elif isinstance(value, Decimal):
            decimal_value = value
        else:
            raise TypeError(f"Value must be int, float, str, or Decimal, got {type(value)}")

        # Apply multiplication by 100 if requested (for decimal to percentage conversion)
        display_value = decimal_value if multiply_by_hundred else decimal_value / 100

        # Convert to float for Babel formatting
        float_value = float(display_value)

        # Create format pattern with specified decimal places
        format_pattern = f"#,##0.{'0' * decimal_places}%"

        # Add positive sign to format if requested
        if show_positive_sign:
            format_pattern = f"+{format_pattern};-{format_pattern}"

        # Perform the percentage formatting using Babel
        return format_percent(
            float_value,
            locale=validated_locale,
            format=format_pattern,
        )

    except (ValueError, TypeError):
        # Provide a fallback
        try:
            fallback_value = float(value) if isinstance(value, (int, float, str)) else 0.0
            if multiply_by_hundred:
                fallback_value = fallback_value * 100
            return f"{fallback_value:.{decimal_places}f}%"
        except Exception:
            return "0.00%"
    except Exception:
        # Handle any unexpected errors
        return "0.00%"


def format_enhanced_number_display(
    number: int | float | str | Decimal,
    decimal_places: int = 0,
    locale_setting: str = "en_US",
    show_thousands_separator: bool = True,
    show_positive_sign: bool = False,
    minimum_digits: int | None = None,
) -> str:
    """Format numbers with enhanced display options and comprehensive locale support.

    This function provides professional number formatting with customizable
    precision, thousands separators, and locale-aware display. It's designed
    for financial and statistical applications requiring consistent number presentation.

    Args:
        number (Union[int, float, str, Decimal]): The number to format
            Accepts various numeric types and string representations
        decimal_places (int, optional): Number of decimal places to display. Defaults to 0
            Range: 0-15 decimal places supported for high precision
        locale_setting (str, optional): Locale for formatting. Defaults to "en_US"
            Affects decimal and thousands separator characters
        show_thousands_separator (bool, optional): Whether to show thousands separators. Defaults to True
            Controls display of comma, period, or space separators based on locale
        show_positive_sign (bool, optional): Whether to show + for positive values. Defaults to False
            Useful for displaying changes, deltas, or scientific notation
        minimum_digits (Optional[int], optional): Minimum number of digits before decimal. Defaults to None
            Pads with leading zeros if necessary for consistent formatting

    Returns
    -------
        str: Formatted number string with proper locale formatting
            Examples: "1,234,567", "1.234.567,89", "1 234 567.0", "+123.45"

    Raises
    ------
        ValueError: If number cannot be converted to a valid numeric value
        TypeError: If decimal_places or minimum_digits are not integers

    Examples
    --------
        Basic number formatting:

        ```python
        # Large number with thousands separators
        result = format_enhanced_number_display(1234567)
        # Output: "1,234,567"

        # Decimal precision control
        result = format_enhanced_number_display(1234.5678, decimal_places=2)
        # Output: "1,234.57"
        ```

        International formatting:

        ```python
        # German locale formatting
        result = format_enhanced_number_display(
            1234567.89, decimal_places=2, locale_setting="de_DE"
        )
        # Output: "1.234.567,89"

        # French locale formatting
        result = format_enhanced_number_display(
            1234567.89, decimal_places=2, locale_setting="fr_FR"
        )
        # Output: "1 234 567,89"
        ```

        Special formatting options:

        ```python
        # Show positive sign for deltas
        result = format_enhanced_number_display(123.45, decimal_places=2, show_positive_sign=True)
        # Output: "+123.45"

        # No thousands separator for compact display
        result = format_enhanced_number_display(1234567, show_thousands_separator=False)
        # Output: "1234567"

        # Minimum digits with zero padding
        result = format_enhanced_number_display(42, minimum_digits=6)
        # Output: "000,042"
        ```

    Note:
        This function automatically handles locale-specific formatting rules
        including decimal separators (. vs ,) and thousands separators (, vs . vs space).

        For financial calculations, consider using higher decimal precision
        to maintain accuracy in intermediate calculations.

    See Also
    --------
        - format_enhanced_currency_display: For currency-specific formatting
        - format_enhanced_percentage_display: For percentage formatting
    """
    try:
        # Input validation
        try:
            validated_locale = validate_locale_setting(locale_setting)
        except (TypeError, ValueError):
            validated_locale = FALLBACK_LOCALE

        # Validate decimal_places parameter
        if not isinstance(decimal_places, int) or not (0 <= decimal_places <= 15):
            decimal_places = 0

        # Validate minimum_digits parameter
        if minimum_digits is not None and (
            not isinstance(minimum_digits, int) or minimum_digits < 1
        ):
            minimum_digits = None

        # Convert and validate the input number
        if isinstance(number, str):
            # Clean string input by removing common formatting
            cleaned_number = number.replace(",", "").strip()
            try:
                decimal_number = Decimal(cleaned_number)
            except InvalidOperation as exc_error:
                raise ValueError(f"Cannot convert '{number}' to a valid number") from exc_error
        elif isinstance(number, (int, float)):
            decimal_number = Decimal(str(number))
        elif isinstance(number, Decimal):
            decimal_number = number
        else:
            raise TypeError(f"Number must be int, float, str, or Decimal, got {type(number)}")

        # Convert to float for Babel formatting
        float_number = float(decimal_number)

        # Build format pattern based on options
        digit_pattern = "0" * minimum_digits if minimum_digits and minimum_digits > 1 else "#"

        # Add thousands separator if requested
        if show_thousands_separator:
            if minimum_digits and minimum_digits > 3:
                # For minimum digits, need to account for grouping
                thousands_pattern = f"{digit_pattern[:-3]},{digit_pattern[-3:]}"
            else:
                thousands_pattern = f"{digit_pattern},##0"
        else:
            thousands_pattern = digit_pattern if minimum_digits else "#0"

        # Add decimal places if specified
        if decimal_places > 0:
            decimal_pattern = "." + "0" * decimal_places
            format_pattern = thousands_pattern + decimal_pattern
        else:
            format_pattern = thousands_pattern

        # Add positive sign formatting if requested
        if show_positive_sign:
            format_pattern = f"+{format_pattern};-{format_pattern}"

        # Perform the number formatting using Babel
        return format_decimal(
            float_number,
            locale=validated_locale,
            format=format_pattern,
        )

    except (ValueError, TypeError):
        # Provide a fallback
        try:
            fallback_number = float(number) if isinstance(number, (int, float, str)) else 0.0
            if show_thousands_separator:
                return f"{fallback_number:,.{decimal_places}f}"
            return f"{fallback_number:.{decimal_places}f}"
        except Exception:
            return "0"
    except Exception:
        # Handle any unexpected errors
        return "0"


def format_enhanced_date_display(
    date_value: Any,
    format_style: str = "medium",
    locale_setting: str = "en_US",
    include_time: bool = False,
) -> str:
    """Format dates with enhanced locale-aware display options.

    This function provides professional date formatting with support for multiple
    locales and various date format styles. It handles different input types
    and provides consistent date presentation across the application.

    Args:
        date_value (Any): The date to format (datetime, date, or ISO string)
            Accepts datetime objects, date objects, or ISO format strings
        format_style (str, optional): Date format style. Defaults to "medium"
            Options: "short", "medium", "long", "full"
        locale_setting (str, optional): Locale for formatting. Defaults to "en_US"
            Affects month names, day names, and date ordering
        include_time (bool, optional): Whether to include time information. Defaults to False
            Only applies when input includes time information

    Returns
    -------
        str: Formatted date string with proper locale formatting
            Examples: "Jan 15, 2024", "15. Januar 2024", "2024年1月15日"

    Examples
    --------
        Basic date formatting:

        ```python
        from datetime import datetime

        # Format current date
        now = datetime.now()
        result = format_enhanced_date_display(now)
        # Output: "Jan 15, 2024" (varies by current date)
        ```

        International date formatting:

        ```python
        # German locale formatting
        result = format_enhanced_date_display(datetime(2024, 1, 15), locale_setting="de_DE")
        # Output: "15. Jan. 2024"
        ```

    Note:
        This function is provided for completeness but date formatting
        is less critical for the current financial dashboard application.
        Consider expanding this function as needed for reporting features.
    """
    try:
        # Basic implementation - can be expanded based on requirements
        from datetime import date, datetime

        try:
            validated_locale = validate_locale_setting(locale_setting)
        except (TypeError, ValueError):
            validated_locale = FALLBACK_LOCALE

        # Convert various input types to datetime
        if isinstance(date_value, str):
            try:
                parsed_date = datetime.fromisoformat(date_value)
            except ValueError:
                return str(date_value)
        elif isinstance(date_value, datetime):
            parsed_date = date_value
        elif isinstance(date_value, date):
            parsed_date = datetime.combine(date_value, datetime.min.time())
        else:
            return str(date_value)

        # Format the date using Babel
        if include_time:
            formatted_result = format_datetime(
                parsed_date,
                format=format_style,
                locale=validated_locale,
            )
        else:
            formatted_result = format_date(
                parsed_date.date(),
                format=format_style,
                locale=validated_locale,
            )

        return formatted_result

    except Exception:
        return str(date_value)


def get_formatting_configuration() -> dict[str, Any]:
    """Get comprehensive formatting configuration for the application.

    This function returns a dictionary containing all formatting settings,
    supported locales, currency codes, and format patterns used throughout
    the application. It's useful for configuration management and testing.

    Returns
    -------
        Dict[str, Any]: Complete formatting configuration including:
            - Supported locales list
            - Default currency formats
            - Fallback settings
            - Format pattern examples

    Examples
    --------
        Retrieve configuration for validation:

        ```python
        config = get_formatting_configuration()
        supported_locales = config["supported_locales"]
        currency_formats = config["currency_formats"]
        ```

    Note:
        This function is primarily for internal use and configuration
        management. External code should use the specific formatting
        functions rather than accessing configuration directly.
    """
    return {
        "supported_locales": SUPPORTED_LOCALES.copy(),
        "currency_formats": DEFAULT_CURRENCY_FORMATS.copy(),
        "fallback_locale": FALLBACK_LOCALE,
        "version": "0.5.1",
        "module_info": {
            "description": "Enhanced formatting utilities for financial applications",
            "dependencies": ["babel", "decimal", "typing"],
            "author": "QWIM Dashboard Development Team",
        },
    }


# Module initialization and configuration
def configure_enhanced_formatting_module(
    custom_locales: list[str] | None = None,
    custom_fallback: str | None = None,
    log_level: str = "INFO",
) -> None:
    """Configure the enhanced formatting module with custom settings.

    This function allows customization of module-level settings including
    supported locales, fallback locale, and logging level. It should be
    called during application initialization if custom configuration is needed.

    Args:
        custom_locales (Optional[List[str]], optional): Custom list of supported locales
            Replaces the default SUPPORTED_LOCALES list
        custom_fallback (Optional[str], optional): Custom fallback locale
            Must be a valid locale identifier
        log_level (str, optional): Logging level for the module. Defaults to "INFO"
            Options: "DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"

    Examples
    --------
        Configure for European markets:

        ```python
        configure_enhanced_formatting_module(
            custom_locales=["en_GB", "de_DE", "fr_FR", "es_ES"],
            custom_fallback="en_GB",
            log_level="DEBUG",
        )
        ```

    Note:
        This function modifies module-level constants and should only
        be called once during application startup to avoid inconsistent behavior.
    """
    global SUPPORTED_LOCALES, FALLBACK_LOCALE  # noqa: PLW0603

    try:
        # Configure logging level
        logger.setLevel(getattr(logging, log_level.upper(), logging.INFO))

        # Update supported locales if provided
        if custom_locales:
            if isinstance(custom_locales, list) and all(
                isinstance(loc, str) for loc in custom_locales
            ):
                SUPPORTED_LOCALES = custom_locales.copy()
            else:
                raise ValueError("Invalid custom_locales format. Must be a list of strings.")

        # Update fallback locale if provided
        if custom_fallback:
            try:
                validate_locale_setting(custom_fallback)
                FALLBACK_LOCALE = custom_fallback
            except (TypeError, ValueError):
                raise ValueError(f"Invalid custom fallback locale: {custom_fallback}")

    except Exception as exc_error:
        raise RuntimeError(
            f"Error configuring enhanced formatting module: {exc_error}",
        ) from exc_error


def format_currency_value(amount: float) -> str:
    """Format numeric value as USD currency string (dollars only, no cents).

    Args:
        amount: Numeric amount to format

    Returns
    -------
        str: Formatted string such as ``"$1,250,000"`` with no decimal places.
    """
    try:
        n = round(float(amount))
        return f"${n:,}"
    except (ValueError, TypeError):
        return "$0"


def extract_numeric_from_currency_string(currency_string: str) -> float:
    """Extract numeric value from formatted currency string.

    Args:
        currency_string: Formatted currency string (e.g., "$6,473")

    Returns
    -------
        float: Extracted numeric value (e.g., 6473.0)
    """
    try:
        if not currency_string or not isinstance(currency_string, str):
            return 0.0

        # Remove all currency symbols and formatting
        cleaned_string = currency_string.replace("$", "").replace(",", "").replace(" ", "").strip()

        if not cleaned_string:
            return 0.0

        # Convert to float
        return float(cleaned_string)

    except (ValueError, AttributeError):
        return 0.0
