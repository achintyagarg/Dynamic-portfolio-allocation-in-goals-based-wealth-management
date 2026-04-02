"""Behave step definitions for ``shiny_utils.feature``.

Tests pure-Python module logic — no live Shiny server required.
All imports are deferred to step functions to minimise startup cost.

Run:
    behave tests/tests_behave/dashboard/shiny_utils/

Author:
    QWIM Development Team

Version:
    0.1.0

Last Modified:
    2026-05-01
"""

from __future__ import annotations

import warnings

from typing import Any

import pandas as pd

from behave import given, then, when  # type: ignore[import-untyped]

from src.utils.custom_exceptions_errors_loggers.logger_custom import get_logger


_logger = get_logger(__name__)


# ===========================================================================
# Background
# ===========================================================================


@given("the shiny_utils modules are importable")
def step_modules_importable(context: Any) -> None:
    """Verify that core shiny_utils modules can be imported."""
    import importlib

    module_names = [
        "src.dashboard.shiny_utils.utils_tab_portfolios",
        "src.dashboard.shiny_utils.utils_tab_clients",
        "src.dashboard.shiny_utils.utils_errors_dashboard",
    ]
    context.modules = {}
    for name in module_names:
        try:
            context.modules[name] = importlib.import_module(name)
        except ImportError as exc:
            raise AssertionError(f"Module '{name}' not importable: {exc}") from exc

    _logger.debug("All shiny_utils modules imported successfully")


# ===========================================================================
# validate_portfolio_data steps
# ===========================================================================


@when("I validate portfolio data with a None value")
def step_validate_none(context: Any) -> None:
    from src.dashboard.shiny_utils.utils_tab_portfolios import validate_portfolio_data

    context.is_valid, context.message = validate_portfolio_data(None)


@when("I validate an empty portfolio DataFrame")
def step_validate_empty(context: Any) -> None:
    from src.dashboard.shiny_utils.utils_tab_portfolios import validate_portfolio_data

    context.is_valid, context.message = validate_portfolio_data(pd.DataFrame())


@when("I validate a portfolio DataFrame without a Date column")
def step_validate_no_date(context: Any) -> None:
    from src.dashboard.shiny_utils.utils_tab_portfolios import validate_portfolio_data

    df = pd.DataFrame({"Value": [100.0, 110.0]})
    context.is_valid, context.message = validate_portfolio_data(df)


@when("I validate portfolio data without a Value column")
def step_validate_no_value_col(context: Any) -> None:
    from src.dashboard.shiny_utils.utils_tab_portfolios import validate_portfolio_data

    df = pd.DataFrame({"Date": ["2024-01-01"], "Price": [100.0]})
    context.is_valid, context.message = validate_portfolio_data(df, dataset_name="portfolio data")


@when("I validate portfolio data with all non-numeric Value entries")
def step_validate_non_numeric(context: Any) -> None:
    from src.dashboard.shiny_utils.utils_tab_portfolios import validate_portfolio_data

    df = pd.DataFrame({"Date": ["2024-01-01", "2024-02-01"], "Value": ["N/A", "missing"]})
    context.is_valid, context.message = validate_portfolio_data(df, dataset_name="portfolio data")


@when("I validate a fully valid portfolio DataFrame")
def step_validate_valid(context: Any) -> None:
    from src.dashboard.shiny_utils.utils_tab_portfolios import validate_portfolio_data

    df = pd.DataFrame({"Date": ["2024-01-01", "2024-02-01"], "Value": [100.0, 110.0]})
    context.is_valid, context.message = validate_portfolio_data(df, dataset_name="portfolio data")


@when("I validate a weights DataFrame with component columns")
def step_validate_weights_valid(context: Any) -> None:
    from src.dashboard.shiny_utils.utils_tab_portfolios import validate_portfolio_data

    df = pd.DataFrame({"Date": ["2024-01-01"], "VTI": [0.6], "AGG": [0.4]})
    context.is_valid, context.message = validate_portfolio_data(df, dataset_name="weights data")


@when("I validate weights data with only the Date column")
def step_validate_weights_no_components(context: Any) -> None:
    from src.dashboard.shiny_utils.utils_tab_portfolios import validate_portfolio_data

    df = pd.DataFrame({"Date": ["2024-01-01", "2024-02-01"]})
    context.is_valid, context.message = validate_portfolio_data(df, dataset_name="weights data")


@then("validation returns False")
def step_result_is_false(context: Any) -> None:
    assert context.is_valid is False, f"Expected False but got True. Message: {context.message}"


@then("validation returns True")
def step_result_is_true(context: Any) -> None:
    assert context.is_valid is True, f"Expected True but got False. Message: {context.message}"


@then('the error message mentions "{keyword}"')
def step_message_contains(context: Any, keyword: str) -> None:
    assert keyword.lower() in context.message.lower(), (
        f"Expected '{keyword}' in message, got: {context.message!r}"
    )


@then("the error message is empty")
def step_message_is_empty(context: Any) -> None:
    assert context.message == "", f"Expected empty message but got: {context.message!r}"


# ===========================================================================
# format_currency_display steps
# ===========================================================================


@when("I format the amount {amount} as currency")
def step_format_currency(context: Any, amount: str) -> None:
    from src.dashboard.shiny_utils.utils_tab_clients import format_currency_display

    raw = None if amount.strip() == "None" else float(amount)
    context.formatted = format_currency_display(raw)


@then('the formatted result is "{expected}"')
def step_formatted_result(context: Any, expected: str) -> None:
    assert context.formatted == expected, (
        f"Expected {expected!r} but got {context.formatted!r}"
    )


# ===========================================================================
# validate_financial_amount steps
# ===========================================================================


@when("I validate the financial amount {amount}")
def step_validate_financial(context: Any, amount: str) -> None:
    from src.dashboard.shiny_utils.utils_tab_clients import validate_financial_amount

    context.exc = None
    context.validated_amount = None
    raw = None if amount.strip() == "None" else float(amount)
    try:
        context.validated_amount = validate_financial_amount(raw)
    except ValueError as exc:
        context.exc = exc


@then("the validated amount is {expected:f}")
def step_validated_amount(context: Any, expected: float) -> None:
    assert context.exc is None, f"Unexpected exception: {context.exc}"
    assert context.validated_amount == expected, (
        f"Expected {expected} but got {context.validated_amount}"
    )


@then('a ValueError is raised mentioning "{keyword}"')
def step_value_error_raised(context: Any, keyword: str) -> None:
    assert context.exc is not None, "Expected ValueError but no exception was raised"
    assert isinstance(context.exc, ValueError), f"Expected ValueError, got {type(context.exc)}"
    assert keyword.lower() in str(context.exc).lower(), (
        f"Expected '{keyword}' in exception message, got: {context.exc!s}"
    )


# ===========================================================================
# validate_age_range steps
# ===========================================================================


@when("I validate the age {age:d} within range {minimum:d} to {maximum:d}")
def step_validate_age(context: Any, age: int, minimum: int, maximum: int) -> None:
    from src.dashboard.shiny_utils.utils_tab_clients import validate_age_range

    context.exc = None
    context.validated_age = None
    try:
        context.validated_age = validate_age_range(
            age,
            minimum_age=minimum,
            maximum_age=maximum,
            age_type_description="age",
        )
    except ValueError as exc:
        context.exc = exc


@then("the validated age is {expected:d}")
def step_validated_age(context: Any, expected: int) -> None:
    assert context.exc is None, f"Unexpected exception: {context.exc}"
    assert context.validated_age == expected, (
        f"Expected {expected} but got {context.validated_age}"
    )


# ===========================================================================
# is_silent_exception steps
# ===========================================================================


@when("I check if an AttributeError is a silent exception")
def step_check_attribute_error(context: Any) -> None:
    from src.dashboard.shiny_utils.utils_errors_dashboard import is_silent_exception

    context.is_silent = is_silent_exception(AttributeError("reactive not ready"))


@when("I check if a ValueError is a silent exception")
def step_check_value_error(context: Any) -> None:
    from src.dashboard.shiny_utils.utils_errors_dashboard import is_silent_exception

    context.is_silent = is_silent_exception(ValueError("bad value"))


@then("the result is True")
def step_result_true(context: Any) -> None:
    assert context.is_silent is True, "Expected True but got False"


@then("the result is False")
def step_result_false(context: Any) -> None:
    assert context.is_silent is False, "Expected False but got True"


# ===========================================================================
# Deprecated exception warning steps
# ===========================================================================


@when("I instantiate Error_Silent_Initialization")
def step_instantiate_silent(context: Any) -> None:
    from src.dashboard.shiny_utils.utils_errors_dashboard import Error_Silent_Initialization

    context.exc_class = Error_Silent_Initialization
    context.warning_class = DeprecationWarning
    with warnings.catch_warnings(record=True) as caught_warnings:
        warnings.simplefilter("always")
        Error_Silent_Initialization("test")
        context.caught_warnings = caught_warnings[:]


@when("I instantiate Error_Dashboard_Initialization")
def step_instantiate_dashboard(context: Any) -> None:
    from src.dashboard.shiny_utils.utils_errors_dashboard import Error_Dashboard_Initialization

    context.exc_class = Error_Dashboard_Initialization
    context.warning_class = DeprecationWarning
    with warnings.catch_warnings(record=True) as caught_warnings:
        warnings.simplefilter("always")
        Error_Dashboard_Initialization("test")
        context.caught_warnings = caught_warnings[:]


@then("a DeprecationWarning is emitted")
def step_deprecation_warning_emitted(context: Any) -> None:
    deprecation_warnings = [
        w for w in context.caught_warnings if issubclass(w.category, DeprecationWarning)
    ]
    assert len(deprecation_warnings) > 0, (
        f"Expected DeprecationWarning for {context.exc_class.__name__} but none was emitted. "
        f"Warnings captured: {context.caught_warnings}"
    )
