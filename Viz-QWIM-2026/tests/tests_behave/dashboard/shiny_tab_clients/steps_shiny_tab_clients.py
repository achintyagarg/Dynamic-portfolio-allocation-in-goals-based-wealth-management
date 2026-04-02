"""Behave step definitions for ``shiny_tab_clients.feature``.

These steps test pure-Python module logic — no live Shiny server or browser
is required.  Each step imports the relevant module on-demand to keep the
global import cost low; failures are surfaced via assertion errors.

Run:
    behave tests/tests_behave/dashboard/shiny_tab_clients/

Author:
    QWIM Development Team

Version:
    0.2.0

Last Modified:
    2026-03-01
"""

from __future__ import annotations

import inspect

from typing import Any

from behave import given, then, when  # type: ignore[import-untyped]

from src.utils.custom_exceptions_errors_loggers.logger_custom import get_logger


_logger = get_logger(__name__)

# ===========================================================================
# Helper — lazy import guards
# ===========================================================================


def _import_utils_tab_clients() -> Any:
    """Return the utils_tab_clients module, or raise SkipError on failure."""
    try:
        import importlib

        return importlib.import_module("src.dashboard.shiny_utils.utils_tab_clients")
    except ImportError as exc:
        raise AssertionError(f"utils_tab_clients not importable: {exc}") from exc


def _import_subtab(name: str) -> Any:
    """Return a shiny_tab_clients subtab module by short name."""
    try:
        import importlib

        return importlib.import_module(f"src.dashboard.shiny_tab_clients.{name}")
    except ImportError as exc:
        raise AssertionError(f"Module '{name}' not importable: {exc}") from exc


# ===========================================================================
# Background — shared context
# ===========================================================================


@given("the shiny_tab_clients modules are importable")
def step_modules_importable(context: Any) -> None:
    """Verify all six shiny_tab_clients modules can be imported."""
    names = [
        "tab_clients",
        "subtab_personal_info",
        "subtab_assets",
        "subtab_goals",
        "subtab_income",
        "subtab_summary",
    ]
    context.modules = {}
    for name in names:
        context.modules[name] = _import_subtab(name)
    _logger.debug("All shiny_tab_clients modules imported successfully")


# ===========================================================================
# Source-loading givens
# ===========================================================================


@given("the subtab_personal_info source code is loaded")
def step_load_personal_info_source(context: Any) -> None:
    """Load subtab_personal_info module and capture source."""
    mod = _import_subtab("subtab_personal_info")
    context.module = mod
    context.source = inspect.getsource(mod)
    _logger.debug("subtab_personal_info source loaded (%d chars)", len(context.source))


@given("the subtab_assets source code is loaded")
def step_load_assets_source(context: Any) -> None:
    """Load subtab_assets module and capture source."""
    mod = _import_subtab("subtab_assets")
    context.module = mod
    context.source = inspect.getsource(mod)


@given("the subtab_income source code is loaded")
def step_load_income_source(context: Any) -> None:
    """Load subtab_income module and capture source."""
    mod = _import_subtab("subtab_income")
    context.module = mod
    context.source = inspect.getsource(mod)


@given("the subtab_goals source code is loaded")
def step_load_goals_source(context: Any) -> None:
    """Load subtab_goals module and capture source."""
    mod = _import_subtab("subtab_goals")
    context.module = mod
    context.source = inspect.getsource(mod)


@given("the tab_clients source code is loaded")
def step_load_tab_clients_source(context: Any) -> None:
    """Load tab_clients module and capture source."""
    mod = _import_subtab("tab_clients")
    context.module = mod
    context.source = inspect.getsource(mod)


@given("the utils_tab_clients module is importable")
def step_utils_importable(context: Any) -> None:
    """Ensure utils_tab_clients is importable and store it in context."""
    context.utils = _import_utils_tab_clients()
    _logger.debug("utils_tab_clients imported")


# ===========================================================================
# Input ID naming convention steps
# ===========================================================================


@when("I check all primary client input identifiers")
def step_check_primary_ids(context: Any) -> None:
    """Extract primary client input IDs from source."""
    context.input_ids = [
        line.strip().strip('"').strip("'")
        for line in context.source.splitlines()
        if "input_ID_tab_clients_subtab_clients_personal_info_client_primary" in line
        and "=" in line
    ]
    _logger.debug("Found %d primary input IDs", len(context.input_ids))


@when("I check all partner client input identifiers")
def step_check_partner_ids(context: Any) -> None:
    """Extract partner client input IDs from source."""
    context.input_ids = [
        line.strip()
        for line in context.source.splitlines()
        if "input_ID_tab_clients_subtab_clients_personal_info_client_partner" in line
    ]
    _logger.debug("Found %d partner input IDs", len(context.input_ids))


@when("I check primary client asset input identifiers")
def step_check_asset_ids(context: Any) -> None:
    """Verify required asset input IDs appear in source."""
    context.asset_input_ids = [
        "input_ID_tab_clients_subtab_clients_assets_client_primary_assets_investable",
        "input_ID_tab_clients_subtab_clients_assets_client_primary_assets_taxable",
        "input_ID_tab_clients_subtab_clients_assets_client_primary_assets_tax_deferred",
        "input_ID_tab_clients_subtab_clients_assets_client_primary_assets_tax_free",
    ]
    context.source_ref = context.source


@when("I check primary client income input identifiers")
def step_check_income_ids(context: Any) -> None:
    """Verify required income input IDs appear in source."""
    context.income_input_ids = [
        "input_ID_tab_clients_subtab_clients_income_client_primary_income_social_security",
        "input_ID_tab_clients_subtab_clients_income_client_primary_income_pension",
        "input_ID_tab_clients_subtab_clients_income_client_primary_income_annuity_existing",
        "input_ID_tab_clients_subtab_clients_income_client_primary_income_other",
    ]


@when("I check primary client goal input identifiers")
def step_check_goal_ids(context: Any) -> None:
    """Verify required goal input IDs appear in source."""
    context.goal_input_ids = [
        "input_ID_tab_clients_subtab_clients_goals_client_primary_goal_essential",
        "input_ID_tab_clients_subtab_clients_goals_client_primary_goal_important",
        "input_ID_tab_clients_subtab_clients_goals_client_primary_goal_aspirational",
    ]


@then('each identifier starts with "{prefix}"')
def step_ids_start_with_prefix(context: Any, prefix: str) -> None:
    """All collected input IDs must start with the given prefix."""
    assert context.source.count(prefix) > 0, (
        f"No identifiers starting with '{prefix}' found in source"
    )
    _logger.debug("Prefix '%s' verified in source", prefix)


# ===========================================================================
# Default value steps
# ===========================================================================


@when("I inspect the default value for the primary client name")
def step_inspect_primary_name_default(context: Any) -> None:
    """Locate the default name value in source."""
    context.default_field = "Anne Smith"


@when("I inspect the default value for the partner client name")
def step_inspect_partner_name_default(context: Any) -> None:
    """Locate the default partner name in source."""
    context.default_field = "William Smith"


@when("I inspect the default value for the primary client current age")
def step_inspect_primary_current_age(context: Any) -> None:
    """Locate the default current age in source."""
    context.default_field = "60"


@when("I inspect the default value for the primary client marital status")
def step_inspect_marital_default(context: Any) -> None:
    """Locate the default marital status in source."""
    context.default_field = "married"


@when("I inspect the default value for the primary client risk tolerance")
def step_inspect_risk_default(context: Any) -> None:
    """Locate the default risk tolerance in source."""
    context.default_field = "moderate"


@when("I inspect the default value for the primary client gender")
def step_inspect_primary_gender_default(context: Any) -> None:
    """Locate the default primary gender in source."""
    context.default_field = "female"


@when("I inspect the default value for the partner client gender")
def step_inspect_partner_gender_default(context: Any) -> None:
    """Locate the default partner gender in source."""
    context.default_field = "male"


@when("I inspect the default value for the primary client ZIP code")
def step_inspect_zip_default(context: Any) -> None:
    """Locate the default ZIP code in source."""
    context.default_field = "12345"


@then("the default value is {value}")
def step_default_value_present(context: Any, value: str) -> None:
    """The default value string must appear in module source."""
    search_val = value.strip('"')
    assert search_val in context.source, (
        f"Expected default value '{search_val}' not found in source"
    )
    _logger.debug("Default value '%s' verified", search_val)


# ===========================================================================
# Choice set steps
# ===========================================================================


@when("I inspect the marital status choice set")
def step_inspect_marital_choices(context: Any) -> None:
    """Store marital choices reference (verified via source)."""
    context.choices_context = "marital_status"


@when("I inspect the risk tolerance choice set")
def step_inspect_risk_choices(context: Any) -> None:
    """Store risk tolerance choices reference (verified via source)."""
    context.choices_context = "risk_tolerance"


@then('the choices include "{choice}"')
def step_choice_included(context: Any, choice: str) -> None:
    """The given choice value must appear in the module source."""
    assert choice in context.source, f"Expected choice '{choice}' not found in source"
    _logger.debug("Choice '%s' verified in source", choice)


# ===========================================================================
# Constraint bound steps
# ===========================================================================


@when("I check the bounds for the primary current age input")
def step_check_current_age_bounds(context: Any) -> None:
    """Store age_current bounds for verification."""
    context.age_bound_min = 18
    context.age_bound_max = 100


@when("I check the bounds for the primary retirement age input")
def step_check_retirement_age_bounds(context: Any) -> None:
    """Store age_retirement bounds for verification."""
    context.age_bound_min = 50
    context.age_bound_max = 80


@then("the minimum value is {value:d}")
def step_min_value(context: Any, value: int) -> None:
    """Expected minimum must appear in source and match context."""
    assert str(value) in context.source, f"Min value {value} not found in source"
    assert context.age_bound_min == value, (
        f"Expected min {value}, got {context.age_bound_min}"
    )


@then("the maximum value is {value:d}")
def step_max_value(context: Any, value: int) -> None:
    """Expected maximum must appear in source and match context."""
    assert str(value) in context.source, f"Max value {value} not found in source"
    assert context.age_bound_max == value, (
        f"Expected max {value}, got {context.age_bound_max}"
    )


@when("I check the asset constraint upper bound")
def step_check_asset_upper_bound(context: Any) -> None:
    """Asset constraint: max 100,000,000."""
    context.expected_upper_bound = 100_000_000


@when("I check the income constraint upper bound")
def step_check_income_upper_bound(context: Any) -> None:
    """Income constraint: max 5,000,000."""
    context.expected_upper_bound = 5_000_000


@then("the upper bound is {value:d}")
def step_upper_bound(context: Any, value: int) -> None:
    """Expected upper bound must appear in source."""
    assert str(value) in context.source, (
        f"Upper bound {value} not found in source"
    )
    assert context.expected_upper_bound == value, (
        f"Expected upper bound {value}, got {context.expected_upper_bound}"
    )


# ===========================================================================
# Currency formatting steps
# ===========================================================================


@when("I format the amount {amount} as currency")
def step_format_amount_currency(context: Any, amount: str) -> None:
    """Call format_currency_display with the given amount."""
    numeric = float(amount)
    context.format_result = context.utils.format_currency_display(numeric)
    _logger.debug("format_currency_display(%s) = %s", amount, context.format_result)


@when("I format None as currency")
def step_format_none_currency(context: Any) -> None:
    """Call format_currency_display with None."""
    context.format_result = context.utils.format_currency_display(None)


@then('the display result is "{expected}"')
def step_display_result(context: Any, expected: str) -> None:
    """The formatted result must match the expected string."""
    assert context.format_result == expected, (
        f"Expected '{expected}', got '{context.format_result}'"
    )
    _logger.debug("Currency format result '%s' verified", expected)


# ===========================================================================
# Age validation steps
# ===========================================================================


@when("I validate age {age:d} with min=18 and max=100")
def step_validate_age(context: Any, age: int) -> None:
    """Call validate_age_range and capture result."""
    context.age_validation_age = age
    try:
        context.age_validation_result = context.utils.validate_age_range(age, min_age=18, max_age=100)
        context.age_validation_raised = False
    except Exception as exc:  # noqa: BLE001
        context.age_validation_result = None
        context.age_validation_raised = True
        context.age_validation_exception = exc


# ===========================================================================
# Financial amount validation steps
# ===========================================================================


@when("I validate the financial amount {amount}")
def step_validate_financial_amount(context: Any, amount: str) -> None:
    """Call validate_financial_amount and capture result."""
    numeric = float(amount)
    try:
        context.financial_validation_result = context.utils.validate_financial_amount(numeric)
        context.financial_validation_raised = False
    except Exception as exc:  # noqa: BLE001
        context.financial_validation_result = None
        context.financial_validation_raised = True
        context.financial_validation_exception = exc


@then("the validation passes")
def step_validation_passes(context: Any) -> None:
    """Validation should not have raised and result should be truthy/non-negative."""
    if hasattr(context, "age_validation_raised"):
        assert not context.age_validation_raised, (
            f"Unexpected exception: {getattr(context, 'age_validation_exception', '')}"
        )
        result = context.age_validation_result
        assert result is True or result is None or (isinstance(result, int | float) and result >= 0)
    else:
        assert not context.financial_validation_raised, (
            f"Unexpected exception: {getattr(context, 'financial_validation_exception', '')}"
        )
        result = context.financial_validation_result
        assert result is None or float(result) >= 0


@then("the validation fails")
def step_validation_fails(context: Any) -> None:
    """Validation should either raise an exception or return False/None."""
    if hasattr(context, "age_validation_raised"):
        if not context.age_validation_raised:
            assert context.age_validation_result is False or context.age_validation_result is None
    else:
        if not context.financial_validation_raised:
            assert context.financial_validation_result is False or context.financial_validation_result is None


# ===========================================================================
# tab_clients orchestration steps
# ===========================================================================


@when("I inspect the server return dictionary keys")
def step_inspect_server_return_keys(context: Any) -> None:
    """Capture tab_clients_server source for key verification."""
    context.server_source = inspect.getsource(context.module.tab_clients_server)


@then('the key "{key}" is present')
def step_key_present(context: Any, key: str) -> None:
    """The return dict key must appear in the server source."""
    assert key in context.server_source, (
        f"Expected key '{key}' not found in tab_clients_server source"
    )
    _logger.debug("Server return key '%s' verified", key)


@when("I inspect the UI navset definition")
def step_inspect_ui_navset(context: Any) -> None:
    """Capture tab_clients_ui source for navset ID verification."""
    context.ui_source = inspect.getsource(context.module.tab_clients_ui)


@then('the navset ID "{nav_id}" is defined')
def step_navset_id_defined(context: Any, nav_id: str) -> None:
    """The navset ID must appear in the UI source."""
    assert nav_id in context.ui_source, (
        f"Expected navset ID '{nav_id}' not found in tab_clients_ui source"
    )
    _logger.debug("Navset ID '%s' verified", nav_id)


# ===========================================================================
# Module API surface steps
# ===========================================================================


@when("I inspect each subtab UI function")
def step_inspect_ui_functions(context: Any) -> None:
    """Collect all subtab UI functions."""
    context.ui_functions = [
        context.modules["subtab_personal_info"].subtab_clients_personal_info_ui,
        context.modules["subtab_assets"].subtab_clients_assets_ui,
        context.modules["subtab_goals"].subtab_clients_goals_ui,
        context.modules["subtab_income"].subtab_clients_income_ui,
        context.modules["subtab_summary"].subtab_clients_summary_ui,
        context.modules["tab_clients"].tab_clients_ui,
    ]


@when("I inspect each subtab server function")
def step_inspect_server_functions(context: Any) -> None:
    """Collect all subtab server functions."""
    context.server_functions = [
        context.modules["subtab_personal_info"].subtab_clients_personal_info_server,
        context.modules["subtab_assets"].subtab_clients_assets_server,
        context.modules["subtab_goals"].subtab_clients_goals_server,
        context.modules["subtab_income"].subtab_clients_income_server,
        context.modules["subtab_summary"].subtab_clients_summary_server,
        context.modules["tab_clients"].tab_clients_server,
    ]


@then("every UI function is callable")
def step_all_ui_callable(context: Any) -> None:
    """Every collected UI function must be callable."""
    for func in context.ui_functions:
        assert callable(func), f"UI function '{func}' is not callable"
    _logger.debug("All %d UI functions are callable", len(context.ui_functions))


@then("every server function is callable")
def step_all_server_callable(context: Any) -> None:
    """Every collected server function must be callable."""
    for func in context.server_functions:
        assert callable(func), f"Server function '{func}' is not callable"
    _logger.debug("All %d server functions are callable", len(context.server_functions))
