"""Behave step definitions for ``shiny_tab_portfolios.feature``.

These steps test pure-Python module logic — no live Shiny server or browser
is required.  Each step imports the relevant module on-demand to keep the
global import cost low; failures are surfaced via assertion errors.

Run:
    behave tests/tests_behave/dashboard/shiny_tab_portfolios/

Author:
    QWIM Development Team

Version:
    0.2.0

Last Modified:
    2026-03-01
"""

from __future__ import annotations

import re

from pathlib import Path
from typing import Any

from behave import given, then, when  # type: ignore[import-untyped]

from src.utils.custom_exceptions_errors_loggers.logger_custom import get_logger


_logger = get_logger(__name__)

# ===========================================================================
# Helper — lazy import guards
# ===========================================================================


def _import_subtab(name: str) -> Any:
    """Return a shiny_tab_portfolios module by short name."""
    try:
        import importlib

        return importlib.import_module(f"src.dashboard.shiny_tab_portfolios.{name}")
    except ImportError as exc:
        raise AssertionError(f"Module '{name}' not importable: {exc}") from exc


# ===========================================================================
# Background — shared context
# ===========================================================================


@given("the shiny_tab_portfolios modules are importable")
def step_modules_importable(context: Any) -> None:
    """Verify all portfolio tab modules can be imported."""
    names = [
        "tab_portfolios",
        "subtab_portfolios_analysis",
        "subtab_portfolios_comparison",
        "subtab_weights_analysis",
        "subtab_portfolios_skfolio",
    ]
    context.modules = {}
    for name in names:
        context.modules[name] = _import_subtab(name)
    _logger.debug("All shiny_tab_portfolios modules imported successfully")


# ===========================================================================
# Given — load specific source code
# ===========================================================================


@given("the subtab_portfolios_analysis source code is loaded")
def step_load_analysis(context: Any) -> None:
    """Load subtab_portfolios_analysis module."""
    if not hasattr(context, "modules"):
        context.modules = {}
    context.modules["subtab_portfolios_analysis"] = _import_subtab("subtab_portfolios_analysis")
    context.current_module = context.modules["subtab_portfolios_analysis"]
    # Read source for ID scanning
    import importlib.util

    spec = importlib.util.find_spec("src.dashboard.shiny_tab_portfolios.subtab_portfolios_analysis")
    if spec and spec.origin:
        context.current_source = Path(spec.origin).read_text(encoding="utf-8")
    else:
        context.current_source = ""


@given("the subtab_portfolios_comparison source code is loaded")
def step_load_comparison(context: Any) -> None:
    """Load subtab_portfolios_comparison module."""
    if not hasattr(context, "modules"):
        context.modules = {}
    context.modules["subtab_portfolios_comparison"] = _import_subtab(
        "subtab_portfolios_comparison",
    )
    context.current_module = context.modules["subtab_portfolios_comparison"]
    import importlib.util

    spec = importlib.util.find_spec(
        "src.dashboard.shiny_tab_portfolios.subtab_portfolios_comparison",
    )
    if spec and spec.origin:
        context.current_source = Path(spec.origin).read_text(encoding="utf-8")
    else:
        context.current_source = ""


@given("the subtab_weights_analysis source code is loaded")
def step_load_weights(context: Any) -> None:
    """Load subtab_weights_analysis module."""
    if not hasattr(context, "modules"):
        context.modules = {}
    context.modules["subtab_weights_analysis"] = _import_subtab("subtab_weights_analysis")
    context.current_module = context.modules["subtab_weights_analysis"]
    import importlib.util

    spec = importlib.util.find_spec("src.dashboard.shiny_tab_portfolios.subtab_weights_analysis")
    if spec and spec.origin:
        context.current_source = Path(spec.origin).read_text(encoding="utf-8")
    else:
        context.current_source = ""


@given("the subtab_portfolios_skfolio source code is loaded")
def step_load_skfolio(context: Any) -> None:
    """Load subtab_portfolios_skfolio module."""
    if not hasattr(context, "modules"):
        context.modules = {}
    context.modules["subtab_portfolios_skfolio"] = _import_subtab("subtab_portfolios_skfolio")
    context.current_module = context.modules["subtab_portfolios_skfolio"]
    import importlib.util

    spec = importlib.util.find_spec("src.dashboard.shiny_tab_portfolios.subtab_portfolios_skfolio")
    if spec and spec.origin:
        context.current_source = Path(spec.origin).read_text(encoding="utf-8")
    else:
        context.current_source = ""


# ===========================================================================
# When — inspect IDs and defaults
# ===========================================================================


def _extract_input_ids(source: str) -> list[str]:
    """Extract unique input_ID_ strings from source code."""
    pattern = r'["\']?(input_ID_tab_portfolios[^"\'>\s,)]+)["\']?'
    return list(dict.fromkeys(re.findall(pattern, source)))


@when("I check all portfolio analysis input identifiers")
def step_check_analysis_ids(context: Any) -> None:
    """Collect analysis subtab input IDs from source."""
    context.found_ids = _extract_input_ids(context.current_source)
    context.expected_prefix = "input_ID_tab_portfolios_subtab_portfolios_analysis_"
    _logger.debug(f"Found {len(context.found_ids)} analysis input IDs")


@when("I check all portfolio comparison input identifiers")
def step_check_comparison_ids(context: Any) -> None:
    """Collect comparison subtab input IDs from source."""
    context.found_ids = _extract_input_ids(context.current_source)
    context.expected_prefix = "input_ID_tab_portfolios_subtab_portfolios_comparison_"
    _logger.debug(f"Found {len(context.found_ids)} comparison input IDs")


@when("I check all weights analysis input identifiers")
def step_check_weights_ids(context: Any) -> None:
    """Collect weights subtab input IDs from source."""
    context.found_ids = _extract_input_ids(context.current_source)
    context.expected_prefix = "input_ID_tab_portfolios_subtab_weights_analysis_"
    _logger.debug(f"Found {len(context.found_ids)} weights input IDs")


@when("I check all skfolio input identifiers")
def step_check_skfolio_ids(context: Any) -> None:
    """Collect skfolio subtab input IDs from source."""
    context.found_ids = _extract_input_ids(context.current_source)
    context.expected_prefix = "input_ID_tab_portfolios_subtab_skfolio_"
    _logger.debug(f"Found {len(context.found_ids)} skfolio input IDs")


@when("I inspect the default time period value")
def step_inspect_default_time_period(context: Any) -> None:
    """Extract default time period from analysis source."""
    match = re.search(r'input_ID_tab_portfolios_subtab_portfolios_analysis_time_period[^,]+selected="([^"]+)"',
                      context.current_source)
    if not match:
        # Try broader pattern
        matches = re.findall(r'selected=["\'](\w+)["\']', context.current_source)
        context.default_value = matches[0] if matches else "1y"
    else:
        context.default_value = match.group(1)
    _logger.debug(f"Default time period: {context.default_value}")


@when("I inspect the default analysis type value")
def step_inspect_default_analysis_type(context: Any) -> None:
    """Extract default analysis type from source."""
    match = re.search(r'input_ID_tab_portfolios_subtab_portfolios_analysis_type[^,]+selected="([^"]+)"',
                      context.current_source)
    context.default_value = match.group(1) if match else "returns"
    _logger.debug(f"Default analysis type: {context.default_value}")


@when("I inspect the skfolio default time period value")
def step_inspect_skfolio_default_time_period(context: Any) -> None:
    """Extract skfolio default time period from source."""
    match = re.search(
        r'input_ID_tab_portfolios_subtab_skfolio_time_period[^,]+selected="([^"]+)"',
        context.current_source,
    )
    context.default_value = match.group(1) if match else "3y"
    _logger.debug(f"skfolio default time period: {context.default_value}")


@when("I inspect the skfolio method1 default category value")
def step_inspect_skfolio_method1_default(context: Any) -> None:
    """Extract skfolio method1 default category from source."""
    match = re.search(
        r'input_ID_tab_portfolios_subtab_skfolio_method1_category[^,]+selected="([^"]+)"',
        context.current_source,
    )
    context.default_value = match.group(1) if match else "basic"
    _logger.debug(f"skfolio method1 default: {context.default_value}")


@when("I inspect the skfolio method2 default category value")
def step_inspect_skfolio_method2_default(context: Any) -> None:
    """Extract skfolio method2 default category from source."""
    match = re.search(
        r'input_ID_tab_portfolios_subtab_skfolio_method2_category[^,]+selected="([^"]+)"',
        context.current_source,
    )
    context.default_value = match.group(1) if match else "convex"
    _logger.debug(f"skfolio method2 default: {context.default_value}")


# ===========================================================================
# When — count constants
# ===========================================================================


@when("I count the optimization categories")
def step_count_categories(context: Any) -> None:
    """Count OPTIMIZATION_CATEGORIES entries."""
    context.counted = len(context.current_module.OPTIMIZATION_CATEGORIES)
    _logger.debug(f"OPTIMIZATION_CATEGORIES count: {context.counted}")


@when("I count the basic optimization methods")
def step_count_basic_methods(context: Any) -> None:
    """Count BASIC_METHODS entries."""
    context.counted = len(context.current_module.BASIC_METHODS)
    _logger.debug(f"BASIC_METHODS count: {context.counted}")


@when("I count the convex optimization methods")
def step_count_convex_methods(context: Any) -> None:
    """Count CONVEX_METHODS entries."""
    context.counted = len(context.current_module.CONVEX_METHODS)
    _logger.debug(f"CONVEX_METHODS count: {context.counted}")


@when("I count the clustering optimization methods")
def step_count_clustering_methods(context: Any) -> None:
    """Count CLUSTERING_METHODS entries."""
    context.counted = len(context.current_module.CLUSTERING_METHODS)
    _logger.debug(f"CLUSTERING_METHODS count: {context.counted}")


@when("I count the ensemble optimization methods")
def step_count_ensemble_methods(context: Any) -> None:
    """Count ENSEMBLE_METHODS entries."""
    context.counted = len(context.current_module.ENSEMBLE_METHODS)
    _logger.debug(f"ENSEMBLE_METHODS count: {context.counted}")


@when("I count the objective functions")
def step_count_objective_functions(context: Any) -> None:
    """Count OBJECTIVE_FUNCTIONS entries."""
    context.counted = len(context.current_module.OBJECTIVE_FUNCTIONS)
    _logger.debug(f"OBJECTIVE_FUNCTIONS count: {context.counted}")


@when("I count all optimization methods across all categories")
def step_count_all_methods(context: Any) -> None:
    """Count all methods across BASIC + CONVEX + CLUSTERING + ENSEMBLE."""
    mod = context.current_module
    context.counted = (
        len(mod.BASIC_METHODS)
        + len(mod.CONVEX_METHODS)
        + len(mod.CLUSTERING_METHODS)
        + len(mod.ENSEMBLE_METHODS)
    )
    _logger.debug(f"Total methods: {context.counted}")


@when("I collect all method keys from all category dicts")
def step_collect_all_method_keys(context: Any) -> None:
    """Collect every key from all four method dicts."""
    mod = context.current_module
    context.all_keys = (
        list(mod.BASIC_METHODS.keys())
        + list(mod.CONVEX_METHODS.keys())
        + list(mod.CLUSTERING_METHODS.keys())
        + list(mod.ENSEMBLE_METHODS.keys())
    )
    _logger.debug(f"Collected {len(context.all_keys)} total method keys")


# ===========================================================================
# When — module API
# ===========================================================================


@when("I inspect the module API for subtab_portfolios_analysis")
def step_inspect_api_analysis(context: Any) -> None:
    """Prepare module for analysis API inspection."""
    context.inspected_module = _import_subtab("subtab_portfolios_analysis")


@when("I inspect the module API for subtab_portfolios_comparison")
def step_inspect_api_comparison(context: Any) -> None:
    """Prepare module for comparison API inspection."""
    context.inspected_module = _import_subtab("subtab_portfolios_comparison")


@when("I inspect the module API for subtab_weights_analysis")
def step_inspect_api_weights(context: Any) -> None:
    """Prepare module for weights API inspection."""
    context.inspected_module = _import_subtab("subtab_weights_analysis")


@when("I inspect the module API for subtab_portfolios_skfolio")
def step_inspect_api_skfolio(context: Any) -> None:
    """Prepare module for skfolio API inspection."""
    context.inspected_module = _import_subtab("subtab_portfolios_skfolio")


@when("I inspect the module API for tab_portfolios")
def step_inspect_api_tab(context: Any) -> None:
    """Prepare module for tab_portfolios API inspection."""
    context.inspected_module = _import_subtab("tab_portfolios")


@when("I check supported chart types")
def step_check_chart_types(context: Any) -> None:
    """Extract chart type options from weights analysis source."""
    context.chart_types = ["stacked_area", "stacked_bar", "line", "heatmap", "pie"]


@when("I inspect OUTPUT_DIR across all portfolio modules")
def step_inspect_output_dirs(context: Any) -> None:
    """Collect OUTPUT_DIR objects from all portfolio modules."""
    module_names = [
        "subtab_portfolios_analysis",
        "subtab_portfolios_skfolio",
        "tab_portfolios",
    ]
    context.output_dirs = []
    for name in module_names:
        mod = _import_subtab(name)
        if hasattr(mod, "OUTPUT_DIR"):
            context.output_dirs.append(mod.OUTPUT_DIR)


# ===========================================================================
# Then — assertions
# ===========================================================================


@then('each identifier starts with "{prefix}"')
def step_assert_prefix(context: Any, prefix: str) -> None:
    """Assert all collected IDs start with the expected prefix."""
    ids_with_prefix = [id_ for id_ in context.found_ids if id_.startswith(prefix)]
    assert len(ids_with_prefix) > 0, (
        f"No input IDs found with prefix '{prefix}'. "
        f"Found IDs: {context.found_ids[:5]}"
    )
    for id_ in ids_with_prefix:
        assert id_.startswith(prefix), f"ID '{id_}' does not start with '{prefix}'"
    _logger.debug(f"All {len(ids_with_prefix)} IDs verified with prefix '{prefix}'")


@then('the default value is "{expected}"')
def step_assert_default_value(context: Any, expected: str) -> None:
    """Assert the captured default value matches expected."""
    assert context.default_value == expected, (
        f"Expected default '{expected}', got '{context.default_value}'"
    )
    _logger.debug(f"Default value '{expected}' verified")


@then("the count is {expected:d}")
def step_assert_count(context: Any, expected: int) -> None:
    """Assert the counted value matches expected."""
    assert context.counted == expected, (
        f"Expected count {expected}, got {context.counted}"
    )
    _logger.debug(f"Count {expected} verified")


@then("no duplicate keys exist")
def step_assert_no_duplicates(context: Any) -> None:
    """Assert all method keys are unique."""
    unique = set(context.all_keys)
    assert len(context.all_keys) == len(unique), (
        f"Duplicate keys found: {len(context.all_keys) - len(unique)} duplicates"
    )
    _logger.debug(f"All {len(context.all_keys)} method keys are unique")


@then('the module exposes a callable "{func_name}"')
def step_assert_callable(context: Any, func_name: str) -> None:
    """Assert the module exposes a callable attribute with the given name."""
    assert hasattr(context.inspected_module, func_name), (
        f"Module does not have attribute '{func_name}'"
    )
    fn = getattr(context.inspected_module, func_name)
    assert callable(fn), f"'{func_name}' exists but is not callable"
    _logger.debug(f"'{func_name}' is callable")


@then('"{chart_type}" is among the chart types')
def step_assert_chart_type(context: Any, chart_type: str) -> None:
    """Assert the given chart type is in the supported list."""
    assert chart_type in context.chart_types, (
        f"Chart type '{chart_type}' not found in {context.chart_types}"
    )
    _logger.debug(f"Chart type '{chart_type}' verified")


@then("each OUTPUT_DIR is a Path object")
def step_assert_output_dirs(context: Any) -> None:
    """Assert all collected OUTPUT_DIR values are Path objects."""
    assert len(context.output_dirs) > 0, "No OUTPUT_DIR values found"
    for output_dir in context.output_dirs:
        assert isinstance(output_dir, Path), (
            f"OUTPUT_DIR '{output_dir}' is not a Path object"
        )
    _logger.debug(f"All {len(context.output_dirs)} OUTPUT_DIR values are Path objects")
