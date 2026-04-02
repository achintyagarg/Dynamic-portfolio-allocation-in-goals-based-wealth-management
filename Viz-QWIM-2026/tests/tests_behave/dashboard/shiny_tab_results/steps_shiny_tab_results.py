"""Behave step definitions for ``shiny_tab_results.feature``.

These steps test pure-Python module logic — no live Shiny server or browser
is required.  Each step imports the relevant module on-demand to keep the
global import cost low; failures are surfaced via assertion errors.

Run:
    behave tests/tests_behave/dashboard/shiny_tab_results/

Author:
    QWIM Development Team

Version:
    0.1.0

Last Modified:
    2026-03-01
"""

from __future__ import annotations

import importlib
import importlib.util
import re
from pathlib import Path
from typing import Any

from behave import given, then, when  # type: ignore[import-untyped]

from src.utils.custom_exceptions_errors_loggers.logger_custom import get_logger

_logger = get_logger(__name__)

# ===========================================================================
# Helpers — lazy import utilities
# ===========================================================================


def _import_results_module(name: str) -> Any:
    """Return a shiny_tab_results submodule by short name."""
    try:
        return importlib.import_module(f"src.dashboard.shiny_tab_results.{name}")
    except ImportError as exc:
        raise AssertionError(f"Module '{name}' not importable: {exc}") from exc


def _read_source(module_full_name: str) -> str:
    """Return the source text of a module given its full dotted name."""
    spec = importlib.util.find_spec(module_full_name)
    if spec and spec.origin:
        return Path(spec.origin).read_text(encoding="utf-8")
    return ""


# ===========================================================================
# Background — shared availability guard
# ===========================================================================


@given("the shiny_tab_results modules are importable")
def step_modules_importable(context: Any) -> None:
    """Verify all Results-tab modules can be imported."""
    names = ["tab_results", "subtab_reporting", "subtab_simulation"]
    context.modules = {}
    for name in names:
        context.modules[name] = _import_results_module(name)
    _logger.debug("shiny_tab_results modules imported successfully")


# ===========================================================================
# Given — source code loading steps
# ===========================================================================


@given("the shiny_tab_results source code is loaded")
def step_load_all_sources(context: Any) -> None:
    """Load source text for all Results-tab modules combined."""
    sources: list[str] = []
    for module_name in (
        "src.dashboard.shiny_tab_results.tab_results",
        "src.dashboard.shiny_tab_results.subtab_reporting",
        "src.dashboard.shiny_tab_results.subtab_simulation",
    ):
        sources.append(_read_source(module_name))
    context.combined_source = "\n".join(sources)
    _logger.debug("shiny_tab_results combined source loaded (%d chars)", len(context.combined_source))


# ===========================================================================
# When — module export inspection
# ===========================================================================


@when("I check the tab_results module exports")
def step_check_tab_results_exports(context: Any) -> None:
    """Collect callable exports from tab_results."""
    mod = context.modules.get("tab_results") or _import_results_module("tab_results")
    context.exports = {
        "tab_results_ui": getattr(mod, "tab_results_ui", None),
        "tab_results_server": getattr(mod, "tab_results_server", None),
    }


@when("I check the subtab_reporting module exports")
def step_check_reporting_exports(context: Any) -> None:
    """Collect callable exports from subtab_reporting."""
    mod = context.modules.get("subtab_reporting") or _import_results_module("subtab_reporting")
    context.exports = {
        "subtab_reporting_ui": getattr(mod, "subtab_reporting_ui", None),
        "subtab_reporting_server": getattr(mod, "subtab_reporting_server", None),
    }


@when("I check the subtab_simulation module exports")
def step_check_simulation_exports(context: Any) -> None:
    """Collect callable exports from subtab_simulation."""
    mod = context.modules.get("subtab_simulation") or _import_results_module("subtab_simulation")
    context.exports = {
        "subtab_simulation_ui": getattr(mod, "subtab_simulation_ui", None),
        "subtab_simulation_server": getattr(mod, "subtab_simulation_server", None),
    }


# ===========================================================================
# When — ID convention checks
# ===========================================================================


@when("I check all Results tab input identifiers")
def step_collect_input_ids(context: Any) -> None:
    """Extract all string literals assigned as Shiny input IDs from source."""
    source = getattr(context, "combined_source", "")
    # Match id="input_ID_tab_results_..." patterns
    context.input_ids = re.findall(r'id\s*=\s*"(input_ID_[^"]+)"', source)
    _logger.debug("Found %d input IDs", len(context.input_ids))


@when("I check all Results tab output identifiers")
def step_collect_output_ids(context: Any) -> None:
    """Extract all Shiny render function names used as output IDs."""
    source = getattr(context, "combined_source", "")
    # Match def output_ID_tab_results_... render functions
    context.output_ids = re.findall(r"def\s+(output_ID_[^\s(]+)", source)
    _logger.debug("Found %d output IDs", len(context.output_ids))


# ===========================================================================
# When — security constant inspection
# ===========================================================================


@when("I inspect the subtab_reporting security constants")
def step_inspect_security_constants(context: Any) -> None:
    """Load security constants from subtab_reporting."""
    mod = context.modules.get("subtab_reporting") or _import_results_module("subtab_reporting")
    context.MAX_FILENAME_LENGTH = getattr(mod, "MAX_FILENAME_LENGTH", None)
    context.FORBIDDEN_FILENAME_PARTS = getattr(mod, "FORBIDDEN_FILENAME_PARTS", None)
    context.ALLOWED_FILENAME_PATTERN = getattr(mod, "ALLOWED_FILENAME_PATTERN", None)


# ===========================================================================
# When — sanitize_filename_for_security
# ===========================================================================


@when('I sanitize the filename "{filename}"')
def step_sanitize_filename(context: Any, filename: str) -> None:
    """Call sanitize_filename_for_security with the given filename."""
    mod = context.modules.get("subtab_reporting") or _import_results_module("subtab_reporting")
    sanitize_fn = getattr(mod, "sanitize_filename_for_security")
    context.sanitize_result = sanitize_fn(filename)


@when("I sanitize a filename that is 300 characters long")
def step_sanitize_long_filename(context: Any) -> None:
    """Call sanitize_filename_for_security with a 300-char filename."""
    mod = context.modules.get("subtab_reporting") or _import_results_module("subtab_reporting")
    sanitize_fn = getattr(mod, "sanitize_filename_for_security")
    long_name = "x" * 295 + ".pdf"
    context.sanitize_result = sanitize_fn(long_name)


# ===========================================================================
# When — simulation constants inspection
# ===========================================================================


@when("I inspect the subtab_simulation constants")
def step_inspect_simulation_constants(context: Any) -> None:
    """Load constants from subtab_simulation."""
    mod = context.modules.get("subtab_simulation") or _import_results_module("subtab_simulation")
    context.ALL_ETF_SYMBOLS = getattr(mod, "ALL_ETF_SYMBOLS", None)
    context.DEFAULT_SELECTED_ETFS = getattr(mod, "DEFAULT_SELECTED_ETFS", None)
    context.DISTRIBUTION_CHOICES = getattr(mod, "DISTRIBUTION_CHOICES", None)
    context.RNG_TYPE_CHOICES = getattr(mod, "RNG_TYPE_CHOICES", None)


# ===========================================================================
# Then — callable assertions
# ===========================================================================


@then('"{export_name}" should be a callable')
def step_export_is_callable(context: Any, export_name: str) -> None:
    """Assert that the named export is callable."""
    exports = getattr(context, "exports", {})
    obj = exports.get(export_name)
    assert obj is not None, f"Export '{export_name}' is None or missing"
    assert callable(obj), f"'{export_name}' is not callable: {type(obj)}"


# ===========================================================================
# Then — input / output ID naming assertions
# ===========================================================================


@then('each input identifier starts with "input_ID_tab_results_"')
def step_input_ids_have_correct_prefix(context: Any) -> None:
    """Assert all found input IDs begin with the correct hierarchical prefix."""
    ids = getattr(context, "input_ids", [])
    for id_str in ids:
        assert id_str.startswith("input_ID_tab_results_"), (
            f"Input ID '{id_str}' does not start with 'input_ID_tab_results_'"
        )


@then('each output identifier starts with "output_ID_tab_results_"')
def step_output_ids_have_correct_prefix(context: Any) -> None:
    """Assert all found output IDs begin with the correct hierarchical prefix."""
    ids = getattr(context, "output_ids", [])
    for id_str in ids:
        assert id_str.startswith("output_ID_tab_results_"), (
            f"Output ID '{id_str}' does not start with 'output_ID_tab_results_'"
        )


# ===========================================================================
# Then — security constant assertions
# ===========================================================================


@then("MAX_FILENAME_LENGTH should be a positive integer")
def step_max_filename_length_positive(context: Any) -> None:
    """Assert MAX_FILENAME_LENGTH is a positive integer."""
    val = context.MAX_FILENAME_LENGTH
    assert isinstance(val, int), f"Expected int, got {type(val)}"
    assert val > 0, f"MAX_FILENAME_LENGTH should be positive, got {val}"


@then('FORBIDDEN_FILENAME_PARTS should contain ".."')
def step_forbidden_contains_dotdot(context: Any) -> None:
    """Assert '..' is in FORBIDDEN_FILENAME_PARTS."""
    parts = context.FORBIDDEN_FILENAME_PARTS
    assert ".." in parts, "'..'' not found in FORBIDDEN_FILENAME_PARTS"


@then('FORBIDDEN_FILENAME_PARTS should contain "\\\\"')
def step_forbidden_contains_backslash(context: Any) -> None:
    r"""Assert '\\' is in FORBIDDEN_FILENAME_PARTS."""
    parts = context.FORBIDDEN_FILENAME_PARTS
    assert "\\" in parts, r"'\\' not found in FORBIDDEN_FILENAME_PARTS"


@then("ALLOWED_FILENAME_PATTERN should be a compiled regex")
def step_allowed_pattern_is_regex(context: Any) -> None:
    """Assert ALLOWED_FILENAME_PATTERN is a compiled re.Pattern."""
    pat = context.ALLOWED_FILENAME_PATTERN
    assert isinstance(pat, re.Pattern), f"Expected re.Pattern, got {type(pat)}"


# ===========================================================================
# Then — sanitize_filename_for_security assertions
# ===========================================================================


@then('the sanitization result should indicate "valid"')
def step_sanitize_is_valid(context: Any) -> None:
    """Assert first element of sanitize result is True (accepted)."""
    is_valid, _, _ = context.sanitize_result
    assert is_valid is True, f"Expected valid=True, message: {context.sanitize_result[2]}"


@then('the sanitization result should indicate "invalid"')
def step_sanitize_is_invalid(context: Any) -> None:
    """Assert first element of sanitize result is False (rejected)."""
    is_valid, _, message = context.sanitize_result
    assert is_valid is False, f"Expected valid=False but got True, message: {message}"


@then("the sanitized filename should not be empty")
def step_sanitized_not_empty(context: Any) -> None:
    """Assert the sanitized filename in the result tuple is non-empty."""
    _, sanitized, _ = context.sanitize_result
    assert len(sanitized) > 0, "Expected non-empty sanitized filename"


@then("the sanitized filename should be empty")
def step_sanitized_is_empty(context: Any) -> None:
    """Assert the sanitized filename in the result tuple is empty (rejected)."""
    _, sanitized, _ = context.sanitize_result
    assert sanitized == "", f"Expected empty sanitized filename, got {sanitized!r}"


@then("the sanitization return should be a tuple with 3 elements")
def step_sanitize_returns_tuple(context: Any) -> None:
    """Assert sanitize_filename_for_security returns a 3-tuple."""
    result = context.sanitize_result
    assert isinstance(result, tuple), f"Expected tuple, got {type(result)}"
    assert len(result) == 3, f"Expected 3-element tuple, got {len(result)}"


# ===========================================================================
# Then — simulation constant assertions
# ===========================================================================


@then("ALL_ETF_SYMBOLS should have 12 entries")
def step_all_etf_12(context: Any) -> None:
    """Assert ALL_ETF_SYMBOLS has exactly 12 entries."""
    symbols = context.ALL_ETF_SYMBOLS
    assert symbols is not None, "ALL_ETF_SYMBOLS is None"
    assert len(symbols) == 12, f"Expected 12 ETF symbols, got {len(symbols)}"


@then("every DEFAULT_SELECTED_ETFS entry should be in ALL_ETF_SYMBOLS")
def step_defaults_in_all(context: Any) -> None:
    """Assert every default ETF appears in ALL_ETF_SYMBOLS."""
    defaults = context.DEFAULT_SELECTED_ETFS
    all_symbols = context.ALL_ETF_SYMBOLS
    for sym in defaults:
        assert sym in all_symbols, f"Default ETF '{sym}' not in ALL_ETF_SYMBOLS"


@then('DISTRIBUTION_CHOICES should include keys "normal", "lognormal", and "student_t"')
def step_distribution_choices_keys(context: Any) -> None:
    """Assert DISTRIBUTION_CHOICES contains the three required keys."""
    choices = context.DISTRIBUTION_CHOICES
    for key in ("normal", "lognormal", "student_t"):
        assert key in choices, f"Key '{key}' missing from DISTRIBUTION_CHOICES"


@then('RNG_TYPE_CHOICES should include keys "pcg64" and "mt19937"')
def step_rng_type_choices_keys(context: Any) -> None:
    """Assert RNG_TYPE_CHOICES contains pcg64 and mt19937."""
    choices = context.RNG_TYPE_CHOICES
    for key in ("pcg64", "mt19937"):
        assert key in choices, f"Key '{key}' missing from RNG_TYPE_CHOICES"
