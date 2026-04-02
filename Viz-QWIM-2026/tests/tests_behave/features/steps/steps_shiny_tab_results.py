"""Behave step definitions for ``shiny_tab_results.feature``.

Tests pure-Python module logic — no live Shiny server or browser required.

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
import io
import re
import sys
from pathlib import Path
from typing import Any

from behave import given, then, when  # type: ignore[import-untyped]

# ---------------------------------------------------------------------------
# Ensure project root on sys.path
# ---------------------------------------------------------------------------
_PROJECT_ROOT: Path = Path(__file__).resolve().parents[4]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# Patch for optional stderr.buffer on some Windows environments
if not hasattr(sys.stderr, "buffer"):
    sys.stderr = io.TextIOWrapper(io.BytesIO(), encoding="utf-8")  # type: ignore[assignment]

# ---------------------------------------------------------------------------
# Lazy-import helper
# ---------------------------------------------------------------------------
MODULE_IMPORT_AVAILABLE: bool = True
_import_error_message: str = ""

try:
    from src.utils.custom_exceptions_errors_loggers.logger_custom import get_logger

    _logger = get_logger(__name__)
except Exception as _exc:
    MODULE_IMPORT_AVAILABLE = False
    _import_error_message = str(_exc)
    import logging

    _logger = logging.getLogger(__name__)


def _import_results_module(name: str) -> Any:
    """Return a shiny_tab_results submodule by short name."""
    try:
        return importlib.import_module(f"src.dashboard.shiny_tab_results.{name}")
    except ImportError as exc:
        raise AssertionError(f"Module '{name}' not importable: {exc}") from exc


# ===========================================================================
# Background
# ===========================================================================


@given("the shiny_tab_results modules are importable")
def step_results_modules_importable(context: Any) -> None:
    """Verify all Results-tab modules can be imported."""
    names = ["tab_results", "subtab_reporting", "subtab_simulation"]
    context.results_modules = {}
    for name in names:
        context.results_modules[name] = _import_results_module(name)
    _logger.debug("shiny_tab_results modules imported OK")


# ===========================================================================
# When — module export inspection
# ===========================================================================


@when("I check the tab_results module exports")
def step_check_tab_results_exports(context: Any) -> None:
    """Collect callable exports from tab_results."""
    mod = (
        context.results_modules.get("tab_results")
        or _import_results_module("tab_results")
    )
    context.results_exports = {
        "tab_results_ui": getattr(mod, "tab_results_ui", None),
        "tab_results_server": getattr(mod, "tab_results_server", None),
    }


@when("I check the subtab_reporting module exports")
def step_check_reporting_exports(context: Any) -> None:
    """Collect callable exports from subtab_reporting."""
    mod = (
        context.results_modules.get("subtab_reporting")
        or _import_results_module("subtab_reporting")
    )
    context.results_exports = {
        "subtab_reporting_ui": getattr(mod, "subtab_reporting_ui", None),
        "subtab_reporting_server": getattr(mod, "subtab_reporting_server", None),
    }


@when("I check the subtab_simulation module exports")
def step_check_simulation_exports(context: Any) -> None:
    """Collect callable exports from subtab_simulation."""
    mod = (
        context.results_modules.get("subtab_simulation")
        or _import_results_module("subtab_simulation")
    )
    context.results_exports = {
        "subtab_simulation_ui": getattr(mod, "subtab_simulation_ui", None),
        "subtab_simulation_server": getattr(mod, "subtab_simulation_server", None),
    }


# ===========================================================================
# When — security constant inspection
# ===========================================================================


@when("I inspect the subtab_reporting security constants")
def step_inspect_security_constants(context: Any) -> None:
    """Load security constants from subtab_reporting."""
    mod = (
        context.results_modules.get("subtab_reporting")
        or _import_results_module("subtab_reporting")
    )
    context.results_MAX_FILENAME_LENGTH = getattr(mod, "MAX_FILENAME_LENGTH", None)
    context.results_FORBIDDEN_FILENAME_PARTS = getattr(mod, "FORBIDDEN_FILENAME_PARTS", None)
    context.results_ALLOWED_FILENAME_PATTERN = getattr(mod, "ALLOWED_FILENAME_PATTERN", None)


# ===========================================================================
# When — sanitize_filename_for_security
# ===========================================================================


@when('I sanitize the results filename "{filename}"')
def step_sanitize_results_filename(context: Any, filename: str) -> None:
    """Call sanitize_filename_for_security with the given filename."""
    mod = (
        context.results_modules.get("subtab_reporting")
        or _import_results_module("subtab_reporting")
    )
    sanitize_fn = getattr(mod, "sanitize_filename_for_security")
    context.results_sanitize_result = sanitize_fn(filename)


@when("I sanitize a results empty filename")
def step_sanitize_results_empty_filename(context: Any) -> None:
    """Call sanitize_filename_for_security with an empty string."""
    mod = (
        context.results_modules.get("subtab_reporting")
        or _import_results_module("subtab_reporting")
    )
    sanitize_fn = getattr(mod, "sanitize_filename_for_security")
    context.results_sanitize_result = sanitize_fn("")


@when("I sanitize a results filename that is 300 characters long")
def step_sanitize_results_long_filename(context: Any) -> None:
    """Call sanitize_filename_for_security with a 300-char filename."""
    mod = (
        context.results_modules.get("subtab_reporting")
        or _import_results_module("subtab_reporting")
    )
    sanitize_fn = getattr(mod, "sanitize_filename_for_security")
    long_name = "x" * 295 + ".pdf"
    context.results_sanitize_result = sanitize_fn(long_name)


# ===========================================================================
# When — simulation constant inspection
# ===========================================================================


@when("I inspect the subtab_simulation constants")
def step_inspect_simulation_constants_results(context: Any) -> None:
    """Load constants from subtab_simulation."""
    mod = (
        context.results_modules.get("subtab_simulation")
        or _import_results_module("subtab_simulation")
    )
    context.results_ALL_ETF_SYMBOLS = getattr(mod, "ALL_ETF_SYMBOLS", None)
    context.results_DEFAULT_SELECTED_ETFS = getattr(mod, "DEFAULT_SELECTED_ETFS", None)
    context.results_DISTRIBUTION_CHOICES = getattr(mod, "DISTRIBUTION_CHOICES", None)
    context.results_RNG_TYPE_CHOICES = getattr(mod, "RNG_TYPE_CHOICES", None)


# ===========================================================================
# Then — callable assertions
# ===========================================================================


@then('"{export_name}" should be a callable')
def step_results_export_is_callable(context: Any, export_name: str) -> None:
    """Assert that the named export is callable."""
    exports = getattr(context, "results_exports", {})
    obj = exports.get(export_name)
    assert obj is not None, f"Export '{export_name}' is None or missing"
    assert callable(obj), f"'{export_name}' is not callable: {type(obj)}"


# ===========================================================================
# Then — security constant assertions
# ===========================================================================


@then("MAX_FILENAME_LENGTH should be a positive integer")
def step_results_max_filename_length_positive(context: Any) -> None:
    """Assert MAX_FILENAME_LENGTH is a positive integer."""
    val = context.results_MAX_FILENAME_LENGTH
    assert isinstance(val, int), f"Expected int, got {type(val)}"
    assert val > 0, f"MAX_FILENAME_LENGTH should be positive, got {val}"


@then('FORBIDDEN_FILENAME_PARTS should contain ".."')
def step_results_forbidden_has_dotdot(context: Any) -> None:
    """Assert '..' is in FORBIDDEN_FILENAME_PARTS."""
    parts = context.results_FORBIDDEN_FILENAME_PARTS
    assert ".." in parts, "'..'' not found in FORBIDDEN_FILENAME_PARTS"


@then('FORBIDDEN_FILENAME_PARTS should contain "\\\\"')
def step_results_forbidden_has_backslash(context: Any) -> None:
    r"""Assert '\\' is in FORBIDDEN_FILENAME_PARTS."""
    parts = context.results_FORBIDDEN_FILENAME_PARTS
    assert "\\" in parts, r"'\\' not found in FORBIDDEN_FILENAME_PARTS"


@then("ALLOWED_FILENAME_PATTERN should be a compiled regex")
def step_results_pattern_is_regex(context: Any) -> None:
    """Assert ALLOWED_FILENAME_PATTERN is a compiled re.Pattern."""
    pat = context.results_ALLOWED_FILENAME_PATTERN
    assert isinstance(pat, re.Pattern), f"Expected re.Pattern, got {type(pat)}"


# ===========================================================================
# Then — sanitize_filename_for_security assertions
# ===========================================================================


@then('the results sanitization result should indicate "valid"')
def step_results_sanitize_is_valid(context: Any) -> None:
    """Assert first element of sanitize result is True (accepted)."""
    is_valid, _, message = context.results_sanitize_result
    assert is_valid is True, f"Expected valid=True, message: {message}"


@then('the results sanitization result should indicate "invalid"')
def step_results_sanitize_is_invalid(context: Any) -> None:
    """Assert first element of sanitize result is False (rejected)."""
    is_valid, _, message = context.results_sanitize_result
    assert is_valid is False, f"Expected valid=False but got True, message: {message}"


@then("the results sanitized filename should not be empty")
def step_results_sanitized_not_empty(context: Any) -> None:
    """Assert the sanitized filename in the result tuple is non-empty."""
    _, sanitized, _ = context.results_sanitize_result
    assert len(sanitized) > 0, "Expected non-empty sanitized filename"


@then("the results sanitized filename should be empty")
def step_results_sanitized_is_empty(context: Any) -> None:
    """Assert the sanitized filename in the result tuple is empty (rejected)."""
    _, sanitized, _ = context.results_sanitize_result
    assert sanitized == "", f"Expected empty sanitized filename, got {sanitized!r}"


@then("the results sanitization return should be a tuple with 3 elements")
def step_results_sanitize_returns_tuple(context: Any) -> None:
    """Assert sanitize_filename_for_security returns a 3-tuple."""
    result = context.results_sanitize_result
    assert isinstance(result, tuple), f"Expected tuple, got {type(result)}"
    assert len(result) == 3, f"Expected 3-element tuple, got {len(result)}"


# ===========================================================================
# Then — simulation constant assertions
# ===========================================================================


@then("ALL_ETF_SYMBOLS should have 12 entries")
def step_results_all_etf_12(context: Any) -> None:
    """Assert ALL_ETF_SYMBOLS has exactly 12 entries."""
    symbols = context.results_ALL_ETF_SYMBOLS
    assert symbols is not None, "ALL_ETF_SYMBOLS is None"
    assert len(symbols) == 12, f"Expected 12 ETF symbols, got {len(symbols)}"


@then("every DEFAULT_SELECTED_ETFS entry should be in ALL_ETF_SYMBOLS")
def step_results_defaults_in_all(context: Any) -> None:
    """Assert every default ETF appears in ALL_ETF_SYMBOLS."""
    defaults = context.results_DEFAULT_SELECTED_ETFS
    all_symbols = context.results_ALL_ETF_SYMBOLS
    for sym in defaults:
        assert sym in all_symbols, f"Default ETF '{sym}' not in ALL_ETF_SYMBOLS"


@then('DISTRIBUTION_CHOICES should include keys "normal", "lognormal", and "student_t"')
def step_results_distribution_keys(context: Any) -> None:
    """Assert DISTRIBUTION_CHOICES contains the three required keys."""
    choices = context.results_DISTRIBUTION_CHOICES
    for key in ("normal", "lognormal", "student_t"):
        assert key in choices, f"Key '{key}' missing from DISTRIBUTION_CHOICES"


@then('RNG_TYPE_CHOICES should include keys "pcg64" and "mt19937"')
def step_results_rng_keys(context: Any) -> None:
    """Assert RNG_TYPE_CHOICES contains pcg64 and mt19937."""
    choices = context.results_RNG_TYPE_CHOICES
    for key in ("pcg64", "mt19937"):
        assert key in choices, f"Key '{key}' missing from RNG_TYPE_CHOICES"
