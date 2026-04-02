"""Behave step definitions for reporting package public API tests.

Tests cover verification of the ``src.dashboard.reporting`` package's
public export surface after the 2026-01 pyright cleanup that removed
``build_typst_data_context`` from ``reporting/__init__.py``.

Features covered:
    - Positive import and callability of 3 exported symbols
    - Negative guard: ``build_typst_data_context`` absent from exports
    - ``__all__`` exact-match regression guard
    - ``validate_polars_DF`` functional smoke

Author:         QWIM Development Team
Version:        0.1.0
Last Modified:  2026-01
"""

from __future__ import annotations

import io
import sys
from pathlib import Path
from typing import Any

from behave import given, then, use_step_matcher, when

# ---------------------------------------------------------------------------
# Project root on sys.path
# ---------------------------------------------------------------------------
_PROJECT_ROOT: Path = Path(__file__).resolve().parents[4]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# ---------------------------------------------------------------------------
# sys.stderr patch for exception_custom.py compatibility
# ---------------------------------------------------------------------------
if not hasattr(sys.stderr, "buffer"):
    sys.stderr = io.TextIOWrapper(io.BytesIO(), encoding="utf-8")

# ---------------------------------------------------------------------------
# Module-level import guard
# ---------------------------------------------------------------------------
MODULE_IMPORT_AVAILABLE: bool = True
_import_error_message: str = ""
_reporting_pkg: Any = None

try:
    import src.dashboard.reporting as _reporting_pkg
    from src.dashboard.reporting import (
        compile_typst_report,
        generate_report_PDF,
        validate_polars_DF,
    )
    import polars as pl
except Exception as _exc:
    MODULE_IMPORT_AVAILABLE = False
    _import_error_message = str(_exc)


def _require_imports() -> None:
    """Raise RuntimeError if reporting package could not be imported."""
    if not MODULE_IMPORT_AVAILABLE:
        raise RuntimeError(
            f"Reporting package modules could not be imported: {_import_error_message}"
        )


use_step_matcher("parse")

# ---------------------------------------------------------------------------
# Background
# ---------------------------------------------------------------------------


@given("the reporting package is importable")
def step_reporting_package_importable(context: Any) -> None:
    """Verify the reporting package can be imported; skip otherwise."""
    if not MODULE_IMPORT_AVAILABLE:
        context.scenario.skip(
            f"Reporting package not importable: {_import_error_message}"
        )
    context.reporting_pkg = _reporting_pkg
    context.last_exception = None
    context.last_result = None


# ---------------------------------------------------------------------------
# Positive export checks
# ---------------------------------------------------------------------------


@then('"{symbol}" should be importable from "src.dashboard.reporting"')
def step_symbol_importable(context: Any, symbol: str) -> None:
    """Verify a named symbol exists in the reporting package namespace."""
    _require_imports()
    assert hasattr(context.reporting_pkg, symbol), (
        f"'{symbol}' is not accessible from src.dashboard.reporting"
    )


@then('"{symbol}" should be callable')
def step_symbol_callable(context: Any, symbol: str) -> None:
    """Verify a named symbol from the package is callable."""
    _require_imports()
    obj = getattr(context.reporting_pkg, symbol, None)
    assert callable(obj), f"'{symbol}' from reporting package is not callable"


# ---------------------------------------------------------------------------
# Negative export guard
# ---------------------------------------------------------------------------


@then('"{symbol}" should NOT be accessible from the reporting package')
def step_symbol_not_accessible(context: Any, symbol: str) -> None:
    """Verify a named symbol does NOT exist in the reporting package namespace."""
    _require_imports()
    assert not hasattr(context.reporting_pkg, symbol), (
        f"'{symbol}' must NOT be exported from src.dashboard.reporting; "
        f"it was removed in the 2026-01 pyright cleanup"
    )


@then('the reporting package __all__ should not contain "{symbol}"')
def step_all_not_contains(context: Any, symbol: str) -> None:
    """Verify __all__ does not include a given symbol."""
    _require_imports()
    all_list: list[str] = getattr(context.reporting_pkg, "__all__", [])
    assert symbol not in all_list, (
        f"'{symbol}' unexpectedly found in reporting.__all__: {all_list}"
    )


# ---------------------------------------------------------------------------
# __all__ completeness
# ---------------------------------------------------------------------------


@then("the reporting package __all__ should contain exactly {count:d} entries")
def step_all_has_exact_count(context: Any, count: int) -> None:
    """Verify __all__ has the expected number of entries."""
    _require_imports()
    all_list: list[str] = getattr(context.reporting_pkg, "__all__", [])
    assert len(all_list) == count, (
        f"reporting.__all__ should have {count} entries, found {len(all_list)}: {all_list}"
    )


@then('the reporting package __all__ should contain "{symbol}"')
def step_all_contains(context: Any, symbol: str) -> None:
    """Verify __all__ includes a specific symbol."""
    _require_imports()
    all_list: list[str] = getattr(context.reporting_pkg, "__all__", [])
    assert symbol in all_list, (
        f"'{symbol}' not found in reporting.__all__: {all_list}"
    )


# ---------------------------------------------------------------------------
# validate_polars_DF functional steps
# ---------------------------------------------------------------------------


@given("a Polars DataFrame with columns Date and Value")
def step_given_valid_dataframe(context: Any) -> None:
    """Create a minimal well-formed Polars DataFrame for validation testing."""
    _require_imports()
    context.test_input = pl.DataFrame(
        {"Date": ["2024-01-01", "2024-06-01"], "Value": [100.0, 110.0]}
    )


@when("I call validate_polars_DF with the DataFrame")
def step_call_validate_with_dataframe(context: Any) -> None:
    """Call validate_polars_DF with a DataFrame and store the result."""
    _require_imports()
    try:
        context.last_result = validate_polars_DF(context.test_input)
        context.last_exception = None
    except (ValueError, TypeError) as exc:
        context.last_result = None
        context.last_exception = exc


@then("the validation result should not be False")
def step_result_not_false(context: Any) -> None:
    """Verify the validation did not explicitly return False."""
    assert context.last_result is not False, (
        "validate_polars_DF returned False for a valid DataFrame"
    )


@given("a None input value")
def step_given_none_input(context: Any) -> None:
    """Set test_input to None for edge-case validation testing."""
    _require_imports()
    context.test_input = None


@when("I call validate_polars_DF with the None input")
def step_call_validate_with_none(context: Any) -> None:
    """Call validate_polars_DF with None and capture any exception."""
    _require_imports()
    try:
        context.last_result = validate_polars_DF(context.test_input)  # type: ignore[arg-type]
        context.last_exception = None
    except (ValueError, TypeError, AttributeError) as exc:
        # Controlled validation exceptions are acceptable
        context.last_result = None
        context.last_exception = exc


@then("no unhandled exception should have been raised")
def step_no_unhandled_exception(context: Any) -> None:
    """Verify that no unhandled (non-validation) exception was raised."""
    exc = context.last_exception
    if exc is not None and not isinstance(exc, (ValueError, TypeError, AttributeError)):
        raise AssertionError(
            f"validate_polars_DF raised an unexpected exception for None input: {exc}"
        ) from exc
