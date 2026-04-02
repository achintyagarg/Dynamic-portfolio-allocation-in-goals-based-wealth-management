"""Behave step definitions for daycount (day-count conventions) feature.

Covers:
  - Daycount_Convention enum members
  - get_daycount_calculator factory
  - year-fraction calculations for 30/360, 30/365, ACT/360, ACT/365, ACT/ACT
  - Multi-period additivity invariant

Author:         QWIM Development Team
Version:        0.1.0
Last Modified:  2026-03-01
"""

from __future__ import annotations

import io
import sys
from datetime import date
from pathlib import Path

from behave import given, then, when

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
    sys.stderr = io.TextIOWrapper(io.BytesIO(), encoding="utf-8")  # type: ignore[assignment]

# ---------------------------------------------------------------------------
# Module-level import guard
# ---------------------------------------------------------------------------
MODULE_IMPORT_AVAILABLE: bool = True
_import_error_message: str = ""

try:
    from src.utils.dates_times_utils.daycount import (
        Daycount_Convention,
        get_daycount_calculator,
    )
except Exception as _exc:
    MODULE_IMPORT_AVAILABLE = False
    _import_error_message = str(_exc)


def _require_imports() -> None:
    if not MODULE_IMPORT_AVAILABLE:
        raise RuntimeError(
            f"daycount modules could not be imported: {_import_error_message}"
        )


def _parse_date(date_str: str) -> date:
    """Parse ISO date string to date object."""
    return date.fromisoformat(date_str)


def _get_convention(name: str) -> "Daycount_Convention":
    """Convert convention name string to Daycount_Convention enum member."""
    return Daycount_Convention[name.upper()]


def _calc_year_fraction(start_str: str, end_str: str, convention_name: str) -> float:
    """Helper: calculate year fraction between two date strings."""
    _require_imports()
    convention = _get_convention(convention_name)
    calculator = get_daycount_calculator(convention)
    start = _parse_date(start_str)
    end = _parse_date(end_str)
    return calculator.calc_year_fraction(start, end)


# ===========================================================================
# Given steps
# ===========================================================================


@given("day-count utility modules are importable")
def step_given_daycount_importable(context) -> None:
    _require_imports()
    context.year_fraction_a = None
    context.year_fraction_b = None


# ===========================================================================
# When steps
# ===========================================================================


@when('I calculate the year fraction from "{start}" to "{end}" using "{convention}"')
def step_calculate_year_fraction(context, start: str, end: str, convention: str) -> None:
    _require_imports()
    yf = _calc_year_fraction(start, end, convention)
    # Store result — first call goes to year_fraction_a, second to year_fraction_b
    if context.year_fraction_a is None:
        context.year_fraction_a = yf
    else:
        context.year_fraction_b = yf


@when('I create a daycount calculator for convention "{convention}"')
def step_create_daycount_calculator(context, convention: str) -> None:
    _require_imports()
    conv_enum = _get_convention(convention)
    context.active_calculator = get_daycount_calculator(conv_enum)
    context.active_convention = convention


# ===========================================================================
# Then steps
# ===========================================================================


@then("the year fraction should be approximately {expected:f}")
def step_year_fraction_approximately(context, expected: float) -> None:
    actual = context.year_fraction_a
    assert actual is not None, "No year fraction has been calculated yet"
    tolerance = max(abs(expected) * 1e-3, 1e-5)
    assert abs(actual - expected) <= tolerance, (
        f"Expected year fraction ≈ {expected}, got {actual:.6f} "
        f"(diff={abs(actual - expected):.6f})"
    )


@then("the year fraction should be greater than {threshold:f}")
def step_year_fraction_greater_than(context, threshold: float) -> None:
    actual = context.year_fraction_a
    assert actual is not None, "No year fraction has been calculated yet"
    assert actual > threshold, (
        f"Expected year fraction > {threshold}, got {actual:.6f}"
    )


@then("the year fraction should be less than {threshold:f}")
def step_year_fraction_less_than(context, threshold: float) -> None:
    actual = context.year_fraction_a
    assert actual is not None, "No year fraction has been calculated yet"
    assert actual < threshold, (
        f"Expected year fraction < {threshold}, got {actual:.6f}"
    )


@then("the calculator should return a positive year fraction for a 6-month period")
def step_calculator_positive_six_month(context) -> None:
    calc = context.active_calculator
    start = date(2024, 1, 1)
    end = date(2024, 7, 1)
    yf = calc.calc_year_fraction(start, end)
    assert yf > 0, (
        f"Expected positive year fraction for 6-month period, got {yf}"
    )


@then("the sum of both year fractions should be approximately {expected:f}")
def step_sum_of_fractions_approximately(context, expected: float) -> None:
    yf_a = context.year_fraction_a
    yf_b = context.year_fraction_b
    assert yf_a is not None, "First year fraction not computed"
    assert yf_b is not None, "Second year fraction not computed"
    total = yf_a + yf_b
    tolerance = max(abs(expected) * 1e-3, 1e-5)
    assert abs(total - expected) <= tolerance, (
        f"Expected sum ≈ {expected}, got {total:.6f} (={yf_a:.6f}+{yf_b:.6f})"
    )
