"""Robot Framework keyword library for daycount (day-count conventions) tests.

Tests cover
-----------
Keyword wrappers for Daycount_Convention enum inspection and year-fraction
calculations for all five standard conventions (30/360, 30/365, ACT/360,
ACT/365, ACT/ACT).  Includes multi-period additivity helpers.

Author:
    QWIM Development Team

Version:
    0.1.0

Last Modified:
    2026-03-01
"""

from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

# ---------------------------------------------------------------------------
# Ensure project root is on sys.path so that "src." imports resolve correctly
# ---------------------------------------------------------------------------
_PROJECT_ROOT: Path = Path(__file__).resolve().parents[4]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# ---------------------------------------------------------------------------
# Module-level import guard following project coding standards
# ---------------------------------------------------------------------------
MODULE_IMPORT_AVAILABLE: bool = True
_import_error_message: str = ""

try:
    from src.utils.dates_times_utils.daycount import (
        Daycount_Convention,
        get_daycount_calculator,
    )
    import logging as _logging
    _logger = _logging.getLogger(__name__)
except ImportError as _exc:
    MODULE_IMPORT_AVAILABLE = False
    _import_error_message = str(_exc)
    import logging as _logging
    _logger = _logging.getLogger(__name__)
    _logger.warning("Import failed — keywords will raise on use: %s", _exc)


def _require_imports() -> None:
    """Raise RuntimeError when source modules could not be imported."""
    if not MODULE_IMPORT_AVAILABLE:
        raise RuntimeError(
            f"daycount source modules could not be imported: {_import_error_message}"
        )


def _parse_date(date_str: str) -> date:
    """Parse ISO date string to a ``datetime.date`` object."""
    return date.fromisoformat(str(date_str))


def _get_convention(name: str) -> "Daycount_Convention":
    """Return ``Daycount_Convention`` member for *name* (case-insensitive)."""
    return Daycount_Convention[name.strip().upper()]


# ===========================================================================
# Year-fraction calculation keywords
# ===========================================================================


def calculate_year_fraction(
    start_date: str,
    end_date: str,
    convention_name: str,
) -> float:
    """Calculate year fraction between two dates for the given convention.

    Parameters
    ----------
    start_date : str
        ISO start date (e.g. ``"2024-01-01"``).
    end_date : str
        ISO end date (e.g. ``"2024-07-01"``).
    convention_name : str
        Name of the ``Daycount_Convention`` member (e.g. ``"THIRTY_360"``).

    Returns
    -------
    float
        Year fraction.
    """
    _require_imports()
    convention = _get_convention(convention_name)
    calculator = get_daycount_calculator(convention)
    return calculator.calc_year_fraction(_parse_date(start_date), _parse_date(end_date))


def calculate_year_fraction_thirty_360(start_date: str, end_date: str) -> float:
    """Year fraction using 30/360 convention.

    Parameters
    ----------
    start_date : str
        ISO start date.
    end_date : str
        ISO end date.

    Returns
    -------
    float
        Year fraction.
    """
    return calculate_year_fraction(start_date, end_date, "THIRTY_360")


def calculate_year_fraction_thirty_365(start_date: str, end_date: str) -> float:
    """Year fraction using 30/365 convention.

    Parameters
    ----------
    start_date : str
        ISO start date.
    end_date : str
        ISO end date.

    Returns
    -------
    float
        Year fraction.
    """
    return calculate_year_fraction(start_date, end_date, "THIRTY_365")


def calculate_year_fraction_actual_360(start_date: str, end_date: str) -> float:
    """Year fraction using ACT/360 convention.

    Parameters
    ----------
    start_date : str
        ISO start date.
    end_date : str
        ISO end date.

    Returns
    -------
    float
        Year fraction.
    """
    return calculate_year_fraction(start_date, end_date, "ACTUAL_360")


def calculate_year_fraction_actual_365(start_date: str, end_date: str) -> float:
    """Year fraction using ACT/365 convention.

    Parameters
    ----------
    start_date : str
        ISO start date.
    end_date : str
        ISO end date.

    Returns
    -------
    float
        Year fraction.
    """
    return calculate_year_fraction(start_date, end_date, "ACTUAL_365")


def calculate_year_fraction_actual_actual(start_date: str, end_date: str) -> float:
    """Year fraction using ACT/ACT convention.

    Parameters
    ----------
    start_date : str
        ISO start date.
    end_date : str
        ISO end date.

    Returns
    -------
    float
        Year fraction.
    """
    return calculate_year_fraction(start_date, end_date, "ACTUAL_ACTUAL")


# ===========================================================================
# Assertion keywords
# ===========================================================================


def year_fraction_should_be_approximately(
    actual: float,
    expected: float,
    rel_tolerance: float = 1e-3,
) -> None:
    """Assert *actual* ≈ *expected* within a relative tolerance.

    Parameters
    ----------
    actual : float
        Computed year fraction.
    expected : float
        Expected year fraction.
    rel_tolerance : float, optional
        Relative tolerance (default 0.001).

    Raises
    ------
    AssertionError
        If the difference exceeds the tolerance.
    """
    actual_f = float(actual)
    expected_f = float(expected)
    tolerance = max(abs(expected_f) * rel_tolerance, 1e-5)
    assert abs(actual_f - expected_f) <= tolerance, (
        f"Expected year fraction ≈ {expected_f}, got {actual_f:.6f} "
        f"(diff={abs(actual_f - expected_f):.6f}, tol={tolerance:.6f})"
    )


def year_fraction_should_be_greater_than(actual: float, threshold: float) -> None:
    """Assert *actual* > *threshold*.

    Parameters
    ----------
    actual : float
        Computed year fraction.
    threshold : float
        Lower bound.

    Raises
    ------
    AssertionError
        If *actual* ≤ *threshold*.
    """
    assert float(actual) > float(threshold), (
        f"Expected year fraction > {threshold}, got {actual}"
    )


def year_fraction_should_be_less_than(actual: float, threshold: float) -> None:
    """Assert *actual* < *threshold*.

    Parameters
    ----------
    actual : float
        Computed year fraction.
    threshold : float
        Upper bound.

    Raises
    ------
    AssertionError
        If *actual* ≥ *threshold*.
    """
    assert float(actual) < float(threshold), (
        f"Expected year fraction < {threshold}, got {actual}"
    )


def two_year_fractions_should_sum_to_approximately(
    yf_a: float,
    yf_b: float,
    expected_sum: float,
    rel_tolerance: float = 1e-3,
) -> None:
    """Assert that the sum of two year fractions equals *expected_sum*.

    Parameters
    ----------
    yf_a : float
        First year fraction.
    yf_b : float
        Second year fraction.
    expected_sum : float
        Expected total.
    rel_tolerance : float, optional
        Relative tolerance.

    Raises
    ------
    AssertionError
        If the sum differs from *expected_sum* by more than the tolerance.
    """
    total = float(yf_a) + float(yf_b)
    expected_f = float(expected_sum)
    tolerance = max(abs(expected_f) * rel_tolerance, 1e-5)
    assert abs(total - expected_f) <= tolerance, (
        f"Expected sum ≈ {expected_f}, got {total:.6f} "
        f"(={float(yf_a):.6f}+{float(yf_b):.6f})"
    )


def daycount_convention_count_should_equal(expected: int) -> None:
    """Assert the number of Daycount_Convention members equals *expected*.

    Parameters
    ----------
    expected : int
        Expected count.

    Raises
    ------
    AssertionError
        If count differs.
    """
    _require_imports()
    actual = len(list(Daycount_Convention))
    assert actual == int(expected), (
        f"Expected {expected} Daycount_Convention members, got {actual}"
    )
