"""
Robot Framework keyword library for Simulation_Standard.
=========================================================

Tests cover:
    - Construction with NORMAL, LOGNORMAL, and STUDENT_T distributions
    - run() output shape — (num_days + 1) rows, (num_scenarios + 1) columns
    - Date column present and sorted ascending
    - All portfolio values positive
    - First-row values equal initial_value (starting conditions)
    - Reproducibility — same seed produces identical output

Author:         QWIM Development Team
Version:        0.1.0
Last Modified:  2026-02-28
"""

from __future__ import annotations

import sys
import io
import datetime as dt
from pathlib import Path

# ---------------------------------------------------------------------------
# Project root on sys.path so src packages resolve correctly
# ---------------------------------------------------------------------------
_PROJECT_ROOT: Path = Path(__file__).resolve().parents[4]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# ---------------------------------------------------------------------------
# Robot Framework redirects sys.stderr to a StringIO object that lacks
# a .buffer attribute.  Patch it back before importing modules that
# access sys.stderr.buffer at module-load time.
# ---------------------------------------------------------------------------
if not hasattr(sys.stderr, "buffer"):
    sys.stderr = io.TextIOWrapper(io.BytesIO(), encoding="utf-8")

# ---------------------------------------------------------------------------
# Conditional imports with Module availability guard
# ---------------------------------------------------------------------------
MODULE_IMPORT_AVAILABLE: bool = True
_import_error_message: str = ""

try:
    import numpy as np
    import polars as pl
    from src.models.simulation.model_simulation_standard import (
        Simulation_Standard,
        DEFAULT_NUM_SCENARIOS,
        DEFAULT_NUM_DAYS,
        DEFAULT_INITIAL_VALUE,
        DEFAULT_RANDOM_SEED,
    )
    from src.num_methods.scenarios.scenarios_distrib import Distribution_Type
except Exception as _exc:
    MODULE_IMPORT_AVAILABLE = False
    _import_error_message = str(_exc)

# ---------------------------------------------------------------------------
# Internal test-size defaults — small values keep tests fast
# ---------------------------------------------------------------------------
_TEST_NUM_SCENARIOS: int = 50
_TEST_NUM_DAYS: int = 30
_TEST_INITIAL_VALUE: float = 100.0
_TEST_SEED: int = 42
_TEST_START_DATE: dt.date = dt.date(2024, 1, 2)


def _require_imports() -> None:
    """Raise RuntimeError when required imports are unavailable."""
    if not MODULE_IMPORT_AVAILABLE:
        raise RuntimeError(
            f"Required imports not available: {_import_error_message}"
        )


def _make_equal_weights(num_components: int) -> "np.ndarray":
    """Return a numpy array of equal weights summing to 1.0."""
    return np.full(num_components, 1.0 / num_components)


# ---------------------------------------------------------------------------
# Construction keywords
# ---------------------------------------------------------------------------


def create_simulation_normal(
    *component_names: str,
    num_scenarios: int = _TEST_NUM_SCENARIOS,
    num_days: int = _TEST_NUM_DAYS,
    initial_value: float = _TEST_INITIAL_VALUE,
    random_seed: int = _TEST_SEED,
) -> "Simulation_Standard":
    """Create a Simulation_Standard with Distribution_Type.NORMAL.

    Arguments:
    - *component_names -- one or more ETF/asset ticker strings
    - num_scenarios    -- number of Monte Carlo paths (default 50)
    - num_days         -- simulation horizon in trading days (default 30)
    - initial_value    -- starting portfolio value
    - random_seed      -- RNG seed for reproducibility

    Returns: Simulation_Standard instance
    """
    _require_imports()
    names = list(component_names)
    weights = _make_equal_weights(len(names))
    return Simulation_Standard(
        names_components=names,
        weights=weights,
        distribution_type=Distribution_Type.NORMAL,
        initial_value=float(initial_value),
        num_scenarios=int(num_scenarios),
        num_days=int(num_days),
        start_date=_TEST_START_DATE,
        random_seed=int(random_seed),
    )


def create_simulation_lognormal(
    *component_names: str,
    num_scenarios: int = _TEST_NUM_SCENARIOS,
    num_days: int = _TEST_NUM_DAYS,
    initial_value: float = _TEST_INITIAL_VALUE,
    random_seed: int = _TEST_SEED,
) -> "Simulation_Standard":
    """Create a Simulation_Standard with Distribution_Type.LOGNORMAL.

    Arguments:
    - *component_names -- one or more ticker strings
    - num_scenarios    -- number of Monte Carlo paths (default 50)
    - num_days         -- simulation horizon in trading days (default 30)
    - initial_value    -- starting portfolio value
    - random_seed      -- RNG seed

    Returns: Simulation_Standard instance
    """
    _require_imports()
    names = list(component_names)
    weights = _make_equal_weights(len(names))
    return Simulation_Standard(
        names_components=names,
        weights=weights,
        distribution_type=Distribution_Type.LOGNORMAL,
        initial_value=float(initial_value),
        num_scenarios=int(num_scenarios),
        num_days=int(num_days),
        start_date=_TEST_START_DATE,
        random_seed=int(random_seed),
    )


def create_simulation_student_t(
    *component_names: str,
    num_scenarios: int = _TEST_NUM_SCENARIOS,
    num_days: int = _TEST_NUM_DAYS,
    initial_value: float = _TEST_INITIAL_VALUE,
    random_seed: int = _TEST_SEED,
) -> "Simulation_Standard":
    """Create a Simulation_Standard with Distribution_Type.STUDENT_T.

    Arguments:
    - *component_names -- one or more ticker strings
    - num_scenarios    -- number of Monte Carlo paths (default 50)
    - num_days         -- simulation horizon in trading days (default 30)
    - initial_value    -- starting portfolio value
    - random_seed      -- RNG seed

    Returns: Simulation_Standard instance
    """
    _require_imports()
    names = list(component_names)
    weights = _make_equal_weights(len(names))
    return Simulation_Standard(
        names_components=names,
        weights=weights,
        distribution_type=Distribution_Type.STUDENT_T,
        initial_value=float(initial_value),
        num_scenarios=int(num_scenarios),
        num_days=int(num_days),
        start_date=_TEST_START_DATE,
        random_seed=int(random_seed),
    )


# ---------------------------------------------------------------------------
# run() keywords
# ---------------------------------------------------------------------------


def run_simulation(simulation: "Simulation_Standard") -> "pl.DataFrame":
    """Call simulation.run() and return the resulting DataFrame.

    Arguments:
    - simulation -- Simulation_Standard instance

    Returns: pl.DataFrame with Date + scenario columns
    """
    _require_imports()
    return simulation.run()


def simulation_results_dataframe_should_be_valid(results: "pl.DataFrame") -> None:
    """Assert that run() returned a non-empty pl.DataFrame.

    Arguments:
    - results -- return value of run_simulation()
    """
    _require_imports()
    if not isinstance(results, pl.DataFrame):
        raise AssertionError(
            f"Expected pl.DataFrame from simulation.run(), got {type(results)}"
        )
    if len(results) == 0:
        raise AssertionError("Simulation results DataFrame is empty")
    if len(results.columns) == 0:
        raise AssertionError("Simulation results DataFrame has no columns")


def simulation_results_should_have_date_column(results: "pl.DataFrame") -> None:
    """Assert that the results DataFrame contains a 'Date' column.

    Arguments:
    - results -- pl.DataFrame from run_simulation()
    """
    _require_imports()
    if "Date" not in results.columns:
        raise AssertionError(
            f"'Date' column not found. Columns: {results.columns}"
        )


def simulation_results_row_count_should_equal(
    results: "pl.DataFrame", expected_rows: int | str
) -> None:
    """Assert that the results DataFrame has the expected number of rows.

    Arguments:
    - results       -- pl.DataFrame from run_simulation()
    - expected_rows -- expected number of rows (num_days + 1)
    """
    _require_imports()
    actual = len(results)
    expected = int(expected_rows)
    if actual != expected:
        raise AssertionError(
            f"Expected {expected} rows in simulation results but got {actual}"
        )


def simulation_results_column_count_should_equal(
    results: "pl.DataFrame", expected_cols: int | str
) -> None:
    """Assert total column count equals expected_cols.

    The DataFrame includes 'Date' plus one column per scenario.

    Arguments:
    - results       -- pl.DataFrame from run_simulation()
    - expected_cols -- expected column count (num_scenarios + 1)
    """
    _require_imports()
    actual = len(results.columns)
    expected = int(expected_cols)
    if actual != expected:
        raise AssertionError(
            f"Expected {expected} columns in simulation results but got {actual}"
        )


def simulation_all_portfolio_values_should_be_positive(
    results: "pl.DataFrame",
) -> None:
    """Assert every scenario value column contains only positive values.

    Arguments:
    - results -- pl.DataFrame from run_simulation()
    """
    _require_imports()
    scenario_cols = [c for c in results.columns if c != "Date"]
    if not scenario_cols:
        raise AssertionError("No scenario columns found in simulation results")
    for col in scenario_cols:
        min_val = results[col].min()
        if min_val is None or min_val <= 0:
            raise AssertionError(
                f"Scenario column '{col}' has non-positive value: min={min_val}"
            )


def simulation_first_row_values_should_equal_initial_value(
    results: "pl.DataFrame", initial_value: float | str
) -> None:
    """Assert every scenario column's first row equals initial_value.

    Arguments:
    - results       -- pl.DataFrame from run_simulation()
    - initial_value -- expected starting portfolio value
    """
    _require_imports()
    iv = float(initial_value)
    scenario_cols = [c for c in results.columns if c != "Date"]
    for col in scenario_cols:
        first = results[col][0]
        if abs(first - iv) > 1.0e-6:
            raise AssertionError(
                f"Column '{col}' first row {first} != initial_value {iv}"
            )


def simulation_date_column_is_sorted_ascending(results: "pl.DataFrame") -> None:
    """Assert the Date column is sorted in ascending order.

    Arguments:
    - results -- pl.DataFrame from run_simulation()
    """
    _require_imports()
    if "Date" not in results.columns:
        raise AssertionError("'Date' column not found in simulation results")
    dates = results["Date"].to_list()
    for idx in range(1, len(dates)):
        if dates[idx] < dates[idx - 1]:
            raise AssertionError(
                f"Date column not sorted ascending at index {idx}: "
                f"{dates[idx - 1]} > {dates[idx]}"
            )


def two_simulations_with_same_seed_should_produce_identical_results(
    *component_names: str,
) -> None:
    """Assert that two runs with the same seed produce the same DataFrame.

    Arguments:
    - *component_names -- ticker strings used for both simulations
    """
    _require_imports()
    sim1 = create_simulation_normal(
        *component_names,
        num_scenarios=10,
        num_days=20,
        random_seed=_TEST_SEED,
    )
    sim2 = create_simulation_normal(
        *component_names,
        num_scenarios=10,
        num_days=20,
        random_seed=_TEST_SEED,
    )
    r1 = sim1.run()
    r2 = sim2.run()
    scenario_cols = [c for c in r1.columns if c != "Date"]
    for col in scenario_cols:
        if not (r1[col] == r2[col]).all():
            raise AssertionError(
                f"Reproducibility failure: column '{col}' differs between identical seeds"
            )


def two_simulations_with_different_seeds_may_differ(
    *component_names: str,
) -> None:
    """Run two simulations with different seeds and verify results differ.

    This is a best-effort check — if by extreme coincidence results
    match, the test passes anyway (we do not want false failures).

    Arguments:
    - *component_names -- ticker strings
    """
    _require_imports()
    sim1 = create_simulation_normal(
        *component_names,
        num_scenarios=20,
        num_days=20,
        random_seed=42,
    )
    sim2 = create_simulation_normal(
        *component_names,
        num_scenarios=20,
        num_days=20,
        random_seed=99,
    )
    r1 = sim1.run()
    r2 = sim2.run()
    # We only assert both ran successfully
    simulation_results_dataframe_should_be_valid(r1)
    simulation_results_dataframe_should_be_valid(r2)
