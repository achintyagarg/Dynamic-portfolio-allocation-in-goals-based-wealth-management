"""Behave step definitions for Simulation_Standard.

Tests cover:
    - Construction with NORMAL, LOGNORMAL, and STUDENT_T distributions
    - run() output: shape (num_days+1 rows, num_scenarios+1 cols), Date column
    - All portfolio values positive; first-row values equal initial_value
    - Reproducibility — same seed produces identical output

Author:         QWIM Development Team
Version:        0.1.0
Last Modified:  2026-02-28
"""

from __future__ import annotations

import datetime as dt
import io
import sys
from pathlib import Path

from behave import given, then, when

# ---------------------------------------------------------------------------
# Project root on sys.path
# ---------------------------------------------------------------------------
_PROJECT_ROOT: Path = Path(__file__).resolve().parents[4]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# ---------------------------------------------------------------------------
# sys.stderr patch — exception_custom.py accesses sys.stderr.buffer
# at module-load time; Behave may replace sys.stderr with StringIO.
# ---------------------------------------------------------------------------
if not hasattr(sys.stderr, "buffer"):
    sys.stderr = io.TextIOWrapper(io.BytesIO(), encoding="utf-8")

# ---------------------------------------------------------------------------
# Module-level import guard
# ---------------------------------------------------------------------------
MODULE_IMPORT_AVAILABLE: bool = True
_import_error_message: str = ""

try:
    import numpy as np
    import polars as pl
    from src.models.simulation.model_simulation_standard import Simulation_Standard
    from src.num_methods.scenarios.scenarios_distrib import Distribution_Type
except Exception as _exc:
    MODULE_IMPORT_AVAILABLE = False
    _import_error_message = str(_exc)

# ---------------------------------------------------------------------------
# Fixed test-size constants — kept small for fast test execution
# ---------------------------------------------------------------------------
_NUM_SCENARIOS: int = 50
_NUM_DAYS: int = 30
_INITIAL_VALUE: float = 100.0
_SEED: int = 42
_START_DATE: dt.date = dt.date(2024, 1, 2)


def _require_imports() -> None:
    if not MODULE_IMPORT_AVAILABLE:
        raise RuntimeError(
            f"Simulation source modules could not be imported: {_import_error_message}"
        )


def _equal_weights(n: int) -> "np.ndarray":
    """Return a 1-D array of equal weights summing to 1.0."""
    return np.full(n, 1.0 / n)


def _make_simulation(
    component_names: list[str],
    distribution_type: "Distribution_Type",
    random_seed: int = _SEED,
    mean_returns: "np.ndarray | None" = None,
) -> "Simulation_Standard":
    """Create a Simulation_Standard instance with default test sizes."""
    _require_imports()
    n = len(component_names)
    # Lognormal mapping formula: sigma_ln = log(1 + cov/mu^2).
    # With the default covariance diag(0.0004), mu must satisfy mu^2 >> cov to
    # yield small sigma_ln values.  Using mu=1.0 (the "multiplicative" mean
    # interpretation) gives sigma_ln ≈ 0.0004 and mu_ln ≈ -0.0002, i.e.
    # near-zero arithmetic daily returns with ~2% daily vol — physically sensible.
    if mean_returns is None and distribution_type == Distribution_Type.LOGNORMAL:
        mean_returns = np.ones(n, dtype=np.float64)  # unit multiplicative mean
    return Simulation_Standard(
        names_components=component_names,
        weights=_equal_weights(n),
        distribution_type=distribution_type,
        mean_returns=mean_returns,
        initial_value=_INITIAL_VALUE,
        num_scenarios=_NUM_SCENARIOS,
        num_days=_NUM_DAYS,
        start_date=_START_DATE,
        random_seed=random_seed,
    )


# ===========================================================================
# When — construction
# ===========================================================================


@when(u'I create a Normal simulation with components "{components}"')
def step_create_normal_simulation(context, components: str) -> None:
    _require_imports()
    names = [c.strip() for c in components.split(",")]
    context.simulation = _make_simulation(names, Distribution_Type.NORMAL)


@when(u'I create a Lognormal simulation with components "{components}"')
def step_create_lognormal_simulation(context, components: str) -> None:
    _require_imports()
    names = [c.strip() for c in components.split(",")]
    context.simulation = _make_simulation(names, Distribution_Type.LOGNORMAL)


@when(u'I create a Student-T simulation with components "{components}"')
def step_create_student_t_simulation(context, components: str) -> None:
    _require_imports()
    names = [c.strip() for c in components.split(",")]
    context.simulation = _make_simulation(names, Distribution_Type.STUDENT_T)


@when(u'I create two identical Normal simulations with components "{components}" and the same seed')
def step_create_two_identical_simulations(context, components: str) -> None:
    _require_imports()
    names = [c.strip() for c in components.split(",")]
    context.sim1 = _make_simulation(names, Distribution_Type.NORMAL, random_seed=_SEED)
    context.sim2 = _make_simulation(names, Distribution_Type.NORMAL, random_seed=_SEED)


@when(u'I create two Normal simulations with components "{components}" using seed {seed1:d} and seed {seed2:d}')
def step_create_two_different_seed_simulations(
    context, components: str, seed1: int, seed2: int
) -> None:
    _require_imports()
    names = [c.strip() for c in components.split(",")]
    context.sim1 = _make_simulation(names, Distribution_Type.NORMAL, random_seed=seed1)
    context.sim2 = _make_simulation(names, Distribution_Type.NORMAL, random_seed=seed2)


# ===========================================================================
# Then — construction assertion
# ===========================================================================


@then(u'the simulation should be created without error')
def step_simulation_created(context) -> None:
    assert context.simulation is not None, "Simulation was not created (is None)"


# ===========================================================================
# When — run
# ===========================================================================


@when(u'I run the simulation')
def step_run_simulation(context) -> None:
    _require_imports()
    context.sim_results = context.simulation.run()


@when(u'I run both simulations')
def step_run_both_simulations(context) -> None:
    _require_imports()
    context.sim1_results = context.sim1.run()
    context.sim2_results = context.sim2.run()


# ===========================================================================
# Then — results validity
# ===========================================================================


@then(u'the simulation results should be a valid non-empty DataFrame')
def step_results_valid_df(context) -> None:
    _require_imports()
    results = context.sim_results
    assert isinstance(results, pl.DataFrame), (
        f"Expected pl.DataFrame from simulation.run(), got {type(results)}"
    )
    assert len(results) > 0, "Simulation results DataFrame is empty"
    assert len(results.columns) > 0, "Simulation results DataFrame has no columns"


@then(u'the simulation results should contain a "{col}" column')
def step_results_has_column(context, col: str) -> None:
    assert col in context.sim_results.columns, (
        f"'{col}' column not found. Columns: {context.sim_results.columns}"
    )


@then(u'the simulation results should have {expected_rows:d} rows')
def step_results_row_count(context, expected_rows: int) -> None:
    actual = len(context.sim_results)
    assert actual == expected_rows, (
        f"Expected {expected_rows} rows in simulation results but got {actual}"
    )


@then(u'the simulation results should have {expected_cols:d} columns')
def step_results_column_count(context, expected_cols: int) -> None:
    actual = len(context.sim_results.columns)
    assert actual == expected_cols, (
        f"Expected {expected_cols} columns in simulation results but got {actual}"
    )


@then(u'all simulation portfolio values should be positive')
def step_all_values_positive(context) -> None:
    results = context.sim_results
    scenario_cols = [c for c in results.columns if c != "Date"]
    assert scenario_cols, "No scenario columns found in simulation results"
    for col in scenario_cols:
        min_val = results[col].min()
        assert min_val is not None and min_val > 0, (
            f"Scenario column '{col}' has non-positive value: min={min_val}"
        )


@then(u'each scenario column first value should equal {expected_value:g}')
def step_first_row_equals_initial(context, expected_value: float) -> None:
    results = context.sim_results
    scenario_cols = [c for c in results.columns if c != "Date"]
    for col in scenario_cols:
        first = results[col][0]
        assert abs(float(first) - float(expected_value)) <= 1e-6, (
            f"Column '{col}' first row {first} ≠ initial_value {expected_value}"
        )

@then(u'each scenario column first value should be within {pct:d} percent of {expected_value:g}')
def step_first_row_within_pct(context, pct: int, expected_value: float) -> None:
    """Check first-row values fall within pct% of expected_value.

    The simulation starts compounding from day 1, so the first row is
    ``initial_value * (1 + return_day1)`` rather than ``initial_value`` itself.
    A generous tolerance ensures the check is meaningful yet not brittle.
    """
    results = context.sim_results
    scenario_cols = [c for c in results.columns if c != "Date"]
    threshold = expected_value * (pct / 100.0)
    for col in scenario_cols:
        first = float(results[col][0])
        assert abs(first - expected_value) <= threshold, (
            f"Column '{col}' first row {first:.4f} not within {pct}% of {expected_value}"
        )

@then(u'the simulation Date column should be sorted ascending')
def step_date_column_sorted(context) -> None:
    results = context.sim_results
    assert "Date" in results.columns, "'Date' column not found in simulation results"
    dates = results["Date"].to_list()
    for idx in range(1, len(dates)):
        assert dates[idx] >= dates[idx - 1], (
            f"Date not sorted ascending at index {idx}: {dates[idx - 1]} > {dates[idx]}"
        )


# ===========================================================================
# Then — reproducibility
# ===========================================================================


@then(u'the two result DataFrames should be identical')
def step_two_results_identical(context) -> None:
    _require_imports()
    r1, r2 = context.sim1_results, context.sim2_results
    scenario_cols = [c for c in r1.columns if c != "Date"]
    for col in scenario_cols:
        assert (r1[col] == r2[col]).all(), (
            f"Reproducibility failure: column '{col}' differs between identical seeds"
        )


@then(u'both simulation result DataFrames should be valid')
def step_both_results_valid(context) -> None:
    _require_imports()
    for results in (context.sim1_results, context.sim2_results):
        assert isinstance(results, pl.DataFrame), (
            f"Expected pl.DataFrame, got {type(results)}"
        )
        assert len(results) > 0, "Simulation results DataFrame is empty"
