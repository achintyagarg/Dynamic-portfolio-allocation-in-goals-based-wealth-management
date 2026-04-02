"""Behave step definitions for num_methods.scenarios.

Tests cover:
    - Scenarios_Distribution (Normal, Student-t, Lognormal) construction
    - Scenarios_CMA construction
    - generate() output shape, column types, null checks
    - Reproducibility with fixed seed
    - validate_scenarios, get_component_series, get_returns_matrix
    - filter_by_date_range
    - Lognormal positivity guarantee

Author:         QWIM Development Team
Version:        0.1.0
Last Modified:  2026-02-28
"""

from __future__ import annotations

import datetime as dt
import sys
from pathlib import Path

import numpy as np
from behave import given, then, when

# ---------------------------------------------------------------------------
# Project root on sys.path
# ---------------------------------------------------------------------------
_PROJECT_ROOT: Path = Path(__file__).resolve().parents[4]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# ---------------------------------------------------------------------------
# Module-level import guard
# ---------------------------------------------------------------------------
MODULE_IMPORT_AVAILABLE: bool = True
_import_error_message: str = ""

try:
    import polars as pl
    from src.num_methods.scenarios.scenarios_CMA import Scenarios_CMA
    from src.num_methods.scenarios.scenarios_distrib import (
        Distribution_Type,
        Scenarios_Distribution,
    )
except ImportError as _exc:
    MODULE_IMPORT_AVAILABLE = False
    _import_error_message = str(_exc)

# ---------------------------------------------------------------------------
# Shared test constants
# ---------------------------------------------------------------------------
_SEED: int = 42
_START_DATE: dt.date = dt.date(2024, 1, 1)
_COMPONENTS: list[str] = ["A", "B", "C"]
_K: int = 3

_MEAN = np.zeros(_K, dtype=np.float64)
_COV = np.array(
    [[1e-4, 5e-5, 2e-5], [5e-5, 1e-4, 3e-5], [2e-5, 3e-5, 1e-4]],
    dtype=np.float64,
)
_LOGNORMAL_MEAN = np.array([1.0005, 1.0004, 1.0002], dtype=np.float64)

_CMA_CORR = np.array(
    [[1.0, 0.5, 0.2], [0.5, 1.0, 0.3], [0.2, 0.3, 1.0]],
    dtype=np.float64,
)
_CMA_RET = np.array([0.06, 0.07, 0.05], dtype=np.float64)
_CMA_VOL = np.array([0.10, 0.12, 0.08], dtype=np.float64)


def _require_imports() -> None:
    if not MODULE_IMPORT_AVAILABLE:
        raise RuntimeError(
            f"num_methods.scenarios modules not importable: {_import_error_message}"
        )


# ===========================================================================
# When — construction
# ===========================================================================


@when(
    u"I create a Normal distribution scenario with "
    u"{num_components:d} components and {num_days:d} days"
)
def step_create_normal_scenario(
    context, num_components: int, num_days: int
) -> None:
    """Create a Normal Scenarios_Distribution and store in context."""
    _require_imports()
    comps = _COMPONENTS[:num_components]
    cov = _COV[:num_components, :num_components].copy()
    context.scenarios_obj = Scenarios_Distribution(
        names_components=comps,
        distribution_type=Distribution_Type.NORMAL,
        mean_returns=_MEAN[:num_components],
        covariance_matrix=cov,
        start_date=_START_DATE,
        num_days=num_days,
        random_seed=_SEED,
    )
    context.num_components = num_components
    context.num_days = num_days
    context.component_names = comps


@when(
    u"I create a Student-t distribution scenario with "
    u"{num_components:d} components and {num_days:d} days"
)
def step_create_student_t_scenario(
    context, num_components: int, num_days: int
) -> None:
    """Create a Student-t Scenarios_Distribution and store in context."""
    _require_imports()
    comps = _COMPONENTS[:num_components]
    cov = _COV[:num_components, :num_components].copy()
    context.scenarios_obj = Scenarios_Distribution(
        names_components=comps,
        distribution_type=Distribution_Type.STUDENT_T,
        mean_returns=_MEAN[:num_components],
        covariance_matrix=cov,
        degrees_of_freedom=5.0,
        start_date=_START_DATE,
        num_days=num_days,
        random_seed=_SEED,
    )
    context.num_components = num_components
    context.num_days = num_days
    context.component_names = comps


@when(
    u"I create a Lognormal distribution scenario with "
    u"{num_components:d} components and {num_days:d} days"
)
def step_create_lognormal_scenario(
    context, num_components: int, num_days: int
) -> None:
    """Create a Lognormal Scenarios_Distribution and store in context."""
    _require_imports()
    comps = _COMPONENTS[:num_components]
    cov = _COV[:num_components, :num_components].copy()
    context.scenarios_obj = Scenarios_Distribution(
        names_components=comps,
        distribution_type=Distribution_Type.LOGNORMAL,
        mean_returns=_LOGNORMAL_MEAN[:num_components],
        covariance_matrix=cov,
        start_date=_START_DATE,
        num_days=num_days,
        random_seed=_SEED,
    )
    context.num_components = num_components
    context.num_days = num_days
    context.component_names = comps


@when(
    u"I create a CMA scenario with "
    u"{num_components:d} asset classes and {num_days:d} days"
)
def step_create_cma_scenario(
    context, num_components: int, num_days: int
) -> None:
    """Create a Scenarios_CMA and store in context."""
    _require_imports()
    comps = _COMPONENTS[:num_components]
    corr = _CMA_CORR[:num_components, :num_components].copy()
    ret = _CMA_RET[:num_components]
    vol = _CMA_VOL[:num_components]
    context.scenarios_obj = Scenarios_CMA(
        names_asset_classes=comps,
        expected_returns_annual=ret,
        expected_vols_annual=vol,
        correlation_matrix=corr,
        start_date=_START_DATE,
        num_days=num_days,
        random_seed=_SEED,
    )
    context.num_components = num_components
    context.num_days = num_days
    context.component_names = comps


@when(u"I call generate on the scenario")
def step_call_generate(context) -> None:
    """Call generate() on the stored scenario object."""
    _require_imports()
    context.generated_df = context.scenarios_obj.generate()


@when(u"I create two Normal distribution scenarios with the same seed")
def step_create_two_same_seed(context) -> None:
    """Create two Normal scenarios sharing seed=42."""
    _require_imports()
    context.scenario_a = Scenarios_Distribution(
        names_components=_COMPONENTS,
        covariance_matrix=_COV,
        start_date=_START_DATE,
        num_days=30,
        random_seed=42,
    )
    context.scenario_b = Scenarios_Distribution(
        names_components=_COMPONENTS,
        covariance_matrix=_COV,
        start_date=_START_DATE,
        num_days=30,
        random_seed=42,
    )
    context.df_a = context.scenario_a.generate()
    context.df_b = context.scenario_b.generate()
    context.component_names = _COMPONENTS


@when(u"I create two Normal distribution scenarios with different seeds")
def step_create_two_different_seeds(context) -> None:
    """Create two Normal scenarios with different seeds."""
    _require_imports()
    context.scenario_a = Scenarios_Distribution(
        names_components=_COMPONENTS,
        covariance_matrix=_COV,
        start_date=_START_DATE,
        num_days=30,
        random_seed=0,
    )
    context.scenario_b = Scenarios_Distribution(
        names_components=_COMPONENTS,
        covariance_matrix=_COV,
        start_date=_START_DATE,
        num_days=30,
        random_seed=999,
    )
    context.df_a = context.scenario_a.generate()
    context.df_b = context.scenario_b.generate()
    context.component_names = _COMPONENTS


# ===========================================================================
# Then — construction
# ===========================================================================


@then(u"the scenario object should be created without error")
def step_scenario_created_ok(context) -> None:
    """Assert scenario object is not None."""
    assert context.scenarios_obj is not None, (
        "Scenario object is None — construction raised an error"
    )


# ===========================================================================
# Then — generate() assertions
# ===========================================================================


@then(u"the generated DataFrame should be a Polars DataFrame")
def step_generated_is_polars(context) -> None:
    """generated_df should be an instance of pl.DataFrame."""
    assert isinstance(context.generated_df, pl.DataFrame), (
        f"Expected pl.DataFrame, got {type(context.generated_df).__name__}"
    )


@then(u"the generated DataFrame should have {count:d} rows")
def step_generated_row_count(context, count: int) -> None:
    """Row count should equal num_days."""
    actual = len(context.generated_df)
    assert actual == count, (
        f"Expected {count} rows, got {actual}"
    )


@then(u'the generated DataFrame should have a "{col}" column')
def step_generated_has_column(context, col: str) -> None:
    """Specified column should be present."""
    assert col in context.generated_df.columns, (
        f"Column '{col}' not found. Available: {context.generated_df.columns}"
    )


@then(u"all component columns should be of type Float64")
def step_component_columns_float64(context) -> None:
    """All non-Date component columns should be pl.Float64."""
    for comp in context.component_names:
        dtype = context.generated_df.schema[comp]
        assert dtype == pl.Float64, (
            f"Column '{comp}' has dtype {dtype}, expected Float64"
        )


@then(u"the generated DataFrame should contain no null values")
def step_no_nulls(context) -> None:
    """No nulls should appear in any column."""
    null_count = context.generated_df.null_count().sum_horizontal()[0]
    assert null_count == 0, (
        f"Found {null_count} null value(s) in generated DataFrame"
    )


# ===========================================================================
# Then — reproducibility
# ===========================================================================


@then(u"the two generated DataFrames should be identical")
def step_dfs_identical(context) -> None:
    """Both DataFrames share identical values for all component columns."""
    for comp in context.component_names:
        arr_a = context.df_a[comp].to_numpy()
        arr_b = context.df_b[comp].to_numpy()
        np.testing.assert_array_equal(
            arr_a,
            arr_b,
            err_msg=f"Column '{comp}' differs between same-seed scenarios",
        )


@then(u"the two generated DataFrames should differ")
def step_dfs_differ(context) -> None:
    """At least one column should differ between different-seed scenarios."""
    comp = context.component_names[0]
    arr_a = context.df_a[comp].to_numpy()
    arr_b = context.df_b[comp].to_numpy()
    assert not np.array_equal(arr_a, arr_b), (
        f"Column '{comp}' is identical for different seeds — expected divergence"
    )


# ===========================================================================
# Then — base-class utilities
# ===========================================================================


@then(u"validate_scenarios should pass without error")
def step_validate_pass(context) -> None:
    """validate_scenarios() should not raise."""
    context.scenarios_obj.validate_scenarios()


@then(
    u"get_component_series for the first component should have "
    u"{count:d} rows"
)
def step_component_series_length(context, count: int) -> None:
    """get_component_series for the first component should have correct length."""
    first_comp = context.component_names[0]
    series = context.scenarios_obj.get_component_series(first_comp)
    assert len(series) == count, (
        f"Expected {count} rows in component series, got {len(series)}"
    )


@then(
    u"get_returns_matrix should have shape {rows:d} by {cols:d}"
)
def step_returns_matrix_shape(context, rows: int, cols: int) -> None:
    """get_returns_matrix() should return an (rows, cols) numpy array."""
    mat = context.scenarios_obj.get_returns_matrix()
    assert mat.shape == (rows, cols), (
        f"Returns matrix shape mismatch: expected ({rows}, {cols}), got {mat.shape}"
    )


@then(u"filtering to half the date range should reduce the row count")
def step_filter_reduces_rows(context) -> None:
    """filter_by_date_range with a mid-range end reduces the DataFrame."""
    start, end = context.scenarios_obj.get_date_range()
    mid = start + dt.timedelta(days=(context.num_days // 2) - 1)
    filtered = context.scenarios_obj.filter_by_date_range(start, mid)
    assert len(filtered) < context.num_days, (
        f"Filter should reduce rows below {context.num_days}, got {len(filtered)}"
    )


# ===========================================================================
# Then — lognormal
# ===========================================================================


@then(u"all component values in the generated DataFrame should be positive")
def step_all_values_positive(context) -> None:
    """All component columns should be strictly positive (lognormal guarantee)."""
    for comp in context.component_names:
        min_val = float(context.generated_df[comp].min())
        assert min_val > 0, (
            f"Column '{comp}' has non-positive value {min_val:.6e}"
        )
