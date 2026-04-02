"""Behave step definitions for num_methods.covariance.

Tests cover:
    - covariance_matrix construction with multiple estimators
    - Shape, component count, and observation count properties
    - validate_covariance_matrix pass/fail
    - Diagonal positivity, symmetry
    - Correlation matrix: diagonal==1, bounds, symmetry
    - Component accessor agreement with full matrix
    - distance_estimator_type classmethods

Author:         QWIM Development Team
Version:        0.1.0
Last Modified:  2026-02-28
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from behave import given, then, when

# ---------------------------------------------------------------------------
# Project root on sys.path — step files live 4 levels below the root:
#   tests/tests_behave/features/steps/<this file>
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
    from src.num_methods.covariance.utils_cov_corr import (
        covariance_estimator,
        covariance_matrix,
        distance_estimator_type,
    )
except ImportError as _exc:
    MODULE_IMPORT_AVAILABLE = False
    _import_error_message = str(_exc)

# Component names used throughout
_COMPONENTS_3 = ["AAPL", "MSFT", "GOOG"]


def _require_imports() -> None:
    if not MODULE_IMPORT_AVAILABLE:
        raise RuntimeError(
            f"num_methods.covariance modules not importable: {_import_error_message}"
        )


def _build_returns_df(num_components: int, num_obs: int) -> "pl.DataFrame":
    """Build a synthetic returns DataFrame with fixed seed."""
    rng = np.random.default_rng(42)
    components = _COMPONENTS_3[:num_components]
    factor = rng.standard_normal(num_obs) * 0.01
    noise = rng.standard_normal((num_obs, num_components)) * 0.005
    raw = factor[:, np.newaxis] + noise

    dates = pl.date_range(
        pl.date(2023, 1, 1),
        pl.date(2023, 1, 1) + pl.duration(days=num_obs - 1),
        interval="1d",
        eager=True,
    )
    return pl.DataFrame(
        {"Date": dates, **{c: raw[:, i].tolist() for i, c in enumerate(components)}}
    )


# ===========================================================================
# Given — data setup
# ===========================================================================


@given(
    u"a returns DataFrame with {num_components:d} components "
    u"and {num_obs:d} observations"
)
def step_given_returns_df(context, num_components: int, num_obs: int) -> None:
    """Build and store synthetic returns DataFrame."""
    _require_imports()
    context.returns_df = _build_returns_df(num_components, num_obs)
    context.num_components = num_components
    context.num_obs = num_obs


@given(
    u"a returns DataFrame with {num_obs:d} observations "
    u"and {num_components:d} components"
)
def step_given_returns_df_alt(context, num_obs: int, num_components: int) -> None:
    """Alias with swapped argument order for natural-language variety."""
    step_given_returns_df(context, num_components, num_obs)


# ===========================================================================
# When — build covariance matrix
# ===========================================================================


@when(u'I build a covariance matrix using the "{estimator_name}" estimator')
def step_build_covariance_matrix(context, estimator_name: str) -> None:
    """Construct a covariance_matrix with the named estimator."""
    _require_imports()
    estimator = covariance_estimator[estimator_name]
    context.cov_matrix_obj = covariance_matrix(
        data_returns=context.returns_df,
        estimator=estimator,
    )


@when(u"I compute the correlation matrix")
def step_compute_correlation_matrix(context) -> None:
    """Compute and store the correlation matrix from the covariance object."""
    _require_imports()
    context.correlation_matrix = context.cov_matrix_obj.get_correlation_matrix()


@when(u"I call get_correlation_based on distance_estimator_type")
def step_call_get_correlation_based(context) -> None:
    """Call distance_estimator_type.get_correlation_based()."""
    _require_imports()
    context.distance_estimator_result = distance_estimator_type.get_correlation_based()


@when(u"I call get_rank_based on distance_estimator_type")
def step_call_get_rank_based(context) -> None:
    """Call distance_estimator_type.get_rank_based()."""
    _require_imports()
    context.distance_estimator_result = distance_estimator_type.get_rank_based()


@when(u"I collect all category members of distance_estimator_type")
def step_collect_all_distance_estimator_members(context) -> None:
    """Aggregate all categorised distance estimator members."""
    _require_imports()
    context.all_categorised = set(
        distance_estimator_type.get_correlation_based()
        + distance_estimator_type.get_covariance_based()
        + distance_estimator_type.get_rank_based()
        + distance_estimator_type.get_information_theoretic()
    )
    context.all_members = set(distance_estimator_type)


# ===========================================================================
# Then — construction assertions
# ===========================================================================


@then(u"the covariance matrix object should be created without error")
def step_cov_matrix_created(context) -> None:
    """Assert the covariance_matrix object is not None."""
    assert context.cov_matrix_obj is not None, (
        "covariance_matrix object is None — construction raised an error"
    )


# ===========================================================================
# Then — shape and property assertions
# ===========================================================================


@then(u"the covariance matrix should have shape {rows:d} by {cols:d}")
def step_cov_matrix_shape(context, rows: int, cols: int) -> None:
    """Assert covariance matrix numpy shape."""
    mat = context.cov_matrix_obj.get_covariance_matrix()
    assert mat.shape == (rows, cols), (
        f"Shape mismatch: expected ({rows}, {cols}), got {mat.shape}"
    )


@then(u"the covariance matrix should report {count:d} components")
def step_cov_num_components(context, count: int) -> None:
    """Assert m_num_components."""
    actual = context.cov_matrix_obj.m_num_components
    assert actual == count, (
        f"Expected {count} components, got {actual}"
    )


@then(u"the covariance matrix should report {count:d} observations")
def step_cov_num_observations(context, count: int) -> None:
    """Assert m_num_observations."""
    actual = context.cov_matrix_obj.m_num_observations
    assert actual == count, (
        f"Expected {count} observations, got {actual}"
    )


# ===========================================================================
# Then — validation
# ===========================================================================


@then(u"the validate_covariance_matrix method should pass without error")
def step_validate_passes(context) -> None:
    """validate_covariance_matrix should not raise."""
    context.cov_matrix_obj.validate_covariance_matrix()


@then(u"all diagonal elements of the covariance matrix should be positive")
def step_cov_diagonal_positive(context) -> None:
    """All variances (diagonal) must be strictly positive."""
    mat = context.cov_matrix_obj.get_covariance_matrix()
    diagonal = np.diag(mat)
    assert np.all(diagonal > 0), (
        f"Non-positive diagonal found: {diagonal.tolist()}"
    )


@then(u"the covariance matrix should be symmetric")
def step_cov_symmetric(context) -> None:
    """Covariance matrix must be symmetric (max asymmetry < 1e-8)."""
    mat = context.cov_matrix_obj.get_covariance_matrix()
    max_asym = float(np.max(np.abs(mat - mat.T)))
    assert max_asym < 1e-8, (
        f"Covariance matrix not symmetric; max asymmetry = {max_asym:.2e}"
    )


# ===========================================================================
# Then — correlation matrix assertions
# ===========================================================================


@then(u"all diagonal elements of the correlation matrix should equal 1.0")
def step_corr_diagonal_one(context) -> None:
    """Correlation diagonal must all be 1.0."""
    diag = np.diag(context.correlation_matrix)
    np.testing.assert_allclose(
        diag,
        np.ones_like(diag),
        atol=1e-10,
        err_msg="Correlation diagonal not all 1.0",
    )


@then(u"all correlation values should be between -1.0 and 1.0")
def step_corr_in_bounds(context) -> None:
    """All correlation values must lie in [-1, 1]."""
    mat = context.correlation_matrix
    assert np.all(mat >= -1.0 - 1e-8) and np.all(mat <= 1.0 + 1e-8), (
        f"Correlation out of bounds: min={mat.min():.6f}, max={mat.max():.6f}"
    )


@then(u"the correlation matrix should be symmetric")
def step_corr_symmetric(context) -> None:
    """Correlation matrix must be symmetric."""
    mat = context.correlation_matrix
    max_asym = float(np.max(np.abs(mat - mat.T)))
    assert max_asym < 1e-10, (
        f"Correlation matrix not symmetric; max asymmetry = {max_asym:.2e}"
    )


# ===========================================================================
# Then — component accessor assertions
# ===========================================================================


@then(
    u'the variance accessor for component "AAPL" should match the matrix diagonal'
)
def step_variance_accessor_aapl(context) -> None:
    """get_component_variance('AAPL') should match diagonal[0]."""
    mat = context.cov_matrix_obj.get_covariance_matrix()
    expected = float(mat[0, 0])
    actual = context.cov_matrix_obj.get_component_variance("AAPL")
    assert abs(actual - expected) < 1e-12, (
        f"Variance mismatch for AAPL: expected {expected}, got {actual}"
    )


@then(
    u'the covariance between "AAPL" and "MSFT" should match the matrix value'
)
def step_covariance_accessor_aapl_msft(context) -> None:
    """get_component_covariance('AAPL','MSFT') should match matrix[0,1]."""
    mat = context.cov_matrix_obj.get_covariance_matrix()
    expected = float(mat[0, 1])
    actual = context.cov_matrix_obj.get_component_covariance("AAPL", "MSFT")
    assert abs(actual - expected) < 1e-12, (
        f"Covariance(AAPL,MSFT) mismatch: expected {expected}, got {actual}"
    )


# ===========================================================================
# Then — distance_estimator_type assertions
# ===========================================================================


@then(u"the distance estimator result should be a non-empty list")
def step_result_non_empty_list(context) -> None:
    """distance_estimator_type category method returns a non-empty list."""
    result = context.distance_estimator_result
    assert isinstance(result, list), f"Expected list, got {type(result).__name__}"
    assert len(result) >= 1, "Expected at least one member in the list"


@then(u"all distance_estimator_type members should be covered")
def step_all_distance_estimator_covered(context) -> None:
    """All enum members appear in at least one category."""
    missing = context.all_members - context.all_categorised
    assert len(missing) == 0, (
        f"Uncategorised distance estimators: {[m.name for m in missing]}"
    )
