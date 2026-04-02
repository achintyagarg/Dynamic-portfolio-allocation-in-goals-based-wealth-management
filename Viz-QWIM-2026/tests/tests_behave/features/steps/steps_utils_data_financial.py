"""Behave step definitions for utils_data_financial feature.

Covers:
  - expected_returns_estimator_type: member count, group classmethods, disjointness, coverage
  - prior_estimator_type: member count, group classmethods, disjointness, coverage
  - distribution_estimator_type: member count, group classmethods, coverage

Author:         QWIM Development Team
Version:        0.1.0
Last Modified:  2026-03-01
"""

from __future__ import annotations

import io
import sys
from pathlib import Path

from behave import then, when

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
    from src.utils.data_utils.utils_data_financial import (
        distribution_estimator_type,
        expected_returns_estimator_type,
        prior_estimator_type,
    )
except Exception as _exc:
    MODULE_IMPORT_AVAILABLE = False
    _import_error_message = str(_exc)


def _require_imports() -> None:
    if not MODULE_IMPORT_AVAILABLE:
        raise RuntimeError(
            f"utils_data_financial could not be imported: {_import_error_message}"
        )


# ---------------------------------------------------------------------------
# Helpers: collect all classmethod groups for each enum
# ---------------------------------------------------------------------------


def _all_groups_expected_returns() -> list[list]:
    """Return all classmethod groups for expected_returns_estimator_type."""
    return [
        expected_returns_estimator_type.get_historical_methods(),
        expected_returns_estimator_type.get_model_based_methods(),
    ]


def _all_groups_prior() -> list[list]:
    """Return all classmethod groups for prior_estimator_type."""
    return [
        prior_estimator_type.get_data_driven(),
        prior_estimator_type.get_equilibrium_based(),
        prior_estimator_type.get_factor_based(),
        prior_estimator_type.get_scenario_based(),
        prior_estimator_type.get_probabilistic(),
        prior_estimator_type.get_bayesian_methods(),
        prior_estimator_type.get_view_incorporation_methods(),
    ]


def _all_groups_distribution() -> list[list]:
    """Return all classmethod groups for distribution_estimator_type."""
    return [
        distribution_estimator_type.get_univariate(),
        distribution_estimator_type.get_bivariate_copulas(),
        distribution_estimator_type.get_multivariate_copulas(),
        distribution_estimator_type.get_fat_tailed_distributions(),
        distribution_estimator_type.get_flexible_distributions(),
    ]


# ===========================================================================
# expected_returns_estimator_type steps
# ===========================================================================


@when("I inspect the expected_returns_estimator_type enum")
def step_inspect_expected_returns_enum(context) -> None:
    _require_imports()
    context.active_enum = expected_returns_estimator_type
    context.enum_members = list(expected_returns_estimator_type)


@when("I call get_historical_methods on expected_returns_estimator_type")
def step_get_historical_methods(context) -> None:
    _require_imports()
    context.result_a = expected_returns_estimator_type.get_historical_methods()


@when("I call get_model_based_methods on expected_returns_estimator_type")
def step_get_model_based_methods(context) -> None:
    _require_imports()
    context.result_b = expected_returns_estimator_type.get_model_based_methods()


@when("I retrieve all classmethod groups for expected_returns_estimator_type")
def step_get_all_groups_expected_returns(context) -> None:
    _require_imports()
    context.all_groups = _all_groups_expected_returns()
    context.active_enum = expected_returns_estimator_type


# ===========================================================================
# prior_estimator_type steps
# ===========================================================================


@when("I inspect the prior_estimator_type enum")
def step_inspect_prior_enum(context) -> None:
    _require_imports()
    context.active_enum = prior_estimator_type
    context.enum_members = list(prior_estimator_type)


@when("I call get_data_driven on prior_estimator_type")
def step_get_data_driven(context) -> None:
    _require_imports()
    context.result_a = prior_estimator_type.get_data_driven()


@when("I call get_equilibrium_based on prior_estimator_type")
def step_get_equilibrium_based(context) -> None:
    _require_imports()
    context.result_b = prior_estimator_type.get_equilibrium_based()


@when("I retrieve all classmethod groups for prior_estimator_type")
def step_get_all_groups_prior(context) -> None:
    _require_imports()
    context.all_groups = _all_groups_prior()
    context.active_enum = prior_estimator_type


# ===========================================================================
# distribution_estimator_type steps
# ===========================================================================


@when("I inspect the distribution_estimator_type enum")
def step_inspect_distribution_enum(context) -> None:
    _require_imports()
    context.active_enum = distribution_estimator_type
    context.enum_members = list(distribution_estimator_type)


@when("I call get_univariate on distribution_estimator_type")
def step_get_univariate(context) -> None:
    _require_imports()
    context.result_a = distribution_estimator_type.get_univariate()


@when("I call get_fat_tailed_distributions on distribution_estimator_type")
def step_get_fat_tailed(context) -> None:
    _require_imports()
    context.result_a = distribution_estimator_type.get_fat_tailed_distributions()


@when("I call get_simulation_ready on distribution_estimator_type")
def step_get_simulation_ready(context) -> None:
    _require_imports()
    context.result_a = distribution_estimator_type.get_simulation_ready()


@when("I retrieve all classmethod groups for distribution_estimator_type")
def step_get_all_groups_distribution(context) -> None:
    _require_imports()
    context.all_groups = _all_groups_distribution()
    context.active_enum = distribution_estimator_type


# ===========================================================================
# Shared Then steps
# ===========================================================================


@then("the enum should have exactly {count:d} members")
def step_enum_has_exactly_n_members(context, count: int) -> None:
    members = list(context.active_enum)
    assert len(members) == count, (
        f"Expected {count} members, got {len(members)}: {members}"
    )


@then("the result should be a non-empty list")
def step_result_is_non_empty(context) -> None:
    # result may be stored as result_a or as generic result
    result = getattr(context, "result_a", getattr(context, "result", None))
    assert result is not None, "No result stored in context"
    assert len(result) > 0, "Expected non-empty list but got empty"


@then("the two result lists should have no common elements")
def step_two_results_are_disjoint(context) -> None:
    set_a = set(context.result_a)
    set_b = set(context.result_b)
    common = set_a & set_b
    assert len(common) == 0, (
        f"Expected disjoint groups but found common elements: {common}"
    )


@then("every enum member should appear in the union of all groups")
def step_all_members_covered(context) -> None:
    union: set = set()
    for group in context.all_groups:
        union.update(group)
    for member in list(context.active_enum):
        assert member in union, (
            f"Member {member!r} does not appear in any classmethod group"
        )
