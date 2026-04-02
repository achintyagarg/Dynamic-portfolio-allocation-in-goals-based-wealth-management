"""Behave step definitions for portfolio_optimization feature.

Covers:
  - portfolio_optimization_type enum membership and classmethods
  - portfolio_optimization_feature_type enum membership and classmethods
  - End-to-end: enum-driven skfolio optimization dispatching
  - Regression: benchmark_pandas unbound-variable fix
  - Regression: aenum → stdlib enum migration
"""

from __future__ import annotations

from datetime import datetime

import numpy as np
import polars as pl
from behave import given, then, when

from src.models.portfolio_optimization.utils_portfolio_optimization import (
    portfolio_optimization_feature_type,
    portfolio_optimization_type,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _build_returns(n_days: int, n_assets: int, seed: int = 42) -> pl.DataFrame:
    """Build synthetic returns with ``n_assets`` columns."""
    asset_names = ["VTI", "VXUS", "AGG", "VNQ", "GLD", "TLT", "GDX", "XLE"][:n_assets]
    rng = np.random.default_rng(seed)
    dates = pl.date_range(
        start=datetime(2023, 1, 2),
        end=datetime(2024, 12, 31),
        interval="1d",
        eager=True,
    )[:n_days]
    data: dict[str, object] = {"Date": dates}
    for asset in asset_names:
        data[asset] = rng.normal(0.0004, 0.015, n_days).tolist()
    return pl.DataFrame(data)


# =============================================================================
# portfolio_optimization_type steps
# =============================================================================


@when("I inspect the portfolio_optimization_type enum")
def step_inspect_opt_type_enum(context) -> None:
    context.opt_type_members = list(portfolio_optimization_type)


@then("the enum should have {count:d} members")
def step_enum_has_n_members(context, count: int) -> None:
    assert len(context.opt_type_members) == count, (
        f"Expected {count} members, got {len(context.opt_type_members)}"
    )


@then("every member value should equal its name")
def step_value_equals_name(context) -> None:
    for member in context.opt_type_members:
        assert member.value == member.name, (
            f"Member {member.name}: value={member.value!r} != name={member.name!r}"
        )


@when("I call get_basic_methods on portfolio_optimization_type")
def step_get_basic_methods(context) -> None:
    context.method_result = portfolio_optimization_type.get_basic_methods()


@when("I call get_convex_methods on portfolio_optimization_type")
def step_get_convex_methods(context) -> None:
    context.method_result = portfolio_optimization_type.get_convex_methods()


@when("I call get_clustering_methods on portfolio_optimization_type")
def step_get_clustering_methods(context) -> None:
    context.method_result = portfolio_optimization_type.get_clustering_methods()


@when("I call get_ensemble_methods on portfolio_optimization_type")
def step_get_ensemble_methods(context) -> None:
    context.method_result = portfolio_optimization_type.get_ensemble_methods()


@then("I should get {count:d} methods")
def step_got_n_methods(context, count: int) -> None:
    assert len(context.method_result) == count, (
        f"Expected {count} methods, got {len(context.method_result)}: {context.method_result}"
    )


@then("I should get {count:d} method")
def step_got_1_method(context, count: int) -> None:
    assert len(context.method_result) == count, (
        f"Expected {count} method, got {len(context.method_result)}"
    )


@then('all methods should have names starting with "{prefix}"')
def step_all_names_start_with(context, prefix: str) -> None:
    for m in context.method_result:
        assert m.name.startswith(prefix), (
            f"Member {m.name!r} does not start with {prefix!r}"
        )


@when("I combine basic, convex, clustering, and ensemble methods")
def step_combine_all_groups(context) -> None:
    context.combined = (
        portfolio_optimization_type.get_basic_methods()
        + portfolio_optimization_type.get_convex_methods()
        + portfolio_optimization_type.get_clustering_methods()
        + portfolio_optimization_type.get_ensemble_methods()
    )


@then("the combined set should equal all 13 portfolio_optimization_type members")
def step_combined_covers_all(context) -> None:
    assert set(context.combined) == set(portfolio_optimization_type), (
        f"Combined groups differ from full enum set:\n"
        f"  Extra: {set(context.combined) - set(portfolio_optimization_type)}\n"
        f"  Missing: {set(portfolio_optimization_type) - set(context.combined)}"
    )


@when('I access portfolio_optimization_type by name "{name}"')
def step_subscript_by_name(context, name: str) -> None:
    context.subscript_result = portfolio_optimization_type[name]


@then("I should get the BASIC_EQUAL_WEIGHTED member")
def step_got_basic_equal_weighted(context) -> None:
    assert context.subscript_result is portfolio_optimization_type.BASIC_EQUAL_WEIGHTED


# =============================================================================
# portfolio_optimization_feature_type steps
# =============================================================================


@when("I inspect the portfolio_optimization_feature_type enum")
def step_inspect_feature_enum(context) -> None:
    context.feature_members = list(portfolio_optimization_feature_type)


@then("the feature enum should have {count:d} members")
def step_feature_enum_n_members(context, count: int) -> None:
    assert len(context.feature_members) == count, (
        f"Expected {count} feature members, got {len(context.feature_members)}"
    )


@when("I call get_objectives on portfolio_optimization_feature_type")
def step_get_objectives(context) -> None:
    context.feature_result = portfolio_optimization_feature_type.get_objectives()


@when("I call get_all_constraints on portfolio_optimization_feature_type")
def step_get_all_constraints(context) -> None:
    context.feature_result = portfolio_optimization_feature_type.get_all_constraints()


@when("I call get_convex_features on portfolio_optimization_feature_type")
def step_get_convex_features(context) -> None:
    context.feature_result = portfolio_optimization_feature_type.get_convex_features()


@then("I should get {count:d} features")
def step_got_n_features(context, count: int) -> None:
    assert len(context.feature_result) == count, (
        f"Expected {count} features, got {len(context.feature_result)}: {context.feature_result}"
    )


@then("the result should include CONSTRAINTS_CUSTOM")
def step_result_includes_custom(context) -> None:
    assert portfolio_optimization_feature_type.CONSTRAINTS_CUSTOM in context.feature_result


@when("I get the integer features")
def step_get_integer_features(context) -> None:
    context.integer_features = set(
        portfolio_optimization_feature_type.get_integer_features()
    )


@when("I get the convex features")
def step_get_convex_features_set(context) -> None:
    context.convex_features = set(
        portfolio_optimization_feature_type.get_convex_features()
    )


@then("integer features and convex features should be disjoint")
def step_integer_convex_disjoint(context) -> None:
    overlap = context.integer_features & context.convex_features
    assert not overlap, (
        f"Integer features found in convex features: {overlap}"
    )


# =============================================================================
# End-to-end optimization steps
# =============================================================================


@given("I have synthetic daily-returns data for {n_assets:d} assets and {n_days:d} days")
def step_given_returns_data(context, n_assets: int, n_days: int) -> None:
    context.returns_data = _build_returns(n_days=n_days, n_assets=n_assets, seed=42)
    context.n_assets = n_assets


@when('I run calc_skfolio_optimization_basic with "{opt_type}"')
def step_run_basic_optimization(context, opt_type: str) -> None:
    from src.models.portfolio_optimization.pkg_skfolio import calc_skfolio_optimization_basic

    context.portfolio = calc_skfolio_optimization_basic(
        returns_data=context.returns_data,
        optimization_type=opt_type,
        portfolio_name=f"Behave Test {opt_type}",
    )


@when('I run calc_skfolio_optimization_convex with "{opt_type}"')
def step_run_convex_optimization(context, opt_type: str) -> None:
    from src.models.portfolio_optimization.pkg_skfolio import calc_skfolio_optimization_convex

    context.portfolio = calc_skfolio_optimization_convex(
        returns_data=context.returns_data,
        optimization_type=opt_type,
        portfolio_name=f"Behave Convex Test {opt_type}",
    )


@when("I run basic optimization with the enum member BASIC_EQUAL_WEIGHTED")
def step_run_basic_with_enum(context) -> None:
    from src.models.portfolio_optimization.pkg_skfolio import calc_skfolio_optimization_basic

    context.portfolio_enum = calc_skfolio_optimization_basic(
        returns_data=context.returns_data,
        optimization_type=portfolio_optimization_type.BASIC_EQUAL_WEIGHTED,
        optimization_date="2024-01-01",
    )


@when('I run basic optimization with the string "BASIC_EQUAL_WEIGHTED"')
def step_run_basic_with_string(context) -> None:
    from src.models.portfolio_optimization.pkg_skfolio import calc_skfolio_optimization_basic

    context.portfolio_str = calc_skfolio_optimization_basic(
        returns_data=context.returns_data,
        optimization_type="BASIC_EQUAL_WEIGHTED",
        optimization_date="2024-01-01",
    )


@then("I should receive a portfolio_QWIM object")
def step_received_portfolio_qwim(context) -> None:
    from src.portfolios.portfolio_QWIM import portfolio_QWIM

    assert isinstance(context.portfolio, portfolio_QWIM), (
        f"Expected portfolio_QWIM, got {type(context.portfolio)}"
    )


@then("the portfolio should have {n:d} components")
def step_portfolio_has_n_components(context, n: int) -> None:
    assert context.portfolio.get_num_components == n, (
        f"Expected {n} components, got {context.portfolio.get_num_components}"
    )


@then("all weights should sum to approximately 1.0")
def step_weights_sum_to_one(context) -> None:
    from src.portfolios.portfolio_QWIM import portfolio_QWIM

    p = context.portfolio
    assert isinstance(p, portfolio_QWIM)
    weights_df = p.get_portfolio_weights()
    weight_sum = sum(
        float(weights_df[asset][0]) for asset in p.get_portfolio_components
    )
    assert abs(weight_sum - 1.0) < 1e-5, (
        f"Weights sum to {weight_sum:.8f}, expected 1.0"
    )


@then("the two portfolios should have identical weights")
def step_two_portfolios_identical_weights(context) -> None:
    p_enum = context.portfolio_enum
    p_str = context.portfolio_str

    w_enum = p_enum.get_portfolio_weights()
    w_str = p_str.get_portfolio_weights()

    components = p_enum.get_portfolio_components
    for asset in components:
        diff = abs(float(w_enum[asset][0]) - float(w_str[asset][0]))
        assert diff < 1e-10, (
            f"Enum vs string produced different weight for {asset}: diff={diff:.2e}"
        )
