"""Behave step definitions for portfolio_QWIM and utils_portfolio.

Tests cover:
    - portfolio_QWIM construction from component names and CSV weights
    - Property access: name, num_components, component list, weights DataFrame
    - Weight-sum validation (each row sums to ~1.0)
    - End-to-end portfolio value calculation with initial value
    - Benchmark creation and divergence check

Author:         QWIM Development Team
Version:        0.1.0
Last Modified:  2026-02-28
"""

from __future__ import annotations

import sys
from pathlib import Path

from behave import given, then, when

# ---------------------------------------------------------------------------
# Project root on sys.path — step files live 4 levels below the root:
#   tests/tests_behave/features/steps/<this file>
# ---------------------------------------------------------------------------
_PROJECT_ROOT: Path = Path(__file__).resolve().parents[4]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

_ETF_DATA_PATH: Path = _PROJECT_ROOT / "inputs" / "raw" / "data_ETFs.csv"
_WEIGHTS_PATH: Path = _PROJECT_ROOT / "inputs" / "raw" / "sample_portfolio_weights_ETFs.csv"

# ---------------------------------------------------------------------------
# Module-level import guard
# ---------------------------------------------------------------------------
MODULE_IMPORT_AVAILABLE: bool = True
_import_error_message: str = ""

try:
    import polars as pl
    from src.portfolios.portfolio_QWIM import portfolio_QWIM
    from src.portfolios.utils_portfolio import (
        calculate_portfolio_values,
        create_benchmark_portfolio_values,
        load_portfolio_weights,
        load_sample_etf_data,
    )
except ImportError as _exc:
    MODULE_IMPORT_AVAILABLE = False
    _import_error_message = str(_exc)


def _require_imports() -> None:
    if not MODULE_IMPORT_AVAILABLE:
        raise RuntimeError(
            f"portfolio_QWIM source modules could not be imported: {_import_error_message}"
        )


# ===========================================================================
# Given / When — construction
# ===========================================================================


@when(u'I create a portfolio named "{name}" with components "{components}"')
def step_create_portfolio(context, name: str, components: str) -> None:
    """Create portfolio_QWIM from comma-separated component names."""
    _require_imports()
    names_list = [c.strip() for c in components.split(",")]
    context.portfolio = portfolio_QWIM(
        name_portfolio=name,
        names_components=names_list,
    )


@when(u'I create a second portfolio named "{name}" with components "{components}"')
def step_create_second_portfolio(context, name: str, components: str) -> None:
    """Create a second portfolio_QWIM and store in context.portfolio2."""
    _require_imports()
    names_list = [c.strip() for c in components.split(",")]
    context.portfolio2 = portfolio_QWIM(
        name_portfolio=name,
        names_components=names_list,
    )


@when(u'I load a portfolio from the sample weights CSV file')
def step_load_portfolio_csv(context) -> None:
    """Load portfolio_QWIM from the default sample weights CSV file."""
    _require_imports()
    weights_df = load_portfolio_weights(filepath=_WEIGHTS_PATH)
    context.portfolio = portfolio_QWIM(
        name_portfolio="CSV Portfolio",
        portfolio_weights=weights_df,
    )


@when(u'I load ETF price data from the sample ETF file')
def step_load_etf_data(context) -> None:
    """Load ETF price data from the default sample ETF CSV file."""
    _require_imports()
    context.etf_data = load_sample_etf_data(filepath=_ETF_DATA_PATH)


@when(u'I calculate portfolio values with initial value {initial_value:g}')
def step_calculate_portfolio_values(context, initial_value: float) -> None:
    """Calculate time-series portfolio values."""
    _require_imports()
    context.portfolio_values = calculate_portfolio_values(
        portfolio_obj=context.portfolio,
        price_data=context.etf_data,
        initial_value=float(initial_value),
    )


@when(u'I create a benchmark from the portfolio values')
def step_create_benchmark(context) -> None:
    """Create benchmark portfolio values from calculated portfolio values."""
    _require_imports()
    context.benchmark_values = create_benchmark_portfolio_values(
        context.portfolio_values
    )


# ===========================================================================
# Then — name and component assertions
# ===========================================================================


@then(u'the portfolio name should be "{expected_name}"')
def step_portfolio_name(context, expected_name: str) -> None:
    actual = context.portfolio.get_portfolio_name
    assert actual == expected_name, (
        f"Portfolio name mismatch: expected '{expected_name}', got '{actual}'"
    )


@then(u'the portfolio should have {count:d} components')
def step_portfolio_component_count(context, count: int) -> None:
    actual = context.portfolio.get_num_components
    assert actual == count, (
        f"Component count mismatch: expected {count}, got {actual}"
    )


@then(u'the component list should contain "{component}"')
def step_component_in_list(context, component: str) -> None:
    components = context.portfolio.get_portfolio_components
    assert component in components, (
        f"Component '{component}' not found. Available: {components}"
    )


@then(u'the two portfolios should be distinct objects')
def step_portfolios_distinct(context) -> None:
    assert context.portfolio is not context.portfolio2, (
        "Expected two distinct portfolio objects but got the same instance"
    )


# ===========================================================================
# Then — weights DataFrame assertions
# ===========================================================================


@then(u'the weights DataFrame should have a "{col}" column')
def step_weights_df_has_column(context, col: str) -> None:
    weights = context.portfolio.get_portfolio_weights()
    assert col in weights.columns, (
        f"Column '{col}' not found in weights DataFrame. Available: {weights.columns}"
    )


@then(u'the weights for each row should sum to approximately 1.0')
def step_weights_sum_to_one(context) -> None:
    weights = context.portfolio.get_portfolio_weights()
    numeric_cols = [c for c in weights.columns if c != "Date"]
    for row_idx in range(weights.height):
        row_sum = sum(float(weights[col][row_idx]) for col in numeric_cols)
        assert abs(row_sum - 1.0) <= 1e-4, (
            f"Row {row_idx} weights sum to {row_sum:.6f}; expected ≈ 1.0"
        )


# ===========================================================================
# Then — CSV / general creation assertions
# ===========================================================================


@then(u'the portfolio should be created without error')
def step_portfolio_created(context) -> None:
    assert context.portfolio is not None, "Portfolio was not created (is None)"


@then(u'the portfolio should have at least 1 component')
def step_portfolio_at_least_one_component(context) -> None:
    count = context.portfolio.get_num_components
    assert count >= 1, f"Expected at least 1 component but got {count}"


# ===========================================================================
# Then — portfolio values assertions
# ===========================================================================


@then(u'the portfolio values DataFrame should not be empty')
def step_portfolio_values_not_empty(context) -> None:
    _require_imports()
    df = context.portfolio_values
    assert isinstance(df, pl.DataFrame), (
        f"Expected pl.DataFrame for portfolio values, got {type(df).__name__}"
    )
    assert not df.is_empty(), "Portfolio values DataFrame is empty"


@then(u'the first portfolio value should equal {expected_value:g}')
def step_first_portfolio_value(context, expected_value: float) -> None:
    first = context.portfolio_values["Portfolio_Value"][0]
    assert first is not None, "First Portfolio_Value is None"
    assert abs(float(first) - float(expected_value)) <= 1e-3, (
        f"Expected first Portfolio_Value ≈ {expected_value}, got {first}"
    )


@then(u'all portfolio values should be positive')
def step_all_portfolio_values_positive(context) -> None:
    min_val = context.portfolio_values["Portfolio_Value"].min()
    assert min_val is not None and float(min_val) > 0.0, (
        f"All Portfolio_Value entries must be > 0. Minimum found: {min_val}"
    )


# ===========================================================================
# Then — benchmark assertions
# ===========================================================================


@then(u'the benchmark values should differ from the source values')
def step_benchmark_differs(context) -> None:
    port_list = context.portfolio_values["Portfolio_Value"].to_list()
    # create_benchmark_portfolio_values returns a DataFrame with a 'Value' column
    bench_col = "Value" if "Value" in context.benchmark_values.columns else "Portfolio_Value"
    bench_list = context.benchmark_values[bench_col].to_list()
    assert port_list != bench_list, (
        "Benchmark values are identical to portfolio values — expected divergence"
    )
