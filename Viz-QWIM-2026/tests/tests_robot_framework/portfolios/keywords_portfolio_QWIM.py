"""Robot Framework keyword library for portfolio_QWIM and utils_portfolio tests.

Tests cover
-----------
Keyword wrappers for portfolio_QWIM construction from component names and
weights DataFrames, property access (name, count, components, raw weights),
weight-sum validation, and end-to-end portfolio value calculation via
calculate_portfolio_values and create_benchmark_portfolio_values.

Author:
    QWIM Development Team

Version:
    0.1.0

Last Modified:
    2026-02-28
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import polars as pl

# ---------------------------------------------------------------------------
# Ensure project root is on sys.path so that "src." imports resolve correctly
# ---------------------------------------------------------------------------
_PROJECT_ROOT: Path = Path(__file__).resolve().parents[3]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# ---------------------------------------------------------------------------
# Module-level import guard following project coding standards
# ---------------------------------------------------------------------------
MODULE_IMPORT_AVAILABLE: bool = True
_import_error_message: str = ""

try:
    from src.portfolios.portfolio_QWIM import portfolio_QWIM
    from src.portfolios.utils_portfolio import (
        calculate_portfolio_values,
        create_benchmark_portfolio_values,
        load_portfolio_weights,
        load_sample_etf_data,
    )
    from src.utils.custom_exceptions_errors_loggers.logger_custom import get_logger
    _logger = get_logger(__name__)
except ImportError as _exc:
    MODULE_IMPORT_AVAILABLE = False
    _import_error_message = str(_exc)
    import logging as _logging
    _logger = _logging.getLogger(__name__)
    _logger.warning("Import failed — keywords will raise on use: %s", _exc)

# ---------------------------------------------------------------------------
# Default data file paths
# ---------------------------------------------------------------------------
_ETF_DATA_PATH: Path = _PROJECT_ROOT / "inputs" / "raw" / "data_ETFs.csv"
_WEIGHTS_PATH: Path = _PROJECT_ROOT / "inputs" / "raw" / "sample_portfolio_weights_ETFs.csv"


# ---------------------------------------------------------------------------
# Internal helper
# ---------------------------------------------------------------------------

def _require_imports() -> None:
    """Raise RuntimeError when source modules could not be imported."""
    if not MODULE_IMPORT_AVAILABLE:
        raise RuntimeError(
            f"portfolio_QWIM source modules could not be imported: {_import_error_message}"
        )


# ===========================================================================
# Portfolio construction keywords
# ===========================================================================


def create_portfolio_from_component_names(name_portfolio: str, *components: str) -> Any:
    """Create a portfolio_QWIM from one or more component name strings (equal weights).

    Arguments:
        name_portfolio  Name to assign to the portfolio.
        *components     One or more ETF/asset component name strings.

    Returns:
        portfolio_QWIM instance with equal weights for all components.
    """
    _require_imports()

    if not components:
        raise ValueError("At least one component name must be supplied to the keyword.")

    _logger.debug(
        "Creating portfolio '%s' from %d components: %s",
        name_portfolio, len(components), list(components),
    )

    return portfolio_QWIM(
        name_portfolio=name_portfolio,
        names_components=list(components),
    )


def create_portfolio_from_weights_csv(name_portfolio: str = "CSV Portfolio") -> Any:
    """Create a portfolio_QWIM by loading weights from the default sample CSV file.

    Arguments:
        name_portfolio  Name to assign to the portfolio (default: 'CSV Portfolio').

    Returns:
        portfolio_QWIM instance initialised with the CSV weights.
    """
    _require_imports()

    weights_df = load_portfolio_weights(filepath=_WEIGHTS_PATH)
    _logger.debug("Loaded weights CSV with %d rows and columns: %s", weights_df.height, weights_df.columns)

    return portfolio_QWIM(
        name_portfolio=name_portfolio,
        portfolio_weights=weights_df,
    )


# ===========================================================================
# Portfolio property / accessor keywords
# ===========================================================================


def get_portfolio_name(portfolio: Any) -> str:
    """Return the name stored in a portfolio object.

    Arguments:
        portfolio   portfolio_QWIM instance.

    Returns:
        Portfolio name string.
    """
    _require_imports()
    return portfolio.get_portfolio_name


def get_portfolio_num_components(portfolio: Any) -> int:
    """Return the integer count of components in a portfolio.

    Arguments:
        portfolio   portfolio_QWIM instance.

    Returns:
        Integer number of components.
    """
    _require_imports()
    return portfolio.get_num_components


def get_portfolio_components_list(portfolio: Any) -> list[str]:
    """Return the list of component names contained in a portfolio.

    Arguments:
        portfolio   portfolio_QWIM instance.

    Returns:
        List of component name strings.
    """
    _require_imports()
    return portfolio.get_portfolio_components


def get_portfolio_weights_dataframe(portfolio: Any) -> Any:
    """Return the Polars DataFrame of portfolio weights.

    Arguments:
        portfolio   portfolio_QWIM instance.

    Returns:
        pl.DataFrame with Date column and one column per component.
    """
    _require_imports()
    return portfolio.get_portfolio_weights()


# ===========================================================================
# Assertion / validation keywords
# ===========================================================================


def portfolio_name_should_equal(portfolio: Any, expected_name: str) -> None:
    """Fail if the portfolio name does not equal the expected string.

    Arguments:
        portfolio       portfolio_QWIM instance.
        expected_name   Expected portfolio name string.
    """
    actual = portfolio.get_portfolio_name
    if actual != expected_name:
        raise AssertionError(
            f"Portfolio name mismatch: expected '{expected_name}', got '{actual}'"
        )


def portfolio_num_components_should_equal(portfolio: Any, expected_count: int) -> None:
    """Fail if the component count does not equal the expected integer.

    Arguments:
        portfolio       portfolio_QWIM instance.
        expected_count  Expected component count (int or int-castable string).
    """
    actual = portfolio.get_num_components
    if actual != int(expected_count):
        raise AssertionError(
            f"Component count mismatch: expected {expected_count}, got {actual}"
        )


def portfolio_should_contain_component(portfolio: Any, component_name: str) -> None:
    """Fail if a specific component name is not present in the portfolio.

    Arguments:
        portfolio        portfolio_QWIM instance.
        component_name   Component name string to check for.
    """
    components = portfolio.get_portfolio_components
    if component_name not in components:
        raise AssertionError(
            f"Component '{component_name}' not found. Available: {components}"
        )


def portfolio_weights_dataframe_should_be_valid(portfolio: Any) -> None:
    """Fail if the portfolio weights DataFrame is not a non-empty pl.DataFrame.

    Arguments:
        portfolio   portfolio_QWIM instance.
    """
    weights = portfolio.get_portfolio_weights()

    if not isinstance(weights, pl.DataFrame):
        raise AssertionError(
            f"Expected pl.DataFrame for portfolio weights, got {type(weights).__name__}"
        )
    if weights.is_empty():
        raise AssertionError("Portfolio weights DataFrame must not be empty.")


def portfolio_weights_should_sum_to_one(portfolio: Any) -> None:
    """Fail if any row of the portfolio weights does not sum to approximately 1.0.

    Arguments:
        portfolio   portfolio_QWIM instance.
    """
    weights = portfolio.get_portfolio_weights()
    numeric_cols = [c for c in weights.columns if c != "Date"]

    for row_idx in range(weights.height):
        row_sum = sum(float(weights[col][row_idx]) for col in numeric_cols)
        if abs(row_sum - 1.0) > 1e-4:
            raise AssertionError(
                f"Row {row_idx} weights sum to {row_sum:.6f}; expected ≈ 1.0"
            )


def validate_and_normalise_portfolio_weights(portfolio: Any) -> None:
    """Call validate_all_weights(normalize=True) and fail if it raises an exception.

    Arguments:
        portfolio   portfolio_QWIM instance.
    """
    _require_imports()
    portfolio.validate_all_weights(normalize=True)


# ===========================================================================
# Portfolio value calculation keywords
# ===========================================================================


def calculate_portfolio_values_from_csv(
    portfolio: Any,
    initial_value: float = 100.0,
) -> Any:
    """Calculate the portfolio value time series using the default ETF price CSV.

    Arguments:
        portfolio       portfolio_QWIM instance.
        initial_value   Starting portfolio value (default: 100.0).

    Returns:
        pl.DataFrame with Date and Portfolio_Value columns.
    """
    _require_imports()

    price_data = load_sample_etf_data(filepath=_ETF_DATA_PATH)
    return calculate_portfolio_values(
        portfolio_obj=portfolio,
        price_data=price_data,
        initial_value=float(initial_value),
    )


def portfolio_values_dataframe_should_be_valid(values_df: Any) -> None:
    """Fail if the portfolio values DataFrame is missing or structurally invalid.

    Arguments:
        values_df   pl.DataFrame returned by calculate_portfolio_values.
    """
    if not isinstance(values_df, pl.DataFrame):
        raise AssertionError(
            f"Expected pl.DataFrame for values, got {type(values_df).__name__}"
        )
    if values_df.is_empty():
        raise AssertionError("Portfolio values DataFrame must not be empty.")
    if "Portfolio_Value" not in values_df.columns:
        raise AssertionError(
            f"'Portfolio_Value' column missing. Available: {values_df.columns}"
        )


def portfolio_values_first_row_should_equal(
    values_df: Any,
    expected_value: float = 100.0,
    tolerance: float = 1e-3,
) -> None:
    """Fail if the first Portfolio_Value row does not match the expected starting value.

    Arguments:
        values_df        pl.DataFrame with Portfolio_Value column.
        expected_value   Expected initial value (default: 100.0).
        tolerance        Absolute tolerance for floating-point comparison (default: 1e-3).
    """
    first = values_df["Portfolio_Value"][0]
    if first is None:
        raise AssertionError("First Portfolio_Value is None — calculation likely failed.")
    if abs(float(first) - float(expected_value)) > float(tolerance):
        raise AssertionError(
            f"Expected first Portfolio_Value ≈ {expected_value}, got {first}"
        )


def portfolio_values_should_all_be_positive(values_df: Any) -> None:
    """Fail if any Portfolio_Value entry is zero or negative.

    Arguments:
        values_df   pl.DataFrame with Portfolio_Value column.
    """
    min_val = values_df["Portfolio_Value"].min()
    if min_val is None or float(min_val) <= 0.0:
        raise AssertionError(
            f"All Portfolio_Value entries must be > 0. Minimum found: {min_val}"
        )


def create_benchmark_from_portfolio_values(values_df: Any) -> Any:
    """Create a benchmark portfolio values DataFrame from an existing portfolio DataFrame.

    Arguments:
        values_df   pl.DataFrame with Portfolio_Value column.

    Returns:
        pl.DataFrame with benchmark Portfolio_Value column.
    """
    _require_imports()
    return create_benchmark_portfolio_values(values_df)


def benchmark_values_should_differ_from_portfolio(
    portfolio_values: Any,
    benchmark_values: Any,
) -> None:
    """Fail if benchmark values are identical to portfolio values.

    Arguments:
        portfolio_values   Original portfolio pl.DataFrame.
        benchmark_values   Benchmark pl.DataFrame.
    """
    port_list = portfolio_values["Portfolio_Value"].to_list()
    bench_list = benchmark_values["Portfolio_Value"].to_list()
    if port_list == bench_list:
        raise AssertionError(
            "Benchmark values are identical to portfolio values — expected divergence."
        )
