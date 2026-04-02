"""Portfolio methods from skfolio Python package.

===============

This module contains portfolio optimization methods using the skfolio Python package.

The module provides four main functions for different optimization categories:

- **Basic Methods**: Simple rule-based allocation (equal weight, inverse volatility, random)
- **Convex Optimization**: Mathematical optimization with convex objectives
- **Clustering Methods**: Hierarchical and graph-based allocation
- **Ensemble Methods**: Combining multiple optimization strategies

Dependencies:
    - skfolio: Portfolio optimization library
    - polars: High-performance DataFrame operations
    - numpy: Numerical computing
    - pandas: Data manipulation (for skfolio compatibility)

Examples
--------
    Basic usage with returns data:

    ```python
    import polars as pl
    from src.models.portfolio_optimization.pkg_skfolio import (
        calc_skfolio_optimization_basic,
        calc_skfolio_optimization_convex,
        calc_skfolio_optimization_clustering,
        calc_skfolio_optimization_ensemble,
    )

    # Load returns data
    returns_df = pl.read_csv("returns.csv")

    # Calculate equal-weighted portfolio
    portfolio = calc_skfolio_optimization_basic(
        returns_data=returns_df,
        optimization_type="BASIC_EQUAL_WEIGHTED",
        portfolio_name="Equal Weight Portfolio",
    )
    ```

References
----------
    - skfolio documentation: https://skfolio.org/
    - Markowitz, H. (1952). Portfolio Selection. Journal of Finance.
    - López de Prado, M. (2016). Building Diversified Portfolios that Outperform Out-of-Sample.
"""

from __future__ import annotations

import logging

from datetime import UTC, datetime
from typing import Any

import pandas as pd
import polars as pl


# skfolio imports
from skfolio.optimization import (
    BenchmarkTracker,
    DistributionallyRobustCVaR,
    EqualWeighted,
    HierarchicalEqualRiskContribution,
    HierarchicalRiskParity,
    InverseVolatility,
    MaximumDiversification,
    MeanRisk,
    NestedClustersOptimization,
    ObjectiveFunction,
    Random,
    RiskBudgeting,
    SchurComplementary,
    StackingOptimization,
)

from src.models.portfolio_optimization.utils_portfolio_optimization import (
    portfolio_optimization_type,
)

# Internal imports
from src.portfolios.portfolio_QWIM import portfolio_QWIM


# Configure logging for the module
logger = logging.getLogger(__name__)


# =============================================================================
# Constants and Configuration
# =============================================================================


BASIC_OPTIMIZATION_MAPPING = {
    portfolio_optimization_type.BASIC_EQUAL_WEIGHTED: EqualWeighted,
    portfolio_optimization_type.BASIC_INVERSE_VOLATILITY: InverseVolatility,
    portfolio_optimization_type.BASIC_RANDOM_DIRICHLET: Random,
}

CONVEX_OPTIMIZATION_MAPPING = {
    portfolio_optimization_type.CONVEX_MEAN_RISK: MeanRisk,
    portfolio_optimization_type.CONVEX_RISK_BUDGETING: RiskBudgeting,
    portfolio_optimization_type.CONVEX_MAXIMUM_DIVERSIFICATION: MaximumDiversification,
    portfolio_optimization_type.CONVEX_DISTRIBUTIONALLY_ROBUST_CVAR: DistributionallyRobustCVaR,
    portfolio_optimization_type.CONVEX_BENCHMARK_TRACKING: BenchmarkTracker,
}

CLUSTERING_OPTIMIZATION_MAPPING = {
    portfolio_optimization_type.CLUSTERING_HIERARCHICAL_RISK_PARITY: HierarchicalRiskParity,
    portfolio_optimization_type.CLUSTERING_HIERARCHICAL_EQUAL_RISK_CONTRIBUTION: HierarchicalEqualRiskContribution,
    portfolio_optimization_type.CLUSTERING_SCHUR_COMPLEMENTARY_ALLOCATION: SchurComplementary,
    portfolio_optimization_type.CLUSTERING_NESTED: NestedClustersOptimization,
}


# =============================================================================
# Helper Functions
# =============================================================================


def _validate_returns_data(returns_data: pl.DataFrame) -> tuple[bool, str]:
    """Validate returns data using defensive programming.

    Args:
        returns_data: Polars DataFrame containing asset returns.

    Returns
    -------
        Tuple of (is_valid, error_message).
    """
    # Input validation with early returns
    if returns_data is None:
        return False, "returns_data cannot be None"

    if not isinstance(returns_data, pl.DataFrame):
        return False, f"returns_data must be a Polars DataFrame, got {type(returns_data).__name__}"

    if returns_data.is_empty():
        return False, "returns_data cannot be empty"

    # Configuration validation - check for Date column
    if "Date" not in returns_data.columns:
        return False, "returns_data must contain a 'Date' column"

    # Business logic validation - need at least one asset column
    asset_columns = [col for col in returns_data.columns if col != "Date"]
    if len(asset_columns) == 0:
        return False, "returns_data must contain at least one asset column besides 'Date'"

    # Validate numeric columns
    for col in asset_columns:
        if returns_data[col].dtype not in [pl.Float64, pl.Float32, pl.Int64, pl.Int32]:
            return False, f"Column '{col}' must be numeric, got {returns_data[col].dtype}"

    return True, ""


def _convert_polars_to_pandas_returns(returns_data: pl.DataFrame) -> pd.DataFrame:
    """Convert Polars DataFrame to pandas for skfolio compatibility.

    Args:
        returns_data: Polars DataFrame with Date column and asset returns.

    Returns
    -------
        pandas DataFrame with Date index and asset columns.
    """
    # Convert to pandas
    df_pandas = returns_data.to_pandas()

    # Set Date as index
    if "Date" in df_pandas.columns:
        df_pandas["Date"] = pd.to_datetime(df_pandas["Date"])
        df_pandas = df_pandas.set_index("Date")

    return df_pandas


def _extract_weights_to_portfolio_qwim(
    skfolio_model: Any,
    asset_names: list[str],
    portfolio_name: str,
    optimization_date: str | datetime | None = None,
) -> portfolio_QWIM:
    """Extract weights from skfolio model and create portfolio_QWIM object.

    Args:
        skfolio_model: Fitted skfolio optimization model.
        asset_names: List of asset names.
        portfolio_name: Name for the portfolio.
        optimization_date: Date for the portfolio weights.

    Returns
    -------
        portfolio_QWIM object with optimized weights.

    Notes
    -----
        This function creates a portfolio_QWIM object directly by providing
        the component names, then manually sets the weights DataFrame to avoid
        initialization validation issues.
    """
    # Get weights from fitted model
    weights = skfolio_model.weights_

    # Use current date if not provided
    if optimization_date is None:
        optimization_date = datetime.now(UTC).strftime("%Y-%m-%d")
    elif isinstance(optimization_date, datetime):
        optimization_date = optimization_date.strftime("%Y-%m-%d")

    # Create weights dictionary
    weights_dict: dict[str, list[object]] = {"Date": [optimization_date]}
    for idx, asset_name in enumerate(asset_names):
        weights_dict[asset_name] = [float(weights[idx])]

    # Create Polars DataFrame
    weights_df = pl.DataFrame(weights_dict)

    # Create portfolio_QWIM from components (equal weights initially)
    portfolio = portfolio_QWIM(
        name_portfolio=portfolio_name,
        names_components=asset_names,
        date_portfolio=optimization_date,
    )

    # Directly set the optimized weights DataFrame
    portfolio.m_portfolio_weights = weights_df

    return portfolio


def _get_optimization_type_enum(
    optimization_type: str | portfolio_optimization_type,
) -> portfolio_optimization_type:
    """Convert string or enum to portfolio_optimization_type enum.

    Args:
        optimization_type: Optimization type as string or enum.

    Returns
    -------
        portfolio_optimization_type enum value.

    Raises
    ------
        ValueError: If optimization_type is invalid.
    """
    if isinstance(optimization_type, portfolio_optimization_type):
        return optimization_type

    if isinstance(optimization_type, str):
        try:
            return portfolio_optimization_type[optimization_type]
        except KeyError as err:
            valid_types = [t.name for t in portfolio_optimization_type]
            raise ValueError(
                f"Invalid optimization_type '{optimization_type}'. Valid types: {valid_types}",
            ) from err

    raise TypeError(
        f"optimization_type must be str or portfolio_optimization_type, "
        f"got {type(optimization_type).__name__}",
    )


# =============================================================================
# Basic Optimization Functions
# =============================================================================


def calc_skfolio_optimization_basic(
    returns_data: pl.DataFrame,
    optimization_type: str
    | portfolio_optimization_type = portfolio_optimization_type.BASIC_EQUAL_WEIGHTED,
    portfolio_name: str | None = None,
    optimization_date: str | datetime | None = None,
    random_state: int | None = None,
    **kwargs: Any,
) -> portfolio_QWIM:
    r"""Calculate optimal portfolio using basic optimization methods from skfolio.

    Basic methods are simple rule-based allocation strategies that don't require
    optimization solvers. They include equal weighting, inverse volatility weighting,
    and random sampling.

    Args:
        returns_data: Polars DataFrame with 'Date' column and asset return columns.
            Returns should be in decimal form (e.g., 0.01 for 1% return).
        optimization_type: Type of basic optimization to use.
            - BASIC_EQUAL_WEIGHTED: Equal weight allocation (1/N portfolio)
            - BASIC_INVERSE_VOLATILITY: Weights inversely proportional to volatility
            - BASIC_RANDOM_DIRICHLET: Random weights from Dirichlet distribution
        portfolio_name: Name for the resulting portfolio. Defaults to optimization type.
        optimization_date: Date for the portfolio weights. Defaults to current date.
        random_state: Random seed for reproducibility (used with BASIC_RANDOM_DIRICHLET).
        **kwargs: Additional keyword arguments passed to the skfolio optimizer.

    Returns
    -------
        portfolio_QWIM: Portfolio object containing optimized weights.

    Raises
    ------
        ValueError: If returns_data is invalid or optimization_type is not a basic method.

    Examples
    --------
        Equal-weighted portfolio:

        ```python
        import polars as pl

        returns = pl.DataFrame(
            {
                "Date": ["2024-01-01", "2024-01-02", "2024-01-03"],
                "AAPL": [0.01, -0.005, 0.02],
                "MSFT": [0.015, 0.01, -0.01],
                "GOOG": [0.005, 0.02, 0.015],
            }
        )

        portfolio = calc_skfolio_optimization_basic(
            returns_data=returns,
            optimization_type="BASIC_EQUAL_WEIGHTED",
            portfolio_name="Equal Weight Tech",
        )
        print(portfolio.get_portfolio_weights)
        ```

        Inverse volatility portfolio:

        ```python
        portfolio = calc_skfolio_optimization_basic(
            returns_data=returns,
            optimization_type="BASIC_INVERSE_VOLATILITY",
            portfolio_name="Inverse Vol Tech",
        )
        ```

    Notes
    -----
        - Equal weighting assigns $w_i = 1/N$ to each asset
        - Inverse volatility assigns $w_i \\propto 1/\\sigma_i$
        - Random Dirichlet samples from $Dir(\\alpha)$ distribution

    References
    ----------
        - DeMiguel, V., Garlappi, L., & Uppal, R. (2009). Optimal versus naive
          diversification: How inefficient is the 1/N portfolio strategy?
    """
    # Input validation with early returns
    is_valid, error_msg = _validate_returns_data(returns_data)
    if not is_valid:
        logger.error(
            "Invalid returns_data",
            extra={"error_message": error_msg},
        )
        raise ValueError(error_msg)

    # Convert optimization_type to enum
    opt_type_enum = _get_optimization_type_enum(optimization_type)

    # Validate optimization type is a basic method
    if opt_type_enum not in BASIC_OPTIMIZATION_MAPPING:
        valid_basic_types = list(BASIC_OPTIMIZATION_MAPPING.keys())
        error_msg = (
            f"optimization_type '{opt_type_enum.name}' is not a basic method. "
            f"Valid basic types: {[t.name for t in valid_basic_types]}"
        )
        logger.error("Invalid optimization type", extra={"error_message": error_msg})
        raise ValueError(error_msg)

    # Set default portfolio name
    if portfolio_name is None:
        portfolio_name = f"Portfolio_{opt_type_enum.name}"

    # Convert to pandas for skfolio
    returns_pandas = _convert_polars_to_pandas_returns(returns_data)
    asset_names = list(returns_pandas.columns)

    logger.info(
        "Calculating basic optimization",
        extra={
            "optimization_type": opt_type_enum.name,
            "num_assets": len(asset_names),
        },
    )

    # Get optimizer class
    optimizer_class = BASIC_OPTIMIZATION_MAPPING[opt_type_enum]

    # Configure optimizer based on type
    # Note: Random optimizer from skfolio doesn't support random_state parameter
    # If reproducibility needed, set numpy/random seed before calling this function
    # The random_state parameter is kept in the signature for API consistency but not used
    _ = random_state  # Mark as intentionally unused

    # Create and fit optimizer
    try:
        optimizer = optimizer_class(**kwargs)
        optimizer.fit(returns_pandas)

        logger.info(
            "Optimization complete",
            extra={
                "portfolio_name": portfolio_name,
                "weights_sum": float(sum(optimizer.weights_)),
            },
        )

        # Extract weights and create portfolio_QWIM
        return _extract_weights_to_portfolio_qwim(
            skfolio_model=optimizer,
            asset_names=asset_names,
            portfolio_name=portfolio_name,
            optimization_date=optimization_date,
        )

    except Exception as e:
        logger.error(
            "Basic optimization failed",
            extra={"error_message": str(e)},
        )
        raise ValueError(f"Optimization failed: {e!s}") from e


# =============================================================================
# Convex Optimization Functions
# =============================================================================


def calc_skfolio_optimization_convex(
    returns_data: pl.DataFrame,
    optimization_type: str
    | portfolio_optimization_type = portfolio_optimization_type.CONVEX_MEAN_RISK,
    portfolio_name: str | None = None,
    optimization_date: str | datetime | None = None,
    objective_function: ObjectiveFunction = ObjectiveFunction.MINIMIZE_RISK,
    risk_measure: str | None = None,
    risk_aversion: float = 1.0,
    min_weights: float | None = 0.0,
    max_weights: float | None = 1.0,
    budget: float = 1.0,
    benchmark_returns: pl.DataFrame | None = None,
    **kwargs: Any,
) -> portfolio_QWIM:
    r"""Calculate optimal portfolio using convex optimization methods from skfolio.

    Convex optimization methods use mathematical optimization with convex objectives
    and constraints. These methods guarantee global optimality and can be solved
    efficiently using convex solvers.

    Args:
        returns_data: Polars DataFrame with 'Date' column and asset return columns.
        optimization_type: Type of convex optimization to use.
            - CONVEX_MEAN_RISK: Mean-variance/mean-risk optimization (Markowitz)
            - CONVEX_RISK_BUDGETING: Risk parity and risk budgeting
            - CONVEX_MAXIMUM_DIVERSIFICATION: Maximize diversification ratio
            - CONVEX_DISTRIBUTIONALLY_ROBUST_CVAR: Robust CVaR with uncertainty
            - CONVEX_BENCHMARK_TRACKING: Minimize tracking error to benchmark
        portfolio_name: Name for the resulting portfolio. Defaults to optimization type.
        optimization_date: Date for the portfolio weights. Defaults to current date.
        objective_function: Objective function for optimization.
            - MINIMIZE_RISK: Minimize portfolio risk
            - MAXIMIZE_RETURN: Maximize expected return
            - MAXIMIZE_UTILITY: Maximize mean-variance utility
            - MAXIMIZE_RATIO: Maximize Sharpe ratio
        risk_measure: Risk measure to use (e.g., "variance", "cvar", "cdar").
        risk_aversion: Risk aversion coefficient for utility maximization.
        min_weights: Minimum weight constraint for each asset.
        max_weights: Maximum weight constraint for each asset.
        budget: Total portfolio budget (sum of weights). Default is 1.0 (fully invested).
        benchmark_returns: Benchmark returns for tracking error optimization.
        **kwargs: Additional keyword arguments passed to the skfolio optimizer.

    Returns
    -------
        portfolio_QWIM: Portfolio object containing optimized weights.

    Raises
    ------
        ValueError: If returns_data is invalid or optimization_type is not a convex method.

    Examples
    --------
        Mean-variance optimization (minimum variance):

        ```python
        portfolio = calc_skfolio_optimization_convex(
            returns_data=returns,
            optimization_type="CONVEX_MEAN_RISK",
            objective_function=ObjectiveFunction.MINIMIZE_RISK,
            portfolio_name="Min Variance Portfolio",
        )
        ```

        Maximum Sharpe ratio:

        ```python
        from skfolio.optimization import ObjectiveFunction

        portfolio = calc_skfolio_optimization_convex(
            returns_data=returns,
            optimization_type="CONVEX_MEAN_RISK",
            objective_function=ObjectiveFunction.MAXIMIZE_RATIO,
            portfolio_name="Max Sharpe Portfolio",
        )
        ```

        Risk budgeting (equal risk contribution):

        ```python
        portfolio = calc_skfolio_optimization_convex(
            returns_data=returns,
            optimization_type="CONVEX_RISK_BUDGETING",
            portfolio_name="Risk Parity Portfolio",
        )
        ```

    Notes
    -----
        Mean-Variance Optimization:
        $$
        \\min_w \\; w^T \\Sigma w - \\lambda \\mu^T w
        $$

        Risk Budgeting (Equal Risk Contribution):
        $$
        \\min_w \\; \\sum_i \\left( w_i \\frac{(\\Sigma w)_i}{w^T \\Sigma w} - b_i \\right)^2
        $$

        Maximum Diversification:
        $$
        \\max_w \\; \\frac{w^T \\sigma}{\\sqrt{w^T \\Sigma w}}
        $$

    References
    ----------
        - Markowitz, H. (1952). Portfolio Selection. Journal of Finance.
        - Maillard, S., Roncalli, T., & Teïletche, J. (2010). The Properties of
          Equally Weighted Risk Contribution Portfolios.
        - Choueifaty, Y., & Coignard, Y. (2008). Toward Maximum Diversification.
    """
    # Input validation with early returns
    is_valid, error_msg = _validate_returns_data(returns_data)
    if not is_valid:
        logger.error(
            "Invalid returns_data",
            extra={"error_message": error_msg},
        )
        raise ValueError(error_msg)

    # Convert optimization_type to enum
    opt_type_enum = _get_optimization_type_enum(optimization_type)

    # Validate optimization type is a convex method
    if opt_type_enum not in CONVEX_OPTIMIZATION_MAPPING:
        valid_convex_types = list(CONVEX_OPTIMIZATION_MAPPING.keys())
        error_msg = (
            f"optimization_type '{opt_type_enum.name}' is not a convex method. "
            f"Valid convex types: {[t.name for t in valid_convex_types]}"
        )
        logger.error("Invalid optimization type", extra={"error_message": error_msg})
        raise ValueError(error_msg)

    # Set default portfolio name
    if portfolio_name is None:
        portfolio_name = f"Portfolio_{opt_type_enum.name}"

    # Convert to pandas for skfolio
    returns_pandas = _convert_polars_to_pandas_returns(returns_data)
    asset_names = list(returns_pandas.columns)

    logger.info(
        "Calculating convex optimization",
        extra={
            "optimization_type": opt_type_enum.name,
            "num_assets": len(asset_names),
            "objective_function": str(objective_function),
        },
    )

    # Get optimizer class
    optimizer_class = CONVEX_OPTIMIZATION_MAPPING[opt_type_enum]

    # Configure optimizer parameters based on type
    # benchmark_pandas initialized here; set inside CONVEX_BENCHMARK_TRACKING branch
    benchmark_pandas: pd.DataFrame | None = None
    optimizer_kwargs = {**kwargs}

    if opt_type_enum == portfolio_optimization_type.CONVEX_MEAN_RISK:
        optimizer_kwargs["objective_function"] = objective_function
        optimizer_kwargs["risk_aversion"] = risk_aversion
        if risk_measure is not None:
            optimizer_kwargs["risk_measure"] = risk_measure
        optimizer_kwargs["min_weights"] = min_weights
        optimizer_kwargs["max_weights"] = max_weights
        optimizer_kwargs["budget"] = budget

    elif opt_type_enum == portfolio_optimization_type.CONVEX_RISK_BUDGETING:
        if risk_measure is not None:
            optimizer_kwargs["risk_measure"] = risk_measure
        optimizer_kwargs["min_weights"] = min_weights
        optimizer_kwargs["max_weights"] = max_weights
        # Note: budget constraint handled via min_weights/max_weights sum

    elif (
        opt_type_enum == portfolio_optimization_type.CONVEX_MAXIMUM_DIVERSIFICATION
        or opt_type_enum == portfolio_optimization_type.CONVEX_DISTRIBUTIONALLY_ROBUST_CVAR
    ):
        optimizer_kwargs["min_weights"] = min_weights
        optimizer_kwargs["max_weights"] = max_weights
        optimizer_kwargs["budget"] = budget

    elif opt_type_enum == portfolio_optimization_type.CONVEX_BENCHMARK_TRACKING:
        if benchmark_returns is None:
            raise ValueError("benchmark_returns required for CONVEX_BENCHMARK_TRACKING")
        # Convert benchmark to pandas
        benchmark_pandas = _convert_polars_to_pandas_returns(benchmark_returns)
        # BenchmarkTracker uses benchmark in fit(), not __init__
        optimizer_kwargs["min_weights"] = min_weights
        optimizer_kwargs["max_weights"] = max_weights

    # Create and fit optimizer
    try:
        optimizer = optimizer_class(**optimizer_kwargs)
        # BenchmarkTracker requires benchmark passed to fit() as y parameter.
        # sklearn expects y as a 1-D array; ravel() avoids DataConversionWarning
        # when benchmark_pandas is a single-column DataFrame (column-vector).
        if opt_type_enum == portfolio_optimization_type.CONVEX_BENCHMARK_TRACKING:
            if benchmark_pandas is None:
                raise ValueError("benchmark_returns required for CONVEX_BENCHMARK_TRACKING")
            optimizer.fit(returns_pandas, y=benchmark_pandas.to_numpy().ravel())
        else:
            optimizer.fit(returns_pandas)

        logger.info(
            "Optimization complete",
            extra={
                "portfolio_name": portfolio_name,
                "weights_sum": float(sum(optimizer.weights_)),
            },
        )

        # Extract weights and create portfolio_QWIM
        return _extract_weights_to_portfolio_qwim(
            skfolio_model=optimizer,
            asset_names=asset_names,
            portfolio_name=portfolio_name,
            optimization_date=optimization_date,
        )

    except Exception as e:
        logger.error(
            "Convex optimization failed",
            extra={"error_message": str(e)},
        )
        raise ValueError(f"Optimization failed: {e!s}") from e


# =============================================================================
# Clustering Optimization Functions
# =============================================================================


def calc_skfolio_optimization_clustering(
    returns_data: pl.DataFrame,
    optimization_type: str
    | portfolio_optimization_type = portfolio_optimization_type.CLUSTERING_HIERARCHICAL_RISK_PARITY,
    portfolio_name: str | None = None,
    optimization_date: str | datetime | None = None,
    risk_measure: str | None = None,
    hierarchical_clustering_estimator: Any | None = None,
    min_weights: float | None = 0.0,
    max_weights: float | None = 1.0,
    **kwargs: Any,
) -> portfolio_QWIM:
    """Calculate optimal portfolio using clustering-based optimization methods from skfolio.

    Clustering methods use hierarchical and graph-based algorithms to allocate
    portfolio weights. These methods are particularly robust to estimation error
    and perform well out-of-sample.

    Args:
        returns_data: Polars DataFrame with 'Date' column and asset return columns.
        optimization_type: Type of clustering optimization to use.
            - CLUSTERING_HIERARCHICAL_RISK_PARITY: HRP using hierarchical clustering
            - CLUSTERING_HIERARCHICAL_EQUAL_RISK_CONTRIBUTION: HERC method
            - CLUSTERING_SCHUR_COMPLEMENTARY_ALLOCATION: Schur complement-based
            - CLUSTERING_NESTED: Nested clustering optimization (NCO)
        portfolio_name: Name for the resulting portfolio. Defaults to optimization type.
        optimization_date: Date for the portfolio weights. Defaults to current date.
        risk_measure: Risk measure for allocation (e.g., "variance", "cvar").
        hierarchical_clustering_estimator: Custom clustering estimator.
        min_weights: Minimum weight constraint for each asset.
        max_weights: Maximum weight constraint for each asset.
        **kwargs: Additional keyword arguments passed to the skfolio optimizer.

    Returns
    -------
        portfolio_QWIM: Portfolio object containing optimized weights.

    Raises
    ------
        ValueError: If returns_data is invalid or optimization_type is not a clustering method.

    Examples
    --------
        Hierarchical Risk Parity:

        ```python
        portfolio = calc_skfolio_optimization_clustering(
            returns_data=returns,
            optimization_type="CLUSTERING_HIERARCHICAL_RISK_PARITY",
            portfolio_name="HRP Portfolio",
        )
        ```

        Nested Clusters Optimization:

        ```python
        portfolio = calc_skfolio_optimization_clustering(
            returns_data=returns,
            optimization_type="CLUSTERING_NESTED",
            portfolio_name="NCO Portfolio",
        )
        ```

    Notes
    -----
        Hierarchical Risk Parity (HRP) Algorithm:
        1. Tree clustering on correlation matrix
        2. Quasi-diagonalization of covariance matrix
        3. Recursive bisection for weight allocation

        Benefits of clustering methods:
        - No matrix inversion required (more stable)
        - Better out-of-sample performance
        - Natural diversification across clusters

    References
    ----------
        - López de Prado, M. (2016). Building Diversified Portfolios that
          Outperform Out-of-Sample. Journal of Portfolio Management.
        - López de Prado, M. (2018). Advances in Financial Machine Learning.
    """
    # Input validation with early returns
    is_valid, error_msg = _validate_returns_data(returns_data)
    if not is_valid:
        logger.error(
            "Invalid returns_data",
            extra={"error_message": error_msg},
        )
        raise ValueError(error_msg)

    # Convert optimization_type to enum
    opt_type_enum = _get_optimization_type_enum(optimization_type)

    # Validate optimization type is a clustering method
    if opt_type_enum not in CLUSTERING_OPTIMIZATION_MAPPING:
        valid_clustering_types = list(CLUSTERING_OPTIMIZATION_MAPPING.keys())
        raise ValueError(
            f"optimization_type '{opt_type_enum.name}' is not a clustering method. "
            f"Valid clustering types: {[t.name for t in valid_clustering_types]}",
        )

    # Set default portfolio name
    if portfolio_name is None:
        portfolio_name = f"Portfolio_{opt_type_enum.name}"

    # Convert to pandas for skfolio
    returns_pandas = _convert_polars_to_pandas_returns(returns_data)
    asset_names = list(returns_pandas.columns)

    logger.info(
        "Calculating clustering optimization",
        extra={
            "optimization_type": opt_type_enum.name,
            "num_assets": len(asset_names),
        },
    )

    # Get optimizer class
    optimizer_class = CLUSTERING_OPTIMIZATION_MAPPING[opt_type_enum]

    # Configure optimizer parameters
    optimizer_kwargs = {**kwargs}

    if risk_measure is not None:
        optimizer_kwargs["risk_measure"] = risk_measure

    if hierarchical_clustering_estimator is not None:
        optimizer_kwargs["hierarchical_clustering_estimator"] = hierarchical_clustering_estimator

    # Add weight constraints where applicable
    if opt_type_enum in [
        portfolio_optimization_type.CLUSTERING_HIERARCHICAL_RISK_PARITY,
        portfolio_optimization_type.CLUSTERING_HIERARCHICAL_EQUAL_RISK_CONTRIBUTION,
    ]:
        optimizer_kwargs["min_weights"] = min_weights
        optimizer_kwargs["max_weights"] = max_weights

    # Create and fit optimizer
    try:
        optimizer = optimizer_class(**optimizer_kwargs)
        optimizer.fit(returns_pandas)

        logger.info(
            "Optimization complete",
            extra={
                "portfolio_name": portfolio_name,
                "weights_sum": float(sum(optimizer.weights_)),
            },
        )

        # Extract weights and create portfolio_QWIM
        return _extract_weights_to_portfolio_qwim(
            skfolio_model=optimizer,
            asset_names=asset_names,
            portfolio_name=portfolio_name,
            optimization_date=optimization_date,
        )

    except Exception as e:
        logger.error(
            "Clustering optimization failed",
            extra={"error_message": str(e)},
        )
        raise ValueError(f"Optimization failed: {e!s}") from e


# =============================================================================
# Ensemble Optimization Functions
# =============================================================================


def calc_skfolio_optimization_ensemble(
    returns_data: pl.DataFrame,
    optimization_type: str
    | portfolio_optimization_type = portfolio_optimization_type.ENSEMBLE_STACKING,
    portfolio_name: str | None = None,
    optimization_date: str | datetime | None = None,
    estimators: list[Any] | None = None,
    final_estimator: Any | None = None,
    cv: int | None = None,
    **kwargs: Any,
) -> portfolio_QWIM:
    """Calculate optimal portfolio using ensemble optimization methods from skfolio.

    Ensemble methods combine multiple optimization strategies to create more
    robust portfolios. Stacking optimization uses cross-validation to learn
    optimal weights for combining base optimizers.

    Args:
        returns_data: Polars DataFrame with 'Date' column and asset return columns.
        optimization_type: Type of ensemble optimization to use.
            - ENSEMBLE_STACKING: Stacked ensemble of multiple optimizers
        portfolio_name: Name for the resulting portfolio. Defaults to optimization type.
        optimization_date: Date for the portfolio weights. Defaults to current date.
        estimators: List of (name, estimator) tuples for base optimizers.
            Default uses HRP, equal weight, and inverse volatility.
        final_estimator: Final estimator for combining base predictions.
            Default uses mean-risk optimization.
        cv: Number of cross-validation folds. Default is 5.
        **kwargs: Additional keyword arguments passed to the skfolio optimizer.

    Returns
    -------
        portfolio_QWIM: Portfolio object containing optimized weights.

    Raises
    ------
        ValueError: If returns_data is invalid or optimization_type is not an ensemble method.

    Examples
    --------
        Default stacking ensemble:

        ```python
        portfolio = calc_skfolio_optimization_ensemble(
            returns_data=returns,
            optimization_type="ENSEMBLE_STACKING",
            portfolio_name="Stacked Portfolio",
        )
        ```

        Custom ensemble with specific base estimators:

        ```python
        from skfolio.optimization import HierarchicalRiskParity, MeanRisk, EqualWeighted

        estimators = [
            ("hrp", HierarchicalRiskParity()),
            ("mean_risk", MeanRisk()),
            ("equal", EqualWeighted()),
        ]

        portfolio = calc_skfolio_optimization_ensemble(
            returns_data=returns,
            optimization_type="ENSEMBLE_STACKING",
            estimators=estimators,
            portfolio_name="Custom Stacked Portfolio",
        )
        ```

    Notes
    -----
        Stacking Optimization Process:
        1. Train multiple base optimizers on training data
        2. Generate portfolio predictions for validation data
        3. Train meta-learner to combine base predictions
        4. Final portfolio is weighted combination of base portfolios

        Benefits:
        - Reduces model risk by diversifying across strategies
        - Can adapt to different market regimes
        - Often more stable out-of-sample performance

    References
    ----------
        - Wolpert, D. H. (1992). Stacked Generalization. Neural Networks.
        - Breiman, L. (1996). Stacked Regressions. Machine Learning.
    """
    # Input validation with early returns
    is_valid, error_msg = _validate_returns_data(returns_data)
    if not is_valid:
        logger.error(
            "Invalid returns_data",
            extra={"error_message": error_msg},
        )
        raise ValueError(error_msg)

    # Convert optimization_type to enum
    opt_type_enum = _get_optimization_type_enum(optimization_type)

    # Validate optimization type is an ensemble method
    valid_ensemble_types = [portfolio_optimization_type.ENSEMBLE_STACKING]
    if opt_type_enum not in valid_ensemble_types:
        raise ValueError(
            f"optimization_type '{opt_type_enum.name}' is not an ensemble method. "
            f"Valid ensemble types: {[t.name for t in valid_ensemble_types]}",
        )

    # Set default portfolio name
    if portfolio_name is None:
        portfolio_name = f"Portfolio_{opt_type_enum.name}"

    # Convert to pandas for skfolio
    returns_pandas = _convert_polars_to_pandas_returns(returns_data)
    asset_names = list(returns_pandas.columns)

    logger.info(
        "Calculating ensemble optimization",
        extra={
            "optimization_type": opt_type_enum.name,
            "num_assets": len(asset_names),
        },
    )

    # Configure default estimators if not provided
    if estimators is None:
        estimators = [
            ("hrp", HierarchicalRiskParity()),
            ("equal", EqualWeighted()),
            ("inverse_vol", InverseVolatility()),
        ]

    # Configure default final estimator if not provided
    if final_estimator is None:
        final_estimator = MeanRisk()

    # Configure optimizer parameters
    optimizer_kwargs = {"estimators": estimators, "final_estimator": final_estimator, **kwargs}

    if cv is not None:
        optimizer_kwargs["cv"] = cv

    # Create and fit optimizer
    try:
        optimizer = StackingOptimization(**optimizer_kwargs)
        optimizer.fit(returns_pandas)

        logger.info(
            "Optimization complete",
            extra={
                "portfolio_name": portfolio_name,
                "weights_sum": float(sum(optimizer.weights_)),
            },
        )

        # Extract weights and create portfolio_QWIM
        return _extract_weights_to_portfolio_qwim(
            skfolio_model=optimizer,
            asset_names=asset_names,
            portfolio_name=portfolio_name,
            optimization_date=optimization_date,
        )

    except Exception as e:
        logger.error(
            "Ensemble optimization failed",
            extra={"error_message": str(e)},
        )
        raise ValueError(f"Optimization failed: {e!s}") from e
