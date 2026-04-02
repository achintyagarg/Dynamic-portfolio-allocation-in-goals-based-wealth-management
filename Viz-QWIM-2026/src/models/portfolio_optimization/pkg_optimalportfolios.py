"""Portfolio optimization methods from optimalportfolios Python package.

This module provides wrapper functions for portfolio optimization methods from the
optimalportfolios package. All functions accept Polars DataFrames as input and return
portfolio_QWIM objects as output.

The optimalportfolios package implements various portfolio optimization methods:
- Minimum variance portfolio
- Maximum quadratic utility
- Budgeted risk contribution (risk parity)
- Maximum diversification
- Maximum Sharpe ratio
- Maximum Cara utility under Gaussian mixture model
- Tracking error minimization

References
----------
- optimalportfolios documentation: https://pypi.org/project/optimalportfolios/
- Markowitz, H. (1952). Portfolio Selection. Journal of Finance.
- Choueifaty, Y., & Coignard, Y. (2008). Toward Maximum Diversification.

Examples
--------
    Basic minimum variance optimization:

    ```python
    import polars as pl
    from src.models.portfolio_optimization.pkg_optimalportfolios import (
        calc_optimalportfolios_minimum_variance,
    )

    returns = pl.DataFrame(
        {
            "Date": ["2024-01-01", "2024-01-02", "2024-01-03"],
            "AAPL": [0.01, -0.005, 0.02],
            "MSFT": [0.015, 0.01, -0.01],
            "GOOG": [0.005, 0.02, 0.015],
        }
    )

    portfolio = calc_optimalportfolios_minimum_variance(
        returns_data=returns, portfolio_name="Min Variance Portfolio"
    )
    ```
"""

from __future__ import annotations

import logging
import warnings

from datetime import UTC, datetime

import cvxpy as cp
import numpy as np
import pandas as pd
import polars as pl

from optimalportfolios import (
    Constraints,
    PortfolioObjective,
    cvx_maximize_portfolio_sharpe,
    cvx_quadratic_optimisation,
    fit_gaussian_mixture,
    opt_maximise_diversification,
    opt_maximize_cara_mixture,
    opt_risk_budgeting,
)

from src.portfolios.portfolio_QWIM import portfolio_QWIM


# Configure logging
logger = logging.getLogger(__name__)


# =============================================================================
# Helper Functions
# =============================================================================


def _validate_returns_data(returns_data: pl.DataFrame) -> tuple[bool, str]:
    """Validate returns DataFrame structure and content.

    Defensive validation following project standards with early returns.

    Args:
        returns_data: Polars DataFrame to validate.

    Returns
    -------
        Tuple of (is_valid, error_message).
        If valid, returns (True, ""). If invalid, returns (False, error_description).
    """
    # Check if None
    if returns_data is None:
        return False, "returns_data cannot be None"

    # Check if DataFrame
    if not isinstance(returns_data, pl.DataFrame):
        return False, f"returns_data must be Polars DataFrame, got {type(returns_data)}"

    # Check if empty
    if len(returns_data) == 0:
        return False, "returns_data cannot be empty"

    # Check for Date column
    if "Date" not in returns_data.columns:
        return False, "returns_data must have 'Date' column"

    # Check for asset columns (at least one beyond Date)
    asset_columns = [col for col in returns_data.columns if col != "Date"]
    if len(asset_columns) == 0:
        return False, "returns_data must have at least one asset column besides 'Date'"

    # Check that asset columns are numeric
    for col in asset_columns:
        if returns_data[col].dtype not in [pl.Float32, pl.Float64, pl.Int32, pl.Int64]:
            return False, f"Asset column '{col}' must be numeric, got {returns_data[col].dtype}"

    return True, ""


def _convert_polars_to_numpy_returns(returns_data: pl.DataFrame) -> np.ndarray:
    """Convert Polars DataFrame to NumPy array for optimalportfolios package.

    Args:
        returns_data: Polars DataFrame with 'Date' and asset return columns.

    Returns
    -------
        NumPy array of returns (n_samples, n_assets).
    """
    # Extract asset columns (exclude Date)
    asset_columns = [col for col in returns_data.columns if col != "Date"]

    # Convert to numpy array
    return returns_data.select(asset_columns).to_numpy().astype(np.float64)


def _compute_covar_and_means(
    returns_array: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute covariance matrix and mean returns from a returns array.

    Parameters
    ----------
    returns_array : np.ndarray
        Returns array with shape (n_samples, n_assets).

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Tuple of (covariance_matrix, mean_returns) where covariance_matrix has
        shape (n_assets, n_assets) and mean_returns has shape (n_assets,).
    """
    covar = np.cov(returns_array.T)
    means = np.mean(returns_array, axis=0)
    # Ensure covar is 2D even for single asset
    if covar.ndim == 0:
        covar = np.array([[float(covar)]])
    return covar, means


def _build_constraints(
    asset_names: list[str],
    is_long_only: bool = True,
    min_w_scalar: float | None = None,
    max_w_scalar: float | None = None,
) -> Constraints:
    """Build an optimalportfolios Constraints object.

    Parameters
    ----------
    asset_names : list[str]
        List of asset names (used as index for pd.Series weight bounds).
    is_long_only : bool
        If True, enforce non-negative weights. Default is True.
    min_w_scalar : float | None
        Scalar minimum weight applied uniformly to all assets. Default is None.
    max_w_scalar : float | None
        Scalar maximum weight applied uniformly to all assets. Default is None.

    Returns
    -------
    Constraints
        Configured Constraints object for optimalportfolios.
    """
    min_weights: pd.Series | None = None
    max_weights: pd.Series | None = None

    if min_w_scalar is not None:
        min_weights = pd.Series([min_w_scalar] * len(asset_names), index=asset_names)

    if max_w_scalar is not None:
        max_weights = pd.Series([max_w_scalar] * len(asset_names), index=asset_names)

    return Constraints(
        is_long_only=is_long_only,
        min_weights=min_weights,  # pyright: ignore[reportArgumentType]
        max_weights=max_weights,  # pyright: ignore[reportArgumentType]
    )


def _extract_weights_to_portfolio_qwim(
    weights: np.ndarray,
    asset_names: list[str],
    portfolio_name: str,
    optimization_date: str | datetime | None = None,
) -> portfolio_QWIM:
    """Extract weights from numpy array and create portfolio_QWIM object.

    Parameters
    ----------
    weights : np.ndarray
        NumPy array of portfolio weights with shape (n_assets,).
    asset_names : list[str]
        List of asset names corresponding to weights.
    portfolio_name : str
        Name for the portfolio.
    optimization_date : str | datetime | None
        Date for the portfolio weights. Defaults to current UTC date.

    Returns
    -------
    portfolio_QWIM
        Portfolio object with the optimized weights.
    """
    if optimization_date is None:
        opt_date = datetime.now(UTC).strftime("%Y-%m-%d")
    elif isinstance(optimization_date, datetime):
        opt_date = optimization_date.strftime("%Y-%m-%d")
    else:
        opt_date = str(optimization_date)

    # Build weights DataFrame
    weights_dict: dict[str, list[object]] = {"Date": [opt_date]}
    for asset_name, weight in zip(asset_names, weights, strict=False):
        weights_dict[asset_name] = [float(weight)]

    weights_df = pl.DataFrame(weights_dict)

    # Create portfolio_QWIM and set optimized weights directly (skfolio pattern)
    port = portfolio_QWIM(
        name_portfolio=portfolio_name,
        names_components=asset_names,
        date_portfolio=opt_date,
    )
    port.m_portfolio_weights = weights_df

    return port


# =============================================================================
# Minimum Variance Portfolio
# =============================================================================


def calc_optimalportfolios_minimum_variance(
    returns_data: pl.DataFrame,
    portfolio_name: str | None = None,
    optimization_date: str | datetime | None = None,
    is_long_only: bool = True,
) -> portfolio_QWIM:
    r"""Calculate minimum variance portfolio using optimalportfolios package.

    The minimum variance portfolio minimizes portfolio variance subject to
    the constraint that weights sum to one. This is the leftmost point on
    the efficient frontier.

    Args:
        returns_data: Polars DataFrame with 'Date' column and asset return columns.
            Returns should be in decimal form (e.g., 0.01 for 1% return).
        portfolio_name: Name for the resulting portfolio. Defaults to "Min Variance Portfolio".
        optimization_date: Date for the portfolio weights. Defaults to current date.
        is_long_only: If True, constrains weights to be non-negative (no short selling).
            Default is True.

    Returns
    -------
        portfolio_QWIM: Portfolio object containing optimized weights.

    Raises
    ------
        ValueError: If returns_data is invalid.

    Notes
    -----
        The minimum variance problem is formulated as:

        $$
        \min_w \; w^T \Sigma w
        $$

        Subject to:

        $$
        \sum_i w_i = 1
        $$

        If long-only constraint:

        $$
        w_i \geq 0 \; \forall i
        $$

        Where $\Sigma$ is the covariance matrix of returns.

    References
    ----------
        - Markowitz, H. (1952). Portfolio Selection. Journal of Finance.
        - Clarke, R., De Silva, H., & Thorley, S. (2006). Minimum-Variance
          Portfolios in the U.S. Equity Market.

    Examples
    --------
        Long-only minimum variance portfolio:

        ```python
        portfolio = calc_optimalportfolios_minimum_variance(
            returns_data=returns, portfolio_name="Min Variance Long-Only"
        )
        ```

        Allow short selling:

        ```python
        portfolio = calc_optimalportfolios_minimum_variance(
            returns_data=returns, portfolio_name="Min Variance Unconstrained", is_long_only=False
        )
        ```
    """
    is_valid, error_msg = _validate_returns_data(returns_data)
    if not is_valid:
        logger.error("Invalid returns_data: %s", error_msg)
        raise ValueError(error_msg)

    if portfolio_name is None:
        portfolio_name = "Min Variance Portfolio"

    asset_names = [col for col in returns_data.columns if col != "Date"]
    returns_array = _convert_polars_to_numpy_returns(returns_data)
    covar, _ = _compute_covar_and_means(returns_array)
    constraints = _build_constraints(asset_names, is_long_only=is_long_only)

    logger.info(
        "Calculating minimum variance portfolio",
        extra={"num_assets": len(asset_names), "is_long_only": is_long_only},
    )

    try:
        weights = cvx_quadratic_optimisation(
            portfolio_objective=PortfolioObjective.MIN_VARIANCE,
            covar=covar,
            constraints=constraints,
        )

        logger.info(
            "Optimization complete",
            extra={"portfolio_name": portfolio_name, "weights_sum": float(np.sum(weights))},
        )

        return _extract_weights_to_portfolio_qwim(
            weights=weights,
            asset_names=asset_names,
            portfolio_name=portfolio_name,
            optimization_date=optimization_date,
        )

    except Exception as e:
        logger.error("Minimum variance optimization failed: %s", str(e))
        raise ValueError(f"Optimization failed: {e!s}") from e


# =============================================================================
# Maximum Quadratic Utility Portfolio
# =============================================================================


def calc_optimalportfolios_maximum_quadratic_utility(
    returns_data: pl.DataFrame,
    portfolio_name: str | None = None,
    optimization_date: str | datetime | None = None,
    risk_aversion: float = 1.0,
    is_long_only: bool = True,
) -> portfolio_QWIM:
    r"""Calculate maximum quadratic utility portfolio using optimalportfolios package.

    Maximizes expected utility with quadratic utility function, balancing expected
    return against portfolio variance scaled by risk aversion.

    Args:
        returns_data: Polars DataFrame with 'Date' column and asset return columns.
        portfolio_name: Name for the resulting portfolio. Defaults to "Max Quadratic Utility".
        optimization_date: Date for the portfolio weights. Defaults to current date.
        risk_aversion: Risk aversion coefficient. Higher values lead to more conservative
            portfolios. Default is 1.0.
        is_long_only: If True, constrains weights to be non-negative. Default is True.

    Returns
    -------
        portfolio_QWIM: Portfolio object containing optimized weights.

    Raises
    ------
        ValueError: If returns_data is invalid or risk_aversion is non-positive.

    Notes
    -----
        The quadratic utility maximization problem:

        $$
        \max_w \; \mu^T w - \frac{\lambda}{2} w^T \Sigma w
        $$

        Subject to:

        $$
        \sum_i w_i = 1, \quad w_i \geq 0 \; \text{(if long-only)}
        $$

        Where:
        - $\mu$ is the vector of expected returns
        - $\Sigma$ is the covariance matrix
        - $\lambda$ is the risk aversion coefficient

    References
    ----------
        - Markowitz, H. (1952). Portfolio Selection. Journal of Finance.
        - Merton, R. C. (1972). An Analytic Derivation of the Efficient Portfolio Frontier.

    Examples
    --------
        Conservative investor (high risk aversion):

        ```python
        portfolio = calc_optimalportfolios_maximum_quadratic_utility(
            returns_data=returns, risk_aversion=5.0, portfolio_name="Conservative Portfolio"
        )
        ```

        Aggressive investor (low risk aversion):

        ```python
        portfolio = calc_optimalportfolios_maximum_quadratic_utility(
            returns_data=returns, risk_aversion=0.5, portfolio_name="Aggressive Portfolio"
        )
        ```
    """
    is_valid, error_msg = _validate_returns_data(returns_data)
    if not is_valid:
        logger.error("Invalid returns_data: %s", error_msg)
        raise ValueError(error_msg)

    if risk_aversion <= 0:
        logger.error("risk_aversion must be positive, got %s", risk_aversion)
        raise ValueError(f"risk_aversion must be positive, got {risk_aversion}")

    if portfolio_name is None:
        portfolio_name = "Max Quadratic Utility Portfolio"

    asset_names = [col for col in returns_data.columns if col != "Date"]
    returns_array = _convert_polars_to_numpy_returns(returns_data)
    covar, means = _compute_covar_and_means(returns_array)
    constraints = _build_constraints(asset_names, is_long_only=is_long_only)

    logger.info(
        "Calculating maximum quadratic utility portfolio",
        extra={
            "num_assets": len(asset_names),
            "risk_aversion": risk_aversion,
            "is_long_only": is_long_only,
        },
    )

    try:
        weights = cvx_quadratic_optimisation(
            portfolio_objective=PortfolioObjective.QUADRATIC_UTILITY,
            covar=covar,
            constraints=constraints,
            means=means,
            carra=risk_aversion,
        )

        logger.info(
            "Optimization complete",
            extra={"portfolio_name": portfolio_name, "weights_sum": float(np.sum(weights))},
        )

        return _extract_weights_to_portfolio_qwim(
            weights=weights,
            asset_names=asset_names,
            portfolio_name=portfolio_name,
            optimization_date=optimization_date,
        )

    except Exception as e:
        logger.error("Maximum quadratic utility optimization failed: %s", str(e))
        raise ValueError(f"Optimization failed: {e!s}") from e


# =============================================================================
# Budgeted Risk Contribution (Risk Parity) Portfolio
# =============================================================================


def calc_optimalportfolios_budgeted_risk_contribution(
    returns_data: pl.DataFrame,
    portfolio_name: str | None = None,
    optimization_date: str | datetime | None = None,
    risk_budgets: np.ndarray | None = None,
) -> portfolio_QWIM:
    r"""Calculate budgeted risk contribution portfolio (risk parity) using optimalportfolios.

    Allocates portfolio risk according to specified risk budgets. When all budgets
    are equal, this produces the equal risk contribution (risk parity) portfolio.

    Args:
        returns_data: Polars DataFrame with 'Date' column and asset return columns.
        portfolio_name: Name for the resulting portfolio. Defaults to "Risk Parity Portfolio".
        optimization_date: Date for the portfolio weights. Defaults to current date.
        risk_budgets: Array of risk budget allocations for each asset. Must sum to 1.0.
            If None, uses equal budgets (risk parity). Default is None.

    Returns
    -------
        portfolio_QWIM: Portfolio object containing optimized weights.

    Raises
    ------
        ValueError: If returns_data is invalid or risk_budgets don't sum to 1.

    Notes
    -----
        The risk contribution of asset $i$ is:

        $$
        RC_i = w_i \frac{\partial \sigma_p}{\partial w_i} = w_i \frac{(\Sigma w)_i}{\sigma_p}
        $$

        The optimization problem minimizes:

        $$
        \min_w \sum_i \left( RC_i - b_i \sigma_p \right)^2
        $$

        Where $b_i$ are the risk budgets with $\sum_i b_i = 1$.

        For equal risk contribution (risk parity), set $b_i = 1/N$ for all assets.

    References
    ----------
        - Maillard, S., Roncalli, T., & Teïletche, J. (2010). The Properties of
          Equally Weighted Risk Contribution Portfolios.
        - Roncalli, T. (2013). Introduction to Risk Parity and Budgeting.

    Examples
    --------
        Equal risk contribution (risk parity):

        ```python
        portfolio = calc_optimalportfolios_budgeted_risk_contribution(
            returns_data=returns, portfolio_name="Risk Parity Portfolio"
        )
        ```

        Custom risk budgets (60% equity, 40% bonds):

        ```python
        risk_budgets = np.array([0.60, 0.40])
        portfolio = calc_optimalportfolios_budgeted_risk_contribution(
            returns_data=returns, risk_budgets=risk_budgets, portfolio_name="60/40 Risk Budget"
        )
        ```
    """
    is_valid, error_msg = _validate_returns_data(returns_data)
    if not is_valid:
        logger.error("Invalid returns_data: %s", error_msg)
        raise ValueError(error_msg)

    asset_names = [col for col in returns_data.columns if col != "Date"]
    n_assets = len(asset_names)

    if risk_budgets is not None:
        # Support dict input: {asset_name: budget, ...} → ordered numpy array
        if isinstance(risk_budgets, dict):
            missing = [a for a in asset_names if a not in risk_budgets]
            if missing:
                msg = f"risk_budgets dict missing keys for assets: {missing}"
                logger.error(msg)
                raise ValueError(msg)
            risk_budgets = np.array([risk_budgets[a] for a in asset_names], dtype=float)
        if len(risk_budgets) != n_assets:
            msg = f"risk_budgets length ({len(risk_budgets)}) must match n_assets ({n_assets})"
            logger.error(msg)
            raise ValueError(msg)
        budget_sum = float(np.sum(risk_budgets))
        if not np.isclose(budget_sum, 1.0, atol=1e-6):
            msg = f"risk_budgets must sum to 1.0, got {budget_sum}"
            logger.error(msg)
            raise ValueError(msg)

    if portfolio_name is None:
        portfolio_name = (
            "Risk Parity Portfolio" if risk_budgets is None else "Risk Budget Portfolio"
        )

    returns_array = _convert_polars_to_numpy_returns(returns_data)
    covar, _ = _compute_covar_and_means(returns_array)
    constraints = _build_constraints(asset_names, is_long_only=True)

    logger.info(
        "Calculating budgeted risk contribution portfolio",
        extra={"num_assets": n_assets, "equal_risk_contribution": risk_budgets is None},
    )

    try:
        # Pass risk_budget only when explicitly provided; the package default (None)
        # creates equal budgets.  Splitting the call avoids a pyright false positive
        # caused by the library annotating "risk_budget: np.ndarray = None" instead
        # of "Optional[np.ndarray] = None".
        if risk_budgets is not None:
            weights = opt_risk_budgeting(
                covar=covar,
                constraints=constraints,
                risk_budget=risk_budgets,
            )
        else:
            weights = opt_risk_budgeting(
                covar=covar,
                constraints=constraints,
            )

        logger.info(
            "Optimization complete",
            extra={"portfolio_name": portfolio_name, "weights_sum": float(np.sum(weights))},
        )

        return _extract_weights_to_portfolio_qwim(
            weights=weights,
            asset_names=asset_names,
            portfolio_name=portfolio_name,
            optimization_date=optimization_date,
        )

    except Exception as e:
        logger.error("Budgeted risk contribution optimization failed: %s", str(e))
        raise ValueError(f"Optimization failed: {e!s}") from e


# =============================================================================
# Maximum Diversification Portfolio
# =============================================================================


def calc_optimalportfolios_maximum_diversification(
    returns_data: pl.DataFrame,
    portfolio_name: str | None = None,
    optimization_date: str | datetime | None = None,
    is_long_only: bool = True,
) -> portfolio_QWIM:
    r"""Calculate maximum diversification portfolio using optimalportfolios package.

    Maximizes the diversification ratio, which is the ratio of weighted average
    volatility to portfolio volatility. This emphasizes diversification benefits.

    Args:
        returns_data: Polars DataFrame with 'Date' column and asset return columns.
        portfolio_name: Name for the resulting portfolio. Defaults to "Max Diversification".
        optimization_date: Date for the portfolio weights. Defaults to current date.
        is_long_only: If True, constrains weights to be non-negative. Default is True.

    Returns
    -------
        portfolio_QWIM: Portfolio object containing optimized weights.

    Raises
    ------
        ValueError: If returns_data is invalid.

    Notes
    -----
        The diversification ratio is defined as:

        $$
        DR = \frac{w^T \sigma}{\sqrt{w^T \Sigma w}}
        $$

        Where:
        - $\sigma$ is the vector of individual asset volatilities
        - $\Sigma$ is the covariance matrix
        - $w$ are the portfolio weights

        The optimization problem:

        $$
        \max_w \; DR(w) = \max_w \; \frac{w^T \sigma}{\sqrt{w^T \Sigma w}}
        $$

        Subject to:

        $$
        \sum_i w_i = 1, \quad w_i \geq 0 \; \text{(if long-only)}
        $$

    References
    ----------
        - Choueifaty, Y., & Coignard, Y. (2008). Toward Maximum Diversification.
          Journal of Portfolio Management.
        - Choueifaty, Y., Froidure, T., & Reynier, J. (2013). Properties of the
          Most Diversified Portfolio.

    Examples
    --------
        Long-only maximum diversification:

        ```python
        portfolio = calc_optimalportfolios_maximum_diversification(
            returns_data=returns, portfolio_name="Max Diversification Long-Only"
        )
        ```

        Allow short selling:

        ```python
        portfolio = calc_optimalportfolios_maximum_diversification(
            returns_data=returns,
            is_long_only=False,
            portfolio_name="Max Diversification Unconstrained",
        )
        ```
    """
    is_valid, error_msg = _validate_returns_data(returns_data)
    if not is_valid:
        logger.error("Invalid returns_data: %s", error_msg)
        raise ValueError(error_msg)

    if portfolio_name is None:
        portfolio_name = "Max Diversification Portfolio"

    asset_names = [col for col in returns_data.columns if col != "Date"]
    returns_array = _convert_polars_to_numpy_returns(returns_data)
    covar, _ = _compute_covar_and_means(returns_array)
    constraints = _build_constraints(asset_names, is_long_only=is_long_only)

    logger.info(
        "Calculating maximum diversification portfolio",
        extra={"num_assets": len(asset_names), "is_long_only": is_long_only},
    )

    try:
        weights = opt_maximise_diversification(
            covar=covar,
            constraints=constraints,
        )

        logger.info(
            "Optimization complete",
            extra={"portfolio_name": portfolio_name, "weights_sum": float(np.sum(weights))},
        )

        return _extract_weights_to_portfolio_qwim(
            weights=weights,
            asset_names=asset_names,
            portfolio_name=portfolio_name,
            optimization_date=optimization_date,
        )

    except Exception as e:
        logger.error("Maximum diversification optimization failed: %s", str(e))
        raise ValueError(f"Optimization failed: {e!s}") from e


# =============================================================================
# Maximum Sharpe Ratio Portfolio
# =============================================================================


def calc_optimalportfolios_maximum_sharpe_ratio(
    returns_data: pl.DataFrame,
    portfolio_name: str | None = None,
    optimization_date: str | datetime | None = None,
    risk_free_rate: float = 0.0,
    is_long_only: bool = True,
) -> portfolio_QWIM:
    r"""Calculate maximum Sharpe ratio portfolio using optimalportfolios package.

    Maximizes the Sharpe ratio, which measures risk-adjusted return as the ratio
    of excess return to volatility.

    Args:
        returns_data: Polars DataFrame with 'Date' column and asset return columns.
        portfolio_name: Name for the resulting portfolio. Defaults to "Max Sharpe Ratio".
        optimization_date: Date for the portfolio weights. Defaults to current date.
        risk_free_rate: Risk-free rate for calculating excess returns. Should match
            the periodicity of returns_data. Default is 0.0.
        is_long_only: If True, constrains weights to be non-negative. Default is True.

    Returns
    -------
        portfolio_QWIM: Portfolio object containing optimized weights.

    Raises
    ------
        ValueError: If returns_data is invalid.

    Notes
    -----
        The Sharpe ratio is defined as:

        $$
        SR = \frac{\mu_p - r_f}{\sigma_p} = \frac{w^T \mu - r_f}{\sqrt{w^T \Sigma w}}
        $$

        Where:
        - $\mu_p$ is the portfolio expected return
        - $r_f$ is the risk-free rate
        - $\sigma_p$ is the portfolio volatility

        The optimization problem:

        $$
        \max_w \; \frac{w^T \mu - r_f}{\sqrt{w^T \Sigma w}}
        $$

        Subject to:

        $$
        \sum_i w_i = 1, \quad w_i \geq 0 \; \text{(if long-only)}
        $$

        This is the tangency portfolio on the efficient frontier.

    References
    ----------
        - Sharpe, W. F. (1966). Mutual Fund Performance. Journal of Business.
        - Sharpe, W. F. (1994). The Sharpe Ratio. Journal of Portfolio Management.

    Examples
    --------
        Maximum Sharpe ratio with zero risk-free rate:

        ```python
        portfolio = calc_optimalportfolios_maximum_sharpe_ratio(
            returns_data=returns, portfolio_name="Max Sharpe Portfolio"
        )
        ```

        With 2% annual risk-free rate (monthly returns):

        ```python
        risk_free_rate_monthly = 0.02 / 12  # Convert annual to monthly
        portfolio = calc_optimalportfolios_maximum_sharpe_ratio(
            returns_data=returns,
            risk_free_rate=risk_free_rate_monthly,
            portfolio_name="Max Sharpe with Risk-Free Rate",
        )
        ```
    """
    is_valid, error_msg = _validate_returns_data(returns_data)
    if not is_valid:
        logger.error("Invalid returns_data: %s", error_msg)
        raise ValueError(error_msg)

    if portfolio_name is None:
        portfolio_name = "Max Sharpe Ratio Portfolio"

    asset_names = [col for col in returns_data.columns if col != "Date"]
    returns_array = _convert_polars_to_numpy_returns(returns_data)
    covar, means = _compute_covar_and_means(returns_array)
    excess_means = means - risk_free_rate
    constraints = _build_constraints(asset_names, is_long_only=is_long_only)

    logger.info(
        "Calculating maximum Sharpe ratio portfolio",
        extra={
            "num_assets": len(asset_names),
            "risk_free_rate": risk_free_rate,
            "is_long_only": is_long_only,
        },
    )

    try:
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="Solution may be inaccurate",
                category=UserWarning,
                module="optimalportfolios",
            )
            weights = cvx_maximize_portfolio_sharpe(
                covar=covar,
                means=excess_means,
                constraints=constraints,
            )

        logger.info(
            "Optimization complete",
            extra={"portfolio_name": portfolio_name, "weights_sum": float(np.sum(weights))},
        )

        return _extract_weights_to_portfolio_qwim(
            weights=weights,
            asset_names=asset_names,
            portfolio_name=portfolio_name,
            optimization_date=optimization_date,
        )

    except Exception as e:
        logger.error("Maximum Sharpe ratio optimization failed: %s", str(e))
        raise ValueError(f"Optimization failed: {e!s}") from e


# =============================================================================
# Maximum CARA Utility under Gaussian Mixture Model
# =============================================================================


def calc_optimalportfolios_maximum_cara_gaussian_mixture(
    returns_data: pl.DataFrame,
    portfolio_name: str | None = None,
    optimization_date: str | datetime | None = None,
    risk_aversion: float = 1.0,
    n_components: int = 2,
    is_long_only: bool = True,
) -> portfolio_QWIM:
    r"""Calculate maximum CARA utility portfolio under Gaussian mixture model.

    Maximizes constant absolute risk aversion (CARA) utility assuming returns
    follow a Gaussian mixture model. This allows for non-normal return distributions
    with multiple regimes.

    Args:
        returns_data: Polars DataFrame with 'Date' column and asset return columns.
        portfolio_name: Name for the resulting portfolio. Defaults to "Max CARA GMM".
        optimization_date: Date for the portfolio weights. Defaults to current date.
        risk_aversion: Risk aversion coefficient. Higher values lead to more
            conservative portfolios. Default is 1.0.
        n_components: Number of components in the Gaussian mixture model.
            Default is 2 (two-regime model).
        is_long_only: If True, constrains weights to be non-negative. Default is True.

    Returns
    -------
        portfolio_QWIM: Portfolio object containing optimized weights.

    Raises
    ------
        ValueError: If returns_data is invalid, risk_aversion is non-positive,
            or n_components < 1.

    Notes
    -----
        CARA utility function:

        $$
        U(W) = -\frac{1}{\lambda} e^{-\lambda W}
        $$

        Where $\lambda$ is the risk aversion coefficient and $W$ is wealth.

        Under Gaussian mixture model, returns follow:

        $$
        r \sim \sum_{k=1}^K \pi_k \mathcal{N}(\mu_k, \Sigma_k)
        $$

        Where:
        - $K$ is the number of components (regimes)
        - $\pi_k$ are the mixing weights
        - $\mu_k, \Sigma_k$ are mean and covariance for regime $k$

        This approach better captures fat tails and regime shifts in returns.

    References
    ----------
        - Arrow, K. J. (1965). Aspects of the Theory of Risk-Bearing.
        - Alexander, C., & Baptista, A. (2002). Economic Implications of Using
          a Mean-VaR Model for Portfolio Selection.
        - McLachlan, G. J., & Peel, D. (2000). Finite Mixture Models.

    Examples
    --------
        Two-regime model (e.g., bull/bear markets):

        ```python
        portfolio = calc_optimalportfolios_maximum_cara_gaussian_mixture(
            returns_data=returns,
            risk_aversion=2.0,
            n_components=2,
            portfolio_name="CARA GMM Bull-Bear",
        )
        ```

        Three-regime model with higher risk aversion:

        ```python
        portfolio = calc_optimalportfolios_maximum_cara_gaussian_mixture(
            returns_data=returns,
            risk_aversion=5.0,
            n_components=3,
            portfolio_name="Conservative CARA GMM",
        )
        ```
    """
    is_valid, error_msg = _validate_returns_data(returns_data)
    if not is_valid:
        logger.error("Invalid returns_data: %s", error_msg)
        raise ValueError(error_msg)

    if risk_aversion <= 0:
        msg = f"risk_aversion must be positive, got {risk_aversion}"
        logger.error(msg)
        raise ValueError(msg)

    if n_components < 1:
        msg = f"n_components must be at least 1, got {n_components}"
        logger.error(msg)
        raise ValueError(msg)

    if portfolio_name is None:
        portfolio_name = "Max CARA GMM Portfolio"

    asset_names = [col for col in returns_data.columns if col != "Date"]
    returns_array = _convert_polars_to_numpy_returns(returns_data)
    constraints = _build_constraints(asset_names, is_long_only=is_long_only)

    logger.info(
        "Calculating maximum CARA utility (GMM) portfolio",
        extra={
            "num_assets": len(asset_names),
            "risk_aversion": risk_aversion,
            "n_components": n_components,
            "is_long_only": is_long_only,
        },
    )

    try:
        # fit_gaussian_mixture returns a Params object with .means, .covars, .probs
        gmm_params = fit_gaussian_mixture(x=returns_array, n_components=n_components)

        weights = opt_maximize_cara_mixture(
            means=gmm_params.means,
            covars=gmm_params.covars,
            probs=gmm_params.probs,
            constraints=constraints,
            carra=risk_aversion,
        )

        logger.info(
            "Optimization complete",
            extra={"portfolio_name": portfolio_name, "weights_sum": float(np.sum(weights))},
        )

        return _extract_weights_to_portfolio_qwim(
            weights=weights,
            asset_names=asset_names,
            portfolio_name=portfolio_name,
            optimization_date=optimization_date,
        )

    except Exception as e:
        logger.error("Maximum CARA GMM optimization failed: %s", str(e))
        raise ValueError(f"Optimization failed: {e!s}") from e


# =============================================================================
# Tracking Error Minimization Portfolio
# =============================================================================


def calc_optimalportfolios_tracking_error_minimization(
    returns_data: pl.DataFrame,
    benchmark_returns: pl.DataFrame,
    portfolio_name: str | None = None,
    optimization_date: str | datetime | None = None,
    is_long_only: bool = True,
) -> portfolio_QWIM:
    r"""Calculate tracking error minimization portfolio using optimalportfolios package.

    Minimizes tracking error (deviation from benchmark) to create a portfolio that
    closely replicates benchmark performance. Useful for index tracking and
    enhanced indexing strategies.

    Args:
        returns_data: Polars DataFrame with 'Date' column and asset return columns.
        benchmark_returns: Polars DataFrame with 'Date' column and benchmark return column.
            Must have same date range as returns_data.
        portfolio_name: Name for the resulting portfolio. Defaults to "Tracking Error Min".
        optimization_date: Date for the portfolio weights. Defaults to current date.
        is_long_only: If True, constrains weights to be non-negative. Default is True.

    Returns
    -------
        portfolio_QWIM: Portfolio object containing optimized weights.

    Raises
    ------
        ValueError: If returns_data or benchmark_returns are invalid, or if date ranges
            don't match.

    Notes
    -----
        Tracking error is the standard deviation of excess returns:

        $$
        TE = \sqrt{\text{Var}(r_p - r_b)} = \sqrt{(w - w_b)^T \Sigma (w - w_b)}
        $$

        Where:
        - $r_p$ is the portfolio return
        - $r_b$ is the benchmark return
        - $w$ are portfolio weights
        - $w_b$ are benchmark weights

        The optimization problem:

        $$
        \min_w \; (w - w_b)^T \Sigma (w - w_b)
        $$

        Subject to:

        $$
        \sum_i w_i = 1, \quad w_i \geq 0 \; \text{(if long-only)}
        $$

        This creates a portfolio that minimally deviates from the benchmark while
        potentially using fewer holdings or satisfying constraints.

    References
    ----------
        - Jorion, P. (2003). Portfolio Optimization with Tracking-Error Constraints.
          Financial Analysts Journal.
        - Roll, R. (1992). A Mean/Variance Analysis of Tracking Error.
          Journal of Portfolio Management.

    Examples
    --------
        Track an index with subset of stocks:

        ```python
        portfolio = calc_optimalportfolios_tracking_error_minimization(
            returns_data=stock_returns,  # 50 stocks
            benchmark_returns=index_returns,  # S&P 500
            portfolio_name="Index Tracker",
        )
        ```

        Enhanced indexing with short selling allowed:

        ```python
        portfolio = calc_optimalportfolios_tracking_error_minimization(
            returns_data=returns,
            benchmark_returns=benchmark,
            is_long_only=False,
            portfolio_name="Enhanced Index",
        )
        ```
    """
    is_valid, error_msg = _validate_returns_data(returns_data)
    if not is_valid:
        logger.error("Invalid returns_data: %s", error_msg)
        raise ValueError(error_msg)

    is_valid_bench, error_msg_bench = _validate_returns_data(benchmark_returns)
    if not is_valid_bench:
        logger.error("Invalid benchmark_returns: %s", error_msg_bench)
        raise ValueError(f"Invalid benchmark_returns: {error_msg_bench}")

    benchmark_columns = [col for col in benchmark_returns.columns if col != "Date"]
    if len(benchmark_columns) != 1:
        msg = f"benchmark_returns must have exactly one asset column, got {len(benchmark_columns)}"
        logger.error(msg)
        raise ValueError(msg)

    if len(returns_data) != len(benchmark_returns):
        msg = (
            f"returns_data ({len(returns_data)} rows) and benchmark_returns "
            f"({len(benchmark_returns)} rows) must have the same length"
        )
        logger.error(msg)
        raise ValueError(msg)

    if portfolio_name is None:
        portfolio_name = "Tracking Error Min Portfolio"

    asset_names = [col for col in returns_data.columns if col != "Date"]
    n_assets = len(asset_names)
    returns_array = _convert_polars_to_numpy_returns(returns_data)
    benchmark_array = benchmark_returns.select(benchmark_columns[0]).to_numpy().flatten()

    logger.info(
        "Calculating tracking error minimization portfolio (cvxpy)",
        extra={
            "num_assets": n_assets,
            "benchmark_col": benchmark_columns[0],
            "is_long_only": is_long_only,
        },
    )

    try:
        covar, _ = _compute_covar_and_means(returns_array)
        cov_combined = np.cov(returns_array.T, benchmark_array)
        cov_with_benchmark = cov_combined[:n_assets, -1]

        w_var = cp.Variable(n_assets)
        objective = cp.Minimize(cp.quad_form(w_var, covar) - 2.0 * cov_with_benchmark @ w_var)
        cvx_constraints = [cp.sum(w_var) == 1.0]
        if is_long_only:
            cvx_constraints.append(w_var >= 0)  # pyright: ignore[reportArgumentType]

        prob = cp.Problem(objective, cvx_constraints)  # pyright: ignore[reportArgumentType]
        prob.solve(solver=cp.ECOS)

        if prob.status not in ("optimal", "optimal_inaccurate"):
            raise RuntimeError(f"cvxpy solver failed with status: {prob.status}")

        weights = np.asarray(w_var.value, dtype=np.float64)

        logger.info(
            "Optimization complete",
            extra={"portfolio_name": portfolio_name, "weights_sum": float(np.sum(weights))},
        )

        return _extract_weights_to_portfolio_qwim(
            weights=weights,
            asset_names=asset_names,
            portfolio_name=portfolio_name,
            optimization_date=optimization_date,
        )

    except Exception as e:
        logger.error("Tracking error minimization failed: %s", str(e))
        raise ValueError(f"Optimization failed: {e!s}") from e
