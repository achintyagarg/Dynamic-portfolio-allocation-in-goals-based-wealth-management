"""Portfolio optimization methods from azapy Python package.

This module provides wrapper functions for portfolio optimization methods from the
azapy package. All functions accept Polars DataFrames as input and return
portfolio_QWIM objects as output.

The azapy package implements portfolio risk measures and associated optimization:
- Mean-Variance (Markowitz) optimization
- Conditional Value at Risk (CVaR / Expected Shortfall)
- Mean Absolute Deviation (MAD)
- Inverse Volatility weighting
- Kelly criterion optimization
- Entropic Value at Risk (EVaR)

Notes
-----
    azapy analyzers use a pandas-DataFrame-based API. The workflow is:
    1. Create analyzer with desired risk measure and optimization type.
    2. Call ``analyzer.set_rrate(returns_df)`` with a pandas DataFrame of returns.
    3. Call ``analyzer.getWeights(rtype=...)`` to obtain weights as ``pd.Series``.

    Input Polars DataFrames are automatically converted to pandas for compatibility.

References
----------
    - azapy documentation: https://azapy.readthedocs.io/
    - Markowitz, H. (1952). Portfolio Selection. Journal of Finance.
    - Rockafellar, R. T., & Uryasev, S. (2000). Optimization of Conditional
      Value-at-Risk. Journal of Risk.

Examples
--------
    Mean-variance minimum risk portfolio:

    ```python
    import polars as pl
    from src.models.portfolio_optimization.pkg_azapy import calc_azapy_mean_variance

    returns = pl.DataFrame(
        {
            "Date": ["2024-01-01", "2024-01-02", "2024-01-03"],
            "AAPL": [0.01, -0.005, 0.02],
            "MSFT": [0.015, 0.01, -0.01],
            "GOOG": [0.005, 0.02, 0.015],
        }
    )

    portfolio = calc_azapy_mean_variance(
        returns_data=returns, rtype="MinRisk", portfolio_name="MV Min Risk Portfolio"
    )
    ```
"""

from __future__ import annotations

import logging
import warnings

from datetime import UTC, datetime
from typing import Literal

import azapy
import pandas as pd  # noqa: TC002
import polars as pl

from src.portfolios.portfolio_QWIM import portfolio_QWIM


# Configure logging
logger = logging.getLogger(__name__)

# Valid rtype values for multi-objective analyzers (MV, CVaR, MAD, EVaR)
RTYPE_OPTIONS = Literal["Sharpe", "MinRisk", "InvNrisk", "RiskAverse", "MaxRet"]


# =============================================================================
# Helper Functions
# =============================================================================


def _validate_returns_data(returns_data: pl.DataFrame) -> tuple[bool, str]:
    """Validate returns DataFrame structure and content.

    Defensive validation following project standards with early returns.

    Parameters
    ----------
    returns_data : pl.DataFrame
        Polars DataFrame to validate.

    Returns
    -------
    tuple[bool, str]
        Tuple of (is_valid, error_message).
        If valid, returns (True, ""). If invalid, returns (False, error_description).
    """
    if returns_data is None:
        return False, "returns_data cannot be None"

    if not isinstance(returns_data, pl.DataFrame):
        return False, f"returns_data must be Polars DataFrame, got {type(returns_data)}"

    if returns_data.is_empty():
        return False, "returns_data cannot be empty"

    if "Date" not in returns_data.columns:
        return False, "returns_data must have 'Date' column"

    asset_columns = [col for col in returns_data.columns if col != "Date"]
    if len(asset_columns) == 0:
        return False, "returns_data must have at least one asset column besides 'Date'"

    for col in asset_columns:
        if returns_data[col].dtype not in [pl.Float32, pl.Float64, pl.Int32, pl.Int64]:
            return False, f"Asset column '{col}' must be numeric, got {returns_data[col].dtype}"

    return True, ""


def _convert_polars_to_pandas_returns(returns_data: pl.DataFrame) -> pd.DataFrame:
    """Convert Polars returns DataFrame to pandas for azapy compatibility.

    azapy analyzers accept a pandas DataFrame where each column corresponds to
    an asset and each row is a return observation. The Date column is dropped
    since azapy only requires the numeric return values.

    Parameters
    ----------
    returns_data : pl.DataFrame
        Polars DataFrame with 'Date' column and asset return columns.

    Returns
    -------
    pd.DataFrame
        Pandas DataFrame with only the asset return columns (no date index).
    """
    asset_columns = [col for col in returns_data.columns if col != "Date"]
    return returns_data.select(asset_columns).to_pandas().astype(float)


def _extract_weights_from_azapy(
    weights_series: pd.Series,
    portfolio_name: str,
    optimization_date: str | datetime | None = None,
) -> portfolio_QWIM:
    """Convert azapy weights pd.Series to portfolio_QWIM object.

    Parameters
    ----------
    weights_series : pd.Series
        Pandas Series of portfolio weights indexed by asset names,
        as returned by ``analyzer.getWeights()``.
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

    asset_names = list(weights_series.index)
    weights_values = weights_series.to_numpy().tolist()

    # Build weights DataFrame
    weights_dict: dict[str, list[object]] = {"Date": [opt_date]}
    for asset_name, weight in zip(asset_names, weights_values, strict=False):
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
# Mean-Variance (Markowitz) Portfolio
# =============================================================================


def calc_azapy_mean_variance(
    returns_data: pl.DataFrame,
    rtype: str = "MinRisk",
    aversion: float | None = None,
    portfolio_name: str | None = None,
    optimization_date: str | datetime | None = None,
) -> portfolio_QWIM:
    r"""Calculate mean-variance portfolio using azapy MVAnalyzer.

    Implements Markowitz mean-variance optimization with various objective
    function types selectable via the ``rtype`` parameter.

    Parameters
    ----------
    returns_data : pl.DataFrame
        Polars DataFrame with 'Date' column and asset return columns.
        Returns should be in decimal form (e.g., 0.01 for 1% return).
    rtype : str
        Optimization objective type. One of:

        - ``"MinRisk"`` : Minimum variance portfolio
        - ``"Sharpe"`` : Maximum Sharpe ratio (tangency portfolio)
        - ``"InvNrisk"`` : Inverse normalized risk allocation
        - ``"RiskAverse"`` : Risk-aversion weighted portfolio
        - ``"MaxRet"`` : Maximum return portfolio

        Default is ``"MinRisk"``.
    aversion : float | None
        Risk aversion coefficient (only used when ``rtype="RiskAverse"``).
        Must be positive. Default is None.
    portfolio_name : str | None
        Name for the resulting portfolio. Defaults to a name based on rtype.
    optimization_date : str | datetime | None
        Date for the portfolio weights. Defaults to current UTC date.

    Returns
    -------
    portfolio_QWIM
        Portfolio object containing optimized weights.

    Raises
    ------
    ValueError
        If returns_data is invalid or rtype is not recognized.

    Notes
    -----
        The mean-variance problem (MinRisk):

        $$
        \min_w \; w^T \Sigma w
        \quad \text{s.t.} \quad \mathbf{1}^T w = 1,\; w \geq 0
        $$

        The Sharpe ratio maximization (tangency portfolio):

        $$
        \max_w \; \frac{\mu_p - r_f}{\sigma_p}
        = \frac{w^T \mu}{\sqrt{w^T \Sigma w}}
        $$

    References
    ----------
        - Markowitz, H. (1952). Portfolio Selection. Journal of Finance.
        - Sharpe, W. F. (1966). Mutual Fund Performance. Journal of Business.

    Examples
    --------
        Minimum variance portfolio:

        ```python
        portfolio = calc_azapy_mean_variance(
            returns_data=returns, rtype="MinRisk", portfolio_name="MV Min Risk"
        )
        ```

        Maximum Sharpe ratio portfolio:

        ```python
        portfolio = calc_azapy_mean_variance(
            returns_data=returns, rtype="Sharpe", portfolio_name="MV Tangency"
        )
        ```
    """
    is_valid, error_msg = _validate_returns_data(returns_data)
    if not is_valid:
        logger.error("Invalid returns_data: %s", error_msg)
        raise ValueError(error_msg)

    valid_rtypes = {
        "MinRisk",
        "Sharpe",
        "Sharpe2",
        "InvNrisk",
        "RiskAverse",
        "Diverse",
        "MaxDiverse",
    }
    if rtype not in valid_rtypes:
        msg = f"rtype must be one of {valid_rtypes}, got '{rtype}'"
        logger.error(msg)
        raise ValueError(msg)

    if portfolio_name is None:
        portfolio_name = f"MV {rtype} Portfolio"

    returns_pandas = _convert_polars_to_pandas_returns(returns_data)

    logger.info(
        "Calculating mean-variance portfolio",
        extra={"rtype": rtype, "num_assets": len(returns_pandas.columns), "aversion": aversion},
    )

    try:
        analyzer = azapy.MVAnalyzer(rtype=rtype, aversion=aversion)
        analyzer.set_rrate(returns_pandas)
        # Suppress azapy-internal FutureWarning from scipy.sparse: azapy's own
        # optimizers call sps.diags([-1] * n) with a Python int list, which
        # scipy casts to float64 and warns.  This is entirely within
        # azapy/scipy internals; it cannot be fixed by changing input dtype.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # azapy-internal scipy.sparse FutureWarning
            weights_series = analyzer.getWeights(rtype=rtype)

        if weights_series is None and rtype == "Sharpe":
            # 'Sharpe' uses a direct SOCP that can be numerically unstable when
            # some assets have near-zero or negative mean returns.  'Sharpe2'
            # (minimisation of inverse Sharpe) is an equivalent but more robust
            # alternative formulation.  Fall back silently so the caller still
            # receives valid weights for rtype='Sharpe' requests.
            logger.warning(
                "MVAnalyzer Sharpe SOCP failed (status %s); retrying with Sharpe2",
                getattr(analyzer, "status", "unknown"),
            )
            analyzer = azapy.MVAnalyzer(rtype="Sharpe2", aversion=aversion)
            analyzer.set_rrate(returns_pandas)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")  # azapy-internal scipy.sparse FutureWarning
                weights_series = analyzer.getWeights(rtype="Sharpe2")

        if weights_series is None:
            raise ValueError("MVAnalyzer returned None weights")

        logger.info(
            "Optimization complete",
            extra={
                "portfolio_name": portfolio_name,
                "weights_sum": float(weights_series.sum()),
            },
        )

        return _extract_weights_from_azapy(
            weights_series=weights_series,
            portfolio_name=portfolio_name,
            optimization_date=optimization_date,
        )

    except Exception as e:
        logger.error("Mean-variance optimization failed: %s", str(e))
        raise ValueError(f"Optimization failed: {e!s}") from e


# =============================================================================
# Conditional Value at Risk (CVaR) Portfolio
# =============================================================================


def calc_azapy_cvar(
    returns_data: pl.DataFrame,
    alpha: float = 0.975,
    rtype: str = "Sharpe",
    portfolio_name: str | None = None,
    optimization_date: str | datetime | None = None,
) -> portfolio_QWIM:
    r"""Calculate CVaR-optimal portfolio using azapy CVaRAnalyzer.

    Minimizes or optimizes Conditional Value at Risk (CVaR), also known as
    Expected Shortfall (ES), which measures the expected loss in the worst
    ``(1 - alpha) * 100%`` of cases.

    Parameters
    ----------
    returns_data : pl.DataFrame
        Polars DataFrame with 'Date' column and asset return columns.
    alpha : float
        Confidence level for CVaR computation. Typical values: 0.95, 0.975, 0.99.
        Default is 0.975 (97.5% confidence level).
    rtype : str
        Optimization objective type. One of ``"MinRisk"``, ``"Sharpe"``,
        ``"InvNrisk"``, ``"RiskAverse"``, ``"MaxRet"``. Default is ``"Sharpe"``.
    portfolio_name : str | None
        Name for the resulting portfolio. Defaults to a name based on rtype.
    optimization_date : str | datetime | None
        Date for the portfolio weights. Defaults to current UTC date.

    Returns
    -------
    portfolio_QWIM
        Portfolio object containing optimized weights.

    Raises
    ------
    ValueError
        If returns_data is invalid or alpha is out of range.

    Notes
    -----
        CVaR at confidence level $\alpha$ is defined as:

        $$
        CVaR_\alpha = \frac{1}{1-\alpha} \int_\alpha^1 VaR_u \, du
        $$

        Equivalently, it is the expected loss given that the loss exceeds $VaR_\alpha$:

        $$
        CVaR_\alpha = E[L \mid L \geq VaR_\alpha]
        $$

    References
    ----------
        - Rockafellar, R. T., & Uryasev, S. (2000). Optimization of Conditional
          Value-at-Risk. Journal of Risk.
        - Rockafellar, R. T., & Uryasev, S. (2002). Conditional Value-at-Risk
          for General Loss Distributions. Journal of Banking and Finance.

    Examples
    --------
        CVaR Sharpe ratio maximization:

        ```python
        portfolio = calc_azapy_cvar(
            returns_data=returns,
            alpha=0.975,
            rtype="Sharpe",
            portfolio_name="CVaR Sharpe Portfolio",
        )
        ```

        Minimum CVaR portfolio:

        ```python
        portfolio = calc_azapy_cvar(
            returns_data=returns, alpha=0.99, rtype="MinRisk", portfolio_name="Min CVaR Portfolio"
        )
        ```
    """
    is_valid, error_msg = _validate_returns_data(returns_data)
    if not is_valid:
        logger.error("Invalid returns_data: %s", error_msg)
        raise ValueError(error_msg)

    if not (0.0 < alpha < 1.0):
        msg = f"alpha must be in (0, 1), got {alpha}"
        logger.error(msg)
        raise ValueError(msg)

    valid_rtypes = {
        "MinRisk",
        "Sharpe",
        "Sharpe2",
        "InvNrisk",
        "RiskAverse",
        "Diverse",
        "MaxDiverse",
    }
    if rtype not in valid_rtypes:
        msg = f"rtype must be one of {valid_rtypes}, got '{rtype}'"
        logger.error(msg)
        raise ValueError(msg)

    if portfolio_name is None:
        portfolio_name = f"CVaR {rtype} Portfolio (alpha={alpha})"

    returns_pandas = _convert_polars_to_pandas_returns(returns_data)

    logger.info(
        "Calculating CVaR portfolio",
        extra={"rtype": rtype, "alpha": alpha, "num_assets": len(returns_pandas.columns)},
    )

    try:
        analyzer = azapy.CVaRAnalyzer(alpha=[alpha], rtype=rtype)
        analyzer.set_rrate(returns_pandas)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # azapy-internal scipy.sparse FutureWarning
            weights_series = analyzer.getWeights(rtype=rtype)

        if weights_series is None:
            raise ValueError("CVaRAnalyzer returned None weights")

        logger.info(
            "Optimization complete",
            extra={
                "portfolio_name": portfolio_name,
                "weights_sum": float(weights_series.sum()),
            },
        )

        return _extract_weights_from_azapy(
            weights_series=weights_series,
            portfolio_name=portfolio_name,
            optimization_date=optimization_date,
        )

    except Exception as e:
        logger.error("CVaR optimization failed: %s", str(e))
        raise ValueError(f"Optimization failed: {e!s}") from e


# =============================================================================
# Mean Absolute Deviation (MAD) Portfolio
# =============================================================================


def calc_azapy_mad(
    returns_data: pl.DataFrame,
    rtype: str = "MinRisk",
    portfolio_name: str | None = None,
    optimization_date: str | datetime | None = None,
) -> portfolio_QWIM:
    r"""Calculate Mean Absolute Deviation portfolio using azapy MADAnalyzer.

    MAD is a robust alternative to variance as a risk measure. It is less
    sensitive to extreme returns / outliers compared to standard deviation.

    Parameters
    ----------
    returns_data : pl.DataFrame
        Polars DataFrame with 'Date' column and asset return columns.
    rtype : str
        Optimization objective type. One of ``"MinRisk"``, ``"Sharpe"``,
        ``"InvNrisk"``, ``"RiskAverse"``, ``"MaxRet"``. Default is ``"MinRisk"``.
    portfolio_name : str | None
        Name for the resulting portfolio. Defaults to a name based on rtype.
    optimization_date : str | datetime | None
        Date for the portfolio weights. Defaults to current UTC date.

    Returns
    -------
    portfolio_QWIM
        Portfolio object containing optimized weights.

    Raises
    ------
    ValueError
        If returns_data is invalid.

    Notes
    -----
        Mean Absolute Deviation risk measure:

        $$
        MAD(w) = E\left[\left| w^T r - E[w^T r] \right|\right]
        $$

        For the minimum MAD portfolio:

        $$
        \min_w \; MAD(w) \quad \text{s.t.} \quad
        \mathbf{1}^T w = 1, \; w \geq 0
        $$

    References
    ----------
        - Konno, H., & Yamazaki, H. (1991). Mean-Absolute Deviation Portfolio
          Optimization Model. Management Science.

    Examples
    --------
        Minimum MAD portfolio:

        ```python
        portfolio = calc_azapy_mad(
            returns_data=returns, rtype="MinRisk", portfolio_name="MAD Min Risk Portfolio"
        )
        ```
    """
    is_valid, error_msg = _validate_returns_data(returns_data)
    if not is_valid:
        logger.error("Invalid returns_data: %s", error_msg)
        raise ValueError(error_msg)

    valid_rtypes = {
        "MinRisk",
        "Sharpe",
        "Sharpe2",
        "InvNrisk",
        "RiskAverse",
        "Diverse",
        "MaxDiverse",
    }
    if rtype not in valid_rtypes:
        msg = f"rtype must be one of {valid_rtypes}, got '{rtype}'"
        logger.error(msg)
        raise ValueError(msg)

    if portfolio_name is None:
        portfolio_name = f"MAD {rtype} Portfolio"

    returns_pandas = _convert_polars_to_pandas_returns(returns_data)

    logger.info(
        "Calculating MAD portfolio",
        extra={"rtype": rtype, "num_assets": len(returns_pandas.columns)},
    )

    try:
        analyzer = azapy.MADAnalyzer(rtype=rtype)
        analyzer.set_rrate(returns_pandas)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # azapy-internal scipy.sparse FutureWarning
            weights_series = analyzer.getWeights(rtype=rtype)

        if weights_series is None:
            raise ValueError("MADAnalyzer returned None weights")

        logger.info(
            "Optimization complete",
            extra={
                "portfolio_name": portfolio_name,
                "weights_sum": float(weights_series.sum()),
            },
        )

        return _extract_weights_from_azapy(
            weights_series=weights_series,
            portfolio_name=portfolio_name,
            optimization_date=optimization_date,
        )

    except Exception as e:
        logger.error("MAD optimization failed: %s", str(e))
        raise ValueError(f"Optimization failed: {e!s}") from e


# =============================================================================
# Inverse Volatility Portfolio
# =============================================================================


def calc_azapy_inverse_volatility(
    returns_data: pl.DataFrame,
    portfolio_name: str | None = None,
    optimization_date: str | datetime | None = None,
) -> portfolio_QWIM:
    r"""Calculate inverse volatility weighted portfolio using azapy InvVolEngine.

    Each asset is weighted inversely proportional to its historical volatility.
    Assets with lower volatility receive higher weights, providing a simple
    risk-adjusted allocation without solving an optimization problem.

    Parameters
    ----------
    returns_data : pl.DataFrame
        Polars DataFrame with 'Date' column and asset return columns.
    portfolio_name : str | None
        Name for the resulting portfolio. Defaults to "Inverse Volatility Portfolio".
    optimization_date : str | datetime | None
        Date for the portfolio weights. Defaults to current UTC date.

    Returns
    -------
    portfolio_QWIM
        Portfolio object containing optimized weights.

    Raises
    ------
    ValueError
        If returns_data is invalid.

    Notes
    -----
        Inverse volatility weights are computed as:

        $$
        w_i = \frac{1/\sigma_i}{\sum_j 1/\sigma_j}
        $$

        Where $\sigma_i$ is the historical standard deviation (volatility) of asset $i$.

        Properties:
        - Simple, no optimization solver required
        - Reduces concentration risk from high-volatility assets
        - Does not account for correlations between assets

    References
    ----------
        - Leote de Carvalho, R., Lu, X., & Moulin, P. (2012). Demystifying Equity
          Risk-Based Strategies. Journal of Portfolio Management.

    Examples
    --------
        Inverse volatility portfolio:

        ```python
        portfolio = calc_azapy_inverse_volatility(
            returns_data=returns, portfolio_name="Inverse Vol Portfolio"
        )
        ```
    """
    is_valid, error_msg = _validate_returns_data(returns_data)
    if not is_valid:
        logger.error("Invalid returns_data: %s", error_msg)
        raise ValueError(error_msg)

    if portfolio_name is None:
        portfolio_name = "Inverse Volatility Portfolio"

    returns_pandas = _convert_polars_to_pandas_returns(returns_data)

    logger.info(
        "Calculating inverse volatility portfolio",
        extra={"num_assets": len(returns_pandas.columns)},
    )

    try:
        analyzer = azapy.InvVolEngine()
        analyzer.set_rrate(returns_pandas)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # azapy-internal scipy.sparse FutureWarning
            weights_series = analyzer.getWeights()

        logger.info(
            "Optimization complete",
            extra={
                "portfolio_name": portfolio_name,
                "weights_sum": float(weights_series.sum()),
            },
        )

        return _extract_weights_from_azapy(
            weights_series=weights_series,
            portfolio_name=portfolio_name,
            optimization_date=optimization_date,
        )

    except Exception as e:
        logger.error("Inverse volatility portfolio failed: %s", str(e))
        raise ValueError(f"Optimization failed: {e!s}") from e


# =============================================================================
# Kelly Criterion Portfolio
# =============================================================================


def calc_azapy_kelly(
    returns_data: pl.DataFrame,
    rtype: str = "ExpCone",
    portfolio_name: str | None = None,
    optimization_date: str | datetime | None = None,
) -> portfolio_QWIM:
    r"""Calculate Kelly criterion portfolio using azapy KellyEngine.

    Maximizes the expected logarithmic growth rate of wealth. The Kelly
    criterion determines the optimal fraction of capital to allocate to
    each asset to maximize long-term growth.

    Parameters
    ----------
    returns_data : pl.DataFrame
        Polars DataFrame with 'Date' column and asset return columns.
    rtype : str
        Solver type for the Kelly optimization:

        - ``"ExpCone"`` : Exponential cone programming (recommended, exact)
        - ``"Full"`` : Full Kelly criterion (unconstrained growth maximization)
        - ``"Order2"`` : Second-order approximation (computationally cheaper)

        Default is ``"ExpCone"``.
    portfolio_name : str | None
        Name for the resulting portfolio. Defaults to a name based on rtype.
    optimization_date : str | datetime | None
        Date for the portfolio weights. Defaults to current UTC date.

    Returns
    -------
    portfolio_QWIM
        Portfolio object containing optimized weights.

    Raises
    ------
    ValueError
        If returns_data is invalid.

    Notes
    -----
        The Kelly criterion maximizes expected log-wealth growth:

        $$
        \max_w \; E\left[\log\left(1 + w^T r\right)\right]
        \quad \text{s.t.} \quad \mathbf{1}^T w = 1, \; w \geq 0
        $$

        This is equivalent to maximizing the geometric mean return.

        The full Kelly criterion can lead to very concentrated portfolios; in
        practice, fractional Kelly (e.g., half-Kelly) is often preferred.

    References
    ----------
        - Kelly, J. L. (1956). A New Interpretation of Information Rate.
          Bell System Technical Journal.
        - Thorp, E. O. (2008). The Kelly Criterion in Blackjack, Sports Betting,
          and the Stock Market. Handbook of Asset and Liability Management.

    Examples
    --------
        Kelly portfolio with exponential cone programming:

        ```python
        portfolio = calc_azapy_kelly(
            returns_data=returns, rtype="ExpCone", portfolio_name="Kelly Portfolio"
        )
        ```
    """
    is_valid, error_msg = _validate_returns_data(returns_data)
    if not is_valid:
        logger.error("Invalid returns_data: %s", error_msg)
        raise ValueError(error_msg)

    valid_rtypes = {"ExpCone", "Full", "Order2"}
    if rtype not in valid_rtypes:
        msg = f"rtype must be one of {valid_rtypes}, got '{rtype}'"
        logger.error(msg)
        raise ValueError(msg)

    if portfolio_name is None:
        portfolio_name = f"Kelly {rtype} Portfolio"

    returns_pandas = _convert_polars_to_pandas_returns(returns_data)

    logger.info(
        "Calculating Kelly portfolio",
        extra={"rtype": rtype, "num_assets": len(returns_pandas.columns)},
    )

    try:
        analyzer = azapy.KellyEngine(rtype=rtype)
        analyzer.set_rrate(returns_pandas)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # azapy-internal scipy.sparse FutureWarning
            weights_series = analyzer.getWeights()

        logger.info(
            "Optimization complete",
            extra={
                "portfolio_name": portfolio_name,
                "weights_sum": float(weights_series.sum()),
            },
        )

        return _extract_weights_from_azapy(
            weights_series=weights_series,
            portfolio_name=portfolio_name,
            optimization_date=optimization_date,
        )

    except Exception as e:
        logger.error("Kelly optimization failed: %s", str(e))
        raise ValueError(f"Optimization failed: {e!s}") from e


# =============================================================================
# Entropic Value at Risk (EVaR) Portfolio
# =============================================================================


def calc_azapy_evar(
    returns_data: pl.DataFrame,
    alpha: float = 0.975,
    rtype: str = "Sharpe",
    portfolio_name: str | None = None,
    optimization_date: str | datetime | None = None,
) -> portfolio_QWIM:
    r"""Calculate EVaR-optimal portfolio using azapy EVaRAnalyzer.

    Entropic Value at Risk (EVaR) is a coherent risk measure that is tighter
    than CVaR and uses the Chernoff inequality. It provides an upper bound on
    CVaR for all confidence levels simultaneously.

    Parameters
    ----------
    returns_data : pl.DataFrame
        Polars DataFrame with 'Date' column and asset return columns.
    alpha : float
        Confidence level for EVaR computation. Typical values: 0.95, 0.975, 0.99.
        Default is 0.975.
    rtype : str
        Optimization objective type. One of ``"MinRisk"``, ``"Sharpe"``,
        ``"InvNrisk"``, ``"RiskAverse"``, ``"MaxRet"``. Default is ``"Sharpe"``.
    portfolio_name : str | None
        Name for the resulting portfolio. Defaults to a name based on rtype.
    optimization_date : str | datetime | None
        Date for the portfolio weights. Defaults to current UTC date.

    Returns
    -------
    portfolio_QWIM
        Portfolio object containing optimized weights.

    Raises
    ------
    ValueError
        If returns_data is invalid or alpha is out of range.

    Notes
    -----
        EVaR at confidence level $\alpha$ is defined via the Chernoff bound:

        $$
        EVaR_\alpha(X) = \inf_{z > 0} \left\{
            z^{-1} \ln\left(\frac{E[e^{-zX}]}{1-\alpha}\right)
        \right\}
        $$

        EVaR satisfies the hierarchy:

        $$
        VaR_\alpha \leq CVaR_\alpha \leq EVaR_\alpha
        $$

        EVaR is more conservative but computationally tractable.

    References
    ----------
        - Ahmadi-Javid, A. (2012). Entropic Value-at-Risk: A New Coherent Risk
          Measure. Journal of Optimization Theory and Applications.

    Examples
    --------
        EVaR Sharpe ratio maximization:

        ```python
        portfolio = calc_azapy_evar(
            returns_data=returns,
            alpha=0.975,
            rtype="Sharpe",
            portfolio_name="EVaR Sharpe Portfolio",
        )
        ```
    """
    is_valid, error_msg = _validate_returns_data(returns_data)
    if not is_valid:
        logger.error("Invalid returns_data: %s", error_msg)
        raise ValueError(error_msg)

    if not (0.0 < alpha < 1.0):
        msg = f"alpha must be in (0, 1), got {alpha}"
        logger.error(msg)
        raise ValueError(msg)

    valid_rtypes = {
        "MinRisk",
        "Sharpe",
        "Sharpe2",
        "InvNrisk",
        "RiskAverse",
        "Diverse",
        "MaxDiverse",
    }
    if rtype not in valid_rtypes:
        msg = f"rtype must be one of {valid_rtypes}, got '{rtype}'"
        logger.error(msg)
        raise ValueError(msg)

    if portfolio_name is None:
        portfolio_name = f"EVaR {rtype} Portfolio (alpha={alpha})"

    returns_pandas = _convert_polars_to_pandas_returns(returns_data)

    logger.info(
        "Calculating EVaR portfolio",
        extra={"rtype": rtype, "alpha": alpha, "num_assets": len(returns_pandas.columns)},
    )

    try:
        analyzer = azapy.EVaRAnalyzer(alpha=[alpha], rtype=rtype)
        analyzer.set_rrate(returns_pandas)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # azapy-internal scipy.sparse FutureWarning
            weights_series = analyzer.getWeights(rtype=rtype)

        if weights_series is None:
            raise ValueError("EVaRAnalyzer returned None weights")

        logger.info(
            "Optimization complete",
            extra={
                "portfolio_name": portfolio_name,
                "weights_sum": float(weights_series.sum()),
            },
        )

        return _extract_weights_from_azapy(
            weights_series=weights_series,
            portfolio_name=portfolio_name,
            optimization_date=optimization_date,
        )

    except Exception as e:
        logger.error("EVaR optimization failed: %s", str(e))
        raise ValueError(f"Optimization failed: {e!s}") from e
