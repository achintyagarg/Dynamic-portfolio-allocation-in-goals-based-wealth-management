r"""Abstract base class for QWIM portfolio rebalancing strategies.

=========================================================

Provides :class:`Portfolio_Rebalancing_Base`, an abstract base class that
defines the common interface for all portfolio rebalancing strategies used
in the QWIM wealth-management and retirement-planning pipeline.

A *portfolio rebalancing strategy* answers three questions:

1. **Configuration** — given target weights and asset names, store and
   validate the rebalancing parameters (``configure``).
2. **Trigger detection** — given the portfolio's *current* weights (and
   optional date), decide whether a rebalance event should occur
   (``should_rebalance``).
3. **Execution** — compute the required trades, estimated transaction
   costs, and new weights, returning a tidy :class:`polars.DataFrame`
   (``rebalance``).

Design goals
------------
* **Consistency** — all strategies share the same ``configure`` /
  ``should_rebalance`` / ``rebalance`` interface, making them
  interchangeable in back-test engines and simulation pipelines.
* **Convention** — weights are represented as 1-D :class:`numpy.ndarray`
  of non-negative floats that sum to 1.0 (within a small tolerance).
  Portfolio value is expressed in the account currency (e.g. USD).
* **Safety** — thorough input validation with early returns and
  descriptive exceptions before any arithmetic.
* **Auditability** — every state transition is written to the project
  logger in ``%s`` format (no f-strings in log calls).

Terminology
-----------
*Current weights*
    The asset allocations actually held at the point of evaluation,
    computed as ``market value of asset / total portfolio value``.
*Target weights*
    The desired steady-state allocations chosen by the investment policy
    statement (IPS) or optimisation engine.
*Drift*
    The signed deviation ``current_weight - target_weight`` for each
    asset.  Positive drift means the asset is over-weight relative to
    target; negative means under-weight.
*Turnover* (one-way)
    ``sum(|trade_weights|) / 2`` — half the total absolute weight
    traded in a single rebalancing event.
*Transaction cost*
    Estimated total cost of executing all required trades, computed as
    ``turnover * portfolio_value * cost_rate``.

References
----------
* Plaxco, L.M., & Arnott, R.D. (2002). Rebalancing a global policy
  benchmark. *Journal of Portfolio Management*, 28(2), 9-22.
* Masters, S.J. (2003). Rebalancing. *Journal of Portfolio Management*,
  29(3), 52-57.
* Dichtl, H., Drobetz, W., & Wambach, M. (2016). When does calendar
  time rebalancing outperform threshold rebalancing?  *Journal of
  Portfolio Management*, 42(2), 37-49.
* Buetow, G.W., Sellers, R., Trotter, D., Hunt, E., & Whipple, W.A.
  (2002). The benefits of rebalancing. *Journal of Portfolio Management*,
  28(2), 23-32.

Author
------
QWIM Team

Version
-------
1.0.0 (2026-07-01)
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import numpy as np


if TYPE_CHECKING:
    from datetime import datetime
import polars as pl

from aenum import Enum

from src.utils.custom_exceptions_errors_loggers.exception_custom import (
    Exception_Calculation,
    Exception_Validation_Input,
)
from src.utils.custom_exceptions_errors_loggers.logger_custom import get_logger


logger = get_logger(__name__)

# ======================================================================
# Tolerances
# ======================================================================

# Weight normalisation: accept sum-to-one within this absolute tolerance
_WEIGHT_SUM_TOLERANCE: float = 1e-6

# Minimum number of assets supported
_MIN_N_ASSETS: int = 1


# ======================================================================
# Enum
# ======================================================================


class Rebalancing_Strategy_Status(Enum):  # noqa: N801  # pyright: ignore[reportGeneralTypeIssues]
    """Lifecycle status of a portfolio rebalancing strategy instance.

    Attributes
    ----------
    NOT_CONFIGURED : str
        Strategy has been constructed but ``configure()`` has not been
        called yet — no target weights are set.
    CONFIGURED : str
        Strategy has been successfully configured with valid target
        weights and is ready for use.
    FAILED : str
        The most recent ``configure()`` or ``rebalance()`` call raised
        an unexpected error.
    """

    NOT_CONFIGURED = "Not Configured"
    CONFIGURED = "Configured"
    FAILED = "Failed"


# ======================================================================
# Abstract base class
# ======================================================================


class Portfolio_Rebalancing_Base(ABC):  # noqa: N801
    r"""Abstract base class for QWIM portfolio rebalancing strategies.

    All concrete rebalancing strategies must inherit from this class and
    implement :meth:`configure`, :meth:`should_rebalance`, and
    :meth:`rebalance`.

    Parameters
    ----------
    name_strategy : str, optional
        Human-readable label for the strategy
        (default ``"Portfolio Rebalancing"``).

    Notes
    -----
    Members follow the project convention of ``m_`` prefix for instance
    attributes that back public properties.

    Weights are represented as 1-D ``float64`` :class:`numpy.ndarray`
    objects.  They must be non-negative and sum to 1.0 within
    :data:`_WEIGHT_SUM_TOLERANCE`.

    Examples
    --------
    Subclasses override ``configure``, ``should_rebalance``, and
    ``rebalance``:

    .. code-block:: python

        class My_Rebalancing_Strategy(Portfolio_Rebalancing_Base):
            def configure(self, target_weights, names_assets):
                # store target_weights and names_assets
                self.m_status = Rebalancing_Strategy_Status.CONFIGURED
                return self

            def should_rebalance(self, current_weights, *, current_date=None):
                # compute drift; return True if rebalance needed
                ...

            def rebalance(self, current_weights, *, current_date=None, portfolio_value=1_000_000.0):
                # compute trades and return result DataFrame
                ...
    """

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def __init__(self, name_strategy: str = "Portfolio Rebalancing") -> None:
        if not isinstance(name_strategy, str) or not name_strategy.strip():
            raise Exception_Validation_Input(
                "name_strategy must be a non-empty string",
                field_name="name_strategy",
                expected_type=str,
                actual_value=name_strategy,
            )

        self.m_name_strategy: str = name_strategy.strip()
        self.m_status: Rebalancing_Strategy_Status = (  # pyright: ignore[reportAttributeAccessIssue]
            Rebalancing_Strategy_Status.NOT_CONFIGURED
        )
        self.m_target_weights: np.ndarray | None = None
        self.m_names_assets: list[str] = []
        self.m_parameters: dict[str, float | int | str] = {}

        logger.info(
            "Portfolio_Rebalancing_Base created: '%s'",
            self.m_name_strategy,
        )

    # ------------------------------------------------------------------
    # Abstract methods
    # ------------------------------------------------------------------

    @abstractmethod
    def configure(
        self,
        target_weights: np.ndarray,
        names_assets: list[str],
    ) -> Portfolio_Rebalancing_Base:
        """Configure the strategy with target weights and asset names.

        Concrete subclasses must:

        1. Validate ``target_weights`` and ``names_assets``.
        2. Store them as instance attributes.
        3. Set any strategy-specific parameters.
        4. Set ``self.m_status = Rebalancing_Strategy_Status.CONFIGURED``.
        5. Return ``self`` for method chaining.

        Parameters
        ----------
        target_weights : np.ndarray
            1-D array of target portfolio weights (non-negative,
            sum to 1.0 within tolerance).  Length must equal
            ``len(names_assets)``.
        names_assets : list[str]
            Ordered list of asset names.  Must be non-empty and
            contain unique, non-empty strings.

        Returns
        -------
        Portfolio_Rebalancing_Base
            The configured strategy instance (``self``).

        Raises
        ------
        Exception_Validation_Input
            If ``target_weights`` or ``names_assets`` are invalid.
        """

    @abstractmethod
    def should_rebalance(
        self,
        current_weights: np.ndarray,
        *,
        current_date: datetime | None = None,
    ) -> bool:
        """Decide whether the portfolio should be rebalanced right now.

        Concrete subclasses implement the strategy-specific trigger
        logic (e.g., threshold test, calendar check, or a combination).

        Parameters
        ----------
        current_weights : np.ndarray
            1-D array of current portfolio weights (same length as
            ``m_names_assets``).  Must be non-negative and sum to 1.0
            within tolerance.
        current_date : datetime or None, keyword-only, optional
            Valuation date.  Used by calendar-aware strategies; ignored
            by pure threshold strategies.

        Returns
        -------
        bool
            ``True`` if the strategy recommends rebalancing now;
            ``False`` otherwise.

        Raises
        ------
        Exception_Calculation
            If the strategy has not been configured yet.
        Exception_Validation_Input
            If ``current_weights`` is invalid.
        """

    @abstractmethod
    def rebalance(
        self,
        current_weights: np.ndarray,
        *,
        current_date: datetime | None = None,
        portfolio_value: float = 1_000_000.0,
    ) -> pl.DataFrame:
        """Compute the trades required to rebalance the portfolio.

        Concrete subclasses must:

        1. Verify the strategy is configured.
        2. Validate ``current_weights`` and ``portfolio_value``.
        3. Compute trades and estimated transaction costs.
        4. Return a tidy per-asset :class:`polars.DataFrame`.

        Parameters
        ----------
        current_weights : np.ndarray
            1-D array of current portfolio weights.
        current_date : datetime or None, keyword-only, optional
            Valuation / trade date.  Used to record the rebalancing
            event and update calendar state.
        portfolio_value : float, keyword-only, optional
            Total portfolio value in account currency (default
            ``1_000_000.0``).  Used to convert weight trades to
            monetary amounts.

        Returns
        -------
        pl.DataFrame
            Per-asset rebalancing table with at minimum:

            * ``"Asset"`` -- :class:`polars.String`
            * ``"Current_Weight"`` -- :class:`polars.Float64`
            * ``"Target_Weight"`` -- :class:`polars.Float64`
            * ``"Drift"`` -- :class:`polars.Float64`
              (``current - target``)
            * ``"Trade_Weight"`` -- :class:`polars.Float64`
              (``target - current``; positive = buy)
            * ``"Trade_Value"`` -- :class:`polars.Float64`
              (``trade_weight * portfolio_value``; account currency)
            * ``"Cost"`` -- :class:`polars.Float64`
              (estimated transaction cost in account currency)

        Raises
        ------
        Exception_Calculation
            If the strategy has not been configured yet.
        Exception_Validation_Input
            If ``current_weights`` is invalid or ``portfolio_value``
            is not a positive finite number.
        """

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def name_strategy(self) -> str:
        """Str : Human-readable strategy label."""
        return self.m_name_strategy

    @property
    def status(self) -> Rebalancing_Strategy_Status:
        """Rebalancing_Strategy_Status : Current lifecycle status."""
        return self.m_status

    @property
    def is_configured(self) -> bool:
        """Bool : ``True`` iff the strategy has been configured."""
        return self.m_status == Rebalancing_Strategy_Status.CONFIGURED

    @property
    def target_weights(self) -> np.ndarray | None:
        """np.ndarray or None : Target weights (copy), or ``None`` before configure."""
        if self.m_target_weights is None:
            return None
        return self.m_target_weights.copy()

    @property
    def names_assets(self) -> list[str]:
        """list[str] : Ordered asset names (copy)."""
        return list(self.m_names_assets)

    @property
    def n_assets(self) -> int:
        """Int : Number of assets in the portfolio."""
        return len(self.m_names_assets)

    @property
    def parameters(self) -> dict[str, float | int | str]:
        """Dict : Strategy parameters (copy)."""
        return self.m_parameters.copy()

    # ------------------------------------------------------------------
    # Shared validation helpers
    # ------------------------------------------------------------------

    def _validate_weights_array(
        self,
        weights: np.ndarray,
        field_name: str = "weights",
        *,
        must_sum_to_one: bool = True,
    ) -> np.ndarray:
        """Validate a 1-D weight array and return it as a ``float64`` copy.

        Checks that *weights* is a 1-D array-like of finite non-negative
        values and, when ``must_sum_to_one`` is ``True``, that the values
        sum to 1.0 within :data:`_WEIGHT_SUM_TOLERANCE`.

        If a configured strategy is present, also checks that the length
        matches :attr:`n_assets`.

        Parameters
        ----------
        weights : np.ndarray
            Weight array to validate.
        field_name : str, optional
            Name of the parameter, used in error messages.
        must_sum_to_one : bool, keyword-only, optional
            If ``True`` (default) the sum-to-one constraint is enforced.

        Returns
        -------
        np.ndarray
            Validated weight array (1-D, ``float64``).

        Raises
        ------
        Exception_Validation_Input
            If ``weights`` is not a valid weight array.
        """
        if not isinstance(weights, (list, np.ndarray)):
            raise Exception_Validation_Input(
                f"{field_name} must be a list or numpy.ndarray",
                field_name=field_name,
                expected_type=np.ndarray,
                actual_value=type(weights).__name__,
            )

        try:
            w = np.asarray(weights, dtype=float)
        except (TypeError, ValueError) as exc:
            raise Exception_Validation_Input(
                f"{field_name} must be convertible to a float array",
                field_name=field_name,
                expected_type=np.ndarray,
                actual_value=type(weights).__name__,
            ) from exc

        if w.ndim != 1 or w.size == 0:
            raise Exception_Validation_Input(
                f"{field_name} must be a non-empty 1-D array",
                field_name=field_name,
                expected_type=np.ndarray,
                actual_value=w.shape,
            )

        if not np.all(np.isfinite(w)):
            raise Exception_Validation_Input(
                f"{field_name} must not contain NaN or Inf values",
                field_name=field_name,
                expected_type=float,
                actual_value="NaN or Inf",
            )

        if np.any(w < 0.0):
            raise Exception_Validation_Input(
                f"{field_name} must be non-negative (all values >= 0)",
                field_name=field_name,
                expected_type=float,
                actual_value="negative weight",
            )

        if must_sum_to_one:
            wsum = float(np.sum(w))
            if abs(wsum - 1.0) > _WEIGHT_SUM_TOLERANCE:
                raise Exception_Validation_Input(
                    f"{field_name} must sum to 1.0 (got {wsum:.6f})",
                    field_name=field_name,
                    expected_type=float,
                    actual_value=wsum,
                )

        # If strategy is configured, verify length matches n_assets
        if self.m_names_assets and w.size != len(self.m_names_assets):
            raise Exception_Validation_Input(
                f"{field_name} length {w.size} does not match n_assets {len(self.m_names_assets)}",
                field_name=field_name,
                expected_type=np.ndarray,
                actual_value=w.size,
            )

        return w

    def _validate_portfolio_value(
        self,
        portfolio_value: float,
        field_name: str = "portfolio_value",
    ) -> float:
        """Validate that *portfolio_value* is a positive finite number.

        Parameters
        ----------
        portfolio_value : float
            Portfolio value to validate.
        field_name : str, optional
            Name of the parameter used in error messages.

        Returns
        -------
        float
            Validated portfolio value.

        Raises
        ------
        Exception_Validation_Input
            If ``portfolio_value`` is not a positive finite number.
        """
        if isinstance(portfolio_value, bool) or not isinstance(portfolio_value, (int, float)):
            raise Exception_Validation_Input(
                f"{field_name} must be a positive finite number",
                field_name=field_name,
                expected_type=float,
                actual_value=type(portfolio_value).__name__,
            )

        fval = float(portfolio_value)
        if not np.isfinite(fval) or fval <= 0.0:
            raise Exception_Validation_Input(
                f"{field_name} must be a positive finite number, got {portfolio_value}",
                field_name=field_name,
                expected_type=float,
                actual_value=portfolio_value,
            )

        return fval

    def _validate_asset_names(
        self,
        names_assets: list[str],
        n_assets_expected: int | None = None,
    ) -> list[str]:
        """Validate a list of asset names.

        Parameters
        ----------
        names_assets : list[str]
            Asset names to validate.
        n_assets_expected : int or None, optional
            If provided, the list must have exactly this many elements.

        Returns
        -------
        list[str]
            Validated list of asset names (stripped copies).

        Raises
        ------
        Exception_Validation_Input
            If ``names_assets`` is not a non-empty list of unique
            non-empty strings.
        """
        if not isinstance(names_assets, list) or len(names_assets) == 0:
            raise Exception_Validation_Input(
                "names_assets must be a non-empty list",
                field_name="names_assets",
                expected_type=list,
                actual_value=type(names_assets).__name__,
            )

        cleaned: list[str] = []
        for i, name in enumerate(names_assets):
            if not isinstance(name, str) or not name.strip():
                raise Exception_Validation_Input(
                    f"names_assets[{i}] must be a non-empty string",
                    field_name=f"names_assets[{i}]",
                    expected_type=str,
                    actual_value=name,
                )
            cleaned.append(name.strip())

        if len(set(cleaned)) != len(cleaned):
            raise Exception_Validation_Input(
                "names_assets must contain unique names (duplicates found)",
                field_name="names_assets",
                expected_type=list,
                actual_value=cleaned,
            )

        if n_assets_expected is not None and len(cleaned) != n_assets_expected:
            raise Exception_Validation_Input(
                f"names_assets length {len(cleaned)} does not match "
                f"expected n_assets {n_assets_expected}",
                field_name="names_assets",
                expected_type=list,
                actual_value=len(cleaned),
            )

        return cleaned

    def _check_is_configured(self) -> None:
        """Assert the strategy has been configured.

        Raises
        ------
        Exception_Calculation
            If :attr:`is_configured` is ``False``.
        """
        if not self.is_configured:
            raise Exception_Calculation(
                f"Strategy '{self.m_name_strategy}' has not been "
                "configured — call configure() first",
            )

    # ------------------------------------------------------------------
    # Shared computation helpers
    # ------------------------------------------------------------------

    def _weight_drifts(self, current_weights: np.ndarray) -> np.ndarray:
        """Compute signed weight drifts from target (``current - target``).

        Parameters
        ----------
        current_weights : np.ndarray
            Validated 1-D float array of current weights.

        Returns
        -------
        np.ndarray
            Signed drifts; positive = over-weight, negative = under-weight.

        Raises
        ------
        Exception_Calculation
            If the strategy has not been configured yet.
        """
        self._check_is_configured()
        return current_weights - self.m_target_weights

    def _compute_turnover(self, trade_weights: np.ndarray) -> float:
        r"""Compute one-way portfolio turnover from an array of trade weights.

        Turnover is defined as half the total absolute weight traded:

        .. math::

            \text{Turnover} = \frac{1}{2}\sum_i |w_i^{\text{trade}}|

        For a pure rebalance-to-target trade the buy-side and sell-side
        amounts are equal, so ``turnover = sum(positive trades)``.

        Parameters
        ----------
        trade_weights : np.ndarray
            1-D array of signed trade weights
            (``target - current``; positive = buy).

        Returns
        -------
        float
            One-way portfolio turnover in weight units (0.0 - 1.0).
        """
        return float(np.sum(np.abs(trade_weights))) / 2.0

    # ------------------------------------------------------------------
    # Utility: summary statistics over a rebalancing result
    # ------------------------------------------------------------------

    def get_summary_statistics(self, rebalancing_result: pl.DataFrame) -> pl.DataFrame:
        r"""Compute scalar summary statistics for a single rebalancing event.

        Parameters
        ----------
        rebalancing_result : pl.DataFrame
            Output of :meth:`rebalance` with at minimum ``"Drift"``,
            ``"Trade_Weight"``, ``"Trade_Value"``, and ``"Cost"``
            columns.

        Returns
        -------
        pl.DataFrame
            Single-row DataFrame with columns:

            * ``"Max_Abs_Drift"`` -- maximum absolute weight deviation
            * ``"Mean_Abs_Drift"`` -- mean absolute weight deviation
            * ``"Turnover"`` -- one-way turnover (weight units)
            * ``"Total_Trade_Value"`` -- sum of absolute trade values
            * ``"Total_Cost"`` -- estimated total transaction cost

        Raises
        ------
        Exception_Validation_Input
            If required columns are absent from ``rebalancing_result``.
        """
        required_cols = {"Drift", "Trade_Weight", "Trade_Value", "Cost"}
        missing = required_cols - set(rebalancing_result.columns)
        if missing:
            raise Exception_Validation_Input(
                f"rebalancing_result is missing columns: {sorted(missing)}",
                field_name="rebalancing_result",
                expected_type=pl.DataFrame,
                actual_value=rebalancing_result.columns,
            )

        drifts = rebalancing_result["Drift"].to_numpy()
        trade_w = rebalancing_result["Trade_Weight"].to_numpy()
        trade_v = np.abs(rebalancing_result["Trade_Value"].to_numpy())
        costs = rebalancing_result["Cost"].to_numpy()

        return pl.DataFrame(
            {
                "Max_Abs_Drift": [float(np.max(np.abs(drifts)))],
                "Mean_Abs_Drift": [float(np.mean(np.abs(drifts)))],
                "Turnover": [float(np.sum(np.abs(trade_w))) / 2.0],
                "Total_Trade_Value": [float(np.sum(trade_v))],
                "Total_Cost": [float(np.sum(costs))],
            },
        )

    # ------------------------------------------------------------------
    # Dunder overrides
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        n = self.n_assets
        assets_str = f"n_assets={n}" if n > 0 else "not configured"
        return (
            f"Portfolio_Rebalancing_Base("
            f"name='{self.m_name_strategy}', "
            f"{assets_str}, "
            f"status={self.m_status.value})"
        )
