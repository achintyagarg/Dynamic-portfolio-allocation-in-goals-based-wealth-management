r"""Standard threshold-band portfolio rebalancing strategy for QWIM.

=========================================================

Provides :class:`Portfolio_Rebalancing_Standard`, a deterministic
rebalancing strategy that combines the two approaches most widely
recommended in the academic and practitioner literature:

**1. Tolerance-band (threshold) rebalancing**

    The portfolio is rebalanced whenever *any* asset's weight drifts
    more than a pre-defined tolerance :math:`\delta` from its target:

    .. math::

        |\, w_i^{\text{current}} - w_i^{\text{target}} \,| > \delta

    This is also known as *range rebalancing*, *band rebalancing*, or
    *corridor rebalancing* in the literature.

**2. Calendar (periodic) rebalancing** (optional, additive trigger)

    Irrespective of drift, the portfolio is also forcibly rebalanced if
    at least ``rebalancing_frequency_days`` calendar days have elapsed
    since the last rebalance.  Setting ``rebalancing_frequency_days=0``
    disables this trigger and produces a pure threshold strategy.

When either (or both) conditions are met, every asset is traded back to
its exact target weight ("full rebalancing").

The combined trigger is the standard approach backed by extensive
empirical evidence:

* Pure threshold strategies minimise unnecessary trading (and therefore
  transaction costs) relative to pure calendar strategies with the same
  average tracking error.
* Adding a calendar floor prevents indefinitely long no-trade periods
  when markets trend without reverting.

Transaction costs
-----------------
A simple proportional cost model is applied:

.. math::

    \\text{Cost}_i = |\\text{Trade Value}_i| \\times c

where :math:`c = \\text{transaction\\_cost\\_bps} / 10{,}000` is the
one-way cost rate expressed in basis points.

References
----------
* Plaxco, L.M., & Arnott, R.D. (2002). Rebalancing a global policy
  benchmark. *Journal of Portfolio Management*, 28(2), 9-22.
* Masters, S.J. (2003). Rebalancing. *Journal of Portfolio Management*,
  29(3), 52-57.
* Dichtl, H., Drobetz, W., & Wambach, M. (2016). When does calendar
  time rebalancing outperform threshold rebalancing?  *Journal of
  Portfolio Management*, 42(2), 37-49.
* Chow, T.M., Hsu, J., Kalesnik, V., & Little, B. (2011). A survey of
  alternative equity index strategies. *Financial Analysts Journal*,
  67(5), 37-57.
* Tsai, C.-Y. (2001). Rebalancing diversified portfolios of various
  risk profiles. *Journal of Financial Planning*, 14(10), 104-110.

Author
------
QWIM Team

Version
-------
1.0.0 (2026-07-01)
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np


if TYPE_CHECKING:
    from datetime import datetime
import polars as pl

from src.models.portfolio_rebalancing.portfolio_rebalancing_base import (
    Portfolio_Rebalancing_Base,
    Rebalancing_Strategy_Status,
)
from src.utils.custom_exceptions_errors_loggers.exception_custom import (
    Exception_Validation_Input,
)
from src.utils.custom_exceptions_errors_loggers.logger_custom import get_logger


logger = get_logger(__name__)

# ======================================================================
# Module-level defaults
# ======================================================================

# Absolute tolerance band width (5 % is the industry default per Plaxco & Arnott)
_DEFAULT_TOLERANCE_ABS: float = 0.05

# Calendar frequency (0 = pure threshold; 91 ~ quarterly; 365 ~ annual)
_DEFAULT_FREQUENCY_DAYS: int = 0

# Proportional one-way transaction cost in basis points (10 bps = 0.10 %)
_DEFAULT_COST_BPS: float = 10.0

# Tolerance band bounds
_TOLERANCE_ABS_MIN: float = 0.001  # 0.1 % — minimum meaningful band
_TOLERANCE_ABS_MAX: float = 0.50  # 50 % — any wider defies purpose

# Transaction cost bounds
_COST_BPS_MIN: float = 0.0  # zero cost (theoretical / frictionless)
_COST_BPS_MAX: float = 500.0  # 5 % — extreme illiquid-market ceiling


# ======================================================================
# Standard rebalancing strategy
# ======================================================================


class Portfolio_Rebalancing_Standard(Portfolio_Rebalancing_Base):  # noqa: N801
    r"""Standard threshold-band (+ optional calendar) rebalancing strategy.

    Combines an **absolute tolerance-band trigger** with an optional
    **calendar (periodic) trigger**:

    * **Threshold trigger**: rebalance if
      :math:`|w_i^{\text{curr}} - w_i^{\text{tgt}}| > \delta` for any
      asset :math:`i`.
    * **Calendar trigger** (when
      ``rebalancing_frequency_days`` > 0): rebalance unconditionally if
      the time since the last rebalance exceeds
      ``rebalancing_frequency_days`` calendar days.

    Full rebalancing back to target weights is performed on every event.

    Parameters
    ----------
    name_strategy : str, optional
        Human-readable label (default ``"Standard Threshold-Band Rebalancing"``).
    tolerance_abs : float, optional
        Absolute tolerance band half-width :math:`\delta` in weight
        units (default ``0.05``; i.e. each asset may drift ±5 % from
        its target before a rebalance is triggered).
        Must be in the range :data:`_TOLERANCE_ABS_MIN` -
        :data:`_TOLERANCE_ABS_MAX`.
    rebalancing_frequency_days : int, optional
        Calendar rebalancing frequency in days (default ``0``).
        Set to ``0`` to disable the calendar trigger and use a pure
        threshold strategy.  Non-negative integer.
    transaction_cost_bps : float, optional
        One-way proportional transaction cost in basis points
        (default ``10.0``; i.e. 10 bp = 0.10 % per unit of absolute
        trade value).  Must be in the range :data:`_COST_BPS_MIN` -
        :data:`_COST_BPS_MAX`.

    Attributes
    ----------
    m_tolerance_abs : float
        Absolute tolerance half-width (validated, stored at construction).
    m_frequency_days : int
        Calendar rebalancing frequency in days.
    m_cost_bps : float
        One-way transaction cost in basis points.
    m_last_rebalance_date : datetime or None
        Date of the most recent rebalance event (``None`` before the
        first call to :meth:`rebalance`).
    m_n_rebalances : int
        Running count of completed rebalancing events.

    Examples
    --------
    Pure threshold strategy (5 % band, 10 bps cost):

    >>> import numpy as np
    >>> from src.models.portfolio_rebalancing.portfolio_rebalancing_standard import (
    ...     Portfolio_Rebalancing_Standard,
    ... )
    >>> strategy = Portfolio_Rebalancing_Standard(tolerance_abs=0.05)
    >>> strategy.configure(
    ...     target_weights=np.array([0.60, 0.40]),
    ...     names_assets=["Equities", "Bonds"],
    ... )
    >>> drifted = np.array([0.67, 0.33])  # equities have drifted +7 %
    >>> strategy.should_rebalance(drifted)
    True
    >>> result = strategy.rebalance(drifted, portfolio_value=1_000_000.0)
    >>> print(result)

    Combined threshold + quarterly calendar:

    >>> from datetime import datetime
    >>> strategy_hybrid = Portfolio_Rebalancing_Standard(
    ...     tolerance_abs=0.05,
    ...     rebalancing_frequency_days=91,
    ... )
    """

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def __init__(
        self,
        name_strategy: str = "Standard Threshold-Band Rebalancing",
        *,
        tolerance_abs: float = _DEFAULT_TOLERANCE_ABS,
        rebalancing_frequency_days: int = _DEFAULT_FREQUENCY_DAYS,
        transaction_cost_bps: float = _DEFAULT_COST_BPS,
    ) -> None:
        super().__init__(name_strategy=name_strategy)

        self.m_tolerance_abs: float = self._init_tolerance_abs(tolerance_abs)
        self.m_frequency_days: int = self._init_frequency_days(rebalancing_frequency_days)
        self.m_cost_bps: float = self._init_cost_bps(transaction_cost_bps)

        self.m_last_rebalance_date: datetime | None = None
        self.m_n_rebalances: int = 0

        logger.info(
            "Portfolio_Rebalancing_Standard constructed: tolerance_abs=%s bps=%s freq_days=%s",
            self.m_tolerance_abs,
            self.m_cost_bps,
            self.m_frequency_days,
        )

    # ------------------------------------------------------------------
    # Constructor-time parameter validators (static helpers)
    # ------------------------------------------------------------------

    @staticmethod
    def _init_tolerance_abs(value: float) -> float:
        """Validate and return the tolerance band half-width."""
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise Exception_Validation_Input(
                "tolerance_abs must be a float",
                field_name="tolerance_abs",
                expected_type=float,
                actual_value=type(value).__name__,
            )
        fval = float(value)
        if not np.isfinite(fval):
            raise Exception_Validation_Input(
                "tolerance_abs must be finite",
                field_name="tolerance_abs",
                expected_type=float,
                actual_value=value,
            )
        if fval < _TOLERANCE_ABS_MIN or fval > _TOLERANCE_ABS_MAX:
            raise Exception_Validation_Input(
                f"tolerance_abs must be in [{_TOLERANCE_ABS_MIN}, {_TOLERANCE_ABS_MAX}], "
                f"got {value}",
                field_name="tolerance_abs",
                expected_type=float,
                actual_value=value,
            )
        return fval

    @staticmethod
    def _init_frequency_days(value: int) -> int:
        """Validate and return the rebalancing frequency in days."""
        if isinstance(value, bool) or not isinstance(value, int):
            raise Exception_Validation_Input(
                "rebalancing_frequency_days must be a non-negative integer",
                field_name="rebalancing_frequency_days",
                expected_type=int,
                actual_value=type(value).__name__,
            )
        if value < 0:
            raise Exception_Validation_Input(
                f"rebalancing_frequency_days must be >= 0, got {value}",
                field_name="rebalancing_frequency_days",
                expected_type=int,
                actual_value=value,
            )
        return value

    @staticmethod
    def _init_cost_bps(value: float) -> float:
        """Validate and return the transaction cost in basis points."""
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise Exception_Validation_Input(
                "transaction_cost_bps must be a float",
                field_name="transaction_cost_bps",
                expected_type=float,
                actual_value=type(value).__name__,
            )
        fval = float(value)
        if not np.isfinite(fval):
            raise Exception_Validation_Input(
                "transaction_cost_bps must be finite",
                field_name="transaction_cost_bps",
                expected_type=float,
                actual_value=value,
            )
        if fval < _COST_BPS_MIN or fval > _COST_BPS_MAX:
            raise Exception_Validation_Input(
                f"transaction_cost_bps must be in [{_COST_BPS_MIN}, {_COST_BPS_MAX}], got {value}",
                field_name="transaction_cost_bps",
                expected_type=float,
                actual_value=value,
            )
        return fval

    # ------------------------------------------------------------------
    # configure
    # ------------------------------------------------------------------

    def configure(
        self,
        target_weights: np.ndarray,
        names_assets: list[str],
    ) -> Portfolio_Rebalancing_Standard:
        r"""Configure the strategy with target weights and asset names.

        Validates, normalises (if necessary), and stores the target
        weights.  After a successful call the strategy status transitions
        to :attr:`Rebalancing_Strategy_Status.CONFIGURED`.

        Parameters
        ----------
        target_weights : np.ndarray
            1-D array of target portfolio weights (non-negative, sum to
            1.0 within :data:`_WEIGHT_SUM_TOLERANCE`).  If the sum
            deviates by more than the tolerance an error is raised rather
            than silently normalising — callers should supply properly
            normalised weights.
        names_assets : list[str]
            Ordered list of unique, non-empty asset names.  Length must
            match ``len(target_weights)``.

        Returns
        -------
        Portfolio_Rebalancing_Standard
            The configured strategy instance (``self``).

        Raises
        ------
        Exception_Validation_Input
            If ``target_weights`` or ``names_assets`` are invalid.
        """
        names = self._validate_asset_names(names_assets)
        w = self._validate_weights_array(
            target_weights,
            field_name="target_weights",
            must_sum_to_one=True,
        )

        if w.size != len(names):
            raise Exception_Validation_Input(
                f"target_weights length {w.size} does not match names_assets length {len(names)}",
                field_name="target_weights",
                expected_type=np.ndarray,
                actual_value=w.size,
            )

        self.m_names_assets = names
        self.m_target_weights = w.copy()
        self.m_status = Rebalancing_Strategy_Status.CONFIGURED  # pyright: ignore[reportAttributeAccessIssue]

        # Populate parameters dict for inspection / reporting
        self.m_parameters = {
            "tolerance_abs": self.m_tolerance_abs,
            "rebalancing_frequency_days": self.m_frequency_days,
            "transaction_cost_bps": self.m_cost_bps,
            "n_assets": len(names),
        }

        logger.info(
            "Portfolio_Rebalancing_Standard configured: n_assets=%d tolerance_abs=%s",
            len(names),
            self.m_tolerance_abs,
        )

        return self

    # ------------------------------------------------------------------
    # should_rebalance
    # ------------------------------------------------------------------

    def should_rebalance(
        self,
        current_weights: np.ndarray,
        *,
        current_date: datetime | None = None,
    ) -> bool:
        r"""Decide whether the portfolio should be rebalanced.

        Two independent trigger conditions are tested:

        1. **Threshold trigger**: any asset's absolute drift from target
           exceeds :attr:`m_tolerance_abs`.
        2. **Calendar trigger** (when ``m_frequency_days`` > 0): the
           number of calendar days elapsed since the last rebalance
           exceeds ``m_frequency_days``.

        The method returns ``True`` if *either* trigger fires.

        Parameters
        ----------
        current_weights : np.ndarray
            1-D array of current portfolio weights.
        current_date : datetime or None, keyword-only, optional
            Valuation date for the calendar trigger.  If ``None``
            (and ``m_frequency_days > 0``), the calendar trigger is
            skipped (treated as not firing) for this evaluation.

        Returns
        -------
        bool
            ``True`` if a rebalance is recommended.

        Raises
        ------
        Exception_Calculation
            If the strategy has not been configured yet.
        Exception_Validation_Input
            If ``current_weights`` is invalid.
        """
        self._check_is_configured()
        w = self._validate_weights_array(current_weights, field_name="current_weights")

        # --- Threshold trigger -------------------------------------------
        drifts = np.abs(w - self.m_target_weights)
        if bool(np.any(drifts > self.m_tolerance_abs)):
            logger.debug(
                "Threshold trigger fired: max_drift=%s > tolerance=%s",
                float(np.max(drifts)),
                self.m_tolerance_abs,
            )
            return True

        # --- Calendar trigger --------------------------------------------
        if (
            self.m_frequency_days > 0
            and current_date is not None
            and self.m_last_rebalance_date is not None
        ):
            days_elapsed = (current_date - self.m_last_rebalance_date).days
            if days_elapsed >= self.m_frequency_days:
                logger.debug(
                    "Calendar trigger fired: days_elapsed=%d >= frequency=%d",
                    days_elapsed,
                    self.m_frequency_days,
                )
                return True

        # Special case: calendar trigger fires on very first evaluation
        # when frequency is set and no prior rebalance has occurred.
        if (
            self.m_frequency_days > 0
            and current_date is not None
            and self.m_last_rebalance_date is None
        ):
            logger.debug("Calendar trigger fired: first evaluation with no prior rebalance")
            return True

        return False

    # ------------------------------------------------------------------
    # rebalance
    # ------------------------------------------------------------------

    def rebalance(
        self,
        current_weights: np.ndarray,
        *,
        current_date: datetime | None = None,
        portfolio_value: float = 1_000_000.0,
    ) -> pl.DataFrame:
        r"""Compute per-asset trades to rebalance back to target weights.

        All assets are traded to their exact target weights (full
        rebalancing).  The proportional transaction-cost model is applied
        to the absolute trade value:

        .. math::

            \text{Cost}_i
            = | w_i^{\text{trade}} \times V | \times
              \frac{c_{\text{bps}}}{10{,}000}

        Parameters
        ----------
        current_weights : np.ndarray
            1-D array of current portfolio weights.
        current_date : datetime or None, keyword-only, optional
            Trade date.  Stored in :attr:`m_last_rebalance_date` and
            used by the calendar trigger on subsequent calls.
        portfolio_value : float, keyword-only, optional
            Total portfolio value in account currency
            (default ``1_000_000.0``).

        Returns
        -------
        pl.DataFrame
            Per-asset rebalancing table with columns:

            * ``"Asset"`` -- asset name
            * ``"Current_Weight"`` -- weight before rebalancing
            * ``"Target_Weight"`` -- target (desired) weight
            * ``"Drift"`` -- ``current - target`` (signed)
            * ``"Trade_Weight"`` -- ``target - current`` (signed;
              positive = buy, negative = sell)
            * ``"Trade_Value"`` -- ``trade_weight * portfolio_value``
            * ``"Cost"`` -- per-asset estimated transaction cost

        Raises
        ------
        Exception_Calculation
            If the strategy has not been configured yet.
        Exception_Validation_Input
            If ``current_weights`` or ``portfolio_value`` are invalid.
        """
        self._check_is_configured()
        assert self.m_target_weights is not None  # guaranteed by _check_is_configured()
        w = self._validate_weights_array(current_weights, field_name="current_weights")
        pv = self._validate_portfolio_value(portfolio_value)

        target: np.ndarray = self.m_target_weights
        drift = w - target
        trade_w = target - w
        trade_v = trade_w * pv
        cost_rate = self.m_cost_bps / 10_000.0
        costs = np.abs(trade_v) * cost_rate

        # Record the event
        self.m_last_rebalance_date = current_date
        self.m_n_rebalances += 1

        logger.info(
            "Portfolio_Rebalancing_Standard.rebalance: event #%d portfolio_value=%s date=%s",
            self.m_n_rebalances,
            pv,
            current_date,
        )

        return pl.DataFrame(
            {
                "Asset": self.m_names_assets,
                "Current_Weight": w.tolist(),
                "Target_Weight": target.tolist(),
                "Drift": drift.tolist(),
                "Trade_Weight": trade_w.tolist(),
                "Trade_Value": trade_v.tolist(),
                "Cost": costs.tolist(),
            },
        )

    # ------------------------------------------------------------------
    # Additional public methods
    # ------------------------------------------------------------------

    def get_turnover(self, current_weights: np.ndarray) -> float:
        r"""Compute the one-way turnover required by a full rebalance.

        .. math::

            \text{Turnover}
            = \frac{1}{2} \sum_i |w_i^{\text{tgt}} - w_i^{\text{curr}}|

        Parameters
        ----------
        current_weights : np.ndarray
            1-D array of current portfolio weights.

        Returns
        -------
        float
            One-way turnover in weight units (between 0.0 and 1.0).

        Raises
        ------
        Exception_Calculation
            If the strategy has not been configured yet.
        Exception_Validation_Input
            If ``current_weights`` is invalid.
        """
        self._check_is_configured()
        w = self._validate_weights_array(current_weights, field_name="current_weights")
        trade_w = self.m_target_weights - w
        return self._compute_turnover(trade_w)

    def get_estimated_cost(
        self,
        current_weights: np.ndarray,
        portfolio_value: float = 1_000_000.0,
    ) -> float:
        r"""Estimate the total transaction cost of a full rebalance.

        .. math::

            \text{Cost}
            = \text{Turnover} \times V \times \frac{c_{\text{bps}}}{10{,}000}

        Parameters
        ----------
        current_weights : np.ndarray
            1-D array of current portfolio weights.
        portfolio_value : float, optional
            Total portfolio value (default ``1_000_000.0``).

        Returns
        -------
        float
            Estimated total transaction cost in account currency.

        Raises
        ------
        Exception_Calculation
            If the strategy has not been configured yet.
        Exception_Validation_Input
            If inputs are invalid.
        """
        self._check_is_configured()
        pv = self._validate_portfolio_value(portfolio_value)
        turnover = self.get_turnover(current_weights)
        return turnover * pv * (self.m_cost_bps / 10_000.0)

    def reset(self) -> None:
        """Reset the calendar state (last rebalance date and event counter).

        Useful at the start of a new simulation period without
        re-configuring the target weights.
        """
        self.m_last_rebalance_date = None
        self.m_n_rebalances = 0
        logger.info(
            "Portfolio_Rebalancing_Standard '%s': calendar state reset.",
            self.m_name_strategy,
        )

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def tolerance_abs(self) -> float:
        """Float : Absolute tolerance band half-width."""
        return self.m_tolerance_abs

    @property
    def rebalancing_frequency_days(self) -> int:
        """Int : Calendar rebalancing frequency in days (0 = disabled)."""
        return self.m_frequency_days

    @property
    def transaction_cost_bps(self) -> float:
        """Float : One-way transaction cost in basis points."""
        return self.m_cost_bps

    @property
    def last_rebalance_date(self) -> datetime | None:
        """Datetime or None : Date of the most recent rebalancing event."""
        return self.m_last_rebalance_date

    @property
    def n_rebalances(self) -> int:
        """Int : Total number of completed rebalancing events."""
        return self.m_n_rebalances

    # ------------------------------------------------------------------
    # Dunder overrides
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"Portfolio_Rebalancing_Standard("
            f"name='{self.m_name_strategy}', "
            f"tolerance_abs={self.m_tolerance_abs}, "
            f"frequency_days={self.m_frequency_days}, "
            f"cost_bps={self.m_cost_bps}, "
            f"n_assets={self.n_assets}, "
            f"status={self.m_status.value})"
        )
