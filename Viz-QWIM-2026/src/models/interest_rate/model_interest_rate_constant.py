r"""Constant short-rate model for the QWIM fixed-income pipeline.

========================================================

Provides :class:`Interest_Rate_Model_Constant`, a deterministic model
that projects a single fixed (constant) annual short rate for every
future period.

This is the simplest possible interest-rate assumption and is commonly
used as a baseline scenario or sensitivity anchor in retirement-income,
discounting, and liability-valuation models.  When historical data is
supplied to :meth:`fit`, the model estimates the rate as the arithmetic
mean of the observed annual short rates; otherwise it uses the
constructor-supplied rate directly.

Author
------
QWIM Team

Version
-------
1.0.0 (2026-07-01)
"""

from __future__ import annotations

from datetime import (
    date as _date,
    timedelta,
)

import numpy as np
import polars as pl

from src.models.interest_rate.model_interest_rate_base import (
    Interest_Rate_Model_Base,
    Interest_Rate_Model_Status,
)
from src.utils.custom_exceptions_errors_loggers.exception_custom import (
    Exception_Validation_Input,
)
from src.utils.custom_exceptions_errors_loggers.logger_custom import get_logger


logger = get_logger(__name__)

# ======================================================================
# Module-level defaults
# ======================================================================

_DEFAULT_ANNUAL_RATE: float = 0.04  # 4 % — approximate long-run US short rate

# Valid range for a short rate (handles negative-rate environments)
_RATE_MIN: float = -0.10  # -10 %
_RATE_MAX: float = 0.50  # 50 %


# ======================================================================
# Constant short-rate model
# ======================================================================


class Interest_Rate_Model_Constant(Interest_Rate_Model_Base):  # noqa: N801
    r"""Deterministic constant short-rate model.

    The model projects the same annual short rate, :math:`r`, for every
    future period:

    .. math::

        r_t = r \quad \forall\; t \geq 0

    where :math:`r` is either supplied directly via the constructor or
    estimated from historical data as the arithmetic mean of observed
    annual short rates.

    Parameters
    ----------
    annual_rate : float, optional
        Fixed annual short rate in decimal form (default ``0.04`` for
        4 %).  Used when ``fit()`` is called without data or when
        acting as a hard-coded assumption.
    name_model : str, optional
        Human-readable label (default ``"Constant Interest Rate"``).

    Examples
    --------
    Hard-coded rate, no fitting required:

    >>> model = Interest_Rate_Model_Constant(annual_rate=0.05)
    >>> model.fit(pl.DataFrame())  # no-op fit
    Interest_Rate_Model_Constant(name='Constant Interest Rate', rate=0.050000, status=Fitted)
    >>> df = model.predict(n_periods=3, start_date="2025-01-01")
    >>> df["short_rate"].to_list()
    [0.05, 0.05, 0.05]

    Fit to historical data:

    >>> import polars as pl
    >>> hist = pl.DataFrame(
    ...     {
    ...         "Date": ["2021-01-01", "2022-01-01", "2023-01-01", "2024-01-01"],
    ...         "short_rate": [0.025, 0.050, 0.053, 0.045],
    ...     }
    ... )
    >>> model = Interest_Rate_Model_Constant()
    >>> model.fit(hist)
    Interest_Rate_Model_Constant(name='Constant Interest Rate', rate=0.043250, status=Fitted)
    """

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def __init__(
        self,
        annual_rate: float = _DEFAULT_ANNUAL_RATE,
        name_model: str = "Constant Interest Rate",
    ) -> None:
        super().__init__(name_model=name_model)

        if not isinstance(annual_rate, float | int) or not np.isfinite(annual_rate):
            raise Exception_Validation_Input(
                "annual_rate must be a finite number",
                field_name="annual_rate",
                expected_type=float,
                actual_value=annual_rate,
            )
        if annual_rate < _RATE_MIN or annual_rate > _RATE_MAX:
            raise Exception_Validation_Input(
                f"annual_rate must be in the range [{_RATE_MIN}, {_RATE_MAX}]",
                field_name="annual_rate",
                expected_type=float,
                actual_value=annual_rate,
            )

        self.m_annual_rate: float = float(annual_rate)

        logger.info(
            "Interest_Rate_Model_Constant constructed with annual_rate=%.6f",
            self.m_annual_rate,
        )

    # ------------------------------------------------------------------
    # Fit
    # ------------------------------------------------------------------

    def fit(self, data: pl.DataFrame) -> Interest_Rate_Model_Constant:
        """Fit (calibrate) the constant model to historical short-rate data.

        If *data* is a non-empty DataFrame with the required columns the
        model updates :attr:`m_annual_rate` to the arithmetic mean of
        the observed ``"short_rate"`` column.  If an empty DataFrame is
        passed the constructor rate is kept unchanged.

        Parameters
        ----------
        data : pl.DataFrame
            Historical data.  If non-empty, must contain columns
            ``"Date"`` and ``"short_rate"`` (annual rates, decimal).
            Pass an empty ``pl.DataFrame()`` to skip estimation and use
            the constructor-supplied rate.

        Returns
        -------
        Interest_Rate_Model_Constant
            The fitted model instance (``self``).

        Raises
        ------
        Exception_Validation_Input
            If *data* is not a :class:`polars.DataFrame`, or if it is
            non-empty and missing required columns or contains invalid
            (non-finite) values.
        """
        if not isinstance(data, pl.DataFrame):
            raise Exception_Validation_Input(
                "data must be a polars.DataFrame",
                field_name="data",
                expected_type=pl.DataFrame,
                actual_value=type(data).__name__,
            )

        if len(data) > 0:
            self._validate_rate_data(data)

            rates = data["short_rate"].to_numpy()

            if not np.all(np.isfinite(rates)):
                raise Exception_Validation_Input(
                    "short_rate column contains non-finite values",
                    field_name="short_rate",
                    expected_type=float,
                    actual_value="non-finite entry",
                )

            self.m_annual_rate = float(np.mean(rates))
            logger.info(
                "Interest_Rate_Model_Constant fitted from %d observations; "
                "estimated annual_rate=%.6f",
                len(rates),
                self.m_annual_rate,
            )
        else:
            logger.info(
                "Interest_Rate_Model_Constant: empty data — using constructor rate=%.6f",
                self.m_annual_rate,
            )

        self.m_status = Interest_Rate_Model_Status.FITTED  # type: ignore[reportAttributeAccessIssue]
        self.m_parameters = {"annual_rate": self.m_annual_rate}
        return self

    # ------------------------------------------------------------------
    # Predict
    # ------------------------------------------------------------------

    def predict(
        self,
        n_periods: int,
        start_date: str | None = None,
        freq: str = "1mo",
    ) -> pl.DataFrame:
        r"""Project a constant short rate over *n_periods* periods.

        Parameters
        ----------
        n_periods : int
            Number of periods to project (must be >= 1).
        start_date : str or None, optional
            ISO-8601 start date of the projection (default
            ``"2025-01-01"``).
        freq : str, optional
            Polars duration string for the period step
            (default ``"1mo"`` for monthly).

        Returns
        -------
        pl.DataFrame
            DataFrame with columns:

            * ``"Date"`` — :class:`polars.Date`
            * ``"short_rate"`` — :class:`polars.Float64`

        Raises
        ------
        Exception_Calculation
            If the model has not been fitted yet.
        Exception_Validation_Input
            If *n_periods* is not a positive integer.

        Examples
        --------
        >>> model = Interest_Rate_Model_Constant(annual_rate=0.05)
        >>> model.fit(pl.DataFrame())
        >>> df = model.predict(n_periods=4, start_date="2025-01-01")
        >>> len(df)
        4
        >>> df["short_rate"].to_list()
        [0.05, 0.05, 0.05, 0.05]
        """
        self._check_is_fitted()
        self._validate_n_periods(n_periods)

        if start_date is None:
            start_date = "2025-01-01"

        dates = _make_date_range(start_date, n_periods, freq)

        logger.info(
            "Interest_Rate_Model_Constant.predict: n_periods=%d, rate=%.6f",
            n_periods,
            self.m_annual_rate,
        )

        return pl.DataFrame(
            {
                "Date": dates,
                "short_rate": [self.m_annual_rate] * n_periods,
            },
        )

    # ------------------------------------------------------------------
    # Scalar rate accessor
    # ------------------------------------------------------------------

    def get_annual_rate(self) -> float:
        """Return the fixed annual short rate.

        Returns
        -------
        float
            Annual short rate in decimal form (e.g. ``0.04``).

        Raises
        ------
        Exception_Calculation
            If the model has not been fitted yet.
        """
        self._check_is_fitted()
        return self.m_annual_rate

    # ------------------------------------------------------------------
    # Dunder overrides
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"Interest_Rate_Model_Constant("
            f"name='{self.m_name_model}', "
            f"rate={self.m_annual_rate:.6f}, "
            f"status={self.m_status.value})"
        )


# ======================================================================
# Private helpers
# ======================================================================


def _make_date_range(start_date: str, n_periods: int, freq: str) -> pl.Series:
    """Build a Polars Date Series of length *n_periods* from *start_date*.

    Parameters
    ----------
    start_date : str
        ISO-8601 start date string (e.g. ``"2025-01-01"``).
    n_periods : int
        Number of dates to generate.
    freq : str
        Polars duration string (``"1d"``, ``"1mo"``, ``"1y"``).

    Returns
    -------
    pl.Series
        Date series of exactly *n_periods* elements.

    Raises
    ------
    Exception_Validation_Input
        If *freq* is not one of ``"1d"``, ``"1mo"``, ``"1y"``.
    """
    supported = {"1d", "1mo", "1y"}
    if freq not in supported:
        raise Exception_Validation_Input(
            f"freq must be one of {sorted(supported)}, got '{freq}'",
            field_name="freq",
            expected_type=str,
            actual_value=freq,
        )

    start = _date.fromisoformat(start_date)

    # Compute a safe end date with buffer
    # (pl.duration does not support months/years as offsets directly)
    if freq == "1d":
        end = start + timedelta(days=n_periods + 2)
    elif freq == "1mo":
        end = start + timedelta(days=(n_periods + 2) * 32)
    else:  # "1y"
        end = start + timedelta(days=(n_periods + 2) * 366)

    raw = pl.date_range(
        start=start,
        end=end,
        interval=freq,
        eager=True,
    )
    return raw.slice(0, n_periods)
