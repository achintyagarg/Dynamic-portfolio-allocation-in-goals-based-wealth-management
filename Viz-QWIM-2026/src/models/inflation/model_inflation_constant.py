r"""Constant inflation model for QWIM retirement-planning pipeline.

========================================================

Provides :class:`Inflation_Model_Constant`, a deterministic model that
projects a single fixed (constant) annual inflation rate for every
future period.

This is the simplest possible inflation assumption and is commonly used
as a baseline scenario or sensitivity anchor in retirement-income
models.  When historical data is supplied to :meth:`fit`, the model
estimates the rate as the arithmetic mean of the observed annual
inflation rates; otherwise it uses the constructor-supplied rate
directly.

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

from src.models.inflation.model_inflation_base import (
    Inflation_Model_Base,
    Inflation_Model_Status,
)
from src.utils.custom_exceptions_errors_loggers.exception_custom import (
    Exception_Validation_Input,
)
from src.utils.custom_exceptions_errors_loggers.logger_custom import get_logger


logger = get_logger(__name__)

# ======================================================================
# Constant inflation model
# ======================================================================

_DEFAULT_ANNUAL_RATE: float = 0.025  # 2.5 % — US long-run average


class Inflation_Model_Constant(Inflation_Model_Base):  # noqa: N801
    r"""Deterministic constant-rate inflation model.

    The model projects the same annual inflation rate, :math:`\pi`, for
    every future period:

    .. math::

        \pi_t = \pi \quad \forall\; t \geq 0

    where :math:`\pi` is either supplied directly via the constructor or
    estimated from historical data as the arithmetic mean of observed
    annual inflation rates.

    Parameters
    ----------
    annual_rate : float, optional
        Fixed annual inflation rate in decimal form (default ``0.025``
        for 2.5 %).  Used when ``fit()`` is called without data or
        when acting as a hard-coded assumption.
    name_model : str, optional
        Human-readable label (default ``"Constant Inflation"``)

    Examples
    --------
    Hard-coded rate, no fitting required:

    >>> model = Inflation_Model_Constant(annual_rate=0.03)
    >>> model.fit(pl.DataFrame())  # no-op fit
    Inflation_Model_Constant(name='Constant Inflation', rate=0.030000, status=Fitted)
    >>> df = model.predict(n_periods=3, start_date="2025-01-01")
    >>> df["inflation_rate"].to_list()
    [0.03, 0.03, 0.03]

    Fit to historical data:

    >>> import polars as pl
    >>> hist = pl.DataFrame(
    ...     {
    ...         "Date": ["2022-01-01", "2023-01-01", "2024-01-01"],
    ...         "inflation_rate": [0.07, 0.04, 0.031],
    ...     }
    ... )
    >>> model = Inflation_Model_Constant()
    >>> model.fit(hist)
    Inflation_Model_Constant(name='Constant Inflation', rate=0.047000, status=Fitted)
    """

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def __init__(
        self,
        annual_rate: float = _DEFAULT_ANNUAL_RATE,
        name_model: str = "Constant Inflation",
    ) -> None:
        super().__init__(name_model=name_model)

        if not isinstance(annual_rate, float | int) or not np.isfinite(annual_rate):
            raise Exception_Validation_Input(
                "annual_rate must be a finite number",
                field_name="annual_rate",
                expected_type=float,
                actual_value=annual_rate,
            )
        if annual_rate < -0.5 or annual_rate > 1.0:
            raise Exception_Validation_Input(
                "annual_rate must be in the range [-0.5, 1.0]",
                field_name="annual_rate",
                expected_type=float,
                actual_value=annual_rate,
            )

        self.m_annual_rate: float = float(annual_rate)

        logger.info(
            "Inflation_Model_Constant constructed with annual_rate=%.6f",
            self.m_annual_rate,
        )

    # ------------------------------------------------------------------
    # Fit
    # ------------------------------------------------------------------

    def fit(self, data: pl.DataFrame) -> Inflation_Model_Constant:
        """Fit (calibrate) the constant model to historical inflation data.

        If *data* is a non-empty DataFrame with the required columns the
        model updates :attr:`m_annual_rate` to the arithmetic mean of
        the observed ``"inflation_rate"`` column.  If an empty DataFrame
        is passed the constructor rate is kept unchanged.

        Parameters
        ----------
        data : pl.DataFrame
            Historical data.  If non-empty, must contain columns
            ``"Date"`` and ``"inflation_rate"`` (annual rates, decimal).
            Pass an empty ``pl.DataFrame()`` to skip estimation and use
            the constructor-supplied rate.

        Returns
        -------
        Inflation_Model_Constant
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
            self._validate_inflation_data(data)

            rates = data["inflation_rate"].to_numpy()

            if not np.all(np.isfinite(rates)):
                raise Exception_Validation_Input(
                    "inflation_rate column contains non-finite values",
                    field_name="inflation_rate",
                    expected_type=float,
                    actual_value="non-finite entry",
                )

            self.m_annual_rate = float(np.mean(rates))
            logger.info(
                "Inflation_Model_Constant fitted from %d observations; estimated annual_rate=%.6f",
                len(rates),
                self.m_annual_rate,
            )
        else:
            logger.info(
                "Inflation_Model_Constant: empty data — using constructor rate=%.6f",
                self.m_annual_rate,
            )

        self.m_status = Inflation_Model_Status.FITTED  # type: ignore[reportAttributeAccessIssue]
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
        r"""Project a constant inflation rate over *n_periods* periods.

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
            * ``"inflation_rate"`` — :class:`polars.Float64`

        Raises
        ------
        Exception_Calculation
            If the model has not been fitted yet.
        Exception_Validation_Input
            If *n_periods* is not a positive integer.

        Examples
        --------
        >>> model = Inflation_Model_Constant(annual_rate=0.02)
        >>> model.fit(pl.DataFrame())
        >>> df = model.predict(n_periods=4, start_date="2025-01-01")
        >>> len(df)
        4
        >>> df["inflation_rate"].to_list()
        [0.02, 0.02, 0.02, 0.02]
        """
        self._check_is_fitted()
        self._validate_n_periods(n_periods)

        if start_date is None:
            start_date = "2025-01-01"

        dates = _make_date_range(start_date, n_periods, freq)

        logger.info(
            "Inflation_Model_Constant.predict: n_periods=%d, rate=%.6f",
            n_periods,
            self.m_annual_rate,
        )

        return pl.DataFrame(
            {
                "Date": dates,
                "inflation_rate": [self.m_annual_rate] * n_periods,
            },
        )

    # ------------------------------------------------------------------
    # Scalar rate accessor
    # ------------------------------------------------------------------

    def get_annual_rate(self) -> float:
        """Return the fixed annual inflation rate.

        Returns
        -------
        float
            Annual inflation rate in decimal form (e.g. ``0.025``).

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
            f"Inflation_Model_Constant("
            f"name='{self.m_name_model}', "
            f"rate={self.m_annual_rate:.6f}, "
            f"status={self.m_status.value})"
        )


# ======================================================================
# Private helpers (shared with model_inflation_standard)
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

    # Compute a safe end date with buffer (months/years not in pl.duration)
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
