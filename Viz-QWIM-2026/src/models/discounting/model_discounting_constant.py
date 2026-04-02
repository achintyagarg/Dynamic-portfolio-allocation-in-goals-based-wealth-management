r"""
Constant-rate discounting model.

===============================

Discounting model that applies a fixed (constant) annual discount rate
over the entire time horizon.  The discount factor for a given time
$t$ (in years) is:

$$
d(t) = \frac{1}{(1 + r)^{t}}
$$

where $r$ is the constant annual discount rate.

This is the simplest and most widely used discounting approach, suitable
when a flat term structure is assumed or when a single representative
rate is appropriate for the analysis.

Author
------
QWIM Team

Version
-------
0.5.1 (2026-03-01)
"""

from __future__ import annotations

import math

from typing import TYPE_CHECKING

from src.utils.custom_exceptions_errors_loggers.exception_custom import (
    Exception_Validation_Input,
)
from src.utils.custom_exceptions_errors_loggers.logger_custom import get_logger
from src.utils.dates_times_utils.daycount import Daycount_Convention

from .model_discounting_base import Discounting_Model_Base


if TYPE_CHECKING:
    from datetime import date


logger = get_logger(__name__)


class Discounting_Model_Constant(Discounting_Model_Base):
    r"""Constant-rate discounting model.

    Applies a fixed annual discount rate $r$ to compute discount factors:

    $$
    d(t) = \frac{1}{(1 + r)^{t}}
    $$

    This model assumes a **flat term structure** — the same rate is used
    regardless of the time horizon.

    Attributes
    ----------
    m_name_model : str
        Human-readable name identifying the discounting model instance
        (inherited).
    m_discount_rate : float
        The constant annual discount rate (as a decimal, e.g. 0.05 for 5 %).
    """

    def __init__(
        self,
        discount_rate: float,
        name_model: str = "Constant Discount Rate",
    ) -> None:
        r"""Initialize the constant-rate discounting model.

        Parameters
        ----------
        discount_rate : float
            The constant annual discount rate as a decimal (e.g. 0.05 for
            5 %).  Must be a finite number strictly greater than $-1$.
            A rate of 0.0 means no discounting (discount factor is always 1).
        name_model : str, optional
            Human-readable name for this model instance
            (default ``"Constant Discount Rate"``).

        Raises
        ------
        Exception_Validation_Input
            If ``discount_rate`` is not a finite number or is $\le -1$.
        """
        super().__init__(name_model)

        # --- Input validation ---
        if not isinstance(discount_rate, (int, float)):
            raise Exception_Validation_Input(
                "discount_rate must be a numeric value (int or float)",
                field_name="discount_rate",
                expected_type=float,
                actual_value=discount_rate,
            )

        rate_value = float(discount_rate)

        if rate_value <= -1.0:
            raise Exception_Validation_Input(
                "discount_rate must be strictly greater than -1.0",
                field_name="discount_rate",
                expected_type=float,
                actual_value=rate_value,
            )

        # Check for non-finite values (NaN, inf)
        if not math.isfinite(rate_value):
            raise Exception_Validation_Input(
                "discount_rate must be a finite number",
                field_name="discount_rate",
                expected_type=float,
                actual_value=rate_value,
            )

        # --- Store member variables ---
        self.m_discount_rate: float = rate_value

        logger.info(
            "Constant discount rate set to %.4f (%.2f %%)",
            self.m_discount_rate,
            self.m_discount_rate * 100,
        )

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def discount_rate(self) -> float:
        """Return the constant annual discount rate (as a decimal)."""
        return self.m_discount_rate

    # ------------------------------------------------------------------
    # Abstract method implementation
    # ------------------------------------------------------------------

    def calc_discount_factor(
        self,
        end_date: date,
        start_date: date | None = None,
        daycount_convention: Daycount_Convention = Daycount_Convention.ACTUAL_ACTUAL,
    ) -> float:
        r"""Calculate the discount factor over a date period.

        $$
        d(t) = \frac{1}{(1 + r)^{t}}
        $$

        where $t$ is the year fraction between ``start_date`` and
        ``end_date`` computed using the specified day-count convention.

        Parameters
        ----------
        end_date : date
            The future date at which the cash flow is received.
        start_date : date | None, optional
            The valuation date (default: today via ``date.today()``).
        daycount_convention : Daycount_Convention, optional
            Day-count convention used to compute the year fraction
            (default: ``Daycount_Convention.ACTUAL_ACTUAL``).

        Returns
        -------
        float
            The discount factor $d(t)$.

        Raises
        ------
        Exception_Validation_Input
            If dates are invalid or ``end_date`` is before ``start_date``.

        Examples
        --------
        >>> from datetime import date
        >>> model = Discounting_Model_Constant(discount_rate=0.05)
        >>> d = model.calc_discount_factor(
        ...     end_date=date(2025, 1, 1),
        ...     start_date=date(2024, 1, 1),
        ... )
        >>> round(d, 6)
        0.951229
        """
        time_years = self._calc_year_fraction(
            end_date,
            start_date,
            daycount_convention,
        )

        return 1.0 / ((1.0 + self.m_discount_rate) ** time_years)

    # ------------------------------------------------------------------
    # Dunder methods
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        """Return a developer-oriented string representation."""
        return (
            f"{self.__class__.__name__}("
            f"name_model='{self.m_name_model}', "
            f"discount_rate={self.m_discount_rate})"
        )
