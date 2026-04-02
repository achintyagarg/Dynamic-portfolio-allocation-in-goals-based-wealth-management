r"""
Base class for discounting models.

===============================

Abstract base class defining the interface and common functionality
for all discounting models.  A discounting model computes the
**present value** of a future cash flow by applying a discount factor:

$$
PV = CF \times d(t)
$$

where $CF$ is the future cash flow, $d(t)$ is the discount factor
at time $t$, and $PV$ is the present value.

The discount factor satisfies $d(0) = 1$ and $0 < d(t) \le 1$
for all $t > 0$ (assuming non-negative rates).

Author
------
QWIM Team

Version
-------
0.5.1 (2026-03-01)
"""

from __future__ import annotations

import datetime

from abc import ABC, abstractmethod
from datetime import date

from src.utils.custom_exceptions_errors_loggers.exception_custom import (
    Exception_Validation_Input,
)
from src.utils.custom_exceptions_errors_loggers.logger_custom import get_logger
from src.utils.dates_times_utils.daycount import (
    Daycount_Convention,
    get_daycount_calculator,
)


logger = get_logger(__name__)


class Discounting_Model_Base(ABC):
    r"""Abstract base class for discounting models.

    Every concrete discounting model must implement :meth:`calc_discount_factor`
    which returns the discount factor $d(t)$ for a given time horizon.

    The present value of a single future cash flow is then:

    $$
    PV = CF \times d(t)
    $$

    Attributes
    ----------
    m_name_model : str
        Human-readable name identifying the discounting model instance.
    """

    def __init__(
        self,
        name_model: str,
    ) -> None:
        """Initialize the base discounting model.

        Parameters
        ----------
        name_model : str
            Human-readable name for this discounting model instance.

        Raises
        ------
        Exception_Validation_Input
            If ``name_model`` is not a non-empty string.
        """
        # --- Input validation ---
        if not isinstance(name_model, str) or len(name_model.strip()) == 0:
            raise Exception_Validation_Input(
                "name_model must be a non-empty string",
                field_name="name_model",
                expected_type=str,
                actual_value=name_model,
            )

        # --- Store member variables ---
        self.m_name_model: str = name_model.strip()

        logger.info(
            "Created discounting model '%s' (type: %s)",
            self.m_name_model,
            self.__class__.__name__,
        )

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def name_model(self) -> str:
        """Return the human-readable name of the discounting model."""
        return self.m_name_model

    # ------------------------------------------------------------------
    # Abstract methods
    # ------------------------------------------------------------------

    @abstractmethod
    def calc_discount_factor(
        self,
        end_date: date,
        start_date: date | None = None,
        daycount_convention: Daycount_Convention = Daycount_Convention.ACTUAL_ACTUAL,
    ) -> float:
        r"""Calculate the discount factor over a date period.

        Concrete implementations must return a value $d(t)$ such that the
        present value of a cash flow $CF$ received at ``end_date`` is
        $PV = CF \times d(t)$, where $t$ is the year fraction between
        ``start_date`` and ``end_date`` computed using the specified
        day-count convention.

        Parameters
        ----------
        end_date : date
            The future date at which the cash flow is received.
        start_date : date | None, optional
            The valuation date (default: today via ``date.today()``).
        daycount_convention : Daycount_Convention, optional
            The day-count convention used to compute the year fraction
            (default: ``Daycount_Convention.ACTUAL_ACTUAL``).

        Returns
        -------
        float
            The discount factor $d(t)$ in the interval $(0, 1]$.

        Raises
        ------
        Exception_Validation_Input
            If ``end_date`` or ``start_date`` are not ``date`` instances.
            If ``end_date`` is before ``start_date``.
        """

    # ------------------------------------------------------------------
    # Concrete methods
    # ------------------------------------------------------------------

    def _resolve_start_date(self, start_date: date | None) -> date:
        """Return ``start_date`` or today's date when ``None``.

        Parameters
        ----------
        start_date : date | None
            Explicit start date, or ``None`` for today.

        Returns
        -------
        date
            The resolved start date.
        """
        if start_date is None:
            return datetime.datetime.now(tz=datetime.UTC).date()
        return start_date

    def _calc_year_fraction(
        self,
        end_date: date,
        start_date: date | None = None,
        daycount_convention: Daycount_Convention = Daycount_Convention.ACTUAL_ACTUAL,
    ) -> float:
        r"""Compute the year fraction between two dates.

        Parameters
        ----------
        end_date : date
            The future date.
        start_date : date | None, optional
            The valuation date (default: today).
        daycount_convention : Daycount_Convention, optional
            Day-count convention (default: ``ACTUAL_ACTUAL``).

        Returns
        -------
        float
            The year fraction $\alpha \ge 0$.

        Raises
        ------
        Exception_Validation_Input
            If dates are invalid or ``end_date`` is before ``start_date``.
        """
        resolved_start = self._resolve_start_date(start_date)

        if not isinstance(end_date, date):
            raise Exception_Validation_Input(
                "end_date must be a datetime.date instance",
                field_name="end_date",
                expected_type=date,
                actual_value=end_date,
            )

        if not isinstance(resolved_start, date):
            raise Exception_Validation_Input(
                "start_date must be a datetime.date instance",
                field_name="start_date",
                expected_type=date,
                actual_value=resolved_start,
            )

        calculator = get_daycount_calculator(daycount_convention)
        return calculator.calc_year_fraction(resolved_start, end_date)

    def calc_present_value(
        self,
        cash_flow: float,
        end_date: date,
        start_date: date | None = None,
        daycount_convention: Daycount_Convention = Daycount_Convention.ACTUAL_ACTUAL,
    ) -> float:
        r"""Calculate the present value of a single future cash flow.

        $$
        PV = CF \times d(t)
        $$

        Parameters
        ----------
        cash_flow : float
            The future cash flow amount.
        end_date : date
            The date at which the cash flow is received.
        start_date : date | None, optional
            The valuation date (default: today via ``date.today()``).
        daycount_convention : Daycount_Convention, optional
            Day-count convention (default: ``ACTUAL_ACTUAL``).

        Returns
        -------
        float
            The present value of the cash flow.

        Raises
        ------
        Exception_Validation_Input
            If ``cash_flow`` is not a finite number.
            If dates are invalid.
        """
        # --- Input validation ---
        if not isinstance(cash_flow, (int, float)):
            raise Exception_Validation_Input(
                "cash_flow must be a numeric value",
                field_name="cash_flow",
                expected_type=float,
                actual_value=cash_flow,
            )

        discount_factor = self.calc_discount_factor(
            end_date,
            start_date,
            daycount_convention,
        )
        return float(cash_flow) * discount_factor

    def calc_present_value_stream(
        self,
        cash_flows: list[float],
        end_dates: list[date],
        start_date: date | None = None,
        daycount_convention: Daycount_Convention = Daycount_Convention.ACTUAL_ACTUAL,
    ) -> float:
        r"""Calculate the present value of a stream of future cash flows.

        $$
        PV = \sum_{i=1}^{n} CF_i \times d(t_i)
        $$

        Parameters
        ----------
        cash_flows : list[float]
            The future cash flow amounts.
        end_dates : list[date]
            Corresponding future dates for each cash flow.
        start_date : date | None, optional
            The common valuation date (default: today via ``date.today()``).
        daycount_convention : Daycount_Convention, optional
            Day-count convention (default: ``ACTUAL_ACTUAL``).

        Returns
        -------
        float
            The total present value of the cash flow stream.

        Raises
        ------
        Exception_Validation_Input
            If ``cash_flows`` and ``end_dates`` have different lengths.
            If either list is empty.
        """
        # --- Input validation ---
        if not isinstance(cash_flows, list) or not isinstance(end_dates, list):
            raise Exception_Validation_Input(
                "cash_flows and end_dates must be lists",
                field_name="cash_flows / end_dates",
                expected_type=list,
                actual_value=type(cash_flows).__name__,
            )

        if len(cash_flows) != len(end_dates):
            raise Exception_Validation_Input(
                "cash_flows and end_dates must have the same length",
                field_name="cash_flows / end_dates",
                expected_type=list,
                actual_value=f"len(cash_flows)={len(cash_flows)}, len(end_dates)={len(end_dates)}",
            )

        if len(cash_flows) == 0:
            raise Exception_Validation_Input(
                "cash_flows must not be empty",
                field_name="cash_flows",
                expected_type=list,
                actual_value="empty list",
            )

        return sum(
            self.calc_present_value(cf, ed, start_date, daycount_convention)
            for cf, ed in zip(cash_flows, end_dates, strict=True)
        )

    # ------------------------------------------------------------------
    # Dunder methods
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        """Return a developer-oriented string representation."""
        return f"{self.__class__.__name__}(name_model='{self.m_name_model}')"
