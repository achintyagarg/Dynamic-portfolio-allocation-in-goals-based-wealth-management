r"""
Functionalities for daycount conventions.

===============================

Day-count conventions determine how the *year fraction* $\alpha$ between
two dates is computed.  They are fundamental building blocks in
fixed-income analytics, discounting, and accrued-interest calculations.

Common daycount conventions include:

- **30/360**: calculates the daily interest using a 360-day year and then multiplies that by 30 (standardized month).
- **30/365**: calculates the daily interest using a 365-day year and then multiplies that by 30 (standardized month).
- **ACTUAL/360**: calculates the daily interest using a 360-day year and then multiplies that by the actual number of days in each time period.
- **ACTUAL/365**: calculates the daily interest using a 365-day year and then multiplies that by the actual number of days in each time period.
- **ACTUAL/ACTUAL**: calculates the daily interest using the actual number of days in the year and then multiplies that by the actual number of days in each time period.

Author
------
QWIM Team

Version
-------
0.5.1 (2026-03-01)
"""

from __future__ import annotations

import calendar

from abc import ABC, abstractmethod
from datetime import date
from enum import Enum

from src.utils.custom_exceptions_errors_loggers.exception_custom import (
    Exception_Validation_Input,
)
from src.utils.custom_exceptions_errors_loggers.logger_custom import get_logger


logger = get_logger(__name__)


# ======================================================================
# Enumerator
# ======================================================================


class Daycount_Convention(Enum):
    """Enumeration of supported day-count conventions.

    Each member maps to a human-readable label that can be used for
    display purposes in dashboards and reports.

    Examples
    --------
    >>> Daycount_Convention.THIRTY_360.value
    '30/360'
    >>> Daycount_Convention.ACTUAL_ACTUAL.value
    'ACTUAL/ACTUAL'
    """

    THIRTY_360 = "30/360"
    THIRTY_365 = "30/365"
    ACTUAL_360 = "ACTUAL/360"
    ACTUAL_365 = "ACTUAL/365"
    ACTUAL_ACTUAL = "ACTUAL/ACTUAL"


# ======================================================================
# Base class
# ======================================================================


class Daycount_Calculator_Base(ABC):
    r"""Abstract base class for day-count calculators.

    Every concrete calculator must implement :meth:`calc_year_fraction`
    which returns the year fraction $\alpha$ between two dates.

    Attributes
    ----------
    m_convention : Daycount_Convention
        The day-count convention used by this calculator.
    """

    def __init__(self, convention: Daycount_Convention) -> None:
        """Initialize the base day-count calculator.

        Parameters
        ----------
        convention : Daycount_Convention
            The day-count convention that this calculator implements.

        Raises
        ------
        Exception_Validation_Input
            If ``convention`` is not a ``Daycount_Convention`` member.
        """
        if not isinstance(convention, Daycount_Convention):
            raise Exception_Validation_Input(
                "convention must be a Daycount_Convention enum member",
                field_name="convention",
                expected_type=Daycount_Convention,
                actual_value=convention,
            )

        self.m_convention: Daycount_Convention = convention

        logger.info(
            "Created day-count calculator (convention: %s)",
            self.m_convention.value,
        )

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def convention(self) -> Daycount_Convention:
        """Return the day-count convention used by this calculator."""
        return self.m_convention

    # ------------------------------------------------------------------
    # Abstract methods
    # ------------------------------------------------------------------

    @abstractmethod
    def calc_year_fraction(
        self,
        date_start: date,
        date_end: date,
    ) -> float:
        r"""Calculate the year fraction between two dates.

        Parameters
        ----------
        date_start : date
            The start date of the period (inclusive).
        date_end : date
            The end date of the period (exclusive).

        Returns
        -------
        float
            The year fraction $\alpha \ge 0$.

        Raises
        ------
        Exception_Validation_Input
            If ``date_start`` or ``date_end`` is not a ``date`` instance.
            If ``date_end`` is before ``date_start``.
        """

    # ------------------------------------------------------------------
    # Concrete helpers
    # ------------------------------------------------------------------

    def _validate_dates(
        self,
        date_start: date,
        date_end: date,
    ) -> None:
        """Validate that inputs are ``date`` objects and properly ordered.

        Parameters
        ----------
        date_start : date
            The start date.
        date_end : date
            The end date.

        Raises
        ------
        Exception_Validation_Input
            If either argument is not a ``date`` instance or if
            ``date_end`` is before ``date_start``.
        """
        if not isinstance(date_start, date):
            raise Exception_Validation_Input(
                "date_start must be a datetime.date instance",
                field_name="date_start",
                expected_type=date,
                actual_value=date_start,
            )

        if not isinstance(date_end, date):
            raise Exception_Validation_Input(
                "date_end must be a datetime.date instance",
                field_name="date_end",
                expected_type=date,
                actual_value=date_end,
            )

        if date_end < date_start:
            raise Exception_Validation_Input(
                "date_end must not be before date_start",
                field_name="date_end",
                expected_type=date,
                actual_value=f"date_start={date_start}, date_end={date_end}",
            )

    # ------------------------------------------------------------------
    # Dunder methods
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        """Return a developer-oriented string representation."""
        return f"{self.__class__.__name__}(convention='{self.m_convention.value}')"


# ======================================================================
# Concrete implementations
# ======================================================================


class Daycount_Thirty_360(Daycount_Calculator_Base):
    r"""Day-count calculator using the **30/360** convention.

    Every month is assumed to have 30 days and every year 360 days.
    The day-count numerator between two dates
    $(Y_1, M_1, D_1)$ and $(Y_2, M_2, D_2)$ is:

    $$
    \text{days} = 360 \times (Y_2 - Y_1) + 30 \times (M_2 - M_1)
                  + (D_2 - D_1)
    $$

    with the standard adjustment: $D_1$ and $D_2$ are capped at 30.

    The year fraction is then $\alpha = \text{days} / 360$.
    """

    def __init__(self) -> None:
        """Initialize the 30/360 day-count calculator."""
        super().__init__(Daycount_Convention.THIRTY_360)

    def calc_year_fraction(
        self,
        date_start: date,
        date_end: date,
    ) -> float:
        r"""Calculate the year fraction using the 30/360 convention.

        Parameters
        ----------
        date_start : date
            The start date of the period (inclusive).
        date_end : date
            The end date of the period (exclusive).

        Returns
        -------
        float
            The year fraction $\alpha$.

        Raises
        ------
        Exception_Validation_Input
            If inputs are invalid (see base class).

        Examples
        --------
        >>> from datetime import date
        >>> calc = Daycount_Thirty_360()
        >>> calc.calc_year_fraction(date(2024, 1, 1), date(2024, 7, 1))
        0.5
        """
        self._validate_dates(date_start, date_end)

        d1 = min(date_start.day, 30)
        d2 = min(date_end.day, 30) if d1 == 30 else date_end.day
        d2 = min(d2, 30)

        days = (
            360 * (date_end.year - date_start.year)
            + 30 * (date_end.month - date_start.month)
            + (d2 - d1)
        )

        return days / 360.0


class Daycount_Thirty_365(Daycount_Calculator_Base):
    r"""Day-count calculator using the **30/365** convention.

    Every month is assumed to have 30 days and the year has 365 days.
    The day-count numerator is computed the same way as 30/360 but the
    denominator is 365:

    $$
    \alpha = \frac{360 (Y_2 - Y_1) + 30 (M_2 - M_1) + (D_2 - D_1)}{365}
    $$
    """

    def __init__(self) -> None:
        """Initialize the 30/365 day-count calculator."""
        super().__init__(Daycount_Convention.THIRTY_365)

    def calc_year_fraction(
        self,
        date_start: date,
        date_end: date,
    ) -> float:
        r"""Calculate the year fraction using the 30/365 convention.

        Parameters
        ----------
        date_start : date
            The start date of the period (inclusive).
        date_end : date
            The end date of the period (exclusive).

        Returns
        -------
        float
            The year fraction $\alpha$.

        Raises
        ------
        Exception_Validation_Input
            If inputs are invalid (see base class).

        Examples
        --------
        >>> from datetime import date
        >>> calc = Daycount_Thirty_365()
        >>> round(calc.calc_year_fraction(date(2024, 1, 1), date(2024, 7, 1)), 6)
        0.493151
        """
        self._validate_dates(date_start, date_end)

        d1 = min(date_start.day, 30)
        d2 = min(date_end.day, 30) if d1 == 30 else date_end.day
        d2 = min(d2, 30)

        days = (
            360 * (date_end.year - date_start.year)
            + 30 * (date_end.month - date_start.month)
            + (d2 - d1)
        )

        return days / 365.0


class Daycount_Actual_360(Daycount_Calculator_Base):
    r"""Day-count calculator using the **ACTUAL/360** convention.

    Uses the actual number of calendar days elapsed as the numerator
    and 360 as the denominator:

    $$
    \alpha = \frac{\text{actual days}}{360}
    $$

    This convention is common in money-market instruments.
    """

    def __init__(self) -> None:
        """Initialize the ACTUAL/360 day-count calculator."""
        super().__init__(Daycount_Convention.ACTUAL_360)

    def calc_year_fraction(
        self,
        date_start: date,
        date_end: date,
    ) -> float:
        r"""Calculate the year fraction using the ACTUAL/360 convention.

        Parameters
        ----------
        date_start : date
            The start date of the period (inclusive).
        date_end : date
            The end date of the period (exclusive).

        Returns
        -------
        float
            The year fraction $\alpha$.

        Raises
        ------
        Exception_Validation_Input
            If inputs are invalid (see base class).

        Examples
        --------
        >>> from datetime import date
        >>> calc = Daycount_Actual_360()
        >>> round(calc.calc_year_fraction(date(2024, 1, 1), date(2024, 7, 1)), 6)
        0.505556
        """
        self._validate_dates(date_start, date_end)

        actual_days = (date_end - date_start).days

        return actual_days / 360.0


class Daycount_Actual_365(Daycount_Calculator_Base):
    r"""Day-count calculator using the **ACTUAL/365** convention.

    Uses the actual number of calendar days elapsed as the numerator
    and a fixed 365 as the denominator (ignoring leap years):

    $$
    \alpha = \frac{\text{actual days}}{365}
    $$

    This convention is also known as **ACTUAL/365 Fixed**.
    """

    def __init__(self) -> None:
        """Initialize the ACTUAL/365 day-count calculator."""
        super().__init__(Daycount_Convention.ACTUAL_365)

    def calc_year_fraction(
        self,
        date_start: date,
        date_end: date,
    ) -> float:
        r"""Calculate the year fraction using the ACTUAL/365 convention.

        Parameters
        ----------
        date_start : date
            The start date of the period (inclusive).
        date_end : date
            The end date of the period (exclusive).

        Returns
        -------
        float
            The year fraction $\alpha$.

        Raises
        ------
        Exception_Validation_Input
            If inputs are invalid (see base class).

        Examples
        --------
        >>> from datetime import date
        >>> calc = Daycount_Actual_365()
        >>> calc.calc_year_fraction(date(2024, 1, 1), date(2025, 1, 1))
        1.0027397260273974
        """
        self._validate_dates(date_start, date_end)

        actual_days = (date_end - date_start).days

        return actual_days / 365.0


class Daycount_Actual_Actual(Daycount_Calculator_Base):
    r"""Day-count calculator using the **ACTUAL/ACTUAL** convention.

    Uses the actual number of calendar days elapsed as the numerator
    and the actual number of days in each year as the denominator.
    When the period spans multiple years, each year's contribution is
    computed separately:

    $$
    \alpha = \sum_{y} \frac{\text{days in year } y}
                           {\text{total days in year } y}
    $$

    where the total days in year $y$ is 366 for a leap year and 365
    otherwise.  This is the **ISDA** variant of the ACTUAL/ACTUAL
    convention.
    """

    def __init__(self) -> None:
        """Initialize the ACTUAL/ACTUAL day-count calculator."""
        super().__init__(Daycount_Convention.ACTUAL_ACTUAL)

    def calc_year_fraction(
        self,
        date_start: date,
        date_end: date,
    ) -> float:
        r"""Calculate the year fraction using the ACTUAL/ACTUAL convention.

        Parameters
        ----------
        date_start : date
            The start date of the period (inclusive).
        date_end : date
            The end date of the period (exclusive).

        Returns
        -------
        float
            The year fraction $\alpha$.

        Raises
        ------
        Exception_Validation_Input
            If inputs are invalid (see base class).

        Examples
        --------
        >>> from datetime import date
        >>> calc = Daycount_Actual_Actual()
        >>> calc.calc_year_fraction(date(2024, 1, 1), date(2025, 1, 1))
        1.0
        """
        self._validate_dates(date_start, date_end)

        if date_start == date_end:
            return 0.0

        year_fraction = 0.0
        current = date_start

        for year in range(date_start.year, date_end.year + 1):
            year_start = max(current, date(year, 1, 1))
            year_end = min(date_end, date(year + 1, 1, 1))

            if year_start >= year_end:
                continue

            days_in_year = 366 if calendar.isleap(year) else 365
            days_in_period = (year_end - year_start).days

            year_fraction += days_in_period / days_in_year

        return year_fraction


# ======================================================================
# Factory function
# ======================================================================


def get_daycount_calculator(
    convention: Daycount_Convention,
) -> Daycount_Calculator_Base:
    """Return a day-count calculator for the given convention.

    This is the recommended way to obtain calculator instances.

    Parameters
    ----------
    convention : Daycount_Convention
        The desired day-count convention.

    Returns
    -------
    Daycount_Calculator_Base
        A concrete calculator implementing the requested convention.

    Raises
    ------
    Exception_Validation_Input
        If ``convention`` is not a valid ``Daycount_Convention`` member.

    Examples
    --------
    >>> from datetime import date
    >>> calc = get_daycount_calculator(Daycount_Convention.ACTUAL_365)
    >>> round(calc.calc_year_fraction(date(2024, 1, 1), date(2024, 4, 1)), 6)
    0.249315
    """
    if not isinstance(convention, Daycount_Convention):
        raise Exception_Validation_Input(
            "convention must be a Daycount_Convention enum member",
            field_name="convention",
            expected_type=Daycount_Convention,
            actual_value=convention,
        )

    if convention is Daycount_Convention.THIRTY_360:
        return Daycount_Thirty_360()
    if convention is Daycount_Convention.THIRTY_365:
        return Daycount_Thirty_365()
    if convention is Daycount_Convention.ACTUAL_360:
        return Daycount_Actual_360()
    if convention is Daycount_Convention.ACTUAL_365:
        return Daycount_Actual_365()

    return Daycount_Actual_Actual()
