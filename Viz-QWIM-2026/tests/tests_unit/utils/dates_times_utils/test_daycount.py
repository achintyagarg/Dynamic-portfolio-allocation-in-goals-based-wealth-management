"""Unit tests for the daycount module.

Tests for day-count convention enumerator, abstract base class,
all five concrete calculators (30/360, 30/365, ACTUAL/360, ACTUAL/365,
ACTUAL/ACTUAL), and the factory function.

Author: QWIM Team
Version: 0.5.1
"""

from __future__ import annotations

from datetime import date

import pytest

from src.utils.custom_exceptions_errors_loggers.exception_custom import (
    Exception_Validation_Input,
)
from src.utils.dates_times_utils.daycount import (
    Daycount_Actual_360,
    Daycount_Actual_365,
    Daycount_Actual_Actual,
    Daycount_Calculator_Base,
    Daycount_Convention,
    Daycount_Thirty_360,
    Daycount_Thirty_365,
    get_daycount_calculator,
)


# ==============================================================================
# Fixtures
# ==============================================================================


@pytest.fixture()
def date_jan1_2024() -> date:
    """Return 2024-01-01 (leap year)."""
    return date(2024, 1, 1)


@pytest.fixture()
def date_jul1_2024() -> date:
    """Return 2024-07-01."""
    return date(2024, 7, 1)


@pytest.fixture()
def date_jan1_2025() -> date:
    """Return 2025-01-01 (non-leap year)."""
    return date(2025, 1, 1)


@pytest.fixture()
def date_jan1_2023() -> date:
    """Return 2023-01-01 (non-leap year)."""
    return date(2023, 1, 1)


@pytest.fixture()
def date_mar1_2024() -> date:
    """Return 2024-03-01."""
    return date(2024, 3, 1)


@pytest.fixture()
def date_feb28_2024() -> date:
    """Return 2024-02-28."""
    return date(2024, 2, 28)


@pytest.fixture()
def date_feb29_2024() -> date:
    """Return 2024-02-29 (leap day)."""
    return date(2024, 2, 29)


@pytest.fixture()
def date_apr1_2024() -> date:
    """Return 2024-04-01."""
    return date(2024, 4, 1)


@pytest.fixture()
def calc_30_360() -> Daycount_Thirty_360:
    """Return a 30/360 calculator."""
    return Daycount_Thirty_360()


@pytest.fixture()
def calc_30_365() -> Daycount_Thirty_365:
    """Return a 30/365 calculator."""
    return Daycount_Thirty_365()


@pytest.fixture()
def calc_act_360() -> Daycount_Actual_360:
    """Return an ACTUAL/360 calculator."""
    return Daycount_Actual_360()


@pytest.fixture()
def calc_act_365() -> Daycount_Actual_365:
    """Return an ACTUAL/365 calculator."""
    return Daycount_Actual_365()


@pytest.fixture()
def calc_act_act() -> Daycount_Actual_Actual:
    """Return an ACTUAL/ACTUAL calculator."""
    return Daycount_Actual_Actual()


# ==============================================================================
# Tests: Daycount_Convention Enum
# ==============================================================================


class Test_Daycount_Convention_Enum:
    """Test the Daycount_Convention enumeration."""

    def test_all_members_exist(self):
        members = list(Daycount_Convention)
        assert len(members) == 5

    @pytest.mark.parametrize(
        ("member", "expected_value"),
        [
            (Daycount_Convention.THIRTY_360, "30/360"),
            (Daycount_Convention.THIRTY_365, "30/365"),
            (Daycount_Convention.ACTUAL_360, "ACTUAL/360"),
            (Daycount_Convention.ACTUAL_365, "ACTUAL/365"),
            (Daycount_Convention.ACTUAL_ACTUAL, "ACTUAL/ACTUAL"),
        ],
    )
    def test_member_values(self, member, expected_value):
        assert member.value == expected_value

    def test_members_are_unique(self):
        values = [m.value for m in Daycount_Convention]
        assert len(values) == len(set(values))


# ==============================================================================
# Tests: Daycount_Calculator_Base
# ==============================================================================


class Test_Daycount_Calculator_Base:
    """Test the abstract base class."""

    def test_cannot_instantiate_abstract_class(self):
        with pytest.raises(TypeError):
            Daycount_Calculator_Base(Daycount_Convention.THIRTY_360)  # type: ignore[abstract]

    def test_convention_property(self, calc_30_360):
        assert calc_30_360.convention is Daycount_Convention.THIRTY_360

    def test_repr(self, calc_30_360):
        assert repr(calc_30_360) == "Daycount_Thirty_360(convention='30/360')"

    def test_invalid_convention_type(self):
        with pytest.raises(Exception_Validation_Input):
            Daycount_Thirty_360.__new__(Daycount_Thirty_360)
            Daycount_Calculator_Base.__init__(
                Daycount_Thirty_360.__new__(Daycount_Thirty_360),
                "not_an_enum",  # type: ignore[arg-type]
            )


# ==============================================================================
# Tests: Date Validation (shared across all calculators)
# ==============================================================================


class Test_Date_Validation:
    """Test date validation logic inherited from the base class."""

    def test_invalid_date_start_type(self, calc_30_360):
        with pytest.raises(Exception_Validation_Input, match="date_start"):
            calc_30_360.calc_year_fraction("2024-01-01", date(2024, 7, 1))  # type: ignore[arg-type]

    def test_invalid_date_end_type(self, calc_30_360, date_jan1_2024):
        with pytest.raises(Exception_Validation_Input, match="date_end"):
            calc_30_360.calc_year_fraction(date_jan1_2024, "2024-07-01")  # type: ignore[arg-type]

    def test_end_before_start_raises(self, calc_30_360, date_jan1_2024, date_jul1_2024):
        with pytest.raises(Exception_Validation_Input, match="date_end must not be before"):
            calc_30_360.calc_year_fraction(date_jul1_2024, date_jan1_2024)

    def test_same_dates_returns_zero(self, calc_30_360, date_jan1_2024):
        assert calc_30_360.calc_year_fraction(date_jan1_2024, date_jan1_2024) == 0.0

    @pytest.mark.parametrize(
        "calc_fixture",
        [
            "calc_30_360",
            "calc_30_365",
            "calc_act_360",
            "calc_act_365",
            "calc_act_act",
        ],
    )
    def test_same_dates_zero_all_conventions(
        self,
        calc_fixture,
        date_jan1_2024,
        request,
    ):
        calc = request.getfixturevalue(calc_fixture)
        assert calc.calc_year_fraction(date_jan1_2024, date_jan1_2024) == 0.0


# ==============================================================================
# Tests: Daycount_Thirty_360
# ==============================================================================


class Test_Daycount_Thirty_360:
    """Test the 30/360 day-count calculator.

    30/360: daily interest = annual_rate / 360, accrued over 30-day months.
    Year fraction = (360*(Y2-Y1) + 30*(M2-M1) + (D2-D1)) / 360.
    """

    def test_convention_attribute(self, calc_30_360):
        assert calc_30_360.convention is Daycount_Convention.THIRTY_360

    def test_half_year(self, calc_30_360, date_jan1_2024, date_jul1_2024):
        # 6 months → 180 / 360 = 0.5
        assert calc_30_360.calc_year_fraction(date_jan1_2024, date_jul1_2024) == 0.5

    def test_full_year(self, calc_30_360, date_jan1_2024, date_jan1_2025):
        # 12 months → 360 / 360 = 1.0
        assert calc_30_360.calc_year_fraction(date_jan1_2024, date_jan1_2025) == 1.0

    def test_one_month(self, calc_30_360):
        # 1 month → 30 / 360
        result = calc_30_360.calc_year_fraction(date(2024, 1, 1), date(2024, 2, 1))
        assert result == pytest.approx(30 / 360)

    def test_quarter(self, calc_30_360, date_jan1_2024, date_apr1_2024):
        # 3 months → 90 / 360 = 0.25
        assert calc_30_360.calc_year_fraction(date_jan1_2024, date_apr1_2024) == 0.25

    def test_day_capping_at_30(self, calc_30_360):
        # Both D1=31 and D2=31 → capped to 30
        result = calc_30_360.calc_year_fraction(date(2024, 1, 31), date(2024, 3, 31))
        expected = (30 * 2 + (30 - 30)) / 360  # 60 / 360
        assert result == pytest.approx(expected)

    def test_two_years(self, calc_30_360, date_jan1_2023, date_jan1_2025):
        # 2 years → 720 / 360 = 2.0
        assert calc_30_360.calc_year_fraction(date_jan1_2023, date_jan1_2025) == 2.0


# ==============================================================================
# Tests: Daycount_Thirty_365
# ==============================================================================


class Test_Daycount_Thirty_365:
    """Test the 30/365 day-count calculator.

    30/365: daily interest = annual_rate / 365, accrued over 30-day months.
    Year fraction = (360*(Y2-Y1) + 30*(M2-M1) + (D2-D1)) / 365.
    """

    def test_convention_attribute(self, calc_30_365):
        assert calc_30_365.convention is Daycount_Convention.THIRTY_365

    def test_half_year(self, calc_30_365, date_jan1_2024, date_jul1_2024):
        # 180 / 365
        assert calc_30_365.calc_year_fraction(date_jan1_2024, date_jul1_2024) == pytest.approx(
            180 / 365,
        )

    def test_full_year(self, calc_30_365, date_jan1_2024, date_jan1_2025):
        # 360 / 365
        assert calc_30_365.calc_year_fraction(date_jan1_2024, date_jan1_2025) == pytest.approx(
            360 / 365,
        )

    def test_one_month(self, calc_30_365):
        result = calc_30_365.calc_year_fraction(date(2024, 1, 1), date(2024, 2, 1))
        assert result == pytest.approx(30 / 365)

    def test_quarter(self, calc_30_365, date_jan1_2024, date_apr1_2024):
        # 90 / 365
        assert calc_30_365.calc_year_fraction(date_jan1_2024, date_apr1_2024) == pytest.approx(
            90 / 365,
        )

    def test_denominator_is_365_not_360(self, calc_30_365, date_jan1_2024, date_jul1_2024):
        calc_360 = Daycount_Thirty_360()
        result_365 = calc_30_365.calc_year_fraction(date_jan1_2024, date_jul1_2024)
        result_360 = calc_360.calc_year_fraction(date_jan1_2024, date_jul1_2024)
        # Same numerator (180), different denominator → 30/365 < 30/360
        assert result_365 < result_360


# ==============================================================================
# Tests: Daycount_Actual_360
# ==============================================================================


class Test_Daycount_Actual_360:
    """Test the ACTUAL/360 day-count calculator.

    ACTUAL/360: daily interest = annual_rate / 360, accrued over actual days.
    Year fraction = actual_days / 360.
    """

    def test_convention_attribute(self, calc_act_360):
        assert calc_act_360.convention is Daycount_Convention.ACTUAL_360

    def test_half_year_leap(self, calc_act_360, date_jan1_2024, date_jul1_2024):
        # 2024 is leap: Jan(31)+Feb(29)+Mar(31)+Apr(30)+May(31)+Jun(30) = 182 days
        actual_days = (date(2024, 7, 1) - date(2024, 1, 1)).days
        assert actual_days == 182
        assert calc_act_360.calc_year_fraction(date_jan1_2024, date_jul1_2024) == pytest.approx(
            182 / 360,
        )

    def test_full_year_leap(self, calc_act_360, date_jan1_2024, date_jan1_2025):
        # 2024 leap → 366 days / 360
        assert calc_act_360.calc_year_fraction(date_jan1_2024, date_jan1_2025) == pytest.approx(
            366 / 360,
        )

    def test_full_year_non_leap(self, calc_act_360, date_jan1_2023, date_jan1_2024):
        # 2023 non-leap → 365 days / 360
        assert calc_act_360.calc_year_fraction(date_jan1_2023, date_jan1_2024) == pytest.approx(
            365 / 360,
        )

    def test_one_day(self, calc_act_360):
        result = calc_act_360.calc_year_fraction(date(2024, 3, 15), date(2024, 3, 16))
        assert result == pytest.approx(1 / 360)

    def test_90_days(self, calc_act_360):
        # 90 actual days / 360 = 0.25
        result = calc_act_360.calc_year_fraction(date(2025, 1, 1), date(2025, 4, 1))
        actual_days = (date(2025, 4, 1) - date(2025, 1, 1)).days  # 90
        assert result == pytest.approx(actual_days / 360)


# ==============================================================================
# Tests: Daycount_Actual_365
# ==============================================================================


class Test_Daycount_Actual_365:
    """Test the ACTUAL/365 day-count calculator.

    ACTUAL/365: daily interest = annual_rate / 365, accrued over actual days.
    Year fraction = actual_days / 365.
    """

    def test_convention_attribute(self, calc_act_365):
        assert calc_act_365.convention is Daycount_Convention.ACTUAL_365

    def test_full_year_non_leap(self, calc_act_365, date_jan1_2023, date_jan1_2024):
        # 365 / 365 = 1.0
        assert calc_act_365.calc_year_fraction(date_jan1_2023, date_jan1_2024) == pytest.approx(
            1.0,
        )

    def test_full_year_leap(self, calc_act_365, date_jan1_2024, date_jan1_2025):
        # 366 / 365 > 1.0 (fixed 365 denominator)
        result = calc_act_365.calc_year_fraction(date_jan1_2024, date_jan1_2025)
        assert result == pytest.approx(366 / 365)
        assert result > 1.0

    def test_half_year_leap(self, calc_act_365, date_jan1_2024, date_jul1_2024):
        actual_days = (date(2024, 7, 1) - date(2024, 1, 1)).days  # 182
        assert calc_act_365.calc_year_fraction(date_jan1_2024, date_jul1_2024) == pytest.approx(
            actual_days / 365,
        )

    def test_one_day(self, calc_act_365):
        result = calc_act_365.calc_year_fraction(date(2024, 6, 15), date(2024, 6, 16))
        assert result == pytest.approx(1 / 365)

    def test_denominator_is_365_not_360(self, calc_act_365, date_jan1_2024, date_jul1_2024):
        calc_360 = Daycount_Actual_360()
        result_365 = calc_act_365.calc_year_fraction(date_jan1_2024, date_jul1_2024)
        result_360 = calc_360.calc_year_fraction(date_jan1_2024, date_jul1_2024)
        # Same numerator (182), different denominator → act/365 < act/360
        assert result_365 < result_360


# ==============================================================================
# Tests: Daycount_Actual_Actual
# ==============================================================================


class Test_Daycount_Actual_Actual:
    """Test the ACTUAL/ACTUAL day-count calculator.

    ACTUAL/ACTUAL: daily interest = annual_rate / actual_days_in_year,
    accrued over actual days.  Splits across year boundaries.
    """

    def test_convention_attribute(self, calc_act_act):
        assert calc_act_act.convention is Daycount_Convention.ACTUAL_ACTUAL

    def test_full_year_leap(self, calc_act_act, date_jan1_2024, date_jan1_2025):
        # Exactly 1 leap year → 366 / 366 = 1.0
        assert calc_act_act.calc_year_fraction(date_jan1_2024, date_jan1_2025) == 1.0

    def test_full_year_non_leap(self, calc_act_act, date_jan1_2023, date_jan1_2024):
        # Exactly 1 non-leap year → 365 / 365 = 1.0
        assert calc_act_act.calc_year_fraction(date_jan1_2023, date_jan1_2024) == 1.0

    def test_two_full_years(self, calc_act_act, date_jan1_2023, date_jan1_2025):
        # 2023 (365/365) + 2024 (366/366) = 2.0
        assert calc_act_act.calc_year_fraction(date_jan1_2023, date_jan1_2025) == 2.0

    def test_half_year_leap(self, calc_act_act, date_jan1_2024, date_jul1_2024):
        # 182 days in a 366-day year → 182/366
        result = calc_act_act.calc_year_fraction(date_jan1_2024, date_jul1_2024)
        assert result == pytest.approx(182 / 366)

    def test_half_year_non_leap(self, calc_act_act):
        # 2023 is not leap: Jan-Jun = 31+28+31+30+31+30 = 181 days / 365
        result = calc_act_act.calc_year_fraction(date(2023, 1, 1), date(2023, 7, 1))
        assert result == pytest.approx(181 / 365)

    def test_cross_year_boundary(self, calc_act_act):
        # Oct 1 2023 → Mar 1 2024
        # 2023 portion: Oct+Nov+Dec = 31+30+31 = 92 days / 365
        # 2024 portion: Jan+Feb = 31+29 = 60 days / 366
        d_start = date(2023, 10, 1)
        d_end = date(2024, 3, 1)
        result = calc_act_act.calc_year_fraction(d_start, d_end)
        expected = 92 / 365 + 60 / 366
        assert result == pytest.approx(expected)

    def test_same_date_returns_zero(self, calc_act_act, date_jan1_2024):
        assert calc_act_act.calc_year_fraction(date_jan1_2024, date_jan1_2024) == 0.0

    def test_one_day_leap_year(self, calc_act_act):
        result = calc_act_act.calc_year_fraction(date(2024, 2, 28), date(2024, 2, 29))
        assert result == pytest.approx(1 / 366)

    def test_one_day_non_leap_year(self, calc_act_act):
        result = calc_act_act.calc_year_fraction(date(2023, 3, 15), date(2023, 3, 16))
        assert result == pytest.approx(1 / 365)

    def test_three_year_span(self, calc_act_act):
        # 2022 (non-leap) + 2023 (non-leap) + 2024 (leap) = 3.0
        result = calc_act_act.calc_year_fraction(date(2022, 1, 1), date(2025, 1, 1))
        assert result == pytest.approx(3.0)


# ==============================================================================
# Tests: get_daycount_calculator factory
# ==============================================================================


class Test_Get_Daycount_Calculator:
    """Test the factory function."""

    @pytest.mark.parametrize(
        ("convention", "expected_cls"),
        [
            (Daycount_Convention.THIRTY_360, Daycount_Thirty_360),
            (Daycount_Convention.THIRTY_365, Daycount_Thirty_365),
            (Daycount_Convention.ACTUAL_360, Daycount_Actual_360),
            (Daycount_Convention.ACTUAL_365, Daycount_Actual_365),
            (Daycount_Convention.ACTUAL_ACTUAL, Daycount_Actual_Actual),
        ],
    )
    def test_returns_correct_type(self, convention, expected_cls):
        calc = get_daycount_calculator(convention)
        assert isinstance(calc, expected_cls)

    @pytest.mark.parametrize(
        "convention",
        list(Daycount_Convention),
    )
    def test_convention_matches(self, convention):
        calc = get_daycount_calculator(convention)
        assert calc.convention is convention

    def test_invalid_convention_raises(self):
        with pytest.raises(Exception_Validation_Input, match="Daycount_Convention"):
            get_daycount_calculator("30/360")  # type: ignore[arg-type]

    def test_none_raises(self):
        with pytest.raises(Exception_Validation_Input):
            get_daycount_calculator(None)  # type: ignore[arg-type]

    def test_factory_produces_working_calculator(self, date_jan1_2024, date_jul1_2024):
        calc = get_daycount_calculator(Daycount_Convention.THIRTY_360)
        result = calc.calc_year_fraction(date_jan1_2024, date_jul1_2024)
        assert result == 0.5


# ==============================================================================
# Tests: Cross-convention comparisons
# ==============================================================================


class Test_Cross_Convention_Comparisons:
    """Compare results across different conventions for the same date pair."""

    def test_full_non_leap_year_ordering(self):
        """For a full non-leap year (365 days), verify relative ordering."""
        d_start = date(2023, 1, 1)
        d_end = date(2024, 1, 1)

        results = {}
        for conv in Daycount_Convention:
            calc = get_daycount_calculator(conv)
            results[conv] = calc.calc_year_fraction(d_start, d_end)

        # 30/360: 360/360 = 1.0
        assert results[Daycount_Convention.THIRTY_360] == pytest.approx(1.0)
        # 30/365: 360/365 < 1.0
        assert results[Daycount_Convention.THIRTY_365] < 1.0
        # ACT/360: 365/360 > 1.0
        assert results[Daycount_Convention.ACTUAL_360] > 1.0
        # ACT/365: 365/365 = 1.0
        assert results[Daycount_Convention.ACTUAL_365] == pytest.approx(1.0)
        # ACT/ACT: 365/365 = 1.0
        assert results[Daycount_Convention.ACTUAL_ACTUAL] == pytest.approx(1.0)

    def test_full_leap_year_ordering(self):
        """For a full leap year (366 days), verify relative ordering."""
        d_start = date(2024, 1, 1)
        d_end = date(2025, 1, 1)

        results = {}
        for conv in Daycount_Convention:
            calc = get_daycount_calculator(conv)
            results[conv] = calc.calc_year_fraction(d_start, d_end)

        # 30/360: 360/360 = 1.0
        assert results[Daycount_Convention.THIRTY_360] == pytest.approx(1.0)
        # ACT/360: 366/360 > 1.0
        assert results[Daycount_Convention.ACTUAL_360] > 1.0
        # ACT/365: 366/365 > 1.0
        assert results[Daycount_Convention.ACTUAL_365] > 1.0
        # ACT/ACT: 366/366 = 1.0
        assert results[Daycount_Convention.ACTUAL_ACTUAL] == pytest.approx(1.0)
