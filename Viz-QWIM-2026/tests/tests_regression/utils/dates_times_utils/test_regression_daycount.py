"""Regression tests for daycount module.

These tests lock in exact year-fraction values computed by each day-count
convention for specific date pairs.  If a refactor accidentally changes the
formula the tests will fail — requiring deliberate baseline regeneration.

Baseline values computed from the current implementation and verified against
standard financial references (e.g., ISDA 2006 Definitions).

Author: QWIM Development Team
Version: 0.1.0
"""

from __future__ import annotations

from datetime import date

import pytest

from src.utils.dates_times_utils.daycount import (
    Daycount_Convention,
    get_daycount_calculator,
)


# ==============================================================================
# Baseline constants
# ==============================================================================

# Date fixtures used across regression scenarios
DATE_2023_01_01 = date(2023, 1, 1)
DATE_2023_07_01 = date(2023, 7, 1)
DATE_2024_01_01 = date(2024, 1, 1)
DATE_2024_02_28 = date(2024, 2, 28)
DATE_2024_02_29 = date(2024, 2, 29)
DATE_2024_03_01 = date(2024, 3, 1)
DATE_2024_07_01 = date(2024, 7, 1)
DATE_2025_01_01 = date(2025, 1, 1)


# ==============================================================================
# Regression: 30/360
# ==============================================================================


class Test_Regression_Thirty_360:
    """Regression baselines for the 30/360 convention."""

    _calc = get_daycount_calculator(Daycount_Convention.THIRTY_360)

    @pytest.mark.regression()
    def test_baseline_full_non_leap_year(self):
        """30/360: 2023-01-01 → 2024-01-01 = 360/360 = 1.0000000000."""
        result = self._calc.calc_year_fraction(DATE_2023_01_01, DATE_2024_01_01)
        assert result == pytest.approx(1.0, rel=1e-12)

    @pytest.mark.regression()
    def test_baseline_full_leap_year(self):
        """30/360: 2024-01-01 → 2025-01-01 = 360/360 = 1.0000000000."""
        result = self._calc.calc_year_fraction(DATE_2024_01_01, DATE_2025_01_01)
        assert result == pytest.approx(1.0, rel=1e-12)

    @pytest.mark.regression()
    def test_baseline_half_year_h1_2024(self):
        """30/360: 2024-01-01 → 2024-07-01 = 180/360 = 0.5000000000."""
        result = self._calc.calc_year_fraction(DATE_2024_01_01, DATE_2024_07_01)
        assert result == pytest.approx(0.5, rel=1e-12)

    @pytest.mark.regression()
    def test_baseline_half_year_h1_2023(self):
        """30/360: 2023-01-01 → 2023-07-01 = 180/360 = 0.5000000000."""
        result = self._calc.calc_year_fraction(DATE_2023_01_01, DATE_2023_07_01)
        assert result == pytest.approx(180 / 360, rel=1e-12)

    @pytest.mark.regression()
    def test_baseline_two_full_years(self):
        """30/360: 2023-01-01 → 2025-01-01 = 720/360 = 2.0000000000."""
        result = self._calc.calc_year_fraction(DATE_2023_01_01, DATE_2025_01_01)
        assert result == pytest.approx(2.0, rel=1e-12)

    @pytest.mark.regression()
    def test_baseline_one_month(self):
        """30/360: 2024-01-01 → 2024-02-01 = 30/360 = 0.083333..."""
        result = self._calc.calc_year_fraction(DATE_2024_01_01, date(2024, 2, 1))
        assert result == pytest.approx(30 / 360, rel=1e-12)

    @pytest.mark.regression()
    def test_baseline_quarter(self):
        """30/360: 2024-01-01 → 2024-04-01 = 90/360 = 0.25."""
        result = self._calc.calc_year_fraction(DATE_2024_01_01, date(2024, 4, 1))
        assert result == pytest.approx(0.25, rel=1e-12)

    @pytest.mark.regression()
    def test_baseline_end_of_month_capping(self):
        """30/360: 2024-01-31 → 2024-03-31 = 60/360 (both days capped to 30)."""
        result = self._calc.calc_year_fraction(date(2024, 1, 31), date(2024, 3, 31))
        assert result == pytest.approx(60 / 360, rel=1e-12)


# ==============================================================================
# Regression: 30/365
# ==============================================================================


class Test_Regression_Thirty_365:
    """Regression baselines for the 30/365 convention."""

    _calc = get_daycount_calculator(Daycount_Convention.THIRTY_365)

    @pytest.mark.regression()
    def test_baseline_full_non_leap_year(self):
        """30/365: 2023-01-01 → 2024-01-01 = 360/365 = 0.9863013699."""
        result = self._calc.calc_year_fraction(DATE_2023_01_01, DATE_2024_01_01)
        assert result == pytest.approx(360 / 365, rel=1e-12)

    @pytest.mark.regression()
    def test_baseline_full_leap_year(self):
        """30/365: 2024-01-01 → 2025-01-01 = 360/365 = 0.9863013699."""
        result = self._calc.calc_year_fraction(DATE_2024_01_01, DATE_2025_01_01)
        assert result == pytest.approx(360 / 365, rel=1e-12)

    @pytest.mark.regression()
    def test_baseline_half_year(self):
        """30/365: 2024-01-01 → 2024-07-01 = 180/365 = 0.4931506849."""
        result = self._calc.calc_year_fraction(DATE_2024_01_01, DATE_2024_07_01)
        assert result == pytest.approx(180 / 365, rel=1e-12)

    @pytest.mark.regression()
    def test_baseline_one_month(self):
        """30/365: 2024-01-01 → 2024-02-01 = 30/365 = 0.0821917808."""
        result = self._calc.calc_year_fraction(DATE_2024_01_01, date(2024, 2, 1))
        assert result == pytest.approx(30 / 365, rel=1e-12)

    @pytest.mark.regression()
    def test_baseline_numerator_same_as_30_360(self):
        """30/365 numerator equals 30/360 numerator; only the denominator differs."""
        calc_360 = get_daycount_calculator(Daycount_Convention.THIRTY_360)
        r_365 = self._calc.calc_year_fraction(DATE_2024_01_01, DATE_2024_07_01)
        r_360 = calc_360.calc_year_fraction(DATE_2024_01_01, DATE_2024_07_01)
        # r_365 = 180/365, r_360 = 180/360 → ratio = 360/365
        assert r_365 / r_360 == pytest.approx(360 / 365, rel=1e-12)


# ==============================================================================
# Regression: ACTUAL/360
# ==============================================================================


class Test_Regression_Actual_360:
    """Regression baselines for the ACTUAL/360 convention."""

    _calc = get_daycount_calculator(Daycount_Convention.ACTUAL_360)

    @pytest.mark.regression()
    def test_baseline_full_non_leap_year(self):
        """ACT/360: 2023-01-01 → 2024-01-01 = 365/360 = 1.0138888..."""
        result = self._calc.calc_year_fraction(DATE_2023_01_01, DATE_2024_01_01)
        assert result == pytest.approx(365 / 360, rel=1e-12)

    @pytest.mark.regression()
    def test_baseline_full_leap_year(self):
        """ACT/360: 2024-01-01 → 2025-01-01 = 366/360 = 1.0166666..."""
        result = self._calc.calc_year_fraction(DATE_2024_01_01, DATE_2025_01_01)
        assert result == pytest.approx(366 / 360, rel=1e-12)

    @pytest.mark.regression()
    def test_baseline_h1_2024_leap(self):
        """ACT/360: 2024-01-01 → 2024-07-01 = 182/360 = 0.5055555..."""
        result = self._calc.calc_year_fraction(DATE_2024_01_01, DATE_2024_07_01)
        assert result == pytest.approx(182 / 360, rel=1e-12)

    @pytest.mark.regression()
    def test_baseline_one_day(self):
        """ACT/360: any single day = 1/360 = 0.002777..."""
        result = self._calc.calc_year_fraction(date(2024, 3, 15), date(2024, 3, 16))
        assert result == pytest.approx(1 / 360, rel=1e-12)

    @pytest.mark.regression()
    def test_baseline_90_days_2025(self):
        """ACT/360: 2025-01-01 → 2025-04-01 = 90/360 = 0.25."""
        actual_days = (date(2025, 4, 1) - date(2025, 1, 1)).days
        result = self._calc.calc_year_fraction(date(2025, 1, 1), date(2025, 4, 1))
        assert actual_days == 90
        assert result == pytest.approx(90 / 360, rel=1e-12)


# ==============================================================================
# Regression: ACTUAL/365
# ==============================================================================


class Test_Regression_Actual_365:
    """Regression baselines for the ACTUAL/365 convention."""

    _calc = get_daycount_calculator(Daycount_Convention.ACTUAL_365)

    @pytest.mark.regression()
    def test_baseline_full_non_leap_year(self):
        """ACT/365: 2023-01-01 → 2024-01-01 = 365/365 = 1.0 exactly."""
        result = self._calc.calc_year_fraction(DATE_2023_01_01, DATE_2024_01_01)
        assert result == pytest.approx(1.0, rel=1e-12)

    @pytest.mark.regression()
    def test_baseline_full_leap_year(self):
        """ACT/365: 2024-01-01 → 2025-01-01 = 366/365 = 1.0027397..."""
        result = self._calc.calc_year_fraction(DATE_2024_01_01, DATE_2025_01_01)
        assert result == pytest.approx(366 / 365, rel=1e-12)

    @pytest.mark.regression()
    def test_baseline_h1_2024_leap(self):
        """ACT/365: 2024-01-01 → 2024-07-01 = 182/365 = 0.4986301..."""
        result = self._calc.calc_year_fraction(DATE_2024_01_01, DATE_2024_07_01)
        assert result == pytest.approx(182 / 365, rel=1e-12)

    @pytest.mark.regression()
    def test_baseline_one_day(self):
        """ACT/365: any single day = 1/365 = 0.002739..."""
        result = self._calc.calc_year_fraction(date(2024, 6, 15), date(2024, 6, 16))
        assert result == pytest.approx(1 / 365, rel=1e-12)

    @pytest.mark.regression()
    def test_baseline_h1_2023_non_leap(self):
        """ACT/365: 2023-01-01 → 2023-07-01 = 181/365 = 0.4958904..."""
        actual_days = (DATE_2023_07_01 - DATE_2023_01_01).days
        result = self._calc.calc_year_fraction(DATE_2023_01_01, DATE_2023_07_01)
        assert actual_days == 181
        assert result == pytest.approx(181 / 365, rel=1e-12)


# ==============================================================================
# Regression: ACTUAL/ACTUAL
# ==============================================================================


class Test_Regression_Actual_Actual:
    """Regression baselines for the ACTUAL/ACTUAL convention."""

    _calc = get_daycount_calculator(Daycount_Convention.ACTUAL_ACTUAL)

    @pytest.mark.regression()
    def test_baseline_full_non_leap_year(self):
        """ACT/ACT: 2023-01-01 → 2024-01-01 = 365/365 = 1.0."""
        result = self._calc.calc_year_fraction(DATE_2023_01_01, DATE_2024_01_01)
        assert result == pytest.approx(1.0, rel=1e-12)

    @pytest.mark.regression()
    def test_baseline_full_leap_year(self):
        """ACT/ACT: 2024-01-01 → 2025-01-01 = 366/366 = 1.0."""
        result = self._calc.calc_year_fraction(DATE_2024_01_01, DATE_2025_01_01)
        assert result == pytest.approx(1.0, rel=1e-12)

    @pytest.mark.regression()
    def test_baseline_two_full_years(self):
        """ACT/ACT: 2023-01-01 → 2025-01-01 = 2.0."""
        result = self._calc.calc_year_fraction(DATE_2023_01_01, DATE_2025_01_01)
        assert result == pytest.approx(2.0, rel=1e-12)

    @pytest.mark.regression()
    def test_baseline_h1_2024_leap(self):
        """ACT/ACT: 2024-01-01 → 2024-07-01 = 182/366 = 0.4972677595..."""
        result = self._calc.calc_year_fraction(DATE_2024_01_01, DATE_2024_07_01)
        assert result == pytest.approx(182 / 366, rel=1e-12)

    @pytest.mark.regression()
    def test_baseline_h1_2023_non_leap(self):
        """ACT/ACT: 2023-01-01 → 2023-07-01 = 181/365 = 0.4958904..."""
        result = self._calc.calc_year_fraction(DATE_2023_01_01, DATE_2023_07_01)
        assert result == pytest.approx(181 / 365, rel=1e-12)

    @pytest.mark.regression()
    def test_baseline_cross_year_oct_2023_to_mar_2024(self):
        """ACT/ACT: 2023-10-01 → 2024-03-01 = 92/365 + 60/366 = 0.4158577..."""
        d_start = date(2023, 10, 1)
        d_end = date(2024, 3, 1)
        expected = 92 / 365 + 60 / 366
        result = self._calc.calc_year_fraction(d_start, d_end)
        assert result == pytest.approx(expected, rel=1e-12)

    @pytest.mark.regression()
    def test_baseline_one_day_leap_year(self):
        """ACT/ACT: 2024-02-28 → 2024-02-29 = 1/366 = 0.002732240..."""
        result = self._calc.calc_year_fraction(DATE_2024_02_28, DATE_2024_02_29)
        assert result == pytest.approx(1 / 366, rel=1e-12)

    @pytest.mark.regression()
    def test_baseline_one_day_non_leap_year(self):
        """ACT/ACT: 2023-03-15 → 2023-03-16 = 1/365 = 0.002739726..."""
        result = self._calc.calc_year_fraction(date(2023, 3, 15), date(2023, 3, 16))
        assert result == pytest.approx(1 / 365, rel=1e-12)

    @pytest.mark.regression()
    def test_baseline_three_year_span_2022_to_2025(self):
        """ACT/ACT: 2022-01-01 → 2025-01-01 = 3.0 (2022+2023 non-leap, 2024 leap)."""
        result = self._calc.calc_year_fraction(date(2022, 1, 1), date(2025, 1, 1))
        assert result == pytest.approx(3.0, rel=1e-12)


# ==============================================================================
# Regression: Cross-convention ordering proofs (locked baseline comparisons)
# ==============================================================================


class Test_Regression_Cross_Convention_Ordering:
    """Locked cross-convention inequality relationships for key date pairs."""

    @pytest.mark.regression()
    def test_non_leap_year_ordering_locked(self):
        """For a full non-leap year the ordering: 30/365 < 30/360 = ACT/365 = ACT/ACT < ACT/360."""
        results = {
            conv: get_daycount_calculator(conv).calc_year_fraction(
                DATE_2023_01_01, DATE_2024_01_01
            )
            for conv in Daycount_Convention
        }
        # 30/365 = 360/365 < 1
        assert results[Daycount_Convention.THIRTY_365] == pytest.approx(360 / 365, rel=1e-9)
        # 30/360 = 1.0
        assert results[Daycount_Convention.THIRTY_360] == pytest.approx(1.0, rel=1e-12)
        # ACT/365 = 365/365 = 1.0
        assert results[Daycount_Convention.ACTUAL_365] == pytest.approx(1.0, rel=1e-12)
        # ACT/ACT = 365/365 = 1.0
        assert results[Daycount_Convention.ACTUAL_ACTUAL] == pytest.approx(1.0, rel=1e-12)
        # ACT/360 = 365/360 > 1
        assert results[Daycount_Convention.ACTUAL_360] == pytest.approx(365 / 360, rel=1e-9)
        # Ordering check
        assert results[Daycount_Convention.THIRTY_365] < results[Daycount_Convention.THIRTY_360]
        assert results[Daycount_Convention.THIRTY_360] < results[Daycount_Convention.ACTUAL_360]

    @pytest.mark.regression()
    def test_leap_year_ordering_locked(self):
        """For a full leap year the ordering: 30/365 < 30/360 = ACT/ACT < ACT/365 < ACT/360."""
        results = {
            conv: get_daycount_calculator(conv).calc_year_fraction(
                DATE_2024_01_01, DATE_2025_01_01
            )
            for conv in Daycount_Convention
        }
        assert results[Daycount_Convention.THIRTY_365] == pytest.approx(360 / 365, rel=1e-9)
        assert results[Daycount_Convention.THIRTY_360] == pytest.approx(1.0, rel=1e-12)
        assert results[Daycount_Convention.ACTUAL_ACTUAL] == pytest.approx(1.0, rel=1e-12)
        # ACT/365 = 366/365 > 1
        assert results[Daycount_Convention.ACTUAL_365] == pytest.approx(366 / 365, rel=1e-9)
        # ACT/360 = 366/360 > ACT/365
        assert results[Daycount_Convention.ACTUAL_360] == pytest.approx(366 / 360, rel=1e-9)
        assert results[Daycount_Convention.ACTUAL_360] > results[Daycount_Convention.ACTUAL_365]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
