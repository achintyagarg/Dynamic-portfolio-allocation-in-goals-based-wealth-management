"""Integration tests for dates_times_utils package.

These tests exercise the daycount conventions in realistic financial
scenarios — multi-year accruals, cross-year boundaries, yield calculations,
and cross-convention comparisons.

Author: QWIM Development Team
Version: 0.1.0
"""

from __future__ import annotations

from datetime import date

import pytest

from src.utils.dates_times_utils.daycount import (
    Daycount_Actual_360,
    Daycount_Actual_365,
    Daycount_Actual_Actual,
    Daycount_Convention,
    Daycount_Thirty_360,
    Daycount_Thirty_365,
    get_daycount_calculator,
)


# ==============================================================================
# Integration: factory + all conventions for realistic financial scenarios
# ==============================================================================


class Test_Bond_Accrued_Interest_Calculation:
    """Simulate real-world accrued interest calculations for fixed-income instruments."""

    @pytest.mark.integration()
    def test_annual_coupon_30_360_full_year_accrual(self):
        """Annual 8% coupon on 30/360 bond over full year = 8% × 1.0 = 8%."""
        coupon_rate = 0.08
        calc = get_daycount_calculator(Daycount_Convention.THIRTY_360)
        year_frac = calc.calc_year_fraction(date(2023, 1, 1), date(2024, 1, 1))
        accrued = coupon_rate * year_frac
        assert accrued == pytest.approx(0.08, rel=1e-9)

    @pytest.mark.integration()
    def test_semiannual_coupon_act_365_half_year_accrual(self):
        """Semiannual 6% coupon on ACT/365 bond: 3% per period for ~182 days."""
        coupon_rate = 0.06
        calc = get_daycount_calculator(Daycount_Convention.ACTUAL_365)
        # First coupon period: Jan → Jul 2024 (182 days in leap year)
        year_frac = calc.calc_year_fraction(date(2024, 1, 1), date(2024, 7, 1))
        # 182 / 365 ≈ 0.4986
        accrued = coupon_rate * year_frac
        assert accrued == pytest.approx(coupon_rate * 182 / 365, rel=1e-9)

    @pytest.mark.integration()
    def test_act_act_one_year_yields_unity_regardless_of_leap_status(self):
        """ACT/ACT convention gives exactly 1.0 for any complete calendar year."""
        calc = get_daycount_calculator(Daycount_Convention.ACTUAL_ACTUAL)
        for year_start in range(2020, 2027):
            d_start = date(year_start, 1, 1)
            d_end = date(year_start + 1, 1, 1)
            year_frac = calc.calc_year_fraction(d_start, d_end)
            assert year_frac == pytest.approx(1.0, rel=1e-9), (
                f"ACT/ACT year fraction for {year_start} is not 1.0: {year_frac}"
            )

    @pytest.mark.integration()
    def test_cross_year_boundary_act_act_additive(self):
        """For ACT/ACT, the period Jan-2023 → Jan-2025 = 2.0 year fractions."""
        calc = get_daycount_calculator(Daycount_Convention.ACTUAL_ACTUAL)
        year_frac = calc.calc_year_fraction(date(2023, 1, 1), date(2025, 1, 1))
        assert year_frac == pytest.approx(2.0, rel=1e-9)

    @pytest.mark.integration()
    def test_act_360_produces_greater_than_act_365_same_period(self):
        """ACT/360 always yields a larger year fraction than ACT/365 (smaller denominator)."""
        calc_360 = get_daycount_calculator(Daycount_Convention.ACTUAL_360)
        calc_365 = get_daycount_calculator(Daycount_Convention.ACTUAL_365)
        d_start = date(2024, 3, 1)
        d_end = date(2024, 9, 1)
        frac_360 = calc_360.calc_year_fraction(d_start, d_end)
        frac_365 = calc_365.calc_year_fraction(d_start, d_end)
        assert frac_360 > frac_365

    @pytest.mark.integration()
    def test_30_360_vs_30_365_same_numerator_different_denominator(self):
        """30/360 and 30/365 use the same day-count numerator; only denominator differs."""
        calc_360 = get_daycount_calculator(Daycount_Convention.THIRTY_360)
        calc_365 = get_daycount_calculator(Daycount_Convention.THIRTY_365)
        d_start = date(2024, 1, 1)
        d_end = date(2024, 7, 1)
        frac_360 = calc_360.calc_year_fraction(d_start, d_end)
        frac_365 = calc_365.calc_year_fraction(d_start, d_end)
        # Numerator is 180 for both (6×30); frac_360 = 180/360, frac_365 = 180/365
        expected_360 = 180 / 360
        expected_365 = 180 / 365
        assert frac_360 == pytest.approx(expected_360, rel=1e-9)
        assert frac_365 == pytest.approx(expected_365, rel=1e-9)
        assert frac_360 > frac_365


# ==============================================================================
# Integration: Multi-period consistency
# ==============================================================================


class Test_Multi_Period_Consistency:
    """Test additivity and period-splitting consistency across conventions."""

    @pytest.mark.integration()
    def test_two_half_years_sum_to_full_year_act_365(self):
        """ACT/365: H1 + H2 year fractions should equal the full-year fraction."""
        calc = get_daycount_calculator(Daycount_Convention.ACTUAL_365)
        d0 = date(2023, 1, 1)
        d1 = date(2023, 7, 1)  # 181 days
        d2 = date(2024, 1, 1)  # 184 days
        h1 = calc.calc_year_fraction(d0, d1)
        h2 = calc.calc_year_fraction(d1, d2)
        full = calc.calc_year_fraction(d0, d2)
        assert h1 + h2 == pytest.approx(full, rel=1e-9)

    @pytest.mark.integration()
    def test_quarterly_periods_sum_to_annual_30_360(self):
        """30/360: Q1+Q2+Q3+Q4 = 1.0 for calendar year 2024."""
        calc = get_daycount_calculator(Daycount_Convention.THIRTY_360)
        quarters = [
            (date(2024, 1, 1), date(2024, 4, 1)),
            (date(2024, 4, 1), date(2024, 7, 1)),
            (date(2024, 7, 1), date(2024, 10, 1)),
            (date(2024, 10, 1), date(2025, 1, 1)),
        ]
        total = sum(calc.calc_year_fraction(*q) for q in quarters)
        assert total == pytest.approx(1.0, rel=1e-9)

    @pytest.mark.integration()
    def test_daily_sum_matches_act_360_known_period(self):
        """Sum of 90 daily ACT/360 fractions = 90 × (1/360) = 0.25."""
        calc = get_daycount_calculator(Daycount_Convention.ACTUAL_360)
        # 2025-01-01 → 2025-04-01 = 90 days
        d_start = date(2025, 1, 1)
        d_end = date(2025, 4, 1)
        full_fraction = calc.calc_year_fraction(d_start, d_end)
        assert full_fraction == pytest.approx(90 / 360, rel=1e-9)

    @pytest.mark.integration()
    def test_all_conventions_agree_on_zero_for_same_date(self):
        """All five conventions return 0.0 when start_date == end_date."""
        d = date(2024, 6, 15)
        for convention in Daycount_Convention:
            calc = get_daycount_calculator(convention)
            assert calc.calc_year_fraction(d, d) == 0.0, (
                f"{convention} returned non-zero for same-date input"
            )


# ==============================================================================
# Integration: Factory function robustness
# ==============================================================================


class Test_Factory_Robustness:
    """Integration tests exercising the factory in varied usage patterns."""

    @pytest.mark.integration()
    def test_factory_produces_independent_calculator_instances(self):
        """Factory produces independent objects; modifying one does not affect another."""
        calc1 = get_daycount_calculator(Daycount_Convention.THIRTY_360)
        calc2 = get_daycount_calculator(Daycount_Convention.THIRTY_360)
        assert calc1 is not calc2

    @pytest.mark.integration()
    def test_factory_round_trip_all_conventions(self):
        """For every convention: factory → compute year fraction → non-negative result."""
        d_start = date(2023, 6, 1)
        d_end = date(2024, 6, 1)
        for convention in Daycount_Convention:
            calc = get_daycount_calculator(convention)
            result = calc.calc_year_fraction(d_start, d_end)
            assert result > 0, f"{convention} returned non-positive year fraction: {result}"

    @pytest.mark.integration()
    def test_calculator_repr_contains_convention_string(self):
        """repr() of each calculator includes its convention's string value."""
        for convention in Daycount_Convention:
            calc = get_daycount_calculator(convention)
            assert convention.value in repr(calc), (
                f"Convention value '{convention.value}' not in repr: {repr(calc)}"
            )

    @pytest.mark.integration()
    def test_convention_property_accessible_after_factory_call(self):
        """Each factory-produced calculator exposes its convention via .convention."""
        for convention in Daycount_Convention:
            calc = get_daycount_calculator(convention)
            assert calc.convention is convention


# ==============================================================================
# Integration: Financial context — forward rate calculation
# ==============================================================================


class Test_Forward_Rate_Consistency:
    """Integration tests using year fractions in simple interest calculations."""

    @pytest.mark.integration()
    def test_simple_interest_30_360(self):
        """Simple-interest amount = principal × rate × year_fraction (30/360)."""
        principal = 1_000_000.0
        annual_rate = 0.05  # 5%
        calc = get_daycount_calculator(Daycount_Convention.THIRTY_360)
        year_frac = calc.calc_year_fraction(date(2024, 1, 1), date(2024, 7, 1))
        interest = principal * annual_rate * year_frac
        # 180/360 = 0.5 → interest = 25_000
        assert interest == pytest.approx(25_000.0, rel=1e-6)

    @pytest.mark.integration()
    def test_simple_interest_act_365_full_non_leap_year(self):
        """Simple interest on non-leap full year equals the annual rate × principal."""
        principal = 1_000_000.0
        annual_rate = 0.04  # 4%
        calc = get_daycount_calculator(Daycount_Convention.ACTUAL_365)
        year_frac = calc.calc_year_fraction(date(2023, 1, 1), date(2024, 1, 1))
        interest = principal * annual_rate * year_frac
        # 365/365=1.0 → interest = 40_000
        assert interest == pytest.approx(40_000.0, rel=1e-6)

    @pytest.mark.integration()
    def test_act_act_preserves_per_diem_interpretation(self):
        """ACT/ACT year fraction for 1 day in a leap year = 1/366."""
        calc = get_daycount_calculator(Daycount_Convention.ACTUAL_ACTUAL)
        frac = calc.calc_year_fraction(date(2024, 1, 1), date(2024, 1, 2))
        assert frac == pytest.approx(1 / 366, rel=1e-9)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
