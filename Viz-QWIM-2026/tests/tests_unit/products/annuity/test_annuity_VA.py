"""Unit tests for Annuity_VA class.

Tests cover initialization, property access, input validation,
charge calculations, account value simulation, surrender charge/value,
benefit base rollup, GMDB death benefit, GLWB payout, annuity payout,
and string representation.

Author
------
QWIM Team

Version
-------
0.7.0 (2026-03-01)
"""

from __future__ import annotations

import polars as pl
import pytest

from src.products.annuity.annuity_base import Annuity_Type, Withdrawal_Rates
from src.products.annuity.annuity_VA import Annuity_VA
from src.utils.custom_exceptions_errors_loggers.exception_custom import (
    Exception_Validation_Input,
)


# ======================================================================
# Fixtures
# ======================================================================


@pytest.fixture()
def va_defaults() -> Annuity_VA:
    """Create an Annuity_VA with default parameters."""
    return Annuity_VA(
        client_age=55,
        annuity_payout_rate=0.05,
        age_income_start=65,
        age_max_ratchet=85,
        rate_rollup_benefit=0.05,
    )


@pytest.fixture()
def va_no_riders() -> Annuity_VA:
    """Create a VA without GMDB or GLWB."""
    return Annuity_VA(
        client_age=55,
        annuity_payout_rate=0.05,
        age_income_start=65,
        age_max_ratchet=85,
        rate_rollup_benefit=0.05,
        has_GMDB=False,
        has_GLWB=False,
    )


@pytest.fixture()
def va_custom() -> Annuity_VA:
    """Create a VA with custom charges and riders."""
    return Annuity_VA(
        client_age=50,
        annuity_payout_rate=0.04,
        age_income_start=65,
        age_max_ratchet=80,
        rate_rollup_benefit=0.06,
        rate_ME_charge=0.015,
        rate_admin_fee=0.002,
        rate_rider_charge=0.012,
        rate_surrender_charge_initial=0.08,
        surrender_charge_schedule_years=8,
        pct_free_withdrawal=0.10,
        has_GMDB=True,
        has_GLWB=True,
        pct_GLWB=0.06,
        payment_frequency=4,
    )


# ======================================================================
# Initialization tests
# ======================================================================


@pytest.mark.unit()
class Test_Annuity_VA_Init:
    """Test Annuity_VA initialization and parameter storage."""

    def test_init_default_parameters(self, va_defaults: Annuity_VA) -> None:
        """Verify default parameter values."""
        assert va_defaults.client_age == 55
        assert va_defaults.annuity_payout_rate == pytest.approx(0.05)
        assert va_defaults.annuity_type == Annuity_Type.ANNUITY_VA
        assert va_defaults.income_start_age == 65
        assert va_defaults.max_ratchet_age == 85
        assert va_defaults.rollup_rate == pytest.approx(0.05)
        assert va_defaults.deferral_years == 10
        assert va_defaults.rate_ME_charge == pytest.approx(0.0125)
        assert va_defaults.rate_admin_fee == pytest.approx(0.0015)
        assert va_defaults.rate_rider_charge == pytest.approx(0.0100)
        assert va_defaults.has_GMDB is True
        assert va_defaults.has_GLWB is True
        assert va_defaults.pct_GLWB == pytest.approx(0.05)
        assert va_defaults.pct_free_withdrawal == pytest.approx(0.10)
        assert va_defaults.payment_frequency == 12
        assert va_defaults.market_proxy == "Equity"

    def test_init_custom_parameters(self, va_custom: Annuity_VA) -> None:
        """Verify custom parameter values."""
        assert va_custom.client_age == 50
        assert va_custom.income_start_age == 65
        assert va_custom.deferral_years == 15
        assert va_custom.rollup_rate == pytest.approx(0.06)
        assert va_custom.rate_ME_charge == pytest.approx(0.015)
        assert va_custom.rate_admin_fee == pytest.approx(0.002)
        assert va_custom.rate_rider_charge == pytest.approx(0.012)
        assert va_custom.pct_GLWB == pytest.approx(0.06)
        assert va_custom.payment_frequency == 4

    def test_init_no_riders(self, va_no_riders: Annuity_VA) -> None:
        """Verify riders can be disabled."""
        assert va_no_riders.has_GMDB is False
        assert va_no_riders.has_GLWB is False


# ======================================================================
# Input validation tests
# ======================================================================


@pytest.mark.unit()
class Test_Annuity_VA_Validation:
    """Test Annuity_VA input validation."""

    def test_invalid_client_age(self) -> None:
        """Non-positive client age should raise."""
        with pytest.raises(Exception_Validation_Input):
            Annuity_VA(
                client_age=0,
                annuity_payout_rate=0.05,
                age_income_start=65,
                age_max_ratchet=85,
                rate_rollup_benefit=0.05,
            )

    def test_income_start_age_not_greater(self) -> None:
        """Income start age must be > client age."""
        with pytest.raises(Exception_Validation_Input):
            Annuity_VA(
                client_age=65,
                annuity_payout_rate=0.05,
                age_income_start=65,
                age_max_ratchet=85,
                rate_rollup_benefit=0.05,
            )

    def test_max_ratchet_age_less_than_client(self) -> None:
        """Max ratchet age < client age should raise."""
        with pytest.raises(Exception_Validation_Input):
            Annuity_VA(
                client_age=55,
                annuity_payout_rate=0.05,
                age_income_start=65,
                age_max_ratchet=50,
                rate_rollup_benefit=0.05,
            )

    def test_negative_rollup_rate(self) -> None:
        """Negative rollup rate should raise."""
        with pytest.raises(Exception_Validation_Input):
            Annuity_VA(
                client_age=55,
                annuity_payout_rate=0.05,
                age_income_start=65,
                age_max_ratchet=85,
                rate_rollup_benefit=-0.01,
            )

    def test_negative_ME_charge(self) -> None:
        """Negative M&E charge should raise."""
        with pytest.raises(Exception_Validation_Input):
            Annuity_VA(
                client_age=55,
                annuity_payout_rate=0.05,
                age_income_start=65,
                age_max_ratchet=85,
                rate_rollup_benefit=0.05,
                rate_ME_charge=-0.01,
            )

    def test_negative_admin_fee(self) -> None:
        """Negative admin fee should raise."""
        with pytest.raises(Exception_Validation_Input):
            Annuity_VA(
                client_age=55,
                annuity_payout_rate=0.05,
                age_income_start=65,
                age_max_ratchet=85,
                rate_rollup_benefit=0.05,
                rate_admin_fee=-0.01,
            )

    def test_negative_rider_charge(self) -> None:
        """Negative rider charge should raise."""
        with pytest.raises(Exception_Validation_Input):
            Annuity_VA(
                client_age=55,
                annuity_payout_rate=0.05,
                age_income_start=65,
                age_max_ratchet=85,
                rate_rollup_benefit=0.05,
                rate_rider_charge=-0.01,
            )

    def test_negative_surrender_charge(self) -> None:
        """Negative surrender charge should raise."""
        with pytest.raises(Exception_Validation_Input):
            Annuity_VA(
                client_age=55,
                annuity_payout_rate=0.05,
                age_income_start=65,
                age_max_ratchet=85,
                rate_rollup_benefit=0.05,
                rate_surrender_charge_initial=-0.01,
            )

    def test_pct_GLWB_out_of_range(self) -> None:
        """GLWB percentage > 1 should raise."""
        with pytest.raises(Exception_Validation_Input):
            Annuity_VA(
                client_age=55,
                annuity_payout_rate=0.05,
                age_income_start=65,
                age_max_ratchet=85,
                rate_rollup_benefit=0.05,
                pct_GLWB=1.5,
            )

    def test_invalid_payment_frequency(self) -> None:
        """Zero payment frequency should raise."""
        with pytest.raises(Exception_Validation_Input):
            Annuity_VA(
                client_age=55,
                annuity_payout_rate=0.05,
                age_income_start=65,
                age_max_ratchet=85,
                rate_rollup_benefit=0.05,
                payment_frequency=0,
            )


# ======================================================================
# Charge calculation tests
# ======================================================================


@pytest.mark.unit()
class Test_Annuity_VA_Charge_Calculations:
    """Test total annual charges and account value after charges."""

    def test_calc_total_annual_charges(self, va_defaults: Annuity_VA) -> None:
        """Total charges = ME + admin + rider."""
        total = va_defaults.calc_total_annual_charges()
        expected = 0.0125 + 0.0015 + 0.0100
        assert total == pytest.approx(expected)

    def test_calc_total_annual_charges_custom(
        self,
        va_custom: Annuity_VA,
    ) -> None:
        """Custom charges sum correctly."""
        total = va_custom.calc_total_annual_charges()
        expected = 0.015 + 0.002 + 0.012
        assert total == pytest.approx(expected)

    def test_calc_account_value_after_charges_single_period(
        self,
        va_defaults: Annuity_VA,
    ) -> None:
        """Single period: V = P * (1 + R_gross - charges)."""
        total_charge = va_defaults.calc_total_annual_charges()
        values = va_defaults.calc_account_value_after_charges(100_000.0, [0.10])
        assert len(values) == 2
        assert values[0] == pytest.approx(100_000.0)
        expected = 100_000.0 * (1 + 0.10 - total_charge)
        assert values[1] == pytest.approx(expected)

    def test_calc_account_value_after_charges_multiple_periods(
        self,
        va_defaults: Annuity_VA,
    ) -> None:
        """Multi-period simulation tracks correctly."""
        gross_returns = [0.10, -0.05, 0.08]
        values = va_defaults.calc_account_value_after_charges(
            100_000.0,
            gross_returns,
        )
        assert len(values) == 4
        assert values[0] == pytest.approx(100_000.0)

        total_charge = va_defaults.calc_total_annual_charges()
        v1 = 100_000.0 * (1 + 0.10 - total_charge)
        v2 = v1 * (1 + (-0.05) - total_charge)
        v3 = v2 * (1 + 0.08 - total_charge)
        assert values[1] == pytest.approx(v1)
        assert values[2] == pytest.approx(max(v2, 0.0))
        assert values[3] == pytest.approx(max(v3, 0.0))

    def test_calc_account_value_after_charges_floor_at_zero(
        self,
        va_defaults: Annuity_VA,
    ) -> None:
        """Account value cannot go below 0."""
        # Extreme negative returns
        values = va_defaults.calc_account_value_after_charges(
            100.0,
            [-2.0],
        )
        assert values[1] == pytest.approx(0.0)

    def test_calc_account_value_invalid_principal(
        self,
        va_defaults: Annuity_VA,
    ) -> None:
        """Non-positive principal should raise."""
        with pytest.raises(Exception_Validation_Input):
            va_defaults.calc_account_value_after_charges(0.0, [0.05])

    def test_calc_account_value_invalid_returns_type(
        self,
        va_defaults: Annuity_VA,
    ) -> None:
        """Non-list returns should raise."""
        with pytest.raises(Exception_Validation_Input):
            va_defaults.calc_account_value_after_charges(100_000.0, 0.05)  # type: ignore[arg-type]


# ======================================================================
# Surrender charge / value tests
# ======================================================================


@pytest.mark.unit()
class Test_Annuity_VA_Surrender:
    """Test surrender charge rate and surrender value."""

    def test_surrender_charge_year_zero(self, va_defaults: Annuity_VA) -> None:
        """Year 0 has the full initial surrender charge."""
        rate = va_defaults.calc_surrender_charge_rate(0)
        assert rate == pytest.approx(0.07)

    def test_surrender_charge_declining(self, va_defaults: Annuity_VA) -> None:
        """Charge declines linearly."""
        rate = va_defaults.calc_surrender_charge_rate(3)
        expected = 0.07 * (1 - 3 / 7)
        assert rate == pytest.approx(expected, rel=1e-9)

    def test_surrender_charge_after_schedule(
        self,
        va_defaults: Annuity_VA,
    ) -> None:
        """After schedule, charge is 0."""
        assert va_defaults.calc_surrender_charge_rate(7) == pytest.approx(0.0)
        assert va_defaults.calc_surrender_charge_rate(20) == pytest.approx(0.0)

    def test_surrender_charge_negative_year(
        self,
        va_defaults: Annuity_VA,
    ) -> None:
        """Negative year returns 0."""
        assert va_defaults.calc_surrender_charge_rate(-1) == pytest.approx(0.0)

    def test_surrender_value_year_zero(self, va_defaults: Annuity_VA) -> None:
        """Surrender value with full charge."""
        sv = va_defaults.calc_surrender_value(100_000.0, 0)
        # free=10000, excess=90000, charge=90000*0.07=6300
        assert sv == pytest.approx(93_700.0)

    def test_surrender_value_after_schedule(
        self,
        va_defaults: Annuity_VA,
    ) -> None:
        """After schedule, full value returned."""
        sv = va_defaults.calc_surrender_value(100_000.0, 10)
        assert sv == pytest.approx(100_000.0)

    def test_surrender_value_invalid_account_value(
        self,
        va_defaults: Annuity_VA,
    ) -> None:
        """Non-positive account value should raise."""
        with pytest.raises(Exception_Validation_Input):
            va_defaults.calc_surrender_value(0.0, 3)


# ======================================================================
# Benefit base and rider tests
# ======================================================================


@pytest.mark.unit()
class Test_Annuity_VA_Benefit_Base:
    """Test benefit base rollup, GMDB, and GLWB calculations."""

    def test_calc_benefit_base_default_deferral(
        self,
        va_defaults: Annuity_VA,
    ) -> None:
        """B = principal * (1 + rollup)^10."""
        bb = va_defaults.calc_benefit_base_with_rollup(100_000.0)
        expected = 100_000.0 * (1.05**10)
        assert bb == pytest.approx(expected, rel=1e-9)

    def test_calc_benefit_base_custom_years(
        self,
        va_defaults: Annuity_VA,
    ) -> None:
        """Override deferral to 5 years."""
        bb = va_defaults.calc_benefit_base_with_rollup(100_000.0, years_deferred=5)
        expected = 100_000.0 * (1.05**5)
        assert bb == pytest.approx(expected, rel=1e-9)

    def test_calc_benefit_base_zero_years(
        self,
        va_defaults: Annuity_VA,
    ) -> None:
        """Zero deferral returns principal."""
        bb = va_defaults.calc_benefit_base_with_rollup(100_000.0, years_deferred=0)
        assert bb == pytest.approx(100_000.0)

    def test_calc_benefit_base_invalid_principal(
        self,
        va_defaults: Annuity_VA,
    ) -> None:
        """Non-positive principal should raise."""
        with pytest.raises(Exception_Validation_Input):
            va_defaults.calc_benefit_base_with_rollup(0.0)

    def test_calc_death_benefit_with_gmdb(
        self,
        va_defaults: Annuity_VA,
    ) -> None:
        """GMDB: max(account_value, premium)."""
        # Account lost value → GMDB protects premium
        db = va_defaults.calc_death_benefit(100_000.0, 80_000.0)
        assert db == pytest.approx(100_000.0)

        # Account gained value → use account value
        db = va_defaults.calc_death_benefit(100_000.0, 120_000.0)
        assert db == pytest.approx(120_000.0)

    def test_calc_death_benefit_without_gmdb(
        self,
        va_no_riders: Annuity_VA,
    ) -> None:
        """Without GMDB, death benefit = account value."""
        db = va_no_riders.calc_death_benefit(100_000.0, 80_000.0)
        assert db == pytest.approx(80_000.0)

    def test_calc_death_benefit_invalid_principal(
        self,
        va_defaults: Annuity_VA,
    ) -> None:
        """Non-positive principal should raise."""
        with pytest.raises(Exception_Validation_Input):
            va_defaults.calc_death_benefit(0.0, 80_000.0)

    def test_calc_death_benefit_negative_account(
        self,
        va_defaults: Annuity_VA,
    ) -> None:
        """Negative account value should raise."""
        with pytest.raises(Exception_Validation_Input):
            va_defaults.calc_death_benefit(100_000.0, -1.0)

    def test_calc_GLWB_payout(self, va_defaults: Annuity_VA) -> None:
        """GLWB payout = benefit_base * pct_GLWB."""
        payout = va_defaults.calc_GLWB_payout(100_000.0)
        benefit_base = 100_000.0 * (1.05**10)
        expected = benefit_base * 0.05
        assert payout == pytest.approx(expected, rel=1e-9)

    def test_calc_GLWB_payout_custom_deferral(
        self,
        va_defaults: Annuity_VA,
    ) -> None:
        """Override deferral years."""
        payout = va_defaults.calc_GLWB_payout(100_000.0, years_deferred=5)
        benefit_base = 100_000.0 * (1.05**5)
        expected = benefit_base * 0.05
        assert payout == pytest.approx(expected, rel=1e-9)

    def test_calc_GLWB_payout_without_glwb(
        self,
        va_no_riders: Annuity_VA,
    ) -> None:
        """Without GLWB, payout is 0."""
        payout = va_no_riders.calc_GLWB_payout(100_000.0)
        assert payout == pytest.approx(0.0)

    def test_calc_GLWB_payout_invalid_principal(
        self,
        va_defaults: Annuity_VA,
    ) -> None:
        """Non-positive principal should raise."""
        with pytest.raises(Exception_Validation_Input):
            va_defaults.calc_GLWB_payout(0.0)


# ======================================================================
# Payout calculation tests
# ======================================================================


@pytest.mark.unit()
class Test_Annuity_VA_Payout_Calculations:
    """Test annuity payout and monthly payout."""

    def test_calc_annuity_payout_during_deferral(
        self,
        va_defaults: Annuity_VA,
    ) -> None:
        """During deferral, payout is 0."""
        payout = va_defaults.calc_annuity_payout(100_000.0)
        assert payout == pytest.approx(0.0)

    def test_calc_annuity_payout_invalid_principal(
        self,
        va_defaults: Annuity_VA,
    ) -> None:
        """Non-positive principal should raise."""
        with pytest.raises(Exception_Validation_Input):
            va_defaults.calc_annuity_payout(0.0)

    def test_calc_monthly_payout_during_deferral(
        self,
        va_defaults: Annuity_VA,
    ) -> None:
        """Monthly payout during deferral is 0."""
        monthly = va_defaults.calc_monthly_payout(100_000.0)
        assert monthly == pytest.approx(0.0)


# ======================================================================
# String representation tests
# ======================================================================


@pytest.mark.unit()
class Test_Annuity_VA_String_Representation:
    """Test get_annuity_as_string output."""

    def test_string_contains_va(self, va_defaults: Annuity_VA) -> None:
        """String should contain 'VA'."""
        result = va_defaults.get_annuity_as_string()
        assert "VA" in result

    def test_string_contains_riders(self, va_defaults: Annuity_VA) -> None:
        """String should mention GMDB and GLWB when active."""
        result = va_defaults.get_annuity_as_string()
        assert "GMDB" in result
        assert "GLWB" in result

    def test_string_no_riders(self, va_no_riders: Annuity_VA) -> None:
        """Without riders, string should say None."""
        result = va_no_riders.get_annuity_as_string()
        assert "None" in result

    def test_string_contains_rollup(self, va_defaults: Annuity_VA) -> None:
        """String should mention rollup rate."""
        result = va_defaults.get_annuity_as_string()
        assert "Rollup" in result

    def test_string_contains_me_charge(self, va_defaults: Annuity_VA) -> None:
        """String should mention M&E charge."""
        result = va_defaults.get_annuity_as_string()
        assert "M&E" in result


# ======================================================================
# Inflation DataFrame type compatibility tests
# ======================================================================


@pytest.fixture()
def inflation_df() -> pl.DataFrame:
    """Minimal valid inflation DataFrame."""
    return pl.DataFrame(
        {
            "Start Date": ["2024-01-01"],
            "End Date": ["2025-01-01"],
            "Inflation Factor": [1.025],
        },
    )


@pytest.mark.unit()
class Test_Annuity_VA_Inflation_Parameter:
    """Verify calc_annuity_payout and calc_monthly_payout accept pl.DataFrame.

    For Variable Annuities the ``obj_inflation`` parameter is unused; it
    exists for interface compatibility.  Tests confirm that passing ``None``
    or a valid Polars DataFrame does not raise and payout is ``0.0`` during
    deferral.
    """

    def test_calc_annuity_payout_accepts_none(
        self,
        va_defaults: Annuity_VA,
    ) -> None:
        """None is accepted as obj_inflation; returns 0.0 during deferral."""
        payout = va_defaults.calc_annuity_payout(100_000.0, obj_inflation=None)
        assert payout == pytest.approx(0.0)

    def test_calc_annuity_payout_accepts_polars_dataframe(
        self,
        va_defaults: Annuity_VA,
        inflation_df: pl.DataFrame,
    ) -> None:
        """pl.DataFrame accepted as obj_inflation; VA ignores it (returns 0.0)."""
        payout = va_defaults.calc_annuity_payout(
            100_000.0,
            obj_inflation=inflation_df,
        )
        assert payout == pytest.approx(0.0)  # still in deferral

    def test_calc_monthly_payout_accepts_polars_dataframe(
        self,
        va_defaults: Annuity_VA,
        inflation_df: pl.DataFrame,
    ) -> None:
        """calc_monthly_payout accepts pl.DataFrame as obj_inflation."""
        monthly = va_defaults.calc_monthly_payout(
            100_000.0,
            obj_inflation=inflation_df,
        )
        assert monthly == pytest.approx(0.0)  # still in deferral


# ======================================================================
# Scenarios DataFrame parameter tests
# ======================================================================


@pytest.fixture()
def va_past_income_age() -> Annuity_VA:
    """Create a VA and advance client past income start age."""
    va = Annuity_VA(
        client_age=64,
        annuity_payout_rate=0.05,
        age_income_start=65,
        age_max_ratchet=64,  # ratchet already past after m_client_age advances to 66
        rate_rollup_benefit=0.05,
    )
    va.m_client_age = 66
    return va


@pytest.fixture()
def scenarios_df_va() -> pl.DataFrame:
    """Minimal valid scenarios DataFrame with Date and Equity columns."""
    return pl.DataFrame(
        {
            "Date": ["2024-01-01", "2025-01-01"],
            "Equity": [0.08, 0.10],
        },
    )


@pytest.mark.unit()
class Test_Annuity_VA_Scenarios_Parameter:
    """Verify calc_annuity_payout handles pl.DataFrame obj_scenarios correctly."""

    def test_market_proxy_default(self, va_defaults: Annuity_VA) -> None:
        """Default market_proxy is 'Equity'."""
        assert va_defaults.market_proxy == "Equity"

    def test_market_proxy_custom(self) -> None:
        """Custom market_proxy is stored and returned."""
        va = Annuity_VA(
            client_age=55,
            annuity_payout_rate=0.05,
            age_income_start=65,
            age_max_ratchet=85,
            rate_rollup_benefit=0.05,
            market_proxy="US Large Cap",
        )
        assert va.market_proxy == "US Large Cap"

    def test_calc_annuity_payout_during_deferral_with_scenarios(
        self,
        va_defaults: Annuity_VA,
        scenarios_df_va: pl.DataFrame,
    ) -> None:
        """Scenarios DataFrame is accepted during deferral; payout remains 0.0."""
        payout = va_defaults.calc_annuity_payout(
            100_000.0,
            obj_scenarios=scenarios_df_va,
        )
        assert payout == pytest.approx(0.0)

    def test_calc_annuity_payout_past_income_age_with_scenarios(
        self,
        va_past_income_age: Annuity_VA,
        scenarios_df_va: pl.DataFrame,
    ) -> None:
        """Past income age with valid scenarios DataFrame returns positive payout."""
        payout = va_past_income_age.calc_annuity_payout(
            100_000.0,
            obj_scenarios=scenarios_df_va,
        )
        assert payout > 0.0

    def test_calc_annuity_payout_missing_date_column(
        self,
        va_past_income_age: Annuity_VA,
    ) -> None:
        """Missing 'Date' column raises Exception_Validation_Input."""
        bad_df = pl.DataFrame({"Equity": [0.08]})
        with pytest.raises(Exception_Validation_Input):
            va_past_income_age.calc_annuity_payout(100_000.0, obj_scenarios=bad_df)

    def test_calc_annuity_payout_missing_proxy_column(
        self,
        va_past_income_age: Annuity_VA,
    ) -> None:
        """Missing market_proxy column raises Exception_Validation_Input."""
        bad_df = pl.DataFrame({"Date": ["2024-01-01"], "NASDAQ": [0.08]})
        with pytest.raises(Exception_Validation_Input):
            va_past_income_age.calc_annuity_payout(100_000.0, obj_scenarios=bad_df)

    def test_calc_annuity_payout_custom_proxy_column(
        self,
        va_past_income_age: Annuity_VA,
    ) -> None:
        """Custom market_proxy requires matching column in scenarios df."""
        va_past_income_age.m_market_proxy = "US Large Cap"
        correct_df = pl.DataFrame(
            {"Date": ["2024-01-01"], "US Large Cap": [0.10]},
        )
        payout = va_past_income_age.calc_annuity_payout(
            100_000.0,
            obj_scenarios=correct_df,
        )
        assert payout > 0.0
        # Wrong column name raises
        wrong_df = pl.DataFrame({"Date": ["2024-01-01"], "Equity": [0.10]})
        with pytest.raises(Exception_Validation_Input):
            va_past_income_age.calc_annuity_payout(100_000.0, obj_scenarios=wrong_df)

    def test_invalid_market_proxy_empty_string(self) -> None:
        """Empty string market_proxy raises Exception_Validation_Input."""
        with pytest.raises(Exception_Validation_Input):
            Annuity_VA(
                client_age=55,
                annuity_payout_rate=0.05,
                age_income_start=65,
                age_max_ratchet=85,
                rate_rollup_benefit=0.05,
                market_proxy="  ",
            )


# ======================================================================
# Withdrawal-rate fixtures
# ======================================================================


@pytest.fixture()
def inflation_wr_df() -> pl.DataFrame:
    """Single-row inflation DataFrame with all four required columns.

    Inflation Factor = 1.03, Inverse Inflation Factor = 1 / 1.03.
    """
    return pl.DataFrame(
        {
            "Start Date": ["2024-01-01"],
            "End Date": ["2025-01-01"],
            "Inflation Factor": [1.03],
            "Inverse Inflation Factor": [1.0 / 1.03],
        },
    )


# ======================================================================
# Withdrawal-rate tests
# ======================================================================


@pytest.mark.unit()
class Test_Annuity_VA_Withdrawal_Rates:
    """Tests for calc_withdrawal_rates on Annuity_VA.

    ``va_defaults`` is in deferral (client_age=55, income_start=65), so
    nominal_WR = 0.0 during deferral.  ``va_past_income_age`` (m_client_age=66)
    produces a positive nominal_WR based on the rolled-up benefit base.
    """

    def test_returns_withdrawal_rates_struct(
        self,
        va_defaults: Annuity_VA,
    ) -> None:
        """calc_withdrawal_rates returns a Withdrawal_Rates instance."""
        result = va_defaults.calc_withdrawal_rates(100_000.0, desired_WR=0.04)
        assert isinstance(result, Withdrawal_Rates)

    def test_nominal_wr_zero_during_deferral(
        self,
        va_defaults: Annuity_VA,
    ) -> None:
        """nominal_WR is 0.0 while client is in the deferral period."""
        result = va_defaults.calc_withdrawal_rates(100_000.0, desired_WR=0.04)
        assert result.nominal_WR == pytest.approx(0.0)

    def test_real_wr_zero_during_deferral(
        self,
        va_defaults: Annuity_VA,
    ) -> None:
        """real_WR is 0.0 while client is in the deferral period."""
        result = va_defaults.calc_withdrawal_rates(100_000.0, desired_WR=0.04)
        assert result.real_WR == pytest.approx(0.0)

    def test_nominal_wr_positive_past_income_age(
        self,
        va_past_income_age: Annuity_VA,
        scenarios_df_va: pl.DataFrame,
    ) -> None:
        """nominal_WR is positive once client has passed the income start age."""
        result = va_past_income_age.calc_withdrawal_rates(
            100_000.0,
            desired_WR=0.04,
            obj_scenarios=scenarios_df_va,
        )
        assert result.nominal_WR > 0.0

    def test_real_wr_equals_nominal_when_no_inflation(
        self,
        va_past_income_age: Annuity_VA,
        scenarios_df_va: pl.DataFrame,
    ) -> None:
        """real_WR equals nominal_WR when no inflation data is provided."""
        result = va_past_income_age.calc_withdrawal_rates(
            100_000.0,
            desired_WR=0.04,
            obj_scenarios=scenarios_df_va,
        )
        assert result.real_WR == pytest.approx(result.nominal_WR, rel=1e-9)

    def test_real_wr_deflated_with_inflation(
        self,
        va_past_income_age: Annuity_VA,
        scenarios_df_va: pl.DataFrame,
        inflation_wr_df: pl.DataFrame,
    ) -> None:
        """real_WR is deflated by the inverse inflation factor."""
        result_no_infl = va_past_income_age.calc_withdrawal_rates(
            100_000.0,
            desired_WR=0.04,
            obj_scenarios=scenarios_df_va,
        )
        result_with_infl = va_past_income_age.calc_withdrawal_rates(
            100_000.0,
            desired_WR=0.04,
            obj_scenarios=scenarios_df_va,
            obj_inflation=inflation_wr_df,
        )
        expected_real = result_no_infl.nominal_WR * (1.0 / 1.03)
        assert result_with_infl.real_WR == pytest.approx(expected_real, rel=1e-9)

    def test_real_wr_less_than_nominal_when_inflation_positive(
        self,
        va_past_income_age: Annuity_VA,
        scenarios_df_va: pl.DataFrame,
        inflation_wr_df: pl.DataFrame,
    ) -> None:
        """real_WR < nominal_WR when the inflation factor is above 1.0."""
        result = va_past_income_age.calc_withdrawal_rates(
            100_000.0,
            desired_WR=0.04,
            obj_scenarios=scenarios_df_va,
            obj_inflation=inflation_wr_df,
        )
        assert result.real_WR < result.nominal_WR

    def test_invalid_principal_zero_raises(
        self,
        va_defaults: Annuity_VA,
    ) -> None:
        """Zero amount_principal raises Exception_Validation_Input."""
        with pytest.raises(Exception_Validation_Input):
            va_defaults.calc_withdrawal_rates(0.0, desired_WR=0.04)

    def test_invalid_principal_negative_raises(
        self,
        va_defaults: Annuity_VA,
    ) -> None:
        """Negative amount_principal raises Exception_Validation_Input."""
        with pytest.raises(Exception_Validation_Input):
            va_defaults.calc_withdrawal_rates(-1.0, desired_WR=0.04)

    def test_invalid_desired_wr_zero_raises(
        self,
        va_defaults: Annuity_VA,
    ) -> None:
        """Zero desired_WR raises Exception_Validation_Input."""
        with pytest.raises(Exception_Validation_Input):
            va_defaults.calc_withdrawal_rates(100_000.0, desired_WR=0.0)

    def test_invalid_desired_wr_negative_raises(
        self,
        va_defaults: Annuity_VA,
    ) -> None:
        """Negative desired_WR raises Exception_Validation_Input."""
        with pytest.raises(Exception_Validation_Input):
            va_defaults.calc_withdrawal_rates(100_000.0, desired_WR=-0.04)

    def test_inflation_df_missing_inverse_column_raises(
        self,
        va_defaults: Annuity_VA,
    ) -> None:
        """obj_inflation lacking 'Inverse Inflation Factor' raises Exception_Validation_Input."""
        incomplete_df = pl.DataFrame(
            {
                "Start Date": ["2024-01-01"],
                "End Date": ["2025-01-01"],
                "Inflation Factor": [1.03],
            },
        )
        with pytest.raises(Exception_Validation_Input):
            va_defaults.calc_withdrawal_rates(
                100_000.0,
                desired_WR=0.04,
                obj_inflation=incomplete_df,
            )


# ======================================================================
# Income starting age tests
# ======================================================================


@pytest.mark.unit()
class Test_Annuity_VA_Income_Starting_Age:
    """Test m_annuity_income_starting_age for Annuity_VA."""

    def test_default_is_client_age(self, va_defaults: Annuity_VA) -> None:
        """Default income starting age equals client_age."""
        assert va_defaults.annuity_income_starting_age == va_defaults.client_age

    def test_custom_income_starting_age_stored(self) -> None:
        """Explicitly provided income starting age >= client_age is stored correctly."""
        va = Annuity_VA(
            client_age=55,
            annuity_payout_rate=0.05,
            age_income_start=65,
            age_max_ratchet=85,
            rate_rollup_benefit=0.05,
            annuity_income_starting_age=60,
        )
        assert va.annuity_income_starting_age == 60

    def test_income_starting_age_equal_to_client_age(self) -> None:
        """Income starting age equal to client_age is accepted."""
        va = Annuity_VA(
            client_age=55,
            annuity_payout_rate=0.05,
            age_income_start=65,
            age_max_ratchet=85,
            rate_rollup_benefit=0.05,
            annuity_income_starting_age=55,
        )
        assert va.annuity_income_starting_age == 55

    def test_income_starting_age_below_client_age_raises(self) -> None:
        """Income starting age below client_age raises Exception_Validation_Input."""
        with pytest.raises(Exception_Validation_Input):
            Annuity_VA(
                client_age=55,
                annuity_payout_rate=0.05,
                age_income_start=65,
                age_max_ratchet=85,
                rate_rollup_benefit=0.05,
                annuity_income_starting_age=50,
            )

    def test_member_variable_directly(self) -> None:
        """m_annuity_income_starting_age member variable equals the provided value."""
        va = Annuity_VA(
            client_age=55,
            annuity_payout_rate=0.05,
            age_income_start=65,
            age_max_ratchet=85,
            rate_rollup_benefit=0.05,
            annuity_income_starting_age=65,
        )
        assert va.m_annuity_income_starting_age == 65
