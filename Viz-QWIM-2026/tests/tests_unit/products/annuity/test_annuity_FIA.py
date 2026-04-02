"""Unit tests for Annuity_FIA class.

Tests cover initialization, property access, input validation,
crediting calculations, account value simulation, benefit base rollup,
surrender charge/value calculations, minimum guaranteed value, payout,
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
from src.products.annuity.annuity_FIA import Annuity_FIA
from src.utils.custom_exceptions_errors_loggers.exception_custom import (
    Exception_Validation_Input,
)


# ======================================================================
# Fixtures
# ======================================================================


@pytest.fixture()
def fia_defaults() -> Annuity_FIA:
    """Create an Annuity_FIA with default parameters."""
    return Annuity_FIA(
        client_age=55,
        annuity_payout_rate=0.05,
        age_income_start=65,
        age_max_ratchet=85,
        rate_rollup_benefit=0.05,
    )


@pytest.fixture()
def fia_custom() -> Annuity_FIA:
    """Create an Annuity_FIA with custom crediting parameters."""
    return Annuity_FIA(
        client_age=55,
        annuity_payout_rate=0.05,
        age_income_start=65,
        age_max_ratchet=85,
        rate_rollup_benefit=0.06,
        cap_rate=0.08,
        participation_rate=0.80,
        spread_rate=0.02,
        floor_rate=0.01,
        rate_surrender_charge_initial=0.10,
        surrender_charge_schedule_years=10,
        rate_minimum_guarantee=0.02,
        pct_minimum_guarantee_base=0.90,
        pct_free_withdrawal=0.15,
        payment_frequency=4,
    )


# ======================================================================
# Initialization tests
# ======================================================================


@pytest.mark.unit()
class Test_Annuity_FIA_Init:
    """Test Annuity_FIA initialization and parameter storage."""

    def test_init_default_parameters(self, fia_defaults: Annuity_FIA) -> None:
        """Verify default parameter values."""
        assert fia_defaults.client_age == 55
        assert fia_defaults.annuity_payout_rate == pytest.approx(0.05)
        assert fia_defaults.annuity_type == Annuity_Type.ANNUITY_FIA
        assert fia_defaults.income_start_age == 65
        assert fia_defaults.max_ratchet_age == 85
        assert fia_defaults.rollup_rate == pytest.approx(0.05)
        assert fia_defaults.cap_rate == pytest.approx(0.05)
        assert fia_defaults.participation_rate == pytest.approx(1.0)
        assert fia_defaults.spread_rate == pytest.approx(0.0)
        assert fia_defaults.floor_rate == pytest.approx(0.0)
        assert fia_defaults.deferral_years == 10
        assert fia_defaults.payment_frequency == 12
        assert fia_defaults.pct_free_withdrawal == pytest.approx(0.10)
        assert fia_defaults.financial_index == "S&P 500"

    def test_init_custom_parameters(self, fia_custom: Annuity_FIA) -> None:
        """Verify custom parameter values."""
        assert fia_custom.rollup_rate == pytest.approx(0.06)
        assert fia_custom.cap_rate == pytest.approx(0.08)
        assert fia_custom.participation_rate == pytest.approx(0.80)
        assert fia_custom.spread_rate == pytest.approx(0.02)
        assert fia_custom.floor_rate == pytest.approx(0.01)
        assert fia_custom.payment_frequency == 4
        assert fia_custom.pct_free_withdrawal == pytest.approx(0.15)


# ======================================================================
# Input validation tests
# ======================================================================


@pytest.mark.unit()
class Test_Annuity_FIA_Validation:
    """Test Annuity_FIA input validation."""

    def test_invalid_client_age(self) -> None:
        """Non-positive client age should raise."""
        with pytest.raises(Exception_Validation_Input):
            Annuity_FIA(
                client_age=0,
                annuity_payout_rate=0.05,
                age_income_start=65,
                age_max_ratchet=85,
                rate_rollup_benefit=0.05,
            )

    def test_income_start_age_not_greater(self) -> None:
        """Income start age must be > client age."""
        with pytest.raises(Exception_Validation_Input):
            Annuity_FIA(
                client_age=65,
                annuity_payout_rate=0.05,
                age_income_start=65,
                age_max_ratchet=85,
                rate_rollup_benefit=0.05,
            )

    def test_max_ratchet_age_less_than_client(self) -> None:
        """Max ratchet age < client age should raise."""
        with pytest.raises(Exception_Validation_Input):
            Annuity_FIA(
                client_age=55,
                annuity_payout_rate=0.05,
                age_income_start=65,
                age_max_ratchet=50,
                rate_rollup_benefit=0.05,
            )

    def test_negative_rollup_rate(self) -> None:
        """Negative rollup rate should raise."""
        with pytest.raises(Exception_Validation_Input):
            Annuity_FIA(
                client_age=55,
                annuity_payout_rate=0.05,
                age_income_start=65,
                age_max_ratchet=85,
                rate_rollup_benefit=-0.01,
            )

    def test_negative_cap_rate(self) -> None:
        """Negative cap rate should raise."""
        with pytest.raises(Exception_Validation_Input):
            Annuity_FIA(
                client_age=55,
                annuity_payout_rate=0.05,
                age_income_start=65,
                age_max_ratchet=85,
                rate_rollup_benefit=0.05,
                cap_rate=-0.01,
            )

    def test_participation_rate_out_of_range(self) -> None:
        """Participation rate > 1 should raise."""
        with pytest.raises(Exception_Validation_Input):
            Annuity_FIA(
                client_age=55,
                annuity_payout_rate=0.05,
                age_income_start=65,
                age_max_ratchet=85,
                rate_rollup_benefit=0.05,
                participation_rate=1.5,
            )

    def test_negative_spread_rate(self) -> None:
        """Negative spread rate should raise."""
        with pytest.raises(Exception_Validation_Input):
            Annuity_FIA(
                client_age=55,
                annuity_payout_rate=0.05,
                age_income_start=65,
                age_max_ratchet=85,
                rate_rollup_benefit=0.05,
                spread_rate=-0.01,
            )

    def test_negative_surrender_charge(self) -> None:
        """Negative surrender charge should raise."""
        with pytest.raises(Exception_Validation_Input):
            Annuity_FIA(
                client_age=55,
                annuity_payout_rate=0.05,
                age_income_start=65,
                age_max_ratchet=85,
                rate_rollup_benefit=0.05,
                rate_surrender_charge_initial=-0.01,
            )

    def test_invalid_payment_frequency(self) -> None:
        """Zero payment frequency should raise."""
        with pytest.raises(Exception_Validation_Input):
            Annuity_FIA(
                client_age=55,
                annuity_payout_rate=0.05,
                age_income_start=65,
                age_max_ratchet=85,
                rate_rollup_benefit=0.05,
                payment_frequency=0,
            )


# ======================================================================
# Crediting calculation tests
# ======================================================================


@pytest.mark.unit()
class Test_Annuity_FIA_Crediting_Calculations:
    """Test calc_credited_rate with various index return scenarios."""

    def test_credited_rate_positive_index(
        self,
        fia_defaults: Annuity_FIA,
    ) -> None:
        """Positive index return: min(max(R,0)*p, cap), capped at 5%."""
        # index=0.10, spread=0, participation=1.0, cap=0.05 → credited=0.05
        credited = fia_defaults.calc_credited_rate(0.10)
        assert credited == pytest.approx(0.05)

    def test_credited_rate_below_cap(self, fia_defaults: Annuity_FIA) -> None:
        """Index return below cap passes through."""
        # index=0.03, spread=0, participation=1.0, cap=0.05 → credited=0.03
        credited = fia_defaults.calc_credited_rate(0.03)
        assert credited == pytest.approx(0.03)

    def test_credited_rate_negative_index(
        self,
        fia_defaults: Annuity_FIA,
    ) -> None:
        """Negative index return floors at 0 (default floor=0)."""
        credited = fia_defaults.calc_credited_rate(-0.20)
        assert credited == pytest.approx(0.0)

    def test_credited_rate_with_spread_and_participation(
        self,
        fia_custom: Annuity_FIA,
    ) -> None:
        """Custom crediting: floor(0.01), spread(0.02), participation(0.80), cap(0.08)."""
        # index=0.10 → max(0.10,0)=0.10, net=0.10-0.02=0.08
        # participated=max(0.08,0)*0.80=0.064, cap=min(0.064,0.08)=0.064
        # final=max(0.064,0.01)=0.064
        credited = fia_custom.calc_credited_rate(0.10)
        assert credited == pytest.approx(0.064)

    def test_credited_rate_with_floor_binding(
        self,
        fia_custom: Annuity_FIA,
    ) -> None:
        """When crediting formula yields less than floor, floor binds."""
        # index=-0.10 → max(-0.10,0)=0, net=0-0.02=-0.02
        # participated=max(-0.02,0)*0.80=0, cap=min(0,0.08)=0
        # final=max(0,0.01)=0.01
        credited = fia_custom.calc_credited_rate(-0.10)
        assert credited == pytest.approx(0.01)

    def test_credited_rate_zero_index(self, fia_defaults: Annuity_FIA) -> None:
        """Zero index return yields floor rate (0 for defaults)."""
        credited = fia_defaults.calc_credited_rate(0.0)
        assert credited == pytest.approx(0.0)

    def test_credited_rate_invalid_input(
        self,
        fia_defaults: Annuity_FIA,
    ) -> None:
        """Non-numeric input should raise."""
        with pytest.raises(Exception_Validation_Input):
            fia_defaults.calc_credited_rate("ten")  # type: ignore[arg-type]


# ======================================================================
# Account value simulation tests
# ======================================================================


@pytest.mark.unit()
class Test_Annuity_FIA_Account_Value:
    """Test calc_account_value over multiple periods."""

    def test_account_value_single_period(
        self,
        fia_defaults: Annuity_FIA,
    ) -> None:
        """One-period simulation."""
        values = fia_defaults.calc_account_value(100_000.0, [0.08])
        assert len(values) == 2
        assert values[0] == pytest.approx(100_000.0)
        # credited=min(0.08,0.05)=0.05 → 100000*1.05=105000
        assert values[1] == pytest.approx(105_000.0)

    def test_account_value_multiple_periods(
        self,
        fia_defaults: Annuity_FIA,
    ) -> None:
        """Multi-period simulation."""
        returns = [0.10, -0.05, 0.03]
        values = fia_defaults.calc_account_value(100_000.0, returns)
        assert len(values) == 4
        assert values[0] == pytest.approx(100_000.0)
        # yr1: credited=0.05 → 105000
        # yr2: credited=0 → 105000
        # yr3: credited=0.03 → 105000*1.03=108150
        assert values[1] == pytest.approx(105_000.0)
        assert values[2] == pytest.approx(105_000.0)
        assert values[3] == pytest.approx(108_150.0)

    def test_account_value_empty_returns(
        self,
        fia_defaults: Annuity_FIA,
    ) -> None:
        """Empty returns list yields only the initial value."""
        values = fia_defaults.calc_account_value(100_000.0, [])
        assert len(values) == 1
        assert values[0] == pytest.approx(100_000.0)

    def test_account_value_invalid_principal(
        self,
        fia_defaults: Annuity_FIA,
    ) -> None:
        """Non-positive principal should raise."""
        with pytest.raises(Exception_Validation_Input):
            fia_defaults.calc_account_value(0.0, [0.05])

    def test_account_value_invalid_returns_type(
        self,
        fia_defaults: Annuity_FIA,
    ) -> None:
        """Non-list returns should raise."""
        with pytest.raises(Exception_Validation_Input):
            fia_defaults.calc_account_value(100_000.0, 0.05)  # type: ignore[arg-type]


# ======================================================================
# Benefit base and guarantee tests
# ======================================================================


@pytest.mark.unit()
class Test_Annuity_FIA_Benefit_Base:
    """Test benefit base rollup and minimum guarantee calculations."""

    def test_calc_benefit_base_default_deferral(
        self,
        fia_defaults: Annuity_FIA,
    ) -> None:
        """Default deferral_years: B = principal * (1 + rollup)^10."""
        bb = fia_defaults.calc_benefit_base_with_rollup(100_000.0)
        expected = 100_000.0 * (1.05**10)
        assert bb == pytest.approx(expected, rel=1e-9)

    def test_calc_benefit_base_custom_years(
        self,
        fia_defaults: Annuity_FIA,
    ) -> None:
        """Override deferral to 5 years."""
        bb = fia_defaults.calc_benefit_base_with_rollup(100_000.0, years_deferred=5)
        expected = 100_000.0 * (1.05**5)
        assert bb == pytest.approx(expected, rel=1e-9)

    def test_calc_benefit_base_zero_years(
        self,
        fia_defaults: Annuity_FIA,
    ) -> None:
        """Zero deferral returns principal unchanged."""
        bb = fia_defaults.calc_benefit_base_with_rollup(100_000.0, years_deferred=0)
        assert bb == pytest.approx(100_000.0)

    def test_calc_benefit_base_invalid_principal(
        self,
        fia_defaults: Annuity_FIA,
    ) -> None:
        """Non-positive principal should raise."""
        with pytest.raises(Exception_Validation_Input):
            fia_defaults.calc_benefit_base_with_rollup(0.0)

    def test_calc_benefit_base_invalid_years(
        self,
        fia_defaults: Annuity_FIA,
    ) -> None:
        """Negative years should raise."""
        with pytest.raises(Exception_Validation_Input):
            fia_defaults.calc_benefit_base_with_rollup(100_000.0, years_deferred=-1)

    def test_calc_minimum_guaranteed_value(
        self,
        fia_defaults: Annuity_FIA,
    ) -> None:
        """V_min = principal * 0.875 * (1.01)^years."""
        mgv = fia_defaults.calc_minimum_guaranteed_value(100_000.0, 10)
        expected = 100_000.0 * 0.875 * (1.01**10)
        assert mgv == pytest.approx(expected, rel=1e-9)

    def test_calc_minimum_guaranteed_value_year_zero(
        self,
        fia_defaults: Annuity_FIA,
    ) -> None:
        """At year 0, minimum guarantee = principal * pct_base."""
        mgv = fia_defaults.calc_minimum_guaranteed_value(100_000.0, 0)
        assert mgv == pytest.approx(87_500.0)

    def test_calc_minimum_guaranteed_value_invalid_inputs(
        self,
        fia_defaults: Annuity_FIA,
    ) -> None:
        """Invalid inputs should raise."""
        with pytest.raises(Exception_Validation_Input):
            fia_defaults.calc_minimum_guaranteed_value(0.0, 10)
        with pytest.raises(Exception_Validation_Input):
            fia_defaults.calc_minimum_guaranteed_value(100_000.0, -1)


# ======================================================================
# Surrender charge / value tests
# ======================================================================


@pytest.mark.unit()
class Test_Annuity_FIA_Surrender:
    """Test surrender charge rate and surrender value."""

    def test_surrender_charge_year_zero(self, fia_defaults: Annuity_FIA) -> None:
        """Year 0 has the full initial surrender charge."""
        rate = fia_defaults.calc_surrender_charge_rate(0)
        assert rate == pytest.approx(0.08)

    def test_surrender_charge_declining(self, fia_defaults: Annuity_FIA) -> None:
        """Charge declines linearly: SC0 * (1 - year/T)."""
        # schedule=7, initial=0.08, year=3 → 0.08 * (1 - 3/7) ≈ 0.04571
        rate = fia_defaults.calc_surrender_charge_rate(3)
        expected = 0.08 * (1 - 3 / 7)
        assert rate == pytest.approx(expected, rel=1e-9)

    def test_surrender_charge_after_schedule(
        self,
        fia_defaults: Annuity_FIA,
    ) -> None:
        """After schedule ends, charge is 0."""
        rate = fia_defaults.calc_surrender_charge_rate(7)
        assert rate == pytest.approx(0.0)
        rate = fia_defaults.calc_surrender_charge_rate(10)
        assert rate == pytest.approx(0.0)

    def test_surrender_charge_negative_year(
        self,
        fia_defaults: Annuity_FIA,
    ) -> None:
        """Negative year returns 0."""
        rate = fia_defaults.calc_surrender_charge_rate(-1)
        assert rate == pytest.approx(0.0)

    def test_surrender_value_year_zero(self, fia_defaults: Annuity_FIA) -> None:
        """Surrender value at year 0 with full charge."""
        sv = fia_defaults.calc_surrender_value(100_000.0, 0)
        # free=10000, excess=90000, charge=90000*0.08=7200
        # surrender=100000-7200=92800
        assert sv == pytest.approx(92_800.0)

    def test_surrender_value_after_schedule(
        self,
        fia_defaults: Annuity_FIA,
    ) -> None:
        """After schedule, surrender value = account value."""
        sv = fia_defaults.calc_surrender_value(100_000.0, 10)
        assert sv == pytest.approx(100_000.0)

    def test_surrender_value_invalid_account_value(
        self,
        fia_defaults: Annuity_FIA,
    ) -> None:
        """Non-positive account value should raise."""
        with pytest.raises(Exception_Validation_Input):
            fia_defaults.calc_surrender_value(0.0, 3)


# ======================================================================
# Payout calculation tests
# ======================================================================


@pytest.mark.unit()
class Test_Annuity_FIA_Payout_Calculations:
    """Test annuity payout and monthly payout methods."""

    def test_calc_annuity_payout_during_deferral(
        self,
        fia_defaults: Annuity_FIA,
    ) -> None:
        """During deferral, payout is 0."""
        payout = fia_defaults.calc_annuity_payout(100_000.0)
        assert payout == pytest.approx(0.0)

    def test_calc_annuity_payout_invalid_principal(
        self,
        fia_defaults: Annuity_FIA,
    ) -> None:
        """Non-positive principal should raise."""
        with pytest.raises(Exception_Validation_Input):
            fia_defaults.calc_annuity_payout(0.0)

    def test_calc_monthly_payout_during_deferral(
        self,
        fia_defaults: Annuity_FIA,
    ) -> None:
        """Monthly payout during deferral is 0."""
        monthly = fia_defaults.calc_monthly_payout(100_000.0)
        assert monthly == pytest.approx(0.0)


# ======================================================================
# String representation tests
# ======================================================================


@pytest.mark.unit()
class Test_Annuity_FIA_String_Representation:
    """Test get_annuity_as_string output."""

    def test_string_contains_fia(self, fia_defaults: Annuity_FIA) -> None:
        """String should contain 'FIA'."""
        result = fia_defaults.get_annuity_as_string()
        assert "FIA" in result

    def test_string_contains_cap_and_participation(
        self,
        fia_defaults: Annuity_FIA,
    ) -> None:
        """String should mention cap and participation."""
        result = fia_defaults.get_annuity_as_string()
        assert "Cap" in result
        assert "Participation" in result

    def test_string_contains_rollup(self, fia_defaults: Annuity_FIA) -> None:
        """String should mention rollup rate."""
        result = fia_defaults.get_annuity_as_string()
        assert "Rollup" in result


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
            "Inflation Factor": [1.03],
        },
    )


@pytest.mark.unit()
class Test_Annuity_FIA_Inflation_Parameter:
    """Verify calc_annuity_payout and calc_monthly_payout accept pl.DataFrame.

    For FIA the ``obj_inflation`` parameter is unused; it exists purely for
    interface compatibility with :class:`~annuity_base.Annuity_Base`.  Tests
    confirm that passing ``None`` or a valid Polars DataFrame does not raise
    and that payout behaviour is unchanged (returns ``0.0`` during deferral).
    """

    def test_calc_annuity_payout_accepts_none(
        self,
        fia_defaults: Annuity_FIA,
    ) -> None:
        """None is accepted as obj_inflation; returns 0.0 during deferral."""
        payout = fia_defaults.calc_annuity_payout(100_000.0, obj_inflation=None)
        assert payout == pytest.approx(0.0)

    def test_calc_annuity_payout_accepts_polars_dataframe(
        self,
        fia_defaults: Annuity_FIA,
        inflation_df: pl.DataFrame,
    ) -> None:
        """pl.DataFrame accepted as obj_inflation; FIA ignores it (returns 0.0)."""
        payout = fia_defaults.calc_annuity_payout(
            100_000.0,
            obj_inflation=inflation_df,
        )
        assert payout == pytest.approx(0.0)  # still in deferral

    def test_calc_monthly_payout_accepts_polars_dataframe(
        self,
        fia_defaults: Annuity_FIA,
        inflation_df: pl.DataFrame,
    ) -> None:
        """calc_monthly_payout accepts pl.DataFrame as obj_inflation."""
        monthly = fia_defaults.calc_monthly_payout(
            100_000.0,
            obj_inflation=inflation_df,
        )
        assert monthly == pytest.approx(0.0)  # still in deferral


# ======================================================================
# Scenarios DataFrame parameter tests
# ======================================================================


@pytest.fixture()
def fia_past_income_age() -> Annuity_FIA:
    """Create an FIA and advance client past income start age."""
    fia = Annuity_FIA(
        client_age=64,
        annuity_payout_rate=0.05,
        age_income_start=65,
        age_max_ratchet=64,  # ratchet already past after m_client_age advances to 66
        rate_rollup_benefit=0.05,
    )
    fia.m_client_age = 66
    return fia


@pytest.fixture()
def scenarios_df_fia() -> pl.DataFrame:
    """Minimal valid scenarios DataFrame with Date and S&P 500 columns."""
    return pl.DataFrame(
        {
            "Date": ["2024-01-01", "2025-01-01"],
            "S&P 500": [0.08, 0.10],
        },
    )


@pytest.mark.unit()
class Test_Annuity_FIA_Scenarios_Parameter:
    """Verify calc_annuity_payout handles pl.DataFrame obj_scenarios correctly."""

    def test_financial_index_default(self, fia_defaults: Annuity_FIA) -> None:
        """Default financial_index is 'S&P 500'."""
        assert fia_defaults.financial_index == "S&P 500"

    def test_financial_index_custom(self) -> None:
        """Custom financial_index is stored and returned."""
        fia = Annuity_FIA(
            client_age=55,
            annuity_payout_rate=0.05,
            age_income_start=65,
            age_max_ratchet=85,
            rate_rollup_benefit=0.05,
            financial_index="Russell 2000",
        )
        assert fia.financial_index == "Russell 2000"

    def test_calc_annuity_payout_during_deferral_with_scenarios(
        self,
        fia_defaults: Annuity_FIA,
        scenarios_df_fia: pl.DataFrame,
    ) -> None:
        """Scenarios DataFrame is accepted during deferral; payout remains 0.0."""
        payout = fia_defaults.calc_annuity_payout(
            100_000.0,
            obj_scenarios=scenarios_df_fia,
        )
        assert payout == pytest.approx(0.0)

    def test_calc_annuity_payout_past_income_age_with_scenarios(
        self,
        fia_past_income_age: Annuity_FIA,
        scenarios_df_fia: pl.DataFrame,
    ) -> None:
        """Past income age with valid scenarios DataFrame returns positive payout."""
        payout = fia_past_income_age.calc_annuity_payout(
            100_000.0,
            obj_scenarios=scenarios_df_fia,
        )
        assert payout > 0.0

    def test_calc_annuity_payout_missing_date_column(
        self,
        fia_past_income_age: Annuity_FIA,
    ) -> None:
        """Missing 'Date' column raises Exception_Validation_Input."""
        bad_df = pl.DataFrame({"S&P 500": [0.08]})
        with pytest.raises(Exception_Validation_Input):
            fia_past_income_age.calc_annuity_payout(100_000.0, obj_scenarios=bad_df)

    def test_calc_annuity_payout_missing_index_column(
        self,
        fia_past_income_age: Annuity_FIA,
    ) -> None:
        """Missing financial_index column raises Exception_Validation_Input."""
        bad_df = pl.DataFrame({"Date": ["2024-01-01"], "NASDAQ": [0.08]})
        with pytest.raises(Exception_Validation_Input):
            fia_past_income_age.calc_annuity_payout(100_000.0, obj_scenarios=bad_df)

    def test_calc_annuity_payout_custom_index_column(
        self,
        fia_past_income_age: Annuity_FIA,
    ) -> None:
        """Custom financial_index requires matching column in scenarios df."""
        fia_past_income_age.m_financial_index = "Russell 2000"
        correct_df = pl.DataFrame(
            {"Date": ["2024-01-01"], "Russell 2000": [0.10]},
        )
        payout = fia_past_income_age.calc_annuity_payout(
            100_000.0,
            obj_scenarios=correct_df,
        )
        assert payout > 0.0
        # Wrong column name now raises
        wrong_df = pl.DataFrame({"Date": ["2024-01-01"], "S&P 500": [0.10]})
        with pytest.raises(Exception_Validation_Input):
            fia_past_income_age.calc_annuity_payout(100_000.0, obj_scenarios=wrong_df)

    def test_invalid_financial_index_empty_string(self) -> None:
        """Empty string financial_index raises Exception_Validation_Input."""
        with pytest.raises(Exception_Validation_Input):
            Annuity_FIA(
                client_age=55,
                annuity_payout_rate=0.05,
                age_income_start=65,
                age_max_ratchet=85,
                rate_rollup_benefit=0.05,
                financial_index="  ",
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
class Test_Annuity_FIA_Withdrawal_Rates:
    """Tests for calc_withdrawal_rates on Annuity_FIA.

    ``fia_defaults`` is in deferral (client_age=55, income_start=65), so
    nominal_WR = 0.0 during deferral.  ``fia_past_income_age`` (m_client_age=66)
    produces a positive nominal_WR based on the rolled-up benefit base.
    """

    def test_returns_withdrawal_rates_struct(
        self,
        fia_defaults: Annuity_FIA,
    ) -> None:
        """calc_withdrawal_rates returns a Withdrawal_Rates instance."""
        result = fia_defaults.calc_withdrawal_rates(100_000.0, desired_WR=0.04)
        assert isinstance(result, Withdrawal_Rates)

    def test_nominal_wr_zero_during_deferral(
        self,
        fia_defaults: Annuity_FIA,
    ) -> None:
        """nominal_WR is 0.0 while client is in the deferral period."""
        result = fia_defaults.calc_withdrawal_rates(100_000.0, desired_WR=0.04)
        assert result.nominal_WR == pytest.approx(0.0)

    def test_real_wr_zero_during_deferral(
        self,
        fia_defaults: Annuity_FIA,
    ) -> None:
        """real_WR is 0.0 while client is in the deferral period."""
        result = fia_defaults.calc_withdrawal_rates(100_000.0, desired_WR=0.04)
        assert result.real_WR == pytest.approx(0.0)

    def test_nominal_wr_positive_past_income_age(
        self,
        fia_past_income_age: Annuity_FIA,
        scenarios_df_fia: pl.DataFrame,
    ) -> None:
        """nominal_WR is positive once client has passed the income start age."""
        result = fia_past_income_age.calc_withdrawal_rates(
            100_000.0,
            desired_WR=0.04,
            obj_scenarios=scenarios_df_fia,
        )
        assert result.nominal_WR > 0.0

    def test_real_wr_equals_nominal_when_no_inflation(
        self,
        fia_past_income_age: Annuity_FIA,
        scenarios_df_fia: pl.DataFrame,
    ) -> None:
        """real_WR equals nominal_WR when no inflation data is provided."""
        result = fia_past_income_age.calc_withdrawal_rates(
            100_000.0,
            desired_WR=0.04,
            obj_scenarios=scenarios_df_fia,
        )
        assert result.real_WR == pytest.approx(result.nominal_WR, rel=1e-9)

    def test_real_wr_deflated_with_inflation(
        self,
        fia_past_income_age: Annuity_FIA,
        scenarios_df_fia: pl.DataFrame,
        inflation_wr_df: pl.DataFrame,
    ) -> None:
        """real_WR is deflated by the inverse inflation factor."""
        result_no_infl = fia_past_income_age.calc_withdrawal_rates(
            100_000.0,
            desired_WR=0.04,
            obj_scenarios=scenarios_df_fia,
        )
        result_with_infl = fia_past_income_age.calc_withdrawal_rates(
            100_000.0,
            desired_WR=0.04,
            obj_scenarios=scenarios_df_fia,
            obj_inflation=inflation_wr_df,
        )
        expected_real = result_no_infl.nominal_WR * (1.0 / 1.03)
        assert result_with_infl.real_WR == pytest.approx(expected_real, rel=1e-9)

    def test_real_wr_less_than_nominal_when_inflation_positive(
        self,
        fia_past_income_age: Annuity_FIA,
        scenarios_df_fia: pl.DataFrame,
        inflation_wr_df: pl.DataFrame,
    ) -> None:
        """real_WR < nominal_WR when the inflation factor is above 1.0."""
        result = fia_past_income_age.calc_withdrawal_rates(
            100_000.0,
            desired_WR=0.04,
            obj_scenarios=scenarios_df_fia,
            obj_inflation=inflation_wr_df,
        )
        assert result.real_WR < result.nominal_WR

    def test_invalid_principal_zero_raises(
        self,
        fia_defaults: Annuity_FIA,
    ) -> None:
        """Zero amount_principal raises Exception_Validation_Input."""
        with pytest.raises(Exception_Validation_Input):
            fia_defaults.calc_withdrawal_rates(0.0, desired_WR=0.04)

    def test_invalid_principal_negative_raises(
        self,
        fia_defaults: Annuity_FIA,
    ) -> None:
        """Negative amount_principal raises Exception_Validation_Input."""
        with pytest.raises(Exception_Validation_Input):
            fia_defaults.calc_withdrawal_rates(-1.0, desired_WR=0.04)

    def test_invalid_desired_wr_zero_raises(
        self,
        fia_defaults: Annuity_FIA,
    ) -> None:
        """Zero desired_WR raises Exception_Validation_Input."""
        with pytest.raises(Exception_Validation_Input):
            fia_defaults.calc_withdrawal_rates(100_000.0, desired_WR=0.0)

    def test_invalid_desired_wr_negative_raises(
        self,
        fia_defaults: Annuity_FIA,
    ) -> None:
        """Negative desired_WR raises Exception_Validation_Input."""
        with pytest.raises(Exception_Validation_Input):
            fia_defaults.calc_withdrawal_rates(100_000.0, desired_WR=-0.04)

    def test_inflation_df_missing_inverse_column_raises(
        self,
        fia_defaults: Annuity_FIA,
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
            fia_defaults.calc_withdrawal_rates(
                100_000.0,
                desired_WR=0.04,
                obj_inflation=incomplete_df,
            )


# ======================================================================
# Income starting age tests
# ======================================================================


@pytest.mark.unit()
class Test_Annuity_FIA_Income_Starting_Age:
    """Test m_annuity_income_starting_age for Annuity_FIA."""

    def test_default_is_client_age(self, fia_defaults: Annuity_FIA) -> None:
        """Default income starting age equals client_age."""
        assert fia_defaults.annuity_income_starting_age == fia_defaults.client_age

    def test_custom_income_starting_age_stored(self) -> None:
        """Explicitly provided income starting age >= client_age is stored correctly."""
        fia = Annuity_FIA(
            client_age=55,
            annuity_payout_rate=0.05,
            age_income_start=65,
            age_max_ratchet=85,
            rate_rollup_benefit=0.05,
            annuity_income_starting_age=60,
        )
        assert fia.annuity_income_starting_age == 60

    def test_income_starting_age_equal_to_client_age(self) -> None:
        """Income starting age equal to client_age is accepted."""
        fia = Annuity_FIA(
            client_age=55,
            annuity_payout_rate=0.05,
            age_income_start=65,
            age_max_ratchet=85,
            rate_rollup_benefit=0.05,
            annuity_income_starting_age=55,
        )
        assert fia.annuity_income_starting_age == 55

    def test_income_starting_age_below_client_age_raises(self) -> None:
        """Income starting age below client_age raises Exception_Validation_Input."""
        with pytest.raises(Exception_Validation_Input):
            Annuity_FIA(
                client_age=55,
                annuity_payout_rate=0.05,
                age_income_start=65,
                age_max_ratchet=85,
                rate_rollup_benefit=0.05,
                annuity_income_starting_age=50,
            )

    def test_member_variable_directly(self) -> None:
        """m_annuity_income_starting_age member variable equals the provided value."""
        fia = Annuity_FIA(
            client_age=55,
            annuity_payout_rate=0.05,
            age_income_start=65,
            age_max_ratchet=85,
            rate_rollup_benefit=0.05,
            annuity_income_starting_age=65,
        )
        assert fia.m_annuity_income_starting_age == 65
