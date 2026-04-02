"""Unit tests for Annuity_DIA class.

Tests cover initialization, property access, input validation,
payout calculations, analytical calculations (future value, PV of
deferred income, commutation, breakeven, mortality credit, death
benefit), and string representation.

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

from src.products.annuity.annuity_base import Annuity_Payout_Option, Annuity_Type, Withdrawal_Rates
from src.products.annuity.annuity_DIA import Annuity_DIA
from src.utils.custom_exceptions_errors_loggers.exception_custom import (
    Exception_Calculation,
    Exception_Validation_Input,
)


# ======================================================================
# Fixtures
# ======================================================================


@pytest.fixture()
def dia_defaults() -> Annuity_DIA:
    """Create an Annuity_DIA with default parameters (in deferral)."""
    return Annuity_DIA(
        client_age=55,
        annuity_payout_rate=0.08,
        age_income_start=65,
    )


@pytest.fixture()
def dia_with_cola() -> Annuity_DIA:
    """Create a DIA with COLA and period certain."""
    return Annuity_DIA(
        client_age=55,
        annuity_payout_rate=0.08,
        age_income_start=65,
        payout_option=Annuity_Payout_Option.LIFE_WITH_PERIOD_CERTAIN,
        guarantee_period_years=10,
        rate_COLA=0.03,
    )


@pytest.fixture()
def dia_with_rop() -> Annuity_DIA:
    """Create a DIA with return-of-premium death benefit."""
    return Annuity_DIA(
        client_age=50,
        annuity_payout_rate=0.10,
        age_income_start=65,
        has_death_benefit_ROP=True,
        rate_mortality_credit=0.02,
    )


# ======================================================================
# Initialization tests
# ======================================================================


@pytest.mark.unit()
class Test_Annuity_DIA_Init:
    """Test Annuity_DIA initialization and parameter storage."""

    def test_init_default_parameters(self, dia_defaults: Annuity_DIA) -> None:
        """Verify default parameter values."""
        assert dia_defaults.client_age == 55
        assert dia_defaults.annuity_payout_rate == pytest.approx(0.08)
        assert dia_defaults.annuity_type == Annuity_Type.ANNUITY_DIA
        assert dia_defaults.income_start_age == 65
        assert dia_defaults.deferral_years == 10
        assert dia_defaults.payout_option == Annuity_Payout_Option.LIFE_ONLY
        assert dia_defaults.guarantee_period_years == 0
        assert dia_defaults.is_inflation_adjusted is False
        assert dia_defaults.rate_COLA == pytest.approx(0.0)
        assert dia_defaults.has_death_benefit_ROP is False
        assert dia_defaults.rate_mortality_credit == pytest.approx(0.0)
        assert dia_defaults.payment_frequency == 12

    def test_init_custom_parameters(self) -> None:
        """Verify custom parameters are stored correctly."""
        dia = Annuity_DIA(
            client_age=50,
            annuity_payout_rate=0.10,
            age_income_start=70,
            payout_option=Annuity_Payout_Option.PERIOD_CERTAIN,
            guarantee_period_years=15,
            is_inflation_adjusted=True,
            rate_COLA=0.02,
            has_death_benefit_ROP=True,
            rate_mortality_credit=0.015,
            payment_frequency=4,
        )
        assert dia.client_age == 50
        assert dia.income_start_age == 70
        assert dia.deferral_years == 20
        assert dia.payout_option == Annuity_Payout_Option.PERIOD_CERTAIN
        assert dia.guarantee_period_years == 15
        assert dia.is_inflation_adjusted is True
        assert dia.rate_COLA == pytest.approx(0.02)
        assert dia.has_death_benefit_ROP is True
        assert dia.rate_mortality_credit == pytest.approx(0.015)
        assert dia.payment_frequency == 4


# ======================================================================
# Input validation tests
# ======================================================================


@pytest.mark.unit()
class Test_Annuity_DIA_Validation:
    """Test Annuity_DIA input validation."""

    def test_invalid_client_age(self) -> None:
        """Non-positive client age should raise."""
        with pytest.raises(Exception_Validation_Input):
            Annuity_DIA(client_age=0, annuity_payout_rate=0.08, age_income_start=65)

    def test_income_start_age_not_greater_than_client_age(self) -> None:
        """Income start age must be > client age."""
        with pytest.raises(Exception_Validation_Input):
            Annuity_DIA(client_age=65, annuity_payout_rate=0.08, age_income_start=65)

    def test_income_start_age_less_than_client_age(self) -> None:
        """Income start age < client age should raise."""
        with pytest.raises(Exception_Validation_Input):
            Annuity_DIA(client_age=70, annuity_payout_rate=0.08, age_income_start=65)

    def test_invalid_payout_option_string(self) -> None:
        """String payout option should raise."""
        with pytest.raises(Exception_Validation_Input):
            Annuity_DIA(
                client_age=55,
                annuity_payout_rate=0.08,
                age_income_start=65,
                payout_option="life_only",  # type: ignore[arg-type]
            )

    def test_period_certain_requires_guarantee(self) -> None:
        """Period Certain with zero guarantee should raise."""
        with pytest.raises(Exception_Validation_Input):
            Annuity_DIA(
                client_age=55,
                annuity_payout_rate=0.08,
                age_income_start=65,
                payout_option=Annuity_Payout_Option.PERIOD_CERTAIN,
                guarantee_period_years=0,
            )

    def test_invalid_cola_rate_negative(self) -> None:
        """Negative COLA rate should raise."""
        with pytest.raises(Exception_Validation_Input):
            Annuity_DIA(
                client_age=55,
                annuity_payout_rate=0.08,
                age_income_start=65,
                rate_COLA=-0.01,
            )

    def test_invalid_mortality_credit_negative(self) -> None:
        """Negative mortality credit rate should raise."""
        with pytest.raises(Exception_Validation_Input):
            Annuity_DIA(
                client_age=55,
                annuity_payout_rate=0.08,
                age_income_start=65,
                rate_mortality_credit=-0.01,
            )

    def test_invalid_payment_frequency(self) -> None:
        """Zero payment frequency should raise."""
        with pytest.raises(Exception_Validation_Input):
            Annuity_DIA(
                client_age=55,
                annuity_payout_rate=0.08,
                age_income_start=65,
                payment_frequency=0,
            )


# ======================================================================
# Payout calculation tests
# ======================================================================


@pytest.mark.unit()
class Test_Annuity_DIA_Payout_Calculations:
    """Test payout calculation methods."""

    def test_calc_annuity_payout_during_deferral(
        self,
        dia_defaults: Annuity_DIA,
    ) -> None:
        """During deferral, payout is 0."""
        payout = dia_defaults.calc_annuity_payout(100_000.0)
        assert payout == pytest.approx(0.0)

    def test_calc_annuity_payout_invalid_principal(
        self,
        dia_defaults: Annuity_DIA,
    ) -> None:
        """Non-positive principal should raise."""
        with pytest.raises(Exception_Validation_Input):
            dia_defaults.calc_annuity_payout(0.0)

    def test_calc_payout_in_year_basic(self, dia_defaults: Annuity_DIA) -> None:
        """Year 1 payout = principal * payout rate (no COLA)."""
        payout = dia_defaults.calc_payout_in_year(100_000.0, 1)
        assert payout == pytest.approx(8_000.0)

    def test_calc_payout_in_year_with_cola(
        self,
        dia_with_cola: Annuity_DIA,
    ) -> None:
        """Year N payout applies COLA."""
        expected = 8_000.0 * (1.03**4)
        payout = dia_with_cola.calc_payout_in_year(100_000.0, 5)
        assert payout == pytest.approx(expected, rel=1e-9)

    def test_calc_payout_in_year_invalid_year(
        self,
        dia_defaults: Annuity_DIA,
    ) -> None:
        """Year < 1 should raise."""
        with pytest.raises(Exception_Validation_Input):
            dia_defaults.calc_payout_in_year(100_000.0, 0)

    def test_calc_monthly_payout_during_deferral(
        self,
        dia_defaults: Annuity_DIA,
    ) -> None:
        """Monthly payout during deferral is 0."""
        monthly = dia_defaults.calc_monthly_payout(100_000.0)
        assert monthly == pytest.approx(0.0)


# ======================================================================
# Analytical calculation tests
# ======================================================================


@pytest.mark.unit()
class Test_Annuity_DIA_Analytical_Calculations:
    """Test future value, PV deferred income, commutation, breakeven, etc."""

    def test_calc_future_value(self, dia_defaults: Annuity_DIA) -> None:
        """FV = principal * (1 + growth_rate) ^ deferral_years."""
        fv = dia_defaults.calc_future_value(100_000.0, 0.05)
        expected = 100_000.0 * (1.05**10)
        assert fv == pytest.approx(expected, rel=1e-9)

    def test_calc_future_value_zero_growth(
        self,
        dia_defaults: Annuity_DIA,
    ) -> None:
        """Zero growth returns the principal unchanged."""
        fv = dia_defaults.calc_future_value(100_000.0, 0.0)
        assert fv == pytest.approx(100_000.0)

    def test_calc_future_value_invalid_principal(
        self,
        dia_defaults: Annuity_DIA,
    ) -> None:
        """Non-positive principal should raise."""
        with pytest.raises(Exception_Validation_Input):
            dia_defaults.calc_future_value(0.0, 0.05)

    def test_calc_future_value_invalid_growth_rate(
        self,
        dia_defaults: Annuity_DIA,
    ) -> None:
        """Non-numeric growth rate should raise."""
        with pytest.raises(Exception_Validation_Input):
            dia_defaults.calc_future_value(100_000.0, "five")  # type: ignore[arg-type]

    def test_calc_present_value_deferred_income(
        self,
        dia_defaults: Annuity_DIA,
    ) -> None:
        """PV of deferred income discounts back from deferral start."""
        pv = dia_defaults.calc_present_value_deferred_income(
            amount_principal=100_000.0,
            discount_rate=0.04,
            life_expectancy_years=30,
        )
        # k=10, payment_years=30-10=20, base_payout=8000
        k = 10
        base = 8_000.0
        expected = sum(base / (1.04 ** (k + t)) for t in range(1, 21))
        assert pv == pytest.approx(expected, rel=1e-9)

    def test_calc_present_value_deferred_income_insufficient_life(
        self,
        dia_defaults: Annuity_DIA,
    ) -> None:
        """If life expectancy <= deferral, PV is 0."""
        pv = dia_defaults.calc_present_value_deferred_income(
            amount_principal=100_000.0,
            discount_rate=0.04,
            life_expectancy_years=10,
        )
        assert pv == pytest.approx(0.0)

    def test_calc_present_value_deferred_income_invalid_inputs(
        self,
        dia_defaults: Annuity_DIA,
    ) -> None:
        """Invalid inputs should raise."""
        with pytest.raises(Exception_Validation_Input):
            dia_defaults.calc_present_value_deferred_income(0.0, 0.04, 30)
        with pytest.raises(Exception_Validation_Input):
            dia_defaults.calc_present_value_deferred_income(100_000.0, -0.01, 30)
        with pytest.raises(Exception_Validation_Input):
            dia_defaults.calc_present_value_deferred_income(100_000.0, 0.04, 0)

    def test_calc_commutation_value(self, dia_defaults: Annuity_DIA) -> None:
        """Commutation value = PV of remaining payments."""
        cv = dia_defaults.calc_commutation_value(
            amount_principal=100_000.0,
            discount_rate=0.04,
            remaining_payment_years=15,
        )
        base = 8_000.0
        expected = sum(base / (1.04**t) for t in range(1, 16))
        assert cv == pytest.approx(expected, rel=1e-9)

    def test_calc_commutation_value_invalid_inputs(
        self,
        dia_defaults: Annuity_DIA,
    ) -> None:
        """Invalid inputs should raise."""
        with pytest.raises(Exception_Validation_Input):
            dia_defaults.calc_commutation_value(0.0, 0.04, 15)
        with pytest.raises(Exception_Validation_Input):
            dia_defaults.calc_commutation_value(100_000.0, -0.01, 15)
        with pytest.raises(Exception_Validation_Input):
            dia_defaults.calc_commutation_value(100_000.0, 0.04, 0)

    def test_calc_breakeven_age(self, dia_defaults: Annuity_DIA) -> None:
        """Breakeven age starts counting from income start age."""
        # 100000 / 8000 = 12.5 → year 13 from income start
        # breakeven_age = 65 + 13 = 78
        breakeven_age = dia_defaults.calc_breakeven_age(100_000.0)
        assert breakeven_age == 78

    def test_calc_breakeven_age_invalid_principal(
        self,
        dia_defaults: Annuity_DIA,
    ) -> None:
        """Non-positive principal should raise."""
        with pytest.raises(Exception_Validation_Input):
            dia_defaults.calc_breakeven_age(0.0)

    def test_calc_breakeven_not_reached(self) -> None:
        """Very low payout rate may not reach breakeven."""
        dia = Annuity_DIA(
            client_age=55,
            annuity_payout_rate=0.001,
            age_income_start=65,
        )
        with pytest.raises(Exception_Calculation):
            dia.calc_breakeven_age(100_000.0)

    def test_calc_mortality_credit_value(
        self,
        dia_with_rop: Annuity_DIA,
    ) -> None:
        """Mortality credit = principal * ((1 + rate)^n - 1)."""
        mc = dia_with_rop.calc_mortality_credit_value(100_000.0)
        n = dia_with_rop.deferral_years  # 15
        expected = 100_000.0 * ((1.02**n) - 1)
        assert mc == pytest.approx(expected, rel=1e-9)

    def test_calc_mortality_credit_zero_rate(
        self,
        dia_defaults: Annuity_DIA,
    ) -> None:
        """Zero mortality credit rate yields 0."""
        mc = dia_defaults.calc_mortality_credit_value(100_000.0)
        assert mc == pytest.approx(0.0)

    def test_calc_death_benefit_with_rop(
        self,
        dia_with_rop: Annuity_DIA,
    ) -> None:
        """ROP death benefit returns the premium."""
        db = dia_with_rop.calc_death_benefit_during_deferral(100_000.0)
        assert db == pytest.approx(100_000.0)

    def test_calc_death_benefit_without_rop(
        self,
        dia_defaults: Annuity_DIA,
    ) -> None:
        """Without ROP, death benefit is 0."""
        db = dia_defaults.calc_death_benefit_during_deferral(100_000.0)
        assert db == pytest.approx(0.0)

    def test_calc_death_benefit_invalid_principal(
        self,
        dia_with_rop: Annuity_DIA,
    ) -> None:
        """Non-positive principal should raise."""
        with pytest.raises(Exception_Validation_Input):
            dia_with_rop.calc_death_benefit_during_deferral(0.0)


# ======================================================================
# String representation tests
# ======================================================================


@pytest.mark.unit()
class Test_Annuity_DIA_String_Representation:
    """Test get_annuity_as_string output."""

    def test_string_contains_dia(self, dia_defaults: Annuity_DIA) -> None:
        """String should contain 'DIA'."""
        result = dia_defaults.get_annuity_as_string()
        assert "DIA" in result

    def test_string_contains_income_start_age(
        self,
        dia_defaults: Annuity_DIA,
    ) -> None:
        """String should include the income start age."""
        result = dia_defaults.get_annuity_as_string()
        assert "65" in result

    def test_string_contains_deferral(self, dia_defaults: Annuity_DIA) -> None:
        """String should include deferral years."""
        result = dia_defaults.get_annuity_as_string()
        assert "10" in result

    def test_string_contains_rop_when_active(
        self,
        dia_with_rop: Annuity_DIA,
    ) -> None:
        """String should mention ROP when active."""
        result = dia_with_rop.get_annuity_as_string()
        assert "ROP" in result


# ======================================================================
# Inflation DataFrame tests
# ======================================================================


@pytest.fixture()
def inflation_df_single() -> pl.DataFrame:
    """Single-row inflation DataFrame representing a 3 % annual factor."""
    return pl.DataFrame(
        {
            "Start Date": ["2024-01-01"],
            "End Date": ["2025-01-01"],
            "Inflation Factor": [1.03],
        },
    )


@pytest.fixture()
def dia_inflation_adjusted() -> Annuity_DIA:
    """DIA with is_inflation_adjusted=True (starts in deferral)."""
    return Annuity_DIA(
        client_age=55,
        annuity_payout_rate=0.08,
        age_income_start=65,
        is_inflation_adjusted=True,
    )


@pytest.mark.unit()
class Test_Annuity_DIA_Inflation_Adjustment:
    """Test calc_annuity_payout and calc_monthly_payout with obj_inflation DataFrame.

    The ``obj_inflation`` parameter must be a Polars DataFrame with columns:
    ``Start Date``, ``End Date``, and ``Inflation Factor``.

    For a DIA object the income deferral check executes before the inflation
    block: while ``client_age < age_income_start`` the payout is always
    ``0.0`` regardless of ``obj_inflation``.  The inflation adjustment is
    applied only once the client reaches the income start age.
    """

    def test_inflation_df_accepted_during_deferral(
        self,
        dia_inflation_adjusted: Annuity_DIA,
        inflation_df_single: pl.DataFrame,
    ) -> None:
        """A valid DataFrame is accepted; deferral returns 0.0 before inflation logic."""
        payout = dia_inflation_adjusted.calc_annuity_payout(
            100_000.0,
            obj_inflation=inflation_df_single,
        )
        assert payout == pytest.approx(0.0)

    def test_none_inflation_accepted_during_deferral(
        self,
        dia_inflation_adjusted: Annuity_DIA,
    ) -> None:
        """None is accepted as obj_inflation; deferral returns 0.0."""
        payout = dia_inflation_adjusted.calc_annuity_payout(
            100_000.0,
            obj_inflation=None,
        )
        assert payout == pytest.approx(0.0)

    def test_inflation_df_wrong_columns_during_deferral(
        self,
        dia_inflation_adjusted: Annuity_DIA,
    ) -> None:
        """Bad schema DataFrame is never validated during deferral (returns 0.0)."""
        bad_df = pl.DataFrame({"Date": ["2024-01-01"], "Factor": [1.03]})
        # The deferral check (client_age < age_income_start) triggers first,
        # so column validation is never reached and 0.0 is returned.
        payout = dia_inflation_adjusted.calc_annuity_payout(
            100_000.0,
            obj_inflation=bad_df,
        )
        assert payout == pytest.approx(0.0)

    def test_non_inflation_adjusted_dia_accepts_df(
        self,
        dia_defaults: Annuity_DIA,
        inflation_df_single: pl.DataFrame,
    ) -> None:
        """DIA with is_inflation_adjusted=False accepts DataFrame; returns 0.0."""
        payout = dia_defaults.calc_annuity_payout(
            100_000.0,
            obj_inflation=inflation_df_single,
        )
        assert payout == pytest.approx(0.0)

    def test_monthly_payout_accepts_polars_dataframe(
        self,
        dia_defaults: Annuity_DIA,
        inflation_df_single: pl.DataFrame,
    ) -> None:
        """calc_monthly_payout accepts pl.DataFrame as obj_inflation."""
        monthly = dia_defaults.calc_monthly_payout(
            100_000.0,
            obj_inflation=inflation_df_single,
        )
        assert monthly == pytest.approx(0.0)  # still in deferral


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
class Test_Annuity_DIA_Withdrawal_Rates:
    """Tests for calc_withdrawal_rates on Annuity_DIA.

    ``dia_defaults`` is in deferral (client_age=55, income_start=65), so
    nominal_WR = 0.0 and real_WR = 0.0 for all deferral-phase tests.
    """

    def test_returns_withdrawal_rates_struct(
        self,
        dia_defaults: Annuity_DIA,
    ) -> None:
        """calc_withdrawal_rates returns a Withdrawal_Rates instance."""
        result = dia_defaults.calc_withdrawal_rates(100_000.0, desired_WR=0.04)
        assert isinstance(result, Withdrawal_Rates)

    def test_nominal_wr_zero_during_deferral(
        self,
        dia_defaults: Annuity_DIA,
    ) -> None:
        """nominal_WR is 0.0 while client is in the deferral period."""
        result = dia_defaults.calc_withdrawal_rates(100_000.0, desired_WR=0.04)
        assert result.nominal_WR == pytest.approx(0.0)

    def test_real_wr_zero_during_deferral(
        self,
        dia_defaults: Annuity_DIA,
    ) -> None:
        """real_WR is 0.0 while client is in the deferral period."""
        result = dia_defaults.calc_withdrawal_rates(100_000.0, desired_WR=0.04)
        assert result.real_WR == pytest.approx(0.0)

    def test_real_wr_zero_during_deferral_with_inflation(
        self,
        dia_defaults: Annuity_DIA,
        inflation_wr_df: pl.DataFrame,
    ) -> None:
        """real_WR remains 0.0 during deferral even with an inflation DataFrame."""
        result = dia_defaults.calc_withdrawal_rates(
            100_000.0,
            desired_WR=0.04,
            obj_inflation=inflation_wr_df,
        )
        assert result.real_WR == pytest.approx(0.0)

    def test_nominal_wr_positive_past_income_age(self) -> None:
        """nominal_WR is positive once client has passed the income start age."""
        dia = Annuity_DIA(
            client_age=55,
            annuity_payout_rate=0.08,
            age_income_start=65,
        )
        dia.m_client_age = 66  # advance past income start
        result = dia.calc_withdrawal_rates(100_000.0, desired_WR=0.04)
        assert result.nominal_WR > 0.0

    def test_real_wr_equals_nominal_when_no_inflation_past_income_age(self) -> None:
        """real_WR equals nominal_WR when no inflation data is provided."""
        dia = Annuity_DIA(
            client_age=55,
            annuity_payout_rate=0.08,
            age_income_start=65,
        )
        dia.m_client_age = 66
        result = dia.calc_withdrawal_rates(100_000.0, desired_WR=0.04)
        assert result.real_WR == pytest.approx(result.nominal_WR, rel=1e-9)

    def test_invalid_principal_zero_raises(
        self,
        dia_defaults: Annuity_DIA,
    ) -> None:
        """Zero amount_principal raises Exception_Validation_Input."""
        with pytest.raises(Exception_Validation_Input):
            dia_defaults.calc_withdrawal_rates(0.0, desired_WR=0.04)

    def test_invalid_principal_negative_raises(
        self,
        dia_defaults: Annuity_DIA,
    ) -> None:
        """Negative amount_principal raises Exception_Validation_Input."""
        with pytest.raises(Exception_Validation_Input):
            dia_defaults.calc_withdrawal_rates(-1.0, desired_WR=0.04)

    def test_invalid_desired_wr_zero_raises(
        self,
        dia_defaults: Annuity_DIA,
    ) -> None:
        """Zero desired_WR raises Exception_Validation_Input."""
        with pytest.raises(Exception_Validation_Input):
            dia_defaults.calc_withdrawal_rates(100_000.0, desired_WR=0.0)

    def test_invalid_desired_wr_negative_raises(
        self,
        dia_defaults: Annuity_DIA,
    ) -> None:
        """Negative desired_WR raises Exception_Validation_Input."""
        with pytest.raises(Exception_Validation_Input):
            dia_defaults.calc_withdrawal_rates(100_000.0, desired_WR=-0.04)

    def test_inflation_df_missing_inverse_column_raises(
        self,
        dia_defaults: Annuity_DIA,
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
            dia_defaults.calc_withdrawal_rates(
                100_000.0,
                desired_WR=0.04,
                obj_inflation=incomplete_df,
            )


# ======================================================================
# Income starting age tests
# ======================================================================


@pytest.mark.unit()
class Test_Annuity_DIA_Income_Starting_Age:
    """Test m_annuity_income_starting_age for Annuity_DIA."""

    def test_default_is_client_age(self, dia_defaults: Annuity_DIA) -> None:
        """Default income starting age equals client_age."""
        assert dia_defaults.annuity_income_starting_age == dia_defaults.client_age

    def test_custom_income_starting_age_stored(self) -> None:
        """Explicitly provided income starting age >= client_age is stored correctly."""
        dia = Annuity_DIA(
            client_age=55,
            annuity_payout_rate=0.08,
            age_income_start=65,
            annuity_income_starting_age=60,
        )
        assert dia.annuity_income_starting_age == 60

    def test_income_starting_age_equal_to_client_age(self) -> None:
        """Income starting age equal to client_age is accepted."""
        dia = Annuity_DIA(
            client_age=55,
            annuity_payout_rate=0.08,
            age_income_start=65,
            annuity_income_starting_age=55,
        )
        assert dia.annuity_income_starting_age == 55

    def test_income_starting_age_below_client_age_raises(self) -> None:
        """Income starting age below client_age raises Exception_Validation_Input."""
        with pytest.raises(Exception_Validation_Input):
            Annuity_DIA(
                client_age=55,
                annuity_payout_rate=0.08,
                age_income_start=65,
                annuity_income_starting_age=50,
            )

    def test_member_variable_directly(self) -> None:
        """m_annuity_income_starting_age member variable equals the provided value."""
        dia = Annuity_DIA(
            client_age=55,
            annuity_payout_rate=0.08,
            age_income_start=65,
            annuity_income_starting_age=65,
        )
        assert dia.m_annuity_income_starting_age == 65
