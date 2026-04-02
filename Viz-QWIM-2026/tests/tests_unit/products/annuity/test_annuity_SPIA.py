"""Unit tests for Annuity_SPIA class.

Tests cover initialization, property access, input validation,
payout calculations, analytical calculations, and string representation.

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
from src.products.annuity.annuity_SPIA import Annuity_SPIA
from src.utils.custom_exceptions_errors_loggers.exception_custom import (
    Exception_Calculation,
    Exception_Validation_Input,
)


# ======================================================================
# Fixtures
# ======================================================================


@pytest.fixture()
def spia_defaults() -> Annuity_SPIA:
    """Create an Annuity_SPIA with default parameters."""
    return Annuity_SPIA(client_age=65, annuity_payout_rate=0.06)


@pytest.fixture()
def spia_with_cola() -> Annuity_SPIA:
    """Create a SPIA with COLA and period certain."""
    return Annuity_SPIA(
        client_age=65,
        annuity_payout_rate=0.06,
        payout_option=Annuity_Payout_Option.LIFE_WITH_PERIOD_CERTAIN,
        guarantee_period_years=10,
        rate_COLA=0.03,
        payment_frequency=12,
    )


@pytest.fixture()
def spia_joint_life() -> Annuity_SPIA:
    """Create a SPIA with joint-life payout."""
    return Annuity_SPIA(
        client_age=65,
        annuity_payout_rate=0.06,
        payout_option=Annuity_Payout_Option.JOINT_LIFE,
        joint_survivor_pct=0.75,
    )


@pytest.fixture()
def spia_inflation_adjusted() -> Annuity_SPIA:
    """Create a SPIA with is_inflation_adjusted=True."""
    return Annuity_SPIA(
        client_age=65,
        annuity_payout_rate=0.06,
        is_inflation_adjusted=True,
    )


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
def inflation_df_multi() -> pl.DataFrame:
    """Multi-row inflation DataFrame; cumulative factor = 1.03 * 1.025 = 1.05575."""
    return pl.DataFrame(
        {
            "Start Date": ["2023-01-01", "2024-01-01"],
            "End Date": ["2024-01-01", "2025-01-01"],
            "Inflation Factor": [1.03, 1.025],
        },
    )


# ======================================================================
# Initialization tests
# ======================================================================


@pytest.mark.unit()
class Test_Annuity_SPIA_Init:
    """Test Annuity_SPIA initialization and parameter storage."""

    def test_init_default_parameters(self, spia_defaults: Annuity_SPIA) -> None:
        """Verify default parameter values are stored correctly."""
        assert spia_defaults.client_age == 65
        assert spia_defaults.annuity_payout_rate == pytest.approx(0.06)
        assert spia_defaults.annuity_type == Annuity_Type.ANNUITY_SPIA
        assert spia_defaults.payout_option == Annuity_Payout_Option.LIFE_ONLY
        assert spia_defaults.guarantee_period_years == 0
        assert spia_defaults.rate_COLA == pytest.approx(0.0)
        assert spia_defaults.joint_survivor_pct == pytest.approx(1.0)
        assert spia_defaults.is_inflation_adjusted is False
        assert spia_defaults.payment_frequency == 12

    def test_init_custom_parameters(self) -> None:
        """Verify custom parameters are stored correctly."""
        spia = Annuity_SPIA(
            client_age=70,
            annuity_payout_rate=0.08,
            payout_option=Annuity_Payout_Option.PERIOD_CERTAIN,
            guarantee_period_years=20,
            rate_COLA=0.02,
            joint_survivor_pct=0.50,
            is_inflation_adjusted=True,
            payment_frequency=4,
        )
        assert spia.client_age == 70
        assert spia.annuity_payout_rate == pytest.approx(0.08)
        assert spia.payout_option == Annuity_Payout_Option.PERIOD_CERTAIN
        assert spia.guarantee_period_years == 20
        assert spia.rate_COLA == pytest.approx(0.02)
        assert spia.joint_survivor_pct == pytest.approx(0.50)
        assert spia.is_inflation_adjusted is True
        assert spia.payment_frequency == 4

    def test_init_with_cola(self, spia_with_cola: Annuity_SPIA) -> None:
        """Verify COLA and guarantee parameters."""
        assert spia_with_cola.rate_COLA == pytest.approx(0.03)
        assert spia_with_cola.guarantee_period_years == 10
        assert spia_with_cola.payout_option == Annuity_Payout_Option.LIFE_WITH_PERIOD_CERTAIN


# ======================================================================
# Input validation tests
# ======================================================================


@pytest.mark.unit()
class Test_Annuity_SPIA_Validation:
    """Test Annuity_SPIA input validation raises correct exceptions."""

    def test_invalid_client_age_zero(self) -> None:
        """Zero age should raise Exception_Validation_Input."""
        with pytest.raises(Exception_Validation_Input):
            Annuity_SPIA(client_age=0, annuity_payout_rate=0.06)

    def test_invalid_client_age_negative(self) -> None:
        """Negative age should raise Exception_Validation_Input."""
        with pytest.raises(Exception_Validation_Input):
            Annuity_SPIA(client_age=-5, annuity_payout_rate=0.06)

    def test_invalid_client_age_non_integer(self) -> None:
        """Non-integer age should raise Exception_Validation_Input."""
        with pytest.raises(Exception_Validation_Input):
            Annuity_SPIA(client_age=65.5, annuity_payout_rate=0.06)  # type: ignore[arg-type]

    def test_invalid_payout_rate_zero(self) -> None:
        """Zero payout rate should raise Exception_Validation_Input."""
        with pytest.raises(Exception_Validation_Input):
            Annuity_SPIA(client_age=65, annuity_payout_rate=0.0)

    def test_invalid_payout_rate_negative(self) -> None:
        """Negative payout rate should raise Exception_Validation_Input."""
        with pytest.raises(Exception_Validation_Input):
            Annuity_SPIA(client_age=65, annuity_payout_rate=-0.05)

    def test_invalid_payout_option_string(self) -> None:
        """String payout option should raise Exception_Validation_Input."""
        with pytest.raises(Exception_Validation_Input):
            Annuity_SPIA(
                client_age=65,
                annuity_payout_rate=0.06,
                payout_option="life_only",  # type: ignore[arg-type]
            )

    def test_invalid_guarantee_period_negative(self) -> None:
        """Negative guarantee period should raise Exception_Validation_Input."""
        with pytest.raises(Exception_Validation_Input):
            Annuity_SPIA(
                client_age=65,
                annuity_payout_rate=0.06,
                guarantee_period_years=-1,
            )

    def test_period_certain_requires_guarantee(self) -> None:
        """Period Certain with zero guarantee should raise."""
        with pytest.raises(Exception_Validation_Input):
            Annuity_SPIA(
                client_age=65,
                annuity_payout_rate=0.06,
                payout_option=Annuity_Payout_Option.PERIOD_CERTAIN,
                guarantee_period_years=0,
            )

    def test_life_with_period_certain_requires_guarantee(self) -> None:
        """Life with Period Certain and zero guarantee should raise."""
        with pytest.raises(Exception_Validation_Input):
            Annuity_SPIA(
                client_age=65,
                annuity_payout_rate=0.06,
                payout_option=Annuity_Payout_Option.LIFE_WITH_PERIOD_CERTAIN,
                guarantee_period_years=0,
            )

    def test_invalid_cola_rate_negative(self) -> None:
        """Negative COLA rate should raise Exception_Validation_Input."""
        with pytest.raises(Exception_Validation_Input):
            Annuity_SPIA(
                client_age=65,
                annuity_payout_rate=0.06,
                rate_COLA=-0.01,
            )

    def test_invalid_joint_survivor_pct_zero(self) -> None:
        """Zero joint survivor pct should raise Exception_Validation_Input."""
        with pytest.raises(Exception_Validation_Input):
            Annuity_SPIA(
                client_age=65,
                annuity_payout_rate=0.06,
                joint_survivor_pct=0.0,
            )

    def test_invalid_joint_survivor_pct_over_one(self) -> None:
        """Joint survivor pct > 1 should raise Exception_Validation_Input."""
        with pytest.raises(Exception_Validation_Input):
            Annuity_SPIA(
                client_age=65,
                annuity_payout_rate=0.06,
                joint_survivor_pct=1.5,
            )

    def test_invalid_payment_frequency_zero(self) -> None:
        """Zero payment frequency should raise Exception_Validation_Input."""
        with pytest.raises(Exception_Validation_Input):
            Annuity_SPIA(
                client_age=65,
                annuity_payout_rate=0.06,
                payment_frequency=0,
            )


# ======================================================================
# Payout calculation tests
# ======================================================================


@pytest.mark.unit()
class Test_Annuity_SPIA_Payout_Calculations:
    """Test payout calculation methods."""

    def test_calc_annuity_payout_basic(self, spia_defaults: Annuity_SPIA) -> None:
        """Annual payout = principal * payout rate."""
        payout = spia_defaults.calc_annuity_payout(100_000.0)
        assert payout == pytest.approx(6_000.0)

    def test_calc_annuity_payout_invalid_principal(
        self,
        spia_defaults: Annuity_SPIA,
    ) -> None:
        """Non-positive principal should raise."""
        with pytest.raises(Exception_Validation_Input):
            spia_defaults.calc_annuity_payout(0.0)

        with pytest.raises(Exception_Validation_Input):
            spia_defaults.calc_annuity_payout(-100.0)

    def test_calc_payout_in_year_first_year(
        self,
        spia_defaults: Annuity_SPIA,
    ) -> None:
        """Year 1 payout equals the base annual payout (no COLA)."""
        payout = spia_defaults.calc_payout_in_year(100_000.0, 1)
        assert payout == pytest.approx(6_000.0)

    def test_calc_payout_in_year_with_cola(
        self,
        spia_with_cola: Annuity_SPIA,
    ) -> None:
        """Year N payout applies COLA compounding."""
        # Year 5: 6000 * (1.03)^4
        expected = 6_000.0 * (1.03**4)
        payout = spia_with_cola.calc_payout_in_year(100_000.0, 5)
        assert payout == pytest.approx(expected, rel=1e-9)

    def test_calc_payout_in_year_invalid_year(
        self,
        spia_defaults: Annuity_SPIA,
    ) -> None:
        """Year number < 1 should raise."""
        with pytest.raises(Exception_Validation_Input):
            spia_defaults.calc_payout_in_year(100_000.0, 0)

    def test_calc_monthly_payout(self, spia_defaults: Annuity_SPIA) -> None:
        """Monthly payout = annual / 12."""
        monthly = spia_defaults.calc_monthly_payout(100_000.0)
        assert monthly == pytest.approx(500.0)


# ======================================================================
# Analytical calculation tests
# ======================================================================


@pytest.mark.unit()
class Test_Annuity_SPIA_Analytical_Calculations:
    """Test present value, breakeven, exclusion ratio, and cumulative."""

    def test_calc_present_value_life_only(
        self,
        spia_defaults: Annuity_SPIA,
    ) -> None:
        """Present value of life-only SPIA with zero COLA uses life expectancy."""
        pv = spia_defaults.calc_present_value_payments(
            amount_principal=100_000.0,
            discount_rate=0.04,
            life_expectancy_years=20,
        )
        expected = sum(6_000.0 / (1.04**t) for t in range(1, 21))
        assert pv == pytest.approx(expected, rel=1e-9)

    def test_calc_present_value_period_certain(self) -> None:
        """Period Certain uses guarantee period, not life expectancy."""
        spia = Annuity_SPIA(
            client_age=65,
            annuity_payout_rate=0.06,
            payout_option=Annuity_Payout_Option.PERIOD_CERTAIN,
            guarantee_period_years=10,
        )
        pv = spia.calc_present_value_payments(
            amount_principal=100_000.0,
            discount_rate=0.04,
            life_expectancy_years=20,
        )
        expected = sum(6_000.0 / (1.04**t) for t in range(1, 11))
        assert pv == pytest.approx(expected, rel=1e-9)

    def test_calc_present_value_life_with_period_certain(
        self,
        spia_with_cola: Annuity_SPIA,
    ) -> None:
        """Life with Period Certain uses max(guarantee, life expectancy)."""
        pv = spia_with_cola.calc_present_value_payments(
            amount_principal=100_000.0,
            discount_rate=0.04,
            life_expectancy_years=20,
        )
        expected = sum(6_000.0 * (1.03 ** (t - 1)) / (1.04**t) for t in range(1, 21))
        assert pv == pytest.approx(expected, rel=1e-9)

    def test_calc_present_value_invalid_inputs(
        self,
        spia_defaults: Annuity_SPIA,
    ) -> None:
        """Invalid inputs should raise."""
        with pytest.raises(Exception_Validation_Input):
            spia_defaults.calc_present_value_payments(0.0, 0.04, 20)
        with pytest.raises(Exception_Validation_Input):
            spia_defaults.calc_present_value_payments(100_000.0, -0.01, 20)
        with pytest.raises(Exception_Validation_Input):
            spia_defaults.calc_present_value_payments(100_000.0, 0.04, 0)

    def test_calc_breakeven_age(self, spia_defaults: Annuity_SPIA) -> None:
        """Breakeven: cumulative payments >= principal."""
        # 100000 / 6000 = 16.67 → year 17, age = 65 + 17 = 82
        breakeven_age = spia_defaults.calc_breakeven_age(100_000.0)
        assert breakeven_age == 82

    def test_calc_breakeven_age_with_cola(
        self,
        spia_with_cola: Annuity_SPIA,
    ) -> None:
        """COLA accelerates breakeven."""
        breakeven_age = spia_with_cola.calc_breakeven_age(100_000.0)
        assert breakeven_age < 82

    def test_calc_breakeven_age_invalid_principal(
        self,
        spia_defaults: Annuity_SPIA,
    ) -> None:
        """Non-positive principal should raise."""
        with pytest.raises(Exception_Validation_Input):
            spia_defaults.calc_breakeven_age(0.0)

    def test_calc_breakeven_not_reached(self) -> None:
        """Very low payout rate may not reach breakeven in 100 years."""
        spia = Annuity_SPIA(client_age=65, annuity_payout_rate=0.001)
        with pytest.raises(Exception_Calculation):
            spia.calc_breakeven_age(100_000.0)

    def test_calc_exclusion_ratio(self, spia_defaults: Annuity_SPIA) -> None:
        """Exclusion ratio = principal / total expected payments, capped at 1."""
        ratio = spia_defaults.calc_exclusion_ratio(100_000.0, 20)
        total_expected = 6_000.0 * 20
        expected_ratio = 100_000.0 / total_expected
        assert ratio == pytest.approx(expected_ratio, rel=1e-9)
        assert 0 < ratio <= 1.0

    def test_calc_exclusion_ratio_capped(self) -> None:
        """Very high payout rate with short expectancy should cap at 1.0."""
        spia = Annuity_SPIA(client_age=65, annuity_payout_rate=0.50)
        ratio = spia.calc_exclusion_ratio(100_000.0, 1)
        assert ratio == pytest.approx(1.0)

    def test_calc_exclusion_ratio_invalid_inputs(
        self,
        spia_defaults: Annuity_SPIA,
    ) -> None:
        """Invalid inputs should raise."""
        with pytest.raises(Exception_Validation_Input):
            spia_defaults.calc_exclusion_ratio(0.0, 20)
        with pytest.raises(Exception_Validation_Input):
            spia_defaults.calc_exclusion_ratio(100_000.0, 0)

    def test_calc_cumulative_payments(self, spia_defaults: Annuity_SPIA) -> None:
        """Cumulative payments over N years without COLA."""
        cumulative = spia_defaults.calc_cumulative_payments(100_000.0, 10)
        assert cumulative == pytest.approx(60_000.0)

    def test_calc_cumulative_payments_with_cola(
        self,
        spia_with_cola: Annuity_SPIA,
    ) -> None:
        """Cumulative payments with COLA compounding."""
        cumulative = spia_with_cola.calc_cumulative_payments(100_000.0, 10)
        expected = sum(6_000.0 * (1.03 ** (t - 1)) for t in range(1, 11))
        assert cumulative == pytest.approx(expected, rel=1e-9)

    def test_calc_cumulative_payments_invalid_inputs(
        self,
        spia_defaults: Annuity_SPIA,
    ) -> None:
        """Invalid inputs should raise."""
        with pytest.raises(Exception_Validation_Input):
            spia_defaults.calc_cumulative_payments(-100.0, 10)
        with pytest.raises(Exception_Validation_Input):
            spia_defaults.calc_cumulative_payments(100_000.0, 0)


# ======================================================================
# String representation tests
# ======================================================================


@pytest.mark.unit()
class Test_Annuity_SPIA_String_Representation:
    """Test get_annuity_as_string output."""

    def test_string_contains_spia(self, spia_defaults: Annuity_SPIA) -> None:
        """String should contain 'SPIA'."""
        result = spia_defaults.get_annuity_as_string()
        assert "SPIA" in result

    def test_string_contains_payout_option(
        self,
        spia_defaults: Annuity_SPIA,
    ) -> None:
        """String should include the payout option name."""
        result = spia_defaults.get_annuity_as_string()
        assert "Life Only" in result

    def test_string_contains_cola_info(
        self,
        spia_with_cola: Annuity_SPIA,
    ) -> None:
        """String should mention COLA when rate > 0."""
        result = spia_with_cola.get_annuity_as_string()
        assert "COLA" in result

    def test_string_contains_fixed_when_no_cola(
        self,
        spia_defaults: Annuity_SPIA,
    ) -> None:
        """String should say 'Fixed' when COLA is 0."""
        result = spia_defaults.get_annuity_as_string()
        assert "Fixed" in result


# ======================================================================
# Inflation DataFrame tests
# ======================================================================


@pytest.mark.unit()
class Test_Annuity_SPIA_Inflation_Adjustment:
    """Test calc_annuity_payout and calc_monthly_payout with obj_inflation DataFrame.

    The ``obj_inflation`` parameter must be a Polars DataFrame with columns:
    ``Start Date``, ``End Date``, and ``Inflation Factor``.

    When ``is_inflation_adjusted`` is ``True`` and a valid DataFrame is
    supplied, the total inflation factor (product of all ``Inflation Factor``
    values) is applied to the base annual payout.
    """

    def test_inflation_adjusted_single_factor(
        self,
        spia_inflation_adjusted: Annuity_SPIA,
        inflation_df_single: pl.DataFrame,
    ) -> None:
        """Single inflation factor multiplies the base payout."""
        payout = spia_inflation_adjusted.calc_annuity_payout(
            100_000.0,
            obj_inflation=inflation_df_single,
        )
        expected = 100_000.0 * 0.06 * 1.03
        assert payout == pytest.approx(expected, rel=1e-9)

    def test_inflation_adjusted_multi_factor(
        self,
        spia_inflation_adjusted: Annuity_SPIA,
        inflation_df_multi: pl.DataFrame,
    ) -> None:
        """Multiple inflation factors are multiplied (product) and applied."""
        payout = spia_inflation_adjusted.calc_annuity_payout(
            100_000.0,
            obj_inflation=inflation_df_multi,
        )
        # cumulative factor = 1.03 * 1.025 = 1.05575
        expected = 100_000.0 * 0.06 * 1.03 * 1.025
        assert payout == pytest.approx(expected, rel=1e-9)

    def test_inflation_not_applied_when_flag_false(
        self,
        spia_defaults: Annuity_SPIA,
        inflation_df_single: pl.DataFrame,
    ) -> None:
        """When is_inflation_adjusted is False, DataFrame is ignored."""
        payout = spia_defaults.calc_annuity_payout(
            100_000.0,
            obj_inflation=inflation_df_single,
        )
        assert payout == pytest.approx(6_000.0)

    def test_inflation_not_applied_when_none(
        self,
        spia_inflation_adjusted: Annuity_SPIA,
    ) -> None:
        """When obj_inflation is None, base payout is returned unchanged."""
        payout = spia_inflation_adjusted.calc_annuity_payout(
            100_000.0,
            obj_inflation=None,
        )
        assert payout == pytest.approx(6_000.0)

    def test_inflation_df_missing_columns_raises(
        self,
        spia_inflation_adjusted: Annuity_SPIA,
    ) -> None:
        """DataFrame missing required columns raises Exception_Validation_Input."""
        bad_df = pl.DataFrame({"Date": ["2024-01-01"], "Factor": [1.03]})
        with pytest.raises(Exception_Validation_Input):
            spia_inflation_adjusted.calc_annuity_payout(
                100_000.0,
                obj_inflation=bad_df,
            )

    def test_inflation_df_partial_columns_raises(
        self,
        spia_inflation_adjusted: Annuity_SPIA,
    ) -> None:
        """DataFrame with only some required columns raises Exception_Validation_Input."""
        partial_df = pl.DataFrame(
            {
                "Start Date": ["2024-01-01"],
                "Inflation Factor": [1.03],
            },
        )
        with pytest.raises(Exception_Validation_Input):
            spia_inflation_adjusted.calc_annuity_payout(
                100_000.0,
                obj_inflation=partial_df,
            )

    def test_inflation_factor_of_one_is_neutral(
        self,
        spia_inflation_adjusted: Annuity_SPIA,
    ) -> None:
        """An inflation factor of 1.0 leaves the payout unchanged."""
        neutral_df = pl.DataFrame(
            {
                "Start Date": ["2024-01-01"],
                "End Date": ["2025-01-01"],
                "Inflation Factor": [1.0],
            },
        )
        payout = spia_inflation_adjusted.calc_annuity_payout(
            100_000.0,
            obj_inflation=neutral_df,
        )
        assert payout == pytest.approx(6_000.0)

    def test_monthly_payout_with_inflation(
        self,
        spia_inflation_adjusted: Annuity_SPIA,
        inflation_df_single: pl.DataFrame,
    ) -> None:
        """Monthly payout propagates the inflation adjustment (annual / 12)."""
        monthly = spia_inflation_adjusted.calc_monthly_payout(
            100_000.0,
            obj_inflation=inflation_df_single,
        )
        expected_annual = 100_000.0 * 0.06 * 1.03
        assert monthly == pytest.approx(expected_annual / 12, rel=1e-9)

    def test_monthly_payout_type_accepts_polars_dataframe(
        self,
        spia_defaults: Annuity_SPIA,
        inflation_df_single: pl.DataFrame,
    ) -> None:
        """calc_monthly_payout accepts pl.DataFrame without raising."""
        monthly = spia_defaults.calc_monthly_payout(
            100_000.0,
            obj_inflation=inflation_df_single,
        )
        assert monthly == pytest.approx(500.0)  # inflation_adjusted=False


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
class Test_Annuity_SPIA_Withdrawal_Rates:
    """Tests for calc_withdrawal_rates on Annuity_SPIA.

    A plain LIFE_ONLY SPIA with payout_rate=0.06 and principal=100_000
    delivers an annual payout of 6_000, giving nominal_WR = 0.06.
    """

    def test_returns_withdrawal_rates_struct(
        self,
        spia_defaults: Annuity_SPIA,
    ) -> None:
        """calc_withdrawal_rates returns a Withdrawal_Rates instance."""
        result = spia_defaults.calc_withdrawal_rates(100_000.0, desired_WR=0.04)
        assert isinstance(result, Withdrawal_Rates)

    def test_nominal_wr_equals_payout_rate(
        self,
        spia_defaults: Annuity_SPIA,
    ) -> None:
        """Nominal WR equals the annuity payout rate for a plain LIFE_ONLY SPIA."""
        result = spia_defaults.calc_withdrawal_rates(100_000.0, desired_WR=0.04)
        assert result.nominal_WR == pytest.approx(0.06, rel=1e-9)

    def test_real_wr_equals_nominal_when_no_inflation(
        self,
        spia_defaults: Annuity_SPIA,
    ) -> None:
        """real_WR equals nominal_WR when obj_inflation is None (deflator = 1.0)."""
        result = spia_defaults.calc_withdrawal_rates(100_000.0, desired_WR=0.04)
        assert result.real_WR == pytest.approx(result.nominal_WR, rel=1e-9)

    def test_real_wr_deflated_with_inflation(
        self,
        spia_defaults: Annuity_SPIA,
        inflation_wr_df: pl.DataFrame,
    ) -> None:
        """real_WR is deflated by the inverse inflation factor (1 / 1.03)."""
        result = spia_defaults.calc_withdrawal_rates(
            100_000.0,
            desired_WR=0.04,
            obj_inflation=inflation_wr_df,
        )
        expected_real = 0.06 * (1.0 / 1.03)
        assert result.nominal_WR == pytest.approx(0.06, rel=1e-9)
        assert result.real_WR == pytest.approx(expected_real, rel=1e-9)

    def test_real_wr_less_than_nominal_when_inflation_positive(
        self,
        spia_defaults: Annuity_SPIA,
        inflation_wr_df: pl.DataFrame,
    ) -> None:
        """real_WR < nominal_WR when the inflation factor is above 1.0."""
        result = spia_defaults.calc_withdrawal_rates(
            100_000.0,
            desired_WR=0.04,
            obj_inflation=inflation_wr_df,
        )
        assert result.real_WR < result.nominal_WR

    def test_invalid_principal_zero_raises(
        self,
        spia_defaults: Annuity_SPIA,
    ) -> None:
        """Zero amount_principal raises Exception_Validation_Input."""
        with pytest.raises(Exception_Validation_Input):
            spia_defaults.calc_withdrawal_rates(0.0, desired_WR=0.04)

    def test_invalid_principal_negative_raises(
        self,
        spia_defaults: Annuity_SPIA,
    ) -> None:
        """Negative amount_principal raises Exception_Validation_Input."""
        with pytest.raises(Exception_Validation_Input):
            spia_defaults.calc_withdrawal_rates(-1.0, desired_WR=0.04)

    def test_invalid_desired_wr_zero_raises(
        self,
        spia_defaults: Annuity_SPIA,
    ) -> None:
        """Zero desired_WR raises Exception_Validation_Input."""
        with pytest.raises(Exception_Validation_Input):
            spia_defaults.calc_withdrawal_rates(100_000.0, desired_WR=0.0)

    def test_invalid_desired_wr_negative_raises(
        self,
        spia_defaults: Annuity_SPIA,
    ) -> None:
        """Negative desired_WR raises Exception_Validation_Input."""
        with pytest.raises(Exception_Validation_Input):
            spia_defaults.calc_withdrawal_rates(100_000.0, desired_WR=-0.04)

    def test_inflation_df_missing_inverse_column_raises(
        self,
        spia_defaults: Annuity_SPIA,
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
            spia_defaults.calc_withdrawal_rates(
                100_000.0,
                desired_WR=0.04,
                obj_inflation=incomplete_df,
            )


# ======================================================================
# Income starting age tests
# ======================================================================


@pytest.mark.unit()
class Test_Annuity_SPIA_Income_Starting_Age:
    """Test m_annuity_income_starting_age for Annuity_SPIA."""

    def test_default_is_client_age(self, spia_defaults: Annuity_SPIA) -> None:
        """Default income starting age equals client_age."""
        assert spia_defaults.annuity_income_starting_age == spia_defaults.client_age

    def test_custom_income_starting_age_stored(self) -> None:
        """Explicitly provided income starting age >= client_age is stored correctly."""
        spia = Annuity_SPIA(
            client_age=60,
            annuity_payout_rate=0.05,
            annuity_income_starting_age=65,
        )
        assert spia.annuity_income_starting_age == 65

    def test_income_starting_age_equal_to_client_age(self) -> None:
        """Income starting age equal to client_age is accepted."""
        spia = Annuity_SPIA(
            client_age=65,
            annuity_payout_rate=0.06,
            annuity_income_starting_age=65,
        )
        assert spia.annuity_income_starting_age == 65

    def test_income_starting_age_below_client_age_raises(self) -> None:
        """Income starting age below client_age raises Exception_Validation_Input."""
        with pytest.raises(Exception_Validation_Input):
            Annuity_SPIA(
                client_age=65,
                annuity_payout_rate=0.06,
                annuity_income_starting_age=60,
            )

    def test_member_variable_directly(self) -> None:
        """m_annuity_income_starting_age member variable equals the provided value."""
        spia = Annuity_SPIA(
            client_age=65,
            annuity_payout_rate=0.06,
            annuity_income_starting_age=70,
        )
        assert spia.m_annuity_income_starting_age == 70
