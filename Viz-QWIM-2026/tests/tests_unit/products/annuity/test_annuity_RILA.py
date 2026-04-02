"""Unit tests for Annuity_RILA class.

Tests cover initialization, property access, input validation,
downside protection (buffer and floor), crediting strategies (cap,
performance trigger, participation rate), annualised credited rate,
account value at term end, interim value, multi-term simulation,
surrender charge/value, death benefit, payout calculations,
worst/best case analysis, and string representation.

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
from src.products.annuity.annuity_RILA import (
    Annuity_RILA,
    Crediting_Strategy,
    Protection_Type,
)
from src.utils.custom_exceptions_errors_loggers.exception_custom import (
    Exception_Validation_Input,
)


# ======================================================================
# Fixtures
# ======================================================================


@pytest.fixture()
def rila_buffer_cap() -> Annuity_RILA:
    """Create an Annuity_RILA with buffer protection + cap strategy."""
    return Annuity_RILA(
        client_age=55,
        annuity_payout_rate=0.05,
        age_income_start=65,
        term_years=6,
        protection_type=Protection_Type.BUFFER,
        buffer_rate=0.10,
        cap_rate=0.15,
    )


@pytest.fixture()
def rila_floor_cap() -> Annuity_RILA:
    """Create an Annuity_RILA with floor protection + cap strategy."""
    return Annuity_RILA(
        client_age=55,
        annuity_payout_rate=0.05,
        age_income_start=65,
        term_years=6,
        protection_type=Protection_Type.FLOOR,
        floor_rate=-0.10,
        cap_rate=0.15,
    )


@pytest.fixture()
def rila_buffer_trigger() -> Annuity_RILA:
    """Create an Annuity_RILA with buffer + performance trigger strategy."""
    return Annuity_RILA(
        client_age=60,
        annuity_payout_rate=0.04,
        age_income_start=65,
        term_years=3,
        protection_type=Protection_Type.BUFFER,
        buffer_rate=0.15,
        crediting_strategy=Crediting_Strategy.PERFORMANCE_TRIGGER,
        performance_trigger_rate=0.08,
    )


@pytest.fixture()
def rila_participation() -> Annuity_RILA:
    """Create an Annuity_RILA with participation rate strategy."""
    return Annuity_RILA(
        client_age=50,
        annuity_payout_rate=0.05,
        age_income_start=65,
        term_years=6,
        protection_type=Protection_Type.BUFFER,
        buffer_rate=0.10,
        crediting_strategy=Crediting_Strategy.PARTICIPATION_RATE,
        participation_rate=1.5,
        cap_rate=0.25,
    )


@pytest.fixture()
def rila_past_income_age() -> Annuity_RILA:
    """Create a RILA and advance client past income start age.

    The constructor requires ``age_income_start > client_age``, so
    we create with valid params and manually advance client age
    to simulate time passing.
    """
    rila = Annuity_RILA(
        client_age=64,
        annuity_payout_rate=0.05,
        age_income_start=65,
        term_years=6,
    )
    # Simulate ageing past income start
    rila.m_client_age = 66
    return rila


@pytest.fixture()
def rila_custom() -> Annuity_RILA:
    """Create an Annuity_RILA with fully customised parameters."""
    return Annuity_RILA(
        client_age=55,
        annuity_payout_rate=0.05,
        age_income_start=65,
        term_years=3,
        protection_type=Protection_Type.FLOOR,
        buffer_rate=0.20,
        floor_rate=-0.15,
        crediting_strategy=Crediting_Strategy.CAP,
        cap_rate=0.20,
        participation_rate=1.0,
        performance_trigger_rate=0.0,
        rate_rider_charge=0.005,
        rate_surrender_charge_initial=0.08,
        surrender_charge_schedule_years=8,
        pct_free_withdrawal=0.15,
        rate_interim_discount=0.03,
        has_GMDB=True,
        has_return_of_premium_death_benefit=True,
        payment_frequency=4,
    )


# ======================================================================
# Enum tests
# ======================================================================


@pytest.mark.unit()
class Test_RILA_Enums:
    """Test RILA-specific enumeration types."""

    def test_protection_type_values(self) -> None:
        """Verify Protection_Type enum values."""
        assert Protection_Type.BUFFER.value == "Buffer"
        assert Protection_Type.FLOOR.value == "Floor"

    def test_crediting_strategy_values(self) -> None:
        """Verify Crediting_Strategy enum values."""
        assert Crediting_Strategy.CAP.value == "Cap"
        assert Crediting_Strategy.PERFORMANCE_TRIGGER.value == "Performance Trigger"
        assert Crediting_Strategy.PARTICIPATION_RATE.value == "Participation Rate"

    def test_annuity_type_rila_exists(self) -> None:
        """Verify ANNUITY_RILA is in the Annuity_Type enum."""
        assert Annuity_Type.ANNUITY_RILA.value == "Registered Index-Linked Annuity"


# ======================================================================
# Initialization tests
# ======================================================================


@pytest.mark.unit()
class Test_Annuity_RILA_Init:
    """Test Annuity_RILA initialization and parameter storage."""

    def test_init_buffer_cap_defaults(self, rila_buffer_cap: Annuity_RILA) -> None:
        """Verify default parameter values with buffer + cap."""
        assert rila_buffer_cap.client_age == 55
        assert rila_buffer_cap.annuity_payout_rate == pytest.approx(0.05)
        assert rila_buffer_cap.annuity_type == Annuity_Type.ANNUITY_RILA
        assert rila_buffer_cap.income_start_age == 65
        assert rila_buffer_cap.term_years == 6
        assert rila_buffer_cap.protection_type == Protection_Type.BUFFER
        assert rila_buffer_cap.buffer_rate == pytest.approx(0.10)
        assert rila_buffer_cap.cap_rate == pytest.approx(0.15)
        assert rila_buffer_cap.crediting_strategy == Crediting_Strategy.CAP
        assert rila_buffer_cap.participation_rate == pytest.approx(1.0)
        assert rila_buffer_cap.performance_trigger_rate == pytest.approx(0.0)
        assert rila_buffer_cap.rate_rider_charge == pytest.approx(0.0)
        assert rila_buffer_cap.pct_free_withdrawal == pytest.approx(0.10)
        assert rila_buffer_cap.rate_interim_discount == pytest.approx(0.02)
        assert rila_buffer_cap.has_GMDB is False
        assert rila_buffer_cap.has_return_of_premium_death_benefit is True
        assert rila_buffer_cap.payment_frequency == 12
        assert rila_buffer_cap.deferral_years == 10
        assert rila_buffer_cap.financial_index == "S&P 500"

    def test_init_floor_cap(self, rila_floor_cap: Annuity_RILA) -> None:
        """Verify floor + cap parameter storage."""
        assert rila_floor_cap.protection_type == Protection_Type.FLOOR
        assert rila_floor_cap.floor_rate == pytest.approx(-0.10)

    def test_init_performance_trigger(self, rila_buffer_trigger: Annuity_RILA) -> None:
        """Verify performance trigger strategy parameters."""
        assert rila_buffer_trigger.crediting_strategy == Crediting_Strategy.PERFORMANCE_TRIGGER
        assert rila_buffer_trigger.performance_trigger_rate == pytest.approx(0.08)
        assert rila_buffer_trigger.term_years == 3

    def test_init_participation_rate(self, rila_participation: Annuity_RILA) -> None:
        """Verify participation rate strategy parameters."""
        assert rila_participation.crediting_strategy == Crediting_Strategy.PARTICIPATION_RATE
        assert rila_participation.participation_rate == pytest.approx(1.5)
        assert rila_participation.cap_rate == pytest.approx(0.25)

    def test_init_custom_parameters(self, rila_custom: Annuity_RILA) -> None:
        """Verify fully customised parameters."""
        assert rila_custom.term_years == 3
        assert rila_custom.protection_type == Protection_Type.FLOOR
        assert rila_custom.floor_rate == pytest.approx(-0.15)
        assert rila_custom.cap_rate == pytest.approx(0.20)
        assert rila_custom.rate_rider_charge == pytest.approx(0.005)
        assert rila_custom.has_GMDB is True
        assert rila_custom.has_return_of_premium_death_benefit is True
        assert rila_custom.payment_frequency == 4
        assert rila_custom.rate_interim_discount == pytest.approx(0.03)
        assert rila_custom.pct_free_withdrawal == pytest.approx(0.15)

    def test_deferral_years(self, rila_buffer_cap: Annuity_RILA) -> None:
        """Verify deferral_years = income_start_age - client_age."""
        assert rila_buffer_cap.deferral_years == 10


# ======================================================================
# Input validation tests
# ======================================================================


@pytest.mark.unit()
class Test_Annuity_RILA_Validation:
    """Test input validation on Annuity_RILA constructor."""

    def test_invalid_client_age_zero(self) -> None:
        """Reject client_age = 0."""
        with pytest.raises(Exception_Validation_Input):
            Annuity_RILA(client_age=0, annuity_payout_rate=0.05, age_income_start=65)

    def test_invalid_client_age_negative(self) -> None:
        """Reject negative client_age."""
        with pytest.raises(Exception_Validation_Input):
            Annuity_RILA(client_age=-5, annuity_payout_rate=0.05, age_income_start=65)

    def test_invalid_payout_rate_zero(self) -> None:
        """Reject annuity_payout_rate = 0."""
        with pytest.raises(Exception_Validation_Input):
            Annuity_RILA(client_age=55, annuity_payout_rate=0, age_income_start=65)

    def test_invalid_payout_rate_negative(self) -> None:
        """Reject negative annuity_payout_rate."""
        with pytest.raises(Exception_Validation_Input):
            Annuity_RILA(client_age=55, annuity_payout_rate=-0.05, age_income_start=65)

    def test_invalid_age_income_start_less_than_client(self) -> None:
        """Reject age_income_start <= client_age."""
        with pytest.raises(Exception_Validation_Input):
            Annuity_RILA(client_age=65, annuity_payout_rate=0.05, age_income_start=60)

    def test_invalid_age_income_start_equal_client(self) -> None:
        """Reject age_income_start == client_age."""
        with pytest.raises(Exception_Validation_Input):
            Annuity_RILA(client_age=65, annuity_payout_rate=0.05, age_income_start=65)

    def test_invalid_term_years_zero(self) -> None:
        """Reject term_years = 0."""
        with pytest.raises(Exception_Validation_Input):
            Annuity_RILA(
                client_age=55,
                annuity_payout_rate=0.05,
                age_income_start=65,
                term_years=0,
            )

    def test_invalid_term_years_negative(self) -> None:
        """Reject negative term_years."""
        with pytest.raises(Exception_Validation_Input):
            Annuity_RILA(
                client_age=55,
                annuity_payout_rate=0.05,
                age_income_start=65,
                term_years=-3,
            )

    def test_invalid_protection_type(self) -> None:
        """Reject non-enum protection_type."""
        with pytest.raises(Exception_Validation_Input):
            Annuity_RILA(
                client_age=55,
                annuity_payout_rate=0.05,
                age_income_start=65,
                protection_type="Buffer",  # type: ignore[arg-type]
            )

    def test_invalid_buffer_rate_negative(self) -> None:
        """Reject negative buffer_rate."""
        with pytest.raises(Exception_Validation_Input):
            Annuity_RILA(
                client_age=55,
                annuity_payout_rate=0.05,
                age_income_start=65,
                buffer_rate=-0.10,
            )

    def test_invalid_buffer_rate_over_one(self) -> None:
        """Reject buffer_rate > 1."""
        with pytest.raises(Exception_Validation_Input):
            Annuity_RILA(
                client_age=55,
                annuity_payout_rate=0.05,
                age_income_start=65,
                buffer_rate=1.5,
            )

    def test_invalid_floor_rate_positive(self) -> None:
        """Reject positive floor_rate."""
        with pytest.raises(Exception_Validation_Input):
            Annuity_RILA(
                client_age=55,
                annuity_payout_rate=0.05,
                age_income_start=65,
                floor_rate=0.10,
            )

    def test_invalid_crediting_strategy(self) -> None:
        """Reject non-enum crediting_strategy."""
        with pytest.raises(Exception_Validation_Input):
            Annuity_RILA(
                client_age=55,
                annuity_payout_rate=0.05,
                age_income_start=65,
                crediting_strategy="Cap",  # type: ignore[arg-type]
            )

    def test_invalid_cap_rate_negative(self) -> None:
        """Reject negative cap_rate."""
        with pytest.raises(Exception_Validation_Input):
            Annuity_RILA(
                client_age=55,
                annuity_payout_rate=0.05,
                age_income_start=65,
                cap_rate=-0.01,
            )

    def test_invalid_participation_rate_negative(self) -> None:
        """Reject negative participation_rate."""
        with pytest.raises(Exception_Validation_Input):
            Annuity_RILA(
                client_age=55,
                annuity_payout_rate=0.05,
                age_income_start=65,
                participation_rate=-0.5,
            )

    def test_invalid_participation_rate_too_high(self) -> None:
        """Reject participation_rate > 3.0."""
        with pytest.raises(Exception_Validation_Input):
            Annuity_RILA(
                client_age=55,
                annuity_payout_rate=0.05,
                age_income_start=65,
                participation_rate=3.5,
            )

    def test_invalid_performance_trigger_rate_negative(self) -> None:
        """Reject negative performance_trigger_rate."""
        with pytest.raises(Exception_Validation_Input):
            Annuity_RILA(
                client_age=55,
                annuity_payout_rate=0.05,
                age_income_start=65,
                performance_trigger_rate=-0.01,
            )

    def test_invalid_rider_charge_negative(self) -> None:
        """Reject negative rate_rider_charge."""
        with pytest.raises(Exception_Validation_Input):
            Annuity_RILA(
                client_age=55,
                annuity_payout_rate=0.05,
                age_income_start=65,
                rate_rider_charge=-0.01,
            )

    def test_invalid_surrender_charge_negative(self) -> None:
        """Reject negative rate_surrender_charge_initial."""
        with pytest.raises(Exception_Validation_Input):
            Annuity_RILA(
                client_age=55,
                annuity_payout_rate=0.05,
                age_income_start=65,
                rate_surrender_charge_initial=-0.01,
            )

    def test_invalid_surrender_schedule_negative(self) -> None:
        """Reject negative surrender_charge_schedule_years."""
        with pytest.raises(Exception_Validation_Input):
            Annuity_RILA(
                client_age=55,
                annuity_payout_rate=0.05,
                age_income_start=65,
                surrender_charge_schedule_years=-1,
            )

    def test_invalid_interim_discount_negative(self) -> None:
        """Reject negative rate_interim_discount."""
        with pytest.raises(Exception_Validation_Input):
            Annuity_RILA(
                client_age=55,
                annuity_payout_rate=0.05,
                age_income_start=65,
                rate_interim_discount=-0.01,
            )

    def test_invalid_payment_frequency_zero(self) -> None:
        """Reject payment_frequency = 0."""
        with pytest.raises(Exception_Validation_Input):
            Annuity_RILA(
                client_age=55,
                annuity_payout_rate=0.05,
                age_income_start=65,
                payment_frequency=0,
            )


# ======================================================================
# Downside protection tests
# ======================================================================


@pytest.mark.unit()
class Test_Downside_Return:
    """Test calc_downside_return with buffer and floor protection."""

    # --- Buffer protection ---

    def test_buffer_positive_return(self, rila_buffer_cap: Annuity_RILA) -> None:
        """Buffer: positive index return → downside = 0."""
        assert rila_buffer_cap.calc_downside_return(0.10) == pytest.approx(0.0)

    def test_buffer_zero_return(self, rila_buffer_cap: Annuity_RILA) -> None:
        """Buffer: zero index return → downside = 0."""
        assert rila_buffer_cap.calc_downside_return(0.0) == pytest.approx(0.0)

    def test_buffer_small_loss_within_buffer(self, rila_buffer_cap: Annuity_RILA) -> None:
        """Buffer: loss within buffer range → downside = 0."""
        assert rila_buffer_cap.calc_downside_return(-0.05) == pytest.approx(0.0)

    def test_buffer_loss_at_buffer_boundary(self, rila_buffer_cap: Annuity_RILA) -> None:
        """Buffer: loss exactly at buffer boundary → downside = 0."""
        assert rila_buffer_cap.calc_downside_return(-0.10) == pytest.approx(0.0)

    def test_buffer_loss_beyond_buffer(self, rila_buffer_cap: Annuity_RILA) -> None:
        """Buffer: loss beyond buffer → downside = return + buffer."""
        # -25% loss with 10% buffer → -15%
        assert rila_buffer_cap.calc_downside_return(-0.25) == pytest.approx(-0.15)

    def test_buffer_total_loss(self, rila_buffer_cap: Annuity_RILA) -> None:
        """Buffer: total wipeout (-100%) → downside = -90%."""
        assert rila_buffer_cap.calc_downside_return(-1.0) == pytest.approx(-0.90)

    # --- Floor protection ---

    def test_floor_positive_return(self, rila_floor_cap: Annuity_RILA) -> None:
        """Floor: positive index return → downside = 0."""
        assert rila_floor_cap.calc_downside_return(0.10) == pytest.approx(0.0)

    def test_floor_small_loss_above_floor(self, rila_floor_cap: Annuity_RILA) -> None:
        """Floor: loss less severe than floor → actual loss passed through."""
        assert rila_floor_cap.calc_downside_return(-0.05) == pytest.approx(-0.05)

    def test_floor_loss_at_floor(self, rila_floor_cap: Annuity_RILA) -> None:
        """Floor: loss exactly at floor → floor_rate."""
        assert rila_floor_cap.calc_downside_return(-0.10) == pytest.approx(-0.10)

    def test_floor_loss_beyond_floor(self, rila_floor_cap: Annuity_RILA) -> None:
        """Floor: loss beyond floor → clamped to floor_rate."""
        assert rila_floor_cap.calc_downside_return(-0.50) == pytest.approx(-0.10)

    def test_floor_total_loss(self, rila_floor_cap: Annuity_RILA) -> None:
        """Floor: total wipeout → clamped to floor_rate."""
        assert rila_floor_cap.calc_downside_return(-1.0) == pytest.approx(-0.10)

    # --- Validation ---

    def test_invalid_index_return_type(self, rila_buffer_cap: Annuity_RILA) -> None:
        """Reject non-numeric index_return."""
        with pytest.raises(Exception_Validation_Input):
            rila_buffer_cap.calc_downside_return("bad")  # type: ignore[arg-type]


# ======================================================================
# Credited rate tests
# ======================================================================


@pytest.mark.unit()
class Test_Credited_Rate:
    """Test calc_credited_rate for all crediting strategies."""

    # --- Cap strategy with buffer ---

    def test_cap_positive_below_cap(self, rila_buffer_cap: Annuity_RILA) -> None:
        """Cap strategy: positive return below cap → return as-is."""
        assert rila_buffer_cap.calc_credited_rate(0.10) == pytest.approx(0.10)

    def test_cap_positive_above_cap(self, rila_buffer_cap: Annuity_RILA) -> None:
        """Cap strategy: positive return above cap → capped."""
        assert rila_buffer_cap.calc_credited_rate(0.25) == pytest.approx(0.15)

    def test_cap_positive_at_cap(self, rila_buffer_cap: Annuity_RILA) -> None:
        """Cap strategy: return exactly at cap."""
        assert rila_buffer_cap.calc_credited_rate(0.15) == pytest.approx(0.15)

    def test_cap_zero_return(self, rila_buffer_cap: Annuity_RILA) -> None:
        """Cap strategy: zero return → 0."""
        assert rila_buffer_cap.calc_credited_rate(0.0) == pytest.approx(0.0)

    def test_cap_loss_within_buffer(self, rila_buffer_cap: Annuity_RILA) -> None:
        """Cap strategy: loss within buffer → 0."""
        assert rila_buffer_cap.calc_credited_rate(-0.08) == pytest.approx(0.0)

    def test_cap_loss_beyond_buffer(self, rila_buffer_cap: Annuity_RILA) -> None:
        """Cap strategy: loss beyond buffer → loss + buffer."""
        assert rila_buffer_cap.calc_credited_rate(-0.30) == pytest.approx(-0.20)

    # --- Cap strategy with floor ---

    def test_cap_floor_positive(self, rila_floor_cap: Annuity_RILA) -> None:
        """Cap + floor: positive return below cap → return as-is."""
        assert rila_floor_cap.calc_credited_rate(0.10) == pytest.approx(0.10)

    def test_cap_floor_loss_above_floor(self, rila_floor_cap: Annuity_RILA) -> None:
        """Cap + floor: loss above (less than) floor → actual loss."""
        assert rila_floor_cap.calc_credited_rate(-0.05) == pytest.approx(-0.05)

    def test_cap_floor_loss_beyond_floor(self, rila_floor_cap: Annuity_RILA) -> None:
        """Cap + floor: loss beyond floor → clamped to floor."""
        assert rila_floor_cap.calc_credited_rate(-0.50) == pytest.approx(-0.10)

    # --- Performance trigger strategy ---

    def test_trigger_positive_return(self, rila_buffer_trigger: Annuity_RILA) -> None:
        """Trigger: any positive return → trigger rate."""
        assert rila_buffer_trigger.calc_credited_rate(0.01) == pytest.approx(0.08)

    def test_trigger_large_positive(self, rila_buffer_trigger: Annuity_RILA) -> None:
        """Trigger: large return → still trigger rate."""
        assert rila_buffer_trigger.calc_credited_rate(0.40) == pytest.approx(0.08)

    def test_trigger_zero_return(self, rila_buffer_trigger: Annuity_RILA) -> None:
        """Trigger: zero return → trigger rate (non-negative)."""
        assert rila_buffer_trigger.calc_credited_rate(0.0) == pytest.approx(0.08)

    def test_trigger_negative_within_buffer(self, rila_buffer_trigger: Annuity_RILA) -> None:
        """Trigger: negative within buffer → 0."""
        assert rila_buffer_trigger.calc_credited_rate(-0.10) == pytest.approx(0.0)

    def test_trigger_negative_beyond_buffer(self, rila_buffer_trigger: Annuity_RILA) -> None:
        """Trigger: negative beyond buffer → loss + buffer."""
        # -25% with 15% buffer → -10%
        assert rila_buffer_trigger.calc_credited_rate(-0.25) == pytest.approx(-0.10)

    # --- Participation rate strategy ---

    def test_participation_positive_below_cap(
        self,
        rila_participation: Annuity_RILA,
    ) -> None:
        """Participation: return * participation_rate < cap → participated return."""
        # 10% * 1.5 = 15% < 25% cap
        assert rila_participation.calc_credited_rate(0.10) == pytest.approx(0.15)

    def test_participation_positive_above_cap(
        self,
        rila_participation: Annuity_RILA,
    ) -> None:
        """Participation: return * participation_rate > cap → capped."""
        # 20% * 1.5 = 30% > 25% cap → 25%
        assert rila_participation.calc_credited_rate(0.20) == pytest.approx(0.25)

    def test_participation_negative(self, rila_participation: Annuity_RILA) -> None:
        """Participation: negative return uses downside, not participation."""
        # -5% with 10% buffer → 0
        assert rila_participation.calc_credited_rate(-0.05) == pytest.approx(0.0)

    # --- Validation ---

    def test_invalid_index_return_type(self, rila_buffer_cap: Annuity_RILA) -> None:
        """Reject non-numeric index_return."""
        with pytest.raises(Exception_Validation_Input):
            rila_buffer_cap.calc_credited_rate("bad")  # type: ignore[arg-type]


# ======================================================================
# Annualised credited rate tests
# ======================================================================


@pytest.mark.unit()
class Test_Annualised_Credited_Rate:
    """Test calc_annualised_credited_rate."""

    def test_annualised_positive(self, rila_buffer_cap: Annuity_RILA) -> None:
        """Annualise a 15% 6-year credited rate."""
        # (1 + 0.15)^(1/6) - 1 ≈ 0.02357
        result = rila_buffer_cap.calc_annualised_credited_rate(0.20)
        # 20% return capped at 15%, then annualised over 6 years
        expected = (1.15) ** (1.0 / 6) - 1
        assert result == pytest.approx(expected, rel=1e-6)

    def test_annualised_zero(self, rila_buffer_cap: Annuity_RILA) -> None:
        """Annualise a 0% credited rate."""
        result = rila_buffer_cap.calc_annualised_credited_rate(0.0)
        assert result == pytest.approx(0.0, abs=1e-10)

    def test_annualised_negative(self, rila_buffer_cap: Annuity_RILA) -> None:
        """Annualise a negative credited rate (loss beyond buffer)."""
        # -30% index, 10% buffer → -20% credited
        result = rila_buffer_cap.calc_annualised_credited_rate(-0.30)
        expected = (1 - 0.20) ** (1.0 / 6) - 1
        assert result == pytest.approx(expected, rel=1e-6)

    def test_annualised_custom_term(self, rila_buffer_cap: Annuity_RILA) -> None:
        """Annualise with a custom term_years override."""
        result = rila_buffer_cap.calc_annualised_credited_rate(0.10, term_years=3)
        expected = (1.10) ** (1.0 / 3) - 1
        assert result == pytest.approx(expected, rel=1e-6)

    def test_annualised_invalid_term_years(self, rila_buffer_cap: Annuity_RILA) -> None:
        """Reject invalid term_years."""
        with pytest.raises(Exception_Validation_Input):
            rila_buffer_cap.calc_annualised_credited_rate(0.10, term_years=0)


# ======================================================================
# Account value at term end tests
# ======================================================================


@pytest.mark.unit()
class Test_Account_Value_At_Term_End:
    """Test calc_account_value_at_term_end."""

    def test_positive_return(self, rila_buffer_cap: Annuity_RILA) -> None:
        """Term end value with positive index return."""
        # 10% return, under 15% cap → 100000 * 1.10
        result = rila_buffer_cap.calc_account_value_at_term_end(100000, 0.10)
        assert result == pytest.approx(110000.0)

    def test_capped_return(self, rila_buffer_cap: Annuity_RILA) -> None:
        """Term end value with return exceeding cap."""
        # 30% return, capped at 15% → 100000 * 1.15
        result = rila_buffer_cap.calc_account_value_at_term_end(100000, 0.30)
        assert result == pytest.approx(115000.0)

    def test_loss_within_buffer(self, rila_buffer_cap: Annuity_RILA) -> None:
        """Term end value with loss within buffer."""
        # -5% return, 10% buffer → 0% credited → 100000
        result = rila_buffer_cap.calc_account_value_at_term_end(100000, -0.05)
        assert result == pytest.approx(100000.0)

    def test_loss_beyond_buffer(self, rila_buffer_cap: Annuity_RILA) -> None:
        """Term end value with loss beyond buffer."""
        # -25% return, 10% buffer → -15% credited → 85000
        result = rila_buffer_cap.calc_account_value_at_term_end(100000, -0.25)
        assert result == pytest.approx(85000.0)

    def test_invalid_principal(self, rila_buffer_cap: Annuity_RILA) -> None:
        """Reject non-positive principal."""
        with pytest.raises(Exception_Validation_Input):
            rila_buffer_cap.calc_account_value_at_term_end(0, 0.10)

        with pytest.raises(Exception_Validation_Input):
            rila_buffer_cap.calc_account_value_at_term_end(-1000, 0.10)


# ======================================================================
# Interim value tests
# ======================================================================


@pytest.mark.unit()
class Test_Interim_Value:
    """Test calc_interim_value."""

    def test_interim_at_term_start(self, rila_buffer_cap: Annuity_RILA) -> None:
        """Interim value at year 0 = discounted for full term."""
        # 100000 * (1 + 0) * 1/(1.02)^6
        result = rila_buffer_cap.calc_interim_value(100000, 0.0, 0)
        expected = 100000 / ((1.02) ** 6)
        assert result == pytest.approx(expected, rel=1e-6)

    def test_interim_at_term_end(self, rila_buffer_cap: Annuity_RILA) -> None:
        """Interim value at term end = no discount (remaining_years = 0)."""
        # years_elapsed = 6 (= term_years) → discount factor = 1.0
        result = rila_buffer_cap.calc_interim_value(100000, 0.10, 6)
        expected = 100000 * 1.10 * 1.0  # No discount
        assert result == pytest.approx(expected, rel=1e-6)

    def test_interim_mid_term(self, rila_buffer_cap: Annuity_RILA) -> None:
        """Interim value at mid-term with positive return."""
        # years_elapsed = 3, remaining = 3
        result = rila_buffer_cap.calc_interim_value(100000, 0.08, 3)
        expected = 100000 * 1.08 / ((1.02) ** 3)
        assert result == pytest.approx(expected, rel=1e-6)

    def test_interim_with_loss(self, rila_buffer_cap: Annuity_RILA) -> None:
        """Interim value with negative index return within buffer."""
        # -5% within 10% buffer → credited = 0 → discounted
        result = rila_buffer_cap.calc_interim_value(100000, -0.05, 2)
        expected = 100000 * 1.0 / ((1.02) ** 4)
        assert result == pytest.approx(expected, rel=1e-6)

    def test_interim_invalid_years_too_large(self, rila_buffer_cap: Annuity_RILA) -> None:
        """Reject years_elapsed > term_years."""
        with pytest.raises(Exception_Validation_Input):
            rila_buffer_cap.calc_interim_value(100000, 0.05, 10)

    def test_interim_invalid_negative_years(self, rila_buffer_cap: Annuity_RILA) -> None:
        """Reject negative years_elapsed."""
        with pytest.raises(Exception_Validation_Input):
            rila_buffer_cap.calc_interim_value(100000, 0.05, -1)

    def test_interim_invalid_principal(self, rila_buffer_cap: Annuity_RILA) -> None:
        """Reject non-positive principal."""
        with pytest.raises(Exception_Validation_Input):
            rila_buffer_cap.calc_interim_value(0, 0.05, 2)


# ======================================================================
# Multi-term account value tests
# ======================================================================


@pytest.mark.unit()
class Test_Account_Values_Multi_Term:
    """Test calc_account_values_multi_term."""

    def test_multi_term_positive_returns(self, rila_buffer_cap: Annuity_RILA) -> None:
        """Multiple positive term returns → compounded growth."""
        returns = [0.10, 0.12, 0.08]
        values = rila_buffer_cap.calc_account_values_multi_term(100000, returns)

        assert len(values) == 4  # initial + 3 terms
        assert values[0] == pytest.approx(100000.0)
        # No rider charge → charge_factor = 1.0
        assert values[1] == pytest.approx(100000 * 1.10)
        assert values[2] == pytest.approx(100000 * 1.10 * 1.12)
        assert values[3] == pytest.approx(100000 * 1.10 * 1.12 * 1.08)

    def test_multi_term_with_losses(self, rila_buffer_cap: Annuity_RILA) -> None:
        """Mixed returns including losses within and beyond buffer."""
        returns = [0.10, -0.05, -0.25]
        values = rila_buffer_cap.calc_account_values_multi_term(100000, returns)

        assert len(values) == 4
        assert values[1] == pytest.approx(110000.0)  # +10%
        assert values[2] == pytest.approx(110000.0)  # -5% within 10% buffer → 0%
        assert values[3] == pytest.approx(110000.0 * 0.85)  # -25% → -15%

    def test_multi_term_with_rider_charge(self, rila_custom: Annuity_RILA) -> None:
        """Multi-term with rider charges deducted per term."""
        returns = [0.10]
        values = rila_custom.calc_account_values_multi_term(100000, returns)

        # rila_custom has rider_charge=0.005, term=3
        charge_factor = (1 - 0.005) ** 3
        expected = 100000 * 1.10 * charge_factor
        assert values[1] == pytest.approx(expected, rel=1e-6)

    def test_multi_term_empty_returns(self, rila_buffer_cap: Annuity_RILA) -> None:
        """Empty returns list → only initial value."""
        values = rila_buffer_cap.calc_account_values_multi_term(100000, [])
        assert values == [100000.0]

    def test_multi_term_capped_returns(self, rila_buffer_cap: Annuity_RILA) -> None:
        """Returns above cap are capped each term."""
        returns = [0.50, 0.50]
        values = rila_buffer_cap.calc_account_values_multi_term(100000, returns)
        # Both capped at 15%
        assert values[1] == pytest.approx(115000.0)
        assert values[2] == pytest.approx(115000.0 * 1.15)

    def test_multi_term_invalid_principal(self, rila_buffer_cap: Annuity_RILA) -> None:
        """Reject non-positive principal."""
        with pytest.raises(Exception_Validation_Input):
            rila_buffer_cap.calc_account_values_multi_term(0, [0.10])

    def test_multi_term_invalid_returns_type(self, rila_buffer_cap: Annuity_RILA) -> None:
        """Reject non-list term_index_returns."""
        with pytest.raises(Exception_Validation_Input):
            rila_buffer_cap.calc_account_values_multi_term(100000, 0.10)  # type: ignore[arg-type]


# ======================================================================
# Surrender charge and value tests
# ======================================================================


@pytest.mark.unit()
class Test_Surrender_Calculations:
    """Test surrender charge and value calculations."""

    def test_surrender_charge_year_0(self, rila_buffer_cap: Annuity_RILA) -> None:
        """Year 0: full initial charge rate."""
        assert rila_buffer_cap.calc_surrender_charge_rate(0) == pytest.approx(0.06)

    def test_surrender_charge_mid_schedule(self, rila_buffer_cap: Annuity_RILA) -> None:
        """Mid-schedule: linearly declining."""
        # Year 3 with 6-year schedule: 0.06 * (1 - 3/6) = 0.03
        assert rila_buffer_cap.calc_surrender_charge_rate(3) == pytest.approx(0.03)

    def test_surrender_charge_past_schedule(self, rila_buffer_cap: Annuity_RILA) -> None:
        """Past schedule: zero charge."""
        assert rila_buffer_cap.calc_surrender_charge_rate(6) == pytest.approx(0.0)
        assert rila_buffer_cap.calc_surrender_charge_rate(10) == pytest.approx(0.0)

    def test_surrender_charge_negative_year(self, rila_buffer_cap: Annuity_RILA) -> None:
        """Negative year: returns 0 (graceful)."""
        assert rila_buffer_cap.calc_surrender_charge_rate(-1) == pytest.approx(0.0)

    def test_surrender_value_year_0(self, rila_buffer_cap: Annuity_RILA) -> None:
        """Surrender value at year 0: full charge on excess."""
        # 100000, 10% free withdrawal = 10000 free
        # 90000 excess * 6% charge = 5400
        # Surrender value = 100000 - 5400 = 94600
        result = rila_buffer_cap.calc_surrender_value(100000, 0)
        assert result == pytest.approx(94600.0)

    def test_surrender_value_past_schedule(self, rila_buffer_cap: Annuity_RILA) -> None:
        """Surrender value past schedule: full value returned."""
        result = rila_buffer_cap.calc_surrender_value(100000, 10)
        assert result == pytest.approx(100000.0)

    def test_surrender_value_invalid_account(self, rila_buffer_cap: Annuity_RILA) -> None:
        """Reject non-positive account_value."""
        with pytest.raises(Exception_Validation_Input):
            rila_buffer_cap.calc_surrender_value(0, 3)


# ======================================================================
# Death benefit tests
# ======================================================================


@pytest.mark.unit()
class Test_Death_Benefit:
    """Test calc_death_benefit."""

    def test_rop_db_account_above_premium(self, rila_buffer_cap: Annuity_RILA) -> None:
        """ROP-DB: account > premium → account value."""
        result = rila_buffer_cap.calc_death_benefit(100000, 120000)
        assert result == pytest.approx(120000.0)

    def test_rop_db_account_below_premium(self, rila_buffer_cap: Annuity_RILA) -> None:
        """ROP-DB: account < premium → premium (return-of-premium)."""
        result = rila_buffer_cap.calc_death_benefit(100000, 80000)
        assert result == pytest.approx(100000.0)

    def test_no_death_benefit(self) -> None:
        """No ROP-DB and no GMDB → account value only."""
        rila = Annuity_RILA(
            client_age=55,
            annuity_payout_rate=0.05,
            age_income_start=65,
            has_GMDB=False,
            has_return_of_premium_death_benefit=False,
        )
        result = rila.calc_death_benefit(100000, 70000)
        assert result == pytest.approx(70000.0)

    def test_gmdb_account_below_premium(self, rila_custom: Annuity_RILA) -> None:
        """GMDB: account < premium → premium guaranteed."""
        # rila_custom has has_GMDB=True and has_return_of_premium_death_benefit=True
        result = rila_custom.calc_death_benefit(100000, 50000)
        assert result == pytest.approx(100000.0)

    def test_death_benefit_invalid_principal(self, rila_buffer_cap: Annuity_RILA) -> None:
        """Reject non-positive principal."""
        with pytest.raises(Exception_Validation_Input):
            rila_buffer_cap.calc_death_benefit(0, 100000)

    def test_death_benefit_invalid_account_value(
        self,
        rila_buffer_cap: Annuity_RILA,
    ) -> None:
        """Reject negative account_value."""
        with pytest.raises(Exception_Validation_Input):
            rila_buffer_cap.calc_death_benefit(100000, -10000)


# ======================================================================
# Payout tests
# ======================================================================


@pytest.mark.unit()
class Test_Payout_Calculations:
    """Test calc_annuity_payout and calc_monthly_payout."""

    def test_payout_during_deferral(self, rila_buffer_cap: Annuity_RILA) -> None:
        """Payout during deferral period → 0."""
        assert rila_buffer_cap.calc_annuity_payout(100000) == pytest.approx(0.0)

    def test_payout_active_no_scenarios(
        self,
        rila_past_income_age: Annuity_RILA,
    ) -> None:
        """Payout past income age without scenarios → principal * payout_rate."""
        result = rila_past_income_age.calc_annuity_payout(100000)
        assert result == pytest.approx(100000 * 0.05)

    def test_payout_active_with_scenarios(
        self,
        rila_past_income_age: Annuity_RILA,
    ) -> None:
        """Payout past income age with scenarios uses simulated account value."""
        scenarios_df = pl.DataFrame(
            {"Date": ["2024-01-01"], "S&P 500": [0.10]},
        )
        result = rila_past_income_age.calc_annuity_payout(
            100000,
            obj_scenarios=scenarios_df,
        )
        # account value after 1 term: 100000 * 1.10 = 110000
        # payout = 110000 * 0.05 = 5500
        assert result == pytest.approx(5500.0)

    def test_payout_active_scenarios_multi_term(
        self,
        rila_past_income_age: Annuity_RILA,
    ) -> None:
        """Payout with multi-term scenario data."""
        scenarios_df = pl.DataFrame(
            {"Date": ["2024-01-01", "2024-07-01"], "S&P 500": [0.10, 0.12]},
        )
        result = rila_past_income_age.calc_annuity_payout(
            100000,
            obj_scenarios=scenarios_df,
        )
        # 100000 * 1.10 * 1.12 = 123200
        # payout = 123200 * 0.05 = 6160
        assert result == pytest.approx(6160.0)

    def test_payout_active_scenarios_with_loss(
        self,
        rila_past_income_age: Annuity_RILA,
    ) -> None:
        """Payout with scenario including loss within buffer."""
        scenarios_df = pl.DataFrame(
            {"Date": ["2024-01-01"], "S&P 500": [-0.05]},
        )
        result = rila_past_income_age.calc_annuity_payout(
            100000,
            obj_scenarios=scenarios_df,
        )
        # 100000 * 1.0 = 100000 → payout = 100000 * 0.05 = 5000
        assert result == pytest.approx(5000.0)

    def test_payout_active_scenarios_missing_date_column(
        self,
        rila_past_income_age: Annuity_RILA,
    ) -> None:
        """Missing 'Date' column raises Exception_Validation_Input."""
        bad_df = pl.DataFrame({"S&P 500": [0.10]})
        with pytest.raises(Exception_Validation_Input):
            rila_past_income_age.calc_annuity_payout(100000, obj_scenarios=bad_df)

    def test_payout_active_scenarios_missing_index_column(
        self,
        rila_past_income_age: Annuity_RILA,
    ) -> None:
        """Missing financial_index column raises Exception_Validation_Input."""
        bad_df = pl.DataFrame({"Date": ["2024-01-01"], "NASDAQ": [0.10]})
        with pytest.raises(Exception_Validation_Input):
            rila_past_income_age.calc_annuity_payout(100000, obj_scenarios=bad_df)

    def test_monthly_payout_deferred(self, rila_buffer_cap: Annuity_RILA) -> None:
        """Monthly payout during deferral → 0."""
        assert rila_buffer_cap.calc_monthly_payout(100000) == pytest.approx(0.0)

    def test_monthly_payout_active(self, rila_past_income_age: Annuity_RILA) -> None:
        """Monthly payout = annual payout / 12."""
        annual = rila_past_income_age.calc_annuity_payout(100000)
        monthly = rila_past_income_age.calc_monthly_payout(100000)
        assert monthly == pytest.approx(annual / 12)
        assert monthly == pytest.approx(5000.0 / 12)

    def test_payout_invalid_principal(self, rila_buffer_cap: Annuity_RILA) -> None:
        """Reject non-positive principal."""
        with pytest.raises(Exception_Validation_Input):
            rila_buffer_cap.calc_annuity_payout(0)

    def test_payout_negative_principal(self, rila_buffer_cap: Annuity_RILA) -> None:
        """Reject negative principal."""
        with pytest.raises(Exception_Validation_Input):
            rila_buffer_cap.calc_annuity_payout(-100000)


# ======================================================================
# Worst-case and best-case tests
# ======================================================================


@pytest.mark.unit()
class Test_Worst_Best_Case:
    """Test worst-case and best-case account value analysis."""

    def test_worst_case_buffer_single_term(self, rila_buffer_cap: Annuity_RILA) -> None:
        """Worst case (buffer, 1 term): total wipeout beyond buffer."""
        # -100% → credited = -90% → 100000 * 0.10
        result = rila_buffer_cap.calc_worst_case_account_value(100000, 1)
        assert result == pytest.approx(10000.0)

    def test_worst_case_floor_single_term(self, rila_floor_cap: Annuity_RILA) -> None:
        """Worst case (floor, 1 term): loss capped at floor."""
        # floor = -10% → 100000 * 0.90
        result = rila_floor_cap.calc_worst_case_account_value(100000, 1)
        assert result == pytest.approx(90000.0)

    def test_worst_case_floor_multi_term(self, rila_floor_cap: Annuity_RILA) -> None:
        """Worst case (floor, 3 terms): compounded floor loss."""
        result = rila_floor_cap.calc_worst_case_account_value(100000, 3)
        expected = 100000 * (0.90**3)
        assert result == pytest.approx(expected, rel=1e-6)

    def test_best_case_cap_single_term(self, rila_buffer_cap: Annuity_RILA) -> None:
        """Best case (cap, 1 term): maximum cap gain."""
        result = rila_buffer_cap.calc_best_case_account_value(100000, 1)
        assert result == pytest.approx(115000.0)

    def test_best_case_cap_multi_term(self, rila_buffer_cap: Annuity_RILA) -> None:
        """Best case (cap, 3 terms): compounded cap gains."""
        result = rila_buffer_cap.calc_best_case_account_value(100000, 3)
        expected = 100000 * (1.15**3)
        assert result == pytest.approx(expected, rel=1e-6)

    def test_best_case_trigger(self, rila_buffer_trigger: Annuity_RILA) -> None:
        """Best case (trigger): compounded trigger rate."""
        result = rila_buffer_trigger.calc_best_case_account_value(100000, 2)
        expected = 100000 * (1.08**2)
        assert result == pytest.approx(expected, rel=1e-6)

    def test_best_case_participation(self, rila_participation: Annuity_RILA) -> None:
        """Best case (participation): min(participation * 100%, cap)."""
        # participation 1.5 * 1.0 = 1.5 > cap 0.25 → capped at 0.25
        result = rila_participation.calc_best_case_account_value(100000, 1)
        assert result == pytest.approx(125000.0)

    def test_worst_case_invalid_principal(self, rila_buffer_cap: Annuity_RILA) -> None:
        """Reject non-positive principal."""
        with pytest.raises(Exception_Validation_Input):
            rila_buffer_cap.calc_worst_case_account_value(0)

    def test_worst_case_invalid_num_terms(self, rila_buffer_cap: Annuity_RILA) -> None:
        """Reject non-positive num_terms."""
        with pytest.raises(Exception_Validation_Input):
            rila_buffer_cap.calc_worst_case_account_value(100000, 0)

    def test_best_case_invalid_principal(self, rila_buffer_cap: Annuity_RILA) -> None:
        """Reject non-positive principal."""
        with pytest.raises(Exception_Validation_Input):
            rila_buffer_cap.calc_best_case_account_value(0)

    def test_best_case_invalid_num_terms(self, rila_buffer_cap: Annuity_RILA) -> None:
        """Reject non-positive num_terms."""
        with pytest.raises(Exception_Validation_Input):
            rila_buffer_cap.calc_best_case_account_value(100000, -1)


# ======================================================================
# String representation tests
# ======================================================================


@pytest.mark.unit()
class Test_String_Representation:
    """Test get_annuity_as_string output."""

    def test_string_buffer_cap(self, rila_buffer_cap: Annuity_RILA) -> None:
        """String for buffer + cap RILA."""
        result = rila_buffer_cap.get_annuity_as_string()
        assert "RILA" in result
        assert "Buffer 10%" in result
        assert "Cap 15%" in result
        assert "ROP-DB" in result
        assert "Term 6yr" in result

    def test_string_floor_cap(self, rila_floor_cap: Annuity_RILA) -> None:
        """String for floor + cap RILA."""
        result = rila_floor_cap.get_annuity_as_string()
        assert "Floor -10%" in result
        assert "Cap 15%" in result

    def test_string_performance_trigger(self, rila_buffer_trigger: Annuity_RILA) -> None:
        """String for performance trigger RILA."""
        result = rila_buffer_trigger.get_annuity_as_string()
        assert "Performance Trigger 8%" in result

    def test_string_participation(self, rila_participation: Annuity_RILA) -> None:
        """String for participation rate RILA."""
        result = rila_participation.get_annuity_as_string()
        assert "Participation Rate 150%" in result
        assert "cap 25%" in result

    def test_string_with_gmdb(self, rila_custom: Annuity_RILA) -> None:
        """String includes GMDB rider when elected."""
        result = rila_custom.get_annuity_as_string()
        assert "GMDB" in result
        assert "ROP-DB" in result

    def test_string_no_riders(self) -> None:
        """String with no riders elected."""
        rila = Annuity_RILA(
            client_age=55,
            annuity_payout_rate=0.05,
            age_income_start=65,
            has_GMDB=False,
            has_return_of_premium_death_benefit=False,
        )
        result = rila.get_annuity_as_string()
        assert "Riders: None" in result


# ======================================================================
# Parametrised edge-case tests
# ======================================================================


@pytest.mark.unit()
class Test_Parametrised_Edge_Cases:
    """Parametrised tests for edge cases and boundary conditions."""

    @pytest.mark.parametrize(
        ("index_return", "expected_credited"),
        [
            (0.0, 0.0),  # Exactly zero
            (0.001, 0.001),  # Very small positive
            (-0.001, 0.0),  # Very small negative (within buffer)
            (0.15, 0.15),  # Exactly at cap
            (-0.10, 0.0),  # Exactly at buffer boundary
            (-0.10001, -0.00001),  # Just past buffer
        ],
    )
    def test_credited_rate_boundary_values(
        self,
        rila_buffer_cap: Annuity_RILA,
        index_return: float,
        expected_credited: float,
    ) -> None:
        """Test credited rate at various boundary points."""
        result = rila_buffer_cap.calc_credited_rate(index_return)
        assert result == pytest.approx(expected_credited, abs=1e-8)

    @pytest.mark.parametrize(
        ("year", "expected_rate"),
        [
            (0, 0.06),
            (1, 0.05),
            (2, 0.04),
            (3, 0.03),
            (4, 0.02),
            (5, 0.01),
            (6, 0.0),  # Schedule expired
            (7, 0.0),
            (100, 0.0),
        ],
    )
    def test_surrender_charge_schedule(
        self,
        rila_buffer_cap: Annuity_RILA,
        year: int,
        expected_rate: float,
    ) -> None:
        """Verify linearly declining surrender charge schedule."""
        result = rila_buffer_cap.calc_surrender_charge_rate(year)
        assert result == pytest.approx(expected_rate, abs=1e-10)

    @pytest.mark.parametrize("term_years", [1, 2, 3, 6])
    def test_common_term_lengths(self, term_years: int) -> None:
        """Verify RILA can be created with common term lengths."""
        rila = Annuity_RILA(
            client_age=55,
            annuity_payout_rate=0.05,
            age_income_start=65,
            term_years=term_years,
        )
        assert rila.term_years == term_years

    @pytest.mark.parametrize("buffer_rate", [0.0, 0.10, 0.15, 0.20, 0.25, 1.0])
    def test_various_buffer_rates(self, buffer_rate: float) -> None:
        """Verify RILA accepts various valid buffer rates."""
        rila = Annuity_RILA(
            client_age=55,
            annuity_payout_rate=0.05,
            age_income_start=65,
            buffer_rate=buffer_rate,
        )
        assert rila.buffer_rate == pytest.approx(buffer_rate)

    @pytest.mark.parametrize("floor_rate", [0.0, -0.05, -0.10, -0.20, -1.0])
    def test_various_floor_rates(self, floor_rate: float) -> None:
        """Verify RILA accepts various valid floor rates."""
        rila = Annuity_RILA(
            client_age=55,
            annuity_payout_rate=0.05,
            age_income_start=65,
            protection_type=Protection_Type.FLOOR,
            floor_rate=floor_rate,
        )
        assert rila.floor_rate == pytest.approx(floor_rate)


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
            "Inflation Factor": [1.02],
        },
    )


@pytest.mark.unit()
class Test_Annuity_RILA_Inflation_Parameter:
    """Verify calc_annuity_payout and calc_monthly_payout accept pl.DataFrame.

    For RILAs the ``obj_inflation`` parameter is unused; it exists for
    interface compatibility.  Tests confirm that passing ``None`` or a valid
    Polars DataFrame does not raise and payout is ``0.0`` during deferral.
    """

    def test_calc_annuity_payout_accepts_none(
        self,
        rila_buffer_cap: Annuity_RILA,
    ) -> None:
        """None is accepted as obj_inflation; returns 0.0 during deferral."""
        payout = rila_buffer_cap.calc_annuity_payout(100_000.0, obj_inflation=None)
        assert payout == pytest.approx(0.0)

    def test_calc_annuity_payout_accepts_polars_dataframe(
        self,
        rila_buffer_cap: Annuity_RILA,
        inflation_df: pl.DataFrame,
    ) -> None:
        """pl.DataFrame accepted as obj_inflation; RILA ignores it (returns 0.0)."""
        payout = rila_buffer_cap.calc_annuity_payout(
            100_000.0,
            obj_inflation=inflation_df,
        )
        assert payout == pytest.approx(0.0)  # still in deferral

    def test_calc_monthly_payout_accepts_polars_dataframe(
        self,
        rila_buffer_cap: Annuity_RILA,
        inflation_df: pl.DataFrame,
    ) -> None:
        """calc_monthly_payout accepts pl.DataFrame as obj_inflation."""
        monthly = rila_buffer_cap.calc_monthly_payout(
            100_000.0,
            obj_inflation=inflation_df,
        )
        assert monthly == pytest.approx(0.0)  # still in deferral

    def test_calc_annuity_payout_past_income_age_accepts_df(
        self,
        rila_past_income_age: Annuity_RILA,
        inflation_df: pl.DataFrame,
    ) -> None:
        """Past income start age, pl.DataFrame obj_inflation is accepted without error."""
        payout = rila_past_income_age.calc_annuity_payout(
            100_000.0,
            obj_inflation=inflation_df,
        )
        # Inflation parameter is unused for RILA; payout is based on benefit base
        assert payout > 0.0


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
class Test_Annuity_RILA_Withdrawal_Rates:
    """Tests for calc_withdrawal_rates on Annuity_RILA.

    ``rila_buffer_cap`` is in deferral (client_age=55, income_start=65), so
    nominal_WR = 0.0 during deferral.  ``rila_past_income_age`` (m_client_age=66)
    produces a positive nominal_WR based on the rolled-up benefit base.
    Note: obj_inflation is unused by RILA's calc_annuity_payout but the
    inverse factor is still applied to real_WR in calc_withdrawal_rates.
    """

    def test_returns_withdrawal_rates_struct(
        self,
        rila_buffer_cap: Annuity_RILA,
    ) -> None:
        """calc_withdrawal_rates returns a Withdrawal_Rates instance."""
        result = rila_buffer_cap.calc_withdrawal_rates(100_000.0, desired_WR=0.04)
        assert isinstance(result, Withdrawal_Rates)

    def test_nominal_wr_zero_during_deferral(
        self,
        rila_buffer_cap: Annuity_RILA,
    ) -> None:
        """nominal_WR is 0.0 while client is in the deferral period."""
        result = rila_buffer_cap.calc_withdrawal_rates(100_000.0, desired_WR=0.04)
        assert result.nominal_WR == pytest.approx(0.0)

    def test_real_wr_zero_during_deferral(
        self,
        rila_buffer_cap: Annuity_RILA,
    ) -> None:
        """real_WR is 0.0 while client is in the deferral period."""
        result = rila_buffer_cap.calc_withdrawal_rates(100_000.0, desired_WR=0.04)
        assert result.real_WR == pytest.approx(0.0)

    def test_nominal_wr_positive_past_income_age(
        self,
        rila_past_income_age: Annuity_RILA,
    ) -> None:
        """nominal_WR is positive once client has passed the income start age."""
        result = rila_past_income_age.calc_withdrawal_rates(100_000.0, desired_WR=0.04)
        assert result.nominal_WR > 0.0

    def test_real_wr_equals_nominal_when_no_inflation(
        self,
        rila_past_income_age: Annuity_RILA,
    ) -> None:
        """real_WR equals nominal_WR when no inflation data is provided."""
        result = rila_past_income_age.calc_withdrawal_rates(100_000.0, desired_WR=0.04)
        assert result.real_WR == pytest.approx(result.nominal_WR, rel=1e-9)

    def test_real_wr_deflated_with_inflation(
        self,
        rila_past_income_age: Annuity_RILA,
        inflation_wr_df: pl.DataFrame,
    ) -> None:
        """real_WR is deflated by the inverse inflation factor."""
        result_no_infl = rila_past_income_age.calc_withdrawal_rates(
            100_000.0,
            desired_WR=0.04,
        )
        result_with_infl = rila_past_income_age.calc_withdrawal_rates(
            100_000.0,
            desired_WR=0.04,
            obj_inflation=inflation_wr_df,
        )
        expected_real = result_no_infl.nominal_WR * (1.0 / 1.03)
        assert result_with_infl.real_WR == pytest.approx(expected_real, rel=1e-9)

    def test_real_wr_less_than_nominal_when_inflation_positive(
        self,
        rila_past_income_age: Annuity_RILA,
        inflation_wr_df: pl.DataFrame,
    ) -> None:
        """real_WR < nominal_WR when the inflation factor is above 1.0."""
        result = rila_past_income_age.calc_withdrawal_rates(
            100_000.0,
            desired_WR=0.04,
            obj_inflation=inflation_wr_df,
        )
        assert result.real_WR < result.nominal_WR

    def test_invalid_principal_zero_raises(
        self,
        rila_buffer_cap: Annuity_RILA,
    ) -> None:
        """Zero amount_principal raises Exception_Validation_Input."""
        with pytest.raises(Exception_Validation_Input):
            rila_buffer_cap.calc_withdrawal_rates(0.0, desired_WR=0.04)

    def test_invalid_principal_negative_raises(
        self,
        rila_buffer_cap: Annuity_RILA,
    ) -> None:
        """Negative amount_principal raises Exception_Validation_Input."""
        with pytest.raises(Exception_Validation_Input):
            rila_buffer_cap.calc_withdrawal_rates(-1.0, desired_WR=0.04)

    def test_invalid_desired_wr_zero_raises(
        self,
        rila_buffer_cap: Annuity_RILA,
    ) -> None:
        """Zero desired_WR raises Exception_Validation_Input."""
        with pytest.raises(Exception_Validation_Input):
            rila_buffer_cap.calc_withdrawal_rates(100_000.0, desired_WR=0.0)

    def test_invalid_desired_wr_negative_raises(
        self,
        rila_buffer_cap: Annuity_RILA,
    ) -> None:
        """Negative desired_WR raises Exception_Validation_Input."""
        with pytest.raises(Exception_Validation_Input):
            rila_buffer_cap.calc_withdrawal_rates(100_000.0, desired_WR=-0.04)

    def test_inflation_df_missing_inverse_column_raises(
        self,
        rila_buffer_cap: Annuity_RILA,
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
            rila_buffer_cap.calc_withdrawal_rates(
                100_000.0,
                desired_WR=0.04,
                obj_inflation=incomplete_df,
            )


# ======================================================================
# Income starting age tests
# ======================================================================


@pytest.mark.unit()
class Test_Annuity_RILA_Income_Starting_Age:
    """Test m_annuity_income_starting_age for Annuity_RILA."""

    def test_default_is_client_age(self, rila_buffer_cap: Annuity_RILA) -> None:
        """Default income starting age equals client_age."""
        assert rila_buffer_cap.annuity_income_starting_age == rila_buffer_cap.client_age

    def test_custom_income_starting_age_stored(self) -> None:
        """Explicitly provided income starting age >= client_age is stored correctly."""
        rila = Annuity_RILA(
            client_age=55,
            annuity_payout_rate=0.05,
            age_income_start=65,
            annuity_income_starting_age=60,
        )
        assert rila.annuity_income_starting_age == 60

    def test_income_starting_age_equal_to_client_age(self) -> None:
        """Income starting age equal to client_age is accepted."""
        rila = Annuity_RILA(
            client_age=55,
            annuity_payout_rate=0.05,
            age_income_start=65,
            annuity_income_starting_age=55,
        )
        assert rila.annuity_income_starting_age == 55

    def test_income_starting_age_below_client_age_raises(self) -> None:
        """Income starting age below client_age raises Exception_Validation_Input."""
        with pytest.raises(Exception_Validation_Input):
            Annuity_RILA(
                client_age=55,
                annuity_payout_rate=0.05,
                age_income_start=65,
                annuity_income_starting_age=50,
            )

    def test_member_variable_directly(self) -> None:
        """m_annuity_income_starting_age member variable equals the provided value."""
        rila = Annuity_RILA(
            client_age=55,
            annuity_payout_rate=0.05,
            age_income_start=65,
            annuity_income_starting_age=65,
        )
        assert rila.m_annuity_income_starting_age == 65
