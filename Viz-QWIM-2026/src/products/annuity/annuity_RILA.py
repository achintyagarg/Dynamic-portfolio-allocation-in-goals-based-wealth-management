"""
Registered Index-Linked Annuity (RILA) class.

===============================

A Registered Index-Linked Annuity (also known as a *buffered annuity*
or *structured annuity*) is an SEC-registered product that sits
between a Fixed Indexed Annuity (FIA) and a Variable Annuity (VA) on
the risk spectrum.  The contract holder assumes limited downside
market risk in exchange for higher upside potential than an FIA.

Key characteristics of RILAs:

- **Index-linked crediting**: returns are tied to a market index
  (e.g. S&P 500, Russell 2000, MSCI EAFE) over a defined term.
- **Buffer protection**: the insurer absorbs the first *B* % of index
  losses; the contract holder bears losses beyond the buffer.
- **Floor protection**: (alternative to buffer) the maximum loss is
  capped at the floor level; the contract holder bears losses up to
  the floor with the insurer absorbing the rest.
- **Cap rate**: the maximum return credited per term.
- **Performance trigger**: (alternative crediting strategy) if the
  index return is non-negative, a fixed rate is credited regardless
  of the magnitude of the gain.
- **Term / segment duration**: crediting is evaluated at the end of
  each term (typically 1, 2, 3, or 6 years).  Investors may hold
  multiple overlapping segments.
- **Interim value**: a market-value-adjusted account value calculated
  during a term (before maturity) that reflects current index
  performance and a discount factor.
- **Surrender charges**: declining schedule of early withdrawal
  penalties similar to FIA/VA products.
- **No guaranteed minimum accumulation**: unlike an FIA, the account
  value can decline (up to the buffer/floor limit).  RILA contracts
  are SEC-registered securities.

Author
------
QWIM Team

Version
-------
0.5.1 (2026-03-01)
"""

from __future__ import annotations

import polars as pl

from aenum import Enum

from src.utils.custom_exceptions_errors_loggers.exception_custom import (
    Exception_Calculation,
    Exception_Validation_Input,
)
from src.utils.custom_exceptions_errors_loggers.logger_custom import get_logger

from .annuity_base import Annuity_Base, Annuity_Type, Withdrawal_Rates


logger = get_logger(__name__)


# ======================================================================
# RILA-specific enumerations
# ======================================================================


class Protection_Type(Enum):
    """Downside protection mechanism for a RILA segment.

    Attributes
    ----------
    BUFFER : str
        The insurer absorbs the first *B* % of index losses.
        The contract holder bears losses beyond the buffer.
    FLOOR : str
        The contract holder bears losses up to the floor percentage.
        The insurer absorbs losses beyond the floor.
    """

    BUFFER = "Buffer"
    FLOOR = "Floor"


class Crediting_Strategy(Enum):
    """Index-return crediting strategy for a RILA segment.

    Attributes
    ----------
    CAP : str
        The credited return is the index return capped at a maximum,
        subject to the downside protection mechanism.
    PERFORMANCE_TRIGGER : str
        If the index return is zero or positive, a predetermined fixed
        trigger rate is credited.  If negative, the downside protection
        mechanism applies.
    PARTICIPATION_RATE : str
        A fraction of the index return is credited up to an optional cap,
        subject to the downside protection mechanism.
    """

    CAP = "Cap"
    PERFORMANCE_TRIGGER = "Performance Trigger"
    PARTICIPATION_RATE = "Participation Rate"


# ======================================================================
# Annuity_RILA class
# ======================================================================


class Annuity_RILA(Annuity_Base):
    r"""Registered Index-Linked Annuity (RILA).

    A RILA provides index-linked returns over defined terms with limited
    downside exposure controlled by a **buffer** or **floor** mechanism.

    **Credited return per term (cap strategy)**:

    With buffer protection:

    $$
    r_{\text{credited}} = \begin{cases}
        \min(R_{\text{index}},\; c)
            & \text{if } R_{\text{index}} \ge 0 \\[4pt]
        0
            & \text{if } -b \le R_{\text{index}} < 0 \\[4pt]
        R_{\text{index}} + b
            & \text{if } R_{\text{index}} < -b
    \end{cases}
    $$

    With floor protection:

    $$
    r_{\text{credited}} = \begin{cases}
        \min(R_{\text{index}},\; c)
            & \text{if } R_{\text{index}} \ge 0 \\[4pt]
        \max(R_{\text{index}},\; f)
            & \text{if } R_{\text{index}} < 0
    \end{cases}
    $$

    **Performance trigger strategy**:

    $$
    r_{\text{credited}} = \begin{cases}
        r_{\text{trigger}}
            & \text{if } R_{\text{index}} \ge 0 \\[4pt]
        \text{downside}(R_{\text{index}})
            & \text{if } R_{\text{index}} < 0
    \end{cases}
    $$

    **Interim value** (market-value-adjusted mid-term):

    $$
    IV = P \times (1 + r_{\text{credited, partial}})
        \times \frac{1}{(1 + d)^{T - t}}
    $$

    Where $d$ = discount rate, $T$ = term length, and $t$ = elapsed years.

    Attributes
    ----------
    m_client_age : int
        The current age of the client (inherited).
    m_annuity_payout_rate : float
        The payout rate of the annuity (inherited).
    m_annuity_type : Annuity_Type
        Set to ``ANNUITY_RILA`` (inherited).
    m_age_income_start : int
        The age at which income payments begin.
    m_term_years : int
        Duration of each crediting term / segment (years).
    m_protection_type : Protection_Type
        Type of downside protection (BUFFER or FLOOR).
    m_buffer_rate : float
        Buffer percentage absorbed by the insurer (e.g. 0.10 for 10 %).
        Only meaningful when ``m_protection_type`` is ``BUFFER``.
    m_floor_rate : float
        Floor / maximum loss percentage for the contract holder (e.g.
        -0.10 for -10 %).  Only meaningful when ``m_protection_type``
        is ``FLOOR``.
    m_crediting_strategy : Crediting_Strategy
        Crediting strategy applied to the index return.
    m_cap_rate : float
        Maximum return credited per term (used with CAP or
        PARTICIPATION_RATE strategies).
    m_participation_rate : float
        Fraction of the index return credited (used with
        PARTICIPATION_RATE strategy).
    m_performance_trigger_rate : float
        Fixed rate credited when the index return >= 0 (used with
        PERFORMANCE_TRIGGER strategy).
    m_rate_rider_charge : float
        Annual rider charge for optional benefits (as a decimal).
    m_rate_surrender_charge_initial : float
        Initial surrender charge rate (year 0).
    m_surrender_charge_schedule_years : int
        Years over which the surrender charge declines to zero.
    m_pct_free_withdrawal : float
        Annual free-withdrawal percentage.
    m_rate_interim_discount : float
        Discount rate used in interim-value calculation.
    m_has_GMDB : bool
        Whether the contract includes a Guaranteed Minimum Death Benefit.
    m_has_return_of_premium_death_benefit : bool
        Whether the death benefit guarantees at least the premium paid.
    m_payment_frequency : int
        Number of payments per year.
    m_financial_index : str
        Name of the financial index column in ``obj_scenarios``.
    """

    def __init__(
        self,
        client_age: int,
        annuity_payout_rate: float,
        age_income_start: int,
        term_years: int = 6,
        protection_type: Protection_Type = Protection_Type.BUFFER,
        buffer_rate: float = 0.10,
        floor_rate: float = -0.10,
        crediting_strategy: Crediting_Strategy = Crediting_Strategy.CAP,
        cap_rate: float = 0.15,
        participation_rate: float = 1.0,
        performance_trigger_rate: float = 0.0,
        rate_rider_charge: float = 0.0,
        rate_surrender_charge_initial: float = 0.06,
        surrender_charge_schedule_years: int = 6,
        pct_free_withdrawal: float = 0.10,
        rate_interim_discount: float = 0.02,
        has_GMDB: bool = False,
        has_return_of_premium_death_benefit: bool = True,
        payment_frequency: int = 12,
        financial_index: str = "S&P 500",
        annuity_income_starting_age: int | None = None,
    ) -> None:
        """Initialize a Registered Index-Linked Annuity object.

        Parameters
        ----------
        client_age : int
            The current age of the client.
        annuity_payout_rate : float
            The annual payout rate of the annuity (as a decimal).
        age_income_start : int
            The age at which income payments begin.
        term_years : int, optional
            Duration of each crediting term in years (default 6).
        protection_type : Protection_Type, optional
            Downside protection mechanism (default ``BUFFER``).
        buffer_rate : float, optional
            Buffer percentage absorbed by insurer (default 0.10 = 10 %).
        floor_rate : float, optional
            Floor / maximum loss percentage; should be negative or zero
            (default -0.10 = -10 %).
        crediting_strategy : Crediting_Strategy, optional
            Index-return crediting strategy (default ``CAP``).
        cap_rate : float, optional
            Maximum return credited per term (default 0.15 = 15 %).
        participation_rate : float, optional
            Fraction of index return credited (default 1.0 = 100 %).
        performance_trigger_rate : float, optional
            Fixed rate credited when index return >= 0 (default 0.0).
        rate_rider_charge : float, optional
            Annual rider charge for optional benefits (default 0.0).
        rate_surrender_charge_initial : float, optional
            Initial surrender charge rate (default 0.06 = 6 %).
        surrender_charge_schedule_years : int, optional
            Years for surrender charge to decline to zero (default 6).
        pct_free_withdrawal : float, optional
            Annual free-withdrawal percentage (default 0.10 = 10 %).
        rate_interim_discount : float, optional
            Discount rate for interim-value calculation (default 0.02).
        has_GMDB : bool, optional
            Include Guaranteed Minimum Death Benefit (default ``False``).
        has_return_of_premium_death_benefit : bool, optional
            Death benefit guarantees at least the premium (default ``True``).
        payment_frequency : int, optional
            Number of payments per year (default 12 — monthly).
        financial_index : str, optional
            Name of the column in ``obj_scenarios`` containing index returns.
            Defaults to ``"S&P 500"``.
        annuity_income_starting_age : int | None, optional
            The age when the client will start receiving income from this
            annuity.  Must be ``>= client_age`` when provided.  Defaults
            to ``client_age`` when ``None``.

        Raises
        ------
        Exception_Validation_Input
            If any input fails validation.
        """
        super().__init__(
            client_age,
            annuity_payout_rate,
            Annuity_Type.ANNUITY_RILA,
            annuity_income_starting_age=annuity_income_starting_age,
        )

        # --- Input validation ---
        if not isinstance(age_income_start, int) or age_income_start <= client_age:
            raise Exception_Validation_Input(
                "age_income_start must be an integer greater than client_age",
                field_name="age_income_start",
                expected_type=int,
                actual_value=age_income_start,
            )

        if not isinstance(term_years, int) or term_years <= 0:
            raise Exception_Validation_Input(
                "term_years must be a positive integer",
                field_name="term_years",
                expected_type=int,
                actual_value=term_years,
            )

        if not isinstance(protection_type, Protection_Type):
            raise Exception_Validation_Input(
                "protection_type must be a valid Protection_Type enum",
                field_name="protection_type",
                expected_type=Protection_Type,
                actual_value=protection_type,
            )

        if not isinstance(buffer_rate, (int, float)) or buffer_rate < 0 or buffer_rate > 1:
            raise Exception_Validation_Input(
                "buffer_rate must be between 0 and 1",
                field_name="buffer_rate",
                expected_type=float,
                actual_value=buffer_rate,
            )

        if not isinstance(floor_rate, (int, float)) or floor_rate > 0:
            raise Exception_Validation_Input(
                "floor_rate must be a non-positive number (e.g. -0.10)",
                field_name="floor_rate",
                expected_type=float,
                actual_value=floor_rate,
            )

        if not isinstance(crediting_strategy, Crediting_Strategy):
            raise Exception_Validation_Input(
                "crediting_strategy must be a valid Crediting_Strategy enum",
                field_name="crediting_strategy",
                expected_type=Crediting_Strategy,
                actual_value=crediting_strategy,
            )

        if not isinstance(cap_rate, (int, float)) or cap_rate < 0:
            raise Exception_Validation_Input(
                "cap_rate must be a non-negative number",
                field_name="cap_rate",
                expected_type=float,
                actual_value=cap_rate,
            )

        if (
            not isinstance(participation_rate, (int, float))
            or participation_rate < 0
            or participation_rate > 3.0
        ):
            raise Exception_Validation_Input(
                "participation_rate must be between 0 and 3.0",
                field_name="participation_rate",
                expected_type=float,
                actual_value=participation_rate,
            )

        if not isinstance(performance_trigger_rate, (int, float)) or performance_trigger_rate < 0:
            raise Exception_Validation_Input(
                "performance_trigger_rate must be a non-negative number",
                field_name="performance_trigger_rate",
                expected_type=float,
                actual_value=performance_trigger_rate,
            )

        if not isinstance(rate_rider_charge, (int, float)) or rate_rider_charge < 0:
            raise Exception_Validation_Input(
                "rate_rider_charge must be a non-negative number",
                field_name="rate_rider_charge",
                expected_type=float,
                actual_value=rate_rider_charge,
            )

        if (
            not isinstance(rate_surrender_charge_initial, (int, float))
            or rate_surrender_charge_initial < 0
        ):
            raise Exception_Validation_Input(
                "rate_surrender_charge_initial must be non-negative",
                field_name="rate_surrender_charge_initial",
                expected_type=float,
                actual_value=rate_surrender_charge_initial,
            )

        if (
            not isinstance(surrender_charge_schedule_years, int)
            or surrender_charge_schedule_years < 0
        ):
            raise Exception_Validation_Input(
                "surrender_charge_schedule_years must be a non-negative integer",
                field_name="surrender_charge_schedule_years",
                expected_type=int,
                actual_value=surrender_charge_schedule_years,
            )

        if not isinstance(rate_interim_discount, (int, float)) or rate_interim_discount < 0:
            raise Exception_Validation_Input(
                "rate_interim_discount must be a non-negative number",
                field_name="rate_interim_discount",
                expected_type=float,
                actual_value=rate_interim_discount,
            )

        if not isinstance(payment_frequency, int) or payment_frequency <= 0:
            raise Exception_Validation_Input(
                "payment_frequency must be a positive integer",
                field_name="payment_frequency",
                expected_type=int,
                actual_value=payment_frequency,
            )

        if not isinstance(financial_index, str) or not financial_index.strip():
            raise Exception_Validation_Input(
                "financial_index must be a non-empty string",
                field_name="financial_index",
                expected_type=str,
                actual_value=financial_index,
            )

        # --- Set member variables ---
        self.m_age_income_start: int = age_income_start
        self.m_term_years: int = term_years
        self.m_protection_type: Protection_Type = protection_type
        self.m_buffer_rate: float = float(buffer_rate)
        self.m_floor_rate: float = float(floor_rate)
        self.m_crediting_strategy: Crediting_Strategy = crediting_strategy
        self.m_cap_rate: float = float(cap_rate)
        self.m_participation_rate: float = float(participation_rate)
        self.m_performance_trigger_rate: float = float(performance_trigger_rate)
        self.m_rate_rider_charge: float = float(rate_rider_charge)
        self.m_rate_surrender_charge_initial: float = float(rate_surrender_charge_initial)
        self.m_surrender_charge_schedule_years: int = surrender_charge_schedule_years
        self.m_pct_free_withdrawal: float = float(pct_free_withdrawal)
        self.m_rate_interim_discount: float = float(rate_interim_discount)
        self.m_has_GMDB: bool = has_GMDB
        self.m_has_return_of_premium_death_benefit: bool = has_return_of_premium_death_benefit
        self.m_payment_frequency: int = payment_frequency
        self.m_financial_index: str = financial_index

        logger.info(
            f"Created RILA with payout rate {annuity_payout_rate:.2%}, "
            f"income at age {age_income_start}, "
            f"term: {term_years}yr, "
            f"protection: {protection_type.value} "
            f"(buffer={buffer_rate:.0%}, floor={floor_rate:.0%}), "
            f"crediting: {crediting_strategy.value} "
            f"(cap={cap_rate:.0%}, participation={participation_rate:.0%}, "
            f"trigger={performance_trigger_rate:.0%}), "
            f"surrender: {rate_surrender_charge_initial:.1%}/"
            f"{surrender_charge_schedule_years}yr, "
            f"index: {financial_index!r}",
        )

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def income_start_age(self) -> int:
        """Int : The age at which income payments begin."""
        return self.m_age_income_start

    @property
    def term_years(self) -> int:
        """Int : Duration of each crediting term in years."""
        return self.m_term_years

    @property
    def protection_type(self) -> Protection_Type:
        """Protection_Type : The downside protection mechanism."""
        return self.m_protection_type

    @property
    def buffer_rate(self) -> float:
        """Float : Buffer percentage absorbed by the insurer."""
        return self.m_buffer_rate

    @property
    def floor_rate(self) -> float:
        """Float : Floor / maximum loss percentage (negative or zero)."""
        return self.m_floor_rate

    @property
    def crediting_strategy(self) -> Crediting_Strategy:
        """Crediting_Strategy : The index-return crediting strategy."""
        return self.m_crediting_strategy

    @property
    def cap_rate(self) -> float:
        """Float : Maximum return credited per term."""
        return self.m_cap_rate

    @property
    def participation_rate(self) -> float:
        """Float : Fraction of the index return credited."""
        return self.m_participation_rate

    @property
    def performance_trigger_rate(self) -> float:
        """Float : Fixed rate credited when the index return >= 0."""
        return self.m_performance_trigger_rate

    @property
    def rate_rider_charge(self) -> float:
        """Float : Annual rider charge for optional benefits."""
        return self.m_rate_rider_charge

    @property
    def pct_free_withdrawal(self) -> float:
        """Float : Annual free-withdrawal percentage."""
        return self.m_pct_free_withdrawal

    @property
    def rate_interim_discount(self) -> float:
        """Float : Discount rate used in interim-value calculation."""
        return self.m_rate_interim_discount

    @property
    def has_GMDB(self) -> bool:
        """Bool : Whether the contract includes a GMDB."""
        return self.m_has_GMDB

    @property
    def has_return_of_premium_death_benefit(self) -> bool:
        """Bool : Whether the death benefit guarantees at least the premium."""
        return self.m_has_return_of_premium_death_benefit

    @property
    def payment_frequency(self) -> int:
        """Int : Number of payments per year."""
        return self.m_payment_frequency

    @property
    def financial_index(self) -> str:
        """Str : Name of the financial index column used in scenario data."""
        return self.m_financial_index

    @property
    def deferral_years(self) -> int:
        """Int : Number of years until income payments begin."""
        return self.m_age_income_start - self.m_client_age

    # ------------------------------------------------------------------
    # Downside protection calculations
    # ------------------------------------------------------------------

    def calc_downside_return(
        self,
        index_return: float,
    ) -> float:
        r"""Calculate the return after applying downside protection.

        **Buffer protection** (insurer absorbs first *b* % of loss):

        $$
        r_{\text{down}} = \begin{cases}
            0 & \text{if } -b \le R < 0 \\
            R + b & \text{if } R < -b
        \end{cases}
        $$

        **Floor protection** (maximum loss = *f*):

        $$
        r_{\text{down}} = \max(R,\; f)
        $$

        Parameters
        ----------
        index_return : float
            The raw index return for the term (as a decimal).

        Returns
        -------
        float
            The return after downside protection is applied.
            Always ``<= 0`` (or ``0`` if the index did not decline).

        Raises
        ------
        Exception_Validation_Input
            If ``index_return`` is not a number.

        Examples
        --------
        >>> rila = Annuity_RILA(60, 0.05, 65, buffer_rate=0.10)
        >>> rila.calc_downside_return(-0.05)
        0.0
        >>> rila.calc_downside_return(-0.25)
        -0.15
        """
        if not isinstance(index_return, (int, float)):
            raise Exception_Validation_Input(
                "index_return must be a number",
                field_name="index_return",
                expected_type=float,
                actual_value=index_return,
            )

        if index_return >= 0:
            return 0.0

        if self.m_protection_type == Protection_Type.BUFFER:
            # Insurer absorbs the first buffer_rate of losses
            if abs(index_return) <= self.m_buffer_rate:
                return 0.0
            return index_return + self.m_buffer_rate

        # Floor protection: loss capped at floor_rate
        return max(index_return, self.m_floor_rate)

    # ------------------------------------------------------------------
    # Credited-rate calculations
    # ------------------------------------------------------------------

    def calc_credited_rate(
        self,
        index_return: float,
    ) -> float:
        r"""Calculate the credited rate for a single term.

        Applies the selected crediting strategy and downside protection.

        **Cap strategy**:

        $$
        r_{\text{credited}} = \begin{cases}
            \min(R_{\text{index}},\; c)
                & \text{if } R_{\text{index}} \ge 0 \\[4pt]
            \text{downside}(R_{\text{index}})
                & \text{if } R_{\text{index}} < 0
        \end{cases}
        $$

        **Performance trigger strategy**:

        $$
        r_{\text{credited}} = \begin{cases}
            r_{\text{trigger}}
                & \text{if } R_{\text{index}} \ge 0 \\[4pt]
            \text{downside}(R_{\text{index}})
                & \text{if } R_{\text{index}} < 0
        \end{cases}
        $$

        **Participation rate strategy**:

        $$
        r_{\text{credited}} = \begin{cases}
            \min(R_{\text{index}} \times p,\; c)
                & \text{if } R_{\text{index}} \ge 0 \\[4pt]
            \text{downside}(R_{\text{index}})
                & \text{if } R_{\text{index}} < 0
        \end{cases}
        $$

        Parameters
        ----------
        index_return : float
            The raw index return for the term (as a decimal).

        Returns
        -------
        float
            The credited rate after applying the crediting strategy
            and downside protection.

        Raises
        ------
        Exception_Validation_Input
            If ``index_return`` is not a number.
        Exception_Calculation
            If the crediting strategy is unrecognised.

        Examples
        --------
        >>> rila = Annuity_RILA(60, 0.05, 65, cap_rate=0.15, buffer_rate=0.10)
        >>> rila.calc_credited_rate(0.20)
        0.15
        >>> rila.calc_credited_rate(-0.05)
        0.0
        >>> rila.calc_credited_rate(-0.25)
        -0.15
        """
        if not isinstance(index_return, (int, float)):
            raise Exception_Validation_Input(
                "index_return must be a number",
                field_name="index_return",
                expected_type=float,
                actual_value=index_return,
            )

        # --- Negative index return: apply downside protection ---
        if index_return < 0:
            return self.calc_downside_return(index_return)

        # --- Non-negative index return: apply crediting strategy ---
        if self.m_crediting_strategy == Crediting_Strategy.CAP:
            return min(index_return, self.m_cap_rate)

        if self.m_crediting_strategy == Crediting_Strategy.PERFORMANCE_TRIGGER:
            return self.m_performance_trigger_rate

        if self.m_crediting_strategy == Crediting_Strategy.PARTICIPATION_RATE:
            participated = index_return * self.m_participation_rate
            return min(participated, self.m_cap_rate)

        raise Exception_Calculation(
            f"Unrecognised crediting strategy: {self.m_crediting_strategy}",
        )

    def calc_annualised_credited_rate(
        self,
        index_return: float,
        term_years: int | None = None,
    ) -> float:
        r"""Annualise the term credited rate.

        $$
        r_{\text{annual}} = (1 + r_{\text{term}})^{1/T} - 1
        $$

        Parameters
        ----------
        index_return : float
            The raw index return for the term.
        term_years : int | None, optional
            Term length to annualise over (default: :attr:`term_years`).

        Returns
        -------
        float
            The annualised credited rate.

        Raises
        ------
        Exception_Validation_Input
            If inputs are invalid.
        """
        if term_years is None:
            term_years = self.m_term_years

        if not isinstance(term_years, int) or term_years <= 0:
            raise Exception_Validation_Input(
                "term_years must be a positive integer",
                field_name="term_years",
                expected_type=int,
                actual_value=term_years,
            )

        credited = self.calc_credited_rate(index_return)
        # Guard: if total return makes (1 + credited) negative, clamp to near-total loss
        growth_factor = max(1 + credited, 1e-10)
        return growth_factor ** (1.0 / term_years) - 1

    # ------------------------------------------------------------------
    # Account value and interim value
    # ------------------------------------------------------------------

    def calc_account_value_at_term_end(
        self,
        amount_principal: float,
        index_return: float,
    ) -> float:
        r"""Calculate the account value at the end of one crediting term.

        $$
        V = P \times (1 + r_{\text{credited}})
        $$

        Parameters
        ----------
        amount_principal : float
            The principal invested at the start of the term.
        index_return : float
            The cumulative index return over the term.

        Returns
        -------
        float
            The account value at term end.

        Raises
        ------
        Exception_Validation_Input
            If inputs are invalid.
        """
        if not isinstance(amount_principal, (int, float)) or amount_principal <= 0:
            raise Exception_Validation_Input(
                "amount_principal must be a positive number",
                field_name="amount_principal",
                expected_type=float,
                actual_value=amount_principal,
            )

        credited = self.calc_credited_rate(index_return)
        return amount_principal * (1 + credited)

    def calc_interim_value(
        self,
        amount_principal: float,
        current_index_return: float,
        years_elapsed: int,
    ) -> float:
        r"""Calculate the market-value-adjusted interim value mid-term.

        The interim value reflects the current index performance and
        applies a discount for the remaining term:

        $$
        IV = P \times (1 + r_{\text{credited, partial}})
            \times \frac{1}{(1 + d)^{T - t}}
        $$

        Where $d$ = ``rate_interim_discount``, $T$ = total term, and
        $t$ = years elapsed.

        Parameters
        ----------
        amount_principal : float
            The principal invested at the start of the term.
        current_index_return : float
            The index return so far within the term.
        years_elapsed : int
            Number of years elapsed within the current term.

        Returns
        -------
        float
            The interim account value.

        Raises
        ------
        Exception_Validation_Input
            If inputs are invalid.
        """
        if not isinstance(amount_principal, (int, float)) or amount_principal <= 0:
            raise Exception_Validation_Input(
                "amount_principal must be a positive number",
                field_name="amount_principal",
                expected_type=float,
                actual_value=amount_principal,
            )

        if not isinstance(years_elapsed, int) or years_elapsed < 0:
            raise Exception_Validation_Input(
                "years_elapsed must be a non-negative integer",
                field_name="years_elapsed",
                expected_type=int,
                actual_value=years_elapsed,
            )

        if years_elapsed > self.m_term_years:
            raise Exception_Validation_Input(
                "years_elapsed cannot exceed term_years",
                field_name="years_elapsed",
                expected_type=int,
                actual_value=years_elapsed,
            )

        # Apply crediting rules to the partial-term index return
        credited_partial = self.calc_credited_rate(current_index_return)

        remaining_years = self.m_term_years - years_elapsed
        discount_factor = 1.0 / ((1 + self.m_rate_interim_discount) ** remaining_years)

        return amount_principal * (1 + credited_partial) * discount_factor

    def calc_account_values_multi_term(
        self,
        amount_principal: float,
        term_index_returns: list[float],
    ) -> list[float]:
        r"""Simulate account value across multiple consecutive terms.

        For each term *i* the account value at the end is:

        $$
        V_i = V_{i-1} \times (1 + r_{\text{credited}, i})
        $$

        Annual rider charges are deducted from the account value once
        per term (compounded over the term length):

        $$
        V_i = V_i \times (1 - c_{\text{rider}})^{T}
        $$

        Parameters
        ----------
        amount_principal : float
            The initial premium invested.
        term_index_returns : list[float]
            A list of cumulative index returns, one per term.

        Returns
        -------
        list[float]
            Account values at the end of each term, starting with the
            initial value at index 0.

        Raises
        ------
        Exception_Validation_Input
            If inputs are invalid.
        """
        if not isinstance(amount_principal, (int, float)) or amount_principal <= 0:
            raise Exception_Validation_Input(
                "amount_principal must be a positive number",
                field_name="amount_principal",
                expected_type=float,
                actual_value=amount_principal,
            )

        if not isinstance(term_index_returns, list):
            raise Exception_Validation_Input(
                "term_index_returns must be a list of numbers",
                field_name="term_index_returns",
                expected_type=list,
                actual_value=term_index_returns,
            )

        values: list[float] = [float(amount_principal)]
        current_value = float(amount_principal)

        charge_factor = (1 - self.m_rate_rider_charge) ** self.m_term_years

        for term_return in term_index_returns:
            credited = self.calc_credited_rate(term_return)
            current_value *= 1 + credited
            # Deduct rider charges accumulated over the term
            current_value *= charge_factor
            current_value = max(current_value, 0.0)
            values.append(current_value)

        return values

    # ------------------------------------------------------------------
    # Surrender calculations
    # ------------------------------------------------------------------

    def calc_surrender_charge_rate(
        self,
        year: int,
    ) -> float:
        r"""Calculate the surrender charge rate for a given contract year.

        Uses a linearly declining schedule:

        $$
        SC(t) = \max\!\Bigl(0,\; SC_0 \times \bigl(1 - \tfrac{t}{T}\bigr)\Bigr)
        $$

        Where $SC_0$ = initial surrender charge, $T$ = schedule duration.

        Parameters
        ----------
        year : int
            The contract year (0-indexed; year 0 = purchase year).

        Returns
        -------
        float
            The surrender charge rate as a decimal.  Returns 0.0 if the
            surrender period has expired.
        """
        if not isinstance(year, int) or year < 0:
            return 0.0

        if self.m_surrender_charge_schedule_years <= 0:
            return 0.0

        if year >= self.m_surrender_charge_schedule_years:
            return 0.0

        return self.m_rate_surrender_charge_initial * (
            1 - year / self.m_surrender_charge_schedule_years
        )

    def calc_surrender_value(
        self,
        account_value: float,
        year: int,
    ) -> float:
        """Calculate the surrender (cash-out) value.

        The free-withdrawal portion is exempt from surrender charges.

        Parameters
        ----------
        account_value : float
            The current account value.
        year : int
            The contract year (0-indexed).

        Returns
        -------
        float
            The net surrender value after charges.

        Raises
        ------
        Exception_Validation_Input
            If ``account_value`` is not a positive number.
        """
        if not isinstance(account_value, (int, float)) or account_value <= 0:
            raise Exception_Validation_Input(
                "account_value must be a positive number",
                field_name="account_value",
                expected_type=float,
                actual_value=account_value,
            )

        charge_rate = self.calc_surrender_charge_rate(year)
        free_amount = account_value * self.m_pct_free_withdrawal
        excess_amount = max(account_value - free_amount, 0.0)
        surrender_charge = excess_amount * charge_rate
        return account_value - surrender_charge

    # ------------------------------------------------------------------
    # Death benefit
    # ------------------------------------------------------------------

    def calc_death_benefit(
        self,
        amount_principal: float,
        account_value: float,
    ) -> float:
        r"""Calculate the death benefit.

        If the return-of-premium death benefit is elected:

        $$
        DB = \max(V_{\text{account}},\; A)
        $$

        Otherwise:

        $$
        DB = V_{\text{account}}
        $$

        Parameters
        ----------
        amount_principal : float
            The initial premium invested.
        account_value : float
            The current account value.

        Returns
        -------
        float
            The death benefit amount.

        Raises
        ------
        Exception_Validation_Input
            If inputs are invalid.
        """
        if not isinstance(amount_principal, (int, float)) or amount_principal <= 0:
            raise Exception_Validation_Input(
                "amount_principal must be a positive number",
                field_name="amount_principal",
                expected_type=float,
                actual_value=amount_principal,
            )

        if not isinstance(account_value, (int, float)) or account_value < 0:
            raise Exception_Validation_Input(
                "account_value must be a non-negative number",
                field_name="account_value",
                expected_type=float,
                actual_value=account_value,
            )

        if self.m_has_return_of_premium_death_benefit:
            return max(account_value, amount_principal)

        if self.m_has_GMDB:
            return max(account_value, amount_principal)

        return account_value

    # ------------------------------------------------------------------
    # Payout calculations
    # ------------------------------------------------------------------

    def calc_annuity_payout(
        self,
        amount_principal: float,
        obj_scenarios: pl.DataFrame | None = None,
        obj_inflation: pl.DataFrame | None = None,
    ) -> float:
        """Calculate the RILA annual payout.

        The payout depends on:

        1. Whether the client has reached the income start age.
        2. The account value after crediting across terms (via scenario
           data if available, otherwise the principal is used directly).
        3. The payout rate applied to the account value or benefit base.

        Parameters
        ----------
        amount_principal : float
            The principal amount invested.
        obj_scenarios : pl.DataFrame | None, optional
            Polars DataFrame containing scenario market data.  The first column
            must be ``Date``; the column named ``self.m_financial_index`` must
            contain index returns per crediting term.
        obj_inflation : pl.DataFrame | None, optional
            Polars DataFrame with pre-calculated inflation factors (unused
            for RILA; kept for interface compatibility).  Expected schema:

            - ``Start Date`` : start of the inflation period.
            - ``End Date``   : end of the inflation period.
            - ``Inflation Factor`` : cumulative inflation already calculated
              for the interval ``[Start Date, End Date]``
            - ``Inverse Inflation Factor`` : inverse of cumulative inflation already calculated
              for the interval ``[Start Date, End Date]``.

        Returns
        -------
        float
            The calculated annual payout.  Returns ``0.0`` while in
            deferral.

        Raises
        ------
        Exception_Validation_Input
            If ``amount_principal`` is not a positive number or if
            ``obj_scenarios`` is missing required columns.
        """
        if not isinstance(amount_principal, (int, float)) or amount_principal <= 0:
            raise Exception_Validation_Input(
                "amount_principal must be a positive number",
                field_name="amount_principal",
                expected_type=float,
                actual_value=amount_principal,
            )

        # Still in deferral period
        if self.m_client_age < self.m_age_income_start:
            return 0.0

        benefit_base: float = float(amount_principal)

        if obj_scenarios is not None:
            required_cols = {"Date", self.m_financial_index}
            if not required_cols.issubset(set(obj_scenarios.columns)):
                missing = required_cols - set(obj_scenarios.columns)
                raise Exception_Validation_Input(
                    f"obj_scenarios is missing required columns: {missing}",
                    field_name="obj_scenarios",
                    expected_type=pl.DataFrame,
                    actual_value=obj_scenarios.columns,
                )
            term_returns = obj_scenarios[self.m_financial_index].to_list()
            account_values = self.calc_account_values_multi_term(
                amount_principal,
                term_returns,
            )
            benefit_base = account_values[-1] if account_values else amount_principal
            logger.info(f"RILA benefit base after index crediting: {benefit_base:.2f}")
        else:
            logger.info(f"RILA benefit base from principal: {benefit_base:.2f}")

        annual_payout: float = benefit_base * self.m_annuity_payout_rate
        logger.info(f"RILA annual payout: {annual_payout:.2f}")

        return annual_payout

    def calc_withdrawal_rates(
        self,
        amount_principal: float,
        desired_WR: float | None = None,
        obj_scenarios: pl.DataFrame | None = None,
        obj_inflation: pl.DataFrame | None = None,
    ) -> Withdrawal_Rates:
        r"""Calculate nominal and real withdrawal rates for this RILA.

        The nominal withdrawal rate is the ratio of the RILA payout
        (account-value driven, index-credited across terms if scenarios
        are supplied) to the invested principal.  The real withdrawal rate
        deflates the nominal rate by the cumulative inverse inflation factor:

        $$
        WR_{\text{nominal}} = \frac{P_{\text{annual}}}{A}
        $$

        $$
        WR_{\text{real}} = WR_{\text{nominal}} \times IF^{-1}
        $$

        where $IF^{-1}$ is the product of the ``Inverse Inflation Factor``
        column in ``obj_inflation`` (defaults to 1.0 when no inflation data
        is supplied).

        Parameters
        ----------
        amount_principal : float
            The principal amount invested.
        desired_WR : float
            The desired withdrawal rate (as a decimal, e.g. 0.04 for 4 %).
            Validated and logged for reference; the returned rates reflect
            what the annuity actually delivers.
        obj_scenarios : pl.DataFrame | None, optional
            Polars DataFrame containing scenario market data.  The first
            column must be ``Date``; the column named ``self.m_financial_index``
            must contain index returns per crediting term.  See
            :meth:`calc_annuity_payout` for full schema details.
        obj_inflation : pl.DataFrame | None, optional
            Polars DataFrame with pre-calculated inflation factors.  Must
            contain exactly four columns:

            - ``Start Date`` : start of the inflation period.
            - ``End Date``   : end of the inflation period.
            - ``Inflation Factor`` : cumulative inflation for the interval.
            - ``Inverse Inflation Factor`` : reciprocal of the inflation factor.

            When multiple rows are present the individual inverse factors are
            multiplied to produce a single cumulative adjustment.

        Returns
        -------
        Withdrawal_Rates
            Struct with:

            - ``nominal_WR``: payout-to-principal ratio (0.0 during deferral).
            - ``real_WR``: nominal rate deflated by the inverse inflation factor.

        Raises
        ------
        Exception_Validation_Input
            If ``amount_principal`` is not a positive number.
            If ``desired_WR`` is not a positive number.
            If ``obj_inflation`` is provided but missing required columns.
        """
        if not isinstance(amount_principal, (int, float)) or amount_principal <= 0:
            raise Exception_Validation_Input(
                "amount_principal must be a positive number",
                field_name="amount_principal",
                expected_type=float,
                actual_value=amount_principal,
            )

        effective_desired_WR: float = (
            desired_WR if desired_WR is not None else self.m_annuity_payout_rate
        )
        if not isinstance(effective_desired_WR, (int, float)) or effective_desired_WR <= 0:
            raise Exception_Validation_Input(
                "desired_WR must be a positive number",
                field_name="desired_WR",
                expected_type=float,
                actual_value=effective_desired_WR,
            )

        # Nominal payout: inflation is not applied inside RILA calc_annuity_payout,
        # so passing None is equivalent to passing obj_inflation here.
        nominal_payout: float = self.calc_annuity_payout(
            amount_principal,
            obj_scenarios,
            None,
        )
        nominal_WR: float = nominal_payout / amount_principal

        # Inverse inflation factor — defaults to 1.0 (no deflation) when
        # no inflation data is provided.
        inverse_inflation_factor: float = 1.0
        if obj_inflation is not None:
            required_cols = {
                "Start Date",
                "End Date",
                "Inflation Factor",
                "Inverse Inflation Factor",
            }
            if not required_cols.issubset(set(obj_inflation.columns)):
                missing = required_cols - set(obj_inflation.columns)
                raise Exception_Validation_Input(
                    f"obj_inflation is missing required columns: {missing}",
                    field_name="obj_inflation",
                    expected_type=pl.DataFrame,
                    actual_value=obj_inflation.columns,
                )
            inverse_inflation_factor = obj_inflation["Inverse Inflation Factor"].product()

        real_WR: float = nominal_WR * inverse_inflation_factor

        logger.info(
            f"RILA withdrawal rates — desired={effective_desired_WR:.2%}, "
            f"nominal={nominal_WR:.4%}, real={real_WR:.4%}, "
            f"inverse_inflation_factor={inverse_inflation_factor:.6f}",
        )

        return Withdrawal_Rates(nominal_WR=nominal_WR, real_WR=real_WR)

    def calc_monthly_payout(
        self,
        amount_principal: float,
        obj_scenarios: pl.DataFrame | None = None,
        obj_inflation: pl.DataFrame | None = None,
    ) -> float:
        """Calculate the monthly equivalent of the annual payout.

        Parameters
        ----------
        amount_principal : float
            The principal amount invested.
        obj_scenarios : object | None, optional
            Scenario data for payout calculations.
        obj_inflation : pl.DataFrame | None, optional
            Polars DataFrame with pre-calculated inflation factors (unused
            for RILA; kept for interface compatibility).  See
            :meth:`calc_annuity_payout` for the expected schema.

        Returns
        -------
        float
            The monthly equivalent payout amount.
        """
        annual_payout = self.calc_annuity_payout(
            amount_principal,
            obj_scenarios,
            obj_inflation,
        )
        return annual_payout / 12

    # ------------------------------------------------------------------
    # Worst-case and best-case outcome analysis
    # ------------------------------------------------------------------

    def calc_worst_case_account_value(
        self,
        amount_principal: float,
        num_terms: int = 1,
    ) -> float:
        r"""Calculate the worst-case account value assuming maximum loss each term.

        For buffer protection:

        $$
        V_{\text{worst}} = P \times (1 - (1 - b))^{n}
        $$

        For floor protection:

        $$
        V_{\text{worst}} = P \times (1 + f)^{n}
        $$

        Parameters
        ----------
        amount_principal : float
            The initial premium invested.
        num_terms : int, optional
            Number of consecutive worst-case terms (default 1).

        Returns
        -------
        float
            The worst-case account value.

        Raises
        ------
        Exception_Validation_Input
            If inputs are invalid.
        """
        if not isinstance(amount_principal, (int, float)) or amount_principal <= 0:
            raise Exception_Validation_Input(
                "amount_principal must be a positive number",
                field_name="amount_principal",
                expected_type=float,
                actual_value=amount_principal,
            )

        if not isinstance(num_terms, int) or num_terms <= 0:
            raise Exception_Validation_Input(
                "num_terms must be a positive integer",
                field_name="num_terms",
                expected_type=int,
                actual_value=num_terms,
            )

        if self.m_protection_type == Protection_Type.BUFFER:
            # Worst case: index drops beyond buffer (theoretically -100 %)
            # Maximum per-term loss = 1 - buffer_rate (i.e. -90 % for 10 % buffer)
            worst_term_return = -1.0  # Total index wipeout
            worst_credited = self.calc_credited_rate(worst_term_return)
        else:
            # Floor protection: worst-case credited = floor_rate
            worst_credited = self.m_floor_rate

        charge_factor = (1 - self.m_rate_rider_charge) ** self.m_term_years
        value = float(amount_principal)
        for _ in range(num_terms):
            value *= (1 + worst_credited) * charge_factor
            value = max(value, 0.0)

        return value

    def calc_best_case_account_value(
        self,
        amount_principal: float,
        num_terms: int = 1,
    ) -> float:
        r"""Calculate the best-case account value assuming maximum gain each term.

        $$
        V_{\text{best}} = P \times (1 + c)^{n}
        $$

        Parameters
        ----------
        amount_principal : float
            The initial premium invested.
        num_terms : int, optional
            Number of consecutive best-case terms (default 1).

        Returns
        -------
        float
            The best-case account value.

        Raises
        ------
        Exception_Validation_Input
            If inputs are invalid.
        """
        if not isinstance(amount_principal, (int, float)) or amount_principal <= 0:
            raise Exception_Validation_Input(
                "amount_principal must be a positive number",
                field_name="amount_principal",
                expected_type=float,
                actual_value=amount_principal,
            )

        if not isinstance(num_terms, int) or num_terms <= 0:
            raise Exception_Validation_Input(
                "num_terms must be a positive integer",
                field_name="num_terms",
                expected_type=int,
                actual_value=num_terms,
            )

        if self.m_crediting_strategy == Crediting_Strategy.PERFORMANCE_TRIGGER:
            best_credited = self.m_performance_trigger_rate
        elif self.m_crediting_strategy == Crediting_Strategy.PARTICIPATION_RATE:
            best_credited = min(
                self.m_participation_rate * 1.0,  # Assume 100 % index gain
                self.m_cap_rate,
            )
        else:
            best_credited = self.m_cap_rate

        charge_factor = (1 - self.m_rate_rider_charge) ** self.m_term_years
        value = float(amount_principal)
        for _ in range(num_terms):
            value *= (1 + best_credited) * charge_factor

        return value

    # ------------------------------------------------------------------
    # String representation
    # ------------------------------------------------------------------

    def get_annuity_as_string(self) -> str:
        """Return a human-readable string representation of the RILA.

        Returns
        -------
        str
            String representation of the Registered Index-Linked Annuity.
        """
        protection_detail = (
            f"Buffer {self.m_buffer_rate:.0%}"
            if self.m_protection_type == Protection_Type.BUFFER
            else f"Floor {self.m_floor_rate:.0%}"
        )

        crediting_detail = f"{self.m_crediting_strategy.value}"
        if self.m_crediting_strategy == Crediting_Strategy.CAP:
            crediting_detail += f" {self.m_cap_rate:.0%}"
        elif self.m_crediting_strategy == Crediting_Strategy.PERFORMANCE_TRIGGER:
            crediting_detail += f" {self.m_performance_trigger_rate:.0%}"
        elif self.m_crediting_strategy == Crediting_Strategy.PARTICIPATION_RATE:
            crediting_detail += f" {self.m_participation_rate:.0%} (cap {self.m_cap_rate:.0%})"

        riders = []
        if self.m_has_GMDB:
            riders.append("GMDB")
        if self.m_has_return_of_premium_death_benefit:
            riders.append("ROP-DB")
        rider_text = ", ".join(riders) if riders else "None"

        return (
            f"RILA: Age {self.m_client_age}, "
            f"Income at {self.m_age_income_start}, "
            f"Term {self.m_term_years}yr, "
            f"Protection: {protection_detail}, "
            f"Crediting: {crediting_detail}, "
            f"Riders: {rider_text}, "
            f"Index: {self.m_financial_index}, "
            f"Payout {self.m_annuity_payout_rate:.2%}, "
            f"Surrender {self.m_rate_surrender_charge_initial:.1%}/"
            f"{self.m_surrender_charge_schedule_years}yr, "
            f"Frequency {self.m_payment_frequency}/year"
        )
