"""
Variable Annuity (VA) class.

===============================

A Variable Annuity invests the premium in sub-accounts (mutual fund
portfolios) whose value fluctuates with the market.  Income guarantees
are layered on through optional riders at an additional cost.

Key characteristics of VAs:

- **Market participation**: account value varies with sub-account
  performance (equities, bonds, money-market, etc.).
- **Annual charges**: Mortality & Expense (M&E) charge, administrative
  fee, and optional rider charges are deducted from the account value.
- **Guaranteed Minimum Death Benefit (GMDB)**: ensures beneficiaries
  receive at least the premium paid (or a stepped-up value).
- **Guaranteed Lifetime Withdrawal Benefit (GLWB)**: guarantees a
  lifetime income stream as a percentage of the benefit base.
- **Rollup benefit**: the benefit base grows at a guaranteed rollup
  rate during deferral regardless of market performance.
- **Ratchet (step-up)**: locks in market gains by resetting the
  benefit base to the account value when it exceeds the rollup base.
- **Surrender charges**: declining schedule that penalises early
  withdrawals beyond the annual free-withdrawal allowance.

Author
------
QWIM Team

Version
-------
0.5.1 (2026-03-01)
"""

from __future__ import annotations

import polars as pl

from src.utils.custom_exceptions_errors_loggers.exception_custom import (
    Exception_Calculation,
    Exception_Validation_Input,
)
from src.utils.custom_exceptions_errors_loggers.logger_custom import get_logger

from .annuity_base import Annuity_Base, Annuity_Type, Withdrawal_Rates


logger = get_logger(__name__)


class Annuity_VA(Annuity_Base):
    r"""Variable Annuity (VA).

    A Variable Annuity invests the premium in market-linked sub-accounts.
    Income guarantees are provided through optional riders (GMDB, GLWB)
    at an additional annual cost.  The benefit base grows via rollup and/or
    ratchet mechanisms.

    The account value after annual charges is:

    $$
    V_t = V_{t-1} \times (1 + R_{\text{gross}} - c_{\text{ME}}
          - c_{\text{admin}} - c_{\text{rider}})
    $$

    Where:

    - $R_{\text{gross}}$ = gross sub-account return
    - $c_{\text{ME}}$ = mortality & expense charge
    - $c_{\text{admin}}$ = administrative fee
    - $c_{\text{rider}}$ = total rider charges

    The benefit base with rollup is:

    $$
    B = A \times (1 + r_{\text{rollup}})^{n}
    $$

    The GLWB payout is:

    $$
    W = B \times w\%
    $$

    Where $w\%$ = guaranteed withdrawal percentage (age-banded).

    Attributes
    ----------
    m_client_age : int
        The current age of the client (inherited).
    m_annuity_payout_rate : float
        The payout rate of the annuity (inherited).
    m_annuity_type : Annuity_Type
        Set to ``ANNUITY_VA`` (inherited).
    m_age_income_start : int
        The age at which income payments begin.
    m_age_max_ratchet : int
        The maximum age until which the ratchet feature applies.
    m_rate_rollup_benefit : float
        The annual rollup rate applied to the benefit base during deferral.
    m_rate_ME_charge : float
        Annual mortality & expense (M&E) charge as a decimal.
    m_rate_admin_fee : float
        Annual administrative fee as a decimal.
    m_rate_rider_charge : float
        Total annual rider charge (GMDB + GLWB etc.) as a decimal.
    m_rate_surrender_charge_initial : float
        Initial surrender charge rate (year 0).
    m_surrender_charge_schedule_years : int
        Number of years over which the surrender charge declines to zero.
    m_pct_free_withdrawal : float
        Annual free-withdrawal percentage.
    m_has_GMDB : bool
        Whether the contract includes a Guaranteed Minimum Death Benefit.
    m_has_GLWB : bool
        Whether the contract includes a Guaranteed Lifetime Withdrawal Benefit.
    m_pct_GLWB : float
        GLWB annual withdrawal percentage applied to the benefit base.
    m_payment_frequency : int
        Number of payments per year.
    m_market_proxy : str
        Name of the market proxy column in ``obj_scenarios``.
    """

    def __init__(
        self,
        client_age: int,
        annuity_payout_rate: float,
        age_income_start: int,
        age_max_ratchet: int,
        rate_rollup_benefit: float,
        rate_ME_charge: float = 0.0125,
        rate_admin_fee: float = 0.0015,
        rate_rider_charge: float = 0.0100,
        rate_surrender_charge_initial: float = 0.07,
        surrender_charge_schedule_years: int = 7,
        pct_free_withdrawal: float = 0.10,
        has_GMDB: bool = True,
        has_GLWB: bool = True,
        pct_GLWB: float = 0.05,
        payment_frequency: int = 12,
        market_proxy: str = "Equity",
        annuity_income_starting_age: int | None = None,
    ) -> None:
        """Initialize a Variable Annuity object.

        Parameters
        ----------
        client_age : int
            The current age of the client.
        annuity_payout_rate : float
            The annual payout rate of the annuity (as a decimal).
        age_income_start : int
            The age at which income payments begin.
        age_max_ratchet : int
            The maximum age until which the ratchet feature applies.
        rate_rollup_benefit : float
            The annual rollup rate applied to the benefit base.
        rate_ME_charge : float, optional
            Annual mortality & expense charge (default 0.0125 = 1.25 %).
        rate_admin_fee : float, optional
            Annual administrative fee (default 0.0015 = 0.15 %).
        rate_rider_charge : float, optional
            Annual rider charge for GMDB/GLWB (default 0.0100 = 1.00 %).
        rate_surrender_charge_initial : float, optional
            Initial surrender charge rate (default 0.07 = 7 %).
        surrender_charge_schedule_years : int, optional
            Years for surrender charge to decline to zero (default 7).
        pct_free_withdrawal : float, optional
            Annual free-withdrawal percentage (default 0.10 = 10 %).
        has_GMDB : bool, optional
            Include Guaranteed Minimum Death Benefit (default ``True``).
        has_GLWB : bool, optional
            Include Guaranteed Lifetime Withdrawal Benefit (default ``True``).
        pct_GLWB : float, optional
            GLWB withdrawal percentage of benefit base (default 0.05 = 5 %).
        payment_frequency : int, optional
            Number of payments per year (default 12 — monthly).
        market_proxy : str, optional
            Name of the column in ``obj_scenarios`` containing gross market
            returns.  Defaults to ``"Equity"``.
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
            Annuity_Type.ANNUITY_VA,
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

        if not isinstance(age_max_ratchet, int) or age_max_ratchet < client_age:
            raise Exception_Validation_Input(
                "age_max_ratchet must be >= client_age",
                field_name="age_max_ratchet",
                expected_type=int,
                actual_value=age_max_ratchet,
            )

        if not isinstance(rate_rollup_benefit, (int, float)) or rate_rollup_benefit < 0:
            raise Exception_Validation_Input(
                "rate_rollup_benefit must be a non-negative number",
                field_name="rate_rollup_benefit",
                expected_type=float,
                actual_value=rate_rollup_benefit,
            )

        if not isinstance(rate_ME_charge, (int, float)) or rate_ME_charge < 0:
            raise Exception_Validation_Input(
                "rate_ME_charge must be a non-negative number",
                field_name="rate_ME_charge",
                expected_type=float,
                actual_value=rate_ME_charge,
            )

        if not isinstance(rate_admin_fee, (int, float)) or rate_admin_fee < 0:
            raise Exception_Validation_Input(
                "rate_admin_fee must be a non-negative number",
                field_name="rate_admin_fee",
                expected_type=float,
                actual_value=rate_admin_fee,
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

        if not isinstance(pct_GLWB, (int, float)) or pct_GLWB < 0 or pct_GLWB > 1:
            raise Exception_Validation_Input(
                "pct_GLWB must be between 0 and 1",
                field_name="pct_GLWB",
                expected_type=float,
                actual_value=pct_GLWB,
            )

        if not isinstance(payment_frequency, int) or payment_frequency <= 0:
            raise Exception_Validation_Input(
                "payment_frequency must be a positive integer",
                field_name="payment_frequency",
                expected_type=int,
                actual_value=payment_frequency,
            )

        if not isinstance(market_proxy, str) or not market_proxy.strip():
            raise Exception_Validation_Input(
                "market_proxy must be a non-empty string",
                field_name="market_proxy",
                expected_type=str,
                actual_value=market_proxy,
            )

        # --- Set member variables ---
        self.m_age_income_start: int = age_income_start
        self.m_age_max_ratchet: int = age_max_ratchet
        self.m_rate_rollup_benefit: float = float(rate_rollup_benefit)
        self.m_rate_ME_charge: float = float(rate_ME_charge)
        self.m_rate_admin_fee: float = float(rate_admin_fee)
        self.m_rate_rider_charge: float = float(rate_rider_charge)
        self.m_rate_surrender_charge_initial: float = float(rate_surrender_charge_initial)
        self.m_surrender_charge_schedule_years: int = surrender_charge_schedule_years
        self.m_pct_free_withdrawal: float = float(pct_free_withdrawal)
        self.m_has_GMDB: bool = has_GMDB
        self.m_has_GLWB: bool = has_GLWB
        self.m_pct_GLWB: float = float(pct_GLWB)
        self.m_payment_frequency: int = payment_frequency
        self.m_market_proxy: str = market_proxy

        logger.info(
            f"Created VA with payout rate {annuity_payout_rate:.2%}, "
            f"income at age {age_income_start}, "
            f"rollup: {rate_rollup_benefit:.2%}, "
            f"M&E: {rate_ME_charge:.2%}, admin: {rate_admin_fee:.2%}, "
            f"rider: {rate_rider_charge:.2%}, "
            f"GMDB: {has_GMDB}, GLWB: {has_GLWB} ({pct_GLWB:.2%}), "
            f"surrender: {rate_surrender_charge_initial:.1%}/"
            f"{surrender_charge_schedule_years}yr, "
            f"market proxy: {market_proxy!r}",
        )

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def income_start_age(self) -> int:
        """Int : The age at which income payments begin."""
        return self.m_age_income_start

    @property
    def max_ratchet_age(self) -> int:
        """Int : The maximum age until which the ratchet feature applies."""
        return self.m_age_max_ratchet

    @property
    def rollup_rate(self) -> float:
        """Float : The annual rollup rate for the benefit base."""
        return self.m_rate_rollup_benefit

    @property
    def payment_frequency(self) -> int:
        """Int : Number of payments per year."""
        return self.m_payment_frequency

    @property
    def deferral_years(self) -> int:
        """Int : Number of years until income payments begin."""
        return self.m_age_income_start - self.m_client_age

    @property
    def rate_ME_charge(self) -> float:
        """Float : Annual mortality & expense charge."""
        return self.m_rate_ME_charge

    @property
    def rate_admin_fee(self) -> float:
        """Float : Annual administrative fee."""
        return self.m_rate_admin_fee

    @property
    def rate_rider_charge(self) -> float:
        """Float : Total annual rider charge."""
        return self.m_rate_rider_charge

    @property
    def has_GMDB(self) -> bool:
        """Bool : Whether the contract includes a GMDB."""
        return self.m_has_GMDB

    @property
    def has_GLWB(self) -> bool:
        """Bool : Whether the contract includes a GLWB."""
        return self.m_has_GLWB

    @property
    def pct_GLWB(self) -> float:
        """Float : GLWB withdrawal percentage applied to benefit base."""
        return self.m_pct_GLWB

    @property
    def pct_free_withdrawal(self) -> float:
        """Float : Annual free-withdrawal percentage."""
        return self.m_pct_free_withdrawal

    @property
    def market_proxy(self) -> str:
        """Str : Name of the market proxy column used in scenario data."""
        return self.m_market_proxy

    # ------------------------------------------------------------------
    # Charge calculations
    # ------------------------------------------------------------------

    def calc_total_annual_charges(self) -> float:
        r"""Calculate the total annual charge rate.

        $$
        c_{\text{total}} = c_{\text{ME}} + c_{\text{admin}} + c_{\text{rider}}
        $$

        Returns
        -------
        float
            The combined annual charge rate as a decimal.
        """
        return self.m_rate_ME_charge + self.m_rate_admin_fee + self.m_rate_rider_charge

    def calc_account_value_after_charges(
        self,
        amount_principal: float,
        gross_returns: list[float],
    ) -> list[float]:
        r"""Simulate the account value net of annual charges.

        $$
        V_t = V_{t-1} \times (1 + R_t - c_{\text{total}})
        $$

        Parameters
        ----------
        amount_principal : float
            The initial premium invested.
        gross_returns : list[float]
            A list of gross annual sub-account returns.

        Returns
        -------
        list[float]
            Account values at the end of each period, starting with the
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

        if not isinstance(gross_returns, list):
            raise Exception_Validation_Input(
                "gross_returns must be a list of numbers",
                field_name="gross_returns",
                expected_type=list,
                actual_value=gross_returns,
            )

        total_charge = self.calc_total_annual_charges()
        values: list[float] = [float(amount_principal)]
        current_value = float(amount_principal)

        for gross_return in gross_returns:
            net_return = gross_return - total_charge
            current_value *= 1 + net_return
            current_value = max(current_value, 0.0)  # Account value cannot go negative
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

        Parameters
        ----------
        year : int
            The contract year (0-indexed; year 0 = purchase year).

        Returns
        -------
        float
            The surrender charge rate as a decimal.
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
    # Benefit base and rider calculations
    # ------------------------------------------------------------------

    def calc_benefit_base_with_rollup(
        self,
        amount_principal: float,
        years_deferred: int | None = None,
    ) -> float:
        r"""Calculate the benefit base after applying the rollup rate.

        $$
        B = A \times (1 + r_{\text{rollup}})^{n}
        $$

        Parameters
        ----------
        amount_principal : float
            The principal amount invested in the annuity.
        years_deferred : int | None, optional
            Number of years to apply the rollup (default: :attr:`deferral_years`).

        Returns
        -------
        float
            The benefit base after applying the rollup.

        Raises
        ------
        Exception_Validation_Input
            If ``amount_principal`` is not positive or ``years_deferred``
            is not a non-negative integer.
        """
        if not isinstance(amount_principal, (int, float)) or amount_principal <= 0:
            raise Exception_Validation_Input(
                "amount_principal must be a positive number",
                field_name="amount_principal",
                expected_type=float,
                actual_value=amount_principal,
            )

        if years_deferred is None:
            years_deferred = self.deferral_years

        if not isinstance(years_deferred, int) or years_deferred < 0:
            raise Exception_Validation_Input(
                "years_deferred must be a non-negative integer",
                field_name="years_deferred",
                expected_type=int,
                actual_value=years_deferred,
            )

        benefit_base: float = amount_principal * (
            (1 + self.m_rate_rollup_benefit) ** years_deferred
        )
        return benefit_base

    def calc_death_benefit(
        self,
        amount_principal: float,
        account_value: float,
    ) -> float:
        r"""Calculate the Guaranteed Minimum Death Benefit (GMDB).

        The return-of-premium GMDB is:

        $$
        DB = \max(V_{\text{account}},\; A)
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
            The death benefit amount.  If GMDB is not elected, returns
            the account value.

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

        if not self.m_has_GMDB:
            return account_value

        return max(account_value, amount_principal)

    def calc_GLWB_payout(
        self,
        amount_principal: float,
        years_deferred: int | None = None,
    ) -> float:
        r"""Calculate the Guaranteed Lifetime Withdrawal Benefit payout.

        The annual GLWB payout is a guaranteed percentage of the benefit base:

        $$
        W = B \times w\%
        $$

        Parameters
        ----------
        amount_principal : float
            The initial premium invested.
        years_deferred : int | None, optional
            Override for deferral years (default: :attr:`deferral_years`).

        Returns
        -------
        float
            The annual GLWB payout.  Returns ``0.0`` if GLWB is not
            elected.

        Raises
        ------
        Exception_Validation_Input
            If ``amount_principal`` is not positive.
        Exception_Calculation
            If payout calculation produces an invalid result.
        """
        if not self.m_has_GLWB:
            return 0.0

        if not isinstance(amount_principal, (int, float)) or amount_principal <= 0:
            raise Exception_Validation_Input(
                "amount_principal must be a positive number",
                field_name="amount_principal",
                expected_type=float,
                actual_value=amount_principal,
            )

        benefit_base = self.calc_benefit_base_with_rollup(
            amount_principal,
            years_deferred,
        )
        payout = benefit_base * self.m_pct_GLWB

        if payout < 0:
            raise Exception_Calculation(
                "GLWB payout calculation produced a negative value",
            )

        logger.info(f"GLWB payout: {payout:.2f} (base: {benefit_base:.2f})")
        return payout

    # ------------------------------------------------------------------
    # Payout calculations
    # ------------------------------------------------------------------

    def calc_annuity_payout(
        self,
        amount_principal: float,
        obj_scenarios: pl.DataFrame | None = None,
        obj_inflation: pl.DataFrame | None = None,
    ) -> float:
        """Calculate the variable annuity annual payout.

        The calculation depends on:

        1. Whether the client has reached the income start age.
        2. The benefit base (rollup or ratchet-enhanced).
        3. Market performance applied through scenario data.
        4. The payout rate applied to the benefit base.

        Parameters
        ----------
        amount_principal : float
            The principal amount invested.
        obj_scenarios : pl.DataFrame | None, optional
            Polars DataFrame containing scenario market data.  The first column
            must be ``Date``; the column named ``self.m_market_proxy`` must
            contain gross sub-account returns used to simulate the account value.
        obj_inflation : pl.DataFrame | None, optional
            Polars DataFrame with pre-calculated inflation factors (unused
            for VA; kept for interface compatibility).  Expected schema:

            - ``Start Date`` : start of the inflation period.
            - ``End Date``   : end of the inflation period.
            - ``Inflation Factor`` : cumulative inflation already calculated
              for the interval ``[Start Date, End Date]``
            - ``Inverse Inflation Factor`` : inverse of cumulative inflation already calculated
              for the interval ``[Start Date, End Date]``.

        Returns
        -------
        float
            The calculated annual payout.  Returns ``0.0`` while in deferral.

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

        benefit_base: float = amount_principal

        if obj_scenarios is not None:
            required_cols = {"Date", self.m_market_proxy}
            if not required_cols.issubset(set(obj_scenarios.columns)):
                missing = required_cols - set(obj_scenarios.columns)
                raise Exception_Validation_Input(
                    f"obj_scenarios is missing required columns: {missing}",
                    field_name="obj_scenarios",
                    expected_type=pl.DataFrame,
                    actual_value=obj_scenarios.columns,
                )
            gross_returns = obj_scenarios[self.m_market_proxy].to_list()
            account_values = self.calc_account_value_after_charges(
                amount_principal,
                gross_returns,
            )
            market_account_value = account_values[-1] if account_values else amount_principal
            if self.m_client_age <= self.m_age_max_ratchet:
                rollup_base = self.calc_benefit_base_with_rollup(amount_principal)
                benefit_base = max(rollup_base, market_account_value)
            else:
                benefit_base = market_account_value
            logger.info(f"VA benefit base after market scenario: {benefit_base:.2f}")
        else:
            benefit_base = self.calc_benefit_base_with_rollup(amount_principal)
            logger.info(f"VA benefit base from rollup: {benefit_base:.2f}")

        annual_payout: float = benefit_base * self.m_annuity_payout_rate
        logger.info(f"VA annual payout: {annual_payout:.2f}")

        return annual_payout

    def calc_withdrawal_rates(
        self,
        amount_principal: float,
        desired_WR: float | None = None,
        obj_scenarios: pl.DataFrame | None = None,
        obj_inflation: pl.DataFrame | None = None,
    ) -> Withdrawal_Rates:
        r"""Calculate nominal and real withdrawal rates for this VA.

        The nominal withdrawal rate is the ratio of the VA payout
        (benefit-base driven, market-scenario-adjusted if scenarios are
        supplied) to the invested principal.  The real withdrawal rate
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
            column must be ``Date``; the column named ``self.m_market_proxy``
            must contain gross sub-account returns.  See
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

        # Nominal payout: inflation is not applied inside VA calc_annuity_payout,
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
            f"VA withdrawal rates — desired={effective_desired_WR:.2%}, "
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
            Scenario data used for payout calculations.
        obj_inflation : pl.DataFrame | None, optional
            Polars DataFrame with pre-calculated inflation factors (unused
            for VA; kept for interface compatibility).  See
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
    # String representation
    # ------------------------------------------------------------------

    def get_annuity_as_string(self) -> str:
        """Return a human-readable string representation of the VA.

        Returns
        -------
        str
            String representation of the Variable Annuity.
        """
        riders = []
        if self.m_has_GMDB:
            riders.append("GMDB")
        if self.m_has_GLWB:
            riders.append(f"GLWB({self.m_pct_GLWB:.1%})")
        rider_text = ", ".join(riders) if riders else "None"

        return (
            f"VA: Age {self.m_client_age}, "
            f"Income at {self.m_age_income_start}, "
            f"Ratchet to {self.m_age_max_ratchet}, "
            f"Rollup {self.m_rate_rollup_benefit:.2%}, "
            f"M&E {self.m_rate_ME_charge:.2%}, "
            f"Admin {self.m_rate_admin_fee:.2%}, "
            f"Riders: {rider_text}, "
            f"Market Proxy: {self.m_market_proxy}, "
            f"Payout {self.m_annuity_payout_rate:.2%}, "
            f"Surrender {self.m_rate_surrender_charge_initial:.1%}/"
            f"{self.m_surrender_charge_schedule_years}yr, "
            f"Frequency {self.m_payment_frequency}/year"
        )
