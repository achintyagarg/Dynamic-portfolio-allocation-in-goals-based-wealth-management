"""
Fixed Indexed Annuity (FIA) class.

===============================

A Fixed Indexed Annuity provides returns linked to a market index
(e.g. S&P 500) with downside protection and guaranteed minimums.
Key characteristics:

- **Floor rate**: a minimum credited rate (typically 0 %) that protects
  against index losses.
- **Cap rate**: limits the maximum index return credited in any period.
- **Participation rate**: the fraction of index growth credited to the
  account.
- **Spread / margin**: a fee deducted from the index return before
  crediting.
- **Rollup benefit**: a guaranteed growth rate applied to the income
  benefit base during the deferral period.
- **Ratchet**: locks in market gains by resetting the benefit base to
  the higher of the current value or the rollup value.
- **Surrender charge schedule**: a declining penalty for early withdrawal.
- **Minimum guaranteed value**: typically 87.5 % of premium at a
  minimum guaranteed interest rate (state-regulated).
- **Free withdrawal**: usually 10 % of the account value may be withdrawn
  each year without surrender charges.

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
    Exception_Validation_Input,
)
from src.utils.custom_exceptions_errors_loggers.logger_custom import get_logger

from .annuity_base import Annuity_Base, Annuity_Type, Withdrawal_Rates


logger = get_logger(__name__)


class Annuity_FIA(Annuity_Base):
    r"""Fixed Indexed Annuity (FIA).

    Returns are linked to a market index with downside protection.
    The credited rate for each period is:

    $$
    r_{\text{credited}} = \max\!\Bigl(
        \min\bigl((\max(R_{\text{index}}, 0) - s) \times p,\; c\bigr),\; f
    \Bigr)
    $$

    Where:

    - $R_{\text{index}}$ = raw index return for the period
    - $s$ = spread / margin
    - $p$ = participation rate
    - $c$ = cap rate
    - $f$ = floor rate (typically 0 %)

    The benefit base with rollup is:

    $$
    B = A \times (1 + r_{\text{rollup}})^{n}
    $$

    The minimum guaranteed value (state-regulated) is:

    $$
    V_{\min} = A \times g \times (1 + r_{\min})^{n}
    $$

    Where $g$ = guarantee base percentage (e.g. 0.875) and
    $r_{\min}$ = minimum guaranteed interest rate.

    Attributes
    ----------
    m_client_age : int
        The current age of the client (inherited).
    m_annuity_payout_rate : float
        The payout rate of the annuity (inherited).
    m_annuity_type : Annuity_Type
        Set to ``ANNUITY_FIA`` (inherited).
    m_age_income_start : int
        The age at which income payments begin.
    m_age_max_ratchet : int
        The maximum age until which the ratchet feature applies.
    m_rate_rollup_benefit : float
        The annual rollup rate applied to the benefit base during deferral.
    m_cap_rate : float
        Maximum annual return that can be credited.
    m_participation_rate : float
        Fraction of index growth credited to the account.
    m_spread_rate : float
        Spread / margin deducted from the index return before crediting.
    m_floor_rate : float
        Minimum credited rate per period (typically 0.0).
    m_rate_surrender_charge_initial : float
        Initial surrender charge rate (year 0).
    m_surrender_charge_schedule_years : int
        Number of years over which the surrender charge declines to zero.
    m_rate_minimum_guarantee : float
        Minimum guaranteed annual interest rate on the guarantee base.
    m_pct_minimum_guarantee_base : float
        Fraction of premium used as the guarantee base (e.g. 0.875).
    m_pct_free_withdrawal : float
        Annual free-withdrawal percentage (e.g. 0.10 for 10 %).
    m_payment_frequency : int
        Number of payments per year.
    m_financial_index : str
        Name of the financial index column in ``obj_scenarios`` used for
        crediting calculations (e.g. ``"S&P 500"``).  The
        :meth:`calc_annuity_payout` method validates that this column
        exists in the supplied DataFrame.
    """

    def __init__(
        self,
        client_age: int,
        annuity_payout_rate: float,
        age_income_start: int,
        age_max_ratchet: int,
        rate_rollup_benefit: float,
        cap_rate: float = 0.05,
        participation_rate: float = 1.0,
        spread_rate: float = 0.0,
        floor_rate: float = 0.0,
        rate_surrender_charge_initial: float = 0.08,
        surrender_charge_schedule_years: int = 7,
        rate_minimum_guarantee: float = 0.01,
        pct_minimum_guarantee_base: float = 0.875,
        pct_free_withdrawal: float = 0.10,
        payment_frequency: int = 12,
        financial_index: str = "S&P 500",
        annuity_income_starting_age: int | None = None,
    ) -> None:
        """Initialize a Fixed Indexed Annuity object.

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
        cap_rate : float, optional
            Maximum annual return that can be credited (default 0.05).
        participation_rate : float, optional
            Fraction of index growth credited (default 1.0 = 100 %).
        spread_rate : float, optional
            Spread / margin deducted from index return (default 0.0).
        floor_rate : float, optional
            Minimum credited rate per period (default 0.0).
        rate_surrender_charge_initial : float, optional
            Initial surrender charge rate (default 0.08 = 8 %).
        surrender_charge_schedule_years : int, optional
            Years for surrender charge to decline to zero (default 7).
        rate_minimum_guarantee : float, optional
            Minimum guaranteed annual interest rate (default 0.01 = 1 %).
        pct_minimum_guarantee_base : float, optional
            Fraction of premium used as guarantee base (default 0.875).
        pct_free_withdrawal : float, optional
            Annual free-withdrawal percentage (default 0.10 = 10 %).
        payment_frequency : int, optional
            Number of payments per year (default 12 — monthly).
        financial_index : str, optional
            Name of the financial index column in ``obj_scenarios``
            (default ``"S&P 500"``).  Must be a non-empty string.
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
            Annuity_Type.ANNUITY_FIA,
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
            or participation_rate > 1
        ):
            raise Exception_Validation_Input(
                "participation_rate must be between 0 and 1",
                field_name="participation_rate",
                expected_type=float,
                actual_value=participation_rate,
            )

        if not isinstance(spread_rate, (int, float)) or spread_rate < 0:
            raise Exception_Validation_Input(
                "spread_rate must be a non-negative number",
                field_name="spread_rate",
                expected_type=float,
                actual_value=spread_rate,
            )

        if not isinstance(floor_rate, (int, float)):
            raise Exception_Validation_Input(
                "floor_rate must be a number",
                field_name="floor_rate",
                expected_type=float,
                actual_value=floor_rate,
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
        self.m_age_max_ratchet: int = age_max_ratchet
        self.m_rate_rollup_benefit: float = float(rate_rollup_benefit)
        self.m_cap_rate: float = float(cap_rate)
        self.m_participation_rate: float = float(participation_rate)
        self.m_spread_rate: float = float(spread_rate)
        self.m_floor_rate: float = float(floor_rate)
        self.m_rate_surrender_charge_initial: float = float(rate_surrender_charge_initial)
        self.m_surrender_charge_schedule_years: int = surrender_charge_schedule_years
        self.m_rate_minimum_guarantee: float = float(rate_minimum_guarantee)
        self.m_pct_minimum_guarantee_base: float = float(pct_minimum_guarantee_base)
        self.m_pct_free_withdrawal: float = float(pct_free_withdrawal)
        self.m_payment_frequency: int = payment_frequency
        self.m_financial_index: str = financial_index

        logger.info(
            f"Created FIA with payout rate {annuity_payout_rate:.2%}, "
            f"income at age {age_income_start}, "
            f"cap: {cap_rate:.2%}, participation: {participation_rate:.2%}, "
            f"spread: {spread_rate:.2%}, floor: {floor_rate:.2%}, "
            f"rollup: {rate_rollup_benefit:.2%}, "
            f"surrender: {rate_surrender_charge_initial:.2%} "
            f"over {surrender_charge_schedule_years}yr, "
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
    def max_ratchet_age(self) -> int:
        """Int : The maximum age until which the ratchet feature applies."""
        return self.m_age_max_ratchet

    @property
    def rollup_rate(self) -> float:
        """Float : The annual rollup rate for the benefit base."""
        return self.m_rate_rollup_benefit

    @property
    def cap_rate(self) -> float:
        """Float : The cap rate for index returns."""
        return self.m_cap_rate

    @property
    def participation_rate(self) -> float:
        """Float : The participation rate for index returns."""
        return self.m_participation_rate

    @property
    def spread_rate(self) -> float:
        """Float : The spread / margin deducted from the index return."""
        return self.m_spread_rate

    @property
    def floor_rate(self) -> float:
        """Float : The minimum credited rate per period."""
        return self.m_floor_rate

    @property
    def payment_frequency(self) -> int:
        """Int : Number of payments per year."""
        return self.m_payment_frequency

    @property
    def deferral_years(self) -> int:
        """Int : Number of years until income payments begin."""
        return self.m_age_income_start - self.m_client_age

    @property
    def pct_free_withdrawal(self) -> float:
        """Float : Annual free-withdrawal percentage."""
        return self.m_pct_free_withdrawal

    @property
    def financial_index(self) -> str:
        """Str : Name of the financial index column used in scenario crediting."""
        return self.m_financial_index

    # ------------------------------------------------------------------
    # Benefit base and crediting calculations
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

    def calc_credited_rate(
        self,
        index_return: float,
    ) -> float:
        r"""Calculate the credited rate based on the index return.

        Applies floor, spread, participation rate, and cap:

        $$
        r_{\text{credited}} = \max\!\Bigl(
            \min\bigl((\max(R, 0) - s) \times p,\; c\bigr),\; f
        \Bigr)
        $$

        Parameters
        ----------
        index_return : float
            The return of the referenced index for the period.

        Returns
        -------
        float
            The credited rate after applying all FIA crediting rules.

        Raises
        ------
        Exception_Validation_Input
            If ``index_return`` is not a number.
        """
        if not isinstance(index_return, (int, float)):
            raise Exception_Validation_Input(
                "index_return must be a number",
                field_name="index_return",
                expected_type=float,
                actual_value=index_return,
            )

        # Step 1: Apply floor on raw index return (downside protection)
        floored_return = max(index_return, 0.0)

        # Step 2: Deduct spread / margin
        net_return = floored_return - self.m_spread_rate

        # Step 3: Apply participation rate
        participated_return = max(net_return, 0.0) * self.m_participation_rate

        # Step 4: Apply cap
        capped_return = min(participated_return, self.m_cap_rate)

        # Step 5: Apply floor rate (guaranteed minimum crediting)
        credited_rate: float = max(capped_return, self.m_floor_rate)

        return credited_rate

    def calc_account_value(
        self,
        amount_principal: float,
        index_returns: list[float],
    ) -> list[float]:
        r"""Simulate the account value over multiple periods.

        For each period the account value grows by the credited rate:

        $$
        V_t = V_{t-1} \times (1 + r_{\text{credited}, t})
        $$

        Parameters
        ----------
        amount_principal : float
            The initial premium invested.
        index_returns : list[float]
            A list of index returns for each period (e.g. annual).

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

        if not isinstance(index_returns, list):
            raise Exception_Validation_Input(
                "index_returns must be a list of numbers",
                field_name="index_returns",
                expected_type=list,
                actual_value=index_returns,
            )

        values: list[float] = [float(amount_principal)]
        current_value = float(amount_principal)

        for year_return in index_returns:
            credited = self.calc_credited_rate(year_return)
            current_value *= 1 + credited
            values.append(current_value)

        return values

    # ------------------------------------------------------------------
    # Guarantee and surrender calculations
    # ------------------------------------------------------------------

    def calc_minimum_guaranteed_value(
        self,
        amount_principal: float,
        years: int,
    ) -> float:
        r"""Calculate the minimum guaranteed value.

        State-regulated minimum, typically 87.5 % of premium compounded
        at the minimum guaranteed rate:

        $$
        V_{\min} = A \times g \times (1 + r_{\min})^{n}
        $$

        Parameters
        ----------
        amount_principal : float
            The premium invested.
        years : int
            Number of years since purchase.

        Returns
        -------
        float
            The minimum guaranteed value.

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

        if not isinstance(years, int) or years < 0:
            raise Exception_Validation_Input(
                "years must be a non-negative integer",
                field_name="years",
                expected_type=int,
                actual_value=years,
            )

        return (
            amount_principal
            * self.m_pct_minimum_guarantee_base
            * ((1 + self.m_rate_minimum_guarantee) ** years)
        )

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

        Surrender value = account value minus the applicable surrender
        charge.  The free-withdrawal amount is not subject to the charge.

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

        # The free-withdrawal portion is exempt from surrender charges
        free_amount = account_value * self.m_pct_free_withdrawal
        excess_amount = max(account_value - free_amount, 0.0)

        surrender_charge = excess_amount * charge_rate
        return account_value - surrender_charge

    # ------------------------------------------------------------------
    # Payout calculations
    # ------------------------------------------------------------------

    def calc_annuity_payout(
        self,
        amount_principal: float,
        obj_scenarios: pl.DataFrame | None = None,
        obj_inflation: pl.DataFrame | None = None,
    ) -> float:
        """Calculate the FIA annual payout.

        The calculation depends on:

        1. Whether the client has reached the income start age.
        2. The benefit base (rollup or ratchet-enhanced).
        3. Index performance from ``obj_scenarios`` subject to participation
           rate, spread, and cap.
        4. The payout rate applied to the benefit base.

        Parameters
        ----------
        amount_principal : float
            The principal amount invested.
        obj_scenarios : pl.DataFrame | None, optional
            Polars DataFrame containing scenario market data.  Must have a
            ``Date`` column and a column named :attr:`financial_index`
            (default ``"S&P 500"``) containing periodic index returns.
            When provided, the account value is simulated using
            :meth:`calc_account_value` and the ratchet mechanism is applied.
        obj_inflation : pl.DataFrame | None, optional
            Polars DataFrame with pre-calculated inflation factors (unused
            for FIA; kept for interface compatibility).  Expected schema:

            - ``Start Date`` : start of the inflation period.
            - ``End Date``   : end of the inflation period.
            - ``Inflation Factor`` : cumulative inflation already calculated
              for the interval ``[Start Date, End Date]``
            - ``Inverse Inflation Factor`` : inverse of cumulative inflation already calculated
              for the interval ``[Start Date, End Date]``.

        Returns
        -------
        float
            The calculated annual payout. Returns ``0.0`` while in deferral.

        Raises
        ------
        Exception_Validation_Input
            If ``amount_principal`` is not a positive number.
            If ``obj_scenarios`` is provided but missing required columns.
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
            required_cols = {"Date", self.m_financial_index}
            if not required_cols.issubset(set(obj_scenarios.columns)):
                missing = required_cols - set(obj_scenarios.columns)
                raise Exception_Validation_Input(
                    f"obj_scenarios is missing required columns: {missing}",
                    field_name="obj_scenarios",
                    expected_type=pl.DataFrame,
                    actual_value=obj_scenarios.columns,
                )
            index_returns = obj_scenarios[self.m_financial_index].to_list()
            account_values = self.calc_account_value(
                amount_principal,
                index_returns,
            )
            account_value = account_values[-1] if account_values else amount_principal

            # Apply ratchet: benefit base = max(rollup, account value)
            if self.m_client_age <= self.m_age_max_ratchet:
                rollup_base = self.calc_benefit_base_with_rollup(amount_principal)
                benefit_base = max(rollup_base, account_value)
            else:
                benefit_base = account_value

            logger.info(
                f"FIA benefit base after index crediting: {benefit_base:.2f}",
            )
        else:
            # Without scenarios, use rollup calculation for the deferral period
            benefit_base = self.calc_benefit_base_with_rollup(amount_principal)
            logger.info(f"FIA benefit base from rollup: {benefit_base:.2f}")

        annual_payout: float = benefit_base * self.m_annuity_payout_rate
        logger.info(f"FIA annual payout: {annual_payout:.2f}")

        return annual_payout

    def calc_withdrawal_rates(
        self,
        amount_principal: float,
        desired_WR: float | None = None,
        obj_scenarios: pl.DataFrame | None = None,
        obj_inflation: pl.DataFrame | None = None,
    ) -> Withdrawal_Rates:
        r"""Calculate nominal and real withdrawal rates for this FIA.

        The nominal withdrawal rate is the ratio of the FIA payout
        (benefit-base driven, index-credited if scenarios are supplied)
        to the invested principal.  The real withdrawal rate deflates the
        nominal rate by the cumulative inverse inflation factor:

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
            Polars DataFrame containing scenario market data.  Must have a
            ``Date`` column and a column named :attr:`financial_index`
            containing periodic index returns.  See :meth:`calc_annuity_payout`
            for full schema details.
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

        # Nominal payout: inflation is not applied inside FIA calc_annuity_payout,
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
            f"FIA withdrawal rates — desired={effective_desired_WR:.2%}, "
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
        obj_scenarios : pl.DataFrame | None, optional
            Polars DataFrame with scenario market data.  See
            :meth:`calc_annuity_payout` for the expected schema.
        obj_inflation : pl.DataFrame | None, optional
            Polars DataFrame with pre-calculated inflation factors (unused
            for FIA; kept for interface compatibility).  See
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
        """Return a human-readable string representation of the FIA.

        Returns
        -------
        str
            String representation of the Fixed Indexed Annuity.
        """
        return (
            f"FIA: Age {self.m_client_age}, "
            f"Income at {self.m_age_income_start}, "
            f"Index: {self.m_financial_index}, "
            f"Ratchet to {self.m_age_max_ratchet}, "
            f"Rollup {self.m_rate_rollup_benefit:.2%}, "
            f"Cap {self.m_cap_rate:.2%}, "
            f"Participation {self.m_participation_rate:.2%}, "
            f"Spread {self.m_spread_rate:.2%}, "
            f"Floor {self.m_floor_rate:.2%}, "
            f"Payout {self.m_annuity_payout_rate:.2%}, "
            f"Surrender {self.m_rate_surrender_charge_initial:.1%}/"
            f"{self.m_surrender_charge_schedule_years}yr, "
            f"Frequency {self.m_payment_frequency}/year"
        )
