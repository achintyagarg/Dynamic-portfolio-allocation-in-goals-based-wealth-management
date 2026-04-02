"""
Deferred Income Annuity (DIA) class.

===============================

A Deferred Income Annuity delays the start of income payments until a
specified future age, typically at or near retirement.  Key characteristics:

- **Deferral period**: no payments are made; the insurer credits mortality
  gains and interest, which produce higher payouts than a SPIA purchased
  at income-start age.
- **Mortality credits**: annuitants who do not survive the deferral period
  effectively subsidise those who do, increasing the payout rate.
- **Return-of-premium (ROP) death benefit**: an optional rider that refunds
  the premium (less any payments made) if the annuitant dies during deferral.
- **Payout options**: life-only, period-certain, or life-with-period-certain.
- **Inflation adjustment**: payments may be linked to an external inflation
  index or include a fixed COLA.

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

from .annuity_base import Annuity_Base, Annuity_Payout_Option, Annuity_Type, Withdrawal_Rates


logger = get_logger(__name__)


class Annuity_DIA(Annuity_Base):
    r"""Deferred Income Annuity (DIA).

    Delays income payments until a specified future age.  During the
    deferral period the premium earns interest and mortality credits,
    producing higher payouts than a SPIA purchased at income-start age.

    The future value of the principal at income start is:

    $$
    FV = A \times (1 + g)^{n}
    $$

    The present value of the deferred income stream discounted to today is:

    $$
    PV = \sum_{t=k}^{k+m-1} \frac{P}{(1 + d)^{t}}
    $$

    Where:

    - $A$ = premium (principal) amount
    - $g$ = annual growth rate during deferral
    - $n$ = number of deferral years
    - $k$ = deferral period in years
    - $m$ = expected payment years after income start
    - $P$ = annual payout
    - $d$ = discount rate

    Attributes
    ----------
    m_client_age : int
        The current age of the client (inherited).
    m_annuity_payout_rate : float
        The payout rate of the annuity (inherited).
    m_annuity_type : Annuity_Type
        Set to ``ANNUITY_DIA`` (inherited).
    m_age_income_start : int
        The age at which annuity income payments begin.
    m_payout_option : Annuity_Payout_Option
        Payout structure (life-only, period-certain, etc.).
    m_guarantee_period_years : int
        Number of years income is guaranteed once payments begin.
    m_is_inflation_adjusted : bool
        Whether the annuity payments are adjusted for inflation.
    m_rate_COLA : float
        Annual cost-of-living adjustment rate (0.0 for fixed payments).
    m_has_death_benefit_ROP : bool
        Whether a return-of-premium death benefit applies during deferral.
    m_rate_mortality_credit : float
        Implicit mortality credit rate earned during the deferral period.
    m_payment_frequency : int
        Number of payments per year.
    """

    def __init__(
        self,
        client_age: int,
        annuity_payout_rate: float,
        age_income_start: int,
        payout_option: Annuity_Payout_Option = Annuity_Payout_Option.LIFE_ONLY,
        guarantee_period_years: int = 0,
        is_inflation_adjusted: bool = False,
        rate_COLA: float = 0.0,
        has_death_benefit_ROP: bool = False,
        rate_mortality_credit: float = 0.0,
        payment_frequency: int = 12,
        annuity_income_starting_age: int | None = None,
    ) -> None:
        """Initialize a DIA object.

        Parameters
        ----------
        client_age : int
            The current age of the client.
        annuity_payout_rate : float
            The annual payout rate of the annuity (as a decimal).
        age_income_start : int
            The age at which the annuity income payments begin.
        payout_option : Annuity_Payout_Option, optional
            The payout structure (default ``LIFE_ONLY``).
        guarantee_period_years : int, optional
            Number of guaranteed payment years once income begins
            (default 0).  Must be > 0 for period-certain options.
        is_inflation_adjusted : bool, optional
            Whether payments are linked to an inflation index
            (default ``False``).
        rate_COLA : float, optional
            Annual cost-of-living adjustment rate (default 0.0 = fixed).
        has_death_benefit_ROP : bool, optional
            Whether a return-of-premium death benefit applies during
            deferral (default ``False``).
        rate_mortality_credit : float, optional
            Implicit annual mortality credit rate during deferral
            (default 0.0).
        payment_frequency : int, optional
            Number of payments per year (default 12 — monthly).
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
            Annuity_Type.ANNUITY_DIA,
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

        if not isinstance(payout_option, Annuity_Payout_Option):
            raise Exception_Validation_Input(
                "payout_option must be a valid Annuity_Payout_Option enum",
                field_name="payout_option",
                expected_type=Annuity_Payout_Option,
                actual_value=payout_option,
            )

        if not isinstance(guarantee_period_years, int) or guarantee_period_years < 0:
            raise Exception_Validation_Input(
                "guarantee_period_years must be a non-negative integer",
                field_name="guarantee_period_years",
                expected_type=int,
                actual_value=guarantee_period_years,
            )

        if (
            payout_option
            in (
                Annuity_Payout_Option.PERIOD_CERTAIN,
                Annuity_Payout_Option.LIFE_WITH_PERIOD_CERTAIN,
            )
            and guarantee_period_years <= 0
        ):
            raise Exception_Validation_Input(
                "guarantee_period_years must be > 0 for period-certain options",
                field_name="guarantee_period_years",
                expected_type=int,
                actual_value=guarantee_period_years,
            )

        if not isinstance(rate_COLA, (int, float)) or rate_COLA < 0:
            raise Exception_Validation_Input(
                "rate_COLA must be a non-negative number",
                field_name="rate_COLA",
                expected_type=float,
                actual_value=rate_COLA,
            )

        if not isinstance(rate_mortality_credit, (int, float)) or rate_mortality_credit < 0:
            raise Exception_Validation_Input(
                "rate_mortality_credit must be a non-negative number",
                field_name="rate_mortality_credit",
                expected_type=float,
                actual_value=rate_mortality_credit,
            )

        if not isinstance(payment_frequency, int) or payment_frequency <= 0:
            raise Exception_Validation_Input(
                "payment_frequency must be a positive integer",
                field_name="payment_frequency",
                expected_type=int,
                actual_value=payment_frequency,
            )

        # --- Set member variables ---
        self.m_age_income_start: int = age_income_start
        self.m_payout_option: Annuity_Payout_Option = payout_option
        self.m_guarantee_period_years: int = guarantee_period_years
        self.m_is_inflation_adjusted: bool = is_inflation_adjusted
        self.m_rate_COLA: float = float(rate_COLA)
        self.m_has_death_benefit_ROP: bool = has_death_benefit_ROP
        self.m_rate_mortality_credit: float = float(rate_mortality_credit)
        self.m_payment_frequency: int = payment_frequency

        logger.info(
            f"Created DIA ({payout_option.value}) with payout rate "
            f"{annuity_payout_rate:.2%}, income at age {age_income_start}, "
            f"COLA: {rate_COLA:.2%}, ROP: {has_death_benefit_ROP}, "
            f"mortality credit: {rate_mortality_credit:.2%}, "
            f"payment frequency: {payment_frequency}/year",
        )

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def income_start_age(self) -> int:
        """Int : The age at which income payments begin."""
        return self.m_age_income_start

    @property
    def deferral_years(self) -> int:
        """Int : Number of years until income payments begin."""
        return self.m_age_income_start - self.m_client_age

    @property
    def payout_option(self) -> Annuity_Payout_Option:
        """payout_option : The payout structure of the annuity."""
        return self.m_payout_option

    @property
    def guarantee_period_years(self) -> int:
        """Int : Number of years income is guaranteed once payments begin."""
        return self.m_guarantee_period_years

    @property
    def is_inflation_adjusted(self) -> bool:
        """Bool : Whether the annuity is inflation-adjusted."""
        return self.m_is_inflation_adjusted

    @property
    def rate_COLA(self) -> float:
        """Float : Annual cost-of-living adjustment rate."""
        return self.m_rate_COLA

    @property
    def has_death_benefit_ROP(self) -> bool:
        """Bool : Whether a return-of-premium death benefit applies."""
        return self.m_has_death_benefit_ROP

    @property
    def rate_mortality_credit(self) -> float:
        """Float : Implicit annual mortality credit rate during deferral."""
        return self.m_rate_mortality_credit

    @property
    def payment_frequency(self) -> int:
        """Int : Number of payments per year."""
        return self.m_payment_frequency

    # ------------------------------------------------------------------
    # Payout calculations
    # ------------------------------------------------------------------

    def calc_annuity_payout(
        self,
        amount_principal: float,
        obj_scenarios: pl.DataFrame | None = None,
        obj_inflation: pl.DataFrame | None = None,
    ) -> float:
        """Calculate the DIA first-year annual payout.

        No payments are made during the deferral period.  Once the client
        reaches ``age_income_start``, the annual payout is the premium
        times the payout rate, optionally adjusted for inflation.

        Parameters
        ----------
        amount_principal : float
            The premium invested in the annuity.
        obj_scenarios : pl.DataFrame | None, optional
            Polars DataFrame with scenario market data (unused for DIA;
            accepted for interface compatibility).  Expected columns: ``Date``
            followed by financial-component return columns.
        obj_inflation : pl.DataFrame | None, optional
            Polars DataFrame with pre-calculated inflation factors.  Must
            contain exactly four columns:

            - ``Start Date`` : start of the inflation period.
            - ``End Date``   : end of the inflation period.
            - ``Inflation Factor`` : cumulative inflation already calculated
              for the interval ``[Start Date, End Date]``
            - ``Inverse Inflation Factor`` : inverse of cumulative inflation already calculated
              for the interval ``[Start Date, End Date]``.

            When multiple rows are present the individual factors are
            multiplied to produce a single cumulative adjustment.

        Returns
        -------
        float
            The calculated first-year annual payout starting at
            ``age_income_start``.  Returns ``0.0`` while still in deferral.

        Raises
        ------
        Exception_Validation_Input
            If ``amount_principal`` is not a positive number.
            If ``obj_inflation`` is provided but missing required columns.
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

        annual_payout: float = amount_principal * self.m_annuity_payout_rate

        # Apply external inflation-index adjustment if configured
        if self.m_is_inflation_adjusted and obj_inflation is not None:
            required_cols = {"Start Date", "End Date", "Inflation Factor"}
            if not required_cols.issubset(set(obj_inflation.columns)):
                missing = required_cols - set(obj_inflation.columns)
                raise Exception_Validation_Input(
                    f"obj_inflation is missing required columns: {missing}",
                    field_name="obj_inflation",
                    expected_type=pl.DataFrame,
                    actual_value=obj_inflation.columns,
                )
            total_factor: float = obj_inflation["Inflation Factor"].product()
            annual_payout *= total_factor
            logger.info(
                f"DIA inflation adjustment applied: deferral={self.deferral_years}yr, "
                f"factor={total_factor:.6f}, adjusted payout={annual_payout:.2f}",
            )

        return annual_payout

    def calc_withdrawal_rates(
        self,
        amount_principal: float,
        desired_WR: float | None = None,
        obj_scenarios: pl.DataFrame | None = None,
        obj_inflation: pl.DataFrame | None = None,
    ) -> Withdrawal_Rates:
        r"""Calculate nominal and real withdrawal rates for this DIA.

        The nominal withdrawal rate is the ratio of the annuity payout
        (computed without inflation adjustment) to the invested principal.
        Returns ``0.0`` for both rates while the client is still in the
        deferral period.  The real withdrawal rate deflates the nominal
        rate by the cumulative inverse inflation factor:

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
            The premium invested in the annuity.
        desired_WR : float
            The desired withdrawal rate (as a decimal, e.g. 0.04 for 4 %).
            Validated and logged for reference; the returned rates reflect
            what the annuity actually delivers.
        obj_scenarios : pl.DataFrame | None, optional
            Polars DataFrame with scenario market data (unused for DIA;
            accepted for interface compatibility).
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

        # Nominal payout: pass obj_inflation=None so the rate is not
        # double-adjusted when inflation is already applied inside
        # calc_annuity_payout for inflation-linked DIAs.
        # Returns 0.0 automatically while still in the deferral period.
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
            f"DIA withdrawal rates — desired={effective_desired_WR:.2%}, "
            f"nominal={nominal_WR:.4%}, real={real_WR:.4%}, "
            f"inverse_inflation_factor={inverse_inflation_factor:.6f}",
        )

        return Withdrawal_Rates(nominal_WR=nominal_WR, real_WR=real_WR)

    def calc_payout_in_year(
        self,
        amount_principal: float,
        year_number: int,
    ) -> float:
        r"""Calculate the payout in a specific payment year accounting for COLA.

        Year numbering starts at 1 = the first year income is received
        (i.e. the year the client reaches ``age_income_start``).

        $$
        P_N = (A \times r) \times (1 + c)^{N - 1}
        $$

        Parameters
        ----------
        amount_principal : float
            The premium invested in the annuity.
        year_number : int
            The payment year (1-indexed from income start).

        Returns
        -------
        float
            The annual payout in the specified year, including COLA.

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

        if not isinstance(year_number, int) or year_number < 1:
            raise Exception_Validation_Input(
                "year_number must be a positive integer (1-indexed)",
                field_name="year_number",
                expected_type=int,
                actual_value=year_number,
            )

        # Base payout at income start (uses payout rate directly)
        base_payout: float = amount_principal * self.m_annuity_payout_rate
        return base_payout * ((1 + self.m_rate_COLA) ** (year_number - 1))

    def calc_future_value(
        self,
        amount_principal: float,
        growth_rate: float,
    ) -> float:
        r"""Calculate the future value of the principal at income start age.

        Uses the compound interest formula:

        $$
        FV = A \times (1 + g)^{n}
        $$

        Parameters
        ----------
        amount_principal : float
            The premium invested in the annuity.
        growth_rate : float
            The annual growth rate during the deferral period (as a decimal).

        Returns
        -------
        float
            The future value of the principal at income start age.

        Raises
        ------
        Exception_Validation_Input
            If ``amount_principal`` is not positive or ``growth_rate`` is
            not a number.
        """
        if not isinstance(amount_principal, (int, float)) or amount_principal <= 0:
            raise Exception_Validation_Input(
                "amount_principal must be a positive number",
                field_name="amount_principal",
                expected_type=float,
                actual_value=amount_principal,
            )

        if not isinstance(growth_rate, (int, float)):
            raise Exception_Validation_Input(
                "growth_rate must be a number",
                field_name="growth_rate",
                expected_type=float,
                actual_value=growth_rate,
            )

        return amount_principal * ((1 + growth_rate) ** self.deferral_years)

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
            The premium invested in the annuity.
        obj_scenarios : pl.DataFrame | None, optional
            Polars DataFrame with scenario market data (unused for DIA;
            accepted for interface compatibility).
        obj_inflation : pl.DataFrame | None, optional
            Polars DataFrame with pre-calculated inflation factors.  See
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
    # Analytical calculations
    # ------------------------------------------------------------------

    def calc_present_value_deferred_income(
        self,
        amount_principal: float,
        discount_rate: float,
        life_expectancy_years: int,
    ) -> float:
        r"""Calculate the present value of the deferred income stream.

        Discounts the future payment stream back to today, accounting
        for the deferral period and the payment horizon:

        $$
        PV = \sum_{t=k+1}^{k+m}
             \frac{P_1 \times (1 + c)^{t - k - 1}}{(1 + d)^{t}}
        $$

        Where $k$ = deferral years, $m$ = payment horizon,
        $c$ = COLA rate, $d$ = discount rate.

        Parameters
        ----------
        amount_principal : float
            The premium invested in the annuity.
        discount_rate : float
            The annual discount rate.
        life_expectancy_years : int
            The client's total remaining life expectancy from current age.

        Returns
        -------
        float
            The present value of the expected deferred income stream.

        Raises
        ------
        Exception_Validation_Input
            If any input is invalid.
        """
        if not isinstance(amount_principal, (int, float)) or amount_principal <= 0:
            raise Exception_Validation_Input(
                "amount_principal must be a positive number",
                field_name="amount_principal",
                expected_type=float,
                actual_value=amount_principal,
            )

        if not isinstance(discount_rate, (int, float)) or discount_rate < 0:
            raise Exception_Validation_Input(
                "discount_rate must be a non-negative number",
                field_name="discount_rate",
                expected_type=float,
                actual_value=discount_rate,
            )

        if not isinstance(life_expectancy_years, int) or life_expectancy_years <= 0:
            raise Exception_Validation_Input(
                "life_expectancy_years must be a positive integer",
                field_name="life_expectancy_years",
                expected_type=int,
                actual_value=life_expectancy_years,
            )

        k = self.deferral_years

        # Payment years after income start
        remaining_after_start = life_expectancy_years - k
        if remaining_after_start <= 0:
            return 0.0

        if self.m_payout_option == Annuity_Payout_Option.PERIOD_CERTAIN:
            payment_years = self.m_guarantee_period_years
        elif self.m_payout_option == Annuity_Payout_Option.LIFE_WITH_PERIOD_CERTAIN:
            payment_years = max(self.m_guarantee_period_years, remaining_after_start)
        else:
            payment_years = remaining_after_start

        base_payout: float = amount_principal * self.m_annuity_payout_rate
        present_value: float = 0.0

        for payment_year in range(1, payment_years + 1):
            t = k + payment_year  # years from today
            payout_t = base_payout * ((1 + self.m_rate_COLA) ** (payment_year - 1))
            present_value += payout_t / ((1 + discount_rate) ** t)

        return present_value

    def calc_commutation_value(
        self,
        amount_principal: float,
        discount_rate: float,
        remaining_payment_years: int,
    ) -> float:
        r"""Calculate the commutation (lump-sum buyout) value.

        The commutation value is the present value of remaining payments
        at a given point in time, typically used when converting a stream
        of annuity payments into a single lump sum:

        $$
        CV = \sum_{t=1}^{n}
             \frac{P_1 \times (1 + c)^{t-1}}{(1 + d)^{t}}
        $$

        Parameters
        ----------
        amount_principal : float
            The premium invested in the annuity.
        discount_rate : float
            The annual discount rate for the commutation calculation.
        remaining_payment_years : int
            Expected remaining years of payments.

        Returns
        -------
        float
            The lump-sum equivalent of remaining payments.

        Raises
        ------
        Exception_Validation_Input
            If any input is invalid.
        """
        if not isinstance(amount_principal, (int, float)) or amount_principal <= 0:
            raise Exception_Validation_Input(
                "amount_principal must be a positive number",
                field_name="amount_principal",
                expected_type=float,
                actual_value=amount_principal,
            )

        if not isinstance(discount_rate, (int, float)) or discount_rate < 0:
            raise Exception_Validation_Input(
                "discount_rate must be a non-negative number",
                field_name="discount_rate",
                expected_type=float,
                actual_value=discount_rate,
            )

        if not isinstance(remaining_payment_years, int) or remaining_payment_years <= 0:
            raise Exception_Validation_Input(
                "remaining_payment_years must be a positive integer",
                field_name="remaining_payment_years",
                expected_type=int,
                actual_value=remaining_payment_years,
            )

        base_payout: float = amount_principal * self.m_annuity_payout_rate
        commutation: float = 0.0

        for t in range(1, remaining_payment_years + 1):
            payout_t = base_payout * ((1 + self.m_rate_COLA) ** (t - 1))
            commutation += payout_t / ((1 + discount_rate) ** t)

        return commutation

    def calc_breakeven_age(
        self,
        amount_principal: float,
    ) -> int:
        r"""Calculate the age at which cumulative payments equal the premium.

        Year counting begins at the income start age.  The breakeven
        year $N^{*}$ satisfies:

        $$
        \sum_{t=1}^{N^{*}} P_1 \times (1 + c)^{t-1} \ge A
        $$

        Parameters
        ----------
        amount_principal : float
            The premium invested in the annuity.

        Returns
        -------
        int
            The client age at which cumulative payments first equal or
            exceed the premium.

        Raises
        ------
        Exception_Validation_Input
            If ``amount_principal`` is not positive.
        Exception_Calculation
            If breakeven is not reached within 100 years of payments.
        """
        if not isinstance(amount_principal, (int, float)) or amount_principal <= 0:
            raise Exception_Validation_Input(
                "amount_principal must be a positive number",
                field_name="amount_principal",
                expected_type=float,
                actual_value=amount_principal,
            )

        base_payout: float = amount_principal * self.m_annuity_payout_rate
        cumulative: float = 0.0

        for year in range(1, 101):
            payout_year = base_payout * ((1 + self.m_rate_COLA) ** (year - 1))
            cumulative += payout_year
            if cumulative >= amount_principal:
                return self.m_age_income_start + year

        raise Exception_Calculation(
            "Breakeven age not reached within 100 years of payments",
        )

    def calc_mortality_credit_value(
        self,
        amount_principal: float,
    ) -> float:
        r"""Estimate the value of mortality credits earned during deferral.

        Mortality credits represent the economic benefit of risk pooling:
        annuitants who die during the deferral period forfeit their
        premiums, which subsidise the payouts for surviving annuitants.

        $$
        MC = A \times \bigl((1 + r_m)^{n} - 1\bigr)
        $$

        Parameters
        ----------
        amount_principal : float
            The premium invested in the annuity.

        Returns
        -------
        float
            Estimated value of accumulated mortality credits.

        Raises
        ------
        Exception_Validation_Input
            If ``amount_principal`` is not positive.
        """
        if not isinstance(amount_principal, (int, float)) or amount_principal <= 0:
            raise Exception_Validation_Input(
                "amount_principal must be a positive number",
                field_name="amount_principal",
                expected_type=float,
                actual_value=amount_principal,
            )

        n = self.deferral_years
        return amount_principal * ((1 + self.m_rate_mortality_credit) ** n - 1)

    def calc_death_benefit_during_deferral(
        self,
        amount_principal: float,
    ) -> float:
        """Calculate the return-of-premium death benefit during deferral.

        If the ROP rider is active the benefit equals the original premium.
        Otherwise the benefit is zero (the premium is forfeited upon death).

        Parameters
        ----------
        amount_principal : float
            The premium invested in the annuity.

        Returns
        -------
        float
            The death benefit amount during the deferral period.

        Raises
        ------
        Exception_Validation_Input
            If ``amount_principal`` is not positive.
        """
        if not isinstance(amount_principal, (int, float)) or amount_principal <= 0:
            raise Exception_Validation_Input(
                "amount_principal must be a positive number",
                field_name="amount_principal",
                expected_type=float,
                actual_value=amount_principal,
            )

        if self.m_has_death_benefit_ROP:
            return float(amount_principal)
        return 0.0

    # ------------------------------------------------------------------
    # String representation
    # ------------------------------------------------------------------

    def get_annuity_as_string(self) -> str:
        """Return a human-readable string representation of the DIA.

        Returns
        -------
        str
            String representation of the DIA.
        """
        cola_text = f"COLA {self.m_rate_COLA:.1%}" if self.m_rate_COLA > 0 else "Fixed"
        inflation_text = ", Inflation-Indexed" if self.m_is_inflation_adjusted else ""
        rop_text = ", ROP" if self.m_has_death_benefit_ROP else ""
        guarantee_text = (
            f", Guarantee: {self.m_guarantee_period_years}yr"
            if self.m_guarantee_period_years > 0
            else ""
        )

        return (
            f"DIA ({self.m_payout_option.value}, {cola_text}{inflation_text}"
            f"{rop_text}{guarantee_text}): "
            f"Current Age: {self.m_client_age}, "
            f"Income Start Age: {self.m_age_income_start}, "
            f"Deferral: {self.deferral_years}yr, "
            f"Payout Rate: {self.m_annuity_payout_rate:.2%}, "
            f"Frequency: {self.m_payment_frequency}/year"
        )
