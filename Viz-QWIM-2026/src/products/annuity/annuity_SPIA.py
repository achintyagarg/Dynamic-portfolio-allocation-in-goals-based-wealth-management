"""
Single Premium Immediate Annuity (SPIA) class.

===============================

A Single Premium Immediate Annuity begins paying income immediately
after purchase.  Key characteristics:

- **Immediate income**: payments begin within one payment period of purchase.
- **Payout options**: life-only, period-certain, life-with-period-certain,
  or joint-life with a configurable survivor benefit.
- **COLA**: an optional cost-of-living adjustment compounds each year.
- **Irrevocable**: the premium cannot be recovered once the contract is issued.
- **Exclusion ratio**: under IRS rules a portion of each payment is considered
  a tax-free return of premium.

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


class Annuity_SPIA(Annuity_Base):
    r"""Single Premium Immediate Annuity (SPIA).

    Pays income immediately after purchase for a guaranteed period, for
    the annuitant's lifetime, or a combination of both.  Joint-life
    options provide continued benefits to a surviving spouse or partner.

    The first-year annual payout is:

    $$
    P_{\text{annual}} = A \times r
    $$

    With a Cost-of-Living Adjustment (COLA) the payout in year $N$ is:

    $$
    P_N = P_1 \times (1 + c)^{N - 1}
    $$

    Where:

    - $A$ = premium (principal) amount invested
    - $r$ = annual payout rate
    - $c$ = annual COLA rate
    - $N$ = payment year (1-indexed)

    Attributes
    ----------
    m_client_age : int
        The age of the client (inherited).
    m_annuity_payout_rate : float
        The annual payout rate (inherited).
    m_annuity_type : Annuity_Type
        Set to ``ANNUITY_SPIA`` (inherited).
    m_payout_option : Annuity_Payout_Option
        Payout structure (life-only, period-certain, etc.).
    m_guarantee_period_years : int
        Number of years income is guaranteed regardless of survival.
    m_rate_COLA : float
        Annual cost-of-living adjustment rate (0.0 for fixed payments).
    m_joint_survivor_pct : float
        Survivor benefit as a fraction of the primary benefit (joint-life).
    m_is_inflation_adjusted : bool
        Whether payments are linked to an external inflation index.
    m_payment_frequency : int
        Number of payments per year (e.g. 12 for monthly).
    """

    def __init__(
        self,
        client_age: int,
        annuity_payout_rate: float,
        payout_option: Annuity_Payout_Option = Annuity_Payout_Option.LIFE_ONLY,
        guarantee_period_years: int = 0,
        rate_COLA: float = 0.0,
        joint_survivor_pct: float = 1.0,
        is_inflation_adjusted: bool = False,
        payment_frequency: int = 12,
        annuity_income_starting_age: int | None = None,
    ) -> None:
        """Initialize a SPIA object.

        Parameters
        ----------
        client_age : int
            The age of the client.
        annuity_payout_rate : float
            The annual payout rate of the annuity (as a decimal).
        payout_option : Annuity_Payout_Option, optional
            The payout structure (default ``LIFE_ONLY``).
        guarantee_period_years : int, optional
            Number of guaranteed payment years (default 0).  Must be > 0
            for ``PERIOD_CERTAIN`` and ``LIFE_WITH_PERIOD_CERTAIN``.
        rate_COLA : float, optional
            Annual cost-of-living adjustment rate (default 0.0 = fixed).
        joint_survivor_pct : float, optional
            Survivor benefit fraction for joint-life option (default 1.0).
            Typical industry values are 0.50, 0.75, or 1.00.
        is_inflation_adjusted : bool, optional
            Whether payments are linked to an external inflation index
            (default ``False``).
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
            Annuity_Type.ANNUITY_SPIA,
            annuity_income_starting_age=annuity_income_starting_age,
        )

        # --- Input validation ---
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

        if (
            not isinstance(joint_survivor_pct, (int, float))
            or joint_survivor_pct <= 0
            or joint_survivor_pct > 1
        ):
            raise Exception_Validation_Input(
                "joint_survivor_pct must be in (0, 1]",
                field_name="joint_survivor_pct",
                expected_type=float,
                actual_value=joint_survivor_pct,
            )

        if not isinstance(payment_frequency, int) or payment_frequency <= 0:
            raise Exception_Validation_Input(
                "payment_frequency must be a positive integer",
                field_name="payment_frequency",
                expected_type=int,
                actual_value=payment_frequency,
            )

        # --- Set member variables ---
        self.m_payout_option: Annuity_Payout_Option = payout_option
        self.m_guarantee_period_years: int = guarantee_period_years
        self.m_rate_COLA: float = float(rate_COLA)
        self.m_joint_survivor_pct: float = float(joint_survivor_pct)
        self.m_is_inflation_adjusted: bool = is_inflation_adjusted
        self.m_payment_frequency: int = payment_frequency

        logger.info(
            f"Created SPIA ({payout_option.value}) with payout rate "
            f"{annuity_payout_rate:.2%}, COLA: {rate_COLA:.2%}, "
            f"guarantee: {guarantee_period_years} yr, "
            f"payment frequency: {payment_frequency}/year",
        )

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def payout_option(self) -> Annuity_Payout_Option:
        """payout_option : The payout structure of the annuity."""
        return self.m_payout_option

    @property
    def guarantee_period_years(self) -> int:
        """Int : Number of years income is guaranteed."""
        return self.m_guarantee_period_years

    @property
    def rate_COLA(self) -> float:
        """Float : Annual cost-of-living adjustment rate."""
        return self.m_rate_COLA

    @property
    def joint_survivor_pct(self) -> float:
        """Float : Survivor benefit fraction for joint-life option."""
        return self.m_joint_survivor_pct

    @property
    def is_inflation_adjusted(self) -> bool:
        """Bool : Whether the annuity is linked to an inflation index."""
        return self.m_is_inflation_adjusted

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
        """Calculate the SPIA first-year annual payout.

        For a SPIA the base calculation is the premium multiplied by
        the payout rate, with an optional inflation-index adjustment.

        Parameters
        ----------
        amount_principal : float
            The premium invested in the annuity.
        obj_scenarios : pl.DataFrame | None, optional
            Polars DataFrame with scenario market data (unused for SPIA;
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
            The calculated first-year annual payout amount.

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

        annual_payout: float = amount_principal * self.m_annuity_payout_rate

        # Apply external inflation-index adjustment if configured
        if self.m_is_inflation_adjusted and obj_inflation is not None:
            required_cols = {
                "Start Date",
                "End Date",
                "Inflation Factor",
            }
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
                f"SPIA inflation adjustment applied: factor={total_factor:.6f}, "
                f"adjusted payout={annual_payout:.2f}",
            )

        return annual_payout

    def calc_withdrawal_rates(
        self,
        amount_principal: float,
        desired_WR: float | None = None,
        obj_scenarios: pl.DataFrame | None = None,
        obj_inflation: pl.DataFrame | None = None,
    ) -> Withdrawal_Rates:
        r"""Calculate nominal and real withdrawal rates for this SPIA.

        The nominal withdrawal rate is the ratio of the annuity payout
        (computed without inflation adjustment) to the invested principal.
        The real withdrawal rate deflates the nominal rate by the cumulative
        inverse inflation factor to express purchasing-power-adjusted income:

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
            Polars DataFrame with scenario market data (unused for SPIA;
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

            - ``nominal_WR``: payout-to-principal ratio (unaffected by inflation).
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
        # calc_annuity_payout for inflation-linked SPIAs.
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
            f"SPIA withdrawal rates — desired={effective_desired_WR:.2%}, "
            f"nominal={nominal_WR:.4%}, real={real_WR:.4%}, "
            f"inverse_inflation_factor={inverse_inflation_factor:.6f}",
        )

        return Withdrawal_Rates(nominal_WR=nominal_WR, real_WR=real_WR)

    def calc_payout_in_year(
        self,
        amount_principal: float,
        year_number: int,
    ) -> float:
        r"""Calculate the payout in a specific year accounting for COLA.

        $$
        P_N = (A \times r) \times (1 + c)^{N - 1}
        $$

        Parameters
        ----------
        amount_principal : float
            The premium invested in the annuity.
        year_number : int
            The payment year (1-indexed; year 1 = first year of payments).

        Returns
        -------
        float
            The annual payout in the specified year, including COLA growth.

        Raises
        ------
        Exception_Validation_Input
            If ``amount_principal`` is not positive or ``year_number`` < 1.
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

        first_year_payout = self.calc_annuity_payout(amount_principal)
        return first_year_payout * ((1 + self.m_rate_COLA) ** (year_number - 1))

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
            Polars DataFrame with scenario market data (unused for SPIA;
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

    def calc_present_value_payments(
        self,
        amount_principal: float,
        discount_rate: float,
        life_expectancy_years: int,
    ) -> float:
        r"""Calculate the present value of the expected future payment stream.

        The payment horizon depends on the payout option:

        - **Life-only / Joint-life**: remaining life expectancy.
        - **Period-certain**: the guaranteed period.
        - **Life-with-period-certain**: the longer of guaranty or life expectancy.

        With COLA the present value is:

        $$
        PV = \sum_{t=1}^{n} \frac{P_1 \times (1 + c)^{t-1}}{(1 + d)^{t}}
        $$

        Parameters
        ----------
        amount_principal : float
            The premium invested in the annuity.
        discount_rate : float
            The annual discount rate.
        life_expectancy_years : int
            The client's remaining life expectancy in years.

        Returns
        -------
        float
            The present value of the expected payment stream.

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

        # Determine the payment horizon based on payout option
        if self.m_payout_option == Annuity_Payout_Option.PERIOD_CERTAIN:
            num_years = self.m_guarantee_period_years
        elif self.m_payout_option == Annuity_Payout_Option.LIFE_WITH_PERIOD_CERTAIN:
            num_years = max(self.m_guarantee_period_years, life_expectancy_years)
        else:
            # LIFE_ONLY and JOINT_LIFE use life expectancy
            num_years = life_expectancy_years

        first_year_payout = self.calc_annuity_payout(amount_principal)

        present_value: float = 0.0
        for t in range(1, num_years + 1):
            payout_t = first_year_payout * ((1 + self.m_rate_COLA) ** (t - 1))
            present_value += payout_t / ((1 + discount_rate) ** t)

        return present_value

    def calc_breakeven_age(
        self,
        amount_principal: float,
    ) -> int:
        r"""Calculate the breakeven age when cumulative payments equal the premium.

        The breakeven year $N^{*}$ satisfies:

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
            If breakeven is not reached within 100 years.
        """
        if not isinstance(amount_principal, (int, float)) or amount_principal <= 0:
            raise Exception_Validation_Input(
                "amount_principal must be a positive number",
                field_name="amount_principal",
                expected_type=float,
                actual_value=amount_principal,
            )

        first_year_payout = self.calc_annuity_payout(amount_principal)
        cumulative: float = 0.0

        for year in range(1, 101):
            payout_year = first_year_payout * ((1 + self.m_rate_COLA) ** (year - 1))
            cumulative += payout_year
            if cumulative >= amount_principal:
                return self.m_client_age + year

        raise Exception_Calculation(
            "Breakeven age not reached within 100 years of payments",
        )

    def calc_exclusion_ratio(
        self,
        amount_principal: float,
        expected_payment_years: int,
    ) -> float:
        r"""Calculate the IRS exclusion ratio for tax purposes.

        The exclusion ratio determines the tax-free portion of each
        payment, representing the return of the original premium:

        $$
        E = \frac{A}{\displaystyle\sum_{t=1}^{n} P_1 \times (1 + c)^{t-1}}
        $$

        The result is capped at 1.0.

        Parameters
        ----------
        amount_principal : float
            The premium invested (cost basis).
        expected_payment_years : int
            Expected number of payment years (from IRS life-expectancy
            tables).

        Returns
        -------
        float
            The exclusion ratio as a decimal in [0, 1].

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

        if not isinstance(expected_payment_years, int) or expected_payment_years <= 0:
            raise Exception_Validation_Input(
                "expected_payment_years must be a positive integer",
                field_name="expected_payment_years",
                expected_type=int,
                actual_value=expected_payment_years,
            )

        first_year_payout = self.calc_annuity_payout(amount_principal)
        total_expected: float = 0.0
        for t in range(1, expected_payment_years + 1):
            total_expected += first_year_payout * ((1 + self.m_rate_COLA) ** (t - 1))

        if total_expected <= 0:
            return 0.0

        return min(amount_principal / total_expected, 1.0)

    def calc_cumulative_payments(
        self,
        amount_principal: float,
        num_years: int,
    ) -> float:
        r"""Calculate total cumulative payments over a specified number of years.

        $$
        C = \sum_{t=1}^{n} P_1 \times (1 + c)^{t-1}
        $$

        Parameters
        ----------
        amount_principal : float
            The premium invested in the annuity.
        num_years : int
            Number of payment years to accumulate.

        Returns
        -------
        float
            The total cumulative payments.

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

        if not isinstance(num_years, int) or num_years <= 0:
            raise Exception_Validation_Input(
                "num_years must be a positive integer",
                field_name="num_years",
                expected_type=int,
                actual_value=num_years,
            )

        first_year_payout = self.calc_annuity_payout(amount_principal)
        cumulative: float = 0.0
        for t in range(1, num_years + 1):
            cumulative += first_year_payout * ((1 + self.m_rate_COLA) ** (t - 1))

        return cumulative

    # ------------------------------------------------------------------
    # String representation
    # ------------------------------------------------------------------

    def get_annuity_as_string(self) -> str:
        """Return a human-readable string representation of the SPIA.

        Returns
        -------
        str
            String representation of the SPIA.
        """
        cola_text = f"COLA {self.m_rate_COLA:.1%}" if self.m_rate_COLA > 0 else "Fixed"
        inflation_text = ", Inflation-Indexed" if self.m_is_inflation_adjusted else ""
        guarantee_text = (
            f", Guarantee: {self.m_guarantee_period_years}yr"
            if self.m_guarantee_period_years > 0
            else ""
        )

        return (
            f"SPIA ({self.m_payout_option.value}, {cola_text}{inflation_text}"
            f"{guarantee_text}): "
            f"Client Age: {self.m_client_age}, "
            f"Payout Rate: {self.m_annuity_payout_rate:.2%}, "
            f"Frequency: {self.m_payment_frequency}/year"
        )
