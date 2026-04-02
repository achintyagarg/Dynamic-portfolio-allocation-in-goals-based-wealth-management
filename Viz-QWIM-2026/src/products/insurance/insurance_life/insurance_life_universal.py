"""
Universal Life Insurance class.

===============================

Universal Life (UL) Insurance provides permanent coverage with flexible
premiums and an adjustable death benefit.  Key characteristics:

- **Flexible premiums**: the policyholder may vary the amount and timing
  of premium payments within policy guidelines (subject to minimum
  premium and maximum funding limits to preserve tax advantages).
- **Adjustable death benefit**: the face amount can be increased
  (subject to evidence of insurability) or decreased.
- **Cash value accumulation**: premiums net of cost-of-insurance (COI)
  charges and expenses are credited to a cash-value account at a
  declared crediting rate, subject to a guaranteed minimum.
- **Transparency**: UL policies disclose the cost-of-insurance charges,
  expense charges, and crediting rate separately (unlike whole life).
- **Policy loans**: the owner may borrow against the cash value.
- **Surrender charges**: early surrender may incur a declining charge
  schedule over the first 10–20 policy years.
- **No-lapse guarantee (NLG)**: some UL policies guarantee the death
  benefit remains in force for a specified period even if the cash
  value drops to zero, provided a minimum premium schedule is met.
- **Indexed UL (IUL) variant**: the crediting rate is tied to a market
  index (e.g. S&P 500) subject to a cap and floor — modelled via
  separate parameters in this class.

Author
------
QWIM Team

Version
-------
0.6.0 (2026-02-13)
"""

from __future__ import annotations

from aenum import Enum

from src.utils.custom_exceptions_errors_loggers.exception_custom import (
    Exception_Validation_Input,
)
from src.utils.custom_exceptions_errors_loggers.logger_custom import get_logger

from .insurance_life_base import (
    Death_Benefit_Option,
    Insurance_Life_Base,
    Insurance_Life_Type,
    Premium_Frequency,
    Underwriting_Class,
)


logger = get_logger(__name__)


# ======================================================================
# UL-specific enumerations
# ======================================================================


class UL_Variant(Enum):
    """Universal Life product variant.

    Attributes
    ----------
    TRADITIONAL : str
        Traditional UL — fixed declared crediting rate.
    INDEXED : str
        Indexed UL (IUL) — crediting rate linked to a market index with
        a cap and floor.
    GUARANTEED : str
        Guaranteed UL (GUL / NLG) — emphasises death benefit guarantee
        over cash-value growth.
    """

    TRADITIONAL = "Traditional UL"
    INDEXED = "Indexed UL"
    GUARANTEED = "Guaranteed UL"


class Insurance_Life_Universal(Insurance_Life_Base):
    r"""Universal Life Insurance product class.

    A flexible-premium permanent policy where the cash value grows at
    a crediting rate net of monthly COI and expense deductions.

    **Monthly cash-value accumulation**:

    $$
    CV_m = (CV_{m-1} + P_m) \times \left(1 + \frac{r_c}{12}\right) - COI_m - E_m
    $$

    Where:

    - $CV_m$ = cash value at end of month $m$
    - $P_m$ = premium paid in month $m$
    - $r_c$ = annual crediting rate
    - $COI_m$ = monthly cost of insurance
    - $E_m$ = monthly expense charge

    For **Indexed UL** the effective crediting rate in a given period is:

    $$
    r_{\text{eff}} = \min\!\bigl(\max(r_{\text{index}},\; f),\; c\bigr)
    $$

    Where $r_{\text{index}}$ = index return, $f$ = floor rate, $c$ = cap rate.

    Attributes
    ----------
    m_ul_variant : UL_Variant
        Traditional, indexed, or guaranteed variant.
    m_rate_crediting_current : float
        Current declared crediting rate.
    m_rate_crediting_guaranteed : float
        Guaranteed minimum crediting rate.
    m_rate_cap : float
        Cap rate for indexed UL (0.0 if traditional).
    m_rate_floor : float
        Floor rate for indexed UL (0.0 if traditional).
    m_participation_rate : float
        Participation rate for indexed UL (1.0 if traditional).
    m_cash_value : float
        Current cash value.
    m_rate_COI : float
        Current monthly COI rate per $1 000 of net amount at risk.
    m_rate_expense_charge : float
        Monthly expense / administrative charge rate.
    m_rate_surrender_charge : float
        Current surrender charge rate (declines over time).
    m_surrender_charge_years : int
        Number of years the surrender charge schedule applies.
    m_target_premium : float
        Target annual premium used for commission / NLG calculations.
    m_minimum_premium : float
        Minimum premium to keep the policy in force.
    m_has_no_lapse_guarantee : bool
        Whether the policy has a no-lapse guarantee (NLG) provision.
    m_nlg_guarantee_years : int
        Number of years the NLG is guaranteed.
    m_death_benefit_option : Death_Benefit_Option
        Level (Option A) or increasing (Option B) death benefit.
    m_amount_loan_outstanding : float
        Current outstanding loan balance.
    m_rate_loan_interest : float
        Loan interest rate.
    m_payment_frequency : int
        Number of premium payments per year.
    """

    def __init__(
        self,
        insured_age: int,
        face_amount: float,
        ul_variant: UL_Variant = UL_Variant.TRADITIONAL,
        rate_crediting_current: float = 0.045,
        rate_crediting_guaranteed: float = 0.02,
        rate_cap: float = 0.0,
        rate_floor: float = 0.0,
        participation_rate: float = 1.0,
        cash_value: float = 0.0,
        rate_COI: float = 3.0,
        rate_expense_charge: float = 0.005,
        rate_surrender_charge: float = 0.0,
        surrender_charge_years: int = 10,
        target_premium: float = 0.0,
        minimum_premium: float = 0.0,
        has_no_lapse_guarantee: bool = False,
        nlg_guarantee_years: int = 0,
        death_benefit_option: Death_Benefit_Option = Death_Benefit_Option.LEVEL,
        amount_loan_outstanding: float = 0.0,
        rate_loan_interest: float = 0.05,
        underwriting_class: Underwriting_Class = Underwriting_Class.STANDARD,
        is_smoker: bool = False,
        payment_frequency: int = 12,
        beneficiary_primary: str = "",
        beneficiary_contingent: str = "",
    ) -> None:
        """Initialize a Universal Life Insurance object.

        Parameters
        ----------
        insured_age : int
            Current age of the insured.
        face_amount : float
            Face amount (death benefit).
        ul_variant : UL_Variant, optional
            Product variant (default ``TRADITIONAL``).
        rate_crediting_current : float, optional
            Current declared annual crediting rate (default 0.045).
        rate_crediting_guaranteed : float, optional
            Guaranteed minimum annual crediting rate (default 0.02).
        rate_cap : float, optional
            Cap rate for IUL (default 0.0).
        rate_floor : float, optional
            Floor rate for IUL (default 0.0).
        participation_rate : float, optional
            Participation rate for IUL (default 1.0 = 100 %).
        cash_value : float, optional
            Current cash value (default 0.0).
        rate_COI : float, optional
            Monthly COI rate per $1 000 of NAR (default 3.0).
        rate_expense_charge : float, optional
            Monthly expense charge rate (default 0.005).
        rate_surrender_charge : float, optional
            Current surrender charge rate (default 0.0).
        surrender_charge_years : int, optional
            Surrender charge schedule duration (default 10).
        target_premium : float, optional
            Target annual premium (default 0.0).
        minimum_premium : float, optional
            Minimum annual premium (default 0.0).
        has_no_lapse_guarantee : bool, optional
            NLG provision flag (default ``False``).
        nlg_guarantee_years : int, optional
            NLG guarantee duration in years (default 0).
        death_benefit_option : Death_Benefit_Option, optional
            Option A (level) or Option B (increasing) (default ``LEVEL``).
        amount_loan_outstanding : float, optional
            Outstanding loan balance (default 0.0).
        rate_loan_interest : float, optional
            Loan interest rate (default 0.05).
        underwriting_class : Underwriting_Class, optional
            Risk classification (default ``STANDARD``).
        is_smoker : bool, optional
            Tobacco usage (default ``False``).
        payment_frequency : int, optional
            Payments per year (default 12).
        beneficiary_primary : str, optional
            Primary beneficiary designation.
        beneficiary_contingent : str, optional
            Contingent beneficiary designation.

        Raises
        ------
        Exception_Validation_Input
            If any input fails validation.
        """
        _freq_map: dict[int, Premium_Frequency] = {
            1: Premium_Frequency.ANNUAL,
            2: Premium_Frequency.SEMI_ANNUAL,
            4: Premium_Frequency.QUARTERLY,
            12: Premium_Frequency.MONTHLY,
            0: Premium_Frequency.SINGLE,
        }
        freq_enum = _freq_map.get(payment_frequency, Premium_Frequency.MONTHLY)

        super().__init__(
            insured_age=insured_age,
            face_amount=face_amount,
            insurance_type=Insurance_Life_Type.UNIVERSAL_LIFE,
            underwriting_class=underwriting_class,
            premium_frequency=freq_enum,
            is_smoker=is_smoker,
            beneficiary_primary=beneficiary_primary,
            beneficiary_contingent=beneficiary_contingent,
        )

        # --- Input validation ---
        if not isinstance(ul_variant, UL_Variant):
            raise Exception_Validation_Input(
                "ul_variant must be a valid UL_Variant enum",
                field_name="ul_variant",
                expected_type=UL_Variant,
                actual_value=ul_variant,
            )

        if not isinstance(rate_crediting_current, (int, float)) or rate_crediting_current < 0:
            raise Exception_Validation_Input(
                "rate_crediting_current must be a non-negative number",
                field_name="rate_crediting_current",
                expected_type=float,
                actual_value=rate_crediting_current,
            )

        if not isinstance(rate_crediting_guaranteed, (int, float)) or rate_crediting_guaranteed < 0:
            raise Exception_Validation_Input(
                "rate_crediting_guaranteed must be a non-negative number",
                field_name="rate_crediting_guaranteed",
                expected_type=float,
                actual_value=rate_crediting_guaranteed,
            )

        if not isinstance(rate_cap, (int, float)) or rate_cap < 0:
            raise Exception_Validation_Input(
                "rate_cap must be a non-negative number",
                field_name="rate_cap",
                expected_type=float,
                actual_value=rate_cap,
            )

        if not isinstance(rate_floor, (int, float)):
            raise Exception_Validation_Input(
                "rate_floor must be a number",
                field_name="rate_floor",
                expected_type=float,
                actual_value=rate_floor,
            )

        if (
            not isinstance(participation_rate, (int, float))
            or participation_rate < 0
            or participation_rate > 3.0
        ):
            raise Exception_Validation_Input(
                "participation_rate must be in [0, 3.0]",
                field_name="participation_rate",
                expected_type=float,
                actual_value=participation_rate,
            )

        if not isinstance(cash_value, (int, float)) or cash_value < 0:
            raise Exception_Validation_Input(
                "cash_value must be a non-negative number",
                field_name="cash_value",
                expected_type=float,
                actual_value=cash_value,
            )

        if not isinstance(rate_COI, (int, float)) or rate_COI < 0:
            raise Exception_Validation_Input(
                "rate_COI must be a non-negative number",
                field_name="rate_COI",
                expected_type=float,
                actual_value=rate_COI,
            )

        if not isinstance(rate_expense_charge, (int, float)) or rate_expense_charge < 0:
            raise Exception_Validation_Input(
                "rate_expense_charge must be a non-negative number",
                field_name="rate_expense_charge",
                expected_type=float,
                actual_value=rate_expense_charge,
            )

        if not isinstance(rate_surrender_charge, (int, float)) or rate_surrender_charge < 0:
            raise Exception_Validation_Input(
                "rate_surrender_charge must be a non-negative number",
                field_name="rate_surrender_charge",
                expected_type=float,
                actual_value=rate_surrender_charge,
            )

        if not isinstance(surrender_charge_years, int) or surrender_charge_years < 0:
            raise Exception_Validation_Input(
                "surrender_charge_years must be a non-negative integer",
                field_name="surrender_charge_years",
                expected_type=int,
                actual_value=surrender_charge_years,
            )

        if not isinstance(death_benefit_option, Death_Benefit_Option):
            raise Exception_Validation_Input(
                "death_benefit_option must be a valid Death_Benefit_Option enum",
                field_name="death_benefit_option",
                expected_type=Death_Benefit_Option,
                actual_value=death_benefit_option,
            )

        if not isinstance(amount_loan_outstanding, (int, float)) or amount_loan_outstanding < 0:
            raise Exception_Validation_Input(
                "amount_loan_outstanding must be a non-negative number",
                field_name="amount_loan_outstanding",
                expected_type=float,
                actual_value=amount_loan_outstanding,
            )

        if not isinstance(rate_loan_interest, (int, float)) or rate_loan_interest < 0:
            raise Exception_Validation_Input(
                "rate_loan_interest must be a non-negative number",
                field_name="rate_loan_interest",
                expected_type=float,
                actual_value=rate_loan_interest,
            )

        if not isinstance(payment_frequency, int) or payment_frequency < 0:
            raise Exception_Validation_Input(
                "payment_frequency must be a non-negative integer",
                field_name="payment_frequency",
                expected_type=int,
                actual_value=payment_frequency,
            )

        # --- Set member variables ---
        self.m_ul_variant: UL_Variant = ul_variant
        self.m_rate_crediting_current: float = float(rate_crediting_current)
        self.m_rate_crediting_guaranteed: float = float(rate_crediting_guaranteed)
        self.m_rate_cap: float = float(rate_cap)
        self.m_rate_floor: float = float(rate_floor)
        self.m_participation_rate: float = float(participation_rate)
        self.m_cash_value: float = float(cash_value)
        self.m_rate_COI: float = float(rate_COI)
        self.m_rate_expense_charge: float = float(rate_expense_charge)
        self.m_rate_surrender_charge: float = float(rate_surrender_charge)
        self.m_surrender_charge_years: int = surrender_charge_years
        self.m_target_premium: float = float(target_premium)
        self.m_minimum_premium: float = float(minimum_premium)
        self.m_has_no_lapse_guarantee: bool = bool(has_no_lapse_guarantee)
        self.m_nlg_guarantee_years: int = nlg_guarantee_years
        self.m_death_benefit_option: Death_Benefit_Option = death_benefit_option
        self.m_amount_loan_outstanding: float = float(amount_loan_outstanding)
        self.m_rate_loan_interest: float = float(rate_loan_interest)
        self.m_payment_frequency: int = payment_frequency

        logger.info(
            f"Created Universal Life ({ul_variant.value}): "
            f"face ${face_amount:,.2f}, "
            f"crediting rate {rate_crediting_current:.2%} "
            f"(guaranteed {rate_crediting_guaranteed:.2%}), "
            f"NLG: {has_no_lapse_guarantee}",
        )

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def ul_variant(self) -> UL_Variant:
        """UL_Variant : Product variant (traditional, indexed, guaranteed)."""
        return self.m_ul_variant

    @property
    def rate_crediting_current(self) -> float:
        """float : Current declared annual crediting rate."""
        return self.m_rate_crediting_current

    @property
    def rate_crediting_guaranteed(self) -> float:
        """float : Guaranteed minimum annual crediting rate."""
        return self.m_rate_crediting_guaranteed

    @property
    def rate_cap(self) -> float:
        """float : Cap rate for indexed UL."""
        return self.m_rate_cap

    @property
    def rate_floor(self) -> float:
        """float : Floor rate for indexed UL."""
        return self.m_rate_floor

    @property
    def participation_rate(self) -> float:
        """float : Participation rate for indexed UL."""
        return self.m_participation_rate

    @property
    def cash_value(self) -> float:
        """float : Current cash value."""
        return self.m_cash_value

    @property
    def rate_COI(self) -> float:
        """float : Monthly COI rate per $1 000 of NAR."""
        return self.m_rate_COI

    @property
    def rate_expense_charge(self) -> float:
        """float : Monthly expense charge rate."""
        return self.m_rate_expense_charge

    @property
    def rate_surrender_charge(self) -> float:
        """float : Current surrender charge rate."""
        return self.m_rate_surrender_charge

    @property
    def surrender_charge_years(self) -> int:
        """int : Surrender charge schedule duration in years."""
        return self.m_surrender_charge_years

    @property
    def target_premium(self) -> float:
        """float : Target annual premium."""
        return self.m_target_premium

    @property
    def minimum_premium(self) -> float:
        """float : Minimum annual premium to keep policy in force."""
        return self.m_minimum_premium

    @property
    def has_no_lapse_guarantee(self) -> bool:
        """bool : Whether the policy has a no-lapse guarantee."""
        return self.m_has_no_lapse_guarantee

    @property
    def nlg_guarantee_years(self) -> int:
        """int : Number of years the NLG is guaranteed."""
        return self.m_nlg_guarantee_years

    @property
    def death_benefit_option(self) -> Death_Benefit_Option:
        """Death_Benefit_Option : Level (A) or increasing (B) death benefit."""
        return self.m_death_benefit_option

    @property
    def amount_loan_outstanding(self) -> float:
        """float : Outstanding loan balance."""
        return self.m_amount_loan_outstanding

    @property
    def rate_loan_interest(self) -> float:
        """float : Loan interest rate."""
        return self.m_rate_loan_interest

    @property
    def payment_frequency(self) -> int:
        """int : Number of premium payments per year."""
        return self.m_payment_frequency

    # ------------------------------------------------------------------
    # Crediting rate calculations
    # ------------------------------------------------------------------

    def calc_effective_crediting_rate(
        self,
        index_return: float = 0.0,
    ) -> float:
        r"""Calculate the effective crediting rate for the current period.

        For **traditional UL**: the current declared rate.
        For **indexed UL**:

        $$
        r_{\text{eff}} = \min\!\bigl(\max(r_{\text{index}} \times p,\; f),\; c\bigr)
        $$

        For **guaranteed UL**: the guaranteed minimum rate.

        Parameters
        ----------
        index_return : float, optional
            The market index return for the period (default 0.0).

        Returns
        -------
        float
            The effective annual crediting rate.
        """
        if self.m_ul_variant == UL_Variant.INDEXED:
            raw_credit = index_return * self.m_participation_rate
            return min(max(raw_credit, self.m_rate_floor), self.m_rate_cap)

        if self.m_ul_variant == UL_Variant.GUARANTEED:
            return self.m_rate_crediting_guaranteed

        return self.m_rate_crediting_current

    # ------------------------------------------------------------------
    # Death benefit calculations
    # ------------------------------------------------------------------

    def calc_death_benefit(self) -> float:
        r"""Calculate the current death benefit payable.

        **Option A (Level)**: death benefit = face amount.
        **Option B (Increasing)**: death benefit = face amount + cash value.
        Both are reduced by the outstanding loan balance.

        $$
        DB = \begin{cases}
            F - L & \text{Option A (Level)} \\
            F + CV - L & \text{Option B (Increasing)}
        \end{cases}
        $$

        Returns
        -------
        float
            The net death benefit (floored at 0).
        """
        if self.m_death_benefit_option == Death_Benefit_Option.INCREASING:
            gross_db = self.m_face_amount + self.m_cash_value
        else:
            gross_db = self.m_face_amount

        return max(gross_db - self.m_amount_loan_outstanding, 0.0)

    # ------------------------------------------------------------------
    # Premium calculations
    # ------------------------------------------------------------------

    def calc_annual_premium(self) -> float:
        r"""Calculate the estimated target annual premium.

        If a target premium was explicitly set, return it.  Otherwise
        estimate using a simplified COI-based approach:

        $$
        P_{\text{annual}} = \frac{F}{1\,000} \times COI \times 12 \times (1 + e)
        $$

        Returns
        -------
        float
            The estimated annual premium.
        """
        if self.m_target_premium > 0:
            return self.m_target_premium

        expense_loading: float = 0.15
        annual_coi = (self.m_face_amount / 1_000) * self.m_rate_COI * 12
        return annual_coi * (1 + expense_loading)

    def calc_modal_premium(self) -> float:
        """Calculate the per-payment premium based on payment frequency.

        Returns
        -------
        float
            The premium per payment period.
        """
        annual = self.calc_annual_premium()
        factor = self.calc_modal_premium_factor()
        if self.m_premium_frequency.value == 0:
            return annual
        return annual * factor

    # ------------------------------------------------------------------
    # Cash value projections
    # ------------------------------------------------------------------

    def calc_monthly_COI_charge(self) -> float:
        r"""Calculate the monthly cost-of-insurance deduction.

        $$
        COI_m = \frac{NAR}{1\,000} \times q_m
        $$

        Returns
        -------
        float
            Monthly COI charge.
        """
        nar = self.calc_net_amount_at_risk()
        return (nar / 1_000) * self.m_rate_COI

    def calc_net_amount_at_risk(self) -> float:
        r"""Calculate the net amount at risk.

        $$
        NAR = DB - CV
        $$

        Returns
        -------
        float
            Net amount at risk (floored at 0).
        """
        return max(self.calc_death_benefit() - self.m_cash_value, 0.0)

    def calc_cash_value_at_month(
        self,
        months: int,
        monthly_premium: float | None = None,
        index_return: float = 0.0,
    ) -> float:
        r"""Project the cash value at the end of a number of months.

        $$
        CV_m = (CV_{m-1} + P_m) \times \left(1 + \frac{r_c}{12}\right) - COI_m - E_m
        $$

        Parameters
        ----------
        months : int
            Number of months to project.
        monthly_premium : float | None, optional
            Monthly premium amount.  If ``None``, uses target / 12.
        index_return : float, optional
            Annualised index return for IUL crediting (default 0.0).

        Returns
        -------
        float
            Projected cash value.

        Raises
        ------
        Exception_Validation_Input
            If ``months`` is not a positive integer.
        """
        if not isinstance(months, int) or months < 1:
            raise Exception_Validation_Input(
                "months must be a positive integer",
                field_name="months",
                expected_type=int,
                actual_value=months,
            )

        if monthly_premium is None:
            monthly_premium = self.calc_annual_premium() / 12

        effective_rate = self.calc_effective_crediting_rate(index_return)
        monthly_rate = effective_rate / 12

        cv: float = self.m_cash_value
        for _ in range(months):
            nar = max(self.m_face_amount - cv, 0.0)
            coi_charge = (nar / 1_000) * self.m_rate_COI
            expense_charge = cv * self.m_rate_expense_charge / 12
            cv = (cv + monthly_premium) * (1 + monthly_rate) - coi_charge - expense_charge

        return max(cv, 0.0)

    def calc_cash_surrender_value(self, policy_year: int = 0) -> float:
        """Calculate the cash surrender value.

        Cash surrender value = cash value - surrender charge - loan balance.

        Parameters
        ----------
        policy_year : int, optional
            Current policy year for surrender charge lookup (default 0).

        Returns
        -------
        float
            Cash surrender value (floored at 0).
        """
        if policy_year > 0 and policy_year <= self.m_surrender_charge_years:
            # Linear decline of surrender charge
            charge_factor = self.m_rate_surrender_charge * (
                1 - (policy_year - 1) / self.m_surrender_charge_years
            )
        else:
            charge_factor = 0.0

        surrender_charge = self.m_cash_value * charge_factor
        return max(
            self.m_cash_value - surrender_charge - self.m_amount_loan_outstanding,
            0.0,
        )

    # ------------------------------------------------------------------
    # Loan calculations
    # ------------------------------------------------------------------

    def calc_max_loan_amount(self) -> float:
        """Calculate the maximum policy loan available.

        Typically 90 % of cash value minus outstanding loans.

        Returns
        -------
        float
            Maximum additional loan amount.
        """
        max_loanable = self.m_cash_value * 0.90
        return max(max_loanable - self.m_amount_loan_outstanding, 0.0)

    def calc_loan_interest_due(self, years: int = 1) -> float:
        r"""Calculate interest due on the outstanding loan.

        $$
        I = L \times r_l \times t
        $$

        Parameters
        ----------
        years : int, optional
            Number of years (default 1).

        Returns
        -------
        float
            Interest due.

        Raises
        ------
        Exception_Validation_Input
            If ``years`` is not a positive integer.
        """
        if not isinstance(years, int) or years < 1:
            raise Exception_Validation_Input(
                "years must be a positive integer",
                field_name="years",
                expected_type=int,
                actual_value=years,
            )

        return self.m_amount_loan_outstanding * self.m_rate_loan_interest * years

    # ------------------------------------------------------------------
    # Policy lapse analysis
    # ------------------------------------------------------------------

    def calc_months_until_lapse(
        self,
        monthly_premium: float = 0.0,
        index_return: float = 0.0,
    ) -> int:
        r"""Estimate months until the policy lapses (cash value hits zero).

        Projects month-by-month until the cash value is exhausted, up to
        a maximum of 1 200 months (100 years).

        Parameters
        ----------
        monthly_premium : float, optional
            Monthly premium being paid (default 0.0 = no premiums).
        index_return : float, optional
            Annualised index return for IUL (default 0.0).

        Returns
        -------
        int
            Estimated months until lapse, or -1 if policy survives
            the 1 200-month horizon.
        """
        effective_rate = self.calc_effective_crediting_rate(index_return)
        monthly_rate = effective_rate / 12

        cv: float = self.m_cash_value
        for month in range(1, 1_201):
            nar = max(self.m_face_amount - cv, 0.0)
            coi_charge = (nar / 1_000) * self.m_rate_COI
            expense_charge = cv * self.m_rate_expense_charge / 12
            cv = (cv + monthly_premium) * (1 + monthly_rate) - coi_charge - expense_charge
            if cv <= 0:
                return month

        return -1

    # ------------------------------------------------------------------
    # String representation
    # ------------------------------------------------------------------

    def get_insurance_as_string(self) -> str:
        """Return a human-readable string representation.

        Returns
        -------
        str
            String representation of the Universal Life policy.
        """
        nlg_text = ", NLG" if self.m_has_no_lapse_guarantee else ""
        opt = "B" if self.m_death_benefit_option == Death_Benefit_Option.INCREASING else "A"
        db_text = f"Option {opt}"
        return (
            f"Universal Life ({self.m_ul_variant.value}, {db_text}{nlg_text}): "
            f"Age {self.m_insured_age}, "
            f"Face ${self.m_face_amount:,.2f}, "
            f"CV ${self.m_cash_value:,.2f}, "
            f"Crediting Rate {self.m_rate_crediting_current:.2%}, "
            f"Class: {self.m_underwriting_class.value}"
        )
