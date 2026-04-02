"""
Whole Life Insurance class.

===============================

Whole Life Insurance provides permanent coverage for the insured's
entire lifetime with level (guaranteed) premiums and a guaranteed
cash-value accumulation.  Key characteristics:

- **Lifetime coverage**: the policy never expires as long as premiums
  are paid.
- **Level premiums**: the annual premium is fixed at issue and never
  increases.
- **Guaranteed cash value**: a savings component grows at a guaranteed
  interest rate and is accessible through loans or withdrawals.
- **Dividends**: participating (par) policies from mutual companies may
  pay annual dividends, which can be taken as cash, used to reduce
  premiums, left to accumulate, or used to purchase paid-up additions.
- **Paid-up additions (PUA)**: additional increments of fully-paid
  whole-life coverage purchased with dividends or extra premiums,
  increasing both the death benefit and cash value.
- **Loan provision**: policyholders may borrow against the cash value
  at a guaranteed or variable loan interest rate.
- **Non-forfeiture options**: if the policy lapses the owner may elect
  reduced paid-up insurance, extended term insurance, or a cash
  surrender.

Author
------
QWIM Team

Version
-------
0.6.0 (2026-02-13)
"""

from __future__ import annotations

from src.utils.custom_exceptions_errors_loggers.exception_custom import (
    Exception_Calculation,
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


class Insurance_Life_Whole(Insurance_Life_Base):
    r"""Whole Life Insurance product class.

    A participating whole-life policy accumulates guaranteed cash value
    at a declared crediting rate.  The annual premium is level (fixed)
    over the premium-paying period.

    **Cash value at end of year** $t$:

    $$
    CV_t = CV_{t-1} \times (1 + r_g) + P_{\text{annual}} - COI_t + D_t
    $$

    Where:

    - $CV_t$ = cash value at end of year $t$
    - $r_g$ = guaranteed crediting rate
    - $P_{\text{annual}}$ = annual premium
    - $COI_t$ = cost of insurance in year $t$
    - $D_t$ = dividend in year $t$ (if participating)

    **Simplified annual premium estimate** (level for life):

    $$
    P = \frac{F}{1\,000} \times COI \times (1 + e)
    $$

    Where $F$ = face amount, $COI$ = cost-of-insurance rate per $1\,000$,
    $e$ = expense loading factor.

    Attributes
    ----------
    m_rate_guaranteed_interest : float
        Guaranteed annual crediting rate on the cash value.
    m_cash_value : float
        Current accumulated cash value.
    m_is_participating : bool
        Whether the policy pays dividends.
    m_rate_dividend : float
        Current annual dividend rate (as a fraction of the cash value).
    m_rate_loan_interest : float
        Interest rate charged on policy loans.
    m_amount_loan_outstanding : float
        Current outstanding loan balance.
    m_paid_up_additions : float
        Total paid-up additions face amount purchased.
    m_premium_paying_years : int
        Number of years over which premiums are paid (e.g. 20, 65-age,
        or 100 for whole-life-to-100).
    m_rate_cost_of_insurance : float
        Approximate annual cost-of-insurance rate per $1 000.
    m_death_benefit_option : Death_Benefit_Option
        Level or increasing death benefit.
    m_payment_frequency : int
        Number of premium payments per year.
    """

    def __init__(
        self,
        insured_age: int,
        face_amount: float,
        rate_guaranteed_interest: float = 0.04,
        cash_value: float = 0.0,
        is_participating: bool = True,
        rate_dividend: float = 0.0,
        rate_loan_interest: float = 0.05,
        amount_loan_outstanding: float = 0.0,
        paid_up_additions: float = 0.0,
        premium_paying_years: int = 100,
        rate_cost_of_insurance: float = 5.0,
        death_benefit_option: Death_Benefit_Option = Death_Benefit_Option.LEVEL,
        underwriting_class: Underwriting_Class = Underwriting_Class.STANDARD,
        is_smoker: bool = False,
        payment_frequency: int = 12,
        beneficiary_primary: str = "",
        beneficiary_contingent: str = "",
    ) -> None:
        """Initialize a Whole Life Insurance object.

        Parameters
        ----------
        insured_age : int
            Current age of the insured.
        face_amount : float
            Face amount (death benefit).
        rate_guaranteed_interest : float, optional
            Guaranteed annual crediting rate on the cash value (default 0.04).
        cash_value : float, optional
            Current accumulated cash value (default 0.0).
        is_participating : bool, optional
            Whether the policy pays dividends (default ``True``).
        rate_dividend : float, optional
            Annual dividend rate as a fraction (default 0.0).
        rate_loan_interest : float, optional
            Loan interest rate (default 0.05).
        amount_loan_outstanding : float, optional
            Current outstanding loan balance (default 0.0).
        paid_up_additions : float, optional
            Total PUA face amount (default 0.0).
        premium_paying_years : int, optional
            Premium-paying period in years (default 100 = whole-life).
        rate_cost_of_insurance : float, optional
            Annual COI rate per $1 000 of coverage (default 5.0).
        death_benefit_option : Death_Benefit_Option, optional
            Level or increasing death benefit (default ``LEVEL``).
        underwriting_class : Underwriting_Class, optional
            Risk classification (default ``STANDARD``).
        is_smoker : bool, optional
            Tobacco usage flag (default ``False``).
        payment_frequency : int, optional
            Number of premium payments per year (default 12).
        beneficiary_primary : str, optional
            Primary beneficiary designation.
        beneficiary_contingent : str, optional
            Contingent beneficiary designation.

        Raises
        ------
        Exception_Validation_Input
            If any input fails validation.
        """
        # Determine modal Premium_Frequency enum from int
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
            insurance_type=Insurance_Life_Type.WHOLE_LIFE,
            underwriting_class=underwriting_class,
            premium_frequency=freq_enum,
            is_smoker=is_smoker,
            beneficiary_primary=beneficiary_primary,
            beneficiary_contingent=beneficiary_contingent,
        )

        # --- Input validation ---
        if not isinstance(rate_guaranteed_interest, (int, float)) or rate_guaranteed_interest < 0:
            raise Exception_Validation_Input(
                "rate_guaranteed_interest must be a non-negative number",
                field_name="rate_guaranteed_interest",
                expected_type=float,
                actual_value=rate_guaranteed_interest,
            )

        if not isinstance(cash_value, (int, float)) or cash_value < 0:
            raise Exception_Validation_Input(
                "cash_value must be a non-negative number",
                field_name="cash_value",
                expected_type=float,
                actual_value=cash_value,
            )

        if not isinstance(rate_dividend, (int, float)) or rate_dividend < 0:
            raise Exception_Validation_Input(
                "rate_dividend must be a non-negative number",
                field_name="rate_dividend",
                expected_type=float,
                actual_value=rate_dividend,
            )

        if not isinstance(rate_loan_interest, (int, float)) or rate_loan_interest < 0:
            raise Exception_Validation_Input(
                "rate_loan_interest must be a non-negative number",
                field_name="rate_loan_interest",
                expected_type=float,
                actual_value=rate_loan_interest,
            )

        if not isinstance(amount_loan_outstanding, (int, float)) or amount_loan_outstanding < 0:
            raise Exception_Validation_Input(
                "amount_loan_outstanding must be a non-negative number",
                field_name="amount_loan_outstanding",
                expected_type=float,
                actual_value=amount_loan_outstanding,
            )

        if not isinstance(paid_up_additions, (int, float)) or paid_up_additions < 0:
            raise Exception_Validation_Input(
                "paid_up_additions must be a non-negative number",
                field_name="paid_up_additions",
                expected_type=float,
                actual_value=paid_up_additions,
            )

        if not isinstance(premium_paying_years, int) or premium_paying_years <= 0:
            raise Exception_Validation_Input(
                "premium_paying_years must be a positive integer",
                field_name="premium_paying_years",
                expected_type=int,
                actual_value=premium_paying_years,
            )

        if not isinstance(rate_cost_of_insurance, (int, float)) or rate_cost_of_insurance < 0:
            raise Exception_Validation_Input(
                "rate_cost_of_insurance must be a non-negative number",
                field_name="rate_cost_of_insurance",
                expected_type=float,
                actual_value=rate_cost_of_insurance,
            )

        if not isinstance(death_benefit_option, Death_Benefit_Option):
            raise Exception_Validation_Input(
                "death_benefit_option must be a valid Death_Benefit_Option enum",
                field_name="death_benefit_option",
                expected_type=Death_Benefit_Option,
                actual_value=death_benefit_option,
            )

        if not isinstance(payment_frequency, int) or payment_frequency < 0:
            raise Exception_Validation_Input(
                "payment_frequency must be a non-negative integer",
                field_name="payment_frequency",
                expected_type=int,
                actual_value=payment_frequency,
            )

        # --- Set member variables ---
        self.m_rate_guaranteed_interest: float = float(rate_guaranteed_interest)
        self.m_cash_value: float = float(cash_value)
        self.m_is_participating: bool = bool(is_participating)
        self.m_rate_dividend: float = float(rate_dividend)
        self.m_rate_loan_interest: float = float(rate_loan_interest)
        self.m_amount_loan_outstanding: float = float(amount_loan_outstanding)
        self.m_paid_up_additions: float = float(paid_up_additions)
        self.m_premium_paying_years: int = premium_paying_years
        self.m_rate_cost_of_insurance: float = float(rate_cost_of_insurance)
        self.m_death_benefit_option: Death_Benefit_Option = death_benefit_option
        self.m_payment_frequency: int = payment_frequency

        logger.info(
            f"Created Whole Life: face ${face_amount:,.2f}, "
            f"guaranteed rate {rate_guaranteed_interest:.2%}, "
            f"participating: {is_participating}, "
            f"premium-paying years: {premium_paying_years}",
        )

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def rate_guaranteed_interest(self) -> float:
        """float : Guaranteed annual crediting rate on the cash value."""
        return self.m_rate_guaranteed_interest

    @property
    def cash_value(self) -> float:
        """float : Current accumulated cash value."""
        return self.m_cash_value

    @property
    def is_participating(self) -> bool:
        """bool : Whether the policy pays dividends."""
        return self.m_is_participating

    @property
    def rate_dividend(self) -> float:
        """float : Annual dividend rate."""
        return self.m_rate_dividend

    @property
    def rate_loan_interest(self) -> float:
        """float : Interest rate charged on policy loans."""
        return self.m_rate_loan_interest

    @property
    def amount_loan_outstanding(self) -> float:
        """float : Outstanding loan balance."""
        return self.m_amount_loan_outstanding

    @property
    def paid_up_additions(self) -> float:
        """float : Total paid-up additions face amount."""
        return self.m_paid_up_additions

    @property
    def premium_paying_years(self) -> int:
        """int : Number of premium-paying years."""
        return self.m_premium_paying_years

    @property
    def rate_cost_of_insurance(self) -> float:
        """float : Annual COI rate per $1 000 of net amount at risk."""
        return self.m_rate_cost_of_insurance

    @property
    def death_benefit_option(self) -> Death_Benefit_Option:
        """Death_Benefit_Option : Level or increasing death benefit."""
        return self.m_death_benefit_option

    @property
    def payment_frequency(self) -> int:
        """int : Number of premium payments per year."""
        return self.m_payment_frequency

    # ------------------------------------------------------------------
    # Death benefit calculations
    # ------------------------------------------------------------------

    def calc_death_benefit(self) -> float:
        r"""Calculate the current death benefit payable.

        For a **level** death benefit the payout is the face amount plus
        any paid-up additions, minus the outstanding loan balance:

        $$
        DB_{\text{level}} = F + PUA - L
        $$

        For an **increasing** death benefit:

        $$
        DB_{\text{inc}} = F + PUA + CV - L
        $$

        Where:

        - $F$ = face amount
        - $PUA$ = paid-up additions face amount
        - $CV$ = cash value
        - $L$ = outstanding loan balance

        Returns
        -------
        float
            The net death benefit amount (floored at 0).
        """
        if self.m_death_benefit_option == Death_Benefit_Option.INCREASING:
            gross_db = self.m_face_amount + self.m_paid_up_additions + self.m_cash_value
        else:
            gross_db = self.m_face_amount + self.m_paid_up_additions

        net_db = gross_db - self.m_amount_loan_outstanding
        return max(net_db, 0.0)

    # ------------------------------------------------------------------
    # Premium calculations
    # ------------------------------------------------------------------

    def calc_annual_premium(self) -> float:
        r"""Calculate the estimated annual premium.

        Uses a simplified net-premium approach:

        $$
        P_{\text{annual}} = \frac{F}{1\,000} \times COI \times (1 + e)
        $$

        Where $COI$ = cost-of-insurance rate per $1\,000$, $e$ = expense
        loading factor (set to 0.20 = 20 % of net premium).

        Returns
        -------
        float
            The estimated annual premium.
        """
        expense_loading: float = 0.20
        net_premium = (self.m_face_amount / 1_000) * self.m_rate_cost_of_insurance
        return net_premium * (1 + expense_loading)

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
    # Cash value calculations
    # ------------------------------------------------------------------

    def calc_cash_value_at_year(self, year: int) -> float:
        r"""Project the cash value at the end of a given policy year.

        Simplified projection assuming level premiums, constant COI,
        and constant dividend rate:

        $$
        CV_t = CV_{t-1} \times (1 + r_g) + P - COI + D
        $$

        Where $D = CV_{t-1} \times r_d$ if participating.

        Parameters
        ----------
        year : int
            The policy year number (1-indexed).

        Returns
        -------
        float
            Projected cash value at end of the specified year.

        Raises
        ------
        Exception_Validation_Input
            If ``year`` is not a positive integer.
        """
        if not isinstance(year, int) or year < 1:
            raise Exception_Validation_Input(
                "year must be a positive integer",
                field_name="year",
                expected_type=int,
                actual_value=year,
            )

        annual_premium = self.calc_annual_premium()
        coi_annual = (self.m_face_amount / 1_000) * self.m_rate_cost_of_insurance
        cv: float = self.m_cash_value

        for _ in range(year):
            dividend = cv * self.m_rate_dividend if self.m_is_participating else 0.0
            cv = cv * (1 + self.m_rate_guaranteed_interest) + annual_premium - coi_annual + dividend

        return max(cv, 0.0)

    def calc_cash_surrender_value(self) -> float:
        """Calculate the current cash surrender value.

        Cash surrender value equals the cash value minus the outstanding
        loan balance.

        Returns
        -------
        float
            The cash surrender value (floored at 0).
        """
        return max(self.m_cash_value - self.m_amount_loan_outstanding, 0.0)

    def calc_net_amount_at_risk(self) -> float:
        r"""Calculate the net amount at risk (NAR) for the insurer.

        $$
        NAR = DB - CV
        $$

        Returns
        -------
        float
            Net amount at risk (floored at 0).
        """
        return max(self.calc_death_benefit() - self.m_cash_value, 0.0)

    # ------------------------------------------------------------------
    # Loan calculations
    # ------------------------------------------------------------------

    def calc_max_loan_amount(self) -> float:
        """Calculate the maximum policy loan available.

        Typically up to 90 % of the cash value minus any outstanding loans.

        Returns
        -------
        float
            Maximum additional loan amount available.
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
            Interest amount due.

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
    # Dividend calculations
    # ------------------------------------------------------------------

    def calc_annual_dividend(self) -> float:
        """Calculate the annual dividend payment.

        Returns
        -------
        float
            The annual dividend amount (0 if non-participating).
        """
        if not self.m_is_participating:
            return 0.0
        return self.m_cash_value * self.m_rate_dividend

    def calc_paid_up_additions_from_dividend(self) -> float:
        r"""Estimate the PUA face amount purchasable with the annual dividend.

        Uses a simplified single-premium rate:

        $$
        PUA = \frac{D}{r_{\text{PUA}}}
        $$

        Where $r_{\text{PUA}}$ = cost per dollar of PUA (approximated
        by the COI rate / 1 000).

        Returns
        -------
        float
            Estimated additional paid-up face amount.
        """
        dividend = self.calc_annual_dividend()
        if dividend <= 0 or self.m_rate_cost_of_insurance <= 0:
            return 0.0

        cost_per_dollar = self.m_rate_cost_of_insurance / 1_000
        return dividend / cost_per_dollar

    # ------------------------------------------------------------------
    # IRR / performance
    # ------------------------------------------------------------------

    def calc_internal_rate_of_return(
        self,
        num_years: int,
    ) -> float:
        r"""Estimate the policy's internal rate of return over a horizon.

        Uses a simplified IRR proxy based on the cash value growth:

        $$
        IRR \approx \left(\frac{CV_n}{P_{\text{total}}}\right)^{1/n} - 1
        $$

        Parameters
        ----------
        num_years : int
            Number of years to project.

        Returns
        -------
        float
            Approximate IRR as a decimal.

        Raises
        ------
        Exception_Validation_Input
            If ``num_years`` is not a positive integer.
        Exception_Calculation
            If total premiums are zero.
        """
        if not isinstance(num_years, int) or num_years < 1:
            raise Exception_Validation_Input(
                "num_years must be a positive integer",
                field_name="num_years",
                expected_type=int,
                actual_value=num_years,
            )

        total_premiums = self.calc_annual_premium() * num_years
        if total_premiums <= 0:
            raise Exception_Calculation(
                "Total premiums are zero; cannot compute IRR",
            )

        projected_cv = self.calc_cash_value_at_year(num_years)
        if projected_cv <= 0:
            return -1.0

        return (projected_cv / total_premiums) ** (1.0 / num_years) - 1

    # ------------------------------------------------------------------
    # String representation
    # ------------------------------------------------------------------

    def get_insurance_as_string(self) -> str:
        """Return a human-readable string representation.

        Returns
        -------
        str
            String representation of the Whole Life policy.
        """
        par_text = "Participating" if self.m_is_participating else "Non-Participating"
        return (
            f"Whole Life ({par_text}): "
            f"Age {self.m_insured_age}, "
            f"Face ${self.m_face_amount:,.2f}, "
            f"CV ${self.m_cash_value:,.2f}, "
            f"Guaranteed Rate {self.m_rate_guaranteed_interest:.2%}, "
            f"Class: {self.m_underwriting_class.value}"
        )
