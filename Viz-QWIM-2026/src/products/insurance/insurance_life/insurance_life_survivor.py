"""
Survivor (Second-to-Die) Life Insurance class.

================================================

Survivor Life Insurance (also called Second-to-Die or Joint Survivor)
covers two insured lives — typically spouses — and pays the death
benefit only upon the death of the *second* insured.

Key characteristics:

- **Payout trigger**: death benefit payable at the second death, not
  the first.
- **Lower premiums**: because the insurer expects to pay much later
  than a single-life policy, premiums are significantly lower.
- **Estate-planning tool**: commonly used to fund estate taxes, provide
  a legacy, or fund an irrevocable life insurance trust (ILIT).
- **Cash value**: typically built on a whole-life or universal-life
  chassis and accumulates cash value.
- **Joint mortality**: the policy's cost of insurance is priced using
  **joint last-to-die mortality** which combines the mortality tables
  of both insureds.
- **Split-option rider**: some policies include a rider that allows the
  policy to split into two individual policies upon a qualifying event
  such as divorce.
- **Guaranteed insurability**: the surviving insured may have the right
  to purchase an individual policy after the first death.

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
# Survivor-specific enumerations
# ======================================================================


class Survivor_Chassis(Enum):
    """Underlying policy structure for the survivor product.

    Attributes
    ----------
    WHOLE_LIFE : str
        Built on a whole-life chassis with guaranteed cash value.
    UNIVERSAL_LIFE : str
        Built on a universal-life chassis with flexible premiums.
    VARIABLE_UNIVERSAL_LIFE : str
        Built on a variable-universal-life chassis with sub-accounts.
    """

    WHOLE_LIFE = "Whole Life"
    UNIVERSAL_LIFE = "Universal Life"
    VARIABLE_UNIVERSAL_LIFE = "Variable Universal Life"


class Insurance_Life_Survivor(Insurance_Life_Base):
    r"""Survivor (Second-to-Die) Life Insurance product class.

    The death benefit is payable upon the death of the second of two
    insured individuals.  Joint last-to-die mortality is used to price
    the policy.

    **Joint last-to-die probability**:

    The probability that *at least one* of two independent lives
    (ages $x$ and $y$) is alive at time $t$ is:

    $$
    {}_tp_{xy}^{\overline{LS}} = 1 - (1 - {}_tp_x)(1 - {}_tp_y)
    $$

    Where ${}_tp_x$ is the probability that life $x$ survives $t$ years.

    **Joint COI** (simplified):

    $$
    COI_{\text{joint}} = COI_1 \times COI_2 \times \frac{1}{COI_1 + COI_2}
    $$

    This reflects the fact that the benefit is paid only at the second
    death, making the joint COI lower than either individual COI.

    Attributes
    ----------
    m_insured_age_second : int
        Age of the second insured.
    m_underwriting_class_second : Underwriting_Class
        Underwriting class of the second insured.
    m_is_smoker_second : bool
        Tobacco usage of the second insured.
    m_first_death_occurred : bool
        Whether the first death has already occurred.
    m_chassis : Survivor_Chassis
        Underlying policy structure.
    m_cash_value : float
        Current cash value.
    m_rate_guaranteed_interest : float
        Guaranteed interest rate on cash value.
    m_rate_COI_insured_1 : float
        COI rate per $1 000 for insured 1.
    m_rate_COI_insured_2 : float
        COI rate per $1 000 for insured 2.
    m_death_benefit_option : Death_Benefit_Option
        Death benefit option (level or increasing).
    m_has_split_option : bool
        Whether the policy includes a split-option rider.
    m_amount_loan_outstanding : float
        Outstanding loan balance.
    m_rate_loan_interest : float
        Loan interest rate.
    m_rate_surrender_charge : float
        Current surrender charge rate.
    m_surrender_charge_years : int
        Surrender charge schedule duration.
    """

    def __init__(
        self,
        insured_age: int,
        insured_age_second: int,
        face_amount: float,
        chassis: Survivor_Chassis = Survivor_Chassis.WHOLE_LIFE,
        underwriting_class: Underwriting_Class = Underwriting_Class.STANDARD,
        underwriting_class_second: Underwriting_Class = Underwriting_Class.STANDARD,
        is_smoker: bool = False,
        is_smoker_second: bool = False,
        first_death_occurred: bool = False,
        cash_value: float = 0.0,
        rate_guaranteed_interest: float = 0.03,
        rate_COI_insured_1: float = 4.0,
        rate_COI_insured_2: float = 4.0,
        death_benefit_option: Death_Benefit_Option = Death_Benefit_Option.LEVEL,
        has_split_option: bool = False,
        amount_loan_outstanding: float = 0.0,
        rate_loan_interest: float = 0.05,
        rate_surrender_charge: float = 0.0,
        surrender_charge_years: int = 10,
        premium_frequency: Premium_Frequency = Premium_Frequency.ANNUAL,
        beneficiary_primary: str = "",
        beneficiary_contingent: str = "",
    ) -> None:
        """Initialize a Survivor Life Insurance object.

        Parameters
        ----------
        insured_age : int
            Age of the first (primary) insured.
        insured_age_second : int
            Age of the second insured.
        face_amount : float
            Face amount / death benefit.
        chassis : Survivor_Chassis, optional
            Underlying policy structure (default ``WHOLE_LIFE``).
        underwriting_class : Underwriting_Class, optional
            Risk classification for insured 1 (default ``STANDARD``).
        underwriting_class_second : Underwriting_Class, optional
            Risk classification for insured 2 (default ``STANDARD``).
        is_smoker : bool, optional
            Tobacco usage for insured 1 (default ``False``).
        is_smoker_second : bool, optional
            Tobacco usage for insured 2 (default ``False``).
        first_death_occurred : bool, optional
            Whether the first insured has already died (default ``False``).
        cash_value : float, optional
            Current cash value (default 0.0).
        rate_guaranteed_interest : float, optional
            Guaranteed annual interest rate on cash value (default 0.03).
        rate_COI_insured_1 : float, optional
            Monthly COI rate per $1 000 for insured 1 (default 4.0).
        rate_COI_insured_2 : float, optional
            Monthly COI rate per $1 000 for insured 2 (default 4.0).
        death_benefit_option : Death_Benefit_Option, optional
            Level or increasing death benefit (default ``LEVEL``).
        has_split_option : bool, optional
            Split-option rider flag (default ``False``).
        amount_loan_outstanding : float, optional
            Outstanding loan balance (default 0.0).
        rate_loan_interest : float, optional
            Loan interest rate (default 0.05).
        rate_surrender_charge : float, optional
            Current surrender charge rate (default 0.0).
        surrender_charge_years : int, optional
            Surrender charge schedule duration (default 10).
        premium_frequency : Premium_Frequency, optional
            Premium payment frequency (default ``ANNUAL``).
        beneficiary_primary : str, optional
            Primary beneficiary.
        beneficiary_contingent : str, optional
            Contingent beneficiary.

        Raises
        ------
        Exception_Validation_Input
            If any input fails validation.
        """
        super().__init__(
            insured_age=insured_age,
            face_amount=face_amount,
            insurance_type=Insurance_Life_Type.SURVIVOR_LIFE,
            underwriting_class=underwriting_class,
            premium_frequency=premium_frequency,
            is_smoker=is_smoker,
            beneficiary_primary=beneficiary_primary,
            beneficiary_contingent=beneficiary_contingent,
        )

        # --- Validate second insured age ---
        if (
            not isinstance(insured_age_second, int)
            or insured_age_second < 0
            or insured_age_second > 120
        ):
            raise Exception_Validation_Input(
                "insured_age_second must be an integer between 0 and 120",
                field_name="insured_age_second",
                expected_type=int,
                actual_value=insured_age_second,
            )

        if not isinstance(underwriting_class_second, Underwriting_Class):
            raise Exception_Validation_Input(
                "underwriting_class_second must be a valid Underwriting_Class",
                field_name="underwriting_class_second",
                expected_type=Underwriting_Class,
                actual_value=underwriting_class_second,
            )

        if not isinstance(chassis, Survivor_Chassis):
            raise Exception_Validation_Input(
                "chassis must be a valid Survivor_Chassis enum",
                field_name="chassis",
                expected_type=Survivor_Chassis,
                actual_value=chassis,
            )

        if not isinstance(cash_value, (int, float)) or cash_value < 0:
            raise Exception_Validation_Input(
                "cash_value must be a non-negative number",
                field_name="cash_value",
                expected_type=float,
                actual_value=cash_value,
            )

        if not isinstance(rate_guaranteed_interest, (int, float)) or rate_guaranteed_interest < 0:
            raise Exception_Validation_Input(
                "rate_guaranteed_interest must be a non-negative number",
                field_name="rate_guaranteed_interest",
                expected_type=float,
                actual_value=rate_guaranteed_interest,
            )

        if not isinstance(rate_COI_insured_1, (int, float)) or rate_COI_insured_1 < 0:
            raise Exception_Validation_Input(
                "rate_COI_insured_1 must be a non-negative number",
                field_name="rate_COI_insured_1",
                expected_type=float,
                actual_value=rate_COI_insured_1,
            )

        if not isinstance(rate_COI_insured_2, (int, float)) or rate_COI_insured_2 < 0:
            raise Exception_Validation_Input(
                "rate_COI_insured_2 must be a non-negative number",
                field_name="rate_COI_insured_2",
                expected_type=float,
                actual_value=rate_COI_insured_2,
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

        # --- Set member variables ---
        self.m_insured_age_second: int = insured_age_second
        self.m_underwriting_class_second: Underwriting_Class = underwriting_class_second
        self.m_is_smoker_second: bool = bool(is_smoker_second)
        self.m_first_death_occurred: bool = bool(first_death_occurred)
        self.m_chassis: Survivor_Chassis = chassis
        self.m_cash_value: float = float(cash_value)
        self.m_rate_guaranteed_interest: float = float(rate_guaranteed_interest)
        self.m_rate_COI_insured_1: float = float(rate_COI_insured_1)
        self.m_rate_COI_insured_2: float = float(rate_COI_insured_2)
        self.m_death_benefit_option: Death_Benefit_Option = death_benefit_option
        self.m_has_split_option: bool = bool(has_split_option)
        self.m_amount_loan_outstanding: float = float(amount_loan_outstanding)
        self.m_rate_loan_interest: float = float(rate_loan_interest)
        self.m_rate_surrender_charge: float = float(rate_surrender_charge)
        self.m_surrender_charge_years: int = surrender_charge_years

        logger.info(
            f"Created Survivor Life ({chassis.value}): "
            f"face ${face_amount:,.2f}, "
            f"insured ages {insured_age}/{insured_age_second}, "
            f"split option: {has_split_option}",
        )

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def insured_age_second(self) -> int:
        """int : Age of the second insured."""
        return self.m_insured_age_second

    @property
    def underwriting_class_second(self) -> Underwriting_Class:
        """Underwriting_Class : Risk classification for insured 2."""
        return self.m_underwriting_class_second

    @property
    def is_smoker_second(self) -> bool:
        """bool : Tobacco usage for insured 2."""
        return self.m_is_smoker_second

    @property
    def first_death_occurred(self) -> bool:
        """bool : Whether the first death has occurred."""
        return self.m_first_death_occurred

    @property
    def chassis(self) -> Survivor_Chassis:
        """Survivor_Chassis : Underlying policy structure."""
        return self.m_chassis

    @property
    def cash_value(self) -> float:
        """float : Current cash value."""
        return self.m_cash_value

    @property
    def rate_guaranteed_interest(self) -> float:
        """float : Guaranteed annual interest rate on cash value."""
        return self.m_rate_guaranteed_interest

    @property
    def rate_COI_insured_1(self) -> float:
        """float : Monthly COI rate per $1 000 for insured 1."""
        return self.m_rate_COI_insured_1

    @property
    def rate_COI_insured_2(self) -> float:
        """float : Monthly COI rate per $1 000 for insured 2."""
        return self.m_rate_COI_insured_2

    @property
    def death_benefit_option(self) -> Death_Benefit_Option:
        """Death_Benefit_Option : Death benefit option."""
        return self.m_death_benefit_option

    @property
    def has_split_option(self) -> bool:
        """bool : Whether the policy includes a split-option rider."""
        return self.m_has_split_option

    @property
    def amount_loan_outstanding(self) -> float:
        """float : Outstanding loan balance."""
        return self.m_amount_loan_outstanding

    @property
    def rate_loan_interest(self) -> float:
        """float : Loan interest rate."""
        return self.m_rate_loan_interest

    @property
    def rate_surrender_charge(self) -> float:
        """float : Current surrender charge rate."""
        return self.m_rate_surrender_charge

    @property
    def surrender_charge_years(self) -> int:
        """int : Surrender charge schedule duration."""
        return self.m_surrender_charge_years

    # ------------------------------------------------------------------
    # Joint mortality / COI calculations
    # ------------------------------------------------------------------

    def calc_joint_COI(self) -> float:
        r"""Calculate the joint last-to-die cost of insurance rate.

        Uses a simplified joint-life formula:

        $$
        COI_{\text{joint}} = \frac{COI_1 \times COI_2}{COI_1 + COI_2}
        $$

        This reflects that the second death must occur before the
        benefit is payable, resulting in a lower effective COI.

        If the first death has already occurred, the COI reverts to the
        surviving insured's individual rate.

        Returns
        -------
        float
            Joint COI rate per $1 000 of NAR.
        """
        if self.m_first_death_occurred:
            # After first death, COI = surviving insured's rate.
            # Simplified: use the higher of the two as a conservative
            # estimate (in practice would depend on which insured died).
            return max(self.m_rate_COI_insured_1, self.m_rate_COI_insured_2)

        coi_sum = self.m_rate_COI_insured_1 + self.m_rate_COI_insured_2
        if coi_sum == 0:
            return 0.0

        return (self.m_rate_COI_insured_1 * self.m_rate_COI_insured_2) / coi_sum

    def calc_joint_equivalent_age(self) -> int:
        """Calculate the joint equivalent age for pricing purposes.

        The joint equivalent age is a single equivalent age that
        produces a similar mortality expectation to the two-life joint
        last-to-die arrangement.  Commonly approximated as:

        age_joint = (age_1 + age_2) / 2  (rounded down)

        A more precise formula would use actual mortality tables.

        Returns
        -------
        int
            Joint equivalent age.
        """
        return (self.m_insured_age + self.m_insured_age_second) // 2

    # ------------------------------------------------------------------
    # Death benefit calculations
    # ------------------------------------------------------------------

    def calc_death_benefit(self) -> float:
        r"""Calculate the current death benefit payable.

        **Level**: DB = face amount.
        **Increasing**: DB = face amount + cash value.
        Both are reduced by the outstanding loan balance.

        $$
        DB = \begin{cases}
            F - L & \text{Level} \\
            F + CV - L & \text{Increasing}
        \end{cases}
        $$

        Returns
        -------
        float
            Net death benefit (floored at 0).
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
        r"""Calculate the estimated annual premium.

        Uses the joint COI rate:

        $$
        P = \frac{F}{1\,000} \times COI_{\text{joint}} \times 12 \times (1 + e)
        $$

        Where $e$ is an expense loading factor (15 %).

        Returns
        -------
        float
            Estimated annual premium.
        """
        expense_loading: float = 0.15
        joint_coi = self.calc_joint_COI()
        annual_coi = (self.m_face_amount / 1_000) * joint_coi * 12
        return annual_coi * (1 + expense_loading)

    def calc_modal_premium(self) -> float:
        """Calculate the per-payment premium based on payment frequency.

        Returns
        -------
        float
            Premium per payment period.
        """
        annual = self.calc_annual_premium()
        factor = self.calc_modal_premium_factor()
        if self.m_premium_frequency.value == 0:
            return annual
        return annual * factor

    def calc_premium_savings_vs_individual(
        self,
        individual_premium_insured_1: float,
        individual_premium_insured_2: float,
    ) -> float:
        r"""Calculate the annual premium savings compared to two individual policies.

        $$
        \Delta P = (P_1 + P_2) - P_{\text{survivor}}
        $$

        Parameters
        ----------
        individual_premium_insured_1 : float
            Annual premium for insured 1's individual policy.
        individual_premium_insured_2 : float
            Annual premium for insured 2's individual policy.

        Returns
        -------
        float
            Annual savings (positive = survivor is cheaper).

        Raises
        ------
        Exception_Validation_Input
            If premiums are negative.
        """
        if individual_premium_insured_1 < 0 or individual_premium_insured_2 < 0:
            raise Exception_Validation_Input(
                "Individual premiums must be non-negative",
                field_name="individual_premium",
                expected_type=float,
                actual_value=(
                    individual_premium_insured_1,
                    individual_premium_insured_2,
                ),
            )

        individual_total = individual_premium_insured_1 + individual_premium_insured_2
        return individual_total - self.calc_annual_premium()

    # ------------------------------------------------------------------
    # Cash value projections
    # ------------------------------------------------------------------

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

    def calc_monthly_COI_charge(self) -> float:
        r"""Calculate the monthly cost-of-insurance deduction.

        Uses the joint COI rate:

        $$
        COI_m = \frac{NAR}{1\,000} \times COI_{\text{joint}}
        $$

        Returns
        -------
        float
            Monthly COI charge.
        """
        nar = self.calc_net_amount_at_risk()
        joint_coi = self.calc_joint_COI()
        return (nar / 1_000) * joint_coi

    def calc_cash_value_at_year(self, years: int) -> float:
        r"""Project cash value at the end of a number of years.

        Assumes annual premium payments and monthly compounding:

        $$
        CV_y = CV_{y-1} \times \left(1 + r_g\right) + P
            - COI_{\text{joint}} \times \frac{NAR}{1\,000} \times 12
        $$

        Parameters
        ----------
        years : int
            Number of years to project.

        Returns
        -------
        float
            Projected cash value.

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

        annual_premium = self.calc_annual_premium()
        joint_coi = self.calc_joint_COI()

        cv: float = self.m_cash_value
        for _ in range(years):
            # Annual interest credit
            cv = cv * (1 + self.m_rate_guaranteed_interest)
            # Add annual premium
            cv += annual_premium
            # Deduct annual COI (12 months)
            nar = max(self.m_face_amount - cv, 0.0)
            annual_coi_charge = (nar / 1_000) * joint_coi * 12
            cv -= annual_coi_charge

        return max(cv, 0.0)

    def calc_cash_surrender_value(self, policy_year: int = 0) -> float:
        """Calculate the cash surrender value.

        Cash surrender value = cash value - surrender charge - loan balance.

        Parameters
        ----------
        policy_year : int, optional
            Current policy year (default 0).

        Returns
        -------
        float
            Cash surrender value (floored at 0).
        """
        if 0 < policy_year <= self.m_surrender_charge_years:
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
    # Estate-planning helpers
    # ------------------------------------------------------------------

    def calc_estate_tax_coverage(
        self,
        estate_value: float,
        estate_tax_rate: float = 0.40,
        unified_credit_exclusion: float = 12_920_000.0,
    ) -> float:
        r"""Estimate whether the death benefit covers projected estate taxes.

        $$
        \text{Tax} = \max\!\bigl(0,\; (E - U) \times t_e\bigr)
        $$

        $$
        \text{Coverage Ratio} = \frac{DB}{\text{Tax}}
        $$

        Parameters
        ----------
        estate_value : float
            Estimated gross estate value.
        estate_tax_rate : float, optional
            Federal estate tax rate (default 0.40 = 40 %).
        unified_credit_exclusion : float, optional
            Unified credit / exemption amount (default $12,920,000).

        Returns
        -------
        float
            Coverage ratio (1.0 = fully covered, > 1.0 = over-insured).

        Raises
        ------
        Exception_Validation_Input
            If ``estate_value`` is negative.
        """
        if not isinstance(estate_value, (int, float)) or estate_value < 0:
            raise Exception_Validation_Input(
                "estate_value must be a non-negative number",
                field_name="estate_value",
                expected_type=float,
                actual_value=estate_value,
            )

        taxable_estate = max(estate_value - unified_credit_exclusion, 0.0)
        estimated_tax = taxable_estate * estate_tax_rate

        if estimated_tax <= 0:
            return float("inf")  # No tax due; coverage is infinite

        db = self.calc_death_benefit()
        return db / estimated_tax

    # ------------------------------------------------------------------
    # String representation
    # ------------------------------------------------------------------

    def get_insurance_as_string(self) -> str:
        """Return a human-readable string representation.

        Returns
        -------
        str
            String representation of the Survivor Life policy.
        """
        split_text = ", Split Option" if self.m_has_split_option else ""
        status_text = " [1st death occurred]" if self.m_first_death_occurred else ""
        return (
            f"Survivor Life ({self.m_chassis.value}{split_text}){status_text}: "
            f"Ages {self.m_insured_age}/{self.m_insured_age_second}, "
            f"Face ${self.m_face_amount:,.2f}, "
            f"CV ${self.m_cash_value:,.2f}, "
            f"Joint COI {self.calc_joint_COI():.2f}, "
            f"Classes: {self.m_underwriting_class.value}"
            f"/{self.m_underwriting_class_second.value}"
        )
