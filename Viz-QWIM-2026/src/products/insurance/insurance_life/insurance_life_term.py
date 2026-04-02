"""
Term Life Insurance class.

===============================

Term Life Insurance provides coverage for a specified number of years
(the "term") with no cash-value accumulation.  Key characteristics:

- **Temporary coverage**: protection for a fixed period (10, 15, 20, 25,
  or 30 years are common).
- **Low initial cost**: premiums are significantly lower than permanent
  insurance at the same face amount because there is no savings component.
- **Level or annually-renewable premiums**: level-term policies lock the
  premium for the full term; annually renewable term (ART) premiums
  increase each year with age.
- **Convertibility**: many term policies include a conversion privilege
  allowing the owner to convert to a permanent policy without evidence
  of insurability.
- **Renewability**: guaranteed-renewable provisions allow renewal at the
  end of the term at attained-age rates without medical underwriting.
- **Return-of-premium (ROP)**: an optional rider that refunds all
  premiums paid if the insured survives the term period.
- **No cash value**: upon lapse or expiration the policy has no residual
  value.

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
    Insurance_Life_Base,
    Insurance_Life_Type,
    Premium_Frequency,
    Underwriting_Class,
)


logger = get_logger(__name__)


# ======================================================================
# Term-specific enumerations
# ======================================================================


class Term_Type(Enum):
    """Term insurance structure variant.

    Attributes
    ----------
    LEVEL : str
        Level-premium term — premium is constant over the full term.
    ANNUALLY_RENEWABLE : str
        Annually Renewable Term (ART) — premium increases each year.
    DECREASING : str
        Decreasing term — face amount declines over the term while
        premiums stay level (common for mortgage protection).
    """

    LEVEL = "Level Term"
    ANNUALLY_RENEWABLE = "Annually Renewable Term"
    DECREASING = "Decreasing Term"


class Insurance_Life_Term(Insurance_Life_Base):
    r"""Term Life Insurance product class.

    Provides a death benefit for a fixed period with no cash-value
    accumulation.

    **Level-term annual premium estimate**:

    $$
    P_{\text{annual}} = \frac{F}{1\,000} \times q_{\text{avg}} \times (1 + e)
    $$

    Where $q_{\text{avg}}$ = average annual mortality cost over the term,
    $e$ = expense loading.

    **Annually Renewable Term premium in year** $t$:

    $$
    P_t = \frac{F}{1\,000} \times q_{x+t-1} \times (1 + e)
    $$

    Where $q_{x+t-1}$ = mortality rate at attained age $x + t - 1$.

    **Decreasing term face amount in year** $t$:

    $$
    F_t = F \times \frac{n - t + 1}{n}
    $$

    Where $n$ = original term length.

    Attributes
    ----------
    m_term_years : int
        Duration of coverage in years.
    m_term_type : Term_Type
        Level, annually-renewable, or decreasing term.
    m_is_convertible : bool
        Whether the policy includes a conversion privilege.
    m_conversion_deadline_year : int
        Last policy year in which conversion is allowed.
    m_is_renewable : bool
        Whether the policy is guaranteed renewable at expiry.
    m_has_return_of_premium : bool
        Whether the ROP rider is attached.
    m_rate_cost_of_insurance : float
        Average annual COI rate per $1 000 of coverage.
    m_payment_frequency : int
        Number of premium payments per year.
    """

    def __init__(
        self,
        insured_age: int,
        face_amount: float,
        term_years: int = 20,
        term_type: Term_Type = Term_Type.LEVEL,
        is_convertible: bool = True,
        conversion_deadline_year: int = 0,
        is_renewable: bool = True,
        has_return_of_premium: bool = False,
        rate_cost_of_insurance: float = 2.5,
        underwriting_class: Underwriting_Class = Underwriting_Class.STANDARD,
        is_smoker: bool = False,
        payment_frequency: int = 12,
        beneficiary_primary: str = "",
        beneficiary_contingent: str = "",
    ) -> None:
        """Initialize a Term Life Insurance object.

        Parameters
        ----------
        insured_age : int
            Current age of the insured.
        face_amount : float
            Face amount (death benefit).
        term_years : int, optional
            Duration of coverage in years (default 20).
        term_type : Term_Type, optional
            Term structure variant (default ``LEVEL``).
        is_convertible : bool, optional
            Conversion privilege flag (default ``True``).
        conversion_deadline_year : int, optional
            Last year conversion is allowed (default 0 = no limit or
            same as term_years).
        is_renewable : bool, optional
            Guaranteed renewable at expiry (default ``True``).
        has_return_of_premium : bool, optional
            Return-of-premium rider attached (default ``False``).
        rate_cost_of_insurance : float, optional
            Average annual COI rate per $1 000 (default 2.5).
        underwriting_class : Underwriting_Class, optional
            Risk classification (default ``STANDARD``).
        is_smoker : bool, optional
            Tobacco usage flag (default ``False``).
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
            insurance_type=Insurance_Life_Type.TERM_LIFE,
            underwriting_class=underwriting_class,
            premium_frequency=freq_enum,
            is_smoker=is_smoker,
            beneficiary_primary=beneficiary_primary,
            beneficiary_contingent=beneficiary_contingent,
        )

        # --- Input validation ---
        if not isinstance(term_years, int) or term_years <= 0:
            raise Exception_Validation_Input(
                "term_years must be a positive integer",
                field_name="term_years",
                expected_type=int,
                actual_value=term_years,
            )

        if not isinstance(term_type, Term_Type):
            raise Exception_Validation_Input(
                "term_type must be a valid Term_Type enum",
                field_name="term_type",
                expected_type=Term_Type,
                actual_value=term_type,
            )

        if not isinstance(conversion_deadline_year, int) or conversion_deadline_year < 0:
            raise Exception_Validation_Input(
                "conversion_deadline_year must be a non-negative integer",
                field_name="conversion_deadline_year",
                expected_type=int,
                actual_value=conversion_deadline_year,
            )

        if not isinstance(rate_cost_of_insurance, (int, float)) or rate_cost_of_insurance < 0:
            raise Exception_Validation_Input(
                "rate_cost_of_insurance must be a non-negative number",
                field_name="rate_cost_of_insurance",
                expected_type=float,
                actual_value=rate_cost_of_insurance,
            )

        if not isinstance(payment_frequency, int) or payment_frequency < 0:
            raise Exception_Validation_Input(
                "payment_frequency must be a non-negative integer",
                field_name="payment_frequency",
                expected_type=int,
                actual_value=payment_frequency,
            )

        # --- Set member variables ---
        self.m_term_years: int = term_years
        self.m_term_type: Term_Type = term_type
        self.m_is_convertible: bool = bool(is_convertible)
        self.m_conversion_deadline_year: int = (
            conversion_deadline_year if conversion_deadline_year > 0 else term_years
        )
        self.m_is_renewable: bool = bool(is_renewable)
        self.m_has_return_of_premium: bool = bool(has_return_of_premium)
        self.m_rate_cost_of_insurance: float = float(rate_cost_of_insurance)
        self.m_payment_frequency: int = payment_frequency

        logger.info(
            f"Created Term Life: face ${face_amount:,.2f}, "
            f"term {term_years} yr ({term_type.value}), "
            f"convertible: {is_convertible}, renewable: {is_renewable}, "
            f"ROP: {has_return_of_premium}",
        )

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def term_years(self) -> int:
        """int : Duration of coverage in years."""
        return self.m_term_years

    @property
    def term_type(self) -> Term_Type:
        """Term_Type : Level, annually-renewable, or decreasing term."""
        return self.m_term_type

    @property
    def is_convertible(self) -> bool:
        """bool : Whether the policy includes a conversion privilege."""
        return self.m_is_convertible

    @property
    def conversion_deadline_year(self) -> int:
        """int : Last policy year conversion is allowed."""
        return self.m_conversion_deadline_year

    @property
    def is_renewable(self) -> bool:
        """bool : Whether the policy is guaranteed renewable."""
        return self.m_is_renewable

    @property
    def has_return_of_premium(self) -> bool:
        """bool : Whether the ROP rider is attached."""
        return self.m_has_return_of_premium

    @property
    def rate_cost_of_insurance(self) -> float:
        """float : Average annual COI rate per $1 000."""
        return self.m_rate_cost_of_insurance

    @property
    def payment_frequency(self) -> int:
        """int : Number of premium payments per year."""
        return self.m_payment_frequency

    # ------------------------------------------------------------------
    # Death benefit calculations
    # ------------------------------------------------------------------

    def calc_death_benefit(self) -> float:
        """Calculate the current death benefit payable.

        For level and ART term the death benefit equals the face amount.
        For decreasing term, it cannot be computed without knowing the
        current policy year; this method returns the full face amount.
        Use :meth:`calc_death_benefit_at_year` for year-specific values.

        Returns
        -------
        float
            The death benefit amount (face amount).
        """
        return self.m_face_amount

    def calc_death_benefit_at_year(self, policy_year: int) -> float:
        r"""Calculate the death benefit in a specific policy year.

        For **decreasing term**:

        $$
        F_t = F \times \frac{n - t + 1}{n}
        $$

        For all other term types the face amount is constant.

        Parameters
        ----------
        policy_year : int
            The policy year (1-indexed).

        Returns
        -------
        float
            The death benefit in the specified year.

        Raises
        ------
        Exception_Validation_Input
            If ``policy_year`` is out of range.
        """
        if not isinstance(policy_year, int) or policy_year < 1:
            raise Exception_Validation_Input(
                "policy_year must be a positive integer",
                field_name="policy_year",
                expected_type=int,
                actual_value=policy_year,
            )

        if policy_year > self.m_term_years:
            return 0.0

        if self.m_term_type == Term_Type.DECREASING:
            return self.m_face_amount * ((self.m_term_years - policy_year + 1) / self.m_term_years)

        return self.m_face_amount

    # ------------------------------------------------------------------
    # Premium calculations
    # ------------------------------------------------------------------

    def calc_annual_premium(self) -> float:
        r"""Calculate the estimated level annual premium.

        $$
        P_{\text{annual}} = \frac{F}{1\,000} \times q \times (1 + e)
        $$

        For ROP policies a surcharge of 60 % is applied.

        Returns
        -------
        float
            The estimated annual premium.
        """
        expense_loading: float = 0.15
        net_premium = (self.m_face_amount / 1_000) * self.m_rate_cost_of_insurance

        gross_premium = net_premium * (1 + expense_loading)

        # ROP rider surcharge
        if self.m_has_return_of_premium:
            gross_premium *= 1.60

        return gross_premium

    def calc_annual_premium_at_year(self, policy_year: int) -> float:
        r"""Calculate the premium for a specific policy year.

        For **level** and **decreasing** term the premium is the same
        each year.  For **ART** the premium increases with age.

        Parameters
        ----------
        policy_year : int
            The policy year (1-indexed).

        Returns
        -------
        float
            The annual premium in the specified year.

        Raises
        ------
        Exception_Validation_Input
            If ``policy_year`` is out of range.
        """
        if not isinstance(policy_year, int) or policy_year < 1:
            raise Exception_Validation_Input(
                "policy_year must be a positive integer",
                field_name="policy_year",
                expected_type=int,
                actual_value=policy_year,
            )

        if policy_year > self.m_term_years:
            return 0.0

        if self.m_term_type == Term_Type.ANNUALLY_RENEWABLE:
            # ART: apply age-based multiplier (simplified ~3 % increase/year)
            base_premium = self.calc_annual_premium()
            return base_premium * (1.03 ** (policy_year - 1))

        return self.calc_annual_premium()

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
    # Total cost / ROP calculations
    # ------------------------------------------------------------------

    def calc_total_premiums_paid(self) -> float:
        """Calculate total premiums over the full term.

        For ART this sums projected year-by-year premiums.

        Returns
        -------
        float
            Total premiums paid over the term.
        """
        if self.m_term_type == Term_Type.ANNUALLY_RENEWABLE:
            total: float = 0.0
            for yr in range(1, self.m_term_years + 1):
                total += self.calc_annual_premium_at_year(yr)
            return total

        return self.calc_annual_premium() * self.m_term_years

    def calc_return_of_premium_benefit(self) -> float:
        """Calculate the ROP benefit at end of term.

        Returns
        -------
        float
            The refund amount (total premiums if ROP, else 0).
        """
        if not self.m_has_return_of_premium:
            return 0.0
        return self.calc_total_premiums_paid()

    # ------------------------------------------------------------------
    # Conversion analysis
    # ------------------------------------------------------------------

    def is_conversion_available(self, current_policy_year: int) -> bool:
        """Check whether conversion is available in the given policy year.

        Parameters
        ----------
        current_policy_year : int
            The current policy year (1-indexed).

        Returns
        -------
        bool
            ``True`` if conversion is still available.
        """
        if not self.m_is_convertible:
            return False
        return current_policy_year <= self.m_conversion_deadline_year

    def calc_remaining_term_years(self, current_policy_year: int) -> int:
        """Calculate remaining years in the term.

        Parameters
        ----------
        current_policy_year : int
            The current policy year (1-indexed).

        Returns
        -------
        int
            Remaining coverage years (floored at 0).
        """
        return max(self.m_term_years - current_policy_year + 1, 0)

    # ------------------------------------------------------------------
    # Cost efficiency analysis
    # ------------------------------------------------------------------

    def calc_cost_per_thousand_per_year(self, policy_year: int = 1) -> float:
        r"""Calculate the cost per $1 000 of coverage in a specific year.

        $$
        \text{Cost per } \$1\,000 = \frac{P_t}{F_t / 1\,000}
        $$

        Parameters
        ----------
        policy_year : int, optional
            The policy year (default 1).

        Returns
        -------
        float
            Cost per $1 000 of coverage for that year.
        """
        premium = self.calc_annual_premium_at_year(policy_year)
        db = self.calc_death_benefit_at_year(policy_year)

        if db <= 0:
            return 0.0

        return premium / (db / 1_000)

    # ------------------------------------------------------------------
    # String representation
    # ------------------------------------------------------------------

    def get_insurance_as_string(self) -> str:
        """Return a human-readable string representation.

        Returns
        -------
        str
            String representation of the Term Life policy.
        """
        features: list[str] = []
        if self.m_is_convertible:
            features.append("Convertible")
        if self.m_is_renewable:
            features.append("Renewable")
        if self.m_has_return_of_premium:
            features.append("ROP")

        features_text = ", ".join(features) if features else "Basic"

        return (
            f"Term Life ({self.m_term_type.value}, {self.m_term_years}yr, "
            f"{features_text}): "
            f"Age {self.m_insured_age}, "
            f"Face ${self.m_face_amount:,.2f}, "
            f"Class: {self.m_underwriting_class.value}"
        )
