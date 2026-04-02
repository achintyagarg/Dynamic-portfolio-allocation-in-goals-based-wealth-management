"""
Life insurance base class.

===============================

Abstract base class defining the interface and common functionality
for all life insurance product types (Whole, Term, Universal, Variable,
Survivor).

Key features common to every life insurance contract:

- **Face amount (death benefit)**: the sum paid to beneficiaries on the
  insured's death.
- **Premium**: regular or single payment(s) that fund the policy.
- **Underwriting class**: preferred-plus, preferred, standard, or substandard
  rating that drives mortality charges.
- **Beneficiary designation**: primary and contingent beneficiaries.
- **Policy riders**: optional add-ons (waiver of premium, accidental death,
  accelerated death benefit, etc.).

Author
------
QWIM Team

Version
-------
0.6.0 (2026-02-13)
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from aenum import Enum

from src.utils.custom_exceptions_errors_loggers.exception_custom import (
    Exception_Validation_Input,
)
from src.utils.custom_exceptions_errors_loggers.logger_custom import get_logger


logger = get_logger(__name__)


# ======================================================================
# Enumerations
# ======================================================================


class Insurance_Life_Type(Enum):
    """Enumeration for different types of life insurance.

    Attributes
    ----------
    WHOLE_LIFE : str
        Whole Life Insurance — permanent coverage with guaranteed cash value.
    TERM_LIFE : str
        Term Life Insurance — coverage for a specified number of years.
    UNIVERSAL_LIFE : str
        Universal Life Insurance — flexible-premium permanent coverage.
    VARIABLE_LIFE : str
        Variable Life Insurance — permanent coverage with market-linked
        sub-accounts.
    SURVIVOR_LIFE : str
        Survivor (Second-to-Die) Life Insurance — pays on the death of
        the second insured.
    """

    WHOLE_LIFE = "Whole Life Insurance"
    TERM_LIFE = "Term Life Insurance"
    UNIVERSAL_LIFE = "Universal Life Insurance"
    VARIABLE_LIFE = "Variable Life Insurance"
    SURVIVOR_LIFE = "Survivor Life Insurance"


class Underwriting_Class(Enum):
    """Underwriting risk classification for life insurance applicants.

    Attributes
    ----------
    PREFERRED_PLUS : str
        Best risk class — excellent health, no tobacco, favourable family history.
    PREFERRED : str
        Above-average health, minor risk factors tolerated.
    STANDARD : str
        Average risk — the baseline rating class.
    SUBSTANDARD : str
        Below-average risk — table-rated; extra mortality charge applied.
    """

    PREFERRED_PLUS = "Preferred Plus"
    PREFERRED = "Preferred"
    STANDARD = "Standard"
    SUBSTANDARD = "Substandard"


class Premium_Frequency(Enum):
    """Payment frequency for insurance premiums.

    Attributes
    ----------
    ANNUAL : int
        One payment per year.
    SEMI_ANNUAL : int
        Two payments per year.
    QUARTERLY : int
        Four payments per year.
    MONTHLY : int
        Twelve payments per year.
    SINGLE : int
        Single lump-sum premium (one-time payment).
    """

    ANNUAL = 1
    SEMI_ANNUAL = 2
    QUARTERLY = 4
    MONTHLY = 12
    SINGLE = 0


class Death_Benefit_Option(Enum):
    """Death benefit options for life insurance policies.

    Attributes
    ----------
    LEVEL : str
        Level death benefit — face amount remains constant.
    INCREASING : str
        Increasing death benefit — face amount plus accumulated cash value.
    RETURN_OF_PREMIUM : str
        Return of premium — face amount plus total premiums paid.
    """

    LEVEL = "Level"
    INCREASING = "Increasing"
    RETURN_OF_PREMIUM = "Return of Premium"


# ======================================================================
# Base class
# ======================================================================


class Insurance_Life_Base(ABC):
    """Abstract base class for life insurance products.

    This class defines the interface and common functionality shared by
    all life insurance types.  Derived classes must implement the
    :meth:`calc_death_benefit` and :meth:`calc_annual_premium` methods.

    Attributes
    ----------
    m_insured_age : int
        Current age of the insured.
    m_face_amount : float
        Face amount (death benefit) of the policy.
    m_insurance_type : Insurance_Life_Type
        The type of life insurance.
    m_underwriting_class : Underwriting_Class
        Underwriting risk classification.
    m_premium_frequency : Premium_Frequency
        How frequently premiums are paid.
    m_is_smoker : bool
        Whether the insured uses tobacco products.
    m_beneficiary_primary : str
        Name or designation of the primary beneficiary.
    m_beneficiary_contingent : str
        Name or designation of the contingent beneficiary.
    """

    def __init__(
        self,
        insured_age: int,
        face_amount: float,
        insurance_type: Insurance_Life_Type,
        underwriting_class: Underwriting_Class = Underwriting_Class.STANDARD,
        premium_frequency: Premium_Frequency = Premium_Frequency.MONTHLY,
        is_smoker: bool = False,
        beneficiary_primary: str = "",
        beneficiary_contingent: str = "",
    ) -> None:
        """Initialize a life insurance base object.

        Parameters
        ----------
        insured_age : int
            Current age of the insured (must be 0 – 120).
        face_amount : float
            Face amount / death benefit (must be positive).
        insurance_type : Insurance_Life_Type
            The specific life insurance type.
        underwriting_class : Underwriting_Class, optional
            Risk classification (default ``STANDARD``).
        premium_frequency : Premium_Frequency, optional
            Premium payment frequency (default ``MONTHLY``).
        is_smoker : bool, optional
            Tobacco usage flag (default ``False``).
        beneficiary_primary : str, optional
            Primary beneficiary designation (default ``""``).
        beneficiary_contingent : str, optional
            Contingent beneficiary designation (default ``""``).

        Raises
        ------
        Exception_Validation_Input
            If any input fails validation.
        """
        # --- Input validation with early returns ---
        if not isinstance(insured_age, int) or insured_age < 0 or insured_age > 120:
            raise Exception_Validation_Input(
                "insured_age must be an integer in [0, 120]",
                field_name="insured_age",
                expected_type=int,
                actual_value=insured_age,
            )

        if not isinstance(face_amount, (int, float)) or face_amount <= 0:
            raise Exception_Validation_Input(
                "face_amount must be a positive number",
                field_name="face_amount",
                expected_type=float,
                actual_value=face_amount,
            )

        if not isinstance(insurance_type, Insurance_Life_Type):
            raise Exception_Validation_Input(
                "insurance_type must be a valid Insurance_Life_Type enum",
                field_name="insurance_type",
                expected_type=Insurance_Life_Type,
                actual_value=insurance_type,
            )

        if not isinstance(underwriting_class, Underwriting_Class):
            raise Exception_Validation_Input(
                "underwriting_class must be a valid Underwriting_Class enum",
                field_name="underwriting_class",
                expected_type=Underwriting_Class,
                actual_value=underwriting_class,
            )

        if not isinstance(premium_frequency, Premium_Frequency):
            raise Exception_Validation_Input(
                "premium_frequency must be a valid Premium_Frequency enum",
                field_name="premium_frequency",
                expected_type=Premium_Frequency,
                actual_value=premium_frequency,
            )

        # --- Set member variables ---
        self.m_insured_age: int = insured_age
        self.m_face_amount: float = float(face_amount)
        self.m_insurance_type: Insurance_Life_Type = insurance_type
        self.m_underwriting_class: Underwriting_Class = underwriting_class
        self.m_premium_frequency: Premium_Frequency = premium_frequency
        self.m_is_smoker: bool = bool(is_smoker)
        self.m_beneficiary_primary: str = str(beneficiary_primary)
        self.m_beneficiary_contingent: str = str(beneficiary_contingent)

        logger.info(
            f"Created {insurance_type.value} for insured age {insured_age}, "
            f"face amount ${face_amount:,.2f}, "
            f"class: {underwriting_class.value}, "
            f"smoker: {is_smoker}",
        )

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def insured_age(self) -> int:
        """int : Current age of the insured."""
        return self.m_insured_age

    @property
    def face_amount(self) -> float:
        """float : Face amount (death benefit) of the policy."""
        return self.m_face_amount

    @property
    def insurance_type(self) -> Insurance_Life_Type:
        """Insurance_Life_Type : The type of life insurance."""
        return self.m_insurance_type

    @property
    def underwriting_class(self) -> Underwriting_Class:
        """Underwriting_Class : Underwriting risk classification."""
        return self.m_underwriting_class

    @property
    def premium_frequency(self) -> Premium_Frequency:
        """Premium_Frequency : Premium payment frequency."""
        return self.m_premium_frequency

    @property
    def is_smoker(self) -> bool:
        """bool : Whether the insured uses tobacco."""
        return self.m_is_smoker

    @property
    def beneficiary_primary(self) -> str:
        """str : Primary beneficiary designation."""
        return self.m_beneficiary_primary

    @property
    def beneficiary_contingent(self) -> str:
        """str : Contingent beneficiary designation."""
        return self.m_beneficiary_contingent

    # ------------------------------------------------------------------
    # Calculation helpers (common)
    # ------------------------------------------------------------------

    def calc_modal_premium_factor(self) -> float:
        """Return the modal premium factor for the payment frequency.

        Insurance companies typically apply a factor to the annual premium
        when premiums are paid more frequently than annually:

        - Annual: 1.000
        - Semi-annual: 0.515
        - Quarterly: 0.265
        - Monthly: 0.0875
        - Single: 1.000

        Returns
        -------
        float
            The modal premium factor.
        """
        modal_factors: dict[Premium_Frequency, float] = {
            Premium_Frequency.ANNUAL: 1.0,
            Premium_Frequency.SEMI_ANNUAL: 0.515,
            Premium_Frequency.QUARTERLY: 0.265,
            Premium_Frequency.MONTHLY: 0.0875,
            Premium_Frequency.SINGLE: 1.0,
        }
        return modal_factors.get(self.m_premium_frequency, 1.0)

    def calc_cost_per_thousand(self, annual_premium: float) -> float:
        r"""Calculate the annual cost per $1 000 of death benefit.

        $$
        \text{Cost per } \$1\,000 = \frac{P_{\text{annual}}}{F / 1\,000}
        $$

        Parameters
        ----------
        annual_premium : float
            The annual premium amount.

        Returns
        -------
        float
            Cost per $1 000 of coverage.

        Raises
        ------
        Exception_Validation_Input
            If ``annual_premium`` is not positive.
        """
        if not isinstance(annual_premium, (int, float)) or annual_premium <= 0:
            raise Exception_Validation_Input(
                "annual_premium must be a positive number",
                field_name="annual_premium",
                expected_type=float,
                actual_value=annual_premium,
            )

        return annual_premium / (self.m_face_amount / 1_000)

    # ------------------------------------------------------------------
    # Abstract methods
    # ------------------------------------------------------------------

    @abstractmethod
    def calc_death_benefit(self) -> float:
        """Calculate the current death benefit payable.

        Must be implemented by each derived class to account for
        product-specific features (e.g. cash-value additions, investment
        gains, or return-of-premium riders).

        Returns
        -------
        float
            The death benefit amount.
        """

    @abstractmethod
    def calc_annual_premium(self) -> float:
        """Calculate the annual premium for the policy.

        Must be implemented by each derived class because premium
        calculation logic varies significantly by product type.

        Returns
        -------
        float
            The annual premium amount.
        """

    # ------------------------------------------------------------------
    # String representation
    # ------------------------------------------------------------------

    def get_insurance_as_string(self) -> str:
        """Return a human-readable string representation.

        Returns
        -------
        str
            String representation of the life insurance policy.
        """
        smoker_text = "Smoker" if self.m_is_smoker else "Non-Smoker"
        return (
            f"{self.m_insurance_type.value} "
            f"(Age: {self.m_insured_age}, "
            f"Face: ${self.m_face_amount:,.2f}, "
            f"Class: {self.m_underwriting_class.value}, "
            f"{smoker_text})"
        )
