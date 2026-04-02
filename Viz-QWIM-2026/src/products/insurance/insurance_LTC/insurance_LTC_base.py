"""
Long Term Care (LTC) insurance base class.

===============================

Abstract base class defining the interface and common functionality
for all Long Term Care (LTC) insurance product types (Traditional,
Hybrid Life, Hybrid Annuity).

Key features common to every LTC insurance contract:

- **Daily / monthly benefit amount**: the maximum amount the policy pays
  per day or per month toward qualifying LTC expenses.
- **Benefit period**: the maximum duration (in years) for which the
  policy will pay benefits (e.g. 2, 3, 5 years, or lifetime).
- **Elimination period**: the number of days the insured must pay
  out-of-pocket before benefits begin (typically 30, 60, or 90 days).
- **Inflation protection**: optional rider that increases benefits over
  time to keep pace with rising care costs.
- **Benefit triggers**: activities of daily living (ADL) impairment or
  cognitive impairment that activate benefit payments.
- **Care settings**: coverage may apply to nursing home, assisted living,
  home health care, adult day care, or hospice.

Author
------
QWIM Team

Version
-------
0.7.0 (2026-02-13)
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


class Insurance_LTC_Type(Enum):
    """Enumeration for different types of LTC insurance.

    Attributes
    ----------
    TRADITIONAL : str
        Stand-alone traditional LTC policy.
    HYBRID_LIFE : str
        LTC rider attached to a life insurance chassis.
    HYBRID_ANNUITY : str
        LTC rider attached to an annuity chassis.
    """

    TRADITIONAL = "Traditional LTC Insurance"
    HYBRID_LIFE = "Hybrid Life LTC Insurance"
    HYBRID_ANNUITY = "Hybrid Annuity LTC Insurance"


class Benefit_Period(Enum):
    """Maximum benefit payment duration.

    Attributes
    ----------
    YEARS_2 : int
        Two-year benefit period.
    YEARS_3 : int
        Three-year benefit period.
    YEARS_4 : int
        Four-year benefit period.
    YEARS_5 : int
        Five-year benefit period.
    YEARS_6 : int
        Six-year benefit period.
    LIFETIME : int
        Unlimited / lifetime benefit period (represented as 100 years).
    """

    YEARS_2 = 2
    YEARS_3 = 3
    YEARS_4 = 4
    YEARS_5 = 5
    YEARS_6 = 6
    LIFETIME = 100


class Elimination_Period(Enum):
    """Waiting period (in days) before benefits begin.

    Attributes
    ----------
    DAYS_0 : int
        Zero-day elimination (immediate benefits).
    DAYS_30 : int
        Thirty-day elimination period.
    DAYS_60 : int
        Sixty-day elimination period.
    DAYS_90 : int
        Ninety-day elimination period.
    DAYS_180 : int
        One-hundred-eighty-day elimination period.
    DAYS_365 : int
        Three-hundred-sixty-five-day elimination period.
    """

    DAYS_0 = 0
    DAYS_30 = 30
    DAYS_60 = 60
    DAYS_90 = 90
    DAYS_180 = 180
    DAYS_365 = 365


class Inflation_Protection(Enum):
    """Inflation protection options for LTC benefits.

    Attributes
    ----------
    NONE : str
        No inflation protection.
    SIMPLE_3 : str
        Simple 3 % annual increase.
    SIMPLE_5 : str
        Simple 5 % annual increase.
    COMPOUND_3 : str
        Compound 3 % annual increase.
    COMPOUND_5 : str
        Compound 5 % annual increase.
    CPI_LINKED : str
        Benefit increases linked to Consumer Price Index.
    FUTURE_PURCHASE : str
        Periodic option to purchase additional coverage at then-current rates.
    """

    NONE = "None"
    SIMPLE_3 = "Simple 3%"
    SIMPLE_5 = "Simple 5%"
    COMPOUND_3 = "Compound 3%"
    COMPOUND_5 = "Compound 5%"
    CPI_LINKED = "CPI-Linked"
    FUTURE_PURCHASE = "Future Purchase Option"


class Care_Setting(Enum):
    """Settings where LTC benefits are payable.

    Attributes
    ----------
    NURSING_HOME : str
        Skilled nursing facility.
    ASSISTED_LIVING : str
        Assisted living facility.
    HOME_HEALTH_CARE : str
        Care provided in the insured's home.
    ADULT_DAY_CARE : str
        Adult day-care centre.
    HOSPICE : str
        Hospice care.
    ALL : str
        Comprehensive — all care settings covered.
    """

    NURSING_HOME = "Nursing Home"
    ASSISTED_LIVING = "Assisted Living"
    HOME_HEALTH_CARE = "Home Health Care"
    ADULT_DAY_CARE = "Adult Day Care"
    HOSPICE = "Hospice"
    ALL = "All Care Settings"


class Premium_Frequency_LTC(Enum):
    """Payment frequency for LTC insurance premiums.

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


# ======================================================================
# Base class
# ======================================================================


class Insurance_LTC_Base(ABC):
    """Abstract base class for Long Term Care insurance products.

    This class defines the interface and common functionality shared by
    all LTC insurance types.  Derived classes must implement the
    :meth:`calc_maximum_lifetime_benefit`, :meth:`calc_annual_premium`,
    and :meth:`calc_daily_benefit_at_year` methods.

    Attributes
    ----------
    m_insured_age : int
        Current age of the insured.
    m_daily_benefit_amount : float
        Maximum daily benefit amount (dollars per day).
    m_insurance_type : Insurance_LTC_Type
        The type of LTC insurance.
    m_benefit_period : Benefit_Period
        Maximum duration of benefit payments.
    m_elimination_period : Elimination_Period
        Waiting period before benefits begin.
    m_inflation_protection : Inflation_Protection
        Type of inflation protection rider.
    m_care_setting : Care_Setting
        Covered care setting(s).
    m_premium_frequency : Premium_Frequency_LTC
        How frequently premiums are paid.
    m_is_smoker : bool
        Whether the insured uses tobacco products.
    m_is_married_discount : bool
        Whether a married / partner discount applies.
    """

    def __init__(
        self,
        insured_age: int,
        daily_benefit_amount: float,
        insurance_type: Insurance_LTC_Type,
        benefit_period: Benefit_Period = Benefit_Period.YEARS_3,
        elimination_period: Elimination_Period = Elimination_Period.DAYS_90,
        inflation_protection: Inflation_Protection = Inflation_Protection.NONE,
        care_setting: Care_Setting = Care_Setting.ALL,
        premium_frequency: Premium_Frequency_LTC = Premium_Frequency_LTC.ANNUAL,
        is_smoker: bool = False,
        is_married_discount: bool = False,
    ) -> None:
        """Initialize an LTC insurance base object.

        Parameters
        ----------
        insured_age : int
            Current age of the insured (must be 18 – 120).
        daily_benefit_amount : float
            Maximum daily benefit in dollars (must be positive).
        insurance_type : Insurance_LTC_Type
            The specific LTC insurance type.
        benefit_period : Benefit_Period, optional
            Maximum benefit duration (default ``YEARS_3``).
        elimination_period : Elimination_Period, optional
            Waiting period in days (default ``DAYS_90``).
        inflation_protection : Inflation_Protection, optional
            Inflation rider type (default ``NONE``).
        care_setting : Care_Setting, optional
            Covered care settings (default ``ALL``).
        premium_frequency : Premium_Frequency_LTC, optional
            Payment frequency (default ``ANNUAL``).
        is_smoker : bool, optional
            Tobacco usage flag (default ``False``).
        is_married_discount : bool, optional
            Whether a marital / partner discount applies (default ``False``).

        Raises
        ------
        Exception_Validation_Input
            If any input fails validation.
        """
        # --- Input validation ---
        if not isinstance(insured_age, int) or insured_age < 18 or insured_age > 120:
            raise Exception_Validation_Input(
                "insured_age must be an integer in [18, 120]",
                field_name="insured_age",
                expected_type=int,
                actual_value=insured_age,
            )

        if not isinstance(daily_benefit_amount, (int, float)) or daily_benefit_amount <= 0:
            raise Exception_Validation_Input(
                "daily_benefit_amount must be a positive number",
                field_name="daily_benefit_amount",
                expected_type=float,
                actual_value=daily_benefit_amount,
            )

        if not isinstance(insurance_type, Insurance_LTC_Type):
            raise Exception_Validation_Input(
                "insurance_type must be a valid Insurance_LTC_Type enum",
                field_name="insurance_type",
                expected_type=Insurance_LTC_Type,
                actual_value=insurance_type,
            )

        if not isinstance(benefit_period, Benefit_Period):
            raise Exception_Validation_Input(
                "benefit_period must be a valid Benefit_Period enum",
                field_name="benefit_period",
                expected_type=Benefit_Period,
                actual_value=benefit_period,
            )

        if not isinstance(elimination_period, Elimination_Period):
            raise Exception_Validation_Input(
                "elimination_period must be a valid Elimination_Period enum",
                field_name="elimination_period",
                expected_type=Elimination_Period,
                actual_value=elimination_period,
            )

        if not isinstance(inflation_protection, Inflation_Protection):
            raise Exception_Validation_Input(
                "inflation_protection must be a valid Inflation_Protection enum",
                field_name="inflation_protection",
                expected_type=Inflation_Protection,
                actual_value=inflation_protection,
            )

        if not isinstance(care_setting, Care_Setting):
            raise Exception_Validation_Input(
                "care_setting must be a valid Care_Setting enum",
                field_name="care_setting",
                expected_type=Care_Setting,
                actual_value=care_setting,
            )

        if not isinstance(premium_frequency, Premium_Frequency_LTC):
            raise Exception_Validation_Input(
                "premium_frequency must be a valid Premium_Frequency_LTC enum",
                field_name="premium_frequency",
                expected_type=Premium_Frequency_LTC,
                actual_value=premium_frequency,
            )

        # --- Set member variables ---
        self.m_insured_age: int = insured_age
        self.m_daily_benefit_amount: float = float(daily_benefit_amount)
        self.m_insurance_type: Insurance_LTC_Type = insurance_type
        self.m_benefit_period: Benefit_Period = benefit_period
        self.m_elimination_period: Elimination_Period = elimination_period
        self.m_inflation_protection: Inflation_Protection = inflation_protection
        self.m_care_setting: Care_Setting = care_setting
        self.m_premium_frequency: Premium_Frequency_LTC = premium_frequency
        self.m_is_smoker: bool = bool(is_smoker)
        self.m_is_married_discount: bool = bool(is_married_discount)

        logger.info(
            f"Created {insurance_type.value} for insured age {insured_age}, "
            f"daily benefit ${daily_benefit_amount:,.2f}, "
            f"benefit period: {benefit_period.value} years, "
            f"elimination: {elimination_period.value} days",
        )

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def insured_age(self) -> int:
        """Int : Current age of the insured."""
        return self.m_insured_age

    @property
    def daily_benefit_amount(self) -> float:
        """Float : Maximum daily benefit amount."""
        return self.m_daily_benefit_amount

    @property
    def insurance_type(self) -> Insurance_LTC_Type:
        """Insurance_LTC_Type : The type of LTC insurance."""
        return self.m_insurance_type

    @property
    def benefit_period(self) -> Benefit_Period:
        """Benefit_Period : Maximum benefit payment duration."""
        return self.m_benefit_period

    @property
    def elimination_period(self) -> Elimination_Period:
        """Elimination_Period : Waiting period before benefits begin."""
        return self.m_elimination_period

    @property
    def inflation_protection(self) -> Inflation_Protection:
        """Inflation_Protection : Type of inflation protection rider."""
        return self.m_inflation_protection

    @property
    def care_setting(self) -> Care_Setting:
        """Care_Setting : Covered care setting(s)."""
        return self.m_care_setting

    @property
    def premium_frequency(self) -> Premium_Frequency_LTC:
        """Premium_Frequency_LTC : Premium payment frequency."""
        return self.m_premium_frequency

    @property
    def is_smoker(self) -> bool:
        """Bool : Whether the insured uses tobacco."""
        return self.m_is_smoker

    @property
    def is_married_discount(self) -> bool:
        """Bool : Whether a marital / partner discount applies."""
        return self.m_is_married_discount

    # ------------------------------------------------------------------
    # Calculation helpers (common)
    # ------------------------------------------------------------------

    def calc_monthly_benefit_amount(self) -> float:
        r"""Calculate the monthly benefit amount.

        $$
        B_{\text{monthly}} = B_{\text{daily}} \times 30
        $$

        Returns
        -------
        float
            The monthly benefit amount.
        """
        return self.m_daily_benefit_amount * 30.0

    def calc_annual_benefit_amount(self) -> float:
        r"""Calculate the annual benefit amount.

        $$
        B_{\text{annual}} = B_{\text{daily}} \times 365
        $$

        Returns
        -------
        float
            The annual benefit amount.
        """
        return self.m_daily_benefit_amount * 365.0

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
        modal_factors: dict[Premium_Frequency_LTC, float] = {
            Premium_Frequency_LTC.ANNUAL: 1.0,
            Premium_Frequency_LTC.SEMI_ANNUAL: 0.515,
            Premium_Frequency_LTC.QUARTERLY: 0.265,
            Premium_Frequency_LTC.MONTHLY: 0.0875,
            Premium_Frequency_LTC.SINGLE: 1.0,
        }
        return modal_factors.get(self.m_premium_frequency, 1.0)

    def calc_inflation_growth_rate(self) -> float:
        """Return the annual inflation growth rate for the selected rider.

        Returns
        -------
        float
            The annual benefit growth rate as a decimal.
        """
        _rates: dict[Inflation_Protection, float] = {
            Inflation_Protection.NONE: 0.0,
            Inflation_Protection.SIMPLE_3: 0.03,
            Inflation_Protection.SIMPLE_5: 0.05,
            Inflation_Protection.COMPOUND_3: 0.03,
            Inflation_Protection.COMPOUND_5: 0.05,
            Inflation_Protection.CPI_LINKED: 0.03,  # estimated average
            Inflation_Protection.FUTURE_PURCHASE: 0.0,  # buyer determines
        }
        return _rates.get(self.m_inflation_protection, 0.0)

    def calc_is_compound_inflation(self) -> bool:
        """Determine whether the inflation rider compounds.

        Returns
        -------
        bool
            ``True`` if compound inflation; ``False`` if simple or none.
        """
        return self.m_inflation_protection in {
            Inflation_Protection.COMPOUND_3,
            Inflation_Protection.COMPOUND_5,
            Inflation_Protection.CPI_LINKED,
        }

    # ------------------------------------------------------------------
    # Abstract methods
    # ------------------------------------------------------------------

    @abstractmethod
    def calc_maximum_lifetime_benefit(self) -> float:
        """Calculate the maximum lifetime benefit pool.

        Must be implemented by each derived class to account for
        product-specific features.

        Returns
        -------
        float
            The total lifetime benefit pool in dollars.
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

    @abstractmethod
    def calc_daily_benefit_at_year(self, year: int) -> float:
        """Calculate the daily benefit amount at a given policy year.

        Accounts for inflation protection if applicable.

        Parameters
        ----------
        year : int
            The policy year number (1-indexed).

        Returns
        -------
        float
            The daily benefit amount at the specified year.
        """

    # ------------------------------------------------------------------
    # String representation
    # ------------------------------------------------------------------

    def get_insurance_as_string(self) -> str:
        """Return a human-readable string representation.

        Returns
        -------
        str
            String representation of the LTC insurance policy.
        """
        smoker_text = "Smoker" if self.m_is_smoker else "Non-Smoker"
        return (
            f"{self.m_insurance_type.value} "
            f"(Age: {self.m_insured_age}, "
            f"Daily Benefit: ${self.m_daily_benefit_amount:,.2f}, "
            f"Benefit Period: {self.m_benefit_period.value} yrs, "
            f"Elimination: {self.m_elimination_period.value} days, "
            f"{smoker_text})"
        )
