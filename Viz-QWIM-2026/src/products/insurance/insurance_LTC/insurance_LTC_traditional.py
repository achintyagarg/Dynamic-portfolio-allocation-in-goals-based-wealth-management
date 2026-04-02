"""
Class for Long Term Care (LTC) Traditional insurance.

===============================

Traditional LTC insurance is a stand-alone product that provides
benefits for long-term care expenses.  Key characteristics:

- **Pay-as-you-go premiums**: premiums are typically level but subject
  to class-wide rate increases approved by state regulators.
- **Use-it-or-lose-it**: there is generally no return of premium or
  cash value if the policy is never used; some non-forfeiture options
  exist.
- **Flexible benefit design**: daily/monthly benefit amount, benefit
  period, elimination period, inflation protection, and care settings
  are all configurable.

Premium Rating Factors
~~~~~~~~~~~~~~~~~~~~~~
Traditional LTC premiums are driven by:

1. **Issue age** — younger applicants receive lower rates.
2. **Health class** — preferred / standard / substandard.
3. **Benefit period** — longer periods cost more.
4. **Elimination period** — shorter waits cost more.
5. **Inflation protection** — compound riders are expensive.
6. **Marital / partner discount** — usually 15 – 30 %.
7. **Gender** — females historically have higher claims.
8. **Tobacco usage** — smokers pay more.

Non-Forfeiture Options
~~~~~~~~~~~~~~~~~~~~~~
If the insured lapses coverage after a minimum number of years they
may retain limited paid-up benefits under a *shortened benefit period*
or *contingent non-forfeiture* provision.

Author
------
QWIM Team

Version
-------
0.7.0 (2026-02-13)
"""

from __future__ import annotations

from aenum import Enum

from src.utils.custom_exceptions_errors_loggers.exception_custom import (
    Exception_Calculation,
    Exception_Validation_Input,
)
from src.utils.custom_exceptions_errors_loggers.logger_custom import get_logger

from .insurance_LTC_base import (
    Benefit_Period,
    Care_Setting,
    Elimination_Period,
    Inflation_Protection,
    Insurance_LTC_Base,
    Insurance_LTC_Type,
    Premium_Frequency_LTC,
)


logger = get_logger(__name__)


# ======================================================================
# Enumerations specific to traditional LTC
# ======================================================================


class Health_Class_LTC(Enum):
    """Underwriting health class for traditional LTC.

    Attributes
    ----------
    PREFERRED : str
        Best health class — lowest premiums.
    STANDARD : str
        Standard health class — normal premiums.
    SUBSTANDARD : str
        Below-average health — rated-up premiums.
    """

    PREFERRED = "Preferred"
    STANDARD = "Standard"
    SUBSTANDARD = "Substandard"


class Non_Forfeiture_Option(Enum):
    """Non-forfeiture options available upon policy lapse.

    Attributes
    ----------
    NONE : str
        No non-forfeiture option (policy lapses with no residual value).
    SHORTENED_BENEFIT : str
        Paid-up coverage with a shortened benefit period equal to the
        premiums already paid.
    RETURN_OF_PREMIUM : str
        Return of premium (in full or partial) upon death or lapse.
    CONTINGENT : str
        Contingent non-forfeiture — triggered only when a substantial
        premium rate increase occurs.
    """

    NONE = "None"
    SHORTENED_BENEFIT = "Shortened Benefit Period"
    RETURN_OF_PREMIUM = "Return of Premium"
    CONTINGENT = "Contingent Non-Forfeiture"


class Gender_LTC(Enum):
    """Gender classification for LTC premium rating.

    Attributes
    ----------
    MALE : str
        Male.
    FEMALE : str
        Female.
    UNISEX : str
        Unisex / gender-neutral.
    """

    MALE = "Male"
    FEMALE = "Female"
    UNISEX = "Unisex"


# ======================================================================
# Traditional LTC insurance class
# ======================================================================


class Insurance_LTC_Traditional(Insurance_LTC_Base):
    r"""Traditional stand-alone Long Term Care (LTC) Insurance.

    Traditional LTC insurance provides a benefit pool that pays a
    maximum daily (or monthly) benefit for qualified long-term-care
    expenses.  The pool is defined by:

    $$
    \text{Pool}_{\text{initial}} = B_{\text{daily}} \times 365
        \times P_{\text{years}}
    $$

    where $B_{\text{daily}}$ is the daily benefit amount and
    $P_{\text{years}}$ is the benefit period in years.

    Premium calculation uses a base rate table indexed by issue age,
    gender, and health class, with multipliers for benefit design
    selections:

    $$
    \text{Premium}_{\text{annual}}
        = R_{\text{base}}
        \times M_{\text{benefit\_period}}
        \times M_{\text{elimination}}
        \times M_{\text{inflation}}
        \times M_{\text{care}}
        \times (1 + S_{\text{smoker}})
        \times (1 - D_{\text{married}})
    $$

    Parameters
    ----------
    insured_age : int
        Issue / current age of the insured (18 – 85).
    daily_benefit_amount : float
        Maximum daily benefit in dollars.
    gender : Gender_LTC
        Gender for premium rating.
    health_class : Health_Class_LTC, optional
        Underwriting health class (default ``STANDARD``).
    benefit_period : Benefit_Period, optional
        Benefit payment duration (default ``YEARS_3``).
    elimination_period : Elimination_Period, optional
        Waiting period in days (default ``DAYS_90``).
    inflation_protection : Inflation_Protection, optional
        Inflation rider (default ``NONE``).
    care_setting : Care_Setting, optional
        Covered care settings (default ``ALL``).
    premium_frequency : Premium_Frequency_LTC, optional
        Payment frequency (default ``ANNUAL``).
    is_smoker : bool, optional
        Tobacco usage flag (default ``False``).
    is_married_discount : bool, optional
        Marital / partner discount flag (default ``False``).
    non_forfeiture : Non_Forfeiture_Option, optional
        Non-forfeiture rider (default ``NONE``).
    shared_care : bool, optional
        Shared-care rider with partner (default ``False``).
    waiver_of_premium : bool, optional
        Premium waived while on claim (default ``True``).
    base_rate_per_unit : float, optional
        Base annual rate per $10 of daily benefit.  If ``None`` a
        default schedule keyed on age and gender is used.
    cumulative_rate_increase_pct : float, optional
        Cumulative class-wide premium rate increase already applied
        (as a decimal, e.g., 0.20 for 20 %).  Default 0.0.

    Attributes
    ----------
    m_gender : Gender_LTC
    m_health_class : Health_Class_LTC
    m_non_forfeiture : Non_Forfeiture_Option
    m_shared_care : bool
    m_waiver_of_premium : bool
    m_base_rate_per_unit : float
    m_cumulative_rate_increase_pct : float
    """

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def __init__(
        self,
        insured_age: int,
        daily_benefit_amount: float,
        gender: Gender_LTC,
        health_class: Health_Class_LTC = Health_Class_LTC.STANDARD,
        benefit_period: Benefit_Period = Benefit_Period.YEARS_3,
        elimination_period: Elimination_Period = Elimination_Period.DAYS_90,
        inflation_protection: Inflation_Protection = Inflation_Protection.NONE,
        care_setting: Care_Setting = Care_Setting.ALL,
        premium_frequency: Premium_Frequency_LTC = Premium_Frequency_LTC.ANNUAL,
        is_smoker: bool = False,
        is_married_discount: bool = False,
        non_forfeiture: Non_Forfeiture_Option = Non_Forfeiture_Option.NONE,
        shared_care: bool = False,
        waiver_of_premium: bool = True,
        base_rate_per_unit: float | None = None,
        cumulative_rate_increase_pct: float = 0.0,
    ) -> None:
        """Initialize a Traditional LTC Insurance instance."""
        # ---- Validate traditional-specific inputs ----
        if not isinstance(gender, Gender_LTC):
            raise Exception_Validation_Input(
                "gender must be a valid Gender_LTC enum",
                field_name="gender",
                expected_type=Gender_LTC,
                actual_value=gender,
            )

        if not isinstance(health_class, Health_Class_LTC):
            raise Exception_Validation_Input(
                "health_class must be a valid Health_Class_LTC enum",
                field_name="health_class",
                expected_type=Health_Class_LTC,
                actual_value=health_class,
            )

        if not isinstance(non_forfeiture, Non_Forfeiture_Option):
            raise Exception_Validation_Input(
                "non_forfeiture must be a valid Non_Forfeiture_Option enum",
                field_name="non_forfeiture",
                expected_type=Non_Forfeiture_Option,
                actual_value=non_forfeiture,
            )

        if insured_age > 85:
            raise Exception_Validation_Input(
                "Traditional LTC is typically not issued above age 85",
                field_name="insured_age",
                expected_type=int,
                actual_value=insured_age,
            )

        if cumulative_rate_increase_pct < 0.0 or cumulative_rate_increase_pct > 5.0:
            raise Exception_Validation_Input(
                "cumulative_rate_increase_pct must be in [0.0, 5.0]",
                field_name="cumulative_rate_increase_pct",
                expected_type=float,
                actual_value=cumulative_rate_increase_pct,
            )

        # ---- Base class initialisation ----
        super().__init__(
            insured_age=insured_age,
            daily_benefit_amount=daily_benefit_amount,
            insurance_type=Insurance_LTC_Type.TRADITIONAL,
            benefit_period=benefit_period,
            elimination_period=elimination_period,
            inflation_protection=inflation_protection,
            care_setting=care_setting,
            premium_frequency=premium_frequency,
            is_smoker=is_smoker,
            is_married_discount=is_married_discount,
        )

        # ---- Traditional-specific member variables ----
        self.m_gender: Gender_LTC = gender
        self.m_health_class: Health_Class_LTC = health_class
        self.m_non_forfeiture: Non_Forfeiture_Option = non_forfeiture
        self.m_shared_care: bool = bool(shared_care)
        self.m_waiver_of_premium: bool = bool(waiver_of_premium)
        self.m_cumulative_rate_increase_pct: float = float(
            cumulative_rate_increase_pct,
        )

        # ---- Default base rate schedule ----
        if base_rate_per_unit is not None:
            if not isinstance(base_rate_per_unit, (int, float)) or base_rate_per_unit <= 0:
                raise Exception_Validation_Input(
                    "base_rate_per_unit must be a positive number",
                    field_name="base_rate_per_unit",
                    expected_type=float,
                    actual_value=base_rate_per_unit,
                )
            self.m_base_rate_per_unit: float = float(base_rate_per_unit)
        else:
            self.m_base_rate_per_unit = self._lookup_base_rate()

        logger.info(
            f"Traditional LTC: gender={gender.value}, "
            f"health_class={health_class.value}, "
            f"non_forfeiture={non_forfeiture.value}, "
            f"shared_care={shared_care}",
        )

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def gender(self) -> Gender_LTC:
        """Gender_LTC : Gender for premium rating."""
        return self.m_gender

    @property
    def health_class(self) -> Health_Class_LTC:
        """Health_Class_LTC : Underwriting health class."""
        return self.m_health_class

    @property
    def non_forfeiture(self) -> Non_Forfeiture_Option:
        """Non_Forfeiture_Option : Non-forfeiture rider selection."""
        return self.m_non_forfeiture

    @property
    def shared_care(self) -> bool:
        """Bool : Whether a shared-care rider is active."""
        return self.m_shared_care

    @property
    def waiver_of_premium(self) -> bool:
        """Bool : Whether premium is waived while on claim."""
        return self.m_waiver_of_premium

    @property
    def base_rate_per_unit(self) -> float:
        """Float : Base annual rate per $10 of daily benefit."""
        return self.m_base_rate_per_unit

    @property
    def cumulative_rate_increase_pct(self) -> float:
        """Float : Cumulative class-wide rate increase (decimal)."""
        return self.m_cumulative_rate_increase_pct

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _lookup_base_rate(self) -> float:
        """Look up the default base-rate per $10 of daily benefit.

        Base rates increase steeply with age.  Female rates are
        typically 40 – 60 % higher than male rates because of longer
        average claim duration.

        Returns
        -------
        float
            Annual base rate per $10 of daily benefit.
        """
        # Simplified schedule (representative of a mid-market carrier).
        # In production this would come from an external rating table.
        _age_band_rates_male: dict[tuple[int, int], float] = {
            (18, 39): 8.00,
            (40, 44): 12.00,
            (45, 49): 18.00,
            (50, 54): 26.00,
            (55, 59): 40.00,
            (60, 64): 62.00,
            (65, 69): 100.00,
            (70, 74): 165.00,
            (75, 79): 260.00,
            (80, 85): 400.00,
        }

        # Female rate multiplier
        _female_multiplier = 1.50

        # Look up age band for male rate
        male_rate = 62.0  # default fallback (age 60-64)
        for (low, high), rate in _age_band_rates_male.items():
            if low <= self.m_insured_age <= high:
                male_rate = rate
                break

        base = male_rate
        if self.m_gender == Gender_LTC.FEMALE:
            base = male_rate * _female_multiplier
        elif self.m_gender == Gender_LTC.UNISEX:
            base = male_rate * (1.0 + _female_multiplier) / 2.0

        return base

    # ------------------------------------------------------------------
    # Calculation methods (abstract overrides)
    # ------------------------------------------------------------------

    def calc_maximum_lifetime_benefit(self) -> float:
        r"""Calculate the maximum (initial) lifetime benefit pool.

        $$
        \text{Pool}_{\text{initial}} = B_{\text{daily}} \times 365
            \times P_{\text{years}}
        $$

        For a **lifetime** benefit period, $P_{\text{years}}$ is capped
        at the Benefit_Period.LIFETIME sentinel (100 years).

        Returns
        -------
        float
            The initial lifetime benefit pool in dollars.
        """
        return self.m_daily_benefit_amount * 365.0 * self.m_benefit_period.value

    def calc_daily_benefit_at_year(self, year: int) -> float:
        r"""Calculate the daily benefit amount at a given policy year.

        With **simple** inflation protection:

        $$
        B(t) = B_0 \times \bigl(1 + r \times t\bigr)
        $$

        With **compound** inflation protection:

        $$
        B(t) = B_0 \times (1 + r)^{t}
        $$

        Parameters
        ----------
        year : int
            Policy year (1-indexed; year 1 = first year).

        Returns
        -------
        float
            Daily benefit amount at the specified policy year.

        Raises
        ------
        Exception_Validation_Input
            If *year* is not a positive integer.
        """
        if not isinstance(year, int) or year < 1:
            raise Exception_Validation_Input(
                "year must be a positive integer",
                field_name="year",
                expected_type=int,
                actual_value=year,
            )

        growth_rate = self.calc_inflation_growth_rate()
        t = year - 1  # years elapsed since policy inception

        if growth_rate == 0.0:
            return self.m_daily_benefit_amount

        if self.calc_is_compound_inflation():
            return self.m_daily_benefit_amount * (1.0 + growth_rate) ** t

        # Simple inflation
        return self.m_daily_benefit_amount * (1.0 + growth_rate * t)

    def calc_annual_premium(self) -> float:
        r"""Calculate the annual premium for the traditional LTC policy.

        The premium is built from a base rate per unit ($10 of daily
        benefit) multiplied by a series of design-selection factors:

        $$
        P_{\text{annual}}
            = \frac{B_{\text{daily}}}{10}
            \times R_{\text{base}}
            \times M_{\text{BP}}
            \times M_{\text{EP}}
            \times M_{\text{Infl}}
            \times M_{\text{Care}}
            \times M_{\text{HC}}
            \times (1 + S_{\text{smoker}})
            \times (1 - D_{\text{married}})
            \times (1 + D_{\text{NF}})
            \times (1 + D_{\text{shared}})
            \times (1 + D_{\text{waiver}})
            \times (1 + I_{\text{cumulative}})
        $$

        Returns
        -------
        float
            The annual premium before modal adjustment.
        """
        units = self.m_daily_benefit_amount / 10.0

        # --- Benefit-period multiplier ---
        _bp_factors: dict[Benefit_Period, float] = {
            Benefit_Period.YEARS_2: 0.72,
            Benefit_Period.YEARS_3: 1.00,
            Benefit_Period.YEARS_4: 1.23,
            Benefit_Period.YEARS_5: 1.42,
            Benefit_Period.YEARS_6: 1.58,
            Benefit_Period.LIFETIME: 2.30,
        }
        m_bp = _bp_factors.get(self.m_benefit_period, 1.0)

        # --- Elimination-period multiplier ---
        _ep_factors: dict[Elimination_Period, float] = {
            Elimination_Period.DAYS_0: 1.45,
            Elimination_Period.DAYS_30: 1.15,
            Elimination_Period.DAYS_60: 1.05,
            Elimination_Period.DAYS_90: 1.00,
            Elimination_Period.DAYS_180: 0.80,
            Elimination_Period.DAYS_365: 0.60,
        }
        m_ep = _ep_factors.get(self.m_elimination_period, 1.0)

        # --- Inflation-protection multiplier ---
        _infl_factors: dict[Inflation_Protection, float] = {
            Inflation_Protection.NONE: 1.00,
            Inflation_Protection.SIMPLE_3: 1.25,
            Inflation_Protection.SIMPLE_5: 1.50,
            Inflation_Protection.COMPOUND_3: 1.55,
            Inflation_Protection.COMPOUND_5: 2.00,
            Inflation_Protection.CPI_LINKED: 1.65,
            Inflation_Protection.FUTURE_PURCHASE: 1.10,
        }
        m_infl = _infl_factors.get(self.m_inflation_protection, 1.0)

        # --- Care-setting multiplier ---
        _care_factors: dict[Care_Setting, float] = {
            Care_Setting.NURSING_HOME: 0.70,
            Care_Setting.ASSISTED_LIVING: 0.80,
            Care_Setting.HOME_HEALTH_CARE: 0.85,
            Care_Setting.ADULT_DAY_CARE: 0.65,
            Care_Setting.HOSPICE: 0.60,
            Care_Setting.ALL: 1.00,
        }
        m_care = _care_factors.get(self.m_care_setting, 1.0)

        # --- Health class multiplier ---
        _hc_factors: dict[Health_Class_LTC, float] = {
            Health_Class_LTC.PREFERRED: 0.85,
            Health_Class_LTC.STANDARD: 1.00,
            Health_Class_LTC.SUBSTANDARD: 1.50,
        }
        m_hc = _hc_factors.get(self.m_health_class, 1.0)

        # --- Smoker surcharge ---
        smoker_surcharge = 0.25 if self.m_is_smoker else 0.0

        # --- Married / partner discount ---
        married_discount = 0.20 if self.m_is_married_discount else 0.0

        # --- Non-forfeiture rider surcharge ---
        _nf_surcharges: dict[Non_Forfeiture_Option, float] = {
            Non_Forfeiture_Option.NONE: 0.0,
            Non_Forfeiture_Option.SHORTENED_BENEFIT: 0.10,
            Non_Forfeiture_Option.RETURN_OF_PREMIUM: 0.35,
            Non_Forfeiture_Option.CONTINGENT: 0.00,
        }
        nf_surcharge = _nf_surcharges.get(self.m_non_forfeiture, 0.0)

        # --- Shared care surcharge ---
        shared_surcharge = 0.12 if self.m_shared_care else 0.0

        # --- Waiver-of-premium surcharge ---
        waiver_surcharge = 0.03 if self.m_waiver_of_premium else 0.0

        # --- Assemble premium ---
        premium = (
            units
            * self.m_base_rate_per_unit
            * m_bp
            * m_ep
            * m_infl
            * m_care
            * m_hc
            * (1.0 + smoker_surcharge)
            * (1.0 - married_discount)
            * (1.0 + nf_surcharge)
            * (1.0 + shared_surcharge)
            * (1.0 + waiver_surcharge)
            * (1.0 + self.m_cumulative_rate_increase_pct)
        )

        return round(premium, 2)

    # ------------------------------------------------------------------
    # Additional calculation methods
    # ------------------------------------------------------------------

    def calc_modal_premium(self) -> float:
        r"""Calculate the premium per payment frequency.

        $$
        P_{\text{modal}} = P_{\text{annual}} \times f_{\text{modal}}
        $$

        Returns
        -------
        float
            The premium amount per payment period.
        """
        annual = self.calc_annual_premium()
        factor = self.calc_modal_premium_factor()
        freq = self.m_premium_frequency.value

        if freq == 0:
            # Single-pay: not typical for traditional LTC
            return annual

        return round(annual * factor, 2)

    def calc_remaining_benefit_pool(
        self,
        benefits_paid_to_date: float,
        policy_year: int = 1,
    ) -> float:
        r"""Calculate the remaining benefit pool at a given point.

        $$
        \text{Pool}_{\text{remaining}} =
            \text{Pool}_{\text{adj}}(t) - \text{Benefits paid}
        $$

        where the adjusted pool accounts for inflation growth of the
        initial pool on a yearly basis.

        Parameters
        ----------
        benefits_paid_to_date : float
            Total benefits already paid out under the policy.
        policy_year : int, optional
            Current policy year (default 1).

        Returns
        -------
        float
            Remaining benefit pool in dollars (floored at 0).

        Raises
        ------
        Exception_Validation_Input
            If inputs are invalid.
        """
        if not isinstance(benefits_paid_to_date, (int, float)) or benefits_paid_to_date < 0:
            raise Exception_Validation_Input(
                "benefits_paid_to_date must be non-negative",
                field_name="benefits_paid_to_date",
                expected_type=float,
                actual_value=benefits_paid_to_date,
            )

        # Adjusted pool: daily benefit at that year x 365 x benefit period
        daily_at_year = self.calc_daily_benefit_at_year(policy_year)
        adjusted_pool = daily_at_year * 365.0 * self.m_benefit_period.value

        remaining = adjusted_pool - float(benefits_paid_to_date)
        return max(remaining, 0.0)

    def calc_cost_of_waiting(
        self,
        future_age: int,
    ) -> float:
        r"""Estimate the cost of waiting to purchase until a later age.

        This compares the cumulative premiums from purchasing today
        versus purchasing at a future age.  A rough first-order
        estimate uses the ratio of base rates:

        $$
        \text{Ratio} = \frac{R_{\text{base}}(\text{future age})}
            {R_{\text{base}}(\text{current age})}
        $$

        Parameters
        ----------
        future_age : int
            The age at which the policy would be purchased instead.

        Returns
        -------
        float
            Ratio of future premium to current premium.

        Raises
        ------
        Exception_Validation_Input
            If *future_age* is not greater than current age.
        """
        if not isinstance(future_age, int) or future_age <= self.m_insured_age:
            raise Exception_Validation_Input(
                "future_age must be an integer > current insured age",
                field_name="future_age",
                expected_type=int,
                actual_value=future_age,
            )

        # Create a temporary policy at the future age to compare rates
        current_premium = self.calc_annual_premium()

        if current_premium == 0:
            raise Exception_Calculation(
                "Current premium is zero; ratio is undefined",
            )

        # Save and temporarily adjust age + base rate
        original_age = self.m_insured_age
        original_rate = self.m_base_rate_per_unit

        self.m_insured_age = future_age
        self.m_base_rate_per_unit = self._lookup_base_rate()
        future_premium = self.calc_annual_premium()

        # Restore originals
        self.m_insured_age = original_age
        self.m_base_rate_per_unit = original_rate

        return round(future_premium / current_premium, 4)

    def calc_non_forfeiture_benefit(
        self,
        years_premiums_paid: int,
    ) -> float:
        r"""Calculate non-forfeiture benefit value.

        Under the **shortened benefit period** option the insured
        retains paid-up coverage equal to the lesser of:

        - Total premiums paid to date, or
        - The original maximum lifetime benefit.

        $$
        \text{NF Benefit} = \min\bigl(
            P_{\text{annual}} \times T_{\text{years\_paid}},\;
            \text{Pool}_{\text{initial}}
        \bigr)
        $$

        Parameters
        ----------
        years_premiums_paid : int
            Number of full years of premiums paid.

        Returns
        -------
        float
            The non-forfeiture benefit value (0 if ``NONE``).

        Raises
        ------
        Exception_Validation_Input
            If *years_premiums_paid* is negative.
        """
        if not isinstance(years_premiums_paid, int) or years_premiums_paid < 0:
            raise Exception_Validation_Input(
                "years_premiums_paid must be a non-negative integer",
                field_name="years_premiums_paid",
                expected_type=int,
                actual_value=years_premiums_paid,
            )

        if self.m_non_forfeiture == Non_Forfeiture_Option.NONE:
            return 0.0

        annual_premium = self.calc_annual_premium()
        total_premiums = annual_premium * years_premiums_paid
        pool = self.calc_maximum_lifetime_benefit()

        if self.m_non_forfeiture == Non_Forfeiture_Option.SHORTENED_BENEFIT:
            return min(total_premiums, pool)

        if self.m_non_forfeiture == Non_Forfeiture_Option.RETURN_OF_PREMIUM:
            return total_premiums

        # Contingent: same as shortened benefit
        return min(total_premiums, pool)

    def calc_elimination_period_out_of_pocket(
        self,
        daily_care_cost: float,
    ) -> float:
        r"""Estimate out-of-pocket costs during the elimination period.

        $$
        \text{OOP} = C_{\text{daily}} \times E_{\text{days}}
        $$

        Parameters
        ----------
        daily_care_cost : float
            Estimated daily cost of care (dollars per day).

        Returns
        -------
        float
            Total out-of-pocket expense during elimination period.

        Raises
        ------
        Exception_Validation_Input
            If *daily_care_cost* is not positive.
        """
        if not isinstance(daily_care_cost, (int, float)) or daily_care_cost <= 0:
            raise Exception_Validation_Input(
                "daily_care_cost must be a positive number",
                field_name="daily_care_cost",
                expected_type=float,
                actual_value=daily_care_cost,
            )

        return float(daily_care_cost) * self.m_elimination_period.value

    def calc_benefit_duration_at_cost(
        self,
        daily_care_cost: float,
        policy_year: int = 1,
    ) -> float:
        r"""Estimate how long benefits last given an actual daily cost.

        If the actual daily cost exceeds the policy's daily benefit,
        the benefit pool is consumed faster.

        $$
        D_{\text{years}} = \frac{\text{Pool}}{C_{\text{daily}}
            \times 365}
        $$

        Parameters
        ----------
        daily_care_cost : float
            Actual daily cost of care in dollars.
        policy_year : int, optional
            Policy year for inflation-adjusted pool (default 1).

        Returns
        -------
        float
            Estimated benefit duration in years.

        Raises
        ------
        Exception_Validation_Input
            If *daily_care_cost* is not positive.
        """
        if not isinstance(daily_care_cost, (int, float)) or daily_care_cost <= 0:
            raise Exception_Validation_Input(
                "daily_care_cost must be a positive number",
                field_name="daily_care_cost",
                expected_type=float,
                actual_value=daily_care_cost,
            )

        pool_remaining = self.calc_remaining_benefit_pool(0.0, policy_year)
        annual_cost = float(daily_care_cost) * 365.0

        if annual_cost == 0:
            return float("inf")

        return round(pool_remaining / annual_cost, 2)

    def calc_total_premiums_to_age(self, target_age: int) -> float:
        r"""Calculate total premiums paid from issue age to a target age.

        $$
        \text{Total} = P_{\text{annual}} \times (A_{\text{target}}
            - A_{\text{issue}})
        $$

        Parameters
        ----------
        target_age : int
            Age up to which premiums are summed.

        Returns
        -------
        float
            Total cumulative premiums paid.

        Raises
        ------
        Exception_Validation_Input
            If *target_age* is not greater than insured age.
        """
        if not isinstance(target_age, int) or target_age <= self.m_insured_age:
            raise Exception_Validation_Input(
                "target_age must be > current insured age",
                field_name="target_age",
                expected_type=int,
                actual_value=target_age,
            )

        years = target_age - self.m_insured_age
        annual_premium = self.calc_annual_premium()
        return round(annual_premium * years, 2)

    # ------------------------------------------------------------------
    # String representation
    # ------------------------------------------------------------------

    def get_insurance_as_string(self) -> str:
        """Return a human-readable string representation.

        Returns
        -------
        str
            Full description of traditional LTC policy.
        """
        smoker_text = "Smoker" if self.m_is_smoker else "Non-Smoker"
        discount_text = "w/ Married Discount" if self.m_is_married_discount else ""
        shared_text = "Shared Care" if self.m_shared_care else ""

        extras = ", ".join(filter(None, [discount_text, shared_text]))
        extras_str = f" ({extras})" if extras else ""

        return (
            f"Traditional LTC Insurance\n"
            f"  Insured Age:        {self.m_insured_age}\n"
            f"  Gender:             {self.m_gender.value}\n"
            f"  Health Class:       {self.m_health_class.value}\n"
            f"  Daily Benefit:      ${self.m_daily_benefit_amount:,.2f}\n"
            f"  Monthly Benefit:    ${self.calc_monthly_benefit_amount():,.2f}\n"
            f"  Benefit Period:     {self.m_benefit_period.value} years\n"
            f"  Elimination Period: {self.m_elimination_period.value} days\n"
            f"  Inflation:          {self.m_inflation_protection.value}\n"
            f"  Care Setting:       {self.m_care_setting.value}\n"
            f"  Non-Forfeiture:     {self.m_non_forfeiture.value}\n"
            f"  {smoker_text}{extras_str}\n"
            f"  Max Lifetime Benefit: "
            f"${self.calc_maximum_lifetime_benefit():,.2f}\n"
            f"  Annual Premium:     ${self.calc_annual_premium():,.2f}\n"
        )
