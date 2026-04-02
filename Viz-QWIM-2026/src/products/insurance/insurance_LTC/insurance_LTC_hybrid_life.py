r"""
Class for Long Term Care (LTC) Hybrid Life insurance.

===============================

A **Hybrid Life / LTC** product combines a permanent life insurance
chassis with long-term-care riders.  Key characteristics:

- **Single or limited-pay premium**: most hybrid life policies are
  funded with a single lump-sum premium or a limited number of annual
  payments (e.g. 5 or 10 years).
- **Death benefit**: if LTC benefits are never used, the full death
  benefit is paid to the beneficiary.
- **Acceleration of Death Benefit (ADB)**: the insured can "accelerate"
  (spend down) the death benefit to pay for qualified LTC expenses.
- **Extension of Benefits**: an optional rider that extends LTC
  benefits beyond the death benefit pool, effectively multiplying
  coverage (2x or 3x the death benefit).
- **Residual death benefit**: after LTC benefits are drawn, a reduced
  death benefit may still be payable.
- **Return of premium**: many policies guarantee a full or partial
  return of premium if the owner surrenders the policy.
- **No rate increases**: premiums are contractually guaranteed never
  to increase, unlike traditional LTC.

Benefit Structure
~~~~~~~~~~~~~~~~~
The total LTC benefit pool is:

$$
\\text{Pool}_{\\text{total}} = DB + \\text{Extension}
$$

where *DB* is the base death benefit and *Extension* is the additional
benefit provided by an extension-of-benefits rider (a multiple of DB).

Author
------
QWIM Team

Version
-------
0.7.0 (2026-02-13)
"""

from __future__ import annotations

import math

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
# Enumerations specific to Hybrid Life / LTC
# ======================================================================


class Premium_Structure_Hybrid_Life(Enum):
    """Premium payment structure for hybrid life / LTC products.

    Attributes
    ----------
    SINGLE : str
        One-time lump-sum premium.
    PAY_5 : str
        Five annual premium payments.
    PAY_10 : str
        Ten annual premium payments.
    PAY_20 : str
        Twenty annual premium payments.
    """

    SINGLE = "Single Premium"
    PAY_5 = "5-Pay"
    PAY_10 = "10-Pay"
    PAY_20 = "20-Pay"


class Extension_Of_Benefits(Enum):
    """Extension-of-benefits rider multiplier.

    The multiplier defines how much additional LTC benefit (as a
    multiple of the base death benefit) is available beyond the
    accelerated death benefit.

    Attributes
    ----------
    NONE : float
        No extension — LTC pool equals the death benefit only.
    TIMES_2 : float
        Extension equal to 2 × the death benefit.
    TIMES_3 : float
        Extension equal to 3 × the death benefit.
    TIMES_4 : float
        Extension equal to 4 × the death benefit.
    """

    NONE = 1.0
    TIMES_2 = 2.0
    TIMES_3 = 3.0
    TIMES_4 = 4.0


class Return_Of_Premium_Type(Enum):
    """Return-of-premium guarantee type.

    Attributes
    ----------
    NONE : str
        No return of premium on surrender.
    FULL : str
        100 % of premiums returned upon surrender.
    GRADED : str
        Percentage of premiums returned increases over time (graded
        schedule, typically reaching 100 % after 10 – 15 years).
    """

    NONE = "None"
    FULL = "Full Return of Premium"
    GRADED = "Graded Return of Premium"


# ======================================================================
# Hybrid Life / LTC insurance class
# ======================================================================


class Insurance_LTC_Hybrid_Life(Insurance_LTC_Base):
    r"""Hybrid Life / Long Term Care (LTC) Insurance.

    Combines a permanent life insurance policy with an accelerated
    death benefit for LTC and an optional extension-of-benefits rider.

    **Total LTC benefit pool**:

    $$
    \text{Pool}_{\text{LTC}} = DB \times M_{\text{extension}}
    $$

    where $DB$ is the base death benefit and $M_{\text{extension}}$
    is the extension-of-benefits multiplier (1×, 2×, 3×, or 4×).

    **Monthly LTC benefit**:

    $$
    B_{\text{monthly}} = \frac{\text{Pool}_{\text{LTC}}}
        {P_{\text{months}}}
    $$

    where $P_{\text{months}}$ is the benefit period in months.

    Parameters
    ----------
    insured_age : int
        Issue / current age of the insured (18 – 85).
    death_benefit : float
        Base death benefit in dollars.
    single_premium : float
        Lump-sum or total premium amount.
    premium_structure : Premium_Structure_Hybrid_Life, optional
        How the premium is structured (default ``SINGLE``).
    extension_of_benefits : Extension_Of_Benefits, optional
        Extension-of-benefits rider multiplier (default ``TIMES_2``).
    return_of_premium : Return_Of_Premium_Type, optional
        Return-of-premium guarantee (default ``FULL``).
    benefit_period : Benefit_Period, optional
        Maximum benefit duration (default ``YEARS_6``).
    elimination_period : Elimination_Period, optional
        Waiting period in days (default ``DAYS_90``).
    inflation_protection : Inflation_Protection, optional
        Inflation rider (default ``NONE``).
    care_setting : Care_Setting, optional
        Covered care settings (default ``ALL``).
    is_smoker : bool, optional
        Tobacco usage (default ``False``).
    is_married_discount : bool, optional
        Marital discount (default ``False``).
    cash_value_growth_rate : float, optional
        Annual guaranteed cash-value growth rate (default 0.02).
    residual_death_benefit_pct : float, optional
        Minimum death benefit retained after LTC claims, expressed as
        a fraction of the original death benefit (default 0.10 = 10 %).

    Attributes
    ----------
    m_death_benefit : float
    m_single_premium : float
    m_premium_structure : Premium_Structure_Hybrid_Life
    m_extension_of_benefits : Extension_Of_Benefits
    m_return_of_premium : Return_Of_Premium_Type
    m_cash_value_growth_rate : float
    m_residual_death_benefit_pct : float
    """

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def __init__(
        self,
        insured_age: int,
        death_benefit: float,
        single_premium: float,
        premium_structure: Premium_Structure_Hybrid_Life = (Premium_Structure_Hybrid_Life.SINGLE),
        extension_of_benefits: Extension_Of_Benefits = (Extension_Of_Benefits.TIMES_2),
        return_of_premium: Return_Of_Premium_Type = (Return_Of_Premium_Type.FULL),
        benefit_period: Benefit_Period = Benefit_Period.YEARS_6,
        elimination_period: Elimination_Period = Elimination_Period.DAYS_90,
        inflation_protection: Inflation_Protection = Inflation_Protection.NONE,
        care_setting: Care_Setting = Care_Setting.ALL,
        is_smoker: bool = False,
        is_married_discount: bool = False,
        cash_value_growth_rate: float = 0.02,
        residual_death_benefit_pct: float = 0.10,
    ) -> None:
        """Initialize a Hybrid Life / LTC Insurance instance."""
        # ---- Validate class-specific inputs ----
        if not isinstance(death_benefit, (int, float)) or death_benefit <= 0:
            raise Exception_Validation_Input(
                "death_benefit must be a positive number",
                field_name="death_benefit",
                expected_type=float,
                actual_value=death_benefit,
            )

        if not isinstance(single_premium, (int, float)) or single_premium <= 0:
            raise Exception_Validation_Input(
                "single_premium must be a positive number",
                field_name="single_premium",
                expected_type=float,
                actual_value=single_premium,
            )

        if not isinstance(premium_structure, Premium_Structure_Hybrid_Life):
            raise Exception_Validation_Input(
                "premium_structure must be a valid Premium_Structure_Hybrid_Life enum",
                field_name="premium_structure",
                expected_type=Premium_Structure_Hybrid_Life,
                actual_value=premium_structure,
            )

        if not isinstance(extension_of_benefits, Extension_Of_Benefits):
            raise Exception_Validation_Input(
                "extension_of_benefits must be a valid Extension_Of_Benefits enum",
                field_name="extension_of_benefits",
                expected_type=Extension_Of_Benefits,
                actual_value=extension_of_benefits,
            )

        if not isinstance(return_of_premium, Return_Of_Premium_Type):
            raise Exception_Validation_Input(
                "return_of_premium must be a valid Return_Of_Premium_Type enum",
                field_name="return_of_premium",
                expected_type=Return_Of_Premium_Type,
                actual_value=return_of_premium,
            )

        if (
            not isinstance(cash_value_growth_rate, (int, float))
            or cash_value_growth_rate < 0.0
            or cash_value_growth_rate > 0.10
        ):
            raise Exception_Validation_Input(
                "cash_value_growth_rate must be in [0.0, 0.10]",
                field_name="cash_value_growth_rate",
                expected_type=float,
                actual_value=cash_value_growth_rate,
            )

        if (
            not isinstance(residual_death_benefit_pct, (int, float))
            or residual_death_benefit_pct < 0.0
            or residual_death_benefit_pct > 1.0
        ):
            raise Exception_Validation_Input(
                "residual_death_benefit_pct must be in [0.0, 1.0]",
                field_name="residual_death_benefit_pct",
                expected_type=float,
                actual_value=residual_death_benefit_pct,
            )

        # ---- Derive daily benefit from pool and benefit period ----
        # For hybrid life, the daily benefit is derived from the pool.
        total_pool = float(death_benefit) * extension_of_benefits.value
        benefit_period_months = benefit_period.value * 12
        monthly_benefit = total_pool / benefit_period_months
        daily_benefit_amount = monthly_benefit / 30.0

        # ---- Base class initialisation ----
        # Premium frequency for hybrid life is always determined by
        # the premium structure (SINGLE maps to SINGLE, multi-pay to
        # ANNUAL).
        _freq_map: dict[Premium_Structure_Hybrid_Life, Premium_Frequency_LTC] = {
            Premium_Structure_Hybrid_Life.SINGLE: Premium_Frequency_LTC.SINGLE,
            Premium_Structure_Hybrid_Life.PAY_5: Premium_Frequency_LTC.ANNUAL,
            Premium_Structure_Hybrid_Life.PAY_10: Premium_Frequency_LTC.ANNUAL,
            Premium_Structure_Hybrid_Life.PAY_20: Premium_Frequency_LTC.ANNUAL,
        }

        super().__init__(
            insured_age=insured_age,
            daily_benefit_amount=daily_benefit_amount,
            insurance_type=Insurance_LTC_Type.HYBRID_LIFE,
            benefit_period=benefit_period,
            elimination_period=elimination_period,
            inflation_protection=inflation_protection,
            care_setting=care_setting,
            premium_frequency=_freq_map.get(
                premium_structure,
                Premium_Frequency_LTC.SINGLE,
            ),
            is_smoker=is_smoker,
            is_married_discount=is_married_discount,
        )

        # ---- Class-specific member variables ----
        self.m_death_benefit: float = float(death_benefit)
        self.m_single_premium: float = float(single_premium)
        self.m_premium_structure: Premium_Structure_Hybrid_Life = premium_structure
        self.m_extension_of_benefits: Extension_Of_Benefits = extension_of_benefits
        self.m_return_of_premium: Return_Of_Premium_Type = return_of_premium
        self.m_cash_value_growth_rate: float = float(cash_value_growth_rate)
        self.m_residual_death_benefit_pct: float = float(
            residual_death_benefit_pct,
        )

        logger.info(
            f"Hybrid Life LTC: DB=${death_benefit:,.2f}, "
            f"premium=${single_premium:,.2f} ({premium_structure.value}), "
            f"extension={extension_of_benefits.value}x, "
            f"ROP={return_of_premium.value}",
        )

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def death_benefit(self) -> float:
        """Float : Base death benefit in dollars."""
        return self.m_death_benefit

    @property
    def single_premium(self) -> float:
        """Float : Total premium amount (lump-sum or total of payments)."""
        return self.m_single_premium

    @property
    def premium_structure(self) -> Premium_Structure_Hybrid_Life:
        """Premium_Structure_Hybrid_Life : Premium payment structure."""
        return self.m_premium_structure

    @property
    def extension_of_benefits(self) -> Extension_Of_Benefits:
        """Extension_Of_Benefits : Extension rider multiplier."""
        return self.m_extension_of_benefits

    @property
    def return_of_premium(self) -> Return_Of_Premium_Type:
        """Return_Of_Premium_Type : Return-of-premium guarantee."""
        return self.m_return_of_premium

    @property
    def cash_value_growth_rate(self) -> float:
        """Float : Annual guaranteed cash-value growth rate."""
        return self.m_cash_value_growth_rate

    @property
    def residual_death_benefit_pct(self) -> float:
        """Float : Minimum DB retained after LTC claims (fraction)."""
        return self.m_residual_death_benefit_pct

    # ------------------------------------------------------------------
    # Calculation methods (abstract overrides)
    # ------------------------------------------------------------------

    def calc_maximum_lifetime_benefit(self) -> float:
        r"""Calculate the total LTC benefit pool.

        $$
        \text{Pool}_{\text{LTC}} = DB \times M_{\text{extension}}
        $$

        Returns
        -------
        float
            Total LTC pool in dollars.
        """
        return self.m_death_benefit * self.m_extension_of_benefits.value

    def calc_annual_premium(self) -> float:
        r"""Calculate the annual (or equivalent) premium.

        For a single-premium product, the "annual premium" equals the
        single premium.  For multi-pay structures:

        $$
        P_{\text{annual}} = \frac{P_{\text{total}}}{N_{\text{years}}}
        $$

        Returns
        -------
        float
            Annual premium amount.
        """
        _pay_years: dict[Premium_Structure_Hybrid_Life, int] = {
            Premium_Structure_Hybrid_Life.SINGLE: 1,
            Premium_Structure_Hybrid_Life.PAY_5: 5,
            Premium_Structure_Hybrid_Life.PAY_10: 10,
            Premium_Structure_Hybrid_Life.PAY_20: 20,
        }
        years = _pay_years.get(self.m_premium_structure, 1)
        return round(self.m_single_premium / years, 2)

    def calc_daily_benefit_at_year(self, year: int) -> float:
        r"""Calculate the daily benefit at a given policy year.

        For hybrid life, the daily benefit is derived from the total
        pool divided by the benefit period.  Inflation protection
        applies on top.

        With **compound** inflation:

        $$
        B_{\text{daily}}(t) = B_{\text{daily,base}}
            \times (1 + r)^{t-1}
        $$

        Parameters
        ----------
        year : int
            Policy year (1-indexed).

        Returns
        -------
        float
            Daily benefit at the specified year.

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
        t = year - 1

        if growth_rate == 0.0:
            return self.m_daily_benefit_amount

        if self.calc_is_compound_inflation():
            return self.m_daily_benefit_amount * (1.0 + growth_rate) ** t

        return self.m_daily_benefit_amount * (1.0 + growth_rate * t)

    # ------------------------------------------------------------------
    # Additional calculation methods
    # ------------------------------------------------------------------

    def calc_monthly_LTC_benefit(self) -> float:
        r"""Calculate the monthly LTC benefit amount.

        $$
        B_{\text{monthly}} = \frac{\text{Pool}_{\text{LTC}}}
            {P_{\text{years}} \times 12}
        $$

        Returns
        -------
        float
            Monthly LTC benefit in dollars.
        """
        pool = self.calc_maximum_lifetime_benefit()
        months = self.m_benefit_period.value * 12
        if months == 0:
            return 0.0
        return round(pool / months, 2)

    def calc_cash_value_at_year(self, year: int) -> float:
        r"""Calculate the guaranteed cash value at a given policy year.

        Cash value grows at a guaranteed rate from the single premium:

        $$
        CV(t) = P_{\text{single}} \times (1 + g)^{t}
        $$

        Parameters
        ----------
        year : int
            Policy year (1-indexed).

        Returns
        -------
        float
            Guaranteed cash value at the specified year.

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

        return round(
            self.m_single_premium * (1.0 + self.m_cash_value_growth_rate) ** year,
            2,
        )

    def calc_surrender_value(self, year: int) -> float:
        r"""Calculate the surrender value at a given policy year.

        For policies with a **full** return-of-premium guarantee:

        $$
        SV(t) = \max\bigl(CV(t),\; P_{\text{single}}\bigr)
        $$

        For **graded** ROP:

        $$
        SV(t) = \max\bigl(CV(t),\;
            P_{\text{single}} \times g_{\text{schedule}}(t)\bigr)
        $$

        where the graded schedule reaches 100 % at year 10.

        Parameters
        ----------
        year : int
            Policy year (1-indexed).

        Returns
        -------
        float
            Surrender value in dollars.

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

        cv = self.calc_cash_value_at_year(year)

        if self.m_return_of_premium == Return_Of_Premium_Type.FULL:
            return max(cv, self.m_single_premium)

        if self.m_return_of_premium == Return_Of_Premium_Type.GRADED:
            # Graded schedule: 80 % in year 1 → 100 % by year 10
            graded_pct = min(0.80 + 0.02 * year, 1.0)
            guaranteed_floor = self.m_single_premium * graded_pct
            return max(cv, guaranteed_floor)

        # No ROP guarantee — just cash value
        return cv

    def calc_residual_death_benefit(
        self,
        LTC_benefits_paid: float,
    ) -> float:
        r"""Calculate the death benefit remaining after LTC claims.

        $$
        DB_{\text{residual}} = \max\bigl(
            DB - \text{LTC paid},\;
            DB \times R_{\text{min}}\bigr)
        $$

        where $R_{\text{min}}$ is the residual death benefit
        percentage floor.

        Parameters
        ----------
        LTC_benefits_paid : float
            Total LTC benefits paid out to date.

        Returns
        -------
        float
            Residual death benefit in dollars.

        Raises
        ------
        Exception_Validation_Input
            If *LTC_benefits_paid* is negative.
        """
        if not isinstance(LTC_benefits_paid, (int, float)) or LTC_benefits_paid < 0:
            raise Exception_Validation_Input(
                "LTC_benefits_paid must be non-negative",
                field_name="LTC_benefits_paid",
                expected_type=float,
                actual_value=LTC_benefits_paid,
            )

        reduced_db = self.m_death_benefit - float(LTC_benefits_paid)
        minimum_db = self.m_death_benefit * self.m_residual_death_benefit_pct

        return max(reduced_db, minimum_db)

    def calc_remaining_LTC_pool(
        self,
        LTC_benefits_paid: float,
    ) -> float:
        r"""Calculate the remaining LTC benefit pool.

        $$
        \text{Pool}_{\text{remaining}} =
            \text{Pool}_{\text{total}} - \text{LTC paid}
        $$

        Parameters
        ----------
        LTC_benefits_paid : float
            Total LTC benefits paid out.

        Returns
        -------
        float
            Remaining LTC pool (floored at 0).

        Raises
        ------
        Exception_Validation_Input
            If *LTC_benefits_paid* is negative.
        """
        if not isinstance(LTC_benefits_paid, (int, float)) or LTC_benefits_paid < 0:
            raise Exception_Validation_Input(
                "LTC_benefits_paid must be non-negative",
                field_name="LTC_benefits_paid",
                expected_type=float,
                actual_value=LTC_benefits_paid,
            )

        pool = self.calc_maximum_lifetime_benefit()
        return max(pool - float(LTC_benefits_paid), 0.0)

    def calc_benefit_leverage_ratio(self) -> float:
        r"""Calculate the benefit-to-premium leverage ratio.

        $$
        \text{Leverage} = \frac{\text{Pool}_{\text{LTC}}}
            {P_{\text{single}}}
        $$

        This shows how many dollars of LTC coverage are obtained per
        dollar of premium.

        Returns
        -------
        float
            LTC pool divided by total premium.
        """
        if self.m_single_premium == 0:
            return 0.0

        pool = self.calc_maximum_lifetime_benefit()
        return round(pool / self.m_single_premium, 4)

    def calc_internal_rate_of_return_death(
        self,
        year_of_death: int,
    ) -> float:
        r"""Estimate the IRR if the insured dies with no LTC claims.

        With zero LTC claims, the beneficiary receives the full death
        benefit.  For a single-premium policy:

        $$
        \text{IRR} = \left(\frac{DB}{P_{\text{single}}}\right)
            ^{1/t} - 1
        $$

        Parameters
        ----------
        year_of_death : int
            Year of death (1-indexed policy year).

        Returns
        -------
        float
            Estimated IRR as a decimal.

        Raises
        ------
        Exception_Validation_Input
            If *year_of_death* is not a positive integer.
        Exception_Calculation
            If calculation is not feasible.
        """
        if not isinstance(year_of_death, int) or year_of_death < 1:
            raise Exception_Validation_Input(
                "year_of_death must be a positive integer",
                field_name="year_of_death",
                expected_type=int,
                actual_value=year_of_death,
            )

        if self.m_single_premium <= 0:
            raise Exception_Calculation(
                "Cannot compute IRR with zero or negative premium",
            )

        ratio = self.m_death_benefit / self.m_single_premium

        if ratio <= 0:
            raise Exception_Calculation(
                "Death benefit must be positive for IRR calculation",
            )

        irr = ratio ** (1.0 / year_of_death) - 1.0
        return round(irr, 6)

    def calc_breakeven_year_vs_traditional(
        self,
        traditional_annual_premium: float,
    ) -> int:
        r"""Estimate the break-even year vs. a traditional LTC policy.

        Finds the year at which cumulative traditional premiums equal
        the hybrid life single premium:

        $$
        N = \left\lceil
            \frac{P_{\text{single}}}{P_{\text{traditional,annual}}}
        \right\rceil
        $$

        Parameters
        ----------
        traditional_annual_premium : float
            Annual premium of the comparable traditional LTC policy.

        Returns
        -------
        int
            Break-even year (rounded up).

        Raises
        ------
        Exception_Validation_Input
            If *traditional_annual_premium* is not positive.
        """
        if (
            not isinstance(traditional_annual_premium, (int, float))
            or traditional_annual_premium <= 0
        ):
            raise Exception_Validation_Input(
                "traditional_annual_premium must be a positive number",
                field_name="traditional_annual_premium",
                expected_type=float,
                actual_value=traditional_annual_premium,
            )

        return math.ceil(
            self.m_single_premium / float(traditional_annual_premium),
        )

    # ------------------------------------------------------------------
    # String representation
    # ------------------------------------------------------------------

    def get_insurance_as_string(self) -> str:
        """Return a human-readable string representation.

        Returns
        -------
        str
            Full description of the hybrid life / LTC policy.
        """
        smoker_text = "Smoker" if self.m_is_smoker else "Non-Smoker"
        pool = self.calc_maximum_lifetime_benefit()
        monthly = self.calc_monthly_LTC_benefit()

        return (
            f"Hybrid Life / LTC Insurance\n"
            f"  Insured Age:          {self.m_insured_age}\n"
            f"  Death Benefit:        ${self.m_death_benefit:,.2f}\n"
            f"  Premium:              ${self.m_single_premium:,.2f} "
            f"({self.m_premium_structure.value})\n"
            f"  Extension:            {self.m_extension_of_benefits.value}x\n"
            f"  Total LTC Pool:       ${pool:,.2f}\n"
            f"  Monthly LTC Benefit:  ${monthly:,.2f}\n"
            f"  Benefit Period:       {self.m_benefit_period.value} years\n"
            f"  Elimination Period:   {self.m_elimination_period.value} days\n"
            f"  Inflation:            {self.m_inflation_protection.value}\n"
            f"  Return of Premium:    {self.m_return_of_premium.value}\n"
            f"  Residual DB Floor:    "
            f"{self.m_residual_death_benefit_pct * 100:.0f}%\n"
            f"  {smoker_text}\n"
            f"  Leverage Ratio:       "
            f"{self.calc_benefit_leverage_ratio():.2f}x\n"
        )
