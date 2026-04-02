r"""
Class for Long Term Care (LTC) Hybrid Annuity insurance.

===============================

A **Hybrid Annuity / LTC** product combines a deferred annuity
chassis with a long-term-care benefit multiplier.  Key characteristics:

- **Single-premium funding**: the owner deposits a lump sum into a
  deferred annuity.
- **LTC benefit multiplier**: if the annuitant qualifies for LTC
  benefits, the policy pays a monthly benefit that is a multiple
  (typically 2× or 3×) of the account value spread over the benefit
  period.  This multiplier effectively creates a benefit pool larger
  than the deposit.
- **Account value growth**: the underlying annuity account grows at
  a guaranteed minimum crediting rate, sometimes supplemented by
  excess interest.
- **Guaranteed minimum surrender value**: even if LTC benefits are
  drawn, a minimum percentage of the original deposit is typically
  available upon surrender.
- **Death benefit**: the greater of the remaining account value or
  a guaranteed minimum is paid to the beneficiary.
- **No rate increases**: premiums are locked in at issue.
- **Tax advantages**: LTC benefits paid from a hybrid annuity can
  qualify as tax-free under IRC §7702B if the contract is a
  qualified long-term-care combination product.

Benefit Structure
~~~~~~~~~~~~~~~~~
$$
\\text{Pool}_{\\text{LTC}} = AV \\times M_{\\text{multiplier}}
$$

where *AV* is the account value at the time of claim and
$M_{\\text{multiplier}}$ is the LTC benefit multiplier (2× or 3×).

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
# Enumerations specific to Hybrid Annuity / LTC
# ======================================================================


class LTC_Multiplier(Enum):
    """LTC benefit multiplier applied to the account value.

    Attributes
    ----------
    TIMES_2 : float
        2× the account value.
    TIMES_3 : float
        3× the account value.
    TIMES_4 : float
        4× the account value.
    """

    TIMES_2 = 2.0
    TIMES_3 = 3.0
    TIMES_4 = 4.0


class Annuity_Type_Hybrid(Enum):
    """Underlying annuity type for the hybrid product.

    Attributes
    ----------
    FIXED : str
        Fixed deferred annuity with guaranteed crediting rate.
    FIXED_INDEXED : str
        Fixed-indexed annuity with index-linked excess interest.
    MULTI_YEAR_GUARANTEED : str
        Multi-year guaranteed annuity (MYGA).
    """

    FIXED = "Fixed Deferred Annuity"
    FIXED_INDEXED = "Fixed-Indexed Annuity"
    MULTI_YEAR_GUARANTEED = "Multi-Year Guaranteed Annuity"


class Surrender_Schedule_Type(Enum):
    """Surrender-charge schedule type.

    Attributes
    ----------
    NONE : str
        No surrender charges.
    STANDARD : str
        Standard declining surrender-charge schedule (typically 7 – 10
        years).
    SHORT : str
        Short surrender period (3 – 5 years).
    """

    NONE = "None"
    STANDARD = "Standard (7-10 yr)"
    SHORT = "Short (3-5 yr)"


# ======================================================================
# Hybrid Annuity / LTC insurance class
# ======================================================================


class Insurance_LTC_Hybrid_Annuity(Insurance_LTC_Base):
    r"""Hybrid Annuity / Long Term Care (LTC) Insurance.

    Combines a single-premium deferred annuity with an LTC benefit
    multiplier.  The annuity account value grows at a guaranteed
    crediting rate and is accessible for LTC benefits or death.

    **Total LTC benefit pool at time of claim**:

    $$
    \text{Pool}_{\text{LTC}}(t) = AV(t) \times M_{\text{mult}}
    $$

    where $AV(t)$ is the account value at policy year $t$ and
    $M_{\text{mult}}$ is the LTC multiplier.

    **Account value growth**:

    $$
    AV(t) = P_{\text{single}} \times (1 + g)^{t}
    $$

    where $g$ is the guaranteed minimum crediting rate.

    **Monthly LTC benefit**:

    $$
    B_{\text{monthly}} = \frac{\text{Pool}_{\text{LTC}}(t)}
        {P_{\text{months}}}
    $$

    Parameters
    ----------
    insured_age : int
        Issue / current age of the insured (18 – 85).
    single_premium : float
        Lump-sum deposit amount.
    LTC_multiplier : LTC_Multiplier, optional
        Benefit multiplier (default ``TIMES_2``).
    annuity_type : Annuity_Type_Hybrid, optional
        Underlying annuity type (default ``FIXED``).
    guaranteed_crediting_rate : float, optional
        Minimum annual crediting rate (default 0.02 = 2 %).
    benefit_period : Benefit_Period, optional
        Maximum LTC benefit duration (default ``YEARS_6``).
    elimination_period : Elimination_Period, optional
        Waiting period in days (default ``DAYS_90``).
    inflation_protection : Inflation_Protection, optional
        Inflation rider (default ``NONE``).
    care_setting : Care_Setting, optional
        Covered care settings (default ``ALL``).
    is_smoker : bool, optional
        Tobacco usage (default ``False``).
    is_married_discount : bool, optional
        Marital / partner discount (default ``False``).
    surrender_schedule : Surrender_Schedule_Type, optional
        Surrender-charge schedule type (default ``STANDARD``).
    guaranteed_min_surrender_pct : float, optional
        Guaranteed minimum surrender value as a fraction of the
        original deposit (default 0.90 = 90 %).
    death_benefit_floor_pct : float, optional
        Minimum death benefit as a fraction of the original deposit
        (default 1.0 = 100 % return of premium on death).

    Attributes
    ----------
    m_single_premium : float
    m_LTC_multiplier : LTC_Multiplier
    m_annuity_type : Annuity_Type_Hybrid
    m_guaranteed_crediting_rate : float
    m_surrender_schedule : Surrender_Schedule_Type
    m_guaranteed_min_surrender_pct : float
    m_death_benefit_floor_pct : float
    """

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def __init__(
        self,
        insured_age: int,
        single_premium: float,
        LTC_multiplier: LTC_Multiplier = LTC_Multiplier.TIMES_2,
        annuity_type: Annuity_Type_Hybrid = Annuity_Type_Hybrid.FIXED,
        guaranteed_crediting_rate: float = 0.02,
        benefit_period: Benefit_Period = Benefit_Period.YEARS_6,
        elimination_period: Elimination_Period = Elimination_Period.DAYS_90,
        inflation_protection: Inflation_Protection = Inflation_Protection.NONE,
        care_setting: Care_Setting = Care_Setting.ALL,
        is_smoker: bool = False,
        is_married_discount: bool = False,
        surrender_schedule: Surrender_Schedule_Type = (Surrender_Schedule_Type.STANDARD),
        guaranteed_min_surrender_pct: float = 0.90,
        death_benefit_floor_pct: float = 1.0,
    ) -> None:
        """Initialize a Hybrid Annuity / LTC Insurance instance."""
        # ---- Validate class-specific inputs ----
        if not isinstance(single_premium, (int, float)) or single_premium <= 0:
            raise Exception_Validation_Input(
                "single_premium must be a positive number",
                field_name="single_premium",
                expected_type=float,
                actual_value=single_premium,
            )

        if not isinstance(LTC_multiplier, LTC_Multiplier):
            raise Exception_Validation_Input(
                "LTC_multiplier must be a valid LTC_Multiplier enum",
                field_name="LTC_multiplier",
                expected_type=LTC_Multiplier,
                actual_value=LTC_multiplier,
            )

        if not isinstance(annuity_type, Annuity_Type_Hybrid):
            raise Exception_Validation_Input(
                "annuity_type must be a valid Annuity_Type_Hybrid enum",
                field_name="annuity_type",
                expected_type=Annuity_Type_Hybrid,
                actual_value=annuity_type,
            )

        if (
            not isinstance(guaranteed_crediting_rate, (int, float))
            or guaranteed_crediting_rate < 0.0
            or guaranteed_crediting_rate > 0.10
        ):
            raise Exception_Validation_Input(
                "guaranteed_crediting_rate must be in [0.0, 0.10]",
                field_name="guaranteed_crediting_rate",
                expected_type=float,
                actual_value=guaranteed_crediting_rate,
            )

        if not isinstance(surrender_schedule, Surrender_Schedule_Type):
            raise Exception_Validation_Input(
                "surrender_schedule must be a valid Surrender_Schedule_Type enum",
                field_name="surrender_schedule",
                expected_type=Surrender_Schedule_Type,
                actual_value=surrender_schedule,
            )

        if (
            not isinstance(guaranteed_min_surrender_pct, (int, float))
            or guaranteed_min_surrender_pct < 0.0
            or guaranteed_min_surrender_pct > 1.0
        ):
            raise Exception_Validation_Input(
                "guaranteed_min_surrender_pct must be in [0.0, 1.0]",
                field_name="guaranteed_min_surrender_pct",
                expected_type=float,
                actual_value=guaranteed_min_surrender_pct,
            )

        if (
            not isinstance(death_benefit_floor_pct, (int, float))
            or death_benefit_floor_pct < 0.0
            or death_benefit_floor_pct > 2.0
        ):
            raise Exception_Validation_Input(
                "death_benefit_floor_pct must be in [0.0, 2.0]",
                field_name="death_benefit_floor_pct",
                expected_type=float,
                actual_value=death_benefit_floor_pct,
            )

        # ---- Derive daily benefit from pool and benefit period ----
        initial_pool = float(single_premium) * LTC_multiplier.value
        benefit_period_months = benefit_period.value * 12
        monthly_benefit = initial_pool / benefit_period_months if benefit_period_months > 0 else 0.0
        daily_benefit_amount = monthly_benefit / 30.0

        # ---- Base class initialisation ----
        super().__init__(
            insured_age=insured_age,
            daily_benefit_amount=daily_benefit_amount,
            insurance_type=Insurance_LTC_Type.HYBRID_ANNUITY,
            benefit_period=benefit_period,
            elimination_period=elimination_period,
            inflation_protection=inflation_protection,
            care_setting=care_setting,
            premium_frequency=Premium_Frequency_LTC.SINGLE,
            is_smoker=is_smoker,
            is_married_discount=is_married_discount,
        )

        # ---- Class-specific member variables ----
        self.m_single_premium: float = float(single_premium)
        self.m_LTC_multiplier: LTC_Multiplier = LTC_multiplier
        self.m_annuity_type: Annuity_Type_Hybrid = annuity_type
        self.m_guaranteed_crediting_rate: float = float(
            guaranteed_crediting_rate,
        )
        self.m_surrender_schedule: Surrender_Schedule_Type = surrender_schedule
        self.m_guaranteed_min_surrender_pct: float = float(
            guaranteed_min_surrender_pct,
        )
        self.m_death_benefit_floor_pct: float = float(
            death_benefit_floor_pct,
        )

        logger.info(
            f"Hybrid Annuity LTC: deposit=${single_premium:,.2f}, "
            f"multiplier={LTC_multiplier.value}x, "
            f"annuity_type={annuity_type.value}, "
            f"crediting_rate={guaranteed_crediting_rate:.2%}",
        )

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def single_premium(self) -> float:
        """Float : Lump-sum deposit amount."""
        return self.m_single_premium

    @property
    def LTC_multiplier(self) -> LTC_Multiplier:
        """LTC_Multiplier : LTC benefit multiplier."""
        return self.m_LTC_multiplier

    @property
    def annuity_type(self) -> Annuity_Type_Hybrid:
        """Annuity_Type_Hybrid : Underlying annuity type."""
        return self.m_annuity_type

    @property
    def guaranteed_crediting_rate(self) -> float:
        """Float : Minimum annual crediting rate."""
        return self.m_guaranteed_crediting_rate

    @property
    def surrender_schedule(self) -> Surrender_Schedule_Type:
        """Surrender_Schedule_Type : Surrender-charge schedule."""
        return self.m_surrender_schedule

    @property
    def guaranteed_min_surrender_pct(self) -> float:
        """Float : Guaranteed minimum surrender value (fraction)."""
        return self.m_guaranteed_min_surrender_pct

    @property
    def death_benefit_floor_pct(self) -> float:
        """Float : Minimum death benefit as fraction of deposit."""
        return self.m_death_benefit_floor_pct

    # ------------------------------------------------------------------
    # Calculation methods (abstract overrides)
    # ------------------------------------------------------------------

    def calc_maximum_lifetime_benefit(self) -> float:
        r"""Calculate the total LTC benefit pool (at issue).

        $$
        \text{Pool}_{\text{LTC}} = P_{\text{single}}
            \times M_{\text{mult}}
        $$

        Returns
        -------
        float
            Total LTC pool at issue in dollars.
        """
        return self.m_single_premium * self.m_LTC_multiplier.value

    def calc_annual_premium(self) -> float:
        r"""Return the single premium.

        For a single-premium annuity hybrid, the "annual premium"
        is simply the one-time deposit.

        Returns
        -------
        float
            The single premium amount.
        """
        return self.m_single_premium

    def calc_daily_benefit_at_year(self, year: int) -> float:
        r"""Calculate the daily benefit at a given policy year.

        The daily benefit grows with the account value (via the
        guaranteed crediting rate) and inflation protection.

        $$
        B_{\text{daily}}(t) = \frac{AV(t) \times M_{\text{mult}}}
            {P_{\text{years}} \times 365}
        $$

        If inflation protection is selected, the larger of the
        AV-derived benefit and the inflation-adjusted benefit is used.

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

        # AV-based daily benefit
        av_at_year = self.calc_account_value_at_year(year)
        pool_at_year = av_at_year * self.m_LTC_multiplier.value
        benefit_days = self.m_benefit_period.value * 365.0
        av_daily = pool_at_year / benefit_days if benefit_days > 0 else 0.0

        # Inflation-adjusted daily benefit (from base amount)
        growth_rate = self.calc_inflation_growth_rate()
        t = year - 1

        if growth_rate == 0.0:
            infl_daily = self.m_daily_benefit_amount
        elif self.calc_is_compound_inflation():
            infl_daily = self.m_daily_benefit_amount * (1.0 + growth_rate) ** t
        else:
            infl_daily = self.m_daily_benefit_amount * (1.0 + growth_rate * t)

        # Use the larger of the two
        return max(av_daily, infl_daily)

    # ------------------------------------------------------------------
    # Additional calculation methods
    # ------------------------------------------------------------------

    def calc_account_value_at_year(self, year: int) -> float:
        r"""Calculate the guaranteed account value at a given year.

        $$
        AV(t) = P_{\text{single}} \times (1 + g)^{t}
        $$

        Parameters
        ----------
        year : int
            Policy year (1-indexed).

        Returns
        -------
        float
            Guaranteed account value at the specified year.

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
            self.m_single_premium * (1.0 + self.m_guaranteed_crediting_rate) ** year,
            2,
        )

    def calc_LTC_pool_at_year(self, year: int) -> float:
        r"""Calculate the total LTC pool at a given policy year.

        $$
        \text{Pool}_{\text{LTC}}(t) = AV(t) \times M_{\text{mult}}
        $$

        Parameters
        ----------
        year : int
            Policy year (1-indexed).

        Returns
        -------
        float
            Total LTC pool at the specified year.

        Raises
        ------
        Exception_Validation_Input
            If *year* is not a positive integer.
        """
        av = self.calc_account_value_at_year(year)
        return round(av * self.m_LTC_multiplier.value, 2)

    def calc_monthly_LTC_benefit_at_year(self, year: int) -> float:
        r"""Calculate the monthly LTC benefit at a given policy year.

        $$
        B_{\text{monthly}}(t) = \frac{AV(t) \times M_{\text{mult}}}
            {P_{\text{years}} \times 12}
        $$

        Parameters
        ----------
        year : int
            Policy year (1-indexed).

        Returns
        -------
        float
            Monthly LTC benefit at the specified year.

        Raises
        ------
        Exception_Validation_Input
            If *year* is not a positive integer.
        """
        pool = self.calc_LTC_pool_at_year(year)
        months = self.m_benefit_period.value * 12

        if months == 0:
            return 0.0

        return round(pool / months, 2)

    def calc_surrender_value(self, year: int) -> float:
        r"""Calculate the surrender value at a given policy year.

        $$
        SV(t) = \max\bigl(
            AV(t) \times (1 - SC(t)),\;
            P_{\text{single}} \times G_{\text{min}}
        \bigr)
        $$

        where $SC(t)$ is the surrender charge for year $t$ and
        $G_{\text{min}}$ is the guaranteed minimum surrender percentage.

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

        av = self.calc_account_value_at_year(year)
        sc = self._get_surrender_charge_pct(year)
        net_av = av * (1.0 - sc)

        guaranteed_floor = self.m_single_premium * self.m_guaranteed_min_surrender_pct

        return round(max(net_av, guaranteed_floor), 2)

    def calc_death_benefit(self, year: int) -> float:
        r"""Calculate the death benefit at a given policy year.

        $$
        DB(t) = \max\bigl(
            AV(t),\;
            P_{\text{single}} \times F_{\text{floor}}
        \bigr)
        $$

        Parameters
        ----------
        year : int
            Policy year (1-indexed).

        Returns
        -------
        float
            Death benefit at the specified year.

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

        av = self.calc_account_value_at_year(year)
        floor = self.m_single_premium * self.m_death_benefit_floor_pct

        return round(max(av, floor), 2)

    def calc_death_benefit_after_LTC(
        self,
        LTC_benefits_paid: float,
        year: int,
    ) -> float:
        r"""Calculate remaining death benefit after LTC draws.

        $$
        DB_{\text{residual}}(t) = \max\bigl(
            AV(t) - \text{LTC paid},\;
            0
        \bigr)
        $$

        Parameters
        ----------
        LTC_benefits_paid : float
            Total LTC benefits drawn to date.
        year : int
            Current policy year (1-indexed).

        Returns
        -------
        float
            Residual death benefit (can be 0).

        Raises
        ------
        Exception_Validation_Input
            If inputs are invalid.
        """
        if not isinstance(LTC_benefits_paid, (int, float)) or LTC_benefits_paid < 0:
            raise Exception_Validation_Input(
                "LTC_benefits_paid must be non-negative",
                field_name="LTC_benefits_paid",
                expected_type=float,
                actual_value=LTC_benefits_paid,
            )

        av = self.calc_account_value_at_year(year)
        return round(max(av - float(LTC_benefits_paid), 0.0), 2)

    def calc_remaining_LTC_pool(
        self,
        LTC_benefits_paid: float,
        year: int = 1,
    ) -> float:
        r"""Calculate remaining LTC pool after partial claims.

        $$
        \text{Pool}_{\text{remaining}} =
            \text{Pool}_{\text{LTC}}(t) - \text{LTC paid}
        $$

        Parameters
        ----------
        LTC_benefits_paid : float
            Total LTC benefits drawn to date.
        year : int, optional
            Current policy year (default 1).

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

        pool = self.calc_LTC_pool_at_year(year)
        return round(max(pool - float(LTC_benefits_paid), 0.0), 2)

    def calc_benefit_leverage_ratio(self) -> float:
        r"""Calculate the benefit-to-deposit leverage ratio.

        $$
        \text{Leverage} = \frac{\text{Pool}_{\text{LTC}}}
            {P_{\text{single}}}
        $$

        Returns
        -------
        float
            LTC pool divided by the premium deposit.
        """
        if self.m_single_premium == 0:
            return 0.0

        pool = self.calc_maximum_lifetime_benefit()
        return round(pool / self.m_single_premium, 4)

    def calc_effective_annual_cost(self, year: int) -> float:
        r"""Calculate the effective annual cost of the LTC rider.

        Defined as the difference between the actual account value
        and the account value that would exist without the LTC rider
        (assuming the rider cost is embedded in a lower crediting
        rate or fees).  Here we approximate by comparing the growth
        to a market alternative:

        $$
        \text{Cost}(t) = P_{\text{single}} \times
            \bigl(r_{\text{market}} - g\bigr) \times t
        $$

        where $r_{\text{market}}$ is an assumed market alternative
        rate (4 %) and $g$ is the guaranteed crediting rate.

        Parameters
        ----------
        year : int
            Policy year (1-indexed).

        Returns
        -------
        float
            Estimated opportunity cost per year.

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

        market_rate = 0.04  # assumed alternative
        rate_diff = market_rate - self.m_guaranteed_crediting_rate
        if rate_diff <= 0:
            return 0.0

        return round(self.m_single_premium * rate_diff, 2)

    def calc_internal_rate_of_return_LTC(
        self,
        total_LTC_benefits_received: float,
        years_on_claim: int,
    ) -> float:
        r"""Estimate the IRR if LTC benefits are utilised.

        $$
        \text{IRR} = \left(
            \frac{\text{Total LTC Benefits}}{P_{\text{single}}}
        \right)^{1/t} - 1
        $$

        Parameters
        ----------
        total_LTC_benefits_received : float
            Total LTC benefits received over the claim period.
        years_on_claim : int
            Number of years benefits were received.

        Returns
        -------
        float
            Estimated IRR as a decimal.

        Raises
        ------
        Exception_Validation_Input
            If inputs are invalid.
        Exception_Calculation
            If calculation is not feasible.
        """
        if (
            not isinstance(total_LTC_benefits_received, (int, float))
            or total_LTC_benefits_received <= 0
        ):
            raise Exception_Validation_Input(
                "total_LTC_benefits_received must be positive",
                field_name="total_LTC_benefits_received",
                expected_type=float,
                actual_value=total_LTC_benefits_received,
            )

        if not isinstance(years_on_claim, int) or years_on_claim < 1:
            raise Exception_Validation_Input(
                "years_on_claim must be a positive integer",
                field_name="years_on_claim",
                expected_type=int,
                actual_value=years_on_claim,
            )

        if self.m_single_premium <= 0:
            raise Exception_Calculation(
                "Cannot compute IRR with zero or negative premium",
            )

        ratio = float(total_LTC_benefits_received) / self.m_single_premium

        if ratio <= 0:
            raise Exception_Calculation(
                "Benefits must be positive for IRR calculation",
            )

        irr = ratio ** (1.0 / years_on_claim) - 1.0
        return round(irr, 6)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _get_surrender_charge_pct(self, year: int) -> float:
        """Return the surrender charge percentage for a given year.

        Parameters
        ----------
        year : int
            Policy year (1-indexed).

        Returns
        -------
        float
            Surrender charge as a decimal (e.g. 0.07 = 7 %).
        """
        if self.m_surrender_schedule == Surrender_Schedule_Type.NONE:
            return 0.0

        if self.m_surrender_schedule == Surrender_Schedule_Type.SHORT:
            # 5-year declining: 5%, 4%, 3%, 2%, 1%, then 0%
            _schedule = {1: 0.05, 2: 0.04, 3: 0.03, 4: 0.02, 5: 0.01}
            return _schedule.get(year, 0.0)

        # Standard 10-year declining: 10%, 9%, ..., 1%, then 0%
        if year <= 10:
            return max(0.11 - 0.01 * year, 0.0)
        return 0.0

    # ------------------------------------------------------------------
    # String representation
    # ------------------------------------------------------------------

    def get_insurance_as_string(self) -> str:
        """Return a human-readable string representation.

        Returns
        -------
        str
            Full description of the hybrid annuity / LTC policy.
        """
        smoker_text = "Smoker" if self.m_is_smoker else "Non-Smoker"
        pool = self.calc_maximum_lifetime_benefit()
        monthly = self.calc_monthly_LTC_benefit_at_year(1)

        return (
            f"Hybrid Annuity / LTC Insurance\n"
            f"  Insured Age:            {self.m_insured_age}\n"
            f"  Single Premium:         ${self.m_single_premium:,.2f}\n"
            f"  Annuity Type:           {self.m_annuity_type.value}\n"
            f"  Crediting Rate:         "
            f"{self.m_guaranteed_crediting_rate:.2%}\n"
            f"  LTC Multiplier:         {self.m_LTC_multiplier.value}x\n"
            f"  Total LTC Pool (issue): ${pool:,.2f}\n"
            f"  Monthly LTC Benefit:    ${monthly:,.2f}\n"
            f"  Benefit Period:         "
            f"{self.m_benefit_period.value} years\n"
            f"  Elimination Period:     "
            f"{self.m_elimination_period.value} days\n"
            f"  Inflation:              "
            f"{self.m_inflation_protection.value}\n"
            f"  Surrender Schedule:     "
            f"{self.m_surrender_schedule.value}\n"
            f"  Min Surrender Value:    "
            f"{self.m_guaranteed_min_surrender_pct * 100:.0f}%\n"
            f"  Death Benefit Floor:    "
            f"{self.m_death_benefit_floor_pct * 100:.0f}%\n"
            f"  {smoker_text}\n"
            f"  Leverage Ratio:         "
            f"{self.calc_benefit_leverage_ratio():.2f}x\n"
        )
