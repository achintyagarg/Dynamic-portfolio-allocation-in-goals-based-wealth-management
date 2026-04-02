"""
Variable Life Insurance class.

============================

Variable Life Insurance (VL / VLI) provides permanent coverage where the
cash value and, optionally, the death benefit fluctuate with the
performance of underlying investment sub-accounts chosen by the
policyholder.

Key characteristics of Variable Life Insurance:

- **Fixed scheduled premiums** (unlike Variable Universal Life which has
  flexible premiums).
- **Investment sub-accounts**: policy cash value is allocated among
  equity, bond, money-market, and balanced sub-accounts similar to
  mutual funds.
- **Investment risk**: the policyholder bears the investment risk — cash
  values are not guaranteed and can decrease.
- **Guaranteed minimum death benefit**: the death benefit cannot drop
  below the face amount as long as premiums are paid as scheduled.
- **Mortality & expense (M&E) charges**: assessed against sub-account
  values to cover insurance risk and administrative costs.
- **Surrender charges**: early surrender incurs declining charges.
- **Policy loans**: available, typically charged a spread over the
  crediting rate on the collateralised portion.
- **SEC-regulated**: classified as a security; must be sold with a
  prospectus.

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
# VL-specific enumerations
# ======================================================================


class Sub_Account_Type(Enum):
    """Investment sub-account categories.

    Attributes
    ----------
    EQUITY_LARGE_CAP : str
        U.S. large-cap equity sub-account.
    EQUITY_SMALL_CAP : str
        U.S. small-cap equity sub-account.
    EQUITY_INTERNATIONAL : str
        International equity sub-account.
    FIXED_INCOME : str
        Bond / fixed-income sub-account.
    BALANCED : str
        Balanced (stock/bond mix) sub-account.
    MONEY_MARKET : str
        Money market / stable value sub-account.
    INDEX : str
        Passively managed index sub-account.
    """

    EQUITY_LARGE_CAP = "Equity - Large Cap"
    EQUITY_SMALL_CAP = "Equity - Small Cap"
    EQUITY_INTERNATIONAL = "Equity - International"
    FIXED_INCOME = "Fixed Income"
    BALANCED = "Balanced"
    MONEY_MARKET = "Money Market"
    INDEX = "Index"


class Insurance_Life_Variable(Insurance_Life_Base):
    r"""Variable Life Insurance product class.

    A fixed-premium permanent policy where cash value is invested in
    policyholder-selected sub-accounts.  The death benefit has a
    guaranteed minimum equal to the face amount, but may increase if
    sub-account performance is strong.

    **Monthly cash-value accumulation** (simplified):

    $$
    CV_m = CV_{m-1} \times (1 + r_{\text{sub},m}) + P_m - COI_m - M\&E_m - A_m
    $$

    Where:

    - $r_{\text{sub},m}$ = weighted sub-account return for month $m$
    - $P_m$ = scheduled premium for month $m$
    - $COI_m$ = cost-of-insurance charge
    - $M\&E_m$ = mortality & expense risk charge
    - $A_m$ = administrative / investment management fees

    Attributes
    ----------
    m_sub_account_allocations : dict[Sub_Account_Type, float]
        Allocation percentages across sub-accounts (summing to 1.0).
    m_cash_value : float
        Current total cash value across all sub-accounts.
    m_guaranteed_min_death_benefit : float
        Guaranteed minimum death benefit (equals initial face amount).
    m_rate_ME_charge : float
        Annual mortality & expense risk charge rate.
    m_rate_admin_fee : float
        Annual administrative fee rate.
    m_rate_investment_management : float
        Annual investment management fee rate.
    m_rate_surrender_charge : float
        Current surrender charge rate.
    m_surrender_charge_years : int
        Duration of surrender charge schedule.
    m_death_benefit_option : Death_Benefit_Option
        Level or increasing (variable) death benefit.
    m_amount_loan_outstanding : float
        Outstanding loan balance.
    m_rate_loan_interest : float
        Loan interest rate.
    m_rate_COI : float
        COI rate per $1 000 of net amount at risk.
    """

    def __init__(
        self,
        insured_age: int,
        face_amount: float,
        sub_account_allocations: dict[Sub_Account_Type, float] | None = None,
        cash_value: float = 0.0,
        rate_ME_charge: float = 0.009,
        rate_admin_fee: float = 0.0015,
        rate_investment_management: float = 0.005,
        rate_surrender_charge: float = 0.0,
        surrender_charge_years: int = 10,
        rate_COI: float = 3.5,
        death_benefit_option: Death_Benefit_Option = Death_Benefit_Option.LEVEL,
        amount_loan_outstanding: float = 0.0,
        rate_loan_interest: float = 0.06,
        underwriting_class: Underwriting_Class = Underwriting_Class.STANDARD,
        premium_frequency: Premium_Frequency = Premium_Frequency.ANNUAL,
        is_smoker: bool = False,
        beneficiary_primary: str = "",
        beneficiary_contingent: str = "",
    ) -> None:
        """Initialize a Variable Life Insurance object.

        Parameters
        ----------
        insured_age : int
            Current age of the insured.
        face_amount : float
            Face amount (also the guaranteed minimum death benefit).
        sub_account_allocations : dict[Sub_Account_Type, float] | None, optional
            Allocation percentages (must sum to 1.0).
            Defaults to 100 % balanced.
        cash_value : float, optional
            Current aggregate cash value (default 0.0).
        rate_ME_charge : float, optional
            Annual M&E risk charge rate (default 0.009 = 90 bps).
        rate_admin_fee : float, optional
            Annual administrative fee rate (default 0.0015 = 15 bps).
        rate_investment_management : float, optional
            Annual investment management fee (default 0.005 = 50 bps).
        rate_surrender_charge : float, optional
            Current surrender charge rate (default 0.0).
        surrender_charge_years : int, optional
            Surrender schedule duration (default 10).
        rate_COI : float, optional
            Monthly COI rate per $1 000 NAR (default 3.5).
        death_benefit_option : Death_Benefit_Option, optional
            Level or increasing death benefit (default ``LEVEL``).
        amount_loan_outstanding : float, optional
            Outstanding loan balance (default 0.0).
        rate_loan_interest : float, optional
            Loan interest rate (default 0.06).
        underwriting_class : Underwriting_Class, optional
            Risk classification (default ``STANDARD``).
        premium_frequency : Premium_Frequency, optional
            Premium payment frequency (default ``ANNUAL``).
        is_smoker : bool, optional
            Tobacco use (default ``False``).
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
            insurance_type=Insurance_Life_Type.VARIABLE_LIFE,
            underwriting_class=underwriting_class,
            premium_frequency=premium_frequency,
            is_smoker=is_smoker,
            beneficiary_primary=beneficiary_primary,
            beneficiary_contingent=beneficiary_contingent,
        )

        # --- Validate sub-account allocations ---
        if sub_account_allocations is None:
            sub_account_allocations = {Sub_Account_Type.BALANCED: 1.0}

        if not isinstance(sub_account_allocations, dict):
            raise Exception_Validation_Input(
                "sub_account_allocations must be a dict",
                field_name="sub_account_allocations",
                expected_type=dict,
                actual_value=sub_account_allocations,
            )

        for key, val in sub_account_allocations.items():
            if not isinstance(key, Sub_Account_Type):
                raise Exception_Validation_Input(
                    "sub_account_allocations keys must be Sub_Account_Type",
                    field_name="sub_account_allocations",
                    expected_type=Sub_Account_Type,
                    actual_value=key,
                )
            if not isinstance(val, (int, float)) or val < 0 or val > 1:
                raise Exception_Validation_Input(
                    "sub_account allocation values must be in [0, 1]",
                    field_name="sub_account_allocations",
                    expected_type=float,
                    actual_value=val,
                )

        alloc_total = sum(sub_account_allocations.values())
        if abs(alloc_total - 1.0) > 1e-6:
            raise Exception_Validation_Input(
                f"sub_account_allocations must sum to 1.0, got {alloc_total:.6f}",
                field_name="sub_account_allocations",
                expected_type=float,
                actual_value=alloc_total,
            )

        if not isinstance(cash_value, (int, float)) or cash_value < 0:
            raise Exception_Validation_Input(
                "cash_value must be a non-negative number",
                field_name="cash_value",
                expected_type=float,
                actual_value=cash_value,
            )

        if not isinstance(rate_ME_charge, (int, float)) or rate_ME_charge < 0:
            raise Exception_Validation_Input(
                "rate_ME_charge must be a non-negative number",
                field_name="rate_ME_charge",
                expected_type=float,
                actual_value=rate_ME_charge,
            )

        if not isinstance(rate_admin_fee, (int, float)) or rate_admin_fee < 0:
            raise Exception_Validation_Input(
                "rate_admin_fee must be a non-negative number",
                field_name="rate_admin_fee",
                expected_type=float,
                actual_value=rate_admin_fee,
            )

        if (
            not isinstance(rate_investment_management, (int, float))
            or rate_investment_management < 0
        ):
            raise Exception_Validation_Input(
                "rate_investment_management must be a non-negative number",
                field_name="rate_investment_management",
                expected_type=float,
                actual_value=rate_investment_management,
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

        if not isinstance(rate_COI, (int, float)) or rate_COI < 0:
            raise Exception_Validation_Input(
                "rate_COI must be a non-negative number",
                field_name="rate_COI",
                expected_type=float,
                actual_value=rate_COI,
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

        # --- Set member variables ---
        self.m_sub_account_allocations: dict[Sub_Account_Type, float] = sub_account_allocations
        self.m_cash_value: float = float(cash_value)
        self.m_guaranteed_min_death_benefit: float = float(face_amount)
        self.m_rate_ME_charge: float = float(rate_ME_charge)
        self.m_rate_admin_fee: float = float(rate_admin_fee)
        self.m_rate_investment_management: float = float(rate_investment_management)
        self.m_rate_surrender_charge: float = float(rate_surrender_charge)
        self.m_surrender_charge_years: int = surrender_charge_years
        self.m_rate_COI: float = float(rate_COI)
        self.m_death_benefit_option: Death_Benefit_Option = death_benefit_option
        self.m_amount_loan_outstanding: float = float(amount_loan_outstanding)
        self.m_rate_loan_interest: float = float(rate_loan_interest)

        logger.info(
            f"Created Variable Life: face ${face_amount:,.2f}, "
            f"{len(sub_account_allocations)} sub-account(s), "
            f"CV ${cash_value:,.2f}, "
            f"M&E {rate_ME_charge:.2%}",
        )

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def sub_account_allocations(self) -> dict[Sub_Account_Type, float]:
        """dict : Sub-account allocation percentages."""
        return self.m_sub_account_allocations

    @property
    def cash_value(self) -> float:
        """float : Current total cash value."""
        return self.m_cash_value

    @property
    def guaranteed_min_death_benefit(self) -> float:
        """float : Guaranteed minimum death benefit."""
        return self.m_guaranteed_min_death_benefit

    @property
    def rate_ME_charge(self) -> float:
        """float : Annual mortality & expense risk charge rate."""
        return self.m_rate_ME_charge

    @property
    def rate_admin_fee(self) -> float:
        """float : Annual administrative fee rate."""
        return self.m_rate_admin_fee

    @property
    def rate_investment_management(self) -> float:
        """float : Annual investment management fee rate."""
        return self.m_rate_investment_management

    @property
    def rate_surrender_charge(self) -> float:
        """float : Current surrender charge rate."""
        return self.m_rate_surrender_charge

    @property
    def surrender_charge_years(self) -> int:
        """int : Surrender charge schedule duration in years."""
        return self.m_surrender_charge_years

    @property
    def rate_COI(self) -> float:
        """float : Monthly COI rate per $1 000 NAR."""
        return self.m_rate_COI

    @property
    def death_benefit_option(self) -> Death_Benefit_Option:
        """Death_Benefit_Option : Death benefit option."""
        return self.m_death_benefit_option

    @property
    def amount_loan_outstanding(self) -> float:
        """float : Outstanding loan balance."""
        return self.m_amount_loan_outstanding

    @property
    def rate_loan_interest(self) -> float:
        """float : Loan interest rate."""
        return self.m_rate_loan_interest

    # ------------------------------------------------------------------
    # Annual fee calculations
    # ------------------------------------------------------------------

    def calc_total_annual_fees(self) -> float:
        r"""Calculate total annual fees charged against the cash value.

        $$
        F_{\text{total}} = CV \times (r_{M\&E} + r_{\text{admin}} + r_{\text{inv}})
        $$

        Returns
        -------
        float
            Total annual fees in dollars.
        """
        total_rate = (
            self.m_rate_ME_charge + self.m_rate_admin_fee + self.m_rate_investment_management
        )
        return self.m_cash_value * total_rate

    def calc_monthly_fees(self) -> float:
        """Calculate monthly fees charged against the cash value.

        Returns
        -------
        float
            Monthly fees in dollars.
        """
        return self.calc_total_annual_fees() / 12

    # ------------------------------------------------------------------
    # Sub-account returns
    # ------------------------------------------------------------------

    def calc_weighted_return(
        self,
        sub_account_returns: dict[Sub_Account_Type, float],
    ) -> float:
        r"""Calculate the weighted portfolio return across sub-accounts.

        $$
        r_w = \sum_{i} w_i \times r_i
        $$

        Parameters
        ----------
        sub_account_returns : dict[Sub_Account_Type, float]
            Return for each sub-account (keyed by Sub_Account_Type).

        Returns
        -------
        float
            The weighted average return.

        Raises
        ------
        Exception_Validation_Input
            If ``sub_account_returns`` is not a dict.
        """
        if not isinstance(sub_account_returns, dict):
            raise Exception_Validation_Input(
                "sub_account_returns must be a dict",
                field_name="sub_account_returns",
                expected_type=dict,
                actual_value=sub_account_returns,
            )

        weighted_return: float = 0.0
        for acct_type, allocation in self.m_sub_account_allocations.items():
            acct_return = sub_account_returns.get(acct_type, 0.0)
            weighted_return += allocation * acct_return

        return weighted_return

    # ------------------------------------------------------------------
    # Death benefit calculations
    # ------------------------------------------------------------------

    def calc_death_benefit(self) -> float:
        r"""Calculate the current death benefit payable.

        **Option Level**: death benefit = max(face amount, cash value +
        corridor factor).  Variable Life guarantees the death benefit
        never falls below the initial face amount while premiums are
        paid.

        **Option Increasing**: death benefit = face amount + cash value.

        Both are reduced by the outstanding loan balance.

        $$
        DB = \max\!\bigl(G,\; DB_{\text{option}}\bigr) - L
        $$

        Returns
        -------
        float
            Net death benefit (floored at 0).
        """
        if self.m_death_benefit_option == Death_Benefit_Option.INCREASING:
            option_db = self.m_face_amount + self.m_cash_value
        else:
            option_db = max(self.m_face_amount, self.m_cash_value)

        gross_db = max(option_db, self.m_guaranteed_min_death_benefit)
        return max(gross_db - self.m_amount_loan_outstanding, 0.0)

    # ------------------------------------------------------------------
    # Premium calculations
    # ------------------------------------------------------------------

    def calc_annual_premium(self) -> float:
        r"""Calculate the annual premium for Variable Life.

        Variable life has fixed scheduled premiums (unlike VUL).
        Estimated as COI-based premium with fee loading:

        $$
        P = \frac{F}{1\,000} \times COI \times 12
            \times (1 + r_{M\&E} + r_{\text{admin}} + r_{\text{inv}} + e)
        $$

        Where $e$ is an expense loading factor (20 %).

        Returns
        -------
        float
            Estimated annual premium.
        """
        expense_loading: float = 0.20
        fee_rate = self.m_rate_ME_charge + self.m_rate_admin_fee + self.m_rate_investment_management
        annual_coi = (self.m_face_amount / 1_000) * self.m_rate_COI * 12
        return annual_coi * (1 + fee_rate + expense_loading)

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

    def calc_cash_value_at_month(
        self,
        months: int,
        sub_account_monthly_return: float = 0.005,
    ) -> float:
        r"""Project cash value at the end of a number of months.

        Assumes a constant weighted monthly sub-account return and
        deducts COI, M&E, admin, and investment management fees monthly.

        $$
        CV_m = CV_{m-1} \times (1 + r_s) + P_m - COI_m - Fees_m
        $$

        Parameters
        ----------
        months : int
            Number of months to project.
        sub_account_monthly_return : float, optional
            Monthly return across sub-accounts (default 0.005 = 0.5 %).

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

        annual_premium = self.calc_annual_premium()
        monthly_premium = annual_premium / 12

        monthly_fee_rate = (
            self.m_rate_ME_charge + self.m_rate_admin_fee + self.m_rate_investment_management
        ) / 12

        cv: float = self.m_cash_value
        for _ in range(months):
            # Investment return
            cv = cv * (1 + sub_account_monthly_return)
            # Add premium
            cv += monthly_premium
            # Deduct COI
            nar = max(self.m_face_amount - cv, 0.0)
            coi_charge = (nar / 1_000) * self.m_rate_COI
            cv -= coi_charge
            # Deduct fees
            cv -= cv * monthly_fee_rate

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
    # String representation
    # ------------------------------------------------------------------

    def get_insurance_as_string(self) -> str:
        """Return a human-readable string representation.

        Returns
        -------
        str
            String representation of the Variable Life policy.
        """
        acct_count = len(self.m_sub_account_allocations)
        opt = "B" if self.m_death_benefit_option == Death_Benefit_Option.INCREASING else "A"
        db_text = f"Option {opt}"
        return (
            f"Variable Life ({db_text}, {acct_count} sub-account(s)): "
            f"Age {self.m_insured_age}, "
            f"Face ${self.m_face_amount:,.2f}, "
            f"CV ${self.m_cash_value:,.2f}, "
            f"Guaranteed Min DB ${self.m_guaranteed_min_death_benefit:,.2f}, "
            f"M&E {self.m_rate_ME_charge:.2%}, "
            f"Class: {self.m_underwriting_class.value}"
        )
