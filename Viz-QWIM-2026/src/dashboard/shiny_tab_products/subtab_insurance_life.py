"""Insurance Life Subtab Module for QWIM Dashboard.

Provides input forms for the five life insurance product types:
Whole Life, Term Life, Universal Life, Variable Life, and
Survivor Life.

Each life insurance type is rendered in its own sub-subtab so users can enter
parameters independently.  All input identifiers follow the hierarchical
naming convention:

    ``input_ID_tab_products_subtab_insurance_life_<type>_<field>``

Author
------
QWIM Team

Version
-------
0.7.0 (2026-03-01)
"""

from __future__ import annotations

import typing

from typing import Any

from shiny import module, reactive, ui

from src.dashboard.shiny_utils.utils_enhanced_ui_components import (
    create_enhanced_card_section,
    create_enhanced_numeric_input,
    create_enhanced_select_input,
)
from src.utils.custom_exceptions_errors_loggers.logger_custom import get_logger


#: Module-level logger instance
_logger = get_logger(__name__)


# ======================================================================
# Shared choice dictionaries
# ======================================================================

_UNDERWRITING_CLASS_CHOICES: dict[str, str] = {
    "preferred_plus": "Preferred Plus",
    "preferred": "Preferred",
    "standard": "Standard",
    "substandard": "Substandard",
}

_DEATH_BENEFIT_OPTION_CHOICES: dict[str, str] = {
    "level": "Level",
    "increasing": "Increasing",
    "return_of_premium": "Return of Premium",
}

_PAYMENT_FREQUENCY_CHOICES: dict[str, str] = {
    "1": "Annual",
    "2": "Semi-Annual",
    "4": "Quarterly",
    "12": "Monthly",
}

_TERM_TYPE_CHOICES: dict[str, str] = {
    "level": "Level Term",
    "annually_renewable": "Annually Renewable Term",
    "decreasing": "Decreasing Term",
}

_UL_VARIANT_CHOICES: dict[str, str] = {
    "traditional": "Traditional UL",
    "indexed": "Indexed UL",
    "guaranteed": "Guaranteed UL",
}

_SURVIVOR_CHASSIS_CHOICES: dict[str, str] = {
    "whole_life": "Whole Life",
    "universal_life": "Universal Life",
    "variable_universal_life": "Variable Universal Life",
}


# ======================================================================
# Whole Life sub-subtab UI helper
# ======================================================================


def _create_whole_life_ui() -> Any:
    """Build the input form for a Whole Life Insurance policy.

    Returns
    -------
    ui.div
        Complete Whole Life input card.
    """
    _pfx = "input_ID_tab_products_subtab_insurance_life_whole_life"
    return create_enhanced_card_section(
        title="Whole Life Parameters",
        content=[
            # --- Core parameters ---
            create_enhanced_numeric_input(
                input_ID=f"{_pfx}_insured_age",
                label_text="Insured Age",
                min_value=0,
                max_value=120,
                step_size=1,
                default_value=40,
                suffix_symbol=" years",
                tooltip_text="Current age of the insured",
                required_field=True,
                input_width="100%",
            ),
            create_enhanced_numeric_input(
                input_ID=f"{_pfx}_face_amount",
                label_text="Face Amount",
                min_value=1_000,
                max_value=50_000_000,
                step_size=10_000,
                default_value=500_000,
                suffix_symbol=" $",
                tooltip_text="Death benefit face amount",
                required_field=True,
                input_width="100%",
            ),
            create_enhanced_numeric_input(
                input_ID=f"{_pfx}_rate_guaranteed_interest",
                label_text="Guaranteed Interest Rate",
                min_value=0.00,
                max_value=0.10,
                step_size=0.005,
                default_value=0.04,
                tooltip_text=("Guaranteed annual crediting rate on cash value (e.g. 0.04 = 4 %)"),
                input_width="100%",
            ),
            create_enhanced_numeric_input(
                input_ID=f"{_pfx}_cash_value",
                label_text="Current Cash Value",
                min_value=0,
                max_value=50_000_000,
                step_size=1_000,
                default_value=0,
                suffix_symbol=" $",
                tooltip_text="Current accumulated cash value",
                input_width="100%",
            ),
            # --- Dividends ---
            create_enhanced_select_input(
                input_ID=f"{_pfx}_is_participating",
                label_text="Participating (Dividends)",
                choices={"true": "Yes", "false": "No"},
                default_selection="true",
                tooltip_text="Whether the policy pays dividends",
            ),
            create_enhanced_numeric_input(
                input_ID=f"{_pfx}_rate_dividend",
                label_text="Dividend Rate",
                min_value=0.00,
                max_value=0.10,
                step_size=0.005,
                default_value=0.00,
                tooltip_text="Annual dividend rate (e.g. 0.02 = 2 %)",
                input_width="100%",
            ),
            create_enhanced_numeric_input(
                input_ID=f"{_pfx}_paid_up_additions",
                label_text="Paid-Up Additions",
                min_value=0,
                max_value=50_000_000,
                step_size=1_000,
                default_value=0,
                suffix_symbol=" $",
                tooltip_text="Total paid-up additions face amount",
                input_width="100%",
            ),
            # --- Loans ---
            create_enhanced_numeric_input(
                input_ID=f"{_pfx}_rate_loan_interest",
                label_text="Loan Interest Rate",
                min_value=0.00,
                max_value=0.12,
                step_size=0.005,
                default_value=0.05,
                tooltip_text="Policy loan interest rate (e.g. 0.05 = 5 %)",
                input_width="100%",
            ),
            create_enhanced_numeric_input(
                input_ID=f"{_pfx}_amount_loan_outstanding",
                label_text="Outstanding Loan",
                min_value=0,
                max_value=50_000_000,
                step_size=1_000,
                default_value=0,
                suffix_symbol=" $",
                tooltip_text="Current outstanding loan balance",
                input_width="100%",
            ),
            # --- Premium / COI ---
            create_enhanced_numeric_input(
                input_ID=f"{_pfx}_premium_paying_years",
                label_text="Premium-Paying Years",
                min_value=1,
                max_value=100,
                step_size=1,
                default_value=100,
                suffix_symbol=" years",
                tooltip_text=(
                    "Number of years premiums are due (100 = whole life, 20 = 20-pay, etc.)"
                ),
                input_width="100%",
            ),
            create_enhanced_numeric_input(
                input_ID=f"{_pfx}_rate_cost_of_insurance",
                label_text="COI Rate (per $1 000)",
                min_value=0.0,
                max_value=50.0,
                step_size=0.5,
                default_value=5.0,
                tooltip_text=("Annual cost-of-insurance rate per $1 000 of coverage"),
                input_width="100%",
            ),
            create_enhanced_select_input(
                input_ID=f"{_pfx}_death_benefit_option",
                label_text="Death Benefit Option",
                choices=_DEATH_BENEFIT_OPTION_CHOICES,
                default_selection="level",
                tooltip_text="Level or increasing death benefit",
            ),
            # --- Underwriting ---
            create_enhanced_select_input(
                input_ID=f"{_pfx}_underwriting_class",
                label_text="Underwriting Class",
                choices=_UNDERWRITING_CLASS_CHOICES,
                default_selection="standard",
                tooltip_text="Risk classification of the insured",
            ),
            create_enhanced_select_input(
                input_ID=f"{_pfx}_is_smoker",
                label_text="Tobacco Use",
                choices={"false": "Non-Smoker", "true": "Smoker"},
                default_selection="false",
                tooltip_text="Whether the insured uses tobacco",
            ),
            create_enhanced_select_input(
                input_ID=f"{_pfx}_payment_frequency",
                label_text="Payment Frequency",
                choices=_PAYMENT_FREQUENCY_CHOICES,
                default_selection="12",
                tooltip_text="Number of premium payments per year",
            ),
        ],
        icon_class="fas fa-shield-alt",
        card_class="shadow-sm border-primary",
    )


# ======================================================================
# Term Life sub-subtab UI helper
# ======================================================================


def _create_term_life_ui() -> Any:
    """Build the input form for a Term Life Insurance policy.

    Returns
    -------
    ui.div
        Complete Term Life input card.
    """
    _pfx = "input_ID_tab_products_subtab_insurance_life_term_life"
    return create_enhanced_card_section(
        title="Term Life Parameters",
        content=[
            # --- Core parameters ---
            create_enhanced_numeric_input(
                input_ID=f"{_pfx}_insured_age",
                label_text="Insured Age",
                min_value=0,
                max_value=120,
                step_size=1,
                default_value=35,
                suffix_symbol=" years",
                tooltip_text="Current age of the insured",
                required_field=True,
                input_width="100%",
            ),
            create_enhanced_numeric_input(
                input_ID=f"{_pfx}_face_amount",
                label_text="Face Amount",
                min_value=1_000,
                max_value=50_000_000,
                step_size=10_000,
                default_value=1_000_000,
                suffix_symbol=" $",
                tooltip_text="Death benefit face amount",
                required_field=True,
                input_width="100%",
            ),
            create_enhanced_numeric_input(
                input_ID=f"{_pfx}_term_years",
                label_text="Term Length",
                min_value=1,
                max_value=40,
                step_size=1,
                default_value=20,
                suffix_symbol=" years",
                tooltip_text="Duration of coverage in years",
                required_field=True,
                input_width="100%",
            ),
            create_enhanced_select_input(
                input_ID=f"{_pfx}_term_type",
                label_text="Term Type",
                choices=_TERM_TYPE_CHOICES,
                default_selection="level",
                tooltip_text="Term premium / benefit structure",
                required_field=True,
            ),
            # --- Features ---
            create_enhanced_select_input(
                input_ID=f"{_pfx}_is_convertible",
                label_text="Convertible",
                choices={"true": "Yes", "false": "No"},
                default_selection="true",
                tooltip_text=("Whether the policy can be converted to permanent insurance"),
            ),
            create_enhanced_numeric_input(
                input_ID=f"{_pfx}_conversion_deadline_year",
                label_text="Conversion Deadline Year",
                min_value=0,
                max_value=40,
                step_size=1,
                default_value=0,
                suffix_symbol=" year",
                tooltip_text=(
                    "Last policy year conversion is available (0 = no limit / same as term)"
                ),
                input_width="100%",
            ),
            create_enhanced_select_input(
                input_ID=f"{_pfx}_is_renewable",
                label_text="Renewable",
                choices={"true": "Yes", "false": "No"},
                default_selection="true",
                tooltip_text=("Guaranteed renewable at term expiry (at higher premium)"),
            ),
            create_enhanced_select_input(
                input_ID=f"{_pfx}_has_return_of_premium",
                label_text="Return-of-Premium Rider",
                choices={"false": "No", "true": "Yes"},
                default_selection="false",
                tooltip_text=("Whether premiums are refunded if the insured survives the term"),
            ),
            # --- COI / Underwriting ---
            create_enhanced_numeric_input(
                input_ID=f"{_pfx}_rate_cost_of_insurance",
                label_text="COI Rate (per $1 000)",
                min_value=0.0,
                max_value=50.0,
                step_size=0.5,
                default_value=2.5,
                tooltip_text=("Average annual cost-of-insurance rate per $1 000"),
                input_width="100%",
            ),
            create_enhanced_select_input(
                input_ID=f"{_pfx}_underwriting_class",
                label_text="Underwriting Class",
                choices=_UNDERWRITING_CLASS_CHOICES,
                default_selection="standard",
                tooltip_text="Risk classification of the insured",
            ),
            create_enhanced_select_input(
                input_ID=f"{_pfx}_is_smoker",
                label_text="Tobacco Use",
                choices={"false": "Non-Smoker", "true": "Smoker"},
                default_selection="false",
                tooltip_text="Whether the insured uses tobacco",
            ),
            create_enhanced_select_input(
                input_ID=f"{_pfx}_payment_frequency",
                label_text="Payment Frequency",
                choices=_PAYMENT_FREQUENCY_CHOICES,
                default_selection="12",
                tooltip_text="Number of premium payments per year",
            ),
        ],
        icon_class="fas fa-hourglass-half",
        card_class="shadow-sm border-info",
    )


# ======================================================================
# Universal Life sub-subtab UI helper
# ======================================================================


def _create_universal_life_ui() -> Any:
    """Build the input form for a Universal Life Insurance policy.

    Returns
    -------
    ui.div
        Complete Universal Life input card.
    """
    _pfx = "input_ID_tab_products_subtab_insurance_life_universal_life"
    return create_enhanced_card_section(
        title="Universal Life Parameters",
        content=[
            # --- Core parameters ---
            create_enhanced_numeric_input(
                input_ID=f"{_pfx}_insured_age",
                label_text="Insured Age",
                min_value=0,
                max_value=120,
                step_size=1,
                default_value=40,
                suffix_symbol=" years",
                tooltip_text="Current age of the insured",
                required_field=True,
                input_width="100%",
            ),
            create_enhanced_numeric_input(
                input_ID=f"{_pfx}_face_amount",
                label_text="Face Amount",
                min_value=1_000,
                max_value=50_000_000,
                step_size=10_000,
                default_value=500_000,
                suffix_symbol=" $",
                tooltip_text="Death benefit face amount",
                required_field=True,
                input_width="100%",
            ),
            create_enhanced_select_input(
                input_ID=f"{_pfx}_ul_variant",
                label_text="UL Variant",
                choices=_UL_VARIANT_CHOICES,
                default_selection="traditional",
                tooltip_text=("Traditional, Indexed (IUL), or Guaranteed (GUL) variant"),
                required_field=True,
            ),
            # --- Crediting rates ---
            create_enhanced_numeric_input(
                input_ID=f"{_pfx}_rate_crediting_current",
                label_text="Current Crediting Rate",
                min_value=0.00,
                max_value=0.15,
                step_size=0.005,
                default_value=0.045,
                tooltip_text=("Current declared annual crediting rate (e.g. 0.045 = 4.5 %)"),
                input_width="100%",
            ),
            create_enhanced_numeric_input(
                input_ID=f"{_pfx}_rate_crediting_guaranteed",
                label_text="Guaranteed Crediting Rate",
                min_value=0.00,
                max_value=0.10,
                step_size=0.005,
                default_value=0.02,
                tooltip_text=("Guaranteed minimum annual crediting rate (e.g. 0.02 = 2 %)"),
                input_width="100%",
            ),
            # --- IUL parameters ---
            create_enhanced_numeric_input(
                input_ID=f"{_pfx}_rate_cap",
                label_text="Cap Rate (IUL)",
                min_value=0.00,
                max_value=0.30,
                step_size=0.005,
                default_value=0.00,
                tooltip_text=(
                    "Cap rate for Indexed UL (e.g. 0.10 = 10 %). Set to 0 for Traditional UL."
                ),
                input_width="100%",
            ),
            create_enhanced_numeric_input(
                input_ID=f"{_pfx}_rate_floor",
                label_text="Floor Rate (IUL)",
                min_value=-0.05,
                max_value=0.05,
                step_size=0.005,
                default_value=0.00,
                tooltip_text=(
                    "Floor rate for Indexed UL (typically 0 %). Set to 0 for Traditional UL."
                ),
                input_width="100%",
            ),
            create_enhanced_numeric_input(
                input_ID=f"{_pfx}_participation_rate",
                label_text="Participation Rate (IUL)",
                min_value=0.00,
                max_value=3.00,
                step_size=0.05,
                default_value=1.00,
                tooltip_text=("Participation rate for Indexed UL (e.g. 1.0 = 100 %)"),
                input_width="100%",
            ),
            # --- Cash value & charges ---
            create_enhanced_numeric_input(
                input_ID=f"{_pfx}_cash_value",
                label_text="Current Cash Value",
                min_value=0,
                max_value=50_000_000,
                step_size=1_000,
                default_value=0,
                suffix_symbol=" $",
                tooltip_text="Current cash value",
                input_width="100%",
            ),
            create_enhanced_numeric_input(
                input_ID=f"{_pfx}_rate_COI",
                label_text="COI Rate (per $1 000)",
                min_value=0.0,
                max_value=50.0,
                step_size=0.5,
                default_value=3.0,
                tooltip_text=("Monthly COI rate per $1 000 of net amount at risk"),
                input_width="100%",
            ),
            create_enhanced_numeric_input(
                input_ID=f"{_pfx}_rate_expense_charge",
                label_text="Expense Charge Rate",
                min_value=0.00,
                max_value=0.05,
                step_size=0.001,
                default_value=0.005,
                tooltip_text=("Monthly expense / administrative charge rate (e.g. 0.005 = 0.5 %)"),
                input_width="100%",
            ),
            # --- Premiums ---
            create_enhanced_numeric_input(
                input_ID=f"{_pfx}_target_premium",
                label_text="Target Premium",
                min_value=0,
                max_value=1_000_000,
                step_size=500,
                default_value=0,
                suffix_symbol=" $",
                tooltip_text=("Target annual premium (0 = auto-calculate from COI)"),
                input_width="100%",
            ),
            create_enhanced_numeric_input(
                input_ID=f"{_pfx}_minimum_premium",
                label_text="Minimum Premium",
                min_value=0,
                max_value=500_000,
                step_size=500,
                default_value=0,
                suffix_symbol=" $",
                tooltip_text=("Minimum annual premium to keep the policy in force"),
                input_width="100%",
            ),
            # --- Surrender ---
            create_enhanced_numeric_input(
                input_ID=f"{_pfx}_rate_surrender_charge",
                label_text="Surrender Charge Rate",
                min_value=0.00,
                max_value=0.20,
                step_size=0.01,
                default_value=0.00,
                tooltip_text=("Current surrender charge rate (e.g. 0.08 = 8 %)"),
                input_width="100%",
            ),
            create_enhanced_numeric_input(
                input_ID=f"{_pfx}_surrender_charge_years",
                label_text="Surrender Schedule Years",
                min_value=0,
                max_value=20,
                step_size=1,
                default_value=10,
                suffix_symbol=" years",
                tooltip_text=("Number of years the surrender charge applies"),
                input_width="100%",
            ),
            # --- NLG ---
            create_enhanced_select_input(
                input_ID=f"{_pfx}_has_no_lapse_guarantee",
                label_text="No-Lapse Guarantee",
                choices={"false": "No", "true": "Yes"},
                default_selection="false",
                tooltip_text=("Whether the policy includes a no-lapse guarantee (NLG)"),
            ),
            create_enhanced_numeric_input(
                input_ID=f"{_pfx}_nlg_guarantee_years",
                label_text="NLG Guarantee Years",
                min_value=0,
                max_value=50,
                step_size=1,
                default_value=0,
                suffix_symbol=" years",
                tooltip_text="Number of years the NLG is guaranteed",
                input_width="100%",
            ),
            # --- Death benefit & loans ---
            create_enhanced_select_input(
                input_ID=f"{_pfx}_death_benefit_option",
                label_text="Death Benefit Option",
                choices={
                    "level": "Option A (Level)",
                    "increasing": "Option B (Increasing)",
                },
                default_selection="level",
                tooltip_text="Level (Option A) or Increasing (Option B)",
            ),
            create_enhanced_numeric_input(
                input_ID=f"{_pfx}_amount_loan_outstanding",
                label_text="Outstanding Loan",
                min_value=0,
                max_value=50_000_000,
                step_size=1_000,
                default_value=0,
                suffix_symbol=" $",
                tooltip_text="Current outstanding loan balance",
                input_width="100%",
            ),
            create_enhanced_numeric_input(
                input_ID=f"{_pfx}_rate_loan_interest",
                label_text="Loan Interest Rate",
                min_value=0.00,
                max_value=0.12,
                step_size=0.005,
                default_value=0.05,
                tooltip_text=("Policy loan interest rate (e.g. 0.05 = 5 %)"),
                input_width="100%",
            ),
            # --- Underwriting ---
            create_enhanced_select_input(
                input_ID=f"{_pfx}_underwriting_class",
                label_text="Underwriting Class",
                choices=_UNDERWRITING_CLASS_CHOICES,
                default_selection="standard",
                tooltip_text="Risk classification of the insured",
            ),
            create_enhanced_select_input(
                input_ID=f"{_pfx}_is_smoker",
                label_text="Tobacco Use",
                choices={"false": "Non-Smoker", "true": "Smoker"},
                default_selection="false",
                tooltip_text="Whether the insured uses tobacco",
            ),
            create_enhanced_select_input(
                input_ID=f"{_pfx}_payment_frequency",
                label_text="Payment Frequency",
                choices=_PAYMENT_FREQUENCY_CHOICES,
                default_selection="12",
                tooltip_text="Number of premium payments per year",
            ),
        ],
        icon_class="fas fa-sliders-h",
        card_class="shadow-sm border-success",
    )


# ======================================================================
# Variable Life sub-subtab UI helper
# ======================================================================


def _create_variable_life_ui() -> Any:
    """Build the input form for a Variable Life Insurance policy.

    Returns
    -------
    ui.div
        Complete Variable Life input card.
    """
    _pfx = "input_ID_tab_products_subtab_insurance_life_variable_life"
    return create_enhanced_card_section(
        title="Variable Life Parameters",
        content=[
            # --- Core parameters ---
            create_enhanced_numeric_input(
                input_ID=f"{_pfx}_insured_age",
                label_text="Insured Age",
                min_value=0,
                max_value=120,
                step_size=1,
                default_value=40,
                suffix_symbol=" years",
                tooltip_text="Current age of the insured",
                required_field=True,
                input_width="100%",
            ),
            create_enhanced_numeric_input(
                input_ID=f"{_pfx}_face_amount",
                label_text="Face Amount",
                min_value=1_000,
                max_value=50_000_000,
                step_size=10_000,
                default_value=500_000,
                suffix_symbol=" $",
                tooltip_text=(
                    "Death benefit face amount (also the guaranteed minimum death benefit)"
                ),
                required_field=True,
                input_width="100%",
            ),
            create_enhanced_numeric_input(
                input_ID=f"{_pfx}_cash_value",
                label_text="Current Cash Value",
                min_value=0,
                max_value=50_000_000,
                step_size=1_000,
                default_value=0,
                suffix_symbol=" $",
                tooltip_text=("Current aggregate cash value across sub-accounts"),
                input_width="100%",
            ),
            # --- Fees & charges ---
            create_enhanced_numeric_input(
                input_ID=f"{_pfx}_rate_ME_charge",
                label_text="M&E Charge",
                min_value=0.00,
                max_value=0.05,
                step_size=0.001,
                default_value=0.009,
                tooltip_text=("Annual mortality & expense risk charge (e.g. 0.009 = 90 bps)"),
                input_width="100%",
            ),
            create_enhanced_numeric_input(
                input_ID=f"{_pfx}_rate_admin_fee",
                label_text="Admin Fee",
                min_value=0.00,
                max_value=0.02,
                step_size=0.0005,
                default_value=0.0015,
                tooltip_text=("Annual administrative fee (e.g. 0.0015 = 15 bps)"),
                input_width="100%",
            ),
            create_enhanced_numeric_input(
                input_ID=f"{_pfx}_rate_investment_management",
                label_text="Investment Management Fee",
                min_value=0.00,
                max_value=0.03,
                step_size=0.001,
                default_value=0.005,
                tooltip_text=("Annual investment management fee (e.g. 0.005 = 50 bps)"),
                input_width="100%",
            ),
            create_enhanced_numeric_input(
                input_ID=f"{_pfx}_rate_COI",
                label_text="COI Rate (per $1 000)",
                min_value=0.0,
                max_value=50.0,
                step_size=0.5,
                default_value=3.5,
                tooltip_text=("Monthly COI rate per $1 000 of net amount at risk"),
                input_width="100%",
            ),
            # --- Surrender ---
            create_enhanced_numeric_input(
                input_ID=f"{_pfx}_rate_surrender_charge",
                label_text="Surrender Charge Rate",
                min_value=0.00,
                max_value=0.20,
                step_size=0.01,
                default_value=0.00,
                tooltip_text=("Current surrender charge rate (e.g. 0.08 = 8 %)"),
                input_width="100%",
            ),
            create_enhanced_numeric_input(
                input_ID=f"{_pfx}_surrender_charge_years",
                label_text="Surrender Schedule Years",
                min_value=0,
                max_value=20,
                step_size=1,
                default_value=10,
                suffix_symbol=" years",
                tooltip_text=("Number of years the surrender charge applies"),
                input_width="100%",
            ),
            # --- Death benefit & loans ---
            create_enhanced_select_input(
                input_ID=f"{_pfx}_death_benefit_option",
                label_text="Death Benefit Option",
                choices={
                    "level": "Option A (Level)",
                    "increasing": "Option B (Increasing)",
                },
                default_selection="level",
                tooltip_text="Level (Option A) or Increasing (Option B)",
            ),
            create_enhanced_numeric_input(
                input_ID=f"{_pfx}_amount_loan_outstanding",
                label_text="Outstanding Loan",
                min_value=0,
                max_value=50_000_000,
                step_size=1_000,
                default_value=0,
                suffix_symbol=" $",
                tooltip_text="Current outstanding loan balance",
                input_width="100%",
            ),
            create_enhanced_numeric_input(
                input_ID=f"{_pfx}_rate_loan_interest",
                label_text="Loan Interest Rate",
                min_value=0.00,
                max_value=0.12,
                step_size=0.005,
                default_value=0.06,
                tooltip_text=("Policy loan interest rate (e.g. 0.06 = 6 %)"),
                input_width="100%",
            ),
            # --- Underwriting ---
            create_enhanced_select_input(
                input_ID=f"{_pfx}_underwriting_class",
                label_text="Underwriting Class",
                choices=_UNDERWRITING_CLASS_CHOICES,
                default_selection="standard",
                tooltip_text="Risk classification of the insured",
            ),
            create_enhanced_select_input(
                input_ID=f"{_pfx}_is_smoker",
                label_text="Tobacco Use",
                choices={"false": "Non-Smoker", "true": "Smoker"},
                default_selection="false",
                tooltip_text="Whether the insured uses tobacco",
            ),
            create_enhanced_select_input(
                input_ID=f"{_pfx}_payment_frequency",
                label_text="Payment Frequency",
                choices=_PAYMENT_FREQUENCY_CHOICES,
                default_selection="1",
                tooltip_text="Number of premium payments per year",
            ),
        ],
        icon_class="fas fa-chart-bar",
        card_class="shadow-sm border-warning",
    )


# ======================================================================
# Survivor Life sub-subtab UI helper
# ======================================================================


def _create_survivor_life_ui() -> Any:
    """Build the input form for a Survivor (Second-to-Die) Life policy.

    Returns
    -------
    ui.div
        Complete Survivor Life input card.
    """
    _pfx = "input_ID_tab_products_subtab_insurance_life_survivor_life"
    return create_enhanced_card_section(
        title="Survivor Life Parameters",
        content=[
            # --- Insured 1 ---
            create_enhanced_numeric_input(
                input_ID=f"{_pfx}_insured_age",
                label_text="Insured 1 — Age",
                min_value=0,
                max_value=120,
                step_size=1,
                default_value=55,
                suffix_symbol=" years",
                tooltip_text="Current age of the first (primary) insured",
                required_field=True,
                input_width="100%",
            ),
            create_enhanced_select_input(
                input_ID=f"{_pfx}_underwriting_class",
                label_text="Insured 1 — Underwriting Class",
                choices=_UNDERWRITING_CLASS_CHOICES,
                default_selection="standard",
                tooltip_text="Risk classification for insured 1",
            ),
            create_enhanced_select_input(
                input_ID=f"{_pfx}_is_smoker",
                label_text="Insured 1 — Tobacco Use",
                choices={"false": "Non-Smoker", "true": "Smoker"},
                default_selection="false",
                tooltip_text="Whether insured 1 uses tobacco",
            ),
            # --- Insured 2 ---
            create_enhanced_numeric_input(
                input_ID=f"{_pfx}_insured_age_second",
                label_text="Insured 2 — Age",
                min_value=0,
                max_value=120,
                step_size=1,
                default_value=53,
                suffix_symbol=" years",
                tooltip_text="Current age of the second insured",
                required_field=True,
                input_width="100%",
            ),
            create_enhanced_select_input(
                input_ID=f"{_pfx}_underwriting_class_second",
                label_text="Insured 2 — Underwriting Class",
                choices=_UNDERWRITING_CLASS_CHOICES,
                default_selection="standard",
                tooltip_text="Risk classification for insured 2",
            ),
            create_enhanced_select_input(
                input_ID=f"{_pfx}_is_smoker_second",
                label_text="Insured 2 — Tobacco Use",
                choices={"false": "Non-Smoker", "true": "Smoker"},
                default_selection="false",
                tooltip_text="Whether insured 2 uses tobacco",
            ),
            # --- Policy parameters ---
            create_enhanced_numeric_input(
                input_ID=f"{_pfx}_face_amount",
                label_text="Face Amount",
                min_value=1_000,
                max_value=50_000_000,
                step_size=10_000,
                default_value=1_000_000,
                suffix_symbol=" $",
                tooltip_text="Death benefit face amount",
                required_field=True,
                input_width="100%",
            ),
            create_enhanced_select_input(
                input_ID=f"{_pfx}_chassis",
                label_text="Policy Chassis",
                choices=_SURVIVOR_CHASSIS_CHOICES,
                default_selection="whole_life",
                tooltip_text="Underlying policy structure",
            ),
            create_enhanced_numeric_input(
                input_ID=f"{_pfx}_cash_value",
                label_text="Current Cash Value",
                min_value=0,
                max_value=50_000_000,
                step_size=1_000,
                default_value=0,
                suffix_symbol=" $",
                tooltip_text="Current cash value",
                input_width="100%",
            ),
            create_enhanced_numeric_input(
                input_ID=f"{_pfx}_rate_guaranteed_interest",
                label_text="Guaranteed Interest Rate",
                min_value=0.00,
                max_value=0.10,
                step_size=0.005,
                default_value=0.03,
                tooltip_text=("Guaranteed annual interest rate on cash value (e.g. 0.03 = 3 %)"),
                input_width="100%",
            ),
            # --- COI (joint) ---
            create_enhanced_numeric_input(
                input_ID=f"{_pfx}_rate_COI_insured_1",
                label_text="COI Rate — Insured 1",
                min_value=0.0,
                max_value=50.0,
                step_size=0.5,
                default_value=4.0,
                tooltip_text=("Monthly COI rate per $1 000 for insured 1"),
                input_width="100%",
            ),
            create_enhanced_numeric_input(
                input_ID=f"{_pfx}_rate_COI_insured_2",
                label_text="COI Rate — Insured 2",
                min_value=0.0,
                max_value=50.0,
                step_size=0.5,
                default_value=4.0,
                tooltip_text=("Monthly COI rate per $1 000 for insured 2"),
                input_width="100%",
            ),
            # --- Death benefit & features ---
            create_enhanced_select_input(
                input_ID=f"{_pfx}_death_benefit_option",
                label_text="Death Benefit Option",
                choices=_DEATH_BENEFIT_OPTION_CHOICES,
                default_selection="level",
                tooltip_text="Level or increasing death benefit",
            ),
            create_enhanced_select_input(
                input_ID=f"{_pfx}_has_split_option",
                label_text="Split-Option Rider",
                choices={"false": "No", "true": "Yes"},
                default_selection="false",
                tooltip_text=(
                    "Whether the policy can split into two individual "
                    "policies upon a qualifying event (e.g. divorce)"
                ),
            ),
            create_enhanced_select_input(
                input_ID=f"{_pfx}_first_death_occurred",
                label_text="First Death Occurred",
                choices={"false": "No", "true": "Yes"},
                default_selection="false",
                tooltip_text=("Whether the first insured has already passed away"),
            ),
            # --- Loans ---
            create_enhanced_numeric_input(
                input_ID=f"{_pfx}_amount_loan_outstanding",
                label_text="Outstanding Loan",
                min_value=0,
                max_value=50_000_000,
                step_size=1_000,
                default_value=0,
                suffix_symbol=" $",
                tooltip_text="Current outstanding loan balance",
                input_width="100%",
            ),
            create_enhanced_numeric_input(
                input_ID=f"{_pfx}_rate_loan_interest",
                label_text="Loan Interest Rate",
                min_value=0.00,
                max_value=0.12,
                step_size=0.005,
                default_value=0.05,
                tooltip_text=("Policy loan interest rate (e.g. 0.05 = 5 %)"),
                input_width="100%",
            ),
            # --- Surrender ---
            create_enhanced_numeric_input(
                input_ID=f"{_pfx}_rate_surrender_charge",
                label_text="Surrender Charge Rate",
                min_value=0.00,
                max_value=0.20,
                step_size=0.01,
                default_value=0.00,
                tooltip_text=("Current surrender charge rate (e.g. 0.08 = 8 %)"),
                input_width="100%",
            ),
            create_enhanced_numeric_input(
                input_ID=f"{_pfx}_surrender_charge_years",
                label_text="Surrender Schedule Years",
                min_value=0,
                max_value=20,
                step_size=1,
                default_value=10,
                suffix_symbol=" years",
                tooltip_text=("Number of years the surrender charge applies"),
                input_width="100%",
            ),
            create_enhanced_select_input(
                input_ID=f"{_pfx}_payment_frequency",
                label_text="Payment Frequency",
                choices=_PAYMENT_FREQUENCY_CHOICES,
                default_selection="1",
                tooltip_text="Number of premium payments per year",
            ),
        ],
        icon_class="fas fa-user-friends",
        card_class="shadow-sm border-danger",
    )


# ======================================================================
# Module UI
# ======================================================================


@module.ui
def subtab_insurance_life_ui(
    data_utils: dict[str, Any],
    data_inputs: dict[str, Any],
) -> Any:
    """Create the Life Insurance subtab UI with five sub-subtabs.

    Parameters
    ----------
    data_utils : dict[str, Any]
        Utility functions and configuration data.
    data_inputs : dict[str, Any]
        Default input values and validation rules.

    Returns
    -------
    ui.div
        Complete Life Insurance subtab UI with five nav-panels.
    """
    return ui.div(
        ui.h3("Life Insurance Products", class_="text-center mb-4"),
        ui.p(
            "Select a life insurance type and enter the corresponding parameters.",
            class_="text-muted text-center mb-3",
        ),
        ui.navset_tab(
            ui.nav_panel("Whole Life", _create_whole_life_ui()),
            ui.nav_panel("Term Life", _create_term_life_ui()),
            ui.nav_panel("Universal Life", _create_universal_life_ui()),
            ui.nav_panel("Variable Life", _create_variable_life_ui()),
            ui.nav_panel("Survivor Life", _create_survivor_life_ui()),
            id="ID_tab_products_subtab_insurance_life_tabs_all",
        ),
    )


# ======================================================================
# Module Server
# ======================================================================


@module.server
def subtab_insurance_life_server(
    input: typing.Any,
    output: typing.Any,
    session: typing.Any,
    data_utils: dict[str, Any],
    data_inputs: dict[str, Any],
    reactives_shiny: dict[str, Any],
) -> None:
    """Server logic for the Life Insurance subtab.

    Stores user-entered life insurance parameters into the shared reactive
    state dictionary so that downstream modules (results, reporting) can
    consume them.

    Parameters
    ----------
    input : typing.Any
        Shiny input object with reactive values from the insurance forms.
    output : typing.Any
        Shiny output object (unused for now — reserved for future renders).
    session : typing.Any
        Shiny session object.
    data_utils : dict[str, Any]
        Utility functions and configuration data.
    data_inputs : dict[str, Any]
        Default input values and validation rules.
    reactives_shiny : dict[str, Any]
        Shared reactive state dictionary for cross-module communication.
    """
    _logger.info("Initializing Life Insurance subtab server")

    # ------------------------------------------------------------------
    # Reactive calculations — gather Whole Life inputs
    # ------------------------------------------------------------------

    @reactive.calc
    def calc_whole_life_params() -> dict[str, Any]:
        """Collect Whole Life input values into a dictionary."""
        _pfx = "input_ID_tab_products_subtab_insurance_life_whole_life"
        return {
            "insured_age": getattr(input, f"{_pfx}_insured_age")(),
            "face_amount": getattr(input, f"{_pfx}_face_amount")(),
            "rate_guaranteed_interest": getattr(
                input,
                f"{_pfx}_rate_guaranteed_interest",
            )(),
            "cash_value": getattr(input, f"{_pfx}_cash_value")(),
            "is_participating": getattr(
                input,
                f"{_pfx}_is_participating",
            )(),
            "rate_dividend": getattr(input, f"{_pfx}_rate_dividend")(),
            "paid_up_additions": getattr(
                input,
                f"{_pfx}_paid_up_additions",
            )(),
            "rate_loan_interest": getattr(
                input,
                f"{_pfx}_rate_loan_interest",
            )(),
            "amount_loan_outstanding": getattr(
                input,
                f"{_pfx}_amount_loan_outstanding",
            )(),
            "premium_paying_years": getattr(
                input,
                f"{_pfx}_premium_paying_years",
            )(),
            "rate_cost_of_insurance": getattr(
                input,
                f"{_pfx}_rate_cost_of_insurance",
            )(),
            "death_benefit_option": getattr(
                input,
                f"{_pfx}_death_benefit_option",
            )(),
            "underwriting_class": getattr(
                input,
                f"{_pfx}_underwriting_class",
            )(),
            "is_smoker": getattr(input, f"{_pfx}_is_smoker")(),
            "payment_frequency": getattr(
                input,
                f"{_pfx}_payment_frequency",
            )(),
        }

    # ------------------------------------------------------------------
    # Reactive calculations — gather Term Life inputs
    # ------------------------------------------------------------------

    @reactive.calc
    def calc_term_life_params() -> dict[str, Any]:
        """Collect Term Life input values into a dictionary."""
        _pfx = "input_ID_tab_products_subtab_insurance_life_term_life"
        return {
            "insured_age": getattr(input, f"{_pfx}_insured_age")(),
            "face_amount": getattr(input, f"{_pfx}_face_amount")(),
            "term_years": getattr(input, f"{_pfx}_term_years")(),
            "term_type": getattr(input, f"{_pfx}_term_type")(),
            "is_convertible": getattr(
                input,
                f"{_pfx}_is_convertible",
            )(),
            "conversion_deadline_year": getattr(
                input,
                f"{_pfx}_conversion_deadline_year",
            )(),
            "is_renewable": getattr(input, f"{_pfx}_is_renewable")(),
            "has_return_of_premium": getattr(
                input,
                f"{_pfx}_has_return_of_premium",
            )(),
            "rate_cost_of_insurance": getattr(
                input,
                f"{_pfx}_rate_cost_of_insurance",
            )(),
            "underwriting_class": getattr(
                input,
                f"{_pfx}_underwriting_class",
            )(),
            "is_smoker": getattr(input, f"{_pfx}_is_smoker")(),
            "payment_frequency": getattr(
                input,
                f"{_pfx}_payment_frequency",
            )(),
        }

    # ------------------------------------------------------------------
    # Reactive calculations — gather Universal Life inputs
    # ------------------------------------------------------------------

    @reactive.calc
    def calc_universal_life_params() -> dict[str, Any]:
        """Collect Universal Life input values into a dictionary."""
        _pfx = "input_ID_tab_products_subtab_insurance_life_universal_life"
        return {
            "insured_age": getattr(input, f"{_pfx}_insured_age")(),
            "face_amount": getattr(input, f"{_pfx}_face_amount")(),
            "ul_variant": getattr(input, f"{_pfx}_ul_variant")(),
            "rate_crediting_current": getattr(
                input,
                f"{_pfx}_rate_crediting_current",
            )(),
            "rate_crediting_guaranteed": getattr(
                input,
                f"{_pfx}_rate_crediting_guaranteed",
            )(),
            "rate_cap": getattr(input, f"{_pfx}_rate_cap")(),
            "rate_floor": getattr(input, f"{_pfx}_rate_floor")(),
            "participation_rate": getattr(
                input,
                f"{_pfx}_participation_rate",
            )(),
            "cash_value": getattr(input, f"{_pfx}_cash_value")(),
            "rate_COI": getattr(input, f"{_pfx}_rate_COI")(),
            "rate_expense_charge": getattr(
                input,
                f"{_pfx}_rate_expense_charge",
            )(),
            "target_premium": getattr(
                input,
                f"{_pfx}_target_premium",
            )(),
            "minimum_premium": getattr(
                input,
                f"{_pfx}_minimum_premium",
            )(),
            "rate_surrender_charge": getattr(
                input,
                f"{_pfx}_rate_surrender_charge",
            )(),
            "surrender_charge_years": getattr(
                input,
                f"{_pfx}_surrender_charge_years",
            )(),
            "has_no_lapse_guarantee": getattr(
                input,
                f"{_pfx}_has_no_lapse_guarantee",
            )(),
            "nlg_guarantee_years": getattr(
                input,
                f"{_pfx}_nlg_guarantee_years",
            )(),
            "death_benefit_option": getattr(
                input,
                f"{_pfx}_death_benefit_option",
            )(),
            "amount_loan_outstanding": getattr(
                input,
                f"{_pfx}_amount_loan_outstanding",
            )(),
            "rate_loan_interest": getattr(
                input,
                f"{_pfx}_rate_loan_interest",
            )(),
            "underwriting_class": getattr(
                input,
                f"{_pfx}_underwriting_class",
            )(),
            "is_smoker": getattr(input, f"{_pfx}_is_smoker")(),
            "payment_frequency": getattr(
                input,
                f"{_pfx}_payment_frequency",
            )(),
        }

    # ------------------------------------------------------------------
    # Reactive calculations — gather Variable Life inputs
    # ------------------------------------------------------------------

    @reactive.calc
    def calc_variable_life_params() -> dict[str, Any]:
        """Collect Variable Life input values into a dictionary."""
        _pfx = "input_ID_tab_products_subtab_insurance_life_variable_life"
        return {
            "insured_age": getattr(input, f"{_pfx}_insured_age")(),
            "face_amount": getattr(input, f"{_pfx}_face_amount")(),
            "cash_value": getattr(input, f"{_pfx}_cash_value")(),
            "rate_ME_charge": getattr(
                input,
                f"{_pfx}_rate_ME_charge",
            )(),
            "rate_admin_fee": getattr(
                input,
                f"{_pfx}_rate_admin_fee",
            )(),
            "rate_investment_management": getattr(
                input,
                f"{_pfx}_rate_investment_management",
            )(),
            "rate_COI": getattr(input, f"{_pfx}_rate_COI")(),
            "rate_surrender_charge": getattr(
                input,
                f"{_pfx}_rate_surrender_charge",
            )(),
            "surrender_charge_years": getattr(
                input,
                f"{_pfx}_surrender_charge_years",
            )(),
            "death_benefit_option": getattr(
                input,
                f"{_pfx}_death_benefit_option",
            )(),
            "amount_loan_outstanding": getattr(
                input,
                f"{_pfx}_amount_loan_outstanding",
            )(),
            "rate_loan_interest": getattr(
                input,
                f"{_pfx}_rate_loan_interest",
            )(),
            "underwriting_class": getattr(
                input,
                f"{_pfx}_underwriting_class",
            )(),
            "is_smoker": getattr(input, f"{_pfx}_is_smoker")(),
            "payment_frequency": getattr(
                input,
                f"{_pfx}_payment_frequency",
            )(),
        }

    # ------------------------------------------------------------------
    # Reactive calculations — gather Survivor Life inputs
    # ------------------------------------------------------------------

    @reactive.calc
    def calc_survivor_life_params() -> dict[str, Any]:
        """Collect Survivor Life input values into a dictionary."""
        _pfx = "input_ID_tab_products_subtab_insurance_life_survivor_life"
        return {
            "insured_age": getattr(input, f"{_pfx}_insured_age")(),
            "insured_age_second": getattr(
                input,
                f"{_pfx}_insured_age_second",
            )(),
            "face_amount": getattr(input, f"{_pfx}_face_amount")(),
            "chassis": getattr(input, f"{_pfx}_chassis")(),
            "underwriting_class": getattr(
                input,
                f"{_pfx}_underwriting_class",
            )(),
            "underwriting_class_second": getattr(
                input,
                f"{_pfx}_underwriting_class_second",
            )(),
            "is_smoker": getattr(input, f"{_pfx}_is_smoker")(),
            "is_smoker_second": getattr(
                input,
                f"{_pfx}_is_smoker_second",
            )(),
            "first_death_occurred": getattr(
                input,
                f"{_pfx}_first_death_occurred",
            )(),
            "cash_value": getattr(input, f"{_pfx}_cash_value")(),
            "rate_guaranteed_interest": getattr(
                input,
                f"{_pfx}_rate_guaranteed_interest",
            )(),
            "rate_COI_insured_1": getattr(
                input,
                f"{_pfx}_rate_COI_insured_1",
            )(),
            "rate_COI_insured_2": getattr(
                input,
                f"{_pfx}_rate_COI_insured_2",
            )(),
            "death_benefit_option": getattr(
                input,
                f"{_pfx}_death_benefit_option",
            )(),
            "has_split_option": getattr(
                input,
                f"{_pfx}_has_split_option",
            )(),
            "amount_loan_outstanding": getattr(
                input,
                f"{_pfx}_amount_loan_outstanding",
            )(),
            "rate_loan_interest": getattr(
                input,
                f"{_pfx}_rate_loan_interest",
            )(),
            "rate_surrender_charge": getattr(
                input,
                f"{_pfx}_rate_surrender_charge",
            )(),
            "surrender_charge_years": getattr(
                input,
                f"{_pfx}_surrender_charge_years",
            )(),
            "payment_frequency": getattr(
                input,
                f"{_pfx}_payment_frequency",
            )(),
        }

    # ------------------------------------------------------------------
    # Store life insurance parameters into reactives_shiny on change
    # ------------------------------------------------------------------

    @reactive.effect
    def _sync_insurance_life_params_to_reactives() -> None:
        """Push the latest life insurance params into the shared state."""
        if "User_Inputs_Shiny" not in reactives_shiny:
            _logger.warning(
                "reactives_shiny missing 'User_Inputs_Shiny' key",
            )
            return

        user_inputs = reactives_shiny["User_Inputs_Shiny"]

        # Initialise reactive.Value containers on first run
        _keys = [
            "Insurance_Life_Whole_Params",
            "Insurance_Life_Term_Params",
            "Insurance_Life_Universal_Params",
            "Insurance_Life_Variable_Params",
            "Insurance_Life_Survivor_Params",
        ]
        for key in _keys:
            if key not in user_inputs:
                user_inputs[key] = reactive.value(None)

        user_inputs["Insurance_Life_Whole_Params"].set(
            calc_whole_life_params(),
        )
        user_inputs["Insurance_Life_Term_Params"].set(
            calc_term_life_params(),
        )
        user_inputs["Insurance_Life_Universal_Params"].set(
            calc_universal_life_params(),
        )
        user_inputs["Insurance_Life_Variable_Params"].set(
            calc_variable_life_params(),
        )
        user_inputs["Insurance_Life_Survivor_Params"].set(
            calc_survivor_life_params(),
        )
