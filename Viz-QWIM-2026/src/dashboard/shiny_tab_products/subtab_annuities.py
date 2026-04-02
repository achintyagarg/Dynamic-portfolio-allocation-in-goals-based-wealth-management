"""Annuities Subtab Module for QWIM Dashboard.

Provides input forms for the five annuity product types:
SPIA (Single Premium Immediate Annuity), DIA (Deferred Income Annuity),
FIA (Fixed Indexed Annuity), VA (Variable Annuity), and
RILA (Registered Index-Linked Annuity).

Each annuity type is rendered in its own sub-subtab so users can enter
parameters independently.  All input identifiers follow the hierarchical
naming convention:

    ``input_ID_tab_products_subtab_annuities_<annuity>_<field>``

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
# Payout option choices shared by SPIA and DIA
# ======================================================================

_PAYOUT_OPTION_CHOICES: dict[str, str] = {
    "life_only": "Life Only",
    "period_certain": "Period Certain",
    "life_with_period_certain": "Life with Period Certain",
    "joint_life": "Joint Life",
}

_PAYMENT_FREQUENCY_CHOICES: dict[str, str] = {
    "1": "Annual",
    "2": "Semi-Annual",
    "4": "Quarterly",
    "12": "Monthly",
}

_PROTECTION_TYPE_CHOICES: dict[str, str] = {
    "buffer": "Buffer",
    "floor": "Floor",
}

_CREDITING_STRATEGY_CHOICES: dict[str, str] = {
    "cap": "Cap",
    "performance_trigger": "Performance Trigger",
    "participation_rate": "Participation Rate",
}


# ======================================================================
# SPIA sub-subtab UI helper
# ======================================================================


def _create_spia_ui() -> Any:
    """Build the input form for a Single Premium Immediate Annuity (SPIA).

    Returns
    -------
    ui.div
        Complete SPIA input card.
    """
    return create_enhanced_card_section(
        title="SPIA Parameters",
        content=[
            create_enhanced_numeric_input(
                input_ID=("input_ID_tab_products_subtab_annuities_SPIA_age_client"),
                label_text="Client Age",
                min_value=18,
                max_value=100,
                step_size=1,
                default_value=65,
                suffix_symbol=" years",
                tooltip_text="Current age of the annuitant",
                required_field=True,
                input_width="100%",
            ),
            create_enhanced_numeric_input(
                input_ID=("input_ID_tab_products_subtab_annuities_SPIA_rate_payout"),
                label_text="Payout Rate",
                min_value=0.00,
                max_value=1.00,
                step_size=0.005,
                default_value=0.06,
                tooltip_text="Annual payout rate as a decimal (e.g. 0.06 = 6 %)",
                required_field=True,
                input_width="100%",
            ),
            create_enhanced_select_input(
                input_ID=("input_ID_tab_products_subtab_annuities_SPIA_payout_option"),
                label_text="Payout Option",
                choices=_PAYOUT_OPTION_CHOICES,
                default_selection="life_only",
                tooltip_text="Payout structure for the SPIA",
                required_field=True,
            ),
            create_enhanced_numeric_input(
                input_ID=("input_ID_tab_products_subtab_annuities_SPIA_guarantee_period_years"),
                label_text="Guarantee Period",
                min_value=0,
                max_value=30,
                step_size=1,
                default_value=0,
                suffix_symbol=" years",
                tooltip_text=(
                    "Number of guaranteed payment years (required for Period Certain options)"
                ),
                input_width="100%",
            ),
            create_enhanced_numeric_input(
                input_ID=("input_ID_tab_products_subtab_annuities_SPIA_rate_COLA"),
                label_text="COLA Rate",
                min_value=0.00,
                max_value=0.20,
                step_size=0.005,
                default_value=0.00,
                tooltip_text="Annual cost-of-living adjustment rate (0.0 = fixed)",
                input_width="100%",
            ),
            create_enhanced_numeric_input(
                input_ID=("input_ID_tab_products_subtab_annuities_SPIA_pct_joint_survivor"),
                label_text="Joint Survivor %",
                min_value=0.00,
                max_value=1.00,
                step_size=0.05,
                default_value=1.00,
                tooltip_text=(
                    "Survivor benefit as a fraction of the primary benefit "
                    "(0.50, 0.75, or 1.00 are common)"
                ),
                input_width="100%",
            ),
            create_enhanced_select_input(
                input_ID=("input_ID_tab_products_subtab_annuities_SPIA_payment_frequency"),
                label_text="Payment Frequency",
                choices=_PAYMENT_FREQUENCY_CHOICES,
                default_selection="12",
                tooltip_text="Number of payments per year",
            ),
        ],
        icon_class="fas fa-money-check-alt",
        card_class="shadow-sm border-primary",
    )


# ======================================================================
# DIA sub-subtab UI helper
# ======================================================================


def _create_dia_ui() -> Any:
    """Build the input form for a Deferred Income Annuity (DIA).

    Returns
    -------
    ui.div
        Complete DIA input card.
    """
    return create_enhanced_card_section(
        title="DIA Parameters",
        content=[
            create_enhanced_numeric_input(
                input_ID=("input_ID_tab_products_subtab_annuities_DIA_age_client"),
                label_text="Client Age",
                min_value=18,
                max_value=100,
                step_size=1,
                default_value=55,
                suffix_symbol=" years",
                tooltip_text="Current age of the annuitant",
                required_field=True,
                input_width="100%",
            ),
            create_enhanced_numeric_input(
                input_ID=("input_ID_tab_products_subtab_annuities_DIA_rate_payout"),
                label_text="Payout Rate",
                min_value=0.00,
                max_value=1.00,
                step_size=0.005,
                default_value=0.08,
                tooltip_text="Annual payout rate as a decimal (e.g. 0.08 = 8 %)",
                required_field=True,
                input_width="100%",
            ),
            create_enhanced_numeric_input(
                input_ID=("input_ID_tab_products_subtab_annuities_DIA_age_income_start"),
                label_text="Income Start Age",
                min_value=50,
                max_value=90,
                step_size=1,
                default_value=65,
                suffix_symbol=" years",
                tooltip_text="Age at which annuity income payments begin",
                required_field=True,
                input_width="100%",
            ),
            create_enhanced_select_input(
                input_ID=("input_ID_tab_products_subtab_annuities_DIA_payout_option"),
                label_text="Payout Option",
                choices=_PAYOUT_OPTION_CHOICES,
                default_selection="life_only",
                tooltip_text="Payout structure for the DIA",
                required_field=True,
            ),
            create_enhanced_numeric_input(
                input_ID=("input_ID_tab_products_subtab_annuities_DIA_guarantee_period_years"),
                label_text="Guarantee Period",
                min_value=0,
                max_value=30,
                step_size=1,
                default_value=0,
                suffix_symbol=" years",
                tooltip_text="Number of guaranteed payment years once income begins",
                input_width="100%",
            ),
            create_enhanced_numeric_input(
                input_ID=("input_ID_tab_products_subtab_annuities_DIA_rate_COLA"),
                label_text="COLA Rate",
                min_value=0.00,
                max_value=0.20,
                step_size=0.005,
                default_value=0.00,
                tooltip_text="Annual cost-of-living adjustment rate (0.0 = fixed)",
                input_width="100%",
            ),
            create_enhanced_numeric_input(
                input_ID=("input_ID_tab_products_subtab_annuities_DIA_rate_mortality_credit"),
                label_text="Mortality Credit Rate",
                min_value=0.00,
                max_value=0.10,
                step_size=0.005,
                default_value=0.00,
                tooltip_text=("Implicit mortality credit rate earned during the deferral period"),
                input_width="100%",
            ),
            create_enhanced_select_input(
                input_ID=("input_ID_tab_products_subtab_annuities_DIA_has_death_benefit_ROP"),
                label_text="Return-of-Premium Death Benefit",
                choices={"false": "No", "true": "Yes"},
                default_selection="false",
                tooltip_text=(
                    "Whether the contract refunds the premium if the "
                    "annuitant dies during the deferral period"
                ),
            ),
            create_enhanced_select_input(
                input_ID=("input_ID_tab_products_subtab_annuities_DIA_payment_frequency"),
                label_text="Payment Frequency",
                choices=_PAYMENT_FREQUENCY_CHOICES,
                default_selection="12",
                tooltip_text="Number of payments per year",
            ),
        ],
        icon_class="fas fa-hourglass-half",
        card_class="shadow-sm border-info",
    )


# ======================================================================
# FIA sub-subtab UI helper
# ======================================================================


def _create_fia_ui() -> Any:
    """Build the input form for a Fixed Indexed Annuity (FIA).

    Returns
    -------
    ui.div
        Complete FIA input card.
    """
    return create_enhanced_card_section(
        title="FIA Parameters",
        content=[
            # --- Core parameters ---
            create_enhanced_numeric_input(
                input_ID=("input_ID_tab_products_subtab_annuities_FIA_age_client"),
                label_text="Client Age",
                min_value=18,
                max_value=100,
                step_size=1,
                default_value=55,
                suffix_symbol=" years",
                tooltip_text="Current age of the client",
                required_field=True,
                input_width="100%",
            ),
            create_enhanced_numeric_input(
                input_ID=("input_ID_tab_products_subtab_annuities_FIA_rate_payout"),
                label_text="Payout Rate",
                min_value=0.00,
                max_value=1.00,
                step_size=0.005,
                default_value=0.05,
                tooltip_text="Annual payout rate as a decimal (e.g. 0.05 = 5 %)",
                required_field=True,
                input_width="100%",
            ),
            create_enhanced_numeric_input(
                input_ID=("input_ID_tab_products_subtab_annuities_FIA_age_income_start"),
                label_text="Income Start Age",
                min_value=50,
                max_value=90,
                step_size=1,
                default_value=65,
                suffix_symbol=" years",
                tooltip_text="Age at which income payments begin",
                required_field=True,
                input_width="100%",
            ),
            create_enhanced_numeric_input(
                input_ID=("input_ID_tab_products_subtab_annuities_FIA_age_max_ratchet"),
                label_text="Max Ratchet Age",
                min_value=55,
                max_value=100,
                step_size=1,
                default_value=85,
                suffix_symbol=" years",
                tooltip_text="Maximum age until which the ratchet feature applies",
                input_width="100%",
            ),
            # --- Crediting parameters ---
            create_enhanced_numeric_input(
                input_ID=("input_ID_tab_products_subtab_annuities_FIA_rate_rollup_benefit"),
                label_text="Rollup Rate",
                min_value=0.00,
                max_value=0.15,
                step_size=0.005,
                default_value=0.05,
                tooltip_text="Annual rollup rate applied to the benefit base",
                input_width="100%",
            ),
            create_enhanced_numeric_input(
                input_ID=("input_ID_tab_products_subtab_annuities_FIA_cap_rate"),
                label_text="Cap Rate",
                min_value=0.00,
                max_value=0.30,
                step_size=0.005,
                default_value=0.05,
                tooltip_text="Maximum annual return that can be credited",
                input_width="100%",
            ),
            create_enhanced_numeric_input(
                input_ID=("input_ID_tab_products_subtab_annuities_FIA_participation_rate"),
                label_text="Participation Rate",
                min_value=0.00,
                max_value=1.00,
                step_size=0.05,
                default_value=1.00,
                tooltip_text="Fraction of index growth credited (1.0 = 100 %)",
                input_width="100%",
            ),
            create_enhanced_numeric_input(
                input_ID=("input_ID_tab_products_subtab_annuities_FIA_spread_rate"),
                label_text="Spread Rate",
                min_value=0.00,
                max_value=0.10,
                step_size=0.005,
                default_value=0.00,
                tooltip_text="Spread / margin deducted from index return",
                input_width="100%",
            ),
            create_enhanced_numeric_input(
                input_ID=("input_ID_tab_products_subtab_annuities_FIA_floor_rate"),
                label_text="Floor Rate",
                min_value=-0.05,
                max_value=0.05,
                step_size=0.005,
                default_value=0.00,
                tooltip_text="Minimum credited rate per period (typically 0 %)",
                input_width="100%",
            ),
            # --- Surrender / guarantee parameters ---
            create_enhanced_numeric_input(
                input_ID=(
                    "input_ID_tab_products_subtab_annuities_FIA_rate_surrender_charge_initial"
                ),
                label_text="Initial Surrender Charge",
                min_value=0.00,
                max_value=0.20,
                step_size=0.01,
                default_value=0.08,
                tooltip_text="Initial surrender charge rate (e.g. 0.08 = 8 %)",
                input_width="100%",
            ),
            create_enhanced_numeric_input(
                input_ID=(
                    "input_ID_tab_products_subtab_annuities_FIA_surrender_charge_schedule_years"
                ),
                label_text="Surrender Schedule Years",
                min_value=0,
                max_value=15,
                step_size=1,
                default_value=7,
                suffix_symbol=" years",
                tooltip_text=("Number of years for the surrender charge to decline to zero"),
                input_width="100%",
            ),
            create_enhanced_numeric_input(
                input_ID=("input_ID_tab_products_subtab_annuities_FIA_rate_minimum_guarantee"),
                label_text="Minimum Guarantee Rate",
                min_value=0.00,
                max_value=0.05,
                step_size=0.005,
                default_value=0.01,
                tooltip_text="Minimum guaranteed annual interest rate (e.g. 0.01 = 1 %)",
                input_width="100%",
            ),
            create_enhanced_numeric_input(
                input_ID=("input_ID_tab_products_subtab_annuities_FIA_pct_minimum_guarantee_base"),
                label_text="Guarantee Base %",
                min_value=0.50,
                max_value=1.00,
                step_size=0.025,
                default_value=0.875,
                tooltip_text=(
                    "Fraction of premium used as the guarantee base (e.g. 0.875 = 87.5 %)"
                ),
                input_width="100%",
            ),
            create_enhanced_numeric_input(
                input_ID=("input_ID_tab_products_subtab_annuities_FIA_pct_free_withdrawal"),
                label_text="Free Withdrawal %",
                min_value=0.00,
                max_value=0.20,
                step_size=0.01,
                default_value=0.10,
                tooltip_text="Annual free-withdrawal percentage (e.g. 0.10 = 10 %)",
                input_width="100%",
            ),
            create_enhanced_select_input(
                input_ID=("input_ID_tab_products_subtab_annuities_FIA_payment_frequency"),
                label_text="Payment Frequency",
                choices=_PAYMENT_FREQUENCY_CHOICES,
                default_selection="12",
                tooltip_text="Number of payments per year",
            ),
        ],
        icon_class="fas fa-chart-line",
        card_class="shadow-sm border-success",
    )


# ======================================================================
# VA sub-subtab UI helper
# ======================================================================


def _create_va_ui() -> Any:
    """Build the input form for a Variable Annuity (VA).

    Returns
    -------
    ui.div
        Complete VA input card.
    """
    return create_enhanced_card_section(
        title="VA Parameters",
        content=[
            # --- Core parameters ---
            create_enhanced_numeric_input(
                input_ID=("input_ID_tab_products_subtab_annuities_VA_age_client"),
                label_text="Client Age",
                min_value=18,
                max_value=100,
                step_size=1,
                default_value=55,
                suffix_symbol=" years",
                tooltip_text="Current age of the client",
                required_field=True,
                input_width="100%",
            ),
            create_enhanced_numeric_input(
                input_ID=("input_ID_tab_products_subtab_annuities_VA_rate_payout"),
                label_text="Payout Rate",
                min_value=0.00,
                max_value=1.00,
                step_size=0.005,
                default_value=0.05,
                tooltip_text="Annual payout rate as a decimal (e.g. 0.05 = 5 %)",
                required_field=True,
                input_width="100%",
            ),
            create_enhanced_numeric_input(
                input_ID=("input_ID_tab_products_subtab_annuities_VA_age_income_start"),
                label_text="Income Start Age",
                min_value=50,
                max_value=90,
                step_size=1,
                default_value=65,
                suffix_symbol=" years",
                tooltip_text="Age at which income payments begin",
                required_field=True,
                input_width="100%",
            ),
            create_enhanced_numeric_input(
                input_ID=("input_ID_tab_products_subtab_annuities_VA_age_max_ratchet"),
                label_text="Max Ratchet Age",
                min_value=55,
                max_value=100,
                step_size=1,
                default_value=85,
                suffix_symbol=" years",
                tooltip_text="Maximum age until which the ratchet feature applies",
                input_width="100%",
            ),
            create_enhanced_numeric_input(
                input_ID=("input_ID_tab_products_subtab_annuities_VA_rate_rollup_benefit"),
                label_text="Rollup Rate",
                min_value=0.00,
                max_value=0.15,
                step_size=0.005,
                default_value=0.05,
                tooltip_text="Annual rollup rate applied to the benefit base",
                input_width="100%",
            ),
            # --- Charges ---
            create_enhanced_numeric_input(
                input_ID=("input_ID_tab_products_subtab_annuities_VA_rate_ME_charge"),
                label_text="M&E Charge",
                min_value=0.00,
                max_value=0.05,
                step_size=0.0025,
                default_value=0.0125,
                tooltip_text="Annual mortality & expense charge (e.g. 0.0125 = 1.25 %)",
                input_width="100%",
            ),
            create_enhanced_numeric_input(
                input_ID=("input_ID_tab_products_subtab_annuities_VA_rate_admin_fee"),
                label_text="Admin Fee",
                min_value=0.00,
                max_value=0.02,
                step_size=0.0005,
                default_value=0.0015,
                tooltip_text="Annual administrative fee (e.g. 0.0015 = 0.15 %)",
                input_width="100%",
            ),
            create_enhanced_numeric_input(
                input_ID=("input_ID_tab_products_subtab_annuities_VA_rate_rider_charge"),
                label_text="Rider Charge",
                min_value=0.00,
                max_value=0.05,
                step_size=0.0025,
                default_value=0.0100,
                tooltip_text=("Total annual rider charge for GMDB / GLWB (e.g. 0.01 = 1.00 %)"),
                input_width="100%",
            ),
            # --- Surrender ---
            create_enhanced_numeric_input(
                input_ID=(
                    "input_ID_tab_products_subtab_annuities_VA_rate_surrender_charge_initial"
                ),
                label_text="Initial Surrender Charge",
                min_value=0.00,
                max_value=0.20,
                step_size=0.01,
                default_value=0.07,
                tooltip_text="Initial surrender charge rate (e.g. 0.07 = 7 %)",
                input_width="100%",
            ),
            create_enhanced_numeric_input(
                input_ID=(
                    "input_ID_tab_products_subtab_annuities_VA_surrender_charge_schedule_years"
                ),
                label_text="Surrender Schedule Years",
                min_value=0,
                max_value=15,
                step_size=1,
                default_value=7,
                suffix_symbol=" years",
                tooltip_text=("Number of years for surrender charge to decline to zero"),
                input_width="100%",
            ),
            create_enhanced_numeric_input(
                input_ID=("input_ID_tab_products_subtab_annuities_VA_pct_free_withdrawal"),
                label_text="Free Withdrawal %",
                min_value=0.00,
                max_value=0.20,
                step_size=0.01,
                default_value=0.10,
                tooltip_text="Annual free-withdrawal percentage (e.g. 0.10 = 10 %)",
                input_width="100%",
            ),
            # --- Guarantees ---
            create_enhanced_select_input(
                input_ID=("input_ID_tab_products_subtab_annuities_VA_has_GMDB"),
                label_text="GMDB Included",
                choices={"true": "Yes", "false": "No"},
                default_selection="true",
                tooltip_text="Include Guaranteed Minimum Death Benefit",
            ),
            create_enhanced_select_input(
                input_ID=("input_ID_tab_products_subtab_annuities_VA_has_GLWB"),
                label_text="GLWB Included",
                choices={"true": "Yes", "false": "No"},
                default_selection="true",
                tooltip_text="Include Guaranteed Lifetime Withdrawal Benefit",
            ),
            create_enhanced_numeric_input(
                input_ID=("input_ID_tab_products_subtab_annuities_VA_pct_GLWB"),
                label_text="GLWB Withdrawal %",
                min_value=0.00,
                max_value=0.15,
                step_size=0.005,
                default_value=0.05,
                tooltip_text=(
                    "GLWB annual withdrawal percentage of benefit base (e.g. 0.05 = 5 %)"
                ),
                input_width="100%",
            ),
            create_enhanced_select_input(
                input_ID=("input_ID_tab_products_subtab_annuities_VA_payment_frequency"),
                label_text="Payment Frequency",
                choices=_PAYMENT_FREQUENCY_CHOICES,
                default_selection="12",
                tooltip_text="Number of payments per year",
            ),
        ],
        icon_class="fas fa-chart-area",
        card_class="shadow-sm border-warning",
    )


# ======================================================================
# RILA sub-subtab UI helper
# ======================================================================


def _create_rila_ui() -> Any:
    """Build the input form for a Registered Index-Linked Annuity (RILA).

    Returns
    -------
    ui.div
        Complete RILA input card.
    """
    return create_enhanced_card_section(
        title="RILA Parameters",
        content=[
            # --- Core parameters ---
            create_enhanced_numeric_input(
                input_ID=("input_ID_tab_products_subtab_annuities_RILA_age_client"),
                label_text="Client Age",
                min_value=18,
                max_value=100,
                step_size=1,
                default_value=55,
                suffix_symbol=" years",
                tooltip_text="Current age of the client",
                required_field=True,
                input_width="100%",
            ),
            create_enhanced_numeric_input(
                input_ID=("input_ID_tab_products_subtab_annuities_RILA_rate_payout"),
                label_text="Payout Rate",
                min_value=0.00,
                max_value=1.00,
                step_size=0.005,
                default_value=0.05,
                tooltip_text="Annual payout rate as a decimal (e.g. 0.05 = 5 %)",
                required_field=True,
                input_width="100%",
            ),
            create_enhanced_numeric_input(
                input_ID=("input_ID_tab_products_subtab_annuities_RILA_age_income_start"),
                label_text="Income Start Age",
                min_value=50,
                max_value=90,
                step_size=1,
                default_value=65,
                suffix_symbol=" years",
                tooltip_text="Age at which income payments begin",
                required_field=True,
                input_width="100%",
            ),
            create_enhanced_numeric_input(
                input_ID=("input_ID_tab_products_subtab_annuities_RILA_term_years"),
                label_text="Term Length",
                min_value=1,
                max_value=10,
                step_size=1,
                default_value=6,
                suffix_symbol=" years",
                tooltip_text="Duration of each crediting term / segment",
                required_field=True,
                input_width="100%",
            ),
            # --- Downside protection ---
            create_enhanced_select_input(
                input_ID=("input_ID_tab_products_subtab_annuities_RILA_protection_type"),
                label_text="Protection Type",
                choices=_PROTECTION_TYPE_CHOICES,
                default_selection="buffer",
                tooltip_text=(
                    "Buffer: insurer absorbs first N % of loss; "
                    "Floor: maximum loss capped at floor level"
                ),
                required_field=True,
            ),
            create_enhanced_numeric_input(
                input_ID=("input_ID_tab_products_subtab_annuities_RILA_buffer_rate"),
                label_text="Buffer Rate",
                min_value=0.00,
                max_value=1.00,
                step_size=0.05,
                default_value=0.10,
                tooltip_text=(
                    "Buffer percentage absorbed by the insurer (e.g. 0.10 = 10 %). "
                    "Applies when Protection Type is Buffer."
                ),
                input_width="100%",
            ),
            create_enhanced_numeric_input(
                input_ID=("input_ID_tab_products_subtab_annuities_RILA_floor_rate"),
                label_text="Floor Rate",
                min_value=-1.00,
                max_value=0.00,
                step_size=0.05,
                default_value=-0.10,
                tooltip_text=(
                    "Maximum loss percentage for the contract holder "
                    "(e.g. -0.10 = −10 %). Applies when Protection Type is Floor."
                ),
                input_width="100%",
            ),
            # --- Crediting strategy ---
            create_enhanced_select_input(
                input_ID=("input_ID_tab_products_subtab_annuities_RILA_crediting_strategy"),
                label_text="Crediting Strategy",
                choices=_CREDITING_STRATEGY_CHOICES,
                default_selection="cap",
                tooltip_text=(
                    "Cap: index return capped at max; "
                    "Performance Trigger: fixed rate if index >= 0; "
                    "Participation Rate: fraction of index return"
                ),
                required_field=True,
            ),
            create_enhanced_numeric_input(
                input_ID=("input_ID_tab_products_subtab_annuities_RILA_cap_rate"),
                label_text="Cap Rate",
                min_value=0.00,
                max_value=0.50,
                step_size=0.005,
                default_value=0.15,
                tooltip_text="Maximum return credited per term (e.g. 0.15 = 15 %)",
                input_width="100%",
            ),
            create_enhanced_numeric_input(
                input_ID=("input_ID_tab_products_subtab_annuities_RILA_participation_rate"),
                label_text="Participation Rate",
                min_value=0.00,
                max_value=3.00,
                step_size=0.05,
                default_value=1.00,
                tooltip_text=(
                    "Fraction of index return credited (e.g. 1.50 = 150 %). "
                    "Applies when Crediting Strategy is Participation Rate."
                ),
                input_width="100%",
            ),
            create_enhanced_numeric_input(
                input_ID=("input_ID_tab_products_subtab_annuities_RILA_performance_trigger_rate"),
                label_text="Performance Trigger Rate",
                min_value=0.00,
                max_value=0.30,
                step_size=0.005,
                default_value=0.08,
                tooltip_text=(
                    "Fixed rate credited when the index return is non-negative "
                    "(e.g. 0.08 = 8 %). Applies when Crediting Strategy is "
                    "Performance Trigger."
                ),
                input_width="100%",
            ),
            # --- Charges ---
            create_enhanced_numeric_input(
                input_ID=("input_ID_tab_products_subtab_annuities_RILA_rate_rider_charge"),
                label_text="Rider Charge",
                min_value=0.00,
                max_value=0.05,
                step_size=0.0025,
                default_value=0.00,
                tooltip_text="Annual rider charge for optional benefits (e.g. 0.005 = 0.50 %)",
                input_width="100%",
            ),
            # --- Surrender ---
            create_enhanced_numeric_input(
                input_ID=(
                    "input_ID_tab_products_subtab_annuities_RILA_rate_surrender_charge_initial"
                ),
                label_text="Initial Surrender Charge",
                min_value=0.00,
                max_value=0.20,
                step_size=0.01,
                default_value=0.06,
                tooltip_text="Initial surrender charge rate (e.g. 0.06 = 6 %)",
                input_width="100%",
            ),
            create_enhanced_numeric_input(
                input_ID=(
                    "input_ID_tab_products_subtab_annuities_RILA_surrender_charge_schedule_years"
                ),
                label_text="Surrender Schedule Years",
                min_value=0,
                max_value=15,
                step_size=1,
                default_value=6,
                suffix_symbol=" years",
                tooltip_text="Number of years for the surrender charge to decline to zero",
                input_width="100%",
            ),
            create_enhanced_numeric_input(
                input_ID=("input_ID_tab_products_subtab_annuities_RILA_pct_free_withdrawal"),
                label_text="Free Withdrawal %",
                min_value=0.00,
                max_value=0.20,
                step_size=0.01,
                default_value=0.10,
                tooltip_text="Annual free-withdrawal percentage (e.g. 0.10 = 10 %)",
                input_width="100%",
            ),
            create_enhanced_numeric_input(
                input_ID=("input_ID_tab_products_subtab_annuities_RILA_rate_interim_discount"),
                label_text="Interim Discount Rate",
                min_value=0.00,
                max_value=0.10,
                step_size=0.005,
                default_value=0.02,
                tooltip_text=(
                    "Discount rate used in interim-value (market-value-adjusted) "
                    "calculation (e.g. 0.02 = 2 %)"
                ),
                input_width="100%",
            ),
            # --- Guarantees ---
            create_enhanced_select_input(
                input_ID=("input_ID_tab_products_subtab_annuities_RILA_has_GMDB"),
                label_text="GMDB Included",
                choices={"false": "No", "true": "Yes"},
                default_selection="false",
                tooltip_text="Include Guaranteed Minimum Death Benefit",
            ),
            create_enhanced_select_input(
                input_ID=("input_ID_tab_products_subtab_annuities_RILA_has_return_of_premium_DB"),
                label_text="Return-of-Premium Death Benefit",
                choices={"true": "Yes", "false": "No"},
                default_selection="true",
                tooltip_text=("Whether the death benefit guarantees at least the premium paid"),
            ),
            create_enhanced_select_input(
                input_ID=("input_ID_tab_products_subtab_annuities_RILA_payment_frequency"),
                label_text="Payment Frequency",
                choices=_PAYMENT_FREQUENCY_CHOICES,
                default_selection="12",
                tooltip_text="Number of payments per year",
            ),
        ],
        icon_class="fas fa-shield-alt",
        card_class="shadow-sm border-danger",
    )


# ======================================================================
# Module UI
# ======================================================================


@module.ui
def subtab_annuities_ui(
    data_utils: dict[str, Any],
    data_inputs: dict[str, Any],
) -> Any:
    """Create the Annuities subtab UI with SPIA / DIA / FIA / VA / RILA sub-subtabs.

    Parameters
    ----------
    data_utils : dict[str, Any]
        Utility functions and configuration data.
    data_inputs : dict[str, Any]
        Default input values and validation rules.

    Returns
    -------
    ui.div
        Complete Annuities subtab UI with five nav-panels.
    """
    return ui.div(
        ui.h3("Annuity Products", class_="text-center mb-4"),
        ui.p(
            "Select an annuity type and enter the corresponding parameters.",
            class_="text-muted text-center mb-3",
        ),
        ui.navset_tab(
            ui.nav_panel("SPIA", _create_spia_ui()),
            ui.nav_panel("DIA", _create_dia_ui()),
            ui.nav_panel("FIA", _create_fia_ui()),
            ui.nav_panel("VA", _create_va_ui()),
            ui.nav_panel("RILA", _create_rila_ui()),
            id="ID_tab_products_subtab_annuities_tabs_all",
        ),
    )


# ======================================================================
# Module Server
# ======================================================================


@module.server
def subtab_annuities_server(
    input: typing.Any,
    output: typing.Any,
    session: typing.Any,
    data_utils: dict[str, Any],
    data_inputs: dict[str, Any],
    reactives_shiny: dict[str, Any],
) -> None:
    """Server logic for the Annuities subtab.

    Stores user-entered annuity parameters into the shared reactive state
    dictionary so that downstream modules (results, reporting) can consume
    them.

    Parameters
    ----------
    input : typing.Any
        Shiny input object with reactive values from the annuity forms.
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
    _logger.info("Initializing Annuities subtab server")

    # ------------------------------------------------------------------
    # Reactive calculations — gather SPIA inputs
    # ------------------------------------------------------------------

    @reactive.calc
    def calc_spia_params() -> dict[str, Any]:
        """Collect SPIA input values into a dictionary."""
        return {
            "client_age": input.input_ID_tab_products_subtab_annuities_SPIA_age_client(),
            "annuity_payout_rate": input.input_ID_tab_products_subtab_annuities_SPIA_rate_payout(),
            "payout_option": input.input_ID_tab_products_subtab_annuities_SPIA_payout_option(),
            "guarantee_period_years": input.input_ID_tab_products_subtab_annuities_SPIA_guarantee_period_years(),
            "rate_COLA": input.input_ID_tab_products_subtab_annuities_SPIA_rate_COLA(),
            "joint_survivor_pct": input.input_ID_tab_products_subtab_annuities_SPIA_pct_joint_survivor(),
            "payment_frequency": input.input_ID_tab_products_subtab_annuities_SPIA_payment_frequency(),
        }

    # ------------------------------------------------------------------
    # Reactive calculations — gather DIA inputs
    # ------------------------------------------------------------------

    @reactive.calc
    def calc_dia_params() -> dict[str, Any]:
        """Collect DIA input values into a dictionary."""
        return {
            "client_age": input.input_ID_tab_products_subtab_annuities_DIA_age_client(),
            "annuity_payout_rate": input.input_ID_tab_products_subtab_annuities_DIA_rate_payout(),
            "age_income_start": input.input_ID_tab_products_subtab_annuities_DIA_age_income_start(),
            "payout_option": input.input_ID_tab_products_subtab_annuities_DIA_payout_option(),
            "guarantee_period_years": input.input_ID_tab_products_subtab_annuities_DIA_guarantee_period_years(),
            "rate_COLA": input.input_ID_tab_products_subtab_annuities_DIA_rate_COLA(),
            "rate_mortality_credit": input.input_ID_tab_products_subtab_annuities_DIA_rate_mortality_credit(),
            "has_death_benefit_ROP": input.input_ID_tab_products_subtab_annuities_DIA_has_death_benefit_ROP(),
            "payment_frequency": input.input_ID_tab_products_subtab_annuities_DIA_payment_frequency(),
        }

    # ------------------------------------------------------------------
    # Reactive calculations — gather FIA inputs
    # ------------------------------------------------------------------

    @reactive.calc
    def calc_fia_params() -> dict[str, Any]:
        """Collect FIA input values into a dictionary."""
        return {
            "client_age": input.input_ID_tab_products_subtab_annuities_FIA_age_client(),
            "annuity_payout_rate": input.input_ID_tab_products_subtab_annuities_FIA_rate_payout(),
            "age_income_start": input.input_ID_tab_products_subtab_annuities_FIA_age_income_start(),
            "age_max_ratchet": input.input_ID_tab_products_subtab_annuities_FIA_age_max_ratchet(),
            "rate_rollup_benefit": input.input_ID_tab_products_subtab_annuities_FIA_rate_rollup_benefit(),
            "cap_rate": input.input_ID_tab_products_subtab_annuities_FIA_cap_rate(),
            "participation_rate": input.input_ID_tab_products_subtab_annuities_FIA_participation_rate(),
            "spread_rate": input.input_ID_tab_products_subtab_annuities_FIA_spread_rate(),
            "floor_rate": input.input_ID_tab_products_subtab_annuities_FIA_floor_rate(),
            "rate_surrender_charge_initial": input.input_ID_tab_products_subtab_annuities_FIA_rate_surrender_charge_initial(),
            "surrender_charge_schedule_years": input.input_ID_tab_products_subtab_annuities_FIA_surrender_charge_schedule_years(),
            "rate_minimum_guarantee": input.input_ID_tab_products_subtab_annuities_FIA_rate_minimum_guarantee(),
            "pct_minimum_guarantee_base": input.input_ID_tab_products_subtab_annuities_FIA_pct_minimum_guarantee_base(),
            "pct_free_withdrawal": input.input_ID_tab_products_subtab_annuities_FIA_pct_free_withdrawal(),
            "payment_frequency": input.input_ID_tab_products_subtab_annuities_FIA_payment_frequency(),
        }

    # ------------------------------------------------------------------
    # Reactive calculations — gather VA inputs
    # ------------------------------------------------------------------

    @reactive.calc
    def calc_va_params() -> dict[str, Any]:
        """Collect VA input values into a dictionary."""
        return {
            "client_age": input.input_ID_tab_products_subtab_annuities_VA_age_client(),
            "annuity_payout_rate": input.input_ID_tab_products_subtab_annuities_VA_rate_payout(),
            "age_income_start": input.input_ID_tab_products_subtab_annuities_VA_age_income_start(),
            "age_max_ratchet": input.input_ID_tab_products_subtab_annuities_VA_age_max_ratchet(),
            "rate_rollup_benefit": input.input_ID_tab_products_subtab_annuities_VA_rate_rollup_benefit(),
            "rate_ME_charge": input.input_ID_tab_products_subtab_annuities_VA_rate_ME_charge(),
            "rate_admin_fee": input.input_ID_tab_products_subtab_annuities_VA_rate_admin_fee(),
            "rate_rider_charge": input.input_ID_tab_products_subtab_annuities_VA_rate_rider_charge(),
            "rate_surrender_charge_initial": input.input_ID_tab_products_subtab_annuities_VA_rate_surrender_charge_initial(),
            "surrender_charge_schedule_years": input.input_ID_tab_products_subtab_annuities_VA_surrender_charge_schedule_years(),
            "pct_free_withdrawal": input.input_ID_tab_products_subtab_annuities_VA_pct_free_withdrawal(),
            "has_GMDB": input.input_ID_tab_products_subtab_annuities_VA_has_GMDB(),
            "has_GLWB": input.input_ID_tab_products_subtab_annuities_VA_has_GLWB(),
            "pct_GLWB": input.input_ID_tab_products_subtab_annuities_VA_pct_GLWB(),
            "payment_frequency": input.input_ID_tab_products_subtab_annuities_VA_payment_frequency(),
        }

    # ------------------------------------------------------------------
    # Reactive calculations — gather RILA inputs
    # ------------------------------------------------------------------

    @reactive.calc
    def calc_rila_params() -> dict[str, Any]:
        """Collect RILA input values into a dictionary."""
        return {
            "client_age": input.input_ID_tab_products_subtab_annuities_RILA_age_client(),
            "annuity_payout_rate": input.input_ID_tab_products_subtab_annuities_RILA_rate_payout(),
            "age_income_start": input.input_ID_tab_products_subtab_annuities_RILA_age_income_start(),
            "term_years": input.input_ID_tab_products_subtab_annuities_RILA_term_years(),
            "protection_type": input.input_ID_tab_products_subtab_annuities_RILA_protection_type(),
            "buffer_rate": input.input_ID_tab_products_subtab_annuities_RILA_buffer_rate(),
            "floor_rate": input.input_ID_tab_products_subtab_annuities_RILA_floor_rate(),
            "crediting_strategy": input.input_ID_tab_products_subtab_annuities_RILA_crediting_strategy(),
            "cap_rate": input.input_ID_tab_products_subtab_annuities_RILA_cap_rate(),
            "participation_rate": input.input_ID_tab_products_subtab_annuities_RILA_participation_rate(),
            "performance_trigger_rate": input.input_ID_tab_products_subtab_annuities_RILA_performance_trigger_rate(),
            "rate_rider_charge": input.input_ID_tab_products_subtab_annuities_RILA_rate_rider_charge(),
            "rate_surrender_charge_initial": input.input_ID_tab_products_subtab_annuities_RILA_rate_surrender_charge_initial(),
            "surrender_charge_schedule_years": input.input_ID_tab_products_subtab_annuities_RILA_surrender_charge_schedule_years(),
            "pct_free_withdrawal": input.input_ID_tab_products_subtab_annuities_RILA_pct_free_withdrawal(),
            "rate_interim_discount": input.input_ID_tab_products_subtab_annuities_RILA_rate_interim_discount(),
            "has_GMDB": input.input_ID_tab_products_subtab_annuities_RILA_has_GMDB(),
            "has_return_of_premium_DB": input.input_ID_tab_products_subtab_annuities_RILA_has_return_of_premium_DB(),
            "payment_frequency": input.input_ID_tab_products_subtab_annuities_RILA_payment_frequency(),
        }

    # ------------------------------------------------------------------
    # Store annuity parameters into reactives_shiny on change
    # ------------------------------------------------------------------

    @reactive.effect
    def _sync_annuity_params_to_reactives() -> None:
        """Push the latest annuity parameters into the shared reactive state."""
        if "User_Inputs_Shiny" not in reactives_shiny:
            _logger.warning("reactives_shiny missing 'User_Inputs_Shiny' key")
            return

        user_inputs = reactives_shiny["User_Inputs_Shiny"]

        # Initialise reactive.Value containers on first run
        if "Annuity_SPIA_Params" not in user_inputs:
            user_inputs["Annuity_SPIA_Params"] = reactive.value(None)
        if "Annuity_DIA_Params" not in user_inputs:
            user_inputs["Annuity_DIA_Params"] = reactive.value(None)
        if "Annuity_FIA_Params" not in user_inputs:
            user_inputs["Annuity_FIA_Params"] = reactive.value(None)
        if "Annuity_VA_Params" not in user_inputs:
            user_inputs["Annuity_VA_Params"] = reactive.value(None)
        if "Annuity_RILA_Params" not in user_inputs:
            user_inputs["Annuity_RILA_Params"] = reactive.value(None)

        user_inputs["Annuity_SPIA_Params"].set(calc_spia_params())
        user_inputs["Annuity_DIA_Params"].set(calc_dia_params())
        user_inputs["Annuity_FIA_Params"].set(calc_fia_params())
        user_inputs["Annuity_VA_Params"].set(calc_va_params())
        user_inputs["Annuity_RILA_Params"].set(calc_rila_params())
