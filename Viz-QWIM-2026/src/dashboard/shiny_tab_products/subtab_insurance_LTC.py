# ruff: noqa: N999
"""Insurance Long Term Care LTC Subtab Module for QWIM Dashboard.

Provides input forms for the three Long Term Care (LTC) insurance product
types: Traditional LTC, Hybrid Life LTC, and Hybrid Annuity LTC.

Each LTC insurance type is rendered in its own sub-subtab so users can enter
parameters independently.  All input identifiers follow the hierarchical
naming convention:

    ``input_ID_tab_products_subtab_insurance_LTC_<type>_<field>``

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
# Shared choice dictionaries — base enums
# ======================================================================

_BENEFIT_PERIOD_CHOICES: dict[str, str] = {
    "2": "2 Years",
    "3": "3 Years",
    "4": "4 Years",
    "5": "5 Years",
    "6": "6 Years",
    "100": "Lifetime",
}

_ELIMINATION_PERIOD_CHOICES: dict[str, str] = {
    "0": "0 Days (No Wait)",
    "30": "30 Days",
    "60": "60 Days",
    "90": "90 Days",
    "180": "180 Days",
    "365": "365 Days",
}

_INFLATION_PROTECTION_CHOICES: dict[str, str] = {
    "none": "None",
    "simple_3": "Simple 3 %",
    "simple_5": "Simple 5 %",
    "compound_3": "Compound 3 %",
    "compound_5": "Compound 5 %",
    "cpi_linked": "CPI-Linked",
    "future_purchase": "Future Purchase Option",
}

_CARE_SETTING_CHOICES: dict[str, str] = {
    "all": "All Care Settings",
    "nursing_home": "Nursing Home",
    "assisted_living": "Assisted Living",
    "home_health_care": "Home Health Care",
    "adult_day_care": "Adult Day Care",
    "hospice": "Hospice",
}

_PREMIUM_FREQUENCY_LTC_CHOICES: dict[str, str] = {
    "1": "Annual",
    "2": "Semi-Annual",
    "4": "Quarterly",
    "12": "Monthly",
}


# ======================================================================
# Traditional LTC — class-specific choice dictionaries
# ======================================================================

_GENDER_LTC_CHOICES: dict[str, str] = {
    "male": "Male",
    "female": "Female",
    "unisex": "Unisex",
}

_HEALTH_CLASS_LTC_CHOICES: dict[str, str] = {
    "preferred": "Preferred",
    "standard": "Standard",
    "substandard": "Substandard",
}

_NON_FORFEITURE_CHOICES: dict[str, str] = {
    "none": "None",
    "shortened_benefit": "Shortened Benefit Period",
    "return_of_premium": "Return of Premium",
    "contingent": "Contingent Non-Forfeiture",
}


# ======================================================================
# Hybrid Life LTC — class-specific choice dictionaries
# ======================================================================

_PREMIUM_STRUCTURE_HYBRID_LIFE_CHOICES: dict[str, str] = {
    "single": "Single Premium",
    "pay_5": "5-Pay",
    "pay_10": "10-Pay",
    "pay_20": "20-Pay",
}

_EXTENSION_OF_BENEFITS_CHOICES: dict[str, str] = {
    "1.0": "None (1x)",
    "2.0": "2x Extension",
    "3.0": "3x Extension",
    "4.0": "4x Extension",
}

_RETURN_OF_PREMIUM_TYPE_CHOICES: dict[str, str] = {
    "none": "None",
    "full": "Full Return of Premium",
    "graded": "Graded Return of Premium",
}


# ======================================================================
# Hybrid Annuity LTC — class-specific choice dictionaries
# ======================================================================

_LTC_MULTIPLIER_CHOICES: dict[str, str] = {
    "2.0": "2x Multiplier",
    "3.0": "3x Multiplier",
    "4.0": "4x Multiplier",
}

_ANNUITY_TYPE_HYBRID_CHOICES: dict[str, str] = {
    "fixed": "Fixed Deferred Annuity",
    "fixed_indexed": "Fixed-Indexed Annuity",
    "multi_year_guaranteed": "Multi-Year Guaranteed Annuity",
}

_SURRENDER_SCHEDULE_CHOICES: dict[str, str] = {
    "none": "None",
    "standard": "Standard (7-10 yr)",
    "short": "Short (3-5 yr)",
}


# ======================================================================
# LTC Traditional sub-subtab UI helper
# ======================================================================


def _create_LTC_traditional_ui() -> Any:
    """Build the input form for a Traditional LTC Insurance policy.

    Returns
    -------
    ui.div
        Complete Traditional LTC input card.
    """
    _pfx = "input_ID_tab_products_subtab_insurance_LTC_traditional"
    return create_enhanced_card_section(
        title="Traditional LTC Parameters",
        content=[
            # --- Core parameters ---
            create_enhanced_numeric_input(
                input_ID=f"{_pfx}_insured_age",
                label_text="Insured Age",
                min_value=18,
                max_value=85,
                step_size=1,
                default_value=55,
                suffix_symbol=" years",
                tooltip_text="Current age of the insured (18-85)",
                required_field=True,
                input_width="100%",
            ),
            create_enhanced_numeric_input(
                input_ID=f"{_pfx}_daily_benefit_amount",
                label_text="Daily Benefit Amount",
                min_value=50,
                max_value=500,
                step_size=10,
                default_value=200,
                suffix_symbol=" $",
                tooltip_text="Maximum daily benefit payable for covered care",
                required_field=True,
                input_width="100%",
            ),
            create_enhanced_select_input(
                input_ID=f"{_pfx}_gender",
                label_text="Gender",
                choices=_GENDER_LTC_CHOICES,
                default_selection="male",
                tooltip_text="Gender rating class (affects premium)",
                required_field=True,
            ),
            # --- Benefit structure ---
            create_enhanced_select_input(
                input_ID=f"{_pfx}_benefit_period",
                label_text="Benefit Period",
                choices=_BENEFIT_PERIOD_CHOICES,
                default_selection="3",
                tooltip_text="Maximum duration of benefit payments",
            ),
            create_enhanced_select_input(
                input_ID=f"{_pfx}_elimination_period",
                label_text="Elimination Period",
                choices=_ELIMINATION_PERIOD_CHOICES,
                default_selection="90",
                tooltip_text=("Waiting period (deductible) before benefits begin"),
            ),
            create_enhanced_select_input(
                input_ID=f"{_pfx}_inflation_protection",
                label_text="Inflation Protection",
                choices=_INFLATION_PROTECTION_CHOICES,
                default_selection="none",
                tooltip_text=("Rider to increase benefits annually to keep pace with inflation"),
            ),
            create_enhanced_select_input(
                input_ID=f"{_pfx}_care_setting",
                label_text="Care Setting",
                choices=_CARE_SETTING_CHOICES,
                default_selection="all",
                tooltip_text="Covered care settings for benefits",
            ),
            # --- Underwriting / rating ---
            create_enhanced_select_input(
                input_ID=f"{_pfx}_health_class",
                label_text="Health Class",
                choices=_HEALTH_CLASS_LTC_CHOICES,
                default_selection="standard",
                tooltip_text=("Health-based underwriting class (affects premium)"),
            ),
            create_enhanced_select_input(
                input_ID=f"{_pfx}_is_smoker",
                label_text="Tobacco Use",
                choices={"false": "Non-Smoker", "true": "Smoker"},
                default_selection="false",
                tooltip_text="Whether the insured uses tobacco",
            ),
            create_enhanced_select_input(
                input_ID=f"{_pfx}_is_married_discount",
                label_text="Married / Partner Discount",
                choices={"false": "No", "true": "Yes"},
                default_selection="false",
                tooltip_text=("Spousal / partner discount on premium (typically 15-40 %)"),
            ),
            # --- Riders & options ---
            create_enhanced_select_input(
                input_ID=f"{_pfx}_non_forfeiture",
                label_text="Non-Forfeiture Option",
                choices=_NON_FORFEITURE_CHOICES,
                default_selection="none",
                tooltip_text=("Guarantees partial benefits if you stop paying premiums"),
            ),
            create_enhanced_select_input(
                input_ID=f"{_pfx}_shared_care",
                label_text="Shared Care Rider",
                choices={"false": "No", "true": "Yes"},
                default_selection="false",
                tooltip_text=("Allows spouses to share benefit pools if one exhausts theirs"),
            ),
            create_enhanced_select_input(
                input_ID=f"{_pfx}_waiver_of_premium",
                label_text="Waiver of Premium",
                choices={"true": "Yes", "false": "No"},
                default_selection="true",
                tooltip_text=("Waives premiums while receiving benefits on claim"),
            ),
            # --- Premium ---
            create_enhanced_select_input(
                input_ID=f"{_pfx}_premium_frequency",
                label_text="Premium Frequency",
                choices=_PREMIUM_FREQUENCY_LTC_CHOICES,
                default_selection="1",
                tooltip_text="Number of premium payments per year",
            ),
            create_enhanced_numeric_input(
                input_ID=f"{_pfx}_cumulative_rate_increase_pct",
                label_text="Cumulative Rate Increase",
                min_value=0.0,
                max_value=5.0,
                step_size=0.1,
                default_value=0.0,
                tooltip_text=(
                    "Cumulative rate increase multiplier from prior "
                    "in-force increases (e.g. 0.25 = 25 %)"
                ),
                input_width="100%",
            ),
        ],
        icon_class="fas fa-hand-holding-medical",
        card_class="shadow-sm border-primary",
    )


# ======================================================================
# LTC Hybrid Life sub-subtab UI helper
# ======================================================================


def _create_LTC_hybrid_life_ui() -> Any:
    """Build the input form for a Hybrid Life LTC Insurance policy.

    Returns
    -------
    ui.div
        Complete Hybrid Life LTC input card.
    """
    _pfx = "input_ID_tab_products_subtab_insurance_LTC_hybrid_life"
    return create_enhanced_card_section(
        title="Hybrid Life LTC Parameters",
        content=[
            # --- Core parameters ---
            create_enhanced_numeric_input(
                input_ID=f"{_pfx}_insured_age",
                label_text="Insured Age",
                min_value=18,
                max_value=120,
                step_size=1,
                default_value=60,
                suffix_symbol=" years",
                tooltip_text="Current age of the insured",
                required_field=True,
                input_width="100%",
            ),
            create_enhanced_numeric_input(
                input_ID=f"{_pfx}_death_benefit",
                label_text="Death Benefit",
                min_value=10_000,
                max_value=5_000_000,
                step_size=10_000,
                default_value=250_000,
                suffix_symbol=" $",
                tooltip_text=("Base death benefit of the underlying life policy"),
                required_field=True,
                input_width="100%",
            ),
            create_enhanced_numeric_input(
                input_ID=f"{_pfx}_single_premium",
                label_text="Single Premium / Deposit",
                min_value=10_000,
                max_value=5_000_000,
                step_size=5_000,
                default_value=100_000,
                suffix_symbol=" $",
                tooltip_text=("Total premium commitment (paid as single or limited-pay)"),
                required_field=True,
                input_width="100%",
            ),
            # --- Premium & extension ---
            create_enhanced_select_input(
                input_ID=f"{_pfx}_premium_structure",
                label_text="Premium Structure",
                choices=_PREMIUM_STRUCTURE_HYBRID_LIFE_CHOICES,
                default_selection="single",
                tooltip_text="How premiums are paid into the policy",
            ),
            create_enhanced_select_input(
                input_ID=f"{_pfx}_extension_of_benefits",
                label_text="Extension of Benefits",
                choices=_EXTENSION_OF_BENEFITS_CHOICES,
                default_selection="2.0",
                tooltip_text=("LTC benefit multiplier on top of death benefit pool"),
            ),
            create_enhanced_select_input(
                input_ID=f"{_pfx}_return_of_premium",
                label_text="Return of Premium",
                choices=_RETURN_OF_PREMIUM_TYPE_CHOICES,
                default_selection="full",
                tooltip_text=("Guarantee type if the policy is surrendered or lapsed"),
            ),
            # --- Benefit structure (base enums) ---
            create_enhanced_select_input(
                input_ID=f"{_pfx}_benefit_period",
                label_text="Benefit Period",
                choices=_BENEFIT_PERIOD_CHOICES,
                default_selection="6",
                tooltip_text="Maximum duration of LTC benefit payments",
            ),
            create_enhanced_select_input(
                input_ID=f"{_pfx}_elimination_period",
                label_text="Elimination Period",
                choices=_ELIMINATION_PERIOD_CHOICES,
                default_selection="90",
                tooltip_text=("Waiting period before LTC benefits begin"),
            ),
            create_enhanced_select_input(
                input_ID=f"{_pfx}_inflation_protection",
                label_text="Inflation Protection",
                choices=_INFLATION_PROTECTION_CHOICES,
                default_selection="none",
                tooltip_text=("Rider to increase LTC benefits annually"),
            ),
            create_enhanced_select_input(
                input_ID=f"{_pfx}_care_setting",
                label_text="Care Setting",
                choices=_CARE_SETTING_CHOICES,
                default_selection="all",
                tooltip_text="Covered care settings for LTC benefits",
            ),
            # --- Growth & residual ---
            create_enhanced_numeric_input(
                input_ID=f"{_pfx}_cash_value_growth_rate",
                label_text="Cash Value Growth Rate",
                min_value=0.00,
                max_value=0.10,
                step_size=0.005,
                default_value=0.02,
                tooltip_text=("Annual cash value growth rate (e.g. 0.02 = 2 %)"),
                input_width="100%",
            ),
            create_enhanced_numeric_input(
                input_ID=f"{_pfx}_residual_death_benefit_pct",
                label_text="Residual Death Benefit %",
                min_value=0.00,
                max_value=1.00,
                step_size=0.05,
                default_value=0.10,
                tooltip_text=(
                    "Minimum death benefit retained during LTC claim "
                    "(e.g. 0.10 = 10 % of original DB)"
                ),
                input_width="100%",
            ),
            # --- Underwriting ---
            create_enhanced_select_input(
                input_ID=f"{_pfx}_is_smoker",
                label_text="Tobacco Use",
                choices={"false": "Non-Smoker", "true": "Smoker"},
                default_selection="false",
                tooltip_text="Whether the insured uses tobacco",
            ),
            create_enhanced_select_input(
                input_ID=f"{_pfx}_is_married_discount",
                label_text="Married / Partner Discount",
                choices={"false": "No", "true": "Yes"},
                default_selection="false",
                tooltip_text=("Spousal / partner discount on premium"),
            ),
        ],
        icon_class="fas fa-heartbeat",
        card_class="shadow-sm border-success",
    )


# ======================================================================
# LTC Hybrid Annuity sub-subtab UI helper
# ======================================================================


def _create_LTC_hybrid_annuity_ui() -> Any:
    """Build the input form for a Hybrid Annuity LTC Insurance policy.

    Returns
    -------
    ui.div
        Complete Hybrid Annuity LTC input card.
    """
    _pfx = "input_ID_tab_products_subtab_insurance_LTC_hybrid_annuity"
    return create_enhanced_card_section(
        title="Hybrid Annuity LTC Parameters",
        content=[
            # --- Core parameters ---
            create_enhanced_numeric_input(
                input_ID=f"{_pfx}_insured_age",
                label_text="Insured Age",
                min_value=18,
                max_value=120,
                step_size=1,
                default_value=62,
                suffix_symbol=" years",
                tooltip_text="Current age of the insured",
                required_field=True,
                input_width="100%",
            ),
            create_enhanced_numeric_input(
                input_ID=f"{_pfx}_single_premium",
                label_text="Single Premium / Deposit",
                min_value=10_000,
                max_value=5_000_000,
                step_size=5_000,
                default_value=150_000,
                suffix_symbol=" $",
                tooltip_text=("Lump-sum deposit into the annuity contract"),
                required_field=True,
                input_width="100%",
            ),
            # --- LTC multiplier & annuity type ---
            create_enhanced_select_input(
                input_ID=f"{_pfx}_LTC_multiplier",
                label_text="LTC Multiplier",
                choices=_LTC_MULTIPLIER_CHOICES,
                default_selection="2.0",
                tooltip_text=("Multiplier applied to deposit to determine total LTC pool"),
            ),
            create_enhanced_select_input(
                input_ID=f"{_pfx}_annuity_type",
                label_text="Annuity Type",
                choices=_ANNUITY_TYPE_HYBRID_CHOICES,
                default_selection="fixed",
                tooltip_text="Type of underlying annuity contract",
            ),
            # --- Crediting & growth ---
            create_enhanced_numeric_input(
                input_ID=f"{_pfx}_guaranteed_crediting_rate",
                label_text="Guaranteed Crediting Rate",
                min_value=0.00,
                max_value=0.10,
                step_size=0.005,
                default_value=0.02,
                tooltip_text=(
                    "Guaranteed annual crediting rate on account value (e.g. 0.02 = 2 %)"
                ),
                input_width="100%",
            ),
            # --- Benefit structure (base enums) ---
            create_enhanced_select_input(
                input_ID=f"{_pfx}_benefit_period",
                label_text="Benefit Period",
                choices=_BENEFIT_PERIOD_CHOICES,
                default_selection="6",
                tooltip_text="Maximum duration of LTC benefit payments",
            ),
            create_enhanced_select_input(
                input_ID=f"{_pfx}_elimination_period",
                label_text="Elimination Period",
                choices=_ELIMINATION_PERIOD_CHOICES,
                default_selection="90",
                tooltip_text=("Waiting period before LTC benefits begin"),
            ),
            create_enhanced_select_input(
                input_ID=f"{_pfx}_inflation_protection",
                label_text="Inflation Protection",
                choices=_INFLATION_PROTECTION_CHOICES,
                default_selection="none",
                tooltip_text=("Rider to increase LTC benefits annually"),
            ),
            create_enhanced_select_input(
                input_ID=f"{_pfx}_care_setting",
                label_text="Care Setting",
                choices=_CARE_SETTING_CHOICES,
                default_selection="all",
                tooltip_text="Covered care settings for LTC benefits",
            ),
            # --- Surrender & death benefit ---
            create_enhanced_select_input(
                input_ID=f"{_pfx}_surrender_schedule",
                label_text="Surrender Schedule",
                choices=_SURRENDER_SCHEDULE_CHOICES,
                default_selection="standard",
                tooltip_text=("Surrender charge schedule type"),
            ),
            create_enhanced_numeric_input(
                input_ID=f"{_pfx}_guaranteed_min_surrender_pct",
                label_text="Guaranteed Min Surrender %",
                min_value=0.00,
                max_value=1.00,
                step_size=0.05,
                default_value=0.90,
                tooltip_text=(
                    "Minimum guaranteed percentage of premium returned on "
                    "surrender (e.g. 0.90 = 90 %)"
                ),
                input_width="100%",
            ),
            create_enhanced_numeric_input(
                input_ID=f"{_pfx}_death_benefit_floor_pct",
                label_text="Death Benefit Floor %",
                min_value=0.00,
                max_value=2.00,
                step_size=0.05,
                default_value=1.00,
                tooltip_text=(
                    "Minimum death benefit as fraction of premium (e.g. 1.0 = 100 % of deposit)"
                ),
                input_width="100%",
            ),
            # --- Underwriting ---
            create_enhanced_select_input(
                input_ID=f"{_pfx}_is_smoker",
                label_text="Tobacco Use",
                choices={"false": "Non-Smoker", "true": "Smoker"},
                default_selection="false",
                tooltip_text="Whether the insured uses tobacco",
            ),
            create_enhanced_select_input(
                input_ID=f"{_pfx}_is_married_discount",
                label_text="Married / Partner Discount",
                choices={"false": "No", "true": "Yes"},
                default_selection="false",
                tooltip_text=("Spousal / partner discount"),
            ),
        ],
        icon_class="fas fa-piggy-bank",
        card_class="shadow-sm border-warning",
    )


# ======================================================================
# Module UI
# ======================================================================


@module.ui
def subtab_insurance_LTC_ui(
    data_utils: dict[str, Any],
    data_inputs: dict[str, Any],
) -> Any:
    """Create the LTC Insurance subtab UI with three sub-subtabs.

    Parameters
    ----------
    data_utils : dict[str, Any]
        Utility functions and configuration data.
    data_inputs : dict[str, Any]
        Default input values and validation rules.

    Returns
    -------
    ui.div
        Complete LTC Insurance subtab UI with three nav-panels.
    """
    return ui.div(
        ui.h3("Long Term Care (LTC) Insurance Products", class_="text-center mb-4"),
        ui.p(
            "Select an LTC insurance type and enter the corresponding parameters.",
            class_="text-muted text-center mb-3",
        ),
        ui.navset_tab(
            ui.nav_panel("LTC Traditional", _create_LTC_traditional_ui()),
            ui.nav_panel("LTC Hybrid Life", _create_LTC_hybrid_life_ui()),
            ui.nav_panel("LTC Hybrid Annuity", _create_LTC_hybrid_annuity_ui()),
            id="ID_tab_products_subtab_insurance_LTC_tabs_all",
        ),
    )


# ======================================================================
# Module Server
# ======================================================================


@module.server
def subtab_insurance_LTC_server(
    input: typing.Any,
    output: typing.Any,
    session: typing.Any,
    data_utils: dict[str, Any],
    data_inputs: dict[str, Any],
    reactives_shiny: dict[str, Any],
) -> None:
    """Server logic for the LTC Insurance subtab.

    Stores user-entered LTC insurance parameters into the shared reactive
    state dictionary so that downstream modules (results, reporting) can
    consume them.

    Parameters
    ----------
    input : typing.Any
        Shiny input object with reactive values from the LTC insurance forms.
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
    _logger.info("Initializing LTC Insurance subtab server")

    # ------------------------------------------------------------------
    # Reactive calculations — gather Traditional LTC inputs
    # ------------------------------------------------------------------

    @reactive.calc
    def calc_LTC_traditional_params() -> dict[str, Any]:
        """Collect Traditional LTC input values into a dictionary."""
        _pfx = "input_ID_tab_products_subtab_insurance_LTC_traditional"
        return {
            "insured_age": getattr(input, f"{_pfx}_insured_age")(),
            "daily_benefit_amount": getattr(
                input,
                f"{_pfx}_daily_benefit_amount",
            )(),
            "gender": getattr(input, f"{_pfx}_gender")(),
            "benefit_period": getattr(input, f"{_pfx}_benefit_period")(),
            "elimination_period": getattr(
                input,
                f"{_pfx}_elimination_period",
            )(),
            "inflation_protection": getattr(
                input,
                f"{_pfx}_inflation_protection",
            )(),
            "care_setting": getattr(input, f"{_pfx}_care_setting")(),
            "health_class": getattr(input, f"{_pfx}_health_class")(),
            "is_smoker": getattr(input, f"{_pfx}_is_smoker")(),
            "is_married_discount": getattr(
                input,
                f"{_pfx}_is_married_discount",
            )(),
            "non_forfeiture": getattr(
                input,
                f"{_pfx}_non_forfeiture",
            )(),
            "shared_care": getattr(input, f"{_pfx}_shared_care")(),
            "waiver_of_premium": getattr(
                input,
                f"{_pfx}_waiver_of_premium",
            )(),
            "premium_frequency": getattr(
                input,
                f"{_pfx}_premium_frequency",
            )(),
            "cumulative_rate_increase_pct": getattr(
                input,
                f"{_pfx}_cumulative_rate_increase_pct",
            )(),
        }

    # ------------------------------------------------------------------
    # Reactive calculations — gather Hybrid Life LTC inputs
    # ------------------------------------------------------------------

    @reactive.calc
    def calc_LTC_hybrid_life_params() -> dict[str, Any]:
        """Collect Hybrid Life LTC input values into a dictionary."""
        _pfx = "input_ID_tab_products_subtab_insurance_LTC_hybrid_life"
        return {
            "insured_age": getattr(input, f"{_pfx}_insured_age")(),
            "death_benefit": getattr(
                input,
                f"{_pfx}_death_benefit",
            )(),
            "single_premium": getattr(
                input,
                f"{_pfx}_single_premium",
            )(),
            "premium_structure": getattr(
                input,
                f"{_pfx}_premium_structure",
            )(),
            "extension_of_benefits": getattr(
                input,
                f"{_pfx}_extension_of_benefits",
            )(),
            "return_of_premium": getattr(
                input,
                f"{_pfx}_return_of_premium",
            )(),
            "benefit_period": getattr(input, f"{_pfx}_benefit_period")(),
            "elimination_period": getattr(
                input,
                f"{_pfx}_elimination_period",
            )(),
            "inflation_protection": getattr(
                input,
                f"{_pfx}_inflation_protection",
            )(),
            "care_setting": getattr(input, f"{_pfx}_care_setting")(),
            "cash_value_growth_rate": getattr(
                input,
                f"{_pfx}_cash_value_growth_rate",
            )(),
            "residual_death_benefit_pct": getattr(
                input,
                f"{_pfx}_residual_death_benefit_pct",
            )(),
            "is_smoker": getattr(input, f"{_pfx}_is_smoker")(),
            "is_married_discount": getattr(
                input,
                f"{_pfx}_is_married_discount",
            )(),
        }

    # ------------------------------------------------------------------
    # Reactive calculations — gather Hybrid Annuity LTC inputs
    # ------------------------------------------------------------------

    @reactive.calc
    def calc_LTC_hybrid_annuity_params() -> dict[str, Any]:
        """Collect Hybrid Annuity LTC input values into a dictionary."""
        _pfx = "input_ID_tab_products_subtab_insurance_LTC_hybrid_annuity"
        return {
            "insured_age": getattr(input, f"{_pfx}_insured_age")(),
            "single_premium": getattr(
                input,
                f"{_pfx}_single_premium",
            )(),
            "LTC_multiplier": getattr(
                input,
                f"{_pfx}_LTC_multiplier",
            )(),
            "annuity_type": getattr(input, f"{_pfx}_annuity_type")(),
            "guaranteed_crediting_rate": getattr(
                input,
                f"{_pfx}_guaranteed_crediting_rate",
            )(),
            "benefit_period": getattr(input, f"{_pfx}_benefit_period")(),
            "elimination_period": getattr(
                input,
                f"{_pfx}_elimination_period",
            )(),
            "inflation_protection": getattr(
                input,
                f"{_pfx}_inflation_protection",
            )(),
            "care_setting": getattr(input, f"{_pfx}_care_setting")(),
            "surrender_schedule": getattr(
                input,
                f"{_pfx}_surrender_schedule",
            )(),
            "guaranteed_min_surrender_pct": getattr(
                input,
                f"{_pfx}_guaranteed_min_surrender_pct",
            )(),
            "death_benefit_floor_pct": getattr(
                input,
                f"{_pfx}_death_benefit_floor_pct",
            )(),
            "is_smoker": getattr(input, f"{_pfx}_is_smoker")(),
            "is_married_discount": getattr(
                input,
                f"{_pfx}_is_married_discount",
            )(),
        }

    # ------------------------------------------------------------------
    # Store LTC insurance parameters into reactives_shiny on change
    # ------------------------------------------------------------------

    @reactive.effect
    def _sync_insurance_LTC_params_to_reactives() -> None:
        """Push the latest LTC insurance params into the shared state."""
        if "User_Inputs_Shiny" not in reactives_shiny:
            _logger.warning(
                "reactives_shiny missing 'User_Inputs_Shiny' key",
            )
            return

        user_inputs = reactives_shiny["User_Inputs_Shiny"]

        # Initialise reactive.Value containers on first run
        _keys = [
            "Insurance_LTC_Traditional_Params",
            "Insurance_LTC_Hybrid_Life_Params",
            "Insurance_LTC_Hybrid_Annuity_Params",
        ]
        for key in _keys:
            if key not in user_inputs:
                user_inputs[key] = reactive.value(None)

        user_inputs["Insurance_LTC_Traditional_Params"].set(
            calc_LTC_traditional_params(),
        )
        user_inputs["Insurance_LTC_Hybrid_Life_Params"].set(
            calc_LTC_hybrid_life_params(),
        )
        user_inputs["Insurance_LTC_Hybrid_Annuity_Params"].set(
            calc_LTC_hybrid_annuity_params(),
        )
