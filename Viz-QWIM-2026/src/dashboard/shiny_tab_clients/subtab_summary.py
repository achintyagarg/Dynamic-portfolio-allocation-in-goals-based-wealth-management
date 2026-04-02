"""client Summary Subtab Module.

This module provides comprehensive summary displays for all client information
collected across the four main client subtabs. It creates professional tables
using the great-tables package to present personal information, assets, goals,
and income in an organized, visually appealing format.

The module includes:
- Comprehensive data collection from all client subtabs
- Professional table generation using great-tables
- Customizable table formatting and styling
- Real-time updates based on input changes
- Currency and percentage formatting for financial data

Key Features:
    - Single customizable function for table generation with polars dataframes
    - Four distinct tables for personal info, assets, goals, and income
    - Professional styling with enhanced readability
    - Responsive design for different screen sizes
    - Currency formatting for financial values
    - Comprehensive error handling with defensive programming

Dependencies:
    - shiny: Web application framework for Python reactive programming
    - great_tables: Professional table generation and formatting
    - polars: High-performance dataframe operations
    - typing: Type hints and annotations for enhanced code clarity

Architecture:
    The module implements a clean separation of concerns with dedicated functions
    for data collection, table generation, and UI rendering. All components follow
    enhanced UI patterns with consistent styling and comprehensive validation.

Example:
    Basic usage in the clients tab:

    ```python
from typing import Any

    from subtab_clients_summary import (
        subtab_clients_summary_ui,
        subtab_clients_summary_server
    )

    # Create the summary subtab UI component
    summary_ui = subtab_clients_summary_ui()

    # Initialize the summary subtab server logic
    def clients_server(input: Any, output: Any, session: Any) -> Any:
        subtab_clients_summary_server(input, output, session)
    ```

Note:
    This module requires all four client subtabs to be properly initialized
    for complete data collection. Tables are generated using polars dataframes
    with 1-2 columns containing both numeric and string data types.

Author:
    QWIM Development Team

Version:
    0.5.1

Last Modified:
    2026-03-01

"""

from __future__ import annotations

import typing

from typing import Any

import polars as pl

from shiny import module, reactive, render, ui

from src.dashboard.shiny_utils.reactives_shiny import get_value_from_reactives_shiny
from src.dashboard.shiny_utils.utils_visuals import (
    create_enhanced_summary_table_multi_column,
)
from src.utils.custom_exceptions_errors_loggers.logger_custom import get_logger


#: Module-level logger instance
_logger = get_logger(__name__)


@module.ui
def subtab_clients_summary_ui(data_utils: dict, data_inputs: dict) -> Any:
    """Create user interface for comprehensive client summary displays.

    Generates a comprehensive UI layout displaying four professional tables
    summarizing all client information collected from the personal info,
    assets, goals, and income subtabs. Each table uses great-tables for
    professional formatting and enhanced readability.

    Args:
        data_utils (dict): Utility functions and configuration data
            for dashboard application functionality.
        data_inputs (dict): Default input values and validation rules
            for input processing and validation.

    Returns
    -------
        shiny.ui: Complete UI layout including:
            - Personal Information Summary Table
            - Financial Assets Summary Table
            - Financial Goals Summary Table
            - Income Sources Summary Table
            - Professional styling and responsive design

    UI Structure:
        ```
        client Summary
        ├── Personal Information Table
        │   ├── Primary client Details
        │   └── Partner client Details
        ├── Financial Assets Table
        │   ├── Asset Categories by Type
        │   └── Combined Asset Totals
        ├── Financial Goals Table
        │   ├── Goal Priorities and Amounts
        │   └── Combined Goal Targets
        └── Income Sources Table
            ├── Income Categories by Source
            └── Combined Income Totals
        ```

    Note:
        All tables are generated dynamically based on current input values
        from the four client subtabs and update in real-time as inputs change.
    """
    return ui.div(
        # Main header for the client summary section
        ui.h3("client Summary", class_="text-center mb-4"),
        # Personal Information Summary Section
        ui.row(
            ui.column(
                12,
                ui.card(
                    ui.card_header(ui.h4("Personal Information Summary", class_="text-center")),
                    ui.card_body(
                        ui.output_ui(
                            "output_ID_tab_clients_subtab_clients_summary_table_personal_info",
                        ),
                        class_="text-center",
                    ),
                ),
            ),
        ),
        # Financial Assets Summary Section
        ui.row(
            ui.column(
                12,
                ui.card(
                    ui.card_header(ui.h4("Financial Assets Summary", class_="text-center")),
                    ui.card_body(
                        ui.output_ui("output_ID_tab_clients_subtab_clients_summary_table_assets"),
                        class_="text-center",
                    ),
                ),
            ),
        ),
        # Financial Goals Summary Section
        ui.row(
            ui.column(
                12,
                ui.card(
                    ui.card_header(ui.h4("Financial Goals Summary", class_="text-center")),
                    ui.card_body(
                        ui.output_ui("output_ID_tab_clients_subtab_clients_summary_table_goals"),
                        class_="text-center",
                    ),
                ),
            ),
        ),
        # Income Sources Summary Section
        ui.row(
            ui.column(
                12,
                ui.card(
                    ui.card_header(ui.h4("Income Sources Summary", class_="text-center")),
                    ui.card_body(
                        ui.output_ui("output_ID_tab_clients_subtab_clients_summary_table_income"),
                        class_="text-center",
                    ),
                ),
            ),
        ),
    )


@module.server
def subtab_clients_summary_server(
    input: typing.Any,
    output: typing.Any,
    session: typing.Any,
    data_utils: dict,
    data_inputs: dict,
    reactives_shiny: dict,
) -> None:
    """Server logic for comprehensive client summary table generation."""

    @reactive.calc
    def calc_table_summary_clients_assets():
        """Generate financial assets summary table with proper reactive dependencies.

        This reactive calculation automatically updates when any of the asset inputs
        change, ensuring the summary table stays synchronized with user inputs.

        Returns
        -------
            pl.DataFrame: Assets data with proper reactive dependencies

        Reactive Dependencies:
            - input_ID_tab_clients_subtab_clients_assets_client_primary_assets_taxable
            - input_ID_tab_clients_subtab_clients_assets_client_primary_assets_tax_deferred
            - input_ID_tab_clients_subtab_clients_assets_client_primary_assets_tax_free
            - input_ID_tab_clients_subtab_clients_assets_client_partner_assets_taxable
            - input_ID_tab_clients_subtab_clients_assets_client_partner_assets_tax_deferred
            - input_ID_tab_clients_subtab_clients_assets_client_partner_assets_tax_free
        """
        try:
            # Input validation with early returns and explicit reactive dependencies
            # Access reactive values using proper Shiny input access patterns

            # Primary client asset information with proper reactive dependencies
            # Pass the actual reactive input object to the safe function
            value_Primary_Taxable = get_value_from_reactives_shiny(
                reactives_shiny=reactives_shiny,
                key_name="Input_Tab_clients_Subtab_clients_Assets_client_Primary_Assets_Taxable",
                key_category="User_Inputs_Shiny",
            )
            value_Primary_Tax_Deferred = get_value_from_reactives_shiny(
                reactives_shiny=reactives_shiny,
                key_name="Input_Tab_clients_Subtab_clients_Assets_client_Primary_Assets_Tax_Deferred",
                key_category="User_Inputs_Shiny",
            )
            value_Primary_Tax_Free = get_value_from_reactives_shiny(
                reactives_shiny=reactives_shiny,
                key_name="Input_Tab_clients_Subtab_clients_Assets_client_Primary_Assets_Tax_Free",
                key_category="User_Inputs_Shiny",
            )

            # Partner client asset information with proper reactive dependencies
            value_Partner_Taxable = get_value_from_reactives_shiny(
                reactives_shiny=reactives_shiny,
                key_name="Input_Tab_clients_Subtab_clients_Assets_client_Partner_Assets_Taxable",
                key_category="User_Inputs_Shiny",
            )
            value_Partner_Tax_Deferred = get_value_from_reactives_shiny(
                reactives_shiny=reactives_shiny,
                key_name="Input_Tab_clients_Subtab_clients_Assets_client_Partner_Assets_Tax_Deferred",
                key_category="User_Inputs_Shiny",
            )
            value_Partner_Tax_Free = get_value_from_reactives_shiny(
                reactives_shiny=reactives_shiny,
                key_name="Input_Tab_clients_Subtab_clients_Assets_client_Partner_Assets_Tax_Free",
                key_category="User_Inputs_Shiny",
            )

            # Configuration validation - ensure all values are non-negative
            value_Primary_Taxable = max(0.0, value_Primary_Taxable)
            value_Primary_Tax_Deferred = max(0.0, value_Primary_Tax_Deferred)
            value_Primary_Tax_Free = max(0.0, value_Primary_Tax_Free)
            value_Partner_Taxable = max(0.0, value_Partner_Taxable)
            value_Partner_Tax_Deferred = max(0.0, value_Partner_Tax_Deferred)
            value_Partner_Tax_Free = max(0.0, value_Partner_Tax_Free)

            # Business logic validation - calculate totals
            value_Primary_Total = (
                value_Primary_Taxable + value_Primary_Tax_Deferred + value_Primary_Tax_Free
            )
            value_Partner_Total = (
                value_Partner_Taxable + value_Partner_Tax_Deferred + value_Partner_Tax_Free
            )

            # Calculate combined totals
            value_Combined_Taxable = value_Primary_Taxable + value_Partner_Taxable
            value_Combined_Tax_Deferred = value_Primary_Tax_Deferred + value_Partner_Tax_Deferred
            value_Combined_Tax_Free = value_Primary_Tax_Free + value_Partner_Tax_Free
            value_Combined_Total = value_Primary_Total + value_Partner_Total

            # Create polars dataframe with four columns using coding standards naming
            data_Assets_DF = pl.DataFrame(
                {
                    "Asset_Category": [
                        "Taxable Assets",
                        "Tax Deferred Assets",
                        "Tax Free Assets",
                        "Total Assets",
                    ],
                    "client_Primary": [
                        value_Primary_Taxable,
                        value_Primary_Tax_Deferred,
                        value_Primary_Tax_Free,
                        value_Primary_Total,
                    ],
                    "client_Partner": [
                        value_Partner_Taxable,
                        value_Partner_Tax_Deferred,
                        value_Partner_Tax_Free,
                        value_Partner_Total,
                    ],
                    "Combined_Total": [
                        value_Combined_Taxable,
                        value_Combined_Tax_Deferred,
                        value_Combined_Tax_Free,
                        value_Combined_Total,
                    ],
                },
            )

            # Configuration validation - verify dataframe was created successfully
            if data_Assets_DF is None or data_Assets_DF.height == 0:
                raise ValueError("Failed to create data_Assets_DF dataframe")

            # Return data_Assets_DF with proper reactive dependencies established
            return data_Assets_DF

        except Exception as exc_error:
            # Enhanced error handling with comprehensive error information
            (f"Error generating assets calculation: {type(exc_error).__name__}: {exc_error!s}")

            # Return fallback empty dataframe following coding standards naming
            return pl.DataFrame(
                {
                    "Asset_Category": ["Error"],
                    "client_Primary": [0.0],
                    "client_Partner": [0.0],
                    "Combined_Total": [0.0],
                },
            )

    @reactive.calc
    def calc_table_summary_clients_personal_info():
        """Calculate summary table for clients personal information using defensive programming.

        This function creates a summary table containing all personal information for both
        primary and partner clients, retrieving data from the reactives_shiny structure
        passed through the closure scope.

        Returns
        -------
            pl.DataFrame: Summary table with client personal information using Polars

        Raises
        ------
            ValueError: If reactives_shiny is invalid or missing required data
            KeyError: If required reactive keys are not found
        """
        # Input validation with early returns
        from src.dashboard.shiny_utils.reactives_shiny import (
            get_value_from_reactives_shiny,
            validate_reactives_shiny_structure,
        )

        validation_result, validation_message = validate_reactives_shiny_structure(reactives_shiny)
        if not validation_result:
            raise ValueError(f"Reactives structure validation failed: {validation_message}")

        # Configuration validation - ensure User_Inputs_Shiny category exists
        user_inputs_category = reactives_shiny.get("User_Inputs_Shiny")
        if user_inputs_category is None:
            available_categories = list(reactives_shiny.keys())
            raise KeyError(
                f"User_Inputs_Shiny category not found. Available categories: {available_categories}",
            )

        # Business logic validation - safely retrieve all personal information values
        # Primary client Personal Information
        primary_client_name = get_value_from_reactives_shiny(
            reactives_shiny=reactives_shiny,
            key_name="Input_Tab_clients_Subtab_clients_Personal_Info_client_Primary_Name",
            key_category="User_Inputs_Shiny",
        )

        primary_client_age_current = get_value_from_reactives_shiny(
            reactives_shiny=reactives_shiny,
            key_name="Input_Tab_clients_Subtab_clients_Personal_Info_client_Primary_Age_Current",
            key_category="User_Inputs_Shiny",
        )

        primary_client_age_retirement = get_value_from_reactives_shiny(
            reactives_shiny=reactives_shiny,
            key_name="Input_Tab_clients_Subtab_clients_Personal_Info_client_Primary_Age_Retirement",
            key_category="User_Inputs_Shiny",
        )

        primary_client_age_income_starting = get_value_from_reactives_shiny(
            reactives_shiny=reactives_shiny,
            key_name="Input_Tab_clients_Subtab_clients_Personal_Info_client_Primary_Age_Income_Starting",
            key_category="User_Inputs_Shiny",
        )

        primary_client_status_marital = get_value_from_reactives_shiny(
            reactives_shiny=reactives_shiny,
            key_name="Input_Tab_clients_Subtab_clients_Personal_Info_client_Primary_Status_Marital",
            key_category="User_Inputs_Shiny",
        )

        primary_client_gender = get_value_from_reactives_shiny(
            reactives_shiny=reactives_shiny,
            key_name="Input_Tab_clients_Subtab_clients_Personal_Info_client_Primary_Gender",
            key_category="User_Inputs_Shiny",
        )

        primary_client_tolerance_risk = get_value_from_reactives_shiny(
            reactives_shiny=reactives_shiny,
            key_name="Input_Tab_clients_Subtab_clients_Personal_Info_client_Primary_Tolerance_Risk",
            key_category="User_Inputs_Shiny",
        )

        primary_client_state = get_value_from_reactives_shiny(
            reactives_shiny=reactives_shiny,
            key_name="Input_Tab_clients_Subtab_clients_Personal_Info_client_Primary_State",
            key_category="User_Inputs_Shiny",
        )

        primary_client_code_zip = get_value_from_reactives_shiny(
            reactives_shiny=reactives_shiny,
            key_name="Input_Tab_clients_Subtab_clients_Personal_Info_client_Primary_Code_Zip",
            key_category="User_Inputs_Shiny",
        )

        # Partner client Personal Information
        partner_client_name = get_value_from_reactives_shiny(
            reactives_shiny=reactives_shiny,
            key_name="Input_Tab_clients_Subtab_clients_Personal_Info_client_Partner_Name",
            key_category="User_Inputs_Shiny",
        )

        partner_client_age_current = get_value_from_reactives_shiny(
            reactives_shiny=reactives_shiny,
            key_name="Input_Tab_clients_Subtab_clients_Personal_Info_client_Partner_Age_Current",
            key_category="User_Inputs_Shiny",
        )

        partner_client_age_retirement = get_value_from_reactives_shiny(
            reactives_shiny=reactives_shiny,
            key_name="Input_Tab_clients_Subtab_clients_Personal_Info_client_Partner_Age_Retirement",
            key_category="User_Inputs_Shiny",
        )

        partner_client_age_income_starting = get_value_from_reactives_shiny(
            reactives_shiny=reactives_shiny,
            key_name="Input_Tab_clients_Subtab_clients_Personal_Info_client_Partner_Age_Income_Starting",
            key_category="User_Inputs_Shiny",
        )

        partner_client_status_marital = get_value_from_reactives_shiny(
            reactives_shiny=reactives_shiny,
            key_name="Input_Tab_clients_Subtab_clients_Personal_Info_client_Partner_Status_Marital",
            key_category="User_Inputs_Shiny",
        )

        partner_client_gender = get_value_from_reactives_shiny(
            reactives_shiny=reactives_shiny,
            key_name="Input_Tab_clients_Subtab_clients_Personal_Info_client_Partner_Gender",
            key_category="User_Inputs_Shiny",
        )

        partner_client_tolerance_risk = get_value_from_reactives_shiny(
            reactives_shiny=reactives_shiny,
            key_name="Input_Tab_clients_Subtab_clients_Personal_Info_client_Partner_Tolerance_Risk",
            key_category="User_Inputs_Shiny",
        )

        partner_client_state = get_value_from_reactives_shiny(
            reactives_shiny=reactives_shiny,
            key_name="Input_Tab_clients_Subtab_clients_Personal_Info_client_Partner_State",
            key_category="User_Inputs_Shiny",
        )

        partner_client_code_zip = get_value_from_reactives_shiny(
            reactives_shiny=reactives_shiny,
            key_name="Input_Tab_clients_Subtab_clients_Personal_Info_client_Partner_Code_Zip",
            key_category="User_Inputs_Shiny",
        )

        # Data processing validation - ensure numeric values are properly formatted
        # Convert numeric values to integers for age fields with defensive programming
        primary_age_current_int = (
            int(primary_client_age_current) if primary_client_age_current else 0
        )
        primary_age_retirement_int = (
            int(primary_client_age_retirement) if primary_client_age_retirement else 0
        )
        primary_age_income_starting_int = (
            int(primary_client_age_income_starting) if primary_client_age_income_starting else 0
        )

        partner_age_current_int = (
            int(partner_client_age_current) if partner_client_age_current else 0
        )
        partner_age_retirement_int = (
            int(partner_client_age_retirement) if partner_client_age_retirement else 0
        )
        partner_age_income_starting_int = (
            int(partner_client_age_income_starting) if partner_client_age_income_starting else 0
        )

        # Business logic validation - create summary table structure using Polars DataFrame
        # Create comprehensive summary table with all personal information using Polars
        # Following coding standards naming convention for columns
        data_Personal_Info_DF = pl.DataFrame(
            {
                "Information_Category": [
                    "Name",
                    "Current Age",
                    "Retirement Age",
                    "Income Starting Age",
                    "Marital Status",
                    "Gender",
                    "Risk Tolerance",
                    "State",
                    "ZIP Code",
                ],
                "Primary_client": [
                    str(primary_client_name) if primary_client_name else "Not Provided",
                    f"{primary_age_current_int} years"
                    if primary_age_current_int > 0
                    else "Not Specified",
                    f"{primary_age_retirement_int} years"
                    if primary_age_retirement_int > 0
                    else "Not Specified",
                    f"{primary_age_income_starting_int} years"
                    if primary_age_income_starting_int > 0
                    else "Not Specified",
                    str(primary_client_status_marital)
                    if primary_client_status_marital
                    else "Not Specified",
                    str(primary_client_gender) if primary_client_gender else "Not Specified",
                    str(primary_client_tolerance_risk)
                    if primary_client_tolerance_risk
                    else "Not Specified",
                    str(primary_client_state) if primary_client_state else "Not Specified",
                    str(primary_client_code_zip) if primary_client_code_zip else "Not Provided",
                ],
                "Partner_client": [
                    str(partner_client_name) if partner_client_name else "Not Provided",
                    f"{partner_age_current_int} years"
                    if partner_age_current_int > 0
                    else "Not Specified",
                    f"{partner_age_retirement_int} years"
                    if partner_age_retirement_int > 0
                    else "Not Specified",
                    f"{partner_age_income_starting_int} years"
                    if partner_age_income_starting_int > 0
                    else "Not Specified",
                    str(partner_client_status_marital)
                    if partner_client_status_marital
                    else "Not Specified",
                    str(partner_client_gender) if partner_client_gender else "Not Specified",
                    str(partner_client_tolerance_risk)
                    if partner_client_tolerance_risk
                    else "Not Specified",
                    str(partner_client_state) if partner_client_state else "Not Specified",
                    str(partner_client_code_zip) if partner_client_code_zip else "Not Provided",
                ],
            },
        )

        # Configuration validation - ensure table has expected structure using Polars methods
        expected_columns = ["Information_Category", "Primary_client", "Partner_client"]
        actual_columns = data_Personal_Info_DF.columns
        if not all(col in actual_columns for col in expected_columns):
            missing_columns = [col for col in expected_columns if col not in actual_columns]
            raise ValueError(f"Summary table missing expected columns: {missing_columns}")

        # Data quality validation - ensure we have data rows using Polars methods
        if data_Personal_Info_DF.height == 0:
            raise ValueError("Summary table is empty - no personal information data available")

        return data_Personal_Info_DF

    @reactive.calc
    def calc_table_summary_clients_goals():
        """Generate financial goals summary table."""
        # Input validation with early returns and direct function calls
        # Access reactive values using proper Shiny input access patterns

        # Primary client goal information with proper reactive dependencies
        value_Primary_Essential = get_value_from_reactives_shiny(
            reactives_shiny=reactives_shiny,
            key_name="Input_Tab_clients_Subtab_clients_Goals_client_Primary_Goal_Essential",
            key_category="User_Inputs_Shiny",
        )
        value_Primary_Important = get_value_from_reactives_shiny(
            reactives_shiny=reactives_shiny,
            key_name="Input_Tab_clients_Subtab_clients_Goals_client_Primary_Goal_Important",
            key_category="User_Inputs_Shiny",
        )
        value_Primary_Aspirational = get_value_from_reactives_shiny(
            reactives_shiny=reactives_shiny,
            key_name="Input_Tab_clients_Subtab_clients_Goals_client_Primary_Goal_Aspirational",
            key_category="User_Inputs_Shiny",
        )

        # Partner client goal information with proper reactive dependencies
        value_Partner_Essential = get_value_from_reactives_shiny(
            reactives_shiny=reactives_shiny,
            key_name="Input_Tab_clients_Subtab_clients_Goals_client_Partner_Goal_Essential",
            key_category="User_Inputs_Shiny",
        )
        value_Partner_Important = get_value_from_reactives_shiny(
            reactives_shiny=reactives_shiny,
            key_name="Input_Tab_clients_Subtab_clients_Goals_client_Partner_Goal_Important",
            key_category="User_Inputs_Shiny",
        )
        value_Partner_Aspirational = get_value_from_reactives_shiny(
            reactives_shiny=reactives_shiny,
            key_name="Input_Tab_clients_Subtab_clients_Goals_client_Partner_Goal_Aspirational",
            key_category="User_Inputs_Shiny",
        )

        # Configuration validation - ensure all values are non-negative
        value_Primary_Essential = max(0.0, value_Primary_Essential)
        value_Primary_Important = max(0.0, value_Primary_Important)
        value_Primary_Aspirational = max(0.0, value_Primary_Aspirational)
        value_Partner_Essential = max(0.0, value_Partner_Essential)
        value_Partner_Important = max(0.0, value_Partner_Important)
        value_Partner_Aspirational = max(0.0, value_Partner_Aspirational)

        # Business logic validation - calculate totals with validated values
        value_Primary_Total = (
            value_Primary_Essential + value_Primary_Important + value_Primary_Aspirational
        )
        value_Partner_Total = (
            value_Partner_Essential + value_Partner_Important + value_Partner_Aspirational
        )

        # Calculate combined totals
        value_Combined_Essential = value_Primary_Essential + value_Partner_Essential
        value_Combined_Important = value_Primary_Important + value_Partner_Important
        value_Combined_Aspirational = value_Primary_Aspirational + value_Partner_Aspirational
        value_Combined_Total = value_Primary_Total + value_Partner_Total

        # Create polars dataframe with four columns using coding standards naming
        data_Goals_DF = pl.DataFrame(
            {
                "Goal_Category": [
                    "Essential Goals",
                    "Important Goals",
                    "Aspirational Goals",
                    "Total Goals",
                ],
                "client_Primary": [
                    value_Primary_Essential,
                    value_Primary_Important,
                    value_Primary_Aspirational,
                    value_Primary_Total,
                ],
                "client_Partner": [
                    value_Partner_Essential,
                    value_Partner_Important,
                    value_Partner_Aspirational,
                    value_Partner_Total,
                ],
                "Combined_Total": [
                    value_Combined_Essential,
                    value_Combined_Important,
                    value_Combined_Aspirational,
                    value_Combined_Total,
                ],
            },
        )

        # Configuration validation - verify dataframe was created successfully
        if data_Goals_DF is None or data_Goals_DF.height == 0:
            raise ValueError("Failed to create data_Goals_DF dataframe")

        # Return data_Goals_DF with proper reactive dependencies established
        return data_Goals_DF

    @reactive.calc
    def calc_table_summary_clients_income():
        """Generate income sources summary table with proper reactive dependencies.

        This reactive calculation automatically updates when any of the income inputs
        change, ensuring the summary table stays synchronized with user inputs.

        Returns
        -------
            pl.DataFrame: Income data with proper reactive dependencies

        Reactive Dependencies:
            - input_ID_tab_clients_subtab_clients_income_client_primary_income_social_security
            - input_ID_tab_clients_subtab_clients_income_client_primary_income_pension
            - input_ID_tab_clients_subtab_clients_income_client_primary_income_annuity_existing
            - input_ID_tab_clients_subtab_clients_income_client_primary_income_other
            - input_ID_tab_clients_subtab_clients_income_client_partner_income_social_security
            - input_ID_tab_clients_subtab_clients_income_client_partner_income_pension
            - input_ID_tab_clients_subtab_clients_income_client_partner_income_annuity_existing
            - input_ID_tab_clients_subtab_clients_income_client_partner_income_other
        """
        try:
            # Input validation with early returns and explicit reactive dependencies
            # Access reactive values using proper Shiny input access patterns

            # Primary client income information with proper reactive dependencies
            value_Primary_Social_Security = get_value_from_reactives_shiny(
                reactives_shiny=reactives_shiny,
                key_name="Input_Tab_clients_Subtab_clients_Income_client_Primary_Income_Social_Security",
                key_category="User_Inputs_Shiny",
            )
            value_Primary_Pension = get_value_from_reactives_shiny(
                reactives_shiny=reactives_shiny,
                key_name="Input_Tab_clients_Subtab_clients_Income_client_Primary_Income_Pension",
                key_category="User_Inputs_Shiny",
            )
            value_Primary_Annuity = get_value_from_reactives_shiny(
                reactives_shiny=reactives_shiny,
                key_name="Input_Tab_clients_Subtab_clients_Income_client_Primary_Income_Annuity_Existing",
                key_category="User_Inputs_Shiny",
            )
            value_Primary_Other = get_value_from_reactives_shiny(
                reactives_shiny=reactives_shiny,
                key_name="Input_Tab_clients_Subtab_clients_Income_client_Primary_Income_Other",
                key_category="User_Inputs_Shiny",
            )

            # Partner client income information with proper reactive dependencies
            value_Partner_Social_Security = get_value_from_reactives_shiny(
                reactives_shiny=reactives_shiny,
                key_name="Input_Tab_clients_Subtab_clients_Income_client_Partner_Income_Social_Security",
                key_category="User_Inputs_Shiny",
            )
            value_Partner_Pension = get_value_from_reactives_shiny(
                reactives_shiny=reactives_shiny,
                key_name="Input_Tab_clients_Subtab_clients_Income_client_Partner_Income_Pension",
                key_category="User_Inputs_Shiny",
            )
            value_Partner_Annuity = get_value_from_reactives_shiny(
                reactives_shiny=reactives_shiny,
                key_name="Input_Tab_clients_Subtab_clients_Income_client_Partner_Income_Annuity_Existing",
                key_category="User_Inputs_Shiny",
            )
            value_Partner_Other = get_value_from_reactives_shiny(
                reactives_shiny=reactives_shiny,
                key_name="Input_Tab_clients_Subtab_clients_Income_client_Partner_Income_Other",
                key_category="User_Inputs_Shiny",
            )

            # Configuration validation - ensure all values are non-negative
            value_Primary_Social_Security = max(0.0, value_Primary_Social_Security)
            value_Primary_Pension = max(0.0, value_Primary_Pension)
            value_Primary_Annuity = max(0.0, value_Primary_Annuity)
            value_Primary_Other = max(0.0, value_Primary_Other)
            value_Partner_Social_Security = max(0.0, value_Partner_Social_Security)
            value_Partner_Pension = max(0.0, value_Partner_Pension)
            value_Partner_Annuity = max(0.0, value_Partner_Annuity)
            value_Partner_Other = max(0.0, value_Partner_Other)

            # Business logic validation - calculate totals
            value_Primary_Total = (
                value_Primary_Social_Security
                + value_Primary_Pension
                + value_Primary_Annuity
                + value_Primary_Other
            )
            value_Partner_Total = (
                value_Partner_Social_Security
                + value_Partner_Pension
                + value_Partner_Annuity
                + value_Partner_Other
            )

            # Calculate combined totals
            value_Combined_Social_Security = (
                value_Primary_Social_Security + value_Partner_Social_Security
            )
            value_Combined_Pension = value_Primary_Pension + value_Partner_Pension
            value_Combined_Annuity = value_Primary_Annuity + value_Partner_Annuity
            value_Combined_Other = value_Primary_Other + value_Partner_Other
            value_Combined_Total = value_Primary_Total + value_Partner_Total

            # Create polars dataframe with four columns using coding standards naming
            data_Income_DF = pl.DataFrame(
                {
                    "Income_Category": [
                        "Social Security",
                        "Pension Income",
                        "Existing Annuity",
                        "Other Income",
                        "Total Income",
                    ],
                    "client_Primary": [
                        value_Primary_Social_Security,
                        value_Primary_Pension,
                        value_Primary_Annuity,
                        value_Primary_Other,
                        value_Primary_Total,
                    ],
                    "client_Partner": [
                        value_Partner_Social_Security,
                        value_Partner_Pension,
                        value_Partner_Annuity,
                        value_Partner_Other,
                        value_Partner_Total,
                    ],
                    "Combined_Total": [
                        value_Combined_Social_Security,
                        value_Combined_Pension,
                        value_Combined_Annuity,
                        value_Combined_Other,
                        value_Combined_Total,
                    ],
                },
            )

            # Configuration validation - verify dataframe was created successfully
            if data_Income_DF is None or data_Income_DF.height == 0:
                raise ValueError("Failed to create data_Income_DF dataframe")

            # Return data_Income_DF with proper reactive dependencies established
            return data_Income_DF

        except Exception as exc_error:
            # Enhanced error handling with comprehensive error information
            (f"Error generating income calculation: {type(exc_error).__name__}: {exc_error!s}")

            # Return fallback empty dataframe following coding standards naming
            return pl.DataFrame(
                {
                    "Income_Category": ["Error"],
                    "client_Primary": [0.0],
                    "client_Partner": [0.0],
                    "Combined_Total": [0.0],
                },
            )

    @output
    @render.ui
    def output_ID_tab_clients_subtab_clients_summary_table_personal_info():
        """Generate personal information summary table using great-tables with reactive dependencies."""
        # Input validation with early returns - get reactive data from calculation
        # Note: No arguments needed for reactive calculations
        data_Personal_Info_DF = calc_table_summary_clients_personal_info()

        # Configuration validation - ensure dataframe exists and has data using Polars methods
        if data_Personal_Info_DF is None:
            error_message = "data_Personal_Info_DF is None from reactive calculation"
            return ui.div(
                ui.div(
                    ui.h5("Personal Info Table Generation Error", class_="text-danger"),
                    ui.p(error_message, class_="text-muted"),
                    ui.p(
                        "Please check that personal information has been entered in the Personal Info tab.",
                        class_="text-info",
                    ),
                    class_="alert alert-warning",
                ),
                ui.HTML("""
                <div class="table-responsive">
                    <table class="table table-striped">
                        <thead class="table-dark">
                            <tr>
                                <th>Information Category</th>
                                <th>Primary client</th>
                                <th>Partner client</th>
                            </tr>
                        </thead>
                        <tbody>
                            <tr><td colspan="3" class="text-center text-muted">Error loading personal information data</td></tr>
                        </tbody>
                    </table>
                </div>
                """),
            )

        if data_Personal_Info_DF.height == 0:
            error_message = "data_Personal_Info_DF is empty from reactive calculation"
            return ui.div(
                ui.div(
                    ui.h5("Personal Info Table Generation Error", class_="text-danger"),
                    ui.p(error_message, class_="text-muted"),
                    ui.p(
                        "Please check that personal information has been entered in the Personal Info tab.",
                        class_="text-info",
                    ),
                    class_="alert alert-warning",
                ),
                ui.HTML("""
                <div class="table-responsive">
                    <table class="table table-striped">
                        <thead class="table-dark">
                            <tr>
                                <th>Information Category</th>
                                <th>Primary client</th>
                                <th>Partner client</th>
                            </tr>
                        </thead>
                        <tbody>
                            <tr><td colspan="3" class="text-center text-muted">Error loading personal information data</td></tr>
                        </tbody>
                    </table>
                </div>
                """),
            )

        primary_values = data_Personal_Info_DF.select("Primary_client").to_series().to_list()
        partner_values = data_Personal_Info_DF.select("Partner_client").to_series().to_list()

        # Check if we have meaningful data (not just defaults)
        # Fixed: Use consistent case matching with DataFrame values
        has_meaningful_data = any(
            val not in ["Not Provided", "Not Specified", "0 years", "0"]
            for val in primary_values + partner_values
        )

        # If no meaningful data, show a message table instead of empty table
        if not has_meaningful_data:
            return ui.div(
                ui.p(
                    "No personal information has been entered yet. Please visit the Personal Info tab to input values.",
                    class_="text-muted text-center mt-3",
                ),
                ui.HTML("""
                <div class="table-responsive">
                    <table class="table table-striped table-hover">
                        <thead class="table-dark">
                            <tr>
                                <th>Information Category</th>
                                <th>Primary client</th>
                                <th>Partner client</th>
                            </tr>
                        </thead>
                        <tbody>
                            <tr>
                                <td>Name</td>
                                <td class="text-muted">Not Provided</td>
                                <td class="text-muted">Not Provided</td>
                            </tr>
                            <tr>
                                <td>Current Age</td>
                                <td class="text-muted">Not Specified</td>
                                <td class="text-muted">Not Specified</td>
                            </tr>
                            <tr>
                                <td>Retirement Age</td>
                                <td class="text-muted">Not Specified</td>
                                <td class="text-muted">Not Specified</td>
                            </tr>
                            <tr>
                                <td>Income Starting Age</td>
                                <td class="text-muted">Not Specified</td>
                                <td class="text-muted">Not Specified</td>
                            </tr>
                            <tr>
                                <td>Marital Status</td>
                                <td class="text-muted">Not Specified</td>
                                <td class="text-muted">Not Specified</td>
                            </tr>
                            <tr>
                                <td>Gender</td>
                                <td class="text-muted">Not Specified</td>
                                <td class="text-muted">Not Specified</td>
                            </tr>
                            <tr>
                                <td>Risk Tolerance</td>
                                <td class="text-muted">Not Specified</td>
                                <td class="text-muted">Not Specified</td>
                            </tr>
                            <tr>
                                <td>State</td>
                                <td class="text-muted">Not Specified</td>
                                <td class="text-muted">Not Specified</td>
                            </tr>
                            <tr>
                                <td>ZIP Code</td>
                                <td class="text-muted">Not Provided</td>
                                <td class="text-muted">Not Provided</td>
                            </tr>
                        </tbody>
                    </table>
                </div>
                """),
                class_="mt-3",
            )

        # Try to create the enhanced table
        table_Personal_Info = create_enhanced_summary_table_multi_column(
            dataframe_input=data_Personal_Info_DF,
            table_title="Personal Information Summary",
            table_subtitle="Comprehensive demographic and personal details for both clients",
            table_theme="professional",
            table_width="100%",
        )

        # Configuration validation - verify table was created successfully
        if table_Personal_Info is None:
            # Fallback to simple HTML table if enhanced table creation fails
            def format_value_fallback(value: Any) -> Any:
                """Format value for fallback table display."""
                return str(value) if value else "Not Provided"

            # Extract values safely from the dataframe using Polars methods
            info_categories = (
                data_Personal_Info_DF.select("Information_Category").to_series().to_list()
            )
            primary_values = data_Personal_Info_DF.select("Primary_client").to_series().to_list()
            partner_values = data_Personal_Info_DF.select("Partner_client").to_series().to_list()

            # Generate table rows using list comprehension
            table_rows = [
                f"""
                    <tr>
                        <td>{format_value_fallback(info_categories[idx])}</td>
                        <td>{format_value_fallback(primary_values[idx])}</td>
                        <td>{format_value_fallback(partner_values[idx])}</td>
                    </tr>
                """
                for idx in range(len(info_categories))
            ]

            return ui.div(
                ui.h5("Personal Information Summary", class_="text-center mb-3"),
                ui.p(
                    "Comprehensive demographic and personal details for both clients",
                    class_="text-muted text-center mb-3",
                ),
                ui.HTML(f"""
                <div class="table-responsive">
                    <table class="table table-striped table-hover">
                        <thead class="table-dark">
                            <tr>
                                <th>Information Category</th>
                                <th>Primary client</th>
                                <th>Partner client</th>
                            </tr>
                        </thead>
                        <tbody>
                            {"".join(table_rows)}
                        </tbody>
                    </table>
                </div>
                """),
                ui.p(
                    "Note: Using fallback table format due to enhanced table creation issue.",
                    class_="text-muted small mt-2",
                ),
                class_="mt-3",
            )

        # Return the GT table as HTML
        return ui.HTML(table_Personal_Info.as_raw_html())

    @output
    @render.ui
    def output_ID_tab_clients_subtab_clients_summary_table_assets():
        """Generate financial assets summary table using great-tables with reactive dependencies."""
        try:
            # Input validation with early returns - get reactive data from calculation
            data_Assets_DF = calc_table_summary_clients_assets()

            # Configuration validation - ensure dataframe exists and has data
            if data_Assets_DF is None:
                raise ValueError("data_Assets_DF is None from reactive calculation")

            if data_Assets_DF.height == 0:
                raise ValueError("data_Assets_DF is empty from reactive calculation")

            # Try to create the enhanced table
            try:
                Assets_Table = create_enhanced_summary_table_multi_column(
                    dataframe_input=data_Assets_DF,
                    table_title="Financial Assets Summary",
                    table_subtitle="Current asset values organized by tax treatment and client",
                    currency_columns=["client_Primary", "client_Partner", "Combined_Total"],
                    table_theme="professional",
                    table_width="100%",
                )

                # Configuration validation - verify table was created successfully
                if Assets_Table is None:
                    raise ValueError("Failed to create Assets_Table using enhanced function")

                # Return the GT table as HTML
                return ui.HTML(Assets_Table.as_raw_html())

            except Exception as table_error:
                # Fallback to simple HTML table if enhanced table creation fails
                def format_currency_fallback(amount: Any) -> str | None:
                    """Format currency for fallback table."""
                    try:
                        return f"${amount:,.0f}" if amount != 0 else "$0"
                    except (ValueError, TypeError):
                        return "$0"

                # Extract values safely from the dataframe
                try:
                    asset_categories = data_Assets_DF.select("Asset_Category").to_series().to_list()
                    primary_values = data_Assets_DF.select("client_Primary").to_series().to_list()
                    partner_values = data_Assets_DF.select("client_Partner").to_series().to_list()
                    combined_values = data_Assets_DF.select("Combined_Total").to_series().to_list()

                    # Generate table rows
                    table_rows = []
                    for idx in range(len(asset_categories)):
                        row_class = "table-info fw-bold" if "Total" in asset_categories[idx] else ""
                        table_rows.append(f"""
                            <tr class="{row_class}">
                                <td>{asset_categories[idx]}</td>
                                <td class="text-end">{format_currency_fallback(primary_values[idx])}</td>
                                <td class="text-end">{format_currency_fallback(partner_values[idx])}</td>
                                <td class="text-end">{format_currency_fallback(combined_values[idx])}</td>
                            </tr>
                        """)

                    return ui.div(
                        ui.h5("Financial Assets Summary", class_="text-center mb-3"),
                        ui.p(
                            "Current asset values organized by tax treatment and client",
                            class_="text-muted text-center mb-3",
                        ),
                        ui.HTML(f"""
                        <div class="table-responsive">
                            <table class="table table-striped table-hover">
                                <thead class="table-dark">
                                    <tr>
                                        <th>Asset Category</th>
                                        <th>client Primary</th>
                                        <th>client Partner</th>
                                        <th>Combined Total</th>
                                    </tr>
                                </thead>
                                <tbody>
                                    {"".join(table_rows)}
                                </tbody>
                            </table>
                        </div>
                        """),
                        class_="mt-3",
                    )

                except Exception as fallback_error:
                    # Ultimate fallback if even simple table generation fails
                    return ui.div(
                        ui.div(
                            ui.h5("Assets Table - Data Processing Error", class_="text-warning"),
                            ui.p(
                                f"Enhanced table error: {table_error!s}",
                                class_="text-muted small",
                            ),
                            ui.p(
                                f"Fallback error: {fallback_error!s}",
                                class_="text-muted small",
                            ),
                            ui.p(
                                "Please check that asset values have been entered in the Assets tab.",
                                class_="text-info",
                            ),
                            class_="alert alert-warning",
                        ),
                        ui.HTML("""
                        <div class="table-responsive">
                            <table class="table table-striped">
                                <thead class="table-dark">
                                    <tr>
                                        <th>Asset Category</th>
                                        <th>client Primary</th>
                                        <th>client Partner</th>
                                        <th>Combined Total</th>
                                    </tr>
                                </thead>
                                <tbody>
                                    <tr><td colspan="4" class="text-center text-muted">Unable to load assets data</td></tr>
                                </tbody>
                            </table>
                        </div>
                        """),
                    )

        except Exception as exc_error:
            # Enhanced error handling with comprehensive error information
            error_message = (
                f"Error generating assets table: {type(exc_error).__name__}: {exc_error!s}"
            )
            return ui.div(
                ui.div(
                    ui.h5("Assets Table Generation Error", class_="text-danger"),
                    ui.p(error_message, class_="text-muted"),
                    ui.p(
                        "Please check that asset values have been entered in the Assets tab.",
                        class_="text-info",
                    ),
                    class_="alert alert-warning",
                ),
                ui.HTML("""
                <div class="table-responsive">
                    <table class="table table-striped">
                        <thead class="table-dark">
                            <tr>
                                <th>Asset Category</th>
                                <th>client Primary</th>
                                <th>client Partner</th>
                                <th>Combined Total</th>
                            </tr>
                        </thead>
                        <tbody>
                            <tr><td colspan="4" class="text-center text-muted">Error loading assets data</td></tr>
                        </tbody>
                    </table>
                </div>
                """),
            )

    @output
    @render.ui
    def output_ID_tab_clients_subtab_clients_summary_table_goals():
        """Generate financial goals summary table using great-tables with reactive dependencies."""
        try:
            # Input validation with early returns - get reactive data from calculation
            data_Goals_DF = calc_table_summary_clients_goals()

            # Configuration validation - ensure dataframe exists and has data
            if data_Goals_DF is None:
                raise ValueError("data_Goals_DF is None from reactive calculation")

            if data_Goals_DF.height == 0:
                raise ValueError("data_Goals_DF is empty from reactive calculation")

            # Try to create the enhanced table
            try:
                table_Goals = create_enhanced_summary_table_multi_column(
                    dataframe_input=data_Goals_DF,
                    table_title="Financial Goals Summary",
                    table_subtitle="Annual goal targets organized by priority level and importance",
                    currency_columns=["client_Primary", "client_Partner", "Combined_Total"],
                    table_theme="professional",
                    table_width="100%",
                )

                # Configuration validation - verify table was created successfully
                if table_Goals is None:
                    raise ValueError("Failed to create table_Goals using enhanced function")

                # Return the GT table as HTML
                return ui.HTML(table_Goals.as_raw_html())

            except Exception as table_error:
                # Fallback to simple HTML table if enhanced table creation fails
                def format_currency_fallback(amount: Any) -> str | None:
                    """Format currency for fallback table."""
                    try:
                        return f"${amount:,.0f}" if amount != 0 else "$0"
                    except (ValueError, TypeError):
                        return "$0"

                # Extract values safely from the dataframe
                try:
                    goal_categories = data_Goals_DF.select("Goal_Category").to_series().to_list()
                    primary_values = data_Goals_DF.select("client_Primary").to_series().to_list()
                    partner_values = data_Goals_DF.select("client_Partner").to_series().to_list()
                    combined_values = data_Goals_DF.select("Combined_Total").to_series().to_list()

                    # Generate table rows
                    table_rows = []
                    for idx in range(len(goal_categories)):
                        row_class = "table-info fw-bold" if "Total" in goal_categories[idx] else ""
                        table_rows.append(f"""
                            <tr class="{row_class}">
                                <td>{goal_categories[idx]}</td>
                                <td class="text-end">{format_currency_fallback(primary_values[idx])}</td>
                                <td class="text-end">{format_currency_fallback(partner_values[idx])}</td>
                                <td class="text-end">{format_currency_fallback(combined_values[idx])}</td>
                            </tr>
                        """)

                    return ui.div(
                        ui.h5("Financial Goals Summary", class_="text-center mb-3"),
                        ui.p(
                            "Annual goal targets organized by priority level and importance",
                            class_="text-muted text-center mb-3",
                        ),
                        ui.HTML(f"""
                        <div class="table-responsive">
                            <table class="table table-striped table-hover">
                                <thead class="table-dark">
                                    <tr>
                                        <th>Goal Category</th>
                                        <th>client Primary</th>
                                        <th>client Partner</th>
                                        <th>Combined Total</th>
                                    </tr>
                                </thead>
                                <tbody>
                                    {"".join(table_rows)}
                                </tbody>
                            </table>
                        </div>
                        """),
                        class_="mt-3",
                    )

                except Exception as fallback_error:
                    # Ultimate fallback if even simple table generation fails
                    return ui.div(
                        ui.div(
                            ui.h5("Goals Table - Data Processing Error", class_="text-warning"),
                            ui.p(
                                f"Enhanced table error: {table_error!s}",
                                class_="text-muted small",
                            ),
                            ui.p(
                                f"Fallback error: {fallback_error!s}",
                                class_="text-muted small",
                            ),
                            ui.p(
                                "Please check that goal values have been entered in the Goals tab.",
                                class_="text-info",
                            ),
                            class_="alert alert-warning",
                        ),
                        ui.HTML("""
                        <div class="table-responsive">
                            <table class="table table-striped">
                                <thead class="table-dark">
                                    <tr>
                                        <th>Goal Category</th>
                                        <th>client Primary</th>
                                        <th>client Partner</th>
                                        <th>Combined Total</th>
                                    </tr>
                                </thead>
                                <tbody>
                                    <tr><td colspan="4" class="text-center text-muted">Unable to load goals data</td></tr>
                                </tbody>
                            </table>
                        </div>
                        """),
                    )

        except Exception as exc_error:
            # Enhanced error handling with comprehensive error information
            error_message = (
                f"Error generating goals table: {type(exc_error).__name__}: {exc_error!s}"
            )
            return ui.div(
                ui.div(
                    ui.h5("Goals Table Generation Error", class_="text-danger"),
                    ui.p(error_message, class_="text-muted"),
                    ui.p(
                        "Please check that goal values have been entered in the Goals tab.",
                        class_="text-info",
                    ),
                    class_="alert alert-warning",
                ),
                ui.HTML("""
                <div class="table-responsive">
                    <table class="table table-striped">
                        <thead class="table-dark">
                            <tr>
                                <th>Goal Category</th>
                                <th>client Primary</th>
                                <th>client Partner</th>
                                <th>Combined Total</th>
                            </tr>
                        </thead>
                        <tbody>
                            <tr><td colspan="4" class="text-center text-muted">Error loading goals data</td></tr>
                        </tbody>
                    </table>
                </div>
                """),
            )

    @output
    @render.ui
    def output_ID_tab_clients_subtab_clients_summary_table_income():
        """Generate income sources summary table using great-tables with reactive dependencies."""
        try:
            # Input validation with early returns - get reactive data from calculation
            data_Income_DF = calc_table_summary_clients_income()

            # Configuration validation - ensure dataframe exists and has data
            if data_Income_DF is None:
                raise ValueError("data_Income_DF is None from reactive calculation")

            if data_Income_DF.height == 0:
                raise ValueError("data_Income_DF is empty from reactive calculation")

            # Try to create the enhanced table
            try:
                Income_Table = create_enhanced_summary_table_multi_column(
                    dataframe_input=data_Income_DF,
                    table_title="Income Sources Summary",
                    table_subtitle="Annual income projections from all sources for retirement planning",
                    currency_columns=["client_Primary", "client_Partner", "Combined_Total"],
                    table_theme="professional",
                    table_width="100%",
                )

                # Configuration validation - verify table was created successfully
                if Income_Table is None:
                    raise ValueError("Failed to create Income_Table using enhanced function")

                # Return the GT table as HTML
                return ui.HTML(Income_Table.as_raw_html())

            except Exception as table_error:
                # Fallback to simple HTML table if enhanced table creation fails
                def format_currency_fallback(amount: Any) -> str | None:
                    """Format currency for fallback table."""
                    try:
                        return f"${amount:,.0f}" if amount != 0 else "$0"
                    except (ValueError, TypeError):
                        return "$0"

                # Extract values safely from the dataframe
                try:
                    income_categories = (
                        data_Income_DF.select("Income_Category").to_series().to_list()
                    )
                    primary_values = data_Income_DF.select("client_Primary").to_series().to_list()
                    partner_values = data_Income_DF.select("client_Partner").to_series().to_list()
                    combined_values = data_Income_DF.select("Combined_Total").to_series().to_list()

                    # Generate table rows
                    table_rows = []
                    for idx in range(len(income_categories)):
                        row_class = (
                            "table-info fw-bold" if "Total" in income_categories[idx] else ""
                        )
                        table_rows.append(f"""
                            <tr class="{row_class}">
                                <td>{income_categories[idx]}</td>
                                <td class="text-end">{format_currency_fallback(primary_values[idx])}</td>
                                <td class="text-end">{format_currency_fallback(partner_values[idx])}</td>
                                <td class="text-end">{format_currency_fallback(combined_values[idx])}</td>
                            </tr>
                        """)

                    return ui.div(
                        ui.h5("Income Sources Summary", class_="text-center mb-3"),
                        ui.p(
                            "Annual income projections from all sources for retirement planning",
                            class_="text-muted text-center mb-3",
                        ),
                        ui.HTML(f"""
                        <div class="table-responsive">
                            <table class="table table-striped table-hover">
                                <thead class="table-dark">
                                    <tr>
                                        <th>Income Category</th>
                                        <th>client Primary</th>
                                        <th>client Partner</th>
                                        <th>Combined Total</th>
                                    </tr>
                                </thead>
                                <tbody>
                                    {"".join(table_rows)}
                                </tbody>
                            </table>
                        </div>
                        """),
                        class_="mt-3",
                    )

                except Exception as fallback_error:
                    # Ultimate fallback if even simple table generation fails
                    return ui.div(
                        ui.div(
                            ui.h5("Income Table - Data Processing Error", class_="text-warning"),
                            ui.p(
                                f"Enhanced table error: {table_error!s}",
                                class_="text-muted small",
                            ),
                            ui.p(
                                f"Fallback error: {fallback_error!s}",
                                class_="text-muted small",
                            ),
                            ui.p(
                                "Please check that income values have been entered in the Income tab.",
                                class_="text-info",
                            ),
                            class_="alert alert-warning",
                        ),
                        ui.HTML("""
                        <div class="table-responsive">
                            <table class="table table-striped">
                                <thead class="table-dark">
                                    <tr>
                                        <th>Income Category</th>
                                        <th>client Primary</th>
                                        <th>client Partner</th>
                                        <th>Combined Total</th>
                                    </tr>
                                </thead>
                                <tbody>
                                    <tr><td colspan="4" class="text-center text-muted">Unable to load income data</td></tr>
                                </tbody>
                            </table>
                        </div>
                        """),
                    )

        except Exception as exc_error:
            # Enhanced error handling with comprehensive error information
            error_message = (
                f"Error generating income table: {type(exc_error).__name__}: {exc_error!s}"
            )
            return ui.div(
                ui.div(
                    ui.h5("Income Table Generation Error", class_="text-danger"),
                    ui.p(error_message, class_="text-muted"),
                    ui.p(
                        "Please check that income values have been entered in the Income tab.",
                        class_="text-info",
                    ),
                    class_="alert alert-warning",
                ),
                ui.HTML("""
                <div class="table-responsive">
                    <table class="table table-striped">
                        <thead class="table-dark">
                            <tr>
                                <th>Income Category</th>
                                <th>client Primary</th>
                                <th>client Partner</th>
                                <th>Combined Total</th>
                            </tr>
                        </thead>
                        <tbody>
                            <tr><td colspan="4" class="text-center text-muted">Error loading income data</td></tr>
                        </tbody>
                    </table>
                </div>
                """),
            )

    @reactive.effect
    def observer_update_shared_reactives_shiny_summary() -> None:
        """Update shared reactive values for cross-module communication."""
        data_Assets_DF = calc_table_summary_clients_assets()

        table_Assets = create_enhanced_summary_table_multi_column(
            dataframe_input=data_Assets_DF,
            table_title="Financial Assets Summary",
            table_subtitle="Current asset values organized by tax treatment and client",
            currency_columns=["client_Primary", "client_Partner", "Combined_Total"],
            table_theme="professional",
            table_width="100%",
        )

        data_Income_DF = calc_table_summary_clients_income()

        table_Income = create_enhanced_summary_table_multi_column(
            dataframe_input=data_Income_DF,
            table_title="Income Sources Summary",
            table_subtitle="Annual income projections from all sources for retirement planning",
            currency_columns=["client_Primary", "client_Partner", "Combined_Total"],
            table_theme="professional",
            table_width="100%",
        )

        data_Personal_Info_DF = calc_table_summary_clients_personal_info()

        table_Personal_Info = create_enhanced_summary_table_multi_column(
            dataframe_input=data_Personal_Info_DF,
            table_title="Personal Information Summary",
            table_subtitle="Comprehensive demographic and personal details for both clients",
            table_theme="professional",
            table_width="100%",
        )

        data_Goals_DF = calc_table_summary_clients_goals()

        table_Goals = create_enhanced_summary_table_multi_column(
            dataframe_input=data_Goals_DF,
            table_title="Financial Goals Summary",
            table_subtitle="Annual goal targets organized by priority level and importance",
            currency_columns=["client_Primary", "client_Partner", "Combined_Total"],
            table_theme="professional",
            table_width="100%",
        )

        # Update shared reactive values if available
        if reactives_shiny and isinstance(reactives_shiny, dict):
            visual_objects_category = reactives_shiny.get("Visual_Objects_Shiny")
            inner_variables_category = reactives_shiny.get("Inner_Variables_Shiny")

            if visual_objects_category and isinstance(visual_objects_category, dict):
                # Store all asset values in the shared reactive structure
                visual_objects_mapping = {
                    "Table_Assets": table_Assets,
                    "Table_Personal_Info": table_Personal_Info,
                    "Table_Goals": table_Goals,
                    "Table_Income": table_Income,
                }

                # Update each reactive value safely
                for reactive_key, current_value in visual_objects_mapping.items():
                    if reactive_key in visual_objects_category:
                        reactive_var = visual_objects_category[reactive_key]
                        if reactive_var is not None and hasattr(reactive_var, "set"):
                            reactive_var.set(current_value)

            if inner_variables_category and isinstance(inner_variables_category, dict):
                # Store all asset values in the shared reactive structure
                inner_variables_mapping = {
                    "Data_Assets_DF": data_Assets_DF,
                    "Data_Personal_Info_DF": data_Personal_Info_DF,
                    "Data_Goals_DF": data_Goals_DF,
                    "Data_Income_DF": data_Income_DF,
                }

                # Update each reactive value safely
                for reactive_key, current_value in inner_variables_mapping.items():
                    if reactive_key in inner_variables_category:
                        reactive_var = inner_variables_category[reactive_key]
                        if reactive_var is not None and hasattr(reactive_var, "set"):
                            reactive_var.set(current_value)
