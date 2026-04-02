"""client Personal Information Subtab Module.

This module provides comprehensive personal information input and management
functionality for the clients tab of the QWIM financial dashboard. It creates
professional input forms for collecting and managing personal details from both
primary and partner clients.

The module includes:
- Personal information input forms
- Contact information management
- Demographic data collection
- Employment and income tracking
- Real-time validation and formatting
- Professional UI components with enhanced styling
- Responsive design for different screen sizes

Key Features:
    - Enhanced input components with comprehensive validation
    - Professional card-based layout design
    - Multi-client support (primary and partner)
    - Real-time form validation and error handling
    - Consistent styling across all components
    - Integration with enhanced UI component system

Dependencies:
    - shiny: Web application framework for Python reactive programming
    - typing: Type hints and annotations for enhanced code clarity
    - utils_enhanced_ui_components: Custom UI components with professional styling
    - utils_enhanced_formatting: Data formatting and display utilities

Architecture:
    The module implements a clean separation of concerns with dedicated functions
    for UI generation and server-side logic. All components follow enhanced UI
    patterns with consistent styling and comprehensive validation.

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

# Import enhanced UI components following project standards
from src.dashboard.shiny_utils.reactives_shiny import (
    get_value_from_shiny_input_numeric,
    get_value_from_shiny_input_text,
)
from src.dashboard.shiny_utils.utils_enhanced_ui_components import (
    create_enhanced_card_section,
    create_enhanced_numeric_input,
    create_enhanced_select_input,
    create_enhanced_text_input,
)
from src.dashboard.shiny_utils.utils_visuals import (
    create_enhanced_summary_table_multi_column,
)
from src.utils.custom_exceptions_errors_loggers.logger_custom import get_logger


#: Module-level logger instance
_logger = get_logger(__name__)


@module.ui
def subtab_clients_personal_info_ui(
    data_utils: dict[str, Any],
    data_inputs: dict[str, Any],
) -> Any:
    """Create user interface for comprehensive client personal information input.

    Generates a professional UI layout for collecting and managing personal
    information from both primary and partner clients. The interface includes
    enhanced input components with validation, professional styling, and
    comprehensive data collection capabilities.

    Args:
        data_utils (Dict[str, Any]): Utility functions and configuration data
            for the dashboard application functionality.
        data_inputs (Dict[str, Any]): Default input values and validation rules
            for personal information input components.

    Returns
    -------
        ui.div: Complete UI layout including:
            - Primary client personal information section
            - Partner client personal information section
            - Contact information forms
            - Employment and income details
            - Professional card-based layout design

    UI Structure:
        ```
        Personal Information Interface
        ├── Primary client Information
        │   ├── Basic Personal Details
        │   ├── Contact Information
        │   └── Employment Details
        ├── Partner client Information
        │   ├── Basic Personal Details
        │   ├── Contact Information
        │   └── Employment Details
        └── Summary Information
            ├── Household Overview
            └── Combined Details
        ```

    Note:
        All input components are fully editable and include comprehensive
        validation, formatting, and real-time updates. The interface
        adapts to different screen sizes for optimal user experience.

    Raises
    ------
        RuntimeError: If UI component creation fails unexpectedly
    """
    try:
        # Define common choice dictionaries for select inputs
        State_Choices = {
            "AL": "Alabama",
            "AK": "Alaska",
            "AZ": "Arizona",
            "AR": "Arkansas",
            "CA": "California",
            "CO": "Colorado",
            "CT": "Connecticut",
            "DE": "Delaware",
            "FL": "Florida",
            "GA": "Georgia",
            "HI": "Hawaii",
            "ID": "Idaho",
            "IL": "Illinois",
            "IN": "Indiana",
            "IA": "Iowa",
            "KS": "Kansas",
            "KY": "Kentucky",
            "LA": "Louisiana",
            "ME": "Maine",
            "MD": "Maryland",
            "MA": "Massachusetts",
            "MI": "Michigan",
            "MN": "Minnesota",
            "MS": "Mississippi",
            "MO": "Missouri",
            "MT": "Montana",
            "NE": "Nebraska",
            "NV": "Nevada",
            "NH": "New Hampshire",
            "NJ": "New Jersey",
            "NM": "New Mexico",
            "NY": "New York",
            "NC": "North Carolina",
            "ND": "North Dakota",
            "OH": "Ohio",
            "OK": "Oklahoma",
            "OR": "Oregon",
            "PA": "Pennsylvania",
            "RI": "Rhode Island",
            "SC": "South Carolina",
            "SD": "South Dakota",
            "TN": "Tennessee",
            "TX": "Texas",
            "UT": "Utah",
            "VT": "Vermont",
            "VA": "Virginia",
            "WA": "Washington",
            "WV": "West Virginia",
            "WI": "Wisconsin",
            "WY": "Wyoming",
        }

        Marital_Status_Choices = {
            "single": "Single",
            "married": "Married",
            "divorced": "Divorced",
            "widowed": "Widowed",
            "separated": "Separated",
            "domestic_partnership": "Domestic Partnership",
        }

        return ui.div(
            # Main section header
            ui.h3("Personal Information", class_="text-center mb-4"),
            # Primary and Partner client sections in responsive layout
            ui.row(
                # Primary client Personal Information Section
                ui.column(
                    6,
                    create_enhanced_card_section(
                        title="Primary client Information",
                        content=[
                            # Name Input (combined first and last name)
                            create_enhanced_text_input(
                                input_ID="input_ID_tab_clients_subtab_clients_personal_info_client_primary_name",
                                label_text="Name",
                                default_value="Anne Smith",
                                placeholder_text="Enter full name",
                                # help_text="Full name as it appears on official documents",
                                tooltip_text="Enter full name for client primary",
                                required_field=True,
                                max_length=100,
                            ),
                            # Current Age Input - Updated identifier following coding standards
                            create_enhanced_numeric_input(
                                input_ID="input_ID_tab_clients_subtab_clients_personal_info_client_primary_age_current",
                                label_text="Current Age",
                                min_value=18,
                                max_value=100,
                                step_size=1,
                                default_value=60,
                                currency_format=False,
                                suffix_symbol=" years",
                                # help_text="Current age in years",
                                tooltip_text="Enter current age for client primary",
                                required_field=True,
                                input_width="100%",
                            ),
                            # Retirement Age Input - Updated identifier following coding standards
                            create_enhanced_numeric_input(
                                input_ID="input_ID_tab_clients_subtab_clients_personal_info_client_primary_age_retirement",
                                label_text="Retirement Age",
                                min_value=50,
                                max_value=80,
                                step_size=1,
                                default_value=65,
                                currency_format=False,
                                suffix_symbol=" years",
                                # help_text="Planned retirement age",
                                tooltip_text="Enter planned retirement age for client primary",
                                required_field=True,
                                input_width="100%",
                            ),
                            # Income Starting Age Input - Updated identifier following coding standards
                            create_enhanced_numeric_input(
                                input_ID="input_ID_tab_clients_subtab_clients_personal_info_client_primary_age_income_starting",
                                label_text="Income Starting Age",
                                min_value=16,
                                max_value=70,
                                step_size=1,
                                default_value=65,
                                currency_format=False,
                                suffix_symbol=" years",
                                # help_text="Age when income started or will start",
                                tooltip_text="Enter age when client primary will start receiving income from retirement portfolio",
                                required_field=True,
                                input_width="100%",
                            ),
                            # Marital Status Select - Updated identifier following coding standards
                            create_enhanced_select_input(
                                input_ID="input_ID_tab_clients_subtab_clients_personal_info_client_primary_status_marital",
                                label_text="Marital Status",
                                choices=Marital_Status_Choices,
                                default_selection="married",
                                # help_text="Current marital status for tax and legal purposes",
                                tooltip_text="Select current marital status for client primary",
                                required_field=True,
                            ),
                            # Gender Select
                            create_enhanced_select_input(
                                input_ID="input_ID_tab_clients_subtab_clients_personal_info_client_primary_gender",
                                label_text="Gender",
                                choices={
                                    "male": "Male",
                                    "female": "Female",
                                    "other": "Other",
                                    "prefer_not_to_say": "Prefer Not to Say",
                                },
                                default_selection="female",
                                # help_text="Gender for demographic purposes",
                                tooltip_text="Select gender for client primary",
                                required_field=False,
                            ),
                            # Risk Tolerance Select - Updated identifier following coding standards
                            create_enhanced_select_input(
                                input_ID="input_ID_tab_clients_subtab_clients_personal_info_client_primary_tolerance_risk",
                                label_text="Risk Tolerance",
                                choices={
                                    "conservative": "Conservative",
                                    "moderate_conservative": "Moderate Conservative",
                                    "moderate": "Moderate",
                                    "moderate_aggressive": "Moderate Aggressive",
                                    "aggressive": "Aggressive",
                                },
                                default_selection="moderate",
                                # help_text="Investment risk tolerance level",
                                tooltip_text="Select risk tolerance for client primary",
                                required_field=True,
                            ),
                            # State Select
                            create_enhanced_select_input(
                                input_ID="input_ID_tab_clients_subtab_clients_personal_info_client_primary_state",
                                label_text="State",
                                choices=State_Choices,
                                default_selection="",
                                # help_text="State of residence",
                                tooltip_text="Select state of residence for client primary",
                                required_field=True,
                            ),
                            # ZIP Code Input - Updated identifier following coding standards
                            create_enhanced_numeric_input(
                                input_ID="input_ID_tab_clients_subtab_clients_personal_info_client_primary_code_zip",
                                label_text="ZIP Code",
                                min_value=1000,
                                max_value=99999,
                                step_size=1,
                                default_value=12345,
                                currency_format=False,
                                # help_text="5-digit ZIP code",
                                tooltip_text="Enter ZIP code for client primary",
                                required_field=True,
                                input_width="100%",
                            ),
                        ],
                        icon_class="fas fa-user",
                        card_class="h-100 shadow-sm border-primary",
                    ),
                ),
                # Partner client Personal Information Section
                ui.column(
                    6,
                    create_enhanced_card_section(
                        title="Partner client Information",
                        content=[
                            # Name Input (combined first and last name)
                            create_enhanced_text_input(
                                input_ID="input_ID_tab_clients_subtab_clients_personal_info_client_partner_name",
                                label_text="Name",
                                default_value="William Smith",
                                placeholder_text="Enter full name",
                                # help_text="Full name as it appears on official documents",
                                tooltip_text="Enter full name for client partner",
                                required_field=False,
                                max_length=100,
                            ),
                            # Current Age Input - Updated identifier following coding standards
                            create_enhanced_numeric_input(
                                input_ID="input_ID_tab_clients_subtab_clients_personal_info_client_partner_age_current",
                                label_text="Current Age",
                                min_value=18,
                                max_value=100,
                                step_size=1,
                                default_value=60,
                                currency_format=False,
                                suffix_symbol=" years",
                                # help_text="Current age in years",
                                tooltip_text="Enter current age for client partner",
                                required_field=False,
                                input_width="100%",
                            ),
                            # Retirement Age Input - Updated identifier following coding standards
                            create_enhanced_numeric_input(
                                input_ID="input_ID_tab_clients_subtab_clients_personal_info_client_partner_age_retirement",
                                label_text="Retirement Age",
                                min_value=50,
                                max_value=80,
                                step_size=1,
                                default_value=65,
                                currency_format=False,
                                suffix_symbol=" years",
                                # help_text="Planned retirement age",
                                tooltip_text="Enter planned retirement age for client partner",
                                required_field=False,
                                input_width="100%",
                            ),
                            # Income Starting Age Input - Updated identifier following coding standards
                            create_enhanced_numeric_input(
                                input_ID="input_ID_tab_clients_subtab_clients_personal_info_client_partner_age_income_starting",
                                label_text="Income Starting Age",
                                min_value=16,
                                max_value=70,
                                step_size=1,
                                default_value=65,
                                currency_format=False,
                                suffix_symbol=" years",
                                # help_text="Age when income started or will start",
                                tooltip_text="Enter age when client partner will start receiving income from retirement portfolio",
                                required_field=False,
                                input_width="100%",
                            ),
                            # Marital Status Select - Updated identifier following coding standards
                            create_enhanced_select_input(
                                input_ID="input_ID_tab_clients_subtab_clients_personal_info_client_partner_status_marital",
                                label_text="Marital Status",
                                choices=Marital_Status_Choices,
                                default_selection="married",
                                # help_text="Current marital status for tax and legal purposes",
                                tooltip_text="Select current marital status for client partner",
                                required_field=False,
                            ),
                            # Gender Select
                            create_enhanced_select_input(
                                input_ID="input_ID_tab_clients_subtab_clients_personal_info_client_partner_gender",
                                label_text="Gender",
                                choices={
                                    "male": "Male",
                                    "female": "Female",
                                    "other": "Other",
                                    "prefer_not_to_say": "Prefer Not to Say",
                                },
                                default_selection="male",  # THE FIX: Changed from "Male" to "male"
                                # help_text="Gender for demographic purposes",
                                tooltip_text="Select gender for client partner",
                                required_field=False,
                            ),
                            # Risk Tolerance Select - Updated identifier following coding standards
                            create_enhanced_select_input(
                                input_ID="input_ID_tab_clients_subtab_clients_personal_info_client_partner_tolerance_risk",
                                label_text="Risk Tolerance",
                                choices={
                                    "conservative": "Conservative",
                                    "moderate_conservative": "Moderate Conservative",
                                    "moderate": "Moderate",
                                    "moderate_aggressive": "Moderate Aggressive",
                                    "aggressive": "Aggressive",
                                },
                                default_selection="moderate",
                                # help_text="Investment risk tolerance level",
                                tooltip_text="Select risk tolerance for client partner",
                                required_field=False,
                            ),
                            # State Select
                            create_enhanced_select_input(
                                input_ID="input_ID_tab_clients_subtab_clients_personal_info_client_partner_state",
                                label_text="State",
                                choices=State_Choices,
                                default_selection="",
                                # help_text="State of residence",
                                tooltip_text="Select state of residence for client partner",
                                required_field=False,
                            ),
                            # ZIP Code Input - Updated identifier following coding standards
                            create_enhanced_numeric_input(
                                input_ID="input_ID_tab_clients_subtab_clients_personal_info_client_partner_code_zip",
                                label_text="ZIP Code",
                                min_value=1000,
                                max_value=99999,
                                step_size=1,
                                default_value=12345,
                                currency_format=False,
                                # help_text="5-digit ZIP code",
                                tooltip_text="Enter ZIP code for client partner",
                                required_field=False,
                                input_width="100%",
                            ),
                        ],
                        icon_class="fas fa-user-friends",
                        card_class="h-100 shadow-sm border-info",
                    ),
                ),
            ),
            ui.row(
                ui.column(
                    12,
                    ui.card(
                        ui.card_header(ui.h4("Test table", class_="text-center")),
                        ui.card_body(
                            ui.output_ui(
                                "output_ID_tab_clients_subtab_clients_personal_info_table_test",
                            ),
                            class_="text-center",
                        ),
                    ),
                ),
            ),
            ui.row(
                ui.column(
                    12,
                    ui.card(
                        ui.card_header(
                            ui.h4("client Personal Information Overview", class_="text-center"),
                        ),
                        ui.card_body(
                            ui.div(
                                ui.output_ui(
                                    "output_ID_tab_clients_subtab_clients_personal_info_table_main",
                                ),
                                style="min-height: 300px; border: 1px dashed #ccc; padding: 10px;",
                            ),
                            class_="text-center",
                        ),
                    ),
                ),
            ),
            # Add custom CSS for enhanced styling with reduced vertical spacing
            ui.tags.style("""
            /* Enhanced styling for personal information forms with 50% reduced spacing */
            .form-group {
                margin-bottom: 0.75rem !important;
            }

            .text-primary {
                color: #0d6efd !important;
            }

            .text-info {
                color: #0dcaf0 !important;
            }

            /* Input field styling */
            .form-control {
                border-radius: 0.375rem;
                border: 1px solid #ced4da;
                transition: border-color 0.15s ease-in-out, box-shadow 0.15s ease-in-out;
            }

            .form-control:focus {
                border-color: #86b7fe;
                box-shadow: 0 0 0 0.25rem rgba(13, 110, 253, 0.25);
            }

            /* Required field indicators */
            .required::after {
                content: " *";
                color: #dc3545;
            }

            /* Section dividers with reduced spacing */
            hr {
                margin: 0.75rem 0;
                opacity: 0.3;
            }

            /* Card styling enhancements */
            .card {
                transition: box-shadow 0.15s ease-in-out;
            }

            .card:hover {
                box-shadow: 0 0.5rem 1rem rgba(0, 0, 0, 0.15) !important;
            }

            /* Reduce spacing between section headers and inputs by 50% */
            .mb-3 {
                margin-bottom: 0.75rem !important;
            }

            /* Additional spacing reductions for compact layout */
            .card-body {
                padding: 1rem !important;
            }

            h5.mb-3 {
                margin-bottom: 0.5rem !important;
            }
            """),
        )

    except Exception as exc_error:
        error_message = f"Unexpected error creating personal info subtab UI: {exc_error}"
        raise RuntimeError(error_message) from exc_error


@module.server
def subtab_clients_personal_info_server(
    input: typing.Any,
    output: typing.Any,
    session: typing.Any,
    data_utils: dict[str, Any],
    data_inputs: dict[str, Any],
    reactives_shiny: dict[str, Any],
) -> None:
    """Server logic for comprehensive client personal information management.

    Implements reactive server-side logic for processing personal information
    inputs, performing validation, and managing data storage. The server handles
    form validation, data formatting, and provides integration with other
    dashboard components.

    Args:
        input (typing.Any): Shiny input object containing reactive values
            from all personal information input UI components.
        output (typing.Any): Shiny output object for rendering reactive
            content and validation messages.
        session (typing.Any): Shiny session object for managing
            application state and user interactions.
        data_utils (Dict[str, Any]): Utility functions and helper methods
            for data processing and validation.
        data_inputs (Dict[str, Any]): Default values and validation rules
            for input processing and storage.
        reactives_shiny (Dict[str, Any]): Shared reactive values and state
            management objects for cross-component communication.

    Server Functions:
        - Real-time form validation and error handling
        - Data formatting and storage management
        - Cross-component reactive updates
        - Integration with shared application state

    Reactive Behavior:
        Personal information inputs are validated in real-time and stored
        in shared reactive values for use by other dashboard components.

    Error Handling:
        Comprehensive error handling ensures graceful management of invalid
        inputs, validation errors, and provides appropriate user feedback
        when issues occur during processing.

    Note:
        This function follows the project's Shiny naming conventions and
        implements proper input validation with defensive programming
        practices for robust operation.
    """

    @reactive.calc
    def calc_table_clients_personal_info_main():
        """Generate personal information table with proper reactive dependencies."""
        # Input validation with early returns following coding standards
        # Access reactive values using proper Shiny input access patterns

        # Primary client personal information with proper reactive dependencies
        value_Primary_Name = get_value_from_shiny_input_text(
            input["input_ID_tab_clients_subtab_clients_personal_info_client_primary_name"],
        )
        value_Primary_Current_Age = get_value_from_shiny_input_numeric(
            input["input_ID_tab_clients_subtab_clients_personal_info_client_primary_age_current"],
        )
        value_Primary_Retirement_Age = get_value_from_shiny_input_numeric(
            input[
                "input_ID_tab_clients_subtab_clients_personal_info_client_primary_age_retirement"
            ],
        )
        value_Primary_Income_Start_Age = get_value_from_shiny_input_numeric(
            input[
                "input_ID_tab_clients_subtab_clients_personal_info_client_primary_age_income_starting"
            ],
        )
        value_Primary_Risk_Tolerance = get_value_from_shiny_input_text(
            input[
                "input_ID_tab_clients_subtab_clients_personal_info_client_primary_tolerance_risk"
            ],
        )
        value_Primary_Gender = get_value_from_shiny_input_text(
            input["input_ID_tab_clients_subtab_clients_personal_info_client_primary_gender"],
        )
        value_Primary_Marital_Status = get_value_from_shiny_input_text(
            input[
                "input_ID_tab_clients_subtab_clients_personal_info_client_primary_status_marital"
            ],
        )
        value_Primary_State = get_value_from_shiny_input_text(
            input["input_ID_tab_clients_subtab_clients_personal_info_client_primary_state"],
        )
        value_Primary_Zip_Code = get_value_from_shiny_input_numeric(
            input["input_ID_tab_clients_subtab_clients_personal_info_client_primary_code_zip"],
        )

        # Partner client personal information with proper reactive dependencies
        value_Partner_Name = get_value_from_shiny_input_text(
            input["input_ID_tab_clients_subtab_clients_personal_info_client_partner_name"],
        )
        value_Partner_Current_Age = get_value_from_shiny_input_numeric(
            input["input_ID_tab_clients_subtab_clients_personal_info_client_partner_age_current"],
        )
        value_Partner_Retirement_Age = get_value_from_shiny_input_numeric(
            input[
                "input_ID_tab_clients_subtab_clients_personal_info_client_partner_age_retirement"
            ],
        )
        value_Partner_Income_Start_Age = get_value_from_shiny_input_numeric(
            input[
                "input_ID_tab_clients_subtab_clients_personal_info_client_partner_age_income_starting"
            ],
        )
        value_Partner_Risk_Tolerance = get_value_from_shiny_input_text(
            input[
                "input_ID_tab_clients_subtab_clients_personal_info_client_partner_tolerance_risk"
            ],
        )
        value_Partner_Gender = get_value_from_shiny_input_text(
            input["input_ID_tab_clients_subtab_clients_personal_info_client_partner_gender"],
        )
        value_Partner_Marital_Status = get_value_from_shiny_input_text(
            input[
                "input_ID_tab_clients_subtab_clients_personal_info_client_partner_status_marital"
            ],
        )
        value_Partner_State = get_value_from_shiny_input_text(
            input["input_ID_tab_clients_subtab_clients_personal_info_client_partner_state"],
        )
        value_Partner_Zip_Code = get_value_from_shiny_input_numeric(
            input["input_ID_tab_clients_subtab_clients_personal_info_client_partner_code_zip"],
        )

        # Configuration validation - ensure all numeric values are non-negative
        value_Primary_Current_Age = max(0.0, float(value_Primary_Current_Age))
        value_Primary_Retirement_Age = max(0.0, float(value_Primary_Retirement_Age))
        value_Primary_Income_Start_Age = max(0.0, float(value_Primary_Income_Start_Age))
        value_Primary_Zip_Code = max(0.0, float(value_Primary_Zip_Code))
        value_Partner_Current_Age = max(0.0, float(value_Partner_Current_Age))
        value_Partner_Retirement_Age = max(0.0, float(value_Partner_Retirement_Age))
        value_Partner_Income_Start_Age = max(0.0, float(value_Partner_Income_Start_Age))
        value_Partner_Zip_Code = max(0.0, float(value_Partner_Zip_Code))

        # Apply defaults for text fields that might be empty (but valid)
        value_Primary_Name = str(value_Primary_Name) if value_Primary_Name else "Not provided"
        value_Primary_Risk_Tolerance = (
            str(value_Primary_Risk_Tolerance) if value_Primary_Risk_Tolerance else "Not specified"
        )
        value_Primary_Gender = (
            str(value_Primary_Gender) if value_Primary_Gender else "Not specified"
        )
        value_Primary_Marital_Status = (
            str(value_Primary_Marital_Status) if value_Primary_Marital_Status else "Not specified"
        )
        value_Primary_State = str(value_Primary_State) if value_Primary_State else "Not specified"

        value_Partner_Name = str(value_Partner_Name) if value_Partner_Name else "Not provided"
        value_Partner_Risk_Tolerance = (
            str(value_Partner_Risk_Tolerance) if value_Partner_Risk_Tolerance else "Not specified"
        )
        value_Partner_Gender = (
            str(value_Partner_Gender) if value_Partner_Gender else "Not specified"
        )
        value_Partner_Marital_Status = (
            str(value_Partner_Marital_Status) if value_Partner_Marital_Status else "Not specified"
        )
        value_Partner_State = str(value_Partner_State) if value_Partner_State else "Not specified"

        # Create polars dataframe with three columns following coding standards
        data_Personal_Info_DF = pl.DataFrame(
            {
                "Information_Category": [
                    "Name",
                    "Current Age",
                    "Retirement Age",
                    "Income Start Age",
                    "Risk Tolerance",
                    "Gender",
                    "Marital Status",
                    "State",
                    "ZIP Code",
                ],
                "client_Primary": [
                    str(value_Primary_Name),
                    f"{int(value_Primary_Current_Age)} years",
                    f"{int(value_Primary_Retirement_Age)} years",
                    f"{int(value_Primary_Income_Start_Age)} years",
                    str(value_Primary_Risk_Tolerance),
                    str(value_Primary_Gender),
                    str(value_Primary_Marital_Status),
                    str(value_Primary_State),
                    str(int(value_Primary_Zip_Code)),
                ],
                "client_Partner": [
                    str(value_Partner_Name),
                    f"{int(value_Partner_Current_Age)} years",
                    f"{int(value_Partner_Retirement_Age)} years",
                    f"{int(value_Partner_Income_Start_Age)} years",
                    str(value_Partner_Risk_Tolerance),
                    str(value_Partner_Gender),
                    str(value_Partner_Marital_Status),
                    str(value_Partner_State),
                    str(int(value_Partner_Zip_Code)),
                ],
            },
        )

        # Configuration validation - verify dataframe was created successfully
        if data_Personal_Info_DF is None or data_Personal_Info_DF.height == 0:
            raise ValueError("Failed to create data_Personal_Info_DF dataframe")

        # Return data_Personal_Info_DF with proper reactive dependencies established
        return data_Personal_Info_DF

    @output
    @render.ui
    def output_ID_tab_clients_subtab_clients_personal_info_table_main():
        """Generate personal information summary table with great-tables with reactive dependencies."""
        try:
            # Input validation with early returns - get reactive data from calculation
            data_Personal_Info_DF = calc_table_clients_personal_info_main()

            # Configuration validation - ensure dataframe exists and has data
            if data_Personal_Info_DF is None:
                raise ValueError("data_Personal_Info_DF is None from reactive calculation")

            if data_Personal_Info_DF.height == 0:
                raise ValueError("data_Personal_Info_DF is empty from reactive calculation")

            # Try to create the enhanced table
            try:
                print("Creating enhanced table...")
                table_Personal_Info = create_enhanced_summary_table_multi_column(
                    dataframe_input=data_Personal_Info_DF,
                    table_title="Personal Information Summary",
                    table_subtitle="Comprehensive demographic and personal details for both clients",
                    table_theme="professional",
                    table_width="100%",
                )

                # Configuration validation - verify table was created successfully
                if table_Personal_Info is None:
                    print("Enhanced table creation returned None")
                    raise ValueError("Failed to create table_Personal_Info using enhanced function")

                print("Enhanced table created successfully")
                print(f"Table type: {type(table_Personal_Info)}")

                table_html = table_Personal_Info.as_raw_html()
                print(f"HTML length: {len(table_html)}")

                # Return with proper container and styling
                return ui.div(
                    ui.HTML(table_html),
                    class_="table-container mt-3",
                    style="width: 100%; overflow-x: auto;",
                )

            except Exception as table_error:
                print(f"Enhanced table creation failed: {table_error}")

                # Fallback to simple HTML table if enhanced table creation fails
                def format_value_fallback(value: Any) -> Any:
                    """Format value for fallback table display."""
                    try:
                        return str(value) if value else "Not provided"
                    except (ValueError, TypeError):
                        return "Not provided"

                # Extract values safely from the dataframe
                try:
                    print("Creating fallback table...")
                    info_categories = (
                        data_Personal_Info_DF.select("Information_Category").to_series().to_list()
                    )
                    primary_values = (
                        data_Personal_Info_DF.select("client_Primary").to_series().to_list()
                    )
                    partner_values = (
                        data_Personal_Info_DF.select("client_Partner").to_series().to_list()
                    )

                    print(
                        f"Categories: {len(info_categories)}, Primary: {len(primary_values)}, Partner: {len(partner_values)}",
                    )

                    # Generate table rows using list comprehension
                    table_rows = [
                        f"""
                            <tr>
                                <td class="text-start">{format_value_fallback(info_categories[idx])}</td>
                                <td class="text-center">{format_value_fallback(primary_values[idx])}</td>
                                <td class="text-center">{format_value_fallback(partner_values[idx])}</td>
                            </tr>
                        """
                        for idx in range(len(info_categories))
                    ]

                    print("Fallback table created successfully")

                    return ui.div(
                        ui.h5("Personal Information Summary", class_="text-center mb-3"),
                        ui.p(
                            "Comprehensive demographic and personal details for both clients",
                            class_="text-muted text-center mb-3",
                        ),
                        ui.div(
                            ui.HTML(f"""
                            <div class="table-responsive">
                                <table class="table table-striped table-hover table-bordered">
                                    <thead class="table-dark">
                                        <tr>
                                            <th class="text-center">Information Category</th>
                                            <th class="text-center">client Primary</th>
                                            <th class="text-center">client Partner</th>
                                        </tr>
                                    </thead>
                                    <tbody>
                                        {"".join(table_rows)}
                                    </tbody>
                                </table>
                            </div>
                            """),
                            style="width: 100%; overflow-x: auto;",
                        ),
                        class_="mt-3",
                    )

                except Exception as fallback_error:
                    print(f"Fallback table creation failed: {fallback_error}")

                    # Ultimate fallback with hardcoded default table
                    return ui.div(
                        ui.div(
                            ui.h5(
                                "Personal Info Table - Data Processing Error",
                                class_="text-warning",
                            ),
                            ui.p(
                                f"Enhanced table error: {table_error!s}",
                                class_="text-muted small",
                            ),
                            ui.p(
                                f"Fallback error: {fallback_error!s}",
                                class_="text-muted small",
                            ),
                            ui.p(
                                "Please check that personal information has been entered in the Personal Info tab.",
                                class_="text-info",
                            ),
                            class_="alert alert-warning",
                        ),
                        ui.div(
                            ui.HTML("""
                            <div class="table-responsive">
                                <table class="table table-striped table-bordered">
                                    <thead class="table-dark">
                                        <tr>
                                            <th class="text-center">Information Category</th>
                                            <th class="text-center">client Primary</th>
                                            <th class="text-center">client Partner</th>
                                        </tr>
                                    </thead>
                                    <tbody>
                                        <tr><td class="text-start">Name</td><td class="text-center">Anne Smith</td><td class="text-center">William Smith</td></tr>
                                        <tr><td class="text-start">Current Age</td><td class="text-center">60 years</td><td class="text-center">35 years</td></tr>
                                        <tr><td class="text-start">Retirement Age</td><td class="text-center">65 years</td><td class="text-center">65 years</td></tr>
                                        <tr><td class="text-start">Income Start Age</td><td class="text-center">65 years</td><td class="text-center">65 years</td></tr>
                                        <tr><td class="text-start">Risk Tolerance</td><td class="text-center">Moderate</td><td class="text-center">Moderate</td></tr>
                                        <tr><td class="text-start">Gender</td><td class="text-center">Female</td><td class="text-center">Male</td></tr>
                                        <tr><td class="text-start">Marital Status</td><td class="text-center">Married</td><td class="text-center">Married</td></tr>
                                        <tr><td class="text-start">State</td><td class="text-center">Not specified</td><td class="text-center">Not specified</td></tr>
                                        <tr><td class="text-start">ZIP Code</td><td class="text-center">12345</td><td class="text-center">12345</td></tr>
                                    </tbody>
                                </table>
                            </div>
                            """),
                            style="width: 100%; overflow-x: auto;",
                        ),
                    )

        except Exception as exc_error:
            print(f"Major error in table generation: {exc_error}")

            # Enhanced error handling with comprehensive error information
            error_message = (
                f"Error generating personal info table: {type(exc_error).__name__}: {exc_error!s}"
            )
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
                ui.div(
                    ui.HTML("""
                    <div class="table-responsive">
                        <table class="table table-striped table-bordered">
                            <thead class="table-dark">
                                <tr>
                                    <th class="text-center">Information Category</th>
                                    <th class="text-center">client Primary</th>
                                    <th class="text-center">client Partner</th>
                                </tr>
                            </thead>
                            <tbody>
                                <tr><td colspan="3" class="text-center text-muted">Error loading personal information data</td></tr>
                            </tbody>
                        </table>
                    </div>
                    """),
                    style="width: 100%; overflow-x: auto;",
                ),
            )

    @output
    @render.ui
    def output_ID_tab_clients_subtab_clients_personal_info_table_test():
        """Generate test table content with enhanced debugging and proper UI structure.

        Creates a test table to verify output rendering functionality in the
        Personal Info subtab. Includes comprehensive debugging and follows
        project coding standards for UI component creation.

        Returns
        -------
            ui.div: Test table content with proper styling and structure
        """
        print("=== DEBUG: Personal Info test table function called ===")

        try:
            # Create enhanced test content with proper UI structure
            test_content = ui.div(
                # Header section with clear identification
                ui.div(
                    ui.h5("Test Table - Personal Info Subtab", class_="text-primary mb-3"),
                    ui.p(
                        "This is a test table to verify output rendering functionality.",
                        class_="text-muted mb-3",
                    ),
                    class_="text-center",
                ),
                # Test table with proper Bootstrap styling
                ui.div(
                    ui.HTML("""
                    <div class="table-responsive">
                        <table class="table table-striped table-hover table-bordered">
                            <thead class="table-dark">
                                <tr>
                                    <th class="text-center">Test Category</th>
                                    <th class="text-center">Test Value</th>
                                    <th class="text-center">Status</th>
                                </tr>
                            </thead>
                            <tbody>
                                <tr>
                                    <td class="text-start">Output Function</td>
                                    <td class="text-center">Working</td>
                                    <td class="text-center"><span class="badge bg-success">✓ Active</span></td>
                                </tr>
                                <tr>
                                    <td class="text-start">UI Rendering</td>
                                    <td class="text-center">Functional</td>
                                    <td class="text-center"><span class="badge bg-success">✓ Active</span></td>
                                </tr>
                                <tr>
                                    <td class="text-start">Module Integration</td>
                                    <td class="text-center">Connected</td>
                                    <td class="text-center"><span class="badge bg-success">✓ Active</span></td>
                                </tr>
                                <tr>
                                    <td class="text-start">Personal Info Subtab</td>
                                    <td class="text-center">Operational</td>
                                    <td class="text-center"><span class="badge bg-info">ℹ Ready</span></td>
                                </tr>
                            </tbody>
                        </table>
                    </div>
                    """),
                    class_="mt-3",
                ),
                # Footer with additional test information
                ui.div(
                    ui.p(
                        "If you can see this content, the output function is working correctly.",
                        class_="text-success small text-center mt-3",
                    ),
                    ui.p(
                        "Module: subtab_clients_personal_info | Function: output_table_test",
                        class_="text-muted small text-center",
                    ),
                    class_="border-top pt-2 mt-3",
                ),
                # Container styling for proper display
                class_="test-table-container p-3",
                style="background-color: #f8f9fa; border-radius: 0.375rem; border: 1px solid #dee2e6;",
            )

            print("=== DEBUG: Test table content created successfully ===")
            return test_content

        except Exception as exc_error:
            print(f"=== DEBUG: Error creating test table content: {exc_error} ===")

            # Fallback content with error information
            return ui.div(
                ui.div(
                    ui.h5("Test Table - Error", class_="text-warning"),
                    ui.p(f"Error creating test content: {exc_error!s}", class_="text-muted"),
                    class_="alert alert-warning text-center",
                ),
                ui.div(
                    "Fallback test content - If you see this, the output function is executing but encountered an error.",
                    class_="text-center p-3 bg-light border rounded",
                ),
            )

    @reactive.effect
    def observer_update_shared_reactives_shiny_personal_info() -> None:
        """Update shared reactive values for cross-module communication.

        This reactive effect automatically updates the shared reactives_shiny structure
        whenever any personal information input values change, ensuring that other modules
        (like the Summary tab) can access the current personal information values for
        calculations and displays.

        Reactive Dependencies:
            - input_ID_tab_clients_subtab_clients_personal_info_client_primary_name
            - input_ID_tab_clients_subtab_clients_personal_info_client_primary_age_current
            - input_ID_tab_clients_subtab_clients_personal_info_client_primary_age_retirement
            - input_ID_tab_clients_subtab_clients_personal_info_client_primary_age_income_starting
            - input_ID_tab_clients_subtab_clients_personal_info_client_primary_status_marital
            - input_ID_tab_clients_subtab_clients_personal_info_client_primary_gender
            - input_ID_tab_clients_subtab_clients_personal_info_client_primary_tolerance_risk
            - input_ID_tab_clients_subtab_clients_personal_info_client_primary_state
            - input_ID_tab_clients_subtab_clients_personal_info_client_primary_code_zip
            - input_ID_tab_clients_subtab_clients_personal_info_client_partner_name
            - input_ID_tab_clients_subtab_clients_personal_info_client_partner_age_current
            - input_ID_tab_clients_subtab_clients_personal_info_client_partner_age_retirement
            - input_ID_tab_clients_subtab_clients_personal_info_client_partner_age_income_starting
            - input_ID_tab_clients_subtab_clients_personal_info_client_partner_status_marital
            - input_ID_tab_clients_subtab_clients_personal_info_client_partner_gender
            - input_ID_tab_clients_subtab_clients_personal_info_client_partner_tolerance_risk
            - input_ID_tab_clients_subtab_clients_personal_info_client_partner_state
            - input_ID_tab_clients_subtab_clients_personal_info_client_partner_code_zip

        Updates:
            Updates the User_Inputs_Shiny category in reactives_shiny with current
            personal information values for cross-module communication and summary calculations.

        Error Handling:
            Uses defensive programming with comprehensive error handling to ensure
            the application continues to function even if reactive updates fail.
        """
        # Get all current personal information values using the safe input access pattern
        # Primary client information
        primary_name = get_value_from_shiny_input_text(
            input["input_ID_tab_clients_subtab_clients_personal_info_client_primary_name"],
        )
        primary_age_current = get_value_from_shiny_input_numeric(
            input["input_ID_tab_clients_subtab_clients_personal_info_client_primary_age_current"],
        )
        primary_age_retirement = get_value_from_shiny_input_numeric(
            input[
                "input_ID_tab_clients_subtab_clients_personal_info_client_primary_age_retirement"
            ],
        )
        primary_age_income_starting = get_value_from_shiny_input_numeric(
            input[
                "input_ID_tab_clients_subtab_clients_personal_info_client_primary_age_income_starting"
            ],
        )
        primary_status_marital = get_value_from_shiny_input_text(
            input[
                "input_ID_tab_clients_subtab_clients_personal_info_client_primary_status_marital"
            ],
        )
        primary_gender = get_value_from_shiny_input_text(
            input["input_ID_tab_clients_subtab_clients_personal_info_client_primary_gender"],
        )
        primary_tolerance_risk = get_value_from_shiny_input_text(
            input[
                "input_ID_tab_clients_subtab_clients_personal_info_client_primary_tolerance_risk"
            ],
        )
        primary_state = get_value_from_shiny_input_text(
            input["input_ID_tab_clients_subtab_clients_personal_info_client_primary_state"],
        )
        primary_code_zip = get_value_from_shiny_input_numeric(
            input["input_ID_tab_clients_subtab_clients_personal_info_client_primary_code_zip"],
        )

        # Partner client information
        partner_name = get_value_from_shiny_input_text(
            input["input_ID_tab_clients_subtab_clients_personal_info_client_partner_name"],
        )
        partner_age_current = get_value_from_shiny_input_numeric(
            input["input_ID_tab_clients_subtab_clients_personal_info_client_partner_age_current"],
        )
        partner_age_retirement = get_value_from_shiny_input_numeric(
            input[
                "input_ID_tab_clients_subtab_clients_personal_info_client_partner_age_retirement"
            ],
        )
        partner_age_income_starting = get_value_from_shiny_input_numeric(
            input[
                "input_ID_tab_clients_subtab_clients_personal_info_client_partner_age_income_starting"
            ],
        )
        partner_status_marital = get_value_from_shiny_input_text(
            input[
                "input_ID_tab_clients_subtab_clients_personal_info_client_partner_status_marital"
            ],
        )
        partner_gender = get_value_from_shiny_input_text(
            input["input_ID_tab_clients_subtab_clients_personal_info_client_partner_gender"],
        )
        partner_tolerance_risk = get_value_from_shiny_input_text(
            input[
                "input_ID_tab_clients_subtab_clients_personal_info_client_partner_tolerance_risk"
            ],
        )
        partner_state = get_value_from_shiny_input_text(
            input["input_ID_tab_clients_subtab_clients_personal_info_client_partner_state"],
        )
        partner_code_zip = get_value_from_shiny_input_numeric(
            input["input_ID_tab_clients_subtab_clients_personal_info_client_partner_code_zip"],
        )

        # Update shared reactive values if available
        if reactives_shiny and isinstance(reactives_shiny, dict):
            user_inputs_category = reactives_shiny.get("User_Inputs_Shiny")
            if user_inputs_category and isinstance(user_inputs_category, dict):
                # Store all personal information values in the shared reactive structure
                # Following the naming convention from reactives_shiny.py
                personal_info_mapping = {
                    "Input_Tab_clients_Subtab_clients_Personal_Info_client_Primary_Name": primary_name,
                    "Input_Tab_clients_Subtab_clients_Personal_Info_client_Primary_Age_Current": primary_age_current,
                    "Input_Tab_clients_Subtab_clients_Personal_Info_client_Primary_Age_Retirement": primary_age_retirement,
                    "Input_Tab_clients_Subtab_clients_Personal_Info_client_Primary_Age_Income_Starting": primary_age_income_starting,
                    "Input_Tab_clients_Subtab_clients_Personal_Info_client_Primary_Status_Marital": primary_status_marital,
                    "Input_Tab_clients_Subtab_clients_Personal_Info_client_Primary_Gender": primary_gender,
                    "Input_Tab_clients_Subtab_clients_Personal_Info_client_Primary_Tolerance_Risk": primary_tolerance_risk,
                    "Input_Tab_clients_Subtab_clients_Personal_Info_client_Primary_State": primary_state,
                    "Input_Tab_clients_Subtab_clients_Personal_Info_client_Primary_Code_Zip": primary_code_zip,
                    "Input_Tab_clients_Subtab_clients_Personal_Info_client_Partner_Name": partner_name,
                    "Input_Tab_clients_Subtab_clients_Personal_Info_client_Partner_Age_Current": partner_age_current,
                    "Input_Tab_clients_Subtab_clients_Personal_Info_client_Partner_Age_Retirement": partner_age_retirement,
                    "Input_Tab_clients_Subtab_clients_Personal_Info_client_Partner_Age_Income_Starting": partner_age_income_starting,
                    "Input_Tab_clients_Subtab_clients_Personal_Info_client_Partner_Status_Marital": partner_status_marital,
                    "Input_Tab_clients_Subtab_clients_Personal_Info_client_Partner_Gender": partner_gender,
                    "Input_Tab_clients_Subtab_clients_Personal_Info_client_Partner_Tolerance_Risk": partner_tolerance_risk,
                    "Input_Tab_clients_Subtab_clients_Personal_Info_client_Partner_State": partner_state,
                    "Input_Tab_clients_Subtab_clients_Personal_Info_client_Partner_Code_Zip": partner_code_zip,
                }

                # Update each reactive value safely using defensive programming
                for reactive_key, current_value in personal_info_mapping.items():
                    if reactive_key in user_inputs_category:
                        reactive_var = user_inputs_category[reactive_key]
                        if reactive_var is not None and hasattr(reactive_var, "set"):
                            reactive_var.set(current_value)
