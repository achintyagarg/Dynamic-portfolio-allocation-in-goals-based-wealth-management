"""client Income Subtab Module.

This module provides comprehensive income input and management functionality for
the clients tab of the QWIM financial dashboard. It creates professional
input forms for collecting and managing income information from both primary
and partner clients across four income categories: Social Security, Pension,
Existing Annuity, and Other Income.

The module includes:
- Social Security income input and management
- Pension income input and management
- Existing Annuity income input and management
- Other income input and management
- Real-time calculation and validation
- Professional UI components with enhanced styling
- Responsive design for different screen sizes

Key Features:
    - Enhanced numeric input components with currency formatting
    - Real-time validation and error handling
    - Professional card-based layout design
    - Four-category income classification system
    - Multi-client support (primary and partner)
    - Automatic income calculations and summaries
    - Integration with summary reporting systems

Dependencies:
    - shiny: Web application framework for Python reactive programming
    - typing: Type hints and annotations for enhanced code clarity
    - utils_enhanced_ui_components: Custom UI components with professional styling
    - utils_enhanced_formatting: Financial formatting and display utilities

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
from src.dashboard.shiny_utils.reactives_shiny import get_value_from_shiny_input_numeric
from src.dashboard.shiny_utils.utils_enhanced_formatting import (
    extract_numeric_from_currency_string,
    format_currency_value,
)
from src.dashboard.shiny_utils.utils_enhanced_ui_components import (
    ComponentVariant,
    create_enhanced_card_section,
    create_enhanced_numeric_input,
    create_enhanced_summary_display,
)
from src.dashboard.shiny_utils.utils_enhanced_validation import (
    validate_and_constrain_numeric_value,
)
from src.dashboard.shiny_utils.utils_visuals import (
    create_enhanced_summary_table_multi_column,
)
from src.utils.custom_exceptions_errors_loggers.logger_custom import get_logger


#: Module-level logger instance
_logger = get_logger(__name__)


# Update string identifiers to conform to coding standards
@module.ui
def subtab_clients_income_ui(data_utils: dict[str, Any], data_inputs: dict[str, Any]) -> Any:
    """Create user interface for comprehensive client income input and management.

    Generates a professional UI layout for collecting and managing income
    information from both primary and partner clients. The interface includes
    enhanced input components with validation, professional styling, and real-time
    calculation capabilities across four income categories: Social Security, Pension,
    Existing Annuity, and Other Income.

    Args:
        data_utils (Dict[str, Any]): Utility functions and configuration data
            for the dashboard application functionality.
        data_inputs (Dict[str, Any]): Default input values and validation rules
            for income input components and form initialization.

    Returns
    -------
        ui.div: Complete UI layout including:
            - Primary client income input section
            - Partner client income input section
            - Four income category inputs per client
            - Real-time summary displays and totals
            - Professional card-based layout design

    UI Structure:
        ```
        Income Management Interface
        ├── Primary client Income
        │   ├── Social Security Income Input
        │   ├── Pension Income Input
        │   ├── Existing Annuity Income Input
        │   └── Other Income Input
        ├── Partner client Income
        │   ├── Social Security Income Input
        │   ├── Pension Income Input
        │   ├── Existing Annuity Income Input
        │   └── Other Income Input
        └── Income Summary Displays
            ├── Total Income for client Primary
            ├── Combined Total Income
            └── Total Income for client Partner
        ```

    Note:
        All input components are fully editable and include comprehensive
        validation, currency formatting, and real-time updates. The interface
        adapts to different screen sizes for optimal user experience.

    Raises
    ------
        RuntimeError: If UI component creation fails unexpectedly
    """
    try:
        return ui.div(
            # Main section header
            ui.h3("Income Information", class_="text-center mb-4"),
            # Primary and Partner client sections in responsive layout
            ui.row(
                # Primary client Income Section
                ui.column(
                    6,
                    create_enhanced_card_section(
                        title="Primary client Income",
                        content=[
                            # Social Security Income Input - Updated identifier following coding standards
                            create_enhanced_numeric_input(
                                input_ID="input_ID_tab_clients_subtab_clients_income_client_primary_income_social_security",
                                label_text="Social Security Income",
                                min_value=0,
                                max_value=5000000,
                                step_size=100,
                                default_value=0,
                                currency_format=True,
                                prefix_symbol="$",
                                # help_text="Annual Social Security benefits received",
                                tooltip_text="Enter annual Social Security income",
                                required_field=False,
                                disabled_state=False,
                                input_width="100%",
                            ),
                            # Pension Income Input - Updated identifier following coding standards
                            create_enhanced_numeric_input(
                                input_ID="input_ID_tab_clients_subtab_clients_income_client_primary_income_pension",
                                label_text="Pension Income",
                                min_value=0,
                                max_value=5000000,
                                step_size=100,
                                default_value=0,
                                currency_format=True,
                                prefix_symbol="$",
                                # help_text="Annual pension income from retirement plans",
                                tooltip_text="Enter annual pension income",
                                required_field=False,
                                disabled_state=False,
                                input_width="100%",
                            ),
                            # Existing Annuity Income Input - Updated identifier following coding standards
                            create_enhanced_numeric_input(
                                input_ID="input_ID_tab_clients_subtab_clients_income_client_primary_income_annuity_existing",
                                label_text="Existing Annuity Income",
                                min_value=0,
                                max_value=5000000,
                                step_size=100,
                                default_value=0,
                                currency_format=True,
                                prefix_symbol="$",
                                # help_text="Annual income from existing annuity contracts",
                                tooltip_text="Enter existing annuity income",
                                required_field=False,
                                disabled_state=False,
                                input_width="100%",
                            ),
                            # Other Income Input - Updated identifier following coding standards
                            create_enhanced_numeric_input(
                                input_ID="input_ID_tab_clients_subtab_clients_income_client_primary_income_other",
                                label_text="Other Income",
                                min_value=0,
                                max_value=5000000,
                                step_size=100,
                                default_value=0,
                                currency_format=True,
                                prefix_symbol="$",
                                # help_text="Annual income from other sources not listed above",
                                tooltip_text="Enter other income sources",
                                required_field=False,
                                disabled_state=False,
                                input_width="100%",
                            ),
                        ],
                        icon_class="fas fa-dollar-sign",
                        card_class="h-100 shadow-sm border-primary",
                    ),
                ),
                # Partner client Income Section
                ui.column(
                    6,
                    create_enhanced_card_section(
                        title="Partner client Income",
                        content=[
                            # Social Security Income Input - Updated identifier following coding standards
                            create_enhanced_numeric_input(
                                input_ID="input_ID_tab_clients_subtab_clients_income_client_partner_income_social_security",
                                label_text="Social Security Income",
                                min_value=0,
                                max_value=5000000,
                                step_size=100,
                                default_value=0,
                                currency_format=True,
                                prefix_symbol="$",
                                # help_text="Annual Social Security benefits received",
                                tooltip_text="Enter annual Social Security income",
                                required_field=False,
                                disabled_state=False,
                                input_width="100%",
                            ),
                            # Pension Income Input - Updated identifier following coding standards
                            create_enhanced_numeric_input(
                                input_ID="input_ID_tab_clients_subtab_clients_income_client_partner_income_pension",
                                label_text="Pension Income",
                                min_value=0,
                                max_value=5000000,
                                step_size=100,
                                default_value=0,
                                currency_format=True,
                                prefix_symbol="$",
                                # help_text="Annual pension income from retirement plans",
                                tooltip_text="Enter annual pension income",
                                required_field=False,
                                disabled_state=False,
                                input_width="100%",
                            ),
                            # Existing Annuity Income Input - Updated identifier following coding standards
                            create_enhanced_numeric_input(
                                input_ID="input_ID_tab_clients_subtab_clients_income_client_partner_income_annuity_existing",
                                label_text="Existing Annuity Income",
                                min_value=0,
                                max_value=5000000,
                                step_size=100,
                                default_value=0,
                                currency_format=True,
                                prefix_symbol="$",
                                # help_text="Annual income from existing annuity contracts",
                                tooltip_text="Enter existing annuity income",
                                required_field=False,
                                disabled_state=False,
                                input_width="100%",
                            ),
                            # Other Income Input - Updated identifier following coding standards
                            create_enhanced_numeric_input(
                                input_ID="input_ID_tab_clients_subtab_clients_income_client_partner_income_other",
                                label_text="Other Income",
                                min_value=0,
                                max_value=5000000,
                                step_size=100,
                                default_value=0,
                                currency_format=True,
                                prefix_symbol="$",
                                # help_text="Annual income from other sources not listed above",
                                tooltip_text="Enter other income sources",
                                required_field=False,
                                disabled_state=False,
                                input_width="100%",
                            ),
                        ],
                        icon_class="fas fa-money-bill-wave",
                        card_class="h-100 shadow-sm border-info",
                    ),
                ),
            ),
            # Income Summary Section - Updated identifiers following coding standards
            ui.row(
                # Total Income for client Primary (leftmost)
                ui.column(
                    4,
                    create_enhanced_summary_display(
                        summary_id="output_ID_tab_clients_subtab_clients_income_client_primary_total",
                        title="Total Income for client Primary",
                        icon_class="fas fa-calculator",
                        background_class="bg-light",
                        border_variant=ComponentVariant.SUCCESS,
                    ),
                ),
                # Combined Total Income (middle)
                ui.column(
                    4,
                    create_enhanced_summary_display(
                        summary_id="output_ID_tab_clients_subtab_clients_income_combined_total",
                        title="Combined Total Income",
                        icon_class="fas fa-coins",
                        background_class="bg-success",
                        text_class="text-white",
                        border_variant=ComponentVariant.SUCCESS,
                    ),
                ),
                # Total Income for client Partner (rightmost)
                ui.column(
                    4,
                    create_enhanced_summary_display(
                        summary_id="output_ID_tab_clients_subtab_clients_income_client_partner_total",
                        title="Total Income for client Partner",
                        icon_class="fas fa-calculator",
                        background_class="bg-light",
                        border_variant=ComponentVariant.INFO,
                    ),
                ),
            ),
            ui.row(
                ui.column(
                    12,
                    ui.card(
                        ui.card_header(ui.h4("client Income Overview", class_="text-center")),
                        ui.card_body(
                            ui.div(
                                ui.output_ui(
                                    "output_ID_tab_clients_subtab_clients_income_table_main",
                                ),
                                style="min-height: 300px; border: 1px dashed #ccc; padding: 10px;",
                            ),
                            class_="text-center",
                        ),
                    ),
                ),
            ),
            # Add custom CSS to ensure inputs are fully interactive
            ui.tags.style("""
            /* Ensure all numeric inputs are fully editable */
            input[type="number"] {
                pointer-events: auto !important;
                user-select: text !important;
                background-color: #ffffff !important;
                opacity: 1 !important;
                cursor: text !important;
            }

            /* Ensure input containers are interactive */
            .form-group {
                pointer-events: auto !important;
            }

            /* Style focused inputs */
            input[type="number"]:focus {
                border-color: #86b7fe !important;
                box-shadow: 0 0 0 0.25rem rgba(13, 110, 253, 0.25) !important;
                outline: 0 !important;
            }

            /* Ensure input labels are properly styled */
            .form-label {
                pointer-events: none;
                user-select: none;
                cursor: default;
            }

            /* Enhanced styling for currency inputs */
            .currency-input input[type="number"] {
                text-align: right;
                font-weight: 500;
            }

            /* Card styling enhancements */
            .card {
                transition: box-shadow 0.15s ease-in-out;
            }

            .card:hover {
                box-shadow: 0 0.5rem 1rem rgba(0, 0, 0, 0.15) !important;
            }
            """),
        )

    except Exception as exc_error:
        error_message = f"Unexpected error creating income subtab UI: {exc_error}"
        raise RuntimeError(error_message) from exc_error


@module.server
def subtab_clients_income_server(
    input: typing.Any,
    output: typing.Any,
    session: typing.Any,
    data_utils: dict[str, Any],
    data_inputs: dict[str, Any],
    reactives_shiny: dict[str, Any],
) -> None:
    """Server logic for comprehensive client income management and calculations.

    Implements reactive server-side logic for processing income inputs, performing
    real-time calculations, and generating summary displays. The server handles
    validation, formatting, and provides immediate feedback to users as they
    input income information for the four income categories: Social Security, Pension,
    Existing Annuity, and Other Income.

    Args:
        input (typing.Any): Shiny input object containing reactive values
            from all income input UI components.
        output (typing.Any): Shiny output object for rendering reactive
            content to summary display components.
        session (typing.Any): Shiny session object for managing
            application state and user interactions.
        data_utils (Dict[str, Any]): Utility functions and helper methods
            for calculations and data processing.
        data_inputs (Dict[str, Any]): Default values and validation rules
            for input processing and validation.
        reactives_shiny (Dict[str, Any]): Shared reactive values and state
            management objects for cross-component communication.

    Server Functions:
        - Real-time income total calculations
        - Input validation and error handling
        - Summary display generation and formatting
        - Cross-component reactive updates

    Reactive Behavior:
        All summary outputs automatically update when any income input value
        changes, providing immediate feedback and real-time calculations
        for enhanced user experience.

    Error Handling:
        Comprehensive error handling ensures graceful management of invalid
        inputs, calculation errors, and provides appropriate user feedback
        when issues occur during processing.

    Note:
        This function follows the project's Shiny naming conventions and
        implements proper input validation with early returns and defensive
        programming practices for robust operation.
    """

    # Reactive observers for currency formatting with proper input/display separation - Updated identifiers following coding standards
    @reactive.effect
    @reactive.event(
        input.input_ID_tab_clients_subtab_clients_income_client_primary_income_social_security,
        ignore_none=True,
    )
    def observer_update_primary_social_security_formatting() -> None:
        """Update currency formatting for primary social security income input."""
        try:
            # Get current raw value directly from input
            current_value = input.input_ID_tab_clients_subtab_clients_income_client_primary_income_social_security()

            if current_value is None:
                return

            # Convert to string for processing
            current_str = str(current_value).strip()

            # Skip if empty
            if not current_str:
                return

            # Extract numeric value (this is the actual input value we want to preserve)
            numeric_value = extract_numeric_from_currency_string(current_str)

            # Validate and constrain
            constrained_value = validate_and_constrain_numeric_value(numeric_value, 0, 5000000)

            # Format for display (this is what the user sees)
            formatted_value = format_currency_value(constrained_value)

            # Only update display if different from current display
            if current_str != formatted_value:
                ui.update_text(
                    "input_ID_tab_clients_subtab_clients_income_client_primary_income_social_security",
                    value=formatted_value,
                )

        except Exception:
            # Silently handle errors to prevent disruption
            pass

    @reactive.effect
    @reactive.event(
        input.input_ID_tab_clients_subtab_clients_income_client_primary_income_pension,
        ignore_none=True,
    )
    def observer_update_primary_pension_formatting() -> None:
        """Update currency formatting for primary pension income input."""
        try:
            # Get current raw value directly from input
            current_value = (
                input.input_ID_tab_clients_subtab_clients_income_client_primary_income_pension()
            )

            if current_value is None:
                return

            # Convert to string for processing
            current_str = str(current_value).strip()

            # Skip if empty
            if not current_str:
                return

            # Extract numeric value (this is the actual input value we want to preserve)
            numeric_value = extract_numeric_from_currency_string(current_str)

            # Validate and constrain
            constrained_value = validate_and_constrain_numeric_value(numeric_value, 0, 5000000)

            # Format for display (this is what the user sees)
            formatted_value = format_currency_value(constrained_value)

            # Only update display if different from current display
            if current_str != formatted_value:
                ui.update_text(
                    "input_ID_tab_clients_subtab_clients_income_client_primary_income_pension",
                    value=formatted_value,
                )

        except Exception:
            # Silently handle errors to prevent disruption
            pass

    @reactive.effect
    @reactive.event(
        input.input_ID_tab_clients_subtab_clients_income_client_primary_income_annuity_existing,
        ignore_none=True,
    )
    def observer_update_primary_existing_annuity_formatting() -> None:
        """Update currency formatting for primary existing annuity income input."""
        try:
            # Get current raw value directly from input
            current_value = input.input_ID_tab_clients_subtab_clients_income_client_primary_income_annuity_existing()

            if current_value is None:
                return

            # Convert to string for processing
            current_str = str(current_value).strip()

            # Skip if empty
            if not current_str:
                return

            # Extract numeric value (this is the actual input value we want to preserve)
            numeric_value = extract_numeric_from_currency_string(current_str)

            # Validate and constrain
            constrained_value = validate_and_constrain_numeric_value(numeric_value, 0, 5000000)

            # Format for display (this is what the user sees)
            formatted_value = format_currency_value(constrained_value)

            # Only update display if different from current display
            if current_str != formatted_value:
                ui.update_text(
                    "input_ID_tab_clients_subtab_clients_income_client_primary_income_annuity_existing",
                    value=formatted_value,
                )

        except Exception:
            # Silently handle errors to prevent disruption
            pass

    @reactive.effect
    @reactive.event(
        input.input_ID_tab_clients_subtab_clients_income_client_primary_income_other,
        ignore_none=True,
    )
    def observer_update_primary_other_formatting() -> None:
        """Update currency formatting for primary other income input."""
        try:
            # Get current raw value directly from input
            current_value = (
                input.input_ID_tab_clients_subtab_clients_income_client_primary_income_other()
            )

            if current_value is None:
                return

            # Convert to string for processing
            current_str = str(current_value).strip()

            # Skip if empty
            if not current_str:
                return

            # Extract numeric value (this is the actual input value we want to preserve)
            numeric_value = extract_numeric_from_currency_string(current_str)

            # Validate and constrain
            constrained_value = validate_and_constrain_numeric_value(numeric_value, 0, 5000000)

            # Format for display (this is what the user sees)
            formatted_value = format_currency_value(constrained_value)

            # Only update display if different from current display
            if current_str != formatted_value:
                ui.update_text(
                    "input_ID_tab_clients_subtab_clients_income_client_primary_income_other",
                    value=formatted_value,
                )

        except Exception:
            # Silently handle errors to prevent disruption
            pass

    @reactive.effect
    @reactive.event(
        input.input_ID_tab_clients_subtab_clients_income_client_partner_income_social_security,
        ignore_none=True,
    )
    def observer_update_partner_social_security_formatting() -> None:
        """Update currency formatting for partner social security income input."""
        try:
            # Get current raw value directly from input
            current_value = input.input_ID_tab_clients_subtab_clients_income_client_partner_income_social_security()

            if current_value is None:
                return

            # Convert to string for processing
            current_str = str(current_value).strip()

            # Skip if empty
            if not current_str:
                return

            # Extract numeric value (this is the actual input value we want to preserve)
            numeric_value = extract_numeric_from_currency_string(current_str)

            # Validate and constrain
            constrained_value = validate_and_constrain_numeric_value(numeric_value, 0, 5000000)

            # Format for display (this is what the user sees)
            formatted_value = format_currency_value(constrained_value)

            # Only update display if different from current display
            if current_str != formatted_value:
                ui.update_text(
                    "input_ID_tab_clients_subtab_clients_income_client_partner_income_social_security",
                    value=formatted_value,
                )

        except Exception:
            # Silently handle errors to prevent disruption
            pass

    @reactive.effect
    @reactive.event(
        input.input_ID_tab_clients_subtab_clients_income_client_partner_income_pension,
        ignore_none=True,
    )
    def observer_update_partner_pension_formatting() -> None:
        """Update currency formatting for partner pension income input."""
        try:
            # Get current raw value directly from input
            current_value = (
                input.input_ID_tab_clients_subtab_clients_income_client_partner_income_pension()
            )

            if current_value is None:
                return

            # Convert to string for processing
            current_str = str(current_value).strip()

            # Skip if empty
            if not current_str:
                return

            # Extract numeric value (this is the actual input value we want to preserve)
            numeric_value = extract_numeric_from_currency_string(current_str)

            # Validate and constrain
            constrained_value = validate_and_constrain_numeric_value(numeric_value, 0, 5000000)

            # Format for display (this is what the user sees)
            formatted_value = format_currency_value(constrained_value)

            # Only update display if different from current display
            if current_str != formatted_value:
                ui.update_text(
                    "input_ID_tab_clients_subtab_clients_income_client_partner_income_pension",
                    value=formatted_value,
                )

        except Exception:
            # Silently handle errors to prevent disruption
            pass

    @reactive.effect
    @reactive.event(
        input.input_ID_tab_clients_subtab_clients_income_client_partner_income_annuity_existing,
        ignore_none=True,
    )
    def observer_update_partner_existing_annuity_formatting() -> None:
        """Update currency formatting for partner existing annuity income input."""
        try:
            # Get current raw value directly from input
            current_value = input.input_ID_tab_clients_subtab_clients_income_client_partner_income_annuity_existing()

            if current_value is None:
                return

            # Convert to string for processing
            current_str = str(current_value).strip()

            # Skip if empty
            if not current_str:
                return

            # Extract numeric value (this is the actual input value we want to preserve)
            numeric_value = extract_numeric_from_currency_string(current_str)

            # Validate and constrain
            constrained_value = validate_and_constrain_numeric_value(numeric_value, 0, 5000000)

            # Format for display (this is what the user sees)
            formatted_value = format_currency_value(constrained_value)

            # Only update display if different from current display
            if current_str != formatted_value:
                ui.update_text(
                    "input_ID_tab_clients_subtab_clients_income_client_partner_income_annuity_existing",
                    value=formatted_value,
                )

        except Exception:
            # Silently handle errors to prevent disruption
            pass

    @reactive.effect
    @reactive.event(
        input.input_ID_tab_clients_subtab_clients_income_client_partner_income_other,
        ignore_none=True,
    )
    def observer_update_partner_other_formatting() -> None:
        """Update currency formatting for partner other income input."""
        try:
            # Get current raw value directly from input
            current_value = (
                input.input_ID_tab_clients_subtab_clients_income_client_partner_income_other()
            )

            if current_value is None:
                return

            # Convert to string for processing
            current_str = str(current_value).strip()

            # Skip if empty
            if not current_str:
                return

            # Extract numeric value (this is the actual input value we want to preserve)
            numeric_value = extract_numeric_from_currency_string(current_str)

            # Validate and constrain
            constrained_value = validate_and_constrain_numeric_value(numeric_value, 0, 5000000)

            # Format for display (this is what the user sees)
            formatted_value = format_currency_value(constrained_value)

            # Only update display if different from current display
            if current_str != formatted_value:
                ui.update_text(
                    "input_ID_tab_clients_subtab_clients_income_client_partner_income_other",
                    value=formatted_value,
                )

        except Exception:
            # Silently handle errors to prevent disruption
            pass

    @output
    @render.text
    def output_ID_tab_clients_subtab_clients_income_client_primary_total():
        """Calculate and display total income for primary client.

        Returns
        -------
            str: Formatted currency string showing total primary client income (dollars only)
        """
        try:
            # Get all primary client income values (numeric values, not display strings) - Updated identifiers following coding standards
            value_Primary_Social_Security = get_value_from_shiny_input_numeric(
                input[
                    "input_ID_tab_clients_subtab_clients_income_client_primary_income_social_security"
                ],
            )
            value_Primary_Pension = get_value_from_shiny_input_numeric(
                input["input_ID_tab_clients_subtab_clients_income_client_primary_income_pension"],
            )
            value_Primary_Existing_Annuity = get_value_from_shiny_input_numeric(
                input[
                    "input_ID_tab_clients_subtab_clients_income_client_primary_income_annuity_existing"
                ],
            )
            value_Primary_Other = get_value_from_shiny_input_numeric(
                input["input_ID_tab_clients_subtab_clients_income_client_primary_income_other"],
            )

            # Calculate total (numeric calculation)
            value_Primary_Total = (
                value_Primary_Social_Security
                + value_Primary_Pension
                + value_Primary_Existing_Annuity
                + value_Primary_Other
            )

            # Format for display (dollars only, no cents)
            return format_currency_value(value_Primary_Total)

        except Exception as exc_error:
            return f"Error: {exc_error!s}"

    @output
    @render.text
    def output_ID_tab_clients_subtab_clients_income_client_partner_total():
        """Calculate and display total income for partner client.

        Returns
        -------
            str: Formatted currency string showing total partner client income (dollars only)
        """
        try:
            # Get all partner client income values (numeric values, not display strings) - Updated identifiers following coding standards
            value_Partner_Social_Security = get_value_from_shiny_input_numeric(
                input[
                    "input_ID_tab_clients_subtab_clients_income_client_partner_income_social_security"
                ],
            )
            value_Partner_Pension = get_value_from_shiny_input_numeric(
                input["input_ID_tab_clients_subtab_clients_income_client_partner_income_pension"],
            )

            value_Partner_Existing_Annuity = get_value_from_shiny_input_numeric(
                input[
                    "input_ID_tab_clients_subtab_clients_income_client_partner_income_annuity_existing"
                ],
            )
            value_Partner_Other = get_value_from_shiny_input_numeric(
                input["input_ID_tab_clients_subtab_clients_income_client_partner_income_other"],
            )
            # Calculate total (numeric calculation)
            value_Partner_Total = (
                value_Partner_Social_Security
                + value_Partner_Pension
                + value_Partner_Existing_Annuity
                + value_Partner_Other
            )

            # Format for display (dollars only, no cents)
            return format_currency_value(value_Partner_Total)

        except Exception as exc_error:
            return f"Error: {exc_error!s}"

    @output
    @render.text
    def output_ID_tab_clients_subtab_clients_income_combined_total():
        """Calculate and display combined total income for both clients.

        Returns
        -------
            str: Formatted currency string showing combined total income (dollars only)
        """
        try:
            # Get all income values (numeric values, not display strings) - Updated identifiers following coding standards
            value_Primary_Social_Security = get_value_from_shiny_input_numeric(
                input[
                    "input_ID_tab_clients_subtab_clients_income_client_primary_income_social_security"
                ],
            )
            value_Primary_Pension = get_value_from_shiny_input_numeric(
                input["input_ID_tab_clients_subtab_clients_income_client_primary_income_pension"],
            )
            value_Primary_Existing_Annuity = get_value_from_shiny_input_numeric(
                input[
                    "input_ID_tab_clients_subtab_clients_income_client_primary_income_annuity_existing"
                ],
            )
            value_Primary_Other = get_value_from_shiny_input_numeric(
                input["input_ID_tab_clients_subtab_clients_income_client_primary_income_other"],
            )

            # Get all partner client income values
            value_Partner_Social_Security = get_value_from_shiny_input_numeric(
                input[
                    "input_ID_tab_clients_subtab_clients_income_client_partner_income_social_security"
                ],
            )
            value_Partner_Pension = get_value_from_shiny_input_numeric(
                input["input_ID_tab_clients_subtab_clients_income_client_partner_income_pension"],
            )
            value_Partner_Existing_Annuity = get_value_from_shiny_input_numeric(
                input[
                    "input_ID_tab_clients_subtab_clients_income_client_partner_income_annuity_existing"
                ],
            )
            value_Partner_Other = get_value_from_shiny_input_numeric(
                input["input_ID_tab_clients_subtab_clients_income_client_partner_income_other"],
            )

            # Calculate combined total (numeric calculation)
            value_Combined_Total = (
                value_Primary_Social_Security
                + value_Primary_Pension
                + value_Primary_Existing_Annuity
                + value_Primary_Other
                + value_Partner_Social_Security
                + value_Partner_Pension
                + value_Partner_Existing_Annuity
                + value_Partner_Other
            )

            # Format for display (dollars only, no cents)
            return format_currency_value(value_Combined_Total)

        except Exception as exc_error:
            return f"Error: {exc_error!s}"

    @reactive.calc
    def calc_table_clients_income_main():
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
            value_Primary_Social_Security = get_value_from_shiny_input_numeric(
                input[
                    "input_ID_tab_clients_subtab_clients_income_client_primary_income_social_security"
                ],
            )
            value_Primary_Pension = get_value_from_shiny_input_numeric(
                input["input_ID_tab_clients_subtab_clients_income_client_primary_income_pension"],
            )
            value_Primary_Annuity = get_value_from_shiny_input_numeric(
                input[
                    "input_ID_tab_clients_subtab_clients_income_client_primary_income_annuity_existing"
                ],
            )
            value_Primary_Other = get_value_from_shiny_input_numeric(
                input["input_ID_tab_clients_subtab_clients_income_client_primary_income_other"],
            )

            # Partner client income information with proper reactive dependencies
            value_Partner_Social_Security = get_value_from_shiny_input_numeric(
                input[
                    "input_ID_tab_clients_subtab_clients_income_client_partner_income_social_security"
                ],
            )
            value_Partner_Pension = get_value_from_shiny_input_numeric(
                input["input_ID_tab_clients_subtab_clients_income_client_partner_income_pension"],
            )
            value_Partner_Annuity = get_value_from_shiny_input_numeric(
                input[
                    "input_ID_tab_clients_subtab_clients_income_client_partner_income_annuity_existing"
                ],
            )
            value_Partner_Other = get_value_from_shiny_input_numeric(
                input["input_ID_tab_clients_subtab_clients_income_client_partner_income_other"],
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
            Income_Data = pl.DataFrame(
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
            if Income_Data is None or Income_Data.height == 0:
                raise ValueError("Failed to create Income_Data dataframe")

            # Return Income_Data with proper reactive dependencies established
            return Income_Data

        except Exception:
            # Log error for debugging but return fallback dataframe
            # Note: Logging not imported to keep module lightweight
            # Suppress exception info since we're returning a safe fallback

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
    def output_ID_tab_clients_subtab_clients_income_table_main():
        """Generate income sources summary table using great-tables with reactive dependencies."""
        try:
            # Input validation with early returns - get reactive data from calculation
            data_Income_DF = calc_table_clients_income_main()

            # Configuration validation - ensure dataframe exists and has data
            if data_Income_DF is None:
                raise ValueError("data_Income_DF is None from reactive calculation")

            if data_Income_DF.height == 0:
                raise ValueError("data_Income_DF is empty from reactive calculation")

            # Try to create the enhanced table
            try:
                table_Income = create_enhanced_summary_table_multi_column(
                    dataframe_input=data_Income_DF,
                    table_title="Income Sources Summary",
                    table_subtitle="Annual income projections from all sources for retirement planning",
                    currency_columns=["client_Primary", "client_Partner", "Combined_Total"],
                    table_theme="professional",
                    table_width="100%",
                )

                # Configuration validation - verify table was created successfully
                if table_Income is None:
                    raise ValueError("Failed to create table_Income using enhanced function")

                # Return the GT table as HTML
                return ui.HTML(table_Income.as_raw_html())

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
    def observer_update_shared_reactives_shiny_income() -> None:
        """Update shared reactive values for cross-module communication.

        This reactive effect automatically updates the shared reactives_shiny structure
        whenever any income input values change, ensuring that other modules (like the
        Summary tab) can access the current income values for calculations and displays.

        Reactive Dependencies:
            - input_ID_tab_clients_subtab_clients_income_client_primary_income_social_security
            - input_ID_tab_clients_subtab_clients_income_client_primary_income_pension
            - input_ID_tab_clients_subtab_clients_income_client_primary_income_annuity_existing
            - input_ID_tab_clients_subtab_clients_income_client_primary_income_other
            - input_ID_tab_clients_subtab_clients_income_client_partner_income_social_security
            - input_ID_tab_clients_subtab_clients_income_client_partner_income_pension
            - input_ID_tab_clients_subtab_clients_income_client_partner_income_annuity_existing
            - input_ID_tab_clients_subtab_clients_income_client_partner_income_other

        Updates:
            Updates the User_Inputs_Shiny category in reactives_shiny with current
            income values for cross-module communication and summary calculations.

        Error Handling:
            Uses defensive programming with comprehensive error handling to ensure
            the application continues to function even if reactive updates fail.
        """
        # Get all current income values using the safe input access pattern
        primary_social_security = get_value_from_shiny_input_numeric(
            input[
                "input_ID_tab_clients_subtab_clients_income_client_primary_income_social_security"
            ],
        )
        primary_pension = get_value_from_shiny_input_numeric(
            input["input_ID_tab_clients_subtab_clients_income_client_primary_income_pension"],
        )
        primary_annuity_existing = get_value_from_shiny_input_numeric(
            input[
                "input_ID_tab_clients_subtab_clients_income_client_primary_income_annuity_existing"
            ],
        )
        primary_other = get_value_from_shiny_input_numeric(
            input["input_ID_tab_clients_subtab_clients_income_client_primary_income_other"],
        )

        partner_social_security = get_value_from_shiny_input_numeric(
            input[
                "input_ID_tab_clients_subtab_clients_income_client_partner_income_social_security"
            ],
        )
        partner_pension = get_value_from_shiny_input_numeric(
            input["input_ID_tab_clients_subtab_clients_income_client_partner_income_pension"],
        )
        partner_annuity_existing = get_value_from_shiny_input_numeric(
            input[
                "input_ID_tab_clients_subtab_clients_income_client_partner_income_annuity_existing"
            ],
        )
        partner_other = get_value_from_shiny_input_numeric(
            input["input_ID_tab_clients_subtab_clients_income_client_partner_income_other"],
        )

        # Update shared reactive values if available
        if reactives_shiny and isinstance(reactives_shiny, dict):
            user_inputs_category = reactives_shiny.get("User_Inputs_Shiny")
            if user_inputs_category and isinstance(user_inputs_category, dict):
                # Store all income values in the shared reactive structure
                # Following the naming convention from reactives_shiny.py
                income_mapping = {
                    "Input_Tab_clients_Subtab_clients_Income_client_Primary_Income_Social_Security": primary_social_security,
                    "Input_Tab_clients_Subtab_clients_Income_client_Primary_Income_Pension": primary_pension,
                    "Input_Tab_clients_Subtab_clients_Income_client_Primary_Income_Annuity_Existing": primary_annuity_existing,
                    "Input_Tab_clients_Subtab_clients_Income_client_Primary_Income_Other": primary_other,
                    "Input_Tab_clients_Subtab_clients_Income_client_Partner_Income_Social_Security": partner_social_security,
                    "Input_Tab_clients_Subtab_clients_Income_client_Partner_Income_Pension": partner_pension,
                    "Input_Tab_clients_Subtab_clients_Income_client_Partner_Income_Annuity_Existing": partner_annuity_existing,
                    "Input_Tab_clients_Subtab_clients_Income_client_Partner_Income_Other": partner_other,
                }

                # Update each reactive value safely using defensive programming
                for reactive_key, current_value in income_mapping.items():
                    if reactive_key in user_inputs_category:
                        reactive_var = user_inputs_category[reactive_key]
                        if reactive_var is not None and hasattr(reactive_var, "set"):
                            reactive_var.set(current_value)
