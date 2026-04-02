"""client Goals Subtab Module.

This module provides comprehensive goal setting and management functionality for
the clients tab of the QWIM financial dashboard. It creates professional
input forms for collecting and managing financial goals from both primary and
partner clients across three goal categories: Essential, Important, and Aspirational.

The module includes:
- Essential goal input and management
- Important goal input and management
- Aspirational goal input and management
- Real-time calculation and validation
- Professional UI components with enhanced styling
- Responsive design for different screen sizes

Key Features:
    - Enhanced numeric input components with currency formatting
    - Real-time validation and error handling
    - Professional card-based layout design
    - Three-tier goal categorization system
    - Multi-client support (primary and partner)
    - Automatic goal calculations and summaries
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


@module.ui
def subtab_clients_goals_ui(data_utils: dict[str, Any], data_inputs: dict[str, Any]) -> Any:
    """Create user interface for comprehensive client goal input and management.

    Generates a professional UI layout for collecting and managing financial goal
    information from both primary and partner clients. The interface includes
    enhanced input components with validation, professional styling, and real-time
    calculation capabilities across three goal categories: Essential, Important, and Aspirational.

    Args:
        data_utils (Dict[str, Any]): Utility functions and configuration data
            for the dashboard application functionality.
        data_inputs (Dict[str, Any]): Default input values and validation rules
            for goal input components and form initialization.

    Returns
    -------
        ui.div: Complete UI layout including:
            - Primary client goal input section
            - Partner client goal input section
            - Three goal category inputs per client
            - Real-time summary displays and totals
            - Professional card-based layout design

    UI Structure:
        ```
        Goal Management Interface
        ├── Primary client Goals
        │   ├── Essential Goal Input
        │   ├── Important Goal Input
        │   └── Aspirational Goal Input
        ├── Partner client Goals
        │   ├── Essential Goal
        │   ├── Important Goal
        │   └── Aspirational Goal
        └── Goal Summary Displays
            ├── Total Goals for client Primary
            ├── Combined Total Goals
            └── Total Goals for client Partner
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
            ui.h3("Financial Goals", class_="text-center mb-4"),
            # Primary and Partner client sections in responsive layout
            ui.row(
                # Primary client Goals Section
                ui.column(
                    6,
                    create_enhanced_card_section(
                        title="Primary client Goals",
                        content=[
                            # Essential Goal Input
                            create_enhanced_numeric_input(
                                input_ID="input_ID_tab_clients_subtab_clients_goals_client_primary_goal_essential",
                                label_text="Essential Goal",
                                min_value=0,
                                max_value=50000000,
                                step_size=1000,
                                default_value=0,
                                currency_format=True,
                                prefix_symbol="$",
                                # help_text="Must-have financial goal that is absolutely necessary",
                                tooltip_text="Enter essential financial goal amount",
                                required_field=False,
                                disabled_state=False,
                                input_width="100%",
                            ),
                            # Important Goal Input
                            create_enhanced_numeric_input(
                                input_ID="input_ID_tab_clients_subtab_clients_goals_client_primary_goal_important",
                                label_text="Important Goal",
                                min_value=0,
                                max_value=50000000,
                                step_size=1000,
                                default_value=0,
                                currency_format=True,
                                prefix_symbol="$",
                                # help_text="Significant financial goal that would greatly improve your situation",
                                tooltip_text="Enter important financial goal amount",
                                required_field=False,
                                disabled_state=False,
                                input_width="100%",
                            ),
                            # Aspirational Goal Input
                            create_enhanced_numeric_input(
                                input_ID="input_ID_tab_clients_subtab_clients_goals_client_primary_goal_aspirational",
                                label_text="Aspirational Goal",
                                min_value=0,
                                max_value=50000000,
                                step_size=1000,
                                default_value=0,
                                currency_format=True,
                                prefix_symbol="$",
                                # help_text="Dream financial goal that would be wonderful to achieve",
                                tooltip_text="Enter aspirational financial goal amount",
                                required_field=False,
                                disabled_state=False,
                                input_width="100%",
                            ),
                        ],
                        icon_class="fas fa-bullseye",
                        card_class="h-100 shadow-sm border-primary",
                    ),
                ),
                # Partner client Goals Section
                ui.column(
                    6,
                    create_enhanced_card_section(
                        title="Partner client Goals",
                        content=[
                            # Essential Goal Input
                            create_enhanced_numeric_input(
                                input_ID="input_ID_tab_clients_subtab_clients_goals_client_partner_goal_essential",
                                label_text="Essential Goal",
                                min_value=0,
                                max_value=50000000,
                                step_size=1000,
                                default_value=0,
                                currency_format=True,
                                prefix_symbol="$",
                                # help_text="Must-have financial goal that is absolutely necessary",
                                tooltip_text="Enter essential financial goal amount",
                                required_field=False,
                                disabled_state=False,
                                input_width="100%",
                            ),
                            # Important Goal Input
                            create_enhanced_numeric_input(
                                input_ID="input_ID_tab_clients_subtab_clients_goals_client_partner_goal_important",
                                label_text="Important Goal",
                                min_value=0,
                                max_value=50000000,
                                step_size=1000,
                                default_value=0,
                                currency_format=True,
                                prefix_symbol="$",
                                # help_text="Significant financial goal that would greatly improve your situation",
                                tooltip_text="Enter important financial goal amount",
                                required_field=False,
                                disabled_state=False,
                                input_width="100%",
                            ),
                            # Aspirational Goal Input
                            create_enhanced_numeric_input(
                                input_ID="input_ID_tab_clients_subtab_clients_goals_client_partner_goal_aspirational",
                                label_text="Aspirational Goal",
                                min_value=0,
                                max_value=50000000,
                                step_size=1000,
                                default_value=0,
                                currency_format=True,
                                prefix_symbol="$",
                                # help_text="Dream financial goal that would be wonderful to achieve",
                                tooltip_text="Enter aspirational financial goal amount",
                                required_field=False,
                                disabled_state=False,
                                input_width="100%",
                            ),
                        ],
                        icon_class="fas fa-chart-line",
                        card_class="h-100 shadow-sm border-info",
                    ),
                ),
            ),
            # Goal Summary Section
            ui.row(
                # Total Goals for client Primary (leftmost)
                ui.column(
                    4,
                    create_enhanced_summary_display(
                        summary_id="output_ID_tab_clients_subtab_clients_goals_client_primary_total",
                        title="Total Goals for client Primary",
                        icon_class="fas fa-calculator",
                        background_class="bg-light",
                        border_variant=ComponentVariant.SUCCESS,
                    ),
                ),
                # Combined Total Goals (middle)
                ui.column(
                    4,
                    create_enhanced_summary_display(
                        summary_id="output_ID_tab_clients_subtab_clients_goals_combined_total",
                        title="Combined Total Goals",
                        icon_class="fas fa-target",
                        background_class="bg-success",
                        text_class="text-white",
                        border_variant=ComponentVariant.SUCCESS,
                    ),
                ),
                # Total Goals for client Partner (rightmost)
                ui.column(
                    4,
                    create_enhanced_summary_display(
                        summary_id="output_ID_tab_clients_subtab_clients_goals_client_partner_total",
                        title="Total Goals for client Partner",
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
                        ui.card_header(ui.h4("client Goals Overview", class_="text-center")),
                        ui.card_body(
                            ui.div(
                                ui.output_ui(
                                    "output_ID_tab_clients_subtab_clients_goals_table_main",
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
        error_message = f"Unexpected error creating goals subtab UI: {exc_error}"
        raise RuntimeError(error_message) from exc_error


@module.server
def subtab_clients_goals_server(
    input: typing.Any,
    output: typing.Any,
    session: typing.Any,
    data_utils: dict[str, Any],
    data_inputs: dict[str, Any],
    reactives_shiny: dict[str, Any],
) -> None:
    """Server logic for comprehensive client goal management and calculations.

    Implements reactive server-side logic for processing goal inputs, performing
    real-time calculations, and generating summary displays. The server handles
    validation, formatting, and provides immediate feedback to users as they
    input goal information for the three goal categories: Essential, Important, and Aspirational.

    Args:
        input (typing.Any): Shiny input object containing reactive values
            from all goal input UI components.
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
        - Real-time goal total calculations
        - Input validation and error handling
        - Summary display generation and formatting
        - Cross-component reactive updates

    Reactive Behavior:
        All summary outputs automatically update when any goal input value
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

    # Reactive observers for currency formatting with proper input/display separation
    @reactive.effect
    @reactive.event(
        input.input_ID_tab_clients_subtab_clients_goals_client_primary_goal_essential,
        ignore_none=True,
    )
    def observer_update_primary_essential_goal_formatting() -> None:
        """Update currency formatting for primary essential goal input."""
        try:
            # Get current raw value directly from input
            current_value = (
                input.input_ID_tab_clients_subtab_clients_goals_client_primary_goal_essential()
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
            constrained_value = validate_and_constrain_numeric_value(numeric_value, 0, 50000000)

            # Format for display (this is what the user sees)
            formatted_value = format_currency_value(constrained_value)

            # Only update display if different from current display
            if current_str != formatted_value:
                ui.update_text(
                    "input_ID_tab_clients_subtab_clients_goals_client_primary_goal_essential",
                    value=formatted_value,
                )

        except Exception:
            # Silently handle errors to prevent disruption
            pass

    @reactive.effect
    @reactive.event(
        input.input_ID_tab_clients_subtab_clients_goals_client_primary_goal_important,
        ignore_none=True,
    )
    def observer_update_primary_important_goal_formatting() -> None:
        """Update currency formatting for primary important goal input."""
        try:
            # Get current raw value directly from input
            current_value = (
                input.input_ID_tab_clients_subtab_clients_goals_client_primary_goal_important()
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
            constrained_value = validate_and_constrain_numeric_value(numeric_value, 0, 50000000)

            # Format for display (this is what the user sees)
            formatted_value = format_currency_value(constrained_value)

            # Only update display if different from current display
            if current_str != formatted_value:
                ui.update_text(
                    "input_ID_tab_clients_subtab_clients_goals_client_primary_goal_important",
                    value=formatted_value,
                )

        except Exception:
            # Silently handle errors to prevent disruption
            pass

    @reactive.effect
    @reactive.event(
        input.input_ID_tab_clients_subtab_clients_goals_client_primary_goal_aspirational,
        ignore_none=True,
    )
    def observer_update_primary_aspirational_goal_formatting() -> None:
        """Update currency formatting for primary aspirational goal input."""
        try:
            # Get current raw value directly from input
            current_value = (
                input.input_ID_tab_clients_subtab_clients_goals_client_primary_goal_aspirational()
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
            constrained_value = validate_and_constrain_numeric_value(numeric_value, 0, 50000000)

            # Format for display (this is what the user sees)
            formatted_value = format_currency_value(constrained_value)

            # Only update display if different from current display
            if current_str != formatted_value:
                ui.update_text(
                    "input_ID_tab_clients_subtab_clients_goals_client_primary_goal_aspirational",
                    value=formatted_value,
                )

        except Exception:
            # Silently handle errors to prevent disruption
            pass

    @reactive.effect
    @reactive.event(
        input.input_ID_tab_clients_subtab_clients_goals_client_partner_goal_essential,
        ignore_none=True,
    )
    def observer_update_partner_essential_goal_formatting() -> None:
        """Update currency formatting for partner essential goal input."""
        try:
            # Get current raw value directly from input
            current_value = (
                input.input_ID_tab_clients_subtab_clients_goals_client_partner_goal_essential()
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
            constrained_value = validate_and_constrain_numeric_value(numeric_value, 0, 50000000)

            # Format for display (this is what the user sees)
            formatted_value = format_currency_value(constrained_value)

            # Only update display if different from current display
            if current_str != formatted_value:
                ui.update_text(
                    "input_ID_tab_clients_subtab_clients_goals_client_partner_goal_essential",
                    value=formatted_value,
                )

        except Exception:
            # Silently handle errors to prevent disruption
            pass

    @reactive.effect
    @reactive.event(
        input.input_ID_tab_clients_subtab_clients_goals_client_partner_goal_important,
        ignore_none=True,
    )
    def observer_update_partner_important_goal_formatting() -> None:
        """Update currency formatting for partner important goal input."""
        try:
            # Get current raw value directly from input
            current_value = (
                input.input_ID_tab_clients_subtab_clients_goals_client_partner_goal_important()
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
            constrained_value = validate_and_constrain_numeric_value(numeric_value, 0, 50000000)

            # Format for display (this is what the user sees)
            formatted_value = format_currency_value(constrained_value)

            # Only update display if different from current display
            if current_str != formatted_value:
                ui.update_text(
                    "input_ID_tab_clients_subtab_clients_goals_client_partner_goal_important",
                    value=formatted_value,
                )

        except Exception:
            # Silently handle errors to prevent disruption
            pass

    @reactive.effect
    @reactive.event(
        input.input_ID_tab_clients_subtab_clients_goals_client_partner_goal_aspirational,
        ignore_none=True,
    )
    def observer_update_partner_aspirational_goal_formatting() -> None:
        """Update currency formatting for partner aspirational goal input."""
        try:
            # Get current raw value directly from input
            current_value = (
                input.input_ID_tab_clients_subtab_clients_goals_client_partner_goal_aspirational()
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
            constrained_value = validate_and_constrain_numeric_value(numeric_value, 0, 50000000)

            # Format for display (this is what the user sees)
            formatted_value = format_currency_value(constrained_value)

            # Only update display if different from current display
            if current_str != formatted_value:
                ui.update_text(
                    "input_ID_tab_clients_subtab_clients_goals_client_partner_goal_aspirational",
                    value=formatted_value,
                )

        except Exception:
            # Silently handle errors to prevent disruption
            pass

    @output
    @render.text
    def output_ID_tab_clients_subtab_clients_goals_client_primary_total():
        """Calculate and display total goals for primary client.

        Returns
        -------
            str: Formatted currency string showing total primary client goals (dollars only)
        """
        try:
            # Get all primary client goal values (numeric values, not display strings)
            value_Primary_Essential_Goal = get_value_from_shiny_input_numeric(
                input["input_ID_tab_clients_subtab_clients_goals_client_primary_goal_essential"],
            )
            value_Primary_Important_Goal = get_value_from_shiny_input_numeric(
                input["input_ID_tab_clients_subtab_clients_goals_client_primary_goal_important"],
            )
            value_Primary_Aspirational_Goal = get_value_from_shiny_input_numeric(
                input["input_ID_tab_clients_subtab_clients_goals_client_primary_goal_aspirational"],
            )

            # Calculate total (numeric calculation)
            value_Primary_Total = (
                value_Primary_Essential_Goal
                + value_Primary_Important_Goal
                + value_Primary_Aspirational_Goal
            )

            # Format for display (dollars only, no cents)
            return format_currency_value(value_Primary_Total)

        except Exception as exc_error:
            return f"Error: {exc_error!s}"

    @output
    @render.text
    def output_ID_tab_clients_subtab_clients_goals_client_partner_total():
        """Calculate and display total goals for partner client.

        Returns
        -------
            str: Formatted currency string showing total partner client goals (dollars only)
        """
        try:
            # Get all partner client goal values (numeric values, not display strings)
            value_Partner_Essential_Goal = get_value_from_shiny_input_numeric(
                input["input_ID_tab_clients_subtab_clients_goals_client_partner_goal_essential"],
            )
            value_Partner_Important_Goal = get_value_from_shiny_input_numeric(
                input["input_ID_tab_clients_subtab_clients_goals_client_partner_goal_important"],
            )
            value_Partner_Aspirational_Goal = get_value_from_shiny_input_numeric(
                input["input_ID_tab_clients_subtab_clients_goals_client_partner_goal_aspirational"],
            )

            # Calculate total (numeric calculation)
            value_Partner_Total = (
                value_Partner_Essential_Goal
                + value_Partner_Important_Goal
                + value_Partner_Aspirational_Goal
            )

            # Format for display (dollars only, no cents)
            return format_currency_value(value_Partner_Total)

        except Exception as exc_error:
            return f"Error: {exc_error!s}"

    @output
    @render.text
    def output_ID_tab_clients_subtab_clients_goals_combined_total():
        """Calculate and display combined total goals for both clients.

        Returns
        -------
            str: Formatted currency string showing combined total goals (dollars only)
        """
        try:
            # Get all goal values (numeric values, not display strings)
            value_Primary_Essential_Goal = get_value_from_shiny_input_numeric(
                input["input_ID_tab_clients_subtab_clients_goals_client_primary_goal_essential"],
            )
            value_Primary_Important_Goal = get_value_from_shiny_input_numeric(
                input["input_ID_tab_clients_subtab_clients_goals_client_primary_goal_important"],
            )
            value_Primary_Aspirational_Goal = get_value_from_shiny_input_numeric(
                input["input_ID_tab_clients_subtab_clients_goals_client_primary_goal_aspirational"],
            )

            value_Partner_Essential_Goal = get_value_from_shiny_input_numeric(
                input["input_ID_tab_clients_subtab_clients_goals_client_partner_goal_essential"],
            )
            value_Partner_Important_Goal = get_value_from_shiny_input_numeric(
                input["input_ID_tab_clients_subtab_clients_goals_client_partner_goal_important"],
            )
            value_Partner_Aspirational_Goal = get_value_from_shiny_input_numeric(
                input["input_ID_tab_clients_subtab_clients_goals_client_partner_goal_aspirational"],
            )

            # Calculate combined total (numeric calculation)
            value_Combined_Total = (
                value_Primary_Essential_Goal
                + value_Primary_Important_Goal
                + value_Primary_Aspirational_Goal
                + value_Partner_Essential_Goal
                + value_Partner_Important_Goal
                + value_Partner_Aspirational_Goal
            )

            # Format for display (dollars only, no cents)
            return format_currency_value(value_Combined_Total)

        except Exception as exc_error:
            return f"Error: {exc_error!s}"

    @reactive.calc
    def calc_table_clients_goals_main():
        """Generate financial goals summary table."""
        # Input validation with early returns and direct function calls
        # Access reactive values using proper Shiny input access patterns

        # Primary client goal information with proper reactive dependencies
        value_Primary_Essential = get_value_from_shiny_input_numeric(
            input["input_ID_tab_clients_subtab_clients_goals_client_primary_goal_essential"],
        )
        value_Primary_Important = get_value_from_shiny_input_numeric(
            input["input_ID_tab_clients_subtab_clients_goals_client_primary_goal_important"],
        )
        value_Primary_Aspirational = get_value_from_shiny_input_numeric(
            input["input_ID_tab_clients_subtab_clients_goals_client_primary_goal_aspirational"],
        )

        # Partner client goal information with proper reactive dependencies
        value_Partner_Essential = get_value_from_shiny_input_numeric(
            input["input_ID_tab_clients_subtab_clients_goals_client_partner_goal_essential"],
        )
        value_Partner_Important = get_value_from_shiny_input_numeric(
            input["input_ID_tab_clients_subtab_clients_goals_client_partner_goal_important"],
        )
        value_Partner_Aspirational = get_value_from_shiny_input_numeric(
            input["input_ID_tab_clients_subtab_clients_goals_client_partner_goal_aspirational"],
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

    @output
    @render.ui
    def output_ID_tab_clients_subtab_clients_goals_table_main():
        """Generate financial goals summary table using great-tables with reactive dependencies."""
        try:
            # Input validation with early returns - get reactive data from calculation
            data_Goals_DF = calc_table_clients_goals_main()

            # Configuration validation - ensure dataframe exists and has data
            if data_Goals_DF is None:
                raise ValueError("data_Goals_DF is None from reactive calculation")

            if data_Goals_DF.height == 0:
                raise ValueError("data_Goals_DF is empty from reactive calculation")

            # Try to create the enhanced table
            try:
                Goals_Table = create_enhanced_summary_table_multi_column(
                    dataframe_input=data_Goals_DF,
                    table_title="Financial Goals Summary",
                    table_subtitle="Annual goal targets organized by priority level and importance",
                    currency_columns=["client_Primary", "client_Partner", "Combined_Total"],
                    table_theme="professional",
                    table_width="100%",
                )

                # Configuration validation - verify table was created successfully
                if Goals_Table is None:
                    raise ValueError("Failed to create Goals_Table using enhanced function")

                # Return the GT table as HTML
                return ui.HTML(Goals_Table.as_raw_html())

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

    @reactive.effect
    def observer_update_shared_reactives_shiny_goals() -> None:
        """Update shared reactive values for cross-module communication.

        This reactive effect automatically updates the shared reactives_shiny structure
        whenever any goal input values change, ensuring that other modules (like the
        Summary tab) can access the current goal values for calculations and displays.

        Reactive Dependencies:
            - input_ID_tab_clients_subtab_clients_goals_client_primary_goal_essential
            - input_ID_tab_clients_subtab_clients_goals_client_primary_goal_important
            - input_ID_tab_clients_subtab_clients_goals_client_primary_goal_aspirational
            - input_ID_tab_clients_subtab_clients_goals_client_partner_goal_essential
            - input_ID_tab_clients_subtab_clients_goals_client_partner_goal_important
            - input_ID_tab_clients_subtab_clients_goals_client_partner_goal_aspirational

        Updates:
            Updates the User_Inputs_Shiny category in reactives_shiny with current
            goal values for cross-module communication and summary calculations.

        Error Handling:
            Uses defensive programming with comprehensive error handling to ensure
            the application continues to function even if reactive updates fail.
        """
        # Get all current goal values using the safe input access pattern
        primary_essential = get_value_from_shiny_input_numeric(
            input["input_ID_tab_clients_subtab_clients_goals_client_primary_goal_essential"],
        )
        primary_important = get_value_from_shiny_input_numeric(
            input["input_ID_tab_clients_subtab_clients_goals_client_primary_goal_important"],
        )
        primary_aspirational = get_value_from_shiny_input_numeric(
            input["input_ID_tab_clients_subtab_clients_goals_client_primary_goal_aspirational"],
        )

        partner_essential = get_value_from_shiny_input_numeric(
            input["input_ID_tab_clients_subtab_clients_goals_client_partner_goal_essential"],
        )
        partner_important = get_value_from_shiny_input_numeric(
            input["input_ID_tab_clients_subtab_clients_goals_client_partner_goal_important"],
        )
        partner_aspirational = get_value_from_shiny_input_numeric(
            input["input_ID_tab_clients_subtab_clients_goals_client_partner_goal_aspirational"],
        )

        # Update shared reactive values if available
        if reactives_shiny and isinstance(reactives_shiny, dict):
            user_inputs_category = reactives_shiny.get("User_Inputs_Shiny")
            if user_inputs_category and isinstance(user_inputs_category, dict):
                # Store all goal values in the shared reactive structure
                # Following the naming convention from reactives_shiny.py
                goal_mapping = {
                    "Input_Tab_clients_Subtab_clients_Goals_client_Primary_Goal_Essential": primary_essential,
                    "Input_Tab_clients_Subtab_clients_Goals_client_Primary_Goal_Important": primary_important,
                    "Input_Tab_clients_Subtab_clients_Goals_client_Primary_Goal_Aspirational": primary_aspirational,
                    "Input_Tab_clients_Subtab_clients_Goals_client_Partner_Goal_Essential": partner_essential,
                    "Input_Tab_clients_Subtab_clients_Goals_client_Partner_Goal_Important": partner_important,
                    "Input_Tab_clients_Subtab_clients_Goals_client_Partner_Goal_Aspirational": partner_aspirational,
                }

                # Update each reactive value safely using defensive programming
                for reactive_key, current_value in goal_mapping.items():
                    if reactive_key in user_inputs_category:
                        reactive_var = user_inputs_category[reactive_key]
                        if reactive_var is not None and hasattr(reactive_var, "set"):
                            reactive_var.set(current_value)
