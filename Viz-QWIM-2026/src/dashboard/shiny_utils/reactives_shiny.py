"""Reactive state management utilities for QWIM Dashboard.

This module provides centralized reactive state management for the Shiny dashboard.
It implements the reactives_shiny dictionary with 4 categories for state organization.
"""

from __future__ import annotations

from typing import Any

from shiny import reactive

from src.utils.custom_exceptions_errors_loggers.logger_custom import get_logger


#: Module-level logger instance
_logger = get_logger(__name__)


def validate_data_utils_parameter(data_utils: Any) -> Any:
    """Validate data_utils parameter using defensive programming."""
    # Input validation with early returns
    if data_utils is None:
        return False, "data_utils parameter cannot be None"

    # Configuration validation
    if not isinstance(data_utils, dict):
        return False, f"data_utils must be a dictionary, got {type(data_utils).__name__}"

    return True, ""


def validate_reactives_shiny_structure(reactives_shiny: Any) -> Any:
    """Validate reactives_shiny dictionary structure using defensive programming."""
    # Input validation with early returns
    if reactives_shiny is None:
        return False, "reactives_shiny dictionary cannot be None"

    if not isinstance(reactives_shiny, dict):
        return False, f"reactives_shiny must be a dictionary, got {type(reactives_shiny).__name__}"

    # Configuration validation - check required categories
    required_categories = [
        "User_Inputs_Shiny",
        "Inner_Variables_Shiny",
        "Triggers_Shiny",
        "Visual_Objects_Shiny",
        "Data_Clients",
        "Data_Results",
    ]

    for category_name in required_categories:
        if category_name not in reactives_shiny:
            available_categories = list(reactives_shiny.keys())
            return (
                False,
                f"Required category '{category_name}' not found. Available: {available_categories}",
            )

        # Business logic validation - ensure each category is a dictionary
        if not isinstance(reactives_shiny[category_name], dict):
            return (
                False,
                f"Category '{category_name}' must be a dictionary, got {type(reactives_shiny[category_name]).__name__}",
            )

    return True, ""


def validate_reactive_key_access(category_dict: Any, key_name: Any, category_name: Any) -> Any:
    """Validate reactive key access using defensive programming."""
    # Input validation with early returns
    if category_dict is None:
        return False, f"Category '{category_name}' is None"

    if not isinstance(category_dict, dict):
        return (
            False,
            f"Category '{category_name}' is not a dictionary, got {type(category_dict).__name__}",
        )

    if key_name is None:
        return False, "key_name cannot be None"

    if not isinstance(key_name, str):
        return False, f"key_name must be string, got {type(key_name).__name__}"

    # Configuration validation
    if len(key_name.strip()) == 0:
        return False, "key_name cannot be empty string"

    # Business logic validation
    if key_name not in category_dict:
        available_keys = list(category_dict.keys())
        return (
            False,
            f"Key '{key_name}' not found in category '{category_name}'. Available keys: {available_keys}",
        )

    return True, ""


def validate_category_name(category_name: Any) -> Any:
    """Validate category name using defensive programming."""
    # Input validation with early returns
    if category_name is None:
        return False, "category_name cannot be None"

    if not isinstance(category_name, str):
        return False, f"category_name must be string, got {type(category_name).__name__}"

    # Configuration validation
    if len(category_name.strip()) == 0:
        return False, "category_name cannot be empty string"

    # Business logic validation
    valid_categories = [
        "User_Inputs_Shiny",
        "Inner_Variables_Shiny",
        "Triggers_Shiny",
        "Visual_Objects_Shiny",
        "Data_Clients",
        "Data_Results",
    ]

    if category_name not in valid_categories:
        return False, f"Invalid category '{category_name}'. Valid categories: {valid_categories}"

    return True, ""


def safe_get_shiny_input_value(input_events: Any, input_identifier: Any) -> Any:
    """Safely get value from Shiny input using defensive programming."""
    # Input validation with early returns
    if input_events is None:
        return None

    if not isinstance(input_identifier, str):
        return None

    if len(input_identifier.strip()) == 0:
        return None

    # Only use try-except for Shiny input access that might fail unpredictably
    try:
        # Check if the input identifier exists as an attribute of input_events
        if hasattr(input_events, input_identifier):
            input_reactive = getattr(input_events, input_identifier)

            # Call the reactive function to get the current value
            if callable(input_reactive):
                return input_reactive()
            return input_reactive
        return None

    except (AttributeError, KeyError, TypeError):
        return None
    except Exception:
        return None


def create_reactive_value_safely(initial_value: Any = None) -> Any:
    """Create a reactive value safely using defensive programming."""
    # Only use try-except for reactive.Value creation that might fail unpredictably
    try:
        return reactive.Value(initial_value)
    except Exception:
        return None


def initialize_reactives_shiny(data_utils: Any) -> Any:
    """Initialize all reactive values for the Shiny dashboard using defensive programming.

    Args:
        data_utils: Initial data for reactive value initialization

    Returns
    -------
        dict: Dictionary containing organized reactive values by category

    Raises
    ------
        ValueError: If data_utils is invalid
    """
    # Input validation with early returns
    validation_result, validation_message = validate_data_utils_parameter(data_utils)
    if not validation_result:
        raise ValueError(f"Data utils validation failed: {validation_message}")

    # Initialize each category using helper functions
    reactive_user_inputs = initialize_reactive_user_inputs(data_utils)
    reactive_inner_variables = initialize_reactive_inner_variables(data_utils)
    reactive_triggers = initialize_reactive_triggers(data_utils)
    reactive_visual_objects = initialize_reactive_visual_objects(data_utils)
    reactive_data_clients = initialize_reactive_data_clients(data_utils)
    reactive_data_results = initialize_reactive_data_results(data_utils)

    # Business logic validation - create organized structure
    return {
        "User_Inputs_Shiny": reactive_user_inputs,
        "Inner_Variables_Shiny": reactive_inner_variables,
        "Triggers_Shiny": reactive_triggers,
        "Visual_Objects_Shiny": reactive_visual_objects,
        "Data_Clients": reactive_data_clients,
        "Data_Results": reactive_data_results,
    }


def initialize_reactive_user_inputs(data_utils: Any) -> Any:
    """Initialize values for User_Inputs_Shiny category in the reactives_shiny structures using defensive programming."""
    # Input validation with early returns
    validation_result, validation_message = validate_data_utils_parameter(data_utils)
    if not validation_result:
        raise ValueError(f"Data utils validation failed: {validation_message}")

    # Configuration validation - define reactive inputs mapping to all subtab modules
    reactive_inputs_config = {
        "Input_Tab_Portfolios_Subtab_Comparison_Time_Period": None,
        "Input_Tab_Portfolios_Subtab_Comparison_Date_Range": None,
        "Input_Tab_Portfolios_Subtab_Comparison_Viz_Type": None,
        "Input_Tab_Portfolios_Subtab_Comparison_Show_Diff": None,
        "Input_Tab_Portfolios_Subtab_Portfolios_Analysis_Time_Period": None,
        "Input_Tab_Portfolios_Subtab_Portfolios_Analysis_Date_Range": None,
        "Input_Tab_Portfolios_Subtab_Portfolios_Analysis_Type": None,
        "Input_Tab_Portfolios_Subtab_Portfolios_Analysis_Rolling_Window": None,
        "Input_Tab_Portfolios_Subtab_Portfolios_Analysis_Include_Benchmark": None,
        "Input_Tab_clients_Subtab_clients_Personal_Info_client_Primary_Name": None,
        "Input_Tab_clients_Subtab_clients_Personal_Info_client_Primary_Age_Current": None,
        "Input_Tab_clients_Subtab_clients_Personal_Info_client_Primary_Age_Retirement": None,
        "Input_Tab_clients_Subtab_clients_Personal_Info_client_Primary_Age_Income_Starting": None,
        "Input_Tab_clients_Subtab_clients_Personal_Info_client_Primary_Status_Marital": None,
        "Input_Tab_clients_Subtab_clients_Personal_Info_client_Primary_Gender": None,
        "Input_Tab_clients_Subtab_clients_Personal_Info_client_Primary_Tolerance_Risk": None,
        "Input_Tab_clients_Subtab_clients_Personal_Info_client_Primary_State": None,
        "Input_Tab_clients_Subtab_clients_Personal_Info_client_Primary_Code_Zip": None,
        "Input_Tab_clients_Subtab_clients_Personal_Info_client_Partner_Name": None,
        "Input_Tab_clients_Subtab_clients_Personal_Info_client_Partner_Age_Current": None,
        "Input_Tab_clients_Subtab_clients_Personal_Info_client_Partner_Age_Retirement": None,
        "Input_Tab_clients_Subtab_clients_Personal_Info_client_Partner_Age_Income_Starting": None,
        "Input_Tab_clients_Subtab_clients_Personal_Info_client_Partner_Status_Marital": None,
        "Input_Tab_clients_Subtab_clients_Personal_Info_client_Partner_Gender": None,
        "Input_Tab_clients_Subtab_clients_Personal_Info_client_Partner_Tolerance_Risk": None,
        "Input_Tab_clients_Subtab_clients_Personal_Info_client_Partner_State": None,
        "Input_Tab_clients_Subtab_clients_Personal_Info_client_Partner_Code_Zip": None,
        "Input_Tab_clients_Subtab_clients_Assets_client_Primary_Assets_Taxable": None,
        "Input_Tab_clients_Subtab_clients_Assets_client_Primary_Assets_Tax_Deferred": None,
        "Input_Tab_clients_Subtab_clients_Assets_client_Primary_Assets_Tax_Free": None,
        "Input_Tab_clients_Subtab_clients_Assets_client_Partner_Assets_Taxable": None,
        "Input_Tab_clients_Subtab_clients_Assets_client_Partner_Assets_Tax_Deferred": None,
        "Input_Tab_clients_Subtab_clients_Assets_client_Partner_Assets_Tax_Free": None,
        "Input_Tab_clients_Subtab_clients_Goals_client_Primary_Goal_Essential": None,
        "Input_Tab_clients_Subtab_clients_Goals_client_Primary_Goal_Important": None,
        "Input_Tab_clients_Subtab_clients_Goals_client_Primary_Goal_Aspirational": None,
        "Input_Tab_clients_Subtab_clients_Goals_client_Partner_Goal_Essential": None,
        "Input_Tab_clients_Subtab_clients_Goals_client_Partner_Goal_Important": None,
        "Input_Tab_clients_Subtab_clients_Goals_client_Partner_Goal_Aspirational": None,
        "Input_Tab_clients_Subtab_clients_Income_client_Primary_Income_Social_Security": None,
        "Input_Tab_clients_Subtab_clients_Income_client_Primary_Income_Pension": None,
        "Input_Tab_clients_Subtab_clients_Income_client_Primary_Income_Annuity_Existing": None,
        "Input_Tab_clients_Subtab_clients_Income_client_Primary_Income_Other": None,
        "Input_Tab_clients_Subtab_clients_Income_client_Partner_Income_Social_Security": None,
        "Input_Tab_clients_Subtab_clients_Income_client_Partner_Income_Pension": None,
        "Input_Tab_clients_Subtab_clients_Income_client_Partner_Income_Annuity_Existing": None,
        "Input_Tab_clients_Subtab_clients_Income_client_Partner_Income_Other": None,
        # ------------------------------------------------------------------
        # Weights Analysis subtab
        # Maps to: input_ID_tab_portfolios_subtab_weights_analysis_*
        # ------------------------------------------------------------------
        "Input_Tab_Portfolios_Subtab_Weights_Analysis_Time_Period": None,
        "Input_Tab_Portfolios_Subtab_Weights_Analysis_Date_Range": None,
        "Input_Tab_Portfolios_Subtab_Weights_Analysis_Viz_Type": None,
        "Input_Tab_Portfolios_Subtab_Weights_Analysis_Show_Pct": None,
        "Input_Tab_Portfolios_Subtab_Weights_Analysis_Sort_Components": None,
        "Input_Tab_Portfolios_Subtab_Weights_Analysis_Select_All_Components": None,
        # ------------------------------------------------------------------
        # Skfolio Portfolio Optimization subtab
        # Maps to: input_ID_tab_portfolios_subtab_skfolio_*
        # ------------------------------------------------------------------
        "Input_Tab_Portfolios_Subtab_Skfolio_Method1_Category": None,
        "Input_Tab_Portfolios_Subtab_Skfolio_Method1_Type": None,
        "Input_Tab_Portfolios_Subtab_Skfolio_Method1_Objective": None,
        "Input_Tab_Portfolios_Subtab_Skfolio_Method1_Risk_Aversion": None,
        "Input_Tab_Portfolios_Subtab_Skfolio_Method2_Category": None,
        "Input_Tab_Portfolios_Subtab_Skfolio_Method2_Type": None,
        "Input_Tab_Portfolios_Subtab_Skfolio_Method2_Objective": None,
        "Input_Tab_Portfolios_Subtab_Skfolio_Method2_Risk_Aversion": None,
        "Input_Tab_Portfolios_Subtab_Skfolio_Time_Period": None,
        # ------------------------------------------------------------------
        # Portfolio Simulation subtab
        # Maps to: input_ID_tab_results_subtab_simulation_*
        # ------------------------------------------------------------------
        "Input_Tab_Results_Subtab_Simulation_Distribution_Type": None,
        "Input_Tab_Results_Subtab_Simulation_Rng_Type": None,
        "Input_Tab_Results_Subtab_Simulation_Num_Scenarios": None,
        "Input_Tab_Results_Subtab_Simulation_Num_Days": None,
        "Input_Tab_Results_Subtab_Simulation_Seed": None,
        "Input_Tab_Results_Subtab_Simulation_Initial_Value": None,
        "Input_Tab_Results_Subtab_Simulation_Degrees_Of_Freedom": None,
        "Input_Tab_Results_Subtab_Simulation_Start_Date": None,
        "Input_Tab_Results_Subtab_Simulation_Select_All_Components": None,
    }

    # Business logic validation - create reactive values safely
    reactive_user_inputs = {}
    for reactive_key_name, initial_value in reactive_inputs_config.items():
        reactive_user_inputs[reactive_key_name] = create_reactive_value_safely(initial_value)

    return reactive_user_inputs


def initialize_reactive_inner_variables(data_utils: Any) -> Any:
    """Initialize reactive values for Inner_Variables_Shiny category in the reactives_shiny structures using defensive programming.

    Args:
        data_utils: Additional data for reactive value initialization

    Returns
    -------
        dict: Dictionary containing reactive values for Inner_Variables_Shiny category in the reactives_shiny structures.

    Raises
    ------
        ValueError: If data_utils is invalid
    """
    # Input validation with early returns
    validation_result, validation_message = validate_data_utils_parameter(data_utils)
    if not validation_result:
        raise ValueError(f"Data utils validation failed: {validation_message}")

    # Configuration validation - define inner variables
    inner_variables_config = {
        "Data_Personal_Info_DF": None,
        "Data_Assets_DF": None,
        "Data_Goals_DF": None,
        "Data_Income_DF": None,
    }

    # Business logic validation - create reactive values safely
    reactive_inner_variables = {}
    for variable_key_name, initial_value in inner_variables_config.items():
        reactive_inner_variables[variable_key_name] = create_reactive_value_safely(initial_value)

    return reactive_inner_variables


def initialize_reactive_triggers(data_utils: Any) -> Any:
    """Initialize reactive values for Triggers_Shiny category in the reactives_shiny structures using defensive programming.

    Args:
        data_utils: Additional data for reactive value initialization

    Returns
    -------
        dict: Dictionary containing reactive values for Triggers_Shiny category in the reactives_shiny structures.

    Raises
    ------
        ValueError: If data_utils is invalid
    """
    # Input validation with early returns
    validation_result, validation_message = validate_data_utils_parameter(data_utils)
    if not validation_result:
        raise ValueError(f"Data utils validation failed: {validation_message}")

    # Configuration validation - define triggers
    triggers_config = {"Temp_Trigger_One": None}

    # Business logic validation - create reactive values safely
    reactive_triggers = {}
    for trigger_key_name, initial_value in triggers_config.items():
        reactive_triggers[trigger_key_name] = create_reactive_value_safely(initial_value)

    return reactive_triggers


def initialize_reactive_visual_objects(data_utils: Any) -> Any:
    """Initialize reactive values for Visual_Objects_Shiny category in the reactives_shiny structures using defensive programming.

    Args:
        data_utils: Additional data for reactive value initialization

    Returns
    -------
        dict: Dictionary containing reactive values for Visual_Objects_Shiny category in the reactives_shiny structures.

    Raises
    ------
        ValueError: If data_utils is invalid
    """
    # Input validation with early returns
    validation_result, validation_message = validate_data_utils_parameter(data_utils)
    if not validation_result:
        raise ValueError(f"Data utils validation failed: {validation_message}")

    # Configuration validation - define visual objects (charts only — saved as SVGs for PDF report)
    visual_objects_config = {
        "Chart_Weights_Analysis": None,
        "Charts_Weights_Composition": None,
        "Chart_Portfolio_Analysis": None,
        "Chart_Portfolio_Comparison": None,
        "Chart_Skfolio_Weights": None,
        "Chart_Skfolio_Performance": None,
        "Chart_Simulation_Fan": None,
        "Chart_Simulation_Histogram": None,
    }

    # Business logic validation - create reactive values safely
    reactive_visual_objects = {}
    for visual_key_name, initial_value in visual_objects_config.items():
        reactive_visual_objects[visual_key_name] = create_reactive_value_safely(initial_value)

    return reactive_visual_objects


def initialize_reactive_data_clients(data_utils: Any) -> Any:
    """Initialize nested reactive values for Data_Clients category in the reactives_shiny structures.

    Creates a nested dictionary structure for client data with three sub-levels:
    - ``Client_Primary``: reactive values for personal info, assets, goals, and income of the
      primary client.
    - ``Client_Partner``: reactive values for personal info, assets, goals, and income of the
      partner client.
    - ``Clients_Combined``: reactive values for combined assets, goals, and income of both clients.

    An additional ``Single_Or_Couple`` reactive value indicates whether the household consists of a
    single client (``"Single"``) or a couple with a primary and partner client (``"Couple"``).

    Args:
        data_utils: Additional data for reactive value initialization

    Returns
    -------
        dict: Nested dictionary containing reactive values for Data_Clients category in the
        reactives_shiny structures with the following layout::

            {
                "Single_Or_Couple": reactive.Value("Single"),
                "Client_Primary": {
                    "Personal_Info": reactive.Value(None),
                    "Assets": reactive.Value(None),
                    "Goals": reactive.Value(None),
                    "Income": reactive.Value(None),
                },
                "Client_Partner": {
                    "Personal_Info": reactive.Value(None),
                    "Assets": reactive.Value(None),
                    "Goals": reactive.Value(None),
                    "Income": reactive.Value(None),
                },
                "Clients_Combined": {
                    "Assets": reactive.Value(None),
                    "Goals": reactive.Value(None),
                    "Income": reactive.Value(None),
                },
            }

    Raises
    ------
        ValueError: If data_utils is invalid
    """
    # Input validation with early returns
    validation_result, validation_message = validate_data_utils_parameter(data_utils)
    if not validation_result:
        raise ValueError(f"Data utils validation failed: {validation_message}")

    # Configuration validation - define per-client data categories
    client_data_categories_config = [
        "Personal_Info",
        "Assets",
        "Goals",
        "Income",
    ]

    # Business logic validation - build Client_Primary sub-dict
    reactive_client_primary: dict[str, Any] = {}
    for category_key_name in client_data_categories_config:
        reactive_client_primary[category_key_name] = create_reactive_value_safely(None)

    # Business logic validation - build Client_Partner sub-dict
    reactive_client_partner: dict[str, Any] = {}
    for category_key_name in client_data_categories_config:
        reactive_client_partner[category_key_name] = create_reactive_value_safely(None)

    # Configuration validation - define combined data categories (no Personal_Info)
    combined_data_categories_config = [
        "Assets",
        "Goals",
        "Income",
    ]

    # Business logic validation - build Clients_Combined sub-dict
    reactive_clients_combined: dict[str, Any] = {}
    for category_key_name in combined_data_categories_config:
        reactive_clients_combined[category_key_name] = create_reactive_value_safely(None)

    # Assemble the Data_Clients nested structure
    return {
        "Single_Or_Couple": create_reactive_value_safely("Single"),
        "Client_Primary": reactive_client_primary,
        "Client_Partner": reactive_client_partner,
        "Clients_Combined": reactive_clients_combined,
    }


def initialize_reactive_data_results(data_utils: Any) -> Any:
    """Initialize nested reactive values for Data_Results category in the reactives_shiny structures.

    Creates a flat dictionary structure for results data with one reactive.Value per
    subtab/direction combination.  Each entry stores whatever data structure
    (dict, Polars DataFrame, or ``None``) that particular subtab needs to persist
    across reactive cycles.

    Args:
        data_utils: Additional data for reactive value initialization

    Returns
    -------
        dict: Dictionary containing reactive values for the Data_Results category with
        the following layout::

            {
                "Portfolio_Analysis_Inputs": reactive.Value(None),
                "Portfolio_Analysis_Outputs": reactive.Value(None),
                "Portfolio_Comparison_Inputs": reactive.Value(None),
                "Portfolio_Comparison_Outputs": reactive.Value(None),
                "Weights_Analysis_Inputs": reactive.Value(None),
                "Weights_Analysis_Outputs": reactive.Value(None),
                "Portfolio_Optimization_Skfolio_Inputs": reactive.Value(None),
                "Portfolio_Optimization_Skfolio_Outputs": reactive.Value(None),
                "Portfolio_Simulation_Inputs": reactive.Value(None),
                "Portfolio_Simulation_Outputs": reactive.Value(None),
            }

    Raises
    ------
        ValueError: If data_utils is invalid
    """
    # Input validation with early returns
    validation_result, validation_message = validate_data_utils_parameter(data_utils)
    if not validation_result:
        raise ValueError(f"Data utils validation failed: {validation_message}")

    # Configuration validation - ordered list of all result subtab keys
    results_subtab_keys_config = [
        "Portfolio_Analysis_Inputs",
        "Portfolio_Analysis_Outputs",
        "Portfolio_Comparison_Inputs",
        "Portfolio_Comparison_Outputs",
        "Weights_Analysis_Inputs",
        "Weights_Analysis_Outputs",
        "Portfolio_Optimization_Skfolio_Inputs",
        "Portfolio_Optimization_Skfolio_Outputs",
        "Portfolio_Simulation_Inputs",
        "Portfolio_Simulation_Outputs",
    ]

    # Business logic validation - build the Data_Results dict
    reactive_data_results: dict[str, Any] = {}
    for subtab_key_name in results_subtab_keys_config:
        reactive_data_results[subtab_key_name] = create_reactive_value_safely(None)

    return reactive_data_results


def get_value_from_reactives_shiny(reactives_shiny: Any, key_name: Any, key_category: Any) -> Any:
    """Get a value from reactives dictionary using specific category with defensive programming.

    This function retrieves values from reactive variables with explicit error handling,
    throwing detailed errors for all failure scenarios rather than returning default values.
    Error messages include the key_name and key_category for better debugging.

    Args:
        reactives_shiny: Dictionary containing reactive values organized by category
        key_name: The key name to search for in the specified category
        key_category: The category to search in ("User_Inputs_Shiny", "Inner_Variables_Shiny",
                     "Triggers_Shiny", or "Visual_Objects_Shiny")

    Returns
    -------
        The value from the reactive variable (any type, including None if that's the actual stored value)

    Raises
    ------
        ValueError: If reactives_shiny is None/invalid, key_category is invalid,
                   key_name is None/empty, or reactive variable is invalid
        KeyError: If key_name is not found in the specified category or category doesn't exist
        RuntimeError: If reactive value retrieval fails unexpectedly
        TypeError: If reactive variable doesn't have required methods
    """
    # Input validation with early returns - validate key_name first for better error messages
    if key_name is None:
        raise ValueError("key_name cannot be None")

    if not isinstance(key_name, str):
        raise ValueError(f"key_name must be string, got {type(key_name).__name__}: '{key_name}'")

    if len(key_name.strip()) == 0:
        raise ValueError(f"key_name cannot be empty string: '{key_name}'")

    # Clean key_name for consistent processing
    key_name_cleaned = key_name.strip()

    # Validate key_category with enhanced error messages
    if key_category is None:
        raise ValueError(
            f"key_category cannot be None (searching for key_name: '{key_name_cleaned}')",
        )

    if not isinstance(key_category, str):
        raise ValueError(
            f"key_category must be string, got {type(key_category).__name__}: '{key_category}' (searching for key_name: '{key_name_cleaned}')",
        )

    if len(key_category.strip()) == 0:
        raise ValueError(
            f"key_category cannot be empty string: '{key_category}' (searching for key_name: '{key_name_cleaned}')",
        )

    # Clean key_category for consistent processing
    key_category_cleaned = key_category.strip()

    # Validate reactives_shiny structure with enhanced error messages
    if reactives_shiny is None:
        raise ValueError(
            f"reactives_shiny dictionary cannot be None (searching for key_name: '{key_name_cleaned}' in key_category: '{key_category_cleaned}')",
        )

    if not isinstance(reactives_shiny, dict):
        raise ValueError(
            f"reactives_shiny must be a dictionary, got {type(reactives_shiny).__name__} (searching for key_name: '{key_name_cleaned}' in key_category: '{key_category_cleaned}')",
        )

    # Configuration validation - check required categories with enhanced error messages
    required_categories = [
        "User_Inputs_Shiny",
        "Inner_Variables_Shiny",
        "Triggers_Shiny",
        "Visual_Objects_Shiny",
        "Data_Clients",
    ]

    # Validate that key_category is a valid category
    if key_category_cleaned not in required_categories:
        raise ValueError(
            f"Invalid key_category: '{key_category_cleaned}'. Valid categories: {required_categories} (searching for key_name: '{key_name_cleaned}')",
        )

    # Check that all required categories exist in reactives_shiny
    missing_categories = []
    for category_name in required_categories:
        if category_name not in reactives_shiny:
            missing_categories.append(category_name)
        elif not isinstance(reactives_shiny[category_name], dict):
            raise ValueError(
                f"Category '{category_name}' must be a dictionary, got {type(reactives_shiny[category_name]).__name__} (searching for key_name: '{key_name_cleaned}' in key_category: '{key_category_cleaned}')",
            )

    if missing_categories:
        available_categories = list(reactives_shiny.keys())
        raise ValueError(
            f"Required categories missing from reactives_shiny: {missing_categories}. Available categories: {available_categories} (searching for key_name: '{key_name_cleaned}' in key_category: '{key_category_cleaned}')",
        )

    # Business logic validation - get the specific category
    category_dict = reactives_shiny.get(key_category_cleaned)
    if category_dict is None:
        # This should not happen due to above validation, but defensive programming
        available_categories = list(reactives_shiny.keys())
        raise KeyError(
            f"Category '{key_category_cleaned}' not found in reactives_shiny. Available categories: {available_categories} (searching for key_name: '{key_name_cleaned}')",
        )

    # Validate that the specific key exists in the category
    if key_name_cleaned not in category_dict:
        available_keys = list(category_dict.keys())
        if not available_keys:
            raise KeyError(
                f"Key '{key_name_cleaned}' not found in category '{key_category_cleaned}' - category is empty (no keys available)",
            )
        # Provide helpful suggestions if key is similar to existing keys
        similar_keys = [
            key
            for key in available_keys
            if key_name_cleaned.lower() in key.lower() or key.lower() in key_name_cleaned.lower()
        ]
        if similar_keys:
            raise KeyError(
                f"Key '{key_name_cleaned}' not found in category '{key_category_cleaned}'. Available keys: {available_keys}. Similar keys found: {similar_keys}",
            )
        raise KeyError(
            f"Key '{key_name_cleaned}' not found in category '{key_category_cleaned}'. Available keys: {available_keys}",
        )

    # Get the reactive variable safely
    reactive_variable = category_dict[key_name_cleaned]

    # Validate reactive variable exists and is not None
    if reactive_variable is None:
        raise ValueError(
            f"Reactive variable is None for key_name: '{key_name_cleaned}' in key_category: '{key_category_cleaned}' - reactive variable was not properly initialized",
        )

    # Validate that reactive variable has the required 'get' method
    if not hasattr(reactive_variable, "get"):
        available_methods = [
            method for method in dir(reactive_variable) if not method.startswith("_")
        ]
        raise TypeError(
            f"Reactive variable for key_name: '{key_name_cleaned}' in key_category: '{key_category_cleaned}' does not have 'get' method. Object type: {type(reactive_variable).__name__}. Available methods: {available_methods}",
        )

    # Validate that the 'get' method is callable
    if not callable(reactive_variable.get):
        raise TypeError(
            f"Reactive variable 'get' attribute for key_name: '{key_name_cleaned}' in key_category: '{key_category_cleaned}' is not callable. Object type: {type(reactive_variable).__name__}. 'get' type: {type(reactive_variable.get).__name__}",
        )

    # Attempt to retrieve the value with comprehensive error handling
    try:
        retrieved_value = reactive_variable.get()

        # Log successful retrieval for debugging purposes
        _logger.debug(
            "Successfully retrieved value from reactive variable",
            extra={
                "key_name": key_name_cleaned,
                "key_category": key_category_cleaned,
                "value_type": type(retrieved_value).__name__,
            },
        )

        # Return the actual value (including None if that's what was stored)
        return retrieved_value

    except AttributeError as exc:
        # Handle cases where get() method doesn't exist or has issues
        raise RuntimeError(
            f"AttributeError accessing reactive value for key_name: '{key_name_cleaned}' in key_category: '{key_category_cleaned}'. Reactive variable type: {type(reactive_variable).__name__}. Error: {exc}",
        ) from exc

    except RuntimeError as exc:
        # Handle Shiny reactive runtime errors (e.g., reactive context issues)
        raise RuntimeError(
            f"RuntimeError accessing reactive value for key_name: '{key_name_cleaned}' in key_category: '{key_category_cleaned}'. This may indicate a reactive context issue. Error: {exc}",
        ) from exc

    except ValueError as exc:
        # Handle value-related errors during retrieval
        raise RuntimeError(
            f"ValueError accessing reactive value for key_name: '{key_name_cleaned}' in key_category: '{key_category_cleaned}'. Error: {exc}",
        ) from exc

    except TypeError as exc:
        # Handle type-related errors during retrieval
        raise RuntimeError(
            f"TypeError accessing reactive value for key_name: '{key_name_cleaned}' in key_category: '{key_category_cleaned}'. Reactive variable type: {type(reactive_variable).__name__}. Error: {exc}",
        ) from exc

    except KeyError as exc:
        # Handle key-related errors during retrieval (though this should be rare for get())
        raise RuntimeError(
            f"KeyError accessing reactive value for key_name: '{key_name_cleaned}' in key_category: '{key_category_cleaned}'. Error: {exc}",
        ) from exc

    except Exception as exc:
        # Handle any other unexpected errors with full context
        exc_type = type(exc).__name__
        exc_module = getattr(type(exc), "__module__", "unknown")
        reactive_type = type(reactive_variable).__name__
        reactive_module = getattr(type(reactive_variable), "__module__", "unknown")

        raise RuntimeError(
            f"Unexpected {exc_type} error accessing reactive value for key_name: '{key_name_cleaned}' in key_category: '{key_category_cleaned}'. Reactive variable type: {reactive_type} from {reactive_module}. Exception type: {exc_type} from {exc_module}. Error: {exc}",
        ) from exc


def set_value_to_reactives_shiny(
    reactives_shiny: Any,
    key_name: Any,
    key_category: Any,
    input_value: Any,
) -> Any:
    """Set value to a variable within reactives dictionary using specific category with defensive programming.

    Args:
        reactives_shiny: Dictionary containing reactive values organized by category
        key_name: The key name to search for in the specified category
        key_category: The category to search in ("User_Inputs_Shiny", "Inner_Variables_Shiny",
                     "Triggers_Shiny", or "Visual_Objects_Shiny")
        input_value: The value to set in the reactive variable

    Returns
    -------
        dict: Updated reactives dictionary

    Raises
    ------
        KeyError: If key_name is not found in the specified category
        ValueError: If reactives_shiny is None, invalid, or key_category is invalid
    """
    # Input validation with early returns
    validation_result, validation_message = validate_reactives_shiny_structure(reactives_shiny)
    if not validation_result:
        error_message = f"Reactives structure validation failed: {validation_message}"
        _logger.error(error_message)
        raise ValueError(error_message)

    validation_result, validation_message = validate_category_name(key_category)
    if not validation_result:
        error_message = f"Category validation failed: {validation_message}"
        _logger.error(error_message)
        raise ValueError(error_message)

    # Business logic validation - get category
    category_dict = reactives_shiny.get(key_category)
    if category_dict is None:
        available_categories = list(reactives_shiny.keys())
        error_message = (
            f"Category '{key_category}' not found. Available categories: {available_categories}"
        )
        _logger.error(error_message)
        raise KeyError(error_message)

    # Validate key access
    validation_result, validation_message = validate_reactive_key_access(
        category_dict,
        key_name,
        key_category,
    )
    if not validation_result:
        error_message = f"Key access validation failed: {validation_message}"
        _logger.error(error_message)
        raise KeyError(error_message)

    # Get the reactive variable safely
    reactive_variable = category_dict[key_name]

    # Defensive programming - safe value setting
    if reactive_variable is None:
        error_message = f"Reactive variable '{key_name}' in category '{key_category}' is None"
        _logger.error(error_message)
        raise ValueError(error_message)

    if not hasattr(reactive_variable, "set"):
        error_message = f"Reactive variable '{key_name}' in category '{key_category}' does not have 'set' method"
        _logger.error(error_message)
        raise ValueError(error_message)

    # Only use try-except for reactive value setting that might fail unpredictably
    try:
        reactive_variable.set(input_value)
        _logger.debug(
            "Set value in reactive variable",
            extra={
                "key_category": key_category,
                "key_name": key_name,
                "input_value": str(input_value),
            },
        )
        return reactives_shiny
    except Exception as exc:
        error_message = f"Unexpected error setting reactive value for key '{key_name}' in category '{key_category}' to '{input_value}': {exc}"
        _logger.error(error_message)
        raise RuntimeError(error_message) from exc


def update_visual_object_in_reactives(
    reactives_shiny: Any,
    chart_key: str,
    figure: Any,
) -> None:
    """Update a Plotly figure stored in the Visual_Objects_Shiny reactive category.

    Called from any dashboard subtab ``render_widget`` function immediately
    before returning a newly computed figure, so the latest chart is always
    available in the shared reactive state for downstream consumers such as the
    PDF report pipeline.

    Uses defensive programming with early returns: no exception is raised on
    failure so that the chart still renders in the UI even if the reactive
    update cannot be completed.

    Parameters
    ----------
    reactives_shiny : Any
        Shared reactive state dictionary.
    chart_key : str
        Key within ``Visual_Objects_Shiny`` to update, e.g.
        ``"Chart_Portfolio_Analysis"``.
    figure : Any
        Plotly ``go.Figure`` (or compatible) object to store.
    """
    # Input validation with early returns
    if reactives_shiny is None:
        _logger.debug("update_visual_object_in_reactives: reactives_shiny is None")
        return
    if not isinstance(reactives_shiny, dict):
        _logger.debug(
            "update_visual_object_in_reactives: reactives_shiny is not a dict (%s)",
            type(reactives_shiny).__name__,
        )
        return
    if not chart_key or not isinstance(chart_key, str):
        _logger.debug("update_visual_object_in_reactives: invalid chart_key '%s'", chart_key)
        return
    if figure is None:
        _logger.debug(
            "update_visual_object_in_reactives: figure is None for key '%s'",
            chart_key,
        )
        return

    # Configuration validation
    visual_objects = reactives_shiny.get("Visual_Objects_Shiny")
    if not isinstance(visual_objects, dict):
        _logger.debug(
            "update_visual_object_in_reactives: Visual_Objects_Shiny missing or is not a dict",
        )
        return

    reactive_value = visual_objects.get(chart_key)
    if reactive_value is None:
        _logger.debug(
            "update_visual_object_in_reactives: key '%s' not found in Visual_Objects_Shiny",
            chart_key,
        )
        return
    if not hasattr(reactive_value, "set"):
        _logger.debug(
            "update_visual_object_in_reactives: reactive for '%s' has no .set() method",
            chart_key,
        )
        return

    # Only use try-except for unpredictable reactive .set() operation
    try:
        reactive_value.set(figure)
        _logger.debug(
            "update_visual_object_in_reactives: updated Visual_Objects_Shiny['%s']",
            chart_key,
        )
    except Exception as exc:
        _logger.warning(
            "update_visual_object_in_reactives: could not update '%s': %s",
            chart_key,
            exc,
        )


def safe_get_reactive_value(reactive_input: Any, default_value: Any = None) -> Any:
    """Safely get value from reactive input with proper error handling.

    Args:
        reactive_input: Shiny reactive input object
        default_value: Default value to return if input is None or inaccessible

    Returns
    -------
        Value from reactive input or default value
    """
    # Input validation with early returns
    if reactive_input is None:
        return default_value

    # Only use try-except for reactive value access that might fail unpredictably
    try:
        Reactive_Value = reactive_input()
        return Reactive_Value if Reactive_Value is not None else default_value
    except Exception:
        return default_value


def safe_get_value_from_shiny_input_text(
    input_reactive_object: Any,
    default_value: str = "",
) -> str:
    """Safely retrieve and validate text input values following coding standards.

    Args:
        input_reactive_object: The actual reactive input object (e.g., input.input_ID_...)
        default_value: Default value if input is invalid or None

    Returns
    -------
        str: Validated text value
    """
    # Input validation with early returns
    if input_reactive_object is None:
        return default_value

    try:
        # Get the reactive value using () to call the reactive function
        raw_value = input_reactive_object()

        # Configuration validation - ensure value exists
        if raw_value is None:
            return default_value

        # Input validation - convert to string and clean
        cleaned_value = raw_value.strip() if isinstance(raw_value, str) else str(raw_value).strip()

        # Return validated value
        return cleaned_value or default_value

    except (AttributeError, RuntimeError, ValueError, TypeError):
        # Defensive programming - handle all expected errors
        return default_value


def safe_get_value_from_shiny_input_numeric(
    input_reactive_object: Any,
    default_value: float = 0.0,
) -> float:
    """Safely retrieve and validate numeric input values following coding standards.

    Args:
        input_reactive_object: The actual reactive input object (e.g., input.input_ID_...)
        default_value: Default value if input is invalid or None

    Returns
    -------
        float: Validated numeric value
    """
    # Input validation with early returns
    if input_reactive_object is None:
        return default_value

    try:
        # Get the reactive value using () to call the reactive function
        raw_value = input_reactive_object()

        # Configuration validation - ensure value exists
        if raw_value is None:
            return default_value

        # Input validation - handle different value types
        if isinstance(raw_value, (int, float)):
            return max(0.0, float(raw_value))

        if isinstance(raw_value, str):
            # Extract numeric value from formatted currency string
            cleaned_value = raw_value.replace("$", "").replace(",", "").strip()
            if cleaned_value:
                try:
                    numeric_value = float(cleaned_value)
                    return max(0.0, numeric_value)
                except (ValueError, TypeError):
                    return default_value
            else:
                return default_value

        # Business logic validation - default for unexpected types
        return default_value

    except (AttributeError, RuntimeError, ValueError, TypeError):
        # Defensive programming - handle all expected errors
        return default_value


def get_value_from_shiny_input_text(input_reactive_object: Any) -> str:
    """Retrieve and validate string value from shiny text_input with explicit error handling.

    This function retrieves text values from Shiny reactive inputs and validates them,
    throwing explicit errors for invalid inputs rather than returning default values.
    The error messages include the reactive input identifier for better debugging.

    Args:
        input_reactive_object: The actual reactive input object (e.g., input.input_ID_...)

    Returns
    -------
        str: Validated text value (guaranteed to be a valid string)

    Raises
    ------
        ValueError: If input_reactive_object is None, invalid, or contains empty text
        RuntimeError: If reactive value retrieval fails
        TypeError: If retrieved value cannot be converted to string
    """
    # Input validation with early returns
    if input_reactive_object is None:
        raise ValueError("input_reactive_object cannot be None")

    # Configuration validation - ensure input object has callable interface
    if not callable(input_reactive_object):
        raise ValueError(
            f"input_reactive_object must be callable, got {type(input_reactive_object).__name__}",
        )

    # Extract reactive input identifier for error reporting using simplified but effective approach
    reactive_input_identifier = "Unknown"
    try:
        # Method 1: Enhanced source code parsing - most reliable for Shiny inputs
        import inspect
        import linecache
        import re

        frame = inspect.currentframe()
        try:
            # Get the calling frame where this function was invoked
            caller_frame = frame.f_back if frame is not None else None
            if caller_frame:
                filename = caller_frame.f_code.co_filename
                line_number = caller_frame.f_lineno

                # Get the actual source code line that called this function
                source_line = linecache.getline(filename, line_number).strip()

                # Enhanced patterns to match various input access styles
                input_patterns = [
                    # Direct input access: input.input_ID_...
                    r"input\s*\.\s*(input_ID_[a-zA-Z0-9_]+)",
                    # Function call with input: get_value_from_shiny_input_text(input.input_ID_...)
                    r"get_value_from_shiny_input_text\s*\(\s*input\s*\.\s*(input_ID_[a-zA-Z0-9_]+)",
                    # Getattr style: getattr(input, 'input_ID_...')
                    r'getattr\s*\(\s*input\s*,\s*["\']?(input_ID_[a-zA-Z0-9_]+)["\']?\s*\)',
                    # Variable assignment: var = input.input_ID_...
                    r"=\s*input\s*\.\s*(input_ID_[a-zA-Z0-9_]+)",
                    # Function parameter: func(input.input_ID_...)
                    r"\(\s*input\s*\.\s*(input_ID_[a-zA-Z0-9_]+)",
                ]

                # Try each pattern to find the identifier
                for pattern in input_patterns:
                    match = re.search(pattern, source_line)
                    if match:
                        reactive_input_identifier = match.group(1)
                        break

                # If not found in current line, check surrounding lines
                if reactive_input_identifier == "Unknown" and line_number > 1:
                    # Check previous lines for multi-line function calls
                    for line_offset in range(1, 4):  # Check 3 lines above
                        prev_line_number = line_number - line_offset
                        if prev_line_number > 0:
                            prev_source_line = linecache.getline(filename, prev_line_number).strip()
                            for pattern in input_patterns:
                                match = re.search(pattern, prev_source_line)
                                if match:
                                    reactive_input_identifier = match.group(1)
                                    break
                            if reactive_input_identifier != "Unknown":
                                break

        finally:
            del frame  # Prevent reference cycles

        # Method 2: Frame variable inspection - backup method
        if reactive_input_identifier == "Unknown":
            frame = inspect.currentframe()
            try:
                caller_frame = frame.f_back if frame is not None else None
                max_depth = 3  # Limit depth to prevent performance issues
                current_depth = 0

                while (
                    caller_frame
                    and reactive_input_identifier == "Unknown"
                    and current_depth < max_depth
                ):
                    current_depth += 1

                    # Check local variables in the calling frame
                    if caller_frame.f_locals:
                        for var_name, var_value in caller_frame.f_locals.items():
                            # Direct match: variable is our reactive object and has input_ID_ in name
                            if var_value is input_reactive_object and "input_ID_" in str(var_name):
                                reactive_input_identifier = str(var_name)
                                break

                            # Input object inspection: look for input objects
                            if var_name == "input" and hasattr(var_value, "__dict__"):
                                # Check all attributes of the input object
                                try:
                                    for attr_name in dir(var_value):
                                        if attr_name.startswith("input_ID_") and hasattr(
                                            var_value,
                                            attr_name,
                                        ):
                                            attr_obj = getattr(var_value, attr_name)
                                            if attr_obj is input_reactive_object:
                                                reactive_input_identifier = attr_name
                                                break
                                except (AttributeError, TypeError):
                                    pass

                                if reactive_input_identifier != "Unknown":
                                    break

                    # Move up one frame
                    caller_frame = caller_frame.f_back

            finally:
                del frame  # Prevent reference cycles

        # Method 3: Object introspection - check common attributes
        if reactive_input_identifier == "Unknown":
            # Check standard identifier attributes
            identifier_attributes = ["__name__", "_name", "name", "_id", "id", "_key", "key"]
            for attr_name in identifier_attributes:
                if hasattr(input_reactive_object, attr_name):
                    try:
                        attr_value = getattr(input_reactive_object, attr_name)
                        if isinstance(attr_value, str) and "input_ID_" in attr_value:
                            reactive_input_identifier = attr_value
                            break
                    except (AttributeError, TypeError):
                        pass

        # Method 4: String representation analysis
        if reactive_input_identifier == "Unknown":
            try:
                # Check object representation
                obj_repr = repr(input_reactive_object)
                obj_str = str(input_reactive_object)

                # Look for input_ID_ patterns in representations
                combined_text = f"{obj_repr} {obj_str}"
                match = re.search(r"input_ID_[a-zA-Z0-9_]+", combined_text)
                if match:
                    reactive_input_identifier = match.group(0)

            except Exception:
                pass

        # Method 5: Enhanced fallback with more context
        if reactive_input_identifier == "Unknown":
            try:
                obj_type = type(input_reactive_object).__name__
                obj_module = getattr(type(input_reactive_object), "__module__", "unknown")

                # Try to get a more meaningful identifier from the calling context
                frame = inspect.currentframe()
                try:
                    caller_frame = frame.f_back if frame is not None else None
                    if caller_frame and caller_frame.f_code:
                        func_name = caller_frame.f_code.co_name
                        filename = caller_frame.f_code.co_filename
                        line_number = caller_frame.f_lineno

                        # Extract just the filename without path
                        import os

                        base_filename = os.path.basename(filename)

                        reactive_input_identifier = (
                            f"Unknown_input_in_{func_name}_at_{base_filename}:{line_number}"
                        )
                    else:
                        reactive_input_identifier = (
                            f"Unknown_reactive_input_{obj_type}_from_{obj_module}"
                        )
                finally:
                    del frame

            except Exception:
                reactive_input_identifier = "Unknown_reactive_input_object"

    except Exception:
        # Final fallback if all extraction methods fail
        reactive_input_identifier = "Unknown_reactive_input_object"

    try:
        # Get the reactive value using () to call the reactive function
        raw_value = input_reactive_object()

    except (AttributeError, RuntimeError) as exc:
        # Defensive programming - handle reactive access failures
        raise RuntimeError(
            f"Failed to retrieve value from reactive input '{reactive_input_identifier}': {exc}",
        ) from exc
    except Exception as exc:
        # Handle any other unexpected errors during reactive access
        raise RuntimeError(
            f"Unexpected error accessing reactive input '{reactive_input_identifier}': {exc}",
        ) from exc

    # Configuration validation - ensure value exists
    if raw_value is None:
        raise ValueError(
            f"Retrieved reactive value is None for input '{reactive_input_identifier}' - no text input provided",
        )

    # Business logic validation - convert to string and clean
    try:
        cleaned_value = raw_value.strip() if isinstance(raw_value, str) else str(raw_value).strip()

    except (ValueError, TypeError) as exc:
        # Handle string conversion failures
        raise TypeError(
            f"Cannot convert retrieved value to string for input '{reactive_input_identifier}': {type(raw_value).__name__} - {exc}",
        ) from exc

    # Final validation - ensure we have a non-empty string
    if not cleaned_value:
        raise ValueError(
            f"Text input '{reactive_input_identifier}' is empty or contains only whitespace",
        )

    # Return validated value
    return cleaned_value


def get_value_from_shiny_input_numeric(input_reactive_object: Any) -> float:
    """Retrieve and validate numeric value from shiny numeric_input with explicit error handling.

    This function retrieves numeric values from Shiny reactive inputs and validates them,
    throwing explicit errors for invalid inputs rather than returning default values.
    The error messages include the reactive input identifier for better debugging.

    Args:
        input_reactive_object: The actual reactive input object (e.g., input.input_ID_...)

    Returns
    -------
        float: Validated numeric value (guaranteed to be a valid non-negative float)

    Raises
    ------
        ValueError: If input_reactive_object is None, invalid, or contains invalid numeric data
        RuntimeError: If reactive value retrieval fails
        TypeError: If retrieved value cannot be converted to float
    """
    # Input validation with early returns
    if input_reactive_object is None:
        raise ValueError("input_reactive_object cannot be None")

    # Configuration validation - ensure input object has callable interface
    if not callable(input_reactive_object):
        raise ValueError(
            f"input_reactive_object must be callable, got {type(input_reactive_object).__name__}",
        )

    # Extract reactive input identifier for error reporting using simplified but effective approach
    reactive_input_identifier = "Unknown"
    try:
        # Method 1: Enhanced source code parsing - most reliable for Shiny inputs
        import inspect
        import linecache
        import re

        frame = inspect.currentframe()
        try:
            # Get the calling frame where this function was invoked
            caller_frame = frame.f_back if frame is not None else None
            if caller_frame:
                filename = caller_frame.f_code.co_filename
                line_number = caller_frame.f_lineno

                # Get the actual source code line that called this function
                source_line = linecache.getline(filename, line_number).strip()

                # Enhanced patterns to match various input access styles
                input_patterns = [
                    # Direct input access: input.input_ID_...
                    r"input\s*\.\s*(input_ID_[a-zA-Z0-9_]+)",
                    # Function call with input: get_value_from_shiny_input_numeric(input.input_ID_...)
                    r"get_value_from_shiny_input_numeric\s*\(\s*input\s*\.\s*(input_ID_[a-zA-Z0-9_]+)",
                    # Getattr style: getattr(input, 'input_ID_...')
                    r'getattr\s*\(\s*input\s*,\s*["\']?(input_ID_[a-zA-Z0-9_]+)["\']?\s*\)',
                    # Variable assignment: var = input.input_ID_...
                    r"=\s*input\s*\.\s*(input_ID_[a-zA-Z0-9_]+)",
                    # Function parameter: func(input.input_ID_...)
                    r"\(\s*input\s*\.\s*(input_ID_[a-zA-Z0-9_]+)",
                ]

                # Try each pattern to find the identifier
                for pattern in input_patterns:
                    match = re.search(pattern, source_line)
                    if match:
                        reactive_input_identifier = match.group(1)
                        break

                # If not found in current line, check surrounding lines
                if reactive_input_identifier == "Unknown" and line_number > 1:
                    # Check previous lines for multi-line function calls
                    for line_offset in range(1, 4):  # Check 3 lines above
                        prev_line_number = line_number - line_offset
                        if prev_line_number > 0:
                            prev_source_line = linecache.getline(filename, prev_line_number).strip()
                            for pattern in input_patterns:
                                match = re.search(pattern, prev_source_line)
                                if match:
                                    reactive_input_identifier = match.group(1)
                                    break
                            if reactive_input_identifier != "Unknown":
                                break

        finally:
            del frame  # Prevent reference cycles

        # Method 2: Frame variable inspection - backup method
        if reactive_input_identifier == "Unknown":
            frame = inspect.currentframe()
            try:
                caller_frame = frame.f_back if frame is not None else None
                max_depth = 3  # Limit depth to prevent performance issues
                current_depth = 0

                while (
                    caller_frame
                    and reactive_input_identifier == "Unknown"
                    and current_depth < max_depth
                ):
                    current_depth += 1

                    # Check local variables in the calling frame
                    if caller_frame.f_locals:
                        for var_name, var_value in caller_frame.f_locals.items():
                            # Direct match: variable is our reactive object and has input_ID_ in name
                            if var_value is input_reactive_object and "input_ID_" in str(var_name):
                                reactive_input_identifier = str(var_name)
                                break

                            # Input object inspection: look for input objects
                            if var_name == "input" and hasattr(var_value, "__dict__"):
                                # Check all attributes of the input object
                                try:
                                    for attr_name in dir(var_value):
                                        if attr_name.startswith("input_ID_") and hasattr(
                                            var_value,
                                            attr_name,
                                        ):
                                            attr_obj = getattr(var_value, attr_name)
                                            if attr_obj is input_reactive_object:
                                                reactive_input_identifier = attr_name
                                                break
                                except (AttributeError, TypeError):
                                    pass

                                if reactive_input_identifier != "Unknown":
                                    break

                    # Move up one frame
                    caller_frame = caller_frame.f_back

            finally:
                del frame  # Prevent reference cycles

        # Method 3: Object introspection - check common attributes
        if reactive_input_identifier == "Unknown":
            # Check standard identifier attributes
            identifier_attributes = ["__name__", "_name", "name", "_id", "id", "_key", "key"]
            for attr_name in identifier_attributes:
                if hasattr(input_reactive_object, attr_name):
                    try:
                        attr_value = getattr(input_reactive_object, attr_name)
                        if isinstance(attr_value, str) and "input_ID_" in attr_value:
                            reactive_input_identifier = attr_value
                            break
                    except (AttributeError, TypeError):
                        pass

        # Method 4: String representation analysis
        if reactive_input_identifier == "Unknown":
            try:
                # Check object representation
                obj_repr = repr(input_reactive_object)
                obj_str = str(input_reactive_object)

                # Look for input_ID_ patterns in representations
                combined_text = f"{obj_repr} {obj_str}"
                match = re.search(r"input_ID_[a-zA-Z0-9_]+", combined_text)
                if match:
                    reactive_input_identifier = match.group(0)

            except Exception:
                pass

        # Method 5: Enhanced fallback with more context
        if reactive_input_identifier == "Unknown":
            try:
                obj_type = type(input_reactive_object).__name__
                obj_module = getattr(type(input_reactive_object), "__module__", "unknown")

                # Try to get a more meaningful identifier from the calling context
                frame = inspect.currentframe()
                try:
                    caller_frame = frame.f_back if frame is not None else None
                    if caller_frame and caller_frame.f_code:
                        func_name = caller_frame.f_code.co_name
                        filename = caller_frame.f_code.co_filename
                        line_number = caller_frame.f_lineno

                        # Extract just the filename without path
                        import os

                        base_filename = os.path.basename(filename)

                        reactive_input_identifier = (
                            f"Unknown_input_in_{func_name}_at_{base_filename}:{line_number}"
                        )
                    else:
                        reactive_input_identifier = (
                            f"Unknown_reactive_input_{obj_type}_from_{obj_module}"
                        )
                finally:
                    del frame

            except Exception:
                reactive_input_identifier = "Unknown_reactive_input_object"

    except Exception:
        # Final fallback if all extraction methods fail
        reactive_input_identifier = "Unknown_reactive_input_object"

    try:
        # Get the reactive value using () to call the reactive function
        raw_value = input_reactive_object()

    except (AttributeError, RuntimeError) as exc:
        # Defensive programming - handle reactive access failures
        raise RuntimeError(
            f"Failed to retrieve value from reactive input '{reactive_input_identifier}': {exc}",
        ) from exc
    except Exception as exc:
        # Handle any other unexpected errors during reactive access
        raise RuntimeError(
            f"Unexpected error accessing reactive input '{reactive_input_identifier}': {exc}",
        ) from exc

    # Configuration validation - ensure value exists
    if raw_value is None:
        raise ValueError(
            f"Retrieved reactive value is None for input '{reactive_input_identifier}' - no numeric input provided",
        )

    # Business logic validation - handle different value types and convert to float
    if isinstance(raw_value, (int, float)):
        try:
            numeric_value = float(raw_value)
            # Ensure non-negative value for financial data
            if numeric_value < 0:
                raise ValueError(
                    f"Numeric input '{reactive_input_identifier}' contains negative value: {numeric_value}",
                )
            return numeric_value
        except (ValueError, TypeError) as exc:
            raise TypeError(
                f"Cannot convert numeric value to float for input '{reactive_input_identifier}': {type(raw_value).__name__} - {exc}",
            ) from exc

    elif isinstance(raw_value, str):
        try:
            # Extract numeric value from formatted currency string
            cleaned_value = raw_value.replace("$", "").replace(",", "").strip()

            if not cleaned_value:
                raise ValueError(
                    f"Numeric input '{reactive_input_identifier}' is empty or contains only formatting characters",
                )

            numeric_value = float(cleaned_value)

            # Ensure non-negative value for financial data
            if numeric_value < 0:
                raise ValueError(
                    f"Numeric input '{reactive_input_identifier}' contains negative value: {numeric_value}",
                )

            return numeric_value

        except ValueError as exc:
            # Re-raise ValueError with context
            if "negative value" in str(exc):
                raise exc
            raise ValueError(
                f"Cannot parse numeric value from string for input '{reactive_input_identifier}': '{raw_value}' - {exc}",
            ) from exc
        except (TypeError, AttributeError) as exc:
            raise TypeError(
                f"Cannot process string value for input '{reactive_input_identifier}': {type(raw_value).__name__} - {exc}",
            ) from exc

    else:
        # Handle unexpected types
        raise TypeError(
            f"Numeric input '{reactive_input_identifier}' has unexpected type: {type(raw_value).__name__} (value: {raw_value})",
        )


# =============================================================================
# Client Input → Reactive State Synchronisation
# =============================================================================


def _reactive_key_to_input_id(reactive_key: str) -> str:
    """Derive the Shiny input element ID from a ``User_Inputs_Shiny`` reactive key.

    Applies the project naming convention::

        Input_Tab_<rest>  →  input_ID_tab_<rest_lowercased>

    Args:
        reactive_key: Key from the ``User_Inputs_Shiny`` sub-dict, e.g.
            ``Input_Tab_clients_Subtab_clients_Personal_Info_client_Primary_Name``

    Returns
    -------
        str: Corresponding Shiny input element ID, e.g.
            ``input_ID_tab_clients_subtab_clients_personal_info_client_primary_name``

    Raises
    ------
        ValueError: If ``reactive_key`` does not start with ``"Input_Tab_"``.
    """
    # Configuration validation
    if not reactive_key.startswith("Input_Tab_"):
        raise ValueError(f"reactive_key must start with 'Input_Tab_', got: '{reactive_key}'")

    # Business logic: strip known prefix, lowercase the rest, re-attach new prefix
    suffix = reactive_key[len("Input_Tab_") :]
    return f"input_ID_tab_{suffix.lower()}"


def _register_single_input_observer(
    input: Any,  # noqa: A002
    reactives_shiny: dict[str, Any],
    input_id: str,
    reactive_key: str,
) -> None:
    """Mirror one Shiny UI input into ``User_Inputs_Shiny[reactive_key]``.

    Registers a ``reactive.effect`` that keeps
    ``reactives_shiny["User_Inputs_Shiny"][reactive_key]`` in sync with the
    ``input_id`` Shiny UI element.

    Uses a dedicated function scope to avoid the Python closure-over-loop-variable
    pitfall: each call binds ``input_id`` and ``reactive_key`` in its own local
    scope, so the inner ``_observer`` captures the correct pair of values.

    Args:
        input: The Shiny ``input`` proxy object available inside a module server.
        reactives_shiny: The central reactive state dictionary.
        input_id: Shiny input element ID, e.g.
            ``input_ID_tab_clients_subtab_clients_personal_info_client_primary_name``
        reactive_key: Corresponding key in ``User_Inputs_Shiny``, e.g.
            ``Input_Tab_clients_Subtab_clients_Personal_Info_client_Primary_Name``
    """
    # Resolve the target reactive.Value once at registration time so the observer
    # body performs no dict look-ups on every reactive invalidation cycle.
    target_reactive = reactives_shiny["User_Inputs_Shiny"].get(reactive_key)

    if target_reactive is None:
        _logger.warning(
            "Skipping observer registration: reactive key not found in User_Inputs_Shiny",
            extra={"reactive_key": reactive_key, "input_id": input_id},
        )
        return

    @reactive.effect
    def _observer() -> None:
        """Forward the current UI input value to the corresponding User_Inputs_Shiny entry."""
        value = safe_get_shiny_input_value(input, input_id)

        # Input validation with early return — do not overwrite with None
        if value is None:
            return

        # Only use try-except for reactive.Value.set() which can fail unpredictably
        try:
            target_reactive.set(value)
            _logger.debug(
                "Synced client input to User_Inputs_Shiny",
                extra={"input_id": input_id, "reactive_key": reactive_key},
            )
        except Exception as exc:
            _logger.error(
                "Failed to sync client input to User_Inputs_Shiny",
                extra={
                    "input_id": input_id,
                    "reactive_key": reactive_key,
                    "error": str(exc),
                },
            )


def register_client_input_observers(
    input: Any,  # noqa: A002
    reactives_shiny: dict[str, Any],
) -> None:
    """Keep ``reactives_shiny`` in sync with all client-related UI inputs.

    Registers reactive observers that fire whenever the user edits a value in
    the dashboard, covering all four client input domains in ``User_Inputs_Shiny``.

    Must be called once from within a Shiny module server (or app server) so that
    the ``@reactive.effect`` decorators are registered inside the active session
    context.  Covers all four client input domains in ``User_Inputs_Shiny``:

    - **Personal Info** — name, age, marital status, gender, risk tolerance, etc.
    - **Assets** — taxable, tax-deferred, and tax-free amounts (primary & partner)
    - **Goals** — essential, important, and aspirational goals (primary & partner)
    - **Income** — Social Security, pension, annuity, and other income (primary & partner)

    For every key in ``User_Inputs_Shiny`` that matches the pattern
    ``Input_Tab_clients_*``, this function registers an observer that:

    1. Reads the current UI input value via the corresponding
       ``input_ID_tab_clients_*`` Shiny element.
    2. Writes the value into ``reactives_shiny["User_Inputs_Shiny"][reactive_key]``
       so it is available to all other dashboard modules through the shared
       reactive state dictionary.

    The input-ID derivation follows the project naming convention::

        Input_Tab_clients_Subtab_clients_Personal_Info_client_Primary_Name
            →  input_ID_tab_clients_subtab_clients_personal_info_client_primary_name

    Args:
        input: The Shiny ``input`` proxy object available inside a module server or
            app server function.
        reactives_shiny: The central reactive state dictionary as returned by
            :func:`initialize_reactives_shiny`.

    Raises
    ------
        ValueError: If ``input`` is ``None`` or ``reactives_shiny`` has an invalid
            structure.

    Examples
    --------
    Call once at the top of each client subtab module server:

    .. code-block:: python

        @module.server
        def subtab_personal_info_server(
            input, output, session, reactives_shiny, data_utils, data_inputs
        ):
            register_client_input_observers(input, reactives_shiny)
            # ... rest of server logic
    """
    # Input validation with early returns
    if input is None:
        raise ValueError("register_client_input_observers: input cannot be None")

    is_valid, error_msg = validate_reactives_shiny_structure(reactives_shiny)
    if not is_valid:
        raise ValueError(f"register_client_input_observers: invalid reactives_shiny — {error_msg}")

    # Configuration validation — gather all client-domain reactive keys
    user_inputs_dict = reactives_shiny["User_Inputs_Shiny"]
    client_reactive_keys = [key for key in user_inputs_dict if key.startswith("Input_Tab_clients_")]

    if not client_reactive_keys:
        _logger.warning(
            "register_client_input_observers: no 'Input_Tab_clients_*' keys found in "
            "User_Inputs_Shiny — no observers registered",
        )
        return

    # Business logic — register one scoped observer per (input_id, reactive_key) pair.
    # _register_single_input_observer creates a new function scope for each pair,
    # preventing the Python closure-over-loop-variable pitfall.
    registered_count = 0
    for reactive_key in client_reactive_keys:
        try:
            input_id = _reactive_key_to_input_id(reactive_key)
        except ValueError as exc:
            _logger.warning(
                "Skipping observer: cannot derive input_id from reactive_key",
                extra={"reactive_key": reactive_key, "error": str(exc)},
            )
            continue

        _register_single_input_observer(
            input=input,
            reactives_shiny=reactives_shiny,
            input_id=input_id,
            reactive_key=reactive_key,
        )
        registered_count += 1

    _logger.info(
        "register_client_input_observers: registered %d client input observers",
        registered_count,
        extra={"registered_count": registered_count},
    )


# =============================================================================
# Portfolio & Simulation Input → Data_Results Synchronisation
# =============================================================================


#: Maps each ``User_Inputs_Shiny`` key prefix to the corresponding base key in
#: ``Data_Results`` (i.e. ``"{base}_Inputs"`` and ``"{base}_Outputs"``).  Order
#: matters: more-specific prefixes must come before less-specific ones so that
#: the longest-match wins when grouping keys by subtab.
_PORTFOLIO_SUBTAB_RESULTS_KEY_MAP: dict[str, str] = {
    "Input_Tab_Portfolios_Subtab_Portfolios_Analysis_": "Portfolio_Analysis",
    "Input_Tab_Portfolios_Subtab_Comparison_": "Portfolio_Comparison",
    "Input_Tab_Portfolios_Subtab_Weights_Analysis_": "Weights_Analysis",
    "Input_Tab_Portfolios_Subtab_Skfolio_": "Portfolio_Optimization_Skfolio",
    "Input_Tab_Results_Subtab_Simulation_": "Portfolio_Simulation",
}


def _register_subtab_snapshot_observer(
    input: Any,  # noqa: A002
    reactives_shiny: dict[str, Any],
    group_keys: list[str],
    results_base_key: str,
) -> None:
    """Snapshot all inputs for one dashboard subtab into ``Data_Results``.

    Registers one ``reactive.effect`` that writes a fresh snapshot dict to
    ``reactives_shiny["Data_Results"]["{results_base_key}_Inputs"]``.

    The snapshot is a plain ``dict`` mapping each ``User_Inputs_Shiny`` reactive
    key to its current UI input value.  Subtab servers read this snapshot via
    ``reactives_shiny["Data_Results"]["{results_base_key}_Inputs"].get()`` and
    write their computed results back into the matching ``"*_Outputs"`` entry.

    Uses a dedicated function scope so each registered observer closure captures
    its own ``group_keys`` and ``results_base_key``, avoiding the Python
    closure-over-loop-variable pitfall.

    Parameters
    ----------
    input : Any
        The Shiny ``input`` proxy available inside a module or app server.
    reactives_shiny : dict[str, Any]
        The central reactive state dictionary.
    group_keys : list[str]
        All ``User_Inputs_Shiny`` keys that belong to this subtab, e.g.
        ``["Input_Tab_Portfolios_Subtab_Comparison_Time_Period", ...]``.
    results_base_key : str
        Base key into ``Data_Results``, e.g. ``"Portfolio_Comparison"``.
        The observer writes to ``Data_Results["{results_base_key}_Inputs"]``.
    """
    inputs_results_key = f"{results_base_key}_Inputs"
    target_reactive = reactives_shiny["Data_Results"].get(inputs_results_key)

    if target_reactive is None:
        _logger.warning(
            "Skipping snapshot observer: Data_Results key not found",
            extra={"inputs_results_key": inputs_results_key, "results_base_key": results_base_key},
        )
        return

    # Pre-resolve (reactive_key → input_id) pairs once at registration time so
    # the inner observer body performs no dict look-ups on every invalidation.
    input_id_pairs: list[tuple[str, str]] = []
    for reactive_key in group_keys:
        try:
            input_id = _reactive_key_to_input_id(reactive_key)
            input_id_pairs.append((reactive_key, input_id))
        except ValueError as exc:
            _logger.warning(
                "Skipping input in snapshot group: cannot derive input_id",
                extra={"reactive_key": reactive_key, "error": str(exc)},
            )

    @reactive.effect
    def _snapshot_observer() -> None:
        """Collect all subtab input values and store as a snapshot dict."""
        snapshot: dict[str, Any] = {}
        for reactive_key, input_id in input_id_pairs:
            value = safe_get_shiny_input_value(input, input_id)
            snapshot[reactive_key] = value

        # Only use try-except for reactive.Value.set() which can fail unpredictably
        try:
            target_reactive.set(snapshot)
            _logger.debug(
                "Snapshot stored in Data_Results",
                extra={
                    "inputs_results_key": inputs_results_key,
                    "num_inputs": len(snapshot),
                },
            )
        except Exception as exc:
            _logger.error(
                "Failed to store snapshot in Data_Results",
                extra={"inputs_results_key": inputs_results_key, "error": str(exc)},
            )


def register_portfolio_results_observers(
    input: Any,  # noqa: A002
    reactives_shiny: dict[str, Any],
) -> None:
    """Keep ``reactives_shiny["Data_Results"]`` in sync with portfolio and simulation UI inputs.

    Registers reactive observers that update ``Data_Results["*_Inputs"]`` entries
    whenever the user changes any portfolio or simulation UI control.

    Must be called once from within the appropriate Shiny module server (or app
    server) so that ``@reactive.effect`` decorators are registered inside the
    active session context.

    **What this function does**

    For each of the five subtabs it:

    1. Finds all ``User_Inputs_Shiny`` keys that belong to that subtab
       (prefix-matched against :data:`_PORTFOLIO_SUBTAB_RESULTS_KEY_MAP`).
    2. Registers one :func:`_register_subtab_snapshot_observer` that reacts
       whenever *any* input in the group changes and stores a fresh snapshot
       dict into ``Data_Results["{subtab}_Inputs"]``.
    3. Registers one :func:`_register_single_input_observer` per input so that
       the corresponding individual ``User_Inputs_Shiny`` entry is also updated.

    **Covered subtabs and Data_Results keys**

    =========================================  ==============================
    Subtab                                     ``Data_Results`` key updated
    =========================================  ==============================
    Portfolio Analysis                         ``Portfolio_Analysis_Inputs``
    Portfolio Comparison                       ``Portfolio_Comparison_Inputs``
    Weights Analysis                           ``Weights_Analysis_Inputs``
    Skfolio Portfolio Optimization             ``Portfolio_Optimization_Skfolio_Inputs``
    Portfolio Simulation                       ``Portfolio_Simulation_Inputs``
    =========================================  ==============================

    Subtab servers read the snapshot from ``Data_Results["*_Inputs"].get()``
    to detect changes and compute results, then write back into
    ``Data_Results["*_Outputs"]``.

    Parameters
    ----------
    input : Any
        The Shiny ``input`` proxy available inside a module or app server.
    reactives_shiny : dict[str, Any]
        The central reactive state dictionary as returned by
        :func:`initialize_reactives_shiny`.

    Raises
    ------
    ValueError
        If ``input`` is ``None`` or ``reactives_shiny`` has an invalid structure.

    Examples
    --------
    Call once at the top of the relevant tab server:

    .. code-block:: python

        @module.server
        def tab_portfolios_server(input, output, session, ...):
            register_portfolio_results_observers(input, reactives_shiny)
            # ... rest of server logic

        @module.server
        def tab_results_server(input, output, session, ...):
            register_portfolio_results_observers(input, reactives_shiny)
    """
    # Input validation with early returns
    if input is None:
        raise ValueError("register_portfolio_results_observers: input cannot be None")

    is_valid, error_msg = validate_reactives_shiny_structure(reactives_shiny)
    if not is_valid:
        raise ValueError(
            f"register_portfolio_results_observers: invalid reactives_shiny — {error_msg}",
        )

    user_inputs_dict = reactives_shiny["User_Inputs_Shiny"]
    data_results_dict = reactives_shiny["Data_Results"]

    # Configuration validation — group User_Inputs_Shiny keys by subtab prefix
    # using longest-match so that e.g. "_Portfolios_Analysis_" is not consumed
    # by the shorter "_Portfolios_" if such a prefix were ever added.
    subtab_groups: dict[str, list[str]] = {
        base_key: [] for base_key in _PORTFOLIO_SUBTAB_RESULTS_KEY_MAP.values()
    }

    for reactive_key in user_inputs_dict:
        for prefix, base_key in _PORTFOLIO_SUBTAB_RESULTS_KEY_MAP.items():
            if reactive_key.startswith(prefix):
                subtab_groups[base_key].append(reactive_key)
                break  # longest-match already guaranteed by dict ordering

    # Business logic — for every subtab group with at least one key:
    # a) register a group snapshot observer → Data_Results["*_Inputs"]
    # b) register individual per-input observers → User_Inputs_Shiny
    total_registered = 0
    for base_key, group_keys in subtab_groups.items():
        if not group_keys:
            _logger.debug(
                "No User_Inputs_Shiny keys found for subtab — skipping",
                extra={"base_key": base_key},
            )
            continue

        inputs_results_key = f"{base_key}_Inputs"
        if inputs_results_key not in data_results_dict:
            _logger.warning(
                "Data_Results key missing — snapshot observer not registered",
                extra={"inputs_results_key": inputs_results_key},
            )
        else:
            _register_subtab_snapshot_observer(
                input=input,
                reactives_shiny=reactives_shiny,
                group_keys=group_keys,
                results_base_key=base_key,
            )

        # Per-input mirroring into User_Inputs_Shiny
        for reactive_key in group_keys:
            try:
                input_id = _reactive_key_to_input_id(reactive_key)
            except ValueError as exc:
                _logger.warning(
                    "Skipping individual observer: cannot derive input_id",
                    extra={"reactive_key": reactive_key, "error": str(exc)},
                )
                continue

            _register_single_input_observer(
                input=input,
                reactives_shiny=reactives_shiny,
                input_id=input_id,
                reactive_key=reactive_key,
            )
            total_registered += 1

    _logger.info(
        "register_portfolio_results_observers: registered %d individual + %d snapshot observers",
        total_registered,
        len([g for g in subtab_groups.values() if g]),
        extra={
            "individual_observers": total_registered,
            "snapshot_observers": len([g for g in subtab_groups.values() if g]),
        },
    )
