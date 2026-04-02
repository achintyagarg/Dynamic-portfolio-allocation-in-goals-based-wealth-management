"""
Utilities for the QWIM Dashboard.

This module implements various utilities for the QWIM Dashboard, including:
- Reactive state management (reactives_shiny)
- Data loading and processing (utils_data)
- Visualization utilities (utils_visuals)
- Error handling classes (utils_errors)
- Enhanced UI components (utils_enhanced_ui_components)
- Enhanced formatting (utils_enhanced_formatting)
- Enhanced validation (utils_enhanced_validation)

Exports:
    From reactives_shiny:
        - initialize_reactives_shiny
        - get_value_from_reactives_shiny
        - validate_reactives_shiny_structure
        - safe_get_shiny_input_value

    From utils_data:
        - get_data_inputs
        - get_data_utils

    From utils_errors:
        - Error_Silent_Initialization
        - Error_Dashboard_Initialization
        - Error_Data_Loading
        - Error_Module_Initialization
        - Error_Validation
        - Error_Configuration
"""

from __future__ import annotations

from .reactives_shiny import (
    get_value_from_reactives_shiny,
    initialize_reactives_shiny,
    safe_get_shiny_input_value,
    validate_reactives_shiny_structure,
)
from .utils_data import get_data_inputs, get_data_utils
from .utils_errors_dashboard import (
    DashboardInitializationError,
    DataLoadingError,
    Error_Configuration,
    Error_Dashboard_Initialization,
    Error_Data_Loading,
    Error_Module_Initialization,
    Error_Silent_Initialization,
    Error_Validation,
    ModuleInitializationError,
    # Backward compatibility aliases
    SilentInitializationException,
)


__all__ = [
    "DashboardInitializationError",
    "DataLoadingError",
    "Error_Configuration",
    "Error_Dashboard_Initialization",
    "Error_Data_Loading",
    "Error_Module_Initialization",
    # Error classes (new naming convention)
    "Error_Silent_Initialization",
    "Error_Validation",
    "ModuleInitializationError",
    # Error classes (backward compatibility)
    "SilentInitializationException",
    # Data utilities
    "get_data_inputs",
    "get_data_utils",
    "get_value_from_reactives_shiny",
    # Reactives
    "initialize_reactives_shiny",
    "safe_get_shiny_input_value",
    "validate_reactives_shiny_structure",
]
