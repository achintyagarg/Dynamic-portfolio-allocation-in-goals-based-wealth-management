"""Utility functions for data analysis tab operations.

Provides parameter validation and data processing functions for
analytical operations in the dashboard.
"""

from __future__ import annotations

from typing import Any


def validate_data_analysis_parameters(
    data_utils: Any = None,
    data_inputs: Any = None,
    reactives_shiny: Any = None,
) -> Any:
    """Validate parameters for data analysis module using defensive programming."""
    # Input validation with early returns
    if data_utils is None:
        return False, "data_utils parameter is required"

    if data_inputs is None:
        return False, "data_inputs parameter is required"

    if reactives_shiny is None:
        return False, "reactives_shiny parameter is required"

    # Configuration validation
    if not isinstance(data_utils, dict):
        return False, f"data_utils must be dict, got {type(data_utils).__name__}"

    if not isinstance(data_inputs, dict):
        return False, f"data_inputs must be dict, got {type(data_inputs).__name__}"

    if not isinstance(reactives_shiny, dict):
        return False, f"reactives_shiny must be dict, got {type(reactives_shiny).__name__}"

    # Business logic validation
    if "Time_Series_Sample" not in data_inputs:
        return False, "data_inputs missing required 'Time_Series_Sample' key"

    return True, ""
