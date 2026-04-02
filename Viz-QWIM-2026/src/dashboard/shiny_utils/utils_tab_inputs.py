"""Utility functions for input processing and validation.

Provides functions for handling user inputs, date range calculations,
and parameter validation across dashboard tabs.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any


def validate_module_inputs(
    data_utils: Any = None,
    data_inputs: Any = None,
    reactives_shiny: Any = None,
) -> Any:
    """Validate module server input parameters using defensive programming."""
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


def validate_time_period_selection(time_period_value: Any) -> Any:
    """Validate time period selection using defensive programming."""
    # Input validation
    if time_period_value is None:
        return False, "Time period selection is None"

    if not isinstance(time_period_value, str):
        return False, f"Time period must be string, got {type(time_period_value).__name__}"

    # Configuration validation
    valid_time_periods = {
        "custom",
        "last_1_year",
        "last_3_years",
        "last_5_years",
        "last_10_years",
        "ytd",
    }

    if time_period_value not in valid_time_periods:
        return (
            False,
            f"Invalid time period: {time_period_value}. Must be one of {valid_time_periods}",
        )

    return True, ""


def validate_date_range_input(date_range_input: Any, time_period_mode: Any = "preset") -> Any:
    """Validate date range input using defensive programming."""
    # Input validation with early returns
    if date_range_input is None:
        if time_period_mode == "custom":
            return False, "Custom mode requires date range input"
        return True, ""  # Preset modes can have None date range

    # Configuration validation
    if not isinstance(date_range_input, (list, tuple)):
        return False, f"Date range must be list or tuple, got {type(date_range_input).__name__}"

    if len(date_range_input) != 2:
        return False, f"Date range must have exactly 2 elements, got {len(date_range_input)}"

    # Business logic validation
    start_date = date_range_input[0]
    end_date = date_range_input[1]

    if start_date is None or end_date is None:
        return False, "Date range contains None values"

    # Defensive programming - safe date comparison
    if hasattr(start_date, "date") and hasattr(end_date, "date"):
        start_date_obj = start_date.date() if hasattr(start_date, "date") else start_date
        end_date_obj = end_date.date() if hasattr(end_date, "date") else end_date

        if start_date_obj > end_date_obj:
            return False, f"Start date {start_date_obj} cannot be after end date {end_date_obj}"

    return True, ""


def safe_get_input_value(
    input_object: Any,
    input_identifier: str,
    default_value: Any = None,
) -> Any:
    """Safely get input value using defensive programming and project Shiny standards."""
    # Input validation
    if input_object is None:
        return default_value

    if not isinstance(input_identifier, str):
        return default_value

    if len(input_identifier.strip()) == 0:
        return default_value

    # Only use try-except for Shiny input access that might fail unpredictably
    try:
        # Use project standard: input[["identifier"]].get()
        result_value = input_object[[input_identifier]].get()
        return result_value if result_value is not None else default_value
    except Exception:
        return default_value


def find_date_column_name(data_frame: Any) -> Any:
    """Find the date column name using defensive programming."""
    # Input validation
    if data_frame is None:
        return None

    if not hasattr(data_frame, "columns"):
        return None

    # Configuration validation
    possible_date_columns = ["date", "Date", "DATE"]

    # Defensive programming - safe column checking
    for date_column_candidate in possible_date_columns:
        if date_column_candidate in data_frame.columns:
            return date_column_candidate

    return None


def calculate_preset_date_range(
    time_period_selection: Any,
    max_date_available: Any,
    min_date_available: Any,
) -> Any:
    """Calculate date range for preset time periods using defensive programming."""
    # Input validation
    if not isinstance(time_period_selection, str):
        return None, None

    if max_date_available is None or min_date_available is None:
        return None, None

    # Configuration validation
    preset_calculations = {
        "last_1_year": 365,
        "last_3_years": 365 * 3,
        "last_5_years": 365 * 5,
        "last_10_years": 365 * 10,
    }

    # Business logic validation
    if time_period_selection == "ytd":
        start_date = datetime(max_date_available.year, 1, 1, tzinfo=UTC).date()
        end_date = max_date_available
    elif time_period_selection in preset_calculations:
        days_back = preset_calculations[time_period_selection]
        start_date = max_date_available - timedelta(days=days_back)
        end_date = max_date_available
    else:
        # Fallback to full range
        start_date = min_date_available
        end_date = max_date_available

    # Defensive programming - ensure dates are within available range
    start_date = max(start_date, min_date_available)
    end_date = min(end_date, max_date_available)

    return start_date, end_date
