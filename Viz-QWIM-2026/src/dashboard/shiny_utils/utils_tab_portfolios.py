"""Utilities for portfolio tab operations."""

from __future__ import annotations

from typing import Any

import pandas as pd


def validate_portfolio_data(portfolio_dataframe: Any, dataset_name: Any = "portfolio data") -> Any:
    """Validate portfolio data meets requirements.

    Parameters
    ----------
    portfolio_dataframe : pd.DataFrame
        DataFrame to validate
    dataset_name : str, optional
        Name of dataset for error messages (descriptive parameter name)

    Returns
    -------
    tuple
        (is_valid, error_message)
    """
    if portfolio_dataframe is None:
        return False, f"No {dataset_name} available"

    if not isinstance(portfolio_dataframe, pd.DataFrame):
        return False, f"Invalid {dataset_name} type: {type(portfolio_dataframe).__name__}"

    if portfolio_dataframe.empty:
        return False, f"{dataset_name.capitalize()} is empty"

    if "Date" not in portfolio_dataframe.columns:
        return False, f"{dataset_name.capitalize()} missing required 'Date' column"

    # For portfolio and benchmark, check for Value column
    if dataset_name.lower() in ["portfolio data", "benchmark data"]:
        if "Value" not in portfolio_dataframe.columns:
            return False, f"{dataset_name.capitalize()} missing required 'Value' column"

        # Check if Value column contains numeric data
        try:
            non_numeric_count = (
                pd.to_numeric(portfolio_dataframe["Value"], errors="coerce").isna().sum()
            )
            if non_numeric_count == len(portfolio_dataframe):
                return False, f"No valid numeric values in {dataset_name} 'Value' column"
        except Exception as validation_error:
            return False, f"Error validating {dataset_name} 'Value' column: {validation_error!s}"

    # For weights data, check for component columns
    if dataset_name.lower() == "weights data":
        component_columns = [
            column_name for column_name in portfolio_dataframe.columns if column_name != "Date"
        ]
        if not component_columns:
            return False, f"No component columns found in {dataset_name}"

    return True, ""
