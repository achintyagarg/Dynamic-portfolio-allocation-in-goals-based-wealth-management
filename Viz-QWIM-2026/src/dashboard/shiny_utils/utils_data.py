"""Data utilities for QWIM Dashboard.

This module provides data loading and utility functions for the QWIM Dashboard.
It handles loading of raw and processed data files with proper validation.

Key Functions
-------------
get_data_utils
    Load utility data (constants, defaults).
get_data_inputs
    Load all input data files.
get_input_data_raw
    Load raw data from inputs/raw folder.
get_input_data_processed
    Load processed data from inputs/processed folder.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import polars as pl

from src.utils.custom_exceptions_errors_loggers.exception_custom import (
    Exception_Data_Not_Found,
    Exception_Severity,
)
from src.utils.custom_exceptions_errors_loggers.logger_custom import get_logger


if TYPE_CHECKING:
    from pathlib import Path


#: Module-level logger instance
_logger = get_logger(__name__)


def find_date_column(df: pl.DataFrame) -> str | None:
    """Find the date column in a DataFrame (case-insensitive).

    Looks for common date column names: 'Date', 'date', 'DATE'.

    Args:
        df: Polars DataFrame to search

    Returns
    -------
        str: Name of the date column if found, None otherwise
    """
    possible_names = ["Date", "date", "DATE"]
    for col_name in possible_names:
        if col_name in df.columns:
            return col_name
    return None


def standardize_date_column(df: pl.DataFrame) -> pl.DataFrame:
    """Standardize the date column name to 'Date'.

    Finds any date column (case-insensitive) and renames it to 'Date'.

    Args:
        df: Polars DataFrame with a date column

    Returns
    -------
        pl.DataFrame: DataFrame with standardized 'Date' column name
    """
    date_col = find_date_column(df)
    if date_col is not None and date_col != "Date":
        df = df.rename({date_col: "Date"})
    return df


def get_data_utils(project_dir: Any) -> Any:
    """Get saved data (such as default parameters) to be used in dashboard."""
    constants_dict = get_data_utils_constants(project_dir=project_dir)
    defaults_dict = get_data_utils_defaults(project_dir=project_dir)

    return {**constants_dict, **defaults_dict}


def get_data_inputs(project_dir: Any) -> Any:
    """Load data (with error handling) from files in subfolders of folder "inputs"."""
    input_data_raw = get_input_data_raw(project_dir=project_dir)
    input_data_processed = get_input_data_processed(project_dir=project_dir)

    return {**input_data_raw, **input_data_processed}


def get_input_data_raw(project_dir: Path) -> dict[str, pl.DataFrame]:
    """Load data from files in "raw" subfolder of folder "inputs".

    The datafiles are stored in polars dataframes, since polars have many advantages
    over pandas (computational speed, memory efficiency, immutability).

    Parameters
    ----------
    project_dir
        Path to the project root directory.

    Returns
    -------
    dict[str, pl.DataFrame]
        Dictionary containing loaded data with standardized keys.

    Raises
    ------
    Exception_Data_Not_Found
        If required data files are not found or cannot be loaded.
    """
    # Define file paths
    path_file_timeseries = project_dir / "inputs" / "raw" / "data_timeseries.csv"
    path_file_etfs = project_dir / "inputs" / "raw" / "data_ETFs.csv"
    path_file_weights_etfs = project_dir / "inputs" / "raw" / "sample_portfolio_weights_ETFs.csv"

    input_data: dict[str, pl.DataFrame] = {}

    # Read data_timeseries.csv with explicit error handling
    try:
        if not path_file_timeseries.exists():
            raise Exception_Data_Not_Found(
                f"Time series data file not found: {path_file_timeseries}",
                severity=Exception_Severity.ERROR,
            )

        data_time_series_sample = pl.read_csv(str(path_file_timeseries))

        # Validate that data was read successfully
        if data_time_series_sample.is_empty():
            _logger.warning(
                "Time series data file is empty",
                extra={"file_path": str(path_file_timeseries)},
            )

        # Check for required date column (case-insensitive)
        if find_date_column(data_time_series_sample) is None:
            _logger.warning(
                "Time series data file missing required date column",
                extra={"file_path": str(path_file_timeseries)},
            )

        # Standardize column name to 'Date'
        data_time_series_sample = standardize_date_column(data_time_series_sample)

        input_data["Time_Series_Sample"] = data_time_series_sample
        _logger.debug("Loaded time series data", extra={"num_rows": len(data_time_series_sample)})

    except pl.exceptions.ComputeError as compute_exc:
        raise Exception_Data_Not_Found(
            f"Failed to read time series data file due to compute error: {path_file_timeseries}",
            severity=Exception_Severity.ERROR,
            cause=compute_exc,
            context={"file_path": str(path_file_timeseries), "error_type": "ComputeError"},
        ) from compute_exc
    except pl.exceptions.SchemaError as schema_exc:
        raise Exception_Data_Not_Found(
            f"Failed to read time series data file due to schema error: {path_file_timeseries}",
            severity=Exception_Severity.ERROR,
            cause=schema_exc,
            context={"file_path": str(path_file_timeseries), "error_type": "SchemaError"},
        ) from schema_exc
    except Exception_Data_Not_Found:
        raise
    except Exception as read_exc:
        raise Exception_Data_Not_Found(
            f"Failed to read time series data file: {path_file_timeseries}",
            severity=Exception_Severity.ERROR,
            cause=read_exc,
            context={"file_path": str(path_file_timeseries)},
        ) from read_exc

    # Read data_ETFs.csv with explicit error handling
    try:
        if not path_file_etfs.exists():
            raise Exception_Data_Not_Found(
                f"ETFs data file not found: {path_file_etfs}",
                severity=Exception_Severity.ERROR,
            )

        data_time_series_etfs = pl.read_csv(str(path_file_etfs))

        # Validate that data was read successfully
        if data_time_series_etfs.is_empty():
            _logger.warning("ETFs data file is empty", extra={"file_path": str(path_file_etfs)})

        # Standardize date column name
        data_time_series_etfs = standardize_date_column(data_time_series_etfs)

        input_data["Time_Series_ETFs"] = data_time_series_etfs
        _logger.debug("Loaded ETFs data", extra={"num_rows": len(data_time_series_etfs)})

    except pl.exceptions.ComputeError as compute_exc:
        raise Exception_Data_Not_Found(
            f"Failed to read ETFs data file due to compute error: {path_file_etfs}",
            severity=Exception_Severity.ERROR,
            cause=compute_exc,
            context={"file_path": str(path_file_etfs), "error_type": "ComputeError"},
        ) from compute_exc
    except pl.exceptions.SchemaError as schema_exc:
        raise Exception_Data_Not_Found(
            f"Failed to read ETFs data file due to schema error: {path_file_etfs}",
            severity=Exception_Severity.ERROR,
            cause=schema_exc,
            context={"file_path": str(path_file_etfs), "error_type": "SchemaError"},
        ) from schema_exc
    except Exception_Data_Not_Found:
        raise
    except Exception as read_exc:
        raise Exception_Data_Not_Found(
            f"Failed to read ETFs data file: {path_file_etfs}",
            severity=Exception_Severity.ERROR,
            cause=read_exc,
            context={"file_path": str(path_file_etfs)},
        ) from read_exc

    # Read sample_portfolio_weights_ETFs.csv with explicit error handling
    try:
        if not path_file_weights_etfs.exists():
            raise Exception_Data_Not_Found(
                f"ETFs weights data file not found: {path_file_weights_etfs}",
                severity=Exception_Severity.ERROR,
            )

        data_weights_etfs = pl.read_csv(str(path_file_weights_etfs))

        # Validate that data was read successfully
        if data_weights_etfs.is_empty():
            _logger.warning(
                "ETFs weights data file is empty",
                extra={"file_path": str(path_file_weights_etfs)},
            )

        # Standardize date column name
        data_weights_etfs = standardize_date_column(data_weights_etfs)

        input_data["Weights_My_Portfolio"] = data_weights_etfs
        _logger.debug("Loaded weights data", extra={"num_rows": len(data_weights_etfs)})

    except pl.exceptions.ComputeError as compute_exc:
        raise Exception_Data_Not_Found(
            f"Failed to read ETFs weights data file due to compute error: {path_file_weights_etfs}",
            severity=Exception_Severity.ERROR,
            cause=compute_exc,
            context={"file_path": str(path_file_weights_etfs), "error_type": "ComputeError"},
        ) from compute_exc
    except pl.exceptions.SchemaError as schema_exc:
        raise Exception_Data_Not_Found(
            f"Failed to read ETFs weights data file due to schema error: {path_file_weights_etfs}",
            severity=Exception_Severity.ERROR,
            cause=schema_exc,
            context={"file_path": str(path_file_weights_etfs), "error_type": "SchemaError"},
        ) from schema_exc
    except Exception_Data_Not_Found:
        raise
    except Exception as read_exc:
        raise Exception_Data_Not_Found(
            f"Failed to read ETFs weights data file: {path_file_weights_etfs}",
            severity=Exception_Severity.ERROR,
            cause=read_exc,
            context={"file_path": str(path_file_weights_etfs)},
        ) from read_exc

    _logger.info(
        "Raw data loaded successfully",
        extra={"datasets": list(input_data.keys())},
    )
    return input_data


def get_input_data_processed(project_dir: Any) -> Any:
    """
    Load data (with error handling) from files in "processed" subfolder of folder "inputs".

    The datafiles are stored in polars dataframes, since polars have many advantages
    over pandas (computational speed, memory efficiency, immutability).

    """
    # Define file paths
    path_file_portfolio_benchmark = (
        project_dir / "inputs" / "processed" / "benchmark_portfolio_values.csv"
    )
    path_file_my_portfolio = project_dir / "inputs" / "processed" / "sample_portfolio_values.csv"

    input_data = {}

    # Read benchmark_portfolio_values.csv with explicit error handling
    try:
        if not path_file_portfolio_benchmark.exists():
            raise FileNotFoundError(
                f"Benchmark portfolio data file not found: {path_file_portfolio_benchmark}",
            )

        data_portfolio_benchmark = pl.read_csv(str(path_file_portfolio_benchmark))

        # Validate that data was read successfully
        if data_portfolio_benchmark.is_empty():
            raise ValueError(
                f"Benchmark portfolio data file is empty: {path_file_portfolio_benchmark}",
            )

        # Check for required date column (case-insensitive)
        if find_date_column(data_portfolio_benchmark) is None:
            raise ValueError(
                f"Benchmark portfolio data file missing required date column: {path_file_portfolio_benchmark}",
            )

        # Standardize date column name
        data_portfolio_benchmark = standardize_date_column(data_portfolio_benchmark)

        input_data["Benchmark_Portfolio"] = data_portfolio_benchmark

    except pl.exceptions.ComputeError as compute_exc:
        raise RuntimeError(
            f"Failed to read benchmark portfolio data file due to compute error: {path_file_portfolio_benchmark}. Error: {compute_exc}",
        ) from compute_exc
    except pl.exceptions.SchemaError as schema_exc:
        raise RuntimeError(
            f"Failed to read benchmark portfolio data file due to schema error: {path_file_portfolio_benchmark}. Error: {schema_exc}",
        ) from schema_exc
    except Exception as read_exc:
        raise RuntimeError(
            f"Failed to read benchmark portfolio data file: {path_file_portfolio_benchmark}. Error: {read_exc}",
        ) from read_exc

    # Read sample_portfolio_values.csv with explicit error handling
    try:
        if not path_file_my_portfolio.exists():
            raise FileNotFoundError(f"My portfolio data file not found: {path_file_my_portfolio}")

        data_my_portfolio = pl.read_csv(str(path_file_my_portfolio))

        # Validate that data was read successfully
        if data_my_portfolio.is_empty():
            raise ValueError(f"Sample portfolio data file is empty: {path_file_my_portfolio}")

        # Check for required date column (case-insensitive)
        if find_date_column(data_my_portfolio) is None:
            raise ValueError(
                f"My portfolio data file missing required date column: {path_file_my_portfolio}",
            )

        # Standardize date column name
        data_my_portfolio = standardize_date_column(data_my_portfolio)

        input_data["My_Portfolio"] = data_my_portfolio

    except pl.exceptions.ComputeError as compute_exc:
        raise RuntimeError(
            f"Failed to read sample portfolio data file due to compute error: {path_file_my_portfolio}. Error: {compute_exc}",
        ) from compute_exc
    except pl.exceptions.SchemaError as schema_exc:
        raise RuntimeError(
            f"Failed to read sample portfolio data file due to schema error: {path_file_my_portfolio}. Error: {schema_exc}",
        ) from schema_exc
    except Exception as read_exc:
        raise RuntimeError(
            f"Failed to read sample portfolio data file: {path_file_my_portfolio}. Error: {read_exc}",
        ) from read_exc

    return input_data


def get_names_time_series_from_DF(polars_DF: Any) -> Any:
    """Extract series names from a polars DataFrame polars_DF."""
    try:
        if polars_DF is None or polars_DF.is_empty():
            return []

        # Define possible date column names (case-insensitive)
        possible_date_columns = ["date", "Date", "DATE"]

        # Find the actual date column name in the DataFrame
        actual_date_column = None
        for date_col_name in possible_date_columns:
            if date_col_name in polars_DF.columns:
                actual_date_column = date_col_name
                break

        # Get all columns except the date column
        if actual_date_column is not None:
            series_names = [jcol for jcol in polars_DF.columns if jcol != actual_date_column]
        else:
            # If no date column found, return all columns
            series_names = list(polars_DF.columns)

        return series_names

    except Exception as exc:
        raise RuntimeError(f"Error getting series names from DataFrame: {exc!s}") from exc


def downsample_dataframe(polars_DF: Any, max_points: Any = 200, date_column: Any = "Date") -> Any:
    """
    Downsample a polars DataFrame to reduce the number of data points for better visualization performance.

    Args:
        polars_DF: Input polars DataFrame to downsample
        max_points: Maximum number of points to keep (default: 200)
        date_column: Name of the date column (default: "Date", case-insensitive)

    Returns
    -------
        Downsampled polars DataFrame

    Raises
    ------
        ValueError: If polars_DF is None or invalid parameters
        TypeError: If polars_DF is not a Polars DataFrame
        RuntimeError: If downsampling operation fails
    """
    # Input validation with early returns
    if polars_DF is None:
        raise ValueError(
            "downsample_dataframe received None dataframe. "
            "Ensure data processing steps return valid DataFrames before downsampling.",
        )

    # Configuration validation - check if it's a Polars DataFrame
    if not isinstance(polars_DF, pl.DataFrame):
        raise TypeError(
            f"polars_DF must be a Polars DataFrame, got {type(polars_DF).__name__}. "
            f"Convert data to Polars DataFrame before calling downsample_dataframe.",
        )

    try:
        # Business logic validation - check if DataFrame is empty
        if polars_DF.is_empty():
            raise ValueError(
                "downsample_dataframe received empty dataframe. "
                "This indicates that data filtering removed all rows. "
                "Common causes:\n"
                "1. Date range filtering excluded all available data\n"
                "2. Component selection filtering removed all rows\n"
                "3. Source data file is empty\n"
                "Solution: Verify date ranges match available data and check data file contents.",
            )

        # Early return - no downsampling needed if below threshold
        num_rows_current = polars_DF.height
        if num_rows_current <= max_points:
            return polars_DF

        # Calculate step size for downsampling
        step_size = max(1, num_rows_current // max_points)

        # Perform downsampling with proper indexing
        return polars_DF[::step_size]

    except ValueError as validation_exc:
        # Re-raise validation errors with context preserved
        raise validation_exc
    except Exception as exc:
        # Handle unexpected errors with comprehensive context
        raise RuntimeError(f"Error in downsample_dataframe: {exc!s}") from exc


def get_safe_num_rows_for_DF(input_DF: Any) -> Any:
    """Safely get the number of rows in a DataFrame input_DF."""
    if input_DF is None:
        return 0

    try:
        if hasattr(input_DF, "height"):  # Polars
            return input_DF.height
        if hasattr(input_DF, "shape"):  # Pandas
            return input_DF.shape[0]
        return 0
    except Exception:
        return 0


def get_data_utils_constants(project_dir: Any) -> Any:
    """Get constants to use in dashboard."""
    return {}


def get_data_utils_defaults(project_dir: Any) -> Any:
    """Get defaults to use in dashboard."""
    return {
        "Select_Time_Period": "Custom",
        "Custom_Date_Range_Start": "2018-01-10",
        "Custom_Date_Range_End": "2023-03-12",
    }


def validate_portfolio_data(data_portfolio: Any) -> Any:
    """Validate portfolio data.

    It checks if the DataFrame is a Polars dataframe which is
    not empty, with 2 columns: "Date" and "Value". The "Date" column should be either a string
    or in date format, and the dates should be increasing.

    Args:
        data_portfolio: The portfolio data to validate (expected to be a Polars DataFrame)

    Returns
    -------
        tuple: (is_valid: bool, error_message: str)
               is_valid is True if validation passes, False otherwise
               error_message contains the validation error description if validation fails

    Raises
    ------
        ValueError: If data_portfolio is None
    """
    # Input validation with early returns
    if data_portfolio is None:
        return False, "Portfolio data is None"

    # Configuration validation - check if it's a Polars DataFrame
    if not isinstance(data_portfolio, pl.DataFrame):
        return (
            False,
            f"Portfolio data must be a Polars DataFrame, got {type(data_portfolio).__name__}",
        )

    # Business logic validation - check if DataFrame is empty
    if data_portfolio.is_empty():
        return False, "Portfolio data DataFrame is empty"

    # Configuration validation - check number of columns
    column_count = len(data_portfolio.columns)
    if column_count != 2:
        return (
            False,
            f"Portfolio data must have exactly 2 columns, got {column_count} columns: {list(data_portfolio.columns)}",
        )

    # Configuration validation - check required columns exist
    required_columns = ["Date", "Value"]
    actual_columns = data_portfolio.columns

    for required_column in required_columns:
        if required_column not in actual_columns:
            return (
                False,
                f"Portfolio data missing required column '{required_column}'. Available columns: {actual_columns}",
            )

    # Business logic validation - check Date column data type
    try:
        date_column = data_portfolio.select("Date").to_series()
        date_dtype = date_column.dtype

        # Check if Date column is string or date type
        valid_date_types = [pl.Utf8, pl.String, pl.Date, pl.Datetime]
        is_valid_date_type = any(date_dtype == valid_type for valid_type in valid_date_types)

        if not is_valid_date_type:
            return False, f"Date column must be string or date type, got {date_dtype}"

    except Exception as date_check_exc:
        return False, f"Error checking Date column type: {date_check_exc}"

    # Business logic validation - check Value column data type
    try:
        value_column = data_portfolio.select("Value").to_series()
        value_dtype = value_column.dtype

        # Check if Value column is numeric
        valid_value_types = [pl.Float32, pl.Float64, pl.Int32, pl.Int64, pl.UInt32, pl.UInt64]
        is_valid_value_type = any(value_dtype == valid_type for valid_type in valid_value_types)

        if not is_valid_value_type:
            return False, f"Value column must be numeric type, got {value_dtype}"

    except Exception as value_check_exc:
        return False, f"Error checking Value column type: {value_check_exc}"

    # Business logic validation - check if dates are increasing
    try:
        # Convert Date column to datetime for comparison if it's string
        if date_dtype in [pl.Utf8, pl.String]:
            try:
                # Try multiple date formats commonly used
                date_formats = [
                    "%Y-%m-%d",  # 2023-12-31
                    "%d/%m/%Y",  # 31/12/2023
                    "%m/%d/%Y",  # 12/31/2023
                    "%Y/%m/%d",  # 2023/12/31
                    "%d-%m-%Y",  # 31-12-2023
                    "%m-%d-%Y",  # 12-31-2023
                    "%Y.%m.%d",  # 2023.12.31
                    "%d.%m.%Y",  # 31.12.2023
                    "%Y-%m-%d %H:%M:%S",  # 2023-12-31 00:00:00
                    "%Y-%m-%d %H:%M:%S%z",  # 2023-12-31 00:00:00-05:00
                    "%Y-%m-%d %H:%M:%S%:z",  # 2023-12-31 00:00:00-05:00 (alternative format)
                ]

                date_series_converted = None
                successful_format = None

                # Try each format until one works
                for date_format in date_formats:
                    try:
                        date_series_converted = date_column.str.to_date(
                            format=date_format,
                            strict=False,
                        )
                        # Check if conversion was successful (no nulls where there shouldn't be)
                        null_count_after = date_series_converted.null_count()
                        original_null_count = date_column.null_count()

                        # If we didn't create new nulls, the format worked
                        if null_count_after == original_null_count:
                            successful_format = date_format
                            break
                    except Exception as _fmt_exc:
                        _logger.debug("Date format '%s' failed: %s", date_format, _fmt_exc)
                        continue

                # If specific formats failed, try datetime conversion first then extract date
                if date_series_converted is None or successful_format is None:
                    try:
                        # Try to parse as datetime first, then convert to date
                        datetime_series = date_column.str.to_datetime(strict=False)
                        date_series_converted = datetime_series.dt.date()

                        # Check if conversion was successful
                        null_count_after = date_series_converted.null_count()
                        original_null_count = date_column.null_count()

                        # If we didn't create excessive new nulls, the conversion worked
                        if null_count_after <= original_null_count + (
                            date_column.len() * 0.1
                        ):  # Allow 10% failure rate
                            successful_format = "datetime_auto_conversion"
                        else:
                            date_series_converted = None

                    except Exception:
                        date_series_converted = None

                # If no format worked, try automatic parsing with strict=False as last resort
                if date_series_converted is None or successful_format is None:
                    try:
                        date_series_converted = date_column.str.to_date(strict=False)
                        # Check for excessive nulls indicating failed parsing
                        null_count_after = date_series_converted.null_count()
                        original_null_count = date_column.null_count()
                        total_rows = date_column.len()

                        # If more than half the dates failed to parse, consider it a failure
                        if (null_count_after - original_null_count) > (total_rows * 0.5):
                            return (
                                False,
                                f"Could not parse more than 50% of Date column strings. Please check date format. Sample values: {date_column.head(3).to_list()}",
                            )
                    except Exception as auto_parse_exc:
                        return (
                            False,
                            f"Error parsing Date column strings. Please check date format. Sample values: {date_column.head(3).to_list()}. Error: {auto_parse_exc}",
                        )

            except Exception as conversion_exc:
                return (
                    False,
                    f"Error converting Date column strings to dates. Please check date format. Sample values: {date_column.head(3).to_list()}. Error: {conversion_exc}",
                )
        else:
            date_series_converted = date_column

        # Check for null dates
        null_dates_count = date_series_converted.null_count()
        original_null_count = date_column.null_count()

        # Only report error if we created new nulls during conversion
        if null_dates_count > original_null_count:
            return (
                False,
                f"Date column contains {null_dates_count - original_null_count} unparseable date values after conversion",
            )

        # Check if dates are increasing (sorted) - only on non-null values
        date_series_clean = date_series_converted.drop_nulls()
        if date_series_clean.len() > 1:
            is_sorted = date_series_clean.is_sorted()
            if not is_sorted:
                return False, "Date column values are not in increasing order"

    except Exception as date_order_exc:
        return False, f"Error checking Date column order: {date_order_exc}"

    # Business logic validation - check for null values in Value column
    try:
        null_values_count = value_column.null_count()
        if null_values_count > 0:
            return False, f"Value column contains {null_values_count} null values"

    except Exception as value_null_exc:
        return False, f"Error checking Value column for nulls: {value_null_exc}"

    # Business logic validation - check for duplicate dates
    try:
        # Only check for duplicates on successfully parsed dates
        if date_dtype in [pl.Utf8, pl.String]:
            # Use the converted dates for duplicate checking
            dates_for_duplicate_check = date_series_converted.drop_nulls()
        else:
            dates_for_duplicate_check = date_column.drop_nulls()

        unique_dates_count = dates_for_duplicate_check.n_unique()
        total_valid_dates = dates_for_duplicate_check.len()

        if unique_dates_count != total_valid_dates:
            duplicate_count = total_valid_dates - unique_dates_count
            return False, f"Date column contains {duplicate_count} duplicate date values"

    except Exception as duplicate_check_exc:
        return False, f"Error checking for duplicate dates: {duplicate_check_exc}"

    # All validations passed
    return True, "Portfolio data validation successful"


def calculate_portfolio_returns(data_portfolio: Any) -> Any:
    """
    Calculate returns from a Polars DataFrame with Date and Value columns.

    Args:
        data_portfolio: Polars DataFrame with "Date" and "Value" columns

    Returns
    -------
        polars.Series: Series containing calculated returns (percentage change)

    Raises
    ------
        ValueError: If data_portfolio is invalid or missing required columns
        RuntimeError: If calculation fails
    """
    # Input validation with early returns
    if data_portfolio is None:
        raise ValueError("Portfolio data is None")

    # Configuration validation - check if it's a Polars DataFrame
    if not isinstance(data_portfolio, pl.DataFrame):
        raise ValueError(
            f"Portfolio data must be a Polars DataFrame, got {type(data_portfolio).__name__}",
        )

    # Business logic validation - check if DataFrame is empty
    if data_portfolio.is_empty():
        return pl.Series("returns", [], dtype=pl.Float64)

    # Configuration validation - check required columns exist
    if "Value" not in data_portfolio.columns:
        raise ValueError(
            f"Portfolio data missing required 'Value' column. Available columns: {data_portfolio.columns}",
        )

    if "Date" not in data_portfolio.columns:
        raise ValueError(
            f"Portfolio data missing required 'Date' column. Available columns: {data_portfolio.columns}",
        )

    # Only use try-except for operations that might fail unpredictably
    try:
        # Business logic validation - sort by Date and prepare data
        portfolio_sorted = data_portfolio.sort("Date")

        # Safe numeric conversion of Value column
        portfolio_with_numeric = portfolio_sorted.with_columns(
            [pl.col("Value").cast(pl.Float64, strict=False).alias("Value_Numeric")],
        )

        # Remove rows with null values after conversion
        portfolio_clean = portfolio_with_numeric.filter(pl.col("Value_Numeric").is_not_null())

        # Check if we have enough data points for returns calculation
        if portfolio_clean.height < 2:
            return pl.Series("returns", [], dtype=pl.Float64)

        # Calculate percentage change (returns) using Polars operations
        portfolio_with_returns = portfolio_clean.with_columns(
            [pl.col("Value_Numeric").pct_change().alias("returns")],
        )

        # Extract returns series and remove the first null value (no previous value for first row)
        returns_series = portfolio_with_returns.select("returns").to_series()
        return returns_series.drop_nulls()

    except Exception as calc_exc:
        raise RuntimeError(f"Error calculating portfolio returns: {calc_exc}") from calc_exc


def validate_portfolio_and_benchmark_data_if_not_already_validated(
    data_portfolio: pl.DataFrame | None,
    data_benchmark: pl.DataFrame | None,
) -> tuple[bool, bool]:
    """Validate input data if not already validated."""
    if data_portfolio is not None:
        validation_portfolio = validate_portfolio_data(data_portfolio=data_portfolio)
        # Check portfolio validation and throw error if validation failed
        if not validation_portfolio[0]:
            raise ValueError(f"Portfolio data validation failed: {validation_portfolio[1]}")

    if data_benchmark is not None:
        validation_benchmark = validate_portfolio_data(data_portfolio=data_benchmark)
        # Check benchmark validation and throw error if validation failed
        if not validation_benchmark[0]:
            raise ValueError(f"Benchmark data validation failed: {validation_benchmark[1]}")

    # Configuration validation - check for empty DataFrames
    has_portfolio_data = (
        data_portfolio is not None
        and isinstance(data_portfolio, pl.DataFrame)
        and not data_portfolio.is_empty()
    )

    has_benchmark_data = (
        data_benchmark is not None
        and isinstance(data_benchmark, pl.DataFrame)
        and not data_benchmark.is_empty()
    )

    return has_portfolio_data, has_benchmark_data
