"""Portfolio Analysis Module.

This module provides the UI and server logic for the Portfolio Analysis subtab
within the QWIM Dashboard's Portfolios section.

## Overview

The Portfolio Analysis module enables comprehensive analysis of portfolio performance
with interactive visualizations and statistical calculations. It supports multiple
analysis types including returns distribution, drawdowns analysis, rolling statistics,
and portfolio vs benchmark comparisons.

## Features

- **Interactive Plotly Visualizations**: Native Shiny integration with plotly widgets
- **Time Period Selection**: Dropdown selection plus custom date range input
- **Multiple Analysis Types**: Returns, drawdowns, rolling statistics, and comparisons
- **Real-time Statistics**: Live performance metrics and basic statistics tables
- **Data Export**: PNG export capabilities for plots and tables
- **Performance Optimization**: Data downsampling for improved rendering speed
- **Responsive Design**: Mobile-friendly layout with proper card structures

## Analysis Types

### Returns Distribution
Analyzes the distribution of portfolio returns with histogram and statistical overlays.

### Drawdowns Analysis
Visualizes portfolio drawdowns over time with maximum drawdown highlighting.

### Rolling Statistics
Shows rolling volatility, returns, and other metrics over configurable time windows.

### Portfolio vs Benchmark
Compares portfolio performance directly against benchmark data.

## Configuration

Analysis type is controlled via the sidebar input controls.

## Dependencies

- `plotly`: Interactive chart generation
- `shiny`: Reactive UI framework
- `shinywidgets`: Plotly widget rendering
- `pandas`: Data manipulation and analysis
- `numpy`: Numerical computations
- `scipy`: Statistical functions
- `polars`: High-performance data processing
- `kaleido`: PNG export functionality

## Usage Example

```python
# UI Component
ui_component = subtab_portfolios_analysis_ui(
    data_utils=data_utilities, data_inputs=portfolio_data_dict
)

# Server Component
server_function = subtab_portfolios_analysis_server(
    input=input_object,
    output=output_object,
    session=session_object,
    data_utils=data_utilities,
    data_inputs=portfolio_data_dict,
    reactives_shiny=reactive_values,
)
```

## File Structure

```
output/
├── temp-graphics/          # PNG exports directory
└── ...
```

## Error Handling

The module implements defensive programming principles with:
- Input validation with early returns
- Configuration validation for user inputs
- Business logic validation for data processing
- Graceful fallbacks for missing or invalid data

## Performance Considerations

- Data downsampling to 200 points maximum for optimal rendering
- Lazy evaluation of reactive calculations
- Efficient Polars DataFrame operations
- Optional PNG saving to reduce processing overhead

---

**Author**: QWIM Dashboard Team
**Version**: 0.5.1
**Last Updated**: 2026-03-01
"""

from __future__ import annotations

import typing

from datetime import UTC, datetime, timedelta
from pathlib import Path

# Core Python libraries for date/time operations and data processing
from typing import Any

# Data processing and analysis libraries
import pandas as pd  # Traditional DataFrame operations and compatibility layer
import polars as pl  # High-performance DataFrame operations


# Shiny reactive web framework components
from shiny import module, reactive, render, ui  # Core Shiny framework
from shinywidgets import output_widget, render_widget  # Plotly widget integration

# Custom logging utilities
from src.utils.custom_exceptions_errors_loggers.logger_custom import get_logger


#: Module-level logger instance
_logger = get_logger(__name__)

# Import visualization utilities for plot and table generation
# Import data processing utilities for portfolio analysis workflows
from src.dashboard.shiny_utils.reactives_shiny import (  # noqa: E402
    update_visual_object_in_reactives,  # Store latest chart in shared reactive state
)
from src.dashboard.shiny_utils.utils_data import (  # noqa: E402
    downsample_dataframe,  # Performance optimization through data reduction
    validate_portfolio_data,  # Portfolio data structure and content validation
)
from src.dashboard.shiny_utils.utils_visuals import (  # noqa: E402
    calculate_table_metrics_performance,  # Performance metrics calculation
    calculate_table_stats_basic,
    create_error_figure,  # Error plot generation for user feedback
    create_plot_drawdowns_analysis,  # Maximum drawdown visualization
    create_plot_portfolios_comparison,  # Portfolio vs benchmark comparison plots
    create_plot_returns_distribution,  # Returns analysis and distribution plots
    create_plot_rolling_statistics,  # Rolling window statistical plots
    format_value_for_display,  # Value formatting utility for financial data
)  # Basic statistical measures calculation


# Constants for file paths and configuration management
# =================================================

# Output directory for generated files and exports
# Used for storing temporary graphics and analysis results
OUTPUT_DIR = Path("output")


@module.ui
def subtab_portfolios_analysis_ui(data_utils: dict, data_inputs: dict) -> Any:
    """
    Create the UI for the Portfolio Analysis subtab.

    Generates a responsive layout with sidebar controls and main content area
    for portfolio analysis visualization and statistics display.

    Args:
        data_utils (dict): Dictionary containing utility functions and configurations
            Expected to contain helper functions and configuration parameters
        data_inputs (dict): Dictionary containing portfolio and benchmark data
            Expected keys:
            - "My_Portfolio": Polars DataFrame with Date and Value columns
            - "Benchmark_Portfolio": Polars DataFrame with Date and Value columns
            - "Weights_My_Portfolio": Polars DataFrame with portfolio weights (optional)
            - "Time_Series_ETFs": Polars DataFrame with ETF time series data (optional)

    Returns
    -------
        shiny.ui.div: Complete UI layout for the portfolio analysis subtab

    Layout Structure:
        - Header section with analysis title
        - Sidebar containing control elements (time period, analysis type, options)
        - Main content area with plot visualization and statistics tables

    UI Components:
        - Time period selection (preset dropdown options + custom date range picker)
        - Analysis type selection (returns, drawdowns, rolling statistics, comparison)
        - Benchmark inclusion toggle checkbox
        - Data information display panel
        - Loading status indicator
        - Main analysis plot container (600px height for optimal display)
        - Performance statistics tables in side-by-side card layout

    Responsive Design Features:
        - Minimum widths enforced for proper table display on all screen sizes
        - Flexible column layout adapting to different viewport dimensions
        - Card-based design for visual separation and professional appearance
        - Scrollable table containers to handle large datasets gracefully
        - Bootstrap grid system for consistent responsive behavior
    """
    return ui.div(
        ui.h3("Portfolio Performance Analysis"),
        ui.layout_sidebar(
            ui.sidebar(
                # Time Period Selection Section
                ui.h5("📅 Time Period Selection", class_="mb-3"),
                # time period dropdown
                ui.input_select(
                    "input_ID_tab_portfolios_subtab_portfolios_analysis_time_period",
                    "Select Time Period",
                    {
                        "custom": "Custom",
                        "1y": "Last 1 Year",
                        "3y": "Last 3 Years",
                        "5y": "Last 5 Years",
                        "10y": "Last 10 Years",
                        "ytd": "Year to Date",
                    },
                    selected="1y",
                ),
                # Date range input - conditionally shown for custom selection
                ui.panel_conditional(
                    "input.input_ID_tab_portfolios_subtab_portfolios_analysis_time_period === 'custom'",
                    ui.div(
                        ui.input_date_range(
                            "input_ID_tab_portfolios_subtab_portfolios_analysis_date_range",
                            "Custom Date Range",
                            start=datetime.now(UTC) - timedelta(days=365),
                            end=datetime.now(UTC),
                            format="yyyy-mm-dd",
                            separator=" to ",
                            width="100%",
                        ),
                        class_="mt-2",
                    ),
                ),
                # Display calculated date range for non-custom selections
                ui.panel_conditional(
                    "input.input_ID_tab_portfolios_subtab_portfolios_analysis_time_period !== 'custom'",
                    ui.div(
                        ui.output_text(
                            "output_ID_tab_portfolios_subtab_portfolios_analysis_calculated_date_range",
                        ),
                        class_="mt-2 p-2 bg-light rounded text-muted small",
                    ),
                ),
                ui.hr(),
                # Analysis Options Section
                ui.h5("📊 Analysis Options", class_="mb-3"),
                ui.input_select(
                    "input_ID_tab_portfolios_subtab_portfolios_analysis_type",
                    "Analysis Type",
                    {
                        "returns": "Returns Distribution",
                        "drawdowns": "Drawdowns Analysis",
                        "rolling": "Rolling Statistics",
                        "comparison": "Portfolio vs Benchmark",
                    },
                    selected="returns",
                ),
                ui.panel_conditional(
                    "input.input_ID_tab_portfolios_subtab_portfolios_analysis_type === 'rolling'",
                    ui.div(
                        ui.input_slider(
                            "input_ID_tab_portfolios_subtab_portfolios_analysis_rolling_window",
                            "Rolling Window (days)",
                            min=7,
                            max=90,
                            value=30,
                            step=1,
                            width="100%",
                        ),
                        class_="mt-3",
                    ),
                ),
                ui.input_checkbox(
                    "input_ID_tab_portfolios_subtab_portfolios_analysis_include_benchmark",
                    "Include Benchmark in Analysis",
                    value=True,
                ),
                # Data Info Section
                ui.hr(),
                ui.h5("ℹ️ Data Information", class_="mb-3"),
                ui.output_ui("output_ID_tab_portfolios_subtab_portfolios_analysis_data_info"),
                width=350,
                position="left",
            ),
            # Main content area
            ui.div(
                # Add loading status at the top of main content
                ui.output_ui("output_ID_tab_portfolios_subtab_portfolios_analysis_loading_status"),
                # Main analysis plot with proper height and styling
                ui.div(
                    ui.h4("Analysis Visualization", class_="mb-3"),
                    ui.div(
                        output_widget(
                            "output_ID_tab_portfolios_subtab_portfolios_analysis_plot_main",
                            width="100%",
                            height="600px",
                        ),
                        style="min-height: 600px; width: 100%; border: 1px solid #dee2e6; border-radius: 0.375rem; padding: 15px; background-color: white;",
                    ),
                    class_="mb-4",
                ),
                # Statistics Tables Section
                ui.div(
                    ui.h4("Performance Statistics", class_="mb-3"),
                    ui.div(
                        ui.div(
                            ui.div(
                                ui.h5("Basic Statistics", class_="card-title text-primary mb-3"),
                                ui.div(
                                    ui.output_table(
                                        "output_ID_tab_portfolios_subtab_portfolios_analysis_table_stats",
                                    ),
                                    style="max-height: 500px; overflow-y: auto; overflow-x: auto; width: 100%;",
                                ),
                                class_="card card-body h-100",
                                style="min-width: 450px;",
                            ),
                            class_="col-lg-6 col-xl-6 mb-3",
                            style="min-width: 450px;",
                        ),
                        ui.div(
                            ui.div(
                                ui.h5("Performance Metrics", class_="card-title text-primary mb-3"),
                                ui.div(
                                    ui.output_table(
                                        "output_ID_tab_portfolios_subtab_portfolios_analysis_table_quantstats_metrics",
                                    ),
                                    style="max-height: 500px; overflow-y: auto; overflow-x: auto; width: 100%;",
                                ),
                                class_="card card-body h-100",
                                style="min-width: 450px;",
                            ),
                            class_="col-lg-6 col-xl-6 mb-3",
                            style="min-width: 450px;",
                        ),
                        class_="row",
                        style="min-width: 900px;",
                    ),
                    class_="mt-4",
                    style="width: 100%; overflow-x: auto;",
                ),
                class_="flex-fill",
            ),
        ),
    )


@module.server
def subtab_portfolios_analysis_server(
    input: typing.Any,
    output: typing.Any,
    session: typing.Any,
    data_utils: dict,
    data_inputs: dict,
    reactives_shiny: dict,
) -> None:
    """
    Server logic for the Portfolio Analysis subtab.

    Implements reactive server functions for data processing, plot generation,
    and table calculations. Handles user interactions and updates visualizations
    based on selected time periods, analysis types, and options.

    Args:
        input (typing.Any): Shiny input object containing user selections
            - Contains reactive values from UI components
            - Accessed via input.component_id() for current values
            - Automatically triggers reactivity when values change
        output (typing.Any): Shiny output object for rendering components
            - Used to define output functions with @output decorator
            - Renders plots, tables, and UI elements
            - Manages reactive updates to the user interface
        session (typing.Any): Shiny session object for reactive context
            - Provides session-specific information and utilities
            - Manages client-server communication
            - Handles reactive graph and execution context
        data_utils (dict): Dictionary containing utility functions and configurations
            - Helper functions for data processing and analysis
            - Configuration parameters and settings
            - Shared utilities across dashboard modules
        data_inputs (dict): Dictionary containing portfolio and benchmark data
            Expected structure:
            - "My_Portfolio": Polars DataFrame with Date and Value columns
            - "Benchmark_Portfolio": Polars DataFrame with Date and Value columns
            - "Weights_My_Portfolio": Portfolio weights data (optional)
            - "Time_Series_ETFs": ETF time series data (optional)
        reactives_shiny (dict): Dictionary containing reactive Shiny variables
            - Shared reactive values across modules
            - Cross-module communication and state management
            - Global application state variables

    Raises
    ------
        ValueError: If portfolio or benchmark data validation fails
            - Thrown when required data structure is invalid
            - Includes detailed validation error messages
            - Prevents server initialization with bad data
        RuntimeError: If reactive calculations encounter errors
            - Thrown when reactive functions fail unexpectedly
            - Includes error context and debugging information
            - Handles unexpected computational failures

    Server Functions:
        The function defines multiple nested reactive functions:

        - **Date Range Calculation**: Converts time period selections to actual dates
        - **Data Filtering**: Processes and filters portfolio/benchmark data by date
        - **Plot Generation**: Creates interactive analysis visualizations
        - **Table Calculation**: Generates performance metrics and statistics
        - **Status Management**: Provides loading states and error messages
        - **Information Display**: Shows data availability and analysis readiness

    Data Processing Pipeline:
        1. **Input Validation**: Validates portfolio and benchmark data on initialization
        2. **Date Processing**: Handles different date formats and time period selections
        3. **Data Filtering**: Applies time-based filtering with proper format conversions
        4. **Performance Optimization**: Applies downsampling for improved rendering
        5. **Analysis Calculation**: Performs statistical calculations and visualizations
        6. **Output Formatting**: Formats results for display with appropriate styling

    Reactive Dependencies:
        The server creates reactive relationships between:

        - User input selections (time period, analysis type, options)
        - Data filtering and processing operations
        - Plot generation and visualization updates
        - Table calculations and formatting
        - Status indicators and information displays

    Error Handling Strategy:
        - **Input Validation**: Early returns for missing or invalid data
        - **Configuration Validation**: Checks for proper user input selections
        - **Business Logic Validation**: Ensures data meets analysis requirements
        - **Defensive Programming**: Graceful fallbacks for edge cases
        - **Exception Handling**: Comprehensive error catching with user feedback

    Performance Considerations:
        - **Data Downsampling**: Limits data points to 200 maximum for optimal rendering
        - **Lazy Evaluation**: Reactive calculations only execute when dependencies change
        - **Efficient Operations**: Uses Polars DataFrame operations for speed
        - **Optional Features**: PNG saving disabled by default to reduce overhead
        - **Memory Management**: Proper cleanup of temporary data structures

    Example Usage:
        ```python
        # Server function is called automatically by Shiny framework
        server_instance = subtab_portfolios_analysis_server(
            input=shiny_input_object,
            output=shiny_output_object,
            session=shiny_session_object,
            data_utils=utility_functions_dict,
            data_inputs=portfolio_data_dict,
            reactives_shiny=shared_reactive_values,
        )
        ```

    Note:
        This function is decorated with @module.server and should only be called
        by the Shiny framework. It establishes the reactive server context for
        the portfolio analysis subtab and manages all server-side logic.
    """
    data_portfolio = data_inputs.get("My_Portfolio")
    data_benchmark = data_inputs.get("Benchmark_Portfolio")
    # Note: Weights_My_Portfolio and Time_Series_ETFs available in data_inputs if needed for future features

    validation_portfolio = validate_portfolio_data(data_portfolio=data_portfolio)
    validation_benchmark = validate_portfolio_data(data_portfolio=data_benchmark)

    # Check portfolio validation and throw error if validation failed
    if not validation_portfolio[0]:
        raise ValueError(f"Portfolio data validation failed: {validation_portfolio[1]}")

    # Check benchmark validation and throw error if validation failed
    if not validation_benchmark[0]:
        raise ValueError(f"Benchmark data validation failed: {validation_benchmark[1]}")

    @output
    @render.text
    def output_ID_tab_portfolios_subtab_portfolios_analysis_calculated_date_range() -> str | None:
        """Display the calculated date range for preset selections."""
        try:
            time_period = input.input_ID_tab_portfolios_subtab_portfolios_analysis_time_period()

            if time_period != "custom":
                today_datetime = datetime.now(UTC)
                today_str = today_datetime.strftime("%Y-%m-%d")

                if time_period == "1y":
                    start_date_datetime = today_datetime - timedelta(days=365)
                elif time_period == "3y":
                    start_date_datetime = today_datetime - timedelta(days=365 * 3)
                elif time_period == "5y":
                    start_date_datetime = today_datetime - timedelta(days=365 * 5)
                elif time_period == "10y":
                    start_date_datetime = today_datetime - timedelta(days=365 * 10)
                elif time_period == "ytd":
                    start_date_datetime = datetime(today_datetime.year, 1, 1, tzinfo=UTC)
                else:
                    start_date_datetime = today_datetime - timedelta(days=365)

                start_date_str = start_date_datetime.strftime("%Y-%m-%d")
                return f"📅 {start_date_str} to {today_str}"
            return ""
        except Exception as exc:
            raise RuntimeError(f"Error displaying calculated date range: {exc}")

    @reactive.calc
    def get_filtered_analysis_data() -> tuple[pl.DataFrame, pl.DataFrame]:
        """
        Get and filter both portfolio and benchmark data for analysis.

        Reactive calculation that processes raw portfolio and benchmark data
        based on user-selected time periods. Handles date filtering, format
        conversion, and data validation to prepare clean datasets for analysis.

        Returns
        -------
            tuple[pl.DataFrame, pl.DataFrame]: Filtered portfolio and benchmark data
                - **Portfolio DataFrame**: Date (Datetime) and Value (Float64) columns
                - **Benchmark DataFrame**: Date (Datetime) and Value (Float64) columns
                - **Empty DataFrames**: Returned with proper schema if no valid data available

        Reactive Dependencies:
            - `input.input_ID_tab_portfolios_subtab_portfolios_analysis_time_period`: Time period selection
            - `input.input_ID_tab_portfolios_subtab_portfolios_analysis_date_range`: Custom date range input

        Data Processing Pipeline:
            1. **Date Range Calculation**: Extract time period and calculate date bounds
            2. **Date Format Handling**: Process different date column formats (string, date, datetime)
            3. **String Conversion**: Convert dates to strings for consistent filtering operations
            4. **Date Range Filtering**: Apply time-based filters using string comparison
            5. **Schema Conversion**: Convert back to Polars Datetime for Plotly compatibility
            6. **Performance Optimization**: Apply downsampling for optimal rendering

        Time Period Options:
            - **custom**: User-defined date range from date picker
            - **1y**: Last 365 days from current date
            - **3y**: Last 3 years (1095 days) from current date
            - **5y**: Last 5 years (1825 days) from current date
            - **10y**: Last 10 years (3650 days) from current date
            - **ytd**: Year-to-date (January 1st to current date)

        Date Format Handling:
            The function handles multiple date column formats:

            - **String Format (pl.Utf8/pl.String)**:
                - Attempts datetime parsing with `str.to_datetime()`
                - Falls back to date parsing with `str.to_date()`
                - Final fallback assumes valid string format
            - **Date Format (pl.Date)**:
                - Direct conversion to string using `dt.strftime()`
            - **Datetime Format (pl.Datetime)**:
                - Conversion to date string using `dt.strftime()`
            - **Unknown Formats**:
                - Casting to string then datetime parsing
                - Final fallback to direct string casting

        Error Handling Strategy:
            - **Input Validation**: Early returns for null or missing data
            - **Configuration Validation**: Proper date format and range validation
            - **Business Logic Validation**: Ensures required columns exist
            - **Defensive Programming**: Graceful fallbacks for date conversion failures
            - **Exception Handling**: Comprehensive error catching with detailed context

        Performance Optimizations:
            - **Data Downsampling**: Limits to 200 points maximum for optimal rendering
            - **Efficient Operations**: Uses Polars DataFrame operations for speed
            - **String-based Filtering**: Consistent approach across different date formats
            - **Early Returns**: Avoids processing when no valid data available
            - **Memory Management**: Proper cleanup of intermediate DataFrames

        Data Schema Requirements:
            Input DataFrames must contain:

            - **Date Column**: Any valid date format (string, Date, Datetime)
            - **Value Column**: Numeric values (convertible to Float64)

            Output DataFrames will have:

            - **Date Column**: Polars Datetime type for Plotly compatibility
            - **Value Column**: Polars Float64 type for numerical operations

        Example Usage:
            ```python
            # Reactive calculation automatically triggered by input changes
            portfolio_data, benchmark_data = get_filtered_analysis_data()

            # Check data availability
            if not portfolio_data.is_empty():
                point_count = portfolio_data.height
                date_range = (portfolio_data["Date"].min(), portfolio_data["Date"].max())
            ```

        Date Range Processing Examples:
            ```python
            # 1 Year selection
            # Input: time_period = "1y"
            # Output: 2024-01-01 to 2025-01-01 (365 days)

            # Custom selection
            # Input: time_period = "custom", date_range = [date(2023,6,1), date(2024,6,1)]
            # Output: 2023-06-01 to 2024-06-01

            # Year-to-date selection
            # Input: time_period = "ytd"
            # Output: 2025-01-01 to 2025-06-10 (current year)
            ```

        Raises
        ------
            RuntimeError: If data processing encounters unexpected errors
                - Includes detailed error context and debugging information
                - Preserves original exception for troubleshooting
                - Provides clear indication of failure point

        Note:
            This function is decorated with `@reactive.calc` and will automatically
            re-execute when any of its reactive dependencies change. The results
            are cached until dependencies change, improving performance.
        """
        # Input validation with early returns
        # Defensive programming - handle missing data gracefully
        if data_portfolio is None and data_benchmark is None:
            # Return empty DataFrames with proper schema if no data available
            empty_schema = {"Date": pl.Datetime, "Value": pl.Float64}
            empty_dataframe_portfolio = pl.DataFrame({"Date": [], "Value": []}, schema=empty_schema)
            empty_dataframe_benchmark = pl.DataFrame({"Date": [], "Value": []}, schema=empty_schema)
            return empty_dataframe_portfolio, empty_dataframe_benchmark

        # Data processing with comprehensive error handling
        # Only use try-except for operations that might fail unpredictably
        try:
            # === DATE RANGE CALCULATION SECTION ===
            # Get date range parameters directly as strings to avoid JSON serialization issues
            time_period = input.input_ID_tab_portfolios_subtab_portfolios_analysis_time_period()
            today_datetime = datetime.now(UTC)
            today_str = today_datetime.strftime("%Y-%m-%d")

            # DEBUG: Log original data availability before filtering
            print(f"\n{'=' * 60}")
            print("🔍 DEBUG: Date Filtering Analysis")
            print(f"{'=' * 60}")
            print(f"Selected time period: {time_period}")

            # Log original portfolio data information
            if data_portfolio is not None and not data_portfolio.is_empty():
                original_portfolio_count = data_portfolio.height
                portfolio_date_column_type = data_portfolio.select("Date").dtypes[0]
                print("\n📊 Original Portfolio Data:")
                print(f"   - Total rows: {original_portfolio_count}")
                print(f"   - Date column type: {portfolio_date_column_type}")

                # Get sample of dates to understand format
                sample_dates = data_portfolio.select("Date").head(5).to_series().to_list()
                print(f"   - Sample dates (first 5): {sample_dates}")

                # Get min/max dates for available range
                try:
                    if portfolio_date_column_type in [pl.Utf8, pl.String]:
                        # String dates - try to parse for min/max
                        try:
                            date_range_portfolio = (
                                data_portfolio.with_columns(
                                    [pl.col("Date").str.to_date(strict=False)],
                                )
                                .select(
                                    [
                                        pl.col("Date").min().alias("min_date"),
                                        pl.col("Date").max().alias("max_date"),
                                    ],
                                )
                                .to_dicts()[0]
                            )
                        except Exception:
                            # Fallback to string comparison
                            date_range_portfolio = data_portfolio.select(
                                [
                                    pl.col("Date").min().alias("min_date"),
                                    pl.col("Date").max().alias("max_date"),
                                ],
                            ).to_dicts()[0]
                    else:
                        date_range_portfolio = data_portfolio.select(
                            [
                                pl.col("Date").min().alias("min_date"),
                                pl.col("Date").max().alias("max_date"),
                            ],
                        ).to_dicts()[0]

                    print(
                        f"   - Available date range: {date_range_portfolio['min_date']} to {date_range_portfolio['max_date']}",
                    )
                except Exception as date_range_exc:
                    print(f"   - Could not determine date range: {date_range_exc}")
            else:
                print("\n❌ Original Portfolio Data: None or empty")

            # Calculate date range as strings to avoid JSON serialization issues
            # Configuration validation for different time period options
            if time_period == "custom":
                # Handle custom date range input with comprehensive validation
                try:
                    date_range = (
                        input.input_ID_tab_portfolios_subtab_portfolios_analysis_date_range()
                    )
                    if date_range and len(date_range) == 2:
                        # Convert date objects to strings immediately to avoid JSON serialization issues
                        # Handle both date objects and string representations
                        if hasattr(date_range[0], "strftime"):
                            # Date object - convert to string format
                            start_date_str = date_range[0].strftime("%Y-%m-%d")
                        else:
                            # Already a string, ensure proper format (YYYY-MM-DD)
                            start_date_str = str(date_range[0])[
                                :10
                            ]  # Take first 10 chars (YYYY-MM-DD)

                        if hasattr(date_range[1], "strftime"):
                            # Date object - convert to string format
                            end_date_str = date_range[1].strftime("%Y-%m-%d")
                        else:
                            # Already a string, ensure proper format (YYYY-MM-DD)
                            end_date_str = str(date_range[1])[
                                :10
                            ]  # Take first 10 chars (YYYY-MM-DD)
                    else:
                        # Defensive programming - fallback to 1 year for invalid custom range
                        start_date_datetime = today_datetime - timedelta(days=365)
                        start_date_str = start_date_datetime.strftime("%Y-%m-%d")
                        end_date_str = today_str
                except Exception:
                    # Defensive programming - fallback to 1 year for any custom range errors
                    start_date_datetime = today_datetime - timedelta(days=365)
                    start_date_str = start_date_datetime.strftime("%Y-%m-%d")
                    end_date_str = today_str

            elif time_period == "1y":
                # Last 1 year calculation (365 days)
                start_date_datetime = today_datetime - timedelta(days=365)
                start_date_str = start_date_datetime.strftime("%Y-%m-%d")
                end_date_str = today_str
            elif time_period == "3y":
                # Last 3 years calculation (1095 days)
                start_date_datetime = today_datetime - timedelta(days=365 * 3)
                start_date_str = start_date_datetime.strftime("%Y-%m-%d")
                end_date_str = today_str
            elif time_period == "5y":
                # Last 5 years calculation (1825 days)
                start_date_datetime = today_datetime - timedelta(days=365 * 5)
                start_date_str = start_date_datetime.strftime("%Y-%m-%d")
                end_date_str = today_str
            elif time_period == "10y":
                # Last 10 years calculation (3650 days)
                start_date_datetime = today_datetime - timedelta(days=365 * 10)
                start_date_str = start_date_datetime.strftime("%Y-%m-%d")
                end_date_str = today_str
            elif time_period == "ytd":
                # Year-to-date calculation (January 1st to current date)
                year_start_datetime = datetime(today_datetime.year, 1, 1, tzinfo=UTC)
                start_date_str = year_start_datetime.strftime("%Y-%m-%d")
                end_date_str = today_str
            else:
                # Configuration validation - default to 1 year for invalid selections
                start_date_datetime = today_datetime - timedelta(days=365)
                start_date_str = start_date_datetime.strftime("%Y-%m-%d")
                end_date_str = today_str

            print("\n📅 Filter Date Range:")
            print(f"   - Start: {start_date_str}")
            print(f"   - End: {end_date_str}")

            # === PORTFOLIO DATA PROCESSING SECTION ===
            # Process portfolio data if available with comprehensive validation
            if (
                data_portfolio is not None
                and isinstance(data_portfolio, pl.DataFrame)
                and not data_portfolio.is_empty()
            ):
                # Business logic validation - ensure required columns exist
                if "Date" in data_portfolio.columns and "Value" in data_portfolio.columns:
                    # Check the current data type of Date column for proper processing
                    date_dtype = data_portfolio.select("Date").dtypes[0]

                    print("\n🔧 Processing Portfolio Data:")
                    print(f"   - Date column type: {date_dtype}")

                    # Handle different date column types - always convert to string for filtering
                    # This approach ensures consistent filtering regardless of input format
                    if date_dtype in [pl.Utf8, pl.String]:
                        # String format - normalize to YYYY-MM-DD format for consistent filtering
                        try:
                            # Try datetime parsing first
                            filtered_portfolio = data_portfolio.with_columns(
                                [
                                    pl.col("Date")
                                    .str.to_datetime(strict=False)
                                    .dt.strftime("%Y-%m-%d")
                                    .alias("Date_String"),
                                ],
                            )
                            print("   ✅ Converted string to datetime then to YYYY-MM-DD format")
                        except Exception:
                            # Fallback to date parsing
                            try:
                                filtered_portfolio = data_portfolio.with_columns(
                                    [
                                        pl.col("Date")
                                        .str.to_date(strict=False)
                                        .dt.strftime("%Y-%m-%d")
                                        .alias("Date_String"),
                                    ],
                                )
                                print("   ✅ Converted string to date then to YYYY-MM-DD format")
                            except Exception:
                                # Last fallback - assume already in valid format, just clean it
                                filtered_portfolio = data_portfolio.with_columns(
                                    [pl.col("Date").str.slice(0, 10).alias("Date_String")],
                                )
                                print("   ⚠️  Using string as-is (first 10 characters)")
                    elif date_dtype == pl.Date:
                        # Already a date, convert to string format for filtering
                        filtered_portfolio = data_portfolio.with_columns(
                            [pl.col("Date").dt.strftime("%Y-%m-%d").alias("Date_String")],
                        )
                        print("   ✅ Converted Date to YYYY-MM-DD string format")
                    elif date_dtype == pl.Datetime:
                        # Convert datetime to date string for filtering
                        filtered_portfolio = data_portfolio.with_columns(
                            [pl.col("Date").dt.strftime("%Y-%m-%d").alias("Date_String")],
                        )
                        print("   ✅ Converted Datetime to YYYY-MM-DD string format")
                    else:
                        # Unknown format - try to cast to string first, then parse
                        try:
                            filtered_portfolio = data_portfolio.with_columns(
                                [
                                    pl.col("Date")
                                    .cast(pl.Utf8)
                                    .str.to_datetime(strict=False)
                                    .dt.strftime("%Y-%m-%d")
                                    .alias("Date_String"),
                                ],
                            )
                            print(
                                "   ✅ Cast unknown type to string, parsed as datetime, converted to YYYY-MM-DD",
                            )
                        except Exception:
                            # Final fallback - cast directly to string and clean
                            filtered_portfolio = data_portfolio.with_columns(
                                [
                                    pl.col("Date")
                                    .cast(pl.Utf8)
                                    .str.slice(0, 10)
                                    .alias("Date_String"),
                                ],
                            )
                            print("   ⚠️  Cast to string and took first 10 characters")

                    # DEBUG: Check date string format before filtering
                    sample_date_strings = (
                        filtered_portfolio.select("Date_String").head(5).to_series().to_list()
                    )
                    print(f"   - Sample Date_String values: {sample_date_strings}")

                    rows_before_filtering = filtered_portfolio.height
                    print(f"   - Rows before date filtering: {rows_before_filtering}")

                    # Apply date range filtering using string comparison only
                    # String comparison is consistent across all date formats
                    filtered_portfolio = filtered_portfolio.filter(
                        pl.col("Date_String").is_not_null(),  # Remove rows with null dates
                    )

                    rows_after_null_removal = filtered_portfolio.height
                    print(f"   - Rows after null removal: {rows_after_null_removal}")

                    filtered_portfolio = filtered_portfolio.filter(
                        (pl.col("Date_String") >= start_date_str)
                        & (pl.col("Date_String") <= end_date_str),
                    )

                    rows_after_date_filtering = filtered_portfolio.height
                    print(f"   - Rows after date filtering: {rows_after_date_filtering}")

                    # Early return validation - check if filtering removed all data
                    if filtered_portfolio.is_empty():
                        print("\n❌ WARNING: Date filtering removed all portfolio data!")
                        print(f"   - Filter range: {start_date_str} to {end_date_str}")
                        print(
                            "   - This usually means the filter date range is outside available data",
                        )
                        print("   - Consider using 'All Time' or adjusting date range")

                        # Create empty DataFrame with proper schema
                        filtered_portfolio = pl.DataFrame(
                            {"Date": [], "Value": []},
                            schema={"Date": pl.Datetime, "Value": pl.Float64},
                        )
                    else:
                        # Convert string back to Polars Datetime type (NOT Date) for Plotly compatibility
                        # Plotly requires Datetime type for proper time series handling
                        try:
                            filtered_portfolio = (
                                filtered_portfolio.with_columns(
                                    [
                                        pl.col("Date_String")
                                        .str.to_datetime(format="%Y-%m-%d")
                                        .alias("Date"),
                                    ],
                                )
                                .select(["Date", "Value"])
                                .sort("Date")
                            )

                            print("   ✅ Successfully converted to Datetime and sorted")
                            print(
                                f"   - Final filtered range: {filtered_portfolio.select('Date').min().item()} to {filtered_portfolio.select('Date').max().item()}",
                            )
                        except Exception as datetime_conversion_exc:
                            print(
                                f"   ❌ Error converting Date_String to Datetime: {datetime_conversion_exc}",
                            )
                            # If conversion fails, create empty DataFrame with Datetime schema
                            filtered_portfolio = pl.DataFrame(
                                {"Date": [], "Value": []},
                                schema={"Date": pl.Datetime, "Value": pl.Float64},
                            )
                else:
                    print("\n❌ Portfolio data missing required columns (Date, Value)")
                    # Configuration validation - missing required columns, create empty DataFrame with proper schema
                    filtered_portfolio = pl.DataFrame(
                        {"Date": [], "Value": []},
                        schema={"Date": pl.Datetime, "Value": pl.Float64},
                    )
            else:
                print("\n❌ Portfolio data is None, not a DataFrame, or empty")
                # Input validation - no valid portfolio data, create empty DataFrame with proper schema
                filtered_portfolio = pl.DataFrame(
                    {"Date": [], "Value": []},
                    schema={"Date": pl.Datetime, "Value": pl.Float64},
                )

            # === BENCHMARK DATA PROCESSING SECTION ===
            # Process benchmark data if available (identical approach as portfolio)
            if (
                data_benchmark is not None
                and isinstance(data_benchmark, pl.DataFrame)
                and not data_benchmark.is_empty()
            ):
                # Business logic validation - ensure required columns exist
                if "Date" in data_benchmark.columns and "Value" in data_benchmark.columns:
                    # Check the current data type of Date column for proper processing
                    date_dtype = data_benchmark.select("Date").dtypes[0]

                    print("\n🔧 Processing Benchmark Data:")
                    print(f"   - Date column type: {date_dtype}")

                    # Handle different date column types using same logic as portfolio
                    if date_dtype in [pl.Utf8, pl.String]:
                        # String format - normalize to YYYY-MM-DD format
                        try:
                            filtered_benchmark = data_benchmark.with_columns(
                                [
                                    pl.col("Date")
                                    .str.to_datetime(strict=False)
                                    .dt.strftime("%Y-%m-%d")
                                    .alias("Date_String"),
                                ],
                            )
                            print("   ✅ Converted string to datetime then to YYYY-MM-DD format")
                        except Exception:
                            try:
                                filtered_benchmark = data_benchmark.with_columns(
                                    [
                                        pl.col("Date")
                                        .str.to_date(strict=False)
                                        .dt.strftime("%Y-%m-%d")
                                        .alias("Date_String"),
                                    ],
                                )
                                print("   ✅ Converted string to date then to YYYY-MM-DD format")
                            except Exception:
                                filtered_benchmark = data_benchmark.with_columns(
                                    [pl.col("Date").str.slice(0, 10).alias("Date_String")],
                                )
                                print("   ⚠️  Using string as-is (first 10 characters)")
                    elif date_dtype == pl.Date:
                        filtered_benchmark = data_benchmark.with_columns(
                            [pl.col("Date").dt.strftime("%Y-%m-%d").alias("Date_String")],
                        )
                        print("   ✅ Converted Date to YYYY-MM-DD string format")
                    elif date_dtype == pl.Datetime:
                        filtered_benchmark = data_benchmark.with_columns(
                            [pl.col("Date").dt.strftime("%Y-%m-%d").alias("Date_String")],
                        )
                        print("   ✅ Converted Datetime to YYYY-MM-DD string format")
                    else:
                        try:
                            filtered_benchmark = data_benchmark.with_columns(
                                [
                                    pl.col("Date")
                                    .cast(pl.Utf8)
                                    .str.to_datetime(strict=False)
                                    .dt.strftime("%Y-%m-%d")
                                    .alias("Date_String"),
                                ],
                            )
                            print(
                                "   ✅ Cast unknown type to string, parsed as datetime, converted to YYYY-MM-DD",
                            )
                        except Exception:
                            filtered_benchmark = data_benchmark.with_columns(
                                [
                                    pl.col("Date")
                                    .cast(pl.Utf8)
                                    .str.slice(0, 10)
                                    .alias("Date_String"),
                                ],
                            )
                            print("   ⚠️  Cast to string and took first 10 characters")

                    rows_before_filtering = filtered_benchmark.height
                    print(f"   - Rows before date filtering: {rows_before_filtering}")

                    # Apply date range filtering using string comparison only
                    filtered_benchmark = filtered_benchmark.filter(
                        pl.col("Date_String").is_not_null(),
                    ).filter(
                        (pl.col("Date_String") >= start_date_str)
                        & (pl.col("Date_String") <= end_date_str),
                    )

                    rows_after_date_filtering = filtered_benchmark.height
                    print(f"   - Rows after date filtering: {rows_after_date_filtering}")

                    # Early return validation
                    if filtered_benchmark.is_empty():
                        print("\n⚠️  Date filtering removed all benchmark data")
                        filtered_benchmark = pl.DataFrame(
                            {"Date": [], "Value": []},
                            schema={"Date": pl.Datetime, "Value": pl.Float64},
                        )
                    else:
                        # Convert string back to Polars Datetime type
                        try:
                            filtered_benchmark = (
                                filtered_benchmark.with_columns(
                                    [
                                        pl.col("Date_String")
                                        .str.to_datetime(format="%Y-%m-%d")
                                        .alias("Date"),
                                    ],
                                )
                                .select(["Date", "Value"])
                                .sort("Date")
                            )
                            print("   ✅ Successfully converted to Datetime and sorted")
                        except Exception as datetime_conversion_exc:
                            print(
                                f"   ❌ Error converting Date_String to Datetime: {datetime_conversion_exc}",
                            )
                            filtered_benchmark = pl.DataFrame(
                                {"Date": [], "Value": []},
                                schema={"Date": pl.Datetime, "Value": pl.Float64},
                            )
                else:
                    print("\n⚠️  Benchmark data missing required columns")
                    filtered_benchmark = pl.DataFrame(
                        {"Date": [], "Value": []},
                        schema={"Date": pl.Datetime, "Value": pl.Float64},
                    )
            else:
                print("\n⚠️  Benchmark data is None, not a DataFrame, or empty")
                filtered_benchmark = pl.DataFrame(
                    {"Date": [], "Value": []},
                    schema={"Date": pl.Datetime, "Value": pl.Float64},
                )

            print(f"\n{'=' * 60}\n")

            # === PERFORMANCE OPTIMIZATION SECTION ===
            # Apply downsampling for performance - only if data exists
            # Input validation with early returns

            # Process portfolio data with defensive programming
            if not filtered_portfolio.is_empty():
                filtered_portfolio_downsampled = downsample_dataframe(
                    filtered_portfolio,
                    max_points=200,
                    date_column="Date",
                )
            else:
                # Early return - preserve empty DataFrame with proper schema
                filtered_portfolio_downsampled = pl.DataFrame(
                    {"Date": [], "Value": []},
                    schema={"Date": pl.Datetime, "Value": pl.Float64},
                )

            # Process benchmark data with defensive programming
            if not filtered_benchmark.is_empty():
                filtered_benchmark_downsampled = downsample_dataframe(
                    filtered_benchmark,
                    max_points=200,
                    date_column="Date",
                )
            else:
                # Early return - preserve empty DataFrame with proper schema
                filtered_benchmark_downsampled = pl.DataFrame(
                    {"Date": [], "Value": []},
                    schema={"Date": pl.Datetime, "Value": pl.Float64},
                )

            return filtered_portfolio_downsampled, filtered_benchmark_downsampled

        except Exception as exc:
            # Comprehensive error handling with detailed context
            # Preserve original exception for debugging while providing clear error message
            raise RuntimeError(f"Error in get_filtered_analysis_data: {exc}") from exc

    @output
    @render.ui
    def output_ID_tab_portfolios_subtab_portfolios_analysis_data_info():
        """Display data information in the sidebar."""
        try:
            data_portfolio_filtered, data_benchmark_filtered = get_filtered_analysis_data()

            # Get current time period with defensive programming
            try:
                time_period = input.input_ID_tab_portfolios_subtab_portfolios_analysis_time_period()
            except Exception:
                time_period = "1y"

            period_labels = {
                "1y": "Last 1 Year",
                "3y": "Last 3 Years",
                "5y": "Last 5 Years",
                "10y": "Last 10 Years",
                "ytd": "Year to Date",
                "custom": "Custom Period",
            }
            period_label = period_labels.get(time_period, "Selected Period")

            # Get analysis type with defensive programming
            try:
                analysis_type = input.input_ID_tab_portfolios_subtab_portfolios_analysis_type()
                analysis_labels = {
                    "returns": "Returns Distribution",
                    "drawdowns": "Drawdowns Analysis",
                    "rolling": "Rolling Statistics",
                    "comparison": "Portfolio vs Benchmark",
                }
                analysis_label = analysis_labels.get(analysis_type, "Analysis")
            except Exception:
                analysis_label = "Analysis"

            # Create info display
            info_items = []

            # Period info
            info_items.append(
                ui.div(ui.strong("Period: "), period_label, class_="small text-muted mb-1"),
            )

            # Analysis type info
            info_items.append(
                ui.div(ui.strong("Analysis: "), analysis_label, class_="small text-muted mb-1"),
            )

            # Portfolio data info - use Polars methods
            portfolio_count = (
                data_portfolio_filtered.height if not data_portfolio_filtered.is_empty() else 0
            )
            info_items.append(
                ui.div(
                    ui.strong("Portfolio Data: "),
                    f"{portfolio_count} points" if portfolio_count > 0 else "No data",
                    class_="small text-muted mb-1",
                ),
            )

            # Benchmark data info - use Polars methods
            # Calculate benchmark count for display
            benchmark_count = (
                data_benchmark_filtered.height if not data_benchmark_filtered.is_empty() else 0
            )
            info_items.append(
                ui.div(
                    ui.strong("Benchmark Data: "),
                    f"{benchmark_count} points" if benchmark_count > 0 else "No data",
                    class_="small text-muted mb-1",
                ),
            )

            # Status indicator
            if portfolio_count > 0:
                if portfolio_count >= 30:
                    status_icon = "✅"
                    status_text = "Ready for analysis"
                    status_class = "text-success"
                else:
                    status_icon = "⚠️"
                    status_text = "Limited data available"
                    status_class = "text-warning"
            else:
                status_icon = "❌"
                status_text = "No portfolio data"
                status_class = "text-danger"

            info_items.append(
                ui.div(
                    ui.span(status_icon, class_="me-1"),
                    ui.span(status_text, class_=f"small {status_class}"),
                    class_="mt-2 p-2 rounded bg-light",
                ),
            )

            return ui.div(*info_items)

        except Exception as exc:
            raise RuntimeError(f"Error generating analysis data info: {exc}") from exc

    @output
    @render.ui
    def output_ID_tab_portfolios_subtab_portfolios_analysis_loading_status():
        """Display loading status and any error messages."""
        try:
            # Check if data is available
            data_portfolio_filtered, _data_benchmark_filtered = get_filtered_analysis_data()

            portfolio_count = (
                data_portfolio_filtered.height if not data_portfolio_filtered.is_empty() else 0
            )
            # Note: benchmark_count available from data_benchmark_filtered.height if needed

            # Get current analysis type with defensive programming
            try:
                analysis_type = input.input_ID_tab_portfolios_subtab_portfolios_analysis_type()
                analysis_labels = {
                    "returns": "Returns Distribution",
                    "drawdowns": "Drawdowns Analysis",
                    "rolling": "Rolling Statistics",
                    "comparison": "Portfolio vs Benchmark",
                }
                current_analysis = analysis_labels.get(analysis_type, analysis_type)
            except Exception:
                current_analysis = "Loading..."

            # Create status message
            if portfolio_count == 0:
                status_class = "alert-danger"
                status_icon = "❌"
                status_message = "No portfolio data available for the selected period"
            elif portfolio_count < 30:
                status_class = "alert-warning"
                status_icon = "⚠️"
                status_message = f"Limited data available ({portfolio_count} points) - Analysis may be less accurate"
            else:
                status_class = "alert-success"
                status_icon = "✅"
                status_message = f"Ready for {current_analysis} ({portfolio_count} data points)"

            return ui.div(
                ui.div(
                    ui.span(status_icon, class_="me-2"),
                    status_message,
                    class_=f"alert {status_class} mb-3 py-2",
                ),
            )

        except Exception:
            return ui.div(
                ui.div("⏳ Loading analysis data...", class_="alert alert-info mb-3 py-2"),
            )

    @output
    @render_widget  # pyright: ignore[reportArgumentType]  # pyrefly: ignore[bad-specialization]  # go.Figure satisfies Widget protocol at runtime
    def output_ID_tab_portfolios_subtab_portfolios_analysis_plot_main():
        """Render the main analysis plot using plotly widgets."""
        try:
            # Get filtered data
            data_portfolio_filtered, data_benchmark_filtered = get_filtered_analysis_data()

            # Input validation with early returns
            if data_portfolio_filtered.is_empty():
                return create_error_figure(
                    "No portfolio data available for the selected time period",
                )

            # Get analysis type with defensive programming
            try:
                analysis_type = input.input_ID_tab_portfolios_subtab_portfolios_analysis_type()
            except Exception:
                analysis_type = "returns"

            # Get include benchmark flag with defensive programming
            try:
                include_benchmark = (
                    input.input_ID_tab_portfolios_subtab_portfolios_analysis_include_benchmark()
                )
            except Exception:
                include_benchmark = True

            # Prepare benchmark data if needed
            benchmark_data = None
            if include_benchmark and not data_benchmark_filtered.is_empty():
                benchmark_data = data_benchmark_filtered

            # Create plot based on analysis type
            if analysis_type == "returns":
                figure = create_plot_returns_distribution(
                    data_portfolio=data_portfolio_filtered,
                    data_benchmark=benchmark_data,
                    period_label="Returns Distribution Analysis",
                    data_was_validated=True,
                    include_benchmark=include_benchmark,
                )
            elif analysis_type == "drawdowns":
                figure = create_plot_drawdowns_analysis(
                    data_portfolio=data_portfolio_filtered,
                    data_benchmark=benchmark_data,
                    period_label="Drawdowns Analysis",
                    data_was_validated=True,
                    include_benchmark=include_benchmark,
                )
            elif analysis_type == "rolling":
                try:
                    window_size = (
                        input.input_ID_tab_portfolios_subtab_portfolios_analysis_rolling_window()
                    )
                except Exception:
                    window_size = 30

                figure = create_plot_rolling_statistics(
                    data_portfolio=data_portfolio_filtered,
                    data_benchmark=benchmark_data,
                    window_size=window_size,
                    period_label=f"Rolling Statistics ({window_size}-day window)",
                    data_was_validated=True,
                    include_benchmark=include_benchmark,
                )
            elif analysis_type == "comparison":
                figure = create_plot_portfolios_comparison(
                    data_portfolio=data_portfolio_filtered,
                    data_benchmark=benchmark_data,
                    period_label="Portfolio vs Benchmark Comparison",
                    data_was_validated=True,
                )
            else:
                # Configuration validation - default to returns distribution
                figure = create_plot_returns_distribution(
                    data_portfolio=data_portfolio_filtered,
                    data_benchmark=benchmark_data,
                    period_label="Returns Distribution Analysis",
                    data_was_validated=True,
                    include_benchmark=include_benchmark,
                )

            # Business logic validation - ensure figure is valid
            if figure is None:
                return create_error_figure("Failed to generate analysis plot")

            # Configure plot layout for better display
            figure.update_layout(
                height=550,
                margin={"l": 50, "r": 50, "t": 60, "b": 50},
                paper_bgcolor="white",
                plot_bgcolor="white",
                showlegend=True,
                legend={
                    "orientation": "h",
                    "yanchor": "bottom",
                    "y": 1.02,
                    "xanchor": "right",
                    "x": 1,
                },
            )

            # Store latest figure in shared reactive state for report pipeline
            update_visual_object_in_reactives(
                reactives_shiny,
                "Chart_Portfolio_Analysis",
                figure,
            )

            return figure

        except Exception as exc:
            error_message = f"Error generating analysis plot: {exc!s}"
            print(f"Plot generation error: {error_message}")
            return create_error_figure(error_message)

    @output
    @render.table  # pyrefly: ignore[bad-argument-type]
    def output_ID_tab_portfolios_subtab_portfolios_analysis_table_quantstats_metrics():
        """Render the performance metrics table using quantstats calculations."""
        try:
            # Get filtered data
            data_portfolio_filtered, data_benchmark_filtered = get_filtered_analysis_data()

            # Input validation with early returns
            if data_portfolio_filtered.is_empty():
                # Return empty table with proper structure
                return pd.DataFrame(
                    {"Metric": ["No Data Available"], "Portfolio": ["-"], "Benchmark": ["-"]},
                )

            # Get include benchmark flag with defensive programming
            try:
                include_benchmark = (
                    input.input_ID_tab_portfolios_subtab_portfolios_analysis_include_benchmark()
                )
            except Exception:
                include_benchmark = True

            # Prepare benchmark data if needed
            benchmark_data = None
            if include_benchmark and not data_benchmark_filtered.is_empty():
                benchmark_data = data_benchmark_filtered

            # Calculate performance metrics using utility function
            metrics_table = calculate_table_metrics_performance(
                data_portfolio=data_portfolio_filtered,
                data_benchmark=benchmark_data,
            )

            # Business logic validation - ensure table is valid
            if metrics_table is None or metrics_table.empty:
                # Return fallback table with error message
                return pd.DataFrame(
                    {
                        "Metric": ["Calculation Error"],
                        "Portfolio": ["-"],
                        "Benchmark": ["-"] if include_benchmark else [],
                    },
                )

            # Format the table for better display
            if "Portfolio" in metrics_table.columns:
                # Apply formatting to numerical values in Portfolio column
                for idx in metrics_table.index:
                    try:
                        value = metrics_table.at[idx, "Portfolio"]
                        if isinstance(value, (int, float)) and not pd.isna(value):
                            # Format different metrics appropriately
                            metric_name = str(metrics_table.at[idx, "Metric"]).lower()
                            if (
                                any(
                                    term in metric_name
                                    for term in ["return", "yield", "ratio", "volatility"]
                                )
                                or "drawdown" in metric_name
                            ):
                                metrics_table.at[idx, "Portfolio"] = format_value_for_display(
                                    value,
                                    format_type="percentage",
                                )
                            elif "days" in metric_name or "count" in metric_name:
                                metrics_table.at[idx, "Portfolio"] = format_value_for_display(
                                    value,
                                    format_type="integer",
                                )
                            else:
                                metrics_table.at[idx, "Portfolio"] = format_value_for_display(
                                    value,
                                    format_type="decimal",
                                )
                    except Exception as _fmt_exc:
                        # Keep original value if formatting fails
                        _logger.debug(
                            "Metrics Portfolio formatting failed, keeping original: %s",
                            _fmt_exc,
                        )
                        continue

            # Format benchmark column if it exists
            if "Benchmark" in metrics_table.columns:
                for idx in metrics_table.index:
                    try:
                        value = metrics_table.at[idx, "Benchmark"]
                        if isinstance(value, (int, float)) and not pd.isna(value):
                            # Format different metrics appropriately
                            metric_name = str(metrics_table.at[idx, "Metric"]).lower()
                            if (
                                any(
                                    term in metric_name
                                    for term in ["return", "yield", "ratio", "volatility"]
                                )
                                or "drawdown" in metric_name
                            ):
                                metrics_table.at[idx, "Benchmark"] = format_value_for_display(
                                    value,
                                    format_type="percentage",
                                )
                            elif "days" in metric_name or "count" in metric_name:
                                metrics_table.at[idx, "Benchmark"] = format_value_for_display(
                                    value,
                                    format_type="integer",
                                )
                            else:
                                metrics_table.at[idx, "Benchmark"] = format_value_for_display(
                                    value,
                                    format_type="decimal",
                                )
                    except Exception as _fmt_exc:
                        # Keep original value if formatting fails
                        _logger.debug(
                            "Metrics Benchmark formatting failed, keeping original: %s",
                            _fmt_exc,
                        )
                        continue

            return metrics_table

        except Exception as exc:
            error_message = f"Error generating performance metrics table: {exc!s}"
            print(f"Metrics table generation error: {error_message}")

            # Return error table
            return pd.DataFrame(
                {"Metric": ["Error"], "Portfolio": [error_message], "Benchmark": ["-"]},
            )

    @output
    @render.table  # pyrefly: ignore[bad-argument-type]
    def output_ID_tab_portfolios_subtab_portfolios_analysis_table_stats():
        """Render the basic statistics table for portfolio and benchmark data."""
        try:
            # Get filtered data
            data_portfolio_filtered, data_benchmark_filtered = get_filtered_analysis_data()

            # Input validation with early returns
            if data_portfolio_filtered.is_empty():
                # Return empty table with proper structure
                return pd.DataFrame(
                    {"Statistic": ["No Data Available"], "Portfolio": ["-"], "Benchmark": ["-"]},
                )

            # Get include benchmark flag with defensive programming
            try:
                include_benchmark = (
                    input.input_ID_tab_portfolios_subtab_portfolios_analysis_include_benchmark()
                )
            except Exception:
                include_benchmark = True

            # Prepare benchmark data if needed
            benchmark_data = None
            if include_benchmark and not data_benchmark_filtered.is_empty():
                benchmark_data = data_benchmark_filtered

            # Calculate basic statistics using utility function
            stats_table = calculate_table_stats_basic(
                data_portfolio=data_portfolio_filtered,
                data_benchmark=benchmark_data,
                include_benchmark=include_benchmark,
            )

            # Business logic validation - ensure table is valid
            if stats_table is None or stats_table.empty:
                # Return fallback table with error message
                return pd.DataFrame(
                    {
                        "Statistic": ["Calculation Error"],
                        "Portfolio": ["-"],
                        "Benchmark": ["-"] if include_benchmark else [],
                    },
                )

            # Format the table for better display
            if "Portfolio" in stats_table.columns:
                # Apply formatting to numerical values in Portfolio column
                for idx in stats_table.index:
                    try:
                        value = stats_table.at[idx, "Portfolio"]
                        if isinstance(value, (int, float)) and not pd.isna(value):
                            # Format different statistics appropriately
                            statistic_name = str(stats_table.at[idx, "Statistic"]).lower()
                            if any(
                                term in statistic_name
                                for term in ["mean", "median", "std", "variance", "volatility"]
                            ) or any(
                                term in statistic_name
                                for term in ["min", "max", "percentile", "quantile"]
                            ):
                                stats_table.at[idx, "Portfolio"] = format_value_for_display(
                                    value,
                                    format_type="decimal",
                                )
                            elif any(
                                term in statistic_name
                                for term in ["count", "observations", "points"]
                            ):
                                stats_table.at[idx, "Portfolio"] = format_value_for_display(
                                    value,
                                    format_type="integer",
                                )
                            elif any(term in statistic_name for term in ["skew", "kurtosis"]):
                                stats_table.at[idx, "Portfolio"] = format_value_for_display(
                                    value,
                                    format_type="decimal",
                                )
                            else:
                                stats_table.at[idx, "Portfolio"] = format_value_for_display(
                                    value,
                                    format_type="decimal",
                                )
                    except Exception as _fmt_exc:
                        # Keep original value if formatting fails
                        _logger.debug(
                            "Stats Portfolio formatting failed, keeping original: %s",
                            _fmt_exc,
                        )
                        continue

            # Format benchmark column if it exists
            if "Benchmark" in stats_table.columns:
                for idx in stats_table.index:
                    try:
                        value = stats_table.at[idx, "Benchmark"]
                        if isinstance(value, (int, float)) and not pd.isna(value):
                            # Format different statistics appropriately
                            statistic_name = str(stats_table.at[idx, "Statistic"]).lower()
                            if any(
                                term in statistic_name
                                for term in ["mean", "median", "std", "variance", "volatility"]
                            ) or any(
                                term in statistic_name
                                for term in ["min", "max", "percentile", "quantile"]
                            ):
                                stats_table.at[idx, "Benchmark"] = format_value_for_display(
                                    value,
                                    format_type="decimal",
                                )
                            elif any(
                                term in statistic_name
                                for term in ["count", "observations", "points"]
                            ):
                                stats_table.at[idx, "Benchmark"] = format_value_for_display(
                                    value,
                                    format_type="integer",
                                )
                            elif any(term in statistic_name for term in ["skew", "kurtosis"]):
                                stats_table.at[idx, "Benchmark"] = format_value_for_display(
                                    value,
                                    format_type="decimal",
                                )
                            else:
                                stats_table.at[idx, "Benchmark"] = format_value_for_display(
                                    value,
                                    format_type="decimal",
                                )
                    except Exception as _fmt_exc:
                        # Keep original value if formatting fails
                        _logger.debug(
                            "Stats Benchmark formatting failed, keeping original: %s",
                            _fmt_exc,
                        )
                        continue

            return stats_table

        except Exception as exc:
            error_message = f"Error generating basic statistics table: {exc!s}"
            print(f"Stats table generation error: {error_message}")

            # Return error table
            return pd.DataFrame(
                {"Statistic": ["Error"], "Portfolio": [error_message], "Benchmark": ["-"]},
            )
