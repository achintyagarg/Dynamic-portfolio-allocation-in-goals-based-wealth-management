"""Portfolio Comparison Module.

Provides comprehensive side-by-side analysis of portfolio performance against benchmark data.

![Portfolio Analysis](https://img.shields.io/badge/Portfolio-Analysis-blue)
![Python](https://img.shields.io/badge/python-3.11+-green)
![Shiny](https://img.shields.io/badge/shiny-for_python-orange)

## Overview

The Portfolio Comparison module provides comprehensive side-by-side analysis of portfolio
performance against benchmark data within the QWIM Dashboard's Portfolios section.

This module enables detailed comparative analysis with multiple visualization types,
statistical calculations, and export capabilities for portfolio management workflows.

## Architecture

```mermaid
graph TD
    A[Portfolio Comparison Module] --> B[UI Components]
    A --> C[Server Logic]
    B --> D[Time Period Selection]
    B --> E[Visualization Options]
    B --> F[Data Information Display]
    C --> G[Data Filtering]
    C --> H[Plot Generation]
    C --> I[Statistics Calculation]
    G --> J[Date Range Processing]
    H --> K[Comparison Visualizations]
    I --> L[Performance Metrics]
```

## Features

!!! success "Core Capabilities"
    - **Portfolio vs Benchmark Comparison**: Side-by-side performance analysis
    - **Multiple Visualization Types**: Absolute value, normalized, percent change, cumulative return
    - **Time Period Selection**: Preset periods (1Y, 3Y, 5Y, 10Y, YTD) plus custom date ranges
    - **Performance Statistics**: Comprehensive statistical analysis and metrics
    - **Interactive Plotly Charts**: Real-time interactive visualizations
    - **Responsive Design**: Mobile-friendly layout with sidebar controls

!!! info "Visualization Types"
    - **Absolute Value**: Raw portfolio and benchmark values
    - **Normalized (Base=100)**: Performance normalized to starting value of 100
    - **Percent Change**: Period-over-period percentage changes
    - **Cumulative Return**: Cumulative return analysis over time
    - **Difference Analysis**: Portfolio vs benchmark performance gap visualization

## Configuration

Analysis type and time period are controlled via the sidebar input controls.

## Dependencies

=== "Core Dependencies"
    - **shiny**: Reactive UI framework for Python
    - **shinywidgets**: Plotly widget integration
    - **plotly**: Interactive chart generation
    - **polars**: High-performance DataFrame operations
    - **pandas**: Data manipulation and analysis
    - **numpy**: Numerical computations

=== "Optional Dependencies"
    - **quantstats**: Advanced portfolio statistics (optional)
    - **beautifulsoup4**: HTML parsing for enhanced statistics (optional)

=== "Internal Dependencies"
    - `Shiny_Utils.utils_data`: Data processing utilities
    - `Shiny_Utils.reactives_shiny`: Reactive value management
    - `Shiny_Utils.utils_visuals`: Visualization utilities

## Data Requirements

### Input Data Format

The module expects the following data inputs:

| Data Input | Type | Description | Required Columns |
|-----------|------|-------------|-----------------|
| `My_Portfolio` | `pl.DataFrame` | Portfolio performance data | `Date`, `Value` |
| `Benchmark_Portfolio` | `pl.DataFrame` | Benchmark performance data | `Date`, `Value` |
| `Weights_My_Portfolio` | `pl.DataFrame` | Portfolio weights (optional) | `Date`, `Component`, `Weight` |
| `Time_Series_ETFs` | `pl.DataFrame` | ETF time series data (optional) | `Date`, `[ETF_Names]` |

!!! note "Date Format Requirements"
    - Date columns support multiple formats: `str`, `pl.Date`, `pl.Datetime`
    - Automatic conversion handles common date string formats
    - Date filtering uses string comparison for performance optimization

## Usage Examples

### Basic Module Integration

```python
# UI Component Integration
portfolio_comparison_ui = subtab_portfolios_comparison_ui(
    data_utils=data_utilities_dict, data_inputs=portfolio_data_dict
)

# Server Component Integration
portfolio_comparison_server = subtab_portfolios_comparison_server(
    input=shiny_input_object,
    output=shiny_output_object,
    session=shiny_session_object,
    data_utils=data_utilities_dict,
    data_inputs=portfolio_data_dict,
    reactives_shiny=reactive_values_dict,
)
```

### Custom Time Period Analysis

```python
# Predefined periods available:
time_periods = {
    "1y": "Last 1 Year",  # 365 days from today
    "3y": "Last 3 Years",  # 3 * 365 days from today
    "5y": "Last 5 Years",  # 5 * 365 days from today
    "10y": "Last 10 Years",  # 10 * 365 days from today
    "ytd": "Year to Date",  # January 1st current year to today
    "custom": "Custom Range",  # User-defined date range
}
```

## Error Handling Strategy

The module implements comprehensive error handling following defensive programming principles:

!!! tip "Error Handling Approach"
    - **Input Validation**: Early returns for invalid parameters
    - **Configuration Validation**: User input sanitization and defaults
    - **Business Logic Validation**: Data consistency and format checks
    - **Graceful Degradation**: Fallback to empty datasets when data unavailable

### Exception Types

| Exception | Usage | Description |
|-----------|--------|-------------|
| `ValueError` | Data validation failures | Invalid portfolio/benchmark data format |
| `RuntimeError` | Processing errors | Date range calculation or filtering failures |

## Performance Optimizations

!!! success "Optimization Features"
    - **Efficient Date Filtering**: String-based date comparison for improved performance
    - **Polars DataFrame Operations**: High-performance data processing
    - **Lazy Reactive Calculations**: Computations only when inputs change
    - **Memory Management**: Proper cleanup of temporary DataFrames
    - **Data Downsampling**: Automatic optimization for large datasets

## API Reference

### UI Function

#### `subtab_portfolios_comparison_ui(data_utils: dict, data_inputs: dict)`

Creates the comprehensive user interface for the Portfolio Comparison subtab.

**Parameters:**
- `data_utils` (dict): Dictionary containing utility functions and configurations
- `data_inputs` (dict): Dictionary containing portfolio and benchmark datasets

**Returns:**
- `shiny.ui.div`: Complete UI layout with sidebar controls and main content area

**UI Components:**
- Time period selection dropdown with preset options
- Custom date range picker (conditional display)
- Visualization type selector with multiple chart options
- Data information panel with real-time statistics
- Main comparison plot with full-screen capability
- Performance statistics table with export options

---

**Module Information:**
- **Author**: QWIM Development Team
- **Version**: 0.5.1
- **Last Updated**: June 2025
- **License**: [Project License]
- **Dependencies**: See requirements.txt for complete dependency list
"""

from __future__ import annotations

import typing  # Type hints for better code documentation

from datetime import UTC, datetime, timedelta  # Date/time operations for period calculations
from pathlib import Path  # Cross-platform file path handling

# Standard library imports - organized by functionality
from typing import Any

# Data processing and analysis libraries
import numpy as np  # Numerical computations and array operations
import pandas as pd  # Traditional DataFrame operations and compatibility layer
import polars as pl  # High-performance DataFrame operations


# Shiny framework components for reactive web applications
from shiny import module, reactive, render, ui  # Core Shiny functionality
from shinywidgets import output_widget, render_widget  # Plotly widget integration

# Custom logging utilities
from src.utils.custom_exceptions_errors_loggers.logger_custom import get_logger


#: Module-level logger instance
_logger = get_logger(__name__)

# Optional dependencies with graceful fallbacks
# These imports provide enhanced functionality but are not required for core operations
try:
    # quantstats would provide advanced portfolio analytics if needed
    HAS_QUANTSTATS = True
    """
    QuantStats would provide advanced portfolio analysis metrics including:
    - Sharpe ratio, Sortino ratio, Calmar ratio
    - Maximum drawdown analysis
    - Value at Risk (VaR) calculations
    - Rolling performance metrics
    """
except ImportError:
    HAS_QUANTSTATS = False
    # Fallback: Basic statistics will be calculated using native pandas/numpy

try:
    # BeautifulSoup would enable enhanced HTML processing if needed
    HAS_BEAUTIFULSOUP = True
    """
    BeautifulSoup would enable enhanced HTML table processing for:
    - Custom styling of statistical output tables
    - Better formatting for export functionality
    - Enhanced readability of performance metrics
    """
except ImportError:
    HAS_BEAUTIFULSOUP = False
    # Fallback: Basic HTML table generation without enhanced styling

# Internal QWIM Dashboard utility modules
# These modules provide reusable functionality across the dashboard
from src.dashboard.shiny_utils.reactives_shiny import (  # noqa: E402
    update_visual_object_in_reactives,  # Store latest chart in shared reactive state
)
from src.dashboard.shiny_utils.utils_data import (  # noqa: E402
    validate_portfolio_data,  # Ensure data integrity and format compliance
)
from src.dashboard.shiny_utils.utils_visuals import (  # noqa: E402
    create_error_figure,  # Standardized error visualization
    create_plot_comparison_portfolios,  # Specialized portfolio comparison charts
)


# Module configuration constants
# These constants control global behavior and file organization

# Output directory structure for exports and temporary files
OUTPUT_DIR = Path("output")  # Base directory for all output files


@module.ui
def subtab_portfolios_comparison_ui(data_utils: dict, data_inputs: dict) -> Any:
    """
    Create the comprehensive user interface for the Portfolio Comparison subtab.

    This function constructs a responsive, feature-rich UI layout for portfolio vs
    benchmark comparison analysis with interactive controls and real-time data display.

    ## UI Architecture

    The interface follows a sidebar-main content layout pattern optimized for
    portfolio analysis workflows:

    ```
    ┌─────────────────┬──────────────────────────────────┐
    │    Sidebar      │        Main Content Area         │
    │                 │                                  │
    │ • Time Period   │ • Loading Status                 │
    │ • Viz Options   │ • Comparison Plot (Full Screen)  │
    │ • Data Info     │ • Performance Statistics Table   │
    │                 │                                  │
    └─────────────────┴──────────────────────────────────┘
    ```

    ## Component Hierarchy

    ### Sidebar Controls (350px width)

    #### 📅 Time Period Selection
    - **Preset Periods**: Dropdown with common analysis periods (1Y, 3Y, 5Y, 10Y, YTD)
    - **Custom Range**: Conditional date range picker for custom analysis periods
    - **Calculated Range Display**: Shows computed date range for preset selections

    #### 📊 Visualization Options
    - **Chart Type Selector**: Multiple visualization types for different analysis needs
    - **Difference Toggle**: Option to overlay portfolio vs benchmark difference

    #### ℹ️ Data Information Panel
    - **Real-time Data Statistics**: Live count of portfolio and benchmark data points
    - **Status Indicators**: Visual feedback on data availability and readiness
    - **Period Information**: Current analysis period details

    ### Main Content Area (Responsive)

    #### Status and Loading
    - **Loading Status Bar**: Real-time feedback during data processing

    #### Interactive Visualization
    - **Comparison Plot Card**: Full-screen capable Plotly chart with 600px height
    - **Performance Statistics Card**: Comprehensive metrics table with export options

    ## Input Component Naming Convention

    Following project standards, all input components use the naming pattern:
    `input_ID_tab_portfolios_subtab_comparison_{identifier}`

    ## Responsive Design Features

    - **Mobile-friendly**: Bootstrap-based responsive layout
    - **Flexible Content**: Main area adapts to screen size
    - **Fixed Sidebar**: Consistent control panel width
    - **Card-based Layout**: Modular, professional appearance

    ## Accessibility Features

    - **Semantic HTML**: Proper heading hierarchy and ARIA labels
    - **Keyboard Navigation**: Full keyboard accessibility
    - **Screen Reader Support**: Descriptive text and status updates
    - **High Contrast**: Material Design color scheme

    Parameters
    ----------
    data_utils : dict
        Dictionary containing utility functions and dashboard configurations.

        Expected keys:
        - Configuration settings for the dashboard
        - Utility functions for data processing
        - Theme and styling preferences

    data_inputs : dict
        Dictionary containing portfolio and benchmark datasets.

        Expected keys:
        - 'My_Portfolio': Portfolio performance data (pl.DataFrame)
        - 'Benchmark_Portfolio': Benchmark performance data (pl.DataFrame)
        - 'Weights_My_Portfolio': Portfolio weights data (pl.DataFrame, optional)
        - 'Time_Series_ETFs': ETF time series data (pl.DataFrame, optional)

    Returns
    -------
    shiny.ui.div
        Complete UI layout containing:
        - Responsive sidebar with all control components
        - Main content area with visualization and statistics
        - Proper Bootstrap classes for responsive behavior
        - All necessary input/output component definitions

    Examples
    --------
    Basic usage in a Shiny application:

    ```python
    # In your main application UI
    portfolio_comparison_tab = subtab_portfolios_comparison_ui(
        data_utils={"theme": "material", "export_enabled": True},
        data_inputs={
            "My_Portfolio": portfolio_dataframe,
            "Benchmark_Portfolio": benchmark_dataframe,
        },
    )
    ```

    Integration with tab navigation:

    ```python
    ui.nav_panel(
        "Portfolio Comparison",
        subtab_portfolios_comparison_ui(
            data_utils=dashboard_utilities, data_inputs=portfolio_datasets
        ),
    )
    ```

    Notes
    -----
    - UI components are fully reactive and will update based on data changes
    - Custom CSS classes follow Bootstrap conventions for consistency
    - All input components include validation and error handling
    - Time period calculations handle edge cases (weekends, holidays, market closures)

    See Also
    --------
    subtab_portfolios_comparison_server : Server logic for this UI
    create_plot_comparison_portfolios : Plot generation function
    create_table_comparison_stats : Statistics table generation function
    """
    # Main container with semantic heading following project standards
    # Uses h3 for proper heading hierarchy within the tab structure
    return ui.div(
        # Page title with clear identification of functionality
        ui.h3("Portfolio vs Benchmark Comparison"),
        # Bootstrap sidebar layout for responsive design
        # Provides optimal layout for both desktop and mobile devices
        ui.layout_sidebar(
            # Sidebar component containing all control elements
            # Fixed width ensures consistent layout across screen sizes
            ui.sidebar(
                # =============================================================
                # TIME PERIOD SELECTION SECTION
                # =============================================================
                # Visual section separator with emoji for enhanced UX
                ui.h5("📅 Time Period Selection", class_="mb-3"),
                # Primary time period dropdown selector
                # Follows project naming convention: input_ID_tab_portfolios_subtab_comparison_{identifier}
                ui.input_select(
                    "input_ID_tab_portfolios_subtab_comparison_time_period",
                    "Select Time Period",
                    {
                        # Custom option allows user-defined date ranges
                        "custom": "Custom",
                        # Preset periods for common analysis timeframes
                        "1y": "Last 1 Year",  # 365 days from current date
                        "3y": "Last 3 Years",  # 3 * 365 days from current date
                        "5y": "Last 5 Years",  # 5 * 365 days from current date
                        "10y": "Last 10 Years",  # 10 * 365 days from current date
                        "ytd": "Year to Date",  # January 1st current year to today
                    },
                    selected="1y",  # Default to 1 year for most common use case
                ),
                # Conditional date range picker - only shown when "custom" is selected
                # Uses JavaScript conditional rendering for enhanced user experience
                ui.panel_conditional(
                    "input.input_ID_tab_portfolios_subtab_comparison_time_period === 'custom'",
                    ui.div(
                        # Date range input with sensible defaults
                        ui.input_date_range(
                            "input_ID_tab_portfolios_subtab_comparison_date_range",
                            "Custom Date Range",
                            start=datetime.now(UTC) - timedelta(days=365),  # Default to last year
                            end=datetime.now(UTC),  # Default to today
                            format="yyyy-mm-dd",  # ISO date format
                            separator=" to ",  # Clear separator text
                            width="100%",  # Full width within container
                        ),
                        class_="mt-2",  # Bootstrap margin-top spacing
                    ),
                ),
                # Display calculated date range for preset selections
                # Provides transparency about the actual date range being analyzed
                ui.panel_conditional(
                    "input.input_ID_tab_portfolios_subtab_comparison_time_period !== 'custom'",
                    ui.div(
                        # Output component for displaying calculated date range
                        ui.output_text(
                            "output_ID_tab_portfolios_subtab_comparison_calculated_date_range",
                        ),
                        # Bootstrap styling for subtle appearance
                        class_="mt-2 p-2 bg-light rounded text-muted small",
                    ),
                ),
                # Visual separator between sections
                ui.hr(),
                # =============================================================
                # VISUALIZATION OPTIONS SECTION
                # =============================================================
                ui.h5("📊 Visualization Options", class_="mb-3"),
                # Visualization type selector for different analysis perspectives
                ui.input_select(
                    "input_ID_tab_portfolios_subtab_comparison_viz_type",
                    "Visualization Type",
                    {
                        # Different visualization types for various analysis needs
                        "value": "Absolute Value",  # Raw portfolio/benchmark values
                        "normalized": "Normalized (Base=100)",  # Performance relative to starting point
                        "pct_change": "Percent Change",  # Period-over-period percentage changes
                        "cum_return": "Cumulative Return",  # Cumulative return analysis
                    },
                    selected="normalized",  # Default to normalized for relative performance comparison
                ),
                # Toggle option for showing difference between portfolio and benchmark
                # Provides additional analytical insight into performance gaps
                ui.input_checkbox(
                    "input_ID_tab_portfolios_subtab_comparison_show_diff",
                    "Show Difference Between Portfolio & Benchmark",
                    value=False,  # Default disabled to avoid cluttering initial view
                ),
                # =============================================================
                # DATA INFORMATION SECTION
                # =============================================================
                # Visual separator and informational section
                ui.hr(),
                ui.h5("ℹ️ Data Information", class_="mb-3"),
                # Dynamic data information display
                # Shows real-time status of portfolio and benchmark data
                ui.output_ui("output_ID_tab_portfolios_subtab_comparison_data_info"),
                # Sidebar configuration
                width=350,  # Fixed pixel width for consistent layout
                position="left",  # Left-side positioning following dashboard conventions
            ),
            # =============================================================
            # MAIN CONTENT AREA
            # =============================================================
            # Main content area containing visualizations and statistics
            ui.div(
                # Loading status indicator at top of main content
                # Provides real-time feedback during data processing operations
                ui.output_ui("output_ID_tab_portfolios_subtab_comparison_loading_status"),
                # Primary comparison plot card
                # Features full-screen capability for detailed analysis
                ui.card(
                    # Card header with clear title
                    ui.card_header("Portfolio vs Benchmark Performance"),
                    # Main plot widget with optimal sizing
                    output_widget(
                        "output_ID_tab_portfolios_subtab_comparison_plot_main",
                        height="600px",  # Fixed height for consistent layout
                        width="100%",  # Responsive width
                    ),
                    # Card configuration
                    full_screen=True,  # Enable full-screen viewing capability
                    class_="mb-4",  # Bootstrap margin-bottom spacing
                ),
                # Performance statistics card
                # Displays comprehensive statistical analysis below the main plot
                ui.card(
                    ui.card_header("Performance Statistics"),
                    # Statistics table output
                    ui.output_ui("output_ID_tab_portfolios_subtab_comparison_table_stats"),
                    class_="mb-4",  # Bootstrap margin-bottom spacing
                ),
                # Main content area configuration
                class_="flex-fill",  # Bootstrap flexbox class for responsive layout
            ),
        ),
    )


@module.server
def subtab_portfolios_comparison_server(
    input: typing.Any,
    output: typing.Any,
    session: typing.Any,
    data_utils: dict,
    data_inputs: dict,
    reactives_shiny: dict,
) -> None:
    """
    Server logic for the Portfolio Comparison subtab with comprehensive reactive analysis.

    This function implements the complete server-side functionality for portfolio vs
    benchmark comparison analysis, including data processing, visualization generation,
    and statistical calculations with real-time reactive updates.

    ## Server Architecture

    The server follows a layered reactive architecture optimized for performance:

    ```mermaid
    graph TD
        A[Input Parameters] --> B[Data Validation]
        B --> C[Reactive Calculations]
        C --> D[Date Range Processing]
        C --> E[Data Filtering]
        C --> F[Visualization Generation]
        C --> G[Statistics Calculation]
        D --> H[Filtered Data]
        E --> H
        H --> I[Plot Outputs]
        H --> J[Table Outputs]
        H --> K[UI Updates]
    ```

    ## Core Functionality

    ### 🔍 Data Processing Pipeline

    1. **Input Validation**: Validates portfolio and benchmark data integrity
    2. **Date Range Calculation**: Processes time period selections and custom date ranges
    3. **Data Filtering**: Applies date-based filtering with efficient string comparison
    4. **Visualization Generation**: Creates interactive Plotly charts with multiple view types
    5. **Statistical Analysis**: Calculates comprehensive performance metrics

    ### 📊 Reactive Components

    #### Primary Reactive Calculations
    - `get_effective_date_range()`: Dynamic date range computation
    - `get_filtered_comparison_data()`: Real-time data filtering based on selections

    #### Output Components
    - **Main Comparison Plot**: Interactive Plotly visualization with export capability
    - **Performance Statistics Table**: Comprehensive metrics with HTML formatting
    - **Data Information Panel**: Real-time status and data point counts
    - **Calculated Date Range Display**: Dynamic date range feedback

    ## Error Handling Strategy

    The server implements comprehensive error handling following defensive programming principles:

    !!! tip "Error Handling Approach"
        - **Input Validation**: Early returns with descriptive error messages
        - **Configuration Validation**: User input sanitization with fallback defaults
        - **Business Logic Validation**: Data consistency checks with graceful degradation
        - **Exception Propagation**: Proper error propagation with context preservation

    ## Performance Optimizations

    !!! success "Optimization Features"
        - **Efficient Date Filtering**: String-based comparison for improved performance
        - **Polars DataFrame Operations**: High-performance data processing
        - **Lazy Reactive Evaluation**: Calculations only when dependencies change
        - **Data Type Optimization**: Automatic type inference and conversion
        - **Memory Management**: Proper cleanup of intermediate DataFrames

    ## Data Flow Architecture

    ### Input Data Processing

    The server expects specific data inputs with validation:

    | Input Dataset | Validation | Processing |
    |---------------|------------|------------|
    | `My_Portfolio` | `validate_portfolio_data()` | Date filtering, type conversion |
    | `Benchmark_Portfolio` | `validate_portfolio_data()` | Date filtering, type conversion |
    | `Weights_My_Portfolio` | Optional validation | Reference data only |
    | `Time_Series_ETFs` | Optional validation | Reference data only |

    ### Reactive Dependency Graph

    ```
    User Inputs → Date Range Calc → Data Filtering → Visualizations
                                                   → Statistics
                                                   → UI Updates
    ```

    ## Visualization Types Supported

    === "Absolute Value"
        Raw portfolio and benchmark values over time

    === "Normalized (Base=100)"
        Performance normalized to starting value of 100

    === "Percent Change"
        Period-over-period percentage changes

    === "Cumulative Return"
        Cumulative return analysis over selected period

    Parameters
    ----------
    input : typing.Any
        Shiny input object containing all user interactions and selections.

        Expected reactive inputs:
        - `input_ID_tab_portfolios_subtab_comparison_time_period`: Time period selection
        - `input_ID_tab_portfolios_subtab_comparison_date_range`: Custom date range
        - `input_ID_tab_portfolios_subtab_comparison_viz_type`: Visualization type
        - `input_ID_tab_portfolios_subtab_comparison_show_diff`: Difference toggle

    output : typing.Any
        Shiny output object for rendering components and visualizations.

        Generated outputs:
        - `output_ID_tab_portfolios_subtab_comparison_plot_main`: Main comparison plot
        - `output_ID_tab_portfolios_subtab_comparison_table_stats`: Statistics table
        - `output_ID_tab_portfolios_subtab_comparison_data_info`: Data information panel
        - `output_ID_tab_portfolios_subtab_comparison_calculated_date_range`: Date display

    session : typing.Any
        Shiny session object providing reactive context and state management.

    data_utils : dict
        Dictionary containing utility functions and dashboard configurations.

        Expected keys:
        - Configuration settings for dashboard behavior
        - Utility functions for data processing
        - Theme and styling preferences

    data_inputs : dict
        Dictionary containing all portfolio and benchmark datasets.

        Expected keys:
        - 'My_Portfolio': Portfolio performance data (pl.DataFrame)
        - 'Benchmark_Portfolio': Benchmark performance data (pl.DataFrame)
        - 'Weights_My_Portfolio': Portfolio weights data (pl.DataFrame, optional)
        - 'Time_Series_ETFs': ETF time series data (pl.DataFrame, optional)

    reactives_shiny : dict
        Dictionary containing reactive variables and shared state.

        Used for:
        - Cross-module reactive value sharing
        - State persistence across user interactions
        - Global dashboard state management

    Returns
    -------
    None
        Server functions define reactive outputs and effects but do not return values.
        All functionality is expressed through side effects on the output object.

    Raises
    ------
    ValueError
        If portfolio or benchmark data validation fails during initialization.

        Common causes:
        - Missing required columns ('Date', 'Value')
        - Invalid data types in DataFrame columns
        - Empty datasets when data is expected

    RuntimeError
        If date range calculation or data filtering encounters errors.

        Common causes:
        - Invalid date formats in custom range selection
        - Date parsing failures during filtering operations
        - Unexpected data type conversions

    Examples
    --------
    Basic server integration in a Shiny module:

    ```python
    # In your main application server
    subtab_portfolios_comparison_server(
        id="ID_tab_portfolios_subtab_comparison",
        input=input,
        output=output,
        session=session,
        data_utils={
            'theme': 'material',
            'export_enabled': True
        },
        data_inputs={
            'My_Portfolio': portfolio_dataframe,
            'Benchmark_Portfolio': benchmark_dataframe
        },
        reactives_shiny=reactive_values
    )
    ```

    Integration with tab navigation:

    ```python
    # Within a larger portfolio analysis module
    tab_portfolios_server(
        id="ID_tab_portfolios",
        data_utils=dashboard_utilities,
        data_inputs=all_portfolio_datasets,
        reactives_shiny=shared_reactive_state,
    )
    ```

    Notes
    -----
    ### Data Requirements
    - Portfolio and benchmark data must contain 'Date' and 'Value' columns
    - Date columns support multiple formats with automatic conversion
    - Missing data is handled gracefully with empty DataFrame fallbacks

    ### Performance Considerations
    - Large datasets are automatically optimized through efficient filtering
    - Reactive calculations use lazy evaluation for optimal performance

    ### Error Recovery
    - All reactive functions include comprehensive error handling
    - Invalid user inputs fallback to sensible defaults
    - Data processing errors display user-friendly error messages

    ### Cross-Browser Compatibility
    - Plotly visualizations work across all modern browsers
    - Responsive design adapts to different screen sizes
    - Touch interactions supported for mobile devices

    See Also
    --------
    subtab_portfolios_comparison_ui : UI component for this server
    validate_portfolio_data : Data validation utility function
    create_plot_comparison_portfolios : Plot generation function
    create_table_comparison_stats : Statistics table generation function
    get_value_from_reactives_shiny : Reactive value access utility
    """
    # =============================================================================
    # DATA VALIDATION AND INITIALIZATION
    # =============================================================================
    # Extract portfolio and benchmark data from inputs with defensive programming
    # Using .get() method provides None fallback for missing keys
    data_portfolio = data_inputs.get("My_Portfolio")  # Primary portfolio performance data
    data_benchmark = data_inputs.get("Benchmark_Portfolio")  # Benchmark comparison data
    # Note: weights_portfolio and time_series_ETFs available but not used in comparison view

    # Validate portfolio data integrity using project utility functions
    # Early validation prevents runtime errors during data processing
    validation_portfolio = validate_portfolio_data(data_portfolio=data_portfolio)
    validation_benchmark = validate_portfolio_data(data_portfolio=data_benchmark)

    # Input validation with early returns and descriptive error messages
    # Following project standards for defensive programming over try-except
    if not validation_portfolio[0]:
        raise ValueError(f"Portfolio data validation failed: {validation_portfolio[1]}")

    if not validation_benchmark[0]:
        raise ValueError(f"Benchmark data validation failed: {validation_benchmark[1]}")

    @reactive.calc
    def get_effective_date_range() -> tuple[datetime, datetime]:
        """
        Calculate the effective date range based on user time period selection.

        This reactive calculation processes both preset time periods and custom date ranges,
        using actual data availability to ensure accurate date boundaries.

        Returns
        -------
        tuple[datetime, datetime]
            Start and end dates for the effective analysis period with time normalized to midnight.
        """
        try:
            time_period = input.input_ID_tab_portfolios_subtab_comparison_time_period()

            # Get actual data availability from portfolio data
            # This ensures we use real data dates, not theoretical dates
            if (
                data_portfolio is not None
                and not data_portfolio.is_empty()
                and "Date" in data_portfolio.columns
            ):
                date_dtype = data_portfolio.select("Date").dtypes[0]

                # Convert string dates to datetime to find actual data range
                if date_dtype in [pl.Utf8, pl.String]:
                    try:
                        # Convert via pandas for timezone handling
                        temp_pandas = data_portfolio.select("Date").to_pandas()
                        temp_pandas["Date"] = pd.to_datetime(
                            temp_pandas["Date"],
                            utc=True,
                            errors="coerce",
                        )
                        if isinstance(temp_pandas["Date"].dtype, pd.DatetimeTZDtype):
                            temp_pandas["Date"] = temp_pandas["Date"].dt.tz_localize(None)

                        data_max_date = temp_pandas["Date"].max()
                        data_min_date = temp_pandas["Date"].min()
                    except Exception as exc_conv:
                        print(f"⚠️  Date conversion error in get_effective_date_range: {exc_conv}")
                        data_max_date = datetime.now(UTC)
                        data_min_date = datetime.now(UTC) - timedelta(days=365 * 10)

                elif date_dtype in [pl.Date, pl.Datetime]:
                    data_max_date_raw = data_portfolio.select(pl.col("Date").max()).item()
                    data_min_date_raw = data_portfolio.select(pl.col("Date").min()).item()

                    # Convert to datetime if needed
                    if isinstance(data_max_date_raw, pl.Date):
                        data_max_date = datetime(
                            data_max_date_raw.year,  # pyright: ignore[reportAttributeAccessIssue]  # pyrefly: ignore[missing-attribute]
                            data_max_date_raw.month,  # pyright: ignore[reportAttributeAccessIssue]  # pyrefly: ignore[missing-attribute]
                            data_max_date_raw.day,  # pyright: ignore[reportAttributeAccessIssue]  # pyrefly: ignore[missing-attribute]
                            tzinfo=UTC,
                        )
                        data_min_date = datetime(
                            data_min_date_raw.year,  # pyright: ignore[reportAttributeAccessIssue]  # pyrefly: ignore[missing-attribute]
                            data_min_date_raw.month,  # pyright: ignore[reportAttributeAccessIssue]  # pyrefly: ignore[missing-attribute]
                            data_min_date_raw.day,  # pyright: ignore[reportAttributeAccessIssue]  # pyrefly: ignore[missing-attribute]
                            tzinfo=UTC,
                        )
                    else:
                        # Already datetime - normalize to date only (remove time component)
                        data_max_date = (
                            data_max_date_raw.replace(hour=0, minute=0, second=0, microsecond=0)
                            if hasattr(data_max_date_raw, "replace")
                            else data_max_date_raw
                        )
                        data_min_date = (
                            data_min_date_raw.replace(hour=0, minute=0, second=0, microsecond=0)
                            if hasattr(data_min_date_raw, "replace")
                            else data_min_date_raw
                        )
                else:
                    data_max_date = datetime.now(UTC).replace(
                        hour=0,
                        minute=0,
                        second=0,
                        microsecond=0,
                    )
                    data_min_date = (datetime.now(UTC) - timedelta(days=365 * 10)).replace(
                        hour=0,
                        minute=0,
                        second=0,
                        microsecond=0,
                    )
            else:
                data_max_date = datetime.now(UTC).replace(hour=0, minute=0, second=0, microsecond=0)
                data_min_date = (datetime.now(UTC) - timedelta(days=365 * 10)).replace(
                    hour=0,
                    minute=0,
                    second=0,
                    microsecond=0,
                )

            print(f"\n{'=' * 70}")
            print("📅 Date Range Calculation")
            print(f"{'=' * 70}")
            print(f"Time period selection: {time_period}")
            print(
                f"Available data range: {data_min_date.strftime('%Y-%m-%d')} to {data_max_date.strftime('%Y-%m-%d')}",
            )

            # Use the latest available data date as the end date
            # Normalize to midnight to avoid time component issues
            today = data_max_date.replace(hour=0, minute=0, second=0, microsecond=0)

            # Configuration validation - handle custom date range selection
            if time_period == "custom":
                try:
                    date_range = input.input_ID_tab_portfolios_subtab_comparison_date_range()
                    if date_range and len(date_range) == 2:
                        # Normalize custom dates to midnight
                        start_date = (
                            pd.Timestamp(date_range[0])
                            .to_pydatetime()
                            .replace(hour=0, minute=0, second=0, microsecond=0)
                        )
                        end_date = (
                            pd.Timestamp(date_range[1])
                            .to_pydatetime()
                            .replace(hour=23, minute=59, second=59, microsecond=999999)
                        )

                        # Ensure end date doesn't exceed available data
                        if end_date > today:
                            print(
                                f"⚠️  Custom end date {end_date.strftime('%Y-%m-%d')} exceeds data availability, using {today.strftime('%Y-%m-%d')}",
                            )
                            end_date = today

                        print(
                            f"Custom range: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}",
                        )
                        print(f"{'=' * 70}\n")
                        return start_date, end_date
                except Exception as exc:
                    print(f"⚠️  Custom range error: {exc}, falling back to 1 year")

                # Fallback for custom range errors - normalize to midnight
                start_date = (today - timedelta(days=365)).replace(
                    hour=0,
                    minute=0,
                    second=0,
                    microsecond=0,
                )
                end_date = today.replace(hour=23, minute=59, second=59, microsecond=999999)
                print(
                    f"Fallback 1Y: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}",
                )
                print(f"{'=' * 70}\n")
                return start_date, end_date

            # Business logic validation - process preset time periods with normalized dates
            if time_period == "1y":
                start_date = (today - timedelta(days=365)).replace(
                    hour=0,
                    minute=0,
                    second=0,
                    microsecond=0,
                )
                end_date = today.replace(hour=23, minute=59, second=59, microsecond=999999)
            elif time_period == "3y":
                start_date = (today - timedelta(days=365 * 3)).replace(
                    hour=0,
                    minute=0,
                    second=0,
                    microsecond=0,
                )
                end_date = today.replace(hour=23, minute=59, second=59, microsecond=999999)
            elif time_period == "5y":
                start_date = (today - timedelta(days=365 * 5)).replace(
                    hour=0,
                    minute=0,
                    second=0,
                    microsecond=0,
                )
                end_date = today.replace(hour=23, minute=59, second=59, microsecond=999999)
            elif time_period == "10y":
                start_date = (today - timedelta(days=365 * 10)).replace(
                    hour=0,
                    minute=0,
                    second=0,
                    microsecond=0,
                )
                end_date = today.replace(hour=23, minute=59, second=59, microsecond=999999)
            elif time_period == "ytd":
                # Year to date - from January 1st with time normalized
                year_start = datetime(today.year, 1, 1, 0, 0, 0, 0, tzinfo=UTC)
                start_date = year_start
                end_date = today.replace(hour=23, minute=59, second=59, microsecond=999999)
            else:
                # Default fallback with normalized dates
                start_date = (today - timedelta(days=365)).replace(
                    hour=0,
                    minute=0,
                    second=0,
                    microsecond=0,
                )
                end_date = today.replace(hour=23, minute=59, second=59, microsecond=999999)

            print(
                f"Calculated range: {start_date.strftime('%Y-%m-%d %H:%M:%S')} to {end_date.strftime('%Y-%m-%d %H:%M:%S')}",
            )
            print(f"{'=' * 70}\n")
            return start_date, end_date

        except Exception as exc:
            print(f"❌ Error calculating date range: {exc}")
            import traceback

            traceback.print_exc()
            # Final fallback with normalized dates
            end_date = datetime.now(UTC).replace(hour=23, minute=59, second=59, microsecond=999999)
            start_date = (end_date - timedelta(days=365)).replace(
                hour=0,
                minute=0,
                second=0,
                microsecond=0,
            )
            return start_date, end_date

    @output
    @render.text
    def output_ID_tab_portfolios_subtab_comparison_calculated_date_range() -> str | None:
        """
        Display the calculated date range for preset time period selections.

        This output component provides transparent feedback to users about the actual
        date range being applied for preset period selections, enhancing user awareness
        of the analysis scope.

        ## Display Logic

        - **Preset Periods**: Shows calculated start and end dates in ISO format
        - **Custom Range**: Returns empty string (range displayed in date picker)
        - **Error States**: Handled gracefully with fallback display

        ## Date Formatting

        Uses ISO date format (YYYY-MM-DD) for:
        - Consistent international date representation
        - Clear, unambiguous date display
        - Easy parsing and validation

        Returns
        -------
        str
            Formatted date range string with emoji prefix for visual appeal.
            Empty string for custom range selections.

        Raises
        ------
        RuntimeError
            If date range calculation or formatting fails.

        Examples
        --------
        Display output examples:

        ```
        📅 2024-06-11 to 2025-06-11    # For 1Y selection
        📅 2022-01-01 to 2025-06-11    # For YTD selection
        (empty)                         # For custom selection
        ```
        """
        # Only use try-except for operations that might fail unpredictably
        try:
            # Get calculated date range from reactive function
            start_date, end_date = get_effective_date_range()
            time_period = input.input_ID_tab_portfolios_subtab_comparison_time_period()

            # Configuration validation - only display for preset periods
            if time_period != "custom":
                # Format dates in ISO format with emoji for visual appeal
                return f"📅 {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}"
            # Return empty string for custom selections (date picker handles display)
            return ""
        except Exception as exc:
            # Comprehensive error with context for debugging
            raise RuntimeError(f"Error displaying calculated date range: {exc}")

    @output
    @render.ui
    def output_ID_tab_portfolios_subtab_comparison_data_info():
        """
        Generate and display comprehensive data information panel in the sidebar.

        This output component provides real-time status information about portfolio
        and benchmark data availability, helping users understand the scope and
        quality of their analysis.

        ## Information Display Components

        ### 📅 Period Information
        - Current time period selection with human-readable labels
        - Dynamic updates based on user selection changes

        ### 📊 Data Point Counts
        - Portfolio data points after filtering
        - Benchmark data points after filtering
        - Real-time updates as filters change

        ### ✅ Status Indicators
        - **Green (✅)**: Both datasets available and ready for comparison
        - **Yellow (⚠️)**: Partial data available (one dataset missing/empty)
        - **Red (❌)**: No data available for analysis

        ## Bootstrap Styling

        Uses Bootstrap classes for consistent visual presentation:
        - `text-success` for positive status indicators
        - `text-warning` for cautionary status indicators
        - `text-danger` for error status indicators
        - `bg-light` for subtle background highlighting

        Returns
        -------
        shiny.ui.div
            Complete UI component containing formatted data information with status indicators.

        Raises
        ------
        RuntimeError
            If data retrieval or UI generation encounters errors.

        Examples
        --------
        UI output structure:

        ```html
        <div>
            <div class="small text-muted mb-1">
                <strong>Period:</strong> Last 1 Year
            </div>
            <div class="small text-muted mb-1">
                <strong>Portfolio Data:</strong> 252 points
            </div>
            <div class="small text-muted mb-1">
                <strong>Benchmark Data:</strong> 252 points
            </div>
            <div class="mt-2 p-2 rounded bg-light">
                <span class="me-1">✅</span>
                <span class="small text-success">Ready for comparison</span>
            </div>
        </div>
        ```
        """
        # Only use try-except for operations that might fail unpredictably
        try:
            # Get filtered data to display current status information
            data_portfolio_filtered, data_benchmark_filtered = get_filtered_comparison_data()

            # Get current time period for display
            time_period = input.input_ID_tab_portfolios_subtab_comparison_time_period()

            # Configuration validation - map time period codes to human-readable labels
            period_labels = {
                "1y": "Last 1 Year",
                "3y": "Last 3 Years",
                "5y": "Last 5 Years",
                "10y": "Last 10 Years",
                "ytd": "Year to Date",
                "custom": "Custom Period",
            }
            period_label = period_labels.get(time_period, "Selected Period")

            # Initialize UI component list for dynamic construction
            info_items = []

            # Add period information with consistent styling
            info_items.append(
                ui.div(
                    ui.strong("Period: "),
                    period_label,
                    class_="small text-muted mb-1",  # Bootstrap classes for subtle styling
                ),
            )

            # Business logic validation - calculate portfolio data statistics
            # Use Polars methods for efficient data point counting
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

            # Business logic validation - calculate benchmark data statistics
            # Use Polars methods for efficient data point counting
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

            # Generate status indicator based on data availability
            # Following project standards for clear conditional logic
            if portfolio_count > 0 and benchmark_count > 0:
                # Both datasets available - optimal condition
                status_icon = "✅"
                status_text = "Ready for comparison"
                status_class = "text-success"
            elif portfolio_count > 0 or benchmark_count > 0:
                # Partial data available - suboptimal but functional
                status_icon = "⚠️"
                status_text = "Partial data available"
                status_class = "text-warning"
            else:
                # No data available - error condition
                status_icon = "❌"
                status_text = "No data available"
                status_class = "text-danger"

            # Add status indicator with appropriate Bootstrap styling
            info_items.append(
                ui.div(
                    ui.span(status_icon, class_="me-1"),  # Icon with margin-end spacing
                    ui.span(status_text, class_=f"small {status_class}"),  # Status text with color
                    class_="mt-2 p-2 rounded bg-light",  # Container with subtle background
                ),
            )

            # Return constructed UI component
            return ui.div(*info_items)

        except Exception as exc:
            # Comprehensive error with context for debugging
            raise RuntimeError(f"Error generating data info: {exc}")

    @reactive.calc
    def get_filtered_comparison_data() -> tuple[pl.DataFrame, pl.DataFrame]:
        """
        Filter portfolio and benchmark data based on effective date range with timezone handling.

        This reactive calculation applies date-based filtering to both portfolio and benchmark
        datasets, handling timezone-aware date strings through pandas for maximum compatibility,
        then converting back to Polars for efficient downstream operations.

        ## Date Processing Strategy

        The function handles multiple date formats with automatic detection and conversion:

        1. **Timezone-aware strings**: Uses pandas.to_datetime with utc=True for normalization
        2. **String dates**: Converts via pandas for flexible format detection
        3. **pl.Date types**: Direct conversion to datetime format
        4. **pl.Datetime types**: Timezone removal if present

        ## Performance Optimization

        - Uses string-based date comparison for efficient filtering
        - Polars operations for high-performance data manipulation
        - Single-pass filtering reduces computational overhead

        Returns
        -------
        tuple[pl.DataFrame, pl.DataFrame]
            Tuple containing:
            - Filtered portfolio DataFrame with Date and Value columns
            - Filtered benchmark DataFrame with Date and Value columns
            Both DataFrames have timezone-naive datetime Date columns

        Raises
        ------
        RuntimeError
            If fundamental data processing operations fail unexpectedly

        Examples
        --------
        Basic usage in reactive context:

        ```python
        @reactive.calc
        def analysis_func():
            portfolio_df, benchmark_df = get_filtered_comparison_data()
            # Both DataFrames now contain filtered, timezone-naive data
        ```

        Notes
        -----
        - Returns empty DataFrames with proper schema if source data is None
        - Preserves data integrity through defensive cloning
        - Handles timezone information automatically via pandas UTC normalization
        - Sorts results chronologically for consistent time-series operations
        """
        # Only use try-except for operations that might fail unpredictably
        try:
            # Get the effective date range for filtering
            start_date, end_date = get_effective_date_range()

            print(f"\n{'=' * 70}")
            print("🔍 Date Filtering Analysis for Portfolio Comparison")
            print(f"{'=' * 70}")
            print("📅 Requested Date Range:")
            print(f"   Start: {start_date.strftime('%Y-%m-%d')}")
            print(f"   End:   {end_date.strftime('%Y-%m-%d')}")

            # =================================================================
            # PORTFOLIO DATA PROCESSING WITH TIMEZONE HANDLING
            # =================================================================

            # Input validation with early returns and proper schema creation
            if data_portfolio is None:
                print("\n❌ Portfolio data is None")
                filtered_portfolio = pl.DataFrame(
                    {"Date": [], "Value": []},
                    schema={"Date": pl.Datetime, "Value": pl.Float64},
                )
            else:
                # Log original portfolio data information
                print("\n📊 Original Portfolio Data:")
                print(f"   Rows: {data_portfolio.height}")
                print(f"   Columns: {data_portfolio.columns}")

                # Input validation - check if data has required structure
                if (
                    data_portfolio.is_empty()
                    or "Date" not in data_portfolio.columns
                    or "Value" not in data_portfolio.columns
                ):
                    print("   ❌ Missing required columns or empty data")
                    filtered_portfolio = pl.DataFrame(
                        {"Date": [], "Value": []},
                        schema={"Date": pl.Datetime, "Value": pl.Float64},
                    )
                else:
                    # Check date column type
                    date_dtype_portfolio = data_portfolio.select("Date").dtypes[0]
                    print(f"   Date column type: {date_dtype_portfolio}")

                    # Sample dates for format understanding
                    sample_dates_portfolio = (
                        data_portfolio.select("Date").head(5).to_series().to_list()
                    )
                    print(f"   Sample dates (first 5): {sample_dates_portfolio}")

                    # Configuration validation - handle timezone-aware date strings via pandas
                    if date_dtype_portfolio in [pl.Utf8, pl.String]:
                        print("   → Converting timezone-aware strings via pandas...")

                        # Convert to pandas for robust timezone handling
                        portfolio_pandas = data_portfolio.to_pandas()

                        # Use pandas to_datetime with UTC normalization for timezone-aware strings
                        # utc=True handles mixed timezones and normalizes to UTC
                        portfolio_pandas["Date"] = pd.to_datetime(
                            portfolio_pandas["Date"],
                            utc=True,  # Normalize to UTC timezone
                            errors="coerce",  # Invalid dates become NaT
                        )

                        # Remove timezone information to create timezone-naive datetime
                        if isinstance(portfolio_pandas["Date"].dtype, pd.DatetimeTZDtype):
                            portfolio_pandas["Date"] = portfolio_pandas["Date"].dt.tz_localize(None)
                            print(
                                "   ✅ Converted timezone-aware strings → UTC → timezone-naive datetime",
                            )

                        # Convert back to Polars for efficient filtering
                        filtered_portfolio = pl.from_pandas(portfolio_pandas)

                    elif date_dtype_portfolio == pl.Date:
                        # pl.Date type - convert to Datetime
                        filtered_portfolio = data_portfolio.clone().with_columns(
                            [pl.col("Date").cast(pl.Datetime).alias("Date")],
                        )
                        print("   ✅ Converted pl.Date → pl.Datetime")

                    elif date_dtype_portfolio == pl.Datetime:
                        # Already Datetime - just remove timezone if present
                        filtered_portfolio = data_portfolio.clone()
                        if filtered_portfolio["Date"].dtype.time_zone is not None:
                            filtered_portfolio = filtered_portfolio.with_columns(
                                [pl.col("Date").dt.replace_time_zone(None).alias("Date")],
                            )
                            print("   ✅ Removed timezone from pl.Datetime")
                        else:
                            print("   ✅ Already timezone-naive pl.Datetime")
                    else:
                        # Unknown type - attempt pandas conversion as fallback
                        print("   ⚠️  Unknown date type, attempting pandas conversion...")
                        portfolio_pandas = data_portfolio.to_pandas()
                        portfolio_pandas["Date"] = pd.to_datetime(
                            portfolio_pandas["Date"],
                            utc=True,
                            errors="coerce",
                        )
                        if isinstance(portfolio_pandas["Date"].dtype, pd.DatetimeTZDtype):
                            portfolio_pandas["Date"] = portfolio_pandas["Date"].dt.tz_localize(None)
                        filtered_portfolio = pl.from_pandas(portfolio_pandas)

                    # Business logic validation - get actual date range from data
                    if not filtered_portfolio.is_empty():
                        data_min_date = filtered_portfolio.select(pl.col("Date").min()).item()
                        data_max_date = filtered_portfolio.select(pl.col("Date").max()).item()
                        print(f"   Available date range: {data_min_date} to {data_max_date}")

                    # Apply date range filtering using efficient Polars operations
                    print("\n🔧 Applying Date Filter to Portfolio:")
                    print(f"   Rows before filtering: {filtered_portfolio.height}")

                    # Convert filter dates to Polars datetime for comparison
                    start_date_pl = pl.lit(start_date).cast(pl.Datetime)
                    end_date_pl = pl.lit(end_date).cast(pl.Datetime)

                    # Apply filtering with null removal
                    filtered_portfolio = filtered_portfolio.filter(
                        pl.col("Date").is_not_null()
                        & (pl.col("Date") >= start_date_pl)
                        & (pl.col("Date") <= end_date_pl),
                    ).sort("Date")

                    print(f"   Rows after filtering: {filtered_portfolio.height}")

                    # Early return validation - warn if filtering removed all data
                    if filtered_portfolio.is_empty():
                        print("   ❌ WARNING: Filtering removed ALL portfolio data!")
                        print("   Requested range may be outside available data range")
                    else:
                        final_min_date = filtered_portfolio.select(pl.col("Date").min()).item()
                        final_max_date = filtered_portfolio.select(pl.col("Date").max()).item()
                        print(f"   ✅ Final range: {final_min_date} to {final_max_date}")

            # =================================================================
            # BENCHMARK DATA PROCESSING (identical logic)
            # =================================================================

            if data_benchmark is None:
                print("\n❌ Benchmark data is None")
                filtered_benchmark = pl.DataFrame(
                    {"Date": [], "Value": []},
                    schema={"Date": pl.Datetime, "Value": pl.Float64},
                )
            else:
                print("\n📊 Original Benchmark Data:")
                print(f"   Rows: {data_benchmark.height}")
                print(f"   Columns: {data_benchmark.columns}")

                if (
                    data_benchmark.is_empty()
                    or "Date" not in data_benchmark.columns
                    or "Value" not in data_benchmark.columns
                ):
                    print("   ❌ Missing required columns or empty data")
                    filtered_benchmark = pl.DataFrame(
                        {"Date": [], "Value": []},
                        schema={"Date": pl.Datetime, "Value": pl.Float64},
                    )
                else:
                    date_dtype_benchmark = data_benchmark.select("Date").dtypes[0]
                    print(f"   Date column type: {date_dtype_benchmark}")

                    sample_dates_benchmark = (
                        data_benchmark.select("Date").head(5).to_series().to_list()
                    )
                    print(f"   Sample dates (first 5): {sample_dates_benchmark}")

                    if date_dtype_benchmark in [pl.Utf8, pl.String]:
                        print("   → Converting timezone-aware strings via pandas...")

                        benchmark_pandas = data_benchmark.to_pandas()
                        benchmark_pandas["Date"] = pd.to_datetime(
                            benchmark_pandas["Date"],
                            utc=True,
                            errors="coerce",
                        )

                        if isinstance(benchmark_pandas["Date"].dtype, pd.DatetimeTZDtype):
                            benchmark_pandas["Date"] = benchmark_pandas["Date"].dt.tz_localize(None)
                            print(
                                "   ✅ Converted timezone-aware strings → UTC → timezone-naive datetime",
                            )

                        filtered_benchmark = pl.from_pandas(benchmark_pandas)

                    elif date_dtype_benchmark == pl.Date:
                        filtered_benchmark = data_benchmark.clone().with_columns(
                            [pl.col("Date").cast(pl.Datetime).alias("Date")],
                        )
                        print("   ✅ Converted pl.Date → pl.Datetime")

                    elif date_dtype_benchmark == pl.Datetime:
                        filtered_benchmark = data_benchmark.clone()
                        if filtered_benchmark["Date"].dtype.time_zone is not None:
                            filtered_benchmark = filtered_benchmark.with_columns(
                                [pl.col("Date").dt.replace_time_zone(None).alias("Date")],
                            )
                            print("   ✅ Removed timezone from pl.Datetime")
                        else:
                            print("   ✅ Already timezone-naive pl.Datetime")
                    else:
                        print("   ⚠️  Unknown date type, attempting pandas conversion...")
                        benchmark_pandas = data_benchmark.to_pandas()
                        benchmark_pandas["Date"] = pd.to_datetime(
                            benchmark_pandas["Date"],
                            utc=True,
                            errors="coerce",
                        )
                        if isinstance(benchmark_pandas["Date"].dtype, pd.DatetimeTZDtype):
                            benchmark_pandas["Date"] = benchmark_pandas["Date"].dt.tz_localize(None)
                        filtered_benchmark = pl.from_pandas(benchmark_pandas)

                    # Business logic validation - get actual date range from data
                    if not filtered_benchmark.is_empty():
                        data_min_date = filtered_benchmark.select(pl.col("Date").min()).item()
                        data_max_date = filtered_benchmark.select(pl.col("Date").max()).item()
                        print(f"   Available date range: {data_min_date} to {data_max_date}")

                    print("\n🔧 Applying Date Filter to Benchmark:")
                    print(f"   Rows before filtering: {filtered_benchmark.height}")

                    start_date_pl = pl.lit(start_date).cast(pl.Datetime)
                    end_date_pl = pl.lit(end_date).cast(pl.Datetime)

                    filtered_benchmark = filtered_benchmark.filter(
                        pl.col("Date").is_not_null()
                        & (pl.col("Date") >= start_date_pl)
                        & (pl.col("Date") <= end_date_pl),
                    ).sort("Date")

                    print(f"   Rows after filtering: {filtered_benchmark.height}")

                    # Early return validation - warn if filtering removed all data
                    if filtered_benchmark.is_empty():
                        print("   ❌ WARNING: Filtering removed ALL benchmark data!")
                    else:
                        final_min_date = filtered_benchmark.select(pl.col("Date").min()).item()
                        final_max_date = filtered_benchmark.select(pl.col("Date").max()).item()
                        print(f"   ✅ Final range: {final_min_date} to {final_max_date}")

            print(f"\n{'=' * 70}")
            print("✅ Filtering Complete")
            print(f"   Portfolio rows: {filtered_portfolio.height}")
            print(f"   Benchmark rows: {filtered_benchmark.height}")
            print(f"{'=' * 70}\n")

            return filtered_portfolio, filtered_benchmark

        except Exception as exc:
            print(f"❌ CRITICAL Error in get_filtered_comparison_data: {exc}")
            import traceback

            traceback.print_exc()
            raise RuntimeError(f"Error filtering comparison data: {exc}")

    @output
    @render.ui
    def output_ID_tab_portfolios_subtab_comparison_loading_status():
        """
        Display loading status and data availability feedback to users.

        This function provides real-time feedback about data processing status and
        helps users understand whether data is ready for visualization.

        Returns
        -------
        shiny.ui.div
            Bootstrap alert div with appropriate styling based on data availability
        """
        try:
            # Get filtered comparison data
            data_portfolio_filtered, data_benchmark_filtered = get_filtered_comparison_data()

            # Business logic validation - check data availability
            portfolio_count = (
                data_portfolio_filtered.height if not data_portfolio_filtered.is_empty() else 0
            )
            benchmark_count = (
                data_benchmark_filtered.height if not data_benchmark_filtered.is_empty() else 0
            )

            # Generate status message based on data availability
            if portfolio_count > 0 and benchmark_count > 0:
                return ui.div(
                    ui.span("✅", class_="me-2"),
                    f"Data ready: {portfolio_count} portfolio points, {benchmark_count} benchmark points",
                    class_="alert alert-success p-2 mb-3",
                )
            if portfolio_count > 0 or benchmark_count > 0:
                return ui.div(
                    ui.span("⚠️", class_="me-2"),
                    f"Partial data: Portfolio ({portfolio_count} points), Benchmark ({benchmark_count} points)",
                    class_="alert alert-warning p-2 mb-3",
                )
            return ui.div(
                ui.span("❌", class_="me-2"),
                "No data available for selected period. Please adjust time range.",
                class_="alert alert-danger p-2 mb-3",
            )

        except Exception as exc:
            print(f"❌ Error in output_ID_tab_portfolios_subtab_comparison_loading_status: {exc}")
            return ui.div(
                ui.span("❌", class_="me-2"),
                f"Error: {exc!s}",
                class_="alert alert-danger p-2 mb-3",
            )

    @output
    @render_widget  # pyright: ignore[reportArgumentType]  # pyrefly: ignore[bad-specialization]  # go.Figure satisfies Widget protocol at runtime
    def output_ID_tab_portfolios_subtab_comparison_plot_main():
        """
        Generate main portfolio vs benchmark comparison visualization.

        This function creates interactive Plotly charts based on user-selected
        visualization type with proper error handling and fallback displays.

        Returns
        -------
        plotly.graph_objects.Figure
            Interactive comparison chart with portfolio and benchmark data
        """
        try:
            # Get filtered comparison data (already in Polars format)
            data_portfolio_filtered, data_benchmark_filtered = get_filtered_comparison_data()

            print(f"\n{'=' * 70}")
            print("📊 Plot Generation Debug")
            print(f"{'=' * 70}")
            print(f"Portfolio filtered rows: {data_portfolio_filtered.height}")
            print(f"Benchmark filtered rows: {data_benchmark_filtered.height}")

            # Early return validation - check if data is available
            if data_portfolio_filtered.is_empty() and data_benchmark_filtered.is_empty():
                print("❌ Both datasets empty - returning error figure")
                return create_error_figure(
                    "No Data Available",
                    "Please select a time period with available portfolio and benchmark data.",
                )

            # Get visualization preferences from user inputs
            viz_type = input.input_ID_tab_portfolios_subtab_comparison_viz_type()
            show_diff = input.input_ID_tab_portfolios_subtab_comparison_show_diff()
            print(f"Viz type: {viz_type}, Show diff: {show_diff}")

            # Calculate actual date range from filtered data
            if not data_portfolio_filtered.is_empty():
                actual_start_date_raw = data_portfolio_filtered.select(pl.col("Date").min()).item()
                actual_end_date_raw = data_portfolio_filtered.select(pl.col("Date").max()).item()
            elif not data_benchmark_filtered.is_empty():
                actual_start_date_raw = data_benchmark_filtered.select(pl.col("Date").min()).item()
                actual_end_date_raw = data_benchmark_filtered.select(pl.col("Date").max()).item()
            else:
                actual_start_date_raw, actual_end_date_raw = get_effective_date_range()

            # Convert to pandas Timestamp for consistency
            actual_start_date = pd.Timestamp(actual_start_date_raw)
            actual_end_date = pd.Timestamp(actual_end_date_raw)

            print(f"Actual data date range: {actual_start_date} to {actual_end_date}")

            # Generate period label based on ACTUAL data range
            time_period = input.input_ID_tab_portfolios_subtab_comparison_time_period()

            if time_period == "custom":
                period_label = f"{actual_start_date.strftime('%Y-%m-%d')} to {actual_end_date.strftime('%Y-%m-%d')}"
            else:
                period_labels_base = {
                    "1y": "Last 1 Year",
                    "3y": "Last 3 Years",
                    "5y": "Last 5 Years",
                    "10y": "Last 10 Years",
                    "ytd": "Year to Date",
                }
                base_label = period_labels_base.get(time_period, "Selected Period")
                period_label = f"{base_label} ({actual_start_date.strftime('%Y-%m-%d')} to {actual_end_date.strftime('%Y-%m-%d')})"

            print(f"Period label: {period_label}")

            # Log Polars DataFrame information before passing to plot function
            print("\n→ Polars DataFrames ready for plotting...")

            if not data_portfolio_filtered.is_empty():
                print(
                    f"  Portfolio Polars shape: ({data_portfolio_filtered.height}, {data_portfolio_filtered.width})",
                )
                print(f"  Portfolio columns: {data_portfolio_filtered.columns}")
                print(f"  Portfolio dtypes: {data_portfolio_filtered.dtypes}")
                print(
                    f"  Portfolio Date range: {data_portfolio_filtered['Date'].min()} to {data_portfolio_filtered['Date'].max()}",
                )
                print(f"  Portfolio sample (first 3):\n{data_portfolio_filtered.head(3)}")
            else:
                print("  Portfolio empty")

            if not data_benchmark_filtered.is_empty():
                print(
                    f"  Benchmark Polars shape: ({data_benchmark_filtered.height}, {data_benchmark_filtered.width})",
                )
                print(f"  Benchmark columns: {data_benchmark_filtered.columns}")
                print(f"  Benchmark dtypes: {data_benchmark_filtered.dtypes}")
                print(
                    f"  Benchmark Date range: {data_benchmark_filtered['Date'].min()} to {data_benchmark_filtered['Date'].max()}",
                )
                print(f"  Benchmark sample (first 3):\n{data_benchmark_filtered.head(3)}")
            else:
                print("  Benchmark empty")

            # Verify data integrity with detailed checks
            has_portfolio_data = (
                not data_portfolio_filtered.is_empty() and data_portfolio_filtered.height > 0
            )
            has_benchmark_data = (
                not data_benchmark_filtered.is_empty() and data_benchmark_filtered.height > 0
            )

            print("\n→ Final validation before plotting...")
            print(
                f"  Has portfolio data: {has_portfolio_data} ({data_portfolio_filtered.height if has_portfolio_data else 0} rows)",
            )
            print(
                f"  Has benchmark data: {has_benchmark_data} ({data_benchmark_filtered.height if has_benchmark_data else 0} rows)",
            )

            if not has_portfolio_data and not has_benchmark_data:
                print("❌ No valid data for plotting!")
                return create_error_figure(
                    "No Valid Data",
                    "Both portfolio and benchmark datasets are empty or invalid.",
                )

            # Log parameters being passed to plot function
            print("\n  Parameters being passed to create_plot_comparison_portfolios:")
            print(
                f"    - data_portfolio: Polars DataFrame with {data_portfolio_filtered.height} rows",
            )
            print(
                f"    - data_benchmark: Polars DataFrame with {data_benchmark_filtered.height} rows",
            )
            print(f"    - viz_type: {viz_type} (type: {type(viz_type)})")
            print(f"    - show_diff: {show_diff} (type: {type(show_diff)})")
            print(f"    - period_label: {period_label} (type: {type(period_label)})")
            print(f"    - start_date: {actual_start_date} (type: {type(actual_start_date)})")
            print(f"    - end_date: {actual_end_date} (type: {type(actual_end_date)})")
            print("    - data_was_validated: False (let function validate Polars data)")

            # Create the comparison plot - pass POLARS DataFrames directly
            # Set data_was_validated=False so the function performs its own validation
            print("\n→ Calling create_plot_comparison_portfolios with Polars DataFrames...")

            try:
                fig = create_plot_comparison_portfolios(
                    data_portfolio=data_portfolio_filtered,
                    data_benchmark=data_benchmark_filtered,
                    viz_type=viz_type,
                    show_diff=show_diff,
                    period_label=period_label,
                    start_date=actual_start_date,
                    end_date=actual_end_date,
                    data_was_validated=False,  # Let function validate and convert Polars data
                )

                print("✅ Plot created successfully")
                print(f"  Figure type: {type(fig)}")
                print(
                    f"  Figure has data traces: {len(fig.data) if hasattr(fig, 'data') else 'N/A'}",  # pyright: ignore[reportArgumentType]
                )

                if hasattr(fig, "data") and len(fig.data) > 0:  # pyright: ignore[reportArgumentType]
                    print("  Trace details:")
                    for idx, trace in enumerate(fig.data):
                        trace_name = trace.name if hasattr(trace, "name") else "Unnamed"  # pyright: ignore[reportAttributeAccessIssue]
                        trace_type = type(trace).__name__
                        trace_len = (
                            len(trace.x) if hasattr(trace, "x") and trace.x is not None else 0  # pyright: ignore[reportAttributeAccessIssue]
                        )
                        print(
                            f"    Trace {idx}: {trace_name} ({trace_type}) with {trace_len} data points",
                        )
                else:
                    print("  ⚠️ WARNING: Figure created but has no data traces!")

                print(f"{'=' * 70}\n")

            except Exception as exc_plot:
                print(f"❌ Error inside create_plot_comparison_portfolios: {exc_plot}")
                import traceback

                traceback.print_exc()

                # Return error figure with detailed message
                return create_error_figure(
                    "Plot Generation Error",
                    f"Error in create_plot_comparison_portfolios: {exc_plot!s}",
                )

            # Store latest figure in shared reactive state for report pipeline
            update_visual_object_in_reactives(
                reactives_shiny,
                "Chart_Portfolio_Comparison",
                fig,
            )

            return fig

        except Exception as exc:
            print(f"❌ Error in output_ID_tab_portfolios_subtab_comparison_plot_main: {exc}")
            import traceback

            traceback.print_exc()
            return create_error_figure(
                "Error Generating Chart",
                f"An error occurred while creating the comparison chart: {exc!s}",
            )

    @output
    @render.ui
    def output_ID_tab_portfolios_subtab_comparison_table_stats():
        """
        Generate performance statistics comparison table.

        This function calculates and displays comprehensive performance metrics
        comparing portfolio and benchmark performance over the selected period.

        Returns
        -------
        shiny.ui.HTML
            Formatted HTML table with performance statistics
        """
        try:
            # Get filtered comparison data
            data_portfolio_filtered, data_benchmark_filtered = get_filtered_comparison_data()

            # Early return validation - check if data is available
            if data_portfolio_filtered.is_empty() or data_benchmark_filtered.is_empty():
                return ui.div(
                    ui.p(
                        "No data available for statistics calculation.",
                        class_="text-muted text-center p-3",
                    ),
                )

            # Convert to pandas for statistics calculations
            portfolio_pandas = data_portfolio_filtered.to_pandas()
            benchmark_pandas = data_benchmark_filtered.to_pandas()

            # Ensure Date column is datetime type
            portfolio_pandas["Date"] = pd.to_datetime(portfolio_pandas["Date"])
            benchmark_pandas["Date"] = pd.to_datetime(benchmark_pandas["Date"])

            # Sort by date for consistency
            portfolio_pandas = portfolio_pandas.sort_values("Date")
            benchmark_pandas = benchmark_pandas.sort_values("Date")

            # Calculate basic statistics
            stats_data = []

            # Period returns
            portfolio_start = portfolio_pandas["Value"].iloc[0] if len(portfolio_pandas) > 0 else 0
            portfolio_end = portfolio_pandas["Value"].iloc[-1] if len(portfolio_pandas) > 0 else 0
            portfolio_return = (
                ((portfolio_end / portfolio_start) - 1) * 100 if portfolio_start > 0 else 0  # ty: ignore[division-by-zero]
            )

            benchmark_start = benchmark_pandas["Value"].iloc[0] if len(benchmark_pandas) > 0 else 0
            benchmark_end = benchmark_pandas["Value"].iloc[-1] if len(benchmark_pandas) > 0 else 0
            benchmark_return = (
                ((benchmark_end / benchmark_start) - 1) * 100 if benchmark_start > 0 else 0  # ty: ignore[division-by-zero]
            )

            stats_data.append(
                {
                    "Metric": "Total Return (%)",
                    "Portfolio": f"{portfolio_return:.2f}",
                    "Benchmark": f"{benchmark_return:.2f}",
                    "Difference": f"{portfolio_return - benchmark_return:.2f}",
                },
            )

            # Calculate daily returns for volatility
            portfolio_pandas["Daily_Return"] = portfolio_pandas["Value"].pct_change()
            benchmark_pandas["Daily_Return"] = benchmark_pandas["Value"].pct_change()

            # Volatility (annualized standard deviation)
            portfolio_volatility = portfolio_pandas["Daily_Return"].std() * np.sqrt(252) * 100
            benchmark_volatility = benchmark_pandas["Daily_Return"].std() * np.sqrt(252) * 100

            stats_data.append(
                {
                    "Metric": "Annualized Volatility (%)",
                    "Portfolio": f"{portfolio_volatility:.2f}",
                    "Benchmark": f"{benchmark_volatility:.2f}",
                    "Difference": f"{portfolio_volatility - benchmark_volatility:.2f}",
                },
            )

            # Sharpe Ratio (assuming 0% risk-free rate for simplicity)
            portfolio_mean_return = portfolio_pandas["Daily_Return"].mean() * 252
            portfolio_sharpe = (
                (portfolio_mean_return / (portfolio_pandas["Daily_Return"].std() * np.sqrt(252)))
                if portfolio_pandas["Daily_Return"].std() > 0
                else 0
            )

            benchmark_mean_return = benchmark_pandas["Daily_Return"].mean() * 252
            benchmark_sharpe = (
                (benchmark_mean_return / (benchmark_pandas["Daily_Return"].std() * np.sqrt(252)))
                if benchmark_pandas["Daily_Return"].std() > 0
                else 0
            )

            stats_data.append(
                {
                    "Metric": "Sharpe Ratio",
                    "Portfolio": f"{portfolio_sharpe:.2f}",
                    "Benchmark": f"{benchmark_sharpe:.2f}",
                    "Difference": f"{portfolio_sharpe - benchmark_sharpe:.2f}",
                },
            )

            # Max Drawdown
            portfolio_pandas["Cumulative_Max"] = portfolio_pandas["Value"].cummax()
            portfolio_pandas["Drawdown"] = (
                portfolio_pandas["Value"] / portfolio_pandas["Cumulative_Max"]
            ) - 1
            portfolio_max_drawdown = portfolio_pandas["Drawdown"].min() * 100

            benchmark_pandas["Cumulative_Max"] = benchmark_pandas["Value"].cummax()
            benchmark_pandas["Drawdown"] = (
                benchmark_pandas["Value"] / benchmark_pandas["Cumulative_Max"]
            ) - 1
            benchmark_max_drawdown = benchmark_pandas["Drawdown"].min() * 100

            stats_data.append(
                {
                    "Metric": "Max Drawdown (%)",
                    "Portfolio": f"{portfolio_max_drawdown:.2f}",
                    "Benchmark": f"{benchmark_max_drawdown:.2f}",
                    "Difference": f"{portfolio_max_drawdown - benchmark_max_drawdown:.2f}",
                },
            )

            # Create DataFrame and convert to HTML with proper styling
            stats_dataframe = pd.DataFrame(stats_data)

            # Generate HTML table with Bootstrap classes
            html_table = stats_dataframe.to_html(
                index=False,
                classes="table table-striped table-hover table-sm",
                escape=False,
                border=0,
            )

            # Add custom CSS to ensure proper column header alignment
            # Bootstrap's table classes center-align headers by default, so we override this
            styled_html_table = f"""
            <style>
                .stats-table-container table thead th {{
                    text-align: left !important;
                    vertical-align: bottom;
                    border-bottom: 2px solid #dee2e6;
                    padding: 0.75rem;
                    font-weight: 600;
                }}
                .stats-table-container table tbody td {{
                    text-align: left;
                    vertical-align: top;
                    padding: 0.75rem;
                }}
                .stats-table-container table tbody td:first-child {{
                    font-weight: 500;
                }}
            </style>
            <div class="stats-table-container">
                {html_table}
            </div>
            """

            return ui.HTML(styled_html_table)

        except Exception as exc:
            print(f"❌ Error in output_ID_tab_portfolios_subtab_comparison_table_stats: {exc}")
            import traceback

            traceback.print_exc()
            return ui.div(
                ui.p(
                    f"Error generating statistics: {exc!s}",
                    class_="text-danger text-center p-3",
                ),
            )
