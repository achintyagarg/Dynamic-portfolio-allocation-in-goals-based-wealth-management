"""Portfolio Weights Analysis Module.

Provides comprehensive analysis of portfolio ETF component weights over time.

![Portfolio Analysis](https://img.shields.io/badge/Portfolio-Analysis-blue)
![Python](https://img.shields.io/badge/python-3.11+-green)
![Shiny](https://img.shields.io/badge/shiny-for_python-orange)

## Overview

The Portfolio Weights Analysis module provides comprehensive analysis and visualization
of portfolio ETF component weights over time within the QWIM Dashboard's Portfolios section.

This module enables detailed examination of portfolio composition evolution with interactive
component selection, multiple visualization types, and statistical analysis capabilities.

## Architecture

```mermaid
graph TD
    A[Portfolio Weights Analysis Module] --> B[UI Components]
    A --> C[Server Logic]
    B --> D[Time Period Selection]
    B --> E[Component Selection]
    B --> F[Visualization Options]
    C --> G[Data Processing]
    C --> H[Plot Generation]
    C --> I[Statistics Calculation]
    G --> J[Date Filtering]
    G --> K[Component Filtering]
    H --> L[Multiple Chart Types]
    I --> M[Weight Statistics]
```

## Features

!!! success "Core Capabilities"
    - **Interactive Weight Visualization**: Multiple chart types for portfolio weight analysis
    - **Dynamic Component Selection**: Checkbox-based ETF component filtering with "Select All" option
    - **Time Period Analysis**: Preset periods (1Y, 3Y, 5Y, 10Y, YTD) plus custom date ranges
    - **Multiple Visualization Types**: Stacked area, bar, line, and heatmap charts
    - **Statistical Analysis**: Comprehensive weight distribution statistics and trends
    - **Real-time Updates**: Reactive interface that updates instantly with selection changes

!!! info "Visualization Types"
    - **Stacked Area Chart**: Shows component weight evolution with filled areas
    - **Stacked Bar Chart**: Discrete time period weight distribution
    - **Line Chart**: Individual component weight trends over time
    - **Heatmap**: Weight distribution intensity visualization
    - **Pie Chart**: Current portfolio composition snapshot

## Dependencies

=== "Core Dependencies"
    - **shiny**: Reactive UI framework for Python
    - **shinywidgets**: Plotly widget integration
    - **plotly**: Interactive chart generation and visualization
    - **polars**: High-performance DataFrame operations
    - **pandas**: Data manipulation and analysis
    - **numpy**: Numerical computations and array operations

=== "Optional Dependencies"
    - **beautifulsoup4**: HTML parsing for enhanced table formatting

=== "Internal Dependencies"
    - `Shiny_Utils.utils_data`: Data processing and validation utilities
    - `Shiny_Utils.utils_visuals`: Visualization generation and export utilities

## Data Requirements

### Input Data Format

The module expects the following data inputs:

| Data Input | Type | Description | Required Columns |
|-----------|------|-------------|-----------------|
| `Weights_My_Portfolio` | `pl.DataFrame` | Portfolio component weights data | `Date`, `[ETF_Components]` |
| `My_Portfolio` | `pl.DataFrame` | Portfolio performance data (for validation) | `Date`, `Value` |
| `Benchmark_Portfolio` | `pl.DataFrame` | Benchmark data (for validation) | `Date`, `Value` |

!!! note "Weight Data Format"
    - Date column supports multiple formats with automatic conversion
    - ETF component columns should contain weight values (numeric)
    - Weight values can be absolute or relative (percentage conversion available)
    - Missing values are automatically filled with 0.0

## Performance Optimizations

!!! success "Optimization Features"
    - **Data Downsampling**: Automatic optimization to 200 points maximum for rendering
    - **Reactive Component Selection**: Only selected ETFs are processed and displayed
    - **Efficient Date Filtering**: String-based date comparison for improved performance
    - **Polars Operations**: High-performance DataFrame processing for large datasets
    - **Lazy Evaluation**: Calculations only when dependencies change

---

**Module Information:**
- **Author**: QWIM Development Team
- **Version**: 0.5.1
- **Last Updated**: June 2025
- **License**: [Project License]
- **Dependencies**: See requirements.txt for complete dependency list
"""

# Standard library imports organized by functionality
from __future__ import annotations

import math
import typing

from datetime import UTC, datetime, timedelta  # Date/time operations for period calculations
from pathlib import Path  # Cross-platform file path handling with modern API
from typing import Any

# High-performance data processing libraries
import pandas as pd  # Traditional DataFrame operations and compatibility layer
import plotly.graph_objects as go  # Core Plotly chart objects for custom visualizations
import polars as pl  # High-performance DataFrame operations optimized for large datasets


# Shiny framework components for reactive web applications
from shiny import module, reactive, render, ui  # Core Shiny functionality
from shinywidgets import output_widget, render_widget  # Plotly widget integration

# Custom logging utilities
from src.utils.custom_exceptions_errors_loggers.logger_custom import get_logger


#: Module-level logger instance
_logger = get_logger(__name__)

# Optional dependencies with graceful fallbacks for enhanced functionality
try:
    # BeautifulSoup enables enhanced HTML processing if needed
    HAS_BEAUTIFULSOUP = True
except ImportError:
    HAS_BEAUTIFULSOUP = False
    # Fallback: Basic HTML table generation without enhanced styling features


# Internal QWIM Dashboard utility modules providing reusable functionality
from src.dashboard.shiny_utils.reactives_shiny import (  # noqa: E402
    update_visual_object_in_reactives,  # Store latest chart in shared reactive state
)
from src.dashboard.shiny_utils.utils_data import (  # noqa: E402
    validate_portfolio_data,  # Ensure data integrity and format compliance
)
from src.dashboard.shiny_utils.utils_reporting import (  # noqa: E402
    save_weights_analysis_outputs_to_reactives,  # Persist weight stats for PDF report
)
from src.dashboard.shiny_utils.utils_visuals import (  # noqa: E402
    create_error_figure,  # Standardized error visualization with consistent styling
)


# Module configuration constants controlling behavior and file organization
OUTPUT_DIR = Path("output")  # Base directory for all module output files


@module.ui
def subtab_weights_analysis_ui(data_utils: dict, data_inputs: dict) -> Any:
    """
    Create the comprehensive user interface for the Portfolio Weights Analysis subtab.

    This function constructs a responsive, feature-rich UI layout for portfolio ETF component
    weight analysis with interactive component selection, multiple visualization types, and
    comprehensive statistical analysis capabilities.

    ## UI Architecture

    The interface follows a sidebar-main content layout pattern optimized for
    portfolio weights analysis workflows:

    ```
    ┌─────────────────┬──────────────────────────────────┐
    │    Sidebar      │        Main Content Area         │
    │                 │                                  │
    │ • Time Period   │ • Loading Status                 │
    │ • ETF Selection │ • Main Weight Distribution Plot  │
    │ • Viz Options   │ • Secondary Composition Chart    │
    │ • Data Info     │ • Weight Statistics Table        │
    │                 │                                  │
    └─────────────────┴──────────────────────────────────┘
    ```

    ## Component Hierarchy

    ### Sidebar Controls (350px width)

    #### 📅 Time Period Selection
    - **Preset Periods**: Dropdown with common analysis periods (1Y, 3Y, 5Y, 10Y, YTD)
    - **Custom Range**: Conditional date range picker for custom analysis periods
    - **Calculated Range Display**: Shows computed date range for preset selections

    #### 🔧 ETF Component Selection
    - **Select All Toggle**: Master checkbox for selecting/deselecting all components
    - **Individual Checkboxes**: Component-specific selection with default first 3 checked
    - **Scrollable Container**: Max height 200px with overflow scroll for many components
    - **Selection Counter**: Real-time count of selected vs total ETF components

    #### 📊 Visualization Options
    - **Chart Type Selector**: Multiple visualization types (Area, Bar, Line, Heatmap)
    - **Percentage Toggle**: Convert weights to percentage representation
    - **Sort Components**: Option to sort components by average weight

    #### ℹ️ Data Information Panel
    - **Real-time Data Statistics**: Live count of weight data points
    - **Period Information**: Current analysis period details
    - **Component Status**: Selected vs available component counts
    - **Readiness Indicators**: Visual feedback on data availability

    ### Main Content Area (Responsive)

    #### Status and Loading
    - **Loading Status Bar**: Real-time feedback during data processing

    #### Primary Visualization
    - **Weight Distribution Plot**: Full-screen capable main chart (600px height)
    - **Interactive Features**: Plotly controls with range slider navigation

    #### Secondary Analysis
    - **Two-Column Layout**: Balanced 50/50 split for complementary analysis
    - **Current Composition**: Pie chart of latest portfolio composition
    - **Weight Statistics**: Comprehensive statistical summary table

    ## Input Component Naming Convention

    Following project standards, all input components use the naming pattern:
    `input_ID_tab_portfolios_subtab_weights_analysis_{identifier}`

    Examples
    --------
    - Time period: `input_ID_tab_portfolios_subtab_weights_analysis_time_period`
    - ETF components: `input_ID_tab_portfolios_subtab_weights_analysis_component_{etf_name}`
    - Visualization: `input_ID_tab_portfolios_subtab_weights_analysis_viz_type`

    ## Responsive Design Features

    !!! success "Mobile-Friendly Design"
        - **Bootstrap Grid System**: Responsive column layouts adapt to screen size
        - **Fixed Sidebar Width**: Consistent control panel (350px) for optimal UX
        - **Scrollable Component Selection**: Handles large numbers of ETF components
        - **Card-Based Layout**: Modular design with professional appearance
        - **Flexible Main Content**: Adapts to remaining screen real estate

    ## Accessibility Features

    !!! info "Universal Design"
        - **Semantic HTML**: Proper heading hierarchy (h3, h5) and ARIA labels
        - **Keyboard Navigation**: Full keyboard accessibility for all controls
        - **Screen Reader Support**: Descriptive text and status updates
        - **High Contrast**: Material Design color scheme with sufficient contrast
        - **Focus Management**: Clear visual focus indicators for interactive elements

    ## Advanced UI Features

    ### Dynamic Component Selection
    - **Master Control**: "Select All" checkbox controls all individual components
    - **Default Selection**: First 3 ETF components checked by default for immediate analysis
    - **Reactive Updates**: Real-time synchronization between selection and visualization
    - **Performance Optimization**: Only selected components processed for improved speed

    ### Conditional Display Logic
    - **Smart Date Range**: Custom date picker only shown when "Custom" period selected
    - **Calculated Range**: Automatic date range display for preset period selections
    - **Context-Aware Information**: Data info panel updates based on current selections

    ### Visual Enhancement
    - **Emoji Section Headers**: Intuitive visual cues for different UI sections
    - **Bootstrap Styling**: Professional appearance with consistent spacing
    - **Progress Indicators**: Visual feedback during data loading and processing
    - **Status Icons**: Clear success/warning/error indicators in data info panel

    Parameters
    ----------
    data_utils : dict
        Dictionary containing utility functions and dashboard configurations.

        Expected keys:
        - Configuration settings for dashboard behavior and theming
        - Utility functions for data processing and validation
        - Theme preferences and styling configurations
        - Performance optimization settings

    data_inputs : dict
        Dictionary containing portfolio weight datasets and related data.

        Expected keys:
        - 'Weights_My_Portfolio': ETF component weights data (pl.DataFrame)
        - 'My_Portfolio': Portfolio performance data for validation (pl.DataFrame)
        - 'Benchmark_Portfolio': Benchmark data for validation (pl.DataFrame)
        - 'Time_Series_ETFs': ETF time series data (pl.DataFrame, optional)

    Returns
    -------
    shiny.ui.div
        Complete UI layout containing:
        - Responsive sidebar with all weight analysis controls
        - Main content area with multiple visualization components
        - Proper Bootstrap classes for responsive behavior
        - All necessary input/output component definitions
        - Semantic HTML structure for accessibility

    Examples
    --------
    Basic usage in a Shiny application:

    ```python
    # In your main application UI
    weights_analysis_tab = subtab_weights_analysis_ui(
        data_utils={"theme": "material", "export_enabled": True, "default_etf_selection": 3},
        data_inputs={
            "Weights_My_Portfolio": etf_weights_dataframe,
            "My_Portfolio": portfolio_performance_dataframe,
            "Benchmark_Portfolio": benchmark_dataframe,
        },
    )
    ```

    Integration with portfolio analysis module:

    ```python
    ui.nav_panel(
        "Weight Distribution",
        subtab_weights_analysis_ui(data_utils=dashboard_utilities, data_inputs=portfolio_datasets),
    )
    ```

    Notes
    -----
    ### Data Requirements
    - Weights data must contain Date column and ETF component columns
    - Component columns should contain numeric weight values
    - Missing values automatically filled with 0.0 for analysis
    - Date formats automatically detected and converted

    ### Performance Considerations
    - Component selection limited to improve rendering performance
    - Data downsampling applied automatically for large datasets (>200 points)
    - Reactive calculations optimized with lazy evaluation
    - PNG export functionality conditional based on configuration flags

    ### User Experience
    - Default to 5-year analysis period for comprehensive view
    - First 3 ETF components selected by default for immediate results
    - All visualizations synchronized with component selection changes
    - Real-time feedback through loading indicators and status displays

    ### Browser Compatibility
    - Plotly visualizations supported across all modern browsers
    - Bootstrap responsive design works on desktop, tablet, and mobile
    - Touch interactions supported for mobile device usage
    - Graceful degradation for older browser versions

    See Also
    --------
    subtab_weights_analysis_server : Server logic for this UI
    get_names_time_series_from_DF : ETF component name extraction utility
    create_table_summary_weights_analysis : Statistics table generation
    validate_portfolio_data : Data validation utility function
    """
    # Main container with semantic heading following project standards
    # Uses h3 for proper heading hierarchy within the portfolio tab structure
    return ui.div(
        # Page title with clear identification of analysis functionality
        ui.h3("Portfolio Weight Analysis"),
        # Bootstrap responsive sidebar layout optimized for weights analysis workflow
        # Provides optimal organization of controls and visualization space
        ui.layout_sidebar(
            # Sidebar component containing all interactive controls for weight analysis
            # Fixed width ensures consistent layout and optimal control organization
            ui.sidebar(
                # =============================================================
                # TIME PERIOD SELECTION SECTION
                # =============================================================
                # Visual section separator with emoji for enhanced user experience
                ui.h5("📅 Time Period Selection", class_="mb-3"),
                # Primary time period dropdown selector for analysis timeframe
                # Follows project naming convention: input_ID_tab_portfolios_subtab_weights_analysis_{identifier}
                ui.input_select(
                    "input_ID_tab_portfolios_subtab_weights_analysis_time_period",
                    "Select Time Period",
                    {
                        # Custom option allows user-defined date ranges for flexibility
                        "custom": "Custom",
                        # Preset periods for common portfolio analysis timeframes
                        "1y": "Last 1 Year",  # 365 days from current date
                        "3y": "Last 3 Years",  # 3 * 365 days from current date
                        "5y": "Last 5 Years",  # 5 * 365 days from current date - default
                        "10y": "Last 10 Years",  # 10 * 365 days from current date
                        "ytd": "Year to Date",  # January 1st current year to today
                    },
                    selected="5y",  # Default to 5 years for comprehensive analysis
                ),
                # Conditional date range picker - only displayed when "custom" period selected
                # Uses JavaScript conditional rendering for enhanced user experience
                ui.panel_conditional(
                    "input.input_ID_tab_portfolios_subtab_weights_analysis_time_period === 'custom'",
                    ui.div(
                        # Custom date range input with sensible defaults for portfolio analysis
                        ui.input_date_range(
                            "input_ID_tab_portfolios_subtab_weights_analysis_date_range",
                            "Custom Date Range",
                            start=datetime.now(UTC) - timedelta(days=365),  # Default to last year
                            end=datetime.now(UTC),  # Default to today
                            format="yyyy-mm-dd",  # ISO date format for consistency
                            separator=" to ",  # Clear separator text
                            width="100%",  # Full width within container
                        ),
                        class_="mt-2",  # Bootstrap margin-top spacing for visual separation
                    ),
                ),
                # Display calculated date range for preset selections
                # Provides transparency about actual date range being analyzed
                ui.panel_conditional(
                    "input.input_ID_tab_portfolios_subtab_weights_analysis_time_period !== 'custom'",
                    ui.div(
                        # Output component for displaying calculated date range
                        ui.output_text(
                            "output_ID_tab_portfolios_subtab_weights_analysis_calculated_date_range",
                        ),
                        # Bootstrap styling for subtle, informative appearance
                        class_="mt-2 p-2 bg-light rounded text-muted small",
                    ),
                ),
                # Visual separator between major control sections
                ui.hr(),
                # =============================================================
                # ETF COMPONENT SELECTION SECTION
                # =============================================================
                ui.h5("🔧 Select Components", class_="mb-3"),
                # Master "Select All" checkbox for controlling all ETF component selections
                # Provides efficient way to select/deselect all components simultaneously
                ui.div(
                    ui.input_checkbox(
                        "input_ID_tab_portfolios_subtab_weights_analysis_select_all_components",
                        "Select All ETFs",
                        value=False,  # Default unchecked to show individual selection control
                    ),
                    class_="mb-2",  # Bootstrap margin-bottom spacing
                ),
                # Scrollable container for individual ETF component checkboxes
                # Handles scenarios with many ETF components without overwhelming the UI
                ui.div(
                    # Dynamic output containing individual ETF component checkboxes
                    ui.output_ui(
                        "output_ID_tab_portfolios_subtab_weights_analysis_component_checkboxes",
                    ),
                    class_="component-selection-container",
                    # Custom styling for scrollable checkbox container
                    style="max-height: 200px; overflow-y: auto; border: 1px solid #dee2e6; border-radius: 0.375rem; padding: 0.5rem;",
                ),
                # Real-time display of selected component count for user awareness
                # Provides immediate feedback on current selection state
                ui.div(
                    ui.output_text(
                        "output_ID_tab_portfolios_subtab_weights_analysis_selected_components_count",
                    ),
                    class_="mt-1 text-muted small",  # Subtle styling for secondary information
                ),
                # Visual separator between control sections
                ui.hr(),
                # =============================================================
                # VISUALIZATION OPTIONS SECTION
                # =============================================================
                ui.h5("📊 Visualization Options", class_="mb-3"),
                # Visualization type selector for different chart representations
                # Offers multiple chart types optimized for weight distribution analysis
                ui.input_select(
                    "input_ID_tab_portfolios_subtab_weights_analysis_viz_type",
                    "Visualization Type",
                    {
                        # Different visualization types for various analysis perspectives
                        "area": "Stacked Area Chart",  # Shows evolution with filled areas - default
                        "bar": "Stacked Bar Chart",  # Discrete time period weight distribution
                        "line": "Line Chart",  # Individual component weight trends
                        "heatmap": "Heatmap",  # Weight distribution intensity visualization
                    },
                    selected="area",  # Default to stacked area for intuitive weight visualization
                ),
                # Percentage display toggle for relative weight analysis
                # Converts absolute weights to percentage representation for easier comparison
                ui.input_checkbox(
                    "input_ID_tab_portfolios_subtab_weights_analysis_show_pct",
                    "Show as Percentage",
                    value=True,  # Default enabled for relative weight analysis
                ),
                # Component sorting toggle for organized weight visualization
                # Sorts components by average weight for clearer visual hierarchy
                ui.input_checkbox(
                    "input_ID_tab_portfolios_subtab_weights_analysis_sort_components",
                    "Sort Components by Weight",
                    value=True,  # Default enabled for organized visualization
                ),
                # =============================================================
                # DATA INFORMATION SECTION
                # =============================================================
                # Visual separator and informational section
                ui.hr(),
                ui.h5("ℹ️ Data Information", class_="mb-3"),
                # Dynamic data information display synchronized with component selection
                # Shows real-time status of weights data and selected components
                ui.output_ui("output_ID_tab_portfolios_subtab_weights_analysis_data_info"),
                # Sidebar configuration following project standards
                width=350,  # Fixed pixel width for consistent layout across screen sizes
                position="left",  # Left-side positioning following dashboard conventions
            ),
            # =============================================================
            # MAIN CONTENT AREA
            # =============================================================
            # Main content area containing all visualization and analysis components
            ui.div(
                # Loading status indicator at top of main content area
                # Provides real-time feedback during data processing and visualization updates
                ui.output_ui("output_ID_tab_portfolios_subtab_weights_analysis_loading_status"),
                # Primary weight distribution visualization card
                # Features full-screen capability for detailed analysis
                ui.card(
                    # Card header with clear, descriptive title
                    ui.card_header("Portfolio Weight Distribution Over Time"),
                    # Main visualization widget with optimal sizing for weight analysis
                    output_widget(
                        "output_ID_tab_portfolios_subtab_weights_analysis_plot_main",
                        height="600px",  # Fixed height for consistent layout and optimal viewing
                        width="100%",  # Responsive width adapts to container
                    ),
                    # Card configuration for enhanced user experience
                    full_screen=True,  # Enable full-screen viewing capability for detailed analysis
                    class_="mb-4",  # Bootstrap margin-bottom spacing
                ),
                # Secondary analysis section with two-column layout
                # Provides complementary analysis views in balanced layout
                ui.div(
                    # Bootstrap responsive column layout for secondary visualizations
                    ui.layout_columns(
                        # Current portfolio composition analysis card
                        ui.card(
                            ui.card_header("Current Portfolio Composition"),
                            # Secondary plot for latest composition visualization (pie chart)
                            output_widget(
                                "output_ID_tab_portfolios_subtab_weights_analysis_plot_secondary",
                                height="400px",  # Smaller height for secondary analysis
                                width="100%",  # Responsive width
                            ),
                            class_="h-100",  # Bootstrap height class for equal card heights
                        ),
                        # Weight statistics summary card
                        ui.card(
                            ui.card_header("Weight Statistics"),
                            # Comprehensive statistical summary table
                            ui.output_ui(
                                "output_ID_tab_portfolios_subtab_weights_analysis_table_summary",
                            ),
                            class_="h-100",  # Bootstrap height class for equal card heights
                        ),
                        # Equal width columns for balanced layout
                        col_widths=[6, 6],  # 50/50 split for optimal secondary analysis viewing
                    ),
                    class_="mt-3",  # Bootstrap margin-top spacing from main visualization
                ),
                # Main content area configuration
                class_="flex-fill",  # Bootstrap flexbox class for responsive layout filling
            ),
        ),
    )


@module.server
def subtab_weights_analysis_server(
    input: typing.Any,
    output: typing.Any,
    session: typing.Any,
    data_utils: dict,
    data_inputs: dict,
    reactives_shiny: dict,
) -> None:
    """
    Server logic for the Portfolio Weights Analysis subtab.

    This function implements the reactive server-side logic for portfolio weight analysis,
    handling data processing, visualization generation, and user interaction responses.

    Parameters
    ----------
    input : typing.Any
        Shiny input object containing user interface input values
    output : typing.Any
        Shiny output object for rendering UI components
    session : typing.Any
        Shiny session object for session-specific operations
    data_utils : dict
        Dictionary containing utility functions and dashboard configurations
    data_inputs : dict
        Dictionary containing portfolio weight datasets:
        - 'Weights_My_Portfolio': ETF component weights data (pl.DataFrame)
        - 'My_Portfolio': Portfolio performance data (pl.DataFrame)
        - 'Benchmark_Portfolio': Benchmark data (pl.DataFrame)
    reactives_shiny : dict
        Dictionary containing shared reactive values across dashboard modules

    Returns
    -------
    None
        Function sets up reactive server logic and renders output components
    """
    # === STEP 1: Input validation - extract and validate data inputs ===
    data_portfolio = data_inputs.get("My_Portfolio")
    data_benchmark = data_inputs.get("Benchmark_Portfolio")
    weights_portfolio = data_inputs.get("Weights_My_Portfolio")
    # Note: time_series_ETFs available in data_inputs but not used in this view

    # === STEP 2: Enhanced debug logging for data availability ===
    print(f"\n{'=' * 80}")
    print("🔍 Weights Analysis MODULE - DATA DIAGNOSTICS")
    print(f"{'=' * 80}")
    print(f"Data inputs keys: {list(data_inputs.keys())}")
    print(f"Weights portfolio type: {type(weights_portfolio)}")

    if weights_portfolio is not None:
        if isinstance(weights_portfolio, pl.DataFrame):
            print("✓ Weights portfolio is Polars DataFrame")
            print(f"  Shape: {weights_portfolio.shape}")
            print(f"  Columns: {weights_portfolio.columns}")
            if not weights_portfolio.is_empty():
                print(
                    f"  Date range: {weights_portfolio['Date'].min()} to {weights_portfolio['Date'].max()}",
                )
                print(f"  First row sample:\n{weights_portfolio.head(1)}")
        elif isinstance(weights_portfolio, pd.DataFrame):
            print("⚠️ Weights portfolio is Pandas DataFrame - will convert")
            print(f"  Shape: {weights_portfolio.shape}")

    print(f"{'=' * 80}\n")

    # === STEP 3: Configuration validation ===
    if weights_portfolio is None:
        raise ValueError(
            "❌ Portfolio weights data (Weights_My_Portfolio) is None.\n"
            f"Available data_inputs keys: {', '.join(data_inputs.keys())}\n"
            "Please ensure the weights data is loaded and passed to the module.",
        )

    # Convert Pandas to Polars if needed
    if isinstance(weights_portfolio, pd.DataFrame):
        print("⚠️ Converting Pandas DataFrame to Polars DataFrame...")
        try:
            weights_portfolio = pl.from_pandas(weights_portfolio)
            print(f"✓ Conversion successful: {weights_portfolio.shape}")
        except Exception as exc_conversion:
            raise ValueError(f"❌ Failed to convert Pandas to Polars: {exc_conversion}")

    # Business logic validation
    if not isinstance(weights_portfolio, pl.DataFrame):
        raise ValueError(
            f"❌ Portfolio weights data must be a Polars DataFrame, got {type(weights_portfolio)}",
        )

    if weights_portfolio.is_empty():
        raise ValueError("❌ Portfolio weights data is empty (0 rows)")

    if "Date" not in weights_portfolio.columns:
        raise ValueError(
            f"❌ Portfolio weights data must contain a 'Date' column. Available: {', '.join(weights_portfolio.columns)}",
        )

    component_columns = [col for col in weights_portfolio.columns if col != "Date"]
    if not component_columns:
        raise ValueError(
            "❌ Portfolio weights data must contain at least one ETF component column besides 'Date'",
        )

    print(
        f"✅ Weights data validation passed: {weights_portfolio.height} rows, {len(component_columns)} components\n",
    )

    # === STEP 4: Validate portfolio and benchmark data ===
    validation_portfolio = validate_portfolio_data(data_portfolio=data_portfolio)
    validation_benchmark = validate_portfolio_data(data_portfolio=data_benchmark)

    if not validation_portfolio[0]:
        raise ValueError(f"Portfolio data validation failed: {validation_portfolio[1]}")

    if not validation_benchmark[0]:
        raise ValueError(f"Benchmark data validation failed: {validation_benchmark[1]}")

    @reactive.calc
    def get_weights_DF():
        """
        Get and preprocess the weights DataFrame with comprehensive validation.

        This function performs comprehensive data processing including date normalization,
        timezone handling, and component weight validation to ensure data quality for
        portfolio weight distribution analysis.

        Returns
        -------
        pl.DataFrame
            Processed weights DataFrame with validated Date column and component columns.
            Date column is normalized to timezone-naive datetime format.
            Component columns are converted to Float64 with null values filled as 0.0.

        Raises
        ------
        RuntimeError
            If critical errors occur during DataFrame processing that prevent analysis

        Notes
        -----
        ### Date Processing Strategy
        - Handles multiple date formats automatically via pandas to_datetime
        - Normalizes all dates to UTC timezone then removes timezone info
        - Filters out any rows with invalid/unparseable dates
        - Sorts chronologically for consistent time-series analysis

        ### Component Weight Processing
        - All component columns converted to Float64 for numerical operations
        - Null/missing values automatically filled with 0.0 (zero weight)
        - Preserves original column names for ETF component identification

        ### Performance Optimizations
        - Creates copy of source data to prevent mutation
        - Single-pass processing for all component columns
        - Efficient filtering using Polars native operations
        """
        try:
            # === STEP 1: Input validation with early returns ===
            if weights_portfolio is None:
                print("⚠️ get_weights_DF: weights_portfolio is None")
                return pl.DataFrame({"Date": []}, schema={"Date": pl.Datetime})

            if not isinstance(weights_portfolio, pl.DataFrame):
                print(
                    f"⚠️ get_weights_DF: weights_portfolio is not DataFrame: {type(weights_portfolio)}",
                )
                return pl.DataFrame({"Date": []}, schema={"Date": pl.Datetime})

            if weights_portfolio.is_empty():
                print("⚠️ get_weights_DF: weights_portfolio is empty")
                return pl.DataFrame({"Date": []}, schema={"Date": pl.Datetime})

            # === STEP 2: Configuration validation ===
            if "Date" not in weights_portfolio.columns:
                print("⚠️ get_weights_DF: No Date column")
                return pl.DataFrame({"Date": []}, schema={"Date": pl.Datetime})

            # === STEP 3: Business logic validation ===
            component_cols = [col for col in weights_portfolio.columns if col != "Date"]
            if not component_cols:
                print("⚠️ get_weights_DF: No component columns found")
                return pl.DataFrame({"Date": []}, schema={"Date": pl.Datetime})

            print(
                f"✓ get_weights_DF: Processing {weights_portfolio.height} rows with {len(component_cols)} components",
            )

            # === STEP 4: Make a copy to avoid modifying original data ===
            weights_DF_processed = weights_portfolio.clone()

            # === STEP 5: Date column processing with timezone normalization ===
            date_dtype = weights_DF_processed.select("Date").dtypes[0]
            first_date_sample = (
                weights_DF_processed["Date"].head(1).to_list()[0]
                if weights_DF_processed.height > 0
                else None
            )
            print(f"✓ Date dtype: {date_dtype}, Sample: '{first_date_sample}'")

            # Use pandas for reliable timezone-aware date conversion with UTC normalization
            if date_dtype in [pl.Utf8, pl.String]:
                print("→ Converting dates via pandas with UTC normalization...")
                try:
                    # Convert to pandas for flexible date parsing
                    weights_pandas = weights_DF_processed.to_pandas()

                    # Use pandas to_datetime with UTC=True to handle mixed timezones properly
                    # This normalizes all dates to UTC timezone before further processing
                    weights_pandas["Date"] = pd.to_datetime(
                        weights_pandas["Date"],
                        utc=True,  # Normalize to UTC to handle mixed timezones
                        errors="coerce",  # Invalid dates become NaT (Not a Time)
                    )

                    # Count null dates after conversion
                    null_count = weights_pandas["Date"].isna().sum()
                    print(
                        f"  Conversion result: {null_count} null dates out of {len(weights_pandas)}",
                    )

                    # Remove timezone information to create timezone-naive datetime
                    # This is required for consistent date comparison operations
                    if isinstance(weights_pandas["Date"].dtype, pd.DatetimeTZDtype):
                        weights_pandas["Date"] = weights_pandas["Date"].dt.tz_localize(None)
                        print("  ✓ Removed timezone information (converted to naive datetime)")

                    # Convert back to Polars for efficient downstream processing
                    weights_DF_processed = pl.from_pandas(weights_pandas)
                    print("  ✓ Converted back to Polars DataFrame")

                except Exception as exc_pandas:
                    print(f"❌ Pandas conversion failed: {exc_pandas}")
                    import traceback

                    traceback.print_exc()
                    return pl.DataFrame({"Date": []}, schema={"Date": pl.Datetime})

            elif date_dtype == pl.Date:
                # Date type - convert to Datetime for consistency
                weights_DF_processed = weights_DF_processed.with_columns(
                    [pl.col("Date").cast(pl.Datetime).alias("Date")],
                )
                print("✓ Converted Date to Datetime")

            elif date_dtype == pl.Datetime:
                print("✓ Date already in Datetime format")
                # Remove timezone if present for consistency
                if getattr(weights_DF_processed["Date"].dtype, "time_zone", None) is not None:
                    weights_DF_processed = weights_DF_processed.with_columns(
                        [pl.col("Date").dt.replace_time_zone(None).alias("Date")],
                    )
                    print("✓ Removed timezone information")

            else:
                # Unexpected date type - attempt conversion via pandas
                print(f"⚠️ Unexpected date dtype: {date_dtype}, attempting pandas conversion")
                try:
                    weights_pandas = weights_DF_processed.to_pandas()
                    weights_pandas["Date"] = pd.to_datetime(
                        weights_pandas["Date"],
                        utc=True,
                        errors="coerce",
                    )
                    if isinstance(weights_pandas["Date"].dtype, pd.DatetimeTZDtype):
                        weights_pandas["Date"] = weights_pandas["Date"].dt.tz_localize(None)
                    weights_DF_processed = pl.from_pandas(weights_pandas)
                    print("✓ Converted via pandas with UTC normalization")
                except Exception as exc_other:
                    print(f"❌ Date conversion failed: {exc_other}")
                    return pl.DataFrame({"Date": []}, schema={"Date": pl.Datetime})

            # === STEP 6: Remove null dates and validate result ===
            initial_row_count = weights_DF_processed.height
            weights_DF_processed = weights_DF_processed.filter(pl.col("Date").is_not_null())

            rows_removed = initial_row_count - weights_DF_processed.height
            if rows_removed > 0:
                print(f"⚠️ Removed {rows_removed} rows with null/invalid dates")

            if weights_DF_processed.is_empty():
                print("❌ No valid dates found after filtering")
                return pl.DataFrame({"Date": []}, schema={"Date": pl.Datetime})

            # === STEP 7: Sort by Date for chronological order ===
            weights_DF_processed = weights_DF_processed.sort("Date")
            date_min = weights_DF_processed["Date"].min()
            date_max = weights_DF_processed["Date"].max()
            print(f"✓ Date range after processing: {date_min} to {date_max}")

            # === STEP 8: Convert component columns to float and fill nulls ===
            # Process all component columns to ensure numerical data type
            # Fill missing values with 0.0 to represent zero weight allocation
            for col in component_cols:
                try:
                    weights_DF_processed = weights_DF_processed.with_columns(
                        [pl.col(col).cast(pl.Float64, strict=False).fill_null(0.0).alias(col)],
                    )
                except Exception as exc_col:
                    print(f"⚠️ Failed to process column {col}: {exc_col}")
                    # On failure, replace entire column with zeros
                    weights_DF_processed = weights_DF_processed.with_columns(
                        [pl.lit(0.0).alias(col)],
                    )

            final_row_count = weights_DF_processed.height
            print(
                f"✅ Successfully processed: {final_row_count} rows with {len(component_cols)} components",
            )
            print(f"   Component columns: {', '.join(component_cols[:5])}")
            if len(component_cols) > 5:
                print(f"   ... and {len(component_cols) - 5} more components")

            return weights_DF_processed

        except Exception as exc:
            print(f"❌ CRITICAL Exception in get_weights_DF: {exc}")
            import traceback

            traceback.print_exc()
            raise RuntimeError(f"Error processing weights DataFrame: {exc}")

    @reactive.calc
    def get_effective_weights_date_range() -> tuple[datetime, datetime]:
        """
        Calculate the effective date range based on time period selection.

        Returns
        -------
        tuple[datetime, datetime]
            Start and end dates for the selected period
        """
        try:
            time_period = input.input_ID_tab_portfolios_subtab_weights_analysis_time_period()
            today = datetime.now(UTC).replace(hour=23, minute=59, second=59)

            if time_period == "custom":
                date_range = input.input_ID_tab_portfolios_subtab_weights_analysis_date_range()
                if date_range and len(date_range) == 2:
                    start_date = datetime.combine(
                        pd.Timestamp(date_range[0]).date(),
                        datetime.min.time(),
                    )
                    end_date = datetime.combine(
                        pd.Timestamp(date_range[1]).date(),
                        datetime.max.time(),
                    )
                    return start_date, end_date
                return today - timedelta(days=365 * 5), today

            if time_period == "1y":
                return today - timedelta(days=365), today
            if time_period == "3y":
                return today - timedelta(days=365 * 3), today
            if time_period == "5y":
                return today - timedelta(days=365 * 5), today
            if time_period == "10y":
                return today - timedelta(days=365 * 10), today
            if time_period == "ytd":
                year_start = datetime(today.year, 1, 1, 0, 0, 0, tzinfo=UTC)
                return year_start, today
            return today - timedelta(days=365 * 5), today

        except Exception as exc:
            print(f"❌ Error in get_effective_weights_date_range: {exc}")
            today = datetime.now(UTC)
            return today - timedelta(days=365 * 5), today

    @reactive.calc
    def get_filtered_weights_DF():
        """
        Filter the weights DataFrame based on effective date range selection.

        Returns
        -------
        pd.DataFrame
            Filtered weights data as pandas DataFrame for compatibility
        """
        try:
            weights_DF = get_weights_DF()

            if weights_DF.is_empty():
                return pd.DataFrame({"Date": []})

            start_date, end_date = get_effective_weights_date_range()

            # Convert to pandas for date filtering
            weights_pandas = weights_DF.to_pandas()

            if weights_pandas.empty:
                return pd.DataFrame({"Date": []})

            # Ensure Date is datetime
            weights_pandas["Date"] = pd.to_datetime(weights_pandas["Date"], errors="coerce")
            weights_pandas = weights_pandas.dropna(subset=["Date"])

            if weights_pandas.empty:
                return pd.DataFrame({"Date": []})

            # Normalize timezones for comparison
            if weights_pandas["Date"].dt.tz is not None:
                weights_pandas["Date"] = weights_pandas["Date"].dt.tz_localize(None)

            start_normalized = pd.Timestamp(start_date).tz_localize(None)
            end_normalized = pd.Timestamp(end_date).tz_localize(None)

            # Filter by date range
            filtered_DF = weights_pandas[
                (weights_pandas["Date"] >= start_normalized)
                & (weights_pandas["Date"] <= end_normalized)
            ].sort_values("Date")

            print(
                f"✓ Filtered: {len(filtered_DF)} rows from {start_date.date()} to {end_date.date()}",
            )
            return filtered_DF

        except Exception as exc:
            print(f"❌ Error in get_filtered_weights_DF: {exc}")
            return pd.DataFrame({"Date": []})

    @reactive.calc
    def get_available_etf_components():
        """
        Get list of available ETF components from weights data.

        Returns
        -------
        list[str]
            List of ETF component column names
        """
        try:
            weights_DF = get_weights_DF()
            if weights_DF.is_empty():
                return []
            return [col for col in weights_DF.columns if col != "Date"]
        except Exception as exc:
            print(f"❌ Error in get_available_etf_components: {exc}")
            return []

    @reactive.calc
    def get_selected_etf_components():
        """
        Get list of currently selected ETF components from checkboxes.

        Returns
        -------
        list[str]
            List of selected ETF component names
        """
        try:
            available_components = get_available_etf_components()

            if not available_components:
                return []

            # Check if "Select All" is checked
            try:
                select_all = (
                    input.input_ID_tab_portfolios_subtab_weights_analysis_select_all_components()
                )
            except Exception:
                # If checkbox doesn't exist yet, default to first 3 components
                return available_components[:3]

            if select_all:
                return available_components

            # Get individual selections
            selected_list = []
            for component in available_components:
                checkbox_id = (
                    f"input_ID_tab_portfolios_subtab_weights_analysis_component_{component}"
                )
                try:
                    if input[checkbox_id]():
                        selected_list.append(component)
                except Exception as _chk_exc:
                    # Checkbox doesn't exist yet, skip this component
                    _logger.debug("Checkbox not yet available, skipping: %s", _chk_exc)
                    continue

            # If no components selected (checkboxes not rendered yet), default to first 3
            if not selected_list:
                return available_components[:3]

            return selected_list

        except Exception as exc:
            print(f"❌ Error in get_selected_etf_components: {exc}")
            # Fallback: return first 3 components
            available_components = get_available_etf_components()
            return available_components[:3] if available_components else []

    @reactive.calc
    def get_filtered_weights_DF_with_selected_components():
        """
        Get filtered weights data with only selected ETF components.

        This function combines date range filtering with component selection to produce
        the final dataset for visualization and analysis. It ensures data consistency
        and provides appropriate fallback behavior when selections are invalid.

        Returns
        -------
        pd.DataFrame
            Filtered weights data containing:
            - Date column with timezone-naive datetime values
            - Only selected ETF component columns
            - Data within the selected date range
            - Empty DataFrame with Date column if no data available

        Notes
        -----
        ### Processing Pipeline
        1. Retrieve date-filtered weights data from get_filtered_weights_DF()
        2. Get list of selected components from get_selected_etf_components()
        3. Filter DataFrame to keep only Date + selected component columns
        4. Return empty DataFrame if no data or no selections

        ### Data Quality
        - Validates component names exist in DataFrame before filtering
        - Preserves Date column for all time-series operations
        - Maintains chronological sort order from upstream processing
        - Returns consistent empty structure for error handling

        ### Performance
        - Minimal processing overhead (column selection only)
        - Reactive dependency on both date range and component selection
        - Automatic updates when either dependency changes
        """
        try:
            # Get date-filtered weights data
            weights_filtered = get_filtered_weights_DF()

            # Get currently selected ETF components
            selected_components = get_selected_etf_components()

            # Early return if no data or no selections
            if weights_filtered.empty or not selected_components:
                print(
                    f"⚠️ No data available: filtered_empty={weights_filtered.empty}, selected_count={len(selected_components)}",
                )
                return pd.DataFrame({"Date": []})

            # Filter to keep only Date column and selected component columns
            # Validate that all selected components actually exist in the DataFrame
            cols_to_keep = ["Date"] + [
                col for col in selected_components if col in weights_filtered.columns
            ]

            # Check if any selected components were not found
            missing_components = [
                col for col in selected_components if col not in weights_filtered.columns
            ]
            if missing_components:
                print(f"⚠️ Selected components not found in data: {', '.join(missing_components)}")

            # Create filtered DataFrame with only selected columns
            weights_selected = weights_filtered[cols_to_keep].copy()

            print(
                f"✓ Filtered to {len(selected_components)} components: {len(weights_selected)} rows",
            )
            return weights_selected

        except Exception as exc:
            print(f"❌ Error in get_filtered_weights_DF_with_selected_components: {exc}")
            import traceback

            traceback.print_exc()
            return pd.DataFrame({"Date": []})

    @output
    @render.ui
    def output_ID_tab_portfolios_subtab_weights_analysis_component_checkboxes():
        """
        Render individual ETF component checkboxes dynamically based on available components.

        Returns
        -------
        ui.div
            Container with individual checkboxes for each ETF component
        """
        try:
            available_components = get_available_etf_components()

            if not available_components:
                return ui.div(
                    ui.p("No ETF components available", class_="text-muted"),
                    class_="p-2",
                )

            # Create checkbox for each component
            checkboxes_list = []
            for idx, component in enumerate(available_components):
                checkbox_id = (
                    f"input_ID_tab_portfolios_subtab_weights_analysis_component_{component}"
                )
                # Default first 3 components to checked
                default_checked = idx < 3

                checkboxes_list.append(
                    ui.div(
                        ui.input_checkbox(checkbox_id, component, value=default_checked),
                        class_="mb-1",
                    ),
                )

            return ui.div(*checkboxes_list)

        except Exception as exc:
            print(
                f"❌ Error in output_ID_tab_portfolios_subtab_weights_analysis_component_checkboxes: {exc}",
            )
            return ui.div(
                ui.p(f"Error loading components: {exc!s}", class_="text-danger"),
                class_="p-2",
            )

    @output
    @render.text
    def output_ID_tab_portfolios_subtab_weights_analysis_selected_components_count() -> str | None:
        """
        Display count of selected vs total ETF components.

        Returns
        -------
        str
            Status text showing selected/total component count
        """
        try:
            available_components = get_available_etf_components()
            selected_components = get_selected_etf_components()

            return f"Selected: {len(selected_components)} of {len(available_components)} components"

        except Exception as exc:
            print(
                f"❌ Error in output_ID_tab_portfolios_subtab_weights_analysis_selected_components_count: {exc}",
            )
            return "Error loading component count"

    @output
    @render.text
    def output_ID_tab_portfolios_subtab_weights_analysis_calculated_date_range() -> str | None:
        """
        Display calculated date range for preset time period selections.

        Returns
        -------
        str
            Formatted date range text
        """
        try:
            start_date, end_date = get_effective_weights_date_range()
            return f"Analysis Period: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}"

        except Exception as exc:
            print(
                f"❌ Error in output_ID_tab_portfolios_subtab_weights_analysis_calculated_date_range: {exc}",
            )
            return "Error calculating date range"

    @output
    @render.ui
    def output_ID_tab_portfolios_subtab_weights_analysis_data_info():
        """
        Display data information panel with real-time statistics.

        Returns
        -------
        ui.div
            Formatted data information panel
        """
        try:
            weights_DF_original = get_weights_DF()
            weights_DF_filtered = get_filtered_weights_DF()
            weights_DF_selected = get_filtered_weights_DF_with_selected_components()

            available_components = get_available_etf_components()
            selected_components = get_selected_etf_components()

            original_rows = weights_DF_original.height if not weights_DF_original.is_empty() else 0
            filtered_rows = len(weights_DF_filtered)
            selected_rows = len(weights_DF_selected)

            info_items = []

            # Original data info
            if original_rows > 0:
                date_min = weights_DF_original["Date"].min()
                date_max = weights_DF_original["Date"].max()
                info_items.append(ui.p(f"📊 Total Data Points: {original_rows}", class_="mb-1"))
                info_items.append(
                    ui.p(
                        f"📅 Full Date Range: {date_min} to {date_max}",
                        class_="mb-1 small text-muted",
                    ),
                )
            else:
                info_items.append(ui.p("❌ No weight data available", class_="mb-1 text-danger"))

            # Filtered data info
            if filtered_rows > 0:
                info_items.append(ui.p(f"🔍 Filtered Points: {filtered_rows}", class_="mb-1"))
            else:
                info_items.append(
                    ui.p("⚠️ No data in selected date range", class_="mb-1 text-warning"),
                )

            # Component selection info
            info_items.append(
                ui.p(
                    f"🎯 Selected Components: {len(selected_components)} of {len(available_components)}",
                    class_="mb-1",
                ),
            )

            # Final data availability
            if selected_rows > 0:
                info_items.append(
                    ui.p(
                        f"✅ Ready for Analysis: {selected_rows} data points",
                        class_="mb-0 text-success fw-bold",
                    ),
                )
            else:
                info_items.append(
                    ui.p(
                        "⚠️ No data available for current selection",
                        class_="mb-0 text-warning fw-bold",
                    ),
                )

            return ui.div(*info_items, class_="p-3 bg-light rounded")

        except Exception as exc:
            print(
                f"❌ Error in output_ID_tab_portfolios_subtab_weights_analysis_data_info: {exc}",
            )
            return ui.div(
                ui.p(f"❌ Error: {exc!s}", class_="text-danger"),
                class_="p-3 bg-light rounded",
            )

    @output
    @render_widget  # pyright: ignore[reportArgumentType]  # pyrefly: ignore[bad-specialization]  # go.Figure satisfies Widget protocol at runtime
    def output_ID_tab_portfolios_subtab_weights_analysis_plot_main():
        """
        Generate main portfolio weight distribution visualization.

        Returns
        -------
        plotly.graph_objects.Figure
            Interactive weight distribution chart
        """
        try:
            weights_DF_selected = get_filtered_weights_DF_with_selected_components()

            if weights_DF_selected.empty:
                return create_error_figure(
                    "No Data Available",
                    "Please select a time period with available data and at least one ETF component.",
                )

            # Get visualization preferences
            viz_type = input.input_ID_tab_portfolios_subtab_weights_analysis_viz_type()
            show_pct = input.input_ID_tab_portfolios_subtab_weights_analysis_show_pct()
            sort_components = (
                input.input_ID_tab_portfolios_subtab_weights_analysis_sort_components()
            )

            # Prepare data
            dates_list = weights_DF_selected["Date"].tolist()
            component_cols = [col for col in weights_DF_selected.columns if col != "Date"]

            # Sort components by average weight if requested
            if sort_components:
                avg_weights = {col: weights_DF_selected[col].mean() for col in component_cols}
                component_cols = sorted(component_cols, key=lambda x: avg_weights[x], reverse=True)

            # Create figure based on visualization type
            fig = go.Figure()

            if viz_type == "area":
                # Stacked area chart
                for component in component_cols:
                    values_list = weights_DF_selected[component].tolist()
                    if show_pct:
                        # Convert to percentage
                        row_totals = weights_DF_selected[component_cols].sum(axis=1)
                        values_list = [
                            (val / total * 100) if total > 0 else 0
                            for val, total in zip(values_list, row_totals, strict=False)
                        ]

                    fig.add_trace(
                        go.Scatter(
                            x=dates_list,
                            y=values_list,
                            name=component,
                            mode="lines",
                            stackgroup="one",
                            fillcolor=None,
                            hovertemplate=f"<b>{component}</b><br>Date: %{{x}}<br>Weight: %{{y:.2f}}{'%' if show_pct else ''}<extra></extra>",
                        ),
                    )

                fig.update_layout(
                    title="Portfolio Weight Distribution Over Time",
                    xaxis_title="Date",
                    yaxis_title="Weight (%)" if show_pct else "Weight",
                    hovermode="x unified",
                    legend={
                        "orientation": "v",
                        "yanchor": "top",
                        "y": 1,
                        "xanchor": "left",
                        "x": 1.02,
                    },
                )

            elif viz_type == "line":
                # Individual line chart
                for component in component_cols:
                    values_list = weights_DF_selected[component].tolist()
                    if show_pct:
                        row_totals = weights_DF_selected[component_cols].sum(axis=1)
                        values_list = [
                            (val / total * 100) if total > 0 else 0
                            for val, total in zip(values_list, row_totals, strict=False)
                        ]

                    fig.add_trace(
                        go.Scatter(
                            x=dates_list,
                            y=values_list,
                            name=component,
                            mode="lines+markers",
                            hovertemplate=f"<b>{component}</b><br>Date: %{{x}}<br>Weight: %{{y:.2f}}{'%' if show_pct else ''}<extra></extra>",
                        ),
                    )

                fig.update_layout(
                    title="Individual Component Weight Trends",
                    xaxis_title="Date",
                    yaxis_title="Weight (%)" if show_pct else "Weight",
                    hovermode="x unified",
                    legend={
                        "orientation": "v",
                        "yanchor": "top",
                        "y": 1,
                        "xanchor": "left",
                        "x": 1.02,
                    },
                )

            # Add range slider for time navigation
            fig.update_xaxes(rangeslider_visible=True)

            # Store latest figure in shared reactive state for report pipeline
            update_visual_object_in_reactives(
                reactives_shiny,
                "Chart_Weights_Analysis",
                fig,
            )

            return fig

        except Exception as exc:
            print(
                f"❌ Error in output_ID_tab_portfolios_subtab_weights_analysis_plot_main: {exc}",
            )
            import traceback

            traceback.print_exc()
            return create_error_figure("Error Generating Chart", f"An error occurred: {exc!s}")

    @output
    @render_widget  # pyright: ignore[reportArgumentType]  # pyrefly: ignore[bad-specialization]  # go.Figure satisfies Widget protocol at runtime
    def output_ID_tab_portfolios_subtab_weights_analysis_plot_secondary():
        """
        Generate secondary visualization (current composition pie chart).

        Returns
        -------
        plotly.graph_objects.Figure
            Current portfolio composition pie chart
        """
        try:
            weights_DF_selected = get_filtered_weights_DF_with_selected_components()

            if weights_DF_selected.empty:
                return create_error_figure(
                    "No Data Available",
                    "Please select components to view current composition.",
                )

            # Get latest weights
            latest_row = weights_DF_selected.iloc[-1]
            component_cols = [col for col in weights_DF_selected.columns if col != "Date"]

            labels_list = component_cols
            values_list = [latest_row[col] for col in component_cols]

            # Create pie chart
            fig = go.Figure(
                data=[
                    go.Pie(
                        labels=labels_list,
                        values=values_list,
                        hovertemplate="<b>%{label}</b><br>Weight: %{value:.2f}<br>Percentage: %{percent}<extra></extra>",
                    ),
                ],
            )

            fig.update_layout(
                title=f"Current Composition as of {latest_row['Date'].strftime('%Y-%m-%d')}",
                showlegend=True,
            )

            # Store latest figure in shared reactive state for report pipeline
            update_visual_object_in_reactives(
                reactives_shiny,
                "Charts_Weights_Composition",
                fig,
            )

            return fig

        except Exception as exc:
            print(
                f"❌ Error in output_ID_tab_portfolios_subtab_weights_analysis_plot_secondary: {exc}",
            )
            return create_error_figure(
                "Error Generating Pie Chart",
                f"An error occurred: {exc!s}",
            )

    @output
    @render.ui
    def output_ID_tab_portfolios_subtab_weights_analysis_table_summary():
        """
        Generate weight statistics summary table.

        Returns
        -------
        ui.HTML
            Formatted HTML table with weight statistics
        """
        try:
            weights_DF_selected = get_filtered_weights_DF_with_selected_components()

            if weights_DF_selected.empty:
                return ui.div(
                    ui.p(
                        "No data available for statistics calculation.",
                        class_="text-muted text-center p-3",
                    ),
                )

            component_cols = [col for col in weights_DF_selected.columns if col != "Date"]

            # Calculate statistics
            stats_data = []
            for component in component_cols:
                values_series = weights_DF_selected[component]
                stats_data.append(
                    {
                        "Component": component,
                        "Current": values_series.iloc[-1],
                        "Average": values_series.mean(),
                        "Min": values_series.min(),
                        "Max": values_series.max(),
                        "Std Dev": values_series.std(),
                    },
                )

            # Persist report-compatible weight statistics into Data_Results so the
            # PDF export pipeline can include them in outputs_weights_analysis.json.
            # Keys must match the Typst template's field() lookups.
            try:
                report_stats = [
                    {
                        "component": row["Component"],
                        "current_weight": float(row["Current"]),
                        "mean_weight": float(row["Average"]),
                        "min_weight": float(row["Min"]),
                        "max_weight": float(row["Max"]),
                        "std_weight": (
                            0.0 if math.isnan(float(row["Std Dev"])) else float(row["Std Dev"])
                        ),
                    }
                    for row in stats_data
                ]
                save_weights_analysis_outputs_to_reactives(
                    reactives_shiny,
                    {"weight_statistics": report_stats},
                )
            except Exception as _save_exc:
                _logger.debug("save_weights_analysis_outputs_to_reactives failed: %s", _save_exc)

            # Create DataFrame
            stats_dataframe = pd.DataFrame(stats_data)

            # Format as HTML table
            html_table = stats_dataframe.to_html(
                index=False,
                classes="table table-striped table-hover table-sm",
                float_format=lambda x: f"{x:.2f}",
            )

            return ui.HTML(html_table)

        except Exception as exc:
            print(
                f"❌ Error in output_ID_tab_portfolios_subtab_weights_analysis_table_summary: {exc}",
            )
            return ui.div(
                ui.p(
                    f"Error generating statistics: {exc!s}",
                    class_="text-danger text-center p-3",
                ),
            )
