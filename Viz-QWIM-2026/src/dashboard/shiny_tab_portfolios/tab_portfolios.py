"""Portfolios Tab Module.

Provides the main portfolios tab functionality for the QWIM Dashboard,
including portfolio analysis, comparison, and Weights Analysis.

## Overview

The Portfolios tab serves as the central hub for portfolio-related functionality within
the QWIM Dashboard. It coordinates three specialized subtabs that provide comprehensive
portfolio analysis capabilities.

## Features

- **Portfolio Performance Analysis**: Detailed analysis of individual portfolio metrics
- **Portfolio vs Benchmark Comparison**: Side-by-side comparison with benchmark portfolios
- **Weights Analysis Visualization**: Interactive visualization of portfolio component weights

## Architecture

The module follows a modular architecture pattern where:

- `tab_portfolios_ui()`: Orchestrates the UI components from specialized subtab modules
- `tab_portfolios_server()`: Coordinates server logic across all portfolio subtabs
- Each subtab handles its own specialized functionality independently

## Configuration

The module behavior is configured through the initialized constants.

## Dependencies

### Internal Dependencies
- `subtab_portfolios_analysis`: Portfolio analysis functionality
- `subtab_portfolios_comparison`: Portfolio comparison functionality
- `subtab_weights_analysis`: Weights Analysis functionality
- `utils_visuals`: Visual utilities for charts and tables

### External Dependencies
- `shiny`: Web application framework for UI and server components
- `polars`: Data manipulation and analysis
- `plotly`: Interactive plotting library
- `pathlib`: Object-oriented filesystem paths

## Usage Example

```python
# In main dashboard application
from typing import Any

from src.dashboard.Shiny_Tab_Portfolios.tab_portfolios import (
    tab_portfolios_ui,
    tab_portfolios_server,
)

# UI setup
portfolios_ui = tab_portfolios_ui(data_utils=data_utils_config, data_inputs=portfolio_data_inputs)

# Server setup
portfolios_server = tab_portfolios_server(
    input=input,
    output=output,
    session=session,
    data_utils=data_utils_config,
    data_inputs=portfolio_data_inputs,
    reactives_shiny=reactive_state,
)
```

## Notes

!!! warning "Performance Considerations"
    Large portfolio datasets may impact performance. Consider data pagination
    or lazy loading for datasets with >10,000 data points.

## Version History

- Initial implementation: Portfolio analysis and comparison functionality
- Enhanced: Added Weights Analysis visualization
- Current: Modular architecture with coordinated subtabs
"""

from __future__ import annotations

import typing

from pathlib import Path
from typing import Any

# Shiny framework imports for web application functionality
from shiny import module, ui

from src.utils.custom_exceptions_errors_loggers.logger_custom import get_logger

# Specialized subtab module imports for portfolio functionality
from .subtab_portfolios_analysis import (
    subtab_portfolios_analysis_server,
    subtab_portfolios_analysis_ui,
)
from .subtab_portfolios_comparison import (
    subtab_portfolios_comparison_server,
    subtab_portfolios_comparison_ui,
)
from .subtab_portfolios_skfolio import (
    subtab_portfolios_skfolio_server,
    subtab_portfolios_skfolio_ui,
)
from .subtab_weights_analysis import (
    subtab_weights_analysis_server,
    subtab_weights_analysis_ui,
)


#: Module-level logger instance
_logger = get_logger(__name__)

# =============================================================================
# Constants and Configuration
# =============================================================================

#: Path to the main output directory for generated files
OUTPUT_DIR = Path("output")


# =============================================================================
# UI Components
# =============================================================================


@module.ui
def tab_portfolios_ui(data_utils: dict, data_inputs: dict) -> Any:
    """
    Create the UI for the Portfolios tab.

    This function orchestrates the user interface for the main Portfolios tab by
    coordinating specialized subtab UI components. It creates a tabbed interface
    that allows users to navigate between different portfolio analysis views.

    ## UI Structure

    The UI is organized as a navigation tab set with three main subtabs:

    1. **Portfolio Analysis**: Individual portfolio performance metrics
    2. **Portfolio Comparison**: Side-by-side portfolio vs benchmark analysis
    3. **Weights Analysis**: Interactive visualization of portfolio weights

    ## Parameters

    Args:
        data_utils (dict): Configuration dictionary containing utility settings.
            Expected to include visualization preferences, export settings,
            and other UI configuration parameters.

        data_inputs (dict): Dictionary containing portfolio data inputs.
            Expected keys include:
            - `My_Portfolio`: Primary portfolio data (DataFrame)
            - `Benchmark_Portfolio`: Benchmark comparison data (DataFrame)
            - `Weights_My_Portfolio`: Portfolio weights data (DataFrame)
            - `Time_Series_ETFs`: ETF time series data (DataFrame)

    ## Returns

    Returns
    -------
        shiny.ui.NavSetTab: A Shiny navigation tab set containing all portfolio
        subtab UI components organized in a tabbed interface.

    ## UI Component Details

    ### Tab Structure
    Each subtab is created with:
    - Descriptive tab title for user navigation
    - Unique module ID following project naming conventions
    - Passed-through data utilities and inputs for consistency

    ### Naming Convention
    All subtab IDs follow the pattern: `ID_tab_portfolios_subtab_{subtab_name}`

    ## Example Usage

    ```python
    # Create portfolios tab UI with sample data
    portfolios_ui = tab_portfolios_ui(
        data_utils={"theme": "default", "export_enabled": True},
        data_inputs={
            "My_Portfolio": portfolio_df,
            "Benchmark_Portfolio": benchmark_df,
            "Weights_My_Portfolio": weights_df,
            "Time_Series_ETFs": etf_df,
        },
    )
    ```

    ## Notes

    !!! info "Data Validation"
        Individual subtab modules are responsible for validating their
        specific data requirements. This function performs no data validation.

    !!! tip "Extensibility"
        To add new subtabs, simply add new `ui.nav_panel()` entries to the
        `tab_panels` list with appropriate subtab UI function calls.
    """
    # Create individual tab panels for each portfolio analysis subtab
    # Each panel contains the UI from its respective specialized module
    tab_panels = [
        ui.nav_panel(
            "Portfolio Analysis",
            subtab_portfolios_analysis_ui(
                id="ID_tab_portfolios_subtab_portfolios_analysis",  # pyright: ignore[reportCallIssue]  # pyrefly: ignore[unexpected-keyword,bad-argument-count]
                data_utils=data_utils,
                data_inputs=data_inputs,
            ),
        ),
        ui.nav_panel(
            "Portfolio Comparison",
            subtab_portfolios_comparison_ui(
                id="ID_tab_portfolios_subtab_portfolios_comparison",  # pyright: ignore[reportCallIssue]  # pyrefly: ignore[unexpected-keyword,bad-argument-count]
                data_utils=data_utils,
                data_inputs=data_inputs,
            ),
        ),
        ui.nav_panel(
            "Weights Analysis",
            subtab_weights_analysis_ui(
                id="ID_tab_portfolios_subtab_weights_analysis",  # pyright: ignore[reportCallIssue]  # pyrefly: ignore[unexpected-keyword,bad-argument-count]
                data_utils=data_utils,
                data_inputs=data_inputs,
            ),
        ),
        ui.nav_panel(
            "skfolio Optimization",
            subtab_portfolios_skfolio_ui(
                id="ID_tab_portfolios_subtab_skfolio",  # pyright: ignore[reportCallIssue]  # pyrefly: ignore[unexpected-keyword,bad-argument-count]
                data_utils=data_utils,
                data_inputs=data_inputs,
            ),
        ),
    ]

    # Return organized navigation tab set with unique identifier
    return ui.navset_tab(
        *tab_panels,  # Unpack all tab panels into the navigation set
        id="ID_tab_portfolios_tabs_all",
    )  # Unique ID for the entire tab set


# =============================================================================
# Server Logic
# =============================================================================


@module.server
def tab_portfolios_server(
    input: typing.Any,
    output: typing.Any,
    session: typing.Any,
    data_utils: dict,
    data_inputs: dict,
    reactives_shiny: dict,
) -> dict | None:
    """
    Server logic for the portfolios tab.

    This function coordinates the server-side logic for all portfolio subtabs,
    ensuring proper initialization and coordination between specialized modules.
    It acts as the central orchestrator for portfolio-related server functionality.

    ## Server Architecture

    The server follows a modular coordination pattern where:

    1. **Module Independence**: Each subtab server operates independently
    2. **Centralized Coordination**: This function manages initialization and communication
    3. **Resource Sharing**: Common data and utilities are shared across all modules
    4. **State Management**: Reactive state is coordinated across all subtabs

    ## Parameters

    Args:
        input (typing.Any): Shiny input object containing user interface inputs.
            Provides access to reactive user input values across all subtabs.

        output (typing.Any): Shiny output object for rendering UI components.
            Used by subtab servers to render their specific output elements.

        session (typing.Any): Shiny session object for session management.
            Handles session state, user authentication, and communication.

        data_utils (dict): Configuration dictionary containing utility settings.
            Shared configuration passed to all subtab servers for consistency.
            Expected to include:
            - Theme and styling preferences
            - Export configuration settings
            - Performance optimization flags

        data_inputs (dict): Dictionary containing all portfolio data inputs.
            Core data shared across all subtab servers. Expected structure:
            ```python
            {
                "My_Portfolio": pl.DataFrame,  # Primary portfolio data
                "Benchmark_Portfolio": pl.DataFrame,  # Benchmark comparison data
                "Weights_My_Portfolio": pl.DataFrame,  # Portfolio component weights
                "Time_Series_ETFs": pl.DataFrame,  # ETF time series data
            }
            ```

        reactives_shiny (dict): Dictionary of reactive variables and state.
            Manages shared reactive state across all portfolio subtabs.
            Enables coordination and communication between modules.

    ## Returns

    Returns
    -------
        dict: Dictionary containing server instances from all portfolio subtabs.
        Structure:
        ```python
        {
            "Analysis_Server": analysis_server_instance,
            "Comparison_Server": comparison_server_instance,
            "Weights_Server": weights_server_instance,
        }
        ```

    ## Server Coordination Details

    ### Initialization Order
    1. Portfolio Analysis server (primary functionality)
    2. Portfolio Comparison server (depends on analysis metrics)
    3. Weights Analysis server (visualization component)

    ### Data Flow
    - All servers receive identical `data_inputs` for consistency
    - Shared `reactives_shiny` enables cross-module communication
    - Individual servers handle their own data validation and processing

    ### Error Handling
    Each subtab server is responsible for its own error handling. This coordination
    function will propagate any initialization errors from individual servers.

    ## Example Usage

    ```python
    # Initialize portfolios tab server coordination
    server_instances = tab_portfolios_server(
        input=shiny_input,
        output=shiny_output,
        session=shiny_session,
        data_utils=configuration_dict,
        data_inputs=portfolio_data_dict,
        reactives_shiny=reactive_state_dict,
    )

    # Access individual server instances
    analysis_server = server_instances["Analysis_Server"]
    comparison_server = server_instances["Comparison_Server"]
    weights_server = server_instances["Weights_Server"]
    ```

    ## Notes

    !!! warning "Module Dependencies"
        Server initialization order may be important if modules have dependencies.
        Currently, all modules are initialized independently.

    !!! info "Resource Management"
        Large datasets in `data_inputs` are shared by reference across all servers.
        Consider memory usage when processing large portfolio datasets.

    !!! tip "Debugging"
        Individual server instances can be accessed from the returned dictionary
        for debugging and testing purposes.

    ## Error Scenarios

    The function may raise exceptions in the following scenarios:

    - **ImportError**: If subtab modules cannot be imported
    - **TypeError**: If required parameters have incorrect types
    - **ValueError**: If data_inputs contains invalid data structures
    - **RuntimeError**: If individual subtab servers fail to initialize

    ## Performance Considerations

    - Server initialization is sequential, not parallel
    - Large datasets in `data_inputs` are shared by reference (memory efficient)
    - Individual servers handle their own performance optimization
    """
    # Initialize Portfolio Analysis server
    # This handles individual portfolio performance metrics and analysis
    subtab_portfolios_analysis_server_instance = subtab_portfolios_analysis_server(
        id="ID_tab_portfolios_subtab_portfolios_analysis",  # pyright: ignore[reportCallIssue]  # pyrefly: ignore[unexpected-keyword,bad-argument-count]
        data_utils=data_utils,
        data_inputs=data_inputs,
        reactives_shiny=reactives_shiny,
    )

    # Initialize Portfolio Comparison server
    # This handles side-by-side portfolio vs benchmark comparison functionality
    subtab_portfolios_comparison_server_instance = subtab_portfolios_comparison_server(
        id="ID_tab_portfolios_subtab_portfolios_comparison",  # pyright: ignore[reportCallIssue]  # pyrefly: ignore[unexpected-keyword,bad-argument-count]
        data_utils=data_utils,
        data_inputs=data_inputs,
        reactives_shiny=reactives_shiny,
    )

    # Initialize Weights Analysis server
    # This handles interactive visualization of portfolio component weights
    subtab_weights_analysis_server_instance = subtab_weights_analysis_server(
        id="ID_tab_portfolios_subtab_weights_analysis",  # pyright: ignore[reportCallIssue]  # pyrefly: ignore[unexpected-keyword,bad-argument-count]
        data_utils=data_utils,
        data_inputs=data_inputs,
        reactives_shiny=reactives_shiny,
    )

    # Initialize skfolio Optimization server
    # This handles portfolio optimization comparison using skfolio methods
    subtab_portfolios_skfolio_server_instance = subtab_portfolios_skfolio_server(
        id="ID_tab_portfolios_subtab_skfolio",  # pyright: ignore[reportCallIssue]  # pyrefly: ignore[unexpected-keyword,bad-argument-count]
        data_utils=data_utils,
        data_inputs=data_inputs,
        reactives_shiny=reactives_shiny,
    )

    # Return all server instances in a structured format for external access
    # This enables testing, debugging, and potential future coordination needs
    return {
        "Portfolios_Analysis_Server": subtab_portfolios_analysis_server_instance,
        "Portfolios_Comparison_Server": subtab_portfolios_comparison_server_instance,
        "Weights_Analysis_Server": subtab_weights_analysis_server_instance,
        "skfolio_Optimization_Server": subtab_portfolios_skfolio_server_instance,
    }
