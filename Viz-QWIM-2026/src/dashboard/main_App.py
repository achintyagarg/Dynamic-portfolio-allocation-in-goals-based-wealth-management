"""QWIM Dashboard Main Application.

===============================

![QWIM Dashboard](https://img.shields.io/badge/QWIM-Dashboard-blue)
![Python](https://img.shields.io/badge/python-3.12+-green)
![Shiny](https://img.shields.io/badge/shiny-for_python-orange)

## Overview

The QWIM Dashboard is a comprehensive time series analysis tool built with Python and Shiny
for visualizing and analyzing multiple time series data with focus on portfolio management
and financial analysis.

## Architecture

The dashboard follows a modular architecture with clear separation of concerns:

```mermaid
graph TD
    A[main_App.py] --> B[Shiny_Tab_Portfolios]
    A --> C[Shiny_Tab_Clients]
    A --> D[Shiny_Utils]
    B --> E[SubTab_Portfolios_Analysis]
    B --> F[SubTab_Portfolios_Comparison]
    B --> G[subtab_weights_analysis]
    C --> H[SubTab_Personal_Info]
    C --> I[SubTab_Assets]
    C --> J[SubTab_Goals]
    D --> K[reactives_shiny]
    D --> L[utils_data]
    D --> M[utils_visuals]
```

## Features

!!! success "Core Features"
    - **Interactive Portfolio Analysis**: Real-time portfolio weight distribution visualization
    - **Time Series Analysis**: Comprehensive time series analysis with multiple visualization types
    - **Component Selection**: Dynamic ETF component selection with checkbox interface
    - **Statistical Analysis**: Advanced statistical metrics and correlation analysis
    - **Export Capabilities**: PNG export for plots and tables
    - **Responsive Design**: Mobile-friendly dashboard with Bootstrap themes

!!! info "Visualization Types"
    - Stacked Area Charts
    - Line Charts
    - Bar Charts
    - Heatmaps
    - Pie Charts for composition analysis

## Technology Stack

=== "Frontend"
    - **Shiny for Python**: Interactive web framework
    - **Bootstrap**: Responsive CSS framework via shinyswatch
    - **Plotly**: Interactive visualizations
    - **HTML/CSS**: Custom styling

=== "Backend"
    - **Python 3.12+**: Core programming language
    - **Polars**: High-performance DataFrame operations
    - **Pandas**: Data manipulation and analysis
    - **NumPy**: Numerical computations

=== "Data Processing"
    - **Time Series Analysis**: Custom time series utilities
    - **Statistical Computing**: NumPy-based calculations
    - **Data Validation**: Defensive programming approach

## Quick Start

### Installation

```bash
# Clone the repository
git clone <repository-url>
cd dashboard-QWIM

# Install dependencies
pip install -r requirements.txt
```

### Running the Application

```bash
# Development mode
python src/dashboard/main_App.py

# Production mode
QWIM_ENVIRONMENT=production python src/dashboard/main_App.py
```

### Access

- **Development**: http://127.0.0.1:8080
- **Production**: http://0.0.0.0:8000

## Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `QWIM_ENVIRONMENT` | `development` | Application environment (`development` or `production`) |

### File Structure

```
src/dashboard/
├── main_App.py                     # Main application entry point
├── Shiny_Tab_Portfolios/           # Portfolio analysis modules
│   ├── Tab_Portfolios.py           # Main portfolio tab
│   ├── SubTab_Portfolios_Analysis.py    # Portfolio analysis subtab
│   ├── SubTab_Portfolios_Comparison.py  # Benchmark comparison subtab
│   └── subtab_weights_analysis.py   # Weight distribution analysis
├── Shiny_Tab_Clients/              # Client/investor modules
│   ├── Tab_Clients.py              # Main clients tab
│   ├── SubTab_Personal_Info.py     # Personal information subtab
│   ├── SubTab_Assets.py            # Assets subtab
│   ├── SubTab_Income.py            # Income subtab
│   ├── SubTab_Goals.py             # Goals subtab
│   └── SubTab_Summary.py           # Summary subtab
└── Shiny_Utils/                    # Utility modules
    ├── reactives_shiny.py          # Reactive value management
    ├── utils_data.py               # Data processing utilities
    ├── utils_visuals.py            # Visualization utilities
    └── utils_errors.py             # Custom exception classes
```

## API Reference

### Classes

#### App
Main Shiny application instance that coordinates all dashboard functionality.

**Attributes:**
- `app_ui`: User interface definition
- `app_server`: Server-side logic

### Functions

#### get_data_inputs()
Load and prepare input data for the dashboard.

**Returns:**
- `dict`: Dictionary containing all input datasets

**Example:**
```python
data_inputs = get_data_inputs(project_dir=project_dir)
portfolio_data = data_inputs.get("Weights_My_Portfolio")
```

#### get_data_utils()
Initialize data utility functions and configurations.

**Parameters:**
- `project_dir` (Path): Root directory of the project

**Returns:**
- `dict`: Dictionary containing utility functions and configurations

#### initialize_reactives_shiny()
Initialize reactive values for Shiny application state management.

**Parameters:**
- `data_utils` (dict): Data utility functions and configurations

**Returns:**
- `dict`: Dictionary containing initialized reactive values

**Raises:**
- `DashboardInitializationError`: If reactive initialization fails

### Server Functions

#### app_server()
Main server function that coordinates all dashboard functionality.

**Parameters:**
- `input`: Shiny input object containing user interactions
- `output`: Shiny output object for rendering results
- `session`: Shiny session object for state management

**Features:**
- Reactive value initialization and management
- Tab navigation tracking
- Modal dialog handling
- Module coordination

## Dashboard Modules

### Portfolio Analysis Module

The portfolio analysis module provides comprehensive tools for analyzing portfolio weights
and performance over time.

**Key Features:**
- Weight distribution visualization
- Component selection interface
- Time period filtering
- Statistical analysis
- Export capabilities

**Usage Example:**
```python
# Access portfolio module in server
portfolios_module_instance = tab_portfolios_server(
    id="ID_tab_portfolios",
    data_utils=data_utils,
    data_inputs=data_inputs,
    reactives_shiny=reactives_shiny,
)
```

### Reactive System

The dashboard uses Shiny's reactive system for real-time updates:

```python
# Reactive value initialization
reactives_shiny = initialize_reactives_shiny(data_utils=data_utils)
```

## Error Handling

The application implements comprehensive error handling following defensive programming principles:

!!! warning "Error Types"
    - `DashboardInitializationError`: Application startup failures
    - `DataLoadingError`: Data loading and processing errors
    - `ModuleInitializationError`: Module-specific initialization errors
    - `SilentInitializationException`: Non-critical initialization warnings

## Performance Considerations

!!! tip "Optimization Strategies"
    - **Data Downsampling**: Automatic downsampling for large datasets (>200 points)
    - **Lazy Loading**: Reactive calculations only when needed
    - **Efficient Data Structures**: Polars for high-performance data operations
    - **Memory Management**: Proper cleanup of temporary variables

## Development Guidelines

### Code Standards

Following the project's coding standards:

- **Naming**: `snake_case` for functions/variables, `CamelCase` for classes
- **Validation**: Input validation with early returns
- **Error Handling**: Defensive programming over try-except
- **Documentation**: Comprehensive docstrings and type hints

### Testing

```bash
# Run tests
python -m pytest tests/

# Run with coverage
python -m pytest tests/ --cov=src
```

## Deployment

### Development Deployment

```bash
# Start development server
python src/dashboard/main_App.py
```

### Production Deployment

```bash
# Set environment
export QWIM_ENVIRONMENT=production

# Start production server
python src/dashboard/main_App.py
```

!!! note "Production Notes"
    - Runs on `0.0.0.0:8000` for external access
    - Debug mode disabled
    - Reload disabled for stability

## Troubleshooting

### Common Issues

??? question "Dashboard won't start"
    - Check Python version (3.11+ required)
    - Verify all dependencies installed
    - Check project directory structure
    - Review error logs in `logs/` directory

??? question "Data not loading"
    - Verify data files exist in expected locations
    - Check file permissions
    - Review data format compatibility

??? question "Visualizations not updating"
    - Check reactive dependencies
    - Verify component selection state
    - Review browser console for JavaScript errors

## Contributing

Please follow the project's coding standards and include:

1. Comprehensive documentation
2. Input validation with early returns
3. Defensive programming practices
4. Type hints for all functions
5. Unit tests for new functionality

## Version History

| Version | Date | Changes |
|---------|------|---------|
| 0.5.1 | 2026-03-01 | Initial release with portfolio analysis |

## License

[Add license information here]

## Support

For support and questions:
- Check the troubleshooting section above
- Review logs in the `logs/` directory
- Contact the QWIM development team
"""

# =============================================================================
# Standard Library Imports
# =============================================================================

from __future__ import annotations

import contextlib
import sys  # MUST be first import to enable UTF-8 encoding fix


# CRITICAL: Force UTF-8 encoding for all I/O on Windows BEFORE any other imports
# This prevents UnicodeEncodeError when logging special characters (❱, ✓, etc.)
# Must execute before importing os, platform, or any third-party packages
if sys.platform == "win32":
    # Import os here only for environment manipulation
    import os as _os_for_encoding

    # Set environment variable for subprocess and future operations
    _os_for_encoding.environ.setdefault("PYTHONIOENCODING", "utf-8")

    # Force UTF-8 for default encoding
    if hasattr(sys, "_enablelegacywindowsfsencoding"):
        # Disable legacy Windows filesystem encoding
        sys._enablelegacywindowsfsencoding = lambda: None

    # Reconfigure all existing standard streams to UTF-8
    if hasattr(sys.stdout, "reconfigure"):
        with contextlib.suppress(Exception):
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # pyright: ignore[reportAttributeAccessIssue]

    if hasattr(sys.stderr, "reconfigure"):
        with contextlib.suppress(Exception):
            sys.stderr.reconfigure(encoding="utf-8", errors="replace")  # pyright: ignore[reportAttributeAccessIssue]

    if hasattr(sys.stdin, "reconfigure"):
        with contextlib.suppress(Exception):
            sys.stdin.reconfigure(encoding="utf-8", errors="replace")  # pyright: ignore[reportAttributeAccessIssue]

    # CRITICAL: Disable tbhandler to prevent UnicodeEncodeError
    # tbhandler uses rich.console which has encoding issues on Windows
    # We'll use our custom exception handling instead
    import io

    # Store original excepthook before any module can replace it
    _original_excepthook = sys.excepthook

    def _utf8_safe_excepthook(
        exc_type: type[BaseException],
        exc_value: BaseException,
        exc_traceback: Any,
    ) -> None:
        """Exception hook that safely handles Unicode characters on Windows."""
        import traceback

        try:
            # Try to format exception with UTF-8 safe output
            lines = traceback.format_exception(exc_type, exc_value, exc_traceback)
            message = "".join(lines)
            # Write to stderr with UTF-8 encoding
            stderr_utf8 = io.TextIOWrapper(
                sys.stderr.buffer,
                encoding="utf-8",
                errors="replace",
                line_buffering=True,
            )
            stderr_utf8.write(message)
            stderr_utf8.flush()
        except Exception:  # noqa: BLE001 - Intentional catch-all for UTF-8 errors
            # Fallback: use original hook
            try:
                _original_excepthook(exc_type, exc_value, exc_traceback)
            except Exception:  # noqa: BLE001 - Last resort error handling
                # Last resort: basic error message
                print(f"Error: {exc_type.__name__}: {exc_value}", file=sys.stderr)

    # Install our safe exception hook
    sys.excepthook = _utf8_safe_excepthook

    # Prevent tbhandler from replacing our excepthook
    # Monkey patch sys to make excepthook read-only for tbhandler
    _sys_setattr = sys.__setattr__  # pyright: ignore[reportAttributeAccessIssue]  # pyrefly: ignore[missing-attribute]

    def _protected_setattr(name: str, value: Any) -> Any:
        # Allow our code to set excepthook, but block tbhandler
        if name == "excepthook":
            import inspect

            frame = inspect.currentframe()
            if frame and frame.f_back:
                caller_file = frame.f_back.f_code.co_filename
                # Only allow our code to modify excepthook
                if "main_App.py" not in caller_file and "tbhandler" in caller_file:
                    return None  # Silently ignore tbhandler's attempt
        return _sys_setattr(name, value)

    # Apply protection (but only for excepthook attribute)
    # Note: This is a workaround to prevent tbhandler from taking over

# Now safe to import other standard library modules
import os  # Operating system interface
import platform  # Access to underlying platform's identifying data
import warnings  # Warning control

from datetime import UTC, datetime  # Date and time handling
from pathlib import Path  # Object-oriented filesystem paths
from typing import Any


# =============================================================================
# Warning Suppression
# =============================================================================
warnings.filterwarnings("ignore", category=FutureWarning, message=".*google.generativeai.*")

# =============================================================================
# Third-Party Imports
# =============================================================================
import shinyswatch  # Bootstrap themes for Shiny applications

from shiny import App, reactive, ui


# =============================================================================
# Project Path Configuration
# =============================================================================

#: Root directory of the QWIM Dashboard project
#: Determined by navigating up from the current file location
project_dir = Path(__file__).parent.parent.parent

# Add project root to Python path for module imports
# This ensures all internal modules can be imported properly
if str(project_dir) not in sys.path:
    sys.path.insert(0, str(project_dir))

# =============================================================================
# Custom Logging and Exception Handling
# =============================================================================

# Custom logging utilities - using loguru-based logger
# Custom exception classes following project coding standards
from src.utils.custom_exceptions_errors_loggers.exception_custom import (
    Exception_Configuration,
    Exception_Severity,
    capture_exception,
)
from src.utils.custom_exceptions_errors_loggers.logger_custom import (
    Performance_Timer,
    get_logger,
    setup_logging,
)


# Initialize logging at module level
setup_logging(
    log_level="INFO",
    environment=os.environ.get("QWIM_ENVIRONMENT", "development"),
    log_dir=Path(__file__).parent.parent.parent / "logs",
    enable_console=True,
    enable_JSON=True,
)

#: Module-level logger instance
_logger = get_logger(__name__)

# =============================================================================
# Internal Module Imports - Utility Functions
# =============================================================================

# Reactive state management utilities
from src.dashboard.shiny_utils.reactives_shiny import (
    initialize_reactives_shiny,
)

# Data processing and utility functions
from src.dashboard.shiny_utils.utils_data import (
    get_data_inputs,
    get_data_utils,
)


# =============================================================================
# Configuration and Environment Setup
# =============================================================================

#: Application environment type (development or production)
#: Controls server configuration, debug settings, and deployment options
#:
#: - development: Local development with debug mode and auto-reload
#: - production: Production deployment with optimized settings
ENVIRONMENT: str = os.environ.get("QWIM_ENVIRONMENT", "development").lower()

# =============================================================================
# Data Initialization
# =============================================================================

_logger.info(
    "Initializing QWIM Dashboard data",
    extra={"environment": ENVIRONMENT, "project_dir": str(project_dir)},
)

#: Dictionary containing data utility functions and configurations
#: Provides access to data processing, validation, and transformation utilities
data_utils: dict[str, Any] = get_data_utils(project_dir=project_dir)

#: Dictionary containing all input datasets for the dashboard
#: Includes portfolio data, benchmark data, weights, and time series information
data_inputs: dict[str, Any] = get_data_inputs(project_dir=project_dir)

# Logs directory is created by setup_logging

# =============================================================================
# Internal Module Imports - Dashboard Components
# =============================================================================

from src.dashboard.shiny_tab_clients.tab_clients import (
    tab_clients_server,
    tab_clients_ui,
)
from src.dashboard.shiny_tab_overview.tab_overview import (
    tab_overview_server,
    tab_overview_ui,
)
from src.dashboard.shiny_tab_portfolios.tab_portfolios import (
    tab_portfolios_server,
    tab_portfolios_ui,
)
from src.dashboard.shiny_tab_products.tab_products import (
    tab_products_server,
    tab_products_ui,
)
from src.dashboard.shiny_tab_results.tab_results import (
    tab_results_server,
    tab_results_ui,
)


_logger.debug("Dashboard components imported successfully")

# =============================================================================
# UI Component Functions
# =============================================================================


def create_about_modal() -> Any:
    """
    Create the About modal dialog with comprehensive application information.

    This modal provides users with detailed information about the dashboard
    including features, technology stack, version details, and platform information.

    ## Modal Content Structure

    The modal is organized into the following sections:

    1. **Features**: Core dashboard capabilities and functionality
    2. **Technology Stack**: Technical components and frameworks used
    3. **Supported Analysis**: Types of analysis available
    4. **Version Information**: Technical details about the current deployment

    ## Returns

    Returns
    -------
        shiny.ui.modal: Configured modal dialog component with comprehensive
        application information formatted using Bootstrap styling.

    ## Usage Example

    ```python
    # Create and display the modal
    about_modal = create_about_modal()
    ui.modal_show(about_modal)
    ```

    ## Modal Features

    - **Responsive Design**: Adapts to different screen sizes
    - **Easy Close**: Can be dismissed by clicking outside or close button
    - **Rich Content**: Includes lists, formatted text, and highlighted sections
    - **Version Info**: Dynamic version and platform information

    ## Styling

    Uses Bootstrap classes for consistent appearance:
    - `bg-light`: Light background for version information section
    - `p-3`: Padding for content spacing
    - `rounded`: Rounded corners for modern appearance
    - `btn-primary`: Primary button styling for close button

    ## Notes

    !!! info "Dynamic Content"
        Version information is generated dynamically based on the current
        Python environment and platform, ensuring accuracy across deployments.

    !!! tip "Customization"
        The modal content can be easily modified to include additional
        sections or updated feature descriptions as the dashboard evolves.
    """
    return ui.modal(
        # Modal header with application title
        ui.h3("About QWIM Dashboard"),
        ui.p(
            "The QWIM Dashboard is a comprehensive time series analysis tool built with Python and Shiny.",
        ),
        # Features section highlighting core capabilities
        ui.h5("🚀 Features:"),
        ui.tags.ul(
            ui.tags.li("Interactive time series visualization with multiple chart types"),
            ui.tags.li("Dynamic ETF component selection and filtering"),
            ui.tags.li("Statistical analysis and portfolio weight distribution"),
            ui.tags.li("Real-time reactive updates and synchronization"),
            ui.tags.li("Customizable time period selection and analysis"),
            ui.tags.li("Export capabilities for plots and statistical tables"),
        ),
        # Technology stack section with technical details
        ui.h5("🛠️ Technology Stack:"),
        ui.tags.ul(
            ui.tags.li("Python 3.12+ for core functionality"),
            ui.tags.li("Shiny for Python for interactive dashboard framework"),
            ui.tags.li("Plotly for interactive and responsive visualizations"),
            ui.tags.li("Polars for high-performance time series processing"),
            ui.tags.li("NumPy and Pandas for numerical computations"),
            ui.tags.li("Bootstrap themes via shinyswatch for responsive design"),
        ),
        # Analysis capabilities section
        ui.h5("📊 Supported Analysis:"),
        ui.tags.ul(
            ui.tags.li("Portfolio weight distribution over time"),
            ui.tags.li("Component-wise statistical analysis"),
            ui.tags.li("Time period filtering and comparison"),
            ui.tags.li("Multiple visualization formats (area, line, bar, heatmap)"),
        ),
        # Dynamic version information section
        ui.h5("ℹ️ Version Information:"),
        ui.div(
            ui.p("**Dashboard Version:** 0.5.1"),
            ui.p(f"**Python Version:** {platform.python_version()}"),
            ui.p(f"**Platform:** {platform.system()} {platform.release()}"),
            ui.p(f"**Environment:** {ENVIRONMENT.title()}"),
            class_="bg-light p-3 rounded",  # Bootstrap styling for highlighted section
        ),
        # Footer with generation timestamp
        ui.hr(),
        ui.p(ui.tags.small(f"Generated on: {datetime.now(UTC).strftime('%Y-%m-%d %H:%M:%S')}")),
        # Modal configuration
        title="About QWIM Dashboard",
        size="l",  # Large modal size for comprehensive content
        easy_close=True,  # Allow closing by clicking outside modal
        footer=ui.modal_button("Close", class_="btn-primary"),
    )


#: Pre-created About modal instance for consistent presentation
#: Avoids recreating the modal on each display request
about_modal = create_about_modal()


def create_app_ui() -> Any:
    """
    Create the main application user interface.

    This function defines the primary UI layout using a navbar-based design
    that provides intuitive navigation and professional appearance.

    ## UI Architecture

    The interface follows a hierarchical structure:

    ```
    page_navbar (main container)
    ├── nav_panel (QWIM Portfolios tab)
    │   └── tab_portfolios_ui (portfolio analysis interface)
    ├── nav_spacer (flexible spacing)
    └── nav_control (About button)
    ```

    ## Layout Components

    ### Main Navigation
    - **Portfolio Tab**: Primary functionality for portfolio analysis
    - **About Button**: Access to application information and help

    ### Theme and Styling
    - **Flatly Theme**: Modern, professional Bootstrap theme via shinyswatch
    - **Responsive Design**: Adapts to different screen sizes and devices
    - **Fillable Layout**: Optimizes space usage for data visualization

    ## Returns

    Returns
    -------
        shiny.ui.page_navbar: Main application UI layout configured with
        portfolio analysis functionality and navigation components.

    ## Configuration Details

    ### Navbar Settings
    - `title`: Application branding displayed in navigation bar
    - `theme`: Bootstrap theme for consistent visual appearance
    - `fillable`: Enables responsive layout optimization
    - `id`: Unique identifier for reactive navigation tracking

    ### Component IDs
    Following project naming conventions:
    - `ID_tab_portfolios`: Main portfolio analysis tab
    - `input_ID_show_about`: About button for modal display
    - `input_ID_navbar_main`: Navigation bar for tab tracking

    ## Usage Example

    ```python
    # Create the main UI
    app_ui = create_app_ui()

    # Use in Shiny app
    app = App(app_ui, app_server)
    ```

    ## Extensibility

    To add new tabs to the dashboard:

    ```python
    # Add additional nav_panel entries before nav_spacer
    (
        ui.nav_panel(
            "New Tab Name", new_tab_ui("ID_new_tab", data_utils=data_utils, data_inputs=data_inputs)
        ),
    )
    ```

    ## Notes

    !!! info "Theme Selection"
        The Flatly theme provides a modern, professional appearance suitable
        for financial and analytical applications.

    !!! tip "Navigation Pattern"
        The nav_spacer() pushes the About button to the right side of the
        navigation bar, following standard web design patterns.

    !!! warning "ID Conventions"
        All component IDs follow the project naming standards with descriptive
        prefixes (input_, output_, ID_) for consistency and maintainability.
    """
    return ui.page_navbar(
        ui.nav_panel(
            "Overview",
            tab_overview_ui(
                "ID_tab_overview",  # Unique module identifier
                data_utils=data_utils,  # Utility functions and configuration
                data_inputs=data_inputs,
            ),
        ),
        ui.nav_panel(
            "Clients",
            tab_clients_ui(
                "ID_tab_clients",  # Unique module identifier
                data_utils=data_utils,  # Utility functions and configuration
                data_inputs=data_inputs,
            ),  # Input datasets for clients
        ),
        # portfolio analysis tab
        ui.nav_panel(
            "Portfolios",
            tab_portfolios_ui(
                "ID_tab_portfolios",  # Unique module identifier
                data_utils=data_utils,  # Utility functions and configuration
                data_inputs=data_inputs,
            ),  # Input datasets for analysis
        ),
        # Products tab (annuities, insurance, etc.)
        ui.nav_panel(
            "Products",
            tab_products_ui(
                "ID_tab_products",  # Unique module identifier
                data_utils=data_utils,  # Utility functions and configuration
                data_inputs=data_inputs,
            ),  # Input datasets for product analysis
        ),
        # Results tab (reporting, PDF generation, etc.)
        ui.nav_panel(
            "Results",
            tab_results_ui(
                "ID_tab_results",  # Unique module identifier
                data_utils=data_utils,  # Utility functions and configuration
                data_inputs=data_inputs,
            ),  # Input datasets for results
        ),
        # Navigation spacer to push subsequent elements to the right
        # Creates proper visual separation between main content and utility buttons
        ui.nav_spacer(),
        # About button in navigation bar for application information
        # Provides users with help, features overview, and version details
        ui.nav_control(
            ui.input_action_button(
                "input_ID_show_about",  # Button identifier following naming convention
                "About",  # Button label
                class_="btn-outline-light me-2",
            ),  # Bootstrap styling classes
        ),
        # Application-level configuration settings
        title="QWIM Dashboard",  # Application title displayed in browser and navbar
        theme=shinyswatch.theme.flatly(),  # Modern Bootstrap theme for professional appearance
        id="input_ID_navbar_main",  # Unique identifier for reactive navigation tracking
        fillable=True,  # Enable responsive fillable layout for optimal space usage
    )


#: Main application UI instance
#: Created once for use by the Shiny App constructor
app_ui = create_app_ui()


# =============================================================================
# Server Logic Functions
# =============================================================================


def app_server(input: Any, output: Any, session: Any) -> None:  # noqa: A002, ARG001
    """Coordinate server-side functionality for the QWIM Dashboard application.

    This function serves as the central coordinator for all server-side functionality,
    managing reactive state, navigation tracking, modal dialogs, and module initialization.

    ## Server Architecture

    The server follows a coordinated modular pattern:

    ```mermaid
    graph TD
        A[app_server] --> B[Reactive Values]
        A --> C[Navigation Tracking]
        A --> D[Modal Handling]
        A --> E[Module Servers]
        B --> F[Portfolio Module]
        E --> F
    ```

    ## Core Responsibilities

    1. **Reactive State Management**: Initialize and coordinate reactive values
    2. **Navigation Tracking**: Monitor and respond to tab changes
    3. **Modal Dialog Handling**: Manage About dialog display
    4. **Module Coordination**: Initialize and coordinate specialized modules

    ## Parameters

    Args:
        input: Shiny input object containing user interface interactions.
            Provides access to reactive values from UI components.
            Access pattern: `input.component_id()` for reactive values,
            `input.component_id.get()` for current values.

        output: Shiny output object for rendering UI components.
            Used by modules to render plots, tables, and other outputs.
            Define outputs using `@output.component_id` decorator pattern.

        session: Shiny session object for session-specific functionality.
            Handles user session state, authentication, and communication.
            Provides session-level utilities and information.

    ## Reactive System Components

    ### State Management
    The server maintains centralized reactive state through:
    - `reactives_shiny`: Dictionary of application-wide reactive values
    - `current_tab`: Reactive value tracking active navigation tab

    ### Reactive Effects
    - `_update_reactives_shiny()`: Updates state based on input events
    - `_track_selected_tab()`: Monitors navigation tab changes
    - `_show_about_modal()`: Displays About dialog on button click

    ## Error Handling

    Implements defensive programming principles:
    - Early validation of reactive value initialization
    - Comprehensive error reporting with descriptive messages
    - Graceful degradation for non-critical functionality

    ## Raises

    DashboardInitializationError: If reactive values cannot be initialized.
        This is a critical error that prevents dashboard functionality.

    ## Usage Example

    ```python
    # This function is automatically called by Shiny framework
    app = App(app_ui, app_server)
    app.run()
    ```

    ## Module Integration

    The server initializes the portfolio analysis module:

    ```python
    portfolios_module_instance = tab_portfolios_server(
        id="ID_tab_portfolios",
        data_utils=data_utils,
        data_inputs=data_inputs,
        reactives_shiny=reactives_shiny,
    )
    ```

    ## Performance Considerations

    - Reactive effects only trigger when relevant inputs change
    - Module initialization is performed once during startup
    - Shared data objects are passed by reference to minimize memory usage

    ## Notes

    !!! info "Reactive Programming"
        The server leverages Shiny's reactive programming model for efficient
        updates and real-time responsiveness to user interactions.

    !!! warning "Error Handling"
        Critical initialization errors will prevent the dashboard from starting.
        Non-critical errors are logged but don't stop application execution.

    !!! tip "Module Extension"
        To add new modules, initialize them after the existing portfolio module
        and pass the same data_utils, data_inputs, and reactives_shiny objects.
    """
    # Initialize reactive values for application state management
    # This creates the centralized state that coordinates all dashboard modules
    try:
        reactives_shiny = initialize_reactives_shiny(data_utils=data_utils)
    except Exception as exc:
        # Critical error - dashboard cannot function without reactive state
        custom_exc = Exception_Configuration(
            "Failed to initialize reactive values",
            severity=Exception_Severity.CRITICAL,
            cause=exc,
            context={"component": "reactives_shiny"},
        )
        custom_exc.log(
            logger=_logger,
            level="error",
        )
        raise custom_exc from exc

    # Initialize navigation tracking reactive value
    # Defaults to the first available tab for consistent user experience
    current_tab = reactive.value("QWIM Portfolios")

    @reactive.effect
    def observer_track_selected_tab() -> None:
        """
        Track and update the currently selected navigation tab.

        This reactive effect monitors changes to the main navigation bar
        and updates the `current_tab` reactive value accordingly.

        ## Functionality

        - Monitors navigation bar state changes
        - Updates current_tab reactive value
        - Enables tab-specific functionality and state management
        - Provides tab context for conditional logic in modules

        ## Implementation Details

        Uses `input.input_ID_navbar_main()` to get the currently selected tab
        and validates the value before updating the reactive state.

        ## Usage Example

        ```python
        # In other reactive contexts, access current tab
        active_tab = current_tab()
        if active_tab == "QWIM Portfolios":
            # Tab-specific logic
        ```

        ## Notes

        !!! info "Validation"
            Includes input validation to ensure only valid tab names
            are set in the reactive value.

        !!! tip "Extension"
            When adding new tabs, this function automatically tracks
            them without modification.
        """
        selected_tab = input.input_ID_navbar_main()
        # Validate that a tab was actually selected before updating
        if selected_tab:
            current_tab.set(selected_tab)

    @reactive.effect
    @reactive.event(input.input_ID_show_about)
    def observer_show_about_modal() -> None:
        """
        Display the About modal dialog when the About button is clicked.

        This reactive effect responds specifically to clicks on the About button
        and displays the modal containing application information.

        ## Event-Driven Pattern

        Uses `@reactive.event` decorator to respond only to specific button clicks
        rather than any input change, ensuring the modal only appears when requested.

        ## Modal Content

        Displays the pre-created `about_modal` which includes:
        - Feature overview and capabilities
        - Technology stack information
        - Version and platform details
        - Usage guidance and support information

        ## User Experience

        - Modal appears immediately upon button click
        - Can be dismissed by clicking outside or using the close button
        - Provides comprehensive application information in accessible format

        ## Implementation

        ```python
        # Button click → Event trigger → Modal display
        ui.modal_show(about_modal)
        ```

        ## Notes

        !!! info "Event Specificity"
            The @reactive.event decorator ensures this function only runs
            when the specific About button is clicked, not on other input changes.

        !!! tip "Modal Management"
            The modal is pre-created for performance and consistency,
            avoiding recreation on each display request.
        """
        ui.modal_show(about_modal)

    # Initialize clients module server
    tab_overview_server(  # pyright: ignore[reportCallIssue]
        id="ID_tab_overview",  # pyrefly: ignore[unexpected-keyword,bad-argument-count]  # Unique module identifier following naming convention
        data_utils=data_utils,  # Utility functions and configuration settings
        data_inputs=data_inputs,  # Input datasets for overview
        reactives_shiny=reactives_shiny,
    )

    # Initialize clients module server
    tab_clients_server(  # pyright: ignore[reportCallIssue]
        id="ID_tab_clients",  # pyrefly: ignore[unexpected-keyword,bad-argument-count]  # Unique module identifier following naming convention
        data_utils=data_utils,  # Utility functions and configuration settings
        data_inputs=data_inputs,  # Input datasets for portfolio analysis
        reactives_shiny=reactives_shiny,
    )  # Centralized reactive state for coordination

    # Initialize portfolio analysis module server
    tab_portfolios_server(  # pyright: ignore[reportCallIssue]
        id="ID_tab_portfolios",  # pyrefly: ignore[unexpected-keyword,bad-argument-count]  # Unique module identifier following naming convention
        data_utils=data_utils,  # Utility functions and configuration settings
        data_inputs=data_inputs,  # Input datasets for portfolio analysis
        reactives_shiny=reactives_shiny,
    )  # Centralized reactive state for coordination

    # Initialize products module server
    tab_products_server(  # pyright: ignore[reportCallIssue]
        id="ID_tab_products",  # pyrefly: ignore[unexpected-keyword,bad-argument-count]  # Unique module identifier following naming convention
        data_utils=data_utils,  # Utility functions and configuration settings
        data_inputs=data_inputs,  # Input datasets for product analysis
        reactives_shiny=reactives_shiny,
    )  # Centralized reactive state for coordination

    # Initialize results module server
    tab_results_server(  # pyright: ignore[reportCallIssue]
        id="ID_tab_results",  # pyrefly: ignore[unexpected-keyword,bad-argument-count]  # Unique module identifier following naming convention
        data_utils=data_utils,  # Utility functions and configuration settings
        data_inputs=data_inputs,  # Input datasets for results
        reactives_shiny=reactives_shiny,
    )  # Centralized reactive state for coordination


#: Main Shiny application instance combining UI and server functionality
#: This creates the complete application ready for deployment
app = App(app_ui, app_server)


# =============================================================================
# Application Entry Point
# =============================================================================


def main() -> None:
    """Handle application startup with environment-specific configuration.

    This function handles application startup with environment-specific configuration,
    providing different settings for development and production deployments.

    ## Environment Configuration

    The application supports two deployment modes:

    === "Development Mode (Default)"
        - **Host**: `127.0.0.1` (localhost access only)
        - **Port**: `8080`
        - **Reload**: `False` (stable operation for testing)
        - **Use Case**: Local development and testing

    === "Production Mode"
        - **Host**: `0.0.0.0` (external network access)
        - **Port**: `8000`
        - **Reload**: `False` (stable operation without automatic restarts)
        - **Use Case**: Production deployment and external access

    ## Environment Control

    Set the deployment mode using the `QWIM_ENVIRONMENT` environment variable:

    ```bash
    # Development mode (default)
    python main_App.py

    # Production mode
    QWIM_ENVIRONMENT=production python main_App.py
    ```

    ## Startup Information

    The function provides comprehensive startup information including:
    - Application mode and configuration
    - Python version and platform details
    - Project directory location
    - Access URLs for the dashboard

    ## Error Handling

    Implements robust error handling with:
    - Detailed error messages for debugging
    - Complete stack trace information
    - Graceful application termination on critical errors
    - System exit codes for process monitoring

    ## Raises

    SystemExit: If the application fails to start due to configuration issues,
    missing dependencies, or other critical errors.

    ## Startup Sequence

    1. **Environment Detection**: Determine deployment mode from environment variable
    2. **Information Display**: Print startup information and configuration details
    3. **Platform Validation**: Display Python and platform version information
    4. **Server Start**: Launch Shiny server with appropriate configuration
    5. **Error Handling**: Catch and report any startup failures

    ## Network Configuration

    ### Development Access
    - URL: http://127.0.0.1:8080
    - Access: Localhost only (secure for development)
    - Features: Stable operation, detailed error messages

    ### Production Access
    - URL: http://0.0.0.0:8000
    - Access: External network access (configure firewall as needed)
    - Features: Optimized performance, stable operation

    ## Performance Considerations

    ### Development Mode
    - Stable operation without file monitoring
    - Detailed logging and error reporting

    ### Production Mode
    - No file system monitoring
    - Optimized for stability

    ## Example Usage

    ```python
    # Direct execution
    if __name__ == "__main__":
        main()

    # Programmatic usage
    import main_App
    main_App.main()
    ```

    ## Security Considerations

    !!! warning "Production Security"
        Production mode provides stable operation for external access.
        Ensure proper firewall configuration when using 0.0.0.0.

    !!! info "Development Safety"
        Development mode restricts access to localhost (127.0.0.1) for security
        during development and testing phases.

    ## Monitoring and Logging

    - Startup information is printed to console for monitoring and debugging
    - Error details are captured with full stack traces
    - Log files are created in the `logs/` directory
    - System exit codes indicate success (0) or failure (1)

    ## Troubleshooting

    Common startup issues and solutions:

    ??? question "Port already in use"
        Another application is using the port. Either stop the other application
        or modify the port number in the function.

    ??? question "Permission denied"
        Insufficient permissions to bind to the port (especially for ports < 1024).
        Use a higher port number or run with appropriate permissions.

    ??? question "Module import errors"
        Missing dependencies or incorrect Python path. Verify all requirements
        are installed and the project structure is correct.
    """
    # Display comprehensive startup information for monitoring and debugging
    _logger.info(
        "Starting QWIM Dashboard",
        extra={
            "environment": ENVIRONMENT,
            "python_version": platform.python_version(),
            "platform": f"{platform.system()} {platform.release()}",
            "project_directory": str(project_dir),
        },
    )

    with Performance_Timer("QWIM Dashboard startup"):
        host: str = "127.0.0.1"  # default; overridden below
        port: int = 8080  # default; overridden below
        try:
            # Launch application with environment-appropriate configuration
            if ENVIRONMENT == "production":
                # Production configuration optimized for stability and security
                host = "0.0.0.0"
                port = 8000
                _logger.info(
                    "Production mode",
                    extra={
                        "url": f"http://{host}:{port}",
                        "features": "External access, optimized performance",
                    },
                )
            else:
                # Development configuration optimized for development workflow
                host = "127.0.0.1"
                port = 8080
                _logger.info(
                    "Development mode",
                    extra={
                        "url": f"http://{host}:{port}",
                        "features": "Stable operation, detailed error messages",
                    },
                )

            app.run(host=host, port=port, reload=False)

        except KeyboardInterrupt:
            _logger.info("Dashboard shutdown requested by user (Ctrl+C)")
            sys.exit(0)

        except Exception as exc:  # noqa: BLE001 - Comprehensive startup error handling
            # Comprehensive error handling with detailed information
            with capture_exception(context={"startup": True, "environment": ENVIRONMENT}):
                custom_exc = Exception_Configuration(
                    "Failed to start QWIM Dashboard",
                    severity=Exception_Severity.CRITICAL,
                    cause=exc,
                    context={"environment": ENVIRONMENT, "host": host, "port": port},
                )
                custom_exc.log(
                    logger=_logger,
                    level="error",
                )
                _logger.error(
                    "Dashboard startup failed",
                    extra={
                        "error": str(exc),
                        "logs_location": str(project_dir / "logs"),
                        "python_version_required": "3.11+",
                    },
                )
            sys.exit(1)  # Exit with error code for process monitoring


# =============================================================================
# Application Execution
# =============================================================================

# Main entry point when script is executed directly
# This enables both direct execution and programmatic usage
if __name__ == "__main__":
    main()
