"""
Backward compatibility module for utils_errors.

This module provides backward compatibility for code that imports from
utils_errors instead of utils_errors_dashboard. All functionality has been
moved to utils_errors_dashboard.py.

This file simply re-exports everything from utils_errors_dashboard.py to
maintain backward compatibility with existing code and tests.

Usage:
    # Old style (still works)
    from src.dashboard.shiny_utils.utils_errors import Error_Dashboard_Initialization

    # New style (preferred)
    from src.dashboard.shiny_utils.utils_errors_dashboard import Error_Dashboard_Initialization

    # Best practice (use package-level import)
    from src.dashboard.shiny_utils import Error_Dashboard_Initialization
"""

# Re-export everything from utils_errors_dashboard
from __future__ import annotations

from .utils_errors_dashboard import *  # noqa: F403


# Explicitly list what we're re-exporting for better IDE support
__all__ = [
    "DashboardInitializationError",
    "DataLoadingError",
    "Error_Configuration",
    "Error_Dashboard_Initialization",
    "Error_Data_Loading",
    "Error_Module_Initialization",
    "Error_Silent_Initialization",
    "Error_Validation",
    "ModuleInitializationError",
    # Backward compatibility aliases
    "SilentInitializationException",
    "is_silent_exception",
]
