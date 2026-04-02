"""Custom Exception Classes for QWIM Dashboard.

This module provides backward-compatible exception aliases for the QWIM Dashboard,
mapping local exception names to the centralized exception classes from
``src.utils.custom_exceptions_errors_loggers.exception_custom``.

This module is DEPRECATED. Use the exception classes directly from
``src.utils.custom_exceptions_errors_loggers.exception_custom`` instead.

Deprecated Aliases
------------------
- ``Error_Silent_Initialization``: Use ``Exception_Custom`` with silent flag
- ``Error_Dashboard_Initialization``: Use ``Exception_Configuration``
- ``Error_Data_Loading``: Use ``Exception_Data_Not_Found``
- ``Error_Module_Initialization``: Use ``Exception_Configuration``
- ``Error_Validation``: Use ``Exception_Validation_Input``
- ``Error_Configuration``: Use ``Exception_Configuration``
"""

from __future__ import annotations

import warnings

from typing import Any

# Import centralized exception classes
from src.utils.custom_exceptions_errors_loggers.exception_custom import (
    Exception_Configuration,
    Exception_Custom,
    Exception_Data_Not_Found,
    Exception_Severity,
    Exception_Validation_Input,
)
from src.utils.custom_exceptions_errors_loggers.logger_custom import get_logger


#: Module-level logger instance
_logger = get_logger(__name__)

# =============================================================================
# Deprecated Exception Aliases (for backward compatibility)
# =============================================================================


class Error_Silent_Initialization(Exception_Custom):
    """DEPRECATED: Use Exception_Custom with silent handling instead.

    Custom exception class for silently handling expected initialization errors.
    This exception is used to signal expected errors during initialization of
    reactive components, where displaying an error message would be inappropriate.

    .. deprecated::
        Use ``Exception_Custom`` with appropriate severity level instead.
    """

    def __init__(self, message: str = "", *args: Any, **kwargs: Any) -> None:
        """Initialize with deprecation warning."""
        warnings.warn(
            "Error_Silent_Initialization is deprecated. "
            "Use Exception_Custom from exception_custom.py instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        super().__init__(message, *args, severity=Exception_Severity.DEBUG, **kwargs)  # pyright: ignore[reportArgumentType]


# Keep old name as alias for backward compatibility
SilentInitializationException = Error_Silent_Initialization


class Error_Dashboard_Initialization(Exception_Configuration):
    """DEPRECATED: Use Exception_Configuration instead.

    Exception raised when dashboard initialization fails.
    This is a critical error that prevents the dashboard from starting properly.

    .. deprecated::
        Use ``Exception_Configuration`` from exception_custom.py instead.
    """

    def __init__(self, message: str = "", *args: Any, **kwargs: Any) -> None:
        """Initialize with deprecation warning."""
        warnings.warn(
            "Error_Dashboard_Initialization is deprecated. "
            "Use Exception_Configuration from exception_custom.py instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        super().__init__(message, *args, severity=Exception_Severity.CRITICAL, **kwargs)


# Keep old name as alias for backward compatibility
DashboardInitializationError = Error_Dashboard_Initialization


class Error_Data_Loading(Exception_Data_Not_Found):
    """DEPRECATED: Use Exception_Data_Not_Found instead.

    Exception raised when data loading fails.
    Used when data files cannot be read, are missing, or have invalid format.

    .. deprecated::
        Use ``Exception_Data_Not_Found`` from exception_custom.py instead.
    """

    def __init__(self, message: str = "", *args: Any, **kwargs: Any) -> None:
        """Initialize with deprecation warning."""
        warnings.warn(
            "Error_Data_Loading is deprecated. "
            "Use Exception_Data_Not_Found from exception_custom.py instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        super().__init__(message, *args, severity=Exception_Severity.ERROR, **kwargs)


# Keep old name as alias for backward compatibility
DataLoadingError = Error_Data_Loading


class Error_Module_Initialization(Exception_Configuration):
    """DEPRECATED: Use Exception_Configuration instead.

    Exception raised when module initialization fails.
    Used when a specific dashboard module (tab, subtab, utility) fails to initialize.

    .. deprecated::
        Use ``Exception_Configuration`` from exception_custom.py instead.
    """

    def __init__(self, message: str = "", *args: Any, **kwargs: Any) -> None:
        """Initialize with deprecation warning."""
        warnings.warn(
            "Error_Module_Initialization is deprecated. "
            "Use Exception_Configuration from exception_custom.py instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        super().__init__(message, *args, severity=Exception_Severity.ERROR, **kwargs)


# Keep old name as alias for backward compatibility
ModuleInitializationError = Error_Module_Initialization


class Error_Validation(Exception_Validation_Input):
    """DEPRECATED: Use Exception_Validation_Input instead.

    Exception raised when input validation fails.
    Used when user input or data validation fails business logic checks.

    .. deprecated::
        Use ``Exception_Validation_Input`` from exception_custom.py instead.
    """

    def __init__(self, message: str = "", *args: Any, **kwargs: Any) -> None:
        """Initialize with deprecation warning."""
        warnings.warn(
            "Error_Validation is deprecated. "
            "Use Exception_Validation_Input from exception_custom.py instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        super().__init__(message, *args, severity=Exception_Severity.WARNING, **kwargs)


class Error_Configuration(Exception_Configuration):
    """DEPRECATED: Use Exception_Configuration instead.

    Exception raised when configuration is invalid.
    Used when application configuration is missing or invalid.

    .. deprecated::
        Use ``Exception_Configuration`` from exception_custom.py instead.
    """

    def __init__(self, message: str = "", *args: Any, **kwargs: Any) -> None:
        """Initialize with deprecation warning."""
        warnings.warn(
            "Error_Configuration is deprecated. "
            "Use Exception_Configuration from exception_custom.py instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        super().__init__(message, *args, severity=Exception_Severity.ERROR, **kwargs)


def is_silent_exception(exception: BaseException) -> bool:
    """Check if an exception should be handled silently during initialization.

    Used to determine if an exception is expected during reactive component
    initialization and should not display error messages to users.

    Parameters
    ----------
    exception
        The exception to check.

    Returns
    -------
    bool
        True if the exception should be handled silently, False otherwise.

    Examples
    --------
    >>> try:
    ...     value = reactive_value.get()
    ... except Exception as exc:
    ...     if is_silent_exception(exc):
    ...         return None  # Silently return None
    ...     raise  # Re-raise unexpected exceptions
    """
    # Check for silent exceptions
    if isinstance(exception, Error_Silent_Initialization):
        return True

    # Check for AttributeError (common during reactive initialization)
    if isinstance(exception, AttributeError):
        return True

    # Check for Exception_Custom with DEBUG severity
    if isinstance(exception, Exception_Custom):
        return exception.severity == Exception_Severity.DEBUG

    return False
