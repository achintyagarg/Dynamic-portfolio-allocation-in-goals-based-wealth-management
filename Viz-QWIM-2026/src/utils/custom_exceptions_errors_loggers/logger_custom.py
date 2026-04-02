"""Custom Logger Module.

This module configures logging to meet regulatory requirements for financial applications.

Features
--------
- Structured logging with timestamps using loguru
- Multiple log files by severity and purpose
- Automatic rotation with retention policies
- Audit trail capabilities for compliance

The module uses loguru and loggerplusplus instead of Python's built-in logging
to provide more powerful and flexible logging capabilities.

Classes
-------
Formatter_Structured_JSON
    JSON formatter for structured logging with audit trail support.
Audit_Logger
    Specialized logger for audit trail events with compliance support.

Functions
---------
setup_logging
    Configure application-wide logging with multiple handlers.
get_logger
    Get a configured logger instance.
log_calculation
    Log financial calculations for audit trail.
log_data_access
    Log data access operations for compliance.

Example
-------
>>> from src.utils.custom_exceptions_errors_loggers.logger_custom import setup_logging, get_logger
>>> setup_logging(log_level="INFO", environment="production")
>>> logger = get_logger(__name__)
>>> logger.info("Application started")

Notes
-----
This module must be initialized once at application startup before any logging occurs.
All timestamps are in UTC for consistency across deployments.

Author: QWIM Dashboard Team
Version: 1.1.0
Last Updated: 2026-02-09
"""

from __future__ import annotations

import contextlib
import json
import sys
import time
import types

from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from functools import wraps
from pathlib import Path
from typing import TYPE_CHECKING, Any, ParamSpec, Self, TypeVar

from loguru import logger
from pydantic import BaseModel, ConfigDict, Field


if TYPE_CHECKING:
    from loguru import Logger


# ==============================================================================
# Type Aliases (Python 3.12+)
# ==============================================================================

type Log_Record_Dict = dict[str, Any]
type Log_Filter_Func = Callable[[Log_Record_Dict], bool]
type Log_Format_Func = Callable[[Log_Record_Dict], str]

P = ParamSpec("P")
R = TypeVar("R")

# ==============================================================================
# Constants
# ==============================================================================

DEFAULT_LOG_LEVEL = "INFO"
DEFAULT_LOG_DIR = Path("logs")
DEFAULT_ROTATION_SIZE = "100 MB"
DEFAULT_RETENTION_DAYS_APPLICATION = "90 days"
DEFAULT_RETENTION_DAYS_AUDIT = "2555 days"  # ~7 years for regulatory compliance
DEFAULT_RETENTION_DAYS_ERROR = "365 days"  # 1 year
DEFAULT_RETENTION_DAYS_PERFORMANCE = "30 days"

# Log file names
LOG_FILE_APPLICATION = "application.log"
LOG_FILE_AUDIT = "audit.log"
LOG_FILE_ERROR = "error.log"
LOG_FILE_PERFORMANCE = "performance.log"
LOG_FILE_DEBUG = "debug.log"

# Log level thresholds (loguru uses numeric values)
LOG_LEVEL_DEBUG = 10
LOG_LEVEL_ERROR = 40


# ==============================================================================
# Pydantic Models for Configuration Validation
# ==============================================================================


class Config_Logging(BaseModel):
    """Configuration model for logging setup.

    Parameters
    ----------
    log_level : str
        Minimum log level (TRACE, DEBUG, INFO, SUCCESS, WARNING, ERROR, CRITICAL).
    environment : str
        Deployment environment (development, staging, production).
    log_dir : Path
        Directory for log files.
    enable_console : bool
        Whether to log to console.
    enable_JSON : bool
        Whether to use JSON format for file logs.
    rotation_size : str
        Size threshold for log rotation.
    """

    log_level: str = Field(
        default=DEFAULT_LOG_LEVEL,
        pattern="^(TRACE|DEBUG|INFO|SUCCESS|WARNING|ERROR|CRITICAL)$",
    )
    environment: str = Field(default="development", pattern="^(development|staging|production)$")
    log_dir: Path = Field(default=DEFAULT_LOG_DIR)
    enable_console: bool = Field(default=True)
    enable_JSON: bool = Field(default=True)
    rotation_size: str = Field(default=DEFAULT_ROTATION_SIZE)

    model_config = ConfigDict(arbitrary_types_allowed=True)


# ==============================================================================
# JSON Serialization Utilities
# ==============================================================================


def serialize_for_JSON(obj: Any) -> Any:
    """Serialize objects for JSON output.

    Parameters
    ----------
    obj : Any
        Object to serialize.

    Returns
    -------
    Any
        JSON-serializable representation of the object.
    """
    if isinstance(obj, datetime):
        return obj.isoformat()
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, bytes):
        return obj.decode("utf-8", errors="replace")
    if hasattr(obj, "__dict__"):
        return obj.__dict__
    return str(obj)


def format_record_as_JSON(record: dict) -> str:
    """Format a loguru record as JSON string.

    Parameters
    ----------
    record : dict
        Loguru record dictionary.

    Returns
    -------
    str
        JSON-formatted log string.
    """
    # Extract timestamp with UTC
    timestamp = record["time"].astimezone(UTC).isoformat()

    # Build base log entry
    log_entry = {
        "timestamp": timestamp,
        "level": record["level"].name,
        "logger": record["name"] or "root",
        "module": record["module"],
        "function": record["function"],
        "line": record["line"],
        "message": record["message"],
    }

    # Add thread/process info for debugging
    if record["level"].no >= LOG_LEVEL_DEBUG:  # DEBUG level or higher
        log_entry["thread_id"] = record["thread"].id
        log_entry["thread_name"] = record["thread"].name
        log_entry["process_id"] = record["process"].id
        log_entry["process_name"] = record["process"].name

    # Add exception info if present
    if record["exception"] is not None:
        log_entry["exception"] = {
            "type": record["exception"].type.__name__ if record["exception"].type else None,
            "value": str(record["exception"].value) if record["exception"].value else None,
            "traceback": record["exception"].traceback or None,
        }

    # Add extra fields from logger.bind() or extra dict
    extra = record.get("extra", {})
    if extra:
        for key, value in extra.items():
            if key not in log_entry:
                log_entry[key] = value

    return json.dumps(log_entry, default=serialize_for_JSON)


def format_sink_JSON(record: dict) -> str:
    """Sink format function for JSON output.

    Parameters
    ----------
    record : dict
        Loguru record dictionary.

    Returns
    -------
    str
        Formatted JSON string with newline.
    """
    return format_record_as_JSON(record) + "\n"


def format_sink_human_readable(record: dict) -> str:
    """Sink format function for human-readable console output.

    Parameters
    ----------
    record : dict
        Loguru record dictionary.

    Returns
    -------
    str
        Human-readable formatted string.
    """
    timestamp = record["time"].strftime("%Y-%m-%d %H:%M:%S")
    level = record["level"].name
    module = record["module"]
    function = record["function"]
    line = record["line"]
    message = record["message"]

    # Escape angle brackets to prevent loguru colorizer from parsing them as color tags
    # This is defensive in case this function is used with colorize=True somewhere
    module = module.replace("<", "\\<").replace(">", "\\>")
    function = function.replace("<", "\\<").replace(">", "\\>")
    message = message.replace("<", "\\<").replace(">", "\\>")

    return f"{timestamp} | {level:<8} | {module}:{function}:{line} | {message}\n"


# ==============================================================================
# Filter Functions for Log Routing
# ==============================================================================


def filter_audit_events(record: dict) -> bool:
    """Filter for audit trail events only.

    Parameters
    ----------
    record : dict
        Loguru record dictionary.

    Returns
    -------
    bool
        True if record is an audit event.
    """
    extra = record.get("extra", {})
    return "event_type" in extra


def filter_performance_events(record: dict) -> bool:
    """Filter for performance metrics events only.

    Parameters
    ----------
    record : dict
        Loguru record dictionary.

    Returns
    -------
    bool
        True if record contains performance metrics.
    """
    extra = record.get("extra", {})
    return "execution_time_ms" in extra


def filter_error_level(record: dict) -> bool:
    """Filter for ERROR level and above.

    Parameters
    ----------
    record : dict
        Loguru record dictionary.

    Returns
    -------
    bool
        True if record is ERROR or CRITICAL level.
    """
    return record["level"].no >= LOG_LEVEL_ERROR  # ERROR = 40, CRITICAL = 50


# ==============================================================================
# Logger Configuration
# ==============================================================================


# Global flag to track if logging has been configured
_logging_configured = False


def setup_logging(
    log_level: str = DEFAULT_LOG_LEVEL,
    environment: str = "development",
    log_dir: Path | None = None,
    enable_console: bool = True,
    enable_JSON: bool = True,
) -> None:
    """Configure application-wide logging using loguru.

    This function must be called once at application startup before any
    logging occurs. It configures multiple log handlers with appropriate
    rotation and retention policies.

    Parameters
    ----------
    log_level : str
        Minimum log level (DEBUG, INFO, WARNING, ERROR, CRITICAL).
        Default is "INFO".
    environment : str
        Deployment environment (development, staging, production).
        Default is "development".
    log_dir : Path, optional
        Directory for log files. Default is ./logs/.
    enable_console : bool
        Whether to log to console. Default is True.
        Set to False for production deployments.
    enable_JSON : bool
        Whether to use JSON format for file logs. Default is True.

    Returns
    -------
    None

    Raises
    ------
    ValueError
        If log_level or environment values are invalid.

    Example
    -------
    >>> setup_logging(log_level="INFO", environment="production")
    >>> logger = get_logger(__name__)
    >>> logger.info("Application started")

    Notes
    -----
    Log files created:

    - application.log: All logs (INFO and above), 90 days retention
    - audit.log: Audit events only, 7 years retention (regulatory compliance)
    - error.log: ERROR and CRITICAL only, 1 year retention
    - performance.log: Performance metrics, 30 days retention
    - debug.log: DEBUG level (development only), 7 days retention
    """
    global _logging_configured

    # Validate configuration using Pydantic
    if log_dir is None:
        log_dir = DEFAULT_LOG_DIR

    config = Config_Logging(
        log_level=log_level.upper(),
        environment=environment.lower(),
        log_dir=log_dir,
        enable_console=enable_console,
        enable_JSON=enable_JSON,
    )

    # Create logs directory
    config.log_dir.mkdir(parents=True, exist_ok=True)

    # Create archive subdirectory
    archive_dir = config.log_dir / "archive"
    archive_dir.mkdir(exist_ok=True)

    # Remove default logger handler
    logger.remove()

    # 1. Application log - All logs (INFO and above)
    logger.add(
        config.log_dir / LOG_FILE_APPLICATION,
        level="INFO",
        format=format_sink_human_readable,  # type: ignore[arg-type]
        serialize=config.enable_JSON,  # Use native JSON serialization
        rotation=config.rotation_size,
        retention=DEFAULT_RETENTION_DAYS_APPLICATION,
        compression="gz",
        encoding="utf-8",
        enqueue=True,  # Thread-safe logging
        colorize=False,  # Disable colorization to prevent <module> parse errors
        backtrace=True,
        diagnose=True,
    )

    # 2. Audit log - INFO level for compliance tracking (filtered)
    logger.add(
        config.log_dir / LOG_FILE_AUDIT,
        level="INFO",
        format=format_sink_human_readable,  # type: ignore[arg-type]
        serialize=True,  # Always JSON for audit (native serialization)
        rotation="1 day",  # Daily rotation for audit
        retention=DEFAULT_RETENTION_DAYS_AUDIT,
        compression="gz",
        encoding="utf-8",
        enqueue=True,
        colorize=False,  # Disable colorization for file output
        filter=filter_audit_events,  # type: ignore[arg-type]
        backtrace=True,
    )

    # 3. Error log - ERROR and above only
    logger.add(
        config.log_dir / LOG_FILE_ERROR,
        level="ERROR",
        format=format_sink_human_readable,  # type: ignore[arg-type]
        serialize=config.enable_JSON,  # Use native JSON serialization
        rotation=config.rotation_size,
        retention=DEFAULT_RETENTION_DAYS_ERROR,
        compression="gz",
        encoding="utf-8",
        enqueue=True,
        colorize=False,  # Disable colorization for file output
        filter=filter_error_level,  # type: ignore[arg-type]
        backtrace=True,
        diagnose=True,
    )

    # 4. Performance log - Performance metrics only
    logger.add(
        config.log_dir / LOG_FILE_PERFORMANCE,
        level="DEBUG",
        format=format_sink_human_readable,  # type: ignore[arg-type]
        serialize=True,  # Always JSON for performance metrics (native serialization)
        rotation=config.rotation_size,
        retention=DEFAULT_RETENTION_DAYS_PERFORMANCE,
        compression="gz",
        encoding="utf-8",
        enqueue=True,
        colorize=False,  # Disable colorization for file output
        filter=filter_performance_events,  # type: ignore[arg-type]
    )

    # 5. Debug log (development only)
    if config.environment == "development":
        logger.add(
            config.log_dir / LOG_FILE_DEBUG,
            level="DEBUG",
            format=format_sink_human_readable,  # type: ignore[arg-type]
            serialize=config.enable_JSON,  # Use native JSON serialization
            rotation=config.rotation_size,
            retention="7 days",
            compression="gz",
            encoding="utf-8",
            enqueue=True,
            colorize=False,  # Disable colorization for file output
            backtrace=True,
            diagnose=True,
        )

    # 6. Console handler (for development or when enabled)
    if config.enable_console:
        # On Windows, special characters in logs (like icons) can cause UnicodeEncodeError
        # if the console encoding is not set to UTF-8. We force reconfigure it here.
        # Additionally, disable colorization on Windows to prevent loguru from parsing
        # <module> and similar text in exception tracebacks as color directives.
        use_colorize = True
        if sys.platform == "win32":
            use_colorize = False  # Disable colorization on Windows to prevent parse errors
            if hasattr(sys.stderr, "reconfigure"):
                # Suppress exception if reconfiguration fails (e.g. in IDEs or non-standard terminals)
                with contextlib.suppress(Exception):
                    sys.stderr.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]

        console_level = "DEBUG" if config.environment == "development" else "INFO"
        logger.add(
            sys.stderr,
            level=console_level,
            format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
            "<level>{message}</level>",
            colorize=use_colorize,
            enqueue=True,
        )

    _logging_configured = True

    # Log startup
    logger.bind(
        event_type="SYSTEM_STARTUP",
        data={
            "log_level": config.log_level,
            "environment": config.environment,
            "log_dir": str(config.log_dir),
            "console_enabled": config.enable_console,
            "JSON_enabled": config.enable_JSON,
        },
    ).info("Logging configured successfully")


def get_logger(name: str) -> Logger:
    """Get a logger instance bound with the given name.

    Parameters
    ----------
    name : str
        Logger name (typically __name__).

    Returns
    -------
    Logger
        Configured loguru logger instance bound with the name.

    Raises
    ------
    RuntimeError
        If setup_logging() has not been called.

    Example
    -------
    >>> app_logger = get_logger(__name__)
    >>> app_logger.info("Processing started")
    """
    if not _logging_configured:
        # Auto-configure with defaults if not configured
        setup_logging()

    return logger.bind(name=name)


# ==============================================================================
# Audit Logger Class
# ==============================================================================


@dataclass
class Audit_Logger:
    """Specialized logger for audit trail events.

    This class provides convenience methods for logging events that require
    audit trails for compliance purposes in financial applications.

    Parameters
    ----------
    logger_name : str
        Name to identify this logger instance.

    Attributes
    ----------
    logger_name : str
        The name of the logger.
    _logger : Logger
        The bound loguru logger instance.

    Example
    -------
    >>> audit = Audit_Logger("portfolio_module")
    >>> audit.log_calculation(
    ...     operation="Option Pricing",
    ...     inputs={"spot": 100, "strike": 105},
    ...     result=3.45,
    ...     execution_time_ms=12.5,
    ...     user_id="analyst_001",
    ... )
    """

    logger_name: str
    _logger: Logger = field(init=False, repr=False)

    def __post_init__(self) -> None:
        """Initialize the bound logger after dataclass initialization."""
        self._logger = get_logger(self.logger_name)

    def log_calculation(
        self,
        operation: str,
        inputs: dict[str, Any],
        result: Any,
        execution_time_ms: float,
        user_id: str | None = None,
        session_id: str | None = None,
    ) -> None:
        """Log a financial calculation for audit trail.

        Parameters
        ----------
        operation : str
            Name of the calculation (e.g., "Black-Scholes Pricing").
        inputs : dict[str, Any]
            Input parameters used in calculation.
        result : Any
            Calculation result.
        execution_time_ms : float
            Execution time in milliseconds.
        user_id : str, optional
            User identifier performing the calculation.
        session_id : str, optional
            Session identifier for tracking.

        Returns
        -------
        None

        Example
        -------
        >>> audit.log_calculation(
        ...     operation="Portfolio Return",
        ...     inputs={"weights": [0.6, 0.4], "returns": [0.05, 0.03]},
        ...     result=0.042,
        ...     execution_time_ms=5.2,
        ...     user_id="analyst_001",
        ... )
        """
        self._logger.bind(
            event_type="CALCULATION",
            user_id=user_id or "system",
            session_id=session_id,
            execution_time_ms=execution_time_ms,
            data={
                "operation": operation,
                "inputs": inputs,
                "result": result,
            },
        ).info(f"Calculation completed: {operation}")

    def log_data_access(
        self,
        source: str,
        operation: str,
        record_count: int,
        user_id: str | None = None,
        session_id: str | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        """Log data access for audit trail.

        Parameters
        ----------
        source : str
            Data source (file path, database, API endpoint).
        operation : str
            Type of operation (READ, WRITE, DELETE, UPDATE).
        record_count : int
            Number of records accessed.
        user_id : str, optional
            User identifier performing the operation.
        session_id : str, optional
            Session identifier for tracking.
        details : dict[str, Any], optional
            Additional details about the operation.

        Returns
        -------
        None

        Example
        -------
        >>> audit.log_data_access(
        ...     source="inputs/raw/data_ETFs.csv",
        ...     operation="READ",
        ...     record_count=252,
        ...     user_id="analyst_001",
        ... )
        """
        self._logger.bind(
            event_type="DATA_ACCESS",
            user_id=user_id or "system",
            session_id=session_id,
            data={
                "source": source,
                "operation": operation,
                "record_count": record_count,
                "details": details or {},
            },
        ).info(f"Data access: {operation} from {source}")

    def log_parameter_change(
        self,
        parameter_name: str,
        old_value: Any,
        new_value: Any,
        user_id: str | None = None,
        session_id: str | None = None,
        reason: str | None = None,
    ) -> None:
        """Log parameter or configuration change for audit trail.

        Parameters
        ----------
        parameter_name : str
            Name of the parameter changed.
        old_value : Any
            Previous value.
        new_value : Any
            New value.
        user_id : str, optional
            User identifier making the change.
        session_id : str, optional
            Session identifier for tracking.
        reason : str, optional
            Reason for the change.

        Returns
        -------
        None

        Example
        -------
        >>> audit.log_parameter_change(
        ...     parameter_name="risk_free_rate",
        ...     old_value=0.02,
        ...     new_value=0.025,
        ...     user_id="admin_001",
        ...     reason="Updated to reflect market conditions",
        ... )
        """
        self._logger.bind(
            event_type="PARAMETER_CHANGE",
            user_id=user_id or "system",
            session_id=session_id,
            data={
                "parameter": parameter_name,
                "old_value": old_value,
                "new_value": new_value,
                "reason": reason,
            },
        ).warning(f"Parameter changed: {parameter_name}")

    def log_user_action(
        self,
        action: str,
        resource: str,
        user_id: str,
        session_id: str | None = None,
        details: dict[str, Any] | None = None,
        success: bool = True,
    ) -> None:
        """Log user action for audit trail.

        Parameters
        ----------
        action : str
            Action performed (e.g., "LOGIN", "LOGOUT", "VIEW", "EXPORT").
        resource : str
            Resource affected by the action.
        user_id : str
            User identifier performing the action.
        session_id : str, optional
            Session identifier for tracking.
        details : dict[str, Any], optional
            Additional details about the action.
        success : bool
            Whether the action was successful.

        Returns
        -------
        None

        Example
        -------
        >>> audit.log_user_action(
        ...     action="EXPORT",
        ...     resource="portfolio_report.pdf",
        ...     user_id="analyst_001",
        ...     details={"format": "PDF", "pages": 15},
        ... )
        """
        log_data = {
            "action": action,
            "resource": resource,
            "success": success,
            "details": details or {},
        }

        bound_logger = self._logger.bind(
            event_type="USER_ACTION",
            user_id=user_id,
            session_id=session_id,
            data=log_data,
        )

        if success:
            bound_logger.info(f"User action: {action} on {resource}")
        else:
            bound_logger.warning(f"User action failed: {action} on {resource}")

    def log_system_event(
        self,
        event_name: str,
        details: dict[str, Any] | None = None,
        severity: str = "INFO",
    ) -> None:
        """Log system event for monitoring.

        Parameters
        ----------
        event_name : str
            Name of the system event.
        details : dict[str, Any], optional
            Additional event details.
        severity : str
            Log severity level (DEBUG, INFO, WARNING, ERROR, CRITICAL).

        Returns
        -------
        None

        Example
        -------
        >>> audit.log_system_event(
        ...     event_name="CACHE_CLEARED",
        ...     details={"cache_size_mb": 256},
        ...     severity="INFO",
        ... )
        """
        bound_logger = self._logger.bind(
            event_type="SYSTEM_EVENT",
            data={
                "event_name": event_name,
                "details": details or {},
            },
        )

        log_method = getattr(bound_logger, severity.lower(), bound_logger.info)
        log_method(f"System event: {event_name}")


# ==============================================================================
# Performance Logging Utilities
# ==============================================================================


def log_performance(
    operation_name: str,
    execution_time_ms: float,
    details: dict[str, Any] | None = None,
    logger_instance: Logger | None = None,
) -> None:
    """Log performance metrics for an operation.

    Parameters
    ----------
    operation_name : str
        Name of the operation being measured.
    execution_time_ms : float
        Execution time in milliseconds.
    details : dict[str, Any], optional
        Additional performance details.
    logger_instance : Logger, optional
        Logger instance to use. If None, uses default logger.

    Returns
    -------
    None

    Example
    -------
    >>> start = time.perf_counter()
    >>> # ... perform operation ...
    >>> elapsed = (time.perf_counter() - start) * 1000
    >>> log_performance("portfolio_calculation", elapsed)
    """
    log = logger_instance or logger

    log.bind(
        event_type="PERFORMANCE",
        execution_time_ms=execution_time_ms,
        data={
            "operation": operation_name,
            "details": details or {},
        },
    ).debug(f"Performance: {operation_name} completed in {execution_time_ms:.2f}ms")


class Performance_Timer:
    """Context manager for timing operations with automatic logging.

    Parameters
    ----------
    operation_name : str
        Name of the operation being timed.
    logger_instance : Logger, optional
        Logger instance to use.
    log_start : bool
        Whether to log when timing starts.
    details : dict[str, Any], optional
        Additional details to include in logs.

    Example
    -------
    >>> with Performance_Timer("portfolio_optimization") as timer:
    ...     result = optimize_portfolio(weights)
    >>> print(f"Optimization took {timer.elapsed_ms:.2f}ms")
    """

    def __init__(
        self,
        operation_name: str,
        logger_instance: Logger | None = None,
        log_start: bool = False,
        details: dict[str, Any] | None = None,
    ) -> None:
        """Initialize the performance timer."""
        self.operation_name = operation_name
        self._logger = logger_instance or logger
        self.log_start = log_start
        self.details = details or {}
        self.start_time: float = 0.0
        self.end_time: float = 0.0
        self.elapsed_ms: float = 0.0

    def __enter__(self) -> Self:
        """Start the timer."""
        self.start_time = time.perf_counter()

        if self.log_start:
            self._logger.bind(
                event_type="PERFORMANCE_START",
                data={
                    "operation": self.operation_name,
                    "details": self.details,
                },
            ).debug(f"Performance: Starting {self.operation_name}")

        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: types.TracebackType | None,
    ) -> None:
        """Stop the timer and log the result."""
        self.end_time = time.perf_counter()
        self.elapsed_ms = (self.end_time - self.start_time) * 1000

        log_data = {
            "operation": self.operation_name,
            "details": self.details,
            "success": exc_type is None,
        }

        if exc_type is not None:
            log_data["error_type"] = exc_type.__name__

        self._logger.bind(
            event_type="PERFORMANCE",
            execution_time_ms=self.elapsed_ms,
            data=log_data,
        ).debug(f"Performance: {self.operation_name} completed in {self.elapsed_ms:.2f}ms")


# ==============================================================================
# Decorator for Function Logging
# ==============================================================================


def log_function_call(
    log_args: bool = True,
    log_result: bool = False,
    log_performance: bool = True,
    audit_event_type: str | None = None,
) -> Callable[[Callable[P, R]], Callable[P, R]]:
    """Decorate a function to automatically log its calls.

    Parameters
    ----------
    log_args : bool
        Whether to log function arguments.
    log_result : bool
        Whether to log function return value.
    log_performance : bool
        Whether to log execution time.
    audit_event_type : str, optional
        If provided, logs as audit event with this type.

    Returns
    -------
    Callable[[Callable[P, R]], Callable[P, R]]
        Decorator that wraps functions with logging.

    Example
    -------
    >>> @log_function_call(audit_event_type="CALCULATION")
    ... def calculate_returns(portfolio_weights, prices):
    ...     # ... calculation logic ...
    ...     return returns
    """

    def decorator(func: Callable[P, R]) -> Callable[P, R]:
        @wraps(func)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            func_name = func.__name__  # ty: ignore[unresolved-attribute]
            module_name = func.__module__

            # Prepare log data
            log_data: dict[str, Any] = {
                "function": func_name,
                "module": module_name,
            }

            if log_args:
                # Sanitize args (avoid logging sensitive data)
                log_data["args_count"] = len(args)
                log_data["kwargs_keys"] = list(kwargs.keys())

            start_time = time.perf_counter()

            try:
                result = func(*args, **kwargs)
                execution_time_ms = (time.perf_counter() - start_time) * 1000

                # Build bound logger
                bound_logger = logger.bind(
                    execution_time_ms=execution_time_ms if log_performance else None,
                    data=log_data,
                )

                if audit_event_type:
                    bound_logger = bound_logger.bind(event_type=audit_event_type)

                if log_result and result is not None:
                    log_data["result_type"] = type(result).__name__

                bound_logger.debug(
                    f"Function {func_name} completed in {execution_time_ms:.2f}ms",
                )

                return result

            except Exception as exc:
                execution_time_ms = (time.perf_counter() - start_time) * 1000
                log_data["error_type"] = type(exc).__name__
                log_data["error_message"] = str(exc)

                logger.bind(
                    event_type="FUNCTION_ERROR" if audit_event_type else None,
                    execution_time_ms=execution_time_ms,
                    data=log_data,
                ).error(f"Function {func_name} failed: {exc}")

                raise

        return wrapper

    return decorator


# ==============================================================================
# Module Initialization
# ==============================================================================


# Export public API
__all__ = [
    # Constants
    "DEFAULT_LOG_DIR",
    "DEFAULT_LOG_LEVEL",
    "DEFAULT_RETENTION_DAYS_APPLICATION",
    "DEFAULT_RETENTION_DAYS_AUDIT",
    "DEFAULT_RETENTION_DAYS_ERROR",
    "DEFAULT_RETENTION_DAYS_PERFORMANCE",
    "DEFAULT_ROTATION_SIZE",
    # Classes
    "Audit_Logger",
    "Config_Logging",
    # Type Aliases
    "Log_Filter_Func",
    "Log_Format_Func",
    "Log_Record_Dict",
    "Performance_Timer",
    # Filters
    "filter_audit_events",
    "filter_error_level",
    "filter_performance_events",
    # Formatters
    "format_record_as_JSON",
    "format_sink_JSON",
    "format_sink_human_readable",
    # Functions
    "get_logger",
    # Decorators
    "log_function_call",
    "log_performance",
    "serialize_for_JSON",
    "setup_logging",
]


# ==============================================================================
# Example Usage (when run directly)
# ==============================================================================


if __name__ == "__main__":
    import os

    # Configure logging at startup
    setup_logging(
        log_level="DEBUG",
        environment=os.getenv("ENVIRONMENT", "development"),
        enable_console=True,
        enable_JSON=True,
    )

    # Get logger instance
    app_logger = get_logger(__name__)

    # Basic logging examples
    app_logger.info("Application initialized")
    app_logger.debug("Debug message with details", extra={"key": "value"})
    app_logger.warning("Warning message")

    # Audit logging example
    audit = Audit_Logger("example_module")
    audit.log_calculation(
        operation="Option Pricing (Black-Scholes)",
        inputs={"spot": 100, "strike": 105, "volatility": 0.2, "rate": 0.05},
        result=3.45,
        execution_time_ms=12.5,
        user_id="analyst_001",
    )

    audit.log_data_access(
        source="inputs/raw/data_ETFs.csv",
        operation="READ",
        record_count=252,
        user_id="analyst_001",
    )

    audit.log_parameter_change(
        parameter_name="risk_free_rate",
        old_value=0.02,
        new_value=0.025,
        user_id="admin_001",
        reason="Updated to reflect current market conditions",
    )

    # Performance timer example
    with Performance_Timer("example_operation", log_start=True) as timer:
        # Simulate some work
        time.sleep(0.1)

    app_logger.info(f"Operation completed in {timer.elapsed_ms:.2f}ms")

    # Decorator example
    @log_function_call(log_performance=True, audit_event_type="CALCULATION")
    def example_calculation(
        value_a: float,
        value_b: float,
    ) -> float:
        """Calculate example values for demonstration."""
        time.sleep(0.05)  # Simulate work
        return value_a + value_b

    result = example_calculation(10.5, 20.3)
    app_logger.info(f"Calculation result: {result}")
