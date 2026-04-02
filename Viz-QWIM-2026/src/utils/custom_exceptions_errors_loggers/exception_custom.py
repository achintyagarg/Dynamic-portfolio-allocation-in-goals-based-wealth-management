"""Custom Exception Classes for QWIM Projects.

============================================

This module defines specialized exception classes with enhanced traceback
capabilities combining `rich.traceback` and `traceback-with-variables` for
detailed error reporting.

Features
--------
- Multiple Exception Formats: Supports simple, complex, traceback, and custom modes
- Automatic Context Capture: Extracts filename, function, line number, and more
- Rich Formatting: Supports JSON, rich table output, and structured logging
- Thread-Safe: Uses locking to ensure safe exception handling
- Variable Inspection: Displays local variables in traceback frames

Classes
-------
Exception_Format
    Enum defining exception output formats.
Exception_Context
    Dataclass capturing exception context information.
Exception_Custom
    Enhanced base exception class with rich traceback support.

Domain-Specific Exceptions
--------------------------
Exception_Validation_Input
    Raised when input validation fails.
Exception_Invalid_Input
    Raised when input is invalid or malformed.
Exception_Configuration
    Raised for configuration-related errors.
Exception_Data_Not_Found
    Raised when required data is missing.
Exception_Not_Found
    Raised when a requested resource is not found.
Exception_Calculation
    Raised for calculation/computation failures.
Exception_Portfolio
    Raised for portfolio-related errors.
Exception_Client
    Raised for client data errors.
Exception_Database
    Raised for database operation failures.
Exception_API
    Raised for API communication failures.
Exception_Authentication
    Raised for authentication failures.
Exception_Authorization
    Raised for authorization/permission failures.
Exception_File_Operation
    Raised for file I/O failures.
Exception_Timeout
    Raised when operations exceed time limits.
Exception_Security_Violation
    Raised for security policy violations.
Exception_Insufficient_Holdings
    Raised when portfolio has insufficient holdings.
Exception_Invalid_Transaction
    Raised when a transaction is invalid.

Aliases
-------
Exception_Validation
    Alias for Exception_Validation_Input.

Example
-------
>>> from src.utils.custom_exceptions_errors_loggers.exception_custom import (
...     Exception_Custom,
...     Exception_Validation_Input,
...     Exception_Format,
...     Exception_Invalid_Input,
...     Exception_Not_Found,
... )
>>>
>>> # Simple exception
>>> raise Exception_Validation_Input("Invalid portfolio weight")
>>>
>>> # Exception with context
>>> raise Exception_Custom(
...     message="Calculation failed",
...     exception_format=Exception_Format.RICH_TRACEBACK,
...     context={"portfolio_id": "PORT_001", "operation": "optimize"},
... )
>>>
>>> # Resource not found
>>> raise Exception_Not_Found(
...     "Portfolio not found",
...     resource_type="Portfolio",
...     resource_id="PORT_001",
... )

Notes
-----
- All exceptions are thread-safe and can be used in multi-threaded applications
- Rich traceback requires terminal support for ANSI colors
- JSON format is suitable for structured logging and API responses
- Use aenum instead of built-in Enum per project coding standards

Author: QWIM Team
Version: 0.6.0
Last Updated: 2026-02-09
"""

from __future__ import annotations

import contextlib
import inspect
import io
import json
import linecache
import os
import sys
import threading

from collections.abc import Callable, Generator
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import UTC, datetime
from io import StringIO
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar, ParamSpec, Self, TypeVar

# Use aenum instead of built-in Enum per project coding standards
from aenum import Enum, auto


# CRITICAL: Force UTF-8 encoding on Windows BEFORE importing rich/tbhandler
# This prevents UnicodeEncodeError when rich console displays special characters
if sys.platform == "win32":
    # Ensure all standard streams use UTF-8
    if hasattr(sys.stdout, "reconfigure"):
        with contextlib.suppress(Exception):
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
    if hasattr(sys.stderr, "reconfigure"):
        with contextlib.suppress(Exception):
            sys.stderr.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
    if hasattr(sys.stdin, "reconfigure"):
        with contextlib.suppress(Exception):
            sys.stdin.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]

# Rich for beautiful traceback formatting
from rich.console import Console
from rich.panel import Panel
from rich.syntax import Syntax
from rich.table import Table
from rich.text import Text

# Traceback with variables for enhanced debugging
from traceback_with_variables import iter_exc_lines


if TYPE_CHECKING:
    from collections.abc import Callable
    from types import TracebackType


# ==============================================================================
# Type Aliases (Python 3.12+)
# ==============================================================================

type Exception_Context_Dict = dict[str, Any]
type Exception_Frame_List = list["Exception_Frame"]
type Sensitive_Field_Set = frozenset[str]


# ==============================================================================
# UTF-8 Console Monkey Patch for Windows
# ==============================================================================

# Monkey patch rich.console.Console to always use UTF-8 on Windows
# This ensures tbhandler and any other code using rich will not encounter
# UnicodeEncodeError with special characters like ❱, ✓, etc.
if sys.platform == "win32":
    # Store original Console.__init__
    _original_console_init = Console.__init__

    def _patched_console_init(
        self: Console,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        """Patched Console.__init__ that forces UTF-8 encoding on Windows."""
        # If file/stderr not specified, wrap stderr with UTF-8
        if "file" not in kwargs and not any(arg == sys.stderr for arg in args):
            # Create UTF-8 wrapper for stderr
            kwargs["file"] = io.TextIOWrapper(
                sys.stderr.buffer,
                encoding="utf-8",
                errors="replace",
                line_buffering=True,
            )
        # Set legacy_windows to False for better Unicode support
        kwargs.setdefault("legacy_windows", False)
        # Call original init
        _original_console_init(self, *args, **kwargs)

    # Apply monkey patch
    Console.__init__ = _patched_console_init


# ==============================================================================
# Constants
# ==============================================================================

# Thread lock for safe exception handling
_EXCEPTION_LOCK = threading.RLock()

# Default console for rich output with UTF-8 encoding
# Force UTF-8 to prevent UnicodeEncodeError with special characters like ❱
_stderr_utf8 = io.TextIOWrapper(
    sys.stderr.buffer,
    encoding="utf-8",
    errors="replace",
    line_buffering=True,
)
_CONSOLE = Console(file=_stderr_utf8, force_terminal=True, legacy_windows=False)

# Maximum frames to display in traceback
MAX_TRACEBACK_FRAMES = 20

# Maximum variable string length before truncation
MAX_VARIABLE_LENGTH = 500

# Sensitive field names to mask in output
SENSITIVE_FIELDS = frozenset(
    {
        "password",
        "secret",
        "token",
        "api_key",
        "apikey",
        "api_secret",
        "private_key",
        "credentials",
        "ssn",
        "social_security",
        "credit_card",
        "card_number",
        "cvv",
        "pin",
    },
)


# ==============================================================================
# Enums
# ==============================================================================


class Exception_Format(Enum):  # type: ignore[misc]
    """Error output format modes.

    Per project coding standards, exception-related enums use Exception_ prefix.

    Attributes
    ----------
    SIMPLE : auto
        Basic error message only.
    STANDARD : auto
        Standard Python traceback format.
    RICH_TRACEBACK : auto
        Rich formatted traceback with colors and syntax highlighting.
    VARIABLES : auto
        Traceback with local variable values displayed.
    JSON : auto
        JSON-formatted error for structured logging.
    TABLE : auto
        Rich table format for console display.
    FULL : auto
        Combines rich traceback with variables and context.
    """

    SIMPLE = auto()
    STANDARD = auto()
    RICH_TRACEBACK = auto()
    VARIABLES = auto()
    JSON = auto()
    TABLE = auto()
    FULL = auto()


class Exception_Severity(Enum):  # type: ignore[misc]
    """Error severity levels.

    Per project coding standards, exception-related enums use Exception_ prefix.

    Attributes
    ----------
    DEBUG : auto
        Debug-level error for development.
    INFO : auto
        Informational error.
    WARNING : auto
        Warning-level error.
    ERROR : auto
        Error-level (default).
    CRITICAL : auto
        Critical error requiring immediate attention.
    """

    DEBUG = auto()
    INFO = auto()
    WARNING = auto()
    ERROR = auto()
    CRITICAL = auto()


# ==============================================================================
# Data Classes
# ==============================================================================


@dataclass(frozen=True, slots=True)
class Exception_Frame:
    """Represents a single frame in the error traceback.

    Per project coding standards, exception-related classes use Exception_ prefix.

    Attributes
    ----------
    filename : str
        Source file path.
    function : str
        Function name where error occurred.
    line_number : int
        Line number in source file.
    code_context : str
        Source code line at error location.
    local_variables : dict[str, Any]
        Local variables at error location.
    module : str
        Module name.
    """

    filename: str
    function: str
    line_number: int
    code_context: str
    local_variables: dict[str, Any] = field(default_factory=dict)
    module: str = ""


@dataclass(slots=True)
class Exception_Context:
    """Captures comprehensive error context information.

    Per project coding standards, exception-related classes use Exception_ prefix.

    Attributes
    ----------
    timestamp : datetime
        UTC timestamp when error occurred.
    exception_type : str
        Error class name.
    message : str
        Error message.
    filename : str
        Source file where error originated.
    function : str
        Function where error originated.
    line_number : int
        Line number where error originated.
    code_context : str
        Source code at error location.
    frames : list[Exception_Frame]
        Stack frames leading to error.
    thread_id : int
        Thread ID where error occurred.
    thread_name : str
        Thread name where error occurred.
    process_id : int
        Process ID.
    user_context : dict[str, Any]
        User-provided context data.
    severity : Exception_Severity
        Error severity level.
    exception_id : str
        Unique error identifier.
    """

    timestamp: datetime = field(default_factory=lambda: datetime.now(UTC))
    exception_type: str = ""
    message: str = ""
    filename: str = ""
    function: str = ""
    line_number: int = 0
    code_context: str = ""
    frames: list[Exception_Frame] = field(default_factory=list)
    thread_id: int = 0
    thread_name: str = ""
    process_id: int = 0
    user_context: dict[str, Any] = field(default_factory=dict)
    severity: Exception_Severity = Exception_Severity.ERROR  # type: ignore[assignment]
    exception_id: str = ""

    def __post_init__(self) -> None:
        """Initialize computed fields after creation."""
        import os

        if not self.thread_id:
            self.thread_id = threading.current_thread().ident or 0
        if not self.thread_name:
            self.thread_name = threading.current_thread().name
        if not self.process_id:
            self.process_id = os.getpid()
        if not self.exception_id:
            self.exception_id = f"EXC_{self.timestamp.strftime('%Y%m%d%H%M%S%f')}"


# ==============================================================================
# Helper Functions
# ==============================================================================


def _mask_sensitive_data(data: dict[str, Any]) -> dict[str, Any]:
    """Mask sensitive data in dictionary values.

    Parameters
    ----------
    data : dict[str, Any]
        Dictionary potentially containing sensitive data.

    Returns
    -------
    dict[str, Any]
        Dictionary with sensitive values masked.
    """
    masked = {}
    for key, value in data.items():
        key_lower = key.lower()
        if any(sensitive in key_lower for sensitive in SENSITIVE_FIELDS):
            masked[key] = "***MASKED***"
        elif isinstance(value, dict):
            masked[key] = _mask_sensitive_data(value)
        else:
            masked[key] = value
    return masked


def _truncate_value(value: Any, max_length: int = MAX_VARIABLE_LENGTH) -> str:
    """Truncate value representation if too long.

    Parameters
    ----------
    value : Any
        Value to represent as string.
    max_length : int
        Maximum string length before truncation.

    Returns
    -------
    str
        Truncated string representation.
    """
    try:
        str_value = repr(value)
        if len(str_value) > max_length:
            return str_value[: max_length - 3] + "..."
        return str_value
    except Exception:
        return "<unrepresentable>"


def _serialize_for_JSON(obj: Any) -> Any:
    """Serialize object for JSON output.

    Parameters
    ----------
    obj : Any
        Object to serialize.

    Returns
    -------
    Any
        JSON-serializable representation.
    """
    if isinstance(obj, datetime):
        return obj.isoformat()
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, bytes):
        return obj.decode("utf-8", errors="replace")
    if isinstance(obj, Enum):
        return obj.name  # type: ignore[attr-defined]
    if isinstance(obj, Exception):
        return str(obj)
    if hasattr(obj, "__dict__"):
        return {k: _serialize_for_JSON(v) for k, v in obj.__dict__.items()}
    return str(obj)


def _extract_frames_from_traceback(
    tb: TracebackType | None,
    max_frames: int = MAX_TRACEBACK_FRAMES,
) -> list[Exception_Frame]:
    """Extract detailed frame information from traceback.

    Parameters
    ----------
    tb : TracebackType | None
        Python traceback object.
    max_frames : int
        Maximum number of frames to extract.

    Returns
    -------
    list[Exception_Frame]
        List of exception frames with context.
    """
    frames = []
    frame_count = 0

    while tb is not None and frame_count < max_frames:
        frame = tb.tb_frame
        lineno = tb.tb_lineno
        filename = frame.f_code.co_filename
        function = frame.f_code.co_name
        module = frame.f_globals.get("__name__", "")

        # Get code context
        code_context = linecache.getline(filename, lineno).strip()

        # Extract local variables (mask sensitive data)
        local_vars = {}
        try:
            for var_name, var_value in frame.f_locals.items():
                if not var_name.startswith("_"):
                    local_vars[var_name] = _truncate_value(var_value)
            local_vars = _mask_sensitive_data(local_vars)
        except Exception:
            pass

        frames.append(
            Exception_Frame(
                filename=filename,
                function=function,
                line_number=lineno,
                code_context=code_context,
                local_variables=local_vars,
                module=module,
            ),
        )

        tb = tb.tb_next
        frame_count += 1

    return frames


# ==============================================================================
# Main Exception Class
# ==============================================================================


class Exception_Custom(Exception):
    """Enhanced base exception class with rich traceback support.

    This exception class provides comprehensive error reporting with multiple
    output formats, automatic context capture, and thread-safe operation.

    Parameters
    ----------
    message : str
        Primary exception message.
    exception_format : Exception_Format
        Output format mode (default: STANDARD).
    severity : Exception_Severity
        Exception severity level (default: ERROR).
    context : dict[str, Any] | None
        Additional context data to include.
    cause : Exception | None
        Original exception that caused this one.
    suppress_traceback : bool
        Whether to suppress traceback output (default: False).

    Attributes
    ----------
    exception_context : Exception_Context
        Captured exception context information.
    exception_format : Exception_Format
        Current output format mode.

    Class Attributes
    ----------------
    default_format : Exception_Format
        Default format for all instances.
    enable_variable_capture : bool
        Whether to capture local variables.
    console : Console
        Rich console for output.

    Examples
    --------
    >>> # Simple usage
    >>> raise Exception_Custom("Something went wrong")

    >>> # With rich traceback
    >>> raise Exception_Custom(
    ...     "Calculation failed",
    ...     exception_format=Exception_Format.RICH_TRACEBACK,
    ...     context={"operation": "portfolio_optimization"},
    ... )

    >>> # Get JSON representation
    >>> try:
    ...     risky_operation()
    ... except Exception as e:
    ...     exc = Exception_Custom.from_exception(e)
    ...     print(exc.to_JSON())
    """

    # Class-level configuration
    default_format: ClassVar[Exception_Format] = Exception_Format.STANDARD  # type: ignore[assignment]
    enable_variable_capture: ClassVar[bool] = True
    console: ClassVar[Console] = _CONSOLE

    def __init__(
        self,
        message: str,
        *args: Any,
        exception_format: Exception_Format | None = None,
        severity: Exception_Severity = Exception_Severity.ERROR,  # type: ignore[assignment]
        context: dict[str, Any] | None = None,
        cause: Exception | None = None,
        suppress_traceback: bool = False,
    ) -> None:
        """Initialize enhanced exception with context capture."""
        super().__init__(message, *args)

        self._message = message

        # Determine output format
        if exception_format is not None:
            self._exception_format = exception_format
        else:
            # Check environment variable for default format override
            env_fmt = os.environ.get("QWIM_EXCEPTION_FORMAT")
            if env_fmt and env_fmt in Exception_Format.__members__:
                self._exception_format = Exception_Format[env_fmt]  # type: ignore[index]
            else:
                self._exception_format = self.default_format

        self._severity = severity
        self._user_context = context or {}
        self._cause = cause
        self._suppress_traceback = suppress_traceback
        self._exception_context: Exception_Context | None = None

        # Capture context automatically
        with _EXCEPTION_LOCK:
            self._capture_context()

        # Set __cause__ for exception chaining
        if cause is not None:
            self.__cause__ = cause

    def _capture_context(self) -> None:
        """Capture exception context from current execution state."""
        # Get current exception info
        exc_type, exc_value, exc_tb = sys.exc_info()

        # If no active exception, use inspection to get caller info
        if exc_tb is None:
            # Get caller frame (skip __init__ and _capture_context)
            frame = inspect.currentframe()
            if frame is not None:
                # Walk up the stack to find the actual caller
                # Base depth: 2 (_capture_context, __init__)
                depth = 2
                # If this is a subclass instantiating via its own __init__, we need to skip one more
                if self.__class__.__name__ != "Exception_Custom":
                    depth = 3

                for _ in range(depth):  # Skip internal frames
                    if frame.f_back is not None:
                        frame = frame.f_back

                filename = frame.f_code.co_filename
                function = frame.f_code.co_name
                line_number = frame.f_lineno
                code_context = linecache.getline(filename, line_number).strip()

                # Capture stack frames from this point
                captured_frames = []
                current_frame = frame
                frame_count = 0

                while current_frame is not None and frame_count < MAX_TRACEBACK_FRAMES:
                    try:
                        f_locals = {}
                        if self.enable_variable_capture:
                            for k, v in current_frame.f_locals.items():
                                if not k.startswith("_"):
                                    f_locals[k] = _truncate_value(v)
                            f_locals = _mask_sensitive_data(f_locals)

                        f_code = current_frame.f_code
                        captured_frames.append(
                            Exception_Frame(
                                filename=f_code.co_filename,
                                function=f_code.co_name,
                                line_number=current_frame.f_lineno,
                                code_context=linecache.getline(
                                    f_code.co_filename,
                                    current_frame.f_lineno,
                                ).strip(),
                                local_variables=f_locals,
                                module=current_frame.f_globals.get("__name__", ""),
                            ),
                        )
                    except Exception:
                        pass

                    current_frame = current_frame.f_back
                    frame_count += 1

                self._exception_context = Exception_Context(
                    exception_type=self.__class__.__name__,
                    message=self._message,
                    filename=filename,
                    function=function,
                    line_number=line_number,
                    code_context=code_context,
                    frames=captured_frames,
                    user_context=_mask_sensitive_data(self._user_context),
                    severity=self._severity,
                )
        else:
            # Extract frames from actual traceback
            frames = _extract_frames_from_traceback(exc_tb)

            # Get origin frame info
            origin_frame = frames[0] if frames else None

            self._exception_context = Exception_Context(
                exception_type=exc_type.__name__ if exc_type else self.__class__.__name__,
                message=str(exc_value) if exc_value else self._message,
                filename=origin_frame.filename if origin_frame else "",
                function=origin_frame.function if origin_frame else "",
                line_number=origin_frame.line_number if origin_frame else 0,
                code_context=origin_frame.code_context if origin_frame else "",
                frames=frames,
                user_context=_mask_sensitive_data(self._user_context),
                severity=self._severity,
            )

    @classmethod
    def from_exception(
        cls,
        exception: Exception,
        exception_format: Exception_Format | None = None,
        context: dict[str, Any] | None = None,
        severity: Exception_Severity = Exception_Severity.ERROR,  # type: ignore[assignment]
    ) -> Self:
        """Create Exception_Custom from an existing exception.

        Parameters
        ----------
        exception : Exception
            Original exception to wrap.
        exception_format : Exception_Format | None
            Output format mode.
        context : dict[str, Any] | None
            Additional context data.
        severity : Exception_Severity
            Exception severity level.

        Returns
        -------
        Self
            New Exception_Custom instance wrapping the original.
        """
        return cls(
            message=str(exception),
            exception_format=exception_format,
            cause=exception,
            context=context,
            severity=severity,
        )

    @property
    def message(self) -> str:
        """Get exception message.

        Returns
        -------
        str
            The exception message.
        """
        return self._message

    @property
    def severity(self) -> Exception_Severity:
        """Get exception severity level.

        Returns
        -------
        Exception_Severity
            The severity level.
        """
        return self._severity

    @property
    def exception_context(self) -> Exception_Context:
        """Get captured exception context."""
        if self._exception_context is None:
            self._capture_context()
        assert self._exception_context is not None, (
            "Exception context must be set after _capture_context()"
        )
        return self._exception_context

    @property
    def exception_format(self) -> Exception_Format:
        """Get current exception format."""
        return self._exception_format

    @exception_format.setter
    def exception_format(self, value: Exception_Format) -> None:
        """Set exception format."""
        self._exception_format = value

    def __str__(self) -> str:
        """Return formatted exception string based on current format."""
        if self._suppress_traceback:
            return self._message

        if self._exception_format == Exception_Format.SIMPLE:
            return self._format_simple()
        if self._exception_format == Exception_Format.JSON:
            return self.to_JSON()
        if self._exception_format in {
            Exception_Format.RICH_TRACEBACK,
            Exception_Format.TABLE,
            Exception_Format.FULL,
        }:
            # Rich formats should use print_rich() for proper display
            return self._format_simple()

        return self._format_standard()

    def __repr__(self) -> str:
        """Return detailed representation."""
        return (
            f"{self.__class__.__name__}("
            f"message={self._message!r}, "
            f"format={self._exception_format.name}, "
            f"severity={self._severity.name})"
        )

    def __reduce__(self) -> tuple[Any, ...]:
        """Support for pickling (serialization).

        Returns
        -------
        tuple
            Reduce tuple for reconstruction.
        """
        return (
            self.__class__,
            (
                self._message,
                None,  # args
                self._exception_format,
                self._severity,
                self._user_context,
                self._cause,
                self._suppress_traceback,
            ),
        )

    def log(self, logger: Any | None = None, level: str | None = None) -> None:
        """Log this exception using a provided logger or default custom logger.

        Parameters
        ----------
        logger : Any | None
            Logger instance. If None, uses src.utils.custom_exceptions_errors_loggers.logger_custom.get_logger.
        level : str | None
            Log level (e.g., "ERROR", "WARNING"). If None, uses exception severity.
        """
        log_level = level or self._severity.name
        message = str(self)

        # Default to custom logger if none provided
        if logger is None:
            try:
                from src.utils.custom_exceptions_errors_loggers.logger_custom import (
                    get_logger,
                )

                # Try to use filename from context as logger name
                name = self.exception_context.filename or "exception_custom"
                logger = get_logger(name)
            except (ImportError, RuntimeError):
                # Fallback if logger_custom not available or setup
                print(f"[{log_level}] {message}", file=sys.stderr)
                return

        if hasattr(logger, "opt"):  # Loguru (from logger_custom)
            # Use opt(exception=self) to include structured traceback info
            # This leverages loguru's backtrace/diagnose features
            logger.opt(exception=self).log(log_level, message)
        elif hasattr(logger, "log"):  # Standard logging fallback
            import logging

            level_mapping = {
                "DEBUG": logging.DEBUG,
                "INFO": logging.INFO,
                "WARNING": logging.WARNING,
                "ERROR": logging.ERROR,
                "CRITICAL": logging.CRITICAL,
            }
            lvl = level_mapping.get(log_level.upper(), logging.ERROR)
            logger.log(lvl, message, exc_info=True)
        else:
            # Simple print fallback
            print(f"[{log_level}] {message}", file=sys.stderr)

    def _format_simple(self) -> str:
        """Format exception as simple message.

        Returns
        -------
        str
            Simple exception message.
        """
        ctx = self.exception_context
        return f"{ctx.exception_type}: {self._message}"

    def _format_standard(self) -> str:
        """Format exception with standard Python traceback style.

        Returns
        -------
        str
            Standard formatted traceback.
        """
        ctx = self.exception_context
        lines = ["Traceback (most recent call last):"]

        for frame in ctx.frames:
            lines.append(
                f'  File "{frame.filename}", line {frame.line_number}, in {frame.function}',
            )
            if frame.code_context:
                lines.append(f"    {frame.code_context}")

        lines.append(f"{ctx.exception_type}: {self._message}")

        if ctx.user_context:
            lines.append("\nContext:")
            for key, value in ctx.user_context.items():
                lines.append(f"  {key}: {value}")

        return "\n".join(lines)

    def to_JSON(self, indent: int = 2) -> str:
        """Convert exception to JSON string.

        Parameters
        ----------
        indent : int
            JSON indentation level (default: 2).

        Returns
        -------
        str
            JSON-formatted exception.
        """
        ctx = self.exception_context
        data = {
            "exception_id": ctx.exception_id,
            "timestamp": ctx.timestamp.isoformat(),
            "exception_type": ctx.exception_type,
            "message": self._message,
            "severity": ctx.severity.name,
            "location": {
                "filename": ctx.filename,
                "function": ctx.function,
                "line_number": ctx.line_number,
                "code_context": ctx.code_context,
            },
            "thread": {
                "id": ctx.thread_id,
                "name": ctx.thread_name,
            },
            "process_id": ctx.process_id,
            "context": ctx.user_context,
            "frames": [
                {
                    "filename": f.filename,
                    "function": f.function,
                    "line_number": f.line_number,
                    "code_context": f.code_context,
                    "module": f.module,
                    "variables": f.local_variables if self.enable_variable_capture else {},
                }
                for f in ctx.frames
            ],
        }

        if self._cause is not None:
            data["caused_by"] = {
                "type": type(self._cause).__name__,
                "message": str(self._cause),
            }

        return json.dumps(data, default=_serialize_for_JSON, indent=indent)

    def to_dict(self) -> dict[str, Any]:
        """Convert exception to dictionary.

        Returns
        -------
        dict[str, Any]
            Dictionary representation of exception.
        """
        return json.loads(self.to_JSON())

    def print_rich(self, console: Console | None = None) -> None:
        """Print rich formatted traceback to console.

        Parameters
        ----------
        console : Console | None
            Rich console to use (default: class console).
        """
        console = console or self.console
        ctx = self.exception_context

        with _EXCEPTION_LOCK:
            if self._exception_format == Exception_Format.TABLE:
                self._print_table(console, ctx)
            elif self._exception_format == Exception_Format.FULL:
                self._print_full(console, ctx)
            else:
                self._print_rich_traceback(console, ctx)

    def _print_rich_traceback(self, console: Console, ctx: Exception_Context) -> None:
        """Print rich-formatted traceback.

        Parameters
        ----------
        console : Console
            Rich console for output.
        ctx : Exception_Context
            Exception context information.
        """
        # Create header panel
        header = Text()
        header.append(f"⚠️  {ctx.exception_type}", style="bold red")
        header.append(f"\n{self._message}", style="white")

        console.print(Panel(header, title="Exception", border_style="red"))

        # Print location info
        location_table = Table(show_header=False, box=None)
        location_table.add_row("📁 File:", ctx.filename)
        location_table.add_row("📍 Function:", ctx.function)
        location_table.add_row("📌 Line:", str(ctx.line_number))
        location_table.add_row("🕐 Time:", ctx.timestamp.strftime("%Y-%m-%d %H:%M:%S UTC"))
        console.print(location_table)

        # Print code context with syntax highlighting
        if ctx.code_context:
            console.print("\n[bold]Code Context:[/bold]")
            syntax = Syntax(
                ctx.code_context,
                "python",
                theme="monokai",
                line_numbers=True,
                start_line=ctx.line_number,
            )
            console.print(syntax)

        # Print user context if available
        if ctx.user_context:
            console.print("\n[bold]Context Data:[/bold]")
            context_table = Table(show_header=True, header_style="bold cyan")
            context_table.add_column("Key")
            context_table.add_column("Value")
            for key, value in ctx.user_context.items():
                context_table.add_row(str(key), str(value))
            console.print(context_table)

        # Print stack frames if available
        if ctx.frames and len(ctx.frames) > 1:
            console.print("\n[bold]Stack Trace:[/bold]")
            for i, frame in enumerate(ctx.frames):
                console.print(
                    f"  [dim]#{i + 1}[/dim] {frame.function} "
                    f"[dim]({frame.filename}:{frame.line_number})[/dim]",
                )

    def _print_table(self, console: Console, ctx: Exception_Context) -> None:
        """Print exception as rich table.

        Parameters
        ----------
        console : Console
            Rich console for output.
        ctx : Exception_Context
            Exception context information.
        """
        table = Table(
            title=f"Exception: {ctx.exception_type}",
            show_header=True,
            header_style="bold red",
        )
        table.add_column("Property", style="cyan")
        table.add_column("Value", style="white")

        table.add_row("Exception ID", ctx.exception_id)
        table.add_row("Timestamp", ctx.timestamp.strftime("%Y-%m-%d %H:%M:%S UTC"))
        table.add_row("Severity", ctx.severity.name)
        table.add_row("Message", self._message)
        table.add_row("File", ctx.filename)
        table.add_row("Function", ctx.function)
        table.add_row("Line", str(ctx.line_number))
        table.add_row("Thread", f"{ctx.thread_name} ({ctx.thread_id})")
        table.add_row("Process ID", str(ctx.process_id))

        if ctx.user_context:
            table.add_row("Context", json.dumps(ctx.user_context, indent=2))

        console.print(table)

    def _print_full(self, console: Console, ctx: Exception_Context) -> None:
        """Print full exception with variables.

        Parameters
        ----------
        console : Console
            Rich console for output.
        ctx : Exception_Context
            Exception context information.
        """
        # Print rich traceback first
        self._print_rich_traceback(console, ctx)

        # Print variables from each frame
        if self.enable_variable_capture and ctx.frames:
            console.print("\n[bold magenta]Local Variables by Frame:[/bold magenta]")
            for i, frame in enumerate(ctx.frames):
                if frame.local_variables:
                    var_table = Table(
                        title=f"Frame #{i + 1}: {frame.function}",
                        show_header=True,
                    )
                    var_table.add_column("Variable", style="green")
                    var_table.add_column("Value", style="white")
                    for var_name, var_value in frame.local_variables.items():
                        var_table.add_row(var_name, str(var_value))
                    console.print(var_table)

        # Print traceback with variables using traceback_with_variables
        if self.__cause__ is not None:
            console.print("\n[bold yellow]Traceback with Variables:[/bold yellow]")
            try:
                output = StringIO()
                for line in iter_exc_lines(self.__cause__):
                    output.write(line + "\n")
                console.print(output.getvalue())
            except Exception:
                pass

    def get_traceback_string(self) -> str:
        """Get traceback as string using traceback-with-variables.

        Returns
        -------
        str
            Formatted traceback string with variables.
        """
        output = StringIO()
        if self.__cause__ is not None:
            for line in iter_exc_lines(self.__cause__):
                output.write(line + "\n")
        else:
            output.write(self._format_standard())
        return output.getvalue()

    @classmethod
    def install_rich_traceback(cls, **kwargs: Any) -> None:
        """Install rich traceback handler globally.

        Parameters
        ----------
        **kwargs : Any
            Arguments passed to rich.traceback.install().
        """
        from rich.traceback import install

        install(
            console=cls.console,
            show_locals=cls.enable_variable_capture,
            **kwargs,
        )


# ==============================================================================
# Domain-Specific Exception Classes
# ==============================================================================


class Exception_Validation_Input(Exception_Custom):
    """Exception raised when input validation fails.

    Use this for validating user inputs, parameters, or data formats.

    Parameters
    ----------
    message : str
        Validation error message.
    field_name : str | None
        Name of the invalid field.
    expected_type : type | None
        Expected type for the field.
    actual_value : Any | None
        Actual value that failed validation.
    **kwargs : Any
        Additional Exception_Custom parameters.

    Examples
    --------
    >>> raise Exception_Validation_Input(
    ...     "Invalid portfolio weight",
    ...     field_name="weight",
    ...     expected_type=float,
    ...     actual_value="not_a_number",
    ... )
    """

    def __init__(
        self,
        message: str,
        field_name: str | None = None,
        expected_type: type | None = None,
        actual_value: Any = None,
        **kwargs: Any,
    ) -> None:
        context = kwargs.pop("context", {})
        if field_name:
            context["field_name"] = field_name
        if expected_type:
            context["expected_type"] = expected_type.__name__
        if actual_value is not None:
            context["actual_value"] = _truncate_value(actual_value)
        super().__init__(message, context=context, **kwargs)


class Exception_Configuration(Exception_Custom):
    """Exception raised for configuration-related errors.

    Use this for missing or invalid configuration settings.

    Parameters
    ----------
    message : str
        Configuration error message.
    config_key : str | None
        Configuration key that caused the error.
    config_file : str | Path | None
        Configuration file path.
    **kwargs : Any
        Additional Exception_Custom parameters.
    """

    def __init__(
        self,
        message: str,
        config_key: str | None = None,
        config_file: str | Path | None = None,
        **kwargs: Any,
    ) -> None:
        context = kwargs.pop("context", {})
        if config_key:
            context["config_key"] = config_key
        if config_file:
            context["config_file"] = str(config_file)
        super().__init__(message, context=context, **kwargs)


class Exception_Data_Not_Found(Exception_Custom):
    """Exception raised when required data is missing.

    Use this for missing files, database records, or API responses.

    Parameters
    ----------
    message : str
        Error message describing missing data.
    data_type : str | None
        Type of data that was not found.
    identifier : str | None
        Identifier used to search for data.
    source : str | None
        Data source (file path, database, API, etc.).
    **kwargs : Any
        Additional Exception_Custom parameters.
    """

    def __init__(
        self,
        message: str,
        data_type: str | None = None,
        identifier: str | None = None,
        source: str | None = None,
        **kwargs: Any,
    ) -> None:
        context = kwargs.pop("context", {})
        if data_type:
            context["data_type"] = data_type
        if identifier:
            context["identifier"] = identifier
        if source:
            context["source"] = source
        super().__init__(message, context=context, **kwargs)


class Exception_Calculation(Exception_Custom):
    """Exception raised for calculation or computation failures.

    Use this for mathematical errors, algorithm failures, or numerical issues.

    Parameters
    ----------
    message : str
        Calculation error message.
    operation : str | None
        Name of the calculation operation.
    inputs : dict[str, Any] | None
        Input values that caused the failure.
    **kwargs : Any
        Additional Exception_Custom parameters.
    """

    def __init__(
        self,
        message: str,
        operation: str | None = None,
        inputs: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        context = kwargs.pop("context", {})
        if operation:
            context["operation"] = operation
        if inputs:
            context["inputs"] = _mask_sensitive_data(inputs)
        super().__init__(message, context=context, **kwargs)


class Exception_Portfolio(Exception_Custom):
    """Exception raised for portfolio-related errors.

    Use this for portfolio validation, optimization, or rebalancing failures.

    Parameters
    ----------
    message : str
        Portfolio error message.
    portfolio_id : str | None
        Portfolio identifier.
    portfolio_name : str | None
        Portfolio name.
    operation : str | None
        Operation that failed.
    **kwargs : Any
        Additional Exception_Custom parameters.
    """

    def __init__(
        self,
        message: str,
        portfolio_id: str | None = None,
        portfolio_name: str | None = None,
        operation: str | None = None,
        **kwargs: Any,
    ) -> None:
        context = kwargs.pop("context", {})
        if portfolio_id:
            context["portfolio_id"] = portfolio_id
        if portfolio_name:
            context["portfolio_name"] = portfolio_name
        if operation:
            context["operation"] = operation
        super().__init__(message, context=context, **kwargs)


class Exception_Client(Exception_Custom):
    """Exception raised for client data errors.

    Use this for client validation, missing client data, or client operations.

    Parameters
    ----------
    message : str
        Client error message.
    client_id : str | None
        Client identifier.
    client_type : str | None
        Type of client (PRIMARY, PARTNER, etc.).
    **kwargs : Any
        Additional Exception_Custom parameters.
    """

    def __init__(
        self,
        message: str,
        client_id: str | None = None,
        client_type: str | None = None,
        **kwargs: Any,
    ) -> None:
        context = kwargs.pop("context", {})
        if client_id:
            context["client_id"] = client_id
        if client_type:
            context["client_type"] = client_type
        super().__init__(message, context=context, **kwargs)


class Exception_Database(Exception_Custom):
    """Exception raised for database operation failures.

    Use this for connection errors, query failures, or transaction issues.

    Parameters
    ----------
    message : str
        Database error message.
    database_name : str | None
        Name of the database.
    operation : str | None
        Database operation (SELECT, INSERT, UPDATE, DELETE).
    table_name : str | None
        Table involved in the operation.
    query : str | None
        Query that failed (will be truncated).
    **kwargs : Any
        Additional Exception_Custom parameters.
    """

    def __init__(
        self,
        message: str,
        database_name: str | None = None,
        operation: str | None = None,
        table_name: str | None = None,
        query: str | None = None,
        **kwargs: Any,
    ) -> None:
        context = kwargs.pop("context", {})
        if database_name:
            context["database_name"] = database_name
        if operation:
            context["operation"] = operation
        if table_name:
            context["table_name"] = table_name
        if query:
            context["query"] = _truncate_value(query, 200)
        super().__init__(message, context=context, **kwargs)


class Exception_API(Exception_Custom):
    """Exception raised for API communication failures.

    Use this for HTTP errors, API timeouts, or response parsing failures.

    Parameters
    ----------
    message : str
        API error message.
    endpoint : str | None
        API endpoint URL.
    method : str | None
        HTTP method (GET, POST, PUT, DELETE).
    status_code : int | None
        HTTP status code.
    response_body : str | None
        Response body (will be truncated).
    **kwargs : Any
        Additional Exception_Custom parameters.
    """

    def __init__(
        self,
        message: str,
        endpoint: str | None = None,
        method: str | None = None,
        status_code: int | None = None,
        response_body: str | None = None,
        **kwargs: Any,
    ) -> None:
        context = kwargs.pop("context", {})
        if endpoint:
            context["endpoint"] = endpoint
        if method:
            context["method"] = method
        if status_code:
            context["status_code"] = status_code
        if response_body:
            context["response_body"] = _truncate_value(response_body, 500)
        super().__init__(message, context=context, **kwargs)


class Exception_Authentication(Exception_Custom):
    """Exception raised for authentication failures.

    Use this for login failures, invalid credentials, or session issues.

    Parameters
    ----------
    message : str
        Authentication error message.
    user_id : str | None
        User identifier (will NOT include credentials).
    auth_method : str | None
        Authentication method used.
    **kwargs : Any
        Additional Exception_Custom parameters.
    """

    def __init__(
        self,
        message: str,
        user_id: str | None = None,
        auth_method: str | None = None,
        **kwargs: Any,
    ) -> None:
        context = kwargs.pop("context", {})
        if user_id:
            context["user_id"] = user_id
        if auth_method:
            context["auth_method"] = auth_method
        super().__init__(
            message,
            severity=Exception_Severity.WARNING,  # type: ignore[arg-type]
            context=context,
            **kwargs,
        )


class Exception_Authorization(Exception_Custom):
    """Exception raised for authorization/permission failures.

    Use this for access denied, insufficient permissions, or role violations.

    Parameters
    ----------
    message : str
        Authorization error message.
    user_id : str | None
        User identifier.
    required_permission : str | None
        Permission that was required.
    resource : str | None
        Resource that access was denied to.
    **kwargs : Any
        Additional Exception_Custom parameters.
    """

    def __init__(
        self,
        message: str,
        user_id: str | None = None,
        required_permission: str | None = None,
        resource: str | None = None,
        **kwargs: Any,
    ) -> None:
        context = kwargs.pop("context", {})
        if user_id:
            context["user_id"] = user_id
        if required_permission:
            context["required_permission"] = required_permission
        if resource:
            context["resource"] = resource
        super().__init__(
            message,
            severity=Exception_Severity.WARNING,  # type: ignore[arg-type]
            context=context,
            **kwargs,
        )


class Exception_File_Operation(Exception_Custom):
    """Exception raised for file I/O failures.

    Use this for file read/write errors, permission issues, or format problems.

    Parameters
    ----------
    message : str
        File operation error message.
    file_path : str | Path | None
        Path to the file.
    operation : str | None
        File operation (read, write, delete, etc.).
    **kwargs : Any
        Additional Exception_Custom parameters.
    """

    def __init__(
        self,
        message: str,
        file_path: str | Path | None = None,
        operation: str | None = None,
        **kwargs: Any,
    ) -> None:
        context = kwargs.pop("context", {})
        if file_path:
            context["file_path"] = str(file_path)
        if operation:
            context["operation"] = operation
        super().__init__(message, context=context, **kwargs)


class Exception_Timeout(Exception_Custom):
    """Exception raised when operations exceed time limits.

    Use this for operation timeouts, deadline exceeded, or slow responses.

    Parameters
    ----------
    message : str
        Timeout error message.
    operation : str | None
        Operation that timed out.
    timeout_seconds : float | None
        Timeout limit in seconds.
    elapsed_seconds : float | None
        Actual elapsed time.
    **kwargs : Any
        Additional Exception_Custom parameters.
    """

    def __init__(
        self,
        message: str,
        operation: str | None = None,
        timeout_seconds: float | None = None,
        elapsed_seconds: float | None = None,
        **kwargs: Any,
    ) -> None:
        context = kwargs.pop("context", {})
        if operation:
            context["operation"] = operation
        if timeout_seconds is not None:
            context["timeout_seconds"] = timeout_seconds
        if elapsed_seconds is not None:
            context["elapsed_seconds"] = elapsed_seconds
        super().__init__(message, context=context, **kwargs)


class Exception_Invalid_Input(Exception_Custom):
    """Exception raised when input is invalid or malformed.

    Use this for general input validation failures beyond type checking.

    Parameters
    ----------
    message : str
        Error message describing the invalid input.
    input_name : str | None
        Name of the invalid input parameter.
    expected_format : str | None
        Description of expected input format.
    actual_value : Any | None
        The actual value that was provided.
    **kwargs : Any
        Additional Exception_Custom parameters.

    Examples
    --------
    >>> raise Exception_Invalid_Input(
    ...     "Portfolio weights must sum to 1.0",
    ...     input_name="weights",
    ...     expected_format="dict[str, float] summing to 1.0",
    ...     actual_value={"AAPL": 0.5, "MSFT": 0.3},
    ... )
    """

    def __init__(
        self,
        message: str,
        input_name: str | None = None,
        expected_format: str | None = None,
        actual_value: Any = None,
        **kwargs: Any,
    ) -> None:
        context = kwargs.pop("context", {})
        if input_name:
            context["input_name"] = input_name
        if expected_format:
            context["expected_format"] = expected_format
        if actual_value is not None:
            context["actual_value"] = _truncate_value(actual_value)
        super().__init__(message, context=context, **kwargs)


class Exception_Not_Found(Exception_Custom):
    """Exception raised when a requested resource or entity is not found.

    General-purpose not-found exception for any type of missing resource.

    Parameters
    ----------
    message : str
        Error message describing what was not found.
    resource_type : str | None
        Type of resource that was not found.
    resource_id : str | None
        Identifier of the missing resource.
    search_criteria : dict[str, Any] | None
        Criteria used to search for the resource.
    **kwargs : Any
        Additional Exception_Custom parameters.

    Examples
    --------
    >>> raise Exception_Not_Found(
    ...     "Portfolio not found",
    ...     resource_type="Portfolio",
    ...     resource_id="PORT_001",
    ... )
    """

    def __init__(
        self,
        message: str,
        resource_type: str | None = None,
        resource_id: str | None = None,
        search_criteria: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        context = kwargs.pop("context", {})
        if resource_type:
            context["resource_type"] = resource_type
        if resource_id:
            context["resource_id"] = resource_id
        if search_criteria:
            context["search_criteria"] = _mask_sensitive_data(search_criteria)
        super().__init__(message, context=context, **kwargs)


class Exception_Security_Violation(Exception_Custom):
    """Exception raised for security policy violations.

    Use this for path traversal attempts, injection attacks, or policy breaches.

    Parameters
    ----------
    message : str
        Security violation description.
    violation_type : str | None
        Type of security violation (path_traversal, injection, policy_breach).
    resource : str | None
        Resource involved in the violation.
    user_id : str | None
        User who triggered the violation.
    **kwargs : Any
        Additional Exception_Custom parameters.

    Examples
    --------
    >>> raise Exception_Security_Violation(
    ...     "Path traversal attempt detected",
    ...     violation_type="path_traversal",
    ...     resource="../../../etc/passwd",
    ... )
    """

    def __init__(
        self,
        message: str,
        violation_type: str | None = None,
        resource: str | None = None,
        user_id: str | None = None,
        **kwargs: Any,
    ) -> None:
        context = kwargs.pop("context", {})
        if violation_type:
            context["violation_type"] = violation_type
        if resource:
            context["resource"] = resource
        if user_id:
            context["user_id"] = user_id
        super().__init__(
            message,
            severity=Exception_Severity.CRITICAL,  # type: ignore[arg-type]
            context=context,
            **kwargs,
        )


class Exception_Insufficient_Holdings(Exception_Custom):
    """Exception raised when portfolio has insufficient holdings for operation.

    Use this for sell/transfer operations that exceed available holdings.

    Parameters
    ----------
    message : str
        Error message describing the insufficiency.
    ticker : str | None
        Ticker symbol of the asset.
    required_quantity : float | None
        Quantity required for the operation.
    available_quantity : float | None
        Quantity currently available.
    operation : str | None
        Operation that was attempted.
    **kwargs : Any
        Additional Exception_Custom parameters.

    Examples
    --------
    >>> raise Exception_Insufficient_Holdings(
    ...     "Insufficient AAPL holdings for sell order",
    ...     ticker="AAPL",
    ...     required_quantity=100.0,
    ...     available_quantity=50.0,
    ...     operation="sell",
    ... )
    """

    def __init__(
        self,
        message: str,
        ticker: str | None = None,
        required_quantity: float | None = None,
        available_quantity: float | None = None,
        operation: str | None = None,
        **kwargs: Any,
    ) -> None:
        context = kwargs.pop("context", {})
        if ticker:
            context["ticker"] = ticker
        if required_quantity is not None:
            context["required_quantity"] = required_quantity
        if available_quantity is not None:
            context["available_quantity"] = available_quantity
        if operation:
            context["operation"] = operation
        super().__init__(message, context=context, **kwargs)


class Exception_Invalid_Transaction(Exception_Custom):
    """Exception raised when a transaction is invalid or cannot be executed.

    Use this for transaction validation failures or business rule violations.

    Parameters
    ----------
    message : str
        Error message describing why transaction is invalid.
    transaction_type : str | None
        Type of transaction (buy, sell, transfer, rebalance).
    transaction_id : str | None
        Unique identifier for the transaction.
    reason : str | None
        Specific reason for invalidation.
    **kwargs : Any
        Additional Exception_Custom parameters.

    Examples
    --------
    >>> raise Exception_Invalid_Transaction(
    ...     "Cannot execute sell order: market closed",
    ...     transaction_type="sell",
    ...     reason="market_closed",
    ... )
    """

    def __init__(
        self,
        message: str,
        transaction_type: str | None = None,
        transaction_id: str | None = None,
        reason: str | None = None,
        **kwargs: Any,
    ) -> None:
        context = kwargs.pop("context", {})
        if transaction_type:
            context["transaction_type"] = transaction_type
        if transaction_id:
            context["transaction_id"] = transaction_id
        if reason:
            context["reason"] = reason
        super().__init__(message, context=context, **kwargs)


# ==============================================================================
# Exception Aliases (for convenience and backward compatibility)
# ==============================================================================

# Alias for validation-related exceptions per coding standards
Exception_Validation = Exception_Validation_Input


# ==============================================================================
# Global Exception Handler Utilities
# ==============================================================================


_ORIGINAL_EXCEPTHOOK = sys.excepthook


def global_exception_handler(
    exc_type: type[BaseException],
    exc_value: BaseException,
    exc_traceback: Any,
) -> None:
    """Global exception handler using Exception_Custom.

    Parameters
    ----------
    exc_type : type[BaseException]
        Exception class.
    exc_value : BaseException
        Exception instance.
    exc_traceback : Any
        Traceback object.
    """
    # Wrap standard exceptions
    if not isinstance(exc_value, Exception_Custom):
        # Determine format from environment
        env_fmt = os.environ.get("QWIM_EXCEPTION_FORMAT", "RICH_TRACEBACK")
        try:
            fmt = Exception_Format[env_fmt]  # type: ignore[index]
        except KeyError:
            fmt = Exception_Format.RICH_TRACEBACK

        # Ensure we wrap standard exceptions properly
        if isinstance(exc_value, Exception):
            exc = Exception_Custom.from_exception(exc_value, exception_format=fmt)
        else:
            # For BaseException (SystemExit, etc), just use standard hook
            _ORIGINAL_EXCEPTHOOK(exc_type, exc_value, exc_traceback)
            return
    else:
        exc = exc_value

    # Log to audit/error log using logger_custom
    try:
        from src.utils.custom_exceptions_errors_loggers.logger_custom import get_logger

        logger = get_logger("global_handler")
        # Log as CRITICAL since it's unhandled
        logger.opt(exception=(exc_type, exc_value, exc_traceback)).critical(
            f"Uncaught exception: {exc_value}",
        )
    except Exception:
        # Fallback if logging fails
        pass

    # Print using rich format if supported, otherwise standard
    if exc.exception_format in {Exception_Format.RICH_TRACEBACK, Exception_Format.FULL}:
        exc.print_rich()
    else:
        print(str(exc), file=sys.stderr)


def install_exception_handler() -> None:
    """Install global exception handler."""
    sys.excepthook = global_exception_handler


def restore_exception_handler() -> None:
    """Restore original exception handler."""
    sys.excepthook = _ORIGINAL_EXCEPTHOOK


# ==============================================================================
# Context Managers & Decorators
# ==============================================================================

T = TypeVar("T")
P = ParamSpec("P")


@contextmanager
def capture_exception(
    context: dict[str, Any] | None = None,
    reraise: bool = True,
    log_severity: Exception_Severity | None = None,
) -> Generator[None, None, None]:
    """Context manager to capture and wrap exceptions.

    Parameters
    ----------
    context : dict[str, Any] | None
        Additional context to add to exceptions.
    reraise : bool
        Whether to reraise the exception (wrapped).
    log_severity : Exception_Severity | None
        Override severity level.

    Yields
    ------
    None
    """
    try:
        yield
    except Exception as e:
        if isinstance(e, Exception_Custom):
            if context:
                e._user_context.update(context)
            if log_severity:
                e._severity = log_severity
            if reraise:
                raise
        else:
            wrapped = Exception_Custom.from_exception(
                e,
                context=context,
                severity=log_severity or Exception_Severity.ERROR,
            )
            if reraise:
                raise wrapped from e


def handle_exceptions(
    context: dict[str, Any] | None = None,
    severity: Exception_Severity | None = None,
) -> Callable[[Callable[P, T]], Callable[P, T]]:
    """Decorator to handle exceptions in functions.

    Parameters
    ----------
    context : dict[str, Any] | None
        Static context data.
    severity : Exception_Severity | None
        Severity level for captured exceptions.

    Returns
    -------
    Callable
        Decorated function.
    """

    def decorator(func: Callable[P, T]) -> Callable[P, T]:
        from functools import wraps

        @wraps(func)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
            try:
                return func(*args, **kwargs)
            except Exception as e:
                ctx = context or {}
                # Ensure context is a copy
                ctx = ctx.copy()

                if isinstance(e, Exception_Custom):
                    e._user_context.update(ctx)
                    if severity:
                        e._severity = severity
                    raise

                wrapped = Exception_Custom.from_exception(
                    e,
                    context=ctx,
                    severity=severity or Exception_Severity.ERROR,
                )
                raise wrapped from e

        return wrapper

    return decorator


# ==============================================================================
# Module Exports
# ==============================================================================

__all__ = [
    "MAX_TRACEBACK_FRAMES",
    "MAX_VARIABLE_LENGTH",
    # Constants
    "SENSITIVE_FIELDS",
    "Exception_API",
    "Exception_Authentication",
    "Exception_Authorization",
    "Exception_Calculation",
    "Exception_Client",
    "Exception_Configuration",
    "Exception_Context",
    # Type aliases
    "Exception_Context_Dict",
    # Main exception class
    "Exception_Custom",
    "Exception_Data_Not_Found",
    "Exception_Database",
    "Exception_File_Operation",
    # Enums
    "Exception_Format",
    # Data classes
    "Exception_Frame",
    "Exception_Frame_List",
    "Exception_Insufficient_Holdings",
    # Additional domain exceptions (per coding-instructions-python.md)
    "Exception_Invalid_Input",
    "Exception_Invalid_Transaction",
    "Exception_Not_Found",
    "Exception_Portfolio",
    "Exception_Security_Violation",
    "Exception_Severity",
    "Exception_Timeout",
    # Aliases (convenience and backward compatibility)
    "Exception_Validation",
    # Domain-specific exceptions
    "Exception_Validation_Input",
    "Sensitive_Field_Set",
    # Utilities
    "capture_exception",
    "handle_exceptions",
    # Global Handler
    "install_exception_handler",
    "restore_exception_handler",
]
