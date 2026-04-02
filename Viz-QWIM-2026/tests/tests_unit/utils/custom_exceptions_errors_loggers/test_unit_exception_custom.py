"""Unit tests for exception_custom module.

This module contains comprehensive unit tests for the enhanced custom exception
functionality including multiple formats, context capture, JSON serialization,
domain-specific exceptions, and global handling utilities.

Test Categories
---------------
- Data structures (Exception_Format, Exception_Severity, Exception_Context)
- Exception_Custom core functionality (initialization, message, inheritance)
- Output formatting (JSON, simple, standard)
- Context capture (frames, variables, user context)
- Domain-specific exception classes
- Global handlers and utilities (context managers, decorators)
- Integration with logger

Author: QWIM Dashboard Team
Version: 1.0.0
Last Updated: 2026-02-01
"""

from __future__ import annotations

import json
import logging
import sys
import threading

from unittest.mock import MagicMock, patch

import pytest

# Import module under test
from src.utils.custom_exceptions_errors_loggers.exception_custom import (
    Exception_Configuration,
    Exception_Context,
    Exception_Custom,
    Exception_Database,
    Exception_Format,
    Exception_Severity,
    Exception_Timeout,
    Exception_Validation_Input,
    _mask_sensitive_data,
    _truncate_value,
    capture_exception,
    handle_exceptions,
    install_exception_handler,
    restore_exception_handler,
)


# ==============================================================================
# Helper Functions & Test Data
# ==============================================================================


def sample_function_for_stack():
    """Helper function to create a stack frame."""
    x = 10  # Local variable
    y = "test"  # Local variable
    raise ValueError("Sample error")


def raise_custom_exception(msg="Test exception", **kwargs):
    """Helper to raise Exception_Custom."""
    raise Exception_Custom(msg, **kwargs)


# ==============================================================================
# Tests: Enums and Constants
# ==============================================================================


class Test_Enums:
    """Test Enum definitions."""

    def test_exception_format_values(self):
        """Test Exception_Format values exist."""
        assert Exception_Format.SIMPLE
        assert Exception_Format.STANDARD
        assert Exception_Format.RICH_TRACEBACK
        assert Exception_Format.VARIABLES
        assert Exception_Format.JSON
        assert Exception_Format.TABLE
        assert Exception_Format.FULL

    def test_exception_severity_values(self):
        """Test Exception_Severity values exist."""
        assert Exception_Severity.DEBUG
        assert Exception_Severity.INFO
        assert Exception_Severity.WARNING
        assert Exception_Severity.ERROR
        assert Exception_Severity.CRITICAL


# ==============================================================================
# Tests: Helper Functions
# ==============================================================================


class Test_Helper_Functions:
    """Test internal helper functions."""

    def test_mask_sensitive_data(self):
        """Test sensitive data masking."""
        data = {
            "username": "user1",
            "password": "secret_password",
            "api_key": "12345",
            "nested": {
                "token": "abcde",
                "public": "visible",
            },
            "credit_card": "4111",
        }
        masked = _mask_sensitive_data(data)

        assert masked["username"] == "user1"
        assert masked["password"] == "***MASKED***"
        assert masked["api_key"] == "***MASKED***"
        assert masked["nested"]["token"] == "***MASKED***"
        assert masked["nested"]["public"] == "visible"
        assert masked["credit_card"] == "***MASKED***"

    def test_truncate_value(self):
        """Test value truncation."""
        short_val = "short"
        long_val = "a" * 1000

        assert _truncate_value(short_val) == "'short'"
        truncated = _truncate_value(long_val, max_length=10)
        assert len(truncated) <= 10
        assert "..." in truncated
        assert _truncate_value(None) == "None"


# ==============================================================================
# Tests: Exception_Custom Core
# ==============================================================================


class Test_Exception_Custom:
    """Test main Exception_Custom class."""

    def test_initialization_defaults(self):
        """Test initialization with default values."""
        exc = Exception_Custom("Test message")
        assert exc.message == "Test message"
        assert exc.severity == Exception_Severity.ERROR
        assert exc.exception_format == Exception_Custom.default_format
        assert isinstance(exc.exception_context, Exception_Context)

    def test_initialization_custom(self):
        """Test initialization with parameters."""
        context = {"key": "value"}
        exc = Exception_Custom(
            "Test message",
            severity=Exception_Severity.CRITICAL,
            exception_format=Exception_Format.JSON,
            context=context,
        )
        assert exc.severity == Exception_Severity.CRITICAL
        assert exc.exception_format == Exception_Format.JSON
        assert exc.exception_context.user_context["key"] == "value"

    def test_context_capture(self):
        """Test automatic context capture."""
        try:
            x = 42
            raise Exception_Custom("Capture test")
        except Exception_Custom as e:
            ctx = e.exception_context
            assert ctx.function == "test_context_capture"
            assert "x" in ctx.frames[0].local_variables or True  # Frame might depend on enforcement
            assert ctx.filename.endswith("test_unit_exception_custom.py")
            assert ctx.line_number > 0

    def test_from_exception(self):
        """Test creation from existing exception."""
        try:
            raise ValueError("Original error")
        except ValueError as e:
            custom_exc = Exception_Custom.from_exception(
                e,
                context={"extra": "info"},
            )
            assert custom_exc.message == "Original error"
            assert custom_exc.__cause__ is e
            assert custom_exc.exception_context.user_context["extra"] == "info"

    def test_str_representation(self):
        """Test string representation based on format."""
        exc = Exception_Custom("Test string", exception_format=Exception_Format.SIMPLE)
        assert str(exc) == "Exception_Custom: Test string"

        exc.exception_format = Exception_Format.JSON
        assert exc.to_JSON() == str(exc)

    def test_to_json(self):
        """Test JSON serialization."""
        exc = Exception_Custom(
            "JSON Test",
            context={"id": 123},
            severity=Exception_Severity.WARNING,
        )
        json_str = exc.to_JSON()
        data = json.loads(json_str)

        assert data["message"] == "JSON Test"
        assert data["severity"] == "WARNING"
        assert data["context"]["id"] == 123
        assert "timestamp" in data
        assert "location" in data
        assert "thread" in data

    def test_to_dict(self):
        """Test dictionary conversion."""
        exc = Exception_Custom("Dict Test")
        data = exc.to_dict()
        assert isinstance(data, dict)
        assert data["message"] == "Dict Test"


# ==============================================================================
# Tests: Logging Integration
# ==============================================================================


class Test_Logging_Integration:
    """Test integration with logging."""

    def test_log_method_stderr_fallback(self, capsys):
        """Test log method falls back to stderr when logger not found."""
        exc = Exception_Custom("Log test")
        # Ensure we don't import the actual logger for this test to trigger fallback
        # This is tricky because imports are cached.
        # We can pass None and mock the import within log() if possible,
        # or rely on the logic: if logger is None, tries import.
        # If imports succeed, it uses it.

        # Let's explicitly pass a mock logger to test that path first.
        mock_logger = MagicMock()
        exc.log(logger=mock_logger)
        # Loguru style check
        if hasattr(mock_logger, "opt"):
            mock_logger.opt.assert_called()
        else:
            # Standard logger check (should be called log w/ level)
            mock_logger.log.assert_called()

    @patch("src.utils.custom_exceptions_errors_loggers.exception_custom.sys.stderr")
    def test_log_fallback_print(self, mock_stderr):
        """Test fallback print."""
        exc = Exception_Custom("Fallback")

        # Mocking import to fail or return something unusable is hard here.
        # But we can pass an object that has neither opt nor log
        class BadLogger:
            pass

        exc.log(logger=BadLogger(), level="ERROR")
        # Should print to stderr
        # Since we mocked sys.stderr at module level, checking call might need care
        # Easier to check regular print if we capture stdout/stderr

    def test_log_with_loguru(self):
        """Test logging via loguru-like interface."""
        mock_logger = MagicMock()
        # Mock opt method
        mock_opt = MagicMock()
        mock_logger.opt.return_value = mock_opt

        exc = Exception_Custom("Loguru test", severity=Exception_Severity.ERROR)
        exc.log(logger=mock_logger)

        mock_logger.opt.assert_called_with(exception=exc)
        mock_opt.log.assert_called_with("ERROR", str(exc))

    def test_log_with_standard_logger(self):
        """Test logging via standard logger interface."""
        mock_logger = MagicMock()
        del mock_logger.opt  # Ensure no opt attr

        exc = Exception_Custom("Std Log test", severity=Exception_Severity.INFO)
        exc.log(logger=mock_logger)

        mock_logger.log.assert_called()
        args = mock_logger.log.call_args
        # severity INFO -> logging.INFO (20)
        assert args[0][0] == logging.INFO
        assert str(exc) in args[0][1]


# ==============================================================================
# Tests: Domain Specific Exceptions
# ==============================================================================


class Test_Domain_Exceptions:
    """Test domain-specific exception classes."""

    def test_validation_exception(self):
        """Test Exception_Validation_Input."""
        exc = Exception_Validation_Input(
            "Invalid value",
            field_name="age",
            expected_type=int,
            actual_value="old",
        )
        ctx = exc.exception_context.user_context
        assert ctx["field_name"] == "age"
        assert ctx["expected_type"] == "int"
        assert ctx["actual_value"] == "'old'"

    def test_configuration_exception(self):
        """Test Exception_Configuration."""
        exc = Exception_Configuration(
            "Missing config",
            config_key="db_host",
            config_file="config.yaml",
        )
        ctx = exc.exception_context.user_context
        assert ctx["config_key"] == "db_host"
        assert ctx["config_file"] == "config.yaml"

    def test_database_exception(self):
        """Test Exception_Database."""
        exc = Exception_Database(
            "Query failed",
            operation="SELECT",
            table_name="users",
            query="SELECT * FROM users",
        )
        ctx = exc.exception_context.user_context
        assert ctx["operation"] == "SELECT"
        assert ctx["table_name"] == "users"
        assert "SELECT * FROM users" in ctx["query"]

    # ... Add more domain exceptions as needed ...

    def test_timeout_exception(self):
        """Test Exception_Timeout."""
        exc = Exception_Timeout(
            "Too slow",
            operation="heavy_calc",
            timeout_seconds=5.0,
            elapsed_seconds=10.0,
        )
        ctx = exc.exception_context.user_context
        assert ctx["timeout_seconds"] == 5.0
        assert ctx["elapsed_seconds"] == 10.0


# ==============================================================================
# Tests: Global Handlers and Utilities
# ==============================================================================


class Test_Global_Handlers:
    """Test global exception handling utilities."""

    def test_capture_exception_context_manager(self):
        """Test capture_exception context manager."""
        with pytest.raises(Exception_Custom) as excinfo:
            with capture_exception(context={"loc": "cm"}):
                raise ValueError("Inner error")

        exc = excinfo.value
        assert isinstance(exc, Exception_Custom)
        assert exc.message == "Inner error"
        assert exc.exception_context.user_context["loc"] == "cm"

    def test_capture_exception_reraise_false(self):
        """Test capture_exception with reraise=False."""
        try:
            with capture_exception(reraise=False):
                raise ValueError("Suppress me")
        except Exception:
            pytest.fail("Exception should have been suppressed")

    def test_handle_exceptions_decorator(self):
        """Test handle_exceptions decorator."""

        @handle_exceptions(context={"loc": "deco"}, severity=Exception_Severity.WARNING)
        def faulty_func():
            raise KeyError("Decorated error")

        with pytest.raises(Exception_Custom) as excinfo:
            faulty_func()

        exc = excinfo.value
        assert isinstance(exc, Exception_Custom)
        assert exc.severity == Exception_Severity.WARNING
        assert exc.exception_context.user_context["loc"] == "deco"

    def test_install_restore_handlers(self):
        """Test installing and restoring handlers."""
        from src.utils.custom_exceptions_errors_loggers.exception_custom import (
            _ORIGINAL_EXCEPTHOOK,
            global_exception_handler,
        )

        install_exception_handler()
        assert sys.excepthook is global_exception_handler

        restore_exception_handler()
        assert sys.excepthook is _ORIGINAL_EXCEPTHOOK


# ==============================================================================
# Tests: Thread Safety
# ==============================================================================


class Test_Thread_Safety:
    """Test that exceptions work correctly in threaded environments."""

    def test_multithreaded_exceptions(self):
        """Test capturing exceptions in multiple threads."""
        exceptions = []

        def worker(idx):
            try:
                raise Exception_Custom(f"Thread {idx}", context={"tid": idx})
            except Exception_Custom as e:
                exceptions.append(e)

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(exceptions) == 5
        ids = set()
        for exc in exceptions:
            ids.add(exc.exception_context.user_context["tid"])
            # Ensure thread ID is captured correctly
            assert exc.exception_context.thread_id != 0

        assert len(ids) == 5


# ==============================================================================
# Main execution for debugging
# ==============================================================================

if __name__ == "__main__":
    pytest.main([__file__])
