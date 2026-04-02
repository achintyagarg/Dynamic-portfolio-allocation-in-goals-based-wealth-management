"""Unit tests for logger_custom module.

This module contains comprehensive unit tests for the custom logging functionality
including configuration validation, audit logging, performance tracking, and
log formatting.

Test Categories
---------------
- Configuration validation (Pydantic models)
- Logger setup and initialization
- Audit logging methods
- Performance tracking
- JSON serialization and formatting
- Filter functions
- Decorator functionality
- Error handling and edge cases

Author: QWIM Dashboard Team
Version: 1.0.0
Last Updated: 2026-02-01
"""

from __future__ import annotations

import json
import time

from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

from loguru import logger
from pydantic import ValidationError

# Import module under test
from src.utils.custom_exceptions_errors_loggers.logger_custom import (
    LOG_FILE_APPLICATION,
    LOG_FILE_AUDIT,
    LOG_FILE_DEBUG,
    LOG_FILE_ERROR,
    LOG_FILE_PERFORMANCE,
    Audit_Logger,
    Config_Logging,
    Performance_Timer,
    filter_audit_events,
    filter_error_level,
    filter_performance_events,
    format_record_as_JSON,
    format_sink_human_readable,
    format_sink_JSON,
    get_logger,
    log_function_call,
    log_performance,
    serialize_for_JSON,
    setup_logging,
)


# ==============================================================================
# Fixtures
# ==============================================================================


@pytest.fixture()
def temp_log_dir(tmp_path: Path) -> Path:
    """Create a temporary directory for log files.

    Parameters
    ----------
    tmp_path : Path
        Pytest temporary directory fixture.

    Returns
    -------
    Path
        Path to temporary log directory.
    """
    log_dir = tmp_path / "logs"
    log_dir.mkdir(exist_ok=True)
    return log_dir


@pytest.fixture(autouse=True)
def cleanup_logger() -> None:
    """Clean up logger handlers after each test.

    This fixture automatically runs after each test to remove all
    logger handlers and reset the logger state.

    Returns
    -------
    None
    """
    yield
    # Remove all handlers
    logger.remove()


@pytest.fixture()
def sample_log_record() -> dict[str, Any]:
    """Create a sample log record for testing.

    Returns
    -------
    dict[str, Any]
        Sample log record with typical structure.
    """
    return {
        "time": datetime.now(UTC),
        "level": type("Level", (), {"name": "INFO", "no": 20})(),
        "name": "test_module",
        "module": "test_module",
        "function": "test_function",
        "line": 42,
        "message": "Test message",
        "exception": None,
        "extra": {},
        "thread": type("Thread", (), {"id": 12345, "name": "MainThread"})(),
        "process": type("Process", (), {"id": 99999, "name": "MainProcess"})(),
    }


@pytest.fixture()
def sample_audit_logger(temp_log_dir: Path) -> Audit_Logger:
    """Create a sample audit logger for testing.

    Parameters
    ----------
    temp_log_dir : Path
        Temporary log directory.

    Returns
    -------
    Audit_Logger
        Configured audit logger instance.
    """
    setup_logging(log_dir=temp_log_dir, enable_console=False)
    return Audit_Logger("test_audit")


# ==============================================================================
# Test Configuration Validation
# ==============================================================================


@pytest.mark.unit()
class Test_Config_Logging_Validation:
    """Test suite for Config_Logging Pydantic model validation."""

    def test_create_config_with_valid_defaults(self) -> None:
        """Test creating configuration with valid default values."""
        config = Config_Logging()

        assert config.log_level == "INFO"
        assert config.environment == "development"
        assert config.log_dir == Path("logs")
        assert config.enable_console is True
        assert config.enable_JSON is True

    @pytest.mark.parametrize(
        "log_level",
        ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
    )
    def test_create_config_with_valid_log_levels(self, log_level: str) -> None:
        """Test creating configuration with all valid log levels.

        Parameters
        ----------
        log_level : str
            Log level to test.
        """
        config = Config_Logging(log_level=log_level)
        assert config.log_level == log_level

    def test_create_config_with_invalid_log_level(self) -> None:
        """Test that invalid log levels raise ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            Config_Logging(log_level="INVALID")

        error = exc_info.value
        assert "log_level" in str(error)

    @pytest.mark.parametrize(
        "environment",
        ["development", "staging", "production"],
    )
    def test_create_config_with_valid_environments(self, environment: str) -> None:
        """Test creating configuration with all valid environments.

        Parameters
        ----------
        environment : str
            Environment to test.
        """
        config = Config_Logging(environment=environment)
        assert config.environment == environment

    def test_create_config_with_invalid_environment(self) -> None:
        """Test that invalid environments raise ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            Config_Logging(environment="invalid_env")

        error = exc_info.value
        assert "environment" in str(error)

    def test_create_config_with_custom_log_dir(self, tmp_path: Path) -> None:
        """Test creating configuration with custom log directory.

        Parameters
        ----------
        tmp_path : Path
            Temporary directory path.
        """
        custom_dir = tmp_path / "custom_logs"
        config = Config_Logging(log_dir=custom_dir)

        assert config.log_dir == custom_dir

    def test_create_config_with_all_custom_values(self, tmp_path: Path) -> None:
        """Test creating configuration with all custom values.

        Parameters
        ----------
        tmp_path : Path
            Temporary directory path.
        """
        config = Config_Logging(
            log_level="DEBUG",
            environment="production",
            log_dir=tmp_path / "logs",
            enable_console=False,
            enable_JSON=True,
            rotation_size="50 MB",
        )

        assert config.log_level == "DEBUG"
        assert config.environment == "production"
        assert config.log_dir == tmp_path / "logs"
        assert config.enable_console is False
        assert config.enable_JSON is True
        assert config.rotation_size == "50 MB"


# ==============================================================================
# Test Serialization Functions
# ==============================================================================


@pytest.mark.unit()
class Test_Serialization_Functions:
    """Test suite for JSON serialization utility functions."""

    def test_serialize_datetime(self) -> None:
        """Test serializing datetime objects."""
        dt = datetime(2024, 1, 15, 10, 30, 45)
        result = serialize_for_JSON(dt)

        assert isinstance(result, str)
        assert "2024-01-15" in result

    def test_serialize_path(self, tmp_path: Path) -> None:
        """Test serializing Path objects.

        Parameters
        ----------
        tmp_path : Path
            Temporary path.
        """
        result = serialize_for_JSON(tmp_path)

        assert isinstance(result, str)
        assert "tmp" in result.lower() or "temp" in result.lower()

    def test_serialize_bytes(self) -> None:
        """Test serializing bytes objects."""
        data = b"test data"
        result = serialize_for_JSON(data)

        assert isinstance(result, str)
        assert result == "test data"

    def test_serialize_object_with_dict(self) -> None:
        """Test serializing objects with __dict__ attribute."""

        class Test_Obj:
            def __init__(self) -> None:
                self.name = "test"
                self.value = 42

        obj = Test_Obj()
        result = serialize_for_JSON(obj)

        assert isinstance(result, dict)
        assert result["name"] == "test"
        assert result["value"] == 42

    def test_serialize_primitive_types(self) -> None:
        """Test serializing primitive types (fallback to str)."""
        result_int = serialize_for_JSON(42)
        result_float = serialize_for_JSON(3.14)
        result_str = serialize_for_JSON("hello")

        assert result_int == "42"
        assert result_float == "3.14"
        assert result_str == "hello"


@pytest.mark.unit()
class Test_Format_Functions:
    """Test suite for log formatting functions."""

    def test_format_record_as_json_basic(self, sample_log_record: dict) -> None:
        """Test basic JSON formatting of log records.

        Parameters
        ----------
        sample_log_record : dict
            Sample log record fixture.
        """
        result = format_record_as_JSON(sample_log_record)
        parsed = json.loads(result)

        assert "timestamp" in parsed
        assert parsed["level"] == "INFO"
        assert parsed["module"] == "test_module"
        assert parsed["function"] == "test_function"
        assert parsed["line"] == 42
        assert parsed["message"] == "Test message"

    def test_format_record_as_json_with_exception(
        self,
        sample_log_record: dict,
    ) -> None:
        """Test JSON formatting with exception information.

        Parameters
        ----------
        sample_log_record : dict
            Sample log record fixture.
        """
        # Create exception info
        exc_type = ValueError
        exc_value = ValueError("Test error")

        sample_log_record["exception"] = type(
            "Exception",
            (),
            {
                "type": exc_type,
                "value": exc_value,
                "traceback": "Traceback info...",
            },
        )()

        result = format_record_as_JSON(sample_log_record)
        parsed = json.loads(result)

        assert "exception" in parsed
        assert parsed["exception"]["type"] == "ValueError"
        assert "Test error" in parsed["exception"]["value"]

    def test_format_record_as_json_with_extra_fields(
        self,
        sample_log_record: dict,
    ) -> None:
        """Test JSON formatting with extra fields.

        Parameters
        ----------
        sample_log_record : dict
            Sample log record fixture.
        """
        sample_log_record["extra"] = {
            "user_id": "analyst_001",
            "session_id": "sess_12345",
            "event_type": "CALCULATION",
        }

        result = format_record_as_JSON(sample_log_record)
        parsed = json.loads(result)

        assert parsed["user_id"] == "analyst_001"
        assert parsed["session_id"] == "sess_12345"
        assert parsed["event_type"] == "CALCULATION"

    def test_format_record_debug_level_includes_thread_info(
        self,
        sample_log_record: dict,
    ) -> None:
        """Test that DEBUG level records include thread/process info.

        Parameters
        ----------
        sample_log_record : dict
            Sample log record fixture.
        """
        # Set to DEBUG level
        sample_log_record["level"] = type("Level", (), {"name": "DEBUG", "no": 10})()

        result = format_record_as_JSON(sample_log_record)
        parsed = json.loads(result)

        assert "thread_id" in parsed
        assert "thread_name" in parsed
        assert "process_id" in parsed
        assert "process_name" in parsed

    def test_format_sink_json(self, sample_log_record: dict) -> None:
        """Test sink JSON formatting adds newline.

        Parameters
        ----------
        sample_log_record : dict
            Sample log record fixture.
        """
        result = format_sink_JSON(sample_log_record)

        assert result.endswith("\n")
        # Remove newline and verify it's valid JSON
        json.loads(result.strip())

    def test_format_sink_human_readable(self, sample_log_record: dict) -> None:
        """Test human-readable formatting.

        Parameters
        ----------
        sample_log_record : dict
            Sample log record fixture.
        """
        result = format_sink_human_readable(sample_log_record)

        assert "INFO" in result
        assert "test_module" in result
        assert "test_function" in result
        assert "42" in result
        assert "Test message" in result
        assert result.endswith("\n")


# ==============================================================================
# Test Filter Functions
# ==============================================================================


@pytest.mark.unit()
class Test_Filter_Functions:
    """Test suite for log filter functions."""

    def test_filter_audit_events_with_event_type(
        self,
        sample_log_record: dict,
    ) -> None:
        """Test audit event filter accepts records with event_type.

        Parameters
        ----------
        sample_log_record : dict
            Sample log record fixture.
        """
        sample_log_record["extra"]["event_type"] = "CALCULATION"

        assert filter_audit_events(sample_log_record) is True

    def test_filter_audit_events_without_event_type(
        self,
        sample_log_record: dict,
    ) -> None:
        """Test audit event filter rejects records without event_type.

        Parameters
        ----------
        sample_log_record : dict
            Sample log record fixture.
        """
        assert filter_audit_events(sample_log_record) is False

    def test_filter_performance_events_with_execution_time(
        self,
        sample_log_record: dict,
    ) -> None:
        """Test performance filter accepts records with execution_time_ms.

        Parameters
        ----------
        sample_log_record : dict
            Sample log record fixture.
        """
        sample_log_record["extra"]["execution_time_ms"] = 123.45

        assert filter_performance_events(sample_log_record) is True

    def test_filter_performance_events_without_execution_time(
        self,
        sample_log_record: dict,
    ) -> None:
        """Test performance filter rejects records without execution_time_ms.

        Parameters
        ----------
        sample_log_record : dict
            Sample log record fixture.
        """
        assert filter_performance_events(sample_log_record) is False

    @pytest.mark.parametrize(
        ("level_no", "expected"),
        [
            (10, False),  # DEBUG
            (20, False),  # INFO
            (30, False),  # WARNING
            (40, True),  # ERROR
            (50, True),  # CRITICAL
        ],
    )
    def test_filter_error_level(
        self,
        sample_log_record: dict,
        level_no: int,
        expected: bool,
    ) -> None:
        """Test error level filter for different log levels.

        Parameters
        ----------
        sample_log_record : dict
            Sample log record fixture.
        level_no : int
            Log level number to test.
        expected : bool
            Expected filter result.
        """
        sample_log_record["level"] = type("Level", (), {"no": level_no})()

        assert filter_error_level(sample_log_record) is expected


# ==============================================================================
# Test Logger Setup and Get Logger
# ==============================================================================


@pytest.mark.unit()
class Test_Logger_Setup:
    """Test suite for logger setup and configuration."""

    def test_setup_logging_creates_log_directory(self, temp_log_dir: Path) -> None:
        """Test that setup_logging creates the log directory.

        Parameters
        ----------
        temp_log_dir : Path
            Temporary log directory fixture.
        """
        log_dir = temp_log_dir / "test_logs"
        setup_logging(log_dir=log_dir, enable_console=False)

        assert log_dir.exists()
        assert log_dir.is_dir()

    def test_setup_logging_creates_archive_directory(
        self,
        temp_log_dir: Path,
    ) -> None:
        """Test that setup_logging creates archive subdirectory.

        Parameters
        ----------
        temp_log_dir : Path
            Temporary log directory fixture.
        """
        setup_logging(log_dir=temp_log_dir, enable_console=False)
        archive_dir = temp_log_dir / "archive"

        assert archive_dir.exists()
        assert archive_dir.is_dir()

    @pytest.mark.parametrize(
        "log_level",
        ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
    )
    def test_setup_logging_with_different_levels(
        self,
        temp_log_dir: Path,
        log_level: str,
    ) -> None:
        """Test setup_logging with different log levels.

        Parameters
        ----------
        temp_log_dir : Path
            Temporary log directory fixture.
        log_level : str
            Log level to test.
        """
        setup_logging(
            log_level=log_level,
            log_dir=temp_log_dir,
            enable_console=False,
        )

        # Log a test message
        test_logger = get_logger(__name__)
        test_logger.info("Test message")

        # Verify application log file was created
        app_log = temp_log_dir / LOG_FILE_APPLICATION
        assert app_log.exists()

    def test_setup_logging_development_environment(
        self,
        temp_log_dir: Path,
    ) -> None:
        """Test setup_logging in development environment.

        Parameters
        ----------
        temp_log_dir : Path
            Temporary log directory fixture.
        """
        setup_logging(
            environment="development",
            log_dir=temp_log_dir,
            enable_console=False,
        )

        test_logger = get_logger(__name__)
        test_logger.debug("Debug message")

        # Development should create debug log
        debug_log = temp_log_dir / LOG_FILE_DEBUG
        assert debug_log.exists()

    def test_setup_logging_production_environment(
        self,
        temp_log_dir: Path,
    ) -> None:
        """Test setup_logging in production environment.

        Parameters
        ----------
        temp_log_dir : Path
            Temporary log directory fixture.
        """
        setup_logging(
            environment="production",
            log_dir=temp_log_dir,
            enable_console=False,
        )

        test_logger = get_logger(__name__)
        test_logger.info("Info message")

        # Production should not create debug log
        debug_log = temp_log_dir / LOG_FILE_DEBUG
        assert not debug_log.exists()

    def test_get_logger_auto_configures(self) -> None:
        """Test that get_logger auto-configures if not set up."""
        # This should trigger auto-configuration
        test_logger = get_logger("test_module")

        assert test_logger is not None

    def test_get_logger_returns_bound_logger(self, temp_log_dir: Path) -> None:
        """Test that get_logger returns a bound logger with name.

        Parameters
        ----------
        temp_log_dir : Path
            Temporary log directory fixture.
        """
        setup_logging(log_dir=temp_log_dir, enable_console=False)
        test_logger = get_logger("test_module")

        # Log a message
        test_logger.info("Test message")

        # Verify log file was created
        app_log = temp_log_dir / LOG_FILE_APPLICATION
        assert app_log.exists()


# ==============================================================================
# Test Audit Logger
# ==============================================================================


@pytest.mark.unit()
class Test_Audit_Logger:
    """Test suite for Audit_Logger class."""

    def test_create_audit_logger(self, sample_audit_logger: Audit_Logger) -> None:
        """Test creating an audit logger instance.

        Parameters
        ----------
        sample_audit_logger : Audit_Logger
            Audit logger fixture.
        """
        assert sample_audit_logger.logger_name == "test_audit"
        assert sample_audit_logger._logger is not None

    def test_log_calculation(
        self,
        sample_audit_logger: Audit_Logger,
        temp_log_dir: Path,
    ) -> None:
        """Test logging a calculation event.

        Parameters
        ----------
        sample_audit_logger : Audit_Logger
            Audit logger fixture.
        temp_log_dir : Path
            Temporary log directory.
        """
        sample_audit_logger.log_calculation(
            operation="Portfolio_Return",
            inputs={"weights": [0.6, 0.4], "returns": [0.05, 0.03]},
            result=0.042,
            execution_time_ms=12.5,
            user_id="analyst_001",
            session_id="sess_123",
        )

        # Wait for log file to be written
        time.sleep(0.1)

        # Verify audit log was created
        audit_log = temp_log_dir / LOG_FILE_AUDIT
        assert audit_log.exists()

        # Read and verify content
        content = audit_log.read_text(encoding="utf-8")
        assert "Portfolio_Return" in content
        assert "analyst_001" in content

    def test_log_calculation_without_optional_params(
        self,
        sample_audit_logger: Audit_Logger,
    ) -> None:
        """Test logging calculation without optional parameters.

        Parameters
        ----------
        sample_audit_logger : Audit_Logger
            Audit logger fixture.
        """
        # Should not raise any exceptions
        sample_audit_logger.log_calculation(
            operation="Simple_Calculation",
            inputs={"a": 1, "b": 2},
            result=3,
            execution_time_ms=5.0,
        )

    def test_log_data_access(
        self,
        sample_audit_logger: Audit_Logger,
        temp_log_dir: Path,
    ) -> None:
        """Test logging data access event.

        Parameters
        ----------
        sample_audit_logger : Audit_Logger
            Audit logger fixture.
        temp_log_dir : Path
            Temporary log directory.
        """
        sample_audit_logger.log_data_access(
            source="inputs/raw/data_ETFs.csv",
            operation="READ",
            record_count=252,
            user_id="analyst_001",
            session_id="sess_123",
            details={"file_size_mb": 2.5},
        )

        time.sleep(0.1)

        audit_log = temp_log_dir / LOG_FILE_AUDIT
        content = audit_log.read_text(encoding="utf-8")

        assert "data_ETFs.csv" in content
        assert "READ" in content
        assert "252" in content

    def test_log_parameter_change(
        self,
        sample_audit_logger: Audit_Logger,
        temp_log_dir: Path,
    ) -> None:
        """Test logging parameter change event.

        Parameters
        ----------
        sample_audit_logger : Audit_Logger
            Audit logger fixture.
        temp_log_dir : Path
            Temporary log directory.
        """
        sample_audit_logger.log_parameter_change(
            parameter_name="risk_free_rate",
            old_value=0.02,
            new_value=0.025,
            user_id="admin_001",
            reason="Updated to reflect market conditions",
        )

        time.sleep(0.1)

        audit_log = temp_log_dir / LOG_FILE_AUDIT
        content = audit_log.read_text(encoding="utf-8")

        assert "risk_free_rate" in content
        assert "0.02" in content or "0.025" in content

    def test_log_user_action_success(
        self,
        sample_audit_logger: Audit_Logger,
        temp_log_dir: Path,
    ) -> None:
        """Test logging successful user action.

        Parameters
        ----------
        sample_audit_logger : Audit_Logger
            Audit logger fixture.
        temp_log_dir : Path
            Temporary log directory.
        """
        sample_audit_logger.log_user_action(
            action="EXPORT",
            resource="portfolio_report.pdf",
            user_id="analyst_001",
            success=True,
            details={"format": "PDF", "pages": 15},
        )

        time.sleep(0.1)

        audit_log = temp_log_dir / LOG_FILE_AUDIT
        content = audit_log.read_text(encoding="utf-8")

        assert "EXPORT" in content
        assert "portfolio_report.pdf" in content

    def test_log_user_action_failure(
        self,
        sample_audit_logger: Audit_Logger,
        temp_log_dir: Path,
    ) -> None:
        """Test logging failed user action.

        Parameters
        ----------
        sample_audit_logger : Audit_Logger
            Audit logger fixture.
        temp_log_dir : Path
            Temporary log directory.
        """
        sample_audit_logger.log_user_action(
            action="DELETE",
            resource="client_data.csv",
            user_id="analyst_001",
            success=False,
            details={"error": "Permission denied"},
        )

        time.sleep(0.1)

        audit_log = temp_log_dir / LOG_FILE_AUDIT
        content = audit_log.read_text(encoding="utf-8")

        assert "DELETE" in content
        assert "client_data.csv" in content

    @pytest.mark.parametrize(
        "severity",
        ["INFO", "WARNING", "ERROR"],
    )
    def test_log_system_event(
        self,
        sample_audit_logger: Audit_Logger,
        severity: str,
    ) -> None:
        """Test logging system events with different severities.

        Parameters
        ----------
        sample_audit_logger : Audit_Logger
            Audit logger fixture.
        severity : str
            Log severity to test.
        """
        sample_audit_logger.log_system_event(
            event_name="CACHE_CLEARED",
            details={"cache_size_mb": 256},
            severity=severity,
        )
        # Should not raise exceptions


# ==============================================================================
# Test Performance Logging
# ==============================================================================


@pytest.mark.unit()
class Test_Performance_Logging:
    """Test suite for performance logging utilities."""

    def test_log_performance_basic(self, temp_log_dir: Path) -> None:
        """Test basic performance logging.

        Parameters
        ----------
        temp_log_dir : Path
            Temporary log directory.
        """
        setup_logging(log_dir=temp_log_dir, enable_console=False)

        log_performance(
            operation_name="test_operation",
            execution_time_ms=123.45,
            details={"rows_processed": 1000},
        )

        time.sleep(0.1)

        perf_log = temp_log_dir / LOG_FILE_PERFORMANCE
        assert perf_log.exists()

    def test_log_performance_with_custom_logger(
        self,
        temp_log_dir: Path,
    ) -> None:
        """Test performance logging with custom logger instance.

        Parameters
        ----------
        temp_log_dir : Path
            Temporary log directory.
        """
        setup_logging(log_dir=temp_log_dir, enable_console=False)
        custom_logger = get_logger("custom_perf")

        log_performance(
            operation_name="custom_operation",
            execution_time_ms=50.0,
            logger_instance=custom_logger,
        )

    def test_performance_timer_context_manager(
        self,
        temp_log_dir: Path,
    ) -> None:
        """Test Performance_Timer as context manager.

        Parameters
        ----------
        temp_log_dir : Path
            Temporary log directory.
        """
        setup_logging(log_dir=temp_log_dir, enable_console=False)

        with Performance_Timer("test_operation") as timer:
            time.sleep(0.01)  # Simulate work

        assert timer.elapsed_ms > 0
        assert timer.start_time > 0
        assert timer.end_time > timer.start_time

    def test_performance_timer_with_log_start(
        self,
        temp_log_dir: Path,
    ) -> None:
        """Test Performance_Timer with log_start enabled.

        Parameters
        ----------
        temp_log_dir : Path
            Temporary log directory.
        """
        setup_logging(log_dir=temp_log_dir, enable_console=False)

        with Performance_Timer("test_operation", log_start=True) as timer:
            time.sleep(0.01)

        assert timer.elapsed_ms > 0

    def test_performance_timer_with_exception(
        self,
        temp_log_dir: Path,
    ) -> None:
        """Test Performance_Timer handles exceptions gracefully.

        Parameters
        ----------
        temp_log_dir : Path
            Temporary log directory.
        """
        setup_logging(log_dir=temp_log_dir, enable_console=False)

        with pytest.raises(ValueError), Performance_Timer("test_operation") as timer:
            msg = "Test exception"
            raise ValueError(msg)

        # Timer should still have recorded time
        assert timer.elapsed_ms > 0

    def test_performance_timer_details(self, temp_log_dir: Path) -> None:
        """Test Performance_Timer with custom details.

        Parameters
        ----------
        temp_log_dir : Path
            Temporary log directory.
        """
        setup_logging(log_dir=temp_log_dir, enable_console=False)

        with Performance_Timer(
            "test_operation",
            details={"input_size": 1000, "algorithm": "quicksort"},
        ) as timer:
            time.sleep(0.01)

        assert timer.elapsed_ms > 0
        assert timer.details["input_size"] == 1000


# ==============================================================================
# Test Decorator
# ==============================================================================


@pytest.mark.unit()
class Test_Log_Function_Call_Decorator:
    """Test suite for log_function_call decorator."""

    def test_decorator_basic_usage(self, temp_log_dir: Path) -> None:
        """Test basic decorator usage.

        Parameters
        ----------
        temp_log_dir : Path
            Temporary log directory.
        """
        setup_logging(log_dir=temp_log_dir, enable_console=False)

        @log_function_call()
        def test_function(a: int, b: int) -> int:
            """Test function."""
            return a + b

        result = test_function(5, 3)
        assert result == 8

    def test_decorator_logs_performance(self, temp_log_dir: Path) -> None:
        """Test decorator logs performance metrics.

        Parameters
        ----------
        temp_log_dir : Path
            Temporary log directory.
        """
        setup_logging(log_dir=temp_log_dir, enable_console=False)

        @log_function_call(log_performance=True)
        def slow_function() -> str:
            """Slow function for testing."""
            time.sleep(0.01)
            return "done"

        result = slow_function()
        assert result == "done"

    def test_decorator_with_exception(self, temp_log_dir: Path) -> None:
        """Test decorator handles exceptions.

        Parameters
        ----------
        temp_log_dir : Path
            Temporary log directory.
        """
        setup_logging(log_dir=temp_log_dir, enable_console=False)

        @log_function_call()
        def failing_function() -> None:
            """Function that raises exception."""
            msg = "Test error"
            raise ValueError(msg)

        with pytest.raises(ValueError):
            failing_function()

    def test_decorator_preserves_function_metadata(self) -> None:
        """Test decorator preserves function name and docstring."""

        @log_function_call()
        def documented_function() -> str:
            """This is a documented function."""
            return "result"

        assert documented_function.__name__ == "documented_function"
        assert documented_function.__doc__ == "This is a documented function."

    def test_decorator_with_audit_event_type(self, temp_log_dir: Path) -> None:
        """Test decorator with audit event type.

        Parameters
        ----------
        temp_log_dir : Path
            Temporary log directory.
        """
        setup_logging(log_dir=temp_log_dir, enable_console=False)

        @log_function_call(audit_event_type="CALCULATION")
        def calculation_function(x: float, y: float) -> float:
            """Calculate sum."""
            return x + y

        result = calculation_function(10.5, 20.3)
        assert result == pytest.approx(30.8)

    def test_decorator_log_args_disabled(self, temp_log_dir: Path) -> None:
        """Test decorator with log_args disabled.

        Parameters
        ----------
        temp_log_dir : Path
            Temporary log directory.
        """
        setup_logging(log_dir=temp_log_dir, enable_console=False)

        @log_function_call(log_args=False)
        def function_with_sensitive_args(password: str) -> bool:
            """Function with sensitive arguments."""
            return len(password) > 8

        result = function_with_sensitive_args("secret123")
        assert result is True


# ==============================================================================
# Integration Tests
# ==============================================================================


@pytest.mark.integration()
class Test_Logger_Integration:
    """Integration tests for complete logging workflows."""

    def test_complete_audit_workflow(self, temp_log_dir: Path) -> None:
        """Test complete audit logging workflow.

        Parameters
        ----------
        temp_log_dir : Path
            Temporary log directory.
        """
        # Setup
        setup_logging(
            log_level="DEBUG",
            environment="production",
            log_dir=temp_log_dir,
            enable_console=False,
        )

        # Create audit logger
        audit = Audit_Logger("integration_test")

        # Log various events
        audit.log_calculation(
            operation="Portfolio_Optimization",
            inputs={"assets": ["AAPL", "MSFT", "GOOG"]},
            result={"weights": [0.4, 0.3, 0.3]},
            execution_time_ms=156.7,
            user_id="analyst_001",
        )

        audit.log_data_access(
            source="database/portfolio_data",
            operation="READ",
            record_count=500,
            user_id="analyst_001",
        )

        audit.log_user_action(
            action="EXPORT",
            resource="portfolio_report.xlsx",
            user_id="analyst_001",
            success=True,
        )

        time.sleep(0.2)

        # Verify audit log exists and contains events
        audit_log = temp_log_dir / LOG_FILE_AUDIT
        assert audit_log.exists()

        content = audit_log.read_text(encoding="utf-8")
        assert "Portfolio_Optimization" in content
        assert "analyst_001" in content
        assert "portfolio_data" in content

    def test_multiple_log_files_created(self, temp_log_dir: Path) -> None:
        """Test that multiple log files are created as expected.

        Parameters
        ----------
        temp_log_dir : Path
            Temporary log directory.
        """
        setup_logging(
            log_level="DEBUG",
            environment="development",
            log_dir=temp_log_dir,
            enable_console=False,
        )

        test_logger = get_logger(__name__)
        audit = Audit_Logger("test")

        # Generate various log entries
        test_logger.debug("Debug message")
        test_logger.info("Info message")
        test_logger.error("Error message")

        audit.log_calculation(
            operation="Test",
            inputs={},
            result=None,
            execution_time_ms=10.0,
        )

        with Performance_Timer("test_op"):
            time.sleep(0.01)

        time.sleep(0.2)

        # Verify multiple log files exist
        assert (temp_log_dir / LOG_FILE_APPLICATION).exists()
        assert (temp_log_dir / LOG_FILE_AUDIT).exists()
        assert (temp_log_dir / LOG_FILE_ERROR).exists()
        assert (temp_log_dir / LOG_FILE_PERFORMANCE).exists()
        assert (temp_log_dir / LOG_FILE_DEBUG).exists()

    def test_json_log_format_is_valid(self, temp_log_dir: Path) -> None:
        """Test that JSON log entries are valid JSON.

        Parameters
        ----------
        temp_log_dir : Path
            Temporary log directory.
        """
        setup_logging(
            log_level="INFO",
            log_dir=temp_log_dir,
            enable_console=False,
            enable_JSON=True,
        )

        test_logger = get_logger(__name__)
        test_logger.info("Test JSON message")

        time.sleep(0.1)

        # Read application log
        app_log = temp_log_dir / LOG_FILE_APPLICATION
        content = app_log.read_text(encoding="utf-8")

        # Each line should be valid JSON
        # Note: loguru's native serialize=True produces a different structure
        # with 'record' and 'text' fields, not the custom format
        for line in content.strip().split("\n"):
            if line:
                parsed = json.loads(line)
                # Loguru's native JSON structure
                assert "record" in parsed
                assert "text" in parsed
                # Verify record contains expected fields
                assert "level" in parsed["record"]
                assert "message" in parsed["record"]


# ==============================================================================
# Edge Cases and Error Handling
# ==============================================================================


@pytest.mark.unit()
class Test_Edge_Cases:
    """Test suite for edge cases and error handling."""

    def test_serialize_none_value(self) -> None:
        """Test serializing None value."""
        result = serialize_for_JSON(None)
        assert result == "None"

    def test_serialize_empty_dict(self) -> None:
        """Test serializing empty dictionary."""
        result = serialize_for_JSON({})
        assert result == "{}"

    def test_serialize_nested_objects(self) -> None:
        """Test serializing nested objects."""
        data = {
            "timestamp": datetime.now(UTC),
            "path": Path("/tmp/test"),
            "nested": {"value": 42},
        }

        result = serialize_for_JSON(data)
        assert isinstance(result, str)

    def test_audit_logger_with_none_values(
        self,
        sample_audit_logger: Audit_Logger,
    ) -> None:
        """Test audit logger handles None values gracefully.

        Parameters
        ----------
        sample_audit_logger : Audit_Logger
            Audit logger fixture.
        """
        # Should not raise exceptions with None values
        sample_audit_logger.log_calculation(
            operation="Test",
            inputs={},
            result=None,
            execution_time_ms=0.0,
            user_id=None,
            session_id=None,
        )

    def test_performance_timer_zero_duration(self, temp_log_dir: Path) -> None:
        """Test Performance_Timer with near-zero duration.

        Parameters
        ----------
        temp_log_dir : Path
            Temporary log directory.
        """
        setup_logging(log_dir=temp_log_dir, enable_console=False)

        with Performance_Timer("instant_op") as timer:
            pass  # No work

        # Should still record some time
        assert timer.elapsed_ms >= 0

    def test_logger_with_unicode_messages(self, temp_log_dir: Path) -> None:
        """Test logger handles Unicode characters correctly.

        Parameters
        ----------
        temp_log_dir : Path
            Temporary log directory.
        """
        setup_logging(log_dir=temp_log_dir, enable_console=False)

        test_logger = get_logger(__name__)
        test_logger.info("Test with émojis: 🚀 📊 💰")

        time.sleep(0.1)

        app_log = temp_log_dir / LOG_FILE_APPLICATION
        assert app_log.exists()


# ==============================================================================
# Run Tests
# ==============================================================================


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
