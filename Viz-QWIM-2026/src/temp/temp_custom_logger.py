
from datetime import datetime


class Formatter_Structured_JSON(logging.Formatter):
    """JSON formatter for structured logging with audit trail support."""
    
    def format_as_JSON(self, record: logging.LogRecord) -> str:
        """Format log record as JSON.
        
        Args:
            record: Log record to format.
            
        Returns:
            JSON-formatted log string.
        """
        # Build base log entry
        log_entry = {
            "timestamp": datetime.fromtimestamp(
                record.created, tz=datetime.timezone.utc
            ).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
            "message": record.getMessage(),
        }
        
        # Add thread/process info for debugging
        if record.levelno >= logging.DEBUG:
            log_entry["thread"] = record.thread
            log_entry["process"] = record.process
        
        # Add exception info if present
        if record.exc_info:
            log_entry["exception"] = self.formatException(record.exc_info)
        
        # Add extra fields from logger.info(..., extra={...})
        if hasattr(record, "event_type"):
            log_entry["event_type"] = record.event_type
        if hasattr(record, "user_id"):
            log_entry["user_id"] = record.user_id
        if hasattr(record, "session_id"):
            log_entry["session_id"] = record.session_id
        if hasattr(record, "execution_time_ms"):
            log_entry["execution_time_ms"] = record.execution_time_ms
        if hasattr(record, "data"):
            log_entry["data"] = record.data
        
        return json.dumps(log_entry, default=str)


class Formatter_Human_Readable(logging.Formatter):
    """Human-readable formatter for console output."""
    
    def __init__(self) -> None:
        super().__init__(
            fmt=(
                "%(asctime)s | %(levelname)-8s | "
                "%(name)s:%(funcName)s:%(lineno)d | %(message)s"
            ),
            datefmt="%Y-%m-%d %H:%M:%S",
        )


def setup_logging(
    log_level: str = "INFO",
    environment: str = "development",
    log_dir: Optional[Path] = None,
    enable_console: bool = True,
) -> None:
    """Configure application-wide logging.
    
    This function must be called once at application startup before any
    logging occurs. It configures:
    - Multiple log files with rotation
    - Structured JSON logging for machine parsing
    - Console logging for development
    - Appropriate retention policies
    
    Args:
        log_level: Minimum log level (DEBUG, INFO, WARNING, ERROR, CRITICAL).
        environment: Deployment environment (development, staging, production).
        log_dir: Directory for log files (default: ./logs/).
        enable_console: Whether to log to console (True for dev, False for prod).
    
    Example:
        >>> setup_logging(log_level="INFO", environment="production")
        >>> logger = logging.getLogger(__name__)
        >>> logger.info("Application started")
    """
    # Create logs directory
    if log_dir is None:
        log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    # Create archive subdirectory
    archive_dir = log_dir / "archive"
    archive_dir.mkdir(exist_ok=True)
    
    # Get root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, log_level.upper()))
    
    # Remove existing handlers to avoid duplicates
    root_logger.handlers.clear()
    
    # Create formatters
    json_formatter = Formatter_Structured_JSON()
    human_formatter = Formatter_Human_Readable()
    
    # 1. Application log - All logs (INFO and above)
    app_handler = logging.handlers.TimedRotatingFileHandler(
        filename=log_dir / "application.log",
        when="midnight",
        interval=1,
        backupCount=90,  # Keep 90 days
        encoding="utf-8",
        utc=True,
    )
    app_handler.setLevel(logging.INFO)
    app_handler.setFormatter(json_formatter)
    app_handler.suffix = "%Y-%m-%d"  # Append date to rotated files
    root_logger.addHandler(app_handler)
    
    # 2. Audit log - INFO level for compliance tracking
    audit_handler = logging.handlers.TimedRotatingFileHandler(
        filename=log_dir / "audit.log",
        when="midnight",
        interval=1,
        backupCount=2555,  # Keep 7 years for regulatory compliance
        encoding="utf-8",
        utc=True,
    )
    audit_handler.setLevel(logging.INFO)
    audit_handler.setFormatter(json_formatter)
    audit_handler.suffix = "%Y-%m-%d"
    # Add filter to only log events with event_type (audit events)
    audit_handler.addFilter(lambda record: hasattr(record, "event_type"))
    root_logger.addHandler(audit_handler)
    
    # 3. Error log - ERROR and above only
    error_handler = logging.handlers.TimedRotatingFileHandler(
        filename=log_dir / "error.log",
        when="midnight",
        interval=1,
        backupCount=365,  # Keep 1 year
        encoding="utf-8",
        utc=True,
    )
    error_handler.setLevel(logging.ERROR)
    error_handler.setFormatter(json_formatter)
    error_handler.suffix = "%Y-%m-%d"
    root_logger.addHandler(error_handler)
    
    # 4. Performance log - Performance metrics only
    perf_handler = logging.handlers.TimedRotatingFileHandler(
        filename=log_dir / "performance.log",
        when="midnight",
        interval=1,
        backupCount=30,  # Keep 30 days
        encoding="utf-8",
        utc=True,
    )
    perf_handler.setLevel(logging.DEBUG)
    perf_handler.setFormatter(json_formatter)
    perf_handler.suffix = "%Y-%m-%d"
    # Filter for performance events
    perf_handler.addFilter(
        lambda record: hasattr(record, "execution_time_ms")
    )
    root_logger.addHandler(perf_handler)
    
    # 5. Console handler (for development)
    if enable_console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.DEBUG if environment == "development" else logging.INFO)
        console_handler.setFormatter(human_formatter)
        root_logger.addHandler(console_handler)
    
    # Log startup
    startup_logger = logging.getLogger(__name__)
    startup_logger.info(
        f"Logging configured",
        extra={
            "event_type": "SYSTEM_STARTUP",
            "data": {
                "log_level": log_level,
                "environment": environment,
                "log_dir": str(log_dir),
                "console_enabled": enable_console,
            },
        },
    )


def get_logger(name: str) -> logging.Logger:
    """Get a logger instance with the given name.
    
    Args:
        name: Logger name (typically __name__).
    
    Returns:
        Configured logger instance.
    
    Example:
        >>> logger = get_logger(__name__)
        >>> logger.info("Processing started")
    """
    return logging.getLogger(name)


class Audit_Logger:
    """Specialized logger for audit trail events.
    
    This class provides convenience methods for logging events that require
    audit trails for compliance purposes.
    """
    
    def __init__(self, logger_name: str) -> None:
        self.logger = logging.getLogger(logger_name)
    
    def log_calculation(
        self,
        operation: str,
        inputs: Dict[str, Any],
        result: Any,
        execution_time_ms: float,
        user_id: Optional[str] = None,
    ) -> None:
        """Log a financial calculation for audit trail.
        
        Args:
            operation: Name of the calculation (e.g., "Black-Scholes Pricing").
            inputs: Input parameters used in calculation.
            result: Calculation result.
            execution_time_ms: Execution time in milliseconds.
            user_id: Optional user identifier.
        """
        self.logger.info(
            f"Calculation: {operation}",
            extra={
                "event_type": "CALCULATION",
                "user_id": user_id or "system",
                "execution_time_ms": execution_time_ms,
                "data": {
                    "operation": operation,
                    "inputs": inputs,
                    "result": result,
                },
            },
        )
    
    def log_data_access(
        self,
        source: str,
        operation: str,
        record_count: int,
        user_id: Optional[str] = None,
    ) -> None:
        """Log data access for audit trail.
        
        Args:
            source: Data source (file path, database, API).
            operation: Type of operation (READ, WRITE, DELETE).
            record_count: Number of records accessed.
            user_id: Optional user identifier.
        """
        self.logger.info(
            f"Data access: {operation} from {source}",
            extra={
                "event_type": "DATA_ACCESS",
                "user_id": user_id or "system",
                "data": {
                    "source": source,
                    "operation": operation,
                    "record_count": record_count,
                },
            },
        )
    
    def log_parameter_change(
        self,
        parameter_name: str,
        old_value: Any,
        new_value: Any,
        user_id: Optional[str] = None,
    ) -> None:
        """Log parameter or configuration change.
        
        Args:
            parameter_name: Name of the parameter changed.
            old_value: Previous value.
            new_value: New value.
            user_id: Optional user identifier.
        """
        self.logger.warning(
            f"Parameter changed: {parameter_name}",
            extra={
                "event_type": "PARAMETER_CHANGE",
                "user_id": user_id or "system",
                "data": {
                    "parameter": parameter_name,
                    "old_value": old_value,
                    "new_value": new_value,
                },
            },
        )


# Example usage in application startup (e.g., src/__init__.py or main script)
if __name__ == "__main__":
    # Configure logging at startup
    setup_logging(
        log_level="INFO",
        environment=os.getenv("ENVIRONMENT", "development"),
        enable_console=True,
    )
    
    # Use in code
    logger = get_logger(__name__)
    logger.info("Application initialized")
    
    # Use audit logger
    audit = Audit_Logger(__name__)
    audit.log_calculation(
        operation="Option Pricing",
        inputs={"spot": 100, "strike": 105, "volatility": 0.2},
        result=3.45,
        execution_time_ms=12.5,
        user_id="analyst_001",
    )
```

**Usage in Application Code**:

```python
# In any module (e.g., src/models/pricing.py)
import logging
import time
from typing import Dict, Any
from decimal import Decimal

logger = logging.getLogger(__name__)

def calculate_option_price(params: Dict[str, Any]) -> Decimal:
    """Calculate option price with full audit logging."""
    start_time = time.perf_counter()
    
    logger.info(
        "Starting option price calculation",
        extra={
            "event_type": "CALCULATION_START",
            "data": {
                "instrument_id": params.get("instrument_id"),
                "model": "Black-Scholes",
            },
        },
    )
    
    try:
        # Perform calculation
        result = Decimal("3.45")  # Actual calculation here
        
        execution_time = (time.perf_counter() - start_time) * 1000
        
        logger.info(
            "Option price calculation completed",
            extra={
                "event_type": "CALCULATION_COMPLETE",
                "execution_time_ms": execution_time,
                "data": {
                    "instrument_id": params.get("instrument_id"),
                    "spot": params.get("spot"),
                    "strike": params.get("strike"),
                    "volatility": params.get("volatility"),
                    "result": float(result),
                    "model_version": "1.0.0",
                },
            },
        )
        
        return result
        
    except Exception as e:
        logger.error(
            f"Option price calculation failed: {str(e)}",
            extra={
                "event_type": "CALCULATION_ERROR",
                "data": {
                    "instrument_id": params.get("instrument_id"),
                    "error_type": type(e).__name__,
                    "inputs": params,
                },
            },
            exc_info=True,
        )
        raise