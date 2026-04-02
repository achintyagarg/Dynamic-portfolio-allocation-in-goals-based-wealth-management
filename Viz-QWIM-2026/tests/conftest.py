"""
Pytest configuration and fixtures for the entire test suite.

This module provides shared fixtures and configuration used across
all test modules in the project.
"""

import sys  # MUST be first import to enable UTF-8 encoding fix


# CRITICAL: Force UTF-8 encoding for all I/O on Windows BEFORE any other imports
# This prevents UnicodeEncodeError when logging special characters during tests
if sys.platform == "win32":
    import os as _os_for_encoding

    # Set environment variable for subprocess and future operations
    _os_for_encoding.environ.setdefault("PYTHONIOENCODING", "utf-8")

    # Force UTF-8 for default encoding
    if hasattr(sys, "_enablelegacywindowsfsencoding"):
        sys._enablelegacywindowsfsencoding = lambda: None

    # Reconfigure all existing standard streams to UTF-8
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

    if hasattr(sys.stderr, "reconfigure"):
        try:
            sys.stderr.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

    if hasattr(sys.stdin, "reconfigure"):
        try:
            sys.stdin.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

    # Custom exception hook for UTF-8 safe error handling
    import io

    _original_excepthook = sys.excepthook

    def _utf8_safe_excepthook(exc_type, exc_value, exc_traceback):
        """Exception hook that safely handles Unicode characters on Windows."""
        import traceback

        try:
            lines = traceback.format_exception(exc_type, exc_value, exc_traceback)
            message = "".join(lines)
            stderr_utf8 = io.TextIOWrapper(
                sys.stderr.buffer,
                encoding="utf-8",
                errors="replace",
                line_buffering=True,
            )
            stderr_utf8.write(message)
            stderr_utf8.flush()
        except Exception:
            try:
                _original_excepthook(exc_type, exc_value, exc_traceback)
            except Exception:
                print(f"Error: {exc_type.__name__}: {exc_value}", file=sys.stderr)

    sys.excepthook = _utf8_safe_excepthook

# Now safe to import other modules
from pathlib import Path

import pytest


# ============================================================================
# Path Configuration
# ============================================================================


# Ensure the src directory is in the Python path for imports
PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ============================================================================
# Pytest Configuration
# ============================================================================


def pytest_configure(config):
    """Configure custom pytest markers."""
    config.addinivalue_line(
        "markers",
        "unit: marks tests as unit tests (fast, isolated, no external dependencies)",
    )
    config.addinivalue_line(
        "markers",
        "integration: marks tests as integration tests (may use external resources)",
    )
    config.addinivalue_line(
        "markers",
        "slow: marks tests as slow (excluded from regular test runs unless specified)",
    )
    config.addinivalue_line(
        "markers",
        "regression: marks tests as regression tests (compare against known benchmark values)",
    )


# ============================================================================
# Common Fixtures
# ============================================================================


@pytest.fixture(scope="session")
def project_root():
    """Return the project root directory path.

    Returns
    -------
    Path
        Absolute path to the project root directory.
    """
    return PROJECT_ROOT


@pytest.fixture(scope="session")
def src_directory(project_root):
    """Return the src directory path.

    Returns
    -------
    Path
        Absolute path to the src directory.
    """
    return project_root / "src"


@pytest.fixture(scope="session")
def inputs_directory(project_root):
    """Return the inputs directory path.

    Returns
    -------
    Path
        Absolute path to the inputs directory.
    """
    return project_root / "inputs"


@pytest.fixture(scope="session")
def outputs_directory(project_root):
    """Return the outputs directory path.

    Returns
    -------
    Path
        Absolute path to the outputs directory.
    """
    return project_root / "outputs"


@pytest.fixture()
def temp_workspace(tmp_path):
    """Create a temporary workspace with standard directory structure.

    Returns
    -------
    dict
        Dictionary containing paths to temporary directories.
    """
    # Create standard project structure
    (tmp_path / "data" / "raw").mkdir(parents=True)
    (tmp_path / "data" / "processed").mkdir(parents=True)
    (tmp_path / "inputs" / "raw").mkdir(parents=True)
    (tmp_path / "inputs" / "processed").mkdir(parents=True)
    (tmp_path / "outputs").mkdir(parents=True)

    return {
        "root": tmp_path,
        "data_raw": tmp_path / "data" / "raw",
        "data_processed": tmp_path / "data" / "processed",
        "inputs_raw": tmp_path / "inputs" / "raw",
        "inputs_processed": tmp_path / "inputs" / "processed",
        "outputs": tmp_path / "outputs",
    }


# ============================================================================
# Test Skip Conditions
# ============================================================================


@pytest.fixture()
def skip_if_no_network():
    """Skip test if no network connection available."""
    import socket

    try:
        socket.create_connection(("8.8.8.8", 53), timeout=3)
    except OSError:
        pytest.skip("No network connection available")


@pytest.fixture()
def skip_if_no_matplotlib():
    """Skip test if matplotlib is not installed."""
    try:
        import matplotlib
    except ImportError:
        pytest.skip("matplotlib not installed")
