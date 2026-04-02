"""
Pytest configuration and fixtures for tests/tests_unit/utils.

This module provides shared fixtures and configuration for testing
the src/utils module.
"""

from datetime import date
from pathlib import Path

import polars as pl
import pytest


# ============================================================================
# Common Test Data Fixtures
# ============================================================================


@pytest.fixture()
def project_root():
    """Return the project root directory."""
    return Path(__file__).resolve().parents[3]


@pytest.fixture()
def sample_date_range():
    """Create a sample date range for testing."""
    return pl.date_range(
        date(2023, 1, 1),
        date(2023, 1, 10),
        interval="1d",
        eager=True,
    )


@pytest.fixture()
def sample_etf_prices(sample_date_range):
    """Create sample ETF price data."""
    return pl.DataFrame(
        {
            "Date": sample_date_range,
            "VTI": [100.0, 101.0, 102.0, 101.5, 103.0, 104.0, 103.5, 105.0, 106.0, 107.0],
            "AGG": [50.0, 50.2, 50.1, 50.3, 50.4, 50.2, 50.5, 50.6, 50.4, 50.7],
            "VNQ": [80.0, 81.0, 79.0, 80.5, 82.0, 81.5, 83.0, 82.5, 84.0, 85.0],
        },
    )


@pytest.fixture()
def sample_portfolio_weights():
    """Create sample portfolio weights data."""
    return pl.DataFrame(
        {
            "Date": [date(2023, 1, 1), date(2023, 1, 5)],
            "VTI": [0.6, 0.5],
            "AGG": [0.3, 0.35],
            "VNQ": [0.1, 0.15],
        },
    )


@pytest.fixture()
def temp_data_directory(tmp_path):
    """Create a temporary data directory structure."""
    raw_dir = tmp_path / "data" / "raw"
    processed_dir = tmp_path / "data" / "processed"
    raw_dir.mkdir(parents=True)
    processed_dir.mkdir(parents=True)

    return {
        "root": tmp_path,
        "raw": raw_dir,
        "processed": processed_dir,
    }


# ============================================================================
# Pytest Markers Configuration
# ============================================================================


def pytest_configure(config):
    """Configure custom pytest markers."""
    config.addinivalue_line(
        "markers",
        "unit: marks tests as unit tests (fast, no external dependencies)",
    )
    config.addinivalue_line(
        "markers",
        "integration: marks tests as integration tests (may use external resources)",
    )
    config.addinivalue_line(
        "markers",
        "slow: marks tests as slow (excluded from regular test runs)",
    )
    config.addinivalue_line(
        "markers",
        "regression: marks tests as regression tests (compare against known values)",
    )
