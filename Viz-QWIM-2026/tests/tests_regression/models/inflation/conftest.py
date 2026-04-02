"""Shared fixtures for inflation model regression tests.

The data fixtures here are **canonical**.  They reproduce the exact same
data used by ``generate_baselines.py``.  Do NOT change them without
regenerating the Parquet baselines first.

Usage
-----
    pytest tests/tests_regression/models/inflation/ -v -m regression
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl
import pytest

from src.models.inflation.model_inflation_constant import Inflation_Model_Constant
from src.models.inflation.model_inflation_standard import Inflation_Model_Standard


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASELINES_DIR: Path = Path(__file__).resolve().parents[3] / "regression_data" / "inflation"

# ---------------------------------------------------------------------------
# Fixed seeds and simulation parameters (MUST match generate_baselines.py)
# ---------------------------------------------------------------------------
RANDOM_SEED: int = 42
N_SIMULATIONS: int = 1_000
PREDICT_SEED: int = 99


# ---------------------------------------------------------------------------
# Canonical data fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def canonical_annual_data() -> pl.DataFrame:
    """Build the canonical 10-year annual inflation dataset.

    Matches ``_build_historical_annual_data()`` in ``generate_baselines.py``.
    """
    rng = np.random.default_rng(RANDOM_SEED)
    n_years = 10
    dates = [f"{yr}-01-01" for yr in range(2015, 2015 + n_years)]
    baseline = 0.025
    rates = baseline + 0.01 * rng.standard_normal(n_years)
    return pl.DataFrame({"Date": dates, "inflation_rate": rates.tolist()})


@pytest.fixture(scope="session")
def canonical_monthly_data() -> pl.DataFrame:
    """Build the canonical 48-month inflation dataset.

    Matches ``_build_historical_monthly_data()`` in ``generate_baselines.py``.
    """
    rng = np.random.default_rng(RANDOM_SEED + 7)
    start = pl.date(2021, 1, 1)
    end = pl.date(2024, 12, 1)
    dates = pl.date_range(start=start, end=end, interval="1mo", eager=True)
    rates = 0.025 + 0.005 * rng.standard_normal(len(dates))
    return pl.DataFrame({"Date": dates, "inflation_rate": rates.tolist()})


@pytest.fixture(scope="session")
def canonical_constant_historical_data() -> pl.DataFrame:
    """Build the canonical 5-year data for the constant inflation model.

    Matches ``_build_constant_historical_data()`` in ``generate_baselines.py``.
    """
    return pl.DataFrame(
        {
            "Date": [
                "2020-01-01",
                "2021-01-01",
                "2022-01-01",
                "2023-01-01",
                "2024-01-01",
            ],
            "inflation_rate": [0.023, 0.047, 0.080, 0.034, 0.032],
        },
    )


# ---------------------------------------------------------------------------
# Fitted model fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def fitted_constant_default() -> Inflation_Model_Constant:
    """Return a constant model fitted with empty data (uses default 2.5 %)."""
    model = Inflation_Model_Constant(annual_rate=0.025)
    model.fit(pl.DataFrame())
    return model


@pytest.fixture(scope="session")
def fitted_constant_historical(
    canonical_constant_historical_data: pl.DataFrame,
) -> Inflation_Model_Constant:
    """Return a constant model fitted on the canonical 5-year data."""
    model = Inflation_Model_Constant()
    model.fit(canonical_constant_historical_data)
    return model


@pytest.fixture(scope="session")
def fitted_standard_annual(
    canonical_annual_data: pl.DataFrame,
) -> Inflation_Model_Standard:
    """Return a standard OU model fitted on canonical annual data."""
    model = Inflation_Model_Standard(n_simulations=N_SIMULATIONS)
    model.fit(canonical_annual_data)
    return model


@pytest.fixture(scope="session")
def fitted_standard_monthly(
    canonical_monthly_data: pl.DataFrame,
) -> Inflation_Model_Standard:
    """Return a standard OU model fitted on canonical monthly data."""
    model = Inflation_Model_Standard(n_simulations=N_SIMULATIONS)
    model.fit(canonical_monthly_data)
    return model


# ---------------------------------------------------------------------------
# Baseline loader
# ---------------------------------------------------------------------------


def load_baseline(name: str) -> pl.DataFrame:
    """Load a named baseline Parquet file.

    Parameters
    ----------
    name:
        Basename without extension, e.g. ``"constant_default_predict_12mo"``.

    Returns
    -------
    pl.DataFrame
        The DataFrame stored in the Parquet baseline file.

    Raises
    ------
    FileNotFoundError
        If the Parquet file does not exist — run ``generate_baselines.py``.
    """
    path = BASELINES_DIR / f"{name}.parquet"
    if not path.exists():
        msg = (
            f"Baseline not found: {path}\n"
            "Run: python tests/regression_data/inflation/generate_baselines.py"
        )
        raise FileNotFoundError(msg)
    return pl.read_parquet(path)
