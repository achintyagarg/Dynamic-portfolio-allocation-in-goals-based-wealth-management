"""Shared fixtures for scenarios regression tests.

The data fixtures here are **canonical**.  They reproduce the exact same
data used by ``generate_baselines.py``.  Do NOT change them without
regenerating the Parquet baselines first.

Usage
-----
    pytest tests/tests_regression/num_methods/scenarios/ -v -m regression
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from src.num_methods.scenarios.scenarios_CMA import (
    Scenarios_CMA,
)
from src.num_methods.scenarios.scenarios_distrib import (
    Distribution_Type,
    Scenarios_Distribution,
)


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASELINES_DIR: Path = (
    Path(__file__).resolve().parents[3] / "regression_data" / "scenarios"
)


# ---------------------------------------------------------------------------
# Fixed seeds and parameters (MUST match generate_baselines.py)
# ---------------------------------------------------------------------------
RANDOM_SEED: int = 42
NUM_DAYS: int = 60
START_DATE: dt.date = dt.date(2024, 1, 2)

# --- Distribution scenario components ---
DISTRIB_COMPONENTS: list[str] = [
    "US_Equity",
    "Intl_Equity",
    "US_Bond",
    "REIT",
    "Gold",
]
MEAN_RETURNS: np.ndarray = np.array(
    [0.0005, 0.0004, 0.0002, 0.0003, 0.0001],
    dtype=np.float64,
)
VOLATILITIES: np.ndarray = np.array(
    [0.012, 0.014, 0.006, 0.010, 0.008],
    dtype=np.float64,
)
CORRELATION: np.ndarray = np.array(
    [
        [1.00, 0.65, 0.20, 0.40, 0.10],
        [0.65, 1.00, 0.15, 0.35, 0.08],
        [0.20, 0.15, 1.00, 0.25, 0.30],
        [0.40, 0.35, 0.25, 1.00, 0.15],
        [0.10, 0.08, 0.30, 0.15, 1.00],
    ],
    dtype=np.float64,
)
COV_MATRIX: np.ndarray = np.outer(VOLATILITIES, VOLATILITIES) * CORRELATION
LOGNORMAL_MEANS: np.ndarray = 1.0 + MEAN_RETURNS

# --- CMA scenario components ---
CMA_ASSET_CLASSES: list[str] = [
    "US Large Cap",
    "International Developed",
    "US Investment Grade Bonds",
    "Real Estate",
]
CMA_EXPECTED_RETURNS_ANNUAL: np.ndarray = np.array(
    [0.08, 0.07, 0.03, 0.06],
    dtype=np.float64,
)
CMA_EXPECTED_VOLS_ANNUAL: np.ndarray = np.array(
    [0.16, 0.18, 0.05, 0.14],
    dtype=np.float64,
)
CMA_CORRELATION: np.ndarray = np.array(
    [
        [1.00, 0.70, 0.10, 0.50],
        [0.70, 1.00, 0.05, 0.40],
        [0.10, 0.05, 1.00, 0.20],
        [0.50, 0.40, 0.20, 1.00],
    ],
    dtype=np.float64,
)


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------


def load_baseline(name: str) -> pl.DataFrame:
    """Load a Parquet baseline file.

    Parameters
    ----------
    name : str
        Stem of the Parquet file (without ``.parquet`` extension).

    Returns
    -------
    pl.DataFrame

    Raises
    ------
    FileNotFoundError
        If the baseline file does not exist.
    """
    path = BASELINES_DIR / f"{name}.parquet"
    if not path.exists():
        raise FileNotFoundError(
            f"Baseline not found: {path}\n"
            "Re-generate with: "
            "python tests/regression_data/scenarios/generate_baselines.py"
        )
    return pl.read_parquet(path)


# ---------------------------------------------------------------------------
# Canonical fixtures — Scenarios_Distribution
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def canonical_distrib_normal() -> "Scenarios_Distribution":
    """Canonical Normal Scenarios_Distribution instance (seed=42)."""
    return Scenarios_Distribution(
        names_components=DISTRIB_COMPONENTS,
        distribution_type=Distribution_Type.NORMAL,
        mean_returns=MEAN_RETURNS,
        covariance_matrix=COV_MATRIX,
        start_date=START_DATE,
        num_days=NUM_DAYS,
        random_seed=RANDOM_SEED,
        name_scenarios="Normal Regression Baseline",
    )


@pytest.fixture(scope="session")
def canonical_distrib_student_t() -> "Scenarios_Distribution":
    """Canonical Student-t Scenarios_Distribution instance (seed=42, nu=5)."""
    return Scenarios_Distribution(
        names_components=DISTRIB_COMPONENTS,
        distribution_type=Distribution_Type.STUDENT_T,
        mean_returns=MEAN_RETURNS,
        covariance_matrix=COV_MATRIX,
        degrees_of_freedom=5.0,
        start_date=START_DATE,
        num_days=NUM_DAYS,
        random_seed=RANDOM_SEED,
        name_scenarios="Student-t Regression Baseline",
    )


@pytest.fixture(scope="session")
def canonical_distrib_lognormal() -> "Scenarios_Distribution":
    """Canonical Lognormal Scenarios_Distribution instance (seed=42)."""
    return Scenarios_Distribution(
        names_components=DISTRIB_COMPONENTS,
        distribution_type=Distribution_Type.LOGNORMAL,
        mean_returns=LOGNORMAL_MEANS,
        covariance_matrix=COV_MATRIX,
        start_date=START_DATE,
        num_days=NUM_DAYS,
        random_seed=RANDOM_SEED,
        name_scenarios="Lognormal Regression Baseline",
    )


@pytest.fixture(scope="session")
def canonical_distrib_corr_vols() -> "Scenarios_Distribution":
    """Canonical correlation+volatilities Scenarios_Distribution (seed=42)."""
    return Scenarios_Distribution.from_correlation_and_volatilities(
        names_components=DISTRIB_COMPONENTS,
        correlation_matrix=CORRELATION,
        volatilities=VOLATILITIES,
        distribution_type=Distribution_Type.NORMAL,
        mean_returns=MEAN_RETURNS,
        start_date=START_DATE,
        num_days=NUM_DAYS,
        random_seed=RANDOM_SEED,
        name_scenarios="Corr+Vols Regression Baseline",
    )


# ---------------------------------------------------------------------------
# Canonical fixtures — Scenarios_CMA
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def canonical_cma() -> "Scenarios_CMA":
    """Canonical Scenarios_CMA instance (seed=42)."""
    return Scenarios_CMA(
        names_asset_classes=CMA_ASSET_CLASSES,
        expected_returns_annual=CMA_EXPECTED_RETURNS_ANNUAL,
        expected_vols_annual=CMA_EXPECTED_VOLS_ANNUAL,
        correlation_matrix=CMA_CORRELATION,
        start_date=START_DATE,
        num_days=NUM_DAYS,
        random_seed=RANDOM_SEED,
        name_scenarios="CMA Regression Baseline",
    )
