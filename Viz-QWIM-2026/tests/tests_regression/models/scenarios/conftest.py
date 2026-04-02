# pyright: reportArgumentType=false
"""Shared fixtures for scenarios regression tests.

The fixtures here use the **canonical** test inputs.  They are identical to
the data used by ``generate_baselines.py`` â€” do NOT change them without also
regenerating the Parquet baselines.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from src.num_methods.scenarios.scenarios_CMA import Scenarios_CMA
from src.num_methods.scenarios.scenarios_distrib import (
    Distribution_Type,
    Scenarios_Distribution,
)


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASELINES_DIR: Path = Path(__file__).resolve().parents[3] / "regression_data" / "scenarios"


# ---------------------------------------------------------------------------
# Fixed-seed input parameters (must match generate_baselines.py constants)
# ---------------------------------------------------------------------------
RANDOM_SEED: int = 42
NUM_DAYS: int = 60
START_DATE: date = date(2024, 1, 2)

# --- Distribution components ---
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

# --- CMA components ---
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
# Fixture helpers
# ---------------------------------------------------------------------------


def load_baseline(name: str) -> pl.DataFrame:
    """Load a named baseline Parquet file.

    Parameters
    ----------
    name : str
        Basename without extension, e.g. ``"distrib_normal_scenarios"``.

    Returns
    -------
    pl.DataFrame

    Raises
    ------
    FileNotFoundError
        If the Parquet file does not exist (baselines not yet generated).
    """
    path = BASELINES_DIR / f"{name}.parquet"
    if not path.exists():
        raise FileNotFoundError(
            f"Baseline not found: {path}\n"
            "Run: python tests/regression_data/scenarios/generate_baselines.py",
        )
    return pl.read_parquet(path)


# ---------------------------------------------------------------------------
# Session-scoped fixtures â€” Scenarios_Distribution
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def baselines_dir() -> Path:
    """Return the path to the regression baselines directory."""
    return BASELINES_DIR


@pytest.fixture(scope="session")
def scen_distrib_normal() -> Scenarios_Distribution:
    """Scenarios_Distribution with Normal distribution â€” generated once per session."""
    scen = Scenarios_Distribution(
        names_components=DISTRIB_COMPONENTS,
        distribution_type=Distribution_Type.NORMAL,
        mean_returns=MEAN_RETURNS,
        covariance_matrix=COV_MATRIX,
        start_date=START_DATE,
        num_days=NUM_DAYS,
        random_seed=RANDOM_SEED,
        name_scenarios="Normal Regression Baseline",
    )
    scen.generate()
    return scen


@pytest.fixture(scope="session")
def scen_distrib_student_t() -> Scenarios_Distribution:
    """Scenarios_Distribution with Student-t distribution."""
    scen = Scenarios_Distribution(
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
    scen.generate()
    return scen


@pytest.fixture(scope="session")
def scen_distrib_lognormal() -> Scenarios_Distribution:
    """Scenarios_Distribution with lognormal distribution."""
    scen = Scenarios_Distribution(
        names_components=DISTRIB_COMPONENTS,
        distribution_type=Distribution_Type.LOGNORMAL,
        mean_returns=LOGNORMAL_MEANS,
        covariance_matrix=COV_MATRIX,
        start_date=START_DATE,
        num_days=NUM_DAYS,
        random_seed=RANDOM_SEED,
        name_scenarios="Lognormal Regression Baseline",
    )
    scen.generate()
    return scen


@pytest.fixture(scope="session")
def scen_distrib_corr_vols() -> Scenarios_Distribution:
    """Scenarios_Distribution constructed via from_correlation_and_volatilities."""
    scen = Scenarios_Distribution.from_correlation_and_volatilities(
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
    scen.generate()
    return scen


# ---------------------------------------------------------------------------
# Session-scoped fixtures â€” Scenarios_CMA
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def scen_CMA() -> Scenarios_CMA:
    """Scenarios_CMA with canonical CMA inputs â€” generated once per session."""
    scen = Scenarios_CMA(
        names_asset_classes=CMA_ASSET_CLASSES,
        expected_returns_annual=CMA_EXPECTED_RETURNS_ANNUAL,
        expected_vols_annual=CMA_EXPECTED_VOLS_ANNUAL,
        correlation_matrix=CMA_CORRELATION,
        start_date=START_DATE,
        num_days=NUM_DAYS,
        random_seed=RANDOM_SEED,
        name_scenarios="CMA Regression Baseline",
    )
    scen.generate()
    return scen
