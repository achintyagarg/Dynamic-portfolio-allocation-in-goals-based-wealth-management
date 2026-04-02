"""Shared fixtures for simulation regression tests.

The parameters here are the **canonical** test inputs.  They are identical
to the data used by ``generate_baselines.py`` — do NOT change them without
also regenerating the Parquet baselines.
"""

from __future__ import annotations

import datetime as dt

from pathlib import Path

import numpy as np
import polars as pl
import pytest

from src.models.simulation.model_simulation_standard import Simulation_Standard
from src.num_methods.scenarios.scenarios_distrib import Distribution_Type


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASELINES_DIR: Path = Path(__file__).resolve().parents[3] / "regression_data" / "simulation"

# ---------------------------------------------------------------------------
# Fixed-seed input parameters (must match generate_baselines.py constants)
# ---------------------------------------------------------------------------
RANDOM_SEED: int = 42
NUM_DAYS: int = 60
NUM_SCENARIOS: int = 50
INITIAL_VALUE: float = 100.0
START_DATE: dt.date = dt.date(2024, 1, 2)
ASSETS: list[str] = ["VTI", "VXUS", "BND", "VNQ", "GLD"]

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

EQUAL_WEIGHTS: np.ndarray = np.array(
    [0.20, 0.20, 0.20, 0.20, 0.20],
    dtype=np.float64,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def baselines_dir() -> Path:
    """Return the path to the simulation regression baselines directory."""
    return BASELINES_DIR


@pytest.fixture(scope="session")
def sim_normal() -> Simulation_Standard:
    """Build and run the canonical Normal-distribution simulation."""
    sim = Simulation_Standard(
        names_components=ASSETS,
        weights=EQUAL_WEIGHTS,
        distribution_type=Distribution_Type.NORMAL,  # type: ignore[reportArgumentType]
        mean_returns=MEAN_RETURNS,
        covariance_matrix=COV_MATRIX,
        initial_value=INITIAL_VALUE,
        num_scenarios=NUM_SCENARIOS,
        num_days=NUM_DAYS,
        start_date=START_DATE,
        random_seed=RANDOM_SEED,
        name_simulation="test_normal",
    )
    sim.run()
    return sim


@pytest.fixture(scope="session")
def sim_student_t() -> Simulation_Standard:
    """Build and run the canonical Student-t simulation."""
    sim = Simulation_Standard(
        names_components=ASSETS,
        weights=EQUAL_WEIGHTS,
        distribution_type=Distribution_Type.STUDENT_T,  # type: ignore[reportArgumentType]
        mean_returns=MEAN_RETURNS,
        covariance_matrix=COV_MATRIX,
        initial_value=INITIAL_VALUE,
        num_scenarios=NUM_SCENARIOS,
        num_days=NUM_DAYS,
        start_date=START_DATE,
        random_seed=RANDOM_SEED,
        degrees_of_freedom=5.0,
        name_simulation="test_student_t",
    )
    sim.run()
    return sim


@pytest.fixture(scope="session")
def sim_lognormal() -> Simulation_Standard:
    """Build and run the canonical Lognormal simulation."""
    lognormal_means = 1.0 + MEAN_RETURNS
    sim = Simulation_Standard(
        names_components=ASSETS,
        weights=EQUAL_WEIGHTS,
        distribution_type=Distribution_Type.LOGNORMAL,  # type: ignore[reportArgumentType]
        mean_returns=lognormal_means,
        covariance_matrix=COV_MATRIX,
        initial_value=INITIAL_VALUE,
        num_scenarios=NUM_SCENARIOS,
        num_days=NUM_DAYS,
        start_date=START_DATE,
        random_seed=RANDOM_SEED,
        name_simulation="test_lognormal",
    )
    sim.run()
    return sim


@pytest.fixture(scope="session")
def sim_unequal_weights() -> Simulation_Standard:
    """Build and run a simulation with unequal portfolio weights."""
    unequal_weights = np.array([0.40, 0.25, 0.15, 0.10, 0.10], dtype=np.float64)
    sim = Simulation_Standard(
        names_components=ASSETS,
        weights=unequal_weights,
        distribution_type=Distribution_Type.NORMAL,  # type: ignore[reportArgumentType]
        mean_returns=MEAN_RETURNS,
        covariance_matrix=COV_MATRIX,
        initial_value=INITIAL_VALUE,
        num_scenarios=NUM_SCENARIOS,
        num_days=NUM_DAYS,
        start_date=START_DATE,
        random_seed=RANDOM_SEED,
        name_simulation="test_unequal_weights",
    )
    sim.run()
    return sim


@pytest.fixture(scope="session")
def sim_high_value() -> Simulation_Standard:
    """Build and run a simulation with 1M initial value."""
    sim = Simulation_Standard(
        names_components=ASSETS,
        weights=EQUAL_WEIGHTS,
        distribution_type=Distribution_Type.NORMAL,  # type: ignore[reportArgumentType]
        mean_returns=MEAN_RETURNS,
        covariance_matrix=COV_MATRIX,
        initial_value=1_000_000.0,
        num_scenarios=NUM_SCENARIOS,
        num_days=NUM_DAYS,
        start_date=START_DATE,
        random_seed=RANDOM_SEED,
        name_simulation="test_high_value",
    )
    sim.run()
    return sim


def load_baseline(name: str) -> pl.DataFrame:
    """Load a named baseline Parquet file.

    Parameters
    ----------
    name:
        Basename without extension, e.g. ``"sim_normal_full_results"``.

    Returns
    -------
    pl.DataFrame
        DataFrame stored in the Parquet file.

    Raises
    ------
    FileNotFoundError
        If the Parquet file does not exist (baselines not yet generated).
    """
    path = BASELINES_DIR / f"{name}.parquet"
    if not path.exists():
        msg = (
            f"Baseline not found: {path}\n"
            "Run: python tests/regression_data/simulation/generate_baselines.py"
        )
        raise FileNotFoundError(msg)
    return pl.read_parquet(path)
