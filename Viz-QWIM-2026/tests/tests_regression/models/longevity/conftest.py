"""Shared fixtures for longevity model regression tests.

The data fixtures here are **canonical**.  They reproduce the exact same
data used by ``generate_baselines.py``.  Do NOT change them without
regenerating the Parquet baselines first.

Usage
-----
    pytest tests/tests_regression/models/longevity/ -v -m regression
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl
import pytest

from src.models.longevity.model_longevity_constant import (
    Longevity_Model_Constant,
)
from src.models.longevity.model_longevity_standard import (
    Longevity_Model_Standard,
)


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASELINES_DIR: Path = Path(__file__).resolve().parents[3] / "regression_data" / "longevity"

# ---------------------------------------------------------------------------
# Fixed constants (MUST match generate_baselines.py)
# ---------------------------------------------------------------------------

# Constant-model defaults
CONSTANT_DEFAULT_QX: float = 0.02
CONSTANT_N_AGES: int = 20
CONSTANT_START_AGE: int = 65

# Gompertz constructor priors
GOMPERTZ_B: float = 5e-5
GOMPERTZ_b: float = 0.09

# Prediction parameters
GOMPERTZ_N_AGES_SHORT: int = 20
GOMPERTZ_N_AGES_LONG: int = 50
GOMPERTZ_START_AGE: int = 40

# Life-expectancy / survival ages
LIFE_EXPECTANCY_AGES: list[int] = [0, 30, 50, 65, 80]
SURVIVAL_AGES: list[int] = [40, 50, 65, 75]
SURVIVAL_HORIZONS: list[float] = [1.0, 5.0, 10.0, 20.0, 30.0]

# Force of mortality ages
FORCE_OF_MORTALITY_AGES: list[int] = [0, 20, 40, 60, 65, 70, 80, 90, 100]


# ---------------------------------------------------------------------------
# Baseline loader
# ---------------------------------------------------------------------------


def load_baseline(name: str) -> pl.DataFrame:
    """Load a Parquet baseline file by *name* (without ``.parquet`` suffix).

    Parameters
    ----------
    name : str
        Basename of the Parquet file (e.g. ``"constant_predict_default"``).

    Returns
    -------
    pl.DataFrame
        Contents of the baseline file.
    """
    path = BASELINES_DIR / f"{name}.parquet"
    return pl.read_parquet(path)


# ---------------------------------------------------------------------------
# Canonical historical data fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def constant_historical_qx() -> pl.DataFrame:
    """Return the canonical 5-row historical qx dataset for the constant model."""
    return pl.DataFrame(
        {
            "Age": [60, 65, 70, 75, 80],
            "qx": [0.010, 0.014, 0.021, 0.032, 0.045],
        },
    )


@pytest.fixture(scope="session")
def gompertz_historical_qx() -> pl.DataFrame:
    """Build the canonical 60-row Gompertz-generated life table for OLS fitting.

    Uses the known Gompertz formula with B=5e-5, b=0.09.
    Matches ``_build_gompertz_historical_qx()`` in ``generate_baselines.py``.
    """
    ages = np.arange(30, 90, dtype=np.float64)
    B_true = 5e-5
    b_true = 0.09

    integral_mu = (B_true / b_true) * np.exp(b_true * ages) * (np.exp(b_true) - 1.0)
    qx_values = 1.0 - np.exp(-integral_mu)

    return pl.DataFrame(
        {
            "Age": ages.astype(int).tolist(),
            "qx": qx_values.tolist(),
        },
    )


# ---------------------------------------------------------------------------
# Fitted model fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def fitted_constant_default() -> Longevity_Model_Constant:
    """Constant model with constructor qx=0.02, empty-fit."""
    model = Longevity_Model_Constant(qx=CONSTANT_DEFAULT_QX)
    model.fit(pl.DataFrame())
    return model


@pytest.fixture(scope="session")
def fitted_constant_historical(
    constant_historical_qx: pl.DataFrame,
) -> Longevity_Model_Constant:
    """Constant model fitted to the canonical 5-row historical data."""
    model = Longevity_Model_Constant()
    model.fit(constant_historical_qx)
    return model


@pytest.fixture(scope="session")
def fitted_gompertz(
    gompertz_historical_qx: pl.DataFrame,
) -> Longevity_Model_Standard:
    """Gompertz model fitted to the canonical 60-row life table."""
    model = Longevity_Model_Standard(B=GOMPERTZ_B, b=GOMPERTZ_b)
    model.fit(gompertz_historical_qx)
    return model
