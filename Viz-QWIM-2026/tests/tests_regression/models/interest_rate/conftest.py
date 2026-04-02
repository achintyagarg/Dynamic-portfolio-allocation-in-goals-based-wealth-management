"""Shared fixtures for interest-rate model regression tests.

The data fixtures here are **canonical**.  They reproduce the exact same
data used by ``generate_baselines.py``.  Do NOT change them without
regenerating the Parquet baselines first.

Usage
-----
    pytest tests/tests_regression/models/interest_rate/ -v -m regression
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl
import pytest

from src.models.interest_rate.model_interest_rate_constant import (
    Interest_Rate_Model_Constant,
)
from src.models.interest_rate.model_interest_rate_standard import (
    Interest_Rate_Model_Standard,
)


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASELINES_DIR: Path = Path(__file__).resolve().parents[3] / "regression_data" / "interest_rate"

# ---------------------------------------------------------------------------
# Fixed seeds and parameters (MUST match generate_baselines.py)
# ---------------------------------------------------------------------------
RANDOM_SEED: int = 42
VASICEK_SEED: int = 42

# Constant-model defaults
CONSTANT_DEFAULT_RATE: float = 0.05
CONSTANT_N_PERIODS: int = 12
CONSTANT_START_DATE: str = "2025-01-01"

# Vasicek constructor priors
VASICEK_KAPPA: float = 0.3
VASICEK_THETA: float = 0.04
VASICEK_SIGMA: float = 0.015
VASICEK_N_SIMS: int = 500
VASICEK_N_PERIODS_SHORT: int = 12
VASICEK_N_PERIODS_LONG: int = 60
VASICEK_START_DATE: str = "2025-01-01"

# Analytical test horizons
ANALYTICAL_HORIZONS: list[float] = [0.5, 1.0, 2.0, 5.0, 10.0]


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
def constant_historical_rates() -> pl.DataFrame:
    """Return the canonical 5-row historical dataset for the constant model."""
    return pl.DataFrame(
        {
            "Date": ["2020-01-01", "2021-01-01", "2022-01-01", "2023-01-01", "2024-01-01"],
            "short_rate": [0.025, 0.030, 0.050, 0.053, 0.045],
        },
    )


@pytest.fixture(scope="session")
def vasicek_historical_rates() -> pl.DataFrame:
    """Build the canonical 120-obs monthly OU series for Vasicek calibration.

    Matches ``_build_vasicek_historical_rates()`` in ``generate_baselines.py``.
    """
    rng = np.random.default_rng(RANDOM_SEED)

    n_obs = 120
    dt = 1.0 / 12.0
    kappa_true = 0.25
    theta_true = 0.04
    sigma_true = 0.012

    rates = np.empty(n_obs, dtype=np.float64)
    rates[0] = 0.035

    e_kdt = np.exp(-kappa_true * dt)
    cond_var = (sigma_true**2 / (2.0 * kappa_true)) * (1.0 - np.exp(-2.0 * kappa_true * dt))
    cond_std = np.sqrt(max(cond_var, 0.0))

    for idx_step in range(1, n_obs):
        cond_mean = theta_true + (rates[idx_step - 1] - theta_true) * e_kdt
        rates[idx_step] = cond_mean + cond_std * rng.standard_normal()

    dates = pl.date_range(
        start=pl.date(2015, 1, 1),
        end=pl.date(2015, 1, 1) + pl.duration(days=(n_obs + 5) * 31),
        interval="1mo",
        eager=True,
    ).slice(0, n_obs)

    return pl.DataFrame(
        {
            "Date": dates,
            "short_rate": rates.tolist(),
        },
    )


# ---------------------------------------------------------------------------
# Fitted model fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def fitted_constant_default() -> Interest_Rate_Model_Constant:
    """Constant model with constructor rate, empty-fit."""
    model = Interest_Rate_Model_Constant(annual_rate=CONSTANT_DEFAULT_RATE)
    model.fit(pl.DataFrame())
    return model


@pytest.fixture(scope="session")
def fitted_constant_historical(
    constant_historical_rates: pl.DataFrame,
) -> Interest_Rate_Model_Constant:
    """Constant model fitted to the canonical 5-row historical data."""
    model = Interest_Rate_Model_Constant()
    model.fit(constant_historical_rates)
    return model


@pytest.fixture(scope="session")
def fitted_vasicek(
    vasicek_historical_rates: pl.DataFrame,
) -> Interest_Rate_Model_Standard:
    """Vasicek model fitted to the canonical 120-month OU series."""
    model = Interest_Rate_Model_Standard(
        kappa=VASICEK_KAPPA,
        theta=VASICEK_THETA,
        sigma=VASICEK_SIGMA,
        n_simulations=VASICEK_N_SIMS,
    )
    model.fit(vasicek_historical_rates)
    return model
