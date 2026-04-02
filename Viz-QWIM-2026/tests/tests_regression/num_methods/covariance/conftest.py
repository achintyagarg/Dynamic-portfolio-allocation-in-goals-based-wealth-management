"""Shared fixtures for covariance/correlation regression tests.

The data fixtures here are **canonical**.  They reproduce the exact same
data used by ``generate_baselines.py``.  Do NOT change them without
regenerating the Parquet baselines first.

Usage
-----
    pytest tests/tests_regression/num_methods/covariance/ -v -m regression
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl
import pytest

from src.num_methods.covariance.utils_cov_corr import (
    covariance_estimator,
    covariance_matrix,
)


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASELINES_DIR: Path = Path(__file__).resolve().parents[3] / "regression_data" / "covariance"

# ---------------------------------------------------------------------------
# Fixed seeds and parameters (MUST match generate_baselines.py)
# ---------------------------------------------------------------------------
RANDOM_SEED: int = 42
NUM_OBS_3_ASSETS: int = 200
NUM_OBS_5_ASSETS: int = 300
COMPONENT_NAMES_3: list[str] = ["AAPL", "MSFT", "GOOG"]
COMPONENT_NAMES_5: list[str] = ["AAPL", "MSFT", "GOOG", "AMZN", "TSLA"]


# ---------------------------------------------------------------------------
# Canonical data fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def canonical_returns_3_assets() -> pl.DataFrame:
    """Build the canonical 200-obs x 3-asset correlated returns dataset.

    Matches ``_build_canonical_returns_3_assets()`` in ``generate_baselines.py``.
    """
    rng = np.random.default_rng(RANDOM_SEED)
    num_obs = NUM_OBS_3_ASSETS

    factor = rng.standard_normal(num_obs) * 0.01
    noise = rng.standard_normal((num_obs, 3)) * 0.005
    returns = factor[:, np.newaxis] + noise

    dates = pl.date_range(
        pl.date(2023, 1, 1),
        pl.date(2023, 1, 1) + pl.duration(days=num_obs - 1),
        eager=True,
    ).alias("Date")

    return pl.DataFrame(
        {
            "Date": dates,
            "AAPL": returns[:, 0].tolist(),
            "MSFT": returns[:, 1].tolist(),
            "GOOG": returns[:, 2].tolist(),
        },
    )


@pytest.fixture(scope="session")
def canonical_returns_5_assets() -> pl.DataFrame:
    """Build the canonical 300-obs x 5-asset correlated returns dataset.

    Matches ``_build_canonical_returns_5_assets()`` in ``generate_baselines.py``.
    """
    rng = np.random.default_rng(RANDOM_SEED)
    num_obs = NUM_OBS_5_ASSETS

    factor_1 = rng.standard_normal(num_obs) * 0.01
    factor_2 = rng.standard_normal(num_obs) * 0.008

    loadings = np.array(
        [
            [1.0, 0.5],
            [0.8, 0.6],
            [0.7, 0.7],
            [1.2, 0.3],
            [1.5, 0.2],
        ]
    )
    factors = np.column_stack([factor_1, factor_2])
    systematic = factors @ loadings.T

    noise = rng.standard_normal((num_obs, 5)) * 0.004
    returns = systematic + noise

    dates = pl.date_range(
        pl.date(2022, 1, 1),
        pl.date(2022, 1, 1) + pl.duration(days=num_obs - 1),
        eager=True,
    ).alias("Date")

    return pl.DataFrame(
        {
            "Date": dates,
            "AAPL": returns[:, 0].tolist(),
            "MSFT": returns[:, 1].tolist(),
            "GOOG": returns[:, 2].tolist(),
            "AMZN": returns[:, 3].tolist(),
            "TSLA": returns[:, 4].tolist(),
        },
    )


# ---------------------------------------------------------------------------
# Fitted covariance_matrix fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def fitted_empirical_3(
    canonical_returns_3_assets: pl.DataFrame,
) -> covariance_matrix:
    """Return covariance_matrix fitted with EMPIRICAL estimator on 3 assets."""
    return covariance_matrix(
        data_returns=canonical_returns_3_assets,
        estimator=covariance_estimator.EMPIRICAL,  # pyright: ignore[reportArgumentType]
    )


@pytest.fixture(scope="session")
def fitted_ledoit_wolf_3(
    canonical_returns_3_assets: pl.DataFrame,
) -> covariance_matrix:
    """Return covariance_matrix fitted with LEDOIT_WOLF estimator on 3 assets."""
    return covariance_matrix(
        data_returns=canonical_returns_3_assets,
        estimator=covariance_estimator.LEDOIT_WOLF,  # pyright: ignore[reportArgumentType]
    )


@pytest.fixture(scope="session")
def fitted_oas_3(
    canonical_returns_3_assets: pl.DataFrame,
) -> covariance_matrix:
    """Return covariance_matrix fitted with OAS estimator on 3 assets."""
    return covariance_matrix(
        data_returns=canonical_returns_3_assets,
        estimator=covariance_estimator.ORACLE_APPROXIMATING_SHRINKAGE,  # pyright: ignore[reportArgumentType]
    )


@pytest.fixture(scope="session")
def fitted_shrunk_3(
    canonical_returns_3_assets: pl.DataFrame,
) -> covariance_matrix:
    """Return covariance_matrix fitted with SHRUNK_COVARIANCE estimator on 3 assets."""
    return covariance_matrix(
        data_returns=canonical_returns_3_assets,
        estimator=covariance_estimator.SHRUNK_COVARIANCE,  # pyright: ignore[reportArgumentType]
    )


@pytest.fixture(scope="session")
def fitted_ew_3(
    canonical_returns_3_assets: pl.DataFrame,
) -> covariance_matrix:
    """Return covariance_matrix fitted with EW estimator on 3 assets."""
    return covariance_matrix(
        data_returns=canonical_returns_3_assets,
        estimator=covariance_estimator.EXPONENTIALLY_WEIGHTED,  # pyright: ignore[reportArgumentType]
    )


@pytest.fixture(scope="session")
def fitted_empirical_5(
    canonical_returns_5_assets: pl.DataFrame,
) -> covariance_matrix:
    """Return covariance_matrix fitted with EMPIRICAL estimator on 5 assets."""
    return covariance_matrix(
        data_returns=canonical_returns_5_assets,
        estimator=covariance_estimator.EMPIRICAL,  # pyright: ignore[reportArgumentType]
    )


@pytest.fixture(scope="session")
def fitted_ledoit_wolf_5(
    canonical_returns_5_assets: pl.DataFrame,
) -> covariance_matrix:
    """Return covariance_matrix fitted with LEDOIT_WOLF estimator on 5 assets."""
    return covariance_matrix(
        data_returns=canonical_returns_5_assets,
        estimator=covariance_estimator.LEDOIT_WOLF,  # pyright: ignore[reportArgumentType]
    )


# ---------------------------------------------------------------------------
# Baseline loader
# ---------------------------------------------------------------------------


def load_baseline(name: str) -> pl.DataFrame:
    """Load a named baseline Parquet file.

    Parameters
    ----------
    name:
        Basename without extension, e.g. ``"cov_empirical_3_assets"``.

    Returns
    -------
    pl.DataFrame
        The DataFrame stored in the Parquet baseline file.

    Raises
    ------
    FileNotFoundError
        If the Parquet file does not exist -- run ``generate_baselines.py``.
    """
    path = BASELINES_DIR / f"{name}.parquet"
    if not path.exists():
        msg = (
            f"Baseline not found: {path}\n"
            "Run: python tests/regression_data/covariance/generate_baselines.py"
        )
        raise FileNotFoundError(msg)
    return pl.read_parquet(path)
