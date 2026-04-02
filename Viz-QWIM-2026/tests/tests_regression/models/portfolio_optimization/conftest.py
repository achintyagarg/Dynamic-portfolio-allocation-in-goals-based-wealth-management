"""Shared fixtures for portfolio optimization regression tests.

The returns fixture here is the **canonical** test input.  It is identical to the
data used by generate_baselines.py — do NOT change it without also regenerating the
Parquet baselines.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import numpy as np
import polars as pl
import pytest

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASELINES_DIR = (
    Path(__file__).resolve().parents[3]
    / "regression_data"
    / "portfolio_optimization"
)

# ---------------------------------------------------------------------------
# Fixed-seed input parameters (must match generate_baselines.py constants)
# ---------------------------------------------------------------------------
RANDOM_SEED: int = 42
N_DAYS: int = 252
ASSETS: list[str] = ["VTI", "VXUS", "BND", "VNQ", "GLD"]
FIXED_DATE: str = "2024-01-15"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def canonical_returns_data() -> pl.DataFrame:
    """Build the fixed-seed returns DataFrame used for all regression comparisons.

    Parameters are pinned to the same values used in generate_baselines.py.
    Changing them breaks all regression baselines.
    """
    rng = np.random.default_rng(RANDOM_SEED)

    mean_returns = np.array([0.0005, 0.0004, 0.0002, 0.0003, 0.0001])
    volatilities = np.array([0.012, 0.014, 0.006, 0.010, 0.008])
    correlation = np.array(
        [
            [1.00, 0.65, 0.20, 0.40, 0.10],
            [0.65, 1.00, 0.15, 0.35, 0.08],
            [0.20, 0.15, 1.00, 0.25, 0.30],
            [0.40, 0.35, 0.25, 1.00, 0.15],
            [0.10, 0.08, 0.30, 0.15, 1.00],
        ],
    )
    cov_matrix = np.outer(volatilities, volatilities) * correlation
    returns_array = rng.multivariate_normal(mean_returns, cov_matrix, N_DAYS)

    dates = pl.date_range(
        start=datetime(2023, 1, 2),
        end=datetime(2024, 6, 30),
        interval="1d",
        eager=True,
    )[:N_DAYS]

    return pl.DataFrame(
        {
            "Date": dates,
            **{asset: returns_array[:, i].tolist() for i, asset in enumerate(ASSETS)},
        }
    )


@pytest.fixture(scope="session")
def canonical_benchmark_returns(canonical_returns_data: pl.DataFrame) -> pl.DataFrame:
    """Single-column benchmark returns aligned with canonical_returns_data."""
    rng = np.random.default_rng(RANDOM_SEED + 1)
    n = len(canonical_returns_data)
    bench = rng.normal(0.0003, 0.010, n)
    return pl.DataFrame(
        {"Date": canonical_returns_data["Date"], "SPY": bench.tolist()}
    )


@pytest.fixture(scope="session")
def baselines_dir() -> Path:
    """Return the path to the regression baselines directory."""
    return BASELINES_DIR


def load_baseline(name: str) -> pl.DataFrame:
    """Load a named baseline Parquet file.

    Parameters
    ----------
    name:
        Basename without extension, e.g. ``"skfolio_basic_equal_weighted"``.

    Returns
    -------
    pl.DataFrame
        Single-row weights DataFrame stored in the Parquet file.

    Raises
    ------
    FileNotFoundError
        If the Parquet file does not exist (baselines not yet generated).
    """
    path = BASELINES_DIR / f"{name}.parquet"
    if not path.exists():
        raise FileNotFoundError(
            f"Baseline not found: {path}\n"
            "Run: python tests/regression_data/portfolio_optimization/generate_baselines.py"
        )
    return pl.read_parquet(path)
