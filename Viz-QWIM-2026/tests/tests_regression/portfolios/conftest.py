"""Conftest for portfolios regression tests.

Provides fixtures that load the pre-generated parquet baseline files and the
JSON metadata so test functions can compare computed results against known-good
snapshots.
"""

from __future__ import annotations

import json
from pathlib import Path

import polars as pl
import pytest

# ---------------------------------------------------------------------------
# Absolute path to the regression baseline artefacts
# ---------------------------------------------------------------------------
_BASELINE_DIR = (
    Path(__file__).resolve().parents[2]  # tests/
    / "regression_data"
    / "portfolios"
)

# ---------------------------------------------------------------------------
# Ensure the real portfolio_QWIM class is available across all tests in this
# suite.  The module uses a relative-import fallback that may not resolve
# under pytest, so we patch it once at session scope.
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session", autouse=True)
def _patch_portfolio_class():
    """Inject portfolio_QWIM into utils_portfolio at session start."""
    from src.portfolios import utils_portfolio  # noqa: PLC0415
    from src.portfolios.portfolio_QWIM import portfolio_QWIM  # noqa: PLC0415

    original = utils_portfolio.portfolio
    utils_portfolio.portfolio = portfolio_QWIM
    yield
    utils_portfolio.portfolio = original


# ---------------------------------------------------------------------------
# Baseline fixture helpers
# ---------------------------------------------------------------------------


def _require_baselines() -> None:
    """Skip the entire test if baselines have not been generated yet."""
    missing = [
        f
        for f in (
            "baseline_portfolio_values.parquet",
            "baseline_benchmark_values.parquet",
            "baseline_weights.parquet",
            "baseline_metadata.json",
        )
        if not (_BASELINE_DIR / f).exists()
    ]
    if missing:
        pytest.skip(
            f"Regression baselines missing: {missing}. "
            "Run tests/regression_data/portfolios/generate_baselines.py first."
        )


@pytest.fixture(scope="session")
def baseline_metadata() -> dict:
    """Return the baseline metadata dictionary (session-scoped)."""
    _require_baselines()
    with (_BASELINE_DIR / "baseline_metadata.json").open(encoding="utf-8") as fh:
        return json.load(fh)


@pytest.fixture(scope="session")
def baseline_portfolio_values() -> pl.DataFrame:
    """Return the baseline portfolio values DataFrame (session-scoped)."""
    _require_baselines()
    return pl.read_parquet(_BASELINE_DIR / "baseline_portfolio_values.parquet")


@pytest.fixture(scope="session")
def baseline_benchmark_values() -> pl.DataFrame:
    """Return the baseline benchmark values DataFrame (session-scoped)."""
    _require_baselines()
    return pl.read_parquet(_BASELINE_DIR / "baseline_benchmark_values.parquet")


@pytest.fixture(scope="session")
def baseline_weights() -> pl.DataFrame:
    """Return the baseline weights DataFrame (session-scoped)."""
    _require_baselines()
    return pl.read_parquet(_BASELINE_DIR / "baseline_weights.parquet")
