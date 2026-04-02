"""
Robot Framework keyword library for num_methods.covariance.
============================================================

Tests cover:
    - covariance_matrix construction with multiple estimators
    - Shape, component count, observation count
    - validate_covariance_matrix pass/fail
    - Diagonal positivity, symmetry
    - Correlation matrix: diagonal==1, symmetry, PSD
    - Component accessor agreement with full matrix
    - distance_estimator_type classmethods

Author:         QWIM Development Team
Version:        0.1.0
Last Modified:  2026-02-28
"""

from __future__ import annotations

import io
import sys
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Project root on sys.path
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parents[4]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# ---------------------------------------------------------------------------
# Conditional imports
# Robot Framework replaces sys.stderr with StringIO; some project modules
# access sys.stderr.buffer at import time. Temporarily restore a real stream.
# ---------------------------------------------------------------------------
MODULE_IMPORT_AVAILABLE: bool = True
_import_error_message: str = ""

_original_stderr = sys.stderr
if not hasattr(sys.stderr, "buffer"):
    sys.stderr = io.TextIOWrapper(io.BytesIO())

try:
    import polars as pl
    from src.num_methods.covariance.utils_cov_corr import (
        covariance_estimator,
        covariance_matrix,
        distance_estimator_type,
    )
except Exception as _exc:  # noqa: BLE001
    MODULE_IMPORT_AVAILABLE = False
    _import_error_message = str(_exc)
finally:
    sys.stderr = _original_stderr

_COMPONENTS_3 = ["AAPL", "MSFT", "GOOG"]
_COMPONENTS_5 = ["AAPL", "MSFT", "GOOG", "AMZN", "TSLA"]


def _require_imports() -> None:
    """Raise RuntimeError when required imports are unavailable."""
    if not MODULE_IMPORT_AVAILABLE:
        raise RuntimeError(
            f"Required imports not available: {_import_error_message}"
        )


def _build_returns_df(
    num_components: int = 3,
    num_obs: int = 60,
    seed: int = 42,
) -> "pl.DataFrame":
    """Build synthetic returns DataFrame.

    Arguments:
    - num_components -- number of asset columns (2-5)
    - num_obs         -- number of observation rows
    - seed            -- random seed for reproducibility

    Returns: pl.DataFrame with Date + component columns
    """
    _require_imports()
    rng = np.random.default_rng(seed)
    components = _COMPONENTS_3[:num_components]
    factor = rng.standard_normal(num_obs) * 0.01
    noise = rng.standard_normal((num_obs, num_components)) * 0.005
    raw = factor[:, np.newaxis] + noise
    dates = pl.date_range(
        pl.date(2023, 1, 1),
        pl.date(2023, 1, 1) + pl.duration(days=num_obs - 1),
        interval="1d",
        eager=True,
    )
    return pl.DataFrame(
        {
            "Date": dates,
            **{c: raw[:, i].tolist() for i, c in enumerate(components)},
        }
    )


# ---------------------------------------------------------------------------
# Construction keywords
# ---------------------------------------------------------------------------


def build_empirical_covariance_matrix(
    num_components: int = 3,
    num_obs: int = 60,
) -> "covariance_matrix":
    """Construct an Empirical covariance_matrix from synthetic data.

    Arguments:
    - num_components -- number of asset columns
    - num_obs         -- number of observation rows

    Returns: covariance_matrix instance
    """
    _require_imports()
    df = _build_returns_df(int(num_components), int(num_obs))
    return covariance_matrix(
        data_returns=df,
        estimator=covariance_estimator.EMPIRICAL,
    )


def build_covariance_matrix_with_estimator(
    estimator_name: str,
    num_components: int = 3,
    num_obs: int = 60,
) -> "covariance_matrix":
    """Construct a covariance_matrix with the named estimator string.

    Arguments:
    - estimator_name  -- member name of covariance_estimator enum (e.g. 'LEDOIT_WOLF')
    - num_components  -- number of asset columns
    - num_obs         -- number of observation rows

    Returns: covariance_matrix instance
    """
    _require_imports()
    df = _build_returns_df(int(num_components), int(num_obs))
    estimator = covariance_estimator[estimator_name]
    return covariance_matrix(
        data_returns=df,
        estimator=estimator,
    )


# ---------------------------------------------------------------------------
# Property / accessor keywords
# ---------------------------------------------------------------------------


def get_covariance_matrix_shape(cov_obj: "covariance_matrix") -> tuple[int, int]:
    """Return the (rows, cols) shape of the covariance matrix.

    Arguments:
    - cov_obj -- covariance_matrix instance

    Returns: tuple (rows, cols)
    """
    _require_imports()
    return cov_obj.get_covariance_matrix().shape


def get_num_components(cov_obj: "covariance_matrix") -> int:
    """Return m_num_components.

    Arguments:
    - cov_obj -- covariance_matrix instance

    Returns: int
    """
    return int(cov_obj.m_num_components)


def get_num_observations(cov_obj: "covariance_matrix") -> int:
    """Return m_num_observations.

    Arguments:
    - cov_obj -- covariance_matrix instance

    Returns: int
    """
    return int(cov_obj.m_num_observations)


def validate_covariance_matrix(cov_obj: "covariance_matrix") -> bool:
    """Call validate_covariance_matrix and return True on success.

    Arguments:
    - cov_obj -- covariance_matrix instance

    Returns: True if validation passes; raises on failure
    """
    _require_imports()
    cov_obj.validate_covariance_matrix()
    return True


def covariance_diagonal_is_positive(cov_obj: "covariance_matrix") -> bool:
    """Check that all diagonal elements are strictly positive.

    Arguments:
    - cov_obj -- covariance_matrix instance

    Returns: True if all variances are positive
    """
    _require_imports()
    mat = cov_obj.get_covariance_matrix()
    return bool(np.all(np.diag(mat) > 0))


def covariance_is_symmetric(cov_obj: "covariance_matrix") -> bool:
    """Check that the covariance matrix is symmetric within 1e-8.

    Arguments:
    - cov_obj -- covariance_matrix instance

    Returns: True if symmetric
    """
    _require_imports()
    mat = cov_obj.get_covariance_matrix()
    return bool(np.allclose(mat, mat.T, atol=1e-8))


# ---------------------------------------------------------------------------
# Correlation matrix keywords
# ---------------------------------------------------------------------------


def get_correlation_matrix(cov_obj: "covariance_matrix") -> np.ndarray:
    """Compute and return the correlation matrix.

    Arguments:
    - cov_obj -- covariance_matrix instance

    Returns: np.ndarray (K, K)
    """
    _require_imports()
    return cov_obj.get_correlation_matrix()


def correlation_diagonal_equals_one(cov_obj: "covariance_matrix") -> bool:
    """Check that correlation matrix diagonal is all 1.0 (atol=1e-10).

    Arguments:
    - cov_obj -- covariance_matrix instance

    Returns: True if diagonal == 1.0
    """
    _require_imports()
    diag = np.diag(cov_obj.get_correlation_matrix())
    return bool(np.allclose(diag, 1.0, atol=1e-10))


def correlation_is_symmetric(cov_obj: "covariance_matrix") -> bool:
    """Check that the correlation matrix is symmetric within 1e-10.

    Arguments:
    - cov_obj -- covariance_matrix instance

    Returns: True if symmetric
    """
    _require_imports()
    corr = cov_obj.get_correlation_matrix()
    return bool(np.allclose(corr, corr.T, atol=1e-10))


def correlation_values_in_bounds(cov_obj: "covariance_matrix") -> bool:
    """Check that all correlation values lie in [-1, 1].

    Arguments:
    - cov_obj -- covariance_matrix instance

    Returns: True if all in bounds
    """
    _require_imports()
    corr = cov_obj.get_correlation_matrix()
    return bool(np.all(corr >= -1.0 - 1e-8) and np.all(corr <= 1.0 + 1e-8))


# ---------------------------------------------------------------------------
# Component accessor keywords
# ---------------------------------------------------------------------------


def get_component_variance(
    cov_obj: "covariance_matrix",
    component_name: str,
) -> float:
    """Return the variance for the named component.

    Arguments:
    - cov_obj        -- covariance_matrix instance
    - component_name -- string name of the component

    Returns: float variance
    """
    _require_imports()
    return float(cov_obj.get_component_variance(component_name))


def get_component_covariance(
    cov_obj: "covariance_matrix",
    component_name_1: str,
    component_name_2: str,
) -> float:
    """Return the covariance between two components.

    Arguments:
    - cov_obj          -- covariance_matrix instance
    - component_name_1 -- first component name
    - component_name_2 -- second component name

    Returns: float covariance
    """
    _require_imports()
    return float(cov_obj.get_component_covariance(component_name_1, component_name_2))


# ---------------------------------------------------------------------------
# distance_estimator_type keywords
# ---------------------------------------------------------------------------


def get_correlation_based_estimators() -> list:
    """Return correlation-based distance estimators.

    Returns: list of distance_estimator_type members
    """
    _require_imports()
    return distance_estimator_type.get_correlation_based()


def get_rank_based_estimators() -> list:
    """Return rank-based distance estimators.

    Returns: list of distance_estimator_type members
    """
    _require_imports()
    return distance_estimator_type.get_rank_based()


def count_distance_estimator_members() -> int:
    """Return the total number of distance_estimator_type members.

    Returns: int
    """
    _require_imports()
    return len(list(distance_estimator_type))


def all_distance_estimators_categorised() -> bool:
    """Check that all enum members appear in at least one category list.

    Returns: True if fully covered
    """
    _require_imports()
    all_members = set(distance_estimator_type)
    categorised = set(
        distance_estimator_type.get_correlation_based()
        + distance_estimator_type.get_covariance_based()
        + distance_estimator_type.get_rank_based()
        + distance_estimator_type.get_information_theoretic()
    )
    return all_members == categorised
