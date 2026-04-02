"""Generate Parquet baseline files for covariance/correlation regression tests.

Run this script whenever an **intentional** change is made to the
covariance estimation module.  Commit the updated Parquet files together
with the corresponding code change so that regression tests continue to
pass.

Usage
-----
    python tests/regression_data/covariance/generate_baselines.py

Baselines written
-----------------
* ``canonical_returns_3_assets.parquet``
* ``canonical_returns_5_assets.parquet``
* ``cov_empirical_3_assets.parquet``
* ``corr_empirical_3_assets.parquet``
* ``cov_ledoit_wolf_3_assets.parquet``
* ``corr_ledoit_wolf_3_assets.parquet``
* ``cov_oas_3_assets.parquet``
* ``corr_oas_3_assets.parquet``
* ``cov_shrunk_3_assets.parquet``
* ``corr_shrunk_3_assets.parquet``
* ``cov_ew_3_assets.parquet``
* ``corr_ew_3_assets.parquet``
* ``cov_empirical_5_assets.parquet``
* ``corr_empirical_5_assets.parquet``
* ``cov_ledoit_wolf_5_assets.parquet``
* ``corr_ledoit_wolf_5_assets.parquet``
* ``component_variances_empirical_3_assets.parquet``
* ``component_covariances_empirical_3_assets.parquet``
* ``metadata_empirical_3_assets.parquet``

Author
------
QWIM Team
"""

from __future__ import annotations

import sys

from pathlib import Path

import numpy as np
import polars as pl


# ---------------------------------------------------------------------------
# Ensure project root is importable
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.num_methods.covariance.utils_cov_corr import (  # noqa: E402
    Covariance_Estimator,
    Covariance_Matrix,
)


# ---------------------------------------------------------------------------
# Path where Parquet baselines are stored
# ---------------------------------------------------------------------------
BASELINES_DIR = Path(__file__).resolve().parent
BASELINES_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Canonical seeds and parameters (MUST NOT change without regenerating)
# ---------------------------------------------------------------------------
RANDOM_SEED: int = 42
NUM_OBS_3_ASSETS: int = 200
NUM_OBS_5_ASSETS: int = 300
COMPONENT_NAMES_3: list[str] = ["AAPL", "MSFT", "GOOG"]
COMPONENT_NAMES_5: list[str] = ["AAPL", "MSFT", "GOOG", "AMZN", "TSLA"]


# ---------------------------------------------------------------------------
# Canonical data builders
# ---------------------------------------------------------------------------


def _build_canonical_returns_3_assets() -> pl.DataFrame:
    """Build the canonical 200-obs x 3-asset correlated returns dataset.

    Uses a factor model with seed 42 to generate correlated returns.
    """
    rng = np.random.default_rng(RANDOM_SEED)
    num_obs = NUM_OBS_3_ASSETS

    # Factor model: shared factor + idiosyncratic noise
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


def _build_canonical_returns_5_assets() -> pl.DataFrame:
    """Build the canonical 300-obs x 5-asset correlated returns dataset.

    Uses a two-factor model with seed 42 to generate returns with
    richer correlation structure.
    """
    rng = np.random.default_rng(RANDOM_SEED)
    num_obs = NUM_OBS_5_ASSETS

    # Two-factor model for richer correlation structure
    factor_1 = rng.standard_normal(num_obs) * 0.01
    factor_2 = rng.standard_normal(num_obs) * 0.008

    # Factor loadings per asset
    loadings = np.array(
        [
            [1.0, 0.5],  # AAPL
            [0.8, 0.6],  # MSFT
            [0.7, 0.7],  # GOOG
            [1.2, 0.3],  # AMZN
            [1.5, 0.2],  # TSLA
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


def _save_parquet(df: pl.DataFrame, name: str) -> None:
    """Write a DataFrame to Parquet and print confirmation."""
    path = BASELINES_DIR / f"{name}.parquet"
    df.write_parquet(path)
    print(f"  Written: {path.name}  ({df.shape[0]} rows x {df.shape[1]} cols)")


def _matrix_to_dataframe(
    matrix: np.ndarray,
    component_names: list[str],
) -> pl.DataFrame:
    """Convert a numpy matrix to a labelled Polars DataFrame."""
    data: dict[str, list[float]] = {}
    for idx_col, name in enumerate(component_names):
        data[name] = matrix[:, idx_col].tolist()
    return pl.DataFrame(data)


# ======================================================================
# 3-asset estimator baselines
# ======================================================================


def _generate_3_asset_baselines(returns_df: pl.DataFrame) -> None:
    """Generate baselines for all estimators with 3-asset data."""
    estimator_configs = [
        ("empirical", Covariance_Estimator.EMPIRICAL),
        ("ledoit_wolf", Covariance_Estimator.LEDOIT_WOLF),
        ("oas", Covariance_Estimator.ORACLE_APPROXIMATING_SHRINKAGE),
        ("shrunk", Covariance_Estimator.SHRUNK_COVARIANCE),
        ("ew", Covariance_Estimator.EXPONENTIALLY_WEIGHTED),
    ]

    for label, estimator in estimator_configs:
        cov_obj = Covariance_Matrix(
            data_returns=returns_df,
            estimator=estimator,  # pyright: ignore[reportArgumentType]
        )

        # Covariance matrix
        cov_df = _matrix_to_dataframe(cov_obj.get_covariance_matrix(), COMPONENT_NAMES_3)
        _save_parquet(cov_df, f"cov_{label}_3_assets")

        # Correlation matrix
        corr_df = _matrix_to_dataframe(cov_obj.get_correlation_matrix(), COMPONENT_NAMES_3)
        _save_parquet(corr_df, f"corr_{label}_3_assets")


def _generate_3_asset_accessor_baselines(returns_df: pl.DataFrame) -> None:
    """Generate baselines for accessor methods with 3-asset EMPIRICAL estimator."""
    cov_obj = Covariance_Matrix(
        data_returns=returns_df,
        estimator=Covariance_Estimator.EMPIRICAL,  # pyright: ignore[reportArgumentType]
    )

    # Component variances
    variances = {
        "Component": COMPONENT_NAMES_3,
        "Variance": [cov_obj.get_component_variance(name) for name in COMPONENT_NAMES_3],
    }
    _save_parquet(pl.DataFrame(variances), "component_variances_empirical_3_assets")

    # Pairwise covariances (upper triangle including diagonal)
    pairs_comp_1: list[str] = []
    pairs_comp_2: list[str] = []
    pairs_cov: list[float] = []
    for idx_i, name_i in enumerate(COMPONENT_NAMES_3):
        for idx_j in range(idx_i, len(COMPONENT_NAMES_3)):
            name_j = COMPONENT_NAMES_3[idx_j]
            pairs_comp_1.append(name_i)
            pairs_comp_2.append(name_j)
            pairs_cov.append(cov_obj.get_component_covariance(name_i, name_j))

    covariances_df = pl.DataFrame(
        {
            "Component_1": pairs_comp_1,
            "Component_2": pairs_comp_2,
            "Covariance": pairs_cov,
        },
    )
    _save_parquet(covariances_df, "component_covariances_empirical_3_assets")

    # Metadata
    metadata_df = pl.DataFrame(
        {
            "Estimator": [cov_obj.m_estimator_type.value],
            "Num_Components": [cov_obj.m_num_components],
            "Num_Observations": [cov_obj.m_num_observations],
            "Matrix_Shape_Rows": [cov_obj.m_cov_matrix.shape[0]],
            "Matrix_Shape_Cols": [cov_obj.m_cov_matrix.shape[1]],
        },
    )
    _save_parquet(metadata_df, "metadata_empirical_3_assets")


# ======================================================================
# 5-asset estimator baselines
# ======================================================================


def _generate_5_asset_baselines(returns_df: pl.DataFrame) -> None:
    """Generate baselines for select estimators with 5-asset data."""
    estimator_configs = [
        ("empirical", Covariance_Estimator.EMPIRICAL),
        ("ledoit_wolf", Covariance_Estimator.LEDOIT_WOLF),
    ]

    for label, estimator in estimator_configs:
        cov_obj = Covariance_Matrix(
            data_returns=returns_df,
            estimator=estimator,  # pyright: ignore[reportArgumentType]
        )

        # Covariance matrix
        cov_df = _matrix_to_dataframe(cov_obj.get_covariance_matrix(), COMPONENT_NAMES_5)
        _save_parquet(cov_df, f"cov_{label}_5_assets")

        # Correlation matrix
        corr_df = _matrix_to_dataframe(cov_obj.get_correlation_matrix(), COMPONENT_NAMES_5)
        _save_parquet(corr_df, f"corr_{label}_5_assets")


# ======================================================================
# Entry point
# ======================================================================


def main() -> None:
    """Generate all covariance/correlation baseline Parquet files."""
    print(f"Generating covariance baselines in: {BASELINES_DIR}")
    print()

    # Build canonical datasets
    returns_3 = _build_canonical_returns_3_assets()
    returns_5 = _build_canonical_returns_5_assets()

    # Save canonical input data
    _save_parquet(returns_3, "canonical_returns_3_assets")
    _save_parquet(returns_5, "canonical_returns_5_assets")
    print()

    print("[3-asset estimator baselines]")
    _generate_3_asset_baselines(returns_3)
    print()

    print("[3-asset accessor baselines]")
    _generate_3_asset_accessor_baselines(returns_3)
    print()

    print("[5-asset estimator baselines]")
    _generate_5_asset_baselines(returns_5)
    print()

    n_files = len(list(BASELINES_DIR.glob("*.parquet")))
    print(f"Done - {n_files} Parquet baseline(s) in {BASELINES_DIR}")


if __name__ == "__main__":
    main()
