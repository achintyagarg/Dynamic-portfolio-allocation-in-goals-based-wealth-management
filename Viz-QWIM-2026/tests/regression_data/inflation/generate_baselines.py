"""Generate Parquet baseline files for inflation model regression tests.

Run this script whenever an **intentional** change is made to the
inflation models.  Commit the updated Parquet files together with the
corresponding code change so that regression tests continue to pass.

Usage
-----
    python tests/regression_data/inflation/generate_baselines.py

Baselines written
-----------------
* ``constant_default_predict_12mo.parquet``
* ``constant_fitted_historical_predict_12mo.parquet``
* ``constant_fitted_historical_summary.parquet``
* ``standard_fitted_annual_predict_12mo.parquet``
* ``standard_fitted_annual_summary.parquet``
* ``standard_fitted_annual_parameters.parquet``
* ``standard_fitted_monthly_predict_24mo.parquet``
* ``standard_analytical_mean_variance.parquet``

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

from src.models.inflation.model_inflation_constant import Inflation_Model_Constant  # noqa: E402
from src.models.inflation.model_inflation_standard import Inflation_Model_Standard  # noqa: E402


# ---------------------------------------------------------------------------
# Path where Parquet baselines are stored
# ---------------------------------------------------------------------------
BASELINES_DIR = Path(__file__).resolve().parent
BASELINES_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Canonical seeds and data (MUST NOT change without regenerating baselines)
# ---------------------------------------------------------------------------
RANDOM_SEED: int = 42
N_SIMULATIONS: int = 1_000
PREDICT_SEED: int = 99


def _build_historical_annual_data() -> pl.DataFrame:
    """Build the canonical 10-year annual inflation dataset."""
    rng = np.random.default_rng(RANDOM_SEED)
    n_years = 10
    dates = [f"{yr}-01-01" for yr in range(2015, 2015 + n_years)]
    baseline = 0.025
    rates = baseline + 0.01 * rng.standard_normal(n_years)
    return pl.DataFrame({"Date": dates, "inflation_rate": rates.tolist()})


def _build_historical_monthly_data() -> pl.DataFrame:
    """Build the canonical 48-month inflation dataset."""
    rng = np.random.default_rng(RANDOM_SEED + 7)
    start = pl.date(2021, 1, 1)
    end = pl.date(2024, 12, 1)
    dates = pl.date_range(start=start, end=end, interval="1mo", eager=True)
    rates = 0.025 + 0.005 * rng.standard_normal(len(dates))
    return pl.DataFrame({"Date": dates, "inflation_rate": rates.tolist()})


def _build_constant_historical_data() -> pl.DataFrame:
    """Build the canonical 5-year annual data for the constant model."""
    return pl.DataFrame(
        {
            "Date": [
                "2020-01-01",
                "2021-01-01",
                "2022-01-01",
                "2023-01-01",
                "2024-01-01",
            ],
            "inflation_rate": [0.023, 0.047, 0.080, 0.034, 0.032],
        },
    )


# ======================================================================
# Constant model baselines
# ======================================================================


def _generate_constant_baselines() -> None:
    """Generate Parquet baselines for Inflation_Model_Constant."""
    # ---- Baseline 1: default rate (2.5%), empty fit, 12-month predict ----
    model_default = Inflation_Model_Constant(annual_rate=0.025)
    model_default.fit(pl.DataFrame())
    df_predict_default = model_default.predict(
        n_periods=12,
        start_date="2025-01-01",
        freq="1mo",
    )
    path_default = BASELINES_DIR / "constant_default_predict_12mo.parquet"
    df_predict_default.write_parquet(path_default)
    print(f"  Written: {path_default.name}")

    # ---- Baseline 2: fitted to historical data, 12-month predict ----
    hist_data = _build_constant_historical_data()
    model_hist = Inflation_Model_Constant()
    model_hist.fit(hist_data)
    df_predict_hist = model_hist.predict(
        n_periods=12,
        start_date="2025-01-01",
        freq="1mo",
    )
    path_hist = BASELINES_DIR / "constant_fitted_historical_predict_12mo.parquet"
    df_predict_hist.write_parquet(path_hist)
    print(f"  Written: {path_hist.name}")

    # ---- Baseline 3: summary statistics of historical prediction ----
    df_summary = model_hist.get_summary_statistics(df_predict_hist)
    path_summary = BASELINES_DIR / "constant_fitted_historical_summary.parquet"
    df_summary.write_parquet(path_summary)
    print(f"  Written: {path_summary.name}")


# ======================================================================
# Standard model baselines
# ======================================================================


def _generate_standard_baselines() -> None:
    """Generate Parquet baselines for Inflation_Model_Standard."""
    # ---- Data ----
    annual_data = _build_historical_annual_data()
    monthly_data = _build_historical_monthly_data()

    # ---- Baseline 1: fitted to annual data, 12-month predict (seeded) ----
    model_annual = Inflation_Model_Standard(n_simulations=N_SIMULATIONS)
    model_annual.fit(annual_data)

    df_predict_annual = model_annual.predict(
        n_periods=12,
        start_date="2025-01-01",
        freq="1mo",
        seed=PREDICT_SEED,
    )
    path_annual = BASELINES_DIR / "standard_fitted_annual_predict_12mo.parquet"
    df_predict_annual.write_parquet(path_annual)
    print(f"  Written: {path_annual.name}")

    # ---- Baseline 2: summary statistics of annual prediction ----
    # Need to use base-class summary (operates on inflation_rate column)
    # For the standard model, predict returns per-path summary columns.
    # We store the predict DataFrame itself as the baseline.
    # Create a separate summary-compatible DataFrame:
    summary_df = pl.DataFrame(
        {
            "Mean_Rate": [df_predict_annual["Mean"].mean()],
            "Median_Rate": [df_predict_annual["Median"].mean()],
            "Mean_Std": [df_predict_annual["Std"].mean()],
            "Mean_P5": [df_predict_annual["P5"].mean()],
            "Mean_P95": [df_predict_annual["P95"].mean()],
        },
    )
    path_summary = BASELINES_DIR / "standard_fitted_annual_summary.parquet"
    summary_df.write_parquet(path_summary)
    print(f"  Written: {path_summary.name}")

    # ---- Baseline 3: fitted parameters ----
    params = model_annual.parameters
    params_df = pl.DataFrame(
        {key: [value] for key, value in params.items()},
    )
    path_params = BASELINES_DIR / "standard_fitted_annual_parameters.parquet"
    params_df.write_parquet(path_params)
    print(f"  Written: {path_params.name}")

    # ---- Baseline 4: fitted to monthly data, 24-month predict ----
    model_monthly = Inflation_Model_Standard(n_simulations=N_SIMULATIONS)
    model_monthly.fit(monthly_data)

    df_predict_monthly = model_monthly.predict(
        n_periods=24,
        start_date="2025-01-01",
        freq="1mo",
        seed=PREDICT_SEED,
    )
    path_monthly = BASELINES_DIR / "standard_fitted_monthly_predict_24mo.parquet"
    df_predict_monthly.write_parquet(path_monthly)
    print(f"  Written: {path_monthly.name}")

    # ---- Baseline 5: analytical mean and variance at several horizons ----
    horizons = [0.5, 1.0, 2.0, 5.0, 10.0]
    analytical_rows = []
    for t_yr in horizons:
        mean_val = model_annual.get_analytical_mean(t_yr)
        var_val = model_annual.get_analytical_variance(t_yr)
        analytical_rows.append(
            {"T_Years": t_yr, "Analytical_Mean": mean_val, "Analytical_Variance": var_val},
        )
    df_analytical = pl.DataFrame(analytical_rows)
    path_analytical = BASELINES_DIR / "standard_analytical_mean_variance.parquet"
    df_analytical.write_parquet(path_analytical)
    print(f"  Written: {path_analytical.name}")


# ======================================================================
# Entry point
# ======================================================================


def main() -> None:
    """Generate all inflation baseline Parquet files."""
    print(f"Generating inflation baselines in: {BASELINES_DIR}")
    print()

    print("[Constant model baselines]")
    _generate_constant_baselines()
    print()

    print("[Standard model baselines]")
    _generate_standard_baselines()
    print()

    n_files = len(list(BASELINES_DIR.glob("*.parquet")))
    print(f"Done — {n_files} Parquet baseline(s) in {BASELINES_DIR}")


if __name__ == "__main__":
    main()
