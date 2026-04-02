"""Generate Parquet baseline files for interest-rate model regression tests.

Run this script whenever an **intentional** change is made to the
interest-rate model modules.  Commit the updated Parquet files together
with the corresponding code change so that regression tests continue to
pass.

Usage
-----
    python tests/regression_data/interest_rate/generate_baselines.py

Baselines written
-----------------
Constant model
~~~~~~~~~~~~~~
* ``constant_historical_rates.parquet``
* ``constant_predict_default.parquet``
* ``constant_predict_fitted.parquet``
* ``constant_summary_default.parquet``
* ``constant_summary_fitted.parquet``
* ``constant_parameters_default.parquet``
* ``constant_parameters_fitted.parquet``
* ``constant_predict_daily.parquet``
* ``constant_predict_yearly.parquet``

Standard (Vasicek) model
~~~~~~~~~~~~~~~~~~~~~~~~
* ``vasicek_historical_rates.parquet``
* ``vasicek_parameters_fitted.parquet``
* ``vasicek_predict_12mo.parquet``
* ``vasicek_predict_60mo.parquet``
* ``vasicek_summary_12mo.parquet``
* ``vasicek_analytical_mean.parquet``
* ``vasicek_analytical_variance.parquet``
* ``vasicek_predict_daily.parquet``

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

from src.models.interest_rate.model_interest_rate_constant import (  # noqa: E402
    Interest_Rate_Model_Constant,
)
from src.models.interest_rate.model_interest_rate_standard import (  # noqa: E402
    Interest_Rate_Model_Standard,
)


# ---------------------------------------------------------------------------
# Fixed seeds and canonical constants — MUST match conftest.py
# ---------------------------------------------------------------------------
RANDOM_SEED: int = 42
VASICEK_SEED: int = 42  # seed for predict()

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

BASELINES_DIR: Path = Path(__file__).resolve().parent


# ---------------------------------------------------------------------------
# Canonical historical data builders
# ---------------------------------------------------------------------------


def _build_constant_historical_rates() -> pl.DataFrame:
    """Build a small deterministic historical dataset for the constant model.

    Returns a 5-row DataFrame with ``Date`` and ``short_rate`` columns.
    Rates are hand-picked to produce a known mean.
    """
    return pl.DataFrame(
        {
            "Date": ["2020-01-01", "2021-01-01", "2022-01-01", "2023-01-01", "2024-01-01"],
            "short_rate": [0.025, 0.030, 0.050, 0.053, 0.045],
        },
    )


def _build_vasicek_historical_rates() -> pl.DataFrame:
    """Build a synthetic historical short-rate series for Vasicek calibration.

    Uses a fixed-seed RNG to produce 120 monthly observations that exhibit
    mean-reverting behaviour around 4 %.
    """
    rng = np.random.default_rng(RANDOM_SEED)

    n_obs = 120
    dt = 1.0 / 12.0
    kappa_true = 0.25
    theta_true = 0.04
    sigma_true = 0.012

    rates = np.empty(n_obs, dtype=np.float64)
    rates[0] = 0.035  # starting rate

    e_kdt = np.exp(-kappa_true * dt)
    cond_var = (sigma_true**2 / (2.0 * kappa_true)) * (1.0 - np.exp(-2.0 * kappa_true * dt))
    cond_std = np.sqrt(max(cond_var, 0.0))

    for idx_step in range(1, n_obs):
        cond_mean = theta_true + (rates[idx_step - 1] - theta_true) * e_kdt
        rates[idx_step] = cond_mean + cond_std * rng.standard_normal()

    # Generate enough months; slice to exactly n_obs
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
# Parquet helpers
# ---------------------------------------------------------------------------


def _save_parquet(df: pl.DataFrame, name: str) -> None:
    """Write *df* to ``BASELINES_DIR / {name}.parquet``."""
    path = BASELINES_DIR / f"{name}.parquet"
    df.write_parquet(path)
    print(f"  [OK] {path.name}  ({len(df)} rows)")


def _dict_to_dataframe(data: dict[str, float]) -> pl.DataFrame:
    """Convert a flat ``{key: value}`` dict to a single-row DataFrame."""
    return pl.DataFrame({key: [val] for key, val in data.items()})


# ---------------------------------------------------------------------------
# Constant-model baselines
# ---------------------------------------------------------------------------


def _generate_constant_baselines() -> None:
    """Generate all Parquet baselines for the constant interest-rate model."""
    print("\n=== Constant Interest Rate Model ===")

    # ---- Historical data ----
    hist = _build_constant_historical_rates()
    _save_parquet(hist, "constant_historical_rates")

    # ---- Default (hard-coded rate, empty-fit) ----
    model_default = Interest_Rate_Model_Constant(annual_rate=CONSTANT_DEFAULT_RATE)
    model_default.fit(pl.DataFrame())

    pred_default = model_default.predict(
        n_periods=CONSTANT_N_PERIODS,
        start_date=CONSTANT_START_DATE,
    )
    _save_parquet(pred_default, "constant_predict_default")

    summary_default = model_default.get_summary_statistics(pred_default)
    _save_parquet(summary_default, "constant_summary_default")

    params_default = _dict_to_dataframe(model_default.parameters)
    _save_parquet(params_default, "constant_parameters_default")

    # ---- Fitted to historical data ----
    model_fitted = Interest_Rate_Model_Constant()
    model_fitted.fit(hist)

    pred_fitted = model_fitted.predict(
        n_periods=CONSTANT_N_PERIODS,
        start_date=CONSTANT_START_DATE,
    )
    _save_parquet(pred_fitted, "constant_predict_fitted")

    summary_fitted = model_fitted.get_summary_statistics(pred_fitted)
    _save_parquet(summary_fitted, "constant_summary_fitted")

    params_fitted = _dict_to_dataframe(model_fitted.parameters)
    _save_parquet(params_fitted, "constant_parameters_fitted")

    # ---- Daily and yearly frequency ----
    pred_daily = model_default.predict(
        n_periods=30,
        start_date=CONSTANT_START_DATE,
        freq="1d",
    )
    _save_parquet(pred_daily, "constant_predict_daily")

    pred_yearly = model_default.predict(
        n_periods=5,
        start_date=CONSTANT_START_DATE,
        freq="1y",
    )
    _save_parquet(pred_yearly, "constant_predict_yearly")


# ---------------------------------------------------------------------------
# Vasicek-model baselines
# ---------------------------------------------------------------------------


def _generate_vasicek_baselines() -> None:
    """Generate all Parquet baselines for the standard (Vasicek) model."""
    print("\n=== Standard (Vasicek) Interest Rate Model ===")

    # ---- Historical data ----
    hist = _build_vasicek_historical_rates()
    _save_parquet(hist, "vasicek_historical_rates")

    # ---- Fit model ----
    model = Interest_Rate_Model_Standard(
        kappa=VASICEK_KAPPA,
        theta=VASICEK_THETA,
        sigma=VASICEK_SIGMA,
        n_simulations=VASICEK_N_SIMS,
    )
    model.fit(hist)

    params_fitted = _dict_to_dataframe(model.parameters)
    _save_parquet(params_fitted, "vasicek_parameters_fitted")

    # ---- 12-month prediction ----
    pred_12 = model.predict(
        n_periods=VASICEK_N_PERIODS_SHORT,
        start_date=VASICEK_START_DATE,
        seed=VASICEK_SEED,
    )
    _save_parquet(pred_12, "vasicek_predict_12mo")

    # ---- 60-month prediction ----
    pred_60 = model.predict(
        n_periods=VASICEK_N_PERIODS_LONG,
        start_date=VASICEK_START_DATE,
        seed=VASICEK_SEED,
    )
    _save_parquet(pred_60, "vasicek_predict_60mo")

    # ---- Summary statistics for 12-month ----
    # Build a single-column DataFrame for get_summary_statistics
    mean_col = pred_12["Mean"]
    summary_input = pl.DataFrame({"short_rate": mean_col})
    summary_12 = model.get_summary_statistics(summary_input)
    _save_parquet(summary_12, "vasicek_summary_12mo")

    # ---- Analytical mean and variance across horizons ----
    horizons = ANALYTICAL_HORIZONS
    ana_means = [model.get_analytical_mean(t_years=t_val) for t_val in horizons]
    ana_vars = [model.get_analytical_variance(t_years=t_val) for t_val in horizons]

    df_mean = pl.DataFrame(
        {
            "t_years": horizons,
            "analytical_mean": ana_means,
        },
    )
    _save_parquet(df_mean, "vasicek_analytical_mean")

    df_var = pl.DataFrame(
        {
            "t_years": horizons,
            "analytical_variance": ana_vars,
        },
    )
    _save_parquet(df_var, "vasicek_analytical_variance")

    # ---- Daily-frequency prediction (short horizon) ----
    pred_daily = model.predict(
        n_periods=30,
        start_date=VASICEK_START_DATE,
        freq="1d",
        seed=VASICEK_SEED,
    )
    _save_parquet(pred_daily, "vasicek_predict_daily")


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def main() -> None:
    """Generate all interest-rate regression baselines."""
    print(f"Baselines directory: {BASELINES_DIR}")
    BASELINES_DIR.mkdir(parents=True, exist_ok=True)

    _generate_constant_baselines()
    _generate_vasicek_baselines()

    print("\n=== All interest-rate baselines generated. ===")


if __name__ == "__main__":
    main()
