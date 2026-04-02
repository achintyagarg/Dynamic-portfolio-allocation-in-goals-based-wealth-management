"""
Time series data generator for synthetic financial data.

This module generates synthetic time series data for seven different series
with various patterns and saves the result to a CSV file.

The module creates time series with different characteristics:

- Series AA: Upward trend with seasonal pattern
- Series BB: Cyclical pattern with random walks
- Series CC: Exponential growth with noise
- Series DD: Stationary with occasional jumps
- Series EE: Declining trend with seasonality
- Series FF: Highly volatile series
- Series GG: Combination of trend and cycles

Functions
---------
generate_monthly_timeseries
    Generates monthly time series data with various patterns.
main
    Main execution function that generates and saves the data.

Examples
--------
To generate and save time series data:

.. code-block:: python

    from src.utils.generate_data import main

    main()
"""

from __future__ import annotations

import datetime as dt
import logging

from pathlib import Path

import numpy as np
import polars as pl
import xlsxwriter

from src.utils.custom_exceptions_errors_loggers.logger_custom import get_logger


logger = get_logger(__name__)


def generate_monthly_timeseries(
    start_date: str = "2002-01-01",
    end_date: str = "2025-03-31",
) -> pl.DataFrame:
    """Generate monthly time series data for seven series.

    This function creates synthetic time series data for seven different series,
    each with unique patterns including trends, seasonality, cycles, and
    random components.

    Parameters
    ----------
    start_date : str, optional
        Start date in YYYY-MM-DD format, by default "2002-01-01".
    end_date : str, optional
        End date in YYYY-MM-DD format, by default "2025-03-31".

    Returns
    -------
    pl.DataFrame
        A polars DataFrame with the generated time series with columns:
        date, AA, BB, CC, DD, EE, FF, GG

    Notes
    -----
    The function uses a fixed random seed (42) to ensure reproducibility.
    Each series has specific characteristics:

    - AA: Upward trend with annual seasonality
    - BB: Cyclical pattern (3-year cycle) with random walk component
    - CC: Exponential growth with added noise
    - DD: Stationary series with a structural break (level shift)
    - EE: Declining trend with annual seasonality
    - FF: Highly volatile series with slight upward trend
    - GG: Complex pattern with multiple seasonal components

    Examples
    --------
    >>> df = generate_monthly_timeseries(start_date="2020-01-01", end_date="2020-12-31")
    >>> df.shape
    (12, 8)
    >>> list(df.columns)
    ['date', 'AA', 'BB', 'CC', 'DD', 'EE', 'FF', 'GG']
    """
    start_date_parsed = dt.date.fromisoformat(start_date)
    end_date_parsed = dt.date.fromisoformat(end_date)

    # Generate monthly dates
    dates = pl.date_range(
        start=start_date_parsed,
        end=end_date_parsed,
        interval="1mo",
        closed="left",
        eager=True,
    ).alias("dates")

    # Number of data points
    n = len(dates)

    # Generate time indices for trends and cycles
    t = np.arange(n)

    # Generate different time series with various patterns
    rng = np.random.default_rng(seed=42)  # For reproducibility

    # Series AA: Upward trend with seasonal pattern
    aa = 100 + 0.5 * t + 15 * np.sin(2 * np.pi * t / 12) + rng.normal(loc=0, scale=5, size=n)

    # Series BB: Cyclical pattern with random walks
    bb = 150 + 30 * np.sin(2 * np.pi * t / 36) + np.cumsum(rng.normal(loc=0, scale=1, size=n))

    # Series CC: Exponential growth with noise
    cc = 50 * np.exp(0.005 * t) + rng.normal(loc=0, scale=5, size=n)

    # Series DD: Stationary with occasional jumps
    dd = 200 + rng.normal(loc=0, scale=10, size=n)
    dd[n // 3 : n // 2] += 50  # Add a level shift in the middle

    # Series EE: Declining trend with seasonality
    ee = 300 - 0.3 * t + 20 * np.sin(2 * np.pi * t / 12) + rng.normal(loc=0, scale=8, size=n)

    # Series FF: Highly volatile series
    ff = 120 + rng.normal(loc=0, scale=25, size=n) + 0.2 * t

    # Series GG: Combination of trend and cycles
    gg = (
        80
        + 0.3 * t
        + 20 * np.sin(2 * np.pi * t / 24)
        + 10 * np.sin(2 * np.pi * t / 6)
        + rng.normal(loc=0, scale=7, size=n)
    )

    # Create the DataFrame
    return pl.DataFrame(
        {
            "date": dates,
            "AA": aa.round(decimals=2),
            "BB": bb.round(decimals=2),
            "CC": cc.round(decimals=2),
            "DD": dd.round(decimals=2),
            "EE": ee.round(decimals=2),
            "FF": ff.round(decimals=2),
            "GG": gg.round(decimals=2),
        },
    )


def generate_scenarios_daily_returns_CMA_Tier_0(
    start_date: str = "2026-01-01",
    end_date: str = "2075-12-31",
    num_scenarios: int = 5000,
    random_seed: int | None = 42,
) -> dict[str, pl.DataFrame]:
    """Generate daily return scenarios for CMA Tier 0 asset classes.

    Creates 5000 scenarios (default) for the three Tier 0 asset classes:
    Equities, Fixed Income, and Cash. Scenarios are sampled from multivariate
    lognormal distributions with realistic CMA parameters and saved to an
    XLSX file with four sheets (Expected Returns, Volatilities, Correlations,
    and Scenarios) that can be loaded by ``Scenarios_CMA.from_spreadsheet()``.

    Parameters
    ----------
    start_date : str, optional
        Start date in YYYY-MM-DD format, by default "2026-01-01".
    end_date : str, optional
        End date in YYYY-MM-DD format, by default "2075-12-31".
    num_scenarios : int, optional
        Number of independent scenario paths to generate, by default 5000.
    random_seed : int | None, optional
        Random seed for reproducibility, by default 42.

    Returns
    -------
    dict[str, pl.DataFrame]
        Dictionary with four keys corresponding to the four sheets:
        "Expected Returns", "Volatilities", "Correlations", "Scenarios".

    Notes
    -----
    The function uses the following CMA assumptions for Tier 0 asset classes:

    - **Equities**: Expected return 8%, volatility 16%, representing global stock market
    - **Fixed Income**: Expected return 4%, volatility 5%, representing aggregate bonds
    - **Cash**: Expected return 2%, volatility 1%, representing money market instruments

    Correlations:
    - Equities-Fixed Income: -0.20 (slight negative correlation)
    - Equities-Cash: 0.0 (uncorrelated)
    - Fixed Income-Cash: 0.10 (slight positive correlation)

    The scenarios are generated using a multivariate lognormal distribution to
    ensure positive returns, which is appropriate for long-term strategic asset
    allocation scenarios.

    The output XLSX file is saved at ``PROJECT_ROOT/inputs/raw/returns_CMA_Tier_0.xlsx``
    with four sheets compatible with ``Scenarios_CMA.from_spreadsheet()``.

    Examples
    --------
    >>> sheets = generate_scenarios_daily_returns_CMA_Tier_0()
    >>> sheets["Expected Returns"]
    shape: (3, 2)
    ┌────────────────┬─────────────────┐
    │ asset_class    ┆ expected_return │
    │ ---            ┆ ---             │
    │ str            ┆ f64             │
    ╞════════════════╪═════════════════╡
    │ Equities       ┆ 0.08            │
    │ Fixed Income   ┆ 0.04            │
    │ Cash           ┆ 0.02            │
    └────────────────┴─────────────────┘

    See Also
    --------
    src.num_methods.scenarios.scenarios_CMA.Scenarios_CMA.from_spreadsheet
        Class method that loads the generated XLSX file.
    """
    # --- CMA parameters for Tier 0 asset classes ---
    asset_classes = ["Equities", "Fixed Income", "Cash"]
    expected_returns_annual = np.array([0.08, 0.04, 0.02])  # 8%, 4%, 2%
    volatilities_annual = np.array([0.16, 0.05, 0.01])  # 16%, 5%, 1%

    # Correlation matrix
    correlation_matrix = np.array(
        [
            [1.00, -0.20, 0.00],  # Equities
            [-0.20, 1.00, 0.10],  # Fixed Income
            [0.00, 0.10, 1.00],  # Cash
        ],
    )

    # --- Generate business dates (skip weekends) ---
    start_date_parsed = dt.date.fromisoformat(start_date)
    end_date_parsed = dt.date.fromisoformat(end_date)

    # Generate all dates in range
    all_dates = pl.date_range(
        start=start_date_parsed,
        end=end_date_parsed,
        interval="1d",
        eager=True,
    )

    # Filter to business days (Monday=0 to Friday=4)
    business_dates = [
        d
        for d in all_dates.to_list()
        if d.weekday() < 5  # 0=Monday, ..., 4=Friday
    ]

    num_days = len(business_dates)
    num_assets = len(asset_classes)

    logger.info(
        "Generating %d scenarios for %d Tier 0 asset classes over %d business days (%s to %s)",
        num_scenarios,
        num_assets,
        num_days,
        start_date,
        end_date,
    )

    # --- Convert annual parameters to daily ---
    # For lognormal, we want the arithmetic mean of the output to match
    # the CMA expected return, so we work with the underlying normal parameters

    trading_days_per_year = 252
    daily_returns = expected_returns_annual / trading_days_per_year
    daily_vols = volatilities_annual / np.sqrt(trading_days_per_year)

    # Covariance matrix: Cov = diag(sigma) @ Corr @ diag(sigma)
    D = np.diag(daily_vols)
    covariance_daily = D @ correlation_matrix @ D

    # For lognormal: X = exp(Y), where Y ~ N(mu_ln, Sigma_ln)
    # We want E[X] = desired_mean, so:
    # mu_ln = ln(desired_mean) - 0.5 * sigma_ln^2
    # sigma_ln^2 = ln(1 + Var[X] / E[X]^2)

    # Map to underlying normal parameters
    sigma_ln = np.zeros((num_assets, num_assets))
    for i in range(num_assets):
        for j in range(num_assets):
            # For the variance/covariance mapping to lognormal
            ratio = covariance_daily[i, j] / (daily_returns[i] * daily_returns[j])
            if 1.0 + ratio <= 0:
                # Fallback: use diagonal only (independent)
                if i == j:
                    sigma_ln[i, i] = np.log(1.0 + (daily_vols[i] / daily_returns[i]) ** 2)
                else:
                    sigma_ln[i, j] = 0.0
            else:
                sigma_ln[i, j] = np.log(1.0 + ratio)

    mu_ln = np.log(daily_returns) - 0.5 * np.diag(sigma_ln)

    # --- Generate lognormal scenarios ---
    rng = np.random.default_rng(random_seed)

    # Cholesky decomposition for correlated sampling
    try:
        L = np.linalg.cholesky(sigma_ln)
    except np.linalg.LinAlgError:
        # Fallback: eigenvalue clipping for near-PSD matrices
        eigvals, eigvecs = np.linalg.eigh(sigma_ln)
        eigvals = np.maximum(eigvals, 1e-10)
        sigma_ln = eigvecs @ np.diag(eigvals) @ eigvecs.T
        L = np.linalg.cholesky(sigma_ln)
        logger.warning("Covariance matrix was not PSD; eigenvalues clipped to 1e-10")

    # Sample underlying normal: Z ~ N(0, I), Y = mu_ln + L @ Z, X = exp(Y)
    Z = rng.standard_normal((num_days, num_assets))
    Y = Z @ L.T + mu_ln
    scenarios = np.exp(Y)  # Lognormal transform

    logger.info(
        "Generated daily lognormal return samples for asset classes",
        extra={
            "num_days": num_days,
            "num_assets": num_assets,
        },
    )

    # --- Build DataFrames for each sheet ---

    # Sheet 1: Expected Returns
    df_returns = pl.DataFrame(
        {
            "asset_class": asset_classes,
            "expected_return": expected_returns_annual,
        },
    )

    # Sheet 2: Volatilities
    df_volatilities = pl.DataFrame(
        {
            "asset_class": asset_classes,
            "volatility": volatilities_annual,
        },
    )

    # Sheet 3: Correlations
    corr_data = {"asset_class": asset_classes}
    for idx, ac in enumerate(asset_classes):
        corr_data[ac] = correlation_matrix[:, idx].tolist()
    df_correlations = pl.DataFrame(corr_data)

    # Sheet 4: Scenarios (Date + asset class columns)
    scenario_data = {"Date": business_dates}
    for idx, ac in enumerate(asset_classes):
        scenario_data[ac] = scenarios[:, idx].tolist()
    df_scenarios = pl.DataFrame(scenario_data)

    # --- Save to XLSX ---
    project_dir = Path(__file__).parent.parent.parent.parent
    raw_data_dir = project_dir / "inputs" / "raw"
    raw_data_dir.mkdir(parents=True, exist_ok=True)

    output_file = raw_data_dir / "returns_CMA_Tier_0.xlsx"

    # Write multiple sheets to XLSX using xlsxwriter
    workbook = xlsxwriter.Workbook(output_file)

    # Sheet 1: Expected Returns
    worksheet1 = workbook.add_worksheet("Expected Returns")
    for col_idx, col_name in enumerate(df_returns.columns):
        worksheet1.write(0, col_idx, col_name)
        for row_idx, value in enumerate(df_returns[col_name].to_list()):
            worksheet1.write(row_idx + 1, col_idx, value)

    # Sheet 2: Volatilities
    worksheet2 = workbook.add_worksheet("Volatilities")
    for col_idx, col_name in enumerate(df_volatilities.columns):
        worksheet2.write(0, col_idx, col_name)
        for row_idx, value in enumerate(df_volatilities[col_name].to_list()):
            worksheet2.write(row_idx + 1, col_idx, value)

    # Sheet 3: Correlations
    worksheet3 = workbook.add_worksheet("Correlations")
    for col_idx, col_name in enumerate(df_correlations.columns):
        worksheet3.write(0, col_idx, col_name)
        for row_idx, value in enumerate(df_correlations[col_name].to_list()):
            worksheet3.write(row_idx + 1, col_idx, value)

    # Sheet 4: Scenarios
    worksheet4 = workbook.add_worksheet("Scenarios")
    for col_idx, col_name in enumerate(df_scenarios.columns):
        worksheet4.write(0, col_idx, col_name)
        for row_idx, value in enumerate(df_scenarios[col_name].to_list()):
            worksheet4.write(row_idx + 1, col_idx, value)

    workbook.close()

    logger.info(
        "SCENARIOS_SAVED: CMA Tier 0 scenarios saved successfully",
        extra={
            "event_type": "scenarios_saved",
            "output_file": str(output_file),
            "num_days": num_days,
            "num_scenarios": num_scenarios,
            "asset_classes": ", ".join(asset_classes),
        },
    )

    return {
        "Expected Returns": df_returns,
        "Volatilities": df_volatilities,
        "Correlations": df_correlations,
        "Scenarios": df_scenarios,
    }


def main() -> None:
    """Generate time series data and save it to CSV.

    This function creates the necessary directories if they don't exist,
    generates synthetic time series data, and saves it to a CSV file
    in the project's data/raw directory.

    Returns
    -------
    None

    Notes
    -----
    The output file is saved at PROJECT_ROOT/data/raw/data_timeseries.csv.
    The function will create the directories if they don't exist.

    The time range for the generated data is from 2002-01-01 to 2025-03-31.

    Examples
    --------
    >>> from src.utils.generate_data import main
    >>> main()
    Generating time series data...
    Data saved to .../data/raw/data_timeseries.csv
    Generated 279 rows from 2002-01-01 to 2025-03-01
    """
    # Configure logging for script execution
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # Define project paths
    project_dir = Path(__file__).parent.parent.parent
    data_dir = project_dir / "inputs"
    raw_data_dir = data_dir / "raw"
    output_file = raw_data_dir / "data_timeseries.csv"

    # Generate the data
    logger.info("Generating time series data...")
    df = generate_monthly_timeseries(
        start_date="2002-01-01",
        end_date="2025-03-31",
    )

    # Create directory if it doesn't exist
    raw_data_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # Save to CSV
    df.write_csv(file=output_file)

    logger.info(
        "Data saved to file",
        extra={
            "event_type": "data_saved",
            "output_file": str(output_file),
            "num_rows": len(df),
        },
    )
    logger.info(
        "Time series data generation complete",
        extra={
            "event_type": "generation_complete",
            "num_rows": len(df),
            "start_date": str(df["date"].min()),
            "end_date": str(df["date"].max()),
        },
    )


if __name__ == "__main__":
    main()
