"""Utility functions for results tab data processing.

Provides data transformation and processing functions specifically
for results visualization and reporting operations.
"""

from __future__ import annotations

from typing import Any

import pandas as pd
import polars as pl


def process_data_for_plot_tab_results(
    data_frame: Any,
    date_column: Any = "date",
    max_points: Any = 5000,
) -> Any:
    """Process data for plotting with downsampling if needed."""
    if data_frame is None or data_frame.is_empty():
        return data_frame

    # If data is within limits, return as-is
    if data_frame.height <= max_points:
        return data_frame

    # Calculate downsampling factor
    downsample_factor = max(1, data_frame.height // max_points)

    # Sort by date and downsample
    sorted_data = data_frame.sort(date_column)
    return sorted_data.gather_every(downsample_factor)


def normalize_data_tab_results(data_frame: Any, method: Any = "none") -> Any:
    """Normalize data using specified method."""
    if method == "none" or data_frame is None or data_frame.is_empty():
        return data_frame

    # Get numeric columns (exclude date)
    numeric_columns = [col for col in data_frame.columns if col != "date"]

    if method == "min_max":
        # Min-Max normalization
        for col in numeric_columns:
            min_val = data_frame.select(pl.col(col).min()).item()
            max_val = data_frame.select(pl.col(col).max()).item()
            if max_val > min_val:
                data_frame = data_frame.with_columns(
                    ((pl.col(col) - min_val) / (max_val - min_val)).alias(col),
                )

    elif method == "z_score":
        # Z-score normalization
        for col in numeric_columns:
            mean_val = data_frame.select(pl.col(col).mean()).item()
            std_val = data_frame.select(pl.col(col).std()).item()
            if std_val > 0:
                data_frame = data_frame.with_columns(
                    ((pl.col(col) - mean_val) / std_val).alias(col),
                )

    return data_frame


def transform_data_tab_results(
    data_frame: Any,
    transformation: Any = "none",
    baseline_date: Any = None,
) -> Any:
    """Transform data based on specified transformation type."""
    if transformation == "none" or data_frame is None or data_frame.is_empty():
        return data_frame

    # Get series columns (exclude date)
    series_columns = [col for col in data_frame.columns if col != "date"]

    if transformation == "percent_change":
        # Calculate percent change from first value
        for col in series_columns:
            first_val = data_frame.select(pl.col(col).first()).item()
            if first_val != 0:
                data_frame = data_frame.with_columns(
                    ((pl.col(col) / first_val - 1) * 100).alias(f"{col}_pct"),
                )

    elif transformation == "cumulative":
        # Calculate cumulative sum
        for col in series_columns:
            data_frame = data_frame.with_columns(
                pl.col(col).cum_sum().alias(f"{col}_cum"),
            )

    elif transformation == "comparative" and baseline_date:
        # Calculate relative change from baseline date
        baseline_dt = pd.to_datetime(baseline_date)

        # Find the closest date to baseline
        data_pd = data_frame.to_pandas()
        data_pd["date"] = pd.to_datetime(data_pd["date"])
        data_pd["date_diff"] = abs(data_pd["date"] - baseline_dt)
        baseline_idx = data_pd["date_diff"].idxmin()

        for col in series_columns:
            baseline_val = data_pd.loc[baseline_idx, col]
            if baseline_val != 0:
                data_pd[f"{col}_rel"] = (data_pd[col] / baseline_val - 1) * 100

        # Convert back to polars
        data_frame = pl.from_pandas(data_pd.drop("date_diff", axis=1))

    return data_frame
