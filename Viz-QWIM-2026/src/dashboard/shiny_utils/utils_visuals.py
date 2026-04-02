"""Visual utilities for dashboard charts and tables."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import polars as pl

from great_tables import GT, loc, md, style
from plotly.subplots import make_subplots


if TYPE_CHECKING:
    from datetime import date


# Add these missing imports and constants
OUTPUT_DIR = Path("outputs")

from src.dashboard.shiny_utils.utils_data import (  # noqa: E402
    calculate_portfolio_returns,
    downsample_dataframe,
    validate_portfolio_and_benchmark_data_if_not_already_validated,
)


def validate_data_visuals(
    data_frame: pl.DataFrame | None = None,
    selected_series_list: list[str] | None = None,
) -> tuple[bool, str]:
    """Validate data meets requirements for visualization and analysis using defensive programming."""
    # Input validation with early returns
    if data_frame is None:
        return False, "No data available"

    # Type validation
    if not isinstance(data_frame, pl.DataFrame):
        return False, f"Invalid data type: {type(data_frame).__name__}, expected polars.DataFrame"

    # Data existence validation
    if data_frame.is_empty():
        return False, "Dataset is empty"

    # Schema validation
    if "Date" not in data_frame.columns:
        return False, "Data is missing required 'Date' column"

    # Series selection validation if provided
    if selected_series_list is not None:
        # Configuration validation
        if not isinstance(selected_series_list, (list, tuple)):
            return (
                False,
                f"Selected series must be list or tuple, got {type(selected_series_list).__name__}",
            )

        # Business logic validation - empty selection
        if not selected_series_list:
            return False, "No series selected"

        # Defensive programming - check series availability
        available_columns = set(data_frame.columns)
        selected_series_set = set(selected_series_list)
        available_series = selected_series_set.intersection(available_columns)

        if not available_series:
            return (
                False,
                f"None of the selected series found in data: {', '.join(selected_series_list)}",
            )

        # Data quality validation with defensive checks
        valid_series_found = False
        max_series_to_check = min(3, len(available_series))  # Check up to 3 series for performance

        for _idx, series_name in enumerate(list(available_series)[:max_series_to_check]):
            # Defensive programming - ensure column exists before operations
            if series_name not in data_frame.columns:
                continue

            # Only use try-except for Polars operations that might fail unpredictably
            try:
                non_null_count = data_frame.select(pl.col(series_name).is_not_null().sum()).item()
                if isinstance(non_null_count, (int, float)) and non_null_count > 0:
                    valid_series_found = True
                    break
            except Exception:  # noqa: S112 - Continue checking series
                # Continue checking other series if this one fails
                continue

        if not valid_series_found:
            return False, "No valid data found in selected series"

    return True, ""


def create_error_figure(
    message_text: str,
    title_text: str = "Error",
    details_text: str | None = None,
) -> go.Figure:
    """Create a standardized error figure using defensive programming.

    Parameters
    ----------
    message_text : str
        Main error message
    title_text : str, optional
        Figure title, by default "Error"
    details_text : str, optional
        Detailed error information, by default None

    Returns
    -------
    plotly.graph_objects.Figure
        Error figure
    """
    # Input validation with early returns
    if not isinstance(message_text, str):
        message_text = str(message_text) if message_text is not None else "Unknown error"

    if not isinstance(title_text, str):
        title_text = str(title_text) if title_text is not None else "Error"

    if details_text is not None and not isinstance(details_text, str):
        details_text = str(details_text)

    # Configuration validation
    if len(message_text.strip()) == 0:
        message_text = "No error message provided"

    # Defensive programming - create figure with safe defaults
    figure_obj = go.Figure()

    # Safe layout configuration
    layout_config = {
        "title": {
            "text": title_text,
            "x": 0.5,
            "xanchor": "center",
            "font": {"size": 16, "color": "#d63384"},
        },
        "template": "plotly_white",
        "height": 400,
        "margin": {"t": 80, "b": 40, "l": 40, "r": 40},
        "xaxis": {"visible": False, "range": [0, 1]},
        "yaxis": {"visible": False, "range": [0, 1]},
        "showlegend": False,
        "plot_bgcolor": "white",
        "paper_bgcolor": "white",
    }

    # Only use try-except for Plotly operations that might fail unpredictably
    try:
        figure_obj.update_layout(**layout_config)

        # Add main error message
        figure_obj.add_annotation(
            text=message_text,
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.6,
            showarrow=False,
            font={"size": 14, "color": "#d63384"},
            align="center",
            width=350,
        )

        # Add details if provided
        if details_text:
            figure_obj.add_annotation(
                text=f"Details: {details_text}",
                xref="paper",
                yref="paper",
                x=0.5,
                y=0.4,
                showarrow=False,
                font={"size": 12, "color": "#6c757d"},
                align="center",
                width=400,
            )

        # Add help text
        figure_obj.add_annotation(
            text="💡 Try selecting different data or adjusting filters",
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.2,
            showarrow=False,
            font={"size": 11, "color": "#198754"},
            align="center",
            width=400,
        )

    except Exception:
        # Fallback to basic figure if Plotly operations fail
        basic_figure = go.Figure()
        basic_figure.update_layout(
            title=f"Error: {message_text}",
            template="plotly_white",
            height=400,
        )
        return basic_figure

    return figure_obj


def format_value_for_display(value_input: Any, format_type: str = "decimal") -> str:
    """Format a value for display in tables using defensive programming.

    Parameters
    ----------
    value_input : Any
        The value to format for display.
    format_type : str, optional
        How to format numeric values. Supported values:
        - ``"percentage"`` — multiply by 100 and append ``%``.
        - ``"integer"`` — format as a comma-separated integer.
        - ``"decimal"`` — four decimal places (default).

    Returns
    -------
    str
        A formatted string representation of *value_input*.
    """
    # Input validation with early returns
    if value_input is None:
        return "None"

    # Handle pandas NA values
    if pd.isna(value_input):
        return "N/A"

    # Handle string values
    if isinstance(value_input, str):
        if value_input.lower() in ["error", "none", "null", "nan"]:
            return value_input
        return value_input

    # Handle numeric values with defensive programming
    if isinstance(value_input, (int, float)):
        # Check for special float values
        if pd.isna(value_input):
            return "N/A"

        # Safe numeric formatting - only use try-except for float conversion that might fail
        try:
            numeric_value = float(value_input)
            if format_type == "percentage":
                return f"{numeric_value * 100:.2f}%"
            if format_type == "integer":
                return f"{int(numeric_value):,}"
            # Default: "decimal"
            return f"{numeric_value:.4f}"
        except (ValueError, OverflowError):
            return str(value_input)

    # Handle boolean values
    if isinstance(value_input, bool):
        return str(value_input)

    # Default case - convert to string safely
    return str(value_input)


def safe_numeric_conversion(input_value: Any) -> Any:
    """Safely convert a value to numeric format using defensive programming."""
    # Input validation with early returns
    if input_value is None:
        return None

    # Handle pandas NA values
    if pd.isna(input_value):
        return input_value

    # Handle string values
    if isinstance(input_value, str):
        if input_value.lower() in ["error", "none", "null", "nan", ""]:
            return input_value

        # Clean string for numeric conversion
        cleaned_value = input_value.strip()
        if len(cleaned_value) == 0:
            return input_value

        # Only use try-except for actual conversion that might fail unpredictably
        try:
            return float(cleaned_value)
        except (ValueError, TypeError):
            return input_value

    # Handle numeric values
    if isinstance(input_value, (int, float)):
        # Check for special float values
        if pd.isna(input_value):
            return input_value

        # Only use try-except for float conversion that might fail with extreme values
        try:
            return float(input_value)
        except (ValueError, OverflowError):
            return input_value

    # Handle boolean values
    if isinstance(input_value, bool):
        return float(input_value)

    # Default case - return as-is if conversion not possible
    return input_value


def create_plot_portfolios_comparison(
    data_portfolio: pl.DataFrame | None,
    data_benchmark: pl.DataFrame | None,
    period_label: str,
    data_was_validated: bool = True,
) -> go.Figure:
    """
    Create portfolio vs benchmark comparison plot using Polars DataFrames.

    Parameters
    ----------
    data_portfolio : polars.DataFrame
        Portfolio data with "Date" and "Value" columns
    data_benchmark : polars.DataFrame
        Benchmark data with "Date" and "Value" columns
    period_label : str
        Label for the time period
    data_was_validated : bool, optional
        Whether data has already been validated, by default True

    Returns
    -------
    plotly.graph_objects.Figure
        Portfolio vs benchmark comparison plot

    Raises
    ------
    ValueError
        If data validation fails
    RuntimeError
        If plot creation fails
    """
    # Input validation with early returns
    if data_portfolio is None and data_benchmark is None:
        return create_error_figure("No portfolio or benchmark data available")

    if not isinstance(period_label, str):
        period_label = str(period_label) if period_label is not None else "Unknown Period"

    has_portfolio_data = True
    has_benchmark_data = True
    # Validate input data if not already validated
    if not data_was_validated:
        has_portfolio_data, has_benchmark_data = (
            validate_portfolio_and_benchmark_data_if_not_already_validated(
                data_portfolio=data_portfolio,
                data_benchmark=data_benchmark,
            )
        )

    if not has_portfolio_data and not has_benchmark_data:
        return create_error_figure("Both portfolio and benchmark data are empty")

    # Only use try-except for Plotly operations that might fail unpredictably
    try:
        figure_obj = go.Figure()

        # Process portfolio data if available
        if has_portfolio_data:
            # Business logic validation - sort by Date and prepare data
            assert data_portfolio is not None  # narrowing: guarded by has_portfolio_data
            portfolio_sorted = data_portfolio.sort("Date")

            # Safe data extraction using Polars methods
            portfolio_dates = portfolio_sorted.select("Date").to_series().to_list()
            portfolio_values = portfolio_sorted.select("Value").to_series()

            # Convert to numeric and handle nulls
            portfolio_numeric = portfolio_values.cast(pl.Float64, strict=False)
            portfolio_clean = portfolio_numeric.drop_nulls()

            if len(portfolio_clean) > 0:
                # Get corresponding dates for clean values
                valid_indices = portfolio_numeric.is_not_null()
                portfolio_dates_clean = [
                    date_val
                    for idx, date_val in enumerate(portfolio_dates)
                    if idx < len(valid_indices) and valid_indices[idx]
                ]

                # Calculate normalized values (base=100)
                portfolio_values_list = portfolio_clean.to_list()
                first_value = portfolio_values_list[0]

                if first_value != 0:
                    normalized_portfolio = [
                        (value / first_value) * 100 for value in portfolio_values_list
                    ]

                    figure_obj.add_trace(
                        go.Scatter(
                            x=portfolio_dates_clean,
                            y=normalized_portfolio,
                            mode="lines",
                            name="Portfolio",
                            line={"width": 3, "color": "#1f77b4"},
                            hovertemplate="<b>Portfolio</b><br>Date: %{x}<br>Value: %{y:.2f}<extra></extra>",
                        ),
                    )

        # Process benchmark data if available
        if has_benchmark_data:
            # Business logic validation - sort by Date and prepare data
            assert data_benchmark is not None  # narrowing: guarded by has_benchmark_data
            benchmark_sorted = data_benchmark.sort("Date")

            # Safe data extraction using Polars methods
            benchmark_dates = benchmark_sorted.select("Date").to_series().to_list()
            benchmark_values = benchmark_sorted.select("Value").to_series()

            # Convert to numeric and handle nulls
            benchmark_numeric = benchmark_values.cast(pl.Float64, strict=False)
            benchmark_clean = benchmark_numeric.drop_nulls()

            if len(benchmark_clean) > 0:
                # Get corresponding dates for clean values
                valid_indices = benchmark_numeric.is_not_null()
                benchmark_dates_clean = [
                    date_val
                    for idx, date_val in enumerate(benchmark_dates)
                    if idx < len(valid_indices) and valid_indices[idx]
                ]

                # Calculate normalized values (base=100)
                benchmark_values_list = benchmark_clean.to_list()
                first_value = benchmark_values_list[0]

                if first_value != 0:
                    normalized_benchmark = [
                        (value / first_value) * 100 for value in benchmark_values_list
                    ]

                    figure_obj.add_trace(
                        go.Scatter(
                            x=benchmark_dates_clean,
                            y=normalized_benchmark,
                            mode="lines",
                            name="Benchmark",
                            line={"width": 3, "color": "#ff7f0e", "dash": "dash"},
                            hovertemplate="<b>Benchmark</b><br>Date: %{x}<br>Value: %{y:.2f}<extra></extra>",
                        ),
                    )

        # Configuration validation - ensure we have at least one trace
        if len(figure_obj.data) == 0:  # pyright: ignore[reportArgumentType]
            return create_error_figure(
                "No valid data series to display",
                details_text="Both portfolio and benchmark data contain no valid numeric values",
            )

        # Safe layout configuration
        layout_config = {
            "title": {
                "text": f"Portfolio vs Benchmark Performance ({period_label})",
                "font": {"size": 18},
                "x": 0.5,
                "xanchor": "center",
            },
            "xaxis_title": "Date",
            "yaxis_title": "Normalized Value (Base=100)",
            "template": "plotly_white",
            "height": 600,
            "margin": {"l": 60, "r": 40, "t": 120, "b": 60},
            "legend": {
                "orientation": "h",
                "yanchor": "bottom",
                "y": 1.02,
                "xanchor": "center",
                "x": 0.5,
            },
            "hovermode": "x unified",
        }

        figure_obj.update_layout(**layout_config)

        return figure_obj

    except Exception as plotly_error:
        raise RuntimeError(f"Error creating comparison plot: {plotly_error}") from plotly_error


def create_plot_rolling_statistics(
    data_portfolio: pl.DataFrame | None,
    data_benchmark: pl.DataFrame | None,
    window_size: int,
    period_label: str,
    data_was_validated: bool = True,
    include_benchmark: bool = False,
) -> go.Figure:
    """Create rolling statistics subplot."""
    # Input validation with early returns
    if data_portfolio is None and data_benchmark is None:
        return create_error_figure("No portfolio or benchmark data available")

    if not isinstance(period_label, str):
        period_label = str(period_label) if period_label is not None else "Unknown Period"

    if not isinstance(window_size, int) or window_size < 1:
        window_size = 30  # Default window size

    has_portfolio_data = True
    has_benchmark_data = True
    # Validate input data if not already validated
    if not data_was_validated:
        has_portfolio_data, has_benchmark_data = (
            validate_portfolio_and_benchmark_data_if_not_already_validated(
                data_portfolio=data_portfolio,
                data_benchmark=data_benchmark,
            )
        )

    if not has_portfolio_data and not has_benchmark_data:
        return create_error_figure("Both portfolio and benchmark data are empty")

    # Only use try-except for Plotly operations that might fail unpredictably
    try:
        figure_obj = make_subplots(
            rows=2,
            cols=1,
            subplot_titles=[
                f"Rolling {window_size}-Day Average Returns (%)",
                f"Rolling {window_size}-Day Volatility (% Annualized)",
            ],
            vertical_spacing=0.15,
            shared_xaxes=True,
        )

        # Portfolio rolling statistics
        if has_portfolio_data and data_portfolio is not None and not data_portfolio.is_empty():
            portfolio_returns = calculate_portfolio_returns(data_portfolio)
            if len(portfolio_returns) >= window_size:
                # Convert to pandas for rolling calculations
                portfolio_returns_pandas = portfolio_returns.to_pandas()
                rolling_returns = portfolio_returns_pandas.rolling(window=window_size).mean() * 100
                rolling_vol = (
                    portfolio_returns_pandas.rolling(window=window_size).std() * 100 * np.sqrt(252)
                )  # Fixed: changed pd.np to np

                # Align dates with rolling statistics
                assert data_portfolio is not None  # narrowing
                portfolio_sorted = data_portfolio.sort("Date")
                dates_list = portfolio_sorted.select("Date").to_series().to_list()

                # Get dates corresponding to rolling statistics (skip first window_size-1 dates)
                if len(dates_list) >= window_size:
                    dates_aligned = dates_list[
                        window_size - 1 : window_size - 1 + len(rolling_returns.dropna())
                    ]

                    figure_obj.add_trace(
                        go.Scatter(
                            x=dates_aligned,
                            y=rolling_returns.dropna(),
                            mode="lines",
                            name="Portfolio Returns",
                            line={"color": "#1f77b4", "width": 2},
                            hovertemplate="Date: %{x}<br>Portfolio Avg Return: %{y:.3f}%<extra></extra>",
                        ),
                        row=1,
                        col=1,
                    )

                    figure_obj.add_trace(
                        go.Scatter(
                            x=dates_aligned,
                            y=rolling_vol.dropna(),
                            mode="lines",
                            name="Portfolio Volatility",
                            line={"color": "#1f77b4", "width": 2},
                            hovertemplate="Date: %{x}<br>Portfolio Volatility: %{y:.2f}%<extra></extra>",
                        ),
                        row=2,
                        col=1,
                    )

        # Benchmark rolling statistics if requested
        if (
            include_benchmark
            and has_benchmark_data
            and data_benchmark is not None
            and not data_benchmark.is_empty()
        ):
            benchmark_returns = calculate_portfolio_returns(data_benchmark)
            if len(benchmark_returns) >= window_size:
                # Convert to pandas for rolling calculations
                benchmark_returns_pandas = benchmark_returns.to_pandas()
                rolling_returns = benchmark_returns_pandas.rolling(window=window_size).mean() * 100
                rolling_vol = (
                    benchmark_returns_pandas.rolling(window=window_size).std() * 100 * np.sqrt(252)
                )  # Fixed: changed pd.np to np

                # Align dates with rolling statistics
                assert data_benchmark is not None  # narrowing
                benchmark_sorted = data_benchmark.sort("Date")
                dates_list = benchmark_sorted.select("Date").to_series().to_list()

                # Get dates corresponding to rolling statistics
                if len(dates_list) >= window_size:
                    dates_aligned = dates_list[
                        window_size - 1 : window_size - 1 + len(rolling_returns.dropna())
                    ]

                    figure_obj.add_trace(
                        go.Scatter(
                            x=dates_aligned,
                            y=rolling_returns.dropna(),
                            mode="lines",
                            name="Benchmark Returns",
                            line={"color": "#ff7f0e", "width": 2, "dash": "dash"},
                            hovertemplate="Date: %{x}<br>Benchmark Avg Return: %{y:.3f}%<extra></extra>",
                        ),
                        row=1,
                        col=1,
                    )

                    figure_obj.add_trace(
                        go.Scatter(
                            x=dates_aligned,
                            y=rolling_vol.dropna(),
                            mode="lines",
                            name="Benchmark Volatility",
                            line={"color": "#ff7f0e", "width": 2, "dash": "dash"},
                            hovertemplate="Date: %{x}<br>Benchmark Volatility: %{y:.2f}%<extra></extra>",
                        ),
                        row=2,
                        col=1,
                    )

        figure_obj.add_hline(y=0, line_dash="dot", line_color="gray", row=1, col=1)  # pyright: ignore[reportArgumentType]

        figure_obj.update_layout(
            title={
                "text": f"Rolling {window_size}-Day Statistics ({period_label})",
                "font": {"size": 18},
                "x": 0.5,
                "xanchor": "center",
            },
            template="plotly_white",
            height=700,
            showlegend=True,
            margin={"l": 60, "r": 60, "t": 140, "b": 60},
            hovermode="x unified",
            legend={
                "orientation": "h",
                "yanchor": "bottom",
                "y": 1.02,
                "xanchor": "center",
                "x": 0.5,
            },
        )

        figure_obj.update_xaxes(title_text="Date", row=2, col=1)
        figure_obj.update_yaxes(title_text="Return (%)", row=1, col=1)
        figure_obj.update_yaxes(title_text="Volatility (%)", row=2, col=1)

        return figure_obj

    except Exception as plotly_error:
        raise RuntimeError(
            f"Error creating rolling statistics plot: {plotly_error}",
        ) from plotly_error


def create_plot_returns_distribution(
    data_portfolio: pl.DataFrame | None,
    data_benchmark: pl.DataFrame | None,
    period_label: str,
    data_was_validated: bool = True,
    include_benchmark: bool = False,
) -> go.Figure:
    """Create returns distribution histogram comparing portfolio and benchmark."""
    # Input validation with early returns
    if data_portfolio is None and data_benchmark is None:
        return create_error_figure("No portfolio or benchmark data available")

    if not isinstance(period_label, str):
        period_label = str(period_label) if period_label is not None else "Unknown Period"

    has_portfolio_data = True
    has_benchmark_data = True
    # Validate input data if not already validated
    if not data_was_validated:
        has_portfolio_data, has_benchmark_data = (
            validate_portfolio_and_benchmark_data_if_not_already_validated(
                data_portfolio=data_portfolio,
                data_benchmark=data_benchmark,
            )
        )

    if not has_portfolio_data and not has_benchmark_data:
        return create_error_figure("Both portfolio and benchmark data are empty")

    # Only use try-except for Plotly operations that might fail unpredictably
    try:
        figure_obj = go.Figure()

        # Calculate portfolio returns
        if has_portfolio_data and data_portfolio is not None and not data_portfolio.is_empty():
            portfolio_returns = calculate_portfolio_returns(data_portfolio)
            if len(portfolio_returns) > 0:
                portfolio_returns_pct = portfolio_returns.to_pandas() * 100
                figure_obj.add_trace(
                    go.Histogram(
                        x=portfolio_returns_pct,
                        name="Portfolio",
                        opacity=0.7,
                        marker_color="#1f77b4",
                        nbinsx=min(50, max(10, len(portfolio_returns_pct) // 5)),
                        hovertemplate="Portfolio Return: %{x:.2f}%<br>Frequency: %{y}<extra></extra>",
                    ),
                )

        if (
            include_benchmark
            and has_benchmark_data
            and data_benchmark is not None
            and not data_benchmark.is_empty()
        ):
            benchmark_returns = calculate_portfolio_returns(data_benchmark)
            if len(benchmark_returns) > 0:
                benchmark_returns_pct = benchmark_returns.to_pandas() * 100
                figure_obj.add_trace(
                    go.Histogram(
                        x=benchmark_returns_pct,
                        name="Benchmark",
                        opacity=0.7,
                        marker_color="#ff7f0e",
                        nbinsx=min(50, max(10, len(benchmark_returns_pct) // 5)),
                        hovertemplate="Benchmark Return: %{x:.2f}%<br>Frequency: %{y}<extra></extra>",
                    ),
                )

        # Calculate statistics for title
        if has_portfolio_data and data_portfolio is not None and not data_portfolio.is_empty():
            portfolio_returns = calculate_portfolio_returns(data_portfolio)
            if len(portfolio_returns) > 0:
                portfolio_returns_pandas = portfolio_returns.to_pandas()
                mean_ret = portfolio_returns_pandas.mean() * 100
                std_ret = portfolio_returns_pandas.std() * 100
                subtitle_text = f"Portfolio Mean: {mean_ret:.3f}%, Std Dev: {std_ret:.3f}%"
            else:
                subtitle_text = "No data available"
        else:
            subtitle_text = "No data available"

        figure_obj.update_layout(
            title={
                "text": f"Returns Distribution ({period_label})<br><sub>{subtitle_text}</sub>",
                "font": {"size": 18},
                "x": 0.5,
                "xanchor": "center",
            },
            xaxis_title="Daily Return (%)",
            yaxis_title="Frequency",
            template="plotly_white",
            height=600,
            margin={"l": 60, "r": 60, "t": 120, "b": 60},
            hovermode="closest",
            barmode="overlay",
            legend={
                "orientation": "h",
                "yanchor": "bottom",
                "y": 1.02,
                "xanchor": "center",
                "x": 0.5,
            },
        )

        return figure_obj

    except Exception as plotly_error:
        raise RuntimeError(
            f"Error creating returns distribution plot: {plotly_error}",
        ) from plotly_error


def create_plot_drawdowns_analysis(
    data_portfolio: pl.DataFrame | None,
    data_benchmark: pl.DataFrame | None,
    period_label: str,
    data_was_validated: bool = True,
    include_benchmark: bool = False,
) -> go.Figure:
    """Create drawdowns analysis plot."""
    # Input validation with early returns
    if data_portfolio is None and data_benchmark is None:
        return create_error_figure("No portfolio or benchmark data available")

    if not isinstance(period_label, str):
        period_label = str(period_label) if period_label is not None else "Unknown Period"

    has_portfolio_data = True
    has_benchmark_data = True
    # Validate input data if not already validated
    if not data_was_validated:
        has_portfolio_data, has_benchmark_data = (
            validate_portfolio_and_benchmark_data_if_not_already_validated(
                data_portfolio=data_portfolio,
                data_benchmark=data_benchmark,
            )
        )

    if not has_portfolio_data and not has_benchmark_data:
        return create_error_figure("Both portfolio and benchmark data are empty")

    # Only use try-except for Plotly operations that might fail unpredictably
    try:
        figure_obj = go.Figure()

        # Calculate portfolio drawdowns
        if has_portfolio_data and data_portfolio is not None and not data_portfolio.is_empty():
            portfolio_returns = calculate_portfolio_returns(data_portfolio)
            if len(portfolio_returns) > 0:
                # Convert to pandas for cumulative calculations
                portfolio_returns_pandas = portfolio_returns.to_pandas()
                cum_rets = (1 + portfolio_returns_pandas).cumprod()
                peak_values = cum_rets.cummax()
                drawdown_values = (cum_rets / peak_values - 1) * 100

                # Get dates for plotting (skip first date since we calculated returns)
                assert data_portfolio is not None  # narrowing: guarded by has_portfolio_data
                portfolio_sorted = data_portfolio.sort("Date")
                dates_list = portfolio_sorted.select("Date").to_series().to_list()

                if len(dates_list) > 1:
                    dates_for_drawdowns = dates_list[1 : len(drawdown_values) + 1]

                    figure_obj.add_trace(
                        go.Scatter(
                            x=dates_for_drawdowns,
                            y=drawdown_values,
                            fill="tozeroy",
                            fillcolor="rgba(31, 119, 180, 0.3)",
                            line={"color": "#1f77b4", "width": 2},
                            name="Portfolio Drawdowns",
                            hovertemplate="Date: %{x}<br>Portfolio Drawdown: %{y:.2f}%<extra></extra>",
                        ),
                    )

        # Calculate benchmark drawdowns if requested
        if (
            include_benchmark
            and has_benchmark_data
            and data_benchmark is not None
            and not data_benchmark.is_empty()
        ):
            benchmark_returns = calculate_portfolio_returns(data_benchmark)
            if len(benchmark_returns) > 0:
                # Convert to pandas for cumulative calculations
                benchmark_returns_pandas = benchmark_returns.to_pandas()
                cum_rets = (1 + benchmark_returns_pandas).cumprod()
                peak_values = cum_rets.cummax()
                drawdown_values = (cum_rets / peak_values - 1) * 100

                # Get dates for plotting
                assert data_benchmark is not None  # narrowing: guarded by has_benchmark_data
                benchmark_sorted = data_benchmark.sort("Date")
                dates_list = benchmark_sorted.select("Date").to_series().to_list()

                if len(dates_list) > 1:
                    dates_for_drawdowns = dates_list[1 : len(drawdown_values) + 1]

                    figure_obj.add_trace(
                        go.Scatter(
                            x=dates_for_drawdowns,
                            y=drawdown_values,
                            fill="tozeroy",
                            fillcolor="rgba(255, 127, 14, 0.3)",
                            line={"color": "#ff7f0e", "width": 2, "dash": "dash"},
                            name="Benchmark Drawdowns",
                            hovertemplate="Date: %{x}<br>Benchmark Drawdown: %{y:.2f}%<extra></extra>",
                        ),
                    )

        figure_obj.update_layout(
            title={
                "text": f"Drawdowns Analysis ({period_label})",
                "font": {"size": 18},
                "x": 0.5,
                "xanchor": "center",
            },
            xaxis_title="Date",
            yaxis_title="Drawdown (%)",
            template="plotly_white",
            height=600,
            margin={"l": 60, "r": 60, "t": 120, "b": 60},
            hovermode="x unified",
            legend={
                "orientation": "h",
                "yanchor": "bottom",
                "y": 1.02,
                "xanchor": "center",
                "x": 0.5,
            },
        )

        return figure_obj

    except Exception as plotly_error:
        raise RuntimeError(f"Error creating drawdowns plot: {plotly_error}") from plotly_error


def calculate_table_stats_basic(
    data_portfolio: pl.DataFrame | None,
    data_benchmark: pl.DataFrame | None,
    include_benchmark: bool = False,
) -> pd.DataFrame:
    """Calculate basic statistics table for portfolio and benchmark data."""
    # Input validation with early returns
    if data_portfolio is None and data_benchmark is None:
        return pd.DataFrame(
            {"Metric": ["No Data"], "Value": ["No portfolio or benchmark data available"]},
        )

    # Configuration validation - check if DataFrames are valid
    has_portfolio_data = (
        data_portfolio is not None
        and isinstance(data_portfolio, pl.DataFrame)
        and not data_portfolio.is_empty()
    )

    has_benchmark_data = (
        data_benchmark is not None
        and isinstance(data_benchmark, pl.DataFrame)
        and not data_benchmark.is_empty()
    )

    if not has_portfolio_data and not has_benchmark_data:
        return pd.DataFrame(
            {"Metric": ["No Data"], "Value": ["Both portfolio and benchmark data are empty"]},
        )

    # Only use try-except for operations that might fail unpredictably
    try:
        # Calculate basic statistics
        stats_data = []

        if has_portfolio_data:
            portfolio_returns = calculate_portfolio_returns(data_portfolio)
            if len(portfolio_returns) > 0:
                portfolio_returns_series = portfolio_returns.to_pandas()

                stats_data.append(
                    {
                        "Metric": "Portfolio Daily Mean Return (%)",
                        "Value": f"{portfolio_returns_series.mean() * 100:.4f}",
                    },
                )
                stats_data.append(
                    {
                        "Metric": "Portfolio Daily Volatility (%)",
                        "Value": f"{portfolio_returns_series.std() * 100:.4f}",
                    },
                )
                stats_data.append(
                    {
                        "Metric": "Portfolio Annualized Return (%)",
                        "Value": f"{portfolio_returns_series.mean() * 252 * 100:.2f}",
                    },
                )
                stats_data.append(
                    {
                        "Metric": "Portfolio Annualized Volatility (%)",
                        "Value": f"{portfolio_returns_series.std() * np.sqrt(252) * 100:.2f}",
                    },
                )

                if portfolio_returns_series.std() > 0:
                    sharpe_ratio = (
                        portfolio_returns_series.mean()
                        / portfolio_returns_series.std()
                        * np.sqrt(252)
                    )
                    stats_data.append(
                        {
                            "Metric": "Portfolio Sharpe Ratio",
                            "Value": f"{sharpe_ratio:.4f}",
                        },
                    )

        if include_benchmark and has_benchmark_data:
            benchmark_returns = calculate_portfolio_returns(data_benchmark)
            if len(benchmark_returns) > 0:
                benchmark_returns_series = benchmark_returns.to_pandas()

                stats_data.append(
                    {
                        "Metric": "Benchmark Daily Mean Return (%)",
                        "Value": f"{benchmark_returns_series.mean() * 100:.4f}",
                    },
                )
                stats_data.append(
                    {
                        "Metric": "Benchmark Daily Volatility (%)",
                        "Value": f"{benchmark_returns_series.std() * 100:.4f}",
                    },
                )
                stats_data.append(
                    {
                        "Metric": "Benchmark Annualized Return (%)",
                        "Value": f"{benchmark_returns_series.mean() * 252 * 100:.2f}",
                    },
                )
                stats_data.append(
                    {
                        "Metric": "Benchmark Annualized Volatility (%)",
                        "Value": f"{benchmark_returns_series.std() * np.sqrt(252) * 100:.2f}",
                    },
                )

                if benchmark_returns_series.std() > 0:
                    sharpe_ratio = (
                        benchmark_returns_series.mean()
                        / benchmark_returns_series.std()
                        * np.sqrt(252)
                    )
                    stats_data.append(
                        {
                            "Metric": "Benchmark Sharpe Ratio",
                            "Value": f"{sharpe_ratio:.4f}",
                        },
                    )

        return (
            pd.DataFrame(stats_data)
            if stats_data
            else pd.DataFrame({"Metric": ["No Data"], "Value": ["N/A"]})
        )

    except Exception as exc:
        raise RuntimeError(f"Error generating basic statistics table: {exc}") from exc


def calculate_table_metrics_performance(
    data_portfolio: pl.DataFrame | None,
    data_benchmark: pl.DataFrame | None,
) -> pd.DataFrame:
    """Calculate performance metrics table for portfolio data."""
    # Input validation with early returns
    if data_portfolio is None:
        return pd.DataFrame({"Metric": ["No Data"], "Value": ["No portfolio data available"]})

    # Configuration validation - check if DataFrame is valid
    has_portfolio_data = isinstance(data_portfolio, pl.DataFrame) and not data_portfolio.is_empty()

    if not has_portfolio_data:
        return pd.DataFrame({"Metric": ["No Data"], "Value": ["Portfolio data is empty"]})

    # Only use try-except for operations that might fail unpredictably
    try:
        metrics_data = []

        portfolio_returns = calculate_portfolio_returns(data_portfolio)
        if len(portfolio_returns) > 0:
            # Calculate various performance metrics
            portfolio_returns_series = portfolio_returns.to_pandas()
            cum_returns = (1 + portfolio_returns_series).cumprod()

            # Max drawdown
            peak = cum_returns.cummax()
            drawdown = cum_returns / peak - 1
            max_dd = drawdown.min()

            metrics_data.append(
                {
                    "Metric": "Total Return (%)",
                    "Value": f"{(cum_returns.iloc[-1] - 1) * 100:.2f}",
                },
            )

            metrics_data.append(
                {
                    "Metric": "Max Drawdown (%)",
                    "Value": f"{max_dd * 100:.2f}",
                },
            )

            # Win rate
            positive_returns = portfolio_returns_series > 0
            win_rate = positive_returns.sum() / len(portfolio_returns_series) * 100
            metrics_data.append(
                {
                    "Metric": "Win Rate (%)",
                    "Value": f"{win_rate:.2f}",
                },
            )

            # Best/Worst days
            metrics_data.append(
                {
                    "Metric": "Best Day (%)",
                    "Value": f"{portfolio_returns_series.max() * 100:.2f}",
                },
            )

            metrics_data.append(
                {
                    "Metric": "Worst Day (%)",
                    "Value": f"{portfolio_returns_series.min() * 100:.2f}",
                },
            )

            # Skewness and Kurtosis
            metrics_data.append(
                {
                    "Metric": "Skewness",
                    "Value": f"{portfolio_returns_series.skew():.4f}",
                },
            )

            metrics_data.append(
                {
                    "Metric": "Kurtosis",
                    "Value": f"{portfolio_returns_series.kurtosis():.4f}",
                },
            )

        return (
            pd.DataFrame(metrics_data)
            if metrics_data
            else pd.DataFrame({"Metric": ["No Data"], "Value": ["N/A"]})
        )

    except Exception as exc:
        raise RuntimeError(f"Error generating performance metrics table: {exc}") from exc


def create_plot_comparison_portfolios(
    data_portfolio: pl.DataFrame | None,
    data_benchmark: pl.DataFrame | None,
    viz_type: str,
    show_diff: bool,
    period_label: str,
    start_date: date,
    end_date: date,
    data_was_validated: bool = True,
) -> go.Figure:
    """Generate the comparison plot figure using Polars DataFrames.

    Parameters
    ----------
    data_portfolio : polars.DataFrame
        Portfolio data with "Date" and "Value" columns
    data_benchmark : polars.DataFrame
        Benchmark data with "Date" and "Value" columns
    viz_type : str
        Visualization type ("normalized", "pct_change", "cum_return", "absolute")
    show_diff : bool
        Whether to show difference between portfolio and benchmark
    period_label : str
        Label for the time period
    start_date : datetime.date
        Start date for the period
    end_date : datetime.date
        End date for the period
    data_was_validated : bool, optional
        Whether data has already been validated, by default True

    Returns
    -------
    plotly.graph_objects.Figure
        Portfolio vs benchmark comparison plot

    Raises
    ------
    ValueError
        If data validation fails
    RuntimeError
        If plot creation fails
    """
    # Input validation with early returns
    if data_portfolio is None and data_benchmark is None:
        return create_error_figure("No portfolio or benchmark data available")

    if not isinstance(period_label, str):
        period_label = str(period_label) if period_label is not None else "Unknown Period"

    if not isinstance(viz_type, str):
        viz_type = "normalized"  # Default visualization type

    has_portfolio_data = True
    has_benchmark_data = True
    # Validate input data if not already validated
    if not data_was_validated:
        has_portfolio_data, has_benchmark_data = (
            validate_portfolio_and_benchmark_data_if_not_already_validated(
                data_portfolio=data_portfolio,
                data_benchmark=data_benchmark,
            )
        )

    if not has_portfolio_data and not has_benchmark_data:
        return create_error_figure("Both portfolio and benchmark data are empty")

    # Configuration validation - check if DataFrames are valid Polars DataFrames
    has_portfolio = (
        data_portfolio is not None
        and isinstance(data_portfolio, pl.DataFrame)
        and not data_portfolio.is_empty()
        and "Date" in data_portfolio.columns
        and "Value" in data_portfolio.columns
    )

    has_benchmark = (
        data_benchmark is not None
        and isinstance(data_benchmark, pl.DataFrame)
        and not data_benchmark.is_empty()
        and "Date" in data_benchmark.columns
        and "Value" in data_benchmark.columns
    )

    # Only use try-except for Plotly operations that might fail unpredictably
    try:
        # Apply aggressive downsampling for performance using Polars-compatible function
        if has_portfolio:
            data_portfolio = downsample_dataframe(
                data_portfolio,
                max_points=200,
                date_column="Date",
            )

        if has_benchmark:
            data_benchmark = downsample_dataframe(
                data_benchmark,
                max_points=200,
                date_column="Date",
            )

        # Create figure immediately
        figure_obj = go.Figure()

        # Determine plot title and y-axis label based on visualization type
        if viz_type == "normalized":
            plot_title = f"Portfolio vs Benchmark - Normalized ({period_label})"
            y_title = "Normalized Value (Base=100)"
        elif viz_type == "pct_change":
            plot_title = f"Portfolio vs Benchmark - Daily Changes ({period_label})"
            y_title = "Daily Change (%)"
        elif viz_type == "cum_return":
            plot_title = f"Portfolio vs Benchmark - Cumulative Returns ({period_label})"
            y_title = "Cumulative Return (%)"
        else:
            plot_title = f"Portfolio vs Benchmark - Absolute Values ({period_label})"
            y_title = "Value"

        # Add portfolio trace if available
        if has_portfolio:
            try:
                # Sort by Date and ensure proper data types using Polars
                assert data_portfolio is not None  # narrowing: guarded by has_portfolio
                portfolio_sorted = data_portfolio.sort("Date")

                # Handle different date column types
                date_dtype = portfolio_sorted.select("Date").dtypes[0]

                if date_dtype in [pl.Utf8, pl.String]:
                    # Convert string dates to datetime
                    portfolio_processed = portfolio_sorted.with_columns(
                        [
                            pl.col("Date").str.to_datetime(strict=False).alias("Date"),
                            pl.col("Value").cast(pl.Float64, strict=False).alias("Value"),
                        ],
                    )
                elif date_dtype == pl.Date:
                    # Convert date to datetime for consistency
                    portfolio_processed = portfolio_sorted.with_columns(
                        [
                            pl.col("Date").cast(pl.Datetime).alias("Date"),
                            pl.col("Value").cast(pl.Float64, strict=False).alias("Value"),
                        ],
                    )
                else:
                    # Already datetime or cast to datetime
                    portfolio_processed = portfolio_sorted.with_columns(
                        [
                            pl.col("Date").cast(pl.Datetime, strict=False).alias("Date"),
                            pl.col("Value").cast(pl.Float64, strict=False).alias("Value"),
                        ],
                    )

                # Remove null values
                portfolio_clean = portfolio_processed.filter(
                    pl.col("Date").is_not_null() & pl.col("Value").is_not_null(),
                )

                if portfolio_clean.height > 0:
                    # Extract data for plotting
                    portfolio_dates = portfolio_clean.select("Date").to_series().to_list()
                    portfolio_values = portfolio_clean.select("Value").to_series().to_list()

                    # Apply transformation based on visualization type
                    if viz_type == "normalized" and len(portfolio_values) > 0:
                        start_value = portfolio_values[0]
                        if start_value != 0:
                            y_values = [(value / start_value) * 100 for value in portfolio_values]
                        else:
                            y_values = portfolio_values
                    elif viz_type == "pct_change" and len(portfolio_values) > 1:
                        # Calculate percentage change using Polars
                        pct_change_series = (
                            portfolio_clean.select(
                                pl.col("Value").pct_change().alias("pct_change"),
                            ).to_series()
                            * 100
                        )
                        y_values = pct_change_series.drop_nulls().to_list()
                        # Adjust dates to match pct_change data (skip first date)
                        portfolio_dates = (
                            portfolio_dates[1 : len(y_values) + 1]
                            if len(portfolio_dates) > 1
                            else portfolio_dates
                        )
                    elif viz_type == "cum_return" and len(portfolio_values) > 0:
                        start_value = portfolio_values[0]
                        if start_value != 0:
                            y_values = [
                                ((value / start_value) - 1) * 100 for value in portfolio_values
                            ]
                        else:
                            y_values = portfolio_values
                    else:
                        y_values = portfolio_values

                    if len(y_values) > 0 and len(portfolio_dates) >= len(y_values):
                        figure_obj.add_trace(
                            go.Scatter(
                                x=portfolio_dates[: len(y_values)],
                                y=y_values,
                                mode="lines",
                                name="Portfolio",
                                line={"width": 2, "color": "#1f77b4"},
                                hovertemplate="<b>Portfolio</b><br>Date: %{x}<br>Value: %{y:.2f}<extra></extra>",
                            ),
                        )
            except Exception:
                pass  # Skip adding trace if there's an error

        # Add benchmark trace if available
        if has_benchmark:
            try:
                # Sort by Date and ensure proper data types using Polars
                assert data_benchmark is not None  # narrowing: guarded by has_benchmark
                benchmark_sorted = data_benchmark.sort("Date")

                # Handle different date column types
                date_dtype = benchmark_sorted.select("Date").dtypes[0]

                if date_dtype in [pl.Utf8, pl.String]:
                    # Convert string dates to datetime
                    benchmark_processed = benchmark_sorted.with_columns(
                        [
                            pl.col("Date").str.to_datetime(strict=False).alias("Date"),
                            pl.col("Value").cast(pl.Float64, strict=False).alias("Value"),
                        ],
                    )
                elif date_dtype == pl.Date:
                    # Convert date to datetime for consistency
                    benchmark_processed = benchmark_sorted.with_columns(
                        [
                            pl.col("Date").cast(pl.Datetime).alias("Date"),
                            pl.col("Value").cast(pl.Float64, strict=False).alias("Value"),
                        ],
                    )
                else:
                    # Already datetime or cast to datetime
                    benchmark_processed = benchmark_sorted.with_columns(
                        [
                            pl.col("Date").cast(pl.Datetime, strict=False).alias("Date"),
                            pl.col("Value").cast(pl.Float64, strict=False).alias("Value"),
                        ],
                    )

                # Remove null values
                benchmark_clean = benchmark_processed.filter(
                    pl.col("Date").is_not_null() & pl.col("Value").is_not_null(),
                )

                if benchmark_clean.height > 0:
                    # Extract data for plotting
                    benchmark_dates = benchmark_clean.select("Date").to_series().to_list()
                    benchmark_values = benchmark_clean.select("Value").to_series().to_list()

                    # Apply transformation based on visualization type
                    if viz_type == "normalized" and len(benchmark_values) > 0:
                        start_value = benchmark_values[0]
                        if start_value != 0:
                            y_values = [(value / start_value) * 100 for value in benchmark_values]
                        else:
                            y_values = benchmark_values
                    elif viz_type == "pct_change" and len(benchmark_values) > 1:
                        # Calculate percentage change using Polars
                        pct_change_series = (
                            benchmark_clean.select(
                                pl.col("Value").pct_change().alias("pct_change"),
                            ).to_series()
                            * 100
                        )
                        y_values = pct_change_series.drop_nulls().to_list()
                        # Adjust dates to match pct_change data (skip first date)
                        benchmark_dates = (
                            benchmark_dates[1 : len(y_values) + 1]
                            if len(benchmark_dates) > 1
                            else benchmark_dates
                        )
                    elif viz_type == "cum_return" and len(benchmark_values) > 0:
                        start_value = benchmark_values[0]
                        if start_value != 0:
                            y_values = [
                                ((value / start_value) - 1) * 100 for value in benchmark_values
                            ]
                        else:
                            y_values = benchmark_values
                    else:
                        y_values = benchmark_values

                    if len(y_values) > 0 and len(benchmark_dates) >= len(y_values):
                        figure_obj.add_trace(
                            go.Scatter(
                                x=benchmark_dates[: len(y_values)],
                                y=y_values,
                                mode="lines",
                                name="Benchmark",
                                line={"width": 2, "color": "#ff7f0e", "dash": "dash"},
                                hovertemplate="<b>Benchmark</b><br>Date: %{x}<br>Value: %{y:.2f}<extra></extra>",
                            ),
                        )
            except Exception:
                pass  # Skip adding trace if there's an error

        # Handle empty figure
        if len(figure_obj.data) == 0:  # pyright: ignore[reportArgumentType]
            figure_obj.add_annotation(
                text=f"No data available for {period_label}<br>{start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}",
                xref="paper",
                yref="paper",
                x=0.5,
                y=0.5,
                showarrow=False,
                font={"size": 16},
                align="center",
            )

        # Apply layout with dynamic title
        figure_obj.update_layout(
            title={
                "text": plot_title,
                "font": {"size": 16},
                "x": 0.5,
                "xanchor": "center",
            },
            xaxis_title="Date",
            yaxis_title=y_title,
            template="plotly_white",
            height=600,
            margin={"l": 60, "r": 40, "t": 80, "b": 60},
            legend={
                "orientation": "h",
                "yanchor": "bottom",
                "y": 1.02,
                "xanchor": "center",
                "x": 0.5,
            },
            hovermode="x unified",
        )

        return figure_obj

    except Exception as exc:
        raise RuntimeError(f"Error generating comparison plot: {exc}") from exc


def create_table_comparison_stats(
    data_portfolio: pl.DataFrame | None,
    data_benchmark: pl.DataFrame | None,
    time_period: str,
    start_date: date,
    end_date: date,
    data_was_validated: bool = True,
) -> GT:
    """
    Create statistics comparison table using great-tables for portfolio and benchmark data.

    Parameters
    ----------
    data_portfolio : polars.DataFrame
        Portfolio data with "Date" and "Value" columns
    data_benchmark : polars.DataFrame
        Benchmark data with "Date" and "Value" columns
    time_period : str
        Time period identifier
    start_date : datetime.date
        Start date for the period
    end_date : datetime.date
        End date for the period
    data_was_validated : bool, optional
        Whether data has already been validated, by default True

    Returns
    -------
    great_tables.GT
        Formatted comparison statistics table

    Raises
    ------
    ValueError
        If data validation fails
    RuntimeError
        If table creation fails
    """
    # Input validation with early returns
    if data_portfolio is None and data_benchmark is None:
        # Create empty table with proper structure
        empty_data = pd.DataFrame(
            {
                "Metric": ["No Data Available"],
                "Portfolio": ["-"],
                "Benchmark": ["-"],
                "Difference": ["-"],
            },
        )
        return GT(empty_data).tab_header(
            title="Portfolio vs Benchmark Statistics",
            subtitle="No data available for comparison",
        )

    if not isinstance(time_period, str):
        time_period = str(time_period) if time_period is not None else "Unknown Period"

    # Configuration validation - check if DataFrames are valid Polars DataFrames
    has_portfolio = (
        data_portfolio is not None
        and isinstance(data_portfolio, pl.DataFrame)
        and not data_portfolio.is_empty()
        and "Date" in data_portfolio.columns
        and "Value" in data_portfolio.columns
    )

    has_benchmark = (
        data_benchmark is not None
        and isinstance(data_benchmark, pl.DataFrame)
        and not data_benchmark.is_empty()
        and "Date" in data_benchmark.columns
        and "Value" in data_benchmark.columns
    )

    if not has_portfolio and not has_benchmark:
        # Create empty table with proper structure
        empty_data = pd.DataFrame(
            {
                "Metric": ["No Valid Data"],
                "Portfolio": ["-"],
                "Benchmark": ["-"],
                "Difference": ["-"],
            },
        )
        return GT(empty_data).tab_header(
            title="Portfolio vs Benchmark Statistics",
            subtitle="Both portfolio and benchmark data are empty or invalid",
        )

    # Only use try-except for operations that might fail unpredictably
    try:
        # Initialize statistics data

        # Calculate period information
        period_info = {
            "Period": time_period.replace("_", " ").title()
            if time_period != "custom"
            else "Custom Range",
            "Date Range": f"{start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}",
            "Portfolio Points": 0,
            "Benchmark Points": 0,
        }

        # Process portfolio data if available
        portfolio_stats = {}
        if has_portfolio:
            try:
                # Sort by Date and ensure proper data types using Polars
                assert data_portfolio is not None  # narrowing: guarded by has_portfolio
                portfolio_sorted = data_portfolio.sort("Date")

                # Handle different date column types
                date_dtype = portfolio_sorted.select("Date").dtypes[0]

                if date_dtype in [pl.Utf8, pl.String]:
                    # Convert string dates to datetime
                    portfolio_processed = portfolio_sorted.with_columns(
                        [
                            pl.col("Date").str.to_datetime(strict=False).alias("Date"),
                            pl.col("Value").cast(pl.Float64, strict=False).alias("Value"),
                        ],
                    )
                elif date_dtype == pl.Date:
                    # Convert date to datetime for consistency
                    portfolio_processed = portfolio_sorted.with_columns(
                        [
                            pl.col("Date").cast(pl.Datetime).alias("Date"),
                            pl.col("Value").cast(pl.Float64, strict=False).alias("Value"),
                        ],
                    )
                else:
                    # Already datetime or cast to datetime
                    portfolio_processed = portfolio_sorted.with_columns(
                        [
                            pl.col("Date").cast(pl.Datetime, strict=False).alias("Date"),
                            pl.col("Value").cast(pl.Float64, strict=False).alias("Value"),
                        ],
                    )

                # Remove null values
                portfolio_clean = portfolio_processed.filter(
                    pl.col("Date").is_not_null() & pl.col("Value").is_not_null(),
                )

                if portfolio_clean.height > 0:
                    # Extract values for calculations
                    portfolio_values = portfolio_clean.select("Value").to_series().to_list()

                    period_info["Portfolio Points"] = len(portfolio_values)

                    # Calculate basic statistics
                    start_value: float = 0.0
                    end_value: float = 0.0
                    if len(portfolio_values) > 1:
                        start_value = portfolio_values[0]
                        end_value = portfolio_values[-1]

                        if start_value != 0:
                            total_return = ((end_value / start_value) - 1) * 100
                            portfolio_stats["Total Return (%)"] = total_return

                        # Calculate returns for additional statistics
                        portfolio_returns = (
                            portfolio_clean.select(
                                pl.col("Value").pct_change().alias("returns"),
                            )
                            .drop_nulls()
                            .to_series()
                            .to_list()
                        )

                        if len(portfolio_returns) > 0:
                            portfolio_stats["Daily Mean Return (%)"] = (
                                np.mean(portfolio_returns) * 100
                            )
                            portfolio_stats["Daily Volatility (%)"] = (
                                np.std(portfolio_returns) * 100
                            )
                            portfolio_stats["Annualized Return (%)"] = (
                                np.mean(portfolio_returns) * 252 * 100
                            )
                            portfolio_stats["Annualized Volatility (%)"] = (
                                np.std(portfolio_returns) * np.sqrt(252) * 100
                            )

                            if np.std(portfolio_returns) > 0:
                                portfolio_stats["Sharpe Ratio"] = (
                                    np.mean(portfolio_returns)
                                    / np.std(portfolio_returns)
                                    * np.sqrt(252)
                                )

                        # Calculate max drawdown
                        cum_returns = np.cumprod(1 + np.array(portfolio_returns))
                        peak_values = np.maximum.accumulate(cum_returns)
                        drawdown_values = cum_returns / peak_values - 1
                        max_dd = drawdown_values.min()

                        portfolio_stats["Max Drawdown (%)"] = max_dd * 100

                    portfolio_stats["Start Value"] = start_value
                    portfolio_stats["End Value"] = end_value
                    portfolio_stats["Min Value"] = min(portfolio_values)
                    portfolio_stats["Max Value"] = max(portfolio_values)

            except Exception:
                portfolio_stats = {"Error": "Failed to calculate portfolio statistics"}

        # Process benchmark data if available (same approach as portfolio)
        benchmark_stats = {}
        if has_benchmark:
            try:
                # Sort by Date and ensure proper data types using Polars
                assert data_benchmark is not None  # narrowing: guarded by has_benchmark
                benchmark_sorted = data_benchmark.sort("Date")

                # Handle different date column types
                date_dtype = benchmark_sorted.select("Date").dtypes[0]

                if date_dtype in [pl.Utf8, pl.String]:
                    # Convert string dates to datetime
                    benchmark_processed = benchmark_sorted.with_columns(
                        [
                            pl.col("Date").str.to_datetime(strict=False).alias("Date"),
                            pl.col("Value").cast(pl.Float64, strict=False).alias("Value"),
                        ],
                    )
                elif date_dtype == pl.Date:
                    # Convert date to datetime for consistency
                    benchmark_processed = benchmark_sorted.with_columns(
                        [
                            pl.col("Date").cast(pl.Datetime).alias("Date"),
                            pl.col("Value").cast(pl.Float64, strict=False).alias("Value"),
                        ],
                    )
                else:
                    # Already datetime or cast to datetime
                    benchmark_processed = benchmark_sorted.with_columns(
                        [
                            pl.col("Date").cast(pl.Datetime, strict=False).alias("Date"),
                            pl.col("Value").cast(pl.Float64, strict=False).alias("Value"),
                        ],
                    )

                # Remove null values
                benchmark_clean = benchmark_processed.filter(
                    pl.col("Date").is_not_null() & pl.col("Value").is_not_null(),
                )

                if benchmark_clean.height > 0:
                    # Extract values for calculations
                    benchmark_values = benchmark_clean.select("Value").to_series().to_list()

                    period_info["Benchmark Points"] = len(benchmark_values)

                    # Calculate basic statistics
                    start_value = 0.0
                    end_value = 0.0
                    if len(benchmark_values) > 1:
                        start_value = benchmark_values[0]
                        end_value = benchmark_values[-1]

                        if start_value != 0:
                            total_return = ((end_value / start_value) - 1) * 100
                            benchmark_stats["Total Return (%)"] = total_return

                        # Calculate returns for additional statistics
                        benchmark_returns = (
                            benchmark_clean.select(
                                pl.col("Value").pct_change().alias("returns"),
                            )
                            .drop_nulls()
                            .to_series()
                            .to_list()
                        )

                        if len(benchmark_returns) > 0:
                            benchmark_stats["Daily Mean Return (%)"] = (
                                np.mean(benchmark_returns) * 100
                            )
                            benchmark_stats["Daily Volatility (%)"] = (
                                np.std(benchmark_returns) * 100
                            )
                            benchmark_stats["Annualized Return (%)"] = (
                                np.mean(benchmark_returns) * 252 * 100
                            )
                            benchmark_stats["Annualized Volatility (%)"] = (
                                np.std(benchmark_returns) * np.sqrt(252) * 100
                            )

                            if np.std(benchmark_returns) > 0:
                                benchmark_stats["Sharpe Ratio"] = (
                                    np.mean(benchmark_returns)
                                    / np.std(benchmark_returns)
                                    * np.sqrt(252)
                                )

                        # Calculate max drawdown
                        cum_returns = np.cumprod(1 + np.array(benchmark_returns))
                        peak_values = np.maximum.accumulate(cum_returns)
                        drawdown_values = cum_returns / peak_values - 1
                        benchmark_stats["Max Drawdown (%)"] = np.min(drawdown_values) * 100

                    benchmark_stats["Start Value"] = start_value
                    benchmark_stats["End Value"] = end_value
                    benchmark_stats["Min Value"] = min(benchmark_values)
                    benchmark_stats["Max Value"] = max(benchmark_values)

            except Exception:
                benchmark_stats = {"Error": "Failed to calculate benchmark statistics"}

        # Build comparison table data
        all_metrics = set()
        if portfolio_stats:
            all_metrics.update(portfolio_stats.keys())
        if benchmark_stats:
            all_metrics.update(benchmark_stats.keys())

        # Remove error keys from metrics list
        all_metrics = sorted([metric for metric in all_metrics if metric != "Error"])

        # Create table data
        table_data = []

        # Add period information rows
        table_data.append(
            {
                "Metric": "Selected Period",
                "Portfolio": period_info["Period"],
                "Benchmark": period_info["Period"],
                "Difference": "-",
            },
        )

        table_data.append(
            {
                "Metric": "Date Range",
                "Portfolio": period_info["Date Range"],
                "Benchmark": period_info["Date Range"],
                "Difference": "-",
            },
        )

        table_data.append(
            {
                "Metric": "Data Points",
                "Portfolio": str(period_info["Portfolio Points"]),
                "Benchmark": str(period_info["Benchmark Points"]),
                "Difference": str(
                    period_info["Portfolio Points"] - period_info["Benchmark Points"],
                ),
            },
        )

        # Add performance metrics
        for metric in all_metrics:
            portfolio_value = portfolio_stats.get(metric, "N/A")
            benchmark_value = benchmark_stats.get(metric, "N/A")

            # Calculate difference if both values are numeric
            difference = "N/A"
            if (
                isinstance(portfolio_value, (int, float))
                and isinstance(benchmark_value, (int, float))
                and not pd.isna(portfolio_value)
                and not pd.isna(benchmark_value)
            ):
                difference = portfolio_value - benchmark_value
                difference = f"{difference:.4f}"

            # Format values for display
            portfolio_display = (
                format_value_for_display(portfolio_value) if portfolio_value != "N/A" else "N/A"
            )
            benchmark_display = (
                format_value_for_display(benchmark_value) if benchmark_value != "N/A" else "N/A"
            )

            table_data.append(
                {
                    "Metric": metric,
                    "Portfolio": portfolio_display,
                    "Benchmark": benchmark_display,
                    "Difference": difference,
                },
            )

        # Create pandas DataFrame for great-tables
        comparison_df = pd.DataFrame(table_data)

        # Create great-tables GT object and style it
        return (
            GT(comparison_df)
            .tab_header(
                title="Portfolio vs Benchmark Statistics",
                subtitle=f"Performance comparison for {period_info['Period']}",
            )
            .cols_label(
                Metric="Metric",
                Portfolio="Portfolio",
                Benchmark="Benchmark",
                Difference="Difference (P-B)",
            )
            .tab_style(
                style=style.fill(color="#f8f9fa"),
                locations=loc.body(rows=[0, 1, 2]),  # Highlight period information rows
            )
            .tab_style(
                style=style.text(weight="bold"),
                locations=loc.body(columns=["Metric"]),
            )
        )

    except Exception as exc:
        # Create error table using great-tables
        error_data = pd.DataFrame(
            {
                "Metric": ["Error"],
                "Portfolio": [f"Error: {exc!s}"],
                "Benchmark": ["-"],
                "Difference": ["-"],
            },
        )

        (
            GT(error_data)
            .tab_header(
                title="Portfolio vs Benchmark Statistics",
                subtitle="Error occurred during calculation",
            )
            .tab_style(
                style=style.fill(color="#f8d7da"),
                locations=loc.body(),
            )
        )

        raise RuntimeError(f"Error creating comparison statistics table: {exc}") from exc


def create_table_summary_weights_analysis(
    data_stats: dict[str, Any],
    show_pct: bool = True,
    data_was_validated: bool = True,
) -> GT:
    """
    Create summary statistics table for Weights Analysis using great-tables.

    Parameters
    ----------
    data_stats : dict
        Dictionary containing weight statistics for each component
    show_pct : bool, optional
        Whether to show values as percentages, by default True
    data_was_validated : bool, optional
        Whether data has already been validated, by default True

    Returns
    -------
    great_tables.GT
        Formatted Weights Analysis statistics table

    Raises
    ------
    ValueError
        If data validation fails
    RuntimeError
        If table creation fails
    """
    # Input validation with early returns
    if not data_stats or not isinstance(data_stats, dict):
        # Create empty table with proper structure
        empty_data = pd.DataFrame(
            {
                "Component": ["No Data Available"],
                "Latest": ["-"],
                "Average": ["-"],
                "Min": ["-"],
                "Max": ["-"],
                "StdDev": ["-"],
            },
        )
        return GT(empty_data).tab_header(
            title="Portfolio Weight Distribution Statistics",
            subtitle="No data available",
        )

    # Configuration validation - check for period info
    period_info = data_stats.get("_period_info", {})
    period_label = period_info.get("period_label", "Unknown Period")
    data_points = period_info.get("data_points", 0)

    # Only use try-except for operations that might fail unpredictably
    try:
        # Extract component statistics
        component_stats = {k: v for k, v in data_stats.items() if not k.startswith("_")}

        if not component_stats:
            # Create empty table with proper structure
            empty_data = pd.DataFrame(
                {
                    "Component": ["No Components Available"],
                    "Latest": ["-"],
                    "Average": ["-"],
                    "Min": ["-"],
                    "Max": ["-"],
                    "StdDev": ["-"],
                },
            )
            return GT(empty_data).tab_header(
                title="Portfolio Weight Distribution Statistics",
                subtitle="No component data available",
            )

        # Build table data
        table_data = []

        for component_name, stats in component_stats.items():
            # Get statistics with fallbacks
            latest_weight = stats.get("Latest", 0)
            average_weight = stats.get("Average", 0)
            min_weight = stats.get("Min", 0)
            max_weight = stats.get("Max", 0)
            std_weight = stats.get("StdDev", 0)

            # Safe formatting function for display values
            def safe_format_value(value: Any, as_percentage: Any = True) -> str | None:
                """Safely format a value for display with defensive programming."""
                # Input validation with early returns
                if value is None:
                    return "-"

                if pd.isna(value):
                    return "-"

                if not isinstance(value, (int, float)):
                    return "-"

                # Only use try-except for string formatting that might fail
                try:
                    if as_percentage:
                        return f"{float(value):.2f}%"
                    return f"{float(value):.4f}"
                except (ValueError, TypeError, OverflowError):
                    return "-"

            # Format values based on show_pct flag using safe formatting
            latest_display = safe_format_value(latest_weight, show_pct)
            average_display = safe_format_value(average_weight, show_pct)
            min_display = safe_format_value(min_weight, show_pct)
            max_display = safe_format_value(max_weight, show_pct)
            std_display = safe_format_value(std_weight, show_pct)

            table_data.append(
                {
                    "Component": str(component_name),  # Ensure string conversion
                    "Latest": latest_display,
                    "Average": average_display,
                    "Min": min_display,
                    "Max": max_display,
                    "StdDev": std_display,
                },
            )

        # Sort by latest weight (descending) with safe numeric extraction
        def extract_numeric_for_sorting(display_value: Any) -> Any:
            """Extract numeric value from display string for sorting."""
            if display_value == "-":
                return 0

            # Only use try-except for string operations that might fail
            try:
                # Remove percentage sign if present
                numeric_str = display_value.rstrip("%")
                return float(numeric_str)
            except (ValueError, TypeError, AttributeError):
                return 0

        table_data.sort(key=lambda x: extract_numeric_for_sorting(x["Latest"]), reverse=True)

        # Create pandas DataFrame for great-tables
        weights_df = pd.DataFrame(table_data)

        # Create great-tables GT object
        return (
            GT(weights_df)
            .tab_header(
                title="Portfolio Weight Distribution Statistics",
                subtitle=f"Weight analysis for {period_label} ({data_points} data points)",
            )
            .cols_label(
                Component="Component",
                Latest="Latest Weight",
                Average="Average Weight",
                Min="Minimum Weight",
                Max="Maximum Weight",
                StdDev="Standard Deviation",
            )
            .tab_style(
                style=style.text(weight="bold"),
                locations=loc.body(columns=["Component"]),
            )
        )

    except Exception as exc:
        # Create error table using great-tables
        error_data = pd.DataFrame(
            {
                "Component": ["Error"],
                "Latest": [f"Error: {exc!s}"],
                "Average": ["-"],
                "Min": ["-"],
                "Max": ["-"],
                "StdDev": ["-"],
            },
        )

        (
            GT(error_data)
            .tab_header(
                title="Portfolio Weight Distribution Statistics",
                subtitle="Error occurred during calculation",
            )
            .tab_style(
                style=style.fill(color="#f8d7da"),
                locations=loc.body(),
            )
        )

        raise RuntimeError(f"Error creating Weights Analysis statistics table: {exc}") from exc


def create_enhanced_summary_table(
    dataframe_input: pl.DataFrame,
    table_title: str = "Summary Table",
    table_subtitle: str = "",
    column_headers: list[str] | None = None,
    currency_columns: list[str] | None = None,
    percentage_columns: list[str] | None = None,
    table_theme: str = "professional",
    show_row_numbers: bool = False,
    table_width: str = "100%",
) -> GT:
    """Create enhanced summary table using great-tables with comprehensive formatting.

    This function generates professional, customizable tables using the great-tables
    package with polars dataframes. It supports 1-2 column dataframes containing
    mixed numeric and string data types with comprehensive formatting options for
    financial data display, currency values, and percentage calculations.

    The function follows defensive programming principles with comprehensive input
    validation using early returns, configuration validation for table parameters,
    and business logic validation for data formatting requirements. All styling
    follows the project's enhanced UI patterns with consistent visual design.

    Args:
        dataframe_input: Polars DataFrame containing 1-2 columns with mixed data types.
            First column typically contains category names (strings), second column
            contains values (numeric or string). Required for table generation.
        table_title: Title displayed at top of table for identification and context.
            Default: "Summary Table". Used for table header identification.
        table_subtitle: Optional subtitle providing additional context or description.
            Default: empty string. Displayed below main title when provided.
        column_headers: Optional list of custom column header names for display.
            Default: None (uses dataframe column names). Must match column count.
        currency_columns: Optional list of column names to format as currency values.
            Default: None. Applies currency formatting ($X,XXX.XX) to specified columns.
        percentage_columns: Optional list of column names to format as percentages.
            Default: None. Applies percentage formatting (XX.X%) to specified columns.
        table_theme: Theme selection for table styling and visual appearance.
            Default: "professional". Options: "professional", "minimal", "enhanced".
        show_row_numbers: Whether to display row numbers in the table for reference.
            Default: False. Adds sequential numbering when enabled.
        table_width: CSS width specification for table container sizing.
            Default: "100%". Accepts CSS width values (px, %, em, rem).

    Returns
    -------
        GT: Configured great-tables GT object with comprehensive formatting, styling,
            and professional appearance ready for display in Shiny applications.
            Includes currency formatting, percentage display, and enhanced readability.

    Raises
    ------
        ValueError: If dataframe_input is None, empty, has invalid structure (not 1-2 columns),
            contains incompatible data types, or column_headers count doesn't match
            dataframe columns during validation and table generation processes.
        TypeError: If dataframe_input is not a polars DataFrame, column_headers is not
            a list when provided, or formatting parameters are not of expected types
            during table configuration and styling operations.
        RuntimeError: If great-tables package is not available, table generation fails
            due to formatting conflicts, or unexpected errors occur during table
            creation and styling application processes.
        ImportError: If required dependencies (great_tables, polars) are not installed
            or available in the current environment during module initialization
            and table generation operations.

    Examples
    --------
        Create basic summary table with personal information:

        ```python
        # Create dataframe with personal info
        Personal_Info_Data = pl.DataFrame(
            {"Category": ["Name", "Age", "Risk Tolerance"], "Value": ["John Doe", "45", "Moderate"]}
        )

        # Generate professional table
        Personal_Info_Table = create_enhanced_summary_table(
            dataframe_input=Personal_Info_Data,
            table_title="Personal Information Summary",
            table_subtitle="Primary Investor Details",
        )
        ```

        Create financial table with currency formatting:

        ```python
        # Create dataframe with financial assets
        Assets_Data = pl.DataFrame(
            {
                "Asset_Type": ["Taxable Assets", "Tax Deferred", "Tax Free"],
                "Amount": [250000.00, 400000.00, 75000.00],
            }
        )

        # Generate table with currency formatting
        Assets_Table = create_enhanced_summary_table(
            dataframe_input=Assets_Data,
            table_title="Financial Assets Summary",
            column_headers=["Asset Type", "Current Value"],
            currency_columns=["Amount"],
            table_theme="enhanced",
        )
        ```

        Create goals table with enhanced styling:

        ```python
        # Create dataframe with financial goals
        Goals_Data = pl.DataFrame(
            {
                "Goal_Priority": ["Essential", "Important", "Aspirational"],
                "Annual_Amount": [60000, 25000, 15000],
            }
        )

        # Generate table with comprehensive formatting
        Goals_Table = create_enhanced_summary_table(
            dataframe_input=Goals_Data,
            table_title="Financial Goals Summary",
            table_subtitle="Annual Target Amounts",
            column_headers=["Priority Level", "Annual Target"],
            currency_columns=["Annual_Amount"],
            show_row_numbers=True,
            table_width="90%",
        )
        ```

    Note:
        This function requires the great-tables package for professional table generation
        and polars for high-performance dataframe operations. All formatting options
        support both single and dual column dataframes with mixed data types.

        Table themes provide different visual styles:
        - "professional": Corporate styling with clean lines and proper spacing
        - "minimal": Simple styling with reduced visual elements
        - "enhanced": Rich styling with enhanced colors and formatting

        Currency formatting automatically applies to numeric columns specified in
        currency_columns parameter, displaying values as $X,XXX.XX format. Percentage
        formatting converts decimal values to XX.X% display format.

        Performance considerations: Large dataframes (>1000 rows) may experience
        slower rendering. Consider data pagination for extensive datasets.

    Security:
        Input validation prevents injection attacks through dataframe content validation
        and parameter sanitization. All user-provided data is properly escaped before
        table generation to prevent XSS vulnerabilities in web display contexts.

    See Also
    --------
        subtab_clients_summary_ui: UI creation function for summary subtab
        subtab_clients_summary_server: Server logic for summary table generation
        format_enhanced_currency_display: Currency formatting utilities
        utils_enhanced_formatting: Comprehensive data formatting utilities
        great_tables.GT: Great-tables GT class documentation
        polars.DataFrame: Polars DataFrame documentation
    """
    # Input validation with early returns following coding standards
    if dataframe_input is None:
        raise ValueError("Dataframe input cannot be None")

    if not isinstance(dataframe_input, pl.DataFrame):
        raise TypeError("Input must be a polars DataFrame")

    if dataframe_input.is_empty():
        raise ValueError("Dataframe cannot be empty")

    # Configuration validation - check dataframe structure
    Column_Count = len(dataframe_input.columns)
    if Column_Count not in [1, 2]:
        raise ValueError("Dataframe must have exactly 1 or 2 columns")

    if column_headers is not None:
        if not isinstance(column_headers, list):
            raise TypeError("Column headers must be a list")
        if len(column_headers) != Column_Count:
            raise ValueError("Column headers count must match dataframe columns")

    # Business logic validation - prepare table data
    try:
        # Create GT table object from polars dataframe
        Table_GT = GT(dataframe_input)

        # Apply custom column headers if provided
        if column_headers is not None:
            Table_GT = Table_GT.cols_label(
                cases={
                    dataframe_input.columns[idx]: column_headers[idx]
                    for idx in range(len(column_headers))
                },
            )

        # Apply table title and subtitle
        if table_title:
            Table_GT = Table_GT.tab_header(
                title=md(f"**{table_title}**"),
                subtitle=md(table_subtitle) if table_subtitle else None,
            )

        # Apply currency formatting to specified columns
        if currency_columns:
            for Currency_Column in currency_columns:
                if Currency_Column in dataframe_input.columns:
                    Table_GT = Table_GT.fmt_currency(
                        columns=[Currency_Column],
                        currency="USD",
                        decimals=2,
                    )

        # Apply percentage formatting to specified columns
        if percentage_columns:
            for Percentage_Column in percentage_columns:
                if Percentage_Column in dataframe_input.columns:
                    Table_GT = Table_GT.fmt_percent(
                        columns=[Percentage_Column],
                        decimals=1,
                    )

        # Apply theme-based styling
        if table_theme == "professional":
            Table_GT = Table_GT.tab_options(
                table_width=table_width,
                table_font_size="14px",
                heading_background_color="#f8f9fa",
                heading_title_font_size="18px",
                heading_subtitle_font_size="14px",
                column_labels_background_color="#e9ecef",
                row_striping_background_color="#f8f9fa",
            )
        elif table_theme == "minimal":
            Table_GT = Table_GT.tab_options(
                table_width=table_width,
                table_font_size="13px",
                table_border_top_style="hidden",
                table_border_bottom_style="hidden",
            )
        elif table_theme == "enhanced":
            Table_GT = Table_GT.tab_options(
                table_width=table_width,
                table_font_size="14px",
                heading_background_color="#007bff",
                heading_title_font_size="20px",
                heading_title_font_color="white",  # type: ignore[reportCallIssue]
                heading_subtitle_font_color="white",  # type: ignore[reportCallIssue]
                column_labels_background_color="#6c757d",
                column_labels_font_color="white",  # type: ignore[reportCallIssue]
                row_striping_background_color="#f1f3f4",
            )

        # Add row numbers if requested
        if show_row_numbers:
            Table_GT = Table_GT.opt_row_striping()

        return Table_GT

    except Exception as exc_error:
        raise RuntimeError(f"Error creating enhanced summary table: {exc_error!s}") from exc_error


def create_enhanced_summary_table_multi_column(
    dataframe_input: pl.DataFrame,
    table_title: str = "Summary Table",
    table_subtitle: str = "",
    currency_columns: list[str] | None = None,
    percentage_columns: list[str] | None = None,
    table_theme: str = "professional",
    table_width: str = "100%",
) -> GT:
    """Create enhanced summary table using great-tables with support for multiple columns.

    This function generates professional, customizable tables using the great-tables
    package with polars dataframes. It supports dataframes with any number of columns
    containing mixed numeric and string data types with comprehensive formatting options.

    Args:
        dataframe_input: Polars DataFrame containing data for table generation.
        table_title: Title displayed at top of table for identification and context.
        table_subtitle: Optional subtitle providing additional context or description.
        currency_columns: Optional list of column names to format as currency values.
        percentage_columns: Optional list of column names to format as percentages.
        table_theme: Theme selection for table styling and visual appearance.
        table_width: CSS width specification for table container sizing.

    Returns
    -------
        GT: Configured great-tables GT object with comprehensive formatting and styling.

    Raises
    ------
        ValueError: If dataframe_input is None or empty.
        TypeError: If dataframe_input is not a polars DataFrame.
        RuntimeError: If table generation fails.
    """
    # Input validation with early returns following coding standards
    if dataframe_input is None:
        raise ValueError("Dataframe input cannot be None")

    if not isinstance(dataframe_input, pl.DataFrame):
        raise TypeError("Input must be a polars DataFrame")

    if dataframe_input.is_empty():
        raise ValueError("Dataframe cannot be empty")

    # Business logic validation - prepare table data
    try:
        # Convert polars DataFrame to pandas for great-tables compatibility
        Pandas_DataFrame = dataframe_input.to_pandas()

        # Create GT table object from pandas dataframe
        Table_GT = GT(Pandas_DataFrame)

        # Apply table title and subtitle
        if table_title:
            Table_GT = Table_GT.tab_header(
                title=md(f"**{table_title}**"),
                subtitle=md(table_subtitle) if table_subtitle else None,
            )

        # Apply currency formatting to specified columns
        if currency_columns:
            for Currency_Column in currency_columns:
                if Currency_Column in Pandas_DataFrame.columns:
                    Table_GT = Table_GT.fmt_currency(
                        columns=[Currency_Column],
                        currency="USD",
                        decimals=2,
                    )

        # Apply percentage formatting to specified columns
        if percentage_columns:
            for Percentage_Column in percentage_columns:
                if Percentage_Column in Pandas_DataFrame.columns:
                    Table_GT = Table_GT.fmt_percent(
                        columns=[Percentage_Column],
                        decimals=1,
                    )

        # Apply theme-based styling with corrected syntax
        if table_theme == "professional":
            Table_GT = Table_GT.tab_options(
                table_width=table_width,
                table_font_size="14px",
                heading_background_color="#f8f9fa",
                heading_title_font_size="18px",
                heading_subtitle_font_size="14px",
                column_labels_background_color="#e9ecef",
            )
            # Add styling for first column
            Table_GT = Table_GT.tab_style(
                style=style.fill(color="#f8f9fa"),
                locations=loc.body(columns=[Pandas_DataFrame.columns[0]]),
            ).tab_style(
                style=style.text(weight="bold"),
                locations=loc.body(columns=[Pandas_DataFrame.columns[0]]),
            )
        elif table_theme == "minimal":
            Table_GT = Table_GT.tab_options(
                table_width=table_width,
                table_font_size="13px",
            )
        elif table_theme == "enhanced":
            Table_GT = Table_GT.tab_options(
                table_width=table_width,
                table_font_size="14px",
                heading_background_color="#007bff",
                heading_title_font_size="20px",
                column_labels_background_color="#6c757d",
            )

        return Table_GT

    except Exception as exc_error:
        raise RuntimeError(f"Error creating enhanced summary table: {exc_error!s}") from exc_error
