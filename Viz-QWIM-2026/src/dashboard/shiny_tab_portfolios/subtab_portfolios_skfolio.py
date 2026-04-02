"""Portfolio Optimization with skfolio Module.

Provides comprehensive portfolio optimization using methods from the skfolio Python package.

![Portfolio Analysis](https://img.shields.io/badge/Portfolio-Analysis-blue)
![Python](https://img.shields.io/badge/python-3.12+-green)
![Shiny](https://img.shields.io/badge/shiny-for_python-orange)

## Overview

The skfolio Portfolio Optimization module enables users to compare different portfolio
optimization strategies using state-of-the-art methods from the skfolio package.

This module provides an interactive interface to:
- Select and configure optimization methods (Basic, Convex, Clustering, Ensemble)
- Compare two different optimization strategies side-by-side
- Visualize portfolio weights and performance metrics
- Export results for further analysis

## Features

- **Multiple Optimization Methods**: Basic, Convex, Clustering, and Ensemble strategies
- **Side-by-Side Comparison**: Compare two optimization strategies simultaneously
- **Interactive Configuration**: Dynamic parameter inputs based on selected method
- **Performance Metrics**: Comprehensive statistics for comparison
- **Visual Analysis**: Interactive Plotly charts for weights and performance

**Module Information:**
- **Author**: QWIM Development Team
- **Version**: 0.5.1
- **Last Updated**: February 2026
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import polars as pl

from shiny import module, reactive, render, ui
from shinywidgets import output_widget, render_widget
from skfolio.optimization import ObjectiveFunction

from src.dashboard.shiny_utils.reactives_shiny import (
    update_visual_object_in_reactives,  # Store latest chart in shared reactive state
)
from src.dashboard.shiny_utils.utils_reporting import (
    save_portfolio_optimization_skfolio_outputs_to_reactives,  # Persist skfolio for PDF
)
from src.dashboard.shiny_utils.utils_visuals import create_error_figure
from src.models.portfolio_optimization.pkg_skfolio import (
    calc_skfolio_optimization_basic,
    calc_skfolio_optimization_clustering,
    calc_skfolio_optimization_convex,
    calc_skfolio_optimization_ensemble,
)
from src.utils.custom_exceptions_errors_loggers.logger_custom import get_logger


qs: Any = None
try:
    import quantstats as qs

    HAS_QUANTSTATS = True
except ImportError:
    HAS_QUANTSTATS = False


# Module-level logger
_logger = get_logger(__name__)

# Configuration
OUTPUT_DIR = Path("output")

# Optimization method categories and mappings
OPTIMIZATION_CATEGORIES = {
    "basic": "Basic Methods (Equal Weight, Inverse Vol, Random)",
    "convex": "Convex Optimization (Mean-Risk, Risk Parity, etc.)",
    "clustering": "Clustering Methods (HRP, HERC, NCO)",
    "ensemble": "Ensemble Methods (Stacking)",
}

BASIC_METHODS = {
    "BASIC_EQUAL_WEIGHTED": "Equal Weighted (1/N)",
    "BASIC_INVERSE_VOLATILITY": "Inverse Volatility",
    "BASIC_RANDOM_DIRICHLET": "Random (Dirichlet)",
}

CONVEX_METHODS = {
    "CONVEX_MEAN_RISK": "Mean-Risk (Markowitz)",
    "CONVEX_RISK_BUDGETING": "Risk Budgeting (Risk Parity)",
    "CONVEX_MAXIMUM_DIVERSIFICATION": "Maximum Diversification",
    "CONVEX_DISTRIBUTIONALLY_ROBUST_CVAR": "Robust CVaR",
    "CONVEX_BENCHMARK_TRACKING": "Benchmark Tracking",
}

CLUSTERING_METHODS = {
    "CLUSTERING_HIERARCHICAL_RISK_PARITY": "Hierarchical Risk Parity (HRP)",
    "CLUSTERING_HIERARCHICAL_EQUAL_RISK_CONTRIBUTION": "Hierarchical Equal Risk Contribution (HERC)",
    "CLUSTERING_SCHUR_COMPLEMENTARY_ALLOCATION": "Schur Complementary Allocation",
    "CLUSTERING_NESTED": "Nested Clusters Optimization (NCO)",
}

ENSEMBLE_METHODS = {
    "ENSEMBLE_STACKING": "Stacking (Meta-Learning)",
}

OBJECTIVE_FUNCTIONS = {
    "MINIMIZE_RISK": "Minimize Risk",
    "MAXIMIZE_RETURN": "Maximize Return",
    "MAXIMIZE_UTILITY": "Maximize Utility",
    "MAXIMIZE_RATIO": "Maximize Sharpe Ratio",
}


@module.ui
def subtab_portfolios_skfolio_ui(data_utils: dict, data_inputs: dict) -> Any:
    """Create UI for skfolio portfolio optimization comparison subtab.

    Parameters
    ----------
    data_utils : dict
        Dictionary containing utility functions and configurations.
    data_inputs : dict
        Dictionary containing input datasets.

    Returns
    -------
    shiny.ui.div
        Complete UI layout for the skfolio optimization subtab.
    """
    return ui.div(
        ui.h3("Portfolio Optimization with skfolio"),
        ui.p(
            "Compare different portfolio optimization strategies using advanced "
            "methods from the skfolio package.",
            class_="text-muted mb-4",
        ),
        ui.layout_sidebar(
            ui.sidebar(
                ui.h5("⚙️ Optimization Configuration", class_="mb-3"),
                # Time period for returns data
                ui.input_select(
                    "input_ID_tab_portfolios_subtab_skfolio_time_period",
                    "Returns Data Period",
                    {
                        "1y": "Last 1 Year",
                        "3y": "Last 3 Years",
                        "5y": "Last 5 Years",
                        "10y": "Last 10 Years",
                        "all": "All Available Data",
                    },
                    selected="3y",
                ),
                ui.hr(),
                # Portfolio 1 Configuration
                ui.h5("📊 Portfolio 1", class_="mb-3 text-primary"),
                ui.input_select(
                    "input_ID_tab_portfolios_subtab_skfolio_method1_category",
                    "Method Category",
                    OPTIMIZATION_CATEGORIES,
                    selected="basic",
                ),
                ui.output_ui("output_ID_tab_portfolios_subtab_skfolio_method1_select"),
                ui.output_ui("output_ID_tab_portfolios_subtab_skfolio_method1_params"),
                ui.hr(),
                # Portfolio 2 Configuration
                ui.h5("📊 Portfolio 2", class_="mb-3 text-success"),
                ui.input_select(
                    "input_ID_tab_portfolios_subtab_skfolio_method2_category",
                    "Method Category",
                    OPTIMIZATION_CATEGORIES,
                    selected="convex",
                ),
                ui.output_ui("output_ID_tab_portfolios_subtab_skfolio_method2_select"),
                ui.output_ui("output_ID_tab_portfolios_subtab_skfolio_method2_params"),
                ui.hr(),
                # Optimize button
                ui.input_action_button(
                    "input_ID_tab_portfolios_subtab_skfolio_btn_optimize",
                    "🚀 Run Optimization",
                    class_="btn-primary w-100",
                ),
                ui.output_ui("output_ID_tab_portfolios_subtab_skfolio_status"),
                width=350,
                position="left",
            ),
            # Main content area
            ui.div(
                # Weights comparison
                ui.card(
                    ui.card_header("Portfolio Weights Comparison"),
                    output_widget(
                        "output_ID_tab_portfolios_subtab_skfolio_plot_weights",
                        height="500px",
                        width="100%",
                    ),
                    full_screen=True,
                    class_="mb-4",
                ),
                # Performance statistics
                ui.card(
                    ui.card_header("Optimization Results"),
                    ui.output_ui("output_ID_tab_portfolios_subtab_skfolio_results_table"),
                    class_="mb-4",
                ),
                # Quantstats portfolio statistics comparison
                ui.card(
                    ui.card_header("Portfolio Statistics Comparison (quantstats)"),
                    ui.output_ui("output_ID_tab_portfolios_subtab_skfolio_statistics_table"),
                    class_="mb-4",
                ),
                # Comparison of Portfolio Performance
                ui.card(
                    ui.card_header("Comparison of Portfolio Performance"),
                    ui.p(
                        "Simulated portfolio value over the selected time period "
                        "using optimized weights applied to historical ETF prices "
                        "(normalized to base = 100).",
                        class_="text-muted small px-3 pt-2",
                    ),
                    output_widget(
                        "output_ID_tab_portfolios_subtab_skfolio_plot_performance",
                        height="500px",
                        width="100%",
                    ),
                    full_screen=True,
                    class_="mb-4",
                ),
                class_="flex-fill",
            ),
        ),
    )


@module.server
def subtab_portfolios_skfolio_server(
    input: Any,
    output: Any,
    session: Any,
    data_utils: dict,
    data_inputs: dict,
    reactives_shiny: dict,
) -> None:
    """Server logic for skfolio portfolio optimization comparison.

    Parameters
    ----------
    input : Any
        Shiny input object.
    output : Any
        Shiny output object.
    session : Any
        Shiny session object.
    data_utils : dict
        Dictionary containing utility functions.
    data_inputs : dict
        Dictionary containing input datasets.
    reactives_shiny : dict
        Dictionary containing reactive values.
    """
    # Reactive values for storing optimization results
    portfolio1_result: reactive.Value[Any] = reactive.Value(None)
    portfolio2_result: reactive.Value[Any] = reactive.Value(None)
    optimization_error: reactive.Value[str | None] = reactive.Value(None)

    # Dynamic UI: Method selection for Portfolio 1
    @output
    @render.ui
    def output_ID_tab_portfolios_subtab_skfolio_method1_select():
        """Render method selection dropdown for Portfolio 1."""
        category = input.input_ID_tab_portfolios_subtab_skfolio_method1_category()

        if category == "basic":
            methods = BASIC_METHODS
        elif category == "convex":
            methods = CONVEX_METHODS
        elif category == "clustering":
            methods = CLUSTERING_METHODS
        elif category == "ensemble":
            methods = ENSEMBLE_METHODS
        else:
            methods = BASIC_METHODS

        return ui.input_select(
            "input_ID_tab_portfolios_subtab_skfolio_method1_type",
            "Optimization Method",
            methods,
        )

    # Dynamic UI: Method selection for Portfolio 2
    @output
    @render.ui
    def output_ID_tab_portfolios_subtab_skfolio_method2_select():
        """Render method selection dropdown for Portfolio 2."""
        category = input.input_ID_tab_portfolios_subtab_skfolio_method2_category()

        if category == "basic":
            methods = BASIC_METHODS
        elif category == "convex":
            methods = CONVEX_METHODS
        elif category == "clustering":
            methods = CLUSTERING_METHODS
        elif category == "ensemble":
            methods = ENSEMBLE_METHODS
        else:
            methods = CONVEX_METHODS

        return ui.input_select(
            "input_ID_tab_portfolios_subtab_skfolio_method2_type",
            "Optimization Method",
            methods,
        )

    # Dynamic UI: Parameters for Portfolio 1
    @output
    @render.ui
    def output_ID_tab_portfolios_subtab_skfolio_method1_params():
        """Render parameter inputs for Portfolio 1 method."""
        category = input.input_ID_tab_portfolios_subtab_skfolio_method1_category()

        if category == "convex":
            return ui.div(
                ui.input_select(
                    "input_ID_tab_portfolios_subtab_skfolio_method1_objective",
                    "Objective Function",
                    OBJECTIVE_FUNCTIONS,
                    selected="MINIMIZE_RISK",
                ),
                ui.input_numeric(
                    "input_ID_tab_portfolios_subtab_skfolio_method1_risk_aversion",
                    "Risk Aversion",
                    value=1.0,
                    min=0.1,
                    max=10.0,
                    step=0.1,
                ),
                class_="mt-2",
            )
        return ui.div()

    # Dynamic UI: Parameters for Portfolio 2
    @output
    @render.ui
    def output_ID_tab_portfolios_subtab_skfolio_method2_params():
        """Render parameter inputs for Portfolio 2 method."""
        category = input.input_ID_tab_portfolios_subtab_skfolio_method2_category()

        if category == "convex":
            return ui.div(
                ui.input_select(
                    "input_ID_tab_portfolios_subtab_skfolio_method2_objective",
                    "Objective Function",
                    OBJECTIVE_FUNCTIONS,
                    selected="MAXIMIZE_RATIO",
                ),
                ui.input_numeric(
                    "input_ID_tab_portfolios_subtab_skfolio_method2_risk_aversion",
                    "Risk Aversion",
                    value=1.0,
                    min=0.1,
                    max=10.0,
                    step=0.1,
                ),
                class_="mt-2",
            )
        return ui.div()

    # Reactive: Get returns data based on time period
    @reactive.Calc
    def get_returns_data():
        """Get returns data for the selected time period."""
        time_period = input.input_ID_tab_portfolios_subtab_skfolio_time_period()

        # Get ETF price data
        etf_data = data_inputs.get("Time_Series_ETFs")
        if etf_data is None or etf_data.is_empty():
            _logger.warning("ETF time series data not available")
            return None

        # Calculate date range
        end_date = datetime.now(UTC).strftime("%Y-%m-%d")
        if time_period == "1y":
            start_date = (datetime.now(UTC) - timedelta(days=365)).strftime("%Y-%m-%d")
        elif time_period == "3y":
            start_date = (datetime.now(UTC) - timedelta(days=3 * 365)).strftime("%Y-%m-%d")
        elif time_period == "5y":
            start_date = (datetime.now(UTC) - timedelta(days=5 * 365)).strftime("%Y-%m-%d")
        elif time_period == "10y":
            start_date = (datetime.now(UTC) - timedelta(days=10 * 365)).strftime("%Y-%m-%d")
        else:  # "all"
            start_date = etf_data["Date"].min()

        # Filter data
        filtered_data = etf_data.filter(
            (pl.col("Date") >= start_date) & (pl.col("Date") <= end_date),
        )

        # Calculate returns (percent change)
        if filtered_data.is_empty():
            return None

        # Convert to returns
        returns_data = filtered_data.clone()
        for col in returns_data.columns:
            if col != "Date":
                # Calculate percent change: (price[t] - price[t-1]) / price[t-1]
                returns_data = returns_data.with_columns(
                    pl.col(col).pct_change().alias(col),
                )

        # Drop first row (NaN from pct_change)
        return returns_data.slice(1, returns_data.height - 1)

    # Reactive: Run optimization when button clicked
    @reactive.Effect
    @reactive.event(input.input_ID_tab_portfolios_subtab_skfolio_btn_optimize)
    def run_optimization():
        """Execute portfolio optimization for both methods."""
        optimization_error.set(None)
        portfolio1_result.set(None)
        portfolio2_result.set(None)

        # Get returns data
        returns_data = get_returns_data()
        if returns_data is None:
            optimization_error.set("No returns data available for selected period")
            return

        try:
            # Portfolio 1
            _logger.info("Optimizing Portfolio 1")
            method1_category = input.input_ID_tab_portfolios_subtab_skfolio_method1_category()
            method1_type = input.input_ID_tab_portfolios_subtab_skfolio_method1_type()

            if method1_category == "basic":
                portfolio1 = calc_skfolio_optimization_basic(
                    returns_data=returns_data,
                    optimization_type=method1_type,
                    portfolio_name=f"{method1_type}",
                )
            elif method1_category == "convex":
                objective_str = input.input_ID_tab_portfolios_subtab_skfolio_method1_objective()
                objective_func = getattr(ObjectiveFunction, objective_str)
                risk_aversion = input.input_ID_tab_portfolios_subtab_skfolio_method1_risk_aversion()

                portfolio1 = calc_skfolio_optimization_convex(
                    returns_data=returns_data,
                    optimization_type=method1_type,
                    portfolio_name=f"{method1_type}",
                    objective_function=objective_func,
                    risk_aversion=risk_aversion,
                )
            elif method1_category == "clustering":
                portfolio1 = calc_skfolio_optimization_clustering(
                    returns_data=returns_data,
                    optimization_type=method1_type,
                    portfolio_name=f"{method1_type}",
                )
            else:  # ensemble
                portfolio1 = calc_skfolio_optimization_ensemble(
                    returns_data=returns_data,
                    optimization_type=method1_type,
                    portfolio_name=f"{method1_type}",
                )

            portfolio1_result.set(portfolio1)
            _logger.info("Portfolio 1 optimization complete")

            # Portfolio 2
            _logger.info("Optimizing Portfolio 2")
            method2_category = input.input_ID_tab_portfolios_subtab_skfolio_method2_category()
            method2_type = input.input_ID_tab_portfolios_subtab_skfolio_method2_type()

            if method2_category == "basic":
                portfolio2 = calc_skfolio_optimization_basic(
                    returns_data=returns_data,
                    optimization_type=method2_type,
                    portfolio_name=f"{method2_type}",
                )
            elif method2_category == "convex":
                objective_str = input.input_ID_tab_portfolios_subtab_skfolio_method2_objective()
                objective_func = getattr(ObjectiveFunction, objective_str)
                risk_aversion = input.input_ID_tab_portfolios_subtab_skfolio_method2_risk_aversion()

                portfolio2 = calc_skfolio_optimization_convex(
                    returns_data=returns_data,
                    optimization_type=method2_type,
                    portfolio_name=f"{method2_type}",
                    objective_function=objective_func,
                    risk_aversion=risk_aversion,
                )
            elif method2_category == "clustering":
                portfolio2 = calc_skfolio_optimization_clustering(
                    returns_data=returns_data,
                    optimization_type=method2_type,
                    portfolio_name=f"{method2_type}",
                )
            else:  # ensemble
                portfolio2 = calc_skfolio_optimization_ensemble(
                    returns_data=returns_data,
                    optimization_type=method2_type,
                    portfolio_name=f"{method2_type}",
                )

            portfolio2_result.set(portfolio2)
            _logger.info("Portfolio 2 optimization complete")

        except Exception as e:
            error_msg = f"Optimization failed: {e!s}"
            _logger.exception(error_msg)
            optimization_error.set(error_msg)

    # Output: Status messages
    @output
    @render.ui
    def output_ID_tab_portfolios_subtab_skfolio_status():
        """Display optimization status messages."""
        error = optimization_error.get()
        if error:
            return ui.div(
                ui.p(f"❌ {error}", class_="text-danger mt-3 small"),
            )

        p1 = portfolio1_result.get()
        p2 = portfolio2_result.get()

        if p1 is not None and p2 is not None:
            return ui.div(
                ui.p("✅ Optimization complete!", class_="text-success mt-3 small"),
            )

        return ui.div()

    # Output: Weights comparison plot
    @output
    @render_widget  # pyright: ignore[reportArgumentType]  # pyrefly: ignore[bad-specialization]  # go.Figure satisfies Widget protocol at runtime
    def output_ID_tab_portfolios_subtab_skfolio_plot_weights():
        """Render weights comparison plot."""
        p1 = portfolio1_result.get()
        p2 = portfolio2_result.get()

        # Check if optimization has been run
        if p1 is None or p2 is None:
            return create_error_figure(
                "Run optimization to see weights comparison",
                "Click 'Run Optimization' button to compare portfolios",
            )

        try:
            # Get weights from portfolios
            weights1 = p1.get_portfolio_weights()
            weights2 = p2.get_portfolio_weights()

            # Create comparison plot
            import plotly.graph_objects as go

            # Get asset names (exclude Date column)
            assets = [col for col in weights1.columns if col != "Date"]

            # Get weights as lists
            w1_values = [weights1[asset][0] * 100 for asset in assets]  # Convert to percentages
            w2_values = [weights2[asset][0] * 100 for asset in assets]

            fig = go.Figure()

            # Portfolio 1 bars
            fig.add_trace(
                go.Bar(
                    name=p1.get_portfolio_name,
                    x=assets,
                    y=w1_values,
                    marker_color="rgb(55, 83, 109)",
                ),
            )

            # Portfolio 2 bars
            fig.add_trace(
                go.Bar(
                    name=p2.get_portfolio_name,
                    x=assets,
                    y=w2_values,
                    marker_color="rgb(26, 118, 255)",
                ),
            )

            fig.update_layout(
                barmode="group",
                title="Portfolio Weights Comparison",
                xaxis_title="Assets",
                yaxis_title="Weight (%)",
                template="plotly_white",
                hovermode="x unified",
                legend={
                    "orientation": "h",
                    "yanchor": "bottom",
                    "y": 1.02,
                    "xanchor": "right",
                    "x": 1,
                },
            )

            # Store latest figure in shared reactive state for report pipeline
            update_visual_object_in_reactives(
                reactives_shiny,
                "Chart_Skfolio_Weights",
                fig,
            )

            return fig

        except Exception as e:
            _logger.exception("Error creating weights plot")
            return create_error_figure("Error creating plot", str(e))

    # Output: Results table
    @output
    @render.ui
    def output_ID_tab_portfolios_subtab_skfolio_results_table():
        """Render optimization results table."""
        p1 = portfolio1_result.get()
        p2 = portfolio2_result.get()

        if p1 is None or p2 is None:
            return ui.p(
                "Run optimization to see results",
                class_="text-muted text-center p-4",
            )

        try:
            # Get weights as DataFrames
            weights1 = p1.get_portfolio_weights()
            weights2 = p2.get_portfolio_weights()

            # Get asset names
            assets = [col for col in weights1.columns if col != "Date"]

            # Build comparison table
            table_rows = []
            table_rows.append(
                "<tr><th>Asset</th><th>Portfolio 1 (%)</th><th>Portfolio 2 (%)</th><th>Difference</th></tr>",
            )

            for asset in assets:
                w1 = weights1[asset][0] * 100
                w2 = weights2[asset][0] * 100
                diff = w2 - w1

                diff_color = (
                    "text-success" if diff > 0 else "text-danger" if diff < 0 else "text-muted"
                )

                table_rows.append(
                    f"<tr>"
                    f"<td><strong>{asset}</strong></td>"
                    f"<td>{w1:.2f}%</td>"
                    f"<td>{w2:.2f}%</td>"
                    f"<td class='{diff_color}'>{diff:+.2f}%</td>"
                    f"</tr>",
                )

            # Persist report-compatible weights so the PDF export pipeline can
            # populate outputs_skfolio_optimization.json.  Weights stored as
            # decimals (0-1); performance metrics filled in by the performance
            # chart renderer when available, defaulting to 0.0 here.
            try:
                import math

                report_weights = [
                    {
                        "asset": asset,
                        "equal_weighted": float(weights1[asset][0]),
                        "mean_risk": float(weights2[asset][0]),
                    }
                    for asset in assets
                    if not math.isnan(float(weights1[asset][0]))
                    and not math.isnan(float(weights2[asset][0]))
                ]
                save_portfolio_optimization_skfolio_outputs_to_reactives(
                    reactives_shiny,
                    {
                        "weights_comparison": report_weights,
                        "statistics_comparison": [],
                        "performance_summary": {
                            "method1": {
                                "label": str(p1.get_portfolio_name),
                                "annualized_return": 0.0,
                                "volatility": 0.0,
                                "sharpe_ratio": 0.0,
                            },
                            "method2": {
                                "label": str(p2.get_portfolio_name),
                                "annualized_return": 0.0,
                                "volatility": 0.0,
                                "sharpe_ratio": 0.0,
                            },
                        },
                    },
                )
            except Exception as _save_exc:
                _logger.debug(
                    "save_portfolio_optimization_skfolio_outputs_to_reactives (table): %s",
                    _save_exc,
                )

            table_html = f"""
            <div class="table-responsive">
                <table class="table table-sm table-hover">
                    <thead class="table-light">
                        {"".join(table_rows[:1])}
                    </thead>
                    <tbody>
                        {"".join(table_rows[1:])}
                    </tbody>
                </table>
            </div>
            <div class="mt-3 p-3 bg-light rounded">
                <p class="mb-1"><strong>Portfolio 1:</strong> {p1.get_portfolio_name}</p>
                <p class="mb-1"><strong>Portfolio 2:</strong> {p2.get_portfolio_name}</p>
                <p class="mb-0 text-muted small">
                    <i>Number of assets: {len(assets)} | Optimization date: {weights1["Date"][0]}</i>
                </p>
            </div>
            """

            return ui.HTML(table_html)

        except Exception as e:
            _logger.exception("Error creating results table")
            return ui.p(f"Error: {e!s}", class_="text-danger")

    @output
    @render.ui
    def output_ID_tab_portfolios_subtab_skfolio_statistics_table():
        """Render side-by-side quantstats portfolio metrics comparison."""
        p1 = portfolio1_result.get()
        p2 = portfolio2_result.get()

        if p1 is None or p2 is None:
            return ui.p(
                "Run optimization to see portfolio statistics",
                class_="text-muted text-center p-4",
            )

        if not HAS_QUANTSTATS:
            return ui.p(
                "quantstats package is not available in this environment.",
                class_="text-warning text-center p-3",
            )

        try:
            perf1, perf2 = _compute_portfolio_performance()
            stats_rows = _build_quantstats_statistics_rows(perf1, perf2)

            if len(stats_rows) == 0:
                return ui.p(
                    "No statistics data available for selected time period.",
                    class_="text-muted text-center p-3",
                )

            def _fmt_pct(value: float) -> str:
                return f"{value * 100:.2f}%"

            def _fmt_ratio(value: float) -> str:
                return f"{value:.3f}"

            table_rows: list[str] = []
            table_rows.append(
                (
                    "<tr>"
                    "<th>Metric</th>"
                    f"<th>{p1.get_portfolio_name}</th>"
                    f"<th>{p2.get_portfolio_name}</th>"
                    "<th>Difference</th>"
                    "</tr>"
                ),
            )

            for row in stats_rows:
                is_pct = row.get("format") == "pct"
                method1_str = (
                    _fmt_pct(float(row["method1"])) if is_pct else _fmt_ratio(float(row["method1"]))
                )
                method2_str = (
                    _fmt_pct(float(row["method2"])) if is_pct else _fmt_ratio(float(row["method2"]))
                )
                diff_val = float(row["difference"])
                diff_str = f"{diff_val * 100:+.2f}%" if is_pct else f"{diff_val:+.3f}"
                diff_color = (
                    "text-success"
                    if diff_val > 0
                    else "text-danger"
                    if diff_val < 0
                    else "text-muted"
                )

                table_rows.append(
                    (
                        "<tr>"
                        f"<td><strong>{row['metric']}</strong></td>"
                        f"<td>{method1_str}</td>"
                        f"<td>{method2_str}</td>"
                        f"<td class='{diff_color}'>{diff_str}</td>"
                        "</tr>"
                    ),
                )

            table_html = f"""
            <div class="table-responsive">
                <table class="table table-sm table-hover">
                    <thead class="table-light">
                        {"".join(table_rows[:1])}
                    </thead>
                    <tbody>
                        {"".join(table_rows[1:])}
                    </tbody>
                </table>
            </div>
            """
            return ui.HTML(table_html)
        except Exception as e:
            _logger.exception("Error creating quantstats statistics table")
            return ui.p(f"Error: {e!s}", class_="text-danger")

    # =================================================================
    # Comparison of Portfolio Performance
    # =================================================================

    @reactive.Calc
    def _compute_portfolio_performance():
        """Compute normalized portfolio value time series for both optimized portfolios.

        Uses the optimized weights from each portfolio and applies them
        to the historical ETF price data over the selected time period.
        Portfolio values are normalized to a base of 100.

        Returns
        -------
        tuple[pl.DataFrame | None, pl.DataFrame | None]
            Two DataFrames each with columns ``Date`` and ``Value``
            (normalized to 100), or ``None`` when data is unavailable.
        """
        p1 = portfolio1_result.get()
        p2 = portfolio2_result.get()

        if p1 is None or p2 is None:
            return None, None

        # Get raw ETF price data (not returns)
        etf_data = data_inputs.get("Time_Series_ETFs")
        if etf_data is None or etf_data.is_empty():
            return None, None

        # Apply the same time-period filter used by get_returns_data
        time_period = input.input_ID_tab_portfolios_subtab_skfolio_time_period()
        end_date = datetime.now(UTC).strftime("%Y-%m-%d")

        if time_period == "1y":
            start_date = (datetime.now(UTC) - timedelta(days=365)).strftime("%Y-%m-%d")
        elif time_period == "3y":
            start_date = (datetime.now(UTC) - timedelta(days=3 * 365)).strftime("%Y-%m-%d")
        elif time_period == "5y":
            start_date = (datetime.now(UTC) - timedelta(days=5 * 365)).strftime("%Y-%m-%d")
        elif time_period == "10y":
            start_date = (datetime.now(UTC) - timedelta(days=10 * 365)).strftime("%Y-%m-%d")
        else:
            start_date = etf_data["Date"].min()

        prices = etf_data.filter(
            (pl.col("Date") >= start_date) & (pl.col("Date") <= end_date),
        ).sort("Date")

        if prices.is_empty() or prices.height < 2:
            return None, None

        def _value_series(
            portfolio_obj: Any,
            price_df: pl.DataFrame,
        ) -> pl.DataFrame | None:
            """Build a normalised (base-100) value series for *one* portfolio."""
            weights_df = portfolio_obj.get_portfolio_weights()
            assets = [c for c in weights_df.columns if c != "Date"]

            # Keep only assets present in both weights and prices
            common_assets = [a for a in assets if a in price_df.columns]
            if not common_assets:
                return None

            # Build a numpy weight vector aligned with common_assets
            w = np.array(
                [float(weights_df[a][0]) for a in common_assets],
                dtype=np.float64,
            )

            # Extract price matrix (rows = dates, cols = assets)
            price_matrix = price_df.select(common_assets).to_numpy().astype(np.float64)

            # Daily returns matrix
            returns_matrix = np.diff(price_matrix, axis=0) / price_matrix[:-1]

            # Weighted daily portfolio returns
            portfolio_returns = returns_matrix @ w

            # Build cumulative value series normalised to 100
            cum_values = np.empty(len(portfolio_returns) + 1, dtype=np.float64)
            cum_values[0] = 100.0
            for i, r in enumerate(portfolio_returns):
                cum_values[i + 1] = cum_values[i] * (1.0 + r)

            dates = price_df["Date"].to_list()

            return pl.DataFrame({"Date": dates, "Value": cum_values.tolist()})

        perf1 = _value_series(p1, prices)
        perf2 = _value_series(p2, prices)

        return perf1, perf2

    def _build_quantstats_statistics_rows(
        series_1: pl.DataFrame | None,
        series_2: pl.DataFrame | None,
    ) -> list[dict[str, Any]]:
        """Build side-by-side quantstats metrics for two value series."""
        if not HAS_QUANTSTATS:
            return []
        if series_1 is None or series_2 is None:
            return []
        if len(series_1) < 2 or len(series_2) < 2:
            return []

        values_1 = np.array(series_1["Value"].to_list(), dtype=np.float64)
        values_2 = np.array(series_2["Value"].to_list(), dtype=np.float64)

        returns_1 = pd.Series(np.diff(values_1) / values_1[:-1])
        returns_2 = pd.Series(np.diff(values_2) / values_2[:-1])

        def _safe_metric(fn: Any, returns_series: pd.Series) -> float:
            try:
                raw_value = fn(returns_series)
                if isinstance(raw_value, pd.Series):
                    if raw_value.empty:
                        return 0.0
                    value = float(raw_value.iloc[-1])
                else:
                    value = float(raw_value)
                if np.isnan(value) or np.isinf(value):
                    return 0.0
                return value
            except Exception:
                return 0.0

        def _metric_cagr(returns_series: pd.Series) -> Any:
            return qs.stats.cagr(returns_series)

        def _metric_volatility(returns_series: pd.Series) -> Any:
            return qs.stats.volatility(returns_series)

        def _metric_sharpe(returns_series: pd.Series) -> Any:
            return qs.stats.sharpe(returns_series, rf=0.02)

        def _metric_sortino(returns_series: pd.Series) -> Any:
            return qs.stats.sortino(returns_series, rf=0.02)

        def _metric_max_drawdown(returns_series: pd.Series) -> Any:
            return qs.stats.max_drawdown(returns_series)

        def _metric_calmar(returns_series: pd.Series) -> Any:
            return qs.stats.calmar(returns_series)

        metric_specs: list[tuple[str, Any, str]] = [
            ("CAGR", _metric_cagr, "pct"),
            ("Volatility", _metric_volatility, "pct"),
            ("Sharpe Ratio", _metric_sharpe, "ratio"),
            ("Sortino Ratio", _metric_sortino, "ratio"),
            ("Max Drawdown", _metric_max_drawdown, "pct"),
            ("Calmar Ratio", _metric_calmar, "ratio"),
        ]

        rows: list[dict[str, Any]] = []
        for metric_name, metric_fn, metric_format in metric_specs:
            metric_1 = _safe_metric(metric_fn, returns_1)
            metric_2 = _safe_metric(metric_fn, returns_2)
            rows.append(
                {
                    "metric": metric_name,
                    "method1": metric_1,
                    "method2": metric_2,
                    "difference": metric_2 - metric_1,
                    "format": metric_format,
                },
            )

        return rows

    @output
    @render_widget  # pyright: ignore[reportArgumentType]  # pyrefly: ignore[bad-specialization]  # go.Figure satisfies Widget protocol at runtime
    def output_ID_tab_portfolios_subtab_skfolio_plot_performance():
        """Render the performance comparison line chart for both optimized portfolios.

        Displays two normalised (base = 100) portfolio value lines over the
        selected time period, using the weights produced by the optimisation
        step and the underlying ETF price history.

        Returns
        -------
        plotly.graph_objects.Figure
            Interactive line chart comparing the two portfolio strategies.
        """
        p1 = portfolio1_result.get()
        p2 = portfolio2_result.get()

        if p1 is None or p2 is None:
            return create_error_figure(
                "Run optimization to see performance comparison",
                "Click 'Run Optimization' to compare portfolio performance over time",
            )

        try:
            import plotly.graph_objects as go

            perf1, perf2 = _compute_portfolio_performance()

            if perf1 is None and perf2 is None:
                return create_error_figure(
                    "Insufficient Price Data",
                    "Could not compute portfolio values — check that ETF price "
                    "data is available for the selected period.",
                )

            fig = go.Figure()

            if perf1 is not None:
                fig.add_trace(
                    go.Scatter(
                        x=perf1["Date"].to_list(),
                        y=perf1["Value"].to_list(),
                        mode="lines",
                        name=p1.get_portfolio_name,
                        line={"color": "rgb(55, 83, 109)", "width": 2},
                        hovertemplate=("<b>%{x|%Y-%m-%d}</b><br>Value: %{y:.2f}<extra></extra>"),
                    ),
                )

            if perf2 is not None:
                fig.add_trace(
                    go.Scatter(
                        x=perf2["Date"].to_list(),
                        y=perf2["Value"].to_list(),
                        mode="lines",
                        name=p2.get_portfolio_name,
                        line={"color": "rgb(26, 118, 255)", "width": 2},
                        hovertemplate=("<b>%{x|%Y-%m-%d}</b><br>Value: %{y:.2f}<extra></extra>"),
                    ),
                )

            fig.update_layout(
                title="Portfolio Performance Comparison (Normalized, Base = 100)",
                xaxis_title="Date",
                yaxis_title="Portfolio Value",
                template="plotly_white",
                hovermode="x unified",
                legend={
                    "orientation": "h",
                    "yanchor": "bottom",
                    "y": 1.02,
                    "xanchor": "right",
                    "x": 1,
                },
                xaxis={
                    "rangeslider": {"visible": True},
                    "rangeselector": {
                        "buttons": [
                            {"count": 1, "label": "1M", "step": "month", "stepmode": "backward"},
                            {"count": 6, "label": "6M", "step": "month", "stepmode": "backward"},
                            {"count": 1, "label": "YTD", "step": "year", "stepmode": "todate"},
                            {"count": 1, "label": "1Y", "step": "year", "stepmode": "backward"},
                            {"step": "all", "label": "All"},
                        ],
                    },
                },
            )

            # Store latest figure in shared reactive state for report pipeline
            update_visual_object_in_reactives(
                reactives_shiny,
                "Chart_Skfolio_Performance",
                fig,
            )

            # Update performance_summary in the stored skfolio outputs with
            # actual annualised metrics computed from the value series.
            try:

                def _perf_stats(series: pl.DataFrame | None) -> dict:
                    """Compute annualised return, volatility, and Sharpe.

                    normalised (base-100) value series.
                    """
                    if series is None or len(series) < 2:
                        return {"annualized_return": 0.0, "volatility": 0.0, "sharpe_ratio": 0.0}
                    vals = np.array(series["Value"].to_list(), dtype=np.float64)
                    daily_returns = np.diff(vals) / vals[:-1]
                    ann_ret = float((vals[-1] / vals[0]) ** (252.0 / len(daily_returns)) - 1)
                    vol = float(np.std(daily_returns, ddof=1) * np.sqrt(252))
                    sharpe = float((ann_ret - 0.02) / vol) if vol > 1e-12 else 0.0
                    return {"annualized_return": ann_ret, "volatility": vol, "sharpe_ratio": sharpe}

                w1_df = p1.get_portfolio_weights()
                w2_df = p2.get_portfolio_weights()
                assets_rep = [c for c in w1_df.columns if c != "Date"]
                report_weights_perf = [
                    {
                        "asset": a,
                        "equal_weighted": float(w1_df[a][0]),
                        "mean_risk": float(w2_df[a][0]),
                    }
                    for a in assets_rep
                ]
                stats1 = _perf_stats(perf1)
                stats2 = _perf_stats(perf2)
                save_portfolio_optimization_skfolio_outputs_to_reactives(
                    reactives_shiny,
                    {
                        "weights_comparison": report_weights_perf,
                        "statistics_comparison": _build_quantstats_statistics_rows(perf1, perf2),
                        "performance_summary": {
                            "method1": {"label": str(p1.get_portfolio_name), **stats1},
                            "method2": {"label": str(p2.get_portfolio_name), **stats2},
                        },
                    },
                )
            except Exception as _save_exc:
                _logger.debug(
                    "save_portfolio_optimization_skfolio_outputs_to_reactives (perf): %s",
                    _save_exc,
                )

            return fig

        except Exception as e:
            _logger.exception("Error creating performance comparison plot")
            return create_error_figure("Error creating performance plot", str(e))
