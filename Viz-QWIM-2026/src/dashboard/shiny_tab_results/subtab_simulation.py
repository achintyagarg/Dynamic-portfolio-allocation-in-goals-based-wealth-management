"""Monte Carlo Simulation Subtab for QWIM Dashboard.

Provides an interactive UI for running Monte Carlo simulations on a
QWIM portfolio.  The user selects ETF components, configures scenario
parameters (distribution type, number of paths, horizon, RNG seed),
and launches the simulation.  Results are presented as a fan-chart
of portfolio value paths with confidence bands plus a summary
statistics table.

Author
------
QWIM Team

Version
-------
0.6.0 (2026-03-01)
"""

from __future__ import annotations

import datetime as dt
import typing

from datetime import datetime
from typing import Any

import numpy as np
import plotly.graph_objects as go
import polars as pl

from shiny import module, reactive, render, ui
from shinywidgets import output_widget, render_widget

from src.dashboard.shiny_utils.reactives_shiny import (
    update_visual_object_in_reactives,  # Store latest chart in shared reactive state
)
from src.models.simulation.model_simulation_standard import (
    DEFAULT_INITIAL_VALUE,
    DEFAULT_NUM_DAYS,
    DEFAULT_NUM_SCENARIOS,
    DEFAULT_RANDOM_SEED,
    Simulation_Standard,
)
from src.num_methods.scenarios.scenarios_distrib import Distribution_Type
from src.utils.custom_exceptions_errors_loggers.logger_custom import get_logger


logger = get_logger(__name__)

# ======================================================================
# Constants
# ======================================================================

#: 12 ETF symbols available in the data
ALL_ETF_SYMBOLS: list[str] = [
    "IVV",
    "IJH",
    "IWM",
    "EFA",
    "EEM",
    "AGG",
    "SPTL",
    "HYG",
    "SPBO",
    "IYR",
    "DBC",
    "GLD",
]

#: Default selected ETFs (first 3)
DEFAULT_SELECTED_ETFS: list[str] = ["IVV", "IJH", "IWM"]

#: Number of default ETFs to pre-select
NUM_DEFAULT_SELECTED: int = 3

#: Mapping of distribution labels → enum values
DISTRIBUTION_CHOICES: dict[str, str] = {
    "normal": "Normal (Gaussian)",
    "lognormal": "Lognormal",
    "student_t": "Student-t",
}

#: RNG type choices
RNG_TYPE_CHOICES: dict[str, str] = {
    "pcg64": "PCG-64 (NumPy default)",
    "mt19937": "Mersenne Twister (MT19937)",
    "philox": "Philox 4×32",
    "sfc64": "SFC-64",
}


# ======================================================================
# Helper — empty Plotly figure
# ======================================================================


def _create_empty_figure(title: str, message: str) -> go.Figure:
    """Return a Plotly figure with a centred annotation and no data."""
    fig = go.Figure()
    fig.add_annotation(
        text=message,
        xref="paper",
        yref="paper",
        x=0.5,
        y=0.5,
        showarrow=False,
        font={"size": 16, "color": "gray"},
    )
    fig.update_layout(
        title=title,
        xaxis={"visible": False},
        yaxis={"visible": False},
        template="plotly_white",
        height=600,
    )
    return fig


def _map_distribution_key(key: str) -> Distribution_Type:
    """Convert UI dropdown key to Distribution_Type enum."""
    mapping: dict[str, Distribution_Type] = {  # pyright: ignore[reportAssignmentType]
        "normal": Distribution_Type.NORMAL,
        "lognormal": Distribution_Type.LOGNORMAL,
        "student_t": Distribution_Type.STUDENT_T,
    }
    if key in mapping:
        return mapping[key]
    return Distribution_Type.NORMAL  # type: ignore[return-value]


def _create_rng(rng_type: str, seed: int) -> np.random.Generator:
    """Create a NumPy Generator with the selected BitGenerator."""
    generators = {
        "pcg64": np.random.PCG64,
        "mt19937": np.random.MT19937,
        "philox": np.random.Philox,
        "sfc64": np.random.SFC64,
    }
    bit_gen_cls = generators.get(rng_type, np.random.PCG64)
    return np.random.Generator(bit_gen_cls(seed))


# ======================================================================
# UI
# ======================================================================


@module.ui
def subtab_simulation_ui(
    data_utils: dict[str, Any],
    data_inputs: dict[str, Any],
) -> ui.Tag:
    """Create the Simulation subtab UI.

    Parameters
    ----------
    data_utils : dict[str, Any]
        Dashboard utility functions.
    data_inputs : dict[str, Any]
        Dashboard input data.

    Returns
    -------
    ui.Tag
        Shiny UI layout for the simulation subtab.
    """
    return ui.div(
        ui.h3("Monte Carlo Simulation"),
        ui.layout_sidebar(
            ui.sidebar(
                # =============================================================
                # ETF COMPONENT SELECTION
                # =============================================================
                ui.h5("Select Components", class_="mb-3"),
                ui.div(
                    ui.input_checkbox(
                        "input_ID_tab_results_subtab_simulation_select_all_components",
                        "Select All ETFs",
                        value=False,
                    ),
                    class_="mb-2",
                ),
                ui.div(
                    ui.output_ui(
                        "output_ID_tab_results_subtab_simulation_component_checkboxes",
                    ),
                    style=(
                        "max-height: 200px; overflow-y: auto; "
                        "border: 1px solid #dee2e6; border-radius: 0.375rem; "
                        "padding: 0.5rem;"
                    ),
                ),
                ui.div(
                    ui.output_text(
                        "output_ID_tab_results_subtab_simulation_selected_components_count",
                    ),
                    class_="mt-1 text-muted small",
                ),
                ui.hr(),
                # =============================================================
                # SIMULATION PARAMETERS
                # =============================================================
                ui.h5("Simulation Parameters", class_="mb-3"),
                ui.input_numeric(
                    "input_ID_tab_results_subtab_simulation_num_scenarios",
                    "Number of Scenarios",
                    value=DEFAULT_NUM_SCENARIOS,
                    min=10,
                    max=100_000,
                    step=100,
                ),
                ui.input_numeric(
                    "input_ID_tab_results_subtab_simulation_num_days",
                    "Simulation Horizon (Trading Days)",
                    value=DEFAULT_NUM_DAYS,
                    min=5,
                    max=2520,
                    step=1,
                ),
                ui.input_date(
                    "input_ID_tab_results_subtab_simulation_start_date",
                    "Simulation Start Date",
                    value=datetime.now(dt.UTC).date().isoformat(),
                ),
                ui.input_numeric(
                    "input_ID_tab_results_subtab_simulation_initial_value",
                    "Initial Portfolio Value ($)",
                    value=DEFAULT_INITIAL_VALUE,
                    min=1.0,
                    max=1_000_000_000.0,
                    step=100.0,
                ),
                ui.hr(),
                # =============================================================
                # DISTRIBUTION & RNG
                # =============================================================
                ui.h5("Distribution & RNG", class_="mb-3"),
                ui.input_select(
                    "input_ID_tab_results_subtab_simulation_distribution_type",
                    "Distribution Type",
                    choices=DISTRIBUTION_CHOICES,
                    selected="normal",
                ),
                ui.panel_conditional(
                    "input.input_ID_tab_results_subtab_simulation_distribution_type === 'student_t'",
                    ui.input_numeric(
                        "input_ID_tab_results_subtab_simulation_degrees_of_freedom",
                        "Degrees of Freedom (v > 2)",
                        value=5.0,
                        min=2.1,
                        max=100.0,
                        step=0.5,
                    ),
                ),
                ui.input_select(
                    "input_ID_tab_results_subtab_simulation_rng_type",
                    "Random Number Generator",
                    choices=RNG_TYPE_CHOICES,
                    selected="pcg64",
                ),
                ui.input_numeric(
                    "input_ID_tab_results_subtab_simulation_seed",
                    "RNG Seed",
                    value=DEFAULT_RANDOM_SEED,
                    min=0,
                    max=2**31 - 1,
                    step=1,
                ),
                ui.hr(),
                # =============================================================
                # RUN BUTTON
                # =============================================================
                ui.input_action_button(
                    "input_ID_tab_results_subtab_simulation_run_btn",
                    "Run Simulation",
                    class_="btn-primary w-100 mb-3",
                ),
                ui.output_text(
                    "output_ID_tab_results_subtab_simulation_status",
                ),
                width=350,
                position="left",
            ),
            # =================================================================
            # MAIN CONTENT
            # =================================================================
            ui.div(
                ui.card(
                    ui.card_header("Portfolio Value Fan Chart"),
                    output_widget(
                        "output_ID_tab_results_subtab_simulation_fan_chart",
                        height="600px",
                        width="100%",
                    ),
                    full_screen=True,
                    class_="mb-4",
                ),
                ui.div(
                    ui.layout_columns(
                        ui.card(
                            ui.card_header("Terminal Value Distribution"),
                            output_widget(
                                "output_ID_tab_results_subtab_simulation_histogram",
                                height="400px",
                                width="100%",
                            ),
                            class_="h-100",
                        ),
                        ui.card(
                            ui.card_header("Summary Statistics"),
                            ui.output_ui(
                                "output_ID_tab_results_subtab_simulation_stats_table",
                            ),
                            class_="h-100",
                        ),
                        col_widths=[6, 6],
                    ),
                    class_="mt-3",
                ),
                class_="flex-fill",
            ),
        ),
    )


# ======================================================================
# Server
# ======================================================================


@module.server
def subtab_simulation_server(
    input: typing.Any,
    output: typing.Any,
    session: typing.Any,
    data_utils: dict[str, Any],
    data_inputs: dict[str, Any],
    reactives_shiny: dict[str, Any],
) -> None:
    """Server logic for the Simulation subtab.

    Parameters
    ----------
    input : typing.Any
        Shiny input object.
    output : typing.Any
        Shiny output object.
    session : typing.Any
        Shiny session object.
    data_utils : dict[str, Any]
        Utility functions.
    data_inputs : dict[str, Any]
        Dashboard input data (must contain ``'ETF_Prices'``
        :class:`pl.DataFrame`).
    reactives_shiny : dict[str, Any]
        Shared reactive state dictionary with standard 4 categories.
    """
    logger.info("Initializing Simulation subtab server")

    # ----- reactive value to store simulation results -----
    simulation_results: reactive.Value[pl.DataFrame | None] = reactive.Value(None)
    simulation_stats: reactive.Value[pl.DataFrame | None] = reactive.Value(None)

    # ------------------------------------------------------------------
    # ETF component helpers (mirror subtab_weights_analysis pattern)
    # ------------------------------------------------------------------

    def _load_etf_price_data() -> pl.DataFrame | None:
        """Try to load ETF price data from data_inputs or CSV."""
        # First try data_inputs dict
        if data_inputs and "ETF_Prices" in data_inputs:
            raw = data_inputs["ETF_Prices"]
            if isinstance(raw, pl.DataFrame) and not raw.is_empty():
                return raw

        # Fallback: load from CSV
        from pathlib import Path

        csv_path = Path(__file__).resolve().parents[3] / "inputs" / "raw" / "data_ETFs.csv"
        if csv_path.exists():
            try:
                df = pl.read_csv(csv_path)
                if "date" in df.columns:
                    df = df.rename({"date": "Date"})
                return df
            except Exception as exc:
                logger.error("Failed to load ETF CSV: %s", exc)
                return None
        return None

    @reactive.calc
    def get_available_etf_components() -> list[str]:
        """Get ETF symbols available in the price data."""
        df = _load_etf_price_data()
        if df is None:
            return ALL_ETF_SYMBOLS
        return [c for c in df.columns if c not in ("Date", "date")]

    @reactive.calc
    def get_selected_etf_components() -> list[str]:
        """Get currently selected ETF components from checkboxes."""
        available = get_available_etf_components()
        if not available:
            return []

        try:
            select_all = input.input_ID_tab_results_subtab_simulation_select_all_components()
        except Exception:
            return available[:NUM_DEFAULT_SELECTED]

        if select_all:
            return available

        selected: list[str] = []
        for comp in available:
            cid = f"input_ID_tab_results_subtab_simulation_component_{comp}"
            try:
                if input[cid]():
                    selected.append(comp)
            except Exception:  # noqa: S112
                continue

        if not selected:
            return available[:NUM_DEFAULT_SELECTED]
        return selected

    # ------------------------------------------------------------------
    # Render component checkboxes (dynamic)
    # ------------------------------------------------------------------

    @output
    @render.ui
    def output_ID_tab_results_subtab_simulation_component_checkboxes() -> ui.Tag:
        """Render ETF component checkboxes."""
        available = get_available_etf_components()
        if not available:
            return ui.div(
                ui.p("No ETF components available", class_="text-muted"),
                class_="p-2",
            )

        boxes: list[ui.Tag] = []
        for idx, comp in enumerate(available):
            cid = f"input_ID_tab_results_subtab_simulation_component_{comp}"
            default_checked = idx < NUM_DEFAULT_SELECTED
            boxes.append(
                ui.div(
                    ui.input_checkbox(cid, comp, value=default_checked),
                    class_="mb-1",
                ),
            )
        return ui.div(*boxes)

    @output
    @render.text
    def output_ID_tab_results_subtab_simulation_selected_components_count() -> str:
        """Show selected / total count."""
        available = get_available_etf_components()
        selected = get_selected_etf_components()
        return f"Selected: {len(selected)} of {len(available)} components"

    # ------------------------------------------------------------------
    # Simulation execution
    # ------------------------------------------------------------------

    @reactive.effect
    @reactive.event(input.input_ID_tab_results_subtab_simulation_run_btn)
    def _run_simulation() -> None:
        """Execute Monte Carlo simulation when the Run button is clicked."""
        selected_components = get_selected_etf_components()
        if not selected_components:
            simulation_results.set(None)
            simulation_stats.set(None)
            return

        # --- gather inputs ---
        num_scenarios = int(
            input.input_ID_tab_results_subtab_simulation_num_scenarios() or DEFAULT_NUM_SCENARIOS,
        )
        num_days = int(
            input.input_ID_tab_results_subtab_simulation_num_days() or DEFAULT_NUM_DAYS,
        )
        initial_value = float(
            input.input_ID_tab_results_subtab_simulation_initial_value() or DEFAULT_INITIAL_VALUE,
        )
        seed = int(
            input.input_ID_tab_results_subtab_simulation_seed() or DEFAULT_RANDOM_SEED,
        )
        dist_key = str(
            input.input_ID_tab_results_subtab_simulation_distribution_type() or "normal",
        )
        distribution_type = _map_distribution_key(dist_key)

        dof = 5.0
        if dist_key == "student_t":
            dof = float(
                input.input_ID_tab_results_subtab_simulation_degrees_of_freedom() or 5.0,
            )

        # Parse start date
        raw_date = input.input_ID_tab_results_subtab_simulation_start_date()
        if isinstance(raw_date, str):
            start_date = dt.date.fromisoformat(raw_date)
        elif isinstance(raw_date, dt.date):
            start_date = raw_date
        else:
            start_date = dt.datetime.now(tz=dt.UTC).date()

        num_k = len(selected_components)

        # --- estimate mean / cov from historical data ---
        etf_data = _load_etf_price_data()
        if etf_data is not None:
            # Filter to selected components + compute returns
            cols_present = [c for c in selected_components if c in etf_data.columns]
            if len(cols_present) < num_k:
                logger.warning(
                    "Some selected ETFs not in data: %s",
                    set(selected_components) - set(cols_present),
                )
            if cols_present:
                prices = etf_data.select(cols_present).to_numpy()
                returns = np.diff(prices, axis=0) / prices[:-1]
                mean_ret = np.mean(returns, axis=0)
                cov_mat = np.cov(returns, rowvar=False)

                # Adjust for lognormal (needs positive means)
                if distribution_type == Distribution_Type.LOGNORMAL:
                    mean_ret = 1.0 + mean_ret

                num_k = len(cols_present)
                selected_components = cols_present
            else:
                mean_ret = np.zeros(num_k, dtype=np.float64)
                cov_mat = np.eye(num_k, dtype=np.float64) * 0.0004
        else:
            mean_ret = np.zeros(num_k, dtype=np.float64)
            cov_mat = np.eye(num_k, dtype=np.float64) * 0.0004

        # Equal weights
        weights = np.ones(num_k, dtype=np.float64) / num_k

        # --- run ---
        try:
            sim = Simulation_Standard(
                names_components=selected_components,
                weights=weights,
                distribution_type=distribution_type,
                mean_returns=mean_ret,
                covariance_matrix=cov_mat,
                initial_value=initial_value,
                num_scenarios=num_scenarios,
                num_days=num_days,
                start_date=start_date,
                random_seed=seed,
                degrees_of_freedom=dof,
                name_simulation="Dashboard MC Simulation",
            )
            results_df = sim.run()
            stats_df = sim.get_summary_statistics()

            simulation_results.set(results_df)
            simulation_stats.set(stats_df)

            # Store in reactives for other tabs
            if "Inner_Variables_Shiny" in reactives_shiny:
                reactives_shiny["Inner_Variables_Shiny"]["Simulation_Results"] = reactive.Value(
                    results_df,
                )
                reactives_shiny["Inner_Variables_Shiny"]["Simulation_Stats"] = reactive.Value(
                    stats_df,
                )

            logger.info(
                "SIMULATION_COMPLETE via dashboard: %d scenarios x %d days",
                num_scenarios,
                num_days,
            )

        except Exception as exc:
            logger.exception("Dashboard simulation failed: %s", exc)
            simulation_results.set(None)
            simulation_stats.set(None)

    # ------------------------------------------------------------------
    # Status text
    # ------------------------------------------------------------------

    @output
    @render.text
    def output_ID_tab_results_subtab_simulation_status() -> str:
        """Show simulation status."""
        results = simulation_results.get()
        if results is None:
            return "Status: Ready — click Run to start simulation"
        scenario_cols = [c for c in results.columns if c != "Date"]
        return f"Status: Completed — {len(scenario_cols)} scenarios, {results.height} time steps"

    # ------------------------------------------------------------------
    # Fan chart
    # ------------------------------------------------------------------

    @output
    @render_widget  # pyright: ignore[reportArgumentType]  # pyrefly: ignore[bad-specialization]  # go.Figure satisfies Widget protocol at runtime
    def output_ID_tab_results_subtab_simulation_fan_chart() -> go.Figure:
        """Render the portfolio value fan chart with confidence bands."""
        stats = simulation_stats.get()
        results = simulation_results.get()

        if stats is None or results is None:
            return _create_empty_figure(
                "Portfolio Value Fan Chart",
                "Click 'Run Simulation' to generate results",
            )

        dates = stats["Date"].to_list()

        fig = go.Figure()

        # 5th-95th percentile band
        fig.add_trace(
            go.Scatter(
                x=dates + dates[::-1],
                y=stats["P95"].to_list() + stats["P5"].to_list()[::-1],
                fill="toself",
                fillcolor="rgba(31, 119, 180, 0.15)",
                line={"width": 0},
                name="5th-95th Percentile",
                hoverinfo="skip",
            ),
        )

        # 25th-75th percentile band
        fig.add_trace(
            go.Scatter(
                x=dates + dates[::-1],
                y=stats["P75"].to_list() + stats["P25"].to_list()[::-1],
                fill="toself",
                fillcolor="rgba(31, 119, 180, 0.30)",
                line={"width": 0},
                name="25th-75th Percentile",
                hoverinfo="skip",
            ),
        )

        # Median
        fig.add_trace(
            go.Scatter(
                x=dates,
                y=stats["Median"].to_list(),
                mode="lines",
                name="Median",
                line={"color": "rgb(31, 119, 180)", "width": 2.5},
                hovertemplate="<b>Median</b><br>Date: %{x}<br>Value: $%{y:,.2f}<extra></extra>",
            ),
        )

        # Mean
        fig.add_trace(
            go.Scatter(
                x=dates,
                y=stats["Mean"].to_list(),
                mode="lines",
                name="Mean",
                line={"color": "rgb(255, 127, 14)", "width": 2, "dash": "dash"},
                hovertemplate="<b>Mean</b><br>Date: %{x}<br>Value: $%{y:,.2f}<extra></extra>",
            ),
        )

        # Sample paths (first 10)
        scenario_cols = [c for c in results.columns if c.startswith("Scenario_")]
        sample_paths = scenario_cols[:10]
        for path_col in sample_paths:
            fig.add_trace(
                go.Scatter(
                    x=dates,
                    y=results[path_col].to_list(),
                    mode="lines",
                    line={"width": 0.5, "color": "rgba(150, 150, 150, 0.3)"},
                    showlegend=False,
                    hoverinfo="skip",
                ),
            )

        # Initial value line
        initial_val = results.select(scenario_cols[0]).to_series()[0]
        fig.add_hline(
            y=initial_val,
            line_dash="dot",
            line_color="gray",
            annotation_text=f"Initial: ${initial_val:,.2f}",
            annotation_position="bottom right",
        )

        fig.update_layout(
            title="Monte Carlo Simulation — Portfolio Value Paths",
            xaxis_title="Date",
            yaxis_title="Portfolio Value ($)",
            hovermode="x unified",
            template="plotly_white",
            height=600,
            legend={
                "orientation": "h",
                "yanchor": "bottom",
                "y": 1.02,
                "xanchor": "right",
                "x": 1,
            },
        )

        fig.update_xaxes(rangeslider_visible=True)

        # Store latest figure in shared reactive state for report pipeline
        update_visual_object_in_reactives(reactives_shiny, "Chart_Simulation_Fan", fig)

        return fig

    # ------------------------------------------------------------------
    # Terminal value histogram
    # ------------------------------------------------------------------

    @output
    @render_widget  # pyright: ignore[reportArgumentType]  # pyrefly: ignore[bad-specialization]  # go.Figure satisfies Widget protocol at runtime
    def output_ID_tab_results_subtab_simulation_histogram() -> go.Figure:
        """Render histogram of terminal portfolio values."""
        results = simulation_results.get()

        if results is None:
            return _create_empty_figure(
                "Terminal Value Distribution",
                "Run a simulation first",
            )

        scenario_cols = [c for c in results.columns if c.startswith("Scenario_")]
        terminal_row = results.tail(1).select(scenario_cols)
        terminal_values = terminal_row.to_numpy().flatten()

        fig = go.Figure()
        fig.add_trace(
            go.Histogram(
                x=terminal_values,
                nbinsx=50,
                marker_color="rgba(31, 119, 180, 0.7)",
                name="Terminal Values",
                hovertemplate="Value: $%{x:,.2f}<br>Count: %{y}<extra></extra>",
            ),
        )

        # Vertical lines for key statistics
        mean_val = float(np.mean(terminal_values))
        median_val = float(np.median(terminal_values))

        fig.add_vline(
            x=mean_val,
            line_dash="dash",
            line_color="orange",
            annotation_text=f"Mean: ${mean_val:,.2f}",
        )
        fig.add_vline(
            x=median_val,
            line_dash="solid",
            line_color="blue",
            annotation_text=f"Median: ${median_val:,.2f}",
        )

        fig.update_layout(
            title="Distribution of Terminal Portfolio Values",
            xaxis_title="Portfolio Value ($)",
            yaxis_title="Frequency",
            template="plotly_white",
            height=400,
            showlegend=False,
        )

        # Store latest figure in shared reactive state for report pipeline
        update_visual_object_in_reactives(reactives_shiny, "Chart_Simulation_Histogram", fig)

        return fig

    # ------------------------------------------------------------------
    # Summary statistics table
    # ------------------------------------------------------------------

    @output
    @render.ui
    def output_ID_tab_results_subtab_simulation_stats_table() -> ui.TagChild:
        """Render summary statistics as an HTML table."""
        stats = simulation_stats.get()
        results = simulation_results.get()

        if stats is None or results is None:
            return ui.div(
                ui.p(
                    "Run a simulation to see statistics.",
                    class_="text-muted text-center p-3",
                ),
            )

        # Terminal row statistics
        scenario_cols = [c for c in results.columns if c.startswith("Scenario_")]
        terminal_values = results.tail(1).select(scenario_cols).to_numpy().flatten()

        initial_val = float(results.select(scenario_cols[0]).to_series()[0])

        rows: list[dict[str, str]] = [
            {"Metric": "Number of Scenarios", "Value": f"{len(scenario_cols):,}"},
            {"Metric": "Simulation Horizon", "Value": f"{results.height} days"},
            {"Metric": "Initial Value", "Value": f"${initial_val:,.2f}"},
            {"Metric": "Mean Terminal Value", "Value": f"${np.mean(terminal_values):,.2f}"},
            {"Metric": "Median Terminal Value", "Value": f"${np.median(terminal_values):,.2f}"},
            {"Metric": "Std Dev", "Value": f"${np.std(terminal_values, ddof=1):,.2f}"},
            {"Metric": "5th Percentile", "Value": f"${np.percentile(terminal_values, 5):,.2f}"},
            {"Metric": "25th Percentile", "Value": f"${np.percentile(terminal_values, 25):,.2f}"},
            {"Metric": "75th Percentile", "Value": f"${np.percentile(terminal_values, 75):,.2f}"},
            {"Metric": "95th Percentile", "Value": f"${np.percentile(terminal_values, 95):,.2f}"},
            {"Metric": "Minimum", "Value": f"${np.min(terminal_values):,.2f}"},
            {"Metric": "Maximum", "Value": f"${np.max(terminal_values):,.2f}"},
            {
                "Metric": "Prob(Loss)",
                "Value": f"{np.mean(terminal_values < initial_val) * 100:.1f}%",
            },
        ]

        # Build HTML table
        header = "<thead><tr><th>Metric</th><th>Value</th></tr></thead>"
        body_rows = "".join(
            f"<tr><td>{r['Metric']}</td><td class='text-end'>{r['Value']}</td></tr>" for r in rows
        )
        table_html = (
            f'<table class="table table-striped table-hover table-sm">'
            f"{header}<tbody>{body_rows}</tbody></table>"
        )

        return ui.HTML(table_html)
