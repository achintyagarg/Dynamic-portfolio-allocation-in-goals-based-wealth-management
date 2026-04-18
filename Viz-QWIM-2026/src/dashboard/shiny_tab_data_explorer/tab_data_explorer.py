"""Data Explorer Tab — GBWM asset universe & efficient frontier.

Shows the three Vanguard index funds used in the GBWM model:
- VTBIX  (Total Bond Market)     mu=4.93%  vol=4.1%
- VGTSX  (Intl Stock Index)      mu=7.70%  vol=19.9%
- VTSMX  (Total Stock Market)    mu=8.86%  vol=19.8%

Displays:
1. Efficient frontier with the 15 portfolio points (action space)
2. Asset risk/return scatter
3. Simulated cumulative wealth paths for the 3 assets
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from shiny import module, ui
from shinywidgets import output_widget, render_widget

# ── Asset parameters (from Das et al. 2024, Table 1) ─────────────────────────
_ASSET_NAMES  = ["VTBIX (Bond)", "VGTSX (Intl)", "VTSMX (US Stock)"]
_MU_ANNUAL    = np.array([0.0493, 0.0770, 0.0886])
_COV_ANNUAL   = np.array([
    [ 0.0017, -0.0017, -0.0021],
    [-0.0017,  0.0396,  0.0309],
    [-0.0021,  0.0309,  0.0392],
])
_SIGMA_ANNUAL = np.sqrt(np.diag(_COV_ANNUAL))

_MU_MIN, _MU_MAX = 0.0526, 0.0886
_N_PORTFOLIOS     = 15


def _build_frontier() -> tuple[np.ndarray, np.ndarray]:
    Si   = np.linalg.inv(_COV_ANNUAL)
    ones = np.ones(3)
    k = _MU_ANNUAL @ Si @ ones
    l = _MU_ANNUAL @ Si @ _MU_ANNUAL
    p = ones        @ Si @ ones
    D = l*p - k**2
    g = (l * Si @ ones        - k * Si @ _MU_ANNUAL) / D
    h = (p * Si @ _MU_ANNUAL  - k * Si @ ones)       / D
    a = float(h @ _COV_ANNUAL @ h)
    b = float(2 * (g @ _COV_ANNUAL @ h))
    c = float(g @ _COV_ANNUAL @ g)
    mu_grid  = np.linspace(_MU_MIN, _MU_MAX, _N_PORTFOLIOS)
    sig_grid = np.sqrt(np.maximum(a*mu_grid**2 + b*mu_grid + c, 0))
    return mu_grid, sig_grid


_PORT_MU, _PORT_SIGMA = _build_frontier()


def _make_frontier_fig() -> go.Figure:
    """Efficient frontier + asset scatter + 15 portfolio points."""
    # Dense frontier curve for smooth line
    mu_dense  = np.linspace(_MU_MIN - 0.002, _MU_MAX + 0.002, 200)
    Si   = np.linalg.inv(_COV_ANNUAL)
    ones = np.ones(3)
    k = _MU_ANNUAL @ Si @ ones
    l = _MU_ANNUAL @ Si @ _MU_ANNUAL
    p = ones        @ Si @ ones
    D = l*p - k**2
    g = (l * Si @ ones        - k * Si @ _MU_ANNUAL) / D
    h = (p * Si @ _MU_ANNUAL  - k * Si @ ones)       / D
    a = float(h @ _COV_ANNUAL @ h)
    b = float(2 * (g @ _COV_ANNUAL @ h))
    c = float(g @ _COV_ANNUAL @ g)
    sig_dense = np.sqrt(np.maximum(a*mu_dense**2 + b*mu_dense + c, 0))

    fig = go.Figure()

    # Efficient frontier curve
    fig.add_trace(go.Scatter(
        x=sig_dense * 100, y=mu_dense * 100,
        mode="lines", name="Efficient Frontier",
        line=dict(color="#2c7bb6", width=2),
    ))

    # 15 portfolio action-space points
    fig.add_trace(go.Scatter(
        x=_PORT_SIGMA * 100, y=_PORT_MU * 100,
        mode="markers+text",
        name="Portfolios (action space)",
        marker=dict(size=8, color="#d7191c", symbol="circle"),
        text=[f"P{i}" for i in range(_N_PORTFOLIOS)],
        textposition="top center",
        textfont=dict(size=9),
    ))

    # Individual assets
    colors = ["#1a9641", "#fdae61", "#d7191c"]
    for i, (name, mu, sig) in enumerate(zip(_ASSET_NAMES, _MU_ANNUAL, _SIGMA_ANNUAL)):
        fig.add_trace(go.Scatter(
            x=[sig * 100], y=[mu * 100],
            mode="markers+text",
            name=name,
            marker=dict(size=14, color=colors[i], symbol="star"),
            text=[name.split()[0]],
            textposition="top right",
        ))

    fig.update_layout(
        title="Efficient Frontier — 15-Portfolio Action Space",
        xaxis_title="Annual Volatility (%)",
        yaxis_title="Annual Return (%)",
        legend=dict(x=0.02, y=0.98),
        template="plotly_white",
        height=480,
    )
    return fig


def _make_simulated_paths_fig(n_years: int = 30, n_paths: int = 50, seed: int = 42) -> go.Figure:
    """Simulated GBM price paths for the 3 assets."""
    rng = np.random.default_rng(seed)
    fig = go.Figure()
    colors_main = ["#2c7bb6", "#d7191c", "#1a9641"]
    colors_fade = ["rgba(44,123,182,0.12)", "rgba(215,25,28,0.12)", "rgba(26,150,65,0.12)"]

    for i, (name, mu, sig) in enumerate(zip(_ASSET_NAMES, _MU_ANNUAL, _SIGMA_ANNUAL)):
        paths = np.zeros((n_paths, n_years + 1))
        paths[:, 0] = 100.0
        for t in range(n_years):
            Z = rng.standard_normal(n_paths)
            paths[:, t+1] = paths[:, t] * np.exp((mu - 0.5*sig**2) + sig*Z)

        years = list(range(n_years + 1))
        for j in range(n_paths):
            fig.add_trace(go.Scatter(
                x=years, y=paths[j],
                mode="lines",
                line=dict(color=colors_fade[i], width=1),
                showlegend=False,
            ))
        # Median path
        fig.add_trace(go.Scatter(
            x=years, y=np.median(paths, axis=0),
            mode="lines",
            name=f"{name} (median)",
            line=dict(color=colors_main[i], width=3),
        ))

    fig.update_layout(
        title=f"Simulated GBM Wealth Paths — {n_paths} Scenarios, 30 Years (W₀=$100)",
        xaxis_title="Year",
        yaxis_title="Portfolio Value ($)",
        template="plotly_white",
        height=420,
        legend=dict(x=0.02, y=0.98),
    )
    return fig


def _make_correlation_fig() -> go.Figure:
    """Correlation heatmap from the paper's covariance matrix."""
    std  = _SIGMA_ANNUAL
    corr = _COV_ANNUAL / np.outer(std, std)
    labels = [n.split("(")[0].strip() for n in _ASSET_NAMES]

    fig = go.Figure(go.Heatmap(
        z=corr,
        x=labels, y=labels,
        colorscale="RdBu",
        zmid=0, zmin=-1, zmax=1,
        text=[[f"{corr[r,c]:.3f}" for c in range(3)] for r in range(3)],
        texttemplate="%{text}",
        textfont=dict(size=14),
        showscale=True,
    ))
    fig.update_layout(
        title="Asset Correlation Matrix (Das et al. 2024)",
        height=350,
        template="plotly_white",
    )
    return fig


# ── Shiny module ──────────────────────────────────────────────────────────────

@module.ui
def tab_data_explorer_ui(data_utils: dict[str, Any], data_inputs: dict[str, Any]) -> Any:
    return ui.div(
        ui.h3("Data Explorer", class_="mb-3 mt-2"),
        ui.p(
            "The GBWM model uses three Vanguard index funds spanning the efficient frontier. "
            "Below are the asset parameters from Das et al. (2024), the 15-portfolio action space, "
            "and simulated GBM wealth paths.",
            class_="text-muted mb-4",
        ),
        # Asset summary cards
        ui.layout_columns(
            ui.card(
                ui.card_header("VTBIX — Total Bond Market"),
                ui.p(ui.strong("Annual return: "), "4.93%"),
                ui.p(ui.strong("Annual volatility: "), "4.1%"),
                ui.p(ui.strong("Role: "), "Conservative anchor (P0)"),
            ),
            ui.card(
                ui.card_header("VGTSX — Intl Stock Index"),
                ui.p(ui.strong("Annual return: "), "7.70%"),
                ui.p(ui.strong("Annual volatility: "), "19.9%"),
                ui.p(ui.strong("Role: "), "International diversifier"),
            ),
            ui.card(
                ui.card_header("VTSMX — US Total Stock Market"),
                ui.p(ui.strong("Annual return: "), "8.86%"),
                ui.p(ui.strong("Annual volatility: "), "19.8%"),
                ui.p(ui.strong("Role: "), "Growth engine (P14)"),
            ),
            col_widths=[4, 4, 4],
        ),
        ui.br(),
        # Charts
        output_widget("plot_frontier"),
        ui.br(),
        ui.layout_columns(
            output_widget("plot_corr"),
            output_widget("plot_paths"),
            col_widths=[4, 8],
        ),
    )


@module.server
def tab_data_explorer_server(
    input: Any, output: Any, session: Any,
    data_utils: dict[str, Any],
    data_inputs: dict[str, Any],
    reactives_shiny: Any,
) -> None:

    @render_widget
    def plot_frontier():
        return _make_frontier_fig()

    @render_widget
    def plot_corr():
        return _make_correlation_fig()

    @render_widget
    def plot_paths():
        return _make_simulated_paths_fig()
