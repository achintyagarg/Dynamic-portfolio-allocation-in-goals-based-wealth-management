"""Investor Simulator Tab — configure a custom investor profile and run Monte Carlo.

The user picks a pre-set persona or manually sets W0, G, T, annual cash flow,
number of episodes, and strategy, then clicks Run Simulation.

Results are shown as a terminal wealth histogram with goal and median lines,
plus summary stat cards (Pr[Goal], Median, Pr[Bankruptcy]).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import plotly.graph_objects as go
from shiny import module, reactive, render, ui
from shinywidgets import output_widget, render_widget

# ── GBWM model constants ──────────────────────────────────────────────────────
_MU_ANNUAL  = np.array([0.0493, 0.0770, 0.0886])
_COV_ANNUAL = np.array([
    [ 0.0017, -0.0017, -0.0021],
    [-0.0017,  0.0396,  0.0309],
    [-0.0021,  0.0309,  0.0392],
])
_MU_MIN, _MU_MAX = 0.0526, 0.0886
_N_PORTFOLIOS     = 15


def _build_frontier():
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

# ── Pre-set investor personas ─────────────────────────────────────────────────
# Each maps to (label, W0, G, T, cf, strategy, description)
_PERSONAS = {
    "custom": {
        "label":    "Custom",
        "W0":       100_000, "G": 200_000, "T": 10, "cf": 0,
        "strategy": "TDF (Target Date Fund)",
        "desc":     "Enter your own values below.",
    },
    "alex_chen": {
        "label":    "Alex Chen — Mid-career couple",
        "W0":       425_000, "G": 1_500_000, "T": 23, "cf": 24_000,
        "strategy": "TDF (Target Date Fund)",
        "desc":     "Age 42, retiring at 65. Combined savings $425K, contributing $24K/yr. "
                    "Essential retirement goal: $1.5M.",
    },
    "young_saver": {
        "label":    "Priya Patel — Young professional",
        "W0":       25_000, "G": 1_000_000, "T": 35, "cf": 12_000,
        "strategy": "Aggressive (P14)",
        "desc":     "Age 30, long horizon with high risk tolerance. "
                    "Saving aggressively toward $1M retirement fund.",
    },
    "near_retirement": {
        "label":    "Robert & Mary Kim — Near retirement",
        "W0":       850_000, "G": 1_200_000, "T": 8, "cf": -45_000,
        "strategy": "Conservative (P0)",
        "desc":     "Age 57, 8 years to retirement. Preserving capital while "
                    "withdrawing $45K/yr for living expenses.",
    },
    "stressed": {
        "label":    "Dana Rivera — Stressed scenario",
        "W0":       100_000, "G": 200_000, "T": 10, "cf": -10_000,
        "strategy": "Greedy Heuristic",
        "desc":     "Withdrawing $10K/yr while trying to double savings. "
                    "This is the hardest scenario — tests RL robustness.",
    },
}

# ── Lightweight GBWM environment ──────────────────────────────────────────────
class _GBWMEnv:
    def __init__(self, W0, G, T, cash_flow_annual, seed=None):
        self.W0         = W0
        self.G          = G
        self.T          = T
        self.cash_flows = np.full(T, cash_flow_annual)
        self.rng        = np.random.default_rng(seed)
        self.W          = W0
        self.t          = 0

    def reset(self, ep_seed=None):
        if ep_seed is not None:
            self.rng = np.random.default_rng(ep_seed)
        self.W = self.W0
        self.t = 0
        return self._state()

    def step(self, action):
        mu    = _PORT_MU[action]
        sig   = _PORT_SIGMA[action]
        cf    = self.cash_flows[self.t]
        W_pre = self.W + cf
        if W_pre <= 0:
            self.W = 0.0
            self.t += 1
            return self._state(), self.t >= self.T
        Z      = self.rng.standard_normal()
        self.W = W_pre * np.exp((mu - 0.5 * sig**2) + sig * Z)
        self.t += 1
        return self._state(), self.t >= self.T

    def _state(self):
        return np.array([
            np.clip(self.W / self.G, 0, 5),
            (self.T - self.t) / self.T,
        ], dtype=np.float32)


# ── Policies ──────────────────────────────────────────────────────────────────
def _tdf(state):
    return int(np.clip(round(2 + state[1] * 10), 0, _N_PORTFOLIOS - 1))

def _aggressive(state):   return _N_PORTFOLIOS - 1
def _conservative(state): return 0
def _moderate(state):     return _N_PORTFOLIOS // 2

def _greedy(state):
    w = state[0]
    if w < 0.7:   return 12
    elif w < 1.0: return 7
    else:         return 2

_POLICIES = {
    "TDF (Target Date Fund)": _tdf,
    "Aggressive (P14)":       _aggressive,
    "Conservative (P0)":      _conservative,
    "Moderate (P7)":          _moderate,
    "Greedy Heuristic":       _greedy,
}


# ── Simulation runner ─────────────────────────────────────────────────────────
def _run_sim(W0, G, T, cf, policy_fn, n_ep, base_seed=42):
    terminal = np.zeros(n_ep)
    for ep in range(n_ep):
        env   = _GBWMEnv(W0, G, T, cf, seed=base_seed + ep)
        state = env.reset()
        done  = False
        while not done:
            state, done = env.step(policy_fn(state))
        terminal[ep] = env.W
    success = terminal >= G
    return {
        "terminal":      terminal,
        "pr_goal":       float(success.mean()),
        "pr_bankruptcy": float((terminal <= 0).mean()),
        "median":        float(np.median(terminal)),
        "p10":           float(np.percentile(terminal, 10)),
        "p90":           float(np.percentile(terminal, 90)),
    }


def _make_chart(res, G, W0, T, cf, strategy_name):
    terminal = res["terminal"]
    n_ep     = len(terminal)

    fig = go.Figure()

    # Histogram
    fig.add_trace(go.Histogram(
        x=terminal,
        nbinsx=60,
        name="Terminal Wealth",
        marker_color="rgba(44,123,182,0.6)",
        marker_line=dict(width=0.5, color="white"),
    ))

    # Goal line
    fig.add_vline(
        x=G, line_width=2, line_dash="dash", line_color="#d7191c",
        annotation_text=f"Goal ${G:,.0f}",
        annotation_position="top right",
        annotation_font_color="#d7191c",
    )

    # Median line
    fig.add_vline(
        x=res["median"], line_width=1.5, line_dash="dot", line_color="#1a9641",
        annotation_text=f"Median ${res['median']:,.0f}",
        annotation_position="top left",
        annotation_font_color="#1a9641",
    )

    # P10 / P90 shaded region
    fig.add_vrect(
        x0=res["p10"], x1=res["p90"],
        fillcolor="rgba(44,123,182,0.08)",
        line_width=0,
        annotation_text="P10–P90",
        annotation_position="top left",
    )

    cf_str = f"+${cf:,.0f}/yr" if cf > 0 else (f"-${abs(cf):,.0f}/yr" if cf < 0 else "no cash flow")
    fig.update_layout(
        title=f"Terminal Wealth Distribution — {strategy_name} | "
              f"W₀=${W0:,.0f}, Goal=${G:,.0f}, T={T}yr, {cf_str} | {n_ep:,} episodes",
        xaxis_title="Terminal Wealth ($)",
        yaxis_title="Count",
        template="plotly_white",
        height=440,
        showlegend=False,
    )
    return fig


# ── Shiny module ──────────────────────────────────────────────────────────────

@module.ui
def tab_investor_simulator_ui(data_utils: dict[str, Any], data_inputs: dict[str, Any]) -> Any:
    persona_choices = {k: v["label"] for k, v in _PERSONAS.items()}
    default = _PERSONAS["alex_chen"]

    return ui.div(
        ui.h3("Investor Simulator", class_="mb-3 mt-2"),
        ui.p(
            "Select a pre-built investor persona or configure a custom profile. "
            "Click Run Simulation to run Monte Carlo episodes under the selected strategy.",
            class_="text-muted mb-3",
        ),
        # Persona selector
        ui.card(
            ui.card_header("Select Investor Persona"),
            ui.layout_columns(
                ui.input_select(
                    "persona", "Persona",
                    choices=persona_choices,
                    selected="alex_chen",
                ),
                ui.output_ui("persona_desc"),
                col_widths=[4, 8],
            ),
            class_="mb-3",
        ),
        ui.layout_sidebar(
            ui.sidebar(
                ui.h5("Investor Profile"),
                ui.input_numeric("w0", "Initial Wealth ($)",   value=default["W0"], min=1_000,  step=10_000),
                ui.input_numeric("g",  "Wealth Goal ($)",      value=default["G"],  min=1_000,  step=10_000),
                ui.input_numeric("t",  "Horizon (years)",      value=default["T"],  min=1, max=40, step=1),
                ui.input_numeric("cf", "Annual Cash Flow ($)", value=default["cf"], step=1_000),
                ui.tags.small(
                    "Positive = contribution (savings), Negative = withdrawal",
                    class_="text-muted",
                ),
                ui.hr(),
                ui.h5("Simulation Settings"),
                ui.input_select(
                    "strategy", "Strategy",
                    choices=list(_POLICIES.keys()),
                    selected=default["strategy"],
                ),
                ui.input_numeric("n_ep", "Episodes", value=2_000, min=100, max=20_000, step=500),
                ui.input_action_button("run_btn", "Run Simulation", class_="btn-primary w-100 mt-2"),
                width=290,
            ),
            ui.div(
                ui.output_ui("stats_cards"),
                ui.br(),
                output_widget("plot_dist"),
            ),
        ),
    )


@module.server
def tab_investor_simulator_server(
    input: Any, output: Any, session: Any,
    data_utils: dict[str, Any],
    data_inputs: dict[str, Any],
    reactives_shiny: Any,
) -> None:

    _result = reactive.value(None)

    # When persona changes, update all input fields
    @reactive.effect
    @reactive.event(input.persona)
    def _load_persona():
        key = input.persona()
        if key not in _PERSONAS:
            return
        p = _PERSONAS[key]
        ui.update_numeric("w0", value=p["W0"])
        ui.update_numeric("g",  value=p["G"])
        ui.update_numeric("t",  value=p["T"])
        ui.update_numeric("cf", value=p["cf"])
        ui.update_select("strategy", selected=p["strategy"])

    @output
    @render.ui
    def persona_desc():
        key = input.persona()
        p = _PERSONAS.get(key, _PERSONAS["custom"])
        return ui.p(p["desc"], class_="text-muted mb-0 mt-1")

    @reactive.effect
    @reactive.event(input.run_btn)
    def _do_sim():
        W0       = float(input.w0())
        G        = float(input.g())
        T        = int(input.t())
        cf       = float(input.cf())
        strategy = input.strategy()
        n_ep     = int(input.n_ep())
        policy   = _POLICIES[strategy]
        res      = _run_sim(W0, G, T, cf, policy, n_ep)
        _result.set((res, W0, G, T, cf, strategy))

    @output
    @render.ui
    def stats_cards():
        val = _result()
        if val is None:
            return ui.div(
                ui.p(
                    "Select a persona and click Run Simulation to see results.",
                    class_="text-muted",
                ),
                # Show the pre-loaded persona values as a hint
                ui.layout_columns(
                    _hint_card("Alex Chen", "Mid-career couple, 23yr horizon"),
                    _hint_card("Priya Patel", "Young professional, 35yr horizon"),
                    _hint_card("Robert & Mary Kim", "Near retirement, 8yr horizon"),
                    _hint_card("Dana Rivera", "Stressed scenario, withdrawals"),
                    col_widths=[3, 3, 3, 3],
                ),
            )
        res, W0, G, T, cf, strategy = val
        pr_g = res["pr_goal"]
        pr_b = res["pr_bankruptcy"]
        med  = res["median"]
        p10  = res["p10"]
        p90  = res["p90"]
        color = "success" if pr_g >= 0.65 else ("warning" if pr_g >= 0.40 else "danger")
        return ui.layout_columns(
            ui.card(
                ui.card_header("Pr[Reach Goal]"),
                ui.h2(f"{pr_g:.1%}", class_=f"text-{color}"),
                ui.p(f"Goal: ${G:,.0f}", class_="text-muted mb-0"),
            ),
            ui.card(
                ui.card_header("Median Terminal Wealth"),
                ui.h2(f"${med:,.0f}"),
                ui.p(f"P10: ${p10:,.0f}  |  P90: ${p90:,.0f}", class_="text-muted mb-0"),
            ),
            ui.card(
                ui.card_header("Pr[Bankruptcy]"),
                ui.h2(f"{pr_b:.1%}",
                      class_="text-danger" if pr_b > 0.05 else "text-success"),
                ui.p("Episodes ending with W(T) ≤ 0", class_="text-muted mb-0"),
            ),
            col_widths=[4, 4, 4],
        )

    @render_widget
    def plot_dist():
        val = _result()
        if val is None:
            fig = go.Figure()
            fig.update_layout(
                title="Select a persona and click Run Simulation",
                template="plotly_white", height=440,
                annotations=[dict(
                    text="Choose a pre-set investor or enter custom values in the sidebar",
                    x=0.5, y=0.5, xref="paper", yref="paper",
                    showarrow=False, font=dict(size=14, color="#999"),
                )],
            )
            return fig
        res, W0, G, T, cf, strategy = val
        return _make_chart(res, G, W0, T, cf, strategy)


def _hint_card(name: str, desc: str):
    return ui.card(
        ui.card_body(
            ui.p(ui.strong(name), class_="mb-1"),
            ui.tags.small(desc, class_="text-muted"),
        ),
        class_="h-100",
    )
