"""RL Efficiency Analysis Tab — how close PPO gets to the DP optimum.

RL Efficiency = PPO Pr[goal] / DP Pr[goal]

Shows:
- Horizontal bar chart per scenario, color-coded by efficiency tier
  (green ≥ 90%, yellow ≥ 70%, red < 70%)
- 90% target line
- Summary table
- Annotation explaining the stressed scenario gap and reward shaping fix
"""

from __future__ import annotations

import json
import pathlib
from typing import Any

import plotly.graph_objects as go
from shiny import module, render, ui
from shinywidgets import output_widget, render_widget

_RESULTS_PATH = pathlib.Path(__file__).parents[4] / "results" / "all_results.json"

_SCENARIO_LABELS = {
    "base":       "Base Case",
    "young":      "Young Investor",
    "retirement": "Retirement",
    "stressed":   "Stressed",
}


def _load_results() -> dict | None:
    if _RESULTS_PATH.exists():
        with open(_RESULTS_PATH) as f:
            return json.load(f)
    return None


def _efficiency_color(eff: float) -> str:
    if eff >= 0.90:
        return "#1a9641"   # green
    elif eff >= 0.70:
        return "#fdae61"   # orange/yellow
    else:
        return "#d7191c"   # red


def _make_efficiency_chart(results: dict) -> go.Figure:
    scenario_keys = list(results.keys())
    labels  = [_SCENARIO_LABELS.get(k, k) for k in scenario_keys]
    ppo_prs = [results[k]["ppo_pr_goal"]  for k in scenario_keys]
    dp_prs  = [results[k]["dp_pr_goal"]   for k in scenario_keys]
    effs    = [results[k]["rl_efficiency"] for k in scenario_keys]
    colors  = [_efficiency_color(e) for e in effs]

    fig = go.Figure()

    # Bars
    fig.add_trace(go.Bar(
        y=labels,
        x=[e * 100 for e in effs],
        orientation="h",
        marker_color=colors,
        text=[f"{e:.1%}" for e in effs],
        textposition="outside",
        name="RL Efficiency",
    ))

    # 90% target line
    fig.add_vline(
        x=90, line_width=2, line_dash="dash", line_color="#2c7bb6",
        annotation_text="90% target",
        annotation_position="top",
        annotation_font_color="#2c7bb6",
    )

    fig.update_layout(
        title="RL Efficiency = PPO Pr[Goal] / DP Pr[Goal]",
        xaxis_title="RL Efficiency (%)",
        xaxis_range=[0, 115],
        yaxis_title="Scenario",
        template="plotly_white",
        height=380,
        showlegend=False,
    )
    return fig


def _make_detail_chart(results: dict) -> go.Figure:
    """Side-by-side PPO vs DP Pr[goal] per scenario."""
    scenario_keys = list(results.keys())
    labels  = [_SCENARIO_LABELS.get(k, k) for k in scenario_keys]
    ppo_prs = [results[k]["ppo_pr_goal"] * 100 for k in scenario_keys]
    dp_prs  = [results[k]["dp_pr_goal"]  * 100 for k in scenario_keys]

    fig = go.Figure()
    fig.add_trace(go.Bar(name="PPO (RL)", x=labels, y=ppo_prs,
                         marker_color="#2c7bb6",
                         text=[f"{v:.1f}%" for v in ppo_prs], textposition="outside"))
    fig.add_trace(go.Bar(name="DP Benchmark", x=labels, y=dp_prs,
                         marker_color="#d7191c", opacity=0.7,
                         text=[f"{v:.1f}%" for v in dp_prs], textposition="outside"))
    fig.update_layout(
        title="PPO vs DP Pr[Goal] Across Scenarios",
        xaxis_title="Scenario",
        yaxis_title="Pr[Reach Goal] (%)",
        yaxis_range=[0, max(max(ppo_prs), max(dp_prs)) + 12],
        barmode="group",
        template="plotly_white",
        height=380,
        legend=dict(x=0.02, y=0.98),
    )
    return fig


def _make_summary_rows(results: dict) -> list[dict]:
    rows = []
    for k, scen in results.items():
        eff  = scen["rl_efficiency"]
        tier = "Good (≥90%)" if eff >= 0.90 else ("Moderate (≥70%)" if eff >= 0.70 else "Needs work (<70%)")
        rows.append({
            "Scenario":        _SCENARIO_LABELS.get(k, k),
            "PPO Pr[Goal]":    f"{scen['ppo_pr_goal']:.1%}",
            "DP Pr[Goal]":     f"{scen['dp_pr_goal']:.1%}",
            "RL Efficiency":   f"{eff:.1%}",
            "Status":          tier,
        })
    return rows


# ── Shiny module ──────────────────────────────────────────────────────────────

@module.ui
def tab_rl_efficiency_ui(data_utils: dict[str, Any], data_inputs: dict[str, Any]) -> Any:
    results = _load_results()

    if results is None:
        return ui.div(
            ui.h3("RL Efficiency Analysis", class_="mb-3 mt-2"),
            ui.div(
                ui.p("Results not yet available. Please run ", ui.code("python run_training.py"),
                     " from the project root to generate ", ui.code("results/all_results.json"), "."),
                class_="alert alert-warning",
            ),
        )

    return ui.div(
        ui.h3("RL Efficiency Analysis", class_="mb-3 mt-2"),
        ui.p(
            "RL Efficiency measures how close the PPO agent gets to the theoretically optimal "
            "Dynamic Programming policy. A value of 100% means PPO matches DP perfectly.",
            class_="text-muted mb-2",
        ),
        ui.div(
            ui.tags.ul(
                ui.tags.li(ui.span("Green", style="color:#1a9641;font-weight:bold"), " — RL Efficiency ≥ 90%  (excellent)"),
                ui.tags.li(ui.span("Orange", style="color:#fdae61;font-weight:bold"), " — 70–89%  (good)"),
                ui.tags.li(ui.span("Red", style="color:#d7191c;font-weight:bold"), " — < 70%  (needs improvement)"),
            ),
            class_="mb-4",
        ),
        ui.layout_columns(
            output_widget("plot_efficiency"),
            output_widget("plot_detail"),
            col_widths=[6, 6],
        ),
        ui.br(),
        ui.h5("Summary Table"),
        ui.output_table("summary_table"),
        ui.br(),
        ui.card(
            ui.card_header("Reward Shaping Fix (Next Step 3)"),
            ui.p(
                "The stressed scenario achieves lower RL efficiency because terminal success is rare "
                "(DP benchmark: 38%), giving the PPO agent almost no reward signal during training. "
                "Fix: add intermediate rewards proportional to W(t)/G, and a bonus when W(t) ≥ 0.5·G. "
                "This has been applied in ", ui.code("Phase_one_fixed.ipynb"), " cell 5 and "
                "in ", ui.code("run_training.py"), ".",
            ),
            class_="border-warning",
        ),
    )


@module.server
def tab_rl_efficiency_server(
    input: Any, output: Any, session: Any,
    data_utils: dict[str, Any],
    data_inputs: dict[str, Any],
    reactives_shiny: Any,
) -> None:
    results = _load_results()

    @render_widget
    def plot_efficiency():
        if results is None:
            fig = go.Figure()
            fig.update_layout(title="No results — run training first", template="plotly_white", height=380)
            return fig
        return _make_efficiency_chart(results)

    @render_widget
    def plot_detail():
        if results is None:
            fig = go.Figure()
            fig.update_layout(title="No results — run training first", template="plotly_white", height=380)
            return fig
        return _make_detail_chart(results)

    @output
    @render.table
    def summary_table():
        import pandas as pd
        if results is None:
            return pd.DataFrame({"Status": ["Run training to see results"]})
        return pd.DataFrame(_make_summary_rows(results))
