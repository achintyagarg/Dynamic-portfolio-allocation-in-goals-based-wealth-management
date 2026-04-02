"""Executive summary Module for QWIM Dashboard.

Provides executive summary.
"""

from __future__ import annotations

import typing

from typing import Any

from shiny import module, ui


@module.ui
def subtab_executive_summary_ui(
    data_utils: dict[str, Any],
    data_inputs: dict[str, Any],
) -> Any:
    """Create the Executive Summary subtab UI for the Overview tab."""
    _ = data_utils
    _ = data_inputs

    return ui.div(
        ui.h3("Executive Summary", class_="text-primary mb-3"),
        ui.p(
            "Quantitative Wealth and Investment Management (QWIM) is a disciplined "
            "medium- and long-term investing approach focused on improving client "
            "experience and client portfolio outcomes.",
            class_="mb-3",
        ),
        ui.p(
            "The process combines quantitative analytics, portfolio construction, "
            "and continuous monitoring to support evidence-based decisions through "
            "different market cycles.",
            class_="mb-3",
        ),
        ui.p(
            "For clients, this means clearer investment guidance, more consistent "
            "communication, and portfolios managed with a long-horizon perspective "
            "aligned to goals, risk tolerance, and changing life needs.",
            class_="mb-0",
        ),
        class_="card card-body bg-light",
    )


@module.server
def subtab_executive_summary_server(
    input: typing.Any,
    output: typing.Any,
    session: typing.Any,
    data_utils: dict[str, Any],
    data_inputs: dict[str, Any],
    reactives_shiny: dict[str, Any],
) -> dict[str, Any]:
    """Server logic for Executive Summary subtab."""
    _ = input
    _ = output
    _ = session
    _ = data_utils
    _ = data_inputs
    _ = reactives_shiny
    return {"status": "initialized"}


SubTab_Executive_Summary = subtab_executive_summary_ui
