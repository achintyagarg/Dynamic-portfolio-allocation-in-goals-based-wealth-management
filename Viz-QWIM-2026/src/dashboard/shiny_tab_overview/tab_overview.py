"""Overview Tab Module.

Overview
"""

from __future__ import annotations

import typing

from typing import Any

from shiny import module, ui

from .subtab_executive_summary import (
    subtab_executive_summary_server,
    subtab_executive_summary_ui,
)


@module.ui
def tab_overview_ui(
    data_utils: dict[str, Any],
    data_inputs: dict[str, Any],
) -> Any:
    """Create the Overview tab UI with Executive Summary subtab."""
    tab_panels_overview = [
        ui.nav_panel(
            "Executive Summary",
            subtab_executive_summary_ui(
                id="ID_tab_overview_subtab_executive_summary",  # pyright: ignore[reportCallIssue]  # pyrefly: ignore[unexpected-keyword, bad-argument-count]
                data_utils=data_utils,
                data_inputs=data_inputs,
            ),
        ),
    ]

    return ui.navset_tab(
        *tab_panels_overview,
        id="ID_tab_overview_tabs_all",
    )


@module.server
def tab_overview_server(
    input: typing.Any,
    output: typing.Any,
    session: typing.Any,
    data_utils: dict[str, Any],
    data_inputs: dict[str, Any],
    reactives_shiny: dict[str, Any],
) -> dict[str, Any]:
    """Server logic for the Overview tab."""
    executive_summary_server_instance = subtab_executive_summary_server(
        id="ID_tab_overview_subtab_executive_summary",  # pyright: ignore[reportCallIssue]  # pyrefly: ignore[unexpected-keyword, bad-argument-count]
        data_utils=data_utils,
        data_inputs=data_inputs,
        reactives_shiny=reactives_shiny,
    )

    return {"Executive_Summary_Server": executive_summary_server_instance}


Tab_Overview = tab_overview_ui
