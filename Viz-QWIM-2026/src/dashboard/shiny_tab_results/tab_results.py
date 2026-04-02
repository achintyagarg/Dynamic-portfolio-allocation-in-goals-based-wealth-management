"""Results Tab Module for QWIM Dashboard.

Provides reporting, simulation, and results visualization functionality.
The tab contains two subtabs:

- **Simulation** — Run Monte Carlo simulations on a QWIM portfolio with
  configurable distribution, RNG, and horizon parameters.
- **Reporting** — Generate Typst-based PDF reports combining investor and
  portfolio data from the Clients and Portfolios tabs.

Author
------
QWIM Team

Version
-------
0.2.0 (2026-06-01)
"""

from __future__ import annotations

import typing

from typing import Any

from shiny import module, ui

from src.utils.custom_exceptions_errors_loggers.logger_custom import get_logger

from .subtab_reporting import subtab_reporting_server, subtab_reporting_ui
from .subtab_simulation import subtab_simulation_server, subtab_simulation_ui


#: Module-level logger instance
_logger = get_logger(__name__)


@module.ui
def tab_results_ui(
    data_utils: dict[str, Any],
    data_inputs: dict[str, Any],
) -> Any:
    """Create the Results tab UI with a Reporting subtab.

    Parameters
    ----------
    data_utils : dict[str, Any]
        Utility functions and configuration data.
    data_inputs : dict[str, Any]
        Default input values and validation rules.

    Returns
    -------
    ui.Tag
        Navigation tab-set containing the Reporting subtab.
    """
    tab_panels_results = [
        ui.nav_panel(
            "Simulation",
            subtab_simulation_ui(
                id="ID_tab_results_subtab_simulation",  # pyright: ignore[reportCallIssue]  # pyrefly: ignore[unexpected-keyword, bad-argument-count]
                data_utils=data_utils,
                data_inputs=data_inputs,
            ),
        ),
        ui.nav_panel(
            "Reporting",
            subtab_reporting_ui(
                id="ID_tab_results_subtab_reporting",  # pyright: ignore[reportCallIssue]  # pyrefly: ignore[unexpected-keyword, bad-argument-count]
                data_utils=data_utils,
                data_inputs=data_inputs,
            ),
        ),
    ]

    return ui.navset_tab(
        *tab_panels_results,
        id="ID_tab_results_tabs_all",
    )


@module.server
def tab_results_server(
    input: typing.Any,
    output: typing.Any,
    session: typing.Any,
    data_utils: dict[str, Any],
    data_inputs: dict[str, Any],
    reactives_shiny: dict[str, Any],
) -> dict[str, Any]:
    """Server logic for the Results tab.

    Initialises and coordinates the Reporting subtab server.

    Parameters
    ----------
    input : typing.Any
        Shiny input object.
    output : typing.Any
        Shiny output object.
    session : typing.Any
        Shiny session object.
    data_utils : dict[str, Any]
        Utility functions and configuration data.
    data_inputs : dict[str, Any]
        Default input values and validation rules.
    reactives_shiny : dict[str, Any]
        Shared reactive state dictionary.

    Returns
    -------
    dict[str, Any]
        Dictionary of initialised subtab server instances.
    """
    _logger.info("Initializing Results tab server")

    reporting_server_instance = subtab_reporting_server(
        id="ID_tab_results_subtab_reporting",  # pyright: ignore[reportCallIssue]  # pyrefly: ignore[unexpected-keyword, bad-argument-count]
        data_utils=data_utils,
        data_inputs=data_inputs,
        reactives_shiny=reactives_shiny,
    )

    simulation_server_instance = subtab_simulation_server(
        id="ID_tab_results_subtab_simulation",  # pyright: ignore[reportCallIssue]  # pyrefly: ignore[unexpected-keyword, bad-argument-count]
        data_utils=data_utils,
        data_inputs=data_inputs,
        reactives_shiny=reactives_shiny,
    )

    return {
        "Reporting_Server": reporting_server_instance,
        "Simulation_Server": simulation_server_instance,
    }
