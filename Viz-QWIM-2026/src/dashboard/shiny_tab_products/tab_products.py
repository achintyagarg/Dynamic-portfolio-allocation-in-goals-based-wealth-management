"""Products Tab Module for QWIM Dashboard.

Provides insurance and annuity product analysis functionality.
The tab contains three subtabs:

- **Annuities** — SPIA, DIA, FIA, VA and RILA input forms.
- **Life Insurance** — Whole Life, Term Life, Universal Life, Variable Life
  and Survivor Life input forms.
- **LTC Insurance** — Traditional LTC, Hybrid Life LTC, and Hybrid Annuity LTC
  input forms.

Author
------
QWIM Team

Version
-------
0.8.0 (2026-03-01)
"""

from __future__ import annotations

import typing

from typing import Any

from shiny import module, ui

from src.utils.custom_exceptions_errors_loggers.logger_custom import get_logger

from .subtab_annuities import subtab_annuities_server, subtab_annuities_ui
from .subtab_insurance_life import (
    subtab_insurance_life_server,
    subtab_insurance_life_ui,
)
from .subtab_insurance_LTC import (
    subtab_insurance_LTC_server,
    subtab_insurance_LTC_ui,
)


#: Module-level logger instance
_logger = get_logger(__name__)


@module.ui
def tab_products_ui(
    data_utils: dict[str, Any],
    data_inputs: dict[str, Any],
) -> Any:
    """Create the Products tab UI with Annuities, Life Insurance, and LTC subtabs.

    Parameters
    ----------
    data_utils : dict[str, Any]
        Utility functions and configuration data.
    data_inputs : dict[str, Any]
        Default input values and validation rules.

    Returns
    -------
    ui.Tag
        Navigation tab-set containing the Annuities, Life Insurance, and
        LTC Insurance subtabs.
    """
    tab_panels_products = [
        ui.nav_panel(
            "Annuities",
            subtab_annuities_ui(
                id="ID_tab_products_subtab_annuities",  # pyright: ignore[reportCallIssue]  # pyrefly: ignore[unexpected-keyword,bad-argument-count]
                data_utils=data_utils,
                data_inputs=data_inputs,
            ),
        ),
        ui.nav_panel(
            "Life Insurance",
            subtab_insurance_life_ui(
                id="ID_tab_products_subtab_insurance_life",  # pyright: ignore[reportCallIssue]  # pyrefly: ignore[unexpected-keyword,bad-argument-count]
                data_utils=data_utils,
                data_inputs=data_inputs,
            ),
        ),
        ui.nav_panel(
            "LTC Insurance",
            subtab_insurance_LTC_ui(
                id="ID_tab_products_subtab_insurance_LTC",  # pyright: ignore[reportCallIssue]  # pyrefly: ignore[unexpected-keyword,bad-argument-count]
                data_utils=data_utils,
                data_inputs=data_inputs,
            ),
        ),
    ]

    return ui.navset_tab(
        *tab_panels_products,
        id="ID_tab_products_tabs_all",
    )


@module.server
def tab_products_server(
    input: typing.Any,
    output: typing.Any,
    session: typing.Any,
    data_utils: dict[str, Any],
    data_inputs: dict[str, Any],
    reactives_shiny: dict[str, Any],
) -> dict[str, Any]:
    """Server logic for the Products tab.

    Initialises and coordinates all product subtab servers.

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
    _logger.info("Initializing Products tab server")

    annuities_server_instance = subtab_annuities_server(
        id="ID_tab_products_subtab_annuities",  # pyright: ignore[reportCallIssue]  # pyrefly: ignore[unexpected-keyword,bad-argument-count]
        data_utils=data_utils,
        data_inputs=data_inputs,
        reactives_shiny=reactives_shiny,
    )

    insurance_life_server_instance = subtab_insurance_life_server(
        id="ID_tab_products_subtab_insurance_life",  # pyright: ignore[reportCallIssue]  # pyrefly: ignore[unexpected-keyword,bad-argument-count]
        data_utils=data_utils,
        data_inputs=data_inputs,
        reactives_shiny=reactives_shiny,
    )

    insurance_LTC_server_instance = subtab_insurance_LTC_server(
        id="ID_tab_products_subtab_insurance_LTC",  # pyright: ignore[reportCallIssue]  # pyrefly: ignore[unexpected-keyword,bad-argument-count]
        data_utils=data_utils,
        data_inputs=data_inputs,
        reactives_shiny=reactives_shiny,
    )

    return {
        "Annuities_Server": annuities_server_instance,
        "Insurance_Life_Server": insurance_life_server_instance,
        "Insurance_LTC_Server": insurance_LTC_server_instance,
    }
