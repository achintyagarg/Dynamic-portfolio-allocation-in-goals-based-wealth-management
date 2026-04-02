from __future__ import annotations


"""Products module for the QWIM Dashboard."""

from .subtab_annuities import subtab_annuities_server, subtab_annuities_ui
from .subtab_insurance_life import (
    subtab_insurance_life_server,
    subtab_insurance_life_ui,
)
from .subtab_insurance_LTC import (
    subtab_insurance_LTC_server,
    subtab_insurance_LTC_ui,
)
from .tab_products import tab_products_server, tab_products_ui


__all__ = [
    "subtab_annuities_server",
    "subtab_annuities_ui",
    "subtab_insurance_LTC_server",
    "subtab_insurance_LTC_ui",
    "subtab_insurance_life_server",
    "subtab_insurance_life_ui",
    "tab_products_server",
    "tab_products_ui",
]
