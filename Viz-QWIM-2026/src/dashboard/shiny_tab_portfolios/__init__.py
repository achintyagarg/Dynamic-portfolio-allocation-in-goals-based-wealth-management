"""
Portfolios module for the QWIM Dashboard.

This module handles capabilities related to QWIM portfolios: comparison, visualization, analysis, reporting.

Exports:
    tab_portfolios_ui: Main portfolios tab UI component
    tab_portfolios_server: Main portfolios tab server logic
"""

from __future__ import annotations

from .tab_portfolios import tab_portfolios_server, tab_portfolios_ui


__all__ = ["tab_portfolios_server", "tab_portfolios_ui"]
