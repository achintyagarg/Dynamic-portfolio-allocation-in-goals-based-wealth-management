"""
Clients module for the QWIM Dashboard.

This module handles client information: assets, goals, income, taxes, personal information, etc.

Exports
-------
tab_clients_ui
    Main clients tab UI component.
tab_clients_server
    Main clients tab server logic.
"""

from __future__ import annotations

from .tab_clients import tab_clients_server, tab_clients_ui


__all__ = ["tab_clients_server", "tab_clients_ui"]
