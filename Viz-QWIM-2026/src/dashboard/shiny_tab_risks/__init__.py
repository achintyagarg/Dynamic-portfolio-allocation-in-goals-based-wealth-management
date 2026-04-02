from __future__ import annotations


"""Risks module for the QWIM Dashboard."""

from .subtab_risks_markets import SubTab_Risks_Markets  # ty: ignore[unresolved-import]  # pyright: ignore[reportAttributeAccessIssue]  # pyrefly: ignore[missing-module-attribute]
from .tab_risks import Tab_Risks  # ty: ignore[unresolved-import]  # pyright: ignore[reportAttributeAccessIssue]  # pyrefly: ignore[missing-module-attribute]


__all__ = ["SubTab_Risks_Markets", "Tab_Risks"]
