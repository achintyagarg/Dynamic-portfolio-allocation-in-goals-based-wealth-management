from __future__ import annotations


"""Setup module for the QWIM Dashboard."""

from .subtab_setup_inputs import SubTab_Setup_Inputs  # ty: ignore[unresolved-import]  # pyright: ignore[reportAttributeAccessIssue]  # pyrefly: ignore[missing-module-attribute]
from .tab_setup import Tab_Setup  # ty: ignore[unresolved-import]  # pyright: ignore[reportAttributeAccessIssue]  # pyrefly: ignore[missing-module-attribute]


__all__ = ["SubTab_Setup_Inputs", "Tab_Setup"]
