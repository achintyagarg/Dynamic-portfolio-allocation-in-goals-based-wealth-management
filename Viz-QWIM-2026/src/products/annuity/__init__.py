"""Annuities Module.

============

This module contains implementations of annuity products:

- :class:`Annuity_Base` — Abstract base class for all annuity types.
- :class:`Annuity_SPIA` — Single Premium Immediate Annuity.
- :class:`Annuity_DIA` — Deferred Income Annuity.
- :class:`Annuity_FIA` — Fixed Indexed Annuity.
- :class:`Annuity_VA` — Variable Annuity.
- :class:`Annuity_RILA` — Registered Index-Linked Annuity.
- :class:`Annuity_Type` — Enum of supported annuity types.
"""

from __future__ import annotations

from .annuity_base import Annuity_Base, Annuity_Payout_Option, Annuity_Type
from .annuity_DIA import Annuity_DIA
from .annuity_FIA import Annuity_FIA
from .annuity_RILA import Annuity_RILA, Crediting_Strategy, Protection_Type
from .annuity_SPIA import Annuity_SPIA
from .annuity_VA import Annuity_VA


__all__ = [
    "Annuity_Base",
    "Annuity_DIA",
    "Annuity_FIA",
    "Annuity_Payout_Option",
    "Annuity_RILA",
    "Annuity_SPIA",
    "Annuity_Type",
    "Annuity_VA",
    "Crediting_Strategy",
    "Protection_Type",
]
