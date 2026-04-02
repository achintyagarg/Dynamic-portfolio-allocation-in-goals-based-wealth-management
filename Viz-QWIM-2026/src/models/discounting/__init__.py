"""
Discounting Models Module.

============

This module contains implementations of discounting models used
for calculating present values of future cash flows.

Classes
-------
Discounting_Model_Base
    Abstract base class defining the discounting model interface.
Discounting_Model_Constant
    Concrete model using a fixed (constant) annual discount rate.
"""

from __future__ import annotations

from .model_discounting_base import Discounting_Model_Base
from .model_discounting_constant import Discounting_Model_Constant


__all__ = [
    "Discounting_Model_Base",
    "Discounting_Model_Constant",
]
