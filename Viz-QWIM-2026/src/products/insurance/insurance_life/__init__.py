"""
Life Insurance Products Module.

================================

This module contains implementations of life insurance products:

- **Insurance_Life_Base** — abstract base class with shared enums
- **Insurance_Life_Whole** — whole life (permanent, guaranteed cash value)
- **Insurance_Life_Term** — term life (temporary coverage, no cash value)
- **Insurance_Life_Universal** — universal life (flexible premium, adjustable DB)
- **Insurance_Life_Variable** — variable life (market-linked sub-accounts)
- **Insurance_Life_Survivor** — survivor / second-to-die (joint last-to-die)
"""

from .insurance_life_base import (
    Death_Benefit_Option,
    Insurance_Life_Base,
    Insurance_Life_Type,
    Premium_Frequency,
    Underwriting_Class,
)
from .insurance_life_survivor import (
    Insurance_Life_Survivor,
    Survivor_Chassis,
)
from .insurance_life_term import (
    Insurance_Life_Term,
    Term_Type,
)
from .insurance_life_universal import (
    Insurance_Life_Universal,
    UL_Variant,
)
from .insurance_life_variable import (
    Insurance_Life_Variable,
    Sub_Account_Type,
)
from .insurance_life_whole import Insurance_Life_Whole

__all__ = [
    # Base & enums
    "Death_Benefit_Option",
    "Insurance_Life_Base",
    "Insurance_Life_Type",
    "Premium_Frequency",
    "Underwriting_Class",
    # Whole life
    "Insurance_Life_Whole",
    # Term life
    "Insurance_Life_Term",
    "Term_Type",
    # Universal life
    "Insurance_Life_Universal",
    "UL_Variant",
    # Variable life
    "Insurance_Life_Variable",
    "Sub_Account_Type",
    # Survivor life
    "Insurance_Life_Survivor",
    "Survivor_Chassis",
]
