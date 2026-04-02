"""Scenarios module.

==================

This module provides classes for generating daily financial time-series
scenarios used in Monte Carlo simulation, portfolio back-testing, and
scenario-based optimisation.

Classes
-------
Scenarios_Base
    Abstract base class for all scenario generators.
Scenarios_CMA
    Scenarios driven by Capital Market Assumptions.
Scenarios_Distribution
    Scenarios sampled from multivariate statistical distributions.

Enums
-----
Scenario_Data_Type
    Distinguishes price, arithmetic return, log return, or index level.
Frequency_Time_Series
    Observation frequency (daily, weekly, monthly, quarterly, annual).
Asset_Class_Tier
    Tier classification for CMA asset classes.
CMA_Source
    Origin of CMA data (spreadsheet or manual).
Distribution_Type
    Statistical distribution type (normal, lognormal, Student-*t*).
Covariance_Input_Type
    How the covariance structure is supplied.
"""

from src.num_methods.scenarios.scenarios_base import (
    Frequency_Time_Series,
    Scenario_Data_Type,
    Scenarios_Base,
)
from src.num_methods.scenarios.scenarios_CMA import (
    DEFAULT_INDEX_MAP,
    DEFAULT_TIER_MAP,
    TIER_0_ASSET_CLASSES,
    TIER_1_ASSET_CLASSES,
    TIER_2_ASSET_CLASSES,
    Asset_Class_Tier,
    CMA_Source,
    Scenarios_CMA,
)
from src.num_methods.scenarios.scenarios_distrib import (
    Covariance_Input_Type,
    Distribution_Type,
    Scenarios_Distribution,
)


__all__: list[str] = [
    "DEFAULT_INDEX_MAP",
    "DEFAULT_TIER_MAP",
    "TIER_0_ASSET_CLASSES",
    "TIER_1_ASSET_CLASSES",
    "TIER_2_ASSET_CLASSES",
    "Asset_Class_Tier",
    "CMA_Source",
    "Covariance_Input_Type",
    "Distribution_Type",
    "Frequency_Time_Series",
    "Scenario_Data_Type",
    # --- base ---
    "Scenarios_Base",
    # --- CMA ---
    "Scenarios_CMA",
    # --- distribution ---
    "Scenarios_Distribution",
]
