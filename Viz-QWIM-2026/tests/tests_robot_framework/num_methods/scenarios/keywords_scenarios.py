"""Robot Framework keyword library for num_methods.scenarios tests.

Provides keywords for Scenarios_Distribution, Scenarios_CMA and the
Scenarios_Base utilities used in Robot Framework test suites.

Arguments are documented inline using Robot Framework conventions.
"""

from __future__ import annotations

import io
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Project root & sys.path setup
# ---------------------------------------------------------------------------

_PROJECT_ROOT = Path(__file__).resolve().parents[4]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# ---------------------------------------------------------------------------
# Lazy imports guarded by availability flag
# Robot Framework replaces sys.stderr with StringIO; some project modules
# access sys.stderr.buffer at import time. Temporarily restore a real stream.
# ---------------------------------------------------------------------------

_MODULE_IMPORT_AVAILABLE = False

_original_stderr = sys.stderr
if not hasattr(sys.stderr, "buffer"):
    sys.stderr = io.TextIOWrapper(io.BytesIO())

try:
    import numpy as np
    import polars as pl
    from src.num_methods.scenarios.scenarios_distrib import (
        Distribution_Type,
        Scenarios_Distribution,
    )
    from src.num_methods.scenarios.scenarios_CMA import Scenarios_CMA
    _MODULE_IMPORT_AVAILABLE = True
except Exception:  # noqa: BLE001
    pass
finally:
    sys.stderr = _original_stderr

# ---------------------------------------------------------------------------
# Fixed test parameters (small, deterministic)
# ---------------------------------------------------------------------------

_DEFAULT_SEED = 42
_DEFAULT_DAYS = 30
_DEFAULT_COMPONENTS_DISTRIB = ["US_Equity", "Intl_Equity", "US_Bond"]
_DEFAULT_COMPONENTS_CMA = ["US Large Cap", "International Developed", "US Bonds"]

_MEAN = [0.0003, 0.0002, 0.0001]
_COV = [
    [1.0e-4, 2.0e-5, 1.0e-5],
    [2.0e-5, 9.0e-5, 1.5e-5],
    [1.0e-5, 1.5e-5, 5.0e-5],
]
_CORR = [
    [1.0, 0.3, 0.1],
    [0.3, 1.0, 0.2],
    [0.1, 0.2, 1.0],
]
_VOLS = [0.10, 0.12, 0.07]  # annual (will be passed directly)

_CMA_RET = [0.07, 0.06, 0.04]
_CMA_VOL = [0.16, 0.18, 0.06]
_CMA_CORR = [
    [1.0, 0.5, -0.1],
    [0.5, 1.0, -0.2],
    [-0.1, -0.2, 1.0],
]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _require_imports() -> None:
    """Raise RuntimeError when modules could not be imported."""
    if not _MODULE_IMPORT_AVAILABLE:
        raise RuntimeError(
            "Required modules could not be imported. "
            "Check that src/num_methods/scenarios is on sys.path."
        )


def _make_cov_matrix(n: int) -> list[list[float]]:
    """Return a simple identity-like PSD covariance matrix of size n×n."""
    mat = [[0.0] * n for _ in range(n)]
    for i in range(n):
        mat[i][i] = 1e-4 * (i + 1)
    return mat


def _get_mean(n: int) -> list[float]:
    """Return a list of n small positive mean returns."""
    return [0.0003 * (i + 1) for i in range(n)]


# ---------------------------------------------------------------------------
# Construction keywords
# ---------------------------------------------------------------------------

def create_normal_distribution_scenario(
    num_components: int = 3,
    num_days: int = _DEFAULT_DAYS,
    seed: int = _DEFAULT_SEED,
) -> "Scenarios_Distribution":
    """Create a Scenarios_Distribution with Distribution_Type.NORMAL.

    Arguments:
    - num_components -- number of asset components (default 3)
    - num_days       -- number of observation dates (default 30)
    - seed           -- random seed for reproducibility (default 42)

    Returns the Scenarios_Distribution object (not yet generated).
    """
    _require_imports()
    n = int(num_components)
    components = _DEFAULT_COMPONENTS_DISTRIB[:n]
    cov = _make_cov_matrix(n)
    return Scenarios_Distribution(
        names_components=components,
        distribution_type=Distribution_Type.NORMAL,
        mean_returns=_get_mean(n),
        covariance_matrix=np.array(cov, dtype=float),
        num_days=int(num_days),
        random_seed=int(seed),
    )


def create_student_t_scenario(
    num_components: int = 3,
    num_days: int = _DEFAULT_DAYS,
    seed: int = _DEFAULT_SEED,
    dof: float = 5.0,
) -> "Scenarios_Distribution":
    """Create a Scenarios_Distribution with Distribution_Type.STUDENT_T.

    Arguments:
    - num_components -- number of asset components (default 3)
    - num_days       -- number of observation dates (default 30)
    - seed           -- random seed for reproducibility (default 42)
    - dof            -- degrees of freedom for Student-t (default 5.0)

    Returns the Scenarios_Distribution object (not yet generated).
    """
    _require_imports()
    n = int(num_components)
    components = _DEFAULT_COMPONENTS_DISTRIB[:n]
    cov = _make_cov_matrix(n)
    return Scenarios_Distribution(
        names_components=components,
        distribution_type=Distribution_Type.STUDENT_T,
        mean_returns=_get_mean(n),
        covariance_matrix=np.array(cov, dtype=float),
        degrees_of_freedom=float(dof),
        num_days=int(num_days),
        random_seed=int(seed),
    )


def create_lognormal_scenario(
    num_components: int = 3,
    num_days: int = _DEFAULT_DAYS,
    seed: int = _DEFAULT_SEED,
) -> "Scenarios_Distribution":
    """Create a Scenarios_Distribution with Distribution_Type.LOGNORMAL.

    Arguments:
    - num_components -- number of asset components (default 3)
    - num_days       -- number of observation dates (default 30)
    - seed           -- random seed for reproducibility (default 42)

    Returns the Scenarios_Distribution object (not yet generated).
    """
    _require_imports()
    n = int(num_components)
    components = _DEFAULT_COMPONENTS_DISTRIB[:n]
    cov = _make_cov_matrix(n)
    # Lognormal requires positive mean returns
    mean = [0.001 * (i + 1) for i in range(n)]
    return Scenarios_Distribution(
        names_components=components,
        distribution_type=Distribution_Type.LOGNORMAL,
        mean_returns=mean,
        covariance_matrix=np.array(cov, dtype=float),
        num_days=int(num_days),
        random_seed=int(seed),
    )


def create_cma_scenario(
    num_components: int = 3,
    num_days: int = _DEFAULT_DAYS,
    seed: int = _DEFAULT_SEED,
) -> "Scenarios_CMA":
    """Create a Scenarios_CMA scenario object.

    Arguments:
    - num_components -- number of asset classes (default 3)
    - num_days       -- number of observation dates (default 30)
    - seed           -- random seed for reproducibility (default 42)

    Returns the Scenarios_CMA object (not yet generated).
    """
    _require_imports()
    n = int(num_components)
    components = _DEFAULT_COMPONENTS_CMA[:n]
    ret = _CMA_RET[:n]
    vol = _CMA_VOL[:n]
    corr = [row[:n] for row in _CMA_CORR[:n]]
    return Scenarios_CMA(
        names_asset_classes=components,
        expected_returns_annual=ret,
        expected_vols_annual=vol,
        correlation_matrix=np.array(corr, dtype=float),
        num_days=int(num_days),
        random_seed=int(seed),
    )


# ---------------------------------------------------------------------------
# Generate keywords
# ---------------------------------------------------------------------------

def generate_scenario(scenario_obj: object) -> "pl.DataFrame":
    """Call generate() on a scenario object and return the resulting DataFrame.

    Arguments:
    - scenario_obj -- a Scenarios_Distribution or Scenarios_CMA instance

    Returns a polars DataFrame.
    """
    _require_imports()
    return scenario_obj.generate()


def generate_and_store(scenario_obj: object) -> object:
    """Call generate() and return the scenario object (with m_df_scenarios set).

    Arguments:
    - scenario_obj -- scenario instance to generate data for

    Returns the same scenario object after generation.
    """
    _require_imports()
    scenario_obj.generate()
    return scenario_obj


# ---------------------------------------------------------------------------
# DataFrame inspection keywords
# ---------------------------------------------------------------------------

def get_generated_row_count(df: "pl.DataFrame") -> int:
    """Return the number of rows in the generated DataFrame.

    Arguments:
    - df -- polars DataFrame returned by generate()
    """
    _require_imports()
    return len(df)


def get_generated_column_names(df: "pl.DataFrame") -> list[str]:
    """Return the column names of the generated DataFrame.

    Arguments:
    - df -- polars DataFrame returned by generate()
    """
    _require_imports()
    return df.columns


def dataframe_has_no_nulls(df: "pl.DataFrame") -> bool:
    """Return True if the DataFrame contains no null values.

    Arguments:
    - df -- polars DataFrame returned by generate()
    """
    _require_imports()
    return df.null_count().sum_horizontal()[0] == 0


def all_values_positive(df: "pl.DataFrame", components: list[str]) -> bool:
    """Return True if all component column values are > 0.

    Used to verify lognormal scenario outputs are strictly positive.

    Arguments:
    - df         -- generated polars DataFrame
    - components -- list of component column names to check
    """
    _require_imports()
    for col in components:
        min_val = df[col].min()
        if min_val is None or min_val <= 0.0:
            return False
    return True


# ---------------------------------------------------------------------------
# Scenario object inspection keywords
# ---------------------------------------------------------------------------

def validate_scenarios(scenario_obj: object) -> bool:
    """Call validate_scenarios() and return True if no exception is raised.

    Arguments:
    - scenario_obj -- scenario object that has been generated
    """
    _require_imports()
    try:
        scenario_obj.validate_scenarios()
        return True
    except Exception:
        return False


def get_component_series_length(scenario_obj: object, comp_name: str) -> int:
    """Return the length of get_component_series(comp_name).

    Arguments:
    - scenario_obj -- generated scenario object
    - comp_name    -- component column name
    """
    _require_imports()
    series = scenario_obj.get_component_series(comp_name)
    return len(series)


def get_returns_matrix_shape(scenario_obj: object) -> tuple[int, int]:
    """Return (T, K) shape of get_returns_matrix().

    Arguments:
    - scenario_obj -- generated scenario object
    """
    _require_imports()
    mat = scenario_obj.get_returns_matrix()
    return mat.shape


# ---------------------------------------------------------------------------
# Reproducibility keywords
# ---------------------------------------------------------------------------

def two_scenarios_identical(
    df_a: "pl.DataFrame",
    df_b: "pl.DataFrame",
    components: list[str],
) -> bool:
    """Return True if both DataFrames have identical values for each component.

    Arguments:
    - df_a       -- first generated DataFrame
    - df_b       -- second generated DataFrame
    - components -- list of component column names to compare
    """
    _require_imports()
    for col in components:
        if not (df_a[col] == df_b[col]).all():
            return False
    return True


def two_scenarios_differ(
    df_a: "pl.DataFrame",
    df_b: "pl.DataFrame",
    comp: str,
) -> bool:
    """Return True if the DataFrames have at least one differing value.

    Arguments:
    - df_a -- first generated DataFrame
    - df_b -- second generated DataFrame
    - comp -- component column name to compare
    """
    _require_imports()
    return not (df_a[comp] == df_b[comp]).all()


# ---------------------------------------------------------------------------
# CMA-specific keywords
# ---------------------------------------------------------------------------

def get_cma_daily_vols_less_than_annual(cma_obj: "Scenarios_CMA") -> bool:
    """Return True if daily vols are strictly less than annual vols.

    Validates the annualisation formula: daily = annual / sqrt(252).

    Arguments:
    - cma_obj -- a Scenarios_CMA instance
    """
    _require_imports()
    daily = cma_obj.calc_daily_volatilities()
    annual = cma_obj.expected_vols_annual
    annual_arr = np.asarray(annual, dtype=float)
    return all(d < a for d, a in zip(daily, annual_arr))


def cma_num_asset_classes(cma_obj: "Scenarios_CMA") -> int:
    """Return the number of asset classes in the CMA scenario.

    Arguments:
    - cma_obj -- a Scenarios_CMA instance
    """
    _require_imports()
    return cma_obj.m_num_components
