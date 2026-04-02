"""Robot Framework keyword library for portfolio optimization tests.

Exposes pure-Python keywords that the robot file calls by name.
"""

from __future__ import annotations

from datetime import datetime

import numpy as np
import polars as pl

from src.models.portfolio_optimization.utils_portfolio_optimization import (
    portfolio_optimization_feature_type,
    portfolio_optimization_type,
)


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------


_ASSETS_3 = ["VTI", "AGG", "VNQ"]
_ASSETS_5 = ["VTI", "VXUS", "AGG", "VNQ", "GLD"]


def _build_returns(n_days: int, assets: list[str], seed: int = 42) -> pl.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pl.date_range(
        start=datetime(2023, 1, 2),
        end=datetime(2024, 12, 31),
        interval="1d",
        eager=True,
    )[:n_days]
    data: dict[str, object] = {"Date": dates}
    for asset in assets:
        data[asset] = rng.normal(0.0004, 0.015, n_days).tolist()
    return pl.DataFrame(data)


# ---------------------------------------------------------------------------
# Enum keywords
# ---------------------------------------------------------------------------


def get_optimization_type_member_count() -> int:
    """Return the number of members in portfolio_optimization_type."""
    return len(portfolio_optimization_type)


def get_feature_type_member_count() -> int:
    """Return the number of members in portfolio_optimization_feature_type."""
    return len(portfolio_optimization_feature_type)


def get_basic_methods_count() -> int:
    """Return the count of basic optimization methods."""
    return len(portfolio_optimization_type.get_basic_methods())


def get_convex_methods_count() -> int:
    """Return the count of convex optimization methods."""
    return len(portfolio_optimization_type.get_convex_methods())


def get_clustering_methods_count() -> int:
    """Return the count of clustering optimization methods."""
    return len(portfolio_optimization_type.get_clustering_methods())


def get_ensemble_methods_count() -> int:
    """Return the count of ensemble optimization methods."""
    return len(portfolio_optimization_type.get_ensemble_methods())


def get_objectives_count() -> int:
    """Return the count of objective feature types."""
    return len(portfolio_optimization_feature_type.get_objectives())


def get_all_constraints_count() -> int:
    """Return the count of all constraint feature types."""
    return len(portfolio_optimization_feature_type.get_all_constraints())


def get_convex_features_count() -> int:
    """Return the count of convex feature types."""
    return len(portfolio_optimization_feature_type.get_convex_features())


def integer_features_disjoint_from_convex_features() -> bool:
    """Return True when integer and convex feature sets do not overlap."""
    integer_f = set(portfolio_optimization_feature_type.get_integer_features())
    convex_f = set(portfolio_optimization_feature_type.get_convex_features())
    return integer_f.isdisjoint(convex_f)


def all_method_groups_cover_all_members() -> bool:
    """Return True when the four method groups together equal all 13 enum members."""
    combined = (
        portfolio_optimization_type.get_basic_methods()
        + portfolio_optimization_type.get_convex_methods()
        + portfolio_optimization_type.get_clustering_methods()
        + portfolio_optimization_type.get_ensemble_methods()
    )
    return set(combined) == set(portfolio_optimization_type)


def subscript_lookup_works() -> bool:
    """Return True when subscript-by-name returns the correct enum member."""
    member = portfolio_optimization_type["BASIC_EQUAL_WEIGHTED"]
    return member is portfolio_optimization_type.BASIC_EQUAL_WEIGHTED


def enum_value_equals_name_for_all_opt_type_members() -> bool:
    """Return True when all portfolio_optimization_type values equal their names."""
    return all(m.value == m.name for m in portfolio_optimization_type)


def enum_value_equals_name_for_all_feature_type_members() -> bool:
    """Return True when all portfolio_optimization_feature_type values equal their names."""
    return all(m.value == m.name for m in portfolio_optimization_feature_type)


# ---------------------------------------------------------------------------
# Optimization keywords
# ---------------------------------------------------------------------------


def run_skfolio_basic_equal_weighted_3_assets(portfolio_name: str = "RF EW") -> object:
    """Run equal-weighted optimization on 3-asset synthetic data."""
    from src.models.portfolio_optimization.pkg_skfolio import calc_skfolio_optimization_basic

    returns = _build_returns(n_days=100, assets=_ASSETS_3)
    return calc_skfolio_optimization_basic(
        returns_data=returns,
        optimization_type=portfolio_optimization_type.BASIC_EQUAL_WEIGHTED,
        portfolio_name=portfolio_name,
    )


def run_skfolio_basic_inverse_vol_5_assets(portfolio_name: str = "RF InvVol") -> object:
    """Run inverse-volatility optimization on 5-asset synthetic data."""
    from src.models.portfolio_optimization.pkg_skfolio import calc_skfolio_optimization_basic

    returns = _build_returns(n_days=120, assets=_ASSETS_5)
    return calc_skfolio_optimization_basic(
        returns_data=returns,
        optimization_type=portfolio_optimization_type.BASIC_INVERSE_VOLATILITY,
        portfolio_name=portfolio_name,
    )


def run_skfolio_convex_risk_budgeting_5_assets(portfolio_name: str = "RF RP") -> object:
    """Run risk-budgeting (risk-parity) optimization — exercises benchmark_pandas fix."""
    from src.models.portfolio_optimization.pkg_skfolio import calc_skfolio_optimization_convex

    returns = _build_returns(n_days=120, assets=_ASSETS_5)
    return calc_skfolio_optimization_convex(
        returns_data=returns,
        optimization_type=portfolio_optimization_type.CONVEX_RISK_BUDGETING,
        portfolio_name=portfolio_name,
    )


def run_skfolio_clustering_hrp_5_assets(portfolio_name: str = "RF HRP") -> object:
    """Run HRP clustering optimization on 5-asset synthetic data."""
    from src.models.portfolio_optimization.pkg_skfolio import (
        calc_skfolio_optimization_clustering,
    )

    returns = _build_returns(n_days=120, assets=_ASSETS_5)
    return calc_skfolio_optimization_clustering(
        returns_data=returns,
        optimization_type=portfolio_optimization_type.CLUSTERING_HIERARCHICAL_RISK_PARITY,
        portfolio_name=portfolio_name,
    )


def portfolio_is_portfolio_qwim(portfolio: object) -> bool:
    """Return True when the object is a portfolio_QWIM instance."""
    from src.portfolios.portfolio_QWIM import portfolio_QWIM

    return isinstance(portfolio, portfolio_QWIM)


def portfolio_num_components(portfolio: object) -> int:
    """Return the number of components in the portfolio."""
    return portfolio.get_num_components  # type: ignore[union-attr]


def portfolio_weights_sum(portfolio: object) -> float:
    """Return the sum of all portfolio weights (should be ~1.0)."""
    weights_df = portfolio.get_portfolio_weights()  # type: ignore[union-attr]
    components = portfolio.get_portfolio_components  # type: ignore[union-attr]
    return sum(float(weights_df[a][0]) for a in components)


def equal_weighted_weights_correct(portfolio: object) -> bool:
    """Return True when equal-weighted portfolio has correct 1/N weights."""
    from src.portfolios.portfolio_QWIM import portfolio_QWIM

    assert isinstance(portfolio, portfolio_QWIM)
    n = portfolio.get_num_components
    weights_df = portfolio.get_portfolio_weights()
    expected = 1.0 / n
    for asset in portfolio.get_portfolio_components:
        if abs(float(weights_df[asset][0]) - expected) > 1e-9:
            return False
    return True


def all_weights_non_negative(portfolio: object) -> bool:
    """Return True when all portfolio weights are >= 0."""
    weights_df = portfolio.get_portfolio_weights()  # type: ignore[union-attr]
    components = portfolio.get_portfolio_components  # type: ignore[union-attr]
    return all(float(weights_df[a][0]) >= -1e-8 for a in components)


def run_azapy_inverse_volatility_5_assets(portfolio_name: str = "RF InvVol Az") -> object:
    """Run azapy inverse-volatility on 5-asset synthetic data."""
    from src.models.portfolio_optimization.pkg_azapy import calc_azapy_inverse_volatility

    returns = _build_returns(n_days=120, assets=_ASSETS_5)
    return calc_azapy_inverse_volatility(
        returns_data=returns,
        portfolio_name=portfolio_name,
    )
