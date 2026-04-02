"""Integration tests for portfolio optimization module.

These tests verify cross-module interactions between the portfolio optimization
wrappers and the Portfolio_QWIM domain objects.

Test Strategy
-------------
- portfolio_optimization functions must produce portfolio_QWIM objects that are
  fully usable downstream: weights accessible, components valid, equal-weight math
  correct.
- Tests cover skfolio and azapy wrappers (optimalportfolios skipped—broken imports).
- All wrappers share the same Portfolio_QWIM output interface; both must satisfy it.
"""

from __future__ import annotations

from datetime import datetime

import numpy as np
import polars as pl
import pytest

from src.models.portfolio_optimization.utils_portfolio_optimization import (
    portfolio_optimization_type,
)
from src.portfolios.portfolio_QWIM import portfolio_QWIM


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_returns(
    n_days: int = 100,
    assets: list[str] | None = None,
    seed: int = 42,
) -> pl.DataFrame:
    """Build a synthetic daily-returns DataFrame."""
    if assets is None:
        assets = ["VTI", "AGG", "VNQ"]
    rng = np.random.default_rng(seed)
    dates = pl.date_range(
        start=datetime(2023, 1, 2),
        end=datetime(2024, 12, 31),
        interval="1d",
        eager=True,
    )[:n_days]
    data = {"Date": dates}
    for asset in assets:
        data[asset] = rng.normal(0.0005, 0.015, n_days).tolist()
    return pl.DataFrame(data)


def _assert_valid_portfolio(portfolio: object, expected_assets: list[str]) -> None:
    """Shared assertions for a valid portfolio_QWIM output."""
    assert isinstance(portfolio, portfolio_QWIM), (
        f"Expected portfolio_QWIM, got {type(portfolio)}"
    )

    # Components present and correct
    components = portfolio.get_portfolio_components
    assert set(components) == set(expected_assets), (
        f"Components mismatch: got {set(components)}, expected {set(expected_assets)}"
    )

    # Weights accessible
    weights_df = portfolio.get_portfolio_weights()
    assert isinstance(weights_df, pl.DataFrame)
    assert "Date" in weights_df.columns

    # Weights sum to 1
    weight_sum = sum(float(weights_df[asset][0]) for asset in expected_assets)
    assert abs(weight_sum - 1.0) < 1e-5, f"Weights sum to {weight_sum:.8f}, expected 1.0"

    # All weights non-negative (long-only)
    for asset in expected_assets:
        w = float(weights_df[asset][0])
        assert w >= -1e-8, f"Negative weight for {asset}: {w:.8f}"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

ASSETS_3 = ["VTI", "AGG", "VNQ"]
ASSETS_5 = ["VTI", "VXUS", "AGG", "VNQ", "GLD"]


@pytest.fixture(scope="module")
def returns_3_assets() -> pl.DataFrame:
    return _build_returns(n_days=120, assets=ASSETS_3, seed=42)


@pytest.fixture(scope="module")
def returns_5_assets() -> pl.DataFrame:
    return _build_returns(n_days=252, assets=ASSETS_5, seed=7)


@pytest.fixture(scope="module")
def benchmark_returns(returns_5_assets: pl.DataFrame) -> pl.DataFrame:
    rng = np.random.default_rng(99)
    n = len(returns_5_assets)
    return pl.DataFrame({
        "Date": returns_5_assets["Date"],
        "SPY": rng.normal(0.0003, 0.012, n).tolist(),
    })


# =============================================================================
# Integration: skfolio outputs → portfolio_QWIM interface
# =============================================================================


@pytest.mark.integration()
class Test_Skfolio_Produces_Valid_Portfolio_QWIM:
    """Verify all skfolio wrappers produce usable portfolio_QWIM objects."""

    def test_basic_equal_weighted_3_assets(self, returns_3_assets: pl.DataFrame) -> None:
        from src.models.portfolio_optimization.pkg_skfolio import (
            calc_skfolio_optimization_basic,
        )

        portfolio = calc_skfolio_optimization_basic(
            returns_data=returns_3_assets,
            optimization_type=portfolio_optimization_type.BASIC_EQUAL_WEIGHTED,
            portfolio_name="EW 3-Asset",
        )
        _assert_valid_portfolio(portfolio, ASSETS_3)

        # Equal-weighted: each weight should be exactly 1/3
        weights_df = portfolio.get_portfolio_weights()
        for asset in ASSETS_3:
            assert abs(float(weights_df[asset][0]) - 1 / 3) < 1e-10, (
                f"{asset}: expected 1/3, got {float(weights_df[asset][0])}"
            )

    def test_basic_inverse_volatility_5_assets(self, returns_5_assets: pl.DataFrame) -> None:
        from src.models.portfolio_optimization.pkg_skfolio import (
            calc_skfolio_optimization_basic,
        )

        portfolio = calc_skfolio_optimization_basic(
            returns_data=returns_5_assets,
            optimization_type=portfolio_optimization_type.BASIC_INVERSE_VOLATILITY,
            portfolio_name="InvVol 5-Asset",
        )
        _assert_valid_portfolio(portfolio, ASSETS_5)

    def test_convex_mean_risk_5_assets(self, returns_5_assets: pl.DataFrame) -> None:
        from skfolio.optimization import ObjectiveFunction

        from src.models.portfolio_optimization.pkg_skfolio import (
            calc_skfolio_optimization_convex,
        )

        portfolio = calc_skfolio_optimization_convex(
            returns_data=returns_5_assets,
            optimization_type=portfolio_optimization_type.CONVEX_MEAN_RISK,
            objective_function=ObjectiveFunction.MINIMIZE_RISK,
            portfolio_name="Min-Risk 5-Asset",
        )
        _assert_valid_portfolio(portfolio, ASSETS_5)

    def test_convex_risk_budgeting_5_assets(self, returns_5_assets: pl.DataFrame) -> None:
        from src.models.portfolio_optimization.pkg_skfolio import (
            calc_skfolio_optimization_convex,
        )

        portfolio = calc_skfolio_optimization_convex(
            returns_data=returns_5_assets,
            optimization_type=portfolio_optimization_type.CONVEX_RISK_BUDGETING,
            portfolio_name="Risk Parity 5-Asset",
        )
        _assert_valid_portfolio(portfolio, ASSETS_5)

    def test_convex_benchmark_tracking_5_assets(
        self,
        returns_5_assets: pl.DataFrame,
        benchmark_returns: pl.DataFrame,
    ) -> None:
        from src.models.portfolio_optimization.pkg_skfolio import (
            calc_skfolio_optimization_convex,
        )

        portfolio = calc_skfolio_optimization_convex(
            returns_data=returns_5_assets,
            optimization_type=portfolio_optimization_type.CONVEX_BENCHMARK_TRACKING,
            benchmark_returns=benchmark_returns,
            portfolio_name="Benchmark Track 5-Asset",
        )
        _assert_valid_portfolio(portfolio, ASSETS_5)

    def test_clustering_hrp_5_assets(self, returns_5_assets: pl.DataFrame) -> None:
        from src.models.portfolio_optimization.pkg_skfolio import (
            calc_skfolio_optimization_clustering,
        )

        portfolio = calc_skfolio_optimization_clustering(
            returns_data=returns_5_assets,
            optimization_type=portfolio_optimization_type.CLUSTERING_HIERARCHICAL_RISK_PARITY,
            portfolio_name="HRP 5-Asset",
        )
        _assert_valid_portfolio(portfolio, ASSETS_5)

    def test_ensemble_stacking_5_assets(self, returns_5_assets: pl.DataFrame) -> None:
        from src.models.portfolio_optimization.pkg_skfolio import (
            calc_skfolio_optimization_ensemble,
        )

        portfolio = calc_skfolio_optimization_ensemble(
            returns_data=returns_5_assets,
            optimization_type=portfolio_optimization_type.ENSEMBLE_STACKING,
            portfolio_name="Stacking 5-Asset",
        )
        _assert_valid_portfolio(portfolio, ASSETS_5)


# =============================================================================
# Integration: azapy outputs → portfolio_QWIM interface
# =============================================================================


@pytest.mark.integration()
class Test_Azapy_Produces_Valid_Portfolio_QWIM:
    """Verify all azapy wrappers produce usable portfolio_QWIM objects."""

    def test_mean_variance_min_risk(self, returns_5_assets: pl.DataFrame) -> None:
        from src.models.portfolio_optimization.pkg_azapy import calc_azapy_mean_variance

        portfolio = calc_azapy_mean_variance(
            returns_data=returns_5_assets,
            rtype="MinRisk",
            portfolio_name="MV MinRisk",
        )
        _assert_valid_portfolio(portfolio, ASSETS_5)

    def test_inverse_volatility(self, returns_5_assets: pl.DataFrame) -> None:
        from src.models.portfolio_optimization.pkg_azapy import calc_azapy_inverse_volatility

        portfolio = calc_azapy_inverse_volatility(
            returns_data=returns_5_assets,
            portfolio_name="InvVol Azapy",
        )
        _assert_valid_portfolio(portfolio, ASSETS_5)

    def test_cvar_sharpe(self, returns_5_assets: pl.DataFrame) -> None:
        from src.models.portfolio_optimization.pkg_azapy import calc_azapy_cvar

        portfolio = calc_azapy_cvar(
            returns_data=returns_5_assets,
            rtype="Sharpe",
            portfolio_name="CVaR Sharpe",
        )
        _assert_valid_portfolio(portfolio, ASSETS_5)


# =============================================================================
# Integration: portfolio_QWIM properties are fully usable downstream
# =============================================================================


@pytest.mark.integration()
class Test_Portfolio_QWIM_Downstream_Usage:
    """After optimization, portfolio_QWIM must be fully usable by downstream code."""

    def test_get_num_components_correct(self, returns_5_assets: pl.DataFrame) -> None:
        from src.models.portfolio_optimization.pkg_skfolio import (
            calc_skfolio_optimization_basic,
        )

        portfolio = calc_skfolio_optimization_basic(
            returns_data=returns_5_assets,
            optimization_type=portfolio_optimization_type.BASIC_EQUAL_WEIGHTED,
        )
        assert portfolio.get_num_components == 5

    def test_weights_frame_has_date_column(self, returns_5_assets: pl.DataFrame) -> None:
        from src.models.portfolio_optimization.pkg_skfolio import (
            calc_skfolio_optimization_basic,
        )

        portfolio = calc_skfolio_optimization_basic(
            returns_data=returns_5_assets,
            optimization_type=portfolio_optimization_type.BASIC_EQUAL_WEIGHTED,
        )
        weights_df = portfolio.get_portfolio_weights()
        assert "Date" in weights_df.columns

    def test_portfolio_name_preserved(self, returns_3_assets: pl.DataFrame) -> None:
        from src.models.portfolio_optimization.pkg_skfolio import (
            calc_skfolio_optimization_basic,
        )

        name = "My Custom Integration Portfolio"
        portfolio = calc_skfolio_optimization_basic(
            returns_data=returns_3_assets,
            optimization_type=portfolio_optimization_type.BASIC_EQUAL_WEIGHTED,
            portfolio_name=name,
        )
        assert portfolio.get_portfolio_name == name

    def test_optimization_date_recorded(self, returns_3_assets: pl.DataFrame) -> None:
        from src.models.portfolio_optimization.pkg_skfolio import (
            calc_skfolio_optimization_basic,
        )

        opt_date = "2024-06-15"
        portfolio = calc_skfolio_optimization_basic(
            returns_data=returns_3_assets,
            optimization_type=portfolio_optimization_type.BASIC_EQUAL_WEIGHTED,
            optimization_date=opt_date,
        )
        weights_df = portfolio.get_portfolio_weights()
        assert opt_date in str(weights_df["Date"][0])

    def test_same_data_different_methods_produce_same_components(
        self,
        returns_5_assets: pl.DataFrame,
    ) -> None:
        """All methods on the same input must produce portfolios with the same assets."""
        from skfolio.optimization import ObjectiveFunction

        from src.models.portfolio_optimization.pkg_skfolio import (
            calc_skfolio_optimization_basic,
            calc_skfolio_optimization_clustering,
            calc_skfolio_optimization_convex,
        )

        portfolios = [
            calc_skfolio_optimization_basic(
                returns_data=returns_5_assets,
                optimization_type=portfolio_optimization_type.BASIC_EQUAL_WEIGHTED,
            ),
            calc_skfolio_optimization_convex(
                returns_data=returns_5_assets,
                optimization_type=portfolio_optimization_type.CONVEX_RISK_BUDGETING,
            ),
            calc_skfolio_optimization_clustering(
                returns_data=returns_5_assets,
                optimization_type=portfolio_optimization_type.CLUSTERING_HIERARCHICAL_RISK_PARITY,
            ),
        ]

        expected_components = set(ASSETS_5)
        for p in portfolios:
            assert set(p.get_portfolio_components) == expected_components


# =============================================================================
# Integration: utils enum types drive wrapper dispatching correctly
# =============================================================================


@pytest.mark.integration()
class Test_Enum_Types_Drive_Dispatching:
    """Verify that portfolio_optimization_type enum members correctly drive method selection.

    Tests that the enum-based dispatch path (aenum→stdlib enum migration) works
    end-to-end: passing enum instances (not strings) to wrappers must produce
    the same result as passing the equivalent string.
    """

    def test_basic_equal_weighted_via_enum_vs_string(
        self, returns_3_assets: pl.DataFrame
    ) -> None:
        from src.models.portfolio_optimization.pkg_skfolio import (
            calc_skfolio_optimization_basic,
        )

        p_enum = calc_skfolio_optimization_basic(
            returns_data=returns_3_assets,
            optimization_type=portfolio_optimization_type.BASIC_EQUAL_WEIGHTED,
            optimization_date="2024-01-01",
        )
        p_str = calc_skfolio_optimization_basic(
            returns_data=returns_3_assets,
            optimization_type="BASIC_EQUAL_WEIGHTED",
            optimization_date="2024-01-01",
        )

        # Both should produce identical weights
        w_enum = p_enum.get_portfolio_weights()
        w_str = p_str.get_portfolio_weights()

        for asset in ASSETS_3:
            assert abs(float(w_enum[asset][0]) - float(w_str[asset][0])) < 1e-10, (
                f"Enum vs string dispatch produced different weights for {asset}"
            )

    def test_clustering_hrp_via_enum_vs_string(self, returns_5_assets: pl.DataFrame) -> None:
        from src.models.portfolio_optimization.pkg_skfolio import (
            calc_skfolio_optimization_clustering,
        )

        p_enum = calc_skfolio_optimization_clustering(
            returns_data=returns_5_assets,
            optimization_type=portfolio_optimization_type.CLUSTERING_HIERARCHICAL_RISK_PARITY,
            optimization_date="2024-01-01",
        )
        p_str = calc_skfolio_optimization_clustering(
            returns_data=returns_5_assets,
            optimization_type="CLUSTERING_HIERARCHICAL_RISK_PARITY",
            optimization_date="2024-01-01",
        )

        w_enum = p_enum.get_portfolio_weights()
        w_str = p_str.get_portfolio_weights()

        for asset in ASSETS_5:
            assert abs(float(w_enum[asset][0]) - float(w_str[asset][0])) < 1e-10
