"""Regression tests for num_methods.scenarios.

These tests verify that Scenarios_Distribution and Scenarios_CMA outputs
do not change unexpectedly across code changes, package upgrades, or
refactors.

Test Strategy
-------------
Each test:
1. Loads a pre-computed baseline Parquet file from
   ``tests/regression_data/scenarios/``.
2. Re-computes the same output with identical fixed-seed input data
   (canonical fixtures from conftest.py).
3. Compares every value against the baseline within the stated tolerance.

Tolerances
----------
* Deterministic values (dates, shapes, discrete metadata): exact equality.
* Floating-point scenario returns:  tolerance = 1e-10.
* Summary statistics (mean/std/etc.): tolerance = 1e-8.
* Correlation / covariance matrices: tolerance = 1e-10.

Baseline Regeneration
---------------------
If a test fails because of an *intentional* algorithmic change::

    python tests/regression_data/scenarios/generate_baselines.py

Then commit both the code changes and the updated Parquet files together.

Author:
    QWIM Development Team

Version:
    0.1.0

Last Modified:
    2026-02-28
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import polars as pl
import pytest

from tests.tests_regression.num_methods.scenarios.conftest import (  # pyright: ignore[reportMissingImports]
    CMA_ASSET_CLASSES,
    DISTRIB_COMPONENTS,
    NUM_DAYS,
    load_baseline,
)

if TYPE_CHECKING:
    from src.num_methods.scenarios.scenarios_CMA import Scenarios_CMA
    from src.num_methods.scenarios.scenarios_distrib import Scenarios_Distribution


# ---------------------------------------------------------------------------
# Tolerances
# ---------------------------------------------------------------------------
_TOL_EXACT: float = 0.0
_TOL_SCENARIO: float = 1e-10
_TOL_SUMMARY: float = 1e-8
_TOL_MATRIX: float = 1e-10


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _assert_df_close(
    current: pl.DataFrame,
    baseline: pl.DataFrame,
    float_cols: list[str],
    *,
    tolerance: float,
    label: str = "",
) -> None:
    """Assert that float columns of *current* match *baseline* within tolerance.

    Parameters
    ----------
    current : pl.DataFrame
        Re-computed DataFrame.
    baseline : pl.DataFrame
        Baseline DataFrame loaded from Parquet.
    float_cols : list[str]
        Column names to compare numerically.
    tolerance : float
        Absolute tolerance for ``np.testing.assert_allclose``.
    label : str
        Optional label for error messages.
    """
    assert len(current) == len(baseline), (
        f"Row count mismatch {label}: current={len(current)}, "
        f"baseline={len(baseline)}"
    )
    mismatches: list[str] = []
    for col in float_cols:
        cur_arr = current[col].to_numpy()
        bas_arr = baseline[col].to_numpy()
        try:
            np.testing.assert_allclose(cur_arr, bas_arr, atol=tolerance, rtol=0)
        except AssertionError as exc:
            mismatches.append(f"column '{col}': {exc}")
    if mismatches:
        raise AssertionError(
            f"Baseline mismatch {label}:\n" + "\n".join(mismatches)
        )


def _matrix_from_df(df: pl.DataFrame, asset_cols: list[str]) -> np.ndarray:
    """Extract a (K, K) numpy matrix from a Parquet-formatted DataFrame."""
    return df.select(asset_cols).to_numpy()


# ===========================================================================
# 1. Normal distribution — scenario values
# ===========================================================================


@pytest.mark.regression()
class TestNormalScenariosRegression:
    """Normal Scenarios_Distribution — scenario values match baseline."""

    def test_scenario_shape(self, canonical_distrib_normal: "Scenarios_Distribution"):
        """Generated DataFrame has (NUM_DAYS, K+1) shape."""
        df = canonical_distrib_normal.generate()
        assert df.shape == (NUM_DAYS, len(DISTRIB_COMPONENTS) + 1)

    def test_scenario_values_match_baseline(
        self, canonical_distrib_normal: "Scenarios_Distribution"
    ):
        """All component return values match the Parquet baseline."""
        df = canonical_distrib_normal.generate()
        baseline = load_baseline("distrib_normal_scenarios")
        _assert_df_close(
            df,
            baseline,
            DISTRIB_COMPONENTS,
            tolerance=_TOL_SCENARIO,
            label="normal scenarios",
        )

    def test_first_row_matches_baseline(
        self, canonical_distrib_normal: "Scenarios_Distribution"
    ):
        """First row of each component column matches the baseline exactly."""
        df = canonical_distrib_normal.generate()
        baseline = load_baseline("distrib_normal_scenarios")
        for comp in DISTRIB_COMPONENTS:
            assert df[comp][0] == pytest.approx(
                baseline[comp][0], abs=_TOL_SCENARIO
            )

    def test_last_row_matches_baseline(
        self, canonical_distrib_normal: "Scenarios_Distribution"
    ):
        """Last row of each component column matches the baseline exactly."""
        df = canonical_distrib_normal.generate()
        baseline = load_baseline("distrib_normal_scenarios")
        for comp in DISTRIB_COMPONENTS:
            assert df[comp][-1] == pytest.approx(
                baseline[comp][-1], abs=_TOL_SCENARIO
            )

    def test_dates_match_baseline(
        self, canonical_distrib_normal: "Scenarios_Distribution"
    ):
        """Date column matches the baseline date sequence."""
        df = canonical_distrib_normal.generate()
        baseline = load_baseline("distrib_normal_scenarios")
        assert df["Date"].to_list() == baseline["Date"].to_list()


# ===========================================================================
# 2. Normal distribution — summary statistics
# ===========================================================================


@pytest.mark.regression()
class TestNormalSummaryStatsRegression:
    """Normal Scenarios_Distribution — summary stats match baseline."""

    def test_summary_stats_shape(
        self, canonical_distrib_normal: "Scenarios_Distribution"
    ):
        """calc_summary_statistics returns one row per component."""
        canonical_distrib_normal.generate()
        df = canonical_distrib_normal.calc_summary_statistics()
        assert len(df) == len(DISTRIB_COMPONENTS)

    def test_mean_matches_baseline(
        self, canonical_distrib_normal: "Scenarios_Distribution"
    ):
        """Component mean values match the baseline."""
        canonical_distrib_normal.generate()
        df = canonical_distrib_normal.calc_summary_statistics()
        baseline = load_baseline("distrib_normal_summary_stats")
        cur_means = df.sort("component")["mean"].to_numpy()
        base_means = baseline.sort("component")["mean"].to_numpy()
        np.testing.assert_allclose(cur_means, base_means, atol=_TOL_SUMMARY, rtol=0)

    def test_std_matches_baseline(
        self, canonical_distrib_normal: "Scenarios_Distribution"
    ):
        """Component standard deviation values match the baseline."""
        canonical_distrib_normal.generate()
        df = canonical_distrib_normal.calc_summary_statistics()
        baseline = load_baseline("distrib_normal_summary_stats")
        cur_std = df.sort("component")["std"].to_numpy()
        base_std = baseline.sort("component")["std"].to_numpy()
        np.testing.assert_allclose(cur_std, base_std, atol=_TOL_SUMMARY, rtol=0)


# ===========================================================================
# 3. Normal distribution — correlation matrix
# ===========================================================================


@pytest.mark.regression()
class TestNormalCorrelationRegression:
    """Normal Scenarios_Distribution — correlation matrix matches baseline."""

    def test_correlation_matrix_values(
        self, canonical_distrib_normal: "Scenarios_Distribution"
    ):
        """Correlation matrix values match the Parquet baseline."""
        canonical_distrib_normal.generate()
        df = canonical_distrib_normal.calc_correlation_matrix()
        baseline = load_baseline("distrib_normal_correlation")
        cur_mat = _matrix_from_df(df, DISTRIB_COMPONENTS)
        bas_mat = _matrix_from_df(baseline, DISTRIB_COMPONENTS)
        np.testing.assert_allclose(cur_mat, bas_mat, atol=_TOL_MATRIX, rtol=0)

    def test_correlation_diagonal_is_one(
        self, canonical_distrib_normal: "Scenarios_Distribution"
    ):
        """Diagonal of the computed correlation matrix is all 1.0."""
        canonical_distrib_normal.generate()
        df = canonical_distrib_normal.calc_correlation_matrix()
        cur_mat = _matrix_from_df(df, DISTRIB_COMPONENTS)
        np.testing.assert_allclose(
            np.diag(cur_mat), np.ones(len(DISTRIB_COMPONENTS)), atol=1e-10
        )


# ===========================================================================
# 4. Normal distribution — covariance matrix
# ===========================================================================


@pytest.mark.regression()
class TestNormalCovarianceRegression:
    """Normal Scenarios_Distribution — covariance matrix matches baseline."""

    def test_covariance_matrix_values(
        self, canonical_distrib_normal: "Scenarios_Distribution"
    ):
        """Covariance matrix values match the Parquet baseline."""
        canonical_distrib_normal.generate()
        df = canonical_distrib_normal.calc_covariance_matrix()
        baseline = load_baseline("distrib_normal_covariance")
        cur_mat = _matrix_from_df(df, DISTRIB_COMPONENTS)
        bas_mat = _matrix_from_df(baseline, DISTRIB_COMPONENTS)
        np.testing.assert_allclose(cur_mat, bas_mat, atol=_TOL_MATRIX, rtol=0)


# ===========================================================================
# 5. Student-t distribution — scenario values
# ===========================================================================


@pytest.mark.regression()
class TestStudentTScenariosRegression:
    """Student-t Scenarios_Distribution — scenario values match baseline."""

    def test_scenario_values_match_baseline(
        self, canonical_distrib_student_t: "Scenarios_Distribution"
    ):
        """All Student-t component return values match the baseline."""
        df = canonical_distrib_student_t.generate()
        baseline = load_baseline("distrib_student_t_scenarios")
        _assert_df_close(
            df,
            baseline,
            DISTRIB_COMPONENTS,
            tolerance=_TOL_SCENARIO,
            label="student_t scenarios",
        )

    def test_first_value_matches_baseline(
        self, canonical_distrib_student_t: "Scenarios_Distribution"
    ):
        """First value of US_Equity matches baseline exactly."""
        df = canonical_distrib_student_t.generate()
        baseline = load_baseline("distrib_student_t_scenarios")
        assert df["US_Equity"][0] == pytest.approx(
            baseline["US_Equity"][0], abs=_TOL_SCENARIO
        )

    def test_summary_stats_match_baseline(
        self, canonical_distrib_student_t: "Scenarios_Distribution"
    ):
        """Student-t summary statistics means match the baseline."""
        canonical_distrib_student_t.generate()
        df = canonical_distrib_student_t.calc_summary_statistics()
        baseline = load_baseline("distrib_student_t_summary_stats")
        cur_means = df.sort("component")["mean"].to_numpy()
        base_means = baseline.sort("component")["mean"].to_numpy()
        np.testing.assert_allclose(cur_means, base_means, atol=_TOL_SUMMARY, rtol=0)

    def test_correlation_matrix_matches_baseline(
        self, canonical_distrib_student_t: "Scenarios_Distribution"
    ):
        """Student-t correlation matrix matches the baseline."""
        canonical_distrib_student_t.generate()
        df = canonical_distrib_student_t.calc_correlation_matrix()
        baseline = load_baseline("distrib_student_t_correlation")
        cur_mat = _matrix_from_df(df, DISTRIB_COMPONENTS)
        bas_mat = _matrix_from_df(baseline, DISTRIB_COMPONENTS)
        np.testing.assert_allclose(cur_mat, bas_mat, atol=_TOL_MATRIX, rtol=0)


# ===========================================================================
# 6. Lognormal distribution — scenario values
# ===========================================================================


@pytest.mark.regression()
class TestLognormalScenariosRegression:
    """Lognormal Scenarios_Distribution — scenario values match baseline."""

    def test_scenario_shape(
        self, canonical_distrib_lognormal: "Scenarios_Distribution"
    ):
        """Generated lognormal DataFrame has correct shape."""
        df = canonical_distrib_lognormal.generate()
        assert df.shape == (NUM_DAYS, len(DISTRIB_COMPONENTS) + 1)

    def test_scenario_values_match_baseline(
        self, canonical_distrib_lognormal: "Scenarios_Distribution"
    ):
        """All lognormal component values match the baseline."""
        df = canonical_distrib_lognormal.generate()
        baseline = load_baseline("distrib_lognormal_scenarios")
        _assert_df_close(
            df,
            baseline,
            DISTRIB_COMPONENTS,
            tolerance=_TOL_SCENARIO,
            label="lognormal scenarios",
        )

    def test_all_values_positive(
        self, canonical_distrib_lognormal: "Scenarios_Distribution"
    ):
        """All lognormal returns are strictly positive (exp transform)."""
        df = canonical_distrib_lognormal.generate()
        for comp in DISTRIB_COMPONENTS:
            assert (df[comp] > 0).all()


# ===========================================================================
# 7. Correlation + volatilities input mode — scenario values
# ===========================================================================


@pytest.mark.regression()
class TestCorrVolsScenariosRegression:
    """Corr+vols Scenarios_Distribution — scenario values match baseline."""

    def test_scenario_values_match_baseline(
        self, canonical_distrib_corr_vols: "Scenarios_Distribution"
    ):
        """Corr+vols scenario values match the baseline."""
        df = canonical_distrib_corr_vols.generate()
        baseline = load_baseline("distrib_corr_vols_scenarios")
        _assert_df_close(
            df,
            baseline,
            DISTRIB_COMPONENTS,
            tolerance=_TOL_SCENARIO,
            label="corr_vols scenarios",
        )


# ===========================================================================
# 8. CMA scenarios — generated values
# ===========================================================================


@pytest.mark.regression()
class TestCMAScenariosRegression:
    """Scenarios_CMA — generated scenario values match baseline."""

    def test_scenario_shape(self, canonical_cma: "Scenarios_CMA"):
        """CMA generated DataFrame has (NUM_DAYS, K+1) shape."""
        df = canonical_cma.generate()
        assert df.shape == (NUM_DAYS, len(CMA_ASSET_CLASSES) + 1)

    def test_scenario_values_match_baseline(self, canonical_cma: "Scenarios_CMA"):
        """All CMA component return values match the baseline."""
        df = canonical_cma.generate()
        baseline = load_baseline("CMA_scenarios")
        _assert_df_close(
            df,
            baseline,
            CMA_ASSET_CLASSES,
            tolerance=_TOL_SCENARIO,
            label="CMA scenarios",
        )

    def test_first_row_matches_baseline(self, canonical_cma: "Scenarios_CMA"):
        """First row of each CMA component column matches baseline."""
        df = canonical_cma.generate()
        baseline = load_baseline("CMA_scenarios")
        for ac in CMA_ASSET_CLASSES:
            assert df[ac][0] == pytest.approx(baseline[ac][0], abs=_TOL_SCENARIO)

    def test_last_row_matches_baseline(self, canonical_cma: "Scenarios_CMA"):
        """Last row of each CMA component column matches baseline."""
        df = canonical_cma.generate()
        baseline = load_baseline("CMA_scenarios")
        for ac in CMA_ASSET_CLASSES:
            assert df[ac][-1] == pytest.approx(baseline[ac][-1], abs=_TOL_SCENARIO)


# ===========================================================================
# 9. CMA summary statistics
# ===========================================================================


@pytest.mark.regression()
class TestCMASummaryStatsRegression:
    """Scenarios_CMA — summary statistics match baseline."""

    def test_summary_stats_shape(self, canonical_cma: "Scenarios_CMA"):
        """CMA summary statistics has one row per asset class."""
        canonical_cma.generate()
        df = canonical_cma.calc_summary_statistics()
        assert len(df) == len(CMA_ASSET_CLASSES)

    def test_mean_matches_baseline(self, canonical_cma: "Scenarios_CMA"):
        """CMA component mean values match the baseline."""
        canonical_cma.generate()
        df = canonical_cma.calc_summary_statistics()
        baseline = load_baseline("CMA_summary_stats")
        cur_means = df.sort("component")["mean"].to_numpy()
        base_means = baseline.sort("component")["mean"].to_numpy()
        np.testing.assert_allclose(cur_means, base_means, atol=_TOL_SUMMARY, rtol=0)


# ===========================================================================
# 10. CMA correlation matrix
# ===========================================================================


@pytest.mark.regression()
class TestCMACorrelationRegression:
    """Scenarios_CMA — correlation matrix matches baseline."""

    def test_correlation_matrix_values(self, canonical_cma: "Scenarios_CMA"):
        """CMA correlation matrix values match the baseline."""
        canonical_cma.generate()
        df = canonical_cma.calc_correlation_matrix()
        baseline = load_baseline("CMA_correlation")
        cur_mat = _matrix_from_df(df, CMA_ASSET_CLASSES)
        bas_mat = _matrix_from_df(baseline, CMA_ASSET_CLASSES)
        np.testing.assert_allclose(cur_mat, bas_mat, atol=_TOL_MATRIX, rtol=0)

    def test_correlation_diagonal_is_one(self, canonical_cma: "Scenarios_CMA"):
        """Diagonal of CMA correlation matrix is 1.0."""
        canonical_cma.generate()
        df = canonical_cma.calc_correlation_matrix()
        cur_mat = _matrix_from_df(df, CMA_ASSET_CLASSES)
        np.testing.assert_allclose(
            np.diag(cur_mat), np.ones(len(CMA_ASSET_CLASSES)), atol=1e-10
        )


# ===========================================================================
# 11. CMA daily parameters
# ===========================================================================


@pytest.mark.regression()
class TestCMADailyParamsRegression:
    """Scenarios_CMA — daily parameters match baseline."""

    def test_daily_returns_match_baseline(self, canonical_cma: "Scenarios_CMA"):
        """Daily expected returns match the CMA_daily_params baseline."""
        baseline = load_baseline("CMA_daily_params")
        cur_daily = canonical_cma.calc_daily_expected_returns()
        base_daily = baseline.sort("asset_class")["daily_return"].to_numpy()
        cur_daily_sorted = cur_daily[
            [CMA_ASSET_CLASSES.index(ac) for ac in sorted(CMA_ASSET_CLASSES)]
        ]
        np.testing.assert_allclose(
            cur_daily_sorted, base_daily, atol=_TOL_MATRIX, rtol=0
        )

    def test_daily_covariance_shape(self, canonical_cma: "Scenarios_CMA"):
        """Daily covariance matrix has shape (K, K)."""
        cov = canonical_cma.calc_daily_covariance()
        assert cov.shape == (len(CMA_ASSET_CLASSES), len(CMA_ASSET_CLASSES))

    def test_daily_covariance_matches_baseline(self, canonical_cma: "Scenarios_CMA"):
        """Daily covariance matrix matches CMA_daily_covariance baseline."""
        baseline = load_baseline("CMA_daily_covariance")
        cur_cov = canonical_cma.calc_daily_covariance()
        bas_cov = _matrix_from_df(baseline, CMA_ASSET_CLASSES)
        np.testing.assert_allclose(cur_cov, bas_cov, atol=_TOL_MATRIX, rtol=0)
