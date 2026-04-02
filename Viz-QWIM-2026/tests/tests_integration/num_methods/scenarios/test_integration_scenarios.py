"""Integration tests for num_methods.scenarios — end-to-end pipelines.

Verifies that Scenarios_Distribution and Scenarios_CMA produce structurally
valid DataFrames from generate(), that base-class utilities (validate_scenarios,
get_component_series, get_returns_matrix, filter_by_date_range) operate
correctly on generated data, and that the distribution→covariance feedback
pipeline works without error.

Test Categories
---------------
- Scenarios_Distribution.generate() produces correct DataFrame schema
- Scenarios_CMA.generate() produces correct DataFrame schema
- validate_scenarios() passes on generated output for all distribution types
- get_component_series / get_returns_matrix operate on generated data
- filter_by_date_range narrows scenario data correctly
- Distribution → covariance_matrix pipeline completes without error
- CMA from_correlation_and_volatilities / annualisation helpers

Author:
    QWIM Development Team

Version:
    0.1.0

Last Modified:
    2026-02-28
"""

from __future__ import annotations

import datetime as dt
from typing import Any

import numpy as np
import polars as pl
import pytest

from src.utils.custom_exceptions_errors_loggers.logger_custom import get_logger

_logger = get_logger(__name__)

MODULE_IMPORT_AVAILABLE: bool = True

try:
    from src.num_methods.scenarios.scenarios_base import (
        Frequency_Time_Series,
        Scenario_Data_Type,
    )
    from src.num_methods.scenarios.scenarios_CMA import (
        Asset_Class_Tier,
        CMA_Source,
        Scenarios_CMA,
    )
    from src.num_methods.scenarios.scenarios_distrib import (
        Distribution_Type,
        Scenarios_Distribution,
    )
except ImportError as exc:
    MODULE_IMPORT_AVAILABLE = False
    _logger.warning("Import failed — tests will be skipped: %s", exc)

_COV_IMPORT_OK: bool = MODULE_IMPORT_AVAILABLE

try:
    from src.num_methods.covariance.utils_cov_corr import (
        covariance_estimator,
        covariance_matrix,
    )
except ImportError:
    _COV_IMPORT_OK = False

pytestmark = pytest.mark.skipif(
    not MODULE_IMPORT_AVAILABLE,
    reason="num_methods.scenarios modules not importable",
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_SEED: int = 0
_NUM_DAYS: int = 60
_COMPONENTS: list[str] = ["A", "B", "C"]
_K: int = len(_COMPONENTS)

_MEAN = np.zeros(_K, dtype=np.float64)
_COV = np.array(
    [[1e-4, 5e-5, 2e-5], [5e-5, 1e-4, 3e-5], [2e-5, 3e-5, 1e-4]],
    dtype=np.float64,
)
_CORR = np.array(
    [[1.0, 0.5, 0.2], [0.5, 1.0, 0.3], [0.2, 0.3, 1.0]],
    dtype=np.float64,
)
_VOLS = np.array([0.01, 0.012, 0.008], dtype=np.float64)

_RET_ANN = np.array([0.06, 0.07, 0.05], dtype=np.float64)
_VOL_ANN = np.array([0.10, 0.12, 0.08], dtype=np.float64)

_START_DATE = dt.date(2024, 1, 1)


# ---------------------------------------------------------------------------
# Shared factories
# ---------------------------------------------------------------------------


def _make_distrib(
    distribution_type: "Distribution_Type" = Distribution_Type.NORMAL,
) -> "Scenarios_Distribution":
    return Scenarios_Distribution(
        names_components=_COMPONENTS,
        distribution_type=distribution_type,
        mean_returns=_MEAN,
        covariance_matrix=_COV,
        start_date=_START_DATE,
        num_days=_NUM_DAYS,
        random_seed=_SEED,
    )


def _make_cma() -> "Scenarios_CMA":
    return Scenarios_CMA(
        names_asset_classes=_COMPONENTS,
        expected_returns_annual=_RET_ANN,
        expected_vols_annual=_VOL_ANN,
        correlation_matrix=_CORR,
        start_date=_START_DATE,
        num_days=_NUM_DAYS,
        random_seed=_SEED,
    )


# ===========================================================================
# 1. Scenarios_Distribution.generate() — schema and shape
# ===========================================================================


@pytest.mark.integration()
class TestScenariosDistributionGenerate:
    """Scenarios_Distribution.generate() returns valid DataFrames."""

    @pytest.mark.parametrize(
        "dist",
        [Distribution_Type.NORMAL, Distribution_Type.STUDENT_T],
        ids=["normal", "student_t"],
    )
    def test_generate_returns_polars_dataframe(self, dist):
        """generate() returns a Polars DataFrame."""
        scen = _make_distrib(dist)
        result = scen.generate()
        assert isinstance(result, pl.DataFrame)

    @pytest.mark.parametrize(
        "dist",
        [Distribution_Type.NORMAL, Distribution_Type.STUDENT_T],
        ids=["normal", "student_t"],
    )
    def test_generated_row_count(self, dist):
        """Row count equals num_days."""
        scen = _make_distrib(dist)
        result = scen.generate()
        assert len(result) == _NUM_DAYS

    @pytest.mark.parametrize(
        "dist",
        [Distribution_Type.NORMAL, Distribution_Type.STUDENT_T],
        ids=["normal", "student_t"],
    )
    def test_generated_columns(self, dist):
        """DataFrame has Date column plus one column per component."""
        scen = _make_distrib(dist)
        result = scen.generate()
        assert "Date" in result.columns
        for comp in _COMPONENTS:
            assert comp in result.columns

    @pytest.mark.parametrize(
        "dist",
        [Distribution_Type.NORMAL, Distribution_Type.STUDENT_T],
        ids=["normal", "student_t"],
    )
    def test_component_columns_float64(self, dist):
        """Component columns are Float64."""
        scen = _make_distrib(dist)
        result = scen.generate()
        for comp in _COMPONENTS:
            assert result[comp].dtype == pl.Float64

    @pytest.mark.parametrize(
        "dist",
        [Distribution_Type.NORMAL, Distribution_Type.STUDENT_T],
        ids=["normal", "student_t"],
    )
    def test_no_nulls_in_generated_data(self, dist):
        """Generated DataFrame contains no null values."""
        scen = _make_distrib(dist)
        result = scen.generate()
        assert result.null_count().sum_horizontal()[0] == 0

    def test_date_column_is_date_type(self):
        """Date column dtype is pl.Date."""
        scen = _make_distrib()
        result = scen.generate()
        assert result.schema["Date"] == pl.Date

    def test_generate_sets_df_scenarios(self):
        """After generate(), df_scenarios property returns the same frame."""
        scen = _make_distrib()
        result = scen.generate()
        assert scen.df_scenarios is not None
        assert len(scen.df_scenarios) == len(result)

    def test_reproducible_with_same_seed(self):
        """Two instances with the same seed produce identical results."""
        scen_a = _make_distrib()
        scen_b = _make_distrib()
        df_a = scen_a.generate()
        df_b = scen_b.generate()
        for comp in _COMPONENTS:
            np.testing.assert_array_equal(
                df_a[comp].to_numpy(), df_b[comp].to_numpy()
            )

    def test_different_seeds_give_different_output(self):
        """Two instances with different seeds differ in generated values."""
        scen_a = Scenarios_Distribution(
            names_components=_COMPONENTS,
            covariance_matrix=_COV,
            start_date=_START_DATE,
            num_days=_NUM_DAYS,
            random_seed=0,
        )
        scen_b = Scenarios_Distribution(
            names_components=_COMPONENTS,
            covariance_matrix=_COV,
            start_date=_START_DATE,
            num_days=_NUM_DAYS,
            random_seed=999,
        )
        df_a = scen_a.generate()
        df_b = scen_b.generate()
        assert not df_a[_COMPONENTS[0]].equals(df_b[_COMPONENTS[0]])


# ===========================================================================
# 2. Scenarios_CMA.generate() — schema and shape
# ===========================================================================


@pytest.mark.integration()
class TestScenariosCMAGenerate:
    """Scenarios_CMA.generate() returns valid DataFrames."""

    def test_generate_returns_polars_dataframe(self):
        """generate() returns a Polars DataFrame."""
        scen = _make_cma()
        result = scen.generate()
        assert isinstance(result, pl.DataFrame)

    def test_generated_row_count(self):
        """Row count equals num_days."""
        scen = _make_cma()
        result = scen.generate()
        assert len(result) == _NUM_DAYS

    def test_generated_columns(self):
        """DataFrame has Date column plus one column per asset class."""
        scen = _make_cma()
        result = scen.generate()
        assert "Date" in result.columns
        for comp in _COMPONENTS:
            assert comp in result.columns

    def test_component_columns_float64(self):
        """Component columns are Float64."""
        scen = _make_cma()
        result = scen.generate()
        for comp in _COMPONENTS:
            assert result[comp].dtype == pl.Float64

    def test_no_nulls_in_generated_data(self):
        """Generated DataFrame contains no null values."""
        scen = _make_cma()
        result = scen.generate()
        assert result.null_count().sum_horizontal()[0] == 0

    def test_reproducible_with_same_seed(self):
        """Two CMA instances with same seed give identical output."""
        scen_a = _make_cma()
        scen_b = _make_cma()
        df_a = scen_a.generate()
        df_b = scen_b.generate()
        for comp in _COMPONENTS:
            np.testing.assert_array_equal(
                df_a[comp].to_numpy(), df_b[comp].to_numpy()
            )

    def test_daily_returns_scale_vs_annual_params(self):
        """Daily returns are roughly within ±5σ of the daily expected return."""
        scen = _make_cma()
        df = scen.generate()
        mu_daily = _RET_ANN[0] / 252
        vol_daily = _VOL_ANN[0] / np.sqrt(252)
        returns_arr = df[_COMPONENTS[0]].to_numpy()
        # Loose sanity check: most returns within 5 daily sigma of daily mean
        assert np.abs(np.mean(returns_arr) - mu_daily) < 5.0 * vol_daily


# ===========================================================================
# 3. Base-class utilities operate on generated data
# ===========================================================================


@pytest.mark.integration()
class TestBaseClassUtilitiesOnGeneratedData:
    """validate_scenarios, get_component_series, get_returns_matrix, filter."""

    def test_validate_scenarios_passes_after_generate(self):
        """validate_scenarios() does not raise after generate()."""
        scen = _make_distrib()
        scen.generate()
        scen.validate_scenarios()  # should not raise

    def test_get_component_series_length(self):
        """get_component_series returns series of correct length."""
        scen = _make_distrib()
        scen.generate()
        series = scen.get_component_series(_COMPONENTS[0])
        assert len(series) == _NUM_DAYS

    def test_get_returns_matrix_shape(self):
        """get_returns_matrix returns (T, K) numpy array."""
        scen = _make_distrib()
        scen.generate()
        mat = scen.get_returns_matrix()
        assert mat.shape == (_NUM_DAYS, _K)

    def test_get_returns_matrix_first_column_matches_series(self):
        """First returns matrix column matches get_component_series for A."""
        scen = _make_distrib()
        scen.generate()
        col_0 = scen.get_returns_matrix()[:, 0]
        series_a = scen.get_component_series(_COMPONENTS[0]).to_numpy()
        np.testing.assert_array_equal(col_0, series_a)

    def test_get_date_range_returns_tuple(self):
        """get_date_range() returns a (start, end) tuple after generate()."""
        scen = _make_distrib()
        scen.generate()
        date_range = scen.get_date_range()
        assert len(date_range) == 2

    def test_filter_by_date_range_reduces_rows(self):
        """filter_by_date_range with a mid-range end date reduces row count."""
        scen = _make_distrib()
        scen.generate()
        start, end = scen.get_date_range()
        mid = start + dt.timedelta(days=(_NUM_DAYS // 2) - 1)
        filtered = scen.filter_by_date_range(start, mid)
        assert len(filtered) < _NUM_DAYS

    def test_filter_by_date_range_keeps_date_column(self):
        """Filtered DataFrame still contains Date column."""
        scen = _make_distrib()
        scen.generate()
        start, end = scen.get_date_range()
        filtered = scen.filter_by_date_range(start, end)
        assert "Date" in filtered.columns

    def test_validate_scenarios_cma(self):
        """validate_scenarios() passes for CMA-generated data."""
        scen = _make_cma()
        scen.generate()
        scen.validate_scenarios()  # should not raise

    def test_get_returns_matrix_cma_shape(self):
        """get_returns_matrix returns (T, K) for CMA scenario."""
        scen = _make_cma()
        scen.generate()
        mat = scen.get_returns_matrix()
        assert mat.shape == (_NUM_DAYS, _K)


# ===========================================================================
# 4. Distribution → covariance_matrix integration pipeline
# ===========================================================================


@pytest.mark.integration()
@pytest.mark.skipif(
    not _COV_IMPORT_OK,
    reason="covariance_matrix module not importable",
)
class TestDistributionToCovariancePipeline:
    """Generated scenario data feeds into covariance_matrix without error."""

    def test_generated_data_feeds_into_covariance_matrix(self):
        """Normal scenario output is a valid covariance_matrix input."""
        scen = Scenarios_Distribution(
            names_components=_COMPONENTS,
            covariance_matrix=_COV,
            start_date=_START_DATE,
            num_days=200,
            random_seed=42,
        )
        df = scen.generate()

        # Feed into covariance_matrix (it expects Date col + float columns)
        cov = covariance_matrix(
            data_returns=df,
            estimator=covariance_estimator.EMPIRICAL,
        )
        cov.validate_covariance_matrix()
        assert cov.m_num_components == _K
        assert cov.m_num_observations == 200

    def test_covariance_shape_matches_components(self):
        """Covariance matrix from scenario output has shape (K, K)."""
        scen = Scenarios_Distribution(
            names_components=_COMPONENTS,
            covariance_matrix=_COV,
            start_date=_START_DATE,
            num_days=300,
            random_seed=42,
        )
        df = scen.generate()
        cov = covariance_matrix(
            data_returns=df,
            estimator=covariance_estimator.LEDOIT_WOLF,
        )
        assert cov.get_covariance_matrix().shape == (_K, _K)


# ===========================================================================
# 5. CMA annualisation helpers
# ===========================================================================


@pytest.mark.integration()
class TestCMAAnnualisationHelpers:
    """calc_daily_expected_returns, calc_daily_covariance, calc_daily_volatilities."""

    def test_daily_expected_returns_scale(self):
        """Daily expected returns are annual / 252."""
        scen = _make_cma()
        daily = scen.calc_daily_expected_returns()
        np.testing.assert_allclose(daily, _RET_ANN / 252.0, rtol=1e-10)

    def test_daily_covariance_scale(self):
        """Daily covariance is annual covariance / 252."""
        scen = _make_cma()
        daily_cov = scen.calc_daily_covariance()
        annual_cov = scen.covariance_matrix_annual
        np.testing.assert_allclose(daily_cov, annual_cov / 252.0, rtol=1e-10)

    def test_daily_volatilities_scale(self):
        """Daily volatilities are annual / sqrt(252)."""
        scen = _make_cma()
        daily_vols = scen.calc_daily_volatilities()
        np.testing.assert_allclose(daily_vols, _VOL_ANN / np.sqrt(252.0), rtol=1e-10)

    def test_get_index_correspondence_table_shape(self):
        """get_index_correspondence_table returns DataFrame with 5 columns."""
        scen = _make_cma()
        tbl = scen.get_index_correspondence_table()
        assert isinstance(tbl, pl.DataFrame)
        assert len(tbl) == _K
        expected_cols = {
            "asset_class",
            "financial_index",
            "tier",
            "expected_return",
            "expected_volatility",
        }
        assert expected_cols.issubset(set(tbl.columns))

    def test_get_asset_classes_by_tier_all_covered(self):
        """All K components appear in some tier."""
        scen = _make_cma()
        all_classes: list[str] = []
        for tier in Asset_Class_Tier:
            all_classes.extend(scen.get_asset_classes_by_tier(tier))
        assert set(all_classes) == set(_COMPONENTS)


# ===========================================================================
# 6. Scenarios_Distribution.from_correlation_and_volatilities convenience ctor
# ===========================================================================


@pytest.mark.integration()
class TestConvenienceConstructors:
    """from_correlation_and_volatilities produces same result as direct ctor."""

    def test_from_corr_and_vols_generates_same_shape(self):
        """Convenience constructor output matches direct constructor shape."""
        direct = Scenarios_Distribution(
            names_components=_COMPONENTS,
            covariance_matrix=_COV,
            start_date=_START_DATE,
            num_days=_NUM_DAYS,
            random_seed=_SEED,
        )
        via_corr = Scenarios_Distribution.from_correlation_and_volatilities(
            names_components=_COMPONENTS,
            correlation_matrix=_CORR,
            volatilities=_VOLS,
            start_date=_START_DATE,
            num_days=_NUM_DAYS,
            random_seed=_SEED,
        )
        df_direct = direct.generate()
        df_corr = via_corr.generate()
        assert df_direct.shape == df_corr.shape

    def test_from_corr_and_vols_covariance_input_type(self):
        """Convenience constructor sets CORRELATION_AND_VOLATILITIES input type."""
        from src.num_methods.scenarios.scenarios_distrib import Covariance_Input_Type

        scen = Scenarios_Distribution.from_correlation_and_volatilities(
            names_components=_COMPONENTS,
            correlation_matrix=_CORR,
            volatilities=_VOLS,
            start_date=_START_DATE,
            num_days=_NUM_DAYS,
        )
        assert scen.covariance_input_type == Covariance_Input_Type.CORRELATION_AND_VOLATILITIES
