"""Integration tests for num_methods.covariance — end-to-end pipelines.

Verifies that all estimators produce structurally valid covariance matrices,
that covariance→correlation conversions are consistent, that the
distance_estimator_type classifier works with live matrices, and that
per-component accessors agree with the full matrix representation.

Test Categories
---------------
- All 10 estimators produce same-shape covariance matrix (parametrised)
- covariance_matrix.get_correlation_matrix returns valid correlation matrix
- Cross-estimator: correlation matrix diagonals are always 1.0
- distance_estimator_type classmethods return non-overlapping subsets
- get_component_variance / get_component_covariance agree with full matrix
- End-to-end: Polars DataFrame → covariance_matrix → validation passes
- Estimators produce symmetric PSD matrix for 3-asset and 5-asset data

Author:
    QWIM Development Team

Version:
    0.1.0

Last Modified:
    2026-02-28
"""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from src.utils.custom_exceptions_errors_loggers.logger_custom import get_logger

_logger = get_logger(__name__)

MODULE_IMPORT_AVAILABLE: bool = True

try:
    from src.num_methods.covariance.utils_cov_corr import (
        covariance_estimator,
        covariance_matrix,
        distance_estimator_type,
    )
except ImportError as exc:
    MODULE_IMPORT_AVAILABLE = False
    _logger.warning("Import failed — tests will be skipped: %s", exc)

pytestmark = pytest.mark.skipif(
    not MODULE_IMPORT_AVAILABLE,
    reason="num_methods.covariance modules not importable",
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_RANDOM_SEED: int = 42
_NUM_OBS_3: int = 200
_NUM_OBS_5: int = 300
_COMPONENTS_3: list[str] = ["AAPL", "MSFT", "GOOG"]
_COMPONENTS_5: list[str] = ["AAPL", "MSFT", "GOOG", "AMZN", "TSLA"]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def returns_3_assets() -> pl.DataFrame:
    """Canonical 200-obs × 3-asset correlated returns DataFrame."""
    rng = np.random.default_rng(_RANDOM_SEED)
    factor = rng.standard_normal(_NUM_OBS_3) * 0.01
    noise = rng.standard_normal((_NUM_OBS_3, 3)) * 0.005
    raw = factor[:, np.newaxis] + noise

    dates = pl.date_range(
        pl.date(2023, 1, 1),
        pl.date(2023, 1, 1) + pl.duration(days=_NUM_OBS_3 - 1),
        interval="1d",
        eager=True,
    )
    return pl.DataFrame(
        {"Date": dates, **{c: raw[:, i].tolist() for i, c in enumerate(_COMPONENTS_3)}}
    )


@pytest.fixture(scope="module")
def returns_5_assets() -> pl.DataFrame:
    """Canonical 300-obs × 5-asset correlated returns DataFrame."""
    rng = np.random.default_rng(_RANDOM_SEED)
    factor = rng.standard_normal(_NUM_OBS_5) * 0.01
    noise = rng.standard_normal((_NUM_OBS_5, 5)) * 0.005
    raw = factor[:, np.newaxis] + noise

    dates = pl.date_range(
        pl.date(2023, 1, 1),
        pl.date(2023, 1, 1) + pl.duration(days=_NUM_OBS_5 - 1),
        interval="1d",
        eager=True,
    )
    return pl.DataFrame(
        {"Date": dates, **{c: raw[:, i].tolist() for i, c in enumerate(_COMPONENTS_5)}}
    )


@pytest.fixture(scope="module")
def empirical_cov_3(returns_3_assets: pl.DataFrame) -> "covariance_matrix":
    """Empirical covariance matrix from the 3-asset fixture."""
    return covariance_matrix(
        data_returns=returns_3_assets,
        estimator=covariance_estimator.EMPIRICAL,
    )


@pytest.fixture(scope="module")
def empirical_cov_5(returns_5_assets: pl.DataFrame) -> "covariance_matrix":
    """Empirical covariance matrix from the 5-asset fixture."""
    return covariance_matrix(
        data_returns=returns_5_assets,
        estimator=covariance_estimator.EMPIRICAL,
    )


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

_ESTIMATORS_REQUIRING_PRIOR: tuple[covariance_estimator, ...] = (
    covariance_estimator.IMPLIED_COVARIANCE,
)


def _build_cov(
    df: pl.DataFrame, estimator: "covariance_estimator"
) -> "covariance_matrix":
    """Build a covariance_matrix; skip IMPLIED_COVARIANCE (needs extra data)."""
    return covariance_matrix(data_returns=df, estimator=estimator)


# ===========================================================================
# 1. All 10 estimators produce same-shape matrix (parametrised)
# ===========================================================================


_ESTIMATORS_SKIPPED = {covariance_estimator.IMPLIED_COVARIANCE}
_ESTIMATORS_TESTABLE = [
    e for e in covariance_estimator if e not in _ESTIMATORS_SKIPPED
]


@pytest.mark.integration()
@pytest.mark.parametrize("estimator", _ESTIMATORS_TESTABLE, ids=lambda e: e.value)
class TestAllEstimatorsShape:
    """All non-implied estimators produce (3,3) matrix for 3-asset input."""

    def test_output_shape_3_assets(self, returns_3_assets, estimator):
        """Covariance matrix has shape (K, K) = (3, 3)."""
        cov = _build_cov(returns_3_assets, estimator)
        mat = cov.get_covariance_matrix()
        assert mat.shape == (3, 3)

    def test_component_count_3_assets(self, returns_3_assets, estimator):
        """m_num_components equals number of return columns (3)."""
        cov = _build_cov(returns_3_assets, estimator)
        assert cov.m_num_components == 3

    def test_observation_count_3_assets(self, returns_3_assets, estimator):
        """m_num_observations equals number of DataFrame rows."""
        cov = _build_cov(returns_3_assets, estimator)
        assert cov.m_num_observations == _NUM_OBS_3

    def test_validates_without_error_3_assets(self, returns_3_assets, estimator):
        """validate_covariance_matrix() passes without raising exceptions."""
        cov = _build_cov(returns_3_assets, estimator)
        cov.validate_covariance_matrix()  # should not raise


# ===========================================================================
# 2. Covariance → Correlation pipeline
# ===========================================================================


@pytest.mark.integration()
class TestCovarianceToCorrPipeline:
    """get_correlation_matrix returns valid correlation matrices."""

    def test_diagonal_all_ones_3_assets(self, empirical_cov_3):
        """Diagonal of correlation matrix must all be 1.0."""
        corr = empirical_cov_3.get_correlation_matrix()
        diag = np.diag(corr)
        np.testing.assert_allclose(diag, 1.0, atol=1e-10)

    def test_off_diagonal_in_bounds_3_assets(self, empirical_cov_3):
        """Off-diagonal correlation values must be in (-1, 1)."""
        corr = empirical_cov_3.get_correlation_matrix()
        assert np.all(corr >= -1.0 - 1e-8)
        assert np.all(corr <= 1.0 + 1e-8)

    def test_symmetric_3_assets(self, empirical_cov_3):
        """Correlation matrix is symmetric."""
        corr = empirical_cov_3.get_correlation_matrix()
        np.testing.assert_allclose(corr, corr.T, atol=1e-10)

    def test_shape_matches_covariance_3_assets(self, empirical_cov_3):
        """Correlation matrix has same shape as covariance matrix."""
        cov_mat = empirical_cov_3.get_covariance_matrix()
        corr = empirical_cov_3.get_correlation_matrix()
        assert corr.shape == cov_mat.shape

    def test_diagonal_all_ones_5_assets(self, empirical_cov_5):
        """Diagonal of correlation matrix must all be 1.0 for 5-asset data."""
        corr = empirical_cov_5.get_correlation_matrix()
        diag = np.diag(corr)
        np.testing.assert_allclose(diag, 1.0, atol=1e-10)

    def test_corr_matrix_is_symmetric_5_assets(self, empirical_cov_5):
        """Correlation matrix is symmetric for 5-asset data."""
        corr = empirical_cov_5.get_correlation_matrix()
        np.testing.assert_allclose(corr, corr.T, atol=1e-10)

    def test_corr_psd_5_assets(self, empirical_cov_5):
        """Correlation matrix is PSD (all eigenvalues ≥ -1e-8)."""
        corr = empirical_cov_5.get_correlation_matrix()
        eigvals = np.linalg.eigvalsh(corr)
        assert np.all(eigvals >= -1e-8)


# ===========================================================================
# 3. Cross-estimator: correlation diagonals consistent
# ===========================================================================


@pytest.mark.integration()
class TestCrossEstimatorConsistency:
    """Different estimators all produce valid correlation diagonals."""

    @pytest.mark.parametrize("estimator", _ESTIMATORS_TESTABLE, ids=lambda e: e.value)
    def test_correlation_diagonal_is_one(self, returns_3_assets, estimator):
        """Correlation diagonal == 1.0 regardless of estimator."""
        cov = _build_cov(returns_3_assets, estimator)
        corr = cov.get_correlation_matrix()
        diag = np.diag(corr)
        np.testing.assert_allclose(diag, 1.0, atol=1e-8)

    @pytest.mark.parametrize("estimator", _ESTIMATORS_TESTABLE, ids=lambda e: e.value)
    def test_covariance_matrix_symmetric(self, returns_3_assets, estimator):
        """Covariance matrix is symmetric for all estimators."""
        cov = _build_cov(returns_3_assets, estimator)
        mat = cov.get_covariance_matrix()
        np.testing.assert_allclose(mat, mat.T, atol=1e-8)

    @pytest.mark.parametrize("estimator", _ESTIMATORS_TESTABLE, ids=lambda e: e.value)
    def test_covariance_positive_diagonal(self, returns_3_assets, estimator):
        """Diagonal elements (variances) are all positive for all estimators."""
        cov = _build_cov(returns_3_assets, estimator)
        mat = cov.get_covariance_matrix()
        assert np.all(np.diag(mat) > 0)


# ===========================================================================
# 4. distance_estimator_type classmethods
# ===========================================================================


@pytest.mark.integration()
class TestDistanceEstimatorClassmethods:
    """distance_estimator_type classmethods return valid, non-overlapping sets."""

    def test_correlation_based_returns_list(self):
        """get_correlation_based() returns a non-empty list."""
        result = distance_estimator_type.get_correlation_based()
        assert isinstance(result, list)
        assert len(result) >= 1

    def test_covariance_based_returns_list(self):
        """get_covariance_based() returns a non-empty list."""
        result = distance_estimator_type.get_covariance_based()
        assert isinstance(result, list)
        assert len(result) >= 1

    def test_rank_based_returns_list(self):
        """get_rank_based() returns a non-empty list."""
        result = distance_estimator_type.get_rank_based()
        assert isinstance(result, list)
        assert len(result) >= 1

    def test_information_theoretic_returns_list(self):
        """get_information_theoretic() returns a non-empty list."""
        result = distance_estimator_type.get_information_theoretic()
        assert isinstance(result, list)
        assert len(result) >= 1

    def test_all_members_covered(self):
        """All enum members appear in at least one category list."""
        all_members = set(distance_estimator_type)
        categorised = set(
            distance_estimator_type.get_correlation_based()
            + distance_estimator_type.get_covariance_based()
            + distance_estimator_type.get_rank_based()
            + distance_estimator_type.get_information_theoretic()
        )
        missing = all_members - categorised
        assert len(missing) == 0, f"Uncategorised distance estimators: {missing}"

    def test_correlation_covariance_disjoint(self):
        """Correlation-based and covariance-based sets are disjoint."""
        corr_set = set(distance_estimator_type.get_correlation_based())
        cov_set = set(distance_estimator_type.get_covariance_based())
        overlap = corr_set & cov_set
        assert len(overlap) == 0, f"Unexpected overlap: {overlap}"


# ===========================================================================
# 5. Per-component accessors agree with full matrix
# ===========================================================================


@pytest.mark.integration()
class TestComponentAccessors:
    """get_component_variance and get_component_covariance agree with full matrix."""

    def test_component_variance_matches_diagonal(self, empirical_cov_3):
        """get_component_variance returns the same value as the diagonal."""
        mat = empirical_cov_3.get_covariance_matrix()
        for idx, name in enumerate(_COMPONENTS_3):
            var_direct = float(mat[idx, idx])
            var_accessor = empirical_cov_3.get_component_variance(name)
            assert var_accessor == pytest.approx(var_direct, rel=1e-10)

    def test_component_covariance_matches_matrix(self, empirical_cov_3):
        """get_component_covariance returns the same off-diagonal value."""
        mat = empirical_cov_3.get_covariance_matrix()
        name_0 = _COMPONENTS_3[0]
        name_1 = _COMPONENTS_3[1]
        expected = float(mat[0, 1])
        result = empirical_cov_3.get_component_covariance(name_0, name_1)
        assert result == pytest.approx(expected, rel=1e-10)

    def test_covariance_symmetric_via_accessor(self, empirical_cov_3):
        """Cov(A,B) == Cov(B,A) via the accessor."""
        name_0 = _COMPONENTS_3[0]
        name_1 = _COMPONENTS_3[1]
        cov_ab = empirical_cov_3.get_component_covariance(name_0, name_1)
        cov_ba = empirical_cov_3.get_component_covariance(name_1, name_0)
        assert cov_ab == pytest.approx(cov_ba, rel=1e-10)

    def test_component_variance_positive(self, empirical_cov_3):
        """All component variances are strictly positive."""
        for name in _COMPONENTS_3:
            var = empirical_cov_3.get_component_variance(name)
            assert var > 0

    def test_component_variance_5_assets(self, empirical_cov_5):
        """All 5-asset component variances match diagonal of full matrix."""
        mat = empirical_cov_5.get_covariance_matrix()
        for idx, name in enumerate(_COMPONENTS_5):
            var_direct = float(mat[idx, idx])
            var_accessor = empirical_cov_5.get_component_variance(name)
            assert var_accessor == pytest.approx(var_direct, rel=1e-10)


# ===========================================================================
# 6. End-to-end: Polars DataFrame → covariance_matrix → validate
# ===========================================================================


@pytest.mark.integration()
class TestEndToEndPipeline:
    """Full pipeline: construct DataFrame, build covariance, validate."""

    def test_full_pipeline_3_assets(self, returns_3_assets):
        """Build and validate covariance matrix without errors (3 assets)."""
        cov = covariance_matrix(
            data_returns=returns_3_assets,
            estimator=covariance_estimator.EMPIRICAL,
        )
        cov.validate_covariance_matrix()
        mat = cov.get_covariance_matrix()
        assert mat.shape == (3, 3)
        assert mat.dtype == np.float64

    def test_full_pipeline_5_assets(self, returns_5_assets):
        """Build and validate covariance matrix without errors (5 assets)."""
        cov = covariance_matrix(
            data_returns=returns_5_assets,
            estimator=covariance_estimator.EMPIRICAL,
        )
        cov.validate_covariance_matrix()
        mat = cov.get_covariance_matrix()
        assert mat.shape == (5, 5)

    def test_ledoit_wolf_shape_equals_empirical(
        self, returns_3_assets
    ):
        """LedoitWolf and Empirical produce same-shape output for same input."""
        cov_emp = covariance_matrix(
            data_returns=returns_3_assets,
            estimator=covariance_estimator.EMPIRICAL,
        )
        cov_lw = covariance_matrix(
            data_returns=returns_3_assets,
            estimator=covariance_estimator.LEDOIT_WOLF,
        )
        assert cov_emp.get_covariance_matrix().shape == cov_lw.get_covariance_matrix().shape

    def test_component_names_stored_correctly(self, returns_3_assets):
        """m_component_names matches the expected column ordering."""
        cov = covariance_matrix(
            data_returns=returns_3_assets,
            estimator=covariance_estimator.EMPIRICAL,
        )
        # Component names should match the non-Date columns
        assert list(cov.m_component_names) == _COMPONENTS_3

    def test_covariance_matrix_copy_not_same_object(self, empirical_cov_3):
        """get_covariance_matrix() returns a copy, not the internal array."""
        mat1 = empirical_cov_3.get_covariance_matrix()
        mat2 = empirical_cov_3.get_covariance_matrix()
        assert mat1 is not mat2

    def test_correlation_matrix_copy_not_same_object(self, empirical_cov_3):
        """get_correlation_matrix() returns a new array each call."""
        corr1 = empirical_cov_3.get_correlation_matrix()
        corr2 = empirical_cov_3.get_correlation_matrix()
        assert corr1 is not corr2

    def test_str_repr_contains_estimator_name(self, empirical_cov_3):
        """__str__ contains the estimator name."""
        s = str(empirical_cov_3)
        assert "empirical" in s.lower() or "Empirical" in s
