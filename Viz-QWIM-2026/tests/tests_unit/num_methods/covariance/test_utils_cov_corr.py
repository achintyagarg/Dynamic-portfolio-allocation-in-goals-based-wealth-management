"""Unit tests for utils_cov_corr module.

This module contains comprehensive tests for the covariance/correlation
matrix utilities in src/num_methods/covariance/utils_cov_corr.py,
covering all three classes:
- covariance_estimator enum
- distance_estimator_type enum
- covariance_matrix class
"""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest


# ---------------------------------------------------------------------------
# Fixtures - lazy imports to avoid top-level resolution issues
# ---------------------------------------------------------------------------


@pytest.fixture()
def covariance_estimator_enum():
    """Fixture to import covariance_estimator enum."""
    from src.num_methods.covariance.utils_cov_corr import covariance_estimator

    return covariance_estimator


@pytest.fixture()
def distance_estimator_type_enum():
    """Fixture to import distance_estimator_type enum."""
    from src.num_methods.covariance.utils_cov_corr import distance_estimator_type

    return distance_estimator_type


@pytest.fixture()
def covariance_matrix_class():
    """Fixture to import covariance_matrix class."""
    from src.num_methods.covariance.utils_cov_corr import covariance_matrix

    return covariance_matrix


@pytest.fixture()
def Exception_Validation_Input_class():
    """Fixture to import custom validation exception."""
    from src.utils.custom_exceptions_errors_loggers.exception_custom import (
        Exception_Validation_Input,
    )

    return Exception_Validation_Input


@pytest.fixture()
def Exception_Calculation_class():
    """Fixture to import custom calculation exception."""
    from src.utils.custom_exceptions_errors_loggers.exception_custom import (
        Exception_Calculation,
    )

    return Exception_Calculation


@pytest.fixture()
def sample_returns_df() -> pl.DataFrame:
    """Fixture providing a minimal valid returns DataFrame (100 obs x 3 assets).

    Generates correlated random returns so covariance estimators produce
    a valid positive-definite matrix.
    """
    rng = np.random.default_rng(42)
    num_obs = 100
    num_assets = 3

    # Generate correlated returns via a factor model
    factor = rng.standard_normal(num_obs) * 0.01
    noise = rng.standard_normal((num_obs, num_assets)) * 0.005
    returns = factor[:, np.newaxis] + noise

    dates = pl.date_range(
        pl.date(2023, 1, 1),
        pl.date(2023, 1, 1) + pl.duration(days=num_obs - 1),
        eager=True,
    ).alias("Date")

    return pl.DataFrame(
        {
            "Date": dates,
            "AAPL": returns[:, 0],
            "MSFT": returns[:, 1],
            "GOOG": returns[:, 2],
        },
    )


@pytest.fixture()
def small_returns_df() -> pl.DataFrame:
    """Fixture with small but valid returns (5 observations x 2 assets)."""
    return pl.DataFrame(
        {
            "Date": pl.date_range(
                pl.date(2024, 1, 1),
                pl.date(2024, 1, 5),
                eager=True,
            ),
            "Stock_A": [0.01, -0.02, 0.015, 0.005, -0.01],
            "Stock_B": [0.005, -0.01, 0.02, 0.01, -0.005],
        },
    )


@pytest.fixture()
def cov_matrix_instance(
    covariance_matrix_class,
    covariance_estimator_enum,
    sample_returns_df,
):
    """Fixture providing a constructed covariance_matrix instance."""
    return covariance_matrix_class(
        data_returns=sample_returns_df,
        estimator=covariance_estimator_enum.EMPIRICAL,
    )


# ===========================================================================
# covariance_estimator enum tests
# ===========================================================================


class Test_Covariance_Estimator_Enum:
    """Tests for the covariance_estimator enumeration."""

    @pytest.mark.unit()
    def test_all_members_exist(self, covariance_estimator_enum):
        """Test that all expected enum members are defined."""
        expected_members = [
            "EMPIRICAL",
            "GERBER",
            "DENOISING",
            "DETONING",
            "EXPONENTIALLY_WEIGHTED",
            "LEDOIT_WOLF",
            "ORACLE_APPROXIMATING_SHRINKAGE",
            "SHRUNK_COVARIANCE",
            "GRAPHICAL_LASSO_CV",
            "IMPLIED_COVARIANCE",
        ]
        for member in expected_members:
            assert hasattr(covariance_estimator_enum, member), f"Missing member: {member}"

    @pytest.mark.unit()
    def test_member_count(self, covariance_estimator_enum):
        """Test that enum has exactly 10 members."""
        members = list(covariance_estimator_enum)
        assert len(members) == 10

    @pytest.mark.unit()
    def test_values_are_strings(self, covariance_estimator_enum):
        """Test that all enum values are non-empty strings."""
        for member in covariance_estimator_enum:
            assert isinstance(member.value, str), f"{member.name} value is not a string"
            assert len(member.value) > 0, f"{member.name} has empty value"

    @pytest.mark.unit()
    def test_unique_values(self, covariance_estimator_enum):
        """Test that all enum values are unique."""
        values = [member.value for member in covariance_estimator_enum]
        assert len(values) == len(set(values)), "Duplicate values found"

    @pytest.mark.unit()
    def test_specific_values(self, covariance_estimator_enum):
        """Test that key enum members have expected values."""
        assert covariance_estimator_enum.EMPIRICAL.value == "Empirical"
        assert covariance_estimator_enum.LEDOIT_WOLF.value == "Ledoit-Wolf"

    @pytest.mark.unit()
    def test_access_by_value(self, covariance_estimator_enum):
        """Test enum member access by value."""
        member = covariance_estimator_enum("Empirical")
        assert member is covariance_estimator_enum.EMPIRICAL

    @pytest.mark.unit()
    def test_access_by_name(self, covariance_estimator_enum):
        """Test enum member access by name."""
        member = covariance_estimator_enum["EMPIRICAL"]
        assert member is covariance_estimator_enum.EMPIRICAL


# ===========================================================================
# distance_estimator_type enum tests
# ===========================================================================


class Test_Distance_Estimator_Type_Enum:
    """Tests for the distance_estimator_type enumeration."""

    @pytest.mark.unit()
    def test_all_members_exist(self, distance_estimator_type_enum):
        """Test that all expected enum members are defined."""
        expected_members = [
            "DISTANCE_PEARSON",
            "DISTANCE_KENDALL",
            "DISTANCE_SPEARMAN",
            "DISTANCE_COV_PEARSON",
            "DISTANCE_COV_KENDALL",
            "DISTANCE_COV_SPEARMAN",
            "VARIATION_OF_INFORMATION",
        ]
        for member in expected_members:
            assert hasattr(distance_estimator_type_enum, member), f"Missing member: {member}"

    @pytest.mark.unit()
    def test_member_count(self, distance_estimator_type_enum):
        """Test that enum has exactly 7 members."""
        members = list(distance_estimator_type_enum)
        assert len(members) == 7

    @pytest.mark.unit()
    def test_values_are_strings(self, distance_estimator_type_enum):
        """Test that all enum values are non-empty strings."""
        for member in distance_estimator_type_enum:
            assert isinstance(member.value, str)
            assert len(member.value) > 0

    @pytest.mark.unit()
    def test_unique_values(self, distance_estimator_type_enum):
        """Test that all enum values are unique."""
        values = [member.value for member in distance_estimator_type_enum]
        assert len(values) == len(set(values))


class Test_Distance_Estimator_Type_Classmethods:
    """Tests for distance_estimator_type @classmethod helpers."""

    @pytest.mark.unit()
    def test_get_correlation_based_count(self, distance_estimator_type_enum):
        """Test that get_correlation_based returns 3 members."""
        result = distance_estimator_type_enum.get_correlation_based()
        assert len(result) == 3

    @pytest.mark.unit()
    def test_get_correlation_based_members(self, distance_estimator_type_enum):
        """Test that get_correlation_based returns expected members."""
        result = distance_estimator_type_enum.get_correlation_based()
        expected_names = {"DISTANCE_PEARSON", "DISTANCE_KENDALL", "DISTANCE_SPEARMAN"}
        result_names = {m.name for m in result}
        assert result_names == expected_names

    @pytest.mark.unit()
    def test_get_covariance_based_count(self, distance_estimator_type_enum):
        """Test that get_covariance_based returns 3 members."""
        result = distance_estimator_type_enum.get_covariance_based()
        assert len(result) == 3

    @pytest.mark.unit()
    def test_get_covariance_based_members(self, distance_estimator_type_enum):
        """Test that get_covariance_based returns expected members."""
        result = distance_estimator_type_enum.get_covariance_based()
        expected_names = {
            "DISTANCE_COV_PEARSON",
            "DISTANCE_COV_KENDALL",
            "DISTANCE_COV_SPEARMAN",
        }
        result_names = {m.name for m in result}
        assert result_names == expected_names

    @pytest.mark.unit()
    def test_get_rank_based_count(self, distance_estimator_type_enum):
        """Test that get_rank_based returns 4 members."""
        result = distance_estimator_type_enum.get_rank_based()
        assert len(result) == 4

    @pytest.mark.unit()
    def test_get_rank_based_members(self, distance_estimator_type_enum):
        """Test that get_rank_based returns Kendall and Spearman variants."""
        result = distance_estimator_type_enum.get_rank_based()
        result_names = {m.name for m in result}
        expected_names = {
            "DISTANCE_KENDALL",
            "DISTANCE_SPEARMAN",
            "DISTANCE_COV_KENDALL",
            "DISTANCE_COV_SPEARMAN",
        }
        assert result_names == expected_names

    @pytest.mark.unit()
    def test_get_information_theoretic_count(self, distance_estimator_type_enum):
        """Test that get_information_theoretic returns 1 member."""
        result = distance_estimator_type_enum.get_information_theoretic()
        assert len(result) == 1

    @pytest.mark.unit()
    def test_get_information_theoretic_member(self, distance_estimator_type_enum):
        """Test that get_information_theoretic returns VARIATION_OF_INFORMATION."""
        result = distance_estimator_type_enum.get_information_theoretic()
        assert result[0].name == "VARIATION_OF_INFORMATION"

    @pytest.mark.unit()
    def test_correlation_and_covariance_disjoint(self, distance_estimator_type_enum):
        """Test that correlation-based and covariance-based sets are disjoint."""
        corr_set = {m.name for m in distance_estimator_type_enum.get_correlation_based()}
        cov_set = {m.name for m in distance_estimator_type_enum.get_covariance_based()}
        assert corr_set.isdisjoint(cov_set), "Correlation and covariance sets should be disjoint"

    @pytest.mark.unit()
    def test_all_rank_based_are_kendall_or_spearman(self, distance_estimator_type_enum):
        """Test that rank-based estimators only contain Kendall/Spearman variants."""
        result = distance_estimator_type_enum.get_rank_based()
        for member in result:
            assert "KENDALL" in member.name or "SPEARMAN" in member.name


# ===========================================================================
# covariance_matrix - construction tests
# ===========================================================================


class Test_Covariance_Matrix_Construction:
    """Tests for covariance_matrix initialization and construction."""

    @pytest.mark.unit()
    def test_basic_construction(
        self,
        covariance_matrix_class,
        covariance_estimator_enum,
        sample_returns_df,
    ):
        """Test that covariance_matrix can be constructed with valid inputs."""
        cov_mat = covariance_matrix_class(
            data_returns=sample_returns_df,
            estimator=covariance_estimator_enum.EMPIRICAL,
        )
        assert cov_mat is not None
        assert cov_mat.m_num_components == 3
        assert cov_mat.m_num_observations == 100

    @pytest.mark.unit()
    def test_component_names_extracted(
        self,
        covariance_matrix_class,
        covariance_estimator_enum,
        sample_returns_df,
    ):
        """Test that component names are correctly extracted from DataFrame."""
        cov_mat = covariance_matrix_class(
            data_returns=sample_returns_df,
            estimator=covariance_estimator_enum.EMPIRICAL,
        )
        assert cov_mat.m_component_names == ["AAPL", "MSFT", "GOOG"]

    @pytest.mark.unit()
    def test_estimator_type_stored(
        self,
        covariance_matrix_class,
        covariance_estimator_enum,
        sample_returns_df,
    ):
        """Test that estimator type is stored correctly."""
        cov_mat = covariance_matrix_class(
            data_returns=sample_returns_df,
            estimator=covariance_estimator_enum.EMPIRICAL,
        )
        assert cov_mat.m_estimator_type == covariance_estimator_enum.EMPIRICAL

    @pytest.mark.unit()
    def test_matrix_shape(
        self,
        covariance_matrix_class,
        covariance_estimator_enum,
        sample_returns_df,
    ):
        """Test that covariance matrix has correct shape (n x n)."""
        cov_mat = covariance_matrix_class(
            data_returns=sample_returns_df,
            estimator=covariance_estimator_enum.EMPIRICAL,
        )
        assert cov_mat.m_cov_matrix.shape == (3, 3)

    @pytest.mark.unit()
    def test_construction_with_small_data(
        self,
        covariance_matrix_class,
        covariance_estimator_enum,
        small_returns_df,
    ):
        """Test construction with small but valid dataset."""
        cov_mat = covariance_matrix_class(
            data_returns=small_returns_df,
            estimator=covariance_estimator_enum.EMPIRICAL,
        )
        assert cov_mat.m_num_components == 2
        assert cov_mat.m_num_observations == 5

    @pytest.mark.unit()
    def test_matrix_is_numpy_array(
        self,
        cov_matrix_instance,
    ):
        """Test that the covariance matrix is stored as numpy ndarray."""
        assert isinstance(cov_matrix_instance.m_cov_matrix, np.ndarray)


# ===========================================================================
# covariance_matrix - input validation tests
# ===========================================================================


class Test_Covariance_Matrix_Input_Validation:
    """Tests for input validation in covariance_matrix constructor."""

    @pytest.mark.unit()
    def test_reject_non_dataframe(
        self,
        covariance_matrix_class,
        covariance_estimator_enum,
        Exception_Validation_Input_class,
    ):
        """Test that non-DataFrame input raises Exception_Validation_Input."""
        with pytest.raises(Exception_Validation_Input_class, match="polars DataFrame"):
            covariance_matrix_class(
                data_returns=np.array([[1, 2], [3, 4]]),
                estimator=covariance_estimator_enum.EMPIRICAL,
            )

    @pytest.mark.unit()
    def test_reject_non_enum_estimator(
        self,
        covariance_matrix_class,
        sample_returns_df,
        Exception_Validation_Input_class,
    ):
        """Test that non-enum estimator raises Exception_Validation_Input."""
        with pytest.raises(Exception_Validation_Input_class, match="covariance_estimator"):
            covariance_matrix_class(
                data_returns=sample_returns_df,
                estimator="Empirical",
            )

    @pytest.mark.unit()
    def test_reject_empty_dataframe(
        self,
        covariance_matrix_class,
        covariance_estimator_enum,
        Exception_Validation_Input_class,
    ):
        """Test that empty DataFrame raises Exception_Validation_Input."""
        empty_df = pl.DataFrame({"Date": [], "A": [], "B": []}).cast(
            {"Date": pl.Date, "A": pl.Float64, "B": pl.Float64},
        )
        with pytest.raises(Exception_Validation_Input_class, match="empty"):
            covariance_matrix_class(
                data_returns=empty_df,
                estimator=covariance_estimator_enum.EMPIRICAL,
            )

    @pytest.mark.unit()
    def test_reject_missing_date_column(
        self,
        covariance_matrix_class,
        covariance_estimator_enum,
        Exception_Validation_Input_class,
    ):
        """Test that missing Date column raises Exception_Validation_Input."""
        no_date_df = pl.DataFrame({"A": [0.01, 0.02, 0.03], "B": [0.02, 0.01, -0.01]})
        with pytest.raises(Exception_Validation_Input_class, match="Date"):
            covariance_matrix_class(
                data_returns=no_date_df,
                estimator=covariance_estimator_enum.EMPIRICAL,
            )

    @pytest.mark.unit()
    def test_reject_single_component(
        self,
        covariance_matrix_class,
        covariance_estimator_enum,
        Exception_Validation_Input_class,
    ):
        """Test that single component (< 2) raises Exception_Validation_Input."""
        one_comp_df = pl.DataFrame(
            {
                "Date": pl.date_range(pl.date(2024, 1, 1), pl.date(2024, 1, 5), eager=True),
                "A": [0.01, 0.02, 0.03, -0.01, 0.005],
            },
        )
        with pytest.raises(Exception_Validation_Input_class, match="at least 2"):
            covariance_matrix_class(
                data_returns=one_comp_df,
                estimator=covariance_estimator_enum.EMPIRICAL,
            )

    @pytest.mark.unit()
    def test_reject_too_few_observations(
        self,
        covariance_matrix_class,
        covariance_estimator_enum,
        Exception_Validation_Input_class,
    ):
        """Test that fewer than 3 observations raises Exception_Validation_Input."""
        two_obs_df = pl.DataFrame(
            {
                "Date": pl.date_range(pl.date(2024, 1, 1), pl.date(2024, 1, 2), eager=True),
                "A": [0.01, 0.02],
                "B": [0.02, -0.01],
            },
        )
        with pytest.raises(Exception_Validation_Input_class, match="at least 3"):
            covariance_matrix_class(
                data_returns=two_obs_df,
                estimator=covariance_estimator_enum.EMPIRICAL,
            )

    @pytest.mark.unit()
    def test_reject_nan_in_returns(
        self,
        covariance_matrix_class,
        covariance_estimator_enum,
        Exception_Validation_Input_class,
    ):
        """Test that NaN values in returns raises Exception_Validation_Input."""
        nan_df = pl.DataFrame(
            {
                "Date": pl.date_range(pl.date(2024, 1, 1), pl.date(2024, 1, 4), eager=True),
                "A": [0.01, None, 0.03, 0.02],
                "B": [0.02, 0.01, -0.01, 0.005],
            },
        )
        with pytest.raises(Exception_Validation_Input_class, match="NaN"):
            covariance_matrix_class(
                data_returns=nan_df,
                estimator=covariance_estimator_enum.EMPIRICAL,
            )

    @pytest.mark.unit()
    def test_reject_infinite_in_returns(
        self,
        covariance_matrix_class,
        covariance_estimator_enum,
        Exception_Validation_Input_class,
    ):
        """Test that infinite values in returns raises Exception_Validation_Input."""
        inf_df = pl.DataFrame(
            {
                "Date": pl.date_range(pl.date(2024, 1, 1), pl.date(2024, 1, 4), eager=True),
                "A": [0.01, float("inf"), 0.03, 0.02],
                "B": [0.02, 0.01, -0.01, 0.005],
            },
        )
        with pytest.raises(Exception_Validation_Input_class, match="Infinite"):
            covariance_matrix_class(
                data_returns=inf_df,
                estimator=covariance_estimator_enum.EMPIRICAL,
            )


# ===========================================================================
# covariance_matrix - matrix properties tests
# ===========================================================================


class Test_Covariance_Matrix_Properties:
    """Tests for mathematical properties of the estimated covariance matrix."""

    @pytest.mark.unit()
    def test_matrix_is_symmetric(self, cov_matrix_instance):
        """Test that the covariance matrix is symmetric."""
        cov = cov_matrix_instance.m_cov_matrix
        np.testing.assert_allclose(cov, cov.T, atol=1e-10)

    @pytest.mark.unit()
    def test_diagonal_is_positive(self, cov_matrix_instance):
        """Test that all diagonal elements (variances) are positive."""
        diagonal = np.diag(cov_matrix_instance.m_cov_matrix)
        assert np.all(diagonal > 0), f"Non-positive diagonal: {diagonal}"

    @pytest.mark.unit()
    def test_positive_semidefinite(self, cov_matrix_instance):
        """Test that all eigenvalues are non-negative (PSD property)."""
        eigenvalues = np.linalg.eigvals(cov_matrix_instance.m_cov_matrix)
        assert np.all(eigenvalues.real >= -1e-10), (
            f"Negative eigenvalues found: {eigenvalues.real[eigenvalues.real < -1e-10]}"
        )

    @pytest.mark.unit()
    def test_no_nan_values(self, cov_matrix_instance):
        """Test that there are no NaN values in the covariance matrix."""
        assert not np.any(np.isnan(cov_matrix_instance.m_cov_matrix))

    @pytest.mark.unit()
    def test_no_infinite_values(self, cov_matrix_instance):
        """Test that there are no infinite values in the covariance matrix."""
        assert np.all(np.isfinite(cov_matrix_instance.m_cov_matrix))

    @pytest.mark.unit()
    def test_valid_correlations(self, cov_matrix_instance):
        """Test that implied correlations are in [-1, 1] range."""
        cov = cov_matrix_instance.m_cov_matrix
        std_dev = np.sqrt(np.diag(cov))
        corr = cov / np.outer(std_dev, std_dev)

        assert np.all(corr >= -1.0 - 1e-6), f"Correlation below -1: {np.min(corr)}"
        assert np.all(corr <= 1.0 + 1e-6), f"Correlation above 1: {np.max(corr)}"

    @pytest.mark.unit()
    def test_correlation_diagonal_is_one(self, cov_matrix_instance):
        """Test that the correlation matrix diagonal is all ones."""
        cov = cov_matrix_instance.m_cov_matrix
        std_dev = np.sqrt(np.diag(cov))
        corr = cov / np.outer(std_dev, std_dev)

        np.testing.assert_allclose(np.diag(corr), 1.0, atol=1e-6)


# ===========================================================================
# covariance_matrix - accessor method tests
# ===========================================================================


class Test_Covariance_Matrix_Accessors:
    """Tests for covariance_matrix accessor methods."""

    @pytest.mark.unit()
    def test_get_covariance_matrix_returns_copy(self, cov_matrix_instance):
        """Test that get_covariance_matrix returns a copy, not a reference."""
        result = cov_matrix_instance.get_covariance_matrix()
        assert result is not cov_matrix_instance.m_cov_matrix

    @pytest.mark.unit()
    def test_get_covariance_matrix_content(self, cov_matrix_instance):
        """Test that get_covariance_matrix returns identical values."""
        result = cov_matrix_instance.get_covariance_matrix()
        np.testing.assert_array_equal(result, cov_matrix_instance.m_cov_matrix)

    @pytest.mark.unit()
    def test_get_covariance_matrix_shape(self, cov_matrix_instance):
        """Test that get_covariance_matrix returns correct shape."""
        result = cov_matrix_instance.get_covariance_matrix()
        assert result.shape == (3, 3)

    @pytest.mark.unit()
    def test_get_correlation_matrix_shape(self, cov_matrix_instance):
        """Test that get_correlation_matrix returns correct shape."""
        result = cov_matrix_instance.get_correlation_matrix()
        assert result.shape == (3, 3)

    @pytest.mark.unit()
    def test_get_correlation_matrix_diagonal(self, cov_matrix_instance):
        """Test that correlation matrix diagonal is all ones."""
        corr = cov_matrix_instance.get_correlation_matrix()
        np.testing.assert_allclose(np.diag(corr), 1.0, atol=1e-6)

    @pytest.mark.unit()
    def test_get_correlation_matrix_range(self, cov_matrix_instance):
        """Test that all correlations are in [-1, 1] range."""
        corr = cov_matrix_instance.get_correlation_matrix()
        assert np.all(corr >= -1.0 - 1e-6)
        assert np.all(corr <= 1.0 + 1e-6)

    @pytest.mark.unit()
    def test_get_correlation_matrix_is_symmetric(self, cov_matrix_instance):
        """Test that the correlation matrix is symmetric."""
        corr = cov_matrix_instance.get_correlation_matrix()
        np.testing.assert_allclose(corr, corr.T, atol=1e-10)

    @pytest.mark.unit()
    def test_get_component_variance_returns_float(self, cov_matrix_instance):
        """Test that get_component_variance returns a Python float."""
        result = cov_matrix_instance.get_component_variance("AAPL")
        assert isinstance(result, float)

    @pytest.mark.unit()
    def test_get_component_variance_positive(self, cov_matrix_instance):
        """Test that component variance is positive."""
        for name in cov_matrix_instance.m_component_names:
            variance = cov_matrix_instance.get_component_variance(name)
            assert variance > 0, f"Variance for {name} not positive: {variance}"

    @pytest.mark.unit()
    def test_get_component_variance_matches_diagonal(self, cov_matrix_instance):
        """Test that component variance matches the diagonal of cov matrix."""
        for i, name in enumerate(cov_matrix_instance.m_component_names):
            variance = cov_matrix_instance.get_component_variance(name)
            expected = float(cov_matrix_instance.m_cov_matrix[i, i])
            assert variance == pytest.approx(expected, rel=1e-10)

    @pytest.mark.unit()
    def test_get_component_variance_unknown_name(
        self,
        cov_matrix_instance,
        Exception_Validation_Input_class,
    ):
        """Test that unknown component name raises Exception_Validation_Input."""
        with pytest.raises(Exception_Validation_Input_class, match="not found"):
            cov_matrix_instance.get_component_variance("UNKNOWN_TICKER")

    @pytest.mark.unit()
    def test_get_component_covariance_returns_float(self, cov_matrix_instance):
        """Test that get_component_covariance returns a Python float."""
        result = cov_matrix_instance.get_component_covariance("AAPL", "MSFT")
        assert isinstance(result, float)

    @pytest.mark.unit()
    def test_get_component_covariance_symmetric(self, cov_matrix_instance):
        """Test that covariance(A, B) == covariance(B, A)."""
        cov_ab = cov_matrix_instance.get_component_covariance("AAPL", "MSFT")
        cov_ba = cov_matrix_instance.get_component_covariance("MSFT", "AAPL")
        assert cov_ab == pytest.approx(cov_ba, rel=1e-10)

    @pytest.mark.unit()
    def test_get_component_covariance_self_equals_variance(self, cov_matrix_instance):
        """Test that cov(A, A) equals the variance of A."""
        for name in cov_matrix_instance.m_component_names:
            cov_self = cov_matrix_instance.get_component_covariance(name, name)
            variance = cov_matrix_instance.get_component_variance(name)
            assert cov_self == pytest.approx(variance, rel=1e-10)

    @pytest.mark.unit()
    def test_get_component_covariance_unknown_first(
        self,
        cov_matrix_instance,
        Exception_Validation_Input_class,
    ):
        """Test that unknown first component raises Exception_Validation_Input."""
        with pytest.raises(Exception_Validation_Input_class, match="not found"):
            cov_matrix_instance.get_component_covariance("UNKNOWN", "AAPL")

    @pytest.mark.unit()
    def test_get_component_covariance_unknown_second(
        self,
        cov_matrix_instance,
        Exception_Validation_Input_class,
    ):
        """Test that unknown second component raises Exception_Validation_Input."""
        with pytest.raises(Exception_Validation_Input_class, match="not found"):
            cov_matrix_instance.get_component_covariance("AAPL", "UNKNOWN")


# ===========================================================================
# covariance_matrix - validate_covariance_matrix tests
# ===========================================================================


class Test_Covariance_Matrix_Validation:
    """Tests for the validate_covariance_matrix sanity checks.

    These tests directly manipulate m_cov_matrix to trigger specific
    validation failures.
    """

    @pytest.mark.unit()
    def test_validate_passes_for_valid_matrix(self, cov_matrix_instance):
        """Test that validation passes for a valid covariance matrix."""
        # Should not raise
        cov_matrix_instance.validate_covariance_matrix()

    @pytest.mark.unit()
    def test_check_rejects_wrong_shape(
        self,
        cov_matrix_instance,
        Exception_Validation_Input_class,
    ):
        """Test that shape mismatch is detected."""
        cov_matrix_instance.m_cov_matrix = np.eye(5)  # Wrong shape for 3 components
        with pytest.raises(Exception_Validation_Input_class, match="shape"):
            cov_matrix_instance.validate_covariance_matrix()

    @pytest.mark.unit()
    def test_check_rejects_nan(
        self,
        cov_matrix_instance,
        Exception_Validation_Input_class,
    ):
        """Test that NaN values in matrix are detected."""
        cov_matrix_instance.m_cov_matrix[0, 0] = np.nan
        with pytest.raises(Exception_Validation_Input_class, match="NaN"):
            cov_matrix_instance.validate_covariance_matrix()

    @pytest.mark.unit()
    def test_check_rejects_infinite(
        self,
        cov_matrix_instance,
        Exception_Validation_Input_class,
    ):
        """Test that infinite values in matrix are detected."""
        cov_matrix_instance.m_cov_matrix[0, 1] = np.inf
        with pytest.raises(Exception_Validation_Input_class, match="infinite"):
            cov_matrix_instance.validate_covariance_matrix()

    @pytest.mark.unit()
    def test_check_rejects_asymmetric(
        self,
        cov_matrix_instance,
        Exception_Validation_Input_class,
    ):
        """Test that asymmetric matrix is detected."""
        cov_matrix_instance.m_cov_matrix[0, 1] += 1.0  # Break symmetry
        with pytest.raises(Exception_Validation_Input_class, match="symmetric"):
            cov_matrix_instance.validate_covariance_matrix()

    @pytest.mark.unit()
    def test_check_rejects_nonpositive_diagonal(
        self,
        cov_matrix_instance,
        Exception_Validation_Input_class,
    ):
        """Test that non-positive diagonal is detected."""
        cov_matrix_instance.m_cov_matrix[0, 0] = -0.001  # Negative variance
        with pytest.raises(Exception_Validation_Input_class, match="non-positive"):
            cov_matrix_instance.validate_covariance_matrix()

    @pytest.mark.unit()
    def test_check_rejects_not_psd(
        self,
        cov_matrix_instance,
        Exception_Validation_Input_class,
    ):
        """Test that non-PSD matrix is detected.

        Construct a symmetric matrix with positive diagonal but negative
        eigenvalue by making off-diagonal elements large.
        """
        n = cov_matrix_instance.m_num_components
        bad_matrix = np.eye(n) * 0.001
        # Off-diagonal much larger than diagonal → guaranteed negative eigenvalue
        bad_matrix[0, 1] = 10.0
        bad_matrix[1, 0] = 10.0
        cov_matrix_instance.m_cov_matrix = bad_matrix
        with pytest.raises(Exception_Validation_Input_class, match="positive semi-definite"):
            cov_matrix_instance.validate_covariance_matrix()


# ===========================================================================
# covariance_matrix - __str__ and __repr__ tests
# ===========================================================================


class Test_Covariance_Matrix_String_Repr:
    """Tests for __str__ and __repr__ methods."""

    @pytest.mark.unit()
    def test_str_contains_estimator(self, cov_matrix_instance):
        """Test that str contains the estimator name."""
        result = str(cov_matrix_instance)
        assert "Empirical" in result

    @pytest.mark.unit()
    def test_str_contains_component_count(self, cov_matrix_instance):
        """Test that str contains the component count."""
        result = str(cov_matrix_instance)
        assert "3" in result

    @pytest.mark.unit()
    def test_repr_contains_component_names(self, cov_matrix_instance):
        """Test that repr contains component names."""
        result = repr(cov_matrix_instance)
        assert "AAPL" in result
        assert "MSFT" in result
        assert "GOOG" in result

    @pytest.mark.unit()
    def test_repr_contains_shape(self, cov_matrix_instance):
        """Test that repr contains matrix shape."""
        result = repr(cov_matrix_instance)
        assert "(3, 3)" in result


# ===========================================================================
# covariance_matrix - multiple estimator types
# ===========================================================================


class Test_Covariance_Matrix_Estimator_Types:
    """Tests for different covariance estimator types.

    Each estimator is tested with the same input data to ensure all
    produce valid covariance matrices.
    """

    @pytest.mark.unit()
    @pytest.mark.parametrize(
        "estimator_name",
        [
            "EMPIRICAL",
            "LEDOIT_WOLF",
            "ORACLE_APPROXIMATING_SHRINKAGE",
            "SHRUNK_COVARIANCE",
            "EXPONENTIALLY_WEIGHTED",
        ],
    )
    def test_estimator_produces_valid_matrix(
        self,
        covariance_matrix_class,
        covariance_estimator_enum,
        sample_returns_df,
        estimator_name,
    ):
        """Test that each estimator produces a valid covariance matrix."""
        estimator = getattr(covariance_estimator_enum, estimator_name)
        cov_mat = covariance_matrix_class(
            data_returns=sample_returns_df,
            estimator=estimator,
        )

        # Shape should be (n, n)
        assert cov_mat.m_cov_matrix.shape == (3, 3)

        # Should be symmetric
        np.testing.assert_allclose(
            cov_mat.m_cov_matrix,
            cov_mat.m_cov_matrix.T,
            atol=1e-10,
        )

        # Diagonal should be positive
        assert np.all(np.diag(cov_mat.m_cov_matrix) > 0)

        # Should be PSD
        eigenvalues = np.linalg.eigvals(cov_mat.m_cov_matrix)
        assert np.all(eigenvalues.real >= -1e-10)

    @pytest.mark.unit()
    @pytest.mark.parametrize(
        "estimator_name",
        [
            "EMPIRICAL",
            "LEDOIT_WOLF",
            "ORACLE_APPROXIMATING_SHRINKAGE",
            "SHRUNK_COVARIANCE",
            "EXPONENTIALLY_WEIGHTED",
        ],
    )
    def test_estimator_correlation_valid(
        self,
        covariance_matrix_class,
        covariance_estimator_enum,
        sample_returns_df,
        estimator_name,
    ):
        """Test that each estimator produces valid correlations in [-1, 1]."""
        estimator = getattr(covariance_estimator_enum, estimator_name)
        cov_mat = covariance_matrix_class(
            data_returns=sample_returns_df,
            estimator=estimator,
        )

        corr = cov_mat.get_correlation_matrix()
        assert np.all(corr >= -1.0 - 1e-6)
        assert np.all(corr <= 1.0 + 1e-6)
        np.testing.assert_allclose(np.diag(corr), 1.0, atol=1e-6)
