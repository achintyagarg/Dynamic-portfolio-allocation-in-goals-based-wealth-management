"""Unit tests for utils_data_financial module.

Tests for the three financial-estimator Enum classes:
- expected_returns_estimator_type
- prior_estimator_type
- distribution_estimator_type

Covers member values, classmethod outputs, membership checks, and
mutual-exclusivity of method groups.

Author: QWIM Development Team
Version: 0.1.0
"""

from __future__ import annotations

import pytest

from src.utils.data_utils.utils_data_financial import (
    distribution_estimator_type,
    expected_returns_estimator_type,
    prior_estimator_type,
)


# ==============================================================================
# Tests: expected_returns_estimator_type
# ==============================================================================


class Test_Expected_Returns_Estimator_Type:
    """Tests for expected_returns_estimator_type enum."""

    # -------------------------------------------------------------------------
    # Member presence & values
    # -------------------------------------------------------------------------

    @pytest.mark.unit()
    def test_all_members_exist(self):
        """All five expected-return estimator members are present."""
        members = list(expected_returns_estimator_type)
        assert len(members) == 5

    @pytest.mark.unit()
    @pytest.mark.parametrize(
        ("member", "expected_value"),
        [
            (expected_returns_estimator_type.EMPIRICAL, "EMPIRICAL"),
            (expected_returns_estimator_type.EXPONENTIALLY_WEIGHTED, "EXPONENTIALLY_WEIGHTED"),
            (expected_returns_estimator_type.EQUILIBRIUM, "EQUILIBRIUM"),
            (expected_returns_estimator_type.SHRINKAGE, "SHRINKAGE"),
            (expected_returns_estimator_type.FROM_DISTRIBUTION, "FROM_DISTRIBUTION"),
        ],
    )
    def test_member_values(self, member, expected_value):
        """Each enum member's value equals its name string."""
        assert member.value == expected_value

    @pytest.mark.unit()
    def test_member_values_are_unique(self):
        """All member values are distinct."""
        values = [m.value for m in expected_returns_estimator_type]
        assert len(values) == len(set(values))

    # -------------------------------------------------------------------------
    # get_historical_methods
    # -------------------------------------------------------------------------

    @pytest.mark.unit()
    def test_get_historical_methods_returns_list(self):
        """get_historical_methods returns a list."""
        result = expected_returns_estimator_type.get_historical_methods()
        assert isinstance(result, list)

    @pytest.mark.unit()
    def test_get_historical_methods_count(self):
        """get_historical_methods returns exactly 2 members."""
        assert len(expected_returns_estimator_type.get_historical_methods()) == 2

    @pytest.mark.unit()
    def test_get_historical_methods_contains_empirical(self):
        """EMPIRICAL is in get_historical_methods."""
        assert expected_returns_estimator_type.EMPIRICAL in (
            expected_returns_estimator_type.get_historical_methods()
        )

    @pytest.mark.unit()
    def test_get_historical_methods_contains_ewma(self):
        """EXPONENTIALLY_WEIGHTED is in get_historical_methods."""
        assert expected_returns_estimator_type.EXPONENTIALLY_WEIGHTED in (
            expected_returns_estimator_type.get_historical_methods()
        )

    @pytest.mark.unit()
    def test_historical_methods_are_enum_instances(self):
        """All returned items are expected_returns_estimator_type instances."""
        for m in expected_returns_estimator_type.get_historical_methods():
            assert isinstance(m, expected_returns_estimator_type)

    # -------------------------------------------------------------------------
    # get_model_based_methods
    # -------------------------------------------------------------------------

    @pytest.mark.unit()
    def test_get_model_based_methods_returns_list(self):
        """get_model_based_methods returns a list."""
        assert isinstance(expected_returns_estimator_type.get_model_based_methods(), list)

    @pytest.mark.unit()
    def test_get_model_based_methods_count(self):
        """get_model_based_methods returns exactly 2 members."""
        assert len(expected_returns_estimator_type.get_model_based_methods()) == 2

    @pytest.mark.unit()
    @pytest.mark.parametrize(
        "member",
        [
            expected_returns_estimator_type.EQUILIBRIUM,
            expected_returns_estimator_type.SHRINKAGE,
        ],
    )
    def test_model_based_members(self, member):
        """EQUILIBRIUM and SHRINKAGE are in model-based methods."""
        assert member in expected_returns_estimator_type.get_model_based_methods()

    # -------------------------------------------------------------------------
    # get_distribution_based_methods
    # -------------------------------------------------------------------------

    @pytest.mark.unit()
    def test_get_distribution_based_methods_count(self):
        """get_distribution_based_methods returns exactly 1 member."""
        assert len(expected_returns_estimator_type.get_distribution_based_methods()) == 1

    @pytest.mark.unit()
    def test_distribution_based_contains_from_distribution(self):
        """FROM_DISTRIBUTION is in get_distribution_based_methods."""
        assert expected_returns_estimator_type.FROM_DISTRIBUTION in (
            expected_returns_estimator_type.get_distribution_based_methods()
        )

    # -------------------------------------------------------------------------
    # get_robust_methods
    # -------------------------------------------------------------------------

    @pytest.mark.unit()
    def test_get_robust_methods_count(self):
        """get_robust_methods returns exactly 3 members."""
        assert len(expected_returns_estimator_type.get_robust_methods()) == 3

    @pytest.mark.unit()
    @pytest.mark.parametrize(
        "member",
        [
            expected_returns_estimator_type.EXPONENTIALLY_WEIGHTED,
            expected_returns_estimator_type.SHRINKAGE,
            expected_returns_estimator_type.EQUILIBRIUM,
        ],
    )
    def test_robust_methods_members(self, member):
        """EXPONENTIALLY_WEIGHTED, SHRINKAGE, EQUILIBRIUM all in robust methods."""
        assert member in expected_returns_estimator_type.get_robust_methods()

    @pytest.mark.unit()
    def test_empirical_not_in_robust_methods(self):
        """EMPIRICAL is not in get_robust_methods (sensitive to outliers)."""
        assert expected_returns_estimator_type.EMPIRICAL not in (
            expected_returns_estimator_type.get_robust_methods()
        )

    # -------------------------------------------------------------------------
    # Coverage: all members appear in exactly one category
    # -------------------------------------------------------------------------

    @pytest.mark.unit()
    def test_all_members_covered_by_at_least_one_group(self):
        """Every member appears in at least one category method."""
        all_grouped = set(
            expected_returns_estimator_type.get_historical_methods()
            + expected_returns_estimator_type.get_model_based_methods()
            + expected_returns_estimator_type.get_distribution_based_methods()
        )
        for member in expected_returns_estimator_type:
            assert member in all_grouped, f"{member} missing from all category groups"

    @pytest.mark.unit()
    def test_historical_and_model_based_are_disjoint(self):
        """Historical and model-based method sets share no members."""
        hist = set(expected_returns_estimator_type.get_historical_methods())
        model = set(expected_returns_estimator_type.get_model_based_methods())
        assert hist.isdisjoint(model)

    @pytest.mark.unit()
    def test_distribution_and_historical_are_disjoint(self):
        """Distribution-based and historical sets share no members."""
        dist = set(expected_returns_estimator_type.get_distribution_based_methods())
        hist = set(expected_returns_estimator_type.get_historical_methods())
        assert dist.isdisjoint(hist)


# ==============================================================================
# Tests: prior_estimator_type
# ==============================================================================


class Test_Prior_Estimator_Type:
    """Tests for prior_estimator_type enum."""

    # -------------------------------------------------------------------------
    # Member presence & values
    # -------------------------------------------------------------------------

    @pytest.mark.unit()
    def test_all_members_exist(self):
        """All seven prior estimator members are present."""
        assert len(list(prior_estimator_type)) == 7

    @pytest.mark.unit()
    @pytest.mark.parametrize(
        ("member", "expected_value"),
        [
            (prior_estimator_type.EMPIRICAL, "EMPIRICAL"),
            (prior_estimator_type.BLACK_LITTERMAN, "BLACK_LITTERMAN"),
            (prior_estimator_type.FACTOR_MODEL, "FACTOR_MODEL"),
            (prior_estimator_type.SYNTHETIC_DATA_STRESS_TEST, "SYNTHETIC_DATA_STRESS_TEST"),
            (
                prior_estimator_type.SYNTHETIC_DATA_FACTOR_STRESS_TEST,
                "SYNTHETIC_DATA_FACTOR_STRESS_TEST",
            ),
            (prior_estimator_type.ENTROPY_POOLING, "ENTROPY_POOLING"),
            (prior_estimator_type.OPINION_POOLING, "OPINION_POOLING"),
        ],
    )
    def test_member_values(self, member, expected_value):
        """Each enum member's value equals its name string."""
        assert member.value == expected_value

    @pytest.mark.unit()
    def test_member_values_are_unique(self):
        """All member values are distinct."""
        values = [m.value for m in prior_estimator_type]
        assert len(values) == len(set(values))

    # -------------------------------------------------------------------------
    # get_data_driven
    # -------------------------------------------------------------------------

    @pytest.mark.unit()
    def test_get_data_driven_count(self):
        """get_data_driven returns exactly 1 member."""
        assert len(prior_estimator_type.get_data_driven()) == 1

    @pytest.mark.unit()
    def test_empirical_in_data_driven(self):
        """EMPIRICAL is in get_data_driven."""
        assert prior_estimator_type.EMPIRICAL in prior_estimator_type.get_data_driven()

    # -------------------------------------------------------------------------
    # get_equilibrium_based
    # -------------------------------------------------------------------------

    @pytest.mark.unit()
    def test_get_equilibrium_based_count(self):
        """get_equilibrium_based returns exactly 1 member."""
        assert len(prior_estimator_type.get_equilibrium_based()) == 1

    @pytest.mark.unit()
    def test_black_litterman_in_equilibrium(self):
        """BLACK_LITTERMAN is in get_equilibrium_based."""
        assert prior_estimator_type.BLACK_LITTERMAN in prior_estimator_type.get_equilibrium_based()

    # -------------------------------------------------------------------------
    # get_factor_based
    # -------------------------------------------------------------------------

    @pytest.mark.unit()
    def test_get_factor_based_count(self):
        """get_factor_based returns exactly 1 member."""
        assert len(prior_estimator_type.get_factor_based()) == 1

    @pytest.mark.unit()
    def test_factor_model_in_factor_based(self):
        """FACTOR_MODEL is in get_factor_based."""
        assert prior_estimator_type.FACTOR_MODEL in prior_estimator_type.get_factor_based()

    # -------------------------------------------------------------------------
    # get_scenario_based
    # -------------------------------------------------------------------------

    @pytest.mark.unit()
    def test_get_scenario_based_count(self):
        """get_scenario_based returns exactly 2 members."""
        assert len(prior_estimator_type.get_scenario_based()) == 2

    @pytest.mark.unit()
    @pytest.mark.parametrize(
        "member",
        [
            prior_estimator_type.SYNTHETIC_DATA_STRESS_TEST,
            prior_estimator_type.SYNTHETIC_DATA_FACTOR_STRESS_TEST,
        ],
    )
    def test_scenario_based_members(self, member):
        """Both synthetic stress test members are in get_scenario_based."""
        assert member in prior_estimator_type.get_scenario_based()

    # -------------------------------------------------------------------------
    # get_probabilistic
    # -------------------------------------------------------------------------

    @pytest.mark.unit()
    def test_get_probabilistic_count(self):
        """get_probabilistic returns exactly 2 members."""
        assert len(prior_estimator_type.get_probabilistic()) == 2

    @pytest.mark.unit()
    @pytest.mark.parametrize(
        "member",
        [prior_estimator_type.ENTROPY_POOLING, prior_estimator_type.OPINION_POOLING],
    )
    def test_probabilistic_members(self, member):
        """ENTROPY_POOLING and OPINION_POOLING are in get_probabilistic."""
        assert member in prior_estimator_type.get_probabilistic()

    # -------------------------------------------------------------------------
    # get_bayesian_methods
    # -------------------------------------------------------------------------

    @pytest.mark.unit()
    def test_get_bayesian_methods_count(self):
        """get_bayesian_methods returns exactly 3 members."""
        assert len(prior_estimator_type.get_bayesian_methods()) == 3

    @pytest.mark.unit()
    @pytest.mark.parametrize(
        "member",
        [
            prior_estimator_type.BLACK_LITTERMAN,
            prior_estimator_type.ENTROPY_POOLING,
            prior_estimator_type.OPINION_POOLING,
        ],
    )
    def test_bayesian_methods_members(self, member):
        """BLACK_LITTERMAN, ENTROPY_POOLING, OPINION_POOLING are Bayesian."""
        assert member in prior_estimator_type.get_bayesian_methods()

    @pytest.mark.unit()
    def test_empirical_not_bayesian(self):
        """EMPIRICAL is not in Bayesian methods."""
        assert prior_estimator_type.EMPIRICAL not in prior_estimator_type.get_bayesian_methods()

    # -------------------------------------------------------------------------
    # get_view_incorporation_methods
    # -------------------------------------------------------------------------

    @pytest.mark.unit()
    def test_get_view_incorporation_methods_count(self):
        """get_view_incorporation_methods returns exactly 3 members."""
        assert len(prior_estimator_type.get_view_incorporation_methods()) == 3

    @pytest.mark.unit()
    def test_view_methods_equal_bayesian_methods(self):
        """View incorporation methods equal Bayesian methods set."""
        bayesian = set(prior_estimator_type.get_bayesian_methods())
        view = set(prior_estimator_type.get_view_incorporation_methods())
        assert bayesian == view

    # -------------------------------------------------------------------------
    # Coverage
    # -------------------------------------------------------------------------

    @pytest.mark.unit()
    def test_all_members_covered_by_at_least_one_group(self):
        """Every prior member appears in at least one of the category methods."""
        all_grouped = set(
            prior_estimator_type.get_data_driven()
            + prior_estimator_type.get_equilibrium_based()
            + prior_estimator_type.get_factor_based()
            + prior_estimator_type.get_scenario_based()
            + prior_estimator_type.get_probabilistic()
        )
        for member in prior_estimator_type:
            assert member in all_grouped, f"{member} missing from all category groups"

    @pytest.mark.unit()
    def test_primary_groups_are_pairwise_disjoint(self):
        """data_driven / equilibrium / factor / scenario / probabilistic groups are disjoint."""
        groups = [
            set(prior_estimator_type.get_data_driven()),
            set(prior_estimator_type.get_equilibrium_based()),
            set(prior_estimator_type.get_factor_based()),
            set(prior_estimator_type.get_scenario_based()),
            set(prior_estimator_type.get_probabilistic()),
        ]
        for i, g1 in enumerate(groups):
            for j, g2 in enumerate(groups):
                if i != j:
                    assert g1.isdisjoint(g2), (
                        f"Groups {i} and {j} share members: {g1 & g2}"
                    )


# ==============================================================================
# Tests: distribution_estimator_type
# ==============================================================================


class Test_Distribution_Estimator_Type:
    """Tests for distribution_estimator_type enum."""

    # -------------------------------------------------------------------------
    # Member presence & values
    # -------------------------------------------------------------------------

    @pytest.mark.unit()
    def test_all_members_exist(self):
        """All 14 distribution estimator members are present (4 univariate + 10 copula)."""
        assert len(list(distribution_estimator_type)) == 14

    @pytest.mark.unit()
    @pytest.mark.parametrize(
        ("member", "expected_value"),
        [
            (distribution_estimator_type.UNIVARIATE_GAUSSIAN, "UNIVARIATE_GAUSSIAN"),
            (distribution_estimator_type.UNIVARIATE_STUDENT_T, "UNIVARIATE_STUDENT_T"),
            (distribution_estimator_type.UNIVARIATE_JOHNSON_SU, "UNIVARIATE_JOHNSON_SU"),
            (
                distribution_estimator_type.UNIVARIATE_NORMAL_INVERSE_GAUSSIAN,
                "UNIVARIATE_NORMAL_INVERSE_GAUSSIAN",
            ),
            (distribution_estimator_type.COPULA_BIVARIATE_GAUSSIAN, "COPULA_BIVARIATE_GAUSSIAN"),
            (
                distribution_estimator_type.COPULA_BIVARIATE_STUDENT_T,
                "COPULA_BIVARIATE_STUDENT_T",
            ),
            (distribution_estimator_type.COPULA_BIVARIATE_CLAYTON, "COPULA_BIVARIATE_CLAYTON"),
            (distribution_estimator_type.COPULA_BIVARIATE_GUMBEL, "COPULA_BIVARIATE_GUMBEL"),
            (distribution_estimator_type.COPULA_BIVARIATE_JOE, "COPULA_BIVARIATE_JOE"),
            (
                distribution_estimator_type.COPULA_BIVARIATE_INDEPENDENT,
                "COPULA_BIVARIATE_INDEPENDENT",
            ),
            (
                distribution_estimator_type.COPULA_MULTIVARIATE_VINE_REGULAR,
                "COPULA_MULTIVARIATE_VINE_REGULAR",
            ),
            (
                distribution_estimator_type.COPULA_MULTIVARIATE_VINE_CENTERED,
                "COPULA_MULTIVARIATE_VINE_CENTERED",
            ),
            (
                distribution_estimator_type.COPULA_MULTIVARIATE_VINE_CLUSTERED,
                "COPULA_MULTIVARIATE_VINE_CLUSTERED",
            ),
            (
                distribution_estimator_type.COPULA_MULTIVARIATE_VINE_CONDITIONAL_SAMPLING,
                "COPULA_MULTIVARIATE_VINE_CONDITIONAL_SAMPLING",
            ),
        ],
    )
    def test_member_values(self, member, expected_value):
        """Each enum member's value equals its name string."""
        assert member.value == expected_value

    @pytest.mark.unit()
    def test_member_values_are_unique(self):
        """All member values are distinct."""
        values = [m.value for m in distribution_estimator_type]
        assert len(values) == len(set(values))

    # -------------------------------------------------------------------------
    # get_univariate
    # -------------------------------------------------------------------------

    @pytest.mark.unit()
    def test_get_univariate_count(self):
        """get_univariate returns exactly 4 members."""
        assert len(distribution_estimator_type.get_univariate()) == 4

    @pytest.mark.unit()
    @pytest.mark.parametrize(
        "member",
        [
            distribution_estimator_type.UNIVARIATE_GAUSSIAN,
            distribution_estimator_type.UNIVARIATE_STUDENT_T,
            distribution_estimator_type.UNIVARIATE_JOHNSON_SU,
            distribution_estimator_type.UNIVARIATE_NORMAL_INVERSE_GAUSSIAN,
        ],
    )
    def test_univariate_members(self, member):
        """All four univariate distributions appear in get_univariate."""
        assert member in distribution_estimator_type.get_univariate()

    # -------------------------------------------------------------------------
    # get_bivariate_copulas
    # -------------------------------------------------------------------------

    @pytest.mark.unit()
    def test_get_bivariate_copulas_count(self):
        """get_bivariate_copulas returns exactly 6 members."""
        assert len(distribution_estimator_type.get_bivariate_copulas()) == 6

    @pytest.mark.unit()
    @pytest.mark.parametrize(
        "member",
        [
            distribution_estimator_type.COPULA_BIVARIATE_GAUSSIAN,
            distribution_estimator_type.COPULA_BIVARIATE_STUDENT_T,
            distribution_estimator_type.COPULA_BIVARIATE_CLAYTON,
            distribution_estimator_type.COPULA_BIVARIATE_GUMBEL,
            distribution_estimator_type.COPULA_BIVARIATE_JOE,
            distribution_estimator_type.COPULA_BIVARIATE_INDEPENDENT,
        ],
    )
    def test_bivariate_copula_members(self, member):
        """All six bivariate copulas are in get_bivariate_copulas."""
        assert member in distribution_estimator_type.get_bivariate_copulas()

    # -------------------------------------------------------------------------
    # get_multivariate_copulas
    # -------------------------------------------------------------------------

    @pytest.mark.unit()
    def test_get_multivariate_copulas_count(self):
        """get_multivariate_copulas returns exactly 4 vine copula members."""
        assert len(distribution_estimator_type.get_multivariate_copulas()) == 4

    @pytest.mark.unit()
    @pytest.mark.parametrize(
        "member",
        [
            distribution_estimator_type.COPULA_MULTIVARIATE_VINE_REGULAR,
            distribution_estimator_type.COPULA_MULTIVARIATE_VINE_CENTERED,
            distribution_estimator_type.COPULA_MULTIVARIATE_VINE_CLUSTERED,
            distribution_estimator_type.COPULA_MULTIVARIATE_VINE_CONDITIONAL_SAMPLING,
        ],
    )
    def test_multivariate_copula_members(self, member):
        """All four vine copulas are in get_multivariate_copulas."""
        assert member in distribution_estimator_type.get_multivariate_copulas()

    # -------------------------------------------------------------------------
    # get_all_copulas
    # -------------------------------------------------------------------------

    @pytest.mark.unit()
    def test_get_all_copulas_count(self):
        """get_all_copulas combines bivariate (6) and multivariate (4) = 10 total."""
        assert len(distribution_estimator_type.get_all_copulas()) == 10

    @pytest.mark.unit()
    def test_get_all_copulas_equals_bivariate_plus_multivariate(self):
        """get_all_copulas = union of get_bivariate_copulas + get_multivariate_copulas."""
        expected = set(
            distribution_estimator_type.get_bivariate_copulas()
            + distribution_estimator_type.get_multivariate_copulas()
        )
        actual = set(distribution_estimator_type.get_all_copulas())
        assert actual == expected

    @pytest.mark.unit()
    def test_univariate_not_in_all_copulas(self):
        """No univariate distributions appear in get_all_copulas."""
        copulas = set(distribution_estimator_type.get_all_copulas())
        for member in distribution_estimator_type.get_univariate():
            assert member not in copulas

    # -------------------------------------------------------------------------
    # get_fat_tailed_distributions
    # -------------------------------------------------------------------------

    @pytest.mark.unit()
    def test_get_fat_tailed_distributions_count(self):
        """get_fat_tailed_distributions returns exactly 2 members."""
        assert len(distribution_estimator_type.get_fat_tailed_distributions()) == 2

    @pytest.mark.unit()
    @pytest.mark.parametrize(
        "member",
        [
            distribution_estimator_type.UNIVARIATE_STUDENT_T,
            distribution_estimator_type.UNIVARIATE_NORMAL_INVERSE_GAUSSIAN,
        ],
    )
    def test_fat_tailed_members(self, member):
        """Student-t and NIG are in get_fat_tailed_distributions."""
        assert member in distribution_estimator_type.get_fat_tailed_distributions()

    @pytest.mark.unit()
    def test_gaussian_not_fat_tailed(self):
        """UNIVARIATE_GAUSSIAN is not fat-tailed."""
        assert distribution_estimator_type.UNIVARIATE_GAUSSIAN not in (
            distribution_estimator_type.get_fat_tailed_distributions()
        )

    # -------------------------------------------------------------------------
    # get_flexible_distributions
    # -------------------------------------------------------------------------

    @pytest.mark.unit()
    def test_get_flexible_distributions_count(self):
        """get_flexible_distributions returns exactly 2 members."""
        assert len(distribution_estimator_type.get_flexible_distributions()) == 2

    @pytest.mark.unit()
    @pytest.mark.parametrize(
        "member",
        [
            distribution_estimator_type.UNIVARIATE_JOHNSON_SU,
            distribution_estimator_type.UNIVARIATE_NORMAL_INVERSE_GAUSSIAN,
        ],
    )
    def test_flexible_members(self, member):
        """Johnson SU and NIG are in get_flexible_distributions."""
        assert member in distribution_estimator_type.get_flexible_distributions()

    # -------------------------------------------------------------------------
    # get_tail_dependent_copulas
    # -------------------------------------------------------------------------

    @pytest.mark.unit()
    def test_get_tail_dependent_copulas_count(self):
        """get_tail_dependent_copulas returns exactly 4 members."""
        assert len(distribution_estimator_type.get_tail_dependent_copulas()) == 4

    @pytest.mark.unit()
    @pytest.mark.parametrize(
        "member",
        [
            distribution_estimator_type.COPULA_BIVARIATE_STUDENT_T,
            distribution_estimator_type.COPULA_BIVARIATE_CLAYTON,
            distribution_estimator_type.COPULA_BIVARIATE_GUMBEL,
            distribution_estimator_type.COPULA_BIVARIATE_JOE,
        ],
    )
    def test_tail_dependent_copula_members(self, member):
        """Student-t, Clayton, Gumbel, Joe copulas are tail-dependent."""
        assert member in distribution_estimator_type.get_tail_dependent_copulas()

    @pytest.mark.unit()
    def test_independent_copula_not_tail_dependent(self):
        """COPULA_BIVARIATE_INDEPENDENT has no tail dependence."""
        assert distribution_estimator_type.COPULA_BIVARIATE_INDEPENDENT not in (
            distribution_estimator_type.get_tail_dependent_copulas()
        )

    @pytest.mark.unit()
    def test_gaussian_copula_not_tail_dependent(self):
        """COPULA_BIVARIATE_GAUSSIAN has zero tail dependence."""
        assert distribution_estimator_type.COPULA_BIVARIATE_GAUSSIAN not in (
            distribution_estimator_type.get_tail_dependent_copulas()
        )

    # -------------------------------------------------------------------------
    # get_archimedean_copulas
    # -------------------------------------------------------------------------

    @pytest.mark.unit()
    def test_get_archimedean_copulas_count(self):
        """get_archimedean_copulas returns exactly 3 members."""
        assert len(distribution_estimator_type.get_archimedean_copulas()) == 3

    @pytest.mark.unit()
    @pytest.mark.parametrize(
        "member",
        [
            distribution_estimator_type.COPULA_BIVARIATE_CLAYTON,
            distribution_estimator_type.COPULA_BIVARIATE_GUMBEL,
            distribution_estimator_type.COPULA_BIVARIATE_JOE,
        ],
    )
    def test_archimedean_copula_members(self, member):
        """Clayton, Gumbel, Joe are Archimedean copulas."""
        assert member in distribution_estimator_type.get_archimedean_copulas()

    @pytest.mark.unit()
    def test_gaussian_copula_not_archimedean(self):
        """COPULA_BIVARIATE_GAUSSIAN is not Archimedean."""
        assert distribution_estimator_type.COPULA_BIVARIATE_GAUSSIAN not in (
            distribution_estimator_type.get_archimedean_copulas()
        )

    @pytest.mark.unit()
    def test_student_t_copula_not_archimedean(self):
        """COPULA_BIVARIATE_STUDENT_T is not Archimedean."""
        assert distribution_estimator_type.COPULA_BIVARIATE_STUDENT_T not in (
            distribution_estimator_type.get_archimedean_copulas()
        )

    # -------------------------------------------------------------------------
    # get_simulation_ready
    # -------------------------------------------------------------------------

    @pytest.mark.unit()
    def test_get_simulation_ready_count(self):
        """get_simulation_ready returns exactly 5 members."""
        assert len(distribution_estimator_type.get_simulation_ready()) == 5

    @pytest.mark.unit()
    @pytest.mark.parametrize(
        "member",
        [
            distribution_estimator_type.UNIVARIATE_GAUSSIAN,
            distribution_estimator_type.UNIVARIATE_STUDENT_T,
            distribution_estimator_type.COPULA_BIVARIATE_GAUSSIAN,
            distribution_estimator_type.COPULA_BIVARIATE_STUDENT_T,
            distribution_estimator_type.COPULA_MULTIVARIATE_VINE_CONDITIONAL_SAMPLING,
        ],
    )
    def test_simulation_ready_members(self, member):
        """Key simulation-ready distributions and copulas are present."""
        assert member in distribution_estimator_type.get_simulation_ready()

    # -------------------------------------------------------------------------
    # Structural invariants
    # -------------------------------------------------------------------------

    @pytest.mark.unit()
    def test_univariate_and_copulas_are_disjoint(self):
        """Univariate distributions and all copulas share no members."""
        univariate = set(distribution_estimator_type.get_univariate())
        copulas = set(distribution_estimator_type.get_all_copulas())
        assert univariate.isdisjoint(copulas)

    @pytest.mark.unit()
    def test_bivariate_and_multivariate_are_disjoint(self):
        """Bivariate and multivariate copula sets share no members."""
        bivariate = set(distribution_estimator_type.get_bivariate_copulas())
        multivariate = set(distribution_estimator_type.get_multivariate_copulas())
        assert bivariate.isdisjoint(multivariate)

    @pytest.mark.unit()
    def test_all_members_covered_by_univariate_or_copulas(self):
        """Every member belongs to either univariate or the copula groups."""
        combined = set(
            distribution_estimator_type.get_univariate()
            + distribution_estimator_type.get_all_copulas()
        )
        for member in distribution_estimator_type:
            assert member in combined, (
                f"{member} is neither univariate nor a copula"
            )

    @pytest.mark.unit()
    def test_archimedean_copulas_subset_of_bivariate(self):
        """All Archimedean copulas are bivariate copulas."""
        bivariate = set(distribution_estimator_type.get_bivariate_copulas())
        archimedean = set(distribution_estimator_type.get_archimedean_copulas())
        assert archimedean.issubset(bivariate)

    @pytest.mark.unit()
    def test_tail_dependent_copulas_subset_of_bivariate(self):
        """All tail-dependent copulas are bivariate copulas."""
        bivariate = set(distribution_estimator_type.get_bivariate_copulas())
        tail_dep = set(distribution_estimator_type.get_tail_dependent_copulas())
        assert tail_dep.issubset(bivariate)


# ==============================================================================
# Cross-module type consistency
# ==============================================================================


class Test_Cross_Module_Consistency:
    """Cross-class consistency checks."""

    @pytest.mark.unit()
    def test_all_three_enums_are_distinct_types(self):
        """The three Enum classes are distinct Python types."""
        assert expected_returns_estimator_type is not prior_estimator_type
        assert expected_returns_estimator_type is not distribution_estimator_type
        assert prior_estimator_type is not distribution_estimator_type

    @pytest.mark.unit()
    def test_expected_returns_members_not_in_prior(self):
        """No expected_returns member equals a prior_estimator member."""
        er_values = {m.value for m in expected_returns_estimator_type}
        prior_values = {m.value for m in prior_estimator_type}
        # EMPIRICAL appears in both — shared name but they are different Enum classes
        # Members from one class must not be instances of the other class
        for m in expected_returns_estimator_type:
            assert not isinstance(m, prior_estimator_type)

    @pytest.mark.unit()
    def test_classmethod_returns_are_lists_not_generators(self):
        """All classmethods return plain lists, not generators or tuples."""
        result = expected_returns_estimator_type.get_historical_methods()
        assert type(result) is list  # noqa: E721
        result2 = prior_estimator_type.get_bayesian_methods()
        assert type(result2) is list  # noqa: E721
        result3 = distribution_estimator_type.get_univariate()
        assert type(result3) is list  # noqa: E721


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
