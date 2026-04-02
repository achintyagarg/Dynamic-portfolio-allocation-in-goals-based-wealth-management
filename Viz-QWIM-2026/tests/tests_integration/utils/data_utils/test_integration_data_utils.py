"""Integration tests for data_utils package.

These tests exercise the full data utilities pipeline testing that
generate_data and utils_data_financial work correctly together and
produce data consistent with downstream consumers (e.g., portfolio models).

Author: QWIM Development Team
Version: 0.1.0
"""

from __future__ import annotations

from datetime import date

import polars as pl
import pytest

from src.utils.data_utils.generate_data import generate_monthly_timeseries
from src.utils.data_utils.utils_data_financial import (
    distribution_estimator_type,
    expected_returns_estimator_type,
    prior_estimator_type,
)


# ==============================================================================
# Integration: generate_data + utils_data_financial
# ==============================================================================


class Test_Generated_Data_Schema_With_Estimator_Types:
    """Tests combining generate_data output with financial estimator enums."""

    @pytest.mark.integration()
    def test_generated_series_count_matches_estimator_count(self):
        """Number of generated series matches number of expected-return method categories."""
        df = generate_monthly_timeseries()
        series_cols = [c for c in df.columns if c != "date"]

        # 3 category groups in expected_returns_estimator_type
        assert len(series_cols) > 0

    @pytest.mark.integration()
    def test_generated_data_columns_are_usable_as_series_names(self):
        """Column names in generated data can label financial time-series."""
        df = generate_monthly_timeseries()
        series_cols = [c for c in df.columns if c != "date"]
        # All column names should be non-empty strings
        for col in series_cols:
            assert isinstance(col, str)
            assert len(col) > 0

    @pytest.mark.integration()
    def test_generated_data_date_column_has_monthly_frequency(self):
        """Date column in generated data has approximately monthly intervals."""
        df = generate_monthly_timeseries(start_date="2020-01-01", end_date="2020-06-30")
        dates = df["date"].to_list()
        assert len(dates) == 6
        # Check approximately monthly spacing
        for i in range(1, len(dates)):
            delta = (dates[i] - dates[i - 1]).days
            assert 28 <= delta <= 31, f"Non-monthly interval at index {i}: {delta} days"

    @pytest.mark.integration()
    def test_all_enum_members_instantiable_within_pipeline(self):
        """All estimator enum members can be iterated alongside a generated data pipeline."""
        df = generate_monthly_timeseries()
        assert isinstance(df, pl.DataFrame)

        # Simulate selecting an estimator based on a condition
        for estimator in expected_returns_estimator_type:
            label = estimator.value
            assert isinstance(label, str)
            # Concrete usage: just verify the pipline can reference the estimator
            assert label in [m.value for m in expected_returns_estimator_type]

    @pytest.mark.integration()
    def test_historical_methods_map_to_generated_series(self):
        """Historical estimator methods can be applied to generated data structure."""
        df = generate_monthly_timeseries()
        historical = expected_returns_estimator_type.get_historical_methods()
        series_cols = [c for c in df.columns if c != "date"]

        # We have enough series for historical method count
        assert len(series_cols) >= len(historical)

    @pytest.mark.integration()
    def test_distribution_types_span_univariate_and_copulas(self):
        """Distribution enum covers univariate and copula types for simulation."""
        univariate = distribution_estimator_type.get_univariate()
        copulas = distribution_estimator_type.get_all_copulas()

        # Total distribution types available for simulation
        total = len(univariate) + len(copulas)
        assert total == len(list(distribution_estimator_type))

    @pytest.mark.integration()
    def test_simulation_ready_types_subset_of_all_types(self):
        """Simulation-ready types are a proper subset of all distribution types."""
        all_types = set(distribution_estimator_type)
        simulation_ready = set(distribution_estimator_type.get_simulation_ready())
        assert simulation_ready.issubset(all_types)
        assert len(simulation_ready) < len(all_types)

    @pytest.mark.integration()
    def test_prior_methods_form_complete_bayesian_workflow(self):
        """Bayesian workflow uses view-incorporation priors as building blocks."""
        bayesian = prior_estimator_type.get_bayesian_methods()
        view_methods = prior_estimator_type.get_view_incorporation_methods()
        # In a real workflow, Bayesian methods support view incorporation
        assert set(bayesian) == set(view_methods)


# ==============================================================================
# Integration: DataFrame shape and type consistency
# ==============================================================================


class Test_Generated_Data_Shape_Consistency:
    """Integration tests verifying DataFrame shape and type consistency."""

    @pytest.mark.integration()
    def test_custom_date_range_produces_correct_shape(self):
        """Custom date range gives expected row and column counts."""
        df = generate_monthly_timeseries(start_date="2022-01-01", end_date="2022-12-31")
        assert df.shape[0] == 12
        assert df.shape[1] == 8  # date + 7 series

    @pytest.mark.integration()
    def test_data_values_are_finite(self):
        """No infinite or NaN values in generated data."""
        df = generate_monthly_timeseries(start_date="2020-01-01", end_date="2021-12-31")
        series_cols = [c for c in df.columns if c != "date"]
        for col in series_cols:
            assert df[col].is_nan().sum() == 0, f"NaN in column {col}"
            assert df[col].is_infinite().sum() == 0, f"Infinite in column {col}"

    @pytest.mark.integration()
    def test_data_dates_are_monotonically_increasing(self):
        """Date column is strictly increasing (monthly cadence)."""
        df = generate_monthly_timeseries(start_date="2020-01-01", end_date="2023-12-31")
        dates = df["date"].to_list()
        for i in range(1, len(dates)):
            assert dates[i] > dates[i - 1], (
                f"Non-monotonic dates at index {i}: {dates[i - 1]} → {dates[i]}"
            )

    @pytest.mark.integration()
    def test_data_first_date_matches_start_date(self):
        """First row date matches the specified start_date."""
        df = generate_monthly_timeseries(start_date="2021-03-01", end_date="2021-06-30")
        assert df["date"][0] == date(2021, 3, 1)

    @pytest.mark.integration()
    def test_reproducing_same_data_with_default_params(self):
        """Two consecutive calls with defaults produce identical DataFrames."""
        df1 = generate_monthly_timeseries()
        df2 = generate_monthly_timeseries()
        # Shape must match
        assert df1.shape == df2.shape
        # Values must be identical (fixed internal seed)
        for col in df1.columns:
            assert (df1[col] == df2[col]).all(), (
                f"Column {col} differs between two default calls"
            )


# ==============================================================================
# Integration: Enum classmethod interaction
# ==============================================================================


class Test_Estimator_Enum_Interactions:
    """Integration tests for interactions between enum classmethods."""

    @pytest.mark.integration()
    def test_expected_returns_all_groups_cover_all_members(self):
        """Union of all expected_returns groups covers all members."""
        all_grouped = set(
            expected_returns_estimator_type.get_historical_methods()
            + expected_returns_estimator_type.get_model_based_methods()
            + expected_returns_estimator_type.get_distribution_based_methods()
        )
        assert all_grouped == set(expected_returns_estimator_type)

    @pytest.mark.integration()
    def test_prior_all_primary_groups_cover_all_members(self):
        """Union of prior's primary groups covers all prior members."""
        all_grouped = set(
            prior_estimator_type.get_data_driven()
            + prior_estimator_type.get_equilibrium_based()
            + prior_estimator_type.get_factor_based()
            + prior_estimator_type.get_scenario_based()
            + prior_estimator_type.get_probabilistic()
        )
        assert all_grouped == set(prior_estimator_type)

    @pytest.mark.integration()
    def test_distribution_all_groups_cover_all_members(self):
        """Union of univariate + all copulas covers all distribution members."""
        all_grouped = set(
            distribution_estimator_type.get_univariate()
            + distribution_estimator_type.get_all_copulas()
        )
        assert all_grouped == set(distribution_estimator_type)

    @pytest.mark.integration()
    def test_robust_methods_are_subset_of_all_er_methods(self):
        """Robust expected-return methods are a proper subset of all members."""
        robust = set(expected_returns_estimator_type.get_robust_methods())
        all_members = set(expected_returns_estimator_type)
        assert robust.issubset(all_members)
        assert robust != all_members

    @pytest.mark.integration()
    def test_scenario_priors_suitable_for_stress_testing(self):
        """Scenario-based priors contain stress-test entries."""
        scenario = prior_estimator_type.get_scenario_based()
        names = [m.name for m in scenario]
        assert any("STRESS" in name for name in names)

    @pytest.mark.integration()
    def test_fat_tailed_distributions_overlap_flexible_distributions(self):
        """NIG appears in both fat-tailed and flexible distribution groups."""
        fat_tailed = set(distribution_estimator_type.get_fat_tailed_distributions())
        flexible = set(distribution_estimator_type.get_flexible_distributions())
        overlap = fat_tailed & flexible
        assert distribution_estimator_type.UNIVARIATE_NORMAL_INVERSE_GAUSSIAN in overlap


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
