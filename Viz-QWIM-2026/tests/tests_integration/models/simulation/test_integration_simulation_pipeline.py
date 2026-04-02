"""Integration tests for the Monte Carlo simulation pipeline.

Verifies that Simulation_Standard initialisation, execution, and results
assembly work correctly end-to-end for all supported distribution types,
including output structure, value positivity, reproducibility, and
distribution-specific behavioural differences.

Test Categories
---------------
- Simulation initialisation with valid parameters
- run() output DataFrame structure (shape, column names, data types)
- Simulation values are strictly positive across all scenarios
- Reproducibility: same random seed produces identical results
- Distribution type differences: NORMAL vs LOGNORMAL vs STUDENT_T
- Default parameter usage (num_scenarios, num_days, initial_value)

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
    from src.models.simulation.model_simulation_standard import (
        DEFAULT_INITIAL_VALUE,
        DEFAULT_NUM_DAYS,
        DEFAULT_NUM_SCENARIOS,
        DEFAULT_RANDOM_SEED,
        Distribution_Type,
        Simulation_Standard,
    )
except ImportError as exc:
    MODULE_IMPORT_AVAILABLE = False
    _logger.warning("Import failed — tests will be skipped: %s", exc)

pytestmark = pytest.mark.skipif(
    not MODULE_IMPORT_AVAILABLE,
    reason="Simulation modules not importable in this environment",
)

# ---------------------------------------------------------------------------
# Shared test parameters for a 3-component portfolio
# ---------------------------------------------------------------------------
_COMPONENTS: list[str] = ["VTI", "AGG", "VNQ"]
_WEIGHTS: np.ndarray = np.array([0.5, 0.3, 0.2])
_MEAN_RETURNS: np.ndarray = np.array([0.10, 0.04, 0.06])  # annualised

# Positive semi-definite covariance matrix built from a valid correlation matrix
# correlations: VTI-AGG=0.5, VTI-VNQ=0.3, AGG-VNQ=0.2
# std devs: VTI=0.20, AGG=0.10, VNQ=0.15
_COVARIANCE_MATRIX: np.ndarray = np.array(
    [
        [0.0400, 0.0100, 0.0090],
        [0.0100, 0.0100, 0.0030],
        [0.0090, 0.0030, 0.0225],
    ]
)
_NUM_SCENARIOS_SMALL: int = 10
_NUM_DAYS_SHORT: int = 20
_START_DATE: dt.date = dt.date(2024, 1, 2)


# ==============================================================================
# Fixtures
# ==============================================================================


@pytest.fixture()
def simulation_normal() -> "Simulation_Standard":
    """Provide a Simulation_Standard configured with the NORMAL distribution.

    Returns
    -------
    Simulation_Standard
        Ready-to-run simulation instance.
    """
    return Simulation_Standard(
        names_components=_COMPONENTS,
        weights=_WEIGHTS,
        distribution_type=Distribution_Type.NORMAL,
        mean_returns=_MEAN_RETURNS,
        covariance_matrix=_COVARIANCE_MATRIX,
        initial_value=100.0,
        num_scenarios=_NUM_SCENARIOS_SMALL,
        num_days=_NUM_DAYS_SHORT,
        start_date=_START_DATE,
        random_seed=DEFAULT_RANDOM_SEED,
    )


@pytest.fixture()
def results_normal(simulation_normal: "Simulation_Standard") -> pl.DataFrame:
    """Provide the run() output for a NORMAL distribution simulation.

    Parameters
    ----------
    simulation_normal : Simulation_Standard
        Configured simulation instance.

    Returns
    -------
    pl.DataFrame
        Simulation results DataFrame.
    """
    return simulation_normal.run()


# ==============================================================================
# Initialisation integration
# ==============================================================================


@pytest.mark.integration()
class Test_Simulation_Initialisation_Integration:
    """Integration tests verifying Simulation_Standard construction and defaults."""

    def test_simulation_creates_successfully_with_all_params(self) -> None:
        """Simulation_Standard constructs without error when all parameters are provided."""
        _logger.debug("Testing Simulation_Standard construction with all parameters")

        sim = Simulation_Standard(
            names_components=_COMPONENTS,
            weights=_WEIGHTS,
            distribution_type=Distribution_Type.NORMAL,
            mean_returns=_MEAN_RETURNS,
            covariance_matrix=_COVARIANCE_MATRIX,
            initial_value=100.0,
            num_scenarios=_NUM_SCENARIOS_SMALL,
            num_days=_NUM_DAYS_SHORT,
            start_date=_START_DATE,
            random_seed=42,
        )

        assert sim is not None, "Simulation_Standard must construct without error"

    def test_default_constants_are_correct(self) -> None:
        """Module-level default constants have the expected documented values."""
        _logger.debug("Testing default simulation constants")

        assert DEFAULT_NUM_SCENARIOS == 1000, f"DEFAULT_NUM_SCENARIOS must be 1000, got {DEFAULT_NUM_SCENARIOS}"
        assert DEFAULT_NUM_DAYS == 252, f"DEFAULT_NUM_DAYS must be 252, got {DEFAULT_NUM_DAYS}"
        assert DEFAULT_INITIAL_VALUE == 100.0, f"DEFAULT_INITIAL_VALUE must be 100.0, got {DEFAULT_INITIAL_VALUE}"
        assert DEFAULT_RANDOM_SEED == 42, f"DEFAULT_RANDOM_SEED must be 42, got {DEFAULT_RANDOM_SEED}"

    def test_all_distribution_types_are_available(self) -> None:
        """Distribution_Type enum exposes NORMAL, LOGNORMAL, and STUDENT_T members."""
        _logger.debug("Testing Distribution_Type enum members")

        assert hasattr(Distribution_Type, "NORMAL"), "Distribution_Type must have NORMAL"
        assert hasattr(Distribution_Type, "LOGNORMAL"), "Distribution_Type must have LOGNORMAL"
        assert hasattr(Distribution_Type, "STUDENT_T"), "Distribution_Type must have STUDENT_T"


# ==============================================================================
# run() output structure
# ==============================================================================


@pytest.mark.integration()
class Test_Simulation_Run_Output_Structure_Integration:
    """Integration tests for DataFrame shape and naming after run()."""

    def test_run_returns_polars_dataframe(
        self, results_normal: pl.DataFrame
    ) -> None:
        """run() returns a Polars DataFrame."""
        _logger.debug("Testing run() return type is Polars DataFrame")

        assert isinstance(results_normal, pl.DataFrame), "run() must return a Polars DataFrame"

    def test_run_output_has_correct_number_of_rows(
        self, results_normal: pl.DataFrame
    ) -> None:
        """run() DataFrame row count equals the requested num_days."""
        _logger.debug("Testing run() row count equals num_days")

        assert results_normal.height == _NUM_DAYS_SHORT, (
            f"Row count must equal num_days={_NUM_DAYS_SHORT}, got {results_normal.height}"
        )

    def test_run_output_has_correct_number_of_columns(
        self, results_normal: pl.DataFrame
    ) -> None:
        """run() DataFrame column count is num_scenarios + 1 (Date column)."""
        _logger.debug("Testing run() column count equals num_scenarios + 1")

        expected_cols = _NUM_SCENARIOS_SMALL + 1  # Date + N scenario columns
        assert results_normal.width == expected_cols, (
            f"Column count must be {expected_cols} (Date + {_NUM_SCENARIOS_SMALL} scenarios), "
            f"got {results_normal.width}"
        )

    def test_run_output_has_date_column(
        self, results_normal: pl.DataFrame
    ) -> None:
        """run() DataFrame contains a Date column."""
        _logger.debug("Testing presence of Date column in run() output")

        assert "Date" in results_normal.columns, "run() output must contain a 'Date' column"

    def test_run_output_scenario_columns_have_correct_names(
        self, results_normal: pl.DataFrame
    ) -> None:
        """Scenario columns are named Scenario_1 through Scenario_N."""
        _logger.debug("Testing scenario column naming pattern")

        scenario_cols = [c for c in results_normal.columns if c != "Date"]
        assert len(scenario_cols) == _NUM_SCENARIOS_SMALL, (
            f"Must have {_NUM_SCENARIOS_SMALL} scenario columns, got {len(scenario_cols)}"
        )

        for i_scenario in range(1, _NUM_SCENARIOS_SMALL + 1):
            col_name = f"Scenario_{i_scenario}"
            assert col_name in results_normal.columns, (
                f"Expected column '{col_name}' not found in results"
            )

    def test_run_output_has_no_null_values(
        self, results_normal: pl.DataFrame
    ) -> None:
        """run() DataFrame contains no null values in any column."""
        _logger.debug("Testing run() output for absence of null values")

        for col in results_normal.columns:
            null_count = results_normal[col].null_count()
            assert null_count == 0, (
                f"Column '{col}' must have no nulls, found {null_count}"
            )


# ==============================================================================
# Simulation value correctness
# ==============================================================================


@pytest.mark.integration()
class Test_Simulation_Value_Correctness_Integration:
    """Integration tests verifying financial properties of simulation output."""

    def test_all_scenario_values_are_positive(
        self, results_normal: pl.DataFrame
    ) -> None:
        """All simulated portfolio path values are strictly positive."""
        _logger.debug("Testing all scenario values are positive")

        scenario_cols = [c for c in results_normal.columns if c != "Date"]
        for col in scenario_cols:
            min_val = results_normal[col].min()
            assert min_val > 0, (
                f"Column '{col}' contains non-positive value {min_val}; all values must be > 0"
            )

    def test_simulation_dates_span_correct_number_of_business_days(
        self, results_normal: pl.DataFrame
    ) -> None:
        """Date column contains num_days distinct business-day dates starting from start_date."""
        _logger.debug("Testing simulation date span and business-day count")

        dates = results_normal["Date"].to_list()
        assert len(dates) == _NUM_DAYS_SHORT, (
            f"Must generate {_NUM_DAYS_SHORT} dates, got {len(dates)}"
        )
        assert len(set(dates)) == _NUM_DAYS_SHORT, "All generated dates must be unique"
        assert dates == sorted(dates), "Simulation dates must be in ascending order"


# ==============================================================================
# Reproducibility
# ==============================================================================


@pytest.mark.integration()
class Test_Simulation_Reproducibility_Integration:
    """Integration tests verifying deterministic output with fixed random seeds."""

    def test_same_seed_produces_identical_results(self) -> None:
        """Two run() calls with the same random seed return identical DataFrames."""
        _logger.debug("Testing run() reproducibility with fixed random seed")

        def _build_sim() -> "Simulation_Standard":
            return Simulation_Standard(
                names_components=_COMPONENTS,
                weights=_WEIGHTS,
                distribution_type=Distribution_Type.NORMAL,
                mean_returns=_MEAN_RETURNS,
                covariance_matrix=_COVARIANCE_MATRIX,
                initial_value=100.0,
                num_scenarios=5,
                num_days=10,
                start_date=_START_DATE,
                random_seed=99,
            )

        results_a = _build_sim().run()
        results_b = _build_sim().run()

        scenario_cols = [c for c in results_a.columns if c != "Date"]
        for col in scenario_cols:
            values_a = results_a[col].to_list()
            values_b = results_b[col].to_list()
            assert values_a == pytest.approx(values_b, rel=1e-9), (
                f"Column '{col}' must be identical across runs with same seed"
            )

    def test_different_seeds_produce_different_results(self) -> None:
        """Two run() calls with different random seeds produce different Scenario_1 paths."""
        _logger.debug("Testing run() produces distinct results with different seeds")

        def _build_sim(seed: int) -> "Simulation_Standard":
            return Simulation_Standard(
                names_components=_COMPONENTS,
                weights=_WEIGHTS,
                distribution_type=Distribution_Type.NORMAL,
                mean_returns=_MEAN_RETURNS,
                covariance_matrix=_COVARIANCE_MATRIX,
                initial_value=100.0,
                num_scenarios=5,
                num_days=15,
                start_date=_START_DATE,
                random_seed=seed,
            )

        results_seed_1 = _build_sim(seed=1).run()
        results_seed_2 = _build_sim(seed=2).run()

        values_seed_1 = results_seed_1["Scenario_1"].to_list()
        values_seed_2 = results_seed_2["Scenario_1"].to_list()

        assert values_seed_1 != values_seed_2, (
            "Different seeds must produce different Scenario_1 paths"
        )


# ==============================================================================
# Distribution type differences
# ==============================================================================


@pytest.mark.integration()
class Test_Simulation_Distribution_Types_Integration:
    """Integration tests verifying all three distribution types run successfully."""

    @pytest.mark.parametrize(
        "distribution_type",
        [
            Distribution_Type.NORMAL,
            Distribution_Type.LOGNORMAL,
            Distribution_Type.STUDENT_T,
        ],
    )
    def test_each_distribution_type_runs_successfully(
        self, distribution_type: "Distribution_Type"
    ) -> None:
        """Each distribution type produces a valid non-empty results DataFrame."""
        _logger.debug("Testing distribution type: %s", distribution_type)

        kwargs: dict[str, Any] = {
            "names_components": _COMPONENTS,
            "weights": _WEIGHTS,
            "distribution_type": distribution_type,
            "mean_returns": _MEAN_RETURNS,
            "covariance_matrix": _COVARIANCE_MATRIX,
            "initial_value": 100.0,
            "num_scenarios": 5,
            "num_days": 10,
            "start_date": _START_DATE,
            "random_seed": DEFAULT_RANDOM_SEED,
        }
        if distribution_type == Distribution_Type.STUDENT_T:
            kwargs["degrees_of_freedom"] = 5

        sim = Simulation_Standard(**kwargs)
        results = sim.run()

        assert isinstance(results, pl.DataFrame), (
            f"{distribution_type} run() must return Polars DataFrame"
        )
        assert results.height == 10, (
            f"{distribution_type} run() must return 10 rows"
        )
        assert "Date" in results.columns, (
            f"{distribution_type} run() result must have Date column"
        )
        scenario_cols = [c for c in results.columns if c != "Date"]
        assert all(results[col].min() > 0 for col in scenario_cols), (
            f"{distribution_type} all scenario values must be positive"
        )

    def test_normal_and_lognormal_produce_different_paths_same_seed(self) -> None:
        """NORMAL and LOGNORMAL distributions produce different Scenario_1 values with same seed."""
        _logger.debug("Testing NORMAL vs LOGNORMAL path differences")

        common_kwargs: dict[str, Any] = {
            "names_components": _COMPONENTS,
            "weights": _WEIGHTS,
            "mean_returns": _MEAN_RETURNS,
            "covariance_matrix": _COVARIANCE_MATRIX,
            "initial_value": 100.0,
            "num_scenarios": 5,
            "num_days": 15,
            "start_date": _START_DATE,
            "random_seed": DEFAULT_RANDOM_SEED,
        }

        results_normal = Simulation_Standard(
            **{**common_kwargs, "distribution_type": Distribution_Type.NORMAL}
        ).run()
        results_lognormal = Simulation_Standard(
            **{**common_kwargs, "distribution_type": Distribution_Type.LOGNORMAL}
        ).run()

        vals_normal = results_normal["Scenario_1"].to_list()
        vals_lognormal = results_lognormal["Scenario_1"].to_list()

        assert vals_normal != vals_lognormal, (
            "NORMAL and LOGNORMAL must produce different Scenario_1 paths"
        )
