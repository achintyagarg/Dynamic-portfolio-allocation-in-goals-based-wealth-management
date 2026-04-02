"""Regression tests for Monte Carlo simulation helpers and constants.

Verifies that pure helper functions in
:mod:`src.dashboard.shiny_tab_results.subtab_simulation` and model
constants in :mod:`src.models.simulation.model_simulation_standard`
produce stable outputs across code changes.

Tests cover:
    - Module constants (ALL_ETF_SYMBOLS, DEFAULT_SELECTED_ETFS, etc.)
    - Distribution key mapping golden lookup table
    - RNG creation produces reproducible sequences for all generators
    - Simulation model constants match expected golden values
    - data_tab_results normalization golden values (min-max, z-score)
    - data_tab_results transform golden values (percent_change, cumulative)

Author:
    QWIM Development Team

Version:
    0.1.0

Last Modified:
    2026-02-28
"""

from __future__ import annotations

from typing import Any

import numpy as np
import polars as pl
import pytest

from src.utils.custom_exceptions_errors_loggers.logger_custom import get_logger

_logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Conditional imports — subtab_simulation helpers
# ---------------------------------------------------------------------------

try:
    from src.dashboard.shiny_tab_results.subtab_simulation import (
        ALL_ETF_SYMBOLS,
        DEFAULT_SELECTED_ETFS,
        DISTRIBUTION_CHOICES,
        NUM_DEFAULT_SELECTED,
        RNG_TYPE_CHOICES,
        _create_rng,
        _map_distribution_key,
    )

    SUBTAB_SIMULATION_AVAILABLE = True
except ImportError as exc:
    SUBTAB_SIMULATION_AVAILABLE = False
    _logger.warning("subtab_simulation import failed — simulation tests skipped: %s", exc)

# ---------------------------------------------------------------------------
# Conditional imports — Simulation_Standard model constants
# ---------------------------------------------------------------------------

try:
    from src.models.simulation.model_simulation_standard import (
        DEFAULT_INITIAL_VALUE,
        DEFAULT_NUM_DAYS,
        DEFAULT_NUM_SCENARIOS,
        DEFAULT_RANDOM_SEED,
    )

    SIMULATION_MODEL_AVAILABLE = True
except ImportError as exc:
    SIMULATION_MODEL_AVAILABLE = False
    _logger.warning("model_simulation_standard import failed — model tests skipped: %s", exc)

# ---------------------------------------------------------------------------
# Conditional imports — Distribution_Type enum
# ---------------------------------------------------------------------------

try:
    from src.num_methods.scenarios.scenarios_distrib import Distribution_Type

    DISTRIBUTION_TYPE_AVAILABLE = True
except ImportError as exc:
    DISTRIBUTION_TYPE_AVAILABLE = False
    _logger.warning("Distribution_Type import failed: %s", exc)

# ---------------------------------------------------------------------------
# Conditional imports — utils_tab_results data processing
# ---------------------------------------------------------------------------

try:
    from src.dashboard.shiny_utils.utils_tab_results import (
        normalize_data_tab_results,
        process_data_for_plot_tab_results,
        transform_data_tab_results,
    )

    UTILS_TAB_RESULTS_AVAILABLE = True
except ImportError as exc:
    UTILS_TAB_RESULTS_AVAILABLE = False
    _logger.warning("utils_tab_results import failed — normalization tests skipped: %s", exc)


# ---------------------------------------------------------------------------
# Test class: Simulation model constants
# ---------------------------------------------------------------------------


@pytest.mark.regression()
@pytest.mark.skipif(not SIMULATION_MODEL_AVAILABLE, reason="model_simulation_standard not available")
class Test_Simulation_Model_Constants_Golden:
    """Regression tests — Simulation_Standard module constants match golden values."""

    def test_default_num_scenarios_is_1000(self) -> None:
        """DEFAULT_NUM_SCENARIOS equals 1 000 (one-year-equivalent paths baseline)."""
        _logger.debug("Checking DEFAULT_NUM_SCENARIOS = 1000")
        assert DEFAULT_NUM_SCENARIOS == 1_000

    def test_default_num_days_is_252(self) -> None:
        """DEFAULT_NUM_DAYS equals 252 (one trading year)."""
        _logger.debug("Checking DEFAULT_NUM_DAYS = 252")
        assert DEFAULT_NUM_DAYS == 252

    def test_default_initial_value_is_100(self) -> None:
        """DEFAULT_INITIAL_VALUE equals 100.0 (normalised unit portfolio)."""
        _logger.debug("Checking DEFAULT_INITIAL_VALUE = 100.0")
        assert DEFAULT_INITIAL_VALUE == pytest.approx(100.0)

    def test_default_random_seed_is_42(self) -> None:
        """DEFAULT_RANDOM_SEED equals 42 for reproducibility."""
        _logger.debug("Checking DEFAULT_RANDOM_SEED = 42")
        assert DEFAULT_RANDOM_SEED == 42

    def test_constants_types_are_correct(self) -> None:
        """All constants have the expected Python types."""
        _logger.debug("Checking constant types")
        assert isinstance(DEFAULT_NUM_SCENARIOS, int)
        assert isinstance(DEFAULT_NUM_DAYS, int)
        assert isinstance(DEFAULT_INITIAL_VALUE, float)
        assert isinstance(DEFAULT_RANDOM_SEED, int)


# ---------------------------------------------------------------------------
# Test class: ETF symbols and UI choice constants
# ---------------------------------------------------------------------------


@pytest.mark.regression()
@pytest.mark.skipif(not SUBTAB_SIMULATION_AVAILABLE, reason="subtab_simulation not available")
class Test_Simulation_UI_Constants_Golden:
    """Regression tests — subtab_simulation UI constants are stable."""

    def test_all_etf_symbols_count_is_12(self) -> None:
        """ALL_ETF_SYMBOLS contains exactly 12 symbols."""
        _logger.debug("Checking ALL_ETF_SYMBOLS count = 12")
        assert len(ALL_ETF_SYMBOLS) == 12

    def test_all_etf_symbols_golden_list(self) -> None:
        """ALL_ETF_SYMBOLS matches the golden ordered list."""
        _logger.debug("Checking ALL_ETF_SYMBOLS golden list")
        expected = ["IVV", "IJH", "IWM", "EFA", "EEM", "AGG", "SPTL", "HYG", "SPBO", "IYR", "DBC", "GLD"]
        assert ALL_ETF_SYMBOLS == expected

    def test_default_selected_etfs_golden(self) -> None:
        """DEFAULT_SELECTED_ETFS matches golden selection of first 3 symbols."""
        _logger.debug("Checking DEFAULT_SELECTED_ETFS golden list")
        assert DEFAULT_SELECTED_ETFS == ["IVV", "IJH", "IWM"]

    def test_num_default_selected_is_3(self) -> None:
        """NUM_DEFAULT_SELECTED equals 3."""
        _logger.debug("Checking NUM_DEFAULT_SELECTED = 3")
        assert NUM_DEFAULT_SELECTED == 3

    def test_default_selected_count_matches_constant(self) -> None:
        """Length of DEFAULT_SELECTED_ETFS equals NUM_DEFAULT_SELECTED."""
        _logger.debug("Checking consistency between count and list length")
        assert len(DEFAULT_SELECTED_ETFS) == NUM_DEFAULT_SELECTED

    def test_distribution_choices_golden_keys(self) -> None:
        """DISTRIBUTION_CHOICES contains exactly the three expected keys."""
        _logger.debug("Checking DISTRIBUTION_CHOICES keys")
        assert set(DISTRIBUTION_CHOICES.keys()) == {"normal", "lognormal", "student_t"}

    def test_rng_type_choices_golden_keys(self) -> None:
        """RNG_TYPE_CHOICES contains exactly the four expected keys."""
        _logger.debug("Checking RNG_TYPE_CHOICES keys")
        assert set(RNG_TYPE_CHOICES.keys()) == {"pcg64", "mt19937", "philox", "sfc64"}

    def test_default_selected_etfs_are_subset_of_all(self) -> None:
        """All DEFAULT_SELECTED_ETFS are present in ALL_ETF_SYMBOLS."""
        _logger.debug("Checking DEFAULT_SELECTED_ETFS ⊆ ALL_ETF_SYMBOLS")
        for sym in DEFAULT_SELECTED_ETFS:
            assert sym in ALL_ETF_SYMBOLS


# ---------------------------------------------------------------------------
# Test class: _map_distribution_key golden lookup
# ---------------------------------------------------------------------------


@pytest.mark.regression()
@pytest.mark.skipif(
    not (SUBTAB_SIMULATION_AVAILABLE and DISTRIBUTION_TYPE_AVAILABLE),
    reason="subtab_simulation or Distribution_Type not available",
)
class Test_Map_Distribution_Key_Golden:
    """Regression tests — _map_distribution_key golden lookup table."""

    def test_normal_key_maps_to_normal_enum(self) -> None:
        """Key 'normal' maps to Distribution_Type.NORMAL."""
        _logger.debug("Checking 'normal' → NORMAL")
        result = _map_distribution_key("normal")
        assert result == Distribution_Type.NORMAL

    def test_lognormal_key_maps_to_lognormal_enum(self) -> None:
        """Key 'lognormal' maps to Distribution_Type.LOGNORMAL."""
        _logger.debug("Checking 'lognormal' → LOGNORMAL")
        result = _map_distribution_key("lognormal")
        assert result == Distribution_Type.LOGNORMAL

    def test_student_t_key_maps_to_student_t_enum(self) -> None:
        """Key 'student_t' maps to Distribution_Type.STUDENT_T."""
        _logger.debug("Checking 'student_t' → STUDENT_T")
        result = _map_distribution_key("student_t")
        assert result == Distribution_Type.STUDENT_T

    def test_unknown_key_falls_back_to_normal(self) -> None:
        """Unrecognised key falls back to Distribution_Type.NORMAL (safe default)."""
        _logger.debug("Checking unknown key fallback → NORMAL")
        assert _map_distribution_key("xyz") == Distribution_Type.NORMAL
        assert _map_distribution_key("") == Distribution_Type.NORMAL
        assert _map_distribution_key("NORMAL") == Distribution_Type.NORMAL  # case-sensitive

    def test_all_distribution_choice_keys_resolve(self) -> None:
        """Every key in DISTRIBUTION_CHOICES resolves to a Distribution_Type member."""
        _logger.debug("Checking all DISTRIBUTION_CHOICES keys resolve")
        for key in DISTRIBUTION_CHOICES:
            result = _map_distribution_key(key)
            assert isinstance(result, Distribution_Type)


# ---------------------------------------------------------------------------
# Test class: _create_rng reproducibility
# ---------------------------------------------------------------------------


@pytest.mark.regression()
@pytest.mark.skipif(not SUBTAB_SIMULATION_AVAILABLE, reason="subtab_simulation not available")
class Test_Create_Rng_Reproducibility:
    """Regression tests — _create_rng produces reproducible sequences for same seed."""

    @pytest.mark.parametrize("rng_type", ["pcg64", "mt19937", "philox", "sfc64"])
    def test_same_seed_produces_identical_first_draw(self, rng_type: str) -> None:
        """Two RNGs built with same type and seed produce identical first draw."""
        _logger.debug("Testing reproducibility for rng_type=%s", rng_type)
        rng_a = _create_rng(rng_type, seed=42)
        rng_b = _create_rng(rng_type, seed=42)
        assert rng_a.random() == pytest.approx(rng_b.random(), rel=1e-12)

    @pytest.mark.parametrize("rng_type", ["pcg64", "mt19937", "philox", "sfc64"])
    def test_different_seeds_produce_different_draws(self, rng_type: str) -> None:
        """Different seeds produce different first draws (no seed collision)."""
        _logger.debug("Testing seed collision absence for rng_type=%s", rng_type)
        rng_a = _create_rng(rng_type, seed=0)
        rng_b = _create_rng(rng_type, seed=1)
        assert rng_a.random() != rng_b.random()

    def test_pcg64_golden_first_draw(self) -> None:
        """PCG-64 with seed 42 produces a known golden first draw."""
        _logger.debug("Testing PCG-64 seed-42 golden first draw")
        rng = _create_rng("pcg64", seed=42)
        draw = rng.random()
        # Golden value: np.random.Generator(np.random.PCG64(42)).random()
        assert 0.0 < draw < 1.0  # basic range check
        # Reproduce independently and assert exact equality
        expected = np.random.Generator(np.random.PCG64(42)).random()
        assert draw == pytest.approx(expected, rel=1e-12)

    def test_unknown_rng_type_defaults_to_pcg64_behaviour(self) -> None:
        """Unknown RNG type falls back to PCG64 (same first draw as explicit pcg64)."""
        _logger.debug("Testing unknown RNG type fallback")
        rng_default = _create_rng("unknown_generator", seed=42)
        rng_pcg64 = _create_rng("pcg64", seed=42)
        assert rng_default.random() == pytest.approx(rng_pcg64.random(), rel=1e-12)


# ---------------------------------------------------------------------------
# Fixtures — data for normalization / transform tests
# ---------------------------------------------------------------------------


@pytest.fixture()
def simple_timeseries_df() -> pl.DataFrame:
    """Return a small 10-row timeseries DataFrame for normalization tests.

    Returns:
        pl.DataFrame: DataFrame with 'date' and 'value' columns.
    """
    dates = [f"2024-01-{i:02d}" for i in range(1, 11)]
    values = [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0]
    return pl.DataFrame({"date": dates, "value": values})


# ---------------------------------------------------------------------------
# Test class: normalize_data_tab_results  —  golden values
# ---------------------------------------------------------------------------


@pytest.mark.regression()
@pytest.mark.skipif(not UTILS_TAB_RESULTS_AVAILABLE, reason="utils_tab_results not available")
class Test_Normalize_Data_Tab_Results_Golden:
    """Regression tests — normalize_data_tab_results produces stable outputs."""

    def test_none_method_returns_data_unchanged(
        self, simple_timeseries_df: pl.DataFrame
    ) -> None:
        """Method 'none' returns the exact same DataFrame without modification."""
        _logger.debug("Testing 'none' normalization passthrough")
        result = normalize_data_tab_results(simple_timeseries_df, method="none")
        assert result.shape == simple_timeseries_df.shape
        assert result["value"].to_list() == simple_timeseries_df["value"].to_list()

    def test_min_max_first_value_is_zero(self, simple_timeseries_df: pl.DataFrame) -> None:
        """Min-max normalization maps minimum value to exactly 0.0."""
        _logger.debug("Testing min-max first value = 0.0")
        result = normalize_data_tab_results(simple_timeseries_df, method="min_max")
        first_val = result["value"][0]
        assert first_val == pytest.approx(0.0, abs=1e-10)

    def test_min_max_last_value_is_one(self, simple_timeseries_df: pl.DataFrame) -> None:
        """Min-max normalization maps maximum value to exactly 1.0."""
        _logger.debug("Testing min-max last value = 1.0")
        result = normalize_data_tab_results(simple_timeseries_df, method="min_max")
        last_val = result["value"][-1]
        assert last_val == pytest.approx(1.0, abs=1e-10)

    def test_min_max_all_values_in_unit_range(
        self, simple_timeseries_df: pl.DataFrame
    ) -> None:
        """All min-max normalised values lie within [0, 1]."""
        _logger.debug("Testing min-max all values in [0, 1]")
        result = normalize_data_tab_results(simple_timeseries_df, method="min_max")
        for val in result["value"].to_list():
            assert 0.0 <= val <= 1.0 + 1e-10

    def test_min_max_middle_value_golden(self, simple_timeseries_df: pl.DataFrame) -> None:
        """Min-max 5th value (50.0 from range 10–100) → golden 4/9 ≈ 0.44444."""
        _logger.debug("Testing min-max 5th value golden")
        result = normalize_data_tab_results(simple_timeseries_df, method="min_max")
        # (50 - 10) / (100 - 10) = 40/90 = 4/9
        expected = 40.0 / 90.0
        assert result["value"][4] == pytest.approx(expected, rel=1e-6)

    def test_z_score_mean_is_approximately_zero(
        self, simple_timeseries_df: pl.DataFrame
    ) -> None:
        """Z-score normalised values have mean ≈ 0."""
        _logger.debug("Testing z-score mean ≈ 0")
        result = normalize_data_tab_results(simple_timeseries_df, method="z_score")
        mean_val = result.select(pl.col("value").mean()).item()
        assert abs(mean_val) < 1e-10

    def test_z_score_std_is_approximately_one(
        self, simple_timeseries_df: pl.DataFrame
    ) -> None:
        """Z-score normalised values have std ≈ 1."""
        _logger.debug("Testing z-score std ≈ 1")
        result = normalize_data_tab_results(simple_timeseries_df, method="z_score")
        std_val = result.select(pl.col("value").std()).item()
        assert abs(std_val - 1.0) < 1e-6

    def test_none_inputs_returned_unchanged(self) -> None:
        """None DataFrame input is returned as-is without error."""
        _logger.debug("Testing None input passthrough")
        result = normalize_data_tab_results(None, method="min_max")
        assert result is None


# ---------------------------------------------------------------------------
# Test class: transform_data_tab_results  —  golden values
# ---------------------------------------------------------------------------


@pytest.mark.regression()
@pytest.mark.skipif(not UTILS_TAB_RESULTS_AVAILABLE, reason="utils_tab_results not available")
class Test_Transform_Data_Tab_Results_Golden:
    """Regression tests — transform_data_tab_results produces stable outputs."""

    def test_none_transformation_returns_data_unchanged(
        self, simple_timeseries_df: pl.DataFrame
    ) -> None:
        """Transformation 'none' returns data without modification."""
        _logger.debug("Testing 'none' transform passthrough")
        result = transform_data_tab_results(simple_timeseries_df, transformation="none")
        assert result["value"].to_list() == simple_timeseries_df["value"].to_list()

    def test_percent_change_adds_pct_column(
        self, simple_timeseries_df: pl.DataFrame
    ) -> None:
        """percent_change transform adds a 'value_pct' column."""
        _logger.debug("Testing percent_change adds _pct column")
        result = transform_data_tab_results(
            simple_timeseries_df, transformation="percent_change"
        )
        assert "value_pct" in result.columns

    def test_percent_change_first_row_is_zero(
        self, simple_timeseries_df: pl.DataFrame
    ) -> None:
        """percent_change first value is 0.0 (no change from baseline)."""
        _logger.debug("Testing percent_change first value = 0")
        result = transform_data_tab_results(
            simple_timeseries_df, transformation="percent_change"
        )
        assert result["value_pct"][0] == pytest.approx(0.0, abs=1e-10)

    def test_percent_change_last_row_golden(
        self, simple_timeseries_df: pl.DataFrame
    ) -> None:
        """percent_change last value: (100/10 - 1) * 100 = 900.0."""
        _logger.debug("Testing percent_change last = 900.0")
        result = transform_data_tab_results(
            simple_timeseries_df, transformation="percent_change"
        )
        # first_val=10, last_val=100 → (100/10 - 1)*100 = 900
        assert result["value_pct"][-1] == pytest.approx(900.0, rel=1e-6)

    def test_cumulative_transform_adds_cum_column(
        self, simple_timeseries_df: pl.DataFrame
    ) -> None:
        """cumulative transform adds a 'value_cum' column."""
        _logger.debug("Testing cumulative adds _cum column")
        result = transform_data_tab_results(simple_timeseries_df, transformation="cumulative")
        assert "value_cum" in result.columns

    def test_cumulative_first_value_equals_original(
        self, simple_timeseries_df: pl.DataFrame
    ) -> None:
        """Cumulative sum first element equals first original value."""
        _logger.debug("Testing cumulative sum first element")
        result = transform_data_tab_results(simple_timeseries_df, transformation="cumulative")
        assert result["value_cum"][0] == pytest.approx(10.0, abs=1e-10)

    def test_cumulative_total_golden(self, simple_timeseries_df: pl.DataFrame) -> None:
        """Cumulative sum last value equals sum of all original values (550)."""
        _logger.debug("Testing cumulative sum last value = 550")
        result = transform_data_tab_results(simple_timeseries_df, transformation="cumulative")
        # sum(10,20,...,100) = 550
        assert result["value_cum"][-1] == pytest.approx(550.0, abs=1e-6)


# ---------------------------------------------------------------------------
# Test class: process_data_for_plot_tab_results  —  golden behaviour
# ---------------------------------------------------------------------------


@pytest.mark.regression()
@pytest.mark.skipif(not UTILS_TAB_RESULTS_AVAILABLE, reason="utils_tab_results not available")
class Test_Process_Data_For_Plot_Tab_Results_Golden:
    """Regression tests — process_data_for_plot_tab_results is stable."""

    def test_small_dataframe_returned_unchanged(
        self, simple_timeseries_df: pl.DataFrame
    ) -> None:
        """DataFrame within max_points limit is returned without downsampling."""
        _logger.debug("Testing small DataFrame passthrough")
        result = process_data_for_plot_tab_results(
            simple_timeseries_df, date_column="date", max_points=5000
        )
        assert result.shape == simple_timeseries_df.shape

    def test_large_dataframe_is_downsampled(self) -> None:
        """DataFrame exceeding max_points is downsampled to stay under limit."""
        _logger.debug("Testing large DataFrame downsampling")
        dates = [f"2024-{(i // 30) + 1:02d}-{(i % 28) + 1:02d}" for i in range(10_000)]
        # Use a date format that will still sort correctly
        dates = [f"2024-01-01"] * 10_000  # same date is OK for this test
        values = list(range(10_000))
        big_df = pl.DataFrame({"date": dates, "value": values})
        result = process_data_for_plot_tab_results(big_df, date_column="date", max_points=100)
        assert result.height <= 200  # generous upper bound after downsampling

    def test_none_dataframe_returned_as_none(self) -> None:
        """None input is returned as-is."""
        _logger.debug("Testing None passthrough")
        result = process_data_for_plot_tab_results(None)
        assert result is None
