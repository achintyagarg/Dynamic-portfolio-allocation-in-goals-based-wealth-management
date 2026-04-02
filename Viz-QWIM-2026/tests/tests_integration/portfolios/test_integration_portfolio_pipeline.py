"""Integration tests for the portfolio pipeline.

Verifies that portfolio_QWIM, utils_portfolio, and utils_data functions work
correctly together across the full end-to-end portfolio data flow — from raw
CSV loading through object construction, value calculation, benchmarking,
CSV persistence, and returns computation.

Test Categories
---------------
- Portfolio construction from raw weights data
- Portfolio value calculation from price data
- Benchmark portfolio value creation from portfolio values
- CSV persistence round-trip (save then reload)
- Validation and returns calculation for computed portfolio data
- Full pipeline convenience function (get_sample_portfolio)

Author:
    QWIM Development Team

Version:
    0.1.0

Last Modified:
    2026-02-28
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import polars as pl
import pytest

from src.utils.custom_exceptions_errors_loggers.logger_custom import get_logger

_logger = get_logger(__name__)

# Project root resolved relative to this file: tests/tests_integration/portfolios/
_PROJECT_ROOT: Path = Path(__file__).resolve().parents[3]
_ETF_DATA_PATH: Path = _PROJECT_ROOT / "inputs" / "raw" / "data_ETFs.csv"
_WEIGHTS_PATH: Path = _PROJECT_ROOT / "inputs" / "raw" / "sample_portfolio_weights_ETFs.csv"

MODULE_IMPORT_AVAILABLE: bool = True

try:
    from src.portfolios.portfolio_QWIM import portfolio_QWIM
    from src.portfolios.utils_portfolio import (
        calculate_portfolio_values,
        create_benchmark_portfolio_values,
        load_portfolio_weights,
        load_sample_etf_data,
        save_portfolio_values_to_csv,
    )
except ImportError as exc:
    MODULE_IMPORT_AVAILABLE = False
    _logger.warning("Import failed — tests will be skipped: %s", exc)

pytestmark = pytest.mark.skipif(
    not MODULE_IMPORT_AVAILABLE,
    reason="Portfolio modules not importable in this environment",
)


# ==============================================================================
# Fixtures
# ==============================================================================


@pytest.fixture()
def sample_weights_df() -> pl.DataFrame:
    """Provide a minimal valid weights DataFrame for portfolio construction.

    Returns
    -------
    pl.DataFrame
        DataFrame with Date and component weight columns, weights summing to 1.0.
    """
    return pl.DataFrame(
        {
            "Date": ["2023-01-01", "2023-07-01"],
            "VTI": [0.50, 0.45],
            "AGG": [0.30, 0.35],
            "VNQ": [0.20, 0.20],
        }
    )


@pytest.fixture()
def sample_price_data() -> pl.DataFrame:
    """Provide aligned ETF price data for portfolio value calculation.

    Returns
    -------
    pl.DataFrame
        DataFrame with a Date column and one price column per ETF component.
    """
    return pl.DataFrame(
        {
            "Date": [
                "2023-01-01",
                "2023-02-01",
                "2023-03-01",
                "2023-04-01",
                "2023-05-01",
                "2023-06-01",
                "2023-07-01",
                "2023-08-01",
                "2023-09-01",
                "2023-10-01",
            ],
            "VTI": [200.0, 202.0, 205.0, 203.0, 207.0, 210.0, 208.0, 212.0, 215.0, 218.0],
            "AGG": [100.0, 99.5, 101.0, 100.0, 101.5, 102.0, 101.0, 102.5, 103.0, 103.5],
            "VNQ": [80.0, 81.0, 82.0, 80.5, 83.0, 84.0, 83.0, 85.0, 86.0, 87.0],
        }
    )


@pytest.fixture()
def built_portfolio(sample_weights_df: pl.DataFrame) -> "portfolio_QWIM":
    """Provide a portfolio_QWIM instance constructed from a weights DataFrame.

    Parameters
    ----------
    sample_weights_df : pl.DataFrame
        Fixture providing test weight data.

    Returns
    -------
    portfolio_QWIM
        Constructed portfolio object.
    """
    return portfolio_QWIM(
        name_portfolio="Integration Test Portfolio",
        portfolio_weights=sample_weights_df,
    )


@pytest.fixture()
def calculated_portfolio_values(
    built_portfolio: "portfolio_QWIM",
    sample_price_data: pl.DataFrame,
) -> pl.DataFrame:
    """Provide pre-calculated portfolio values for downstream tests.

    Parameters
    ----------
    built_portfolio : portfolio_QWIM
        Constructed portfolio object.
    sample_price_data : pl.DataFrame
        ETF price data aligned to portfolio date range.

    Returns
    -------
    pl.DataFrame
        DataFrame with Date and portfolio value columns.
    """
    return calculate_portfolio_values(
        portfolio_obj=built_portfolio,
        price_data=sample_price_data,
        initial_value=100.0,
    )


# ==============================================================================
# Portfolio construction integration
# ==============================================================================


@pytest.mark.integration()
class Test_Portfolio_Construction_Integration:
    """Integration tests for portfolio_QWIM construction from raw data."""

    def test_load_weights_and_construct_portfolio(self) -> None:
        """Raw CSV weights load and portfolio_QWIM construction produce valid object."""
        _logger.debug("Testing load_portfolio_weights -> portfolio_QWIM construction")

        weights_df = load_portfolio_weights(filepath=_WEIGHTS_PATH)

        assert isinstance(weights_df, pl.DataFrame), "Expected Polars DataFrame from load_portfolio_weights"
        assert "Date" in weights_df.columns, "Weights DataFrame must contain Date column"
        assert weights_df.height > 0, "Weights DataFrame must have at least one row"

        portfolio = portfolio_QWIM(
            name_portfolio="Integration Test Portfolio",
            portfolio_weights=weights_df,
        )

        assert portfolio is not None, "portfolio_QWIM must construct without error"
        assert portfolio.get_num_components > 0, "Portfolio must have at least one component"
        assert portfolio.get_portfolio_name is not None, "Portfolio must have a name"

    def test_portfolio_from_weights_df_has_correct_components(
        self, sample_weights_df: pl.DataFrame
    ) -> None:
        """Portfolio built from a weights DataFrame correctly identifies all components."""
        _logger.debug("Testing portfolio component extraction from DataFrame construction")

        portfolio = portfolio_QWIM(
            name_portfolio="Component Test",
            portfolio_weights=sample_weights_df,
        )

        components = portfolio.get_portfolio_components
        assert "VTI" in components, "VTI must be identified as a component"
        assert "AGG" in components, "AGG must be identified as a component"
        assert "VNQ" in components, "VNQ must be identified as a component"
        assert portfolio.get_num_components == 3, "Portfolio must have exactly 3 components"

    def test_portfolio_from_component_names_creates_equal_weights(self) -> None:
        """Portfolio constructed from names_components assigns equal initial weights."""
        _logger.debug("Testing equal-weight construction from names_components")

        components = ["VTI", "AGG", "VNQ"]
        portfolio = portfolio_QWIM(
            name_portfolio="Equal Weight Portfolio",
            names_components=components,
            date_portfolio="2024-01-01",
        )

        weights_df = portfolio.get_portfolio_weights()
        assert weights_df is not None, "get_portfolio_weights must not return None"
        assert weights_df.height >= 1, "Weights DataFrame must have at least one row"

        component_cols = [c for c in weights_df.columns if c != "Date"]
        assert len(component_cols) == 3, "Must have 3 component columns"

        total_weight = sum(weights_df[col][0] for col in component_cols)
        assert abs(total_weight - 1.0) < 1e-6, f"Equal weights must sum to 1.0, got {total_weight}"

    def test_portfolio_get_portfolio_weights_returns_dataframe(
        self, built_portfolio: "portfolio_QWIM"
    ) -> None:
        """get_portfolio_weights returns a non-empty Polars DataFrame."""
        _logger.debug("Testing get_portfolio_weights return type and structure")

        weights = built_portfolio.get_portfolio_weights()

        assert isinstance(weights, pl.DataFrame), "get_portfolio_weights must return pl.DataFrame"
        assert "Date" in weights.columns, "Weights DataFrame must contain Date column"
        assert weights.height == 2, "Must contain exactly 2 weight rows"
        assert weights.width == 4, "Must have Date + 3 component columns = 4 total"


# ==============================================================================
# Portfolio value calculation integration
# ==============================================================================


@pytest.mark.integration()
class Test_Portfolio_Value_Calculation_Integration:
    """Integration tests for calculating portfolio values from prices and weights."""

    def test_calculate_portfolio_values_returns_dataframe(
        self,
        built_portfolio: "portfolio_QWIM",
        sample_price_data: pl.DataFrame,
    ) -> None:
        """calculate_portfolio_values returns a Polars DataFrame with Date and value columns."""
        _logger.debug("Testing calculate_portfolio_values return structure")

        result = calculate_portfolio_values(
            portfolio_obj=built_portfolio,
            price_data=sample_price_data,
            initial_value=100.0,
        )

        assert isinstance(result, pl.DataFrame), "Result must be a Polars DataFrame"
        assert result.height > 0, "Result must have at least one row"
        assert "Date" in result.columns, "Result must contain a Date column"
        assert result.width == 2, "Result must have exactly 2 columns: Date and value"

    def test_calculate_portfolio_values_starts_at_initial_value(
        self,
        built_portfolio: "portfolio_QWIM",
        sample_price_data: pl.DataFrame,
    ) -> None:
        """First portfolio value equals the specified initial_value."""
        _logger.debug("Testing initial value correctness in calculate_portfolio_values")

        result = calculate_portfolio_values(
            portfolio_obj=built_portfolio,
            price_data=sample_price_data,
            initial_value=100.0,
        )

        value_col = [c for c in result.columns if c != "Date"][0]
        first_value = result[value_col][0]
        assert abs(first_value - 100.0) < 1e-4, f"First value must be 100.0, got {first_value}"

    def test_calculate_portfolio_values_all_positive(
        self,
        calculated_portfolio_values: pl.DataFrame,
    ) -> None:
        """All calculated portfolio values are strictly positive."""
        _logger.debug("Testing that all portfolio values are positive")

        value_col = [c for c in calculated_portfolio_values.columns if c != "Date"][0]
        values = calculated_portfolio_values[value_col]

        assert values.min() > 0, f"All portfolio values must be positive, min was {values.min()}"

    def test_calculate_portfolio_values_no_nulls(
        self,
        calculated_portfolio_values: pl.DataFrame,
    ) -> None:
        """Calculated portfolio DataFrame contains no null values."""
        _logger.debug("Testing for absence of null values in portfolio values")

        for col in calculated_portfolio_values.columns:
            null_count = calculated_portfolio_values[col].null_count()
            assert null_count == 0, f"Column '{col}' must have no nulls, found {null_count}"

    def test_calculate_portfolio_values_dates_are_sorted(
        self,
        calculated_portfolio_values: pl.DataFrame,
    ) -> None:
        """Portfolio values DataFrame is sorted by Date in ascending order."""
        _logger.debug("Testing date sorting in portfolio values")

        dates = calculated_portfolio_values["Date"].to_list()
        assert dates == sorted(dates), "Portfolio values must be sorted by Date ascending"

    def test_full_pipeline_etf_data_to_portfolio_values(self) -> None:
        """Loading ETF data and weights then computing values produces a valid DataFrame."""
        _logger.debug("Testing full ETF → portfolio values pipeline")

        etf_data = load_sample_etf_data(filepath=_ETF_DATA_PATH)
        weights_df = load_portfolio_weights(filepath=_WEIGHTS_PATH)

        assert isinstance(etf_data, pl.DataFrame), "ETF data must be a Polars DataFrame"
        assert isinstance(weights_df, pl.DataFrame), "Weights must be a Polars DataFrame"

        portfolio = portfolio_QWIM(
            name_portfolio="Sample Portfolio",
            portfolio_weights=weights_df,
        )
        result = calculate_portfolio_values(
            portfolio_obj=portfolio,
            price_data=etf_data,
            initial_value=100.0,
        )

        assert isinstance(result, pl.DataFrame), "Pipeline result must be a Polars DataFrame"
        assert result.height > 0, "Pipeline result must have rows"
        value_col = [c for c in result.columns if c != "Date"][0]
        assert result[value_col][0] > 0, "First portfolio value must be positive"


# ==============================================================================
# Benchmark creation integration
# ==============================================================================


@pytest.mark.integration()
class Test_Benchmark_Creation_Integration:
    """Integration tests for benchmark portfolio value generation."""

    def test_benchmark_has_same_shape_as_portfolio_values(
        self,
        calculated_portfolio_values: pl.DataFrame,
    ) -> None:
        """Benchmark DataFrame has the same shape as the source portfolio values."""
        _logger.debug("Testing benchmark DataFrame shape matches source")

        benchmark = create_benchmark_portfolio_values(calculated_portfolio_values)

        assert isinstance(benchmark, pl.DataFrame), "Benchmark must be a Polars DataFrame"
        assert benchmark.height == calculated_portfolio_values.height, (
            "Benchmark must have same number of rows as portfolio values"
        )
        assert benchmark.width == calculated_portfolio_values.width, (
            "Benchmark must have same number of columns as portfolio values"
        )

    def test_benchmark_has_same_date_column(
        self,
        calculated_portfolio_values: pl.DataFrame,
    ) -> None:
        """Benchmark DataFrame contains same Date values as portfolio values."""
        _logger.debug("Testing benchmark date column alignment")

        benchmark = create_benchmark_portfolio_values(calculated_portfolio_values)

        assert "Date" in benchmark.columns, "Benchmark must contain Date column"
        portfolio_dates = calculated_portfolio_values["Date"].to_list()
        benchmark_dates = benchmark["Date"].to_list()
        assert portfolio_dates == benchmark_dates, "Benchmark dates must match portfolio dates"

    def test_benchmark_values_differ_from_portfolio(
        self,
        calculated_portfolio_values: pl.DataFrame,
    ) -> None:
        """Benchmark values differ from source portfolio values (performance variation applied)."""
        _logger.debug("Testing that benchmark values are distinct from source portfolio values")

        benchmark = create_benchmark_portfolio_values(calculated_portfolio_values)

        portfolio_val_col = [c for c in calculated_portfolio_values.columns if c != "Date"][0]
        benchmark_val_col = [c for c in benchmark.columns if c != "Date"][0]

        portfolio_last = calculated_portfolio_values[portfolio_val_col][-1]
        benchmark_last = benchmark[benchmark_val_col][-1]

        assert portfolio_last != benchmark_last, (
            "Benchmark final value must differ from portfolio final value"
        )

    def test_benchmark_all_positive_values(
        self,
        calculated_portfolio_values: pl.DataFrame,
    ) -> None:
        """Benchmark portfolio values are all strictly positive."""
        _logger.debug("Testing benchmark values are all positive")

        benchmark = create_benchmark_portfolio_values(calculated_portfolio_values)
        val_col = [c for c in benchmark.columns if c != "Date"][0]

        assert benchmark[val_col].min() > 0, "All benchmark values must be positive"


# ==============================================================================
# CSV persistence integration
# ==============================================================================


@pytest.mark.integration()
class Test_Portfolio_CSV_Persistence_Integration:
    """Integration tests for saving and reloading portfolio values via CSV."""

    def test_save_and_reload_portfolio_values_preserves_row_count(
        self,
        calculated_portfolio_values: pl.DataFrame,
        tmp_path: Path,
    ) -> None:
        """Saving then reloading portfolio values via CSV preserves the row count."""
        _logger.debug("Testing save → reload row count preservation")

        output_path = tmp_path / "test_portfolio_values.csv"
        saved_path = save_portfolio_values_to_csv(
            portfolio_values=calculated_portfolio_values,
            output_path=output_path,
        )

        assert saved_path is not None, "save_portfolio_values_to_csv must return a Path"
        assert saved_path.exists(), f"Saved file must exist at {saved_path}"

        reloaded = pl.read_csv(saved_path)
        assert reloaded.height == calculated_portfolio_values.height, (
            "Reloaded CSV must have the same number of rows as the original"
        )

    def test_save_and_reload_csv_has_date_and_value_columns(
        self,
        calculated_portfolio_values: pl.DataFrame,
        tmp_path: Path,
    ) -> None:
        """Reloaded portfolio CSV contains Date and Value columns."""
        _logger.debug("Testing save → reload column names")

        output_path = tmp_path / "test_portfolio_values.csv"
        saved_path = save_portfolio_values_to_csv(
            portfolio_values=calculated_portfolio_values,
            output_path=output_path,
        )

        reloaded = pl.read_csv(saved_path)
        assert "Date" in reloaded.columns, "Reloaded CSV must have Date column"
        assert reloaded.width == 2, "Reloaded CSV must have exactly 2 columns"


# ==============================================================================
# get_sample_portfolio convenience pipeline
# ==============================================================================


@pytest.mark.integration()
class Test_Sample_Portfolio_Full_Pipeline_Integration:
    """Integration tests for the full sample portfolio pipeline using explicit paths.

    These tests replicate the get_sample_portfolio() pipeline explicitly with
    correct input paths, verifying the integration between loading, construction,
    value calculation, and benchmarking steps.
    """

    @pytest.fixture()
    def full_pipeline_outputs(self) -> tuple:
        """Run the complete pipeline: load → construct → calculate → benchmark.

        Returns
        -------
        tuple
            (portfolio_obj, etf_data, portfolio_values, benchmark_values)
        """
        etf_data = load_sample_etf_data(filepath=_ETF_DATA_PATH)
        weights_df = load_portfolio_weights(filepath=_WEIGHTS_PATH)
        portfolio_obj = portfolio_QWIM(
            name_portfolio="Sample Portfolio",
            portfolio_weights=weights_df,
        )
        portfolio_values = calculate_portfolio_values(
            portfolio_obj=portfolio_obj,
            price_data=etf_data,
            initial_value=100.0,
        )
        benchmark_values = create_benchmark_portfolio_values(portfolio_values)
        return portfolio_obj, etf_data, portfolio_values, benchmark_values

    def test_full_pipeline_produces_valid_portfolio_object(
        self, full_pipeline_outputs: tuple
    ) -> None:
        """Full pipeline produces a portfolio_QWIM with at least one component."""
        _logger.debug("Testing full pipeline portfolio object validity")

        portfolio_obj, _etf, _vals, _bench = full_pipeline_outputs

        assert portfolio_obj is not None, "Portfolio object must not be None"
        assert portfolio_obj.get_num_components > 0, "Portfolio must have at least one component"

    def test_full_pipeline_etf_data_is_non_empty_dataframe(
        self, full_pipeline_outputs: tuple
    ) -> None:
        """Full pipeline ETF data is a non-empty Polars DataFrame with Date column."""
        _logger.debug("Testing full pipeline ETF data structure")

        _portfolio, etf_data, _vals, _bench = full_pipeline_outputs

        assert isinstance(etf_data, pl.DataFrame), "ETF data must be Polars DataFrame"
        assert etf_data.height > 0, "ETF data must have rows"
        assert "Date" in etf_data.columns, "ETF data must contain Date column"

    def test_full_pipeline_portfolio_values_are_valid(
        self, full_pipeline_outputs: tuple
    ) -> None:
        """Full pipeline portfolio values DataFrame has 2 columns and positive values."""
        _logger.debug("Testing full pipeline portfolio values")

        _portfolio, _etf, portfolio_values, _bench = full_pipeline_outputs

        assert isinstance(portfolio_values, pl.DataFrame), "Portfolio values must be Polars DataFrame"
        assert portfolio_values.height > 0, "Portfolio values must have rows"
        assert "Date" in portfolio_values.columns, "Portfolio values must contain Date column"
        assert portfolio_values.width == 2, "Portfolio values must have exactly 2 columns"
        val_col = [c for c in portfolio_values.columns if c != "Date"][0]
        assert portfolio_values[val_col][0] > 0, "First portfolio value must be positive"

    def test_full_pipeline_etf_components_match_portfolio(
        self, full_pipeline_outputs: tuple
    ) -> None:
        """ETF data columns include every component referenced by the portfolio."""
        _logger.debug("Testing ETF data covers all portfolio components")

        portfolio_obj, etf_data, _vals, _bench = full_pipeline_outputs

        for component in portfolio_obj.get_portfolio_components:
            assert component in etf_data.columns, (
                f"Component '{component}' must be present in ETF data columns"
            )

    def test_full_pipeline_initial_value_is_100(
        self, full_pipeline_outputs: tuple
    ) -> None:
        """First portfolio value in the pipeline output is approximately 100.0."""
        _logger.debug("Testing initial portfolio value in full pipeline")

        _portfolio, _etf, portfolio_values, _bench = full_pipeline_outputs

        val_col = [c for c in portfolio_values.columns if c != "Date"][0]
        first_value = portfolio_values[val_col][0]

        assert abs(first_value - 100.0) < 1e-2, (
            f"Initial portfolio value must be ~100.0, got {first_value}"
        )

    def test_full_pipeline_benchmark_aligns_with_portfolio_values(
        self, full_pipeline_outputs: tuple
    ) -> None:
        """Benchmark produced from portfolio values has matching row count and Date column."""
        _logger.debug("Testing full pipeline benchmark alignment")

        _portfolio, _etf, portfolio_values, benchmark_values = full_pipeline_outputs

        assert isinstance(benchmark_values, pl.DataFrame), "Benchmark must be Polars DataFrame"
        assert benchmark_values.height == portfolio_values.height, (
            "Benchmark row count must match portfolio values row count"
        )
        assert "Date" in benchmark_values.columns, "Benchmark must contain Date column"
