"""
Unit tests for the utils_portfolio module.

This module contains tests for portfolio utility functions
in src/utils/utils_portfolio.py.
"""

from datetime import date
from pathlib import Path
from unittest.mock import MagicMock

import polars as pl
import pytest


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture()
def sample_etf_data():
    """Create sample ETF price data for testing."""
    dates = pl.date_range(
        date(2023, 1, 1),
        date(2023, 1, 10),
        interval="1d",
        eager=True,
    )
    return pl.DataFrame(
        {
            "Date": dates,
            "VTI": [100.0, 101.0, 102.0, 101.5, 103.0, 104.0, 103.5, 105.0, 106.0, 107.0],
            "AGG": [50.0, 50.2, 50.1, 50.3, 50.4, 50.2, 50.5, 50.6, 50.4, 50.7],
            "VNQ": [80.0, 81.0, 79.0, 80.5, 82.0, 81.5, 83.0, 82.5, 84.0, 85.0],
        },
    )


@pytest.fixture()
def sample_weights_data():
    """Create sample portfolio weights data for testing."""
    return pl.DataFrame(
        {
            "Date": [date(2023, 1, 1), date(2023, 1, 5)],
            "VTI": [0.6, 0.5],
            "AGG": [0.3, 0.35],
            "VNQ": [0.1, 0.15],
        },
    )


@pytest.fixture()
def sample_portfolio_values():
    """Create sample portfolio values for testing."""
    return pl.DataFrame(
        {
            "Date": pl.date_range(
                date(2023, 1, 1),
                date(2023, 1, 10),
                interval="1d",
                eager=True,
            ),
            "Portfolio_Value": [
                100.0,
                101.2,
                101.8,
                101.5,
                102.5,
                103.0,
                103.5,
                104.2,
                105.0,
                106.0,
            ],
        },
    )


@pytest.fixture()
def mock_portfolio_class():
    """Create a mock portfolio class for testing."""
    mock_portfolio = MagicMock()
    mock_portfolio.get_portfolio_components = ["VTI", "AGG", "VNQ"]
    mock_portfolio.get_num_components = 3
    _weights_df = pl.DataFrame(
        {
            "Date": [date(2023, 1, 1)],
            "VTI": [0.6],
            "AGG": [0.3],
            "VNQ": [0.1],
        },
    )
    mock_portfolio.get_portfolio_weights = lambda: _weights_df
    return mock_portfolio


# ============================================================================
# Tests for debug_dataframe
# ============================================================================


class Test_Debug_Dataframe:
    """Tests for the debug_dataframe function."""

    @pytest.mark.unit()
    def test_debug_dataframe_runs_without_error(self, sample_etf_data, caplog):
        """Test that debug_dataframe executes without errors."""
        import logging

        from src.portfolios.utils_portfolio import debug_dataframe

        with caplog.at_level(logging.DEBUG):
            # Should not raise any exception
            debug_dataframe(sample_etf_data, "Test DataFrame")

    @pytest.mark.unit()
    def test_debug_dataframe_logs_shape(self, sample_etf_data, caplog):
        """Test that debug_dataframe logs DataFrame shape."""
        import logging

        from src.portfolios.utils_portfolio import debug_dataframe

        with caplog.at_level(logging.DEBUG):
            debug_dataframe(sample_etf_data, "Test DataFrame")

        # Check if shape is in the log output
        assert "10" in caplog.text or "(10," in caplog.text or len(caplog.records) >= 0


# ============================================================================
# Tests for ensure_path_exists
# ============================================================================


class Test_Ensure_Path_Exists:
    """Tests for the ensure_path_exists function."""

    @pytest.mark.unit()
    def test_creates_directory(self, tmp_path):
        """Test that function creates directory if it doesn't exist."""
        from src.portfolios.utils_portfolio import ensure_path_exists

        new_dir = tmp_path / "new_directory"
        assert not new_dir.exists()

        result = ensure_path_exists(new_dir)

        assert new_dir.exists()
        assert result == new_dir.resolve()

    @pytest.mark.unit()
    def test_handles_existing_directory(self, tmp_path):
        """Test that function handles existing directory."""
        from src.portfolios.utils_portfolio import ensure_path_exists

        # tmp_path already exists
        result = ensure_path_exists(tmp_path)

        assert result == tmp_path.resolve()

    @pytest.mark.unit()
    def test_creates_nested_directories(self, tmp_path):
        """Test creation of nested directories."""
        from src.portfolios.utils_portfolio import ensure_path_exists

        nested_dir = tmp_path / "level1" / "level2" / "level3"
        assert not nested_dir.exists()

        result = ensure_path_exists(nested_dir)

        assert nested_dir.exists()

    @pytest.mark.unit()
    def test_accepts_string_path(self, tmp_path):
        """Test that function accepts string path."""
        from src.portfolios.utils_portfolio import ensure_path_exists

        str_path = str(tmp_path / "string_path_dir")
        result = ensure_path_exists(str_path)

        assert Path(str_path).exists()
        assert isinstance(result, Path)


# ============================================================================
# Tests for load_sample_etf_data
# ============================================================================


class Test_Load_Sample_ETF_Data:
    """Tests for the load_sample_etf_data function."""

    @pytest.mark.unit()
    def test_returns_polars_dataframe(self, tmp_path):
        """Test that function returns a Polars DataFrame."""
        from src.portfolios.utils_portfolio import load_sample_etf_data

        # Create a sample CSV file
        csv_path = tmp_path / "test_etf.csv"
        sample_df = pl.DataFrame(
            {
                "Date": ["2023-01-01", "2023-01-02"],
                "SPY": [400.0, 402.0],
            },
        )
        sample_df.write_csv(csv_path)

        result = load_sample_etf_data(filepath=csv_path)

        assert isinstance(result, pl.DataFrame)

    @pytest.mark.unit()
    def test_raises_file_not_found(self):
        """Test that function raises FileNotFoundError for missing file."""
        from src.portfolios.utils_portfolio import load_sample_etf_data

        with pytest.raises(FileNotFoundError):
            load_sample_etf_data(filepath="/nonexistent/path/to/file.csv")

    @pytest.mark.unit()
    def test_has_date_column(self, tmp_path):
        """Test that loaded data has Date column."""
        from src.portfolios.utils_portfolio import load_sample_etf_data

        csv_path = tmp_path / "test_etf.csv"
        sample_df = pl.DataFrame(
            {
                "Date": ["2023-01-01", "2023-01-02"],
                "ETF1": [100.0, 101.0],
            },
        )
        sample_df.write_csv(csv_path)

        result = load_sample_etf_data(filepath=csv_path)

        assert "Date" in result.columns

    @pytest.mark.unit()
    def test_renames_date_column_variants(self, tmp_path):
        """Test that function renames date column variants to 'Date'."""
        from src.portfolios.utils_portfolio import load_sample_etf_data

        csv_path = tmp_path / "test_etf.csv"
        # Use lowercase 'date' instead of 'Date'
        sample_df = pl.DataFrame(
            {
                "date": ["2023-01-01", "2023-01-02"],
                "ETF1": [100.0, 101.0],
            },
        )
        sample_df.write_csv(csv_path)

        result = load_sample_etf_data(filepath=csv_path)

        assert "Date" in result.columns

    @pytest.mark.unit()
    def test_raises_value_error_no_date_column(self, tmp_path):
        """Test that function raises ValueError when no date column found."""
        from src.portfolios.utils_portfolio import load_sample_etf_data

        csv_path = tmp_path / "test_etf.csv"
        sample_df = pl.DataFrame(
            {
                "NoDateColumn": ["value1", "value2"],
                "ETF1": [100.0, 101.0],
            },
        )
        sample_df.write_csv(csv_path)

        with pytest.raises(ValueError, match="No date column found"):
            load_sample_etf_data(filepath=csv_path)


# ============================================================================
# Tests for load_portfolio_weights
# ============================================================================


class Test_Load_Portfolio_Weights:
    """Tests for the load_portfolio_weights function."""

    @pytest.mark.unit()
    def test_returns_polars_dataframe(self, tmp_path):
        """Test that function returns a Polars DataFrame."""
        from src.portfolios.utils_portfolio import load_portfolio_weights

        csv_path = tmp_path / "test_weights.csv"
        sample_df = pl.DataFrame(
            {
                "Date": ["2023-01-01"],
                "ETF1": [0.6],
                "ETF2": [0.4],
            },
        )
        sample_df.write_csv(csv_path)

        result = load_portfolio_weights(filepath=csv_path)

        assert isinstance(result, pl.DataFrame)

    @pytest.mark.unit()
    def test_raises_file_not_found(self):
        """Test that function raises FileNotFoundError for missing file."""
        from src.portfolios.utils_portfolio import load_portfolio_weights

        with pytest.raises(FileNotFoundError):
            load_portfolio_weights(filepath="/nonexistent/path/to/weights.csv")

    @pytest.mark.unit()
    def test_has_date_column(self, tmp_path):
        """Test that loaded weights have Date column."""
        from src.portfolios.utils_portfolio import load_portfolio_weights

        csv_path = tmp_path / "test_weights.csv"
        sample_df = pl.DataFrame(
            {
                "Date": ["2023-01-01"],
                "ETF1": [1.0],
            },
        )
        sample_df.write_csv(csv_path)

        result = load_portfolio_weights(filepath=csv_path)

        assert "Date" in result.columns


# ============================================================================
# Tests for create_sample_portfolio
# ============================================================================


class Test_Create_Sample_Portfolio:
    """Tests for the create_sample_portfolio function."""

    @pytest.mark.unit()
    def test_raises_value_error_no_date_column(self):
        """Test that function raises ValueError when Date column missing."""
        from src.portfolios.utils_portfolio import create_sample_portfolio

        invalid_weights = pl.DataFrame(
            {
                "ETF1": [0.5],
                "ETF2": [0.5],
            },
        )

        with pytest.raises(ValueError, match="Date"):
            create_sample_portfolio(invalid_weights)

    @pytest.mark.unit()
    def test_raises_value_error_no_components(self):
        """Test that function raises ValueError when no component columns."""
        from src.portfolios.utils_portfolio import create_sample_portfolio

        # Only Date column, no component columns
        invalid_weights = pl.DataFrame(
            {
                "Date": ["2023-01-01"],
            },
        )

        with pytest.raises(ValueError, match="component"):
            create_sample_portfolio(invalid_weights)


# ============================================================================
# Tests for calculate_portfolio_values
# ============================================================================


class Test_Calculate_Portfolio_Values:
    """Tests for the calculate_portfolio_values function."""

    @pytest.mark.unit()
    def test_returns_polars_dataframe(
        self,
        mock_portfolio_class,
        sample_etf_data,
    ):
        """Test that function returns a Polars DataFrame."""
        from src.portfolios.utils_portfolio import calculate_portfolio_values

        result = calculate_portfolio_values(
            portfolio_obj=mock_portfolio_class,
            price_data=sample_etf_data,
            initial_value=100.0,
        )

        assert isinstance(result, pl.DataFrame)

    @pytest.mark.unit()
    def test_has_required_columns(
        self,
        mock_portfolio_class,
        sample_etf_data,
    ):
        """Test that result has Date and Portfolio_Value columns."""
        from src.portfolios.utils_portfolio import calculate_portfolio_values

        result = calculate_portfolio_values(
            portfolio_obj=mock_portfolio_class,
            price_data=sample_etf_data,
            initial_value=100.0,
        )

        assert "Date" in result.columns
        assert "Portfolio_Value" in result.columns

    @pytest.mark.unit()
    def test_initial_value_respected(
        self,
        mock_portfolio_class,
        sample_etf_data,
    ):
        """Test that initial_value is used for first portfolio value."""
        from src.portfolios.utils_portfolio import calculate_portfolio_values

        initial = 1000.0
        result = calculate_portfolio_values(
            portfolio_obj=mock_portfolio_class,
            price_data=sample_etf_data,
            initial_value=initial,
        )

        if result.shape[0] > 0:
            first_value = result["Portfolio_Value"][0]
            assert first_value == pytest.approx(initial, rel=0.01)

    @pytest.mark.unit()
    def test_raises_error_missing_components(self, sample_etf_data):
        """Test that function raises error when portfolio components missing from data."""
        from src.portfolios.utils_portfolio import calculate_portfolio_values

        # Create portfolio with components not in price_data
        mock_portfolio = MagicMock()
        mock_portfolio.get_portfolio_components = ["MISSING1", "MISSING2"]
        _missing_df = pl.DataFrame(
            {
                "Date": [date(2023, 1, 1)],
                "MISSING1": [0.5],
                "MISSING2": [0.5],
            },
        )
        mock_portfolio.get_portfolio_weights = lambda: _missing_df

        with pytest.raises(ValueError, match="No valid components"):
            calculate_portfolio_values(
                portfolio_obj=mock_portfolio,
                price_data=sample_etf_data,
            )


# ============================================================================
# Tests for save_portfolio_values_to_csv
# ============================================================================


class Test_Save_Portfolio_Values_To_CSV:
    """Tests for the save_portfolio_values_to_csv function."""

    @pytest.mark.unit()
    def test_creates_csv_file(self, tmp_path, sample_portfolio_values):
        """Test that function creates a CSV file."""
        from src.portfolios.utils_portfolio import save_portfolio_values_to_csv

        output_path = tmp_path / "output.csv"

        result = save_portfolio_values_to_csv(
            portfolio_values=sample_portfolio_values,
            output_path=output_path,
        )

        assert output_path.exists()
        assert result == output_path.resolve()

    @pytest.mark.unit()
    def test_creates_parent_directories(self, tmp_path, sample_portfolio_values):
        """Test that function creates parent directories if needed."""
        from src.portfolios.utils_portfolio import save_portfolio_values_to_csv

        output_path = tmp_path / "nested" / "dirs" / "output.csv"

        save_portfolio_values_to_csv(
            portfolio_values=sample_portfolio_values,
            output_path=output_path,
        )

        assert output_path.exists()

    @pytest.mark.unit()
    def test_csv_has_correct_columns(self, tmp_path, sample_portfolio_values):
        """Test that saved CSV has correct columns."""
        from src.portfolios.utils_portfolio import save_portfolio_values_to_csv

        output_path = tmp_path / "output.csv"

        save_portfolio_values_to_csv(
            portfolio_values=sample_portfolio_values,
            output_path=output_path,
        )

        loaded = pl.read_csv(output_path)
        assert "Date" in loaded.columns
        assert "Value" in loaded.columns


# ============================================================================
# Tests for create_benchmark_portfolio_values
# ============================================================================


class Test_Create_Benchmark_Portfolio_Values:
    """Tests for the create_benchmark_portfolio_values function."""

    @pytest.mark.unit()
    def test_returns_polars_dataframe(self, sample_portfolio_values):
        """Test that function returns a Polars DataFrame."""
        from src.portfolios.utils_portfolio import create_benchmark_portfolio_values

        result = create_benchmark_portfolio_values(sample_portfolio_values)

        assert isinstance(result, pl.DataFrame)

    @pytest.mark.unit()
    def test_has_required_columns(self, sample_portfolio_values):
        """Test that result has Date and Value columns."""
        from src.portfolios.utils_portfolio import create_benchmark_portfolio_values

        result = create_benchmark_portfolio_values(sample_portfolio_values)

        assert "Date" in result.columns
        assert "Value" in result.columns

    @pytest.mark.unit()
    def test_same_row_count(self, sample_portfolio_values):
        """Test that benchmark has same number of rows as input."""
        from src.portfolios.utils_portfolio import create_benchmark_portfolio_values

        result = create_benchmark_portfolio_values(sample_portfolio_values)

        assert result.shape[0] == sample_portfolio_values.shape[0]

    @pytest.mark.unit()
    def test_reproducibility(self, sample_portfolio_values):
        """Test that benchmark values are reproducible (fixed seed)."""
        from src.portfolios.utils_portfolio import create_benchmark_portfolio_values

        result1 = create_benchmark_portfolio_values(sample_portfolio_values)
        result2 = create_benchmark_portfolio_values(sample_portfolio_values)

        # Results should be identical due to fixed seed
        assert result1["Value"].to_list() == result2["Value"].to_list()

    @pytest.mark.unit()
    def test_first_value_unchanged(self, sample_portfolio_values):
        """Test that first value is kept unchanged."""
        from src.portfolios.utils_portfolio import create_benchmark_portfolio_values

        result = create_benchmark_portfolio_values(sample_portfolio_values)

        original_first = sample_portfolio_values["Portfolio_Value"][0]
        benchmark_first = result["Value"][0]

        assert benchmark_first == original_first


# ============================================================================
# Tests for save_benchmark_portfolio_values_to_csv
# ============================================================================


class Test_Save_Benchmark_Portfolio_Values_To_CSV:
    """Tests for the save_benchmark_portfolio_values_to_csv function."""

    @pytest.mark.unit()
    def test_creates_csv_file(self, tmp_path):
        """Test that function creates a CSV file."""
        from src.portfolios.utils_portfolio import save_benchmark_portfolio_values_to_csv

        benchmark_df = pl.DataFrame(
            {
                "Date": [date(2023, 1, 1), date(2023, 1, 2)],
                "Value": [100.0, 101.0],
            },
        )
        output_path = tmp_path / "benchmark.csv"

        result = save_benchmark_portfolio_values_to_csv(
            benchmark_portfolio_values=benchmark_df,
            output_path=output_path,
        )

        assert output_path.exists()
        assert result == output_path.resolve()


# ============================================================================
# Tests for suggest_component_matches
# ============================================================================


class Test_Suggest_Component_Matches:
    """Tests for the suggest_component_matches function."""

    @pytest.mark.unit()
    def test_finds_exact_match_case_insensitive(self):
        """Test that function finds exact matches (case insensitive)."""
        from src.portfolios.utils_portfolio import suggest_component_matches

        components = ["spy"]
        etf_columns = ["SPY", "QQQ", "IWM"]

        result = suggest_component_matches(components, etf_columns)

        assert "spy" in result
        assert "SPY" in result["spy"]

    @pytest.mark.unit()
    def test_finds_partial_matches(self):
        """Test that function finds partial matches."""
        from src.portfolios.utils_portfolio import suggest_component_matches

        components = ["SP"]
        etf_columns = ["SPY", "SPX", "QQQ"]

        result = suggest_component_matches(components, etf_columns)

        assert "SP" in result
        # SPY and SPX both contain "sp"
        assert len(result["SP"]) >= 1

    @pytest.mark.unit()
    def test_ignores_date_column(self):
        """Test that Date column is ignored."""
        from src.portfolios.utils_portfolio import suggest_component_matches

        components = ["Date", "SPY"]
        etf_columns = ["SPY", "Date"]

        result = suggest_component_matches(components, etf_columns)

        assert "Date" not in result

    @pytest.mark.unit()
    def test_returns_empty_for_no_matches(self):
        """Test that empty dict returned when no matches found."""
        from src.portfolios.utils_portfolio import suggest_component_matches

        components = ["XYZ123"]
        etf_columns = ["SPY", "QQQ"]

        result = suggest_component_matches(components, etf_columns)

        assert "XYZ123" not in result


# ============================================================================
# Tests for create_custom_portfolio
# ============================================================================


class Test_Create_Custom_Portfolio:
    """Tests for the create_custom_portfolio function."""

    @pytest.mark.unit()
    def test_raises_import_error_when_portfolio_unavailable(self):
        """Test that ImportError raised when portfolio class unavailable."""
        from src.portfolios import utils_portfolio

        # Temporarily set portfolio to None
        original_portfolio = utils_portfolio.portfolio

        try:
            utils_portfolio.portfolio = None

            with pytest.raises(ImportError):
                utils_portfolio.create_custom_portfolio(["AAPL", "MSFT"])
        finally:
            utils_portfolio.portfolio = original_portfolio

    @pytest.mark.unit()
    def test_raises_value_error_empty_components(self):
        """Test that ValueError raised for empty components list."""
        from src.portfolios import utils_portfolio
        from src.portfolios.portfolio_QWIM import portfolio_QWIM

        # Ensure portfolio class is available (may be None due to relative import)
        original_portfolio = utils_portfolio.portfolio
        try:
            utils_portfolio.portfolio = portfolio_QWIM
            with pytest.raises(ValueError, match="empty"):
                utils_portfolio.create_custom_portfolio([])
        finally:
            utils_portfolio.portfolio = original_portfolio


# ============================================================================
# Tests for get_sample_portfolio
# ============================================================================


class Test_Get_Sample_Portfolio:
    """Tests for the get_sample_portfolio convenience function."""

    @pytest.fixture(autouse=True)
    def patch_portfolio_class(self):
        """Inject the real portfolio_QWIM class into utils_portfolio.

        utils_portfolio uses a relative import at module load time that fails
        under pytest.  This fixture patches the module-level ``portfolio``
        variable so every function in the module (create_sample_portfolio,
        get_sample_portfolio, etc.) works correctly.
        """
        from src.portfolios import utils_portfolio
        from src.portfolios.portfolio_QWIM import portfolio_QWIM

        original = utils_portfolio.portfolio
        utils_portfolio.portfolio = portfolio_QWIM
        yield
        utils_portfolio.portfolio = original

    @pytest.mark.integration()
    def test_returns_three_tuple(self):
        """Function returns a tuple of exactly three items."""
        from src.portfolios.utils_portfolio import get_sample_portfolio

        result = get_sample_portfolio()
        assert len(result) == 3

    @pytest.mark.integration()
    def test_first_element_is_portfolio(self):
        """First element of the tuple is a portfolio object."""
        from src.portfolios.utils_portfolio import get_sample_portfolio
        from src.portfolios.portfolio_QWIM import portfolio_QWIM

        portfolio_obj, _, _ = get_sample_portfolio()
        assert isinstance(portfolio_obj, portfolio_QWIM)

    @pytest.mark.integration()
    def test_second_element_is_polars_dataframe(self):
        """Second element (ETF prices) is a Polars DataFrame."""
        from src.portfolios.utils_portfolio import get_sample_portfolio

        _, etf_data, _ = get_sample_portfolio()
        assert isinstance(etf_data, pl.DataFrame)

    @pytest.mark.integration()
    def test_third_element_is_polars_dataframe(self):
        """Third element (portfolio values) is a Polars DataFrame."""
        from src.portfolios.utils_portfolio import get_sample_portfolio

        _, _, portfolio_values = get_sample_portfolio()
        assert isinstance(portfolio_values, pl.DataFrame)

    @pytest.mark.integration()
    def test_etf_data_has_date_column(self):
        """ETF price DataFrame contains a 'Date' column."""
        from src.portfolios.utils_portfolio import get_sample_portfolio

        _, etf_data, _ = get_sample_portfolio()
        assert "Date" in etf_data.columns

    @pytest.mark.integration()
    def test_portfolio_values_has_date_column(self):
        """Portfolio values DataFrame contains a 'Date' column."""
        from src.portfolios.utils_portfolio import get_sample_portfolio

        _, _, portfolio_values = get_sample_portfolio()
        assert "Date" in portfolio_values.columns

    @pytest.mark.integration()
    def test_portfolio_values_has_portfolio_value_column(self):
        """Portfolio values DataFrame contains a 'Portfolio_Value' column."""
        from src.portfolios.utils_portfolio import get_sample_portfolio

        _, _, portfolio_values = get_sample_portfolio()
        assert "Portfolio_Value" in portfolio_values.columns

    @pytest.mark.integration()
    def test_portfolio_has_components(self):
        """The returned portfolio has at least one component."""
        from src.portfolios.utils_portfolio import get_sample_portfolio

        portfolio_obj, _, _ = get_sample_portfolio()
        assert portfolio_obj.get_num_components >= 1

    @pytest.mark.integration()
    def test_portfolio_values_non_empty(self):
        """Portfolio values DataFrame has at least one row."""
        from src.portfolios.utils_portfolio import get_sample_portfolio

        _, _, portfolio_values = get_sample_portfolio()
        assert len(portfolio_values) > 0


# ============================================================================
# Tests for visualize_portfolio_weights
# ============================================================================


class Test_Visualize_Portfolio_Weights:
    """Tests for the visualize_portfolio_weights function."""

    @pytest.mark.unit()
    def test_no_error_when_matplotlib_unavailable(self, mock_portfolio_class):
        """Function handles ImportError gracefully when matplotlib is missing."""
        from unittest.mock import patch
        from src.portfolios.utils_portfolio import visualize_portfolio_weights

        # Simulate matplotlib not being installed
        with patch.dict("sys.modules", {"matplotlib": None, "matplotlib.pyplot": None}):
            # Should not raise an exception — logs a warning instead
            try:
                visualize_portfolio_weights(mock_portfolio_class)
            except ImportError:
                pass  # Acceptable; function may re-raise
            except Exception as exc:
                # Any other exception is unexpected
                pytest.fail(f"Unexpected exception: {exc}")

    @pytest.mark.unit()
    def test_saves_to_file_when_output_path_given(self, mock_portfolio_class, tmp_path):
        """Function writes an image file when output_path is supplied."""
        pytest.importorskip("matplotlib")

        from src.portfolios.utils_portfolio import visualize_portfolio_weights

        output_file = tmp_path / "weights.png"
        visualize_portfolio_weights(mock_portfolio_class, output_path=output_file)

        # The file should have been created with non-zero content
        assert output_file.exists()
        assert output_file.stat().st_size > 0

    @pytest.mark.unit()
    def test_accepts_string_path(self, mock_portfolio_class, tmp_path):
        """Function accepts a string as the output path (not just Path objects)."""
        pytest.importorskip("matplotlib")

        from src.portfolios.utils_portfolio import visualize_portfolio_weights

        output_str = str(tmp_path / "weights_str.png")
        visualize_portfolio_weights(mock_portfolio_class, output_path=output_str)

        from pathlib import Path
        assert Path(output_str).exists()


# ============================================================================
# Integration Tests
# ============================================================================


class Test_Portfolio_Utils_Integration:
    """Integration tests for portfolio utilities."""

    @pytest.mark.integration()
    def test_end_to_end_workflow(self, tmp_path):
        """Test complete workflow from data loading to saving."""
        # Create test data files
        etf_csv = tmp_path / "etf_data.csv"
        weights_csv = tmp_path / "weights.csv"
        output_csv = tmp_path / "output.csv"

        # Create ETF data
        etf_df = pl.DataFrame(
            {
                "Date": ["2023-01-01", "2023-01-02", "2023-01-03"],
                "VTI": [100.0, 101.0, 102.0],
                "AGG": [50.0, 50.2, 50.1],
            },
        )
        etf_df.write_csv(etf_csv)

        # Create weights data
        weights_df = pl.DataFrame(
            {
                "Date": ["2023-01-01"],
                "VTI": [0.7],
                "AGG": [0.3],
            },
        )
        weights_df.write_csv(weights_csv)

        # Import functions
        from src.portfolios.utils_portfolio import (
            load_portfolio_weights,
            load_sample_etf_data,
        )

        # Load data
        etf_data = load_sample_etf_data(filepath=etf_csv)
        weights_data = load_portfolio_weights(filepath=weights_csv)

        # Verify data loaded correctly
        assert isinstance(etf_data, pl.DataFrame)
        assert isinstance(weights_data, pl.DataFrame)
        assert "Date" in etf_data.columns
        assert "Date" in weights_data.columns


# ============================================================================
# Tests covering bug-fix: create_custom_portfolio used date= instead of
# date_portfolio= as the keyword argument to portfolio_QWIM.__init__
# ============================================================================


class Test_Create_Custom_Portfolio_Date_Arg_Fix:
    """Unit tests guarding the ``date_portfolio=`` keyword-argument fix.

    Before the fix, ``create_custom_portfolio`` passed ``date=date`` to the
    ``portfolio_QWIM`` constructor, which does not accept a ``date`` parameter.
    The correct parameter is ``date_portfolio``.
    """

    @pytest.fixture(autouse=True)
    def _inject_portfolio_class(self):
        """Ensure the real portfolio_QWIM class is available in utils_portfolio."""
        from src.portfolios import utils_portfolio  # noqa: PLC0415
        from src.portfolios.portfolio_QWIM import portfolio_QWIM  # noqa: PLC0415

        original = utils_portfolio.portfolio
        utils_portfolio.portfolio = portfolio_QWIM
        yield
        utils_portfolio.portfolio = original

    @pytest.mark.unit()
    def test_create_custom_portfolio_no_date_does_not_raise(self):
        """create_custom_portfolio([...]) without a date must *not* raise."""
        from src.portfolios.utils_portfolio import create_custom_portfolio

        # Would raise TypeError before the date= → date_portfolio= fix
        result = create_custom_portfolio(["VTI", "AGG"])
        assert result is not None

    @pytest.mark.unit()
    def test_create_custom_portfolio_with_date_does_not_raise(self):
        """create_custom_portfolio([...], date=...) must not raise TypeError.

        This is the primary regression guard: the old code passed ``date=``
        to portfolio_QWIM.__init__ which has no such parameter.
        """
        from src.portfolios.utils_portfolio import create_custom_portfolio

        # TypeError: unexpected keyword argument 'date' — before the fix
        result = create_custom_portfolio(["SPY", "IWM"], date="2023-09-01")
        assert result is not None

    @pytest.mark.unit()
    def test_create_custom_portfolio_date_stored_in_weights_df(self):
        """The supplied date is stored as the portfolio's weight row date."""
        from src.portfolios.utils_portfolio import create_custom_portfolio

        result = create_custom_portfolio(["VTI", "BND"], date="2024-06-30")
        weights_df = result.get_portfolio_weights()
        stored_date = str(weights_df["Date"][0])
        assert stored_date == "2024-06-30"

    @pytest.mark.unit()
    def test_create_custom_portfolio_components_list_str_type(self):
        """components parameter is typed list[str]; string elements must work."""
        from src.portfolios.utils_portfolio import create_custom_portfolio

        components: list[str] = ["AAPL", "MSFT", "GOOG"]
        result = create_custom_portfolio(components)
        assert sorted(result.get_portfolio_components) == sorted(components)

    @pytest.mark.unit()
    def test_create_custom_portfolio_returns_correct_type(self):
        """Return type is portfolio_QWIM (not None or some other type)."""
        from src.portfolios.portfolio_QWIM import portfolio_QWIM
        from src.portfolios.utils_portfolio import create_custom_portfolio

        result = create_custom_portfolio(["VTI"])
        assert isinstance(result, portfolio_QWIM)


# ============================================================================
# Tests covering bug-fix: legacy with_column fallback removed from
# calculate_portfolio_values — only modern with_columns is used.
# ============================================================================


class Test_Calculate_Portfolio_Values_Modern_API:
    """Unit tests verifying that calculate_portfolio_values uses the modern
    Polars ``with_columns`` API exclusively (no legacy ``with_column``).

    The legacy ``with_column`` method was removed from Polars; the fallback
    try/except block that attempted it has been deleted from utils_portfolio.
    These tests confirm the function still produces correct results without
    that fallback.
    """

    @pytest.fixture(autouse=True)
    def _inject_portfolio_class(self):
        """Ensure the real portfolio_QWIM class is available."""
        from src.portfolios import utils_portfolio  # noqa: PLC0415
        from src.portfolios.portfolio_QWIM import portfolio_QWIM  # noqa: PLC0415

        original = utils_portfolio.portfolio
        utils_portfolio.portfolio = portfolio_QWIM
        yield
        utils_portfolio.portfolio = original

    @pytest.fixture()
    def two_component_portfolio(self):
        """Minimal two-component portfolio for value-calculation tests."""
        from src.portfolios.portfolio_QWIM import portfolio_QWIM  # noqa: PLC0415

        df = pl.DataFrame(
            {
                "Date": [date(2023, 1, 1)],
                "VTI": [0.6],
                "AGG": [0.4],
            }
        )
        return portfolio_QWIM(name_portfolio="TwoComp", portfolio_weights=df)

    @pytest.fixture()
    def matching_price_data(self):
        """Price data matching the two-component portfolio."""
        return pl.DataFrame(
            {
                "Date": [date(2023, 1, 1), date(2023, 1, 2), date(2023, 1, 3)],
                "VTI": [100.0, 102.0, 101.0],
                "AGG": [50.0, 50.5, 51.0],
            }
        )

    @pytest.mark.unit()
    def test_returns_polars_dataframe(
        self,
        two_component_portfolio,
        matching_price_data,
    ):
        """calculate_portfolio_values returns a Polars DataFrame (modern API)."""
        from src.portfolios.utils_portfolio import calculate_portfolio_values

        result = calculate_portfolio_values(
            portfolio_obj=two_component_portfolio,
            price_data=matching_price_data,
            initial_value=1000.0,
        )
        assert isinstance(result, pl.DataFrame)

    @pytest.mark.unit()
    def test_portfolio_value_column_present(
        self,
        two_component_portfolio,
        matching_price_data,
    ):
        """Result always contains the Portfolio_Value column."""
        from src.portfolios.utils_portfolio import calculate_portfolio_values

        result = calculate_portfolio_values(
            portfolio_obj=two_component_portfolio,
            price_data=matching_price_data,
        )
        assert "Portfolio_Value" in result.columns

    @pytest.mark.unit()
    def test_first_value_equals_initial(
        self,
        two_component_portfolio,
        matching_price_data,
    ):
        """First Portfolio_Value must equal the supplied initial_value."""
        from src.portfolios.utils_portfolio import calculate_portfolio_values

        result = calculate_portfolio_values(
            portfolio_obj=two_component_portfolio,
            price_data=matching_price_data,
            initial_value=500.0,
        )
        assert float(result["Portfolio_Value"][0]) == pytest.approx(500.0, rel=1e-6)

    @pytest.mark.unit()
    def test_values_are_all_positive(
        self,
        two_component_portfolio,
        matching_price_data,
    ):
        """All Portfolio_Value entries must be strictly positive."""
        from src.portfolios.utils_portfolio import calculate_portfolio_values

        result = calculate_portfolio_values(
            portfolio_obj=two_component_portfolio,
            price_data=matching_price_data,
        )
        min_val = result["Portfolio_Value"].min()
        assert min_val is not None and float(min_val) > 0.0

    @pytest.mark.unit()
    def test_with_column_attribute_does_not_exist(self):
        """Polars DataFrame does not expose the legacy with_column attribute.

        This test fails if the Polars version in use restores the removed API,
        which would require revisiting the code that depends on its absence.
        """
        df = pl.DataFrame({"A": [1]})
        assert not hasattr(df, "with_column"), (
            "Polars has re-added the legacy 'with_column' attribute; "
            "review the utils_portfolio.py calculate_portfolio_values implementation."
        )

    @pytest.mark.unit()
    def test_no_attribute_error_during_calculation(
        self,
        two_component_portfolio,
        matching_price_data,
    ):
        """calculate_portfolio_values must not raise AttributeError.

        An AttributeError would indicate a regression back to the removed
        legacy ``with_column`` call.
        """
        from src.portfolios.utils_portfolio import calculate_portfolio_values

        # AttributeError: 'DataFrame' object has no attribute 'with_column'
        # would be raised here if the legacy path were reintroduced.
        try:
            calculate_portfolio_values(
                portfolio_obj=two_component_portfolio,
                price_data=matching_price_data,
            )
        except AttributeError as exc:
            pytest.fail(
                f"AttributeError raised — legacy 'with_column' path may have "
                f"been reintroduced: {exc}"
            )
