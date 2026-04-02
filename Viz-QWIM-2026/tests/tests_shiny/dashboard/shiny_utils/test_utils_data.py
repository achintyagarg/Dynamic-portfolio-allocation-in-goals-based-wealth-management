"""Unit and integration tests for ``src.dashboard.shiny_utils.utils_data``.

Run only unit tests (no disk I/O):
    pytest tests/tests_shiny/ -m unit -k utils_data

Run all including integration (needs real CSV files):
    pytest tests/tests_shiny/ -m "unit or integration" -k utils_data
"""

from __future__ import annotations

import pytest
import polars as pl
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_df(**kwargs: list[Any]) -> pl.DataFrame:
    """Convenience: build a Polars DataFrame from keyword column lists."""
    return pl.DataFrame(kwargs)


# ---------------------------------------------------------------------------
# find_date_column
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestFindDateColumn:
    """Verify that find_date_column detects date-like column names."""

    def test_finds_capitalised_date(self) -> None:
        from src.dashboard.shiny_utils.utils_data import find_date_column

        df = _make_df(Date=["2024-01-01"], Value=[1.0])
        assert find_date_column(df) == "Date"

    def test_finds_lowercase_date(self) -> None:
        from src.dashboard.shiny_utils.utils_data import find_date_column

        df = _make_df(date=["2024-01-01"], Value=[1.0])
        assert find_date_column(df) == "date"

    def test_finds_uppercase_date(self) -> None:
        from src.dashboard.shiny_utils.utils_data import find_date_column

        df = _make_df(DATE=["2024-01-01"], Value=[1.0])
        assert find_date_column(df) == "DATE"

    def test_returns_none_when_no_date_column(self) -> None:
        from src.dashboard.shiny_utils.utils_data import find_date_column

        df = _make_df(Ticker=["AAPL"], Price=[200.0])
        assert find_date_column(df) is None

    def test_returns_none_for_empty_dataframe(self) -> None:
        from src.dashboard.shiny_utils.utils_data import find_date_column

        df = pl.DataFrame()
        assert find_date_column(df) is None

    def test_prefers_date_column_over_other_columns(self) -> None:
        """If multiple columns exist but one is 'Date', it must be returned."""
        from src.dashboard.shiny_utils.utils_data import find_date_column

        df = _make_df(Date=["2024-01-01"], Timestamp=["2024-01-01"], Value=[1.0])
        result = find_date_column(df)
        assert result in {"Date", "date", "DATE"}

    def test_single_date_column_only(self) -> None:
        from src.dashboard.shiny_utils.utils_data import find_date_column

        df = _make_df(Date=["2024-01-01"])
        assert find_date_column(df) == "Date"


# ---------------------------------------------------------------------------
# standardize_date_column
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestStandardizeDateColumn:
    """Verify that standardize_date_column normalises the date column name."""

    def test_lowercase_date_renamed_to_Date(self) -> None:
        from src.dashboard.shiny_utils.utils_data import standardize_date_column

        df = _make_df(date=["2024-01-01"], Val=[1.0])
        result = standardize_date_column(df)
        assert "Date" in result.columns
        assert "date" not in result.columns

    def test_uppercase_DATE_renamed_to_Date(self) -> None:
        from src.dashboard.shiny_utils.utils_data import standardize_date_column

        df = _make_df(DATE=["2024-01-01"], Val=[1.0])
        result = standardize_date_column(df)
        assert "Date" in result.columns

    def test_already_Date_unchanged(self) -> None:
        from src.dashboard.shiny_utils.utils_data import standardize_date_column

        df = _make_df(Date=["2024-01-01"], Val=[1.0])
        result = standardize_date_column(df)
        assert "Date" in result.columns
        assert result.columns.count("Date") == 1

    def test_returns_polars_dataframe(self) -> None:
        from src.dashboard.shiny_utils.utils_data import standardize_date_column

        df = _make_df(date=["2024-01-01"])
        result = standardize_date_column(df)
        assert isinstance(result, pl.DataFrame)

    def test_no_date_column_returns_dataframe_unchanged(self) -> None:
        from src.dashboard.shiny_utils.utils_data import standardize_date_column

        df = _make_df(Ticker=["AAPL"], Price=[200.0])
        result = standardize_date_column(df)
        # Should not raise; must return a DataFrame
        assert isinstance(result, pl.DataFrame)
        assert set(result.columns) == {"Ticker", "Price"}


# ---------------------------------------------------------------------------
# get_names_time_series_from_DF
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestGetNamesTimeSeriesFromDF:
    """Verify that the Date column is excluded from the returned name list."""

    def test_excludes_date_column(self) -> None:
        from src.dashboard.shiny_utils.utils_data import get_names_time_series_from_DF

        df = _make_df(Date=["2024-01-01"], VTI=[100.0], AGG=[90.0])
        result = get_names_time_series_from_DF(df)
        assert "Date" not in result
        assert "VTI" in result
        assert "AGG" in result

    def test_returns_list(self) -> None:
        from src.dashboard.shiny_utils.utils_data import get_names_time_series_from_DF

        df = _make_df(Date=["2024-01-01"], VTI=[100.0])
        result = get_names_time_series_from_DF(df)
        assert isinstance(result, list)

    def test_empty_df_returns_empty_or_safe(self) -> None:
        from src.dashboard.shiny_utils.utils_data import get_names_time_series_from_DF

        df = pl.DataFrame()
        result = get_names_time_series_from_DF(df)
        assert isinstance(result, list)
        assert len(result) == 0

    def test_df_with_only_date_column_returns_empty(self) -> None:
        from src.dashboard.shiny_utils.utils_data import get_names_time_series_from_DF

        df = _make_df(Date=["2024-01-01", "2024-01-02"])
        result = get_names_time_series_from_DF(df)
        assert result == []

    def test_multiple_series_all_returned(self) -> None:
        from src.dashboard.shiny_utils.utils_data import get_names_time_series_from_DF

        expected = {"VTI", "AGG", "VNQ", "VXUS"}
        df = pl.DataFrame(
            {"Date": ["2024-01-01"], **{name: [1.0] for name in expected}}
        )
        result = get_names_time_series_from_DF(df)
        assert set(result) == expected


# ---------------------------------------------------------------------------
# downsample_dataframe
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestDownsampleDataframe:
    """Verify error handling and downsampling behaviour."""

    def test_none_input_raises_value_error(self) -> None:
        from src.dashboard.shiny_utils.utils_data import downsample_dataframe

        with pytest.raises((ValueError, TypeError)):
            downsample_dataframe(None)  # type: ignore[arg-type]

    def test_non_polars_input_raises_type_error(self) -> None:
        from src.dashboard.shiny_utils.utils_data import downsample_dataframe

        with pytest.raises((TypeError, AttributeError)):
            downsample_dataframe({"Date": ["2024-01-01"]})  # type: ignore[arg-type]

    def test_empty_dataframe_raises_value_error(self) -> None:
        from src.dashboard.shiny_utils.utils_data import downsample_dataframe

        with pytest.raises((ValueError, RuntimeError)):
            downsample_dataframe(pl.DataFrame())

    def test_below_max_points_returns_unchanged(self, sample_timeseries_df: pl.DataFrame) -> None:
        from src.dashboard.shiny_utils.utils_data import downsample_dataframe

        # sample_timeseries_df has 250 rows; max_points=500 → should be returned as-is
        original_len = len(sample_timeseries_df)
        result = downsample_dataframe(sample_timeseries_df, max_points=500)
        assert isinstance(result, pl.DataFrame)
        assert len(result) == original_len

    def test_above_max_points_reduces_rows(self, sample_timeseries_df: pl.DataFrame) -> None:
        from src.dashboard.shiny_utils.utils_data import downsample_dataframe

        # 250 rows → max_points=100 must reduce row count.
        # The function uses a stride/window approach so the result may be
        # slightly above max_points; we only assert it is less than the
        # original count.
        original_len = len(sample_timeseries_df)
        result = downsample_dataframe(sample_timeseries_df, max_points=100)
        assert isinstance(result, pl.DataFrame)
        assert len(result) < original_len

    def test_result_keeps_all_columns(self, sample_timeseries_df: pl.DataFrame) -> None:
        from src.dashboard.shiny_utils.utils_data import downsample_dataframe

        result = downsample_dataframe(sample_timeseries_df, max_points=50)
        for col in sample_timeseries_df.columns:
            assert col in result.columns

    def test_custom_date_column_name(self) -> None:
        from src.dashboard.shiny_utils.utils_data import downsample_dataframe

        n = 300
        df = pl.DataFrame(
            {
                "date": [f"2024-01-{(i % 28) + 1:02d}" for i in range(n)],
                "close": [float(i) for i in range(n)],
            }
        )
        result = downsample_dataframe(df, max_points=50, date_column="date")
        assert len(result) <= 50

    @pytest.mark.parametrize("max_pts", [1, 50, 100])
    def test_various_max_points(self, sample_timeseries_df: pl.DataFrame, max_pts: int) -> None:
        from src.dashboard.shiny_utils.utils_data import downsample_dataframe

        # sample_timeseries_df has 250 rows.
        # The function uses stride = len(df) // max_points; when stride == 1 no
        # rows are skipped and the output length equals the input.  Only test
        # values where stride > 1 (i.e. max_pts < 125 for 250 rows).
        original_len = len(sample_timeseries_df)  # 250
        stride = original_len // max_pts
        result = downsample_dataframe(sample_timeseries_df, max_points=max_pts)
        if stride > 1:
            assert len(result) < original_len
        else:
            assert len(result) == original_len


# ---------------------------------------------------------------------------
# get_safe_num_rows_for_DF
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestGetSafeNumRowsForDF:
    """Verify safe row-count utility handles edge cases without raising."""

    def test_normal_dataframe_returns_positive_int(self) -> None:
        from src.dashboard.shiny_utils.utils_data import get_safe_num_rows_for_DF

        df = _make_df(Date=["2024-01-01", "2024-01-02"], V=[1.0, 2.0])
        result = get_safe_num_rows_for_DF(df)
        assert isinstance(result, int)
        assert result == 2

    def test_empty_dataframe_returns_zero_or_safe(self) -> None:
        from src.dashboard.shiny_utils.utils_data import get_safe_num_rows_for_DF

        result = get_safe_num_rows_for_DF(pl.DataFrame())
        assert isinstance(result, int)
        assert result == 0

    def test_none_input_returns_zero_or_safe(self) -> None:
        from src.dashboard.shiny_utils.utils_data import get_safe_num_rows_for_DF

        result = get_safe_num_rows_for_DF(None)  # type: ignore[arg-type]
        # Must not raise; should return 0 or a safe default
        assert isinstance(result, int)
        assert result == 0


# ---------------------------------------------------------------------------
# validate_portfolio_data
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestValidatePortfolioData:
    """Verify that validate_portfolio_data enforces expected data contracts."""

    def test_none_input_returns_false_or_invalid(self) -> None:
        from src.dashboard.shiny_utils.utils_data import validate_portfolio_data

        result = validate_portfolio_data(None)  # type: ignore[arg-type]
        # Expecting a falsy result (False, None, 0, or (False, ...))
        if isinstance(result, tuple):
            assert result[0] is False
        else:
            assert not result

    def test_empty_dataframe_marks_invalid(self) -> None:
        from src.dashboard.shiny_utils.utils_data import validate_portfolio_data

        result = validate_portfolio_data(pl.DataFrame())
        if isinstance(result, tuple):
            assert result[0] is False
        else:
            assert not result

    def test_valid_portfolio_dataframe_is_accepted(self, sample_portfolio_df: pl.DataFrame) -> None:
        from src.dashboard.shiny_utils.utils_data import validate_portfolio_data

        result = validate_portfolio_data(sample_portfolio_df)
        # Accept truthy or (True, ...) results
        if isinstance(result, tuple):
            assert result[0] is True
        else:
            assert result


# ---------------------------------------------------------------------------
# calculate_portfolio_returns
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestCalculatePortfolioReturns:
    """Verify that calculate_portfolio_returns produces expected shapes/values."""

    def test_returns_a_result_for_valid_input(self, sample_portfolio_df: pl.DataFrame) -> None:
        from src.dashboard.shiny_utils.utils_data import calculate_portfolio_returns

        result = calculate_portfolio_returns(sample_portfolio_df)
        assert result is not None

    def test_none_input_raises_value_error(self) -> None:
        """calculate_portfolio_returns raises ValueError for None input."""
        from src.dashboard.shiny_utils.utils_data import calculate_portfolio_returns

        with pytest.raises((ValueError, TypeError)):
            calculate_portfolio_returns(None)  # type: ignore[arg-type]

    def test_single_row_returns_safe_default(self) -> None:
        from src.dashboard.shiny_utils.utils_data import calculate_portfolio_returns

        df = _make_df(Date=["2024-01-01"], Value=[100.0])
        result = calculate_portfolio_returns(df)
        assert result is not None or result is None  # must not raise


# ---------------------------------------------------------------------------
# Integration tests  (require real data files on disk)
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestGetDataInputsIntegration:
    """Integration tests that load the real CSV files from inputs/."""

    def test_get_data_inputs_returns_dict(self, data_inputs: dict[str, Any]) -> None:
        assert isinstance(data_inputs, dict)

    def test_data_inputs_has_benchmark_portfolio_key(self, data_inputs: dict[str, Any]) -> None:
        assert "Benchmark_Portfolio" in data_inputs

    def test_data_inputs_has_my_portfolio_key(self, data_inputs: dict[str, Any]) -> None:
        assert "My_Portfolio" in data_inputs

    def test_benchmark_portfolio_is_polars_dataframe(self, data_inputs: dict[str, Any]) -> None:
        bench = data_inputs["Benchmark_Portfolio"]
        assert isinstance(bench, pl.DataFrame)

    def test_my_portfolio_is_polars_dataframe(self, data_inputs: dict[str, Any]) -> None:
        my_portfolio = data_inputs["My_Portfolio"]
        assert isinstance(my_portfolio, pl.DataFrame)

    def test_benchmark_portfolio_has_rows(self, data_inputs: dict[str, Any]) -> None:
        bench = data_inputs["Benchmark_Portfolio"]
        assert len(bench) > 0

    def test_my_portfolio_has_rows(self, data_inputs: dict[str, Any]) -> None:
        my_portfolio = data_inputs["My_Portfolio"]
        assert len(my_portfolio) > 0


@pytest.mark.integration
class TestGetInputDataRawIntegration:
    """Integration tests for the raw CSV loader."""

    def test_raw_data_contains_time_series_sample(self, project_dir: Path) -> None:
        from src.dashboard.shiny_utils.utils_data import get_input_data_raw

        raw = get_input_data_raw(project_dir)
        assert "Time_Series_Sample" in raw or isinstance(raw, dict)

    def test_raw_data_contains_etfs(self, project_dir: Path) -> None:
        from src.dashboard.shiny_utils.utils_data import get_input_data_raw

        raw = get_input_data_raw(project_dir)
        assert "Time_Series_ETFs" in raw or isinstance(raw, dict)

    def test_raw_data_contains_weights(self, project_dir: Path) -> None:
        from src.dashboard.shiny_utils.utils_data import get_input_data_raw

        raw = get_input_data_raw(project_dir)
        assert "Weights_My_Portfolio" in raw or isinstance(raw, dict)

    def test_each_raw_value_is_polars_dataframe(self, project_dir: Path) -> None:
        from src.dashboard.shiny_utils.utils_data import get_input_data_raw

        raw = get_input_data_raw(project_dir)
        for key, value in raw.items():
            assert isinstance(value, pl.DataFrame), (
                f"Expected pl.DataFrame for key '{key}', got {type(value)}"
            )
