"""
Unit tests for the generate_data module.

This module contains tests for the time series data generation functions
in src/utils/generate_data.py.
"""

from pathlib import Path

import numpy as np
import polars as pl
import pytest


@pytest.fixture()
def generate_monthly_timeseries():
    """Fixture to import the generate_monthly_timeseries function."""
    from src.utils.data_utils.generate_data import generate_monthly_timeseries

    return generate_monthly_timeseries


@pytest.fixture()
def generate_scenarios_daily_returns_CMA_Tier_0():
    """Fixture to import the generate_scenarios_daily_returns_CMA_Tier_0 function."""
    from src.utils.data_utils.generate_data import (
        generate_scenarios_daily_returns_CMA_Tier_0,
    )

    return generate_scenarios_daily_returns_CMA_Tier_0


class Test_Generate_Monthly_Timeseries:
    """Tests for the generate_monthly_timeseries function."""

    @pytest.mark.unit()
    def test_returns_polars_dataframe(self, generate_monthly_timeseries):
        """Test that the function returns a Polars DataFrame."""
        result = generate_monthly_timeseries()

        assert isinstance(result, pl.DataFrame)

    @pytest.mark.unit()
    def test_default_date_range(self, generate_monthly_timeseries):
        """Test default date range produces expected number of rows."""
        result = generate_monthly_timeseries()

        # Default range is 2002-01-01 to 2025-03-31 (about 279 months)
        assert result.shape[0] > 200
        assert result.shape[0] < 300

    @pytest.mark.unit()
    def test_custom_date_range(self, generate_monthly_timeseries):
        """Test custom date range produces correct number of rows."""
        result = generate_monthly_timeseries(
            start_date="2020-01-01",
            end_date="2020-12-31",
        )

        # Should have 12 months for year 2020
        assert result.shape[0] == 12

    @pytest.mark.unit()
    def test_column_names(self, generate_monthly_timeseries):
        """Test that DataFrame has expected column names."""
        result = generate_monthly_timeseries()

        expected_columns = ["date", "AA", "BB", "CC", "DD", "EE", "FF", "GG"]
        assert list(result.columns) == expected_columns

    @pytest.mark.unit()
    def test_column_count(self, generate_monthly_timeseries):
        """Test that DataFrame has 8 columns (date + 7 series)."""
        result = generate_monthly_timeseries()

        assert result.shape[1] == 8

    @pytest.mark.unit()
    def test_date_column_type(self, generate_monthly_timeseries):
        """Test that date column is a Date type."""
        result = generate_monthly_timeseries()

        assert result["date"].dtype == pl.Date

    @pytest.mark.unit()
    def test_series_columns_are_numeric(self, generate_monthly_timeseries):
        """Test that all series columns are numeric (Float64)."""
        result = generate_monthly_timeseries()

        series_columns = ["AA", "BB", "CC", "DD", "EE", "FF", "GG"]
        for col in series_columns:
            assert result[col].dtype == pl.Float64

    @pytest.mark.unit()
    def test_no_null_values(self, generate_monthly_timeseries):
        """Test that there are no null values in the DataFrame."""
        result = generate_monthly_timeseries()

        for col in result.columns:
            null_count = result[col].null_count()
            assert null_count == 0, f"Column {col} has {null_count} null values"

    @pytest.mark.unit()
    def test_reproducibility(self, generate_monthly_timeseries):
        """Test that results are reproducible (due to fixed random seed)."""
        result1 = generate_monthly_timeseries(
            start_date="2020-01-01",
            end_date="2020-12-31",
        )
        result2 = generate_monthly_timeseries(
            start_date="2020-01-01",
            end_date="2020-12-31",
        )

        # Compare AA column values (should be identical due to fixed seed)
        assert result1["AA"].to_list() == result2["AA"].to_list()

    @pytest.mark.unit()
    def test_series_aa_upward_trend(self, generate_monthly_timeseries):
        """Test that series AA has an upward trend on average."""
        result = generate_monthly_timeseries(
            start_date="2002-01-01",
            end_date="2025-03-31",
        )

        aa_values = result["AA"].to_list()
        # Compare first 10% average with last 10% average
        n = len(aa_values)
        first_avg = np.mean(aa_values[: n // 10])
        last_avg = np.mean(aa_values[-n // 10 :])

        assert last_avg > first_avg, "Series AA should have upward trend"

    @pytest.mark.unit()
    def test_series_ee_downward_trend(self, generate_monthly_timeseries):
        """Test that series EE has a downward trend on average."""
        result = generate_monthly_timeseries(
            start_date="2002-01-01",
            end_date="2025-03-31",
        )

        ee_values = result["EE"].to_list()
        # Compare first 10% average with last 10% average
        n = len(ee_values)
        first_avg = np.mean(ee_values[: n // 10])
        last_avg = np.mean(ee_values[-n // 10 :])

        assert last_avg < first_avg, "Series EE should have downward trend"

    @pytest.mark.unit()
    def test_series_ff_high_volatility(self, generate_monthly_timeseries):
        """Test that series FF has high volatility (detrended noise component).

        FF is designed with scale=25 noise, which should be visible in
        the month-to-month differences (removes trend effect).
        """
        result = generate_monthly_timeseries()

        # Compare month-to-month differences to isolate noise component
        # This removes the trend effect that would inflate AA's std
        ff_diffs = result["FF"].diff().drop_nulls().std()
        aa_diffs = result["AA"].diff().drop_nulls().std()

        # FF has scale=25 noise, AA has scale=5 - FF diffs should be higher
        assert ff_diffs > aa_diffs, "Series FF should have more volatile changes than AA"

    @pytest.mark.unit()
    def test_series_dd_level_shift(self, generate_monthly_timeseries):
        """Test that series DD has a level shift (higher values in middle section)."""
        result = generate_monthly_timeseries()

        dd_values = result["DD"].to_list()
        n = len(dd_values)

        # The level shift is applied from n//3 to n//2
        before_shift = np.mean(dd_values[: n // 4])
        during_shift = np.mean(dd_values[n // 3 : n // 2])

        assert during_shift > before_shift, "Series DD should have level shift in middle"

    @pytest.mark.unit()
    def test_values_are_rounded(self, generate_monthly_timeseries):
        """Test that values are rounded to 2 decimal places."""
        result = generate_monthly_timeseries(
            start_date="2020-01-01",
            end_date="2020-03-31",
        )

        # Check that all values have at most 2 decimal places
        for col in ["AA", "BB", "CC", "DD", "EE", "FF", "GG"]:
            values = result[col].to_list()
            for val in values:
                rounded = round(val, 2)
                assert abs(val - rounded) < 1e-10, f"Value {val} not rounded to 2 decimals"

    @pytest.mark.unit()
    def test_dates_are_monthly(self, generate_monthly_timeseries):
        """Test that dates are approximately monthly (first of month or close)."""
        result = generate_monthly_timeseries(
            start_date="2020-01-01",
            end_date="2020-12-31",
        )

        dates = result["date"].to_list()

        # Check that dates are in consecutive months
        for i in range(len(dates) - 1):
            current_date = dates[i]
            next_date = dates[i + 1]

            # Month should increment (or wrap around year)
            if current_date.month == 12:
                assert next_date.month == 1
            else:
                assert next_date.month == current_date.month + 1

    @pytest.mark.unit()
    @pytest.mark.parametrize(
        "start,end,expected_months",
        [
            ("2020-01-01", "2020-06-30", 6),
            ("2023-01-01", "2023-12-31", 12),
            ("2020-07-01", "2020-12-31", 6),
        ],
    )
    def test_various_date_ranges(
        self,
        generate_monthly_timeseries,
        start,
        end,
        expected_months,
    ):
        """Test various date ranges produce expected number of rows."""
        result = generate_monthly_timeseries(start_date=start, end_date=end)

        assert result.shape[0] == expected_months

    @pytest.mark.unit()
    def test_series_cc_exponential_growth(self, generate_monthly_timeseries):
        """Test that series CC shows exponential growth pattern."""
        result = generate_monthly_timeseries()

        cc_values = result["CC"].to_list()
        n = len(cc_values)

        # Exponential growth: later values should be much larger
        early_avg = np.mean(cc_values[: n // 4])
        late_avg = np.mean(cc_values[-n // 4 :])

        # The growth should be significant for exponential
        growth_ratio = late_avg / early_avg
        assert growth_ratio > 1.5, "Series CC should show exponential growth"

    @pytest.mark.unit()
    def test_empty_date_range_raises_or_empty(self, generate_monthly_timeseries):
        """Test behavior with invalid or very short date range."""
        # Single day range should produce 0 or 1 rows
        result = generate_monthly_timeseries(
            start_date="2020-01-01",
            end_date="2020-01-01",
        )

        assert result.shape[0] <= 1


class Test_Generate_Data_Main_Function:
    """Tests for the main() function (integration tests)."""

    @pytest.mark.integration()
    @pytest.mark.slow()
    def test_main_creates_output_file(self, tmp_path, monkeypatch):
        """Test that main() creates output file in expected location."""
        import src.utils.data_utils.generate_data as module

        # Redirect __file__ so Path(__file__).parent.parent.parent → tmp_path
        # main() uses 3x .parent: file→dir→dir→project_root
        fake_file = tmp_path / "a" / "b" / "generate_data.py"
        fake_file.parent.mkdir(parents=True, exist_ok=True)
        fake_file.touch()
        monkeypatch.setattr(module, "__file__", str(fake_file))

        module.main()

        output_file = tmp_path / "inputs" / "raw" / "data_timeseries.csv"
        assert output_file.exists()
        # Verify file is not empty
        assert output_file.stat().st_size > 0


class Test_Generate_Scenarios_Daily_Returns_CMA_Tier_0:
    """Tests for the generate_scenarios_daily_returns_CMA_Tier_0 function."""

    @pytest.mark.unit()
    def test_returns_dict_with_expected_keys(
        self,
        generate_scenarios_daily_returns_CMA_Tier_0,
    ):
        """Test that function returns dict with all expected keys."""
        result = generate_scenarios_daily_returns_CMA_Tier_0(
            start_date="2024-01-01",
            end_date="2024-06-30",
            num_scenarios=10,
            random_seed=42,
        )

        assert isinstance(result, dict)
        expected_keys = {
            "Expected Returns",
            "Volatilities",
            "Correlations",
            "Scenarios",
        }
        assert set(result.keys()) == expected_keys

    @pytest.mark.unit()
    def test_df_returns_structure(self, generate_scenarios_daily_returns_CMA_Tier_0):
        """Test that df_returns DataFrame has correct structure."""
        result = generate_scenarios_daily_returns_CMA_Tier_0(
            start_date="2024-01-01",
            end_date="2024-12-31",
            num_scenarios=5,
            random_seed=42,
        )

        df_returns = result["Expected Returns"]
        assert isinstance(df_returns, pl.DataFrame)

        # Should have 2 columns: asset_class + expected_return
        assert df_returns.shape[1] == 2
        expected_columns = ["asset_class", "expected_return"]
        assert list(df_returns.columns) == expected_columns

        # Should have 3 rows (one for each Tier 0 asset)
        assert df_returns.shape[0] == 3

        # Check asset classes are correct
        asset_classes = df_returns["asset_class"].to_list()
        assert set(asset_classes) == {"Equities", "Fixed Income", "Cash"}

    @pytest.mark.unit()
    def test_df_volatilities_structure(
        self,
        generate_scenarios_daily_returns_CMA_Tier_0,
    ):
        """Test that df_volatilities DataFrame has correct structure."""
        result = generate_scenarios_daily_returns_CMA_Tier_0(
            start_date="2024-01-01",
            end_date="2024-12-31",
            num_scenarios=5,
            random_seed=42,
        )

        df_vols = result["Volatilities"]
        assert isinstance(df_vols, pl.DataFrame)

        # Should have 2 columns: asset_class + volatility
        assert df_vols.shape[1] == 2
        expected_columns = ["asset_class", "volatility"]
        assert list(df_vols.columns) == expected_columns

        # Should have 3 rows (one for each Tier 0 asset)
        assert df_vols.shape[0] == 3

        # Check asset classes are correct
        asset_classes = df_vols["asset_class"].to_list()
        assert set(asset_classes) == {"Equities", "Fixed Income", "Cash"}

    @pytest.mark.unit()
    def test_df_correlations_structure(
        self,
        generate_scenarios_daily_returns_CMA_Tier_0,
    ):
        """Test that df_correlations DataFrame has correct structure."""
        result = generate_scenarios_daily_returns_CMA_Tier_0(
            start_date="2024-01-01",
            end_date="2024-12-31",
            num_scenarios=5,
            random_seed=42,
        )

        df_corr = result["Correlations"]
        assert isinstance(df_corr, pl.DataFrame)

        # Should be 3x4 (3 assets rows, 1 asset_class column + 3 asset correlation columns)
        assert df_corr.shape == (3, 4)
        expected_columns = ["asset_class", "Equities", "Fixed Income", "Cash"]
        assert list(df_corr.columns) == expected_columns

        # Diagonal should be 1.0
        # Note: df_corr is in long format where each row is an asset class
        # and columns are correlations with all assets
        # Row 0 (Equities) -> Equities column should be 1.0
        # Row 1 (Fixed Income) -> Fixed Income column should be 1.0
        # Row 2 (Cash) -> Cash column should be 1.0
        equities_row = df_corr.filter(pl.col("asset_class") == "Equities")
        assert equities_row["Equities"][0] == pytest.approx(1.0)

        fi_row = df_corr.filter(pl.col("asset_class") == "Fixed Income")
        assert fi_row["Fixed Income"][0] == pytest.approx(1.0)

        cash_row = df_corr.filter(pl.col("asset_class") == "Cash")
        assert cash_row["Cash"][0] == pytest.approx(1.0)

    @pytest.mark.unit()
    def test_df_scenarios_structure(
        self,
        generate_scenarios_daily_returns_CMA_Tier_0,
    ):
        """Test that df_scenarios DataFrame has correct structure."""
        result = generate_scenarios_daily_returns_CMA_Tier_0(
            start_date="2024-01-01",
            end_date="2024-06-30",
            num_scenarios=10,
            random_seed=42,
        )

        df_scenarios = result["Scenarios"]
        assert isinstance(df_scenarios, pl.DataFrame)

        # Should have: Date column + 3 asset class columns
        # Note: num_scenarios parameter doesn't create multiple scenario columns
        # The function creates one set of returns per asset class
        assert df_scenarios.shape[1] == 4

        # Check Date column exists and is first
        assert df_scenarios.columns[0] == "Date"
        assert df_scenarios["Date"].dtype == pl.Date

        # Check asset class columns exist
        assert "Equities" in df_scenarios.columns
        assert "Fixed Income" in df_scenarios.columns
        assert "Cash" in df_scenarios.columns

    @pytest.mark.unit()
    def test_business_dates_no_weekends(
        self,
        generate_scenarios_daily_returns_CMA_Tier_0,
    ):
        """Test that generated dates are only business days (no weekends)."""
        result = generate_scenarios_daily_returns_CMA_Tier_0(
            start_date="2024-01-01",
            end_date="2024-01-31",
            num_scenarios=5,
            random_seed=42,
        )

        df_scenarios = result["Scenarios"]
        dates = df_scenarios["Date"].to_list()

        # Check that no dates are Saturday (5) or Sunday (6)
        for d in dates:
            weekday = d.weekday()
            assert weekday < 5, f"Date {d} is a weekend (weekday={weekday})"

    @pytest.mark.unit()
    def test_cma_parameters_values(self, generate_scenarios_daily_returns_CMA_Tier_0):
        """Test that CMA parameters match expected values."""
        result = generate_scenarios_daily_returns_CMA_Tier_0(
            start_date="2024-01-01",
            end_date="2024-12-31",
            num_scenarios=5,
            random_seed=42,
        )

        # Check expected returns (as decimals, not percentages)
        df_returns = result["Expected Returns"]
        equities_return = df_returns.filter(pl.col("asset_class") == "Equities")["expected_return"][
            0
        ]
        fi_return = df_returns.filter(pl.col("asset_class") == "Fixed Income")["expected_return"][0]
        cash_return = df_returns.filter(pl.col("asset_class") == "Cash")["expected_return"][0]

        assert equities_return == pytest.approx(0.08)  # 8% as decimal
        assert fi_return == pytest.approx(0.04)  # 4% as decimal
        assert cash_return == pytest.approx(0.02)  # 2% as decimal

        # Check volatilities (as decimals, not percentages)
        df_vols = result["Volatilities"]
        equities_vol = df_vols.filter(pl.col("asset_class") == "Equities")["volatility"][0]
        fi_vol = df_vols.filter(pl.col("asset_class") == "Fixed Income")["volatility"][0]
        cash_vol = df_vols.filter(pl.col("asset_class") == "Cash")["volatility"][0]

        assert equities_vol == pytest.approx(0.16)  # 16% as decimal
        assert fi_vol == pytest.approx(0.05)  # 5% as decimal
        assert cash_vol == pytest.approx(0.01)  # 1% as decimal

        # Check correlations
        df_corr = result["Correlations"]

        # Equities-Fixed Income correlation
        eq_row = df_corr.filter(pl.col("asset_class") == "Equities")
        eq_fi_corr = eq_row["Fixed Income"][0]
        assert eq_fi_corr == pytest.approx(-0.20)

        # Equities-Cash correlation
        eq_cash_corr = eq_row["Cash"][0]
        assert eq_cash_corr == pytest.approx(0.0)

        # Fixed Income-Cash correlation
        fi_row = df_corr.filter(pl.col("asset_class") == "Fixed Income")
        fi_cash_corr = fi_row["Cash"][0]
        assert fi_cash_corr == pytest.approx(0.10)

    @pytest.mark.unit()
    def test_reproducibility_with_seed(
        self,
        generate_scenarios_daily_returns_CMA_Tier_0,
    ):
        """Test that same random seed produces identical results."""
        result1 = generate_scenarios_daily_returns_CMA_Tier_0(
            start_date="2024-01-01",
            end_date="2024-06-30",
            num_scenarios=10,
            random_seed=42,
        )
        result2 = generate_scenarios_daily_returns_CMA_Tier_0(
            start_date="2024-01-01",
            end_date="2024-06-30",
            num_scenarios=10,
            random_seed=42,
        )

        # Check that scenarios are identical
        df1 = result1["Scenarios"]
        df2 = result2["Scenarios"]

        assert df1.shape == df2.shape

        # Compare Equities column
        values1 = df1["Equities"].to_list()
        values2 = df2["Equities"].to_list()

        # Use numpy for element-wise comparison with tolerance
        assert np.allclose(values1, values2, rtol=1e-10)

    @pytest.mark.unit()
    def test_different_seeds_produce_different_results(
        self,
        generate_scenarios_daily_returns_CMA_Tier_0,
    ):
        """Test that different random seeds produce different results."""
        result1 = generate_scenarios_daily_returns_CMA_Tier_0(
            start_date="2024-01-01",
            end_date="2024-06-30",
            num_scenarios=10,
            random_seed=42,
        )
        result2 = generate_scenarios_daily_returns_CMA_Tier_0(
            start_date="2024-01-01",
            end_date="2024-06-30",
            num_scenarios=10,
            random_seed=123,
        )

        # Check that scenarios are different
        df1 = result1["Scenarios"]
        df2 = result2["Scenarios"]

        values1 = df1["Equities"].to_list()
        values2 = df2["Equities"].to_list()

        assert not np.allclose(values1, values2, rtol=1e-5)

    @pytest.mark.unit()
    def test_custom_num_scenarios(self, generate_scenarios_daily_returns_CMA_Tier_0):
        """Test with custom number of scenarios.

        Note: num_scenarios parameter is currently metadata-only and doesn't
        affect the output DataFrame structure. The function generates one
        scenario (set of returns) per asset class.
        """
        num_scenarios = 25
        result = generate_scenarios_daily_returns_CMA_Tier_0(
            start_date="2024-01-01",
            end_date="2024-06-30",
            num_scenarios=num_scenarios,
            random_seed=42,
        )

        df_scenarios = result["Scenarios"]

        # Should have: Date + 3 asset class columns (num_scenarios doesn't multiply columns)
        expected_cols = 4  # Date + Equities + Fixed Income + Cash
        assert df_scenarios.shape[1] == expected_cols

        # Check that all asset class columns exist
        assert "Equities" in df_scenarios.columns
        assert "Fixed Income" in df_scenarios.columns
        assert "Cash" in df_scenarios.columns

    @pytest.mark.unit()
    def test_custom_date_range_short(
        self,
        generate_scenarios_daily_returns_CMA_Tier_0,
    ):
        """Test with short custom date range (1 year)."""
        result = generate_scenarios_daily_returns_CMA_Tier_0(
            start_date="2024-01-01",
            end_date="2024-12-31",
            num_scenarios=5,
            random_seed=42,
        )

        df_scenarios = result["Scenarios"]

        # Should have approximately 252 business days (1 trading year)
        num_days = df_scenarios.shape[0]
        assert 240 <= num_days <= 265, f"Expected ~252 days, got {num_days}"

    @pytest.mark.unit()
    def test_scenario_values_are_numeric(
        self,
        generate_scenarios_daily_returns_CMA_Tier_0,
    ):
        """Test that all scenario values are numeric."""
        result = generate_scenarios_daily_returns_CMA_Tier_0(
            start_date="2024-01-01",
            end_date="2024-06-30",
            num_scenarios=5,
            random_seed=42,
        )

        df_scenarios = result["Scenarios"]

        # Check all columns except Date are Float64
        for col in df_scenarios.columns:
            if col != "Date":
                assert df_scenarios[col].dtype == pl.Float64

    @pytest.mark.unit()
    def test_no_null_values(self, generate_scenarios_daily_returns_CMA_Tier_0):
        """Test that there are no null values in any output."""
        result = generate_scenarios_daily_returns_CMA_Tier_0(
            start_date="2024-01-01",
            end_date="2024-06-30",
            num_scenarios=5,
            random_seed=42,
        )

        # Check all DataFrames for null values
        for key, df in result.items():
            for col in df.columns:
                null_count = df[col].null_count()
                assert null_count == 0, f"{key} - Column {col} has {null_count} null values"

    @pytest.mark.integration()
    @pytest.mark.slow()
    def test_xlsx_file_creation(
        self,
        generate_scenarios_daily_returns_CMA_Tier_0,
        tmp_path,
        monkeypatch,
    ):
        """Test that XLSX file is created in correct location."""
        import src.utils.data_utils.generate_data as module

        # Redirect __file__ so Path(__file__).parent.parent.parent.parent → tmp_path
        # generate_scenarios uses 4x .parent: file→dir→dir→dir→project_root
        fake_file = tmp_path / "a" / "b" / "c" / "generate_data.py"
        fake_file.parent.mkdir(parents=True, exist_ok=True)
        fake_file.touch()
        monkeypatch.setattr(module, "__file__", str(fake_file))

        result = module.generate_scenarios_daily_returns_CMA_Tier_0(
            start_date="2024-01-01",
            end_date="2024-03-31",
            num_scenarios=5,
            random_seed=42,
        )

        output_file = tmp_path / "inputs" / "raw" / "returns_CMA_Tier_0.xlsx"
        assert output_file.exists()
        assert output_file.stat().st_size > 0
        assert isinstance(result, dict)
        assert "Scenarios" in result

    @pytest.mark.integration()
    @pytest.mark.slow()
    def test_integration_with_scenarios_CMA(
        self,
        generate_scenarios_daily_returns_CMA_Tier_0,
    ):
        """Test that generated data structure is compatible with Scenarios_CMA."""
        import tempfile

        import src.utils.data_utils.generate_data as module

        # Redirect XLSX output to a temp directory
        with tempfile.TemporaryDirectory() as tmpdir:
            fake_file = Path(tmpdir) / "a" / "b" / "c" / "generate_data.py"
            fake_file.parent.mkdir(parents=True, exist_ok=True)
            fake_file.touch()
            original_file = module.__file__
            module.__file__ = str(fake_file)
            try:
                result = module.generate_scenarios_daily_returns_CMA_Tier_0(
                    start_date="2024-01-01",
                    end_date="2024-12-31",
                    num_scenarios=10,
                    random_seed=42,
                )
            finally:
                module.__file__ = original_file

        # Verify structure is compatible with Scenarios_CMA expectations
        df_scenarios = result["Scenarios"]
        assert "Date" in df_scenarios.columns
        assert df_scenarios.shape[0] > 0
        assert df_scenarios.shape[1] > 1
        # Must have the 3 asset class columns
        for ac in ["Equities", "Fixed Income", "Cash"]:
            assert ac in df_scenarios.columns

    @pytest.mark.unit()
    @pytest.mark.slow()
    def test_statistical_properties_approximate_cma(
        self,
        generate_scenarios_daily_returns_CMA_Tier_0,
    ):
        """Test that sample statistics approximate CMA parameters (large sample)."""
        # Generate scenarios for one year
        result = generate_scenarios_daily_returns_CMA_Tier_0(
            start_date="2024-01-01",
            end_date="2024-12-31",
            num_scenarios=1000,  # Note: This is metadata, doesn't create multiple scenarios
            random_seed=42,
        )

        df_scenarios = result["Scenarios"]

        # Get Equities column (there's only one, not multiple scenario columns)
        equities_returns = df_scenarios["Equities"].to_list()

        # Calculate mean return
        sample_mean = np.mean(equities_returns)

        # Expected daily return for Equities: 0.08 annual / 252 days ≈ 0.000317
        # Due to lognormal distribution properties, this is approximate
        # We allow wide tolerance since this is a statistical test
        expected_daily_return = 0.08 / 252

        # Check that mean is within reasonable range (allow ±5% absolute deviation)
        assert abs(sample_mean - expected_daily_return) < 0.005, (
            f"Sample mean {sample_mean:.6f} differs from expected {expected_daily_return:.6f} "
            f"by more than tolerance"
        )

    @pytest.mark.unit()
    def test_scenarios_contain_reasonable_values(
        self,
        generate_scenarios_daily_returns_CMA_Tier_0,
    ):
        """Test that scenario values are within reasonable bounds."""
        result = generate_scenarios_daily_returns_CMA_Tier_0(
            start_date="2024-01-01",
            end_date="2024-12-31",
            num_scenarios=10,
            random_seed=42,
        )

        df_scenarios = result["Scenarios"]

        # Daily returns should typically be between -10% and +10%
        # (very conservative bounds for daily equity returns)
        for col in df_scenarios.columns:
            if col != "Date":
                values = df_scenarios[col].to_list()
                assert all(-0.10 <= v <= 0.10 for v in values), (
                    f"Column {col} has values outside reasonable bounds"
                )
