"""Unit tests for scenarios_base module.

This module contains comprehensive tests for the Scenarios_Base abstract base class
and related utilities in src/num_methods/scenarios/scenarios_base.py.
"""

import datetime as dt

import numpy as np
import polars as pl
import pytest


@pytest.fixture()
def Scenarios_Base_class():
    """Fixture to import Scenarios_Base class."""
    from src.num_methods.scenarios.scenarios_base import Scenarios_Base

    return Scenarios_Base


@pytest.fixture()
def Scenario_Data_Type_enum():
    """Fixture to import Scenario_Data_Type enum."""
    from src.num_methods.scenarios.scenarios_base import Scenario_Data_Type

    return Scenario_Data_Type


@pytest.fixture()
def Frequency_Time_Series_enum():
    """Fixture to import Frequency_Time_Series enum."""
    from src.num_methods.scenarios.scenarios_base import Frequency_Time_Series

    return Frequency_Time_Series


@pytest.fixture()
def concrete_scenarios_class(
    Scenarios_Base_class, Scenario_Data_Type_enum, Frequency_Time_Series_enum
):
    """Fixture creating a concrete implementation for testing.

    The concrete class accepts data_type, frequency, and dates as keyword
    arguments and forwards them to Scenarios_Base.__init__ in the correct
    positional order:

        super().__init__(names_components, dates, data_type, frequency, ...)
    """

    class ConcreteScenarios(Scenarios_Base_class):
        """Concrete implementation for testing abstract base class."""

        def __init__(
            self,
            names_components,
            data_type=Scenario_Data_Type_enum.RETURN_ARITHMETIC,
            frequency=Frequency_Time_Series_enum.DAILY,
            dates=None,
            num_days=252,
            num_scenarios=1,
        ):
            """Initialize with num_days and num_scenarios."""
            # If no dates supplied, generate business dates so base class
            # can derive num_dates.
            if dates is None:
                dates = self._generate_business_dates(dt.date(2024, 1, 1), num_days)

            super().__init__(
                names_components=names_components,
                dates=dates,
                data_type=data_type,
                frequency=frequency,
                num_scenarios=num_scenarios,
            )
            self._num_days_requested = num_days

        def generate(self):
            """Simple implementation that generates random returns."""
            dates = self.dates
            num_days = len(dates)
            n_components = self.num_components

            # Generate random returns
            data = np.random.randn(num_days, n_components) * 0.01

            # Build DataFrame with Date + component columns only (no Scenario column)
            scenario_data = {"Date": dates}
            for idx, component in enumerate(self.names_components):
                scenario_data[component] = data[:, idx]

            self.m_df_scenarios = pl.DataFrame(scenario_data)
            return self.m_df_scenarios

    return ConcreteScenarios


@pytest.fixture()
def sample_scenario_data():
    """Fixture providing sample scenario data."""
    dates = [dt.date(2024, 1, i) for i in range(1, 6)]
    return pl.DataFrame(
        {
            "Date": dates,
            "Stock": [0.01, -0.005, 0.02, 0.015, -0.01],
            "Bond": [0.002, 0.001, 0.003, 0.002, 0.001],
        }
    )


class Test_Scenario_Data_Type_Enum:
    """Tests for Scenario_Data_Type enum."""

    @pytest.mark.unit()
    def test_enum_members_exist(self, Scenario_Data_Type_enum):
        """Test that all expected enum members exist."""
        assert hasattr(Scenario_Data_Type_enum, "RETURN_ARITHMETIC")
        assert hasattr(Scenario_Data_Type_enum, "RETURN_LOG")
        assert hasattr(Scenario_Data_Type_enum, "PRICE")
        assert hasattr(Scenario_Data_Type_enum, "INDEX_LEVEL")

    @pytest.mark.unit()
    def test_enum_values(self, Scenario_Data_Type_enum):
        """Test enum values are strings as expected."""
        assert isinstance(Scenario_Data_Type_enum.RETURN_ARITHMETIC.value, str)
        assert "Return" in Scenario_Data_Type_enum.RETURN_ARITHMETIC.value
        assert "Log" in Scenario_Data_Type_enum.RETURN_LOG.value


class Test_Frequency_Time_Series_Enum:
    """Tests for Frequency_Time_Series enum."""

    @pytest.mark.unit()
    def test_enum_members_exist(self, Frequency_Time_Series_enum):
        """Test that frequency enum members exist."""
        assert hasattr(Frequency_Time_Series_enum, "DAILY")
        assert hasattr(Frequency_Time_Series_enum, "WEEKLY")
        assert hasattr(Frequency_Time_Series_enum, "MONTHLY")

    @pytest.mark.unit()
    def test_enum_values_are_integers(self, Frequency_Time_Series_enum):
        """Test that frequency values represent periods per year."""
        # Should be approximate trading days per year
        assert Frequency_Time_Series_enum.DAILY.value == 252
        assert Frequency_Time_Series_enum.MONTHLY.value == 12


class Test_Scenarios_Base_Instantiation:
    """Tests for Scenarios_Base initialization."""

    @pytest.mark.unit()
    def test_cannot_instantiate_abstract_class(self, Scenarios_Base_class):
        """Test that abstract base class cannot be instantiated directly."""
        with pytest.raises(TypeError, match="Can't instantiate abstract class"):
            Scenarios_Base_class(
                names_components=["Stock", "Bond"],
                data_type="Arithmetic Return",
                frequency=252,
            )

    @pytest.mark.unit()
    def test_concrete_class_can_be_instantiated(
        self,
        concrete_scenarios_class,
        Scenario_Data_Type_enum,
        Frequency_Time_Series_enum,
    ):
        """Test that concrete implementation can be instantiated."""
        obj = concrete_scenarios_class(
            names_components=["Stock", "Bond"],
            data_type=Scenario_Data_Type_enum.RETURN_ARITHMETIC,
            frequency=Frequency_Time_Series_enum.DAILY,
            num_days=10,
            num_scenarios=2,
        )
        assert obj is not None
        assert obj.num_components == 2

    @pytest.mark.unit()
    def test_initialization_with_dates(
        self,
        concrete_scenarios_class,
        Scenario_Data_Type_enum,
        Frequency_Time_Series_enum,
    ):
        """Test initialization with explicit dates."""
        dates = [dt.date(2024, 1, i) for i in range(1, 4)]
        obj = concrete_scenarios_class(
            names_components=["Stock"],
            data_type=Scenario_Data_Type_enum.RETURN_ARITHMETIC,
            frequency=Frequency_Time_Series_enum.DAILY,
            dates=dates,
            num_scenarios=1,
        )
        assert obj.num_dates == 3

    @pytest.mark.unit()
    def test_empty_components_raises_error(
        self,
        concrete_scenarios_class,
        Scenario_Data_Type_enum,
        Frequency_Time_Series_enum,
    ):
        """Test that empty component names raise validation error."""
        with pytest.raises(Exception):  # Exception_Validation_Input
            concrete_scenarios_class(
                names_components=[],
                data_type=Scenario_Data_Type_enum.RETURN_ARITHMETIC,
                frequency=Frequency_Time_Series_enum.DAILY,
            )


class Test_Scenarios_Base_Properties:
    """Tests for Scenarios_Base property accessors."""

    @pytest.mark.unit()
    def test_names_components(
        self, concrete_scenarios_class, Scenario_Data_Type_enum, Frequency_Time_Series_enum
    ):
        """Test names_components property returns correct list."""
        obj = concrete_scenarios_class(
            names_components=["Stock", "Bond", "Cash"],
            data_type=Scenario_Data_Type_enum.RETURN_ARITHMETIC,
            frequency=Frequency_Time_Series_enum.DAILY,
        )
        assert obj.names_components == ["Stock", "Bond", "Cash"]

    @pytest.mark.unit()
    def test_num_components(
        self, concrete_scenarios_class, Scenario_Data_Type_enum, Frequency_Time_Series_enum
    ):
        """Test num_components property returns correct count."""
        obj = concrete_scenarios_class(
            names_components=["A", "B", "C", "D"],
            data_type=Scenario_Data_Type_enum.RETURN_ARITHMETIC,
            frequency=Frequency_Time_Series_enum.DAILY,
        )
        assert obj.num_components == 4

    @pytest.mark.unit()
    def test_data_type(
        self, concrete_scenarios_class, Scenario_Data_Type_enum, Frequency_Time_Series_enum
    ):
        """Test data_type property returns enum value."""
        obj = concrete_scenarios_class(
            names_components=["Stock"],
            data_type=Scenario_Data_Type_enum.RETURN_LOG,
            frequency=Frequency_Time_Series_enum.DAILY,
        )
        assert obj.data_type == Scenario_Data_Type_enum.RETURN_LOG

    @pytest.mark.unit()
    def test_frequency(
        self, concrete_scenarios_class, Scenario_Data_Type_enum, Frequency_Time_Series_enum
    ):
        """Test frequency property returns enum value."""
        obj = concrete_scenarios_class(
            names_components=["Stock"],
            data_type=Scenario_Data_Type_enum.RETURN_ARITHMETIC,
            frequency=Frequency_Time_Series_enum.MONTHLY,
        )
        assert obj.frequency == Frequency_Time_Series_enum.MONTHLY


class Test_Scenarios_Base_Data_Access:
    """Tests for data access methods."""

    @pytest.mark.unit()
    def test_df_scenarios_returns_dataframe(
        self,
        concrete_scenarios_class,
        Scenario_Data_Type_enum,
        Frequency_Time_Series_enum,
    ):
        """Test that df_scenarios returns Polars DataFrame after generation."""
        obj = concrete_scenarios_class(
            names_components=["Stock", "Bond"],
            data_type=Scenario_Data_Type_enum.RETURN_ARITHMETIC,
            frequency=Frequency_Time_Series_enum.DAILY,
            num_days=10,
            num_scenarios=1,
        )
        obj.generate()

        result = obj.df_scenarios
        assert isinstance(result, pl.DataFrame)
        assert "Date" in result.columns
        assert "Stock" in result.columns
        assert "Bond" in result.columns

    @pytest.mark.unit()
    def test_get_component_series(
        self,
        concrete_scenarios_class,
        Scenario_Data_Type_enum,
        Frequency_Time_Series_enum,
    ):
        """Test get_component_series method for single component."""
        obj = concrete_scenarios_class(
            names_components=["Stock", "Bond"],
            data_type=Scenario_Data_Type_enum.RETURN_ARITHMETIC,
            frequency=Frequency_Time_Series_enum.DAILY,
            num_days=10,
            num_scenarios=1,
        )
        obj.generate()

        stock_series = obj.get_component_series("Stock")
        assert isinstance(stock_series, pl.Series)
        assert len(stock_series) == 10


class Test_Statistical_Methods:
    """Tests for statistical calculation methods."""

    @pytest.mark.unit()
    def test_calc_correlation_matrix(
        self,
        concrete_scenarios_class,
        Scenario_Data_Type_enum,
        Frequency_Time_Series_enum,
    ):
        """Test correlation matrix calculation."""
        obj = concrete_scenarios_class(
            names_components=["Stock", "Bond"],
            data_type=Scenario_Data_Type_enum.RETURN_ARITHMETIC,
            frequency=Frequency_Time_Series_enum.DAILY,
            num_days=100,
            num_scenarios=1,
        )

        # Generate data
        obj.generate()

        corr_df = obj.calc_correlation_matrix()

        assert isinstance(corr_df, pl.DataFrame)

    @pytest.mark.unit()
    def test_calc_summary_statistics(
        self,
        concrete_scenarios_class,
        Scenario_Data_Type_enum,
        Frequency_Time_Series_enum,
    ):
        """Test summary statistics calculation."""
        obj = concrete_scenarios_class(
            names_components=["Stock"],
            data_type=Scenario_Data_Type_enum.RETURN_ARITHMETIC,
            frequency=Frequency_Time_Series_enum.DAILY,
            num_days=100,
            num_scenarios=1,
        )

        # Generate data
        obj.generate()

        stats_df = obj.calc_summary_statistics()

        assert isinstance(stats_df, pl.DataFrame)


class Test_Conversion_Methods:
    """Tests for data conversion methods (returns to prices)."""

    @pytest.mark.unit()
    def test_convert_returns_to_prices_arithmetic(
        self,
        concrete_scenarios_class,
        Scenario_Data_Type_enum,
        Frequency_Time_Series_enum,
    ):
        """Test converting arithmetic returns to prices."""
        obj = concrete_scenarios_class(
            names_components=["Stock"],
            data_type=Scenario_Data_Type_enum.RETURN_ARITHMETIC,
            frequency=Frequency_Time_Series_enum.DAILY,
            dates=[dt.date(2024, 1, i) for i in range(1, 4)],
            num_scenarios=1,
        )

        # Override generated data with known returns
        returns_data = pl.DataFrame(
            {
                "Date": [dt.date(2024, 1, i) for i in range(1, 4)],
                "Stock": [0.10, 0.05, -0.05],  # 10%, 5%, -5%
            }
        )
        obj.m_df_scenarios = returns_data

        prices = obj.convert_returns_to_prices(initial_prices=[100.0])

        assert isinstance(prices, pl.DataFrame)
        assert "Stock" in prices.columns
        # Check price calculation: 100 * 1.1 = 110, 110 * 1.05 = 115.5, 115.5 * 0.95 = 109.725
        stock_prices = prices["Stock"].to_list()
        assert np.isclose(stock_prices[0], 110.0, rtol=1e-5)
        assert np.isclose(stock_prices[1], 115.5, rtol=1e-5)
        assert np.isclose(stock_prices[2], 109.725, rtol=1e-5)

    @pytest.mark.unit()
    def test_convert_returns_to_prices_log(
        self,
        concrete_scenarios_class,
        Scenario_Data_Type_enum,
        Frequency_Time_Series_enum,
    ):
        """Test converting log returns to prices."""
        obj = concrete_scenarios_class(
            names_components=["Stock"],
            data_type=Scenario_Data_Type_enum.RETURN_LOG,
            frequency=Frequency_Time_Series_enum.DAILY,
            dates=[dt.date(2024, 1, i) for i in range(1, 4)],
            num_scenarios=1,
        )

        # Create log return data
        log_returns_data = pl.DataFrame(
            {
                "Date": [dt.date(2024, 1, i) for i in range(1, 4)],
                "Stock": [0.10, 0.05, -0.05],
            }
        )
        obj.m_df_scenarios = log_returns_data

        prices = obj.convert_returns_to_prices(initial_prices=[100.0])

        assert isinstance(prices, pl.DataFrame)
        # Log returns use exponential: price = 100 * exp(cumulative_log_return)
        stock_prices = prices["Stock"].to_list()
        assert stock_prices[0] > 100.0  # Should increase with positive log return


class Test_CSV_Input_Output:
    """Tests for CSV save/load functionality."""

    @pytest.mark.unit()
    def test_save_scenarios_to_csv(
        self,
        concrete_scenarios_class,
        Scenario_Data_Type_enum,
        Frequency_Time_Series_enum,
        tmp_path,
    ):
        """Test saving scenarios to CSV file."""
        obj = concrete_scenarios_class(
            names_components=["Stock"],
            data_type=Scenario_Data_Type_enum.RETURN_ARITHMETIC,
            frequency=Frequency_Time_Series_enum.DAILY,
            num_days=5,
            num_scenarios=1,
        )

        obj.generate()

        # Save to temporary path
        output_path = tmp_path / "test_scenarios.csv"
        obj.to_csv(str(output_path))

        assert output_path.exists()

        # Load and verify
        loaded = pl.read_csv(output_path)
        assert "Date" in loaded.columns
        assert "Stock" in loaded.columns

    @pytest.mark.unit()
    def test_load_scenarios_from_csv(
        self,
        Scenarios_Base_class,
        Scenario_Data_Type_enum,
        Frequency_Time_Series_enum,
        tmp_path,
    ):
        """Test loading scenarios from CSV file."""
        # Create a sample CSV (no Scenario column - base class does not use it)
        sample_data = pl.DataFrame(
            {
                "Date": ["2024-01-01", "2024-01-02"],
                "Stock": [0.01, -0.005],
                "Bond": [0.002, 0.001],
            }
        )

        csv_path = tmp_path / "sample.csv"
        sample_data.write_csv(csv_path)

        # Load using class method (from_csv, not load_from_csv)
        obj = Scenarios_Base_class.from_csv(
            str(csv_path),
            data_type=Scenario_Data_Type_enum.RETURN_ARITHMETIC,
            frequency=Frequency_Time_Series_enum.DAILY,
        )

        assert obj is not None
        assert obj.num_components == 2
        assert "Stock" in obj.names_components
        assert "Bond" in obj.names_components


class Test_Validation_Methods:
    """Tests for validation functionality."""

    @pytest.mark.unit()
    def test_validate_scenarios_after_generation(
        self,
        concrete_scenarios_class,
        Scenario_Data_Type_enum,
        Frequency_Time_Series_enum,
    ):
        """Test that validation passes after scenario generation."""
        obj = concrete_scenarios_class(
            names_components=["Stock"],
            data_type=Scenario_Data_Type_enum.RETURN_ARITHMETIC,
            frequency=Frequency_Time_Series_enum.DAILY,
            num_days=10,
            num_scenarios=1,
        )

        obj.generate()

        # validate_scenarios should return True on well-formed data
        assert obj.validate_scenarios() is True

    @pytest.mark.unit()
    def test_validate_component_names(
        self,
        concrete_scenarios_class,
        Scenario_Data_Type_enum,
        Frequency_Time_Series_enum,
    ):
        """Test validation of component names."""
        # Should not raise error with valid names
        obj = concrete_scenarios_class(
            names_components=["Stock_US", "Bond_10Y", "Cash_USD"],
            data_type=Scenario_Data_Type_enum.RETURN_ARITHMETIC,
            frequency=Frequency_Time_Series_enum.DAILY,
        )
        assert obj.num_components == 3


class Test_Business_Dates_Generation:
    """Tests for business dates generation."""

    @pytest.mark.unit()
    def test_generate_business_dates_excludes_weekends(
        self,
        concrete_scenarios_class,
        Scenario_Data_Type_enum,
        Frequency_Time_Series_enum,
    ):
        """Test that generated business dates exclude weekends."""
        obj = concrete_scenarios_class(
            names_components=["Stock"],
            data_type=Scenario_Data_Type_enum.RETURN_ARITHMETIC,
            frequency=Frequency_Time_Series_enum.DAILY,
        )

        # Generate 10 business days starting from Monday, Jan 1, 2024
        dates = obj._generate_business_dates(dt.date(2024, 1, 1), 10)

        assert len(dates) == 10

        # Check no weekends (Saturday=5, Sunday=6)
        for date in dates:
            assert date.weekday() < 5  # Monday=0, Friday=4

    @pytest.mark.unit()
    def test_generate_business_dates_sequential(
        self,
        concrete_scenarios_class,
        Scenario_Data_Type_enum,
        Frequency_Time_Series_enum,
    ):
        """Test that generated dates are in sequential order."""
        obj = concrete_scenarios_class(
            names_components=["Stock"],
            data_type=Scenario_Data_Type_enum.RETURN_ARITHMETIC,
            frequency=Frequency_Time_Series_enum.DAILY,
        )

        dates = obj._generate_business_dates(dt.date(2024, 1, 1), 20)

        # Check dates are strictly increasing
        for i in range(len(dates) - 1):
            assert dates[i] < dates[i + 1]


class Test_Edge_Cases:
    """Tests for edge cases and error handling."""

    @pytest.mark.unit()
    def test_single_component_scenarios(
        self,
        concrete_scenarios_class,
        Scenario_Data_Type_enum,
        Frequency_Time_Series_enum,
    ):
        """Test scenarios with single component."""
        obj = concrete_scenarios_class(
            names_components=["Stock"],
            data_type=Scenario_Data_Type_enum.RETURN_ARITHMETIC,
            frequency=Frequency_Time_Series_enum.DAILY,
            num_days=5,
            num_scenarios=1,
        )

        data = obj.generate()

        assert data.shape[0] == 5
        assert "Stock" in data.columns

    @pytest.mark.unit()
    def test_large_number_of_components(
        self,
        concrete_scenarios_class,
        Scenario_Data_Type_enum,
        Frequency_Time_Series_enum,
    ):
        """Test scenarios with large number of components."""
        # 50 components
        components = [f"Asset_{i:02d}" for i in range(50)]

        obj = concrete_scenarios_class(
            names_components=components,
            data_type=Scenario_Data_Type_enum.RETURN_ARITHMETIC,
            frequency=Frequency_Time_Series_enum.DAILY,
        )

        assert obj.num_components == 50

    @pytest.mark.unit()
    def test_invalid_initial_prices_length(
        self,
        concrete_scenarios_class,
        Scenario_Data_Type_enum,
        Frequency_Time_Series_enum,
    ):
        """Test error when initial prices don't match components."""
        obj = concrete_scenarios_class(
            names_components=["Stock", "Bond"],
            data_type=Scenario_Data_Type_enum.RETURN_ARITHMETIC,
            frequency=Frequency_Time_Series_enum.DAILY,
            num_days=5,
            num_scenarios=1,
        )

        obj.generate()

        # Provide wrong number of initial prices
        with pytest.raises(Exception):  # Should raise validation error
            obj.convert_returns_to_prices(initial_prices=[100.0])  # Need 2, gave 1
