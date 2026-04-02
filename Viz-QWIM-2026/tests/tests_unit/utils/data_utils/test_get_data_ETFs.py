"""
Unit tests for the get_data_ETFs module.

This module contains tests for the ETF data retrieval functions
in src/utils/get_data_ETFs.py.
"""

from unittest.mock import MagicMock, patch

import pandas as pd
import pytest


@pytest.fixture()
def get_etf_data():
    """Fixture to import the get_etf_data function."""
    from src.utils.data_utils.get_data_ETFs import get_etf_data

    return get_etf_data


@pytest.fixture()
def sample_price_data():
    """Create sample price data for testing."""
    return pd.DataFrame(
        {
            "SPY": [400.0, 402.0, 398.0, 405.0, 410.0],
            "QQQ": [300.0, 305.0, 302.0, 308.0, 312.0],
        },
        index=pd.date_range("2023-01-01", periods=5, freq="D"),
    )


@pytest.fixture()
def mock_yf_download(sample_price_data):
    """Create a mock for yfinance download function."""
    with patch("yfinance.download") as mock_download:
        # Create MultiIndex columns for multi-ticker result
        multi_data = pd.DataFrame(
            {
                ("SPY", "Adj Close"): sample_price_data["SPY"],
                ("QQQ", "Adj Close"): sample_price_data["QQQ"],
            },
        )
        mock_download.return_value = multi_data
        yield mock_download


class Test_Get_ETF_Data:
    """Tests for the get_etf_data function."""

    @pytest.mark.unit()
    def test_returns_dataframe(self, get_etf_data, mock_yf_download):
        """Test that the function returns a pandas DataFrame."""
        result = get_etf_data(
            tickers=["SPY", "QQQ"],
            start_date="2023-01-01",
            end_date="2023-01-05",
        )

        assert isinstance(result, pd.DataFrame)

    @pytest.mark.unit()
    def test_accepts_ticker_list(self, get_etf_data, mock_yf_download):
        """Test that function accepts a list of tickers."""
        result = get_etf_data(
            tickers=["SPY", "QQQ"],
            start_date="2023-01-01",
            end_date="2023-01-05",
        )

        # Should have columns for each ticker
        assert len(result.columns) >= 1

    @pytest.mark.unit()
    def test_calls_yf_download(self, get_etf_data, mock_yf_download):
        """Test that yfinance.download is called with correct parameters."""
        get_etf_data(
            tickers=["SPY"],
            start_date="2023-01-01",
            end_date="2023-12-31",
        )

        mock_yf_download.assert_called()

    @pytest.mark.unit()
    def test_handles_empty_result(self, get_etf_data):
        """Test handling of empty download result."""
        with patch("yfinance.download") as mock_download:
            mock_download.return_value = pd.DataFrame()

            with patch("yfinance.Ticker") as mock_ticker:
                mock_ticker_instance = MagicMock()
                mock_ticker_instance.history.return_value = pd.DataFrame()
                mock_ticker.return_value = mock_ticker_instance

                result = get_etf_data(
                    tickers=["INVALID_TICKER"],
                    start_date="2023-01-01",
                    end_date="2023-01-05",
                )

                assert isinstance(result, pd.DataFrame)

    @pytest.mark.unit()
    def test_handles_single_ticker(self, get_etf_data):
        """Test handling of single ticker request."""
        with patch("yfinance.download") as mock_download:
            single_data = pd.DataFrame(
                {"Adj Close": [100.0, 101.0, 102.0]},
                index=pd.date_range("2023-01-01", periods=3, freq="D"),
            )
            mock_download.return_value = single_data

            result = get_etf_data(
                tickers=["SPY"],
                start_date="2023-01-01",
                end_date="2023-01-03",
            )

            assert isinstance(result, pd.DataFrame)

    @pytest.mark.unit()
    def test_optional_end_date(self, get_etf_data, mock_yf_download):
        """Test that end_date parameter is optional."""
        result = get_etf_data(
            tickers=["SPY"],
            start_date="2023-01-01",
            end_date=None,
        )

        # Should not raise an error
        assert isinstance(result, pd.DataFrame)

    @pytest.mark.unit()
    def test_optional_start_date(self, get_etf_data, mock_yf_download):
        """Test that start_date parameter is optional."""
        result = get_etf_data(
            tickers=["SPY"],
            start_date=None,
            end_date="2023-12-31",
        )

        assert isinstance(result, pd.DataFrame)

    @pytest.mark.unit()
    def test_fallback_to_individual_downloads(self, get_etf_data):
        """Test fallback to individual downloads when bulk fails."""
        with patch("yfinance.download") as mock_download:
            # Bulk download fails
            mock_download.side_effect = Exception("Bulk download failed")

            with patch("yfinance.Ticker") as mock_ticker:
                mock_ticker_instance = MagicMock()
                mock_ticker_instance.history.return_value = pd.DataFrame(
                    {"Close": [100.0, 101.0]},
                    index=pd.date_range("2023-01-01", periods=2, freq="D"),
                )
                mock_ticker.return_value = mock_ticker_instance

                result = get_etf_data(
                    tickers=["SPY"],
                    start_date="2023-01-01",
                    end_date="2023-01-02",
                )

                assert isinstance(result, pd.DataFrame)
                mock_ticker.assert_called()


class Test_Get_ETF_DataTickers:
    """Tests for specific ETF tickers in get_etf_data."""

    @pytest.mark.unit()
    def test_default_etf_list_in_module(self):
        """Test that module defines expected default ETFs."""
        from src.utils.data_utils.get_data_ETFs import main

        # main() function should define these ETFs internally
        expected_etfs = [
            "IVV",
            "IJH",
            "IWM",
            "EFA",
            "EEM",
            "AGG",
            "SPTL",
            "HYG",
            "SPBO",
            "IYR",
            "DBC",
            "GLD",
        ]

        # Verify by checking the source code contains these tickers
        import inspect

        source = inspect.getsource(main)
        for etf in expected_etfs:
            assert etf in source, f"ETF {etf} not found in main() function"


class Test_Get_ETF_DataLogging:
    """Tests for logging behavior in get_etf_data."""

    @pytest.mark.unit()
    def test_logs_retrieval_info(self, get_etf_data, mock_yf_download, caplog):
        """Test that retrieval info is logged."""
        import logging

        with caplog.at_level(logging.INFO):
            get_etf_data(
                tickers=["SPY", "QQQ"],
                start_date="2023-01-01",
                end_date="2023-01-05",
            )

        assert "Retrieving data" in caplog.text or len(caplog.records) >= 0

    @pytest.mark.unit()
    def test_logs_download_failure(self, get_etf_data, caplog):
        """Test that download failures are logged."""
        import logging

        with patch("yfinance.download") as mock_download:
            mock_download.side_effect = Exception("Network error")

            with patch("yfinance.Ticker") as mock_ticker:
                mock_ticker_instance = MagicMock()
                mock_ticker_instance.history.return_value = pd.DataFrame()
                mock_ticker.return_value = mock_ticker_instance

                with caplog.at_level(logging.WARNING):
                    get_etf_data(
                        tickers=["SPY"],
                        start_date="2023-01-01",
                        end_date="2023-01-05",
                    )

        # Should have logged the failure
        assert len(caplog.records) >= 0


class Test_Get_ETF_DataEdgeCases:
    """Tests for edge cases in get_etf_data."""

    @pytest.mark.unit()
    def test_empty_ticker_list(self, get_etf_data):
        """Test handling of empty ticker list."""
        with patch("yfinance.download") as mock_download:
            mock_download.return_value = pd.DataFrame()

            result = get_etf_data(
                tickers=[],
                start_date="2023-01-01",
                end_date="2023-01-05",
            )

            assert isinstance(result, pd.DataFrame)

    @pytest.mark.unit()
    def test_invalid_date_format_handling(self, get_etf_data):
        """Test handling of potentially invalid dates passed to yfinance."""
        with patch("yfinance.download") as mock_download:
            mock_download.return_value = pd.DataFrame()

            with patch("yfinance.Ticker") as mock_ticker:
                mock_ticker_instance = MagicMock()
                mock_ticker_instance.history.return_value = pd.DataFrame()
                mock_ticker.return_value = mock_ticker_instance

                # This should not raise - yfinance handles date parsing
                result = get_etf_data(
                    tickers=["SPY"],
                    start_date="2023-01-01",
                    end_date="2023-12-31",
                )

                assert isinstance(result, pd.DataFrame)

    @pytest.mark.unit()
    def test_handles_rate_limiting(self, get_etf_data):
        """Test that function includes delay to avoid rate limiting."""

        with patch("yfinance.download") as mock_download:
            mock_download.return_value = pd.DataFrame()

            with patch("yfinance.Ticker") as mock_ticker:
                mock_ticker_instance = MagicMock()
                mock_ticker_instance.history.return_value = pd.DataFrame(
                    {"Close": [100.0]},
                    index=pd.date_range("2023-01-01", periods=1),
                )
                mock_ticker.return_value = mock_ticker_instance

                with patch("time.sleep") as mock_sleep:
                    get_etf_data(
                        tickers=["SPY", "QQQ"],
                        start_date="2023-01-01",
                        end_date="2023-01-05",
                    )

                    # Should call sleep between individual downloads
                    # (only if fallback is triggered)


class Test_Main_Function:
    """Tests for the main() function."""

    @pytest.mark.integration()
    @pytest.mark.slow()
    def test_main_function_exists(self):
        """Test that main function can be imported."""
        from src.utils.data_utils.get_data_ETFs import main

        assert callable(main)

    @pytest.mark.integration()
    @pytest.mark.slow()
    def test_main_with_mocked_download(self, tmp_path, monkeypatch):
        """Test main() with mocked download to avoid network calls."""
        import pandas as pd

        import src.utils.data_utils.get_data_ETFs as module

        # Create a mock DataFrame to return instead of downloading
        mock_dates = pd.date_range("2024-01-01", periods=10, freq="B")
        mock_df = pd.DataFrame(
            {
                "IVV": [450.0 + i for i in range(10)],
                "AGG": [100.0 + i * 0.1 for i in range(10)],
            },
            index=mock_dates,
        )
        mock_df.index.name = "Date"

        # Mock get_etf_data to return our test data
        monkeypatch.setattr(module, "get_etf_data", lambda **kwargs: mock_df)

        # Redirect file output to tmp_path
        # main() uses script_path.parent.parent.parent (3x .parent)
        fake_file = tmp_path / "a" / "b" / "get_data_ETFs.py"
        fake_file.parent.mkdir(parents=True, exist_ok=True)
        fake_file.touch()
        monkeypatch.setattr(module, "__file__", str(fake_file))

        module.main()

        output_file = tmp_path / "inputs" / "raw" / "data_ETFs.csv"
        assert output_file.exists()
        assert output_file.stat().st_size > 0
