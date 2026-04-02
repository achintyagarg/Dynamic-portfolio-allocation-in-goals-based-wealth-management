"""
ETF data retrieval module for fetching historical prices from Yahoo Finance.

This module retrieves historical price data for a selection of ETFs from Yahoo Finance
and saves it to a CSV file in a wide-table format.

The script fetches daily closing prices for the following ETFs:

- IVV: iShares Core S&P 500 ETF
- IJH: iShares Core S&P Mid-Cap ETF
- IWM: iShares Russell 2000 ETF
- EFA: iShares MSCI EAFE ETF
- EEM: iShares MSCI Emerging Markets ETF
- AGG: iShares Core U.S. Aggregate Bond ETF
- SPTL: SPDR Portfolio Long Term Treasury ETF
- HYG: iShares iBoxx $ High Yield Corporate Bond ETF
- SPBO: SPDR Portfolio Corporate Bond ETF
- IYR: iShares U.S. Real Estate ETF
- DBC: Invesco DB Commodity Index Tracking Fund
- GLD: SPDR Gold Shares

Functions
---------
get_etf_data
    Retrieves historical ETF price data from Yahoo Finance.
main
    Main execution function that retrieves data and saves to CSV.

Examples
--------
To fetch ETF data and save to CSV:

.. code-block:: python

    from src.utils.get_data_ETFs import main

    main()
"""

import logging
import time
import traceback

from datetime import datetime
from pathlib import Path

import pandas as pd
import yfinance as yf


# Configure module logger
logger = logging.getLogger(__name__)


def get_etf_data(
    tickers: list[str],
    start_date: str | None = None,
    end_date: str | None = None,
) -> pd.DataFrame:
    """Retrieve historical ETF data from Yahoo Finance.

    This function downloads historical price data for the specified ETF tickers
    from Yahoo Finance. It attempts a bulk download first and falls back to
    individual ticker downloads if needed.

    Parameters
    ----------
    tickers : List[str]
        List of ETF ticker symbols to retrieve.
    start_date : str, optional
        Start date in YYYY-MM-DD format, by default None (gets all available data).
    end_date : str, optional
        End date in YYYY-MM-DD format, by default None (up to today).

    Returns
    -------
    pd.DataFrame
        DataFrame with date index and ticker columns containing closing prices.

    Notes
    -----
    The function downloads closing price data ('Adj Close') for each ticker
    and combines them into a single DataFrame. If the bulk download fails,
    it attempts to download each ticker individually, which can be more reliable
    but slower.

    Examples
    --------
    >>> tickers = ["SPY", "QQQ"]
    >>> df = get_etf_data(tickers=tickers, start_date="2023-01-01", end_date="2023-12-31")
    >>> df.shape[1]  # Should have 2 columns
    2
    """
    logger.info(
        f"Retrieving data for {len(tickers)} ETFs from {start_date} to {end_date or 'present'}...",
    )

    # Create an empty DataFrame to store results
    all_data = pd.DataFrame()

    # Try first with bulk download
    try:
        # Download data for all tickers at once
        data = yf.download(
            tickers=tickers,
            start=start_date,
            end=end_date,
            auto_adjust=True,
            progress=True,
            group_by="ticker",  # Group by ticker to ensure proper column structure
        )

        # Check the structure of the data
        if isinstance(data, pd.DataFrame) and not data.empty:
            # Handle different data structures
            if isinstance(data.columns, pd.MultiIndex):
                # Multi-ticker result with MultiIndex columns
                prices = pd.DataFrame()
                for ticker in tickers:
                    if ticker in data.columns.levels[0]:
                        try:
                            prices[ticker] = data[(ticker, "Adj Close")]
                            logger.debug(f"Successfully downloaded {ticker} data from bulk request")
                        except (KeyError, ValueError) as e:
                            logger.warning(f"Could not extract Adj Close for {ticker}: {e}")

                if not prices.empty:
                    logger.info(
                        f"Bulk download successful: retrieved {len(prices)} rows "
                        f"for {len(prices.columns)} tickers",
                    )
                    return prices
            else:
                # Single ticker result or non-MultiIndex columns
                if "Adj Close" in data.columns:
                    prices = data["Adj Close"]
                    if isinstance(prices, pd.Series):
                        prices = prices.to_frame()
                        prices.columns = [tickers[0]]
                    logger.info(f"Single ticker bulk download successful: {len(prices)} rows")
                    return prices
    except Exception as e:
        logger.warning(f"Bulk download failed: {e}, trying individual downloads...")

    # If bulk download failed or returned no data, try individual downloads
    logger.info("Downloading each ticker individually...")

    for ticker in tickers:
        try:
            logger.info(f"Downloading {ticker}...")
            ticker_obj = yf.Ticker(ticker)
            ticker_data = ticker_obj.history(
                start=start_date,
                end=end_date,
                auto_adjust=True,
            )

            if not ticker_data.empty:
                if "Close" in ticker_data.columns:
                    all_data[ticker] = ticker_data["Close"]
                    logger.debug(
                        f"Successfully downloaded {ticker} data: {len(ticker_data)} rows",
                    )
                else:
                    logger.warning(f"No Close column found for {ticker}")
            else:
                logger.warning(f"No data found for {ticker}")

            # Avoid hitting rate limits
            time.sleep(0.5)

        except Exception as e:
            logger.error(f"Error downloading {ticker}: {e}")

    if not all_data.empty:
        logger.info(
            f"Individual downloads successful: retrieved {len(all_data)} rows "
            f"for {len(all_data.columns)} tickers",
        )
        return all_data

    logger.warning("Could not retrieve any data from Yahoo Finance")
    return pd.DataFrame()


def main() -> None:
    """Retrieve ETF data and save to CSV.

    This function retrieves historical price data for a set of ETFs,
    processes the data into a wide-table format, and saves it to a CSV file.

    Returns
    -------
    None

    Notes
    -----
    The output file is saved at PROJECT_ROOT/inputs/raw/data_ETFs.csv.
    The function will create the directories if they don't exist.

    The data includes daily prices from January 1, 2012 to the present date.
    All prices are adjusted for splits and dividends.

    Examples
    --------
    >>> from src.utils.get_data_ETFs import main
    >>> main()  # Downloads data and saves to CSV
    """
    # Configure logging for main execution
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # Define ETF tickers to retrieve
    etf_tickers = [
        "IVV",  # iShares Core S&P 500 ETF
        "IJH",  # iShares Core S&P Mid-Cap ETF
        "IWM",  # iShares Russell 2000 ETF
        "EFA",  # iShares MSCI EAFE ETF
        "EEM",  # iShares MSCI Emerging Markets ETF
        "AGG",  # iShares Core U.S. Aggregate Bond ETF
        "SPTL",  # SPDR Portfolio Long Term Treasury ETF
        "HYG",  # iShares iBoxx $ High Yield Corporate Bond ETF
        "SPBO",  # SPDR Portfolio Corporate Bond ETF
        "IYR",  # iShares U.S. Real Estate ETF
        "DBC",  # Invesco DB Commodity Index Tracking Fund
        "GLD",  # SPDR Gold Shares
    ]

    # Define date range: January 1, 2012 to present
    start_date = "2012-01-01"
    # No need to specify end_date as it defaults to today

    try:
        # Get script location for absolute paths
        script_path = Path(__file__).resolve()

        # Define project paths explicitly
        project_dir = script_path.parent.parent.parent
        raw_data_dir = project_dir.joinpath("inputs", "raw")
        output_file = raw_data_dir.joinpath("data_ETFs.csv")

        # Log absolute paths for verification
        logger.info(f"Project directory: {project_dir.absolute()}")
        logger.info(f"Raw data directory: {raw_data_dir.absolute()}")
        logger.info(f"Output file: {output_file.absolute()}")

        # Create directory if it doesn't exist
        raw_data_dir.mkdir(parents=True, exist_ok=True)

        # Get ETF data with specified date range
        df = get_etf_data(tickers=etf_tickers, start_date=start_date)

        # Check if we have data
        if df.empty:
            logger.error("No data was retrieved from Yahoo Finance.")
            return

        # Reset index to make date a column
        df = df.reset_index(drop=False)

        # Rename the date column to match requirements
        if "Date" in df.columns:
            df = df.rename(columns={"Date": "date"})
        elif "index" in df.columns and pd.api.types.is_datetime64_any_dtype(df["index"]):
            df = df.rename(columns={"index": "date"})

        # Determine date column name
        date_column = "date" if "date" in df.columns else df.columns[0]

        # Log date range information
        logger.info(f"Data date range: {df[date_column].min()} to {df[date_column].max()}")

        # Save to CSV with explicit argument names
        df.to_csv(path_or_buf=output_file, index=False)

        logger.info(f"Data saved to {output_file.absolute()}")
        logger.info(
            f"Retrieved {len(df)} rows from {df[date_column].min()} to {df[date_column].max()}",
        )
        logger.info(f"Columns: {', '.join(df.columns)}")

        # Log completion time
        logger.info(f"Download completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    except Exception as e:
        logger.error(f"Error occurred: {e}")
        logger.debug(traceback.format_exc())


if __name__ == "__main__":
    main()
