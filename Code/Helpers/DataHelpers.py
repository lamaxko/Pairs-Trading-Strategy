import yfinance as yf
import yfinance.shared as shared
import pandas as pd


def download_data(tickers: list, start_date: pd.Timestamp, end_date: pd.Timestamp) -> pd.DataFrame:
    """
    Downloads historical stock data for a list of tickers from Yahoo Finance.
    
    Args:
        tickers (list): List of stock tickers to download.
        start_date (datetime): Start date for the historical data.
        end_date (datetime): End date for the historical data.
        
    Returns:
        DataFrame: A DataFrame containing the adjusted closing prices for each stock.
    """
    data = yf.download(tickers, start=start_date, end=end_date)['Adj Close']
    failed_downloads = list(shared._ERRORS.keys())

    for failed in failed_downloads:
        tickers.remove(failed)

    data.drop(failed_downloads, axis=1, inplace=True)

    return data


def train_test_split(data: pd.DataFrame, train_start: pd.Timestamp, train_end: pd.Timestamp, test_start: pd.Timestamp, test_end: pd.Timestamp) -> tuple:
    """
    Splits data into training and testing sets based on predefined date ranges.

    Args:
        data (DataFrame): The dataset to be split.
        train_start (datetime): Start date for the training data.
        train_end (datetime): End date for the training data.
        test_start (datetime): Start date for the testing data.
        test_end (datetime): End date for the testing data.

    Returns:
        tuple: Two DataFrames, where both the training and testing datasets.
    """
    train_data = data.loc[train_start:train_end].copy()
    test_data = data.loc[test_start:test_end].copy()

    return train_data, test_data


def calculate_cumulative_returns(data: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate the cumulative returns of a DataFrame of stock prices.

    Args:
        data (DataFrame): A DataFrame containing stock prices.

    Returns:
        DataFrame: A DataFrame containing the cumulative returns of the input data.
    """
    

    data = data.copy()
    # calculate percentage change without filling NA values automatically
    data = data.pct_change(fill_method=None).fillna(0)
    data = (1 + data).cumprod()

    return data