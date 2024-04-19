import pandas as pd
import numpy as np

from Methods.DmHelpers import *
from Methods.CointHelpers import *
from Methods.CopulasHelpers import *

from Helpers.DataHelpers import *
from Helpers.PlotHelpers import *
from Helpers.ModuleHelpers import *
from Helpers.BacktestHelpers import *


def equal_weighing_scheme(pairs: list) -> dict: 
    """Weights each pair equally based on the number of pairs.
    
    Args:
        pairs (list): A list of tuples, each containing the names of two stocks.
        
    Returns:
        dict: A dictionary with pair as key and equal weight as value.
    """
    number_of_pairs = len(pairs)
    equal_weight = 1 / number_of_pairs  # Calculate equal weight for each pair
    
    # Create a dictionary with pair as key and equal weight as value
    weights = {f'{pair[0]}_{pair[1]}': equal_weight for pair in pairs}
    
    return weights


def train_test_dates(start_date):
    """Return train and test dates based on the start date. Train is 12 months and test is 6 months.

    Args:
        start_date (datetime): Start date for the training data.

    Returns:
        tuple: A tuple containing the start and end dates for training and testing data.
    """
    train_start = pd.to_datetime(start_date)
    train_end = pd.to_datetime(start_date + pd.DateOffset(months=12))
    test_start = train_end 
    test_end = pd.to_datetime(start_date + pd.DateOffset(months=18))
        
    return train_start, train_end, test_start, test_end

def update_dates(start_date):
    """Increment the start date by 1 month and return the new start and end dates.

    Args:
        start_date (datetime): Start date for the training data.

    Returns:
        tuple: A tuple containing the updated start and end dates.
    """
    start_date = pd.to_datetime(start_date) + pd.DateOffset(months=1)
    end_date = pd.to_datetime(start_date) + pd.DateOffset(months=18)

    return start_date, end_date


def print_backtest_info(i, start_date, end_date):
    """Show the backtest period information.

    Args:
        i (int): The backtest period number.
        start_date (datetime): Start date for the training data.
        end_date (datetime): End date for the testing data.
    """
    print(f'Backtest Period: {i+1}')
    print(f"Start Date: {pd.to_datetime(start_date).strftime('%Y-%m-%d')}", 
          f"End Date: {pd.to_datetime(end_date).strftime('%Y-%m-%d')}")
    print('-------------------------------------------')
    return

def run_rolling_window_backtest(data, start_date, end_date, END_DATE, dm_threshold, coint_threshold, copula_threshold):

    dm_backtest_results = pd.DataFrame()
    coint_backtest_results = pd.DataFrame()
    copula_backtest_results = pd.DataFrame()

    counter = 0

    while end_date < END_DATE:
        # Split data into training and testing
        train_start, train_end, test_start, test_end = train_test_dates(start_date)
        train_data, test_data = train_test_split(data, train_start, train_end, test_start, test_end)
        
        # Run backtests for each strategy
        dm_signals, dm_pairs = dm_get_signals_backtest(train_data, test_data, threshold=dm_threshold) # was 1.5
        dm_signals_test = dm_signals.loc[test_start:test_end]
        coint_signals, coint_pairs = coint_get_signals_backtest(train_data, test_data, threshold=coint_threshold) 
        coint_signals_test = coint_signals.loc[test_start:test_end]
        copula_signals, copula_pairs = copula_get_signals_backtest(train_data, test_data, threshold=copula_threshold)
        copula_signals_test = copula_signals.loc[test_start:test_end]

        # Helper function to process each set of signals
        def process_signals(signals, results_df, label):
            tuples = [(label, col) for col in signals.columns]
            multi_index = pd.MultiIndex.from_tuples(tuples, names=['Period', 'Signal'])
            period_signals = pd.DataFrame(signals.values, index=signals.index, columns=multi_index)
            return pd.concat([results_df, period_signals], axis=1)

        # Format the period label for the big column
        period_label = f"{start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}"

        # Process each method's signals and update the respective DataFrame
        dm_backtest_results = process_signals(dm_signals_test, dm_backtest_results, period_label)
        coint_backtest_results = process_signals(coint_signals_test, coint_backtest_results, period_label)
        copula_backtest_results = process_signals(copula_signals_test, copula_backtest_results, period_label)

        # Update for next period
        print_backtest_info(counter, start_date, end_date)
        start_date, end_date = update_dates(start_date)
        counter += 1

    return dm_backtest_results, coint_backtest_results, copula_backtest_results


def get_daily_returns_backtest(backtesting_results: pd.DataFrame, data: pd.DataFrame) -> pd.DataFrame:
    # Calculate daily returns for the price data
    data_daily_returns = data.pct_change()

    # Retrieve the unique periods from the backtesting results
    period_labels = backtesting_results.columns.levels[0]
    returns_list = []

    for period in period_labels:
        # Loop through each column within the period and calculate the daily returns
        for column in backtesting_results[period].columns:
            if 'positions' in column:
                # Extract the stock symbol from the column name
                position_type = column.split('_')[1]
                stock = column.split('_')[0] if 'positions1' in column else column.split('_')[1]
                
                # Ensure that the stock is in the data_daily_returns to avoid KeyError
                if stock in data_daily_returns.columns:
                    # Calculate daily returns by multiplying the daily returns with the shifted positions
                    # Shift positions to align with trading execution on the next day
                    daily_returns = data_daily_returns[stock] * backtesting_results[period][column].shift(1)
                    daily_returns.name = f'{stock}_{period}_{position_type}'
                    returns_list.append(daily_returns)

    # Combine all Series into one DataFrame using concat to avoid high fragmentation
    if returns_list:
        backtest_daily_returns = pd.concat(returns_list, axis=1)
    else:
        backtest_daily_returns = pd.DataFrame()

    return backtest_daily_returns

def average_daily_returns(backtest_daily_returns: pd.DataFrame) -> pd.Series:
    """
    Calculate the average daily returns across all assets for each date, ignoring NaN values.

    Args:
        backtest_daily_returns (pd.DataFrame): DataFrame containing daily returns for multiple assets.

    Returns:
        pd.Series: A series containing the average daily returns for each date.
    """
    # Calculate the mean of returns across columns for each row, ignoring NaN values
    average_returns = backtest_daily_returns.mean(axis=1, skipna=True)

    return average_returns

def calculate_cumulative_return(average_returns: pd.Series) -> pd.Series:
    """
    Calculate the cumulative return of the portfolio from average daily returns.

    Args:
        average_returns (pd.Series): A series containing the average daily returns for each date.

    Returns:
        pd.Series: A series containing the cumulative return for each date.
    """
    # Calculate cumulative product of (1 + average_returns) to get cumulative returns
    cumulative_returns = (1 + average_returns).cumprod()

    return cumulative_returns

def get_daily_returns_backtest_transaction_costs(backtesting_results: pd.DataFrame, data: pd.DataFrame, transaction_cost_rate=0.0009) -> pd.DataFrame:
    # Calculate daily returns for the price data
    data_daily_returns = data.pct_change()

    # Retrieve the unique periods from the backtesting results
    period_labels = backtesting_results.columns.levels[0]
    returns_list = []

    for period in period_labels:
        # Loop through each column within the period and calculate the daily returns
        for column in backtesting_results[period].columns:
            if 'positions' in column:
                # Extract the stock symbol from the column name
                position_type = column.split('_')[1]
                stock = column.split('_')[0] if 'positions1' in column else column.split('_')[1]
                
                # Ensure that the stock is in the data_daily_returns to avoid KeyError
                if stock in data_daily_returns.columns:
                    # Calculate daily returns by multiplying the daily returns with the shifted positions
                    # Shift positions to align with trading execution on the next day
                    positions = backtesting_results[period][column].shift(1)
                    daily_returns = data_daily_returns[stock] * positions

                    # Identify transitions from 0 to 1 which indicate a new position is taken
                    position_changes = (positions.diff() == 1) & (positions.shift(-1) == 0)
                    
                    # Ensure position_changes aligns with the index of daily_returns
                    position_changes = position_changes.reindex(daily_returns.index, fill_value=False)
                    
                    # Apply transaction cost on the days where a new position is opened
                    daily_returns[position_changes] -= transaction_cost_rate
                    
                    daily_returns.name = f'{stock}_{period}_{position_type}'
                    returns_list.append(daily_returns)

    # Combine all Series into one DataFrame using concat to avoid high fragmentation
    if returns_list:
        backtest_daily_returns = pd.concat(returns_list, axis=1)
    else:
        backtest_daily_returns = pd.DataFrame()

    return backtest_daily_returns
