import pandas as pd
import numpy as np


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