from .FunctHelpers import *
from .PlotHelpers import *
import pandas as pd

def dm_get_signals(train_data, test_data):
    """Get trading signals for pairs based on the distance method.

    Args:
        train_data (DataFrame): A DataFrame containing the training data.
        test_data (DataFrame): A DataFrame containing the testing data.

    Returns:
        DataFrame: A DataFrame containing the trading signals for the pairs.
    """
    # Find ssd and choose 20 pairs with lowest ssd
    train_data_filled = train_data.ffill().bfill() #! Temporary works for now, I know not good practice
    train_data_cum = train_data_filled.pct_change(fill_method=None).cumsum()

    ssd_train = find_ssd(train_data_cum)
    pairs = select_lowest_ssd(ssd_train, train_data_cum)

    # Get stdev from tet and get trading signals
    train_spread_std_dev = find_std_dev_spread(train_data_cum, pairs)
    all_pair_signals = get_trading_signals_dm(test_data, pairs, train_spread_std_dev).dropna(axis=0)

    return all_pair_signals

def coint_get_signals(train_data, test_data):
    # First, clean the data by filling or dropping NaNs/Infs
    train_data.replace([np.inf, -np.inf], np.nan, inplace=True)
    test_data.replace([np.inf, -np.inf], np.nan, inplace=True)
    train_data.dropna(axis=1, inplace=True)
    test_data.dropna(axis=1, inplace=True)

    # Check for stationarity and drop non-stationary columns
    I0_list = adf_check_stationarity(train_data)
    columns_to_drop = [col for col in train_data.columns if col not in I0_list]

    train_data.drop(columns=columns_to_drop, inplace=True, errors='ignore')
    test_data.drop(columns=columns_to_drop, inplace=True, errors='ignore')
    
    # Assuming you have functions to find cointegrated pairs and get signals
    pvalue_matrix, cointegrated_pairs, coint_constants = find_cointegrated_pairs(train_data, autolag='AIC')

    train_data_filled = train_data.ffill().bfill() #! Temporary works for now, I know not good practice
    train_data_cum = train_data_filled.pct_change(fill_method=None).cumsum()

    ssd_train = find_ssd_pair(train_data_cum, cointegrated_pairs)
    pairs = select_lowest_ssd_pair(ssd_train)

    all_pair_signals = get_trading_signals_coint(test_data, pairs, train_data)

    return all_pair_signals

def copula_get_signals(train_data, test_data):

    
    return