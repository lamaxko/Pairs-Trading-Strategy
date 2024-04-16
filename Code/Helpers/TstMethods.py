from .FunctHelpers import *
from .PlotHelpers import *
import pandas as pd


def dm_get_signals_pairs(train_data, test_data):
    """Get trading signals for pairs based on the distance method.

    Args:
        train_data (DataFrame): A DataFrame containing the training data.
        test_data (DataFrame): A DataFrame containing the testing data.

    Returns:
        DataFrame: A DataFrame containing the trading signals for the pairs.
    """

    # Create copy of data to avoid modifying original data
    train_data = train_data.copy()
    test_data = test_data.copy()

    # Find ssd and choose 20 pairs with lowest ssd
    train_data_filled = train_data.ffill().bfill() #! Temporary works for now, I know not good practice
    train_data_cum = train_data_filled.pct_change(fill_method=None).cumsum()

    ssd_train = find_ssd(train_data_cum)
    pairs = select_lowest_ssd(ssd_train, train_data_cum)

    # Get stdev from tet and get trading signals
    train_spread_std_dev = find_std_dev_spread(train_data_cum, pairs)
    all_pair_signals = get_trading_signals_dm(test_data, pairs, train_spread_std_dev).dropna(axis=0)

    return all_pair_signals, pairs


def coint_get_signals_pairs(train_data, test_data):
    """Get trading signals for pairs based on the cointegration method.

    Args:
        train_data (DataFrame): A DataFrame containing the training data.
        test_data (DataFrame): A DataFrame containing the testing data.

    Returns:
        DataFrame: A DataFrame containing the trading signals for the pairs.
        list: A list the 20 pairs traded.
    """

    # Create copy of data to avoid modifying original data
    train_data = train_data.copy()
    test_data = test_data.copy()


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

    get_trading_signals_coint(test_data, pairs, train_data)
    all_pair_signals = get_trading_signals_coint(test_data, pairs, train_data)

    return all_pair_signals, pairs

def copula_get_signals(train_data, test_data):
    """Get trading signals for pairs based on the copula method.

    Args:
        train_data (DataFrame): A DataFrame containing the training data.
        test_data (DataFrame): A DataFrame containing the testing data.

    Returns:
        DataFrame: A DataFrame containing the trading signals for the pairs.
    """
    
    # Create copy of data to avoid modifying original data
    train_data = train_data.copy()
    test_data = test_data.copy()

    # Find ssd and choose 20 pairs with lowest ssd
    train_data_filled = train_data.ffill().bfill() #! Temporary works for now, I know not good practice
    train_data_cum = train_data_filled.pct_change(fill_method=None).cumsum()

    ssd_train = find_ssd(train_data_cum)
    pairs = select_lowest_ssd(ssd_train, train_data_cum)

    raise NotImplementedError("Copula method not implemented yet.")

def calculate_returns(signals, pairs, test_data):
    """Calculate the returns of the trading strategy.

    Args:
        signals (DataFrame): A DataFrame containing the trading signals for the pairs.
        pairs (list): A list of pairs for trading.
        test_data (DataFrame): A DataFrame containing the testing data.

    Returns:
        DataFrame: A DataFrame containing the returns of the trading strategy.
    """
    daily_returns = pd.DataFrame()
    cumulative_returns = pd.DataFrame()
    
    for pair in pairs:
        asset_one, asset_two = pair
        pair_key = f'{asset_one}_{asset_two}'
        
        positions1 = signals[f'{pair_key}_positions1']
        positions2 = signals[f'{pair_key}_positions2']
        
        returns1 = positions1.shift(1) * test_data[asset_one].pct_change()
        returns2 = positions2.shift(1) * test_data[asset_two].pct_change()
        daily_returns[pair_key] = returns1 + returns2
        cumulative_returns[pair_key] = daily_returns[pair_key].cumsum()

        # Of equally weighted portfolio returns in percent
        daily_returns['portfolio'] = daily_returns.sum(axis=1)
        cumulative_returns['portfolio'] = daily_returns['portfolio'].cumsum()


    # Example usage 
    portfoio_return_at_end_of_test = cumulative_returns['portfolio'].iloc[-1]


    return daily_returns, cumulative_returns