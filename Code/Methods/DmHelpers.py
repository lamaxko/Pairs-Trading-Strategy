from Helpers.ModuleHelpers import *
from Helpers.DataHelpers import *

def find_ssd(data: pd.DataFrame) -> np.ndarray:
    """
    Computes the sum of squared differences (SSD) between the cumulative returns of each pair of stocks in a dataset, 
    useful for analyzing variance among stocks. It generates an upper triangular matrix where each entry (i, j) 
    represents the SSD between stocks i and j.

    Args:
        data (DataFrame): Cumulative returns for each stock, with stocks as columns.

    Returns:
        ndarray: Upper triangular matrix of SSD values among stocks.
    """
    cum_data = calculate_cumulative_returns(data)

    n = len(cum_data.columns)
    ssd = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            ssd[i, j] = np.sum((cum_data.iloc[:, i] - cum_data.iloc[:, j])**2)
            
    return ssd

def select_lowest_ssd_pairs(ssd: np.ndarray, data: pd.DataFrame, num_pairs=20) -> list:
    """
    Identifies and selects the 20 pairs of stocks with the lowest sum of squared differences (SSD) in their 
    cumulative returns from a given SSD matrix. It helps in identifying pairs of stocks that move similarly.

    Args:
        ssd (ndarray): A matrix of SSD values between each pair of stocks.
        data (DataFrame): The dataset with stocks as columns, used to map indices to stock names.

    Returns:
        list: A list of tuples, where each tuple contains the names of a pair of stocks with the lowest SSD.
    """
    
    cum_data = calculate_cumulative_returns(data)

    pairs = []
    ssd_copy = ssd.copy()

    for _ in range(num_pairs):
        min_ssd = np.min(ssd_copy[np.nonzero(ssd_copy)])
        idx = np.where(ssd_copy == min_ssd)
        pair = (cum_data.columns[idx[0][0]], cum_data.columns[idx[1][0]])
        pairs.append(pair)
        ssd_copy[idx[0][0], idx[1][0]] = np.inf
    
    return pairs


def find_dev_spread(data: pd.DataFrame , pairs: list) -> list:
    """
    Calculates the standard deviation of the spread between each pair of stocks in a list. This metric can be 
    crucial for strategies that involve pairs trading, as it helps to quantify the variability of the spread and 
    can be used to generate trading signals based on deviation from a mean.

    Args:
        data (DataFrame): The dataset with cumulative returns for each stock, with stocks as columns.
        pairs (list): A list of tuples, each containing the names of two stocks for which to calculate the spread's 
                      standard deviation.

    Returns:
        list: A list of standard deviation values for the spread of each stock pair.

    Note:
        This function assumes `pairs` contains valid pairs present in `data`.
    """

    data = calculate_cumulative_returns(data)

    std_dev = []
    for pair in pairs:
        stock1, stock2 = pair
        spread = data[stock1] - data[stock2]
        std_dev.append(np.std(spread))

    return std_dev

def get_trading_signals_dm(test_data: pd.DataFrame, pairs: list, dev_train_spread: list, threshold_dev: int) -> pd.DataFrame:
    """
    Generate trading signals for pairs based on the distance method.
    Args:
        test_data (DataFrame): The test dataset containing price data for each stock.
        pairs (list): A list of tuples, each containing the names of two stocks to be used in the pair trading strategy.
        dev_train_spread (list): A list of standard deviation values corresponding to each pair in `pairs`.
        threshold_dev (int): Multiplier for the standard deviation to set divergence thresholds.
    Returns:
        DataFrame: A DataFrame containing trading signals and positions for each pair of stocks.
    """

    # Rescale test_data prices to start at $1 for normalization
    normalized_test_data = test_data / test_data.iloc[0]
    
    all_pair_signals = pd.DataFrame()

    for ((stock1, stock2), std_dev) in zip(pairs, dev_train_spread):
        # Calculate the spread and set divergence threshold
        spread = normalized_test_data[stock2] - normalized_test_data[stock1]
        divergence_threshold = threshold_dev * std_dev
        
        # Generate trading signals based on divergence from the threshold
        signal = np.where(spread.abs() > divergence_threshold, np.where(spread > 0, -1, 1), 0)

        # Create a DataFrame to hold signals and positions for the current pair
        pair_signals = pd.DataFrame({
            'signal1': signal,
            'signal2': -signal,
        }, index=normalized_test_data.index)
        
        # Define positions: Maintain positions as long as signal is active, reset to 0 when signal is 0
        pair_signals['positions1'] = pair_signals['signal1'].where(pair_signals['signal1'] != 0).ffill().fillna(0)
        pair_signals['positions2'] = pair_signals['signal2'].where(pair_signals['signal2'] != 0).ffill().fillna(0)

        # Handle case where signal returns within thresholds
        pair_signals['positions1'] = np.where(pair_signals['signal1'] == 0, 0, pair_signals['positions1'])
        pair_signals['positions2'] = np.where(pair_signals['signal2'] == 0, 0, pair_signals['positions2'])

        # Key for identifying signals in the final DataFrame
        pair_key = f'{stock1}_{stock2}'
        all_pair_signals[f'{pair_key}_signal1'] = pair_signals['signal1']
        all_pair_signals[f'{pair_key}_positions1'] = pair_signals['positions1']
        all_pair_signals[f'{pair_key}_signal2'] = pair_signals['signal2']
        all_pair_signals[f'{pair_key}_positions2'] = pair_signals['positions2']

    return all_pair_signals


def dm_get_signals_backtest(train_data: pd.DataFrame, test_data: pd.DataFrame, threshold: int) -> pd.DataFrame:
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

    ssd_train = find_ssd(train_data)
    pairs = select_lowest_ssd_pairs(ssd_train, train_data)

    # Get stdev from tet and get trading signals
    train_spread_std_dev = find_dev_spread(train_data, pairs)
    all_pair_signals = get_trading_signals_dm(test_data, pairs, train_spread_std_dev, threshold).dropna(axis=0)

    return all_pair_signals, pairs