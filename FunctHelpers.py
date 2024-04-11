import yfinance as yf
import pandas as pd
import numpy as np
import yfinance.shared as shared
import statsmodels.api as sm
import statsmodels.tsa.stattools as ts

def download_data(tickers, start_date, end_date):
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


def train_test_split(data, train_start, train_end, test_start, test_end):
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

def find_ssd(data):
    """
    Computes the sum of squared differences (SSD) between the cumulative returns of each pair of stocks in a dataset, 
    useful for analyzing variance among stocks. It generates an upper triangular matrix where each entry (i, j) 
    represents the SSD between stocks i and j.

    Args:
        data (DataFrame): Cumulative returns for each stock, with stocks as columns.

    Returns:
        ndarray: Upper triangular matrix of SSD values among stocks.
    """
    n = len(data.columns)
    ssd = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            ssd[i, j] = np.sum((data.iloc[:, i] - data.iloc[:, j])**2)
            
    return ssd

def select_lowest_ssd(ssd, data):
    """
    Identifies and selects the 20 pairs of stocks with the lowest sum of squared differences (SSD) in their 
    cumulative returns from a given SSD matrix. It helps in identifying pairs of stocks that move similarly.

    Args:
        ssd (ndarray): A matrix of SSD values between each pair of stocks.
        data (DataFrame): The dataset with stocks as columns, used to map indices to stock names.

    Returns:
        list: A list of tuples, where each tuple contains the names of a pair of stocks with the lowest SSD.
    """
    n = len(data.columns)
    pairs = []
    ssd_copy = ssd.copy()

    for _ in range(20):
        min_ssd = np.min(ssd_copy[np.nonzero(ssd_copy)])
        idx = np.where(ssd_copy == min_ssd)
        pair = (data.columns[idx[0][0]], data.columns[idx[1][0]])
        pairs.append(pair)
        ssd_copy[idx[0][0], idx[1][0]] = np.inf
    
    return pairs


def find_ssd_pair(data, coint_pairs):
    """Find the sum of squared differences (SSD) directly between the cumulative returns of pairs of stocks.

    Args:
        data (DataFrame): The dataset with cumulative returns for each stock, with stocks as columns.
        coint_pairs (list): A list containing tuples of the names of two stocks for each pair to calculate the SSD.

    Returns:
        dict: A dictionary with each pair as keys and their SSD as values.
    """
    ssd_dict = {}

    for pair in coint_pairs:
        if pair[0] in data.columns and pair[1] in data.columns:
            spread = data[pair[0]] - data[pair[1]]
            spread = spread.dropna()
            ssd = np.sum(spread ** 2)  # Direct calculation of SSD

            ssd_dict[pair] = ssd

    return ssd_dict


def select_lowest_ssd_pair(ssd_dict):
    """Selects the 20 pairs with the lowest sum of squared differences (SSD) from a dictionary of SSD values.

    Args:
        ssd_dict (dict): A dictionary containing the pairs and their corresponding SSD values.

    Returns:
        list: A list of 20 pairs with the lowest SSD values.
    """
    ssd_sorted = sorted(ssd_dict.items(), key=lambda x: x[1])
    ssd_sorted = ssd_sorted[:20]
    # choose the 20 highest ssd
    #ssd_sorted = ssd_sorted[-20:]
    pairs = [pair[0] for pair in ssd_sorted]

    return pairs

def find_std_dev_spread(data, pairs):
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
    std_dev = []
    for pair in pairs:
        stock1, stock2 = pair
        spread = data[stock1] - data[stock2]
        std_dev.append(np.std(spread))
    return std_dev

def get_trading_signals_dm(test_data, pairs, train_spread_std_dev):
    """
    Performs the Pair Trading Strategy (PTS) using historical standard deviation to generate trade signals.
    This backtest scales the test data prices to start at $1 for comparison and calculates signals based on the spread
    between each pair of stocks, considering a divergence threshold of 2 historical standard deviations.

    Args:
        test_data (DataFrame): The test dataset containing price data for each stock.
        pairs (list): A list of tuples, each containing the names of two stocks to be used in the pair trading strategy.
        train_std_dev (list): A list of standard deviation values corresponding to each pair in `pairs`.

    Returns:
        DataFrame: A DataFrame containing trading signals and positions for each pair of stocks. Each pair generates four
                   columns: 'signal1' and 'signal2' for the trading signals of the first and second stock, respectively,
                   and 'positions1' and 'positions2' for the corresponding position changes based on the signals.

    Note:
        This function rescales `test_data` prices for each stock to start at $1 for normalization, facilitating
        comparison across different stocks and time periods.
    """
    # Rescale test_data prices to start at $1 for normalization
    normalized_test_data = test_data / test_data.iloc[0]
    
    all_pair_signals = pd.DataFrame()

    for ((stock1, stock2), std_dev) in zip(pairs, train_spread_std_dev):
        # Initialize a DataFrame to hold signals for the current pair
        pair_signals = pd.DataFrame({
            stock1: normalized_test_data[stock1],
            stock2: normalized_test_data[stock2]
        })
        
        # Calculate the spread and set divergence threshold
        spread = pair_signals[stock2] - pair_signals[stock1]
        divergence_threshold = 2 * std_dev  # Defined as 2 historical standard deviations
        
        # Generate trading signals based on divergence from the threshold
        pair_signals['signal1'] = np.select(
            [spread.abs() > divergence_threshold],  # Condition for divergence
            [np.where(spread > 0, -1, 1)],  # Determines the direction of the signal
            default=0  # No signal if condition is not met
        )
        pair_signals['positions1'] = pair_signals['signal1'].diff()

        # Inverse signals for the paired stock
        pair_signals['signal2'] = -pair_signals['signal1']
        pair_signals['positions2'] = pair_signals['signal2'].diff()

        # Key for identifying signals in the final DataFrame
        pair_key = f'{stock1}_{stock2}'
        all_pair_signals[f'{pair_key}_signal1'] = pair_signals['signal1']
        all_pair_signals[f'{pair_key}_positions1'] = pair_signals['positions1']
        all_pair_signals[f'{pair_key}_signal2'] = pair_signals['signal2']
        all_pair_signals[f'{pair_key}_positions2'] = pair_signals['positions2']
        
        # Remove NaN values that result from differencing
        pair_signals.dropna(inplace=True)
    
    return all_pair_signals

def equal_weighing_scheme(pairs):
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

def adf_check_stationarity(data):
    """Check for stationarity using the Augmented Dickey-Fuller test."""
    I0_list = []
    for col in data.columns:
        try:
            result = ts.adfuller(data[col].dropna())  # Ensure to drop NaNs
            if result[1] < 0.1:  # P-value less than 5%
                I0_list.append(col)
        except Exception as e:
            print(f"Error processing column {col}: {e}")
    return I0_list


def find_cointegrated_pairs(data, autolag='AIC'):
    """Find cointegrated pairs in a dataset using the Engle-Granger two-step method.

    Args:
        data (DataFrame): The dataset to check for cointegration.
        autolag (str): The method to use for automatic lag length selection.

    Returns:
        ndarray: A matrix of p-values from the cointegration test.
        list: A list of tuples containing the names of cointegrated pairs.
        list: A list of cointegration constants for each pair. (used for weighting the short positions)
    """
    n = data.shape[1]
    pvalue_matrix = np.ones((n, n))
    keys = data.keys()
    pairs = []
    coint_constants = []

    for i in range(n):
        for j in range(i+1, n):
            result = ts.coint(data[keys[i]], data[keys[j]], autolag=autolag)
            pvalue_matrix[i, j] = result[1]
            if result[1] < 0.05:
                pairs.append((keys[i], keys[j]))
                coint_constants.append(result[0])

    return pvalue_matrix, pairs, coint_constants

def get_spread_pairs(data, pairs):
    """Calculate the spread between each pair of stocks using linear regression.

    Args:
        data (DataFrame): The dataset with cumulative returns for each stock, with stocks as columns.
        pairs (list): A list of tuples, each containing the names of two stocks to calculate the spread.

    Returns:
        list: A list of spread values for each pair of stocks.
    """
    spread_pairs = []
    
    for pair in pairs:
        asset_one = pair[0]
        asset_two = pair[1]
        model = sm.OLS(data[asset_one], sm.add_constant(data[asset_two])).fit()
        spread = model.resid
        spread_pairs.append(spread)
    return spread_pairs

def get_trading_signals_coint(test_data, pairs, train_data):
    """
    Generate trading signals for pairs based on the cointegration method.
    Args:
        test_data (DataFrame): The test dataset containing price data for each stock.
        pairs (list): A list of tuples, each containing the names of two stocks to be used in the pair trading strategy.
        train_data (DataFrame): The training dataset used to calculate the hedge ratio for each pair.
    Returns:
        DataFrame: A DataFrame containing trading signals, z-score, upper and lower limit as well as positions for each pair of stocks.
    """
    results = []  # List to store each pair's signals DataFrame

    for stock1, stock2 in pairs:
        # Fit OLS model on training data to get the hedge ratio
        model = sm.OLS(train_data[stock1], sm.add_constant(train_data[stock2])).fit()

        # Correctly accessing model parameters using labels
        intercept = model.params['const']
        slope = model.params[stock2]

        # Use the hedge ratio to calculate the spread on test data
        spread = test_data[stock2] - intercept - slope * test_data[stock1]
        
        # Function to calculate z-score
        def zscore(series):
            return (series - series.mean()) / series.std()

        signals = pd.DataFrame(index=test_data.index)
        signals['spread'] = spread
        signals['zscore'] = zscore(spread)  # Normalize spread using zscore

        std_shift = 1  # Define standard deviation threshold shift

        # Create the upper and lower thresholds using mean and standard deviation of z-score
        signals['z upper limit'] = signals['zscore'].mean() + std_shift * signals['zscore'].std()
        signals['z lower limit'] = signals['zscore'].mean() - std_shift * signals['zscore'].std()

        # Generate signals based on the z-score exceeding these thresholds
        signals['signal1'] = np.select([signals['zscore'] > signals['z upper limit'], signals['zscore'] < signals['z lower limit']], [1, -1], default=0)
        signals['positions1'] = signals['signal1'].diff()
        signals['signal2'] = -signals['signal1']
        signals['positions2'] = signals['signal2'].diff()

        # Prepare data for concatenation with standardized column names
        pair_key = f'{stock1}_{stock2}'
        data_dict = {
            f'{pair_key}_signal1': signals['signal1'],
            f'{pair_key}_positions1': signals['positions1'],
            f'{pair_key}_signal2': signals['signal2'],
            f'{pair_key}_positions2': signals['positions2'],
            f'{pair_key}_spread': signals['spread'],
            f'{pair_key}_zscore': signals['zscore'],
            f'{pair_key}_z_upper_limit': signals['z upper limit'],
            f'{pair_key}_z_lower_limit': signals['z lower limit']
        }
        results.append(pd.DataFrame(data_dict))

    # Concatenate all results into a single DataFrame
    all_signals = pd.concat(results, axis=1)
    return all_signals