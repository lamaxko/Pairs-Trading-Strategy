from Helpers.ModuleHelpers import *
from Helpers.DataHelpers import *

def adf_check_stationarity(data: pd.DataFrame) -> list[str]:
    """Check for stationarity using the Augmented Dickey-Fuller test."""
    I0_list = []
    for col in data.columns:
        try:
            result = ts.adfuller(data[col].dropna(), regression='ct')  # Ensure to drop NaNs
            if result[1] < 0.1:  # P-value less than 5%
                I0_list.append(col)
        except Exception as e:
            print(f"Error processing column {col}: {e}")
    return I0_list


def find_cointegrated_pairs(data: pd.DataFrame, autolag='AIC') -> tuple[np.ndarray, list[tuple[str, str]], list[float]]:
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

def find_ssd_coint_pairs(data: pd.DataFrame, coint_pairs: list[tuple[str, str]]) -> dict[tuple[str, str], float]:
    """Find the sum of squared differences (SSD) directly between the cumulative returns of pairs of stocks.

    Args:
        data (DataFrame): The dataset with cumulative returns for each stock, with stocks as columns.
        coint_pairs (list): A list containing tuples of the names of two stocks for each pair to calculate the SSD.

    Returns:
        dict: A dictionary with each pair as keys and their SSD as values.
    """
    data = calculate_cumulative_returns(data)

    ssd_dict = {}

    for pair in coint_pairs:
        if pair[0] in data.columns and pair[1] in data.columns:
            spread = data[pair[0]] - data[pair[1]]
            spread = spread.dropna()
            if not spread.empty:  # Check if the spread is not empty
                ssd = np.sum(spread ** 2)  # Direct calculation of SSD
                ssd_dict[pair] = ssd
            else:
                print(f"Spread for pair {pair} is empty. Skipping SSD calculation.")
    
    return ssd_dict


def select_lowest_ssd_coint_pair(ssd_dict: dict[tuple[str, str], float]) -> list[tuple[str, str]]:
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



def get_spread_pairs(data: pd.DataFrame, pairs: list[tuple[str, str]]) -> list[pd.Series]:
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


def get_trading_signals_coint(test_data, pairs, train_data, threshold):
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
        if stock1 in train_data.columns and stock2 in train_data.columns:
            # Fit OLS model on training data to get the hedge ratio
            model = sm.OLS(train_data[stock1], sm.add_constant(train_data[stock2])).fit()

            # Correctly accessing model parameters using labels
            intercept = model.params['const']
            slope = model.params[stock2]

            # Use the hedge ratio to calculate the spread on test data
            spread = test_data[stock2] - intercept - slope * test_data[stock1]
            
            # Function to calculate z-score
            def zscore(series):
                if series.std() != 0:
                    return (series - series.mean()) / series.std()
                else:
                    return pd.Series(index=series.index)  # Return an empty series

            signals = pd.DataFrame(index=test_data.index)
            signals['spread'] = spread
            signals['zscore'] = zscore(spread.dropna())  # Normalize spread using zscore and handle NA

            std_shift = threshold  # Define standard deviation threshold shift

            # Check if zscore calculation was successful
            if not signals['zscore'].empty:
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
                    f'{pair_key}_positions1': signals['positions1'], # positions1 is the position for stock1
                    f'{pair_key}_signal2': signals['signal2'],
                    f'{pair_key}_positions2': signals['positions2'],
                    f'{pair_key}_spread': signals['spread'],
                    f'{pair_key}_zscore': signals['zscore'],
                    f'{pair_key}_z_upper_limit': signals['z upper limit'],
                    f'{pair_key}_z_lower_limit': signals['z lower limit']
                }
                results.append(pd.DataFrame(data_dict))
            else:
                print(f"Z-score calculation failed for pair {pair_key} due to zero variance.")
        else:
            print(f"One or both of the stocks {stock1}, {stock2} are not in the training data.")

    # Concatenate all results into a single DataFrame
    if results:
        all_signals = pd.concat(results, axis=1)
    else:
        all_signals = pd.DataFrame()

    return all_signals



def coint_get_signals_backtest(train_data, test_data, threshold):
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

    train_data.dropna(axis=1, inplace=True)
    test_data.dropna(axis=1, inplace=True)

    # Check for stationarity and drop non-stationary columns
    I0_list = adf_check_stationarity(train_data)
    columns_to_drop = [col for col in train_data.columns if col not in I0_list]

    train_data.drop(columns=columns_to_drop, inplace=True, errors='ignore')
    test_data.drop(columns=columns_to_drop, inplace=True, errors='ignore')
    
    # Assuming you have functions to find cointegrated pairs and get signals
    pvalue_matrix, cointegrated_pairs, coint_constants = find_cointegrated_pairs(train_data, autolag='AIC')



    ssd_train = find_ssd_coint_pairs(train_data, cointegrated_pairs)
    pairs = select_lowest_ssd_coint_pair(ssd_train)

    all_pair_signals = get_trading_signals_coint(test_data, pairs, train_data, threshold)

    return all_pair_signals, pairs