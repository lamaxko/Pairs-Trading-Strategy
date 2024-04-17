from Helpers.ModuleHelpers import *
from Methods.DmHelpers import *

import pandas as pd
import numpy as np
import rpy2 as rpy2
import scipy.stats as stats
import rpy2.robjects as ro

from rpy2.robjects.packages import importr
from rpy2.robjects import numpy2ri
from rpy2.rinterface_lib.embedded import RRuntimeError

def fit_distributions(stock_returns):
    """Fit distributions to the stock returns and select the best fit based on AIC and BIC.

    Args:
        stock_returns (pd.Series): Stock returns.


    Returns:
        str: Name of the best fit distribution.
        dict: Parameters of the best fit distribution.
    """
    # List of distributions to check
    distributions = {
        'extreme_value': stats.genextreme,
        'generalized_extreme_value': stats.gumbel_r,
        'logistic': stats.logistic,
        'normal': stats.norm
    }
    
    results = {}
    for dist_name, dist in distributions.items():
        # Fit distribution to the data
        params = dist.fit(stock_returns)
        # Calculate the negative log likelihood
        nll = -np.sum(dist.logpdf(stock_returns, *params))
        # Calculate AIC and BIC
        k = len(params)
        n = len(stock_returns)
        aic = 2 * k + 2 * nll
        bic = k * np.log(n) + 2 * nll
        results[dist_name] = {'params': params, 'aic': aic, 'bic': bic}
    
    # Select the distribution with the lowest AIC and BIC
    best_fit = sorted(results.items(), key=lambda x: (x[1]['aic'], x[1]['bic']))[0]
    return best_fit[0], results[best_fit[0]]['params']


def fit_best_distribution_for_pairs(train_data, pairs):
    """Fit distributions to the returns of the selected pairs.

    Args:
        train_data (pd.DataFrame): Train data.
        pairs (list): List of pairs.

    Returns:
        dict: Best fit distributions for the pairs.
    """
    best_fit_distributions = {}
    for pair in pairs:
        best_fit_distributions[pair] = {}
        for stock in pair:
            stock_returns = train_data[stock].pct_change().dropna()
            best_dist, best_params = fit_distributions(stock_returns)
            best_fit_distributions[pair][stock] = {'distribution': best_dist, 'params': best_params}
    return best_fit_distributions

def transform_to_uniform(train_data, best_fit_distributions):
    """Transform the stock returns to uniform margins using the best fit distributions.

    Args:
        train_data (pd.DataFrame): Train data.
        best_fit_distributions (dict): Best fit distributions for the pairs.

    Raises:
        ValueError: If the distribution is not supported.

    Returns:
        dict: Uniform margins.
    """
    uniform_margins = {}
    for pair, fits in best_fit_distributions.items():
        uniform_margins[pair] = {}
        for stock, fit in fits.items():
            dist_name = fit['distribution']
            params = fit['params']
            
            # Get the appropriate distribution from scipy.stats
            if dist_name == 'extreme_value':
                dist = stats.genextreme
            elif dist_name == 'generalized_extreme_value':
                dist = stats.gumbel_r
            elif dist_name == 'logistic':
                dist = stats.logistic
            elif dist_name == 'normal':
                dist = stats.norm
            else:
                raise ValueError("Distribution not supported")
            
            # Calculate the CDF values for the stock returns, which will be uniform
            stock_returns = train_data[stock].pct_change().dropna()
            cdf_values = dist.cdf(stock_returns, *params)
            uniform_margins[pair][stock] = cdf_values

    return uniform_margins

def fit_best_copulas_to_pairs(uniform_data):
    """Fit the best fitting copulas based on BIC to the transformed data.

    Args:
        uniform_data (dict): Uniform margins.

    Returns:
        dict: Results of fitting copulas to the pairs.
    """

    numpy2ri.activate()

    try:
        copula = importr('copula')
        #print("Copula package loaded successfully.")
    except RRuntimeError:
        print("Installing copula package...")
        ro.r('install.packages("copula", repos="http://cran.r-project.org")')
        copula = importr('copula')


    results = {}
    for pair, data in uniform_data.items():
        u1 = ro.FloatVector(data[pair[0]])
        u2 = ro.FloatVector(data[pair[1]])
        data_matrix = ro.r['cbind'](u1, u2)

        copulas = {
            'Clayton': copula.claytonCopula(),
            'Gumbel': copula.gumbelCopula(),
            'StudentT': copula.tCopula(dim=2)
        }

        copula_fits = {}
        for name, copula_model in copulas.items():
            try:
                fit = copula.fitCopula(copula_model, data_matrix, method="ml")
                logLik = ro.r['logLik'](fit)
                aic = ro.r['AIC'](fit)
                bic = ro.r['BIC'](fit)
                copula_fits[name] = {
                    'logLik': logLik[0],
                    'AIC': aic[0],
                    'BIC': bic[0],
                    'model': fit  # Store the fitted model for later use
                }
            except RRuntimeError as e:
                print(f"Error fitting {name} copula for pair {pair}: {e}")

        # Select the best copula based on BIC
        best_copula = min(copula_fits, key=lambda x: copula_fits[x]['BIC'])
        results[pair] = {
            'Best Copula': best_copula,
            'Fits': copula_fits[best_copula],
            'Fitted Model': copula_fits[best_copula]['model']  # Include the fitted model
        }
    
    return results

def calculate_returns(data):
    """Calculate returns from the data.

    Args:
        data (pd.DataFrame): Data.

    Returns:
        pd.DataFrame: Returns.
    """
    return data.pct_change().fillna(0)

def get_trading_signals_copula(test_data, pairs, copula_results, diff_threshold=0.025, logLik_threshold=50):
    """Generate trading signals based on copula-based mispricing.

    Args:
        test_data (pd.DataFrame): Test data.
        pairs (list): List of pairs.
        copula_results (dict): Results of fitting copulas to the pairs.
        diff_threshold (float): Threshold for the mean difference.
        logLik_threshold (float): Threshold for the copula log likelihood.

    Returns:
        pd.DataFrame: Trading signals.
    """
    normalized_test_data = test_data / test_data.iloc[0]
    all_pair_signals = pd.DataFrame(index=normalized_test_data.index)

    for pair in pairs:
        if pair in copula_results:
            stock1, stock2 = pair
            data = copula_results[pair]
            logLik = data['Fits']['logLik']

            if stock1 in normalized_test_data.columns and stock2 in normalized_test_data.columns:
                pair_signals = pd.DataFrame({
                    stock1: normalized_test_data[stock1],
                    stock2: normalized_test_data[stock2]
                })

                returns = calculate_returns(pair_signals)
                mean_diff = returns[stock1] - returns[stock2]

                # Generate trading signals based on copula log likelihood and mean difference
                signal1 = np.where((mean_diff.abs() > diff_threshold) & (logLik > logLik_threshold),
                                   np.where(mean_diff > 0, 1, -1), 0)
                signal2 = -signal1

                # Calculate positions
                positions1 = np.zeros_like(signal1)
                positions2 = np.zeros_like(signal2)
                positions1[1:] = np.diff(signal1)
                positions2[1:] = np.diff(signal2)

                # Normalize positions to 1 whenever a change occurs
                positions1 = np.where(positions1 != 0, np.sign(positions1), 0)
                positions2 = np.where(positions2 != 0, np.sign(positions2), 0)

                all_pair_signals[f'{stock1}_{stock2}_signal1'] = signal1
                all_pair_signals[f'{stock1}_{stock2}_positions1'] = positions1
                all_pair_signals[f'{stock1}_{stock2}_signal2'] = signal2
                all_pair_signals[f'{stock1}_{stock2}_positions2'] = positions2
            else:
                print(f"Warning: Columns {stock1} or {stock2} not found in test data.")
        else:
            print(f"Warning: No copula data found for pair {pair}.")

    return all_pair_signals


def copula_get_signals_backtest(train_data, test_data):
    
    # Create copy of data to avoid modifying original data
    train_data = train_data.copy()
    test_data = test_data.copy()

    train_data.dropna(axis=1, inplace=True)
    test_data.dropna(axis=1, inplace=True)

    ssd_train = find_ssd(train_data)
    pairs = select_lowest_ssd_pairs(ssd_train, train_data)

    best_fits = fit_best_distribution_for_pairs(train_data, pairs)
    uniform_data = transform_to_uniform(train_data, best_fits)
    copula_results = fit_best_copulas_to_pairs(uniform_data)

    signals = get_trading_signals_copula(test_data, pairs, copula_results) 

    return signals, pairs