from Helpers.ModuleHelpers import *
from Methods.DmHelpers import *

import pandas as pd
import numpy as np
import rpy2 as rpy2
import scipy.stats as stats
import rpy2.robjects as ro

from scipy.stats import t
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
            stock_returns = train_data[stock].pct_change().fillna(0)
            #stock_returns = train_data[stock].pct_change().dropna()

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

    from rpy2.robjects.packages import importr
    from rpy2.robjects import numpy2ri
    from rpy2.rinterface_lib.embedded import RRuntimeError
    import rpy2.robjects as ro

    numpy2ri.activate()

    try:
        copula = importr('copula')
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
                
                # Common attributes to store
                fit_attributes = {
                    'logLik': logLik[0],
                    'AIC': aic[0],
                    'BIC': bic[0],
                    'model': fit  # Store the fitted model for later use
                }

                # Extracting coefficients using r['coef']
                parameters = ro.r['coef'](fit)
                
                if name in ['Clayton', 'Gumbel']:
                    theta = parameters[0]  #! Check if correct index
                    fit_attributes['theta'] = theta
                elif name == 'StudentT':
                    rho = parameters[0] #! Check if correct index
                    nu = parameters[1] #! Check if correct index
                    fit_attributes['rho'] = rho 
                    fit_attributes['nu'] = nu

                copula_fits[name] = fit_attributes

            except RRuntimeError as e:
                print(f"Error fitting {name} copula for pair {pair}: {e}")

        best_copula = min(copula_fits, key=lambda x: copula_fits[x]['BIC'])
        results[pair] = {
            'Best Copula': best_copula,
            'Fits': copula_fits[best_copula]
        }

    return results


def get_probabilities_studentt(data, pair, copula_results):
    # Get the uniform marginals
    u1 = data[pair[0]]
    u2 = data[pair[1]]
    
    
    # Get the copula parameters rho (correlation) and nu (degrees of freedom)
    rho = copula_results[pair]['Fits']['rho']
    nu = copula_results[pair]['Fits']['nu']
    
    # Transform the uniform marginals to the t-scores
    x1 = t.ppf(u1, df=nu)
    x2 = t.ppf(u2, df=nu)

    # Calculate the numerator of the conditional CDF
    numerator = x1 - rho * x2
    
    # Calculate the denominator of the conditional CDF
    denominator = ((nu + x2**2) / (nu + 1) * (1 - rho**2))**0.5
    
    # Compute the conditional CDF using the Student's t-distribution CDF
    conditional_cdf = t.cdf(numerator / denominator, df=nu + 1)
    
    return conditional_cdf


def get_probabilities_gumbel(data, pair, copula_results):
    # Extract uniform marginals for the pair of assets
    u1 = data[pair[0]]
    u2 = data[pair[1]]

    # Extract the parameter theta from the copula results
    theta = copula_results[pair]['Fits']['theta']

    # Calculate the components of the Gumbel copula function
    # Here, we use np.log to compute the natural log (ln)
    term1 = (-np.log(u1)) ** theta
    term2 = (-np.log(u2)) ** theta
    sum_terms = (term1 + term2) ** (1 / theta)
    
    # Compute C_theta(u1, u2)
    C_theta = np.exp(-sum_terms)

    # Compute the partial derivative of C_theta with respect to u2
    part_derivative = sum_terms ** (1 - theta) * (1 - theta) / theta * term2 ** (theta - 1) / u2

    # Compute h(u1, u2; theta)
    h_u1_u2_theta = C_theta * part_derivative

    return h_u1_u2_theta

def get_probabilities_clayton(data, pair, copula_results):
    # Extract the uniform marginals for the pair of assets
    u1 = data[pair[0]]
    u2 = data[pair[1]]

    # Extract the parameter theta from the copula results
    theta = copula_results[pair]['Fits']['theta']

    # Calculate the survival function for Clayton copula
    h_u1_u2_theta = u2 ** (-theta - 1) * (u1 ** (-theta) + u2 ** (-theta) - 1) ** (-1 / theta - 1)

    return h_u1_u2_theta


def calculate_returns(data):
    """Calculate returns from the data.

    Args:
        data (pd.DataFrame): Data.

    Returns:
        pd.DataFrame: Returns.
    """
    return data.pct_change().fillna(0)


def get_trading_signals_copula(test_data, pairs, copula_results, best_fit_distributions, threshold=0.5):
    results = []
    for stock1, stock2 in pairs:
        uniform_data = transform_to_uniform(test_data, best_fit_distributions)
        pair = (stock1, stock2)

        copula_type = copula_results[pair]['Best Copula']
        uniform_data_pair = uniform_data[pair]

        # Evaluate conditional probabilities using copula
        if copula_type == 'StudentT':
            h1 = get_probabilities_studentt(uniform_data_pair, pair, copula_results)
            h2 = get_probabilities_studentt(uniform_data_pair, pair[::-1], copula_results)
        elif copula_type == 'Gumbel':
            h1 = get_probabilities_gumbel(uniform_data_pair, pair, copula_results)
            h2 = get_probabilities_gumbel(uniform_data_pair, pair[::-1], copula_results)
        elif copula_type == 'Clayton':
            h1 = get_probabilities_clayton(uniform_data_pair, pair, copula_results)
            h2 = get_probabilities_clayton(uniform_data_pair, pair[::-1], copula_results)


       # Compute daily mispricing indices
        m1 = h1 - 0.5
        m2 = h2 - 0.5
        
        # Initialize cumulative mispricing indices
        M1 = np.cumsum(m1)
        M2 = np.cumsum(m2)

        # Convert numpy arrays to pandas Series to use the .diff() method
        signal1 = pd.Series(np.where((M1 > threshold) & (M2 < -threshold), 1, np.where((M1 < -threshold) & (M2 > threshold), -1, 0)), index=test_data.index)
        position1 = signal1.diff()
        
        signal2 = pd.Series(-signal1.values, index=test_data.index)
        position2 = signal2.diff()

        # Initialize a DataFrame for the signals
        signals = pd.DataFrame({
            f'{stock1}_{stock2}_signal1': signal1,
            f'{stock1}_{stock2}_positions1': position1,
            f'{stock1}_{stock2}_signal2': signal2,
            f'{stock1}_{stock2}_positions2': position2,
            f'{stock1}_{stock2}_M1': pd.Series(M1, index=test_data.index),
            f'{stock1}_{stock2}_M2': pd.Series(M2, index=test_data.index),
        })

        results.append(signals)

    # Concatenate all results into a single DataFrame
    all_signals = pd.concat(results, axis=1) if results else pd.DataFrame()
    return all_signals


def copula_get_signals_backtest(train_data, test_data, threshold):
    
    # Create copy of data to avoid modifying original data
    train_data = train_data.copy()
    test_data = test_data.copy()

    train_data.dropna(axis=1, inplace=True)
    test_data.dropna(axis=1, inplace=True)

    ssd_train = find_ssd(train_data)
    pairs = select_lowest_ssd_pairs(ssd_train, train_data)

    # Cor conditional probabilities using going boht ways
    flipped_pairs = [(pair[1], pair[0]) for pair in pairs]

    best_fits = fit_best_distribution_for_pairs(train_data, pairs)
    best_fits_flipped_pairs = fit_best_distribution_for_pairs(train_data, flipped_pairs)

    uniform_data = transform_to_uniform(train_data, best_fits)
    uniform_data_flipped_pairs = transform_to_uniform(train_data, best_fits_flipped_pairs)

    # Combine the uniform data for the pairs and the flipped pairs again
    uniform_data = {**uniform_data, **uniform_data_flipped_pairs}

    copula_results = fit_best_copulas_to_pairs(uniform_data)

    signals = get_trading_signals_copula(test_data, pairs, copula_results, best_fits, threshold=threshold)

    return signals, pairs
