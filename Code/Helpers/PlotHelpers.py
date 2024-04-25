from Helpers.ModuleHelpers import *
from Helpers.DataHelpers import *

def plot_min_ssd_pairs(pairs, data):
    """
    Visualizes the cumulative returns of selected stock pairs to illustrate their movement over time and 
    the spread between them. The function generates a grid of line plots, each showing a pair of stocks,
    making it easier to compare their performance visually.

    Args:
        pairs (list): A list of tuples, each containing the names of two stocks to be plotted.
        data (DataFrame): The dataset with cumulative returns for each stock, with stocks as columns.

    Note:
        The function creates a 5x4 subplot grid, so it expects at least 20 pairs in the `pairs` list.
    """

    data = calculate_cumulative_returns(data)

    fig, axs = plt.subplots(5, 4, figsize=(20, 20))

    for i in range(5):
        for j in range(4):
            ax = axs[i, j]
            pair_index = i*4+j
            stock1, stock2 = pairs[pair_index]
            
            ax.plot(data[stock1], label=stock1, color='red')
            ax.plot(data[stock2], label=stock2, color='blue')
            
            ax.grid(True, which='major', linestyle='--', linewidth=0.5, color='grey')
            ax.minorticks_on()
            ax.grid(True, which='minor', linestyle=':', linewidth=0.5, color='lightgrey')
            ax.set_title(f'Pair {pair_index + 1}', fontsize=16, fontweight='bold', color='navy')
            
            ax.tick_params(axis='x', rotation=45)

            ax.legend()

    plt.tight_layout()
    plt.show()

def plot_spread_dm(pairs: list, data: pd.DataFrame, dev_train_spread: list, threshold_dev: int):
    """
    Visualizes the spread between selected stock pairs over time and includes upper and lower threshold lines indicating 
    when the spread exceeds one historical standard deviation.
    Args:
        pairs (list): A list of tuples, each containing the names of two stocks to be plotted.
        data (DataFrame): The dataset with cumulative returns for each stock, with stocks as columns.
        dev_train_spread (list): List of historical standard deviations for the spreads of each pair.
        threshold_dev (int): The number of standard deviations to set as the threshold.

    Note:
        The function creates a 5x4 subplot grid, so it expects at least 20 pairs in the `pairs` list.
    """
    data = calculate_cumulative_returns(data)

    fig, axs = plt.subplots(5, 4, figsize=(20, 20))

    for i in range(5):
        for j in range(4):
            ax = axs[i, j]
            pair_index = i*4+j
            stock1, stock2 = pairs[pair_index]
            
            # Calculate spread and thresholds
            spread = data[stock1] - data[stock2]
            upper_threshold = dev_train_spread[pair_index] * threshold_dev
            lower_threshold = -upper_threshold
            
            # Plotting the spread and thresholds
            ax.plot(data.index, spread, label='Spread', color='blue')
            ax.axhline(y=upper_threshold, color='red', linestyle='--', label='Upper Threshold')
            ax.axhline(y=lower_threshold, color='green', linestyle='--', label='Lower Threshold')
            
            # Customizing the plot
            ax.grid(True, which='major', linestyle='--', linewidth=0.5, color='grey')
            ax.minorticks_on()
            ax.grid(True, which='minor', linestyle=':', linewidth=0.5, color='lightgrey')
            ax.set_title(f'Pair {pair_index + 1}: {stock1} vs {stock2}', fontsize=16, fontweight='bold', color='navy')
            ax.tick_params(axis='x', rotation=45)
            ax.legend()

    plt.tight_layout()
    plt.show()


def plot_signals_pairs(signals, pairs):
    """Plot the trading signals generated for each pair of stocks.

    Args:
        signals (DataFrame): A DataFrame containing trading signals and positions for each pair of stocks. Each pair generates four
                   columns: 'signal1' and 'signal2' for the trading signals of the first and second stock, respectively,
                   and 'positions1' and 'positions2' for the corresponding position changes based on the signals.
        pairs (list): A list with all pairs of stocks used in the strategy.

    Note:
        The function creates a 5x4 subplot grid, so it expects at least 20 pairs in the `pairs` list.
    """

    fig, axs = plt.subplots(5, 4, figsize=(20, 20))

    for i in range(5):
        for j in range(4):
            ax = axs[i, j]
            pair_index = i*4+j
            stock1, stock2 = pairs[pair_index]
            
            ax.plot(signals[f'{stock1}_{stock2}_signal1'], label=f'{stock1}_{stock2}_signal1', color='blue')
            ax.plot(signals[f'{stock1}_{stock2}_signal2'], label=f'{stock1}_{stock2}_signal2', color='orange')
            
            ax.grid(True, which='major', linestyle='--', linewidth=0.5, color='grey')
            ax.minorticks_on()
            ax.grid(True, which='minor', linestyle=':', linewidth=0.5, color='lightgrey')
            ax.set_title(f'Pair {pair_index + 1}', fontsize=16, fontweight='bold', color='navy')
            ax.tick_params(axis='x', rotation=45)
            #ax.legend()

    plt.tight_layout()
    plt.show()


def plot_signals_prices_pairs(test_data, test_signals, pairs):
    """Plot the trading signals generated for each pair of stocks on top of price development of the pair.

    Args:
        test_data (DataFrame): The test dataset containing the Adj Close price data for each stock.
        signals (DataFrame): A DataFrame containing trading signals and positions for each pair of stocks. Each pair generates four
                   columns: 'signal1' and 'signal2' for the trading signals of the first and second stock, respectively,
                   and 'positions1' and 'positions2' for the corresponding position changes based on the signals.
        pairs (list): A list with all pairs of stocks used in the strategy.

    Note:
        The function creates a 5x4 subplot grid, so it expects at least 20 pairs in the `pairs` list.
    """
    fig, axs = plt.subplots(5, 4, figsize=(20, 25))
    axs = axs.flatten() 
    
    for idx, (asset_one, asset_two) in enumerate(pairs):
        signals = pd.DataFrame()
        signals[asset_one] = test_data[asset_one]
        signals[asset_two] = test_data[asset_two]
                
        pair_key = f'{asset_one}_{asset_two}'
        signals['positions1'] = test_signals[f'{pair_key}_positions1']
        signals['positions2'] = test_signals[f'{pair_key}_positions2']

        ax = axs[idx]
        ax2 = ax.twinx()

        ax.plot(signals.index, signals[asset_one], label=asset_one, color='blue', linewidth=1)
        ax2.plot(signals.index, signals[asset_two], label=asset_two, color='orange', linewidth=1)
                
        ax.plot(signals[asset_one][signals['positions1'] == 1].index, 
                signals[asset_one][signals['positions1'] == 1], 
                '^', markersize=7, color='g', label=f'Long {asset_one}', linewidth=0, alpha=0.7)

        ax.plot(signals[asset_one][signals['positions1'] == -1].index,
                signals[asset_one][signals['positions1'] == -1], 
                'v', markersize=7, color='r', label=f'Short {asset_one}', linewidth=0, alpha=0.7)

        ax2.plot(signals[asset_two][signals['positions2'] == 1].index,
                signals[asset_two][signals['positions2'] == 1], 
                '^', markersize=7, color='g', label=f'Long {asset_two}', linewidth=0, alpha=0.7)

        ax2.plot(signals[asset_two][signals['positions2'] == -1].index,
                signals[asset_two][signals['positions2'] == -1], 
                'v', markersize=7, color='r', label=f'Short {asset_two}', linewidth=0, alpha=0.7)

        ax.set_title(f'Pair {idx + 1}: {asset_one} & {asset_two}', fontsize=16, fontweight='bold', color='navy')
        ax.grid(True, which='major', linestyle='--', linewidth=0.5, color='grey')
        ax.minorticks_on()
        ax.grid(True, which='minor', linestyle=':', linewidth=0.5, color='lightgrey')
        ax.tick_params(axis='x', rotation=45)

        lines, labels = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax2.legend(lines + lines2, labels + labels2, fontsize=8)

    plt.tight_layout()
    plt.show()

def plot_cumulative_payoff_idividual(test_data, test_signals, pairs):
    """Plot the cumulative payoff for the individual pairs.
    
    Args:
        test_data (DataFrame): The test dataset containing the Adj Close price data for each stock.
        test_signals (DataFrame): A DataFrame containing trading signals and positions for each pair of stocks. Each pair generates four
                   columns: 'signal1' and 'signal2' for the trading signals of the first and second stock, respectively,
                   and 'positions1' and 'positions2' for the corresponding position changes based on the signals.
        pairs (list): A list with all pairs of stocks used in the strategy.
        
        Note:
        The function creates a 5x4 subplot grid, so it expects at least 20 pairs in the `pairs` list.
    """
    fig, axs = plt.subplots(5, 4, figsize=(20, 20))
    
    for idx, (pair) in enumerate(pairs):
        asset_one, asset_two = pair
        ax = axs[idx // 4, idx % 4] 
        
        positions1 = test_signals[f'{asset_one}_{asset_two}_positions1']
        positions2 = test_signals[f'{asset_one}_{asset_two}_positions2']
        
        returns1 = positions1.shift(1) * test_data[asset_one].pct_change()
        returns2 = positions2.shift(1) * test_data[asset_two].pct_change()
        cumulative_returns = (returns1.fillna(0) + returns2.fillna(0)).cumsum()

        ax.plot(cumulative_returns.index, cumulative_returns, label='Cumulative Payoff', color='darkcyan')
        
        ax.set_title(f'{asset_one} & {asset_two}', fontsize=16, fontweight='bold', color='navy')
        ax.grid(True, which='major', linestyle='--', linewidth=0.5, color='grey')
        ax.minorticks_on()
        ax.grid(True, which='minor', linestyle=':', linewidth=0.5, color='lightgrey')
        ax.tick_params(axis='x', rotation=45)

        if idx == 0 or idx % 4 == 0: 
            ax.legend()

    plt.tight_layout()
    plt.show()

def plot_cumulative_payoff_portfolio_weighted(test_data, test_signals, pairs, weights):
    """Plot the payoff of the portfolio.
    
    Args:
        test_data (DataFrame): The test dataset containing the Adj Close price data for each stock.
        test_signals (DataFrame): A DataFrame containing trading signals and positions for each pair of stocks. Each pair generates four
                   columns: 'signal1' and 'signal2' for the trading signals of the first and second stock, respectively,
                   and 'positions1' and 'positions2' for the corresponding position changes based on the signals.
        pairs (list): A list with all pairs of stocks used in the strategy.
        weights (dict): A dictionary containing the weights for each pair of stocks in the portfolio.
        
    Note:
        The function creates a 5x4 subplot grid, so it expects at least 20 pairs in the `pairs` list.
    """
    
    weighted_cumulative_returns = pd.DataFrame()
    
    for pair in pairs:
        asset_one, asset_two = pair
        pair_key = f'{asset_one}_{asset_two}'
        
        positions1 = test_signals[f'{pair_key}_positions1']
        positions2 = test_signals[f'{pair_key}_positions2']
        
        returns1 = positions1.shift(1) * test_data[asset_one].pct_change()
        returns2 = positions2.shift(1) * test_data[asset_two].pct_change()
        cumulative_returns = (returns1.fillna(0) + returns2.fillna(0)).cumsum()
        
        weighted_cumulative_returns[pair_key] = cumulative_returns * weights[pair_key]

    portfolio_cumulative_returns = weighted_cumulative_returns.sum(axis=1)

    plt.figure(figsize=(14, 7))
    plt.plot(portfolio_cumulative_returns.index, portfolio_cumulative_returns, label='Weighted Portfolio Cumulative Payoff', color='darkcyan')
    plt.title('Weighted Cumulative Payoff for Portfolio', fontsize=16, fontweight='bold', color='navy')
    plt.xlabel('Date', fontsize=14, fontweight='bold', color='darkgreen')
    plt.ylabel('Cumulative Payoff', fontsize=14, fontweight='bold', color='darkgreen')
    plt.grid(True, which='major', linestyle='--', linewidth=0.5, color='grey')
    plt.minorticks_on()
    plt.grid(True, which='minor', linestyle=':', linewidth=0.5, color='lightgrey')
    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_spread_pairs(spread_pairs):
    """
    Visualizes the spread of selected stock pairs to illustrate their differences over time.
    Each subplot will show the spread for one pair with a red dotted line indicating the mean of the spread,
    centered in the middle of the y-axis.
    The function generates a grid of line plots in a 5x4 format.

    Args:
        spread_pairs (list): A list of series, each containing the spread of a stock pair.

    Note:
        The function creates a 5x4 subplot grid, so it expects at least 20 spread series in the `spread_pairs` list.
    """
    fig, axs = plt.subplots(5, 4, figsize=(20, 20))  # Create a 5x4 grid of plots

    for i in range(5):
        for j in range(4):
            ax = axs[i, j]
            spread_index = i * 4 + j
            if spread_index < len(spread_pairs):
                spread = spread_pairs[spread_index]
                max_abs_spread = np.max(np.abs(spread))  # Find the maximum absolute value for setting y-limits

                ax.plot(spread, label='Spread', color='blue')
                ax.axhline(spread.mean(), color='red', linestyle='--', label='Mean')
                
                ax.set_ylim(-max_abs_spread, max_abs_spread)  # Set y-limits to be symmetrical around the mean line
                
                ax.set_title(f'Pair {spread_index + 1}', fontsize=16, fontweight='bold', color='navy')
                
                ax.grid(True, which='major', linestyle='--', linewidth=0.5, color='grey')
                ax.minorticks_on()
                ax.grid(True, which='minor', linestyle=':', linewidth=0.5, color='lightgrey')
                
                ax.tick_params(axis='x', rotation=45)
                ax.legend()

    plt.tight_layout()
    plt.show()


def plot_spread_dm(pairs: list, data: pd.DataFrame, dev_train_spread: list, threshold_dev: int):
    """
    Visualizes the spread between selected stock pairs over time and includes upper and lower threshold lines indicating 
    when the spread exceeds one historical standard deviation.
    """
    data = calculate_cumulative_returns(data)
    
    # Pre-calculate global min and max for spread consistency
    spreads = [data[pair[0]] - data[pair[1]] for pair in pairs]
    global_spread_max = max(spread.max() for spread in spreads)
    global_spread_min = min(spread.min() for spread in spreads)
    global_limit = max(abs(global_spread_max), abs(global_spread_min))
    
    fig, axs = plt.subplots(5, 4, figsize=(20, 20))

    for i in range(5):
        for j in range(4):
            ax = axs[i, j]
            pair_index = i*4+j
            stock1, stock2 = pairs[pair_index]
            
            # Calculate spread and thresholds
            spread = data[stock1] - data[stock2]
            upper_threshold = dev_train_spread[pair_index] * threshold_dev
            lower_threshold = -upper_threshold
            
            # Plotting the spread and thresholds
            ax.plot(data.index, spread, label='Spread', color='blue')
            ax.axhline(y=upper_threshold, color='red', linestyle='--', label='Upper Threshold')
            ax.axhline(y=lower_threshold, color='green', linestyle='--', label='Lower Threshold')
            
            # Set consistent y-limits based on global max/min spread
            ax.set_ylim(-global_limit, global_limit)
            
            # Customizing the plot
            ax.grid(True, which='major', linestyle='--', linewidth=0.5, color='grey')
            ax.minorticks_on()
            ax.grid(True, which='minor', linestyle=':', linewidth=0.5, color='lightgrey')
            ax.set_title(f'Pair {pair_index + 1}: {stock1} vs {stock2}', fontsize=16, fontweight='bold', color='navy')
            ax.tick_params(axis='x', rotation=45)
            ax.legend()

    plt.tight_layout()
    plt.show()

def plot_cumulative_indices(signals_df, pairs, threshold=0.8, num_rows=5, num_cols=4):
    """
    Visualizes the cumulative mispricing indices M1 and M2 for stock pairs and includes horizontal lines for thresholds.
    """
    fig, axs = plt.subplots(num_rows, num_cols, figsize=(20, 20))

    for i in range(num_rows):
        for j in range(num_cols):
            ax = axs[i, j]
            pair_index = i * num_cols + j
            if pair_index >= len(pairs):
                fig.delaxes(ax)  # Remove excess subplots
                continue
            
            stock1, stock2 = pairs[pair_index]
            col_M1 = f'{stock1}_{stock2}_M1'
            col_M2 = f'{stock1}_{stock2}_M2'
            
            # Plotting cumulative mispricing indices
            ax.plot(signals_df.index, signals_df[col_M1], label=f'{stock1} M1', color='blue')
            ax.plot(signals_df.index, signals_df[col_M2], label=f'{stock2} M2', color='orange')
            
            # Adding threshold lines
            ax.axhline(y=threshold, color='red', linestyle='--', label=f'+{threshold} Threshold', linewidth=1)
            ax.axhline(y=-threshold, color='green', linestyle='--', label=f'-{threshold} Threshold', linewidth=1)

            # Customizing the plot
            ax.grid(True, which='major', linestyle='--', linewidth=0.5, color='grey')
            ax.minorticks_on()
            ax.grid(True, which='minor', linestyle=':', linewidth=0.5, color='lightgrey')
            ax.set_title(f'Pair {pair_index + 1}: {stock1} vs {stock2}', fontsize=16, fontweight='bold', color='navy')
            ax.tick_params(axis='x', rotation=45)
            ax.legend()

    plt.tight_layout()
    plt.show()


def plot_zscores(signals, pairs, threshold_dev=1.0):
    """
    Visualizes the z-scores between selected stock pairs over time and includes horizontal lines indicating 
    when the z-score exceeds the threshold.
    """
    num_plots = len(pairs)
    num_rows, num_cols = 5, 4  # Customize based on your requirement
    fig, axs = plt.subplots(num_rows, num_cols, figsize=(20, 20))
    
    for i, (stock1, stock2) in enumerate(pairs):
        ax = axs[i // num_cols, i % num_cols]
        pair_key = f'{stock1}_{stock2}'
        zscore_col = f'{pair_key}_zscore'
        
        # Calculate thresholds
        upper_threshold = threshold_dev
        lower_threshold = -threshold_dev
        
        # Plotting the z-scores and thresholds
        ax.plot(signals.index, signals[zscore_col], label='Z-Score', color='blue')
        ax.axhline(y=upper_threshold, color='red', linestyle='--', label='Upper Threshold', linewidth=1)
        ax.axhline(y=lower_threshold, color='green', linestyle='--', label='Lower Threshold', linewidth=1)
        
        # Customizing the plot
        ax.grid(True, which='major', linestyle='--', linewidth=0.5, color='grey')
        ax.minorticks_on()
        ax.grid(True, which='minor', linestyle=':', linewidth=0.5, color='lightgrey')
        ax.set_title(f'Pair {i + 1}: {stock1} vs {stock2}', fontsize=16, fontweight='bold', color='navy')
        ax.tick_params(axis='x', rotation=45)
        ax.legend()

        # Remove empty subplots if the number of pairs is less than the number of subplots
        if i >= num_plots - 1:
            for j in range(i + 1, num_rows * num_cols):
                fig.delaxes(axs[j // num_cols, j % num_cols])
            break
    
    plt.tight_layout()
    plt.show()

def plot_single_pair_analysis_dm(pair_index, pairs, data, dev_train_spread, threshold_dev, test_data, test_signals):
    """
    Plots three aspects of a single stock pair side by side: spread and deviations, signals and prices, and cumulative payoff.

    Args:
        pair_index (int): Index of the pair to be plotted.
        pairs (list): List of tuples, each containing the names of two stocks.
        data (DataFrame): DataFrame containing historical data for each stock.
        dev_train_spread (list): List of standard deviations of the spreads for each pair.
        threshold_dev (int): The number of standard deviations to set as the threshold.
        test_data (DataFrame): Test dataset containing the adjusted close prices for each stock.
        test_signals (DataFrame): DataFrame containing trading signals and positions for each pair of stocks.
    """
    if pair_index >= len(pairs):
        raise ValueError("Pair index out of range.")

    stock1, stock2 = pairs[pair_index]
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))

    # Plot 1: Spread and deviations
    data = calculate_cumulative_returns(data)
    spread = data[stock2] - data[stock1]
    upper_threshold = dev_train_spread[pair_index] * threshold_dev
    lower_threshold = -upper_threshold

    ax1.plot(data.index, spread, label='Spread', color='blue')
    ax1.axhline(y=upper_threshold, color='red', linestyle='--', label='Upper Threshold')
    ax1.axhline(y=lower_threshold, color='green', linestyle='--', label='Lower Threshold')
    ax1.set_title(f'Spread & Deviations: {stock1} vs {stock2}', fontsize=15, fontweight='bold', color='navy')
    ax1.grid(True, which='major', linestyle='--', linewidth=0.5, color='grey')
    ax1.minorticks_on()
    ax1.grid(True, which='minor', linestyle=':', linewidth=0.5, color='lightgrey')
    ax1.tick_params(axis='x', rotation=45)
    ax1.legend()

    # Plot 2: Signals and prices
    ax2.plot(test_data.index, test_data[stock1], label=stock1, color='orange')
    ax2.plot(test_data.index, test_data[stock2], label=stock2, color='blue')

    # Correct indexing for signals plotting
    long_signals_stock1 = test_signals[test_signals[f'{stock1}_{stock2}_positions1'] == 1].index
    short_signals_stock1 = test_signals[test_signals[f'{stock1}_{stock2}_positions1'] == -1].index
    long_signals_stock2 = test_signals[test_signals[f'{stock1}_{stock2}_positions2'] == 1].index
    short_signals_stock2 = test_signals[test_signals[f'{stock1}_{stock2}_positions2'] == -1].index

    ax2.plot(long_signals_stock1, 
         test_data.loc[long_signals_stock1, stock1], 
         '^', markersize=10, color='green', label=f'Long {stock1}', alpha=0.7)
    ax2.plot(short_signals_stock1,
         test_data.loc[short_signals_stock1, stock1], 
         'v', markersize=10, color='red', label=f'Short {stock1}', alpha=0.7)
    ax2.plot(long_signals_stock2,
            test_data.loc[long_signals_stock2, stock2], 
            '^', markersize=10, color='green', label=f'Long {stock2}', alpha=0.7)
    ax2.plot(short_signals_stock2,
            test_data.loc[short_signals_stock2, stock2], 
            'v', markersize=10, color='red', label=f'Short {stock2}', alpha=0.7)

    ax2.set_title(f'Signals & Prices: {stock1} & {stock2}', fontsize=15, fontweight='bold', color='navy')
    ax2.grid(True, which='major', linestyle='--', linewidth=0.5, color='grey')
    ax2.minorticks_on()
    ax2.grid(True, which='minor', linestyle=':', linewidth=0.5, color='lightgrey')
    ax2.tick_params(axis='x', rotation=45)
    ax2.legend()

    # Plot 3: Cumulative Payoff
    positions1 = test_signals[f'{stock1}_{stock2}_positions1']
    positions2 = test_signals[f'{stock1}_{stock2}_positions2']
    #positions1 = test_signals[f'{stock1}_{stock2}_signal1']
    #positions2 = test_signals[f'{stock1}_{stock2}_signal2']
    returns1 = positions1.shift(1) * test_data[stock1].pct_change()
    returns2 = positions2.shift(1) * test_data[stock2].pct_change()
    cumulative_returns = (returns1.fillna(0) + returns2.fillna(0)).cumsum()

    ax3.plot(cumulative_returns.index, cumulative_returns, label='Cumulative Payoff', color='darkcyan')
    ax3.set_title(f'Cumulative Payoff: {stock1} & {stock2}', fontweight='bold', color='navy')
    ax3.legend()
    ax3.plot(cumulative_returns.index, cumulative_returns, label='Cumulative Payoff', color='darkcyan')
    ax3.set_title(f'Cumulative Payoff: {stock1} & {stock2}', fontsize=15, fontweight='bold', color='navy')
    ax3.grid(True, which='major', linestyle='--', linewidth=0.5, color='grey')
    ax3.minorticks_on()
    ax3.grid(True, which='minor', linestyle=':', linewidth=0.5, color='lightgrey')
    ax3.tick_params(axis='x', rotation=45)
    ax3.legend()

    plt.tight_layout()
    plt.show()

def plot_single_pair_analysis_coint(pair_index, pairs, signals, threshold_dev, test_data, test_signals):
    """
    Plots three aspects of a single stock pair side by side: z-scores, signals and prices, and cumulative payoff.

    Args:
        pair_index (int): Index of the pair to be plotted.
        pairs (list): List of tuples, each containing the names of two stocks.
        data (DataFrame): DataFrame containing historical data for each stock.
        signals (DataFrame): DataFrame containing z-scores and other metrics.
        threshold_dev (int): The number of standard deviations to set as the threshold for z-score plotting.
        test_data (DataFrame): Test dataset containing the adjusted close prices for each stock.
        test_signals (DataFrame): DataFrame containing trading signals and positions for each pair of stocks.
    """
    if pair_index >= len(pairs):
        raise ValueError("Pair index out of range.")

    stock1, stock2 = pairs[pair_index]
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))

    # Plot 1: Z-scores
    zscore_col = f'{stock1}_{stock2}_zscore'
    ax1.plot(signals.index, signals[zscore_col], label='Z-Score', color='blue')
    upper_threshold = threshold_dev
    lower_threshold = -threshold_dev
    ax1.axhline(y=upper_threshold, color='red', linestyle='--', label='Upper Threshold')
    ax1.axhline(y=lower_threshold, color='green', linestyle='--', label='Lower Threshold')
    ax1.set_title(f'Z-Scores: {stock1} vs {stock2}', fontsize=15, fontweight='bold', color='navy')
    ax1.grid(True, which='major', linestyle='--', linewidth=0.5, color='grey')
    ax1.minorticks_on()
    ax1.grid(True, which='minor', linestyle=':', linewidth=0.5, color='lightgrey')
    ax1.tick_params(axis='x', rotation=45)
    ax1.legend()

    # Plot 2: Signals and prices
    ax2.plot(test_data.index, test_data[stock1], label=stock1, color='orange')
    ax2.plot(test_data.index, test_data[stock2], label=stock2, color='blue')

    # Correct indexing for signals plotting
    long_signals_stock1 = test_signals[test_signals[f'{stock1}_{stock2}_positions1'] == 1].index
    short_signals_stock1 = test_signals[test_signals[f'{stock1}_{stock2}_positions1'] == -1].index
    long_signals_stock2 = test_signals[test_signals[f'{stock1}_{stock2}_positions2'] == 1].index
    short_signals_stock2 = test_signals[test_signals[f'{stock1}_{stock2}_positions2'] == -1].index

    ax2.plot(long_signals_stock1, 
         test_data.loc[long_signals_stock1, stock1], 
         '^', markersize=10, color='green', label=f'Long {stock1}', alpha=0.7)
    ax2.plot(short_signals_stock1,
         test_data.loc[short_signals_stock1, stock1], 
         'v', markersize=10, color='red', label=f'Short {stock1}', alpha=0.7)
    ax2.plot(long_signals_stock2,
            test_data.loc[long_signals_stock2, stock2], 
            '^', markersize=10, color='green', label=f'Long {stock2}', alpha=0.7)
    ax2.plot(short_signals_stock2,
            test_data.loc[short_signals_stock2, stock2], 
            'v', markersize=10, color='red', label=f'Short {stock2}', alpha=0.7)

    ax2.set_title(f'Signals & Prices: {stock1} & {stock2}', fontsize=15, fontweight='bold', color='navy')
    ax2.grid(True, which='major', linestyle='--', linewidth=0.5, color='grey')
    ax2.minorticks_on()
    ax2.grid(True, which='minor', linestyle=':', linewidth=0.5, color='lightgrey')
    ax2.tick_params(axis='x', rotation=45)
    ax2.legend()

    # Plot 3: Cumulative Payoff
    positions1 = test_signals[f'{stock1}_{stock2}_positions1']
    positions2 = test_signals[f'{stock1}_{stock2}_positions2']
    #positions1 = test_signals[f'{stock1}_{stock2}_signal1']
    #positions2 = test_signals[f'{stock1}_{stock2}_signal2']
    returns1 = positions1.shift(1) * test_data[stock1].pct_change()
    returns2 = positions2.shift(1) * test_data[stock2].pct_change()
    cumulative_returns = (returns1.fillna(0) + returns2.fillna(0)).cumsum()

    ax3.plot(cumulative_returns.index, cumulative_returns, label='Cumulative Payoff', color='darkcyan')
    ax3.set_title(f'Cumulative Payoff: {stock1} & {stock2}', fontweight='bold', color='navy')
    ax3.legend()
    ax3.plot(cumulative_returns.index, cumulative_returns, label='Cumulative Payoff', color='darkcyan')
    ax3.set_title(f'Cumulative Payoff: {stock1} & {stock2}', fontsize=15, fontweight='bold', color='navy')
    ax3.grid(True, which='major', linestyle='--', linewidth=0.5, color='grey')
    ax3.minorticks_on()
    ax3.grid(True, which='minor', linestyle=':', linewidth=0.5, color='lightgrey')
    ax3.tick_params(axis='x', rotation=45)
    ax3.legend()

    plt.tight_layout()
    plt.show()


def plot_single_pair_analysis_copulas(pair_index, pairs, signals_df, threshold, test_data, test_signals):
    """
    Plots three aspects of a single stock pair side by side: cumulative mispricing indices, signals and prices, and cumulative payoff.

    Args:
        pair_index (int): Index of the pair to be plotted.
        pairs (list): List of tuples, each containing the names of two stocks.
        signals_df (DataFrame): DataFrame containing the mispricing indices and other metrics.
        threshold (float): Threshold for the mispricing indices.
        test_data (DataFrame): Test dataset containing the adjusted close prices for each stock.
        test_signals (DataFrame): DataFrame containing trading signals and positions for each pair of stocks.
    """
    if pair_index >= len(pairs):
        raise ValueError("Pair index out of range.")

    stock1, stock2 = pairs[pair_index]
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))

    # Plot 1: Cumulative Mispricing Indices
    col_M1 = f'{stock1}_{stock2}_M1'
    col_M2 = f'{stock1}_{stock2}_M2'
    ax1.plot(signals_df.index, signals_df[col_M1], label=f'{stock1} M1', color='orange')
    ax1.plot(signals_df.index, signals_df[col_M2], label=f'{stock2} M2', color='blue')
    ax1.axhline(y=threshold, color='red', linestyle='--', label=f'+{threshold} Threshold')
    ax1.axhline(y=-threshold, color='green', linestyle='--', label=f'-{threshold} Threshold')
    ax1.set_title(f'Cumulative Mispricing Indices: {stock1} vs {stock2}', fontsize=15, fontweight='bold', color='navy')
    ax1.grid(True, which='major', linestyle='--', linewidth=0.5, color='grey')
    ax1.minorticks_on()
    ax1.grid(True, which='minor', linestyle=':', linewidth=0.5, color='lightgrey')
    ax1.tick_params(axis='x', rotation=45)
    ax1.legend()
    ax1.legend()

    # Plot 2: Signals and prices
    ax2.plot(test_data.index, test_data[stock1], label=stock1, color='orange')
    ax2.plot(test_data.index, test_data[stock2], label=stock2, color='blue')

    # Correct indexing for signals plotting
    long_signals_stock1 = test_signals[test_signals[f'{stock1}_{stock2}_positions1'] == 1].index
    short_signals_stock1 = test_signals[test_signals[f'{stock1}_{stock2}_positions1'] == -1].index
    long_signals_stock2 = test_signals[test_signals[f'{stock1}_{stock2}_positions2'] == 1].index
    short_signals_stock2 = test_signals[test_signals[f'{stock1}_{stock2}_positions2'] == -1].index

    ax2.plot(long_signals_stock1, 
         test_data.loc[long_signals_stock1, stock1], 
         '^', markersize=10, color='green', label=f'Long {stock1}', alpha=0.7)
    ax2.plot(short_signals_stock1,
         test_data.loc[short_signals_stock1, stock1], 
         'v', markersize=10, color='red', label=f'Short {stock1}', alpha=0.7)
    ax2.plot(long_signals_stock2,
            test_data.loc[long_signals_stock2, stock2], 
            '^', markersize=10, color='green', label=f'Long {stock2}', alpha=0.7)
    ax2.plot(short_signals_stock2,
            test_data.loc[short_signals_stock2, stock2], 
            'v', markersize=10, color='red', label=f'Short {stock2}', alpha=0.7)

    ax2.set_title(f'Signals & Prices: {stock1} & {stock2}', fontsize=15, fontweight='bold', color='navy')
    ax2.grid(True, which='major', linestyle='--', linewidth=0.5, color='grey')
    ax2.minorticks_on()
    ax2.grid(True, which='minor', linestyle=':', linewidth=0.5, color='lightgrey')
    ax2.tick_params(axis='x', rotation=45)
    ax2.legend()

    # Plot 3: Cumulative Payoff
    positions1 = test_signals[f'{stock1}_{stock2}_positions1']
    positions2 = test_signals[f'{stock1}_{stock2}_positions2']
    #positions1 = test_signals[f'{stock1}_{stock2}_signal1']
    #positions2 = test_signals[f'{stock1}_{stock2}_signal2']
    returns1 = positions1.shift(1) * test_data[stock1].pct_change()
    returns2 = positions2.shift(1) * test_data[stock2].pct_change()
    cumulative_returns = (returns1.fillna(0) + returns2.fillna(0)).cumsum()

    ax3.plot(cumulative_returns.index, cumulative_returns, label='Cumulative Payoff', color='darkcyan')
    ax3.set_title(f'Cumulative Payoff: {stock1} & {stock2}', fontweight='bold', color='navy')
    ax3.legend()
    ax3.plot(cumulative_returns.index, cumulative_returns, label='Cumulative Payoff', color='darkcyan')
    ax3.set_title(f'Cumulative Payoff: {stock1} & {stock2}', fontsize=15, fontweight='bold', color='navy')
    ax3.grid(True, which='major', linestyle='--', linewidth=0.5, color='grey')
    ax3.minorticks_on()
    ax3.grid(True, which='minor', linestyle=':', linewidth=0.5, color='lightgrey')
    ax3.tick_params(axis='x', rotation=45)
    ax3.legend()

    plt.tight_layout()
    plt.show()
