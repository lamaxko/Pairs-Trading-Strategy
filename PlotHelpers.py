import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance.shared as shared

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
            
            ax.plot(signals[f'{stock1}_{stock2}_signal1'], label=f'{stock1}_{stock2}_signal1', color='red')
            ax.plot(signals[f'{stock1}_{stock2}_signal2'], label=f'{stock1}_{stock2}_signal2', color='blue')
            
            ax.grid(True, which='major', linestyle='--', linewidth=0.5, color='grey')
            ax.minorticks_on()
            ax.grid(True, which='minor', linestyle=':', linewidth=0.5, color='lightgrey')
            ax.set_title(f'Pair {pair_index + 1}', fontsize=16, fontweight='bold', color='navy')
            ax.tick_params(axis='x', rotation=45)
            ax.legend()

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

        ax.plot(signals.index, signals[asset_one], label=asset_one, color='#1300CF', linewidth=1)
        ax2.plot(signals.index, signals[asset_two], label=asset_two, color='#B000CF', linewidth=1)
                
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
    Each subplot will show the spread for one pair with a red dotted line indicating the mean of the spread.
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
                ax.plot(spread, label='Spread', color='blue')
                ax.axhline(spread.mean(), color='red', linestyle='--', label='Mean')
                ax.set_title(f'Pair {spread_index + 1}', fontsize=16, fontweight='bold', color='navy')
                
                ax.grid(True, which='major', linestyle='--', linewidth=0.5, color='grey')
                ax.minorticks_on()
                ax.grid(True, which='minor', linestyle=':', linewidth=0.5, color='lightgrey')
                
                ax.tick_params(axis='x', rotation=45) 
                ax.legend()

    plt.tight_layout()
    plt.show()

def plot_zscores(signals, pairs):
    """
    Visualizes the z-scores, upper, and lower thresholds of selected stock pairs to illustrate their normalized spreads over time.
    The function generates a grid of line plots in a 5x4 format, each representing a different pair.

    Args:
        signals (DataFrame): A DataFrame containing the z-scores and thresholds for multiple stock pairs.
        pairs (list): A list of tuples, each representing a pair of stock names.
    """
    num_pairs = len(pairs)  # Determine the number of pairs
    rows, cols = 5, 4
    fig, axs = plt.subplots(rows, cols, figsize=(20, 20))  # Create a 5x4 grid of plots
    pair_idx = 0

    for i in range(rows):
        for j in range(cols):
            ax = axs[i, j]
            if pair_idx < num_pairs:
                stock1, stock2 = pairs[pair_idx]
                pair_key = f'{stock1}_{stock2}'

                zscore_col = f'{pair_key}_zscore'
                upper_limit_col = f'{pair_key}_z_upper_limit'
                lower_limit_col = f'{pair_key}_z_lower_limit'

                # Plot z-score
                if zscore_col in signals.columns:
                    ax.plot(signals.index, signals[zscore_col], label='Zscore', color='blue')
                    # Plot thresholds
                    ax.axhline(signals[upper_limit_col].mean(), color='red', linestyle='--', label='Upper Threshold')
                    ax.axhline(signals[lower_limit_col].mean(), color='green', linestyle='--', label='Lower Threshold')
                    ax.set_title(f'Pair {stock1} vs {stock2}', fontsize=16, fontweight='bold', color='navy')

                    ax.grid(True, which='major', linestyle='--', linewidth=0.5, color='grey')
                    ax.minorticks_on()
                    ax.grid(True, which='minor', linestyle=':', linewidth=0.5, color='lightgrey')
                    
                    ax.tick_params(axis='x', rotation=45)  # Rotate date labels if needed
                    ax.legend()
                pair_idx += 1

    plt.tight_layout()  # Adjust layout to make room for all subplots
    plt.show()