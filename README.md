# Pairs Trading Strategy

This repository offers implementations and backtests for various Pairs Trading Strategies, such as the Distance, Cointegration, and Copulas Methods. It serves as a comprehensive toolkit for analyzing and visualizing pairs trading dynamics along with their historical performance.

## Installation

To install the required dependencies, run:

```bash
pip install -r requirements.txt
```

## Visualization Notebooks

Explore detailed visualizations of the trading strategies in Jupyter notebooks located in the `Code` folder:

- `VisCointegrationMethod.ipynb`: Visualization of the Cointegration Method.
- `VisCopulasMethod.ipynb`: Visualization of the Copulas Method.
- `VisDistanceMethod.ipynb`: Visualization of the Distance Method.

## Helper Functions

The repository provides helper functions organized in the `Helpers` directory:

- `FunctHelpers.py`: Utility functions for calculating values and metrics used in various methods.
- `PlotHelpers.py`: Assists in plotting the outputs within the visualization notebooks.
- `TstMethods.py`: Implementation of the three methods in a Python file without visualization, used in backtesting by `EvalBacktest.ipynb`.

## Backtesting (IN PROGRESS NOT FINISHED)

- `EvalBacktest.ipynb`: Utilizes the functions from `TstMethods.py`. The backtest assesses the performance of the Distance, Cointegration, and Copulas Methods across multiple overlapping periods. Using a 12-month calibration period and a 6-month trading period.

---

**Note**: Project is under development and not finished
