# Pairs Trading Strategy

This repository implements and backtests various Pairs Trading Strategies, including the Distance, Cointegration, and Copulas Methods. It serves as a complete toolkit for analyzing and visualizing pairs trading dynamics and their historical performance.

## Installation

pip install -r requirements.txt

## Visualization Notebooks inside the Code folder

The repository includes Jupyter notebooks for a detailed visualization of the trading strategies:

- `VisCointegrationMethod.ipynb`: Visualization of the Cointegration Method.
- `VisCopulasMethod.ipynb`: Visualization of the Copulas Method.
- `VisDistanceMethod.ipynb`: Visualization of the Distance Method.

## Helper Functions

Helper functions are organized in the `Helpers` directory:

- `FunctHelpers.py`: Contains utility functions for calculating values and metrics used in the various methods.
- `PlotHelpers.py`: Aids in plotting the outputs within the visualization notebooks.
- `TstMethods.py`: Is a implementation of the three methods in a python file without the visualization so that it can be used in the backtest.

## Backtesting (IN PROGRESS NOT FINISHED)

- `EvalBacktest.ipynb` is the visualization of Backtest which uses the functions from `TstMethods.py`. This is a set of backtests to assess the performance of the Distance, Cointegration, and Copulas Methods in a historical context. It tests for multiple overlapping periods with a 12 month callibration period and a 6 month traiding period.


---

**Note**: This project is under active development, and some components are yet to be finished. The repository will be updated regularly with improvements and new features.
