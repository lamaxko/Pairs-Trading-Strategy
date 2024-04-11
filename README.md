# Pairs Trading Strategy

This repository is dedicated to the implementation and backtesting of Pairs Trading Strategies. It includes the Distance Method, Cointegration Method, and the upcoming Copulas Method. The aim is to provide a comprehensive suite for analyzing and visualizing pairs trading opportunities and their performance over time.

## Strategies

- **Distance Method(Finished)**: A statistical arbitrage strategy that selects pairs with a high degree of historical price convergence.
- **Cointegration Method(Finished)**: Identifies pairs of securities whose prices are typically moving together, implying a long-run equilibrium.
- **Copulas Method (TBD)**: A method to be developed that will focus on capturing the dependencies between the movements of pairs.

## Visualization Notebooks

The repository includes Jupyter notebooks for a detailed visualization of the trading strategies:

- `VisCointegrationMethod.ipynb`: Visualization of the Cointegration Method.
- `VisCopulasMethod.ipynb` (In Progress): Visualization of the Copulas Method.
- `VisDistanceMethod.ipynb`: Visualization of the Distance Method.

## Helper Functions

Helper functions are organized in the `Helpers` directory:

- `FunctHelpers.py`: Contains utility functions for calculating values and metrics used in the various methods.
- `PlotHelpers.py`: Aids in plotting the outputs within the visualization notebooks.

## Backtesting

For evaluation purposes, `EvalBacktest.ipynb` is included which uses the functions from `TstMethods.py`. This is a set of backtests to assess the performance of the Distance, Cointegration, and Copulas Methods in a historical context. It tests for multiple overlapping periods with a 12 month callibration period and a 6 month traiding period.

The backtesting process is yet to be completed and will be documented accordingly once finalized.

---

**Note**: This project is under active development, and some components are yet to be finished. The repository will be updated regularly with improvements and new features.
