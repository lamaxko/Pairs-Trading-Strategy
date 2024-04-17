
# Pairs Trading Strategy

This repository implements and backtests Distance, Cointegration, and Copulas Pairs Trading Strategies. The strategies are separated into distinct methods, each located in its respective folder within the `Methods` directory.

## Installation

To install the required dependencies, run:

```bash
pip install -r requirements.txt
```

## Visualization Notebooks

Detailed visualizations of the trading strategies are provided in Jupyter notebooks located in the root directory:

- `00_EvalBacktest.ipynb`: Evaluation of backtest results for the strategies (In PROGRESS).
- `01_VisDistanceMethod.ipynb`: Visualization of the Distance Method.
- `02_VisCointegrationMethod.ipynb`: Visualization of the Cointegration Method.
- `03_VisCopulasMethod.ipynb`: Visualization of the Copulas Method.

## Methods

Implementation of trading strategies:

- `DistanceMethod.py`: Implements the Distance Method strategy.
- `CointegrationMethod.py`: Implements the Cointegration Method strategy.
- `CopulasMethod.py`: Implements the Copulas Method strategy.

## Helper Functions

The repository provides helper functions organized in the `Helpers` directory:

- `BacktestHelpers.py`: Functions specific to backtesting the strategies.
- `DataHelpers.py`: Functions to handle and manipulate data.
- `ModuleHelpers.py`: General utility functions that support various aspects of the strategies.
- `PlotHelpers.py`: Functions that assist in plotting the outputs within the visualization notebooks.

---

**Note**: The project is under development and not all components are final.
