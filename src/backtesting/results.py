from dataclasses import dataclass

import pandas as pd


@dataclass
class BacktestResult:
    """
    Container for the outputs of a portfolio backtest.
    """

    returns: pd.Series
    weights: pd.DataFrame
    turnover: pd.Series
    transaction_costs: pd.Series
    slippage: pd.Series
