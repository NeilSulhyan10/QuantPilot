from __future__ import annotations

import pandas as pd

from src.backtesting.config import BacktestConfig
from src.backtesting.returns import calculate_net_portfolio_return
from src.backtesting.results import BacktestResult
from src.portfolio.drift import drift_weights


def backtest_goal_portfolio(
    returns: pd.DataFrame,
    portfolio,
    config: BacktestConfig | None = None,
) -> BacktestResult:
    """
    Evaluate a generated goal portfolio using fixed initial weights.

    This is a static-weight historical analysis. The portfolio is
    generated separately and then evaluated historically. It is not
    a walk-forward goal optimization.
    """
    if returns.empty:
        raise ValueError("Returns must not be empty.")

    if returns.isna().any().any():
        raise ValueError("Returns must not contain NaN values.")

    if portfolio.weights.empty:
        raise ValueError("Goal portfolio weights must not be empty.")

    weights = portfolio.weights.copy()
    weights.index = weights.index.astype(str).str.upper()

    missing = weights.index.difference(returns.columns)
    if len(missing) > 0:
        raise ValueError(
            f"Portfolio contains assets missing from returns: {list(missing)}"
        )

    selected_returns = returns.loc[:, weights.index].copy()

    if abs(float(weights.sum()) - 1.0) > 1e-5:
        raise ValueError("Portfolio weights must sum to 1.")

    if (weights < -1e-5).any():
        raise ValueError("Portfolio weights must be long-only.")

    current_weights = weights.copy()

    portfolio_returns = []
    weight_history = []

    for date, asset_returns in selected_returns.iterrows():
        gross_return = float(
            (current_weights * asset_returns).sum()
        )

        portfolio_returns.append(gross_return)
        weight_history.append(current_weights.copy())

        current_weights = drift_weights(
            current_weights,
            asset_returns,
        )

    gross_returns = pd.Series(
        portfolio_returns,
        index=selected_returns.index,
        name="return",
    )

    weights_df = pd.DataFrame(
        weight_history,
        index=selected_returns.index,
    )

    # Static portfolio has only the initial deployment cost.
    turnover = pd.Series(
        1.0,
        index=[selected_returns.index[0]],
        name="turnover",
    )

    config = config or BacktestConfig()

    transaction_cost = (
        1.0 * config.transaction_cost_rate
    )

    slippage_cost = (
        1.0 * config.slippage_rate
    )

    transaction_costs = pd.Series(
        transaction_cost,
        index=[selected_returns.index[0]],
        name="transaction_cost",
    )

    slippage = pd.Series(
        slippage_cost,
        index=[selected_returns.index[0]],
        name="slippage",
    )

    net_returns = gross_returns.copy()

    first_date = selected_returns.index[0]

    net_returns.loc[first_date] = calculate_net_portfolio_return(
        gross_returns.loc[first_date],
        transaction_cost=transaction_cost,
        slippage=slippage_cost,
    )

    return BacktestResult(
        returns=net_returns,
        weights=weights_df,
        turnover=turnover,
        transaction_costs=transaction_costs,
        slippage=slippage,
    )