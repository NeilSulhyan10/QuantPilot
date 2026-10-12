from __future__ import annotations

import pandas as pd

from src.backtesting.config import BacktestConfig
from src.backtesting.costs import calculate_transaction_cost
from src.backtesting.returns import apply_rebalance_costs, calculate_net_portfolio_return
from src.backtesting.results import BacktestResult
from src.backtesting.slippage import calculate_slippage
from src.portfolio.drift import drift_weights
from src.portfolio.rebalance import calculate_rebalance_turnover
from src.portfolio.schedule import monthly_rebalance_dates


def backtest_goal_portfolio(
    returns: pd.DataFrame,
    portfolio,
    config: BacktestConfig | None = None,
    rebalance_monthly: bool = False,
) -> BacktestResult:
    """
    Evaluate a generated goal portfolio using fixed target weights.

    The target weights may either drift between the initial deployment and
    the end of the backtest, or be restored monthly. The portfolio is generated
    separately and evaluated historically; this is not walk-forward optimization.
    """
    if returns.empty:
        raise ValueError("Returns must not be empty.")

    if returns.isna().any().any():
        raise ValueError("Returns must not contain NaN values.")

    if portfolio.weights.empty:
        raise ValueError("Goal portfolio weights must not be empty.")

    # Handle cash weight if present
    cash_weight = getattr(portfolio, 'cash_weight', 0.0)
    if cash_weight < 0 or cash_weight > 1:
        raise ValueError("Cash weight must be between 0 and 1.")

    weights = portfolio.weights.copy()
    weights.index = weights.index.astype(str).str.upper()

    missing = weights.index.difference(returns.columns)
    if len(missing) > 0:
        raise ValueError(
            f"Portfolio contains assets missing from returns: {list(missing)}"
        )

    # Weights should sum to 1 - cash_weight (stocks only)
    expected_sum = 1.0 - cash_weight
    if abs(float(weights.sum()) - expected_sum) > 1e-5:
        raise ValueError(
            f"Portfolio stock weights must sum to {expected_sum:.6f} "
            f"(cash_weight={cash_weight:.6f}). Got {float(weights.sum()):.6f}."
        )

    if (weights < -1e-5).any():
        raise ValueError("Portfolio weights must be long-only.")

    # Add cash as an asset with zero returns
    cash_returns = pd.Series(0.0, index=returns.index, name="CASH")
    returns_with_cash = pd.concat([returns, cash_returns], axis=1)
    selected_returns = returns_with_cash.loc[:, list(weights.index) + ["CASH"]].copy()

    # Combine stock weights with cash weight
    all_weights = weights.copy()
    all_weights["CASH"] = cash_weight

    config = config or BacktestConfig()

    if rebalance_monthly:
        return _backtest_monthly_rebalanced(
            selected_returns=selected_returns,
            target_weights=all_weights,
            config=config,
        )

    current_weights = all_weights.copy()

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


def _backtest_monthly_rebalanced(
    selected_returns: pd.DataFrame,
    target_weights: pd.Series,
    config: BacktestConfig,
) -> BacktestResult:
    """Backtest target weights using the same monthly schedule as equal weight."""
    rebalance_dates = monthly_rebalance_dates(selected_returns.index)
    portfolio_returns = pd.Series(index=selected_returns.index, dtype=float)
    weight_history: dict[pd.Timestamp, pd.Series] = {}
    turnovers = pd.Series(0.0, index=rebalance_dates, dtype=float)
    transaction_costs = pd.Series(0.0, index=rebalance_dates, dtype=float)
    slippage = pd.Series(0.0, index=rebalance_dates, dtype=float)

    for i, rebalance_date in enumerate(rebalance_dates):
        current_weights = target_weights.copy()
        if i + 1 < len(rebalance_dates):
            next_rebalance_date = rebalance_dates[i + 1]
            period_dates = selected_returns.index[
                (selected_returns.index > rebalance_date)
                & (selected_returns.index < next_rebalance_date)
            ]
        else:
            period_dates = selected_returns.index[selected_returns.index > rebalance_date]

        for date in period_dates:
            asset_returns = selected_returns.loc[date]
            portfolio_returns.loc[date] = float((current_weights * asset_returns).sum())
            weight_history[date] = current_weights.copy()
            current_weights = drift_weights(current_weights, asset_returns)

        if i + 1 < len(rebalance_dates):
            next_rebalance_date = rebalance_dates[i + 1]
            turnover = calculate_rebalance_turnover(current_weights, target_weights)
            turnovers.loc[next_rebalance_date] = turnover
            transaction_costs.loc[next_rebalance_date] = calculate_transaction_cost(
                turnover, config.transaction_cost_rate
            )
            slippage.loc[next_rebalance_date] = calculate_slippage(
                turnover, config.slippage_rate
            )

    first_rebalance_date = rebalance_dates[0]
    turnovers.loc[first_rebalance_date] = 1.0
    transaction_costs.loc[first_rebalance_date] = calculate_transaction_cost(
        1.0, config.transaction_cost_rate
    )
    slippage.loc[first_rebalance_date] = calculate_slippage(
        1.0, config.slippage_rate
    )

    portfolio_returns = portfolio_returns.dropna()
    weights = pd.DataFrame.from_dict(weight_history, orient="index")
    weights.index.name = "date"
    net_returns = apply_rebalance_costs(
        gross_returns=portfolio_returns,
        rebalance_dates=rebalance_dates,
        transaction_costs=transaction_costs,
        slippage=slippage,
    )

    return BacktestResult(
        returns=net_returns,
        weights=weights,
        turnover=turnovers,
        transaction_costs=transaction_costs,
        slippage=slippage,
    )
