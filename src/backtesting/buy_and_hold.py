import pandas as pd

from src.backtesting.results import BacktestResult
from src.backtesting.costs import calculate_transaction_cost
from src.backtesting.slippage import calculate_slippage
from src.portfolio.drift import drift_weights


class BuyAndHoldBacktester:
    """True buy-and-hold benchmark."""

    def run(self, returns: pd.DataFrame) -> BacktestResult:
        if returns.empty:
            raise ValueError("Returns dataframe is empty.")

        tickers = list(returns.columns)
        initial_weights = pd.Series(
            1.0 / len(tickers),
            index=tickers,
            dtype=float,
        )

        portfolio_returns = pd.Series(
            index=returns.index,
            dtype=float,
        )

        current_weights = initial_weights.copy()

        for date in returns.index:
            asset_returns = returns.loc[date]

            portfolio_return = float(
                (current_weights * asset_returns).sum()
            )

            portfolio_returns.loc[date] = portfolio_return

            current_weights = drift_weights(
                current_weights,
                asset_returns,
            )

        initial_turnover = 1.0

        first_date = returns.index[0]

        turnover = pd.Series(
            {first_date: initial_turnover},
            dtype=float,
        )

        transaction_costs = pd.Series(
            {
                first_date: calculate_transaction_cost(
                   initial_turnover
               )
            },
            dtype=float,
        )

        slippage = pd.Series(
           {
               first_date: calculate_slippage(
                   initial_turnover
                )
           },
            dtype=float,
        )

        turnover.loc[first_date] = initial_turnover

        transaction_costs.loc[first_date] = calculate_transaction_cost(
            initial_turnover
        )

        slippage.loc[first_date] = calculate_slippage(
            initial_turnover
        )

        net_returns = portfolio_returns.copy()

        net_returns.loc[first_date] -= (
            transaction_costs.loc[first_date]
            + slippage.loc[first_date]
        )

        weights = pd.DataFrame(
            [initial_weights],
            index=[first_date],
        )

        weights.index.name = "date"

        return BacktestResult(
            returns=net_returns,
            weights=weights,
            turnover=turnover,
            transaction_costs=transaction_costs,
            slippage=slippage,
        )