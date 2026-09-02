from dataclasses import dataclass

import pandas as pd

from src.backtesting.config import BacktestConfig
from src.backtesting.costs import calculate_transaction_cost
from src.backtesting.results import BacktestResult
from src.backtesting.returns import apply_rebalance_costs
from src.backtesting.slippage import calculate_slippage
from src.portfolio.drift import drift_weights
from src.portfolio.rebalance import calculate_rebalance_turnover
from src.portfolio.schedule import monthly_rebalance_dates

@dataclass
class EqualWeightBacktester:
    config: BacktestConfig

    def run(
        self,
        returns: pd.DataFrame,
        rebalance_dates: pd.DatetimeIndex | None = None,
    ) -> BacktestResult:

        if returns.empty:
            raise ValueError("Returns dataframe is empty.")

        tickers = list(returns.columns)

        if rebalance_dates is None:
            rebalance_dates = monthly_rebalance_dates(returns.index)

        if len(rebalance_dates) == 0:
            raise ValueError("No rebalance dates available.")

        rebalance_dates = pd.DatetimeIndex(
            rebalance_dates
        ).sort_values()

        target_weight = pd.Series(
            1.0 / len(tickers),
            index=tickers,
            dtype=float,
        )

        target_weights = {
            date: target_weight.copy()
            for date in rebalance_dates
        }

        portfolio_returns = pd.Series(
            index=returns.index,
            dtype=float,
        )

        turnovers = pd.Series(
            0.0,
            index=rebalance_dates,
            dtype=float,
        )

        transaction_costs = pd.Series(
            0.0,
            index=rebalance_dates,
            dtype=float,
        )

        slippage = pd.Series(
            0.0,
            index=rebalance_dates,
            dtype=float,
        )

        for i, rebalance_date in enumerate(rebalance_dates):

            current_weights = target_weights[
                rebalance_date
            ].copy()

            if i + 1 < len(rebalance_dates):

                next_rebalance_date = rebalance_dates[i + 1]

                period_dates = returns.index[
                    (returns.index > rebalance_date)
                    & (returns.index < next_rebalance_date)
                ]

            else:

                period_dates = returns.index[
                    returns.index > rebalance_date
                ]

            for date in period_dates:

                asset_returns = returns.loc[date]

                portfolio_returns.loc[date] = float(
                    (current_weights * asset_returns).sum()
                )

                current_weights = drift_weights(
                    current_weights,
                    asset_returns,
                )

            # Calculate turnover at the next rebalance.
            if i + 1 < len(rebalance_dates):

                next_rebalance_date = rebalance_dates[i + 1]

                target = target_weights[
                    next_rebalance_date
                ]

                turnover = calculate_rebalance_turnover(
                    current_weights,
                    target,
                )

                turnovers.loc[next_rebalance_date] = turnover

                transaction_costs.loc[next_rebalance_date] = (
                    calculate_transaction_cost(
                        turnover,
                        self.config.transaction_cost_rate,
                    )
                )

                slippage.loc[next_rebalance_date] = (
                    calculate_slippage(
                        turnover,
                        self.config.slippage_rate,
                    )
                )

        # Initial portfolio deployment.
        first_rebalance_date = rebalance_dates[0]

        turnovers.loc[first_rebalance_date] = 1.0

        transaction_costs.loc[first_rebalance_date] = (
            calculate_transaction_cost(
                1.0,
                self.config.transaction_cost_rate,
            )
        )

        slippage.loc[first_rebalance_date] = (
            calculate_slippage(
                1.0,
                self.config.slippage_rate,
            )
        )

        portfolio_returns = portfolio_returns.dropna()

        net_returns = apply_rebalance_costs(
            gross_returns=portfolio_returns,
            rebalance_dates=rebalance_dates,
            transaction_costs=transaction_costs,
            slippage=slippage,
        )

        weights = pd.DataFrame(target_weights).T
        weights.index.name = "date"

        return BacktestResult(
            returns=net_returns,
            weights=weights,
            turnover=turnovers,
            transaction_costs=transaction_costs,
            slippage=slippage,
        )