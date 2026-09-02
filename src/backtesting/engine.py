from dataclasses import dataclass

import pandas as pd

from src.backtesting.config import BacktestConfig
from src.backtesting.results import BacktestResult
from src.optimization.mean_variance import optimize_mean_variance
from src.portfolio.schedule import monthly_rebalance_dates
from src.backtesting.costs import calculate_transaction_cost
from src.backtesting.slippage import calculate_slippage
from src.backtesting.returns import apply_rebalance_costs
from src.portfolio.drift import drift_weights
from src.portfolio.rebalance import calculate_rebalance_turnover

@dataclass
class WalkForwardBacktester:
    config: BacktestConfig

    def _get_rebalance_dates(
        self,
        returns: pd.DataFrame,
        expected_returns: pd.DataFrame,
        covariance_matrices: dict[pd.Timestamp, pd.DataFrame],
    ) -> pd.DatetimeIndex:
        dates = monthly_rebalance_dates(returns.index)

        available_dates = (
            set(expected_returns.index)
            & set(covariance_matrices.keys())
        )

        return pd.DatetimeIndex(
            [date for date in dates if date in available_dates]
        )

    def run(
        self,
        returns: pd.DataFrame,
        expected_returns: pd.DataFrame,
        covariance_matrices: dict[pd.Timestamp, pd.DataFrame],
    ) -> BacktestResult:

        if returns.empty:
            raise ValueError("Returns dataframe is empty.")

        rebalance_dates = self._get_rebalance_dates(
            returns=returns,
            expected_returns=expected_returns,
            covariance_matrices=covariance_matrices,
        )

        if len(rebalance_dates) == 0:
            raise ValueError("No valid rebalance dates available.")

        target_weights = {}

        for date in rebalance_dates:
            target_weights[date] = self._calculate_target_weights(
                date=date,
                expected_returns=expected_returns,
                covariance_matrices=covariance_matrices,
            )

        daily_returns = self._calculate_daily_returns(
            returns=returns,
            rebalance_dates=rebalance_dates,
            target_weights=target_weights,
        )

        turnovers, transaction_costs, slippage = (
            self._calculate_rebalance_costs(
                returns=returns,
                rebalance_dates=rebalance_dates,
                target_weights=target_weights,
            )
        )

        net_returns = apply_rebalance_costs(
            gross_returns=daily_returns,
            rebalance_dates=rebalance_dates,
            transaction_costs=transaction_costs,
            slippage=slippage,
        )


        weights = pd.DataFrame(
            target_weights
        ).T

        weights.index.name = "date"

        return BacktestResult(
            returns=net_returns,
            weights=weights,
            turnover=turnovers,
            transaction_costs=transaction_costs,
            slippage=slippage,
        )
    
    def _calculate_target_weights(
            self,
            date: pd.Timestamp,
            expected_returns: pd.DataFrame,
            covariance_matrices: dict[pd.Timestamp, pd.DataFrame],
    ) -> pd.Series:
        if date not in expected_returns.index:
            raise ValueError(f"No expected returns available for {date}.")

        if date not in covariance_matrices:
            raise ValueError(f"No covariance matrix available for {date}.")

        expected = expected_returns.loc[date]

        covariance = covariance_matrices[date]

        tickers = list(expected.index)

        if list(covariance.index) != tickers:
            raise ValueError(
                "Expected returns and covariance matrix tickers do not match."
            )

        weights = optimize_mean_variance(
            expected_returns=expected.to_numpy(),
            covariance=covariance.to_numpy(),
            tickers=tickers,
            risk_aversion=self.config.risk_aversion,
            max_weight=self.config.max_weight,
        )

        return weights

    def _calculate_daily_returns(
        self,
        returns: pd.DataFrame,
        rebalance_dates: pd.DatetimeIndex,
        target_weights: dict[pd.Timestamp, pd.Series],
    ) -> pd.Series:
        portfolio_returns = pd.Series(
            index=returns.index,
            dtype=float,
        )

        current_weights = None

        for i, rebalance_date in enumerate(rebalance_dates):

            # Set the portfolio to the optimized target
            # at the rebalance date.
            current_weights = target_weights[rebalance_date].copy()

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
    
                # Calculate return using weights at the
                # beginning of the trading day.
                portfolio_returns.loc[date] = float(
                    (current_weights * asset_returns).sum()
                )

                # Allow weights to drift after the day's return.
                current_weights = drift_weights(
                    current_weights,
                    asset_returns,
                )

        return portfolio_returns.dropna()
    
    def _calculate_rebalance_costs(
        self,
        returns: pd.DataFrame,
        rebalance_dates: pd.DatetimeIndex,
        target_weights: dict[pd.Timestamp, pd.Series],
    ) -> tuple[pd.Series, pd.Series, pd.Series]:

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

        # Initial portfolio deployment.
        first_date = rebalance_dates[0]

        turnovers.loc[first_date] = 1.0

        transaction_costs.loc[first_date] = calculate_transaction_cost(
            1.0,
            self.config.transaction_cost_rate,
        )

        slippage.loc[first_date] = calculate_slippage(
            1.0,
            self.config.slippage_rate,
        )

        # The portfolio starts at the first optimized target.
        current_weights = target_weights[first_date].copy()

        for i in range(1, len(rebalance_dates)):

            previous_rebalance_date = rebalance_dates[i - 1]
            rebalance_date = rebalance_dates[i]

            period_dates = returns.index[
                (returns.index > previous_rebalance_date)
                & (returns.index < rebalance_date)
            ]

            # Let portfolio weights drift between rebalances.
            for date in period_dates:
                current_weights = drift_weights(
                    current_weights,
                    returns.loc[date],
                )

            target = target_weights[rebalance_date]

            turnover = calculate_rebalance_turnover(
                current_weights,
                target,
            )

            turnovers.loc[rebalance_date] = turnover

            transaction_costs.loc[rebalance_date] = (
                calculate_transaction_cost(
                    turnover,
                    self.config.transaction_cost_rate,
                )
            )

            slippage.loc[rebalance_date] = (
                calculate_slippage(
                    turnover,
                    self.config.slippage_rate,
                )
            )

            # Rebalance to the optimized target.
            current_weights = target.copy()

        return turnovers, transaction_costs, slippage