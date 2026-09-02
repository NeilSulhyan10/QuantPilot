import pandas as pd

from src.backtesting.engine import (
    BacktestConfig,
    BacktestResult,
    WalkForwardBacktester,
)


class MLWalkForwardBacktester(WalkForwardBacktester):
    """
    Walk-forward backtester using ML-generated expected returns.

    The inherited engine handles:
    - monthly rebalancing
    - portfolio drift
    - turnover
    - transaction costs
    - slippage
    - portfolio returns

    This class only provides the ML expected-return estimates.
    """

    def run_ml(
        self,
        returns: pd.DataFrame,
        ml_expected_returns: pd.DataFrame,
        covariance_matrices: dict,
    ) -> BacktestResult:
        """
        Run the portfolio backtest using ML expected returns.

        ML predictions are used only when they are available at the
        corresponding rebalance date.
        """

        if returns.empty:
            raise ValueError("Returns dataframe is empty.")

        if ml_expected_returns.empty:
            raise ValueError(
                "ML expected returns dataframe is empty."
            )

        if not returns.index.equals(
            ml_expected_returns.index
        ):
            raise ValueError(
                "Returns and ML expected returns "
                "must have identical indices."
            )

        if list(returns.columns) != list(
            ml_expected_returns.columns
        ):
            raise ValueError(
                "Returns and ML expected returns "
                "must have identical columns."
            )

        if not covariance_matrices:
            raise ValueError(
                "Covariance matrices are empty."
            )

        # Only dates where ML predictions actually exist
        # can be used for portfolio optimization.
        available_predictions = (
            ml_expected_returns
            .dropna(how="any")
        )

        if available_predictions.empty:
            raise ValueError(
                "No complete ML predictions are available."
            )

        return super().run(
            returns=returns,
            expected_returns=available_predictions,
            covariance_matrices=covariance_matrices,
        )